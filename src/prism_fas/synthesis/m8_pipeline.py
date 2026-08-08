from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import numpy as np
import torch
import yaml
from prism_fas.data.loader.config import load_loader_config
from prism_fas.data.loader.transforms import read_image
from prism_fas.data.package.priors import load_prior
from prism_fas.recipes.bank import load_bank
from prism_fas.recipes.compile import compile_recipe
from prism_fas.recipes.conditioning import conditioning_vector
from .gpat_contracts import GPATBatch
from .masks import RegionMaskBuilder
from .pair_plan import SOURCE_SPLIT, load_pair_manifest

M8_PIPELINE_SCHEMA_VERSION = "m8-pipeline-v1"
PROJECT_ROOT = Path(__file__).parents[3]
LOADER_CONFIG = PROJECT_ROOT / "configs" / "data" / "loader_m4.yaml"
FORBIDDEN_SPLITS = ("source_dev", "target_test")
ALLOWED_DATASETS = ("casia_fasd", "msu_mfsd")


class PipelineError(RuntimeError):
    """An M8 pipeline stage was asked for data it is not allowed to open."""


class SourceOnlyAudit:
    """Records every package path M8 opens so isolation can be proven, not claimed."""
    def __init__(self) -> None:
        self.opened: list[str] = []
    def record(self, relative_path: str) -> str:
        text = str(relative_path).replace("\\", "/")
        lowered = text.lower()
        for token in FORBIDDEN_SPLITS:
            if token in lowered: raise PipelineError(f"M8 refused to open a forbidden artifact: {text}")
        if "siw" in lowered or "target" in lowered: raise PipelineError(f"M8 refused to open target data: {text}")
        self.opened.append(text)
        return text
    def report(self) -> dict[str, Any]:
        manifests = sorted({path for path in self.opened if path.startswith("manifests/")})
        return {"source_train_opened": True, "source_dev_opened": False, "target_test_opened": False,
                "target_label_artifact_opened": False, "raw_dataset_path_opened": False,
                "manifests_opened": manifests, "distinct_paths": len(set(self.opened)),
                "total_opens": len(self.opened),
                "path_prefixes": sorted({path.split("/", 1)[0] for path in self.opened})}


def load_gpat_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict): raise PipelineError("gpat config must be a YAML mapping")
    data = payload.get("data", {})
    if data.get("package_split") != SOURCE_SPLIT:
        raise PipelineError(f"gpat config package_split must be {SOURCE_SPLIT!r}")
    for dataset in data.get("allowed_datasets", []):
        if dataset not in ALLOWED_DATASETS: raise PipelineError(f"gpat config allows dataset {dataset!r}")
    text = json.dumps(payload, sort_keys=True)
    for token in FORBIDDEN_SPLITS + ("siw", "target_test"):
        if token in text.lower() and token not in json.dumps(data.get("forbidden_splits", []), sort_keys=True).lower():
            raise PipelineError(f"gpat config references {token!r} outside its forbidden list")
    return payload


def config_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass
class SampleStore:
    """Reads source_train image/prior payloads only, with an open-audit trail."""
    package_root: Path
    audit: SourceOnlyAudit
    _rows: dict[str, dict[str, Any]]
    _cache: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]
    _mask_cache: dict[tuple[str, str, str], np.ndarray] | None = None

    @classmethod
    def open(cls, package_root: Path, audit: SourceOnlyAudit | None = None) -> "SampleStore":
        import pyarrow.parquet as pq
        audit = audit or SourceOnlyAudit()
        root = Path(package_root)
        relative = f"manifests/{SOURCE_SPLIT}.parquet"
        audit.record(relative)
        table = pq.read_table(root / relative).to_pydict()
        rows: dict[str, dict[str, Any]] = {}
        for index in range(len(table["sample_id"])):
            if table["project_split"][index] != SOURCE_SPLIT:
                raise PipelineError(f"row {index} is not {SOURCE_SPLIT}")
            rows[table["sample_id"][index]] = {key: table[key][index] for key in table}
        return cls(package_root=root, audit=audit, _rows=rows, _cache={}, _mask_cache={})

    def row(self, sample_id: str) -> dict[str, Any]:
        try: return self._rows[sample_id]
        except KeyError: raise PipelineError(f"sample {sample_id} is not in {SOURCE_SPLIT}") from None

    def load(self, sample_id: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if sample_id in self._cache: return self._cache[sample_id]
        row = self.row(sample_id)
        loader = load_loader_config(LOADER_CONFIG)
        image = read_image(self.package_root / self.audit.record(row["image_relative_path"]), loader.image)
        arrays = load_prior(self.package_root / self.audit.record(row["prior_relative_path"]))
        self._cache[sample_id] = (image, arrays)
        return image, arrays

    def mask_builder(self, sample_id: str) -> RegionMaskBuilder:
        _, arrays = self.load(sample_id)
        return RegionMaskBuilder(height=224, width=224, parsing=arrays["parsing_labels"],
                                 landmarks=arrays["landmarks"], bbox=arrays["bbox"], crop_box=arrays["crop_box"])

    def cached_mask(self, sample_id: str, role: str, graph: Any, *, coverage: float, seed_scope: str,
                    use_support: bool) -> np.ndarray:
        """Deterministic region mask with memoization.

        Mask building is the dominant CPU cost of a GPAT epoch and is a pure
        function of (sample, recipe, role), so the result is cached. The cache
        can only change timing, never values.
        """
        if self._mask_cache is None: self._mask_cache = {}
        key = (sample_id, graph.recipe_id, role)
        hit = self._mask_cache.get(key)
        if hit is not None: return hit
        policy = graph.region_mask_policy
        result = self.mask_builder(sample_id).build(
            list(graph.requested_regions), geometry_shape=str(policy["geometry_shape"]),
            coverage=float(coverage), seed=graph.node_seed(graph.nodes[0], f"{sample_id}|{seed_scope}"))
        mask = np.asarray(result.operator_support_mask if use_support else result.requested_region_mask,
                          dtype=np.float32)
        self._mask_cache[key] = mask
        return mask


def build_batch(store: SampleStore, pairs: list[dict[str, Any]], bank: dict[str, Any],
                identity_model: Any, *, device: str = "cpu") -> GPATBatch:
    """Materialize one GPAT batch from real source_train pairs.

    The live embedding is produced by the same differentiable wrapper used for
    the generated embedding, then detached and cached, so the identity cosine is
    self-consistent.
    """
    recipes = {recipe.recipe_id: recipe for recipe in bank["recipes"]}
    live_images, spoof_images, conditionings, supports, styles, strengths = [], [], [], [], [], []
    for pair in pairs:
        recipe = recipes[pair["recipe_id"]]
        graph = compile_recipe(recipe, bank["ontology"], bank_id=bank["bank_id"])
        live_image, _ = store.load(pair["live_sample_id"])
        spoof_image, _ = store.load(pair["spoof_sample_id"])
        policy = graph.region_mask_policy
        support = store.cached_mask(pair["live_sample_id"], "live", graph,
                                    coverage=float(policy["requested_coverage"]),
                                    seed_scope="region_mask", use_support=True)
        style = store.cached_mask(pair["spoof_sample_id"], "spoof", graph, coverage=1.0,
                                  seed_scope="style_mask", use_support=False)
        live_images.append(live_image)
        spoof_images.append(spoof_image)
        conditionings.append(conditioning_vector(recipe, bank["ontology"]))
        supports.append(support)
        styles.append(style)
        strengths.append(float(np.mean([spec.strength for spec in recipe.artifacts])))
    live = torch.from_numpy(np.stack(live_images)).to(device)
    spoof = torch.from_numpy(np.stack(spoof_images)).to(device)
    with torch.no_grad():
        live_embedding = identity_model(live).detach()
    return GPATBatch(live_image=live, source_spoof_image=spoof,
                     recipe_conditioning=torch.from_numpy(np.stack(conditionings)).to(device),
                     target_support_mask=torch.from_numpy(np.stack(supports)).to(device),
                     source_style_mask=torch.from_numpy(np.stack(styles)).to(device),
                     recipe_strength=torch.tensor(strengths, dtype=torch.float32, device=device),
                     live_identity_embedding=live_embedding,
                     pair_ids=tuple(pair["pair_id"] for pair in pairs),
                     live_sample_ids=tuple(pair["live_sample_id"] for pair in pairs),
                     spoof_sample_ids=tuple(pair["spoof_sample_id"] for pair in pairs),
                     recipe_ids=tuple(pair["recipe_id"] for pair in pairs)).validate()


def load_pairs(pairs_root: Path, partition: str) -> list[dict[str, Any]]:
    name = {"train": "pair_manifest_train.parquet", "validation": "pair_manifest_validation.parquet"}[partition]
    return load_pair_manifest(Path(pairs_root) / name)


def resolve_bank(bank_root: Path) -> dict[str, Any]:
    return load_bank(bank_root)


# --- synthetic bank drivers ------------------------------------------------
# The correctness pilot, the resume audit and the determinism audit all drive the
# same `SyntheticBankGenerator`; nothing here re-implements generation.

PILOT_COUNT = 32
PILOT_PHYSICS = 16
PILOT_GPAT = 16
DEFAULT_INTERRUPT_AFTER = 64


def build_generator(*, package_root: Path, bank_root: Path, work_root: Path, plan_root: Path,
                    calibration_path: Path, gpat_checkpoint_path: Path | None = None,
                    backends: Any = None, device: str = "cpu",
                    bank_config_path: Path | None = None, quality_config_path: Path | None = None,
                    gpat_config_path: Path | None = None,
                    expected_gpat_sha256: str | None = None,
                    expected_pair_plan_identity: str | None = None,
                    reuse_root: Path | None = None, bank_id_prefix: str | None = None,
                    calibration_files: dict[str, Path] | None = None,
                    conditioning_control: Any = None) -> Any:
    """Assemble a `SyntheticBankGenerator` from the frozen configs and artifacts.

    `conditioning_control` defaults to `None`, which is every M8/M9/B08 path: the
    GPAT identity guard stays complete and a recipe-bank mismatch fails closed. Only
    the declared A02 control passes a validated `ConditioningBankControl`.
    """
    from .candidate_plan import load_bank_config
    from .quality_calibration import load_quality_config
    from .synthetic_bank import SyntheticBankGenerator
    configs = PROJECT_ROOT / "configs" / "synthesis"
    bank_config = load_bank_config(Path(bank_config_path or configs / "synthetic_bank_m8.yaml"))
    quality_config = load_quality_config(Path(quality_config_path or configs / "quality_gate_m8.yaml"))
    gpat_config = load_gpat_config(Path(gpat_config_path or configs / "gpat_m8.yaml"))
    return SyntheticBankGenerator(
        package_root=Path(package_root), bank_root=Path(bank_root), work_root=Path(work_root),
        plan_path=Path(plan_root) / "candidate_plan.parquet", calibration_path=Path(calibration_path),
        bank_config=bank_config, quality_config=quality_config, backends=backends,
        gpat_checkpoint_path=None if gpat_checkpoint_path is None else Path(gpat_checkpoint_path),
        gpat_config=gpat_config, device=device, expected_gpat_sha256=expected_gpat_sha256,
        expected_pair_plan_identity=expected_pair_plan_identity,
        reuse_root=None if reuse_root is None else Path(reuse_root),
        bank_id_prefix=bank_id_prefix, calibration_files=dict(calibration_files or {}),
        conditioning_control=conditioning_control)


def _pilot_key(row: dict[str, Any]) -> int:
    return int.from_bytes(hashlib.sha256(f"pilot|{row['synthetic_id']}".encode("utf-8")).digest()[:8], "big")


def select_pilot_rows(plan_rows: list[dict[str, Any]], bank: dict[str, Any], *,
                      count: int = PILOT_COUNT) -> list[dict[str, Any]]:
    """A deterministic, coverage-greedy 32-candidate correctness subset.

    Both routes, both live-target domains and both GPAT domain relations are
    represented, and the greedy step maximizes new (artifact type, region)
    coverage so an implementation defect in any operator or region shows up.
    """
    recipes = {recipe.recipe_id: recipe for recipe in bank["recipes"]}
    quotas: list[tuple[str, str, str]] = []
    for dataset in ("casia_fasd", "msu_mfsd"):
        quotas += [("physics", dataset, "not_applicable")] * (PILOT_PHYSICS // 2)
        for relation in ("same_domain", "cross_domain"):
            quotas += [("gpat", dataset, relation)] * (PILOT_GPAT // 4)
    quotas = quotas[:count]
    pools: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in plan_rows:
        key = (row["route"], row["live_target_dataset"], row["domain_relation"])
        pools.setdefault(key, []).append(row)
    for key in pools: pools[key].sort(key=lambda row: (_pilot_key(row), row["synthetic_id"]))
    covered: set[tuple[str, str]] = set()
    chosen: list[dict[str, Any]] = []
    taken: set[str] = set()
    for key in quotas:
        pool = [row for row in pools.get(key, []) if row["synthetic_id"] not in taken]
        if not pool: raise PipelineError(f"the frozen plan has no candidate for pilot bucket {key}")
        def gain(row: dict[str, Any]) -> tuple[int, int, str]:
            recipe = recipes[row["recipe_id"]]
            pairs = {(spec.name, region) for spec in recipe.artifacts for region in recipe.regions}
            return (-len(pairs - covered), _pilot_key(row), row["synthetic_id"])
        pick = min(pool, key=gain)
        recipe = recipes[pick["recipe_id"]]
        covered |= {(spec.name, region) for spec in recipe.artifacts for region in recipe.regions}
        chosen.append(pick); taken.add(pick["synthetic_id"])
    chosen.sort(key=lambda row: row["synthetic_id"])
    return chosen


def _pilot_coverage(rows: list[dict[str, Any]], bank: dict[str, Any]) -> dict[str, Any]:
    recipes = {recipe.recipe_id: recipe for recipe in bank["recipes"]}
    artifacts = sorted({spec.name for row in rows for spec in recipes[row["recipe_id"]].artifacts})
    regions = sorted({region for row in rows for region in recipes[row["recipe_id"]].regions})
    counts: dict[str, int] = {}
    for row in rows:
        for key in (f"route:{row['route']}", f"live:{row['live_target_dataset']}",
                    f"relation:{row['domain_relation']}"):
            counts[key] = counts.get(key, 0) + 1
    return {"candidates": len(rows), "artifact_types": artifacts, "artifact_type_count": len(artifacts),
            "regions": regions, "region_count": len(regions), "distinct_recipes": len({row["recipe_id"] for row in rows}),
            "bucket_counts": dict(sorted(counts.items()))}


def _record_fingerprint(record: dict[str, Any], root: Path) -> dict[str, Any]:
    """Everything a determinism comparison must agree on for one candidate."""
    from .synthetic_bank import decode_npz
    payload: dict[str, Any] = {
        "terminal_state": record["terminal_state"], "route": record["route"],
        "plan": record["plan"], "identity": record["identity"],
        "recipe": record.get("recipe"), "geometry": record.get("geometry"),
        "quality": None if "quality" not in record else {
            "accepted": record["quality"]["accepted"], "failed_gates": record["quality"]["failed_gates"],
            "q": record["quality"]["q"], "quality_components": record["quality"]["quality_components"],
            "metrics": record["quality"]["metrics"], "threshold_hash": record["quality"]["threshold_hash"]},
        "failure": record.get("failure"), "outputs": record.get("outputs", {})}
    if record["terminal_state"] == "accepted":
        outputs = record["outputs"]
        payload["bytes"] = {}
        for name in ("image", "mask", "artifact_map"):
            path = Path(root) / outputs[f"{name}_relative_path"]
            payload["bytes"][name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        map_path = Path(root) / outputs["artifact_map_relative_path"]
        if map_path.is_file():
            array = decode_npz(map_path.read_bytes())
            payload["artifact_map_array_sha256"] = hashlib.sha256(
                np.ascontiguousarray(array).tobytes()).hexdigest()
        metadata = Path(root) / "metadata" / f"{record['synthetic_id']}.json"
        payload["metadata_sha256"] = hashlib.sha256(metadata.read_bytes()).hexdigest() if metadata.is_file() else None
    return payload


def compare_records(left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Field-level comparison of two generations of the same candidate ids."""
    mismatches: list[dict[str, Any]] = []
    if sorted(left) != sorted(right):
        mismatches.append({"field": "candidate_ids", "left_only": sorted(set(left) - set(right))[:10],
                           "right_only": sorted(set(right) - set(left))[:10]})
    for synthetic_id in sorted(set(left) & set(right)):
        a, b = left[synthetic_id], right[synthetic_id]
        for field in sorted(set(a) | set(b)):
            if a.get(field) != b.get(field):
                mismatches.append({"synthetic_id": synthetic_id, "field": field,
                                   "left": str(a.get(field))[:200], "right": str(b.get(field))[:200]})
    return {"compared": len(set(left) & set(right)), "mismatches": mismatches[:40],
            "mismatch_count": len(mismatches), "identical": not mismatches}


def run_pilot(generator: Any, *, pilot_root: Path, count: int = PILOT_COUNT,
              progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """The correctness pilot. Its acceptance rate never changes a threshold."""
    from .synthetic_bank import IMAGE_SIZE, decode_npz, decode_png
    rows = select_pilot_rows(generator.plan_rows, generator.bank, count=count)
    restore = Path(generator.work_root)
    generator.work_root = Path(pilot_root)
    try:
        result = generator.run(rows=rows, resume=True, progress=progress)
    finally:
        # The pilot writes into its own ignored namespace; the generator goes back
        # to the real work root so a later stage cannot inherit the pilot's.
        generator.work_root = restore
    records = {record["synthetic_id"]: record for record in result["records"]}
    planned = {row["synthetic_id"] for row in rows}
    checks: dict[str, bool] = {
        "candidate_ids_match_the_frozen_plan": set(records) == planned,
        "terminal_accounting": len(records) == len(rows),
        "every_candidate_has_a_terminal_state": all(
            record["terminal_state"] in ("accepted", "rejected", "failed_generation") for record in records.values()),
        "gpat_rows_bind_the_frozen_checkpoint": all(
            record["plan"]["gpat_checkpoint_sha256"] == generator.plan_lock["gpat_checkpoint_sha256"]
            for record in records.values() if record["route"] == "gpat"),
        "physics_rows_bind_the_physics_engine": all(
            record["plan"]["physics_engine_version"] == "m7-physics-v1"
            for record in records.values() if record["route"] == "physics"),
        "rejected_rows_name_failed_gates": all(
            record["quality"]["failed_gates"] for record in records.values()
            if record["terminal_state"] == "rejected"),
        "accepted_rows_pass_every_gate": all(
            all(record["quality"]["gates"].values()) for record in records.values()
            if record["terminal_state"] == "accepted"),
        "quality_metrics_finite": all(
            all(np.isfinite(float(value)) for value in record["quality"]["metrics"].values()
                if isinstance(value, (int, float)))
            for record in records.values() if "quality" in record),
        "q_finite_in_unit_interval": all(
            np.isfinite(record["quality"]["q"]) and 0.0 <= record["quality"]["q"] <= 1.0
            for record in records.values() if "quality" in record),
        "recipe_match_not_applicable": all(
            record["quality"]["recipe_match"] == "not_applicable"
            for record in records.values() if "quality" in record)}
    payload_errors: list[str] = []
    accepted = [record for record in records.values() if record["terminal_state"] == "accepted"]
    for record in accepted:
        outputs = record["outputs"]
        image = decode_png((Path(pilot_root) / outputs["image_relative_path"]).read_bytes())
        mask = decode_png((Path(pilot_root) / outputs["mask_relative_path"]).read_bytes())
        array = decode_npz((Path(pilot_root) / outputs["artifact_map_relative_path"]).read_bytes())
        if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3): payload_errors.append(f"{record['synthetic_id']}: image shape")
        if set(np.unique(mask).tolist()) - {0, 255}: payload_errors.append(f"{record['synthetic_id']}: mask values")
        if array.dtype != np.float16 or array.shape != (1, IMAGE_SIZE, IMAGE_SIZE):
            payload_errors.append(f"{record['synthetic_id']}: artifact map dtype/shape")
        outside = mask != 255
        if outside.any() and float(np.abs(np.asarray(array[0], dtype=np.float32)[outside]).max()) != 0.0:
            payload_errors.append(f"{record['synthetic_id']}: artifact map outside the exact mask")
        original, _ = generator.store.load(record["plan"]["live_target_sample_id"])
        from .synthetic_bank import to_uint8
        if outside.any():
            error = int(np.abs(image.astype(np.int32) - to_uint8(original).astype(np.int32))[outside].max())
            if error != 0: payload_errors.append(f"{record['synthetic_id']}: outside-mask error {error}")
    checks["accepted_payloads_valid"] = not payload_errors
    states: dict[str, int] = {}
    for record in records.values(): states[record["terminal_state"]] = states.get(record["terminal_state"], 0) + 1
    return {"pilot_schema_version": "m8-generation-pilot-v1", "requested": count,
            "planned": len(rows), "terminal_counts": dict(sorted(states.items())),
            "coverage": _pilot_coverage(rows, generator.bank),
            "checks": checks, "payload_errors": payload_errors[:20],
            "passed": all(checks.values()),
            "acceptance_rate_is_not_an_implementation_verdict": True,
            "thresholds_unchanged": True,
            "identity": generator.identity(),
            "source_isolation": result.get("source_isolation", generator.audit.report()),
            "fingerprints": {synthetic_id: _record_fingerprint(record, Path(pilot_root))
                             for synthetic_id, record in sorted(records.items())}}


def run_pilot_determinism(generator: Any, *, first: dict[str, Any], rerun_root: Path,
                          count: int = PILOT_COUNT,
                          progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Regenerate the same 32 candidates in a separate ignored namespace."""
    rows = select_pilot_rows(generator.plan_rows, generator.bank, count=count)
    restore = Path(generator.work_root)
    generator.work_root = Path(rerun_root)
    try:
        result = generator.run(rows=rows, resume=False, progress=progress)
    finally:
        generator.work_root = restore
    second = {record["synthetic_id"]: _record_fingerprint(record, Path(rerun_root))
              for record in result["records"]}
    comparison = compare_records(first["fingerprints"], second)
    return {"determinism_schema_version": "m8-generation-pilot-determinism-v1",
            "candidates": len(rows), **comparison,
            "passed": comparison["identical"], "identity": generator.identity()}


def run_full_generation(generator: Any, *, interrupt_after: int | None = DEFAULT_INTERRUPT_AFTER,
                        progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Generate the whole frozen plan through a real interruption and a resume.

    The interruption is genuine — the first pass stops after a deterministic
    number of rebuilds and the second pass has to reuse what survived, rebuild
    the one deliberately corrupted candidate and finish the rest.
    """
    from .synthetic_bank import record_is_reusable
    work = Path(generator.work_root)
    identity = generator.identity()
    phases: list[dict[str, Any]] = []
    corruption: dict[str, Any] = {"performed": False}
    if interrupt_after:
        first = generator.run(resume=True, progress=progress, stop_after=interrupt_after)
        phases.append({"phase": "interrupted", **{key: first[key] for key in
                                                  ("status", "planned", "examined", "reused", "rebuilt")}})
        accepted = [record for record in first["records"] if record["terminal_state"] == "accepted"]
        if accepted:
            victim = sorted(accepted, key=lambda record: record["synthetic_id"])[0]
            target = work / victim["outputs"]["image_relative_path"]
            original_bytes = target.read_bytes()
            target.write_bytes(original_bytes[: max(1, len(original_bytes) // 2)])
            usable, reason = record_is_reusable(victim, identity, work)
            corruption = {"performed": True, "synthetic_id": victim["synthetic_id"],
                          "detected_as_unusable": not usable, "reason": reason,
                          "rebuilt_bytes_identical": None}
    second = generator.run(resume=True, progress=progress)
    phases.append({"phase": "resumed", **{key: second[key] for key in
                                          ("status", "planned", "examined", "reused", "rebuilt")},
                   "reuse_reasons": second["reuse_reasons"]})
    if corruption["performed"]:
        record = generator.load_record(corruption["synthetic_id"])
        path = work / record["outputs"]["image_relative_path"] if record.get("outputs") else None
        corruption["rebuilt_bytes_identical"] = bool(
            path is not None and path.is_file()
            and hashlib.sha256(path.read_bytes()).hexdigest() == record["outputs"]["image_sha256"])
        corruption["terminal_state_after_rebuild"] = record["terminal_state"]
    rerun = generator.run(resume=True, progress=progress)
    phases.append({"phase": "completed_rerun", **{key: rerun[key] for key in
                                                  ("status", "planned", "examined", "reused", "rebuilt")}})
    records = rerun["records"]
    ids = [record["synthetic_id"] for record in records]
    states: dict[str, int] = {}
    for record in records: states[record["terminal_state"]] = states.get(record["terminal_state"], 0) + 1
    checks = {
        "all_candidates_examined": rerun["examined"] == len(generator.plan_rows),
        "completed_rerun_rebuilt_nothing": rerun["rebuilt"] == 0,
        "completed_rerun_reused_everything": rerun["reused"] == len(generator.plan_rows),
        "no_duplicate_candidate_ids": len(set(ids)) == len(ids),
        "terminal_accounting": sum(states.values()) == len(generator.plan_rows),
        "resume_reused_valid_candidates": bool(interrupt_after) is False or phases[1]["reused"] > 0,
        "corrupted_candidate_detected": corruption.get("detected_as_unusable", True),
        "corrupted_candidate_rebuilt": corruption.get("rebuilt_bytes_identical", True) is not False}
    return {"resume_schema_version": "m8-resume-audit-v1", "phases": phases, "corruption_probe": corruption,
            "terminal_counts": dict(sorted(states.items())), "planned": len(generator.plan_rows),
            "checks": checks, "passed": all(checks.values()),
            "identity": identity, "records": records,
            "source_isolation": rerun.get("source_isolation", generator.audit.report())}


def run_determinism_audit(generator: Any, *, work_root: Path, rerun_root: Path, count: int = PILOT_COUNT,
                          progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Regenerate a deterministic 32-candidate subset of the completed bank under
    a separate ignored namespace and compare everything."""
    rows = select_pilot_rows(generator.plan_rows, generator.bank, count=count)
    baseline: dict[str, dict[str, Any]] = {}
    generator.work_root = Path(work_root)
    for row in rows:
        record = generator.load_record(row["synthetic_id"])
        if record is None: raise PipelineError(f"{row['synthetic_id']} has no record in the completed bank")
        baseline[row["synthetic_id"]] = _record_fingerprint(record, Path(work_root))
    generator.work_root = Path(rerun_root)
    result = generator.run(rows=rows, resume=False, progress=progress)
    rerun = {record["synthetic_id"]: _record_fingerprint(record, Path(rerun_root))
             for record in result["records"]}
    comparison = compare_records(baseline, rerun)
    generator.work_root = Path(work_root)
    return {"determinism_schema_version": "m8-determinism-audit-v1", "candidates": len(rows),
            "compared_fields": ["candidate_ids", "image bytes/hash", "mask bytes/hash",
                                "artifact map array/hash", "quality metrics", "q", "decisions",
                                "metadata", "recipe/checkpoint bindings"],
            **comparison, "passed": comparison["identical"], "identity": generator.identity()}
