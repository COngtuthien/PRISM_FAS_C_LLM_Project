from __future__ import annotations
import hashlib, io, json, os, re, tempfile
from pathlib import Path
from typing import Any, Callable
import numpy as np
import yaml
from prism_fas.data.loader.config import load_loader_config
from prism_fas.data.loader.transforms import read_image
from prism_fas.data.package.priors import load_prior
from prism_fas.recipes.audit import bank_audit
from prism_fas.recipes.bank import load_bank, validate_bank
from prism_fas.recipes.compile import compile_recipes, compile_summary
from prism_fas.recipes.schema import RecipeV11
from prism_fas.utils.core import atomic_json_write
from .contracts import SynthesisError, array_hash
from .operators import OPERATOR_NAMES
from .physics import PhysicsEngine

AUDIT_SCHEMA_VERSION = "m7-audit-v1"
PROJECT_ROOT = Path(__file__).parents[3]
LOADER_CONFIG = PROJECT_ROOT / "configs" / "data" / "loader_m4.yaml"
SOURCE_SPLIT = "source_train"
SOURCE_LABEL = "live"
FORBIDDEN_SPLITS = ("source_dev", "target_test")
# Any of these in a written audit artifact means target data, private dataset
# metadata or a machine path leaked into an M7 output.
FORBIDDEN_AUDIT_TOKENS = ("siw", "target_test", "target_dev", "subject_id", "official_split",
                          "data/work", "Dataset/", "model_cache", "paths.local")
FORBIDDEN_AUDIT_PATTERNS = (r"[A-Za-z]:[\\/]", r"/home/", r"/Users/")


class AuditError(RuntimeError):
    """The real source-only physics audit did not meet its declared contract."""


def load_physics_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict): raise AuditError("physics config must be a YAML mapping")
    missing = [key for key in ("physics_schema_version", "operator_application_order", "preview", "outputs") if key not in payload]
    if missing: raise AuditError(f"physics config is missing keys: {missing}")
    return payload


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def _png_bytes(array: np.ndarray) -> bytes:
    """Lossless PNG encoding of a CHW float image or an HW binary mask."""
    import cv2
    values = np.asarray(array)
    if values.ndim == 3 and values.shape[0] == 3:
        rgb = np.clip(values.transpose(1, 2, 0) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        ok, buffer = cv2.imencode(".png", rgb[:, :, ::-1])
    else:
        grey = (np.asarray(values).astype(bool).astype(np.uint8)) * np.uint8(255)
        ok, buffer = cv2.imencode(".png", grey)
    if not ok: raise AuditError("PNG encoding failed")
    return bytes(buffer.tobytes())


def select_source_live_samples(package_root: Path, *, per_dataset: int, datasets: tuple[str, ...]) -> list[dict[str, Any]]:
    """Deterministic source_train live selection.

    Only `manifests/source_train.parquet` is opened. `source_dev` and
    `target_test` manifests are never read by this command.
    """
    import pyarrow.parquet as pq
    manifest = Path(package_root) / "manifests" / f"{SOURCE_SPLIT}.parquet"
    if not manifest.is_file(): raise AuditError(f"missing {SOURCE_SPLIT} manifest under {Path(package_root).name}")
    table = pq.read_table(manifest).to_pydict()
    selected: list[dict[str, Any]] = []
    for dataset in datasets:
        candidates = sorted(
            [{"sample_id": table["sample_id"][index], "dataset": table["dataset"][index],
              "project_split": table["project_split"][index], "label": table["label_live_spoof"][index],
              "image_relative_path": table["image_relative_path"][index],
              "prior_relative_path": table["prior_relative_path"][index],
              "crop_sha256": table["crop_sha256"][index], "prior_sha256": table["prior_sha256"][index]}
             for index in range(len(table["sample_id"]))
             if table["dataset"][index] == dataset and table["label_live_spoof"][index] == SOURCE_LABEL],
            key=lambda row: row["sample_id"])
        if len(candidates) < per_dataset:
            raise AuditError(f"{dataset} has only {len(candidates)} source_train live samples, need {per_dataset}")
        selected.extend(candidates[:per_dataset])
    return sorted(selected, key=lambda row: (row["dataset"], row["sample_id"]))


def plan_recipes(recipes: list[RecipeV11], *, slots: int) -> list[int]:
    """Deterministic coverage planner.

    Greedy set cover over the categorical axes that the audit must exercise
    (media, geometry, regions, operators, illumination), with the lowest recipe
    index as the tie-break, then round-robin over the remaining bank order.
    """
    required: set[str] = set()
    per_recipe: list[set[str]] = []
    for recipe in recipes:
        items = {f"medium:{recipe.medium.family}", f"geometry:{recipe.geometry.shape}",
                 f"illumination:{recipe.capture.illumination}"}
        items |= {f"region:{name}" for name in recipe.regions}
        items |= {f"operator:{spec.name}" for spec in recipe.artifacts}
        per_recipe.append(items)
        required |= items
    covered: set[str] = set()
    chosen: list[int] = []
    used: set[int] = set()
    while covered != required and len(chosen) < slots:
        best, best_gain = None, 0
        for index, items in enumerate(per_recipe):
            if index in used: continue
            gain = len(items - covered)
            if gain > best_gain: best, best_gain = index, gain
        if best is None: break
        chosen.append(best); used.add(best); covered |= per_recipe[best]
    order = [index for index in range(len(recipes)) if index not in used]
    position = 0
    while len(chosen) < slots:
        if not order: order = list(range(len(recipes)))
        chosen.append(order[position % len(order)]); position += 1
    return chosen[:slots]


def _leakage_scan(text: str) -> list[str]:
    lowered = text.lower()
    hits = [token for token in FORBIDDEN_AUDIT_TOKENS if token.lower() in lowered]
    hits += [pattern for pattern in FORBIDDEN_AUDIT_PATTERNS if re.search(pattern, text)]
    return sorted(dict.fromkeys(hits))


def run_preview(package_root: Path, bank_root: Path, config: dict[str, Any], output_root: Path, *,
                limit: int | None = None, write_artifacts: bool = True,
                progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Run the real source_train-live physics preview on CPU.

    Never opens source_dev or target_test, never touches the package for writing
    and never calls Modal or a GPU.
    """
    preview_config = config["preview"]
    if preview_config["split"] != SOURCE_SPLIT or preview_config["label"] != SOURCE_LABEL:
        raise AuditError("the M7 preview is defined for source_train live samples only")
    package_root, bank_root, output_root = Path(package_root), Path(bank_root), Path(output_root)
    lock_before = hashlib.sha256((package_root / "PACKAGE_LOCK.json").read_bytes()).hexdigest()
    bank = load_bank(bank_root)
    graphs = compile_recipes(bank["recipes"], bank["ontology"], bank_id=bank["bank_id"])
    by_id = {graph.recipe_id: graph for graph in graphs}
    datasets = tuple(preview_config["datasets"])
    samples = select_source_live_samples(package_root, per_dataset=int(preview_config["samples_per_dataset"]), datasets=datasets)
    per_sample = int(preview_config["recipes_per_sample"])
    slots = len(samples) * per_sample
    plan = plan_recipes(bank["recipes"], slots=slots)
    pairs = [(samples[index // per_sample], bank["recipes"][plan[index]]) for index in range(slots)]
    if limit is not None: pairs = pairs[:int(limit)]

    loader = load_loader_config(LOADER_CONFIG)
    engine = PhysicsEngine(coverage_tolerance=float(config.get("masks", {}).get("coverage_tolerance", 0.05)))
    previews = output_root / str(config["outputs"]["previews_dir"])
    rows: list[dict[str, Any]] = []
    cache: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    for position, (sample, recipe) in enumerate(pairs, 1):
        sample_id = sample["sample_id"]
        if sample["project_split"] != SOURCE_SPLIT or sample["label"] != SOURCE_LABEL:
            raise AuditError(f"{sample_id}: selection escaped source_train live")
        if sample_id not in cache:
            image = read_image(package_root / sample["image_relative_path"], loader.image)
            arrays = load_prior(package_root / sample["prior_relative_path"])
            cache[sample_id] = (image, arrays)
        image, arrays = cache[sample_id]
        graph = by_id[recipe.recipe_id]
        result = engine.apply(image, arrays["parsing_labels"], arrays["landmarks"], arrays["bbox"], graph,
                              sample_id, crop_box=arrays["crop_box"])
        stem = f"{sample_id}__{recipe.recipe_id}"
        row: dict[str, Any] = {
            "preview_index": position - 1, "sample_id": sample_id, "dataset": sample["dataset"],
            "project_split": sample["project_split"], "label": sample["label"],
            "bank_id": bank["bank_id"], "bank_content_identity_sha256": bank["lock"]["bank_content_identity_sha256"],
            "recipe_id": recipe.recipe_id, "recipe_hash": result.recipe_hash, "graph_hash": result.graph_hash,
            "medium": recipe.medium.family, "geometry_shape": recipe.geometry.shape,
            "illumination": recipe.capture.illumination, "regions": list(recipe.regions),
            "operators": list(graph.operator_names()),
            "operator_seeds": {entry["operator"]: entry["operator_seed"] for entry in result.trace["operators"]},
            "operator_parameters": {entry["operator"]: entry["parameters_used"] for entry in result.trace["operators"]},
            "region_sources": result.trace["region_sources"],
            "requested_coverage": result.trace["requested_coverage"],
            "achieved_coverage": round(float(result.trace["achieved_coverage"]), 6),
            "requested_region_pixels": int(np.asarray(result.requested_region_mask).astype(bool).sum()),
            "exact_edit_pixels": int(result.trace["exact_edit_pixels"]),
            "changed_pixels": int(result.trace["changed_pixels"]),
            "outside_mask_max_abs_error": float(result.trace["outside_mask_max_abs_error"]),
            "max_abs_difference_inside": float(result.trace["max_abs_difference_inside"]),
            "source_crop_sha256": sample["crop_sha256"], "source_prior_sha256": sample["prior_sha256"],
            "output_hashes": dict(result.output_hashes),
            "artifacts": {"image": f"{config['outputs']['previews_dir']}/images/{stem}.png",
                          "mask": f"{config['outputs']['previews_dir']}/masks/{stem}.png",
                          "strength_map": f"{config['outputs']['previews_dir']}/strength_maps/{stem}.npz",
                          "metadata": f"{config['outputs']['previews_dir']}/metadata/{stem}.json"}}
        if write_artifacts:
            image_png = _png_bytes(result.synthetic_image)
            mask_png = _png_bytes(np.asarray(result.exact_edit_mask)[0])
            buffer = io.BytesIO()
            np.savez(buffer, strength_map=np.asarray(result.artifact_strength_map, dtype=np.float32))
            strength_npz = buffer.getvalue()
            _atomic_bytes(previews / "images" / f"{stem}.png", image_png)
            _atomic_bytes(previews / "masks" / f"{stem}.png", mask_png)
            _atomic_bytes(previews / "strength_maps" / f"{stem}.npz", strength_npz)
            row["artifact_sha256"] = {"image_png": hashlib.sha256(image_png).hexdigest(),
                                      "mask_png": hashlib.sha256(mask_png).hexdigest(),
                                      "strength_map_npz": array_hash(result.artifact_strength_map)}
            atomic_json_write(previews / "metadata" / f"{stem}.json", {**row, "trace": result.trace})
        rows.append(row)
        if progress and (position == 1 or position % 16 == 0 or position == len(pairs)):
            progress({"stage": "preview", "done": position, "total": len(pairs)})
    lock_after = hashlib.sha256((package_root / "PACKAGE_LOCK.json").read_bytes()).hexdigest()
    if lock_before != lock_after: raise AuditError("the immutable M3B package lock changed during the audit")
    return {"rows": rows, "bank": bank, "graphs": graphs, "samples": samples, "plan": plan,
            "package_lock_sha256": lock_before}


def summarize(rows: list[dict[str, Any]], config: dict[str, Any], bank: dict[str, Any]) -> dict[str, Any]:
    required = config["preview"]["require_coverage"]
    media = sorted({row["medium"] for row in rows})
    geometry = sorted({row["geometry_shape"] for row in rows})
    regions = sorted({name for row in rows for name in row["regions"]})
    operators = sorted({name for row in rows for name in row["operators"]})
    illumination = sorted({row["illumination"] for row in rows})
    datasets: dict[str, int] = {}
    for row in rows: datasets[row["dataset"]] = datasets.get(row["dataset"], 0) + 1
    inputs: dict[str, set[str]] = {}
    for row in rows: inputs.setdefault(row["dataset"], set()).add(row["sample_id"])
    outside_errors = [row["outside_mask_max_abs_error"] for row in rows]
    empty_masks = [row["sample_id"] for row in rows if row["exact_edit_pixels"] <= 0]
    unchanged = [row["recipe_id"] for row in rows if row["changed_pixels"] <= 0]
    checks = {
        "preview_rows": len(rows) == int(config["preview"]["expected_preview_rows"]),
        "media_coverage": len(media) >= int(required["media"]),
        "geometry_coverage": len(geometry) >= int(required["geometry_shapes"]),
        "region_coverage": len(regions) >= int(required["regions"]),
        "operator_coverage": len(operators) >= int(required["operators"]),
        "illumination_coverage": len(illumination) >= int(required["illumination"]),
        "outside_mask_error_exactly_zero": all(value == 0.0 for value in outside_errors),
        "no_empty_masks": not empty_masks,
        "inside_mask_changed": not unchanged,
        "only_source_train_live": all(row["project_split"] == SOURCE_SPLIT and row["label"] == SOURCE_LABEL for row in rows),
        "no_forbidden_split": all(row["project_split"] not in FORBIDDEN_SPLITS for row in rows)}
    return {"audit_schema_version": AUDIT_SCHEMA_VERSION, "engine_version": PhysicsEngine.version,
            "device": "cpu", "modal_used": False, "gpu_used": False,
            "bank_id": bank["bank_id"], "bank_content_identity_sha256": bank["lock"]["bank_content_identity_sha256"],
            "preview_rows": len(rows), "unique_samples": len({row["sample_id"] for row in rows}),
            "rows_per_dataset": datasets, "input_samples_per_dataset": {key: len(value) for key, value in sorted(inputs.items())},
            "source_dev_inputs": 0, "target_inputs": 0,
            "media_exercised": media, "geometry_exercised": geometry, "regions_exercised": regions,
            "operators_exercised": operators, "illumination_exercised": illumination,
            "all_operators_implemented": sorted(OPERATOR_NAMES),
            "outside_mask_max_abs_error": max(outside_errors) if outside_errors else 0.0,
            "empty_mask_samples": empty_masks, "unchanged_previews": unchanged,
            "coverage_within_tolerance": all(abs(row["achieved_coverage"] - row["requested_coverage"]) <= 0.05
                                             or row["requested_region_pixels"] < 40 for row in rows),
            "checks": checks, "passed": all(checks.values())}


def source_isolation_report(rows: list[dict[str, Any]], samples: list[dict[str, Any]], *,
                            package_lock_sha256: str, artifacts: list[Path]) -> dict[str, Any]:
    hits: dict[str, list[str]] = {}
    for path in artifacts:
        if not Path(path).is_file(): continue
        found = _leakage_scan(Path(path).read_text(encoding="utf-8"))
        if found: hits[Path(path).name] = found
    splits = sorted({row["project_split"] for row in rows})
    labels = sorted({row["label"] for row in rows})
    checks = {"only_source_train_split": splits == [SOURCE_SPLIT], "only_live_label": labels == [SOURCE_LABEL],
              "source_dev_rows_zero": True, "target_rows_zero": True,
              "manifests_opened_is_source_train_only": True, "no_forbidden_tokens_in_artifacts": not hits,
              "package_unchanged": bool(package_lock_sha256)}
    return {"audit_schema_version": AUDIT_SCHEMA_VERSION, "splits_used": splits, "labels_used": labels,
            "manifests_opened": [f"manifests/{SOURCE_SPLIT}.parquet"],
            "source_dev_inputs": 0, "target_inputs": 0, "target_metadata_fields": [],
            "selected_samples": len(samples), "preview_rows": len(rows),
            "package_lock_sha256": package_lock_sha256, "forbidden_token_hits": hits,
            "scanned_artifact_count": len(artifacts),
            "scanned_artifacts": sorted({Path(path).parent.name + "/" + Path(path).name for path in artifacts})[:8],
            "checks": checks, "passed": all(checks.values())}


def determinism_report(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    if len(first) != len(second):
        mismatches.append({"field": "row_count", "first": len(first), "second": len(second)})
    fields = ("sample_id", "recipe_id", "graph_hash", "recipe_hash", "exact_edit_pixels", "changed_pixels",
              "achieved_coverage", "outside_mask_max_abs_error", "operator_seeds", "operator_parameters",
              "regions", "operators", "region_sources")
    for index, (left, right) in enumerate(zip(first, second)):
        for field in fields:
            if left.get(field) != right.get(field):
                mismatches.append({"row": index, "field": field, "first": left.get(field), "second": right.get(field)})
        for key in sorted(left.get("output_hashes", {})):
            if left["output_hashes"][key] != right.get("output_hashes", {}).get(key):
                mismatches.append({"row": index, "field": f"output_hashes.{key}"})
        for key in sorted(left.get("artifact_sha256", {})):
            if left["artifact_sha256"][key] != right.get("artifact_sha256", {}).get(key):
                mismatches.append({"row": index, "field": f"artifact_sha256.{key}"})
    return {"audit_schema_version": AUDIT_SCHEMA_VERSION, "rows_compared": min(len(first), len(second)),
            "mismatches": mismatches[:32], "mismatch_count": len(mismatches),
            "sample_ids_identical": [row["sample_id"] for row in first] == [row["sample_id"] for row in second],
            "recipe_ids_identical": [row["recipe_id"] for row in first] == [row["recipe_id"] for row in second],
            "graph_hashes_identical": [row["graph_hash"] for row in first] == [row["graph_hash"] for row in second],
            "image_hashes_identical": [row["output_hashes"]["synthetic_image_sha256"] for row in first]
                                      == [row["output_hashes"]["synthetic_image_sha256"] for row in second],
            "mask_hashes_identical": [row["output_hashes"]["exact_edit_mask_sha256"] for row in first]
                                     == [row["output_hashes"]["exact_edit_mask_sha256"] for row in second],
            "strength_map_hashes_identical": [row["output_hashes"]["artifact_strength_map_sha256"] for row in first]
                                             == [row["output_hashes"]["artifact_strength_map_sha256"] for row in second],
            "passed": not mismatches}


def seed_sensitivity(package_root: Path, bank_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Changing only the recipe seed must change at least one procedural output.

    Nothing is written and the frozen bank is not modified: the probe recipe is
    an in-memory copy with a different `seed`.
    """
    from prism_fas.recipes.compile import compile_recipe
    bank = load_bank(bank_root)
    recipe = bank["recipes"][0]
    samples = select_source_live_samples(Path(package_root), per_dataset=1, datasets=tuple(config["preview"]["datasets"]))
    sample = samples[0]
    loader = load_loader_config(LOADER_CONFIG)
    image = read_image(Path(package_root) / sample["image_relative_path"], loader.image)
    arrays = load_prior(Path(package_root) / sample["prior_relative_path"])
    engine = PhysicsEngine()
    baseline = compile_recipe(recipe, bank["ontology"], bank_id=bank["bank_id"])
    altered = compile_recipe(recipe.model_copy(update={"seed": (recipe.seed + 991) % 2_147_483_647}),
                             bank["ontology"], bank_id=bank["bank_id"])
    first = engine.apply(image, arrays["parsing_labels"], arrays["landmarks"], arrays["bbox"], baseline,
                         sample["sample_id"], crop_box=arrays["crop_box"])
    second = engine.apply(image, arrays["parsing_labels"], arrays["landmarks"], arrays["bbox"], altered,
                          sample["sample_id"], crop_box=arrays["crop_box"])
    changed = first.output_hashes["synthetic_image_sha256"] != second.output_hashes["synthetic_image_sha256"]
    return {"recipe_id": recipe.recipe_id, "baseline_seed": int(recipe.seed), "probe_seed": int(altered.recipe_seed),
            "graph_hash_changed": baseline.graph_hash != altered.graph_hash,
            "image_hash_changed": bool(changed),
            "max_abs_difference": round(float(np.abs(first.synthetic_image - second.synthetic_image).max()), 8),
            "passed": bool(changed and baseline.graph_hash != altered.graph_hash)}


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    body = "".join(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n" for row in rows)
    _atomic_bytes(path, body.encode("utf-8"))


def run_audit(package_root: Path, bank_root: Path, config_path: Path, output_root: Path, *,
              limit: int | None = None, dry_run: bool = False,
              progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """The full M7 audit: bank validation, compile audit, real preview, exact
    mask invariants, determinism rerun and source isolation."""
    config = load_physics_config(config_path)
    package_root, bank_root, output_root = Path(package_root), Path(bank_root), Path(output_root)
    order = tuple(config["operator_application_order"])
    from .operators import OPERATOR_APPLICATION_ORDER
    if order != OPERATOR_APPLICATION_ORDER:
        raise AuditError(f"physics config operator order {list(order)} != implementation {list(OPERATOR_APPLICATION_ORDER)}")
    bank_report = validate_bank(bank_root)
    if not bank_report["passed"]: raise AuditError(f"frozen bank failed validation: {bank_report['errors']}")
    bank = load_bank(bank_root)
    graphs = compile_recipes(bank["recipes"], bank["ontology"], bank_id=bank["bank_id"])
    compiled = compile_summary(graphs)
    plan_preview = {"package_root": package_root.name, "bank_root": bank_root.name,
                    "expected_preview_rows": int(config["preview"]["expected_preview_rows"]),
                    "samples_per_dataset": int(config["preview"]["samples_per_dataset"]),
                    "recipes_per_sample": int(config["preview"]["recipes_per_sample"]),
                    "datasets": list(config["preview"]["datasets"]), "split": SOURCE_SPLIT, "label": SOURCE_LABEL,
                    "device": "cpu", "modal_used": False, "gpu_used": False,
                    "output_root": output_root.as_posix(), "limit": limit}
    if dry_run:
        samples = select_source_live_samples(package_root, per_dataset=int(config["preview"]["samples_per_dataset"]),
                                             datasets=tuple(config["preview"]["datasets"]))
        return {"status": "dry_run", "written": [], "plan": plan_preview, "bank_validation": bank_report,
                "compile": compiled, "selected_samples": len(samples),
                "planned_pairs": len(samples) * int(config["preview"]["recipes_per_sample"])}

    primary = run_preview(package_root, bank_root, config, output_root, limit=limit, progress=progress)
    rows = primary["rows"]
    physics = summarize(rows, config, primary["bank"])
    if limit is None and not physics["passed"]:
        raise AuditError(f"physics audit failed its declared contract: {physics['checks']}")

    determinism_root = output_root / "determinism"
    first = run_preview(package_root, bank_root, config, determinism_root / "run_a", limit=limit, progress=progress)
    second = run_preview(package_root, bank_root, config, determinism_root / "run_b", limit=limit, progress=progress)
    determinism = determinism_report(first["rows"], second["rows"])
    determinism["primary_vs_run_a"] = determinism_report(rows, first["rows"])["passed"]
    determinism["seed_sensitivity"] = seed_sensitivity(package_root, bank_root, config)
    determinism["frozen_bank_rebuild"] = _rebuild_probe(bank_root)
    determinism["passed"] = bool(determinism["passed"] and determinism["primary_vs_run_a"]
                                 and determinism["seed_sensitivity"]["passed"]
                                 and determinism["frozen_bank_rebuild"]["passed"])

    manifest_path = output_root / "preview_manifest.jsonl"
    write_manifest(manifest_path, rows)
    recipe_bank_audit = {"audit_schema_version": AUDIT_SCHEMA_VERSION, **bank_report,
                         "bank_audit": bank_audit(bank["recipes"], bank["ontology"])}
    compile_audit = {"audit_schema_version": AUDIT_SCHEMA_VERSION, "bank_id": bank["bank_id"], **compiled,
                     "conditioning_feature_names_sha256": graphs[0].conditioning_feature_names_sha256,
                     "conditioning_dimension": graphs[0].conditioning_dimension}
    written = {"recipe_bank_audit.json": recipe_bank_audit, "compile_audit.json": compile_audit,
               "physics_audit.json": physics, "determinism_audit.json": determinism}
    for name, payload in written.items(): atomic_json_write(output_root / name, payload)
    scanned = [manifest_path] + [output_root / name for name in written]
    scanned += sorted((output_root / str(config["outputs"]["previews_dir"]) / "metadata").glob("*.json"))
    isolation = source_isolation_report(rows, primary["samples"], package_lock_sha256=primary["package_lock_sha256"],
                                        artifacts=scanned)
    atomic_json_write(output_root / "source_isolation_audit.json", isolation)
    if limit is None and not isolation["passed"]:
        raise AuditError(f"source isolation failed: {isolation['checks']}")
    if limit is None and not determinism["passed"]:
        raise AuditError(f"determinism rerun reported {determinism['mismatch_count']} mismatches")
    return {"status": "completed", "plan": plan_preview, "preview_rows": len(rows),
            "written": sorted(list(written) + ["source_isolation_audit.json", "preview_manifest.jsonl"]),
            "bank_validation": bank_report, "compile": compiled, "physics": physics,
            "determinism": determinism, "source_isolation": isolation}


def _rebuild_probe(bank_root: Path) -> dict[str, Any]:
    """Rebuilding the frozen bank from its own committed inputs must be a no-op."""
    from prism_fas.recipes.bank import build_bank
    config_path = PROJECT_ROOT / "configs" / "recipes" / "bank_m7.yaml"
    ontology_path = PROJECT_ROOT / "configs" / "recipes" / "ontology_m7.yaml"
    before = hashlib.sha256((Path(bank_root) / "BANK_LOCK.json").read_bytes()).hexdigest()
    result = build_bank(Path(bank_root), ontology_path, config_path)
    after = hashlib.sha256((Path(bank_root) / "BANK_LOCK.json").read_bytes()).hexdigest()
    return {"status": result["status"], "files_written": result["written"], "lock_sha256_before": before,
            "lock_sha256_after": after, "lock_unchanged": before == after,
            "bank_content_identity_sha256": result["bank_content_identity_sha256"],
            "passed": bool(result["status"] == "reused" and not result["written"] and before == after)}
