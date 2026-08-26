"""C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V1 — bounded, source-only, mechanistic
diagnostics run AFTER the observed BA_sep Option-1 V2 scientific FAILURE
(`reports/readiness/C9_BA_SEP_OPTION1_V2_SCIENTIFIC_FAILURE_CLOSURE.md`).

**THIS IS NOT A BA_sep REVISION, A RELIABILITY-BARRIER RESCUE PROTOCOL, A C9
PASS PROTOCOL, OR A TARGET PROTOCOL.** `synthetic_vs_real_spoof_probe` has
already, permanently, FAILED. Nothing here reruns it, overwrites it, or can
make `DETECTOR_RELIABILITY_LOCK_C` PASS — no function in this module writes
to `reports/full/c8/reliability/synthetic_vs_real_spoof_probe/` or
`reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json`.

Four tests are frozen executable
(`configs/evaluation/c9_post_failure_source_diagnostics_v1.yaml`):
`benign_jpeg_corruption`, `benign_resize_corruption`, `benign_color_corruption`
(deterministic perturbation + `reliability.score_shift`, a preregistered
source-only calibration/evaluation split derives the acceptance threshold —
never the diagnostic evaluation population, never target data) and
`cross_route_synthetic` (reuses the EXACT frozen BA_sep linear-probe
mechanics — `fit_linear_probe`, `compute_ba_sep_for_seed`, the same
hyperparameters — fit on one synthesis route's evidence, scored on the
other). The remaining four
(`residual_scale_zero`, `recipe_region_shift`, `artifact_map_swap`,
`crop_padding_interpolation`) are NOT executable today and carry a precise,
code-grounded `blocked_reason` in the frozen config rather than an invented
proxy.

Every checkpoint construction and evidence-forwarding step reuses
`synthetic_real_probe.construct_row_trainer` /
`synthetic_real_probe.forward_evidence_for_records` exactly — never a second
implementation. `run_scientific_diagnostics` is the one function that would
load real checkpoints and forward real images; nothing calls it from this
laptop, and no test in this repository calls it against real data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

#: Where every diagnostic artifact lives — deliberately NEVER the BA_sep
#: reliability directory or the DETECTOR_RELIABILITY_LOCK_C path.
DIAGNOSTICS_DIR = "reports/full/c8/reliability/post_failure_source_diagnostics_v1"
PROTOCOL_BINDING_PATH = f"{DIAGNOSTICS_DIR}/DIAGNOSTICS_PROTOCOL_BINDING.json"
POPULATION_BINDING_PATH = f"{DIAGNOSTICS_DIR}/DIAGNOSTICS_POPULATION_BINDING.json"
CHECKPOINT_BINDING_PATH = f"{DIAGNOSTICS_DIR}/DIAGNOSTICS_CHECKPOINT_BINDING.json"
RESULT_PATH = f"{DIAGNOSTICS_DIR}/DIAGNOSTICS_RESULT.json"
PER_TEST_PATH = f"{DIAGNOSTICS_DIR}/DIAGNOSTICS_PER_TEST.json"
PROVENANCE_PATH = f"{DIAGNOSTICS_DIR}/DIAGNOSTICS_PROVENANCE.json"
VERDICT_PATH = f"{DIAGNOSTICS_DIR}/DIAGNOSTICS_VERDICT.json"

RESULT_ARTIFACT_PATHS: dict[str, str] = {
    "result": RESULT_PATH, "per_test": PER_TEST_PATH,
    "provenance": PROVENANCE_PATH, "verdict": VERDICT_PATH,
}

PROTOCOL_CONFIG_PATH = "configs/evaluation/c9_post_failure_source_diagnostics_v1.yaml"
EXECUTABLE_TESTS: tuple[str, ...] = (
    "benign_jpeg_corruption", "benign_resize_corruption", "benign_color_corruption",
    "cross_route_synthetic",
)
BENIGN_CORRUPTION_TESTS: tuple[str, ...] = (
    "benign_jpeg_corruption", "benign_resize_corruption", "benign_color_corruption",
)
ROUTES: tuple[str, ...] = ("physics", "gpat")


class PostFailureDiagnosticsError(RuntimeError):
    """A post-failure diagnostic cannot proceed with the inputs given."""


# ==============================================================================
# 1. Protocol
# ==============================================================================

def load_protocol(repo: Path) -> dict[str, Any]:
    """The frozen post-failure diagnostics protocol, or a refusal naming why
    it is absent. Never invents a value."""
    import yaml

    path = Path(repo) / PROTOCOL_CONFIG_PATH
    if not path.is_file():
        raise PostFailureDiagnosticsError(
            f"C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V1 is not frozen (expected "
            f"{PROTOCOL_CONFIG_PATH} to exist and declare status: FROZEN_NOT_RUN)")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise PostFailureDiagnosticsError(f"{PROTOCOL_CONFIG_PATH} did not parse: {error}") from error
    if not isinstance(payload, dict) or payload.get("status") != "FROZEN_NOT_RUN":
        raise PostFailureDiagnosticsError(
            f"{PROTOCOL_CONFIG_PATH} does not declare status: FROZEN_NOT_RUN")
    return payload


#: Metadata keys excluded from the protocol identity — provenance, never a
#: result-affecting field. Mirrors detector_reliability's own exclusion
#: policy, with this protocol's own metadata keys.
_PROTOCOL_IDENTITY_EXCLUDED_KEYS = frozenset({
    "frozen_on", "approved_by", "status", "schema_version", "decision_id",
    "document_kind", "no_diagnostic_metric_observed_before_freeze",
})


def protocol_identity(protocol: Mapping[str, Any]) -> str:
    """sha256 over every result-affecting protocol field, sorted keys, no
    timestamps. Changes if and only if a result-affecting field changes."""
    material = {key: value for key, value in protocol.items()
               if key not in _PROTOCOL_IDENTITY_EXCLUDED_KEYS}
    return hashlib.sha256(json.dumps(
        material, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def active_protocol_identity(repo: Path) -> str:
    return protocol_identity(load_protocol(repo))


# ==============================================================================
# 2. Deterministic benign-corruption perturbations — exact, frozen operations
# ==============================================================================

def jpeg_corrupt(image: np.ndarray, *, quality: int = 50) -> np.ndarray:
    """Encode/decode through JPEG at a frozen quality. `image` is `[3,H,W]`
    or `[H,W,3]` float32 in `[0,1]`; returns the same shape/dtype/range."""
    from PIL import Image

    array, was_chw = _to_hwc_uint8(image)
    from io import BytesIO
    buffer = BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    decoded = np.asarray(Image.open(buffer).convert("RGB"))
    return _from_hwc_uint8(decoded, was_chw)


def resize_corrupt(image: np.ndarray, *, downscale_factor: float = 0.5) -> np.ndarray:
    """Downscale then upscale back to the original size, bilinear both ways —
    a benign resampling chain, never a crop or a padding change."""
    from PIL import Image

    array, was_chw = _to_hwc_uint8(image)
    height, width = array.shape[0], array.shape[1]
    small = (max(1, round(width * downscale_factor)), max(1, round(height * downscale_factor)))
    pil_image = Image.fromarray(array, mode="RGB")
    downscaled = pil_image.resize(small, Image.BILINEAR)
    upscaled = np.asarray(downscaled.resize((width, height), Image.BILINEAR))
    return _from_hwc_uint8(upscaled, was_chw)


def color_corrupt(image: np.ndarray, *,
                  gain: tuple[float, float, float] = (1.15, 1.00, 0.90)) -> np.ndarray:
    """A fixed per-channel RGB gain, clipped to `[0,1]` — deterministic, no
    external codec, no color-space round trip."""
    array = np.asarray(image, dtype=np.float64)
    is_chw = array.shape[0] == 3 and array.ndim == 3
    if is_chw:
        gains = np.asarray(gain, dtype=np.float64).reshape(3, 1, 1)
    else:
        gains = np.asarray(gain, dtype=np.float64).reshape(1, 1, 3)
    return np.clip(array * gains, 0.0, 1.0).astype(np.float32)


def _to_hwc_uint8(image: np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(image, dtype=np.float64)
    was_chw = array.shape[0] == 3 and array.ndim == 3
    if was_chw:
        array = np.transpose(array, (1, 2, 0))
    return np.clip(array * 255.0, 0, 255).astype(np.uint8), was_chw


def _from_hwc_uint8(array: np.ndarray, was_chw: bool) -> np.ndarray:
    out = array.astype(np.float32) / 255.0
    if was_chw:
        out = np.transpose(out, (2, 0, 1))
    return out.astype(np.float32)


CORRUPTION_FUNCTIONS: dict[str, Any] = {
    "benign_jpeg_corruption": jpeg_corrupt,
    "benign_resize_corruption": resize_corrupt,
    "benign_color_corruption": color_corrupt,
}


# ==============================================================================
# 3. Group-safe calibration/evaluation split — reuses the exact BA_sep rule
# ==============================================================================

def calibration_evaluation_split(stable_group_identities: Sequence[str], *,
                                 namespace: str, seed: int,
                                 calibration_fraction: float = 0.5
                                 ) -> dict[str, list[str]]:
    """Deterministic, group-safe 50/50 split — the SAME hash rule
    `synthetic_real_probe.split_bucket` uses (never a second split rule),
    applied here to `(namespace, seed, "diagnostics", group_identity)`."""
    from prism_fas.evaluation.synthetic_real_probe import split_bucket

    calibration: list[str] = []
    evaluation: list[str] = []
    for group_id in stable_group_identities:
        bucket = split_bucket(namespace, seed, "diagnostics", group_id,
                              train_fraction=calibration_fraction)
        (calibration if bucket == "train" else evaluation).append(group_id)
    if not calibration or not evaluation:
        raise PostFailureDiagnosticsError(
            "calibration/evaluation split produced an empty group; fail closed "
            "rather than derive a threshold or evaluate over zero samples")
    return {"calibration": calibration, "evaluation": evaluation}


# ==============================================================================
# 4. Benign-corruption threshold derivation and verdict — pure arithmetic
# ==============================================================================

def derive_corruption_threshold(calibration_shifts: Sequence[float]) -> dict[str, Any]:
    """`threshold = calibration_mean_shift + 3 * calibration_std_shift`
    (population std, ddof=0) — computed ONLY from the calibration group,
    never the evaluation group. Fails closed on an empty calibration set."""
    values = np.asarray(calibration_shifts, dtype=np.float64)
    if values.size == 0:
        raise PostFailureDiagnosticsError(
            "cannot derive a threshold from zero calibration samples")
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    threshold = mean + 3.0 * std
    return {"calibration_mean_shift": mean, "calibration_std_shift": std,
           "threshold": threshold, "formula": "calibration_mean_shift + 3 * calibration_std_shift",
           "calibration_samples": int(values.size)}


def corruption_verdict(evaluation_mean_shift: float, threshold: float) -> str:
    """`PASS` iff the evaluation group's mean shift does not exceed the
    calibration-derived threshold. Ties PASS (`<=`), matching the project's
    existing convention for BA_sep's own ceiling."""
    return "PASS" if float(evaluation_mean_shift) <= float(threshold) else "FAIL"


# ==============================================================================
# 5. cross_route_synthetic — reuses the exact frozen BA_sep probe mechanics
# ==============================================================================

def resolve_synthetic_population_by_route(repo: Path, arm: str, *,
                                          domains: Sequence[str] = ("casia_fasd", "msu_mfsd")
                                          ) -> dict[str, list[Any]]:
    """`ARM`'s frozen C6 matched bank, partitioned by its existing `route`
    field — never a second bank resolver, never bank regeneration.

    Reuses `synthetic_real_probe.resolve_synthetic_population`'s exact
    group-identity resolution (`source_record_id` via
    `live_target_sample_id`), then partitions the already-resolved records
    by route using the bank's own `route` field, read directly here since
    `PopulationRecord` itself carries no route (route is not part of the
    BA_sep evidence contract and must not leak into it).
    """
    from prism_fas.detector.c6_bank import open_arm_bank
    from prism_fas.evaluation import c6_evidence
    from prism_fas.evaluation.synthetic_real_probe import (ARMS, SYNTHETIC_SPOOF_CLASS,
                                                            PopulationRecord,
                                                            SyntheticRealProbeError,
                                                            _source_record_id_by_sample_id)
    from prism_fas.pipeline.adapters import sources

    if arm not in ARMS:
        raise PostFailureDiagnosticsError(f"unknown arm {arm!r}; covers {ARMS}")
    inputs = sources.verify_detector_inputs(repo, arms=(arm,))
    evidence = c6_evidence.verify_c6_evidence(repo).bank(arm)
    bank = open_arm_bank(
        repo, arm=arm, evidence=evidence,
        candidates_root=Path(repo) / inputs["candidates_root"],
        package_identity=inputs["package_identity"],
        recipe_bank_identity=inputs["recipe_bank_identity"])
    try:
        lookup = _source_record_id_by_sample_id(repo, domains=domains)
    except SyntheticRealProbeError as error:
        raise PostFailureDiagnosticsError(str(error)) from error

    by_route: dict[str, list[Any]] = {route: [] for route in ROUTES}
    unmapped: list[str] = []
    for row in bank.rows:
        if row["live_target_dataset"] not in domains:
            continue
        route = str(row.get("route") or "")
        if route not in ROUTES:
            continue
        synthetic_id = str(row["synthetic_id"])
        live_target_sample_id = str(row.get("live_target_sample_id") or "").strip()
        source_record_id = lookup.get(live_target_sample_id) if live_target_sample_id else None
        if not source_record_id:
            unmapped.append(synthetic_id)
            continue
        by_route[route].append(PopulationRecord(
            sample_identity=synthetic_id, stable_group_identity=source_record_id,
            source_domain=str(row["live_target_dataset"]), label=SYNTHETIC_SPOOF_CLASS))
    if unmapped:
        raise PostFailureDiagnosticsError(
            f"arm {arm!r}: {len(unmapped)} synthetic candidate(s) have an unmappable "
            f"live_target_sample_id; fail closed")
    for route in ROUTES:
        if not by_route[route]:
            raise PostFailureDiagnosticsError(
                f"arm {arm!r}: route {route!r} resolves zero synthetic candidates; "
                "fail closed rather than run a cross-route comparison against an "
                "empty population")
    return by_route


def cross_route_ba(train_features: np.ndarray, train_labels: np.ndarray,
                   evaluate_features: np.ndarray, evaluate_labels: np.ndarray) -> float:
    """Fit the FROZEN BA_sep linear probe on one route's evidence, score it
    on the other's — reuses `compute_ba_sep_for_seed` verbatim (same
    z-score/train-only normalization, same LBFGS config, same threshold);
    never a new numeric rule."""
    from prism_fas.evaluation.synthetic_real_probe import compute_ba_sep_for_seed

    result = compute_ba_sep_for_seed(train_features, train_labels,
                                     evaluate_features, evaluate_labels)
    return float(result["balanced_accuracy"])


# ==============================================================================
# 6. source_dev LIVE population — for the three benign-corruption tests
# ==============================================================================

def resolve_source_dev_live_records(repo: Path, *,
                                    domains: Sequence[str] = ("casia_fasd", "msu_mfsd")
                                    ) -> list[dict[str, str]]:
    """Every `source_dev` LIVE row's identities — read-only metadata
    resolution, mirrors `synthetic_real_probe.resolve_real_spoof_population`'s
    pattern exactly, but over `VALIDATION_SPLIT`
    (`source_dev`, never `source_train`) and `label_live_spoof == "live"`
    (never spoof, never target). Fails closed on a row with no
    `source_record_id`.
    """
    from prism_fas.data.loader.config import VALIDATION_SPLIT, load_loader_config
    from prism_fas.data.loader.loose_dataset import CanonicalPackageDataset
    from prism_fas.pipeline.adapters import sources

    inputs = sources.verify_detector_inputs(repo)
    package_root = Path(repo) / inputs["package_root"]
    loader_config = load_loader_config(Path(repo) / "configs/data/loader_m4.yaml")
    dataset = CanonicalPackageDataset(package_root, VALIDATION_SPLIT, loader_config, mode="validation")

    records: list[dict[str, str]] = []
    for row in dataset.index.rows:
        if row["dataset"] not in domains or row["label_live_spoof"] != "live":
            continue
        source_record_id = str(row.get("source_record_id") or "").strip()
        if not source_record_id:
            raise PostFailureDiagnosticsError(
                f"source_dev sample_id={row['sample_id']!r} has no source_record_id; "
                "fail closed")
        records.append({"sample_id": str(row["sample_id"]),
                        "stable_group_identity": source_record_id,
                        "source_domain": str(row["dataset"])})
    return records


# ==============================================================================
# 7. Real forward passes — the ONE boundary genuinely unverifiable here.
#
# Reuses `synthetic_real_probe.construct_row_trainer` (the exact C8
# row-construction path) for checkpoint loading, and
# `prism_fas.detector.dataset.M9ValidationDataset`/`collate_items` (the exact
# class C8's own cross-source evaluation already uses for source_dev) for
# real image access — never a second construction path, never a second
# image pipeline. Nothing in this repository's test suite calls these
# against real data; every test exercises the pure functions above with
# fixtures, and mocks this boundary exactly as
# `synthetic_real_probe.construct_row_trainer`/`forward_evidence_for_records`
# already are in the BA_sep test suite.
# ==============================================================================

def forward_corruption_evidence_for_arm(repo: Path, checkpoints: Sequence[Any], *,
                                        sample_ids: Sequence[str], corruption_fn: Any,
                                        domains: Sequence[str] = ("casia_fasd", "msu_mfsd")
                                        ) -> dict[str, dict[str, np.ndarray]]:
    """For one arm's 5 checkpoints and a set of `source_dev` LIVE
    `sample_id`s: the BEFORE and AFTER (corrupted) evidence, averaged over
    the 5 checkpoints exactly as `average_checkpoint_evidence` does for
    BA_sep. Returns `{sample_id: {"before": vector, "after": vector}}`.
    """
    import torch
    from dataclasses import replace

    from prism_fas.detector.dataset import LIVE, M9ValidationDataset, collate_items
    from prism_fas.evaluation.synthetic_real_probe import (average_checkpoint_evidence,
                                                            construct_row_trainer,
                                                            forward_checkpoint_evidence)
    from prism_fas.pipeline.adapters import sources

    if not checkpoints:
        raise PostFailureDiagnosticsError("at least one checkpoint is required")
    inputs = sources.verify_detector_inputs(repo)
    package_root = Path(repo) / inputs["package_root"]

    before_by_checkpoint: list[dict[str, np.ndarray]] = []
    after_by_checkpoint: list[dict[str, np.ndarray]] = []
    for binding in checkpoints:
        trainer = construct_row_trainer(repo, binding)
        dataset = M9ValidationDataset(package_root, trainer.loader_config,
                                      cache_root=trainer.cache_root, domains=domains)
        wanted = set(sample_ids)
        positions = [position for position in dataset.positions
                    if dataset.sample_id_of(position) in wanted]
        found = {dataset.sample_id_of(p) for p in positions}
        missing = wanted - found
        if missing:
            raise PostFailureDiagnosticsError(
                f"{len(missing)} requested source_dev sample_id(s) not found "
                f"(first offenders: {sorted(missing)[:5]}); fail closed")
        before: dict[str, np.ndarray] = {}
        after: dict[str, np.ndarray] = {}
        with torch.no_grad():
            for position in positions:
                item = dataset.item(position)
                if item.label != LIVE:
                    raise PostFailureDiagnosticsError(
                        f"{item.sample_id!r} is not LIVE; benign corruption tests never "
                        "forward a spoof sample")
                sample_id = item.sample_id
                clean_batch = collate_items([item]).to(trainer.device)
                before[sample_id] = forward_checkpoint_evidence(trainer.model, clean_batch)
                corrupted_item = replace(item, image=corruption_fn(item.image))
                corrupted_batch = collate_items([corrupted_item]).to(trainer.device)
                after[sample_id] = forward_checkpoint_evidence(trainer.model, corrupted_batch)
        before_by_checkpoint.append(before)
        after_by_checkpoint.append(after)

    merged: dict[str, dict[str, np.ndarray]] = {}
    for sample_id in sample_ids:
        before_vectors = [d[sample_id] for d in before_by_checkpoint]
        after_vectors = [d[sample_id] for d in after_by_checkpoint]
        merged[sample_id] = {"before": average_checkpoint_evidence(before_vectors),
                             "after": average_checkpoint_evidence(after_vectors)}
    return merged


# ==============================================================================
# 8. cross_route_synthetic — the real, joint, per-arm orchestration.
# ==============================================================================

def _group_split(records: Sequence[Any], *, namespace: str, seed: int) -> dict[str, list[Any]]:
    from prism_fas.evaluation.synthetic_real_probe import assign_splits, verify_group_safe_split

    split = assign_splits(records, namespace=namespace, probe_seed=seed)
    verify_group_safe_split(split)
    return split


def _balanced_pair(real: Sequence[Any], synthetic: Sequence[Any]) -> tuple[list[Any], list[Any]]:
    """Deterministic 1:1 balance by `sample_identity` order — the diagnostic
    analogue of `balance_classes`, scoped to one arm/one direction rather
    than three arms jointly."""
    n = min(len(real), len(synthetic))
    if n == 0:
        raise PostFailureDiagnosticsError(
            "a cross-route cell resolves N=0; refusing to fit or score over an "
            "empty population")
    ordered_real = sorted(real, key=lambda r: r.sample_identity)[:n]
    ordered_synthetic = sorted(synthetic, key=lambda r: r.sample_identity)[:n]
    return ordered_real, ordered_synthetic


def run_cross_route_diagnostic_for_arm(repo: Path, arm: str, checkpoints: Sequence[Any], *,
                                       protocol: Mapping[str, Any]) -> dict[str, Any]:
    """One arm's full `cross_route_synthetic` diagnostic: both directions,
    all three frozen probe seeds, reusing `construct_row_trainer` /
    `forward_evidence_for_records` / `fit_linear_probe` /
    `compute_ba_sep_for_seed` exactly."""
    from prism_fas.evaluation.synthetic_real_probe import (forward_evidence_for_records,
                                                            resolve_real_spoof_population)

    test_config = protocol["tests"]["cross_route_synthetic"]
    namespace = test_config["split_hash_namespace"]
    seeds = [int(seed) for seed in test_config["probe_seed_values"]]

    real_spoof = resolve_real_spoof_population(repo)
    by_route = resolve_synthetic_population_by_route(repo, arm)

    # One forward pass per unique sample per checkpoint, across every seed's
    # split — evidence does not depend on the probe seed, only membership
    # does, exactly the same optimization BA_sep's own execution already
    # relies on.
    needed = list(real_spoof) + list(by_route["physics"]) + list(by_route["gpat"])
    per_checkpoint_evidence: list[dict[str, np.ndarray]] = []
    for binding in checkpoints:
        from prism_fas.evaluation.synthetic_real_probe import construct_row_trainer

        trainer = construct_row_trainer(repo, binding)
        per_checkpoint_evidence.append(forward_evidence_for_records(trainer, needed))
    if len(per_checkpoint_evidence) != len(checkpoints):
        raise PostFailureDiagnosticsError(f"arm {arm!r}: checkpoint forwarding incomplete")

    from prism_fas.evaluation.synthetic_real_probe import average_checkpoint_evidence

    evidence: dict[str, np.ndarray] = {}
    for record in needed:
        vectors = [d[record.sample_identity] for d in per_checkpoint_evidence]
        evidence[record.sample_identity] = average_checkpoint_evidence(vectors)

    per_seed_ba: dict[int, dict[str, float]] = {}
    for seed in seeds:
        real_split = _group_split(real_spoof, namespace=namespace, seed=seed)
        route_splits = {route: _group_split(by_route[route], namespace=namespace, seed=seed)
                        for route in ROUTES}
        directions: dict[str, float] = {}
        for train_route, eval_route in (("physics", "gpat"), ("gpat", "physics")):
            train_real, train_synth = _balanced_pair(
                real_split["train"], route_splits[train_route]["train"])
            eval_real, eval_synth = _balanced_pair(
                real_split["validation"], route_splits[eval_route]["validation"])
            train_features = np.array([evidence[r.sample_identity] for r in train_real]
                                      + [evidence[r.sample_identity] for r in train_synth])
            train_labels = np.array([0] * len(train_real) + [1] * len(train_synth))
            eval_features = np.array([evidence[r.sample_identity] for r in eval_real]
                                     + [evidence[r.sample_identity] for r in eval_synth])
            eval_labels = np.array([0] * len(eval_real) + [1] * len(eval_synth))
            directions[f"{train_route}_to_{eval_route}"] = cross_route_ba(
                train_features, train_labels, eval_features, eval_labels)
        per_seed_ba[seed] = directions

    from prism_fas.evaluation.detector_reliability import BA_SEP_CEILING

    mean_by_direction = {
        direction: float(np.mean([per_seed_ba[seed][direction] for seed in seeds]))
        for direction in ("physics_to_gpat", "gpat_to_physics")}
    mean_cross_route_ba = float(np.mean(list(mean_by_direction.values())))
    # Reuses the SAME frozen 0.75 ceiling BA_sep Option-1 V2 already froze
    # (§ of the diagnostics protocol: "reuses a frozen threshold rather than
    # inventing one") — never a second, independently-invented ceiling.
    verdict = "PASS" if mean_cross_route_ba <= BA_SEP_CEILING else "FAIL"
    return {"arm": arm, "per_seed_ba_by_direction": per_seed_ba,
           "mean_ba_by_direction": mean_by_direction,
           "mean_cross_route_ba": mean_cross_route_ba, "verdict": verdict}


# ==============================================================================
# 9. Benign corruption — the real, joint, per-arm orchestration.
# ==============================================================================

def run_benign_corruption_diagnostic_for_arm(repo: Path, test_id: str, arm: str,
                                             checkpoints: Sequence[Any], *,
                                             calibration_ids: Sequence[str],
                                             evaluation_ids: Sequence[str]) -> dict[str, Any]:
    """One arm's full benign-corruption diagnostic: forwards every
    calibration + evaluation `source_dev` LIVE sample, clean and corrupted,
    through all 5 of the arm's checkpoints; derives the threshold from the
    calibration group's `p_global` shift only; scores the evaluation group
    against it via `reliability.score_shift`."""
    from prism_fas.evaluation.reliability import score_shift

    corruption_fn = CORRUPTION_FUNCTIONS[test_id]
    sample_ids = list(calibration_ids) + list(evaluation_ids)
    evidence = forward_corruption_evidence_for_arm(
        repo, checkpoints, sample_ids=sample_ids, corruption_fn=corruption_fn)

    def _p_global(vector: np.ndarray) -> float:
        return float(vector[1])   # [global_logit_G, p_global]

    calibration_shifts = [_p_global(evidence[sid]["after"]) - _p_global(evidence[sid]["before"])
                          for sid in calibration_ids]
    threshold_report = derive_corruption_threshold(calibration_shifts)

    evaluation_before = [_p_global(evidence[sid]["before"]) for sid in evaluation_ids]
    evaluation_after = [_p_global(evidence[sid]["after"]) for sid in evaluation_ids]
    shift_report = score_shift(evaluation_before, evaluation_after)

    verdict = corruption_verdict(shift_report["mean_shift"], threshold_report["threshold"])
    return {"arm": arm, "test_id": test_id, "threshold": threshold_report,
           "evaluation": shift_report, "verdict": verdict}


__all__ = ["DIAGNOSTICS_DIR", "PROTOCOL_BINDING_PATH", "POPULATION_BINDING_PATH",
           "CHECKPOINT_BINDING_PATH", "RESULT_PATH", "PER_TEST_PATH", "PROVENANCE_PATH",
           "VERDICT_PATH", "RESULT_ARTIFACT_PATHS", "PROTOCOL_CONFIG_PATH",
           "EXECUTABLE_TESTS", "BENIGN_CORRUPTION_TESTS", "ROUTES",
           "PostFailureDiagnosticsError", "load_protocol", "protocol_identity",
           "active_protocol_identity", "jpeg_corrupt", "resize_corrupt", "color_corrupt",
           "CORRUPTION_FUNCTIONS", "calibration_evaluation_split",
           "derive_corruption_threshold", "corruption_verdict",
           "resolve_synthetic_population_by_route", "cross_route_ba",
           "resolve_source_dev_live_records", "forward_corruption_evidence_for_arm",
           "run_cross_route_diagnostic_for_arm", "run_benign_corruption_diagnostic_for_arm"]
