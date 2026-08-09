"""G7 — label-free target prediction, and the PREDICTION_LOCK that freezes it.

G7 may open exactly four things: the frozen selected checkpoint, the frozen source
calibration, the target FEATURE manifest, and the target images/priors that
inference needs. It cannot resolve a target label path — not because it chooses not
to, but because the firewall's permission table denies the label root to `G7` and
because `TargetInferenceBatch` has no field a label could live in.

Prediction schema: Table 57, versioned and nullable
(`docs/M10_TARGET_EVALUATION_CONTRACT.md` section 2). A baseline without a regional
branch or a PromptHead writes `null` for the components it does not have, never
`0.0`: a fabricated zero would enter the fusion arithmetic and the report as a
measured value, and `null` cannot.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
import numpy as np
import pyarrow as pa
import yaml
from prism_fas.data.package.manifests import write_manifest
from prism_fas.utils.core import atomic_json_write
from .contracts import (DECISIONS, M10_PREDICTION_LOCK_SCHEMA_VERSION, M10_PREDICTION_SCHEMA_VERSION,
                        M10ContractError, PredictionLockError, stable_identity)
from .firewall import TargetLabelFirewall
from .video_aggregation import aggregate_frames, aggregation_config

PREDICTION_FILE = "target_predictions.parquet"
PREDICTION_LOCK_FILE = "PREDICTION_LOCK.json"

# Table 57 columns, plus the applicability status fields the nullable schema needs.
PREDICTION_SCHEMA = pa.schema([
    ("sample_id", pa.string()), ("video_id", pa.string()), ("frame_id", pa.int64()),
    ("p_global", pa.float64()), ("s_region", pa.float64()), ("p_prompt", pa.float64()),
    ("s_final", pa.float64()), ("decision_score", pa.float64()),
    ("confidence", pa.float64()), ("decision", pa.string()),
    ("top_region_ids", pa.list_(pa.int64())), ("region_distances", pa.list_(pa.float64())),
    ("checkpoint_hash", pa.string()), ("calibration_hash", pa.string()),
    ("inference_config_hash", pa.string()), ("region_status", pa.string()),
    ("prompt_status", pa.string()), ("variant", pa.string())])

# A prediction file may never carry any of these, whatever produced it.
FORBIDDEN_PREDICTION_COLUMNS = frozenset({
    "label", "true_label", "label_live_spoof", "true_target", "target", "class_target",
    "attack_type", "attack_family", "taxonomy", "subject_id", "session_id",
    "identity_embedding", "source_path", "crop_relative_path", "official_split",
    "private_label", "dataset"})
NULLABLE_COLUMNS = ("s_region", "p_prompt")
APPLICABILITY = ("computed", "not_applicable")


@dataclass(frozen=True)
class TargetInferenceBatch:
    """A target batch with NO field a label could live in.

    The M9 `DetectorBatch` carries `label`; reusing it here would mean a target
    batch that structurally *could* hold one. This mirrors the M4 decision to give
    target samples their own contract rather than a nullable label column.
    """
    image: Any                      # [B,3,224,224] float in [0,1]
    region_priors: Any              # [B,R,Ht,Wt]
    visibility: Any                 # [B,R]
    is_synthetic: Any               # [B] bool, always False here
    sample_ids: tuple[str, ...]
    video_ids: tuple[str, ...]
    frame_ids: tuple[int, ...]
    attack_region_mask: Any = None  # always None: a target attack mask does not exist

    def __post_init__(self) -> None:
        if self.attack_region_mask is not None:
            raise M10ContractError("a target inference batch cannot carry an attack region mask")


@dataclass(frozen=True)
class VariantCapabilities:
    """Which evidence terms the variant that produced a row actually has.

    Derived from the ONE resolved variant object the trainer used, so a prediction
    can never claim a term the model it came from does not compute.

    `has_region_detail` is separate on purpose. A single global center DOES fuse a
    regional evidence term — `s_region` is measured and belongs in the row — but it
    has no per-region decomposition, so `top_region_ids` and `region_distances` stay
    empty rather than reporting site 0 as though it were `left_eye`.
    """
    has_region: bool
    has_prompt: bool
    has_region_detail: bool = True

    @classmethod
    def from_variant(cls, variant: Any) -> "VariantCapabilities":
        return cls(has_region=bool(variant.fuses_region_evidence),
                   has_prompt=bool(variant.fuses_prompt_evidence),
                   has_region_detail=variant.manifold_scope == "regional")

    @classmethod
    def from_flags(cls, flags: dict[str, Any]) -> "VariantCapabilities":
        from prism_fas.detector.variant import ResolvedExperimentVariant
        return cls.from_variant(ResolvedExperimentVariant.resolve(dict(flags)))


def load_evaluation_config(path: Path,
                           reveal_path: Path = Path("reports/m10/TARGET_LABEL_REVEAL.json")
                           ) -> dict[str, Any]:
    """Load the frozen evaluation config and enforce the one-way reveal flag.

    `target_labels_revealed` may be `true` ONLY once the authorized reveal record
    exists, so the flag cannot be flipped by editing a YAML file: the artifact that
    records the first authorized read is what permits it. It is one-way — nothing
    here or anywhere else sets it back, because a label cannot be un-opened.
    """
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict): raise M10ContractError(f"{Path(path).name} is not a mapping")
    if payload.get("evaluation_schema_version") != "m10-target-evaluation-v1":
        raise M10ContractError(f"evaluation schema {payload.get('evaluation_schema_version')!r} "
                               f"!= 'm10-target-evaluation-v1'")
    revealed = payload.get("target_labels_revealed")
    if revealed is True:
        if not Path(reveal_path).is_file():
            raise M10ContractError("the evaluation config declares target_labels_revealed: true but "
                                   f"{Path(reveal_path).name} does not exist; the flag is one-way and "
                                   "is set by the authorized reveal, never by hand")
    elif revealed is not False:
        raise M10ContractError("the evaluation config must declare target_labels_revealed: false "
                               "until the authorized reveal")
    return payload


def build_prediction_row(*, sample_id: str, video_id: str, frame_id: int, p_global: float,
                         s_region: float | None, p_prompt: float | None,
                         threshold: float, unknown_threshold: float | None,
                         confidence: float | None = None,
                         top_region_ids: Sequence[int], region_distances: Sequence[float],
                         checkpoint_hash: str, calibration_hash: str, inference_config_hash: str,
                         variant: str) -> dict[str, Any]:
    """One Table 57 row. `s_final` is the Table 34 fusion with absent factors omitted.

    `1 - (1 - p_global)` collapses to `p_global` when the regional and prompt
    evidence terms do not exist. That is the same formula with fewer factors, not a
    second formula, and it is why an absent term is `null` rather than `0.0`.

    **`decision_score` is `p_global`, not `s_final`** (contract section 2b). The
    frozen temperature and the frozen threshold were both fitted by `run_g6` on
    `output.global_logit` alone, and the checkpoint was selected on
    `sigmoid(global_logit)`, so `p_global` is the only quantity the frozen operating
    point belongs to. `s_final >= threshold` is a category error: `s_final` is
    pointwise >= `p_global`, and on `source_dev` it puts every bona-fide sample above
    the threshold (BPCER 1.0, ACER 0.5). For a variant with no regional or prompt
    branch the two are equal by construction, so this changes nothing for B00-B05.
    `s_final` is still recorded and is still reported for the threshold-free metrics.
    """
    factors = [float(p_global)]
    if s_region is not None: factors.append(float(s_region))
    if p_prompt is not None: factors.append(float(p_prompt))
    s_final = 1.0
    for value in factors: s_final *= (1.0 - value)
    s_final = 1.0 - s_final
    decision_score = float(p_global)
    # One definition of confidence, derived from the DECISION score, so a caller can
    # never hand in a confidence that disagrees with the decision it belongs to.
    if confidence is None: confidence = max(decision_score, 1.0 - decision_score)
    decision = ("reject" if unknown_threshold is not None and float(confidence) < float(unknown_threshold)
                else ("spoof" if decision_score >= float(threshold) else "live"))
    return {"sample_id": str(sample_id), "video_id": str(video_id), "frame_id": int(frame_id),
            "p_global": float(p_global),
            "s_region": (None if s_region is None else float(s_region)),
            "p_prompt": (None if p_prompt is None else float(p_prompt)),
            "s_final": float(s_final), "decision_score": decision_score,
            "confidence": float(confidence), "decision": decision,
            "top_region_ids": [int(value) for value in top_region_ids],
            "region_distances": [float(value) for value in region_distances],
            "checkpoint_hash": str(checkpoint_hash), "calibration_hash": str(calibration_hash),
            "inference_config_hash": str(inference_config_hash),
            "region_status": "computed" if s_region is not None else "not_applicable",
            "prompt_status": "computed" if p_prompt is not None else "not_applicable",
            "variant": str(variant)}


def validate_predictions(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Schema, finiteness, applicability and the no-label rule. Raises on any breach."""
    if not rows: raise M10ContractError("a prediction file cannot be empty")
    expected = {field.name for field in PREDICTION_SCHEMA}
    leaked = sorted(FORBIDDEN_PREDICTION_COLUMNS & {key for row in rows for key in row})
    if leaked: raise M10ContractError(f"target predictions expose forbidden columns: {leaked}")
    seen_ids: set[str] = set()
    for row in rows:
        missing = sorted(expected - set(row))
        if missing: raise M10ContractError(f"prediction row is missing {missing}")
        extra = sorted(set(row) - expected)
        if extra: raise M10ContractError(f"prediction row carries undeclared columns {extra}")
        if row["sample_id"] in seen_ids: raise M10ContractError(f"duplicate sample_id {row['sample_id']!r}")
        seen_ids.add(str(row["sample_id"]))
        for name in ("p_global", "s_final", "decision_score", "confidence"):
            value = float(row[name])
            if not np.isfinite(value): raise M10ContractError(f"{name} is not finite")
            if value < -1e-9 or value > 1.0 + 1e-9: raise M10ContractError(f"{name}={value} outside [0,1]")
        for name, status in (("s_region", "region_status"), ("p_prompt", "prompt_status")):
            if row[status] not in APPLICABILITY: raise M10ContractError(f"{status}={row[status]!r}")
            if (row[name] is None) != (row[status] == "not_applicable"):
                raise M10ContractError(f"{name} and {status} disagree; an absent term is null, never 0.0")
            if row[name] is not None and not (0.0 - 1e-9 <= float(row[name]) <= 1.0 + 1e-9):
                raise M10ContractError(f"{name}={row[name]} outside [0,1]")
        if row["decision"] not in DECISIONS: raise M10ContractError(f"decision={row['decision']!r}")
        if (row["region_status"] == "not_applicable"
                and (row["top_region_ids"] or row["region_distances"])):
            raise M10ContractError("a variant without regions cannot report region ids or distances")
    videos = sorted({str(row["video_id"]) for row in rows})
    return {"rows": len(rows), "videos": len(videos), "unique_sample_ids": len(seen_ids),
            "region_status": sorted({str(row["region_status"]) for row in rows}),
            "prompt_status": sorted({str(row["prompt_status"]) for row in rows}),
            "labels_present": False}


def prediction_logical_identity(rows: Sequence[dict[str, Any]]) -> str:
    """Hashes the LOGICAL rows, not the parquet bytes.

    Same reason M7's pair plan and M8's bank identity bind logical rows: pyarrow
    versions write different bytes for identical data, and a prediction must not
    change identity because of the writer's version.
    """
    body = [[str(row["sample_id"]), str(row["video_id"]), int(row["frame_id"]),
             round(float(row["s_final"]), 12), round(float(row["decision_score"]), 12),
             round(float(row["confidence"]), 12), str(row["decision"])] for row in rows]
    return stable_identity({"schema_version": M10_PREDICTION_SCHEMA_VERSION,
                            "rows": sorted(body, key=lambda item: item[0])})


def write_predictions(path: Path, rows: Sequence[dict[str, Any]], *, variant: str,
                      firewall: TargetLabelFirewall | None = None, dry_run: bool = False) -> Path:
    target = Path(path)
    if firewall is not None: firewall.check_write("G7", target)
    validate_predictions(rows)
    if dry_run: return target
    write_manifest(target, [dict(row) for row in rows], PREDICTION_SCHEMA,
                   {"stage": "M10_G7", "split": "target_test", "variant": str(variant),
                    "labels": "not_accessed",
                    "prediction_schema_version": M10_PREDICTION_SCHEMA_VERSION})
    return target


def read_predictions(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    table = pq.read_table(Path(path))
    rows = table.to_pylist()
    validate_predictions(rows)
    return rows


# --- PREDICTION_LOCK ---------------------------------------------------------

def build_prediction_lock(*, experiment_id: str, variant: str, seed: int | None,
                          rows: Sequence[dict[str, Any]], checkpoint_sha256: str,
                          source_calibration_sha256: str, calibration_hash: str,
                          inference_config_hash: str, target_feature_package_identity: str,
                          target_package_id: str, threshold: float,
                          unknown_threshold: float | None,
                          scientific_config_hash: str = "",
                          source_matrix_lock_identity: str = "",
                          code_commit: str = "",
                          engineering_smoke: bool = False) -> dict[str, Any]:
    """Freeze one prediction.

    `scientific_config_hash`, `source_matrix_lock_identity` and `code_commit` bind
    the prediction to the frozen source-side decision that produced it, so a lock
    can be traced back to a matrix row and the code that ran it without consulting
    anything outside the file. They are empty only for an engineering smoke, which
    has no matrix row to bind.
    """
    aggregates = aggregate_frames(rows, threshold=threshold, unknown_threshold=unknown_threshold)
    body = {"prediction_lock_schema_version": M10_PREDICTION_LOCK_SCHEMA_VERSION,
            "experiment_id": str(experiment_id), "variant": str(variant), "seed": seed,
            "scientific_config_hash": str(scientific_config_hash),
            "source_matrix_lock_identity": str(source_matrix_lock_identity),
            "code_commit": str(code_commit),
            "checkpoint_sha256": str(checkpoint_sha256),
            "source_calibration_sha256": str(source_calibration_sha256),
            "calibration_hash": str(calibration_hash),
            "inference_config_hash": str(inference_config_hash),
            "target_feature_package_identity": str(target_feature_package_identity),
            "target_package_id": str(target_package_id),
            "prediction_logical_identity": prediction_logical_identity(rows),
            "prediction_schema_version": M10_PREDICTION_SCHEMA_VERSION,
            "row_count": len(rows), "video_count": len(aggregates),
            "aggregation": aggregation_config(threshold, unknown_threshold),
            # An engineering smoke is a real prediction produced for engineering
            # verification, not a scientific result. It is labelled here so a
            # scientific scoring pass refuses it instead of quietly counting it.
            "engineering_smoke": bool(engineering_smoke),
            "scientific_use": not bool(engineering_smoke),
            "target_labels_opened": False, "status": "LOCKED"}
    return {**body, "prediction_lock_identity": stable_identity(body)}


def write_prediction_lock(path: Path, lock: dict[str, Any], *, dry_run: bool = False) -> Path:
    target = Path(path)
    if not dry_run: atomic_json_write(target, lock)
    return target


def validate_prediction_lock(lock: dict[str, Any], rows: Sequence[dict[str, Any]], *,
                             expected_checkpoint_sha256: str | None = None,
                             expected_calibration_sha256: str | None = None,
                             expected_calibration_hash: str | None = None,
                             expected_inference_config_hash: str | None = None,
                             expected_package_identity: str | None = None,
                             expected_scientific_config_hash: str | None = None,
                             expected_source_matrix_lock_identity: str | None = None
                             ) -> dict[str, Any]:
    """Every refusal condition of the target evaluation contract section 5.

    A refusal raises. It is never a warning and never a fallback path.
    """
    if lock.get("prediction_lock_schema_version") != M10_PREDICTION_LOCK_SCHEMA_VERSION:
        raise PredictionLockError(f"prediction lock schema {lock.get('prediction_lock_schema_version')!r} "
                                  f"!= {M10_PREDICTION_LOCK_SCHEMA_VERSION!r}")
    if lock.get("status") != "LOCKED":
        raise PredictionLockError(f"refusing to score an unlocked prediction (status "
                                  f"{lock.get('status')!r})")
    recomputed = stable_identity({key: value for key, value in lock.items()
                                  if key != "prediction_lock_identity"})
    if recomputed != lock.get("prediction_lock_identity"):
        raise PredictionLockError("the prediction lock does not hash to its own identity")
    checks = {
        "checkpoint_sha256": (expected_checkpoint_sha256, lock.get("checkpoint_sha256")),
        "source_calibration_sha256": (expected_calibration_sha256, lock.get("source_calibration_sha256")),
        "calibration_hash": (expected_calibration_hash, lock.get("calibration_hash")),
        "inference_config_hash": (expected_inference_config_hash, lock.get("inference_config_hash")),
        "target_feature_package_identity": (expected_package_identity,
                                            lock.get("target_feature_package_identity")),
        "scientific_config_hash": (expected_scientific_config_hash,
                                   lock.get("scientific_config_hash")),
        "source_matrix_lock_identity": (expected_source_matrix_lock_identity,
                                        lock.get("source_matrix_lock_identity"))}
    mismatched = {name: {"expected": expected, "lock": actual}
                  for name, (expected, actual) in checks.items()
                  if expected is not None and expected != actual}
    if mismatched: raise PredictionLockError(f"prediction/frozen-artifact mismatch: {mismatched}")
    if int(lock.get("row_count", -1)) != len(rows):
        raise PredictionLockError(f"lock row_count {lock.get('row_count')} != {len(rows)} rows on disk")
    videos = len({str(row["video_id"]) for row in rows})
    if int(lock.get("video_count", -1)) != videos:
        raise PredictionLockError(f"lock video_count {lock.get('video_count')} != {videos} videos on disk")
    identity = prediction_logical_identity(rows)
    if identity != lock.get("prediction_logical_identity"):
        raise PredictionLockError("the prediction file does not reproduce its locked logical identity")
    if lock.get("target_labels_opened") is not False:
        raise PredictionLockError("a prediction lock may never record an opened target label")
    return {"passed": True, "prediction_lock_identity": lock["prediction_lock_identity"],
            "row_count": len(rows), "video_count": videos}


def build_lockset(locks: Sequence[dict[str, Any]], *, matrix_identity: str,
                  registry_identity: str, source_matrix_lock_identity: str = "",
                  target_feature_package_identity: str = "",
                  row_statuses: dict[str, str] | None = None,
                  blocked_rows: Sequence[dict[str, Any]] = (),
                  rows_without_prediction: Sequence[dict[str, Any]] = (),
                  code_lineage: dict[str, Any] | None = None) -> dict[str, Any]:
    """`TARGET_PREDICTION_LOCKSET.json` — the gate the label reveal waits behind.

    It binds the whole source-side decision, not just the predictions: the frozen
    `SOURCE_MATRIX_LOCK` identity, the matrix identity, the target FEATURE identity,
    every row's terminal status, the BLOCKED rows with their reasons, and the rows
    that legitimately produce no prediction with the reason they do not. A reader
    can therefore tell a missing prediction from an absent one.
    """
    if not locks: raise M10ContractError("a lockset cannot be empty")
    identifiers = [str(lock["experiment_id"]) for lock in locks]
    duplicates = sorted({name for name in identifiers if identifiers.count(name) > 1})
    if duplicates: raise M10ContractError(f"duplicate experiment ids in the lockset: {duplicates}")
    bound = {str(lock.get("source_matrix_lock_identity") or "") for lock in locks}
    if source_matrix_lock_identity and bound != {str(source_matrix_lock_identity)}:
        raise M10ContractError("every prediction lock must bind the same SOURCE_MATRIX_LOCK identity; "
                               f"found {sorted(bound)}")
    if any(lock.get("engineering_smoke") for lock in locks):
        raise M10ContractError("an engineering-smoke prediction may never enter the lockset")
    body = {"lockset_schema_version": "m10-prediction-lockset-v1",
            "m10_matrix_identity": str(matrix_identity), "registry_identity": str(registry_identity),
            "source_matrix_lock_identity": str(source_matrix_lock_identity),
            "target_feature_package_identity": str(target_feature_package_identity),
            "entries": sorted(({"experiment_id": str(lock["experiment_id"]),
                                "variant": str(lock.get("variant", "")),
                                "seed": lock.get("seed"),
                                "scientific_config_hash": str(lock.get("scientific_config_hash", "")),
                                "checkpoint_sha256": str(lock.get("checkpoint_sha256", "")),
                                "source_calibration_sha256":
                                    str(lock.get("source_calibration_sha256", "")),
                                "inference_config_hash": str(lock.get("inference_config_hash", "")),
                                "prediction_lock_identity": str(lock["prediction_lock_identity"]),
                                "prediction_logical_identity": str(lock["prediction_logical_identity"]),
                                "row_count": int(lock["row_count"]),
                                "video_count": int(lock["video_count"])} for lock in locks),
                              key=lambda entry: entry["experiment_id"]),
            "entry_count": len(locks),
            "row_statuses": dict(sorted((row_statuses or {}).items())),
            "blocked_rows": sorted(({"experiment_id": str(row["experiment_id"]),
                                     "reason": str(row.get("reason") or row.get("blocked_reason") or "")}
                                    for row in blocked_rows),
                                   key=lambda entry: entry["experiment_id"]),
            "rows_without_prediction": sorted(({"experiment_id": str(row["experiment_id"]),
                                                "reason": str(row.get("reason", ""))}
                                               for row in rows_without_prediction),
                                              key=lambda entry: entry["experiment_id"]),
            "frame_rows_total": sum(int(lock["row_count"]) for lock in locks),
            "code_lineage": dict(code_lineage or {}),
            "target_labels_opened": False, "status": "FROZEN"}
    return {**body, "lockset_identity": stable_identity(body)}


# --- the inference driver ----------------------------------------------------

def inference_config_hash(*, variant: str, flags: dict[str, Any], threshold: float,
                          unknown_threshold: float | None, temperature: float | None,
                          package_identity: str, architecture_identity: str) -> str:
    return stable_identity({"schema_version": M10_PREDICTION_SCHEMA_VERSION, "variant": str(variant),
                            "flags": {str(k): v for k, v in sorted(flags.items())},
                            "threshold": float(threshold),
                            "unknown_threshold": (None if unknown_threshold is None
                                                  else float(unknown_threshold)),
                            "temperature": (None if temperature is None else float(temperature)),
                            "target_feature_package_identity": str(package_identity),
                            "architecture_identity": str(architecture_identity)})


def load_checkpoint_for_inference(path: Path, model: Any) -> dict[str, Any]:
    """Open a frozen checkpoint for G7 and verify what inference actually depends on.

    The architecture identity, the SigLIP2 identity and the recipe-text-cache
    identity must match the model being loaded into; `strict=True` then guarantees
    there is no partial restore. The training-side identities (config hash, batch
    contract, dataset contract) are recorded as provenance — G7 is not resuming a
    training run, so requiring them to match a *training* configuration would be a
    check of the wrong thing.
    """
    import torch
    from prism_fas.detector.checkpoint import M9_CHECKPOINT_SCHEMA_VERSION, sha256_file
    target = Path(path)
    if not target.is_file(): raise M10ContractError(f"checkpoint {target.name} does not exist")
    payload = torch.load(target, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != M9_CHECKPOINT_SCHEMA_VERSION:
        raise M10ContractError(f"checkpoint schema {payload.get('schema_version')!r} != "
                               f"{M9_CHECKPOINT_SCHEMA_VERSION!r}")
    stored = dict(payload.get("identity") or {})
    expected = {"architecture_identity": model.architecture_identity(),
                "recipe_text_cache_identity": model.text_cache_identity}
    mismatched = {name: {"checkpoint": stored.get(name), "model": value}
                  for name, value in expected.items() if stored.get(name) != value}
    if mismatched:
        raise M10ContractError(f"refusing to predict: inference identity mismatch {mismatched}")
    model.load_state_dict(payload["model_state"], strict=True)
    from prism_fas.detector.checkpoint import _restore_prototypes
    _restore_prototypes(model.manifold, payload["prototype_state"])
    model.eval()
    return {"checkpoint_sha256": sha256_file(target), "stage": str(payload.get("stage")),
            "global_step": int(payload.get("global_step", 0)),
            "prototypes_initialized": bool((payload.get("prototype_state") or {}).get("initialized", False)),
            "identity": stored, "verified": sorted(expected)}


def target_batches(package_root: Path, loader_config: Any, *, cache_root: Path, limit: int | None = None,
                   batch_size: int = 16, firewall: TargetLabelFirewall | None = None,
                   video_ids: Sequence[str] | None = None,
                   progress: Callable[[dict[str, Any]], None] | None = None
                   ) -> Iterable[TargetInferenceBatch]:
    """Stream target FEATURE batches. No label is read, because none is reachable.

    The region priors come from the same deterministic cache builder M9 uses for
    `source_train`/`source_dev`; the split is the only difference.
    """
    import torch
    from prism_fas.data.loader.config import INFERENCE_SPLIT
    from prism_fas.data.loader.loose_dataset import CanonicalPackageDataset
    from prism_fas.detector.region_cache import load_or_build_region_prior_cache
    root = Path(package_root)
    if firewall is not None:
        firewall.check_read("G7", root)
        firewall.assert_cannot_resolve_labels("G7")
    dataset = CanonicalPackageDataset(root, INFERENCE_SPLIT, loader_config, mode="inference")
    # Keep the DATASET position beside every selected row. Selecting a subset makes
    # the filtered list and the dataset's own indexing diverge, and `dataset[i]`
    # always indexes the dataset — so the two must be carried together rather than
    # assumed equal.
    selected = list(enumerate(dataset.index.rows))
    # `video_ids` selects whole videos by OPAQUE id. It is label-free by
    # construction: the caller derives the set from feature-manifest row counts,
    # which carry no label, family or path. It exists so a smoke can deliberately
    # cover both the 4-frame and the 3-frame video cases.
    if video_ids is not None:
        wanted = {str(value) for value in video_ids}
        selected = [entry for entry in selected if str(entry[1]["source_record_id"]) in wanted]
    if limit is not None: selected = selected[:int(limit)]
    rows = [row for _, row in selected]
    cache, _ = load_or_build_region_prior_cache(Path(cache_root), root, rows,
                                                package_identity=dataset.index.content_identity,
                                                split=INFERENCE_SPLIT)
    # A cache built over a different row set must never be silently reused: the
    # positions would still index, and every prior would belong to another sample.
    if cache.sample_ids != tuple(row["sample_id"] for row in rows):
        raise M10ContractError("the region prior cache does not align with the target rows it was "
                               "asked for; refusing to index priors by position")
    # The frozen package manifest does not carry the real in-video frame index —
    # M3A/M3B keep only the opaque `sample_id`, which HASHES that index without
    # exposing it. `frame_id` is therefore the deterministic ordinal of the frame
    # within its video in frozen `sample_id` order. It is an ordering key for the
    # aggregation join and is never read as a timestamp or a video position.
    ordinal: dict[str, int] = {}
    for row in sorted(rows, key=lambda item: (str(item["source_record_id"]), str(item["sample_id"]))):
        video = str(row["source_record_id"])
        ordinal[str(row["sample_id"])] = ordinal.get(f"__count__{video}", 0)
        ordinal[f"__count__{video}"] = ordinal[str(row["sample_id"])] + 1
    positions = list(range(len(selected)))
    for start in range(0, len(positions), int(batch_size)):
        window = positions[start:start + int(batch_size)]
        samples = [dataset[selected[position][0]] for position in window]
        # The prior is indexed by the SELECTED position and the image by the DATASET
        # position. Asserting the sample ids agree is what makes a future divergence
        # a loud failure instead of every prior silently belonging to another sample.
        for position, sample in zip(window, samples):
            if sample.sample_id != rows[position]["sample_id"]:
                raise M10ContractError(
                    f"target row alignment broke: prior {rows[position]['sample_id']!r} does not "
                    f"belong to sample {sample.sample_id!r}")
        yield TargetInferenceBatch(
            image=torch.from_numpy(np.stack([np.asarray(item.image, dtype=np.float32) for item in samples])),
            region_priors=torch.from_numpy(np.stack([cache.prior(position) for position in window])),
            visibility=torch.from_numpy(np.stack([cache.visible(position) for position in window])),
            is_synthetic=torch.zeros(len(window), dtype=torch.bool),
            sample_ids=tuple(item.sample_id for item in samples),
            video_ids=tuple(item.source_record_id for item in samples),
            frame_ids=tuple(int(ordinal[item.sample_id]) for item in samples))
        if progress: progress({"stage": "G7", "done": start + len(window), "total": len(positions)})


def predict_target(model: Any, batches: Iterable[TargetInferenceBatch], *, capabilities: VariantCapabilities,
                   threshold: float, unknown_threshold: float | None, temperature: float | None,
                   checkpoint_hash: str, calibration_hash: str, inference_config_hash: str,
                   variant: str, device: str = "cpu", top_region_count: int = 2) -> list[dict[str, Any]]:
    """Run the frozen detector over target features and emit Table 57 rows.

    Nothing here can consult a label: the batch has no label field, and the only
    thresholds used are the frozen source-side ones passed in.
    """
    import torch
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for batch in batches:
            moved = TargetInferenceBatch(
                image=batch.image.to(device), region_priors=batch.region_priors.to(device),
                visibility=batch.visibility.to(device), is_synthetic=batch.is_synthetic.to(device),
                sample_ids=batch.sample_ids, video_ids=batch.video_ids, frame_ids=batch.frame_ids)
            output = model(moved)
            p_global = output.p_global.detach().float().cpu().numpy()
            if temperature is not None:
                logit = output.global_logit.detach().float().cpu().numpy().reshape(-1)
                p_global = 1.0 / (1.0 + np.exp(-logit / float(temperature)))
            # A term the variant does not produce is absent here too, so an absent
            # component can never be read out of a zero-filled array.
            s_region = (output.s_region.detach().float().cpu().numpy()
                        if output.s_region is not None else None)
            p_prompt = (output.p_prompt_spoof.detach().float().cpu().numpy()
                        if output.p_prompt_spoof is not None else None)
            distances = (output.aux["normalized_distances"].detach().float().cpu().numpy()
                         if "normalized_distances" in output.aux else None)
            valid = (output.region_valid.detach().cpu().numpy()
                     if output.region_valid is not None else None)
            # PromptHead applicability is `is_synthetic AND attacked region AND
            # visible`. A target sample is never synthetic and carries no attack
            # mask, so no region is applicable and the head returns an EXACT
            # structural zero. Writing that 0.0 as a computed value would present a
            # constant as a measurement, which is the same mistake as writing 0.0
            # for a branch the variant does not have.
            applicable = (output.aux["prompt_applicable"].detach().float().cpu().numpy()
                          if "prompt_applicable" in output.aux else None)
            for position, sample_id in enumerate(moved.sample_ids):
                top: list[int] = []
                region_distances: list[float] = []
                region_value: float | None = None
                if capabilities.has_region and s_region is not None:
                    region_value = float(s_region[position])
                    # Per-region ids and distances exist only for the nine-site
                    # regional manifold; a single global center reports the fused
                    # score without pretending to a regional decomposition.
                    if capabilities.has_region_detail and distances is not None and valid is not None:
                        ranked = np.argsort(-np.where(valid[position], distances[position], -np.inf))
                        top = [int(index) for index in ranked[:int(top_region_count)]
                               if bool(valid[position][int(index)])]
                        region_distances = [float(value) for value in distances[position]]
                prompt_value = (float(p_prompt[position])
                                if capabilities.has_prompt and p_prompt is not None
                                and applicable is not None and float(applicable[position].sum()) > 0.0
                                else None)
                rows.append(build_prediction_row(
                    sample_id=sample_id, video_id=moved.video_ids[position],
                    frame_id=moved.frame_ids[position], p_global=float(p_global[position]),
                    s_region=region_value, p_prompt=prompt_value,
                    threshold=threshold, unknown_threshold=unknown_threshold,
                    top_region_ids=top, region_distances=region_distances,
                    checkpoint_hash=checkpoint_hash, calibration_hash=calibration_hash,
                    inference_config_hash=inference_config_hash, variant=variant))
    validate_predictions(rows)
    return rows
