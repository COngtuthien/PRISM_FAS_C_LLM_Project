"""G8 — the isolated target scorer.

G8 is the ONLY stage permitted to resolve an evaluation-only label path, and it is
permitted to do nothing else. Spec Table 54: "evaluator: không update model,
optimizer hoặc calibration từ target labels".

Structural guarantees, not promises:

*   **No training capability.** This module imports numpy and pyarrow. It does not
    import torch, does not import `prism_fas.detector.trainer`, and holds no
    optimizer, scheduler, scaler or model. `static_import_audit` proves that from
    the module's own AST rather than from a comment.
*   **No model-state write.** Every write goes through
    `TargetLabelFirewall.check_write("G8", path)`, which refuses `.pt`, `.pth`,
    `.ckpt`, `.safetensors`, `optimizer`, `calibration/` and `checkpoints/`
    whatever root they are aimed at.
*   **No unlocked scoring.** `validate_prediction_lock` runs before a single label
    is read, and a refusal raises.

The label join is by the opaque `siw_<16hex>` video id and nothing else. A label is
never joined by filename, path, attack name or directory structure.
"""
from __future__ import annotations
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from prism_fas.utils.core import atomic_json_write
from .contracts import (M10_SCORING_SCHEMA_VERSION, M10ContractError, ScoringRefusal,
                        is_not_applicable, not_applicable, stable_identity)
from .firewall import TargetLabelFirewall
from .metrics import (attack_wise_apcer, calibration_metrics, core_metrics, open_set_metrics,
                      population, region_wise_summary, rejection_metrics, risk_coverage,
                      threshold_table)
from .target_prediction import read_predictions, validate_prediction_lock
from .video_aggregation import aggregate_frames

LABEL_COLUMNS = ("video_id", "label")
OPTIONAL_LABEL_COLUMNS = ("attack_family",)
# Names this module must never import. Asserted from the AST, not from a text scan.
FORBIDDEN_IMPORTS = ("torch", "prism_fas.detector.trainer", "prism_fas.train.trainer",
                     "prism_fas.train.b00_pipeline", "torch.optim")


def static_import_audit(module_path: Path) -> dict[str, Any]:
    """Parse a module and list what it imports, structurally.

    This is an AST walk of the import graph, not a search for a word in a file —
    the M8/M9 lesson applies here too, and a docstring that names `torch` must not
    be able to fail this check.
    """
    tree = ast.parse(Path(module_path).read_text(encoding="utf-8"))

    def names(node: ast.AST) -> set[str]:
        if isinstance(node, ast.Import): return {alias.name for alias in node.names}
        if isinstance(node, ast.ImportFrom):
            base = node.module or ""
            return {base} | {f"{base}.{alias.name}" for alias in node.names if base}
        return set()

    # Only MODULE-LEVEL imports are pulled in when the module is imported. A
    # function-local `import torch` inside a G7 helper is not a G8 capability, and
    # it is reported separately rather than counted as one.
    module_level: set[str] = set()
    for node in tree.body: module_level |= names(node)
    deferred: set[str] = set()
    for node in ast.walk(tree):
        deferred |= names(node)
    deferred -= module_level
    violations = sorted({name for name in module_level for forbidden in FORBIDDEN_IMPORTS
                         if name == forbidden or name.startswith(forbidden + ".")})
    return {"module": Path(module_path).name, "module_level_imports": sorted(module_level),
            "deferred_imports": sorted(deferred), "forbidden_imports": violations,
            "passed": not violations}


def import_closure_audit() -> dict[str, Any]:
    """Audit this module AND the first-party modules it imports at module level.

    Auditing `scoring.py` alone would be satisfied by a module that imports a
    module that imports torch. The evaluation package's own modules keep every
    torch import function-local, so importing G8 pulls in no training runtime, and
    this walk is what proves it.
    """
    here = Path(__file__).parent
    modules = [Path(__file__)] + [here / f"{name}.py" for name in
                                  ("contracts", "firewall", "metrics", "target_prediction",
                                   "video_aggregation")]
    audits = [static_import_audit(path) for path in modules if path.is_file()]
    failed = [audit for audit in audits if not audit["passed"]]
    return {"modules_audited": [audit["module"] for audit in audits], "passed": not failed,
            "failures": failed}


def assert_no_training_capability() -> dict[str, Any]:
    """G8 cannot train, because the capability is absent from this module."""
    audit = static_import_audit(Path(__file__))
    closure = import_closure_audit()
    if not closure["passed"]:
        raise ScoringRefusal(f"importing G8 would pull in a training runtime: {closure['failures']}")
    if not audit["passed"]:
        raise ScoringRefusal(f"G8 must have no training capability; found {audit['forbidden_imports']}")
    globals_here = set(globals())
    forbidden_symbols = sorted(name for name in globals_here
                               if name.lower() in {"optimizer", "scheduler", "scaler", "trainer",
                                                   "model", "save_checkpoint"})
    if forbidden_symbols:
        raise ScoringRefusal(f"G8 holds training symbols: {forbidden_symbols}")
    return {**audit, "import_closure": closure, "holds_optimizer": False, "holds_model": False,
            "can_save_checkpoint": False}


# --- evaluation-only labels --------------------------------------------------

@dataclass(frozen=True)
class EvaluationLabels:
    """Video-level ground truth. Reachable only from G8."""
    by_video: dict[str, int]
    families: dict[str, str]
    source: str

    def validate(self) -> "EvaluationLabels":
        if not self.by_video: raise M10ContractError("the evaluation label artifact is empty")
        bad = sorted({value for value in self.by_video.values()} - {0, 1})
        if bad: raise M10ContractError(f"label values outside live=0/spoof=1: {bad}")
        return self

    def counts(self) -> dict[str, int]:
        values = list(self.by_video.values())
        return {"videos": len(values), "live": values.count(0), "spoof": values.count(1),
                "families": len(set(self.families.values()))}


def load_evaluation_labels(path: Path, *, firewall: TargetLabelFirewall,
                           stage: str = "G8") -> EvaluationLabels:
    """Open the evaluator-only label artifact. Any stage but G8 is refused."""
    if stage != "G8":
        raise ScoringRefusal(f"{stage} may not open the evaluation-only label artifact")
    target = firewall.check_read(stage, Path(path))
    if not Path(target).is_file():
        raise M10ContractError(f"the evaluation label artifact does not exist: {Path(path).name}")
    import pyarrow.parquet as pq
    table = pq.read_table(target)
    missing = [name for name in LABEL_COLUMNS if name not in table.column_names]
    if missing: raise M10ContractError(f"the label artifact is missing columns {missing}")
    rows = table.to_pylist()
    mapping = {"live": 0, "spoof": 1}
    by_video, families = {}, {}
    for row in rows:
        raw = row["label"]
        value = mapping[str(raw)] if isinstance(raw, str) else int(raw)
        by_video[str(row["video_id"])] = value
        if row.get("attack_family"): families[str(row["video_id"])] = str(row["attack_family"])
    return EvaluationLabels(by_video=by_video, families=families, source=Path(path).name).validate()


# --- the scorer --------------------------------------------------------------

def _join(video_ids: Sequence[str], labels: EvaluationLabels) -> tuple[list[int], list[str]]:
    missing = sorted({str(value) for value in video_ids} - set(labels.by_video))
    if missing:
        raise ScoringRefusal(f"{len(missing)} predicted videos have no label; refusing to score a "
                             f"partially joined population (first: {missing[:3]})")
    return ([int(labels.by_video[str(value)]) for value in video_ids],
            [labels.families.get(str(value), "live") for value in video_ids])


def score(*, predictions: Sequence[dict[str, Any]], lock: dict[str, Any], labels: EvaluationLabels,
          threshold: float, unknown_threshold: float | None = None,
          region_order: Sequence[str] = (), expected: dict[str, str] | None = None,
          allow_engineering_smoke: bool = False) -> dict[str, Any]:
    """Frame and video metrics for ONE experiment. Nothing is written here.

    The lock is validated BEFORE a single label is touched, so an unlocked or
    mismatched prediction can never reach the join.
    """
    assert_no_training_capability()
    expected = expected or {}
    validate_prediction_lock(
        lock, predictions,
        expected_checkpoint_sha256=expected.get("checkpoint_sha256"),
        expected_calibration_sha256=expected.get("source_calibration_sha256"),
        expected_calibration_hash=expected.get("calibration_hash"),
        expected_inference_config_hash=expected.get("inference_config_hash"),
        expected_package_identity=expected.get("target_feature_package_identity"))
    if lock.get("engineering_smoke") and not allow_engineering_smoke:
        raise ScoringRefusal("this prediction was produced as an engineering smoke "
                             "(scientific_use: false); refusing to score it as a scientific result")

    frame_videos = [str(row["video_id"]) for row in predictions]
    frame_labels, frame_families = _join(frame_videos, labels)
    frame_scores = [float(row["s_final"]) for row in predictions]
    frame_confidence = [float(row["confidence"]) for row in predictions]

    aggregates = aggregate_frames(predictions, threshold=threshold, unknown_threshold=unknown_threshold)
    video_ids = [row["video_id"] for row in aggregates]
    video_labels, video_families = _join(video_ids, labels)
    video_scores = [float(row["video_score"]) for row in aggregates]
    video_confidence = [float(row["video_confidence"]) for row in aggregates]

    frame_core = core_metrics(frame_scores, frame_labels, threshold=threshold)
    video_core = core_metrics(video_scores, video_labels, threshold=threshold)
    scorable = not is_not_applicable(video_core["acer"])
    body = {
        "scoring_schema_version": M10_SCORING_SCHEMA_VERSION,
        "experiment_id": str(lock["experiment_id"]), "variant": str(lock["variant"]),
        "seed": lock.get("seed"), "threshold": float(threshold),
        "unknown_threshold": (None if unknown_threshold is None else float(unknown_threshold)),
        "prediction_lock_identity": str(lock["prediction_lock_identity"]),
        "label_source": labels.source, "label_counts": labels.counts(),
        "frame": {**frame_core, "population": population(frame_scores, frame_labels),
                  "calibration": calibration_metrics(frame_scores, frame_labels)},
        "video": {**video_core, "population": population(video_scores, video_labels),
                  "calibration": calibration_metrics(video_scores, video_labels),
                  "risk_coverage": risk_coverage(video_scores, video_labels, video_confidence,
                                                 threshold=threshold),
                  **rejection_metrics([row["decision"] for row in aggregates])},
        "attack_wise": (attack_wise_apcer(video_scores, video_labels, video_families,
                                          threshold=threshold) if labels.families
                        else not_applicable("the label artifact carries no attack_family column; "
                                            "attack-wise APCER is optional metadata")),
        "region_wise": (region_wise_summary(predictions, region_order=region_order) if region_order
                        else not_applicable("no region order was supplied to the scorer")),
        "open_set": open_set_metrics(),
        "threshold_table": (threshold_table(video_scores, video_labels,
                                            thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                            if scorable else not_applicable(
                                "a threshold table needs both bona-fide and attack presentations",
                                population=population(video_scores, video_labels))),
        "video_scores": [{"video_id": row["video_id"], "video_score": row["video_score"],
                          "video_confidence": row["video_confidence"], "decision": row["decision"],
                          "label": label} for row, label in zip(aggregates, video_labels)],
        "target_labels_opened": True,
        "wrote_checkpoint": False, "wrote_optimizer_state": False, "modified_calibration": False,
        "modified_predictions": False}
    return {**body, "scoring_identity": stable_identity(
        {key: value for key, value in body.items() if key != "video_scores"})}


def score_from_disk(*, predictions_path: Path, lock: dict[str, Any], labels: EvaluationLabels,
                    threshold: float, **kwargs: Any) -> dict[str, Any]:
    return score(predictions=read_predictions(Path(predictions_path)), lock=lock, labels=labels,
                 threshold=threshold, **kwargs)


def write_scoring_report(path: Path, payload: dict[str, Any], *, firewall: TargetLabelFirewall,
                         dry_run: bool = False) -> Path:
    """Every G8 write passes the firewall, which refuses model state by pattern."""
    target = firewall.check_write("G8", Path(path))
    if not dry_run: atomic_json_write(Path(target), payload)
    return Path(target)


def target_label_reveal(*, lockset: dict[str, Any], authorized_by: str,
                        reason: str = "the frozen TARGET_PREDICTION_LOCKSET is complete") -> dict[str, Any]:
    """The one-way transition, recorded exactly once.

    Refuses unless the lockset is FROZEN, so labels cannot be opened before every
    prediction in the matrix is locked.
    """
    if lockset.get("status") != "FROZEN":
        raise ScoringRefusal("target labels may be opened only after the prediction lockset is FROZEN")
    if not authorized_by: raise M10ContractError("the reveal must record who authorized it")
    body = {"reveal_schema_version": "m10-target-label-reveal-v1",
            "lockset_identity": str(lockset["lockset_identity"]),
            "entry_count": int(lockset["entry_count"]), "authorized_by": str(authorized_by),
            "reason": str(reason), "target_labels_revealed": True, "one_way": True,
            "may_be_reset": False}
    return {**body, "reveal_identity": stable_identity(body)}


def isolation_report() -> dict[str, Any]:
    """The evidence a reviewer needs, asserted field by field."""
    audit = assert_no_training_capability()
    return {"scoring_schema_version": M10_SCORING_SCHEMA_VERSION,
            "g8_imports_torch": False, "g8_imports_trainer": False,
            "g8_can_save_checkpoint": False, "g8_can_modify_calibration": False,
            "g8_can_modify_predictions": False, "g8_reads_labels": True,
            "static_import_audit": audit,
            "enforcement": "ast_import_audit_plus_firewall_write_check"}
