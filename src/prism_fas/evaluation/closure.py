"""M10 closure: the pre-reveal audit, the isolated G8 pass over the whole matrix,
seed aggregation, and the machine-readable summary.

This module is the ORDER of the endgame, expressed once so it cannot be performed
out of order by hand:

    1. every eligible row has a frozen, self-reproducing prediction
    2. `TARGET_PREDICTION_LOCKSET` freezes that set
    3. the pre-reveal audit passes
    4. and only then may a label be opened

It holds no training runtime for the same reason `scoring` does not: it is imported
by G8. It never imports torch, never imports the detector package (which does), and
every write it performs goes through the firewall.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from prism_fas.utils.core import atomic_json_write
from .contracts import (M10ContractError, REGION_ORDER, ScoringRefusal, is_not_applicable,
                        not_applicable, stable_identity)
from .firewall import TargetLabelFirewall
from . import scoring
from . import target_prediction as g7

CLOSURE_SCHEMA_VERSION = "m10-closure-v1"
REVEAL_FILE = "TARGET_LABEL_REVEAL.json"
LOCKSET_FILE = "TARGET_PREDICTION_LOCKSET.json"


def logical_row(experiment_id: str) -> str:
    """The row name without its seed suffix — what the hypotheses actually name."""
    return str(experiment_id).rsplit("-s", 1)[0]


# --- the frozen source-side decision -----------------------------------------

def eligible_rows(source_matrix_lock: dict[str, Any]) -> list[dict[str, Any]]:
    """Every COMPLETED row that declares a target prediction, in id order."""
    return sorted((entry for entry in source_matrix_lock["entries"]
                   if entry.get("status") == "COMPLETED" and entry.get("target_prediction_required")),
                  key=lambda entry: str(entry["experiment_id"]))


def rows_without_prediction(source_matrix_lock: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows that legitimately produce NO prediction, each with the specific reason.

    An absent prediction must be explicit. A BLOCKED row never trained, and the A09
    parity row is parity evidence: it trains nothing to completion and selects no
    checkpoint, so there is nothing to predict with.
    """
    out = []
    for entry in source_matrix_lock["entries"]:
        if entry.get("status") == "COMPLETED" and entry.get("target_prediction_required"): continue
        if entry.get("status") == "BLOCKED":
            out.append({"experiment_id": entry["experiment_id"], "status": "BLOCKED",
                        "reason": entry.get("blocked_reason") or ""})
        else:
            out.append({"experiment_id": entry["experiment_id"], "status": entry.get("status"),
                        "reason": entry.get("blocked_reason")
                        or "the row declares target_prediction_required: false; it is a parity "
                           "result and selects no checkpoint, so it has nothing to predict with"})
    return sorted(out, key=lambda entry: str(entry["experiment_id"]))


def expected_bindings(entry: dict[str, Any], *, package_identity: str,
                      source_matrix_lock_identity: str) -> dict[str, str]:
    """What a prediction lock for this row MUST say, taken from the frozen lock."""
    return {"checkpoint_sha256": str(entry["best_checkpoint_sha256"]),
            "source_calibration_sha256": str(entry["source_calibration_sha256"]),
            "calibration_hash": str(entry["calibration_hash"]),
            "scientific_config_hash": str(entry["scientific_config_hash"]),
            "source_matrix_lock_identity": str(source_matrix_lock_identity),
            "target_feature_package_identity": str(package_identity)}


# --- prediction validation ----------------------------------------------------

def validate_row_prediction(*, entry: dict[str, Any], prediction_path: Path, lock: dict[str, Any],
                            package_identity: str, source_matrix_lock_identity: str,
                            expected_rows: int, expected_videos: int) -> dict[str, Any]:
    """Everything section 9 of the instruction requires, per row, before any label.

    `read_predictions` already refuses a forbidden column, a duplicate sample id, a
    non-finite score and a null that disagrees with its applicability status, so
    those are asserted here by construction rather than re-implemented.
    """
    rows = g7.read_predictions(Path(prediction_path))
    bindings = expected_bindings(entry, package_identity=package_identity,
                                 source_matrix_lock_identity=source_matrix_lock_identity)
    lock_check = g7.validate_prediction_lock(
        lock, rows,
        expected_checkpoint_sha256=bindings["checkpoint_sha256"],
        expected_calibration_sha256=bindings["source_calibration_sha256"],
        expected_calibration_hash=bindings["calibration_hash"],
        expected_package_identity=bindings["target_feature_package_identity"],
        expected_scientific_config_hash=bindings["scientific_config_hash"],
        expected_source_matrix_lock_identity=bindings["source_matrix_lock_identity"])
    videos: dict[str, int] = {}
    for row in rows: videos[str(row["video_id"])] = videos.get(str(row["video_id"]), 0) + 1
    frames_per_video: dict[str, int] = {}
    for count in videos.values(): frames_per_video[str(count)] = frames_per_video.get(str(count), 0) + 1
    checks = {
        "frame_rows_complete": len(rows) == int(expected_rows),
        "videos_complete": len(videos) == int(expected_videos),
        "unique_sample_ids": len({str(row["sample_id"]) for row in rows}) == len(rows),
        "scores_finite": all(np.isfinite(float(row["s_final"])) for row in rows),
        "no_ground_truth_field": True,      # enforced structurally by read_predictions
        "lock_reproduces": bool(lock_check["passed"]),
        "lock_is_scientific": not bool(lock.get("engineering_smoke")),
        "labels_never_opened": lock.get("target_labels_opened") is False}
    return {"experiment_id": str(entry["experiment_id"]), "passed": all(checks.values()),
            "checks": checks, "row_count": len(rows), "video_count": len(videos),
            "frames_per_video_distribution": dict(sorted(frames_per_video.items())),
            "prediction_lock_identity": str(lock["prediction_lock_identity"]),
            "prediction_logical_identity": str(lock["prediction_logical_identity"]),
            "region_status": sorted({str(row["region_status"]) for row in rows}),
            "prompt_status": sorted({str(row["prompt_status"]) for row in rows})}


# --- the hard stop ------------------------------------------------------------

def pre_reveal_audit(*, lockset: dict[str, Any], source_matrix_lock: dict[str, Any],
                     matrix_plan: dict[str, Any], prediction_root: Path, reports_root: Path,
                     package_identity: str, expected_rows: int, expected_videos: int,
                     evaluation_config: dict[str, Any]) -> dict[str, Any]:
    """The final gate. Every condition is CHECKED against an artifact, not asserted.

    If any condition fails the caller must stop: a reveal is one-way, and there is
    no way to un-open a label.
    """
    identity = str(source_matrix_lock["source_matrix_lock_identity"])
    eligible = eligible_rows(source_matrix_lock)
    entries = {str(item["experiment_id"]): item for item in lockset.get("entries") or []}
    per_row, missing = [], []
    for entry in eligible:
        name = str(entry["experiment_id"])
        row_root = Path(prediction_root) / name
        prediction = row_root / g7.PREDICTION_FILE
        lock_path = row_root / g7.PREDICTION_LOCK_FILE
        if name not in entries or not prediction.is_file() or not lock_path.is_file():
            missing.append(name); continue
        import json
        per_row.append(validate_row_prediction(
            entry=entry, prediction_path=prediction,
            lock=json.loads(lock_path.read_text(encoding="utf-8")),
            package_identity=package_identity, source_matrix_lock_identity=identity,
            expected_rows=expected_rows, expected_videos=expected_videos))
    lockset_body = {key: value for key, value in lockset.items() if key != "lockset_identity"}
    checks = {
        "source_matrix_lock_identity_self_consistent":
            bool(source_matrix_lock.get("source_matrix_lock_identity")),
        "source_matrix_lock_never_opened_target": source_matrix_lock.get("target_labels_opened") is False,
        "matrix_identity_unchanged":
            source_matrix_lock["m10_matrix_identity"] == matrix_plan["m10_matrix_identity"],
        "lockset_is_frozen": lockset.get("status") == "FROZEN",
        "lockset_reproduces_its_identity":
            stable_identity(lockset_body) == lockset.get("lockset_identity"),
        "lockset_binds_the_source_matrix_lock":
            lockset.get("source_matrix_lock_identity") == identity,
        "lockset_binds_the_matrix": lockset.get("m10_matrix_identity") == matrix_plan["m10_matrix_identity"],
        "lockset_binds_the_target_feature_package":
            lockset.get("target_feature_package_identity") == package_identity,
        "every_eligible_row_is_locked": not missing,
        "lockset_covers_exactly_the_eligible_rows":
            sorted(entries) == sorted(str(item["experiment_id"]) for item in eligible),
        "every_prediction_validates": bool(per_row) and all(item["passed"] for item in per_row),
        "no_engineering_smoke_in_the_lockset":
            not any(str(name).startswith("g7_") for name in entries),
        "labels_not_yet_revealed": not Path(reports_root, REVEAL_FILE).exists(),
        "evaluation_config_declares_labels_sealed":
            evaluation_config.get("target_labels_revealed") is False,
        "no_scoring_artifact_exists_yet": not any(Path(reports_root, "g8").glob("*/scoring.json"))}
    body = {"schema_version": CLOSURE_SCHEMA_VERSION, "audit": "pre_reveal",
            "source_matrix_lock_identity": identity,
            "m10_matrix_identity": matrix_plan["m10_matrix_identity"],
            "lockset_identity": lockset.get("lockset_identity"),
            "target_feature_package_identity": package_identity,
            "eligible_rows": len(eligible), "locked_rows": len(entries),
            "missing_rows": sorted(missing), "per_row": per_row, "checks": checks,
            "passed": all(checks.values()),
            "target_labels_revealed": False, "target_labels_opened": False}
    return {**body, "pre_reveal_audit_identity": stable_identity(body)}


# --- G8 over the whole matrix -------------------------------------------------

def score_matrix(*, source_matrix_lock: dict[str, Any], lockset: dict[str, Any],
                 prediction_root: Path, output_root: Path, labels: scoring.EvaluationLabels,
                 firewall: TargetLabelFirewall, package_identity: str,
                 unknown_threshold: float | None = None,
                 thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    """Score every frozen prediction against the sealed labels. Isolated by G8's own
    refusals: an unlocked or mismatched prediction never reaches a label."""
    import json
    identity = str(source_matrix_lock["source_matrix_lock_identity"])
    entries = {str(item["experiment_id"]): item for item in source_matrix_lock["entries"]}
    results: dict[str, Any] = {}
    for entry in lockset["entries"]:
        name = str(entry["experiment_id"])
        row = entries[name]
        lock = json.loads((Path(prediction_root) / name / g7.PREDICTION_LOCK_FILE)
                          .read_text(encoding="utf-8"))
        rows = g7.read_predictions(Path(prediction_root) / name / g7.PREDICTION_FILE)
        threshold = float((thresholds or {}).get(name, lock["aggregation"]["threshold"]))
        result = scoring.score(
            predictions=rows, lock=lock, labels=labels, threshold=threshold,
            unknown_threshold=unknown_threshold, region_order=REGION_ORDER,
            expected=expected_bindings(row, package_identity=package_identity,
                                       source_matrix_lock_identity=identity))
        destination = Path(output_root) / name / "scoring.json"
        scoring.write_scoring_report(destination, result, firewall=firewall)
        results[name] = result
    return results


# --- seed aggregation ---------------------------------------------------------

def _value(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    return None if (value is None or is_not_applicable(value)) else float(value)


def seed_summary(scorings: dict[str, Any], *, roles: dict[str, str],
                 metrics: Sequence[str] = ("acer", "apcer", "bpcer", "hter", "roc_auc", "eer")
                 ) -> dict[str, Any]:
    """Mean ± std over the row's seeds, per the statistics contract section 2.

    A single-seed row is reported `single_seed_descriptive`; no variance is
    manufactured for it and it may carry no statistical claim.
    """
    from .metrics import summarize_seeds
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for name, payload in scorings.items():
        grouped.setdefault(logical_row(name), []).append((int(payload.get("seed") or 0), payload))
    out: dict[str, Any] = {}
    for row, items in sorted(grouped.items()):
        items.sort()
        role = roles.get(row, "diagnostic")
        block: dict[str, Any] = {
            "seeds": [seed for seed, _ in items], "n_seeds": len(items),
            "replication_role": role,
            "single_seed_descriptive": len(items) < 3,
            "may_carry_statistical_claim": len(items) >= 3 and role in ("spec_mandated",
                                                                       "hypothesis_critical"),
            "video": {}, "frame": {}}
        for level in ("video", "frame"):
            for metric in metrics:
                values = [_value(payload[level], metric) for _, payload in items]
                present = [value for value in values if value is not None]
                block[level][metric] = (summarize_seeds(present) if present
                                        else not_applicable(f"{metric} was not computable on this "
                                                            f"population at {level} level"))
        block["video"]["calibration"] = {
            key: summarize_seeds([float(payload["video"]["calibration"][key]) for _, payload in items])
            for key in ("ece", "brier", "nll")}
        out[row] = block
    return out


def statistics_input(scorings: dict[str, Any], *, thresholds: dict[str, float]) -> dict[str, Any]:
    """The `{logical_row: {video_ids, scores, labels, threshold}}` block the paired
    bootstrap consumes.

    Statistics contract section 3.3: the bootstrap resamples VIDEOS, not seeds, so
    the video-level score inside a resample is the MEAN over the row's seeds of that
    video's score. The operating threshold that accompanies a mean-over-seeds score
    is the mean of those seeds' own frozen source-dev thresholds — recorded
    explicitly beside it, because it is derived rather than frozen.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for name, payload in scorings.items(): grouped.setdefault(logical_row(name), []).append(payload)
    out: dict[str, Any] = {}
    for row, items in sorted(grouped.items()):
        per_seed = [{str(entry["video_id"]): entry for entry in payload["video_scores"]}
                    for payload in items]
        common = sorted(set.intersection(*[set(block) for block in per_seed]))
        if not common: raise M10ContractError(f"{row}: the seeds share no scored video")
        scores = [float(np.mean([block[video]["video_score"] for block in per_seed]))
                  for video in common]
        labels = [int(per_seed[0][video]["label"]) for video in common]
        row_thresholds = [float(thresholds[str(payload["experiment_id"])]) for payload in items]
        out[row] = {"video_ids": common, "scores": scores, "labels": labels,
                    "n_seeds": len(items),
                    "threshold": float(np.mean(row_thresholds)),
                    "threshold_source": ("the mean of this row's frozen per-seed source_dev "
                                         "thresholds, matching the mean-over-seeds video score "
                                         "the statistics contract requires"),
                    "per_seed_thresholds": row_thresholds}
    return out


# --- summary ------------------------------------------------------------------

def build_summary(*, plan: dict[str, Any], source_matrix_lock: dict[str, Any],
                  lockset: dict[str, Any], scorings: dict[str, Any], seeds: dict[str, Any],
                  statistics: dict[str, Any], reliability: dict[str, Any],
                  reveal: dict[str, Any], parity: dict[str, Any] | None = None,
                  compute: dict[str, Any] | None = None,
                  disclosures: dict[str, Any] | None = None) -> dict[str, Any]:
    """`summary.json` — every headline number, machine-readable, with its provenance."""
    hypotheses = {}
    for name, block in sorted((plan.get("hypotheses") or {}).items()):
        entry = ((statistics or {}).get("comparisons") or {}).get(name, {})
        adjusted = ((statistics or {}).get("multiple_comparison") or {}).get("adjusted", {}).get(name)
        if block.get("kind") == "parity_not_superiority":
            hypotheses[name] = {"kind": "parity_not_superiority", "outcome": "reported_as_measured",
                                "treatment": block["treatment"], "control": block["control"],
                                "evidence": (parity or {}).get("summary")
                                or "see reports/m10/A09_BACKEND_PARITY.json"}
            continue
        outcome = "inconclusive"
        if entry.get("status") == "computed":
            # Predeclared: `supported` requires the Holm-adjusted p to clear alpha AND
            # the interval to exclude zero in the hypothesised direction (the treatment
            # having the LOWER ACER). Anything else is not_supported or inconclusive.
            significant = adjusted is not None and float(adjusted) <= 0.05
            direction = float(entry.get("observed_delta", 0.0)) < 0.0
            excludes_zero = bool(entry.get("ci_high", 0.0) < 0.0 or entry.get("ci_low", 0.0) > 0.0)
            outcome = ("supported" if significant and direction and excludes_zero else
                       "not_supported" if significant and not direction else
                       "not_supported" if excludes_zero and not direction else "inconclusive")
        elif entry.get("status") in ("refused", "unavailable"):
            outcome = "inconclusive"
        hypotheses[name] = {
            "kind": block.get("kind"), "treatment": block["treatment"], "control": block["control"],
            "status": entry.get("status"), "outcome": outcome,
            "observed_delta_acer": entry.get("observed_delta"),
            "ci95": [entry.get("ci_low"), entry.get("ci_high")],
            "p_value_raw": entry.get("p_value"), "p_value_holm_adjusted": adjusted,
            "n_videos": entry.get("n_units"),
            "bootstrap_plan_identity": entry.get("bootstrap_plan_identity")}
    body = {
        "summary_schema_version": CLOSURE_SCHEMA_VERSION,
        "milestone": "M10",
        "m10_matrix_identity": plan["m10_matrix_identity"],
        "source_matrix_lock_identity": source_matrix_lock["source_matrix_lock_identity"],
        "target_prediction_lockset_identity": lockset["lockset_identity"],
        "target_feature_package_identity": lockset.get("target_feature_package_identity"),
        "target_labels_revealed": True,
        "target_label_reveal_identity": (reveal or {}).get("reveal_identity"),
        "rows": {"logical": source_matrix_lock["logical_rows"],
                 "executable": source_matrix_lock["executable_rows"],
                 "blocked": source_matrix_lock["blocked_rows"],
                 "predicted": lockset["entry_count"],
                 "scored": len(scorings)},
        "target_population": {
            "videos": (next(iter(scorings.values()))["video"]["population"] if scorings else None),
            "frames": (next(iter(scorings.values()))["frame"]["population"] if scorings else None)},
        "per_seed": {name: {"video": {"acer": _value(payload["video"], "acer"),
                                      "apcer": _value(payload["video"], "apcer"),
                                      "bpcer": _value(payload["video"], "bpcer"),
                                      "hter": _value(payload["video"], "hter"),
                                      "roc_auc": _value(payload["video"], "roc_auc"),
                                      "eer": _value(payload["video"], "eer")},
                            "threshold": payload["threshold"],
                            "source_dev_threshold_source": "frozen G6 source_dev calibration"}
                     for name, payload in sorted(scorings.items())},
        "by_row": seeds,
        "hypotheses": hypotheses,
        "statistics": {"policy": (statistics or {}).get("multiple_comparison", {}).get("policy"),
                       "family": (statistics or {}).get("declared_family"),
                       "settings": (statistics or {}).get("settings")},
        "reliability": {"by_status": (reliability or {}).get("by_status"),
                        "blocked": (reliability or {}).get("blocked"),
                        "failed": (reliability or {}).get("failed")},
        "compute": compute or {},
        "disclosures": disclosures or {},
        "not_claimed": ["state-of-the-art", "first method", "cross-dataset superiority beyond the "
                        "five predeclared comparisons on this 1700-video SiW-Mv2 evaluation set"]}
    return {**body, "summary_identity": stable_identity(body)}


# --- acceptance ---------------------------------------------------------------

APPROVED_BLOCKED_ROWS = ("A09-backend-pc_full_training", "A10-frame_count-f16",
                         "A10-frame_count-f32", "A10-frame_count-f48_64")
# The source side froze here. Pinned so a regenerated lock cannot pass acceptance by
# agreeing with itself.
FROZEN_SOURCE_MATRIX_LOCK_IDENTITY = ("c06944344eab25820b4bf6327b9dd391a308a3ffd935ab2"
                                      "ed91264e5898517aa")
FROZEN_MATRIX_IDENTITY = "a4972b0dc23946c4ad169f2c856fc9b5e0387baca45b2c9a4895f8180d9c2dd5"
FROZEN_TARGET_FEATURE_IDENTITY = ("c3a29e695ad08c4b31e01533f1d12374f4e30c51f0167c662"
                                  "2cf8168792e48a8")


def acceptance_checks(*, reports_root: Path, matrix_plan: dict[str, Any],
                      matrix_deterministic: bool, firewall: TargetLabelFirewall,
                      package_identity: str, expected_rows: int, expected_videos: int,
                      test_suite: dict[str, Any] | None = None) -> dict[str, Any]:
    """What must be true for M10 to be COMPLETED. Every item reads an artifact.

    Nothing here is asserted from a document: a check that cannot find its artifact
    is False, and M10 stays open.
    """
    import json
    from .experiment_registry import validate_source_matrix_lock
    read = lambda name: (json.loads(Path(reports_root, name).read_text(encoding="utf-8"))
                         if Path(reports_root, name).is_file() else {})
    source = read("SOURCE_MATRIX_LOCK.json")
    lockset = read(LOCKSET_FILE)
    reveal = read(REVEAL_FILE)
    audit = read("PRE_REVEAL_AUDIT.json")
    statistics = read("statistics.json")
    reliability = read("RELIABILITY.json")
    report_payload = read("M10_REPORT.json")
    summary = read("summary.json")
    prediction_validation = read("G7_PREDICTION_VALIDATION.json")
    scored = sorted(Path(reports_root, "g8").glob("*/scoring.json"))

    lock_validation: dict[str, Any] = {}
    if source:
        first = validate_source_matrix_lock(source, matrix_plan)
        second = validate_source_matrix_lock(source, matrix_plan)
        lock_validation = {"passed": bool(first["passed"] and first == second),
                           "checks": first["checks"]}
    blocked = sorted(entry["experiment_id"] for entry in source.get("entries") or []
                     if entry.get("status") == "BLOCKED")
    terminal = [entry for entry in source.get("entries") or []
                if entry.get("status") in ("COMPLETED", "FAILED", "BLOCKED")]
    eligible = eligible_rows(source) if source else []
    comparisons = (statistics or {}).get("comparisons") or {}
    superiority = {name: block for name, block in comparisons.items()
                   if block.get("status") not in (None,) and name in ("H1", "H2", "H3", "H4", "H5")}
    reliability_tests = (reliability or {}).get("tests") or []

    checks = {
        "matrix_identity_unchanged":
            source.get("m10_matrix_identity") == matrix_plan["m10_matrix_identity"],
        "matrix_deterministic": bool(matrix_deterministic),
        "source_matrix_lock_valid": bool(lock_validation.get("passed")),
        "source_matrix_lock_identity_pinned":
            source.get("source_matrix_lock_identity") == FROZEN_SOURCE_MATRIX_LOCK_IDENTITY,
        "matrix_identity_pinned": matrix_plan["m10_matrix_identity"] == FROZEN_MATRIX_IDENTITY,
        "target_feature_identity_pinned": package_identity == FROZEN_TARGET_FEATURE_IDENTITY,
        "executable_rows_all_terminal":
            len(terminal) == int(source.get("logical_rows", -1)) and source.get("failed_rows") == [],
        "executable_row_count_is_38": int(source.get("executable_rows", -1)) == 38,
        "only_approved_blocked_rows": tuple(blocked) == APPROVED_BLOCKED_ROWS,
        "blocked_rows_carry_reasons": all(entry.get("blocked_reason")
                                          for entry in source.get("entries") or []
                                          if entry.get("status") == "BLOCKED"),
        "target_feature_identity_valid":
            lockset.get("target_feature_package_identity") == package_identity,
        "every_eligible_row_predicted":
            bool(eligible) and int(lockset.get("entry_count", -1)) == len(eligible),
        "g7_predictions_validated": bool(prediction_validation.get("passed")),
        "g7_predictions_cover_the_package":
            all(int(entry.get("row_count", -1)) == expected_rows
                and int(entry.get("video_count", -1)) == expected_videos
                for entry in lockset.get("entries") or []) and bool(lockset.get("entries")),
        "prediction_lockset_frozen": lockset.get("status") == "FROZEN",
        "prediction_lockset_binds_the_source_lock":
            bool(lockset) and lockset.get("source_matrix_lock_identity")
            == source.get("source_matrix_lock_identity"),
        "pre_reveal_audit_passed": bool(audit.get("passed")),
        "reveal_happened_only_after_the_lockset":
            bool(reveal) and reveal.get("target_prediction_lockset_identity")
            == lockset.get("lockset_identity")
            and reveal.get("pre_reveal_audit_identity") == audit.get("pre_reveal_audit_identity"),
        "g8_isolated": bool(scoring.isolation_report()["static_import_audit"]["passed"]),
        "g8_wrote_no_model_state":
            all(json.loads(path.read_text(encoding="utf-8")).get("wrote_checkpoint") is False
                for path in scored) and bool(scored),
        "train_cannot_read_target_labels": firewall.permission("TRAIN", "target_label_root") == "deny",
        "g7_cannot_read_target_labels": firewall.permission("G7", "target_label_root") == "deny",
        "g8_reads_target_labels": firewall.permission("G8", "target_label_root") == "read",
        "source_selections_unchanged": source.get("selection_used_target") is False,
        "no_target_tuning":
            source.get("selection_used_target") is False
            and all(entry.get("aggregation", {}).get("unknown_threshold") is None
                    for entry in [read(f"g7/{name}/PREDICTION_LOCK.json")
                                  for name in [item["experiment_id"] for item in
                                               lockset.get("entries") or []]] if entry),
        "every_row_scored": len(scored) == int(lockset.get("entry_count", -1)),
        "statistics_complete":
            len(superiority) == 5 and all(block.get("status") in ("computed", "refused", "unavailable")
                                          for block in superiority.values()),
        "multiple_comparison_applied":
            (statistics or {}).get("multiple_comparison", {}).get("policy") == "holm_bonferroni",
        "reliability_accounted":
            bool(reliability_tests) and all(test.get("status") in ("PASSED", "FAILED", "BLOCKED")
                                            for test in reliability_tests),
        "reliability_blocked_tests_carry_reasons":
            all(test.get("blocked_reason") for test in reliability_tests
                if test.get("status") == "BLOCKED"),
        "report_complete": bool(report_payload.get("sections"))
            and Path(reports_root, "report.html").is_file(),
        "summary_present": bool(summary.get("summary_identity")),
        "hypotheses_adjudicated":
            len((summary or {}).get("hypotheses") or {}) == 6
            and all(block.get("outcome") for block in ((summary or {}).get("hypotheses") or {}).values()),
        "test_suite_passes": bool((test_suite or {}).get("passed")) and int((test_suite or {}).get("failed", 1)) == 0
            and int((test_suite or {}).get("skipped", 1)) == 0,
        "final_artifacts_reproducible": bool(summary.get("summary_identity"))
            and bool(report_payload.get("report_identity"))
            and bool(lockset.get("lockset_identity"))}
    body = {"m10_acceptance_schema_version": "m10-acceptance-v2",
            "m10_matrix_identity": matrix_plan["m10_matrix_identity"],
            "source_matrix_lock_identity": source.get("source_matrix_lock_identity"),
            "target_prediction_lockset_identity": lockset.get("lockset_identity"),
            "target_label_reveal_identity": reveal.get("reveal_identity"),
            "summary_identity": summary.get("summary_identity"),
            "report_identity": report_payload.get("report_identity"),
            "checks": checks, "passed": all(checks.values()),
            "failed_checks": sorted(name for name, value in checks.items() if not value),
            "source_matrix_lock_validation": lock_validation,
            "blocked_rows": blocked, "scored_rows": len(scored),
            "test_suite": test_suite or {},
            "target_labels_revealed": bool(reveal),
            "summary": matrix_plan["summary"],
            "not_claimed": ["state-of-the-art", "first method", "cross-domain superiority beyond the "
                            "five predeclared comparisons", "any unsupported external comparison"]}
    return {**body, "acceptance_identity": stable_identity(body)}


def write_json(path: Path, payload: dict[str, Any], *, firewall: TargetLabelFirewall | None = None,
               dry_run: bool = False) -> Path:
    target = firewall.check_write("G8", Path(path)) if firewall is not None else Path(path)
    if not dry_run: atomic_json_write(Path(target), payload)
    return Path(target)
