"""POST_FAILURE_EXPLORATORY_TARGET_V1 — Phase E2, label unlock + scoring.

**Structural guarantee: this module holds no training capability.** It
imports numpy and pyarrow. It does NOT import `torch`,
`prism_fas.detector.trainer`, `prism_fas.detector.checkpoint`, or
`prism_fas.evaluation.synthetic_real_probe` (the module that constructs and
loads a real checkpoint) — `static_import_audit`, reused verbatim from
`prism_fas.evaluation.scoring`, proves that from this module's own AST
rather than from a comment.

Phase E2 may run only after a valid, `FROZEN` `TARGET_PREDICTION_LOCK`
exists (Phase E1, `post_failure_exploratory_target.py`). It reuses
`prism_fas.evaluation.scoring.score`/`load_evaluation_labels`/
`target_label_reveal` verbatim — the exact G8 implementation the legacy M10
confirmatory path already froze — and never retrains, recalibrates, selects
a checkpoint, or mutates a prediction file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EXIT_PASS, EXIT_BLOCKED, EXIT_USAGE = 0, 2, 3

DIAGNOSTICS_DIR = "reports/full/exploratory_target_v1"
PREDICTION_LOCK_PATH = f"{DIAGNOSTICS_DIR}/TARGET_PREDICTION_LOCK.json"
SCORE_RESULT_PATH = f"{DIAGNOSTICS_DIR}/EXPLORATORY_TARGET_SCORE_RESULT.json"
LABEL_REVEAL_PATH = f"{DIAGNOSTICS_DIR}/TARGET_LABEL_REVEAL.json"

#: Names this module must never import. Reused from `scoring.py`'s own list
#: plus the two paths specific to this branch (the checkpoint-construction
#: functions the predictor uses).
FORBIDDEN_IMPORTS = ("torch", "prism_fas.detector.trainer", "prism_fas.detector.checkpoint",
                    "prism_fas.train.trainer", "prism_fas.train.b00_pipeline", "torch.optim",
                    "prism_fas.evaluation.synthetic_real_probe")


class ExploratoryScoringError(RuntimeError):
    """The exploratory scorer cannot proceed with the inputs given."""


def assert_no_training_capability() -> dict[str, Any]:
    """This module's own AST audit, reusing `scoring.static_import_audit`
    verbatim but applied to THIS file and THIS forbidden-import list."""
    from prism_fas.evaluation.scoring import static_import_audit

    audit = static_import_audit(Path(__file__))
    violations = sorted({name for name in audit["module_level_imports"]
                         for forbidden in FORBIDDEN_IMPORTS
                         if name == forbidden or name.startswith(forbidden + ".")})
    if violations:
        raise ExploratoryScoringError(
            f"the exploratory scorer must have no training/checkpoint-loading capability; "
            f"found {violations}")
    return {**audit, "forbidden_imports_checked": list(FORBIDDEN_IMPORTS), "violations": violations}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def require_frozen_prediction_lock(repo: Path) -> dict[str, Any]:
    """The one precondition Phase E2 may never bypass: a `FROZEN`,
    structurally valid `TARGET_PREDICTION_LOCK` — never a partial or
    unlocked one, and never one this module recomputed."""
    path = Path(repo) / PREDICTION_LOCK_PATH
    lock = _read_json(path)
    if lock is None:
        raise ExploratoryScoringError(
            f"{PREDICTION_LOCK_PATH} does not exist; Phase E1 (blind prediction) has not "
            "produced a lock yet — refusing to open any target label before it does")
    if lock.get("status") != "FROZEN":
        raise ExploratoryScoringError(
            f"the prediction lock's status is {lock.get('status')!r}, not FROZEN; refusing "
            "to open a target label before every prediction in the matrix is locked")
    if lock.get("target_labels_opened") is not False:
        raise ExploratoryScoringError(
            "the prediction lock does not record target_labels_opened: false; fail closed")
    return lock


# ==============================================================================
# Video-level, arm-paired exploratory comparisons (E-H1..E-H4)
# ==============================================================================

def paired_bootstrap_acer_difference(acer_a: Any, acer_b: Any, *, seed: int = 20260810,
                                     resamples: int = 10000, confidence_level: float = 0.95
                                     ) -> dict[str, Any]:
    """Paired, video-level bootstrap CI for `mean(acer_a) - mean(acer_b)`
    over matched videos — pure numpy, no label access, no model. Used only
    AFTER Phase E2 scoring has produced per-video ACER contributions for
    both arms of one exploratory comparison."""
    import numpy as np

    a = np.asarray(acer_a, dtype=np.float64)
    b = np.asarray(acer_b, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise ExploratoryScoringError("paired bootstrap requires two equal-length, non-empty arrays")
    observed = float(a.mean() - b.mean())
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    n = a.size
    diffs = np.empty(int(resamples), dtype=np.float64)
    for i in range(int(resamples)):
        indices = rng.integers(0, n, size=n)
        diffs[i] = float(a[indices].mean() - b[indices].mean())
    alpha = 1.0 - float(confidence_level)
    lower = float(np.percentile(diffs, 100 * (alpha / 2)))
    upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    p_value = float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {"observed_difference": observed, "ci_lower": lower, "ci_upper": upper,
           "p_value_two_sided": min(1.0, p_value), "resamples": int(resamples), "seed": int(seed),
           "paired": True, "bootstrap_unit": "video"}


def holm_bonferroni(p_values: dict[str, float], *, alpha: float = 0.05) -> dict[str, dict[str, Any]]:
    """Holm-Bonferroni step-down correction over the named exploratory
    comparison family. Pure arithmetic; never touches a score or a label."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered)
    out: dict[str, dict[str, Any]] = {}
    reject_seen_false = False
    for rank, (name, p_value) in enumerate(ordered, start=1):
        adjusted_alpha = alpha / (m - rank + 1)
        significant = (not reject_seen_false) and (p_value <= adjusted_alpha)
        if not significant:
            reject_seen_false = True
        out[name] = {"p_value": float(p_value), "rank": rank,
                    "adjusted_alpha": float(adjusted_alpha), "significant": bool(significant)}
    return out


# ==============================================================================
# CLI
# ==============================================================================

def _preflight_score(repo: Path) -> tuple[int, dict[str, Any]]:
    report: dict[str, Any] = {"prediction_lock_valid": False, "target_labels_opened": False,
                              "target_access": 0, "checkpoint_weights_loaded": False,
                              "model_loaded": False}
    try:
        assert_no_training_capability()
        report["no_training_capability_verified"] = True
    except ExploratoryScoringError as error:
        report["error"] = str(error)
        return EXIT_BLOCKED, report

    try:
        lock = require_frozen_prediction_lock(repo)
        report["prediction_lock_valid"] = True
        report["prediction_lock_entry_count"] = lock.get("entry_count")
    except ExploratoryScoringError as error:
        report["error"] = str(error)
        return EXIT_BLOCKED, report

    score_result = _read_json(Path(repo) / SCORE_RESULT_PATH)
    report["score_result_exists"] = score_result is not None
    return EXIT_PASS, report


SCORE_ROWS_DIR = f"{DIAGNOSTICS_DIR}/scores"


def _build_firewall(repo: Path, protocol: dict[str, Any]) -> Any:
    from prism_fas.evaluation.firewall import FirewallConfig, TargetLabelFirewall

    roots = {name: Path(str(protocol["roots"][name])) for name in
            ("source_package_root", "target_feature_root", "target_label_root", "prediction_root")}
    config = FirewallConfig(roots=roots, permissions={stage: dict(protocol["permissions"][stage])
                                                      for stage in ("TRAIN", "G7", "G8")}).validate()
    return TargetLabelFirewall(config=config, project_root=Path(repo))


def score_one_row(repo: Path, row_id: str, *, labels: Any) -> dict[str, Any]:
    """One row's real frame+video metrics, reusing `scoring.score` verbatim.
    The row's OWN `PREDICTION_LOCK.json` and locked predictions are read
    from disk — never recomputed, never re-inferred."""
    from prism_fas.evaluation.scoring import score
    from prism_fas.evaluation.target_prediction import PREDICTION_LOCK_FILE, read_predictions

    row_run_dir = Path(repo) / "runs/exploratory_target_v1" / row_id
    row_lock = _read_json(row_run_dir / PREDICTION_LOCK_FILE)
    if row_lock is None:
        raise ExploratoryScoringError(f"{row_id}: no per-row PREDICTION_LOCK.json on disk")
    predictions = read_predictions(row_run_dir / "target_predictions.parquet")
    threshold = float(row_lock["aggregation"]["threshold"])
    return score(predictions=predictions, lock=row_lock, labels=labels, threshold=threshold)


def compute_exploratory_comparisons(scored_rows: dict[str, dict[str, Any]],
                                    row_meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """E-H1..E-H4 (section 6), computed ONLY from already-scored rows'
    `video_scores` — never a label, never a model. Paired by shared
    `video_id` between the two arms being compared, bootstrapped per
    `paired_bootstrap_acer_difference`."""
    def _video_error(video_scores: list[dict[str, Any]]) -> dict[str, float]:
        return {str(row["video_id"]): float(row["decision"] != ("spoof" if row["label"] else "live"))
               for row in video_scores}

    def _paired(arm_a_rows: list[str], arm_b_rows: list[str]) -> dict[str, Any]:
        errors_a: dict[str, float] = {}
        for row_id in arm_a_rows:
            errors_a.update(_video_error(scored_rows[row_id]["video_scores"]))
        errors_b: dict[str, float] = {}
        for row_id in arm_b_rows:
            errors_b.update(_video_error(scored_rows[row_id]["video_scores"]))
        shared = sorted(set(errors_a) & set(errors_b))
        if not shared:
            raise ExploratoryScoringError("no shared video_id between the two comparison arms")
        a = [errors_a[video_id] for video_id in shared]
        b = [errors_b[video_id] for video_id in shared]
        return paired_bootstrap_acer_difference(a, b)

    def _rows_for(track: str, arm: str) -> list[str]:
        return sorted(row_id for row_id, meta in row_meta.items()
                     if meta["track"] == track and meta["arm"] == arm)

    comparisons: dict[str, Any] = {}
    comparisons["E-H1"] = {
        "RND_vs_DET": _paired(_rows_for("G", "RND"), _rows_for("G", "DET")),
        "RND_vs_LLM": _paired(_rows_for("G", "RND"), _rows_for("G", "LLM")),
        "DET_vs_LLM": _paired(_rows_for("G", "DET"), _rows_for("G", "LLM"))}
    comparisons["E-H2"] = _paired(_rows_for("R", "DET"), _rows_for("R", "LLM"))
    noprompt_rows = sorted(row_id for row_id, meta in row_meta.items()
                          if meta["experiment_id"] == "C-R-NOPROMPT")
    comparisons["E-H3"] = _paired(_rows_for("R", "LLM"), noprompt_rows)
    comparisons["E-H4"] = {"DET": _paired(_rows_for("G", "DET"), _rows_for("R", "DET")),
                           "LLM": _paired(_rows_for("G", "LLM"), _rows_for("R", "LLM"))}
    p_values = {"E-H2": comparisons["E-H2"]["p_value_two_sided"],
               "E-H3": comparisons["E-H3"]["p_value_two_sided"],
               "E-H4_DET": comparisons["E-H4"]["DET"]["p_value_two_sided"],
               "E-H4_LLM": comparisons["E-H4"]["LLM"]["p_value_two_sided"]}
    return {"comparisons": comparisons, "holm_bonferroni": holm_bonferroni(p_values)}


def _score(repo: Path) -> tuple[int, dict[str, Any]]:
    """Phase E2 execution. NEVER invoked on this laptop for real — no
    genuine `TARGET_PREDICTION_LOCK` or label artifact exists here.
    No-rerun: a complete score result re-reports; a partial per-row score
    set BLOCKS rather than being silently completed or overwritten."""
    from prism_fas.pipeline.state import atomic_write_json

    report: dict[str, Any] = {"scored": False, "target_access": 0,
                              "checkpoint_weights_loaded": False, "model_loaded": False}
    result_path = Path(repo) / SCORE_RESULT_PATH
    existing = _read_json(result_path)
    if existing is not None:
        report.update({"scored": True, "reused_existing_score_result": True,
                      "target_labels_reopened": False})
        return EXIT_PASS, report

    try:
        assert_no_training_capability()
        lock = require_frozen_prediction_lock(repo)
    except ExploratoryScoringError as error:
        report["error"] = str(error)
        return EXIT_BLOCKED, report

    expected_rows = int(lock["entry_count"])
    rows_dir = Path(repo) / SCORE_ROWS_DIR
    existing_row_files = sorted(rows_dir.glob("*.json")) if rows_dir.is_dir() else []
    if 0 < len(existing_row_files) < expected_rows:
        report.update({"error": "PARTIAL_SCIENTIFIC_RESULT_SET",
                      "present": len(existing_row_files), "expected": expected_rows})
        return EXIT_BLOCKED, report

    try:
        from prism_fas.evaluation.post_failure_exploratory_target import load_protocol
        from prism_fas.evaluation.scoring import load_evaluation_labels

        protocol = load_protocol(repo)
        firewall = _build_firewall(repo, protocol)
        label_path = (Path(repo) / protocol["target_label_root"]["path"]
                     / protocol["target_label_root"]["artifact"])
        labels = load_evaluation_labels(label_path, firewall=firewall, stage="G8")
    except Exception as error:                            # noqa: BLE001
        report["error"] = f"{type(error).__name__}: {error}"
        return EXIT_BLOCKED, report

    try:
        from prism_fas.evaluation.post_failure_exploratory_target import resolve_target_matrix

        # The real 24-row matrix, resolved the SAME way Phase E1 did — never
        # reconstructed from the lockset's summarized entries, which carry
        # no `row_id` field of their own.
        rows = resolve_target_matrix(repo)
        row_meta = {row.row_id: {"track": row.track, "arm": row.arm,
                                 "experiment_id": row.experiment_id} for row in rows}
        if len(row_meta) != expected_rows:
            raise ExploratoryScoringError(
                f"resolved {len(row_meta)} target rows, lock declares entry_count={expected_rows}")
        scored_rows: dict[str, dict[str, Any]] = {}
        for row_id in sorted(row_meta):
            row_result = score_one_row(repo, row_id, labels=labels)
            scored_rows[row_id] = row_result
            atomic_write_json(rows_dir / f"{row_id}.json", row_result)
    except Exception as error:                            # noqa: BLE001
        report["error"] = f"{type(error).__name__}: {error}"
        return EXIT_BLOCKED, report

    try:
        comparisons = compute_exploratory_comparisons(scored_rows, row_meta)
    except ExploratoryScoringError as error:
        report["error"] = f"comparisons could not be computed: {error}"
        return EXIT_BLOCKED, report

    combined = {"schema_version": "post-failure-exploratory-target-v1-score-result-v1",
               "prediction_lock_identity": lock["lockset_identity"],
               "row_count": len(scored_rows), "rows": scored_rows,
               "exploratory_comparisons": comparisons,
               "target_labels_opened": True, "c9_may_close": False,
               "ba_sep_observed_verdict": "FAIL",
               "detector_reliability_lock_c_observed_overall": "FAILED",
               "post_failure_diagnostics_v2": "FAIL",
               "c9_original_confirmatory_path": "BLOCKED",
               "exploratory_target_status": "POST_FAILURE_EXPLORATORY",
               "target_access": 0}
    atomic_write_json(result_path, combined)
    report.update({"scored": True, "reused_existing_score_result": False, "row_count": len(scored_rows),
                  "score_result_path": SCORE_RESULT_PATH})
    return EXIT_PASS, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m prism_fas.evaluation.post_failure_exploratory_target_scorer",
        description="POST_FAILURE_EXPLORATORY_TARGET_V1 — Phase E2 label-unlock scoring. "
                    "Holds no training or checkpoint-loading capability.")
    parser.add_argument("--repo", default=".", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-score", action="store_true")
    mode.add_argument("--score", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.preflight_score:
        exit_code, payload = _preflight_score(args.repo)
    else:
        exit_code, payload = _score(args.repo)

    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["DIAGNOSTICS_DIR", "PREDICTION_LOCK_PATH", "SCORE_RESULT_PATH", "LABEL_REVEAL_PATH",
           "FORBIDDEN_IMPORTS", "ExploratoryScoringError", "assert_no_training_capability",
           "require_frozen_prediction_lock", "paired_bootstrap_acer_difference",
           "holm_bonferroni", "EXIT_PASS", "EXIT_BLOCKED", "EXIT_USAGE"]
