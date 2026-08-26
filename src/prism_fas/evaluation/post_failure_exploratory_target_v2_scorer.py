"""POST_FAILURE_EXPLORATORY_TARGET_V2 — Phase E2, label unlock + scoring.

Corrects three statistical/scoring defects found by pre-target audit of V1's
scorer:

  H. `compute_exploratory_comparisons` converted each video to a binary
     error and took `mean(error_A) - mean(error_B)` — a raw, class-
     unweighted accuracy difference, NOT ACER, and wrong whenever live/spoof
     counts differ (785 live vs 915 spoof on the frozen target). Corrected:
     `class_stratified_paired_bootstrap` computes real
     `APCER`/`BPCER`/`ACER = 0.5*(APCER+BPCER)` from the correct class
     populations.
  I. The same function built `video_id -> error` via repeated `.update()`
     across every seed sharing an arm, so later seeds silently overwrote
     earlier ones — most replication evidence was discarded. Corrected:
     every matched seed's per-video decisions are kept in a separate
     mapping and averaged only at the final `delta_seed` step; no seed ever
     overwrites another.
  J. The Holm-Bonferroni family was `family_size: 4`, one p-value per E-H,
     undercounting E-H1's three pairwise arms and E-H4's two matched-arm
     pairs. Corrected: all seven atomic comparisons enter one Holm family.

**Structural guarantee, unchanged from V1: this module holds no training
capability.** It imports numpy and pyarrow, never `torch`,
`prism_fas.detector.trainer`, `prism_fas.detector.checkpoint`, or
`prism_fas.evaluation.synthetic_real_probe` — `static_import_audit`, reused
verbatim, proves it from the AST.

Phase E2 may run only after a valid, `FROZEN`, row_id-keyed V2
`TARGET_PREDICTION_LOCK.json` exists (`post_failure_exploratory_target_v2.py`).
Row metadata (`row_id`/`experiment_id`/`track`/`arm`/`seed`/`threshold`/
`prediction_variant_id`) is read from that validated frozen lock — never
re-resolved from a possibly-drifted source repository.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from prism_fas.evaluation.contracts import stable_identity

EXIT_PASS, EXIT_BLOCKED, EXIT_USAGE = 0, 2, 3

DIAGNOSTICS_DIR = "reports/full/exploratory_target_v2"
RUN_DIR = "runs/exploratory_target_v2"
PREDICTION_PLAN_BINDING_PATH = f"{DIAGNOSTICS_DIR}/PREDICTION_PLAN_BINDING.json"
PREDICTION_LOCK_PATH = f"{DIAGNOSTICS_DIR}/TARGET_PREDICTION_LOCK.json"
SCORE_RESULT_PATH = f"{DIAGNOSTICS_DIR}/EXPLORATORY_TARGET_SCORE_RESULT.json"
SCORE_ROWS_DIR = f"{DIAGNOSTICS_DIR}/scores"
LABEL_REVEAL_PATH = f"{DIAGNOSTICS_DIR}/TARGET_LABEL_REVEAL.json"

EXPECTED_TOTAL_ROWS = 24
EXPECTED_ATOMIC_COMPARISONS = 7

#: Reused from V1's scorer plus the V2 predictor (which this module must
#: also never import at module level, to stay checkpoint-loading-free).
FORBIDDEN_IMPORTS = ("torch", "prism_fas.detector.trainer", "prism_fas.detector.checkpoint",
                    "prism_fas.train.trainer", "prism_fas.train.b00_pipeline", "torch.optim",
                    "prism_fas.evaluation.synthetic_real_probe")


class ExploratoryScoringV2Error(RuntimeError):
    """The V2 exploratory scorer cannot proceed with the inputs given."""


def assert_no_training_capability() -> dict[str, Any]:
    from prism_fas.evaluation.scoring import static_import_audit

    audit = static_import_audit(Path(__file__))
    violations = sorted({name for name in audit["module_level_imports"]
                         for forbidden in FORBIDDEN_IMPORTS
                         if name == forbidden or name.startswith(forbidden + ".")})
    if violations:
        raise ExploratoryScoringV2Error(
            f"the V2 exploratory scorer must have no training/checkpoint-loading capability; "
            f"found {violations}")
    return {**audit, "forbidden_imports_checked": list(FORBIDDEN_IMPORTS), "violations": violations}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def require_frozen_prediction_lockset(repo: Path) -> dict[str, Any]:
    """The one precondition Phase E2 may never bypass: a `FROZEN`,
    structurally valid, row_id-keyed V2 lockset with exactly 24 entries."""
    from prism_fas.evaluation.post_failure_exploratory_target_v2 import (
        EXPECTED_TOTAL_ROWS as expected, validate_existing_exploratory_prediction_result)

    validation = validate_existing_exploratory_prediction_result(repo)
    if not validation["valid"]:
        raise ExploratoryScoringV2Error(
            f"the V2 prediction lockset failed validation: {validation['problems']}")
    lockset = validation["lockset"]
    if lockset.get("status") != "FROZEN":
        raise ExploratoryScoringV2Error(f"lockset status is {lockset.get('status')!r}, not FROZEN")
    if int(lockset.get("entry_count", -1)) != expected:
        raise ExploratoryScoringV2Error(f"lockset entry_count is {lockset.get('entry_count')}, expected {expected}")
    if lockset.get("target_labels_opened") is not False:
        raise ExploratoryScoringV2Error("lockset does not record target_labels_opened: false")
    return lockset


# ==============================================================================
# Class-stratified paired video bootstrap (Defects H, I; protocol section
# "statistics.design_steps")
# ==============================================================================

def apcer_bpcer_acer(decisions: Mapping[str, str], *, live_ids: list[str],
                     spoof_ids: list[str]) -> dict[str, float]:
    """`APCER = count(spoof predicted NOT spoof) / count(spoof)`,
    `BPCER = count(live predicted spoof) / count(live)`,
    `ACER = 0.5*(APCER+BPCER)` — never a raw, class-unweighted error rate."""
    if not live_ids or not spoof_ids:
        raise ExploratoryScoringV2Error("ACER requires both non-empty LIVE and SPOOF populations")
    apcer = sum(1 for video_id in spoof_ids if decisions[video_id] != "spoof") / len(spoof_ids)
    bpcer = sum(1 for video_id in live_ids if decisions[video_id] == "spoof") / len(live_ids)
    return {"apcer": float(apcer), "bpcer": float(bpcer), "acer": float(0.5 * (apcer + bpcer))}


def class_stratified_paired_bootstrap(decisions_a_by_seed: Mapping[int, Mapping[str, str]],
                                      decisions_b_by_seed: Mapping[int, Mapping[str, str]],
                                      labels: Mapping[str, int], *, seed: int = 20260810,
                                      resamples: int = 10000, confidence_level: float = 0.95
                                      ) -> dict[str, Any]:
    """The frozen class-stratified paired video bootstrap. Every matched
    seed contributes its own `ACER_A(seed) - ACER_B(seed)`; seeds are FIXED
    replications, never themselves resampled, and never overwrite one
    another (Defect I). Live and spoof video IDs are resampled with
    replacement WITHIN their own class only, using the SAME resampled IDs
    for both compared methods and every matched seed (Defect H).

    Tie handling: a bootstrap replicate statistic exactly equal to zero
    counts toward BOTH tail probabilities (`<=0` and `>=0` are each
    inclusive) — the standard two-sided bootstrap-sign convention, applied
    by this one, single, frozen implementation only.
    """
    import numpy as np

    matched_seeds = sorted(set(decisions_a_by_seed) & set(decisions_b_by_seed))
    if not matched_seeds:
        raise ExploratoryScoringV2Error("no matched seed between the two compared arms")

    live_ids = sorted(video_id for video_id, label in labels.items() if label == 0)
    spoof_ids = sorted(video_id for video_id, label in labels.items() if label == 1)
    if not live_ids or not spoof_ids:
        raise ExploratoryScoringV2Error("labels must contain both LIVE and SPOOF videos")

    per_seed_observed: dict[int, dict[str, Any]] = {}
    observed_deltas: list[float] = []
    live_a: dict[int, "np.ndarray"] = {}
    spoof_a: dict[int, "np.ndarray"] = {}
    live_b: dict[int, "np.ndarray"] = {}
    spoof_b: dict[int, "np.ndarray"] = {}
    for matched_seed in matched_seeds:
        decisions_a = decisions_a_by_seed[matched_seed]
        decisions_b = decisions_b_by_seed[matched_seed]
        missing_a = (set(live_ids) | set(spoof_ids)) - set(decisions_a)
        missing_b = (set(live_ids) | set(spoof_ids)) - set(decisions_b)
        if missing_a or missing_b:
            raise ExploratoryScoringV2Error(
                f"seed {matched_seed}: compared rows do not score the same locked target video IDs")
        metrics_a = apcer_bpcer_acer(decisions_a, live_ids=live_ids, spoof_ids=spoof_ids)
        metrics_b = apcer_bpcer_acer(decisions_b, live_ids=live_ids, spoof_ids=spoof_ids)
        delta = metrics_a["acer"] - metrics_b["acer"]
        per_seed_observed[matched_seed] = {"acer_a": metrics_a["acer"], "acer_b": metrics_b["acer"],
                                           "apcer_a": metrics_a["apcer"], "bpcer_a": metrics_a["bpcer"],
                                           "apcer_b": metrics_b["apcer"], "bpcer_b": metrics_b["bpcer"],
                                           "delta": delta}
        observed_deltas.append(delta)
        live_a[matched_seed] = np.array([decisions_a[video_id] == "spoof" for video_id in live_ids])
        spoof_a[matched_seed] = np.array([decisions_a[video_id] != "spoof" for video_id in spoof_ids])
        live_b[matched_seed] = np.array([decisions_b[video_id] == "spoof" for video_id in live_ids])
        spoof_b[matched_seed] = np.array([decisions_b[video_id] != "spoof" for video_id in spoof_ids])

    observed_statistic = float(np.mean(observed_deltas))

    rng = np.random.Generator(np.random.PCG64(int(seed)))
    n_live, n_spoof = len(live_ids), len(spoof_ids)
    replicate_statistics = np.empty(int(resamples), dtype=np.float64)
    for i in range(int(resamples)):
        live_sample = rng.integers(0, n_live, size=n_live)
        spoof_sample = rng.integers(0, n_spoof, size=n_spoof)
        deltas = np.empty(len(matched_seeds), dtype=np.float64)
        for position, matched_seed in enumerate(matched_seeds):
            bpcer_a = float(live_a[matched_seed][live_sample].mean())
            apcer_a = float(spoof_a[matched_seed][spoof_sample].mean())
            bpcer_b = float(live_b[matched_seed][live_sample].mean())
            apcer_b = float(spoof_b[matched_seed][spoof_sample].mean())
            deltas[position] = 0.5 * (apcer_a + bpcer_a) - 0.5 * (apcer_b + bpcer_b)
        replicate_statistics[i] = float(deltas.mean())

    alpha = 1.0 - float(confidence_level)
    lower = float(np.percentile(replicate_statistics, 100 * (alpha / 2)))
    upper = float(np.percentile(replicate_statistics, 100 * (1 - alpha / 2)))
    p_value = float(min(1.0, 2 * min((replicate_statistics <= 0).mean(),
                                     (replicate_statistics >= 0).mean())))
    return {"matched_seeds": matched_seeds, "per_seed_observed": per_seed_observed,
           "observed_statistic": observed_statistic, "ci_lower": lower, "ci_upper": upper,
           "p_value_two_sided": p_value, "resamples": int(resamples), "seed": int(seed),
           "confidence_level": float(confidence_level), "paired": True, "class_stratified": True,
           "bootstrap_unit": "video"}


def holm_bonferroni(p_values: Mapping[str, float], *, alpha: float = 0.05) -> dict[str, dict[str, Any]]:
    """Holm-Bonferroni step-down correction — unchanged from V1 (not
    flagged defective), reused here over the corrected 7-member family."""
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
# Per-row scoring, reusing scoring.score verbatim
# ==============================================================================

def score_one_row(repo: Path, row_id: str, *, labels: Any, run_root: Path) -> dict[str, Any]:
    from prism_fas.evaluation.scoring import score
    from prism_fas.evaluation.target_prediction import PREDICTION_LOCK_FILE, read_predictions

    row_run_dir = Path(run_root) / row_id
    row_lock = _read_json(row_run_dir / PREDICTION_LOCK_FILE)
    if row_lock is None:
        raise ExploratoryScoringV2Error(f"{row_id}: no per-row PREDICTION_LOCK.json on disk")
    predictions = read_predictions(row_run_dir / "target_predictions.parquet")
    threshold = float(row_lock["aggregation"]["threshold"])
    return score(predictions=predictions, lock=row_lock, labels=labels, threshold=threshold)


# ==============================================================================
# The seven atomic comparisons (Defect J)
# ==============================================================================

def compute_exploratory_comparisons_v2(scored_rows: Mapping[str, Mapping[str, Any]],
                                       row_meta: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """All seven atomic E-H comparisons, each via
    `class_stratified_paired_bootstrap`. Never converts a video to a
    class-blind binary error; never lets one seed overwrite another."""
    decisions_by_row = {row_id: {row["video_id"]: row["decision"]
                                 for row in scored_rows[row_id]["video_scores"]}
                        for row_id in scored_rows}
    labels_by_row_id = next(iter(scored_rows.values()))["video_scores"]
    labels = {row["video_id"]: int(row["label"]) for row in labels_by_row_id}

    def _by_seed(track: str, arm: str | None = None, experiment_id: str | None = None
                ) -> dict[int, Mapping[str, str]]:
        matches = [row_id for row_id, meta in row_meta.items()
                  if meta["track"] == track
                  and (arm is None or meta["arm"] == arm)
                  and (experiment_id is None or meta["experiment_id"] == experiment_id)]
        return {int(row_meta[row_id]["seed"]): decisions_by_row[row_id] for row_id in matches}

    comparisons: dict[str, Any] = {}
    comparisons["E-H1_RND_vs_DET"] = class_stratified_paired_bootstrap(
        _by_seed("G", arm="RND"), _by_seed("G", arm="DET"), labels)
    comparisons["E-H1_RND_vs_LLM"] = class_stratified_paired_bootstrap(
        _by_seed("G", arm="RND"), _by_seed("G", arm="LLM"), labels)
    comparisons["E-H1_DET_vs_LLM"] = class_stratified_paired_bootstrap(
        _by_seed("G", arm="DET"), _by_seed("G", arm="LLM"), labels)
    comparisons["E-H2"] = class_stratified_paired_bootstrap(
        _by_seed("R", experiment_id="C-R-DET"), _by_seed("R", experiment_id="C-R-LLM"), labels)
    comparisons["E-H3"] = class_stratified_paired_bootstrap(
        _by_seed("R", experiment_id="C-R-LLM"), _by_seed("R", experiment_id="C-R-NOPROMPT"), labels)
    comparisons["E-H4_DET"] = class_stratified_paired_bootstrap(
        _by_seed("G", arm="DET"), _by_seed("R", experiment_id="C-R-DET"), labels)
    comparisons["E-H4_LLM"] = class_stratified_paired_bootstrap(
        _by_seed("G", arm="LLM"), _by_seed("R", experiment_id="C-R-LLM"), labels)

    if len(comparisons) != EXPECTED_ATOMIC_COMPARISONS:
        raise ExploratoryScoringV2Error(
            f"expected {EXPECTED_ATOMIC_COMPARISONS} atomic comparisons, computed {len(comparisons)}")
    p_values = {name: result["p_value_two_sided"] for name, result in comparisons.items()}
    return {"comparisons": comparisons, "holm_bonferroni": holm_bonferroni(p_values),
           "atomic_comparison_count": len(comparisons)}


# ==============================================================================
# Existing-score-result validation (section 15)
# ==============================================================================

def validate_existing_exploratory_score_result(repo: Path) -> dict[str, Any]:
    """Canonical validation of an on-disk `EXPLORATORY_TARGET_SCORE_RESULT.json`.
    Never rescoring; only re-deriving cheap identities and cross-checking
    the recorded content."""
    problems: list[str] = []
    repo = Path(repo)
    result = _read_json(repo / SCORE_RESULT_PATH)
    if result is None:
        return {"valid": False, "problems": ["no EXPLORATORY_TARGET_SCORE_RESULT.json on disk"], "result": None}

    try:
        lockset = require_frozen_prediction_lockset(repo)
    except ExploratoryScoringV2Error as error:
        return {"valid": False, "problems": [f"prediction lockset no longer valid: {error}"], "result": result}

    if result.get("prediction_lock_identity") != lockset.get("lockset_identity"):
        problems.append("score result's prediction_lock_identity does not match the current lockset")
    if int(result.get("row_count", -1)) != EXPECTED_TOTAL_ROWS:
        problems.append(f"score result row_count is {result.get('row_count')}, expected {EXPECTED_TOTAL_ROWS}")
    rows = dict(result.get("rows") or {})
    if set(rows) != set(lockset.get("entries") or {}):
        problems.append("score result row_ids do not exactly match the lockset's row_ids")
    comparisons = dict((result.get("exploratory_comparisons") or {}).get("comparisons") or {})
    if len(comparisons) != EXPECTED_ATOMIC_COMPARISONS:
        problems.append(f"score result has {len(comparisons)} comparisons, expected {EXPECTED_ATOMIC_COMPARISONS}")
    holm = dict((result.get("exploratory_comparisons") or {}).get("holm_bonferroni") or {})
    if set(holm) != set(comparisons):
        problems.append("Holm-Bonferroni correction does not cover exactly the recorded comparisons")
    for field, expected in (("ba_sep_observed_verdict", "FAIL"),
                            ("detector_reliability_lock_c_observed_overall", "FAILED"),
                            ("post_failure_diagnostics_v2", "FAIL"),
                            ("c9_original_confirmatory_path", "BLOCKED"),
                            ("c9_may_close", False), ("target_access", 0)):
        if result.get(field) != expected:
            problems.append(f"result.{field} is not {expected!r}")

    return {"valid": not problems, "problems": problems, "result": result}


# ==============================================================================
# CLI
# ==============================================================================

def _build_firewall(repo: Path, protocol: dict[str, Any]) -> Any:
    from prism_fas.evaluation.firewall import FirewallConfig, TargetLabelFirewall

    roots = {name: Path(str(protocol["roots"][name])) for name in
            ("source_package_root", "target_feature_root", "target_label_root", "prediction_root")}
    config = FirewallConfig(roots=roots, permissions={stage: dict(protocol["permissions"][stage])
                                                      for stage in ("TRAIN", "G7", "G8")}).validate()
    return TargetLabelFirewall(config=config, project_root=Path(repo))


def _preflight_score(repo: Path) -> tuple[int, dict[str, Any]]:
    report: dict[str, Any] = {"prediction_lockset_valid": False, "target_labels_opened": False,
                              "target_access": 0, "checkpoint_weights_loaded": False,
                              "model_loaded": False}
    try:
        assert_no_training_capability()
        report["no_training_capability_verified"] = True
    except ExploratoryScoringV2Error as error:
        report["error"] = str(error)
        return EXIT_BLOCKED, report

    try:
        lockset = require_frozen_prediction_lockset(repo)
        report["prediction_lockset_valid"] = True
        report["prediction_lockset_entry_count"] = lockset.get("entry_count")
    except ExploratoryScoringV2Error as error:
        report["error"] = str(error)
        return EXIT_BLOCKED, report

    score_result = _read_json(Path(repo) / SCORE_RESULT_PATH)
    report["score_result_exists"] = score_result is not None
    return EXIT_PASS, report


def _score(repo: Path) -> tuple[int, dict[str, Any]]:
    """Phase E2 execution. NEVER invoked on this laptop for real. Row
    metadata comes from the validated frozen lockset (Defect: 'scorer must
    use frozen prediction metadata'), never a fresh source-matrix
    resolution."""
    from prism_fas.pipeline.state import atomic_write_json

    report: dict[str, Any] = {"scored": False, "target_access": 0,
                              "checkpoint_weights_loaded": False, "model_loaded": False}
    result_path = Path(repo) / SCORE_RESULT_PATH
    if result_path.is_file():
        validation = validate_existing_exploratory_score_result(repo)
        if not validation["valid"]:
            report.update({"error": "EXISTING_RESULT_FAILED_VALIDATION", "problems": validation["problems"]})
            return EXIT_BLOCKED, report
        report.update({"scored": True, "reused_existing_score_result": True, "target_labels_reopened": False})
        return EXIT_PASS, report

    try:
        assert_no_training_capability()
        lockset = require_frozen_prediction_lockset(repo)
    except ExploratoryScoringV2Error as error:
        report["error"] = str(error)
        return EXIT_BLOCKED, report

    expected_rows = int(lockset["entry_count"])
    rows_dir = Path(repo) / SCORE_ROWS_DIR
    existing_row_files = sorted(rows_dir.glob("*.json")) if rows_dir.is_dir() else []
    if 0 < len(existing_row_files) < expected_rows:
        report.update({"error": "PARTIAL_SCIENTIFIC_RESULT_SET",
                      "present": len(existing_row_files), "expected": expected_rows})
        return EXIT_BLOCKED, report

    try:
        from prism_fas.evaluation.post_failure_exploratory_target_v2 import load_protocol
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
        # Row metadata from the VALIDATED FROZEN LOCKSET — never a fresh
        # source_matrix resolution, per the corrected scoring-authority rule.
        entries = dict(lockset["entries"])
        row_meta = {row_id: {"track": entry["track"], "arm": entry["arm"],
                             "experiment_id": entry["experiment_id"], "seed": entry["seed"],
                             "prediction_variant_id": entry["prediction_variant_id"]}
                   for row_id, entry in entries.items()}
        run_root = Path(repo) / RUN_DIR
        scored_rows: dict[str, dict[str, Any]] = {}
        for row_id in sorted(row_meta):
            row_result = score_one_row(repo, row_id, labels=labels, run_root=run_root)
            scored_rows[row_id] = row_result
            atomic_write_json(rows_dir / f"{row_id}.json", row_result)
    except Exception as error:                            # noqa: BLE001
        report["error"] = f"{type(error).__name__}: {error}"
        return EXIT_BLOCKED, report

    try:
        comparisons = compute_exploratory_comparisons_v2(scored_rows, row_meta)
    except ExploratoryScoringV2Error as error:
        report["error"] = f"comparisons could not be computed: {error}"
        return EXIT_BLOCKED, report

    combined = {"schema_version": "post-failure-exploratory-target-v2-score-result-v1",
               "prediction_lock_identity": lockset["lockset_identity"],
               "row_count": len(scored_rows), "rows": scored_rows,
               "exploratory_comparisons": comparisons,
               "target_labels_opened": True, "c9_may_close": False,
               "ba_sep_observed_verdict": "FAIL",
               "detector_reliability_lock_c_observed_overall": "FAILED",
               "post_failure_diagnostics_v2": "FAIL",
               "c9_original_confirmatory_path": "BLOCKED",
               "exploratory_target_status": "POST_FAILURE_EXPLORATORY",
               "target_access": 0}
    combined["score_result_identity"] = stable_identity(
        {key: value for key, value in combined.items() if key != "score_result_identity"})
    atomic_write_json(result_path, combined)
    report.update({"scored": True, "reused_existing_score_result": False, "row_count": len(scored_rows),
                  "atomic_comparison_count": comparisons["atomic_comparison_count"],
                  "score_result_path": SCORE_RESULT_PATH})
    return EXIT_PASS, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m prism_fas.evaluation.post_failure_exploratory_target_v2_scorer",
        description="POST_FAILURE_EXPLORATORY_TARGET_V2 — Phase E2 label-unlock scoring, "
                    "pre-target corrected. Holds no training or checkpoint-loading capability.")
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


__all__ = ["DIAGNOSTICS_DIR", "PREDICTION_LOCK_PATH", "SCORE_RESULT_PATH", "SCORE_ROWS_DIR",
           "LABEL_REVEAL_PATH", "EXPECTED_TOTAL_ROWS", "EXPECTED_ATOMIC_COMPARISONS",
           "FORBIDDEN_IMPORTS", "ExploratoryScoringV2Error", "assert_no_training_capability",
           "require_frozen_prediction_lockset", "apcer_bpcer_acer",
           "class_stratified_paired_bootstrap", "holm_bonferroni", "score_one_row",
           "compute_exploratory_comparisons_v2", "validate_existing_exploratory_score_result",
           "EXIT_PASS", "EXIT_BLOCKED", "EXIT_USAGE"]
