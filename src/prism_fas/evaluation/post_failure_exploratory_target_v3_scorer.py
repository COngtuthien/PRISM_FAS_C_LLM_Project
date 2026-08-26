"""POST_FAILURE_EXPLORATORY_TARGET_V3 — Phase E2, final pre-target
provenance/access/statistics hardening.

Corrects the remaining defects found by a final pre-target audit of V2's
scorer:

  D. `TARGET_LABEL_REVEAL.json` was declared in prose but never written —
     the scorer called `load_evaluation_labels` directly. Corrected:
     `reveal_target_labels` writes a real, one-way reveal artifact BEFORE
     the first label load, binding the prediction lock identity and the
     label artifact's own SHA-256, computed WITHOUT scoring.
  E. A complete 24-row score-file set with no final result would be
     rescored (reopening labels) on the next `--score` call. Corrected:
     that state is `INCOMPLETE_FINALIZATION` and BLOCKS.
  F. `validate_existing_exploratory_score_result` is hardened
     (`_v3`) to also verify the label-reveal identity, the prediction
     execution code commit, per-row score artifact identity, and to
     recompute Holm-Bonferroni from the RECORDED randomization p-values.
  G/H/I/J. The bootstrap now produces a CI ONLY.  A separate, frozen
     paired video-level sign-flip randomization test produces the
     exploratory p-value for each of the seven atomic comparisons, and
     Holm-Bonferroni is applied to those seven randomization p-values.
     Matched-seed sets are asserted EXACTLY (never a silent set
     intersection), and a missing seed BLOCKS. A canonical cross-seed
     summary (mean/std, `ddof=0`, every frozen seed present) is emitted for
     each of the six configurations and each of the eight metrics.

Structural guarantee, unchanged: this module holds no training capability.
Row metadata comes from the validated frozen V3 lockset, never a fresh
source-matrix resolution.

IMPLEMENTATION RECONCILIATION (no protocol-identity change): row metadata is
now built exclusively from the frozen lockset's own entries (no
`resolve_target_matrix` call survives in `_score`); the label reveal binds
two DISTINCT commits (`prediction_execution_code_commit` from the E1 lock,
`first_authorized_reveal_code_commit` freshly read at reveal time); every
per-row score artifact is wrapped in an identity envelope and validated
exactly-24/no-extras/per-file-hash; the final result additionally binds
`scoring_execution_code_commit`, `target_label_artifact_sha256`,
`target_feature_package_identity`, per-row artifact identities/hashes, and
an explicit `access_state`; scoring is crash-recoverable via a disposable
`.score_staging/<execution_id>/` namespace and a
`SCORE_PROMOTION_TRANSACTION_<id>.json` manifest, recovering with zero
rescoring and without reopening an already-revealed label a second time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from prism_fas.evaluation.contracts import stable_identity

EXIT_PASS, EXIT_BLOCKED, EXIT_USAGE = 0, 2, 3

DIAGNOSTICS_DIR = "reports/full/exploratory_target_v3"
RUN_DIR = "runs/exploratory_target_v3"
PREDICTION_PLAN_BINDING_PATH = f"{DIAGNOSTICS_DIR}/PREDICTION_PLAN_BINDING.json"
PREDICTION_LOCK_PATH = f"{DIAGNOSTICS_DIR}/TARGET_PREDICTION_LOCK.json"
LABEL_REVEAL_PATH = f"{DIAGNOSTICS_DIR}/TARGET_LABEL_REVEAL.json"
SCORE_RESULT_PATH = f"{DIAGNOSTICS_DIR}/EXPLORATORY_TARGET_SCORE_RESULT.json"
SCORE_ROWS_DIR = f"{DIAGNOSTICS_DIR}/scores"

EXPECTED_TOTAL_ROWS = 24
EXPECTED_ATOMIC_COMPARISONS = 7
CONFIGURATIONS: tuple[str, ...] = ("C-G-RND", "C-G-DET", "C-G-LLM", "C-R-DET", "C-R-LLM", "C-R-NOPROMPT")
CROSS_SEED_METRICS: tuple[str, ...] = ("apcer", "bpcer", "acer", "roc_auc", "eer", "ece", "brier", "nll")

REQUIRED_MATCHED_SEEDS: dict[str, tuple[int, ...]] = {
    "E-H1_RND_vs_DET": (20260806, 20260807, 20260808, 20260809, 20260810),
    "E-H1_RND_vs_LLM": (20260806, 20260807, 20260808, 20260809, 20260810),
    "E-H1_DET_vs_LLM": (20260806, 20260807, 20260808, 20260809, 20260810),
    "E-H2": (20260806, 20260807, 20260808),
    "E-H3": (20260806, 20260807, 20260808),
    "E-H4_DET": (20260806, 20260807, 20260808),
    "E-H4_LLM": (20260806, 20260807, 20260808),
}

FORBIDDEN_IMPORTS = ("torch", "prism_fas.detector.trainer", "prism_fas.detector.checkpoint",
                    "prism_fas.train.trainer", "prism_fas.train.b00_pipeline", "torch.optim",
                    "prism_fas.evaluation.synthetic_real_probe")


class ExploratoryScoringV3Error(RuntimeError):
    """The V3 exploratory scorer cannot proceed with the inputs given."""


def assert_no_training_capability() -> dict[str, Any]:
    from prism_fas.evaluation.scoring import static_import_audit

    audit = static_import_audit(Path(__file__))
    violations = sorted({name for name in audit["module_level_imports"]
                         for forbidden in FORBIDDEN_IMPORTS
                         if name == forbidden or name.startswith(forbidden + ".")})
    if violations:
        raise ExploratoryScoringV3Error(
            f"the V3 exploratory scorer must have no training/checkpoint-loading capability; "
            f"found {violations}")
    return {**audit, "forbidden_imports_checked": list(FORBIDDEN_IMPORTS), "violations": violations}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _row_meta_from_lockset(lockset: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Defect B: row metadata comes SOLELY from the validated, frozen
    lockset entries — never a fresh `resolve_target_matrix` call. After E1
    closes, the lockset (and the plan binding it is built from) is the sole
    scientific authority for row identity."""
    entries = lockset.get("entries") or {}
    return {row_id: {
        "row_id": row_id, "experiment_id": entry["experiment_id"], "track": entry["track"],
        "arm": entry["arm"], "seed": int(entry["seed"]),
        "prediction_variant_id": entry["prediction_variant_id"],
        "threshold": float(entry["threshold"]), "checkpoint_sha256": entry["checkpoint_sha256"],
        "calibration_hash": entry["calibration_hash"],
        "prediction_lock_identity": entry["prediction_lock_identity"],
        "prediction_logical_identity": entry["prediction_logical_identity"],
    } for row_id, entry in entries.items()}


def _current_scorer_git_commit(repo: Path) -> str:
    """Defect C: a FRESH git HEAD read at reveal/scoring-execution time, via
    a small local subprocess helper duplicated from
    `detector.checkpoint.git_commit` on purpose — that module is (and stays)
    forbidden in this scorer (see `FORBIDDEN_IMPORTS`) because importing it
    can pull in model/checkpoint machinery transitively."""
    import subprocess

    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo), capture_output=True,
                                text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def require_frozen_prediction_lockset(repo: Path) -> dict[str, Any]:
    from prism_fas.evaluation.post_failure_exploratory_target_v3 import (
        EXPECTED_TOTAL_ROWS as expected, validate_existing_exploratory_prediction_result_v3)

    validation = validate_existing_exploratory_prediction_result_v3(repo)
    if not validation["valid"]:
        raise ExploratoryScoringV3Error(f"the V3 prediction lockset failed validation: {validation['problems']}")
    lockset = validation["lockset"]
    if lockset.get("status") != "FROZEN":
        raise ExploratoryScoringV3Error(f"lockset status is {lockset.get('status')!r}, not FROZEN")
    if int(lockset.get("entry_count", -1)) != expected:
        raise ExploratoryScoringV3Error(f"lockset entry_count is {lockset.get('entry_count')}, expected {expected}")
    if lockset.get("target_labels_opened") is not False:
        raise ExploratoryScoringV3Error("lockset does not record target_labels_opened: false")
    return lockset


# ==============================================================================
# Label reveal — the real, one-way artifact (Defect D)
# ==============================================================================

def compute_label_artifact_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_label_reveal(*, protocol_id: str, plan_binding_identity: str, prediction_lock_identity: str,
                       label_relative_path: str, label_sha256: str, prediction_execution_code_commit: str,
                       first_authorized_reveal_code_commit: str) -> dict[str, Any]:
    """Defect C: two DISTINCT commits are bound — `prediction_execution_code_commit`
    (the E1 inference commit, taken verbatim from the frozen lockset) and
    `first_authorized_reveal_code_commit` (a FRESH read of the current git
    HEAD at reveal time — the E2 commit, never conflated with E1's)."""
    body = {"reveal_schema_version": "post-failure-exploratory-target-v3-label-reveal-v2",
           "protocol_identity": protocol_id, "prediction_plan_binding_identity": plan_binding_identity,
           "target_prediction_lock_identity": prediction_lock_identity,
           "target_label_artifact_relative_path": label_relative_path,
           "target_label_artifact_sha256": label_sha256,
           "prediction_execution_code_commit": prediction_execution_code_commit,
           "first_authorized_reveal_code_commit": first_authorized_reveal_code_commit,
           "target_labels_accessed": True, "one_way": True, "may_be_reset": False,
           "reason": "POST_FAILURE_EXPLORATORY_E2_SCORING",
           "access_state": {"target_feature_identity_accessed": True, "target_prediction_features_accessed": True,
                            "target_labels_accessed": True, "target_feature_access_count": 1,
                            "target_label_access_count": 1},
           "ba_sep_observed_verdict": "FAIL", "detector_reliability_overall": "FAILED",
           "post_failure_diagnostics_v2": "FAIL", "c9_original_confirmatory_path": "BLOCKED"}
    return {**body, "reveal_identity": stable_identity(body)}


def reveal_target_labels(repo: Path, *, label_path: Path) -> dict[str, Any]:
    """The real, one-way reveal sequence (Defect D, section 5): validate the
    full E1 lock, verify no model capability, verify the label artifact
    exists, hash it WITHOUT scoring, then atomically create the reveal —
    exact match reuses it (never re-incrementing `target_label_access_count`),
    any mismatch BLOCKS, it is never overwritten."""
    from prism_fas.evaluation.post_failure_exploratory_target_v3 import active_protocol_identity
    from prism_fas.pipeline.state import atomic_write_json

    assert_no_training_capability()
    lockset = require_frozen_prediction_lockset(repo)
    if not Path(label_path).is_file():
        raise ExploratoryScoringV3Error(f"the target label artifact does not exist: {label_path}")
    label_sha256 = compute_label_artifact_sha256(label_path)
    protocol_id = active_protocol_identity(repo)
    binding = _read_json(Path(repo) / PREDICTION_PLAN_BINDING_PATH) or {}
    reveal = build_label_reveal(
        protocol_id=protocol_id, plan_binding_identity=binding.get("prediction_plan_binding_identity", ""),
        prediction_lock_identity=lockset["lockset_identity"],
        label_relative_path=str(Path(label_path).relative_to(Path(repo))) if Path(label_path).is_absolute()
        else str(label_path),
        label_sha256=label_sha256,
        prediction_execution_code_commit=lockset.get("prediction_execution_code_commit", ""),
        first_authorized_reveal_code_commit=_current_scorer_git_commit(repo))

    reveal_path = Path(repo) / LABEL_REVEAL_PATH
    existing = _read_json(reveal_path)
    if existing is not None:
        if existing != reveal:
            raise ExploratoryScoringV3Error(
                "an existing TARGET_LABEL_REVEAL.json differs from the one just computed; "
                "refusing to overwrite a one-way reveal")
        return existing
    atomic_write_json(reveal_path, reveal)
    return reveal


# ==============================================================================
# Bootstrap CI (Defect G — CI only, never a p-value)
# ==============================================================================

def apcer_bpcer_acer(decisions: Mapping[str, str], *, live_ids: list[str], spoof_ids: list[str]) -> dict[str, float]:
    if not live_ids or not spoof_ids:
        raise ExploratoryScoringV3Error("ACER requires both non-empty LIVE and SPOOF populations")
    apcer = sum(1 for video_id in spoof_ids if decisions[video_id] != "spoof") / len(spoof_ids)
    bpcer = sum(1 for video_id in live_ids if decisions[video_id] == "spoof") / len(live_ids)
    return {"apcer": float(apcer), "bpcer": float(bpcer), "acer": float(0.5 * (apcer + bpcer))}


def class_stratified_bootstrap_ci(decisions_a_by_seed: Mapping[int, Mapping[str, str]],
                                  decisions_b_by_seed: Mapping[int, Mapping[str, str]],
                                  labels: Mapping[str, int], *, seed: int = 20260810,
                                  resamples: int = 10000, confidence_level: float = 0.95
                                  ) -> dict[str, Any]:
    """CI ONLY — never a p-value (Defect G). Every matched seed's decisions
    are kept separate; seeds are fixed replications, never resampled."""
    import numpy as np

    matched_seeds = sorted(set(decisions_a_by_seed) & set(decisions_b_by_seed))
    if not matched_seeds:
        raise ExploratoryScoringV3Error("no matched seed between the two compared arms")
    live_ids = sorted(v for v, l in labels.items() if l == 0)
    spoof_ids = sorted(v for v, l in labels.items() if l == 1)
    if not live_ids or not spoof_ids:
        raise ExploratoryScoringV3Error("labels must contain both LIVE and SPOOF videos")

    observed_deltas = []
    live_a, spoof_a, live_b, spoof_b = {}, {}, {}, {}
    for matched_seed in matched_seeds:
        da, db = decisions_a_by_seed[matched_seed], decisions_b_by_seed[matched_seed]
        metrics_a = apcer_bpcer_acer(da, live_ids=live_ids, spoof_ids=spoof_ids)
        metrics_b = apcer_bpcer_acer(db, live_ids=live_ids, spoof_ids=spoof_ids)
        observed_deltas.append(metrics_a["acer"] - metrics_b["acer"])
        live_a[matched_seed] = np.array([da[v] == "spoof" for v in live_ids])
        spoof_a[matched_seed] = np.array([da[v] != "spoof" for v in spoof_ids])
        live_b[matched_seed] = np.array([db[v] == "spoof" for v in live_ids])
        spoof_b[matched_seed] = np.array([db[v] != "spoof" for v in spoof_ids])
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
    return {"matched_seeds": matched_seeds, "observed_statistic": observed_statistic,
           "ci_lower": lower, "ci_upper": upper, "resamples": int(resamples), "seed": int(seed),
           "confidence_level": float(confidence_level), "paired": True, "class_stratified": True,
           "bootstrap_unit": "video", "used_for": "CI_ONLY"}


# ==============================================================================
# Paired randomization test (Defects G/H/I — the exploratory p-value)
# ==============================================================================

def paired_randomization_test(decisions_a_by_seed: Mapping[int, Mapping[str, str]],
                              decisions_b_by_seed: Mapping[int, Mapping[str, str]],
                              labels: Mapping[str, int], *, seed: int = 20260810,
                              resamples: int = 10000) -> dict[str, Any]:
    """The frozen paired video-level sign-flip randomization test (section
    9). Every video's contribution is averaged across ALL matched seeds
    first (linear, so sign-flipping the averaged contribution is
    mathematically identical to flipping the sign consistently across every
    seed for that video — the video is the paired observational unit).
    `p = (1 + count(|T_perm| >= |T_obs|)) / (1 + resamples)`, `>=` inclusive.
    """
    import numpy as np

    matched_seeds = sorted(set(decisions_a_by_seed) & set(decisions_b_by_seed))
    if not matched_seeds:
        raise ExploratoryScoringV3Error("no matched seed between the two compared arms")
    live_ids = sorted(v for v, l in labels.items() if l == 0)
    spoof_ids = sorted(v for v, l in labels.items() if l == 1)
    if not live_ids or not spoof_ids:
        raise ExploratoryScoringV3Error("labels must contain both LIVE and SPOOF videos")
    n_live, n_spoof = len(live_ids), len(spoof_ids)

    contributions: dict[str, float] = {}
    for video_id in live_ids:
        deltas = [(1.0 if decisions_a_by_seed[s][video_id] == "spoof" else 0.0)
                 - (1.0 if decisions_b_by_seed[s][video_id] == "spoof" else 0.0) for s in matched_seeds]
        contributions[video_id] = (sum(deltas) / len(matched_seeds)) * (1.0 / (2 * n_live))
    for video_id in spoof_ids:
        deltas = [(1.0 if decisions_a_by_seed[s][video_id] != "spoof" else 0.0)
                 - (1.0 if decisions_b_by_seed[s][video_id] != "spoof" else 0.0) for s in matched_seeds]
        contributions[video_id] = (sum(deltas) / len(matched_seeds)) * (1.0 / (2 * n_spoof))

    video_ids = sorted(contributions)
    values = np.array([contributions[v] for v in video_ids], dtype=np.float64)
    observed_statistic = float(values.sum())
    observed_abs = abs(observed_statistic)

    rng = np.random.Generator(np.random.PCG64(int(seed)))
    signs = rng.integers(0, 2, size=(int(resamples), len(values))).astype(np.float64) * 2.0 - 1.0
    permuted = np.abs((signs * values[None, :]).sum(axis=1))
    count_ge = int((permuted >= observed_abs).sum())
    p_value = float((1 + count_ge) / (1 + int(resamples)))

    return {"matched_seeds": matched_seeds, "observed_statistic": observed_statistic,
           "observed_absolute_statistic": observed_abs, "p_value_two_sided": p_value,
           "resamples": int(resamples), "seed": int(seed), "count_ge": count_ge,
           "tie_handling": ">= inclusive", "bootstrap_unit": "video", "used_for": "P_VALUE_ONLY"}


def holm_bonferroni(p_values: Mapping[str, float], *, alpha: float = 0.05) -> dict[str, dict[str, Any]]:
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
# Per-row scoring, cross-seed summaries, the seven atomic comparisons
# ==============================================================================

def score_one_row(repo: Path, row_id: str, *, labels: Any, run_root: Path,
                  row_meta_entry: Mapping[str, Any]) -> dict[str, Any]:
    """Defect F: the raw `scoring.score()` payload is wrapped in an
    identity-bearing envelope — row_id, the frozen per-row prediction lock
    and logical identities, checkpoint/calibration hashes, seed/track/arm/
    variant, and a self-hash (`score_artifact_identity`) computed over the
    whole envelope. The raw metrics live unchanged under `metrics`."""
    from prism_fas.evaluation.scoring import score
    from prism_fas.evaluation.target_prediction import PREDICTION_LOCK_FILE, read_predictions

    row_run_dir = Path(run_root) / row_id
    row_lock = _read_json(row_run_dir / PREDICTION_LOCK_FILE)
    if row_lock is None:
        raise ExploratoryScoringV3Error(f"{row_id}: no per-row PREDICTION_LOCK.json on disk")
    predictions = read_predictions(row_run_dir / "target_predictions.parquet")
    threshold = float(row_lock["aggregation"]["threshold"])
    metrics = score(predictions=predictions, lock=row_lock, labels=labels, threshold=threshold)
    body = {"row_id": row_id,
           "prediction_lock_identity": row_meta_entry["prediction_lock_identity"],
           "prediction_logical_identity": row_meta_entry["prediction_logical_identity"],
           "checkpoint_sha256": row_meta_entry["checkpoint_sha256"],
           "calibration_hash": row_meta_entry["calibration_hash"],
           "seed": int(row_meta_entry["seed"]), "track": row_meta_entry["track"],
           "arm": row_meta_entry["arm"], "prediction_variant_id": row_meta_entry["prediction_variant_id"],
           "metrics": metrics}
    body["score_artifact_identity"] = stable_identity(body)
    return body


def build_cross_seed_summary(scored_rows: Mapping[str, Mapping[str, Any]],
                             row_meta: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Section 12: for each of the 6 configurations and each of the 8
    metrics, the per-seed values plus `mean`/`std(ddof=0)` over the
    complete frozen seed family — NEVER predictions pooled across seeds."""
    import numpy as np

    summary: dict[str, Any] = {}
    for experiment_id in CONFIGURATIONS:
        rows_for_config = sorted((row_id for row_id, meta in row_meta.items()
                                 if meta["experiment_id"] == experiment_id),
                                 key=lambda row_id: row_meta[row_id]["seed"])
        if not rows_for_config:
            raise ExploratoryScoringV3Error(f"{experiment_id}: no scored rows found for cross-seed summary")
        per_metric: dict[str, Any] = {}
        for metric in CROSS_SEED_METRICS:
            per_seed = {}
            for row_id in rows_for_config:
                video = scored_rows[row_id]["metrics"]["video"]
                value = video[metric] if metric in video else video["calibration"][metric]
                per_seed[int(row_meta[row_id]["seed"])] = float(value)
            values = np.array(list(per_seed.values()), dtype=np.float64)
            per_metric[metric] = {"seeds": sorted(per_seed), "per_seed": per_seed,
                                  "mean": float(values.mean()), "std_ddof0": float(values.std(ddof=0))}
        summary[experiment_id] = per_metric
    return summary


def _by_seed_decisions(scored_rows: Mapping[str, Mapping[str, Any]],
                       row_meta: Mapping[str, Mapping[str, Any]], *, track: str,
                       arm: str | None = None, experiment_id: str | None = None
                       ) -> dict[int, Mapping[str, str]]:
    matches = [row_id for row_id, meta in row_meta.items()
              if meta["track"] == track and (arm is None or meta["arm"] == arm)
              and (experiment_id is None or meta["experiment_id"] == experiment_id)]
    return {int(row_meta[row_id]["seed"]): {row["video_id"]: row["decision"]
                                            for row in scored_rows[row_id]["metrics"]["video_scores"]}
           for row_id in matches}


def _assert_exact_matched_seeds(name: str, decisions_a: Mapping[int, Any], decisions_b: Mapping[int, Any]) -> list[int]:
    """Every REQUIRED seed must be present on BOTH sides — a side may
    legitimately carry additional seeds it needs for a different
    comparison (e.g. Track-G DET has 5 seeds for E-H1 but only 3 are
    required here for E-H4_DET); the caller filters to exactly the
    required set before computing anything. Only a genuinely MISSING
    required seed blocks — never a silent narrowing to whatever seeds
    happen to intersect."""
    required = set(REQUIRED_MATCHED_SEEDS[name])
    missing = required - (set(decisions_a) & set(decisions_b))
    if missing:
        raise ExploratoryScoringV3Error(
            f"{name}: missing required matched seed(s) {sorted(missing)}; refusing to silently drop a seed")
    return sorted(required)


def compute_exploratory_comparisons_v3(scored_rows: Mapping[str, Mapping[str, Any]],
                                       row_meta: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """All seven atomic E-H comparisons: exact matched-seed assertion,
    class-stratified bootstrap for CI, paired randomization for p-value."""
    labels_source = next(iter(scored_rows.values()))["metrics"]["video_scores"]
    labels = {row["video_id"]: int(row["label"]) for row in labels_source}

    pairs = {
        "E-H1_RND_vs_DET": (_by_seed_decisions(scored_rows, row_meta, track="G", arm="RND"),
                           _by_seed_decisions(scored_rows, row_meta, track="G", arm="DET")),
        "E-H1_RND_vs_LLM": (_by_seed_decisions(scored_rows, row_meta, track="G", arm="RND"),
                           _by_seed_decisions(scored_rows, row_meta, track="G", arm="LLM")),
        "E-H1_DET_vs_LLM": (_by_seed_decisions(scored_rows, row_meta, track="G", arm="DET"),
                           _by_seed_decisions(scored_rows, row_meta, track="G", arm="LLM")),
        "E-H2": (_by_seed_decisions(scored_rows, row_meta, track="R", experiment_id="C-R-DET"),
                _by_seed_decisions(scored_rows, row_meta, track="R", experiment_id="C-R-LLM")),
        "E-H3": (_by_seed_decisions(scored_rows, row_meta, track="R", experiment_id="C-R-LLM"),
                _by_seed_decisions(scored_rows, row_meta, track="R", experiment_id="C-R-NOPROMPT")),
        "E-H4_DET": (_by_seed_decisions(scored_rows, row_meta, track="G", arm="DET"),
                    _by_seed_decisions(scored_rows, row_meta, track="R", experiment_id="C-R-DET")),
        "E-H4_LLM": (_by_seed_decisions(scored_rows, row_meta, track="G", arm="LLM"),
                    _by_seed_decisions(scored_rows, row_meta, track="R", experiment_id="C-R-LLM")),
    }
    if set(pairs) != set(REQUIRED_MATCHED_SEEDS):
        raise ExploratoryScoringV3Error("the atomic comparison set does not match the frozen seven")

    comparisons: dict[str, Any] = {}
    for name, (decisions_a, decisions_b) in pairs.items():
        matched_seeds = _assert_exact_matched_seeds(name, decisions_a, decisions_b)
        filtered_a = {s: decisions_a[s] for s in matched_seeds}
        filtered_b = {s: decisions_b[s] for s in matched_seeds}
        ci = class_stratified_bootstrap_ci(filtered_a, filtered_b, labels)
        randomization = paired_randomization_test(filtered_a, filtered_b, labels)
        comparisons[name] = {"bootstrap_ci": ci, "randomization": randomization,
                            "matched_seeds": matched_seeds}

    if len(comparisons) != EXPECTED_ATOMIC_COMPARISONS:
        raise ExploratoryScoringV3Error(
            f"expected {EXPECTED_ATOMIC_COMPARISONS} atomic comparisons, computed {len(comparisons)}")
    p_values = {name: result["randomization"]["p_value_two_sided"] for name, result in comparisons.items()}
    return {"comparisons": comparisons, "holm_bonferroni": holm_bonferroni(p_values),
           "atomic_comparison_count": len(comparisons), "holm_input": "randomization_p_values"}


# ==============================================================================
# Score staging + crash-recoverable promotion (Defect H)
# ==============================================================================

SCORE_STAGING_ROOT = f"{DIAGNOSTICS_DIR}/.score_staging"
SCORE_PROMOTION_TRANSACTION_SCHEMA_VERSION = "post-failure-exploratory-target-v3-score-promotion-transaction-v1"


def score_execution_identity(*, prediction_lock_identity: str, scoring_execution_code_commit: str) -> str:
    """Deliberately independent of `label_reveal_identity`: this identity
    must be computable from the lockset and the current commit ALONE, with
    no label access, so the score-state checks (extra/partial/incomplete)
    can run — and a genuine transaction can be located — before any label
    byte is ever touched."""
    return stable_identity({"prediction_lock_identity": prediction_lock_identity,
                            "scoring_execution_code_commit": scoring_execution_code_commit})[:16]


def _score_promotion_transaction_path(repo: Path, execution_id: str) -> Path:
    return Path(repo) / SCORE_STAGING_ROOT / f"SCORE_PROMOTION_TRANSACTION_{execution_id}.json"


def _validate_score_file_against_transaction(path: Path, staged_record: Mapping[str, Any], row_id: str) -> None:
    if not path.is_file():
        raise ExploratoryScoringV3Error(f"{row_id}: score file missing at {path}")
    real_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if real_hash != staged_record["score_file_sha256"]:
        raise ExploratoryScoringV3Error(f"{row_id}: score file hash disagrees with the promotion transaction")


def promote_staged_score_rows(repo: Path, staging_root: Path, row_ids: Sequence[str], *,
                              prediction_lock_identity: str, label_reveal_identity: str,
                              scoring_execution_code_commit: str, execution_identity: str,
                              staged_hashes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Defect H, mirroring Defect A: a `SCORE_PROMOTION_TRANSACTION_<id>.json`
    manifest (state READY_TO_PROMOTE) is written BEFORE any row file is
    moved out of `.score_staging/<execution_id>/`, binding the exact row
    set and each staged file's SHA-256. Recovery validates already-promoted
    and still-staged rows against those hashes and resumes file-renames
    only — no rescoring, no label reopen."""
    from prism_fas.pipeline.state import atomic_write_json

    transaction_path = _score_promotion_transaction_path(repo, execution_identity)
    transaction = _read_json(transaction_path)
    if transaction is None:
        body = {"schema_version": SCORE_PROMOTION_TRANSACTION_SCHEMA_VERSION,
               "prediction_lock_identity": prediction_lock_identity,
               "label_reveal_identity": label_reveal_identity,
               "scoring_execution_code_commit": scoring_execution_code_commit,
               "execution_identity": execution_identity,
               "row_ids": sorted(row_ids),
               "staged_artifacts": {row_id: dict(staged_hashes[row_id]) for row_id in sorted(row_ids)}}
        body["transaction_identity"] = stable_identity(body)
        body["state"] = "READY_TO_PROMOTE"
        transaction = body
        atomic_write_json(transaction_path, transaction)
    elif transaction.get("state") not in ("READY_TO_PROMOTE", "COMPLETE"):
        raise ExploratoryScoringV3Error(f"unrecognized score promotion transaction state {transaction.get('state')!r}")
    elif sorted(transaction["row_ids"]) != sorted(row_ids):
        raise ExploratoryScoringV3Error(
            "an existing score promotion transaction does not match the requested row set")

    final_rows_dir = Path(repo) / SCORE_ROWS_DIR
    for row_id in transaction["row_ids"]:
        staged_record = transaction["staged_artifacts"][row_id]
        final_path = final_rows_dir / f"{row_id}.json"
        staging_path = Path(staging_root) / f"{row_id}.json"
        if final_path.is_file():
            _validate_score_file_against_transaction(final_path, staged_record, row_id)
            continue
        if not staging_path.is_file():
            raise ExploratoryScoringV3Error(f"{row_id}: neither staged nor promoted; score transaction cannot recover")
        _validate_score_file_against_transaction(staging_path, staged_record, row_id)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        staging_path.rename(final_path)

    transaction["state"] = "COMPLETE"
    atomic_write_json(transaction_path, transaction)
    return transaction


def _recover_scored_rows_from_disk(repo: Path, staging_root: Path, row_ids: Sequence[str]) -> dict[str, Any]:
    """Recovery-only: read each row's already-computed score envelope back
    from wherever it currently sits (promoted or still staged) — zero
    rescoring, zero label access."""
    final_rows_dir = Path(repo) / SCORE_ROWS_DIR
    scored_rows: dict[str, Any] = {}
    for row_id in row_ids:
        final_path = final_rows_dir / f"{row_id}.json"
        staging_path = Path(staging_root) / f"{row_id}.json"
        path = final_path if final_path.is_file() else staging_path
        body = _read_json(path)
        if body is None:
            raise ExploratoryScoringV3Error(f"{row_id}: cannot recover — no score file at {path}")
        scored_rows[row_id] = body
    return scored_rows


# ==============================================================================
# Existing-score-result validation (Defect F, section 7)
# ==============================================================================

def validate_existing_exploratory_score_result_v3(repo: Path) -> dict[str, Any]:
    problems: list[str] = []
    repo = Path(repo)
    result = _read_json(repo / SCORE_RESULT_PATH)
    if result is None:
        return {"valid": False, "problems": ["no EXPLORATORY_TARGET_SCORE_RESULT.json on disk"], "result": None}

    try:
        lockset = require_frozen_prediction_lockset(repo)
    except ExploratoryScoringV3Error as error:
        return {"valid": False, "problems": [f"prediction lockset no longer valid: {error}"], "result": result}

    reveal = _read_json(repo / LABEL_REVEAL_PATH)
    if reveal is None:
        problems.append("no TARGET_LABEL_REVEAL.json on disk")
    else:
        if result.get("label_reveal_identity") != reveal.get("reveal_identity"):
            problems.append("score result's label_reveal_identity does not match the recorded reveal")
        # Defect E: the label artifact itself must not have been tampered
        # with since the one-way reveal — this hashes CURRENT bytes only
        # because `reveal` already exists (post-reveal state); no code path
        # here touches label bytes before a reveal is on disk.
        label_relative_path = reveal.get("target_label_artifact_relative_path")
        label_path = (repo / label_relative_path) if label_relative_path else None
        if label_path is None or not label_path.is_file():
            problems.append("the target label artifact referenced by the reveal is missing or unresolvable")
        else:
            current_label_sha256 = compute_label_artifact_sha256(label_path)
            if current_label_sha256 != reveal.get("target_label_artifact_sha256"):
                problems.append("the target label artifact's current SHA-256 no longer matches the frozen "
                               "reveal — possible tampering")
        recomputed_reveal_identity = stable_identity(
            {key: value for key, value in reveal.items() if key != "reveal_identity"})
        if recomputed_reveal_identity != reveal.get("reveal_identity"):
            problems.append("the label reveal does not hash to its own recorded reveal_identity")

    if result.get("prediction_lock_identity") != lockset.get("lockset_identity"):
        problems.append("score result's prediction_lock_identity does not match the current lockset")
    if result.get("prediction_execution_code_commit") != lockset.get("prediction_execution_code_commit"):
        problems.append("score result's prediction_execution_code_commit does not match the lockset")
    if not result.get("scoring_execution_code_commit"):
        problems.append("score result does not record a scoring_execution_code_commit")
    if result.get("target_feature_package_identity") != lockset.get("target_feature_package_identity"):
        problems.append("score result's target_feature_package_identity does not match the lockset")
    if reveal is not None and result.get("target_label_artifact_sha256") != reveal.get("target_label_artifact_sha256"):
        problems.append("score result's target_label_artifact_sha256 does not match the reveal")
    if int(result.get("row_count", -1)) != EXPECTED_TOTAL_ROWS:
        problems.append(f"score result row_count is {result.get('row_count')}, expected {EXPECTED_TOTAL_ROWS}")
    rows = dict(result.get("rows") or {})
    lockset_entries = dict(lockset.get("entries") or {})
    if set(rows) != set(lockset_entries):
        problems.append("score result row_ids do not exactly match the lockset's row_ids")

    # Defect F: exactly 24 per-row score files, no extras, current hash and
    # internal identity of each, cross-checked against the frozen lockset
    # entry and against the final result's own references.
    rows_dir = repo / SCORE_ROWS_DIR
    existing_files = {p.stem: p for p in rows_dir.glob("*.json")} if rows_dir.is_dir() else {}
    expected_row_ids = set(lockset_entries)
    extra_files = set(existing_files) - expected_row_ids
    if extra_files:
        problems.append(f"unexpected per-row score files not in the frozen row set: {sorted(extra_files)}")
    missing_files = expected_row_ids - set(existing_files)
    if missing_files:
        problems.append(f"missing per-row score files: {sorted(missing_files)}")
    per_row_score_artifacts = dict(result.get("per_row_score_artifacts") or {})
    if set(per_row_score_artifacts) != expected_row_ids:
        problems.append("result.per_row_score_artifacts does not exactly cover the frozen row set")
    for row_id in sorted(expected_row_ids & set(existing_files)):
        path = existing_files[row_id]
        real_file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        recorded = per_row_score_artifacts.get(row_id, {})
        if real_file_sha256 != recorded.get("score_file_sha256"):
            problems.append(f"{row_id}: per-row score file sha256 no longer matches the frozen result "
                           "(possible tampering)")
        body = _read_json(path)
        if body is None:
            problems.append(f"{row_id}: per-row score file failed to parse")
            continue
        recomputed_artifact_identity = stable_identity(
            {key: value for key, value in body.items() if key != "score_artifact_identity"})
        if recomputed_artifact_identity != body.get("score_artifact_identity"):
            problems.append(f"{row_id}: per-row score file does not hash to its own recorded score_artifact_identity")
        if recorded.get("score_artifact_identity") != body.get("score_artifact_identity"):
            problems.append(f"{row_id}: result's recorded score_artifact_identity disagrees with the file")
        lock_entry = lockset_entries.get(row_id, {})
        if body.get("prediction_lock_identity") != lock_entry.get("prediction_lock_identity"):
            problems.append(f"{row_id}: per-row prediction_lock_identity disagrees with the frozen lockset entry")
        if body.get("prediction_logical_identity") != lock_entry.get("prediction_logical_identity"):
            problems.append(f"{row_id}: per-row prediction_logical_identity disagrees with the frozen lockset entry")
        if int(body.get("seed", -1)) != int(lock_entry.get("seed", -2)):
            problems.append(f"{row_id}: per-row seed disagrees with the frozen lockset entry")
        if body.get("track") != lock_entry.get("track") or body.get("arm") != lock_entry.get("arm"):
            problems.append(f"{row_id}: per-row track/arm disagrees with the frozen lockset entry")
        if body.get("prediction_variant_id") != lock_entry.get("prediction_variant_id"):
            problems.append(f"{row_id}: per-row prediction_variant_id disagrees with the frozen lockset entry")
        if row_id in rows and rows[row_id] != body:
            problems.append(f"{row_id}: result.rows entry disagrees with the current per-row score file")

    comparisons = dict((result.get("exploratory_comparisons") or {}).get("comparisons") or {})
    if len(comparisons) != EXPECTED_ATOMIC_COMPARISONS:
        problems.append(f"score result has {len(comparisons)} comparisons, expected {EXPECTED_ATOMIC_COMPARISONS}")
    for name, expected_seeds in REQUIRED_MATCHED_SEEDS.items():
        entry = comparisons.get(name) or {}
        if sorted(entry.get("matched_seeds") or []) != sorted(expected_seeds):
            problems.append(f"{name}: recorded matched_seeds does not match the frozen requirement")

    recorded_p_values = {name: entry.get("randomization", {}).get("p_value_two_sided")
                         for name, entry in comparisons.items()}
    recomputed_holm = holm_bonferroni({k: v for k, v in recorded_p_values.items() if v is not None})
    stored_holm = dict((result.get("exploratory_comparisons") or {}).get("holm_bonferroni") or {})
    if recomputed_holm != stored_holm:
        problems.append("Holm-Bonferroni recomputed from the RECORDED randomization p-values does not "
                        "match the stored correction")

    cross_seed = dict(result.get("cross_seed_summary") or {})
    if set(cross_seed) != set(CONFIGURATIONS):
        problems.append("cross_seed_summary does not cover exactly the six frozen configurations")

    expected_access_state = {"target_feature_identity_accessed": True, "target_prediction_features_accessed": True,
                             "target_labels_accessed": True, "target_feature_access_count": 1,
                             "target_label_access_count": 1}
    if result.get("access_state") != expected_access_state:
        problems.append(f"result.access_state is {result.get('access_state')!r}, expected {expected_access_state!r}")

    for field, expected in (("ba_sep_observed_verdict", "FAIL"),
                            ("detector_reliability_lock_c_observed_overall", "FAILED"),
                            ("post_failure_diagnostics_v2", "FAIL"),
                            ("c9_original_confirmatory_path", "BLOCKED"),
                            ("c9_may_close", False)):
        if result.get(field) != expected:
            problems.append(f"result.{field} is not {expected!r}")

    recomputed_identity = stable_identity(
        {key: value for key, value in result.items() if key != "score_result_identity"})
    if recomputed_identity != result.get("score_result_identity"):
        problems.append("score result does not hash to its own recorded score_result_identity")

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
    report: dict[str, Any] = {"prediction_lockset_valid": False, "checkpoint_weights_loaded": False,
                              "model_loaded": False}
    try:
        assert_no_training_capability()
        report["no_training_capability_verified"] = True
    except ExploratoryScoringV3Error as error:
        report["error"] = str(error)
        return EXIT_BLOCKED, report

    try:
        lockset = require_frozen_prediction_lockset(repo)
        report["prediction_lockset_valid"] = True
        report["prediction_lockset_entry_count"] = lockset.get("entry_count")
    except ExploratoryScoringV3Error as error:
        report["error"] = str(error)
        return EXIT_BLOCKED, report

    report["label_reveal_exists"] = (Path(repo) / LABEL_REVEAL_PATH).is_file()
    report["score_result_exists"] = (Path(repo) / SCORE_RESULT_PATH).is_file()
    return EXIT_PASS, report


def _score(repo: Path) -> tuple[int, dict[str, Any]]:
    """Phase E2 execution. NEVER invoked on this laptop for real.

    Defect B: row metadata comes solely from the frozen lockset entries.
    Defect H: all 24 rows are scored into a disposable
    `.score_staging/<execution_id>/` namespace first; a
    `SCORE_PROMOTION_TRANSACTION_<id>.json` manifest is written before any
    row file is promoted; a crash mid-promotion recovers by validating
    already-promoted and still-staged rows against that manifest and
    resuming file-renames only — no rescoring, no second label reopen (the
    existing, already-revealed `TARGET_LABEL_REVEAL.json` is read directly
    rather than re-derived through `reveal_target_labels`)."""
    from prism_fas.pipeline.state import atomic_write_json

    report: dict[str, Any] = {"scored": False, "checkpoint_weights_loaded": False, "model_loaded": False}
    result_path = Path(repo) / SCORE_RESULT_PATH
    if result_path.is_file():
        validation = validate_existing_exploratory_score_result_v3(repo)
        if not validation["valid"]:
            report.update({"error": "EXISTING_RESULT_FAILED_VALIDATION", "problems": validation["problems"]})
            return EXIT_BLOCKED, report
        report.update({"scored": True, "reused_existing_score_result": True, "labels_reopened": False})
        return EXIT_PASS, report

    try:
        assert_no_training_capability()
        lockset = require_frozen_prediction_lockset(repo)
    except ExploratoryScoringV3Error as error:
        report["error"] = str(error)
        return EXIT_BLOCKED, report

    row_meta = _row_meta_from_lockset(lockset)
    expected_rows = int(lockset["entry_count"])
    if len(row_meta) != expected_rows:
        report["error"] = f"lockset row_meta has {len(row_meta)} entries, expected {expected_rows}"
        return EXIT_BLOCKED, report

    # `execution_id` depends only on the lockset and the current commit —
    # NEVER on the label reveal — so a transaction can be located, and the
    # score-state checks below can run, with zero label access.
    existing_reveal = _read_json(Path(repo) / LABEL_REVEAL_PATH)
    scoring_execution_code_commit = _current_scorer_git_commit(repo)
    execution_id = score_execution_identity(
        prediction_lock_identity=lockset["lockset_identity"], scoring_execution_code_commit=scoring_execution_code_commit)
    staging_root = Path(repo) / SCORE_STAGING_ROOT / execution_id
    transaction = _read_json(_score_promotion_transaction_path(repo, execution_id))
    recovered_from_score_promotion_transaction = transaction is not None

    final_rows_dir = Path(repo) / SCORE_ROWS_DIR
    if transaction is None:
        existing_row_files = sorted(final_rows_dir.glob("*.json")) if final_rows_dir.is_dir() else []
        extra = {p.stem for p in existing_row_files} - set(row_meta)
        if extra:
            report.update({"error": "UNEXPECTED_SCORE_ROW_FILES", "extra": sorted(extra)})
            return EXIT_BLOCKED, report
        if 0 < len(existing_row_files) < expected_rows:
            report.update({"error": "PARTIAL_SCIENTIFIC_RESULT_SET",
                          "present": len(existing_row_files), "expected": expected_rows})
            return EXIT_BLOCKED, report
        if len(existing_row_files) == expected_rows:
            report.update({"error": "INCOMPLETE_FINALIZATION",
                          "detail": "all row score files exist but the final result is absent; "
                                   "refusing to reopen labels or rewrite rows"})
            return EXIT_BLOCKED, report

    staged_hashes: dict[str, dict[str, Any]] = {}
    try:
        if recovered_from_score_promotion_transaction:
            # Recovery: the reveal MUST already exist (it always precedes
            # any staging) — read it back, never re-derive or re-hash it.
            reveal = existing_reveal
            if reveal is None:
                raise ExploratoryScoringV3Error(
                    "a score promotion transaction exists but no TARGET_LABEL_REVEAL.json is on disk")
            scored_rows = _recover_scored_rows_from_disk(repo, staging_root, transaction["row_ids"])
        else:
            from prism_fas.evaluation.post_failure_exploratory_target_v3 import load_protocol
            from prism_fas.evaluation.scoring import load_evaluation_labels

            protocol = load_protocol(repo)
            firewall = _build_firewall(repo, protocol)
            label_path = (Path(repo) / protocol["target_label_root"]["path"]
                         / protocol["target_label_root"]["artifact"])
            reveal = existing_reveal if existing_reveal is not None else reveal_target_labels(
                repo, label_path=label_path)
            labels = load_evaluation_labels(label_path, firewall=firewall, stage="G8")
            run_root = Path(repo) / RUN_DIR
            scored_rows = {}
            for row_id in sorted(row_meta):
                body = score_one_row(repo, row_id, labels=labels, run_root=run_root,
                                     row_meta_entry=row_meta[row_id])
                staging_path = staging_root / f"{row_id}.json"
                staging_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_json(staging_path, body)
                on_disk = _read_json(staging_path)
                if on_disk is None:
                    raise ExploratoryScoringV3Error(f"{row_id}: staged score file failed to write/parse")
                recomputed = stable_identity({key: value for key, value in on_disk.items()
                                             if key != "score_artifact_identity"})
                if recomputed != on_disk.get("score_artifact_identity"):
                    raise ExploratoryScoringV3Error(f"{row_id}: staged score file failed self-identity validation")
                staged_hashes[row_id] = {"score_file_sha256": hashlib.sha256(staging_path.read_bytes()).hexdigest(),
                                        "score_artifact_identity": body["score_artifact_identity"]}
                scored_rows[row_id] = body
    except Exception as error:                            # noqa: BLE001
        report["error"] = f"{type(error).__name__}: {error}"
        return EXIT_BLOCKED, report

    try:
        comparisons = compute_exploratory_comparisons_v3(scored_rows, row_meta)
        cross_seed_summary = build_cross_seed_summary(scored_rows, row_meta)
    except ExploratoryScoringV3Error as error:
        report["error"] = f"comparisons/summary could not be computed: {error}"
        return EXIT_BLOCKED, report

    try:
        transaction = promote_staged_score_rows(
            repo, staging_root, sorted(scored_rows),
            prediction_lock_identity=lockset["lockset_identity"], label_reveal_identity=reveal["reveal_identity"],
            scoring_execution_code_commit=scoring_execution_code_commit, execution_identity=execution_id,
            staged_hashes=staged_hashes or transaction["staged_artifacts"])
        if staging_root.is_dir() and not any(staging_root.iterdir()):
            staging_root.rmdir()
    except Exception as error:                            # noqa: BLE001
        report["error"] = f"{type(error).__name__}: {error}"
        return EXIT_BLOCKED, report

    per_row_score_artifacts = {
        row_id: {"score_artifact_identity": scored_rows[row_id]["score_artifact_identity"],
                "score_file_sha256": transaction["staged_artifacts"][row_id]["score_file_sha256"]}
        for row_id in sorted(scored_rows)}
    access_state = {"target_feature_identity_accessed": True, "target_prediction_features_accessed": True,
                    "target_labels_accessed": True, "target_feature_access_count": 1,
                    "target_label_access_count": 1}

    combined = {"schema_version": "post-failure-exploratory-target-v3-score-result-v2",
               "prediction_lock_identity": lockset["lockset_identity"],
               "prediction_execution_code_commit": lockset["prediction_execution_code_commit"],
               "scoring_execution_code_commit": scoring_execution_code_commit,
               "label_reveal_identity": reveal["reveal_identity"],
               "target_label_artifact_sha256": reveal["target_label_artifact_sha256"],
               "target_feature_package_identity": lockset["target_feature_package_identity"],
               "row_count": len(scored_rows), "rows": scored_rows,
               "per_row_score_artifacts": per_row_score_artifacts,
               "cross_seed_summary": cross_seed_summary,
               "exploratory_comparisons": comparisons,
               "target_labels_opened": True, "c9_may_close": False,
               "access_state": access_state,
               "ba_sep_observed_verdict": "FAIL",
               "detector_reliability_lock_c_observed_overall": "FAILED",
               "post_failure_diagnostics_v2": "FAIL",
               "c9_original_confirmatory_path": "BLOCKED",
               "exploratory_target_status": "POST_FAILURE_EXPLORATORY"}
    combined["score_result_identity"] = stable_identity(
        {key: value for key, value in combined.items() if key != "score_result_identity"})
    atomic_write_json(result_path, combined)
    report.update({"scored": True, "reused_existing_score_result": False, "row_count": len(scored_rows),
                  "atomic_comparison_count": comparisons["atomic_comparison_count"],
                  "score_result_path": SCORE_RESULT_PATH, "access_state": access_state,
                  "recovered_from_score_promotion_transaction": recovered_from_score_promotion_transaction,
                  "labels_reopened": not recovered_from_score_promotion_transaction and existing_reveal is None})
    return EXIT_PASS, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m prism_fas.evaluation.post_failure_exploratory_target_v3_scorer",
        description="POST_FAILURE_EXPLORATORY_TARGET_V3 — Phase E2 label-unlock scoring, "
                    "final pre-target hardened. Holds no training or checkpoint-loading capability.")
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


__all__ = ["DIAGNOSTICS_DIR", "PREDICTION_LOCK_PATH", "LABEL_REVEAL_PATH", "SCORE_RESULT_PATH",
           "SCORE_ROWS_DIR", "SCORE_STAGING_ROOT", "EXPECTED_TOTAL_ROWS", "EXPECTED_ATOMIC_COMPARISONS",
           "CONFIGURATIONS", "CROSS_SEED_METRICS", "REQUIRED_MATCHED_SEEDS", "FORBIDDEN_IMPORTS",
           "SCORE_PROMOTION_TRANSACTION_SCHEMA_VERSION",
           "ExploratoryScoringV3Error", "assert_no_training_capability", "require_frozen_prediction_lockset",
           "_row_meta_from_lockset", "_current_scorer_git_commit",
           "compute_label_artifact_sha256", "build_label_reveal", "reveal_target_labels",
           "apcer_bpcer_acer", "class_stratified_bootstrap_ci", "paired_randomization_test",
           "holm_bonferroni", "score_one_row", "build_cross_seed_summary",
           "compute_exploratory_comparisons_v3", "validate_existing_exploratory_score_result_v3",
           "score_execution_identity", "promote_staged_score_rows",
           "EXIT_PASS", "EXIT_BLOCKED", "EXIT_USAGE"]
