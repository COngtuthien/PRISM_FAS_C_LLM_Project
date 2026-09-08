"""PRISM-FAS-C EXT-Q1Q2 -- E8 G8: the isolated target scorer, plus arm
aggregation/paired comparison/closure for the 15 completed E8 q-matched runs.

Structural guarantees, proven from THIS file's own AST (never a comment,
mirroring ``prism_fas.evaluation.scoring``'s own self-audit exactly):

* No training capability. This module holds no optimizer/scheduler/scaler/
  model/trainer and never imports torch or ``prism_fas.detector.trainer`` at
  module level (``assert_g8_scorer_has_no_training_capability`` proves this
  from the file's own import graph, not from a comment).
* No unlocked scoring. The frozen 15-row ``E8_TARGET_PREDICTION_LOCKSET.json``
  is fully validated (``c_ext_e8_target_evaluation.require_valid_lockset_for_g8``)
  BEFORE a single prediction lock -- let alone a label -- is read.
* Every write goes through ``TargetLabelFirewall.check_write("G8", ...)``,
  which structurally refuses model-state patterns regardless of destination.
* Reuses ``prism_fas.evaluation.scoring.score`` UNCHANGED for every run's
  metrics (APCER/BPCER/ACER/ROC-AUC/EER already come from ``core_metrics``;
  ECE/Brier/NLL from ``calibration_metrics`` -- both already inside
  ``score()``'s ``video`` block); reuses
  ``prism_fas.evaluation.bootstrap.paired_bootstrap``/``holm_bonferroni``
  for the paired same-seed comparisons rather than inventing a new
  significance test.

Important disclosure (Part C): the project-level
``configs/evaluation/m10_target.yaml`` already declares
``target_labels_revealed: true`` -- target labels were revealed HISTORICALLY,
before E8 existed (see ``HISTORICAL_REVEAL_ARTIFACT_RELATIVE_PATH``). E8
training, q-matching, membership selection, calibration correction and G7
prediction never use target labels; this module's G8 pass is SCORING USE of
an already-revealed label artifact, never a first-ever blind reveal. If the
historical reveal artifact cannot be found, ``build_target_label_use_record``
fails closed with an explicit ``PROVENANCE_GAP`` status rather than
inventing a historical event.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prism_fas.evaluation import c_ext_common as cc  # noqa: E402
from prism_fas.evaluation import c_ext_e8_calibration_authority as auth  # noqa: E402
from prism_fas.evaluation import c_ext_e8_target_evaluation as g7  # noqa: E402
from prism_fas.evaluation import c_ext_e8_training_runner as runner  # noqa: E402
from prism_fas.evaluation import scoring  # noqa: E402
from prism_fas.evaluation.bootstrap import BootstrapSettings, holm_bonferroni, paired_bootstrap  # noqa: E402
from prism_fas.evaluation.target_prediction import read_predictions  # noqa: E402

EVIDENCE_ROOT_RELATIVE = "reports/c_ext_q1q2_v1/e8_qmatched/target_eval_v1"
LABEL_USE_RECORD_PATH = f"{EVIDENCE_ROOT_RELATIVE}/E8_TARGET_LABEL_USE_RECORD.json"
SCORE_RESULT_PATH = f"{EVIDENCE_ROOT_RELATIVE}/E8_TARGET_SCORE_RESULT.json"
PER_RUN_CSV_PATH = f"{EVIDENCE_ROOT_RELATIVE}/E8_TARGET_PER_RUN.csv"
ARM_SUMMARY_CSV_PATH = f"{EVIDENCE_ROOT_RELATIVE}/E8_TARGET_ARM_SUMMARY.csv"
PAIRED_COMPARISONS_PATH = f"{EVIDENCE_ROOT_RELATIVE}/E8_TARGET_PAIRED_COMPARISONS.json"
CLOSURE_PATH = f"{EVIDENCE_ROOT_RELATIVE}/E8_TARGET_EVALUATION_CLOSURE.json"
EVIDENCE_SHA256_PATH = f"{EVIDENCE_ROOT_RELATIVE}/E8_TARGET_EVALUATION_EVIDENCE.sha256"
SUMMARY_MD_PATH = f"{EVIDENCE_ROOT_RELATIVE}/E8_TARGET_EVALUATION_SUMMARY.md"

#: The historical, already-authorized target-label reveal that governs the
#: WHOLE project. E8 discloses, never repeats, this event.
HISTORICAL_REVEAL_ARTIFACT_RELATIVE_PATH = "reports/full/exploratory_target_v3/TARGET_LABEL_REVEAL.json"
PROJECT_TARGET_CONFIG_RELATIVE_PATH = "configs/evaluation/m10_target.yaml"

REQUIRED_METRICS = ("apcer", "bpcer", "acer", "roc_auc", "eer", "ece", "brier", "nll")
ARM_PAIRS = (("RND", "DET"), ("RND", "LLM"), ("DET", "LLM"))


class E8TargetScoringError(RuntimeError):
    """G8 refuses to proceed."""


def _repo_root(root: Path | None) -> Path:
    return root or cc.repo_root()


# --------------------------------------------------------------------------- #
# No-training-capability self-audit (mirrors scoring.assert_no_training_capability,
# parametrized over THIS file since scoring.import_closure_audit is hardcoded
# to its own module path)
# --------------------------------------------------------------------------- #

_SELF_DOTTED = "prism_fas.evaluation.c_ext_e8_target_scoring"


def import_closure_audit_for_this_module() -> dict[str, Any]:
    start = Path(__file__)
    seen: dict[str, Path] = {_SELF_DOTTED: start}
    audits: list[dict[str, Any]] = []
    pending = [(_SELF_DOTTED, start)]
    while pending:
        dotted, path = pending.pop()
        audit = scoring.static_import_audit(path, dotted=dotted)
        audits.append({**audit, "dotted": dotted})
        for name in audit["module_level_imports"]:
            target = scoring._module_file(name)  # noqa: SLF001 -- same-package reuse, no public equivalent
            if target is None or name in seen:
                continue
            package = name.rsplit(".", 1)[0]
            if package.startswith("prism_fas.") and package not in seen:
                init = scoring._module_file(package)  # noqa: SLF001
                if init is not None:
                    seen[package] = init
                    pending.append((package, init))
            seen[name] = target
            pending.append((name, target))
    failed = [audit for audit in audits if not audit["passed"]]
    return {"modules_audited": sorted(audit["dotted"] for audit in audits), "module_count": len(audits),
            "passed": not failed, "failures": failed}


def assert_g8_scorer_has_no_training_capability() -> dict[str, Any]:
    audit = scoring.static_import_audit(Path(__file__), dotted=_SELF_DOTTED)
    closure = import_closure_audit_for_this_module()
    if not closure["passed"]:
        raise E8TargetScoringError(f"importing the E8 G8 scorer would pull in a training runtime: "
                                   f"{closure['failures']}")
    if not audit["passed"]:
        raise E8TargetScoringError(f"the E8 G8 scorer must have no training capability; found "
                                   f"{audit['forbidden_imports']}")
    globals_here = set(globals())
    forbidden_symbols = sorted(name for name in globals_here
                               if name.lower() in {"optimizer", "scheduler", "scaler", "trainer",
                                                   "model", "save_checkpoint"})
    if forbidden_symbols:
        raise E8TargetScoringError(f"the E8 G8 scorer holds training symbols: {forbidden_symbols}")
    return {**audit, "import_closure": closure, "holds_optimizer": False, "holds_model": False,
            "can_save_checkpoint": False}


# --------------------------------------------------------------------------- #
# Part C: target label use record (disclosure, never a fabricated reveal)
# --------------------------------------------------------------------------- #

def build_target_label_use_record(
    repo: Path, *, lockset: dict[str, Any], scoring_code_commit: str = "",
    reveal_artifact_relative_path: str = HISTORICAL_REVEAL_ARTIFACT_RELATIVE_PATH,
) -> dict[str, Any]:
    """Binds the E8 prediction lockset to the historical, already-authorized
    label reveal. Never opens the label artifact itself -- the label's own
    SHA256 is copied verbatim from the historical reveal record, never
    recomputed here. Fails closed (``PROVENANCE_GAP``) if the project
    declares ``target_labels_revealed: true`` but no historical reveal
    artifact can be found, rather than inventing a historical event.
    """
    lockset_identity = lockset.get("lockset_identity") or lockset.get("lock_identity")
    reveal_path = repo / reveal_artifact_relative_path
    if not reveal_path.is_file():
        return {
            "schema_version": "e8-target-label-use-record-v1",
            "status": "PROVENANCE_GAP",
            "reason": ("the project config declares target_labels_revealed: true but no historical "
                      f"reveal artifact was found at {reveal_artifact_relative_path!r}; refusing to "
                      "invent a historical reveal event"),
            "e8_prediction_lockset_identity": lockset_identity,
            "scoring_code_commit": scoring_code_commit,
            "is_first_ever_blind_reveal": False,
            "e8_uses_target_labels_in_training_or_g7": False,
        }
    reveal = cc.read_json(reveal_path)
    return {
        "schema_version": "e8-target-label-use-record-v1",
        "status": "BOUND",
        "historical_reveal_artifact_path": reveal_artifact_relative_path,
        "historical_reveal_artifact_sha256": cc.sha256_file(reveal_path),
        "historical_reveal_identity": reveal.get("reveal_identity"),
        "historical_first_authorized_reveal_code_commit": reveal.get("first_authorized_reveal_code_commit"),
        "target_label_artifact_relative_path": reveal.get("target_label_artifact_relative_path"),
        "target_label_artifact_sha256": reveal.get("target_label_artifact_sha256"),
        "e8_prediction_lockset_identity": lockset_identity,
        "scoring_code_commit": scoring_code_commit,
        "statement": (
            "E8 training, q-matching, membership selection, calibration correction and G7 prediction "
            "never use target labels. Target labels were already revealed historically in the larger "
            "project (see historical_reveal_artifact_path) BEFORE this E8 scoring pass. This E8 target "
            "evaluation is E8 SCORING USE of an already-revealed label artifact, NOT a first-ever "
            "blind target reveal. G8 remains structurally isolated from training and from G7."
        ),
        "is_first_ever_blind_reveal": False,
        "e8_uses_target_labels_in_training_or_g7": False,
        "e8_g8_structurally_isolated_from_training_and_g7": True,
    }


# --------------------------------------------------------------------------- #
# Per-run scoring (reuses prism_fas.evaluation.scoring.score UNCHANGED)
# --------------------------------------------------------------------------- #

def score_run(*, repo: Path, run_id: str, lockset_entry: dict[str, Any], calibration_row: dict[str, Any],
             labels: Any, firewall: Any | None = None) -> dict[str, Any]:
    assert_g8_scorer_has_no_training_capability()
    run_dir = repo / g7.PREDICTION_RUN_ROOT / run_id
    predictions = read_predictions(run_dir / "target_predictions.parquet")
    lock = cc.read_json(run_dir / "PREDICTION_LOCK.json")
    return scoring.score(
        predictions=predictions, lock=lock, labels=labels,
        threshold=float(calibration_row["selected_threshold"]), unknown_threshold=None,
        expected={
            "checkpoint_sha256": lockset_entry["best_checkpoint_sha256"],
            "source_calibration_sha256": lockset_entry["calibration_sha256"],
            "calibration_hash": lockset_entry["calibration_hash"],
            "target_feature_package_identity": g7.TARGET_FEATURE_PACKAGE_IDENTITY,
        })


def score_all_e8_runs(repo: Path, *, labels: Any, firewall: Any | None = None) -> dict[str, dict[str, Any]]:
    """Validates the complete frozen 15-row lockset BEFORE a single label is
    read, then scores every run. Refuses (raises) on an incomplete matrix."""
    assert_g8_scorer_has_no_training_capability()
    lockset = g7.require_valid_lockset_for_g8(repo)
    cal_lock = auth.load_calibration_authority_lock_if_usable(repo)
    if cal_lock is None:
        raise E8TargetScoringError("no valid calibration authority lock found; G8 may not score")
    cal_by_run = {row["run_id"]: row for row in cal_lock["rows"]}
    firewall = firewall or g7.build_target_firewall(repo)
    results: dict[str, dict[str, Any]] = {}
    for entry in lockset["entries"]:
        run_id = entry["run_id"]
        if run_id not in cal_by_run:
            raise E8TargetScoringError(f"{run_id}: no calibration authority row -- refusing to score")
        results[run_id] = score_run(repo=repo, run_id=run_id, lockset_entry=entry,
                                    calibration_row=cal_by_run[run_id], labels=labels, firewall=firewall)
    return results


# --------------------------------------------------------------------------- #
# Part E: aggregation, paired comparisons, closure
# --------------------------------------------------------------------------- #

def build_per_run_table(results_by_run: dict[str, dict[str, Any]], lockset: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for entry in lockset["entries"]:
        run_id = entry["run_id"]
        result = results_by_run[run_id]
        video = result["video"]
        rows.append({
            "run_id": run_id, "arm": entry["arm"], "seed": entry["seed"],
            "apcer": float(video["apcer"]), "bpcer": float(video["bpcer"]), "acer": float(video["acer"]),
            "roc_auc": float(video["roc_auc"]), "eer": float(video["eer"]),
            "ece": float(video["calibration"]["ece"]), "brier": float(video["calibration"]["brier"]),
            "nll": float(video["calibration"]["nll"]), "threshold": float(result["threshold"]),
        })
    return sorted(rows, key=lambda row: row["run_id"])


def aggregate_arm(per_run_rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    import numpy as np

    rows = [row for row in per_run_rows if row["arm"] == arm]
    if len(rows) != 5:
        raise E8TargetScoringError(f"{arm}: expected exactly 5 seed rows for aggregation, got {len(rows)}")
    out: dict[str, Any] = {"arm": arm, "n": len(rows)}
    for metric in REQUIRED_METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        out[f"{metric}_mean"] = float(values.mean())
        out[f"{metric}_std"] = float(values.std(ddof=0))
    return out


def build_arm_summary(per_run_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {arm: aggregate_arm(per_run_rows, arm) for arm in runner.ARMS}


def paired_same_seed_comparisons(results_by_run: dict[str, dict[str, Any]],
                                 per_run_rows: list[dict[str, Any]],
                                 settings: BootstrapSettings | None = None) -> dict[str, Any]:
    """Reuses ``bootstrap.paired_bootstrap`` (frozen video-level paired
    ACER-difference bootstrap) for each same-seed arm pair -- the 1700
    target videos ARE the shared, paired population for two runs at the same
    seed. Never invents a new post-hoc test."""
    import numpy as np

    by_arm_seed = {(row["arm"], row["seed"]): row["run_id"] for row in per_run_rows}
    threshold_by_run = {row["run_id"]: row["threshold"] for row in per_run_rows}
    comparisons: dict[str, Any] = {}
    for arm_a, arm_b in ARM_PAIRS:
        per_seed = []
        for seed in runner.SEEDS:
            run_a, run_b = by_arm_seed[(arm_a, seed)], by_arm_seed[(arm_b, seed)]
            result_a, result_b = results_by_run[run_a], results_by_run[run_b]
            video_a = {row["video_id"]: row for row in result_a["video_scores"]}
            video_b = {row["video_id"]: row for row in result_b["video_scores"]}
            shared_ids = sorted(set(video_a) & set(video_b))
            if not shared_ids:
                raise E8TargetScoringError(f"{arm_a} vs {arm_b} seed {seed}: no shared scored videos")
            scores_a = [float(video_a[v]["video_score"]) for v in shared_ids]
            scores_b = [float(video_b[v]["video_score"]) for v in shared_ids]
            labels = [int(video_a[v]["label"]) for v in shared_ids]
            boot = paired_bootstrap(
                video_ids=shared_ids, scores_a=scores_a, scores_b=scores_b, labels=labels,
                threshold=threshold_by_run[run_a], threshold_a=threshold_by_run[run_a],
                threshold_b=threshold_by_run[run_b], settings=settings)
            per_seed.append({"seed": seed, "run_a": run_a, "run_b": run_b,
                             "observed_acer_delta": boot["observed_delta"],
                             "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
                             "p_value": boot["p_value"], "significant_at_alpha": boot["significant_at_alpha"]})
        deltas = np.asarray([row["observed_acer_delta"] for row in per_seed], dtype=np.float64)
        comparisons[f"{arm_a}_vs_{arm_b}"] = {
            "per_seed": per_seed, "mean_paired_acer_delta": float(deltas.mean()),
            "std_paired_acer_delta": float(deltas.std(ddof=0)),
            # A single representative p-value for the Holm family: the seed-20260810
            # bootstrap's p-value is the one already declared as this project's
            # frozen bootstrap seed (see bootstrap.DEFAULT_SEED); the family
            # correction below is what actually governs significance, not this
            # one number in isolation.
            "family_p_value": next(row["p_value"] for row in per_seed if row["seed"] == runner.SEEDS[-1]),
        }
    return comparisons


def build_holm_correction(paired_comparisons: dict[str, Any]) -> dict[str, Any]:
    p_values = {name: float(entry["family_p_value"]) for name, entry in paired_comparisons.items()}
    return holm_bonferroni(p_values)


def build_closure(arm_summary: dict[str, dict[str, Any]], holm_result: dict[str, Any]) -> dict[str, Any]:
    acer_means = {arm: arm_summary[arm]["acer_mean"] for arm in runner.ARMS}
    best_arm = min(acer_means, key=lambda arm: acer_means[arm])
    significant_pairs = list(holm_result["rejected_null"])
    differences_remain = bool(significant_pairs)
    llm_superior = (best_arm == "LLM"
                    and any(name in significant_pairs for name in ("RND_vs_LLM", "DET_vs_LLM")))
    return {
        "schema_version": "e8-target-evaluation-closure-v1",
        "question": "After q matching, do RND/DET/LLM performance differences remain?",
        "answer": (f"YES -- statistically significant paired ACER differences remain after "
                  f"Holm-Bonferroni correction: {significant_pairs}" if differences_remain else
                  "NO -- no paired ACER difference between RND/DET/LLM survives Holm-Bonferroni "
                  "correction at alpha=0.05 after q-matching"),
        "differences_remain": differences_remain,
        "significant_pairs_holm": significant_pairs,
        "acer_means_by_arm": acer_means,
        "best_arm_by_acer_mean": best_arm,
        "llm_superiority_supported": bool(llm_superior),
        "pre_qmatch_frozen_f1_comparison": {
            "available": False,
            "reason": ("no legitimate, comparable pre-q-match frozen-F1 TARGET evaluation result (real "
                      "target predictions plus G8 scoring against the held-out SiW-Mv2 target) was "
                      "found anywhere in this repository; marked unavailable rather than guessed"),
        },
        "ranking_change_vs_pre_qmatch": "UNAVAILABLE",
        "controls_additional_f2_f3_work": False,
        "f2_f3_note": ("no comparable pre-q-match result exists to detect a ranking change against, so "
                      "this E8 pass alone does not trigger additional F2/F3 work under this rule; a "
                      "separate historical audit would be needed before that decision could be made"),
    }


# --------------------------------------------------------------------------- #
# Writers -- every write goes through the G8 firewall check
# --------------------------------------------------------------------------- #

def _write_csv(path: Path, rows: list[dict[str, Any]], *, firewall: Any) -> Path:
    target = firewall.check_write("G8", path)
    if not rows:
        raise E8TargetScoringError(f"refusing to write an empty CSV at {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return target


def write_all_evidence(
    repo: Path, *, per_run_rows: list[dict[str, Any]], arm_summary: dict[str, dict[str, Any]],
    paired_comparisons: dict[str, Any], holm_result: dict[str, Any], closure: dict[str, Any],
    label_use_record: dict[str, Any], firewall: Any | None = None,
) -> dict[str, str]:
    firewall = firewall or g7.build_target_firewall(repo)
    written: dict[str, str] = {}

    result_body = {"schema_version": "e8-target-score-result-v1", "required_metrics": list(REQUIRED_METRICS),
                  "per_run": per_run_rows, "arm_summary": arm_summary}
    firewall.check_write("G8", repo / SCORE_RESULT_PATH)
    written["score_result"] = cc.write_json_atomic(SCORE_RESULT_PATH, result_body, root=repo)

    written["per_run_csv"] = str(_write_csv(repo / PER_RUN_CSV_PATH, per_run_rows, firewall=firewall))
    arm_rows = [arm_summary[arm] for arm in runner.ARMS]
    written["arm_summary_csv"] = str(_write_csv(repo / ARM_SUMMARY_CSV_PATH, arm_rows, firewall=firewall))

    firewall.check_write("G8", repo / PAIRED_COMPARISONS_PATH)
    written["paired_comparisons"] = cc.write_json_atomic(
        PAIRED_COMPARISONS_PATH, {"comparisons": paired_comparisons, "holm_bonferroni": holm_result}, root=repo)

    firewall.check_write("G8", repo / CLOSURE_PATH)
    written["closure"] = cc.write_json_atomic(CLOSURE_PATH, closure, root=repo)

    firewall.check_write("G8", repo / LABEL_USE_RECORD_PATH)
    written["label_use_record"] = cc.write_json_atomic(LABEL_USE_RECORD_PATH, label_use_record, root=repo)

    evidence_files = [SCORE_RESULT_PATH, PER_RUN_CSV_PATH, ARM_SUMMARY_CSV_PATH,
                      PAIRED_COMPARISONS_PATH, CLOSURE_PATH, LABEL_USE_RECORD_PATH]
    lines = [f"{cc.sha256_file(repo / rel)}  {Path(rel).name}" for rel in evidence_files]
    evidence_text = "\n".join(lines) + "\n"
    evidence_path = repo / EVIDENCE_SHA256_PATH
    firewall.check_write("G8", evidence_path)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(evidence_text, encoding="utf-8")
    written["evidence_sha256"] = str(evidence_path)

    md = _build_summary_markdown(per_run_rows=per_run_rows, arm_summary=arm_summary,
                                 paired_comparisons=paired_comparisons, holm_result=holm_result,
                                 closure=closure)
    summary_path = repo / SUMMARY_MD_PATH
    firewall.check_write("G8", summary_path)
    summary_path.write_text(md, encoding="utf-8")
    written["summary_md"] = str(summary_path)
    return written


def _build_summary_markdown(*, per_run_rows: list[dict[str, Any]], arm_summary: dict[str, dict[str, Any]],
                            paired_comparisons: dict[str, Any], holm_result: dict[str, Any],
                            closure: dict[str, Any]) -> str:
    lines = ["# E8 Q-Matched Target Evaluation Summary", "",
            "## Arm summary (n=5 seeds/arm, mean +/- std ddof=0)", "",
            "| arm | ACER | APCER | BPCER | ROC-AUC |", "|---|---|---|---|---|"]
    for arm in runner.ARMS:
        row = arm_summary[arm]
        lines.append(f"| {arm} | {row['acer_mean']:.4f} +/- {row['acer_std']:.4f} "
                     f"| {row['apcer_mean']:.4f} +/- {row['apcer_std']:.4f} "
                     f"| {row['bpcer_mean']:.4f} +/- {row['bpcer_std']:.4f} "
                     f"| {row['roc_auc_mean']:.4f} +/- {row['roc_auc_std']:.4f} |")
    lines += ["", "## Per-run (15 rows)", "", "| run_id | arm | seed | ACER |", "|---|---|---|---|"]
    for row in per_run_rows:
        lines.append(f"| {row['run_id']} | {row['arm']} | {row['seed']} | {row['acer']:.4f} |")
    lines += ["", "## Paired same-seed ACER comparisons", ""]
    for name, entry in paired_comparisons.items():
        lines.append(f"- **{name}**: mean paired ACER delta = {entry['mean_paired_acer_delta']:.4f}")
    lines += ["", f"Holm-Bonferroni rejected null: {holm_result['rejected_null']}", "",
            "## Closure", "", f"**{closure['question']}**", "", closure["answer"], ""]
    return "\n".join(lines)
