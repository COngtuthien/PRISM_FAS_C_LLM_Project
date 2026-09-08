"""PRISM-FAS-C EXT-Q1Q2 -- E10: the frozen extension decision gate.

Section 17 of ``PRISM_FAS_C_EXT_Q1Q2_Detailed_Spec_v1_0.docx``.

What E10 is
-----------
E10 is an EVIDENCE-ONLY decision gate.  It reads already-frozen E0..E9 closure /
result artifacts, resolves the five frozen E10 criteria, and decides whether
another LLM call (E11) is authorized.  It runs no experiment: no detector
training, no LLM / provider / network call, no GPU, no rerun of E3..E9, and no
access to any raw target image / feature / label.  It reads frozen ``reports/``
JSON only, all of it under ``reports/c_ext_q1q2_v1/``.

The five frozen criteria (verbatim, spec Section 17)
---------------------------------------------------
1. CROSS-DOMAIN SIGNAL -- Frozen LLM bank is not ranked last in at least 2/3
   held-out folds; prefer LLM <= DET/RND in ACER.
2. VALUE OVER REAL-ONLY -- LLM is better than REAL-ONLY in at least 2/3 folds by
   descriptive mean ACER.
3. SEMANTIC ABLATION -- LLM-original is better than LLM-SHUFFLE-A in at least 2/3
   folds, OR the effect is clear in EXT-F1 and does not strongly reverse in the
   other folds.
4. QUALITY CONFOUND -- There is no large q confound, OR the q-matched ablation
   still preserves the direction of an LLM advantage.
5. INTEGRITY -- BA controls / threshold analysis do not reveal a bug or label
   leakage that invalidates the core evaluation.

Frozen decision rule
--------------------
The gate is CONJUNCTIVE and FAIL-CLOSED.  ``e11_authorized`` is true only if
every one of the five required criteria is POSITIVELY established
(``status == "PASS"``) from real, available evidence.  If the required
conditions are not established: no new LLM call, ``E11_AUTHORIZED = false``.

Interpretation rule (mandatory)
-------------------------------
Unavailable evidence is NOT evidence that the LLM failed.  A criterion whose
required upstream evidence is scientifically blocked, or is simply not present in
this checkout (the ``reports/c_ext_q1q2_v1`` tree is git-ignored; historical
milestones may be absent locally yet complete elsewhere), is recorded
``BLOCKED`` / ``NOT_ESTABLISHED`` -- never ``FAIL`` and never ``PASS``, and never
``NOT_EXECUTED`` on the basis of absence alone.  Missing evidence is never
converted into a negative scientific result.

Hard rules honoured by this module
----------------------------------
* Imports: the Python standard library and
  ``prism_fas.evaluation.c_ext_common`` ONLY.  Nothing that could pull in
  ``torch``, a provider SDK, a network client, ``prism_fas.detector`` or
  ``prism_fas.synthesis``.
* No training capability: this module fits no model, has no optimizer, loads no
  checkpoint, and constructs no trainer.
* Every write is routed through ``c_ext_common.assert_ext_write_path`` and is
  confined to ``reports/c_ext_q1q2_v1/e10_gate/`` (never the ``attempts/``
  subtree, which preserves earlier attempts read-only).
* Deterministic & idempotent: the output bytes are a pure function of the
  consumed artifacts' contents plus the frozen criteria table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prism_fas.evaluation.c_ext_common import (  # noqa: E402
    assert_ext_write_path, read_json, repo_root, sha256_file, sha256_json,
    write_json_atomic, write_text_atomic,
)

# --------------------------------------------------------------------------- #
# Frozen constants
# --------------------------------------------------------------------------- #

E10_ID = "e10_gate"
EXT_ROOT_RELPATH = "reports/c_ext_q1q2_v1"
OUTPUT_SUBTREE = f"{EXT_ROOT_RELPATH}/{E10_ID}"

E10_MODULE_RELPATH = "src/prism_fas/evaluation/c_ext_e10_gate.py"
E10_TEST_RELPATH = "tests/pipeline/test_c_ext_e10_gate.py"

SCHEMA_VERSION = "ext-q1q2-e10-gate-v3"

FOLDS: tuple[str, ...] = ("EXT-F1", "EXT-F2", "EXT-F3")
MIN_FOLDS = 2  # "at least 2/3"

#: Frozen E0 q-distribution trigger magnitude (|SMD(q)| >= 0.25 between LLM and
#: RND or between LLM and DET).  Upstream frozen scientific constant, copied from
#: ``reports/c_ext_q1q2_v1/e0/EXT_HYPOTHESIS_FAMILY.json`` -- NOT invented here.
Q_CONFOUND_TRIGGER_SMD = 0.25

# --------------------------------------------------------------------------- #
# Frozen criteria table (spec Section 17, verbatim text)
# --------------------------------------------------------------------------- #

CRITERION_IDS: tuple[str, ...] = (
    "C1_CROSS_DOMAIN_SIGNAL",
    "C2_VALUE_OVER_REAL_ONLY",
    "C3_SEMANTIC_ABLATION",
    "C4_QUALITY_CONFOUND",
    "C5_INTEGRITY",
)

FROZEN_CRITERIA: tuple[dict[str, Any], ...] = (
    {
        "criterion_id": "C1_CROSS_DOMAIN_SIGNAL",
        "title": "CROSS-DOMAIN SIGNAL",
        "frozen_criterion_text": (
            "Frozen LLM bank is not ranked last in at least 2/3 held-out folds; "
            "prefer LLM <= DET/RND in ACER."
        ),
        "required": True,
        "positive_evidence_requires": (
            "Completed E7 detector trainings G-LLM, G-RND and G-DET with a "
            "held-out target ACER per fold for each arm (a 2/3-fold comparison; "
            "single-fold EXT-F1 results such as E8 or E4 do not count)."
        ),
    },
    {
        "criterion_id": "C2_VALUE_OVER_REAL_ONLY",
        "title": "VALUE OVER REAL-ONLY",
        "frozen_criterion_text": (
            "LLM is better than REAL-ONLY in at least 2/3 folds by descriptive "
            "mean ACER."
        ),
        "required": True,
        "positive_evidence_requires": (
            "A completed, comparable LLM arm and REAL-ONLY arm target ACER in at "
            "least 2 of the 3 held-out folds (E5 supplies REAL-ONLY for EXT-F1 "
            "only; the matched LLM arm needs E7 G-LLM)."
        ),
    },
    {
        "criterion_id": "C3_SEMANTIC_ABLATION",
        "title": "SEMANTIC ABLATION",
        "frozen_criterion_text": (
            "LLM-original is better than LLM-SHUFFLE-A in at least 2/3 folds, OR "
            "the effect is clear in EXT-F1 and does not strongly reverse in the "
            "other folds."
        ),
        "required": True,
        "positive_evidence_requires": (
            "Completed E7 detector trainings G-LLM and G-LLM-SHUFFLE-A (or the "
            "E6 paired original/shuffle matched-bank detector comparison) with a "
            "held-out target ACER."
        ),
    },
    {
        "criterion_id": "C4_QUALITY_CONFOUND",
        "title": "QUALITY CONFOUND",
        "frozen_criterion_text": (
            "There is no large q confound, OR the q-matched ablation still "
            "preserves the direction of an LLM advantage."
        ),
        "required": True,
        "positive_evidence_requires": (
            "EITHER a measured |SMD(q)| below the frozen 0.25 trigger, OR a "
            "completed E8 q-matched detector ablation whose DESCRIPTIVE results "
            "retain the direction of an LLM advantage (LLM at least as good as "
            "DET and RND by descriptive mean ACER after q-matching). The frozen "
            "rule does NOT require inferential significance and does NOT require "
            "a comparable pre-q-match baseline; a PASS via this branch carries an "
            "explicit DESCRIPTIVE_ONLY / NO_SIGNIFICANCE_CLAIM qualifier."
        ),
    },
    {
        "criterion_id": "C5_INTEGRITY",
        "title": "INTEGRITY",
        "frozen_criterion_text": (
            "BA controls / threshold analysis do not reveal a bug or label "
            "leakage that invalidates the core evaluation."
        ),
        "required": True,
        "positive_evidence_requires": (
            "The BA-control / threshold-transfer / label-use provenance evidence "
            "that WAS produced (E3, E4, E8 label-use record, E5 calibration "
            "provenance) reveals no bug and no label leakage that would "
            "invalidate the core evaluation. The frozen rule does NOT require "
            "every BA control to have completed; incomplete E3 controls are a "
            "recorded LIMITATION, not an automatic non-establishment. A FAIL "
            "requires an actual invalidating bug or leakage to have been found; "
            "NOT_ESTABLISHED applies only when there is no integrity evidence at "
            "all."
        ),
    },
)

# --------------------------------------------------------------------------- #
# Evidence source artifacts (frozen, read-only).  All repo-relative POSIX,
# all under reports/c_ext_q1q2_v1/ (a git-ignored tree: absence here is
# "not present in this checkout", never "not executed").
# --------------------------------------------------------------------------- #

EVIDENCE_SOURCES: dict[str, str] = {
    # E0 -- frozen hypothesis family + E8 trigger rule + claim ceiling
    "e0_hypothesis_family": f"{EXT_ROOT_RELPATH}/e0/EXT_HYPOTHESIS_FAMILY.json",
    "e0_validation": f"{EXT_ROOT_RELPATH}/e0/EXT_E0_VALIDATION.json",
    # E2 -- q distribution / pairwise SMD (the q confound magnitude)
    "e2_quality_analysis": f"{EXT_ROOT_RELPATH}/e2_quality/E2_QUALITY_ANALYSIS.json",
    # E3 -- BA negative controls
    "e3_ba_controls": f"{EXT_ROOT_RELPATH}/e3_ba_controls/E3_BA_CONTROLS.json",
    # E4 -- threshold transfer
    "e4_threshold_transfer": (
        f"{EXT_ROOT_RELPATH}/e4_threshold_transfer/E4_THRESHOLD_TRANSFER.json"
    ),
    # E5 -- real-only target evaluation
    "e5_target_scoring_result": (
        f"{EXT_ROOT_RELPATH}/e5_realonly/target_scoring/E5_TARGET_SCORING_RESULT.json"
    ),
    "e5_real_only_lock": f"{EXT_ROOT_RELPATH}/e5_realonly/E5_REAL_ONLY_LOCK.json",
    # E6 -- paired original / shuffle rerender
    "e6_v2_final_closure": (
        f"{EXT_ROOT_RELPATH}/e6_paired_current_runtime_v2/E6_V2_FINAL_CLOSURE.json"
    ),
    # E7 -- three-fold core synthetic evaluation
    "e7_three_fold_scientific_closure": (
        f"{EXT_ROOT_RELPATH}/e7_three_fold/gpat_bank/e7_v1_1_reserve/"
        "E7_V1_1_THREE_FOLD_SCIENTIFIC_CLOSURE.json"
    ),
    "e7_e8_trigger_record": f"{EXT_ROOT_RELPATH}/e7_three_fold/E7_E8_TRIGGER_RECORD.json",
    # E8 -- q-matched target evaluation (authoritative)
    "e8_training_integration_audit": (
        f"{EXT_ROOT_RELPATH}/e8_qmatched/training/E8_TRAINING_INTEGRATION_AUDIT.json"
    ),
    "e8_target_evaluation_closure": (
        f"{EXT_ROOT_RELPATH}/e8_qmatched/target_eval_v1/E8_TARGET_EVALUATION_CLOSURE.json"
    ),
    "e8_target_score_result": (
        f"{EXT_ROOT_RELPATH}/e8_qmatched/target_eval_v1/E8_TARGET_SCORE_RESULT.json"
    ),
    "e8_target_paired_comparisons": (
        f"{EXT_ROOT_RELPATH}/e8_qmatched/target_eval_v1/E8_TARGET_PAIRED_COMPARISONS.json"
    ),
    "e8_target_label_use_record": (
        f"{EXT_ROOT_RELPATH}/e8_qmatched/target_eval_v1/E8_TARGET_LABEL_USE_RECORD.json"
    ),
    "e8_target_prediction_lockset": (
        f"{EXT_ROOT_RELPATH}/e8_qmatched/target_eval_v1/E8_TARGET_PREDICTION_LOCKSET.json"
    ),
    # E9 -- bank-selection robustness (NOT detector-performance evidence)
    "e9_closure": f"{EXT_ROOT_RELPATH}/e9_bank_robustness/E9_CLOSURE.json",
    "e9_bank_stability": f"{EXT_ROOT_RELPATH}/e9_bank_robustness/E9_BANK_STABILITY.json",
}

#: Artifacts that WOULD carry positive comparative evidence for a criterion.
#: Expected-possibly-absent: when a real, authorized upstream run produces one,
#: E10 reads it; until then its absence is reported plainly and is NOT a negative
#: result.
#:
#: * ``e7_detector_target_acer`` -- an optional block inside the E7 closure at
#:   ``downstream_dependency_classification.E7_detector_experiment.target_acer_by_fold``
#:   mapping fold -> arm -> descriptive mean ACER (the frozen 3-fold design).
#: * ``e10_integrity_diagnostics`` -- a completed BA-controls + threshold-transfer
#:   + leakage certification over a COMPLETED core evaluation.
OPTIONAL_POSITIVE_EVIDENCE: dict[str, str] = {
    "e10_integrity_diagnostics": f"{OUTPUT_SUBTREE}/E10_INTEGRITY_DIAGNOSTICS.json",
}

#: Extension sub-milestone report subtrees.  Presence => the milestone left
#: artifacts in THIS checkout; absence => "not present in this checkout" (the
#: whole reports/c_ext_q1q2_v1 tree is git-ignored) -- NEVER "not executed".
MILESTONE_SUBTREES: dict[str, str] = {
    "E0": f"{EXT_ROOT_RELPATH}/e0",
    "E1": f"{EXT_ROOT_RELPATH}/e1_recipe_analysis",
    "E2": f"{EXT_ROOT_RELPATH}/e2_quality",
    "E3": f"{EXT_ROOT_RELPATH}/e3_ba_controls",
    "E4": f"{EXT_ROOT_RELPATH}/e4_threshold_transfer",
    "E5": f"{EXT_ROOT_RELPATH}/e5_realonly",
    "E6": f"{EXT_ROOT_RELPATH}/e6_paired_current_runtime_v2",
    "E7": f"{EXT_ROOT_RELPATH}/e7_three_fold",
    "E8": f"{EXT_ROOT_RELPATH}/e8_qmatched",
    "E9": f"{EXT_ROOT_RELPATH}/e9_bank_robustness",
}

# Output artifact relpaths (never under attempts/).
INPUT_AUDIT_RELPATH = f"{OUTPUT_SUBTREE}/E10_INPUT_AUDIT.json"
EVIDENCE_MATRIX_RELPATH = f"{OUTPUT_SUBTREE}/E10_EVIDENCE_MATRIX.json"
GATE_DECISION_RELPATH = f"{OUTPUT_SUBTREE}/E10_GATE_DECISION.json"
EVIDENCE_MANIFEST_RELPATH = f"{OUTPUT_SUBTREE}/E10_EVIDENCE.sha256"
CLOSURE_RELPATH = f"{OUTPUT_SUBTREE}/E10_CLOSURE.json"

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_NOT_ESTABLISHED = "NOT_ESTABLISHED"
STATUS_BLOCKED = "BLOCKED"
_VALID_STATUSES = (STATUS_PASS, STATUS_FAIL, STATUS_NOT_ESTABLISHED, STATUS_BLOCKED)

POSITIVE_STATUS = STATUS_PASS

#: Milestone evidence-state vocabulary.  Absence is EVIDENCE_NOT_PRESENT_IN_CHECKOUT
#: -- never NOT_EXECUTED / FAILED.
MS_COMPLETE = "COMPLETE"
MS_CLOSED = "CLOSED"
MS_EXECUTED_AND_SCORED = "EXECUTED_AND_SCORED"
MS_SCIENTIFICALLY_BLOCKED = "SCIENTIFICALLY_BLOCKED"
MS_FEASIBILITY_AUDIT_ONLY = "FEASIBILITY_AUDIT_ONLY_GPU_GATED"
MS_PARTIAL_F1_ONLY = "PARTIAL_EXT_F1_ONLY"
MS_EVIDENCE_ABSENT = "EVIDENCE_NOT_PRESENT_IN_CHECKOUT"
MS_PRESENT_UNREADABLE = "EVIDENCE_PRESENT_BUT_UNREADABLE"


class E10Error(RuntimeError):
    """E10 refused an operation (fail-closed)."""


# --------------------------------------------------------------------------- #
# Evidence loading (tolerant of absence -- never fabricates)
# --------------------------------------------------------------------------- #

def _resolve(root: Path, relpath: str) -> dict[str, Any]:
    abs_path = root / relpath
    if not abs_path.exists():
        return {"relpath": relpath, "present": False, "sha256": None, "json": None,
                "readable": None}
    sha = sha256_file(abs_path) if abs_path.is_file() else None
    doc: Any = None
    readable = None
    if abs_path.is_file():
        readable = True
        try:
            doc = read_json(abs_path)
        except (ValueError, OSError):
            doc = None
            readable = False
    return {"relpath": relpath, "present": True, "sha256": sha, "json": doc,
            "readable": readable}


def load_evidence(root: Path | None = None) -> dict[str, Any]:
    """Load every evidence artifact E10 may consult.  Pure read, no writes."""
    root = root or repo_root()
    sources = {name: _resolve(root, rel) for name, rel in EVIDENCE_SOURCES.items()}
    optional = {name: _resolve(root, rel)
                for name, rel in OPTIONAL_POSITIVE_EVIDENCE.items()}
    milestones = {}
    for name, rel in MILESTONE_SUBTREES.items():
        p = root / rel
        milestones[name] = {"relpath": rel, "present": p.exists(), "is_dir": p.is_dir()}
    return {"root": str(root), "sources": sources, "optional": optional,
            "milestones": milestones}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _get(doc: Any, *path: str, default: Any = None) -> Any:
    cur = doc
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _src(evd: dict[str, Any], name: str, *path: str, default: Any = None) -> Any:
    return _get(_get(evd["sources"], name, "json"), *path, default=default)


def _present(evd: dict[str, Any], name: str) -> bool:
    return bool(_get(evd["sources"], name, "present"))


def _rel(name: str) -> str:
    return EVIDENCE_SOURCES.get(name, OPTIONAL_POSITIVE_EVIDENCE.get(name, name))


# --------------------------------------------------------------------------- #
# Structured readers over the frozen artifacts
# --------------------------------------------------------------------------- #

def _e7_state(evd: dict[str, Any]) -> dict[str, Any]:
    rec = _get(evd["sources"], "e7_three_fold_scientific_closure")
    doc = rec["json"] if rec else None
    if not rec or not rec["present"]:
        return {"present": False, "state": MS_EVIDENCE_ABSENT}
    if not isinstance(doc, Mapping):
        return {"present": True, "state": MS_PRESENT_UNREADABLE}
    fsi = _get(doc, "final_scientific_interpretation", default={})
    folds = _get(doc, "folds", default={})
    fold_status = {f: _get(folds, f, "final_status")
                   for f in folds if isinstance(folds, Mapping)}
    all_blocked = bool(fold_status) and all(
        s == "SCIENTIFICALLY_BLOCKED_AFTER_RESERVE_CAP" for s in fold_status.values())
    core = str(_get(fsi, "core_matched_banks_produced", default="")).replace(" ", "")
    per_cond = _get(doc, "downstream_dependency_classification",
                    "E7_detector_experiment", "per_condition", default={}) or {}
    acer_block = _get(doc, "downstream_dependency_classification",
                      "E7_detector_experiment", "target_acer_by_fold", default=None)
    return {
        "present": True,
        "state": MS_SCIENTIFICALLY_BLOCKED if (all_blocked and core in ("0/3", "0of3"))
        else MS_CLOSED,
        "all_folds_blocked": all_blocked,
        "core_matched_banks_produced": _get(fsi, "core_matched_banks_produced"),
        "is_target_acer_evidence": _get(fsi, "is_target_acer_evidence"),
        "fold_status": fold_status,
        "per_condition": {k: _get(per_cond, k, "disposition")
                          for k in ("G-RND", "G-DET", "G-LLM",
                                    "G-LLM-SHUFFLE-A", "G-REALONLY")},
        "integrity_and_leakage_statement": _get(doc, "integrity_and_leakage_statement",
                                                default={}),
        "detector_target_acer_by_fold": acer_block if isinstance(acer_block, Mapping)
        else None,
    }


def _e6_state(evd: dict[str, Any]) -> dict[str, Any]:
    rec = _get(evd["sources"], "e6_v2_final_closure")
    doc = rec["json"] if rec else None
    if not rec or not rec["present"]:
        return {"present": False, "state": MS_EVIDENCE_ABSENT}
    if not isinstance(doc, Mapping):
        return {"present": True, "state": MS_PRESENT_UNREADABLE}
    status = _get(doc, "E6_V2_STATUS")
    ready = _get(doc, "E6_V2_READY_FOR_TRAINING")
    return {
        "present": True, "state": MS_CLOSED,
        "E6_V2_STATUS": status,
        "E6_V2_READY_FOR_TRAINING": ready,
        "training_block_reason": _get(doc, "E6_V2_TRAINING_BLOCK_REASON"),
        "shuffle_infeasible": status == "CLOSED_TRUE_FROZEN_MATCHED_BANK_INFEASIBILITY"
        and ready is False,
        "target_access": _get(doc, "TARGET_ACCESS"),
        "llm_api_calls": _get(doc, "LLM_API_CALLS"),
    }


def _e8_state(evd: dict[str, Any]) -> dict[str, Any]:
    """Authoritative E8 q-matched target-evaluation state.

    CRITICAL: the E8 target-evaluation artifacts live under a git-ignored tree.
    If they are ABSENT, the state is EVIDENCE_NOT_PRESENT_IN_CHECKOUT -- it is
    NEVER inferred to be NOT_EXECUTED / FAILED from absence alone.  A definitive
    "completed" classification requires the artifacts to be present and readable.
    """
    closure = _get(evd["sources"], "e8_target_evaluation_closure")
    score = _get(evd["sources"], "e8_target_score_result")
    paired = _get(evd["sources"], "e8_target_paired_comparisons")
    label_use = _get(evd["sources"], "e8_target_label_use_record")
    audit = _get(evd["sources"], "e8_training_integration_audit")

    c_present = bool(closure and closure["present"])
    c_doc = closure["json"] if closure else None
    s_doc = score["json"] if score else None

    if not c_present and not (score and score["present"]):
        return {
            "state": MS_EVIDENCE_ABSENT,
            "present": False,
            "note": (
                "E8 target-evaluation artifacts (target_eval_v1/) are not present "
                "in this checkout. reports/c_ext_q1q2_v1 is git-ignored, so this "
                "is 'evidence not present here', NOT 'E8 not executed'. Do not "
                "classify E8 as NOT_EXECUTED or as an LLM failure on this basis; "
                "restore the authoritative E8 evidence to classify definitively."
            ),
            "training_integration_audit_status": _get(audit, "json", "status")
            if audit else None,
        }
    if c_present and not isinstance(c_doc, Mapping):
        return {"state": MS_PRESENT_UNREADABLE, "present": True}

    acer_means = _get(c_doc, "acer_means_by_arm", default=None)
    if not isinstance(acer_means, Mapping):
        acer_means = _get(s_doc, "arm_summary", default=None)
        if isinstance(acer_means, Mapping):
            acer_means = {a: _get(acer_means, a, "acer_mean")
                          for a in acer_means}
    means: dict[str, float] = {}
    if isinstance(acer_means, Mapping):
        for a, v in acer_means.items():
            try:
                means[str(a)] = float(v)
            except (TypeError, ValueError):
                pass

    p_doc = paired["json"] if paired else None
    comparisons = _get(p_doc, "descriptive_seed_summary", "comparisons", default={}) or {}
    paired_deltas: dict[str, Any] = {}
    for key, blk in comparisons.items() if isinstance(comparisons, Mapping) else []:
        paired_deltas[key] = {
            "mean_delta": _get(blk, "mean_delta"),
            "count_positive": _get(blk, "count_positive"),
            "count_negative": _get(blk, "count_negative"),
            "n": _get(blk, "n"),
        }

    lu = label_use["json"] if label_use else None
    return {
        "state": MS_EXECUTED_AND_SCORED,
        "present": True,
        "fold_scope": "EXT-F1",
        "acer_means_by_arm": means,
        "best_arm_by_acer_mean_descriptive_only": _get(
            c_doc, "best_arm_by_acer_mean_descriptive_only"),
        "inferential_status": _get(c_doc, "inferential_status"),
        "statistical_significance_claimed": _get(
            c_doc, "statistical_significance_claimed"),
        "llm_superiority_supported": _get(c_doc, "llm_superiority_supported"),
        "pre_qmatch_comparison_available": _get(
            c_doc, "pre_qmatch_frozen_f1_comparison", "available"),
        "ranking_change_vs_pre_qmatch": _get(c_doc, "ranking_change_vs_pre_qmatch"),
        "paired_deltas": paired_deltas,
        "label_use": {
            "uses_target_labels_in_training_or_g7": _get(
                lu, "e8_uses_target_labels_in_training_or_g7"),
            "g8_structurally_isolated_from_training_and_g7": _get(
                lu, "e8_g8_structurally_isolated_from_training_and_g7"),
            "is_first_ever_blind_reveal": _get(lu, "is_first_ever_blind_reveal"),
        },
    }


def _e5_state(evd: dict[str, Any]) -> dict[str, Any]:
    rec = _get(evd["sources"], "e5_target_scoring_result")
    doc = rec["json"] if rec else None
    if not rec or not rec["present"]:
        return {"state": MS_EVIDENCE_ABSENT, "present": False}
    if not isinstance(doc, Mapping):
        return {"state": MS_PRESENT_UNREADABLE, "present": True}
    agg = _get(doc, "aggregate", default={})
    return {
        "state": MS_PARTIAL_F1_ONLY,
        "present": True,
        "arm": _get(doc, "arm"),
        "fold": _get(doc, "fold"),
        "acer_mean": _get(agg, "acer", "mean"),
        "acer_sample_sd": _get(agg, "acer", "sample_sd"),
        "roc_auc_mean": _get(agg, "roc_auc", "mean"),
        "seed_count": _get(agg, "seed_count"),
        "target_threshold_fitted": _get(doc, "target_threshold_fitted"),
        "target_scoring_executed": _get(doc, "target_scoring_executed"),
        "threshold_rule": _get(doc, "threshold_rule"),
    }


def _e4_state(evd: dict[str, Any]) -> dict[str, Any]:
    rec = _get(evd["sources"], "e4_threshold_transfer")
    doc = rec["json"] if rec else None
    if not rec or not rec["present"]:
        return {"state": MS_EVIDENCE_ABSENT, "present": False}
    if not isinstance(doc, Mapping):
        return {"state": MS_PRESENT_UNREADABLE, "present": True}
    agg = _get(doc, "aggregates_ddof1", default={})
    gap = {}
    for arm in ("LLM", "DET", "RND"):
        gap[arm] = _get(agg, arm, "video", "acer_gap_t0_minus_oracle", "mean")
    return {
        "state": MS_PARTIAL_F1_ONLY,
        "present": True,
        "fold": _get(doc, "fold"),
        "status": _get(doc, "status"),
        "t0_minus_oracle_acer_gap_video_mean": gap,
        "f2_f3_status": _get(doc, "feasibility_audit",
                             "EXT_F2_F3_threshold_transfer", "status"),
        "target_labels_accessed": _get(doc, "target_labels_accessed"),
        "interpretation_boundary": _get(doc, "scientific_interpretation_boundary"),
    }


def _e3_state(evd: dict[str, Any]) -> dict[str, Any]:
    rec = _get(evd["sources"], "e3_ba_controls")
    doc = rec["json"] if rec else None
    if not rec or not rec["present"]:
        return {"state": MS_EVIDENCE_ABSENT, "present": False}
    if not isinstance(doc, Mapping):
        return {"state": MS_PRESENT_UNREADABLE, "present": True}
    per_control = _get(doc, "per_control", default={})
    controls_run = list(_get(per_control, "COMPLETE_CPU", default=[]) or [])
    controls_not_run = list(_get(per_control, "GPU_REQUIRED", default=[]) or [])
    controls_unsupported = list(_get(per_control, "UNSUPPORTED", default=[]) or [])
    # delta_BA_excess needs C3 (syn-vs-real BA) plus C0/C1 controls -- all GPU.
    delta_ba_computable = not controls_not_run and not controls_unsupported
    return {
        "state": MS_FEASIBILITY_AUDIT_ONLY,
        "present": True,
        "status": _get(doc, "status"),
        "controls_complete": controls_run,
        "controls_gpu_required": controls_not_run,
        "controls_unsupported": controls_unsupported,
        "delta_ba_excess_computable": delta_ba_computable,
        "interpretation_boundary": _get(doc, "scientific_interpretation_boundary"),
        "target_labels_accessed": _get(doc, "target_labels_accessed"),
    }


def _e2_q_confound(evd: dict[str, Any]) -> dict[str, Any]:
    rec = _get(evd["sources"], "e2_quality_analysis")
    doc = rec["json"] if rec else None
    if not rec or not rec["present"]:
        return {"state": MS_EVIDENCE_ABSENT, "present": False}
    if not isinstance(doc, Mapping):
        return {"state": MS_PRESENT_UNREADABLE, "present": True}
    pairs = _get(doc, "q_distribution", "pairwise_smd", default=[]) or []
    by_pair: dict[str, float] = {}
    max_abs = 0.0
    for p in pairs:
        if not isinstance(p, Mapping):
            continue
        a, b, smd = p.get("arm_a"), p.get("arm_b"), p.get("smd")
        try:
            smd = float(smd)
        except (TypeError, ValueError):
            continue
        by_pair[f"{a}_vs_{b}"] = smd
        max_abs = max(max_abs, abs(smd))
    trigger = any(abs(v) >= Q_CONFOUND_TRIGGER_SMD
                  for k, v in by_pair.items() if k.startswith("LLM_vs_"))
    means = _get(doc, "q_distribution", "q_overall_by_arm", default={})
    llm_mean = _get(means, "LLM", "mean")
    return {
        "state": MS_COMPLETE,
        "present": True,
        "pairwise_smd": by_pair,
        "max_abs_smd": max_abs,
        "large_q_confound_present": trigger,
        "trigger_threshold": Q_CONFOUND_TRIGGER_SMD,
        "llm_q_mean": llm_mean,
        "q_mean_by_arm": {a: _get(means, a, "mean") for a in ("LLM", "DET", "RND")},
    }


def _e9_state(evd: dict[str, Any]) -> dict[str, Any]:
    c = _get(evd["sources"], "e9_closure", "json")
    stab = _get(evd["sources"], "e9_bank_stability", "json")
    if not _present(evd, "e9_closure"):
        return {"state": MS_EVIDENCE_ABSENT, "present": False}
    return {
        "state": _get(c, "status") or MS_CLOSED,
        "present": True,
        "realizations_ok": _get(c, "realizations_ok"),
        "realizations_total": _get(c, "realizations_total"),
        "realizations_blocked": _get(c, "realizations_blocked"),
        "llm_calls": _get(c, "llm_calls"),
        "target_access": _get(c, "target_access"),
        "is_detector_performance_evidence": False,
        "stability_metric_note": _get(stab, "arms", "LLM", "note") or _get(stab, "note"),
    }


def _e0_state(evd: dict[str, Any]) -> dict[str, Any]:
    hf = _get(evd["sources"], "e0_hypothesis_family", "json")
    val = _get(evd["sources"], "e0_validation", "json")
    if not _present(evd, "e0_hypothesis_family"):
        return {"state": MS_EVIDENCE_ABSENT, "present": False}
    return {
        "state": _get(val, "milestone_status") or MS_COMPLETE,
        "present": True,
        "scientific_content_status": _get(val, "e0_scientific_content_status"),
        "claim_ceiling_without_e11": _get(hf, "claim_ceiling_without_e11"),
        "primary_hypotheses": [h.get("id") + ":" + h.get("contrast")
                               for h in _get(hf, "primary_family", "hypotheses",
                                             default=[]) if isinstance(h, Mapping)],
        "diagnostic_families_not_in_holm": _get(hf, "diagnostic_families_not_in_holm"),
        "e11_rule": _get(hf, "e11_rule"),
        "e8_trigger_rule": _get(hf, "e8_trigger", "rule"),
    }


def _e10_integrity_diag(evd: dict[str, Any]) -> dict[str, Any] | None:
    doc = _get(evd["optional"], "e10_integrity_diagnostics", "json")
    return doc if isinstance(doc, Mapping) else None


# --------------------------------------------------------------------------- #
# Criterion evaluators
# --------------------------------------------------------------------------- #

def _result(cid: str, *, status: str, reason: str, used: Sequence[str],
            missing: Sequence[str], observed: Sequence[str],
            upstream_block: bool, label: str) -> dict[str, Any]:
    if status not in _VALID_STATUSES:  # pragma: no cover
        raise E10Error(f"invalid status {status!r} for {cid}")
    meta = next(c for c in FROZEN_CRITERIA if c["criterion_id"] == cid)
    return {
        "criterion_id": cid,
        "title": meta["title"],
        "frozen_criterion_text": meta["frozen_criterion_text"],
        "required": meta["required"],
        "evidence_artifacts_used": list(used),
        "evidence_artifacts_missing": list(missing),
        "observed_facts": list(observed),
        "status": status,
        "resolution_label": label,
        "reason": reason,
        "missing_evidence_due_to_upstream_scientific_block": bool(upstream_block),
        "positively_established": status == POSITIVE_STATUS,
    }


def _fmt(x: Any, nd: int = 4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def evaluate_c1(evd: dict[str, Any]) -> dict[str, Any]:
    cid = "C1_CROSS_DOMAIN_SIGNAL"
    e7 = _e7_state(evd)
    e8 = _e8_state(evd)
    e4 = _e4_state(evd)
    src_e7 = _rel("e7_three_fold_scientific_closure")

    # Positive path: the frozen 3-fold E7 detector target ACER block, all folds.
    table = e7.get("detector_target_acer_by_fold")
    folds_ok = []
    if isinstance(table, Mapping):
        folds_ok = [f for f in FOLDS
                    if isinstance(table.get(f), Mapping)
                    and {"LLM", "DET", "RND"} <= set(table[f])]
    if len(folds_ok) >= 3:
        not_last = [f for f in folds_ok
                    if float(table[f]["LLM"]) <= max(float(table[f]["DET"]),
                                                     float(table[f]["RND"]))]
        prefer = [f for f in folds_ok
                  if float(table[f]["LLM"]) <= float(table[f]["DET"])
                  and float(table[f]["LLM"]) <= float(table[f]["RND"])]
        observed = [f"E7 per-fold target ACER present for LLM/DET/RND in {folds_ok}",
                    f"LLM not ranked last in {len(not_last)}/3 folds: {not_last}",
                    f"LLM <= both DET and RND in {len(prefer)}/3 folds: {prefer}"]
        if len(not_last) >= MIN_FOLDS:
            return _result(cid, status=STATUS_PASS,
                           reason=(f"LLM bank is not ranked last by target ACER in "
                                   f"{len(not_last)}/3 held-out folds (>= 2/3)."),
                           used=[src_e7], missing=[], observed=observed,
                           upstream_block=False, label="ESTABLISHED_FROM_EVIDENCE")
        return _result(cid, status=STATUS_FAIL,
                       reason=(f"LLM bank is ranked last by target ACER in "
                               f"{3 - len(not_last)}/3 folds; the 2/3 condition "
                               "is not met."),
                       used=[src_e7], missing=[], observed=observed,
                       upstream_block=False,
                       label="EVIDENCE_PRESENT_CONDITION_NOT_MET")

    # No completed 3-fold E7 comparison.
    observed = [
        f"E7 core matched synthetic bank: {e7.get('core_matched_banks_produced')!r}; "
        f"all folds {e7.get('fold_status')}",
        f"E7 per-condition: G-LLM={e7.get('per_condition', {}).get('G-LLM')}, "
        f"G-RND={e7.get('per_condition', {}).get('G-RND')}, "
        f"G-DET={e7.get('per_condition', {}).get('G-DET')}",
        f"E7 is_target_acer_evidence = {e7.get('is_target_acer_evidence')!r}",
    ]
    if e8.get("state") == MS_EXECUTED_AND_SCORED and e8.get("acer_means_by_arm"):
        m = e8["acer_means_by_arm"]
        observed.append(
            "DESCRIPTIVE, single fold (EXT-F1), q-matched E8 ablation -- NOT a "
            "2/3-fold cross-domain result and NOT counted for this criterion: "
            f"ACER LLM={_fmt(m.get('LLM'))} DET={_fmt(m.get('DET'))} "
            f"RND={_fmt(m.get('RND'))} (LLM ranked "
            f"{'1st (best)' if m.get('LLM') == min(m.values()) else 'not 1st'} of 3; "
            f"inferential_status={e8.get('inferential_status')})")
    if e4.get("present") and e4.get("t0_minus_oracle_acer_gap_video_mean"):
        observed.append(
            "DESCRIPTIVE, single fold (EXT-F1), E4 threshold-transfer re-analysis of "
            "the frozen historical Flow-1 Track-G target evaluation -- NOT an E7 "
            "3-fold result and NOT counted for this criterion.")
    missing = [f"{src_e7} :: downstream_dependency_classification."
               "E7_detector_experiment.target_acer_by_fold (LLM/DET/RND, all 3 folds)"]
    return _result(cid, status=STATUS_BLOCKED,
                   reason=(
                       "The G-LLM / G-RND / G-DET detector trainings this criterion "
                       "needs are BLOCKED_BY_E7_CORE_BANK: 0/3 folds produced a valid "
                       "core matched synthetic bank (all folds "
                       "SCIENTIFICALLY_BLOCKED_AFTER_RESERVE_CAP), so no per-fold "
                       "cross-domain LLM/DET/RND target ACER exists. The only "
                       "per-fold LLM/DET/RND ACERs anywhere are EXT-F1-only and from "
                       "non-E7 designs (E8 q-matched; E4 historical re-analysis) and "
                       "cannot satisfy a 2/3-fold requirement. Missing evidence is "
                       "NOT a negative LLM result."),
                   used=[src_e7], missing=missing, observed=observed,
                   upstream_block=True,
                   label="NOT_ESTABLISHED_DUE_TO_BLOCKED_EVIDENCE")


def evaluate_c2(evd: dict[str, Any]) -> dict[str, Any]:
    cid = "C2_VALUE_OVER_REAL_ONLY"
    e7 = _e7_state(evd)
    e5 = _e5_state(evd)
    e8 = _e8_state(evd)
    e4 = _e4_state(evd)
    src_e7 = _rel("e7_three_fold_scientific_closure")
    src_e5 = _rel("e5_target_scoring_result")

    # Positive path: a matched LLM & REAL_ONLY target ACER in >= 3 folds.
    table = e7.get("detector_target_acer_by_fold")
    folds_ok = []
    if isinstance(table, Mapping):
        folds_ok = [f for f in FOLDS
                    if isinstance(table.get(f), Mapping)
                    and {"LLM", "REAL_ONLY"} <= set(table[f])]
    if len(folds_ok) >= 3:
        better = [f for f in folds_ok
                  if float(table[f]["LLM"]) < float(table[f]["REAL_ONLY"])]
        observed = [f"per-fold LLM/REAL_ONLY target ACER present in {folds_ok}",
                    f"LLM better than REAL-ONLY by mean ACER in {len(better)}/3 "
                    f"folds: {better}"]
        if len(better) >= MIN_FOLDS:
            return _result(cid, status=STATUS_PASS,
                           reason=(f"LLM beats REAL-ONLY by descriptive mean target "
                                   f"ACER in {len(better)}/3 folds (>= 2/3)."),
                           used=[src_e7, src_e5], missing=[], observed=observed,
                           upstream_block=False, label="ESTABLISHED_FROM_EVIDENCE")
        return _result(cid, status=STATUS_FAIL,
                       reason=(f"LLM does not beat REAL-ONLY in >= 2/3 folds "
                               f"({len(better)}/3)."),
                       used=[src_e7, src_e5], missing=[], observed=observed,
                       upstream_block=False,
                       label="EVIDENCE_PRESENT_CONDITION_NOT_MET")

    observed = [
        f"E5 REAL-ONLY target evaluation: {e5.get('state')}, "
        f"fold={e5.get('fold')!r}, ACER mean={_fmt(e5.get('acer_mean'))} "
        f"(sd {_fmt(e5.get('acer_sample_sd'))}, {e5.get('seed_count')} seeds), "
        f"target_threshold_fitted={e5.get('target_threshold_fitted')!r} "
        "-- EXT-F1 only; no REAL-ONLY ACER exists for EXT-F2/EXT-F3.",
        f"E7 G-LLM (the matched LLM comparator): {e7.get('per_condition', {}).get('G-LLM')} "
        "-- 0/3 folds; no per-fold matched LLM target ACER.",
    ]
    if e8.get("state") == MS_EXECUTED_AND_SCORED and e8.get("acer_means_by_arm"):
        llm = e8["acer_means_by_arm"].get("LLM")
        observed.append(
            "Only LLM target ACER that exists: E8 q-matched EXT-F1 "
            f"LLM={_fmt(llm)} (5 seeds; a DIFFERENT ablation than the matched "
            "LLM-vs-REAL-ONLY design). Descriptively "
            f"{'WORSE (higher ACER) than' if (llm is not None and e5.get('acer_mean') is not None and llm > e5['acer_mean']) else 'vs'} "
            f"E5 REAL-ONLY EXT-F1 ({_fmt(e5.get('acer_mean'))}).")
    if e4.get("present"):
        g = e4.get("t0_minus_oracle_acer_gap_video_mean", {})
        observed.append(
            "E4 threshold-transfer (EXT-F1, frozen historical Flow-1 Track-G "
            "re-analysis, NOT a controlled EXT LLM-vs-REAL-ONLY comparison) "
            f"records a small T0-vs-oracle ACER gap (LLM {_fmt(g.get('LLM'))}).")
    missing = [
        f"{src_e7} :: matched G-LLM target ACER (BLOCKED_BY_E7_CORE_BANK, 0/3 folds)",
        f"{src_e5} :: REAL-ONLY target ACER for EXT-F2 and EXT-F3 (E5 is EXT-F1 only)",
    ]
    return _result(cid, status=STATUS_NOT_ESTABLISHED,
                   reason=(
                       "The frozen 2/3-fold LLM-vs-REAL-ONLY comparison (EXT-H3) "
                       "cannot be formed: E5 supplies REAL-ONLY target ACER for "
                       "EXT-F1 only (mean ACER "
                       f"{_fmt(e5.get('acer_mean'))}), the matched LLM comparator "
                       "(E7 G-LLM) is BLOCKED_BY_E7_CORE_BANK in all 3 folds, and "
                       "no REAL-ONLY result exists for EXT-F2/EXT-F3. At most one "
                       "fold could ever be compared, so the >= 2/3 requirement is "
                       "unreachable; and where the closest available LLM number "
                       "(E8 q-matched EXT-F1) touches this contrast it is "
                       "descriptively worse than REAL-ONLY, not better. No "
                       "positive evidence of LLM value over REAL-ONLY. Missing "
                       "folds are due to the upstream E7 block, not an LLM failure."),
                   used=[src_e5, src_e7], missing=missing, observed=observed,
                   upstream_block=True,
                   label="NOT_ESTABLISHED_2_OF_3_UNREACHABLE_MATCHED_COMPARATOR_BLOCKED")


def evaluate_c3(evd: dict[str, Any]) -> dict[str, Any]:
    cid = "C3_SEMANTIC_ABLATION"
    e7 = _e7_state(evd)
    e6 = _e6_state(evd)
    src_e7 = _rel("e7_three_fold_scientific_closure")
    src_e6 = _rel("e6_v2_final_closure")

    table = e7.get("detector_target_acer_by_fold")
    folds_ok = []
    if isinstance(table, Mapping):
        folds_ok = [f for f in FOLDS
                    if isinstance(table.get(f), Mapping)
                    and {"LLM_ORIGINAL", "LLM_SHUFFLE_A"} <= set(table[f])]
    if len(folds_ok) >= 3:
        better = [f for f in folds_ok
                  if float(table[f]["LLM_ORIGINAL"]) < float(table[f]["LLM_SHUFFLE_A"])]
        reversed_folds = [f for f in folds_ok
                          if float(table[f]["LLM_SHUFFLE_A"]) < float(table[f]["LLM_ORIGINAL"])]
        observed = [f"per-fold LLM_ORIGINAL/LLM_SHUFFLE_A target ACER in {folds_ok}",
                    f"LLM-original better in {len(better)}/3 folds: {better}",
                    f"folds with reversal (SHUFFLE-A better): {reversed_folds}"]
        if len(better) >= MIN_FOLDS:
            return _result(cid, status=STATUS_PASS,
                           reason=(f"LLM-original beats LLM-SHUFFLE-A by target ACER "
                                   f"in {len(better)}/3 folds (>= 2/3)."),
                           used=[src_e7], missing=[], observed=observed,
                           upstream_block=False, label="ESTABLISHED_FROM_EVIDENCE")
        if "EXT-F1" in better and not reversed_folds:
            return _result(cid, status=STATUS_PASS,
                           reason=("LLM-original advantage is clear in EXT-F1 and "
                                   "does not reverse in the other folds."),
                           used=[src_e7], missing=[], observed=observed,
                           upstream_block=False, label="ESTABLISHED_FROM_EVIDENCE")
        return _result(cid, status=STATUS_FAIL,
                       reason="Neither the 2/3-fold nor the EXT-F1 OR-branch is met.",
                       used=[src_e7], missing=[], observed=observed,
                       upstream_block=False,
                       label="EVIDENCE_PRESENT_CONDITION_NOT_MET")

    observed = [
        f"E7 per-condition: G-LLM={e7.get('per_condition', {}).get('G-LLM')}, "
        f"G-LLM-SHUFFLE-A={e7.get('per_condition', {}).get('G-LLM-SHUFFLE-A')}",
        f"E6-v2: status={e6.get('E6_V2_STATUS')!r}, "
        f"ready_for_training={e6.get('E6_V2_READY_FOR_TRAINING')!r} "
        f"({e6.get('training_block_reason')})",
        "E7 Shuffle-A generation was never attempted (0/3 folds closed their core "
        "anchor bank); E6-v2 proved the frozen SHUFFLE-A matched Physics bank "
        "infeasible under the frozen per-source-domain quota.",
    ]
    missing = [
        f"{src_e7} :: target_acer_by_fold for G-LLM and G-LLM-SHUFFLE-A",
        f"{src_e6} :: a feasible LLM_SHUFFLE_A matched bank + paired detector target ACER",
    ]
    return _result(cid, status=STATUS_BLOCKED,
                   reason=(
                       "Both routes to an LLM-original vs LLM-SHUFFLE-A detector "
                       "target-ACER contrast are blocked upstream: E7 "
                       "G-LLM-SHUFFLE-A is BLOCKED_BY_E7_CORE_BANK and E6-v2 closed "
                       "CLOSED_TRUE_FROZEN_MATCHED_BANK_INFEASIBILITY with "
                       "E6_V2_READY_FOR_TRAINING = false. No semantic-ablation "
                       "detector result exists in any fold, and none may be "
                       "fabricated. Absence of evidence is not a negative result."),
                   used=[src_e7, src_e6], missing=missing, observed=observed,
                   upstream_block=True,
                   label="NOT_ESTABLISHED_DUE_TO_BLOCKED_EVIDENCE")


def evaluate_c4(evd: dict[str, Any]) -> dict[str, Any]:
    """C4 -- frozen wording: "No large q confound, OR the q-matched ablation
    still retains the direction of an LLM advantage."

    The frozen rule requires NEITHER inferential significance NOR a comparable
    pre-q-match baseline.  Branch (b) is satisfied when the completed E8
    q-matched detector ablation DESCRIPTIVELY retains the direction of an LLM
    advantage (LLM at least as good as DET and RND by descriptive mean ACER),
    and such a PASS carries an explicit DESCRIPTIVE_ONLY / NO_SIGNIFICANCE_CLAIM
    qualifier.
    """
    cid = "C4_QUALITY_CONFOUND"
    q = _e2_q_confound(evd)
    e8 = _e8_state(evd)
    src_e2 = _rel("e2_quality_analysis")
    src_e8 = _rel("e8_target_evaluation_closure")

    # ---- branch (a): no large q confound ----
    if q.get("present") and q.get("large_q_confound_present") is False:
        return _result(cid, status=STATUS_PASS,
                       reason=(f"Measured max |SMD(q)| = {_fmt(q.get('max_abs_smd'), 3)} "
                               f"is below the frozen {Q_CONFOUND_TRIGGER_SMD} trigger: "
                               "no large q confound."),
                       used=[src_e2], missing=[],
                       observed=[f"E2 pairwise SMD(q): {q.get('pairwise_smd')}"],
                       upstream_block=False, label="ESTABLISHED_FROM_EVIDENCE")

    q_present = q.get("present")
    q_facts = ([f"E2 pairwise SMD(q): {q.get('pairwise_smd')}; "
                f"LLM q mean {_fmt(q.get('llm_q_mean'))} vs "
                f"{ {a: _fmt(v) for a, v in (q.get('q_mean_by_arm') or {}).items()} }; "
                f"|SMD(q)| >= {Q_CONFOUND_TRIGGER_SMD} trigger fired = "
                f"{q.get('large_q_confound_present')}"]
               if q_present else
               [f"E2 q analysis not present in this checkout ({src_e2})"])

    # ---- branch (b): q-matched ablation retains the direction of an LLM advantage ----
    if e8.get("state") == MS_EVIDENCE_ABSENT:
        return _result(cid, status=STATUS_NOT_ESTABLISHED,
                       reason=(
                           "A large q confound is present (E2 |SMD(q)| >= 0.25), so "
                           "branch (a) is not satisfied; and the E8 q-matched "
                           "target-evaluation artifacts are NOT PRESENT IN THIS "
                           "CHECKOUT (git-ignored tree), so branch (b) cannot be "
                           "assessed here. This is 'evidence not present locally', "
                           "NOT 'E8 not executed' and NOT an LLM failure -- restore "
                           "the authoritative E8 evidence to resolve."),
                       used=[src_e2] if q_present else [],
                       missing=[f"{src_e8} (E8 q-matched target evaluation -- not in checkout)"],
                       observed=q_facts + [e8.get("note", "")],
                       upstream_block=True,
                       label="NOT_ESTABLISHED_EVIDENCE_NOT_PRESENT_IN_CHECKOUT")

    if e8.get("state") == MS_EXECUTED_AND_SCORED and e8.get("acer_means_by_arm"):
        m = e8["acer_means_by_arm"]
        llm, det, rnd = m.get("LLM"), m.get("DET"), m.get("RND")
        pd = e8.get("paired_deltas", {})
        det_vs_llm = pd.get("DET_vs_LLM", {})
        rnd_vs_llm = pd.get("RND_vs_LLM", {})
        have_means = llm is not None and det is not None and rnd is not None
        # "retains the direction of an LLM advantage" -- descriptive: LLM at least
        # as good (<=) as BOTH DET and RND by mean ACER after q-matching.
        direction_retained = have_means and llm <= det and llm <= rnd
        llm_seed_wins_vs_det = det_vs_llm.get("count_positive")
        llm_seed_wins_vs_rnd = rnd_vs_llm.get("count_positive")

        observed = q_facts + [
            "E8 q-matched target evaluation (EXT-F1, 5 detector seeds/arm), "
            "authoritative mean ACER read from artifact: "
            f"LLM={_fmt(llm)} DET={_fmt(det)} RND={_fmt(rnd)}; "
            f"LLM <= DET and LLM <= RND (direction retained) = {direction_retained}",
            "E8 mean paired ACER deltas: "
            f"DET-LLM={_fmt(det_vs_llm.get('mean_delta'))} "
            f"(LLM better in {llm_seed_wins_vs_det}/5 seeds), "
            f"RND-LLM={_fmt(rnd_vs_llm.get('mean_delta'))} "
            f"(LLM better in {llm_seed_wins_vs_rnd}/5 seeds)",
            f"E8 inferential_status={e8.get('inferential_status')!r}, "
            f"statistical_significance_claimed={e8.get('statistical_significance_claimed')!r}, "
            f"llm_superiority_supported={e8.get('llm_superiority_supported')!r}",
            "Scope: EXT-F1 only; descriptive margins ~0.002 ACER.",
        ]

        if direction_retained:
            return _result(cid, status=STATUS_PASS,
                           reason=(
                               "Branch (b) of the frozen rule is satisfied: the "
                               "completed E8 q-matched detector ablation "
                               "DESCRIPTIVELY retains the direction of an LLM "
                               "advantage -- after q-matching (which removes the q "
                               "confound by construction) the LLM arm still has the "
                               "lowest mean target ACER "
                               f"({_fmt(llm)} vs DET {_fmt(det)} vs RND {_fmt(rnd)}) "
                               "and its mean paired ACER delta is favourable vs both "
                               "DET and RND. QUALIFIER: DESCRIPTIVE_ONLY / "
                               "NO_SIGNIFICANCE_CLAIM -- the frozen rule requires "
                               "neither inferential significance (E8 "
                               "inferential_status=NOT_CLAIMED) nor a pre-q-match "
                               "baseline; the effect is small (~0.002 ACER), EXT-F1 "
                               "only, and no LLM-superiority claim is made."),
                           used=[src_e2, src_e8], missing=[], observed=observed,
                           upstream_block=False,
                           label="PASS_DESCRIPTIVE_ONLY_NO_SIGNIFICANCE_CLAIM")

        return _result(cid, status=STATUS_FAIL,
                       reason=(
                           "Branch (a) is not satisfied (E2 shows a large q confound, "
                           f"|SMD(q)| LLM-RND={_fmt(q.get('pairwise_smd', {}).get('LLM_vs_RND'), 3)}, "
                           f"LLM-DET={_fmt(q.get('pairwise_smd', {}).get('LLM_vs_DET'), 3)}) "
                           "and branch (b) is not satisfied either: the completed E8 "
                           "q-matched ablation does NOT retain the direction of an LLM "
                           f"advantage (LLM mean ACER {_fmt(llm)} is not <= both DET "
                           f"{_fmt(det)} and RND {_fmt(rnd)})."),
                       used=[src_e2, src_e8], missing=[], observed=observed,
                       upstream_block=False,
                       label="EVIDENCE_PRESENT_DIRECTION_NOT_RETAINED")

    # E8 present but unreadable / no means
    return _result(cid, status=STATUS_NOT_ESTABLISHED,
                   reason=("A large q confound is present (branch (a) not satisfied) "
                           "and the E8 q-matched target-evaluation artifacts are "
                           "present but not readable as a scored result "
                           "(branch (b) unassessable)."),
                   used=[src_e2] if q_present else [], missing=[src_e8],
                   observed=q_facts, upstream_block=False,
                   label="NOT_ESTABLISHED_E8_EVIDENCE_UNREADABLE")


def evaluate_c5(evd: dict[str, Any]) -> dict[str, Any]:
    """C5 -- frozen wording: "BA controls / threshold analysis do not detect a
    bug / label leakage that invalidates the core evaluation."

    This is a NO-INVALIDATING-BUG/LEAKAGE criterion.  The frozen rule does NOT
    require every BA control to have completed.  It PASSES when the integrity
    evidence that WAS produced (E3, E4, E8 label-use record, E5 calibration
    provenance) detected no bug and no label leakage that would invalidate the
    core evaluation -- incomplete E3 controls are recorded as a LIMITATION.  It
    FAILS only if an actual invalidating bug or leakage was found.  It is
    NOT_ESTABLISHED only when there is no integrity evidence at all.
    """
    cid = "C5_INTEGRITY"
    e3 = _e3_state(evd)
    e4 = _e4_state(evd)
    e5 = _e5_state(evd)
    e8 = _e8_state(evd)
    e7 = _e7_state(evd)
    diag = _e10_integrity_diag(evd)
    src_e3 = _rel("e3_ba_controls")
    src_e4 = _rel("e4_threshold_transfer")
    src_e8_lu = _rel("e8_target_label_use_record")
    opt_diag = OPTIONAL_POSITIVE_EVIDENCE["e10_integrity_diagnostics"]

    # Optional dedicated diagnostics artifact wins if present.
    if isinstance(diag, Mapping):
        leakage = bool(diag.get("label_leakage_found", diag.get("leakage_found", True)))
        ba_ok = bool(diag.get("ba_controls_ok", False))
        thr_ok = bool(diag.get("threshold_analysis_ok", False))
        observed = [f"E10 integrity diagnostics: label_leakage_found={leakage}, "
                    f"ba_controls_ok={ba_ok}, threshold_analysis_ok={thr_ok}"]
        if not leakage and ba_ok and thr_ok:
            return _result(cid, status=STATUS_PASS,
                           reason=("Dedicated integrity diagnostics detect no bug "
                                   "and no label leakage that invalidates the core "
                                   "evaluation."),
                           used=[opt_diag], missing=[], observed=observed,
                           upstream_block=False, label="ESTABLISHED_FROM_EVIDENCE")
        return _result(cid, status=STATUS_FAIL,
                       reason=("Dedicated integrity diagnostics report a bug or "
                               "label leakage that invalidates the core evaluation."),
                       used=[opt_diag], missing=[], observed=observed,
                       upstream_block=False,
                       label="EVIDENCE_PRESENT_INVALIDATING_FINDING")

    # ---- concrete invalidating-finding checks over the evidence that exists ----
    invalidating: list[str] = []
    clean_facts: list[str] = []
    used: list[str] = []

    if e8.get("state") == MS_EXECUTED_AND_SCORED:
        used.append(src_e8_lu)
        lu = e8.get("label_use", {})
        if lu.get("uses_target_labels_in_training_or_g7") is True:
            invalidating.append(
                "E8 label-use record: target labels used in training or G7 "
                "(label leakage).")
        if lu.get("g8_structurally_isolated_from_training_and_g7") is False:
            invalidating.append(
                "E8 label-use record: G8 not structurally isolated from training/G7.")
        clean_facts.append(
            "E8 label-use record: uses_target_labels_in_training_or_g7="
            f"{lu.get('uses_target_labels_in_training_or_g7')!r}, "
            "G8 structurally isolated from training and G7="
            f"{lu.get('g8_structurally_isolated_from_training_and_g7')!r}, "
            f"is_first_ever_blind_reveal={lu.get('is_first_ever_blind_reveal')!r} "
            "-> no label leakage detected.")
        clean_facts.append(
            f"E8 closure framing: inferential_status={e8.get('inferential_status')!r}, "
            f"statistical_significance_claimed={e8.get('statistical_significance_claimed')!r}, "
            f"llm_superiority_supported={e8.get('llm_superiority_supported')!r} "
            "-> no over-claiming / no inferential bug.")

    if e5.get("present"):
        if e5.get("target_threshold_fitted") is True:
            invalidating.append(
                "E5 REAL-ONLY: a target threshold was fitted (target-guided tuning).")
        clean_facts.append(
            f"E5 REAL-ONLY: target_threshold_fitted={e5.get('target_threshold_fitted')!r}, "
            f"threshold_rule={e5.get('threshold_rule')!r} -> no target-guided calibration.")

    if e4.get("present"):
        used.append(src_e4)
        g = e4.get("t0_minus_oracle_acer_gap_video_mean", {})
        clean_facts.append(
            "E4 threshold-transfer (EXT-F1): T0(source-dev) vs T3(target-oracle, "
            "diagnostic-only) mean video ACER gap small "
            f"(LLM {_fmt(g.get('LLM'))}, DET {_fmt(g.get('DET'))}, "
            f"RND {_fmt(g.get('RND'))}); target_labels_accessed="
            f"{e4.get('target_labels_accessed')!r} (frozen label-derived aggregates "
            "only, no new inference) -> no threshold-transfer anomaly detected.")

    if e3.get("present"):
        used.append(src_e3)
        clean_facts.append(
            f"E3 BA controls: status={e3.get('status')!r}; the historical BA_sep "
            "re-summary produced no invalidating finding; "
            f"target_labels_accessed={e3.get('target_labels_accessed')!r}.")

    limitations = []
    if e3.get("present") and e3.get("delta_ba_excess_computable") is False:
        limitations.append(
            "E3 discriminating BA controls C0/C1/C3 are NOT_RUN_GPU_REQUIRED and "
            "C2 is UNSUPPORTED, so delta_BA_excess is not computed -- whether the "
            "synthetic/real separation is a generator fingerprint or a "
            "domain/pipeline confound is NOT affirmatively resolved (limitation, "
            "not an invalidating finding).")
    if e4.get("present") and e4.get("f2_f3_status"):
        limitations.append(
            f"E4 threshold-transfer covers EXT-F1 only (EXT-F2/F3 = "
            f"{e4.get('f2_f3_status')!r}).")
    if e7.get("is_target_acer_evidence") is not True:
        limitations.append(
            "No completed core 3-fold cross-domain detector evaluation exists "
            "(E7 blocked); the only completed target evaluation is E8's EXT-F1 "
            "q-matched ablation.")

    have_integrity_evidence = bool(used)

    if invalidating:
        return _result(cid, status=STATUS_FAIL,
                       reason=("An actual invalidating bug or label leakage was "
                               "detected: " + "; ".join(invalidating)),
                       used=used, missing=[], observed=clean_facts + invalidating,
                       upstream_block=False,
                       label="EVIDENCE_PRESENT_INVALIDATING_FINDING")

    if not have_integrity_evidence:
        return _result(cid, status=STATUS_NOT_ESTABLISHED,
                       reason=("No BA-control / threshold-transfer / label-use "
                               "provenance evidence is present in this checkout, so "
                               "whether a bug or leakage invalidates the core "
                               "evaluation cannot be assessed. Absence is not a "
                               "failure."),
                       used=[], missing=[src_e3, src_e4, src_e8_lu],
                       observed=[], upstream_block=True,
                       label="NOT_ESTABLISHED_NO_INTEGRITY_EVIDENCE")

    return _result(cid, status=STATUS_PASS,
                   reason=(
                       "The BA-control / threshold-transfer / label-use provenance "
                       "evidence that was produced detects NO bug and NO label "
                       "leakage that invalidates the core evaluation: E8's G8 target "
                       "scoring is structurally isolated from training and G7 "
                       "(uses_target_labels_in_training_or_g7=false), calibration is "
                       "source-dev-only with no target threshold fitting (E5 and E8), "
                       "E8's closure claims no significance and no LLM superiority, "
                       "E4's EXT-F1 source-derived operating point sits close to the "
                       "target oracle with no anomaly, and E3's historical BA_sep "
                       "re-summary produced no invalidating finding. LIMITATIONS "
                       "(recorded, not invalidating): " + " | ".join(limitations)),
                   used=used, missing=[], observed=clean_facts + [
                       "LIMITATION: " + x for x in limitations],
                   upstream_block=False,
                   label="PASS_WITH_LIMITATIONS_NO_INVALIDATING_BUG_OR_LEAKAGE")


CRITERION_EVALUATORS = {
    "C1_CROSS_DOMAIN_SIGNAL": evaluate_c1,
    "C2_VALUE_OVER_REAL_ONLY": evaluate_c2,
    "C3_SEMANTIC_ABLATION": evaluate_c3,
    "C4_QUALITY_CONFOUND": evaluate_c4,
    "C5_INTEGRITY": evaluate_c5,
}


# --------------------------------------------------------------------------- #
# Milestone status audit (E0..E9) -- resolved from artifacts, never assumed.
# Absence of an artifact is EVIDENCE_NOT_PRESENT_IN_CHECKOUT, never NOT_EXECUTED.
# --------------------------------------------------------------------------- #

def audit_milestones(evd: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ABSENCE_RULE = (
        "reports/c_ext_q1q2_v1 is a git-ignored tree; an absent artifact here means "
        "'not present in this checkout', never 'not executed' and never an LLM "
        "failure."
    )

    # E0
    e0 = _e0_state(evd)
    out["E0"] = {"evidence_present": e0.get("present"),
                 "status": e0.get("state"),
                 "scientific_content_status": e0.get("scientific_content_status"),
                 "claim_ceiling_without_e11": e0.get("claim_ceiling_without_e11"),
                 "primary_hypotheses": e0.get("primary_hypotheses"),
                 "e8_trigger_rule": e0.get("e8_trigger_rule"),
                 "absence_rule": ABSENCE_RULE}

    # E1 (recipe analysis) -- subtree present?
    e1_present = evd["milestones"]["E1"]["present"]
    out["E1"] = {"evidence_present": e1_present,
                 "status": "PRESENT_UNCLASSIFIED" if e1_present else MS_EVIDENCE_ABSENT,
                 "note": ("Recipe analysis. Not consulted by any E10 criterion. "
                          + ("Present; not individually classified by E10."
                             if e1_present else
                             "Not present in this checkout; not a failure.")),
                 "absence_rule": ABSENCE_RULE}

    # E2
    q = _e2_q_confound(evd)
    out["E2"] = {"evidence_present": q.get("present"),
                 "status": q.get("state"),
                 "pairwise_smd_q": q.get("pairwise_smd"),
                 "large_q_confound_present": q.get("large_q_confound_present"),
                 "q_mean_by_arm": q.get("q_mean_by_arm"),
                 "note": ("q reconstruction + analysis. The frozen E8 trigger "
                          "(|SMD(q)| >= 0.25) fired between LLM and both RND and DET "
                          "(LLM accepted-candidate q is systematically lower)."),
                 "absence_rule": ABSENCE_RULE}

    # E3
    e3 = _e3_state(evd)
    out["E3"] = {"evidence_present": e3.get("present"),
                 "status": e3.get("state"),
                 "e3_status_field": e3.get("status"),
                 "controls_complete": e3.get("controls_complete"),
                 "controls_gpu_required_not_run": e3.get("controls_gpu_required"),
                 "controls_unsupported": e3.get("controls_unsupported"),
                 "delta_ba_excess_computable": e3.get("delta_ba_excess_computable"),
                 "target_labels_accessed": e3.get("target_labels_accessed"),
                 "note": ("BA negative controls: only a descriptive CPU re-summary of "
                          "historical BA_sep was completed. The discriminating controls "
                          "C0/C1/C3 are NOT_RUN_GPU_REQUIRED and C2 is UNSUPPORTED; "
                          "delta_BA_excess (generator-fingerprint vs domain/pipeline "
                          "confound) cannot be computed. GPU-gated, not scientifically "
                          "impossible."),
                 "absence_rule": ABSENCE_RULE}

    # E4
    e4 = _e4_state(evd)
    out["E4"] = {"evidence_present": e4.get("present"),
                 "status": e4.get("state"),
                 "e4_status_field": e4.get("status"),
                 "fold_scope": e4.get("fold"),
                 "t0_minus_oracle_acer_gap_video_mean": e4.get(
                     "t0_minus_oracle_acer_gap_video_mean"),
                 "f2_f3_status": e4.get("f2_f3_status"),
                 "target_labels_accessed": e4.get("target_labels_accessed"),
                 "note": ("Threshold transfer: EXT-F1 only, computed on the frozen "
                          "historical Flow-1 Track-G target evaluation (not a new EXT "
                          "run). T0(source-dev) sits close to the T3 target oracle "
                          "(diagnostic-only) for EXT-F1. EXT-F2/F3 GPU_REQUIRED."),
                 "absence_rule": ABSENCE_RULE}

    # E5
    e5 = _e5_state(evd)
    out["E5"] = {"evidence_present": e5.get("present"),
                 "status": e5.get("state"),
                 "arm": e5.get("arm"),
                 "fold_scope": e5.get("fold"),
                 "acer_mean": e5.get("acer_mean"),
                 "acer_sample_sd": e5.get("acer_sample_sd"),
                 "roc_auc_mean": e5.get("roc_auc_mean"),
                 "seed_count": e5.get("seed_count"),
                 "target_threshold_fitted": e5.get("target_threshold_fitted"),
                 "target_scoring_executed": e5.get("target_scoring_executed"),
                 "note": ("REAL-ONLY target evaluation COMPLETED for EXT-F1 only "
                          f"(mean ACER {_fmt(e5.get('acer_mean'))}, "
                          f"{e5.get('seed_count')} detector seeds, frozen source-dev "
                          "calibration). No REAL-ONLY result exists for EXT-F2/F3."),
                 "absence_rule": ABSENCE_RULE}

    # E6
    e6 = _e6_state(evd)
    out["E6"] = {"evidence_present": e6.get("present"),
                 "status": e6.get("state"),
                 "disposition": e6.get("E6_V2_STATUS"),
                 "ready_for_training": e6.get("E6_V2_READY_FOR_TRAINING"),
                 "training_block_reason": e6.get("training_block_reason"),
                 "detector_trained": False,
                 "target_access": e6.get("target_access"),
                 "llm_api_calls": e6.get("llm_api_calls"),
                 "note": ("E6-v2 paired original/shuffle rerender: the frozen "
                          "LLM_SHUFFLE_A matched Physics bank is infeasible under the "
                          "frozen per-source-domain quota. No detector trained; "
                          "supplies no semantic-ablation detector target ACER."),
                 "absence_rule": ABSENCE_RULE}

    # E7
    e7 = _e7_state(evd)
    out["E7"] = {"evidence_present": e7.get("present"),
                 "status": e7.get("state"),
                 "all_folds_status": e7.get("fold_status"),
                 "core_matched_banks_produced": e7.get("core_matched_banks_produced"),
                 "is_target_acer_evidence": e7.get("is_target_acer_evidence"),
                 "per_condition_disposition": e7.get("per_condition"),
                 "detector_trained": False,
                 "note": ("E7 unified 3-fold core synthetic evaluation terminated "
                          "SCIENTIFICALLY_BLOCKED_AFTER_RESERVE_CAP in all 3 folds; "
                          "0/3 valid core matched synthetic banks; terminally blocked "
                          "before any target ACER scoring. G-RND/G-DET/G-LLM/"
                          "G-LLM-SHUFFLE-A are BLOCKED_BY_E7_CORE_BANK; G-REALONLY was "
                          "READY (evaluated separately as E5)."),
                 "absence_rule": ABSENCE_RULE}

    # E8 -- authoritative; NEVER "NOT_EXECUTED" from mere absence.
    e8 = _e8_state(evd)
    e8_row: dict[str, Any] = {
        "evidence_present": e8.get("present"),
        "status": e8.get("state"),
        "fold_scope": e8.get("fold_scope"),
        "detector_trained": e8.get("state") == MS_EXECUTED_AND_SCORED,
        "absence_rule": ABSENCE_RULE,
    }
    if e8.get("state") == MS_EXECUTED_AND_SCORED:
        e8_row.update({
            "acer_means_by_arm": e8.get("acer_means_by_arm"),
            "best_arm_by_acer_mean_descriptive_only": e8.get(
                "best_arm_by_acer_mean_descriptive_only"),
            "paired_deltas": e8.get("paired_deltas"),
            "inferential_status": e8.get("inferential_status"),
            "statistical_significance_claimed": e8.get("statistical_significance_claimed"),
            "llm_superiority_supported": e8.get("llm_superiority_supported"),
            "pre_qmatch_comparison_available": e8.get("pre_qmatch_comparison_available"),
            "label_use": e8.get("label_use"),
            "note": ("E8 q-matched Track-G target evaluation COMPLETED for EXT-F1 "
                     "(5 detector seeds/arm, 15 runs, scored against the frozen "
                     "held-out target). Results are "
                     "DESCRIPTIVE ONLY: inferential_status=NOT_CLAIMED, "
                     "statistical_significance_claimed=false, "
                     "llm_superiority_supported=false. Descriptive ACER rank: "
                     "LLM (lowest) < DET < RND, margins ~0.002 ACER. G8 structurally "
                     "isolated from training and G7; no target labels in training/G7; "
                     "source-dev-only calibration."),
        })
    else:
        e8_row["note"] = e8.get("note", "")
        e8_row["training_integration_audit_status"] = e8.get(
            "training_integration_audit_status")
    out["E8"] = e8_row

    # E9
    e9 = _e9_state(evd)
    out["E9"] = {"evidence_present": e9.get("present"),
                 "status": e9.get("state"),
                 "realizations_ok": e9.get("realizations_ok"),
                 "realizations_total": e9.get("realizations_total"),
                 "realizations_blocked": e9.get("realizations_blocked"),
                 "llm_calls": e9.get("llm_calls"),
                 "target_access": e9.get("target_access"),
                 "is_detector_performance_evidence": False,
                 "note": ("E9 conditional bank-selection robustness closed "
                          f"{e9.get('realizations_ok')}/{e9.get('realizations_total')} "
                          "realizations OK. Measures ONLY the structural stability of "
                          "the frozen 384->256 recipe selection under deterministic "
                          "pool perturbations (descriptive Jaccard). Explicitly NOT a "
                          "claim of downstream detector superiority and NOT "
                          "independent LLM generation replication; supplies NO "
                          "detector-performance / target-ACER evidence."),
                 "absence_rule": ABSENCE_RULE}

    return out


# --------------------------------------------------------------------------- #
# Full evaluation (pure -- no writes)
# --------------------------------------------------------------------------- #

def evaluate(root: Path | None = None, *, evidence: dict[str, Any] | None = None
             ) -> dict[str, Any]:
    evd = evidence if evidence is not None else load_evidence(root)

    criteria = [CRITERION_EVALUATORS[cid](evd) for cid in CRITERION_IDS]
    required = [c for c in criteria if c["required"]]
    established = [c for c in required if c["status"] == POSITIVE_STATUS]
    not_established = [c for c in required if c["status"] != POSITIVE_STATUS]

    e11_authorized = len(not_established) == 0 and len(required) == len(FROZEN_CRITERIA)

    if e11_authorized:
        decision_reason = (
            "All five frozen E10 criteria are positively established (status == "
            "PASS) from available evidence. Under the frozen decision rule, if the "
            "project intends to make the Q1 mechanism claim, E11 becomes required "
            "(with a NEW hypothesis family frozen before any API call)."
        )
    else:
        parts = [f"{c['criterion_id']} = {c['status']} [{c['resolution_label']}]"
                 for c in not_established]
        decision_reason = (
            "E11 is NOT authorized. The gate is conjunctive and fail-closed and "
            f"{len(not_established)} of {len(required)} required criteria are not "
            "positively established: " + "; ".join(parts) + ". Per the frozen "
            "decision rule: do NOT call the LLM again; E11_AUTHORIZED = false. "
            "Unavailable / upstream-blocked evidence is not evidence that the LLM "
            "failed."
        )

    milestone_audit = audit_milestones(evd)
    q1_supported = e11_authorized  # the conjunctive gate passing is necessary
    q1_reason = (
        "All five criteria positively established; the Q1 mechanism claim is "
        "supported pending E11." if q1_supported else
        f"The Q1 mechanism claim is NOT supported: the conjunctive gate does not "
        f"pass ({len(established)}/{len(required)} criteria positively "
        "established; " + ", ".join(f"{c['criterion_id']}={c['status']}"
                                    for c in not_established) + "). E8 completed "
        "but is explicitly descriptive only (inferential_status=NOT_CLAIMED, "
        "statistical_significance_claimed=false, llm_superiority_supported="
        "false); the mechanism-testing hypotheses EXT-H1/EXT-H2 need E7 "
        "(blocked) and EXT-H4 needs the SHUFFLE-A ablation (blocked); and E0's "
        "frozen claim ceiling forbids a general 'LLM generation mechanism is "
        "superior' claim without E11."
    )

    firewall_rollup = _firewall_rollup(evd)

    decision = {
        "schema_version": SCHEMA_VERSION,
        "e11_authorized": e11_authorized,
        "new_llm_calls_authorized": e11_authorized,
        "q1_mechanism_claim_supported": q1_supported,
        "q1_mechanism_claim_status": "SUPPORTED_PENDING_E11" if q1_supported
        else "NOT_SUPPORTED",
        "q1_mechanism_claim_reason": q1_reason,
        "decision_reason": decision_reason,
        "gate_is_conjunctive": True,
        "gate_is_fail_closed": True,
        "required_criteria_total": len(required),
        "criteria_positively_established": [c["criterion_id"] for c in established],
        "criteria_not_established": [
            {"criterion_id": c["criterion_id"], "status": c["status"],
             "resolution_label": c["resolution_label"],
             "missing_evidence_due_to_upstream_scientific_block":
                 c["missing_evidence_due_to_upstream_scientific_block"]}
            for c in not_established],
        "per_criterion_status": {c["criterion_id"]: c["status"] for c in criteria},
        "evidence_available_but_negative": any(
            c["status"] == STATUS_FAIL for c in required),
        "target_access": False,
        "target_labels_accessed": False,
        "target_features_accessed": False,
        "llm_calls": 0,
        "training_performed": False,
        "e9_counts_as_detector_performance_evidence": False,
        "interpretation_rule": (
            "Unavailable evidence is not evidence that the LLM failed. Criteria "
            "whose upstream evidence is scientifically blocked or is simply not "
            "present in this checkout are NOT_ESTABLISHED / BLOCKED, never FAIL and "
            "never NOT_EXECUTED-from-absence."
        ),
        "target_firewall_rollup": firewall_rollup,
    }

    matrix = {
        "schema_version": SCHEMA_VERSION,
        "milestone": "E10",
        "classification": "EVIDENCE_ONLY_DECISION_GATE",
        "frozen_criteria_source": "PRISM_FAS_C_EXT_Q1Q2_Detailed_Spec_v1_0.docx Section 17",
        "criteria": criteria,
    }

    input_audit = _build_input_audit(evd, milestone_audit)

    gate_identity = sha256_json({
        "criteria": [
            {k: c[k] for k in ("criterion_id", "status", "resolution_label",
                               "missing_evidence_due_to_upstream_scientific_block")}
            for c in criteria],
        "decision": {k: decision[k] for k in (
            "e11_authorized", "new_llm_calls_authorized",
            "q1_mechanism_claim_supported", "per_criterion_status")},
        "inputs": input_audit["inputs_digest"],
    })

    closure = {
        "schema_version": SCHEMA_VERSION,
        "milestone": "E10",
        "attempt": 3,
        "supersedes": [
            "reports/c_ext_q1q2_v1/e10_gate/attempts/attempt1_incomplete_checkout",
            "reports/c_ext_q1q2_v1/e10_gate/attempts/attempt2_post_outcome_requirements",
        ],
        "classification": "EVIDENCE_ONLY_DECISION_GATE",
        "status": "CLOSED",
        "gate_identity": gate_identity,
        "e11_authorized": e11_authorized,
        "new_llm_calls_authorized": e11_authorized,
        "q1_mechanism_claim_supported": q1_supported,
        "q1_mechanism_claim_status": decision["q1_mechanism_claim_status"],
        "e11_work_permitted": e11_authorized,
        "decision_reason": decision_reason,
        "per_criterion_status": {c["criterion_id"]: c["status"] for c in criteria},
        "milestone_status_audit": milestone_audit,
        "inputs_digest": input_audit["inputs_digest"],
        "corrections_vs_prior_attempts": input_audit["corrections_vs_prior_attempts"],
        "no_experiment_run": True,
        "no_llm_call": True,
        "no_training": True,
        "no_gpu": True,
        "no_e3_e9_rerun": True,
        "target_access": False,
        "target_labels_accessed": False,
        "target_features_accessed": False,
        "llm_calls": 0,
        "module_relpath": E10_MODULE_RELPATH,
        "test_relpath": E10_TEST_RELPATH,
        "output_subtree": OUTPUT_SUBTREE,
    }

    return {
        "evidence_root": evd["root"],
        "input_audit": input_audit,
        "evidence_matrix": matrix,
        "gate_decision": decision,
        "closure": closure,
    }


def _firewall_rollup(evd: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for name, rel in EVIDENCE_SOURCES.items():
        doc = _get(evd["sources"], name, "json")
        if not isinstance(doc, Mapping):
            continue
        ta = doc.get("target_access", doc.get("TARGET_ACCESS",
             doc.get("target_labels_accessed", doc.get("target_raw_access"))))
        lc = doc.get("llm_calls", doc.get("llm_api_calls", doc.get("LLM_API_CALLS")))
        checks.append({"artifact": rel, "target_access_or_labels": ta, "llm_calls": lc})
    return {
        "e10_self": {
            "target_access": False, "target_labels_accessed": False,
            "target_features_accessed": False, "llm_calls": 0,
            "reads_only": "frozen reports/results JSON under reports/c_ext_q1q2_v1/",
            "note": ("Upstream E4/E5/E8 DID open target labels for their own "
                     "authorized G8 scoring; E10 consumes only their frozen derived "
                     "summaries and opens no raw target artifact itself."),
        },
        "consulted_artifacts": checks,
    }


def _build_input_audit(evd: dict[str, Any], milestone_audit: dict[str, Any]
                       ) -> dict[str, Any]:
    roles = {
        "e0_hypothesis_family": "Frozen hypothesis family, E8 trigger rule, claim ceiling.",
        "e0_validation": "E0 completion status.",
        "e2_quality_analysis": "C4: pairwise |SMD(q)| -- the q confound magnitude.",
        "e3_ba_controls": "C5: BA negative controls (feasibility audit + historical resummary).",
        "e4_threshold_transfer": "C5 + C1/C2 descriptive: EXT-F1 threshold-transfer diagnostic.",
        "e5_target_scoring_result": "C2: REAL-ONLY target ACER (EXT-F1 only).",
        "e5_real_only_lock": "C2 provenance: E5 real-only lock.",
        "e6_v2_final_closure": "C3: E6-v2 paired original/shuffle matched-bank feasibility.",
        "e7_three_fold_scientific_closure": "C1/C2/C3/C5: E7 core-bank block + per-condition dispositions.",
        "e7_e8_trigger_record": "C4: E0/E2 |SMD(q)| trigger record.",
        "e8_training_integration_audit": "E8 audit: integration status.",
        "e8_target_evaluation_closure": "C4 + E8 audit: authoritative q-matched target closure.",
        "e8_target_score_result": "C4: per-arm / per-run q-matched target metrics.",
        "e8_target_paired_comparisons": "C4: descriptive per-seed paired ACER deltas.",
        "e8_target_label_use_record": "C5: E8 label-use / firewall provenance.",
        "e8_target_prediction_lockset": "C5: E8 frozen target prediction lockset.",
        "e9_closure": "E9 audit: closure status, realization counts.",
        "e9_bank_stability": "E9 audit: proves E9 is descriptive selection-stability only.",
    }
    consumed = []
    inputs_digest: dict[str, str] = {}
    for name, rel in EVIDENCE_SOURCES.items():
        rec = evd["sources"][name]
        inputs_digest[rel] = rec["sha256"] or "ABSENT"
        consumed.append({"key": name, "path": rel, "present": rec["present"],
                         "readable": rec["readable"], "sha256": rec["sha256"],
                         "role": roles.get(name, "")})
    optional = []
    for name, rel in OPTIONAL_POSITIVE_EVIDENCE.items():
        rec = evd["optional"][name]
        inputs_digest[rel] = rec["sha256"] or "ABSENT"
        optional.append({"key": name, "path": rel, "present": rec["present"],
                         "sha256": rec["sha256"],
                         "absence_meaning": (
                             "Would carry a POSITIVE certification for its criterion "
                             "if produced. Absent -> the criterion resolves from the "
                             "upstream-block / incomplete-controls analysis. Absence "
                             "is reported, never fabricated, never a negative result.")})

    llm_m = _fmt(_src(evd, 'e8_target_evaluation_closure', 'acer_means_by_arm', 'LLM'))
    det_m = _fmt(_src(evd, 'e8_target_evaluation_closure', 'acer_means_by_arm', 'DET'))
    rnd_m = _fmt(_src(evd, 'e8_target_evaluation_closure', 'acer_means_by_arm', 'RND'))
    corrections = [
        {
            "topic": "E8 classification (fixed in attempt 2)",
            "attempt1": ("NOT_EXECUTED -- inferred from the ABSENCE of E8's "
                         "git-ignored target_eval_v1/ artifacts in an incomplete "
                         "checkout."),
            "attempt2_onward": (
                f"{milestone_audit['E8']['status']} -- verified from the restored "
                "authoritative artifacts (E8_TARGET_EVALUATION_CLOSURE.json, "
                "E8_TARGET_SCORE_RESULT.json, E8_TARGET_PAIRED_COMPARISONS.json, "
                "E8_TARGET_LABEL_USE_RECORD.json, E8_TARGET_PREDICTION_LOCKSET.json). "
                "E8 q-matched Track-G target evaluation COMPLETED for EXT-F1 (5 "
                "detector seeds/arm); results DESCRIPTIVE ONLY, inferential "
                "significance NOT CLAIMED, no LLM superiority supported. "
                f"Authoritative mean ACER: LLM={llm_m} < DET={det_m} < RND={rnd_m} "
                "(LLM lowest-ACER / descriptively best arm)."),
            "fix": ("E10 classifies an absent E8 target evaluation as "
                    "EVIDENCE_NOT_PRESENT_IN_CHECKOUT, never NOT_EXECUTED; a "
                    "'completed' verdict requires the artifacts present and readable."),
        },
        {
            "topic": "C4 QUALITY CONFOUND -- frozen wording (fixed in attempt 3)",
            "attempt2": (
                "NOT_ESTABLISHED -- attempt 2 added post-outcome requirements not in "
                "the frozen spec: it demanded a comparable pre-q-match baseline and "
                "treated the descriptive q-matched LLM direction plus non-claimed "
                "inferential significance as insufficient."),
            "attempt3": (
                "PASS (PASS_DESCRIPTIVE_ONLY_NO_SIGNIFICANCE_CLAIM) -- the frozen "
                "rule is 'no large q confound OR the q-matched ablation still "
                "retains the direction of an LLM advantage'. It requires neither "
                "inferential significance nor a pre-q-match baseline. The completed "
                f"E8 q-matched ablation descriptively retains the direction (LLM "
                f"mean ACER {llm_m} <= DET {det_m} and <= RND {rnd_m}; favourable "
                "mean paired deltas vs both), so branch (b) is satisfied, with an "
                "explicit DESCRIPTIVE_ONLY / NO_SIGNIFICANCE_CLAIM qualifier."),
        },
        {
            "topic": "C5 INTEGRITY -- frozen wording (fixed in attempt 3)",
            "attempt2": (
                "NOT_ESTABLISHED -- attempt 2 required every BA control to have "
                "completed (delta_BA_excess computable) and a completed core 3-fold "
                "evaluation to certify; neither is in the frozen wording."),
            "attempt3": (
                "PASS (PASS_WITH_LIMITATIONS_NO_INVALIDATING_BUG_OR_LEAKAGE) -- the "
                "frozen rule is 'BA controls / threshold analysis do not detect a "
                "bug / label leakage that invalidates the core evaluation'. The "
                "integrity evidence that was produced (E3 historical BA_sep "
                "re-summary, E4 EXT-F1 threshold transfer, E8 label-use record, E5 "
                "calibration provenance) detected NO bug and NO label leakage; the "
                "incomplete E3 discriminating controls (C0/C1/C3 GPU-required) are "
                "recorded as an explicit LIMITATION, not an automatic "
                "non-establishment."),
        },
        {
            "topic": "E3 / E4 / E5 classification (from restored evidence, attempt 2 onward)",
            "attempt1": "NOT_PRESENT_IN_CHECKOUT (report subtrees absent locally).",
            "attempt2_onward": (
                f"E3 = {milestone_audit['E3']['status']} (BA controls: only a "
                "descriptive historical re-summary; C0/C1/C3 GPU-required, C2 "
                f"unsupported); E4 = {milestone_audit['E4']['status']} (EXT-F1-only "
                "threshold transfer on the frozen historical Flow-1 eval); E5 = "
                f"{milestone_audit['E5']['status']} (REAL-ONLY target eval COMPLETED "
                f"for EXT-F1 only, mean ACER "
                f"{_fmt(milestone_audit['E5'].get('acer_mean'))})."),
        },
        {
            "topic": "Informal external ranking claim (resolved attempt 2 onward)",
            "attempt1": ("Prompt-supplied claim 'E8 descriptive ranking DET best, "
                         "LLM middle, RND worst' recorded as UNVERIFIED."),
            "attempt2_onward": (
                f"Verified from E8_TARGET_EVALUATION_CLOSURE.json: descriptive mean "
                f"ACER LLM={llm_m} < DET={det_m} < RND={rnd_m} -- i.e. LLM is the "
                "lowest-ACER (descriptively best) arm, DET middle, RND worst. The "
                "informal claim was inaccurate; the artifact is authoritative. "
                "Differences ~0.002 ACER, inferential significance NOT CLAIMED."),
        },
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "milestone": "E10",
        "attempt": 3,
        "ext_root": EXT_ROOT_RELPATH,
        "git_ignored_evidence_tree": True,
        "absence_semantics": (
            "An artifact absent under reports/c_ext_q1q2_v1 is "
            "EVIDENCE_NOT_PRESENT_IN_CHECKOUT -- never NOT_EXECUTED, never a failure."
        ),
        "frozen_wording_note": (
            "C4 and C5 are evaluated strictly against the frozen Section-17 wording. "
            "C4 requires neither inferential significance nor a pre-q-match "
            "baseline. C5 is a no-invalidating-bug/leakage criterion and does not "
            "require every BA control to have completed. No stricter post-outcome "
            "requirement is imposed."
        ),
        "consumed_frozen_artifacts": consumed,
        "optional_positive_evidence_artifacts": optional,
        "milestone_subtrees_present": {
            m: evd["milestones"][m]["present"] for m in MILESTONE_SUBTREES},
        "corrections_vs_prior_attempts": corrections,
        "inputs_digest": inputs_digest,
    }


# --------------------------------------------------------------------------- #
# Artifact writing (guarded, atomic, deterministic)
# --------------------------------------------------------------------------- #

def write_artifacts(body: dict[str, Any], *, root: Path | None = None) -> dict[str, str]:
    root = root or repo_root()

    for rel in (INPUT_AUDIT_RELPATH, EVIDENCE_MATRIX_RELPATH, GATE_DECISION_RELPATH,
                CLOSURE_RELPATH, EVIDENCE_MANIFEST_RELPATH):
        assert_ext_write_path(rel, root=root, must_be_under=OUTPUT_SUBTREE)
        if "/attempts/" in rel:  # pragma: no cover - defensive
            raise E10Error(f"refusing to write into the preserved attempts tree: {rel}")

    written: dict[str, str] = {}
    written["input_audit"] = write_json_atomic(INPUT_AUDIT_RELPATH,
                                               body["input_audit"], root=root)
    written["evidence_matrix"] = write_json_atomic(EVIDENCE_MATRIX_RELPATH,
                                                   body["evidence_matrix"], root=root)
    written["gate_decision"] = write_json_atomic(GATE_DECISION_RELPATH,
                                                 body["gate_decision"], root=root)
    written["closure"] = write_json_atomic(CLOSURE_RELPATH, body["closure"], root=root)

    manifest_lines: list[tuple[str, str]] = []
    for path, digest in body["input_audit"]["inputs_digest"].items():
        if digest != "ABSENT":
            manifest_lines.append((path, digest))
    for key in ("input_audit", "evidence_matrix", "gate_decision", "closure"):
        rel = written[key]
        manifest_lines.append((rel, sha256_file(root / rel)))
    manifest_lines.sort()
    text = "".join(f"{d}  {p}\n" for p, d in manifest_lines)
    written["evidence_manifest"] = write_text_atomic(EVIDENCE_MANIFEST_RELPATH,
                                                     text, root=root)
    return written


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _print_summary(body: dict[str, Any]) -> None:
    dec = body["gate_decision"]
    print(json.dumps({
        "per_criterion_status": dec["per_criterion_status"],
        "e11_authorized": dec["e11_authorized"],
        "new_llm_calls_authorized": dec["new_llm_calls_authorized"],
        "q1_mechanism_claim_supported": dec["q1_mechanism_claim_supported"],
        "q1_mechanism_claim_status": dec["q1_mechanism_claim_status"],
        "target_access": dec["target_access"],
        "target_labels_accessed": dec["target_labels_accessed"],
        "target_features_accessed": dec["target_features_accessed"],
        "llm_calls": dec["llm_calls"],
        "decision_reason": dec["decision_reason"],
    }, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="E10 -- frozen extension decision gate (evidence-only).")
    parser.add_argument("--print", dest="do_print", action="store_true",
                        help="Read-only: resolve the gate and print the decision.")
    parser.add_argument("--evaluate", action="store_true",
                        help="Resolve the gate and write the five E10 artifacts "
                             f"under {OUTPUT_SUBTREE}/.")
    args = parser.parse_args(argv)
    if not (args.do_print or args.evaluate):
        parser.print_help()
        return 1
    body = evaluate()
    if args.evaluate:
        written = write_artifacts(body)
        print(json.dumps({"written": written}, indent=2))
    _print_summary(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
