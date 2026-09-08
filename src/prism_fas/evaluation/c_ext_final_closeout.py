"""PRISM-FAS-C EXT-Q1Q2 -- FINAL scientific closeout (after E10).

This module produces the additive final-closeout namespace

    reports/c_ext_q1q2_v1/final_closeout/

from the already-frozen E0..E10 closure / result artifacts.  It is a
READ-ONLY evidence synthesis: it runs no experiment, trains no detector,
calls no LLM / provider / network, uses no GPU, reruns nothing, and never
opens a raw target image / feature / label.  It reads frozen ``reports/``
JSON only, all of it under ``reports/c_ext_q1q2_v1/``, and it never writes
outside ``reports/c_ext_q1q2_v1/final_closeout/``.

Authoritative final state (independently re-derived here from the artifacts)
--------------------------------------------------------------------------
* E7  -- unified 3-fold Track-G experiment SCIENTIFICALLY BLOCKED; 0/3 core
  matched synthetic banks; terminated before any target ACER scoring; no E7
  target ACER evidence exists.
* E8  -- q-matched EXT-F1 Track-G target evaluation COMPLETED and scored
  (5 detector seeds/arm).  DESCRIPTIVE ONLY: inferential_status=NOT_CLAIMED,
  no statistical-significance claim, no general LLM-superiority claim.
* E9  -- CLOSED.  9/9 deterministic candidate-pool perturbation realizations.
  Bank-SELECTION robustness only (structural 256-of-384 selection stability
  within the frozen 384-candidate pools).  NOT detector-performance evidence;
  NOT independent LLM-generation replication.
* E10 -- CLOSED decision gate.  C1 BLOCKED, C2 NOT_ESTABLISHED, C3 BLOCKED,
  C4 PASS (descriptive only), C5 PASS (with limitations).  E11_AUTHORIZED=false,
  new_llm_calls_authorized=false, q1_mechanism_claim_supported=false.

Interpretation rule (mandatory, inherited from E10)
--------------------------------------------------
Unavailable / upstream-blocked evidence is NOT evidence that the LLM failed.
A blocked milestone is recorded BLOCKED / NOT_ESTABLISHED -- never FAIL, never
"negative result", never "LLM inferior".

Hard rules honoured by this module
----------------------------------
* Imports: the Python standard library and
  ``prism_fas.evaluation.c_ext_common`` ONLY.  Nothing that could pull in
  ``torch``, a provider SDK, a network client, ``prism_fas.detector``,
  ``prism_fas.synthesis`` or ``prism_fas.recipes``.
* No training / generation / inference capability whatsoever.
* Every write is routed through ``c_ext_common.assert_ext_write_path`` and is
  confined to ``reports/c_ext_q1q2_v1/final_closeout/``.
* Deterministic & idempotent: the output bytes are a pure function of the
  consumed artifacts' contents.  No wall-clock timestamp is serialized.
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

FINAL_ID = "final_closeout"
EXT_ROOT_RELPATH = "reports/c_ext_q1q2_v1"
OUTPUT_SUBTREE = f"{EXT_ROOT_RELPATH}/{FINAL_ID}"

MODULE_RELPATH = "src/prism_fas/evaluation/c_ext_final_closeout.py"
TEST_RELPATH = "tests/pipeline/test_c_ext_final_closeout.py"

SCHEMA_STATUS = "c-ext-q1q2-final-closeout-status-v1"
SCHEMA_CLAIM_MATRIX = "c-ext-q1q2-final-closeout-claim-matrix-v1"
SCHEMA_CLOSURE = "c-ext-q1q2-final-closure-v1"
RESULTS_TABLE_SCHEMA = "c-ext-q1q2-final-results-table-v1"

#: The closeout is defined ONLY at the E10 scientific-closure HEAD.  If the
#: working tree is at a different commit the closeout fails closed.
E10_CLOSURE_HEAD_COMMIT = "7723fc1444cd69a5c4db1fe73c4ee4edf96313dd"
#: git-log-only provenance anchors (no identity field inside the artifacts).
E7_EVIDENCE_PACKAGE_COMMIT = "86c32a2a870f4ceda47c26de30a161eb19550219"
E7_RESERVE_STAGE_CLOSE_COMMIT = "bf4d3ee6d1c95f380c225033899f7f75e9ebd88d"
E9_ARCHIVE_COMMIT = "ebff72596fe56bb77ed494a4aaba943135a98a0d"

FOLDS: tuple[str, ...] = ("EXT-F1", "EXT-F2", "EXT-F3")
ARMS: tuple[str, ...] = ("LLM", "DET", "RND")

E11_STATUS = "NOT_AUTHORIZED_NOT_RUN"
EXTENSION_STATUS = "CLOSED_WITH_BLOCKED_PRIMARY_MECHANISM_EVIDENCE"

# Claim-matrix classification vocabulary (frozen).
CLS_SUPPORTED = "SUPPORTED"
CLS_DESCRIPTIVE_ONLY = "DESCRIPTIVELY_SUPPORTED_ONLY"
CLS_NOT_SUPPORTED = "NOT_SUPPORTED"
CLS_NOT_ESTABLISHED = "NOT_ESTABLISHED"
CLS_BLOCKED = "BLOCKED_BY_PROTOCOL_FEASIBILITY"
CLASSIFICATIONS: tuple[str, ...] = (
    CLS_SUPPORTED, CLS_DESCRIPTIVE_ONLY, CLS_NOT_SUPPORTED,
    CLS_NOT_ESTABLISHED, CLS_BLOCKED,
)
#: classifications that assert the claim is (at least descriptively) carried.
AFFIRMATIVE_CLASSIFICATIONS: frozenset[str] = frozenset(
    {CLS_SUPPORTED, CLS_DESCRIPTIVE_ONLY})
#: evidence states that can NEVER be promoted to an affirmative classification.
NON_AFFIRMATIVE_EVIDENCE: frozenset[str] = frozenset({
    "BLOCKED", "NOT_PRESENT", "PARTIAL_UNREACHABLE",
    "DESCRIPTIVE_ONLY_GATE_NEGATIVE",
})

CONSULTED_SOURCES: dict[str, str] = {
    "e0_hypothesis_family": f"{EXT_ROOT_RELPATH}/e0/EXT_HYPOTHESIS_FAMILY.json",
    "e0_validation": f"{EXT_ROOT_RELPATH}/e0/EXT_E0_VALIDATION.json",
    "e0_protocol": f"{EXT_ROOT_RELPATH}/e0/EXT_PROTOCOL.json",
    "e1_recipe_analysis":
        f"{EXT_ROOT_RELPATH}/e1_recipe_analysis/E1_RECIPE_ANALYSIS.json",
    "e2_quality_analysis": f"{EXT_ROOT_RELPATH}/e2_quality/E2_QUALITY_ANALYSIS.json",
    "e3_ba_controls": f"{EXT_ROOT_RELPATH}/e3_ba_controls/E3_BA_CONTROLS.json",
    "e4_threshold_transfer":
        f"{EXT_ROOT_RELPATH}/e4_threshold_transfer/E4_THRESHOLD_TRANSFER.json",
    "e5_target_scoring_result":
        f"{EXT_ROOT_RELPATH}/e5_realonly/target_scoring/E5_TARGET_SCORING_RESULT.json",
    "e5_real_only_lock": f"{EXT_ROOT_RELPATH}/e5_realonly/E5_REAL_ONLY_LOCK.json",
    "e6_v2_final_closure":
        f"{EXT_ROOT_RELPATH}/e6_paired_current_runtime_v2/E6_V2_FINAL_CLOSURE.json",
    "e7_three_fold_scientific_closure":
        f"{EXT_ROOT_RELPATH}/e7_three_fold/gpat_bank/e7_v1_1_reserve/"
        "E7_V1_1_THREE_FOLD_SCIENTIFIC_CLOSURE.json",
    "e7_e8_trigger_record":
        f"{EXT_ROOT_RELPATH}/e7_three_fold/E7_E8_TRIGGER_RECORD.json",
    "e8_target_evaluation_closure":
        f"{EXT_ROOT_RELPATH}/e8_qmatched/target_eval_v1/E8_TARGET_EVALUATION_CLOSURE.json",
    "e8_target_score_result":
        f"{EXT_ROOT_RELPATH}/e8_qmatched/target_eval_v1/E8_TARGET_SCORE_RESULT.json",
    "e8_target_paired_comparisons":
        f"{EXT_ROOT_RELPATH}/e8_qmatched/target_eval_v1/E8_TARGET_PAIRED_COMPARISONS.json",
    "e8_target_label_use_record":
        f"{EXT_ROOT_RELPATH}/e8_qmatched/target_eval_v1/E8_TARGET_LABEL_USE_RECORD.json",
    "e9_closure": f"{EXT_ROOT_RELPATH}/e9_bank_robustness/E9_CLOSURE.json",
    "e9_bank_stability": f"{EXT_ROOT_RELPATH}/e9_bank_robustness/E9_BANK_STABILITY.json",
    "e10_closure": f"{EXT_ROOT_RELPATH}/e10_gate/E10_CLOSURE.json",
    "e10_gate_decision": f"{EXT_ROOT_RELPATH}/e10_gate/E10_GATE_DECISION.json",
    "e10_evidence_matrix": f"{EXT_ROOT_RELPATH}/e10_gate/E10_EVIDENCE_MATRIX.json",
    "e10_input_audit": f"{EXT_ROOT_RELPATH}/e10_gate/E10_INPUT_AUDIT.json",
}

STATUS_RELPATH = f"{OUTPUT_SUBTREE}/C_EXT_FINAL_STATUS.json"
RESULTS_TABLE_RELPATH = f"{OUTPUT_SUBTREE}/C_EXT_FINAL_RESULTS_TABLE.csv"
CLAIM_MATRIX_RELPATH = f"{OUTPUT_SUBTREE}/C_EXT_FINAL_CLAIM_MATRIX.json"
PAPER_READY_RELPATH = f"{OUTPUT_SUBTREE}/C_EXT_PAPER_READY_SUMMARY.md"
CLOSURE_RELPATH = f"{OUTPUT_SUBTREE}/C_EXT_FINAL_CLOSURE.json"
EVIDENCE_MANIFEST_RELPATH = f"{OUTPUT_SUBTREE}/C_EXT_FINAL_EVIDENCE.sha256"

OUTPUT_RELPATHS: tuple[str, ...] = (
    STATUS_RELPATH, RESULTS_TABLE_RELPATH, CLAIM_MATRIX_RELPATH,
    PAPER_READY_RELPATH, CLOSURE_RELPATH, EVIDENCE_MANIFEST_RELPATH,
)


class FinalCloseoutError(RuntimeError):
    """A closeout precondition failed; fail closed rather than emit a wrong closeout."""


# --------------------------------------------------------------------------- #
# Input loading
# --------------------------------------------------------------------------- #

def _head_commit(root: Path) -> str | None:
    """Best-effort HEAD resolution from the git plumbing (no subprocess)."""
    dotgit = root / ".git"
    try:
        if dotgit.is_file():
            gitdir = Path(dotgit.read_text(encoding="utf-8").split("gitdir:", 1)[1].strip())
            if not gitdir.is_absolute():
                gitdir = (root / gitdir).resolve()
        else:
            gitdir = dotgit
        head = (gitdir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head or None
        ref = head[5:].strip()
        for base in (gitdir, gitdir.parent.parent if gitdir.name else gitdir):
            cand = base / ref
            if cand.is_file():
                return cand.read_text(encoding="utf-8").strip()
        commondir = gitdir / "commondir"
        if commondir.is_file():
            common = (gitdir / commondir.read_text(encoding="utf-8").strip()).resolve()
            cand = common / ref
            if cand.is_file():
                return cand.read_text(encoding="utf-8").strip()
            packed = common / "packed-refs"
            if packed.is_file():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if line and not line.startswith(("#", "^")) and line.endswith(ref):
                        return line.split(" ", 1)[0]
    except (OSError, IndexError):
        return None
    return None


def load_inputs(root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    inputs: dict[str, Any] = {}
    missing: list[str] = []
    for name, rel in CONSULTED_SOURCES.items():
        abs_path = root / rel
        if not abs_path.is_file():
            missing.append(rel)
            continue
        inputs[name] = read_json(abs_path)
    if missing:
        e1_rel = CONSULTED_SOURCES["e1_recipe_analysis"]
        hint = ""
        if e1_rel in missing:
            hint = (" NOTE: the E1 recipe-analysis evidence is a required authoritative "
                    "input -- if its git-ignored subtree exists on disk it MUST be read "
                    "and audited, never silently recorded NOT_PRESENT_IN_THIS_CHECKOUT; "
                    "restore reports/c_ext_q1q2_v1/e1_recipe_analysis/ before generating "
                    "the closeout.")
        raise FinalCloseoutError(
            "authoritative closeout input(s) not present in this checkout: "
            + ", ".join(sorted(missing))
            + " -- the final closeout may only be generated where every E0..E10 "
            "closure/analysis artifact is available." + hint)
    resolved_head = _head_commit(root)
    if resolved_head is not None and resolved_head != E10_CLOSURE_HEAD_COMMIT:
        raise FinalCloseoutError(
            f"working tree HEAD {resolved_head} != E10 scientific-closure HEAD "
            f"{E10_CLOSURE_HEAD_COMMIT}; the final closeout is defined only at the "
            "E10 closure commit.")
    inputs["_head_commit"] = resolved_head or E10_CLOSURE_HEAD_COMMIT
    inputs["_head_commit_resolved"] = resolved_head is not None
    inputs["_source_sha256"] = {
        rel: sha256_file(root / rel) for rel in CONSULTED_SOURCES.values()
    }
    return inputs


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def _f(x: Any) -> float:
    return round(float(x), 6)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise FinalCloseoutError(msg)


# --------------------------------------------------------------------------- #
# 1. C_EXT_FINAL_STATUS.json
# --------------------------------------------------------------------------- #

def build_final_status(inp: Mapping[str, Any]) -> dict[str, Any]:
    e7 = inp["e7_three_fold_scientific_closure"]
    e8c = inp["e8_target_evaluation_closure"]
    e9 = inp["e9_closure"]
    e10c = inp["e10_closure"]
    e10d = inp["e10_gate_decision"]

    e7_fsi = e7["final_scientific_interpretation"]
    _require(e7_fsi["core_matched_banks_produced"].replace(" ", "") == "0/3",
             "E7 core_matched_banks_produced is not '0 / 3' -- artifact changed.")
    _require(e7_fsi["is_target_acer_evidence"] is False,
             "E7 is_target_acer_evidence is not False -- artifact changed.")
    _require(e8c["inferential_status"] == "NOT_CLAIMED"
             and e8c["statistical_significance_claimed"] is False
             and e8c["llm_superiority_supported"] is False,
             "E8 closure no longer descriptive-only -- artifact changed.")
    _require(int(e9["realizations_ok"]) == 9 and int(e9["realizations_total"]) == 9
             and int(e9["realizations_blocked"]) == 0,
             "E9 realizations are not 9/9 OK -- artifact changed.")
    _require(e10d["e11_authorized"] is False
             and e10d["new_llm_calls_authorized"] is False
             and e10c["q1_mechanism_claim_supported"] is False,
             "E10 decision is no longer gate-negative -- artifact changed.")

    per_criterion = dict(e10d["per_criterion_status"])
    _require(per_criterion == {
        "C1_CROSS_DOMAIN_SIGNAL": "BLOCKED",
        "C2_VALUE_OVER_REAL_ONLY": "NOT_ESTABLISHED",
        "C3_SEMANTIC_ABLATION": "BLOCKED",
        "C4_QUALITY_CONFOUND": "PASS",
        "C5_INTEGRITY": "PASS",
    }, "E10 per-criterion status is not the authoritative frozen set.")

    e5 = inp["e5_target_scoring_result"]
    e5_acer = _f(e5["aggregate"]["acer"]["mean"])

    # E1 recipe analysis -- re-audited strictly from the restored artifact.
    e1 = inp["e1_recipe_analysis"]
    _require(e1.get("milestone") == "E1",
             "E1_RECIPE_ANALYSIS.json milestone field is not 'E1' -- wrong artifact.")
    e1_status_field = str(e1.get("status", ""))
    _require(e1_status_field.startswith("COMPLETE"),
             f"E1_RECIPE_ANALYSIS.json status={e1_status_field!r} is not a COMPLETE* "
             "state -- the restored E1 evidence is not a completed analysis.")
    _require(e1.get("gpu_used") is False and int(e1.get("llm_api_calls", -1)) == 0
             and e1.get("target_labels_accessed") is False,
             "E1_RECIPE_ANALYSIS.json integrity flags changed (gpu/llm/target).")
    e1_rows = e1.get("row_counts", {})
    e1_urr = e1.get("unique_recipe_ratio", {})
    e1_gower = e1.get("pairwise_gower_distance", {})

    milestones = {
        "E0": {
            "status": "COMPLETE",
            "kind": "protocol / hypothesis-family / binding freeze",
            "reason": (
                "EXT_E0_VALIDATION.json milestone_status=COMPLETE, "
                "e0_scientific_content_status=COMPLETE_AND_LOCKED; frozen hypothesis "
                "family EXT-H1..EXT-H4, E8 trigger rule, and the claim ceiling "
                "(no general 'LLM generation mechanism is superior' claim without E11)."),
        },
        "E1": {
            "status": "COMPLETE",
            "artifact_status_field": e1_status_field,
            "kind": "recipe-bank structural analysis (CPU, descriptive)",
            "reason": (
                "Re-audited from the restored authoritative artifact "
                "reports/c_ext_q1q2_v1/e1_recipe_analysis/E1_RECIPE_ANALYSIS.json "
                f"(status={e1_status_field}, gpu_used=false, llm_api_calls=0, "
                "target_labels_accessed=false). All 12 planned structural metrics were "
                "generated for the 3 frozen 256-recipe selected banks (RND/DET/LLM), "
                "including the RND/DET raw->selected distribution-shift submetric that a "
                "prior attempt (history/pre_pydantic_blocked_20260831/) had left "
                "BLOCKED_TOOL -- it was later completed by deterministic reconstruction "
                "of the RND/DET raw-384 pools, verified byte-identical to the frozen "
                "schedule contract. Findings are STRUCTURAL statistics of the frozen "
                "recipe banks only: per E1's own scientific_interpretation_boundary they "
                "do NOT measure semantic plausibility or a causal role of LLM reasoning, "
                "and a semantic-mechanism reading is supported only if E6 LLM-SHUFFLE "
                "reduces performance downstream (E6 is blocked). E1 is a diagnostic "
                "(not in the Holm family), upstream of every target metric, and is not "
                "consulted by any E10 gate criterion."),
            "e10_reconciliation": (
                "E10's own milestone_status_audit records E1 as "
                "EVIDENCE_NOT_PRESENT_IN_CHECKOUT because the git-ignored E1 subtree was "
                "absent from the E10 checkout. That does not change any E10 gate "
                "criterion (none consult E1). This closeout re-audits E1 from the now-"
                "restored artifact and records COMPLETE; the disagreement is recorded, "
                "not silently reconciled."),
            "structural_facts": {
                "selected_recipes_per_arm": e1_rows,
                "unique_recipe_ratio": {
                    a: e1_urr.get(a, {}).get("unique_ratio") for a in ARMS},
                "pairwise_gower_distance_mean": {
                    a: _f(e1_gower.get(a, {}).get("mean")) for a in ARMS
                    if e1_gower.get(a, {}).get("mean") is not None},
            },
            "not_a_failure": True,
            "not_consulted_by_any_e10_criterion": True,
        },
        "E2": {
            "status": "COMPLETE",
            "kind": "q-distribution reconstruction + confound analysis",
            "reason": (
                "E2_QUALITY_ANALYSIS.json status=COMPLETE_CPU. The frozen E0 E8 "
                "trigger |SMD(q)|>=0.25 fired: LLM-vs-RND SMD(q)=-0.326 and "
                "LLM-vs-DET SMD(q)=-0.364 (LLM accepted-candidate q systematically "
                "lower). No target labels accessed."),
        },
        "E3": {
            "status": "FEASIBILITY_AUDIT_ONLY_GPU_GATED",
            "kind": "BA-separation negative controls",
            "reason": (
                "Only the CPU-safe historical BA_sep re-summary (ddof=1) completed. "
                "The discriminating controls C0_same_dataset_null, C1_cross_domain_real "
                "and C3_matched_domain_syn_real are NOT_RUN_GPU_REQUIRED; C2_pipeline_noop "
                "is UNSUPPORTED (no verified renderer no-op path). delta_BA_excess "
                "cannot be computed. GPU-gated, not scientifically impossible. No "
                "invalidating finding in the evidence that was produced."),
            "blocked_components": [
                "C0_same_dataset_null (GPU_REQUIRED)",
                "C1_cross_domain_real (GPU_REQUIRED)",
                "C3_matched_domain_syn_real (GPU_REQUIRED)",
                "C2_pipeline_noop (UNSUPPORTED)",
                "delta_BA_excess (not computable without C0/C1/C3)",
            ],
        },
        "E4": {
            "status": "PARTIAL_EXT_F1_ONLY",
            "kind": "threshold-transfer diagnostic",
            "reason": (
                "Computed on the FROZEN historical Flow-1 Track-G SiW-Mv2 evaluation, "
                "re-analysed for threshold transfer -- NOT a new EXT training run. "
                "Covers EXT-F1 only; EXT-F2/EXT-F3 threshold transfer is GPU_REQUIRED "
                "(no target inference for those folds exists). T3-oracle is diagnostic "
                "only, never deployable, never a method. No threshold-transfer anomaly "
                "detected."),
            "blocked_components": ["EXT-F2 threshold transfer (GPU_REQUIRED)",
                                   "EXT-F3 threshold transfer (GPU_REQUIRED)"],
        },
        "E5": {
            "status": "COMPLETED_PARTIAL_EXT_F1_ONLY",
            "kind": "REAL-ONLY target evaluation",
            "reason": (
                f"REAL-ONLY (G-REALONLY) target evaluation COMPLETED and scored for "
                f"EXT-F1 only: mean ACER {e5_acer:.4f} over 5 detector seeds, frozen "
                "source-dev calibration, target_threshold_fitted=false. No REAL-ONLY "
                "result exists for EXT-F2/EXT-F3."),
            "blocked_components": ["EXT-F2 REAL-ONLY (not run)",
                                   "EXT-F3 REAL-ONLY (not run)"],
        },
        "E6": {
            "status": "CLOSED_MATCHED_BANK_INFEASIBLE",
            "kind": "paired original / LLM-SHUFFLE-A rerender",
            "reason": (
                "E6_V2_STATUS=CLOSED_TRUE_FROZEN_MATCHED_BANK_INFEASIBILITY. The frozen "
                "LLM_SHUFFLE_A matched Physics bank fills only 479/512 under the frozen "
                "EXACT per-source-domain quota (CASIA deficit 33, MSU surplus 33, not "
                "fungible). E6_V2_READY_FOR_TRAINING=false; no detector trained; no "
                "semantic-ablation target ACER. No scientific protocol / quota / quality "
                "threshold was changed."),
            "blocked_components": [
                "LLM-SHUFFLE-A matched Physics bank (SCIENTIFIC_INFEASIBILITY under frozen domain quota)",
                "paired semantic-ablation detector training (no bank to train on)",
            ],
        },
        "E7": {
            "status": "SCIENTIFICALLY_BLOCKED",
            "kind": "unified 3-fold Track-G core synthetic evaluation",
            "reason": (
                "All 3 folds terminated SCIENTIFICALLY_BLOCKED_AFTER_RESERVE_CAP; "
                "0/3 valid core matched synthetic banks. The binding constraint was the "
                "LLM-Physics accepted count in every fold (EXT-F1 295, EXT-F2 467, "
                "EXT-F3 472) against the frozen common 512-Physics quota (shortfall "
                "217 / 45 / 40; GPAT feasible 512/512 in all folds). Terminated before "
                "any target ACER scoring. G-RND / G-DET / G-LLM / G-LLM-SHUFFLE-A are "
                "BLOCKED_BY_E7_CORE_BANK; G-REALONLY was READY and evaluated separately "
                "as E5. is_target_acer_evidence=false; "
                "is_evidence_llm_recipes_generally_inferior=false. A source-only, "
                "target-blind generation/quality-gate/matched-bank FEASIBILITY result "
                "under the frozen v1.0+v1.1 protocol -- nothing more."),
            "blocked_components": [
                "G-LLM target ACER (all 3 folds)",
                "G-DET target ACER (all 3 folds)",
                "G-RND target ACER (all 3 folds)",
                "G-LLM-SHUFFLE-A target ACER (all 3 folds; Shuffle-A generation never attempted)",
                "EXT-H1 (LLM vs RND), EXT-H2 (LLM vs DET), EXT-H4 (LLM-original vs LLM-SHUFFLE)",
            ],
        },
        "E8": {
            "status": "COMPLETED_DESCRIPTIVE_ONLY",
            "kind": "q-matched EXT-F1 Track-G target evaluation",
            "reason": (
                "q-matched Track-G target evaluation COMPLETED and scored for EXT-F1 "
                "(arms RND/DET/LLM, 5 detector seeds/arm, 15 runs, scored against the "
                "frozen held-out target). Authoritative mean ACER "
                f"LLM={e8c['acer_means_by_arm']['LLM']:.4f} < "
                f"DET={e8c['acer_means_by_arm']['DET']:.4f} < "
                f"RND={e8c['acer_means_by_arm']['RND']:.4f} (margins ~0.002 ACER). "
                "inferential_status=NOT_CLAIMED; statistical_significance_claimed=false; "
                "llm_superiority_supported=false; no comparable pre-q-match target "
                "baseline exists (ranking_change_vs_pre_qmatch=UNAVAILABLE). G8 scoring "
                "structurally isolated from training and G7; no target labels in "
                "training/G7; source-dev-only calibration. SHUFFLE excluded from E8."),
            "descriptive_only": True,
        },
        "E9": {
            "status": "CLOSED_BANK_SELECTION_ROBUSTNESS_ONLY",
            "kind": "conditional candidate-pool perturbation of the 384->256 selector",
            "reason": (
                "9/9 deterministic candidate-pool perturbation realizations completed "
                "(arms RND/DET/LLM x masks P1/P2/P3; 0 blocked). Measures ONLY the "
                "structural stability of the frozen 384->256 recipe selection when "
                "64 of the 384 frozen candidates are deterministically masked "
                "(retained fraction 320/384), reported as descriptive Jaccard. "
                "llm_calls=0, no target access. Explicitly NOT downstream "
                "detector-performance / target-ACER evidence and NOT independent LLM "
                "generation replication."),
            "is_detector_performance_evidence": False,
            "is_independent_llm_generation_replication": False,
        },
        "E10": {
            "status": "CLOSED_DECISION_GATE_NEGATIVE",
            "kind": "frozen Section-17 evidence-only decision gate",
            "reason": (
                "Conjunctive, fail-closed gate over 5 frozen criteria. "
                "C1_CROSS_DOMAIN_SIGNAL=BLOCKED (needs completed E7 G-LLM/G-DET/G-RND "
                "per-fold target ACER; E7 blocked). "
                "C2_VALUE_OVER_REAL_ONLY=NOT_ESTABLISHED (E5 REAL-ONLY is EXT-F1 only; "
                "matched LLM comparator blocked; the >=2/3-fold comparison is "
                "unreachable). C3_SEMANTIC_ABLATION=BLOCKED (E7 G-LLM-SHUFFLE-A blocked "
                "and E6-v2 matched bank infeasible). C4_QUALITY_CONFOUND=PASS "
                "(DESCRIPTIVE_ONLY / NO_SIGNIFICANCE_CLAIM: after q-matching the LLM arm "
                "still has the lowest mean target ACER in EXT-F1). "
                "C5_INTEGRITY=PASS (WITH LIMITATIONS: the BA-control / threshold / "
                "label-use evidence that was produced detects no invalidating bug or "
                "label leakage; incomplete E3 discriminating controls are a recorded "
                "limitation). Result: E11 NOT authorized; do not call the LLM again."),
            "e10_gate_identity": e10c.get("gate_identity"),
        },
        "E11": {
            "status": E11_STATUS,
            "kind": "independent multi-bank LLM-generation study (optional; spec Section 18)",
            "reason": (
                "E10's conjunctive gate did not pass (2/5 criteria positively "
                "established). Per the frozen decision rule new_llm_calls_authorized="
                "false and e11_authorized=false. Launching E11 would require a separate, "
                "dated protocol amendment with its own new pre-API hash-locked "
                "hypothesis family (EXT-H1..EXT-H4 may not be reused). No E11 work was "
                "performed. No new LLM / provider / API call was made by this closeout."),
        },
    }

    return {
        "schema_version": SCHEMA_STATUS,
        "extension_id": "c_ext_q1q2_v1",
        "title": "PRISM-FAS-C EXT-Q1Q2 -- final scientific closeout status (after E10)",
        "generated_from_head_commit": inp["_head_commit"],
        "governing_spec": {
            "ext_q1q2_detailed_spec_path": inp["e0_protocol"]["authoritative_spec"]["path"],
            "ext_q1q2_detailed_spec_sha256":
                inp["e0_protocol"]["authoritative_spec"]["sha256"],
            "governing_version_c_spec_sha256":
                inp["e0_protocol"]["governing_version_c_spec_sha256"],
            "frozen_criteria_source":
                inp["e10_evidence_matrix"].get("frozen_criteria_source"),
        },
        "closeout_task": {
            "read_only_evidence_synthesis": True,
            "llm_calls": 0,
            "provider_or_network_calls": 0,
            "gpu_jobs": 0,
            "detectors_trained": 0,
            "synthetic_data_generated": 0,
            "experiments_rerun": 0,
            "target_image_feature_or_label_access": False,
            "historical_e0_e10_evidence_modified": False,
            "frozen_protocol_modified": False,
            "additive_namespace": OUTPUT_SUBTREE,
        },
        "milestones": milestones,
        "e10_gate_criteria": per_criterion,
        "e11": {
            "status": E11_STATUS,
            "authorized": False,
            "run": False,
            "new_llm_calls_authorized": False,
            "q1_mechanism_claim_supported": False,
        },
        "interpretation_rule": (
            "Unavailable / upstream-blocked evidence is not evidence that the LLM "
            "failed. Blocked milestones are BLOCKED / NOT_ESTABLISHED, never FAIL and "
            "never a negative scientific result."),
    }


# --------------------------------------------------------------------------- #
# 2. C_EXT_FINAL_RESULTS_TABLE.csv
# --------------------------------------------------------------------------- #

_CSV_HEADER = (
    "evidence_class,milestone,fold,arm_or_pair,metric,value,n,statistic_type,"
    "inferential_status,scope_note"
)


def _row(evidence_class: str, milestone: str, fold: str, arm: str, metric: str,
         value: str, n: str, stat: str, inferential: str, note: str) -> str:
    cells = [evidence_class, milestone, fold, arm, metric, value, n, stat,
             inferential, note]
    out: list[str] = []
    for c in cells:
        c = str(c)
        if any(ch in c for ch in (',', '"', '\n')):
            c = '"' + c.replace('"', '""') + '"'
        out.append(c)
    return ",".join(out)


def build_results_table(inp: Mapping[str, Any]) -> str:
    rows: list[str] = [_CSV_HEADER]

    # --- (0) E1 recipe-bank structural analysis (descriptive; NOT a target metric) --
    e1 = inp["e1_recipe_analysis"]
    e1_note = ("STRUCTURAL statistic of the frozen 256-recipe selected banks only; "
               "NOT a target/detector metric; NOT a semantic-mechanism or LLM-reasoning "
               "claim (E1 scientific_interpretation_boundary); diagnostic, not in the "
               "Holm family")
    for arm in ARMS:
        urr = e1.get("unique_recipe_ratio", {}).get(arm, {})
        if urr.get("unique_ratio") is not None:
            rows.append(_row(
                "e1_recipe_structural_descriptive", "E1", "frozen-256-bank", arm,
                "unique_recipe_ratio", f"{_f(urr['unique_ratio']):.6f}",
                str(urr.get("n", "")), "descriptive_ratio", "not_claimed",
                f"{urr.get('duplicated_key_count', 0)} duplicated recipe keys; " + e1_note))
        gd = e1.get("pairwise_gower_distance", {}).get(arm, {})
        if gd.get("mean") is not None:
            rows.append(_row(
                "e1_recipe_structural_descriptive", "E1", "frozen-256-bank", arm,
                "pairwise_gower_distance_mean", f"{_f(gd['mean']):.6f}",
                str(gd.get("n", "")), "descriptive_mean_ddof1", "not_claimed",
                f"sd_ddof1={_f(gd.get('sd_ddof1')):.6f}; within-bank recipe dispersion; "
                + e1_note))
    for pair in e1.get("cross_arm_js_divergence", []):
        if pair.get("distribution") == "cooccurrence:artifact_x_severity_bin":
            rows.append(_row(
                "e1_recipe_structural_descriptive", "E1", "frozen-256-bank",
                f"{pair['arm_a']}_vs_{pair['arm_b']}",
                "js_divergence_artifact_x_severity_bin",
                f"{_f(pair['js_divergence']):.6f}", "", "descriptive_js_divergence",
                "not_claimed",
                "Jensen-Shannon divergence between arms' joint artifact x severity "
                "co-occurrence; " + e1_note))

    # --- (a) q-confound magnitude (E2, descriptive; diagnostic family) --------
    for pair in inp["e2_quality_analysis"]["q_distribution"]["pairwise_smd"]:
        a, b = pair["arm_a"], pair["arm_b"]
        rows.append(_row(
            "e2_quality_confound_magnitude", "E2", "frozen-F1-bank", f"{a}_vs_{b}",
            "SMD_q", f"{_f(pair['smd']):.6f}", "1024+1024", "descriptive_effect_size",
            "not_claimed",
            "standardized mean difference of accepted-candidate quality q; "
            "|SMD|>=0.25 is the frozen E8 trigger, not a performance result"))

    # --- (b) historical descriptive BA-separation re-summary (E3) -------------
    e3 = inp["e3_ba_controls"]["historical_resummary"]["arm_level_3_probe_seeds"]
    for arm in ARMS:
        rows.append(_row(
            "e3_historical_descriptive_integrity", "E3", "historical", arm,
            "BA_sep_mean", f"{_f(e3[arm]['mean']):.6f}", str(e3[arm]["n"]),
            "descriptive_mean_ddof1", "not_claimed",
            "re-summary of FROZEN historical Flow-1/Flow-2 BA_sep; discriminating "
            "controls C0/C1/C3 GPU-gated so delta_BA_excess is not computable; "
            "no invalidating finding"))

    # --- (c) historical / partial EXT-F1 threshold-transfer diagnostic (E4) ---
    e4agg = inp["e4_threshold_transfer"]["aggregates_ddof1"]
    for arm in ARMS:
        v = e4agg[arm]["video"]
        rows.append(_row(
            "e4_historical_partial_ext_f1_threshold_transfer", "E4", "EXT-F1", arm,
            "T0_source_dev_video_ACER_mean", f"{_f(v['T0_acer']['mean']):.6f}", "5",
            "descriptive_mean_ddof1", "not_claimed",
            "re-analysis of the FROZEN historical Flow-1 Track-G SiW-Mv2 evaluation; "
            "NOT a new EXT run; EXT-F1 only; EXT-F2/F3 GPU_REQUIRED"))
        rows.append(_row(
            "e4_historical_partial_ext_f1_threshold_transfer", "E4", "EXT-F1", arm,
            "T0_minus_oracle_video_ACER_gap_mean",
            f"{_f(v['acer_gap_t0_minus_oracle']['mean']):.6f}", "5",
            "descriptive_mean_ddof1", "not_claimed",
            "T3-oracle is ORACLE_DIAGNOSTIC_ONLY / NOT_DEPLOYABLE / NOT_A_METHOD; "
            "no threshold-transfer anomaly detected"))

    # --- (d) partial EXT-F1 REAL-ONLY completed target evaluation (E5) --------
    e5 = inp["e5_target_scoring_result"]["aggregate"]
    n5 = str(inp["e5_target_scoring_result"]["aggregate"]["seed_count"])
    rows.append(_row(
        "e5_partial_ext_f1_real_only_completed", "E5", "EXT-F1", "REAL_ONLY",
        "target_ACER_mean", f"{_f(e5['acer']['mean']):.6f}", n5,
        "descriptive_mean_ddof1", "not_claimed",
        f"completed & scored; sample_sd(ddof1)={_f(e5['acer']['sample_sd']):.6f}; "
        "frozen source-dev calibration, target_threshold_fitted=false; EXT-F1 only"))
    rows.append(_row(
        "e5_partial_ext_f1_real_only_completed", "E5", "EXT-F1", "REAL_ONLY",
        "target_ROC_AUC_mean", f"{_f(e5['roc_auc']['mean']):.6f}", n5,
        "descriptive_mean_ddof1", "not_claimed",
        "REAL-ONLY (G-REALONLY) arm; no REAL-ONLY result exists for EXT-F2/EXT-F3"))

    # --- (e) E8 q-matched EXT-F1 (descriptive only) --------------------------
    e8c = inp["e8_target_evaluation_closure"]["acer_means_by_arm"]
    e8s = inp["e8_target_score_result"]["arm_summary"]
    for arm in ARMS:
        rows.append(_row(
            "e8_qmatched_ext_f1_descriptive", "E8", "EXT-F1", arm,
            "target_ACER_mean", f"{_f(e8c[arm]):.6f}", "5",
            "descriptive_mean", "not_claimed",
            "q-matched Track-G target evaluation; DESCRIPTIVE ONLY; "
            "inferential_status=NOT_CLAIMED; no significance & no LLM-superiority claim; "
            "margins ~0.002 ACER; EXT-F1 only"))
        rows.append(_row(
            "e8_qmatched_ext_f1_descriptive", "E8", "EXT-F1", arm,
            "target_ROC_AUC_mean", f"{_f(e8s[arm]['roc_auc_mean']):.6f}", "5",
            "descriptive_mean", "not_claimed",
            "descriptive rank LLM(lowest ACER) < DET < RND; no comparable pre-q-match "
            "target baseline exists"))
    for cmp_key, cmp in inp["e8_target_paired_comparisons"][
            "descriptive_seed_summary"]["comparisons"].items():
        rows.append(_row(
            "e8_qmatched_ext_f1_descriptive", "E8", "EXT-F1", cmp_key,
            "mean_paired_ACER_delta", f"{_f(cmp['mean_delta']):.6f}", str(cmp["n"]),
            "descriptive_mean_paired_delta", "not_claimed",
            f"{cmp['count_positive']}/{cmp['n']} seeds positive; descriptive only, "
            "no inferential test computed"))

    # --- (f) E9 bank-SELECTION robustness (structural, descriptive) ----------
    e9arms = inp["e9_bank_stability"]["arms"]
    for arm in ARMS:
        d = e9arms[arm]["descriptive_jaccard_vs_reference"]
        rows.append(_row(
            "e9_bank_selection_robustness_structural", "E9", "frozen-384-pool", arm,
            "jaccard_vs_reference_mean", f"{_f(d['mean']):.6f}", str(d["n_non_blocked"]),
            "descriptive_mean", "not_claimed",
            "structural 256-of-384 selection stability within the frozen 384-candidate "
            "pool (64 masked); NOT detector-performance evidence; NOT independent "
            "LLM-generation replication"))
        rows.append(_row(
            "e9_bank_selection_robustness_structural", "E9", "frozen-384-pool", arm,
            "jaccard_vs_reference_min", f"{_f(d['min']):.6f}", str(d["n_non_blocked"]),
            "descriptive_min", "not_claimed",
            "9/9 realizations OK; bank size never lowered; llm_calls=0; no target access"))

    # --- (g) unavailable E7 3-fold results ---------------------------------
    for cond in ("G-LLM", "G-DET", "G-RND", "G-LLM-SHUFFLE-A"):
        for fold in FOLDS:
            rows.append(_row(
                "e7_three_fold_unavailable", "E7", fold, cond,
                "target_ACER_mean", "UNAVAILABLE_UPSTREAM_BLOCKED", "0",
                "none", "not_applicable",
                "E7 terminated SCIENTIFICALLY_BLOCKED_AFTER_RESERVE_CAP with 0/3 core "
                "matched synthetic banks; no detector trained; no target ACER scored; "
                "unavailable evidence is NOT negative evidence"))

    return "\n".join(rows) + "\n"


# --------------------------------------------------------------------------- #
# 3. C_EXT_FINAL_CLAIM_MATRIX.json
# --------------------------------------------------------------------------- #

def build_claim_matrix(inp: Mapping[str, Any]) -> dict[str, Any]:
    e8c = inp["e8_target_evaluation_closure"]["acer_means_by_arm"]
    e5_acer = _f(inp["e5_target_scoring_result"]["aggregate"]["acer"]["mean"])
    e8_llm = _f(e8c["LLM"])

    claims: list[dict[str, Any]] = [
        {
            "claim_id": "general_llm_superiority",
            "claim_text": (
                "The LLM recipe-generation mechanism is generally superior to the "
                "RND / DET generators for cross-domain face anti-spoofing."),
            "classification": CLS_NOT_SUPPORTED,
            "evidence_status": "DESCRIPTIVE_ONLY_GATE_NEGATIVE",
            "supporting_artifacts": ["e10_gate_decision", "e0_hypothesis_family",
                                     "e8_target_evaluation_closure"],
            "reason": (
                "E0 froze a claim ceiling that forbids a general 'LLM generation "
                "mechanism is superior' claim without E11. The E10 conjunctive gate "
                "did not pass (2/5 criteria established). E8 completed but is "
                "explicitly descriptive only (inferential_status=NOT_CLAIMED, "
                "statistical_significance_claimed=false, llm_superiority_supported="
                "false). The mechanism hypotheses EXT-H1/EXT-H2 depend on the blocked "
                "E7 and EXT-H4 on the blocked SHUFFLE-A ablation."),
            "allowed_statement": (
                "No general LLM-superiority claim is supported. The only completed "
                "target comparison (E8, EXT-F1, q-matched) is descriptive and its "
                "~0.002-ACER LLM-lowest ordering carries no inferential weight."),
            "prohibited_statements": [
                "LLM is superior overall",
                "the LLM mechanism is proven",
                "LLM generation generally outperforms RND/DET",
            ],
        },
        {
            "claim_id": "q_confound_robustness",
            "claim_text": (
                "The descriptive LLM ACER advantage is not an artefact of the LLM "
                "arm's lower accepted-candidate quality q (it survives q-matching)."),
            "classification": CLS_DESCRIPTIVE_ONLY,
            "evidence_status": "DESCRIPTIVE_PRESENT",
            "supporting_artifacts": ["e2_quality_analysis",
                                     "e8_target_evaluation_closure",
                                     "e8_target_paired_comparisons",
                                     "e10_evidence_matrix"],
            "reason": (
                "E2 shows a real q confound (|SMD(q)|>=0.25 for LLM-vs-RND and "
                "LLM-vs-DET). After the E8 q-matched construction (which removes the "
                "confound by design) the LLM arm still has the lowest mean target "
                f"ACER in EXT-F1 (LLM={e8_llm:.4f} <= DET/RND) with favourable mean "
                "paired deltas. E10 C4 = PASS with a DESCRIPTIVE_ONLY / "
                "NO_SIGNIFICANCE_CLAIM qualifier: the effect is ~0.002 ACER, EXT-F1 "
                "only, no inferential test, no pre-q-match baseline."),
            "allowed_statement": (
                "The q-matched EXT-F1 ablation descriptively retains the direction of "
                "the LLM ordering; this is a single-fold descriptive result, not an "
                "LLM-advantage claim."),
            "prohibited_statements": [
                "q-matching proves the LLM advantage is real",
                "the LLM advantage is statistically significant after q-matching",
            ],
        },
        {
            "claim_id": "semantic_coupling_shuffle",
            "claim_text": (
                "LLM-original recipes beat LLM-SHUFFLE-A (field-group-shuffled) "
                "recipes, i.e. the LLM's semantic field coupling carries signal."),
            "classification": CLS_BLOCKED,
            "evidence_status": "BLOCKED",
            "supporting_artifacts": ["e6_v2_final_closure",
                                     "e7_three_fold_scientific_closure",
                                     "e1_recipe_analysis", "e10_evidence_matrix"],
            "reason": (
                "Both routes to an LLM-original vs LLM-SHUFFLE-A detector target-ACER "
                "contrast are blocked upstream: E6-v2 closed "
                "CLOSED_TRUE_FROZEN_MATCHED_BANK_INFEASIBILITY (the SHUFFLE-A matched "
                "Physics bank fills only 479/512 under the frozen exact per-domain "
                "quota) and E7 G-LLM-SHUFFLE-A is BLOCKED_BY_E7_CORE_BANK in all 3 "
                "folds (Shuffle-A generation was never even attempted). No "
                "semantic-ablation detector result exists; EXT-H4 is untestable under "
                "this frozen protocol. E1 shows the LLM recipe bank is structurally "
                "distinguishable from RND/DET at the recipe level (larger cross-arm JS "
                "divergence, notably on the joint artifact x severity co-occurrence, "
                "and lower within-bank Gower dispersion), but E1's own "
                "scientific_interpretation_boundary states these structural statistics "
                "do NOT measure semantic plausibility or a causal role of LLM "
                "reasoning and are only a mechanism signal if E6 LLM-SHUFFLE reduces "
                "downstream performance -- which is exactly the blocked contrast. E1 "
                "therefore cannot lift this claim off BLOCKED."),
            "allowed_statement": (
                "The semantic-coupling (Shuffle) ablation is blocked by matched-bank "
                "feasibility under the frozen quota, before any target evaluation."),
            "prohibited_statements": [
                "the Shuffle ablation shows semantic coupling matters",
                "LLM-original beats LLM-SHUFFLE-A",
                "the Shuffle ablation failed / reversed",
            ],
        },
        {
            "claim_id": "value_over_real_only",
            "claim_text": (
                "Adding LLM-generated synthetic data beats REAL-ONLY training in at "
                "least 2/3 held-out folds by descriptive mean ACER (EXT-H3)."),
            "classification": CLS_NOT_ESTABLISHED,
            "evidence_status": "PARTIAL_UNREACHABLE",
            "supporting_artifacts": ["e5_target_scoring_result",
                                     "e7_three_fold_scientific_closure",
                                     "e10_evidence_matrix"],
            "reason": (
                f"E5 supplies a completed REAL-ONLY target ACER for EXT-F1 only "
                f"(mean {e5_acer:.4f}). The matched LLM comparator (E7 G-LLM) is "
                "BLOCKED_BY_E7_CORE_BANK in all 3 folds and no REAL-ONLY result exists "
                "for EXT-F2/EXT-F3, so the >=2/3-fold comparison is unreachable. Where "
                f"the closest available LLM number touches this contrast (E8 q-matched "
                f"EXT-F1 LLM ACER {e8_llm:.4f}, a different ablation) it is "
                "descriptively worse (higher ACER) than REAL-ONLY, not better. No "
                "positive evidence of LLM value over REAL-ONLY; the missing folds are "
                "the upstream E7 block, not an LLM failure."),
            "allowed_statement": (
                "The LLM-vs-REAL-ONLY comparison could not be formed on >=2/3 folds "
                "(matched LLM comparator blocked; REAL-ONLY is EXT-F1 only)."),
            "prohibited_statements": [
                "LLM synthetic data beats REAL-ONLY",
                "LLM synthetic data is worse than REAL-ONLY",
            ],
        },
        {
            "claim_id": "cross_domain_three_fold_robustness",
            "claim_text": (
                "The frozen LLM bank is not ranked last on at least 2/3 held-out "
                "folds; LLM <= DET/RND in target ACER across folds (EXT-H1/EXT-H2)."),
            "classification": CLS_BLOCKED,
            "evidence_status": "BLOCKED",
            "supporting_artifacts": ["e7_three_fold_scientific_closure",
                                     "e10_evidence_matrix"],
            "reason": (
                "E7 terminated SCIENTIFICALLY_BLOCKED_AFTER_RESERVE_CAP in all 3 folds "
                "with 0/3 valid core matched synthetic banks; G-LLM / G-DET / G-RND "
                "detector trainings were never run and no per-fold cross-domain target "
                "ACER exists. is_target_acer_evidence=false. The only per-fold LLM/DET/"
                "RND ACERs anywhere are EXT-F1-only and from non-E7 designs (E8 "
                "q-matched; E4 historical re-analysis) and cannot satisfy a 2/3-fold "
                "requirement."),
            "allowed_statement": (
                "The 3-fold cross-domain robustness contrast is blocked by "
                "source-only matched-bank feasibility, before any target evaluation."),
            "prohibited_statements": [
                "E7 shows the LLM bank failed",
                "the LLM bank is robust / not robust across folds",
                "the 3-fold result is negative",
            ],
        },
        {
            "claim_id": "bank_selection_subset_robustness",
            "claim_text": (
                "The frozen 384->256 recipe selection is not brittle to which exact "
                "256-recipe subset was drawn from the frozen 384-candidate pools."),
            "classification": CLS_DESCRIPTIVE_ONLY,
            "evidence_status": "DESCRIPTIVE_PRESENT",
            "supporting_artifacts": ["e9_closure", "e9_bank_stability"],
            "reason": (
                "E9 closed 9/9 deterministic candidate-pool perturbation realizations "
                "(RND/DET/LLM x P1/P2/P3, 0 blocked). With 64 of 384 candidates "
                "deterministically masked the 256-recipe selection retains a large "
                "descriptive Jaccard overlap with the reference selection (per-arm "
                "means ~0.63-0.70). This is a descriptive structural-stability "
                "statement about the selector on the SAME frozen candidate pool."),
            "allowed_statement": (
                "E9 tests sensitivity to the exact selected 256-recipe subset within "
                "the frozen 384-candidate pools only, via three deterministic "
                "perturbation masks; the selection is descriptively stable."),
            "prohibited_statements": [
                "E9 proves detector robustness",
                "E9 shows the LLM bank yields robust detector performance",
                "E9 is independent LLM-generation replication",
            ],
            "is_detector_performance_evidence": False,
            "is_independent_llm_generation_replication": False,
        },
        {
            "claim_id": "independent_llm_generation_robustness",
            "claim_text": (
                "The LLM-generation result reproduces under independent re-generation "
                "with fresh LLM calls / multiple independent banks (spec Section 18 "
                "E11)."),
            "classification": CLS_NOT_ESTABLISHED,
            "evidence_status": "NOT_PRESENT",
            "supporting_artifacts": ["e10_gate_decision", "e9_closure"],
            "reason": (
                "E11 is NOT_AUTHORIZED_NOT_RUN: the E10 gate did not pass and "
                "new_llm_calls_authorized=false. No independent re-generation exists. "
                "E9 is explicitly NOT independent generator replication -- it perturbs "
                "the selection of an already-frozen candidate pool and makes zero LLM "
                "calls."),
            "allowed_statement": (
                "Independent LLM-generation replication was not attempted; E10 did "
                "not authorize E11 and no new LLM calls were made."),
            "prohibited_statements": [
                "the LLM generation result was independently replicated",
                "E9 replicates the LLM generation",
            ],
        },
        {
            "claim_id": "integrity_leakage_status",
            "claim_text": (
                "The completed EXT-Q1Q2 target evaluations contain no bug or target-"
                "label leakage that invalidates them."),
            "classification": CLS_SUPPORTED,
            "evidence_status": "PRESENT_WITH_LIMITATIONS",
            "supporting_artifacts": ["e8_target_label_use_record",
                                     "e5_target_scoring_result",
                                     "e4_threshold_transfer", "e3_ba_controls",
                                     "e10_evidence_matrix"],
            "reason": (
                "E10 C5 = PASS. The integrity evidence that was produced detects no "
                "invalidating bug and no label leakage: E8 G8 scoring is structurally "
                "isolated from training and G7 (uses_target_labels_in_training_or_g7="
                "false), calibration is source-dev-only with no target threshold "
                "fitting (E5, E8), E8 claims no significance and no LLM superiority, "
                "E4's EXT-F1 source-derived operating point sits close to the target "
                "oracle with no anomaly, and E3's historical BA_sep re-summary "
                "produced no invalidating finding. LIMITATIONS (recorded, not "
                "invalidating): E3 discriminating BA controls C0/C1/C3 are GPU-gated "
                "so delta_BA_excess is not computed; E4 threshold transfer is EXT-F1 "
                "only; no completed core 3-fold evaluation exists."),
            "allowed_statement": (
                "No bug or target-label leakage that would invalidate the completed "
                "EXT-F1 target evaluations was detected; some discriminating integrity "
                "controls remain GPU-gated and are listed as limitations."),
            "prohibited_statements": [
                "all integrity controls passed",
                "leakage was fully ruled out",
            ],
        },
    ]

    matrix = {
        "schema_version": SCHEMA_CLAIM_MATRIX,
        "extension_id": "c_ext_q1q2_v1",
        "generated_from_head_commit": inp["_head_commit"],
        "classification_vocabulary": list(CLASSIFICATIONS),
        "classification_legend": {
            CLS_SUPPORTED: "carried by real, available, non-blocked evidence "
                           "(may still carry stated limitations)",
            CLS_DESCRIPTIVE_ONLY: "a completed measurement descriptively points this "
                                  "way, with no inferential / significance claim",
            CLS_NOT_SUPPORTED: "the available evidence does not carry the claim",
            CLS_NOT_ESTABLISHED: "the comparison needed for the claim could not be "
                                 "formed (partial / unreachable evidence)",
            CLS_BLOCKED: "the evidence is blocked upstream by protocol / feasibility, "
                         "before any target evaluation",
        },
        "hard_invariants": {
            "blocked_or_unavailable_evidence_cannot_be_affirmative": True,
            "e9_is_not_detector_performance_evidence": True,
            "e9_is_not_independent_llm_generation_replication": True,
            "e7_block_is_not_a_negative_llm_result": True,
            "no_statistical_significance_claimed_anywhere": True,
            "e11_status": E11_STATUS,
        },
        "claims": claims,
    }
    assert_claim_matrix_invariants(matrix)
    return matrix


def assert_claim_matrix_invariants(matrix: Mapping[str, Any]) -> None:
    """Fail closed if the claim matrix would promote blocked evidence, mis-state E9,
    or claim statistical significance."""
    ids = {c["claim_id"] for c in matrix["claims"]}
    required = {
        "general_llm_superiority", "q_confound_robustness",
        "semantic_coupling_shuffle", "value_over_real_only",
        "cross_domain_three_fold_robustness", "bank_selection_subset_robustness",
        "independent_llm_generation_robustness", "integrity_leakage_status",
    }
    _require(required.issubset(ids),
             f"claim matrix is missing required claims: {sorted(required - ids)}")

    for c in matrix["claims"]:
        cid = c["claim_id"]
        cls = c["classification"]
        _require(cls in CLASSIFICATIONS, f"{cid}: unknown classification {cls!r}")
        if c["evidence_status"] in NON_AFFIRMATIVE_EVIDENCE:
            _require(cls not in AFFIRMATIVE_CLASSIFICATIONS,
                     f"{cid}: evidence_status={c['evidence_status']} may not be "
                     f"promoted to {cls} -- blocked/unavailable evidence can never be "
                     "SUPPORTED or DESCRIPTIVELY_SUPPORTED_ONLY.")
        blob = " ".join([
            c["claim_text"], c["reason"], c["allowed_statement"],
        ]).lower()
        _require("statistically significant" not in blob,
                 f"{cid}: an affirmative-context statement contains "
                 "'statistically significant'.")

    # E7 / cross-domain / semantic-ablation must stay non-affirmative.
    for cid in ("cross_domain_three_fold_robustness", "semantic_coupling_shuffle"):
        c = next(x for x in matrix["claims"] if x["claim_id"] == cid)
        _require(c["classification"] == CLS_BLOCKED,
                 f"{cid}: must be classified {CLS_BLOCKED}, got {c['classification']}.")

    # E9-derived claims may never be affirmed as detector-performance or as
    # independent generation replication.
    bsr = next(x for x in matrix["claims"]
               if x["claim_id"] == "bank_selection_subset_robustness")
    _require(bsr.get("is_detector_performance_evidence") is False
             and bsr.get("is_independent_llm_generation_replication") is False,
             "bank_selection_subset_robustness must flag E9 as neither "
             "detector-performance evidence nor independent generation replication.")
    ind = next(x for x in matrix["claims"]
               if x["claim_id"] == "independent_llm_generation_robustness")
    _require(ind["classification"] not in AFFIRMATIVE_CLASSIFICATIONS,
             "independent_llm_generation_robustness must not be affirmed (no E11).")

    _require(matrix["hard_invariants"]["e11_status"] == E11_STATUS,
             "claim matrix e11_status must be NOT_AUTHORIZED_NOT_RUN.")


# --------------------------------------------------------------------------- #
# 4. C_EXT_PAPER_READY_SUMMARY.md
# --------------------------------------------------------------------------- #

def build_paper_ready_summary(inp: Mapping[str, Any]) -> str:
    e8c = inp["e8_target_evaluation_closure"]["acer_means_by_arm"]
    e8s = inp["e8_target_score_result"]["arm_summary"]
    e5 = inp["e5_target_scoring_result"]["aggregate"]
    e2 = {p["arm_a"] + "_vs_" + p["arm_b"]: p["smd"]
          for p in inp["e2_quality_analysis"]["q_distribution"]["pairwise_smd"]}
    e9 = inp["e9_bank_stability"]["arms"]
    e1 = inp["e1_recipe_analysis"]
    e1_gd = e1.get("pairwise_gower_distance", {})
    e1_js = {(p["arm_a"], p["arm_b"]): p["js_divergence"]
             for p in e1.get("cross_arm_js_divergence", [])
             if p.get("distribution") == "cooccurrence:artifact_x_severity_bin"}
    head = inp["_head_commit"]

    def r4(x: Any) -> str:
        return f"{float(x):.4f}"

    lines: list[str] = []
    A = lines.append
    A("# PRISM-FAS-C EXT-Q1Q2 -- paper-ready scientific summary (final closeout)")
    A("")
    A(f"_Read-only evidence synthesis at the E10 scientific-closure commit "
      f"`{head}`. No experiment was run; no LLM / provider / API call was made; "
      f"no detector was trained; no synthetic data was generated; no raw target "
      f"image, feature or label was accessed._")
    A("")

    A("## What was attempted")
    A("")
    A("An additive Q1/Q2 evidence-strengthening extension to Version C, executed "
      "under the frozen **Q2-NoLLM** profile (`ALLOW_LLM_API = FALSE` for the "
      "whole E0-E10 core). The pre-registered primary family (frozen at E0) is:")
    A("")
    A("- **EXT-H1** LLM vs RND, **EXT-H2** LLM vs DET, **EXT-H3** LLM vs "
      "REAL-ONLY, **EXT-H4** LLM-original vs LLM-SHUFFLE-A;")
    A("- primary endpoint: video-level ACER by fold/seed on the held-out target "
      "SiW-Mv2, **conditional on the frozen recipe banks**;")
    A("- Holm-Bonferroni, alpha = 0.05;")
    A("- E0 claim ceiling: **no general 'LLM generation mechanism is superior' "
      "claim without E11.**")
    A("")
    A("The intended mechanism test was a unified 3-fold cross-domain Track-G "
      "detector experiment (E7) with matched RND/DET/LLM/LLM-SHUFFLE-A synthetic "
      "banks, plus REAL-ONLY (E5), a recipe-bank structural analysis (E1), a "
      "q-confound ablation (E8), a bank-selection sensitivity check (E9), "
      "BA-separation negative controls (E3) and a threshold-transfer diagnostic "
      "(E4); then the E10 decision gate.")
    A("")

    A("## What completed")
    A("")
    A("- **E0** protocol / hypothesis-family / binding freeze -- COMPLETE and locked.")
    A("- **E1** recipe-bank structural analysis -- COMPLETE (CPU, descriptive). "
      f"All 12 structural metrics generated for the 3 frozen 256-recipe banks "
      f"(status `{e1.get('status')}`, 0 LLM calls, no target access). The LLM bank "
      "is structurally distinguishable from RND/DET at the recipe level: larger "
      "cross-arm Jensen-Shannon divergence on the joint artifact x severity "
      "co-occurrence (RND-vs-LLM "
      f"{r4(e1_js.get(('RND', 'LLM'), float('nan')))}, DET-vs-LLM "
      f"{r4(e1_js.get(('DET', 'LLM'), float('nan')))} vs RND-vs-DET "
      f"{r4(e1_js.get(('RND', 'DET'), float('nan')))}) and lower within-bank "
      f"pairwise Gower dispersion (LLM mean {r4(e1_gd.get('LLM', {}).get('mean', float('nan')))} "
      f"vs RND {r4(e1_gd.get('RND', {}).get('mean', float('nan')))} / "
      f"DET {r4(e1_gd.get('DET', {}).get('mean', float('nan')))}). "
      "**Structural only** -- per E1's own interpretation boundary this does not "
      "measure semantic plausibility or a causal role of LLM reasoning, and is a "
      "mechanism signal only if the (blocked) E6 LLM-SHUFFLE ablation reduces "
      "downstream performance. E1 is a diagnostic, not in the Holm family, and is "
      "not consulted by any E10 gate criterion.")
    A("- **E2** quality-distribution reconstruction -- COMPLETE. A real quality "
      f"confound is present: SMD(q) LLM-vs-RND = {r4(e2['LLM_vs_RND'])}, "
      f"LLM-vs-DET = {r4(e2['LLM_vs_DET'])} (both |SMD| >= 0.25; RND-vs-DET "
      f"= {r4(e2['RND_vs_DET'])}). This fired the frozen E8 trigger.")
    A("- **E5** REAL-ONLY target evaluation -- COMPLETED and scored for **EXT-F1 "
      f"only**: mean target ACER {r4(e5['acer']['mean'])} "
      f"(sample SD {r4(e5['acer']['sample_sd'])}, 5 detector seeds), "
      f"mean ROC-AUC {r4(e5['roc_auc']['mean'])}. Frozen source-dev calibration, "
      "no target threshold fitted.")
    A("- **E8** q-matched Track-G target evaluation -- COMPLETED and scored for "
      "**EXT-F1 only** (arms RND/DET/LLM, 5 detector seeds/arm, 15 runs). "
      f"Authoritative descriptive mean target ACER: **LLM = {r4(e8c['LLM'])} < "
      f"DET = {r4(e8c['DET'])} < RND = {r4(e8c['RND'])}** "
      f"(mean ROC-AUC LLM {r4(e8s['LLM']['roc_auc_mean'])}, "
      f"DET {r4(e8s['DET']['roc_auc_mean'])}, RND {r4(e8s['RND']['roc_auc_mean'])}). "
      "Margins are ~0.002 ACER. **Descriptive only:** inferential_status = "
      "NOT_CLAIMED, no statistical-significance claim, no LLM-superiority claim, "
      "no comparable pre-q-match target baseline exists.")
    A("- **E9** conditional bank-selection robustness -- CLOSED. 9/9 deterministic "
      "candidate-pool perturbation realizations (RND/DET/LLM x masks P1/P2/P3, 0 "
      "blocked). With 64 of 384 frozen candidates masked, the 384->256 selection "
      "keeps a large descriptive Jaccard overlap with the reference selection "
      f"(per-arm means RND {r4(e9['RND']['descriptive_jaccard_vs_reference']['mean'])}, "
      f"DET {r4(e9['DET']['descriptive_jaccard_vs_reference']['mean'])}, "
      f"LLM {r4(e9['LLM']['descriptive_jaccard_vs_reference']['mean'])}). "
      "**Structural selection stability only** -- 0 LLM calls, no target access.")
    A("- **E10** decision gate -- CLOSED. Criteria: C1 BLOCKED, C2 NOT_ESTABLISHED, "
      "C3 BLOCKED, C4 PASS (descriptive only), C5 PASS (with limitations). "
      "**E11 NOT authorized; new LLM calls NOT authorized; Q1 mechanism claim "
      "NOT supported.**")
    A("")

    A("## What was blocked")
    A("")
    A("- **E7 (primary mechanism test) -- SCIENTIFICALLY BLOCKED.** All three folds "
      "terminated `SCIENTIFICALLY_BLOCKED_AFTER_RESERVE_CAP` with **0/3** valid core "
      "matched synthetic banks. The binding constraint was the LLM-Physics accepted "
      "count in every fold (EXT-F1 295, EXT-F2 467, EXT-F3 472) against the frozen "
      "common 512-Physics quota (GPAT was feasible 512/512 in all folds). E7 "
      "terminated **before any target ACER scoring**; G-RND/G-DET/G-LLM/"
      "G-LLM-SHUFFLE-A are `BLOCKED_BY_E7_CORE_BANK`. This is a source-only, "
      "target-blind generation/quality-gate/matched-bank **feasibility** result "
      "under the frozen protocol -- `is_target_acer_evidence = false`, "
      "`is_evidence_llm_recipes_generally_inferior = false`.")
    A("- **E6 (paired original / LLM-SHUFFLE-A rerender) -- CLOSED, matched bank "
      "infeasible.** The frozen SHUFFLE-A matched Physics bank fills only 479/512 "
      "under the frozen EXACT per-source-domain quota (CASIA deficit 33, MSU "
      "surplus 33, not fungible). No detector trained; no semantic-ablation target "
      "ACER.")
    A("- **E3 discriminating BA controls (C0/C1/C3) and C2** -- GPU-gated / "
      "unsupported; `delta_BA_excess` not computable.")
    A("- **E4 EXT-F2/EXT-F3 threshold transfer** -- GPU-required; only the EXT-F1 "
      "re-analysis of the frozen historical Flow-1 evaluation exists.")
    A("- **E11** -- NOT_AUTHORIZED_NOT_RUN (gate did not pass).")
    A("")

    A("## Valid quantitative results")
    A("")
    A("All numbers below are descriptive; none carries an inferential or "
      "significance claim. Full rows: `C_EXT_FINAL_RESULTS_TABLE.csv`.")
    A("")
    A("| Evidence class | Fold | Quantity | Value |")
    A("|---|---|---|---|")
    A(f"| E1 recipe structural (descriptive) | frozen-256 bank | within-bank Gower "
      f"dispersion mean LLM / DET / RND | "
      f"{r4(e1_gd.get('LLM', {}).get('mean', float('nan')))} / "
      f"{r4(e1_gd.get('DET', {}).get('mean', float('nan')))} / "
      f"{r4(e1_gd.get('RND', {}).get('mean', float('nan')))} |")
    A(f"| E1 recipe structural (descriptive) | frozen-256 bank | JS divergence "
      f"artifact x severity: RND-LLM / DET-LLM / RND-DET | "
      f"{r4(e1_js.get(('RND', 'LLM'), float('nan')))} / "
      f"{r4(e1_js.get(('DET', 'LLM'), float('nan')))} / "
      f"{r4(e1_js.get(('RND', 'DET'), float('nan')))} |")
    A(f"| E2 quality confound | frozen-F1 bank | SMD(q) LLM-vs-RND / LLM-vs-DET | "
      f"{r4(e2['LLM_vs_RND'])} / {r4(e2['LLM_vs_DET'])} |")
    A(f"| E5 REAL-ONLY (completed, partial) | EXT-F1 | mean target ACER (5 seeds) | "
      f"{r4(e5['acer']['mean'])} |")
    A(f"| E8 q-matched (descriptive) | EXT-F1 | mean target ACER LLM / DET / RND | "
      f"{r4(e8c['LLM'])} / {r4(e8c['DET'])} / {r4(e8c['RND'])} |")
    A(f"| E9 bank selection (structural) | frozen-384 pool | mean Jaccard vs "
      f"reference RND / DET / LLM | "
      f"{r4(e9['RND']['descriptive_jaccard_vs_reference']['mean'])} / "
      f"{r4(e9['DET']['descriptive_jaccard_vs_reference']['mean'])} / "
      f"{r4(e9['LLM']['descriptive_jaccard_vs_reference']['mean'])} |")
    A("| E7 3-fold G-LLM/G-DET/G-RND/G-SHUFFLE target ACER | EXT-F1/F2/F3 | "
      "mean target ACER | UNAVAILABLE (upstream-blocked) |")
    A("")

    A("## Limitations")
    A("")
    A("- The primary mechanism test (E7) produced **no target evidence**: the "
      "matched-bank construction was infeasible under the frozen quota in every "
      "fold. EXT-H1, EXT-H2 and EXT-H4 are therefore untestable under this frozen "
      "protocol.")
    A("- Every completed target evaluation (E5, E8) is **EXT-F1 only** and "
      "**5 detector seeds**; effect sizes in E8 are ~0.002 ACER with per-seed SD "
      "an order of magnitude larger.")
    A("- E8 is a q-matched ablation with **no comparable pre-q-match target "
      "baseline**, so a ranking change cannot be assessed.")
    A("- No inferential procedure for multi-detector-seed arm-level comparison "
      "existed in the tracked frozen repository before target results, so **no "
      "significance is claimed anywhere**.")
    A("- E9 is **structural selection stability**, not detector performance and "
      "not independent generation replication.")
    A("- E3's discriminating BA controls are GPU-gated, so whether the "
      "synthetic/real separation is a generator fingerprint or a domain/pipeline "
      "confound is **not affirmatively resolved**.")
    A("- E4's operating-point diagnostic covers EXT-F1 only; T3-oracle is "
      "diagnostic-only and never a method.")
    A("")

    A("## Claims that ARE allowed")
    A("")
    A("- E1 shows the LLM recipe bank is **structurally distinguishable** from "
      "RND/DET at the recipe level (cross-arm JS divergence, lower within-bank "
      "Gower dispersion); this is a descriptive structural fact only and is **not** "
      "evidence of a semantic mechanism -- that would need the blocked E6 Shuffle "
      "ablation.")
    A("- E7 was **blocked by source-only matched-bank feasibility before target "
      "evaluation**; it is not evidence about LLM recipe quality either way.")
    A("- E8 provides **descriptive EXT-F1 q-matched evidence only**: after "
      "q-matching, the LLM arm has the lowest mean target ACER in that single fold, "
      "with no significance and no superiority claim.")
    A("- After q-matching, the descriptive LLM ordering **does not reverse** in "
      "EXT-F1 (the E10 C4 PASS, descriptive-only).")
    A("- E9 shows the frozen 384->256 recipe selection is **descriptively stable** "
      "to which exact 256-recipe subset is drawn from the frozen 384-candidate "
      "pools, under three deterministic perturbation masks.")
    A("- No bug or target-label leakage that would invalidate the completed EXT-F1 "
      "target evaluations was detected (E10 C5 PASS, with GPU-gated controls "
      "listed as limitations).")
    A("- **E10 did not authorize E11**; unavailable evidence is not negative "
      "evidence.")
    A("")

    A("## Claims that must NOT be made")
    A("")
    A("- “The LLM is superior overall” / “LLM generation generally "
      "outperforms RND/DET”.")
    A("- “The LLM mechanism is proven” / “the Q1 mechanism claim is "
      "supported”.")
    A("- “E7 shows the LLM recipes failed / are inferior”, or any reading "
      "of the E7 block as a negative LLM result.")
    A("- “E9 proves detector robustness” or “E9 shows robust LLM "
      "detector performance”.")
    A("- “E9 is an independent LLM-generation replication”.")
    A("- “LLM synthetic data beats (or loses to) REAL-ONLY”.")
    A("- “The Shuffle / semantic-coupling ablation shows an effect” (or "
      "that it reversed).")
    A("- “Statistically significant” anywhere -- no authoritative artifact "
      "supports it.")
    A("")

    A("## Bottom line")
    A("")
    A("The EXT-Q1Q2 extension is **CLOSED with the primary mechanism evidence "
      "blocked**. The frozen matched-bank protocol could not construct the E7 "
      "cross-domain banks in any fold, so the pre-registered LLM-vs-RND/DET/SHUFFLE "
      "mechanism comparison was never evaluated on the target. The completed "
      "descriptive work (E5, E8, E9) does not, and under the E0 claim ceiling "
      "cannot, support a general LLM-superiority or LLM-mechanism claim. E10 "
      "correctly declined to authorize another LLM call (E11).")
    A("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 6. C_EXT_FINAL_CLOSURE.json
# --------------------------------------------------------------------------- #

def build_final_closure(inp: Mapping[str, Any]) -> dict[str, Any]:
    e7 = inp["e7_three_fold_scientific_closure"]
    e8lu = inp["e8_target_label_use_record"]
    e9 = inp["e9_closure"]
    e10c = inp["e10_closure"]
    e10d = inp["e10_gate_decision"]
    ssha = inp["_source_sha256"]

    def src(name: str) -> str:
        return CONSULTED_SOURCES[name]

    body = {
        "schema_version": SCHEMA_CLOSURE,
        "extension_id": "c_ext_q1q2_v1",
        "title": "PRISM-FAS-C EXT-Q1Q2 -- final closure (post-E10)",
        "generated_from_head_commit": inp["_head_commit"],
        "head_commit_resolved_from_git": inp["_head_commit_resolved"],
        "governing_spec": {
            "ext_q1q2_detailed_spec": {
                "path": inp["e0_protocol"]["authoritative_spec"]["path"],
                "sha256": inp["e0_protocol"]["authoritative_spec"]["sha256"],
                "frozen_criteria_source":
                    inp["e10_evidence_matrix"].get("frozen_criteria_source"),
            },
            "version_c_v1_5_spec": {
                "path": "docs/PRISM_FAS_C_LLM_v1_5_FINAL_ComputeConstrained_"
                        "FullPipeline_Spec_2026.docx",
                "sha256": inp["e0_protocol"]["governing_version_c_spec_sha256"],
            },
            "execution_profile": inp["e0_protocol"].get("execution_profile"),
        },
        "closeout_task_state": {
            "read_only_evidence_synthesis": True,
            "target_access": False,
            "target_features_accessed": False,
            "target_labels_accessed": False,
            "llm_calls": 0,
            "provider_or_network_calls": 0,
            "gpu_jobs": 0,
            "no_new_scientific_experiment_performed": True,
            "no_detector_trained": True,
            "no_synthetic_data_generated": True,
            "no_e0_e10_evidence_modified": True,
            "no_frozen_protocol_modified": True,
        },
        "closure_provenance": {
            "E7": {
                "artifact": src("e7_three_fold_scientific_closure"),
                "artifact_sha256": ssha[src("e7_three_fold_scientific_closure")],
                "status": "SCIENTIFICALLY_BLOCKED",
                "core_matched_banks_produced":
                    e7["final_scientific_interpretation"]["core_matched_banks_produced"],
                "is_target_acer_evidence":
                    e7["final_scientific_interpretation"]["is_target_acer_evidence"],
                "reserve_schedule_rule_identity":
                    e7["provenance"]["reserve_schedule_rule_identity"],
                "ext_f1_scientific_execution_commit":
                    e7["provenance"]["ext_f1_scientific_execution_commit"],
                "ext_f2_execution_commit": e7["provenance"]["ext_f2_execution_commit"],
                "ext_f3_execution_commit": e7["provenance"]["ext_f3_execution_commit"],
                "evidence_package_commit": E7_EVIDENCE_PACKAGE_COMMIT,
                "reserve_stage_close_commit": E7_RESERVE_STAGE_CLOSE_COMMIT,
            },
            "E8": {
                "artifact": src("e8_target_evaluation_closure"),
                "artifact_sha256": ssha[src("e8_target_evaluation_closure")],
                "status": "EXECUTED_AND_SCORED_DESCRIPTIVE_ONLY",
                "fold_scope": "EXT-F1",
                "inferential_status": "NOT_CLAIMED",
                "statistical_significance_claimed": False,
                "llm_superiority_supported": False,
                "acer_means_by_arm":
                    inp["e8_target_evaluation_closure"]["acer_means_by_arm"],
                "e8_prediction_lockset_identity":
                    e8lu.get("e8_prediction_lockset_identity"),
                "scoring_code_commit": e8lu.get("scoring_code_commit"),
                "uses_target_labels_in_training_or_g7":
                    e8lu.get("e8_uses_target_labels_in_training_or_g7"),
                "g8_structurally_isolated_from_training_and_g7":
                    e8lu.get("e8_g8_structurally_isolated_from_training_and_g7"),
                "is_first_ever_blind_reveal": e8lu.get("is_first_ever_blind_reveal"),
            },
            "E9": {
                "artifact": src("e9_closure"),
                "artifact_sha256": ssha[src("e9_closure")],
                "status": "CLOSED_BANK_SELECTION_ROBUSTNESS_ONLY",
                "realizations_ok": e9["realizations_ok"],
                "realizations_total": e9["realizations_total"],
                "realizations_blocked": e9["realizations_blocked"],
                "is_detector_performance_evidence": False,
                "is_independent_llm_generation_replication": False,
                "execution_commit": e9["execution_commit"],
                "implementation_source_commit": e9["implementation_source_commit"],
                "e9_protocol_amendment_identity":
                    e9["e9_protocol_amendment_identity"],
                "protocol_lock_identity": e9["protocol_lock_identity"],
                "archive_commit": E9_ARCHIVE_COMMIT,
                "llm_calls": e9["llm_calls"],
                "target_access": e9["target_access"],
            },
            "E10": {
                "artifact": src("e10_closure"),
                "artifact_sha256": ssha[src("e10_closure")],
                "status": "CLOSED_DECISION_GATE_NEGATIVE",
                "gate_identity": e10c.get("gate_identity"),
                "per_criterion_status": e10d["per_criterion_status"],
                "e11_authorized": e10d["e11_authorized"],
                "new_llm_calls_authorized": e10d["new_llm_calls_authorized"],
                "q1_mechanism_claim_supported":
                    e10c["q1_mechanism_claim_supported"],
                "q1_mechanism_claim_status": e10c["q1_mechanism_claim_status"],
                "closure_commit": E10_CLOSURE_HEAD_COMMIT,
                "llm_calls": e10d["llm_calls"],
            },
        },
        "milestones_status": {
            "E0": "COMPLETE",
            "E1": "COMPLETE",
            "E2": "COMPLETE",
            "E3": "FEASIBILITY_AUDIT_ONLY_GPU_GATED",
            "E4": "PARTIAL_EXT_F1_ONLY",
            "E5": "COMPLETED_PARTIAL_EXT_F1_ONLY",
            "E6": "CLOSED_MATCHED_BANK_INFEASIBLE",
            "E7": "SCIENTIFICALLY_BLOCKED",
            "E8": "COMPLETED_DESCRIPTIVE_ONLY",
            "E9": "CLOSED_BANK_SELECTION_ROBUSTNESS_ONLY",
            "E10": "CLOSED_DECISION_GATE_NEGATIVE",
            "E11": E11_STATUS,
        },
        "milestones_status_note": (
            "E0-E10 all resolved from available authoritative evidence at this HEAD. "
            "E1 was re-audited from the restored "
            "reports/c_ext_q1q2_v1/e1_recipe_analysis/E1_RECIPE_ANALYSIS.json "
            f"(artifact status={inp['e1_recipe_analysis'].get('status')}); E10's own "
            "milestone_status_audit still records E1 EVIDENCE_NOT_PRESENT_IN_CHECKOUT "
            "(its checkout lacked the git-ignored subtree) -- this does not touch any "
            "E10 gate criterion. E11 is governed solely by the authoritative E10 "
            "decision."),
        "e11_authorized": False,
        "new_llm_calls_authorized": False,
        "q1_mechanism_claim_supported": False,
        "e11_status": E11_STATUS,
        "extension_status": EXTENSION_STATUS,
        "extension_status_reason": (
            "The pre-registered primary mechanism test (E7 unified 3-fold Track-G) "
            "is SCIENTIFICALLY_BLOCKED: 0/3 core matched synthetic banks under the "
            "frozen matched-bank protocol, terminated before any target ACER scoring. "
            "The semantic-ablation route (E6/E7 SHUFFLE-A) is likewise blocked by "
            "matched-bank feasibility. The completed work (E5 REAL-ONLY EXT-F1, E8 "
            "q-matched EXT-F1, E9 bank-selection robustness) is descriptive / "
            "structural only and, under the E0 claim ceiling, cannot support a "
            "general LLM-superiority or LLM-mechanism claim. E10 fail-closed and did "
            "not authorize E11. Unavailable / upstream-blocked evidence is not "
            "evidence that the LLM failed."),
        "consulted_artifacts": [
            {"name": n, "path": CONSULTED_SOURCES[n], "sha256": ssha[CONSULTED_SOURCES[n]]}
            for n in sorted(CONSULTED_SOURCES)
        ],
    }
    body["closeout_identity"] = sha256_json(
        {k: v for k, v in body.items() if k != "closeout_identity"})
    return body


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def evaluate(root: Path | None = None, *, inputs: Mapping[str, Any] | None = None
             ) -> dict[str, Any]:
    inp = dict(inputs) if inputs is not None else load_inputs(root)
    final_status = build_final_status(inp)
    results_table_csv = build_results_table(inp)
    claim_matrix = build_claim_matrix(inp)
    paper_ready_md = build_paper_ready_summary(inp)
    final_closure = build_final_closure(inp)
    return {
        "final_status": final_status,
        "results_table_csv": results_table_csv,
        "claim_matrix": claim_matrix,
        "paper_ready_summary_md": paper_ready_md,
        "final_closure": final_closure,
        "consulted_sha256": dict(inp["_source_sha256"]),
        "head_commit": inp["_head_commit"],
    }


def write_artifacts(body: Mapping[str, Any], *, root: Path | None = None
                    ) -> dict[str, str]:
    root = root or repo_root()
    for rel in OUTPUT_RELPATHS:
        assert_ext_write_path(rel, root=root, must_be_under=OUTPUT_SUBTREE)

    written: dict[str, str] = {}
    written["final_status"] = write_json_atomic(
        STATUS_RELPATH, body["final_status"], root=root)
    written["results_table"] = write_text_atomic(
        RESULTS_TABLE_RELPATH, body["results_table_csv"], root=root)
    written["claim_matrix"] = write_json_atomic(
        CLAIM_MATRIX_RELPATH, body["claim_matrix"], root=root)
    written["paper_ready_summary"] = write_text_atomic(
        PAPER_READY_RELPATH, body["paper_ready_summary_md"], root=root)
    written["final_closure"] = write_json_atomic(
        CLOSURE_RELPATH, body["final_closure"], root=root)

    manifest: list[tuple[str, str]] = []
    for rel, digest in body["consulted_sha256"].items():
        manifest.append((rel, digest))
    for key in ("final_status", "results_table", "claim_matrix",
                "paper_ready_summary", "final_closure"):
        rel = written[key]
        manifest.append((rel, sha256_file(root / rel)))
    manifest.sort()
    text = "".join(f"{d}  {p}\n" for p, d in manifest)
    written["evidence_manifest"] = write_text_atomic(
        EVIDENCE_MANIFEST_RELPATH, text, root=root)
    return written


def _print_summary(body: Mapping[str, Any]) -> None:
    fc = body["final_closure"]
    print(json.dumps({
        "extension_status": fc["extension_status"],
        "e11_status": fc["e11_status"],
        "e11_authorized": fc["e11_authorized"],
        "new_llm_calls_authorized": fc["new_llm_calls_authorized"],
        "q1_mechanism_claim_supported": fc["q1_mechanism_claim_supported"],
        "e10_per_criterion_status":
            fc["closure_provenance"]["E10"]["per_criterion_status"],
        "closeout_identity": fc["closeout_identity"],
        "llm_calls": 0,
        "target_access": False,
    }, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="EXT-Q1Q2 final scientific closeout (read-only evidence synthesis).")
    parser.add_argument("--print", dest="do_print", action="store_true",
                        help="Resolve the closeout and print the summary (no writes).")
    parser.add_argument("--write", action="store_true",
                        help="Resolve the closeout and write the six artifacts "
                             f"under {OUTPUT_SUBTREE}/.")
    args = parser.parse_args(argv)
    if not (args.do_print or args.write):
        parser.print_help()
        return 1
    body = evaluate()
    if args.write:
        written = write_artifacts(body)
        print(json.dumps({"written": written}, indent=2))
    _print_summary(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
