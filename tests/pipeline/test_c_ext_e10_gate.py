"""Tests for E10 -- the frozen extension decision gate
(``src/prism_fas/evaluation/c_ext_e10_gate.py``), attempt 3
(strict frozen Section-17 wording; restored evidence).

These tests prove:

* the gate is conjunctive and fail-closed;
* C4 / C5 follow the FROZEN wording exactly -- no stricter post-outcome
  requirement is imposed:
    - C4 PASSES when there is no large q confound OR the completed E8 q-matched
      ablation DESCRIPTIVELY retains the direction of an LLM advantage
      (no inferential significance and no pre-q-match baseline required); such a
      PASS carries a DESCRIPTIVE_ONLY / NO_SIGNIFICANCE_CLAIM qualifier;
    - C5 PASSES when the BA-control / threshold / label-use evidence that WAS
      produced detects no invalidating bug or label leakage; incomplete E3
      controls are a recorded LIMITATION, not an automatic non-establishment;
      C5 FAILS only on an actual invalidating finding;
* C1 / C2 / C3 are unchanged (BLOCKED / NOT_ESTABLISHED / BLOCKED) and are NOT
  softened to improve the gate outcome;
* an existing-but-absent git-ignored milestone is NEVER classified NOT_EXECUTED /
  FAILED from absence alone;
* E9 bank-selection robustness never counts as detector-performance evidence;
* the module imports nothing that could reach the network, a provider, a GPU or
  the target labels, and has no training capability;
* no historical artifact is ever written (and never the preserved attempts/ trees);
* the decision is deterministic / idempotent;
* E11 is authorized only when EVERY frozen criterion passes.

No GPU, no LLM, no network, no target data.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from prism_fas.evaluation import c_ext_e10_gate as e10  # noqa: E402
from prism_fas.evaluation.c_ext_common import ExtPathSafetyError  # noqa: E402

MODULE_PATH = REPO / "src" / "prism_fas" / "evaluation" / "c_ext_e10_gate.py"


# --------------------------------------------------------------------------- #
# fixture-tree builder
# --------------------------------------------------------------------------- #

def _write(root: Path, relpath: str, obj) -> None:
    p = root / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _e7_closure(*, blocked: bool, acer_by_fold: dict | None = None,
                is_target_acer_evidence: bool | None = None) -> dict:
    if is_target_acer_evidence is None:
        is_target_acer_evidence = acer_by_fold is not None
    fs = "SCIENTIFICALLY_BLOCKED_AFTER_RESERVE_CAP" if blocked else "CLOSED_VALID"
    disp = "BLOCKED_BY_E7_CORE_BANK" if blocked else "CLOSED_VALID"
    e7det: dict = {"per_condition": {
        "G-RND": {"disposition": disp}, "G-DET": {"disposition": disp},
        "G-LLM": {"disposition": disp}, "G-LLM-SHUFFLE-A": {"disposition": disp},
        "G-REALONLY": {"disposition": "READY"}}}
    if acer_by_fold is not None:
        e7det["target_acer_by_fold"] = acer_by_fold
    return {
        "final_scientific_interpretation": {
            "core_matched_banks_produced": "0 / 3" if blocked else "3 / 3",
            "is_target_acer_evidence": is_target_acer_evidence},
        "folds": {f: {"final_status": fs} for f in e10.FOLDS},
        "downstream_dependency_classification": {"E7_detector_experiment": e7det},
        "integrity_and_leakage_statement": {
            "target_access": False, "llm_api_calls": 0}}


def _e6_closure(*, infeasible: bool = True) -> dict:
    return {"E6_V2_STATUS": ("CLOSED_TRUE_FROZEN_MATCHED_BANK_INFEASIBILITY"
                             if infeasible else "CLOSED_FEASIBLE"),
            "E6_V2_READY_FOR_TRAINING": not infeasible,
            "E6_V2_TRAINING_BLOCK_REASON": "x" if infeasible else None,
            "TARGET_ACCESS": False, "LLM_API_CALLS": 0}


def _e2_quality(*, smd_llm_rnd=-0.33, smd_llm_det=None) -> dict:
    if smd_llm_det is None:
        smd_llm_det = smd_llm_rnd - 0.03
    return {"q_distribution": {
        "pairwise_smd": [
            {"arm_a": "LLM", "arm_b": "RND", "smd": smd_llm_rnd},
            {"arm_a": "LLM", "arm_b": "DET", "smd": smd_llm_det},
            {"arm_a": "RND", "arm_b": "DET", "smd": -0.05}],
        "q_overall_by_arm": {"LLM": {"mean": 0.71}, "DET": {"mean": 0.75},
                             "RND": {"mean": 0.74}}}}


def _e3_ba(*, controls_run: bool = False) -> dict:
    per = ({"COMPLETE_CPU": ["historical_ba_sep_resummary_ddof1",
                             "C0_same_dataset_null", "C1_cross_domain_real",
                             "C3_matched_domain_syn_real", "C2_pipeline_noop"],
            "GPU_REQUIRED": [], "UNSUPPORTED": []}
           if controls_run else
           {"COMPLETE_CPU": ["historical_ba_sep_resummary_ddof1"],
            "GPU_REQUIRED": ["C0_same_dataset_null", "C1_cross_domain_real",
                             "C3_matched_domain_syn_real"],
            "UNSUPPORTED": ["C2_pipeline_noop"]})
    return {"status": "E3_STATUS", "per_control": per,
            "target_labels_accessed": False,
            "scientific_interpretation_boundary": "descriptive only"}


def _e4_transfer(*, f2_f3="GPU_REQUIRED") -> dict:
    arm = {"video": {"acer_gap_t0_minus_oracle": {"mean": 0.005}}}
    return {"fold": "EXT-F1", "status": "E4_STATUS",
            "aggregates_ddof1": {"LLM": arm, "DET": arm, "RND": arm},
            "feasibility_audit": {"EXT_F2_F3_threshold_transfer": {"status": f2_f3}},
            "target_labels_accessed": True,
            "scientific_interpretation_boundary": "EXT-F1 only"}


def _e5_result(*, acer_mean=0.20, threshold_fitted=False) -> dict:
    return {"arm": "REAL_ONLY", "fold": "EXT-F1",
            "aggregate": {"acer": {"mean": acer_mean, "sample_sd": 0.005},
                          "roc_auc": {"mean": 0.88}, "seed_count": 5},
            "target_threshold_fitted": threshold_fitted,
            "target_scoring_executed": True,
            "threshold_rule": "FROZEN_ROOT_CALIBRATION_SOURCE_DEV",
            "target_labels_accessed": True}


def _e8_closure(*, acer_means, best="LLM") -> dict:
    return {"acer_means_by_arm": acer_means,
            "best_arm_by_acer_mean_descriptive_only": best,
            "inferential_status": "NOT_CLAIMED",
            "statistical_significance_claimed": False,
            "llm_superiority_supported": False,
            "pre_qmatch_frozen_f1_comparison": {"available": False},
            "ranking_change_vs_pre_qmatch": "UNAVAILABLE"}


def _e8_paired(*, det_llm_pos=2, rnd_llm_pos=3) -> dict:
    def blk(pos, mean):
        return {"mean_delta": mean, "count_positive": pos, "count_negative": 5 - pos,
                "count_zero": 0, "n": 5}
    return {"descriptive_seed_summary": {"comparisons": {
        "DET_vs_LLM": blk(det_llm_pos, 0.002),
        "RND_vs_LLM": blk(rnd_llm_pos, 0.003),
        "RND_vs_DET": blk(3, 0.001)}}}


def _e8_label_use(*, leakage=False) -> dict:
    return {"e8_uses_target_labels_in_training_or_g7": bool(leakage),
            "e8_g8_structurally_isolated_from_training_and_g7": not leakage,
            "is_first_ever_blind_reveal": False}


def _e9_closure() -> dict:
    return {"status": "CLOSED", "realizations_ok": 9, "realizations_total": 9,
            "realizations_blocked": 0, "llm_calls": 0, "target_access": False,
            "target_features_accessed": False, "target_labels_accessed": False}


def _e0_hypothesis() -> dict:
    return {"claim_ceiling_without_e11": "NO general superiority claim",
            "diagnostic_families_not_in_holm": ["E3", "E4"],
            "e11_rule": "new family before any API call",
            "e8_trigger": {"rule": "|SMD(q)| >= 0.25"},
            "primary_family": {"hypotheses": [
                {"id": "EXT-H1", "contrast": "LLM vs RND"},
                {"id": "EXT-H2", "contrast": "LLM vs DET"},
                {"id": "EXT-H3", "contrast": "LLM vs REAL-ONLY"},
                {"id": "EXT-H4", "contrast": "LLM-original vs LLM-SHUFFLE"}]}}


def build_tree(root: Path, *, e7_blocked=True, acer_by_fold=None, e6_infeasible=True,
               e2=True, e3_controls_run=False, e4=True, e4_f2_f3="GPU_REQUIRED",
               e5=True, e5_acer=0.20, e5_threshold_fitted=False,
               e8_present=True, e8_acer_means=None, e8_det_llm_pos=2,
               e8_rnd_llm_pos=3, e8_label_leakage=False, smd_llm_rnd=-0.33,
               integrity=None, e0=True, e9=True) -> None:
    S = e10.EVIDENCE_SOURCES
    if e0:
        _write(root, S["e0_hypothesis_family"], _e0_hypothesis())
        _write(root, S["e0_validation"], {"milestone_status": "COMPLETE",
                                          "e0_scientific_content_status": "COMPLETE_AND_LOCKED"})
    if e2:
        _write(root, S["e2_quality_analysis"], _e2_quality(smd_llm_rnd=smd_llm_rnd))
    _write(root, S["e3_ba_controls"], _e3_ba(controls_run=e3_controls_run))
    if e4:
        _write(root, S["e4_threshold_transfer"], _e4_transfer(f2_f3=e4_f2_f3))
    if e5:
        _write(root, S["e5_target_scoring_result"],
               _e5_result(acer_mean=e5_acer, threshold_fitted=e5_threshold_fitted))
        _write(root, S["e5_real_only_lock"], {"arm": "REAL_ONLY"})
    _write(root, S["e6_v2_final_closure"], _e6_closure(infeasible=e6_infeasible))
    _write(root, S["e7_three_fold_scientific_closure"],
           _e7_closure(blocked=e7_blocked, acer_by_fold=acer_by_fold))
    _write(root, S["e7_e8_trigger_record"],
           {"E8_TRIGGER_FROM_E2": True, "E8_EXECUTED": e8_present,
            "training_performed": e8_present, "target_access": False, "llm_api_calls": 0})
    _write(root, S["e8_training_integration_audit"],
           {"status": "CLOSED" if e8_present else "READY", "training_performed": e8_present,
            "target_access": False})
    if e8_present:
        means = e8_acer_means or {"LLM": 0.3487, "DET": 0.3506, "RND": 0.3516}
        _write(root, S["e8_target_evaluation_closure"], _e8_closure(acer_means=means))
        _write(root, S["e8_target_score_result"],
               {"arm_summary": {a: {"acer_mean": v} for a, v in means.items()}})
        _write(root, S["e8_target_paired_comparisons"],
               _e8_paired(det_llm_pos=e8_det_llm_pos, rnd_llm_pos=e8_rnd_llm_pos))
        _write(root, S["e8_target_label_use_record"],
               _e8_label_use(leakage=e8_label_leakage))
        _write(root, S["e8_target_prediction_lockset"], {"identity": "x"})
    if e9:
        _write(root, S["e9_closure"], _e9_closure())
        _write(root, S["e9_bank_stability"], {"arms": {"LLM": {"note": "Descriptive only."}}})
    if integrity is not None:
        _write(root, e10.OPTIONAL_POSITIVE_EVIDENCE["e10_integrity_diagnostics"], integrity)


ALL_PASS_ACER = {
    "EXT-F1": {"LLM": 0.10, "DET": 0.20, "RND": 0.25, "REAL_ONLY": 0.30,
               "LLM_ORIGINAL": 0.10, "LLM_SHUFFLE_A": 0.22},
    "EXT-F2": {"LLM": 0.12, "DET": 0.19, "RND": 0.24, "REAL_ONLY": 0.28,
               "LLM_ORIGINAL": 0.12, "LLM_SHUFFLE_A": 0.20},
    "EXT-F3": {"LLM": 0.11, "DET": 0.18, "RND": 0.23, "REAL_ONLY": 0.27,
               "LLM_ORIGINAL": 0.11, "LLM_SHUFFLE_A": 0.19},
}
GOOD_INTEGRITY = {"label_leakage_found": False, "ba_controls_ok": True,
                  "threshold_analysis_ok": True}


def _crit(body: dict, cid: str) -> dict:
    return next(c for c in body["evidence_matrix"]["criteria"]
               if c["criterion_id"] == cid)


# --------------------------------------------------------------------------- #
# real (restored) repository behaviour -- expected conceptual outcome
# --------------------------------------------------------------------------- #

def test_real_repo_per_criterion_status_matches_frozen_wording():
    dec = e10.evaluate()["gate_decision"]
    assert dec["per_criterion_status"] == {
        "C1_CROSS_DOMAIN_SIGNAL": "BLOCKED",
        "C2_VALUE_OVER_REAL_ONLY": "NOT_ESTABLISHED",
        "C3_SEMANTIC_ABLATION": "BLOCKED",
        "C4_QUALITY_CONFOUND": "PASS",
        "C5_INTEGRITY": "PASS",
    }


def test_real_repo_gate_still_fails_conjunctively():
    dec = e10.evaluate()["gate_decision"]
    assert dec["e11_authorized"] is False
    assert dec["new_llm_calls_authorized"] is False
    assert dec["q1_mechanism_claim_supported"] is False
    assert dec["q1_mechanism_claim_status"] == "NOT_SUPPORTED"
    assert dec["llm_calls"] == 0
    assert dec["target_access"] is False
    assert dec["target_labels_accessed"] is False
    assert dec["target_features_accessed"] is False
    ne = [c["criterion_id"] for c in dec["criteria_not_established"]]
    assert set(ne) == {"C1_CROSS_DOMAIN_SIGNAL", "C2_VALUE_OVER_REAL_ONLY",
                       "C3_SEMANTIC_ABLATION"}
    assert set(dec["criteria_positively_established"]) == {"C4_QUALITY_CONFOUND",
                                                          "C5_INTEGRITY"}


def test_real_repo_c4_pass_is_descriptive_only_qualified():
    c = _crit(e10.evaluate(), "C4_QUALITY_CONFOUND")
    assert c["status"] == e10.STATUS_PASS
    assert c["resolution_label"] == "PASS_DESCRIPTIVE_ONLY_NO_SIGNIFICANCE_CLAIM"
    blob = c["reason"] + " ".join(c["observed_facts"])
    # authoritative means read from artifact, not hardcoded assumptions
    assert "0.3487" in blob and "0.3506" in blob and "0.3516" in blob
    assert "direction retained) = True" in blob or "retains the direction" in blob
    assert "NOT_CLAIMED" in blob
    assert "neither inferential significance" in c["reason"]
    assert c["missing_evidence_due_to_upstream_scientific_block"] is False


def test_real_repo_c5_pass_with_limitations_no_invalidating_finding():
    c = _crit(e10.evaluate(), "C5_INTEGRITY")
    assert c["status"] == e10.STATUS_PASS
    assert c["resolution_label"] == "PASS_WITH_LIMITATIONS_NO_INVALIDATING_BUG_OR_LEAKAGE"
    assert "no bug and NO label leakage".lower() in c["reason"].lower() \
        or "detects NO bug and NO label leakage" in c["reason"]
    # incomplete E3 controls are recorded as a LIMITATION, not an invalidator
    lim = [o for o in c["observed_facts"] if o.startswith("LIMITATION:")]
    assert any("delta_BA_excess" in o for o in lim)
    assert any("EXT-F1 only" in o for o in lim)


def test_real_repo_c1_c2_c3_unchanged_and_upstream_blocked():
    body = e10.evaluate()
    assert _crit(body, "C1_CROSS_DOMAIN_SIGNAL")["status"] == e10.STATUS_BLOCKED
    assert _crit(body, "C2_VALUE_OVER_REAL_ONLY")["status"] == e10.STATUS_NOT_ESTABLISHED
    assert _crit(body, "C3_SEMANTIC_ABLATION")["status"] == e10.STATUS_BLOCKED
    for cid in ("C1_CROSS_DOMAIN_SIGNAL", "C2_VALUE_OVER_REAL_ONLY",
                "C3_SEMANTIC_ABLATION"):
        assert _crit(body, cid)["missing_evidence_due_to_upstream_scientific_block"] is True
    # C1 still refuses to count the single-fold E8 result
    assert "NOT counted" in " ".join(_crit(body, "C1_CROSS_DOMAIN_SIGNAL")["observed_facts"])


def test_real_repo_e8_executed_and_scored_descriptive_only():
    e8 = e10.evaluate()["closure"]["milestone_status_audit"]["E8"]
    assert e8["status"] == e10.MS_EXECUTED_AND_SCORED
    assert e8["inferential_status"] == "NOT_CLAIMED"
    assert e8["statistical_significance_claimed"] is False
    assert e8["llm_superiority_supported"] is False
    m = e8["acer_means_by_arm"]
    assert m["LLM"] < m["DET"] < m["RND"]
    assert round(m["LLM"], 4) == 0.3487


def test_real_repo_milestone_audit_e3_e4_e5_e7():
    a = e10.evaluate()["closure"]["milestone_status_audit"]
    assert a["E3"]["status"] == e10.MS_FEASIBILITY_AUDIT_ONLY
    assert a["E3"]["delta_ba_excess_computable"] is False
    assert a["E4"]["status"] == e10.MS_PARTIAL_F1_ONLY
    assert a["E5"]["status"] == e10.MS_PARTIAL_F1_ONLY
    assert round(a["E5"]["acer_mean"], 4) == 0.1999
    assert a["E7"]["status"] == e10.MS_SCIENTIFICALLY_BLOCKED
    assert a["E7"]["core_matched_banks_produced"] == "0 / 3"


def test_real_repo_corrections_vs_prior_attempts_recorded():
    corr = e10.evaluate()["closure"]["corrections_vs_prior_attempts"]
    topics = " ".join(c["topic"] for c in corr)
    assert "C4 QUALITY CONFOUND -- frozen wording" in topics
    assert "C5 INTEGRITY -- frozen wording" in topics
    c4 = next(c for c in corr if c["topic"].startswith("C4"))
    assert "PASS" in c4["attempt3"]
    assert "pre-q-match baseline" in c4["attempt2"]
    c5 = next(c for c in corr if c["topic"].startswith("C5"))
    assert "PASS" in c5["attempt3"]


def test_real_repo_e9_not_detector_performance_evidence():
    body = e10.evaluate()
    assert body["gate_decision"]["e9_counts_as_detector_performance_evidence"] is False
    for c in body["evidence_matrix"]["criteria"]:
        joined = " ".join(c["evidence_artifacts_used"] + c["evidence_artifacts_missing"])
        assert "e9_bank_robustness" not in joined


# --------------------------------------------------------------------------- #
# C4 -- frozen wording (no significance / no baseline required)
# --------------------------------------------------------------------------- #

def test_c4_branch_a_small_smd_passes(tmp_path):
    build_tree(tmp_path, smd_llm_rnd=-0.10, e8_present=False)
    c = _crit(e10.evaluate(tmp_path), "C4_QUALITY_CONFOUND")
    assert c["status"] == e10.STATUS_PASS
    assert "no large q confound" in c["reason"]


def test_c4_large_smd_but_descriptive_direction_retained_passes(tmp_path):
    build_tree(tmp_path, smd_llm_rnd=-0.40,
               e8_acer_means={"LLM": 0.3487, "DET": 0.3506, "RND": 0.3516},
               e8_det_llm_pos=2, e8_rnd_llm_pos=3)
    c = _crit(e10.evaluate(tmp_path), "C4_QUALITY_CONFOUND")
    assert c["status"] == e10.STATUS_PASS
    assert c["resolution_label"] == "PASS_DESCRIPTIVE_ONLY_NO_SIGNIFICANCE_CLAIM"
    # no inferential significance and no pre-q-match baseline were demanded
    assert "neither inferential significance" in c["reason"]
    assert "pre-q-match baseline" in c["reason"]


def test_c4_large_smd_and_direction_not_retained_fails(tmp_path):
    build_tree(tmp_path, smd_llm_rnd=-0.40,
               e8_acer_means={"LLM": 0.40, "DET": 0.20, "RND": 0.22})
    c = _crit(e10.evaluate(tmp_path), "C4_QUALITY_CONFOUND")
    assert c["status"] == e10.STATUS_FAIL
    assert c["resolution_label"] == "EVIDENCE_PRESENT_DIRECTION_NOT_RETAINED"


def test_c4_large_smd_and_e8_absent_is_not_established_not_failure(tmp_path):
    build_tree(tmp_path, smd_llm_rnd=-0.40, e8_present=False)
    c = _crit(e10.evaluate(tmp_path), "C4_QUALITY_CONFOUND")
    assert c["status"] == e10.STATUS_NOT_ESTABLISHED
    assert c["status"] != e10.STATUS_FAIL
    assert c["resolution_label"] == "NOT_ESTABLISHED_EVIDENCE_NOT_PRESENT_IN_CHECKOUT"


# --------------------------------------------------------------------------- #
# C5 -- frozen wording (no-invalidating-bug/leakage criterion)
# --------------------------------------------------------------------------- #

def test_c5_incomplete_e3_controls_do_not_block_pass(tmp_path):
    build_tree(tmp_path, e3_controls_run=False)   # C0/C1/C3 GPU-required, not run
    c = _crit(e10.evaluate(tmp_path), "C5_INTEGRITY")
    assert c["status"] == e10.STATUS_PASS
    assert c["resolution_label"] == "PASS_WITH_LIMITATIONS_NO_INVALIDATING_BUG_OR_LEAKAGE"
    assert any("delta_BA_excess" in o for o in c["observed_facts"])


def test_c5_label_leakage_in_label_use_record_fails(tmp_path):
    build_tree(tmp_path, e8_label_leakage=True)
    c = _crit(e10.evaluate(tmp_path), "C5_INTEGRITY")
    assert c["status"] == e10.STATUS_FAIL
    assert "label leakage" in c["reason"].lower()


def test_c5_target_threshold_fitted_fails(tmp_path):
    build_tree(tmp_path, e5_threshold_fitted=True)
    c = _crit(e10.evaluate(tmp_path), "C5_INTEGRITY")
    assert c["status"] == e10.STATUS_FAIL
    assert "target threshold was fitted" in c["reason"]


def test_c5_no_integrity_evidence_at_all_is_not_established(tmp_path):
    build_tree(tmp_path, e3_controls_run=False, e4=False, e8_present=False, e5=False)
    # remove the E3 file too so there is genuinely no integrity evidence
    (tmp_path / e10.EVIDENCE_SOURCES["e3_ba_controls"]).unlink()
    c = _crit(e10.evaluate(tmp_path), "C5_INTEGRITY")
    assert c["status"] == e10.STATUS_NOT_ESTABLISHED
    assert c["resolution_label"] == "NOT_ESTABLISHED_NO_INTEGRITY_EVIDENCE"


def test_c5_dedicated_diagnostics_leakage_fails(tmp_path):
    build_tree(tmp_path, integrity={"label_leakage_found": True,
                                    "ba_controls_ok": True,
                                    "threshold_analysis_ok": True})
    c = _crit(e10.evaluate(tmp_path), "C5_INTEGRITY")
    assert c["status"] == e10.STATUS_FAIL


# --------------------------------------------------------------------------- #
# conjunctive / fail-closed structure
# --------------------------------------------------------------------------- #

def test_all_five_criteria_pass_authorizes_e11(tmp_path):
    build_tree(tmp_path, e7_blocked=False, acer_by_fold=ALL_PASS_ACER,
               e3_controls_run=True, e4_f2_f3="COMPLETE",
               e8_acer_means={"LLM": 0.10, "DET": 0.20, "RND": 0.22},
               smd_llm_rnd=-0.10, integrity=GOOD_INTEGRITY)
    dec = e10.evaluate(tmp_path)["gate_decision"]
    assert dec["per_criterion_status"] == {c: "PASS" for c in e10.CRITERION_IDS}
    assert dec["e11_authorized"] is True
    assert dec["new_llm_calls_authorized"] is True
    assert dec["q1_mechanism_claim_supported"] is True
    assert dec["q1_mechanism_claim_status"] == "SUPPORTED_PENDING_E11"
    assert dec["llm_calls"] == 0
    assert dec["target_labels_accessed"] is False


@pytest.mark.parametrize("drop", list(e10.CRITERION_IDS))
def test_gate_is_conjunctive_single_failing_criterion_blocks_e11(tmp_path, drop):
    kw = dict(e7_blocked=False,
              acer_by_fold={f: dict(v) for f, v in ALL_PASS_ACER.items()},
              e3_controls_run=True, e4_f2_f3="COMPLETE",
              e8_acer_means={"LLM": 0.10, "DET": 0.20, "RND": 0.22},
              smd_llm_rnd=-0.10, integrity=dict(GOOD_INTEGRITY))
    if drop == "C1_CROSS_DOMAIN_SIGNAL":
        for f in kw["acer_by_fold"]:
            kw["acer_by_fold"][f]["LLM"] = 0.99
    elif drop == "C2_VALUE_OVER_REAL_ONLY":
        for f in kw["acer_by_fold"]:
            kw["acer_by_fold"][f]["REAL_ONLY"] = 0.01
    elif drop == "C3_SEMANTIC_ABLATION":
        for f in kw["acer_by_fold"]:
            kw["acer_by_fold"][f]["LLM_SHUFFLE_A"] = 0.01
    elif drop == "C4_QUALITY_CONFOUND":
        kw["smd_llm_rnd"] = -0.40
        kw["e8_acer_means"] = {"LLM": 0.40, "DET": 0.20, "RND": 0.22}  # not retained
    elif drop == "C5_INTEGRITY":
        kw["integrity"] = None
        kw["e8_label_leakage"] = True   # an actual invalidating finding
    build_tree(tmp_path, **kw)
    dec = e10.evaluate(tmp_path)["gate_decision"]
    assert dec["e11_authorized"] is False
    assert drop in [c["criterion_id"] for c in dec["criteria_not_established"]]


def test_c1_c2_c3_not_softened_by_frozen_wording_change(tmp_path):
    # real-repo shape: E7 blocked, no 3-fold ACER, E8 present
    build_tree(tmp_path, e7_blocked=True, acer_by_fold=None)
    body = e10.evaluate(tmp_path)
    assert _crit(body, "C1_CROSS_DOMAIN_SIGNAL")["status"] == e10.STATUS_BLOCKED
    assert _crit(body, "C2_VALUE_OVER_REAL_ONLY")["status"] == e10.STATUS_NOT_ESTABLISHED
    assert _crit(body, "C3_SEMANTIC_ABLATION")["status"] == e10.STATUS_BLOCKED
    assert body["gate_decision"]["e11_authorized"] is False


# --------------------------------------------------------------------------- #
# missing evidence vs real negative result / absence never NOT_EXECUTED
# --------------------------------------------------------------------------- #

def test_absent_e8_target_eval_is_not_classified_not_executed(tmp_path):
    build_tree(tmp_path, e8_present=False)
    e8 = e10.evaluate(tmp_path)["closure"]["milestone_status_audit"]["E8"]
    assert e8["status"] == e10.MS_EVIDENCE_ABSENT
    assert e8["status"] != "NOT_EXECUTED"
    assert e8["detector_trained"] is False
    assert "not executed" in e8["note"].lower()  # explicitly says it is NOT that


@pytest.mark.parametrize("milestone_key,src_keys", [
    ("E2", ["e2_quality_analysis"]),
    ("E3", ["e3_ba_controls"]),
    ("E4", ["e4_threshold_transfer"]),
    ("E5", ["e5_target_scoring_result", "e5_real_only_lock"]),
    ("E8", ["e8_target_evaluation_closure", "e8_target_score_result",
            "e8_target_paired_comparisons", "e8_target_label_use_record",
            "e8_target_prediction_lockset"]),
    ("E9", ["e9_closure", "e9_bank_stability"]),
])
def test_no_milestone_absence_yields_failure_or_not_executed(tmp_path, milestone_key,
                                                             src_keys):
    build_tree(tmp_path)
    for k in src_keys:
        (tmp_path / e10.EVIDENCE_SOURCES[k]).unlink()
    st = e10.evaluate(tmp_path)["closure"]["milestone_status_audit"][milestone_key]["status"]
    assert st in (e10.MS_EVIDENCE_ABSENT, e10.MS_PRESENT_UNREADABLE)
    for bad in ("NOT_EXECUTED", "FAIL", "FAILED", "NOT_RUN"):
        assert bad not in st


def test_missing_evidence_is_blocked_not_fail(tmp_path):
    build_tree(tmp_path, e7_blocked=True, acer_by_fold=None, e8_present=False)
    for cid in ("C1_CROSS_DOMAIN_SIGNAL", "C3_SEMANTIC_ABLATION"):
        c = _crit(e10.evaluate(tmp_path), cid)
        assert c["status"] == e10.STATUS_BLOCKED
        assert c["status"] != e10.STATUS_FAIL


def test_real_negative_result_is_fail_not_blocked(tmp_path):
    bad = {f: {"LLM": 0.90, "DET": 0.20, "RND": 0.25, "REAL_ONLY": 0.10,
               "LLM_ORIGINAL": 0.90, "LLM_SHUFFLE_A": 0.20} for f in e10.FOLDS}
    build_tree(tmp_path, e7_blocked=False, acer_by_fold=bad, e3_controls_run=True)
    body = e10.evaluate(tmp_path)
    for cid in ("C1_CROSS_DOMAIN_SIGNAL", "C2_VALUE_OVER_REAL_ONLY",
                "C3_SEMANTIC_ABLATION"):
        assert _crit(body, cid)["status"] == e10.STATUS_FAIL
    assert body["gate_decision"]["evidence_available_but_negative"] is True
    assert body["gate_decision"]["e11_authorized"] is False


# --------------------------------------------------------------------------- #
# determinism / idempotency / confinement
# --------------------------------------------------------------------------- #

def test_decision_is_deterministic_on_same_inputs():
    a, b = e10.evaluate(), e10.evaluate()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a["closure"]["gate_identity"] == b["closure"]["gate_identity"]


def test_write_artifacts_is_idempotent_byte_for_byte(tmp_path):
    build_tree(tmp_path)
    w1 = e10.write_artifacts(e10.evaluate(tmp_path), root=tmp_path)
    first = {k: (tmp_path / v).read_bytes() for k, v in w1.items()}
    w2 = e10.write_artifacts(e10.evaluate(tmp_path), root=tmp_path)
    second = {k: (tmp_path / v).read_bytes() for k, v in w2.items()}
    assert first == second


def test_all_outputs_confined_to_e10_subtree_and_never_attempts(tmp_path):
    build_tree(tmp_path)
    written = e10.write_artifacts(e10.evaluate(tmp_path), root=tmp_path)
    for rel in written.values():
        assert rel.startswith(e10.OUTPUT_SUBTREE + "/")
        assert "/attempts/" not in rel


def test_print_mode_writes_nothing():
    gate_dir = REPO / "reports/c_ext_q1q2_v1/e10_gate"
    before = sorted(p.name for p in gate_dir.glob("*")) if gate_dir.exists() else []
    assert e10.main(["--print"]) == 0
    after = sorted(p.name for p in gate_dir.glob("*")) if gate_dir.exists() else []
    assert before == after


def test_prior_attempt_evidence_is_never_written_by_e10():
    for rel in (e10.INPUT_AUDIT_RELPATH, e10.EVIDENCE_MATRIX_RELPATH,
                e10.GATE_DECISION_RELPATH, e10.CLOSURE_RELPATH,
                e10.EVIDENCE_MANIFEST_RELPATH):
        assert "/attempts/" not in rel


def test_write_guard_rejects_non_e10_paths():
    from prism_fas.evaluation.c_ext_common import write_json_atomic
    for bad in ("reports/full/c6/HACK.json", "reports/c3/HACK.json"):
        with pytest.raises(ExtPathSafetyError):
            write_json_atomic(bad, {"x": 1})


# --------------------------------------------------------------------------- #
# no forbidden capability / imports
# --------------------------------------------------------------------------- #

FORBIDDEN_IMPORT_ROOTS = {
    "torch", "tensorflow", "jax", "requests", "urllib", "http", "socket",
    "aiohttp", "httpx", "websocket", "grpc", "boto3", "openai", "anthropic",
    "google", "vertexai", "cohere", "sklearn", "pandas", "pyarrow", "yaml",
}
FORBIDDEN_IMPORT_SUBMODULES = {
    "prism_fas.detector", "prism_fas.synthesis", "prism_fas.providers",
    "prism_fas.llm", "prism_fas.recipes",
}


def test_module_imports_no_network_provider_gpu_or_target_surface():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    for m in mods:
        assert m.split(".")[0] not in FORBIDDEN_IMPORT_ROOTS, f"forbidden: {m}"
        for sub in FORBIDDEN_IMPORT_SUBMODULES:
            assert not m.startswith(sub), f"forbidden: {m}"
    assert {m for m in mods if m.startswith("prism_fas")} == {
        "prism_fas.evaluation.c_ext_common"}


def test_module_has_no_training_capability():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            continue
        if isinstance(node, ast.Attribute):
            names.append(node.attr)
        if isinstance(node, ast.Name):
            names.append(node.id)
    for token in ("M9Trainer", "backward", "load_state_dict", "state_dict",
                  "build_matched_banks", "SyntheticBankGenerator", "fit", "step",
                  "zero_grad", "optimizer"):
        assert token not in names, f"training-capability symbol: {token!r}"
    assert "torch" not in " ".join(names)


def test_module_reads_only_frozen_reports_under_ext_root():
    for rel in list(e10.EVIDENCE_SOURCES.values()) + \
            list(e10.OPTIONAL_POSITIVE_EVIDENCE.values()):
        assert rel.startswith("reports/c_ext_q1q2_v1/")


def test_no_target_label_or_siw_resolution_in_module():
    src = MODULE_PATH.read_text(encoding="utf-8").lower()
    for token in ("siw_mv2", "siw-mv2", "target_label_root", "target_feature_root",
                  "data/evaluation_only", "prism_target_eval"):
        assert token not in src, f"target-resolution token present: {token!r}"


# --------------------------------------------------------------------------- #
# criteria table integrity
# --------------------------------------------------------------------------- #

def test_frozen_criteria_table_is_the_five_spec_conditions():
    ids = [c["criterion_id"] for c in e10.FROZEN_CRITERIA]
    assert ids == list(e10.CRITERION_IDS)
    assert len(ids) == 5 and all(c["required"] for c in e10.FROZEN_CRITERIA)
    txt = {c["criterion_id"]: c["frozen_criterion_text"] for c in e10.FROZEN_CRITERIA}
    assert "not ranked last in at least 2/3 held-out folds" in txt["C1_CROSS_DOMAIN_SIGNAL"]
    assert "better than REAL-ONLY in at least 2/3 folds" in txt["C2_VALUE_OVER_REAL_ONLY"]
    assert "LLM-SHUFFLE-A" in txt["C3_SEMANTIC_ABLATION"]
    assert "q confound" in txt["C4_QUALITY_CONFOUND"]
    assert "label leakage" in txt["C5_INTEGRITY"]
    # frozen-wording alignment recorded in positive_evidence_requires
    c4pe = next(c for c in e10.FROZEN_CRITERIA
                if c["criterion_id"] == "C4_QUALITY_CONFOUND")["positive_evidence_requires"]
    assert "does NOT require inferential significance" in c4pe
    assert "does NOT require a comparable pre-q-match baseline" in c4pe
    c5pe = next(c for c in e10.FROZEN_CRITERIA
                if c["criterion_id"] == "C5_INTEGRITY")["positive_evidence_requires"]
    assert "does NOT require every BA control to have completed" in c5pe


def test_only_pass_status_is_positive_for_the_gate():
    assert e10.POSITIVE_STATUS == e10.STATUS_PASS
    for st in (e10.STATUS_FAIL, e10.STATUS_BLOCKED, e10.STATUS_NOT_ESTABLISHED):
        assert st != e10.POSITIVE_STATUS
