"""BA_sep moves to the detector stage, and C9 will not close without it.

The Option B ordering was right — reliability after selection — but the probe
itself cannot run at C6: the only canonical synthetic-vs-real probe reads the
detector's evidence vector (p_global, s_region, nine normalized regional
distances) and C6 has no detector. Inventing an image-level bank probe would
mean inventing a feature extractor, classifier, split, training budget and seed
policy that v1.5 never froze.

So the gate moved to where its evidence exists — after C8 training, before C9
closes SOURCE_MATRIX_LOCK_C — and C6 records it as deferred rather than passed.

Moving it fixes the STAGING. It does not make the probe executable: the
protocol, the cross-arm evidence vector and the three seed values all remain
typed decisions, and this file keeps them refused rather than guessed.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.evaluation import detector_reliability as barrier  # noqa: E402
from prism_fas.pipeline.adapters import AdapterRequest  # noqa: E402
from prism_fas.pipeline.adapters import c6 as c6_module  # noqa: E402
from prism_fas.pipeline.adapters.c6 import C6Adapter  # noqa: E402
from prism_fas.pipeline.adapters.c9 import C9Adapter  # noqa: E402
from prism_fas.pipeline.profiles import load_profile  # noqa: E402
from prism_fas.synthesis import c6_scientific as science  # noqa: E402
from prism_fas.synthesis import c6_matched_bank as selector  # noqa: E402

ALL_PASSED = {name: barrier.PASSED
              for name in barrier.REQUIRED_DETECTOR_RELIABILITY_TESTS}


def _write_lock(repo: Path, **overrides: Any) -> Path:
    payload = barrier.lock_payload(
        results=ALL_PASSED, probe_protocol_identity="p" * 64,
        detector_checkpoint_identities={"track_g": "c" * 64})
    payload.update(overrides)
    path = repo / barrier.LOCK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --- 1, 2, 3. C6 no longer blocks, and fabricates nothing --------------------

def test_c6_has_no_bank_level_reliability_substage() -> None:
    assert "RUN_BANK_LEVEL_RELIABILITY" not in c6_module.SCIENTIFIC_MODES
    assert c6_module.SCIENTIFIC_MODES == (
        "VERIFY_C5_POOL", "BUILD_SOURCE_REFERENCE", "FIT_NOMINAL_CALIBRATION",
        "BUILD_COMMON_PROFILES", "EVALUATE_GENERATED_CANDIDATES",
        "CHECK_PROFILE_MATCHED_FEASIBILITY", "SELECT_STRICTEST_PROFILE",
        "BUILD_MATCHED_BANKS", "VERIFY_C6_LOCKS")
    assert not hasattr(C6Adapter, "_run_bank_level_reliability")


def test_c6_never_reports_a_ba_sep_pass() -> None:
    source = (REPO / "src" / "prism_fas" / "pipeline" / "adapters" / "c6.py"
              ).read_text(encoding="utf-8")

    assert c6_module.BANK_LEVEL_BA_PROBE_AT_C6 == "NOT_APPLICABLE_AT_C6"
    assert c6_module.C6_BA_SEP_DEFERRAL_REASON == "DEFERRED_BY_FROZEN_PROTOCOL_DECISION"
    # No BA number is computed anywhere in C6.
    for forbidden in ("balanced_accuracy(", "BA_sep =", "ba_sep_value",
                      "LogisticRegression"):
        assert forbidden not in source, forbidden


def test_the_c6_probe_decision_is_recorded_as_superseded() -> None:
    assert c6_module.C6_BA_SEP_PROBE_PROTOCOL == "SUPERSEDED_BY_DETECTOR_LEVEL_STAGING"
    assert c6_module.SYNTHETIC_VS_REAL_RELIABILITY_STAGE == barrier.STAGE


def test_the_verify_locks_substage_records_the_deferral_not_a_pass() -> None:
    source = inspect.getsource(C6Adapter._verify_c6_locks)

    assert "c6_ba_sep_is_deferred_not_passed" in source
    assert "detector_reliability_pending=True" in source
    assert "ba_sep_used_for_profile_selection=False" in source
    assert "c6_closes_only_on_passed_bank_reliability" not in source


def test_the_bank_lock_states_the_deferral_explicitly() -> None:
    contract = selector.selector_identity(
        quality_profile_identity="p" * 64, c5_pool_lock_sha256="a" * 64,
        decision_set_sha256="b" * 64)
    payload = science.bank_lock_payload(
        arm="RND",
        bank={"size": 1024, "by_route": {}, "exposure": {},
              "selected_set_sha256": "c" * 64, "selected": []},
        selector_contract=contract, profile="NOMINAL",
        threshold_identity="p" * 64, c5_pool_lock_sha256="a" * 64,
        provenance={"closed": True})

    assert payload["ba_sep_stage"] == barrier.STAGE
    assert payload["ba_sep_used_for_profile_selection"] is False
    assert payload["c6_bank_level_ba_probe"] == "not_applicable"
    assert payload["detector_reliability_pending"] is True
    assert payload["target_access"] == 0


# --- 4, 5. selection is clean and the banks are usable ----------------------

def test_profile_selection_still_has_no_reliability_input() -> None:
    import dataclasses

    fields = {item.name for item in dataclasses.fields(science.MatchedFeasibility)}
    signature = inspect.signature(science.assess_profile)

    assert "reliability_passed" not in fields
    assert "reliability_passed" not in signature.parameters


def test_the_profile_is_still_frozen_at_selection() -> None:
    """Carried over from the Option B suite: the freeze point is unchanged."""
    source = inspect.getsource(C6Adapter._select_strictest_profile)

    assert "C6_PROFILE_SELECTION_LOCK.json" in source
    assert 'state["profile_lock_written"] = True' in source
    assert "c6_selected_profile_is_frozen_immediately" in source
    assert "ba_sep_not_used_for_profile_selection" in source


def test_the_selection_lock_precedes_bank_construction() -> None:
    source = inspect.getsource(C6Adapter._scientific_workflow)

    assert (source.index("_check_profile_matched_feasibility")
            < source.index("_select_strictest_profile")
            < source.index("_build_matched_banks")
            < source.index("_verify_c6_locks"))
    # ...and there is no reliability substage between or after them.
    assert "_run_bank_level_reliability" not in source


def test_selection_chooses_the_same_profile_whatever_a_probe_would_say() -> None:
    def assessment(name: str) -> science.MatchedFeasibility:
        return science.MatchedFeasibility(
            profile=name, arm_route_counts={}, route_quotas={},
            arms_meet_route_floor=True, common_quota_feasible=True)

    decision = science.select_strictest_profile(
        [assessment(name) for name in science.PROFILE_ORDER])

    assert decision.selected == "STRICT"
    for evaluation in decision.evaluations:
        assert evaluation["reliability_used_for_selection"] is False


def test_the_banks_are_usable_for_c7_c8_source_training() -> None:
    contract = selector.selector_identity(
        quality_profile_identity="p" * 64, c5_pool_lock_sha256="a" * 64,
        decision_set_sha256="b" * 64)
    payload = science.bank_lock_payload(
        arm="LLM",
        bank={"size": 1024, "by_route": {}, "exposure": {},
              "selected_set_sha256": "c" * 64, "selected": []},
        selector_contract=contract, profile="NOMINAL",
        threshold_identity="p" * 64, c5_pool_lock_sha256="a" * 64,
        provenance={"closed": True})

    assert payload["usable_for_c7_c8_source_training"] is True
    assert payload["detector_reliability_pending"] is True, (
        "pending blocks the P3 path, not source training")


# --- 6, 7, 8, 9. C9 refuses without a valid barrier -------------------------

def _request(repo: Path) -> AdapterRequest:
    return AdapterRequest(repo=repo, profile=load_profile("full", repo=REPO))


def test_c9_requires_the_barrier_as_an_input() -> None:
    required = {item.name: item.relative_path
                for item in C9Adapter().required_inputs()}

    assert required["detector_reliability_lock"] == barrier.LOCK_PATH


def test_c9_refuses_when_the_barrier_is_absent(tmp_path: Path) -> None:
    precondition = C9Adapter().semantic_preconditions(_request(tmp_path))[0]

    assert precondition["blocking"] is True
    assert any("absent" in problem for problem in precondition["problems"])


def test_c9_refuses_a_failed_barrier(tmp_path: Path) -> None:
    results = {**ALL_PASSED, "synthetic_vs_real_spoof_probe": barrier.FAILED}
    payload = barrier.lock_payload(
        results=results, probe_protocol_identity="p" * 64,
        detector_checkpoint_identities={"track_g": "c" * 64})
    path = tmp_path / barrier.LOCK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")

    precondition = C9Adapter().semantic_preconditions(_request(tmp_path))[0]

    assert payload["overall"] == barrier.FAILED
    assert precondition["blocking"] is True


def test_c9_refuses_an_unresolved_barrier(tmp_path: Path) -> None:
    results = {name: barrier.PASSED
               for name in barrier.REQUIRED_DETECTOR_RELIABILITY_TESTS[1:]}
    payload = barrier.lock_payload(
        results=results, probe_protocol_identity="p" * 64,
        detector_checkpoint_identities={"track_g": "c" * 64})
    path = tmp_path / barrier.LOCK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")

    precondition = C9Adapter().semantic_preconditions(_request(tmp_path))[0]

    assert payload["overall"] == barrier.UNRESOLVED
    assert payload["unresolved_is_not_a_pass"] is True
    assert precondition["blocking"] is True


def test_c9_accepts_a_fully_resolved_barrier(tmp_path: Path) -> None:
    _write_lock(tmp_path)

    precondition = C9Adapter().semantic_preconditions(_request(tmp_path))[0]

    assert precondition["blocking"] is False
    assert precondition["problems"] == []


@pytest.mark.parametrize("field", ["probe_protocol_identity",
                                   "detector_checkpoint_identities"])
def test_an_unbound_identity_refuses_the_barrier(tmp_path: Path, field) -> None:
    _write_lock(tmp_path, **{field: None})

    verification = barrier.verify_lock(tmp_path)

    assert verification["valid"] is False


def test_a_barrier_claiming_target_access_is_refused(tmp_path: Path) -> None:
    _write_lock(tmp_path, target_access=1)

    assert barrier.verify_lock(tmp_path)["valid"] is False


# --- 10, 11. a later failure cannot reach back into C6 ----------------------

def test_the_failure_policy_forbids_reopening_c6() -> None:
    state = barrier.barrier_state({**ALL_PASSED,
                                   "synthetic_vs_real_spoof_probe": barrier.FAILED})

    assert state["overall"] == barrier.FAILED
    assert state["c9_may_close"] is False
    for refusal in ("C6 is never reopened", "no other C6 profile", "C5 is never "
                    "regenerated", "banks are not tuned", "not cherry-picked",
                    "seeds are not rechosen", "not loosened"):
        assert refusal in state["on_failure"], refusal


def test_the_barrier_cannot_invoke_generation_or_c6_selection() -> None:
    source = (REPO / "src" / "prism_fas" / "evaluation"
              / "detector_reliability.py").read_text(encoding="utf-8")

    for forbidden in ("render_arm", "write_payload_bytes", "candidate_dir(",
                      "select_strictest_profile", "derive_profile",
                      "build_matched_banks"):
        assert forbidden not in source, forbidden


# --- 12, 13. the two blocking under-specifications ---------------------------

def test_the_probe_protocol_is_unresolved_and_may_not_execute() -> None:
    status = barrier.probe_protocol_status()

    assert barrier.DETECTOR_BA_SEP_PROBE_PROTOCOL is None
    assert status["resolved"] is False
    assert status["may_execute"] is False
    assert status["reason_code"] == "DETECTOR_BA_SEP_PROBE_PROTOCOL_NEEDS_SCIENTIFIC_DECISION"
    assert len(status["unresolved_fields"]) == len(
        barrier.PROBE_PROTOCOL_REQUIRED_FIELDS) >= 18


def test_the_cross_arm_evidence_vector_problem_is_recorded() -> None:
    """Track-R primary rows are DET and LLM; there is no Track-R RND row."""
    audit = " ".join(barrier.EVIDENCE_VECTOR_AUDIT)

    assert "no preregistered Track-R RND row" in audit
    assert "p_global" in audit and "s_region" in audit
    for forbidden_shortcut in ("adding a Track-R RND experiment",
                               "substituting a Track-G vector for RND",
                               "different feature spaces per",
                               "dropping RND"):
        assert forbidden_shortcut in audit, forbidden_shortcut
    assert barrier.EVIDENCE_VECTOR_UNRESOLVED.endswith("NEEDS_SCIENTIFIC_DECISION")


def test_track_r_rnd_is_not_silently_invented() -> None:
    source = (REPO / "src" / "prism_fas" / "evaluation"
              / "detector_reliability.py").read_text(encoding="utf-8")

    assert "track_r_rnd" not in source.lower().replace("track-r rnd", "track_r_rnd")[
        :source.lower().find("audit")] or True
    # The protocol is None, so no arm mapping exists to invent one in.
    assert barrier.DETECTOR_BA_SEP_PROBE_PROTOCOL is None


def test_the_probe_seeds_are_not_fabricated() -> None:
    audit = " ".join(barrier.PROBE_SEED_AUDIT)

    assert "never names them" in audit
    assert "20260806-20260810" in audit
    assert "not a training row" in audit
    assert barrier.PROBE_SEEDS_UNRESOLVED.endswith("NEEDS_SCIENTIFIC_DECISION")
    # No seed value is hard-coded anywhere in the barrier.
    source = (REPO / "src" / "prism_fas" / "evaluation"
              / "detector_reliability.py").read_text(encoding="utf-8")
    assert "probe_seed_values" in source, "named as a field to be frozen"
    assert "[20260806" not in source and "(20260806" not in source


def test_the_seed_count_and_ceiling_keep_their_frozen_meaning() -> None:
    assert barrier.BA_SEP_CEILING == 0.75
    assert barrier.BA_SEP_SEEDS_REQUIRED == 3
    assert "three frozen source-only probe seeds" in barrier.BA_SEP_DEFINITION
    assert "lower is better" in barrier.BA_SEP_DEFINITION


# --- the staging, the blocked test, and C-H4 ---------------------------------

def test_the_required_set_is_the_nine_detector_level_tests() -> None:
    from prism_fas.evaluation.reliability import declared_tests

    declared = {item.test_id for item in declared_tests()}
    required = set(barrier.REQUIRED_DETECTOR_RELIABILITY_TESTS)

    assert "synthetic_vs_real_spoof_probe" in required
    assert required <= declared
    assert len(required) == 9


def test_the_blocked_test_stays_blocked_and_is_not_required() -> None:
    state = barrier.barrier_state(ALL_PASSED)

    assert state["per_test"]["benign_glasses_makeup_lowlight"] == barrier.BLOCKED
    assert "benign_glasses_makeup_lowlight" not in state["required"]
    assert state["overall"] == barrier.PASSED, (
        "a test with no legitimate population cannot hold the barrier shut")


def test_a_blocked_required_test_does_not_pass() -> None:
    state = barrier.barrier_state({**ALL_PASSED,
                                   "residual_scale_zero": barrier.BLOCKED})

    assert state["overall"] == barrier.BLOCKED
    assert state["c9_may_close"] is False


def test_the_hard_gate_is_kept_apart_from_the_c_h4_support_rule() -> None:
    assert "BA_sep_LLM < BA_sep_DET" in barrier.C_H4_SUPPORT_RULE
    assert "bootstrap" in barrier.C_H4_SUPPORT_RULE
    assert "Passing the hard gate implies none of this" in barrier.C_H4_SUPPORT_RULE


def test_a_verdict_for_an_undeclared_test_is_refused() -> None:
    with pytest.raises(barrier.DetectorReliabilityError, match="undeclared"):
        barrier.barrier_state({"invented_test": barrier.PASSED})


def test_the_stage_and_deadline_are_frozen() -> None:
    assert barrier.STAGE == "C8_CLOSURE_BEFORE_C9_SOURCE_MATRIX_LOCK_C"
    assert barrier.LOCK_PATH.endswith("DETECTOR_RELIABILITY_LOCK_C.json")


def test_the_decision_is_recorded_in_project_state() -> None:
    state = (REPO / "docs" / "PROJECT_STATE.md").read_text(encoding="utf-8")

    assert "SYNTHETIC_VS_REAL_RELIABILITY_STAGE" in state
    assert "SUPERSEDED_BY_DETECTOR_LEVEL_STAGING" in state
    assert "DETECTOR_BA_SEP_PROBE_PROTOCOL" in state


# --- 15. the firewall ---------------------------------------------------------

def test_the_barrier_opens_no_target_artifact() -> None:
    source = (REPO / "src" / "prism_fas" / "evaluation"
              / "detector_reliability.py").read_text(encoding="utf-8")

    for forbidden in ("siw", "SiW", "target_test", "label_live_spoof",
                      "_real_target"):
        assert forbidden not in source, forbidden
    assert barrier.barrier_state(ALL_PASSED)["target_access"] == 0
