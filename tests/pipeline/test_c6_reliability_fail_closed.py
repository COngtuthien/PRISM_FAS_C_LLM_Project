"""C6_RELIABILITY_SEQUENCE = OPTION_B_POST_SELECTION_CLOSURE_GATE.

The profile is chosen on the frozen cardinality contract alone, frozen
immediately, and the three final banks are built. Only then does the bank-level
probe run, as a CLOSURE gate: it cannot steer the selection (which is what
§3.1.1 asks) and it can still stop C6 (which is what §17 asks).

Two fail-closed properties are pinned. Reliability is not an input to selection —
`MatchedFeasibility` has no field for it and `assess_profile` has no parameter
for it, so it cannot leak in. And `NOT_YET_APPLICABLE` is never a pass: C6 closes
only on `PASSED`.

The probe itself does not execute. `BA_SEP_PROBE_PROTOCOL` is None because the
protocol is not uniquely recoverable — see `BA_SEP_PROBE_AUDIT` — so the gate
BLOCKS rather than inventing a classifier. These tests keep that refusal in
place and will keep holding once a protocol is frozen, because they assert
"unresolved does not pass", not "the protocol is missing".
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

from prism_fas.pipeline.adapters import AdapterRequest  # noqa: E402
from prism_fas.pipeline.adapters import c6 as c6_module  # noqa: E402
from prism_fas.pipeline.adapters.c6 import C6Adapter  # noqa: E402
from prism_fas.pipeline.profiles import load_profile  # noqa: E402
from prism_fas.synthesis import c6_matched_bank as selector  # noqa: E402
from prism_fas.synthesis import c6_scientific as science  # noqa: E402


def _bank() -> dict[str, Any]:
    return {"size": selector.FINAL_BANK_PER_ARM,
            "by_route": {selector.PHYSICS: 512, selector.GPAT: 512},
            "exposure": {}, "selected_set_sha256": "c" * 64, "selected": []}


def _state(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "decision": science.ProfileDecision(selected="STRICT"),
        "banks": {"banks": {arm: _bank() for arm in selector.ARMS}},
        "profile_lock_written": True,
        "profile_lock": "reports/full/c6/C6_PROFILE_SELECTION_LOCK.json"}
    base.update(overrides)
    return base


def _run(tmp_path: Path, **overrides: Any) -> tuple[Any, dict[str, Any], Path]:
    reports = tmp_path / "reports" / "full" / "c6"
    reports.mkdir(parents=True, exist_ok=True)
    request = AdapterRequest(repo=tmp_path, profile=load_profile("full", repo=REPO))
    state = _state(**overrides)
    result = C6Adapter()._run_bank_level_reliability(request, state, reports)
    return result, state, reports


# --- 1, 2. reliability cannot reach profile selection -------------------------

def test_matched_feasibility_has_no_reliability_field() -> None:
    import dataclasses

    fields = {item.name for item in dataclasses.fields(science.MatchedFeasibility)}

    assert "reliability_passed" not in fields
    assert fields >= {"arms_meet_route_floor", "common_quota_feasible"}


def test_assess_profile_cannot_receive_a_reliability_verdict() -> None:
    signature = inspect.signature(science.assess_profile)

    assert "reliability_passed" not in signature.parameters
    with pytest.raises(TypeError):
        science.assess_profile("STRICT", {}, {},          # type: ignore[call-arg]
                               reliability_passed=False)


def test_selection_chooses_the_same_profile_whatever_the_probe_would_say() -> None:
    """There is no probe input to vary — which is the guarantee."""
    def assessment(name: str, feasible: bool = True) -> science.MatchedFeasibility:
        return science.MatchedFeasibility(
            profile=name, arm_route_counts={}, route_quotas={},
            arms_meet_route_floor=feasible, common_quota_feasible=feasible)

    decision = science.select_strictest_profile(
        [assessment(name) for name in science.PROFILE_ORDER])

    assert decision.selected == "STRICT"
    for evaluation in decision.evaluations:
        assert evaluation["reliability_used_for_selection"] is False


def test_the_selection_substage_records_that_ba_sep_was_not_used() -> None:
    source = inspect.getsource(C6Adapter._select_strictest_profile)

    assert "ba_sep_not_used_for_profile_selection" in source
    assert c6_module.C6_PROFILE_SELECTION_RELIABILITY_INPUTS == ()


# --- 3, 4. the profile is frozen and the banks exist before the probe --------

def test_the_workflow_runs_the_probe_after_selection_and_bank_construction() -> None:
    source = inspect.getsource(C6Adapter._scientific_workflow)

    assert (source.index("_select_strictest_profile")
            < source.index("_build_matched_banks")
            < source.index("_run_bank_level_reliability")
            < source.index("_verify_c6_locks"))


def test_the_declared_substage_order_matches(tmp_path: Path) -> None:
    modes = c6_module.SCIENTIFIC_MODES

    assert modes.index("SELECT_STRICTEST_PROFILE") < modes.index("BUILD_MATCHED_BANKS")
    assert modes.index("BUILD_MATCHED_BANKS") < modes.index("RUN_BANK_LEVEL_RELIABILITY")
    assert modes.index("RUN_BANK_LEVEL_RELIABILITY") < modes.index("VERIFY_C6_LOCKS")


def test_the_selection_lock_is_written_before_any_probe_runs() -> None:
    source = inspect.getsource(C6Adapter._select_strictest_profile)

    assert "C6_PROFILE_SELECTION_LOCK.json" in source
    assert 'state["profile_lock_written"] = True' in source
    assert "c6_selected_profile_is_frozen_immediately" in source


def test_the_probe_checks_the_profile_was_frozen_first(tmp_path: Path) -> None:
    result, _, _ = _run(tmp_path)
    frozen = next(item for item in result.checks
                  if item["check_id"] == "c6_profile_is_frozen_before_the_probe")

    assert frozen["ok"] is True
    assert frozen["detail"]["sequence"] == "OPTION_B_POST_SELECTION_CLOSURE_GATE"


def test_an_unfrozen_profile_fails_the_probe_precondition(tmp_path: Path) -> None:
    result, _, _ = _run(tmp_path, profile_lock_written=False)
    frozen = next(item for item in result.checks
                  if item["check_id"] == "c6_profile_is_frozen_before_the_probe")

    assert frozen["ok"] is False
    assert result.status == "BLOCKED"


def test_the_probe_requires_the_three_final_banks(tmp_path: Path) -> None:
    result, _, _ = _run(tmp_path)
    banks = next(item for item in result.checks
                 if item["check_id"] == "c6_final_banks_exist_before_the_probe")

    assert banks["ok"] is True
    assert banks["detail"]["sizes"] == {arm: 1024 for arm in selector.ARMS}


# --- 12, 13. unresolved is not a pass -----------------------------------------

def test_an_unresolved_probe_protocol_blocks_rather_than_passing(tmp_path: Path) -> None:
    result, state, _ = _run(tmp_path)

    assert result.status == "BLOCKED"
    assert result.mode == "C6_BA_SEP_PROBE_PROTOCOL_NEEDS_SCIENTIFIC_DECISION"
    assert state["bank_reliability"] == c6_module.RELIABILITY_BLOCKED
    assert state["halt"] is True


def test_not_yet_applicable_is_never_a_pass(tmp_path: Path) -> None:
    _, _, reports = _run(tmp_path)
    payload = json.loads((reports / "C6_BANK_RELIABILITY.json")
                         .read_text(encoding="utf-8"))

    assert payload["not_yet_applicable_is_not_a_pass"] is True
    assert payload["overall"] != c6_module.RELIABILITY_PASSED
    assert payload["is_scientific_lock"] is False


def test_c6_closes_only_on_a_passed_bank_reliability() -> None:
    source = inspect.getsource(C6Adapter._verify_c6_locks)

    assert "c6_closes_only_on_passed_bank_reliability" in source
    assert 'state.get("bank_reliability") == RELIABILITY_PASSED' in source


@pytest.mark.parametrize("status", ["NOT_YET_APPLICABLE", "BLOCKED", "FAILED"])
def test_no_non_passed_status_closes_c6(status) -> None:
    assert status != c6_module.RELIABILITY_PASSED


# --- 6, 7, 8. failure is terminal --------------------------------------------

def test_the_failure_policy_forbids_reopening_the_profile(tmp_path: Path) -> None:
    _, _, reports = _run(tmp_path)
    payload = json.loads((reports / "C6_BANK_RELIABILITY.json")
                         .read_text(encoding="utf-8"))

    for refusal in ("stays frozen", "no looser profile", "no stricter profile",
                    "no changed candidate selection", "no discarded arm",
                    "no reseeded probe", "no widened ceiling"):
        assert refusal in payload["on_failure"], refusal


def test_the_probe_substage_never_reaches_generation_or_selection_code() -> None:
    source = inspect.getsource(C6Adapter._run_bank_level_reliability)

    for forbidden in ("render_arm", "write_payload_bytes", "build_matched_banks(",
                      "select_strictest_profile(", "derive_profile("):
        assert forbidden not in source, forbidden


def test_a_blocked_gate_does_not_advance_to_another_profile(tmp_path: Path) -> None:
    result, state, _ = _run(tmp_path)

    assert state["decision"].selected == "STRICT", "the selection is untouched"
    assert result.status == "BLOCKED"


# --- 9, 10, 11. the frozen pass rule -----------------------------------------

def test_the_ceiling_and_seed_count_are_frozen() -> None:
    assert c6_module.BA_SEP_CEILING == 0.75
    assert c6_module.BA_SEP_PROBE_SEEDS_REQUIRED == 3


def test_every_arm_must_satisfy_the_ceiling(tmp_path: Path) -> None:
    _, _, reports = _run(tmp_path)
    payload = json.loads((reports / "C6_BANK_RELIABILITY.json")
                         .read_text(encoding="utf-8"))

    assert payload["ba_sep_ceiling"] == 0.75
    assert "every arm" in payload["pass_rule"]
    assert "max over RND/DET/LLM" in payload["pass_rule"]
    assert sorted(payload["per_arm"]) == sorted(selector.ARMS)


def test_the_c6_gate_is_not_the_full_c_h4_rule(tmp_path: Path) -> None:
    """The hard gate asks only that no arm is trivially identifiable."""
    _, _, reports = _run(tmp_path)
    payload = json.loads((reports / "C6_BANK_RELIABILITY.json")
                         .read_text(encoding="utf-8"))

    assert "C-H4" in payload["distinct_from"]
    assert "validity" in payload["distinct_from"]


# --- the audit is recorded, not invented --------------------------------------

def test_the_probe_protocol_is_unresolved_and_says_why() -> None:
    assert c6_module.BA_SEP_PROBE_PROTOCOL is None
    assert c6_module.BA_SEP_PROBE_PROTOCOL_UNRESOLVED.endswith(
        "NEEDS_SCIENTIFIC_DECISION")
    audit = " ".join(c6_module.BA_SEP_PROBE_AUDIT)
    assert "DETECTOR'S OWN EVIDENCE VECTOR" in audit
    assert "p_global" in audit and "s_region" in audit
    assert "three" in audit.lower()


def test_the_sequence_and_staging_are_frozen() -> None:
    assert c6_module.C6_RELIABILITY_SEQUENCE == "OPTION_B_POST_SELECTION_CLOSURE_GATE"
    assert c6_module.DETECTOR_RELIABILITY_STAGE == (
        "C8_CLOSURE_BEFORE_C9_SOURCE_MATRIX_LOCK_C")


def test_the_declared_tests_are_split_bank_level_versus_detector_level() -> None:
    from prism_fas.evaluation.reliability import declared_tests

    declared = {item.test_id for item in declared_tests()}
    bank = set(c6_module.BANK_LEVEL_RELIABILITY_TESTS)
    detector = set(c6_module.DETECTOR_LEVEL_RELIABILITY_TESTS)

    assert bank | detector <= declared
    assert not (bank & detector)
    assert bank == {"synthetic_vs_real_spoof_probe"}
    assert detector == {"residual_scale_zero", "recipe_region_shift",
                        "artifact_map_swap", "cross_route_synthetic",
                        "benign_jpeg_corruption", "benign_resize_corruption",
                        "benign_color_corruption", "crop_padding_interpolation"}


def test_the_blocked_test_keeps_its_canonical_reason() -> None:
    """`benign_glasses_makeup_lowlight` has no legitimate population."""
    from prism_fas.evaluation.reliability import declared_tests

    blocked = next(item for item in declared_tests()
                   if item.test_id == "benign_glasses_makeup_lowlight")

    assert blocked.status == "BLOCKED"
    assert blocked.test_id not in c6_module.BANK_LEVEL_RELIABILITY_TESTS
    assert blocked.test_id not in c6_module.DETECTOR_LEVEL_RELIABILITY_TESTS


def test_no_reliability_test_is_executed_yet() -> None:
    from prism_fas.evaluation.reliability import declared_tests

    assert {item.status for item in declared_tests()} <= {"PLANNED", "BLOCKED"}
    source = inspect.getsource(C6Adapter._run_bank_level_reliability)
    for forbidden in ("balanced_accuracy(", "train_probe", "LogisticRegression"):
        assert forbidden not in source, forbidden


def test_the_decision_is_recorded_in_project_state() -> None:
    state = (REPO / "docs" / "PROJECT_STATE.md").read_text(encoding="utf-8")

    assert "OPTION_B_POST_SELECTION_CLOSURE_GATE" in state
    assert "C6_BA_SEP_PROBE_PROTOCOL" in state


# --- 15, 16, 17. firewall and immutability ------------------------------------

def test_the_probe_substage_opens_no_target_or_candidate_pool() -> None:
    source = inspect.getsource(C6Adapter._run_bank_level_reliability)

    for forbidden in ("candidate_dir(", "write_record(", "siw", "SiW",
                      "target_test", "label_live_spoof", "source_dev"):
        assert forbidden not in source, forbidden


def test_the_reliability_artifact_records_zero_target_access(tmp_path: Path) -> None:
    _, _, reports = _run(tmp_path)
    payload = json.loads((reports / "C6_BANK_RELIABILITY.json")
                         .read_text(encoding="utf-8"))

    assert payload["target_access"] == 0


def test_version_b_and_c5_are_untouched() -> None:
    import subprocess

    version_b = REPO.parent / "PRISM_FAS_B_Project"
    if not (version_b / ".git").exists():
        pytest.skip("the Version-B tree is not mounted beside this repository")

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=version_b,
                          capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=version_b,
                           capture_output=True, text=True, check=True).stdout.strip()

    assert head == "7799f7decd35db6987ce4578824e5bd8d9eab4ae"
    assert dirty == ""
    source = (REPO / "src" / "prism_fas" / "pipeline" / "adapters" / "c6.py"
              ).read_text(encoding="utf-8")
    assert "render_arm" not in source and "write_payload_bytes" not in source
