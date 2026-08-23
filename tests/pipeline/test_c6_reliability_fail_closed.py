"""An empty reliability set is not a pass.

`_run_reliability_gates` set `state["reliability"] = {}` and reported success;
the consumer then read `all({}.values()) if {} else True`. Zero executed gates
therefore meant every gate passed, and §11.4 requires the SELECTED profile to
pass the mandatory gates — "none were run" is not a pass.

Which gates are mandatory at selection time is not resolvable from the frozen
spec (see `RELIABILITY_SEQUENCE_AUDIT`), so `C6_REQUIRED_RELIABILITY_GATES` is
None and the stage blocks. These tests hold the refusal in place, so the gap
cannot close by accident — and they will keep holding once a set is frozen,
because the assertion is "empty is not a pass", not "the set is empty".
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
from prism_fas.synthesis import c6_scientific as science  # noqa: E402


def _run(tmp_path: Path) -> tuple[Any, dict[str, Any], Path]:
    reports = tmp_path / "reports" / "full" / "c6"
    reports.mkdir(parents=True, exist_ok=True)
    request = AdapterRequest(repo=tmp_path, profile=load_profile("full", repo=REPO))
    state: dict[str, Any] = {}
    result = C6Adapter()._run_reliability_gates(request, state, reports)
    return result, state, reports


# --- the refusal ---------------------------------------------------------------

def test_zero_executed_gates_blocks_instead_of_passing(tmp_path: Path) -> None:
    result, state, _ = _run(tmp_path)

    assert result.status == "BLOCKED"
    assert result.mode == "C6_RELIABILITY_SEQUENCE_NEEDS_SCIENTIFIC_DECISION"
    assert state["halt"] is True
    assert state["reliability"] == {}


def test_the_stage_says_plainly_that_empty_is_not_a_pass(tmp_path: Path) -> None:
    result, _, reports = _run(tmp_path)
    explicit = next(item for item in result.checks
                    if item["check_id"] == "c6_reliability_evidence_is_explicit")

    assert explicit["ok"] is False
    assert explicit["detail"]["executed_count"] == 0
    payload = json.loads((reports / "C6_RELIABILITY.json").read_text(encoding="utf-8"))
    assert payload["empty_is_not_a_pass"] is True
    assert payload["required_set_frozen"] is False
    assert payload["is_scientific_lock"] is False


def test_the_unresolved_required_set_is_reported_with_its_audit(tmp_path: Path) -> None:
    result, _, _ = _run(tmp_path)
    frozen = next(item for item in result.checks
                  if item["check_id"] == "c6_required_reliability_set_is_frozen")

    assert frozen["ok"] is False
    assert frozen["detail"]["required"] is None
    assert len(frozen["detail"]["audit"]) >= 5
    assert frozen["detail"]["bank_level"] == ["synthetic_vs_real_spoof_probe"]
    assert len(frozen["detail"]["detector_level"]) >= 6


def test_the_scientific_workflow_stops_at_the_reliability_substage() -> None:
    source = inspect.getsource(C6Adapter._scientific_workflow)

    assert "self._run_reliability_gates" in source
    assert 'state.get("halt")' in source
    # ...and it runs before profile assessment, so nothing is selected without it.
    assert (source.index("_run_reliability_gates")
            < source.index("_check_profile_matched_feasibility"))


# --- the consumer no longer treats an empty dict as success --------------------

def test_the_consumer_does_not_read_empty_as_true() -> None:
    source = inspect.getsource(C6Adapter._check_profile_matched_feasibility)

    assert "bool(reliability) and all(reliability.values())" in source
    assert 'if state["reliability"] else True' not in source


def test_assess_profile_requires_an_explicit_verdict() -> None:
    """The default used to be True, which is a fail-open of its own."""
    signature = inspect.signature(science.assess_profile)

    assert signature.parameters["reliability_passed"].default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        science.assess_profile("STRICT", {}, {})          # type: ignore[call-arg]


@pytest.mark.parametrize("gates,expected", [
    ({}, False),
    ({"synthetic_vs_real_spoof_probe": True}, True),
    ({"synthetic_vs_real_spoof_probe": False}, False),
    ({"a": True, "b": False}, False),
])
def test_the_verdict_rule_is_conjunctive_and_empty_is_false(gates, expected) -> None:
    assert (bool(gates) and all(gates.values())) is expected


def test_a_profile_with_a_failed_reliability_gate_is_not_feasible() -> None:
    unreliable = science.MatchedFeasibility(
        profile="STRICT", arm_route_counts={}, route_quotas={},
        arms_meet_route_floor=True, common_quota_feasible=True,
        reliability_passed=False)

    assert unreliable.feasible is False


# --- the audit is recorded, not invented --------------------------------------

def test_the_required_set_is_unresolved_and_named_as_such() -> None:
    assert c6_module.C6_REQUIRED_RELIABILITY_GATES is None
    assert c6_module.RELIABILITY_SEQUENCE_UNRESOLVED.endswith(
        "NEEDS_SCIENTIFIC_DECISION")


def test_the_declared_tests_are_split_bank_level_versus_detector_level() -> None:
    from prism_fas.evaluation.reliability import declared_tests

    declared = {item.test_id for item in declared_tests()}
    classified = (set(c6_module.BANK_LEVEL_RELIABILITY_TESTS)
                  | set(c6_module.DETECTOR_LEVEL_RELIABILITY_TESTS))

    assert classified <= declared, "every classified test really is declared"
    assert not (set(c6_module.BANK_LEVEL_RELIABILITY_TESTS)
                & set(c6_module.DETECTOR_LEVEL_RELIABILITY_TESTS))
    # The one test that needs no detector is the synthetic-vs-real probe.
    assert c6_module.BANK_LEVEL_RELIABILITY_TESTS == (
        "synthetic_vs_real_spoof_probe",)


def test_no_reliability_test_is_actually_executed_yet() -> None:
    """The framework declares; nothing in C6 runs a probe."""
    from prism_fas.evaluation.reliability import declared_tests

    assert {item.status for item in declared_tests()} <= {"PLANNED", "BLOCKED"}
    source = inspect.getsource(C6Adapter._run_reliability_gates)
    assert "executed: dict[str, bool] = {}" in source
    for forbidden in ("balanced_accuracy", "train_probe", "fit(", "BA_sep"):
        assert forbidden not in source, forbidden


def test_the_ambiguity_is_recorded_in_project_state() -> None:
    state = (REPO / "docs" / "PROJECT_STATE.md").read_text(encoding="utf-8")

    assert "C6_RELIABILITY_SEQUENCE" in state
    assert "NEEDS_SCIENTIFIC_DECISION" in state


def test_nothing_here_touches_the_candidate_pool_or_the_target() -> None:
    source = inspect.getsource(C6Adapter._run_reliability_gates)

    for forbidden in ("candidate_dir(", "write_record(", "siw", "SiW",
                      "target_test", "label_live_spoof"):
        assert forbidden not in source, forbidden
