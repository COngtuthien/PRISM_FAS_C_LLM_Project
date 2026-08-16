"""Historical adapters: C0, C1, C2, C2B, C2C (required tests 1-5, 27-28).

These milestones are finished and their provider calls are paid for. The
property under test is that the adapters *verify* them and cannot *repeat*
them — so alongside the ordinary pass checks, every test here asserts the
provider-call count is zero.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism_fas.pipeline.adapters import ProviderBinding
from prism_fas.pipeline.adapters.historical import build_adapters
from prism_fas.pipeline.adapters.registry import ADAPTED_SUBSTAGE_IDS, build_registry

from conftest_adapters import make_sandbox, request_for


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory) -> Path:
    return make_sandbox(tmp_path_factory.mktemp("historical"))


def _run(sandbox: Path, stage_id: str, profile_name: str = "smoke"):
    adapter = build_adapters()[stage_id]
    return adapter.run(request_for(sandbox, profile_name))


# --- 1. C0 -------------------------------------------------------------------

def test_c0_adapter_verification_passes(sandbox: Path) -> None:
    results = _run(sandbox, "C0")
    assert len(results) == 1
    assert results[0].ok, results[0].failed_checks
    assert results[0].substage == "C0"


def test_c0_verifies_the_version_b_snapshot_and_inherited_failures(sandbox: Path) -> None:
    checks = {check["check_id"] for check in _run(sandbox, "C0")[0].checks}
    assert "c0_version_b_snapshot_present" in checks
    assert "c0_inherited_failures_documented" in checks


# --- 2. C1 -------------------------------------------------------------------

def test_c1_adapter_reproduces_contract_identities(sandbox: Path) -> None:
    result = _run(sandbox, "C1")[0]
    assert result.ok, result.failed_checks
    identity_check = next(check for check in result.checks
                          if check["check_id"] == "c1_contract_identities_reproduce")
    assert identity_check["ok"]
    assert identity_check["detail"]["drifted"] == []


def test_c1_identity_check_fails_on_drift(sandbox: Path, tmp_path: Path) -> None:
    """Rewrite the lock's expected identity; the adapter must disagree with it."""
    broken = make_sandbox(tmp_path / "broken")
    lock = broken / "reports/c3/C3_BANK_LOCK.json"
    payload = json.loads(lock.read_text(encoding="utf-8"))
    payload["components"]["route_policy_identity"] = "0" * 64
    lock.write_text(json.dumps(payload), encoding="utf-8")

    result = build_adapters()["C1"].run(request_for(broken, "smoke"))[0]
    assert not result.ok


# --- 3, 4, 5. C2 / C2B / C2C -------------------------------------------------

def test_c2_adapter_covers_all_three_substages(sandbox: Path) -> None:
    results = _run(sandbox, "C2")
    assert [result.substage for result in results] == ["C2", "C2B", "C2C"]
    assert all(result.ok for result in results), [
        check for result in results for check in result.failed_checks]


def test_c2_pilot_stays_disposable(sandbox: Path) -> None:
    result = next(item for item in _run(sandbox, "C2") if item.substage == "C2")
    check = next(item for item in result.checks
                 if item["check_id"] == "c2_pilot_is_disposable")
    assert "never enter" in check["detail"]["rule"]


def test_c2b_negative_outcome_is_preserved(sandbox: Path) -> None:
    """BATCH_SHAPE_FAIL is the finding; an adapter that lost it would be wrong."""
    result = next(item for item in _run(sandbox, "C2") if item.substage == "C2B")
    preserved = next(item for item in result.checks
                     if item["check_id"] == "c2b_negative_outcome_preserved")
    assert preserved["ok"]
    rejection = next(item for item in result.checks
                     if item["check_id"] == "c2b_envelope_rejection_preserved")
    assert rejection["ok"]


def test_c2b_deviation_detected_if_rewritten(tmp_path: Path) -> None:
    broken = make_sandbox(tmp_path / "rewritten")
    acceptance = broken / "reports/c2b/C2B_ACCEPTANCE.json"
    payload = json.loads(acceptance.read_text(encoding="utf-8"))
    scrubbed = json.loads(json.dumps(payload).replace("BATCH_SHAPE_FAIL", "PASS"))
    acceptance.write_text(json.dumps(scrubbed), encoding="utf-8")

    results = build_adapters()["C2"].run(request_for(broken, "smoke"))
    c2b = next(item for item in results if item.substage == "C2B")
    assert not c2b.ok


def test_c2c_route_identity_is_exact(sandbox: Path) -> None:
    result = next(item for item in _run(sandbox, "C2") if item.substage == "C2C")
    route = next(item for item in result.checks
                 if item["check_id"] == "c2c_route_contract_exact")
    assert route["ok"]
    assert route["detail"]["actual"] == ["physics", "gpat"]


def test_c2c_contract_context_loads_and_self_verifies(sandbox: Path) -> None:
    result = next(item for item in _run(sandbox, "C2") if item.substage == "C2C")
    context = next(item for item in result.checks
                   if item["check_id"] == "c2c_contract_context_loads")
    assert context["ok"]


# --- no historical provider call is ever repeated ----------------------------

@pytest.mark.parametrize("stage_id", ["C0", "C1", "C2"])
def test_historical_adapters_make_zero_provider_calls(sandbox: Path, stage_id: str) -> None:
    for result in _run(sandbox, stage_id):
        assert result.provider_calls == 0
        assert result.provider_binding is ProviderBinding.NONE


@pytest.mark.parametrize("stage_id", ["C0", "C1", "C2"])
def test_historical_adapters_cannot_bind_a_provider(sandbox: Path, stage_id: str) -> None:
    adapter = build_adapters()[stage_id]
    for profile_name in ("validate", "smoke", "full"):
        assert adapter.default_binding(profile_name) is ProviderBinding.NONE


def test_archived_provider_evidence_is_verified_not_reissued(sandbox: Path) -> None:
    result = next(item for item in _run(sandbox, "C2") if item.substage == "C2")
    evidence = next(item for item in result.checks
                    if item["check_id"] == "c2_provider_evidence_intact")
    assert evidence["ok"]
    assert evidence["detail"]["provider_calls_made_by_this_adapter"] == 0


# --- status reconstruction ---------------------------------------------------

def test_a_historical_verdict_is_not_promoted_to_an_engineering_status(
        sandbox: Path) -> None:
    """These milestones predate L.3; reconstructing a status is not measuring one."""
    for stage_id in ("C0", "C1", "C2"):
        for result in _run(sandbox, stage_id):
            assert result.status_axes.engineering == "NOT_TESTED"
            assert result.status_axes.scientific == "NOT_RUN"
            assert "historical_recorded_verdict" in result.detail


# --- registry ----------------------------------------------------------------

def test_registry_covers_exactly_c0_to_c3(sandbox: Path) -> None:
    assert sorted(build_registry()) == ["C0", "C1", "C2", "C3"]


def test_registry_substages_are_the_established_lineage() -> None:
    assert ADAPTED_SUBSTAGE_IDS == ("C0", "C1", "C2", "C2B", "C2C", "C3")


def test_c4_to_c13_have_no_adapter() -> None:
    registry = build_registry()
    for index in range(4, 14):
        assert f"C{index}" not in registry
