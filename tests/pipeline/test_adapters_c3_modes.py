"""C3 adapter modes and the live-generation gate (required tests 6-13, 18-19).

The gate is the reason this file exists. Between "the adapter can generate" and
"the adapter did generate" stand four independent conditions, and each one is
tested in isolation, because a gate that only holds when all four fail is not a
gate.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism_fas.pipeline.adapters import (AdapterRequest, LiveProviderRefused,
                                         ProviderBinding, permitted_bindings)
from prism_fas.pipeline.adapters.c3 import (C3Adapter, C3Mode, C3ModeRefused, live_dir_for,
                                            resolve_binding, resolve_mode)
from prism_fas.pipeline.adapters.c3_live import build_plan
from prism_fas.pipeline.adapters.quota import (QUOTA_SNAPSHOT_SCHEMA_VERSION, UNKNOWN,
                                               build_template, validate_payload)

from conftest_adapters import make_sandbox, profile, request_for, schedule_for


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory) -> Path:
    return make_sandbox(tmp_path_factory.mktemp("c3modes"))


# --- 6. pre-live verify ------------------------------------------------------

def test_c3_pre_live_adapter_passes(sandbox: Path) -> None:
    results = C3Adapter().run(request_for(sandbox, "validate"))
    assert len(results) == 1
    assert results[0].ok, results[0].failed_checks
    assert results[0].mode == C3Mode.PRE_LIVE_VERIFY.value


def test_pre_live_verify_makes_no_provider_call(sandbox: Path) -> None:
    result = C3Adapter().run(request_for(sandbox, "validate"))[0]
    assert result.provider_calls == 0
    assert result.provider_binding is ProviderBinding.NONE


def test_pre_live_verify_reports_zero_scientific_counters(sandbox: Path) -> None:
    detail = C3Adapter().run(request_for(sandbox, "validate"))[0].detail
    assert detail["c3_scientific_logical_requests"] == 0
    assert detail["c3_scientific_candidate_slots"] == 0


def test_pre_live_verify_checks_the_ancestor_chain(sandbox: Path) -> None:
    result = C3Adapter().run(request_for(sandbox, "validate"))[0]
    chain = next(check for check in result.checks
                 if check["check_id"] == "c3_ancestor_chain_accepted")
    assert chain["ok"]
    assert {row["stage"] for row in chain["detail"]["ancestors"]} == {
        "C0", "C1", "C2", "C2B", "C2C"}


def test_c3_blocks_when_an_ancestor_is_missing(tmp_path: Path) -> None:
    broken = make_sandbox(tmp_path / "no_ancestor")
    (broken / "reports/c1/C1_ACCEPTANCE.json").unlink()
    result = C3Adapter().run(request_for(broken, "validate"))[0]
    assert not result.ok
    chain = next(check for check in result.checks
                 if check["check_id"] == "c3_ancestor_chain_accepted")
    assert "C1" in chain["detail"]["missing"]


# --- 7, 8, 9. mode availability by profile -----------------------------------

@pytest.mark.parametrize("mode", ["LIVE_GENERATE", "RESUME_LIVE_GENERATE"])
def test_live_mode_unavailable_under_validate(sandbox: Path, mode: str) -> None:
    with pytest.raises(C3ModeRefused, match="not available under the validate profile"):
        resolve_mode(request_for(sandbox, "validate", mode=mode))


def test_finalize_unavailable_under_validate(sandbox: Path) -> None:
    with pytest.raises(C3ModeRefused, match="not available under the validate profile"):
        resolve_mode(request_for(sandbox, "validate", mode="FINALIZE_BANKS"))


def test_live_provider_binding_unavailable_under_smoke(sandbox: Path) -> None:
    """Smoke may drive the live CODE PATH, never the live PROVIDER."""
    request = request_for(sandbox, "smoke", mode="LIVE_GENERATE",
                          provider_binding=ProviderBinding.LIVE,
                          authorized_live_generation=True)
    with pytest.raises(LiveProviderRefused, match="forbids a live provider entirely"):
        resolve_binding(request, C3Mode.LIVE_GENERATE)


def test_smoke_may_drive_the_live_code_path_with_a_mock(sandbox: Path) -> None:
    request = request_for(sandbox, "smoke", mode="LIVE_GENERATE")
    assert resolve_mode(request) is C3Mode.LIVE_GENERATE
    assert resolve_binding(request, C3Mode.LIVE_GENERATE) is ProviderBinding.MOCK


def test_full_requires_explicit_authorization_for_a_live_binding(sandbox: Path) -> None:
    unauthorized = request_for(sandbox, "full", mode="LIVE_GENERATE",
                               provider_binding=ProviderBinding.LIVE,
                               authorized_live_generation=False)
    with pytest.raises(LiveProviderRefused, match="not been explicitly authorized"):
        resolve_binding(unauthorized, C3Mode.LIVE_GENERATE)


def test_full_with_authorization_permits_a_live_binding(sandbox: Path) -> None:
    authorized = request_for(sandbox, "full", mode="LIVE_GENERATE",
                             provider_binding=ProviderBinding.LIVE,
                             authorized_live_generation=True)
    assert resolve_binding(authorized, C3Mode.LIVE_GENERATE) is ProviderBinding.LIVE


def test_live_is_never_the_default_even_when_permitted(sandbox: Path) -> None:
    """Spending scientific quota is asked for by name, never acquired by omission."""
    request = request_for(sandbox, "full", mode="LIVE_GENERATE",
                          authorized_live_generation=True)
    assert resolve_binding(request, C3Mode.LIVE_GENERATE) is ProviderBinding.MOCK


def test_only_c3_may_ever_bind_a_live_provider(sandbox: Path) -> None:
    full = profile("full", sandbox)
    for stage_id in ("C0", "C1", "C2", "C4", "C8", "C13"):
        allowed = permitted_bindings(full, authorized_live_generation=True,
                                     stage_id=stage_id)
        assert ProviderBinding.LIVE not in allowed
    assert ProviderBinding.LIVE in permitted_bindings(
        full, authorized_live_generation=True, stage_id="C3")


def test_an_unknown_mode_is_refused(sandbox: Path) -> None:
    with pytest.raises(C3ModeRefused, match="unknown C3 mode"):
        resolve_mode(request_for(sandbox, "smoke", mode="GENERATE_EVERYTHING"))


# --- 10, 11. credential and quota gates --------------------------------------

def test_live_binding_blocks_without_a_quota_snapshot(tmp_path: Path, monkeypatch) -> None:
    """Requirements 10 and 11 together: both gates hold before any call.

    The sandbox is built fresh and its quota snapshot removed, because the real
    repository now carries a materialized one. Relying on the file merely being
    absent would make this test silently stop testing the gate the moment the
    snapshot was created — which is exactly what happened once already.
    """
    from prism_fas.pipeline.adapters.c3 import _live_generate
    from prism_fas.pipeline.adapters.quota import RELATIVE_PATH

    sandbox = make_sandbox(tmp_path / "no_quota")
    snapshot = sandbox / RELATIVE_PATH
    assert snapshot.exists(), "the fixture repo should carry the real snapshot to remove"
    snapshot.unlink()

    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    request = AdapterRequest(
        repo=sandbox, profile=profile("full", sandbox), mode="LIVE_GENERATE",
        provider_binding=ProviderBinding.LIVE, resume=True,
        authorized_live_generation=True, options={})
    result = _live_generate(request, C3Mode.LIVE_GENERATE, ProviderBinding.LIVE)

    assert result.status == "BLOCKED"
    assert result.provider_calls == 0
    quota = next(check for check in result.checks if check["check_id"] == "c3_quota_snapshot")
    assert not quota["ok"]


def test_live_binding_blocks_without_a_credential(sandbox: Path, monkeypatch) -> None:
    from prism_fas.pipeline.adapters.c3 import _credential_gate

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    gate = _credential_gate(sandbox, required=True)
    assert not gate["ok"]
    assert gate["detail"]["credential"] == "MISSING"


def test_the_credential_gate_never_serializes_a_key(sandbox: Path, monkeypatch) -> None:
    from prism_fas.pipeline.adapters.c3 import _credential_gate

    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyTESTTESTTESTTESTTESTTESTTESTTEST")
    gate = _credential_gate(sandbox, required=True)
    assert "AIza" not in json.dumps(gate)
    assert gate["detail"]["credential"] == "PRESENT"


# --- 12, 13. the frozen plan -------------------------------------------------

def test_the_plan_is_exactly_twelve_logical_requests(sandbox: Path) -> None:
    schedule = schedule_for(sandbox)
    assert schedule["requests"] == 12
    assert schedule["objects_per_request"] == 32
    assert schedule["raw_slots"] == 384

    plan = build_plan(requests=schedule["requests"],
                      objects_per_request=schedule["objects_per_request"],
                      raw_slots=schedule["raw_slots"])
    assert len(plan) == 12


def test_slot_mapping_covers_384_slots_exactly_once(sandbox: Path) -> None:
    schedule = schedule_for(sandbox)
    plan = build_plan(requests=schedule["requests"],
                      objects_per_request=schedule["objects_per_request"],
                      raw_slots=schedule["raw_slots"])
    covered: list[int] = []
    for record in plan:
        assert record.slot_count == 32
        covered.extend(range(record.slot_start, record.slot_end + 1))
    assert covered == list(range(384))
    assert len(set(covered)) == 384


def test_a_plan_that_does_not_multiply_out_is_refused() -> None:
    """36 requests would be the damaging silent error; it must not be possible."""
    from prism_fas.pipeline.adapters.c3_live import LiveStateError

    with pytest.raises(LiveStateError, match="does not multiply out"):
        build_plan(requests=36, objects_per_request=32, raw_slots=384)
    with pytest.raises(LiveStateError, match="does not multiply out"):
        build_plan(requests=12, objects_per_request=31, raw_slots=384)


def test_pool_and_bank_sizes_are_the_frozen_ones(sandbox: Path) -> None:
    schedule = schedule_for(sandbox)
    assert schedule["minimum_unique_pool"] == 320
    assert schedule["final_bank"] == 256


# --- 19. control arms consume no provider call -------------------------------

def test_rnd_and_det_consume_zero_provider_calls(sandbox: Path) -> None:
    result = C3Adapter().run(request_for(sandbox, "validate"))[0]
    arms = next(check for check in result.checks
                if check["check_id"] == "c3_control_arms_offline")
    assert arms["ok"]
    assert arms["detail"]["provider_calls_total"] == 0
    for arm in ("RND", "DET"):
        assert arms["detail"]["arms"][arm]["provider_calls"] == 0
        assert arms["detail"]["arms"][arm]["slots"] == 384


# --- namespace isolation -----------------------------------------------------

def test_a_non_eligible_profile_never_writes_into_the_c3_scientific_namespace(
        sandbox: Path) -> None:
    for name in ("validate", "smoke"):
        directory = live_dir_for(profile(name, sandbox))
        assert not directory.as_posix().startswith("reports/c3")
    assert live_dir_for(profile("full", sandbox)).as_posix() == "reports/c3/live"


# --- quota snapshot schema (K) ----------------------------------------------

def test_the_template_is_not_an_observation() -> None:
    template = build_template(model="gemini-3.6-flash")
    assert template["materialized"] is False
    assert template["current_remaining_rpd"] == UNKNOWN
    assert template["observed_at_utc"] == ""


def test_unknown_remaining_quota_is_valid() -> None:
    payload = build_template(model="gemini-3.6-flash")
    payload.update({"project": "Color classification", "tier": "Free",
                    "rpm_limit": 5, "tpm_limit": 250000, "rpd_limit": 20,
                    "observation_window": "1 Day",
                    "usage_dashboard": "No data available",
                    "observed_by": "user_observed",
                    "observed_at_utc": "2026-08-16T00:00:00.000000Z",
                    "materialized": True})
    assert validate_payload(payload) == []


def test_an_inferred_remaining_quota_is_refused() -> None:
    """The specific fabrication this schema exists to prevent."""
    payload = build_template(model="gemini-3.6-flash")
    payload.update({"current_remaining_rpd": 20, "observed_by": "inferred",
                    "observed_at_utc": "2026-08-16T00:00:00.000000Z"})
    problems = validate_payload(payload)
    assert any("may only accompany 'user_observed'" in problem for problem in problems)


def test_a_negative_remaining_quota_is_refused() -> None:
    payload = build_template(model="gemini-3.6-flash")
    payload.update({"current_remaining_rpd": -1, "observed_by": "user_observed"})
    assert any("cannot be negative" in problem for problem in validate_payload(payload))


def test_a_quota_snapshot_is_never_a_tuning_signal() -> None:
    payload = build_template(model="gemini-3.6-flash")
    payload["is_scientific_tuning_signal"] = True
    assert any("never a tuning signal" in problem for problem in validate_payload(payload))


def test_schema_version_is_enforced() -> None:
    payload = build_template(model="gemini-3.6-flash")
    payload["schema_version"] = "something-else"
    assert any(QUOTA_SNAPSHOT_SCHEMA_VERSION in problem
               for problem in validate_payload(payload))
