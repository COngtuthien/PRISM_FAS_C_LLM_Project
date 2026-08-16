"""C3 crash and resume scenarios A-F (required tests 14-17, 20-21).

Every provider interaction here is mocked. The scenarios walk the failure modes
a free-tier daily quota actually produces: a process that dies mid-run, a
transient error, a hard daily stop, and a restart that must not re-spend
anything.

The assertion that recurs is the provider call count. Correct resume behaviour
is not "the run finished" — it is "the run finished without paying twice", and
only the call count can tell those apart.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism_fas.llm.providers.mock import (ScriptedResponse, quota_exhausted_error,
                                          rate_limit_error, transport_error)
from prism_fas.pipeline.adapters.c3_live import (CompletedRequestDrift, LiveGenerationState,
                                                 RequestStatus)

from conftest_adapters import (crash_after, live_state_path, make_sandbox, run_c3,
                               schedule_for, valid_script)


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    return make_sandbox(tmp_path / "repo")


def _state(sandbox: Path) -> LiveGenerationState:
    return LiveGenerationState.load(live_state_path(sandbox))


def _statuses(sandbox: Path) -> list[str]:
    return [record.status for record in _state(sandbox).requests]


# --- Scenario A: crash after 5, resume at 6 ----------------------------------

def test_scenario_a_crash_after_five_resumes_at_six(sandbox: Path) -> None:
    crash_after(sandbox, 5)

    state = _state(sandbox)
    assert len(state.completed) == 5
    cursor = state.resume_cursor
    assert cursor["next_logical_request_index"] == 5          # zero-based: request 6
    assert cursor["next_logical_request_id"] == "c3-llm-req-06"
    assert cursor["next_slot_start"] == 160
    assert cursor["next_slot_end"] == 191

    second = run_c3(sandbox, script=valid_script(sandbox, 7, salt_base=100))
    assert second.provider_calls == 7                          # 6..12 only
    assert _state(sandbox).all_complete


def test_scenario_a_never_re_issues_the_first_five(sandbox: Path) -> None:
    crash_after(sandbox, 5)
    before = {record.logical_request_id: record.raw_response_sha256
              for record in _state(sandbox).completed}
    assert len(before) == 5

    run_c3(sandbox, script=valid_script(sandbox, 7, salt_base=100))
    after = {record.logical_request_id: record.raw_response_sha256
             for record in _state(sandbox).completed}

    for request_id, sha in before.items():
        assert after[request_id] == sha, f"{request_id} was regenerated"


def test_scenario_a_the_interrupted_request_is_resumable_not_lost(
        sandbox: Path) -> None:
    """The crash lands mid-request 6; it must come back as work still to do."""
    crash_after(sandbox, 5)
    sixth = _state(sandbox).requests[5]
    assert sixth.status == RequestStatus.IN_PROGRESS.value
    assert _state(sandbox).next_request.logical_request_id == "c3-llm-req-06"


# --- Scenario B: retryable error keeps the same logical request --------------

def test_scenario_b_retryable_error_retains_request_identity(sandbox: Path) -> None:
    schedule = schedule_for(sandbox)
    script = valid_script(sandbox, 5)
    # Request 6 hits transport failures until the transport budget is spent.
    script.extend(ScriptedResponse(error=transport_error()) for _ in range(8))

    run_c3(sandbox, script=script)
    state = _state(sandbox)
    sixth = state.requests[5]

    assert sixth.status == RequestStatus.FAILED_RETRYABLE.value
    assert sixth.next_permitted_action == "RETRY_SAME_LOGICAL_REQUEST"
    assert sixth.slot_start == 160 and sixth.slot_end == 191
    assert sixth.slot_count == schedule["objects_per_request"]
    identity_after_failure = sixth.request_identity

    run_c3(sandbox, script=valid_script(sandbox, 7, salt_base=200))
    resumed = _state(sandbox).requests[5]
    assert resumed.status == RequestStatus.COMPLETED_VALID.value
    assert resumed.request_identity == identity_after_failure
    assert resumed.slot_start == 160 and resumed.slot_end == 191


def test_scenario_b_a_rate_limit_is_retryable_not_blocking(sandbox: Path) -> None:
    script = valid_script(sandbox, 2)
    script.extend(ScriptedResponse(error=rate_limit_error(0.0)) for _ in range(8))
    run_c3(sandbox, script=script)
    assert _state(sandbox).requests[2].status == RequestStatus.FAILED_RETRYABLE.value


# --- Scenario C: blocking quota stops cleanly --------------------------------

def test_scenario_c_quota_block_preserves_completed_and_points_at_six(
        sandbox: Path) -> None:
    script = valid_script(sandbox, 5)
    script.append(ScriptedResponse(error=quota_exhausted_error()))

    result = run_c3(sandbox, script=script)
    state = _state(sandbox)

    assert len(state.completed) == 5
    assert all(record.status == RequestStatus.COMPLETED_VALID.value
               for record in state.requests[:5])
    assert state.requests[5].status == RequestStatus.FAILED_BLOCKING.value
    assert state.blocked_reason is not None
    assert state.resume_cursor["next_logical_request_index"] == 5
    assert state.requests[5].next_permitted_action == "STOP_AND_RESUME_LATER"
    assert result.status == "BLOCKED"


# --- Scenario D: request 6 completes after resume, run continues -------------

def test_scenario_d_resume_completes_six_then_continues_to_seven(
        sandbox: Path) -> None:
    script = valid_script(sandbox, 5)
    script.append(ScriptedResponse(error=quota_exhausted_error()))
    run_c3(sandbox, script=script)
    assert len(_state(sandbox).completed) == 5

    run_c3(sandbox, script=valid_script(sandbox, 7, salt_base=300))
    state = _state(sandbox)
    assert state.requests[5].status == RequestStatus.COMPLETED_VALID.value
    assert state.requests[6].status == RequestStatus.COMPLETED_VALID.value
    assert state.all_complete
    assert len(state.completed) == 12


# --- Scenario E: a second invocation after 12/12 makes zero calls ------------

def test_scenario_e_second_invocation_makes_zero_provider_calls(
        sandbox: Path) -> None:
    first = run_c3(sandbox, script=valid_script(sandbox, 12))
    assert first.provider_calls == 12
    assert _state(sandbox).all_complete

    # An empty script: any provider call at all would raise from the mock.
    second = run_c3(sandbox, script=[])
    assert second.provider_calls == 0
    assert second.detail["logical_requests_executed_this_run"] == 0
    assert _state(sandbox).provider_calls_total == 12


def test_scenario_e_a_third_invocation_still_makes_zero_calls(sandbox: Path) -> None:
    run_c3(sandbox, script=valid_script(sandbox, 12))
    run_c3(sandbox, script=[])
    third = run_c3(sandbox, script=[])
    assert third.provider_calls == 0
    assert _state(sandbox).provider_calls_total == 12


# --- Scenario F: a corrupt completed artifact fails closed -------------------

def test_scenario_f_corrupt_completed_archive_fails_closed(sandbox: Path) -> None:
    run_c3(sandbox, script=valid_script(sandbox, 12))

    archive = (live_state_path(sandbox).parent / "raw_responses" / "c3-llm-req-03.json")
    payload = json.loads(archive.read_text(encoding="utf-8"))
    payload["raw_response"] = payload["raw_response"].replace("paper-like", "tampered")
    archive.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CompletedRequestDrift, match="failing closed"):
        run_c3(sandbox, script=valid_script(sandbox, 12, salt_base=400))


def test_scenario_f_a_missing_completed_archive_fails_closed(sandbox: Path) -> None:
    run_c3(sandbox, script=valid_script(sandbox, 12))
    (live_state_path(sandbox).parent / "raw_responses" / "c3-llm-req-05.json").unlink()

    with pytest.raises(CompletedRequestDrift):
        run_c3(sandbox, script=valid_script(sandbox, 12, salt_base=500))


def test_scenario_f_does_not_regenerate_the_drifted_request(sandbox: Path) -> None:
    """The point of failing closed: the bad archive is still there afterwards."""
    run_c3(sandbox, script=valid_script(sandbox, 12))
    archive = live_state_path(sandbox).parent / "raw_responses" / "c3-llm-req-03.json"
    archive.write_text(json.dumps({"raw_response": "tampered"}), encoding="utf-8")

    with pytest.raises(CompletedRequestDrift):
        run_c3(sandbox, script=valid_script(sandbox, 12, salt_base=600))
    assert json.loads(archive.read_text(encoding="utf-8"))["raw_response"] == "tampered"


# --- 18. no extra logical request is ever created ----------------------------

def test_no_extra_logical_request_is_created_across_many_resumes(
        sandbox: Path) -> None:
    crash_after(sandbox, 3)
    crash_after(sandbox, 4, salt_base=700)
    run_c3(sandbox, script=valid_script(sandbox, 5, salt_base=800))

    state = _state(sandbox)
    assert len(state.requests) == 12
    assert state.all_complete
    assert state.provider_calls_total == 12
    assert len({record.logical_request_id for record in state.requests}) == 12


def test_total_provider_calls_never_exceed_twelve_for_twelve_requests(
        sandbox: Path) -> None:
    run_c3(sandbox, script=valid_script(sandbox, 12))
    assert _state(sandbox).provider_calls_total == 12


def test_a_retry_does_not_add_a_logical_request(sandbox: Path) -> None:
    """Attempts may exceed 12; logical requests may not."""
    script = valid_script(sandbox, 2)
    script.extend(ScriptedResponse(error=transport_error()) for _ in range(8))
    run_c3(sandbox, script=script)

    state = _state(sandbox)
    assert len(state.requests) == 12
    assert state.requests[2].attempt_count >= 1
    assert state.provider_calls_total > 2      # attempts, not logical requests


# --- starting fresh over completed work is refused ---------------------------

def test_starting_fresh_over_completed_work_is_refused(sandbox: Path) -> None:
    from prism_fas.pipeline.adapters.c3_live import LiveStateError

    crash_after(sandbox, 5)
    with pytest.raises(LiveStateError, match="would re-issue them"):
        LiveGenerationState.open(live_state_path(sandbox), arm="LLM",
                                 schedule=schedule_for(sandbox), resume=False)


def test_an_unreadable_state_file_fails_closed(sandbox: Path) -> None:
    from prism_fas.pipeline.adapters.c3_live import LiveStateError

    crash_after(sandbox, 2)
    live_state_path(sandbox).write_text("{ truncated", encoding="utf-8")
    with pytest.raises(LiveStateError, match="ambiguous state"):
        LiveGenerationState.load(live_state_path(sandbox))


# --- state is written atomically after every transition ----------------------

def test_state_is_durable_after_each_completed_request(sandbox: Path) -> None:
    crash_after(sandbox, 4)
    on_disk = json.loads(live_state_path(sandbox).read_text(encoding="utf-8"))
    assert on_disk["status_counts"]["COMPLETED_VALID"] == 4
    assert len(on_disk["history"]) >= 8          # in-progress + completed per request


def test_every_completed_request_records_its_archive_identity(sandbox: Path) -> None:
    run_c3(sandbox, script=valid_script(sandbox, 12))
    for record in _state(sandbox).requests:
        assert record.archive_identity
        assert record.raw_response_sha256 == record.archive_identity
        assert record.accepted_recipe_count == 32
