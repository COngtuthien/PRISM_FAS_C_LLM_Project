"""Pipeline state and the master run index (v1.5 Appendix L.10, L.8, L.11).

Two properties are load-bearing and both are tested against the failure that
would break them rather than only the happy path: an interrupted write must not
leave a half-file, and appending to the index must not drop a row that a later
reader needs to find a losing or failing configuration.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism_fas.pipeline.profiles import load_profile
from prism_fas.pipeline.registry import (RegistryError, RunRecord, annotate_historical,
                                         index_path, read_index, record)
from prism_fas.pipeline.state import (PipelineState, StateError, atomic_write_json,
                                      read_state, state_path, write_state)
from prism_fas.pipeline.status import StatusError


def _row(run_id: str, status: str = "PASS", **kwargs) -> RunRecord:
    defaults = dict(stage_id="C0", execution_profile="validate", scientific_eligible=False,
                    status=status, started_at_utc="2026-01-01T00:00:00Z",
                    finished_at_utc="2026-01-01T00:00:01Z")
    defaults.update(kwargs)
    return RunRecord(run_id=run_id, **defaults)


# --- atomicity --------------------------------------------------------------

def _unserializable() -> dict:
    """A payload json.dumps cannot write, whatever `default` is set to.

    A plain unknown object would be absorbed by the writer's `default=str`,
    which exists so Path and similar values serialize. A circular reference
    fails regardless, so it exercises the write path rather than the encoder.
    """
    payload: dict = {"generation": 2}
    payload["self"] = payload
    return payload


def test_a_failed_write_leaves_no_partial_file(tmp_path: Path) -> None:
    target = tmp_path / "state" / "PIPELINE_STATE.json"
    with pytest.raises(ValueError, match="Circular reference"):
        atomic_write_json(target, _unserializable())
    assert not target.exists()
    assert not list((tmp_path / "state").glob("*.tmp"))
    assert not list((tmp_path / "state").glob(".*"))


def test_a_failed_overwrite_preserves_the_previous_file(tmp_path: Path) -> None:
    target = tmp_path / "state" / "PIPELINE_STATE.json"
    atomic_write_json(target, {"generation": 1})
    with pytest.raises(ValueError, match="Circular reference"):
        atomic_write_json(target, _unserializable())
    assert json.loads(target.read_text(encoding="utf-8")) == {"generation": 1}


def test_no_temp_file_survives_a_successful_write(tmp_path: Path) -> None:
    target = tmp_path / "state" / "MASTER_RUN_INDEX.json"
    atomic_write_json(target, {"runs": []})
    assert [item.name for item in (tmp_path / "state").iterdir()] == ["MASTER_RUN_INDEX.json"]


# --- pipeline state ---------------------------------------------------------

def test_state_round_trips_and_carries_the_profile_stamp(tmp_path: Path, repo: Path) -> None:
    profile = load_profile("validate", repo=repo)
    state = PipelineState(profile="validate", phase="preflight", stage_id="C3",
                          stage_index=4, completed_stages=["C0", "C1", "C2", "C3"],
                          last_updated_utc="2026-01-01T00:00:00Z", outcome="PASS")
    write_state(tmp_path, state, profile)

    stored = read_state(tmp_path)
    assert stored is not None
    assert stored["execution_profile"] == "validate"
    assert stored["scientific_eligible"] is False
    assert stored["stage_id"] == "C3"
    assert stored["completed_stages"] == ["C0", "C1", "C2", "C3"]


def test_state_is_absent_before_the_pipeline_has_ever_run(tmp_path: Path) -> None:
    assert read_state(tmp_path) is None


def test_a_corrupt_state_file_fails_closed_rather_than_reading_as_absent(
        tmp_path: Path) -> None:
    """L.11: an ambiguous cursor must not be silently treated as 'never ran'."""
    path = state_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{ truncated", encoding="utf-8")
    with pytest.raises(StateError, match="refusing to resume from an ambiguous cursor"):
        read_state(tmp_path)


def test_state_declares_itself_a_navigation_aid(tmp_path: Path, repo: Path) -> None:
    profile = load_profile("validate", repo=repo)
    write_state(tmp_path, PipelineState(profile="validate", phase="preflight",
                                        stage_id=None, stage_index=0), profile)
    assert "navigation aid" in read_state(tmp_path)["authority"]


# --- master run index -------------------------------------------------------

def test_appending_preserves_every_earlier_row(tmp_path: Path, repo: Path) -> None:
    profile = load_profile("validate", repo=repo)
    record(tmp_path, [_row("a", "FAIL"), _row("b", "DIVERGED")], profile=profile,
           generated_at_utc="2026-01-01T00:00:00Z")
    record(tmp_path, [_row("c", "PASS")], profile=profile,
           generated_at_utc="2026-01-01T00:00:02Z")

    index = read_index(tmp_path)
    assert [item["run_id"] for item in index["runs"]] == ["a", "b", "c"]
    assert index["run_count"] == 3


def test_a_losing_row_survives_a_later_winner(tmp_path: Path, repo: Path) -> None:
    """L.8 forbids winner-only cleanup; the loser stays addressable."""
    profile = load_profile("validate", repo=repo)
    record(tmp_path, [_row("loser", "FAIL"), _row("blocked", "BLOCKED")], profile=profile,
           generated_at_utc="2026-01-01T00:00:00Z")
    record(tmp_path, [_row("winner", "PASS")], profile=profile,
           generated_at_utc="2026-01-01T00:00:02Z")

    statuses = {item["run_id"]: item["status"] for item in read_index(tmp_path)["runs"]}
    assert statuses == {"loser": "FAIL", "blocked": "BLOCKED", "winner": "PASS"}


def test_a_rerun_replaces_only_its_own_row(tmp_path: Path, repo: Path) -> None:
    profile = load_profile("validate", repo=repo)
    record(tmp_path, [_row("a", "FAIL"), _row("b", "FAIL")], profile=profile,
           generated_at_utc="2026-01-01T00:00:00Z")
    record(tmp_path, [_row("a", "PASS")], profile=profile,
           generated_at_utc="2026-01-01T00:00:02Z")

    statuses = {item["run_id"]: item["status"] for item in read_index(tmp_path)["runs"]}
    assert statuses == {"a": "PASS", "b": "FAIL"}


def test_an_unreadable_index_is_never_silently_replaced(tmp_path: Path, repo: Path) -> None:
    path = index_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{ truncated", encoding="utf-8")
    with pytest.raises(RegistryError, match="existing\n?\\s*rows cannot be preserved|"
                                            "rows cannot be preserved"):
        record(tmp_path, [_row("a")], profile=load_profile("validate", repo=repo),
               generated_at_utc="2026-01-01T00:00:00Z")


def test_a_row_outside_the_l8_outcome_vocabulary_is_refused() -> None:
    with pytest.raises(StatusError, match="run status"):
        _row("a", "SUCCEEDED")


def test_a_row_cannot_claim_eligibility_its_profile_cannot_grant() -> None:
    with pytest.raises(StatusError, match="only the full profile"):
        _row("a", execution_profile="smoke", scientific_eligible=True)


def test_historical_artifacts_are_annotated_and_never_edited() -> None:
    annotation = annotate_historical("reports/c0/C0_ACCEPTANCE.json", milestone="C0",
                                     reason="predates L.9")
    assert annotation["bytes_modified"] is False
    assert annotation["execution_profile"] == "UNSTAMPED_PRE_V15"
    assert annotation["scientific_eligible"] is False


def test_the_index_records_the_l8_preservation_rule(tmp_path: Path, repo: Path) -> None:
    record(tmp_path, [_row("a")], profile=load_profile("validate", repo=repo),
           generated_at_utc="2026-01-01T00:00:00Z")
    assert "no winner-only cleanup" in read_index(tmp_path)["preservation_rule"]
