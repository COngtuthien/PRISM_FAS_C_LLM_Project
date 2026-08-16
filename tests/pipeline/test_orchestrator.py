"""Orchestration behaviour (v1.5 Appendix L.4, L.5, L.8) and its honesty properties.

The orchestrator's most valuable property right now is what it refuses to claim.
Fourteen stages exist as declarations and none has an adapter, so the tests that
matter most assert that a clean run does NOT read as a working pipeline: no
stage reports an engineering pass, no unimplemented stage reports success, and
no blocked run leaves artifacts that look like scientific evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism_fas.pipeline.orchestrator import VALIDATE_GATE, OrchestratorError, run
from prism_fas.pipeline.stages import STAGE_IDS, STAGES, StageError, stage_slice
from prism_fas.pipeline.status import scientifically_complete


@pytest.fixture(scope="module")
def validate_run(repo: Path, tmp_path_factory) -> dict:
    """One real validate run against the live repository, output redirected.

    The checks read the real repository — that is the point of them — but the
    artifacts land in a temp tree so the suite never overwrites the committed
    evidence produced by an actual `train.py` invocation.
    """
    import shutil

    sandbox = tmp_path_factory.mktemp("repo")
    for relative in ("configs", "docs", "reports", "src"):
        source = repo / relative
        if source.exists():
            shutil.copytree(source, sandbox / relative,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc",
                                                          "raw_responses"))
    result = run(repo=sandbox, profile_name="validate")
    return {"result": result, "sandbox": sandbox}


# --- what a validate run must NOT claim -------------------------------------

def test_no_stage_reports_an_engineering_pass(validate_run: dict) -> None:
    """L.3 has no value meaning 'validate passed'; NOT_TESTED is the truth."""
    for outcome in validate_run["result"].outcomes:
        assert outcome.status.engineering == "NOT_TESTED"


def test_no_stage_reports_scientific_completion(validate_run: dict) -> None:
    result = validate_run["result"]
    for outcome in result.outcomes:
        assert outcome.status.scientific == "NOT_RUN"
        assert not scientifically_complete(outcome.status, profile=result.profile.name)


def test_stages_without_an_adapter_are_not_applicable_rather_than_passing(
        validate_run: dict) -> None:
    gates = {outcome.stage.stage_id: outcome.validate_gate
             for outcome in validate_run["result"].outcomes}
    for stage_id in ("C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13"):
        assert gates[stage_id] == "NOT_APPLICABLE"


def test_every_stage_still_declares_no_adapter(validate_run: dict) -> None:
    """If this fails, an adapter landed and these expectations need revisiting."""
    assert all(not stage.adapter_implemented for stage in STAGES)


def test_the_summary_says_it_is_not_scientific_evidence(validate_run: dict) -> None:
    summary = json.loads(
        (validate_run["sandbox"] / "reports/validate/VALIDATE_RUN.json").read_text(
            encoding="utf-8"))
    assert summary["not_scientific_evidence"] is True
    assert summary["artifact_kind"] == "ENGINEERING_READINESS_EVIDENCE"
    assert summary["scientific_eligible"] is False
    assert len(summary["stages_without_adapter"]) == 14


# --- what a validate run does establish -------------------------------------

def test_the_run_passes_against_the_live_repository(validate_run: dict) -> None:
    result = validate_run["result"]
    assert result.outcome == "PASS", result.blockers
    assert result.ok


def test_the_four_prepared_milestones_carry_real_checks(validate_run: dict) -> None:
    gates = {outcome.stage.stage_id: outcome.validate_gate
             for outcome in validate_run["result"].outcomes}
    assert [gates[stage_id] for stage_id in ("C0", "C1", "C2", "C3")] == ["PASS"] * 4


def test_every_gate_value_is_from_the_declared_vocabulary(validate_run: dict) -> None:
    for outcome in validate_run["result"].outcomes:
        assert outcome.validate_gate in VALIDATE_GATE


def test_all_fourteen_stages_are_traversed(validate_run: dict) -> None:
    assert [outcome.stage.stage_id for outcome in validate_run["result"].outcomes] \
        == list(STAGE_IDS)


def test_each_stage_writes_its_own_durable_artifact(validate_run: dict) -> None:
    """L.8: a milestone emits its artifact before the orchestrator advances."""
    sandbox = validate_run["sandbox"]
    for stage_id in STAGE_IDS:
        path = sandbox / "reports/validate" / stage_id.lower() / f"{stage_id}_VALIDATE.json"
        assert path.exists(), f"{stage_id} wrote no artifact"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["execution_profile"] == "validate"
        assert payload["scientific_eligible"] is False


def test_state_and_index_are_written(validate_run: dict) -> None:
    sandbox = validate_run["sandbox"]
    state = json.loads((sandbox / "state/PIPELINE_STATE.json").read_text(encoding="utf-8"))
    index = json.loads((sandbox / "state/MASTER_RUN_INDEX.json").read_text(encoding="utf-8"))
    assert state["execution_profile"] == "validate"
    assert state["outcome"] == "PASS"
    assert index["run_count"] == 14
    assert not any(row["scientific_eligible"] for row in index["runs"])


def test_historical_acceptance_files_are_annotated_not_edited(validate_run: dict) -> None:
    index = json.loads(
        (validate_run["sandbox"] / "state/MASTER_RUN_INDEX.json").read_text(encoding="utf-8"))
    annotations = index["historical_annotations"]
    assert annotations
    assert all(item["bytes_modified"] is False for item in annotations)


# --- profiles that cannot run yet -------------------------------------------

@pytest.mark.parametrize("profile_name", ["smoke", "full"])
def test_a_profile_needing_adapters_is_blocked_not_passed(
        repo: Path, tmp_path: Path, profile_name: str) -> None:
    import shutil

    for relative in ("configs", "docs", "reports", "src"):
        if (repo / relative).exists():
            shutil.copytree(repo / relative, tmp_path / relative,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc",
                                                          "raw_responses"))
    result = run(repo=tmp_path, profile_name=profile_name)

    assert result.outcome == "BLOCKED"
    assert not result.ok
    assert any("none exists yet" in blocker for blocker in result.blockers)
    assert all(outcome.status.engineering == "BLOCKED" for outcome in result.outcomes)


def test_a_blocked_run_writes_no_per_stage_artifacts(repo: Path, tmp_path: Path) -> None:
    """A reports/full/ tree of stubs would read as science having started."""
    import shutil

    for relative in ("configs", "docs", "reports", "src"):
        if (repo / relative).exists():
            shutil.copytree(repo / relative, tmp_path / relative,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc",
                                                          "raw_responses"))
    run(repo=tmp_path, profile_name="full")

    assert (tmp_path / "reports/full/FULL_RUN.json").exists()
    assert not (tmp_path / "reports/full/c0").exists()
    summary = json.loads((tmp_path / "reports/full/FULL_RUN.json").read_text(encoding="utf-8"))
    assert summary["scientific_eligible"] is False
    assert summary["profile_permits_scientific_evidence"] is True


def test_the_full_profile_names_the_outstanding_c3_decision(
        repo: Path, tmp_path: Path) -> None:
    import shutil

    for relative in ("configs", "docs", "reports", "src"):
        if (repo / relative).exists():
            shutil.copytree(repo / relative, tmp_path / relative,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc",
                                                          "raw_responses"))
    result = run(repo=tmp_path, profile_name="full")
    assert any("NEEDS_SCIENTIFIC_DECISION" in blocker for blocker in result.blockers)


# --- scope flags ------------------------------------------------------------

def test_from_and_to_select_a_contiguous_range() -> None:
    assert [stage.stage_id for stage in stage_slice(first="C2", last="C5")] \
        == ["C2", "C3", "C4", "C5"]


def test_an_inverted_range_is_refused() -> None:
    with pytest.raises(StageError, match="is after"):
        stage_slice(first="C8", last="C2")


def test_an_unknown_stage_is_refused() -> None:
    with pytest.raises(StageError, match="unknown stage"):
        stage_slice(first="C42")


def test_a_phase_with_no_stage_in_range_is_refused(repo: Path) -> None:
    with pytest.raises(OrchestratorError, match="no stage in the requested range"):
        run(repo=repo, profile_name="validate", first_stage="C0", last_stage="C1",
            phase="target-eval")
