"""Orchestration behaviour (v1.5 Appendix L.4, L.5, L.8) and its honesty properties.

The orchestrator's most valuable property is what it refuses to claim. Every
C0-C13 stage now has an adapter, which makes that property harder to see and
more important to hold: a clean validate or smoke run over all fourteen stages
must still NOT read as a working scientific pipeline. So the tests that matter
most assert the refusals — a validate run reports no engineering pass on any
stage, a stage whose adapter is missing is blocked rather than skipped, the full
profile blocks on absent scientific inputs rather than substituting fixtures, and
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
    # `assets` carries the frozen C3 recipe banks. C3's bank check and C5's route
    # check both read them, so a sandbox without them is corrupt rather than
    # minimal and the checks correctly refuse it.
    for relative in ("assets", "configs", "docs", "reports", "src"):
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


def test_a_stage_with_no_declared_check_is_not_applicable_rather_than_passing(
        validate_run: dict) -> None:
    """The gate follows the CHECKS, not the adapter.

    Every stage now declares at least one validate check, so nothing is
    NOT_APPLICABLE today. The property under test is the mapping itself: a stage
    with no declared check must never report PASS, because a PASS earned by
    having nothing to check is the exact confusion the gate exists to prevent.
    """
    for outcome in validate_run["result"].outcomes:
        if not outcome.stage.validate_checks:
            assert outcome.validate_gate == "NOT_APPLICABLE", outcome.stage.stage_id


def test_every_stage_has_an_adapter_and_declares_what_that_does_not_mean(
        validate_run: dict) -> None:
    """C0-C13 are all adapted, and none of them claims scientific evidence."""
    adapted = {stage.stage_id for stage in STAGES if stage.adapter_implemented}
    assert adapted == set(STAGE_IDS)
    for outcome in validate_run["result"].outcomes:
        assert outcome.status.scientific == "NOT_RUN", outcome.stage.stage_id
        assert outcome.status.engineering == "NOT_TESTED", outcome.stage.stage_id


def test_the_summary_says_it_is_not_scientific_evidence(validate_run: dict) -> None:
    summary = json.loads(
        (validate_run["sandbox"] / "reports/validate/VALIDATE_RUN.json").read_text(
            encoding="utf-8"))
    assert summary["not_scientific_evidence"] is True
    assert summary["artifact_kind"] == "ENGINEERING_READINESS_EVIDENCE"
    assert summary["scientific_eligible"] is False
    # Every stage is adapted now, so the list is empty — and it must still be
    # PRESENT and empty rather than absent, so a reader can tell "none missing"
    # from "the field was dropped".
    assert summary["stages_without_adapter"] == []


# --- what a validate run does establish -------------------------------------

def test_the_run_passes_against_the_live_repository(validate_run: dict) -> None:
    result = validate_run["result"]
    assert result.outcome == "PASS", result.blockers
    assert result.ok


def test_every_milestone_carries_real_checks(validate_run: dict) -> None:
    gates = {outcome.stage.stage_id: outcome.validate_gate
             for outcome in validate_run["result"].outcomes}
    assert [gates[stage_id] for stage_id in STAGE_IDS] == ["PASS"] * len(STAGE_IDS)
    for stage in STAGES:
        assert stage.validate_checks, f"{stage.stage_id} declares no validate check"


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
    assert index["run_count"] == len(STAGE_IDS)
    assert not any(row["scientific_eligible"] for row in index["runs"])


def test_historical_acceptance_files_are_annotated_not_edited(validate_run: dict) -> None:
    index = json.loads(
        (validate_run["sandbox"] / "state/MASTER_RUN_INDEX.json").read_text(encoding="utf-8"))
    annotations = index["historical_annotations"]
    assert annotations
    assert all(item["bytes_modified"] is False for item in annotations)


# --- profiles that cannot run yet -------------------------------------------

def test_a_stage_without_an_adapter_would_be_blocked_not_skipped(
        repo: Path, tmp_path: Path, monkeypatch) -> None:
    """The refusal survives even though nothing triggers it today.

    Every C0-C13 stage is adapted, so this constructs the condition instead of
    waiting for it: with one adapter removed from the registry, the run must
    report BLOCKED rather than quietly traversing the gap. Deleting this test
    when the last unadapted stage disappeared would have removed the guarantee
    along with its last live example.
    """
    from conftest_adapters import make_sandbox
    from prism_fas.pipeline.adapters import registry as registry_module

    sandbox = make_sandbox(tmp_path / "missing_adapter")
    real = registry_module.build_registry

    def without_c9() -> dict:
        table = real()
        table.pop("C9")
        return table

    monkeypatch.setattr(registry_module, "build_registry", without_c9)
    result = run(repo=sandbox, profile_name="smoke", first_stage="C8", last_stage="C9")

    assert result.outcome == "BLOCKED"
    assert not result.ok
    blocked = {outcome.stage.stage_id for outcome in result.outcomes
               if outcome.status.engineering == "BLOCKED"}
    assert blocked == {"C9"}
    assert any("has no adapter" in note
               for outcome in result.outcomes for note in outcome.notes)


@pytest.mark.parametrize("profile_name", ["full"])
def test_the_full_profile_blocks_on_absent_scientific_inputs(
        repo: Path, tmp_path: Path, profile_name: str) -> None:
    """Full refuses to start C4-C13 without the real inputs, and names them."""
    from conftest_adapters import make_sandbox

    sandbox = make_sandbox(tmp_path / f"range_{profile_name}")
    result = run(repo=sandbox, profile_name=profile_name, first_stage="C4",
                 last_stage="C13")

    assert result.outcome == "BLOCKED"
    blocked = {outcome.stage.stage_id for outcome in result.outcomes
               if outcome.status.engineering == "BLOCKED"}
    assert blocked == {f"C{index}" for index in range(4, 14)}
    # The refusal must name what is missing rather than fail vaguely.
    named = [check for outcome in result.outcomes
             for adapter_result in outcome.adapter_results
             for check in adapter_result.checks if not check["ok"]]
    assert named, "a blocked full run named no missing input"


def test_a_blocked_run_writes_no_per_stage_artifacts(repo: Path, tmp_path: Path) -> None:
    """A reports/full/ tree of stubs would read as science having started."""
    import shutil

    for relative in ("assets", "configs", "docs", "reports", "src"):
        if (repo / relative).exists():
            shutil.copytree(repo / relative, tmp_path / relative,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc",
                                                          "raw_responses"))
    run(repo=tmp_path, profile_name="full", first_stage="C4", last_stage="C13")

    assert (tmp_path / "reports/full/FULL_RUN.json").exists()
    assert not (tmp_path / "reports/full/c4").exists()
    summary = json.loads((tmp_path / "reports/full/FULL_RUN.json").read_text(encoding="utf-8"))
    assert summary["scientific_eligible"] is False
    assert summary["profile_permits_scientific_evidence"] is True


def test_the_full_profile_names_the_outstanding_c3_decision(
        repo: Path, tmp_path: Path) -> None:
    import shutil

    for relative in ("assets", "configs", "docs", "reports", "src"):
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
