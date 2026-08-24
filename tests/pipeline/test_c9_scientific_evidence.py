"""Scientific C9 freezes what C8 produced, or refuses. It never invents a row.

C9's rehearsal builds a complete evidence set with `_complete_evidence` and uses
it to reach seven refusal branches — which is legitimate, and the only way to
exercise branches a complete real matrix does not have. What would not be
legitimate is that same helper producing the evidence a real
`SOURCE_MATRIX_LOCK_C` closes over: the freeze would then vouch for an experiment
nobody performed, and the lock would validate perfectly because it is
self-consistent.

So the separation is structural. `_complete_evidence` raises under a scientific
context, and the scientific path's only route to evidence is
`source_evidence.load_row_evidence`, which reads C8's run manifests off disk and
re-hashes the checkpoints they name.

C9 is expected to stay BLOCKED regardless, because DETECTOR_RELIABILITY_LOCK_C is
unresolved. That is a precondition, not a defect, and it is asserted here so a
future change that quietly satisfies it is visible.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from conftest_adapters import make_sandbox, request_for  # noqa: E402
from prism_fas.evaluation import source_evidence  # noqa: E402
from prism_fas.evaluation.source_matrix import build_plan  # noqa: E402
from prism_fas.pipeline.adapters import c9  # noqa: E402
from prism_fas.pipeline.adapters.common import FixtureInScientificContext  # noqa: E402
from prism_fas.pipeline.state import atomic_write_json  # noqa: E402


@pytest.fixture
def repo(tmp_path):
    return make_sandbox(tmp_path / "repo")


def _write_row(repo: Path, row: Any, *, status: str = "PASS",
               **overrides: Any) -> Path:
    """One C8-shaped run manifest, at the path C8's scheduler addresses."""
    directory = source_evidence.row_directory(repo / source_evidence.C8_RUNS, row)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"checkpoint::{row.run_identity}".encode())
    atomic_write_json(directory / "calibration.json", {"split": "source_dev"})

    manifest = {
        "schema_version": "c8-run-manifest-v1",
        "run_identity": row.run_identity, "row_id": row.row_id,
        "protocol": row.protocol, "method": row.experiment_id, "track": row.track,
        "arm": row.arm, "config_identity": row.config_identity, "seed": int(row.seed),
        "status": status, "fixture_backed": False, "scientific_eligible": True,
        "decision_logit_name": "fused_logit_R" if row.track == "R" else "global_logit_G",
        "decision_score_name": "p_R" if row.track == "R" else "p_G",
        "checkpoint": {"path": checkpoint.relative_to(repo).as_posix(),
                       "sha256": source_evidence._sha256_file(checkpoint)},
        "calibration": {"split": "source_dev", "calibration_hash": "c" * 64,
                        "temperature": 1.0, "threshold": 0.5},
        "metrics": {"source_dev": {"ranking_tuple": {"video_ACER": 0.1}}},
        "parent_identities": {"source_matrix": "m" * 64},
        "target_labels_resolved": 0,
    }
    manifest.update(overrides)
    atomic_write_json(directory / source_evidence.RUN_MANIFEST, manifest)
    return directory


def _write_acceptance(repo: Path, plan: Any, **overrides: Any) -> None:
    payload = {
        "schema_version": "c8-acceptance-v1", "accepted": True,
        "execution_profile": "full", "scientific_eligible": True,
        "fixture_backed": False, "matrix_identity": plan.identity,
        "rows_declared": len(plan.rows), "rows_terminal": len(plan.rows),
        "rows_passed": len(plan.rows), "rows_failed": [], "hidden_rows": [],
        "missing_rows": [], "target_access": 0,
        "c7_detector_config_sha256": "7" * 64,
        "c6_selector_identity_sha256": "6" * 64,
        "source_package_identity": "3" * 64,
    }
    payload.update(overrides)
    atomic_write_json(repo / source_evidence.C8_REPORTS / source_evidence.ACCEPTANCE,
                      payload)


def _complete_matrix(repo: Path):
    plan = build_plan()
    for row in plan.rows:
        _write_row(repo, row)
    _write_acceptance(repo, plan)
    return plan


# --- the fixture/scientific separation ---------------------------------------

def test_constructed_evidence_raises_under_a_scientific_context(repo) -> None:
    """The one line that stops a freeze over rows nobody ran."""
    plan = build_plan()
    scientific = request_for(repo, "full").context
    rehearsal = request_for(repo, "smoke").context

    assert len(c9._complete_evidence(plan, rehearsal)) == len(plan.rows)
    with pytest.raises(FixtureInScientificContext):
        c9._complete_evidence(plan, scientific)


def test_the_scientific_workflow_never_names_the_constructed_helper() -> None:
    import ast
    import inspect

    source = inspect.getsource(c9)
    tree = ast.parse(source)
    node = next(item for item in ast.walk(tree)
                if isinstance(item, ast.FunctionDef)
                and item.name == "_scientific_workflow")
    body = ast.get_source_segment(source, node) or ""

    assert "_complete_evidence" not in body
    assert "_refusals" not in body


# --- the real evidence loader ------------------------------------------------

def test_a_complete_real_matrix_loads_and_freezes(repo) -> None:
    plan = _complete_matrix(repo)
    report = source_evidence.evidence_report(repo, plan)

    assert report["available"] is True
    assert report["problems"] == []
    assert report["rows_found"] == report["rows_planned"] == 42
    assert report["acceptance_accepted"] is True

    request = request_for(repo, "full")
    state, result = c9.C9Adapter()._scientific_evidence(
        request, repo / "reports/full/c9")
    assert result.status == "PASS", [c for c in result.checks if not c["ok"]]
    assert state is not None

    lock, freeze = c9.C9Adapter()._scientific_freeze(
        request, state, repo / "reports/full/c9")
    assert freeze.status == "PASS", [c for c in freeze.checks if not c["ok"]]
    assert lock is not None
    payload = json.loads(
        (repo / "reports/full/c9" / c9.SOURCE_MATRIX_LOCK).read_text(encoding="utf-8"))
    assert payload["is_scientific_lock"] is True
    assert payload["fixture_backed"] is False
    assert payload["row_count"] == 42
    assert payload["evidence_source"].endswith("load_row_evidence")


def test_an_absent_acceptance_blocks_the_freeze(repo) -> None:
    plan = build_plan()
    for row in plan.rows:
        _write_row(repo, row)

    request = request_for(repo, "full")
    state, result = c9.C9Adapter()._scientific_evidence(
        request, repo / "reports/full/c9")
    assert state is None
    assert result.status != "PASS"
    absent = next(item for item in result.checks
                  if item["check_id"] == "c9_c8_acceptance_present_and_scientific")
    assert absent["ok"] is False


@pytest.mark.parametrize("overrides,expected", [
    ({"fixture_backed": True}, "does not declare fixture_backed=false"),
    ({"scientific_eligible": False}, "not scientifically eligible"),
])
def test_a_rehearsal_acceptance_may_never_govern_the_freeze(repo, overrides,
                                                            expected) -> None:
    plan = build_plan()
    _write_acceptance(repo, plan, **overrides)

    with pytest.raises(source_evidence.SourceEvidenceError) as caught:
        source_evidence.acceptance_report(repo)
    assert expected in str(caught.value)


@pytest.mark.parametrize("overrides,problem", [
    ({"fixture_backed": True}, "FIXTURE_BACKED"),
    ({"run_identity": "0" * 64}, "RUN_IDENTITY_MISMATCH"),
    ({"config_identity": "0" * 64}, "CONFIG_IDENTITY_MISMATCH"),
    ({"calibration": {"split": "source_train", "calibration_hash": "c" * 64}},
     "CALIBRATION_NOT_SOURCE_DEV"),
    ({"target_labels_resolved": 3}, "TARGET_LABELS_RESOLVED"),
])
def test_each_way_a_manifest_fails_to_be_evidence(repo, overrides, problem) -> None:
    plan = _complete_matrix(repo)
    _write_row(repo, plan.rows[0], **overrides)

    _evidence, problems = source_evidence.load_row_evidence(repo, plan)
    assert problem in {item["problem"] for item in problems}


def test_a_checkpoint_that_moved_is_refused(repo) -> None:
    """The frozen thing must be the thing on disk."""
    plan = _complete_matrix(repo)
    directory = source_evidence.row_directory(
        repo / source_evidence.C8_RUNS, plan.rows[0])
    (directory / "checkpoints" / "best.pt").write_bytes(b"different bytes")

    _evidence, problems = source_evidence.load_row_evidence(repo, plan)
    assert "CHECKPOINT_MOVED" in {item["problem"] for item in problems}


def test_a_missing_row_is_absent_rather_than_synthesized(repo) -> None:
    """The refusal comes from the ABSENCE, never from a placeholder standing in."""
    plan = _complete_matrix(repo)
    directory = source_evidence.row_directory(
        repo / source_evidence.C8_RUNS, plan.rows[-1])
    (directory / source_evidence.RUN_MANIFEST).unlink()

    evidence, problems = source_evidence.load_row_evidence(repo, plan)
    assert len(evidence) == len(plan.rows) - 1
    assert "MANIFEST_ABSENT" in {item["problem"] for item in problems}

    request = request_for(repo, "full")
    state, result = c9.C9Adapter()._scientific_evidence(
        request, repo / "reports/full/c9")
    assert state is None
    complete = next(item for item in result.checks
                    if item["check_id"] == "c9_every_planned_row_has_readable_evidence")
    assert complete["ok"] is False


def test_a_failed_row_is_evidence_and_refuses_the_freeze(repo) -> None:
    """A real failure is loaded as a row, and `source_lock.audit` refuses on it."""
    from prism_fas.evaluation.source_lock import audit

    plan = _complete_matrix(repo)
    _write_row(repo, plan.rows[0], status="FAIL")

    evidence, problems = source_evidence.load_row_evidence(repo, plan)
    assert len(evidence) == len(plan.rows)
    assert problems == []
    assert audit(plan, evidence)["freezable"] is False


# --- the barrier C9 stays blocked on -----------------------------------------

def test_c9_is_blocked_while_detector_reliability_is_unresolved(repo) -> None:
    from prism_fas.evaluation import detector_reliability

    request = request_for(repo, "full")
    rows = {row["name"]: row for row in c9.C9Adapter().semantic_preconditions(request)}
    barrier = rows["detector_reliability_resolved"]

    assert barrier["present"] is False
    assert barrier["blocking"] is True
    assert barrier["path"] == detector_reliability.LOCK_PATH
    assert not (repo / detector_reliability.LOCK_PATH).exists(), (
        "a DETECTOR_RELIABILITY_LOCK_C exists; its probe protocol, evidence vector "
        "and seeds are still NEEDS_SCIENTIFIC_DECISION, so nothing may have written "
        "one")


def test_the_full_precondition_gate_blocks_c9_on_this_repository(repo) -> None:
    """End to end: C9 under `--profile full` reports BLOCKED, naming its inputs."""
    request = request_for(repo, "full")
    gate = c9.C9Adapter().full_precondition_gate(request)

    assert gate is not None
    assert gate.status == "BLOCKED"
    missing = {item["name"] for item in gate.detail["missing_inputs"]}
    assert "detector_reliability_resolved" in missing
