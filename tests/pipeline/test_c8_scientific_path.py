"""The C8 scientific matrix, rehearsed row by row without a GPU.

The defect this file is the regression for: C8 had ONE workflow whose executor
called `audit_batch` and `build_audit_detector` unconditionally. The scheduler
was already correct — under a scientific context `ExecutionContext.limit`
returns the full 42 and never reads `SMOKE_ROWS` — so a `--profile full` run
would have trained all 42 rows on fixture batches through an audit model, written
42 PASS manifests, and handed them to C9 to freeze. Every check would have
passed, because the fixture execution is correct engineering.

So two things are proved here. That the scientific path is a DIFFERENT path,
which cannot reach the fixture executor at all; and that the path it is instead
does the whole atomic-row lifecycle — resolve the row's typed variant, the frozen
C7 configuration, the row's own C6 matched bank and the row's protocol splits;
train; select and calibrate on source_dev alone; evaluate cross-source as a
diagnostic; and write a manifest, calibration, history, checkpoint identity,
complexity and resources per row.

The C7 lock C8 consumes is produced by actually running C7's scientific path
first, so the producer/consumer contract between the two stages is exercised
rather than asserted against a hand-written lock.

Nothing here is scientific evidence: the trainer is stubbed at the two points a
laptop cannot reach, and every artifact lands in a tmp_path sandbox.
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

from conftest_adapters import request_for  # noqa: E402
from prism_fas.evaluation import source_selection  # noqa: E402
from prism_fas.evaluation.source_matrix import build_plan  # noqa: E402
from prism_fas.pipeline.adapters import c7, c8  # noqa: E402
from test_c7_scientific_path import (_StubTrainer, _approve, _flow,  # noqa: E402
                                     _frames, _run, scientific)  # noqa: F401

#: The four rows the rehearsal exercises, chosen to cover every axis §18 varies:
#: both tracks, a P1/P2 protocol and a P3-ready one, and the PromptHead ablation.
REHEARSED = ("C-G-RND-P1-s20260806",       # Track G, P1, single-domain selection
             "C-G-LLM-P2-s20260807",       # Track G, P2, the other single domain
             "C-R-DET-P3READY-s20260806",  # Track R, P3-ready, two-domain equal weight
             "C-R-NOPROMPT-P3READY-s20260808")  # the C-H5 PromptHead OFF ablation


@pytest.fixture
def with_c7_lock(scientific, monkeypatch):  # noqa: F811
    """A sandbox whose C7 scientific run really produced the detector config lock."""
    _approve(scientific)
    _run(scientific)
    assert (scientific / c7.SCIENTIFIC_CONFIG_LOCK_PATH).is_file()

    from prism_fas.detector import trainer as trainer_module

    # C8's row runner reads the same two things C7's trial runner did.
    monkeypatch.setattr(trainer_module, "M9Trainer", _StubTrainer)
    monkeypatch.setattr(trainer_module, "run_source_only_flow", _flow)
    monkeypatch.setattr(
        source_selection, "source_dev_frame_rows",
        lambda trainer: _frames(1.0, tuple(trainer.config.source_domains)))
    monkeypatch.setattr("prism_fas.reporting.complexity.profile_model",
                        lambda model, batch, **kwargs: {
                            "name": kwargs.get("name", ""), "total_parameters": 1,
                            "trainable_parameters": 1,
                            "complexity": {"status": "stubbed"}})

    def cross(request, *, trainer, inputs, row, domains, calibration, epoch):
        """The cross-source diagnostic, over the real evaluator on stub frames.

        Stubbed only where it opens a second `M9ValidationDataset` over the real
        package. `source_selection.evaluate` itself runs, so the payload's shape,
        its `role` and its `is_selection_signal` flag are the real ones.
        """
        frames = _frames(0.8, tuple(domains))
        return source_selection.evaluate(
            frames, protocol=row.protocol, temperature=calibration["temperature"],
            threshold=calibration["threshold"], epoch=epoch,
            decision_logit_name=trainer.decision_logit_name,
            decision_score_name=trainer.decision_score_name,
            domains=domains, role="cross_source_diagnostic")

    monkeypatch.setattr(c8, "_cross_source_evaluation", cross)
    _StubTrainer.instances = []
    return scientific


def _prepare(repo: Path, **kwargs: Any):
    request = request_for(repo, "full", **kwargs)
    inputs, result = c8.C8Adapter()._scientific_prepare(
        request, repo / "reports/full/c8")
    return request, inputs, result


def _rows(row_ids=REHEARSED):
    plan = build_plan()
    by_id = {row.row_id: row for row in plan.rows}
    return plan, [by_id[row_id] for row_id in row_ids]


# --- the input contract ------------------------------------------------------

def test_c8_verifies_the_c7_lock_with_c7s_own_verifier(with_c7_lock) -> None:
    _request, inputs, result = _prepare(with_c7_lock)

    assert result.status == "PASS", [c for c in result.checks if not c["ok"]]
    assert inputs is not None
    shared = next(item for item in result.checks
                  if item["check_id"] == "c8_uses_c7s_own_lock_verifier")
    assert shared["detail"]["verifier"].endswith("verify_detector_config_lock")
    assert shared["detail"]["valid"] is True

    binding = next(item for item in result.checks
                   if item["check_id"] == "c8_c7_lock_binds_this_c6_closure")
    assert binding["ok"] is True


def test_a_drifted_c7_lock_blocks_c8_before_any_row(with_c7_lock) -> None:
    """The 42 rows train at the configuration the lock names, so a lock that does
    not verify must stop the stage rather than the first row."""
    path = with_c7_lock / c7.SCIENTIFIC_CONFIG_LOCK_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tracks"]["R"]["winner_checkpoint_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    _request, inputs, result = _prepare(with_c7_lock)
    assert inputs is None
    assert result.status != "PASS"
    assert "does not verify" in result.summary


def test_the_precondition_gate_names_a_bad_c7_lock(with_c7_lock) -> None:
    path = with_c7_lock / c7.SCIENTIFIC_CONFIG_LOCK_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tracks"]["G"]["retained_trials"] = []
    path.write_text(json.dumps(payload), encoding="utf-8")

    request = request_for(with_c7_lock, "full")
    rows = {row["name"]: row for row in c8.C8Adapter().semantic_preconditions(request)}
    lock = rows["c7_config_lock_verified"]
    assert lock["present"] is False and lock["blocking"] is True
    assert "c7_config_lock_track_g_retains_every_trial" in lock["problems"]


# --- the atomic row lifecycle ------------------------------------------------

def test_every_rehearsed_row_runs_end_to_end(with_c7_lock) -> None:
    request, inputs, _ = _prepare(with_c7_lock)
    _plan, rows = _rows()
    runs = with_c7_lock / "runs/full/c8"

    executed = [c8._run_scientific_row(request, inputs=inputs, row=row, root=runs)
                for row in rows]

    assert [item["status"] for item in executed] == ["PASS"] * len(rows)
    for item, row in zip(executed, rows):
        directory = with_c7_lock / item["path"]
        for name in ("run_manifest.json", "calibration.json", "train_history.jsonl",
                     "model_complexity.json", "compute_resources.json"):
            assert (directory / name).is_file(), f"{row.row_id} is missing {name}"

        manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["run_identity"] == row.run_identity
        assert manifest["config_identity"] == row.config_identity
        assert manifest["seed"] == row.seed
        assert manifest["fixture_backed"] is False
        assert manifest["trainer"] == "M9Trainer"
        assert manifest["target_labels_resolved"] == 0
        assert manifest["checkpoint"]["sha256"]
        assert manifest["calibration"]["calibration_hash"]
        assert set(manifest["parent_identities"]) >= {
            "source_package", "c6_bank", "c7_detector_config", "c7_search_plan"}


def test_selection_uses_only_the_protocols_own_domains(with_c7_lock) -> None:
    """§19: P1 selects on CASIA-dev, P2 on MSU-dev, P3-ready on both."""
    request, inputs, _ = _prepare(with_c7_lock)
    _plan, rows = _rows()
    runs = with_c7_lock / "runs/full/c8"

    for row in rows:
        item = c8._run_scientific_row(request, inputs=inputs, row=row, root=runs)
        expected = list(source_selection.domains_for(row.protocol))
        assert item["selection_domains"] == expected, row.row_id
        assert item["expected_selection_domains"] == expected
        # And the trainer really was configured for those domains.
        assert tuple(_StubTrainer.instances[-1].config.source_domains) == tuple(expected)


def test_the_p3_ready_tuple_is_equal_weight_across_domains(with_c7_lock) -> None:
    request, inputs, _ = _prepare(with_c7_lock)
    _plan, rows = _rows(("C-R-DET-P3READY-s20260806",))
    item = c8._run_scientific_row(request, inputs=inputs, row=rows[0],
                                  root=with_c7_lock / "runs/full/c8")

    selection = item["metrics"]["source_dev"]
    assert selection["selection_tuple"] == list(source_selection.P3_READY_TUPLE)
    assert set(selection["per_domain"]) == {"casia_fasd", "msu_mfsd"}
    assert selection["is_selection_signal"] is True
    # Equal weight: the mean over the per-domain numbers, not over pooled rows.
    per_domain = [item["video_ACER"] for item in selection["per_domain"].values()]
    assert selection["mean_domain_video_ACER"] == pytest.approx(
        sum(per_domain) / len(per_domain))


def test_cross_source_is_a_diagnostic_and_never_a_selection_signal(with_c7_lock) -> None:
    request, inputs, _ = _prepare(with_c7_lock)
    _plan, rows = _rows(("C-G-RND-P1-s20260806", "C-R-DET-P3READY-s20260806"))
    runs = with_c7_lock / "runs/full/c8"

    p1 = c8._run_scientific_row(request, inputs=inputs, row=rows[0], root=runs)
    p3 = c8._run_scientific_row(request, inputs=inputs, row=rows[1], root=runs)

    assert p1["cross_source"]["role"] == "cross_source_diagnostic"
    assert p1["cross_source"]["is_selection_signal"] is False
    assert p1["cross_source"]["domains"] == ["msu_mfsd"]
    # The diagnostic carries the row's OWN frozen temperature and threshold.
    assert p1["cross_source"]["temperature"] == p1["calibration"]["temperature"]
    assert p1["cross_source"]["threshold"] == p1["calibration"]["threshold"]
    # A P3-ready row's test domain is the held-out target; nothing is evaluated.
    assert p3["cross_source"] == {}


def test_calibration_fits_and_thresholds_the_same_quantity(with_c7_lock) -> None:
    """§16.2, the Version-B G7 defect, checked per row and per track."""
    request, inputs, _ = _prepare(with_c7_lock)
    _plan, rows = _rows()
    runs = with_c7_lock / "runs/full/c8"

    for row in rows:
        item = c8._run_scientific_row(request, inputs=inputs, row=row, root=runs)
        calibration = item["calibration"]
        assert calibration["split"] == "source_dev"
        assert calibration["uses_target"] is False
        assert calibration["decision_logit_name"] == item["decision_logit_name"]
        assert calibration["thresholded_quantity"] == item["decision_score_name"]


def test_a_row_that_genuinely_fails_is_retained_not_constructed(with_c7_lock,
                                                                monkeypatch) -> None:
    """A real failure keeps its own manifest. Nothing constructs one."""
    from prism_fas.detector import trainer as trainer_module

    request, inputs, _ = _prepare(with_c7_lock)
    _plan, rows = _rows(("C-G-RND-P1-s20260806",))

    def explodes(trainer, *, resume=True):
        raise RuntimeError("planted row failure: the detector did not train")

    monkeypatch.setattr(trainer_module, "run_source_only_flow", explodes)
    item = c8._run_scientific_row(request, inputs=inputs, row=rows[0],
                                  root=with_c7_lock / "runs/full/c8")

    assert item["status"] == "FAIL"
    manifest = json.loads(
        (with_c7_lock / item["path"] / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "FAIL"
    assert manifest["failure"]["constructed"] is False
    assert "planted row failure" in manifest["failure"]["reason"]
    assert manifest["run_identity"] == rows[0].run_identity


def test_a_completed_row_is_reused_by_identity_not_rerun(with_c7_lock) -> None:
    """L.11 at row granularity: existence is not enough, identity is."""
    from prism_fas.pipeline.adapters.common import resume_decision

    request, inputs, _ = _prepare(with_c7_lock)
    _plan, rows = _rows(("C-G-RND-P1-s20260806",))
    runs = with_c7_lock / "runs/full/c8"
    item = c8._run_scientific_row(request, inputs=inputs, row=rows[0], root=runs)
    directory = with_c7_lock / item["path"]

    decision = resume_decision(
        request_for(with_c7_lock, "full", resume=True), rows[0].row_id,
        directory / "run_manifest.json",
        expected_identity=rows[0].run_identity, identity_key="run_identity")
    assert decision["action"] == "SKIP_VALID_COMPLETE"

    # A manifest recording a DIFFERENT identity is not a completion for this row.
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["run_identity"] = "0" * 64
    (directory / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    drifted = resume_decision(
        request_for(with_c7_lock, "full", resume=True), rows[0].row_id,
        directory / "run_manifest.json",
        expected_identity=rows[0].run_identity, identity_key="run_identity")
    assert drifted["action"] == "EXECUTE"


# --- the aggregation modes ---------------------------------------------------

def _executed(repo: Path, row_ids=REHEARSED) -> tuple[Any, Any, list[dict[str, Any]]]:
    request, inputs, _ = _prepare(repo)
    plan, rows = _rows(row_ids)
    runs = repo / "runs/full/c8"
    return request, plan, [c8._run_scientific_row(request, inputs=inputs, row=row,
                                                  root=runs) for row in rows]


def test_cross_source_diagnostics_cover_every_p1_p2_row(with_c7_lock) -> None:
    request, _plan, executed = _executed(with_c7_lock)
    result = c8.C8Adapter()._scientific_diagnostics(
        request, executed, with_c7_lock / "reports/full/c8")

    assert result.status == "PASS", [c for c in result.checks if not c["ok"]]
    payload = json.loads(
        (with_c7_lock / "reports/full/c8/C8_CROSS_SOURCE_DIAGNOSTICS.json")
        .read_text(encoding="utf-8"))
    assert {row["row_id"] for row in payload["rows"]} == {
        "C-G-RND-P1-s20260806", "C-G-LLM-P2-s20260807"}


def test_calibration_stability_groups_seeds_of_one_configuration(with_c7_lock) -> None:
    request, _plan, executed = _executed(
        with_c7_lock, ("C-G-RND-P1-s20260806", "C-G-RND-P1-s20260807",
                       "C-G-RND-P1-s20260808"))
    result = c8.C8Adapter()._scientific_calibration_stability(
        request, executed, with_c7_lock / "reports/full/c8")

    assert result.status == "PASS", [c for c in result.checks if not c["ok"]]
    payload = json.loads(
        (with_c7_lock / "reports/full/c8/C8_CALIBRATION_STABILITY.json")
        .read_text(encoding="utf-8"))
    assert len(payload["configurations"]) == 1
    group = payload["configurations"][0]
    assert group["seeds"] == [20260806, 20260807, 20260808]
    assert "stdev" in group["temperature"] and "stdev" in group["threshold"]


def test_acceptance_refuses_a_partial_matrix(with_c7_lock) -> None:
    """Four rows out of 42 is not a matrix, and acceptance must say so."""
    request, inputs, _ = _prepare(with_c7_lock)
    plan, _rows_ = _rows()
    _request, plan, executed = _executed(with_c7_lock)
    reports = with_c7_lock / "reports/full/c8"

    result = c8.C8Adapter()._scientific_acceptance(
        request, inputs, plan, executed, reports, gates=[])
    payload = json.loads((reports / "C8_ACCEPTANCE.json").read_text(encoding="utf-8"))

    assert payload["accepted"] is False
    assert payload["rows_declared"] == 42
    assert payload["rows_terminal"] == len(executed)
    assert len(payload["missing_rows"]) == 42 - len(executed)
    assert payload["hidden_rows"] == []
    terminal = next(item for item in result.checks
                    if item["check_id"] == "c8_every_declared_row_is_terminal")
    assert terminal["ok"] is False


def test_acceptance_refuses_a_hidden_row(with_c7_lock) -> None:
    request, inputs, _ = _prepare(with_c7_lock)
    _r, plan, executed = _executed(with_c7_lock)
    executed.append({**executed[0], "row_id": "UNPLANNED-ROW"})

    result = c8.C8Adapter()._scientific_acceptance(
        request, inputs, plan, executed, with_c7_lock / "reports/full/c8", gates=[])
    hidden = next(item for item in result.checks if item["check_id"] == "c8_no_hidden_row")
    assert hidden["ok"] is False
    assert hidden["detail"]["hidden"] == ["UNPLANNED-ROW"]


def test_acceptance_never_claims_detector_reliability(with_c7_lock) -> None:
    """C8 may finish; C9 stays blocked on a barrier C8 does not resolve."""
    request, inputs, _ = _prepare(with_c7_lock)
    _r, plan, executed = _executed(with_c7_lock)

    result = c8.C8Adapter()._scientific_acceptance(
        request, inputs, plan, executed, with_c7_lock / "reports/full/c8", gates=[])
    barrier = next(item for item in result.checks
                   if item["check_id"] == "c8_detector_reliability_is_not_claimed_here")
    assert barrier["ok"] is True
    assert set(barrier["detail"]["unresolved"]) == {
        "DETECTOR_BA_SEP_PROBE_PROTOCOL", "DETECTOR_BA_SEP_EVIDENCE_VECTOR",
        "DETECTOR_BA_SEP_PROBE_SEEDS"}

    payload = json.loads(
        (with_c7_lock / "reports/full/c8/C8_ACCEPTANCE.json").read_text(encoding="utf-8"))
    assert "DETECTOR_RELIABILITY_LOCK_C" in payload["next_gate"]
    assert not (with_c7_lock / "reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json").exists()


def test_the_matrix_is_the_canonical_42_rows(with_c7_lock) -> None:
    """Not hard-coded here: read off the canonical planner and checked against §18."""
    import collections

    plan = build_plan()
    counts = collections.Counter((row.experiment_id, row.protocol) for row in plan.rows)

    assert len(plan.rows) == 42
    for arm in ("RND", "DET", "LLM"):
        assert counts[(f"C-G-{arm}", "P1")] == 3
        assert counts[(f"C-G-{arm}", "P2")] == 3
        assert counts[(f"C-G-{arm}", "P3")] == 5
    for arm in ("DET", "LLM"):
        assert counts[(f"C-R-{arm}", "P3")] == 3
    assert counts[("C-R-NOPROMPT", "P3")] == 3
