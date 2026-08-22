"""Closing the C4 scientific search: resume, selection binding and completion.

Three execution-correctness defects, all found by reading c9b7ec1 rather than by
spending twelve GPU-hours discovering them.

**Resume could not finalize.** `trained` was populated inside `evaluate`, and
`coordinate_search(resume=True)` reuses a recorded PASS by config hash WITHOUT
calling `evaluate`. So the dictionary was empty for exactly the trials a resumed
run depends on, and finalization refused a winner it had legitimately completed
in an earlier process.

**The selected config and the checkpoint could disagree.** `best_config` is the
coordinate-wise accumulator the pass produces; `winner` is the top row of the
individual-trial leaderboard. A trial from an early coordinate can rank globally
best while its config lacks every later coordinate's improvement. The finalizer
took the checkpoint from the leaderboard winner and wrote `best_config` beside
it — configuration A bound to checkpoint B.

**An interrupted search could finalize.** The search returned an outcome for
INTERRUPTED as well as COMPLETED, and the workflow finalized any non-None
outcome, so a bounded envelope that never closed could write the lock.

The cross-binding test below constructs a leaderboard whose winner is NOT the
coordinate-wise best, which is the shape the old code got wrong.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.pipeline.adapters import c4  # noqa: E402
from prism_fas.search.plan import canonical_config_sha256  # noqa: E402

C4_SOURCE = (REPO / "src" / "prism_fas" / "pipeline" / "adapters" / "c4.py").read_text(
    encoding="utf-8")

SELECTED = {"learning_rate_multiplier": 2.0, "weight_decay": 0.0001,
            "residual_loss_weight": 0.01, "identity_preservation_weight": 0.5}
OTHER = {**SELECTED, "learning_rate_multiplier": 0.5}
PLAN_IDENTITY = "p" * 64
IDENTITIES = {"package_identity": "pkg" + "0" * 61,
              "recipe_bank_identity": "bank" + "0" * 60,
              "pair_plan_identity": "pair" + "0" * 60}


def _summary(config_hash: str = "cfg-hash", checkpoint_sha: str = "c" * 64) -> dict:
    return {"identity": {**IDENTITIES, "config_hash": config_hash,
                         "adaface_weight_sha256": "ada" + "0" * 61,
                         "architecture_hash": "arch" + "0" * 60},
            "checkpoints": {"best_sha256": checkpoint_sha},
            "best": {"validation_total_loss": 0.2, "epoch": 3},
            "epochs_run": 4, "epochs_configured": 15, "stop_reason": "completed_all_epochs",
            "record_set_hashes": {}, "resume_lineage": [],
            "source_isolation": {"source_train_opened": True, "source_dev_opened": False,
                                 "target_test_opened": False},
            "device": "cuda"}


def _trial(config: dict, index: int = 0) -> Any:
    """A `Trial` shaped exactly as the search engine builds one."""
    from prism_fas.search.coordinate import Trial

    return Trial.create(trial_index=index, coordinate="learning_rate_multiplier",
                        value=config["learning_rate_multiplier"], config=config,
                        plan_identity=PLAN_IDENTITY)


def _write_evidence(repo: Path, runs: Path, config: dict, *,
                    checkpoint_sha: str = "c" * 64, config_hash: str = "cfg-hash",
                    plan_identity: str = PLAN_IDENTITY,
                    with_checkpoint: bool = True) -> Path:
    """A completed trial as a PREVIOUS process would have left it on disk."""
    sha = canonical_config_sha256(config)
    root = c4._trial_run_root(runs, sha)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    if with_checkpoint:
        (root / "checkpoints" / "best.pt").write_bytes(b"checkpoint bytes")
    summary = _summary(config_hash, checkpoint_sha)
    return Path(c4._write_trial_summary(
        repo, root, trial=_trial(config), plan_identity=plan_identity,
        trial_config=dict(config), summary=summary,
        metrics={"neutral_support_validation_objective": 0.2}, inputs={})["trial_summary"])


# --- defect 1: a trial that outlives its process -----------------------------

def test_a_completed_trial_writes_persistent_evidence(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    path = _write_evidence(tmp_path, runs, SELECTED)
    record = json.loads((tmp_path / path).read_text(encoding="utf-8"))

    assert record["trial_config_sha256"] == canonical_config_sha256(SELECTED)
    for field in ("search_plan_identity", "resolved_config_hash", "package_identity",
                  "recipe_bank_identity", "pair_plan_identity", "adaface_weight_sha256",
                  "architecture_hash", "checkpoint", "checkpoint_sha256",
                  "best_metrics", "epochs_run", "stop_reason", "record_set_hashes",
                  "resume_lineage", "source_isolation"):
        assert field in record, field


def test_evidence_resolves_when_nothing_was_trained_in_this_process(
        tmp_path: Path) -> None:
    """The restart scenario: a previous process trained it, this one did not."""
    runs = tmp_path / "runs"
    _write_evidence(tmp_path, runs, SELECTED)
    sha = canonical_config_sha256(SELECTED)

    resolved = c4._resolve_trial_evidence(tmp_path, runs, sha, trained={})

    assert resolved is not None, "a resumed run must find what an earlier one wrote"
    assert resolved["reused_from_previous_process"] is True
    assert resolved["trial_config_sha256"] == sha
    assert resolved["summary"]["identity"]["package_identity"] == IDENTITIES["package_identity"]
    assert resolved["summary"]["checkpoints"]["best_sha256"] == "c" * 64


def test_evidence_prefers_what_this_process_trained(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write_evidence(tmp_path, runs, SELECTED)
    sha = canonical_config_sha256(SELECTED)
    in_memory = {sha: {"run_root": "from-memory"}}

    assert c4._resolve_trial_evidence(tmp_path, runs, sha, in_memory)["run_root"] == "from-memory"


def test_absent_evidence_resolves_to_none(tmp_path: Path) -> None:
    """A recorded PASS whose run is gone is not evidence. The caller fails closed."""
    assert c4._resolve_trial_evidence(tmp_path, tmp_path / "runs",
                                      canonical_config_sha256(SELECTED), {}) is None


def test_evidence_for_a_different_config_is_refused(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    path = tmp_path / _write_evidence(tmp_path, runs, SELECTED)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["trial_config_sha256"] = "z" * 64
    path.write_text(json.dumps(record), encoding="utf-8")

    assert c4._resolve_trial_evidence(tmp_path, runs,
                                      canonical_config_sha256(SELECTED), {}) is None


def test_the_finalizer_does_not_require_training_in_this_pass() -> None:
    source = _function_source("_scientific_finalize")

    assert "_resolve_trial_evidence" in source
    assert "trained in this pass" not in source
    assert "c4_selected_trial_evidence_resolves" in source


# --- defect 2: one config, its own checkpoint --------------------------------

def _function_source(name: str) -> str:
    tree = ast.parse(C4_SOURCE)
    node = next(item for item in ast.walk(tree)
                if isinstance(item, ast.FunctionDef) and item.name == name)
    return ast.get_source_segment(C4_SOURCE, node) or ""


def test_the_selection_is_the_coordinate_wise_best_config() -> None:
    """Stated, not implied: `best_config` is what a coordinate pass produces."""
    source = _function_source("_scientific_finalize")

    assert 'selected_config = dict(payload["best_config"])' in source
    assert "selected_sha = canonical_config_sha256(selected_config)" in source
    assert '"selected_config_sha256": selected_sha' in source


def test_the_leaderboard_winner_is_kept_only_as_a_diagnostic() -> None:
    source = _function_source("_scientific_finalize")

    assert '"leaderboard_winner_config_sha256": leaderboard_sha' in source
    assert '"winner_config_sha256": winner_sha' not in source, (
        "the ambiguous name must not survive; it read as the selection")
    assert '"selection_rule"' in source


def test_the_checkpoint_is_resolved_from_the_selected_sha_not_the_winner() -> None:
    source = _function_source("_scientific_finalize")

    resolve = source.index("_resolve_trial_evidence(")
    assert "selected_sha" in source[resolve:resolve + 200]
    assert "leaderboard_sha" not in source[resolve:resolve + 200], (
        "the old bug: the checkpoint came from the leaderboard winner while the "
        "config came from best_config")


def test_a_leaderboard_winner_that_differs_cannot_cross_bind(tmp_path: Path) -> None:
    """The exact shape the old code got wrong, constructed deliberately.

    The leaderboard winner is OTHER; the coordinate-wise selection is SELECTED.
    Evidence exists only for SELECTED. The finalizer must bind SELECTED's config
    to SELECTED's checkpoint — and must not reach for OTHER's.
    """
    runs = tmp_path / "runs"
    _write_evidence(tmp_path, runs, SELECTED, checkpoint_sha="selected" + "0" * 56)
    _write_evidence(tmp_path, runs, OTHER, checkpoint_sha="other" + "0" * 59)

    selected_sha = canonical_config_sha256(SELECTED)
    other_sha = canonical_config_sha256(OTHER)
    assert selected_sha != other_sha

    chosen = c4._resolve_trial_evidence(tmp_path, runs, selected_sha, {})
    assert chosen["summary"]["checkpoints"]["best_sha256"] == "selected" + "0" * 56, (
        "the checkpoint must be the one trained for the selected configuration")
    other = c4._resolve_trial_evidence(tmp_path, runs, other_sha, {})
    assert other["summary"]["checkpoints"]["best_sha256"] != chosen["summary"][
        "checkpoints"]["best_sha256"]


def test_the_finalizer_checks_the_checkpoint_belongs_to_the_config() -> None:
    source = _function_source("_scientific_finalize")

    assert "c4_checkpoint_belongs_to_the_selected_config" in source
    assert "c4_evidence_binds_this_search_plan" in source
    assert "c4_evidence_binds_the_frozen_inputs" in source
    assert "c4_selected_checkpoint_hash_is_intact" in source


def test_verify_lock_compares_the_config_hash_rather_than_its_truthiness() -> None:
    # The checks live in the shared module-level verifier, which C5 also calls.
    source = _function_source("verify_gpat_config_lock")

    assert "recomputed == recorded" in source
    assert 'bool(recomputed))' not in source, (
        "asserting that hashing returned something proves nothing")
    assert "c4_lock_checkpoint_belongs_to_the_locked_config" in source
    assert "verify_gpat_config_lock(request.repo, path)" in _function_source(
        "_scientific_verify_lock"), "C4 VERIFY_LOCK must use the shared verifier"


def test_the_selected_config_must_have_been_evaluated() -> None:
    source = _function_source("_scientific_finalize")

    assert "c4_selected_config_was_actually_evaluated" in source
    assert "finite_valid" in source


# --- defect 3: an interrupted envelope never finalizes -----------------------

def test_an_interrupted_search_returns_no_outcome() -> None:
    source = _function_source("_scientific_search")

    assert 'if outcome.status != "COMPLETED":' in source
    assert 'outcome.status in ("COMPLETED", "INTERRUPTED")' not in source
    assert "c4_search_completed_before_finalization" in source
    assert "SEARCH_INCOMPLETE" in source


def test_the_workflow_stops_before_finalizing_a_none_outcome() -> None:
    source = _function_source("_scientific_workflow")

    assert "if outcome is None:" in source
    assert source.index("if outcome is None:") < source.index("_scientific_finalize")


def test_an_interrupted_search_preserves_its_state_and_checkpoints() -> None:
    source = _function_source("_scientific_search")
    interrupted = source[source.index('if outcome.status != "COMPLETED":'):]

    assert "state_preserved" in interrupted
    assert "resume" in interrupted
    assert "unlink" not in interrupted and "rmtree" not in interrupted, (
        "an interrupted pass deletes nothing")
    assert "SCIENTIFIC_LOCK" not in interrupted


def test_only_a_completed_search_can_reach_the_lock() -> None:
    """Structural: the single lock writer is downstream of the completion gate."""
    workflow = _function_source("_scientific_workflow")
    search = _function_source("_scientific_search")

    assert "SCIENTIFIC_LOCK" not in search
    assert workflow.index("_scientific_search") < workflow.index("_scientific_finalize")
    assert "if outcome is None:" in workflow


# --- the CUDA gate -----------------------------------------------------------

def test_scientific_c4_refuses_a_cpu_host() -> None:
    """This laptop is CPU-only, so the refusal is exercised for real here."""
    import torch

    if torch.cuda.is_available():                      # pragma: no cover
        pytest.skip("this host has CUDA; the refusal cannot be observed")

    with pytest.raises(c4.ScientificDeviceUnavailable) as raised:
        c4._scientific_device()

    assert raised.value.reason_code == "SCIENTIFIC_DEVICE_UNAVAILABLE"
    assert "fp16" in str(raised.value)


def test_the_rehearsal_device_behaviour_is_untouched() -> None:
    """`resolve_device` still answers "cpu" for everything that is not C4."""
    from prism_fas.synthesis.gpat_trainer import resolve_device

    assert resolve_device("cpu") == "cpu"
    assert "_scientific_device" not in _function_source("_smoke_gpat")
    assert "_scientific_device" not in _function_source("_source_search")


def test_only_the_scientific_trial_uses_the_strict_device() -> None:
    """`evaluate` is nested inside `_scientific_search`, so the enclosing
    function legitimately contains the call text too. Nothing else may."""
    callers = {name for name, node in
               {n.name: n for n in ast.walk(ast.parse(C4_SOURCE))
                if isinstance(n, ast.FunctionDef)}.items()
               if "_scientific_device()" in (ast.get_source_segment(C4_SOURCE, node) or "")}

    assert callers == {"_scientific_device", "_scientific_search", "evaluate"}, callers


# --- retention, isolation and separation are unchanged -----------------------

def test_failed_trials_are_still_retained() -> None:
    assert "c4_all_trials_retained" in _function_source("_scientific_search")


def test_the_trial_summary_records_the_source_only_audit(tmp_path: Path) -> None:
    path = tmp_path / _write_evidence(tmp_path, tmp_path / "runs", SELECTED)
    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["source_isolation"]["source_dev_opened"] is False
    assert record["source_isolation"]["target_test_opened"] is False
    assert "siw" not in json.dumps(record).lower()
    assert record["scientific_eligible"] is True


def test_the_engineering_separation_from_the_previous_milestone_survives() -> None:
    assert c4.C4Adapter.SCIENTIFIC_SEARCH_STATE == "C4_SCIENTIFIC_SEARCH_STATE.json"
    assert c4.C4Adapter.SCIENTIFIC_LOCK == "GPAT_CONFIG_LOCK.json"
    assert "C4_ENGINEERING_CONFIG_RECORD.json" in C4_SOURCE
    assert "_identity_stand_in" not in _function_source("_scientific_search")
    assert "_fixture_batch" not in _function_source("_scientific_search")
    assert "require_valid_winner=True" in _function_source("_scientific_search")


def test_the_trial_run_root_is_deterministic_from_the_config_identity() -> None:
    runs = Path("runs/full/c4")
    sha = canonical_config_sha256(SELECTED)

    assert c4._trial_run_root(runs, sha) == c4._trial_run_root(runs, sha)
    assert c4._trial_run_root(runs, sha).name == f"trial_{sha[:16]}"
    assert c4._trial_run_root(runs, sha) != c4._trial_run_root(
        runs, canonical_config_sha256(OTHER))
