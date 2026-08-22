"""C4's two workflows, and the wall between them.

The defect: C4 had ONE workflow, built as an engineering rehearsal, and
`--profile full` ran it. It evaluated each search trial with one optimizer step
on a `_fixture_batch`, scored it with `_identity_stand_in` instead of the frozen
AdaFace, called `coordinate_search(require_valid_winner=False)`, and finished by
writing `C4_ENGINEERING_CONFIG_RECORD.json` while asserting the scientific
`GPAT_CONFIG_LOCK.json` did NOT exist. Every check passed — the engineering path
is correct engineering — so a real RTX 5090 run reported

    C4 PASS        with        sci=NOT_RUN

and C5 blocked on a lock nothing had written. The engineering path passed; the
scientific C4 never executed.

Two things had to be true and neither was: a scientific context must take a
different branch, and the scientific status axis must be able to say something
other than NOT_RUN. It was hard-coded in two places.

Nothing here runs GPAT training. The trainer wiring is asserted structurally —
which arguments reach `GPATTrainer`, which evaluator the search engine is given,
which flag `coordinate_search` receives — because a laptop cannot execute this
stage and pretending otherwise is how the last six defects were shipped.
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
from prism_fas.pipeline.adapters.common import (FixtureInScientificContext,  # noqa: E402
                                                assert_fixture_permitted)
from prism_fas.pipeline.execution import ExecutionContext  # noqa: E402
from prism_fas.pipeline.status import StatusError  # noqa: E402
from prism_fas.search.lr_decision import COMMON_MULTIPLIER, load_decision  # noqa: E402

C4_SOURCE = (REPO / "src" / "prism_fas" / "pipeline" / "adapters" / "c4.py").read_text(
    encoding="utf-8")


def _context(scientific: bool) -> ExecutionContext:
    from dataclasses import dataclass

    @dataclass
    class _Profile:
        name: str = "full" if scientific else "rehearsal"
        scientific_eligible: bool = scientific
        reports_namespace: str = "full" if scientific else "rehearsal"
        runs_namespace: str = "full" if scientific else "rehearsal"
        # Only the rehearsal branch builds a budget; a scientific run truncates
        # nothing, which `ExecutionContext` enforces by leaving `budget` None.
        engineering_budget: Any = None

    return ExecutionContext.for_profile(_Profile())


# --- 1-2. the two workflows are different code -------------------------------

def test_a_scientific_context_takes_the_scientific_workflow(
        monkeypatch: pytest.MonkeyPatch) -> None:
    taken: list[str] = []
    monkeypatch.setattr(c4.C4Adapter, "_scientific_workflow",
                        lambda self, request, context: taken.append("scientific") or [])
    monkeypatch.setattr(c4.C4Adapter, "_engineering_workflow",
                        lambda self, request, context: taken.append("engineering") or [])

    c4.C4Adapter().workflow(object(), _context(True))
    assert taken == ["scientific"]

    taken.clear()
    c4.C4Adapter().workflow(object(), _context(False))
    assert taken == ["engineering"]


def test_the_rehearsal_workflow_still_runs_every_engineering_mode() -> None:
    """The engineering path is preserved, not replaced."""
    tree = ast.parse(C4_SOURCE)
    engineering = next(node for node in ast.walk(tree)
                       if isinstance(node, ast.FunctionDef)
                       and node.name == "_engineering_workflow")
    called = {node.func.attr for node in ast.walk(engineering)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}

    assert {"_prepare_support", "_validate_support", "_smoke_gpat",
            "_source_search", "_finalize", "_verify_lock"} <= called


def test_the_scientific_workflow_shares_no_engineering_mode() -> None:
    tree = ast.parse(C4_SOURCE)
    scientific = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef)
                      and node.name == "_scientific_workflow")
    called = {node.func.attr for node in ast.walk(scientific)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}

    forbidden = {"_prepare_support", "_validate_support", "_smoke_gpat",
                 "_source_search", "_finalize", "_verify_lock"}
    assert not (called & forbidden), sorted(called & forbidden)
    assert {"_scientific_prepare", "_scientific_plan", "_scientific_search",
            "_scientific_finalize", "_scientific_verify_lock"} <= called


# --- 3-4. no fixture, no stand-in, in the scientific branch ------------------

def _scientific_functions() -> dict[str, ast.FunctionDef]:
    tree = ast.parse(C4_SOURCE)
    return {node.name: node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name.startswith("_scientific")}


@pytest.mark.parametrize("forbidden", ["_fixture_batch", "_identity_stand_in",
                                       "_support_conditioning"])
def test_no_scientific_function_calls_a_fixture_producer(forbidden: str) -> None:
    for name, node in _scientific_functions().items():
        called = {item.func.id for item in ast.walk(node)
                  if isinstance(item, ast.Call) and isinstance(item.func, ast.Name)}
        assert forbidden not in called, f"{name} calls {forbidden}"


def test_the_fixture_guard_refuses_a_scientific_context() -> None:
    with pytest.raises(FixtureInScientificContext) as raised:
        assert_fixture_permitted(_context(True), "a fixture batch")

    assert raised.value.reason_code == "FIXTURE_IN_SCIENTIFIC_CONTEXT"
    # ...and permits the rehearsal it exists for.
    assert assert_fixture_permitted(_context(False), "a fixture batch") is None


def test_the_engineering_prepare_is_guarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Second lock on the door: even if `workflow` were re-edited to route a
    scientific run here, the fixture producer itself refuses."""
    request = type("R", (), {"repo": REPO, "context": _context(True),
                             "profile": None, "resume": False})()

    with pytest.raises(FixtureInScientificContext):
        c4.C4Adapter()._prepare_support(request, REPO / "reports", None)


# --- 5-7. the approved LR decision, and the 2:1:2 ratio ----------------------

def test_the_scientific_plan_binds_the_approved_c4_lr_decision() -> None:
    tree = ast.parse(C4_SOURCE)
    plan_fn = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == "_scientific_plan")
    source = ast.get_source_segment(C4_SOURCE, plan_fn) or ""

    assert "load_decision" in source
    assert 'for_component("C4")' in source
    assert "lr_decision=decision" in source, (
        "gpat_search_plan must receive the decision; without it the learning-rate "
        "coordinate stays AMBIGUOUS and contributes no trials")


def test_the_engineering_plan_does_not_bind_it() -> None:
    """The rehearsal keeps the honest pre-decision shape; that is not a defect
    there, and it is why the two plans have different identities."""
    tree = ast.parse(C4_SOURCE)
    fn = next(node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef) and node.name == "_source_search")
    assert "lr_decision" not in (ast.get_source_segment(C4_SOURCE, fn) or "")


def test_the_lr_multiplier_holds_the_two_one_two_ratio() -> None:
    decision = load_decision(REPO).for_component("C4")

    assert decision.interpretation == COMMON_MULTIPLIER
    assert decision.candidates == (0.5, 1.0, 2.0)
    for multiplier in decision.candidates:
        groups = decision.lr_for_groups(multiplier)
        assert decision.ratio_preserved(multiplier)
        assert groups["encoder_lr"] / groups["recipe_lr"] == pytest.approx(2.0)
        assert groups["generator_lr"] / groups["recipe_lr"] == pytest.approx(2.0)
    # m = 1.0 reproduces the inherited anchor exactly.
    assert decision.lr_for_groups(1.0) == {"encoder_lr": 2.0e-4, "recipe_lr": 1.0e-4,
                                           "generator_lr": 2.0e-4}


def test_the_trial_config_never_writes_an_independent_per_group_rate() -> None:
    from prism_fas.pipeline.adapters.c4 import _load_config, _scientific_trial_config

    decision = load_decision(REPO).for_component("C4")
    config = _load_config(REPO)
    trial = type("T", (), {"config": {"learning_rate_multiplier": 2.0,
                                      "weight_decay": 0.5}})()

    resolved = _scientific_trial_config(config, trial, decision)

    assert resolved["optimizer"]["encoder_lr"] == pytest.approx(4.0e-4)
    assert resolved["optimizer"]["recipe_lr"] == pytest.approx(2.0e-4)
    assert resolved["optimizer"]["generator_lr"] == pytest.approx(4.0e-4)
    assert resolved["optimizer"]["weight_decay"] == 0.5
    # Everything the envelope does not search is the frozen value.
    for key in ("batch_size", "epochs", "seed"):
        assert resolved[key] == config[key]
    assert resolved["precision"] == config["precision"]
    assert resolved["early_stopping"] == config["early_stopping"]
    assert config["optimizer"]["encoder_lr"] == 2.0e-4, "the frozen config is not mutated"


def test_no_anchor_is_invented_for_the_absent_geometry_coordinate() -> None:
    """`gpat_m8.yaml` has no `loss.geometry`; §15.2.3 skips an absent scalar.
    Mapping it onto the nearest-looking key would invent an inherited anchor."""
    from prism_fas.pipeline.adapters.c4 import _TRIAL_CONFIG_PATHS

    assert "geometry_preservation_weight" not in _TRIAL_CONFIG_PATHS
    assert set(_TRIAL_CONFIG_PATHS) == {"weight_decay", "residual_loss_weight",
                                        "identity_preservation_weight"}


# --- 8-9. the search engine, and state separation ----------------------------

def test_the_scientific_search_requires_a_valid_winner() -> None:
    tree = ast.parse(C4_SOURCE)
    fn = next(node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef) and node.name == "_scientific_search")
    source = ast.get_source_segment(C4_SOURCE, fn) or ""

    assert "require_valid_winner=True" in source
    assert "require_valid_winner=False" not in source
    assert "EnvelopeExhausted" in source
    assert "NEEDS_SCIENTIFIC_DECISION" in source


def test_the_scientific_search_uses_the_canonical_trainer() -> None:
    tree = ast.parse(C4_SOURCE)
    fn = next(node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef) and node.name == "_scientific_search")
    source = ast.get_source_segment(C4_SOURCE, fn) or ""

    assert "from prism_fas.synthesis.gpat_trainer import GPATTrainer" in source
    assert "GPATTrainer(" in source
    assert "trainer.fit(" in source
    assert "package_root=" in source and "bank_root=" in source and "pairs_root=" in source


def test_the_trainer_owns_both_pair_manifests() -> None:
    """C4 hands the trainer a pairs ROOT; the trainer loads train and validation
    from it, so the search cannot select on a set it chose itself."""
    trainer_source = (REPO / "src" / "prism_fas" / "synthesis" / "gpat_trainer.py"
                      ).read_text(encoding="utf-8")

    assert 'load_pairs(self.pairs_root, "train")' in trainer_source
    assert 'load_pairs(self.pairs_root, "validation")' in trainer_source


def test_engineering_search_state_cannot_resume_a_scientific_search() -> None:
    """Two mechanisms, both required to hold."""
    assert c4.C4Adapter.SCIENTIFIC_SEARCH_STATE == "C4_SCIENTIFIC_SEARCH_STATE.json"
    assert c4.C4Adapter.SCIENTIFIC_SEARCH_STATE != "C4_SEARCH_STATE.json"

    # ...and the plans differ, so `coordinate_search.load_state` refuses anyway.
    from prism_fas.pipeline.adapters.c4 import _load_config
    from prism_fas.search.plan import gpat_search_plan

    config = _load_config(REPO)
    scientific, _ = gpat_search_plan(config,
                                     lr_decision=load_decision(REPO).for_component("C4"))
    engineering, _ = gpat_search_plan(config)
    assert scientific.identity != engineering.identity
    assert scientific.total_trials > engineering.total_trials, (
        "the decision adds the learning-rate coordinate the rehearsal cannot search")


def test_a_state_from_another_plan_is_refused(tmp_path: Path) -> None:
    from prism_fas.pipeline.adapters.c4 import _load_config
    from prism_fas.search.coordinate import SearchError, load_state
    from prism_fas.search.plan import gpat_search_plan

    config = _load_config(REPO)
    scientific, _ = gpat_search_plan(config,
                                     lr_decision=load_decision(REPO).for_component("C4"))
    engineering, _ = gpat_search_plan(config)
    state = tmp_path / "C4_SEARCH_STATE.json"
    state.write_text(json.dumps({"search_plan_identity": engineering.identity}),
                     encoding="utf-8")

    with pytest.raises(SearchError):
        load_state(state, scientific)


# --- 10-12. the lock, and what may write it ----------------------------------

def test_only_the_scientific_path_writes_the_gpat_config_lock() -> None:
    tree = ast.parse(C4_SOURCE)
    writers: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        source = ast.get_source_segment(C4_SOURCE, node) or ""
        if "SCIENTIFIC_LOCK" in source and "write_artifact" in source:
            writers.append(node.name)

    assert writers == ["_scientific_finalize"], writers


def test_the_engineering_record_still_declares_it_is_not_the_lock() -> None:
    tree = ast.parse(C4_SOURCE)
    fn = next(node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef) and node.name == "_finalize")
    source = ast.get_source_segment(C4_SOURCE, fn) or ""

    assert "C4_ENGINEERING_CONFIG_RECORD.json" in source
    assert '"is_scientific_lock": False' in source
    assert "GPAT_CONFIG_LOCK.json" in source, "it asserts the scientific lock is absent"


def test_the_scientific_lock_binds_every_frozen_identity() -> None:
    tree = ast.parse(C4_SOURCE)
    fn = next(node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef) and node.name == "_scientific_finalize")
    source = ast.get_source_segment(C4_SOURCE, fn) or ""

    for field in ("search_plan_identity", "lr_decision_identity", "winner_config_sha256",
                  "package_identity", "recipe_bank_identity", "pair_plan_identity",
                  "adaface_weight_sha256", "architecture_hash", "config_hash",
                  "winning_checkpoint", "winning_checkpoint_sha256",
                  "source_isolation", "no_target_capability_proof",
                  "attempted_config_ids", "resume_lineage"):
        assert f'"{field}"' in source, field
    assert '"is_scientific_lock": True' in source


def test_the_lock_is_not_written_without_verified_trial_evidence() -> None:
    """The requirement is "valid scientific trial evidence exists and matches
    this frozen plan", not "was trained in this process" — a resumed run must be
    able to finalize a trial an earlier process completed. Every gate still
    precedes the write."""
    tree = ast.parse(C4_SOURCE)
    fn = next(node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef) and node.name == "_scientific_finalize")
    source = ast.get_source_segment(C4_SOURCE, fn) or ""

    for gate in ("c4_selected_config_was_actually_evaluated",
                 "c4_selected_trial_evidence_resolves",
                 "c4_selected_checkpoint_present",
                 "c4_selected_checkpoint_hash_is_intact",
                 "c4_checkpoint_belongs_to_the_selected_config",
                 "c4_evidence_binds_this_search_plan",
                 "c4_evidence_binds_the_frozen_inputs"):
        assert gate in source, gate
        assert source.index(gate) < source.index("write_artifact"), gate


def test_verify_lock_checks_the_checkpoint_hash_and_the_input_identities() -> None:
    tree = ast.parse(C4_SOURCE)
    fn = next(node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef)
              and node.name == "_scientific_verify_lock")
    source = ast.get_source_segment(C4_SOURCE, fn) or ""

    assert "c4_lock_checkpoint_hash_matches" in source
    assert "sha256_file" in source
    assert "c4_lock_inputs_still_agree" in source
    assert "verify_support_inputs" in source


# --- 13-14. the status axis follows the evidence -----------------------------

def test_scientific_evidence_is_refused_on_a_non_eligible_profile() -> None:
    from prism_fas.pipeline.adapters.common import EngineeringAdapter

    class _Adapter(EngineeringAdapter):
        stage_id = "C4"
        substages = ("C4",)
        title = "t"
        modes = ("VERIFY_LOCK",)

        def required_inputs(self):
            return ()

        def workflow(self, request, context):
            return []

    request = type("R", (), {"profile": type("P", (), {
        "name": "rehearsal", "scientific_eligible": False})()})()

    with pytest.raises(StatusError, match="scientific evidence"):
        _Adapter().result(request, mode="VERIFY_LOCK", checks=[],
                          scientific_evidence=True)


def test_only_verify_lock_claims_scientific_evidence() -> None:
    claimants = [name for name, node in
                 {n.name: n for n in ast.walk(ast.parse(C4_SOURCE))
                  if isinstance(n, ast.FunctionDef)}.items()
                 if "scientific_evidence=" in (ast.get_source_segment(C4_SOURCE, node) or "")]

    assert claimants == ["_scientific_verify_lock"], claimants


def test_the_stage_scientific_axis_follows_the_adapter_results() -> None:
    """It was hard-coded NOT_RUN, which is the other half of `PASS / sci=NOT_RUN`."""
    orchestrator = (REPO / "src" / "prism_fas" / "pipeline" / "orchestrator.py"
                    ).read_text(encoding="utf-8")

    assert 'scientific="NOT_RUN"' not in orchestrator.split("_validate_stage")[0], (
        "the executing branch must derive the axis, not hard-code it")
    assert "item.status_axes.scientific" in orchestrator


def test_an_engineering_record_cannot_satisfy_the_scientific_lock_path() -> None:
    """C5 blocks on `reports/full/c4/GPAT_CONFIG_LOCK.json`. The engineering
    record has a different name, a different schema and says so."""
    assert c4.C4Adapter.SCIENTIFIC_LOCK == "GPAT_CONFIG_LOCK.json"
    assert "C4_ENGINEERING_CONFIG_RECORD.json" != c4.C4Adapter.SCIENTIFIC_LOCK
    assert "c4-engineering-config-record-v1" in C4_SOURCE
    assert "c4-gpat-config-lock-v1" in C4_SOURCE


# --- 15-16. source-only, and the selection tuple -----------------------------

def test_the_scientific_path_opens_no_target_and_no_source_dev() -> None:
    for name, node in _scientific_functions().items():
        source = ast.get_source_segment(C4_SOURCE, node) or ""
        lowered = source.lower()
        assert "siw" not in lowered, name
        # The two forbidden split names may appear only as an explicit
        # NOT-opened assertion, or as a key read out of the trainer's own
        # source-only audit report — never as a path being resolved.
        for split in ("target_test", "source_dev"):
            for line in source.splitlines():
                if split not in line.lower():
                    continue
                assert (f"{split}_opened" in line or "isolation" in line
                        or "manifests_opened" in line or "are not read" in line), (
                    f"{name}: {line.strip()}")


def test_the_selection_tuple_is_read_from_the_trainer_not_invented() -> None:
    """Every field of `GPAT_SELECTION_TUPLE` already exists in what
    `GPATTrainer.validate` returns, so no new metric is introduced."""
    from prism_fas.pipeline.adapters.c4 import _selection_metrics
    from prism_fas.search.plan import GPAT_SELECTION_TUPLE

    summary = {"best": {"validation_total_loss": 0.25, "epoch": 3},
               "history": [{"epoch": 3, "validation_total_loss": 0.25,
                            "validation_identity": 0.10,
                            "validation_ll_invariant_max_abs_error": 1e-7,
                            "validation_outside_mask_max_abs_error": 0.0}]}
    config = {"invariants": {"ll_max_abs_error": 1e-5,
                             "outside_mask_max_abs_error": 0.0}}

    metrics = _selection_metrics(summary, config)

    assert set(metrics) == set(GPAT_SELECTION_TUPLE)
    assert metrics["hard_invariant_failure"] is False
    assert metrics["neutral_support_validation_objective"] == 0.25
    assert metrics["identity_drift"] == 0.10
    assert metrics["low_frequency_geometry_drift"] == 1e-7
    assert metrics["outside_mask_error"] == 0.0


def test_an_invariant_violation_ranks_last() -> None:
    from prism_fas.pipeline.adapters.c4 import _selection_metrics

    summary = {"best": {"validation_total_loss": 0.01, "epoch": 0},
               "history": [{"epoch": 0, "validation_total_loss": 0.01,
                            "validation_identity": 0.0,
                            "validation_ll_invariant_max_abs_error": 1.0,
                            "validation_outside_mask_max_abs_error": 0.0}]}
    metrics = _selection_metrics(summary, {"invariants": {"ll_max_abs_error": 1e-5,
                                                          "outside_mask_max_abs_error": 0.0}})

    assert metrics["hard_invariant_failure"] is True, (
        "the leading tuple field ranks any invariant failure after every passing "
        "configuration, however good its loss"
    )
