"""Identity-aware resume (L.11) and operational budget (L.12).

The resume tests are written from the consequence backwards. Skipping a unit
that should have re-run silently reuses stale science; re-running a unit that
should have been skipped spends live provider quota or GPU hours on evidence
that already exists. Both directions are tested.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from prism_fas.pipeline.budget import (BudgetError, BudgetLedger, assert_reduction_allowed,
                                       ledger_for)
from prism_fas.pipeline.profiles import load_profile
from prism_fas.pipeline.resume import (EXECUTE, INVALIDATE, SKIP, CompletedUnit,
                                       ResumeError, decide, invalidation_subtree,
                                       protected_from_regeneration)
from prism_fas.pipeline.stages import get_stage

PARENTS = {"bank_contract": "aaa", "ontology": "bbb"}


def _unit(**kwargs) -> CompletedUnit:
    defaults = dict(unit_id="C3/RND/seed0", stage_id="C3", parent_identities=dict(PARENTS),
                    config_identity="cfg-1", content_hash="hash-1",
                    acceptance_state="PASS")
    defaults.update(kwargs)
    return CompletedUnit(**defaults)


def _decide(unit: CompletedUnit, **overrides):
    kwargs = dict(expected_parents=dict(PARENTS), expected_config_identity="cfg-1",
                  expected_content_hash="hash-1")
    kwargs.update(overrides)
    return decide(unit, **kwargs)


# --- resume -----------------------------------------------------------------

def test_all_four_conditions_together_permit_a_skip() -> None:
    decision = _decide(_unit())
    assert decision.action == SKIP
    assert decision.may_skip
    assert all(decision.checks.values())


def test_a_drifted_parent_identity_invalidates_the_subtree() -> None:
    decision = _decide(_unit(parent_identities={"bank_contract": "CHANGED", "ontology": "bbb"}))
    assert decision.action == INVALIDATE
    assert decision.invalidation_subtree[0] == "C3"
    assert "C13" in decision.invalidation_subtree


def test_a_drifted_config_identity_invalidates_the_subtree() -> None:
    decision = _decide(_unit(config_identity="cfg-2"))
    assert decision.action == INVALIDATE
    assert not decision.checks["config_identity_validates"]


def test_a_changed_content_hash_re_executes_without_invalidating() -> None:
    """The identities still agree, so only this unit's work is redone."""
    decision = _decide(_unit(content_hash="hash-2"))
    assert decision.action == EXECUTE
    assert decision.invalidation_subtree == ()


def test_an_unknown_expected_hash_is_not_treated_as_a_match() -> None:
    decision = _decide(_unit(), expected_content_hash=None)
    assert decision.action == EXECUTE
    assert not decision.checks["content_hash_validates"]


def test_an_unaccepted_unit_is_not_skipped() -> None:
    decision = _decide(_unit(acceptance_state="FAIL"))
    assert decision.action == EXECUTE
    assert not decision.checks["acceptance_state_validates"]


def test_a_missing_artifact_is_not_skipped_even_when_identities_agree(
        tmp_path: Path) -> None:
    decision = _decide(_unit(artifact_path="reports/c3/gone.json"), repo=tmp_path)
    assert decision.action == EXECUTE
    assert not decision.checks["artifact_present"]


def test_a_present_artifact_with_valid_identities_is_skipped(tmp_path: Path) -> None:
    artifact = tmp_path / "reports" / "c3" / "here.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")
    decision = _decide(_unit(artifact_path="reports/c3/here.json"), repo=tmp_path)
    assert decision.action == SKIP


def test_the_subtree_is_the_suffix_of_the_declared_stage_order() -> None:
    assert invalidation_subtree("C11") == ("C11", "C12", "C13")
    assert invalidation_subtree("C0")[0] == "C0"
    assert len(invalidation_subtree("C0")) == 14


def test_an_unknown_stage_cannot_produce_a_subtree() -> None:
    with pytest.raises(ResumeError, match="unknown stage"):
        invalidation_subtree("C99")


def test_c3_and_c4_are_protected_from_silent_regeneration() -> None:
    """L.11 names both: live Gemini re-generation and GPAT retraining."""
    assert protected_from_regeneration(get_stage("C3"))
    assert protected_from_regeneration(get_stage("C4"))
    assert not protected_from_regeneration(get_stage("C0"))


def test_a_valid_c3_archive_is_skipped_rather_than_regenerated() -> None:
    """The concrete L.11 case: a restart must not re-spend provider quota."""
    decision = _decide(_unit(stage_id="C3"))
    assert decision.may_skip


# --- budget -----------------------------------------------------------------

def test_validate_has_no_budget_to_exhaust(repo: Path) -> None:
    ledger = ledger_for(load_profile("validate", repo=repo))
    assert not ledger.bounded
    assert ledger.exceeded() == []


def test_smoke_is_bounded_by_its_engineering_budget(repo: Path) -> None:
    ledger = ledger_for(load_profile("smoke", repo=repo))
    assert ledger.bounded
    assert ledger.limits["max_epochs"] == 1


def test_full_is_unbounded_so_no_cap_can_shrink_the_science(repo: Path) -> None:
    ledger = ledger_for(load_profile("full", repo=repo))
    assert not ledger.bounded
    assert ledger.exceeded() == []


def test_exceeding_a_smoke_cap_is_detected(repo: Path) -> None:
    ledger = ledger_for(load_profile("smoke", repo=repo))
    ledger.spend("max_epochs", 5)
    assert "max_epochs" in ledger.exceeded()


def test_a_budget_is_never_a_treatment_factor(repo: Path) -> None:
    ledger = ledger_for(load_profile("smoke", repo=repo))
    assert ledger.as_dict()["treatment_factor"] is False


def test_an_eligible_profile_carrying_a_budget_is_refused() -> None:
    from prism_fas.pipeline.profiles import ComputePolicy, ExecutionProfile

    forged = ExecutionProfile(
        name="full", purpose="", scientific_eligible=True,
        may_select_scientific_winner=True, winner_selection_rule=None,
        reports_namespace="reports/full", runs_namespace="runs/full",
        compute_policy=ComputePolicy(True, "required", True, "gated", ("C3",),
                                     "c12_scorer_only", "c12_scorer_only"),
        engineering_budget={"max_epochs": 1}, reduction_permitted=False,
        preserved_under_reduction=(), phases=(), config_path="forged",
        profile_identity="0" * 64)
    with pytest.raises(BudgetError, match="must not carry an operational budget cap"):
        ledger_for(forged)


def test_validate_may_not_reduce_anything(repo: Path) -> None:
    with pytest.raises(BudgetError, match="does not permit budget reduction"):
        assert_reduction_allowed(load_profile("validate", repo=repo), "samples")


def test_smoke_may_reduce_an_authorized_dimension(repo: Path) -> None:
    assert_reduction_allowed(load_profile("smoke", repo=repo), "seeds")


def test_smoke_may_not_reduce_an_unauthorized_dimension(repo: Path) -> None:
    """Reducing a loss term is a scientific change wearing an engineering label."""
    with pytest.raises(BudgetError, match="is not a reducible dimension"):
        assert_reduction_allowed(load_profile("smoke", repo=repo), "loss_terms")


def test_an_unbounded_ledger_reports_no_exhaustion_action() -> None:
    ledger = BudgetLedger(profile_name="full")
    assert "no operational cap" in ledger.as_dict()["on_exhaustion"]
