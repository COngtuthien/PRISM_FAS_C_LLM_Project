"""The learning-rate coordinate is searched under BOTH approved interpretations.

The defect this file is the regression for, found before the first C7 scientific
trial: `search/lr_decision.py` derived "is the learning-rate coordinate searched"
from "does the multiplier expand over more than one parameter group", so
`UNIQUE_INHERITED_ANCHOR` produced an inapplicable coordinate and Track G's
bounded pass silently omitted `learning_rate` — the FIRST coordinate of the
frozen §15.2.2 order. Track G would have run 12 trials instead of 15 and frozen
`config_G`'s learning rate at the inherited anchor without ever evaluating 0.5x
or 2x, under a lock that recorded a complete-looking one-pass envelope.

Two questions, and this file keeps them apart:

* **How was the anchor resolved?** `B_common_multiplier` when several inherited
  scalars are simultaneously applicable; `UNIQUE_INHERITED_ANCHOR` when exactly
  one is. That is what the approved decision record answers, and Track G needed
  no user decision because its answer is unique.
* **Is the coordinate searched?** Always, when an applicable anchor exists.
  §15.2.2 declares no exemption for a component whose anchor happens to be
  unique.

Every count here is derived from the canonical plan builders. Nothing is
hard-coded as the authority: the numbers are asserted against what
`detector_search_plan` and `gpat_search_plan` actually produce.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.evaluation import source_selection  # noqa: E402
from prism_fas.pipeline.adapters.c7 import (TRACK_G_FLAGS, TRACK_R_FLAGS,  # noqa: E402
                                            _active_terms, _variant)
from prism_fas.search import lr_decision  # noqa: E402
from prism_fas.search.c7_decision import load_decision as load_c7  # noqa: E402
from prism_fas.search.lr_decision import (COMMON_MULTIPLIER,  # noqa: E402
                                          UNIQUE_ANCHOR, load_decision, lr_coordinate)
from prism_fas.search.plan import (DETECTOR_COORDINATE_ORDER,  # noqa: E402
                                   K4_ONLY_WEIGHTS, detector_search_plan,
                                   gpat_search_plan)


@pytest.fixture(scope="module")
def record():
    return load_decision(REPO)


@pytest.fixture(scope="module")
def detector_config():
    return yaml.safe_load(
        (REPO / "configs/train/m9_reference.yaml").read_text(encoding="utf-8"))


def _plan(record, detector_config, track: str):
    variant = _variant(TRACK_R_FLAGS if track == "R" else TRACK_G_FLAGS)
    decision = load_c7(REPO)
    plan, _resolutions = detector_search_plan(
        detector_config, active_terms=_active_terms(variant),
        k4_weights=K4_ONLY_WEIGHTS,
        selection_tuple=source_selection.TUPLES[decision.selection_tuple_name],
        lr_decision=record.for_component(f"C7_TRACK_{track}"))
    return plan


# --- (1) the interpretation no longer decides whether we search --------------

def test_unique_inherited_anchor_is_not_an_inapplicable_coordinate(record) -> None:
    """The exact defect, stated as its own assertion."""
    track_g = record.for_component("C7_TRACK_G")
    assert track_g.interpretation == UNIQUE_ANCHOR

    coordinate = lr_coordinate(track_g)
    assert coordinate.applicable is True, (
        "UNIQUE_INHERITED_ANCHOR produced an inapplicable learning-rate "
        "coordinate; it describes how the anchor was RESOLVED, not whether the "
        "frozen §15.2.2 coordinate is searched")
    assert coordinate.candidates == (0.5, 1.0, 2.0)


def test_the_two_questions_are_separate_properties(record) -> None:
    track_g = record.for_component("C7_TRACK_G")
    track_r = record.for_component("C7_TRACK_R")

    # "does the multiplier expand over several groups" — differs by track
    assert track_r.searches_a_multiplier is True
    assert track_g.searches_a_multiplier is False
    # "is the frozen coordinate searched" — true for both
    assert track_r.searches_the_learning_rate is True
    assert track_g.searches_the_learning_rate is True


def test_a_component_with_no_applicable_anchor_is_still_skipped() -> None:
    """The one case that legitimately contributes no coordinate stays skipped.

    An ABSENT anchor is a different thing from a uniquely RESOLVED one, and
    §15.2.3 skips an absent scalar rather than inventing it. Losing that
    distinction in the other direction would be as wrong as the original defect.
    """
    absent = lr_decision.LRAnchorDecision(
        component="TEST", interpretation=UNIQUE_ANCHOR, anchor_vector={},
        multipliers=(0.5, 1.0, 2.0), coordinate_name="learning_rate_multiplier",
        preserved_ratio=(), parameter_groups=(), compliance_class="TEST",
        anchor_source="")

    assert absent.searches_the_learning_rate is False
    coordinate = lr_coordinate(absent)
    assert coordinate.applicable is False
    assert "ABSENT" in coordinate.skip_reason


# --- (2)(3)(4)(5)(6)(7) Track G's corrected contract --------------------------

def test_track_g_has_exactly_one_applicable_lr_coordinate(record,
                                                          detector_config) -> None:
    plan = _plan(record, detector_config, "G")
    lr_coordinates = [item for item in plan.coordinates
                      if "learning_rate" in item.name]

    assert len(lr_coordinates) == 1, (
        f"expected exactly one learning-rate coordinate, found "
        f"{[item.name for item in lr_coordinates]}")
    assert lr_coordinates[0].applicable is True


def test_track_g_lr_multipliers_are_the_frozen_set(record, detector_config) -> None:
    plan = _plan(record, detector_config, "G")
    coordinate = next(item for item in plan.coordinates
                      if item.name == "learning_rate_multiplier")

    assert list(coordinate.candidates) == [0.5, 1.0, 2.0]


def test_track_g_effective_learning_rates_are_exact(record) -> None:
    """anchor x {0.5, 1.0, 2.0} over the one applicable group."""
    track_g = record.for_component("C7_TRACK_G")

    assert dict(track_g.anchor_vector) == {"head_lr": 1.0e-4}
    assert track_g.lr_for_groups(0.5) == {"head_lr": 5.0e-5}
    assert track_g.lr_for_groups(1.0) == {"head_lr": 1.0e-4}
    assert track_g.lr_for_groups(2.0) == {"head_lr": 2.0e-4}


def test_the_track_g_anchor_trial_reproduces_the_inherited_rate(record) -> None:
    """m = 1.0 IS the inherited configuration, so the search starts where
    §15.2.2 says it starts."""
    track_g = record.for_component("C7_TRACK_G")

    assert track_g.lr_for_groups(1.0) == {name: float(value) for name, value
                                          in track_g.anchor_vector.items()}
    assert track_g.as_dict()["anchor_trial_reproduces_version_b"] is True


def test_track_g_introduces_no_backbone_lr_group(record) -> None:
    """§13.4.1: Track G instantiates no ConvNeXt, so backbone_lr controls nothing."""
    track_g = record.for_component("C7_TRACK_G")

    assert "backbone_lr" not in track_g.anchor_vector
    assert list(track_g.parameter_groups) == ["heads"]
    for multiplier in track_g.candidates:
        assert set(track_g.lr_for_groups(multiplier)) == {"head_lr"}


def test_track_g_lr_coordinate_is_first_in_the_frozen_order(record,
                                                            detector_config) -> None:
    plan = _plan(record, detector_config, "G")
    track_g = record.for_component("C7_TRACK_G")

    assert DETECTOR_COORDINATE_ORDER[0] == "learning_rate"
    assert plan.coordinate_order[0] == track_g.coordinate_name
    assert [item.name for item in plan.coordinates if item.applicable][0] == (
        track_g.coordinate_name)


def test_track_g_tunes_no_structurally_inactive_term(record, detector_config) -> None:
    plan = _plan(record, detector_config, "G")
    applicable = {item.name for item in plan.coordinates if item.applicable}

    assert applicable == {"learning_rate_multiplier", "weight_decay", "warmup",
                          "lambda_syn", "lambda_risk"}
    for name in ("lambda_local", "lambda_MIL", "lambda_P", *K4_ONLY_WEIGHTS):
        assert name not in applicable


# --- (8)(9)(10)(11) the corrected declared size ------------------------------

def test_track_g_declares_fifteen_trials(record, detector_config) -> None:
    plan = _plan(record, detector_config, "G")

    # 5 applicable coordinates x 3 candidates, derived rather than asserted flat.
    assert plan.total_trials == sum(len(item.candidates)
                                    for item in plan.coordinates if item.applicable)
    assert plan.total_trials == 15


def test_track_r_still_declares_twenty_four_trials(record, detector_config) -> None:
    plan = _plan(record, detector_config, "R")

    assert plan.total_trials == 24


def test_the_total_c7_envelope_is_thirty_nine_trials(record,
                                                     detector_config) -> None:
    decision = load_c7(REPO)
    total = sum(_plan(record, detector_config, track).total_trials
                for track in decision.tracks)

    assert sorted(decision.tracks) == ["G", "R"]
    assert total == 39


def test_the_declared_optimizer_step_count_derives_to_61425(record,
                                                            detector_config) -> None:
    decision = load_c7(REPO)
    total = sum(_plan(record, detector_config, track).total_trials
                for track in decision.tracks)
    epochs = int(detector_config["stages"]["total_epochs"])
    steps = int(detector_config["batch"]["steps_per_epoch"])

    assert epochs == 35
    assert steps == 45
    assert total * epochs * steps == 61_425


# --- (12)(13) Track R and C4 are untouched -----------------------------------

def test_track_r_semantics_and_ratio_are_unchanged(record, detector_config) -> None:
    track_r = record.for_component("C7_TRACK_R")

    assert track_r.interpretation == COMMON_MULTIPLIER
    assert dict(track_r.anchor_vector) == {"backbone_lr": 1.0e-5, "head_lr": 1.0e-4}
    assert list(track_r.candidates) == [0.5, 1.0, 2.0]
    for multiplier in track_r.candidates:
        rates = track_r.lr_for_groups(multiplier)
        assert track_r.ratio_preserved(multiplier)
        assert rates["head_lr"] / rates["backbone_lr"] == pytest.approx(10.0)
    assert track_r.lr_for_groups(1.0) == {"backbone_lr": 1.0e-5, "head_lr": 1.0e-4}


def test_c4_gpat_semantics_and_ratio_are_unchanged(record) -> None:
    c4 = record.for_component("C4")

    assert c4.interpretation == COMMON_MULTIPLIER
    assert dict(c4.anchor_vector) == {"encoder_lr": 2.0e-4, "recipe_lr": 1.0e-4,
                                      "generator_lr": 2.0e-4}
    assert list(c4.candidates) == [0.5, 1.0, 2.0]
    for multiplier in c4.candidates:
        rates = c4.lr_for_groups(multiplier)
        assert c4.ratio_preserved(multiplier)
        assert rates["encoder_lr"] / rates["recipe_lr"] == pytest.approx(2.0)
        assert rates["generator_lr"] / rates["recipe_lr"] == pytest.approx(2.0)


def test_the_c4_frozen_search_plan_identity_did_not_move(record) -> None:
    """C4 has already executed scientifically. Its plan identity is frozen.

    `Coordinate.anchor_source` and `spec_clause` enter the plan identity, so a
    cosmetic decoration of either would move a hash C5 depends on. This is the
    guard for that, with the value C4's own scientific run hashed.
    """
    config = yaml.safe_load(
        (REPO / "configs/synthesis/gpat_m8.yaml").read_text(encoding="utf-8"))
    plan, _resolutions = gpat_search_plan(config,
                                          lr_decision=record.for_component("C4"))

    assert plan.identity == (
        "71bfff29bfe1e7ba71d083831a0337a6ae6e0dcfc7f7a75eb9e6f3f3a4ac2b6a")
    assert plan.total_trials == 12


# --- (16) exactly one LR coordinate, both tracks -----------------------------

@pytest.mark.parametrize("track", ("G", "R"))
def test_no_second_learning_rate_coordinate_is_introduced(record, detector_config,
                                                          track) -> None:
    plan = _plan(record, detector_config, track)
    order = list(plan.coordinate_order)

    assert order.count("learning_rate_multiplier") == 1
    assert "learning_rate" not in order, (
        "the multiplier replaces the per-scalar coordinate IN PLACE; both present "
        "would be two learning-rate steps in a one-pass envelope")
    assert len(order) == len(set(order))
    assert len(order) == len(DETECTOR_COORDINATE_ORDER) + len(K4_ONLY_WEIGHTS)


# --- the injection test: restore the defect, prove it is caught --------------

def test_restoring_the_defective_rule_fails_the_track_g_plan(record, detector_config,
                                                             monkeypatch) -> None:
    """The test that would have caught this before a GPU hour was spent.

    Re-imposes the exact defective rule — UNIQUE_INHERITED_ANCHOR yields no
    candidates — and asserts the Track-G plan degrades in precisely the way the
    real defect degraded it: the learning rate leaves the applicable set and the
    envelope shrinks from 15 trials to 12.
    """
    defective = lr_decision.LRAnchorDecision(
        **{**{field: getattr(record.for_component("C7_TRACK_G"), field)
              for field in ("component", "interpretation", "anchor_vector",
                            "coordinate_name", "preserved_ratio", "parameter_groups",
                            "compliance_class", "anchor_source", "rationale")},
           "multipliers": ()})

    assert defective.searches_the_learning_rate is False   # the defect, restored
    coordinate = lr_coordinate(defective)
    assert coordinate.applicable is False

    variant = _variant(TRACK_G_FLAGS)
    plan, _resolutions = detector_search_plan(
        detector_config, active_terms=_active_terms(variant),
        k4_weights=K4_ONLY_WEIGHTS,
        selection_tuple=source_selection.TUPLES["P3_READY"], lr_decision=defective)
    applicable = [item.name for item in plan.coordinates if item.applicable]

    assert "learning_rate_multiplier" not in applicable, (
        "the defective rule no longer removes the coordinate — this injection "
        "test has stopped testing anything")
    assert plan.total_trials == 12

    # And the corrected decision, on the same inputs, restores all 15.
    corrected = _plan(record, detector_config, "G")
    assert "learning_rate_multiplier" in [item.name for item in corrected.coordinates
                                          if item.applicable]
    assert corrected.total_trials == 15
    assert corrected.identity != plan.identity


# --- (14)(15) resume behaviour across the corrected envelope -----------------

def test_a_defective_search_state_fails_closed_against_the_corrected_plan(
        record, detector_config, tmp_path) -> None:
    """(14). The old 12-trial state may not be resumed into the 15-trial envelope."""
    from prism_fas.search.coordinate import SearchError, TrialResult, coordinate_search

    defective = lr_decision.LRAnchorDecision(
        **{**{field: getattr(record.for_component("C7_TRACK_G"), field)
              for field in ("component", "interpretation", "anchor_vector",
                            "coordinate_name", "preserved_ratio", "parameter_groups",
                            "compliance_class", "anchor_source", "rationale")},
           "multipliers": ()})
    variant = _variant(TRACK_G_FLAGS)
    old_plan, _r = detector_search_plan(
        detector_config, active_terms=_active_terms(variant),
        k4_weights=K4_ONLY_WEIGHTS,
        selection_tuple=source_selection.TUPLES["P3_READY"], lr_decision=defective)

    def objective(trial):
        return TrialResult(trial=trial, status="PASS",
                           metrics={name: 0.1 for name in old_plan.selection_tuple})

    state = tmp_path / "C7_SCIENTIFIC_SEARCH_STATE_G.json"
    coordinate_search(old_plan, objective, state_path=state, resume=False)
    assert state.is_file()

    corrected = _plan(record, detector_config, "G")
    with pytest.raises(SearchError, match="failing closed"):
        coordinate_search(corrected, objective, state_path=state, resume=True)


def test_a_completed_corrected_track_g_pass_resumes_with_zero_re_execution(
        record, detector_config, tmp_path) -> None:
    """(15). The corrected 15-trial envelope resumes as a completed unit."""
    from prism_fas.search.coordinate import TrialResult, coordinate_search

    plan = _plan(record, detector_config, "G")

    def objective(trial):
        return TrialResult(trial=trial, status="PASS",
                           metrics={name: 0.1 for name in plan.selection_tuple})

    state = tmp_path / "C7_SCIENTIFIC_SEARCH_STATE_G.json"
    first = coordinate_search(plan, objective, state_path=state, resume=False)
    assert first.status == "COMPLETED"
    assert len(first.results) == 15

    executed: list[str] = []

    def counting(trial):
        executed.append(trial.config_sha256)
        return objective(trial)

    second = coordinate_search(plan, counting, state_path=state, resume=True)

    assert executed == []
    assert second.plan.identity == plan.identity
    assert second.best_config == first.best_config
    assert ([item.trial.config_sha256 for item in second.results]
            == [item.trial.config_sha256 for item in first.results])
    assert second.winner.trial.config_sha256 == first.winner.trial.config_sha256


# --- (17)(18)(19) the surrounding contract is untouched ----------------------

def test_the_search_arm_is_still_det_and_there_is_no_per_arm_search() -> None:
    decision = load_c7(REPO)

    assert decision.training_arm == "DET"
    assert decision.decision_status == "FROZEN"
    assert decision.permits_arm("RND") is False
    assert decision.permits_arm("LLM") is False
    assert sorted(decision.tracks) == ["G", "R"]
    assert decision.as_dict()["per_arm_search_authorized"] is False


def test_the_correction_record_reports_zero_affected_trials() -> None:
    import json

    payload = json.loads(
        (REPO / "reports/handoff/LR_ANCHOR_DECISION_CORRECTION.json").read_text(
            encoding="utf-8"))
    affected = payload["affected_scientific_evidence"]

    assert payload["classification"] == "IMPLEMENTATION_CORRECTION"
    assert affected["c7_scientific_trials_executed_before_correction"] == 0
    assert affected["detector_config_lock_written_before_correction"] is False
    assert affected["target_accessed"] is False
    assert affected["target_access"] == 0
    assert payload["identity_change"]["decision_config_bytes"]["changed"] is False
    assert payload["identity_change"]["c4_gpat_search_plan_identity"]["changed"] is False
    assert payload["identity_change"]["lr_decision_identity"]["changed"] is True
    assert payload["supersedes"]["record_preserved_unchanged"] is True


def test_the_frozen_decision_record_was_not_rewritten() -> None:
    """The approved scientific values were already correct; no byte changed."""
    import hashlib
    import json

    correction = json.loads(
        (REPO / "reports/handoff/LR_ANCHOR_DECISION_CORRECTION.json").read_text(
            encoding="utf-8"))
    measured = hashlib.sha256(
        (REPO / "configs/search/lr_anchor_decision.yaml").read_bytes()).hexdigest()
    recorded = json.loads(
        (REPO / "reports/handoff/LR_ANCHOR_DECISION_RECORD.json").read_text(
            encoding="utf-8"))

    assert measured == recorded["decision_config_sha256"]
    assert measured == correction["identity_change"]["decision_config_bytes"]["sha256"]
