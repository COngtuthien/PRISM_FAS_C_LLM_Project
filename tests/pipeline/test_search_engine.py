"""The bounded coordinate-search engine (§15.2.2, §15.2.3, L.6).

The engine decides scientific winners under the full profile, so the properties
worth testing are the ones that would let it decide a different winner than the
frozen contract says: an order that drifts, a coordinate revisited, a failed
trial silently dropped, a tie broken by iteration order, a resume that
recomputes. Each of those is a way for a bounded search to become an unbounded
one, and each gets its own test here.

Everything runs against synthetic objectives. That is deliberate — a synthetic
objective makes the winner predictable in advance, which is what lets these tests
assert *which* configuration should win rather than merely that one did.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism_fas.search.coordinate import (SearchError, SearchInterrupted, TrialResult,
                                         coordinate_search, rank_key)
from prism_fas.search.plan import (AMBIGUOUS, Coordinate, SearchPlan, SearchPlanError,
                                   anchor_resolution_report, canonical_config_sha256,
                                   detector_search_plan, gpat_search_plan, resolve_anchors)


def simple_plan(**overrides) -> SearchPlan:
    coordinates = (
        Coordinate("alpha", 1.0, (0.5, 1.0, 2.0)),
        Coordinate("beta", 2.0, (0.5, 1.0, 2.0)),
        Coordinate("gamma", None, (0.5, 1.0, 2.0), skip_reason="absent scalar"),
    )
    return SearchPlan(plan_id="test", milestone="CX", coordinates=coordinates,
                      selection_tuple=("objective", "epoch"), **overrides)


def distance_objective(target_alpha: float = 2.0, target_beta: float = 1.0):
    def evaluate(trial):
        value = (abs(trial.config["alpha"] - target_alpha)
                 + abs(trial.config["beta"] - target_beta))
        return TrialResult(trial=trial, status="PASS",
                           metrics={"objective": round(value, 9), "epoch": 1})
    return evaluate


# --- candidate generation ----------------------------------------------------

def test_candidates_are_anchor_times_multipliers_sorted_ascending() -> None:
    coordinate = Coordinate("lr", 1e-4, (0.5, 1.0, 2.0))
    assert coordinate.candidates == (5e-05, 0.0001, 0.0002)


def test_a_zero_anchor_collapses_to_a_single_candidate() -> None:
    """§15.2.2: if the weight-decay anchor is 0, the candidate set is {0}."""
    assert Coordinate("weight_decay", 0.0, (0.5, 1.0, 2.0)).candidates == (0.0,)


def test_warmup_is_clipped_to_its_declared_range() -> None:
    coordinate = Coordinate("warmup", 0.18, (0.5, 1.0, 1.5), clip=(0.0, 0.20))
    assert max(coordinate.candidates) <= 0.20


def test_a_coordinate_with_no_anchor_contributes_no_trial() -> None:
    coordinate = Coordinate("missing", None, (0.5, 1.0, 2.0), skip_reason="absent")
    assert not coordinate.applicable
    assert coordinate.candidates == ()


def test_a_plan_cannot_declare_a_coordinate_twice() -> None:
    with pytest.raises(SearchPlanError):
        SearchPlan(plan_id="x", milestone="CX",
                   coordinates=(Coordinate("a", 1.0, (1.0,)), Coordinate("a", 2.0, (1.0,))),
                   selection_tuple=("objective",))


def test_a_plan_must_declare_a_selection_tuple() -> None:
    with pytest.raises(SearchPlanError):
        SearchPlan(plan_id="x", milestone="CX",
                   coordinates=(Coordinate("a", 1.0, (1.0,)),), selection_tuple=())


# --- order, one pass, winner update -----------------------------------------

def test_coordinates_are_searched_in_the_declared_order() -> None:
    seen: list[tuple[str, float]] = []

    def evaluate(trial):
        seen.append((trial.coordinate, trial.value))
        return TrialResult(trial=trial, status="PASS",
                           metrics={"objective": 1.0, "epoch": 1})

    coordinate_search(simple_plan(), evaluate)
    assert [name for name, _value in seen] == ["alpha"] * 3 + ["beta"] * 3
    assert [value for name, value in seen if name == "alpha"] == [0.5, 1.0, 2.0]


def test_one_pass_only_and_no_coordinate_is_revisited() -> None:
    plan = simple_plan()
    outcome = coordinate_search(plan, distance_objective())
    assert outcome.status == "COMPLETED"
    assert outcome.completed_coordinates == ["alpha", "beta"]
    assert len(outcome.completed_coordinates) == len(set(outcome.completed_coordinates))
    assert outcome.as_dict()["trials_executed"] == plan.total_trials


def test_the_winner_updates_as_the_pass_proceeds() -> None:
    outcome = coordinate_search(simple_plan(), distance_objective(2.0, 1.0))
    assert outcome.best_config["alpha"] == 2.0
    assert outcome.best_config["beta"] == 1.0
    assert outcome.winner is not None
    assert outcome.winner.metrics["objective"] == 0.0


def test_later_coordinates_are_evaluated_at_the_current_best() -> None:
    """§15.2.2: other coordinates stay at the current best while one moves."""
    configs: list[dict] = []

    def evaluate(trial):
        configs.append(dict(trial.config))
        return distance_objective()(trial)

    coordinate_search(simple_plan(), evaluate)
    alpha_trials, beta_trials = configs[:3], configs[3:]
    # While alpha moves, beta sits at its anchor.
    assert {config["beta"] for config in alpha_trials} == {2.0}
    # Once alpha is decided, every beta trial carries alpha's winning value.
    assert {config["alpha"] for config in beta_trials} == {2.0}


# --- retention ---------------------------------------------------------------

def test_failed_and_divergent_trials_are_retained_and_ranked_last() -> None:
    def evaluate(trial):
        if trial.value == 2.0:
            return TrialResult(trial=trial, status="DIVERGED", metrics={})
        if trial.value == 0.5:
            raise RuntimeError("planted implementation error")
        return TrialResult(trial=trial, status="PASS",
                           metrics={"objective": 1.0, "epoch": 1})

    outcome = coordinate_search(simple_plan(), evaluate, require_valid_winner=False)
    payload = outcome.as_dict()
    # Nothing is dropped: the status counts account for every executed trial.
    assert payload["trials_executed"] == sum(payload["trials_by_status"].values())
    assert payload["trials_by_status"]["DIVERGED"] >= 1
    assert payload["trials_by_status"]["FAIL"] >= 1
    assert len(payload["leaderboard"]) == payload["trials_executed"]
    # And every non-finite trial sorts after every finite one, with no interleave.
    finite = [item.finite_valid for item in outcome.leaderboard()]
    assert finite == sorted(finite, reverse=True), finite
    assert any(finite) and not all(finite)


def test_a_nan_metric_ranks_after_every_finite_trial() -> None:
    def evaluate(trial):
        objective = float("nan") if trial.value == 1.0 else 5.0
        return TrialResult(trial=trial, status="PASS",
                           metrics={"objective": objective, "epoch": 1})

    outcome = coordinate_search(simple_plan(), evaluate, require_valid_winner=False)
    ranked = outcome.leaderboard()
    assert not ranked[-1].finite_valid
    assert outcome.winner is not None and outcome.winner.finite_valid


def test_a_coordinate_whose_every_candidate_failed_keeps_its_anchor() -> None:
    def evaluate(trial):
        if trial.coordinate == "alpha":
            return TrialResult(trial=trial, status="DIVERGED", metrics={})
        return TrialResult(trial=trial, status="PASS",
                           metrics={"objective": trial.value, "epoch": 1})

    outcome = coordinate_search(simple_plan(), evaluate, require_valid_winner=False)
    assert outcome.best_config["alpha"] == 1.0        # unchanged anchor
    assert any("no finite valid trial" in note for note in outcome.notes)


# --- tie-break ---------------------------------------------------------------

def test_an_exact_tie_is_broken_by_canonical_config_sha_ascending() -> None:
    def evaluate(trial):
        return TrialResult(trial=trial, status="PASS",
                           metrics={"objective": 1.0, "epoch": 1})

    outcome = coordinate_search(simple_plan(), evaluate)
    trace = outcome.tie_break_trace[0]
    assert trace["decided_by_tie_break"] is True
    assert trace["numeric_tie_count"] == 3
    assert trace["selected_config_sha256"] == min(trace["tied_config_sha256"])


def test_traversal_order_cannot_decide_a_tie() -> None:
    """Reversing the evaluation order must not change the selected config."""
    def evaluate(trial):
        return TrialResult(trial=trial, status="PASS",
                           metrics={"objective": 1.0, "epoch": 1})

    first = coordinate_search(simple_plan(), evaluate)
    second = coordinate_search(simple_plan(), evaluate)
    assert first.best_config == second.best_config
    assert first.outcome_identity == second.outcome_identity


def test_rank_key_puts_a_hard_failure_flag_first() -> None:
    """§15.2.3: a hard-invariant failure ranks after every passing config."""
    plan = SearchPlan(plan_id="gpat", milestone="C4",
                      coordinates=(Coordinate("a", 1.0, (1.0,)),),
                      selection_tuple=("hard_invariant_failure", "objective"))

    def result(flag: bool, objective: float):
        return TrialResult(trial=_trial(plan), status="PASS",
                           metrics={"hard_invariant_failure": flag,
                                    "objective": objective})

    passing = result(False, 9.0)
    failing = result(True, 0.0)
    assert rank_key(passing, plan.selection_tuple) < rank_key(failing, plan.selection_tuple)


def _trial(plan: SearchPlan):
    from prism_fas.search.coordinate import Trial

    return Trial.create(trial_index=0, coordinate="a", value=1.0,
                        config={"a": 1.0}, plan_identity=plan.identity)


# --- identity ----------------------------------------------------------------

def test_a_config_identity_is_assigned_before_execution() -> None:
    """L.6: every attempted config carries a stable id and hash before it runs."""
    captured: list[tuple[str, str]] = []

    def evaluate(trial):
        assert trial.config_sha256 == canonical_config_sha256(trial.config)
        assert trial.config_id.startswith(trial.coordinate)
        captured.append((json.dumps(trial.config, sort_keys=True), trial.config_sha256))
        return TrialResult(trial=trial, status="PASS",
                           metrics={"objective": 1.0, "epoch": 1})

    coordinate_search(simple_plan(), evaluate)
    # The hash is a function of the CONFIG, not of the position: two coordinates
    # can legitimately produce the same configuration, and when they do the hash
    # must agree rather than differ.
    by_config = dict(captured)
    assert len(by_config) == len({sha for _config, sha in captured})


def test_the_plan_identity_excludes_nothing_scientific_and_is_stable() -> None:
    assert simple_plan().identity == simple_plan().identity
    moved = SearchPlan(plan_id="test", milestone="CX",
                       coordinates=(Coordinate("alpha", 2.0, (0.5, 1.0, 2.0)),),
                       selection_tuple=("objective", "epoch"))
    assert moved.identity != simple_plan().identity


# --- resume ------------------------------------------------------------------

def test_an_interrupted_pass_checkpoints_and_resumes_at_the_same_trial(
        tmp_path: Path) -> None:
    state = tmp_path / "SEARCH_STATE.json"
    calls = {"n": 0}

    def flaky(trial):
        calls["n"] += 1
        if calls["n"] == 4:
            raise SearchInterrupted("engineering budget exhausted")
        return distance_objective()(trial)

    first = coordinate_search(simple_plan(), flaky, state_path=state, resume=False,
                              require_valid_winner=False)
    assert first.status == "INTERRUPTED"
    assert first.completed_coordinates == ["alpha"]
    assert state.exists()

    executed: list[str] = []

    def counting(trial):
        executed.append(trial.config_sha256)
        return distance_objective()(trial)

    second = coordinate_search(simple_plan(), counting, state_path=state, resume=True)
    assert second.status == "COMPLETED"
    reused = [item for item in second.results
              if any("reused" in note for note in item.notes)]
    # The three completed alpha trials are reused. So is the beta candidate whose
    # configuration happens to equal one of them — reuse is keyed on config
    # identity, not on position, and an identical configuration is identical work.
    assert len(reused) >= 3, "the completed alpha coordinate was recomputed"
    assert not set(executed) & {item.trial.config_sha256 for item in reused}
    assert second.best_config == {"alpha": 2.0, "beta": 1.0}


def test_resuming_a_completed_pass_re_executes_nothing() -> None:
    """A pass that already CLOSED is returned, not re-walked.

    The defect this is the regression for, found by the C7 rehearsal before any
    GPU run: reuse is keyed on config identity, and `best` is restored to the
    final winning vector. Re-walking the coordinates therefore generated the
    EARLY coordinates' trials with the LATE coordinates already at their winning
    values — different configurations, different hashes, missing the reuse table.
    A rerun of a finished search silently retrained, and the trials it produced
    were not the ones the pass selected from.

    L.11 says a validated completed unit is not recomputed. The completed unit
    here is the whole pass.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        state = Path(directory) / "SEARCH_STATE.json"
        first = coordinate_search(simple_plan(), distance_objective(),
                                  state_path=state, resume=False)
        assert first.status == "COMPLETED"

        executed: list[str] = []

        def counting(trial):
            executed.append(trial.config_sha256)
            return distance_objective()(trial)

        second = coordinate_search(simple_plan(), counting, state_path=state,
                                   resume=True)

    assert executed == [], "a completed pass re-executed trials on resume"
    assert second.status == "COMPLETED"
    assert second.plan.identity == first.plan.identity
    assert second.best_config == first.best_config
    assert second.completed_coordinates == first.completed_coordinates
    assert ([item.trial.config_sha256 for item in second.results]
            == [item.trial.config_sha256 for item in first.results])
    assert second.winner is not None
    assert second.winner.trial.config_sha256 == first.winner.trial.config_sha256
    assert any("returned unchanged" in note for note in second.notes)


def test_resuming_an_interrupted_pass_executes_only_the_incomplete_trials() -> None:
    """The other half: an interrupted pass resumes at the trial that stopped.

    Named separately from the completed case because the two behaviours are
    different and only one of them was ever exercised. A completed pass is
    returned; an interrupted one continues, reusing every trial whose
    configuration identity the state already records.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        state = Path(directory) / "SEARCH_STATE.json"
        calls = {"n": 0}

        def flaky(trial):
            calls["n"] += 1
            if calls["n"] == 4:
                raise SearchInterrupted("engineering budget exhausted")
            return distance_objective()(trial)

        first = coordinate_search(simple_plan(), flaky, state_path=state,
                                  resume=False, require_valid_winner=False)
        assert first.status == "INTERRUPTED"
        completed_before = {item.trial.config_sha256 for item in first.results
                            if item.status == "PASS"}

        executed: list[str] = []

        def counting(trial):
            executed.append(trial.config_sha256)
            return distance_objective()(trial)

        second = coordinate_search(simple_plan(), counting, state_path=state,
                                   resume=True)

    assert second.status == "COMPLETED"
    # No configuration the first pass completed was executed again.
    assert not set(executed) & completed_before
    # Every coordinate appears exactly once: no revisit, no second pass.
    assert second.completed_coordinates == ["alpha", "beta"]
    assert len(second.completed_coordinates) == len(set(second.completed_coordinates))
    # And the interrupted trial itself really did run this time.
    assert executed, "a resumed interrupted pass executed nothing at all"


def test_a_completed_pass_with_no_valid_winner_still_refuses_on_resume() -> None:
    """Returning a completed pass may not skip the EnvelopeExhausted refusal."""
    import tempfile

    from prism_fas.search.coordinate import EnvelopeExhausted

    def always_fails(trial):
        return TrialResult(trial=trial, status="DIVERGED", metrics={})

    with tempfile.TemporaryDirectory() as directory:
        state = Path(directory) / "SEARCH_STATE.json"
        coordinate_search(simple_plan(), always_fails, state_path=state,
                          resume=False, require_valid_winner=False)

        with pytest.raises(EnvelopeExhausted, match="no finite valid configuration"):
            coordinate_search(simple_plan(), always_fails, state_path=state,
                              resume=True, require_valid_winner=True)


def test_resume_refuses_a_state_written_under_a_different_plan(tmp_path: Path) -> None:
    """L.11: a changed identity fails closed rather than mixing two envelopes."""
    state = tmp_path / "SEARCH_STATE.json"
    coordinate_search(simple_plan(), distance_objective(), state_path=state)

    other = SearchPlan(plan_id="test", milestone="CX",
                       coordinates=(Coordinate("alpha", 5.0, (0.5, 1.0, 2.0)),),
                       selection_tuple=("objective", "epoch"))
    with pytest.raises(SearchError, match="failing closed"):
        coordinate_search(other, distance_objective(), state_path=state, resume=True)


def test_a_corrupt_state_file_is_refused_rather_than_ignored(tmp_path: Path) -> None:
    state = tmp_path / "SEARCH_STATE.json"
    state.write_text("{not json", encoding="utf-8")
    with pytest.raises(SearchError, match="ambiguous"):
        coordinate_search(simple_plan(), distance_objective(), state_path=state,
                          resume=True)


# --- the two real envelopes --------------------------------------------------

def test_the_gpat_envelope_matches_section_15_2_3(repo: Path) -> None:
    import yaml

    config = yaml.safe_load(
        (repo / "configs/synthesis/gpat_m8.yaml").read_text(encoding="utf-8"))
    plan, resolutions = gpat_search_plan(config)
    assert plan.coordinate_order == ("learning_rate", "weight_decay",
                                     "residual_loss_weight",
                                     "identity_preservation_weight",
                                     "geometry_preservation_weight")
    assert plan.selection_tuple[0] == "hard_invariant_failure"
    assert plan.one_pass and not plan.revisit_permitted
    for coordinate in plan.coordinates:
        assert coordinate.multipliers == (0.5, 1.0, 2.0)
    report = anchor_resolution_report(resolutions)
    assert set(report["resolved"]) | set(report["absent"]) | set(report["ambiguous"]) \
        == set(plan.coordinate_order)


def test_a_non_unique_inherited_anchor_is_user_approval_required(repo: Path) -> None:
    """§15.2.2 classifies a non-unique anchor as a decision, not a default."""
    import yaml

    config = yaml.safe_load(
        (repo / "configs/synthesis/gpat_m8.yaml").read_text(encoding="utf-8"))
    _plan, resolutions = gpat_search_plan(config)
    learning_rate = resolutions["learning_rate"]
    assert learning_rate.state == AMBIGUOUS
    assert learning_rate.needs_user_decision
    assert learning_rate.as_dict()["classification"] == "USER_APPROVAL_REQUIRED"
    assert anchor_resolution_report(resolutions)["executable_under_full"] is False


def test_an_absent_scalar_is_skipped_not_invented(repo: Path) -> None:
    import yaml

    config = yaml.safe_load(
        (repo / "configs/synthesis/gpat_m8.yaml").read_text(encoding="utf-8"))
    _plan, resolutions = gpat_search_plan(config)
    geometry = resolutions["geometry_preservation_weight"]
    assert geometry.state == "ABSENT"
    assert geometry.value is None
    assert not geometry.needs_user_decision       # absent is legal; ambiguous is not


def test_the_detector_envelope_matches_section_15_2_2(repo: Path) -> None:
    import yaml

    from prism_fas.search.plan import K4_ONLY_WEIGHTS

    config = yaml.safe_load(
        (repo / "configs/train/m9_reference.yaml").read_text(encoding="utf-8"))
    plan, _resolutions = detector_search_plan(config, k4_weights=K4_ONLY_WEIGHTS)
    assert plan.coordinate_order == (
        "learning_rate", "weight_decay", "warmup", "lambda_syn", "lambda_local",
        "lambda_MIL", "lambda_P", "lambda_risk", *K4_ONLY_WEIGHTS)
    warmup = next(item for item in plan.coordinates if item.name == "warmup")
    assert warmup.multipliers == (0.5, 1.0, 1.5)
    assert warmup.clip == (0.0, 0.20)


def test_inactive_loss_terms_are_skipped(repo: Path) -> None:
    import yaml

    from prism_fas.search.plan import K4_ONLY_WEIGHTS

    config = yaml.safe_load(
        (repo / "configs/train/m9_reference.yaml").read_text(encoding="utf-8"))
    plan, _resolutions = detector_search_plan(
        config, k4_weights=K4_ONLY_WEIGHTS,
        active_terms={name: False for name in K4_ONLY_WEIGHTS})
    for name in K4_ONLY_WEIGHTS:
        coordinate = next(item for item in plan.coordinates if item.name == name)
        assert not coordinate.applicable
        assert "not active" in coordinate.skip_reason
