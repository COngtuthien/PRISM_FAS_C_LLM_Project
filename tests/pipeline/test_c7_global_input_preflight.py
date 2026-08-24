"""The first real GPU C7 attempt, as regressions.

That run produced a scientific-looking verdict from a missing file. The frozen
recipe text cache was absent from the host; every one of Track G's 15 candidates
raised the identical `TextCacheError`; the engine's generic `except Exception`
turned each into a retained FAIL; the coordinate pass "completed" with zero
finite-valid trials; and `EnvelopeExhausted` — a §15.2.2 verdict meaning *these
detector configurations do not work* — was raised and then mislabelled
`SEARCH_STATE_IDENTITY_MISMATCH` because the broad handler sat first.

Four defects in one run, each tested here:

* the text cache was never a verified detector input, so nothing blocked;
* a GLOBAL dependency failure was recorded as a per-configuration FAIL;
* `EnvelopeExhausted` subclasses `SearchError`, so its own handler was dead code;
* 15 logical search positions shared 11 config-keyed artifacts, and the later
  occurrence overwrote the earlier one's provenance.

Nothing here trains anything. The stubs sit exactly where a laptop cannot go.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from conftest_adapters import request_for  # noqa: E402
from prism_fas.detector.heads import RECIPE_TEXT_CACHE_RELPATHS, TextCacheError  # noqa: E402
from prism_fas.evaluation import source_selection  # noqa: E402
from prism_fas.pipeline.adapters import c7, sources  # noqa: E402
from prism_fas.search.coordinate import (EnvelopeExhausted,  # noqa: E402
                                         FatalDependencyError, SearchError,
                                         TrialResult, coordinate_search)
from prism_fas.search.plan import Coordinate, SearchPlan  # noqa: E402


@pytest.fixture(scope="module")
def pin() -> dict[str, Any]:
    """The canonical pin. Read from the config that owns it, never restated."""
    model = yaml.safe_load(
        (REPO / "configs/models/m9_detector.yaml").read_text(encoding="utf-8"))
    return dict((model.get("model") or {}).get("prompt") or model.get("prompt") or {})


# --- (1)(3)(4)(5) the text cache is a verified detector input ----------------

def test_the_canonical_pin_owns_both_expected_values(pin) -> None:
    """No new magic constants: both hashes already lived in the model config."""
    assert pin["cache_file"] == "recipe_text_cache.npz"
    assert pin["cache_file_sha256"] == (
        "bb7d3fb4b82ad6ac89ebb06eeac9eb679e2fbb3bab500112cd1e304c187683aa")
    assert pin["cache_identity_sha256"] == (
        "10f4ec35b7563b2b658cacc94599d35b9f93b531963a065459d4694d5dc2c141")


def test_an_absent_cache_blocks_and_refuses_to_rebuild(tmp_path, pin) -> None:
    """(1)(3). The failure names the artifact and forbids regenerating it."""
    repo = tmp_path / "repo"
    (repo / "configs" / "models").mkdir(parents=True)
    (repo / "configs/models/m9_detector.yaml").write_text(
        (REPO / "configs/models/m9_detector.yaml").read_text(encoding="utf-8"),
        encoding="utf-8")
    (repo / "weights").mkdir()

    with pytest.raises(sources.DetectorInputsUnavailable) as caught:
        sources._frozen_recipe_text_cache(repo)

    message = str(caught.value)
    assert "recipe text cache is absent" in message
    assert pin["cache_file_sha256"] in message
    assert "may NOT be rebuilt" in message
    assert all(relative in message for relative in RECIPE_TEXT_CACHE_RELPATHS)


def test_wrong_file_bytes_are_refused_before_any_semantic_check(tmp_path,
                                                                pin) -> None:
    """(3). A file at the right path with the wrong bytes is not the artifact."""
    repo = tmp_path / "repo"
    (repo / "configs" / "models").mkdir(parents=True)
    (repo / "configs/models/m9_detector.yaml").write_text(
        (REPO / "configs/models/m9_detector.yaml").read_text(encoding="utf-8"),
        encoding="utf-8")
    (repo / "weights").mkdir()
    (repo / "weights/recipe_text_cache.npz").write_bytes(b"not the frozen artifact")

    with pytest.raises(sources.DetectorInputsUnavailable) as caught:
        sources._frozen_recipe_text_cache(repo)

    message = str(caught.value)
    assert "hashes to" in message
    assert pin["cache_file_sha256"] in message
    assert "Do not rebuild it" in message


def test_the_verifier_checks_bytes_and_semantics_independently() -> None:
    """(4). Two checks, because a file could satisfy either one alone.

    Structural: the verifier must both hash the file AND re-derive the semantic
    identity through the canonical loader. Dropping either leaves a hole — bytes
    alone would accept a correctly-hashed file whose binding was edited, and
    identity alone would accept a re-encoded cache that happened to agree.
    """
    import inspect

    body = inspect.getsource(sources._frozen_recipe_text_cache)

    assert "hashlib.sha256(path.read_bytes()).hexdigest()" in body
    assert "resolve_recipe_text_cache" in body
    assert "expected_identity=expected_identity" in body
    assert "cache_file_sha256" in body and "cache_identity_sha256" in body
    # And it must never fall back, download or rebuild. Checked as CALLS, not as
    # substrings: the function legitimately DECLARES `downloaded_during_run:
    # False`, and a blunt substring ban would flag the very field that records
    # the guarantee.
    for forbidden in ("build_recipe_text_cache(", "allow_text_cache_build=True",
                      "urlretrieve(", "hf_hub_download(", "snapshot_download("):
        assert forbidden not in body, forbidden
    assert body.count("rebuilt_at_runtime") == 1
    assert '"rebuilt_at_runtime": False' in body


def test_the_evidence_binds_every_declared_field() -> None:
    """(5). What a passing verification must record for the lock to bind."""
    import inspect

    body = inspect.getsource(sources._frozen_recipe_text_cache)
    for field in ('"path"', '"file_sha256"', '"cache_identity_sha256"',
                  '"recipe_count"', '"rebuilt_at_runtime": False'):
        assert field in body, field


def test_detector_inputs_expose_the_cache_to_c7_and_c8() -> None:
    """One verifier, shared: fixing C7's input hole closes C8's too."""
    import inspect

    body = inspect.getsource(sources.verify_detector_inputs)
    assert "_frozen_recipe_text_cache" in body
    assert '"recipe_text_cache": text_cache' in body


# --- (2) no trial is emitted for a missing global input ----------------------

def test_a_global_dependency_failure_never_becomes_a_trial_result() -> None:
    """(2)(6). The engine propagates it instead of converting it to FAIL."""
    plan = SearchPlan(
        plan_id="test", milestone="CX",
        coordinates=(Coordinate("alpha", 1.0, (0.5, 1.0, 2.0)),),
        selection_tuple=("objective", "epoch"))
    seen: list[int] = []

    def evaluate(trial):
        seen.append(trial.trial_index)
        raise FatalDependencyError("the frozen recipe text cache is missing",
                                   dependency="TextCacheError")

    with pytest.raises(FatalDependencyError) as caught:
        coordinate_search(plan, evaluate, require_valid_winner=False)

    assert caught.value.reason_code == "GLOBAL_DEPENDENCY_UNAVAILABLE"
    assert caught.value.dependency == "TextCacheError"
    # It aborted on the FIRST candidate. It did not spend all three.
    assert seen == [0], "the pass continued after a global dependency failure"


def test_a_configuration_specific_failure_is_still_a_retained_fail() -> None:
    """(7). The distinction has to cut both ways or it is not a distinction."""
    plan = SearchPlan(
        plan_id="test", milestone="CX",
        coordinates=(Coordinate("alpha", 1.0, (0.5, 1.0, 2.0)),),
        selection_tuple=("objective", "epoch"))

    def evaluate(trial):
        if trial.value == 2.0:
            raise RuntimeError("this learning rate diverged")
        return TrialResult(trial=trial, status="PASS",
                           metrics={"objective": trial.value, "epoch": 1})

    outcome = coordinate_search(plan, evaluate, require_valid_winner=False)
    statuses = [item.status for item in outcome.results]

    assert len(outcome.results) == 3, "one bad config lost the others"
    assert statuses.count("FAIL") == 1
    assert statuses.count("PASS") == 2


def test_a_non_finite_metric_is_still_diverged() -> None:
    """(8). DIVERGED is retained and ranks after every finite-valid trial."""
    plan = SearchPlan(
        plan_id="test", milestone="CX",
        coordinates=(Coordinate("alpha", 1.0, (0.5, 1.0, 2.0)),),
        selection_tuple=("objective", "epoch"))

    def evaluate(trial):
        if trial.value == 0.5:
            return TrialResult(trial=trial, status="DIVERGED", metrics={})
        return TrialResult(trial=trial, status="PASS",
                           metrics={"objective": trial.value, "epoch": 1})

    outcome = coordinate_search(plan, evaluate, require_valid_winner=False)
    ranked = outcome.leaderboard()

    assert [item.status for item in outcome.results].count("DIVERGED") == 1
    assert ranked[-1].status == "DIVERGED"
    assert ranked[0].finite_valid is True


def test_the_classifier_is_typed_and_not_a_string_match() -> None:
    """(6). Prose is not a contract; the allowlist is exception types."""
    import inspect

    types = c7._global_dependency_errors()
    names = {item.__name__ for item in types}

    assert {"TextCacheError", "PretrainedError", "C6BankError",
            "ScientificDeviceUnavailable"} <= names
    body = inspect.getsource(c7._run_scientific_trial)
    assert "except global_errors" in body
    assert "FatalDependencyError" in body
    # No prose matching anywhere in the classification path.
    for forbidden in ('in str(error)', 'error.args[0] ==', '"missing" in'):
        assert forbidden not in body, forbidden


def test_the_engine_preserves_state_when_a_dependency_aborts(tmp_path) -> None:
    """The pass aborts, but what it already wrote survives for recovery."""
    plan = SearchPlan(
        plan_id="test", milestone="CX",
        coordinates=(Coordinate("alpha", 1.0, (0.5, 1.0, 2.0)),),
        selection_tuple=("objective", "epoch"))
    state = tmp_path / "SEARCH_STATE.json"

    def evaluate(trial):
        raise FatalDependencyError("gone", dependency="TextCacheError")

    with pytest.raises(FatalDependencyError):
        coordinate_search(plan, evaluate, state_path=state, require_valid_winner=False)

    assert state.is_file()
    payload = json.loads(state.read_text(encoding="utf-8"))
    assert payload["status"] == "BLOCKED_ON_DEPENDENCY"
    assert payload["results"] == [], "a candidate was consumed by a global failure"


# --- (9)(10)(11) the exception order ----------------------------------------

def test_envelope_exhausted_is_a_search_error_subclass() -> None:
    """The fact that makes handler ORDER load-bearing."""
    assert issubclass(EnvelopeExhausted, SearchError)
    assert not issubclass(FatalDependencyError, SearchError)


def test_c7_catches_the_specific_handlers_before_the_broad_one() -> None:
    """(11). The defect, as a structural assertion.

    On the first GPU run a genuine `EnvelopeExhausted` was reported as
    `SEARCH_STATE_IDENTITY_MISMATCH`, because `except SearchError` sat first and
    `EnvelopeExhausted` subclasses it. Ordering is the whole fix, so ordering is
    what this pins.
    """
    import inspect

    body = inspect.getsource(c7.C7Adapter._scientific_search)
    fatal = body.index("except FatalDependencyError")
    exhausted = body.index("except EnvelopeExhausted")
    broad = body.index("except SearchError")

    assert fatal < exhausted < broad, (
        "EnvelopeExhausted must be caught before the broader SearchError, or its "
        "handler is dead code")


def test_the_two_search_errors_map_to_different_verdicts() -> None:
    """(9)(10). One is a scientific decision; the other is a fail-closed stop."""
    import inspect

    body = inspect.getsource(c7.C7Adapter._scientific_search)
    exhausted = body[body.index("except EnvelopeExhausted"):body.index("except SearchError")]
    mismatch = body[body.index("except SearchError"):]

    assert "NEEDS_SCIENTIFIC_DECISION" in exhausted
    assert "SEARCH_STATE_IDENTITY_MISMATCH" not in exhausted
    assert "SEARCH_STATE_IDENTITY_MISMATCH" in mismatch
    assert "NEEDS_SCIENTIFIC_DECISION" not in mismatch


def test_a_plan_identity_mismatch_still_raises_the_broad_search_error(tmp_path) -> None:
    """(10). Reordering must not stop a real mismatch from being caught."""
    first = SearchPlan(plan_id="test", milestone="CX",
                       coordinates=(Coordinate("alpha", 1.0, (0.5, 1.0, 2.0)),),
                       selection_tuple=("objective", "epoch"))
    other = SearchPlan(plan_id="test", milestone="CX",
                       coordinates=(Coordinate("alpha", 5.0, (0.5, 1.0, 2.0)),),
                       selection_tuple=("objective", "epoch"))
    state = tmp_path / "SEARCH_STATE.json"

    def evaluate(trial):
        return TrialResult(trial=trial, status="PASS",
                           metrics={"objective": trial.value, "epoch": 1})

    coordinate_search(first, evaluate, state_path=state, require_valid_winner=False)
    with pytest.raises(SearchError, match="failing closed") as caught:
        coordinate_search(other, evaluate, state_path=state, resume=True)

    assert not isinstance(caught.value, EnvelopeExhausted)


def test_a_genuine_envelope_exhaustion_still_raises_envelope_exhausted() -> None:
    """(9). Every candidate really failed on its own merits."""
    plan = SearchPlan(plan_id="test", milestone="CX",
                      coordinates=(Coordinate("alpha", 1.0, (0.5, 1.0, 2.0)),),
                      selection_tuple=("objective", "epoch"))

    def evaluate(trial):
        return TrialResult(trial=trial, status="DIVERGED", metrics={})

    with pytest.raises(EnvelopeExhausted, match="no finite valid configuration"):
        coordinate_search(plan, evaluate, require_valid_winner=True)


# --- (12)(13)(14)(15) the logical-occurrence evidence model ------------------

def _plan_for(track: str):
    from prism_fas.search.c7_decision import load_decision as load_c7
    from prism_fas.search.lr_decision import load_decision as load_lr
    from prism_fas.search.plan import K4_ONLY_WEIGHTS, detector_search_plan

    config = yaml.safe_load(
        (REPO / "configs/train/m9_reference.yaml").read_text(encoding="utf-8"))
    decision = load_c7(REPO)
    variant = c7._variant(c7.TRACK_R_FLAGS if track == "R" else c7.TRACK_G_FLAGS)
    plan, _resolutions = detector_search_plan(
        config, active_terms=c7._active_terms(variant), k4_weights=K4_ONLY_WEIGHTS,
        selection_tuple=source_selection.TUPLES[decision.selection_tuple_name],
        lr_decision=load_lr(REPO).for_component(f"C7_TRACK_{track}"))
    return plan


@pytest.mark.parametrize("track,occurrences,unique", [("G", 15, 11), ("R", 24, 17)])
def test_both_counts_are_derived_from_the_canonical_plan(track, occurrences,
                                                         unique) -> None:
    """(12). The 11 artifacts the GPU run left were not a bug in the count.

    A coordinate pass evaluates each coordinate's candidates while the others sit
    at the current best, so when the anchor wins a coordinate the anchor
    configuration recurs at the next one: same canonical SHA, different search
    position. Both numbers come from the plan, neither is written down.
    """
    plan = _plan_for(track)

    assert plan.total_trials == occurrences
    assert c7._unique_configurations(plan) == unique
    assert c7._unique_configurations(plan) < plan.total_trials


def test_the_total_envelope_reports_both_counts_and_the_honest_compute() -> None:
    """(12). 39 occurrences over 28 configurations; compute uses the second."""
    config = yaml.safe_load(
        (REPO / "configs/train/m9_reference.yaml").read_text(encoding="utf-8"))
    epochs = int(config["stages"]["total_epochs"])
    steps = int(config["batch"]["steps_per_epoch"])

    occurrences = sum(_plan_for(track).total_trials for track in ("G", "R"))
    unique = sum(c7._unique_configurations(_plan_for(track)) for track in ("G", "R"))

    assert occurrences == 39
    assert unique == 28
    assert unique * epochs * steps == 44_100
    assert occurrences * epochs * steps == 61_425


def test_two_occurrences_of_one_config_get_distinct_records() -> None:
    """(13). Occurrence roots are keyed by search position, not configuration."""
    from prism_fas.search.coordinate import Trial

    plan = _plan_for("G")
    first = Trial.create(trial_index=3, coordinate="weight_decay", value=0.05,
                         config=dict(plan.base_config), plan_identity=plan.identity)
    later = Trial.create(trial_index=6, coordinate="warmup", value=0.05,
                         config=dict(plan.base_config), plan_identity=plan.identity)

    # Same configuration, by construction.
    assert first.config_sha256 == later.config_sha256
    assert c7._trial_run_root(Path("runs"), first.config_sha256) == \
        c7._trial_run_root(Path("runs"), later.config_sha256)
    # Different logical occurrence, so different record.
    assert c7._occurrence_root(Path("runs"), "G", first) != \
        c7._occurrence_root(Path("runs"), "G", later)


def test_the_occurrence_record_declares_reuse_explicitly() -> None:
    """(14). Reuse is recorded, not inferred from a missing file."""
    import inspect

    body = inspect.getsource(c7._run_scientific_trial)
    for field in ('"config_evidence_reused"', '"config_evidence_produced_here"',
                  '"config_run_summary"', '"trial_index"', '"coordinate"'):
        assert field in body, field
    # Reuse requires the SAME config sha under the same plan, not merely a file.
    assert 'existing.get("trial_config_sha256") == trial.config_sha256' in body


def test_a_completed_unique_config_is_not_retrained(tmp_path) -> None:
    """(15). L.11 at trial granularity, over the corrected occurrence model."""
    plan = SearchPlan(plan_id="test", milestone="CX",
                      coordinates=(Coordinate("alpha", 1.0, (0.5, 1.0, 2.0)),
                                   Coordinate("beta", 2.0, (0.5, 1.0, 2.0))),
                      selection_tuple=("objective", "epoch"))
    state = tmp_path / "SEARCH_STATE.json"

    def objective(trial):
        return TrialResult(trial=trial, status="PASS",
                           metrics={"objective": abs(trial.config["alpha"] - 1.0),
                                    "epoch": 1})

    first = coordinate_search(plan, objective, state_path=state, resume=False)
    executed: list[str] = []

    def counting(trial):
        executed.append(trial.config_sha256)
        return objective(trial)

    second = coordinate_search(plan, counting, state_path=state, resume=True)

    assert executed == []
    assert second.best_config == first.best_config


# --- (16)(17)(18) the recovery procedure ------------------------------------

def _write_state(repo: Path, track: str, rows: list[dict[str, Any]], *,
                 status: str = "COMPLETED",
                 plan_identity: str = "a" * 64) -> Path:
    from prism_fas.pipeline.adapters.c7 import _search_state_name
    from prism_fas.pipeline.state import atomic_write_json

    path = repo / "reports/full/c7" / _search_state_name(track)
    atomic_write_json(path, {
        "schema_version": "prism-coordinate-search-v1",
        "search_plan_identity": plan_identity, "status": status,
        "completed_coordinates": ["learning_rate_multiplier"], "results": rows,
        "best_config": {}})
    return path


def _failed_row(index: int, note: str) -> dict[str, Any]:
    return {"trial_index": index, "coordinate": "weight_decay", "value": 0.05,
            "config": {}, "config_id": f"weight_decay@{index:012d}",
            "config_sha256": f"{index:064d}", "status": "FAIL", "metrics": {},
            "finite_valid": False, "notes": [note], "artifacts": []}


def test_recovery_accepts_a_pure_global_precondition_failure(tmp_path) -> None:
    """(16). The exact shape the first GPU attempt left behind."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "recover_c7", REPO / "scripts/recover_c7_invalid_search_state.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    repo = tmp_path / "repo"
    note = ("TextCacheError: the frozen recipe text cache is missing under weights")
    state = _write_state(repo, "G", [_failed_row(index, note) for index in range(15)])

    report = module.assess(repo, "G")
    assert report["eligible"] is True, report["problems"]
    assert report["results"] == 15
    assert report["finite_valid"] == 0
    assert report["global_failure_types"] == ["TextCacheError"]
    assert report["training_progress"]["optimizer_step_evidence"] is False

    result = module.quarantine(repo, "G", report, reason="missing frozen text cache")

    # Preserved with hashes, and only the active state cleared.
    assert result["preserved_count"] >= 1
    assert result["classification"] == "ENGINEERING_GLOBAL_INPUT_FAILURE"
    assert result["scientific_envelope_exhausted"] is False
    assert result["candidates_consumed"] == 0
    assert result["target_access"] == 0
    assert not state.exists()
    preserved = repo / result["preserved"][0]["preserved_as"]
    assert preserved.is_file()
    assert len(result["preserved"][0]["sha256"]) == 64
    assert (repo / result["quarantine_root"] / "RECOVERY_RECORD.json").is_file()


@pytest.mark.parametrize("rows,expected", [
    # (17) a real scientific outcome anywhere refuses the quarantine
    ([{**_failed_row(0, "TextCacheError: missing"), "status": "PASS",
       "finite_valid": True}], "finite-valid"),
    # mixed causes are not one global precondition failure
    ([_failed_row(0, "TextCacheError: missing"),
      _failed_row(1, "RuntimeError: this learning rate diverged")],
     "no recognised global failure type"),
])
def test_recovery_refuses_anything_that_might_be_scientific(tmp_path, rows,
                                                            expected) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "recover_c7", REPO / "scripts/recover_c7_invalid_search_state.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    repo = tmp_path / "repo"
    state = _write_state(repo, "G", rows)
    report = module.assess(repo, "G")

    assert report["eligible"] is False
    assert any(expected in problem for problem in report["problems"]), report["problems"]
    assert state.is_file(), "an ineligible state was touched"


def test_recovery_refuses_a_different_search_plan_identity(tmp_path) -> None:
    """(18). Recovering the wrong envelope's state would erase real evidence."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "recover_c7", REPO / "scripts/recover_c7_invalid_search_state.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    repo = tmp_path / "repo"
    note = "TextCacheError: the frozen recipe text cache is missing"
    _write_state(repo, "G", [_failed_row(0, note)], plan_identity="a" * 64)

    report = module.assess(repo, "G", expected_plan_identity="b" * 64)
    assert report["eligible"] is False
    assert any("search_plan_identity" in problem for problem in report["problems"])


def test_recovery_refuses_when_a_detector_config_lock_exists(tmp_path) -> None:
    from prism_fas.pipeline.state import atomic_write_json
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "recover_c7", REPO / "scripts/recover_c7_invalid_search_state.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    repo = tmp_path / "repo"
    note = "TextCacheError: the frozen recipe text cache is missing"
    _write_state(repo, "G", [_failed_row(0, note)])
    atomic_write_json(repo / c7.SCIENTIFIC_CONFIG_LOCK_PATH, {"is_scientific_lock": True})

    report = module.assess(repo, "G")
    assert report["eligible"] is False
    assert any("DETECTOR_CONFIG_LOCK" in problem for problem in report["problems"])


def test_recovery_touches_no_frozen_decision(tmp_path) -> None:
    """The recovery is not a result-driven restart of anything."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "recover_c7", REPO / "scripts/recover_c7_invalid_search_state.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    repo = tmp_path / "repo"
    note = "TextCacheError: missing"
    _write_state(repo, "G", [_failed_row(index, note) for index in range(3)])
    result = module.quarantine(repo, "G", module.assess(repo, "G"), reason="test")

    untouched = " ".join(result["not_touched"])
    for name in ("c7_source_search_decision", "lr_anchor_decision", "c6", "c5"):
        assert name in untouched
    assert result["cleared"] == ["reports/full/c7/C7_SCIENTIFIC_SEARCH_STATE_G.json"]


# --- (19)(20)(21) nothing frozen moved --------------------------------------

def test_the_c4_scientific_plan_identity_is_unchanged() -> None:
    """(19). C4 has executed scientifically; none of this may reach it."""
    from prism_fas.search.lr_decision import load_decision
    from prism_fas.search.plan import gpat_search_plan

    config = yaml.safe_load(
        (REPO / "configs/synthesis/gpat_m8.yaml").read_text(encoding="utf-8"))
    plan, _resolutions = gpat_search_plan(
        config, lr_decision=load_decision(REPO).for_component("C4"))

    assert plan.identity == (
        "71bfff29bfe1e7ba71d083831a0337a6ae6e0dcfc7f7a75eb9e6f3f3a4ac2b6a")
    assert plan.total_trials == 12


def test_the_c7_search_arm_decision_is_unchanged() -> None:
    """(20). DET, frozen, identity untouched by any of this."""
    from prism_fas.search.c7_decision import load_decision

    decision = load_decision(REPO)
    assert decision.training_arm == "DET"
    assert decision.decision_status == "FROZEN"
    assert decision.identity == (
        "ed4f6b777d9f95f089a76191b863e2fb2df0b9e13434470ffd736d6e511b474e")


def test_target_access_is_absent_from_every_path_touched_here() -> None:
    """(21)."""
    import inspect

    for function in (sources._frozen_recipe_text_cache, sources.verify_detector_inputs,
                     c7._run_scientific_trial):
        body = inspect.getsource(function)
        for forbidden in ("siw", "target_test", "target_label"):
            assert forbidden not in body.lower().replace("target_labels_resolved", ""), (
                f"{function.__name__} mentions {forbidden}")


def test_the_c4_evaluator_cannot_raise_the_new_fatal_exception() -> None:
    """C4 shares the engine, so the new contract must not change its behaviour."""
    import inspect

    from prism_fas.pipeline.adapters import c4

    body = inspect.getsource(c4)
    assert "FatalDependencyError" not in body, (
        "C4 now raises the fatal-dependency exception; its scientific behaviour "
        "would change and it has already executed")
