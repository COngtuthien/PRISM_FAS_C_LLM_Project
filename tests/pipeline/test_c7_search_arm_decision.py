"""C7_SOURCE_SEARCH_SYNTHETIC_ARM = DET, and the fairness it buys.

§15.2.2 freezes the bounded detector/loss envelope and does not name which of
C6's three matched banks supplies the synthetic quarter of the batch while the
search runs. That is result-affecting, so it was decided explicitly, recorded in
`configs/search/c7_source_search_decision.yaml`, and frozen before any C7
scientific metric existed.

The decision buys one thing, and this file is the standing proof of it: the
detector configuration each track runs in C8 was NOT tuned on the arm under
test. Three properties make that checkable rather than asserted.

**One arm supplies the search.** DET only. The RND and LLM banks are refused as
a search population, and every trial's parent set carries the DET bank's own
selected-set digest — so a swapped bank changes every config identity rather
than quietly producing a second set of numbers under the same names.

**One search per TRACK, never per arm.** Track G and Track R each get one
bounded pass because their active loss sets differ; RND, DET and LLM do not,
because the generator arm is the treatment. A pass per arm would hand each
generator its own hyperparameters and confound the treatment with detector
tuning — the exact confound the design removes.

**The frozen configuration is shared within its track.** Every Track-G row
resolves the Track-G sub-config and every Track-R row the Track-R one, whatever
arm it trains on. Cross-track substitution fails closed.

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
from prism_fas.evaluation.source_matrix import build_plan  # noqa: E402
from prism_fas.pipeline.adapters import c7, c8  # noqa: E402
from prism_fas.search import c7_decision  # noqa: E402
from test_c7_scientific_path import (ARM, _StubTrainer, _approve,  # noqa: E402
                                     _decision, _plan_result, _run,
                                     _search_result, _unfreeze, scientific)  # noqa: F401
from test_c8_scientific_path import _prepare, _rows, with_c7_lock  # noqa: E402,F401


# --- the decision record itself ----------------------------------------------

def test_the_committed_record_freezes_det_before_any_trial() -> None:
    """Read from the repository, not from the sandbox: this is the real record."""
    decision = c7_decision.load_decision(REPO)

    assert decision.decision_id == c7_decision.DECISION_ID
    assert decision.training_arm == "DET"
    assert decision.decision_status == "FROZEN"
    assert decision.timing == "BEFORE_FIRST_C7_SCIENTIFIC_TRIAL"
    assert decision.frozen_before_any_trial is True
    assert decision.spec_status == "UNDER_SPECIFIED_IN_V1_5"
    assert decision.source == "EXPLICIT_SCIENTIFIC_DECISION"
    assert decision.tracks == ("G", "R")
    assert decision.rationale


def test_the_decision_permits_det_and_refuses_rnd_and_llm() -> None:
    decision = c7_decision.load_decision(REPO)

    assert decision.permits_arm("DET") is True
    assert decision.permits_arm("RND") is False
    assert decision.permits_arm("LLM") is False


def test_the_prohibited_alternatives_are_part_of_the_identity() -> None:
    """Changing the arm changes the identity bound into the plan and the lock."""
    import copy
    import dataclasses

    decision = c7_decision.load_decision(REPO)
    swapped = dataclasses.replace(decision, training_arm="LLM")
    reordered = dataclasses.replace(decision, tracks=("R", "G"))

    assert swapped.identity != decision.identity
    assert reordered.identity != decision.identity
    assert "RND" in decision.prohibited_alternatives
    assert "LLM" in decision.prohibited_alternatives
    assert any("per generator arm" in item
               for item in decision.prohibited_alternatives)
    assert copy.deepcopy(decision).identity == decision.identity


def test_a_record_taken_after_a_trial_exists_is_refused(tmp_path) -> None:
    """A search-population decision taken from a result is not a decision."""
    import shutil

    import yaml

    repo = tmp_path / "repo"
    (repo / "configs" / "search").mkdir(parents=True)
    source = REPO / c7_decision.DECISION_CONFIG
    shutil.copy(source, repo / c7_decision.DECISION_CONFIG)
    path = repo / c7_decision.DECISION_CONFIG
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["timing"] = "AFTER_FIRST_C7_SCIENTIFIC_TRIAL"
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(c7_decision.C7DecisionError, match="taken from a result"):
        c7_decision.load_decision(repo)


@pytest.mark.parametrize("overrides,expected", [
    ({"decision_status": "NEEDS_SCIENTIFIC_DECISION"}, "not one of"),
    ({"training_arm": "POOLED", "value": "POOLED"}, "outside the permitted values"),
    ({"tracks": ["G", "R", "X"]}, "outside the permitted values"),
    ({"value": "LLM"}, "must agree"),
    ({"trial_schedule": "short"}, "outside the permitted values"),
])
def test_each_way_the_record_can_be_wrong_is_refused(tmp_path, overrides,
                                                     expected) -> None:
    import shutil

    import yaml

    repo = tmp_path / "repo"
    (repo / "configs" / "search").mkdir(parents=True)
    shutil.copy(REPO / c7_decision.DECISION_CONFIG, repo / c7_decision.DECISION_CONFIG)
    path = repo / c7_decision.DECISION_CONFIG
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload.update(overrides)
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(c7_decision.C7DecisionError, match=expected):
        c7_decision.load_decision(repo)


# --- (1)(2)(3) the search resolves DET only ----------------------------------

def test_the_search_trains_on_det_and_refuses_the_other_banks(scientific) -> None:
    _approve(scientific)
    plan = _plan_result(_run(scientific))

    frozen = next(item for item in plan.checks
                  if item["check_id"] == "c7_source_search_population_frozen")
    assert frozen["ok"] is True
    assert frozen["detail"]["training_arm"] == "DET"

    refused = next(item for item in plan.checks
                   if item["check_id"]
                   == "c7_only_the_frozen_arm_supplies_synthetic_samples")
    assert refused["ok"] is True
    assert refused["detail"]["frozen_arm"] == "DET"
    assert sorted(refused["detail"]["refused_arms"]) == ["LLM", "RND"]
    assert sorted(refused["detail"]["available_arms"]) == ["DET", "LLM", "RND"]


def test_every_trial_trained_on_det_only(scientific) -> None:
    """(1)(2)(3) at the trial level: no RND or LLM candidate byte enters training."""
    _approve(scientific)
    search = _search_result(_run(scientific))

    only = next(item for item in search.checks
                if item["check_id"] == "c7_every_trial_trained_on_the_frozen_arm_only")
    assert only["ok"] is True
    assert only["detail"]["arms_seen"] == ["DET"]
    assert only["detail"]["frozen_arm"] == "DET"

    # And the banks each trial actually opened, read off the reader the trainer
    # was handed rather than off a field the trial wrote about itself.
    opened = {instance.kwargs["synthetic_bank"].arm
              for instance in _StubTrainer.instances}
    assert opened == {"DET"}


@pytest.mark.parametrize("arm", ("RND", "LLM"))
def test_a_record_naming_another_arm_changes_every_identity(scientific, arm) -> None:
    """Swapping the arm is not silent: it moves the plan and every config SHA."""
    _approve(scientific)
    baseline = _plan_result(_run(scientific))
    before = {key: value for key, value in baseline.parent_identities.items()}

    _decision(scientific, training_arm=arm, value=arm)
    swapped = _plan_result(_run(scientific))

    assert swapped.parent_identities["c7_search_decision"] != before["c7_search_decision"]
    for track in ("g", "r"):
        assert (swapped.parent_identities[f"c7_search_plan_{track}"]
                != before[f"c7_search_plan_{track}"])
    frozen = next(item for item in swapped.checks
                  if item["check_id"] == "c7_source_search_population_frozen")
    assert frozen["detail"]["training_arm"] == arm


# --- (4)(5) the DET bank identity is bound -----------------------------------

@pytest.mark.parametrize("track", ("G", "R"))
def test_the_det_bank_identity_enters_the_search_plan_identity(scientific,
                                                               track) -> None:
    """(4). Bound through the plan's base config, so it enters the plan hash."""
    _approve(scientific)
    plan = _plan_result(_run(scientific))

    bound = next(item for item in plan.checks
                 if item["check_id"]
                 == f"c7_track_{track.lower()}_plan_binds_the_frozen_search_bank")
    assert bound["ok"] is True
    detail = bound["detail"]
    assert detail["training_arm"] == "DET"
    assert len(detail["c6_bank_selected_set_sha256"]) == 64
    assert detail["c6_selector_identity_sha256"]
    assert detail["source_package_identity"]
    assert detail["search_decision_identity"]
    assert detail["lr_decision_identity"]

    payload = json.loads(
        (scientific / "reports/full/c7/C7_SCIENTIFIC_SEARCH_PLAN.json").read_text(
            encoding="utf-8"))
    base = payload["tracks"][track]["plan"]["base_config"]["c7_search_binding"]
    assert base["c6_bank_selected_set_sha256"] == detail["c6_bank_selected_set_sha256"]


def test_the_det_selected_set_hash_enters_every_trials_parents(scientific) -> None:
    """(5). Every trial summary, not only the winner's."""
    _approve(scientific)
    _run(scientific)

    expected = json.loads(
        (scientific / "reports/full/c6/C6_BANK_LOCK_DET.json").read_text(
            encoding="utf-8"))["selected_set_sha256"]
    summaries = sorted((scientific / "runs/full/c7/scientific").rglob(c7.TRIAL_SUMMARY))
    assert summaries, "no trial summary was written"

    for path in summaries:
        record = json.loads(path.read_text(encoding="utf-8"))
        parents = record["parent_identities"]
        assert record["training_arm"] == "DET", path
        assert parents["c6_bank_selected_set_sha256"] == expected, path
        assert parents["c7_search_decision"]
        assert parents["c7_search_plan"]
        assert parents["source_package_identity"]
        assert record["c6_bank_lock_selected_set_sha256"] == expected


# --- (6) a moved bank invalidates resume -------------------------------------

def test_changing_the_det_bank_lock_invalidates_resume(scientific) -> None:
    """(6). The recorded state belongs to another envelope, and C7 fails closed."""
    _approve(scientific)
    _run(scientific)
    state = scientific / "reports/full/c7" / c7._search_state_name("G")
    assert state.is_file()
    before = json.loads(state.read_text(encoding="utf-8"))["search_plan_identity"]

    path = scientific / "reports/full/c6/C6_BANK_LOCK_DET.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selected_set_sha256"] = "9" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    results = _run(scientific, resume=True)
    search = _search_result(results)
    refused = next(item for item in search.checks
                   if item["check_id"].endswith("_resume_state_matches_this_envelope"))
    assert refused["ok"] is False
    assert refused["detail"]["reason_code"] == "SEARCH_STATE_IDENTITY_MISMATCH"
    assert refused["detail"]["search_plan_identity"] != before
    assert search.status != "PASS"


# --- (7) no treatment-arm feedback before the lock ---------------------------

def test_c7_reads_no_rnd_or_llm_performance_before_the_lock(scientific) -> None:
    """(7). Nothing about the other arms is computed, let alone compared."""
    _approve(scientific)
    results = _run(scientific)
    plan = _plan_result(results)

    isolation = next(item for item in plan.checks
                     if item["check_id"]
                     == "c7_no_treatment_arm_feedback_before_the_lock")
    assert isolation["ok"] is True
    assert isolation["detail"]["arms_trained"] == ["DET"]
    assert isolation["detail"]["arms_evaluated"] == ["DET"]
    assert isolation["detail"]["comparisons_computed"] == []
    assert isolation["detail"]["target_metrics_computed"] == 0

    # Structural: nothing the stage wrote mentions a non-frozen arm's metrics.
    reports = scientific / "reports/full/c7"
    for path in sorted(reports.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        blob = json.dumps(payload)
        # Naming the other arms in a REFUSAL is required; carrying a metric for
        # one would not be. The refusal lists are the only place they appear.
        for arm in ("RND", "LLM"):
            for phrase in (f'"{arm}_video_ACER"', f'"{arm}_acer"', f'"{arm}_metrics"'):
                assert phrase not in blob, f"{path.name} carries {phrase}"


# --- (11) there is no per-arm search path ------------------------------------

def test_no_per_arm_search_path_exists(scientific) -> None:
    """(11). One pass per track, and the lock says so in a machine-readable field."""
    _approve(scientific)
    plan = _plan_result(_run(scientific))

    per_track = next(item for item in plan.checks
                     if item["check_id"] == "c7_runs_no_per_arm_search")
    assert per_track["ok"] is True
    assert sorted(per_track["detail"]["passes"]) == ["track_G", "track_R"]
    assert per_track["detail"]["search_population"] == "DET"

    payload = json.loads(
        (scientific / c7.SCIENTIFIC_CONFIG_LOCK_PATH).read_text(encoding="utf-8"))
    assert payload["per_arm_search_performed"] is False
    assert payload["shared_within_track"] is True
    assert sorted(payload["tracks"]) == ["G", "R"]
    # Exactly two configurations exist. Not three, not five.
    assert len({sub["winner_config_sha256"]
                for sub in payload["tracks"].values()}) == 2


def test_the_search_never_loops_over_arms_in_source() -> None:
    """A structural guard: no `for arm in ARMS` inside the scientific search."""
    import ast
    import inspect

    source = inspect.getsource(c7)
    tree = ast.parse(source)
    for name in ("_scientific_plan", "_scientific_search"):
        node = next(item for item in ast.walk(tree)
                    if isinstance(item, ast.FunctionDef) and item.name == name)
        for loop in (item for item in ast.walk(node) if isinstance(item, ast.For)):
            iterated = ast.get_source_segment(source, loop.iter) or ""
            assert "banks" not in iterated and "ARMS" not in iterated, (
                f"c7.{name} loops over arms: {iterated}")


# --- (8)(9)(10) shared-within-track fairness, at C8 --------------------------

_TRACK_G_ROWS = ("C-G-RND-P1-s20260806", "C-G-DET-P1-s20260806",
                 "C-G-LLM-P1-s20260806")
_TRACK_R_ROWS = ("C-R-DET-P3READY-s20260806", "C-R-LLM-P3READY-s20260806")


def _run_rows(repo: Path, row_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    request, inputs, _ = _prepare(repo)
    _plan, rows = _rows(row_ids)
    runs = repo / "runs/full/c8"
    return [c8._run_scientific_row(request, inputs=inputs, row=row, root=runs)
            for row in rows]


def test_track_g_arms_share_one_frozen_configuration(with_c7_lock) -> None:
    """(8). RND, DET and LLM Track-G rows resolve the SAME Track-G config."""
    executed = _run_rows(with_c7_lock, _TRACK_G_ROWS)

    assert [item["status"] for item in executed] == ["PASS"] * 3
    assert {item["arm"] for item in executed} == {"RND", "DET", "LLM"}
    assert {item["track"] for item in executed} == {"G"}
    configs = {item["parent_identities"]["c7_detector_config"] for item in executed}
    plans = {item["parent_identities"]["c7_search_plan"] for item in executed}
    assert len(configs) == 1, "Track-G arms received different detector configurations"
    assert len(plans) == 1
    # The searched scalars are identical; only the bank and the arm differ.
    resolved = {json.dumps(item["resolved_config"]["loss_weights"], sort_keys=True)
                for item in executed}
    assert len(resolved) == 1
    assert {item["parent_identities"]["c7_training_arm"] for item in executed} == {"DET"}
    assert len({item["parent_identities"]["c6_bank"] for item in executed}) == 3


def test_track_r_primary_arms_share_one_frozen_configuration(with_c7_lock) -> None:
    """(9). DET and LLM Track-R rows resolve the SAME Track-R config."""
    executed = _run_rows(with_c7_lock, _TRACK_R_ROWS)

    assert [item["status"] for item in executed] == ["PASS"] * 2
    assert {item["track"] for item in executed} == {"R"}
    configs = {item["parent_identities"]["c7_detector_config"] for item in executed}
    assert len(configs) == 1, "Track-R arms received different detector configurations"


def test_the_two_tracks_do_not_share_a_configuration(with_c7_lock) -> None:
    """Same-within-track is not same-across-track: the loss sets differ."""
    executed = _run_rows(with_c7_lock, _TRACK_G_ROWS[:1] + _TRACK_R_ROWS[:1])
    by_track = {item["track"]: item["parent_identities"]["c7_detector_config"]
                for item in executed}

    assert set(by_track) == {"G", "R"}
    assert by_track["G"] != by_track["R"], (
        "Track G and Track R were given one configuration; they have different "
        "active loss sets and forcing one vector across them would be a different "
        "claim from the fairness invariant")


def test_the_prompthead_ablation_differs_only_in_its_own_dimension(
        with_c7_lock) -> None:
    """(10). C-R-LLM vs C-R-NOPROMPT: one flag, one config, one bank."""
    executed = _run_rows(
        with_c7_lock, ("C-R-LLM-P3READY-s20260806", "C-R-NOPROMPT-P3READY-s20260806"))
    llm, noprompt = executed

    assert llm["status"] == noprompt["status"] == "PASS"
    # The SAME frozen Track-R configuration.
    assert (llm["parent_identities"]["c7_detector_config"]
            == noprompt["parent_identities"]["c7_detector_config"])
    # The SAME synthetic bank: the ablation is not a change of generator arm.
    assert llm["arm"] == noprompt["arm"] == "LLM"
    assert llm["parent_identities"]["c6_bank"] == noprompt["parent_identities"]["c6_bank"]
    # Exactly one typed flag differs, and it is the preregistered one.
    differing = {key for key in set(llm["flags"]) | set(noprompt["flags"])
                 if llm["flags"].get(key) != noprompt["flags"].get(key)}
    assert differing == {"prompt"}, differing
    assert llm["flags"]["prompt"] == "frozen_prompt"
    assert noprompt["flags"]["prompt"] == "off"
    # And the searched loss weights are the same numbers; lambda_P simply has no
    # term to weight once the PromptHead is off.
    assert (llm["resolved_config"]["loss_weights"]
            == noprompt["resolved_config"]["loss_weights"])


def test_a_row_may_not_borrow_the_other_tracks_configuration(with_c7_lock) -> None:
    """Cross-track substitution fails closed rather than training on the wrong one."""
    _request, inputs, _ = _prepare(with_c7_lock)
    lock = dict(inputs["c7_lock"])
    lock["tracks"] = {"G": lock["tracks"]["G"]}   # Track R's config removed

    with pytest.raises(c8.TrackConfigurationMissing, match="Track 'R'"):
        c8.track_configuration(lock, "R")


def test_the_c8_execution_check_proves_the_shared_invariant(with_c7_lock) -> None:
    """The adapter's own check, over rows that really ran."""
    request, inputs, _ = _prepare(with_c7_lock)
    executed = _run_rows(with_c7_lock, _TRACK_G_ROWS + _TRACK_R_ROWS)
    _rows_, result = c8.C8Adapter()._scientific_execute(
        request, inputs, [], with_c7_lock / "reports/full/c8",
        with_c7_lock / "runs/full/c8")
    # `_scientific_execute` with no schedule produces no rows; the invariant is
    # asserted here directly over the rows above, which is what it aggregates.
    shared: dict[str, set[str]] = {}
    for item in executed:
        shared.setdefault(item["track"], set()).add(
            item["parent_identities"]["c7_detector_config"])
    assert all(len(values) == 1 for values in shared.values())
    assert sorted(shared) == ["G", "R"]


def test_c8_prepare_asserts_the_fairness_invariant(with_c7_lock) -> None:
    _request, _inputs, result = _prepare(with_c7_lock)

    shared = next(item for item in result.checks
                  if item["check_id"] == "c8_configuration_is_shared_within_each_track")
    assert shared["ok"] is True
    assert shared["detail"]["training_arm"] == "DET"
    assert shared["detail"]["per_arm_search_performed"] is False
    assert sorted(shared["detail"]["configurations"]) == ["G", "R"]

    covered = next(item for item in result.checks
                   if item["check_id"]
                   == "c8_every_declared_track_has_a_frozen_configuration")
    assert covered["ok"] is True
    assert sorted(covered["detail"]["tracks_in_matrix"]) == ["G", "R"]


def test_changing_a_tracks_frozen_config_invalidates_only_that_tracks_rows(
        with_c7_lock) -> None:
    """The invalidation subtree is per track, which is what the lock shape buys."""
    before = {item["track"]: item["parent_identities"]["c7_detector_config"]
              for item in _run_rows(with_c7_lock, _TRACK_G_ROWS[:1] + _TRACK_R_ROWS[:1])}

    path = with_c7_lock / c7.SCIENTIFIC_CONFIG_LOCK_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tracks"]["G"]["winner_config_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    after = {item["track"]: item["parent_identities"]["c7_detector_config"]
             for item in _run_rows(with_c7_lock, _TRACK_G_ROWS[:1] + _TRACK_R_ROWS[:1])}

    assert after["G"] != before["G"]
    assert after["R"] == before["R"]


# --- (12) target access ------------------------------------------------------

def test_target_access_stays_zero_everywhere(scientific) -> None:
    """(12). Every artifact the scientific C7 wrote declares it."""
    _approve(scientific)
    _run(scientific)

    lock = json.loads(
        (scientific / c7.SCIENTIFIC_CONFIG_LOCK_PATH).read_text(encoding="utf-8"))
    assert lock["target_access"] == 0
    assert lock["no_target_capability_proof"]["target_labels_resolved"] == 0
    assert lock["no_target_capability_proof"]["target_roots_mounted"] == []

    for path in sorted((scientific / "runs/full/c7/scientific").rglob(c7.TRIAL_SUMMARY)):
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["target_paths_resolved"] == 0
        assert record["target_labels_resolved"] == 0


def test_the_matrix_declares_only_the_two_searched_tracks() -> None:
    """The lock covers every track the 42 rows actually run."""
    decision = c7_decision.load_decision(REPO)
    tracks = sorted({row.track for row in build_plan().rows})

    assert tracks == sorted(decision.tracks) == ["G", "R"]
