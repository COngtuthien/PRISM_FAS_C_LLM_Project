"""Backend portability, the C6 gate profiles, the C8 matrix and the C9 freeze.

Four modules, one shared concern: each of them is a place where an engineering
convenience could quietly become a scientific change. A batch shrunk to fit a
card, a gate relaxed for one arm, a seed added after seeing a result, a lock
built over incomplete evidence. Every test below is aimed at one of those.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from prism_fas.evaluation.source_lock import RowEvidence, SourceLockError
from prism_fas.evaluation.source_lock import audit as lock_audit
from prism_fas.evaluation.source_lock import build as build_lock
from prism_fas.evaluation.source_lock import validate as validate_lock
from prism_fas.evaluation.source_matrix import (ARMS, SEED_FAMILY, SourceMatrixError,
                                                SourceRow, build_plan)
from prism_fas.pipeline import portability
from prism_fas.synthesis.gate_profiles import (ARMS as GATE_ARMS, GPAT_PER_ARM,
                                               PHYSICS_PER_ARM, PROFILE_ORDER, RANGE_SAFE,
                                               ArmFeasibility, GateProfileError,
                                               build_profiles, derive_profile,
                                               matched_bank_plan, select_profile)

NOMINAL = {"tau_fd": 0.50, "tau_id": 0.80, "tau_lm": 0.08,
           "tau_parse": 0.90, "tau_out": 0.0, "tau_fp": 0.50}


# --- portability -------------------------------------------------------------

def test_the_frozen_batch_composition_comes_from_its_canonical_owner() -> None:
    assert portability.frozen_composition() == {"real_live": 12, "real_spoof": 12,
                                                "synthetic": 8}


def test_a_microbatch_always_reconstitutes_the_effective_batch() -> None:
    for vram in (None, 6.0, 12.0, 24.0, 80.0):
        backend = portability.BackendProfile("probe", device="cuda", vram_gb=vram)
        plan = portability.resolve_microbatch(backend=backend)
        assert plan.microbatch * plan.accumulation_steps == 32
        assert plan.preserves_effective_batch


def test_a_backend_too_small_for_any_divisor_is_refused_not_shrunk() -> None:
    """L.12: a scientific budget is never reduced to fit the hardware."""
    backend = portability.BackendProfile("tiny", device="cuda", vram_gb=24.0)
    with pytest.raises(portability.PortabilityError, match="not reduced to fit"):
        portability.resolve_microbatch(backend=backend, effective_batch=32,
                                       max_microbatch=0)


def test_a_plan_that_does_not_divide_the_batch_cannot_be_constructed() -> None:
    with pytest.raises(portability.PortabilityError, match="never change the batch"):
        portability.MicrobatchPlan(effective_batch=32, composition={}, microbatch=5,
                                   accumulation_steps=6, backend="x")


def test_a_drifted_composition_is_refused() -> None:
    backend = portability.KNOWN_BACKENDS["local_cpu"]
    plan = portability.resolve_microbatch(backend=backend, effective_batch=32,
                                          composition={"real_live": 16, "real_spoof": 16,
                                                       "synthetic": 0})
    with pytest.raises(portability.PortabilityError, match="frozen composition"):
        portability.assert_composition_preserved(plan)


def test_a_scientific_identity_does_not_move_with_the_backend() -> None:
    payload = {"config_hash": "abc", "seed": 20260806, "notes": ["x"]}
    report = portability.identity_is_backend_invariant(
        payload, portability.KNOWN_BACKENDS.values())
    assert report["invariant"]
    assert report["distinct_identity_count"] == 1
    assert report["baseline_identity"] == next(iter(report["identities"].values()))


def test_operational_fields_are_stripped_recursively() -> None:
    payload = {"seed": 7, "run": {"device": "cuda", "gpu_uuid": "GPU-x", "steps": 10}}
    stripped = portability.scientific_identity_material(payload)
    assert stripped == {"seed": 7, "run": {"steps": 10}}


def test_a_checkpoint_carrying_a_machine_path_is_not_portable() -> None:
    report = portability.checkpoint_portability_audit({
        "weights": r"D:\AI on IOT\weights.pt",
        "mount": "/modal/vol/bank",
        "device": "GPU-1a2b3c4d-0000-0000-0000-000000000000"})
    assert not report["portable"]
    assert {finding["key_path"] for finding in report["findings"]} == {
        "weights", "mount", "device"}


def test_recorded_provenance_is_not_a_portability_defect() -> None:
    """Recording where a run happened is provenance; needing it is the defect."""
    report = portability.checkpoint_portability_audit({
        "operational_provenance": {"recorded": r"D:\AI on IOT\somewhere"},
        "config_identity": "abc"})
    assert report["portable"], report["findings"]


def test_a_path_under_no_declared_root_cannot_be_made_portable() -> None:
    with pytest.raises(portability.PortabilityError, match="none of the declared roots"):
        portability.portable_path("/elsewhere/x.pt", {"project": "/project"})


def test_the_longest_matching_root_wins() -> None:
    logical = portability.portable_path(
        "/project/data/packages/x.parquet",
        {"project": "/project", "packages": "/project/data/packages"})
    assert logical.root == "packages"
    assert logical.relative == "x.parquet"


# --- C6 gate profiles --------------------------------------------------------

def test_the_derivation_formula_is_the_one_section_11_4_declares() -> None:
    strict = derive_profile(NOMINAL, "STRICT")
    permissive = derive_profile(NOMINAL, "PERMISSIVE")
    # higher-is-better lower bound a: STRICT = a + 0.10 * (1 - a)
    assert strict["tau_fd"] == pytest.approx(0.50 + 0.10 * 0.50)
    assert permissive["tau_fd"] == pytest.approx(0.90 * 0.50)
    # nonnegative lower-is-better upper bound a: STRICT = 0.90a, PERMISSIVE = 1.10a
    assert strict["tau_lm"] == pytest.approx(0.90 * 0.08)
    assert permissive["tau_lm"] == pytest.approx(1.10 * 0.08)


def test_a_range_safe_threshold_is_identical_in_every_profile() -> None:
    profiles = build_profiles(NOMINAL)
    for name in RANGE_SAFE:
        assert len({profiles[profile].thresholds[name] for profile in PROFILE_ORDER}) == 1


def test_a_threshold_with_no_declared_direction_is_refused() -> None:
    with pytest.raises(GateProfileError, match="no declared direction"):
        derive_profile({**NOMINAL, "tau_mystery": 0.5}, "STRICT")


def test_the_strictest_qualifying_profile_is_selected() -> None:
    profiles = build_profiles(NOMINAL)
    feasibility = {
        "STRICT": [ArmFeasibility(arm, 2048, PHYSICS_PER_ARM if arm != "RND" else 10,
                                  GPAT_PER_ARM) for arm in GATE_ARMS],
        "NOMINAL": [ArmFeasibility(arm, 2048, PHYSICS_PER_ARM, GPAT_PER_ARM)
                    for arm in GATE_ARMS],
        "PERMISSIVE": [ArmFeasibility(arm, 2048, PHYSICS_PER_ARM, GPAT_PER_ARM)
                       for arm in GATE_ARMS]}
    selection = select_profile(profiles, feasibility)
    assert selection.selected == "NOMINAL"
    assert not selection.failed


def test_one_short_arm_disqualifies_a_profile_for_every_arm() -> None:
    """§11.4: the gate is COMMON; a strong arm cannot carry a weak one."""
    profiles = build_profiles(NOMINAL)
    short = {name: [ArmFeasibility(arm, 2048,
                                   PHYSICS_PER_ARM if arm != "LLM" else 1,
                                   GPAT_PER_ARM) for arm in GATE_ARMS]
             for name in PROFILE_ORDER}
    selection = select_profile(profiles, short)
    assert selection.failed
    assert "C6 FAILS" in selection.failure_reason


def test_reaching_the_total_does_not_satisfy_the_route_split() -> None:
    item = ArmFeasibility("LLM", 2048, accepted_physics=900, accepted_gpat=124)
    assert item.accepted_total == 1024
    assert not item.feasible


def test_a_matched_plan_is_matched_only_when_every_arm_is() -> None:
    good = {arm: ArmFeasibility(arm, 2048, PHYSICS_PER_ARM, GPAT_PER_ARM)
            for arm in GATE_ARMS}
    assert matched_bank_plan(good)["matched"]
    bad = {**good, "LLM": ArmFeasibility("LLM", 2048, 10, GPAT_PER_ARM)}
    assert not matched_bank_plan(bad)["matched"]


def test_the_arm_identity_is_not_an_input_to_the_derivation() -> None:
    """The strongest form of 'common': the function cannot see the arm."""
    import inspect

    signature = inspect.signature(derive_profile)
    assert "arm" not in signature.parameters


# --- C8 source matrix --------------------------------------------------------

def test_the_matrix_satisfies_the_replication_policy() -> None:
    report = build_plan().validate()
    assert report["valid"], report["problems"]
    for arm in ARMS:
        assert report["seed_counts"][f"C-G-{arm}:P3"] == 5
        assert report["seed_counts"][f"C-G-{arm}:P1"] == 3
        assert report["seed_counts"][f"C-G-{arm}:P2"] == 3
    for experiment in ("C-R-DET", "C-R-LLM", "C-R-NOPROMPT"):
        assert report["seed_counts"][f"{experiment}:P3"] == 3


def test_a_seed_outside_the_family_cannot_be_planned() -> None:
    """§18.3: the family is not extended after a result is seen."""
    with pytest.raises(SourceMatrixError, match="outside the fixed family"):
        SourceRow(row_id="x", experiment_id="C-G-LLM", track="G", arm="LLM",
                  protocol="P3", seed=19700101, replication_role="diagnostic",
                  hypotheses=(), flags={}, selection_tuple=())


def test_rows_that_differ_only_by_seed_share_one_configuration_identity() -> None:
    plan = build_plan()
    by_config: dict[str, set[int]] = {}
    for row in plan.rows:
        by_config.setdefault(row.config_identity, set()).add(row.seed)
    assert any(len(seeds) == 5 for seeds in by_config.values())
    assert len(by_config) < len(plan.rows)


def test_the_matrix_identity_is_stable_and_excludes_the_clock() -> None:
    assert build_plan().identity == build_plan().identity


def test_no_matrix_row_resolves_a_target_label_or_metric() -> None:
    plan = build_plan()
    for row in plan.rows:
        assert "label" not in str(row.flags).lower()
        for field in row.selection_tuple:
            assert "siw" not in field.lower()


# --- C9 source freeze --------------------------------------------------------

def complete_evidence(plan) -> list[RowEvidence]:
    return [RowEvidence(row_id=row.row_id, run_identity=row.run_identity,
                        config_identity=row.config_identity, status="PASS",
                        checkpoint_sha256="ckpt", calibration_sha256="cal",
                        calibration_hash="calhash") for row in plan.rows]


def test_a_complete_matrix_freezes_and_the_lock_reproduces() -> None:
    plan = build_plan()
    lock = build_lock(plan, complete_evidence(plan))
    report = validate_lock(lock, plan, complete_evidence(plan))
    assert report["valid"], report["problems"]
    assert report["identity_reproduces"]
    assert lock["row_count"] == len(plan.rows)
    assert lock["immutability"]["rewrite_permitted"] is False


@pytest.mark.parametrize("mutate,expected", [
    (lambda rows: rows[:-1], "required_row_missing"),
    (lambda rows: [*rows[:-1], replace(rows[-1], status="FAIL")], "row_failed"),
    (lambda rows: [*rows[:-1], replace(rows[-1], status="RUNNING")], "row_not_terminal"),
    (lambda rows: [*rows[:-1], replace(rows[-1], run_identity="0" * 64)],
     "run_identity_mismatch"),
    (lambda rows: [*rows[:-1], replace(rows[-1], checkpoint_sha256=None)],
     "checkpoint_missing"),
    (lambda rows: [*rows[:-1], replace(rows[-1], calibration_hash=None)],
     "calibration_missing"),
])
def test_the_freeze_refuses_incomplete_evidence(mutate, expected) -> None:
    plan = build_plan()
    evidence = mutate(complete_evidence(plan))
    assert lock_audit(plan, evidence)["refusals"][expected]
    with pytest.raises(SourceLockError):
        build_lock(plan, evidence)


def test_a_hidden_row_the_plan_never_declared_blocks_the_freeze() -> None:
    """§C9's 'zero failed hidden rows': an unplanned run cannot be locked in."""
    plan = build_plan()
    evidence = [*complete_evidence(plan),
                RowEvidence(row_id="UNPLANNED", run_identity="a", config_identity="b",
                            status="PASS", checkpoint_sha256="c",
                            calibration_sha256="d", calibration_hash="e")]
    assert lock_audit(plan, evidence)["refusals"]["hidden_row_not_in_plan"] == ["UNPLANNED"]
    with pytest.raises(SourceLockError, match="hidden_row_not_in_plan"):
        build_lock(plan, evidence)


def test_the_refusal_is_total_rather_than_partial() -> None:
    """There is no partial lock and no force flag."""
    import inspect

    assert "force" not in inspect.signature(build_lock).parameters


def test_validation_detects_evidence_that_drifted_after_the_freeze() -> None:
    plan = build_plan()
    lock = build_lock(plan, complete_evidence(plan))
    moved = [replace(item, checkpoint_sha256="moved")
             for item in complete_evidence(plan)]
    report = validate_lock(lock, plan, moved)
    assert not report["valid"]
    assert any("drifted" in problem for problem in report["problems"])


def test_a_lock_cannot_vouch_for_itself() -> None:
    plan = build_plan()
    lock = dict(build_lock(plan, complete_evidence(plan)))
    lock["rows"] = lock["rows"][:-1]           # edit the body, keep the identity
    report = validate_lock(lock, plan)
    assert not report["valid"]
    assert any("does not hash" in problem for problem in report["problems"])
