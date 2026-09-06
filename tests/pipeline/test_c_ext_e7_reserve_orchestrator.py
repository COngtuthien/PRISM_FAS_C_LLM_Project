"""Tests for `prism_fas.evaluation.c_ext_e7_reserve_orchestrator` (E7-v1.1
RESERVE AMENDMENT orchestration). Every test builds a self-contained fake
repo under `tmp_path`. This module never renders for real on this laptop
(the GPU-boundary/mask-compatibility/evaluation seams are the SAME ones
`test_c_ext_e7_gpat_bank.py` already mocks for v1.0 candidate generation);
tests that exercise `open_and_close_tranche` reuse that exact mocking
philosophy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_c_ext_e7_gpat_bank as fx  # noqa: E402

from prism_fas.evaluation import c_ext_e7_reserve_orchestrator as orch  # noqa: E402
from prism_fas.evaluation import c_ext_e7_reserve_schedule as rs  # noqa: E402
from prism_fas.synthesis import c5_source_pair_plan as spp  # noqa: E402

REPO = fx.REPO


def _accept_shortfall_then_reserve_closes(row) -> bool:
    """v1.0's own GPAT candidates: only the first 100 accepted (far short of
    512, forcing BLOCKED_PROTOCOL_GAP_RESERVE_SCHEDULE). Every reserve GPAT
    candidate: accepted unconditionally (256/tranche), so tranche 1 alone
    (100 + 256 = 356) still falls short, and tranche 2 (100 + 512 = 612)
    closes the fold -- a clean, deterministic 2-tranche demonstration.
    Physics is accepted unconditionally throughout (floor trivially met by
    v1.0's own 1024 alone)."""
    if row["route"] == spp.PHYSICS:
        return True
    return row["position"] < 100 or row["position"] >= rs.RESERVE_POSITION_BASE


def _build_orchestrator_ready_fixture(tmp_path: Path, monkeypatch, fold_id: str):
    """A fold with a REAL v1.0 BLOCKED_PROTOCOL_GAP_RESERVE_SCHEDULE
    terminal state (built via the REAL, unmocked `generate_and_match`, with
    only its GPU/model boundaries mocked -- exactly the existing v1.0 test
    fixture's own pattern) plus the global, fold-independent amendment
    protocol lock. Returns (repo, tracking)."""
    repo = fx._build_generation_ready_fixture(tmp_path, monkeypatch, fold_id)
    tracking = fx._patch_generation_boundary(monkeypatch, repo, accept=_accept_shortfall_then_reserve_closes,
                                             skip=fx._skip_none)
    result = fx.e7g.generate_and_match(repo, fold_id, authorize=True)
    assert result["status"] == fx.e7g.BLOCKED_PROTOCOL_GAP_RESERVE_SCHEDULE
    orch.rs.write_reserve_protocol_lock(repo)
    return repo, tracking


# --- A. v1.0 initial 2048 IDs untouched --------------------------------------

def test_a_v1_0_ids_untouched_after_schedule_materialization(tmp_path, monkeypatch):
    repo, _ = _build_orchestrator_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    before = rs.rebuild_v1_0_pool_for_invariance_check(
        repo, "EXT-F1", gpat_checkpoint_sha256=json.loads(
            fx.e7g.gpat_fit_lock_path(repo, "EXT-F1").read_text())["best_checkpoint_sha256"])
    orch.write_reserve_schedule_lock(repo, "EXT-F1")
    after = rs.rebuild_v1_0_pool_for_invariance_check(
        repo, "EXT-F1", gpat_checkpoint_sha256=json.loads(
            fx.e7g.gpat_fit_lock_path(repo, "EXT-F1").read_text())["best_checkpoint_sha256"])
    assert before == after
    invariance = orch.verify_v1_0_invariance(repo, "EXT-F1")
    assert invariance["invariant"] is True


# --- B. first reserve candidate is exactly position 2048 --------------------

def test_b_first_reserve_position_is_2048(tmp_path, monkeypatch):
    repo, _ = _build_orchestrator_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    orch.write_reserve_schedule_lock(repo, "EXT-F1")
    package_root = repo / fx.e7g.GPAT_INPUT_ROOT / "EXT-F1"
    base_plan = spp.build_source_pair_plan(package_root)
    reserve_base = rs.build_reserve_base_schedule(repo, "EXT-F1", base_plan=base_plan)
    assert min(row["position"] for row in reserve_base["rows"]) == 2048 == rs.RESERVE_POSITION_BASE


# --- C. tranche boundaries ----------------------------------------------------

def test_c_tranche_boundaries_exact(tmp_path, monkeypatch):
    repo, _ = _build_orchestrator_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    package_root = repo / fx.e7g.GPAT_INPUT_ROOT / "EXT-F1"
    base_plan = spp.build_source_pair_plan(package_root)
    reserve_base = rs.build_reserve_base_schedule(repo, "EXT-F1", base_plan=base_plan)
    expected_ranges = {1: (2048, 2559), 2: (2560, 3071), 3: (3072, 3583), 4: (3584, 4095)}
    for tranche, (lo, hi) in expected_ranges.items():
        positions = [row["position"] for row in reserve_base["rows"] if row["reserve_tranche"] == tranche]
        assert min(positions) == lo
        assert max(positions) == hi
        assert len(positions) == 512


# --- D. exactly +256 Physics +256 GPAT per arm/tranche -----------------------

def test_d_exact_256_256_per_arm_tranche(tmp_path, monkeypatch):
    repo, _ = _build_orchestrator_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    lock = orch.write_reserve_schedule_lock(repo, "EXT-F1")
    package_root = repo / fx.e7g.GPAT_INPUT_ROOT / "EXT-F1"
    base_plan = spp.build_source_pair_plan(package_root)
    reserve_base = rs.build_reserve_base_schedule(repo, "EXT-F1", base_plan=base_plan)
    for tranche in range(1, 5):
        rows = [row for row in reserve_base["rows"] if row["reserve_tranche"] == tranche]
        assert sum(1 for row in rows if row["route"] == spp.PHYSICS) == 256
        assert sum(1 for row in rows if row["route"] == spp.GPAT) == 256


# --- E. all three core arms required -----------------------------------------

def test_e_all_three_arms_required_structurally():
    import inspect

    assert "arm" not in inspect.signature(orch.open_and_close_tranche).parameters
    assert "route" not in inspect.signature(orch.open_and_close_tranche).parameters


# --- F. partial arm or partial route execution rejected ----------------------

def test_f_reserve_schedule_lock_always_covers_all_three_arms(tmp_path, monkeypatch):
    repo, _ = _build_orchestrator_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    lock = orch.write_reserve_schedule_lock(repo, "EXT-F1")
    assert set(lock["arm_schedule_identities"]) == set(spp.ARMS)
    assert set(lock["candidate_ids_by_arm_and_tranche"]) == set(spp.ARMS)


def test_f_condition_check_fails_closed_without_schedule(tmp_path, monkeypatch):
    repo, _ = _build_orchestrator_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    with pytest.raises(orch.E7ReserveOrchestratorError, match="conditions not met"):
        orch.assert_tranche_open_conditions(repo, "EXT-F1")


# --- G/H. resume / ordering ---------------------------------------------------

def test_gh_tranche_1_then_2_full_lifecycle_and_ordering_enforced(tmp_path, monkeypatch):
    repo, tracking = _build_orchestrator_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    orch.write_reserve_schedule_lock(repo, "EXT-F1")

    # H: tranche 2 cannot open before tranche 1.
    with pytest.raises(orch.E7ReserveOrchestratorError, match="next allowed tranche is 1"):
        orch.open_and_close_tranche(repo, "EXT-F1", 2, authorize=True)

    first = orch.open_and_close_tranche(repo, "EXT-F1", 1, authorize=True)
    assert first["status"] == orch.STATUS_NEEDS_NEXT_TRANCHE
    state = orch.read_run_state(repo, "EXT-F1")
    assert state["next_tranche"] == 2

    # H again: tranche 1 cannot be reopened once tranche 2 is next.
    with pytest.raises(orch.E7ReserveOrchestratorError, match="next allowed tranche is 2"):
        orch.open_and_close_tranche(repo, "EXT-F1", 1, authorize=True)

    second = orch.open_and_close_tranche(repo, "EXT-F1", 2, authorize=True)
    assert second["status"] == orch.STATUS_CLOSED_MATCHED
    assert orch.read_run_state(repo, "EXT-F1")["status"] == orch.STATUS_CLOSED_MATCHED

    # L: closed at tranche 2 -- no further tranche may open.
    with pytest.raises(orch.E7ReserveOrchestratorError, match="already closed"):
        orch.open_and_close_tranche(repo, "EXT-F1", 3, authorize=True)


def test_g_tranche_lock_written_before_rendering_and_ids_stable(tmp_path, monkeypatch):
    repo, tracking = _build_orchestrator_ready_fixture(tmp_path, monkeypatch, "EXT-F2")
    orch.write_reserve_schedule_lock(repo, "EXT-F2")

    import prism_fas.synthesis.c5_render as c5_render_mod

    inner = c5_render_mod.render_arm
    seen = {"checked": False}

    def _checking_render_arm(**kwargs):
        lock_path = orch.reserve_tranche_lock_path(repo, "EXT-F2", 1)
        assert lock_path.is_file(), "tranche lock must exist before the first render call"
        seen["checked"] = True
        return inner(**kwargs)

    monkeypatch.setattr("prism_fas.synthesis.c5_render.render_arm", _checking_render_arm)
    orch.open_and_close_tranche(repo, "EXT-F2", 1, authorize=True)
    assert seen["checked"] is True

    lock_before = json.loads(orch.reserve_tranche_lock_path(repo, "EXT-F2", 1).read_text())
    # Re-deriving the same tranche's candidate ids (idempotent schedule construction) must
    # reproduce EXACTLY what was pre-locked -- "resume" at the schedule level.
    package_root = repo / fx.e7g.GPAT_INPUT_ROOT / "EXT-F2"
    base_plan = spp.build_source_pair_plan(package_root)
    reserve_base = rs.build_reserve_base_schedule(repo, "EXT-F2", base_plan=base_plan)
    from prism_fas.synthesis import c5_arm_plan

    gpat_lock = json.loads(fx.e7g.gpat_fit_lock_path(repo, "EXT-F2").read_text())
    for arm in spp.ARMS:
        rebuilt = rs.build_reserve_arm_schedule(
            repo, "EXT-F2", arm, reserve_base_schedule=reserve_base,
            gpat_checkpoint_sha256=gpat_lock["best_checkpoint_sha256"],
            physics_engine_version="m7-physics-v1")
        rebuilt_ids = [row["candidate_id"] for row in rebuilt["rows"] if row["reserve_tranche"] == 1]
        assert rebuilt_ids == lock_before["candidate_ids"][arm]


# --- I. outcome counts cannot alter future schedule membership --------------

def test_i_outcome_counts_never_alter_schedule_membership(tmp_path, monkeypatch):
    repo, _ = _build_orchestrator_ready_fixture(tmp_path, monkeypatch, "EXT-F3")
    lock_a = orch.write_reserve_schedule_lock(repo, "EXT-F3")
    # Simulate a wildly different "observed outcome" on disk (a fabricated run-state) --
    # `materialize_real_reserve_schedule` never reads run-state at all.
    orch.write_run_state(repo, "EXT-F3", {**orch.build_initial_run_state("EXT-F3"),
                                          "status": orch.STATUS_CAP_BLOCKED})
    lock_b = orch.materialize_real_reserve_schedule(repo, "EXT-F3")
    assert lock_a["candidate_ids_by_arm_and_tranche"] == lock_b["candidate_ids_by_arm_and_tranche"]
    assert lock_a["arm_schedule_identities"] == lock_b["arm_schedule_identities"]


# --- J. quality threshold/profile identities unchanged -----------------------

def test_j_quality_profile_identities_bound_and_unmodified(tmp_path, monkeypatch):
    repo, _ = _build_orchestrator_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    binding = orch.write_reserve_protocol_binding(repo, "EXT-F1")
    from prism_fas.synthesis import gate_profiles

    assert set(binding["quality_profile_identities"]) == set(gate_profiles.PROFILE_ORDER)
    for name in gate_profiles.PROFILE_ORDER:
        assert isinstance(binding["quality_profile_identities"][name], str)
        assert len(binding["quality_profile_identities"][name]) == 64


# --- K/L. cumulative feasibility; success prevents next tranche -------------

def test_kl_cumulative_feasibility_closes_at_tranche_2_not_before(tmp_path, monkeypatch):
    repo, _ = _build_orchestrator_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    orch.write_reserve_schedule_lock(repo, "EXT-F1")
    first = orch.open_and_close_tranche(repo, "EXT-F1", 1, authorize=True)
    assert first["status"] == orch.STATUS_NEEDS_NEXT_TRANCHE  # K: 100+256=356 < 512, GPAT still short
    second = orch.open_and_close_tranche(repo, "EXT-F1", 2, authorize=True)
    assert second["status"] == orch.STATUS_CLOSED_MATCHED  # K: 100+512=612 >= 512, now feasible
    for arm in spp.ARMS:
        bank = second["closure"]["banks"][arm]
        assert bank["final_bank_size"] == 1024
        assert bank["by_route"] == {"physics": 512, "gpat": 512}
    final = json.loads(orch.reserve_final_closure_path(repo, "EXT-F1").read_text())
    assert final["closed_at_tranche"] == 2
    # L: tranche 3 is refused now that the fold is closed.
    with pytest.raises(orch.E7ReserveOrchestratorError, match="already closed"):
        orch.open_and_close_tranche(repo, "EXT-F1", 3, authorize=True)


# --- M. tranche 4 infeasibility -> terminal cap-block status -----------------

def test_m_tranche_4_infeasibility_produces_cap_block(tmp_path, monkeypatch):
    repo, _ = _build_orchestrator_ready_fixture(tmp_path, monkeypatch, "EXT-F2")

    def _never_enough_gpat(row):
        if row["route"] == spp.PHYSICS:
            return True
        return False  # every GPAT candidate, v1.0 and reserve alike, rejected -- never reaches 512

    fx._patch_generation_boundary(monkeypatch, repo, accept=_never_enough_gpat, skip=fx._skip_none)
    orch.write_reserve_schedule_lock(repo, "EXT-F2")
    for tranche in (1, 2, 3):
        result = orch.open_and_close_tranche(repo, "EXT-F2", tranche, authorize=True)
        assert result["status"] == orch.STATUS_NEEDS_NEXT_TRANCHE
    final_tranche_result = orch.open_and_close_tranche(repo, "EXT-F2", 4, authorize=True)
    assert final_tranche_result["status"] == orch.STATUS_CAP_BLOCKED
    final = json.loads(orch.reserve_final_closure_path(repo, "EXT-F2").read_text())
    assert final["final_status"] == "SCIENTIFICALLY_BLOCKED_AFTER_RESERVE_CAP"
    with pytest.raises(orch.E7ReserveOrchestratorError, match="already closed"):
        orch.open_and_close_tranche(repo, "EXT-F2", 4, authorize=True)


# --- N/O. firewall / LLM ------------------------------------------------------

def test_no_target_domain_and_zero_llm_calls(tmp_path, monkeypatch):
    repo, _ = _build_orchestrator_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    orch.write_reserve_schedule_lock(repo, "EXT-F1")
    firewall = orch.validate_source_domain_firewall(repo, "EXT-F1")
    assert firewall["status"] == "VALID"
    assert firewall["target_access"] is False
    assert firewall["llm_api_calls"] == 0
    result = orch.open_and_close_tranche(repo, "EXT-F1", 1, authorize=True)
    assert result["closure"]["target_access"] is False
    assert result["closure"]["llm_api_calls"] == 0


# --- P. historical v1.0 closure files byte-identical -------------------------

def test_p_v1_0_closure_files_byte_unchanged_after_tranche(tmp_path, monkeypatch):
    repo, _ = _build_orchestrator_ready_fixture(tmp_path, monkeypatch, "EXT-F3")
    before = {condition: fx.e7g.generation_closure_path(repo, "EXT-F3", condition).read_bytes()
             for condition in ("G-RND", "G-DET", "G-LLM")}
    orch.write_reserve_schedule_lock(repo, "EXT-F3")
    orch.open_and_close_tranche(repo, "EXT-F3", 1, authorize=True)
    after = {condition: fx.e7g.generation_closure_path(repo, "EXT-F3", condition).read_bytes()
            for condition in ("G-RND", "G-DET", "G-LLM")}
    assert before == after
    for condition in ("G-RND", "G-DET", "G-LLM"):
        assert not fx.e7g.bank_lock_path(repo, "EXT-F3", condition).is_file()


# --- Q. F1 historical Shuffle block unchanged --------------------------------

def test_q_f1_shuffle_block_unchanged_by_orchestrator(tmp_path, monkeypatch):
    repo, _ = _build_orchestrator_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    before = fx.e7g.resolved_shuffle_status(repo, "EXT-F1")
    assert before == "BLOCKED_TRUE_FROZEN_MATCHED_BANK_INFEASIBILITY"
    orch.write_reserve_schedule_lock(repo, "EXT-F1")
    orch.open_and_close_tranche(repo, "EXT-F1", 1, authorize=True)
    after = fx.e7g.resolved_shuffle_status(repo, "EXT-F1")
    assert after == before
    assert not fx.e7g.bank_lock_path(repo, "EXT-F1", fx.e7g.SHUFFLE_CONDITION).is_file()
    for name in dir(orch):
        assert "shuffle" not in name.lower()


# --- R. dry-run performs zero candidate rendering ----------------------------

def test_r_dry_run_preflight_zero_rendering(tmp_path, monkeypatch):
    repo, tracking = _build_orchestrator_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    orch.write_reserve_schedule_lock(repo, "EXT-F1")
    calls_before = len(tracking["render_arm_calls"])
    report = orch.dry_run_reserve_preflight(repo, "EXT-F1")
    assert report["rendering_performed"] is False
    assert len(tracking["render_arm_calls"]) == calls_before  # zero new render_arm calls
    assert report["8_9_authorization"]["authorized"] is True
    assert report["6_tranche_counts"]["next_tranche"] == 1


def test_r_preflight_reports_next_tranche_per_fold(tmp_path, monkeypatch):
    repo, _ = _build_orchestrator_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    assert orch.next_allowed_tranche(repo, "EXT-F1") == 1
    orch.write_reserve_schedule_lock(repo, "EXT-F1")
    orch.open_and_close_tranche(repo, "EXT-F1", 1, authorize=True)
    assert orch.next_allowed_tranche(repo, "EXT-F1") == 2


# --- protocol-condition / lock content proofs -------------------------------

def test_check_conditions_reports_all_eight(tmp_path, monkeypatch):
    repo, _ = _build_orchestrator_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    orch.write_reserve_schedule_lock(repo, "EXT-F1")
    report = orch.check_tranche_open_conditions(repo, "EXT-F1")
    for key in ("1_v1_0_terminal_status", "2_gpat_fit_lock_valid", "3_quality_calibration_valid",
               "4_amendment_protocol_lock_valid", "5_reserve_schedule_materialized",
               "6_v1_0_invariance_verified", "7_no_target_domain_access", "8_synchronized_arms"):
        assert key in report
        assert report[key]["satisfied"] if "satisfied" in report[key] else True
    assert_ = orch.assert_tranche_open_conditions(repo, "EXT-F1")
    assert assert_ == report


def test_disjointness_report_all_true(tmp_path, monkeypatch):
    repo, _ = _build_orchestrator_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    orch.write_reserve_schedule_lock(repo, "EXT-F1")
    report = orch.candidate_id_disjointness_report(repo, "EXT-F1")
    assert report["all_disjoint"] is True
    for arm_report in report["by_arm"].values():
        assert arm_report["disjoint_from_v1_0"] is True
        assert arm_report["disjoint_across_tranches"] is True
        assert arm_report["total_reserve_candidates"] == 2048  # 4 tranches * 512


def test_frozen_c5_c6_gpat_and_v1_0_module_unchanged():
    import subprocess

    for relative in (
        "src/prism_fas/synthesis/c5_source_pair_plan.py", "src/prism_fas/synthesis/c5_arm_plan.py",
        "src/prism_fas/synthesis/c5_raw_generation.py", "src/prism_fas/synthesis/c5_render.py",
        "src/prism_fas/synthesis/synthetic_bank.py", "src/prism_fas/synthesis/quality_gate.py",
        "src/prism_fas/synthesis/quality_calibration.py", "src/prism_fas/synthesis/c6_scientific.py",
        "src/prism_fas/synthesis/c6_matched_bank.py", "src/prism_fas/synthesis/gpat_trainer.py",
        "src/prism_fas/synthesis/gpat_model.py", "src/prism_fas/synthesis/gpat_losses.py",
        "src/prism_fas/synthesis/m8_pipeline.py", "src/prism_fas/synthesis/masks.py",
        "src/prism_fas/synthesis/pair_plan.py", "src/prism_fas/synthesis/gate_profiles.py",
        "src/prism_fas/evaluation/c_ext_e7_gpat_bank.py",
        "src/prism_fas/evaluation/c_ext_e7_reserve_schedule.py",
    ):
        committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=REPO, check=True,
                                   capture_output=True, text=True).stdout
        on_disk = (REPO / relative).read_text(encoding="utf-8")
        assert committed == on_disk, f"{relative} differs from HEAD"
