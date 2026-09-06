"""Tests for `prism_fas.evaluation.c_ext_e7_reserve_schedule` (E7-v1.1 RESERVE
AMENDMENT). Every test builds a self-contained fake repo under `tmp_path`
unless it explicitly checks the REAL committed repo (frozen-primitive-
unchanged checks, which can only be meaningfully verified against real
bytes). This module never renders, never evaluates quality, never trains,
never calls an LLM, and never opens a target path.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_c_ext_e7_gpat_bank as fx  # noqa: E402 -- shares this module's own fixtures verbatim

from prism_fas.evaluation import c_ext_e7_reserve_schedule as rs  # noqa: E402
from prism_fas.synthesis import c5_arm_plan  # noqa: E402
from prism_fas.synthesis import c5_source_pair_plan as spp  # noqa: E402

REPO = fx.REPO


def _build_reserve_ready_fixture(tmp_path: Path, monkeypatch, fold_id: str) -> Path:
    """A fold with a real, unmocked GPAT-input package plus real frozen C3
    arm banks for RND/DET/LLM -- everything `materialize_reserve_schedule_
    for_fold` needs. No GPAT fit, no rendering, no quality calibration --
    the reserve schedule is pure position/identity arithmetic and needs
    none of those."""
    repo = fx._build_gpat_ready_fixture(tmp_path, monkeypatch, fold_id)
    result = fx.e7g.materialize_gpat_input_package(repo, fold_id, authorize=True)
    assert result["status"] == "MATERIALIZED"
    for arm in spp.ARMS:
        fx._write_c3_arm_bank(repo, arm)
    return repo


_FAKE_CHECKPOINT_SHA = "f" * 64
_FAKE_PHYSICS_VERSION = "m7-physics-v1"


def _fold_base_plan(repo: Path, fold_id: str) -> dict:
    package_root = repo / fx.e7g.GPAT_INPUT_ROOT / fold_id
    return spp.build_source_pair_plan(package_root)


# --- A. INITIAL POOL INVARIANCE ---------------------------------------------

def test_a_initial_pool_invariance_before_and_after_reserve_schedule(tmp_path, monkeypatch):
    repo = _build_reserve_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    before = rs.rebuild_v1_0_pool_for_invariance_check(
        repo, "EXT-F1", gpat_checkpoint_sha256=_FAKE_CHECKPOINT_SHA)

    base_plan = _fold_base_plan(repo, "EXT-F1")
    rs.materialize_reserve_schedule_for_fold(
        repo, "EXT-F1", base_plan=base_plan, gpat_checkpoint_sha256=_FAKE_CHECKPOINT_SHA,
        physics_engine_version=_FAKE_PHYSICS_VERSION)

    after = rs.rebuild_v1_0_pool_for_invariance_check(
        repo, "EXT-F1", gpat_checkpoint_sha256=_FAKE_CHECKPOINT_SHA)

    assert before == after
    assert after["position_count"] == spp.CANDIDATES_PER_ARM == 2048
    assert after["max_position"] == 2047
    for arm in spp.ARMS:
        assert len(after["candidate_ids_by_arm"][arm]) == 2048


def test_a_reserve_positions_never_below_2048(tmp_path, monkeypatch):
    repo = _build_reserve_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    base_plan = _fold_base_plan(repo, "EXT-F1")
    schedule = rs.materialize_reserve_schedule_for_fold(
        repo, "EXT-F1", base_plan=base_plan, gpat_checkpoint_sha256=_FAKE_CHECKPOINT_SHA,
        physics_engine_version=_FAKE_PHYSICS_VERSION)
    for arm_schedule in schedule["arms"].values():
        for row in arm_schedule["rows"]:
            assert row["position"] >= rs.RESERVE_POSITION_BASE == 2048


def test_a_reserve_candidate_ids_disjoint_from_v1_0(tmp_path, monkeypatch):
    repo = _build_reserve_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    v1_0 = rs.rebuild_v1_0_pool_for_invariance_check(
        repo, "EXT-F1", gpat_checkpoint_sha256=_FAKE_CHECKPOINT_SHA)
    base_plan = _fold_base_plan(repo, "EXT-F1")
    reserve = rs.materialize_reserve_schedule_for_fold(
        repo, "EXT-F1", base_plan=base_plan, gpat_checkpoint_sha256=_FAKE_CHECKPOINT_SHA,
        physics_engine_version=_FAKE_PHYSICS_VERSION)
    for arm in spp.ARMS:
        v1_0_ids = set(v1_0["candidate_ids_by_arm"][arm])
        reserve_ids = {row["candidate_id"] for row in reserve["arms"][arm]["rows"]}
        assert v1_0_ids.isdisjoint(reserve_ids)


# --- B. RESERVE DETERMINISM -------------------------------------------------

def test_b_rule_identity_deterministic():
    assert rs.reserve_schedule_rule_identity() == rs.reserve_schedule_rule_identity()


def test_b_reserve_schedule_byte_identical_across_two_builds(tmp_path, monkeypatch):
    repo = _build_reserve_ready_fixture(tmp_path, monkeypatch, "EXT-F2")
    base_plan = _fold_base_plan(repo, "EXT-F2")
    first = rs.materialize_reserve_schedule_for_fold(
        repo, "EXT-F2", base_plan=base_plan, gpat_checkpoint_sha256=_FAKE_CHECKPOINT_SHA,
        physics_engine_version=_FAKE_PHYSICS_VERSION)
    second = rs.materialize_reserve_schedule_for_fold(
        repo, "EXT-F2", base_plan=base_plan, gpat_checkpoint_sha256=_FAKE_CHECKPOINT_SHA,
        physics_engine_version=_FAKE_PHYSICS_VERSION)
    for arm in spp.ARMS:
        assert first["arms"][arm]["rows"] == second["arms"][arm]["rows"]
        assert first["arms"][arm]["reserve_schedule_identity"] == second["arms"][arm]["reserve_schedule_identity"]


def test_b_protocol_lock_stable_on_rewrite(tmp_path, monkeypatch):
    repo = fx._base_repo(tmp_path)
    first = rs.write_reserve_protocol_lock(repo)
    second = rs.build_reserve_protocol_lock()
    assert first == second
    validation = rs.validate_reserve_protocol_lock(repo)
    assert validation["status"] == "VALID"


# --- C. ARM-INDEPENDENT SOURCE SCHEDULE -------------------------------------

def test_c_rnd_det_llm_share_the_reserve_source_schedule(tmp_path, monkeypatch):
    repo = _build_reserve_ready_fixture(tmp_path, monkeypatch, "EXT-F3")
    base_plan = _fold_base_plan(repo, "EXT-F3")
    schedule = rs.materialize_reserve_schedule_for_fold(
        repo, "EXT-F3", base_plan=base_plan, gpat_checkpoint_sha256=_FAKE_CHECKPOINT_SHA,
        physics_engine_version=_FAKE_PHYSICS_VERSION)
    signatures = {
        arm: [(row["position"], row["route"], row["domain_relation"], row["live_target_sample_id"],
              row["spoof_source_sample_id"]) for row in schedule["arms"][arm]["rows"]]
        for arm in spp.ARMS}
    reference = signatures["RND"]
    assert signatures["DET"] == reference
    assert signatures["LLM"] == reference
    # ...but recipe content/candidate identity differ per arm, because each arm's own frozen C3
    # bank differs.
    recipe_ids = {arm: [row["recipe_id"] for row in schedule["arms"][arm]["rows"]] for arm in spp.ARMS}
    assert recipe_ids["RND"] != recipe_ids["DET"]
    candidate_ids = {arm: {row["candidate_id"] for row in schedule["arms"][arm]["rows"]} for arm in spp.ARMS}
    assert candidate_ids["RND"].isdisjoint(candidate_ids["DET"])
    assert candidate_ids["RND"].isdisjoint(candidate_ids["LLM"])


def test_c_fairness_violation_detected_not_merely_assumed(tmp_path, monkeypatch):
    repo = _build_reserve_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    base_plan = _fold_base_plan(repo, "EXT-F1")
    reserve_base = rs.build_reserve_base_schedule(repo, "EXT-F1", base_plan=base_plan)
    good = rs.build_reserve_arm_schedule(repo, "EXT-F1", "RND", reserve_base_schedule=reserve_base,
                                         gpat_checkpoint_sha256=_FAKE_CHECKPOINT_SHA,
                                         physics_engine_version=_FAKE_PHYSICS_VERSION)
    corrupted = json.loads(json.dumps(good))
    corrupted["rows"][0]["position"] = corrupted["rows"][0]["position"] + 1
    with pytest.raises(rs.E7ReserveScheduleError, match="differs from"):
        rs.assert_reserve_arms_share_the_schedule({"RND": corrupted, "DET": good, "LLM": good})


# --- D. FOLD FIREWALL --------------------------------------------------------

def test_d_reserve_rows_never_carry_the_target_dataset(tmp_path, monkeypatch):
    for fold_id, target_domain in fx.e7g.FOLD_TARGET_DOMAIN.items():
        sub = tmp_path / fold_id
        sub.mkdir()
        repo = _build_reserve_ready_fixture(sub, monkeypatch, fold_id)
        base_plan = _fold_base_plan(repo, fold_id)
        reserve_base = rs.build_reserve_base_schedule(repo, fold_id, base_plan=base_plan)
        target_slug = {"CASIA-FASD": "casia_fasd", "MSU-MFSD": "msu_mfsd",
                      "SiW-Mv2": "siw_mv2"}[target_domain]
        for row in reserve_base["rows"]:
            assert row["live_dataset"] != target_slug
            if row["spoof_dataset"] is not None:
                assert row["spoof_dataset"] != target_slug


def test_d_load_source_rows_reads_source_train_manifest_only(tmp_path, monkeypatch):
    repo = _build_reserve_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    package_root = repo / fx.e7g.GPAT_INPUT_ROOT / "EXT-F1"
    # The frozen `load_source_rows` itself asserts every row's project_split == "source_train"
    # and reads nothing else -- reused verbatim, never re-implemented, by this module.
    import inspect as _inspect

    source = _inspect.getsource(spp.load_source_rows)
    assert "source_train" in source
    # The docstring says target manifests are "never opened"; the CODE BODY (after the closing
    # docstring) must never reference a target path/manifest at all.
    body = source.split('"""', 2)[-1]
    assert "target" not in body.lower()


# --- E. SYNCHRONIZED TRANCHE -------------------------------------------------

def test_e_no_single_arm_or_single_route_api_exists():
    sig = inspect.signature(rs.materialize_reserve_schedule_for_fold)
    names = set(sig.parameters)
    assert "arm" not in names
    assert "route" not in names
    sig2 = inspect.signature(rs.build_reserve_base_schedule)
    assert "arm" not in set(sig2.parameters)


def test_e_every_tranche_synchronized_across_arms_and_routes(tmp_path, monkeypatch):
    repo = _build_reserve_ready_fixture(tmp_path, monkeypatch, "EXT-F2")
    base_plan = _fold_base_plan(repo, "EXT-F2")
    schedule = rs.materialize_reserve_schedule_for_fold(
        repo, "EXT-F2", base_plan=base_plan, gpat_checkpoint_sha256=_FAKE_CHECKPOINT_SHA,
        physics_engine_version=_FAKE_PHYSICS_VERSION, max_tranches=2)
    assert set(schedule["arms"]) == set(spp.ARMS)
    for arm in spp.ARMS:
        rows = schedule["arms"][arm]["rows"]
        by_tranche: dict[int, dict[str, int]] = {}
        for row in rows:
            counts = by_tranche.setdefault(row["reserve_tranche"], {spp.PHYSICS: 0, spp.GPAT: 0})
            counts[row["route"]] += 1
        assert set(by_tranche) == {1, 2}
        for counts in by_tranche.values():
            assert counts[spp.PHYSICS] == 256
            assert counts[spp.GPAT] == 256


# --- F. NO OUTCOME ADAPTATION -----------------------------------------------

def test_f_no_outcome_shaped_parameters_anywhere():
    forbidden_substrings = ("accept", "quality", "q_value", "pass_count", "outcome", "reject", "score")
    for fn in (rs.build_reserve_base_schedule, rs.build_reserve_arm_schedule,
              rs.materialize_reserve_schedule_for_fold, rs.reserve_schedule_rule_identity,
              rs.reserve_position_for, rs.reserve_domain_relation_for_tranche, rs.reserve_slot_for):
        for name in inspect.signature(fn).parameters:
            lowered = name.lower()
            assert not any(bad in lowered for bad in forbidden_substrings), f"{fn.__name__} takes {name!r}"


def test_f_schedule_unaffected_by_unrelated_observed_generation_state(tmp_path, monkeypatch):
    repo = _build_reserve_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    base_plan = _fold_base_plan(repo, "EXT-F1")
    before = rs.materialize_reserve_schedule_for_fold(
        repo, "EXT-F1", base_plan=base_plan, gpat_checkpoint_sha256=_FAKE_CHECKPOINT_SHA,
        physics_engine_version=_FAKE_PHYSICS_VERSION)

    # Simulate a completely different "observed outcome" on disk (a fake, very different
    # GENERATION_CLOSURE.json for G-RND) -- the reserve schedule must not react to it in any way,
    # since it never reads any generation-closure/candidate-record path at all.
    closure_path = fx.e7g.generation_closure_path(repo, "EXT-F1", "G-RND")
    closure_path.parent.mkdir(parents=True, exist_ok=True)
    closure_path.write_text(json.dumps({"status": "MATCHED_BANK_INFEASIBLE", "fabricated": True}),
                            encoding="utf-8")

    after = rs.materialize_reserve_schedule_for_fold(
        repo, "EXT-F1", base_plan=base_plan, gpat_checkpoint_sha256=_FAKE_CHECKPOINT_SHA,
        physics_engine_version=_FAKE_PHYSICS_VERSION)
    for arm in spp.ARMS:
        assert before["arms"][arm]["rows"] == after["arms"][arm]["rows"]


# --- G. CAP -------------------------------------------------------------------

def test_g_cap_enforced_at_four_tranches(tmp_path, monkeypatch):
    repo = _build_reserve_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    base_plan = _fold_base_plan(repo, "EXT-F1")
    with pytest.raises(rs.E7ReserveScheduleError, match="outside 1.."):
        rs.build_reserve_base_schedule(repo, "EXT-F1", base_plan=base_plan, max_tranches=5)
    with pytest.raises(rs.E7ReserveScheduleError):
        rs.reserve_position_for(5, spp.PHYSICS, 0)
    with pytest.raises(rs.E7ReserveScheduleError):
        rs.reserve_tranche_of_position(rs.RESERVE_POSITION_BASE + rs.POSITIONS_PER_TRANCHE * rs.MAX_TRANCHES)


def test_g_no_schedule_beyond_2048_candidates_per_route_arm(tmp_path, monkeypatch):
    repo = _build_reserve_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    base_plan = _fold_base_plan(repo, "EXT-F1")
    schedule = rs.materialize_reserve_schedule_for_fold(
        repo, "EXT-F1", base_plan=base_plan, gpat_checkpoint_sha256=_FAKE_CHECKPOINT_SHA,
        physics_engine_version=_FAKE_PHYSICS_VERSION, max_tranches=rs.MAX_TRANCHES)
    for arm in spp.ARMS:
        by_route = {spp.PHYSICS: 0, spp.GPAT: 0}
        for row in schedule["arms"][arm]["rows"]:
            by_route[row["route"]] += 1
        # 1024 initial (not part of this schedule) + this reserve schedule's own count must not
        # exceed the cap of 2048/route.
        assert 1024 + by_route[spp.PHYSICS] == 2048
        assert 1024 + by_route[spp.GPAT] == 2048


# --- H. RESUME (schedule-level idempotency; no rendering exists yet) --------

def test_h_schedule_construction_idempotent_never_reassigns(tmp_path, monkeypatch):
    repo = _build_reserve_ready_fixture(tmp_path, monkeypatch, "EXT-F3")
    base_plan = _fold_base_plan(repo, "EXT-F3")
    runs = [rs.materialize_reserve_schedule_for_fold(
                repo, "EXT-F3", base_plan=base_plan, gpat_checkpoint_sha256=_FAKE_CHECKPOINT_SHA,
                physics_engine_version=_FAKE_PHYSICS_VERSION)
            for _ in range(3)]
    for arm in spp.ARMS:
        reference = runs[0]["arms"][arm]["rows"]
        for run in runs[1:]:
            assert run["arms"][arm]["rows"] == reference


def test_h_protocol_lock_write_then_validate_repeated(tmp_path, monkeypatch):
    repo = fx._base_repo(tmp_path)
    rs.write_reserve_protocol_lock(repo)
    for _ in range(3):
        assert rs.validate_reserve_protocol_lock(repo)["status"] == "VALID"


# --- I. V1.0 IMMUTABILITY ----------------------------------------------------

def test_i_existing_generation_closure_files_byte_unchanged(tmp_path, monkeypatch):
    repo = _build_reserve_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    fake_closures = {}
    for condition in ("G-RND", "G-DET", "G-LLM"):
        path = fx.e7g.generation_closure_path(repo, "EXT-F1", condition)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps({"status": "BLOCKED_PROTOCOL_GAP_RESERVE_SCHEDULE", "condition": condition,
                          "fold_id": "EXT-F1"})
        path.write_text(body, encoding="utf-8")
        fake_closures[path] = body

    base_plan = _fold_base_plan(repo, "EXT-F1")
    rs.materialize_reserve_schedule_for_fold(
        repo, "EXT-F1", base_plan=base_plan, gpat_checkpoint_sha256=_FAKE_CHECKPOINT_SHA,
        physics_engine_version=_FAKE_PHYSICS_VERSION)
    rs.write_reserve_protocol_lock(repo)

    for path, body in fake_closures.items():
        assert path.read_text(encoding="utf-8") == body


def test_i_existing_candidate_dir_untouched(tmp_path, monkeypatch):
    repo = _build_reserve_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    candidate_root = fx.e7g.generation_candidate_root(repo, "EXT-F1")
    fake_candidate = candidate_root / "RND" / "c5syn_deadbeefdeadbeefdead" / "CANDIDATE.json"
    fake_candidate.parent.mkdir(parents=True, exist_ok=True)
    fake_candidate.write_text('{"status": "generated"}', encoding="utf-8")
    before = fake_candidate.read_bytes()

    base_plan = _fold_base_plan(repo, "EXT-F1")
    rs.materialize_reserve_schedule_for_fold(
        repo, "EXT-F1", base_plan=base_plan, gpat_checkpoint_sha256=_FAKE_CHECKPOINT_SHA,
        physics_engine_version=_FAKE_PHYSICS_VERSION)

    assert fake_candidate.read_bytes() == before


# --- protocol-lock content proofs -------------------------------------------

def test_lock_names_amendment_and_v1_0_reference_correctly(tmp_path):
    repo = fx._base_repo(tmp_path)
    lock = rs.write_reserve_protocol_lock(repo)
    assert lock["amendment_name"] == "E7-v1.1 RESERVE AMENDMENT"
    assert lock["not_e0_preregistered"] is True
    assert lock["amendment_status"] == "FROZEN_POST_OBSERVATION_PRE_RENDER"
    for fold_id in fx.e7g.FOLD_IDS:
        for condition in ("G-RND", "G-DET", "G-LLM"):
            assert lock["v1_0_terminal_observations_referenced"][fold_id][condition] == \
                "BLOCKED_PROTOCOL_GAP_RESERVE_SCHEDULE"
    assert lock["v1_0_evidence_immutable"] is True
    assert lock["rendering_authorized"] is False
    assert lock["target_access"] is False
    assert lock["llm_api_calls"] == 0
    assert lock["budget"]["hard_cap_per_route_arm"] == 2048
    assert lock["budget"]["hard_cap_total_per_arm"] == 4096
    assert lock["budget"]["reserve_tranche_per_route"] == 256
    assert lock["budget"]["max_tranches"] == 4


def test_lock_shuffle_policy_never_rescues_f1(tmp_path):
    repo = fx._base_repo(tmp_path)
    lock = rs.write_reserve_protocol_lock(repo)
    assert "not rescued" in lock["shuffle_policy"] or "NOT rescued" in lock["shuffle_policy"]
    assert lock["rendering_authorized"] is False


# --- frozen primitives unchanged --------------------------------------------

def test_frozen_c5_c6_gpat_primitives_unchanged_by_reserve_module():
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
    ):
        committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=REPO, check=True,
                                   capture_output=True, text=True).stdout
        on_disk = (REPO / relative).read_text(encoding="utf-8")
        assert committed == on_disk, f"{relative} differs from HEAD"


def test_no_rendering_or_quality_or_llm_function_in_module():
    import prism_fas.evaluation.c_ext_e7_reserve_schedule as module

    names = [name for name in dir(module) if not name.startswith("_")]
    forbidden = ("render", "evaluate_pool", "gate_candidates", "select_route_bank", "llm", "train")
    offending = [name for name in names if any(bad in name.lower() for bad in forbidden)]
    assert offending == [], f"reserve-schedule module unexpectedly defines: {offending}"


def test_no_target_access_no_llm_calls_declared_in_lock(tmp_path):
    repo = fx._base_repo(tmp_path)
    lock = rs.write_reserve_protocol_lock(repo)
    assert lock["target_access"] is False
    assert lock["llm_api_calls"] == 0
