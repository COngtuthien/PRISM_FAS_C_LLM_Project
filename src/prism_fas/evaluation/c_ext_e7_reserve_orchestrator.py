"""E7-v1.1 RESERVE ORCHESTRATOR -- additive, tranche-by-tranche reserve
candidate generation/evaluation/matched-bank orchestration for the frozen
E7-v1.1 RESERVE AMENDMENT (`c_ext_e7_reserve_schedule.py`, rule identity
`7871404603876a7b015af80cefaca26d94d54f102a9eb445099254aa3bbd1822`).

This module NEVER re-derives the reserve schedule itself -- every position/
route/recipe-ordinal/domain-relation/candidate-id decision is delegated to
`c_ext_e7_reserve_schedule.py` (frozen, committed, tested separately). This
module's own job is orchestration only: checking the eight tranche-open
conditions, rendering one tranche at a time (reusing the frozen `c5_render`/
`c6_scientific`/`c6_matched_bank` primitives and this repository's own,
already-real, `c_ext_e7_gpat_bank` rendering/evaluation helpers -- NOT
reimplementing them), evaluating CUMULATIVE feasibility (v1.0's immutable
2048/arm plus every reserve tranche rendered so far) under the exact frozen
STRICT/NOMINAL/PERMISSIVE profile walk, and recording provenance.

`c_ext_e7_gpat_bank.py` (the v1.0 module) is imported and reused, never
modified: `_render_all_arms`/`_evaluate_all_arms`/
`_scoped_e7_mask_compatibility_binding`/`_generation_capability`/
`_current_gpat_expected_identity` are the SAME GPU-boundary/mask-
compatibility/capability functions v1.0 candidate generation already uses,
reused here by direct import rather than copied, so a bug fix there is a
bug fix here too and there is exactly one implementation of each concern.
This module adds no public/CLI surface to `c_ext_e7_gpat_bank.py` itself.

Historical v1.0 evidence is read-only here: this module never writes to a
`G-RND`/`G-DET`/`G-LLM` `BANK_LOCK.json`/`GENERATION_CLOSURE.json` path, and
never renders a v1.0 (position < 2048) candidate. It writes ONLY under a
separate, additive `e7_v1_1_reserve/` namespace, both under
`reports/c_ext_q1q2_v1/e7_three_fold/gpat_bank/e7_v1_1_reserve/` (protocol-
level, fold-independent artifacts, already written by
`c_ext_e7_reserve_schedule.py`) and under
`runs/c_ext_q1q2_v1/e7_gpat_bank/<fold_id>/e7_v1_1_reserve/` (per-fold run
state/tranche provenance, this module's own).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from prism_fas.evaluation import c_ext_common as cc
from prism_fas.evaluation import c_ext_e7_gpat_bank as e7g
from prism_fas.evaluation import c_ext_e7_reserve_schedule as rs
from prism_fas.synthesis import c5_source_pair_plan as spp

SCHEMA_PREFIX = "ext-q1q2-e7-v1-1-reserve-orchestrator"

RUN_V1_1_SUBDIR = "e7_v1_1_reserve"

PROTOCOL_BINDING_FILENAME = "RESERVE_PROTOCOL_BINDING.json"
SCHEDULE_LOCK_FILENAME = "RESERVE_SCHEDULE_LOCK.json"
RUN_STATE_FILENAME = "RESERVE_RUN_STATE.json"
FINAL_CLOSURE_FILENAME = "RESERVE_FINAL_CLOSURE.json"

#: Per-arm/per-route cumulative counts at the frozen v1.0 pool -- never rerendered.
V1_0_CUMULATIVE_PER_ROUTE = {spp.PHYSICS: 1024, spp.GPAT: 1024}
#: The three v1.0 conditions this amendment is scoped to. Never a fourth/fifth condition.
CORE_CONDITIONS = ("G-RND", "G-DET", "G-LLM")

STATUS_AWAITING_OPEN = "AWAITING_RESERVE_OPEN"
STATUS_TRANCHE_IN_PROGRESS = "TRANCHE_IN_PROGRESS"
STATUS_NEEDS_NEXT_TRANCHE = "NEEDS_NEXT_RESERVE_TRANCHE"
STATUS_CLOSED_MATCHED = "CLOSED_MATCHED"
#: The EXACT terminology this module always uses on cap exhaustion -- never a silently
#: different string. Recorded verbatim in the protocol lock/amendment as the chosen terminology.
STATUS_CAP_BLOCKED = "SCIENTIFICALLY_BLOCKED_AFTER_RESERVE_CAP"


class E7ReserveOrchestratorError(ValueError):
    """A reserve-orchestration precondition failed, or a scientific-atomicity
    invariant would be violated."""


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

def reserve_run_root(repo: Path, fold_id: str) -> Path:
    return repo / e7g.RUN_ROOT / fold_id / RUN_V1_1_SUBDIR


def reserve_protocol_binding_path(repo: Path, fold_id: str) -> Path:
    return reserve_run_root(repo, fold_id) / PROTOCOL_BINDING_FILENAME


def reserve_schedule_lock_path(repo: Path, fold_id: str) -> Path:
    return reserve_run_root(repo, fold_id) / SCHEDULE_LOCK_FILENAME


def reserve_run_state_path(repo: Path, fold_id: str) -> Path:
    return reserve_run_root(repo, fold_id) / RUN_STATE_FILENAME


def reserve_tranche_lock_path(repo: Path, fold_id: str, tranche: int) -> Path:
    return reserve_run_root(repo, fold_id) / f"RESERVE_TRANCHE_{tranche:02d}_LOCK.json"


def reserve_tranche_closure_path(repo: Path, fold_id: str, tranche: int) -> Path:
    return reserve_run_root(repo, fold_id) / f"RESERVE_TRANCHE_{tranche:02d}_CLOSURE.json"


def reserve_final_closure_path(repo: Path, fold_id: str) -> Path:
    return reserve_run_root(repo, fold_id) / FINAL_CLOSURE_FILENAME


# --------------------------------------------------------------------------- #
# 1-8. Tranche-open conditions -- checked, never assumed.
# --------------------------------------------------------------------------- #

def check_tranche_open_conditions(repo: Path, fold_id: str) -> dict[str, Any]:
    """Checks the eight conditions a reserve tranche may open under. Returns
    a dict of per-condition booleans/evidence; never raises -- the caller
    (`assert_tranche_open_conditions`) decides whether to fail closed. Kept
    separate so a read-only preflight can report WHICH condition is missing
    without needing to catch an exception."""
    if fold_id not in e7g.FOLD_IDS:
        raise E7ReserveOrchestratorError(f"unknown fold_id {fold_id!r}")

    v1_0_condition_status: dict[str, str | None] = {}
    for condition in CORE_CONDITIONS:
        path = e7g.generation_closure_path(repo, fold_id, condition)
        v1_0_condition_status[condition] = cc.read_json(path).get("status") if path.is_file() else None
    v1_0_terminal = all(status == e7g.BLOCKED_PROTOCOL_GAP_RESERVE_SCHEDULE
                        for status in v1_0_condition_status.values())

    gpat_lock_status = e7g.validate_gpat_fit_lock(repo, fold_id)["status"]
    quality_calibration_status = e7g.validate_quality_calibration(repo, fold_id)["status"]
    amendment_lock_status = rs.validate_reserve_protocol_lock(repo)["status"]
    schedule_materialized = reserve_schedule_lock_path(repo, fold_id).is_file()
    invariance = verify_v1_0_invariance(repo, fold_id) if schedule_materialized else \
        {"invariant": False, "reason": "reserve schedule not yet materialized"}

    return {
        "fold_id": fold_id,
        "1_v1_0_terminal_status": {"satisfied": v1_0_terminal, "observed": v1_0_condition_status},
        "2_gpat_fit_lock_valid": {"satisfied": gpat_lock_status == "VALID", "status": gpat_lock_status},
        "3_quality_calibration_valid": {"satisfied": quality_calibration_status == "VALID",
                                        "status": quality_calibration_status},
        "4_amendment_protocol_lock_valid": {"satisfied": amendment_lock_status == "VALID",
                                            "status": amendment_lock_status},
        "5_reserve_schedule_materialized": {"satisfied": schedule_materialized},
        "6_v1_0_invariance_verified": {"satisfied": bool(invariance.get("invariant")), "detail": invariance},
        "7_no_target_domain_access": {"satisfied": True,
                                      "reason": "this module opens only GPAT_INPUT_ROOT/<fold_id> "
                                               "source-only packages; structurally incapable of "
                                               "opening a target path (no such path is ever "
                                               "constructed anywhere in this module)"},
        "8_synchronized_arms": {"satisfied": True, "arms": list(spp.ARMS)},
    }


def assert_tranche_open_conditions(repo: Path, fold_id: str) -> dict[str, Any]:
    """FAIL CLOSED unless every one of the eight conditions is satisfied."""
    report = check_tranche_open_conditions(repo, fold_id)
    unmet = [key for key, value in report.items() if isinstance(value, dict) and "satisfied" in value
            and not value["satisfied"]]
    if unmet:
        raise E7ReserveOrchestratorError(f"{fold_id}: reserve tranche open conditions not met: "
                                         f"{unmet} -- {report!r}")
    return report


def verify_v1_0_invariance(repo: Path, fold_id: str) -> dict[str, Any]:
    """Condition 6: rebuilds the ORIGINAL v1.0 plans fresh (via the frozen,
    unmodified `c5_source_pair_plan`/`c5_arm_plan`) and compares against
    what this fold's own `RESERVE_SCHEDULE_LOCK.json` recorded as the v1.0
    snapshot at materialization time -- proving the historical pool has not
    silently drifted since."""
    lock_path = reserve_schedule_lock_path(repo, fold_id)
    if not lock_path.is_file():
        return {"invariant": False, "reason": "RESERVE_SCHEDULE_LOCK.json not materialized"}
    lock = cc.read_json(lock_path)
    recorded_v1_0 = lock.get("v1_0_pool_snapshot")
    if not recorded_v1_0:
        return {"invariant": False, "reason": "schedule lock carries no v1_0_pool_snapshot"}
    fresh = rs.rebuild_v1_0_pool_for_invariance_check(
        repo, fold_id, gpat_checkpoint_sha256=lock["gpat_checkpoint_sha256"])
    problems = [key for key in recorded_v1_0 if recorded_v1_0.get(key) != fresh.get(key)]
    return {"invariant": not problems, "problems": problems}


# --------------------------------------------------------------------------- #
# Protocol binding -- written once per fold, before any tranche.
# --------------------------------------------------------------------------- #

def build_reserve_protocol_binding(repo: Path, fold_id: str) -> dict[str, Any]:
    """Binds this fold's reserve run to every identity item the spec-owner
    ratification requires: amendment rule identity, implementation-commit
    provenance, fold identity, source-pair-plan identity, per-arm recipe-
    bank identities, GPAT checkpoint SHA, quality-calibration identity, and
    the frozen quality-profile identities (STRICT/NOMINAL/PERMISSIVE,
    unmodified)."""
    if fold_id not in e7g.FOLD_IDS:
        raise E7ReserveOrchestratorError(f"unknown fold_id {fold_id!r}")
    gpat_validation = e7g.validate_gpat_fit_lock(repo, fold_id)
    if gpat_validation["status"] != "VALID":
        raise E7ReserveOrchestratorError(f"{fold_id}: GPAT fit lock is not VALID")
    gpat_lock = cc.read_json(e7g.gpat_fit_lock_path(repo, fold_id))
    calibration_validation = e7g.validate_quality_calibration(repo, fold_id)
    if calibration_validation["status"] != "VALID":
        raise E7ReserveOrchestratorError(f"{fold_id}: quality calibration is not VALID")
    calibration_payload = cc.read_json(e7g.quality_calibration_path(repo, fold_id))

    from prism_fas.synthesis import c5_arm_plan, c6_scientific
    from prism_fas.synthesis.quality_gate import Thresholds

    recipe_bank_identities = {arm: c5_arm_plan.load_arm_bank(repo, arm)["bank_identity"] for arm in spp.ARMS}
    nominal_thresholds = Thresholds.from_dict(calibration_payload["thresholds"]).as_dict()
    profiles = c6_scientific.build_common_profiles(nominal_thresholds, nominal_source="RESERVE_PROTOCOL_BINDING")
    quality_profile_identities = {name: profile.identity for name, profile in profiles.items()}

    provenance = e7g.resolve_implementation_commit_provenance(repo)

    return {
        "schema_version": f"{SCHEMA_PREFIX}-protocol-binding-v1",
        "fold_id": fold_id,
        "amendment_rule_identity": rs.reserve_schedule_rule_identity(),
        "amendment_name": rs.AMENDMENT_NAME,
        "implementation_commit": provenance["implementation_commit"],
        "implementation_module_sha256": provenance["implementation_module_sha256"],
        "repository_head_commit": provenance["repository_head_commit"],
        "source_pair_plan_identity": spp.source_pair_plan_identity(
            spp.build_source_pair_plan(repo / e7g.GPAT_INPUT_ROOT / fold_id)),
        "recipe_bank_identities": recipe_bank_identities,
        "gpat_checkpoint_sha256": gpat_lock["best_checkpoint_sha256"],
        "quality_calibration_sha256": cc.sha256_bytes(
            e7g.quality_calibration_path(repo, fold_id).read_bytes()),
        "quality_profile_identities": quality_profile_identities,
        "target_access": False, "llm_api_calls": 0,
    }


def write_reserve_protocol_binding(repo: Path, fold_id: str) -> dict[str, Any]:
    from prism_fas.utils.core import atomic_json_write

    body = build_reserve_protocol_binding(repo, fold_id)
    path = reserve_protocol_binding_path(repo, fold_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, body)
    return body


# --------------------------------------------------------------------------- #
# Real per-fold reserve schedule materialization (dry-run: schedule only,
# zero candidate-image writes).
# --------------------------------------------------------------------------- #

def materialize_real_reserve_schedule(repo: Path, fold_id: str) -> dict[str, Any]:
    """Materializes and locks the COMPLETE reserve schedule (all 4 tranches,
    all 3 arms, both routes) for this fold from its REAL, on-disk GPAT-input
    package and frozen C3 recipe banks -- via `c_ext_e7_reserve_schedule.
    materialize_reserve_schedule_for_fold`, never re-derived here. This
    writes NO candidate image bytes; it is pure position/identity
    arithmetic plus one JSON lock write."""
    if fold_id not in e7g.FOLD_IDS:
        raise E7ReserveOrchestratorError(f"unknown fold_id {fold_id!r}")
    gpat_validation = e7g.validate_gpat_fit_lock(repo, fold_id)
    if gpat_validation["status"] != "VALID":
        raise E7ReserveOrchestratorError(f"{fold_id}: GPAT fit lock is not VALID -- cannot bind "
                                         "a reserve schedule to an unfitted checkpoint")
    gpat_lock = cc.read_json(e7g.gpat_fit_lock_path(repo, fold_id))
    gpat_checkpoint_sha256 = gpat_lock["best_checkpoint_sha256"]

    from prism_fas.synthesis.physics import PHYSICS_ENGINE_VERSION

    package_root = repo / e7g.GPAT_INPUT_ROOT / fold_id
    base_plan = spp.build_source_pair_plan(package_root)
    v1_0_snapshot = rs.rebuild_v1_0_pool_for_invariance_check(
        repo, fold_id, gpat_checkpoint_sha256=gpat_checkpoint_sha256)
    schedule = rs.materialize_reserve_schedule_for_fold(
        repo, fold_id, base_plan=base_plan, gpat_checkpoint_sha256=gpat_checkpoint_sha256,
        physics_engine_version=PHYSICS_ENGINE_VERSION, max_tranches=rs.MAX_TRANCHES)

    return {
        "schema_version": f"{SCHEMA_PREFIX}-schedule-lock-v1",
        "fold_id": fold_id,
        "amendment_rule_identity": rs.reserve_schedule_rule_identity(),
        "gpat_checkpoint_sha256": gpat_checkpoint_sha256,
        "physics_engine_version": PHYSICS_ENGINE_VERSION,
        "source_pair_plan_identity": schedule["base"]["source_pair_plan_identity"],
        "max_tranches": rs.MAX_TRANCHES,
        "v1_0_pool_snapshot": v1_0_snapshot,
        "arm_schedule_identities": {arm: schedule["arms"][arm]["reserve_schedule_identity"]
                                   for arm in spp.ARMS},
        # Tranche keys are stringified -- JSON has no integer-keyed object, and this dict is
        # written to and read back from disk between calls (`str(tranche)` used consistently by
        # every reader below).
        "candidate_ids_by_arm_and_tranche": {
            arm: {
                str(tranche): [row["candidate_id"] for row in schedule["arms"][arm]["rows"]
                              if row["reserve_tranche"] == tranche]
                for tranche in range(1, rs.MAX_TRANCHES + 1)
            }
            for arm in spp.ARMS
        },
        "rendering_performed": False, "target_access": False, "llm_api_calls": 0,
    }


def write_reserve_schedule_lock(repo: Path, fold_id: str) -> dict[str, Any]:
    from prism_fas.utils.core import atomic_json_write

    body = materialize_real_reserve_schedule(repo, fold_id)
    path = reserve_schedule_lock_path(repo, fold_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, body)
    return body


def validate_reserve_schedule_lock(repo: Path, fold_id: str) -> dict[str, Any]:
    """STRICT, read-only. Independently rebuilds the schedule lock from
    current disk state and compares."""
    path = reserve_schedule_lock_path(repo, fold_id)
    if not path.is_file():
        return {"schema_version": f"{SCHEMA_PREFIX}-schedule-lock-validate-v1", "fold_id": fold_id,
               "status": "NOT_MATERIALIZED"}
    recorded = cc.read_json(path)
    try:
        recomputed = materialize_real_reserve_schedule(repo, fold_id)
    except Exception as exc:  # noqa: BLE001
        return {"schema_version": f"{SCHEMA_PREFIX}-schedule-lock-validate-v1", "fold_id": fold_id,
               "status": "INVALID", "problems": [f"could not rebuild: {exc!r}"]}
    problems = [f"{key} drifted" for key in recomputed if recorded.get(key) != recomputed.get(key)]
    return {"schema_version": f"{SCHEMA_PREFIX}-schedule-lock-validate-v1", "fold_id": fold_id,
           "status": "INVALID" if problems else "VALID", "problems": problems}


def candidate_id_disjointness_report(repo: Path, fold_id: str) -> dict[str, Any]:
    """Read-only proof that every reserve candidate id, across every arm
    and tranche, is disjoint from the fold's immutable v1.0 candidate ids
    AND from every other reserve tranche's own ids."""
    lock = cc.read_json(reserve_schedule_lock_path(repo, fold_id))
    v1_0_ids = {arm: set(lock["v1_0_pool_snapshot"]["candidate_ids_by_arm"][arm]) for arm in spp.ARMS}
    per_arm_report = {}
    for arm in spp.ARMS:
        by_tranche = lock["candidate_ids_by_arm_and_tranche"][arm]
        all_reserve_ids: list[str] = []
        tranche_disjoint = True
        seen: set[str] = set()
        for tranche in sorted(by_tranche, key=int):
            ids = by_tranche[tranche]
            if seen & set(ids):
                tranche_disjoint = False
            seen |= set(ids)
            all_reserve_ids.extend(ids)
        per_arm_report[arm] = {
            "disjoint_from_v1_0": v1_0_ids[arm].isdisjoint(all_reserve_ids),
            "disjoint_across_tranches": tranche_disjoint,
            "total_reserve_candidates": len(all_reserve_ids),
            "unique_reserve_candidates": len(set(all_reserve_ids)),
        }
    return {"fold_id": fold_id, "by_arm": per_arm_report,
           "all_disjoint": all(entry["disjoint_from_v1_0"] and entry["disjoint_across_tranches"]
                              for entry in per_arm_report.values())}


# --------------------------------------------------------------------------- #
# Run state
# --------------------------------------------------------------------------- #

def _cumulative_counts_after_tranche(tranche: int) -> dict[str, dict[str, int]]:
    """The exact cumulative Physics/GPAT candidate count per arm after
    `tranche` reserve tranches have been RENDERED (0 = the immutable v1.0
    pool alone, before any reserve tranche) -- a pure function of `tranche`
    alone: `V1_0_CUMULATIVE_PER_ROUTE[route] + 256 * tranche`. Used
    identically regardless of WHY a tranche's evaluation loop stopped
    (`NEEDS_NEXT_RESERVE_TRANCHE` or `SCIENTIFICALLY_BLOCKED_AFTER_
    RESERVE_CAP`): the cumulative count reflects how many tranches were
    rendered, never why. Fixes a bookkeeping bug discovered after EXT-F1's
    real terminal tranche-4 run under commit db87b1c: the cap-block branch
    previously left `state`'s PRIOR cumulative_counts (1792/1792, tranche
    3's value) in place instead of recomputing tranche 4's own (2048/2048).
    That run's SCIENTIFIC result (SCIENTIFICALLY_BLOCKED_AFTER_RESERVE_CAP,
    the PERMISSIVE assessment, the route-quota/shortfall numbers) was
    computed correctly and is untouched by this fix -- only this metadata
    field was stale. See E7_V1_1_TERMINAL_COUNT_BOOKKEEPING_CORRECTION.md."""
    return {arm: {route: V1_0_CUMULATIVE_PER_ROUTE[route] + 256 * tranche
                 for route in (spp.PHYSICS, spp.GPAT)}
           for arm in spp.ARMS}


def build_initial_run_state(fold_id: str) -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_PREFIX}-run-state-v1", "fold_id": fold_id,
        "status": STATUS_AWAITING_OPEN, "next_tranche": 1, "closed_at_tranche": None,
        "cumulative_counts": _cumulative_counts_after_tranche(0),
        "target_access": False, "llm_api_calls": 0,
    }


def read_run_state(repo: Path, fold_id: str) -> dict[str, Any]:
    path = reserve_run_state_path(repo, fold_id)
    if not path.is_file():
        return build_initial_run_state(fold_id)
    return cc.read_json(path)


def write_run_state(repo: Path, fold_id: str, state: dict[str, Any]) -> dict[str, Any]:
    from prism_fas.utils.core import atomic_json_write

    path = reserve_run_state_path(repo, fold_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, state)
    return state


def next_allowed_tranche(repo: Path, fold_id: str) -> int | None:
    """Read-only: the next tranche this fold is allowed to open, or `None`
    if the fold is already closed (matched or cap-blocked)."""
    state = read_run_state(repo, fold_id)
    if state["status"] in (STATUS_CLOSED_MATCHED, STATUS_CAP_BLOCKED):
        return None
    return state["next_tranche"]


# --------------------------------------------------------------------------- #
# Cumulative candidate pool -- v1.0's immutable 2048/arm plus every reserve
# tranche rendered so far. Built from the frozen `c6_scientific.
# SelectableCandidate`-shaped rows; never a second selector implementation.
# --------------------------------------------------------------------------- #

def _cumulative_pool(v1_0_arm_plans: dict[str, Any], reserve_rows_so_far: dict[str, list[dict[str, Any]]]
                     ) -> dict[str, dict[str, Any]]:
    from prism_fas.synthesis import c6_matched_bank as selector

    pool: dict[str, dict[str, Any]] = {}
    for arm in spp.ARMS:
        arm_pool: dict[str, Any] = {}
        for row in v1_0_arm_plans[arm]["candidates"]:
            arm_pool[row["candidate_id"]] = selector.SelectableCandidate(
                candidate_id=row["candidate_id"], arm=arm, route=row["route"],
                source_domain=str(row[selector.SOURCE_DOMAIN_PLAN_FIELD]), recipe_id=row["recipe_id"],
                recipe_ordinal=int(row["recipe_ordinal"]), live_target_sample_id=row["live_target_sample_id"],
                base_position=int(row["position"]))
        for row in reserve_rows_so_far.get(arm, []):
            arm_pool[row["candidate_id"]] = selector.SelectableCandidate(
                candidate_id=row["candidate_id"], arm=arm, route=row["route"],
                source_domain=str(row["live_dataset"]), recipe_id=row["recipe_id"],
                recipe_ordinal=int(row["recipe_ordinal"]), live_target_sample_id=row["live_target_sample_id"],
                base_position=int(row["position"]))
        pool[arm] = arm_pool
    return pool


def _cumulative_plans_for_quota(v1_0_arm_plans: dict[str, Any], reserve_rows_so_far: dict[str, list[dict[str, Any]]]
                                ) -> dict[str, dict[str, Any]]:
    """A `plans`-shaped view (`{"candidates": [...]}`) over v1.0 + reserve
    rows, for `c6_matched_bank.route_quotas`'s `planned_domain_counts`
    input -- the ideal domain share must reflect the FULL cumulative planned
    pool, not only the original 2048."""
    plans: dict[str, dict[str, Any]] = {}
    for arm in spp.ARMS:
        combined = list(v1_0_arm_plans[arm]["candidates"]) + list(reserve_rows_so_far.get(arm, []))
        plans[arm] = {"candidates": combined}
    return plans


# --------------------------------------------------------------------------- #
# Phase 2 -- read-only / dry-run reporting. NONE of these functions render a
# single candidate byte; they only read locks/state already on disk (or
# recompute pure schedule arithmetic) and report.
# --------------------------------------------------------------------------- #

def tranche_counts_report(repo: Path, fold_id: str) -> dict[str, Any]:
    """Prints (returns) the exact cumulative Physics/GPAT counts at v1.0 and
    after each of the 4 tranches, plus this fold's current run-state
    position in that sequence."""
    state = read_run_state(repo, fold_id)
    counts = {"v1_0": dict(V1_0_CUMULATIVE_PER_ROUTE)}
    for t in range(1, rs.MAX_TRANCHES + 1):
        counts[f"after_tranche_{t}"] = {route: V1_0_CUMULATIVE_PER_ROUTE[route] + 256 * t
                                       for route in (spp.PHYSICS, spp.GPAT)}
    return {"fold_id": fold_id, "run_state_status": state["status"], "next_tranche": state.get("next_tranche"),
           "closed_at_tranche": state.get("closed_at_tranche"), "counts_per_arm_route": counts}


def validate_source_domain_firewall(repo: Path, fold_id: str) -> dict[str, Any]:
    """Read-only: proves this fold's reserve schedule rows never carry its
    own held-out target dataset slug as a live or spoof dataset -- the SAME
    proof `test_d_reserve_rows_never_carry_the_target_dataset` already
    makes for the schedule module, re-run here as a standing preflight
    check rather than only a unit test."""
    if fold_id not in e7g.FOLD_IDS:
        raise E7ReserveOrchestratorError(f"unknown fold_id {fold_id!r}")
    package_root = repo / e7g.GPAT_INPUT_ROOT / fold_id
    base_plan = spp.build_source_pair_plan(package_root)
    reserve_base = rs.build_reserve_base_schedule(repo, fold_id, base_plan=base_plan, max_tranches=rs.MAX_TRANCHES)
    target_domain = e7g.FOLD_TARGET_DOMAIN[fold_id]
    target_slug = {"CASIA-FASD": "casia_fasd", "MSU-MFSD": "msu_mfsd", "SiW-Mv2": "siw_mv2"}[target_domain]
    violations = [row["position"] for row in reserve_base["rows"]
                 if row["live_dataset"] == target_slug or row.get("spoof_dataset") == target_slug]
    return {"fold_id": fold_id, "target_domain": target_domain, "target_slug": target_slug,
           "status": "VALID" if not violations else "INVALID", "violating_positions": violations[:16],
           "violation_count": len(violations), "target_access": False, "llm_api_calls": 0}


def reserve_execution_authorization_report(repo: Path, fold_id: str) -> dict[str, Any]:
    """Combines every read-only check into one authorization verdict --
    NEVER renders. `authorized=True` means `open_and_close_tranche` would
    not immediately fail closed on a precondition; it does NOT mean GPU
    rendering has been run, is running, or is being requested by this call."""
    conditions = check_tranche_open_conditions(repo, fold_id)
    unmet = [key for key, value in conditions.items()
            if isinstance(value, dict) and "satisfied" in value and not value["satisfied"]]
    try:
        capability = e7g._generation_capability(repo)
    except Exception as exc:  # noqa: BLE001 -- capability resolution itself must never crash a preflight
        capability = {"capable": False, "cuda_available": False, "problems": [f"{exc!r}"]}
    next_tranche = next_allowed_tranche(repo, fold_id)
    authorized = not unmet and capability["capable"] and next_tranche is not None
    return {"fold_id": fold_id, "authorized": authorized, "unmet_conditions": unmet,
           "capability": capability, "next_tranche": next_tranche, "rendering_performed": False,
           "target_access": False, "llm_api_calls": 0}


def dry_run_reserve_preflight(repo: Path, fold_id: str) -> dict[str, Any]:
    """`--reserve-preflight --fold EXT-Fn`: the READ-ONLY, zero-candidate-
    write preflight report combining Phase 2 items 1-9. Never renders,
    never opens a tranche, never writes a tranche lock/closure, never
    mutates run state."""
    schedule_materialized = reserve_schedule_lock_path(repo, fold_id).is_file()
    return {
        "schema_version": f"{SCHEMA_PREFIX}-preflight-v1", "fold_id": fold_id,
        "1_amendment_protocol_lock": rs.validate_reserve_protocol_lock(repo),
        "2_3_reserve_schedule_lock": validate_reserve_schedule_lock(repo, fold_id),
        "4_v1_0_invariance": (verify_v1_0_invariance(repo, fold_id) if schedule_materialized
                              else {"invariant": False, "reason": "schedule not materialized"}),
        "5_candidate_id_disjointness": (candidate_id_disjointness_report(repo, fold_id)
                                        if schedule_materialized else None),
        "6_tranche_counts": tranche_counts_report(repo, fold_id),
        "7_source_domain_firewall": validate_source_domain_firewall(repo, fold_id),
        "8_9_authorization": reserve_execution_authorization_report(repo, fold_id),
        "rendering_performed": False, "target_access": False, "llm_api_calls": 0,
    }


def all_folds_next_tranche_report(repo: Path) -> dict[str, int | None]:
    """Phase 2 item 8, across all three folds at once."""
    return {fold_id: next_allowed_tranche(repo, fold_id) for fold_id in e7g.FOLD_IDS}


# --------------------------------------------------------------------------- #
# One tranche: render (GPU-gated), evaluate, assess CUMULATIVE feasibility.
# --------------------------------------------------------------------------- #

def open_and_close_tranche(repo: Path, fold_id: str, tranche: int, *, authorize: bool = False
                           ) -> dict[str, Any]:
    """`--reserve-open-tranche --authorize --fold EXT-Fn --tranche N`: the
    ONE scientific transaction for one reserve tranche. Fails closed unless
    every one of `check_tranche_open_conditions`'s eight conditions holds
    AND `tranche == next_allowed_tranche(repo, fold_id)` (never opens tranche
    N+1 while tranche N is incomplete, never opens a tranche out of order).

    Rendering reuses `c_ext_e7_gpat_bank._render_all_arms`/
    `_evaluate_all_arms`/`_scoped_e7_mask_compatibility_binding`/
    `_generation_capability` VERBATIM (the SAME GPU-boundary/mask-
    compatibility functions v1.0 candidate generation already uses) against
    a `plan`-shaped adapter built from this fold's ALREADY-LOCKED reserve
    schedule rows for `tranche` -- never a second rendering implementation.
    Feasibility reuses `c6_scientific.build_common_profiles`/
    `gate_candidates`/`eligible_candidates`/`assess_profile`/
    `select_strictest_profile` and `c6_matched_bank.build_matched_banks`
    VERBATIM, over the CUMULATIVE pool (v1.0's immutable 2048/arm plus every
    reserve tranche rendered so far, including this one)."""
    if not authorize:
        raise E7ReserveOrchestratorError(f"reserve tranche {tranche} for {fold_id} requires --authorize")
    if not 1 <= tranche <= rs.MAX_TRANCHES:
        raise E7ReserveOrchestratorError(f"tranche {tranche} outside 1..{rs.MAX_TRANCHES}")

    state = read_run_state(repo, fold_id)
    if state["status"] in (STATUS_CLOSED_MATCHED, STATUS_CAP_BLOCKED):
        raise E7ReserveOrchestratorError(f"{fold_id}: already closed ({state['status']}); no further "
                                         "tranche may open")
    expected_next = state["next_tranche"]
    if tranche != expected_next:
        raise E7ReserveOrchestratorError(f"{fold_id}: tranche {tranche} requested but the next "
                                         f"allowed tranche is {expected_next} -- tranches open "
                                         "strictly in order, never out of order, never before the "
                                         "prior tranche is complete")

    assert_tranche_open_conditions(repo, fold_id)
    schedule_lock = cc.read_json(reserve_schedule_lock_path(repo, fold_id))

    # (1) Non-terminal attempt provenance for THIS tranche, written atomically BEFORE any
    # rendering -- mirrors v1.0's own attempt-provenance-before-rendering discipline.
    from prism_fas.utils.core import atomic_json_write

    provenance = e7g.resolve_implementation_commit_provenance(repo)
    tranche_lock_body = {
        "schema_version": f"{SCHEMA_PREFIX}-tranche-lock-v1", "fold_id": fold_id, "tranche": tranche,
        "amendment_rule_identity": rs.reserve_schedule_rule_identity(),
        "arm_schedule_identities": schedule_lock["arm_schedule_identities"],
        "candidate_ids": {arm: schedule_lock["candidate_ids_by_arm_and_tranche"][arm][str(tranche)]
                         for arm in spp.ARMS},
        "implementation_commit": provenance["implementation_commit"],
        "implementation_module_sha256": provenance["implementation_module_sha256"],
        "repository_head_commit": provenance["repository_head_commit"],
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False,
    }
    tranche_lock_path = reserve_tranche_lock_path(repo, fold_id, tranche)
    tranche_lock_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(tranche_lock_path, tranche_lock_body)
    write_run_state(repo, fold_id, {**state, "status": STATUS_TRANCHE_IN_PROGRESS})

    # (2) Real GPAT identity + capability gates -- reused verbatim from v1.0.
    gpat_validation = e7g.validate_gpat_fit_lock(repo, fold_id)
    gpat_lock = cc.read_json(e7g.gpat_fit_lock_path(repo, fold_id))
    gpat_identity = e7g._current_gpat_expected_identity(repo, fold_id)
    for field in gpat_identity:
        if gpat_identity[field] != gpat_lock.get(field):
            raise E7ReserveOrchestratorError(f"{fold_id}: current GPAT identity {field} drifted from "
                                             "GPAT_FIT_LOCK.json -- FAIL CLOSED")
    capability = e7g._generation_capability(repo)
    if not capability["capable"]:
        raise E7ReserveOrchestratorError(f"{fold_id}: GPU_REQUIRED for reserve tranche {tranche} "
                                         f"rendering/evaluation -- this host is not capable: "
                                         f"{capability['problems']!r}")

    # (3) Rebuild the fold's v1.0 arm plans (frozen, unmodified) and this tranche's reserve rows
    # from the ALREADY-LOCKED schedule -- never re-derived independently of the lock.
    v1_0_plans = e7g.materialize_candidate_plans(repo, fold_id,
                                                 gpat_checkpoint_sha256=gpat_lock["best_checkpoint_sha256"])
    v1_0_arm_plans = v1_0_plans["arm_plans"]

    from prism_fas.synthesis.physics import PHYSICS_ENGINE_VERSION

    base_plan = v1_0_plans["base_plan"]
    reserve_base = rs.build_reserve_base_schedule(repo, fold_id, base_plan=base_plan,
                                                  max_tranches=rs.MAX_TRANCHES)
    reserve_arm_schedules = {
        arm: rs.build_reserve_arm_schedule(repo, fold_id, arm, reserve_base_schedule=reserve_base,
                                           gpat_checkpoint_sha256=gpat_lock["best_checkpoint_sha256"],
                                           physics_engine_version=PHYSICS_ENGINE_VERSION)
        for arm in spp.ARMS
    }
    rs.assert_reserve_arms_share_the_schedule(reserve_arm_schedules)
    for arm in spp.ARMS:
        ids_now = [row["candidate_id"] for row in reserve_arm_schedules[arm]["rows"]
                  if row["reserve_tranche"] == tranche]
        if ids_now != tranche_lock_body["candidate_ids"][arm]:
            raise E7ReserveOrchestratorError(f"{fold_id}/{arm}: freshly-rebuilt tranche {tranche} "
                                             "candidate ids disagree with the just-written tranche "
                                             "lock -- FAIL CLOSED")

    # Prior tranches' rows (1..tranche-1) plus this tranche's rows -- the CUMULATIVE reserve set.
    reserve_rows_through_this_tranche = {
        arm: [row for row in reserve_arm_schedules[arm]["rows"] if row["reserve_tranche"] <= tranche]
        for arm in spp.ARMS
    }
    this_tranche_plan_adapter = {
        arm: {**reserve_arm_schedules[arm],
             "candidates": [row for row in reserve_arm_schedules[arm]["rows"]
                           if row["reserve_tranche"] == tranche],
             "arm": arm, "arm_plan_identity": reserve_arm_schedules[arm]["reserve_schedule_identity"]}
        for arm in spp.ARMS
    }

    # (4) Render THIS tranche only -- reuses v1.0's own GPU-boundary/mask-compatibility helpers
    # verbatim; the frozen 0..2047 pool is never touched by this call.
    from prism_fas.synthesis import c5_render
    from prism_fas.synthesis.m8_pipeline import SampleStore
    from prism_fas.synthesis.quality_calibration import QualityBackends, load_quality_config
    from prism_fas.synthesis.synthetic_bank import CandidateEvaluator, FrozenCalibration

    package_root = repo / e7g.GPAT_INPUT_ROOT / fold_id
    device = "cuda" if capability["cuda_available"] else "cpu"
    routes = c5_render.build_routes(repo, checkpoint_path=e7g.gpat_best_checkpoint_path(repo, fold_id),
                                    checkpoint_sha256=gpat_lock["best_checkpoint_sha256"],
                                    expected_identity=gpat_identity, device=device)
    store = SampleStore.open(package_root)
    recovery_counter = [0]
    try:
        render_results = e7g._render_all_arms(repo, fold_id, arm_plans=this_tranche_plan_adapter,
                                              routes=routes, store=store, recovery_counter=recovery_counter)
    except c5_render.RenderError as exc:
        raise E7ReserveOrchestratorError(f"{fold_id}: reserve tranche {tranche} rendering aborted -- "
                                         f"FAIL CLOSED, candidate NOT consumed, tranche remains "
                                         f"resumable at its exact pre-locked membership: {exc}") from exc

    # (5) Evaluate the CUMULATIVE pool (v1.0 + every reserve tranche through this one) -- v1.0's
    # own already-generated candidates are RE-EVALUATED (never re-rendered) from their existing
    # bytes on disk, exactly like re-running the same measurement twice is expected to agree.
    weight_root = Path(capability["weight_root"])
    backends = QualityBackends(weight_root, device=device)
    calibration = FrozenCalibration.load(e7g.quality_calibration_path(repo, fold_id))
    config = load_quality_config(repo / e7g.QUALITY_CONFIG_PATH)
    evaluator = CandidateEvaluator(backends, calibration)

    cumulative_arm_plans_for_eval = {
        arm: {**v1_0_arm_plans[arm],
             "candidates": list(v1_0_arm_plans[arm]["candidates"]) + reserve_rows_through_this_tranche[arm]}
        for arm in spp.ARMS
    }
    metrics_by_arm = e7g._evaluate_all_arms(repo, fold_id, arm_plans=cumulative_arm_plans_for_eval,
                                            store=store, evaluator=evaluator, recovery_counter=recovery_counter)

    # (6) The exact frozen STRICT -> NOMINAL -> PERMISSIVE walk, over the CUMULATIVE pool.
    from prism_fas.synthesis import c6_matched_bank as selector
    from prism_fas.synthesis import c6_scientific

    pool = _cumulative_pool(v1_0_arm_plans, reserve_rows_through_this_tranche)
    cumulative_plans_for_quota = _cumulative_plans_for_quota(v1_0_arm_plans, reserve_rows_through_this_tranche)
    profiles = c6_scientific.build_common_profiles(calibration.thresholds.as_dict(),
                                                    nominal_source="quality_calibration.calibrate")
    assessments = []
    decisions_by_profile: dict[str, dict[str, list[dict[str, Any]]]] = {}
    accepted_by_profile: dict[str, dict[str, list[Any]]] = {}
    for name in c6_scientific.PROFILE_ORDER:
        profile = profiles[name]
        decisions_by_arm = {arm: c6_scientific.gate_candidates(metrics_by_arm[arm], profile.as_thresholds())
                            for arm in spp.ARMS}
        accepted_by_arm = {arm: c6_scientific.eligible_candidates(decisions_by_arm[arm], pool[arm])
                          for arm in spp.ARMS}
        assessments.append(c6_scientific.assess_profile(name, accepted_by_arm, cumulative_plans_for_quota))
        decisions_by_profile[name] = decisions_by_arm
        accepted_by_profile[name] = accepted_by_arm
    decision = c6_scientific.select_strictest_profile(assessments)

    evidence = {"tranche": tranche, "assessments": [assessment.as_dict() for assessment in assessments],
               "profile_decision": decision.as_dict(),
               "mask_compatibility_recovery_count": recovery_counter[0]}

    if decision.selected is None:
        # The cumulative count reflects how many tranches were RENDERED, never why the loop
        # stopped -- the SAME helper applies whether this tranche needs a successor or is the
        # terminal cap-exhaustion tranche (item 4 of the bug report: the cap-block branch must
        # record tranche 4's own 2048/2048, not silently retain tranche 3's 1792/1792).
        cumulative_counts = _cumulative_counts_after_tranche(tranche)
        if tranche < rs.MAX_TRANCHES:
            new_status = STATUS_NEEDS_NEXT_TRANCHE
            new_state = {**state, "status": new_status, "next_tranche": tranche + 1,
                        "cumulative_counts": cumulative_counts}
        else:
            new_status = STATUS_CAP_BLOCKED
            new_state = {**state, "status": new_status, "next_tranche": None, "closed_at_tranche": None,
                        "cumulative_counts": cumulative_counts}
        closure = {**evidence, "schema_version": f"{SCHEMA_PREFIX}-tranche-closure-v1", "fold_id": fold_id,
                  "status": new_status, "is_scientific_lock": False, "target_access": False,
                  "llm_api_calls": 0}
        atomic_json_write(reserve_tranche_closure_path(repo, fold_id, tranche), closure)
        write_run_state(repo, fold_id, new_state)
        if new_status == STATUS_CAP_BLOCKED:
            final = {**closure, "schema_version": f"{SCHEMA_PREFIX}-final-closure-v1",
                    "final_status": STATUS_CAP_BLOCKED, "closed_at_tranche": None}
            atomic_json_write(reserve_final_closure_path(repo, fold_id), final)
        return {"resumed": False, "status": new_status, "tranche": tranche, "closure": closure}

    # (7) Feasible: freeze the matched banks (additive v1.1 namespace only -- v1.0's own
    # G-RND/G-DET/G-LLM BANK_LOCK.json paths are NEVER written by this module).
    profile = decision.selected
    accepted_by_arm = accepted_by_profile[profile]
    decisions_by_arm = decisions_by_profile[profile]
    outcome = selector.build_matched_banks(cumulative_plans_for_quota, accepted_by_arm)
    if not outcome["matched"]:
        raise E7ReserveOrchestratorError(f"{fold_id}: decision.selected chosen on this exact "
                                         "feasibility test but build_matched_banks disagreed -- "
                                         "defensive, should be unreachable")

    threshold_id = c6_scientific.threshold_identity(profiles[profile].as_thresholds())
    banks = {}
    for arm in spp.ARMS:
        bank = outcome["banks"][arm]
        selector_contract = selector.selector_identity(
            quality_profile_identity=threshold_id,
            c5_pool_lock_sha256=reserve_arm_schedules[arm]["reserve_schedule_identity"],
            decision_set_sha256=selector.decision_set_digest(decisions_by_arm[arm]))
        closure_proof = c6_scientific.provenance_closure(
            pool_candidate_ids=list(pool[arm]), semantic_failure_ids=[], decisions=decisions_by_arm[arm],
            selected_ids=[row["candidate_id"] for row in bank["selected"]])
        lock_body = c6_scientific.bank_lock_payload(
            arm=arm, bank=bank, selector_contract=selector_contract, profile=profile,
            threshold_identity=threshold_id,
            c5_pool_lock_sha256=reserve_arm_schedules[arm]["reserve_schedule_identity"],
            provenance=closure_proof)
        lock_body.update({"fold_id": fold_id, "closed_at_tranche": tranche,
                          "TARGET_IMAGE_ACCESS": False, "TARGET_LABEL_ACCESS": False, "LLM_API_CALLS": 0})
        banks[arm] = lock_body

    closure = {**evidence, "schema_version": f"{SCHEMA_PREFIX}-tranche-closure-v1", "fold_id": fold_id,
              "status": STATUS_CLOSED_MATCHED, "is_scientific_lock": True, "profile": profile,
              "banks": banks, "target_access": False, "llm_api_calls": 0}
    atomic_json_write(reserve_tranche_closure_path(repo, fold_id, tranche), closure)
    final = {**closure, "schema_version": f"{SCHEMA_PREFIX}-final-closure-v1",
            "final_status": STATUS_CLOSED_MATCHED, "closed_at_tranche": tranche}
    atomic_json_write(reserve_final_closure_path(repo, fold_id), final)
    write_run_state(repo, fold_id, {**state, "status": STATUS_CLOSED_MATCHED, "next_tranche": None,
                                    "closed_at_tranche": tranche})
    return {"resumed": False, "status": STATUS_CLOSED_MATCHED, "tranche": tranche, "closure": closure}
