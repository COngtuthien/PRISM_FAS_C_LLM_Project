"""E7-v1.1 RESERVE AMENDMENT -- the additive, deterministic reserve-tranche
candidate schedule for the governing EXT-Q1Q2 Detailed Spec v1.0 section 6.1
"Unified synthetic candidate budget" reserve-tranche mechanism.

`reports/c_ext_q1q2_v1/e7_three_fold/gpat_bank/RESERVE_SCHEDULE_AUTHORITY_
AUDIT.md` found NO authoritative exact reserve schedule anywhere in the
governing spec, the frozen C5/C6/GPAT synthesis primitives, or repository
history -- section 6.1 states the numeric budget/trigger/cap but specifies
nothing about which recipe ordinal, live sample, GPAT spoof partner, route,
domain relation, or candidate identity a reserve position beyond the frozen
2048/arm receives. That audit's "Option 1" draft has since been RATIFIED by
the protocol owner as this amendment, E7-v1.1 RESERVE AMENDMENT:

* reserve scheduling is an ADDITIVE, deterministic extension of the existing
  frozen C5 position-keyed source-scheduling semantics;
* `c5_source_pair_plan.load_source_rows`/`live_for_position`/
  `spoof_for_position`/`candidate_identity` and `c5_arm_plan.load_arm_bank`/
  `_recipe_id` are reused VERBATIM, unmodified, over a NEW, disjoint,
  strictly-higher position range;
* positions `0..2047` (§10.4's frozen initial 2048/arm) and everything
  derived from them are NEVER touched, renumbered or reinterpreted;
* BLOCK route assignment (the first 256 positions of a tranche are Physics,
  the next 256 are GPAT), as proposed in the audit;
* NO frozen C5/C6/GPAT synthesis primitive file is modified by this module.

This amendment is explicitly FROZEN AFTER observing, under the existing
source-only/target-blind protocol, that EXT-F1/EXT-F2/EXT-F3's initial v1.0
2048-candidate pass each terminated `BLOCKED_PROTOCOL_GAP_RESERVE_SCHEDULE`
on every one of G-RND/G-DET/G-LLM, and BEFORE any reserve candidate is
rendered. It is a genuine mid-study protocol amendment and is NEVER an E0
preregistration. The three historical v1.0 terminal observations remain
immutable and are never rewritten, rescued, or reinterpreted as successful
by this module.

This module builds and locks the DETERMINISTIC SCHEDULE ONLY. It contains
NO rendering, NO quality evaluation, and NO matched-bank orchestration --
those stages reuse `c5_render.render_arm`/`c6_scientific.evaluate_pool`/
`c6_matched_bank.build_matched_banks` unmodified, over this schedule's rows,
in a LATER, separately-authorized task. Actually opening (rendering) the
first reserve tranche requires that later, explicit authorization; this
module's job ends at proving the complete schedule and its identities are
determined before that happens.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from prism_fas.evaluation import c_ext_common as cc
from prism_fas.evaluation import c_ext_e7_gpat_bank as e7g
from prism_fas.synthesis import c5_arm_plan
from prism_fas.synthesis import c5_source_pair_plan as spp

SCHEMA_PREFIX = "ext-q1q2-e7-v1-1-reserve"

AMENDMENT_NAME = "E7-v1.1 RESERVE AMENDMENT"
#: Never "PREREGISTERED_AT_E0" -- this status name is itself part of the audit trail.
AMENDMENT_STATUS = "FROZEN_POST_OBSERVATION_PRE_RENDER"
GOVERNING_SPEC_SECTION = "EXT-Q1Q2 Detailed Spec v1.0 section 6.1 (Unified synthetic candidate budget)"
AUTHORITY_AUDIT_PATH = "reports/c_ext_q1q2_v1/e7_three_fold/gpat_bank/RESERVE_SCHEDULE_AUTHORITY_AUDIT.json"

#: The v1.0 status this amendment exists to resolve. Referenced, never rewritten: v1.0's own
#: GENERATION_CLOSURE.json files stay exactly as they are.
V1_0_TERMINAL_STATUS_REFERENCE = "BLOCKED_PROTOCOL_GAP_RESERVE_SCHEDULE"
V1_0_CONDITIONS = ("G-RND", "G-DET", "G-LLM")

#: One past the last frozen v1.0 position (`c5_source_pair_plan.CANDIDATES_PER_ARM` = 2048).
#: Reserve positions occupy `RESERVE_POSITION_BASE ..` exclusively; `0..RESERVE_POSITION_BASE-1`
#: is the frozen v1.0 pool and is never produced by this module.
RESERVE_POSITION_BASE = spp.CANDIDATES_PER_ARM
POSITIONS_PER_ROUTE_PER_TRANCHE = 256
ROUTES_PER_TRANCHE = 2
POSITIONS_PER_TRANCHE = POSITIONS_PER_ROUTE_PER_TRANCHE * ROUTES_PER_TRANCHE  # 512
#: §6.1 cap: 1024 initial + 4*256 reserve = 2048 candidates/route/arm.
MAX_TRANCHES = 4
RECIPES_PER_ARM = spp.RECIPES_PER_ARM  # 256, the frozen C3 bank size -- never changed
ORIGINAL_SLOTS_PER_RECIPE = 8  # the frozen v1.0 RENDERS_PER_RECIPE


class E7ReserveScheduleError(ValueError):
    """A reserve-schedule precondition failed under the frozen E7-v1.1 rule."""


# --------------------------------------------------------------------------- #
# Pure position arithmetic -- the entire deterministic mapping. No repo, no
# I/O, no fold, no arm, no observed outcome anywhere in these signatures.
# --------------------------------------------------------------------------- #

def reserve_position_for(tranche: int, route: str, recipe_ordinal: int) -> int:
    """The ONE global candidate position for (tranche, route, recipe_ordinal).
    Inverse of `reserve_route_for_position`/`reserve_recipe_ordinal_for_position`/
    `reserve_tranche_of_position`."""
    if not 1 <= tranche <= MAX_TRANCHES:
        raise E7ReserveScheduleError(f"tranche {tranche} outside 1..{MAX_TRANCHES}")
    if route not in (spp.PHYSICS, spp.GPAT):
        raise E7ReserveScheduleError(f"unknown route {route!r}")
    if not 0 <= recipe_ordinal < RECIPES_PER_ARM:
        raise E7ReserveScheduleError(f"recipe_ordinal {recipe_ordinal} outside 0..{RECIPES_PER_ARM - 1}")
    route_offset = 0 if route == spp.PHYSICS else POSITIONS_PER_ROUTE_PER_TRANCHE
    return RESERVE_POSITION_BASE + POSITIONS_PER_TRANCHE * (tranche - 1) + route_offset + recipe_ordinal


def reserve_tranche_of_position(position: int) -> int:
    if position < RESERVE_POSITION_BASE:
        raise E7ReserveScheduleError(
            f"position {position} belongs to the frozen v1.0 pool (0..{RESERVE_POSITION_BASE - 1}), "
            "not a reserve position")
    tranche = (position - RESERVE_POSITION_BASE) // POSITIONS_PER_TRANCHE + 1
    if tranche > MAX_TRANCHES:
        raise E7ReserveScheduleError(f"position {position} exceeds the frozen {MAX_TRANCHES}-tranche cap")
    return tranche


def reserve_route_for_position(position: int) -> str:
    reserve_tranche_of_position(position)  # range/cap validation
    offset_in_tranche = (position - RESERVE_POSITION_BASE) % POSITIONS_PER_TRANCHE
    return spp.PHYSICS if offset_in_tranche < POSITIONS_PER_ROUTE_PER_TRANCHE else spp.GPAT


def reserve_recipe_ordinal_for_position(position: int) -> int:
    """recipe_ordinal = the position's 0-based rank within its own route's
    block of the tranche (0..255) -- BLOCK route assignment makes this
    directly the local offset, one full additional cycle through the frozen
    256-recipe bank per tranche per route."""
    reserve_tranche_of_position(position)
    offset_in_tranche = (position - RESERVE_POSITION_BASE) % POSITIONS_PER_TRANCHE
    return offset_in_tranche if offset_in_tranche < POSITIONS_PER_ROUTE_PER_TRANCHE \
        else offset_in_tranche - POSITIONS_PER_ROUTE_PER_TRANCHE


def reserve_domain_relation_for_tranche(tranche: int) -> str:
    """Whole-tranche alternation: SAME_DOMAIN for odd tranches (1, 3),
    CROSS_DOMAIN for even tranches (2, 4) -- at the 4-tranche cap this is
    exactly 2 SAME + 2 CROSS tranches (512 + 512 GPAT reserve candidates),
    the same 50/50 balance the frozen 4-GPAT-slots/recipe v1.0 schedule
    already guarantees, achieved cumulatively rather than within one
    tranche."""
    if not 1 <= tranche <= MAX_TRANCHES:
        raise E7ReserveScheduleError(f"tranche {tranche} outside 1..{MAX_TRANCHES}")
    return spp.SAME_DOMAIN if (tranche - 1) % 2 == 0 else spp.CROSS_DOMAIN


def reserve_slot_for(tranche: int, route: str) -> int:
    """Extends the frozen v1.0 per-recipe slot numbering (0..7) naturally:
    each tranche gives recipe ordinal r exactly one more Physics render and
    one more GPAT render, so slot 8, 9 is tranche 1's Physics/GPAT render,
    10, 11 is tranche 2's, and so on -- never colliding with a v1.0 slot
    (0..7) and never reusing a slot across tranches/routes."""
    if route not in (spp.PHYSICS, spp.GPAT):
        raise E7ReserveScheduleError(f"unknown route {route!r}")
    if not 1 <= tranche <= MAX_TRANCHES:
        raise E7ReserveScheduleError(f"tranche {tranche} outside 1..{MAX_TRANCHES}")
    return ORIGINAL_SLOTS_PER_RECIPE + 2 * (tranche - 1) + (0 if route == spp.PHYSICS else 1)


def reserve_schedule_rule_identity() -> str:
    """The AMENDMENT's own identity: a hash over the deterministic RULE
    parameters alone (fold-independent, arm-independent, outcome-
    independent) -- true regardless of which fold/arm/tranche is realized.
    Two independent evaluations of this function, today or a year from now,
    on this exact module, return the SAME value; it changes only if the
    rule itself is amended again."""
    material = {
        "schema_version": f"{SCHEMA_PREFIX}-rule-identity-v1",
        "amendment_name": AMENDMENT_NAME,
        "governing_spec_section": GOVERNING_SPEC_SECTION,
        "reserve_position_base": RESERVE_POSITION_BASE,
        "positions_per_route_per_tranche": POSITIONS_PER_ROUTE_PER_TRANCHE,
        "routes_per_tranche": ROUTES_PER_TRANCHE,
        "max_tranches": MAX_TRANCHES,
        "recipes_per_arm": RECIPES_PER_ARM,
        "original_slots_per_recipe": ORIGINAL_SLOTS_PER_RECIPE,
        "route_assignment_rule": "BLOCK: the first 256 positions of a tranche are PHYSICS, the "
                                 "next 256 are GPAT",
        "recipe_ordinal_rule": "recipe_ordinal = the position's 0-based rank within its own "
                               "route's block of the tranche (0..255) -- one additional full "
                               "cycle through the frozen 256-recipe bank per tranche per route",
        "domain_relation_rule": "SAME_DOMAIN for odd tranches (1, 3), CROSS_DOMAIN for even "
                                "tranches (2, 4) -- whole-tranche alternation",
        "slot_rule": "slot = 8 + 2*(tranche-1) + (0 if PHYSICS else 1)",
        "seed": spp.PLAN_SEED,
        "reused_frozen_functions": [
            "c5_source_pair_plan.load_source_rows", "c5_source_pair_plan.live_for_position",
            "c5_source_pair_plan.spoof_for_position", "c5_source_pair_plan.candidate_identity",
            "c5_source_pair_plan.source_pair_plan_identity", "c5_arm_plan.load_arm_bank",
            "c5_arm_plan._recipe_id",
        ],
        "frozen_primitives_not_modified": [
            "c5_source_pair_plan.py", "c5_arm_plan.py", "c5_raw_generation.py", "c5_render.py",
            "synthetic_bank.py", "quality_gate.py", "quality_calibration.py", "c6_scientific.py",
            "c6_matched_bank.py", "gpat_trainer.py", "gpat_model.py", "gpat_losses.py",
            "m8_pipeline.py", "masks.py", "pair_plan.py", "gate_profiles.py",
        ],
    }
    return cc.sha256_bytes(cc.canonical_json_bytes(material))


def _reserve_arm_schedule_identity(*, fold_id: str, arm: str, source_pair_plan_identity: str,
                                   recipe_bank_identity: str, gpat_checkpoint_sha256: str,
                                   physics_engine_version: str, ontology_identity: str,
                                   rule_identity: str) -> str:
    """One arm/fold's reserve-schedule identity: the amendment's fold/arm-
    independent RULE identity, bound to this fold's own source schedule and
    this arm's own recipe bank -- mirrors `c5_source_pair_plan.
    arm_candidate_plan_identity`'s shape (a SEPARATE function: that frozen
    function's `arm in {RND,DET,LLM}` check must never be widened, though
    here `arm` already is one of those three)."""
    material = {"schema_version": f"{SCHEMA_PREFIX}-arm-schedule-identity-v1", "fold_id": fold_id,
               "arm": arm, "source_pair_plan_identity": source_pair_plan_identity,
               "recipe_bank_identity": recipe_bank_identity,
               "gpat_checkpoint_sha256": gpat_checkpoint_sha256,
               "physics_engine_version": physics_engine_version,
               "ontology_identity": ontology_identity, "rule_identity": rule_identity}
    return cc.sha256_bytes(cc.canonical_json_bytes(material))


# --------------------------------------------------------------------------- #
# Schedule construction -- reuses the frozen C5 functions verbatim.
# --------------------------------------------------------------------------- #

def build_reserve_base_schedule(repo: Path, fold_id: str, *, base_plan: dict[str, Any],
                                max_tranches: int = MAX_TRANCHES) -> dict[str, Any]:
    """The ARM-INDEPENDENT reserve position -> (route, recipe_ordinal,
    domain_relation, live, spoof) mapping for tranches `1..max_tranches`,
    over this fold's OWN already-validated base schedule (`base_plan`, the
    SAME object `c_ext_e7_gpat_bank.materialize_candidate_plans` builds for
    v1.0). Reuses `load_source_rows`/`live_for_position`/`spoof_for_position`
    VERBATIM, unmodified -- nothing here decides how a position becomes a
    live/spoof pair, only WHICH new positions exist.

    Fails closed if the live/spoof pools resolved right now have drifted
    from what `base_plan` already locked (`live_list_identity`/
    `spoof_list_identity`) -- reserve positions must draw from the EXACT
    SAME source-only pools the frozen v1.0 schedule used, never a silently
    different one.
    """
    if not 1 <= max_tranches <= MAX_TRANCHES:
        raise E7ReserveScheduleError(f"max_tranches {max_tranches} outside 1..{MAX_TRANCHES}")
    package_root = repo / e7g.GPAT_INPUT_ROOT / fold_id
    live_list, spoof_list = spp.load_source_rows(package_root)
    live_identity = spp._list_identity([row.sample_id for row in live_list])
    spoof_identity = spp._list_identity([row.sample_id for row in spoof_list])
    if live_identity != base_plan["live_list_identity"]:
        raise E7ReserveScheduleError(
            f"{fold_id}: the live-sample pool resolved right now (identity {live_identity!r}) has "
            f"drifted from the frozen base plan's live_list_identity {base_plan['live_list_identity']!r}")
    if spoof_identity != base_plan["spoof_list_identity"]:
        raise E7ReserveScheduleError(
            f"{fold_id}: the spoof-sample pool resolved right now (identity {spoof_identity!r}) has "
            f"drifted from the frozen base plan's spoof_list_identity {base_plan['spoof_list_identity']!r}")

    rows: list[dict[str, Any]] = []
    for tranche in range(1, max_tranches + 1):
        relation = reserve_domain_relation_for_tranche(tranche)
        for route in (spp.PHYSICS, spp.GPAT):
            for recipe_ordinal in range(RECIPES_PER_ARM):
                position = reserve_position_for(tranche, route, recipe_ordinal)
                live = spp.live_for_position(live_list, position)
                domain_relation = relation if route == spp.GPAT else None
                spoof = (spp.spoof_for_position(spoof_list, live, position, domain_relation,
                                                seed=spp.PLAN_SEED) if route == spp.GPAT else None)
                rows.append({
                    "reserve_tranche": tranche, "position": position, "recipe_ordinal": recipe_ordinal,
                    "slot": reserve_slot_for(tranche, route), "route": route,
                    "domain_relation": domain_relation, "live_target_sample_id": live.sample_id,
                    "live_dataset": live.dataset, "live_source_record_id": live.source_record_id,
                    "spoof_source_sample_id": spoof.sample_id if spoof else None,
                    "spoof_dataset": spoof.dataset if spoof else None,
                    "spoof_source_record_id": spoof.source_record_id if spoof else None,
                })
    _assert_reserve_base_schedule(rows, max_tranches)
    return {"schema_version": f"{SCHEMA_PREFIX}-base-schedule-v1", "fold_id": fold_id,
           "max_tranches": max_tranches, "positions_per_tranche": POSITIONS_PER_TRANCHE,
           "source_pair_plan_identity": spp.source_pair_plan_identity(base_plan),
           "package_identity": base_plan["package_identity"], "rows": rows}


def _assert_reserve_base_schedule(rows: list[dict[str, Any]], max_tranches: int) -> None:
    expected = POSITIONS_PER_TRANCHE * max_tranches
    if len(rows) != expected:
        raise E7ReserveScheduleError(f"expected {expected} reserve rows for {max_tranches} tranches, built {len(rows)}")
    positions = [row["position"] for row in rows]
    if len(set(positions)) != len(positions):
        raise E7ReserveScheduleError("duplicate reserve positions produced")
    if min(positions) < RESERVE_POSITION_BASE:
        raise E7ReserveScheduleError("a reserve row occupies a frozen v1.0 position (< 2048)")
    by_tranche: dict[int, dict[str, int]] = {}
    for row in rows:
        counts = by_tranche.setdefault(row["reserve_tranche"], {spp.PHYSICS: 0, spp.GPAT: 0})
        counts[row["route"]] += 1
    for tranche, counts in by_tranche.items():
        if counts[spp.PHYSICS] != POSITIONS_PER_ROUTE_PER_TRANCHE or counts[spp.GPAT] != POSITIONS_PER_ROUTE_PER_TRANCHE:
            raise E7ReserveScheduleError(f"tranche {tranche} route split {counts} != "
                                         f"{POSITIONS_PER_ROUTE_PER_TRANCHE}/{POSITIONS_PER_ROUTE_PER_TRANCHE}")
    for row in rows:
        if row["route"] == spp.GPAT and row["live_source_record_id"] == row["spoof_source_record_id"]:
            raise E7ReserveScheduleError(f"position {row['position']} pairs a live and a spoof from "
                                         "the same source record")


def build_reserve_arm_schedule(repo: Path, fold_id: str, arm: str, *,
                               reserve_base_schedule: dict[str, Any],
                               gpat_checkpoint_sha256: str, physics_engine_version: str
                               ) -> dict[str, Any]:
    """The ARM-SPECIFIC reserve schedule: binds `reserve_base_schedule`'s
    arm-independent rows to `arm`'s own frozen C3 recipe bank and computes
    each reserve candidate's `candidate_id` via the frozen `candidate_
    identity()` VERBATIM -- the SAME function v1.0 candidates already use,
    over the SAME kind of inputs, differing only in `position`/`recipe_id`/
    `recipe_bank_identity`, which is exactly what makes a reserve candidate
    unconditionally distinct from every v1.0 candidate."""
    bank = c5_arm_plan.load_arm_bank(repo, arm)
    if len(bank["recipes"]) != RECIPES_PER_ARM:
        raise E7ReserveScheduleError(f"the {arm} C3 bank holds {len(bank['recipes'])} recipes, "
                                     f"not the frozen {RECIPES_PER_ARM}")
    base_identity = reserve_base_schedule["source_pair_plan_identity"]
    rule_identity = reserve_schedule_rule_identity()
    schedule_identity = _reserve_arm_schedule_identity(
        fold_id=fold_id, arm=arm, source_pair_plan_identity=base_identity,
        recipe_bank_identity=bank["bank_identity"], gpat_checkpoint_sha256=gpat_checkpoint_sha256,
        physics_engine_version=physics_engine_version, ontology_identity=bank["ontology_identity"],
        rule_identity=rule_identity)

    rows: list[dict[str, Any]] = []
    for row in reserve_base_schedule["rows"]:
        ordinal = row["recipe_ordinal"]
        recipe_id = c5_arm_plan._recipe_id(bank["recipes"][ordinal], ordinal)
        binding = gpat_checkpoint_sha256 if row["route"] == spp.GPAT else physics_engine_version
        candidate_id = spp.candidate_identity(
            source_pair_plan_identity=base_identity, arm=arm, recipe_bank_identity=bank["bank_identity"],
            recipe_id=recipe_id, recipe_ordinal=ordinal, slot=row["slot"], position=row["position"],
            route=row["route"], live_target_sample_id=row["live_target_sample_id"],
            spoof_source_sample_id=row["spoof_source_sample_id"],
            package_identity=reserve_base_schedule["package_identity"],
            ontology_identity=bank["ontology_identity"], generator_binding=binding)
        rows.append({**row, "arm": arm, "recipe_id": recipe_id, "recipe_bank_identity": bank["bank_identity"],
                    "generator_binding": binding, "candidate_id": candidate_id})
    _assert_reserve_arm_schedule(rows, arm)
    return {"schema_version": f"{SCHEMA_PREFIX}-arm-schedule-v1", "fold_id": fold_id, "arm": arm,
           "reserve_schedule_identity": schedule_identity, "rule_identity": rule_identity,
           "source_pair_plan_identity": base_identity,
           "package_identity": reserve_base_schedule["package_identity"],
           "recipe_bank_identity": bank["bank_identity"], "ontology_identity": bank["ontology_identity"],
           "gpat_checkpoint_sha256": gpat_checkpoint_sha256,
           "physics_engine_version": physics_engine_version,
           "planned_reserve_candidates": len(rows), "rows": rows}


def _assert_reserve_arm_schedule(rows: list[dict[str, Any]], arm: str) -> None:
    ids = [row["candidate_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise E7ReserveScheduleError(f"{arm} reserve schedule produced duplicate candidate ids")


def materialize_reserve_schedule_for_fold(repo: Path, fold_id: str, *, base_plan: dict[str, Any],
                                          gpat_checkpoint_sha256: str, physics_engine_version: str,
                                          max_tranches: int = MAX_TRANCHES) -> dict[str, Any]:
    """The SYNCHRONIZED reserve schedule for one fold: ALL THREE of
    `spp.ARMS` (RND, DET, LLM), over BOTH routes, for tranches `1..
    max_tranches`, built together. There is deliberately NO API here (and
    nowhere else in this module) to build a schedule for a single arm or a
    single route alone -- §6.1's "for ALL arms in that fold" synchronization
    requirement is a structural guarantee of this function's signature, not
    a runtime check performed after the fact."""
    reserve_base = build_reserve_base_schedule(repo, fold_id, base_plan=base_plan, max_tranches=max_tranches)
    arm_schedules = {
        arm: build_reserve_arm_schedule(repo, fold_id, arm, reserve_base_schedule=reserve_base,
                                        gpat_checkpoint_sha256=gpat_checkpoint_sha256,
                                        physics_engine_version=physics_engine_version)
        for arm in spp.ARMS
    }
    assert_reserve_arms_share_the_schedule(arm_schedules)
    return {"schema_version": f"{SCHEMA_PREFIX}-fold-schedule-v1", "fold_id": fold_id,
           "max_tranches": max_tranches, "base": reserve_base, "arms": arm_schedules}


def assert_reserve_arms_share_the_schedule(arm_schedules: dict[str, dict[str, Any]]) -> None:
    """The SAME fairness invariant `c5_arm_plan.assert_arms_share_the_
    schedule` proves for the frozen v1.0 pool, re-checked (never merely
    assumed) for the reserve extension: RND/DET/LLM see the IDENTICAL
    (position, route, domain_relation, live, spoof) tuple at every reserve
    ordinal, differing only in recipe content."""
    if set(arm_schedules) != set(spp.ARMS):
        raise E7ReserveScheduleError(f"expected schedules for {spp.ARMS}, got {sorted(arm_schedules)}")
    signatures = {
        arm: [(row["position"], row["route"], row["domain_relation"], row["live_target_sample_id"],
              row["spoof_source_sample_id"]) for row in schedule["rows"]]
        for arm, schedule in arm_schedules.items()}
    reference = signatures[spp.ARMS[0]]
    for arm in spp.ARMS[1:]:
        if signatures[arm] != reference:
            differing = next(index for index, (left, right) in enumerate(zip(reference, signatures[arm]))
                             if left != right)
            raise E7ReserveScheduleError(f"the {arm} reserve schedule differs from {spp.ARMS[0]} at "
                                         f"reserve row {differing}; the base reserve schedule must be "
                                         "arm-independent")
    identities = {schedule["reserve_schedule_identity"] for schedule in arm_schedules.values()}
    if len(identities) != len(spp.ARMS):
        raise E7ReserveScheduleError("two arms produced the same reserve schedule identity")


# --------------------------------------------------------------------------- #
# v1.0 pool invariance -- proves the frozen initial 2048/arm is untouched.
# --------------------------------------------------------------------------- #

def rebuild_v1_0_pool_for_invariance_check(repo: Path, fold_id: str, *, gpat_checkpoint_sha256: str
                                           ) -> dict[str, Any]:
    """Rebuilds the ORIGINAL v1.0 base+arm plans via the frozen, UNCHANGED
    `c5_source_pair_plan`/`c5_arm_plan` functions -- this module never calls
    a different code path for v1.0 positions than v1.0 itself already used.
    A caller compares two invocations of this function (e.g. one before and
    one after building a reserve schedule) to prove nothing shifted."""
    from prism_fas.synthesis.physics import PHYSICS_ENGINE_VERSION

    package_root = repo / e7g.GPAT_INPUT_ROOT / fold_id
    base_plan = spp.build_source_pair_plan(package_root)
    if len(base_plan["positions"]) != spp.CANDIDATES_PER_ARM:
        raise E7ReserveScheduleError(f"{fold_id}: v1.0 base plan holds {len(base_plan['positions'])} "
                                     f"positions, not the frozen {spp.CANDIDATES_PER_ARM}")
    arm_plans = c5_arm_plan.build_all_arm_plans(
        repo, base_plan, gpat_checkpoint_sha256=gpat_checkpoint_sha256,
        physics_engine_version=PHYSICS_ENGINE_VERSION)
    return {"fold_id": fold_id, "source_pair_plan_identity": spp.source_pair_plan_identity(base_plan),
           "arm_plan_identities": {arm: plan["arm_plan_identity"] for arm, plan in arm_plans.items()},
           "candidate_ids_by_arm": {arm: [row["candidate_id"] for row in plan["candidates"]]
                                   for arm, plan in arm_plans.items()},
           "max_position": max(row["position"] for row in base_plan["positions"]),
           "position_count": len(base_plan["positions"])}


# --------------------------------------------------------------------------- #
# Protocol lock -- the fold/arm-independent RULE definition, materialized
# BEFORE any per-fold schedule is computed and BEFORE any rendering.
# --------------------------------------------------------------------------- #

E7_V1_1_RESERVE_ROOT = f"{e7g.REPORT_DIR}/e7_v1_1_reserve"
RESERVE_PROTOCOL_LOCK_FILENAME = "E7_V1_1_RESERVE_PROTOCOL_LOCK.json"


def reserve_protocol_lock_path(repo: Path) -> Path:
    return repo / E7_V1_1_RESERVE_ROOT / RESERVE_PROTOCOL_LOCK_FILENAME


def build_reserve_protocol_lock() -> dict[str, Any]:
    """The AMENDMENT's protocol lock: fold-independent and arm-independent
    by construction (the rule is the same for every fold and arm), so it
    requires no real per-fold GPAT-input package to build or verify --
    exactly like the rule itself. Per-fold/arm REALIZED schedules
    (`materialize_reserve_schedule_for_fold`) are computed separately, on
    whichever host holds the real source packages, and are verified against
    this lock's `reserve_schedule_rule_identity` rather than re-deriving it."""
    rule_identity = reserve_schedule_rule_identity()
    return {
        "schema_version": f"{SCHEMA_PREFIX}-protocol-lock-v1",
        "amendment_name": AMENDMENT_NAME,
        "amendment_status": AMENDMENT_STATUS,
        "not_e0_preregistered": True,
        "frozen_after": "observing BLOCKED_PROTOCOL_GAP_RESERVE_SCHEDULE on the v1.0 initial-pool "
                       "attempt for EXT-F1, EXT-F2 and EXT-F3 (all of G-RND/G-DET/G-LLM, every fold)",
        "frozen_before": "any reserve candidate rendering",
        "v1_0_terminal_observations_referenced": {
            fold_id: {condition: V1_0_TERMINAL_STATUS_REFERENCE for condition in V1_0_CONDITIONS}
            for fold_id in e7g.FOLD_IDS
        },
        "v1_0_evidence_immutable": True,
        "v1_0_evidence_rewritten_or_rescued": False,
        "governing_spec_section": GOVERNING_SPEC_SECTION,
        "authority_audit_path": AUTHORITY_AUDIT_PATH,
        "reserve_schedule_rule_identity": rule_identity,
        "budget": {
            "initial_per_arm": {"physics": 1024, "gpat": 1024},
            "goal_accepted_per_arm": {"physics": 512, "gpat": 512},
            "reserve_tranche_per_route": POSITIONS_PER_ROUTE_PER_TRANCHE,
            "max_tranches": MAX_TRANCHES,
            "hard_cap_per_route_arm": RESERVE_POSITION_BASE // 2 + POSITIONS_PER_ROUTE_PER_TRANCHE * MAX_TRANCHES,
            "hard_cap_total_per_arm": RESERVE_POSITION_BASE + POSITIONS_PER_TRANCHE * MAX_TRANCHES,
            "never_lower_quality_threshold": True,
            "underfill_after_cap": "SCIENTIFICALLY_BLOCKED",
            "no_oversampling_to_fake_a_full_bank": True,
        },
        "route_assignment": "BLOCK: within one tranche's 512 new positions, the first 256 (by "
                            "position order) are PHYSICS, the next 256 are GPAT",
        "synchronization_rule": "a reserve tranche is ALWAYS built for all three RND/DET/LLM arms "
                                "and both routes together, via materialize_reserve_schedule_for_"
                                "fold -- this module exposes no API to build a partial (single-arm "
                                "or single-route) tranche",
        "outcome_independence": "no function in this module accepts an accepted-count, a quality "
                                "score, a q value, a prior tranche's result, target data, or any "
                                "other observed outcome as an input; membership is a pure function "
                                "of (fold_id, tranche, route, recipe_ordinal, arm) only",
        "v1_0_pool_invariance": "positions 0..2047 are never produced, touched, or renumbered by "
                                "this module; reserve positions occupy 2048.. exclusively "
                                "(RESERVE_POSITION_BASE)",
        "reused_frozen_functions": [
            "c5_source_pair_plan.load_source_rows", "c5_source_pair_plan.live_for_position",
            "c5_source_pair_plan.spoof_for_position", "c5_source_pair_plan.candidate_identity",
            "c5_source_pair_plan.source_pair_plan_identity", "c5_arm_plan.load_arm_bank",
            "c5_arm_plan._recipe_id", "c5_arm_plan.build_all_arm_plans (invariance check only)",
        ],
        "frozen_primitives_not_modified": [
            "c5_source_pair_plan.py", "c5_arm_plan.py", "c5_raw_generation.py", "c5_render.py",
            "synthetic_bank.py", "quality_gate.py", "quality_calibration.py", "c6_scientific.py",
            "c6_matched_bank.py", "gpat_trainer.py", "gpat_model.py", "gpat_losses.py",
            "m8_pipeline.py", "masks.py", "pair_plan.py", "gate_profiles.py",
        ],
        "shuffle_policy": "the core RND/DET/LLM anchor bank must reach a common matched-feasible "
                          "profile BEFORE Shuffle-A is evaluated for that fold; this amendment "
                          "does not implement Shuffle reserve as an independent rescue mechanism; "
                          "EXT-F1's historical Shuffle scientific infeasibility "
                          "(BLOCKED_TRUE_FROZEN_MATCHED_BANK_INFEASIBILITY) is NOT rescued by it",
        "orchestration_procedure_future_work": [
            "open tranche t for all 3 arms x 2 routes synchronously (this module)",
            "render (reuses c5_render.render_arm unmodified -- NOT implemented by this module)",
            "evaluate (reuses c6_scientific.evaluate_pool/gate_candidates unmodified -- NOT "
            "implemented by this module)",
            "recompute the exact STRICT/NOMINAL/PERMISSIVE matched-bank feasibility over the "
            "CUMULATIVE pool (reuses c6_scientific.assess_profile/select_strictest_profile/"
            "c6_matched_bank.build_matched_banks unmodified -- NOT implemented by this module)",
            "if still infeasible and t < 4: open tranche t+1",
            "if infeasible at t = 4: SCIENTIFICALLY_BLOCKED, no oversampling, no threshold change",
        ],
        "not_implemented_this_task": "rendering, quality evaluation, and the tranche "
                                     "orchestration LOOP are NOT implemented by this module -- "
                                     "only the deterministic schedule/identity construction. "
                                     "Opening (rendering) the first reserve tranche requires a "
                                     "separate, later, explicit authorization.",
        "rendering_authorized": False,
        "target_access": False, "llm_api_calls": 0,
    }


def validate_reserve_protocol_lock(repo: Path) -> dict[str, Any]:
    """STRICT, read-only. Independently rebuilds the lock and compares."""
    path = reserve_protocol_lock_path(repo)
    if not path.is_file():
        return {"schema_version": f"{SCHEMA_PREFIX}-protocol-lock-validate-v1", "status": "NOT_MATERIALIZED"}
    recorded = cc.read_json(path)
    recomputed = build_reserve_protocol_lock()
    problems = [f"{key} drifted" for key in recomputed if recorded.get(key) != recomputed.get(key)]
    return {"schema_version": f"{SCHEMA_PREFIX}-protocol-lock-validate-v1",
           "status": "INVALID" if problems else "VALID", "problems": problems,
           "target_access": False, "llm_api_calls": 0}


RESERVE_PROTOCOL_FILENAME = "E7_V1_1_RESERVE_PROTOCOL.json"


def reserve_protocol_path(repo: Path) -> Path:
    return repo / E7_V1_1_RESERVE_ROOT / RESERVE_PROTOCOL_FILENAME


def write_reserve_protocol_document(repo: Path) -> dict[str, Any]:
    """The narrative protocol document -- SAME content as the frozen lock
    (`build_reserve_protocol_lock`; the "lock" is exactly a frozen snapshot
    of this same protocol definition), written under its own conventionally
    named file so the namespace carries both a human-facing protocol
    description and its frozen lock."""
    from prism_fas.utils.core import atomic_json_write

    body = build_reserve_protocol_lock()
    path = reserve_protocol_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, body)
    return body


def write_reserve_protocol_lock(repo: Path) -> dict[str, Any]:
    from prism_fas.utils.core import atomic_json_write

    body = build_reserve_protocol_lock()
    path = reserve_protocol_lock_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, body)
    validation = validate_reserve_protocol_lock(repo)
    if validation["status"] != "VALID":
        raise E7ReserveScheduleError(f"freshly-written {RESERVE_PROTOCOL_LOCK_FILENAME} FAILED "
                                     f"strict validation -- {validation['problems']!r}")
    return body
