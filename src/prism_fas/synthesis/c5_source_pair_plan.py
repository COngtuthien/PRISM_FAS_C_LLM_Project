"""C5_SOURCE_PAIR_PLAN_V1 — the frozen Version-C synthesis schedule.

The Version-B M8 planner (`candidate_plan.py`, 1120 rows) is keyed on LIVE
SAMPLES and has no arm dimension. Version-C C5 is a different experiment: §10.4
fixes the budget per RECIPE at 2048 renders/arm = 256 recipes x 8 renders, with
exactly 4 Physics and 4 GPAT per recipe, across three arms. The legacy planner is
left exactly as it is; nothing here imports or modifies it.

Two properties this module exists to guarantee.

**The schedule is arm-independent.** Which live sample a position renders, which
route it takes, and which spoof source a GPAT position pairs with are all
functions of the global position alone. RND, DET and LLM therefore differ only in
recipe CONTENT, which is the treatment under test. If the arm could influence the
source assignment, an acceptance-rate difference at C6 would be uninterpretable —
exactly the Version-B confound §11.3 exists to remove.

**Generation identity does not bind the C6 quality gate.** The frozen stage
boundary puts rendering at C5 and gating at C6, and C6 selects its profile from
three preregistered candidates AFTER the candidates exist. So a C5 candidate
identity binds only what can change its pixels. Changing a C6 threshold changes
which candidates are accepted; it must never change which candidates exist, what
they are called, or what they contain. The Version-B `SyntheticBankGenerator`
binds three calibration hashes into its generation identity; that class and its
identities are untouched, and this module is the Version-C path that does not.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

SCHEMA_VERSION = "prism-c5-source-pair-plan-v1"
PLAN_NAME = "C5_SOURCE_PAIR_PLAN_V1"
PLAN_SEED = 20260806

#: §10.4. 256 recipes x 8 renders = 2048 candidates per arm.
RECIPES_PER_ARM = 256
RENDERS_PER_RECIPE = 8
CANDIDATES_PER_ARM = RECIPES_PER_ARM * RENDERS_PER_RECIPE

#: The three treatment arms, in their frozen order.
ARMS: tuple[str, ...] = ("RND", "DET", "LLM")

PHYSICS, GPAT = "physics", "gpat"
PHYSICS_NONE = "physics-none"
SAME_DOMAIN, CROSS_DOMAIN = "same_domain", "cross_domain"

#: The frozen render-slot schedule. Even slots are Physics, odd slots are GPAT,
#: which yields exactly 4 and 4 per recipe. The GPAT slots alternate same-domain
#: and cross-domain, giving exactly 2 and 2.
ROUTE_BY_SLOT: tuple[str, ...] = (PHYSICS, GPAT, PHYSICS, GPAT,
                                  PHYSICS, GPAT, PHYSICS, GPAT)
DOMAIN_RELATION_BY_SLOT: dict[int, str] = {1: SAME_DOMAIN, 3: CROSS_DOMAIN,
                                           5: SAME_DOMAIN, 7: CROSS_DOMAIN}

SOURCE_SPLIT = "source_train"
LIVE, SPOOF = "live", "spoof"

#: Named so a reader can see what the digest is a digest OF, and so a change to
#: the assignment rule changes the plan identity.
ASSIGNMENT_ALGORITHM = "c5_position_modulo_live_list_v1"
SPOOF_SELECTION_ALGORITHM = "c5_position_keyed_eligible_index_v1"


class SourcePairPlanError(ValueError):
    """The frozen C5 schedule cannot be built as declared."""


def _digest(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _list_identity(sample_ids: Sequence[str]) -> str:
    """Identity over an ORDERED list, so a reordering is a different list."""
    return hashlib.sha256(
        json.dumps(list(sample_ids), separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceRow:
    sample_id: str
    dataset: str
    source_record_id: str
    subject_id: str | None
    label: str


def load_source_rows(package_root: Path) -> tuple[list[SourceRow], list[SourceRow]]:
    """The ordered live and spoof lists, from `manifests/source_train.parquet` only.

    `source_dev` and every target manifest are never opened. The sort is by
    `sample_id` ascending, which is stable and independent of row order on disk,
    so two machines build the same schedule.
    """
    import pyarrow.parquet as pq

    manifest = Path(package_root) / "manifests" / f"{SOURCE_SPLIT}.parquet"
    if not manifest.is_file():
        raise SourcePairPlanError(f"missing {SOURCE_SPLIT} manifest at {manifest}")
    table = pq.read_table(manifest).to_pydict()

    rows: list[SourceRow] = []
    for index in range(len(table["sample_id"])):
        split = table["project_split"][index]
        if split != SOURCE_SPLIT:
            raise SourcePairPlanError(
                f"row {index} carries project_split {split!r}; C5 reads "
                f"{SOURCE_SPLIT} and nothing else")
        subject = table["subject_id"][index]
        rows.append(SourceRow(
            sample_id=str(table["sample_id"][index]),
            dataset=str(table["dataset"][index]),
            source_record_id=str(table["source_record_id"][index]),
            subject_id=str(subject) if subject not in (None, "") else None,
            label=str(table["label_live_spoof"][index])))

    live = sorted((row for row in rows if row.label == LIVE), key=lambda r: r.sample_id)
    spoof = sorted((row for row in rows if row.label == SPOOF), key=lambda r: r.sample_id)
    if not live:
        raise SourcePairPlanError(f"{SOURCE_SPLIT} carries no live rows")
    if not spoof:
        raise SourcePairPlanError(f"{SOURCE_SPLIT} carries no spoof rows")
    return live, spoof


def live_for_position(live_list: Sequence[SourceRow], position: int) -> SourceRow:
    """`LIVE_LIST[p % len(LIVE_LIST)]`. A function of the position alone.

    Walking the ordered list by position spreads exposure as evenly as a fixed
    schedule can: over 2048 consecutive positions every live sample is used
    either floor(2048/N) or ceil(2048/N) times, so no live identity is
    over-represented in one arm relative to another.
    """
    return live_list[position % len(live_list)]


def route_for_slot(slot: int) -> str:
    if not 0 <= slot < RENDERS_PER_RECIPE:
        raise SourcePairPlanError(f"slot {slot} is outside 0..{RENDERS_PER_RECIPE - 1}")
    return ROUTE_BY_SLOT[slot]


def domain_relation_for_slot(slot: int) -> str | None:
    return DOMAIN_RELATION_BY_SLOT.get(slot)


def eligible_spoof_sources(spoof_list: Sequence[SourceRow], live: SourceRow,
                           relation: str) -> list[SourceRow]:
    """Every spoof row this live target may legitimately be paired with.

    Three constraints, none of which may be relaxed to fill a pool: the spoof
    must come from a different source record, from a different subject whenever
    both subject ids are known, and from the domain the slot's relation names.
    """
    same_domain = relation == SAME_DOMAIN
    eligible = [
        row for row in spoof_list
        if row.source_record_id != live.source_record_id
        and (live.subject_id is None or row.subject_id is None
             or row.subject_id != live.subject_id)
        and ((row.dataset == live.dataset) if same_domain
             else (row.dataset != live.dataset))
    ]
    return sorted(eligible, key=lambda row: row.sample_id)


def spoof_for_position(spoof_list: Sequence[SourceRow], live: SourceRow,
                       position: int, relation: str, *,
                       seed: int = PLAN_SEED) -> SourceRow:
    """The frozen GPAT spoof source for one global position.

    Keyed on the position, the live id and the relation — never on the arm or on
    anything from the recipe — so RND, DET and LLM pair the same position with
    the same spoof source. An empty pool fails closed rather than dropping a
    constraint.
    """
    eligible = eligible_spoof_sources(spoof_list, live, relation)
    if not eligible:
        raise SourcePairPlanError(
            f"no eligible {relation} spoof source for live {live.sample_id} at "
            f"position {position}: the pool is empty after the different-record "
            "and different-subject rules. C5 fails closed rather than relaxing a "
            "constraint or choosing another render policy.")
    key = _digest(PLAN_NAME, int(seed), int(position), live.sample_id, relation)
    return eligible[int(key[:16], 16) % len(eligible)]


def build_source_pair_plan(package_root: Path, *,
                           seed: int = PLAN_SEED) -> dict[str, Any]:
    """The BASE schedule: 2048 positions, identical for every arm.

    It carries no arm, no recipe bank and no recipe content on purpose — that is
    what makes it provably the same schedule for all three arms.
    """
    live_list, spoof_list = load_source_rows(package_root)
    package_identity = json.loads(
        (Path(package_root) / "PACKAGE_LOCK.json").read_text(encoding="utf-8")
    )["content_identity_sha256"]

    positions: list[dict[str, Any]] = []
    for recipe_ordinal in range(RECIPES_PER_ARM):
        for slot in range(RENDERS_PER_RECIPE):
            position = RENDERS_PER_RECIPE * recipe_ordinal + slot
            live = live_for_position(live_list, position)
            route = route_for_slot(slot)
            relation = domain_relation_for_slot(slot)
            spoof = (spoof_for_position(spoof_list, live, position, relation, seed=seed)
                     if route == GPAT else None)
            positions.append({
                "position": position, "recipe_ordinal": recipe_ordinal, "slot": slot,
                "route": route, "domain_relation": relation,
                "live_target_sample_id": live.sample_id,
                "live_dataset": live.dataset,
                "live_source_record_id": live.source_record_id,
                "spoof_source_sample_id": spoof.sample_id if spoof else None,
                "spoof_dataset": spoof.dataset if spoof else None,
                "spoof_source_record_id": spoof.source_record_id if spoof else None,
            })

    _assert_schedule(positions)
    return {
        "schema_version": SCHEMA_VERSION, "plan_name": PLAN_NAME, "plan_seed": int(seed),
        "package_identity": package_identity,
        "live_list_identity": _list_identity([row.sample_id for row in live_list]),
        "live_list_size": len(live_list),
        "spoof_list_identity": _list_identity([row.sample_id for row in spoof_list]),
        "spoof_list_size": len(spoof_list),
        "recipes_per_arm": RECIPES_PER_ARM, "renders_per_recipe": RENDERS_PER_RECIPE,
        "positions_per_arm": CANDIDATES_PER_ARM,
        "route_schedule": list(ROUTE_BY_SLOT),
        "domain_relation_schedule": {str(slot): relation for slot, relation
                                     in sorted(DOMAIN_RELATION_BY_SLOT.items())},
        "assignment_algorithm": ASSIGNMENT_ALGORITHM,
        "spoof_selection_algorithm": SPOOF_SELECTION_ALGORITHM,
        "eligibility_rules": {
            "source_split": SOURCE_SPLIT,
            "different_source_record": True,
            "different_subject_when_available": True,
            "domain_relation_enforced": True,
            "empty_pool": "fail_closed"},
        "arm_independent": True,
        "binds_quality_calibration": False,
        "positions": positions,
        "source_only": {"source_dev_opened": False, "target_test_opened": False,
                        "target_labels_opened": False, "target_access": 0},
    }


def _assert_schedule(positions: Sequence[dict[str, Any]]) -> None:
    """The frozen cardinalities, checked on the plan rather than promised."""
    if len(positions) != CANDIDATES_PER_ARM:
        raise SourcePairPlanError(
            f"expected {CANDIDATES_PER_ARM} positions, built {len(positions)}")
    by_route: dict[str, int] = {}
    per_recipe: dict[int, list[str]] = {}
    for row in positions:
        by_route[row["route"]] = by_route.get(row["route"], 0) + 1
        per_recipe.setdefault(row["recipe_ordinal"], []).append(row["route"])
    expected = CANDIDATES_PER_ARM // 2
    if by_route.get(PHYSICS) != expected or by_route.get(GPAT) != expected:
        raise SourcePairPlanError(f"route counts are not {expected}/{expected}: {by_route}")
    for ordinal, routes in per_recipe.items():
        if routes.count(PHYSICS) != 4 or routes.count(GPAT) != 4:
            raise SourcePairPlanError(
                f"recipe ordinal {ordinal} has {routes.count(PHYSICS)} Physics and "
                f"{routes.count(GPAT)} GPAT renders; §10.4 fixes both at 4")
    for row in positions:
        if row["route"] == GPAT:
            if row["live_source_record_id"] == row["spoof_source_record_id"]:
                raise SourcePairPlanError(
                    f"position {row['position']} pairs a live and a spoof from the "
                    "same source record")


#: Fields excluded from the plan identity: they describe the plan rather than
#: determine it. The positions themselves are covered by the schedule fields and
#: the two list identities, so a different assignment is a different identity.
IDENTITY_EXCLUDED_FIELDS = ("positions", "source_only")


def source_pair_plan_identity(plan: dict[str, Any]) -> str:
    material = {key: value for key, value in plan.items()
                if key not in IDENTITY_EXCLUDED_FIELDS}
    material["positions_digest"] = _digest(*[
        f"{row['position']}:{row['route']}:{row['live_target_sample_id']}:"
        f"{row['spoof_source_sample_id'] or PHYSICS_NONE}"
        for row in plan["positions"]])
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode("utf-8")).hexdigest()


def arm_candidate_plan_identity(*, source_pair_plan_identity: str, arm: str,
                                recipe_bank_identity: str,
                                gpat_checkpoint_sha256: str,
                                physics_engine_version: str,
                                ontology_identity: str) -> str:
    """One arm's plan: the shared base schedule plus that arm's own inputs.

    The base schedule identity is an INPUT here rather than something recomputed,
    which is what makes "the arms share a schedule" checkable: three arm
    identities that differ must still name the same base.
    """
    if arm not in ARMS:
        raise SourcePairPlanError(f"unknown arm {arm!r}; the frozen arms are {ARMS}")
    return _digest(SCHEMA_VERSION, source_pair_plan_identity, arm,
                   recipe_bank_identity, gpat_checkpoint_sha256,
                   physics_engine_version, ontology_identity)


def candidate_identity(*, source_pair_plan_identity: str, arm: str,
                       recipe_bank_identity: str, recipe_id: str,
                       recipe_ordinal: int, slot: int, position: int, route: str,
                       live_target_sample_id: str,
                       spoof_source_sample_id: str | None,
                       package_identity: str, ontology_identity: str,
                       generator_binding: str, seed: int = PLAN_SEED) -> str:
    """`c5syn_<24 hex>` over everything that can change this candidate's pixels.

    `generator_binding` is the C4 winning GPAT checkpoint SHA-256 on the GPAT
    route and the PhysicsEngine version on the Physics route — each route binds
    the generator that actually produced it and not the other one.

    Deliberately ABSENT: every C6 quality-gate field. `threshold_sha256`,
    `fingerprint_reference_sha256`, `calibration_sha256` and the selected profile
    describe how a candidate is JUDGED, not how it is MADE, and C6 chooses them
    after these candidates exist. Binding them would make the candidate set a
    function of the gate and create a C5 -> C6 -> C5 cycle.

    Also absent: paths, filenames, timestamps, subject ids and anything from the
    target.
    """
    if route not in (PHYSICS, GPAT):
        raise SourcePairPlanError(f"unknown route {route!r}")
    return "c5syn_" + _digest(
        SCHEMA_VERSION, source_pair_plan_identity, arm, recipe_bank_identity,
        recipe_id, int(recipe_ordinal), int(slot), int(position), route,
        live_target_sample_id, spoof_source_sample_id or PHYSICS_NONE,
        package_identity, ontology_identity, generator_binding, int(seed))[:24]


__all__ = ["SCHEMA_VERSION", "PLAN_NAME", "PLAN_SEED", "ARMS", "RECIPES_PER_ARM",
           "RENDERS_PER_RECIPE", "CANDIDATES_PER_ARM", "ROUTE_BY_SLOT",
           "DOMAIN_RELATION_BY_SLOT", "PHYSICS", "GPAT", "PHYSICS_NONE",
           "SAME_DOMAIN", "CROSS_DOMAIN", "SOURCE_SPLIT", "SourceRow",
           "SourcePairPlanError", "load_source_rows", "live_for_position",
           "route_for_slot", "domain_relation_for_slot", "eligible_spoof_sources",
           "spoof_for_position", "build_source_pair_plan",
           "source_pair_plan_identity", "arm_candidate_plan_identity",
           "candidate_identity"]
