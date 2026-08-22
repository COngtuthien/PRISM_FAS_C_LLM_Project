"""The three C5 arm candidate plans, built on the one frozen base schedule.

C5's treatment banks are the three FROZEN C3 banks — RND, DET and LLM, 256
selected recipes each. `prism_recipe_bank_m7_v1` is not one of them: M7 is the
neutral support bank the shared GPAT generator was TRAINED on, and using it as a
treatment arm would compare the generator against itself.

Every arm consumes the same `C5_SOURCE_PAIR_PLAN_V1` object. The base schedule
takes no arm, no recipe bank and no recipe content, so the position → live map,
the route sequence and the GPAT spoof pairing are identical across arms by
construction rather than by convention. What differs is the 256 recipes, which is
the treatment under test.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .c5_source_pair_plan import (ARMS, CANDIDATES_PER_ARM, GPAT, PHYSICS,
                                  RECIPES_PER_ARM, SourcePairPlanError,
                                  arm_candidate_plan_identity, candidate_identity,
                                  source_pair_plan_identity)

SCHEMA_VERSION = "prism-c5-arm-candidate-plan-v1"

#: The C3 treatment banks, by arm. Directory names are lowercase on disk.
C3_BANK_ROOT = "assets/recipe_banks/c3"

#: Named so a test can assert it is refused rather than merely absent.
NEUTRAL_SUPPORT_BANK = "assets/recipe_banks/prism_recipe_bank_m7_v1"


class ArmPlanError(ValueError):
    """An arm candidate plan cannot be built as declared."""


def arm_bank_root(repo: Path, arm: str) -> Path:
    if arm not in ARMS:
        raise ArmPlanError(f"unknown arm {arm!r}; the frozen arms are {ARMS}")
    return Path(repo) / C3_BANK_ROOT / arm.lower()


def load_arm_bank(repo: Path, arm: str) -> dict[str, Any]:
    """One frozen C3 treatment bank, in its canonical frozen recipe order.

    The order matters: recipe ordinal r is an index into this list, and the base
    schedule pairs ordinal r with a fixed set of positions. A reordered bank is a
    different experiment.
    """
    root = arm_bank_root(repo, arm)
    lock_path, recipes_path = root / "C3_BANK.json", root / "recipes.jsonl"
    if not lock_path.is_file() or not recipes_path.is_file():
        raise ArmPlanError(f"{root.as_posix()} is not a frozen C3 arm bank")

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if str(lock.get("arm")) != arm:
        raise ArmPlanError(
            f"{root.as_posix()} declares arm {lock.get('arm')!r}, not {arm!r}")
    if not lock.get("scientific_eligible", False):
        raise ArmPlanError(f"the {arm} C3 bank is not scientifically eligible")

    lines = [line for line in recipes_path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if len(lines) != RECIPES_PER_ARM:
        raise ArmPlanError(
            f"the {arm} C3 bank holds {len(lines)} recipes; §10.4 fixes the "
            f"selected bank at {RECIPES_PER_ARM}")

    recipes = [json.loads(line) for line in lines]
    return {"arm": arm, "root": root, "lock": lock, "recipes": recipes,
            "bank_identity": str(lock["bank_identity"]),
            "selected_set_identity": str(lock.get("selected_set_identity", "")),
            "ontology_identity": str(lock.get("ontology_identity", ""))}


def _recipe_id(recipe: dict[str, Any], ordinal: int) -> str:
    for key in ("recipe_id", "id", "recipe_hash"):
        if recipe.get(key):
            return str(recipe[key])
    raise ArmPlanError(f"recipe at ordinal {ordinal} carries no identifier")


def build_arm_plan(repo: Path, arm: str, base_plan: dict[str, Any], *,
                   gpat_checkpoint_sha256: str,
                   physics_engine_version: str) -> dict[str, Any]:
    """One arm's 2048 candidates, over the shared base schedule.

    `base_plan` is passed in rather than rebuilt so all three arms provably share
    one object — and its identity is an input to this plan's identity, so three
    differing arm identities still name the same base.
    """
    bank = load_arm_bank(repo, arm)
    base_identity = source_pair_plan_identity(base_plan)
    positions = base_plan["positions"]
    if len(positions) != CANDIDATES_PER_ARM:
        raise ArmPlanError(
            f"the base schedule holds {len(positions)} positions, not {CANDIDATES_PER_ARM}")

    plan_identity = arm_candidate_plan_identity(
        source_pair_plan_identity=base_identity, arm=arm,
        recipe_bank_identity=bank["bank_identity"],
        gpat_checkpoint_sha256=gpat_checkpoint_sha256,
        physics_engine_version=physics_engine_version,
        ontology_identity=bank["ontology_identity"])

    rows: list[dict[str, Any]] = []
    for row in positions:
        ordinal = int(row["recipe_ordinal"])
        recipe_id = _recipe_id(bank["recipes"][ordinal], ordinal)
        # Each route binds the generator that actually produced it, never both.
        binding = (gpat_checkpoint_sha256 if row["route"] == GPAT
                   else physics_engine_version)
        rows.append({
            **row, "arm": arm, "recipe_id": recipe_id,
            "recipe_bank_identity": bank["bank_identity"],
            "generator_binding": binding,
            "candidate_id": candidate_identity(
                source_pair_plan_identity=base_identity, arm=arm,
                recipe_bank_identity=bank["bank_identity"], recipe_id=recipe_id,
                recipe_ordinal=ordinal, slot=int(row["slot"]),
                position=int(row["position"]), route=row["route"],
                live_target_sample_id=row["live_target_sample_id"],
                spoof_source_sample_id=row["spoof_source_sample_id"],
                package_identity=base_plan["package_identity"],
                ontology_identity=bank["ontology_identity"],
                generator_binding=binding)})

    _assert_arm_plan(rows, arm)
    return {
        "schema_version": SCHEMA_VERSION, "arm": arm,
        "arm_plan_identity": plan_identity,
        "source_pair_plan_identity": base_identity,
        "package_identity": base_plan["package_identity"],
        "recipe_bank_identity": bank["bank_identity"],
        "recipe_bank_root": bank["root"].relative_to(Path(repo)).as_posix(),
        "selected_set_identity": bank["selected_set_identity"],
        "ontology_identity": bank["ontology_identity"],
        "gpat_checkpoint_sha256": gpat_checkpoint_sha256,
        "physics_engine_version": physics_engine_version,
        "planned_candidates": len(rows),
        "binds_quality_calibration": False,
        "candidates": rows,
    }


def _assert_arm_plan(rows: Sequence[dict[str, Any]], arm: str) -> None:
    if len(rows) != CANDIDATES_PER_ARM:
        raise ArmPlanError(f"{arm} planned {len(rows)} candidates, not {CANDIDATES_PER_ARM}")
    ids = [row["candidate_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ArmPlanError(f"{arm} produced duplicate candidate ids")
    routes = [row["route"] for row in rows]
    half = CANDIDATES_PER_ARM // 2
    if routes.count(PHYSICS) != half or routes.count(GPAT) != half:
        raise ArmPlanError(f"{arm} route split is not {half}/{half}")
    per_recipe: dict[int, int] = {}
    for row in rows:
        per_recipe[row["recipe_ordinal"]] = per_recipe.get(row["recipe_ordinal"], 0) + 1
    if set(per_recipe.values()) != {8} or len(per_recipe) != RECIPES_PER_ARM:
        raise ArmPlanError(f"{arm} does not render exactly 8 candidates per recipe")


def build_all_arm_plans(repo: Path, base_plan: dict[str, Any], *,
                        gpat_checkpoint_sha256: str,
                        physics_engine_version: str) -> dict[str, dict[str, Any]]:
    """All three arms over ONE base plan, then the cross-arm agreement check."""
    plans = {arm: build_arm_plan(repo, arm, base_plan,
                                 gpat_checkpoint_sha256=gpat_checkpoint_sha256,
                                 physics_engine_version=physics_engine_version)
             for arm in ARMS}
    assert_arms_share_the_schedule(plans)
    return plans


def assert_arms_share_the_schedule(plans: dict[str, dict[str, Any]]) -> None:
    """The fairness invariant, checked rather than assumed.

    If the arm could influence which live sample or spoof source a position
    receives, a C6 acceptance-rate difference would be uninterpretable. This is
    the Version-B confound §11.3 exists to remove.
    """
    if set(plans) != set(ARMS):
        raise ArmPlanError(f"expected plans for {ARMS}, got {sorted(plans)}")
    signatures = {
        arm: [(row["position"], row["route"], row["domain_relation"],
               row["live_target_sample_id"], row["spoof_source_sample_id"])
              for row in plan["candidates"]]
        for arm, plan in plans.items()}
    reference = signatures[ARMS[0]]
    for arm in ARMS[1:]:
        if signatures[arm] != reference:
            differing = next(index for index, (left, right)
                             in enumerate(zip(reference, signatures[arm]))
                             if left != right)
            raise ArmPlanError(
                f"the {arm} schedule differs from {ARMS[0]} at position "
                f"{differing}; the base schedule must be arm-independent")
    bases = {plan["source_pair_plan_identity"] for plan in plans.values()}
    if len(bases) != 1:
        raise ArmPlanError(f"the arms name {len(bases)} different base schedules")
    identities = {plan["arm_plan_identity"] for plan in plans.values()}
    if len(identities) != len(ARMS):
        raise ArmPlanError("two arms produced the same arm plan identity")


def global_candidate_count(plans: dict[str, dict[str, Any]]) -> int:
    return sum(plan["planned_candidates"] for plan in plans.values())


__all__ = ["SCHEMA_VERSION", "C3_BANK_ROOT", "NEUTRAL_SUPPORT_BANK", "ArmPlanError",
           "arm_bank_root", "load_arm_bank", "build_arm_plan", "build_all_arm_plans",
           "assert_arms_share_the_schedule", "global_candidate_count"]
