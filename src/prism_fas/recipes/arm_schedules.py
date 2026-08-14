"""Frozen 384-slot generation schedules for the RND and DET control arms.

v1.5 §7.8.5. Both control arms produce exactly 384 offline candidate slots under
frozen deterministic schedules, using the same ontology, item schema, ranges,
route policy, compiler eligibility and target firewall as the LLM arm. Neither
generator may depend on LLM output or on any target information.

This module defines and freezes the SCHEDULES — canonical slot ids, the seed
family and the sampling contract — and can materialize candidate payloads from
them. It does not, by itself, constitute C3 scientific generation: producing the
scientific archives is a separate authorized action that writes immutable
artifacts under the frozen bank contract.

Determinism is per-slot, not per-run. Every slot's value is derived from
SHA-256(schedule_seed_material | slot_id), so slot 137 is identical whether it is
generated alone, in order, in reverse, or in parallel. No global RNG is ever
consulted.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from prism_fas.recipes.ontology import Ontology

SCHEDULE_VERSION = "prism_c3_arm_schedule_v1"
SLOTS_PER_ARM = 384

#: Frozen seed family. Source-independent constants: they encode nothing about
#: any dataset, target or measured result.
SEED_FAMILY: dict[str, int] = {"RND": 20260812, "DET": 20260813}

#: The scientific route contract every arm must declare (§7.3.1).
SCIENTIFIC_ROUTE: tuple[str, ...] = ("physics", "gpat")


class ScheduleError(ValueError):
    """A schedule is malformed or disagrees with the ontology."""


def slot_ids(arm: str, count: int = SLOTS_PER_ARM) -> list[str]:
    """Canonical slot ids, frozen order: `RND_SLOT_000` … `RND_SLOT_383`."""
    if arm not in SEED_FAMILY:
        raise ScheduleError(f"unknown control arm {arm!r}; expected one of {sorted(SEED_FAMILY)}")
    return [f"{arm}_SLOT_{index:03d}" for index in range(count)]


def _digest(arm: str, slot_id: str, field: str) -> int:
    """Per-slot, per-field deterministic integer. No global RNG."""
    material = f"{SCHEDULE_VERSION}|{arm}|{SEED_FAMILY[arm]}|{slot_id}|{field}"
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def _unit(arm: str, slot_id: str, field: str) -> float:
    """Deterministic value in [0, 1) with fixed precision."""
    return (_digest(arm, slot_id, field) % 10**9) / 10**9


def _pick(arm: str, slot_id: str, field: str, options: Sequence[str]) -> str:
    return options[_digest(arm, slot_id, field) % len(options)]


def _bounded(arm: str, slot_id: str, field: str, low: float, high: float,
             decimals: int = 4) -> float:
    return round(low + (high - low) * _unit(arm, slot_id, field), decimals)


@dataclass(frozen=True)
class ArmSchedule:
    """One control arm's frozen generation schedule."""

    arm: str
    version: str
    slots: int
    seed: int
    slot_ids: tuple[str, ...]
    strategy: str
    ontology_identity: str
    route: tuple[str, ...] = SCIENTIFIC_ROUTE

    def identity_material(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "arm": self.arm,
            "slots": self.slots,
            "seed": self.seed,
            "strategy": self.strategy,
            "route": list(self.route),
            "ontology_identity": self.ontology_identity,
            "slot_id_pattern": f"{self.arm}_SLOT_%03d",
            "first_slot_id": self.slot_ids[0],
            "last_slot_id": self.slot_ids[-1],
            "slot_id_list_sha256": hashlib.sha256(
                "\n".join(self.slot_ids).encode("utf-8")).hexdigest(),
            "derivation": "SHA-256(version|arm|seed|slot_id|field); per-slot, order-independent",
            "target_information_used": False,
            "llm_output_used": False,
        }

    @property
    def schedule_identity(self) -> str:
        text = json.dumps(self.identity_material(), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {**self.identity_material(), "schedule_identity": self.schedule_identity}


def build_schedule(arm: str, ontology: Ontology, *, slots: int = SLOTS_PER_ARM) -> ArmSchedule:
    strategy = {
        "RND": "rule-valid random sampling over the frozen ontology, ranges and compatibility "
               "tables; every draw derived per slot from the frozen seed family",
        "DET": "deterministic structured enumeration over the frozen ontology in a fixed "
               "category order; no randomness beyond the frozen per-slot derivation",
    }[arm]
    return ArmSchedule(arm=arm, version=SCHEDULE_VERSION, slots=slots, seed=SEED_FAMILY[arm],
                       slot_ids=tuple(slot_ids(arm, slots)), strategy=strategy,
                       ontology_identity=ontology.sha256)


# ------------------------------------------------------------ candidate drafting
def _compatible_artifacts(ontology: Ontology, medium: str) -> tuple[str, ...]:
    return ontology.artifacts_for_medium(medium)


def _compatible_regions(ontology: Ontology, shape: str) -> tuple[str, ...]:
    return ontology.regions_for_geometry(shape)


def _fit_severity_budget(specs: list[dict[str, Any]], ontology: Ontology,
                         budget: float) -> None:
    """Clamp total artifact strength to the frozen budget, in place.

    Proportional scaling followed by rounding can land just above the budget,
    which the validator rejects. Floor to canonical precision, hold every
    artifact at or above its safe minimum, and shave the largest above-minimum
    artifact if needed. The ontology self-check guarantees the sum of minima for
    the maximum artifact count fits, so this terminates.
    """
    import math

    step = 1e-4
    for _ in range(10_000):
        total = round(sum(spec["strength"] for spec in specs), 6)
        if total <= budget:
            return
        scale = budget / total
        shaved = False
        for spec in specs:
            band = ontology.strength_range(spec["name"])
            floored = math.floor(spec["strength"] * scale * 10_000) / 10_000
            new = max(band.minimum, floored)
            if new < spec["strength"]:
                spec["strength"] = round(new, 4)
                shaved = True
        if shaved:
            continue
        candidates = [spec for spec in specs
                      if spec["strength"] > ontology.strength_range(spec["name"]).minimum]
        if not candidates:
            return
        target = max(candidates, key=lambda spec: spec["strength"])
        target["strength"] = round(target["strength"] - step, 4)


def draft_candidate(arm: str, slot_id: str, ontology: Ontology) -> dict[str, Any]:
    """Deterministically draft one candidate payload for a slot.

    The draft respects the ontology's compatibility tables and safe ranges, and
    always declares the scientific route. It is still subject to the SAME common
    eligibility gate as the LLM arm — nothing here bypasses validation, and a
    draft that fails is rejected like any other candidate (§7.8.5).
    """
    limits = ontology.limits
    index = int(slot_id.rsplit("_", 1)[1])

    if arm == "DET":
        # Structured enumeration: walk the category grid in frozen order so the
        # arm covers the ontology systematically rather than randomly.
        medium = ontology.media[index % len(ontology.media)]
        shape = ontology.geometry_shapes[(index // len(ontology.media))
                                         % len(ontology.geometry_shapes)]
        illumination = ontology.illumination[(index // (len(ontology.media)
                                                        * len(ontology.geometry_shapes)))
                                             % len(ontology.illumination)]
    else:
        medium = _pick(arm, slot_id, "medium", ontology.media)
        shape = _pick(arm, slot_id, "geometry", ontology.geometry_shapes)
        illumination = _pick(arm, slot_id, "illumination", ontology.illumination)

    allowed_artifacts = _compatible_artifacts(ontology, medium)
    allowed_regions = _compatible_regions(ontology, shape)

    max_artifacts = min(int(limits["max_artifacts_per_recipe"]), len(allowed_artifacts))
    max_regions = min(int(limits["max_regions_per_recipe"]), len(allowed_regions))
    n_artifacts = 1 + (_digest(arm, slot_id, "n_artifacts") % max_artifacts)
    n_regions = 1 + (_digest(arm, slot_id, "n_regions") % max_regions)

    def take(options: Sequence[str], how_many: int, field: str) -> list[str]:
        ordered = sorted(options, key=lambda name: _digest(arm, slot_id, f"{field}:{name}"))
        return ordered[:how_many]

    artifacts = take(allowed_artifacts, n_artifacts, "artifact")
    regions = ontology.sort_regions(take(allowed_regions, n_regions, "region"))

    # Severity: inside each artifact's safe band, and scaled so the total cannot
    # exceed the frozen budget.
    budget = float(limits["max_total_artifact_strength"])
    specs = []
    for name in artifacts:
        band = ontology.strength_range(name)
        specs.append({"name": name,
                      "strength": _bounded(arm, slot_id, f"strength:{name}",
                                           band.minimum, band.maximum)})
    _fit_severity_budget(specs, ontology, budget)

    capture_ranges = ontology.capture_ranges
    medium_ranges = ontology.medium_ranges
    geometry_ranges = ontology.geometry_ranges
    return {
        "schema_version": ontology.recipe_schema_version,
        "medium": {
            "family": medium,
            "transparency": _bounded(arm, slot_id, "transparency",
                                     medium_ranges["transparency"].minimum,
                                     medium_ranges["transparency"].maximum),
            "roughness": _bounded(arm, slot_id, "roughness",
                                  medium_ranges["roughness"].minimum,
                                  medium_ranges["roughness"].maximum),
        },
        "geometry": {
            "shape": shape,
            "rigidity": _bounded(arm, slot_id, "rigidity",
                                 geometry_ranges["rigidity"].minimum,
                                 geometry_ranges["rigidity"].maximum),
            "coverage": _bounded(arm, slot_id, "coverage",
                                 geometry_ranges["coverage"].minimum,
                                 geometry_ranges["coverage"].maximum),
        },
        "regions": regions,
        "artifacts": specs,
        "capture": {
            "yaw": _bounded(arm, slot_id, "yaw", capture_ranges["yaw"].minimum,
                            capture_ranges["yaw"].maximum, decimals=2),
            "illumination": illumination,
            "compression_q": int(capture_ranges["compression_q"].minimum
                                 + _digest(arm, slot_id, "compression_q")
                                 % (int(capture_ranges["compression_q"].maximum)
                                    - int(capture_ranges["compression_q"].minimum) + 1)),
            "scale": _bounded(arm, slot_id, "scale", capture_ranges["scale"].minimum,
                              capture_ranges["scale"].maximum, decimals=3),
            "motion": _bounded(arm, slot_id, "motion", capture_ranges["motion"].minimum,
                               capture_ranges["motion"].maximum, decimals=3),
            "defocus": _bounded(arm, slot_id, "defocus", capture_ranges["defocus"].minimum,
                                capture_ranges["defocus"].maximum, decimals=3),
        },
        "forbidden_shortcuts": [],
        "generator_route": list(SCIENTIFIC_ROUTE),
        "seed": _digest(arm, slot_id, "seed") % (int(limits["seed_max"]) + 1),
    }


def draft_schedule(arm: str, ontology: Ontology, *,
                   slots: int = SLOTS_PER_ARM) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (slot_id, candidate payload) for the whole frozen schedule."""
    for slot_id in slot_ids(arm, slots):
        yield slot_id, draft_candidate(arm, slot_id, ontology)


__all__ = ["SCHEDULE_VERSION", "SLOTS_PER_ARM", "SEED_FAMILY", "SCIENTIFIC_ROUTE",
           "ScheduleError", "ArmSchedule", "slot_ids", "build_schedule", "draft_candidate",
           "draft_schedule"]
