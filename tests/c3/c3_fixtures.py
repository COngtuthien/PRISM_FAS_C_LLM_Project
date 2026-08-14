"""Deterministic offline fixtures for the C3 selection tests.

Everything here is synthetic and generated from the ontology. No provider, no
network, no dataset, no target information.

The selector's real budget is 384 -> >=320 -> 256, which is far too slow to solve
repeatedly in a unit suite. The tests therefore exercise the SAME code path at a
scaled-down bank size with scaled quotas, plus separate exact-arithmetic tests
that use the real constants. Scaling is a test-harness device only: the shipped
constants in `prism_fas.recipes.selection` are never modified.
"""
from __future__ import annotations

import hashlib
from typing import Any, Sequence

from prism_fas.recipes import selection as sel
from prism_fas.recipes.ontology import Ontology
from prism_fas.recipes.schema import RecipeV11
from prism_fas.recipes.validate import validate_payload


def _digest(*parts: Any) -> int:
    material = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def fit_severity_budget(specs: list[dict[str, Any]], ontology: Ontology) -> None:
    """Clamp total artifact strength to the frozen budget, in place.

    Scaling then rounding can land a whisker above the budget, which the
    validator rejects. Floor to the canonical precision, keep every artifact at
    or above its safe minimum, and shave the largest above-minimum artifact until
    the total fits. The ontology self-check guarantees the sum of minima for the
    maximum artifact count is within budget, so this terminates.
    """
    import math

    budget = float(ontology.limits["max_total_artifact_strength"])
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
        # Every artifact is at its minimum under the proportional scale; take the
        # largest one down a single step instead.
        candidates = [spec for spec in specs
                      if spec["strength"] > ontology.strength_range(spec["name"]).minimum]
        if not candidates:
            return
        target = max(candidates, key=lambda spec: spec["strength"])
        target["strength"] = round(target["strength"] - step, 4)


def make_payload(ontology: Ontology, index: int, *, tag: str = "fx",
                 medium: str | None = None, geometry: str | None = None,
                 illumination: str | None = None,
                 artifacts: Sequence[str] | None = None,
                 regions: Sequence[str] | None = None,
                 route: Sequence[str] | None = None) -> dict[str, Any]:
    """A schema/ontology/compatibility-valid candidate payload.

    Category choices walk the ontology grid so a generated pool covers every
    category, which is what the hard quotas need.
    """
    media = ontology.media
    shapes = ontology.geometry_shapes
    lights = ontology.illumination

    # Strides chosen so each axis cycles through every category quickly: a pool
    # that never shows an illumination mode cannot satisfy a per-category floor,
    # and the tests would then be measuring the fixture rather than the selector.
    family = medium if medium is not None else media[index % len(media)]
    shape = geometry if geometry is not None else shapes[(index // 5) % len(shapes)]
    light = (illumination if illumination is not None
             else lights[(index // 3) % len(lights)])

    allowed_artifacts = list(ontology.artifacts_for_medium(family))
    allowed_regions = list(ontology.regions_for_geometry(shape))

    def rotate(options: list[str], offset: int, how_many: int) -> list[str]:
        start = offset % len(options)
        return [options[(start + step) % len(options)] for step in range(how_many)]

    if artifacts is None:
        count = 1 + (_digest(tag, index, "na") % min(3, len(allowed_artifacts)))
        picked = rotate(allowed_artifacts, index, count)
    else:
        picked = list(artifacts)

    if regions is None:
        count = 1 + (_digest(tag, index, "nr") % min(3, len(allowed_regions)))
        chosen = rotate(allowed_regions, index, count)
    else:
        chosen = list(regions)
    chosen = ontology.sort_regions(chosen)

    specs = []
    for name in picked:
        band = ontology.strength_range(name)
        span = band.maximum - band.minimum
        value = band.minimum + span * ((_digest(tag, index, "s", name) % 1000) / 1000.0)
        specs.append({"name": name, "strength": round(value, 4)})
    fit_severity_budget(specs, ontology)

    capture = ontology.capture_ranges
    return {
        "schema_version": ontology.recipe_schema_version,
        "medium": {"family": family,
                   "transparency": round((_digest(tag, index, "t") % 1000) / 1000.0, 4),
                   "roughness": round((_digest(tag, index, "ro") % 1000) / 1000.0, 4)},
        "geometry": {"shape": shape,
                     "rigidity": round((_digest(tag, index, "ri") % 1000) / 1000.0, 4),
                     "coverage": round(0.10 + 0.90 * ((_digest(tag, index, "c") % 1000)
                                                      / 1000.0), 4)},
        "regions": chosen,
        "artifacts": specs,
        "capture": {
            "yaw": round(-45.0 + 90.0 * ((_digest(tag, index, "y") % 1000) / 1000.0), 2),
            "illumination": light,
            "compression_q": 30 + (_digest(tag, index, "q") % 71),
            "scale": round(0.75 + 0.50 * ((_digest(tag, index, "sc") % 1000) / 1000.0), 3),
            "motion": round((_digest(tag, index, "mo") % 1000) / 1000.0, 3),
            "defocus": round((_digest(tag, index, "df") % 1000) / 1000.0, 3),
        },
        "forbidden_shortcuts": [],
        "generator_route": list(route) if route is not None
        else list(sel.__dict__.get("SCIENTIFIC_ROUTE", ("physics", "gpat"))),
        "seed": _digest(tag, index, "seed") % 2_147_483_647,
    }


def make_recipe(ontology: Ontology, index: int, **kwargs: Any) -> RecipeV11:
    """A parsed, fully valid recipe. Raises if the fixture is not valid."""
    payload = make_payload(ontology, index, **kwargs)
    payload["recipe_id"] = f"R-{index:06d}"
    recipe, issues = validate_payload(payload, ontology, canonicalize=False)
    if recipe is None or issues:
        raise AssertionError(f"fixture {index} is not valid: {issues}")
    return recipe


def make_pool(ontology: Ontology, size: int, *, tag: str = "fx") -> list[RecipeV11]:
    """A pool of distinct valid recipes covering every ontology category."""
    return [make_recipe(ontology, index, tag=tag) for index in range(size)]


def scaled_quotas(bank_size: int, ontology: Ontology) -> dict[str, dict[str, int | None]]:
    """Quotas scaled to a small bank, preserving the real shape.

    Real contract at 256: medium 32/80, geometry 24/64, illumination 24/64,
    artifact 8/128 (pref 32), region 8/128 (pref 24). Scaling keeps each axis
    satisfiable at `bank_size` so the harness exercises real feasibility logic.
    """
    factor = bank_size / sel.FINAL_BANK_SIZE_PER_ARM
    def scale(value: int | None, floor: int = 1) -> int | None:
        return None if value is None else max(floor, int(value * factor))

    return {
        "medium": {"hard_min": scale(32), "hard_max": bank_size, "preferred_min": None},
        "geometry": {"hard_min": scale(24), "hard_max": bank_size, "preferred_min": None},
        "illumination": {"hard_min": scale(24), "hard_max": bank_size, "preferred_min": None},
        "artifact": {"hard_min": scale(8), "hard_max": bank_size * 3,
                     "preferred_min": scale(32)},
        "region": {"hard_min": scale(8), "hard_max": bank_size * 3,
                   "preferred_min": scale(24)},
    }
