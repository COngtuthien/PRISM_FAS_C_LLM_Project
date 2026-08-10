"""The strict JSON Schema sent to the provider as the structured-output contract.

Built from the frozen ontology so the enums can never drift from the validator.
The schema is a request-side constraint, not the authority: whatever the provider
returns is re-validated locally by `prism_fas.recipes.validate`, which owns the
ranges, the compatibility rules and the severity budget. A provider that ignores
the schema produces a rejected candidate, never an accepted one.

Two fields are deliberately absent from the candidate object:

* `recipe_id`  - assigned by the system. Spec section 7.3: recipe id, content hash,
  provenance and validation status are computed, never trusted from the model.
* `parameters` - per-artifact operator overrides. The planner chooses artifacts,
  regions, media, geometry, capture and severity; operator internals stay with
  the compiler and the ontology.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from prism_fas.recipes.ontology import Ontology

CANDIDATE_SCHEMA_VERSION = "c1-llm-candidate-schema-v1"


def _range(ontology_range: Any) -> dict[str, float]:
    return {"minimum": float(ontology_range.minimum), "maximum": float(ontology_range.maximum)}


def candidate_object_schema(ontology: Ontology) -> dict[str, Any]:
    """One candidate recipe, exactly as the model must emit it."""
    limits = ontology.limits
    medium_ranges = ontology.medium_ranges
    geometry_ranges = ontology.geometry_ranges
    capture_ranges = ontology.capture_ranges
    strength_floor = min(band.minimum for band in ontology.artifact_strength_ranges.values())
    strength_ceiling = max(band.maximum for band in ontology.artifact_strength_ranges.values())

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "medium", "geometry", "regions", "artifacts",
                     "capture", "forbidden_shortcuts", "generator_route", "seed"],
        "properties": {
            "schema_version": {"type": "string", "enum": [ontology.recipe_schema_version]},
            "medium": {
                "type": "object",
                "additionalProperties": False,
                "required": ["family", "transparency", "roughness"],
                "properties": {
                    "family": {"type": "string", "enum": list(ontology.media)},
                    "transparency": {"type": "number", **_range(medium_ranges["transparency"])},
                    "roughness": {"type": "number", **_range(medium_ranges["roughness"])},
                },
            },
            "geometry": {
                "type": "object",
                "additionalProperties": False,
                "required": ["shape", "rigidity", "coverage"],
                "properties": {
                    "shape": {"type": "string", "enum": list(ontology.geometry_shapes)},
                    "rigidity": {"type": "number", **_range(geometry_ranges["rigidity"])},
                    "coverage": {"type": "number", **_range(geometry_ranges["coverage"])},
                },
            },
            "regions": {
                "type": "array",
                "minItems": int(limits["min_regions_per_recipe"]),
                "maxItems": int(limits["max_regions_per_recipe"]),
                "items": {"type": "string", "enum": list(ontology.regions)},
            },
            "artifacts": {
                "type": "array",
                "minItems": int(limits["min_artifacts_per_recipe"]),
                "maxItems": int(limits["max_artifacts_per_recipe"]),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "strength"],
                    "properties": {
                        "name": {"type": "string", "enum": list(ontology.artifacts)},
                        "strength": {"type": "number", "minimum": strength_floor,
                                     "maximum": strength_ceiling},
                    },
                },
            },
            "capture": {
                "type": "object",
                "additionalProperties": False,
                "required": ["yaw", "illumination", "compression_q", "scale", "motion", "defocus"],
                "properties": {
                    "yaw": {"type": "number", **_range(capture_ranges["yaw"])},
                    "illumination": {"type": "string", "enum": list(ontology.illumination)},
                    "compression_q": {"type": "integer", **_range(capture_ranges["compression_q"])},
                    "scale": {"type": "number", **_range(capture_ranges["scale"])},
                    "motion": {"type": "number", **_range(capture_ranges["motion"])},
                    "defocus": {"type": "number", **_range(capture_ranges["defocus"])},
                },
            },
            "forbidden_shortcuts": {
                "type": "array",
                "maxItems": len(ontology.forbidden_shortcuts),
                "items": {"type": "string", "enum": list(ontology.forbidden_shortcuts)},
            },
            "generator_route": {
                "type": "array",
                "minItems": 1,
                "maxItems": len(ontology.routes),
                "items": {"type": "string", "enum": list(ontology.routes)},
            },
            "seed": {"type": "integer", "minimum": int(limits["seed_min"]),
                     "maximum": int(limits["seed_max"])},
        },
    }


def candidate_json_schema(ontology: Ontology, *, recipes_requested: int,
                          array_bounds: bool = True) -> dict[str, Any]:
    """The batch envelope: exactly `recipes_requested` candidate objects.

    `array_bounds=False` omits `minItems`/`maxItems`. C2B finding, observed live
    on 2026-08-10: the provider answers `400 INVALID_ARGUMENT` for this envelope
    once the bound reaches 32, while the byte-identical schema with the bound at
    1 is accepted. Google documents array length limits as a source of schema
    complexity rejection, and the two envelopes differ by nothing else - 2695 vs
    2697 bytes, identical item schema.

    Dropping the bound costs nothing scientifically. The request-side schema was
    never the authority: `RecipePlanner.validate_response` rejects a response
    whose recipe count is not exactly `recipes_requested`, so "exactly 32" is
    still enforced locally, on the response, where it always was.
    """
    if not isinstance(recipes_requested, int) or isinstance(recipes_requested, bool) or recipes_requested < 1:
        raise ValueError("recipes_requested must be a positive integer")
    recipes: dict[str, Any] = {"type": "array"}
    if array_bounds:
        recipes["minItems"] = recipes_requested
        recipes["maxItems"] = recipes_requested
    recipes["items"] = candidate_object_schema(ontology)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["recipes"],
        "properties": {"recipes": recipes},
    }


def json_schema_identity(schema: dict[str, Any]) -> str:
    """SHA-256 over the canonical schema text. Part of the request identity, so a
    changed schema invalidates downstream scientific generation."""
    text = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
