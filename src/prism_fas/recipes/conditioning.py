from __future__ import annotations
import hashlib
from typing import Any
import numpy as np
from .ontology import Ontology
from .schema import RecipeV11

CONDITIONING_VERSION = "recipe_conditioning_v1"
CONDITIONING_DIM = 41
# Fixed block layout. Blocks are concatenated in this order and never reordered:
#   medium 5 | geometry 6 | regions 9 | artifact strengths 8 | illumination 6
#   | yaw 1 | compression 1 | scale 1 | motion 1 | defocus 1 | routes 2  = 41
CONDITIONING_BLOCKS = (("medium", 5), ("geometry", 6), ("region", 9), ("artifact_strength", 8),
                       ("illumination", 6), ("yaw", 1), ("compression", 1), ("scale", 1),
                       ("motion", 1), ("defocus", 1), ("route", 2))


class ConditioningError(ValueError):
    """A recipe carries a category the fixed conditioning layout cannot encode."""


def feature_names(ontology: Ontology) -> tuple[str, ...]:
    """Stable feature-name list. An ontology that changed a category order or
    count would change this list, and the saved SHA would stop matching."""
    names: list[str] = []
    names += [f"medium={value}" for value in ontology.media]
    names += [f"geometry={value}" for value in ontology.geometry_shapes]
    names += [f"region={value}" for value in ontology.regions]
    names += [f"artifact_strength={value}" for value in ontology.artifacts]
    names += [f"illumination={value}" for value in ontology.illumination]
    names += ["capture.yaw_normalized", "capture.compression_normalized", "capture.scale_normalized",
              "capture.motion", "capture.defocus"]
    names += [f"route={value}" for value in ontology.routes]
    resolved = tuple(names)
    if len(resolved) != CONDITIONING_DIM:
        raise ConditioningError(f"ontology yields {len(resolved)} conditioning features, expected {CONDITIONING_DIM}; "
                                f"the fixed {CONDITIONING_VERSION} layout cannot absorb a changed category count")
    return resolved


def feature_names_sha256(ontology: Ontology) -> str:
    return hashlib.sha256("\n".join(feature_names(ontology)).encode("utf-8")).hexdigest()


def _index(values: tuple[str, ...], value: str, block: str) -> int:
    try: return values.index(value)
    except ValueError:
        raise ConditioningError(f"unknown {block} category {value!r}; the fixed conditioning layout must fail rather "
                                f"than shift indices") from None


def normalize_yaw(yaw: float, ontology: Ontology) -> float:
    return float(np.clip(float(yaw) / max(ontology.max_abs_yaw(), 1e-9), -1.0, 1.0))


def normalize_compression(value: float, ontology: Ontology) -> float:
    band = ontology.capture_ranges["compression_q"]
    span = max(band.maximum - band.minimum, 1e-9)
    return float(np.clip((float(value) - band.minimum) / span, 0.0, 1.0))


def normalize_scale(value: float, ontology: Ontology) -> float:
    band = ontology.capture_ranges["scale"]
    span = max(band.maximum - band.minimum, 1e-9)
    return float(np.clip(2.0 * (float(value) - band.minimum) / span - 1.0, -1.0, 1.0))


def conditioning_vector(recipe: RecipeV11, ontology: Ontology) -> np.ndarray:
    """Fixed float32 [41] conditioning vector for later M8 GPAT conditioning.

    M7 only produces and freezes it; no generator consumes it yet.
    """
    names = feature_names(ontology)  # also enforces the 41-dimension invariant
    vector = np.zeros(CONDITIONING_DIM, dtype=np.float32)
    offset = 0
    vector[offset + _index(ontology.media, recipe.medium.family, "medium")] = 1.0
    offset += len(ontology.media)
    vector[offset + _index(ontology.geometry_shapes, recipe.geometry.shape, "geometry")] = 1.0
    offset += len(ontology.geometry_shapes)
    for region in recipe.regions:
        vector[offset + _index(ontology.regions, region, "region")] = 1.0
    offset += len(ontology.regions)
    for spec in recipe.artifacts:
        vector[offset + _index(ontology.artifacts, spec.name, "artifact")] = np.float32(spec.strength)
    offset += len(ontology.artifacts)
    vector[offset + _index(ontology.illumination, recipe.capture.illumination, "illumination")] = 1.0
    offset += len(ontology.illumination)
    vector[offset + 0] = np.float32(normalize_yaw(recipe.capture.yaw, ontology))
    vector[offset + 1] = np.float32(normalize_compression(recipe.capture.compression_q, ontology))
    vector[offset + 2] = np.float32(normalize_scale(recipe.capture.scale, ontology))
    vector[offset + 3] = np.float32(recipe.capture.motion)
    vector[offset + 4] = np.float32(recipe.capture.defocus)
    offset += 5
    for route in recipe.generator_route:
        vector[offset + _index(ontology.routes, route, "route")] = 1.0
    offset += len(ontology.routes)
    if offset != CONDITIONING_DIM: raise ConditioningError(f"conditioning layout produced {offset} slots, expected {CONDITIONING_DIM}")
    if vector.shape != (CONDITIONING_DIM,) or vector.dtype != np.float32:
        raise ConditioningError(f"conditioning vector must be float32 [{CONDITIONING_DIM}]")
    if not np.isfinite(vector).all(): raise ConditioningError("conditioning vector is not finite")
    assert len(names) == CONDITIONING_DIM
    return vector


def decode_conditioning(vector: np.ndarray, ontology: Ontology) -> dict[str, Any]:
    """Audit helper: explain every non-zero component of a conditioning vector."""
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vector.shape != (CONDITIONING_DIM,): raise ConditioningError(f"expected a [{CONDITIONING_DIM}] vector")
    names = feature_names(ontology)
    blocks: dict[str, Any] = {}
    offset = 0
    for block, width in CONDITIONING_BLOCKS:
        segment = vector[offset:offset + width]
        blocks[block] = {names[offset + i]: round(float(segment[i]), 6) for i in range(width) if float(segment[i]) != 0.0}
        offset += width
    return {"conditioning_version": CONDITIONING_VERSION, "dimension": CONDITIONING_DIM,
            "feature_names_sha256": feature_names_sha256(ontology), "active": blocks,
            "nonzero": int((vector != 0).sum())}
