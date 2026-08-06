from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from typing import Any
import numpy as np

SYNTHESIS_SCHEMA_VERSION = "m7-synthesis-v1"


class SynthesisError(ValueError):
    """An image, mask or operator result violates the M7 synthesis contract."""


class MaskBuildError(SynthesisError):
    """A semantic region mask could not be built deterministically."""


def validate_image(image: np.ndarray, *, name: str = "image") -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.float32: raise SynthesisError(f"{name} dtype {array.dtype} != float32")
    if array.ndim != 3 or array.shape[0] != 3: raise SynthesisError(f"{name} shape {array.shape} is not [3,H,W]")
    if not np.isfinite(array).all(): raise SynthesisError(f"{name} contains non-finite values")
    if float(array.min()) < 0.0 or float(array.max()) > 1.0: raise SynthesisError(f"{name} values outside [0,1]")
    return array


def validate_mask(mask: np.ndarray, *, name: str = "mask", height: int | None = None, width: int | None = None) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 2: array = array[None]
    if array.ndim != 3 or array.shape[0] != 1: raise SynthesisError(f"{name} shape {array.shape} is not [1,H,W]")
    if array.dtype == np.bool_: array = array.astype(np.float32)
    array = array.astype(np.float32, copy=False)
    unique = np.unique(array)
    if unique.size and not np.isin(unique, (0.0, 1.0)).all(): raise SynthesisError(f"{name} must contain exactly 0 or 1")
    if height is not None and array.shape[1] != height: raise SynthesisError(f"{name} height {array.shape[1]} != {height}")
    if width is not None and array.shape[2] != width: raise SynthesisError(f"{name} width {array.shape[2]} != {width}")
    return array


def mask_hash(mask: np.ndarray) -> str:
    """Stable hash of a binary mask: packed bits, shape-prefixed."""
    array = np.asarray(mask).astype(bool).reshape(-1)
    header = "x".join(str(dimension) for dimension in np.asarray(mask).shape).encode("utf-8")
    return hashlib.sha256(header + b"|" + np.packbits(array).tobytes()).hexdigest()


def array_hash(array: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    header = f"{values.dtype.str}|{values.shape}".encode("utf-8")
    return hashlib.sha256(header + values.tobytes()).hexdigest()


@dataclass(frozen=True)
class OperatorResult:
    """One operator's contribution. `image` is already exactly composited: every
    pixel outside `actual_support_mask` equals the operator's input pixel."""
    image: np.ndarray
    actual_support_mask: np.ndarray
    strength_map: np.ndarray
    parameters_used: dict[str, float]
    operator_seed: int
    trace: dict[str, Any]

    def validate(self, source: np.ndarray) -> "OperatorResult":
        validate_image(self.image, name="operator image")
        validate_mask(self.actual_support_mask, name="operator support", height=self.image.shape[1], width=self.image.shape[2])
        smap = np.asarray(self.strength_map, dtype=np.float32)
        if smap.shape != (1, self.image.shape[1], self.image.shape[2]): raise SynthesisError("operator strength_map must be [1,H,W]")
        if not np.isfinite(smap).all() or float(smap.min()) < 0.0 or float(smap.max()) > 1.0:
            raise SynthesisError("operator strength_map outside [0,1]")
        outside = np.asarray(self.actual_support_mask, dtype=np.float32)[0] < 0.5
        if outside.any():
            delta = np.abs(self.image - source)[:, outside]
            if delta.size and float(delta.max()) != 0.0:
                raise SynthesisError("operator altered pixels outside its declared support")
        if float(smap[0][outside].max(initial=0.0)) != 0.0:
            raise SynthesisError("operator strength_map is non-zero outside its declared support")
        return self

    def support_pixels(self) -> int: return int(np.asarray(self.actual_support_mask).astype(bool).sum())


@dataclass(frozen=True)
class RegionMaskResult:
    """Deterministic semantic-region masks for one sample/recipe pair."""
    requested_region_mask: np.ndarray
    operator_support_mask: np.ndarray
    per_region_masks: dict[str, np.ndarray]
    region_sources: dict[str, str]
    requested_coverage: float
    achieved_coverage: float
    mask_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "RegionMaskResult":
        height, width = self.requested_region_mask.shape[1], self.requested_region_mask.shape[2]
        validate_mask(self.requested_region_mask, name="requested_region_mask")
        validate_mask(self.operator_support_mask, name="operator_support_mask", height=height, width=width)
        for name, mask in self.per_region_masks.items():
            validate_mask(mask, name=f"region[{name}]", height=height, width=width)
        if int(np.asarray(self.operator_support_mask).astype(bool).sum()) == 0:
            raise MaskBuildError("operator support mask is empty")
        return self

    def support_pixels(self) -> int: return int(np.asarray(self.operator_support_mask).astype(bool).sum())
    def requested_pixels(self) -> int: return int(np.asarray(self.requested_region_mask).astype(bool).sum())


@dataclass(frozen=True)
class PhysicsResult:
    """Output of `PhysicsEngine.apply` for one (sample, compiled recipe) pair."""
    synthetic_image: np.ndarray
    exact_edit_mask: np.ndarray
    artifact_strength_map: np.ndarray
    requested_region_mask: np.ndarray
    per_operator_support_masks: dict[str, np.ndarray]
    recipe_id: str
    recipe_hash: str
    graph_hash: str
    sample_id: str
    trace: dict[str, Any]
    output_hashes: dict[str, str]

    def validate(self) -> "PhysicsResult":
        validate_image(self.synthetic_image, name="synthetic_image")
        height, width = self.synthetic_image.shape[1], self.synthetic_image.shape[2]
        validate_mask(self.exact_edit_mask, name="exact_edit_mask", height=height, width=width)
        validate_mask(self.requested_region_mask, name="requested_region_mask", height=height, width=width)
        strength = np.asarray(self.artifact_strength_map, dtype=np.float32)
        if strength.shape != (1, height, width): raise SynthesisError("artifact_strength_map must be [1,H,W]")
        if not np.isfinite(strength).all() or float(strength.min()) < 0.0 or float(strength.max()) > 1.0:
            raise SynthesisError("artifact_strength_map outside [0,1]")
        outside = np.asarray(self.exact_edit_mask, dtype=np.float32)[0] < 0.5
        if outside.any() and float(strength[0][outside].max(initial=0.0)) != 0.0:
            raise SynthesisError("artifact_strength_map is non-zero outside the exact edit mask")
        return self

    def changed_pixels(self) -> int: return int(np.asarray(self.exact_edit_mask).astype(bool).sum())
