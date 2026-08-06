from __future__ import annotations
from typing import Any
import numpy as np
from ..contracts import OperatorResult, SynthesisError, validate_image, validate_mask

# Illumination direction unit vectors in normalized image coordinates
# (u to the right, v downward). `mixed` keeps a deterministic diagonal.
ILLUMINATION_DIRECTION = {"front": (0.0, 0.0), "left": (-1.0, 0.0), "right": (1.0, 0.0),
                          "top": (0.0, -1.0), "bottom": (0.0, 1.0), "mixed": (0.7071, -0.7071)}


def coordinate_grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalized pixel-centre coordinates in [0,1); deterministic and
    resolution independent so an operator's look does not depend on H,W."""
    v = ((np.arange(height, dtype=np.float32) + 0.5) / np.float32(height))[:, None]
    u = ((np.arange(width, dtype=np.float32) + 0.5) / np.float32(width))[None, :]
    return np.broadcast_to(u, (height, width)).astype(np.float32), np.broadcast_to(v, (height, width)).astype(np.float32)


def gaussian_kernel1d(sigma: float) -> np.ndarray:
    sigma = float(max(sigma, 1e-3))
    radius = int(max(1, np.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(offsets ** 2) / np.float32(2.0 * sigma * sigma)).astype(np.float32)
    return (kernel / kernel.sum()).astype(np.float32)


def _convolve1d(array: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = kernel.shape[0] // 2
    padding = [(0, 0)] * array.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(array, padding, mode="reflect")
    out = np.zeros_like(array, dtype=np.float32)
    for offset in range(kernel.shape[0]):
        index = [slice(None)] * array.ndim
        index[axis] = slice(offset, offset + array.shape[axis])
        out += np.float32(kernel[offset]) * padded[tuple(index)]
    return out.astype(np.float32)


def gaussian_blur(array: np.ndarray, sigma: float) -> np.ndarray:
    """Separable float32 Gaussian with reflect padding. Implemented here rather
    than through OpenCV so the numerics cannot drift with a library version."""
    values = np.asarray(array, dtype=np.float32)
    if sigma <= 0: return values.copy()
    kernel = gaussian_kernel1d(sigma)
    return _convolve1d(_convolve1d(values, kernel, values.ndim - 1), kernel, values.ndim - 2)


def _shift2d(array: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
    """Edge-replicating shift: never wraps content from the opposite border."""
    values = np.asarray(array, dtype=np.float32)
    pad_y, pad_x = abs(int(shift_y)), abs(int(shift_x))
    padding = [(0, 0)] * values.ndim
    padding[values.ndim - 2] = (pad_y, pad_y)
    padding[values.ndim - 1] = (pad_x, pad_x)
    padded = np.pad(values, padding, mode="edge")
    index = [slice(None)] * values.ndim
    index[values.ndim - 2] = slice(pad_y - int(shift_y), pad_y - int(shift_y) + values.shape[values.ndim - 2])
    index[values.ndim - 1] = slice(pad_x - int(shift_x), pad_x - int(shift_x) + values.shape[values.ndim - 1])
    return padded[tuple(index)].astype(np.float32)


def _shift2d_subpixel(array: np.ndarray, shift_y: float, shift_x: float) -> np.ndarray:
    """Bilinear sub-pixel shift built from four integer shifts.

    Integer rounding would collapse every short streak onto the same lattice
    offsets and silently discard the motion angle, so the taps are interpolated.
    """
    base_y, base_x = int(np.floor(shift_y)), int(np.floor(shift_x))
    fraction_y, fraction_x = np.float32(shift_y - base_y), np.float32(shift_x - base_x)
    top_left = _shift2d(array, base_y, base_x)
    top_right = _shift2d(array, base_y, base_x + 1)
    bottom_left = _shift2d(array, base_y + 1, base_x)
    bottom_right = _shift2d(array, base_y + 1, base_x + 1)
    top = top_left * (np.float32(1.0) - fraction_x) + top_right * fraction_x
    bottom = bottom_left * (np.float32(1.0) - fraction_x) + bottom_right * fraction_x
    return (top * (np.float32(1.0) - fraction_y) + bottom * fraction_y).astype(np.float32)


def directional_blur(array: np.ndarray, length: float, angle_radians: float) -> np.ndarray:
    """Deterministic motion blur: a normalized line kernel sampled at sub-pixel
    positions along `angle_radians`, with edge replication at the border."""
    values = np.asarray(array, dtype=np.float32)
    taps = int(max(1, round(float(length))))
    if taps <= 1: return values.copy()
    dx, dy = float(np.cos(angle_radians)), float(np.sin(angle_radians))
    offsets = np.linspace(-(float(length) - 1.0) / 2.0, (float(length) - 1.0) / 2.0, taps, dtype=np.float32)
    out = np.zeros_like(values, dtype=np.float32)
    for offset in offsets:
        out += _shift2d_subpixel(values, float(offset) * dy, float(offset) * dx)
    return (out / np.float32(taps)).astype(np.float32)


def _disk_offsets(radius: int) -> list[tuple[int, int]]:
    radius = int(radius)
    return [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)
            if dy * dy + dx * dx <= radius * radius]


def binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    values = np.asarray(mask).astype(bool)
    radius = int(radius)
    if radius <= 0: return values.copy()
    height, width = values.shape
    padded = np.pad(values, ((radius, radius), (radius, radius)), mode="constant", constant_values=False)
    out = np.zeros_like(values)
    for dy, dx in _disk_offsets(radius):
        out |= padded[radius + dy:radius + dy + height, radius + dx:radius + dx + width]
    return out


def binary_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    """Erosion with an outside-is-background border, so a region touching the
    crop edge is treated as bounded by the crop."""
    values = np.asarray(mask).astype(bool)
    radius = int(radius)
    if radius <= 0: return values.copy()
    return ~binary_dilate(~values, radius)


def boundary_band(mask: np.ndarray, radius: int) -> np.ndarray:
    """Deterministic ring straddling the edge of `mask`."""
    values = np.asarray(mask).astype(bool)
    radius = max(1, int(radius))
    band = binary_dilate(values, radius) & ~binary_erode(values, radius)
    return band if band.any() else values.copy()


def composite(original: np.ndarray, transformed: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Exact support composite: pixels outside `support` keep their input bytes."""
    mask = np.asarray(support).astype(bool)
    if mask.ndim == 3: mask = mask[0]
    out = np.where(mask[None, :, :], np.asarray(transformed, dtype=np.float32), np.asarray(original, dtype=np.float32))
    return np.ascontiguousarray(out.astype(np.float32))


def clamp01(array: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(array, dtype=np.float32), np.float32(0.0), np.float32(1.0)).astype(np.float32)


def luminance(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    return (np.float32(0.299) * values[0] + np.float32(0.587) * values[1] + np.float32(0.114) * values[2]).astype(np.float32)


class PhysicsOperator:
    """Base class for the eight deterministic M7 physics operators.

    Subclasses implement `_transform`, which returns the full-frame transformed
    image plus the support it actually acted on. The base class performs the
    exact composite, builds the strength map and validates the result, so no
    operator can silently modify a pixel outside its declared support.
    """
    name: str = "operator"
    support_policy: str = "requested_region"

    def apply(self, image: np.ndarray, mask: np.ndarray, strength: float, capture: dict[str, Any],
              rng: np.random.Generator, parameters: dict[str, float] | None = None,
              *, seed: int = 0) -> OperatorResult:
        source = validate_image(image, name=f"{self.name} input")
        support_in = validate_mask(mask, name=f"{self.name} input mask", height=source.shape[1], width=source.shape[2])
        strength = float(strength)
        if not 0.0 <= strength <= 1.0: raise SynthesisError(f"{self.name}: strength {strength} outside [0,1]")
        params = dict(parameters or {})
        base = support_in[0].astype(bool)
        transformed, support, trace = self._transform(source, base, strength, capture, rng, params)
        support = np.asarray(support).astype(bool)
        if support.shape != base.shape: raise SynthesisError(f"{self.name}: support shape {support.shape} != {base.shape}")
        transformed = clamp01(transformed)
        output = composite(source, transformed, support)
        strength_map = (support.astype(np.float32) * np.float32(min(max(strength, 0.0), 1.0)))[None]
        result = OperatorResult(image=output, actual_support_mask=support.astype(np.float32)[None],
                                strength_map=strength_map, parameters_used={key: float(value) for key, value in sorted(params.items())},
                                operator_seed=int(seed),
                                trace={"operator": self.name, "support_policy": self.support_policy,
                                       "strength": round(strength, 6), "support_pixels": int(support.sum()), **trace})
        return result.validate(source)

    def _transform(self, image: np.ndarray, support: np.ndarray, strength: float, capture: dict[str, Any],
                   rng: np.random.Generator, parameters: dict[str, float]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        raise NotImplementedError
