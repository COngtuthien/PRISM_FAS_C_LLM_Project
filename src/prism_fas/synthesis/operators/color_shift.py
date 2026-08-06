from __future__ import annotations
from typing import Any
import numpy as np
from .base import PhysicsOperator


class ColorShiftOperator(PhysicsOperator):
    """Bounded per-channel gain and colour-temperature transform.

    The channel direction is drawn from the node-local RNG and the temperature
    axis from the compiled parameter, so the operator does not impose one fixed
    colour cast across every recipe.
    """
    name = "color_shift"
    support_policy = "requested_region"
    channel_gain = 0.25
    temperature_gain = 0.20

    def _transform(self, image: np.ndarray, support: np.ndarray, strength: float, capture: dict[str, Any],
                   rng: np.random.Generator, parameters: dict[str, float]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        direction = np.asarray(rng.uniform(-1.0, 1.0, size=3), dtype=np.float32)
        norm = float(np.abs(direction).max())
        direction = direction / np.float32(norm) if norm > 1e-6 else np.zeros(3, dtype=np.float32)
        temperature = float(np.clip(parameters.get("temperature", 0.0), -1.0, 1.0))
        axis = np.asarray([1.0, 0.0, -1.0], dtype=np.float32)
        gain = (np.float32(1.0) + np.float32(strength) *
                (np.float32(self.channel_gain) * direction + np.float32(self.temperature_gain * temperature) * axis))
        transformed = image * gain[:, None, None]
        return transformed.astype(np.float32), support, {
            "channel_gain": [round(float(value), 6) for value in gain],
            "direction": [round(float(value), 6) for value in direction],
            "temperature": round(temperature, 6)}
