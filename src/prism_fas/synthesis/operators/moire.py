from __future__ import annotations
from typing import Any
import numpy as np
from .base import PhysicsOperator, coordinate_grid


class MoireOperator(PhysicsOperator):
    """Interference between the display lattice and the camera sensor grid.

    Built from two bounded sinusoidal fields whose frequencies differ by the
    compiled ratio; their product is the visible beat pattern. Moire is not
    applied to every recipe, and a recipe declaring `always_moire` forbidden
    cannot use moire as its only artifact (enforced in recipe validation).
    """
    name = "moire"
    support_policy = "requested_region"
    amplitude = 0.30

    def _transform(self, image: np.ndarray, support: np.ndarray, strength: float, capture: dict[str, Any],
                   rng: np.random.Generator, parameters: dict[str, float]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        height, width = image.shape[1], image.shape[2]
        u, v = coordinate_grid(height, width)
        ratio = float(np.clip(parameters.get("frequency_ratio", 1.08), 1.01, 1.6))
        base_cycles = float(rng.uniform(9.0, 26.0))
        angle_a = float(rng.uniform(0.0, np.pi))
        angle_b = angle_a + float(rng.uniform(0.02, 0.25))
        phase_a, phase_b = float(rng.uniform(0.0, 2.0 * np.pi)), float(rng.uniform(0.0, 2.0 * np.pi))
        first = (u * np.float32(np.cos(angle_a)) + v * np.float32(np.sin(angle_a))) * np.float32(base_cycles)
        second = (u * np.float32(np.cos(angle_b)) + v * np.float32(np.sin(angle_b))) * np.float32(base_cycles * ratio)
        field = (np.sin(2.0 * np.pi * first + np.float32(phase_a)) *
                 np.sin(2.0 * np.pi * second + np.float32(phase_b))).astype(np.float32)
        delta = np.float32(strength * self.amplitude) * field
        return image + delta[None, :, :], support, {
            "frequency_ratio": round(ratio, 6), "base_cycles": round(base_cycles, 4),
            "angles_radians": [round(angle_a, 6), round(angle_b, 6)],
            "phase": [round(phase_a, 6), round(phase_b, 6)], "amplitude": self.amplitude}
