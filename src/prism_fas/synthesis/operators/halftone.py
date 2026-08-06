from __future__ import annotations
from typing import Any
import numpy as np
from .base import PhysicsOperator, coordinate_grid, luminance


class HalftoneOperator(PhysicsOperator):
    """Printed-ink dot screen.

    A rotated cross-screen modulates mid-tones most (the printable dot area is
    largest where luminance is mid-grey). Orientation, frequency and phase all
    come from the compiled parameters and the node-local RNG, so no fixed
    always-on halftone signature can be learned from the operator itself.
    """
    name = "halftone"
    support_policy = "requested_region"
    amplitude = 0.35

    def _transform(self, image: np.ndarray, support: np.ndarray, strength: float, capture: dict[str, Any],
                   rng: np.random.Generator, parameters: dict[str, float]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        height, width = image.shape[1], image.shape[2]
        u, v = coordinate_grid(height, width)
        frequency = float(parameters.get("screen_frequency", 0.12))
        angle = np.deg2rad(float(parameters.get("screen_angle_deg", 45.0)) + float(rng.uniform(-7.5, 7.5)))
        cycles = float(max(2.0, frequency * min(height, width)))
        phase_a, phase_b = float(rng.uniform(0.0, 2.0 * np.pi)), float(rng.uniform(0.0, 2.0 * np.pi))
        cos, sin = np.float32(np.cos(angle)), np.float32(np.sin(angle))
        first = (u * cos + v * sin) * np.float32(cycles)
        second = (-u * sin + v * cos) * np.float32(cycles)
        screen = (np.sin(2.0 * np.pi * first + np.float32(phase_a)) *
                  np.sin(2.0 * np.pi * second + np.float32(phase_b))).astype(np.float32)
        grey = luminance(image)
        dot_area = (np.float32(4.0) * grey * (np.float32(1.0) - grey)).astype(np.float32)
        delta = np.float32(strength * self.amplitude) * screen * dot_area
        return image + delta[None, :, :], support, {
            "screen_frequency": round(frequency, 6), "screen_cycles": round(cycles, 4),
            "screen_angle_radians": round(float(angle), 6), "phase": [round(phase_a, 6), round(phase_b, 6)],
            "amplitude": self.amplitude}
