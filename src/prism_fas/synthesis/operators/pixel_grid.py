from __future__ import annotations
from typing import Any
import numpy as np
from .base import PhysicsOperator


class PixelGridOperator(PhysicsOperator):
    """Emissive-panel pixel lattice (screen-door effect).

    A dark inter-pixel lattice at the compiled period plus an RGB sub-pixel
    stripe. The period is bounded by the ontology and the lattice phase is
    drawn from the node-local RNG.
    """
    name = "pixel_grid"
    support_policy = "requested_region"
    line_darkening = 0.50
    subpixel_gain = 0.12

    def _transform(self, image: np.ndarray, support: np.ndarray, strength: float, capture: dict[str, Any],
                   rng: np.random.Generator, parameters: dict[str, float]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        height, width = image.shape[1], image.shape[2]
        period = float(np.clip(parameters.get("period_px", 3.0) * float(parameters.get("capture_scale", 1.0)), 2.0, 12.0))
        phase_x, phase_y = float(rng.uniform(0.0, period)), float(rng.uniform(0.0, period))
        columns = np.arange(width, dtype=np.float32)[None, :]
        rows = np.arange(height, dtype=np.float32)[:, None]
        fx = np.mod(columns + np.float32(phase_x), np.float32(period)) / np.float32(period)
        fy = np.mod(rows + np.float32(phase_y), np.float32(period)) / np.float32(period)
        # distance to the nearest lattice line, normalized so 0 sits on the line
        dx = np.clip(np.minimum(fx, 1.0 - fx) * np.float32(period), 0.0, 1.0).astype(np.float32)
        dy = np.clip(np.minimum(fy, 1.0 - fy) * np.float32(period), 0.0, 1.0).astype(np.float32)
        lattice = (np.float32(1.0) - np.minimum(dx, dy)).astype(np.float32)
        darken = (np.float32(1.0) - np.float32(strength * self.line_darkening) * lattice).astype(np.float32)
        stripe_index = np.mod(columns + np.float32(phase_x), np.float32(3.0)).astype(np.float32)
        channels = []
        for channel in range(3):
            weight = np.clip(np.float32(1.0) - np.abs(stripe_index - np.float32(channel)), 0.0, 1.0).astype(np.float32)
            gain = np.float32(1.0) + np.float32(strength * self.subpixel_gain) * (weight - np.float32(1.0 / 3.0))
            channels.append(image[channel] * darken * gain)
        return np.stack(channels, axis=0).astype(np.float32), support, {
            "period_px": round(period, 6), "phase_px": [round(phase_x, 6), round(phase_y, 6)],
            "line_darkening": self.line_darkening, "subpixel_gain": self.subpixel_gain}
