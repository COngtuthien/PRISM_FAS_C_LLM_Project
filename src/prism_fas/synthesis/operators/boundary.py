from __future__ import annotations
from typing import Any
import numpy as np
from .base import PhysicsOperator, boundary_band, gaussian_blur


class BoundaryInconsistencyOperator(PhysicsOperator):
    """Seam where a re-presented surface meets real skin.

    Unlike the other operators this one acts on an explicit deterministic edge
    band straddling the requested region, so its actual support is the band and
    not the region interior. Pixels outside that band are still bit-identical to
    the input.
    """
    name = "boundary_inconsistency"
    support_policy = "boundary_band"
    offset_gain = 0.18

    def _transform(self, image: np.ndarray, support: np.ndarray, strength: float, capture: dict[str, Any],
                   rng: np.random.Generator, parameters: dict[str, float]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        band_px = float(np.clip(parameters.get("band_px", 3.0), 1.0, 12.0))
        radius = int(max(1, round(band_px * (0.5 + 0.5 * float(strength)))))
        band = boundary_band(support, radius)
        sign = 1.0 if float(rng.random()) < 0.5 else -1.0
        seam_sigma = float(np.clip(0.8 + 1.2 * float(strength), 0.2, 4.0))
        blurred = gaussian_blur(image, seam_sigma)
        # Soft ramp across the band so the seam is graded, not a hard step.
        weight = gaussian_blur(band.astype(np.float32), max(radius / 2.0, 0.5))
        peak = float(weight.max())
        weight = (weight / np.float32(peak)) if peak > 1e-6 else weight
        offset = np.float32(sign * self.offset_gain * float(strength))
        blend = np.float32(np.clip(float(strength), 0.0, 1.0))
        transformed = image * (np.float32(1.0) - blend) + blurred * blend + offset * weight[None, :, :]
        return transformed.astype(np.float32), band, {
            "band_px": round(band_px, 6), "band_radius_px": radius, "seam_sign": sign,
            "seam_sigma_px": round(seam_sigma, 6), "seam_blend": round(float(blend), 6),
            "offset_gain": self.offset_gain, "band_pixels": int(band.sum())}
