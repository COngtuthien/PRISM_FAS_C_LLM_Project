from __future__ import annotations
from typing import Any
import numpy as np
from .base import ILLUMINATION_DIRECTION, PhysicsOperator, coordinate_grid


class SpecularReflectionOperator(PhysicsOperator):
    """Smooth specular highlight from a glossy medium.

    The highlight is an anisotropic Gaussian lobe placed inside the support and
    elongated along the capture illumination direction. A rough surface scatters
    the lobe (lower peak); a transparent medium passes light through instead of
    reflecting it, so both attenuate the effect. The blend is `screen`-like, so
    the result can approach but never exceed 1.
    """
    name = "specular_reflection"
    support_policy = "requested_region"
    peak = 0.85

    def _transform(self, image: np.ndarray, support: np.ndarray, strength: float, capture: dict[str, Any],
                   rng: np.random.Generator, parameters: dict[str, float]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        height, width = image.shape[1], image.shape[2]
        u, v = coordinate_grid(height, width)
        rows, columns = np.nonzero(support)
        if rows.size:
            centre_v = float(rows.mean() + 0.5) / float(height)
            centre_u = float(columns.mean() + 0.5) / float(width)
            span_v = max(float(rows.max() - rows.min() + 1) / float(height), 1e-3)
            span_u = max(float(columns.max() - columns.min() + 1) / float(width), 1e-3)
        else:
            centre_u, centre_v, span_u, span_v = 0.5, 0.5, 1.0, 1.0
        jitter_u, jitter_v = float(rng.uniform(-0.25, 0.25)), float(rng.uniform(-0.25, 0.25))
        centre_u = float(np.clip(centre_u + jitter_u * span_u, 0.0, 1.0))
        centre_v = float(np.clip(centre_v + jitter_v * span_v, 0.0, 1.0))
        sigma_ratio = float(np.clip(parameters.get("sigma_ratio", 0.18), 0.03, 0.6))
        direction = ILLUMINATION_DIRECTION.get(str(capture.get("illumination", "front")), (0.0, 0.0))
        elongation = 1.0 + 0.8 * float(np.hypot(*direction))
        sigma_u = max(sigma_ratio * span_u * elongation, 1e-3)
        sigma_v = max(sigma_ratio * span_v, 1e-3)
        centre_u = float(np.clip(centre_u + 0.15 * direction[0] * span_u, 0.0, 1.0))
        centre_v = float(np.clip(centre_v + 0.15 * direction[1] * span_v, 0.0, 1.0))
        du = (u - np.float32(centre_u)) / np.float32(sigma_u)
        dv = (v - np.float32(centre_v)) / np.float32(sigma_v)
        lobe = np.exp(np.float32(-0.5) * (du * du + dv * dv)).astype(np.float32)
        roughness = float(np.clip(parameters.get("medium_roughness", 0.3), 0.0, 1.0))
        transparency = float(np.clip(parameters.get("medium_transparency", 0.0), 0.0, 1.0))
        alpha = float(strength) * self.peak * (1.0 - 0.6 * roughness) * (1.0 - 0.3 * transparency)
        transformed = image + np.float32(alpha) * lobe[None, :, :] * (np.float32(1.0) - image)
        return transformed.astype(np.float32), support, {
            "centre": [round(centre_u, 6), round(centre_v, 6)], "sigma": [round(sigma_u, 6), round(sigma_v, 6)],
            "alpha": round(alpha, 6), "roughness": round(roughness, 6), "transparency": round(transparency, 6),
            "illumination": str(capture.get("illumination", "front"))}
