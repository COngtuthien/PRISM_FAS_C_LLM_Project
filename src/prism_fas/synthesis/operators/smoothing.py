from __future__ import annotations
from typing import Any
import numpy as np
from .base import PhysicsOperator, gaussian_blur


class TextureSmoothingOperator(PhysicsOperator):
    """Loss of skin micro-texture on a re-presented surface.

    The smoothing is a bounded blend towards a Gaussian-blurred copy, confined
    to the support by the exact composite. The blend weight is capped well below
    1 so texture is attenuated, never globally erased.
    """
    name = "texture_smoothing"
    support_policy = "requested_region"
    max_blend = 0.85

    def _transform(self, image: np.ndarray, support: np.ndarray, strength: float, capture: dict[str, Any],
                   rng: np.random.Generator, parameters: dict[str, float]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        sigma_base = float(np.clip(parameters.get("sigma_px", 1.5), 0.3, 6.0))
        jitter = float(rng.uniform(0.85, 1.15))
        sigma = float(np.clip(sigma_base * (0.4 + 0.6 * float(strength)) * jitter, 0.2, 6.0))
        blend = float(np.clip(self.max_blend * float(strength), 0.0, 1.0))
        blurred = gaussian_blur(image, sigma)
        transformed = image * np.float32(1.0 - blend) + blurred * np.float32(blend)
        return transformed.astype(np.float32), support, {
            "sigma_px": round(sigma, 6), "sigma_base_px": round(sigma_base, 6),
            "jitter": round(jitter, 6), "blend": round(blend, 6), "max_blend": self.max_blend}
