from __future__ import annotations
from typing import Any
import numpy as np
from .base import PhysicsOperator, directional_blur, gaussian_blur


class BlurOperator(PhysicsOperator):
    """Capture-side blur.

    The variant is chosen deterministically from the compiled capture: a motion
    streak when `capture.motion` dominates, otherwise a Gaussian defocus. The
    kernel size follows both the capture parameter and the artifact strength.
    """
    name = "blur"
    support_policy = "requested_region"

    def _transform(self, image: np.ndarray, support: np.ndarray, strength: float, capture: dict[str, Any],
                   rng: np.random.Generator, parameters: dict[str, float]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        motion = float(np.clip(parameters.get("capture_motion", capture.get("motion", 0.0)), 0.0, 1.0))
        defocus = float(np.clip(parameters.get("capture_defocus", capture.get("defocus", 0.0)), 0.0, 1.0))
        sigma_base = float(np.clip(parameters.get("sigma_px", 1.2), 0.2, 5.0))
        blend = float(np.clip(float(strength), 0.0, 1.0))
        if motion > defocus:
            variant = "motion"
            length = 1.0 + 10.0 * motion * float(strength)
            yaw = float(parameters.get("capture_yaw", capture.get("yaw", 0.0)))
            angle = float(np.deg2rad(yaw)) * 0.5 + float(rng.uniform(-0.35, 0.35))
            filtered = directional_blur(image, length, angle)
            trace: dict[str, Any] = {"variant": variant, "length_px": round(length, 6), "angle_radians": round(angle, 6)}
        else:
            variant = "defocus"
            sigma = float(np.clip(sigma_base * (0.3 + 0.7 * float(strength)) * (0.5 + defocus) * float(rng.uniform(0.9, 1.1)), 0.1, 6.0))
            filtered = gaussian_blur(image, sigma)
            trace = {"variant": variant, "sigma_px": round(sigma, 6)}
        transformed = image * np.float32(1.0 - blend) + filtered * np.float32(blend)
        trace.update({"blend": round(blend, 6), "capture_motion": round(motion, 6), "capture_defocus": round(defocus, 6),
                      "sigma_base_px": round(sigma_base, 6)})
        return transformed.astype(np.float32), support, trace
