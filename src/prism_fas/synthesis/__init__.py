"""M7 CPU physics engine: deterministic semantic-region masks and operators.

M7 produces preview/audit artifacts only. The M8 quality gate, GPAT route and
versioned synthetic bank are not implemented here.
"""
from __future__ import annotations
from .contracts import (MaskBuildError, OperatorResult, PhysicsResult, RegionMaskResult, SynthesisError,
                        array_hash, mask_hash, validate_image, validate_mask)
from .masks import MASK_POLICY_VERSION, PARSING_LABELS, RegionMaskBuilder, builder_from_geometry
from .operators import OPERATOR_APPLICATION_ORDER, OPERATOR_CLASSES, OPERATOR_NAMES, build_operator
from .physics import PHYSICS_ENGINE_VERSION, PhysicsEngine

__all__ = ["MASK_POLICY_VERSION", "MaskBuildError", "OPERATOR_APPLICATION_ORDER", "OPERATOR_CLASSES",
           "OPERATOR_NAMES", "PARSING_LABELS", "PHYSICS_ENGINE_VERSION", "OperatorResult", "PhysicsEngine",
           "PhysicsResult", "RegionMaskBuilder", "RegionMaskResult", "SynthesisError", "array_hash",
           "build_operator", "builder_from_geometry", "mask_hash", "validate_image", "validate_mask"]
