from __future__ import annotations
from .base import PhysicsOperator
from .blur import BlurOperator
from .boundary import BoundaryInconsistencyOperator
from .color_shift import ColorShiftOperator
from .halftone import HalftoneOperator
from .moire import MoireOperator
from .pixel_grid import PixelGridOperator
from .reflection import SpecularReflectionOperator
from .smoothing import TextureSmoothingOperator

# Fixed, physically motivated application order: what happens at the medium
# surface first, then the optical interaction, then the capture chain. The
# compiler sorts a recipe's artifacts into this order and the order is part of
# the graph hash, so it can never drift silently.
OPERATOR_APPLICATION_ORDER = ("texture_smoothing", "halftone", "pixel_grid", "moire", "color_shift",
                              "specular_reflection", "boundary_inconsistency", "blur")
OPERATOR_CLASSES: dict[str, type[PhysicsOperator]] = {
    "halftone": HalftoneOperator, "pixel_grid": PixelGridOperator, "moire": MoireOperator,
    "specular_reflection": SpecularReflectionOperator, "texture_smoothing": TextureSmoothingOperator,
    "color_shift": ColorShiftOperator, "boundary_inconsistency": BoundaryInconsistencyOperator, "blur": BlurOperator}
OPERATOR_NAMES = tuple(sorted(OPERATOR_CLASSES))
OPERATOR_SUPPORT_POLICIES = {name: cls.support_policy for name, cls in sorted(OPERATOR_CLASSES.items())}
assert set(OPERATOR_APPLICATION_ORDER) == set(OPERATOR_NAMES), "operator order must cover exactly the implemented operators"


def build_operator(name: str) -> PhysicsOperator:
    try: return OPERATOR_CLASSES[name]()
    except KeyError: raise KeyError(f"no implemented physics operator named {name!r}; have {list(OPERATOR_NAMES)}") from None


__all__ = ["OPERATOR_APPLICATION_ORDER", "OPERATOR_CLASSES", "OPERATOR_NAMES", "OPERATOR_SUPPORT_POLICIES",
           "PhysicsOperator", "build_operator", "BlurOperator", "BoundaryInconsistencyOperator", "ColorShiftOperator",
           "HalftoneOperator", "MoireOperator", "PixelGridOperator", "SpecularReflectionOperator",
           "TextureSmoothingOperator"]
