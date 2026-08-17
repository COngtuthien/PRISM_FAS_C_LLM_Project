"""Which stages have adapters, and which do not.

Every C0-C13 stage now has one. That is an *engineering* statement and nothing
more: an adapter means the stage's control path can be executed and audited, not
that the stage has scientific evidence. C4-C13 have adapters and have never run
under the full profile, and the two facts sit side by side without contradiction
precisely because the dual-status model keeps them on different axes.

The registry stays a plain mapping with no fallback. Asking for a stage that has
no adapter returns nothing rather than a stub that reports success — that was
true when C4-C13 were missing and it remains the rule for anything added later.
"""
from __future__ import annotations

from typing import Any

from prism_fas.pipeline.adapters.c3 import C3Adapter
from prism_fas.pipeline.adapters.c4 import C4Adapter
from prism_fas.pipeline.adapters.c5 import C5Adapter
from prism_fas.pipeline.adapters.c6 import C6Adapter
from prism_fas.pipeline.adapters.c7 import C7Adapter
from prism_fas.pipeline.adapters.c8 import C8Adapter
from prism_fas.pipeline.adapters.c9 import C9Adapter
from prism_fas.pipeline.adapters.c10 import C10Adapter
from prism_fas.pipeline.adapters.c11 import C11Adapter
from prism_fas.pipeline.adapters.c12 import C12Adapter
from prism_fas.pipeline.adapters.c13 import C13Adapter
from prism_fas.pipeline.adapters.historical import build_adapters as _historical


def build_registry() -> dict[str, Any]:
    """Stage id -> adapter, for every stage that has one."""
    registry: dict[str, Any] = dict(_historical())
    registry["C3"] = C3Adapter()
    registry["C4"] = C4Adapter()
    registry["C5"] = C5Adapter()
    registry["C6"] = C6Adapter()
    registry["C7"] = C7Adapter()
    registry["C8"] = C8Adapter()
    registry["C9"] = C9Adapter()
    registry["C10"] = C10Adapter()
    registry["C11"] = C11Adapter()
    registry["C12"] = C12Adapter()
    registry["C13"] = C13Adapter()
    return registry


#: The stages an adapter exists for, in C-order.
ADAPTED_STAGE_IDS: tuple[str, ...] = ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7",
                                      "C8", "C9", "C10", "C11", "C12", "C13")

#: Every substage the registry can execute, in order. C2B and C2C are C2's.
ADAPTED_SUBSTAGE_IDS: tuple[str, ...] = ("C0", "C1", "C2", "C2B", "C2C", "C3", "C4",
                                         "C5", "C6", "C7", "C8", "C9", "C10", "C11",
                                         "C12", "C13")


def has_adapter(stage_id: str) -> bool:
    return stage_id in ADAPTED_STAGE_IDS


__all__ = ["build_registry", "ADAPTED_STAGE_IDS", "ADAPTED_SUBSTAGE_IDS", "has_adapter"]
