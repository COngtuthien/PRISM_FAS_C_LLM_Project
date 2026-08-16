"""Which stages have adapters, and which do not.

The registry is deliberately sparse. C0-C3 have adapters; C4-C13 do not, and
asking for one returns nothing rather than a stub that reports success. That
absence is what keeps a smoke run over the implemented range from reading as a
smoke run over the whole pipeline.
"""
from __future__ import annotations

from typing import Any

from prism_fas.pipeline.adapters.c3 import C3Adapter
from prism_fas.pipeline.adapters.historical import build_adapters as _historical


def build_registry() -> dict[str, Any]:
    """Stage id -> adapter, for every stage that has one."""
    registry: dict[str, Any] = dict(_historical())
    registry["C3"] = C3Adapter()
    return registry


#: The stages an adapter exists for, in C-order. Everything else is
#: NOT_IMPLEMENTED and says so.
ADAPTED_STAGE_IDS: tuple[str, ...] = ("C0", "C1", "C2", "C3")

#: Every substage the registry can execute, in order. C2B and C2C are C2's.
ADAPTED_SUBSTAGE_IDS: tuple[str, ...] = ("C0", "C1", "C2", "C2B", "C2C", "C3")


def has_adapter(stage_id: str) -> bool:
    return stage_id in ADAPTED_STAGE_IDS


__all__ = ["build_registry", "ADAPTED_STAGE_IDS", "ADAPTED_SUBSTAGE_IDS", "has_adapter"]
