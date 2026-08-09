"""M5 B00 local baseline: model, loss, trainer, calibration, inference, report.

The re-exports are LAZY (PEP 562). Importing them eagerly would mean that
`from prism_fas.train.metrics import roc_auc` — the numpy-only primitives M10's G8
binds so the two milestones cannot drift — pulls `.models`, and with it torch, into
the scoring process. G8's whole isolation claim is that it holds no training
runtime, so the package that owns the trainer must not force one on a module that
only wants a metric. The public names are unchanged.
"""
from __future__ import annotations
from typing import Any

_LAZY = {"B00Config": "config", "TRAIN_SCHEMA_VERSION": "config", "load_b00_config": "config",
         "b00_binary_cross_entropy": "losses", "B00ConvNeXtBinaryClassifier": "models",
         "B00Output": "models", "build_b00_model": "models"}
__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    if name not in _LAZY: raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)


def __dir__() -> list[str]: return sorted(set(globals()) | set(_LAZY))
