"""Model complexity: parameters, MACs and FLOPs, counted or honestly unknown.

Two rules govern this module, and the second is the one that matters.

**Never fabricate a FLOP count.** MACs are accumulated by forward hooks over the
layer types this project actually uses. Any module that reaches a hook and is not
in the supported set is recorded by name in `unsupported_operations`, and the
result is downgraded to PARTIAL. A number that silently skipped half the network
would be worse than no number, because it would be quoted.

**Compute diagnostics may never select a scientific winner.** Every payload
carries `selection_input: false`. The §15.2.3 and §15.4 selection tuples contain
no complexity term, and nothing here is allowed to become one — a model chosen
for being cheap is a different experiment from the one that was preregistered.

Parameter counts are exact and always available; MACs are best-effort. The two
are reported separately so a reader can tell which is which.
"""
from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "prism-model-complexity-v1"

#: Counting convention, stated because the field is ambiguous in the literature.
#: One MAC is one multiply-accumulate. FLOPs are reported as 2 x MACs, which is
#: the convention that counts the multiply and the add separately.
FLOPS_PER_MAC = 2

COMPLETE = "COMPLETE"
PARTIAL = "PARTIAL"
UNKNOWN = "UNKNOWN"

#: Modules that carry no arithmetic worth counting. Reaching one of these is not
#: an unsupported-operation finding.
_TRANSPARENT = ("Sequential", "ModuleList", "ModuleDict", "Identity", "Dropout",
                "Dropout2d", "Flatten", "Sigmoid", "Tanh", "SiLU", "GELU", "ReLU",
                "Softmax", "Upsample", "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d")


def parameter_counts(model: Any) -> dict[str, Any]:
    """Exact parameter accounting. Always available, never estimated."""
    trainable = sum(item.numel() for item in model.parameters() if item.requires_grad)
    frozen = sum(item.numel() for item in model.parameters() if not item.requires_grad)
    bytes_total = sum(item.numel() * item.element_size() for item in model.parameters())
    buffers = sum(item.numel() for item in model.buffers())
    return {
        "total_parameters": trainable + frozen,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "buffer_elements": buffers,
        "parameter_bytes": bytes_total,
        "parameter_megabytes": round(bytes_total / (1024 ** 2), 4),
        "dtypes": sorted({str(item.dtype) for item in model.parameters()}),
    }


def _hook_macs(module: Any, inputs: Any, output: Any, sink: dict[str, Any]) -> None:
    import torch
    from torch import nn

    name = type(module).__name__
    if isinstance(module, nn.Conv2d):
        out = output.shape
        kernel = module.kernel_size[0] * module.kernel_size[1]
        per_position = kernel * (module.in_channels // module.groups)
        sink["macs"] += int(per_position * module.out_channels
                            * out[-1] * out[-2] * out[0])
    elif isinstance(module, nn.Linear):
        elements = 1
        for dimension in output.shape[:-1]:
            elements *= int(dimension)
        sink["macs"] += int(elements * module.in_features * module.out_features)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        sink["macs"] += int(output.numel())
    elif isinstance(module, nn.MultiheadAttention):
        # Reported rather than counted: the shape depends on the call site, and a
        # guess here would be indistinguishable from a measurement.
        sink["unsupported"].add(name)
    elif name not in _TRANSPARENT and not list(module.children()):
        sink["unsupported"].add(name)


def count_macs(model: Any, sample_input: Any, *,
               forward: Any | None = None) -> dict[str, Any]:
    """Accumulate MACs over one forward pass. Reports what it could not count.

    `forward` lets a caller drive a model whose `forward` takes a structured
    batch rather than a tensor — the detector and the GPAT generator both do.
    """
    import torch

    sink: dict[str, Any] = {"macs": 0, "unsupported": set()}
    handles = []
    for module in model.modules():
        handles.append(module.register_forward_hook(
            lambda m, i, o, sink=sink: _hook_macs(m, i, o, sink)))
    error = ""
    try:
        with torch.no_grad():
            if forward is not None:
                forward(model, sample_input)
            else:
                model(sample_input)
    except Exception as failure:                             # noqa: BLE001 - reported
        error = f"{type(failure).__name__}: {failure}"
    finally:
        for handle in handles:
            handle.remove()

    unsupported = sorted(sink["unsupported"])
    if error:
        status = UNKNOWN
    elif unsupported:
        status = PARTIAL
    else:
        status = COMPLETE
    return {
        "status": status,
        "macs": int(sink["macs"]) if status != UNKNOWN else None,
        "flops": int(sink["macs"]) * FLOPS_PER_MAC if status != UNKNOWN else None,
        "flops_per_mac": FLOPS_PER_MAC,
        "counting_convention": "one MAC = one multiply-accumulate; FLOPs = 2 x MACs. "
                               "Counted by forward hooks over Conv2d, Linear and "
                               "normalization layers",
        "unsupported_operations": unsupported,
        "warnings": ([f"{len(unsupported)} module type(s) were not counted; the MAC "
                      "total is a LOWER BOUND"] if unsupported else [])
                    + ([f"forward pass failed: {error}"] if error else []),
        "tool": "prism_fas.reporting.complexity (in-tree forward hooks)",
        "tool_version": SCHEMA_VERSION,
    }


def profile_model(model: Any, sample_input: Any, *, name: str,
                  input_shape: Any = None, forward: Any | None = None) -> dict[str, Any]:
    """One model's complete complexity record."""
    import torch

    shape = list(input_shape) if input_shape is not None else (
        list(sample_input.shape) if hasattr(sample_input, "shape") else None)
    counts = parameter_counts(model)
    macs = count_macs(model, sample_input, forward=forward)
    return {
        "schema_version": SCHEMA_VERSION,
        "model": name,
        "canonical_input_shape": shape,
        **counts,
        "complexity": macs,
        "torch_version": torch.__version__,
        "selection_input": False,
        "selection_note": "compute diagnostics never enter scientific winner selection. "
                          "The §15.2.3 and §15.4 selection tuples contain no complexity "
                          "term",
    }


__all__ = ["SCHEMA_VERSION", "FLOPS_PER_MAC", "COMPLETE", "PARTIAL", "UNKNOWN",
           "parameter_counts", "count_macs", "profile_model"]
