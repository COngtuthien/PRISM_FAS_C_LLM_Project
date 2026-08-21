"""The GPU proof-of-work that runs before scientific training, inside `train.py`.

Selecting a dependency profile from `nvidia-smi` output proves that a driver is
installed. It does not prove that this PyTorch build can launch a kernel on this
card, that a checkpoint round-trips, or that the memory counters report anything.
Those fail at the first real step — after the data has loaded, an hour into a run
the operator believed had started correctly.

So this module does the small version of the real thing first: allocate, matmul,
build the actual detector, forward, backward, save a checkpoint, read it back,
synchronize, and read the memory counters. Every operation here is engineering.
It touches no dataset, no target, no lock and no scientific artifact, and it runs
for a second or two.

It is deliberately called from the zero-argument path rather than exposed as a
separate command. The user asked for one command; a preflight they have to
remember to run is a preflight that does not run.

A failure stops the pipeline BEFORE C4 with a specific reason code, because
"CUDA out of memory at step 3000" and "this build cannot launch sm_120 kernels"
need different responses from the operator.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "prism-gpu-preflight-v1"

#: Reason codes. Each is a real runtime condition, never "not implemented".
CUDA_UNAVAILABLE = "CUDA_UNAVAILABLE"
KERNEL_LAUNCH_FAILED = "KERNEL_LAUNCH_FAILED"
CAPABILITY_UNSUPPORTED_BY_BUILD = "CAPABILITY_UNSUPPORTED_BY_BUILD"
INSUFFICIENT_GPU_MEMORY = "INSUFFICIENT_GPU_MEMORY"
CHECKPOINT_ROUNDTRIP_FAILED = "CHECKPOINT_ROUNDTRIP_FAILED"
AUTOGRAD_FAILED = "AUTOGRAD_FAILED"

#: Below this, no permitted microbatch/accumulation combination fits the smallest
#: declared scientific composition. Chosen from the C7/C8 detector footprint, not
#: from a vendor spec sheet.
MINIMUM_USABLE_VRAM_MB = 8000

#: The stage whose batch contract and loss graph the autograd probe exercises.
#: Named rather than inlined so a test can prove the probe and the trainer speak
#: about the same stage.
PREFLIGHT_STAGE = "G5"


class GPUPreflightError(RuntimeError):
    def __init__(self, reason: str, message: str, detail: dict[str, Any] | None = None):
        super().__init__(message)
        self.reason = reason
        self.detail = detail or {}


def _probe(name: str, ok: bool, summary: str, **detail: Any) -> dict[str, Any]:
    return {"probe": name, "ok": bool(ok), "summary": summary, "detail": detail}


def run_preflight(repo: Path, *, required_vram_mb: int = MINIMUM_USABLE_VRAM_MB,
                  strict: bool = True) -> dict[str, Any]:
    """Prove this host can actually train, then report what it measured.

    `strict=False` lets a CPU host produce the same report shape with every GPU
    probe marked not-applicable, so the rehearsal path exercises this code too
    rather than skipping it entirely.
    """
    started = time.time()
    probes: list[dict[str, Any]] = []
    device_info: dict[str, Any] = {}

    try:
        import torch
    except ImportError as error:
        raise GPUPreflightError(
            CUDA_UNAVAILABLE, f"torch does not import: {error}") from error

    available = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    probes.append(_probe("cuda_is_available", available or not strict,
                         "torch.cuda.is_available()" if available else
                         "no CUDA device is visible to torch",
                         torch_version=torch.__version__,
                         torch_cuda_version=getattr(torch.version, "cuda", None),
                         available=available))

    if not available:
        if strict:
            raise GPUPreflightError(
                CUDA_UNAVAILABLE,
                "torch reports no CUDA device. The driver may be present while this "
                "torch build is the CPU wheel; check state/ENVIRONMENT_MANIFEST.json "
                "for the profile that was installed.",
                {"torch_version": torch.__version__,
                 "torch_cuda_version": getattr(torch.version, "cuda", None)})
        return _report(started, probes, device_info, applicable=False)

    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    capability = f"{properties.major}.{properties.minor}"
    total_mb = int(properties.total_memory / (1024 ** 2))
    device_info = {
        "device_count": torch.cuda.device_count(),
        "device_index": index,
        # Operational provenance. Recorded, never used to decide anything.
        "gpu_name": properties.name,
        "compute_capability": capability,
        "total_memory_mb": total_mb,
        "multi_processor_count": properties.multi_processor_count,
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
    }

    probes.append(_probe(
        "device_enumeration", torch.cuda.device_count() >= 1,
        f"{torch.cuda.device_count()} CUDA device(s); using index {index}",
        **device_info))

    # The build must actually carry kernels for this architecture. An arch list
    # missing this capability is the failure that otherwise appears as a cryptic
    # "no kernel image is available for execution on the device".
    try:
        arch_list = torch.cuda.get_arch_list()
    except Exception:                                        # noqa: BLE001
        arch_list = []
    arch_ok = (not arch_list) or any(
        item.replace("sm_", "") == capability.replace(".", "") for item in arch_list)
    probes.append(_probe(
        "build_supports_capability", arch_ok,
        f"this torch build carries kernels for sm_{capability.replace('.', '')}"
        if arch_ok else
        f"this torch build has no sm_{capability.replace('.', '')} kernels",
        arch_list=arch_list, capability=capability))
    if not arch_ok and strict:
        raise GPUPreflightError(
            CAPABILITY_UNSUPPORTED_BY_BUILD,
            f"the installed torch was not built for compute capability {capability}. "
            f"It carries {arch_list}. Reinstall with the CUDA profile whose "
            "compute_capabilities include this device, then rerun `python train.py`.",
            {"capability": capability, "arch_list": arch_list})

    if total_mb < required_vram_mb:
        raise GPUPreflightError(
            INSUFFICIENT_GPU_MEMORY,
            f"the device reports {total_mb} MB of VRAM; no permitted "
            f"microbatch/accumulation combination fits the declared scientific "
            f"composition below {required_vram_mb} MB. The run stops rather than "
            "shrinking the model, the loss, the resolution or the batch composition.",
            {"total_memory_mb": total_mb, "required_mb": required_vram_mb})

    torch.cuda.reset_peak_memory_stats(index)
    torch.cuda.synchronize(index)

    # 1. a tiny tensor op
    try:
        left = torch.randn(256, 256, device="cuda")
        product = (left @ left.T).sum().item()
        torch.cuda.synchronize(index)
        probes.append(_probe("tensor_kernel_launch", True,
                             "a 256x256 matmul launched and synchronized",
                             checksum_is_finite=bool(product == product)))
    except Exception as error:                               # noqa: BLE001
        raise GPUPreflightError(
            KERNEL_LAUNCH_FAILED,
            f"a minimal CUDA matmul failed: {type(error).__name__}: {error}",
            {"capability": capability}) from error

    # 2-4. a representative model, forward, backward
    try:
        forward_detail = _model_roundtrip(torch, torch.device("cuda", index))
        probes.append(_probe("model_forward_backward", True,
                             "the real detector instantiated, forwarded through the "
                             "real batch contract and backpropagated on the device",
                             **forward_detail))
    except GPUPreflightError:
        raise
    except Exception as error:                               # noqa: BLE001
        raise GPUPreflightError(
            AUTOGRAD_FAILED,
            f"the representative model could not complete a forward/backward step: "
            f"{type(error).__name__}: {error}") from error

    # 5. checkpoint write and read
    try:
        checkpoint_detail = _checkpoint_roundtrip(repo, torch)
        probes.append(_probe("checkpoint_roundtrip", True,
                             "a checkpoint was written, reloaded and compared equal",
                             **checkpoint_detail))
    except Exception as error:                               # noqa: BLE001
        raise GPUPreflightError(
            CHECKPOINT_ROUNDTRIP_FAILED,
            f"a checkpoint could not be written and read back: "
            f"{type(error).__name__}: {error}") from error

    # 6. synchronized timing and the memory counters
    torch.cuda.synchronize(index)
    peak_allocated = int(torch.cuda.max_memory_allocated(index) / (1024 ** 2))
    peak_reserved = int(torch.cuda.max_memory_reserved(index) / (1024 ** 2))
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    probes.append(_probe(
        "memory_counters", peak_allocated > 0,
        "the CUDA memory counters reset and read back non-zero after real work",
        peak_allocated_mb=peak_allocated, peak_reserved_mb=peak_reserved,
        free_mb=int(free_bytes / (1024 ** 2)),
        total_mb=int(total_bytes / (1024 ** 2))))

    device_info.update({"peak_allocated_mb": peak_allocated,
                        "peak_reserved_mb": peak_reserved,
                        "free_memory_mb": int(free_bytes / (1024 ** 2))})
    return _report(started, probes, device_info, applicable=True)


def _same_device(actual: Any, expected: Any) -> bool:
    """Device equality that tolerates an unindexed request for the current device."""
    if actual.type != expected.type:
        return False
    if expected.index is None or actual.index is None:
        return True
    return int(actual.index) == int(expected.index)


def _require_device(device: Any, named: dict[str, Any]) -> dict[str, str]:
    """Every named tensor must sit on the device the run selected.

    This is the probe that catches a silent CPU fallback. A stub, a buffer or a
    collate step that quietly stayed on the host either fails with a device
    mismatch far from its cause, or — when the whole subgraph drifts together —
    succeeds on the CPU while the operator believes the GPU is training.
    """
    wrong = {name: str(tensor.device) for name, tensor in named.items()
             if tensor is not None and not _same_device(tensor.device, device)}
    if wrong:
        raise GPUPreflightError(
            AUTOGRAD_FAILED,
            f"the run selected {device} but {len(wrong)} tensor(s) executed "
            f"elsewhere: {wrong}. That is a silent CPU fallback; the run stops "
            "rather than spending the GPU budget on the host.",
            {"selected_device": str(device), "misplaced": wrong})
    return {name: str(tensor.device) for name, tensor in named.items()
            if tensor is not None}


def _audit_gradients(torch: Any, trainable: list[tuple[str, Any]]) -> dict[str, Any]:
    """Every parameter the run declares trainable must carry a finite gradient.

    "Some parameter somewhere got a gradient" is the weak version of this check
    and it passes on a graph that has silently detached a whole branch. A NaN at
    step zero is an arithmetic fault on this device, not a training problem to be
    tuned away, so it stops the run here rather than at hour three.
    """
    if not trainable:
        raise GPUPreflightError(
            AUTOGRAD_FAILED, "the representative variant has no trainable parameters")
    missing = [name for name, parameter in trainable if parameter.grad is None]
    if missing:
        raise GPUPreflightError(
            AUTOGRAD_FAILED,
            f"the backward pass left {len(missing)} of {len(trainable)} trainable "
            f"parameters without a gradient, starting at {missing[:5]}",
            {"missing_gradients": len(missing), "trainable": len(trainable),
             "first_missing": missing[:5]})
    nonfinite = [name for name, parameter in trainable
                 if not bool(torch.isfinite(parameter.grad).all())]
    if nonfinite:
        raise GPUPreflightError(
            AUTOGRAD_FAILED,
            f"{len(nonfinite)} parameter gradient(s) are not finite, starting at "
            f"{nonfinite[:5]}",
            {"nonfinite_gradients": len(nonfinite), "trainable": len(trainable),
             "first_nonfinite": nonfinite[:5]})
    norm = float(torch.sqrt(sum((parameter.grad.detach().float() ** 2).sum()
                                for _, parameter in trainable)))
    return {"trainable_parameters": len(trainable),
            "parameters_with_gradients": len(trainable),
            "gradients_are_finite": True,
            "gradient_global_norm": round(norm, 6)}


def _representative_batch(variant: Any, device: Any) -> tuple[Any, Any]:
    """One batch built through the REAL `DetectorBatch` contract, on `device`.

    The detector's `forward` takes a `DetectorBatch` — the image, its nine region
    priors, visibility, labels and the synthetic provenance the declared losses
    read — never a bare image tensor. Handing it `torch.randn(2, 3, 224, 224)` is
    exactly how this probe failed on the RTX 5090 with `'Tensor' object has no
    attribute 'image'`: the probe had invented an input contract that the trainer
    does not use, so it could only ever have passed by accident.

    The composition comes from `batch_contract_for`, so the probe carries the
    declared real-live / real-spoof / synthetic mix rather than a shape the
    trainer would never produce. No dataset is opened and no target is resolved:
    `audit_batch` is the same seeded fixture the CPU variant audit already uses.
    """
    from prism_fas.detector.trainer import M9TrainingConfig, batch_contract_for
    from prism_fas.evaluation.variant_audit import audit_batch

    config = M9TrainingConfig(run_id="gpu_preflight", variant=variant, steps_per_epoch=1)
    contract = batch_contract_for(PREFLIGHT_STAGE, config)
    return audit_batch(variant, contract).to(device), contract


def _model_roundtrip(torch: Any, device: Any) -> dict[str, Any]:
    """Instantiate the real detector and push one real batch through it.

    The detector rather than a toy `nn.Linear`: the operators that fail on a new
    architecture are the ones the real topology uses, and a probe that passes
    while the real model would fail is worse than no probe. The input is the real
    batch contract and the loss is the real loss graph for the same reason — a
    probe that invents either proves nothing about the run that follows it.

    Nothing here is scientific: the detector is at audit fixture scale, the batch
    is seeded noise, no optimizer steps, and no artifact is written.
    """
    from prism_fas.detector.losses import compute_losses
    from prism_fas.detector.trainer import enabled_terms
    from prism_fas.detector.variant import ResolvedExperimentVariant
    from prism_fas.evaluation.variant_audit import build_audit_detector
    from prism_fas.pipeline.adapters.c7 import TRACK_R_FLAGS

    # Track R is the heavier of the two decision heads, so it exercises the
    # widest set of operators. Passing here implies Track G passes.
    variant = ResolvedExperimentVariant.resolve(TRACK_R_FLAGS)
    model = build_audit_detector(variant).to(device)
    model.train()

    batch, contract = _representative_batch(variant, device)
    groups = model.parameter_groups(backbone_lr=1e-5, head_lr=1e-4, weight_decay=0.05)

    started = time.time()
    output = model(batch)
    result = compute_losses(output, batch, model.manifold,
                            text_embeddings=model.text_matrix(),
                            enabled=enabled_terms(PREFLIGHT_STAGE, variant),
                            variant=variant)
    loss = result.total
    if loss.dim() != 0:
        raise GPUPreflightError(
            AUTOGRAD_FAILED,
            f"the loss graph produced a tensor of shape {tuple(loss.shape)}; "
            "backward needs a scalar")
    if not bool(torch.isfinite(loss)):
        raise GPUPreflightError(
            AUTOGRAD_FAILED,
            f"the forward pass produced a non-finite loss ({float(loss.detach())})")

    model.zero_grad(set_to_none=True)
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.time() - started

    trainable = [(name, parameter) for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    gradient_detail = _audit_gradients(torch, trainable)

    # None of the above proves the work happened where the operator was told it did.
    devices = _require_device(device, {
        "batch.image": batch.image, "batch.region_priors": batch.region_priors,
        "output.s_final": output.s_final, "loss": loss,
        "first_parameter": trainable[0][1], "first_gradient": trainable[0][1].grad})

    return {"variant": variant.identity(),
            "stage": PREFLIGHT_STAGE,
            "batch_contract": contract.payload(),
            "batch_size": batch.batch_size,
            "parameters": sum(p.numel() for p in model.parameters()),
            "optimizer_groups": [group["name"] for group in groups],
            "active_loss_terms": dict(result.active),
            "forward_backward_seconds": round(elapsed, 4),
            "loss": float(loss.detach()),
            "loss_is_finite": True,
            "executed_on": devices,
            **gradient_detail}


def _checkpoint_roundtrip(repo: Path, torch: Any) -> dict[str, Any]:
    """Write a checkpoint to the real output root and read it back."""
    import tempfile

    scratch = repo / "state" / "preflight"
    scratch.mkdir(parents=True, exist_ok=True)
    payload = {"probe": torch.randn(64, 64), "step": 1}
    handle = tempfile.NamedTemporaryFile(dir=scratch, suffix=".pt", delete=False)
    handle.close()
    path = Path(handle.name)
    try:
        torch.save(payload, path)
        reloaded = torch.load(path, map_location="cpu", weights_only=False)
        identical = bool(torch.equal(payload["probe"], reloaded["probe"]))
        if not identical:
            raise GPUPreflightError(
                CHECKPOINT_ROUNDTRIP_FAILED,
                "a checkpoint reloaded with different contents than were written")
        return {"bytes": path.stat().st_size, "round_trip_identical": identical,
                "written_under": scratch.relative_to(repo).as_posix()}
    finally:
        path.unlink(missing_ok=True)


def _report(started: float, probes: list[dict[str, Any]], device: dict[str, Any],
            *, applicable: bool) -> dict[str, Any]:
    failed = [item for item in probes if not item["ok"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "applicable": applicable,
        "outcome": "PASS" if not failed else "FAIL",
        "probes_run": len(probes),
        "probes_failed": len(failed),
        "probes": probes,
        "device": device,
        "elapsed_seconds": round(time.time() - started, 3),
        "scientific_eligible": False,
        "meaning": "an engineering proof that this host can execute the pipeline. "
                   "It opens no dataset, resolves no target and produces no "
                   "scientific evidence.",
    }


def write_report(repo: Path, report: dict[str, Any]) -> Path:
    from prism_fas.pipeline.state import atomic_write_json

    path = repo / "reports" / "preflight" / "GPU_PREFLIGHT.json"
    atomic_write_json(path, report)
    return path


__all__ = ["run_preflight", "write_report", "GPUPreflightError", "SCHEMA_VERSION",
           "MINIMUM_USABLE_VRAM_MB", "PREFLIGHT_STAGE", "CUDA_UNAVAILABLE",
           "KERNEL_LAUNCH_FAILED",
           "CAPABILITY_UNSUPPORTED_BY_BUILD", "INSUFFICIENT_GPU_MEMORY",
           "CHECKPOINT_ROUNDTRIP_FAILED", "AUTOGRAD_FAILED"]
