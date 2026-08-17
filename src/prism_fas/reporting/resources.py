"""Compute resources: what the hardware did, recorded as provenance only.

Three things get measured here — the device, the training run's memory and
timing, and a deterministic inference benchmark — and all three are operational
provenance. They are recorded on every artifact and excluded from every
scientific identity, because L.12 makes compute a non-treatment factor: a run is
the same experiment whether it took an hour on a 5090 or a week on a laptop.

The measurements that are easy to get wrong are done carefully:

* **Peak memory** is read after an explicit reset, so it describes the window
  being measured rather than everything since the process started.
* **Latency** is timed with CUDA synchronization around the timed region.
  Without it the timer measures how fast Python can enqueue kernels.
* **Model-only latency** is reported separately from end-to-end. Conflating them
  is the usual way an inference number ends up flattering.

On CPU the same fields are produced where they are meaningful and omitted where
they are not. A CPU number and a GPU number are never presented as a controlled
comparison — they are not one.
"""
from __future__ import annotations

import platform
import time
from typing import Any

SCHEMA_VERSION = "prism-compute-resources-v1"

#: Benchmark shape. Fixed so two runs of the benchmark are comparable to each
#: other; it is a diagnostic constant, not a scientific one.
DEFAULT_WARMUP = 5
DEFAULT_ITERATIONS = 20


def device_report() -> dict[str, Any]:
    """Everything about the machine that a later reader may need, and nothing else."""
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "cpu_count": None,
        "classification": "OPERATIONAL_PROVENANCE_ONLY",
        "enters_scientific_identity": False,
    }
    try:
        import os

        report["cpu_count"] = os.cpu_count()
    except Exception:                                        # noqa: BLE001
        pass
    try:
        import torch

        report["torch_version"] = torch.__version__
        report["torch_cuda_build"] = torch.version.cuda
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["thread_count"] = torch.get_num_threads()
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(index)
            report.update({
                "device": "cuda",
                "gpu_name": properties.name,
                "gpu_total_vram_mb": round(properties.total_memory / (1024 ** 2)),
                "compute_capability": f"{properties.major}.{properties.minor}",
                "multi_processor_count": properties.multi_processor_count,
                "cuda_runtime": torch.version.cuda,
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            })
        else:
            report["device"] = "cpu"
            report["precision"] = "fp32"
    except ImportError:
        report["torch_version"] = None
    try:
        import shutil

        report["disk_free_gb"] = round(shutil.disk_usage(".").free / (1024 ** 3), 2)
    except Exception:                                        # noqa: BLE001
        pass
    return report


def reset_peak_memory() -> None:
    """Start a fresh measurement window. Without this, peaks are cumulative."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    except Exception:                                        # noqa: BLE001
        pass


def peak_memory() -> dict[str, Any]:
    """Peak allocated and reserved bytes since the last reset."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"device": "cpu", "peak_allocated_mb": None, "peak_reserved_mb": None,
                    "note": "CUDA memory counters are unavailable on CPU"}
        torch.cuda.synchronize()
        return {
            "device": "cuda",
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024 ** 2), 2),
            "current_allocated_mb": round(torch.cuda.memory_allocated() / (1024 ** 2), 2),
            "semantics": "measured after reset_peak_memory_stats and a synchronize, so "
                         "the peak describes this window only",
        }
    except ImportError:
        return {"device": "unknown", "peak_allocated_mb": None, "peak_reserved_mb": None}


def _synchronize() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:                                        # noqa: BLE001
        pass


class TrainingTimer:
    """Wall-clock and throughput for a training window, with correct peaks."""

    def __init__(self, *, samples_per_step: int) -> None:
        self.samples_per_step = int(samples_per_step)
        self.steps = 0
        self._start = 0.0
        self._elapsed = 0.0

    def __enter__(self) -> "TrainingTimer":
        reset_peak_memory()
        _synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exception: Any) -> None:
        _synchronize()
        self._elapsed = time.perf_counter() - self._start

    def step(self, count: int = 1) -> None:
        self.steps += count

    def as_dict(self) -> dict[str, Any]:
        seconds = max(self._elapsed, 1e-9)
        return {
            "schema_version": SCHEMA_VERSION,
            "wall_clock_seconds": round(self._elapsed, 6),
            "steps": self.steps,
            "samples": self.steps * self.samples_per_step,
            "steps_per_second": round(self.steps / seconds, 6),
            "samples_per_second": round(self.steps * self.samples_per_step / seconds, 6),
            "seconds_per_step": round(seconds / max(self.steps, 1), 6),
            "memory": peak_memory(),
            "classification": "OPERATIONAL_PROVENANCE_ONLY",
        }


def benchmark_inference(model: Any, sample_input: Any, *, batch_size: int,
                        input_resolution: Any = None, warmup: int = DEFAULT_WARMUP,
                        iterations: int = DEFAULT_ITERATIONS,
                        forward: Any | None = None) -> dict[str, Any]:
    """Model-only inference latency, synchronized and reported with percentiles.

    Model-only on purpose: the timed region contains the forward pass and nothing
    else — no data loading, no post-processing. An end-to-end number is a
    different measurement and is labelled as such wherever it appears.
    """
    import torch

    model.eval()
    reset_peak_memory()

    def once() -> None:
        with torch.no_grad():
            if forward is not None:
                forward(model, sample_input)
            else:
                model(sample_input)

    error = ""
    latencies: list[float] = []
    try:
        for _ in range(max(0, warmup)):
            once()
        _synchronize()
        for _ in range(max(1, iterations)):
            start = time.perf_counter()
            once()
            _synchronize()
            latencies.append((time.perf_counter() - start) * 1000.0)
    except Exception as failure:                             # noqa: BLE001 - reported
        error = f"{type(failure).__name__}: {failure}"

    if not latencies:
        return {"schema_version": SCHEMA_VERSION, "status": "UNKNOWN", "error": error,
                "batch_size": batch_size, "classification": "OPERATIONAL_PROVENANCE_ONLY"}

    ordered = sorted(latencies)
    mean = sum(ordered) / len(ordered)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PARTIAL" if error else "COMPLETE",
        "error": error,
        "batch_size": batch_size,
        "input_resolution": list(input_resolution) if input_resolution else None,
        "warmup_iterations": warmup,
        "timed_iterations": len(ordered),
        "model_only_latency_ms_mean": round(mean, 4),
        "model_only_latency_ms_p50": round(ordered[len(ordered) // 2], 4),
        "model_only_latency_ms_p95": round(ordered[min(len(ordered) - 1,
                                                       int(len(ordered) * 0.95))], 4),
        "model_only_latency_ms_min": round(ordered[0], 4),
        "model_only_latency_ms_max": round(ordered[-1], 4),
        "throughput_samples_per_second": round(batch_size / (mean / 1000.0), 4),
        "fps": round(batch_size / (mean / 1000.0), 4),
        "fps_note": "frames per second at the stated batch size, from model-only "
                    "latency. Not an end-to-end pipeline rate",
        "latency_scope": "MODEL_ONLY",
        "end_to_end_measured": False,
        "synchronization": "CUDA synchronize around every timed region; on CPU the "
                           "call is a no-op and the timer is already exact",
        "memory": peak_memory(),
        "classification": "OPERATIONAL_PROVENANCE_ONLY",
        "enters_scientific_identity": False,
    }


def resource_record(*, microbatch_plan: Any = None, timer: Any = None,
                    inference: dict[str, Any] | None = None) -> dict[str, Any]:
    """One combined record for a run's compute provenance."""
    return {
        "schema_version": SCHEMA_VERSION,
        "device": device_report(),
        "microbatch_plan": (microbatch_plan.as_dict()
                            if hasattr(microbatch_plan, "as_dict") else microbatch_plan),
        "training": timer.as_dict() if hasattr(timer, "as_dict") else timer,
        "inference": inference,
        "classification": "OPERATIONAL_PROVENANCE_ONLY",
        "enters_scientific_identity": False,
        "cpu_gpu_comparison": "CPU and GPU runtimes are never presented as a controlled "
                              "treatment comparison; they are different machines, not "
                              "different conditions",
    }


__all__ = ["SCHEMA_VERSION", "DEFAULT_WARMUP", "DEFAULT_ITERATIONS", "device_report",
           "reset_peak_memory", "peak_memory", "TrainingTimer", "benchmark_inference",
           "resource_record"]
