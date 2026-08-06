"""M6 Modal wrapper for the PRISM-FAS-B B00 baseline.

The wrapper only orchestrates: every function calls the same TrainerCore and
package loaders as the local M5 pipeline. `src/prism_fas/train/**` must never
import modal.
"""
from __future__ import annotations
import json
from pathlib import Path

import modal

APP_NAME = "prism-fas-b-m6"
DATA_VOLUME, MODELS_VOLUME, RUNS_VOLUME = "prism-fas-b-data", "prism-fas-b-models", "prism-fas-b-runs"
DATA_MOUNT, MODELS_MOUNT, RUNS_MOUNT = "/vol/data", "/vol/models", "/vol/runs"
REMOTE_PACKAGE = f"{DATA_MOUNT}/packages/prism_data_v1_m3b"
REMOTE_PARITY_INPUTS = f"{RUNS_MOUNT}/parity_inputs/b00_local_seed42"
REMOTE_RUNS_ROOT = f"{RUNS_MOUNT}/runs"
REMOTE_WEIGHT = f"{MODELS_MOUNT}/pretrained/b00/convnextv2_atto_fcmae_ft_in1k.safetensors"
EXPECTED_PACKAGE_IDENTITY = "b1cf29b69a165ed5d9e074fc8127c17fbf057723edf9e272048ec3a564eb9dc6"
EXPECTED_WEIGHT_SHA = "6389c2f5a427b01a922e66e6d352c707424cccb62390c6936bc612e3d10b7ebb"
EXPECTED_PARENT_IDENTITY = "a968caeb8e6e55a2afdba724923073161d2315e33c57733cf1be2b967b469769"
# Modal jobs stream the 9 tar shards instead of opening 13k small files.
DATA_BACKEND = "shard"
GPU_ALLOW_LIST = ("L4", "T4", "A10G")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1", "timm==1.0.28", "numpy==2.1.3", "pyarrow==18.1.0",
        "opencv-python-headless==4.10.0.84", "pydantic==2.10.3", "typer==0.15.1", "PyYAML==6.0.2",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .env({"HF_HUB_OFFLINE": "1", "PYTHONPATH": "/root/project/src"})
    # Source and config only: never .git, data/, runs/, reports/ or model_cache/.
    .add_local_dir("src", "/root/project/src")
    .add_local_dir("configs", "/root/project/configs")
)
app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME, create_if_missing=False)
models_volume = modal.Volume.from_name(MODELS_VOLUME, create_if_missing=False)
runs_volume = modal.Volume.from_name(RUNS_VOLUME, create_if_missing=False)
VOLUMES = {DATA_MOUNT: data_volume, MODELS_MOUNT: models_volume, RUNS_MOUNT: runs_volume}


def _project_paths() -> None:
    import sys
    if "/root/project/src" not in sys.path: sys.path.insert(0, "/root/project/src")


def _require_cuda(strict_fp32: bool = True) -> dict:
    """A GPU function must never silently fall back to CPU.

    strict_fp32 disables TF32: Ada/Ampere GPUs default to TF32 for conv and
    matmul, whose ~10-bit mantissa produces ~1e-2 logit drift versus CPU fp32.
    Parity is declared in fp32, so TF32 is switched off rather than the
    tolerance being widened.
    """
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the GPU function; refusing to run on CPU")
    if strict_fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "cudnn": str(torch.backends.cudnn.version()), "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)}


def _assert_gpu(gpu: str) -> str:
    if gpu not in GPU_ALLOW_LIST: raise ValueError(f"gpu {gpu!r} not in allow-list {GPU_ALLOW_LIST}")
    return gpu


def _safe_run_id(run_id: str) -> str:
    import re
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", run_id): raise ValueError(f"unsafe run id: {run_id!r}")
    return run_id


@app.function(image=image, volumes=VOLUMES, timeout=600)
def environment_probe() -> dict:
    """CPU probe: report the image contents and what the volumes actually hold."""
    _project_paths()
    import platform, sys, cv2, numpy, timm, torch
    package_lock = Path(REMOTE_PACKAGE) / "PACKAGE_LOCK.json"
    return {"python": platform.python_version(), "sys_version": sys.version.split()[0],
            "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda, "timm": timm.__version__, "numpy": numpy.__version__,
            "opencv": cv2.__version__,
            "package_present": package_lock.is_file(),
            "package_identity": json.loads(package_lock.read_text())["content_identity_sha256"] if package_lock.is_file() else None,
            "model_present": (Path(MODELS_MOUNT) / "pretrained/b00/convnextv2_atto_fcmae_ft_in1k.safetensors").is_file(),
            "parity_inputs_present": (Path(REMOTE_PARITY_INPUTS) / "checkpoints/best.pt").is_file()}


@app.function(image=image, volumes=VOLUMES, timeout=1800)
def verify_package() -> dict:
    """Shard-first remote verification (profile: remote_parity).

    Verifies the bytes the remote trainer actually reads. The exhaustive
    loose-file validator stays local; hashing 13k small files over a network
    volume previously exceeded the container lifetime and got preempted.
    """
    _project_paths()
    import hashlib, time
    from prism_fas.cloud.remote_verify import verify_remote_package
    from prism_fas.train.checkpoint import load_checkpoint
    started = time.time()
    report = verify_remote_package(Path(REMOTE_PACKAGE), expected_identity=EXPECTED_PACKAGE_IDENTITY,
                                   expected_parent=EXPECTED_PARENT_IDENTITY)
    weight = Path(MODELS_MOUNT) / "pretrained/b00/convnextv2_atto_fcmae_ft_in1k.safetensors"
    weight_sha = hashlib.sha256(weight.read_bytes()).hexdigest()
    checkpoint = Path(REMOTE_PARITY_INPUTS) / "checkpoints/best.pt"
    payload = load_checkpoint(checkpoint)
    calibration = json.loads((Path(REMOTE_PARITY_INPUTS) / "calibration/source_dev.json").read_text())
    extra = {"weight_sha256": weight_sha, "weight_sha_matches": weight_sha == EXPECTED_WEIGHT_SHA,
             "checkpoint_readable": True, "checkpoint_package_identity": payload.get("package_content_identity"),
             "checkpoint_model_name": payload.get("model_name"), "checkpoint_epoch": payload.get("epoch"),
             "checkpoint_global_step": payload.get("global_step"),
             "checkpoint_matches_package": payload.get("package_content_identity") == EXPECTED_PACKAGE_IDENTITY,
             "calibration_parsed": bool(calibration.get("calibration_hash")),
             "calibration_temperature": calibration.get("temperature"),
             "calibration_threshold": calibration.get("selected_threshold"),
             "raw_dataset_mounted": any(Path(m).exists() for m in ("/vol/data/Dataset", "/vol/data/raw")),
             "data_backend": DATA_BACKEND, "elapsed_seconds": round(time.time() - started, 1)}
    for key in ("weight_sha_matches", "checkpoint_matches_package", "calibration_parsed"):
        if not extra[key]: report["errors"].append(key); report["passed"] = False
    if extra["raw_dataset_mounted"]: report["errors"].append("raw_dataset_mounted"); report["passed"] = False
    return {**report, **extra}


@app.function(image=image, gpu="L4", volumes=VOLUMES, timeout=1200)
def forward_parity() -> dict:
    """fp32 eval-mode forward over the same fixed sample IDs as the local reference."""
    _project_paths()
    gpu_info = _require_cuda()
    from prism_fas.cloud.artifacts import run_parity_forward
    from prism_fas.train.config import load_b00_config
    config = load_b00_config(Path("/root/project/configs/train/b00_local.yaml"))
    calibration = json.loads((Path(REMOTE_PARITY_INPUTS) / "calibration/source_dev.json").read_text())
    result = run_parity_forward(Path(REMOTE_PACKAGE), Path(REMOTE_PARITY_INPUTS) / "checkpoints/best.pt",
                                calibration, config, device="cuda", source_count=32, target_count=16,
                                loader_config_path=Path("/root/project/configs/data/loader_m4.yaml"),
                                backend=DATA_BACKEND, weight_file=REMOTE_WEIGHT)
    return {**result, "gpu": gpu_info}


@app.function(image=image, gpu="L4", volumes=VOLUMES, timeout=1800)
def train_smoke(run_id: str = "b00_modal_smoke_seed42", steps: int = 5, resume_steps: int = 6,
                num_workers: int = 2) -> dict:
    """Real B00 GPU smoke through the shared TrainerCore, then a resume."""
    _project_paths()
    gpu_info = _require_cuda(strict_fp32=False)
    from prism_fas.train.config import load_b00_config
    from prism_fas.train.trainer import train_b00
    run_id = _safe_run_id(run_id)
    config = load_b00_config(Path("/root/project/configs/train/b00_local.yaml"))
    run_root = Path(REMOTE_RUNS_ROOT) / run_id
    loader_config = Path("/root/project/configs/data/loader_m4.yaml")
    first = train_b00(Path(REMOTE_PACKAGE), run_root, config, device="cuda", limit_steps=steps,
                      limit_dev_samples=128, workers=num_workers, loader_config_path=loader_config,
                      weight_file=REMOTE_WEIGHT)
    resumed = train_b00(Path(REMOTE_PACKAGE), run_root, config, device="cuda", resume=True,
                        limit_steps=resume_steps, limit_dev_samples=128, workers=num_workers,
                        loader_config_path=loader_config, weight_file=REMOTE_WEIGHT)
    metrics = [json.loads(line) for line in (run_root / "logs" / "metrics.jsonl").read_text().splitlines() if line.strip()]
    runs_volume.commit()
    return {"run_id": run_id, "gpu": gpu_info, "steps_first": first["global_step"],
            "steps_after_resume": resumed["global_step"],
            "resume_continued": resumed["global_step"] > first["global_step"],
            "batch_compositions": [row["batch_composition"] for row in metrics],
            "losses": [row["train_loss"] for row in metrics], "grad_norms": [row["grad_norm"] for row in metrics],
            "amp": first["amp"], "last_checkpoint_sha256": resumed["last_checkpoint_sha256"],
            "package_content_identity": first["package_content_identity"],
            "remote_run_path": str(run_root).replace(RUNS_MOUNT, "")}


@app.function(image=image, gpu="L4", volumes=VOLUMES, timeout=1200)
def inference_parity() -> dict:
    """Remote inference under the frozen local checkpoint and calibration."""
    _project_paths()
    gpu_info = _require_cuda()
    from prism_fas.cloud.artifacts import run_parity_forward
    from prism_fas.train.config import load_b00_config
    config = load_b00_config(Path("/root/project/configs/train/b00_local.yaml"))
    calibration = json.loads((Path(REMOTE_PARITY_INPUTS) / "calibration/source_dev.json").read_text())
    result = run_parity_forward(Path(REMOTE_PACKAGE), Path(REMOTE_PARITY_INPUTS) / "checkpoints/best.pt",
                                calibration, config, device="cuda", source_count=32, target_count=16,
                                loader_config_path=Path("/root/project/configs/data/loader_m4.yaml"),
                                backend=DATA_BACKEND, weight_file=REMOTE_WEIGHT)
    result.pop("features", None)
    return {**result, "gpu": gpu_info, "calibration_refitted": False, "threshold_changed": False}


@app.function(image=image, gpu="L4", volumes=VOLUMES, timeout=3600)
def train_entrypoint(config_name: str = "b00_local", run_id: str = "b00_modal", package_subpath: str = "packages/prism_data_v1_m3b",
                     model_subpath: str = "pretrained/b00", seed: int = 42, resume: bool = False,
                     limit_steps: int | None = None, gpu_type: str = "L4", run_subpath: str = "runs") -> dict:
    """Reusable training entrypoint for later milestones (not run fully in M6)."""
    _project_paths()
    _assert_gpu(gpu_type); run_id = _safe_run_id(run_id)
    for value in (package_subpath, model_subpath, run_subpath):
        if value.startswith("/") or ".." in value.split("/") or ":" in value:
            raise ValueError(f"unsafe subpath: {value!r}")
        for marker in ("Dataset", "data/work", "casia", "msu", "siw"):
            if marker.lower() in value.lower(): raise ValueError("raw dataset paths are not permitted")
    _require_cuda()
    from prism_fas.train.config import load_b00_config
    from prism_fas.train.trainer import train_b00
    config = load_b00_config(Path(f"/root/project/configs/train/{config_name}.yaml"))
    package_root = Path(DATA_MOUNT) / package_subpath
    lock = json.loads((package_root / "PACKAGE_LOCK.json").read_text())
    if lock["status"] != "validated": raise RuntimeError("package is not validated")
    if lock["content_identity_sha256"] != EXPECTED_PACKAGE_IDENTITY: raise RuntimeError("package identity mismatch")
    result = train_b00(package_root, Path(RUNS_MOUNT) / run_subpath / run_id, config, device="cuda",
                       resume=resume, limit_steps=limit_steps, workers=2,
                       loader_config_path=Path("/root/project/configs/data/loader_m4.yaml"),
                       weight_file=REMOTE_WEIGHT)
    runs_volume.commit()
    return {k: v for k, v in result.items() if k != "history"}


@app.local_entrypoint()
def main(stage: str = "probe"):
    """Run one M6 stage: probe | verify | forward | smoke | inference."""
    dispatch = {"probe": environment_probe, "verify": verify_package, "forward": forward_parity,
                "smoke": train_smoke, "inference": inference_parity}
    if stage not in dispatch: raise SystemExit(f"unknown stage {stage!r}; choose from {sorted(dispatch)}")
    result = dispatch[stage].remote()
    Path("reports/m6").mkdir(parents=True, exist_ok=True)
    Path(f"reports/m6/remote_{stage}.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    compact = {k: v for k, v in result.items() if k not in ("source", "target", "features")}
    print(json.dumps(compact, default=str)[:2000])
