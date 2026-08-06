"""M8 Modal wrapper: GPAT training, quality calibration and synthetic bank.

The wrapper only orchestrates. Every function calls the same core modules the
local CPU smoke uses; `src/prism_fas/synthesis/**` must never import modal.

Source-only: these functions open `source_train` payloads, the frozen M7 recipe
bank and the pinned quality weights. `source_dev` and `target_test` are never
opened.
"""
from __future__ import annotations
import json
from pathlib import Path

import modal

APP_NAME = "prism-fas-b-m8"
DATA_VOLUME, MODELS_VOLUME, RUNS_VOLUME = "prism-fas-b-data", "prism-fas-b-models", "prism-fas-b-runs"
DATA_MOUNT, MODELS_MOUNT, RUNS_MOUNT = "/vol/data", "/vol/models", "/vol/runs"
REMOTE_PACKAGE = f"{DATA_MOUNT}/packages/prism_data_v1_m3b"
REMOTE_WEIGHT_ROOT = f"{MODELS_MOUNT}/pretrained/m8"
REMOTE_RUNS_ROOT = f"{RUNS_MOUNT}/runs"
REMOTE_SYNTHETIC_WORK = f"{RUNS_MOUNT}/synthetic_banks/m8_work"
REMOTE_EXPORTS = f"{RUNS_MOUNT}/exports"
EXPECTED_PACKAGE_IDENTITY = "b1cf29b69a165ed5d9e074fc8127c17fbf057723edf9e272048ec3a564eb9dc6"
EXPECTED_BANK_IDENTITY = "fa989938cafdc4887518cc45c35d559d00278358439dc68c2486da10309210cb"
GPU_ALLOW_LIST = ("L4", "L40S")
DEFAULT_GPU = "L4"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1", "numpy==2.1.3", "pyarrow==18.1.0",
        "opencv-python-headless==4.10.0.84", "onnxruntime==1.20.1", "pydantic==2.10.3",
        "typer==0.15.1", "PyYAML==6.0.2",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .env({"HF_HUB_OFFLINE": "1", "PYTHONPATH": "/root/project/src", "PYTHONIOENCODING": "utf-8"})
    # Source, config and the small committed recipe bank only: never .git,
    # data/, runs/, reports/ or model_cache/.
    .add_local_dir("src", "/root/project/src")
    .add_local_dir("configs", "/root/project/configs")
    .add_local_dir("assets", "/root/project/assets")
)
app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME, create_if_missing=False)
models_volume = modal.Volume.from_name(MODELS_VOLUME, create_if_missing=False)
runs_volume = modal.Volume.from_name(RUNS_VOLUME, create_if_missing=False)
VOLUMES = {DATA_MOUNT: data_volume, MODELS_MOUNT: models_volume, RUNS_MOUNT: runs_volume}
PROJECT = Path("/root/project")


def _paths() -> None:
    import sys
    if "/root/project/src" not in sys.path: sys.path.insert(0, "/root/project/src")


def _require_cuda() -> dict:
    """A GPU function must never silently fall back to CPU."""
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is not available inside a GPU function")
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {"device": "cuda", "gpu_name": properties.name,
            "gpu_memory_gb": round(properties.total_memory / 1024 ** 3, 2),
            "cuda_runtime": torch.version.cuda, "cudnn": str(torch.backends.cudnn.version()),
            "torch": torch.__version__}


def _verify_inputs() -> dict:
    """Fail before allocating work if the package or recipe bank is not the
    exact frozen artifact this milestone was declared against."""
    _paths()
    lock = json.loads((Path(REMOTE_PACKAGE) / "PACKAGE_LOCK.json").read_text(encoding="utf-8"))
    identity = str(lock["content_identity_sha256"])
    if identity != EXPECTED_PACKAGE_IDENTITY:
        raise RuntimeError(f"remote package identity {identity} != {EXPECTED_PACKAGE_IDENTITY}")
    bank_lock = json.loads((PROJECT / "assets" / "recipe_banks" / "prism_recipe_bank_m7_v1" / "BANK_LOCK.json").read_text(encoding="utf-8"))
    bank_identity = str(bank_lock["bank_content_identity_sha256"])
    if bank_identity != EXPECTED_BANK_IDENTITY:
        raise RuntimeError(f"recipe bank identity {bank_identity} != {EXPECTED_BANK_IDENTITY}")
    from prism_fas.synthesis.quality_models import QualityModelRegistry
    registry = QualityModelRegistry.resolve(Path(REMOTE_WEIGHT_ROOT), roles=("identity",))
    return {"package_identity": identity, "recipe_bank_identity": bank_identity,
            "package_status": lock["status"], "package_split_counts": lock["per_split_counts"],
            "quality_models": registry.manifest()}


def _pairs_root(run_root: Path) -> Path:
    """Materialize the deterministic pair plan inside the run root.

    The plan is a pure function of the package, the frozen bank and the seed, so
    rebuilding it remotely reproduces the local manifests byte for byte.
    """
    _paths()
    from prism_fas.synthesis.pair_plan import write_pair_plan
    pairs = run_root / "pairs"
    # Always rewrite: the plan is deterministic and cheap, so re-deriving it is
    # self-healing if a run root still holds a plan from an earlier code version.
    write_pair_plan(Path(REMOTE_PACKAGE), PROJECT / "assets" / "recipe_banks" / "prism_recipe_bank_m7_v1",
                    pairs, config_hash="modal_m8")
    return pairs


def _config() -> dict:
    _paths()
    from prism_fas.synthesis.m8_pipeline import load_gpat_config
    return load_gpat_config(PROJECT / "configs" / "synthesis" / "gpat_m8.yaml")


@app.function(image=image, volumes=VOLUMES, gpu=DEFAULT_GPU, timeout=600)
def m8_environment_probe() -> dict:
    _paths()
    gpu = _require_cuda()
    verified = _verify_inputs()
    import numpy, pyarrow, torch
    return {"stage": "environment_probe", "gpu": gpu, **verified,
            "python_packages": {"torch": torch.__version__, "numpy": numpy.__version__,
                                "pyarrow": pyarrow.__version__},
            "weight_root": REMOTE_WEIGHT_ROOT, "source_dev_opened": False, "target_test_opened": False}


@app.function(image=image, volumes=VOLUMES, gpu=DEFAULT_GPU, timeout=3600)
def m8_gpat_smoke(steps: int = 5, resume_steps: int = 6, run_id: str = "gpat_m8_smoke_seed20260806") -> dict:
    _paths()
    gpu = _require_cuda()
    verified = _verify_inputs()
    from prism_fas.synthesis.gpat_trainer import GPATTrainer
    run_root = Path(REMOTE_RUNS_ROOT) / run_id
    trainer = GPATTrainer(config=_config(), package_root=Path(REMOTE_PACKAGE),
                          bank_root=PROJECT / "assets" / "recipe_banks" / "prism_recipe_bank_m7_v1",
                          pairs_root=_pairs_root(run_root), run_root=run_root,
                          weight_root=Path(REMOTE_WEIGHT_ROOT), device="cuda")
    result = trainer.smoke(steps=steps, resume_steps=resume_steps, run_id=run_id)
    runs_volume.commit()
    return {"stage": "gpat_smoke", "gpu": gpu, **verified, **result,
            "remote_run_path": str(run_root), "cuda_available": True}


@app.function(image=image, volumes=VOLUMES, gpu=DEFAULT_GPU, timeout=21600)
def m8_train_gpat(run_id: str = "gpat_m8_seed20260806", max_epochs: int | None = None,
                  resume: bool = True, limit_steps_per_epoch: int | None = None) -> dict:
    _paths()
    gpu = _require_cuda()
    verified = _verify_inputs()
    from prism_fas.synthesis.gpat_trainer import GPATTrainer
    from prism_fas.utils.core import atomic_json_write
    run_root = Path(REMOTE_RUNS_ROOT) / run_id
    trainer = GPATTrainer(config=_config(), package_root=Path(REMOTE_PACKAGE),
                          bank_root=PROJECT / "assets" / "recipe_banks" / "prism_recipe_bank_m7_v1",
                          pairs_root=_pairs_root(run_root), run_root=run_root,
                          weight_root=Path(REMOTE_WEIGHT_ROOT), device="cuda")

    def progress(payload: dict) -> None:
        print(json.dumps({"progress": payload}), flush=True)
        if payload.get("stage") == "epoch": runs_volume.commit()

    result = trainer.fit(run_id=run_id, progress=progress, max_epochs=max_epochs,
                         limit_steps_per_epoch=limit_steps_per_epoch, resume=resume)
    payload = {"stage": "train_gpat", "gpu": gpu, **verified, **result, "remote_run_path": str(run_root)}
    atomic_json_write(run_root / "gpat_training.json", payload)
    runs_volume.commit()
    return payload


@app.local_entrypoint()
def main(stage: str = "probe", steps: int = 5, resume_steps: int = 6, max_epochs: int = 0,
         limit_steps_per_epoch: int = 0, run_id: str = "", resume: bool = True) -> None:
    if stage == "probe":
        print(json.dumps(m8_environment_probe.remote(), indent=2, default=str))
    elif stage == "smoke":
        print(json.dumps(m8_gpat_smoke.remote(steps=steps, resume_steps=resume_steps,
                                              run_id=run_id or "gpat_m8_smoke_seed20260806"), indent=2, default=str))
    elif stage == "train":
        print(json.dumps(m8_train_gpat.remote(run_id=run_id or "gpat_m8_seed20260806",
                                              max_epochs=max_epochs or None,
                                              limit_steps_per_epoch=limit_steps_per_epoch or None),
                         indent=2, default=str))
    elif stage == "train_spawn":
        # Detached launch: prints the function call id so a later poll can attach
        # to the SAME run instead of starting a duplicate.
        call = m8_train_gpat.spawn(run_id=run_id or "gpat_m8_seed20260806",
                                   max_epochs=max_epochs or None, resume=resume,
                                   limit_steps_per_epoch=limit_steps_per_epoch or None)
        print(json.dumps({"spawned": True, "function_call_id": call.object_id,
                          "run_id": run_id or "gpat_m8_seed20260806"}))
    else:
        raise SystemExit(f"unknown stage {stage!r}")
