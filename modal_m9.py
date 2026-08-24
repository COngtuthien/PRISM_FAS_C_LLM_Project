"""M9 Modal wrapper: regional detector smoke, prototype initialization and the
single reference training run.

The wrapper only orchestrates. Every function calls the same core modules the local
CPU smoke uses; `src/prism_fas/detector/**` must never import modal (spec Table 41).

Source-only: these functions open `source_train`, the frozen validated M8 v3 bank
and — for checkpoint selection and the G6 calibration only — `source_dev`.
`target_test` is never opened.

Volumes are REUSED, never duplicated: `prism-fas-b-data`, `prism-fas-b-models`,
`prism-fas-b-runs` already exist and already hold the M3B package and the frozen
bank, so M9 re-uploads neither.
"""
from __future__ import annotations
import json
from pathlib import Path

import modal

APP_NAME = "prism-fas-b-m9"
DATA_VOLUME, MODELS_VOLUME, RUNS_VOLUME = "prism-fas-b-data", "prism-fas-b-models", "prism-fas-b-runs"
DATA_MOUNT, MODELS_MOUNT, RUNS_MOUNT = "/vol/data", "/vol/models", "/vol/runs"
REMOTE_PACKAGE = f"{DATA_MOUNT}/packages/prism_data_v1_m3b"
REMOTE_BANK = f"{DATA_MOUNT}/synthetic_banks/prism_synthetic_bank_m8_v3_e84c78cd2a9b"
REMOTE_WEIGHT_ROOT = f"{MODELS_MOUNT}/pretrained/m9"
REMOTE_RUNS_ROOT = f"{RUNS_MOUNT}/runs"
REMOTE_CACHE_ROOT = f"{RUNS_MOUNT}/m9_cache"
EXPECTED_PACKAGE_IDENTITY = "b1cf29b69a165ed5d9e074fc8127c17fbf057723edf9e272048ec3a564eb9dc6"
EXPECTED_BANK_IDENTITY = "e84c78cd2a9b548244e243de0380998d04bc6770b91caf32ac7be96f489bb542"
EXPECTED_BANK_ID = "prism_synthetic_bank_m8_v3_e84c78cd2a9b"
EXPECTED_RECIPE_BANK_IDENTITY = "fa989938cafdc4887518cc45c35d559d00278358439dc68c2486da10309210cb"
EXPECTED_SIGLIP2_IDENTITY = "7e059e40dcc34913b51fc8d7bd25e6f0c023bc238261effee9bfb87b33f04822"
EXPECTED_TEXT_CACHE_IDENTITY = "10f4ec35b7563b2b658cacc94599d35b9f93b531963a065459d4694d5dc2c141"
EXPECTED_TEXT_CACHE_SHA256 = "bb7d3fb4b82ad6ac89ebb06eeac9eb679e2fbb3bab500112cd1e304c187683aa"
GPU_ALLOW_LIST = ("L4", "L40S")
DEFAULT_GPU = "L4"
SMOKE_RUN_ID = "m9_reference_smoke_seed20260806"
REFERENCE_RUN_ID = "m9_reference_seed20260806"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1", "numpy==2.1.3", "pyarrow==18.1.0",
        "opencv-python-headless==4.10.0.84", "timm==1.0.11", "transformers==4.49.0",
        "safetensors==0.4.5", "sentencepiece==0.2.0", "pydantic==2.10.3", "PyYAML==6.0.2",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    # HF_HUB_OFFLINE: a silent Hub fetch would replace the verified SigLIP2 pin with
    # whatever is current upstream. That is a failure, not a fallback.
    .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONPATH": "/root/project/src",
          "PYTHONIOENCODING": "utf-8"})
    # Source, config and the small committed recipe bank only: never .git, data/,
    # runs/, reports/ or model_cache/.
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
RECIPE_BANK = PROJECT / "assets" / "recipe_banks" / "prism_recipe_bank_m7_v1"


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
            "torch": torch.__version__, "bf16_supported": bool(torch.cuda.is_bf16_supported())}


def _verify_siglip2() -> dict:
    """Re-hash every pinned SigLIP2 file INSIDE the container. A mismatch fails the
    function rather than downloading a newer model."""
    _paths()
    from prism_fas.detector.pretrained import SIGLIP2_PIN, SigLIP2Artifacts
    artifacts = SigLIP2Artifacts.resolve(Path(REMOTE_WEIGHT_ROOT), verify=True)
    identity = artifacts.identity()
    if identity != EXPECTED_SIGLIP2_IDENTITY:
        raise RuntimeError(f"in-container SigLIP2 identity {identity} != pinned {EXPECTED_SIGLIP2_IDENTITY}")
    return {"siglip2_identity_sha256": identity, "model_id": SIGLIP2_PIN["model_id"],
            "revision": SIGLIP2_PIN["revision"], "verified_sha256": dict(sorted(artifacts.digests.items())),
            "downloaded_at_runtime": False}


def _verify_inputs() -> dict:
    """Fail before allocating work if any input is not the exact frozen artifact."""
    _paths()
    lock = json.loads((Path(REMOTE_PACKAGE) / "PACKAGE_LOCK.json").read_text(encoding="utf-8"))
    package_identity = str(lock["content_identity_sha256"])
    if package_identity != EXPECTED_PACKAGE_IDENTITY:
        raise RuntimeError(f"remote package identity {package_identity} != {EXPECTED_PACKAGE_IDENTITY}")
    bank_lock = json.loads((Path(REMOTE_BANK) / "BANK_LOCK.json").read_text(encoding="utf-8"))
    bank_identity = str(bank_lock["bank_content_identity_sha256"])
    if bank_identity != EXPECTED_BANK_IDENTITY:
        raise RuntimeError(f"remote synthetic bank identity {bank_identity} != {EXPECTED_BANK_IDENTITY}")
    if str(bank_lock["bank_id"]) != EXPECTED_BANK_ID or str(bank_lock["status"]) != "validated":
        raise RuntimeError(f"remote bank is {bank_lock.get('bank_id')!r} / {bank_lock.get('status')!r}")
    recipe_lock = json.loads((RECIPE_BANK / "BANK_LOCK.json").read_text(encoding="utf-8"))
    recipe_identity = str(recipe_lock["bank_content_identity_sha256"])
    if recipe_identity != EXPECTED_RECIPE_BANK_IDENTITY:
        raise RuntimeError(f"recipe bank identity {recipe_identity} != {EXPECTED_RECIPE_BANK_IDENTITY}")
    from prism_fas.detector.pretrained import resolve_convnext_weight, sha256_file
    convnext = sha256_file(resolve_convnext_weight(Path(REMOTE_WEIGHT_ROOT)))
    # The frozen recipe text cache is an uploaded artifact, verified by both its file
    # bytes and its own recomputed identity. It is never rebuilt inside a run.
    cache_path = Path(REMOTE_WEIGHT_ROOT) / "recipe_text_cache.npz"
    cache_sha = sha256_file(cache_path)
    if cache_sha != EXPECTED_TEXT_CACHE_SHA256:
        raise RuntimeError(f"recipe text cache SHA {cache_sha} != pinned {EXPECTED_TEXT_CACHE_SHA256}")
    from prism_fas.detector.heads import resolve_recipe_text_cache
    cache = resolve_recipe_text_cache(Path(REMOTE_WEIGHT_ROOT), expected_identity=EXPECTED_TEXT_CACHE_IDENTITY)
    return {"package_identity": package_identity, "package_status": lock["status"],
            "recipe_text_cache_sha256": cache_sha, "recipe_text_cache_identity": cache.identity,
            "recipe_text_cache_recipes": cache.count, "recipe_text_cache_rebuilt": False,
            "package_split_counts": lock["per_split_counts"],
            "m8_bank_id": str(bank_lock["bank_id"]), "m8_bank_identity": bank_identity,
            "m8_bank_accepted": int(bank_lock["accepted_count"]),
            "recipe_bank_identity": recipe_identity, "convnext_weight_sha256": convnext,
            **_verify_siglip2()}


def _trainer(run_id: str, *, device: str = "cuda", overrides: dict | None = None,
             validation_limit: int | None = None):
    _paths()
    from dataclasses import replace
    from prism_fas.detector.config import load_m9_configs, verify_pinned_identities
    from prism_fas.detector.trainer import M9Trainer
    configs = load_m9_configs(PROJECT / "configs/models/m9_detector.yaml",
                              PROJECT / "configs/train/m9_reference.yaml")
    training = replace(configs["training_config"], run_id=run_id, **(overrides or {}))
    trainer = M9Trainer(config=training, detector_config=configs["detector_config"],
                        package_root=Path(REMOTE_PACKAGE), bank_root=Path(REMOTE_BANK),
                        recipe_bank_root=RECIPE_BANK, run_root=Path(REMOTE_RUNS_ROOT) / run_id,
                        cache_root=Path(REMOTE_CACHE_ROOT), weight_root=Path(REMOTE_WEIGHT_ROOT),
                        loader_config_path=PROJECT / "configs/data/loader_m4.yaml", device=device,
                        validation_limit=validation_limit,
                        progress=lambda payload: print(json.dumps(payload), flush=True))
    pins = verify_pinned_identities(
        configs["model_payload"], package_identity=trainer.dataset.package_identity,
        bank_identity=trainer.dataset.bank.identity, recipe_bank_identity=trainer.recipe_bank_identity,
        siglip2_identity=trainer.siglip.identity(), text_cache_identity=trainer.text_cache.identity)
    return trainer, {"pinned_identities_verified": pins,
                     "resolved_config_hash": configs["resolved_config_hash"]}


@app.function(image=image, volumes=VOLUMES, gpu=DEFAULT_GPU, timeout=900)
def m9_environment_probe() -> dict:
    _paths()
    gpu = _require_cuda()
    verified = _verify_inputs()
    import numpy, pyarrow, timm, torch, transformers
    return {"stage": "environment_probe", "app": APP_NAME, "gpu": gpu, **verified,
            "python_packages": {"torch": torch.__version__, "numpy": numpy.__version__,
                                "pyarrow": pyarrow.__version__, "timm": timm.__version__,
                                "transformers": transformers.__version__},
            "weight_root": REMOTE_WEIGHT_ROOT, "source_dev_opened": False,
            "target_test_opened": False}


@app.function(image=image, volumes=VOLUMES, gpu=DEFAULT_GPU, timeout=5400)
def m9_detector_smoke(steps: int = 5, resume_steps: int = 6, run_id: str = SMOKE_RUN_ID) -> dict:
    """5 real training steps, checkpoint, strict resume, then at least one more."""
    _paths()
    gpu = _require_cuda()
    verified = _verify_inputs()
    trainer, pins = _trainer(run_id, overrides={"steps_per_epoch": 45, "checkpoint_every_steps": 0},
                             validation_limit=256)
    result = trainer.smoke(steps=int(steps), resume_steps=int(resume_steps))
    summary = trainer.run_summary()
    runs_volume.commit()
    return {"stage": "detector_smoke", "gpu": gpu, **verified, **pins, "smoke": result,
            "run_summary": summary, "remote_run_path": str(Path(REMOTE_RUNS_ROOT) / run_id),
            "cuda_available": True, "target_test_opened": False}


@app.function(image=image, volumes=VOLUMES, gpu=DEFAULT_GPU, timeout=5400)
def m9_initialize_prototypes(run_id: str = "m9_prototype_init_seed20260806", repeats: int = 2) -> dict:
    """Run the G2 initialization `repeats` times independently and require an
    identical prototype content identity every time."""
    _paths()
    gpu = _require_cuda()
    verified = _verify_inputs()
    from prism_fas.detector.manifold import initialize_prototypes
    runs = []
    for index in range(int(repeats)):
        trainer, pins = _trainer(f"{run_id}_r{index}", overrides={"checkpoint_every_steps": 0})
        embeddings, valid, audit = trainer.collect_live_embeddings()
        state = initialize_prototypes(embeddings, valid, k=trainer.detector_config.prototype_k,
                                      epsilon=trainer.detector_config.covariance_epsilon,
                                      seed=trainer.config.prototype_seed)
        population = trainer.dataset.population_identity(trainer.dataset.live_positions())
        identity = state.identity(config_hash=trainer.config.hash(), population_identity=population)
        runs.append({"repeat": index, "prototype_identity_sha256": identity,
                     "population_identity_sha256": population, "k": state.k, "dim": state.dim,
                     "epsilon": state.epsilon, "counts": state.counts.tolist(),
                     "valid": state.valid.astype(int).tolist(), "audit": audit,
                     "centers_sha256": __import__("hashlib").sha256(
                         state.centers.round(10).tobytes()).hexdigest(),
                     "variances_sha256": __import__("hashlib").sha256(
                         state.variances.round(10).tobytes()).hexdigest(),
                     **pins})
    identities = {row["prototype_identity_sha256"] for row in runs}
    runs_volume.commit()
    return {"stage": "initialize_prototypes", "gpu": gpu, **verified, "repeats": int(repeats),
            "runs": runs, "identical_identity": len(identities) == 1,
            "prototype_identity_sha256": sorted(identities)[0] if len(identities) == 1 else None,
            "target_test_opened": False}


@app.function(image=image, volumes=VOLUMES, gpu=DEFAULT_GPU, timeout=86400)
def m9_train_reference(run_id: str = REFERENCE_RUN_ID, resume: bool = True) -> dict:
    """The single M9 reference run: G1 -> G2 -> G5 -> G6.

    One configuration, one seed. No K search, no baselines, no ablations, no target.
    """
    _paths()
    gpu = _require_cuda()
    verified = _verify_inputs()
    trainer, pins = _trainer(run_id)
    # The G1 -> G2 -> G5 -> G6 sequence, its resume rule and its per-stage evidence
    # merge live in `detector.trainer` because C7's search trials and C8's matrix
    # rows run the same flow. This entrypoint supplies the GPU, the volumes and
    # the pins; it does not own the flow.
    from prism_fas.detector.trainer import run_source_only_flow
    flow = run_source_only_flow(trainer, resume=resume)
    if flow["resumed_from"] is not None:
        print(json.dumps({"resumed": flow["resumed_from"], "stage": flow["resumed_stage"]}),
              flush=True)
    runs_volume.commit()
    return {"stage": "train_reference", "gpu": gpu, **verified, **pins,
            "resumed_from": flow["resumed_from"],
            "stages": flow["stages"], "run_summary": flow["run_summary"],
            "run_closure": flow["run_closure"],
            "source_isolation": flow["source_isolation"],
            "remote_run_path": str(Path(REMOTE_RUNS_ROOT) / run_id),
            "best_checkpoint": str(trainer.checkpoint_path("best")),
            "last_checkpoint": str(trainer.checkpoint_path("last")),
            "target_test_opened": False}


@app.function(image=image, volumes=VOLUMES, gpu=DEFAULT_GPU, timeout=5400)
def m9_validate_checkpoint(run_id: str = REFERENCE_RUN_ID, kind: str = "best") -> dict:
    """Re-open a checkpoint under strict identity and report its source-only
    metrics. Never reads target data."""
    _paths()
    gpu = _require_cuda()
    verified = _verify_inputs()
    from prism_fas.detector.checkpoint import checkpoint_summary, sha256_file
    trainer, pins = _trainer(run_id)
    path = trainer.checkpoint_path(kind)
    if not path.is_file(): raise RuntimeError(f"checkpoint {kind!r} does not exist for run {run_id!r}")
    restored = trainer.resume(kind)
    metrics = trainer.validate()
    return {"stage": "validate_checkpoint", "gpu": gpu, **verified, **pins, "kind": kind,
            "checkpoint_sha256": sha256_file(path), "checkpoint": checkpoint_summary(path),
            "restored": {key: value for key, value in restored.items() if key != "history"},
            "source_dev_metrics": metrics, "target_test_opened": False}


@app.local_entrypoint()
def main(action: str = "probe", run_id: str = "", steps: int = 5, resume_steps: int = 6,
         repeats: int = 2, kind: str = "best", resume: bool = True, report: str = "") -> None:
    """Thin dispatcher. Every action calls exactly one remote function.

    `--report` writes the returned JSON locally so the evidence is a file rather
    than console scrollback that has to be re-run to recover.
    """
    if action == "probe":
        result = m9_environment_probe.remote()
    elif action == "smoke":
        result = m9_detector_smoke.remote(steps=steps, resume_steps=resume_steps,
                                          run_id=run_id or SMOKE_RUN_ID)
    elif action == "prototypes":
        result = m9_initialize_prototypes.remote(run_id=run_id or "m9_prototype_init_seed20260806",
                                                 repeats=repeats)
    elif action == "train":
        result = m9_train_reference.remote(run_id=run_id or REFERENCE_RUN_ID, resume=resume)
    elif action == "validate":
        result = m9_validate_checkpoint.remote(run_id=run_id or REFERENCE_RUN_ID, kind=kind)
    else:
        raise SystemExit(f"unknown action {action!r}; expected probe|smoke|prototypes|train|validate")
    payload = json.dumps(result, indent=1, sort_keys=True, default=str)
    if report:
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        Path(report).write_text(payload, encoding="utf-8")
        print(f"wrote {report}")
    print(payload)
