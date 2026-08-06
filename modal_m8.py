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


@app.function(image=image, volumes=VOLUMES, gpu=DEFAULT_GPU, timeout=10800)
def m8_calibrate_quality(limit_live: int | None = None, limit_spoof: int | None = None) -> dict:
    """Source-train-only quality calibration. Opens source_train payloads only."""
    _paths()
    gpu = _require_cuda()
    verified = _verify_inputs()
    from prism_fas.synthesis.quality_calibration import QualityBackends, calibrate, load_quality_config, write_calibration
    config = load_quality_config(PROJECT / "configs" / "synthesis" / "quality_gate_m8.yaml")
    backends = QualityBackends(Path(REMOTE_WEIGHT_ROOT), device="cuda")
    payload = calibrate(Path(REMOTE_PACKAGE), config, backends, limit_live=limit_live, limit_spoof=limit_spoof,
                        progress=lambda item: print(json.dumps({"progress": item}), flush=True))
    target = Path(REMOTE_SYNTHETIC_WORK) / "calibration" / "quality_gate.json"
    enriched = write_calibration(target, {"gpu": gpu, **verified, **payload}, config=config)
    runs_volume.commit()
    return enriched


# --- M8 synthetic bank -------------------------------------------------------
GPAT_RUN_ID = "gpat_m8_seed20260806"
GPAT_CHECKPOINT = f"{REMOTE_RUNS_ROOT}/{GPAT_RUN_ID}/checkpoints/best.pt"
EXPECTED_GPAT_CHECKPOINT_SHA = "2047cdb513767010cfdf368c6f53a3664922451c56e1e837ec59cb96918a5b63"
EXPECTED_PAIR_PLAN_IDENTITY = "301868301dd11739ec018eed438704f9e4da7896ea52a0e60d50de563f2ccad3"
EXPECTED_CANDIDATE_PLAN_IDENTITY = "b167c169dcb92426c0dc2ee96a80eb69f4645fbf887360a1b67abfc8890f40b8"
REMOTE_GENERATION = f"{REMOTE_SYNTHETIC_WORK}/generation"
REMOTE_CALIBRATION = f"{REMOTE_SYNTHETIC_WORK}/calibration/quality_gate.json"
REMOTE_FROZEN_BANKS = f"{DATA_MOUNT}/synthetic_banks"
BANK_ROOT = "assets/recipe_banks/prism_recipe_bank_m7_v1"


def _bank_root():
    return PROJECT / BANK_ROOT


def _plan_root() -> "Path":
    """Rebuild the deterministic candidate plan in-container and assert its frozen
    identity. The plan is a pure function of the package, the bank, the seed and
    the checkpoint SHA, so this reproduces the local artifact exactly."""
    _paths()
    from prism_fas.synthesis.candidate_plan import write_candidate_plan
    root = Path(REMOTE_SYNTHETIC_WORK) / "plan"
    result = write_candidate_plan(Path(REMOTE_PACKAGE), _bank_root(),
                                  PROJECT / "configs" / "synthesis" / "synthetic_bank_m8.yaml", root,
                                  gpat_checkpoint_sha256=EXPECTED_GPAT_CHECKPOINT_SHA)
    identity = result["lock"]["candidate_plan_identity_sha256"]
    if identity != EXPECTED_CANDIDATE_PLAN_IDENTITY:
        raise RuntimeError(f"rebuilt candidate plan identity {identity} != {EXPECTED_CANDIDATE_PLAN_IDENTITY}")
    return root


def _pair_plan_root() -> "Path":
    root = _pairs_root(Path(REMOTE_SYNTHETIC_WORK))
    identity = json.loads((root / "PAIR_PLAN_LOCK.json").read_text(encoding="utf-8"))["pair_plan_identity_sha256"]
    if identity != EXPECTED_PAIR_PLAN_IDENTITY:
        raise RuntimeError(f"rebuilt pair plan identity {identity} != {EXPECTED_PAIR_PLAN_IDENTITY}")
    return root


def _generator(work_root: str, *, device: str = "cuda"):
    _paths()
    from prism_fas.synthesis.m8_pipeline import build_generator
    from prism_fas.synthesis.quality_calibration import QualityBackends
    return build_generator(
        package_root=Path(REMOTE_PACKAGE), bank_root=_bank_root(), work_root=Path(work_root),
        plan_root=_plan_root(), calibration_path=Path(REMOTE_CALIBRATION),
        gpat_checkpoint_path=Path(GPAT_CHECKPOINT),
        backends=QualityBackends(Path(REMOTE_WEIGHT_ROOT), device=device), device=device,
        expected_gpat_sha256=EXPECTED_GPAT_CHECKPOINT_SHA,
        expected_pair_plan_identity=EXPECTED_PAIR_PLAN_IDENTITY)


def _progress(payload: dict) -> None:
    print(json.dumps({"progress": payload}), flush=True)


REMOTE_SYNTHETIC_WORK_V2 = f"{RUNS_MOUNT}/synthetic_banks/m8_work_v2"
REMOTE_CALIBRATION_V2 = f"{REMOTE_SYNTHETIC_WORK_V2}/calibration"


@app.function(image=image, volumes=VOLUMES, gpu=DEFAULT_GPU, timeout=7200)
def m8_calibrate_identity_v2() -> dict:
    """Source-train-only cross-observation identity calibration.

    Opens the `source_train` manifest and `source_train` live images only. No
    generated candidate, no `source_dev`, no target. Runs twice and refuses to
    write unless both runs agree.
    """
    _paths()
    gpu = _require_cuda()
    verified = _verify_inputs()
    from prism_fas.synthesis.identity_calibration import (IdentityBackend, calibrate_identity,
                                                          compare_calibrations, load_identity_config,
                                                          write_calibration_artifacts)
    from prism_fas.synthesis.synthetic_bank import FrozenCalibration
    config = load_identity_config(PROJECT / "configs" / "synthesis" / "quality_gate_m8_v2.yaml")
    v1 = FrozenCalibration.load(Path(REMOTE_CALIBRATION))
    backends = IdentityBackend(Path(REMOTE_WEIGHT_ROOT), device="cuda")
    runs = [calibrate_identity(Path(REMOTE_PACKAGE), config, backends, v1_thresholds=v1.thresholds,
                               progress=_progress, device_report=gpu) for _ in range(2)]
    comparison = compare_calibrations(runs[0], runs[1])
    if not comparison["identical"]:
        raise RuntimeError(f"v2 calibration is not reproducible: {comparison['mismatches'][:5]}")
    written = write_calibration_artifacts(Path(REMOTE_CALIBRATION_V2), runs[0],
                                          package_identity=verified["package_identity"], config=config)
    runs_volume.commit()
    summary = {key: value for key, value in written["calibration"].items() if key != "identity_structure"}
    return {"stage": "calibrate_identity_v2", "gpu": gpu, **verified,
            "calibration": summary, "lock": written["lock"], "determinism": comparison,
            "identity_structure": written["calibration"]["identity_structure"],
            "remote_calibration_path": REMOTE_CALIBRATION_V2, "written": written["written"],
            "v1_tau_id": v1.thresholds.tau_id, "v1_threshold_sha256": v1.threshold_sha256}


@app.function(image=image, volumes=VOLUMES, gpu=DEFAULT_GPU, timeout=7200)
def m8_generation_pilot(count: int = 32, determinism: bool = True) -> dict:
    """Deterministic correctness pilot. Never changes a frozen threshold."""
    _paths()
    gpu = _require_cuda()
    verified = _verify_inputs()
    from prism_fas.synthesis.m8_pipeline import run_pilot, run_pilot_determinism
    generator = _generator(f"{REMOTE_SYNTHETIC_WORK}/_pilot")
    pilot = run_pilot(generator, pilot_root=Path(f"{REMOTE_SYNTHETIC_WORK}/_pilot"), count=count,
                      progress=_progress)
    audit = {"passed": True, "mismatch_count": 0, "skipped": True}
    if determinism:
        audit = run_pilot_determinism(generator, first=pilot,
                                      rerun_root=Path(f"{REMOTE_SYNTHETIC_WORK}/_pilot_rerun"), count=count)
    runs_volume.commit()
    return {"stage": "generation_pilot", "gpu": gpu, **verified,
            "pilot": {key: value for key, value in pilot.items() if key != "fingerprints"},
            "determinism": {key: value for key, value in audit.items() if key != "fingerprints"}}


@app.function(image=image, volumes=VOLUMES, gpu=DEFAULT_GPU, timeout=21600)
def m8_generate_bank(interrupt_after: int | None = 64, determinism_audit: bool = True) -> dict:
    """Generate the whole frozen plan, run the resume and determinism audits, then
    assemble, validate, freeze and export the bank."""
    _paths()
    gpu = _require_cuda()
    verified = _verify_inputs()
    from prism_fas.synthesis.m8_pipeline import run_determinism_audit, run_full_generation
    from prism_fas.synthesis.synthetic_bank import assemble_bank
    from prism_fas.synthesis.synthetic_validation import validate_bank
    from prism_fas.utils.core import atomic_json_write
    generator = _generator(REMOTE_GENERATION)
    audit = run_full_generation(generator, interrupt_after=interrupt_after, progress=_progress)
    runs_volume.commit()
    assembled = assemble_bank(generator, audit["records"], pairs_root=_pair_plan_root())
    runs_volume.commit()
    bank_root = Path(assembled["bank_root"])
    validation = validate_bank(bank_root, package_root=Path(REMOTE_PACKAGE), recipe_bank_root=_bank_root(),
                               gpat_checkpoint_path=Path(GPAT_CHECKPOINT))
    determinism = {"skipped": True, "passed": True, "mismatch_count": 0}
    if determinism_audit:
        determinism = run_determinism_audit(generator, work_root=Path(REMOTE_GENERATION),
                                            rerun_root=Path(f"{REMOTE_SYNTHETIC_WORK}/_determinism"),
                                            progress=_progress)
    resume_report = {key: value for key, value in audit.items() if key != "records"}
    for name, payload in (("resume_audit.json", resume_report),
                          ("determinism_audit.json", determinism),
                          ("synthetic_bank_validation.json", validation)):
        atomic_json_write(Path(REMOTE_SYNTHETIC_WORK) / "reports" / name, payload)
    runs_volume.commit()
    return {"stage": "generate_bank", "gpu": gpu, **verified,
            "bank": {key: assembled[key] for key in ("status", "bank_id", "bank_root")},
            "lock": assembled["lock"], "operational_minimums": assembled["operational_minimums"],
            "quality_summary": assembled["quality_summary"],
            "generation_summary": assembled["generation_summary"],
            "resume_audit": resume_report, "determinism_audit": determinism,
            "validation": {key: value for key, value in validation.items() if key != "checks"},
            "validation_checks": validation["checks"]}


@app.function(image=image, volumes=VOLUMES, gpu=DEFAULT_GPU, timeout=10800)
def m8_reassemble_bank(export: bool = True, allow_unvalidated: bool = False) -> dict:
    """Re-assemble and re-validate the bank from the terminal records already on the
    volume, then write the transport archive.

    Every valid record is reused, so nothing is regenerated: this is the completed
    rerun, and it is how a retained run is re-locked after a code fix.
    """
    _paths()
    gpu = _require_cuda()
    from prism_fas.synthesis.synthetic_bank import assemble_bank
    from prism_fas.synthesis.synthetic_export import export_archive
    from prism_fas.synthesis.synthetic_validation import validate_bank
    from prism_fas.utils.core import atomic_json_write
    generator = _generator(REMOTE_GENERATION)
    rerun = generator.run(resume=True, progress=_progress)
    if rerun["rebuilt"]:
        raise RuntimeError(f"a completed rerun must rebuild nothing, rebuilt {rerun['rebuilt']}")
    assembled = assemble_bank(generator, rerun["records"], pairs_root=_pair_plan_root())
    runs_volume.commit()
    bank_root = Path(assembled["bank_root"])
    validation = validate_bank(bank_root, package_root=Path(REMOTE_PACKAGE), recipe_bank_root=_bank_root(),
                               gpat_checkpoint_path=Path(GPAT_CHECKPOINT))
    atomic_json_write(Path(REMOTE_SYNTHETIC_WORK) / "reports" / "synthetic_bank_validation.json", validation)
    archive = {"status": "skipped"}
    if export:
        Path(REMOTE_EXPORTS).mkdir(parents=True, exist_ok=True)
        archive = export_archive(bank_root, Path(REMOTE_EXPORTS),
                                 require_validated=not allow_unvalidated)
        atomic_json_write(Path(REMOTE_SYNTHETIC_WORK) / "reports" / "export.json", archive)
    runs_volume.commit()
    return {"stage": "reassemble_bank", "gpu": gpu,
            "rerun": {key: rerun[key] for key in ("status", "planned", "examined", "reused", "rebuilt")},
            "bank": {key: assembled[key] for key in ("status", "bank_id", "bank_root")},
            "lock_status": assembled["lock"]["status"],
            "bank_content_identity_sha256": assembled["lock"]["bank_content_identity_sha256"],
            "operational_minimums": assembled["operational_minimums"],
            "validation_passed": validation["passed"], "validation_errors": validation["errors"],
            "validation_checks": validation["checks"],
            "archive": archive,
            "archive_remote_path": (f"{REMOTE_EXPORTS}/{assembled['bank_id']}.tar"
                                    .replace(RUNS_MOUNT, "prism-fas-b-runs:"))}


@app.function(image=image, volumes=VOLUMES, timeout=7200)
def m8_validate_bank(bank_id: str = "", bank_root: str = "") -> dict:
    _paths()
    verified = _verify_inputs()
    from prism_fas.synthesis.synthetic_validation import validate_bank
    root = Path(bank_root or f"{REMOTE_GENERATION}/{bank_id}")
    report = validate_bank(root, package_root=Path(REMOTE_PACKAGE), recipe_bank_root=_bank_root(),
                           gpat_checkpoint_path=Path(GPAT_CHECKPOINT))
    return {"stage": "validate_bank", **verified, **report}


@app.function(image=image, volumes=VOLUMES, timeout=7200)
def m8_export_bank(bank_id: str) -> dict:
    """Freeze the validated bank under its immutable versioned path and write the
    single deterministic transport archive."""
    _paths()
    from prism_fas.synthesis.synthetic_export import export_archive, freeze_bank
    from prism_fas.synthesis.synthetic_validation import validate_bank
    source = Path(REMOTE_GENERATION) / bank_id
    report = validate_bank(source, package_root=Path(REMOTE_PACKAGE), recipe_bank_root=_bank_root(),
                           gpat_checkpoint_path=Path(GPAT_CHECKPOINT))
    if not report["passed"]:
        raise RuntimeError(f"refusing to export an invalid bank: {report['errors'][:5]}")
    frozen = freeze_bank(source, Path(REMOTE_FROZEN_BANKS))
    data_volume.commit()
    Path(REMOTE_EXPORTS).mkdir(parents=True, exist_ok=True)
    archive = export_archive(source, Path(REMOTE_EXPORTS))
    runs_volume.commit()
    return {"stage": "export_bank", "bank_id": bank_id, "frozen": frozen, "archive": archive,
            "frozen_remote_path": f"{REMOTE_FROZEN_BANKS}/{bank_id}".replace(DATA_MOUNT, "prism-fas-b-data:"),
            "archive_remote_path": f"{REMOTE_EXPORTS}/{bank_id}.tar".replace(RUNS_MOUNT, "prism-fas-b-runs:"),
            "validation_passed": report["passed"],
            "bank_content_identity_sha256": report["bank_content_identity_sha256"]}


@app.local_entrypoint()
def main(stage: str = "probe", steps: int = 5, resume_steps: int = 6, max_epochs: int = 0,
         limit_steps_per_epoch: int = 0, run_id: str = "", resume: bool = True,
         count: int = 32, interrupt_after: int = 64, bank_id: str = "",
         allow_unvalidated: bool = False) -> None:
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
    elif stage == "calibrate":
        print(json.dumps(m8_calibrate_quality.remote(), indent=2, default=str))
    elif stage == "calibrate_identity_v2":
        print(json.dumps(m8_calibrate_identity_v2.remote(), indent=2, default=str))
    elif stage == "pilot":
        print(json.dumps(m8_generation_pilot.remote(count=count), indent=2, default=str))
    elif stage == "pilot_spawn":
        call = m8_generation_pilot.spawn(count=count)
        print(json.dumps({"spawned": True, "function_call_id": call.object_id, "stage": "pilot"}))
    elif stage == "generate":
        print(json.dumps(m8_generate_bank.remote(interrupt_after=interrupt_after or None), indent=2, default=str))
    elif stage == "generate_spawn":
        # Detached launch: the printed call id lets a later poll attach to the SAME
        # run instead of starting a duplicate generation job.
        call = m8_generate_bank.spawn(interrupt_after=interrupt_after or None)
        print(json.dumps({"spawned": True, "function_call_id": call.object_id, "stage": "generate"}))
    elif stage == "reassemble":
        print(json.dumps(m8_reassemble_bank.remote(allow_unvalidated=allow_unvalidated), indent=2, default=str))
    elif stage == "validate_bank":
        print(json.dumps(m8_validate_bank.remote(bank_id=bank_id), indent=2, default=str))
    elif stage == "export":
        if not bank_id: raise SystemExit("--bank-id is required to export")
        print(json.dumps(m8_export_bank.remote(bank_id=bank_id), indent=2, default=str))
    elif stage == "calibrate_spawn":
        call = m8_calibrate_quality.spawn()
        print(json.dumps({"spawned": True, "function_call_id": call.object_id, "stage": "calibrate"}))
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
