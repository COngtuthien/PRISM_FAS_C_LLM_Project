"""M10 A02 control artifact: the random-operator synthetic bank, source-only.

Table 60's recipe ablation and hypothesis H4 need a control at EQUAL sample count
and EQUAL detector whose only difference from the structured arm is how the
operator composition was chosen. This wrapper builds it.

It reuses the M8 pipeline unchanged — the same `SyntheticBankGenerator`, the same
frozen `PhysicsEngine`, the same frozen GPAT checkpoint and the same frozen M8 v3
quality gate. The ONLY substitution is the recipe bank: `prism_recipe_bank_m10_random_v1`
in place of `prism_recipe_bank_m7_v1`.

Nothing here modifies an M7 or M8 artifact. The frozen M8 v3 bank is opened
READ-ONLY, purely to read the calibration files it already ships, and the M7 bank
is opened read-only to read the ontology. The output is a new, separately
identified, immutable bank.

Source-only: `source_train` payloads, the frozen recipe ontology and the pinned
quality/GPAT weights. `source_dev` and `target_test` are never opened, and no
target path, taxonomy or label is reachable from any function here.
"""
from __future__ import annotations
import json
from pathlib import Path

import modal

APP_NAME = "prism-fas-b-m10-a02"
DATA_VOLUME, MODELS_VOLUME, RUNS_VOLUME = "prism-fas-b-data", "prism-fas-b-models", "prism-fas-b-runs"
DATA_MOUNT, MODELS_MOUNT, RUNS_MOUNT = "/vol/data", "/vol/models", "/vol/runs"

REMOTE_PACKAGE = f"{DATA_MOUNT}/packages/prism_data_v1_m3b"
# The frozen M8 v3 bank. READ-ONLY, and only for the calibration files it ships.
REMOTE_M8_V3_BANK = f"{DATA_MOUNT}/synthetic_banks/prism_synthetic_bank_m8_v3_e84c78cd2a9b"
REMOTE_M8_WEIGHTS = f"{MODELS_MOUNT}/pretrained/m8"
REMOTE_RUNS_ROOT = f"{RUNS_MOUNT}/runs"
REMOTE_FROZEN_BANKS = f"{DATA_MOUNT}/synthetic_banks"
# The A02 working root. Separate from every M8 work root so a rerun of this
# artifact can never touch M8 state.
REMOTE_A02_WORK = f"{RUNS_MOUNT}/synthetic_banks/m10_a02_work"
REMOTE_A02_RECIPE_BANK = f"{REMOTE_A02_WORK}/recipe_bank"

GPAT_RUN_ID = "gpat_m8_seed20260806"
GPAT_CHECKPOINT = f"{REMOTE_RUNS_ROOT}/{GPAT_RUN_ID}/checkpoints/best.pt"
EXPECTED_GPAT_CHECKPOINT_SHA = "2047cdb513767010cfdf368c6f53a3664922451c56e1e837ec59cb96918a5b63"
EXPECTED_PAIR_PLAN_IDENTITY = "301868301dd11739ec018eed438704f9e4da7896ea52a0e60d50de563f2ccad3"
EXPECTED_PACKAGE_IDENTITY = "b1cf29b69a165ed5d9e074fc8127c17fbf057723edf9e272048ec3a564eb9dc6"
EXPECTED_M8_V3_BANK_IDENTITY = "e84c78cd2a9b548244e243de0380998d04bc6770b91caf32ac7be96f489bb542"
EXPECTED_M7_BANK_IDENTITY = "fa989938cafdc4887518cc45c35d559d00278358439dc68c2486da10309210cb"
# The A02 recipe bank is deterministic, so its identity is a pin, not an output.
EXPECTED_RANDOM_RECIPE_BANK_IDENTITY = "9351d08ac824cc67021445d1bb59bd9dc14ef7eb3dfa606414500d8fac49603f"
BANK_ID_PREFIX = "prism_synthetic_bank_m10_random"
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
M7_BANK = PROJECT / "assets" / "recipe_banks" / "prism_recipe_bank_m7_v1"


def _paths() -> None:
    import sys
    if "/root/project/src" not in sys.path: sys.path.insert(0, "/root/project/src")


def _require_cuda() -> dict:
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is not available inside a GPU function")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {"device": "cuda", "gpu_name": properties.name,
            "gpu_memory_gb": round(properties.total_memory / 1024 ** 3, 2),
            "torch": torch.__version__}


def _verify_inputs() -> dict:
    """Every frozen input this artifact binds, checked before any work starts."""
    _paths()
    package = json.loads((Path(REMOTE_PACKAGE) / "PACKAGE_LOCK.json").read_text(encoding="utf-8"))
    m8 = json.loads((Path(REMOTE_M8_V3_BANK) / "BANK_LOCK.json").read_text(encoding="utf-8"))
    m7 = json.loads((M7_BANK / "BANK_LOCK.json").read_text(encoding="utf-8"))
    problems = []
    if package.get("content_identity_sha256") != EXPECTED_PACKAGE_IDENTITY:
        problems.append("source package identity")
    if m8.get("bank_content_identity_sha256") != EXPECTED_M8_V3_BANK_IDENTITY:
        problems.append("M8 v3 bank identity")
    if m7.get("bank_content_identity_sha256") != EXPECTED_M7_BANK_IDENTITY:
        problems.append("M7 recipe bank identity")
    if problems: raise RuntimeError(f"frozen input mismatch: {problems}")
    return {"source_package_identity": package["content_identity_sha256"],
            "m8_v3_bank_identity": m8["bank_content_identity_sha256"],
            "m7_recipe_bank_identity": m7["bank_content_identity_sha256"],
            "source_dev_opened": False, "target_test_opened": False,
            "target_labels_opened": False}


def _build_recipe_bank() -> dict:
    """Materialize the deterministic A02 recipe bank and assert its pinned identity."""
    _paths()
    from prism_fas.synthesis.random_operator_bank import (build_random_operator_bank,
                                                          validate_random_operator_bank)
    result = build_random_operator_bank(Path(REMOTE_A02_RECIPE_BANK), M7_BANK / "ontology.yaml")
    identity = result["bank_content_identity_sha256"]
    if identity != EXPECTED_RANDOM_RECIPE_BANK_IDENTITY:
        raise RuntimeError(f"A02 recipe bank identity {identity} != pinned "
                           f"{EXPECTED_RANDOM_RECIPE_BANK_IDENTITY}")
    validation = validate_random_operator_bank(Path(REMOTE_A02_RECIPE_BANK))
    if not validation["passed"]: raise RuntimeError(f"A02 recipe bank failed validation: {validation['errors']}")
    return {"recipe_bank": result, "validation": validation}


def _plan_root() -> Path:
    """The A02 candidate plan: the same shape as M8's, over the same live targets,
    with the random recipes substituted. Rebuilt deterministically in-container."""
    _paths()
    from prism_fas.synthesis.candidate_plan import write_candidate_plan
    root = Path(REMOTE_A02_WORK) / "plan"
    write_candidate_plan(Path(REMOTE_PACKAGE), Path(REMOTE_A02_RECIPE_BANK),
                         PROJECT / "configs" / "synthesis" / "synthetic_bank_m8.yaml", root,
                         gpat_checkpoint_sha256=EXPECTED_GPAT_CHECKPOINT_SHA)
    return root


def _pairs_root() -> Path:
    """The frozen M8 pair plan, read-only. The GPAT spoof-source pairing is an M8
    artifact and is NOT re-derived here: A02 changes the recipe, not the pairing."""
    return Path(RUNS_MOUNT) / "synthetic_banks" / "m8_work" / "pairs"


def _conditioning_control():
    """The ONE declared A02 scientific control, constructed from frozen pins.

    The frozen GPAT checkpoint binds the STRUCTURED M7 bank it was trained on, so
    feeding it the random-operator conditioning vector needs this single named
    exemption. It authorizes exactly one pairing of exact identities; every other
    path in M8/M9/B08/M10 passes no control and keeps the full guard, and the object
    validates its own pins, so it cannot be used for anything else.

    The weights are unchanged, the quality gate is unchanged, and the
    out-of-training-conditioning-distribution caveat is recorded inside the
    identity this control contributes to the artifact.
    """
    _paths()
    from prism_fas.synthesis.conditioning_control import ConditioningBankControl
    return ConditioningBankControl.for_a02_random_operators(
        conditioning_recipe_bank_identity=EXPECTED_RANDOM_RECIPE_BANK_IDENTITY,
        gpat_checkpoint_sha256=EXPECTED_GPAT_CHECKPOINT_SHA,
        source_package_identity=EXPECTED_PACKAGE_IDENTITY)


def _generator(work_root: str, *, device: str = "cuda"):
    """The M8 v3 generator with ONLY the recipe bank substituted.

    The quality gate is the exact `quality_gate_v3.json` the frozen M8 v3 bank
    already ships, so acceptance is decided by the same thresholds that decided the
    structured arm. Nothing is re-calibrated and no threshold moves.
    """
    _paths()
    from prism_fas.synthesis.m8_pipeline import build_generator
    from prism_fas.synthesis.quality_calibration import QualityBackends
    calibration = Path(REMOTE_M8_V3_BANK) / "calibration"
    configs = PROJECT / "configs" / "synthesis"
    return build_generator(
        conditioning_control=_conditioning_control(),
        package_root=Path(REMOTE_PACKAGE), bank_root=Path(REMOTE_A02_RECIPE_BANK),
        work_root=Path(work_root), plan_root=_plan_root(),
        calibration_path=calibration / "quality_gate_v3.json",
        gpat_checkpoint_path=Path(GPAT_CHECKPOINT),
        backends=QualityBackends(Path(REMOTE_M8_WEIGHTS), device=device), device=device,
        quality_config_path=configs / "quality_gate_m8_v3.yaml",
        expected_gpat_sha256=EXPECTED_GPAT_CHECKPOINT_SHA,
        expected_pair_plan_identity=EXPECTED_PAIR_PLAN_IDENTITY,
        bank_id_prefix=BANK_ID_PREFIX,
        calibration_files={
            "identity_calibration_v2.json": calibration / "identity_calibration_v2.json",
            "structural_calibration_v3.json": calibration / "structural_calibration_v3.json",
            "IDENTITY_CALIBRATION_V2_LOCK.json": calibration / "IDENTITY_CALIBRATION_V2_LOCK.json",
            "STRUCTURAL_CALIBRATION_V3_LOCK.json": calibration / "STRUCTURAL_CALIBRATION_V3_LOCK.json"})


def _progress(payload: dict) -> None:
    print(json.dumps({"progress": payload}), flush=True)


@app.function(image=image, volumes=VOLUMES, timeout=1800, cpu=2.0, memory=8192)
def a02_plan() -> dict:
    """Build and validate the recipe bank and the candidate plan. No GPU, no pixels.

    Cheap enough to run before committing GPU time, and it is what proves the
    control is a real control: `off_manifold_any` counts the recipes whose
    composition the structured ontology would have refused.
    """
    _paths()
    verified = _verify_inputs()
    built = _build_recipe_bank()
    from prism_fas.synthesis.candidate_plan import load_candidate_plan
    plan_root = _plan_root()
    rows = load_candidate_plan(plan_root / "candidate_plan.parquet")
    lock = json.loads((plan_root / "CANDIDATE_PLAN_LOCK.json").read_text(encoding="utf-8"))
    runs_volume.commit()
    routes: dict = {}
    for row in rows: routes[row["route"]] = routes.get(row["route"], 0) + 1
    return {"stage": "a02_plan", **verified,
            "recipe_bank_identity": built["recipe_bank"]["bank_content_identity_sha256"],
            "recipe_bank_status": built["recipe_bank"]["status"],
            "recipe_bank_validation": built["validation"],
            "composition": built["recipe_bank"]["composition"],
            "candidate_plan_identity": lock["candidate_plan_identity_sha256"],
            "candidate_count": len(rows), "route_counts": routes,
            "distinct_live_samples": len({row["live_target_sample_id"] for row in rows}),
            "distinct_recipes": len({row["recipe_id"] for row in rows})}


@app.function(image=image, volumes=VOLUMES, gpu=DEFAULT_GPU, timeout=10800)
def a02_pilot(count: int = 32) -> dict:
    """A bounded correctness + determinism pilot before the full 1120."""
    _paths()
    gpu = _require_cuda()
    verified = _verify_inputs()
    _build_recipe_bank()
    from prism_fas.synthesis.m8_pipeline import run_pilot, run_pilot_determinism
    from prism_fas.utils.core import atomic_json_write
    generator = _generator(f"{REMOTE_A02_WORK}/_pilot")
    pilot = run_pilot(generator, pilot_root=Path(f"{REMOTE_A02_WORK}/_pilot"), count=count,
                      progress=_progress)
    audit = run_pilot_determinism(generator, first=pilot,
                                  rerun_root=Path(f"{REMOTE_A02_WORK}/_pilot_rerun"), count=count)
    strip = lambda payload: {key: value for key, value in payload.items() if key != "fingerprints"}
    for name, payload in (("a02_pilot.json", strip(pilot)), ("a02_pilot_determinism.json", strip(audit))):
        atomic_json_write(Path(REMOTE_A02_WORK) / "reports" / name, payload)
    runs_volume.commit()
    control = _conditioning_control()
    return {"stage": "a02_pilot", "gpu": gpu, **verified,
            "conditioning_control": control.policy_payload(),
            "conditioning_control_identity": control.identity(),
            "pilot": strip(pilot), "determinism": strip(audit)}


@app.function(image=image, volumes=VOLUMES, gpu=DEFAULT_GPU, timeout=21600)
def a02_generate_bank(interrupt_after: int | None = 96, determinism_audit: bool = True) -> dict:
    """Generate all 1120 A02 candidates, gate them with the FROZEN M8 v3 thresholds
    and freeze the resulting bank.

    `interrupt_after` deliberately interrupts the run once so the resume path is
    exercised on the real artifact rather than assumed, exactly as M8 did.
    """
    _paths()
    gpu = _require_cuda()
    verified = _verify_inputs()
    built = _build_recipe_bank()
    from prism_fas.synthesis.m8_pipeline import run_determinism_audit, run_full_generation
    from prism_fas.synthesis.synthetic_bank import assemble_bank
    from prism_fas.synthesis.synthetic_validation import validate_bank
    from prism_fas.utils.core import atomic_json_write
    generator = _generator(f"{REMOTE_A02_WORK}/generation")
    audit = run_full_generation(generator, interrupt_after=interrupt_after, progress=_progress)
    runs_volume.commit()
    assembled = assemble_bank(generator, audit["records"], pairs_root=_pairs_root())
    runs_volume.commit()
    bank_root = Path(assembled["bank_root"])
    validation = validate_bank(bank_root, package_root=Path(REMOTE_PACKAGE),
                               recipe_bank_root=Path(REMOTE_A02_RECIPE_BANK),
                               gpat_checkpoint_path=Path(GPAT_CHECKPOINT))
    determinism = {"skipped": True, "passed": True, "mismatch_count": 0}
    if determinism_audit:
        determinism = run_determinism_audit(generator, work_root=Path(f"{REMOTE_A02_WORK}/generation"),
                                            rerun_root=Path(f"{REMOTE_A02_WORK}/_determinism"),
                                            progress=_progress)
    resume_report = {key: value for key, value in audit.items() if key != "records"}
    for name, payload in (("a02_resume_audit.json", resume_report),
                          ("a02_determinism_audit.json", determinism),
                          ("a02_bank_validation.json", validation)):
        atomic_json_write(Path(REMOTE_A02_WORK) / "reports" / name, payload)
    runs_volume.commit()
    data_volume.commit()
    return {"stage": "a02_generate_bank", "gpu": gpu, **verified,
            "recipe_bank_identity": built["recipe_bank"]["bank_content_identity_sha256"],
            "bank": {key: assembled[key] for key in ("status", "bank_id", "bank_root")},
            "lock": assembled["lock"], "operational_minimums": assembled["operational_minimums"],
            "quality_summary": assembled["quality_summary"],
            "generation_summary": assembled["generation_summary"],
            "resume_audit": resume_report, "determinism_audit": determinism,
            "validation": {key: value for key, value in validation.items() if key != "checks"},
            "validation_checks": validation["checks"]}


@app.function(image=image, volumes=VOLUMES, timeout=3600, cpu=4.0, memory=16384)
def a02_verify(bank_id: str) -> dict:
    """Re-derive the frozen A02 bank's identity from the bytes on the volume.

    The same fail-closed check the training rows will perform, run once here so a
    row never discovers a broken artifact mid-training.
    """
    _paths()
    from prism_fas.detector.synthetic_bank import SyntheticBankReader
    from prism_fas.synthesis.random_operator_bank import validate_random_operator_bank
    root = Path(REMOTE_FROZEN_BANKS) / bank_id
    lock = json.loads((root / "BANK_LOCK.json").read_text(encoding="utf-8"))
    reader = SyntheticBankReader.open(root, expected_identity=lock["bank_content_identity_sha256"],
                                      expected_bank_id=bank_id)
    routes: dict = {}
    for row in reader.rows: routes[str(row["route"])] = routes.get(str(row["route"]), 0) + 1
    recipe_validation = validate_random_operator_bank(Path(REMOTE_A02_RECIPE_BANK))
    return {"stage": "a02_verify", "bank_id": reader.bank_id,
            "bank_content_identity_sha256": reader.identity,
            "status": lock.get("status"), "accepted": len(reader.rows),
            "route_counts": dict(sorted(routes.items())),
            "rows_identity_sha256": reader.rows_identity(),
            "recipe_bank_identity": recipe_validation["bank_content_identity_sha256"],
            "recipe_bank_passed": recipe_validation["passed"],
            "off_manifold_any": recipe_validation["composition_report"]["off_manifold_any"],
            "package_identity": str(lock.get("package_identity")),
            "target_labels_opened": False}


@app.function(image=image, volumes=VOLUMES, timeout=1800, cpu=2.0, memory=8192)
def a02_failures(work: str = "_pilot") -> dict:
    """Summarize the terminal states and the FAILURE REASONS of a generation root.

    A control whose candidates fail to generate is telling us something about the
    composition policy, and the report has to say what. This reads the per-candidate
    records rather than guessing from counts.
    """
    _paths()
    import collections
    root = Path(REMOTE_A02_WORK) / work / "records"
    states: collections.Counter = collections.Counter()
    reasons: collections.Counter = collections.Counter()
    stages: collections.Counter = collections.Counter()
    by_route: dict = {}
    example: dict = {}
    for path in sorted(root.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        state = str(record.get("terminal_state"))
        states[state] += 1
        route = str(record.get("route") or (record.get("plan") or {}).get("route") or "?")
        by_route.setdefault(route, collections.Counter())[state] += 1
        if state != "failed_generation": continue
        failure = record.get("failure") or {}
        stage = str(failure.get("stage") or record.get("stage") or "?")
        stages[stage] += 1
        reason = str(failure.get("reason") or failure.get("error") or record.get("error") or "")
        # Keep the operator/condition, drop the per-sample ids so reasons group.
        key = reason.split(":")[-1].strip()[:120] or reason[:120]
        reasons[key] += 1
        example.setdefault(key, {"synthetic_id": record.get("synthetic_id"), "route": route,
                                 "recipe_id": record.get("recipe_id"), "failure": failure})
    return {"stage": "a02_failures", "work_root": work,
            "terminal_states": dict(states),
            "failure_reasons": dict(reasons.most_common()),
            "failure_stages": dict(stages),
            "by_route": {route: dict(counter) for route, counter in sorted(by_route.items())},
            "examples": example, "records": sum(states.values())}


@app.local_entrypoint()
def main(action: str = "plan", count: int = 32, interrupt_after: int | None = 96,
         determinism_audit: bool = True, bank_id: str = "", work_root: str = "_pilot") -> None:
    if action == "plan":
        print(json.dumps(a02_plan.remote(), indent=1, default=str))
    elif action == "pilot":
        print(json.dumps(a02_pilot.remote(count), indent=1, default=str))
    elif action == "generate":
        print(json.dumps(a02_generate_bank.remote(interrupt_after, determinism_audit), indent=1, default=str))
    elif action == "failures":
        print(json.dumps(a02_failures.remote(work_root), indent=1, default=str))
    elif action == "verify":
        if not bank_id: raise SystemExit("--bank-id is required for verify")
        print(json.dumps(a02_verify.remote(bank_id), indent=1, default=str))
    else:
        raise SystemExit(f"unknown action {action!r}; use plan | pilot | generate | verify")
