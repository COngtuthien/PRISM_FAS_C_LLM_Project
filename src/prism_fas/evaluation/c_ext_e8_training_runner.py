"""PRISM-FAS-C EXT-Q1Q2 -- E8 EXT-F1 fixed-config scientific training runner.

Orchestrates ONE frozen Track-G training run against an E8 q-matched bank,
using ONLY existing primitives:

    run spec (arm, seed) -> derive run_id -> validate against the frozen
    15-run matrix -> verify the E8 GPU execution plan identity -> load the
    frozen winner Track-G config (prism_fas.detector.config.load_m9_configs
    + the C7 winner_config delta, exactly the pattern
    prism_fas.pipeline.adapters.c7._scientific_trial_config already uses) ->
    open the E8 arm bank (prism_fas.evaluation.c_ext_e8_training_adapter.
    open_e8_arm_bank) -> classify run-directory collision state -> construct
    prism_fas.detector.trainer.M9Trainer(..., synthetic_bank=e8_bank) -> run
    the EXISTING trainer.

This module implements NO new training loop, optimizer, scheduler, dataset,
sampler, checkpoint-selection rule, q-weighting, or metric. It never
modifies trainer.py, dataset.py, sampler.py, c6_bank.py, synthetic_bank.py,
or any C7/E7 module. Scientific-mode run IDs are DERIVED, never freeform;
scientific mode exposes no hyperparameter flags (no --lr/--epochs/--batch-
size/--weight-decay) -- its whole purpose is to prevent config drift.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prism_fas.evaluation import c_ext_common as cc  # noqa: E402
from prism_fas.evaluation import c_ext_e8_training_adapter as adapter  # noqa: E402
from prism_fas.evaluation.c_ext_e8_qmatched import PRIMARY_FOLD  # noqa: E402

# --------------------------------------------------------------------------- #
# Frozen constants -- ratified upstream (E8 amendment, selector, adapter,
# execution plan); never redefine, never recompute drifted values here.
# --------------------------------------------------------------------------- #

ARMS = adapter.ARMS
CONDITION_BY_ARM = adapter.CONDITION_BY_ARM
SEEDS: tuple[int, ...] = (20260806, 20260807, 20260808, 20260809, 20260810)
SHUFFLE_SEEDS_EXCLUDED: tuple[int, ...] = (20260911, 20260912, 20260913)

FROZEN_MEMBERSHIP_PARQUET_SHA256 = adapter.FROZEN_MEMBERSHIP_PARQUET_SHA256
FROZEN_MEMBERSHIP_LOCK_SHA256 = adapter.FROZEN_MEMBERSHIP_LOCK_SHA256
SELECTOR_RULE_IDENTITY = adapter.SELECTOR_RULE_IDENTITY
INPUT_BINDING_RULE_IDENTITY = adapter.INPUT_BINDING_RULE_IDENTITY
EXPECTED_ADAPTER_RULE_IDENTITY = "56e649a49ba03febbadce6fc62d4362be753d95ccd0fce03d1886f8c5384bbdb"
ADAPTER_IMPLEMENTATION_COMMIT = "9fb4e2fc7d30ad940e42999524f0d09b81bbf0e5"
EXPECTED_EXECUTION_PLAN_IDENTITY = "74f9503f1fa22ffe96eca6585ff1184b3f6bc122380a3e39c485a35dc02a39de"
TRACK_G_VARIANT_IDENTITY = adapter.TRACK_G_VARIANT_IDENTITY
C7_WINNER_CONFIG_SHA256 = adapter.C7_WINNER_CONFIG_SHA256

EXPECTED_PHYSICS_PER_ARM = adapter.EXPECTED_PHYSICS_PER_ARM
EXPECTED_GPAT_PER_ARM = adapter.EXPECTED_GPAT_PER_ARM
EXPECTED_COUNT_PER_ARM = adapter.EXPECTED_COUNT_PER_ARM
FROZEN_TOTAL_EPOCHS = 35
FROZEN_STEPS_PER_EPOCH = 45
FROZEN_TOTAL_OPTIMIZER_UPDATES = 1575
FROZEN_SYNTHETIC_DRAWS_PER_RUN = 10800

RUN_ROOT_RELATIVE = "runs/c_ext_q1q2_v1/e8_qmatched/ext_f1"
SMOKE_ROOT_RELATIVE = "runs/c_ext_q1q2_v1/e8_qmatched/smoke"
SMOKE_RUN_ID = "e8_ext_f1_adapter_smoke"

EXECUTION_PLAN_RELATIVE_PATH = "reports/c_ext_q1q2_v1/e8_qmatched/training/execution/E8_GPU_EXECUTION_PLAN.json"
EXECUTION_PLAN_EVIDENCE_RELATIVE_PATH = "reports/c_ext_q1q2_v1/e8_qmatched/training/execution/E8_GPU_EXECUTION_PLAN.sha256"
DETECTOR_CONFIG_LOCK_RELATIVE_PATH = "reports/full/c7/DETECTOR_CONFIG_LOCK.json"
MODEL_CONFIG_RELATIVE_PATH = "configs/models/m9_detector.yaml"
TRAINING_CONFIG_RELATIVE_PATH = "configs/train/m9_reference.yaml"
LOADER_CONFIG_RELATIVE_PATH = "configs/data/loader_m4.yaml"
SOURCE_PACKAGE_RELATIVE_PATH = "data/processed/c_ext_q1q2_v1/e7_gpat_bank/gpat_input/EXT-F1"
EXT_F1_SOURCE_PACKAGE_IDENTITY = "955b630fec438c80f284ecbcb30fbf10c83251a23fd31d8ab1a52e0f8ce8383b"
ALLOWED_SOURCE_DOMAINS = frozenset({"casia_fasd", "msu_mfsd"})

RUNNER_RULE_NAME = "E8_FIXED_TRACK_G_RUNNER_V1"
FAILURE_MARKER_NAME = "E8_RUN_FAILED.json"


class E8RunnerError(RuntimeError):
    """The E8 training runner refuses to proceed."""


class RunState(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED_TECHNICAL = "FAILED_TECHNICAL"
    BLOCKED_COLLISION = "BLOCKED_COLLISION"


def _repo_root(root: Path | None) -> Path:
    return root or cc.repo_root()


# --------------------------------------------------------------------------- #
# Run-ID derivation and validation (scientific mode: no freeform IDs)
# --------------------------------------------------------------------------- #

def derive_run_id(arm: str, seed: int) -> str:
    if arm not in ARMS:
        raise E8RunnerError(f"unknown arm {arm!r}; expected one of {ARMS!r}")
    if seed not in SEEDS:
        raise E8RunnerError(f"unknown seed {seed!r}; expected one of {SEEDS!r}")
    return f"e8_ext_f1_g_{arm.lower()}_qmatch_s{seed}"


# --------------------------------------------------------------------------- #
# Fixed, typed, immutable run specification
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class E8RunSpec:
    fold_id: str
    condition: str
    arm: str
    seed: int
    run_id: str
    run_root: str  # repo-relative POSIX path, as a string
    membership_sha256: str
    membership_lock_sha256: str
    selector_rule_identity: str
    input_binding_rule_identity: str
    adapter_rule_identity: str
    adapter_implementation_commit: str
    execution_plan_identity: str
    track_g_variant_identity: str
    c7_winner_config_sha256: str


def verify_execution_plan(root: Path | None = None) -> dict:
    """Load + evidence-verify the frozen E8 GPU execution plan. Hard-fails on
    any evidence mismatch or identity drift. Never regenerates the plan."""
    repo = _repo_root(root)
    evidence_path = repo / EXECUTION_PLAN_EVIDENCE_RELATIVE_PATH
    if not evidence_path.is_file():
        raise E8RunnerError(f"{evidence_path}: execution-plan evidence manifest missing")
    for line in evidence_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        expected_sha, name = line.split(None, 1)
        candidate = evidence_path.parent / name.strip()
        actual = cc.sha256_file(candidate)
        if actual != expected_sha:
            raise E8RunnerError(f"{candidate}: execution-plan evidence SHA256 {actual} != {expected_sha}")
    plan_path = repo / EXECUTION_PLAN_RELATIVE_PATH
    plan = cc.read_json(plan_path)
    actual_identity = plan.get("e8_gpu_execution_plan_identity")
    if actual_identity != EXPECTED_EXECUTION_PLAN_IDENTITY:
        raise E8RunnerError(
            f"execution-plan identity {actual_identity!r} != frozen expected "
            f"{EXPECTED_EXECUTION_PLAN_IDENTITY!r} -- refusing to run against a drifted plan"
        )
    return plan


def build_run_spec(arm: str, seed: int, *, run_id: str | None = None,
                   root: Path | None = None, verify_plan: bool = True) -> E8RunSpec:
    """Build and validate one scientific run's immutable spec.

    Hard-fails on: unknown arm; unknown seed; a supplied ``run_id`` that
    disagrees with the derived one; an adapter rule identity that has
    drifted from the frozen value; an execution plan that has drifted from
    its frozen identity (unless ``verify_plan=False``, used only by tests
    that intentionally exercise a corrupted plan fixture).
    """
    derived = derive_run_id(arm, seed)
    if run_id is not None and run_id != derived:
        raise E8RunnerError(f"run_id {run_id!r} != derived {derived!r} for arm={arm!r} seed={seed!r}")

    actual_adapter_identity = adapter.adapter_rule_identity(root)
    if actual_adapter_identity != EXPECTED_ADAPTER_RULE_IDENTITY:
        raise E8RunnerError(
            f"adapter_rule_identity {actual_adapter_identity!r} != frozen expected "
            f"{EXPECTED_ADAPTER_RULE_IDENTITY!r}"
        )
    if verify_plan:
        verify_execution_plan(root)

    return E8RunSpec(
        fold_id=PRIMARY_FOLD, condition=CONDITION_BY_ARM[arm], arm=arm, seed=int(seed), run_id=derived,
        run_root=f"{RUN_ROOT_RELATIVE}/{derived}",
        membership_sha256=FROZEN_MEMBERSHIP_PARQUET_SHA256,
        membership_lock_sha256=FROZEN_MEMBERSHIP_LOCK_SHA256,
        selector_rule_identity=SELECTOR_RULE_IDENTITY,
        input_binding_rule_identity=INPUT_BINDING_RULE_IDENTITY,
        adapter_rule_identity=EXPECTED_ADAPTER_RULE_IDENTITY,
        adapter_implementation_commit=ADAPTER_IMPLEMENTATION_COMMIT,
        execution_plan_identity=EXPECTED_EXECUTION_PLAN_IDENTITY,
        track_g_variant_identity=TRACK_G_VARIANT_IDENTITY,
        c7_winner_config_sha256=C7_WINNER_CONFIG_SHA256,
    )


def all_scientific_run_specs(root: Path | None = None, *, verify_plan: bool = True) -> tuple[E8RunSpec, ...]:
    """The frozen 15 (arm, seed) run specs, in RND/DET/LLM x seed-ascending order."""
    return tuple(
        build_run_spec(arm, seed, root=root, verify_plan=verify_plan)
        for arm in ARMS for seed in SEEDS
    )


# --------------------------------------------------------------------------- #
# Frozen winner Track-G config (loaded, never hand-duplicated)
# --------------------------------------------------------------------------- #

def load_frozen_winner_track_g_config(spec: E8RunSpec, synthetic_bank_identity: str,
                                      root: Path | None = None):
    """Load the base M9 config via the repository's own
    ``prism_fas.detector.config.load_m9_configs``, then apply the frozen C7
    Track-G winner delta -- exactly the ``dataclasses.replace`` pattern
    ``prism_fas.pipeline.adapters.c7._scientific_trial_config`` already uses
    for arm-conditioned production training. Returns
    ``(training_config, detector_config)``.

    ``backbone_lr``/``head_lr`` are intentionally NOT re-derived through the
    LR-anchor-multiplier mechanism (``prism_fas.search.lr_decision``):
    Track G's component there is ``UNIQUE_INHERITED_ANCHOR``, and this
    implementation did not conclusively resolve what a nominal
    ``learning_rate_multiplier=2.0`` means for that interpretation (the
    literal ``anchor * multiplier`` computation disagrees with the value
    already resolved and frozen in ``EXT_MODEL_BINDING.json``). Rather than
    guess, this function asserts that the base config's own LR defaults
    already equal the frozen, independently-resolved binding
    (backbone_lr=1e-5, head_lr=1e-4) and leaves them untouched -- a real
    equality check, not a fabricated derivation.
    """
    from prism_fas.detector.config import load_m9_configs

    repo = _repo_root(root)
    configs = load_m9_configs(repo / MODEL_CONFIG_RELATIVE_PATH, repo / TRAINING_CONFIG_RELATIVE_PATH)
    base = configs["training_config"]

    lock = cc.read_json(repo / DETECTOR_CONFIG_LOCK_RELATIVE_PATH)
    track_g = lock["tracks"]["G"]
    winner_config = track_g["winner_config"]
    recomputed = cc.sha256_bytes(json.dumps(winner_config, sort_keys=True,
                                            separators=(",", ":")).encode("utf-8"))
    if recomputed != C7_WINNER_CONFIG_SHA256:
        raise E8RunnerError(
            f"recomputed Track-G winner_config SHA256 {recomputed} != frozen expected "
            f"{C7_WINNER_CONFIG_SHA256} -- refusing to train against a drifted winner config"
        )
    if track_g.get("winner_config_sha256") != C7_WINNER_CONFIG_SHA256:
        raise E8RunnerError("DETECTOR_CONFIG_LOCK.json's own recorded winner_config_sha256 disagrees "
                            "with the frozen expected value")
    if track_g.get("variant_identity") != TRACK_G_VARIANT_IDENTITY:
        raise E8RunnerError("DETECTOR_CONFIG_LOCK.json's own recorded variant_identity disagrees "
                            "with the frozen expected value")

    if float(base.backbone_lr) != 1e-05 or float(base.head_lr) != 1e-04:
        raise E8RunnerError(
            f"base m9_reference.yaml LR values (backbone_lr={base.backbone_lr}, head_lr={base.head_lr}) "
            "no longer match the frozen winner LR binding (1e-5/1e-4) -- refusing to silently "
            "re-derive via the LR-anchor multiplier"
        )

    weights = dict(base.loss_weights)
    for name in ("lambda_syn", "lambda_risk"):
        if name in winner_config:
            weights[name] = float(winner_config[name])

    overrides: dict[str, Any] = {
        "run_id": spec.run_id,
        "seed": int(spec.seed),
        "weight_decay": float(winner_config["weight_decay"]),
        "warmup_fraction": float(winner_config["warmup"]),
        "loss_weights": weights,
        "variant": configs["variant"],
        "synthetic_bank_identity": str(synthetic_bank_identity),
    }
    training_config = replace(base, **overrides)

    if training_config.total_epochs != FROZEN_TOTAL_EPOCHS:
        raise E8RunnerError(f"resolved total_epochs {training_config.total_epochs} != frozen "
                            f"{FROZEN_TOTAL_EPOCHS}")
    if training_config.steps_per_epoch != FROZEN_STEPS_PER_EPOCH:
        raise E8RunnerError(f"resolved steps_per_epoch {training_config.steps_per_epoch} != frozen "
                            f"{FROZEN_STEPS_PER_EPOCH}")
    if training_config.warmup_detector_epochs != 3 or training_config.manifold_warmup_epochs != 2 \
            or training_config.mixed_epochs != 30:
        raise E8RunnerError("resolved stage schedule disagrees with the frozen G1=3/G2=2/G5=30 contract")

    return training_config, configs["detector_config"]


# --------------------------------------------------------------------------- #
# Source-package binding (metadata-only preflight; never fabricates presence)
# --------------------------------------------------------------------------- #

def source_package_binding(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root(root)
    package_root = repo / SOURCE_PACKAGE_RELATIVE_PATH
    manifests = {
        name: (package_root / "manifests" / name)
        for name in ("source_train.parquet", "source_dev.parquet")
    }
    return {
        "package_root": SOURCE_PACKAGE_RELATIVE_PATH,
        "package_identity": EXT_F1_SOURCE_PACKAGE_IDENTITY,
        "allowed_domains": sorted(ALLOWED_SOURCE_DOMAINS),
        "package_present_locally": package_root.is_dir(),
        "manifests_present": {name: path.is_file() for name, path in manifests.items()},
    }


# --------------------------------------------------------------------------- #
# E8 bank opening (delegates entirely to the adapter -- no duplicate filter)
# --------------------------------------------------------------------------- #

def resolve_e8_bank_counts(spec: E8RunSpec, root: Path | None = None) -> adapter.E8ArmBinding:
    """Resolve and hard-assert this run's E8 bank counts via the adapter's
    own ``resolve_arm_binding`` -- never re-implements membership filtering
    here."""
    binding = adapter.resolve_arm_binding(spec.arm, root)
    if binding.membership_count != EXPECTED_COUNT_PER_ARM:
        raise E8RunnerError(f"{spec.arm}: bank count {binding.membership_count} != "
                            f"{EXPECTED_COUNT_PER_ARM}")
    if binding.physics_count != EXPECTED_PHYSICS_PER_ARM:
        raise E8RunnerError(f"{spec.arm}: physics count {binding.physics_count} != "
                            f"{EXPECTED_PHYSICS_PER_ARM}")
    if binding.gpat_count != EXPECTED_GPAT_PER_ARM:
        raise E8RunnerError(f"{spec.arm}: gpat count {binding.gpat_count} != {EXPECTED_GPAT_PER_ARM}")
    return binding


# --------------------------------------------------------------------------- #
# Collision / resume policy
# --------------------------------------------------------------------------- #

def classify_run_state(run_root: Path) -> RunState:
    """Fail-closed state classification. Never deletes, never silently
    overwrites, never silently resumes.

    Grounded in verified ``M9Trainer`` behavior (``src/prism_fas/detector/
    trainer.py``): a stage's ``stages/<stage>/output_hashes.json`` is that
    stage's own authoritative completion marker (per
    ``reconcile_lineage_from_disk``'s docstring); the run's final stage is
    G6 (``calibration_stage: G6`` in the frozen stage schedule), so
    ``stages/G6/output_hashes.json`` plus ``run.json`` together mark a
    COMPLETED run. ``FAILED_TECHNICAL`` is recognized only via an explicit
    ``E8_RUN_FAILED.json`` marker -- a contract this runner's own future
    launch wrapper writes on a caught technical failure; ``M9Trainer``
    itself has no such concept.
    """
    if not run_root.is_dir():
        return RunState.NOT_STARTED
    entries = list(run_root.iterdir())
    if not entries:
        return RunState.NOT_STARTED

    if (run_root / FAILURE_MARKER_NAME).is_file():
        return RunState.FAILED_TECHNICAL

    completion_marker = run_root / "stages" / "G6" / "output_hashes.json"
    run_json = run_root / "run.json"
    if completion_marker.is_file() and run_json.is_file():
        return RunState.COMPLETED

    if (run_root / "stages").is_dir() or (run_root / "checkpoints").is_dir() or run_json.is_file():
        return RunState.IN_PROGRESS

    return RunState.BLOCKED_COLLISION


def assert_no_collision(run_root: Path) -> RunState:
    """Fail-closed pre-launch guard. Only NOT_STARTED may proceed to
    training. Every other state raises -- this function never deletes,
    overwrites, or resumes anything itself."""
    state = classify_run_state(run_root)
    if state is not RunState.NOT_STARTED:
        raise E8RunnerError(
            f"{run_root}: run state is {state.value}, not NOT_STARTED -- refusing to launch. "
            "No automatic overwrite, resume, or deletion is ever performed by this runner."
        )
    return state


# --------------------------------------------------------------------------- #
# Preflight (dry-run) mode -- resolves everything, never trains
# --------------------------------------------------------------------------- #

def preflight_e8_run(spec: E8RunSpec, root: Path | None = None) -> dict[str, Any]:
    """Resolve run specification, source package binding, E8 bank, exact
    bank counts, frozen training config, output path, collision state,
    model identity and the target firewall -- then STOP. Never imports
    ``M9Trainer`` and never opens a pixel payload."""
    repo = _repo_root(root)
    source_binding = source_package_binding(repo)
    bank_binding = resolve_e8_bank_counts(spec, repo)
    training_config, detector_config = load_frozen_winner_track_g_config(
        spec, synthetic_bank_identity=bank_binding.c6_bank_lock_sha256, root=repo)
    run_root = repo / spec.run_root
    state = classify_run_state(run_root)

    return {
        "schema_version": "ext-q1q2-e8-training-runner-preflight-result-v1",
        "run_id": spec.run_id, "arm": spec.arm, "condition": spec.condition, "seed": spec.seed,
        "run_root": spec.run_root, "collision_state": state.value,
        "source_package_binding": source_binding,
        "bank_counts": {"total": bank_binding.membership_count, "physics": bank_binding.physics_count,
                        "gpat": bank_binding.gpat_count},
        "frozen_schedule": {
            "total_epochs": training_config.total_epochs,
            "steps_per_epoch": training_config.steps_per_epoch,
            "total_optimizer_updates": training_config.total_epochs * training_config.steps_per_epoch,
            "synthetic_draws_per_run": FROZEN_SYNTHETIC_DRAWS_PER_RUN,
        },
        "training_config_hash": training_config.hash(),
        "model_identity": "google/siglip2-base-patch16-224",
        "target_firewall": {"target_access": False, "target_labels_accessed": False,
                            "allowed_domains": sorted(ALLOWED_SOURCE_DOMAINS)},
        "training_started": False,
        "checkpoint_created": False,
    }


# --------------------------------------------------------------------------- #
# Scientific launch (implemented, never invoked by this task)
# --------------------------------------------------------------------------- #

def launch_scientific_run(spec: E8RunSpec, *, root: Path | None = None,
                          candidates_root: Path | None = None,
                          recipes: Any = None, recipe_bank_identity: str | None = None,
                          _trainer_cls: Callable[..., Any] | None = None) -> Any:
    """Construct and run the EXISTING ``M9Trainer`` against the E8 bank.

    ``_trainer_cls`` is a test-only constructor-injection seam (defaults to
    the real ``prism_fas.detector.trainer.M9Trainer``); it exists so tests
    can verify the exact construction kwargs without instantiating the full
    SigLIP2 model. This task does not call this function for real.
    """
    repo = _repo_root(root)
    run_root = repo / spec.run_root
    assert_no_collision(run_root)

    e8_bank = adapter.open_e8_arm_bank(
        spec.arm, candidates_root=candidates_root, recipes=recipes or (),
        package_identity=EXT_F1_SOURCE_PACKAGE_IDENTITY,
        recipe_bank_identity=recipe_bank_identity or "", root=repo,
    )
    training_config, detector_config = load_frozen_winner_track_g_config(
        spec, synthetic_bank_identity=e8_bank.identity, root=repo)

    if _trainer_cls is None:
        from prism_fas.detector.trainer import M9Trainer as _trainer_cls  # noqa: N806

    trainer = _trainer_cls(
        config=training_config, detector_config=detector_config,
        package_root=repo / SOURCE_PACKAGE_RELATIVE_PATH,
        bank_root=repo / SOURCE_PACKAGE_RELATIVE_PATH,  # unused: synthetic_bank= is supplied
        recipe_bank_root=repo, run_root=run_root, cache_root=run_root / "cache",
        weight_root=repo / "weights", loader_config_path=repo / LOADER_CONFIG_RELATIVE_PATH,
        synthetic_bank=e8_bank,
    )
    return trainer


# --------------------------------------------------------------------------- #
# Smoke mode -- distinct namespace, distinct ID, never one of the 15
# --------------------------------------------------------------------------- #

def smoke_run_root(root: Path | None = None) -> Path:
    return _repo_root(root) / SMOKE_ROOT_RELATIVE / SMOKE_RUN_ID


def preflight_e8_smoke(arm: str, seed: int, root: Path | None = None) -> dict[str, Any]:
    """Preflight for the ONE engineering smoke. Uses a distinct namespace
    and a distinct run identity that can never collide with the 15
    scientific run IDs."""
    if arm not in ARMS:
        raise E8RunnerError(f"unknown arm {arm!r}")
    repo = _repo_root(root)
    run_root = smoke_run_root(repo)
    if run_root == (repo / RUN_ROOT_RELATIVE / derive_run_id(arm, seed)):
        raise E8RunnerError("smoke output must never collide with a scientific run root")
    return {
        "schema_version": "ext-q1q2-e8-smoke-preflight-result-v1",
        "smoke_run_id": SMOKE_RUN_ID, "smoke_run_root": f"{SMOKE_ROOT_RELATIVE}/{SMOKE_RUN_ID}",
        "arm": arm, "seed": seed,
        "is_scientific_result": False,
        "mechanism": "M9Trainer.smoke(steps=..., resume_steps=..., stage=...) -- an existing, "
                    "already-frozen reduced-step engineering mechanism (src/prism_fas/detector/"
                    "trainer.py); it overrides only engineering execution length, never batch "
                    "composition, LR, weight decay, or any other frozen scientific config value",
        "status": "SUPPORTED_VIA_M9TRAINER_SMOKE_METHOD",
        "target_access": False,
    }


# --------------------------------------------------------------------------- #
# Runner rule identity (frozen before any scientific execution)
# --------------------------------------------------------------------------- #

def build_runner_rule_payload() -> dict[str, Any]:
    return {
        "runner_rule_name": RUNNER_RULE_NAME,
        "execution_plan_identity": EXPECTED_EXECUTION_PLAN_IDENTITY,
        "adapter_implementation_commit": ADAPTER_IMPLEMENTATION_COMMIT,
        "adapter_rule_identity": EXPECTED_ADAPTER_RULE_IDENTITY,
        "adapter_source_sha256": "7fa4be3beec6be6118faa830b39ef1d579a4715aba11c84a08a7a813a4ebd520",
        "membership_sha256": FROZEN_MEMBERSHIP_PARQUET_SHA256,
        "membership_lock_sha256": FROZEN_MEMBERSHIP_LOCK_SHA256,
        "selector_rule_identity": SELECTOR_RULE_IDENTITY,
        "input_binding_rule_identity": INPUT_BINDING_RULE_IDENTITY,
        "track_g_variant_identity": TRACK_G_VARIANT_IDENTITY,
        "c7_winner_config_sha256": C7_WINNER_CONFIG_SHA256,
        "source_fold_identity": EXT_F1_SOURCE_PACKAGE_IDENTITY,
        "allowed_run_ids": sorted(derive_run_id(arm, seed) for arm in ARMS for seed in SEEDS),
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "run_root": RUN_ROOT_RELATIVE,
        "collision_policy": [state.value for state in RunState],
        "target_firewall": {"target_access": False, "target_labels_accessed": False,
                            "allowed_domains": sorted(ALLOWED_SOURCE_DOMAINS)},
    }


def runner_rule_identity() -> str:
    return cc.sha256_bytes(cc.canonical_json_bytes(build_runner_rule_payload()))
