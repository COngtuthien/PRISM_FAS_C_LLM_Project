"""PRISM-FAS-C EXT-Q1Q2 - E5: REAL-ONLY / NO-SYN step-matched baseline.

E4 (frozen, closed) established that threshold transfer is a partial confound
of the LLM ranking but not a sufficient explanation of it. E5 asks a
different, orthogonal question: does synthetic spoof supervision provide
value beyond simply giving the training procedure more optimization
steps/samples? Scope for this milestone is EXT-F1 / Track G only (the fuller
three-fold rollout belongs to E7, whose own conditions.yaml already declares
this same condition as ``G-REALONLY``).

The step-matching audit (``E5_STEP_MATCHING_AUDIT.json``, written by a prior
phase of this milestone) established, by reading the existing frozen code and
the real frozen C7 lock -- not by guessing -- that:

* ``synthetic: "none"`` is an ALREADY existing, ALREADY validated flag value
  (``detector.variant.ResolvedExperimentVariant``), used historically by six
  Version-B ablation baselines.
* Whenever a variant declares ``synthetic: "none"``,
  ``detector.trainer.batch_contract_for`` already returns the SAME real-only
  16/16/0 batch contract for every stage -- the same shape Track G's own G1
  warm-up already uses -- with zero new sampler code.
* ``M9TrainingConfig.total_epochs`` and ``.steps_per_epoch`` are schedule
  constants, untouched by ``pipeline.adapters.c8._detector_config_for_row``'s
  per-row overrides and unaffected by variant/batch composition, so
  ``M9Trainer.total_steps = total_epochs * steps_per_epoch`` is identical
  between a historical mixed Track G row and its REAL_ONLY counterpart by
  construction.
* ``pipeline.adapters.c8.run_source_only_flow`` (via ``M9Trainer.resume``
  and ``trainer.lineage``) already gives safe, idempotent resume with no
  overwrite of a completed run.

This module therefore does not invent a new training mechanism. It builds the
one new input the existing mechanism needs -- a Track-G variant with synthetic
supervision structurally removed -- and wires it through the SAME
``M9Trainer`` / ``run_source_only_flow`` orchestration ``c8._run_scientific_row``
already uses, under the additive ``e5_realonly`` extension namespace.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from prism_fas.evaluation import c_ext_common as cc

# --------------------------------------------------------------------------- #
# Identity constants
# --------------------------------------------------------------------------- #

EXPERIMENT = "E5_REAL_ONLY_STEP_MATCHED"
FOLD = "EXT-F1"
TRACK = "G"
ARM = "REAL_ONLY"
SOURCE_DATASETS = "CASIA-FASD + MSU-MFSD"
TARGET_DATASET = "SiW-Mv2"
SEEDS: tuple[int, ...] = (20260806, 20260807, 20260808, 20260809, 20260810)

HISTORICAL_TOTAL_EPOCHS = 35
HISTORICAL_STEPS_PER_EPOCH = 45
HISTORICAL_ACCUMULATION_STEPS = 1
HISTORICAL_TRAINING_BUDGET = HISTORICAL_TOTAL_EPOCHS * HISTORICAL_STEPS_PER_EPOCH

#: Track G's real, current source package identity (matches every historical
#: C-G-{RND,DET,LLM} row). E5 trains against this SAME package.
TRACK_G_SOURCE_PACKAGE_IDENTITY = "08d9d289eb4b462006afcff37cd4750a7c4eeb402c83de5599eda38df44168c9"
#: The M9 reference (M8 v3) bank's package identity -- an OLDER package build,
#: incompatible with Track G's current source package. Never opened by E5.
STALE_M8_BANK_PACKAGE_IDENTITY = "b1cf29b69a165ed5d9e074fc8127c17fbf057723edf9e272048ec3a564eb9dc6"

#: The real, frozen Track G architecture flags this audit read from
#: ``reports/full/c7/DETECTOR_CONFIG_LOCK.json``. A row-construction function
#: below refuses to proceed if the CURRENT lock's flags ever disagree with
#: this snapshot, rather than silently training against a changed contract.
EXPECTED_TRACK_G_VARIANT_FLAGS: dict[str, Any] = {
    "frames_per_video": 4,
    "fusion": "single_logit",
    "global_branch": "siglip2_frozen",
    "local_branch": "off",
    "manifold": "off",
    "outlier_loss": "off",
    "prompt": "off",
    "prototype_k": 0,
    "quality_weighting": "q_weighted",
    "recipe_conditioning": "structured",
    "region": "off",
    "sampler": "domain_class_balanced",
    "synthetic": "bank_physics_gpat",
}

#: The M9 "reference" bank (`prism_synthetic_bank_m8_v3_e84c78cd2a9b`) is bound
#: to an OLDER source package identity than the current Track G source
#: package (see `E5_BANK_DEPENDENCY_REAUDIT.json`) and must never be opened by
#: E5. `variant.synthetic == "none"` makes `M9TrainingDataset` build its own
#: empty, package-matched bank internally instead -- E5 opens no bank at all.

C7_LOCK_PATH = "reports/full/c7/DETECTOR_CONFIG_LOCK.json"

E5_DIR = "reports/c_ext_q1q2_v1/e5_realonly"
E5_RUNS_DIR = f"{E5_DIR}/runs"
E5_LOCK_PATH = f"{E5_DIR}/E5_REAL_ONLY_LOCK.json"
E5_STEP_MATCHING_AUDIT_PATH = f"{E5_DIR}/E5_STEP_MATCHING_AUDIT.json"
E5_RUN_ROOT = "runs/c_ext_q1q2_v1/EXT-F1/e5_realonly"


class E5Error(RuntimeError):
    """A precondition for building or running the E5 REAL_ONLY row failed."""


# --------------------------------------------------------------------------- #
# Track G lock verification (read-only; never mutates the frozen C7 lock)
# --------------------------------------------------------------------------- #

def load_c7_lock(repo: Path) -> dict[str, Any]:
    path = repo / C7_LOCK_PATH
    if not path.is_file():
        raise E5Error(f"missing frozen C7 detector config lock at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_track_g_lock(lock: dict[str, Any]) -> dict[str, Any]:
    """The frozen Track G sub-configuration, with its flags pinned against the audit.

    A silent drift here (someone re-running C7's search, or editing the lock)
    would make E5 train against an architecture the step-matching audit never
    examined -- so this fails closed rather than trusting the current lock.
    """
    tracks = dict(lock.get("tracks") or {})
    if "G" not in tracks:
        raise E5Error(
            f"the C7 detector config lock names no Track G configuration; "
            f"it carries {sorted(tracks)}"
        )
    g = dict(tracks["G"])
    actual_flags = dict(g.get("variant_flags") or {})
    if actual_flags != EXPECTED_TRACK_G_VARIANT_FLAGS:
        raise E5Error(
            "Track G variant_flags in the C7 lock disagree with the flags the E5 "
            f"step-matching audit examined.\nexpected: {EXPECTED_TRACK_G_VARIANT_FLAGS}\n"
            f"found:    {actual_flags}\n"
            "Re-run the step-matching audit against the current lock before proceeding."
        )
    return g


# --------------------------------------------------------------------------- #
# The REAL_ONLY variant
# --------------------------------------------------------------------------- #

def e5_variant_flags(track_g_flags: dict[str, Any]) -> dict[str, Any]:
    """Track G's own flags with synthetic supervision structurally removed.

    Only ``synthetic`` is a deliberate choice here. ``recipe_conditioning``
    and ``quality_weighting`` are forced to ``"off"`` because
    ``ResolvedExperimentVariant.validate()`` REQUIRES this whenever
    ``synthetic == "none"`` (there is nothing left to condition on or weight)
    -- it is a mechanical consequence of removing synthetic data, not an
    independent ablation choice, and it is a no-op on the loss graph for
    Track G either way (``L_cls_syn`` already becomes structurally inactive
    once ``synthetic == "none"``, regardless of these two flags). Every other
    flag -- region, manifold, fusion, local/global branch, prompt,
    outlier_loss, sampler, frames_per_video -- is untouched, so the only
    experimental factor between E5 and the historical Track G arms is
    synthetic supervision present vs absent.
    """
    flags = {key: value for key, value in track_g_flags.items() if key != "recipe_arm"}
    flags["synthetic"] = "none"
    flags["recipe_conditioning"] = "off"
    flags["quality_weighting"] = "off"
    return flags


def build_e5_variant(track_g_flags: dict[str, Any]) -> Any:
    from prism_fas.detector.variant import ResolvedExperimentVariant

    return ResolvedExperimentVariant.resolve(e5_variant_flags(track_g_flags))


# --------------------------------------------------------------------------- #
# Config construction (mirrors pipeline.adapters.c8._detector_config_for_row)
# --------------------------------------------------------------------------- #

def build_e5_training_config(
    *, repo: Path, lock: dict[str, Any], seed: int, run_id: str,
) -> tuple[Any, dict[str, Any]]:
    """The frozen Track G schedule/optimizer scalars, at the REAL_ONLY variant and this seed.

    Deliberately mirrors ``pipeline.adapters.c8._detector_config_for_row``:
    every scalar C7 froze for Track G (the LR multiplier x anchor, weight
    decay, warm-up fraction, loss weights) is read from the SAME winner
    config and reused unchanged. No schedule field (``steps_per_epoch``,
    ``warmup_detector_epochs``, ``manifold_warmup_epochs``, ``mixed_epochs``,
    ``accumulation_steps``) is ever touched by this function or by the one it
    mirrors -- that is what step-matching means here. The only overrides
    novel to E5 are the variant (synthetic supervision removed) and
    ``synthetic_bank_identity`` (left empty: `variant.synthetic == "none"`
    means `M9TrainingDataset` opens no bank at all -- see
    `detector.dataset._empty_synthetic_bank` -- so no bank identity is ever
    pinned or checked for this row).
    """
    from prism_fas.detector.config import load_m9_configs
    from prism_fas.pipeline.adapters.c7 import _TRIAL_LOSS_WEIGHTS

    g = verify_track_g_lock(lock)
    variant = build_e5_variant(dict(g["variant_flags"]))
    configs = load_m9_configs(
        repo / "configs/models/m9_detector.yaml",
        repo / "configs/train/m9_reference.yaml",
        variant=variant,
    )
    reference_config = configs["training_config"]
    winner = dict(g.get("winner_config") or {})

    weights = dict(reference_config.loss_weights)
    for name in _TRIAL_LOSS_WEIGHTS:
        if name in winner:
            weights[name] = float(winner[name])

    overrides: dict[str, Any] = {
        "run_id": run_id,
        "seed": int(seed),
        "prototype_seed": int(seed),
        "variant": variant,
        "loss_weights": weights,
        "synthetic_bank_identity": "",
        "source_domains": reference_config.source_domains,
    }
    multiplier = winner.get("learning_rate_multiplier")
    anchor = dict(g.get("lr_anchor_vector") or {})
    if multiplier is not None and anchor:
        for group, value in anchor.items():
            overrides[group] = float(value) * float(multiplier)
    if "weight_decay" in winner:
        overrides["weight_decay"] = float(winner["weight_decay"])
    if "warmup" in winner:
        overrides["warmup_fraction"] = float(winner["warmup"])

    config = replace(reference_config, **overrides)
    assert_step_matched(config)
    return config, configs


# --------------------------------------------------------------------------- #
# Step-matching / real-only invariants (asserted, never assumed)
# --------------------------------------------------------------------------- #

def assert_step_matched(config: Any) -> None:
    """Every schedule field the historical Track G rows used, unchanged."""
    if int(config.total_epochs) != HISTORICAL_TOTAL_EPOCHS:
        raise E5Error(
            f"E5 config total_epochs={config.total_epochs} != historical "
            f"{HISTORICAL_TOTAL_EPOCHS}; step matching is broken"
        )
    if int(config.steps_per_epoch) != HISTORICAL_STEPS_PER_EPOCH:
        raise E5Error(
            f"E5 config steps_per_epoch={config.steps_per_epoch} != historical "
            f"{HISTORICAL_STEPS_PER_EPOCH}; step matching is broken"
        )
    if int(config.accumulation_steps) != HISTORICAL_ACCUMULATION_STEPS:
        raise E5Error(
            f"E5 config accumulation_steps={config.accumulation_steps} != historical "
            f"{HISTORICAL_ACCUMULATION_STEPS}; step matching is broken"
        )
    actual_budget = int(config.total_epochs) * int(config.steps_per_epoch)
    if actual_budget != HISTORICAL_TRAINING_BUDGET:
        raise E5Error(
            f"E5 expected_optimizer_steps={actual_budget} != historical "
            f"{HISTORICAL_TRAINING_BUDGET}"
        )


def assert_real_only(config: Any) -> None:
    """Zero synthetic supervision anywhere in the schedule, for every stage the variant runs."""
    from prism_fas.detector.trainer import batch_contract_for

    if config.variant.uses_synthetic:
        raise E5Error("E5 variant declares synthetic supervision; expected synthetic='none'")
    if config.variant.recipe_conditioning != "off":
        raise E5Error("E5 variant must resolve recipe_conditioning='off'")
    if config.variant.quality_weighting != "off":
        raise E5Error("E5 variant must resolve quality_weighting='off'")
    for stage in config.variant.required_stages():
        contract = batch_contract_for(stage, config)
        if contract.synthetic != 0:
            raise E5Error(f"stage {stage!r} batch contract still requests synthetic samples")
        if contract.phase != "real_only":
            raise E5Error(f"stage {stage!r} batch contract phase={contract.phase!r} != 'real_only'")
        if contract.real_live != 16 or contract.real_spoof != 16:
            raise E5Error(
                f"stage {stage!r} real-only composition {contract.real_live}/"
                f"{contract.real_spoof} != the expected 16/16"
            )


def expected_optimizer_steps(config: Any) -> int:
    return int(config.total_epochs) * int(config.steps_per_epoch)


# --------------------------------------------------------------------------- #
# Per-seed identity payload
# --------------------------------------------------------------------------- #

def seed_run_id(seed: int) -> str:
    return f"EXT-F1-G-REALONLY-s{seed}"


def build_identity_record(*, seed: int, config: Any, actual_optimizer_steps: int | None) -> dict[str, Any]:
    expected = expected_optimizer_steps(config)
    return {
        "schema_version": "e5-real-only-identity-v1",
        "experiment": EXPERIMENT,
        "fold": FOLD,
        "track": TRACK,
        "arm": ARM,
        "detector_seed": int(seed),
        "run_id": seed_run_id(seed),
        "synthetic_samples_used": 0,
        "synthetic_recipes_used": 0,
        "llm_api_calls": 0,
        "step_matched": True,
        "historical_reference_training_budget": HISTORICAL_TRAINING_BUDGET,
        "expected_optimizer_steps": expected,
        "actual_optimizer_steps": actual_optimizer_steps,
        "source_datasets": SOURCE_DATASETS,
        "target_dataset": TARGET_DATASET,
        "target_used_for_training": False,
    }


# --------------------------------------------------------------------------- #
# Resumable per-seed runner (real GPU training; not executed this milestone)
# --------------------------------------------------------------------------- #

def run_root_for_seed(repo: Path, seed: int) -> Path:
    return repo / E5_RUN_ROOT / f"s{seed}"


def seed_result_path(repo: Path, seed: int) -> Path:
    return run_root_for_seed(repo, seed) / "E5_REAL_ONLY_RESULT.json"


def seed_already_complete(repo: Path, seed: int) -> bool:
    """A completed seed carries a result file with every declared stage PASS-equivalent.

    Read-only check; never deletes or recomputes a completed result.
    """
    path = seed_result_path(repo, seed)
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return bool(payload.get("status") == "COMPLETE")


#: `M9Trainer.bank_root` is a required dataclass field (converted to a `Path`
#: unconditionally in `__post_init__`), but `M9TrainingDataset` never reads it
#: for a `synthetic: none` variant (see `detector.dataset._empty_synthetic_bank`
#: and `E5_BANK_DEPENDENCY_REAUDIT.json`). This sentinel documents that at the
#: call site rather than asking the caller to supply a real bank directory that
#: will never be opened.
UNUSED_BANK_ROOT_SENTINEL = Path("<synthetic-none-unused-bank-root>")


def run_e5_seed(
    *,
    repo: Path,
    lock: dict[str, Any],
    seed: int,
    package_root: Path,
    recipe_bank_root: Path,
    weight_root: Path,
    loader_config_path: Path,
    device: str = "cuda",
    resume: bool = True,
    trainer_factory: Callable[..., Any] | None = None,
    flow_runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One REAL_ONLY seed, trained with the SAME M9Trainer / run_source_only_flow C8 uses.

    No M8/C6 synthetic bank is opened or required: `variant.synthetic == "none"`
    makes `M9TrainingDataset` build its own empty, package-matched bank object
    internally, so this call site never needs a `bank_root`, a bank identity or
    a bank content pin at all -- there is no bank whose package identity could
    ever disagree with the source package this row actually trains on.

    ``trainer_factory`` / ``flow_runner`` are injectable seams for CPU/fixture
    testing (E5-C); the real (E5-D) call site leaves them ``None`` and gets
    the genuine ``detector.trainer.M9Trainer`` / ``run_source_only_flow``.

    Safe-resume / no-overwrite: a seed already marked COMPLETE is returned
    verbatim without touching the trainer, exactly mirroring
    ``run_source_only_flow``'s own "only outstanding stages execute" contract
    one layer up, for a seed-granularity skip that never needs to instantiate
    a model to discover a run is done.
    """
    if device != "cuda" and trainer_factory is None:
        raise E5Error(
            "a real (non-fixture) E5 run requires device='cuda'; GPU is required for "
            "real training and target-inaccessible-during-training is unaffected either way"
        )

    destination = run_root_for_seed(repo, seed)
    if seed_already_complete(repo, seed):
        return json.loads(seed_result_path(repo, seed).read_text(encoding="utf-8"))

    destination.mkdir(parents=True, exist_ok=True)
    run_id = seed_run_id(seed)
    config, configs = build_e5_training_config(repo=repo, lock=lock, seed=seed, run_id=run_id)
    assert_real_only(config)

    if trainer_factory is None:
        from prism_fas.detector.trainer import M9Trainer

        trainer_factory = M9Trainer
    if flow_runner is None:
        from prism_fas.detector.trainer import run_source_only_flow

        flow_runner = run_source_only_flow

    trainer = trainer_factory(
        config=config,
        detector_config=configs["detector_config"],
        package_root=package_root,
        bank_root=UNUSED_BANK_ROOT_SENTINEL,
        recipe_bank_root=recipe_bank_root,
        run_root=destination,
        cache_root=destination / "cache",
        weight_root=weight_root,
        loader_config_path=loader_config_path,
        device=device,
    )
    flow = flow_runner(trainer, resume=resume)

    actual_steps = int((flow.get("run_summary") or {}).get("global_step", 0))
    identity = build_identity_record(seed=seed, config=config, actual_optimizer_steps=actual_steps)
    if actual_steps != identity["expected_optimizer_steps"]:
        status = "INCOMPLETE"
    else:
        status = "COMPLETE"

    result = {
        "schema_version": "e5-real-only-result-v1",
        "status": status,
        "identity": identity,
        "flow": flow,
        "path": destination.relative_to(repo).as_posix(),
    }
    cc.write_json_atomic(seed_result_path(repo, seed), result, root=repo)
    return result


def run_all_seeds(*, repo: Path, seeds: tuple[int, ...] = SEEDS, **kwargs: Any) -> list[dict[str, Any]]:
    return [run_e5_seed(repo=repo, seed=seed, **kwargs) for seed in seeds]


# --------------------------------------------------------------------------- #
# E5-C: CPU dry-run / fixture validation (config construction only, no torch)
# --------------------------------------------------------------------------- #

def dry_run_config_audit(repo: Path) -> dict[str, Any]:
    """Builds and validates every seed's E5 config against the real frozen C7 lock.

    Pure config construction and assertion -- no dataset, no model, no torch,
    no GPU. This is what E5-C actually checks: that the REAL_ONLY variant and
    schedule resolve, for every seed, to a step-matched, synthetic-free
    configuration, using the real frozen lock rather than a fixture.
    """
    lock = load_c7_lock(repo)
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        run_id = seed_run_id(seed)
        config, _ = build_e5_training_config(repo=repo, lock=lock, seed=seed, run_id=run_id)
        assert_real_only(config)
        rows.append(build_identity_record(seed=seed, config=config, actual_optimizer_steps=None))
    return {
        "schema_version": "e5-dry-run-audit-v1",
        "execution_profile": "smoke",
        "scientific_eligible": False,
        "rows": rows,
        "historical_training_budget": HISTORICAL_TRAINING_BUDGET,
        "all_step_matched": all(row["step_matched"] for row in rows),
        "all_synthetic_free": all(row["synthetic_samples_used"] == 0 for row in rows),
    }


def write_dry_run_audit(repo: Path) -> str:
    payload = dry_run_config_audit(repo)
    return cc.write_json_atomic(f"{E5_DIR}/E5_DRY_RUN_CONFIG_AUDIT.json", payload, root=repo)


# --------------------------------------------------------------------------- #
# GPU command preparation (never executed here; no SSH, no subprocess)
# --------------------------------------------------------------------------- #

def gpu_training_command(seed: int) -> str:
    """The exact command to run manually on the GPU host. Never executed by this process.

    Invokes THIS module's own ``main()`` real-mode entry point (see below), not
    a separate script -- there is one runner, not two copies of the same
    orchestration that could quietly disagree. Carries no ``--bank-root``: a
    ``synthetic: none`` variant opens no bank at all (see
    ``E5_BANK_DEPENDENCY_REAUDIT.json`` / ``detector.dataset._empty_synthetic_bank``),
    so there is no bank directory for this command to name.
    """
    return (
        "python -m prism_fas.evaluation.c_ext_e5_real_only "
        f"--execute --authorize-gpu-training --seed {seed} "
        "--package-root /vol/data/packages/prism_data_v1_m3b "
        "--recipe-bank-root /vol/data/recipe_banks/prism_recipe_bank_m7_v1 "
        "--weight-root /vol/models/pretrained/m9 "
        "--loader-config configs/data/loader_m4.yaml "
        "--device cuda --resume"
    )


def files_to_sync_to_gpu() -> list[str]:
    return [
        "src/prism_fas/",
        "configs/models/m9_detector.yaml",
        "configs/train/m9_reference.yaml",
        "configs/data/loader_m4.yaml",
        "configs/cloud/modal_m9.yaml",
        C7_LOCK_PATH,
        E5_LOCK_PATH,
    ]


def build_e5_lock(repo: Path) -> dict[str, Any]:
    """The frozen E5 plan: variant identity, expected step budget, GPU command per seed.

    This locks the DESIGN (E5-A/B/C), not an executed training result -- no
    GPU run backs it yet. E5-D will read this lock and refuse to proceed if
    its own recomputed config disagrees with what is frozen here.

    FROZEN as of the first E5-C dry run: ``write_e5_lock`` / ``E5_LOCK_PATH``
    are not called again after that point (see ``write_e5_runtime_correction_lock``
    below for how a later technical fix is recorded instead), so this function
    is kept for its historical shape and for tests, not re-invoked in `main()`.
    """
    lock = load_c7_lock(repo)
    g = verify_track_g_lock(lock)
    variant_flags = e5_variant_flags(dict(g["variant_flags"]))
    seeds_payload = []
    for seed in SEEDS:
        config, _ = build_e5_training_config(repo=repo, lock=lock, seed=seed, run_id=seed_run_id(seed))
        seeds_payload.append({
            "seed": seed,
            "run_id": seed_run_id(seed),
            "expected_optimizer_steps": expected_optimizer_steps(config),
            "gpu_command": gpu_training_command(seed),
        })
    payload = {
        "schema_version": "e5-real-only-lock-v1",
        "experiment": EXPERIMENT,
        "fold": FOLD,
        "track": TRACK,
        "arm": ARM,
        "variant_flags": variant_flags,
        "historical_training_budget": HISTORICAL_TRAINING_BUDGET,
        "seeds": seeds_payload,
        "files_to_sync_to_gpu": [f for f in files_to_sync_to_gpu() if f != E5_LOCK_PATH],
        "gpu_real_run_executed": False,
    }
    payload["lock_identity"] = cc.sha256_bytes(cc.canonical_json_bytes(payload))
    return payload


def write_e5_lock(repo: Path) -> str:
    payload = build_e5_lock(repo)
    return cc.write_json_atomic(E5_LOCK_PATH, payload, root=repo)


# --------------------------------------------------------------------------- #
# Additive technical-correction lock (the bank-dependency fix, TASK D)
# --------------------------------------------------------------------------- #

E5_RUNTIME_CORRECTION_LOCK_PATH = f"{E5_DIR}/E5_REAL_ONLY_RUNTIME_CORRECTION_LOCK.json"


def build_e5_runtime_correction_lock(repo: Path) -> dict[str, Any]:
    """Documents the bank-dependency bugfix WITHOUT rewriting the original design lock.

    The original ``E5_REAL_ONLY_LOCK.json`` (frozen at the end of E5-C) carried
    a GPU command with a mandatory ``--bank-root`` that a real GPU run showed
    was invalid: `M9TrainingDataset` unconditionally required a bank whose
    package identity matched the source package, and the only bank this call
    site could reach by default (the M9 reference / M8 v3 bank) is bound to a
    DIFFERENT, older package identity than Track G's actual source package.
    The fix removes the bank requirement entirely for `synthetic: none`
    (`detector.dataset._empty_synthetic_bank`); this lock records that as an
    additive correction, references the untouched original lock by its
    identity, and carries the corrected GPU command. It never overwrites
    ``E5_REAL_ONLY_LOCK.json``.
    """
    original_path = repo / E5_LOCK_PATH
    if not original_path.is_file():
        raise E5Error(f"missing original E5 design lock at {original_path}; run --dry-run first")
    original = json.loads(original_path.read_text(encoding="utf-8"))

    lock = load_c7_lock(repo)
    seeds_payload = []
    for seed in SEEDS:
        config, _ = build_e5_training_config(repo=repo, lock=lock, seed=seed, run_id=seed_run_id(seed))
        seeds_payload.append({
            "seed": seed,
            "run_id": seed_run_id(seed),
            "expected_optimizer_steps": expected_optimizer_steps(config),
            "corrected_gpu_command": gpu_training_command(seed),
        })

    payload = {
        "schema_version": "e5-real-only-runtime-correction-lock-v1",
        "experiment": EXPERIMENT,
        "original_lock_path": E5_LOCK_PATH,
        "original_lock_identity": original.get("lock_identity"),
        "original_lock_untouched": True,
        "original_design_issue":
            "incompatible mandatory M8 bank on synthetic:none path -- "
            "M9TrainingDataset.__init__ unconditionally opened/checked a "
            "synthetic bank even when the variant declares synthetic='none' "
            "and draws zero synthetic samples; the M9 reference (M8 v3) bank "
            "this call site fell back to is bound to source package "
            f"{STALE_M8_BANK_PACKAGE_IDENTITY}, not the current Track G "
            f"source package {TRACK_G_SOURCE_PACKAGE_IDENTITY}",
        "fix_summary":
            "M9TrainingDataset now builds an empty, package-matched bank "
            "object internally whenever variant.synthetic == 'none' "
            "(detector.dataset._empty_synthetic_bank), ignoring any bank / "
            "bank_root / bank_identity / bank_id argument for that variant "
            "only. Every other variant's bank-opening code and the identity "
            "check itself are unchanged.",
        "bank_dependency_reaudit_path":
            f"{E5_DIR}/E5_BANK_DEPENDENCY_REAUDIT.json",
        "scientific_design_unchanged": True,
        "source_package_unchanged": True,
        "source_package_identity": TRACK_G_SOURCE_PACKAGE_IDENTITY,
        "training_budget_unchanged": True,
        "training_budget": HISTORICAL_TRAINING_BUDGET,
        "synthetic_supervision_unchanged": "ZERO",
        "historical_synthetic_path_changed": False,
        "real_only_batch_contract_changed": False,
        "bank_required_for_synthetic_none": False,
        "attack_cache_required_for_synthetic_none": False,
        "seeds": seeds_payload,
        "files_to_sync_to_gpu": [f for f in files_to_sync_to_gpu() if f != E5_LOCK_PATH],
        "gpu_real_run_executed": False,
    }
    payload["lock_identity"] = cc.sha256_bytes(cc.canonical_json_bytes(payload))
    return payload


def write_e5_runtime_correction_lock(repo: Path) -> str:
    payload = build_e5_runtime_correction_lock(repo)
    return cc.write_json_atomic(E5_RUNTIME_CORRECTION_LOCK_PATH, payload, root=repo)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E5 REAL_ONLY / NO-SYN step-matched baseline")
    parser.add_argument("--dry-run", action="store_true", help="CPU config-only audit (E5-A/E5-C)")
    parser.add_argument("--execute", action="store_true",
                        help="E5-D: real GPU training for one seed. Requires --authorize-gpu-training.")
    parser.add_argument("--authorize-gpu-training", action="store_true",
                        help="Explicit second flag required alongside --execute (never set by CI/tests).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--package-root", type=Path, default=None)
    parser.add_argument("--recipe-bank-root", type=Path, default=None)
    parser.add_argument("--weight-root", type=Path, default=None)
    parser.add_argument("--loader-config", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args(argv)
    repo = cc.repo_root()

    if args.dry_run:
        rel_audit = write_dry_run_audit(repo)
        # NOTE: E5_REAL_ONLY_LOCK.json (write_e5_lock) is the frozen E5-C design
        # lock and is NOT rewritten here or anywhere else after its first write
        # -- a technical correction is recorded additively instead. See
        # write_e5_runtime_correction_lock.
        rel_correction = write_e5_runtime_correction_lock(repo)
        print(f"wrote {rel_audit}")
        print(f"wrote {rel_correction}")
        return 0

    if args.execute:
        if not args.authorize_gpu_training:
            print("--execute requires --authorize-gpu-training as an explicit second flag.")
            return 2
        if args.seed is None or None in (args.package_root, args.recipe_bank_root,
                                         args.weight_root, args.loader_config):
            print("--execute requires --seed, --package-root, --recipe-bank-root, "
                 "--weight-root and --loader-config. No --bank-root is accepted: a "
                 "synthetic:none row opens no bank.")
            return 2
        lock = load_c7_lock(repo)
        result = run_e5_seed(
            repo=repo, lock=lock, seed=args.seed,
            package_root=args.package_root,
            recipe_bank_root=args.recipe_bank_root, weight_root=args.weight_root,
            loader_config_path=args.loader_config, device=args.device, resume=args.resume)
        print(json.dumps({"status": result["status"], "path": result["path"]}))
        return 0 if result["status"] == "COMPLETE" else 1

    print("E5-D (real GPU training) is not executed unless --execute --authorize-gpu-training is passed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
