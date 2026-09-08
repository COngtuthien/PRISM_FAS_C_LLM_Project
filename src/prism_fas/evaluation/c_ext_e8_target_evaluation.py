"""PRISM-FAS-C EXT-Q1Q2 -- E8 G7: label-free target prediction for the 15
completed E8 q-matched Track-G runs (RND/DET/LLM x 5 seeds, EXT-F1 only).

Does NOT modify generic M10 ``target_prediction.py`` semantics. Every generic
building block is REUSED VERBATIM rather than reinvented:

* ``evaluation.target_prediction``: ``TargetInferenceBatch``,
  ``VariantCapabilities``, ``target_batches``, ``predict_target``,
  ``validate_predictions``, ``write_predictions``, ``inference_config_hash``,
  ``build_prediction_lock``, ``write_prediction_lock``.
* ``evaluation.post_failure_exploratory_target_v3.verify_locked_target_feature_package``
  / ``build_verified_target_loader_config`` -- fully generic, protocol-independent
  target feature package verification and loader-policy rebinding.
* ``evaluation.firewall.FirewallConfig`` / ``TargetLabelFirewall``.
* The plan -> stage -> crash-safe promotion transaction -> lockset shape
  ``c_ext_e5_target_inference.py`` already established for a flat run set;
  mirrored here for E8's 15 (arm, seed) runs, with E8's OWN row resolution
  (the calibration authority lock, never a C8 matrix row).

This module NEVER resolves a target label: ``TargetInferenceBatch`` has no
field a label could live in, ``target_label_root`` is declared to the
firewall only so it can be DENIED to G7, and ``unknown_threshold`` is always
``None`` (no reject policy is fitted or applied here).

Checkpoint + calibration provenance for every run come EXCLUSIVELY from the
frozen, validated ``E8_CALIBRATION_AUTHORITY_LOCK.json``
(``c_ext_e8_calibration_authority.require_valid_calibration_authority_for_g7``)
-- this module never re-derives which checkpoint or calibration to use.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prism_fas.evaluation import c_ext_common as cc  # noqa: E402
from prism_fas.evaluation import c_ext_e8_calibration_authority as auth  # noqa: E402
from prism_fas.evaluation import c_ext_e8_training_runner as runner  # noqa: E402

# --------------------------------------------------------------------------- #
# Namespace / identity constants
# --------------------------------------------------------------------------- #

TARGET_PACKAGE_ID = "prism_target_eval_v2"
TARGET_FEATURE_PACKAGE_IDENTITY = "c3a29e695ad08c4b31e01533f1d12374f4e30c51f0167c6622cf8168792e48a8"
TARGET_FEATURE_ROOT = "data/processed/prism_target_eval_v2"
#: Declared ONLY so the firewall has a root to DENY G7 (and TRAIN); G7 never
#: resolves this path for real.
TARGET_LABEL_ROOT = "data/evaluation_only/prism_target_v2_labels"

PREDICTION_RUN_ROOT = "runs/c_ext_q1q2_v1/e8_qmatched/target_eval_v1"
STAGING_ROOT_RELATIVE = f"{PREDICTION_RUN_ROOT}/.staging"
#: A throwaway per-run trainer run_root/cache_root for G7 model construction
#: ONLY -- never a scientific run root, never the calibration-authority
#: correction namespace.
INFERENCE_CACHE_ROOT_RELATIVE = f"{PREDICTION_RUN_ROOT}/.inference_cache"

EVIDENCE_ROOT_RELATIVE = "reports/c_ext_q1q2_v1/e8_qmatched/target_eval_v1"
PLAN_BINDING_PATH = f"{EVIDENCE_ROOT_RELATIVE}/E8_TARGET_PREDICTION_PLAN_BINDING.json"
LOCKSET_PATH = f"{EVIDENCE_ROOT_RELATIVE}/E8_TARGET_PREDICTION_LOCKSET.json"

PLAN_SCHEMA_VERSION = "e8-target-prediction-plan-v1"
LOCKSET_SCHEMA_VERSION = "e8-target-prediction-lockset-v1"

EXPECTED_RUN_COUNT = 15
EXPECTED_PER_ARM = 5
EXPECTED_SEEDS = tuple(runner.SEEDS)

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class E8TargetEvaluationError(RuntimeError):
    """A precondition for E8 target prediction failed. Fails closed."""


def _repo_root(root: Path | None) -> Path:
    return root or cc.repo_root()


def resolve_e8_code_commit(root: Path | None = None) -> str:
    """Resolves git HEAD exactly once, via the existing lightweight
    ``prism_fas.utils.core.git_commit`` helper -- a subprocess-only
    ``git rev-parse HEAD`` wrapper with no torch/trainer import, so reusing
    it introduces no training capability. Fails closed (raises) unless the
    result is a non-empty 40-hex-char SHA: a scientific G7 plan may never be
    frozen with an empty or malformed ``code_commit``.
    """
    from prism_fas.utils.core import git_commit as _git_commit

    repo = _repo_root(root)
    commit = _git_commit(repo)
    if not commit or not _GIT_SHA_RE.match(commit):
        raise E8TargetEvaluationError(
            f"could not resolve a valid 40-hex-char git HEAD commit for {repo} (got {commit!r}); "
            "refusing to freeze a scientific G7 plan with an unbound code_commit"
        )
    return commit


# --------------------------------------------------------------------------- #
# A. Target feature package verification (reused, never re-derived)
# --------------------------------------------------------------------------- #

def verify_target_feature_package(repo: Path) -> dict[str, Any]:
    from prism_fas.evaluation.post_failure_exploratory_target_v3 import verify_locked_target_feature_package

    result = verify_locked_target_feature_package(
        repo / TARGET_FEATURE_ROOT, expected_package_id=TARGET_PACKAGE_ID,
        expected_content_identity=TARGET_FEATURE_PACKAGE_IDENTITY)
    if not result.get("present_on_this_host"):
        return result
    if not result.get("verified"):
        raise E8TargetEvaluationError(f"target feature package failed verification: {result}")
    return result


# --------------------------------------------------------------------------- #
# B. The target prediction plan -- bound to the frozen calibration authority
# --------------------------------------------------------------------------- #

def prediction_output_path(run_id: str) -> str:
    return f"{PREDICTION_RUN_ROOT}/{run_id}/target_predictions.parquet"


def build_target_prediction_plan(repo: Path, *, code_commit: str = "") -> dict[str, Any]:
    """Binds exactly the 15 E8 runs to their frozen calibration-authority
    checkpoint/calibration, plus the frozen target feature package identity.
    Fails closed if the calibration authority lock is missing/invalid, or if
    it does not cover exactly the 15 frozen run ids.
    """
    lock = auth.require_valid_calibration_authority_for_g7(repo)
    rows_by_id = {row["run_id"]: row for row in lock["rows"]}
    specs = runner.all_scientific_run_specs(repo)
    if sorted(spec.run_id for spec in specs) != sorted(rows_by_id):
        raise E8TargetEvaluationError(
            "the calibration authority lock does not cover exactly the 15 frozen E8 run ids")

    package_check = verify_target_feature_package(repo)

    entries = []
    for spec in specs:
        row = rows_by_id[spec.run_id]
        entries.append({
            "run_id": spec.run_id, "arm": spec.arm, "seed": spec.seed,
            "best_checkpoint_path": row["best_checkpoint_path"],
            "best_checkpoint_sha256": row["best_checkpoint_sha256"],
            "authoritative_calibration_kind": row["authoritative_calibration_kind"],
            "calibration_sha256": row["calibration_file_sha256"],
            "calibration_hash": row["calibration_hash"],
            "temperature": row["temperature"],
            "selected_threshold": row["selected_threshold"],
            "prediction_output_path": prediction_output_path(spec.run_id),
        })

    per_arm_counts = {arm: sum(1 for entry in entries if entry["arm"] == arm) for arm in runner.ARMS}
    if per_arm_counts != {arm: EXPECTED_PER_ARM for arm in runner.ARMS}:
        raise E8TargetEvaluationError(f"expected 5 runs per arm; got {per_arm_counts}")

    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "experiment": "E8_QMATCHED_TRACK_G", "fold": "EXT-F1",
        "arms": list(runner.ARMS), "seeds": list(EXPECTED_SEEDS),
        "calibration_authority_identity": lock["lock_identity"],
        "target_package_id": TARGET_PACKAGE_ID,
        "target_feature_package_identity": TARGET_FEATURE_PACKAGE_IDENTITY,
        "target_feature_package_present_on_this_host": bool(package_check.get("present_on_this_host")),
        "unknown_threshold": None,
        "prediction_schema_version": "m10-prediction-v1",
        "code_commit": code_commit,
        "run_count": len(entries),
        "runs": sorted(entries, key=lambda row: row["run_id"]),
        "target_labels_opened": False,
    }
    plan["plan_identity"] = cc.sha256_bytes(cc.canonical_json_bytes(plan))
    return plan


def freeze_target_prediction_plan(repo: Path, *, code_commit: str | None = None) -> str:
    """Freezes the plan exactly once. A later call recomputes the plan and
    refuses (rather than silently overwrites) if it would now differ.

    ``code_commit=None`` (the production default) resolves git HEAD exactly
    once via ``resolve_e8_code_commit`` -- fail-closed if it cannot be
    resolved to a non-empty 40-hex-char SHA. Engineering/unit fixtures may
    pass an explicit deterministic fake SHA instead; either way, an empty
    ``code_commit`` is refused here, before anything is written.
    """
    resolved_commit = code_commit if code_commit is not None else resolve_e8_code_commit(repo)
    if not resolved_commit:
        raise E8TargetEvaluationError(
            "refusing to freeze a scientific G7 plan with an empty code_commit")
    plan = build_target_prediction_plan(repo, code_commit=resolved_commit)
    existing_path = repo / PLAN_BINDING_PATH
    if existing_path.is_file():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        if existing.get("plan_identity") != plan["plan_identity"]:
            raise E8TargetEvaluationError(
                f"a DIFFERENT target prediction plan is already frozen at {PLAN_BINDING_PATH}; "
                f"refusing to silently overwrite it (frozen plan_identity={existing.get('plan_identity')!r}, "
                f"recomputed={plan['plan_identity']!r})")
        return PLAN_BINDING_PATH
    return cc.write_json_atomic(PLAN_BINDING_PATH, plan, root=repo)


def load_frozen_plan(repo: Path) -> dict[str, Any]:
    path = repo / PLAN_BINDING_PATH
    if not path.is_file():
        raise E8TargetEvaluationError(f"missing frozen plan at {path.as_posix()}; run --preflight first")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not plan.get("code_commit"):
        raise E8TargetEvaluationError(
            "the frozen plan has an empty code_commit; refusing to run scientific inference against "
            "an unbound implementation commit")
    recomputed = build_target_prediction_plan(repo, code_commit=plan.get("code_commit", ""))
    if (recomputed["runs"] != plan["runs"]
            or recomputed["calibration_authority_identity"] != plan["calibration_authority_identity"]):
        raise E8TargetEvaluationError(
            "the frozen plan no longer matches a fresh recomputation from the calibration authority; "
            "refusing to run inference against a drifted plan")
    return plan


# --------------------------------------------------------------------------- #
# C. Target label firewall
# --------------------------------------------------------------------------- #

def build_target_firewall(repo: Path) -> Any:
    from prism_fas.evaluation.firewall import FirewallConfig, TargetLabelFirewall

    config = FirewallConfig(
        roots={
            "source_package_root": repo / runner.M3B_RUNTIME_PACKAGE_RELATIVE_PATH,
            "target_feature_root": repo / TARGET_FEATURE_ROOT,
            "target_label_root": repo / TARGET_LABEL_ROOT,
            "prediction_root": repo / PREDICTION_RUN_ROOT,
        },
        permissions={
            "TRAIN": {"source_package_root": "read", "target_feature_root": "deny",
                     "target_label_root": "deny", "prediction_root": "deny"},
            "G7": {"source_package_root": "deny", "target_feature_root": "read",
                  "target_label_root": "deny", "prediction_root": "write"},
            "G8": {"source_package_root": "deny", "target_feature_root": "deny",
                  "target_label_root": "read", "prediction_root": "read"},
        }).validate()
    return TargetLabelFirewall(config=config, project_root=repo)


# --------------------------------------------------------------------------- #
# Per-run model construction (GPU-only; injectable for tests)
# --------------------------------------------------------------------------- #

def default_e8_model_provider(*, repo: Path, run_id: str, arm: str, seed: int,
                              best_checkpoint_path: str, best_checkpoint_sha256: str,
                              _trainer_provider: Callable[..., Any] | None = None,
                              _checkpoint_loader: Callable[[Path, Any], dict[str, Any]] | None = None,
                              _device_resolver: Callable[[], str] | None = None,
                              _override_detector_inputs: dict[str, Any] | None = None,
                              _override_c3_bank: dict[str, Any] | None = None,
                              _skip_m3b_guard: bool = False) -> dict[str, Any]:
    """The REAL, GPU-only checkpoint load: reuses the SAME canonical
    ``_resolve_e8_launch_bindings`` resolver the scientific launcher and the
    calibration authority correction both use, then
    ``target_prediction.load_checkpoint_for_inference`` to restore
    ``best.pt`` -- never ``trainer.resume()``, which resumes full training
    state this module has no business touching. The trainer's own
    ``run_root``/``cache_root`` are a THROWAWAY per-run inference cache
    directory, never a scientific run root. Never invoked in a test, which
    injects a fake ``model_provider`` instead.
    """
    from prism_fas.evaluation.c_ext_e8_calibration_authority import default_trainer_provider
    from prism_fas.evaluation.c_ext_e8_training_runner import _resolve_e8_launch_bindings
    from prism_fas.evaluation.target_prediction import VariantCapabilities, load_checkpoint_for_inference

    spec = runner.build_run_spec(arm, seed)
    bindings = _resolve_e8_launch_bindings(
        spec, root=repo, _skip_m3b_guard=_skip_m3b_guard,
        _override_detector_inputs=_override_detector_inputs, _override_c3_bank=_override_c3_bank,
        _device_resolver=_device_resolver)
    inference_run_root = repo / INFERENCE_CACHE_ROOT_RELATIVE / run_id

    provider = _trainer_provider or default_trainer_provider
    trainer = provider(spec, bindings, inference_run_root)

    loader = _checkpoint_loader or load_checkpoint_for_inference
    load_result = loader(repo / best_checkpoint_path, trainer.model)
    if load_result.get("checkpoint_sha256") != best_checkpoint_sha256:
        raise E8TargetEvaluationError(
            f"{best_checkpoint_path}: loaded checkpoint SHA256 {load_result.get('checkpoint_sha256')!r} "
            f"!= frozen expected {best_checkpoint_sha256!r} -- refusing to predict against a drifted checkpoint"
        )
    return {"model": trainer.model, "capabilities": VariantCapabilities.from_variant(trainer.config.variant),
           "device": bindings["device"], "loader_config": trainer.loader_config, "cache_root": trainer.cache_root,
           "architecture_identity": trainer.model.architecture_identity()}


# --------------------------------------------------------------------------- #
# Per-run prediction to staging
# --------------------------------------------------------------------------- #

def predict_e8_run_to_staging(*, repo: Path, plan_entry: dict[str, Any], package_root: Path,
                              firewall: Any, staging_root: Path, code_commit: str,
                              calibration_authority_identity: str,
                              model_provider: Callable[..., dict[str, Any]] | None = None
                              ) -> dict[str, Any]:
    """Real, label-free target inference for ONE E8 run, writing ONLY into
    the disposable staging namespace."""
    from prism_fas.evaluation.post_failure_exploratory_target_v3 import build_verified_target_loader_config
    from prism_fas.evaluation.target_prediction import (PREDICTION_LOCK_FILE, build_prediction_lock,
                                                         inference_config_hash, predict_target,
                                                         target_batches, write_prediction_lock,
                                                         write_predictions)

    provider = model_provider or default_e8_model_provider
    run_id = plan_entry["run_id"]
    resolved = provider(repo=repo, run_id=run_id, arm=plan_entry["arm"], seed=plan_entry["seed"],
                        best_checkpoint_path=plan_entry["best_checkpoint_path"],
                        best_checkpoint_sha256=plan_entry["best_checkpoint_sha256"])

    target_loader_config = build_verified_target_loader_config(
        resolved["loader_config"], target_package_id=TARGET_PACKAGE_ID,
        target_content_identity=TARGET_FEATURE_PACKAGE_IDENTITY)
    batches = target_batches(Path(package_root), target_loader_config,
                             cache_root=resolved["cache_root"], firewall=firewall)

    variant = f"E8_QMATCH_{plan_entry['arm']}"
    config_hash = inference_config_hash(
        variant=variant, flags={}, threshold=plan_entry["selected_threshold"],
        unknown_threshold=None, temperature=plan_entry["temperature"],
        package_identity=TARGET_FEATURE_PACKAGE_IDENTITY,
        architecture_identity=resolved["architecture_identity"])

    rows = predict_target(
        resolved["model"], batches, capabilities=resolved["capabilities"],
        threshold=plan_entry["selected_threshold"], unknown_threshold=None,
        temperature=plan_entry["temperature"],
        checkpoint_hash=plan_entry["best_checkpoint_sha256"],
        calibration_hash=plan_entry["calibration_hash"],
        inference_config_hash=config_hash, variant=variant, device=resolved["device"])

    run_staging_dir = Path(staging_root) / run_id
    run_staging_dir.mkdir(parents=True, exist_ok=True)
    (run_staging_dir / "STAGING_MARKER.json").write_text(
        json.dumps({"marker": "E8_TARGET_STAGING", "run_id": run_id}), encoding="utf-8")
    prediction_path = run_staging_dir / "target_predictions.parquet"
    write_predictions(prediction_path, rows, variant=variant, firewall=firewall)
    prediction_file_sha256 = cc.sha256_file(prediction_path)

    lock = build_prediction_lock(
        experiment_id=run_id, variant=variant, seed=plan_entry["seed"], rows=rows,
        checkpoint_sha256=plan_entry["best_checkpoint_sha256"],
        source_calibration_sha256=plan_entry["calibration_sha256"],
        calibration_hash=plan_entry["calibration_hash"],
        inference_config_hash=config_hash,
        target_feature_package_identity=TARGET_FEATURE_PACKAGE_IDENTITY,
        target_package_id=TARGET_PACKAGE_ID, threshold=plan_entry["selected_threshold"],
        unknown_threshold=None, source_matrix_lock_identity=calibration_authority_identity,
        code_commit=code_commit, engineering_smoke=False)
    write_prediction_lock(run_staging_dir / PREDICTION_LOCK_FILE, lock)
    return {"run_id": run_id, "staging_dir": str(run_staging_dir),
           "prediction_file_sha256": prediction_file_sha256, "row_count": len(rows), "lock": lock}


# --------------------------------------------------------------------------- #
# Staging -> atomic promotion -> global lock
# --------------------------------------------------------------------------- #

def build_promotion_transaction(*, plan_identity: str, run_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    from prism_fas.evaluation.target_prediction import PREDICTION_LOCK_FILE

    run_ids = sorted(run_results)
    staged = {}
    for run_id in run_ids:
        result = run_results[run_id]
        lock_path = Path(result["staging_dir"]) / PREDICTION_LOCK_FILE
        staged[run_id] = {
            "prediction_file_sha256": result["prediction_file_sha256"],
            "prediction_lock_identity": result["lock"]["prediction_lock_identity"],
            "lock_file_sha256": cc.sha256_file(lock_path),
        }
    body = {"schema_version": "e8-target-prediction-promotion-transaction-v1",
           "plan_identity": plan_identity, "run_ids": run_ids, "staged_artifacts": staged}
    body["transaction_identity"] = cc.sha256_bytes(cc.canonical_json_bytes(body))
    body["state"] = "READY_TO_PROMOTE"
    return body


def _validate_promoted_run(directory: Path, staged_record: dict[str, Any], run_id: str) -> None:
    from prism_fas.evaluation.target_prediction import PREDICTION_LOCK_FILE

    prediction_path = directory / "target_predictions.parquet"
    if not prediction_path.is_file():
        raise E8TargetEvaluationError(f"{run_id}: prediction file missing at {directory}")
    if cc.sha256_file(prediction_path) != staged_record["prediction_file_sha256"]:
        raise E8TargetEvaluationError(f"{run_id}: prediction file hash disagrees with the promotion transaction")
    lock_path = directory / PREDICTION_LOCK_FILE
    if not lock_path.is_file():
        raise E8TargetEvaluationError(f"{run_id}: lock file missing at {directory}")
    if cc.sha256_file(lock_path) != staged_record["lock_file_sha256"]:
        raise E8TargetEvaluationError(f"{run_id}: lock file hash disagrees with the promotion transaction")


def promote_staged_runs(repo: Path, *, staging_root: Path, plan_identity: str,
                        run_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Crash-recoverable promotion: writes the transaction (READY_TO_PROMOTE)
    before any rename; a run already promoted is validated in place by hash
    and left untouched, never re-inferred."""
    transaction_path = (Path(staging_root) / ".." / "PREDICTION_PROMOTION_TRANSACTION.json").resolve()
    if transaction_path.is_file():
        transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
        if sorted(transaction["run_ids"]) != sorted(run_results):
            raise E8TargetEvaluationError("an existing promotion transaction does not match the requested run set")
    else:
        transaction = build_promotion_transaction(plan_identity=plan_identity, run_results=run_results)
        transaction_path.write_text(json.dumps(transaction), encoding="utf-8")

    run_root = repo / PREDICTION_RUN_ROOT
    for run_id in transaction["run_ids"]:
        staged_record = transaction["staged_artifacts"][run_id]
        final_dir = run_root / run_id
        staging_dir = Path(staging_root) / run_id
        if final_dir.is_dir():
            _validate_promoted_run(final_dir, staged_record, run_id)
            continue
        if not staging_dir.is_dir():
            raise E8TargetEvaluationError(f"{run_id}: neither staged nor promoted; cannot recover")
        _validate_promoted_run(staging_dir, staged_record, run_id)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir.rename(final_dir)
        (final_dir / "STAGING_MARKER.json").unlink(missing_ok=True)

    transaction["state"] = "COMPLETE"
    transaction_path.write_text(json.dumps(transaction), encoding="utf-8")
    return transaction


# --------------------------------------------------------------------------- #
# Global manifest / lockset
# --------------------------------------------------------------------------- #

def build_prediction_manifest(plan: dict[str, Any], run_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expected_ids = {entry["run_id"] for entry in plan["runs"]}
    if set(run_results) != expected_ids:
        raise E8TargetEvaluationError(
            f"expected exactly the {len(expected_ids)} frozen run ids, got {sorted(run_results)}")
    plan_commit = plan.get("code_commit") or ""
    if not plan_commit:
        raise E8TargetEvaluationError("refusing to build a manifest from a plan with an empty code_commit")
    rows = []
    for entry in plan["runs"]:
        run_id = entry["run_id"]
        result = run_results[run_id]
        lock = result["lock"]
        if lock.get("code_commit") != plan_commit:
            raise E8TargetEvaluationError(
                f"{run_id}: prediction lock code_commit {lock.get('code_commit')!r} disagrees with the "
                f"frozen plan's code_commit {plan_commit!r}")
        rows.append({
            "run_id": run_id, "arm": entry["arm"], "seed": entry["seed"],
            "best_checkpoint_sha256": entry["best_checkpoint_sha256"],
            "calibration_sha256": entry["calibration_sha256"], "calibration_hash": entry["calibration_hash"],
            "target_feature_package_identity": TARGET_FEATURE_PACKAGE_IDENTITY,
            "prediction_output_path": entry["prediction_output_path"],
            "prediction_file_sha256": result["prediction_file_sha256"],
            "prediction_lock_identity": lock["prediction_lock_identity"],
            "code_commit": lock["code_commit"],
            "row_count": result["row_count"], "video_count": lock["video_count"],
            "no_label_audit": {"labels_present": False, "checked": True},
        })
    return {"schema_version": "e8-target-prediction-manifest-v1", "plan_identity": plan["plan_identity"],
           "calibration_authority_identity": plan["calibration_authority_identity"],
           "code_commit": plan_commit,
           "run_count": len(rows), "runs": sorted(rows, key=lambda row: row["run_id"])}


def build_target_prediction_lockset(manifest: dict[str, Any]) -> dict[str, Any]:
    if int(manifest.get("run_count", -1)) != EXPECTED_RUN_COUNT:
        raise E8TargetEvaluationError(f"expected exactly {EXPECTED_RUN_COUNT} runs, got {manifest.get('run_count')}")
    manifest_commit = manifest.get("code_commit") or ""
    if not manifest_commit:
        raise E8TargetEvaluationError("refusing to freeze a scientific lockset with an empty code_commit")
    by_arm: dict[str, list[int]] = {}
    for row in manifest["runs"]:
        by_arm.setdefault(row["arm"], []).append(row["seed"])
        if row.get("code_commit") != manifest_commit:
            raise E8TargetEvaluationError(
                f"{row['run_id']}: code_commit {row.get('code_commit')!r} disagrees with the manifest's "
                f"{manifest_commit!r}; every prediction lock must bind the SAME code_commit")
    if set(by_arm) != set(runner.ARMS) or any(len(seeds) != EXPECTED_PER_ARM for seeds in by_arm.values()):
        raise E8TargetEvaluationError(f"expected 5 RND + 5 DET + 5 LLM; got {[(a, len(s)) for a, s in by_arm.items()]}")
    for arm, seeds in by_arm.items():
        if sorted(seeds) != sorted(EXPECTED_SEEDS):
            raise E8TargetEvaluationError(f"{arm}: seed set {sorted(seeds)} != frozen {sorted(EXPECTED_SEEDS)}")

    body = {
        "lockset_schema_version": LOCKSET_SCHEMA_VERSION,
        "experiment": "E8_QMATCHED_TRACK_G", "fold": "EXT-F1", "arms": list(runner.ARMS), "seeds": list(EXPECTED_SEEDS),
        "plan_identity": manifest["plan_identity"],
        "calibration_authority_identity": manifest["calibration_authority_identity"],
        "target_feature_package_identity": TARGET_FEATURE_PACKAGE_IDENTITY,
        "code_commit": manifest_commit,
        "run_count": manifest["run_count"],
        "entries": sorted(({"run_id": row["run_id"], "arm": row["arm"], "seed": row["seed"],
                            "best_checkpoint_sha256": row["best_checkpoint_sha256"],
                            "calibration_sha256": row["calibration_sha256"],
                            "calibration_hash": row["calibration_hash"],
                            "prediction_lock_identity": row["prediction_lock_identity"],
                            "prediction_file_sha256": row["prediction_file_sha256"],
                            "row_count": row["row_count"]} for row in manifest["runs"]),
                          key=lambda entry: entry["run_id"]),
        "frame_rows_total": sum(int(row["row_count"]) for row in manifest["runs"]),
        "target_features_accessed": True, "target_labels_accessed": False,
        "no_engineering_smoke_entries": True,
        "status": "FROZEN",
    }
    body["lockset_identity"] = cc.sha256_bytes(cc.canonical_json_bytes(body))
    return {**body, "lock_identity": body["lockset_identity"]}


def is_usable_lockset(payload: dict[str, Any]) -> bool:
    if payload.get("status") != "FROZEN":
        return False
    if int(payload.get("run_count", -1)) != EXPECTED_RUN_COUNT:
        return False
    if payload.get("target_labels_accessed") is not False:
        return False
    if not payload.get("code_commit"):
        return False
    body = {key: value for key, value in payload.items() if key not in ("lockset_identity", "lock_identity")}
    recomputed = cc.sha256_bytes(cc.canonical_json_bytes(body))
    return recomputed == payload.get("lockset_identity") == payload.get("lock_identity")


def load_lockset_if_usable(root: Path | None = None) -> dict[str, Any] | None:
    repo = _repo_root(root)
    path = repo / LOCKSET_PATH
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if is_usable_lockset(payload) else None


def require_valid_lockset_for_g8(root: Path | None = None) -> dict[str, Any]:
    lockset = load_lockset_if_usable(root)
    if lockset is None:
        raise E8TargetEvaluationError(
            f"no valid, complete 15-row E8 target prediction lockset found at {LOCKSET_PATH}; G8 may not score")
    return lockset


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_target_prediction(repo: Path, *, package_root: Path | None = None,
                          model_provider: Callable[..., dict[str, Any]] | None = None) -> dict[str, str]:
    """Builds everything in memory first (a mid-run failure writes nothing
    into the final namespace -- staged rows stay staged), then publishes the
    manifest, and the LOCKSET LAST."""
    plan = load_frozen_plan(repo)
    firewall = build_target_firewall(repo)
    staging_root = repo / STAGING_ROOT_RELATIVE
    run_results: dict[str, dict[str, Any]] = {}
    for entry in plan["runs"]:
        run_results[entry["run_id"]] = predict_e8_run_to_staging(
            repo=repo, plan_entry=entry, package_root=package_root or (repo / TARGET_FEATURE_ROOT),
            firewall=firewall, staging_root=staging_root, code_commit=plan["code_commit"],
            calibration_authority_identity=plan["calibration_authority_identity"], model_provider=model_provider)
    promote_staged_runs(repo, staging_root=staging_root, plan_identity=plan["plan_identity"],
                        run_results=run_results)

    manifest = build_prediction_manifest(plan, run_results)
    lockset = build_target_prediction_lockset(manifest)
    written = {"lockset": cc.write_json_atomic(LOCKSET_PATH, lockset, root=repo)}
    return written


def preflight_e8_target_prediction(root: Path | None = None) -> dict[str, Any]:
    """Metadata-only preflight. Never opens ``target_batches``/``predict_target``,
    never touches ``target_label_root``, never requires a real GPU."""
    repo = _repo_root(root)
    auth_lock = auth.load_calibration_authority_lock_if_usable(repo)
    package_check = None
    package_error = None
    try:
        package_check = verify_target_feature_package(repo)
    except E8TargetEvaluationError as exc:
        package_error = str(exc)
    code_commit = None
    code_commit_error = None
    try:
        code_commit = resolve_e8_code_commit(repo)
    except E8TargetEvaluationError as exc:
        code_commit_error = str(exc)
    plan_path = repo / PLAN_BINDING_PATH
    return {
        "schema_version": "e8-target-prediction-preflight-v1",
        "calibration_authority_valid": auth_lock is not None,
        "calibration_authority_identity": (auth_lock or {}).get("lock_identity"),
        "target_package_id": TARGET_PACKAGE_ID,
        "target_feature_package_identity": TARGET_FEATURE_PACKAGE_IDENTITY,
        "target_feature_package_check": package_check, "target_feature_package_error": package_error,
        "code_commit": code_commit, "code_commit_error": code_commit_error,
        "plan_frozen": plan_path.is_file(),
        "unknown_threshold": None, "target_labels_accessed": False, "target_features_opened": False,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E8 G7 target prediction (label-free)")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--authorize-target-feature-inference", action="store_true")
    args = parser.parse_args(argv)
    repo = cc.repo_root()

    if args.preflight:
        result = preflight_e8_target_prediction(repo)
        print(json.dumps(result, indent=2))
        if result["calibration_authority_valid"] and result["code_commit"]:
            freeze_target_prediction_plan(repo, code_commit=result["code_commit"])
        elif result["calibration_authority_valid"] and not result["code_commit"]:
            print(f"refusing to freeze the G7 plan: {result['code_commit_error']}")
            return 2
        return 0

    if args.predict:
        if not args.authorize_target_feature_inference:
            print("--predict requires --authorize-target-feature-inference as an explicit second flag.")
            return 2
        written = run_target_prediction(repo)
        print(json.dumps(written))
        return 0

    print("Pass --preflight to validate + freeze the plan, or --predict "
         "--authorize-target-feature-inference to run the real GPU-side inference.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
