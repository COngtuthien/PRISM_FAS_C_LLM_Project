"""PRISM-FAS-C EXT-Q1Q2 -- E8 calibration authority (additive correction).

A read-only GPU audit over all 15 completed E8 scientific runs
(``runs/c_ext_q1q2_v1/e8_qmatched/ext_f1/``) found that 14/15 runs have their
``best.pt`` checkpoint at the SAME (epoch, global_step) position as the
final in-memory G5 model, and exactly one -- ``e8_ext_f1_g_det_qmatch_s20260809``
-- does not (best=epoch30/global_step1350, last=epoch35/global_step1575).

``M9Trainer.run_g6()`` never loads ``checkpoint`` from disk: it only uses the
``checkpoint`` argument to compute ``checkpoint_sha`` for provenance, while
``self.source_dev_logits()`` reads whatever model state happens to be in
memory. For the 14 runs where best==last this is harmless (the in-memory
model IS the best-checkpoint model); for the one mismatched run, the
existing ``calibration/source_dev.json`` was fitted against the FINAL
in-memory model while its own recorded ``checkpoint_sha256`` claims
``best.pt`` -- a provenance/computation mismatch, not a scientific-protocol
defect (no membership, bank, seed, model, optimizer, schedule or
threshold-*rule* changed).

This module NEVER overwrites the historical evidence
(``run_root/calibration/source_dev.json``, ``run_root/stages/G6/*``,
``run_root/run.json``, ``run_root/checkpoints/*``): for the 14 valid runs it
retains the existing calibration as authoritative; for the one affected run
it recomputes calibration SOURCE-ONLY from the frozen ``best.pt`` itself,
writing ONLY into a dedicated additive correction namespace
(``runs/c_ext_q1q2_v1/e8_qmatched/calibration_authority_v1/<run_id>/``),
never the scientific training run root.

The correction reconstructs the exact E8 Track-G runtime environment via the
EXISTING canonical resolver (``c_ext_e8_training_runner._resolve_e8_launch_bindings``,
reused, never duplicated) -- never ``launch_scientific_run`` itself, because
the 15 canonical run roots are already COMPLETED and must not be mutated.
Model/prototype restore prefers
``prism_fas.evaluation.target_prediction.load_checkpoint_for_inference``.
Forward inference is ``source_dev`` only, under ``torch.no_grad()``
(``M9Trainer.source_dev_logits``), reusing the EXACT existing
``prism_fas.train.calibration.calibrate_source_dev`` fitting/selection logic
E6's own ``run_g6`` uses. No gradient, no optimizer step, no target feature,
no target label, ever.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prism_fas.evaluation import c_ext_common as cc  # noqa: E402
from prism_fas.evaluation import c_ext_e8_training_runner as runner  # noqa: E402

# --------------------------------------------------------------------------- #
# Namespace / identity constants
# --------------------------------------------------------------------------- #

CORRECTION_RUN_ROOT_RELATIVE = "runs/c_ext_q1q2_v1/e8_qmatched/calibration_authority_v1"
EVIDENCE_ROOT_RELATIVE = "reports/c_ext_q1q2_v1/e8_qmatched/calibration_authority_v1"
LOCK_RELATIVE_PATH = f"{EVIDENCE_ROOT_RELATIVE}/E8_CALIBRATION_AUTHORITY_LOCK.json"
LOCK_SCHEMA_VERSION = "e8-calibration-authority-lock-v1"

#: Frozen by the GPU read-only audit. The ONLY run this module may ever treat
#: as needing correction; every other run is asserted to agree (see
#: ``build_row_for_run``'s fail-closed cross-checks below).
CORRECTED_RUN_ID = "e8_ext_f1_g_det_qmatch_s20260809"
EXPECTED_RUN_COUNT = 15
EXPECTED_CORRECTED_COUNT = 1


class CalibrationAuthorityError(RuntimeError):
    """The E8 calibration authority refuses to proceed."""


def _repo_root(root: Path | None) -> Path:
    return root or cc.repo_root()


def _display_path(path: Path, repo: Path) -> str:
    """Repo-relative when possible (the common case); the absolute path
    otherwise -- e.g. a test fixture root that legitimately lives outside
    ``repo`` (a tmp_path correction namespace override)."""
    try:
        return str(Path(path).relative_to(repo))
    except ValueError:
        return str(Path(path))


# --------------------------------------------------------------------------- #
# Enumeration -- reuses the EXISTING E8 runner authority; never re-derives.
# --------------------------------------------------------------------------- #

def enumerate_e8_run_specs(root: Path | None = None) -> tuple[Any, ...]:
    """The frozen 15 E8 scientific run specs, via the existing runner authority."""
    specs = runner.all_scientific_run_specs(root)
    if len(specs) != EXPECTED_RUN_COUNT:
        raise CalibrationAuthorityError(
            f"expected exactly {EXPECTED_RUN_COUNT} E8 scientific run specs, got {len(specs)}")
    return specs


def run_root_for_spec(spec: Any, root: Path | None = None) -> Path:
    return _repo_root(root) / spec.run_root


def correction_run_root(spec: Any, root: Path | None = None) -> Path:
    return _repo_root(root) / CORRECTION_RUN_ROOT_RELATIVE / spec.run_id


def verify_run_completed(run_root: Path) -> Any:
    """Fail-closed: only a COMPLETED run may enter the calibration authority."""
    state = runner.classify_run_state(run_root)
    if state is not runner.RunState.COMPLETED:
        raise CalibrationAuthorityError(f"{run_root}: run state {state.value} != COMPLETED")
    return state


def read_run_summary(run_root: Path) -> dict[str, Any]:
    path = run_root / "run.json"
    if not path.is_file():
        raise CalibrationAuthorityError(f"{path}: run.json missing -- run is not COMPLETED")
    return cc.read_json(path)


def read_historical_calibration(run_root: Path) -> tuple[dict[str, Any], Path]:
    path = run_root / "calibration" / "source_dev.json"
    if not path.is_file():
        raise CalibrationAuthorityError(f"{path}: historical calibration/source_dev.json missing")
    return cc.read_json(path), path


def best_and_last_positions(run_summary: dict[str, Any]) -> dict[str, Any]:
    """Extracted from ``run.json`` alone -- never opens a checkpoint just to
    compare positions: ``best_metrics.{epoch,global_step}`` is written by
    ``M9Trainer.run_g5()`` at the moment ``save(\"best\")`` is called, and the
    run-level ``epoch``/``global_step`` are the FINAL (``last.pt``) position.
    """
    best = run_summary.get("best_metrics") or {}
    best_epoch, best_step = best.get("epoch"), best.get("global_step")
    last_epoch, last_step = run_summary.get("epoch"), run_summary.get("global_step")
    return {"best_epoch": best_epoch, "best_global_step": best_step,
            "last_epoch": last_epoch, "last_global_step": last_step,
            "positions_match": (best_epoch == last_epoch and best_step == last_step)}


# --------------------------------------------------------------------------- #
# Best-checkpoint-only correction (the ONE affected run)
# --------------------------------------------------------------------------- #

def default_trainer_provider(spec: Any, bindings: dict[str, Any], run_root: Path) -> Any:
    """Real construction: the SAME canonical inputs
    ``_resolve_e8_launch_bindings`` already resolved, pointed at the
    dedicated correction namespace run_root -- NEVER the scientific training
    run root. Never invoked in a test, which injects a fake
    ``trainer_provider`` instead."""
    repo = bindings["repo"]
    detector_inputs = bindings["detector_inputs"]
    trainer_cls = bindings["trainer_cls"]
    return trainer_cls(
        config=bindings["training_config"], detector_config=bindings["detector_config"],
        package_root=repo / detector_inputs["package_root"],
        bank_root=repo / detector_inputs["candidates_root"],
        recipe_bank_root=repo / detector_inputs["recipe_bank_root"],
        run_root=run_root, cache_root=run_root / "cache",
        weight_root=repo / detector_inputs["weight_root"],
        loader_config_path=repo / runner.LOADER_CONFIG_RELATIVE_PATH,
        synthetic_bank=bindings["e8_bank"], device=bindings["device"],
    )


def correct_calibration_via_best_checkpoint(
    spec: Any, root: Path | None = None, *, best_checkpoint_path: Path, best_checkpoint_sha256: str,
    trainer_provider: Callable[[Any, dict[str, Any], Path], Any] | None = None,
    checkpoint_loader: Callable[[Path, Any], dict[str, Any]] | None = None,
    _correction_root: Path | None = None,
    _device_resolver: Callable[[], str] | None = None,
    _override_detector_inputs: dict[str, Any] | None = None,
    _override_c3_bank: dict[str, Any] | None = None,
    _skip_m3b_guard: bool = False,
) -> dict[str, Any]:
    """Reconstruct the exact E8 Track-G environment, load ``best.pt`` itself,
    run ``source_dev`` forward ONLY (no gradient, no target), and fit a NEW
    calibration record into the additive correction namespace.

    Fail-closed if the loaded checkpoint's own recomputed SHA256 disagrees
    with ``best_checkpoint_sha256`` -- the correction must be fitted against
    the EXACT frozen best checkpoint, never a drifted substitute.
    """
    from prism_fas.evaluation.c_ext_e8_training_runner import _resolve_e8_launch_bindings
    from prism_fas.evaluation.target_prediction import load_checkpoint_for_inference

    repo = _repo_root(root)
    bindings = _resolve_e8_launch_bindings(
        spec, root=repo, _skip_m3b_guard=_skip_m3b_guard,
        _override_detector_inputs=_override_detector_inputs, _override_c3_bank=_override_c3_bank,
        _device_resolver=_device_resolver,
    )
    run_root = Path(_correction_root) if _correction_root is not None else correction_run_root(spec, repo)

    provider = trainer_provider or default_trainer_provider
    trainer = provider(spec, bindings, run_root)

    loader = checkpoint_loader or load_checkpoint_for_inference
    load_result = loader(best_checkpoint_path, trainer.model)
    if load_result.get("checkpoint_sha256") != best_checkpoint_sha256:
        raise CalibrationAuthorityError(
            f"{best_checkpoint_path}: loaded checkpoint SHA256 {load_result.get('checkpoint_sha256')!r} "
            f"!= expected {best_checkpoint_sha256!r} -- refusing to calibrate against a drifted checkpoint"
        )

    logits, targets = trainer.source_dev_logits()

    import hashlib
    import json as _json
    prediction_hash = hashlib.sha256(
        _json.dumps([[format(float(value), ".12g") for value in logits.tolist()],
                    [int(value) for value in targets.tolist()]],
                   separators=(",", ":")).encode("utf-8")).hexdigest()

    from prism_fas.train.calibration import calibrate_source_dev
    record = calibrate_source_dev(
        logits, targets, checkpoint_sha=best_checkpoint_sha256,
        package_identity=trainer.dataset.package_identity,
        prediction_hash=prediction_hash, run_root=run_root)
    calibration_path = run_root / "calibration" / "source_dev.json"
    return {
        "record": record,
        "calibration_path": calibration_path,
        "calibration_file_sha256": cc.sha256_file(calibration_path),
        "device": bindings["device"],
        "source_dev_opened": True, "target_feature_accessed": False, "target_labels_accessed": False,
        "produced_gradient": False,
    }


# --------------------------------------------------------------------------- #
# Per-row authority builder -- 14 retained, 1 corrected
# --------------------------------------------------------------------------- #

def build_row_for_run(
    spec: Any, root: Path | None = None, *,
    trainer_provider: Callable[[Any, dict[str, Any], Path], Any] | None = None,
    checkpoint_loader: Callable[[Path, Any], dict[str, Any]] | None = None,
    _run_root_resolver: Callable[[Any], Path] | None = None,
    _correction_root: Path | None = None,
    _device_resolver: Callable[[], str] | None = None,
    _override_detector_inputs: dict[str, Any] | None = None,
    _override_c3_bank: dict[str, Any] | None = None,
    _skip_m3b_guard: bool = False,
) -> dict[str, Any]:
    repo = _repo_root(root)
    run_root = _run_root_resolver(spec) if _run_root_resolver else run_root_for_spec(spec, repo)
    verify_run_completed(run_root)
    run_summary = read_run_summary(run_root)
    positions = best_and_last_positions(run_summary)
    historical_record, historical_path = read_historical_calibration(run_root)
    historical_sha256 = cc.sha256_file(historical_path)

    best_checkpoint_path = run_root / "checkpoints" / "best.pt"
    if not best_checkpoint_path.is_file():
        raise CalibrationAuthorityError(f"{best_checkpoint_path}: best checkpoint missing")
    best_checkpoint_sha256 = cc.sha256_file(best_checkpoint_path)

    # Fail-closed cross-check: the frozen audit names EXACTLY one affected
    # run. A drift in either direction (the named run now agreeing, or a
    # DIFFERENT run now disagreeing) means the audit is stale and this
    # module must refuse rather than silently apply/skip a correction
    # outside its authorized scope.
    if spec.run_id == CORRECTED_RUN_ID and positions["positions_match"]:
        raise CalibrationAuthorityError(
            f"{spec.run_id}: expected the frozen audited best/last position mismatch, but best and "
            "last now agree -- refusing to apply a stale correction rule")
    if spec.run_id != CORRECTED_RUN_ID and not positions["positions_match"]:
        raise CalibrationAuthorityError(
            f"{spec.run_id}: a best/last position mismatch was found on a run NOT in the frozen "
            f"correction set ({CORRECTED_RUN_ID}!r); refusing to silently apply or skip a correction "
            "outside the audited scope -- this needs a new explicit audit entry first"
        )

    correction_required = spec.run_id == CORRECTED_RUN_ID
    base_row = {
        "run_id": spec.run_id, "arm": spec.arm, "seed": spec.seed,
        "best_checkpoint_path": _display_path(best_checkpoint_path, repo),
        "best_checkpoint_sha256": best_checkpoint_sha256,
        "best_epoch": positions["best_epoch"], "best_global_step": positions["best_global_step"],
        "historical_calibration_path": _display_path(historical_path, repo),
        "historical_calibration_hash": historical_record.get("calibration_hash"),
        "correction_required": correction_required,
        "target_feature_accessed": False, "target_labels_accessed": False, "produced_gradient": False,
    }

    if not correction_required:
        return {
            **base_row,
            "authoritative_calibration_kind": "HISTORICAL_VALID",
            "authoritative_calibration_path": _display_path(historical_path, repo),
            "calibration_file_sha256": historical_sha256,
            "calibration_hash": historical_record.get("calibration_hash"),
            "temperature": historical_record.get("temperature"),
            "selected_threshold": historical_record.get("selected_threshold"),
            "source_dev_prediction_hash": historical_record.get("source_dev_prediction_hash"),
        }

    corrected = correct_calibration_via_best_checkpoint(
        spec, repo, best_checkpoint_path=best_checkpoint_path, best_checkpoint_sha256=best_checkpoint_sha256,
        trainer_provider=trainer_provider, checkpoint_loader=checkpoint_loader,
        _correction_root=_correction_root, _device_resolver=_device_resolver,
        _override_detector_inputs=_override_detector_inputs, _override_c3_bank=_override_c3_bank,
        _skip_m3b_guard=_skip_m3b_guard,
    )
    return {
        **base_row,
        "authoritative_calibration_kind": "BEST_CHECKPOINT_CORRECTION_V1",
        "authoritative_calibration_path": _display_path(corrected["calibration_path"], repo),
        "calibration_file_sha256": corrected["calibration_file_sha256"],
        "calibration_hash": corrected["record"]["calibration_hash"],
        "temperature": corrected["record"]["temperature"],
        "selected_threshold": corrected["record"]["selected_threshold"],
        "source_dev_prediction_hash": corrected["record"]["source_dev_prediction_hash"],
    }


# --------------------------------------------------------------------------- #
# The 15-row lock
# --------------------------------------------------------------------------- #

def build_calibration_authority_lock(
    root: Path | None = None, *,
    trainer_provider: Callable[[Any, dict[str, Any], Path], Any] | None = None,
    checkpoint_loader: Callable[[Path, Any], dict[str, Any]] | None = None,
    _run_root_resolver: Callable[[Any], Path] | None = None,
    _correction_root_resolver: Callable[[Any], Path] | None = None,
    _device_resolver: Callable[[], str] | None = None,
    _override_detector_inputs_by_run: dict[str, dict[str, Any]] | None = None,
    _override_c3_bank_by_arm: dict[str, dict[str, Any]] | None = None,
    _skip_m3b_guard: bool = False,
) -> dict[str, Any]:
    """Enumerate EXACTLY the existing 15 E8 scientific run specs, build one
    authority row per run, and freeze the whole set as an immutable lock.
    Refuses (raises) unless exactly 15 rows resolve and exactly 1 required a
    correction, for the frozen ``CORRECTED_RUN_ID``.
    """
    repo = _repo_root(root)
    specs = enumerate_e8_run_specs(repo)
    rows = []
    for spec in specs:
        override_inputs = ((_override_detector_inputs_by_run or {}).get(spec.run_id))
        override_c3 = ((_override_c3_bank_by_arm or {}).get(spec.arm))
        correction_root = (_correction_root_resolver(spec) if _correction_root_resolver else None)
        rows.append(build_row_for_run(
            spec, repo, trainer_provider=trainer_provider, checkpoint_loader=checkpoint_loader,
            _run_root_resolver=_run_root_resolver, _correction_root=correction_root,
            _device_resolver=_device_resolver, _override_detector_inputs=override_inputs,
            _override_c3_bank=override_c3, _skip_m3b_guard=_skip_m3b_guard,
        ))

    corrected = [row for row in rows if row["correction_required"]]
    if len(corrected) != EXPECTED_CORRECTED_COUNT or corrected[0]["run_id"] != CORRECTED_RUN_ID:
        raise CalibrationAuthorityError(
            f"expected exactly {EXPECTED_CORRECTED_COUNT} correction, for {CORRECTED_RUN_ID!r}; "
            f"got {[row['run_id'] for row in corrected]}"
        )

    body = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "entry_count": len(rows),
        "corrected_entry_count": len(corrected),
        "corrected_run_ids": sorted(row["run_id"] for row in corrected),
        "source_only": True,
        "target_access": False,
        "rows": sorted(rows, key=lambda row: row["run_id"]),
    }
    return {**body, "lock_identity": cc.sha256_bytes(cc.canonical_json_bytes(body))}


def write_calibration_authority_lock(root: Path | None = None, **kwargs: Any) -> tuple[str, dict[str, Any]]:
    repo = _repo_root(root)
    lock = build_calibration_authority_lock(repo, **kwargs)
    written = cc.write_json_atomic(LOCK_RELATIVE_PATH, lock, root=repo)
    return written, lock


def is_usable_lock(payload: dict[str, Any]) -> bool:
    if int(payload.get("entry_count", -1)) != EXPECTED_RUN_COUNT:
        return False
    if int(payload.get("corrected_entry_count", -1)) != EXPECTED_CORRECTED_COUNT:
        return False
    if payload.get("corrected_run_ids") != [CORRECTED_RUN_ID]:
        return False
    if payload.get("source_only") is not True or payload.get("target_access") is not False:
        return False
    body = {key: value for key, value in payload.items() if key != "lock_identity"}
    return cc.sha256_bytes(cc.canonical_json_bytes(body)) == payload.get("lock_identity")


def load_calibration_authority_lock_if_usable(root: Path | None = None) -> dict[str, Any] | None:
    repo = _repo_root(root)
    path = repo / LOCK_RELATIVE_PATH
    if not path.is_file():
        return None
    try:
        payload = cc.read_json(path)
    except (OSError, ValueError):
        return None
    return payload if is_usable_lock(payload) else None


def require_valid_calibration_authority_for_g7(root: Path | None = None) -> dict[str, Any]:
    """The G7 preflight gate: refuses (raises) unless a valid, self-consistent
    15-row calibration authority lock (exactly 1 correction) is present."""
    lock = load_calibration_authority_lock_if_usable(root)
    if lock is None:
        raise CalibrationAuthorityError(
            "no valid 15-row E8 calibration authority lock found; G7 may not proceed. Build one "
            f"via write_calibration_authority_lock() first (expected at {LOCK_RELATIVE_PATH})."
        )
    return lock
