"""POST_FAILURE_EXPLORATORY_TARGET_V1 — Phase E1, blind target prediction.

**NOT the original C9-C13 confirmatory path. NOT a C9 PASS. NOT a
reliability waiver.** `synthetic_vs_real_spoof_probe` has already,
permanently, FAILED (`C9_DETECTOR_BA_SEP_OPTION1_V2`); `DETECTOR_RELIABILITY_LOCK_C`
remains `overall=FAILED`; the original C9 confirmatory path remains BLOCKED.
This module implements a SEPARATE, EXPLORATORY, POST-FAILURE branch that
evaluates the already-frozen C8 P3 matrix against the external SiW-Mv2
target domain, in a namespace fully disjoint from C9-C13 and from the
legacy M10 confirmatory code path's own artifacts.

This module is Phase E1 ONLY: blind, label-free target prediction. It
resolves the frozen 24-row P3-ready C8 matrix
(`SourceRow.target_prediction_required`), each row's real checkpoint and
source-dev calibration (reusing `source_evidence.load_row_evidence` and
`synthetic_real_probe.construct_row_trainer` verbatim — both are track-
agnostic and were never limited to Track G), and drives real, label-free
inference through `target_prediction.target_batches`/`predict_target`
(reused verbatim from the legacy M10 G7 implementation). It NEVER imports
or calls anything that can resolve a target label — `scoring.py`'s
`EvaluationLabels`/`load_evaluation_labels` are Phase E2 (the separate
scorer module) only.

Nothing in this module can set `c9_may_close: true`, alter
`DETECTOR_RELIABILITY_LOCK_C.json`, or write into any canonical C9-C13
artifact namespace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

EXIT_PASS, EXIT_BLOCKED, EXIT_USAGE = 0, 2, 3

DIAGNOSTICS_DIR = "reports/full/exploratory_target_v1"
RUN_DIR = "runs/exploratory_target_v1"
PROTOCOL_CONFIG_PATH = "configs/evaluation/post_failure_exploratory_target_v1.yaml"

PREDICTION_PLAN_BINDING_PATH = f"{DIAGNOSTICS_DIR}/PREDICTION_PLAN_BINDING.json"
PREDICTION_LOCK_PATH = f"{DIAGNOSTICS_DIR}/TARGET_PREDICTION_LOCK.json"

C8_ACCEPTANCE_PATH = "reports/full/c8/C8_ACCEPTANCE.json"

EXPECTED_TOTAL_ROWS = 24
EXPECTED_TRACK_G_ROWS = 15
EXPECTED_TRACK_R_ROWS = 9
TRACK_G_ARMS: tuple[str, ...] = ("RND", "DET", "LLM")
TRACK_R_EXPERIMENTS: tuple[str, ...] = ("C-R-DET", "C-R-LLM", "C-R-NOPROMPT")


class ExploratoryTargetError(RuntimeError):
    """The exploratory target protocol cannot proceed with the inputs given."""


# ==============================================================================
# 1. Protocol
# ==============================================================================

def load_protocol(repo: Path) -> dict[str, Any]:
    """The frozen exploratory protocol, or a refusal naming why it is absent.
    Never invents a value, and never reads `configs/evaluation/m10_target.yaml`
    — that file is legacy contract prose only, never this protocol's state."""
    import yaml

    path = Path(repo) / PROTOCOL_CONFIG_PATH
    if not path.is_file():
        raise ExploratoryTargetError(
            f"POST_FAILURE_EXPLORATORY_TARGET_V1 is not frozen (expected "
            f"{PROTOCOL_CONFIG_PATH} to exist and declare status: FROZEN_NOT_RUN)")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, Exception) as error:                # noqa: BLE001
        raise ExploratoryTargetError(f"{PROTOCOL_CONFIG_PATH} did not parse: {error}") from error
    if not isinstance(payload, dict) or payload.get("status") != "FROZEN_NOT_RUN":
        raise ExploratoryTargetError(f"{PROTOCOL_CONFIG_PATH} does not declare status: FROZEN_NOT_RUN")
    if payload.get("target_labels_revealed") is not False or payload.get("target_labels_opened") is not False:
        raise ExploratoryTargetError(
            "the exploratory protocol must declare target_labels_revealed: false and "
            "target_labels_opened: false; refusing to load a protocol that starts opened")
    return payload


_PROTOCOL_IDENTITY_EXCLUDED_KEYS = frozenset({
    "frozen_on", "approved_by", "status", "schema_version", "decision_id", "document_kind",
})


def protocol_identity(protocol: Mapping[str, Any]) -> str:
    material = {key: value for key, value in protocol.items()
               if key not in _PROTOCOL_IDENTITY_EXCLUDED_KEYS}
    return hashlib.sha256(json.dumps(
        material, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def active_protocol_identity(repo: Path) -> str:
    return protocol_identity(load_protocol(repo))


# ==============================================================================
# 2. The 24-row target matrix (section 5) — reuses source_matrix verbatim
# ==============================================================================

def resolve_target_matrix(repo: Path) -> list[Any]:
    """Every `SourceRow` the frozen C8 matrix marks
    `target_prediction_required` — never a hand-picked or renamed subset.
    Fails closed unless the count and track/arm breakdown match the frozen
    expectation exactly."""
    from prism_fas.evaluation.source_matrix import build_plan

    plan = build_plan()
    rows = [row for row in plan.rows if row.target_prediction_required]
    if len(rows) != EXPECTED_TOTAL_ROWS:
        raise ExploratoryTargetError(
            f"expected {EXPECTED_TOTAL_ROWS} target_prediction_required rows, found "
            f"{len(rows)}; the C8 source matrix plan may have drifted")
    track_g = [row for row in rows if row.track == "G"]
    track_r = [row for row in rows if row.track == "R"]
    if len(track_g) != EXPECTED_TRACK_G_ROWS:
        raise ExploratoryTargetError(
            f"expected {EXPECTED_TRACK_G_ROWS} Track-G target rows, found {len(track_g)}")
    if len(track_r) != EXPECTED_TRACK_R_ROWS:
        raise ExploratoryTargetError(
            f"expected {EXPECTED_TRACK_R_ROWS} Track-R target rows, found {len(track_r)}")
    for arm in TRACK_G_ARMS:
        count = sum(1 for row in track_g if row.arm == arm)
        if count != 5:
            raise ExploratoryTargetError(f"Track-G arm {arm!r} has {count} target rows, expected 5")
    for experiment in TRACK_R_EXPERIMENTS:
        count = sum(1 for row in track_r if row.experiment_id == experiment)
        if count != 3:
            raise ExploratoryTargetError(
                f"Track-R experiment {experiment!r} has {count} target rows, expected 3")
    return sorted(rows, key=lambda row: row.row_id)


def target_matrix_identity(rows: Sequence[Any]) -> str:
    """Identity over the 24-row subset alone — deliberately distinct from
    `source_matrix.matrix_identity`, which covers the full 42-row plan."""
    return hashlib.sha256(json.dumps(
        [row.as_dict() for row in rows], sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def bind_c8_matrix_identity(repo: Path) -> dict[str, Any]:
    """Reused verbatim from the diagnostics V2 correction — the real
    `source_matrix.build_plan().identity`, cross-checked against
    `C8_ACCEPTANCE.json`. Never a second implementation."""
    from prism_fas.evaluation.post_failure_diagnostics_v2 import bind_c8_matrix_identity as _bind

    try:
        return _bind(repo)
    except Exception as error:                            # noqa: BLE001
        raise ExploratoryTargetError(str(error)) from error


# ==============================================================================
# 3. Per-row checkpoint + source-dev calibration binding (section 5, 7)
# ==============================================================================

def resolve_row_binding(repo: Path, row: Any) -> dict[str, Any]:
    """One row's real checkpoint path/hash and source-dev calibration
    (threshold, temperature), read from its own `run_manifest.json` and
    `calibration.json` — the exact files C8 itself wrote. Fails closed on
    any missing field, a calibration split other than `source_dev`, or a
    manifest that ever resolved a target label."""
    from prism_fas.evaluation import source_evidence

    directory = source_evidence.row_directory(Path(repo) / source_evidence.C8_RUNS, row)
    manifest_path = directory / source_evidence.RUN_MANIFEST
    if not manifest_path.is_file():
        raise ExploratoryTargetError(f"{row.row_id}: run_manifest.json is absent at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("fixture_backed") is not False:
        raise ExploratoryTargetError(f"{row.row_id}: manifest does not declare fixture_backed=false")
    if manifest.get("status") != "PASS":
        raise ExploratoryTargetError(f"{row.row_id}: manifest status is {manifest.get('status')!r}, not PASS")
    if int(manifest.get("target_labels_resolved", -1)) != 0:
        raise ExploratoryTargetError(f"{row.row_id}: manifest recorded target_labels_resolved != 0")

    checkpoint = dict(manifest.get("checkpoint") or {})
    if not checkpoint.get("sha256") or not checkpoint.get("path") or not checkpoint.get("kind"):
        raise ExploratoryTargetError(f"{row.row_id}: checkpoint identity incomplete in manifest")

    calibration_path = directory / "calibration.json"
    if not calibration_path.is_file():
        raise ExploratoryTargetError(f"{row.row_id}: calibration.json is absent at {calibration_path}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("split") != "source_dev":
        raise ExploratoryTargetError(
            f"{row.row_id}: calibration split is {calibration.get('split')!r}, not 'source_dev'; "
            "the exploratory branch may never bind a target-derived threshold")
    if not calibration.get("calibration_hash"):
        raise ExploratoryTargetError(f"{row.row_id}: calibration_hash is absent")
    if "threshold" not in calibration:
        raise ExploratoryTargetError(f"{row.row_id}: calibration.json carries no threshold")

    return {
        "row_id": row.row_id, "experiment_id": row.experiment_id, "track": row.track,
        "arm": row.arm, "protocol": row.protocol, "seed": row.seed,
        "config_identity": row.config_identity, "run_identity": manifest["run_identity"],
        "checkpoint_sha256": checkpoint["sha256"], "checkpoint_relative_path": checkpoint["path"],
        "checkpoint_kind": checkpoint["kind"],
        "decision_logit_name": str(manifest.get("decision_logit_name", "")),
        "decision_score_name": str(manifest.get("decision_score_name", "")),
        "decision_graph_hash": str(manifest.get("decision_graph_hash", "")),
        "calibration_hash": calibration["calibration_hash"],
        "calibration_split": calibration["split"],
        "threshold": float(calibration["threshold"]),
        "temperature": (None if calibration.get("temperature") is None
                        else float(calibration["temperature"])),
        "flags": dict(row.flags),
    }


def resolve_all_row_bindings(repo: Path, rows: Sequence[Any]) -> dict[str, dict[str, Any]]:
    """Every one of the 24 rows' bindings, or a fail-closed report naming
    every row that could not resolve — never a silent partial matrix."""
    bindings: dict[str, dict[str, Any]] = {}
    problems: list[dict[str, Any]] = []
    for row in rows:
        try:
            bindings[row.row_id] = resolve_row_binding(repo, row)
        except ExploratoryTargetError as error:
            problems.append({"row_id": row.row_id, "problem": str(error)})
    if problems:
        raise ExploratoryTargetError(
            f"{len(problems)}/{len(rows)} target rows failed to bind: {problems}")
    return bindings


# ==============================================================================
# 4. Target feature package identity — verified WITHOUT opening labels
# ==============================================================================

def compute_target_feature_package_identity(root: Path) -> str:
    """Sorted `(relative_path, sha256)` pairs over every file under the
    package root, hashed. The same algorithm
    `prism_fas.pipeline.adapters.c10._package_identity` uses (independently
    implemented here since that function is private and never imported)."""
    root = Path(root)
    entries = sorted((path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest())
                     for path in root.rglob("*") if path.is_file())
    return hashlib.sha256(json.dumps(
        entries, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def verify_target_feature_package_expected(repo: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Compares the REAL package identity to the frozen expected one, WITHOUT
    ever opening `target_label_root`. Never fabricates a verified identity
    for a package that is not present on this host."""
    declared = protocol["target_feature_package"]
    root = Path(repo) / declared["target_feature_root"]
    expected = str(declared["expected_identity"])
    if not root.is_dir():
        return {"present_on_this_host": False, "verified": False,
               "expected_identity": expected, "computed_identity": None,
               "reason": "NOT_PRESENT_ON_THIS_HOST"}
    computed = compute_target_feature_package_identity(root)
    if computed != expected:
        raise ExploratoryTargetError(
            f"target feature package identity mismatch: expected {expected!r}, computed {computed!r}; "
            "fail closed rather than bind a drifted target package")
    return {"present_on_this_host": True, "verified": True,
           "expected_identity": expected, "computed_identity": computed, "reason": ""}


def verify_target_label_root_sealed(repo: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Structural proof the label root is declared-but-forbidden to `G7`
    (this module's own stage), reusing `TargetLabelFirewall` verbatim."""
    firewall = build_firewall(repo, protocol)
    proof = firewall.assert_cannot_resolve_labels("G7")
    return proof


def build_firewall(repo: Path, protocol: Mapping[str, Any]) -> Any:
    from prism_fas.evaluation.firewall import FirewallConfig

    roots = {name: Path(str(protocol["roots"][name])) for name in
            ("source_package_root", "target_feature_root", "target_label_root", "prediction_root")}
    config = FirewallConfig(
        roots=roots, permissions={stage: dict(protocol["permissions"][stage])
                                  for stage in ("TRAIN", "G7", "G8")},
        g8_forbidden_write_patterns=tuple(protocol.get("g8_forbidden_write_patterns") or ())).validate()
    from prism_fas.evaluation.firewall import TargetLabelFirewall

    return TargetLabelFirewall(config=config, project_root=Path(repo))


# ==============================================================================
# 5. The prediction-plan binding — zero scientific metric, zero target access
# ==============================================================================

def build_prediction_plan_binding(repo: Path) -> dict[str, Any]:
    protocol = load_protocol(repo)
    protocol_id = protocol_identity(protocol)
    rows = resolve_target_matrix(repo)
    matrix_id = target_matrix_identity(rows)
    c8_matrix = bind_c8_matrix_identity(repo)
    row_bindings = resolve_all_row_bindings(repo, rows)
    package_check = verify_target_feature_package_expected(repo, protocol)
    label_seal = verify_target_label_root_sealed(repo, protocol)

    binding = {
        "schema_version": "post-failure-exploratory-target-v1-prediction-plan-binding-v1",
        "protocol_identity": protocol_id,
        "target_matrix_identity": matrix_id,
        "c8_matrix_identity": c8_matrix["c8_matrix_identity"],
        "row_count": len(rows),
        "rows": {row_id: {k: v for k, v in binding.items() if k != "flags"}
                for row_id, binding in sorted(row_bindings.items())},
        "target_feature_package": package_check,
        "target_label_root_seal": label_seal,
        "immutable_upstream_state": dict(protocol["immutable_upstream_state"]),
        "target_labels_opened": False,
        "target_access": 0,
    }
    return binding


# ==============================================================================
# 6. Phase E1 real inference driver — reused verbatim from target_prediction.py
# ==============================================================================

def _checkpoint_binding_for_row(binding: Mapping[str, Any]) -> Any:
    from prism_fas.evaluation.synthetic_real_probe import CheckpointBinding

    return CheckpointBinding(
        arm=str(binding["arm"]), seed=int(binding["seed"]), row_id=str(binding["row_id"]),
        run_identity=str(binding["run_identity"]), config_identity=str(binding["config_identity"]),
        checkpoint_sha256=str(binding["checkpoint_sha256"]),
        checkpoint_path=str(binding["checkpoint_relative_path"]),
        checkpoint_kind=str(binding["checkpoint_kind"]),
        decision_logit_name=str(binding["decision_logit_name"]),
        decision_graph_hash=str(binding["decision_graph_hash"]))


def predict_one_row(repo: Path, binding: Mapping[str, Any], *, firewall: Any) -> dict[str, Any]:
    """Real, label-free target inference for ONE row — reuses
    `synthetic_real_probe.construct_row_trainer` (track-agnostic; verified
    against both Track G and Track R rows), then
    `target_prediction.target_batches`/`predict_target`/`write_predictions`/
    `build_prediction_lock` verbatim. Nothing here can resolve a target
    label: `TargetInferenceBatch` has no label field, and this function
    imports nothing from `scoring.py`.
    """
    from prism_fas.evaluation.synthetic_real_probe import construct_row_trainer
    from prism_fas.evaluation.target_prediction import (PREDICTION_LOCK_FILE, VariantCapabilities,
                                                         build_prediction_lock,
                                                         inference_config_hash,
                                                         predict_target, target_batches,
                                                         write_predictions, write_prediction_lock)

    checkpoint_binding = _checkpoint_binding_for_row(binding)
    trainer = construct_row_trainer(repo, checkpoint_binding)
    capabilities = VariantCapabilities.from_flags(binding["flags"])

    package_root = Path(repo) / "data/processed/prism_target_eval_v2"
    batches = target_batches(package_root, trainer.loader_config, cache_root=trainer.cache_root,
                             firewall=firewall)

    config_hash = inference_config_hash(
        variant=binding["arm"], flags=binding["flags"], threshold=binding["threshold"],
        unknown_threshold=None, temperature=binding["temperature"],
        package_identity=binding.get("target_feature_package_identity", ""),
        architecture_identity=trainer.model.architecture_identity())

    rows = predict_target(
        trainer.model, batches, capabilities=capabilities, threshold=binding["threshold"],
        unknown_threshold=None, temperature=binding["temperature"],
        checkpoint_hash=binding["checkpoint_sha256"], calibration_hash=binding["calibration_hash"],
        inference_config_hash=config_hash, variant=binding["arm"], device=trainer.device)

    prediction_path = (Path(repo) / RUN_DIR / binding["row_id"] / "target_predictions.parquet")
    write_predictions(prediction_path, rows, variant=binding["arm"], firewall=firewall)

    lock = build_prediction_lock(
        experiment_id=binding["experiment_id"], variant=binding["arm"], seed=binding["seed"],
        rows=rows, checkpoint_sha256=binding["checkpoint_sha256"],
        source_calibration_sha256=binding["calibration_hash"], calibration_hash=binding["calibration_hash"],
        inference_config_hash=config_hash,
        target_feature_package_identity=binding.get("target_feature_package_identity", ""),
        target_package_id="prism_target_eval_v2", threshold=binding["threshold"],
        unknown_threshold=None, scientific_config_hash=binding["config_identity"],
        source_matrix_lock_identity=binding["run_identity"])
    write_prediction_lock(prediction_path.parent / PREDICTION_LOCK_FILE, lock)
    return {"row_id": binding["row_id"], "prediction_path": str(prediction_path),
           "row_count": len(rows), "lock": lock}


def build_overall_prediction_lock(repo: Path, *, protocol_id: str, matrix_id: str,
                                  package_identity: str, per_row_locks: Sequence[dict[str, Any]]
                                  ) -> dict[str, Any]:
    """`TARGET_PREDICTION_LOCK.json` — reuses
    `target_prediction.build_lockset` verbatim. `registry_identity` binds
    THIS protocol's own identity (the exploratory branch has no separate
    M10 experiment registry) — an intentional, documented semantic
    adaptation of the legacy parameter, not a second implementation."""
    from prism_fas.evaluation.target_prediction import build_lockset

    return build_lockset(per_row_locks, matrix_identity=matrix_id, registry_identity=protocol_id,
                         target_feature_package_identity=package_identity)


# ==============================================================================
# 7. CLI
# ==============================================================================

def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _preflight(repo: Path) -> tuple[int, dict[str, Any]]:
    report: dict[str, Any] = {
        "protocol_resolved": False, "protocol_identity": None,
        "matrix_resolved": False, "matrix_error": "", "row_count": None,
        "rows_bindable": False, "rows_binding_error": "",
        "c8_matrix_identity_resolvable": False, "c8_matrix_identity_error": "",
        "target_feature_package": None, "target_label_root_sealed": None,
        "checkpoint_weights_loaded": False, "images_forwarded": False,
        "target_access": 0,
    }
    try:
        protocol = load_protocol(repo)
        report["protocol_resolved"] = True
        report["protocol_identity"] = protocol_identity(protocol)
    except ExploratoryTargetError as error:
        report["protocol_error"] = str(error)
        return EXIT_BLOCKED, report

    try:
        rows = resolve_target_matrix(repo)
        report["matrix_resolved"] = True
        report["row_count"] = len(rows)
    except ExploratoryTargetError as error:
        report["matrix_error"] = str(error)
        return EXIT_BLOCKED, report

    try:
        resolve_all_row_bindings(repo, rows)
        report["rows_bindable"] = True
    except ExploratoryTargetError as error:
        report["rows_binding_error"] = str(error)

    try:
        c8_matrix = bind_c8_matrix_identity(repo)
        report["c8_matrix_identity_resolvable"] = True
        report["c8_matrix_identity"] = c8_matrix["c8_matrix_identity"]
    except ExploratoryTargetError as error:
        report["c8_matrix_identity_error"] = str(error)

    try:
        report["target_feature_package"] = verify_target_feature_package_expected(repo, protocol)
    except ExploratoryTargetError as error:
        report["target_feature_package"] = {"verified": False, "error": str(error)}

    try:
        report["target_label_root_sealed"] = verify_target_label_root_sealed(repo, protocol)
    except Exception as error:                            # noqa: BLE001
        report["target_label_root_sealed"] = {"error": f"{type(error).__name__}: {error}"}

    report["ready_for_bind"] = bool(
        report["rows_bindable"] and report["c8_matrix_identity_resolvable"]
        and isinstance(report["target_label_root_sealed"], dict)
        and report["target_label_root_sealed"].get("target_labels_opened") is False)
    exit_code = EXIT_PASS if report["ready_for_bind"] else EXIT_BLOCKED
    return exit_code, report


def _bind_prediction_plan(repo: Path) -> tuple[int, dict[str, Any]]:
    from prism_fas.pipeline.state import atomic_write_json

    report: dict[str, Any] = {"bound": False, "reused": False, "target_access": 0}
    try:
        binding = build_prediction_plan_binding(repo)
    except ExploratoryTargetError as error:
        report["error"] = str(error)
        return EXIT_BLOCKED, report

    path = Path(repo) / PREDICTION_PLAN_BINDING_PATH
    existing = _read_json(path)
    if existing is not None:
        if existing != binding:
            report["error"] = ("an existing prediction plan binding differs from the one just "
                               "resolved; refusing to silently overwrite a prior preregistration")
            return EXIT_BLOCKED, report
        report.update({"bound": True, "reused": True, "protocol_identity": binding["protocol_identity"],
                      "target_matrix_identity": binding["target_matrix_identity"], "row_count": binding["row_count"]})
        return EXIT_PASS, report

    atomic_write_json(path, binding)
    report.update({"bound": True, "reused": False, "protocol_identity": binding["protocol_identity"],
                  "target_matrix_identity": binding["target_matrix_identity"], "row_count": binding["row_count"],
                  "binding_path": PREDICTION_PLAN_BINDING_PATH})
    return EXIT_PASS, report


def _status(repo: Path) -> tuple[int, dict[str, Any]]:
    binding = _read_json(Path(repo) / PREDICTION_PLAN_BINDING_PATH)
    lock = _read_json(Path(repo) / PREDICTION_LOCK_PATH)
    report: dict[str, Any] = {
        "prediction_plan_bound": binding is not None,
        "prediction_lock_exists": lock is not None,
        "prediction_lock_status": (lock or {}).get("status"),
        "target_labels_opened": False, "target_access": 0,
    }
    if binding is None:
        report["reason"] = "NO_PREDICTION_PLAN_BINDING"
        return EXIT_BLOCKED, report
    if lock is None:
        report["reason"] = "NO_PREDICTION_LOCK_YET"
        return EXIT_BLOCKED, report
    return EXIT_PASS, report


def _predict(repo: Path) -> tuple[int, dict[str, Any]]:
    """Phase E1 execution. NEVER invoked on this laptop for real — no genuine
    C8 checkpoint or target feature package exists here. No-rerun: a
    complete lock re-reports; a partial prediction set BLOCKS."""
    from prism_fas.pipeline.state import atomic_write_json

    report: dict[str, Any] = {"executed": False, "target_access": 0}
    lock_path = Path(repo) / PREDICTION_LOCK_PATH
    existing_lock = _read_json(lock_path)
    if existing_lock is not None:
        report.update({"executed": True, "reused_existing_lock": True,
                      "checkpoint_weights_loaded": False, "images_forwarded": False,
                      "lock_status": existing_lock.get("status")})
        return EXIT_PASS, report

    binding_path = Path(repo) / PREDICTION_PLAN_BINDING_PATH
    binding = _read_json(binding_path)
    if binding is None:
        report["error"] = "no prediction plan binding on disk; run --bind-prediction-plan first"
        return EXIT_BLOCKED, report

    run_root = Path(repo) / RUN_DIR
    partial = [p for p in run_root.glob("*/target_predictions.parquet")] if run_root.is_dir() else []
    if partial and len(partial) != binding["row_count"]:
        report["error"] = "PARTIAL_SCIENTIFIC_RESULT_SET"
        report["present"] = len(partial)
        report["expected"] = binding["row_count"]
        return EXIT_BLOCKED, report

    try:
        protocol = load_protocol(repo)
        firewall = build_firewall(repo, protocol)
        rows = resolve_target_matrix(repo)
        row_bindings = resolve_all_row_bindings(repo, rows)
        per_row_locks = []
        for row_id in sorted(row_bindings):
            result = predict_one_row(repo, row_bindings[row_id], firewall=firewall)
            per_row_locks.append(result["lock"])
        package_check = verify_target_feature_package_expected(repo, protocol)
        overall_lock = build_overall_prediction_lock(
            repo, protocol_id=binding["protocol_identity"], matrix_id=binding["target_matrix_identity"],
            package_identity=str(package_check.get("computed_identity") or ""),
            per_row_locks=per_row_locks)
    except Exception as error:                            # noqa: BLE001
        report["error"] = f"{type(error).__name__}: {error}"
        return EXIT_BLOCKED, report

    atomic_write_json(lock_path, overall_lock)
    report.update({"executed": True, "reused_existing_lock": False,
                  "row_count": len(per_row_locks), "lock_path": PREDICTION_LOCK_PATH,
                  "target_labels_opened": False})
    return EXIT_PASS, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m prism_fas.evaluation.post_failure_exploratory_target",
        description="POST_FAILURE_EXPLORATORY_TARGET_V1 — Phase E1 blind target "
                    "prediction. Never a C9 pass path.")
    parser.add_argument("--repo", default=".", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--bind-prediction-plan", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--predict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.preflight_only:
        exit_code, payload = _preflight(args.repo)
    elif args.bind_prediction_plan:
        exit_code, payload = _bind_prediction_plan(args.repo)
    elif args.status:
        exit_code, payload = _status(args.repo)
    else:
        exit_code, payload = _predict(args.repo)

    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "DIAGNOSTICS_DIR", "RUN_DIR", "PROTOCOL_CONFIG_PATH", "PREDICTION_PLAN_BINDING_PATH",
    "PREDICTION_LOCK_PATH", "C8_ACCEPTANCE_PATH", "EXPECTED_TOTAL_ROWS",
    "EXPECTED_TRACK_G_ROWS", "EXPECTED_TRACK_R_ROWS", "TRACK_G_ARMS", "TRACK_R_EXPERIMENTS",
    "ExploratoryTargetError", "load_protocol", "protocol_identity", "active_protocol_identity",
    "resolve_target_matrix", "target_matrix_identity", "bind_c8_matrix_identity",
    "resolve_row_binding", "resolve_all_row_bindings",
    "compute_target_feature_package_identity", "verify_target_feature_package_expected",
    "verify_target_label_root_sealed", "build_firewall", "build_prediction_plan_binding",
    "predict_one_row", "build_overall_prediction_lock",
    "EXIT_PASS", "EXIT_BLOCKED", "EXIT_USAGE",
]
