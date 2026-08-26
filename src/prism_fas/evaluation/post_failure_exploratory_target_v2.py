"""POST_FAILURE_EXPLORATORY_TARGET_V2 — Phase E1, pre-target scientific and
execution correction of V1.

V1 (`configs/evaluation/post_failure_exploratory_target_v1.yaml`, identity
`8fb806d25a80ecd3c7d44cfeba8c893a5f115b8b51797220a51132ba16708b51`) was
never scientifically executed: no SiW-Mv2 prediction has run, no target
label has been opened. This module corrects six pre-target defects found by
audit:

  A. `--predict` re-resolved scientific inputs live instead of trusting the
     frozen `PREDICTION_PLAN_BINDING.json` — corrected: the binding is now
     the sole authoritative execution source, and a read-only
     recomputation must match it EXACTLY before `--predict` proceeds.
  B. `--bind-prediction-plan` could succeed with an unverified target
     feature package — corrected: binding now REQUIRES
     `present_on_this_host`/`verified`/identity-match before writing.
  C. The legacy `target_prediction.build_lockset` rejects duplicate
     `experiment_id` values, incompatible with 3-5 seeded rows per arm —
     corrected: a new, `row_id`-keyed V2 lockset is used; the legacy
     function is left completely unchanged.
  D. `variant = row.arm` collapsed `C-R-LLM` and `C-R-NOPROMPT` (both
     `arm=LLM`) into the same identifier — corrected:
     `prediction_variant_id = row.experiment_id`.
  E. An existing `TARGET_PREDICTION_LOCK.json` was re-reported without
     validation — corrected:
     `validate_existing_exploratory_prediction_result`.
  F. Partial-result detection only counted prediction files — corrected:
     checks both prediction files and per-row locks, and the
     lock-without-rows / rows-without-lock cases.
  (Defect G, the hard-coded target feature root, is also corrected here:
  `predict_one_row` now reads the root from the frozen binding.)

This module reuses V1's own row/matrix/package/firewall resolution
functions VERBATIM (imported, never duplicated) — those were not found
defective — and the entire legacy M10 per-row prediction machinery
(`target_prediction.py`) verbatim. `synthetic_real_probe.construct_row_trainer`
remains track-agnostic and unchanged. Nothing here can resolve a target
label — Phase E2 (`post_failure_exploratory_target_v2_scorer.py`) is
separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from prism_fas.evaluation import post_failure_exploratory_target as v1
from prism_fas.evaluation.contracts import stable_identity

EXIT_PASS, EXIT_BLOCKED, EXIT_USAGE = 0, 2, 3

DIAGNOSTICS_DIR = "reports/full/exploratory_target_v2"
RUN_DIR = "runs/exploratory_target_v2"
PROTOCOL_CONFIG_PATH = "configs/evaluation/post_failure_exploratory_target_v2.yaml"

PREDICTION_PLAN_BINDING_PATH = f"{DIAGNOSTICS_DIR}/PREDICTION_PLAN_BINDING.json"
PREDICTION_LOCK_PATH = f"{DIAGNOSTICS_DIR}/TARGET_PREDICTION_LOCK.json"

EXPECTED_TOTAL_ROWS = 24
BINDING_REQUIRED_ROW_FIELDS: tuple[str, ...] = (
    "row_id", "experiment_id", "track", "arm", "protocol", "seed", "flags",
    "config_identity", "run_identity", "checkpoint_relative_path", "checkpoint_sha256",
    "checkpoint_kind", "decision_logit_name", "decision_score_name", "decision_graph_hash",
    "calibration_hash", "calibration_split", "threshold", "temperature", "prediction_variant_id",
)


class ExploratoryTargetV2Error(RuntimeError):
    """The V2 exploratory target protocol cannot proceed with the inputs given."""


# ==============================================================================
# 1. Protocol
# ==============================================================================

def load_protocol(repo: Path) -> dict[str, Any]:
    """The frozen V2 protocol. Never reads V1's config or the legacy
    `m10_target.yaml` as its own state."""
    import yaml

    path = Path(repo) / PROTOCOL_CONFIG_PATH
    if not path.is_file():
        raise ExploratoryTargetV2Error(
            f"POST_FAILURE_EXPLORATORY_TARGET_V2 is not frozen (expected {PROTOCOL_CONFIG_PATH} "
            "to exist and declare status: FROZEN_NOT_RUN)")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, Exception) as error:                # noqa: BLE001
        raise ExploratoryTargetV2Error(f"{PROTOCOL_CONFIG_PATH} did not parse: {error}") from error
    if not isinstance(payload, dict) or payload.get("status") != "FROZEN_NOT_RUN":
        raise ExploratoryTargetV2Error(f"{PROTOCOL_CONFIG_PATH} does not declare status: FROZEN_NOT_RUN")
    if payload.get("target_labels_revealed") is not False or payload.get("target_labels_opened") is not False:
        raise ExploratoryTargetV2Error(
            "the V2 protocol must declare target_labels_revealed: false and "
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
# 2. Row bindings — reused verbatim from V1, plus prediction_variant_id
#    (Defect D) and full flags retention (Defect A's field list)
# ==============================================================================

def resolve_all_row_bindings_v2(repo: Path, rows: Sequence[Any]) -> dict[str, dict[str, Any]]:
    """V1's `resolve_row_binding`/`resolve_all_row_bindings`, reused
    verbatim, then annotated with `prediction_variant_id = experiment_id`
    (Defect D) — `arm` is kept as a separate field, never overloaded as the
    variant identity."""
    bindings = v1.resolve_all_row_bindings(repo, rows)
    for row_id, binding in bindings.items():
        binding["prediction_variant_id"] = binding["experiment_id"]
    return bindings


# ==============================================================================
# 3. Target feature package — verified BEFORE bind is allowed (Defect B)
# ==============================================================================

def verify_target_feature_package_required(repo: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Unlike `--preflight-only` (which may report the package absent or
    unverified), `--bind-prediction-plan` requires
    `present_on_this_host AND verified AND computed_identity == expected`,
    else it fails closed. Package-identity verification never opens
    `target_label_root` — `target_label_access` is recorded separately and
    is always 0 here."""
    check = v1.verify_target_feature_package_expected(repo, protocol)
    if not (check.get("present_on_this_host") and check.get("verified")):
        raise ExploratoryTargetV2Error(
            f"target feature package is not present-and-verified on this host "
            f"({check}); --bind-prediction-plan requires verification before binding, "
            "per the corrected V2 protocol")
    return {**check, "target_feature_package_identity_verified": True, "target_label_access": 0}


# ==============================================================================
# 4. The prediction-plan binding — the sole authoritative execution source
# ==============================================================================

def build_prediction_plan_binding(repo: Path) -> dict[str, Any]:
    protocol = load_protocol(repo)
    protocol_id = protocol_identity(protocol)
    rows = v1.resolve_target_matrix(repo)
    matrix_id = v1.target_matrix_identity(rows)
    c8_matrix = v1.bind_c8_matrix_identity(repo)
    row_bindings = resolve_all_row_bindings_v2(repo, rows)
    package_check = verify_target_feature_package_required(repo, protocol)
    label_seal = v1.verify_target_label_root_sealed(repo, protocol)

    for row_id, binding in row_bindings.items():
        missing = [field for field in BINDING_REQUIRED_ROW_FIELDS if field not in binding]
        if missing:
            raise ExploratoryTargetV2Error(f"{row_id}: binding is missing required fields {missing}")

    binding = {
        "schema_version": "post-failure-exploratory-target-v2-prediction-plan-binding-v1",
        "protocol_identity": protocol_id,
        "target_matrix_identity": matrix_id,
        "c8_matrix_identity": c8_matrix["c8_matrix_identity"],
        "row_count": len(rows),
        # Flags ARE retained here (Defect A's field list) — V1 stripped them.
        "rows": {row_id: dict(sorted(binding.items())) for row_id, binding in sorted(row_bindings.items())},
        "target_feature_package": package_check,
        "target_label_root_seal": label_seal,
        "immutable_upstream_state": dict(protocol["immutable_upstream_state"]),
        "target_labels_opened": False,
        "target_label_access": 0,
        "target_access": 0,
    }
    binding["prediction_plan_binding_identity"] = stable_identity(
        {key: value for key, value in binding.items() if key != "prediction_plan_binding_identity"})
    return binding


def verify_binding_unchanged(repo: Path, frozen_binding: Mapping[str, Any]) -> dict[str, Any]:
    """Defect A's core correction: recompute the candidate binding
    READ-ONLY and require EXACT equality with the frozen
    `PREDICTION_PLAN_BINDING.json` before `--predict` may proceed. Any
    difference — a moved checkpoint, a changed calibration, a drifted
    source matrix — BLOCKS rather than silently executing from fresh
    values."""
    fresh = build_prediction_plan_binding(repo)
    if fresh != dict(frozen_binding):
        return {"unchanged": False, "fresh_prediction_plan_binding_identity":
                fresh.get("prediction_plan_binding_identity"),
               "frozen_prediction_plan_binding_identity":
                   frozen_binding.get("prediction_plan_binding_identity")}
    return {"unchanged": True}


# ==============================================================================
# 5. Phase E1 real inference driver — path from the binding (Defect G),
#    prediction_variant_id (Defect D)
# ==============================================================================

def predict_one_row(repo: Path, binding: Mapping[str, Any], *, package_root: Path,
                    firewall: Any) -> dict[str, Any]:
    """Real, label-free target inference for ONE row, driven ENTIRELY by
    the frozen binding document — reuses `synthetic_real_probe.construct_row_trainer`
    and `target_prediction.target_batches`/`predict_target`/`write_predictions`/
    `build_prediction_lock` verbatim. `package_root` comes from the caller's
    already-verified protocol/binding root — never hard-coded here."""
    from prism_fas.evaluation.synthetic_real_probe import CheckpointBinding, construct_row_trainer
    from prism_fas.evaluation.target_prediction import (PREDICTION_LOCK_FILE, VariantCapabilities,
                                                         build_prediction_lock,
                                                         inference_config_hash,
                                                         predict_target, target_batches,
                                                         write_prediction_lock, write_predictions)

    checkpoint_binding = CheckpointBinding(
        arm=str(binding["arm"]), seed=int(binding["seed"]), row_id=str(binding["row_id"]),
        run_identity=str(binding["run_identity"]), config_identity=str(binding["config_identity"]),
        checkpoint_sha256=str(binding["checkpoint_sha256"]),
        checkpoint_path=str(binding["checkpoint_relative_path"]),
        checkpoint_kind=str(binding["checkpoint_kind"]),
        decision_logit_name=str(binding["decision_logit_name"]),
        decision_graph_hash=str(binding["decision_graph_hash"]))
    trainer = construct_row_trainer(repo, checkpoint_binding)
    capabilities = VariantCapabilities.from_flags(binding["flags"])
    variant = str(binding["prediction_variant_id"])

    batches = target_batches(Path(package_root), trainer.loader_config, cache_root=trainer.cache_root,
                             firewall=firewall)

    config_hash = inference_config_hash(
        variant=variant, flags=binding["flags"], threshold=binding["threshold"],
        unknown_threshold=None, temperature=binding["temperature"],
        package_identity=binding.get("target_feature_package_identity", ""),
        architecture_identity=trainer.model.architecture_identity())

    rows = predict_target(
        trainer.model, batches, capabilities=capabilities, threshold=binding["threshold"],
        unknown_threshold=None, temperature=binding["temperature"],
        checkpoint_hash=binding["checkpoint_sha256"], calibration_hash=binding["calibration_hash"],
        inference_config_hash=config_hash, variant=variant, device=trainer.device)

    prediction_path = Path(repo) / RUN_DIR / binding["row_id"] / "target_predictions.parquet"
    write_predictions(prediction_path, rows, variant=variant, firewall=firewall)
    prediction_file_sha256 = hashlib.sha256(prediction_path.read_bytes()).hexdigest()

    lock = build_prediction_lock(
        experiment_id=binding["experiment_id"], variant=variant, seed=binding["seed"],
        rows=rows, checkpoint_sha256=binding["checkpoint_sha256"],
        source_calibration_sha256=binding["calibration_hash"], calibration_hash=binding["calibration_hash"],
        inference_config_hash=config_hash,
        target_feature_package_identity=binding.get("target_feature_package_identity", ""),
        target_package_id="prism_target_eval_v2", threshold=binding["threshold"],
        unknown_threshold=None, scientific_config_hash=binding["config_identity"],
        source_matrix_lock_identity=binding["run_identity"])
    write_prediction_lock(prediction_path.parent / PREDICTION_LOCK_FILE, lock)
    return {"row_id": binding["row_id"], "prediction_path": str(prediction_path),
           "prediction_file_sha256": prediction_file_sha256, "row_count": len(rows), "lock": lock}


# ==============================================================================
# 6. V2 prediction lockset — row_id-keyed (Defect C); legacy build_lockset
#    is NEVER called here and remains completely unchanged
# ==============================================================================

LOCKSET_SCHEMA_VERSION = "post-failure-exploratory-target-v2-prediction-lockset-v1"


def build_v2_prediction_lockset(*, protocol_id: str, matrix_id: str, c8_matrix_id: str,
                                package_identity: str, plan_binding_identity: str,
                                row_bindings: Mapping[str, Mapping[str, Any]],
                                row_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The exploratory V2 lockset: primary key `row_id`, so 3-5 seeded rows
    sharing one `experiment_id` are all represented — the legacy
    `experiment_id`-keyed `target_prediction.build_lockset` cannot do this
    and is not used here."""
    if len(row_results) != EXPECTED_TOTAL_ROWS:
        raise ExploratoryTargetV2Error(
            f"expected exactly {EXPECTED_TOTAL_ROWS} row results, got {len(row_results)}")
    entries: dict[str, Any] = {}
    for row_id, result in row_results.items():
        binding = row_bindings[row_id]
        lock = result["lock"]
        entries[row_id] = {
            "row_id": row_id, "experiment_id": binding["experiment_id"], "track": binding["track"],
            "arm": binding["arm"], "seed": int(binding["seed"]),
            "prediction_variant_id": binding["prediction_variant_id"],
            "checkpoint_sha256": binding["checkpoint_sha256"], "calibration_hash": binding["calibration_hash"],
            "threshold": float(binding["threshold"]),
            "temperature": (None if binding["temperature"] is None else float(binding["temperature"])),
            "inference_config_hash": lock["inference_config_hash"],
            "prediction_logical_identity": lock["prediction_logical_identity"],
            "prediction_file_sha256": result["prediction_file_sha256"],
            "prediction_lock_identity": lock["prediction_lock_identity"],
            "row_count": lock["row_count"], "video_count": lock["video_count"],
        }
    body = {
        "lockset_schema_version": LOCKSET_SCHEMA_VERSION,
        "protocol_identity": protocol_id, "target_matrix_identity": matrix_id,
        "c8_matrix_identity": c8_matrix_id, "target_feature_package_identity": package_identity,
        "prediction_plan_binding_identity": plan_binding_identity,
        "entries": dict(sorted(entries.items())), "entry_count": len(entries),
        "frame_rows_total": sum(int(e["row_count"]) for e in entries.values()),
        "target_labels_opened": False,
        "ba_sep_observed_verdict": "FAIL",
        "detector_reliability_lock_c_observed_overall": "FAILED",
        "post_failure_diagnostics_v2": "FAIL",
        "c9_original_confirmatory_path": "BLOCKED",
        "exploratory_target_status": "POST_FAILURE_EXPLORATORY",
        "status": "FROZEN", "target_access": 0,
    }
    return {**body, "lockset_identity": stable_identity(body)}


# ==============================================================================
# 7. Existing-result validation (Defect E)
# ==============================================================================

def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def validate_existing_exploratory_prediction_result(repo: Path) -> dict[str, Any]:
    """Canonical, comprehensive validation of an on-disk V2
    `TARGET_PREDICTION_LOCK.json` — used by BOTH a second `--predict` call
    and `--status`. Never re-runs inference; only re-derives cheap
    identities/hashes and compares them to the recorded lockset.

    Returns `{"valid": bool, "problems": [...], "lockset": {...} | None}`.
    """
    problems: list[str] = []
    repo = Path(repo)

    binding = _read_json(repo / PREDICTION_PLAN_BINDING_PATH)
    if binding is None:
        return {"valid": False, "problems": ["no PREDICTION_PLAN_BINDING.json on disk"], "lockset": None}
    lockset = _read_json(repo / PREDICTION_LOCK_PATH)
    if lockset is None:
        return {"valid": False, "problems": ["no TARGET_PREDICTION_LOCK.json on disk"], "lockset": None}

    try:
        active_id = active_protocol_identity(repo)
    except ExploratoryTargetV2Error as error:
        return {"valid": False, "problems": [f"active protocol unresolvable: {error}"], "lockset": lockset}

    if lockset.get("status") != "FROZEN":
        problems.append(f"lockset status is {lockset.get('status')!r}, not FROZEN")
    if lockset.get("protocol_identity") != active_id:
        problems.append("lockset.protocol_identity does not match the active V2 protocol identity")
    if binding.get("protocol_identity") != active_id:
        problems.append("binding.protocol_identity does not match the active V2 protocol identity")
    if lockset.get("prediction_plan_binding_identity") != binding.get("prediction_plan_binding_identity"):
        problems.append("lockset.prediction_plan_binding_identity does not match the bound plan")
    if lockset.get("target_matrix_identity") != binding.get("target_matrix_identity"):
        problems.append("lockset.target_matrix_identity does not match the bound plan")
    if lockset.get("c8_matrix_identity") != binding.get("c8_matrix_identity"):
        problems.append("lockset.c8_matrix_identity does not match the bound plan")
    if int(lockset.get("entry_count", -1)) != EXPECTED_TOTAL_ROWS:
        problems.append(f"lockset.entry_count is {lockset.get('entry_count')}, expected {EXPECTED_TOTAL_ROWS}")
    if lockset.get("target_labels_opened") is not False:
        problems.append("lockset.target_labels_opened is not False")
    for field, expected in (("ba_sep_observed_verdict", "FAIL"),
                            ("detector_reliability_lock_c_observed_overall", "FAILED"),
                            ("post_failure_diagnostics_v2", "FAIL"),
                            ("c9_original_confirmatory_path", "BLOCKED")):
        if lockset.get(field) != expected:
            problems.append(f"lockset.{field} is not {expected!r}")

    entries = dict(lockset.get("entries") or {})
    binding_rows = dict(binding.get("rows") or {})
    if set(entries) != set(binding_rows):
        problems.append("lockset row_ids do not exactly match the bound plan's row_ids")

    from prism_fas.evaluation.target_prediction import read_predictions, validate_predictions

    for row_id, entry in sorted(entries.items()):
        row_binding = binding_rows.get(row_id)
        if row_binding is None:
            continue
        if entry.get("experiment_id") != row_binding.get("experiment_id") or \
                entry.get("track") != row_binding.get("track") or \
                entry.get("arm") != row_binding.get("arm") or \
                int(entry.get("seed", -1)) != int(row_binding.get("seed", -2)) or \
                entry.get("prediction_variant_id") != row_binding.get("prediction_variant_id"):
            problems.append(f"{row_id}: lockset entry disagrees with the bound row's identity fields")
        if entry.get("checkpoint_sha256") != row_binding.get("checkpoint_sha256"):
            problems.append(f"{row_id}: checkpoint_sha256 mismatch")
        if entry.get("calibration_hash") != row_binding.get("calibration_hash"):
            problems.append(f"{row_id}: calibration_hash mismatch")

        prediction_path = repo / RUN_DIR / row_id / "target_predictions.parquet"
        if not prediction_path.is_file():
            problems.append(f"{row_id}: target_predictions.parquet is missing")
            continue
        real_hash = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
        if real_hash != entry.get("prediction_file_sha256"):
            problems.append(f"{row_id}: prediction file sha256 no longer matches the locked value")
        try:
            rows = read_predictions(prediction_path)   # re-validates schema + forbidden columns
            validate_predictions(rows)
        except Exception as error:                        # noqa: BLE001
            problems.append(f"{row_id}: prediction file failed schema validation: {error}")

        from prism_fas.evaluation.target_prediction import PREDICTION_LOCK_FILE

        row_lock = _read_json(repo / RUN_DIR / row_id / PREDICTION_LOCK_FILE)
        if row_lock is None:
            problems.append(f"{row_id}: per-row PREDICTION_LOCK.json is missing")
        else:
            if row_lock.get("prediction_lock_identity") != entry.get("prediction_lock_identity"):
                problems.append(f"{row_id}: per-row prediction_lock_identity mismatch")
            if row_lock.get("prediction_logical_identity") != entry.get("prediction_logical_identity"):
                problems.append(f"{row_id}: per-row prediction_logical_identity mismatch")

    recomputed_identity = stable_identity(
        {key: value for key, value in lockset.items() if key != "lockset_identity"})
    if recomputed_identity != lockset.get("lockset_identity"):
        problems.append("lockset does not hash to its own recorded lockset_identity")

    return {"valid": not problems, "problems": problems, "lockset": lockset}


# ==============================================================================
# 8. CLI
# ==============================================================================

def _preflight(repo: Path) -> tuple[int, dict[str, Any]]:
    report: dict[str, Any] = {
        "protocol_resolved": False, "protocol_identity": None,
        "matrix_resolved": False, "row_count": None, "rows_bindable": False,
        "c8_matrix_identity_resolvable": False, "target_feature_package": None,
        "target_label_root_sealed": None, "checkpoint_weights_loaded": False,
        "images_forwarded": False, "target_access": 0,
    }
    try:
        protocol = load_protocol(repo)
        report["protocol_resolved"] = True
        report["protocol_identity"] = protocol_identity(protocol)
    except ExploratoryTargetV2Error as error:
        report["protocol_error"] = str(error)
        return EXIT_BLOCKED, report

    try:
        rows = v1.resolve_target_matrix(repo)
        report["matrix_resolved"] = True
        report["row_count"] = len(rows)
    except v1.ExploratoryTargetError as error:
        report["matrix_error"] = str(error)
        return EXIT_BLOCKED, report

    try:
        resolve_all_row_bindings_v2(repo, rows)
        report["rows_bindable"] = True
    except v1.ExploratoryTargetError as error:
        report["rows_binding_error"] = str(error)

    try:
        c8_matrix = v1.bind_c8_matrix_identity(repo)
        report["c8_matrix_identity_resolvable"] = True
        report["c8_matrix_identity"] = c8_matrix["c8_matrix_identity"]
    except v1.ExploratoryTargetError as error:
        report["c8_matrix_identity_error"] = str(error)

    try:
        report["target_feature_package"] = v1.verify_target_feature_package_expected(repo, protocol)
    except v1.ExploratoryTargetError as error:
        report["target_feature_package"] = {"verified": False, "error": str(error)}

    try:
        report["target_label_root_sealed"] = v1.verify_target_label_root_sealed(repo, protocol)
    except Exception as error:                            # noqa: BLE001
        report["target_label_root_sealed"] = {"error": f"{type(error).__name__}: {error}"}

    report["ready_for_bind"] = bool(
        report["rows_bindable"] and report["c8_matrix_identity_resolvable"]
        and isinstance(report["target_feature_package"], dict)
        and report["target_feature_package"].get("present_on_this_host")
        and report["target_feature_package"].get("verified")
        and isinstance(report["target_label_root_sealed"], dict)
        and report["target_label_root_sealed"].get("target_labels_opened") is False)
    exit_code = EXIT_PASS if report["ready_for_bind"] else EXIT_BLOCKED
    return exit_code, report


def _bind_prediction_plan(repo: Path) -> tuple[int, dict[str, Any]]:
    from prism_fas.pipeline.state import atomic_write_json

    report: dict[str, Any] = {"bound": False, "reused": False, "target_access": 0}
    try:
        binding = build_prediction_plan_binding(repo)
    except (ExploratoryTargetV2Error, v1.ExploratoryTargetError) as error:
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
                  "binding_path": PREDICTION_PLAN_BINDING_PATH,
                  "target_feature_package_identity_verified":
                      binding["target_feature_package"]["target_feature_package_identity_verified"],
                  "target_label_access": 0})
    return EXIT_PASS, report


def _status(repo: Path) -> tuple[int, dict[str, Any]]:
    binding = _read_json(Path(repo) / PREDICTION_PLAN_BINDING_PATH)
    lock_present = (Path(repo) / PREDICTION_LOCK_PATH).is_file()
    report: dict[str, Any] = {
        "prediction_plan_bound": binding is not None,
        "prediction_lock_exists": lock_present,
        "target_labels_opened": False, "target_access": 0,
    }
    if binding is None:
        report["reason"] = "NO_PREDICTION_PLAN_BINDING"
        return EXIT_BLOCKED, report
    if not lock_present:
        report["reason"] = "NO_PREDICTION_LOCK_YET"
        return EXIT_BLOCKED, report

    validation = validate_existing_exploratory_prediction_result(repo)
    report["existing_result_validation"] = {"valid": validation["valid"], "problems": validation["problems"]}
    if not validation["valid"]:
        report["reason"] = "EXISTING_RESULT_FAILED_VALIDATION"
        return EXIT_BLOCKED, report
    report["prediction_lock_status"] = validation["lockset"]["status"]
    return EXIT_PASS, report


def _detect_partial_state(repo: Path, expected_rows: int) -> str | None:
    """Defect F: checks BOTH prediction files and per-row locks for all
    rows, and the two asymmetric completion cases. Returns a problem
    string, or `None` if the host is either clean or fully complete."""
    from prism_fas.evaluation.target_prediction import PREDICTION_LOCK_FILE

    run_root = Path(repo) / RUN_DIR
    prediction_files = sorted(run_root.glob("*/target_predictions.parquet")) if run_root.is_dir() else []
    lock_files = sorted(run_root.glob(f"*/{PREDICTION_LOCK_FILE}")) if run_root.is_dir() else []
    overall_lock_present = (Path(repo) / PREDICTION_LOCK_PATH).is_file()

    if not prediction_files and not lock_files and not overall_lock_present:
        return None   # clean host
    if 0 < len(prediction_files) < expected_rows or 0 < len(lock_files) < expected_rows:
        return (f"PARTIAL_SCIENTIFIC_RESULT_SET: {len(prediction_files)} prediction file(s), "
               f"{len(lock_files)} per-row lock(s), expected {expected_rows} of each")
    if len(prediction_files) == expected_rows and len(lock_files) == expected_rows \
            and not overall_lock_present:
        return "all row artifacts present but the overall TARGET_PREDICTION_LOCK.json is missing"
    if overall_lock_present and (len(prediction_files) < expected_rows or len(lock_files) < expected_rows):
        return (f"overall TARGET_PREDICTION_LOCK.json exists but only {len(prediction_files)} prediction "
               f"file(s) / {len(lock_files)} per-row lock(s) of {expected_rows} exist")
    return None   # fully complete; handled by the caller's own existing-lock branch


def _predict(repo: Path) -> tuple[int, dict[str, Any]]:
    """Phase E1 execution. NEVER invoked on this laptop for real. Defect A:
    inference is driven ENTIRELY by the frozen `PREDICTION_PLAN_BINDING.json`,
    after a read-only recomputation proves it has not drifted. Defect E: an
    existing lock is validated, not merely re-reported. Defect F: partial
    state is detected from both prediction files and per-row locks."""
    from prism_fas.pipeline.state import atomic_write_json

    report: dict[str, Any] = {"executed": False, "target_access": 0}
    binding_path = Path(repo) / PREDICTION_PLAN_BINDING_PATH
    frozen_binding = _read_json(binding_path)
    if frozen_binding is None:
        report["error"] = "no prediction plan binding on disk; run --bind-prediction-plan first"
        return EXIT_BLOCKED, report

    lock_path = Path(repo) / PREDICTION_LOCK_PATH
    if lock_path.is_file():
        validation = validate_existing_exploratory_prediction_result(repo)
        if not validation["valid"]:
            report.update({"error": "EXISTING_RESULT_FAILED_VALIDATION", "problems": validation["problems"]})
            return EXIT_BLOCKED, report
        report.update({"executed": True, "reused_existing_lock": True,
                      "checkpoint_weights_loaded": False, "images_forwarded": False,
                      "prediction_recomputed": False, "lock_status": validation["lockset"]["status"]})
        return EXIT_PASS, report

    partial_problem = _detect_partial_state(repo, int(frozen_binding["row_count"]))
    if partial_problem is not None:
        report["error"] = partial_problem
        return EXIT_BLOCKED, report

    unchanged = verify_binding_unchanged(repo, frozen_binding)
    if not unchanged["unchanged"]:
        report.update({"error": "PREDICTION_PLAN_BINDING_DRIFTED", **unchanged})
        return EXIT_BLOCKED, report

    try:
        protocol = load_protocol(repo)
        firewall = v1.build_firewall(repo, protocol)
        package_root = Path(repo) / protocol["roots"]["target_feature_root"]
        row_bindings = frozen_binding["rows"]   # the FROZEN document is now authoritative
        row_results: dict[str, Any] = {}
        for row_id in sorted(row_bindings):
            row_results[row_id] = predict_one_row(repo, row_bindings[row_id],
                                                  package_root=package_root, firewall=firewall)
        lockset = build_v2_prediction_lockset(
            protocol_id=frozen_binding["protocol_identity"], matrix_id=frozen_binding["target_matrix_identity"],
            c8_matrix_id=frozen_binding["c8_matrix_identity"],
            package_identity=frozen_binding["target_feature_package"]["computed_identity"],
            plan_binding_identity=frozen_binding["prediction_plan_binding_identity"],
            row_bindings=row_bindings, row_results=row_results)
    except Exception as error:                            # noqa: BLE001
        report["error"] = f"{type(error).__name__}: {error}"
        return EXIT_BLOCKED, report

    atomic_write_json(lock_path, lockset)
    report.update({"executed": True, "reused_existing_lock": False, "row_count": len(row_results),
                  "lock_path": PREDICTION_LOCK_PATH, "target_labels_opened": False})
    return EXIT_PASS, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m prism_fas.evaluation.post_failure_exploratory_target_v2",
        description="POST_FAILURE_EXPLORATORY_TARGET_V2 — Phase E1 blind target "
                    "prediction, pre-target corrected. Never a C9 pass path.")
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
    "PREDICTION_LOCK_PATH", "EXPECTED_TOTAL_ROWS", "BINDING_REQUIRED_ROW_FIELDS",
    "LOCKSET_SCHEMA_VERSION", "ExploratoryTargetV2Error", "load_protocol", "protocol_identity",
    "active_protocol_identity", "resolve_all_row_bindings_v2",
    "verify_target_feature_package_required", "build_prediction_plan_binding",
    "verify_binding_unchanged", "predict_one_row", "build_v2_prediction_lockset",
    "validate_existing_exploratory_prediction_result",
    "EXIT_PASS", "EXIT_BLOCKED", "EXIT_USAGE",
]
