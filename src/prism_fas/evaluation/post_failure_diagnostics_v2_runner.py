"""C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2 runner CLI.

    python -m prism_fas.evaluation.post_failure_diagnostics_v2_runner --repo . --preflight-only
    python -m prism_fas.evaluation.post_failure_diagnostics_v2_runner --repo . --bind-only
    python -m prism_fas.evaluation.post_failure_diagnostics_v2_runner --repo . --status
    python -m prism_fas.evaluation.post_failure_diagnostics_v2_runner --repo . --execute

Pre-execution scientific correction of `post_failure_diagnostics_runner`
(V1). V1's four modes and fail-closed/no-rerun contract are mirrored exactly,
on a SEPARATE artifact namespace (`post_failure_diagnostics_v2.DIAGNOSTICS_DIR`),
with five corrections: (A) the benign-corruption threshold is now a disjoint
reference calibration, never self-normalizing; (B) `cross_route_synthetic` is
`NEEDS_SCIENTIFIC_DECISION`, not executed; (C) a second-or-later `--execute`
and every `--status` call run
`post_failure_diagnostics_v2.validate_existing_diagnostics_result` rather
than trusting bare verdict/per-test presence; (D) the C8 matrix identity is
a real, cross-checked binding; (E) the calibration/evaluation split is
proven group-safe per domain, not just globally.

**NOT A BA_sep REVISION. NOT A RELIABILITY-BARRIER RESCUE. NOT A C9 PASS
PATH.** No mode of this CLI reads, writes, or depends on writing to
`reports/full/c8/reliability/synthetic_vs_real_spoof_probe/`,
`reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json`, or the V1 diagnostics
namespace. Even a diagnostic that PASSES on every arm cannot make
`c9_may_close` true.

Exit codes: 0 all executed diagnostics PASS, 1 at least one executed
diagnostic FAILED (a real result, not an error), 2 BLOCKED, 3 USAGE error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EXIT_PASS, EXIT_FAIL, EXIT_BLOCKED, EXIT_USAGE = 0, 1, 2, 3

from prism_fas.evaluation.post_failure_diagnostics_v2 import (  # noqa: E402
    CHECKPOINT_BINDING_PATH, EXECUTABLE_TESTS, PER_TEST_PATH,
    POPULATION_BINDING_PATH, PROTOCOL_BINDING_PATH, PROVENANCE_PATH, RESULT_PATH,
    VERDICT_PATH)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m prism_fas.evaluation.post_failure_diagnostics_v2_runner",
        description="C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2 runner. Pre-execution "
                    "scientific correction of V1. Never a C9 pass path.")
    parser.add_argument("--repo", default=".", type=Path,
                        help="repository root (default: current directory)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--bind-only", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _ba_sep_canary(repo: Path) -> dict[str, Any]:
    """Read-only, informational: the CURRENT BA_sep protocol identity and
    reliability-lock overall. Never written by this module."""
    from prism_fas.evaluation import detector_reliability, synthetic_real_probe

    canary: dict[str, Any] = {
        "ba_sep_protocol_identity": None, "ba_sep_protocol_identity_error": "",
        "detector_reliability_lock_overall": None, "detector_reliability_lock_present": False,
    }
    try:
        canary["ba_sep_protocol_identity"] = synthetic_real_probe.protocol_identity(repo)
    except Exception as error:                        # noqa: BLE001
        canary["ba_sep_protocol_identity_error"] = f"{type(error).__name__}: {error}"
    lock = _read_json(Path(repo) / detector_reliability.LOCK_PATH)
    if lock is not None:
        canary["detector_reliability_lock_present"] = True
        canary["detector_reliability_lock_overall"] = lock.get("overall")
    return canary


# ==============================================================================
# --preflight-only
# ==============================================================================

def _preflight(repo: Path) -> tuple[int, dict[str, Any]]:
    from prism_fas.evaluation import post_failure_diagnostics_v2 as diag2
    from prism_fas.evaluation import synthetic_real_probe as probe
    from prism_fas.pipeline.adapters import sources

    report: dict[str, Any] = {
        "protocol_resolved": False, "protocol_identity": None,
        "source_inputs_resolved": False, "source_inputs_error": "",
        "checkpoints_resolved": False, "checkpoints_error": "",
        "c8_matrix_identity_resolvable": False, "c8_matrix_identity_error": "",
        "target_firewall_clean": False, "per_test_gpu_ready": {},
        "ba_metric_computed": False, "images_forwarded": False,
        "scientific_artifacts_written": False, "state_modified": False,
        "target_access": 0,
    }
    try:
        protocol = diag2.load_protocol(repo)
        report["protocol_resolved"] = True
        report["protocol_identity"] = diag2.protocol_identity(protocol)
        report["target_firewall_clean"] = int(protocol.get("target_access", -1)) == 0
        report["per_test_gpu_ready"] = {test_id: bool(cfg.get("gpu_ready"))
                                        for test_id, cfg in protocol["tests"].items()}
    except diag2.PostFailureDiagnosticsError as error:
        report["protocol_error"] = str(error)
        return EXIT_BLOCKED, report

    try:
        sources.verify_detector_inputs(repo, arms=probe.ARMS)
        report["source_inputs_resolved"] = True
    except Exception as error:                        # noqa: BLE001
        report["source_inputs_error"] = f"{type(error).__name__}: {error}"

    try:
        probe.resolve_all_checkpoint_sets(repo)
        report["checkpoints_resolved"] = True
    except probe.SyntheticRealProbeError as error:
        report["checkpoints_error"] = str(error)

    try:
        diag2.bind_c8_matrix_identity(repo)
        report["c8_matrix_identity_resolvable"] = True
    except diag2.PostFailureDiagnosticsError as error:
        report["c8_matrix_identity_error"] = str(error)

    report["ba_sep_canary"] = _ba_sep_canary(repo)
    report["ready_for_bind"] = bool(
        report["protocol_resolved"] and report["source_inputs_resolved"]
        and report["checkpoints_resolved"] and report["c8_matrix_identity_resolvable"]
        and report["target_firewall_clean"])
    exit_code = EXIT_PASS if report["ready_for_bind"] else EXIT_BLOCKED
    return exit_code, report


# ==============================================================================
# --bind-only
# ==============================================================================

def _bind_only(repo: Path) -> tuple[int, dict[str, Any]]:
    from prism_fas.evaluation import post_failure_diagnostics_v2 as diag2
    from prism_fas.evaluation import synthetic_real_probe as probe
    from prism_fas.pipeline.state import atomic_write_json

    report: dict[str, Any] = {"bound": False, "reused": False, "artifacts_written": False,
                              "target_access": 0}
    try:
        protocol = diag2.load_protocol(repo)
        protocol_id = diag2.protocol_identity(protocol)
        c8_matrix = diag2.bind_c8_matrix_identity(repo)   # Defect D: real, cross-checked

        ba_sep_binding = probe.build_checkpoint_binding(repo)   # reused, not duplicated
        checkpoint_binding = {
            "schema_version": "c9-post-failure-diagnostics-v2-checkpoint-binding-v1",
            "protocol_identity": protocol_id,
            "source_package_identity": ba_sep_binding["source_package_identity"],
            "c6_bank_identities": ba_sep_binding["c6_bank_identities"],
            "checkpoints": ba_sep_binding["checkpoints"],
            "checkpoints_per_arm": ba_sep_binding["checkpoints_per_arm"],
            "total_checkpoints": ba_sep_binding["total_checkpoints"],
            "target_access": 0,
        }
        checkpoint_binding["checkpoint_binding_identity_sha256"] = diag2.protocol_identity(
            {k: v for k, v in checkpoint_binding.items() if k != "protocol_identity"})

        live_records = diag2.resolve_source_dev_live_records(repo)
        group_ids = sorted({r["stable_group_identity"] for r in live_records})
        shared = protocol["benign_corruption_shared"]
        split = diag2.calibration_evaluation_split(
            group_ids, namespace=shared["split_hash_namespace"], seed=int(shared["split_seed"]))
        per_domain_safety = diag2.verify_per_domain_group_safety(
            live_records, split, domains=diag2.DOMAINS)   # Defect E: fails closed on degeneracy
        calibration_groups, evaluation_groups = set(split["calibration"]), set(split["evaluation"])
        calibration_ids = sorted(r["sample_id"] for r in live_records
                                 if r["stable_group_identity"] in calibration_groups)
        evaluation_ids = sorted(r["sample_id"] for r in live_records
                                if r["stable_group_identity"] in evaluation_groups)

        population_binding = {
            "schema_version": "c9-post-failure-diagnostics-v2-population-binding-v1",
            "protocol_identity": protocol_id,
            "benign_corruption": {"calibration_sample_ids": calibration_ids,
                                  "evaluation_sample_ids": evaluation_ids},
            "per_domain_group_safety": per_domain_safety,
            "target_access": 0,
        }
        population_binding["population_binding_identity_sha256"] = diag2.protocol_identity(
            {k: v for k, v in population_binding.items() if k != "protocol_identity"})

        protocol_binding = {
            "schema_version": "c9-post-failure-diagnostics-v2-protocol-binding-v1",
            "protocol_identity": protocol_id,
            "checkpoint_binding_identity": checkpoint_binding["checkpoint_binding_identity_sha256"],
            "population_binding_identity": population_binding["population_binding_identity_sha256"],
            "executable_tests": list(EXECUTABLE_TESTS),
            "c8_matrix_identity": c8_matrix["c8_matrix_identity"],
            "target_access": 0,
        }
    except Exception as error:                        # noqa: BLE001
        report["error"] = f"{type(error).__name__}: {error}"
        return EXIT_BLOCKED, report

    paths = {"protocol": Path(repo) / PROTOCOL_BINDING_PATH,
            "population": Path(repo) / POPULATION_BINDING_PATH,
            "checkpoint": Path(repo) / CHECKPOINT_BINDING_PATH}
    docs = {"protocol": protocol_binding, "population": population_binding,
           "checkpoint": checkpoint_binding}
    existing = {key: _read_json(path) for key, path in paths.items()}

    for key in paths:
        if existing[key] is not None and existing[key] != docs[key]:
            report["error"] = (
                f"an existing {key} binding differs from the one just resolved; "
                "refusing to silently overwrite a prior preregistration")
            return EXIT_BLOCKED, report

    if all(existing[key] is not None for key in paths):
        report.update({"bound": True, "reused": True, "artifacts_written": False,
                       "protocol_identity": protocol_id, "c8_matrix_identity": c8_matrix["c8_matrix_identity"],
                       "checkpoint_weights_loaded": False, "images_forwarded": False,
                       "ba_metric_computed": False})
        return EXIT_PASS, report

    for key, path in paths.items():
        atomic_write_json(path, docs[key])
    report.update({"bound": True, "reused": False, "artifacts_written": True,
                  "protocol_identity": protocol_id, "c8_matrix_identity": c8_matrix["c8_matrix_identity"],
                  "protocol_binding_path": PROTOCOL_BINDING_PATH,
                  "population_binding_path": POPULATION_BINDING_PATH,
                  "checkpoint_binding_path": CHECKPOINT_BINDING_PATH,
                  "checkpoint_weights_loaded": False, "images_forwarded": False,
                  "ba_metric_computed": False})
    return EXIT_PASS, report


# ==============================================================================
# --status
# ==============================================================================

def _status(repo: Path) -> tuple[int, dict[str, Any]]:
    from prism_fas.evaluation import post_failure_diagnostics_v2 as diag2

    presence = {name: (Path(repo) / relative).is_file()
               for name, relative in diag2.RESULT_ARTIFACT_PATHS.items()}
    count = sum(presence.values())
    report: dict[str, Any] = {
        "diagnostics_result_available": count == len(presence),
        "diagnostics_result_presence": presence,
        "ba_sep_canary": _ba_sep_canary(repo),
        "target_access": 0,
        "c9_may_close": False,
    }
    if count == 0:
        report["reason"] = "NO_DIAGNOSTICS_RESULT_ON_THIS_HOST"
        return EXIT_BLOCKED, report
    if count < len(presence):
        report["reason"] = "PARTIAL_SCIENTIFIC_RESULT_SET"
        return EXIT_BLOCKED, report

    validation = diag2.validate_existing_diagnostics_result(repo)
    report["existing_result_validation"] = {"valid": validation["valid"],
                                            "problems": validation["problems"]}
    if not validation["valid"]:
        report["reason"] = "EXISTING_RESULT_FAILED_VALIDATION"
        return EXIT_BLOCKED, report

    verdict_doc = validation["docs"]["verdict"]
    per_test_doc = validation["docs"]["per_test"]
    report["per_test"] = per_test_doc.get("per_test")
    report["overall_diagnostics_verdict"] = verdict_doc.get("overall_diagnostics_verdict")
    exit_code = EXIT_PASS if report["overall_diagnostics_verdict"] == "PASS" else EXIT_FAIL
    return exit_code, report


# ==============================================================================
# --execute
# ==============================================================================

def _execute(repo: Path) -> tuple[int, dict[str, Any]]:
    from prism_fas.evaluation import post_failure_diagnostics_v2 as diag2
    from prism_fas.pipeline.state import atomic_write_json

    report: dict[str, Any] = {"executed": False, "target_access": 0}

    presence = {name: (Path(repo) / relative).is_file()
               for name, relative in diag2.RESULT_ARTIFACT_PATHS.items()}
    count = sum(presence.values())

    # Defect C: a complete existing result is VALIDATED, never trusted bare,
    # before it is re-reported — and never recomputed or overwritten if
    # validation fails.
    if count == len(presence):
        validation = diag2.validate_existing_diagnostics_result(repo)
        if not validation["valid"]:
            report.update({"error": "EXISTING_RESULT_FAILED_VALIDATION",
                          "problems": validation["problems"]})
            return EXIT_BLOCKED, report
        verdict_doc = validation["docs"]["verdict"]
        per_test_doc = validation["docs"]["per_test"]
        report.update({
            "executed": True, "reused_existing_diagnostics_result": True,
            "checkpoint_weights_loaded": False, "images_forwarded": False,
            "ba_metric_recomputed": False,
            "per_test": per_test_doc.get("per_test"),
            "overall_diagnostics_verdict": verdict_doc.get("overall_diagnostics_verdict"),
            "c9_may_close": False,
        })
        exit_code = (EXIT_PASS if report["overall_diagnostics_verdict"] == "PASS" else EXIT_FAIL)
        return exit_code, report
    if 0 < count < len(presence):
        report.update({"error": "PARTIAL_SCIENTIFIC_RESULT_SET",
                       "present": {k: v for k, v in presence.items() if v},
                       "missing": {k: v for k, v in presence.items() if not v}})
        return EXIT_BLOCKED, report

    protocol_binding = _read_json(Path(repo) / PROTOCOL_BINDING_PATH)
    population_binding = _read_json(Path(repo) / POPULATION_BINDING_PATH)
    checkpoint_binding = _read_json(Path(repo) / CHECKPOINT_BINDING_PATH)
    if protocol_binding is None or population_binding is None or checkpoint_binding is None:
        report["error"] = "no diagnostics binding on disk; run --bind-only first"
        return EXIT_BLOCKED, report

    try:
        protocol = diag2.load_protocol(repo)
        active_id = diag2.protocol_identity(protocol)
    except diag2.PostFailureDiagnosticsError as error:
        report["error"] = str(error)
        return EXIT_BLOCKED, report
    if protocol_binding.get("protocol_identity") != active_id or \
            population_binding.get("protocol_identity") != active_id or \
            checkpoint_binding.get("protocol_identity") != active_id:
        report["error"] = "bound artifacts are not bound to the currently active protocol identity"
        return EXIT_BLOCKED, report

    # Defect D: the C8 matrix identity is re-verified fresh at execution time
    # and must still match the identity bound at --bind-only.
    try:
        current_c8 = diag2.bind_c8_matrix_identity(repo)
    except diag2.PostFailureDiagnosticsError as error:
        report["error"] = f"C8 matrix identity re-verification failed: {error}"
        return EXIT_BLOCKED, report
    if protocol_binding.get("c8_matrix_identity") != current_c8["c8_matrix_identity"]:
        report["error"] = ("the bound c8_matrix_identity no longer matches the current canonical "
                           "C8 matrix identity; fail closed rather than execute against a drifted "
                           "C8 matrix")
        return EXIT_BLOCKED, report

    try:
        from prism_fas.evaluation.synthetic_real_probe import ARMS, CheckpointBinding

        raw_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
        for item in checkpoint_binding["checkpoints"]:
            raw_by_arm[str(item["arm"])].append(item)
        checkpoints_by_arm: dict[str, list[Any]] = {
            arm: [CheckpointBinding(
                arm=str(item["arm"]), seed=int(item["seed"]), row_id=str(item["row_id"]),
                run_identity=str(item["run_identity"]), config_identity=str(item["config_identity"]),
                checkpoint_sha256=str(item["checkpoint_sha256"]),
                checkpoint_path=str(item["checkpoint_relative_path"]),
                checkpoint_kind=Path(str(item["checkpoint_relative_path"])).stem,
                decision_logit_name=str(item["decision_logit_name"]),
                decision_graph_hash=str(item["decision_graph_hash"]))
                for item in raw_by_arm[arm]]
            for arm in ARMS}

        evaluation_ids = population_binding["benign_corruption"]["evaluation_sample_ids"]
        calibration_ids = population_binding["benign_corruption"]["calibration_sample_ids"]

        # Reference threshold: computed ONCE per arm from the calibration
        # group's reference-variant delta_plus — never from the tested
        # corruption, and shared across all three benign-corruption tests.
        reference_threshold_by_arm: dict[str, dict[str, Any]] = {}
        for arm in ARMS:
            reference_delta_plus = diag2.reference_delta_plus_for_arm(
                repo, arm, checkpoints_by_arm[arm], calibration_ids)
            reference_threshold_by_arm[arm] = diag2.derive_reference_threshold(reference_delta_plus)

        per_test: dict[str, Any] = {}
        for test_id, config in protocol["tests"].items():
            if config["classification"] != "EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL":
                per_test[test_id] = {"status": "BLOCKED", "classification": config["classification"],
                                     "blocked_reason": config.get("blocked_reason", "")}
                continue
            per_arm: dict[str, Any] = {}
            for arm in ARMS:
                per_arm[arm] = diag2.run_benign_corruption_diagnostic_for_arm(
                    repo, test_id, arm, checkpoints_by_arm[arm],
                    evaluation_ids=evaluation_ids,
                    reference_threshold=reference_threshold_by_arm[arm])
            test_verdict = "PASS" if all(per_arm[arm]["verdict"] == "PASS" for arm in ARMS) else "FAIL"
            per_test[test_id] = {"status": test_verdict, "classification": config["classification"],
                                 "per_arm": per_arm}
    except Exception as error:                        # noqa: BLE001
        report["error"] = f"{type(error).__name__}: {error}"
        return EXIT_BLOCKED, report

    import prism_fas.detector.checkpoint as checkpoint_module

    common = {
        "protocol_identity": active_id,
        "checkpoint_binding_identity": checkpoint_binding["checkpoint_binding_identity_sha256"],
        "population_binding_identity": population_binding["population_binding_identity_sha256"],
        "source_package_identity": checkpoint_binding["source_package_identity"],
        "c6_bank_identities": checkpoint_binding["c6_bank_identities"],
        "code_commit": checkpoint_module.git_commit(),
        "c8_matrix_identity": current_c8["c8_matrix_identity"],
        "ba_sep_protocol_identity": "720a2e344017d588d71005b81fdf0e7d2062081ae2f3881a61a306d952dc4ac8",
        "ba_sep_observed_verdict": "FAIL",
        "detector_reliability_lock_c_observed_overall": "FAILED",
        "target_access": 0,
    }
    executed_verdicts = [per_test[t]["status"] for t in EXECUTABLE_TESTS if t in per_test]
    overall = "PASS" if all(v == "PASS" for v in executed_verdicts) else "FAIL"

    atomic_write_json(Path(repo) / RESULT_PATH, {**common, "per_test": per_test})
    atomic_write_json(Path(repo) / PER_TEST_PATH, {**common, "per_test": per_test})
    atomic_write_json(Path(repo) / PROVENANCE_PATH, {**common})
    atomic_write_json(Path(repo) / VERDICT_PATH, {
        **common, "overall_diagnostics_verdict": overall,
        "c9_may_close": False,
        "note": ("this verdict is diagnostic only; it can never reopen C9 or change "
                "DETECTOR_RELIABILITY_LOCK_C, which remains FAILED regardless")})

    report.update({"executed": True, "reused_existing_diagnostics_result": False,
                  "per_test": per_test, "overall_diagnostics_verdict": overall,
                  "c9_may_close": False,
                  "result_path": RESULT_PATH, "per_test_path": PER_TEST_PATH,
                  "provenance_path": PROVENANCE_PATH, "verdict_path": VERDICT_PATH})
    exit_code = EXIT_PASS if overall == "PASS" else EXIT_FAIL
    return exit_code, report


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.preflight_only:
        exit_code, payload = _preflight(args.repo)
    elif args.bind_only:
        exit_code, payload = _bind_only(args.repo)
    elif args.status:
        exit_code, payload = _status(args.repo)
    else:
        exit_code, payload = _execute(args.repo)

    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
