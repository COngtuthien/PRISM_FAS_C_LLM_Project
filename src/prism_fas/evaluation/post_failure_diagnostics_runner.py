"""C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V1 runner CLI.

    python -m prism_fas.evaluation.post_failure_diagnostics_runner --repo . --preflight-only
    python -m prism_fas.evaluation.post_failure_diagnostics_runner --repo . --bind-only
    python -m prism_fas.evaluation.post_failure_diagnostics_runner --repo . --status
    python -m prism_fas.evaluation.post_failure_diagnostics_runner --repo . --execute

**NOT A BA_sep REVISION. NOT A RELIABILITY-BARRIER RESCUE. NOT A C9 PASS
PATH.** `synthetic_vs_real_spoof_probe` has already, permanently, FAILED
under `C9_DETECTOR_BA_SEP_OPTION1_V2`. No mode of this CLI reads, writes, or
depends on writing to `reports/full/c8/reliability/synthetic_vs_real_spoof_probe/`
or `reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json` — a fully SEPARATE
artifact namespace (`post_failure_diagnostics.DIAGNOSTICS_DIR`) is used
throughout. Even a diagnostic that PASSES on every arm cannot make
`c9_may_close` true or the detector-reliability barrier's `overall` anything
but `FAILED` — every artifact this runner writes says so explicitly.

Four modes:

`--preflight-only`
    Read-only. Validates the frozen diagnostics protocol, the source
    package, all 15 checkpoints (hash-verified), and the target firewall.
    Reports each test's GPU-readiness exactly as the frozen protocol
    declares it — never computes or infers a readiness this CLI did not
    already decide when the protocol was frozen. Never fits anything, never
    forwards an image, never writes a file.

`--bind-only`
    Resolves and atomically writes three binding artifacts (protocol,
    population, checkpoint) under `post_failure_diagnostics.DIAGNOSTICS_DIR`
    — zero scientific metric in any of them, pure identity and population
    membership. Idempotent: identical existing bindings are reused; a
    DIFFERENT existing binding blocks rather than being overwritten.

`--status`
    Read-only report of whatever diagnostics result exists on this host —
    none, partial, or complete — plus the CURRENT (informational,
    read-only) BA_sep protocol identity and reliability-lock overall, so a
    caller can see at a glance that neither has moved.

`--execute`
    SCIENTIFICALLY NO-RERUN, exactly like the BA_sep runner: with all four
    result artifacts present, re-reports the existing result (zero
    recomputation); with some-but-not-all present, BLOCKS
    (`PARTIAL_SCIENTIFIC_RESULT_SET`); only on a clean host does it run each
    `EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL` test, once, for every arm, and
    write the four result artifacts. Exit code reflects whether every
    EXECUTED diagnostic PASSED — it NEVER implies C9 may close.

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

from prism_fas.evaluation.post_failure_diagnostics import (  # noqa: E402
    CHECKPOINT_BINDING_PATH, EXECUTABLE_TESTS, PER_TEST_PATH, POPULATION_BINDING_PATH,
    PROTOCOL_BINDING_PATH, PROVENANCE_PATH, RESULT_PATH, VERDICT_PATH)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m prism_fas.evaluation.post_failure_diagnostics_runner",
        description="C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V1 runner. Bounded, "
                    "source-only, mechanistic diagnostics AFTER the already-"
                    "observed BA_sep FAILURE. Never a C9 pass path.")
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
    reliability-lock overall, so every artifact this runner writes carries
    proof neither moved. Never written by this module; never derived from
    anything this module computes."""
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
    from prism_fas.evaluation import post_failure_diagnostics as diag
    from prism_fas.evaluation import synthetic_real_probe as probe
    from prism_fas.pipeline.adapters import sources

    report: dict[str, Any] = {
        "protocol_resolved": False, "protocol_identity": None,
        "source_inputs_resolved": False, "source_inputs_error": "",
        "checkpoints_resolved": False, "checkpoints_error": "",
        "target_firewall_clean": False, "per_test_gpu_ready": {},
        "ba_metric_computed": False, "images_forwarded": False,
        "scientific_artifacts_written": False, "state_modified": False,
        "target_access": 0,
    }
    try:
        protocol = diag.load_protocol(repo)
        report["protocol_resolved"] = True
        report["protocol_identity"] = diag.protocol_identity(protocol)
        report["target_firewall_clean"] = int(protocol.get("target_access", -1)) == 0
        report["per_test_gpu_ready"] = {test_id: bool(cfg.get("gpu_ready"))
                                        for test_id, cfg in protocol["tests"].items()}
    except diag.PostFailureDiagnosticsError as error:
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

    report["ba_sep_canary"] = _ba_sep_canary(repo)
    report["ready_for_bind"] = bool(
        report["protocol_resolved"] and report["source_inputs_resolved"]
        and report["checkpoints_resolved"] and report["target_firewall_clean"])
    exit_code = EXIT_PASS if report["ready_for_bind"] else EXIT_BLOCKED
    return exit_code, report


# ==============================================================================
# --bind-only
# ==============================================================================

def _bind_only(repo: Path) -> tuple[int, dict[str, Any]]:
    from prism_fas.evaluation import post_failure_diagnostics as diag
    from prism_fas.evaluation import synthetic_real_probe as probe
    from prism_fas.pipeline.state import atomic_write_json

    report: dict[str, Any] = {"bound": False, "reused": False, "artifacts_written": False,
                              "target_access": 0}
    try:
        protocol = diag.load_protocol(repo)
        protocol_id = diag.protocol_identity(protocol)

        ba_sep_binding = probe.build_checkpoint_binding(repo)   # reused, not duplicated
        checkpoint_binding = {
            "schema_version": "c9-post-failure-diagnostics-checkpoint-binding-v1",
            "protocol_identity": protocol_id,
            "source_package_identity": ba_sep_binding["source_package_identity"],
            "c6_bank_identities": ba_sep_binding["c6_bank_identities"],
            "checkpoints": ba_sep_binding["checkpoints"],
            "checkpoints_per_arm": ba_sep_binding["checkpoints_per_arm"],
            "total_checkpoints": ba_sep_binding["total_checkpoints"],
            "target_access": 0,
        }
        checkpoint_binding["checkpoint_binding_identity_sha256"] = diag.protocol_identity(
            {k: v for k, v in checkpoint_binding.items() if k != "protocol_identity"})

        live_records = diag.resolve_source_dev_live_records(repo)
        group_ids = sorted({r["stable_group_identity"] for r in live_records})
        split = diag.calibration_evaluation_split(
            group_ids, namespace=protocol["benign_corruption_shared"]["split_hash_namespace"],
            seed=int(protocol["benign_corruption_shared"]["split_seed"]))
        calibration_groups, evaluation_groups = set(split["calibration"]), set(split["evaluation"])
        calibration_ids = sorted(r["sample_id"] for r in live_records
                                 if r["stable_group_identity"] in calibration_groups)
        evaluation_ids = sorted(r["sample_id"] for r in live_records
                                if r["stable_group_identity"] in evaluation_groups)

        cross_route_population: dict[str, Any] = {}
        for arm in probe.ARMS:
            by_route = diag.resolve_synthetic_population_by_route(repo, arm)
            cross_route_population[arm] = {
                route: sorted(r.sample_identity for r in records)
                for route, records in by_route.items()}

        population_binding = {
            "schema_version": "c9-post-failure-diagnostics-population-binding-v1",
            "protocol_identity": protocol_id,
            "benign_corruption": {"calibration_sample_ids": calibration_ids,
                                  "evaluation_sample_ids": evaluation_ids},
            "cross_route_synthetic": cross_route_population,
            "target_access": 0,
        }
        population_binding["population_binding_identity_sha256"] = diag.protocol_identity(
            {k: v for k, v in population_binding.items() if k != "protocol_identity"})

        protocol_binding = {
            "schema_version": "c9-post-failure-diagnostics-protocol-binding-v1",
            "protocol_identity": protocol_id,
            "checkpoint_binding_identity": checkpoint_binding["checkpoint_binding_identity_sha256"],
            "population_binding_identity": population_binding["population_binding_identity_sha256"],
            "executable_tests": list(EXECUTABLE_TESTS),
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

    # Strict safety: compare the FULL document for exact equality rather
    # than trusting a single identity field.
    for key in paths:
        if existing[key] is not None and existing[key] != docs[key]:
            report["error"] = (
                f"an existing {key} binding differs from the one just resolved; "
                "refusing to silently overwrite a prior preregistration")
            return EXIT_BLOCKED, report

    if all(existing[key] is not None for key in paths):
        report.update({"bound": True, "reused": True, "artifacts_written": False,
                       "protocol_identity": protocol_id,
                       "checkpoint_weights_loaded": False, "images_forwarded": False,
                       "ba_metric_computed": False})
        return EXIT_PASS, report

    for key, path in paths.items():
        atomic_write_json(path, docs[key])
    report.update({"bound": True, "reused": False, "artifacts_written": True,
                  "protocol_identity": protocol_id,
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
    from prism_fas.evaluation import post_failure_diagnostics as diag

    presence = {name: (Path(repo) / relative).is_file()
               for name, relative in diag.RESULT_ARTIFACT_PATHS.items()}
    count = sum(presence.values())
    report: dict[str, Any] = {
        "diagnostics_result_available": count == len(presence),
        "diagnostics_result_presence": presence,
        "ba_sep_canary": _ba_sep_canary(repo),
        "target_access": 0,
    }
    if count == 0:
        report["reason"] = "NO_DIAGNOSTICS_RESULT_ON_THIS_HOST"
        return EXIT_BLOCKED, report
    if count < len(presence):
        report["reason"] = "PARTIAL_SCIENTIFIC_RESULT_SET"
        return EXIT_BLOCKED, report

    verdict_doc = _read_json(Path(repo) / VERDICT_PATH)
    per_test_doc = _read_json(Path(repo) / PER_TEST_PATH)
    report["per_test"] = (per_test_doc or {}).get("per_test")
    report["overall_diagnostics_verdict"] = (verdict_doc or {}).get("overall_diagnostics_verdict")
    report["c9_may_close"] = False   # always, regardless of diagnostic outcome
    exit_code = EXIT_PASS if report["overall_diagnostics_verdict"] == "PASS" else EXIT_FAIL
    return exit_code, report


# ==============================================================================
# --execute
# ==============================================================================

def _execute(repo: Path) -> tuple[int, dict[str, Any]]:
    from prism_fas.evaluation import post_failure_diagnostics as diag
    from prism_fas.pipeline.state import atomic_write_json

    report: dict[str, Any] = {"executed": False, "target_access": 0}

    presence = {name: (Path(repo) / relative).is_file()
               for name, relative in diag.RESULT_ARTIFACT_PATHS.items()}
    count = sum(presence.values())
    if count == len(presence):
        verdict_doc = _read_json(Path(repo) / VERDICT_PATH)
        per_test_doc = _read_json(Path(repo) / PER_TEST_PATH)
        report.update({
            "executed": True, "reused_existing_diagnostics_result": True,
            "checkpoint_weights_loaded": False, "images_forwarded": False,
            "ba_metric_recomputed": False,
            "per_test": (per_test_doc or {}).get("per_test"),
            "overall_diagnostics_verdict": (verdict_doc or {}).get("overall_diagnostics_verdict"),
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
        protocol = diag.load_protocol(repo)
        active_id = diag.protocol_identity(protocol)
    except diag.PostFailureDiagnosticsError as error:
        report["error"] = str(error)
        return EXIT_BLOCKED, report
    if protocol_binding.get("protocol_identity") != active_id or \
            population_binding.get("protocol_identity") != active_id or \
            checkpoint_binding.get("protocol_identity") != active_id:
        report["error"] = "bound artifacts are not bound to the currently active protocol identity"
        return EXIT_BLOCKED, report

    checkpoints_by_arm: dict[str, list[Any]] = {}
    try:
        from prism_fas.evaluation.synthetic_real_probe import ARMS, CheckpointBinding

        raw_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
        for item in checkpoint_binding["checkpoints"]:
            raw_by_arm[str(item["arm"])].append(item)
        for arm in ARMS:
            checkpoints_by_arm[arm] = [
                CheckpointBinding(
                    arm=str(item["arm"]), seed=int(item["seed"]), row_id=str(item["row_id"]),
                    run_identity=str(item["run_identity"]), config_identity=str(item["config_identity"]),
                    checkpoint_sha256=str(item["checkpoint_sha256"]),
                    checkpoint_path=str(item["checkpoint_relative_path"]),
                    checkpoint_kind=Path(str(item["checkpoint_relative_path"])).stem,
                    decision_logit_name=str(item["decision_logit_name"]),
                    decision_graph_hash=str(item["decision_graph_hash"]))
                for item in raw_by_arm[arm]]

        per_test: dict[str, Any] = {}
        for test_id, config in protocol["tests"].items():
            if config["classification"] != "EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL":
                per_test[test_id] = {"status": "BLOCKED", "classification": config["classification"],
                                     "blocked_reason": config.get("blocked_reason", "")}
                continue
            per_arm: dict[str, Any] = {}
            if test_id in diag.BENIGN_CORRUPTION_TESTS:
                calibration_ids = population_binding["benign_corruption"]["calibration_sample_ids"]
                evaluation_ids = population_binding["benign_corruption"]["evaluation_sample_ids"]
                for arm in ARMS:
                    per_arm[arm] = diag.run_benign_corruption_diagnostic_for_arm(
                        repo, test_id, arm, checkpoints_by_arm[arm],
                        calibration_ids=calibration_ids, evaluation_ids=evaluation_ids)
            else:   # cross_route_synthetic
                for arm in ARMS:
                    per_arm[arm] = diag.run_cross_route_diagnostic_for_arm(
                        repo, arm, checkpoints_by_arm[arm], protocol=protocol)
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
        "c8_matrix_identity": "see reports/full/c8/C8_ACCEPTANCE.json (unaltered by this runner)",
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
