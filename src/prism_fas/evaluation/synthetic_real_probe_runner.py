"""The C9 BA_sep Option-1 V2 scientific runner CLI.

    python -m prism_fas.evaluation.synthetic_real_probe_runner --repo . --preflight-only
    python -m prism_fas.evaluation.synthetic_real_probe_runner --repo . --bind-only
    python -m prism_fas.evaluation.synthetic_real_probe_runner --repo . --execute

The frozen protocol's balancing rule is JOINT across RND/DET/LLM
(`synthetic_real_probe.balance_classes` requires all three arms' synthetic
pools simultaneously: `N = min(real, RND, DET, LLM)`). An earlier version of
this CLI took `--bind-only --arm {RND,DET,LLM}` and bound one arm at a time
— which forced the OTHER two arms' pools to `[]`, and therefore `N = 0`,
for every cell. That was an implementation contradiction with the
already-frozen protocol, not a scientific decision, and is fixed here: there
is no `--arm` flag on any mode. The synthetic-vs-real reliability test is
ONE preregistered three-arm experiment, bound and executed as one unit.

Three modes, mutually exclusive:

`--preflight-only`
    Read-only, all arms. PASSES only if the frozen V2 protocol resolves, the
    source package and all three C6 arm banks resolve, all 15 C8
    checkpoints resolve and hash-verify, the real/synthetic
    `source_record_id` group-identity mapping validates end to end, and the
    target firewall is clean. Never fits a probe, never opens an image,
    never loads a checkpoint's weights, never writes a file.

`--bind-only`
    Resolves the real checkpoint manifests and the joint three-arm
    population plan, then atomically writes exactly two artifacts under
    `reports/full/c8/reliability/synthetic_vs_real_spoof_probe/`:
    `C9_BA_SEP_EXECUTION_BINDING.json` (all 15 checkpoint bindings, the
    package identity, all three C6 bank identities) and
    `C9_BA_SEP_POPULATION_PLAN.json` (the exact preselected sample_identity
    lists for every `(probe_seed, source_domain, split)` cell, for the real
    population and each arm's synthetic population, group-safe by
    construction and re-audited for leakage). Both are built fully in
    memory first; either both are written or neither is. A required cell
    that resolves zero samples, or a group-safety leak, blocks the whole
    bind — nothing partial is ever written. If matching artifacts already
    exist (identical identity), they are verified and reused rather than
    rewritten; if existing artifacts carry a DIFFERENT identity, binding is
    refused rather than silently overwriting a prior preregistration.

`--execute`
    Runs the real, joint, three-arm probe: for every arm, strict-loads all
    five bound checkpoints (reusing the exact C8 row construction path),
    forwards every required real and synthetic sample through each,
    averages evidence across the five checkpoints, fits the frozen linear
    probe per `(arm, probe_seed)` on the prebound population plan, computes
    `BA_sep_arm_seed` then `BA_sep_arm`, and applies the frozen all-arm hard
    verdict rule. Requires `--bind-only` to have already produced both
    artifacts, bound to the CURRENTLY active protocol identity; re-verifies
    every checkpoint's bytes on disk before any forward pass. A scientific
    FAILED verdict is a real result, written honestly — it is not the same
    as a BLOCKED precondition failure.

Exit codes: 0 PASS (successful preflight/bind, or a scientific PASS
verdict), 1 FAIL (a real scientific FAILED verdict — not an error), 2
BLOCKED (a precondition, resolution or execution failure), 3 USAGE error —
the same convention `train.py` uses.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EXIT_PASS, EXIT_FAIL, EXIT_BLOCKED, EXIT_USAGE = 0, 1, 2, 3

#: Every artifact this runner reads or writes lives here — beside the other
#: post-C8, pre-C9 reliability evidence
#: (`detector_reliability.LOCK_PATH` = reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json)
#: rather than inside the frozen C8 run tree itself.
RELIABILITY_DIR = "reports/full/c8/reliability/synthetic_vs_real_spoof_probe"
EXECUTION_BINDING_PATH = f"{RELIABILITY_DIR}/C9_BA_SEP_EXECUTION_BINDING.json"
POPULATION_PLAN_PATH = f"{RELIABILITY_DIR}/C9_BA_SEP_POPULATION_PLAN.json"
RESULT_PATH = f"{RELIABILITY_DIR}/BA_SEP_RESULT.json"
PER_SEED_PATH = f"{RELIABILITY_DIR}/BA_SEP_PER_SEED.json"
PARAMETERS_PATH = f"{RELIABILITY_DIR}/BA_SEP_PROBE_PARAMETERS.json"
EVIDENCE_MANIFEST_PATH = f"{RELIABILITY_DIR}/BA_SEP_EVIDENCE_MANIFEST.json"
VERDICT_PATH = f"{RELIABILITY_DIR}/SYNTHETIC_VS_REAL_SPOOF_PROBE_VERDICT.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m prism_fas.evaluation.synthetic_real_probe_runner",
        description="C9_DETECTOR_BA_SEP_OPTION1_V2 scientific runner. The "
                    "synthetic-vs-real probe is one joint three-arm "
                    "experiment; no mode takes --arm.")
    parser.add_argument("--repo", default=".", type=Path,
                        help="repository root (default: current directory)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true",
                      help="read-only production readiness check, all arms")
    mode.add_argument("--bind-only", action="store_true",
                      help="resolve and atomically write the joint checkpoint binding "
                           "and population plan; never loads checkpoint weights or "
                           "opens an image")
    mode.add_argument("--execute", action="store_true",
                      help="run the real joint three-arm probe against the bound "
                           "artifacts and write the scientific result")
    return parser


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ==============================================================================
# --preflight-only
# ==============================================================================

def _preflight(repo: Path) -> tuple[int, dict[str, Any]]:
    from prism_fas.evaluation import synthetic_real_probe as probe
    from prism_fas.pipeline.adapters import sources

    report: dict[str, Any] = {
        "protocol_resolved": False, "protocol_identity": None,
        "source_inputs_resolved": False, "source_inputs_error": "",
        "checkpoints_resolved": False, "checkpoints_error": "",
        "checkpoints_per_arm": {}, "total_checkpoints": 0,
        "group_identity_mapping_resolved": False, "group_identity_mapping_error": "",
        "target_firewall_clean": False,
        "probe_fit_executed": False, "ba_metric_computed": False,
        "scientific_artifacts_written": False, "state_modified": False,
        "detector_reliability_lock_created": False,
        "target_access": 0,
    }

    try:
        protocol = probe.load_protocol(repo)
        report["protocol_resolved"] = True
        report["protocol_identity"] = probe.protocol_identity(repo)
        report["target_firewall_clean"] = int(protocol.get("target_access", -1)) == 0 and \
            int((protocol.get("target_firewall") or {}).get("target_access", -1)) == 0
    except probe.SyntheticRealProbeError as error:
        report["protocol_error"] = str(error)
        return EXIT_BLOCKED, report

    try:
        sources.verify_detector_inputs(repo, arms=probe.ARMS)
        report["source_inputs_resolved"] = True
    except Exception as error:                        # noqa: BLE001
        report["source_inputs_error"] = f"{type(error).__name__}: {error}"

    try:
        by_arm = probe.resolve_all_checkpoint_sets(repo)
        report["checkpoints_resolved"] = True
        report["checkpoints_per_arm"] = {arm: len(items) for arm, items in by_arm.items()}
        report["total_checkpoints"] = sum(len(items) for items in by_arm.values())
    except probe.SyntheticRealProbeError as error:
        report["checkpoints_error"] = str(error)

    try:
        probe.resolve_joint_populations(repo)
        report["group_identity_mapping_resolved"] = True
    except Exception as error:                        # noqa: BLE001
        report["group_identity_mapping_error"] = f"{type(error).__name__}: {error}"

    report["ready_for_bind"] = bool(
        report["protocol_resolved"] and report["source_inputs_resolved"]
        and report["checkpoints_resolved"] and report["group_identity_mapping_resolved"]
        and report["target_firewall_clean"])
    exit_code = EXIT_PASS if report["ready_for_bind"] else EXIT_BLOCKED
    return exit_code, report


# ==============================================================================
# --bind-only
# ==============================================================================

def _bind_only(repo: Path) -> tuple[int, dict[str, Any]]:
    from prism_fas.evaluation import synthetic_real_probe as probe
    from prism_fas.pipeline.state import atomic_write_json

    report: dict[str, Any] = {"bound": False, "reused": False, "artifacts_written": False,
                              "target_access": 0}
    try:
        protocol = probe.load_protocol(repo)
        checkpoint_binding = probe.build_checkpoint_binding(repo)
        population_plan = probe.build_population_plan(repo, protocol=protocol)
    except Exception as error:                        # noqa: BLE001
        # Resolution spans several modules (sources/c6_evidence/dataset), each
        # with its own RuntimeError subclass; any one of them failing to
        # resolve on this host must become a reported BLOCKED result, never
        # an uncaught crash — the same boundary c8.py's own row executor
        # draws around real-artifact resolution.
        report["error"] = f"{type(error).__name__}: {error}"
        return EXIT_BLOCKED, report

    binding_path = Path(repo) / EXECUTION_BINDING_PATH
    plan_path = Path(repo) / POPULATION_PLAN_PATH
    existing_binding = _read_json(binding_path)
    existing_plan = _read_json(plan_path)

    if existing_binding is not None and existing_binding.get(
            "checkpoint_binding_identity_sha256") != checkpoint_binding[
            "checkpoint_binding_identity_sha256"]:
        report["error"] = (
            "an existing execution binding carries a DIFFERENT identity than the one "
            "just resolved; refusing to silently overwrite a prior preregistration")
        return EXIT_BLOCKED, report
    if existing_plan is not None and existing_plan.get(
            "population_plan_identity_sha256") != population_plan[
            "population_plan_identity_sha256"]:
        report["error"] = (
            "an existing population plan carries a DIFFERENT identity than the one "
            "just resolved; refusing to silently overwrite a prior preregistration")
        return EXIT_BLOCKED, report

    if existing_binding is not None and existing_plan is not None:
        report.update({
            "bound": True, "reused": True, "artifacts_written": False,
            "checkpoint_binding_identity": checkpoint_binding["checkpoint_binding_identity_sha256"],
            "population_plan_identity": population_plan["population_plan_identity_sha256"],
            "execution_binding_path": EXECUTION_BINDING_PATH,
            "population_plan_path": POPULATION_PLAN_PATH,
            "checkpoint_weights_loaded": False, "images_forwarded": False,
            "ba_metric_computed": False,
        })
        return EXIT_PASS, report

    # Both built fully in memory and validated above; write atomically now —
    # either both land or neither does.
    atomic_write_json(binding_path, checkpoint_binding)
    atomic_write_json(plan_path, population_plan)
    report.update({
        "bound": True, "reused": False, "artifacts_written": True,
        "checkpoint_binding_identity": checkpoint_binding["checkpoint_binding_identity_sha256"],
        "population_plan_identity": population_plan["population_plan_identity_sha256"],
        "execution_binding_path": EXECUTION_BINDING_PATH,
        "population_plan_path": POPULATION_PLAN_PATH,
        "checkpoints_per_arm": checkpoint_binding["checkpoints_per_arm"],
        "total_checkpoints": checkpoint_binding["total_checkpoints"],
        "population_cells": len(population_plan["cells"]),
        "checkpoint_weights_loaded": False, "images_forwarded": False,
        "ba_metric_computed": False,
    })
    return EXIT_PASS, report


# ==============================================================================
# --execute
# ==============================================================================

def _execute(repo: Path) -> tuple[int, dict[str, Any]]:
    from prism_fas.evaluation import detector_reliability
    from prism_fas.evaluation import synthetic_real_probe as probe
    from prism_fas.pipeline.state import atomic_write_json

    report: dict[str, Any] = {"executed": False, "target_access": 0}

    try:
        protocol = probe.load_protocol(repo)
    except probe.SyntheticRealProbeError as error:
        report["error"] = str(error)
        return EXIT_BLOCKED, report

    checkpoint_binding = _read_json(Path(repo) / EXECUTION_BINDING_PATH)
    population_plan = _read_json(Path(repo) / POPULATION_PLAN_PATH)
    if checkpoint_binding is None or population_plan is None:
        report["error"] = (
            "no joint execution binding and/or population plan on disk; run "
            "--bind-only first")
        return EXIT_BLOCKED, report

    protocol_id = probe.protocol_identity(repo)
    if checkpoint_binding.get("protocol_identity") != protocol_id or \
            population_plan.get("protocol_identity") != protocol_id:
        report["error"] = (
            "the bound artifacts are not bound to the currently active protocol "
            "identity; re-run --bind-only")
        return EXIT_BLOCKED, report

    try:
        result = probe.execute_joint_probe(
            repo, checkpoint_binding=checkpoint_binding, population_plan=population_plan)
    except Exception as error:                        # noqa: BLE001
        # Real checkpoint construction spans several modules (c7/c8 config
        # resolution, c6_bank, detector.trainer, detector.checkpoint); any
        # one of them refusing must become a reported BLOCKED result, never
        # an uncaught crash. No scientific artifact is written on this path.
        report["error"] = f"{type(error).__name__}: {error}"
        return EXIT_BLOCKED, report

    import prism_fas.detector.checkpoint as checkpoint_module

    code_commit = checkpoint_module.git_commit()
    common_binding = {
        "protocol_identity": result["protocol_identity"],
        "checkpoint_binding_identity": result["checkpoint_binding_identity"],
        "population_plan_identity": result["population_plan_identity"],
        "source_package_identity": checkpoint_binding["source_package_identity"],
        "c6_bank_identities": checkpoint_binding["c6_bank_identities"],
        "checkpoint_sha256_by_arm": {
            arm: sorted(item["checkpoint_sha256"] for item in checkpoint_binding["checkpoints"]
                       if item["arm"] == arm)
            for arm in probe.ARMS},
        "code_commit": code_commit,
        "target_access": 0,
    }

    atomic_write_json(Path(repo) / RESULT_PATH, {
        **common_binding, "ba_sep_by_arm": result["ba_sep_by_arm"]})
    atomic_write_json(Path(repo) / PER_SEED_PATH, {
        **common_binding, "per_seed_by_arm": result["per_seed_by_arm"]})
    atomic_write_json(Path(repo) / PARAMETERS_PATH, {
        **common_binding,
        "evidence_fields": list(probe.EVIDENCE_FIELDS),
        "probe_seed_values": list(protocol["probe_seed_values"]),
        "lbfgs_config": probe.LBFGS_CONFIG, "l2_lambda": probe.L2_LAMBDA,
        "classifier_threshold": probe.CLASSIFIER_THRESHOLD,
        "ba_ceiling": protocol["ba_ceiling"]})
    atomic_write_json(Path(repo) / EVIDENCE_MANIFEST_PATH, {
        **common_binding, "seed_details": result["seed_details"]})
    atomic_write_json(Path(repo) / VERDICT_PATH, {
        **common_binding, "verdict": result["verdict"],
        "c_h4_support_rule_is_separate": detector_reliability.C_H4_SUPPORT_RULE})

    report.update({
        "executed": True,
        "ba_sep_by_arm": result["ba_sep_by_arm"],
        "verdict": result["verdict"]["verdict"],
        "result_path": RESULT_PATH, "per_seed_path": PER_SEED_PATH,
        "parameters_path": PARAMETERS_PATH, "evidence_manifest_path": EVIDENCE_MANIFEST_PATH,
        "verdict_path": VERDICT_PATH,
    })
    exit_code = EXIT_PASS if result["verdict"]["verdict"] == "PASS" else EXIT_FAIL
    return exit_code, report


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.preflight_only:
        exit_code, payload = _preflight(args.repo)
    elif args.bind_only:
        exit_code, payload = _bind_only(args.repo)
    else:
        exit_code, payload = _execute(args.repo)

    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
