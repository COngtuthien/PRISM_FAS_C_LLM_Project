"""The C9 BA_sep Option-1 V2 scientific runner CLI.

    python -m prism_fas.evaluation.synthetic_real_probe_runner \\
        --repo . (--preflight-only | --bind-only --arm {RND,DET,LLM} | --execute --arm {RND,DET,LLM})

This module is the ONLY sanctioned entrypoint for moving the
`C9_DETECTOR_BA_SEP_OPTION1_V2` protocol from "frozen" toward "run" — every
other module in this project either resolves protocol/population/checkpoint
METADATA (`synthetic_real_probe.py`) or refuses to compute a real BA_sep
value at all (`run_scientific_probe` there deliberately raises
`NotImplementedError`). This CLI does not change that: `--execute` still
refuses, on every host, until a future task explicitly wires and tests the
real forward pass on a machine that has the GPU C8 artifacts to test it
against. What this CLI adds is the strict, fail-closed, three-mode CONTRACT
around that refusal, so a GPU host can run exactly the same commands a
laptop preflight-checks, with the same identity and firewall guarantees.

Three modes, mutually exclusive:

`--preflight-only`
    Read-only, all arms. Validates the frozen protocol resolves, reports its
    identity, and best-effort reports whether the 15 real C8 checkpoints are
    resolvable on THIS host (never fatal if they are not — a development
    clone with no `runs/full/c8/` is expected to report them unresolved).
    Never fits a probe, never computes a BA value, never writes a file,
    never touches `state/`. Exit 0 if the protocol resolves, 2 if it does
    not.

`--bind-only --arm {RND,DET,LLM}`
    Resolves the real checkpoint manifests and real populations for ONE arm
    through the exact canonical readers `synthetic_real_probe.py` already
    wraps (never a second implementation), builds the group-safe split and
    the balanced selection for every `(probe_seed, source_domain)` cell, and
    writes ONE binding-record artifact
    (`reports/full/c9/ba_sep_option1_v2/BINDING_{ARM}.json`) recording every
    resolved identity — but never loads a checkpoint's weights and never
    opens an image. On this development laptop, which has no
    `runs/full/c8/`, checkpoint resolution fails closed and NO artifact is
    written; that is the correct, expected outcome here. Exit 0 on a full
    bind, 2 if any input cannot be resolved (nothing partial is ever
    written).

`--execute --arm {RND,DET,LLM}`
    Would run the real probe end to end and write the scientific result
    artifacts. Calls `synthetic_real_probe.run_scientific_probe`, which
    raises `NotImplementedError` on every host today — this CLI reports that
    refusal clearly and exits 2. It never falls through to a partial or
    approximate computation.

Exit codes follow `train.py`: 0 pass, 2 blocked, 3 usage error. This runner
never returns 1 (FAIL) — there is no scientific FAIL state reachable from
metadata resolution or a deliberately-unwired execute call.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EXIT_PASS, EXIT_BLOCKED, EXIT_USAGE = 0, 2, 3

#: Where a successful --bind-only writes its one artifact per arm. Chosen to
#: sit beside the other post-C8, pre-C9 reliability evidence
#: (`detector_reliability.LOCK_PATH` = reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json)
#: without writing into the frozen C8 run tree itself.
BINDING_ARTIFACT_DIR = "reports/full/c9/ba_sep_option1_v2"


def _binding_artifact_path(repo: Path, arm: str) -> Path:
    return Path(repo) / BINDING_ARTIFACT_DIR / f"BINDING_{arm}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m prism_fas.evaluation.synthetic_real_probe_runner",
        description="C9_DETECTOR_BA_SEP_OPTION1_V2 scientific runner. "
                    "--execute is deliberately unwired on every host today; "
                    "see this module's docstring.")
    parser.add_argument("--repo", default=".", type=Path,
                        help="repository root (default: current directory)")
    parser.add_argument("--arm", choices=("RND", "DET", "LLM"),
                        help="required for --bind-only and --execute; ignored "
                             "(and must be omitted) for --preflight-only, which "
                             "always covers all three arms")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true",
                      help="read-only protocol/checkpoint-cardinality check, all arms")
    mode.add_argument("--bind-only", action="store_true",
                      help="resolve real checkpoints+populations for --arm and write "
                           "one binding-record artifact; never loads checkpoint weights "
                           "or opens an image")
    mode.add_argument("--execute", action="store_true",
                      help="would run the real probe for --arm; refuses on every host "
                           "today (run_scientific_probe is deliberately unwired)")
    return parser


def _preflight(repo: Path) -> tuple[int, dict[str, Any]]:
    from prism_fas.evaluation import synthetic_real_probe as probe

    result = probe.preflight(repo)
    checkpoint_report: dict[str, Any] = {}
    for arm in probe.ARMS:
        try:
            bindings = probe.resolve_checkpoint_set(repo, arm)
            checkpoint_report[arm] = {"resolved": True, "count": len(bindings)}
        except probe.SyntheticRealProbeError as error:
            checkpoint_report[arm] = {"resolved": False, "error": str(error)}
    result["checkpoints_by_arm"] = checkpoint_report
    result["all_checkpoints_resolved"] = all(
        entry["resolved"] for entry in checkpoint_report.values())
    exit_code = EXIT_PASS if result["protocol_resolved"] else EXIT_BLOCKED
    return exit_code, result


def _bind_only(repo: Path, arm: str) -> tuple[int, dict[str, Any]]:
    from prism_fas.evaluation import detector_reliability
    from prism_fas.evaluation import synthetic_real_probe as probe
    from prism_fas.pipeline.state import atomic_write_json

    report: dict[str, Any] = {
        "arm": arm, "bound": False, "artifact_written": False,
        "artifact_path": None, "target_access": 0,
    }
    try:
        protocol = probe.load_protocol(repo)
        protocol_id = probe.protocol_identity(repo)
        required_logit = protocol["detector_checkpoint_identity"]["required_decision_logit_name"]
        if required_logit != "global_logit_G":
            raise probe.SyntheticRealProbeError(
                f"protocol declares required_decision_logit_name={required_logit!r}, "
                "expected 'global_logit_G'; refusing to bind under a mismatched contract")

        checkpoints = probe.resolve_checkpoint_set(repo, arm)
        real_population, synthetic_population = probe.resolve_arm_populations(repo, arm)

        split_namespace = protocol["matched_source_split"]["split_hash_namespace"]
        probe_seeds = protocol["probe_seed_values"]
        source_domains = protocol["source_domains"]

        cells: list[dict[str, Any]] = []
        for seed in probe_seeds:
            real_split = probe.assign_splits(real_population, namespace=split_namespace,
                                             probe_seed=seed)
            probe.verify_group_safe_split(real_split)
            synth_split = probe.assign_splits(synthetic_population, namespace=split_namespace,
                                              probe_seed=seed)
            probe.verify_group_safe_split(synth_split)
            for domain in source_domains:
                for split_label in (probe.TRAIN_LABEL, probe.VALIDATION_LABEL):
                    real_cell = [r for r in real_split[split_label] if r.source_domain == domain]
                    synth_cell = [r for r in synth_split[split_label] if r.source_domain == domain]
                    balance = probe.balance_report(
                        protocol_id=protocol_id, probe_seed=seed, split=split_label,
                        source_domain=domain, real_spoof=real_cell,
                        synthetic_by_arm={a: (synth_cell if a == arm else [])
                                         for a in probe.ARMS})
                    cells.append({
                        "probe_seed": seed, "source_domain": domain, "split": split_label,
                        "n": balance["n"],
                        "unique_source_record_id_counts": balance["unique_source_record_id_counts"],
                    })

        report.update({
            "bound": True,
            "protocol_identity": protocol_id,
            "checkpoint_bindings": [
                {"seed": c.seed, "row_id": c.row_id, "run_identity": c.run_identity,
                 "checkpoint_sha256": c.checkpoint_sha256,
                 "checkpoint_path": c.checkpoint_path,
                 "decision_logit_name": "global_logit_G"}
                for c in checkpoints],
            "real_population_count": len(real_population),
            "synthetic_population_count": len(synthetic_population),
            "split_cells": cells,
            "group_safety_verified": True,
            "checkpoint_weights_loaded": False,
            "images_forwarded": False,
            "ba_metric_computed": False,
        })
    except (probe.SyntheticRealProbeError, detector_reliability.DetectorReliabilityError) as error:
        report["error"] = str(error)
        return EXIT_BLOCKED, report

    artifact_path = _binding_artifact_path(repo, arm)
    atomic_write_json(artifact_path, report)
    report["artifact_written"] = True
    report["artifact_path"] = artifact_path.relative_to(Path(repo)).as_posix()
    return EXIT_PASS, report


def _execute(repo: Path, arm: str) -> tuple[int, dict[str, Any]]:
    from prism_fas.evaluation import synthetic_real_probe as probe

    try:
        probe.run_scientific_probe(repo, arm)
        # Unreachable today: run_scientific_probe always raises. If a future
        # task wires it, this branch becomes the real success path.
        return EXIT_PASS, {"arm": arm, "executed": True}
    except NotImplementedError as error:
        return EXIT_BLOCKED, {
            "arm": arm, "executed": False,
            "reason": ("run_scientific_probe is deliberately unwired on this host "
                      "(and on every host today) — see synthetic_real_probe.py's "
                      "module and function docstrings. No BA_sep value was computed, "
                      "no checkpoint weights were loaded, no image was opened, no "
                      "scientific artifact was written."),
            "detail": str(error), "target_access": 0,
        }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.preflight_only:
        if args.arm is not None:
            print("usage error: --preflight-only covers all arms; do not pass --arm",
                 file=sys.stderr)
            return EXIT_USAGE
        exit_code, payload = _preflight(args.repo)
    else:
        if args.arm is None:
            print("usage error: --bind-only and --execute require --arm {RND,DET,LLM}",
                 file=sys.stderr)
            return EXIT_USAGE
        if args.bind_only:
            exit_code, payload = _bind_only(args.repo, args.arm)
        else:
            exit_code, payload = _execute(args.repo, args.arm)

    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
