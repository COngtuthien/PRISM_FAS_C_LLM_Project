"""GPU-side registration of an observed BA_sep scientific result into the
`DETECTOR_RELIABILITY_LOCK_C` barrier.

    python -m prism_fas.evaluation.detector_reliability_runner --repo . --status
    python -m prism_fas.evaluation.detector_reliability_runner --repo . --register-ba-sep-result

Two modes, mutually exclusive:

`--status`
    Strictly read-only. Reads whatever `synthetic_vs_real_spoof_probe`
    result artifacts exist on this host (via
    `synthetic_real_probe.validate_existing_scientific_result`) and reports
    honestly whether a result is available, whether it validates, and what
    it says — never fabricates a result from constants or from any prior
    conversation, and never loads a checkpoint, opens an image, fits a
    probe or writes a file.

`--register-ba-sep-result`
    Requires the complete, valid, already-written BA_sep result set (the
    seven artifacts `validate_existing_scientific_result` cross-checks) and
    binds its OBSERVED scientific verdict — PASSED or FAILED, exactly as
    recorded, never inferred or chosen — into
    `reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json` via the existing,
    unchanged `detector_reliability.lock_payload`. Only
    `synthetic_vs_real_spoof_probe` is ever set from this registration; the
    other eight `REQUIRED_DETECTOR_RELIABILITY_TESTS` stay `UNRESOLVED`
    unless a separate, later scientific execution resolves them — this
    module never infers or fabricates their outcome. Idempotent: identical
    scientific content already on disk is verified and reused; DIFFERENT
    existing content blocks rather than being silently overwritten. NEVER
    mutates the seven BA_sep artifacts it reads — read-only with respect to
    them.

Exit codes: 0 the registered/verified barrier is PASSED, 1 it is FAILED (a
real registered negative result, not an error), 2 BLOCKED (the result set
is absent, invalid, or an existing lock disagrees), 3 USAGE error — the
same convention every runner in this project uses.

NEVER exercised against real GPU artifacts on this development laptop:
there is no `runs/full/c8/`, no real BA_sep result, and `--register-ba-sep-result`
therefore always reports the result set absent and refuses, exactly as it
should.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EXIT_PASS, EXIT_FAIL, EXIT_BLOCKED, EXIT_USAGE = 0, 1, 2, 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m prism_fas.evaluation.detector_reliability_runner",
        description="Register an observed BA_sep scientific verdict into "
                    "DETECTOR_RELIABILITY_LOCK_C. Never fabricates a result; "
                    "only synthetic_vs_real_spoof_probe is ever set here.")
    parser.add_argument("--repo", default=".", type=Path,
                        help="repository root (default: current directory)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--status", action="store_true",
                      help="read-only report of whatever BA_sep result exists on this host")
    mode.add_argument("--register-ba-sep-result", action="store_true",
                      help="bind the observed, already-validated BA_sep verdict into "
                           "DETECTOR_RELIABILITY_LOCK_C")
    return parser


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ==============================================================================
# --status
# ==============================================================================

def _status(repo: Path) -> tuple[int, dict[str, Any]]:
    from prism_fas.evaluation import synthetic_real_probe as probe

    presence = probe.result_artifact_presence(repo)
    report: dict[str, Any] = {
        "ba_sep_result_available": presence["all_present"],
        "ba_sep_result_presence": presence["present"],
        "target_access": 0,
    }
    if not presence["all_present"]:
        report["ba_sep_result_valid"] = False
        report["reason"] = ("PARTIAL_SCIENTIFIC_RESULT_SET" if presence["partial"]
                            else "NO_SCIENTIFIC_RESULT_ON_THIS_HOST")
        return EXIT_BLOCKED, report

    validation = probe.validate_existing_scientific_result(repo)
    report.update({
        "ba_sep_result_valid": validation["valid"],
        "problems": validation["problems"],
        "scientific_verdict": validation["scientific_verdict"],
        "ba_sep_by_arm": validation["ba_sep_by_arm"],
        "protocol_identity": validation["protocol_identity"],
        "checkpoint_binding_identity": validation["checkpoint_binding_identity"],
        "population_plan_identity": validation["population_plan_identity"],
        "source_package_identity": validation["source_package_identity"],
        "checkpoints_per_arm": validation["checkpoints_per_arm"],
        "total_checkpoints": validation["total_checkpoints"],
    })
    if not validation["valid"]:
        report["expected_barrier_state"] = None
        return EXIT_BLOCKED, report

    report["expected_barrier_state"] = (
        "FAILED" if validation["scientific_verdict"] == "FAIL" else
        "UNRESOLVED (synthetic_vs_real_spoof_probe would PASS, but the other eight "
        "required tests remain unresolved unless separately executed)")
    exit_code = EXIT_PASS if validation["scientific_verdict"] == "PASS" else EXIT_FAIL
    return exit_code, report


# ==============================================================================
# --register-ba-sep-result
# ==============================================================================

def _register(repo: Path) -> tuple[int, dict[str, Any]]:
    from prism_fas.evaluation import detector_reliability
    from prism_fas.evaluation import synthetic_real_probe as probe
    from prism_fas.pipeline.state import atomic_write_json

    report: dict[str, Any] = {"registered": False, "target_access": 0}

    validation = probe.validate_existing_scientific_result(repo)
    if not validation["valid"]:
        report.update({
            "reason": "BA_SEP_RESULT_ABSENT_OR_INVALID",
            "problems": validation["problems"],
        })
        return EXIT_BLOCKED, report

    binding = _read_json(Path(repo) / probe.EXECUTION_BINDING_PATH)
    checkpoints = list((binding or {}).get("checkpoints") or [])
    checkpoint_identities = {str(item["row_id"]): str(item["checkpoint_sha256"])
                             for item in checkpoints}
    if len(checkpoint_identities) != probe.TOTAL_CHECKPOINTS:
        report.update({
            "reason": "CHECKPOINT_IDENTITY_COUNT_MISMATCH",
            "expected": probe.TOTAL_CHECKPOINTS, "found": len(checkpoint_identities),
        })
        return EXIT_BLOCKED, report

    verdict = validation["scientific_verdict"]
    results = {"synthetic_vs_real_spoof_probe":
              detector_reliability.PASSED if verdict == "PASS" else detector_reliability.FAILED}
    payload = detector_reliability.lock_payload(
        results=results,
        probe_protocol_identity=validation["protocol_identity"],
        detector_checkpoint_identities=checkpoint_identities,
        ba_sep_by_arm=validation["ba_sep_by_arm"])

    lock_path = Path(repo) / detector_reliability.LOCK_PATH
    existing = _read_json(lock_path)
    if existing is not None:
        if existing.get("identity_sha256") == payload["identity_sha256"]:
            report.update({
                "registered": True, "reused": True,
                "overall": existing.get("overall"),
                "per_test": existing.get("per_test"),
                "lock_path": detector_reliability.LOCK_PATH,
            })
            exit_code = (EXIT_PASS if existing.get("overall") == detector_reliability.PASSED
                        else EXIT_FAIL)
            return exit_code, report
        report.update({
            "reason": "EXISTING_LOCK_HAS_DIFFERENT_SCIENTIFIC_CONTENT",
            "existing_identity_sha256": existing.get("identity_sha256"),
            "new_identity_sha256": payload["identity_sha256"],
        })
        return EXIT_BLOCKED, report

    atomic_write_json(lock_path, payload)
    report.update({
        "registered": True, "reused": False,
        "overall": payload["overall"], "per_test": payload["per_test"],
        "c9_may_close": payload["c9_may_close"],
        "lock_path": detector_reliability.LOCK_PATH,
        "identity_sha256": payload["identity_sha256"],
    })
    exit_code = EXIT_PASS if payload["overall"] == detector_reliability.PASSED else EXIT_FAIL
    return exit_code, report


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.status:
        exit_code, payload = _status(args.repo)
    else:
        exit_code, payload = _register(args.repo)

    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
