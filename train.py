#!/usr/bin/env python
"""The canonical root orchestration entrypoint (v1.5 Appendix L.4).

The normal user workflow is one command with no arguments:

    python train.py

On a fresh machine that bootstraps the environment, detects the hardware,
verifies the folder is complete, picks the right execution intent, resumes
whatever is already done and runs to completion. The expert flags are still
there for debugging:

    python train.py --profile validate
    python train.py --profile smoke --from C0 --to C13 --resume
    python train.py --profile full --from C4 --to C4 --resume

L.4 requires this file to delegate rather than grow. It parses arguments,
bootstraps, and calls into `src/prism_fas/`; every decision it appears to make is
made in the package. The one thing it must own is the part that runs *before* the
project environment exists, so everything above the re-exec point is stdlib only
— a single third-party import here would break the entrypoint on exactly the host
it is meant to serve.

Zero-argument execution resolves one of two intents, and the difference is a
safety boundary rather than a convenience:

* **CPU_FULL_REHEARSAL** on a host with no GPU matching a declared profile. It
  exercises the real implementation and writes only to `reports/rehearsal` and
  `runs/rehearsal`. It is not scientifically eligible and cannot become a
  scientific ancestor.
* **GPU_SCIENTIFIC_FULL** on a host whose GPU matches one. It runs the frozen
  scientific pipeline from the first incomplete milestone through C13.

A rehearsal can never produce a Version-C P3 result, because it never reaches the
real target package at all.

Exit codes: 0 pass, 1 fail, 2 blocked, 3 usage or configuration error.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
EXIT_PASS, EXIT_FAIL, EXIT_BLOCKED, EXIT_USAGE = 0, 1, 2, 3

#: Set on the re-exec so a bootstrapped child never bootstraps again.
REEXEC_FLAG = "PRISM_BOOTSTRAPPED"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="PRISM-FAS-C-LLM C0-C13 orchestrator (v1.5 Appendix L). "
                    "Run with no arguments for the normal portable workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="With no arguments this bootstraps the environment, detects the "
               "hardware and runs the appropriate pipeline with resume enabled. "
               "Only --profile full produces scientifically eligible evidence.")
    parser.add_argument("--profile", choices=("validate", "smoke", "rehearsal", "full"),
                        help="execution profile (L.2). Omit to resolve automatically.")
    parser.add_argument("--resume", action="store_true",
                        help="identity-aware resume (L.11). Implied in zero-argument mode.")
    parser.add_argument("--from", dest="first_stage", metavar="Cx",
                        help="first stage to execute (debugging scope only)")
    parser.add_argument("--to", dest="last_stage", metavar="Cy",
                        help="last stage to execute (debugging scope only)")
    parser.add_argument("--phase", metavar="PHASE",
                        help="restrict to one L.5 phase (debugging scope only)")
    parser.add_argument("--mode", metavar="MODE",
                        help="stage mode where the adapter has several")
    parser.add_argument("--i-authorize-live-scientific-generation",
                        dest="authorized_live_generation", action="store_true",
                        help="authorize LIVE provider calls for C3 scientific generation. "
                             "Requires --profile full, a materialized quota snapshot and "
                             "a present credential. Without it every run is offline.")
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="use the current interpreter as-is; do not create or "
                             "re-exec into .venv")
    parser.add_argument("--preflight-only", action="store_true",
                        help="resolve the intent, print the preflight summary and stop "
                             "without executing anything")
    return parser


# --- everything above the re-exec is stdlib only -----------------------------

def _bootstrap_and_reexec(argv: list[str], *, quiet: bool) -> int | None:
    """Prepare the environment and hand control to the project interpreter.

    Returns an exit code when the run should stop here, or None when the caller
    should continue in-process (already inside the environment, or bootstrap
    explicitly disabled).
    """
    sys.path.insert(0, str(REPO))
    import bootstrap as boot

    if os.environ.get(REEXEC_FLAG) == "1" or boot.running_inside_project_venv():
        return None

    try:
        report = boot.ensure_environment(quiet=quiet)
    except boot.BootstrapError as error:
        print(f"\n[{error.reason}] {error}", file=sys.stderr)
        return EXIT_BLOCKED if error.reason == boot.CUDA_NOT_VALIDATED else EXIT_USAGE

    interpreter = Path(report["interpreter"])
    if not interpreter.exists():
        print(f"\n[{boot.BOOTSTRAP_FAILED}] the project interpreter is absent at "
              f"{interpreter}", file=sys.stderr)
        return EXIT_USAGE

    if report["action"] == "INSTALLED":
        print(f"  environment         INSTALLED  profile={report['profile_id']}  "
              f"id={report['environment_identity'][:16]}")
    environment = {**os.environ, REEXEC_FLAG: "1"}
    completed = subprocess.run([str(interpreter), str(REPO / "train.py"), *argv],
                               cwd=str(REPO), env=environment)
    return completed.returncode


def _zero_argument(args: argparse.Namespace) -> int:
    """The portable workflow: resolve, preflight, execute, summarize."""
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO))

    import bootstrap as boot
    from prism_fas.pipeline import runner
    from prism_fas.pipeline.orchestrator import run

    try:
        environment = boot.ensure_environment(quiet=True, allow_install=False)
    except boot.BootstrapError as error:
        print(f"\n[{error.reason}] {error}", file=sys.stderr)
        return EXIT_BLOCKED if error.reason == boot.CUDA_NOT_VALIDATED else EXIT_USAGE

    plan = runner.resolve(REPO, environment)
    print(runner.preflight_summary(REPO, plan, git_identity=_git_identity()))

    if not plan.ready:
        return EXIT_BLOCKED
    if args.preflight_only:
        print("\n  --preflight-only: nothing was executed.")
        return EXIT_PASS
    if plan.first_stage is None:
        print("\n  Nothing to execute: every milestone is scientifically complete.")
        return EXIT_PASS

    result = run(repo=REPO, profile_name=plan.profile_name, resume=True,
                 first_stage=plan.first_stage, last_stage=plan.last_stage)
    _print_stage_table(result)

    # Figures, tables, report and bundle, drawn from what the run just stored.
    # Always attempted: a run whose reporting failed should still say so rather
    # than finish silently and leave the operator to discover the gap.
    from prism_fas import reporting

    try:
        summary = reporting.generate(REPO, profile_name=plan.profile_name,
                                     execution_intent=plan.intent)
        print(f"\n  reporting           {summary['plots']['written_count']} plot(s), "
              f"{len(summary['tables']['tables'])} table(s), report + bundle")
        if summary["missing_evidence"]:
            print(f"  evidence gaps       {summary['missing_evidence']}")
    except Exception as error:                               # noqa: BLE001 - reported
        print(f"\n  reporting           FAILED: {type(error).__name__}: {error}",
              file=sys.stderr)

    print(runner.completion_summary(REPO, plan, result))
    return {"PASS": EXIT_PASS, "BLOCKED": EXIT_BLOCKED}.get(result.outcome, EXIT_FAIL)


def _explicit(args: argparse.Namespace) -> int:
    """The expert path: exactly the flags the operator asked for."""
    sys.path.insert(0, str(REPO / "src"))

    from prism_fas.pipeline.adapters import AdapterError, ProviderBinding
    from prism_fas.pipeline.orchestrator import OrchestratorError, run
    from prism_fas.pipeline.profiles import ProfileError
    from prism_fas.pipeline.stages import StageError

    binding = ProviderBinding.LIVE if args.authorized_live_generation else None
    try:
        result = run(repo=REPO, profile_name=args.profile, resume=args.resume,
                     first_stage=args.first_stage, last_stage=args.last_stage,
                     phase=args.phase, mode=args.mode, provider_binding=binding,
                     authorized_live_generation=args.authorized_live_generation)
    except (ProfileError, StageError, OrchestratorError, AdapterError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_USAGE

    profile = result.profile
    print(f"profile              {profile.name}  ({profile.config_path})")
    print(f"profile_identity     {profile.profile_identity}")
    print(f"scientific_eligible  {profile.scientific_eligible}")
    print(f"run_id               {result.run_id}")
    print(f"phase                {result.phase}")
    print()
    _print_stage_table(result)
    print()
    for path in result.written:
        print(f"  wrote {path}")
    print("  wrote state/PIPELINE_STATE.json")
    print("  wrote state/MASTER_RUN_INDEX.json")
    if result.blockers:
        print()
        for blocker in result.blockers:
            print(f"  BLOCKER {blocker}")
    print()
    print(f"outcome              {result.outcome}")
    if profile.name == "validate":
        print("meaning              engineering readiness evidence only. This run "
              "executed no\n                     scientific work and completes no "
              "milestone.")
    return {"PASS": EXIT_PASS, "BLOCKED": EXIT_BLOCKED}.get(result.outcome, EXIT_FAIL)


def _print_stage_table(result) -> None:
    for outcome in result.outcomes:
        adapter = "adapter" if outcome.stage.adapter_implemented else "no-adapter"
        counts = (f"{len(outcome.check_results) - len(outcome.failed_checks)}"
                  f"/{len(outcome.check_results)}")
        print(f"  {outcome.stage.stage_id:<4} {outcome.validate_gate:<15} "
              f"checks {counts:<7} {adapter:<11} "
              f"eng={outcome.status.engineering} sci={outcome.status.scientific}")
        for adapter_result in outcome.adapter_results:
            passed = len(adapter_result.checks) - len(adapter_result.failed_checks)
            print(f"         {adapter_result.substage:<5} {adapter_result.mode:<24}"
                  f" {adapter_result.status:<8} "
                  f"checks {passed}/{len(adapter_result.checks)}"
                  f"  binding={adapter_result.provider_binding.value}"
                  f"  provider_calls={adapter_result.provider_calls}")
            for failure in adapter_result.failed_checks:
                print(f"           FAILED {failure['check_id']}: {failure['summary']}")
        for failure in outcome.failed_checks:
            print(f"         FAILED {failure.check_id}: {failure.summary}")


def _git_identity() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(REPO), text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw)

    if not args.no_bootstrap:
        code = _bootstrap_and_reexec(raw, quiet=bool(args.profile))
        if code is not None:
            return code

    if args.profile is None:
        return _zero_argument(args)
    return _explicit(args)


if __name__ == "__main__":
    raise SystemExit(main())
