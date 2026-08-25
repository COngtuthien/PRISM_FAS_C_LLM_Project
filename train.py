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

The interpreter that runs this file is not necessarily the interpreter that runs
the project. On Windows, PATH `python` may be MSYS2/MinGW Python, which builds a
POSIX-scheme environment and cannot host the CUDA wheels; `bootstrap` classifies
it, finds a supported standard Windows CPython through the Python Launcher and
builds `.venv` with that instead. The command the operator types never changes.

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
    parser.add_argument("--diagnose-data", action="store_true",
                        help="print a read-only forensic view of the derived-data "
                             "trees on this machine (M2 namespaces, manifest row "
                             "counts, crop counts, packages) and stop. Builds "
                             "nothing, deletes nothing, reads no target.")
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
        return EXIT_BLOCKED if error.reason in boot.BLOCKING_REASONS else EXIT_USAGE

    interpreter = Path(report["interpreter"])
    if not interpreter.exists():
        print(f"\n[{boot.BOOTSTRAP_FAILED}] the project interpreter is absent at "
              f"{interpreter}", file=sys.stderr)
        return EXIT_USAGE

    host = report.get("host_interpreter") or {}
    if host.get("fallback"):
        # The operator typed `python train.py` and got a different interpreter.
        # Saying so is the difference between a runner that works and a runner
        # that appears to ignore the environment it was started in.
        fallback = host["fallback"]
        print(f"  host interpreter    {fallback['from_classification']} at "
              f"{fallback['from']}")
        print(f"  using instead       standard Windows CPython "
              f"{fallback.get('to_version')} at {fallback['to']}")
    recovery = report.get("venv_recovery") or {}
    if recovery.get("rebuilt"):
        print(f"  environment         REBUILT  ({recovery.get('state')}: "
              f"{recovery.get('why')})")
    if report["action"] == "INSTALLED":
        print(f"  environment         INSTALLED  profile={report['profile_id']}  "
              f"id={report['environment_identity'][:16]}")
    sys.stdout.flush()          # the child writes to the same stream
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
        return EXIT_BLOCKED if error.reason in boot.BLOCKING_REASONS else EXIT_USAGE

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

    # On a scientific host, prove the GPU can actually train before starting C4.
    # Selecting a wheel from nvidia-smi shows a driver exists; it does not show
    # that a kernel launches, that autograd runs or that a checkpoint round-trips.
    # This is the same command, not a second one the operator must remember.
    from prism_fas.pipeline import gpu_preflight

    try:
        report = gpu_preflight.run_preflight(REPO, strict=plan.is_scientific)
        path = gpu_preflight.write_report(REPO, report)
        if report["applicable"]:
            device = report["device"]
            print(f"\n  GPU preflight       PASS  {report['probes_run']} probe(s) in "
                  f"{report['elapsed_seconds']}s")
            print(f"  Device              {device['gpu_name']}  cc {device['compute_capability']}"
                  f"  {device['total_memory_mb']} MB")
            print(f"  Report              {path.relative_to(REPO).as_posix()}")
    except gpu_preflight.GPUPreflightError as error:
        print(f"\n[{error.reason}] {error}", file=sys.stderr)
        print("\n  Stopped BEFORE C4. No scientific work was started.", file=sys.stderr)
        return EXIT_BLOCKED

    # Build any missing derived data tree from the raw datasets that travelled in
    # the folder. Deterministic, resumable, and delegated to the canonical
    # builders — this is the step that used to be the collaborator's homework.
    from prism_fas.pipeline import preparation

    try:
        # A rehearsal runs on fixtures and needs no derived tree, so it only
        # REPORTS what a scientific run would have to build. Building hours of
        # preprocessing that the rehearsal will not read would be pure waste.
        prepared = preparation.prepare(REPO, resume=True,
                                       dry_run=not plan.is_scientific)
        preparation.write_report(REPO, prepared)
        if prepared["outcome"] not in ("NOTHING_TO_DO",):
            print(f"\n  Derived data        {prepared['outcome']}  {prepared['summary']}")
            for step in prepared["steps"]:
                print(f"      {step['step']:16s} {step['action']:18s} {step['seconds']}s")
        if prepared["outcome"] == "BLOCKED" and plan.is_scientific:
            print(f"\n[{prepared['reason_code']}] {prepared['summary']}", file=sys.stderr)
            print("\n  Stopped BEFORE C4. No scientific work was started.",
                  file=sys.stderr)
            return EXIT_BLOCKED
    except preparation.PreparationError as error:
        print(f"\n[{error.reason}] {error}", file=sys.stderr)
        print("\n  Stopped BEFORE C4. No scientific work was started.", file=sys.stderr)
        return EXIT_BLOCKED

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
                     authorized_live_generation=args.authorized_live_generation,
                     preflight_only=args.preflight_only)
    except (ProfileError, StageError, OrchestratorError, AdapterError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_USAGE

    profile = result.profile
    print(f"profile              {profile.name}  ({profile.config_path})")
    print(f"profile_identity     {profile.profile_identity}")
    print(f"scientific_eligible  {profile.scientific_eligible}")
    print(f"run_id               {result.run_id}")
    print(f"phase                {result.phase}")
    if args.preflight_only:
        print("preflight_only       true  (workflow() was never called for any stage; "
              "see below)")
    print()
    _print_stage_table(result)
    print()
    if args.preflight_only:
        # Genuinely nothing was written: orchestrator.run() returns before
        # _write_reports / record / write_state when preflight_only is set.
        print("  --preflight-only: nothing was executed and nothing was written "
              "(no reports/full/*, no state/PIPELINE_STATE.json, no "
              "state/MASTER_RUN_INDEX.json).")
    else:
        for path in result.written:
            print(f"  wrote {path}")
        print("  wrote state/PIPELINE_STATE.json")
        print("  wrote state/MASTER_RUN_INDEX.json")
    if result.blockers:
        print()
        for blocker in result.blockers:
            print(f"  BLOCKER {blocker}")
    print()
    if args.preflight_only:
        verdict = "PASS" if result.outcome == "PASS" else "BLOCKED"
        print(f"preflight            {verdict}")
        print("outcome              " + result.outcome +
              "  (preflight-only: no scientific work of any kind was started)")
    else:
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


def _diagnose_data() -> int:
    """Say what derived data this machine actually has, and where.

    A remote host's partial layout cannot be assumed from here, and the operator
    should never have to guess which of two M2 namespaces their hours went into.
    Read-only: it builds nothing, deletes nothing and resolves no target.
    """
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO))

    import json

    from prism_fas.pipeline import preparation

    report = preparation.diagnose(REPO)
    path = REPO / "reports" / "preflight" / "DERIVED_DATA_DIAGNOSIS.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if report.get("error"):
        print(f"\n  diagnosis unavailable: {report['error']}", file=sys.stderr)
        return EXIT_USAGE

    legacy, full = report["legacy_m2a"], report["full_preprocessing"]
    status = report["m2_status"]
    print(f"\n  M2 config root      {report['m2_config_root']}")
    print(f"  namespaces present  {', '.join(report['namespaces']) or 'none'}")
    print(f"\n  legacy m2a          {'present' if legacy['present'] else 'absent'}  "
          f"{legacy['crops']} crop(s), results {legacy['result_files'] or 'none'}")
    print("  reusable as M2      NO: it holds JSONL results in the frozen "
          "acceptance namespace, not")
    print("                      the canonical parquet manifests M3A reads. "
          "Left untouched.")
    print(f"\n  full_preprocessing  {'present' if full['present'] else 'absent'}  "
          f"{full['crops_on_disk']} crop(s)")
    print(f"  manifests           {full['manifests_root']}")
    for name, rows in full["manifests"].items():
        print(f"      {name:24s} {'absent' if rows is None else str(rows) + ' row(s)'}")
    print(f"  completion marker   {'present' if full['completion_marker'] else 'absent'}")
    print(f"  M2 status           {'COMPLETE' if status['complete'] else status['reason']}")
    if status.get("outstanding_records"):
        print(f"  records outstanding {status['outstanding_records']}")
    for name, package in report["packages"].items():
        # Never just "locked": a package whose lock says `building` is present,
        # locked, and not scientific input. Saying only "locked" is what hid the
        # M3B lifecycle defect until C4 refused the package three steps later.
        if not package["present"]:
            print(f"  {name:24s} absent")
            continue
        verdict = "REUSABLE" if package["reusable_as_scientific_input"] else "NOT USABLE"
        print(f"  {name:24s} status={package['status'] or 'unlocked'}  "
              f"validation={package['package_validation'] or 'none'}  "
              f"{package['files']} file(s)")
        print(f"  {'':24s} identity={(package['content_identity_sha256'] or 'none')[:16]}  "
              f"scientific input: {verdict}")
        if package["why"]:
            print(f"  {'':24s} {package['why']}")
    pairs = report["gpat_pairs"]
    print(f"  gpat_pairs               "
          f"{'locked' if pairs['locked'] else 'absent'}  "
          f"reusable: {'YES' if pairs['reusable'] else 'NO'}")
    if pairs["locked"] and pairs["why"]:
        print(f"  {'':24s} {pairs['why']}")
    print(f"\n  data/processed      {report['data_processed']['role']}")
    print(f"  Report              {path.relative_to(REPO).as_posix()}")
    print("\n  Nothing was built, deleted or read from the target.")
    return EXIT_PASS


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw)

    if not args.no_bootstrap:
        code = _bootstrap_and_reexec(raw, quiet=bool(args.profile))
        if code is not None:
            return code

    if args.diagnose_data:
        return _diagnose_data()
    if args.profile is None:
        return _zero_argument(args)
    return _explicit(args)


if __name__ == "__main__":
    raise SystemExit(main())
