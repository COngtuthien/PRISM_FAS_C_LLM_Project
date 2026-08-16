#!/usr/bin/env python
"""The canonical root orchestration entrypoint (v1.5 Appendix L.4).

    python train.py --profile validate
    python train.py --profile smoke --resume
    python train.py --profile full --resume

The name is historical. This is an ORCHESTRATOR for the complete C0-C13
workflow, and L.4 requires it to delegate to importable modules under
`src/prism_fas/` rather than grow into a monolith that duplicates recipe, GPAT,
synthesis, detector, evaluation or lock logic. So this file parses arguments,
calls `prism_fas.pipeline.orchestrator.run`, and prints what happened. Every
decision it appears to make is made in the package.

`--from`, `--to` and `--phase` change debugging scope only. L.4 forbids them
from changing scientific data, model topology, loss semantics, search space,
seed family, selection rule, target accessibility or acceptance criteria, and a
partial range cannot by itself produce a full-pipeline acceptance.

Exit codes: 0 for a passing run, 1 for a failing one, 2 for a blocked one and 3
for a usage or configuration error. A blocked run is distinguished from a failing
one because "this cannot run yet" and "this ran and was wrong" call for
different responses.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prism_fas.pipeline.adapters import AdapterError  # noqa: E402
from prism_fas.pipeline.orchestrator import OrchestratorError, run  # noqa: E402
from prism_fas.pipeline.profiles import PROFILE_NAMES, ProfileError  # noqa: E402
from prism_fas.pipeline.stages import STAGE_IDS, StageError  # noqa: E402

EXIT_PASS, EXIT_FAIL, EXIT_BLOCKED, EXIT_USAGE = 0, 1, 2, 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="PRISM-FAS-C-LLM C0-C13 orchestrator (v1.5 Appendix L).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Only --profile full produces scientifically eligible evidence, and only "
               "through the frozen source-only selector before source freeze.")
    parser.add_argument("--profile", required=True, choices=PROFILE_NAMES,
                        help="execution profile (L.2)")
    parser.add_argument("--resume", action="store_true",
                        help="identity-aware resume; a completed artifact is skipped only "
                             "after its parents, config identity, content hash and "
                             "acceptance state validate (L.11)")
    parser.add_argument("--from", dest="first_stage", metavar="Cx", choices=STAGE_IDS,
                        help="first stage to execute (debugging scope only)")
    parser.add_argument("--to", dest="last_stage", metavar="Cy", choices=STAGE_IDS,
                        help="last stage to execute (debugging scope only)")
    parser.add_argument("--phase", metavar="PHASE",
                        help="restrict to one L.5 phase (debugging scope only)")
    parser.add_argument("--mode", metavar="MODE",
                        help="stage mode where the adapter has several (C3: "
                             "PRE_LIVE_VERIFY, LIVE_GENERATE, RESUME_LIVE_GENERATE, "
                             "FINALIZE_BANKS)")
    # Deliberately verbose and deliberately not a boolean shorthand. This flag is
    # the human authorization that lets C3 spend live provider quota on the
    # frozen 12x32 schedule; it must be impossible to type by accident and
    # obvious in a shell history.
    parser.add_argument("--i-authorize-live-scientific-generation",
                        dest="authorized_live_generation", action="store_true",
                        help="authorize LIVE provider calls for C3 scientific generation. "
                             "Requires --profile full, a materialized quota snapshot and a "
                             "present credential. Without it every run is offline.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        binding = None
        if args.authorized_live_generation:
            # The flag is the authorization; binding LIVE here makes the intent
            # explicit rather than leaving it to a default.
            from prism_fas.pipeline.adapters import ProviderBinding

            binding = ProviderBinding.LIVE

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

    for outcome in result.outcomes:
        adapter = "adapter" if outcome.stage.adapter_implemented else "no-adapter"
        counts = (f"{len(outcome.check_results) - len(outcome.failed_checks)}"
                  f"/{len(outcome.check_results)}")
        print(f"  {outcome.stage.stage_id:<4} {outcome.validate_gate:<15} "
              f"checks {counts:<7} {adapter:<11} "
              f"eng={outcome.status.engineering} sci={outcome.status.scientific}")
        for adapter_result in outcome.adapter_results:
            passed = len(adapter_result.checks) - len(adapter_result.failed_checks)
            print(f"         {adapter_result.substage:<5} {adapter_result.mode:<22}"
                  f" {adapter_result.status:<8} checks {passed}/{len(adapter_result.checks)}"
                  f"  binding={adapter_result.provider_binding.value}"
                  f"  provider_calls={adapter_result.provider_calls}")
            for failure in adapter_result.failed_checks:
                print(f"           FAILED {failure['check_id']}: {failure['summary']}")
        for failure in outcome.failed_checks:
            print(f"         FAILED {failure.check_id}: {failure.summary}")

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
        print("meaning              engineering readiness evidence only. This run executed "
              "no\n                     scientific work and completes no milestone.")

    if result.outcome == "PASS":
        return EXIT_PASS
    if result.outcome == "BLOCKED":
        return EXIT_BLOCKED
    return EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
