"""Quarantine an invalid C7 search state, after proving it is invalid.

    python scripts/recover_c7_invalid_search_state.py --track G            # inspect
    python scripts/recover_c7_invalid_search_state.py --track G --apply    # quarantine

Run this on the GPU host, deliberately, and read the inspection output before
passing `--apply`. It is not wired into `train.py` and never runs automatically:
clearing a search state is the one operation that can destroy scientific evidence,
so it is a decision a person takes with the eligibility report in front of them.

WHY IT EXISTS. The first real C7 GPU attempt left
`reports/full/c7/C7_SCIENTIFIC_SEARCH_STATE_G.json` in status COMPLETED with 15
FAIL rows and zero finite-valid trials, because the frozen recipe text cache was
missing from the host and every candidate raised the identical error before
training. Under `--resume` that state is a closed envelope, so the next run would
re-raise EnvelopeExhausted forever without training anything. It is not a
scientific result and it must not be treated as one — but it is also evidence
about how the run failed, so it is preserved rather than deleted.

WHAT IT REFUSES. Eligibility is proven, not assumed. The state must record no
finite-valid trial, every result row must share ONE typed global failure, no
DETECTOR_CONFIG_LOCK may exist, and the caller's expected search-plan identity
must match. Any real scientific outcome in the state — a single PASS, or FAIL
rows with differing causes — refuses the quarantine and prints why.

WHAT IT DOES NOT TOUCH. The C7 search arm (DET), the search envelope, the LR
decision, C6, C5. This is not a result-driven restart: there was no
configuration-specific scientific result to restart from.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

SCHEMA_VERSION = "prism-c7-search-state-recovery-v1"

#: Where a scientific C7 writes, and where a quarantined state is preserved.
C7_REPORTS = "reports/full/c7"
C7_RUNS = "runs/full/c7"
QUARANTINE_ROOT = "reports/evidence/quarantine"

#: Failure signatures this procedure recognises as GLOBAL — true of every
#: candidate, and therefore not a scientific outcome for any of them. Matched
#: against the exception TYPE name recorded in the trial notes, never against
#: free prose.
GLOBAL_FAILURE_TYPES: tuple[str, ...] = (
    "TextCacheError", "PretrainedError", "C6BankError", "C6EvidenceError",
    "DatasetError", "RegionCacheError", "SourceUnavailable",
    "DetectorInputsUnavailable", "ScientificDeviceUnavailable",
    "FatalDependencyError",
)

_TYPE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*Error|[A-Za-z_][A-Za-z0-9_]*Unavailable)")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_path(repo: Path, track: str) -> Path:
    from prism_fas.pipeline.adapters.c7 import _search_state_name

    return Path(repo) / C7_REPORTS / _search_state_name(track)


def _failure_types(row: dict[str, Any]) -> set[str]:
    """The exception type names a result row records, from its notes."""
    found: set[str] = set()
    for note in row.get("notes") or ():
        found.update(_TYPE.findall(str(note)))
    return found


def assess(repo: Path, track: str, *,
           expected_plan_identity: str | None = None) -> dict[str, Any]:
    """Is this state invalid because of ONE global input failure, and only that?"""
    from prism_fas.pipeline.adapters.c7 import (DETECTOR_CONFIG_LOCK, SCIENTIFIC_REPORTS,
                                                TRIAL_SUMMARY)

    repo = Path(repo)
    path = _state_path(repo, track)
    problems: list[str] = []

    if not path.is_file():
        return {"eligible": False, "state_present": False,
                "state": path.relative_to(repo).as_posix(),
                "problems": ["no search state exists for this track; nothing to "
                             "recover"], "results": 0}

    payload = json.loads(path.read_text(encoding="utf-8"))
    results = list(payload.get("results") or ())
    statuses = {str(row.get("status")) for row in results}
    finite_valid = [row for row in results if row.get("finite_valid")]
    passing = [row for row in results if str(row.get("status")) == "PASS"]

    types: set[str] = set()
    rows_without_global_type: list[str] = []
    for row in results:
        found = _failure_types(row) & set(GLOBAL_FAILURE_TYPES)
        if found:
            types |= found
        else:
            rows_without_global_type.append(str(row.get("config_id")))

    # --- the refusals --------------------------------------------------------
    if not results:
        problems.append("the state records no result rows at all")
    if finite_valid:
        problems.append(
            f"{len(finite_valid)} result row(s) are finite-valid: this state contains "
            "a real scientific outcome and may not be quarantined")
    if passing:
        problems.append(
            f"{len(passing)} result row(s) are PASS: a trained configuration is "
            "scientific evidence, not an engineering artifact")
    if rows_without_global_type:
        problems.append(
            f"{len(rows_without_global_type)} result row(s) record no recognised "
            f"global failure type, starting at {rows_without_global_type[:3]}. Rows "
            "with differing causes may be configuration-specific findings")
    if len(types) > 1:
        problems.append(
            f"result rows record more than one global failure type {sorted(types)}; "
            "a single global precondition failure is expected")
    if expected_plan_identity and payload.get("search_plan_identity") != expected_plan_identity:
        problems.append(
            f"state search_plan_identity {payload.get('search_plan_identity')!r} != "
            f"expected {expected_plan_identity!r}")

    lock = repo / SCIENTIFIC_REPORTS / DETECTOR_CONFIG_LOCK
    if lock.is_file():
        problems.append(
            f"{SCIENTIFIC_REPORTS}/{DETECTOR_CONFIG_LOCK} exists: a frozen detector "
            "configuration was produced and this state is not recoverable")

    # --- what training progress the artifacts actually prove -----------------
    summaries = sorted((repo / C7_RUNS / "scientific").rglob(TRIAL_SUMMARY))
    progress = _training_progress(repo, summaries)
    if progress["checkpoints_present"] or progress["rows_with_selection_metrics"]:
        problems.append(
            "trial artifacts show real training progress "
            f"(checkpoints={progress['checkpoints_present']}, "
            f"rows_with_metrics={progress['rows_with_selection_metrics']}); this is "
            "not a pure precondition failure")

    return {
        "eligible": not problems,
        "state_present": True,
        "state": path.relative_to(repo).as_posix(),
        "state_sha256": _sha256_file(path),
        "search_plan_identity": payload.get("search_plan_identity"),
        "state_status": payload.get("status"),
        "results": len(results),
        "result_statuses": {value: sum(1 for row in results
                                       if str(row.get("status")) == value)
                            for value in sorted(statuses)},
        "finite_valid": len(finite_valid),
        "global_failure_types": sorted(types),
        "completed_coordinates": list(payload.get("completed_coordinates") or ()),
        "trial_summaries_on_disk": len(summaries),
        "logical_result_rows": len(results),
        "unique_config_sha256_in_state": len(
            {str(row.get("config_sha256")) for row in results}),
        "training_progress": progress,
        "detector_config_lock_present": lock.is_file(),
        "problems": problems,
    }


def _training_progress(repo: Path, summaries: list[Path]) -> dict[str, Any]:
    """What the trial artifacts PROVE about how far training actually got.

    Asserted from the artifacts rather than assumed from the failure message: a
    precondition failure should leave no checkpoint, no completed stage and no
    source-selection metric, and if any of those exist the boundary was further
    along than the message suggests.
    """
    checkpoints = 0
    with_metrics = 0
    completed_stages: set[str] = set()
    phases: dict[str, int] = {}
    for path in summaries:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if record.get("checkpoint_sha256"):
            checkpoints += 1
        if record.get("selection_metrics"):
            with_metrics += 1
        for stage, item in dict((record.get("flow") or {}).get("stages") or {}).items():
            if str(dict(item).get("status")) == "COMPLETED":
                completed_stages.add(str(stage))
        phase = str(record.get("failure_phase", "unrecorded"))
        phases[phase] = phases.get(phase, 0) + 1
    return {
        "summaries_read": len(summaries),
        "checkpoints_present": checkpoints,
        "rows_with_selection_metrics": with_metrics,
        "completed_training_stages": sorted(completed_stages),
        "failure_phases": phases,
        "optimizer_step_evidence": bool(checkpoints or completed_stages),
    }


def quarantine(repo: Path, track: str, report: dict[str, Any], *,
               reason: str) -> dict[str, Any]:
    """Preserve the invalid state and its artifacts, then clear ONLY the state."""
    from prism_fas.pipeline.adapters.c7 import TRIAL_SUMMARY
    from prism_fas.pipeline.state import atomic_write_json

    repo = Path(repo)
    stamp = _utc().replace(":", "").replace("-", "")
    root = repo / QUARANTINE_ROOT / f"c7_search_state_{track.lower()}_{stamp}"
    root.mkdir(parents=True, exist_ok=True)

    state = _state_path(repo, track)
    preserved: list[dict[str, Any]] = []

    shutil.copy2(state, root / state.name)
    preserved.append({"artifact": state.relative_to(repo).as_posix(),
                      "preserved_as": (root / state.name).relative_to(repo).as_posix(),
                      "sha256": _sha256_file(state)})

    summaries_root = root / "trial_summaries"
    for path in sorted((repo / C7_RUNS / "scientific").rglob(TRIAL_SUMMARY)):
        relative = path.relative_to(repo / C7_RUNS / "scientific")
        destination = summaries_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        preserved.append({"artifact": path.relative_to(repo).as_posix(),
                          "preserved_as": destination.relative_to(repo).as_posix(),
                          "sha256": _sha256_file(path)})

    for candidate in sorted(repo.glob("gpu_c7_*.log")):
        destination = root / candidate.name
        shutil.copy2(candidate, destination)
        preserved.append({"artifact": candidate.name,
                          "preserved_as": destination.relative_to(repo).as_posix(),
                          "sha256": _sha256_file(candidate)})

    record = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc(),
        "track": track,
        "classification": "ENGINEERING_GLOBAL_INPUT_FAILURE",
        "scientific_negative_result": False,
        "scientific_envelope_exhausted": False,
        "scientific_candidate_quality_observed": False,
        "candidates_consumed": 0,
        "reason": reason,
        "eligibility": report,
        "preserved": preserved,
        "preserved_count": len(preserved),
        "cleared": [],
        "not_touched": [
            "configs/search/c7_source_search_decision.yaml (arm = DET)",
            "configs/search/lr_anchor_decision.yaml",
            "the §15.2.2 search envelope",
            "reports/full/c6 (C6 is closed)",
            "runs/full/c5/scientific/candidates (C5 is closed)",
        ],
        "target_access": 0,
        "immutability": {"rewrite_permitted": False},
    }

    # Cleared LAST, and only the active execution state. Trial artifacts stay in
    # place: they are addressable engineering evidence, and the next pass writes
    # under identities derived from a plan whose identity is unchanged.
    state.unlink()
    record["cleared"] = [state.relative_to(repo).as_posix()]
    atomic_write_json(root / "RECOVERY_RECORD.json", record)
    return {**record, "quarantine_root": root.relative_to(repo).as_posix()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", required=True, choices=("G", "R"))
    parser.add_argument("--apply", action="store_true",
                        help="quarantine and clear; without it, only inspect")
    parser.add_argument("--expect-plan-identity", default=None,
                        help="refuse unless the state records this search plan")
    parser.add_argument("--reason",
                        default="the frozen recipe text cache was absent from the "
                                "host, so every candidate failed identically before "
                                "training and no configuration-specific scientific "
                                "result exists")
    args = parser.parse_args()

    report = assess(REPO, args.track,
                    expected_plan_identity=args.expect_plan_identity)
    print(json.dumps(report, indent=2, sort_keys=True))

    if not report["eligible"]:
        print("\nNOT ELIGIBLE — nothing was changed.")
        for problem in report["problems"]:
            print(f"  - {problem}")
        return 1
    if not args.apply:
        print("\nELIGIBLE. Re-run with --apply to preserve and clear.")
        return 0

    result = quarantine(REPO, args.track, report, reason=args.reason)
    print(f"\npreserved {result['preserved_count']} artifact(s) under "
          f"{result['quarantine_root']}")
    print(f"cleared: {result['cleared']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
