"""The C0-C13 orchestrator (v1.5 Appendix L.4, L.5, L.8).

`train.py` is the human-facing name; this is where the work is decided. L.4 is
explicit that the entrypoint must delegate to importable modules and must not
absorb recipe, GPAT, synthesis, detector, evaluation or lock logic, so this
module only sequences stages, records what happened and stops when it should.

Three refusals are built in, and each exists because the quiet alternative would
be worse than a failure:

* **No adapter, no pass.** Every stage in `stages.py` currently declares
  ``adapter_implemented=False``. Smoke and full therefore cannot run, and the
  orchestrator says so and exits BLOCKED rather than traversing fourteen stages
  that do nothing and reporting success.
* **No validate result is an engineering pass.** L.3 fixes the engineering
  vocabulary at NOT_TESTED / RUNNING / SMOKE_PASS / SMOKE_FAIL / BLOCKED. None
  of those means "validate passed", so a clean validate run leaves
  ``engineering_status=NOT_TESTED`` and reports its outcome on a separate
  ``validate_gate`` axis. Reusing SMOKE_PASS would claim a smoke execution that
  never happened; inventing a sixth value would edit a frozen vocabulary.
* **No profile promotion.** Every artifact is stamped from the profile itself,
  and `assert_not_promoted` refuses a non-full artifact that claims scientific
  eligibility.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prism_fas.pipeline import checks as check_module
from prism_fas.pipeline.budget import BudgetLedger, ledger_for
from prism_fas.pipeline.profiles import ExecutionProfile, load_profile
from prism_fas.pipeline.registry import RunRecord, annotate_historical, record
from prism_fas.pipeline.stages import STAGES, Stage, stage_slice
from prism_fas.pipeline.state import PipelineState, write_state
from prism_fas.pipeline.status import (NEEDS_SCIENTIFIC_DECISION, DualStatus,
                                       assert_not_promoted, evidence_eligibility,
                                       scientifically_complete)

RUN_SCHEMA_VERSION = "prism-orchestrator-run-v1"

#: The validate profile reports on its own axis rather than borrowing one of the
#: five L.3 engineering values. See the module docstring.
VALIDATE_GATE = ("PASS", "FAIL", "NOT_APPLICABLE")

#: Historical acceptance files written before L.9 existed. They are annotated in
#: the master index, never edited: their bytes are inputs to frozen identities.
HISTORICAL_ACCEPTANCE = (
    ("C0", "reports/c0/C0_ACCEPTANCE.json"),
    ("C1", "reports/c1/C1_ACCEPTANCE.json"),
    ("C2", "reports/c2/C2_ACCEPTANCE.json"),
    ("C2B", "reports/c2b/C2B_ACCEPTANCE.json"),
    ("C2C", "reports/c2c/C2C_ACCEPTANCE.json"),
    ("C3", "reports/c3/v15_selection_contract/C3_SELECTION_CONTRACT_ACCEPTANCE.json"),
)


class OrchestratorError(RuntimeError):
    """The requested execution cannot proceed as asked."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class StageOutcome:
    """What one stage produced in one orchestration run."""

    stage: Stage
    status: DualStatus
    validate_gate: str
    check_results: list[check_module.CheckResult] = field(default_factory=list)
    started_at_utc: str = ""
    finished_at_utc: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def failed_checks(self) -> list[check_module.CheckResult]:
        return [result for result in self.check_results if not result.ok]

    def as_dict(self, profile: ExecutionProfile) -> dict[str, Any]:
        payload = {
            "schema_version": RUN_SCHEMA_VERSION,
            **profile.stamp(),
            # As in the run summary: the profile permits, the outcome earns.
            "scientific_eligible": evidence_eligibility(
                profile_eligible=profile.scientific_eligible,
                outcome="PASS" if self.validate_gate == "PASS" else self.validate_gate),
            "profile_permits_scientific_evidence": profile.scientific_eligible,
            **self.stage.as_dict(),
            **self.status.as_dict(),
            "validate_gate": self.validate_gate,
            "scientifically_complete": scientifically_complete(self.status,
                                                               profile=profile.name),
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "checks_run": len(self.check_results),
            "checks_passed": len(self.check_results) - len(self.failed_checks),
            "checks_failed": len(self.failed_checks),
            "checks": [result.as_dict() for result in self.check_results],
            "notes": list(self.notes),
        }
        assert_not_promoted(payload)
        return payload


@dataclass
class RunResult:
    """The whole orchestration run, and whether it succeeded."""

    profile: ExecutionProfile
    phase: str
    outcomes: list[StageOutcome]
    outcome: str
    started_at_utc: str
    finished_at_utc: str
    run_id: str
    ledger: BudgetLedger
    blockers: list[str] = field(default_factory=list)
    written: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.outcome == "PASS"


def _validate_stage(stage: Stage, repo: Path) -> StageOutcome:
    """Run one stage's declared readiness checks.

    A stage with no declared checks is reported as NOT_APPLICABLE rather than
    PASS. That distinction is the whole reason C4-C13 cannot make a validate run
    look complete.
    """
    started = _utc()
    results = [check_module.run_check(check_id, repo) for check_id in stage.validate_checks]

    if not stage.validate_checks:
        gate = "NOT_APPLICABLE"
        notes = [f"no validate checks are declared for {stage.stage_id}; "
                 f"{stage.adapter_note}"]
    else:
        gate = "PASS" if all(result.ok for result in results) else "FAIL"
        notes = []

    # L.3: validate execution never produces SMOKE_PASS. The engineering axis
    # stays NOT_TESTED until a smoke run actually executes the code path.
    status = DualStatus(engineering="NOT_TESTED", scientific="NOT_RUN")
    return StageOutcome(stage=stage, status=status, validate_gate=gate,
                        check_results=results, started_at_utc=started,
                        finished_at_utc=_utc(), notes=notes)


def _blocked_stage(stage: Stage, reason: str) -> StageOutcome:
    now = _utc()
    return StageOutcome(stage=stage,
                        status=DualStatus(engineering="BLOCKED", scientific="BLOCKED"),
                        validate_gate="NOT_APPLICABLE", started_at_utc=now,
                        finished_at_utc=now, notes=[reason])


def stage_artifact_path(profile: ExecutionProfile, stage: Stage) -> Path:
    """Where one stage's artifact lives, named for the profile that wrote it.

    The profile is in the filename because these files get read out of context
    later, and `C3_FULL.json` sitting next to `C3_SMOKE.json` is the difference
    between evidence and a fixture.
    """
    return (Path(profile.reports_namespace) / stage.stage_id.lower()
            / f"{stage.stage_id}_{profile.name.upper()}.json")


def _write_reports(repo: Path, result: RunResult) -> list[str]:
    """Per-stage artifacts plus one summary, all under the profile's namespace.

    L.8 requires every milestone to emit its own durable artifact before the
    orchestrator advances — but only for a stage that actually ran. A blocked
    run executed nothing, so it writes its summary and no per-stage files: a
    `reports/full/` tree full of stubs would read like scientific work had
    started, which is precisely the confusion the dual-status model exists to
    prevent. The blocked outcome itself is preserved, in the summary and in the
    master index (L.8).
    """
    from prism_fas.pipeline.state import atomic_write_json

    profile = result.profile
    base = profile.reports_dir(repo)
    written: list[str] = []

    if result.outcome != "BLOCKED":
        for outcome in result.outcomes:
            path = repo / stage_artifact_path(profile, outcome.stage)
            atomic_write_json(path, outcome.as_dict(profile))
            written.append(path.relative_to(repo).as_posix())

    summary: dict[str, Any] = {
        "schema_version": RUN_SCHEMA_VERSION,
        **profile.stamp(),
        # The profile's declaration is what it PERMITS; this run's outcome is
        # what it EARNED. A blocked full run permits eligibility and earns none.
        "scientific_eligible": evidence_eligibility(
            profile_eligible=profile.scientific_eligible, outcome=result.outcome),
        "profile_permits_scientific_evidence": profile.scientific_eligible,
        "run_id": result.run_id,
        "phase": result.phase,
        "outcome": result.outcome,
        "started_at_utc": result.started_at_utc,
        "finished_at_utc": result.finished_at_utc,
        "artifact_kind": "ENGINEERING_READINESS_EVIDENCE",
        "not_scientific_evidence": True,
        "meaning": (
            "This run measured static readiness only: identities, locks, contracts and "
            "environment. It executed no scientific work, and it is not evidence that any "
            "milestone can run, let alone that one is scientifically complete."),
        "stages_total": len(result.outcomes),
        "stages_with_validate_checks": sum(
            1 for outcome in result.outcomes if outcome.stage.validate_checks),
        "stages_without_adapter": [
            outcome.stage.stage_id for outcome in result.outcomes
            if not outcome.stage.adapter_implemented],
        "validate_gate_counts": {
            gate: sum(1 for outcome in result.outcomes if outcome.validate_gate == gate)
            for gate in VALIDATE_GATE},
        "checks_run": sum(len(outcome.check_results) for outcome in result.outcomes),
        "checks_failed": sum(len(outcome.failed_checks) for outcome in result.outcomes),
        "failed_check_ids": [failed.check_id for outcome in result.outcomes
                             for failed in outcome.failed_checks],
        "blockers": list(result.blockers),
        "budget": result.ledger.as_dict(),
        "stages": [
            {**outcome.stage.as_dict(), **outcome.status.as_dict(),
             "validate_gate": outcome.validate_gate,
             "checks_failed": len(outcome.failed_checks)}
            for outcome in result.outcomes],
        "engineering_status_note": (
            "L.3 fixes the engineering vocabulary at NOT_TESTED / RUNNING / SMOKE_PASS / "
            "SMOKE_FAIL / BLOCKED. None of those means 'validate passed', so a clean "
            "validate run leaves engineering_status=NOT_TESTED and reports on the separate "
            "validate_gate axis instead of borrowing SMOKE_PASS."),
        "per_stage_artifacts_written": result.outcome != "BLOCKED",
        "per_stage_artifacts_note": (
            "a blocked run executes no stage, so it writes no per-stage artifact; the "
            "blocked outcome is preserved here and in state/MASTER_RUN_INDEX.json (L.8)"),
        "artifacts": written,
    }
    assert_not_promoted(summary)
    summary_path = base / f"{profile.name.upper()}_RUN.json"
    atomic_write_json(summary_path, summary)
    written.append(summary_path.relative_to(repo).as_posix())
    return written


def run(*, repo: Path, profile_name: str, resume: bool = False,
        first_stage: str | None = None, last_stage: str | None = None,
        phase: str | None = None) -> RunResult:
    """Execute one orchestration run and return its result.

    Smoke and full stop immediately while no stage adapter exists. That is not a
    placeholder: running them would mean traversing the phase machine with
    nothing to execute and writing artifacts that assert stages completed. A
    BLOCKED result naming the missing adapters is the accurate outcome.
    """
    repo = Path(repo).resolve()
    profile = load_profile(profile_name, repo=repo)
    ledger = ledger_for(profile)
    run_id = f"{profile.name}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    started = _utc()

    selected = stage_slice(first=first_stage, last=last_stage)
    if phase:
        selected = tuple(stage for stage in selected if stage.phase == phase)
        if not selected:
            raise OrchestratorError(
                f"no stage in the requested range belongs to phase {phase!r}")

    blockers: list[str] = []
    outcomes: list[StageOutcome] = []

    missing_adapters = [stage.stage_id for stage in STAGES if not stage.adapter_implemented]
    if profile.name in {"smoke", "full"}:
        blockers.append(
            f"profile {profile.name!r} executes stage adapters, and none exists yet "
            f"(missing: {', '.join(missing_adapters)}). Stage adapters are step 4 of "
            "docs/V15_PIPELINE_RESTRUCTURE_PLAN.md and are separately authorized.")
        if profile.name == "full":
            blockers.append(
                "C3 scientific generation additionally remains gated on user approval of "
                "the superseding bank-contract lock; the full profile cannot start while "
                f"that approval is outstanding ({NEEDS_SCIENTIFIC_DECISION}).")
        outcomes = [_blocked_stage(stage, blockers[0]) for stage in selected]
        outcome = "BLOCKED"
    else:
        for stage in selected:
            outcomes.append(_validate_stage(stage, repo))
            ledger.spend("stages")
        failed = [item for item in outcomes if item.validate_gate == "FAIL"]
        outcome = "FAIL" if failed else "PASS"
        if failed:
            blockers.extend(
                f"{item.stage.stage_id}: " + "; ".join(
                    failure.summary for failure in item.failed_checks)
                for item in failed)

    finished = _utc()
    result = RunResult(profile=profile, phase=phase or "preflight", outcomes=outcomes,
                       outcome=outcome, started_at_utc=started, finished_at_utc=finished,
                       run_id=run_id, ledger=ledger, blockers=blockers)

    result.written = _write_reports(repo, result)

    def _row_status(item: StageOutcome) -> str:
        if outcome == "BLOCKED":
            return "BLOCKED"
        return {"PASS": "PASS", "FAIL": "FAIL",
                "NOT_APPLICABLE": "SKIPPED_VALID"}[item.validate_gate]

    rows = [
        RunRecord(
            run_id=f"{run_id}:{item.stage.stage_id}",
            stage_id=item.stage.stage_id,
            execution_profile=profile.name,
            scientific_eligible=evidence_eligibility(
                profile_eligible=profile.scientific_eligible,
                outcome=_row_status(item)),
            status=_row_status(item),
            started_at_utc=item.started_at_utc,
            finished_at_utc=item.finished_at_utc,
            paths=() if outcome == "BLOCKED" else
                  (stage_artifact_path(profile, item.stage).as_posix(),),
            notes="; ".join(item.notes),
            extra={"validate_gate": item.validate_gate,
                   "adapter_implemented": item.stage.adapter_implemented,
                   **item.status.as_dict()},
        )
        for item in outcomes
    ]
    annotations = [
        annotate_historical(path, milestone=milestone,
                            reason="written before the L.9 dual-status artifact schema "
                                   "existed; editing it would change bytes that frozen "
                                   "identities were computed over")
        for milestone, path in HISTORICAL_ACCEPTANCE if (repo / path).exists()
    ]
    record(repo, rows, profile=profile, generated_at_utc=finished,
           historical_annotations=annotations)

    write_state(repo, PipelineState(
        profile=profile.name,
        phase=result.phase,
        stage_id=outcomes[-1].stage.stage_id if outcomes else None,
        stage_index=len(outcomes),
        completed_stages=[item.stage.stage_id for item in outcomes
                          if item.validate_gate == "PASS"],
        blocked_stages=[item.stage.stage_id for item in outcomes
                        if item.status.engineering == "BLOCKED"],
        last_run_id=run_id,
        last_updated_utc=finished,
        outcome=outcome,
        notes=blockers,
        extra={"resume_requested": resume,
               "resume_note": "validate checks are measurements of current repository "
                              "state and are always re-executed; there is no completed "
                              "artifact for --resume to skip (L.11 applies to executed "
                              "work, not to measurement)"},
    ), profile)

    return result


__all__ = ["RUN_SCHEMA_VERSION", "VALIDATE_GATE", "HISTORICAL_ACCEPTANCE",
           "OrchestratorError", "StageOutcome", "RunResult", "stage_artifact_path", "run"]
