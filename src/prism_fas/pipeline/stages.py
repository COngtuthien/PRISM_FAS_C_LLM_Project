"""The C0-C13 stage declarations (v1.5 Appendix L.9).

This module declares what each stage IS and what the full profile owes for it.
It deliberately does not execute anything, and it deliberately does not pretend
that a stage exists because it is listed here.

Two flags carry that honesty:

* ``adapter_implemented`` — whether a stage adapter exists that can execute this
  milestone under the smoke or full profile. Today every one of them is False;
  the adapters are a separately authorized step of the restructure plan. A
  stage with no adapter reports NOT_TESTED, never a pass.
* ``validate_checks`` — the readiness checks the validate profile can run for
  this stage right now. C0-C3 have frozen contracts, locks and identities that
  can be re-derived offline, so they have real checks. C4-C13 have nothing to
  check yet, and an empty tuple is the truthful representation of that.

The gap between the two is the point. A green validate run proves the frozen
contracts still reproduce; it does not prove that C4-C13 can execute, and no
reader of the artifacts should be able to confuse the two.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: L.5 phase each stage belongs to under the full profile.
PREFLIGHT = "preflight"
SOURCE_SEARCH = "source-search"
SOURCE_FREEZE = "source-freeze"
SCIENTIFIC = "scientific"
TARGET_EVAL = "target-eval"
FINAL_REPORT = "final-report"


@dataclass(frozen=True)
class Stage:
    """One milestone, its L.9 obligations and its current implementation state."""

    stage_id: str
    title: str
    phase: str
    mandatory_full_outputs: str
    adapter_implemented: bool = False
    validate_checks: tuple[str, ...] = ()
    #: Why no adapter exists yet, so the artifact explains itself.
    adapter_note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def index(self) -> int:
        return int(self.stage_id[1:])

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "title": self.title,
            "phase": self.phase,
            "mandatory_full_outputs": self.mandatory_full_outputs,
            "adapter_implemented": self.adapter_implemented,
            "validate_checks": list(self.validate_checks),
            "adapter_note": self.adapter_note,
        }


_NO_ADAPTER = ("no stage adapter exists; adapters are step 4 of "
               "docs/V15_PIPELINE_RESTRUCTURE_PLAN.md and are separately authorized")

#: The L.9 table, transcribed as declarations. The "representative mandatory
#: full outputs" column is kept verbatim in spirit so a future adapter has the
#: obligation in front of it rather than in a document.
STAGES: tuple[Stage, ...] = (
    Stage("C0", "Spec and repository reconciliation", PREFLIGHT,
          "spec/repo reconciliation, compliance and deviation matrix, environment and "
          "source-package identities, C0_ACCEPTANCE",
          validate_checks=("spec_sha256", "version_b_integrity", "environment",
                           "c0_acceptance_present"),
          adapter_note=_NO_ADAPTER),
    Stage("C1", "Provider, schema and prompt contracts", PREFLIGHT,
          "provider/schema/prompt contracts, tests, provider and mock evidence, "
          "identities, C1_ACCEPTANCE",
          validate_checks=("contract_identities", "c1_acceptance_present"),
          adapter_note=_NO_ADAPTER),
    Stage("C2", "Disposable generation pilot", PREFLIGHT,
          "disposable pilot archive and audit, validity/coverage/retry/quota evidence, "
          "C2_ACCEPTANCE",
          validate_checks=("c2_acceptance_present", "route_contract_exact"),
          adapter_note=_NO_ADAPTER),
    Stage("C3", "LLM recipe bank generation and selection", PREFLIGHT,
          "all 384 raw slots per arm, selected and rejected manifests, selector audit, "
          "256 banks per arm, locks and identities, C3_ACCEPTANCE",
          validate_checks=("c3_contract_identities", "c3_locks_verify",
                           "c3_generation_not_started"),
          adapter_note=_NO_ADAPTER),
    Stage("C4", "GPAT source search and final checkpoint", SOURCE_SEARCH,
          "every allowed GPAT source-search run, configs/metrics/checkpoints, winner "
          "selection and lock, final GPAT checkpoint, C4_ACCEPTANCE",
          adapter_note=_NO_ADAPTER),
    Stage("C5", "Route rendering and synthetic candidates", SCIENTIFIC,
          "per-arm and per-recipe route render manifests, synthetic candidate identities, "
          "failures, C5_ACCEPTANCE",
          adapter_note=_NO_ADAPTER),
    Stage("C6", "Quality gate and matched banks", SCIENTIFIC,
          "quality-gate search and audit where permitted, all candidate decisions and q "
          "values, matched 1024-per-arm banks, reliability, locks, C6_ACCEPTANCE",
          adapter_note=_NO_ADAPTER),
    Stage("C7", "Detector readiness and configuration search", SOURCE_SEARCH,
          "all declared detector/config readiness and search runs, forward/backward/resume/"
          "dependency audits, selected config locks, C7_ACCEPTANCE",
          adapter_note=_NO_ADAPTER),
    Stage("C8", "Source matrix over arms, tracks, configs and seeds", SCIENTIFIC,
          "all P1/P2/P3-ready source runs by arm/track/config/seed, complete source "
          "leaderboard and selection evidence, calibration stability, C8_ACCEPTANCE",
          adapter_note=_NO_ADAPTER),
    Stage("C9", "Source matrix freeze", SOURCE_FREEZE,
          "validated SOURCE_MATRIX_LOCK_C plus ancestry and index of every frozen "
          "source-side artifact, C9_ACCEPTANCE",
          adapter_note=_NO_ADAPTER),
    Stage("C10", "Target package and label firewall", TARGET_EVAL,
          "target package and capability lock, label-firewall audit, C10_ACCEPTANCE",
          adapter_note=_NO_ADAPTER),
    Stage("C11", "Label-isolated P3 prediction", TARGET_EVAL,
          "every preregistered P3 prediction artifact by method and seed plus per-row "
          "PREDICTION_LOCK and global lockset, C11_ACCEPTANCE",
          adapter_note=_NO_ADAPTER),
    Stage("C12", "Scoring, statistics and hypothesis tests", TARGET_EVAL,
          "frame and video metrics, bootstrap and Holm, hypotheses, scorer isolation and "
          "unlock evidence, C12_ACCEPTANCE",
          adapter_note=_NO_ADAPTER),
    Stage("C13", "Acceptance, evidence package and report", FINAL_REPORT,
          "C_ACCEPTANCE, master final summary, reports/tables/plots/paper evidence, "
          "preserved negative, failed and blocked results",
          adapter_note=_NO_ADAPTER),
)

STAGE_IDS: tuple[str, ...] = tuple(stage.stage_id for stage in STAGES)
STAGES_BY_ID: dict[str, Stage] = {stage.stage_id: stage for stage in STAGES}


class StageError(ValueError):
    """An unknown stage id, or a range that does not describe a real slice."""


def get_stage(stage_id: str) -> Stage:
    try:
        return STAGES_BY_ID[stage_id.upper()]
    except KeyError:
        raise StageError(f"unknown stage {stage_id!r}; expected one of {STAGE_IDS}") from None


def stage_slice(*, first: str | None = None, last: str | None = None) -> tuple[Stage, ...]:
    """The contiguous C-range for ``--from``/``--to``.

    L.4 permits these flags to change debugging scope but never scientific
    content, so this returns a plain slice and carries no other meaning. A
    partial slice cannot by itself create a full-pipeline acceptance.
    """
    start = get_stage(first).index if first else STAGES[0].index
    stop = get_stage(last).index if last else STAGES[-1].index
    if start > stop:
        raise StageError(f"--from {first} is after --to {last}")
    return tuple(stage for stage in STAGES if start <= stage.index <= stop)


def implemented_stages() -> tuple[Stage, ...]:
    return tuple(stage for stage in STAGES if stage.adapter_implemented)


def validatable_stages() -> tuple[Stage, ...]:
    return tuple(stage for stage in STAGES if stage.validate_checks)


__all__ = ["Stage", "STAGES", "STAGE_IDS", "STAGES_BY_ID", "StageError", "get_stage",
           "stage_slice", "implemented_stages", "validatable_stages",
           "PREFLIGHT", "SOURCE_SEARCH", "SOURCE_FREEZE", "SCIENTIFIC", "TARGET_EVAL",
           "FINAL_REPORT"]
