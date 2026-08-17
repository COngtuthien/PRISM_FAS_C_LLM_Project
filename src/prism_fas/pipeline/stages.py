"""The C0-C13 stage declarations (v1.5 Appendix L.9).

This module declares what each stage IS and what the full profile owes for it.
It deliberately does not execute anything, and it deliberately does not pretend
that a stage exists because it is listed here.

Two flags carry that honesty:

* ``adapter_implemented`` — whether a stage adapter exists that can execute this
  milestone's control path under the smoke or full profile. Every C0-C13 stage
  now has one. A stage without an adapter is BLOCKED, never skipped and never
  passed.
* ``validate_checks`` — the readiness checks the validate profile can run for
  this stage. Each is a pure read-and-re-derive over the repository: no
  training, no provider, no GPU, no target label.

The gap between the two is the point, and it did not close when the adapters
landed. An adapter proves a stage CAN execute; it says nothing about whether the
stage HAS executed scientifically. C4-C13 have adapters and have never run under
the full profile, so their ``scientific_status`` is NOT_RUN and stays there until
a full run produces evidence. A green validate run and a green smoke run together
mean ENGINEERING_READY — not one milestone of scientific completion.
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
    #: Substages that carry their own evidence row. C2B and C2C are C2's later
    #: batches, not stages of their own: L.9 fixes the sequence at C0..C13, and
    #: inventing a C2B stage would change a numbering the spec owns.
    substages: tuple[str, ...] = ()
    #: Why no adapter exists yet, so the artifact explains itself.
    adapter_note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def index(self) -> int:
        return int(self.stage_id[1:])

    @property
    def evidence_units(self) -> tuple[str, ...]:
        """The ids this stage produces evidence under — itself, or its substages."""
        return self.substages or (self.stage_id,)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "title": self.title,
            "phase": self.phase,
            "mandatory_full_outputs": self.mandatory_full_outputs,
            "adapter_implemented": self.adapter_implemented,
            "validate_checks": list(self.validate_checks),
            "substages": list(self.substages),
            "adapter_note": self.adapter_note,
        }


#: Kept for a stage added later that has no adapter yet. Nothing uses it today —
#: every C0-C13 stage is adapted — and it stays so the honest answer for a future
#: stage is a value that already exists rather than one someone has to remember.
_NO_ADAPTER = ("no stage adapter exists; a stage without one is BLOCKED under smoke "
               "and full rather than skipped or passed")

_ENGINEERING_ADAPTER = (
    "engineering adapter: the stage's control path is executable and auditable under "
    "smoke on tiny fixtures, and refuses to start under full when a real scientific "
    "input is absent. An adapter is an ENGINEERING statement — it does not mean the "
    "stage has scientific evidence, and scientific_status stays NOT_RUN until the full "
    "profile executes it.")

_VERIFY_ADAPTER = ("verification-only adapter over a completed milestone: it checks "
                   "artifacts, identities and acceptance, and can never re-issue the "
                   "milestone's archived provider calls")

#: The L.9 table, transcribed as declarations. The "representative mandatory
#: full outputs" column is kept verbatim in spirit so a future adapter has the
#: obligation in front of it rather than in a document.
STAGES: tuple[Stage, ...] = (
    Stage("C0", "Spec and repository reconciliation", PREFLIGHT,
          "spec/repo reconciliation, compliance and deviation matrix, environment and "
          "source-package identities, C0_ACCEPTANCE",
          adapter_implemented=True,
          validate_checks=("spec_sha256", "version_b_integrity", "environment",
                           "c0_acceptance_present"),
          substages=("C0",),
          adapter_note=_VERIFY_ADAPTER),
    Stage("C1", "Provider, schema and prompt contracts", PREFLIGHT,
          "provider/schema/prompt contracts, tests, provider and mock evidence, "
          "identities, C1_ACCEPTANCE",
          adapter_implemented=True,
          validate_checks=("contract_identities", "c1_acceptance_present"),
          substages=("C1",),
          adapter_note=_VERIFY_ADAPTER),
    Stage("C2", "Disposable generation pilot", PREFLIGHT,
          "disposable pilot archive and audit, validity/coverage/retry/quota evidence, "
          "C2_ACCEPTANCE",
          adapter_implemented=True,
          validate_checks=("c2_acceptance_present", "route_contract_exact"),
          substages=("C2", "C2B", "C2C"),
          adapter_note=_VERIFY_ADAPTER),
    Stage("C3", "LLM recipe bank generation and selection", PREFLIGHT,
          "all 384 raw slots per arm, selected and rejected manifests, selector audit, "
          "256 banks per arm, locks and identities, C3_ACCEPTANCE",
          adapter_implemented=True,
          validate_checks=("c3_contract_identities", "c3_locks_verify",
                           "c3_scientific_banks_frozen"),
          substages=("C3",),
          adapter_note="adapter has four modes: PRE_LIVE_VERIFY, LIVE_GENERATE, "
                       "RESUME_LIVE_GENERATE, FINALIZE_BANKS. Only PRE_LIVE_VERIFY is "
                       "reachable under validate; a live provider binding additionally "
                       "requires explicit user authorization."),
    Stage("C4", "GPAT source search and final checkpoint", SOURCE_SEARCH,
          "every allowed GPAT source-search run, configs/metrics/checkpoints, winner "
          "selection and lock, final GPAT checkpoint, C4_ACCEPTANCE",
          adapter_implemented=True,
          validate_checks=("c4_search_plan",),
          adapter_note=_ENGINEERING_ADAPTER +
                       "Modes: PREPARE_SUPPORT, VALIDATE_SUPPORT, SMOKE_GPAT, "
                       "SOURCE_SEARCH, FINALIZE_GPAT, VERIFY_LOCK."),
    Stage("C5", "Route rendering and synthetic candidates", SCIENTIFIC,
          "per-arm and per-recipe route render manifests, synthetic candidate identities, "
          "failures, C5_ACCEPTANCE",
          adapter_implemented=True,
          validate_checks=("c5_route_contract",),
          adapter_note=_ENGINEERING_ADAPTER +
                       "Modes: LOAD_RECIPES, RESOLVE_ROUTES, RENDER_PHYSICS, "
                       "RENDER_GPAT, CANDIDATE_IDENTITY, FAILURE_RECORDING."),
    Stage("C6", "Quality gate and matched banks", SCIENTIFIC,
          "quality-gate search and audit where permitted, all candidate decisions and q "
          "values, matched 1024-per-arm banks, reliability, locks, C6_ACCEPTANCE",
          adapter_implemented=True,
          validate_checks=("c6_gate_profiles",),
          adapter_note=_ENGINEERING_ADAPTER +
                       "Modes: APPLY_COMMON_GATE, PROFILE_SELECTION, "
                       "RELIABILITY_GATES, MATCHED_BANKS, CARDINALITY_REFUSAL."),
    Stage("C7", "Detector readiness and configuration search", SOURCE_SEARCH,
          "all declared detector/config readiness and search runs, forward/backward/resume/"
          "dependency audits, selected config locks, C7_ACCEPTANCE",
          adapter_implemented=True,
          validate_checks=("c7_tracks_resolve",),
          adapter_note=_ENGINEERING_ADAPTER +
                       "Modes: TRACK_G_READINESS, TRACK_R_READINESS, "
                       "DECISION_DEPENDENCY_AUDIT, CALIBRATION_GUARDS, "
                       "VARIANT_MATRIX_AUDIT, SOURCE_SEARCH."),
    Stage("C8", "Source matrix over arms, tracks, configs and seeds", SCIENTIFIC,
          "all P1/P2/P3-ready source runs by arm/track/config/seed, complete source "
          "leaderboard and selection evidence, calibration stability, C8_ACCEPTANCE",
          adapter_implemented=True,
          validate_checks=("c8_source_matrix",),
          adapter_note=_ENGINEERING_ADAPTER +
                       "Modes: PLAN_MATRIX, SCHEDULE, EXECUTE_ROWS, "
                       "FAILURE_PRESERVATION, TARGET_ISOLATION."),
    Stage("C9", "Source matrix freeze", SOURCE_FREEZE,
          "validated SOURCE_MATRIX_LOCK_C plus ancestry and index of every frozen "
          "source-side artifact, C9_ACCEPTANCE",
          adapter_implemented=True,
          validate_checks=("c9_source_lock_refuses",),
          adapter_note=_ENGINEERING_ADAPTER +
                       " Modes: BUILD_LOCK, VALIDATE_LOCK, REFUSAL_CASES."),
    Stage("C10", "Target package and label firewall", TARGET_EVAL,
          "target package and capability lock, label-firewall audit, C10_ACCEPTANCE",
          adapter_implemented=True,
          validate_checks=("c10_firewall_config",),
          adapter_note=_ENGINEERING_ADAPTER +
                       "Modes: BUILD_FIXTURE_PACKAGE, FIREWALL_PERMISSIONS, "
                       "PACKAGE_IDENTITY, TARGET_LOCK, TAMPER_DETECTION. The real "
                       "target package is never opened."),
    Stage("C11", "Label-isolated P3 prediction", TARGET_EVAL,
          "every preregistered P3 prediction artifact by method and seed plus per-row "
          "PREDICTION_LOCK and global lockset, C11_ACCEPTANCE",
          adapter_implemented=True,
          validate_checks=("c11_prediction_schema",),
          adapter_note=_ENGINEERING_ADAPTER +
                       "Modes: BUILD_PREDICTIONS, LABEL_ISOLATION_AUDIT, "
                       "PREDICTION_LOCKS, DOUBLE_VALIDATION. No real target inference "
                       "is performed."),
    Stage("C12", "Scoring, statistics and hypothesis tests", TARGET_EVAL,
          "frame and video metrics, bootstrap and Holm, hypotheses, scorer isolation and "
          "unlock evidence, C12_ACCEPTANCE",
          adapter_implemented=True,
          validate_checks=("c12_scorer_isolation",),
          adapter_note=_ENGINEERING_ADAPTER +
                       "Modes: SCORER_ISOLATION, DRY_RUN, UNLOCK_AND_SCORE, "
                       "STATISTICS, NO_FEEDBACK. No real SiW label is opened."),
    Stage("C13", "Acceptance, evidence package and report", FINAL_REPORT,
          "C_ACCEPTANCE, master final summary, reports/tables/plots/paper evidence, "
          "preserved negative, failed and blocked results",
          adapter_implemented=True,
          validate_checks=("c13_acceptance_refuses",),
          adapter_note=_ENGINEERING_ADAPTER +
                       "Modes: ACCEPTANCE_MATRIX, NEGATIVE_PRESERVATION, "
                       "ARTIFACT_INTEGRITY, CLAIM_POLICY, FINAL_REPORT. It refuses "
                       "acceptance while any milestone is scientifically incomplete."),
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
