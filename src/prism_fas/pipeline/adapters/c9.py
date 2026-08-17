"""C9 — the source freeze.

C9 builds and validates SOURCE_MATRIX_LOCK_C. Its acceptance is short and
absolute: all checkpoints, calibrations and identities frozen, zero failed hidden
rows. So this adapter spends most of its effort on the refusals, because a lock
that only ever succeeds proves nothing about the gate it is supposed to be.

Six refusal cases are constructed and each is asserted to block the freeze: a
required row missing, a row that failed, a run-identity mismatch, a missing
calibration, a missing checkpoint, and a hidden row that the plan never declared.
The last one is the reason "hidden" appears in the acceptance criterion at all —
an unplanned run entering the frozen set would put a configuration into the
comparison that was never preregistered.

The smoke lock is written under the profile's own namespace and declares
`is_scientific_lock: false`. The real SOURCE_MATRIX_LOCK_C is built at C9 under
the full profile from real runs, and nothing here may occupy that position.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from prism_fas.pipeline.adapters import AdapterRequest, AdapterResult
from prism_fas.pipeline.adapters.common import (EngineeringAdapter, RequiredInput, check,
                                                resume_decision, stage_reports_dir, utc,
                                                write_artifact)

STAGE_ID = "C9"

BUILD_LOCK = "BUILD_LOCK"
VALIDATE_LOCK = "VALIDATE_LOCK"
REFUSAL_CASES = "REFUSAL_CASES"

MODES: tuple[str, ...] = (BUILD_LOCK, VALIDATE_LOCK, REFUSAL_CASES)


def _complete_evidence(plan: Any) -> list[Any]:
    """Synthetic complete evidence for every planned row.

    Fixture evidence, clearly: the hashes are derived from each row's own
    identity rather than from a trained checkpoint. What it exercises is the
    lock's completeness logic, which is a pure function of the evidence shape.
    """
    from prism_fas.evaluation.source_lock import RowEvidence

    return [RowEvidence(
        row_id=row.row_id, run_identity=row.run_identity,
        config_identity=row.config_identity, status="PASS",
        checkpoint_sha256=f"fixture-checkpoint-{row.run_identity[:16]}",
        calibration_sha256=f"fixture-calibration-{row.run_identity[:16]}",
        calibration_hash=f"fixture-calhash-{row.config_identity[:16]}",
        decision_logit_name="fused_logit_R" if row.track == "R" else "global_logit_G",
        decision_score_name="p_R" if row.track == "R" else "p_G",
        parent_identities={"source_matrix": plan.identity},
        metrics={"source_dev_acer": 0.1}) for row in plan.rows]


@dataclass
class C9Adapter(EngineeringAdapter):
    """The C9 execution adapter. The lock logic is imported, not restated."""

    stage_id: str = STAGE_ID
    substages: tuple[str, ...] = (STAGE_ID,)
    title: str = "Source matrix freeze"
    modes: tuple[str, ...] = MODES
    requires_gpu: bool = False

    def required_inputs(self) -> tuple[RequiredInput, ...]:
        return (
            RequiredInput("c8_runs", "runs/full/c8",
                          "every executed source run's manifest, checkpoint and calibration"),
            RequiredInput("c8_acceptance", "reports/full/c8/C8_ACCEPTANCE.json",
                          "C8's own acceptance verdict over the completed matrix"),
        )

    def run_smoke(self, request: AdapterRequest) -> list[AdapterResult]:
        reports = stage_reports_dir(request, STAGE_ID)
        lock, build_result = self._build(request, reports)
        return [build_result,
                self._validate(request, lock, reports),
                self._refusals(request, reports)]

    # --- modes ----------------------------------------------------------------

    def _build(self, request: AdapterRequest, reports: Path) -> tuple[dict[str, Any] | None,
                                                                      AdapterResult]:
        from prism_fas.evaluation.source_lock import SourceLockError, audit, build
        from prism_fas.evaluation.source_matrix import build_plan

        checks: list[dict[str, Any]] = []
        plan = build_plan()
        evidence = _complete_evidence(plan)
        report = audit(plan, evidence)

        checks.append(check(
            "c9_audit_finds_no_blocking_refusal", report["freezable"],
            "complete evidence produces no blocking refusal",
            planned_rows=report["planned_rows"], evidence_rows=report["evidence_rows"],
            accepted_rows=report["accepted_rows"],
            blocking=report["blocking_refusals"]))

        lock: dict[str, Any] | None = None
        try:
            lock = build(plan, evidence,
                         code_lineage={"engineering_smoke": True},
                         artifact_identities={"c3_recipe_banks": "see C3 scientific lock"})
            built = True
            error = ""
        except SourceLockError as failure:
            built, error = False, str(failure)

        checks.append(check(
            "c9_lock_builds_from_complete_evidence", built,
            "SOURCE_MATRIX_LOCK_C builds when every row is present, terminal and passing",
            error=error, lock_identity=(lock or {}).get("lock_identity")))
        if lock is not None:
            checks.append(check(
                "c9_lock_covers_every_planned_row",
                lock["row_count"] == len(plan.rows),
                "the lock covers every preregistered row",
                row_count=lock["row_count"], planned=len(plan.rows)))
            checks.append(check(
                "c9_lock_declares_no_target_capability",
                lock["target_capability"]["target_labels_resolved"] == 0,
                "the lock carries a no-target-capability proof",
                **lock["target_capability"]))

        artifact = write_artifact(request, reports / "C9_SOURCE_MATRIX_LOCK.json", {
            **(lock or {"status": "NOT_BUILT", "error": error}),
            "generated_at_utc": utc(), "mode": BUILD_LOCK,
            "is_scientific_lock": False,
            "why_not": ("the real SOURCE_MATRIX_LOCK_C is built at C9 under the full "
                        "profile from real runs. This lock was built over fixture evidence "
                        "and may never occupy that position"),
            "fixture_backed": True})
        return lock, self.result(request, mode=BUILD_LOCK, checks=checks,
                                 artifacts=[artifact],
                                 parent_identities={"c8_source_matrix": plan.identity})

    def _validate(self, request: AdapterRequest, lock: dict[str, Any] | None,
                  reports: Path) -> AdapterResult:
        from prism_fas.evaluation.source_lock import validate
        from prism_fas.evaluation.source_matrix import build_plan

        checks: list[dict[str, Any]] = []
        plan = build_plan()
        if lock is None:
            checks.append(check("c9_lock_available_to_validate", False,
                                "no lock was produced by the build mode"))
            return self.result(request, mode=VALIDATE_LOCK, checks=checks)

        report = validate(lock, plan, _complete_evidence(plan))
        checks.append(check(
            "c9_lock_identity_reproduces", report["identity_reproduces"],
            "the lock body hashes to its own recorded identity",
            lock_identity=report["lock_identity"],
            recomputed=report["lock_identity_recomputed"]))
        checks.append(check(
            "c9_lock_validates_against_the_plan", report["valid"],
            "the lock validates against the matrix plan and current evidence",
            problems=report["problems"], rows_locked=report["rows_locked"]))

        drifted = validate(lock, plan, [replace(item, checkpoint_sha256="moved")
                                        for item in _complete_evidence(plan)])
        checks.append(check(
            "c9_validation_detects_drifted_evidence", not drifted["valid"],
            "changing a frozen checkpoint hash makes validation fail",
            problems=drifted["problems"][:1],
            rule="L.11: if an expected identity changed, fail closed rather than reuse"))

        artifact = write_artifact(request, reports / "C9_LOCK_VALIDATION.json", {
            "schema_version": "c9-lock-validation-v1", "generated_at_utc": utc(),
            "mode": VALIDATE_LOCK, "validation": report,
            "drift_detection": {"detected": not drifted["valid"],
                                "problems": drifted["problems"]},
            "fixture_backed": True})

        decision = resume_decision(request, "c9_source_matrix_lock",
                                   reports / "C9_SOURCE_MATRIX_LOCK.json",
                                   expected_identity=lock["lock_identity"],
                                   identity_key="lock_identity")
        checks.append(check(
            "c9_resume_is_identity_aware", decision["identity_matches"],
            "resume validates the lock by its recorded identity", **decision))
        return self.result(request, mode=VALIDATE_LOCK, checks=checks, artifacts=[artifact])

    def _refusals(self, request: AdapterRequest, reports: Path) -> AdapterResult:
        """Every condition §C9 says must block a freeze, constructed and asserted."""
        from prism_fas.evaluation.source_lock import RowEvidence, SourceLockError, build
        from prism_fas.evaluation.source_matrix import build_plan

        checks: list[dict[str, Any]] = []
        plan = build_plan()
        base = _complete_evidence(plan)

        def refuses(name: str, evidence: list[Any]) -> dict[str, Any]:
            try:
                build(plan, evidence)
                return {"case": name, "refused": False, "error": ""}
            except SourceLockError as error:
                return {"case": name, "refused": True, "error": str(error)[:400]}

        cases = [
            refuses("required_row_missing", base[:-1]),
            refuses("row_failed", [*base[:-1], replace(base[-1], status="FAIL")]),
            refuses("row_not_terminal", [*base[:-1], replace(base[-1], status="RUNNING")]),
            refuses("run_identity_mismatch",
                    [*base[:-1], replace(base[-1], run_identity="0" * 64)]),
            refuses("calibration_missing",
                    [*base[:-1], replace(base[-1], calibration_sha256=None,
                                         calibration_hash=None)]),
            refuses("checkpoint_missing",
                    [*base[:-1], replace(base[-1], checkpoint_sha256=None)]),
            refuses("hidden_row_not_in_plan",
                    [*base, RowEvidence(row_id="UNPLANNED-ROW", run_identity="x",
                                        config_identity="y", status="PASS",
                                        checkpoint_sha256="c", calibration_sha256="d",
                                        calibration_hash="e")]),
        ]
        for case in cases:
            checks.append(check(
                f"c9_refuses_{case['case']}", case["refused"],
                f"the freeze is refused when {case['case'].replace('_', ' ')}",
                error=case["error"]))

        checks.append(check(
            "c9_refusal_is_total_not_partial", all(case["refused"] for case in cases),
            "no refusal produces a partial lock; the build raises instead of returning one",
            cases=[case["case"] for case in cases],
            rule="§C9: all checkpoints, calibrations and identities frozen; 0 failed "
                 "hidden rows"))

        artifact = write_artifact(request, reports / "C9_REFUSAL_CASES.json", {
            "schema_version": "c9-refusal-cases-v1", "generated_at_utc": utc(),
            "mode": REFUSAL_CASES, "cases": cases, "fixture_backed": True,
            "meaning": "constructed refusal cases proving the gate blocks. They are not "
                       "findings about any real run"})
        return self.result(request, mode=REFUSAL_CASES, checks=checks, artifacts=[artifact])


__all__ = ["STAGE_ID", "MODES", "BUILD_LOCK", "VALIDATE_LOCK", "REFUSAL_CASES", "C9Adapter"]
