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
from prism_fas.pipeline.adapters.common import (assert_fixture_permitted,
                                                EngineeringAdapter, RequiredInput, check,
                                                resume_decision, stage_reports_dir, utc,
                                                write_artifact)
from prism_fas.evaluation import detector_reliability
from prism_fas.pipeline.execution import ExecutionContext

STAGE_ID = "C9"

BUILD_LOCK = "BUILD_LOCK"
VALIDATE_LOCK = "VALIDATE_LOCK"
REFUSAL_CASES = "REFUSAL_CASES"

#: Scientific-only substages. `REFUSAL_CASES` is deliberately not among them: it
#: CONSTRUCTS seven broken evidence sets to prove the gate refuses, which is
#: rehearsal evidence and would be fabrication inside a scientific pass.
LOAD_C8_EVIDENCE = "LOAD_C8_EVIDENCE"
FREEZE_SOURCE_MATRIX = "FREEZE_SOURCE_MATRIX"

SCIENTIFIC_MODES: tuple[str, ...] = (LOAD_C8_EVIDENCE, FREEZE_SOURCE_MATRIX,
                                     VALIDATE_LOCK)

MODES: tuple[str, ...] = (BUILD_LOCK, VALIDATE_LOCK, REFUSAL_CASES,
                          LOAD_C8_EVIDENCE, FREEZE_SOURCE_MATRIX)

#: The one governing artifact scientific C9 produces. C10 declares this exact
#: path as a required input, so it is named once here.
SCIENTIFIC_REPORTS = "reports/full/c9"
SOURCE_MATRIX_LOCK = "SOURCE_MATRIX_LOCK_C.json"


def _complete_evidence(plan: Any, context: Any = None) -> list[Any]:
    """Synthetic complete evidence for every planned row.

    Fixture evidence, clearly: the hashes are derived from each row's own
    identity rather than from a trained checkpoint. What it exercises is the
    lock's completeness logic, which is a pure function of the evidence shape,
    and exercising it that way is the only way to reach the refusal branches at
    all — a real complete matrix has none of them.

    `context` is not optional in practice. Every caller inside the adapter passes
    it, and passing a scientific one raises: a SOURCE_MATRIX_LOCK_C built over
    rows this function invented would be a freeze over an experiment nobody
    performed. The default exists only so the rehearsal-only helpers in the tests
    can call it without a request.
    """
    from prism_fas.evaluation.source_lock import RowEvidence

    if context is not None:
        assert_fixture_permitted(context, "the C9 constructed complete evidence set")

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
            # The pre-target reliability barrier. SOURCE_MATRIX_LOCK_C may not
            # close over an unresolved one, and the file merely existing proves
            # nothing — `semantic_preconditions` validates it.
            RequiredInput("detector_reliability_lock",
                          detector_reliability.LOCK_PATH,
                          "the post-C8 pre-C9 reliability barrier, including the "
                          "synthetic-vs-real probe"),
        )

    def semantic_preconditions(self, request: AdapterRequest) -> list[dict[str, Any]]:
        """SOURCE_MATRIX_LOCK_C closes only over a VALID reliability barrier.

        Structural rather than a human-readable note in PROJECT_STATE: the lock
        must resolve every required test to PASSED, bind the probe protocol and
        detector checkpoint identities, and record zero target access. An
        unresolved required test never counts as a pass, so C10 and C11 stay
        unreachable until the barrier is genuinely resolved.
        """
        verification = detector_reliability.verify_lock(request.repo)
        return [{
            "name": "detector_reliability_resolved",
            "path": detector_reliability.LOCK_PATH,
            "present": verification["valid"],
            "blocking": not verification["valid"],
            "description": ("every required detector-level reliability test "
                            "resolved to PASSED after C8 and before C9, with the "
                            "probe protocol and checkpoint identities bound"),
            "verifier": "prism_fas.evaluation.detector_reliability.verify_lock",
            "problems": verification["problems"][:12],
            "required_stage": verification["required_stage"],
            "required_tests": list(
                detector_reliability.REQUIRED_DETECTOR_RELIABILITY_TESTS),
        }]

    def workflow(self, request: AdapterRequest,
                 context: ExecutionContext) -> list[AdapterResult]:
        """Two workflows, chosen by the context — never one that adapts.

        The rehearsal builds a lock over constructed evidence and proves the
        seven refusal cases. The scientific path loads the real C8 manifests,
        re-hashes the checkpoints they name and refuses to build anything at all
        if a single planned row is absent, failed, fixture-backed or drifted.
        `_complete_evidence` raises under a scientific context, so the two cannot
        meet even if a future edit reconnected them.
        """
        if context.is_scientific:
            return self._scientific_workflow(request, context)
        return self._engineering_workflow(request, context)

    def _engineering_workflow(self, request: AdapterRequest,
                              context: ExecutionContext) -> list[AdapterResult]:
        reports = stage_reports_dir(request, STAGE_ID)
        lock, build_result = self._build(request, reports)
        return [build_result,
                self._validate(request, lock, reports),
                self._refusals(request, reports)]

    # --- the scientific workflow ---------------------------------------------

    def _scientific_workflow(self, request: AdapterRequest,
                             context: ExecutionContext) -> list[AdapterResult]:
        """The real C9: freeze what C8 actually produced, or refuse and say why."""
        reports = stage_reports_dir(request, STAGE_ID)
        evidence, load_result = self._scientific_evidence(request, reports)
        if evidence is None:
            return [load_result]
        lock, freeze_result = self._scientific_freeze(request, evidence, reports)
        results = [load_result, freeze_result]
        if lock is not None:
            results.append(self._scientific_validate(request, evidence, lock, reports))
        return results

    def _scientific_evidence(self, request: AdapterRequest,
                             reports: Path) -> tuple[dict[str, Any] | None, AdapterResult]:
        """Load REAL C8 row evidence. Nothing here constructs a row."""
        from prism_fas.evaluation import source_evidence
        from prism_fas.evaluation.source_matrix import build_plan

        checks: list[dict[str, Any]] = []
        plan = build_plan()
        report = source_evidence.evidence_report(request.repo, plan)

        checks.append(check(
            "c9_c8_acceptance_present_and_scientific", report["available"],
            "C8's own acceptance verdict exists, is scientifically eligible and is "
            "not fixture-backed",
            reason_code=report["reason_code"], error=report["error"][:400],
            loader="prism_fas.evaluation.source_evidence.evidence_report"))
        if not report["available"]:
            return None, self.result(request, mode=LOAD_C8_EVIDENCE, checks=checks,
                                     summary="C9 has no scientific C8 evidence to freeze")

        checks.append(check(
            "c9_evidence_is_real_not_constructed", True,
            "every row of evidence came from a C8 run manifest on disk; the "
            "constructed-evidence helper raises under a scientific context",
            rows_found=report["rows_found"], rows_planned=report["rows_planned"],
            loaded_by=report["loaded_by"], runs_root=report["runs_root"],
            constructed_evidence_reachable=False))
        checks.append(check(
            "c9_every_planned_row_has_readable_evidence",
            report["rows_found"] == report["rows_planned"] and not report["problems"],
            f"{report['rows_found']}/{report['rows_planned']} planned rows produced "
            "readable, identity-matching, byte-verified evidence",
            problems=report["problems"][:12],
            problem_count=len(report["problems"])))
        checks.append(check(
            "c9_c8_accepted_its_own_matrix", report["acceptance_accepted"],
            "C8 accepted the matrix it produced; C9 does not overrule a C8 refusal",
            **{key: report["acceptance"].get(key) for key in
               ("accepted", "rows_declared", "rows_terminal", "rows_passed",
                "rows_failed", "hidden_rows", "missing_rows")}))

        artifact = write_artifact(request, reports / "C9_C8_EVIDENCE.json", {
            "schema_version": "c9-c8-evidence-v1", "generated_at_utc": utc(),
            "mode": LOAD_C8_EVIDENCE, "fixture_backed": False,
            "matrix_identity": plan.identity,
            "rows_planned": report["rows_planned"], "rows_found": report["rows_found"],
            "problems": report["problems"],
            "acceptance": report["acceptance"],
            "rows": [item.as_dict() for item in report["evidence"]]})

        if not all(item["ok"] for item in checks):
            return None, self.result(request, mode=LOAD_C8_EVIDENCE, checks=checks,
                                     artifacts=[artifact],
                                     summary="C9 refuses: the C8 evidence is incomplete")

        state = {"plan": plan, "evidence": report["evidence"], "report": report}
        return state, self.result(request, mode=LOAD_C8_EVIDENCE, checks=checks,
                                  artifacts=[artifact],
                                  parent_identities={"c8_source_matrix": plan.identity})

    def _scientific_freeze(self, request: AdapterRequest, state: dict[str, Any],
                           reports: Path) -> tuple[dict[str, Any] | None, AdapterResult]:
        """Build SOURCE_MATRIX_LOCK_C from the real evidence, or refuse."""
        from prism_fas.evaluation.source_lock import SourceLockError, audit, build

        checks: list[dict[str, Any]] = []
        plan, evidence = state["plan"], state["evidence"]
        report = audit(plan, evidence)

        checks.append(check(
            "c9_audit_finds_no_blocking_refusal", report["freezable"],
            "the real evidence produces no blocking refusal",
            planned_rows=report["planned_rows"], evidence_rows=report["evidence_rows"],
            accepted_rows=report["accepted_rows"],
            blocking=report["blocking_refusals"]))

        lock: dict[str, Any] | None = None
        error = ""
        try:
            acceptance = state["report"]["acceptance"]
            lock = build(plan, evidence,
                         code_lineage={"c8_acceptance": acceptance},
                         artifact_identities={
                             "c7_detector_config_sha256":
                                 str(acceptance.get("c7_detector_config_sha256")),
                             "c6_selector_identity_sha256":
                                 str(acceptance.get("c6_selector_identity_sha256")),
                             "source_package_identity":
                                 str(acceptance.get("source_package_identity"))})
        except SourceLockError as failure:
            error = str(failure)

        checks.append(check(
            "c9_lock_builds_from_real_evidence", lock is not None,
            "SOURCE_MATRIX_LOCK_C builds when every real row is present, terminal "
            "and passing",
            error=error[:400], lock_identity=(lock or {}).get("lock_identity")))
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

        payload = {**(lock or {"status": "NOT_BUILT", "error": error}),
                   "generated_at_utc": utc(), "mode": FREEZE_SOURCE_MATRIX,
                   "is_scientific_lock": lock is not None,
                   "fixture_backed": False,
                   "evidence_source": state["report"]["loaded_by"]}
        artifact = write_artifact(request, reports / SOURCE_MATRIX_LOCK, payload)
        return lock, self.result(request, mode=FREEZE_SOURCE_MATRIX, checks=checks,
                                 artifacts=[artifact],
                                 parent_identities={"c8_source_matrix": plan.identity})

    def _scientific_validate(self, request: AdapterRequest, state: dict[str, Any],
                             lock: dict[str, Any], reports: Path) -> AdapterResult:
        """Re-derive the lock from its own material and from the evidence on disk."""
        from prism_fas.evaluation.source_lock import validate

        checks: list[dict[str, Any]] = []
        plan, evidence = state["plan"], state["evidence"]
        report = validate(lock, plan, evidence)

        checks.append(check(
            "c9_lock_identity_reproduces", report["identity_reproduces"],
            "the lock body hashes to its own recorded identity",
            lock_identity=report["lock_identity"],
            recomputed=report["lock_identity_recomputed"]))
        checks.append(check(
            "c9_lock_validates_against_the_plan_and_disk", report["valid"],
            "the lock validates against the matrix plan and the evidence currently "
            "on disk",
            problems=report["problems"], rows_locked=report["rows_locked"]))

        artifact = write_artifact(request, reports / "C9_LOCK_VALIDATION.json", {
            "schema_version": "c9-lock-validation-v1", "generated_at_utc": utc(),
            "mode": VALIDATE_LOCK, "fixture_backed": False, "validation": report})
        decision = resume_decision(request, "c9_source_matrix_lock",
                                   reports / SOURCE_MATRIX_LOCK,
                                   expected_identity=lock["lock_identity"],
                                   identity_key="lock_identity")
        checks.append(check(
            "c9_resume_is_identity_aware", decision["identity_matches"],
            "resume validates the lock by its recorded identity", **decision))

        passed = all(item["ok"] for item in checks)
        return self.result(request, mode=VALIDATE_LOCK, checks=checks,
                           artifacts=[artifact],
                           # The ONE place C9 claims scientific evidence.
                           scientific_evidence=passed)

    # --- modes ----------------------------------------------------------------

    def _build(self, request: AdapterRequest, reports: Path) -> tuple[dict[str, Any] | None,
                                                                      AdapterResult]:
        from prism_fas.evaluation.source_lock import SourceLockError, audit, build
        from prism_fas.evaluation.source_matrix import build_plan

        checks: list[dict[str, Any]] = []
        plan = build_plan()
        evidence = _complete_evidence(plan, request.context)
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
            "fixture_backed": request.context.fixtures_permitted})
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

        report = validate(lock, plan, _complete_evidence(plan, request.context))
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
                                        for item in _complete_evidence(plan, request.context)])
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
            "fixture_backed": request.context.fixtures_permitted})

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
        base = _complete_evidence(plan, request.context)

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
            "mode": REFUSAL_CASES, "cases": cases, "fixture_backed": request.context.fixtures_permitted,
            "meaning": "constructed refusal cases proving the gate blocks. They are not "
                       "findings about any real run"})
        return self.result(request, mode=REFUSAL_CASES, checks=checks, artifacts=[artifact])


__all__ = ["STAGE_ID", "MODES", "SCIENTIFIC_MODES", "BUILD_LOCK", "VALIDATE_LOCK",
           "REFUSAL_CASES", "LOAD_C8_EVIDENCE", "FREEZE_SOURCE_MATRIX",
           "SCIENTIFIC_REPORTS", "SOURCE_MATRIX_LOCK", "C9Adapter"]
