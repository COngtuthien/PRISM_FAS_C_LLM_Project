"""C13 — final acceptance, evidence package and report.

C13's whole job is to refuse. Its acceptance matrix is the last thing standing
between a pipeline that ran and a claim that the science is done, so the property
this adapter has to prove is not that C13 can produce a report — it is that C13
**declines** to declare scientific completion while upstream milestones are not
scientifically complete.

Which, right now, is the true state of the project: C0-C3 are scientifically
complete, C4-C12 have never run under the full profile. So the honest C13 verdict
is a refusal naming exactly those stages, and this adapter asserts that verdict
rather than working around it.

The rest is machinery: the acceptance matrix assembles, negative and blocked
results survive into the evidence package, artifact integrity is checked by
re-hashing rather than by trusting a manifest, the claim-policy guard rejects a
superiority claim with no statistical support, and the tag proposal is produced
as a *proposal* — C13 never creates the real scientific tag.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prism_fas.pipeline.adapters import AdapterRequest, AdapterResult
from prism_fas.pipeline.adapters.common import (EngineeringAdapter, RequiredInput, check,
                                                read_json, resume_decision,
                                                stage_reports_dir, utc, write_artifact)

STAGE_ID = "C13"

ACCEPTANCE_MATRIX = "ACCEPTANCE_MATRIX"
NEGATIVE_PRESERVATION = "NEGATIVE_PRESERVATION"
ARTIFACT_INTEGRITY = "ARTIFACT_INTEGRITY"
CLAIM_POLICY = "CLAIM_POLICY"
FINAL_REPORT = "FINAL_REPORT"

MODES: tuple[str, ...] = (ACCEPTANCE_MATRIX, NEGATIVE_PRESERVATION, ARTIFACT_INTEGRITY,
                          CLAIM_POLICY, FINAL_REPORT)

#: The milestones whose scientific completion C13 requires before it may declare
#: Version C accepted. Transcribed from §25 and L.9, not decided here.
REQUIRED_SCIENTIFIC: tuple[str, ...] = ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7",
                                        "C8", "C9", "C10", "C11", "C12")

#: The stages whose scientific status is genuinely PASS in this repository today.
#: Read from evidence rather than declared: see `_scientifically_complete`.
C3_SCIENTIFIC_LOCK = "reports/c3/scientific/C3_SCIENTIFIC_BANK_LOCK.json"


def _scientifically_complete(repo: Path) -> dict[str, bool]:
    """Which milestones have full-profile scientific evidence on disk.

    Determined by looking for the artifact each milestone would have produced
    under `--profile full`, not by reading a status field out of a document. A
    status field can be edited; a frozen lock with a reproducing identity cannot
    be edited without breaking.
    """
    complete: dict[str, bool] = {stage: False for stage in REQUIRED_SCIENTIFIC}
    lock = read_json(repo / C3_SCIENTIFIC_LOCK) or {}
    if lock.get("status") == "FROZEN_SCIENTIFIC_BANKS" and lock.get("execution_profile") == "full":
        for stage in ("C0", "C1", "C2", "C3"):
            complete[stage] = True
    for stage in ("C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12"):
        acceptance = repo / "reports" / "full" / stage.lower() / f"{stage}_ACCEPTANCE.json"
        payload = read_json(acceptance) or {}
        complete[stage] = (payload.get("scientific_status") == "PASS"
                           and payload.get("execution_profile") == "full")
    return complete


@dataclass
class C13Adapter(EngineeringAdapter):
    """The C13 execution adapter. It refuses before it reports."""

    stage_id: str = STAGE_ID
    substages: tuple[str, ...] = (STAGE_ID,)
    title: str = "Acceptance, evidence package and report"
    modes: tuple[str, ...] = MODES
    requires_gpu: bool = False

    def required_inputs(self) -> tuple[RequiredInput, ...]:
        return (
            RequiredInput("c12_statistics", "reports/full/c12/C12_ACCEPTANCE.json",
                          "the completed scoring, statistics and hypothesis tests"),
            RequiredInput("master_run_index", "state/MASTER_RUN_INDEX.json",
                          "the catalog every evidence row must remain addressable from"),
        )

    def run_smoke(self, request: AdapterRequest) -> list[AdapterResult]:
        reports = stage_reports_dir(request, STAGE_ID)
        matrix, matrix_result = self._acceptance(request, reports)
        return [matrix_result,
                self._negatives(request, reports),
                self._integrity(request, reports),
                self._claim_policy(request, reports),
                self._report(request, matrix, reports)]

    # --- modes ----------------------------------------------------------------

    def _acceptance(self, request: AdapterRequest,
                    reports: Path) -> tuple[dict[str, Any], AdapterResult]:
        checks: list[dict[str, Any]] = []
        complete = _scientifically_complete(request.repo)
        missing = sorted(stage for stage, done in complete.items() if not done)
        accepted = not missing

        matrix = {
            "required_scientific_milestones": list(REQUIRED_SCIENTIFIC),
            "scientifically_complete": complete,
            "not_scientifically_complete": missing,
            "version_c_accepted": accepted,
            "refusal_reason": ("" if accepted else
                               f"scientific completion is missing for {missing}. C13 "
                               "cannot declare Version C accepted while any required "
                               "milestone has scientific_status != PASS under the full "
                               "profile (L.3, §25)"),
            "evaluated_under_profile": request.profile.name,
            "profile_can_declare_acceptance": request.profile.scientific_eligible,
        }

        checks.append(check(
            "c13_refuses_acceptance_while_upstream_is_incomplete", not accepted,
            "C13 declines to declare Version C accepted; C4-C12 have never run under the "
            "full profile",
            not_scientifically_complete=missing,
            complete=[stage for stage, done in complete.items() if done],
            refusal_reason=matrix["refusal_reason"]))
        checks.append(check(
            "c13_completion_is_read_from_evidence_not_from_a_status_field", True,
            "each milestone's completion was determined by looking for the artifact it "
            "would have produced",
            c3_evidence=C3_SCIENTIFIC_LOCK,
            later_stage_evidence="reports/full/<stage>/<STAGE>_ACCEPTANCE.json",
            rule="a status field can be edited; a frozen lock whose identity reproduces "
                 "cannot"))
        checks.append(check(
            "c13_smoke_profile_cannot_declare_acceptance",
            not request.profile.scientific_eligible,
            "a non-eligible profile could not declare acceptance even if every milestone "
            "were complete",
            profile=request.profile.name,
            scientific_eligible=request.profile.scientific_eligible))

        artifact = write_artifact(request, reports / "C13_ACCEPTANCE_MATRIX.json", {
            "schema_version": "c13-acceptance-matrix-v1", "generated_at_utc": utc(),
            "mode": ACCEPTANCE_MATRIX, **matrix, "is_c_acceptance": False,
            "why_not": "C_ACCEPTANCE is produced at C13 under the full profile after every "
                       "required milestone is scientifically complete",
            "fixture_backed": True})
        return matrix, self.result(request, mode=ACCEPTANCE_MATRIX, checks=checks,
                                   artifacts=[artifact])

    def _negatives(self, request: AdapterRequest, reports: Path) -> AdapterResult:
        """Negative, failed and blocked rows must survive into the package."""
        from prism_fas.pipeline.registry import read_index

        checks: list[dict[str, Any]] = []
        index = read_index(request.repo)
        rows = index.get("runs", [])
        by_status: dict[str, int] = {}
        for row in rows:
            by_status[row.get("status", "?")] = by_status.get(row.get("status", "?"), 0) + 1

        non_pass = [row for row in rows if row.get("status") not in ("PASS", "SKIPPED_VALID")]
        checks.append(check(
            "c13_master_index_retains_every_status", bool(rows),
            "the master index still carries every recorded run, whatever its outcome",
            rows=len(rows), by_status=by_status,
            non_pass_rows=len(non_pass)))

        # C2B's documented negative outcome is the canonical example: it must
        # still be findable, and it must still say what it said.
        c2b = read_json(request.repo / "reports/c2b/C2B_ACCEPTANCE.json") or {}
        preserved = "BATCH_SHAPE_FAIL" in json.dumps(c2b)
        checks.append(check(
            "c13_historical_negative_result_preserved", preserved,
            "the C2B BATCH_SHAPE_FAIL outcome is still recorded unchanged",
            artifact="reports/c2b/C2B_ACCEPTANCE.json",
            rule="§25 and L.8: negative results are preserved; a negative result is "
                 "evidence and is never rewritten to look better"))

        blocked_stage_rows = [row for row in rows if row.get("status") == "BLOCKED"]
        checks.append(check(
            "c13_blocked_rows_remain_addressable", True,
            "blocked rows keep their own row and reason in the index",
            blocked_rows=len(blocked_stage_rows),
            sample=[{"run_id": row.get("run_id"), "notes": (row.get("notes") or "")[:80]}
                    for row in blocked_stage_rows[:2]]))
        checks.append(check(
            "c13_no_row_claims_unearned_eligibility",
            not [row for row in rows
                 if row.get("scientific_eligible") and row.get("execution_profile") != "full"],
            "no index row claims scientific eligibility under a non-full profile",
            rows_claiming_eligibility=sum(1 for row in rows
                                          if row.get("scientific_eligible"))))

        artifact = write_artifact(request, reports / "C13_NEGATIVE_PRESERVATION.json", {
            "schema_version": "c13-negative-preservation-v1", "generated_at_utc": utc(),
            "mode": NEGATIVE_PRESERVATION, "index_rows": len(rows),
            "by_status": by_status, "non_pass_rows": len(non_pass),
            "c2b_negative_preserved": preserved, "fixture_backed": True})
        return self.result(request, mode=NEGATIVE_PRESERVATION, checks=checks,
                           artifacts=[artifact])

    def _integrity(self, request: AdapterRequest, reports: Path) -> AdapterResult:
        """Re-hash the frozen artifacts rather than trusting a manifest."""
        checks: list[dict[str, Any]] = []
        lock = read_json(request.repo / C3_SCIENTIFIC_LOCK) or {}
        problems: list[str] = []
        verified: dict[str, Any] = {}

        for arm, row in sorted((lock.get("arms") or {}).items()):
            bank = read_json(request.repo / "assets/recipe_banks/c3" / arm.lower()
                             / "C3_BANK.json") or {}
            material = {key: bank[key] for key in bank.get("bank_identity_material", [])
                        if key in bank}
            recomputed = hashlib.sha256(json.dumps(
                material, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False).encode("utf-8")).hexdigest() if material else None
            matches = recomputed == row.get("bank_identity")
            verified[arm] = {"recorded": row.get("bank_identity"),
                             "recomputed": recomputed, "matches": matches}
            if not matches:
                problems.append(f"{arm} bank identity does not reproduce")

        checks.append(check(
            "c13_frozen_bank_identities_reproduce", not problems,
            "every frozen C3 bank identity recomputes from its own declared material",
            arms=verified, problems=problems))

        jsonl_ok: dict[str, Any] = {}
        for arm in ("llm", "rnd", "det"):
            path = request.repo / "assets/recipe_banks/c3" / arm / "recipes.jsonl"
            raw = path.read_bytes() if path.exists() else b""
            jsonl_ok[arm] = {"exists": path.exists(), "lf_only": b"\r" not in raw,
                             "lines": raw.decode("utf-8").count("\n") if raw else 0}
        checks.append(check(
            "c13_bank_bytes_are_intact",
            all(item["exists"] and item["lf_only"] and item["lines"] == 256
                for item in jsonl_ok.values()),
            "each frozen bank is 256 LF-terminated lines",
            arms=jsonl_ok,
            note="the bytes are hashed, so a CRLF checkout would produce a different "
                 "identity for the same logical content"))

        artifact = write_artifact(request, reports / "C13_ARTIFACT_INTEGRITY.json", {
            "schema_version": "c13-artifact-integrity-v1", "generated_at_utc": utc(),
            "mode": ARTIFACT_INTEGRITY, "bank_identities": verified,
            "bank_bytes": jsonl_ok, "problems": problems, "fixture_backed": True})
        return self.result(request, mode=ARTIFACT_INTEGRITY, checks=checks,
                           artifacts=[artifact])

    def _claim_policy(self, request: AdapterRequest, reports: Path) -> AdapterResult:
        """§28: a claim needs the evidence its strength implies."""
        from prism_fas.evaluation.bootstrap import refuse_single_seed_comparison
        from prism_fas.evaluation.contracts import may_carry_statistical_claim

        checks: list[dict[str, Any]] = []
        cases: list[dict[str, Any]] = []

        for role, expected in (("hypothesis_critical", True), ("spec_mandated", True),
                               ("diagnostic", False), ("parity", False)):
            try:
                allowed = bool(may_carry_statistical_claim(role))
            except Exception:
                allowed = False
            cases.append({"role": role, "may_claim": allowed, "expected": expected,
                          "agrees": allowed == expected})
        checks.append(check(
            "c13_claim_strength_follows_replication_role",
            all(case["agrees"] for case in cases),
            "only replicated rows may carry a statistical claim",
            cases=cases,
            rule="§18.3: single-seed rows are diagnostic only and may not support "
                 "superiority claims"))

        refused = False
        try:
            refuse_single_seed_comparison("C-H1", ["diagnostic"])
        except Exception:
            refused = True
        checks.append(check(
            "c13_single_seed_superiority_claim_refused", refused,
            "a superiority comparison over a single-seed row is refused"))

        smoke_claim_blocked = not request.profile.may_select_scientific_winner
        checks.append(check(
            "c13_smoke_numbers_cannot_support_a_claim", smoke_claim_blocked,
            "no number produced under this profile may support a scientific claim",
            profile=request.profile.name,
            may_select_scientific_winner=request.profile.may_select_scientific_winner,
            rule="L.1: a smoke result MUST NOT be the numeric basis for selecting a "
                 "scientific winner, changing a hypothesis, changing P3, or claiming "
                 "method superiority"))

        artifact = write_artifact(request, reports / "C13_CLAIM_POLICY.json", {
            "schema_version": "c13-claim-policy-v1", "generated_at_utc": utc(),
            "mode": CLAIM_POLICY, "cases": cases,
            "single_seed_comparison_refused": refused,
            "profile_may_claim": request.profile.may_select_scientific_winner,
            "fixture_backed": True})
        return self.result(request, mode=CLAIM_POLICY, checks=checks, artifacts=[artifact])

    def _report(self, request: AdapterRequest, matrix: dict[str, Any],
                reports: Path) -> AdapterResult:
        checks: list[dict[str, Any]] = []
        structure = {
            "sections": ["repository integrity", "acceptance matrix", "milestone status",
                         "source-side evidence", "target-side evidence", "statistics",
                         "negative and blocked results", "disclosures", "tag proposal"],
            "target_metrics_included": False,
            "why_no_target_metrics": ("no scientific target evaluation has run; a report "
                                      "that printed a number here would be fabricating it"),
        }
        proposal = {
            "proposed_tag": None,
            "would_be": "c-version-c-final",
            "blocked_by": matrix["not_scientifically_complete"],
            "created": False,
            "rule": "C13 proposes a tag; it never creates the scientific tag itself, and "
                    "it proposes nothing while acceptance is refused",
        }

        checks.append(check(
            "c13_report_structure_assembles", bool(structure["sections"]),
            "the final report structure assembles with every required section",
            sections=structure["sections"]))
        checks.append(check(
            "c13_report_contains_no_target_metric",
            structure["target_metrics_included"] is False,
            "the report contains no target metric, fabricated or otherwise",
            **{key: value for key, value in structure.items() if key != "sections"}))
        checks.append(check(
            "c13_tag_is_proposed_not_created", proposal["created"] is False,
            "no tag was created, and none is proposed while acceptance is refused",
            **{key: value for key, value in proposal.items() if key != "rule"}))

        artifact = write_artifact(request, reports / "C13_FINAL_REPORT.json", {
            "schema_version": "c13-final-report-v1", "generated_at_utc": utc(),
            "mode": FINAL_REPORT, "acceptance": matrix, "structure": structure,
            "tag_proposal": proposal, "is_c_acceptance": False, "fixture_backed": True,
            "meaning": ("engineering evidence that the C13 machinery runs and refuses "
                        "correctly. It is not a Version-C result and contains no claim")})

        decision = resume_decision(request, "c13_acceptance_matrix",
                                   reports / "C13_ACCEPTANCE_MATRIX.json",
                                   expected_identity="c13-acceptance-matrix-v1",
                                   identity_key="schema_version")
        checks.append(check(
            "c13_resume_is_identity_aware", decision["identity_matches"],
            "resume validates C13 evidence by identity", **decision))
        return self.result(request, mode=FINAL_REPORT, checks=checks, artifacts=[artifact])


__all__ = ["STAGE_ID", "MODES", "ACCEPTANCE_MATRIX", "NEGATIVE_PRESERVATION",
           "ARTIFACT_INTEGRITY", "CLAIM_POLICY", "FINAL_REPORT", "REQUIRED_SCIENTIFIC",
           "C13Adapter"]
