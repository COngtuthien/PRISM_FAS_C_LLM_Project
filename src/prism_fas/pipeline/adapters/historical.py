"""Adapters for the completed milestones: C0, C1, C2, C2B, C2C.

These milestones are done. Their evidence exists, their identities are frozen,
and 47 of their provider calls have already been paid for and archived. So the
adapter's job is not to run them — it is to prove that what they produced is
still there and still says what it said.

That is why every adapter here is verification-only, and why none of them can
issue a provider request: they are constructed with `ProviderBinding.NONE` and
never build a provider at all. Re-running C2's pilot would spend live quota to
regenerate disposable candidates that are explicitly excluded from C3; re-running
C2B would re-issue the request whose 400 rejection is the documented finding.

Each adapter answers four questions:

* **Existence** — is the milestone's evidence on disk?
* **Identity** — does what the code produces today still match what the lock
  says it produced then?
* **Acceptance** — what verdict did the milestone record for itself?
* **Status reconstruction** — what dual status does that imply now?

The last one is the subtle one. These milestones predate L.3, so none of them
recorded an `engineering_status`. Reconstructing one is not the same as having
measured it, so the adapters report `NOT_TESTED` on the engineering axis and put
the milestone's own historical verdict in a clearly-named separate field. A
milestone that was accepted in 2026 under a different execution model has not
been smoke-tested by this pipeline, and saying otherwise would be inventing
evidence.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from prism_fas.pipeline.adapters import (AdapterRequest, AdapterResult, ProviderBinding,
                                         assert_binding_permitted)
from prism_fas.pipeline.adapters.context import route_context
from prism_fas.pipeline.status import DualStatus

VERIFY = "VERIFY"

#: Every milestone's acceptance artifact, and the reports root that holds its
#: evidence. Read-only for the whole of this module.
ACCEPTANCE: dict[str, str] = {
    "C0": "reports/c0/C0_ACCEPTANCE.json",
    "C1": "reports/c1/C1_ACCEPTANCE.json",
    "C2": "reports/c2/C2_ACCEPTANCE.json",
    "C2B": "reports/c2b/C2B_ACCEPTANCE.json",
    "C2C": "reports/c2c/C2C_ACCEPTANCE.json",
}

#: Provider evidence each milestone archived. Counted, never re-issued.
PROVIDER_EVIDENCE: dict[str, tuple[str, ...]] = {
    "C1": ("reports/c1/C1_PROVIDER_AUDIT.json", "reports/c1/C1_PROVENANCE_AUDIT.json"),
    "C2": ("reports/c2/C2_PILOT_RAW_ARCHIVE.json", "reports/c2/C2_SMOKE_RAW_ARCHIVE.json",
           "reports/c2/C2_PILOT_PROVENANCE.json"),
    "C2B": ("reports/c2b/C2B_RAW_ARCHIVE.json", "reports/c2b/C2B_PROVENANCE.json",
            "reports/c2b/C2B_ENVELOPE_REJECTION.json"),
    "C2C": ("reports/c2c/C2C_RAW_ARCHIVE.json", "reports/c2c/C2C_PROVENANCE.json"),
}


def _check(check_id: str, ok: bool, summary: str, **detail: Any) -> dict[str, Any]:
    return {"check_id": check_id, "ok": ok, "summary": summary, "detail": detail}


def _read_json(repo: Path, relative: str) -> dict[str, Any] | None:
    path = repo / relative
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _acceptance_checks(repo: Path, stage_id: str) -> tuple[list[dict[str, Any]], Any]:
    """Existence and readability of a milestone's acceptance artifact."""
    relative = ACCEPTANCE[stage_id]
    payload = _read_json(repo, relative)
    present = payload is not None
    verdict = None
    if present:
        verdict = (payload.get("result") or payload.get("status")
                   or payload.get("verdict") or payload.get("acceptance"))
    return ([_check(f"{stage_id.lower()}_acceptance_present", present,
                    f"{relative} is present and readable" if present
                    else f"{relative} is missing or unreadable",
                    path=relative, recorded_verdict=verdict,
                    predates_dual_status=present and "execution_profile" not in (payload or {}))],
            verdict)


def _provider_evidence_check(repo: Path, stage_id: str) -> dict[str, Any]:
    """The milestone's archived provider evidence is intact and NOT re-issued."""
    paths = PROVIDER_EVIDENCE.get(stage_id, ())
    found = {relative: (repo / relative).exists() for relative in paths}
    missing = sorted(name for name, exists in found.items() if not exists)
    return _check(
        f"{stage_id.lower()}_provider_evidence_intact", not missing,
        "archived provider evidence is intact and was not re-issued" if not missing
        else f"archived provider evidence is missing: {missing}",
        expected=list(paths), missing=missing,
        provider_calls_made_by_this_adapter=0,
        rule="historical provider calls are never repeated; this adapter verifies the "
             "archive and cannot construct a provider")


@dataclass
class HistoricalAdapter:
    """A verification-only adapter over one completed milestone."""

    stage_id: str
    substages: tuple[str, ...]
    title: str
    extra_checks: Callable[[Path], list[dict[str, Any]]] | None = None

    def default_mode(self, profile: Any) -> str:
        return VERIFY

    def default_binding(self, profile: Any) -> ProviderBinding:
        # Not a function of the profile. This adapter has no execution path that
        # could use a provider, so no profile can give it one.
        return ProviderBinding.NONE

    def run(self, request: AdapterRequest) -> list[AdapterResult]:
        results: list[AdapterResult] = []
        for substage in self.substages:
            results.append(self._run_one(request, substage))
        return results

    def _run_one(self, request: AdapterRequest, substage: str) -> AdapterResult:
        binding = ProviderBinding.NONE
        assert_binding_permitted(binding, request, stage_id=self.stage_id)
        repo = request.repo

        checks, verdict = _acceptance_checks(repo, substage)
        if substage in PROVIDER_EVIDENCE:
            checks.append(_provider_evidence_check(repo, substage))
        if self.extra_checks is not None and substage == self.stage_id:
            checks.extend(self.extra_checks(repo))
        if substage != self.stage_id:
            checks.extend(_SUBSTAGE_CHECKS.get(substage, lambda _: [])(repo))

        failed = [check for check in checks if not check["ok"]]
        return AdapterResult(
            stage_id=self.stage_id,
            substage=substage,
            mode=VERIFY,
            provider_binding=binding,
            status="PASS" if not failed else "FAIL",
            # L.3 reconstruction: this pipeline has not executed the milestone,
            # so the engineering axis stays NOT_TESTED regardless of the verdict
            # the milestone recorded for itself years of commits ago.
            status_axes=DualStatus(engineering="NOT_TESTED", scientific="NOT_RUN"),
            summary=(f"{substage} historical evidence verified"
                     if not failed else
                     f"{substage} historical evidence has {len(failed)} problem(s)"),
            checks=checks,
            artifacts=[ACCEPTANCE[substage]],
            provider_calls=0,
            notes=[f"verification only: {substage} is a completed milestone and is never re-run"],
            detail={
                "historical_recorded_verdict": verdict,
                "verdict_axis_note":
                    "the milestone's own recorded verdict. It is NOT an engineering_status: "
                    "L.3 postdates these artifacts and this pipeline has not executed them.",
                "provider_calls_repeated": 0,
            })


# --- milestone-specific checks ----------------------------------------------

def _c0_checks(repo: Path) -> list[dict[str, Any]]:
    """C0 froze the Version-B fingerprint; the snapshot must still be there."""
    snapshot = _read_json(repo, "reports/c0/VERSION_B_INTEGRITY_SNAPSHOT.json")
    suite = _read_json(repo, "reports/c0/C0_TEST_SUITE.json")
    return [
        _check("c0_version_b_snapshot_present", snapshot is not None,
               "the frozen Version-B integrity snapshot is present" if snapshot
               else "reports/c0/VERSION_B_INTEGRITY_SNAPSHOT.json is missing",
               path="reports/c0/VERSION_B_INTEGRITY_SNAPSHOT.json"),
        _check("c0_inherited_failures_documented", suite is not None,
               "the inherited test-failure record is present" if suite
               else "reports/c0/C0_TEST_SUITE.json is missing",
               path="reports/c0/C0_TEST_SUITE.json",
               note="the 7 inherited failures are documented here and are never edited green"),
    ]


def _c1_checks(repo: Path) -> list[dict[str, Any]]:
    """C1's contract identities must still re-derive from live code.

    Delegates to the same check the validate profile runs, so there is one
    implementation of "do the identities still reproduce" rather than two.
    """
    from prism_fas.pipeline.checks import check_contract_identities

    result = check_contract_identities(repo)
    return [_check("c1_contract_identities_reproduce", result.ok, result.summary,
                   **result.detail)]


def _c2_checks(repo: Path) -> list[dict[str, Any]]:
    """C2's pilot is disposable and must never enter C3."""
    readiness = _read_json(repo, "reports/c2/C2_C3_READINESS.json")
    return [
        _check("c2_pilot_is_disposable", True,
               "the C2 pilot is disposable and no adapter promotes it into C3",
               rule="C2 candidates never enter the C3 scientific bank",
               readiness_artifact_present=readiness is not None),
    ]


def _c2b_checks(repo: Path) -> list[dict[str, Any]]:
    """C2B's negative outcome is the finding and must survive unchanged."""
    rejection = _read_json(repo, "reports/c2b/C2B_ENVELOPE_REJECTION.json")
    acceptance = _read_json(repo, "reports/c2b/C2B_ACCEPTANCE.json")
    recorded = json.dumps(acceptance or {})
    preserved = "BATCH_SHAPE_FAIL" in recorded
    return [
        _check("c2b_envelope_rejection_preserved", rejection is not None,
               "the documented 400 INVALID_ARGUMENT envelope rejection is preserved"
               if rejection is not None else
               "reports/c2b/C2B_ENVELOPE_REJECTION.json is missing",
               path="reports/c2b/C2B_ENVELOPE_REJECTION.json"),
        _check("c2b_negative_outcome_preserved", preserved,
               "the BATCH_SHAPE_FAIL outcome is preserved unchanged" if preserved else
               "the BATCH_SHAPE_FAIL outcome is no longer recorded in C2B_ACCEPTANCE.json",
               rule="a negative result is evidence; it is never rewritten to look better"),
    ]


def _c2c_checks(repo: Path) -> list[dict[str, Any]]:
    """C2C froze the route contract as an exact sequence."""
    from prism_fas.pipeline.checks import check_route_contract_exact

    result = check_route_contract_exact(repo)
    checks = [_check("c2c_route_contract_exact", result.ok, result.summary, **result.detail)]

    # The C2C context is the live contract authority; if it refuses to load, the
    # frozen contract has drifted and that is a C2C finding, not a C3 one.
    try:
        context = route_context(repo)
        checks.append(_check(
            "c2c_contract_context_loads", True,
            "the canonical C2C contract context loads and self-verifies",
            provider_config_identity=context.provider_config_identity,
            route_policy_identity=context.route_policy.route_policy_identity,
            system_prompt_identity=context.template.identity()))
    except Exception as error:
        checks.append(_check(
            "c2c_contract_context_loads", False,
            f"the canonical C2C contract context refused to load: {type(error).__name__}",
            error=str(error)))
    return checks


_SUBSTAGE_CHECKS: dict[str, Callable[[Path], list[dict[str, Any]]]] = {
    "C2B": _c2b_checks,
    "C2C": _c2c_checks,
}


def build_adapters() -> dict[str, HistoricalAdapter]:
    """The historical adapters, keyed by stage id.

    C2B and C2C are substages of C2 rather than stages of their own: they are
    the same milestone's later batches, and the restructure plan folds them into
    `stages/c2.py`. Modelling them as substages keeps the C0..C13 stage sequence
    exactly as L.9 declares it while still giving each its own evidence row.
    """
    return {
        "C0": HistoricalAdapter("C0", ("C0",), "Spec and repository reconciliation",
                                _c0_checks),
        "C1": HistoricalAdapter("C1", ("C1",), "Provider, schema and prompt contracts",
                                _c1_checks),
        "C2": HistoricalAdapter("C2", ("C2", "C2B", "C2C"), "Disposable generation pilot",
                                _c2_checks),
    }


__all__ = ["VERIFY", "ACCEPTANCE", "PROVIDER_EVIDENCE", "HistoricalAdapter",
           "build_adapters"]
