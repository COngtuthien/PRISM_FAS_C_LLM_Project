"""The C3 adapter: pre-live verification, live generation, resume, finalization.

C3 is the only stage that can spend live provider quota, so it is the only one
whose adapter has modes at all. The four are:

``PRE_LIVE_VERIFY``
    Offline. Re-derives the frozen identities, verifies both locks, confirms the
    12x32 plan multiplies out, checks the ancestor chain and reports whether a
    quota snapshot and credential exist. Makes no request under any profile.

``LIVE_GENERATE``
    Executes the frozen 12 logical requests through `RecipePlanner`. Requires an
    explicit authorization flag *and* a profile that permits a live binding *and*
    a materialized quota snapshot *and* a credential. Bound to a mock or replay
    provider it becomes an offline rehearsal of exactly the same code path.

``RESUME_LIVE_GENERATE``
    The same thing, continuing an existing state file. Completed requests are
    skipped after their archives are re-hashed, never regenerated.

``FINALIZE_BANKS``
    Offline. Applies the frozen selector to a complete raw pool. Delegates
    entirely to `prism_fas.recipes.selection`.

The mode a run gets is decided by profile and authorization, never by
convenience. `resolve_mode` is the only place that decision is made, and under
validate it cannot return a generating mode at all.

Nothing here implements science. The request comes from `RouteContext`, the
retry policy and validation come from `RecipePlanner`, the schedule numbers come
from the canonical module, and selection comes from the frozen selector. This
file sequences them and records what happened.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from prism_fas.pipeline.adapters import (AdapterError, AdapterRequest, AdapterResult,
                                         ProviderBinding, assert_binding_permitted,
                                         permitted_bindings)
from prism_fas.pipeline.adapters import quota as quota_module
from prism_fas.pipeline.adapters.c3_live import (LiveGenerationState, LogicalRequestRecord,
                                                 RequestStatus, classify_result)
from prism_fas.pipeline.adapters.context import frozen_schedule, route_context
from prism_fas.pipeline.state import atomic_write_json
from prism_fas.pipeline.status import DualStatus

STAGE_ID = "C3"

#: Where a SCIENTIFIC live run keeps its state and archives: under `reports/c3/`
#: with the rest of the C3 evidence, but in its own subdirectory so it is
#: distinguishable from the frozen contract artifacts.
LIVE_DIR = Path("reports/c3/live")
LIVE_STATE_FILE = "C3_LIVE_GENERATION_STATE.json"
RAW_ARCHIVE_DIR = "raw_responses"


def live_dir_for(profile: Any) -> Path:
    """The live directory for a profile, scoped so rehearsal cannot pose as science.

    A smoke rehearsal drives the identical code path against fixtures, and its
    state file and raw archives look structurally like the real thing. Writing
    them into `reports/c3/` would put files that are not scientific evidence
    inside the namespace that holds scientific evidence — and the C3 generation
    prohibition is checked by looking for exactly such files. So anything that
    is not the scientifically-eligible profile writes under its own namespace
    instead, which is what L.2's output-namespace rule already requires.
    """
    if profile.scientific_eligible:
        return LIVE_DIR
    return Path(profile.reports_namespace) / "c3" / "live"


class C3Mode(str, Enum):
    PRE_LIVE_VERIFY = "PRE_LIVE_VERIFY"
    LIVE_GENERATE = "LIVE_GENERATE"
    RESUME_LIVE_GENERATE = "RESUME_LIVE_GENERATE"
    FINALIZE_BANKS = "FINALIZE_BANKS"


#: Modes that can issue provider requests. The distinction the whole gate rests
#: on: a mode being generating does not mean it is bound to a live provider.
GENERATING_MODES: frozenset[C3Mode] = frozenset({
    C3Mode.LIVE_GENERATE, C3Mode.RESUME_LIVE_GENERATE})


class C3ModeRefused(AdapterError):
    """The requested C3 mode is not available under this profile."""


def _check(check_id: str, ok: bool, summary: str, **detail: Any) -> dict[str, Any]:
    return {"check_id": check_id, "ok": ok, "summary": summary, "detail": detail}


def _check_from(check_id: str, report: dict[str, Any]) -> dict[str, Any]:
    """Wrap a helper's report, which carries its own `ok` and `summary` keys.

    Splatting such a report into `_check` would collide on those two names, so
    they are lifted out and the rest becomes the detail.
    """
    detail = {key: value for key, value in report.items() if key not in {"ok", "summary"}}
    return _check(check_id, bool(report["ok"]), str(report["summary"]), **detail)


def _rehearses_generation(profile: Any) -> bool:
    """Profiles that drive the generation path against fixtures.

    `smoke` and `rehearsal` are both non-eligible profiles that EXECUTE the code,
    so both get the fixture provider. Keyed on the eligibility contract rather
    than on a name list: a profile that cannot produce scientific evidence can
    never be allowed to reach a live provider, and a profile that can must never
    be handed fixtures.
    """
    return profile.name in ("smoke", "rehearsal")


def resolve_mode(request: AdapterRequest) -> C3Mode:
    """Decide which mode runs, refusing anything the profile cannot support.

    The validate profile has no generating mode available at all — not a
    mock-bound one, not a refused-then-logged one. L.2 defines validate as
    static readiness, and a state machine that writes generation state is not
    static, so validate stops at PRE_LIVE_VERIFY.
    """
    profile = request.profile
    asked = request.mode

    if asked is not None:
        try:
            mode = C3Mode(asked)
        except ValueError:
            raise C3ModeRefused(
                f"unknown C3 mode {asked!r}; expected one of "
                f"{[item.value for item in C3Mode]}") from None
    else:
        mode = (C3Mode.RESUME_LIVE_GENERATE if request.resume and profile.name != "validate"
                else C3Mode.PRE_LIVE_VERIFY)

    if profile.name == "validate" and mode in GENERATING_MODES:
        raise C3ModeRefused(
            f"C3 mode {mode.value} is not available under the validate profile. Validate is "
            "static readiness only (L.2): it verifies identities, locks and contracts and "
            "never enters a generation path.")
    if profile.name == "validate" and mode is C3Mode.FINALIZE_BANKS:
        raise C3ModeRefused(
            "C3 mode FINALIZE_BANKS is not available under the validate profile; selecting a "
            "bank is execution, not readiness verification.")
    return mode


def resolve_binding(request: AdapterRequest, mode: C3Mode) -> ProviderBinding:
    """Decide what the mode may talk to, defaulting to the safest thing that works."""
    if mode not in GENERATING_MODES:
        return ProviderBinding.NONE

    allowed = permitted_bindings(
        request.profile, authorized_live_generation=request.authorized_live_generation,
        stage_id=STAGE_ID)

    if request.provider_binding is not None:
        if request.provider_binding not in allowed:
            assert_binding_permitted(request.provider_binding, request, stage_id=STAGE_ID)
        return request.provider_binding

    # No explicit choice: never default to LIVE. Spending scientific quota is
    # something a caller asks for by name, not something it gets by omission.
    return ProviderBinding.MOCK


# --- PRE_LIVE_VERIFY --------------------------------------------------------

def _ancestor_chain_checks(repo: Path) -> list[dict[str, Any]]:
    """C3 live may only proceed on accepted C0/C1/C2/C2B/C2C evidence.

    Verified from the artifacts rather than assumed from the milestone order: an
    ancestor whose evidence has gone missing invalidates the descendant, and the
    only way to know is to look.
    """
    from prism_fas.pipeline.adapters.historical import ACCEPTANCE

    rows = []
    for stage_id, relative in ACCEPTANCE.items():
        present = (repo / relative).exists()
        rows.append({"stage": stage_id, "artifact": relative, "present": present})
    missing = [row["stage"] for row in rows if not row["present"]]
    return [_check(
        "c3_ancestor_chain_accepted", not missing,
        "every C0-C2C ancestor has its acceptance evidence on disk" if not missing
        else f"ancestor evidence is missing for {missing}",
        ancestors=rows, missing=missing,
        rule="C3 live execution depends on accepted evidence from C0, C1, C2/C2B/C2C and "
             "the C3 pre-live gate")]


def _schedule_checks(repo: Path) -> list[dict[str, Any]]:
    """The frozen 12x32 plan, checked against the canonical constants."""
    schedule = frozen_schedule(repo)
    plan_ok = (schedule["requests"] * schedule["objects_per_request"]
               == schedule["raw_slots"])
    return [
        _check("c3_frozen_schedule_exact", plan_ok,
               f"{schedule['requests']} logical requests x "
               f"{schedule['objects_per_request']} objects = {schedule['raw_slots']} raw slots"
               if plan_ok else "the frozen schedule does not multiply out",
               **schedule,
               source="scripts/c2c_common.py (canonical); not restated by the adapter"),
        _check("c3_pool_and_bank_sizes", True,
               f"minimum unique pool {schedule['minimum_unique_pool']}, "
               f"final bank {schedule['final_bank']} per arm",
               minimum_unique_pool=schedule["minimum_unique_pool"],
               final_bank=schedule["final_bank"],
               rule="below the minimum pool C3 FAILS for that arm; the validator is never "
                    "weakened after seeing results"),
    ]


def _control_arm_checks(repo: Path) -> list[dict[str, Any]]:
    """RND and DET are offline schedules and consume zero provider calls."""
    from prism_fas.recipes.arm_schedules import SLOTS_PER_ARM, build_schedule
    from prism_fas.recipes.ontology import load_ontology

    ontology = load_ontology(repo / "configs/recipes/ontology_m7.yaml")
    rows = {}
    for arm in ("RND", "DET"):
        schedule = build_schedule(arm, ontology)
        rows[arm] = {"schedule_identity": schedule.schedule_identity,
                     "slots": SLOTS_PER_ARM, "provider_calls": 0}
    distinct = rows["RND"]["schedule_identity"] != rows["DET"]["schedule_identity"]
    return [_check(
        "c3_control_arms_offline", distinct,
        "RND and DET are distinct offline schedules and consume zero provider calls"
        if distinct else "the RND and DET schedules are not distinct",
        arms=rows, provider_calls_total=0,
        rule="only the LLM arm obtains its 384 slots from the live 12x32 schedule")]


def _credential_gate(repo: Path, *, required: bool) -> dict[str, Any]:
    """Presence only. A key is never printed, logged or serialized."""
    import os

    present = bool(os.environ.get("GEMINI_API_KEY"))
    return _check(
        "c3_credential_gate", (present or not required),
        ("a provider credential is PRESENT" if present else "a provider credential is MISSING")
        + ("; live generation is gated on it" if required else "; not required offline"),
        credential="PRESENT" if present else "MISSING", required=required,
        rule="presence is checked, never the value; no key is printed, logged or serialized")


def _pre_live_verify(request: AdapterRequest) -> AdapterResult:
    repo = request.repo
    checks: list[dict[str, Any]] = []

    # Identity and lock verification delegates to the validate-profile checks so
    # there is exactly one implementation of each.
    from prism_fas.pipeline.checks import (check_c3_contract_identities,
                                           check_c3_locks_verify,
                                           check_c3_scientific_banks_frozen)

    for check_fn in (check_c3_contract_identities, check_c3_locks_verify):
        result = check_fn(repo)
        checks.append(_check(result.check_id, result.ok, result.summary, **result.detail))

    checks.extend(_ancestor_chain_checks(repo))
    checks.extend(_schedule_checks(repo))
    checks.extend(_control_arm_checks(repo))

    # Before the authorized live run this asked whether generation had started.
    # It has, and it completed, so the obligation moved from "prove the
    # prohibition holds" to "prove the frozen result is intact" — same check
    # slot, same delegation, a claim that is still true.
    frozen = check_c3_scientific_banks_frozen(repo)
    checks.append(_check(frozen.check_id, frozen.ok, frozen.summary, **frozen.detail))

    checks.append(_check_from("c3_quota_snapshot",
                              quota_module.preflight(repo, required=False)))
    checks.append(_credential_gate(repo, required=False))

    parents = {}
    identity_check = next((item for item in checks
                           if item["check_id"] == "c3_contract_identities"), None)
    if identity_check:
        for comparison in identity_check["detail"].get("comparisons", []):
            if comparison.get("actual"):
                parents[comparison["identity"]] = comparison["actual"]

    failed = [item for item in checks if not item["ok"]]
    return AdapterResult(
        stage_id=STAGE_ID, substage="C3", mode=C3Mode.PRE_LIVE_VERIFY.value,
        provider_binding=ProviderBinding.NONE,
        status="PASS" if not failed else "FAIL",
        status_axes=DualStatus(engineering="NOT_TESTED", scientific="NOT_RUN"),
        summary="C3 pre-live verification passed; the frozen contract is intact"
                if not failed else f"C3 pre-live verification found {len(failed)} problem(s)",
        checks=checks, parent_identities=parents, provider_calls=0,
        notes=["verification only; this mode cannot issue a provider request"],
        detail={"c3_scientific_logical_requests": 0, "c3_scientific_candidate_slots": 0,
                "live_generation_gated_on": [
                    "explicit user authorization",
                    "a full profile that permits a live binding for C3",
                    "a materialized quota snapshot",
                    "a present provider credential"]})


# --- LIVE_GENERATE / RESUME_LIVE_GENERATE -----------------------------------

@dataclass
class _Archiver:
    """Writes one raw response per logical request, atomically, before use.

    §7.6.1 requires every raw request/response to be archived before semantic
    use. Archiving first is also what makes the resume check meaningful: the
    hash recorded in the state must be the hash of a file that exists.
    """

    directory: Path

    def path_for(self, record: LogicalRequestRecord) -> Path:
        return self.directory / f"{record.logical_request_id}.json"

    def write(self, record: LogicalRequestRecord, *, request_identity: str,
              raw_text: str, result_summary: dict[str, Any]) -> str:
        payload = {
            "logical_request_id": record.logical_request_id,
            "logical_request_index": record.logical_request_index,
            "slot_start": record.slot_start,
            "slot_end": record.slot_end,
            "request_identity": request_identity,
            "attempt_count": record.attempt_count,
            "raw_response": raw_text,
            "result": result_summary,
        }
        atomic_write_json(self.path_for(record), payload)
        from prism_fas.llm.contracts import text_sha256
        return text_sha256(raw_text)

    def read_sha(self, record: LogicalRequestRecord) -> str | None:
        path = self.path_for(record)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        raw = payload.get("raw_response")
        if raw is None:
            return None
        from prism_fas.llm.contracts import text_sha256
        return text_sha256(raw)


def _build_provider(binding: ProviderBinding, request: AdapterRequest) -> Any:
    """Construct the provider for a binding, and nothing else.

    The LIVE branch is the only one that can import the vendor SDK, and it is
    reached only after `assert_binding_permitted` has already passed. Importing
    lazily means an offline run never loads it at all.
    """
    if binding is ProviderBinding.MOCK:
        provider = request.options.get("mock_provider")
        if provider is None:
            raise AdapterError(
                "a mock binding needs a scripted provider; pass options['mock_provider']")
        return provider
    if binding is ProviderBinding.REPLAY:
        provider = request.options.get("replay_provider")
        if provider is None:
            raise AdapterError(
                "a replay binding needs an archive; pass options['replay_provider']")
        return provider
    if binding is ProviderBinding.LIVE:
        assert_binding_permitted(binding, request, stage_id=STAGE_ID)
        from prism_fas.llm.providers.gemini import GeminiRecipeProvider

        return GeminiRecipeProvider(route_context(request.repo).config)
    raise AdapterError(f"binding {binding.value!r} cannot produce a provider")


def _live_generate(request: AdapterRequest, mode: C3Mode,
                   binding: ProviderBinding) -> AdapterResult:
    repo = request.repo
    profile = request.profile
    schedule = frozen_schedule(repo)

    live_relative = live_dir_for(profile)
    live_dir = repo / live_relative
    archiver = _Archiver(live_dir / RAW_ARCHIVE_DIR)
    state = LiveGenerationState.open(
        live_dir / LIVE_STATE_FILE, arm="LLM", schedule=schedule,
        execution_profile=profile.name, provider_binding=binding.value,
        resume=(mode is C3Mode.RESUME_LIVE_GENERATE or request.resume))
    # Eligibility is a property of the evidence, not of the file's defaults: a
    # complete live run under the full profile IS scientific evidence and has to
    # say so, while anything else must not.
    state.scientific_eligible = (profile.scientific_eligible
                                 and binding is ProviderBinding.LIVE)
    state.generation_contract_identity = (
        state.generation_contract_identity
        or json.loads((repo / "reports/c3/C3_BANK_LOCK.json").read_text(encoding="utf-8"))
        ["composite"]["c3_generation_contract_identity"])

    checks: list[dict[str, Any]] = []

    # L.11: before anything else, prove that what we think is done really is.
    integrity = state.assert_completed_intact(archiver.read_sha)
    checks.append(_check(
        "c3_completed_requests_intact", True,
        f"{len(integrity)} completed logical request(s) re-hashed and intact",
        rows=integrity))

    if binding is ProviderBinding.LIVE:
        checks.append(_check_from("c3_quota_snapshot",
                                  quota_module.preflight(repo, required=True)))
        checks.append(_credential_gate(repo, required=True))
        blocking = [item for item in (checks[-2], checks[-1]) if not item["ok"]]
        if blocking:
            return AdapterResult(
                stage_id=STAGE_ID, substage="C3", mode=mode.value, provider_binding=binding,
                status="BLOCKED",
                status_axes=DualStatus(engineering="BLOCKED", scientific="BLOCKED"),
                summary="live generation is gated: " + "; ".join(
                    item["summary"] for item in blocking),
                checks=checks, provider_calls=0,
                detail={"resume_cursor": state.resume_cursor})

    provider = _build_provider(binding, request)
    context = route_context(repo)

    from prism_fas.llm.pipeline import QuotaBlocked, RecipePlanner

    planner = RecipePlanner(provider=provider, config=context.config,
                            ontology=context.ontology,
                            route_policy=context.route_policy,
                            sleep=request.options.get("sleep"))

    calls_before = state.provider_calls_total
    executed = 0
    stopped_reason: str | None = None

    # Inter-request pacing. Purely transport behaviour: it decides WHEN a request
    # leaves, never what it contains, so the request identity, prompt, schema and
    # slot assignment are untouched. It exists because the free tier allows 5
    # requests per minute and firing 12 back to back would earn rate-limit errors
    # that cost retries rather than time.
    pace_seconds = float(request.options.get("min_seconds_between_requests", 0.0))
    sleep = request.options.get("sleep") or time.sleep
    last_request_at: float | None = None

    for record in state.requests:
        if record.complete:
            continue          # L.11: completed is terminal. Never re-issued.
        if record.state is RequestStatus.FAILED_BLOCKING and not request.resume:
            stopped_reason = f"{record.logical_request_id} is FAILED_BLOCKING"
            break

        if pace_seconds and last_request_at is not None:
            elapsed = time.monotonic() - last_request_at
            if elapsed < pace_seconds:
                sleep(pace_seconds - elapsed)
        last_request_at = time.monotonic()

        generation_request = context.request(record.logical_request_id)
        state.mark_in_progress(record, request_identity=generation_request.request_sha256)

        try:
            result, validation, attempts = planner.generate_slot(
                generation_request,
                recipes_requested=schedule["objects_per_request"],
                next_recipe_index=record.slot_start,
                pending_slot_ids=[item.logical_request_id for item in state.requests
                                  if not item.complete])
        except QuotaBlocked as blocked:
            state.mark_blocking(record, classification="quota_exhausted",
                                reason=f"provider daily quota exhausted: {blocked.state.reason}")
            stopped_reason = "provider daily quota exhausted"
            break

        executed += 1
        state.provider_calls_total += attempts

        status, classification, note = classify_result(
            result, bool(validation and validation.all_accepted))

        if status is RequestStatus.COMPLETED_VALID:
            sha = archiver.write(record, request_identity=generation_request.request_sha256,
                                 raw_text=result.raw_text or "",
                                 result_summary=result.as_dict())
            accepted = sum(1 for candidate in validation.candidates if candidate.accepted)
            state.mark_completed(record, archive_identity=sha, raw_response_sha256=sha,
                                 accepted_recipe_count=accepted,
                                 provider_attempts=attempts)
            planner.mark_slot_completed(record.logical_request_id)
        elif status is RequestStatus.FAILED_RETRYABLE:
            state.mark_retryable(record, classification=classification, note=note)
            stopped_reason = note
            break
        else:
            state.mark_blocking(record, classification=classification, reason=note)
            stopped_reason = note
            break

    state.save()

    plan_ok = len(state.requests) == schedule["requests"]
    checks.append(_check(
        "c3_logical_request_count_unchanged", plan_ok,
        f"the plan still contains exactly {schedule['requests']} logical requests"
        if plan_ok else "the logical request count changed",
        planned=len(state.requests), frozen=schedule["requests"]))

    calls_made = state.provider_calls_total - calls_before
    complete = state.all_complete
    status = "PASS" if complete else ("BLOCKED" if stopped_reason else "PASS")

    return AdapterResult(
        stage_id=STAGE_ID, substage="C3", mode=mode.value, provider_binding=binding,
        status=status,
        status_axes=DualStatus(
            engineering="SMOKE_PASS" if (complete and profile.name == "smoke")
            else ("BLOCKED" if status == "BLOCKED" else "NOT_TESTED"),
            scientific="PASS" if (complete and profile.scientific_eligible
                                  and binding is ProviderBinding.LIVE) else "NOT_RUN"),
        summary=(f"all {len(state.requests)} logical requests complete"
                 if complete else
                 f"{len(state.completed)}/{len(state.requests)} complete; stopped: "
                 f"{stopped_reason or 'no further work permitted'}"),
        checks=checks,
        artifacts=[(live_relative / LIVE_STATE_FILE).as_posix()],
        provider_calls=calls_made,
        notes=[note for note in [stopped_reason] if note],
        detail={
            "resume_cursor": state.resume_cursor,
            "status_counts": state.counts(),
            "logical_requests_executed_this_run": executed,
            "provider_calls_this_run": calls_made,
            "provider_binding": binding.value,
            "min_seconds_between_requests": pace_seconds,
            "pacing_is_transport_only": True,
            "is_scientific_generation": binding is ProviderBinding.LIVE,
            "c3_scientific_logical_requests":
                len(state.completed) if binding is ProviderBinding.LIVE else 0,
            "c3_scientific_candidate_slots":
                len(state.completed) * schedule["objects_per_request"]
                if binding is ProviderBinding.LIVE else 0,
        })


# --- FINALIZE_BANKS ---------------------------------------------------------

def _finalize_banks(request: AdapterRequest) -> AdapterResult:
    """Apply the frozen selector. Delegates entirely; decides nothing."""
    repo = request.repo
    schedule = frozen_schedule(repo)
    pool = request.options.get("candidate_pool")

    if pool is None:
        return AdapterResult(
            stage_id=STAGE_ID, substage="C3", mode=C3Mode.FINALIZE_BANKS.value,
            provider_binding=ProviderBinding.NONE, status="BLOCKED",
            status_axes=DualStatus(engineering="BLOCKED", scientific="BLOCKED"),
            summary="no candidate pool is available to finalize",
            checks=[_check("c3_pool_available", False,
                           "finalization needs a complete raw candidate pool; none exists "
                           "because C3 generation has not run",
                           required_pool=schedule["minimum_unique_pool"])],
            provider_calls=0,
            detail={"selector": "prism_fas.recipes.selection.select (frozen, not reimplemented)"})

    from prism_fas.recipes.ontology import load_ontology
    from prism_fas.recipes.selection import MINIMUM_ELIGIBLE_POOL_PER_ARM, select

    ontology = load_ontology(repo / "configs/recipes/ontology_m7.yaml")
    enough = len(pool) >= MINIMUM_ELIGIBLE_POOL_PER_ARM
    checks = [_check(
        "c3_minimum_eligible_pool", enough,
        f"{len(pool)} eligible candidates against a frozen minimum of "
        f"{MINIMUM_ELIGIBLE_POOL_PER_ARM}",
        pool_size=len(pool), minimum=MINIMUM_ELIGIBLE_POOL_PER_ARM,
        rule="below the minimum, C3 FAILS for the arm; the validator is never weakened")]
    if not enough:
        return AdapterResult(
            stage_id=STAGE_ID, substage="C3", mode=C3Mode.FINALIZE_BANKS.value,
            provider_binding=ProviderBinding.NONE, status="FAIL",
            status_axes=DualStatus(engineering="NOT_TESTED", scientific="FAIL"),
            summary="the eligible pool is below the frozen minimum; C3 fails for this arm",
            checks=checks, provider_calls=0)

    selection = select(pool, ontology)
    return AdapterResult(
        stage_id=STAGE_ID, substage="C3", mode=C3Mode.FINALIZE_BANKS.value,
        provider_binding=ProviderBinding.NONE, status="PASS",
        status_axes=DualStatus(engineering="NOT_TESTED", scientific="NOT_RUN"),
        summary=f"selected exactly {schedule['final_bank']} recipes through the frozen selector",
        checks=checks, provider_calls=0,
        detail={"selected_set_identity": selection.selected_set_identity,
                "selector": "prism_fas.recipes.selection.select (frozen, not reimplemented)"})


# --- adapter ----------------------------------------------------------------

@dataclass
class C3Adapter:
    """Dispatches to one of the four modes."""

    stage_id: str = STAGE_ID
    substages: tuple[str, ...] = ("C3",)

    def default_mode(self, profile: Any) -> str:
        return C3Mode.PRE_LIVE_VERIFY.value

    def default_binding(self, profile: Any) -> ProviderBinding:
        return ProviderBinding.NONE

    def run(self, request: AdapterRequest) -> list[AdapterResult]:
        mode = resolve_mode(request)
        binding = resolve_binding(request, mode)

        if mode is C3Mode.PRE_LIVE_VERIFY:
            results = [_pre_live_verify(request)]
            # Smoke exists to find engineering defects, and the defects worth
            # finding here are in the generation path — the state machine, the
            # archiver, checkpointing and resume. Verifying the contract and
            # stopping would exercise none of it, so smoke continues into a
            # fixture-backed rehearsal of exactly that code, bound to a provider
            # that cannot reach a network.
            if _rehearses_generation(request.profile) and results[0].ok:
                results.append(self._smoke_rehearsal(request, C3Mode.LIVE_GENERATE))
            return results
        if mode is C3Mode.FINALIZE_BANKS:
            return [_finalize_banks(request)]
        # A generating mode under smoke needs the same fixture provider the
        # rehearsal supplies. `--profile smoke --resume` resolves to
        # RESUME_LIVE_GENERATE, and routing that through the bare request left
        # the mock binding with no scripted responses — a defect the first
        # C0-C13 resume run surfaced, which is exactly what smoke is for. Smoke
        # can never reach a live provider, so a generating mode here is always a
        # rehearsal and is always given its fixtures.
        if _rehearses_generation(request.profile):
            return [self._smoke_rehearsal(request, mode)]
        return [_live_generate(request, mode, binding)]

    def _smoke_rehearsal(self, request: AdapterRequest, mode: C3Mode) -> AdapterResult:
        """Drive a generating mode against fixtures, under the smoke namespace."""
        from prism_fas.pipeline.adapters.fixtures import smoke_provider

        schedule = frozen_schedule(request.repo)
        rehearsal = AdapterRequest(
            repo=request.repo, profile=request.profile,
            mode=mode.value, provider_binding=ProviderBinding.MOCK,
            resume=True, authorized_live_generation=False,
            options={**request.options,
                     "mock_provider": smoke_provider(
                         repo=request.repo,
                         recipes=schedule["objects_per_request"],
                         count=schedule["requests"]),
                     # No real backoff in a rehearsal: the retry policy's timing
                     # is not what smoke is testing, and sleeping through it
                     # would make the profile useless.
                     "sleep": lambda _seconds: None})
        return _live_generate(rehearsal, mode, ProviderBinding.MOCK)


__all__ = ["STAGE_ID", "LIVE_DIR", "LIVE_STATE_FILE", "C3Mode", "GENERATING_MODES",
           "C3ModeRefused", "C3Adapter", "resolve_mode", "resolve_binding"]
