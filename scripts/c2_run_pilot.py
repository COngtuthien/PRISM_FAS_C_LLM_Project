"""C2 disposable Gemini recipe pilot: exactly 32 slots.

    python scripts/c2_run_pilot.py

`pilot_000` .. `pilot_031`. Exactly 32 slots, fixed before the first call. A
failed attempt never creates a new slot: retries stay attached to the slot that
produced them, and every attempt - accepted, rejected or errored - is archived.

These are DISPOSABLE C2 development recipes. They must never enter the C3 384
candidate slots, the final 256-recipe LLM bank, detector training or any
synthetic training bank.

Retry policy is the frozen C1 one, unchanged:

    semantic invalid  -> at most 2 retries (3 semantic attempts per slot)
    transport / 5xx   -> bounded transport attempts, separate budget
    transient 429     -> bounded exponential backoff, same frozen request
    quota_exceeded    -> checkpoint, write API_QUOTA_BLOCKED.json, stop

Nothing here repairs a semantic field. An invalid candidate is recorded exactly
as the provider produced it and the only recovery is another candidate under the
same frozen contract.
"""
from __future__ import annotations

import sys
import time

from c2_pilot_common import (RECIPES_PER_PILOT_SLOT, REPORTS, FrozenContext, RecordingProvider,
                             api_key_present, git, read_json, replay_fields, utc_now, write_json,
                             write_raw_response_files)

from prism_fas.llm.contracts import ErrorClass
from prism_fas.llm.pipeline import QuotaBlocked, RecipePlanner, compile_accepted
from prism_fas.llm.provenance import build_provenance
from prism_fas.llm.providers.gemini import GeminiRecipeProvider
from prism_fas.recipes.compile import CompileError

PILOT_SLOT_COUNT = 32
PILOT_BANK_ID = "c2-pilot-disposable"

#: Client-side pacing between slots. This is operational scheduling, NOT part of
#: the frozen retry policy and not a scientific parameter: it changes when a
#: request is sent, never what is sent. The live Free Tier limit for the frozen
#: model replenishes over minutes (observed 2026-08-10: a 429 body hinting
#: "retry in 8-18s" against a 20-request window), so pacing keeps the pilot under
#: the limit instead of relying on the bounded retry budget to absorb it.
PILOT_MIN_CALL_INTERVAL_SECONDS = 75.0
PILOT_MAX_CALL_INTERVAL_SECONDS = 300.0
PACING_GROWTH = 1.4
PASS_COOLDOWN_SECONDS = 180.0
MAX_PASSES = 12
STATE_PATH = REPORTS / "C2_PILOT_STATE.json"
ARCHIVE_PATH = REPORTS / "C2_PILOT_RAW_ARCHIVE.json"
PROVENANCE_PATH = REPORTS / "C2_PILOT_PROVENANCE.json"

SLOT_IDS = [f"pilot_{index:03d}" for index in range(PILOT_SLOT_COUNT)]

#: Non-retryable classes that stop the whole pilot with a named block status.
HARD_BLOCK = {
    ErrorClass.AUTH: "BLOCKED_AUTH",
    ErrorClass.MODEL_UNAVAILABLE: "BLOCKED_MODEL",
    ErrorClass.QUOTA_EXHAUSTED: "BLOCKED_QUOTA",
}


def pace(last_started: float | None, interval: float) -> None:
    """Hold the minimum interval between slots. Never changes what is sent."""
    if last_started is None:
        return
    remaining = interval - (time.monotonic() - last_started)
    if remaining > 0:
        time.sleep(remaining)


def is_transport_only_failure(slot: dict) -> bool:
    """A recorded slot that never actually received a response.

    An earlier revision of this runner recorded such a slot as `exhausted`,
    which would have counted a rate-limit condition as semantic retry
    exhaustion. It is reopened on resume; its attempts stay in the archive.
    """
    if slot.get("final_status") != "exhausted":
        return False
    if slot.get("responses_received") is not None:
        return slot["responses_received"] == 0
    error = slot.get("final_error") or {}
    return error.get("error_class") in {"rate_limit", "transport", "server_error"}


def load_state() -> dict:
    """Resume state. Completed slots are never regenerated."""
    if STATE_PATH.exists() and ARCHIVE_PATH.exists():
        state = read_json(STATE_PATH)
        archive = read_json(ARCHIVE_PATH)
        kept, reopened = [], []
        for slot in state.get("slots", []):
            (reopened if is_transport_only_failure(slot) else kept).append(slot)
        for slot in reopened:
            print(f"reopening {slot['slot_id']}: it never received a response "
                  f"({(slot.get('final_error') or {}).get('error_class')}), so it is a pending "
                  "slot, not a semantic failure")
        records = list(archive.get("records", []))
        seen: dict[str, int] = {}
        for record in records:              # backfill the per-slot call counter
            seen[record["slot_id"]] = seen.get(record["slot_id"], 0) + 1
            record.setdefault("sequence", seen[record["slot_id"]])
        return {"slots": kept,
                "records": records,
                "provenance": list(read_json(PROVENANCE_PATH).get("records", []))
                if PROVENANCE_PATH.exists() else []}
    return {"slots": [], "records": [], "provenance": []}


def persist(context: FrozenContext, slots: list[dict], records: list[dict],
            provenance: list[dict], status: str, commit: str) -> None:
    archived = [record for record in records if record["raw_text"] is not None]
    if archived:
        write_raw_response_files(archived, "pilot")
    write_json(ARCHIVE_PATH, {
        "schema_version": "c2-pilot-raw-archive-v1",
        "note": "verbatim provider output for every pilot attempt, in call order. The same bytes "
                "are on disk under reports/c2/raw_responses/pilot/ (git-ignored by repository "
                "policy); this committed copy is what the offline replay tests read.",
        "disposable": True,
        "enters_c3_or_final_bank": False,
        "record_count": len(records),
        "records": records,
        "replay_records": [replay_fields(record) for record in archived]})
    write_json(PROVENANCE_PATH, {
        "schema_version": "c2-pilot-provenance-v1",
        "note": "one immutable record per provider attempt, successful or not",
        "record_count": len(provenance),
        "records": provenance})
    write_json(STATE_PATH, {
        "schema_version": "c2-pilot-state-v1",
        "generated_at_utc": utc_now(),
        "generator_code_commit": commit,
        "status": status,
        "slot_count": PILOT_SLOT_COUNT,
        "slot_ids": SLOT_IDS,
        "completed_slot_ids": [slot["slot_id"] for slot in slots],
        "pending_slot_ids": [slot_id for slot_id in SLOT_IDS
                             if slot_id not in {slot["slot_id"] for slot in slots}],
        "bank_id": PILOT_BANK_ID,
        "recipes_per_slot": RECIPES_PER_PILOT_SLOT,
        "frozen_contract": context.as_frozen_record(),
        "slots": slots})


def main() -> int:
    context = FrozenContext()
    config = context.config
    commit = git("rev-parse", "HEAD")

    if not api_key_present(config):
        print("BLOCKED_AUTH: no credential in the environment")
        return 2

    state = load_state()
    slots: list[dict] = state["slots"]
    records: list[dict] = state["records"]
    provenance: list[dict] = state["provenance"]
    done = {slot["slot_id"] for slot in slots}
    if done:
        print(f"resuming: {len(done)} slot(s) already complete and preserved")

    provider = RecordingProvider(GeminiRecipeProvider(config), phase="pilot")
    planner = RecipePlanner(provider=provider, config=config, ontology=context.ontology)

    # A resumed run must keep the duplicate registry of the slots it inherited,
    # or a repeat of an already-accepted recipe would be scored as unique.
    for slot in slots:
        if slot.get("content_identity"):
            planner.register(slot["content_identity"], slot["slot_id"])
        planner.mark_slot_completed(slot["slot_id"])

    status = "COMPLETE"
    last_call_started: float | None = None
    interval = PILOT_MIN_CALL_INTERVAL_SECONDS
    stop = False

    for pass_number in range(1, MAX_PASSES + 1):
        remaining = [slot_id for slot_id in SLOT_IDS if slot_id not in done]
        if not remaining or stop:
            break
        if pass_number > 1:
            print(f"\n--- pass {pass_number}: {len(remaining)} slot(s) still pending, "
                  f"pacing at {interval:.0f}s ---")
        advanced = False

        for slot_id in remaining:
            index = SLOT_IDS.index(slot_id)
            pending = [name for name in SLOT_IDS if name not in done]
            request = context.request(slot_id, metadata={"phase": "c2_pilot", "disposable": True,
                                                         "enters_c3": False})
            pace(last_call_started, interval)
            last_call_started = time.monotonic()
            before = len(provider.records)
            print(f"[{utc_now()}] {slot_id} ({index + 1}/{PILOT_SLOT_COUNT})", end="", flush=True)

            try:
                result, validation, attempts = planner.generate_slot(
                    request, recipes_requested=RECIPES_PER_PILOT_SLOT,
                    next_recipe_index=index, pending_slot_ids=pending)
            except QuotaBlocked as blocked:
                write_json(REPORTS / config.quota.quota_block_filename, {
                    **blocked.state.as_dict(), "phase": "c2_pilot",
                    "milestone": "C2", "generator_code_commit": commit,
                    "frozen_contract": context.as_frozen_record()})
                blocked_records = provider.records[before:]
                for offset, record in enumerate(blocked_records, start=1):
                    record["sequence"] = sum(1 for item in records
                                             if item["slot_id"] == slot_id) + offset
                    # A blocked attempt is still an attempt. Breaking out before
                    # recording it would leave an archived call with no
                    # provenance, which is exactly what "every attempt has
                    # provenance" forbids.
                    provenance.append(build_provenance(
                        request=request, result=_result_view(record),
                        config_summary=context.config_summary(),
                        prompt_provenance=context.template.as_provenance(),
                        schema_identity=context.schema_identity_12x32,
                        validation_result="provider_error", validation_errors=[],
                        retry_count=offset - 1, generator_code_commit=commit,
                        request_schedule_id=f"C2_PILOT:{slot_id}:pass{pass_number}",
                        raw_response_path=None,
                        billing_tier=config.quota.billing_tier).as_dict())
                records.extend(blocked_records)
                status = "BLOCKED_QUOTA"
                stop = True
                print("  QUOTA EXHAUSTED - checkpointed and stopped")
                break

            new_records = provider.records[before:]
            for offset, record in enumerate(new_records, start=1):
                # `attempt` is reset by the frozen policy after a transport
                # failure, so it does not identify a call. `sequence` does.
                record["sequence"] = sum(1 for item in records
                                         if item["slot_id"] == slot_id) + offset
            records.extend(new_records)

            issues = ([issue for candidate in validation.candidates for issue in candidate.issues]
                      + validation.response_issues) if validation is not None else []
            accepted = (validation.accepted[0]
                        if (validation is not None and validation.accepted) else None)
            responses = [record for record in new_records if record["raw_text"] is not None]

            for offset, record in enumerate(new_records, start=1):
                final = offset == len(new_records)
                provenance.append(build_provenance(
                    request=request, result=_result_view(record),
                    config_summary=context.config_summary(),
                    prompt_provenance=context.template.as_provenance(),
                    schema_identity=context.schema_identity_12x32,
                    validation_result=("accepted" if (final and accepted is not None)
                                       else "rejected" if record["raw_text"] is not None
                                       else "provider_error"),
                    validation_errors=issues if final else [],
                    parsed_recipe_sha256=[accepted.recipe_identity] if (final and accepted) else [],
                    retry_count=offset - 1, generator_code_commit=commit,
                    request_schedule_id=f"C2_PILOT:{slot_id}:pass{pass_number}",
                    raw_response_path=f"reports/c2/raw_responses/pilot/"
                                      f"{slot_id}__seq{record['sequence']:02d}.txt",
                    billing_tier=config.quota.billing_tier).as_dict())

            if result.error is not None and result.error.error_class in HARD_BLOCK:
                status = HARD_BLOCK[result.error.error_class]
                stop = True
                print(f"  {status}: {result.error}")
                break

            if accepted is None and not responses:
                # Not a semantic failure: the provider never answered. Recording
                # this as an exhausted slot would inflate the retry-exhaustion
                # rate with a rate-limit condition that says nothing about the
                # prompt. The slot stays pending and the pacing widens.
                interval = min(interval * PACING_GROWTH, PILOT_MAX_CALL_INTERVAL_SECONDS)
                print(f"  no response after {len(new_records)} call(s) "
                      f"({result.error.error_class.value if result.error else 'unknown'}); "
                      f"slot stays pending, pacing -> {interval:.0f}s")
                persist(context, slots, records, provenance, "IN_PROGRESS", commit)
                continue

            slot: dict = {
                "slot_id": slot_id,
                "slot_index": index,
                "provider_calls": len(new_records),
                "responses_received": len(responses),
                "semantic_attempts": len(responses),
                "accepted_on_attempt": (responses.index(new_records[-1]) + 1
                                        if accepted is not None and new_records[-1] in responses
                                        else len(responses) if accepted is not None else None),
                "final_status": "accepted" if accepted is not None else "exhausted",
                "recipe_id": accepted.recipe.recipe_id if accepted and accepted.recipe else None,
                "recipe_identity": accepted.recipe_identity if accepted else None,
                "content_identity": (RecipePlanner.content_identity(accepted.recipe)
                                     if accepted and accepted.recipe else None),
                "canonical_recipe": accepted.canonical_text if accepted else None,
                "final_error": result.error.as_dict() if result.error is not None else None,
                "final_issues": issues,
                "compiler_status": "not_attempted",
                "graph_hash": None,
            }

            if accepted is not None and accepted.recipe is not None:
                try:
                    graph = compile_accepted(accepted.recipe, context.ontology,
                                             bank_id=PILOT_BANK_ID)
                    slot["compiler_status"] = "compiled"
                    slot["graph_hash"] = graph.graph_hash
                    slot["operator_names"] = list(graph.operator_names())
                    slot["conditioning_dimension"] = graph.conditioning_dimension
                    slot["region_mask_policy"] = graph.region_mask_policy
                except CompileError as exc:
                    slot["compiler_status"] = "failed"
                    slot["compiler_error"] = str(exc)
                planner.mark_slot_completed(slot_id)

            slots.append(slot)
            done.add(slot_id)
            advanced = True
            print(f"  {slot['final_status']} after {slot['provider_calls']} call(s)"
                  f"  [{slot['compiler_status']}]")
            persist(context, slots, records, provenance, "IN_PROGRESS", commit)

        if not stop and not advanced and any(slot_id not in done for slot_id in SLOT_IDS):
            interval = min(interval * PACING_GROWTH, PILOT_MAX_CALL_INTERVAL_SECONDS)
            print(f"pass {pass_number} advanced no slot; cooling down "
                  f"{PASS_COOLDOWN_SECONDS:.0f}s and widening pacing to {interval:.0f}s")
            time.sleep(PASS_COOLDOWN_SECONDS)

    slots.sort(key=lambda item: item["slot_index"])
    if len(slots) < PILOT_SLOT_COUNT and status == "COMPLETE":
        status = "INCOMPLETE"
    persist(context, slots, records, provenance, status, commit)

    accepted_count = sum(1 for slot in slots if slot["final_status"] == "accepted")
    print(f"\npilot status: {status}   slots {len(slots)}/{PILOT_SLOT_COUNT}   "
          f"accepted {accepted_count}   provider calls {len(records)}")
    return 0 if status == "COMPLETE" else 3


def _result_view(record: dict):
    """Rebuild the ProviderGenerationResult a provenance record needs.

    The archive is the authority: provenance is derived from the archived bytes
    rather than from a live object, so a replayed record and a live one agree.
    """
    from prism_fas.llm.contracts import ProviderError, ProviderGenerationResult
    error = None
    if record["error"] is not None:
        error = ProviderError(ErrorClass(record["error"]["error_class"]),
                              record["error"]["message"],
                              status_code=record["error"].get("status_code"),
                              retry_after_seconds=record["error"].get("retry_after_seconds"))
    return ProviderGenerationResult(
        slot_id=record["slot_id"], attempt=record["attempt"], provider=record["provider"],
        model_id=record["model_id"], raw_text=record["raw_text"], parsed=None,
        finish_reason=record["finish_reason"], latency_seconds=record["latency_seconds"],
        usage=dict(record["usage"]), provider_request_id=record["provider_request_id"],
        model_version=record["model_version"], provider_seed=record["provider_seed"],
        error=error, sdk_version=record["sdk_version"], api_surface=record["api_surface"])


if __name__ == "__main__":
    sys.exit(main())
