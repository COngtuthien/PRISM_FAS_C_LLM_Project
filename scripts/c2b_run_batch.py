"""C2B: run exactly ONE logical 32-recipe Gemini batch.

    python scripts/c2b_run_batch.py

One logical batch, `C2B_BATCH_000`, asking for 32 recipe objects in a single
structured response. Transport failures before a completed semantic response are
retries of the SAME logical batch under the frozen bounded transport policy; they
are not a second experiment.

The rule that matters most here is the anti-cherry-picking one: once a completed
semantic response arrives, this script will not issue another batch, and refuses
to run again at all. A disappointing coverage result is a result. Re-rolling
until the numbers look good would make the coverage evidence meaningless, so the
capability simply does not exist in this script.

Semantic retry is deliberately NOT used. `RecipePlanner.generate_slot` would ask
the provider for a fresh batch when validation fails, which is exactly what C2B
must not do. The retry loop below therefore handles transport classes only.
"""
from __future__ import annotations

import sys
import time

from c2b_common import (BATCH_SIZE, C2B_BANK_ID, LOGICAL_BATCH_ID, REPORTS, BatchContext,
                        RecordingProvider, api_key_present, git, read_json, replay_fields,
                        utc_now, write_json, write_raw_response_files)

from prism_fas.llm.contracts import ErrorClass
from prism_fas.llm.pipeline import RecipePlanner, compile_accepted
from prism_fas.llm.provenance import build_provenance, identity_chain
from prism_fas.llm.providers.gemini import GeminiRecipeProvider
from prism_fas.recipes.compile import CompileError

STATE_PATH = REPORTS / "C2B_BATCH_STATE.json"
ARCHIVE_PATH = REPORTS / "C2B_RAW_ARCHIVE.json"
PROVENANCE_PATH = REPORTS / "C2B_PROVENANCE.json"

#: Classes that stop C2B with a named block status rather than a retry.
HARD_BLOCK = {
    ErrorClass.AUTH: "BLOCKED_PROVIDER",
    ErrorClass.MODEL_UNAVAILABLE: "BLOCKED_PROVIDER",
    ErrorClass.UNSUPPORTED_CONFIG: "BLOCKED_PROVIDER",
    ErrorClass.CONTRACT_VIOLATION: "BLOCKED_PROVIDER",
    ErrorClass.LOCAL_ERROR: "BLOCKED_PROVIDER",
    ErrorClass.FORBIDDEN_REQUEST: "BLOCKED_PROVIDER",
    ErrorClass.QUOTA_EXHAUSTED: "BLOCKED_QUOTA",
}

RETRYABLE = {ErrorClass.TRANSPORT, ErrorClass.SERVER_ERROR, ErrorClass.RATE_LIMIT}


def already_completed() -> bool:
    """True once a completed semantic response exists on disk."""
    if not STATE_PATH.exists():
        return False
    state = read_json(STATE_PATH)
    return bool(state.get("semantic_response_received"))


def main() -> int:
    context = BatchContext()
    config = context.config
    commit = git("rev-parse", "HEAD")

    if already_completed():
        print("C2B_BATCH_000 already produced a completed semantic response.\n"
              "C2B allows exactly one logical batch, so this script will not run again.\n"
              "Re-running after seeing the coverage result would be cherry-picking.\n"
              "Delete reports/c2b/ only if you intend to discard the recorded experiment.")
        return 0

    if not api_key_present(config):
        print("BLOCKED_PROVIDER: no credential in the environment")
        return 2

    # A previous run that never received a semantic response still made real
    # provider calls. Those attempts are evidence and are carried forward rather
    # than overwritten, so the archive holds every call this batch ever cost.
    prior_records: list[dict] = []
    if ARCHIVE_PATH.exists():
        prior_records = list(read_json(ARCHIVE_PATH).get("records", []))
        if prior_records:
            print(f"carrying forward {len(prior_records)} archived attempt(s) from an earlier "
                  "run that received no semantic response")

    provider = RecordingProvider(GeminiRecipeProvider(config), phase="c2b_batch")
    planner = RecipePlanner(provider=provider, config=config, ontology=context.ontology)
    request = context.request(LOGICAL_BATCH_ID)

    retry = config.retry
    transport_attempts = 0
    status = "COMPLETE"
    result = None
    rate_limit_events: list[dict] = []

    print(f"[{utc_now()}] {LOGICAL_BATCH_ID}: requesting {BATCH_SIZE} recipe objects "
          f"in one structured response")

    while True:
        attempt = len(provider.records) + 1
        result = provider.generate(request, attempt=attempt)
        record = provider.records[-1]

        if result.error is None:
            print(f"  response received: {len(result.raw_text or '')} chars "
                  f"in {record['latency_seconds']}s")
            break

        error = result.error
        print(f"  attempt {attempt} failed: {error.error_class.value}")
        if error.error_class is ErrorClass.RATE_LIMIT:
            rate_limit_events.append({"attempt": attempt,
                                      "retry_after_seconds": error.retry_after_seconds,
                                      "recorded_at_utc": record["recorded_at_utc"]})

        if error.error_class in RETRYABLE:
            transport_attempts += 1
            if transport_attempts >= retry.transport_max_attempts:
                status = "BLOCKED_PROVIDER"
                print(f"  transport budget exhausted after {transport_attempts} attempts")
                break
            delay = (error.retry_after_seconds if error.retry_after_seconds is not None
                     else retry.backoff_initial_seconds
                     * (retry.backoff_multiplier ** max(0, transport_attempts - 1)))
            delay = float(min(delay, retry.backoff_max_seconds)
                          if error.retry_after_seconds is None else delay)
            print(f"  transport retry {transport_attempts}/{retry.transport_max_attempts} "
                  f"in {delay:.1f}s (same logical batch)")
            time.sleep(delay)
            continue

        status = HARD_BLOCK.get(error.error_class, "BLOCKED_PROVIDER")
        if error.error_class is ErrorClass.QUOTA_EXHAUSTED:
            write_json(REPORTS / config.quota.quota_block_filename, {
                "blocked": True, "reason": str(error), "phase": "c2b_batch",
                "logical_batch_id": LOGICAL_BATCH_ID, "blocked_at_utc": utc_now(),
                "completed_slot_ids": [], "pending_slot_ids": [LOGICAL_BATCH_ID],
                "billing_tier": config.quota.billing_tier, "auto_enable_paid": False,
                "user_decision_required": "wait for the quota reset, or explicitly enable a "
                                          "Paid Tier. Code never enables billing."})
        print(f"  {status}: {error}")
        break

    # --- validate every returned object independently -----------------------
    validation = None
    per_recipe: list[dict] = []
    accepted_recipes = []
    if result is not None and result.raw_text is not None:
        validation = planner.validate_response(result.raw_text, slot_id=LOGICAL_BATCH_ID,
                                               recipes_requested=BATCH_SIZE,
                                               next_recipe_index=0)
        for candidate in validation.candidates:
            entry: dict = {
                "batch_index": candidate.index,
                "status": candidate.outcome.value,
                "recipe_id": candidate.recipe.recipe_id if candidate.recipe else None,
                "canonical_identity": candidate.recipe_identity,
                "canonical_recipe": candidate.canonical_text,
                "validation_failures": candidate.issues,
                "compiler_status": "not_attempted",
                "graph_hash": None,
            }
            if candidate.accepted and candidate.recipe is not None:
                try:
                    graph = compile_accepted(candidate.recipe, context.ontology,
                                             bank_id=C2B_BANK_ID)
                    entry["compiler_status"] = "compiled"
                    entry["graph_hash"] = graph.graph_hash
                    entry["operator_names"] = list(graph.operator_names())
                    entry["conditioning_dimension"] = graph.conditioning_dimension
                    entry["region_mask_policy"] = graph.region_mask_policy
                except CompileError as exc:
                    entry["compiler_status"] = "failed"
                    entry["compiler_error"] = str(exc)
                accepted_recipes.append(candidate.recipe)
            per_recipe.append(entry)

    # --- provenance for every attempt ---------------------------------------
    provenance: list[dict] = []
    for record in provider.records:
        final = record is provider.records[-1]
        provenance.append(build_provenance(
            request=request, result=_result_view(record),
            config_summary=context.config_summary(),
            prompt_provenance=context.template.as_provenance(),
            schema_identity=context.batch_envelope_schema_identity,
            validation_result=("accepted" if (final and validation is not None
                                              and validation.all_accepted)
                               else "partially_accepted" if (final and validation is not None)
                               else "provider_error"),
            validation_errors=([issue for candidate in validation.candidates
                                for issue in candidate.issues] + validation.response_issues)
            if (final and validation is not None) else [],
            parsed_recipe_sha256=[entry["canonical_identity"] for entry in per_recipe
                                  if entry["canonical_identity"]] if final else [],
            retry_count=record["sequence"] - 1, generator_code_commit=commit,
            request_schedule_id=f"C2B:{LOGICAL_BATCH_ID}",
            raw_response_path=(f"reports/c2b/raw_responses/{LOGICAL_BATCH_ID}"
                               f"__seq{record['sequence']:02d}.json")
            if record["raw_text"] else None,
            billing_tier=config.quota.billing_tier).as_dict())

    all_records = prior_records + provider.records
    for offset, record in enumerate(all_records, start=1):
        record["sequence"] = offset
    archived = [record for record in all_records if record["raw_text"] is not None]
    if archived:
        write_raw_response_files(archived)

    write_json(ARCHIVE_PATH, {
        "schema_version": "c2b-raw-archive-v1",
        "logical_batch_id": LOGICAL_BATCH_ID,
        "note": "verbatim provider output for every attempt of the one logical C2B batch, "
                "including attempts from an earlier run that received no semantic response; the "
                "same bytes are on disk under reports/c2b/raw_responses/ (git-ignored by "
                "repository policy)",
        "disposable": True,
        "enters_c3_or_final_bank": False,
        "record_count": len(all_records),
        "records": all_records,
        "replay_records": [replay_fields(record) for record in archived]})

    write_json(PROVENANCE_PATH, {
        "schema_version": "c2b-provenance-v1",
        "note": "one immutable record per provider attempt, successful or not",
        "record_count": len(provenance),
        "records": provenance})

    chain = identity_chain(
        ontology_identity=context.ontology.sha256,
        schema_identity=context.batch_envelope_schema_identity,
        prompt_template_identity=context.template.identity(),
        provider_config_identity=context.provider_config_identity,
        generation_request_identity=request.request_sha256,
        raw_response_identity=result.raw_response_sha256 if result is not None else None,
        canonical_recipe_identity=None)

    accepted = sum(1 for entry in per_recipe if entry["status"] == "accepted")
    write_json(STATE_PATH, {
        "schema_version": "c2b-batch-state-v1",
        "milestone": "C2B",
        "generated_at_utc": utc_now(),
        "generator_code_commit": commit,
        "logical_batch_id": LOGICAL_BATCH_ID,
        "logical_batches_executed": 1,
        "second_batch_issued": False,
        "status": status,
        "semantic_response_received": result is not None and result.raw_text is not None,
        "requested_objects": BATCH_SIZE,
        "returned_objects": len(per_recipe),
        "accepted_objects": accepted,
        "provider_attempts": len(provider.records),
        "transport_retries": transport_attempts,
        "rate_limit_events": rate_limit_events,
        "response_issues": validation.response_issues if validation is not None else [],
        "batch_contract": context.as_contract_record(),
        "coverage_quotas": context.quotas.as_dict(),
        "identity_chain": chain,
        "recipes": per_recipe,
        "disposable": True,
        "enters_c3": False,
        "enters_final_bank": False,
    })

    print(f"\nC2B batch status: {status}   returned {len(per_recipe)}/{BATCH_SIZE}   "
          f"accepted {accepted}   attempts {len(provider.records)}")
    return 0 if status == "COMPLETE" else 3


def _result_view(record: dict):
    """The ProviderGenerationResult a provenance record needs, from the archive."""
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
