"""C2 live provider smoke: at most two disposable Gemini calls.

    python scripts/c2_live_smoke.py

These calls are marked C2_SMOKE_ONLY. They are NOT pilot slots, NOT C3 candidate
slots and NOT part of the final 256-recipe bank; nothing they produce may be
reused downstream.

The point is to exercise the WHOLE live path once, end to end, before spending a
32-slot budget on it:

    Gemini -> Interactions API -> structured JSON response -> raw archive
    -> JSON parsing -> schema validation -> ontology validation -> range checks
    -> compatibility checks -> canonicalization -> recipe identity
    -> inherited compiler -> operator graph / mask policy / 41-D conditioning

Failure policy (C2 instruction section 4) is enforced here, not improvised:
AUTH stops with BLOCKED_AUTH, an unavailable model stops with BLOCKED_MODEL and
no substitute is tried, quota exhaustion writes the block artifact and stops with
BLOCKED_QUOTA, and a transient 429 uses the frozen bounded backoff.
"""
from __future__ import annotations

import sys

from c2_pilot_common import (REPORTS, FrozenContext, RecordingProvider, api_key_present,
                             git, read_json, replay_fields, utc_now, write_json,
                             write_raw_response_files)

from prism_fas.llm.contracts import ErrorClass
from prism_fas.llm.pipeline import RecipePlanner, compile_accepted
from prism_fas.llm.provenance import build_provenance, identity_chain
from prism_fas.llm.providers.gemini import GeminiRecipeProvider

SMOKE_MARKER = "C2_SMOKE_ONLY"
MAX_SMOKE_CALLS = 2
SMOKE_BANK_ID = "c2-smoke-disposable"

#: Non-retryable classes that must stop C2 with a named block status.
BLOCKING = {
    ErrorClass.AUTH: "BLOCKED_AUTH",
    ErrorClass.MODEL_UNAVAILABLE: "BLOCKED_MODEL",
    ErrorClass.QUOTA_EXHAUSTED: "BLOCKED_QUOTA",
    ErrorClass.UNSUPPORTED_CONFIG: "BLOCKED_REQUEST_SHAPE",
}


def stage_report(name: str, ok: bool, detail: str) -> dict:
    return {"stage": name, "ok": bool(ok), "detail": detail}


def load_prior_smoke() -> dict:
    """Earlier smoke evidence, if this is a resume.

    A completed smoke call is historical evidence: it is never re-made and never
    overwritten, and its raw response stays in the archive exactly as recorded.
    """
    empty = {"records": [], "calls": [], "provenance": [], "stages": [], "accepted": None,
             "result": None}
    archive_path = REPORTS / "C2_SMOKE_RAW_ARCHIVE.json"
    audit_path = REPORTS / "C2_LIVE_SMOKE_AUDIT.json"
    if not archive_path.exists() or not audit_path.exists():
        return empty
    archive = read_json(archive_path)
    audit = read_json(audit_path)
    if not archive.get("records"):
        return empty
    return {"records": list(archive["records"]),
            "calls": list(audit.get("calls", [])),
            "provenance": list(audit.get("provenance_records", [])),
            "stages": list(audit.get("pipeline_stages", [])),
            "accepted": audit.get("accepted_smoke_recipe"),
            "result": audit.get("result")}


def main() -> int:
    context = FrozenContext()
    config = context.config
    commit = git("rev-parse", "HEAD")

    if not api_key_present(config):
        write_json(REPORTS / "C2_LIVE_SMOKE_AUDIT.json", {
            "schema_version": "c2-live-smoke-audit-v1", "milestone": "C2",
            "marker": SMOKE_MARKER, "result": "BLOCKED_AUTH",
            "reason": f"{config.api_key_env} is not present in the process environment",
            "live_calls": 0, "generated_at_utc": utc_now()})
        print("BLOCKED_AUTH: no credential in the environment")
        return 2

    provider = RecordingProvider(GeminiRecipeProvider(config), phase="smoke")
    planner = RecipePlanner(provider=provider, config=config, ontology=context.ontology)

    # Resume: earlier smoke calls are historical evidence and are never redone or
    # overwritten. Only the unspent part of the <=2 call budget is used.
    prior = load_prior_smoke()
    spent = len({record["slot_id"] for record in prior["records"]})
    calls: list[dict] = list(prior["calls"])
    provenance: list[dict] = list(prior["provenance"])
    stages: list[dict] = list(prior["stages"])
    accepted_summary: dict | None = prior["accepted"]
    result_status = prior["result"] or "SMOKE_NO_VALID_CANDIDATE"
    if spent >= MAX_SMOKE_CALLS:
        print(f"smoke budget already spent ({spent}/{MAX_SMOKE_CALLS}); nothing to do")
        return 0 if result_status == "SMOKE_PASS" else 3

    for index in range(spent, MAX_SMOKE_CALLS):
        slot_id = f"smoke_{index:03d}"
        request = context.request(slot_id, metadata={"marker": SMOKE_MARKER,
                                                     "disposable": True,
                                                     "counts_towards_pilot": False})
        print(f"[{utc_now()}] {SMOKE_MARKER} call {index + 1}/{MAX_SMOKE_CALLS}: {slot_id}")

        # One provider call per smoke slot. The provider classifies every failure
        # into the result rather than raising, so there is nothing to catch here.
        result = provider.generate(request, attempt=1)
        record = provider.records[-1]
        calls.append({
            "slot_id": slot_id, "attempt": 1, "marker": SMOKE_MARKER,
            "request_identity": request.request_sha256,
            "raw_response_sha256": record["raw_response_sha256"],
            "latency_seconds": record["latency_seconds"],
            "finish_reason": record["finish_reason"],
            "model_version": record["model_version"],
            "usage": record["usage"],
            "error": record["error"],
        })

        if result.error is not None:
            error_class = result.error.error_class
            stages.append(stage_report("provider_call", False,
                                       f"{error_class.value}: {result.error}"))
            if error_class is ErrorClass.QUOTA_EXHAUSTED:
                write_json(REPORTS / config.quota.quota_block_filename, {
                    "blocked": True, "reason": str(result.error), "phase": "smoke",
                    "marker": SMOKE_MARKER, "blocked_at_utc": utc_now(),
                    "completed_slot_ids": [], "pending_slot_ids": [slot_id],
                    "billing_tier": config.quota.billing_tier, "auto_enable_paid": False,
                    "user_decision_required": "wait for the quota reset, or explicitly enable a "
                                              "Paid Tier. Code never enables billing."})
            if error_class in BLOCKING:
                result_status = BLOCKING[error_class]
                break
            # transport / server / rate-limit: the second smoke call is the retry
            result_status = f"SMOKE_PROVIDER_ERROR_{error_class.value.upper()}"
            continue

        stages.append(stage_report("provider_call", True,
                                   f"interaction returned {len(result.raw_text or '')} chars"))
        stages.append(stage_report("raw_response_archive", True,
                                   f"sha256 {record['raw_response_sha256']}"))

        validation = planner.validate_response(result.raw_text or "", slot_id=slot_id,
                                               recipes_requested=request.recipes_requested)
        stages.append(stage_report("json_parsing", not any(
            issue["stage"] == "json_parsing" for issue in validation.response_issues),
            "response parsed as JSON" if not any(
                issue["stage"] == "json_parsing" for issue in validation.response_issues)
            else "response was not valid JSON"))
        stages.append(stage_report("envelope_schema", not validation.response_issues,
                                   f"{len(validation.response_issues)} envelope issues"))

        provenance.append(build_provenance(
            request=request, result=result, config_summary=context.config_summary(),
            prompt_provenance=context.template.as_provenance(),
            schema_identity=context.schema_identity_12x32,
            validation_result="accepted" if validation.all_accepted else "rejected",
            validation_errors=[issue for candidate in validation.candidates
                               for issue in candidate.issues] + validation.response_issues,
            parsed_recipe_sha256=[candidate.recipe_identity for candidate in validation.candidates
                                  if candidate.recipe_identity],
            retry_count=0, generator_code_commit=commit,
            request_schedule_id=f"{SMOKE_MARKER}:{slot_id}",
            raw_response_path=f"reports/c2/raw_responses/smoke/{slot_id}__attempt01.txt",
            billing_tier=config.quota.billing_tier).as_dict())

        if not validation.all_accepted:
            stages.append(stage_report("candidate_validation", False,
                                       f"{len(validation.accepted)}/{len(validation.candidates)} accepted"))
            result_status = "SMOKE_INVALID_CANDIDATE"
            continue

        candidate = validation.accepted[0]
        recipe = candidate.recipe
        assert recipe is not None
        stages.append(stage_report("typed_recipe_schema", True, "strict v1.1 schema satisfied"))
        stages.append(stage_report("ontology_membership", True,
                                   "every enum value is canonical under the frozen ontology"))
        stages.append(stage_report("range_checks", True, "every numeric value inside its safe band"))
        stages.append(stage_report("compatibility_checks", True,
                                   "medium/artifact and geometry/region compatible"))
        stages.append(stage_report("canonicalization", candidate.canonical_text is not None,
                                   "canonical JSON produced"))
        stages.append(stage_report("recipe_identity", bool(candidate.recipe_identity),
                                   f"recipe identity {candidate.recipe_identity}"))

        graph = compile_accepted(recipe, context.ontology, bank_id=SMOKE_BANK_ID)
        stages.append(stage_report("inherited_compiler", True,
                                   f"compiled to {len(graph.nodes)} operator nodes"))
        stages.append(stage_report("operator_graph", bool(graph.graph_hash),
                                   f"graph hash {graph.graph_hash}"))
        stages.append(stage_report("region_mask_policy",
                                   graph.region_mask_policy.get("policy") is not None,
                                   f"policy {graph.region_mask_policy.get('policy')!r}"))
        stages.append(stage_report("conditioning_41d", graph.conditioning_dimension == 41,
                                   f"conditioning dimension {graph.conditioning_dimension}"))

        accepted_summary = {
            "slot_id": slot_id,
            "recipe_identity": candidate.recipe_identity,
            "canonical_recipe": candidate.canonical_text,
            "medium": recipe.medium.family,
            "geometry": recipe.geometry.shape,
            "regions": list(recipe.regions),
            "artifacts": [{"name": spec.name, "strength": spec.strength} for spec in recipe.artifacts],
            "illumination": recipe.capture.illumination,
            "graph_hash": graph.graph_hash,
            "operator_names": list(graph.operator_names()),
            "conditioning_dimension": graph.conditioning_dimension,
            "conditioning_version": graph.conditioning_version,
            "disposable": True,
            "enters_pilot": False,
            "enters_c3": False,
            "enters_final_bank": False,
        }
        result_status = "SMOKE_PASS"
        break

    # A pass already recorded by an earlier smoke call is not undone by a later
    # one; the per-call records below carry what each call actually did.
    if accepted_summary is not None and result_status not in BLOCKING.values():
        result_status = "SMOKE_PASS"

    chain = identity_chain(
        ontology_identity=context.ontology.sha256,
        schema_identity=context.schema_identity_12x32,
        prompt_template_identity=context.template.identity(),
        provider_config_identity=context.provider_config_identity,
        generation_request_identity=calls[-1]["request_identity"] if calls else None,
        raw_response_identity=calls[-1]["raw_response_sha256"] if calls else None,
        canonical_recipe_identity=accepted_summary["recipe_identity"] if accepted_summary else None)

    all_records = prior["records"] + provider.records
    archived = [record for record in all_records if record["raw_text"] is not None]
    if archived:
        write_raw_response_files(archived, "smoke")
    write_json(REPORTS / "C2_SMOKE_RAW_ARCHIVE.json", {
        "schema_version": "c2-smoke-raw-archive-v1", "marker": SMOKE_MARKER,
        "note": "verbatim provider output for every smoke attempt; the same bytes are on disk "
                "under reports/c2/raw_responses/smoke/ (git-ignored by repository policy)",
        "records": all_records,
        "replay_records": [replay_fields(record) for record in archived]})

    write_json(REPORTS / "C2_LIVE_SMOKE_AUDIT.json", {
        "schema_version": "c2-live-smoke-audit-v1",
        "milestone": "C2",
        "marker": SMOKE_MARKER,
        "generated_at_utc": utc_now(),
        "generator_code_commit": commit,
        "result": result_status,
        "disposable": True,
        "scope_exclusion": {
            "counts_towards_32_pilot_slots": False,
            "counts_towards_c3_384_slots": False,
            "enters_final_256_recipe_bank": False,
            "enters_detector_training": False,
        },
        "budget": {"max_calls": MAX_SMOKE_CALLS, "calls_made": len(calls)},
        "frozen_contract": context.as_frozen_record(),
        "calls": calls,
        "pipeline_stages": stages,
        "accepted_smoke_recipe": accepted_summary,
        "identity_chain": chain,
        "provenance_records": provenance,
        "credential": {"env_var": config.api_key_env, "present": True,
                       "value_read_into_artifact": False},
    })

    print(f"\nsmoke result: {result_status}  ({len(calls)} live call(s))")
    return 0 if result_status == "SMOKE_PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
