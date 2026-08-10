"""Build every C2C audit artifact from the archived batch, offline.

    python scripts/c2c_build_reports.py

Nothing here calls a provider. The archived response is replayed through a fresh
pipeline by the replay provider, which holds no client and no credential, and
every number is derived from what that replay produces.

The acceptance rule is not negotiable and is not weakened after the fact: if any
ACCEPTED scientific recipe fails to compile, C2C fails.
"""
from __future__ import annotations

import json
import sys
from typing import Any

from c2c_common import (BATCH_SIZE, C2B_COVERAGE, C2C_BANK_ID, C3_FINAL_BANK,
                        C3_MIN_UNIQUE_POOL, C3_RAW_SLOTS, C3_REQUESTS, DOCS, LOGICAL_BATCH_ID,
                        REPORTS, RouteContext, git, read_json, utc_now, write_json)

from prism_fas.llm.coverage_quotas import evaluate
from prism_fas.llm.pilot_audit import axis_pair_table, coverage_audit, duplicate_audit
from prism_fas.llm.pipeline import RecipePlanner, compile_accepted
from prism_fas.llm.providers.replay import ReplayArchive, ReplayRecipeProvider
from prism_fas.llm.route_policy import audit as route_audit
from prism_fas.recipes.compile import CompileError

#: Fixed before the batch returned. Not weakened afterwards.
CRITERIA = {
    "returned_objects_exact": BATCH_SIZE,
    "min_semantic_validity": 0.90,
    "max_response_issues": 0,
    "max_compiler_failures_among_accepted": 0,
    "max_duplicate_rate": 0.10,
    "max_axis_share_percent": 60.0,
    "axes_requiring_full_presence": ["media", "geometry", "illumination", "artifacts", "regions"],
}

PAIR_TABLES = (("artifacts", "media"), ("artifacts", "geometry"), ("artifacts", "regions"),
               ("media", "geometry"), ("media", "illumination"), ("geometry", "illumination"))

REPLAY_KEYS = ("slot_id", "attempt", "raw_text", "provider", "model_id", "model_version",
               "finish_reason", "usage", "provider_request_id", "provider_seed", "sdk_version",
               "api_surface", "request_sha256")


def replay_batch(context: RouteContext, archive_records: list[dict]) -> dict[str, Any]:
    served = [record for record in archive_records if record["raw_text"] is not None]
    if not served:
        return {"rows": [], "recipes": [], "identities": [], "response_issues": []}
    archive = ReplayArchive.from_records([{key: record[key] for key in REPLAY_KEYS}
                                          for record in served])
    provider = ReplayRecipeProvider(archive, strict=False)
    planner = RecipePlanner(provider=provider, config=context.config, ontology=context.ontology,
                            sleep=lambda _s: None, route_policy=context.route_policy)
    request = context.request(LOGICAL_BATCH_ID)
    result = provider.generate(request, attempt=served[-1]["attempt"])
    validation = planner.validate_response(result.raw_text or "", slot_id=LOGICAL_BATCH_ID,
                                           recipes_requested=BATCH_SIZE, next_recipe_index=0)

    rows: list[dict[str, Any]] = []
    recipes = []
    identities: list[str] = []
    for candidate in validation.candidates:
        row: dict[str, Any] = {
            "batch_index": candidate.index,
            "status": candidate.outcome.value,
            "recipe_id": candidate.recipe.recipe_id if candidate.recipe else None,
            "generator_route": (list(candidate.recipe.generator_route)
                                if candidate.recipe else None),
            "canonical_identity": candidate.recipe_identity,
            "canonical_recipe": candidate.canonical_text,
            "validation_failures": candidate.issues,
            "compiler_status": "not_attempted",
            "graph_hash": None,
        }
        if candidate.accepted and candidate.recipe is not None:
            try:
                graph = compile_accepted(candidate.recipe, context.ontology,
                                         bank_id=C2C_BANK_ID)
                row["compiler_status"] = "compiled"
                row["graph_hash"] = graph.graph_hash
                row["operator_names"] = list(graph.operator_names())
                row["conditioning_dimension"] = graph.conditioning_dimension
                row["region_mask_policy"] = graph.region_mask_policy
            except CompileError as exc:
                row["compiler_status"] = "failed"
                row["compiler_error"] = str(exc)
            recipes.append(candidate.recipe)
            identities.append(RecipePlanner.content_identity(candidate.recipe))
        rows.append(row)
    return {"rows": rows, "recipes": recipes, "identities": identities,
            "response_issues": validation.response_issues, "provider": provider}


def verify_replay(live_rows: list[dict], replay_rows: list[dict]) -> dict[str, Any]:
    mismatches = []
    for live, replayed in zip(live_rows, replay_rows):
        for field in ("batch_index", "status", "canonical_identity", "compiler_status",
                      "graph_hash", "generator_route"):
            if live.get(field) != replayed.get(field):
                mismatches.append({"batch_index": live.get("batch_index"), "field": field,
                                   "live": live.get(field), "replay": replayed.get(field)})
    return {"objects_compared": min(len(live_rows), len(replay_rows)),
            "count_matches": len(live_rows) == len(replay_rows),
            "mismatches": mismatches,
            "identical": not mismatches and len(live_rows) == len(replay_rows)}


def coverage_comparison(quota: dict) -> dict[str, Any]:
    """Did the minimal route fix damage the coverage C2B achieved?"""
    rows = []
    for axis, before in C2B_COVERAGE.items():
        entry = quota["axes"][axis]
        rows.append({
            "axis": axis,
            "total_categories": before["total"],
            "c2b_present": before["present"],
            "c2c_present": entry["categories_present"],
            "presence_delta": entry["categories_present"] - before["present"],
            "c2b_max_share_percent": before["max_share_percent"],
            "c2c_max_share_percent": entry["max_share_percent"],
            "share_delta": round(entry["max_share_percent"] - before["max_share_percent"], 4),
            "c2c_missing": entry["categories_missing"],
            "still_fully_covered": entry["categories_present"] == before["total"],
            "damaged": entry["categories_present"] < before["present"],
        })
    damaged = [row["axis"] for row in rows if row["damaged"]]
    return {
        "scope": "C2B batch versus C2C batch, both source-independent prompt-development "
                 "evidence under identical generic quotas. No dataset was consulted.",
        "axes": rows,
        "axes_still_fully_covered": sum(1 for row in rows if row["still_fully_covered"]),
        "axes_total": len(rows),
        "damaged_axes": damaged,
        "route_fix_damaged_coverage": bool(damaged),
        "verdict": "coverage preserved" if not damaged else f"coverage reduced on {damaged}",
    }


def classify_outcome(state: dict, replay: dict, quota: dict, duplicates: dict) -> dict[str, Any]:
    if state["status"] == "BLOCKED_QUOTA":
        return {"outcome": "BLOCKED_QUOTA", "structural": {}, "coverage": {}, "failed": []}
    if not state.get("semantic_response_received"):
        return {"outcome": "BLOCKED_PROVIDER", "structural": {}, "coverage": {}, "failed": []}

    rows = replay["rows"]
    accepted = [row for row in rows if row["status"] == "accepted"]
    validity = len(accepted) / len(rows) if rows else 0.0
    compiler_failures = sum(1 for row in accepted if row["compiler_status"] == "failed")

    structural = {
        "returned_exactly_32": len(rows) == CRITERIA["returned_objects_exact"],
        "no_response_level_issues": len(replay["response_issues"]) <= CRITERIA["max_response_issues"],
        "semantic_validity_at_least_threshold": validity >= CRITERIA["min_semantic_validity"],
        "zero_compiler_failures_among_accepted":
            compiler_failures <= CRITERIA["max_compiler_failures_among_accepted"],
        "duplicate_rate_within_threshold":
            duplicates["exact_duplicate_rate"] <= CRITERIA["max_duplicate_rate"],
    }
    max_share = max((entry["max_share_percent"] for entry in quota["axes"].values()), default=0.0)
    coverage_checks = {
        "all_axes_fully_represented": all(
            quota["axes"][axis]["categories_missing"] == []
            for axis in CRITERIA["axes_requiring_full_presence"]),
        "no_severe_mode_collapse": max_share <= CRITERIA["max_axis_share_percent"],
        "quota_required_bounds_satisfied": quota["required_pass"],
    }
    failed = ([name for name, ok in structural.items() if not ok]
              + [name for name, ok in coverage_checks.items() if not ok])
    outcome = "PASS" if not failed else "FAIL"
    return {"outcome": outcome, "criteria": CRITERIA, "structural": structural,
            "coverage": coverage_checks, "failed": failed,
            "semantic_validity": round(validity, 6),
            "compiler_failures_among_accepted": compiler_failures,
            "max_axis_share_percent": max_share,
            "criteria_weakened_after_seeing_results": False}


def freeze_candidate(context: RouteContext, verdict: dict, state: dict) -> dict[str, Any]:
    """The C3 generation contract, prepared but NOT frozen and NOT executed."""
    contract = context.as_contract_record()
    components = {
        "provider": contract["provider_name"],
        "model_id": contract["model_id"],
        "api_surface": contract["api_surface"],
        "sdk_package": contract["sdk_package"],
        "thinking_level": contract["thinking_level"],
        "response_mime_type": contract["response_mime_type"],
        "max_output_tokens": contract["max_output_tokens"],
        "system_prompt_identity": contract["system_prompt_identity"],
        "batch_generation_template_identity": contract["batch_generation_template_identity"],
        "coverage_quota_identity": contract["coverage_quota_identity"],
        "single_recipe_schema_identity": contract["single_recipe_schema_identity"],
        "batch_envelope_schema_identity": contract["batch_envelope_schema_identity"],
        "ontology_identity": contract["ontology_identity"],
        "route_policy_identity": contract["route_policy_identity"],
        "allow_ontology_aliases": contract["allow_ontology_aliases"],
        "provider_config_identity": contract["provider_config_identity"],
        "request_schedule": {"requests": C3_REQUESTS, "objects_per_request": BATCH_SIZE,
                             "raw_slots": C3_RAW_SLOTS,
                             "minimum_unique_pool": C3_MIN_UNIQUE_POOL,
                             "final_bank": C3_FINAL_BANK},
        "retry_policy": contract["retry_policy"],
    }
    import hashlib

    canonical = json.dumps(components, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    identity = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "schema_version": "c2c-c3-freeze-candidate-v1",
        "milestone": "C2C",
        "generated_at_utc": utc_now(),
        "status": "CANDIDATE - NOT FROZEN, NOT EXECUTED",
        "frozen_by_this_session": False,
        "requires_explicit_user_approval": True,
        "c2c_outcome": verdict["outcome"],
        "components": components,
        "canonical_text": canonical,
        "c3_generation_contract_identity": identity,
        "invalidation_rule": "changing ANY component above changes "
                             "c3_generation_contract_identity, and therefore invalidates any "
                             "C3 generation carried out under the previous value.",
        "c3_requests_executed": 0,
        "evidence": {
            "one_logical_batch": state["logical_batches_executed"] == 1,
            "returned_objects": state["returned_objects"],
            "accepted_objects": state["accepted_objects"],
            "compiler_failures_among_accepted": verdict.get(
                "compiler_failures_among_accepted"),
        },
    }


# ------------------------------------------------------------------- markdown
def _table(headers: list[str], rows: list[list[Any]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join([line, rule, body])


def describe_physically(recipe) -> str:
    medium_words = {"paper-like": "a printed paper surface",
                    "display-like": "an emissive display panel",
                    "plastic-like": "a moulded plastic surface",
                    "fabric-like": "a woven fabric surface",
                    "reflective-film-like": "a reflective film overlay"}
    shape_words = {"flat": "held flat", "curved": "curved", "partial-curved": "partly curved",
                   "flexible": "flexible and deformable", "rigid": "rigid",
                   "boundary-only": "present only at the face edge"}
    light_words = {"front": "frontal light", "left": "light from the left",
                   "right": "light from the right", "top": "light from above",
                   "bottom": "light from below", "mixed": "mixed lighting"}
    artifacts = ", ".join(f"{spec.name} at {round(float(spec.strength), 3)}"
                          for spec in recipe.artifacts)
    return (f"{medium_words.get(recipe.medium.family, recipe.medium.family)} "
            f"({shape_words.get(recipe.geometry.shape, recipe.geometry.shape)}, "
            f"transparency {recipe.medium.transparency}, roughness {recipe.medium.roughness}) "
            f"covering {', '.join(recipe.regions)} at coverage {recipe.geometry.coverage}, "
            f"producing {artifacts}, under "
            f"{light_words.get(recipe.capture.illumination, recipe.capture.illumination)} "
            f"at yaw {recipe.capture.yaw} deg, scale {recipe.capture.scale}, "
            f"compression q{recipe.capture.compression_q}.")


def build_report(context: RouteContext, state: dict, replay: dict, quota: dict, coverage: dict,
                 pairs: dict, duplicates: dict, comparison: dict, verdict: dict,
                 routes: dict, verification: dict, archive: dict, c2b_replay: dict) -> str:
    contract = context.as_contract_record()
    diff = context.prompt_diff()
    lines: list[str] = []
    add = lines.append

    add("# C2C - scientific route contract repair")
    add("")
    add("C2B produced 32 schema-valid recipes and then could not compile 10 of them, because")
    add("they declared `generator_route` without `physics`. The validator and the compiler")
    add("disagreed about what an acceptable recipe is. C2C resolves that in favour of the")
    add("synthesis design: `generator_route` is a frozen execution contract, and the only")
    add("accepted declaration is exactly `[\"physics\", \"gpat\"]`.")
    add("")
    add(f"**Outcome: {verdict['outcome']}**")
    add("")

    add("## The route policy")
    add("")
    policy = context.route_policy
    add(_table(["field", "value"], [
        ["version", policy.version],
        ["required generator_route", f"`{list(policy.allowed_scientific_generator_route)}`"],
        ["require exact order", str(policy.require_exact_order)],
        ["subset allowed", str(policy.allow_subset)],
        ["GPAT-only accepted class", str(policy.allow_gpat_only_class)],
        ["silent repair permitted", str(policy.silent_repair_permitted)],
        ["**route policy identity**", f"`{policy.route_policy_identity}`"],
    ]))
    add("")
    add(f"Canonical text: `{policy.canonical_text()}`")
    add("")
    add(f"> {policy.rationale}")
    add("")
    add("Enforcement sits in the validation pipeline, between the inherited validator's pass")
    add("and canonicalization. A route-invalid candidate is rejected as `rejected_route_policy`,")
    add("is never canonicalized, never registered in the duplicate registry, and is never")
    add("handed to the compiler. **Nothing is repaired**: a recipe declaring `[\"gpat\"]` is")
    add("recorded exactly as the provider wrote it.")
    add("")

    add("## Prompt amendment")
    add("")
    add(_table(["field", "value"], [
        ["reason", diff["reason"]],
        ["old prompt identity", f"`{diff['old_prompt_identity']}`"],
        ["new prompt identity", f"`{diff['new_prompt_identity']}`"],
        ["old system-prompt sha256", f"`{diff['old_system_prompt_sha256']}`"],
        ["new system-prompt sha256", f"`{diff['new_system_prompt_sha256']}`"],
        ["generation template changed", str(diff["generation_template_changed"])],
        ["coverage quotas changed", str(diff["coverage_quotas_changed"])],
        ["characters added", diff["chars_added"]],
        ["lines added / removed", f"{diff['lines_added']} / {diff['lines_removed']}"],
    ]))
    add("")
    add(f"> {diff['classification']}")
    add("")
    add("Exact byte-level diff:")
    add("")
    add("```diff")
    add(diff["unified_diff"].rstrip())
    add("```")
    add("")

    add("## C2B replay under the new policy (offline, zero network)")
    add("")
    without = c2b_replay["without_route_policy_as_c2b_ran_it"]
    with_policy = c2b_replay["with_route_policy"]
    add(_table(["measure", "as C2B ran it", "under the C2C route policy"], [
        ["accepted", without["accepted"], with_policy["accepted"]],
        ["rejected", without["rejected"], with_policy["rejected"]],
        ["rejected by route policy", without["rejected_by_route_policy"],
         with_policy["rejected_by_route_policy"]],
        ["compiler attempted", without["compiler_attempted"], with_policy["compiler_attempted"]],
        ["compiler compiled", without["compiler_compiled"], with_policy["compiler_compiled"]],
        ["**compiler failed**", without["compiler_failed"], with_policy["compiler_failed"]],
    ]))
    add("")
    add(f"- no recipe was altered: **{c2b_replay['no_recipe_was_altered']}**")
    add(f"- silent repairs: **{c2b_replay['silent_repairs_performed']}**")
    add("- C2B artifacts were read only and were not modified.")
    add("")
    add("The observed split is reported as measured. It is **not** the 22/10 an initial reading")
    add("might expect: the frozen policy rejects physics-only as well as gpat-only, and C2B's")
    add("route distribution was 16 physics-only, 10 gpat-only and only 6 physics+gpat. So 6")
    add("recipes comply and 26 do not, and the compiler is never offered a recipe it cannot")
    add("build.")
    add("")

    add("## Live batch")
    add("")
    add(_table(["measure", "value"], [
        ["logical batches executed", state["logical_batches_executed"]],
        ["second batch issued", str(state["second_batch_issued"])],
        ["provider attempts", state["provider_attempts"]],
        ["transport retries", state["transport_retries"]],
        ["429 events", len(state["rate_limit_events"])],
        ["status", state["status"]],
        ["requested objects", state["requested_objects"]],
        ["returned objects", len(replay["rows"])],
        ["accepted", sum(1 for row in replay["rows"] if row["status"] == "accepted")],
        ["route-policy rejections",
         sum(1 for row in replay["rows"] if row["status"] == "rejected_route_policy")],
        ["other rejections",
         sum(1 for row in replay["rows"]
             if row["status"] not in ("accepted", "rejected_route_policy"))],
        ["duplicates", duplicates["exact_duplicate_groups"]],
        ["compiled", sum(1 for row in replay["rows"] if row["compiler_status"] == "compiled")],
        ["**compiler failures among accepted**",
         verdict.get("compiler_failures_among_accepted", 0)],
    ]))
    add("")
    served = [record for record in archive["records"] if record["raw_text"] is not None]
    if served:
        record = served[-1]
        add(_table(["provenance field", "value"], [
            ["latency (s)", record["latency_seconds"]],
            ["model revision", record["model_version"]],
            ["finish reason", record["finish_reason"]],
            ["raw response sha256", f"`{record['raw_response_sha256']}`"],
            ["request sha256", f"`{record['request_sha256']}`"],
            ["route policy identity in request", f"`{record.get('route_policy_identity')}`"],
            *[[key, value] for key, value in sorted((record["usage"] or {}).items())
              if isinstance(value, (int, float))],
        ]))
        add("")
    add(f"- offline replay identical to the live run: **{verification['identical']}**")
    add("")

    add("## Route compliance of the live batch")
    add("")
    add(_table(["generator_route", "recipes"],
               [[route, count] for route, count in routes["route_counts"].items()]))
    add("")
    add(f"- accepted recipes compliant with the contract: "
        f"**{routes['compliant_count']}/{routes['recipes_examined']}**")
    add(f"- silent repairs: **{routes['silent_repairs_performed']}** · "
        f"GPAT-only class created: **{routes['gpat_only_class_created']}**")
    add("")

    add("## Coverage and quota compliance")
    add("")
    for axis, entry in quota["axes"].items():
        add(f"### {axis} - {entry['categories_present']}/{entry['category_count']} present, "
            f"max share {entry['max_share_percent']}% "
            f"({'PASS' if entry['required_pass'] else 'FAIL'})")
        add("")
        add(_table(["category", "count", "% of recipes", "min", "preferred", "max", "required"],
                   [[name, cell["count"], cell["percent_of_recipes"],
                     cell["quota_minimum"] if cell["quota_minimum"] is not None else "-",
                     cell["quota_preferred_minimum"]
                     if cell["quota_preferred_minimum"] is not None else "-",
                     cell["quota_maximum"] if cell["quota_maximum"] is not None else "-",
                     "pass" if cell["required_pass"] else "**FAIL**"]
                    for name, cell in entry["categories"].items()]))
        add("")
    add(_table(["measure", "min", "max", "mean", "histogram"], [
        ["artifacts per recipe", coverage["artifacts_per_recipe"]["min"],
         coverage["artifacts_per_recipe"]["max"], coverage["artifacts_per_recipe"]["mean"],
         json.dumps(coverage["artifacts_per_recipe"]["histogram"])],
        ["regions per recipe", coverage["regions_per_recipe"]["min"],
         coverage["regions_per_recipe"]["max"], coverage["regions_per_recipe"]["mean"],
         json.dumps(coverage["regions_per_recipe"]["histogram"])],
    ]))
    add("")

    add("## Did the route fix damage coverage?")
    add("")
    add(_table(["axis", "categories", "C2B present", "C2C present", "C2B max share",
                "C2C max share", "still fully covered"],
               [[row["axis"], row["total_categories"], row["c2b_present"], row["c2c_present"],
                 f"{row['c2b_max_share_percent']}%", f"{row['c2c_max_share_percent']}%",
                 "yes" if row["still_fully_covered"] else "**no**"]
                for row in comparison["axes"]]))
    add("")
    add(f"**{comparison['verdict']}.** The quota values were not changed: the same C2B generic")
    add("quotas were used unmodified.")
    add("")

    add("## Co-occurrence")
    add("")
    for name, table in pairs.items():
        add(f"### {name}")
        add("")
        add(_table([table["row_axis"], *table["columns"], "total"],
                   [[row, *[table["cells"][row][column] for column in table["columns"]],
                     table["row_totals"][row]] for row in table["rows"]]))
        add("")
        dominant = table["dominant_cell"]
        add(f"occupied cells: {table['occupied_cells']}/{table['total_cells']}"
            + (f" · dominant: {dominant['row']} x {dominant['column']} "
               f"({dominant['count']}, {dominant['share_percent']}%)" if dominant else ""))
        add("")

    add("## All 32 returned objects")
    add("")
    by_id = {recipe.recipe_id: recipe for recipe in replay["recipes"]}
    for row in replay["rows"]:
        add(f"### index {row['batch_index']}")
        add("")
        recipe = by_id.get(row["recipe_id"])
        table_rows = [["status", row["status"]],
                      ["generator_route", json.dumps(row["generator_route"])],
                      ["compiler", row["compiler_status"]]]
        if recipe is not None:
            table_rows.extend([
                ["recipe id", row["recipe_id"]],
                ["artifact(s)", ", ".join(spec.name for spec in recipe.artifacts)],
                ["strengths", ", ".join(f"{spec.name}={round(float(spec.strength), 4)}"
                                        for spec in recipe.artifacts)
                 + f" (total {round(sum(float(s.strength) for s in recipe.artifacts), 4)})"],
                ["region(s)", ", ".join(recipe.regions)],
                ["medium", f"{recipe.medium.family} (transparency {recipe.medium.transparency}, "
                           f"roughness {recipe.medium.roughness})"],
                ["geometry", f"{recipe.geometry.shape} (rigidity {recipe.geometry.rigidity}, "
                             f"coverage {recipe.geometry.coverage})"],
                ["illumination", recipe.capture.illumination],
                ["canonical identity", f"`{row['canonical_identity']}`"],
                ["graph hash", f"`{row['graph_hash']}`"],
            ])
        else:
            table_rows.append(["failure", json.dumps(row["validation_failures"])[:400]])
        add(_table(["field", "value"], table_rows))
        add("")
        if recipe is not None:
            add(f"*Physical reading:* {describe_physically(recipe)}")
            add("")

    add("## Verdict detail")
    add("")
    add(_table(["check", "result"],
               [[name, "pass" if ok else "**FAIL**"]
                for name, ok in {**verdict["structural"], **verdict["coverage"]}.items()]))
    add("")
    add(f"Thresholds fixed before the batch returned: `{json.dumps(CRITERIA)}`")
    add("")
    add("These recipes are disposable validation evidence. None enters the C3 raw 384 slots,")
    add("the final LLM bank, an RND/DET bank, a synthetic bank or any training.")
    add("")
    return "\n".join(lines) + "\n"


def build_freeze_doc(context: RouteContext, candidate: dict, verdict: dict,
                     comparison: dict) -> str:
    lines: list[str] = []
    add = lines.append
    components = candidate["components"]
    add("# C2C - C3 freeze candidate")
    add("")
    add("**Nothing here is frozen.** This document prepares the exact identities for explicit")
    add("user approval. No C3 request was executed.")
    add("")
    add(f"C2C outcome: **{verdict['outcome']}**")
    add("")
    add("## C3 generation contract - candidate components")
    add("")
    add(_table(["component", "candidate value"], [
        ["provider", components["provider"]],
        ["model", f"`{components['model_id']}`"],
        ["SDK / API surface", f"{components['sdk_package']} / {components['api_surface']}"],
        ["thinking config", f"thinking_level = {components['thinking_level']}, "
                            f"max_output_tokens = {components['max_output_tokens']}, "
                            f"no sampling controls sent"],
        ["system prompt identity", f"`{components['system_prompt_identity']}`"],
        ["batch generation-template identity",
         f"`{components['batch_generation_template_identity']}`"],
        ["coverage quota identity", f"`{components['coverage_quota_identity']}`"],
        ["single-recipe schema identity", f"`{components['single_recipe_schema_identity']}`"],
        ["batch-envelope schema identity", f"`{components['batch_envelope_schema_identity']}`"],
        ["ontology identity", f"`{components['ontology_identity']}`"],
        ["**route policy identity**", f"`{components['route_policy_identity']}`"],
        ["alias policy", f"allow_ontology_aliases = {components['allow_ontology_aliases']}"],
        ["provider config identity", f"`{components['provider_config_identity']}`"],
        ["request schedule", f"{components['request_schedule']['requests']} x "
                             f"{components['request_schedule']['objects_per_request']} = "
                             f"{components['request_schedule']['raw_slots']} raw slots; "
                             f"min unique pool "
                             f"{components['request_schedule']['minimum_unique_pool']}; "
                             f"final bank {components['request_schedule']['final_bank']}"],
        ["retry policy", json.dumps(components["retry_policy"])],
    ]))
    add("")
    add("## Composite identity")
    add("")
    add(f"**C3_GENERATION_CONTRACT_IDENTITY**")
    add("")
    add(f"    {candidate['c3_generation_contract_identity']}")
    add("")
    add(f"> {candidate['invalidation_rule']}")
    add("")
    add("## Evidence supporting this candidate")
    add("")
    evidence = candidate["evidence"]
    add(_table(["measure", "value"], [
        ["exactly one logical batch", str(evidence["one_logical_batch"])],
        ["returned objects", evidence["returned_objects"]],
        ["accepted objects", evidence["accepted_objects"]],
        ["compiler failures among accepted", evidence["compiler_failures_among_accepted"]],
        ["coverage", comparison["verdict"]],
    ]))
    add("")
    add("## What still requires an explicit user decision")
    add("")
    add("1. Approve every identity in the table above as the frozen C3 generation contract.")
    add("2. Approve `C3_GENERATION_CONTRACT_IDENTITY` as the value the C3 BANK_LOCK binds to.")
    add("3. Confirm the Free-Tier quota position for 12 batch requests before C3 begins; the")
    add("   RPM/TPM/RPD limits must be read from AI Studio and were not invented here.")
    add("")
    add("**C3 was not started.**")
    add("")
    return "\n".join(lines) + "\n"


def build_correction_note(context: RouteContext) -> str:
    lines: list[str] = []
    add = lines.append
    add("# C2C - schema identity naming correction (prospective)")
    add("")
    add("This note corrects a NAMING error going forward. It edits no historical artifact:")
    add("the C1, C2 and C2B reports are left exactly as they were recorded, including the")
    add("original wording, so the record of how the mistake arose stays intact.")
    add("")
    add("## The error")
    add("")
    add("C1 recorded a value under a name that reads as a single-recipe identity:")
    add("")
    add("    llm_schema_identity_12x32 = 7afc3abd29178bb07e83538bdf1a9f15f1ce3c626ed3f5d467841f7038b777c4")
    add("")
    add("and later instructions referred to it as \"the single-recipe schema identity\". It is")
    add("not. It is the identity of the **32-object batch envelope**: the whole")
    add("`{\"recipes\": [...]}` object with `minItems = maxItems = 32`.")
    add("")
    add("## The correct values")
    add("")
    add(_table(["schema", "identity", "what it actually is"], [
        ["single-recipe ITEM schema",
         f"`{context.single_recipe_schema_identity}`",
         "one recipe object; the thing that carries recipe semantics"],
        ["C2 singleton envelope (n=1)",
         "`e9f66067c2de2deda5373a99dc6c92689c0ab2d2163b80adcde57af83df9bbd1`",
         "envelope C2 sent, accepted 42 times"],
        ["C1-recorded 32-object envelope",
         "`7afc3abd29178bb07e83538bdf1a9f15f1ce3c626ed3f5d467841f7038b777c4`",
         "envelope with the array bound; **rejected by the provider**, 400 INVALID_ARGUMENT"],
        ["C2B/C2C batch envelope (sent)",
         f"`{context.batch_envelope_schema_identity}`",
         "same envelope without the array length bound"],
    ]))
    add("")
    add("## Why the distinction matters")
    add("")
    add("The item schema is what recipe semantics depend on. The envelope only says how many")
    add("of those items a response carries. Conflating them made two separate facts look like")
    add("one:")
    add("")
    add("- the envelope had to change, because the provider rejects the bounded form;")
    add("- the item schema did **not** change, and is byte-identical inside the 1-object")
    add("  envelope C2 used and the envelope C2B and C2C send.")
    add("")
    add("C2C enforces the route contract at the validation layer rather than in the schema")
    add("precisely so the item identity stays put.")
    add("")
    add("## Naming used from C2C onward")
    add("")
    add(_table(["name", "meaning"], [
        ["`single_recipe_schema_identity`", "the item schema, one recipe object"],
        ["`batch_envelope_schema_identity`", "the envelope actually sent"],
        ["`bounded_batch_envelope_identity`", "the envelope with array bounds, kept only to "
                                              "name what the provider refuses"],
    ]))
    add("")
    add("No historical report was edited to hide the original naming.")
    add("")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------- main
def main() -> int:
    context = RouteContext()
    state = read_json(REPORTS / "C2C_BATCH_STATE.json")
    archive = read_json(REPORTS / "C2C_RAW_ARCHIVE.json")
    c2b_replay = read_json(REPORTS / "C2C_C2B_REPLAY_AUDIT.json")
    commit = git("rev-parse", "HEAD")

    replay = replay_batch(context, archive["records"])
    verification = verify_replay(state["recipes"], replay["rows"])
    recipes = replay["recipes"]

    quota = evaluate(context.quotas, recipes, context.ontology)
    coverage = coverage_audit(recipes, context.ontology)
    duplicates = duplicate_audit(replay["identities"], recipes)
    pairs = {f"{row}_x_{column}": axis_pair_table(recipes, context.ontology, row, column)
             for row, column in PAIR_TABLES}
    routes = route_audit(context.route_policy, recipes)
    comparison = coverage_comparison(quota)
    verdict = classify_outcome(state, replay, quota, duplicates)
    candidate = freeze_candidate(context, verdict, state)

    write_json(REPORTS / "C2C_ROUTE_POLICY_AUDIT.json", {
        "schema_version": "c2c-route-policy-audit-v1",
        "milestone": "C2C",
        "generated_at_utc": utc_now(),
        "generator_code_commit": commit,
        "route_policy": context.route_policy.as_dict(),
        "prompt_amendment": context.prompt_diff(),
        "enforcement_point": "prism_fas.llm.pipeline.RecipePlanner._validate_candidate, after "
                             "the inherited validator pass and before canonicalization, "
                             "duplicate detection and the compiler",
        "pipeline_stages": list(__import__("prism_fas.llm.pipeline", fromlist=["PIPELINE_STAGES"])
                                .PIPELINE_STAGES),
        "live_batch_route_audit": routes,
        "route_policy_rejections": [
            {"batch_index": row["batch_index"], "generator_route": row["generator_route"],
             "reasons": [issue["reason"] for issue in row["validation_failures"]],
             "compiler_ever_called": row["compiler_status"] != "not_attempted"}
            for row in replay["rows"] if row["status"] == "rejected_route_policy"],
        "silent_repairs_performed": 0,
        "gpat_only_accepted_class_created": False,
        "item_schema_identity_unchanged": True,
        "single_recipe_schema_identity": context.single_recipe_schema_identity,
    })

    write_json(REPORTS / "C2C_LIVE_BATCH_AUDIT.json", {
        "schema_version": "c2c-live-batch-audit-v1",
        "milestone": "C2C",
        "generated_at_utc": utc_now(),
        "generator_code_commit": commit,
        "derivation": "every number was produced by replaying the archived raw response "
                      "offline through a fresh validation pipeline",
        "logical_batch_id": LOGICAL_BATCH_ID,
        "logical_batches_executed": state["logical_batches_executed"],
        "second_batch_issued": False,
        "batch_contract": context.as_contract_record(),
        "status": state["status"],
        "provider_attempts": state["provider_attempts"],
        "transport_retries": state["transport_retries"],
        "rate_limit_events": state["rate_limit_events"],
        "requested_objects": BATCH_SIZE,
        "returned_objects": len(replay["rows"]),
        "accepted_objects": sum(1 for row in replay["rows"] if row["status"] == "accepted"),
        "route_policy_rejections": sum(1 for row in replay["rows"]
                                       if row["status"] == "rejected_route_policy"),
        "other_rejections": sum(1 for row in replay["rows"]
                                if row["status"] not in ("accepted", "rejected_route_policy")),
        "compiled_objects": sum(1 for row in replay["rows"]
                                if row["compiler_status"] == "compiled"),
        "compiler_failures_among_accepted": verdict.get("compiler_failures_among_accepted"),
        "response_issues": replay["response_issues"],
        "replay_verification": verification,
        "duplicates": duplicates,
        "verdict": verdict,
        "recipes": replay["rows"],
        "disposable": True,
        "enters_c3": False,
    })

    write_json(REPORTS / "C2C_COVERAGE_AUDIT.json", {
        "schema_version": "c2c-coverage-audit-v1",
        "milestone": "C2C",
        "generated_at_utc": utc_now(),
        "scope": "batch-level coverage over the frozen ontology only",
        "quota_values_changed_in_c2c": False,
        "compared_against_dataset_attack_families": False,
        "target_information_used": False,
        "coverage_quotas": context.quotas.as_dict(),
        "quota_compliance": quota,
        "coverage": coverage,
        "cooccurrence": pairs,
        "c2b_versus_c2c": comparison,
    })

    write_json(REPORTS / "C2C_C3_FREEZE_CANDIDATE.json", candidate)

    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "C2C_ROUTE_CONTRACT_REPORT.md").write_text(
        build_report(context, state, replay, quota, coverage, pairs, duplicates, comparison,
                     verdict, routes, verification, archive, c2b_replay), encoding="utf-8")
    print("wrote docs/c2c/C2C_ROUTE_CONTRACT_REPORT.md")
    (DOCS / "C2C_C3_FREEZE_CANDIDATE.md").write_text(
        build_freeze_doc(context, candidate, verdict, comparison), encoding="utf-8")
    print("wrote docs/c2c/C2C_C3_FREEZE_CANDIDATE.md")
    (DOCS / "C2C_IDENTITY_CORRECTION_NOTE.md").write_text(
        build_correction_note(context), encoding="utf-8")
    print("wrote docs/c2c/C2C_IDENTITY_CORRECTION_NOTE.md")

    print(f"\noutcome: {verdict['outcome']}")
    print(f"returned {len(replay['rows'])}/{BATCH_SIZE}, "
          f"accepted {sum(1 for row in replay['rows'] if row['status'] == 'accepted')}, "
          f"route rejections {sum(1 for row in replay['rows'] if row['status'] == 'rejected_route_policy')}")
    print(f"compiler failures among accepted: {verdict.get('compiler_failures_among_accepted')}")
    print(f"replay identical: {verification['identical']}")
    print(f"coverage: {comparison['verdict']}")
    print(f"C3_GENERATION_CONTRACT_IDENTITY: {candidate['c3_generation_contract_identity']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
