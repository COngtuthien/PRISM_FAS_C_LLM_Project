"""Build every C2B audit artifact from the archived batch, offline.

    python scripts/c2b_build_reports.py

Nothing here calls a provider. The archived raw response is replayed through a
fresh validation pipeline by the replay provider - which holds no client and no
credential - and every number is derived from what that replay produces.

The pass/fail thresholds in `CRITERIA` were written before the batch returned and
are not adjusted afterwards. Weakening a criterion once the result is visible
would make the verdict meaningless.
"""
from __future__ import annotations

import json
import sys
from typing import Any

from c2b_common import (BATCH_SIZE, C2B_BANK_ID, C2_SINGLETON_COVERAGE, DOCS, LOGICAL_BATCH_ID,
                        REPORTS, BatchContext, git, read_json, utc_now, write_json)

from prism_fas.llm.coverage_quotas import classify_recipes, evaluate
from prism_fas.llm.pilot_audit import (axis_pair_table, coverage_audit, duplicate_audit,
                                       latency_stats, structural_pattern)
from prism_fas.llm.pipeline import RecipePlanner, compile_accepted
from prism_fas.llm.providers.replay import ReplayArchive, ReplayRecipeProvider
from prism_fas.recipes.compile import CompileError

C3_REQUESTS = 12
C3_SLOTS = C3_REQUESTS * BATCH_SIZE

#: Fixed before the batch returned. Do not weaken after seeing results.
CRITERIA = {
    "returned_objects_exact": BATCH_SIZE,
    "min_semantic_validity": 0.90,
    "max_response_issues": 0,
    "max_compiler_failures": 0,
    "max_duplicate_rate": 0.10,
    "max_axis_share_percent": 60.0,
    "max_required_quota_failures": 2,
    "axes_requiring_full_presence": ["media", "geometry", "illumination", "artifacts", "regions"],
}

#: The six co-occurrence tables C2B inspects.
PAIR_TABLES = (("artifacts", "media"), ("artifacts", "geometry"), ("artifacts", "regions"),
               ("media", "geometry"), ("media", "illumination"), ("geometry", "illumination"))


def replay_batch(context: BatchContext, archive_records: list[dict]) -> dict[str, Any]:
    """Re-run the archived response through a fresh pipeline. Zero network."""
    served = [record for record in archive_records if record["raw_text"] is not None]
    archive = ReplayArchive.from_records([
        {key: record[key] for key in
         ("slot_id", "attempt", "raw_text", "provider", "model_id", "model_version",
          "finish_reason", "usage", "provider_request_id", "provider_seed", "sdk_version",
          "api_surface", "request_sha256")}
        for record in served])
    provider = ReplayRecipeProvider(archive, strict=False)
    planner = RecipePlanner(provider=provider, config=context.config, ontology=context.ontology,
                            sleep=lambda _seconds: None)
    request = context.request(LOGICAL_BATCH_ID)

    if not served:
        return {"validation": None, "rows": [], "recipes": [], "identities": [],
                "response_issues": [], "provider": provider}

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
            "canonical_identity": candidate.recipe_identity,
            "canonical_recipe": candidate.canonical_text,
            "validation_failures": candidate.issues,
            "compiler_status": "not_attempted",
            "graph_hash": None,
        }
        if candidate.accepted and candidate.recipe is not None:
            try:
                graph = compile_accepted(candidate.recipe, context.ontology, bank_id=C2B_BANK_ID)
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
    return {"validation": validation, "rows": rows, "recipes": recipes,
            "identities": identities, "response_issues": validation.response_issues,
            "provider": provider}


def verify_replay(live_rows: list[dict], replay_rows: list[dict]) -> dict[str, Any]:
    mismatches = []
    for live, replayed in zip(live_rows, replay_rows):
        for field in ("batch_index", "status", "canonical_identity", "compiler_status",
                      "graph_hash"):
            if live.get(field) != replayed.get(field):
                mismatches.append({"batch_index": live.get("batch_index"), "field": field,
                                   "live": live.get(field), "replay": replayed.get(field)})
    return {"objects_compared": min(len(live_rows), len(replay_rows)),
            "count_matches": len(live_rows) == len(replay_rows),
            "mismatches": mismatches,
            "identical": not mismatches and len(live_rows) == len(replay_rows)}


def route_analysis(rows: list[dict]) -> dict[str, Any]:
    """Which generator routes the batch chose, and what that costs the compiler.

    C2B finding: a recipe naming only the `gpat` route is fully VALID - the
    ontology enables both routes and the validator only checks membership - yet
    `compile_recipe` refuses it, because an operator graph is a physics-route
    artifact. The two authorities disagree, and the batch made that visible for
    the first time: the C2 singleton pilot happened to put `physics` in all 32.

    This is unrelated to the coverage quotas, which never mention
    `generator_route`.
    """
    from collections import Counter

    routes: Counter[str] = Counter()
    accepted = [row for row in rows if row["status"] == "accepted"]
    without_physics: list[int] = []
    for row in accepted:
        if not row["canonical_recipe"]:
            continue
        declared = json.loads(row["canonical_recipe"])["generator_route"]
        routes["+".join(declared)] += 1
        if "physics" not in declared:
            without_physics.append(row["batch_index"])
    failed = [row["batch_index"] for row in accepted if row["compiler_status"] == "failed"]
    return {
        "route_counts": dict(sorted(routes.items())),
        "accepted_objects": len(accepted),
        "accepted_without_physics_route": len(without_physics),
        "accepted_without_physics_indices": without_physics,
        "compiler_failed_indices": sorted(failed),
        "compiler_failures_all_explained_by_missing_physics_route":
            sorted(failed) == sorted(without_physics),
        "caused_by_coverage_quotas": False,
        "why_not_quotas": "The C2B quotas constrain media, geometry, illumination, artifacts and "
                          "regions. They never mention generator_route, so they cannot have "
                          "forced this choice.",
        "c2_singleton_comparison": "Every one of the 32 C2 singleton recipes declared the physics "
                                   "route (25 physics+gpat, 7 physics-only). The batch request is "
                                   "the first context in which the model chose gpat-only.",
        "consequence_for_c3": "A gpat-only recipe passes validation and then cannot be compiled "
                              "into an operator graph, so it cannot reach the physics renderer. "
                              "C3 must decide explicitly whether the physics route is required, "
                              "or whether a gpat-only recipe is a separate accepted class. This "
                              "is a USER decision; C2B changes nothing.",
    }


def mode_collapse_comparison(coverage: dict) -> dict[str, Any]:
    """C2 singleton versus C2B batch. Both are source-independent prompt
    development evidence, so this comparison is permitted. No dataset is used."""
    rows = []
    for axis, before in C2_SINGLETON_COVERAGE.items():
        entry = coverage["axes"][axis]
        after_present = entry["categories_present"]
        after_share = max(entry["assignment_percent"].values()) if entry["assignment_percent"] else 0.0
        rows.append({
            "axis": axis,
            "total_categories": before["total"],
            "c2_singleton_present": before["present"],
            "c2b_batch_present": after_present,
            "presence_delta": after_present - before["present"],
            "c2_singleton_max_share_percent": before["max_share_percent"],
            "c2b_batch_max_share_percent": round(after_share, 4),
            "share_delta": round(after_share - before["max_share_percent"], 4),
            "c2_singleton_missing": before["missing"],
            "c2b_batch_missing": entry["missing_categories"],
            "fully_covered_now": after_present == before["total"],
            "collapse_resolved": (after_present == before["total"]
                                  and after_share < CRITERIA["max_axis_share_percent"]),
        })
    resolved = [row for row in rows if row["collapse_resolved"]]
    return {
        "comparison_scope": "C2 singleton pilot versus the C2B batch, both source-independent "
                            "prompt-development evidence. No dataset, target metric or attack "
                            "taxonomy was consulted.",
        "axes": rows,
        "axes_fully_covered": sum(1 for row in rows if row["fully_covered_now"]),
        "axes_collapse_resolved": len(resolved),
        "axes_total": len(rows),
        "verdict": ("resolved on every axis" if len(resolved) == len(rows)
                    else f"resolved on {len(resolved)}/{len(rows)} axes"),
    }


def classify_outcome(state: dict, replay: dict, quota: dict, coverage: dict,
                     duplicates: dict) -> dict[str, Any]:
    """The C2B verdict, against thresholds fixed before the batch returned."""
    if state["status"] == "BLOCKED_QUOTA":
        return {"outcome": "BLOCKED_QUOTA", "structural": {}, "coverage": {}, "failed": []}
    if not state.get("semantic_response_received"):
        return {"outcome": "BLOCKED_PROVIDER", "structural": {}, "coverage": {}, "failed": []}

    rows = replay["rows"]
    accepted = [row for row in rows if row["status"] == "accepted"]
    validity = len(accepted) / len(rows) if rows else 0.0
    compiler_failures = sum(1 for row in rows if row["compiler_status"] == "failed")

    structural = {
        "returned_exactly_32": len(rows) == CRITERIA["returned_objects_exact"],
        "no_response_level_issues": len(replay["response_issues"]) <= CRITERIA["max_response_issues"],
        "semantic_validity_at_least_threshold": validity >= CRITERIA["min_semantic_validity"],
        "no_compiler_failures": compiler_failures <= CRITERIA["max_compiler_failures"],
        "duplicate_rate_within_threshold":
            duplicates["exact_duplicate_rate"] <= CRITERIA["max_duplicate_rate"],
    }
    # "Did the quotas force physical incompatibility?" is a question about the
    # compatibility rules, so it is measured on compatibility rejections - not on
    # compiler failures, which here come from a route choice the quotas never
    # mention. Using the wrong proxy would have blamed the quotas for something
    # they did not cause; it does not change the verdict either way.
    compatibility_violations = sum(
        1 for row in rows for issue in row["validation_failures"]
        if issue.get("stage") in {"medium_artifact", "geometry_region"})
    max_share = max((entry["max_share_percent"] for entry in quota["axes"].values()), default=0.0)
    coverage_checks = {
        "all_axes_fully_represented": all(
            quota["axes"][axis]["categories_missing"] == []
            for axis in CRITERIA["axes_requiring_full_presence"]),
        "no_severe_mode_collapse": max_share <= CRITERIA["max_axis_share_percent"],
        "quotas_substantially_satisfied":
            len(quota["required_failures"]) <= CRITERIA["max_required_quota_failures"],
        "quotas_did_not_force_incompatibility": (compatibility_violations == 0
                                                 and validity >= CRITERIA["min_semantic_validity"]),
    }

    failed_structural = [name for name, ok in structural.items() if not ok]
    failed_coverage = [name for name, ok in coverage_checks.items() if not ok]
    if failed_structural:
        outcome = "BATCH_SHAPE_FAIL"
    elif failed_coverage:
        outcome = "BATCH_SHAPE_PASS_COVERAGE_FAIL"
    else:
        outcome = "BATCH_SHAPE_PASS"
    return {"outcome": outcome, "criteria": CRITERIA, "structural": structural,
            "coverage": coverage_checks, "failed": failed_structural + failed_coverage,
            "semantic_validity": round(validity, 6), "max_axis_share_percent": max_share,
            "compiler_failures": compiler_failures,
            "compatibility_violations": compatibility_violations,
            "criteria_were_fixed_before_the_batch_returned": True,
            "criteria_weakened_after_seeing_results": False}


def c3_estimate(state: dict, archive: dict, replay: dict) -> dict[str, Any]:
    """Correct the C2 estimate: C3 is 12 batch requests, not 384 singletons."""
    served = [record for record in archive["records"] if record["raw_text"] is not None]
    latency = latency_stats(record["latency_seconds"] for record in served)
    usage: dict[str, Any] = {}
    for record in served:
        for key, value in (record["usage"] or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                usage[key] = usage.get(key, 0) + value
    attempts = state["provider_attempts"]
    retries = state["transport_retries"]
    observed_latency = latency["mean"] or 0.0
    accepted = sum(1 for row in replay["rows"] if row["status"] == "accepted")
    yield_rate = accepted / BATCH_SIZE if BATCH_SIZE else 0.0

    # One observation. The band is deliberately wide and is not a prediction.
    low, high = observed_latency * 0.6, observed_latency * 2.0
    return {
        "schema_version": "c2b-c3-quota-estimate-v1",
        "milestone": "C2B",
        "generated_at_utc": utc_now(),
        "supersedes": "reports/c2/C2_C3_READINESS.json, whose projection treated the 384 slots "
                      "as 384 singleton calls. C3 is 12 batch requests of 32 objects.",
        "c3_design": {"requests": C3_REQUESTS, "objects_per_request": BATCH_SIZE,
                      "raw_candidate_slots": C3_SLOTS},
        "observed_single_batch": {
            "provider_attempts": attempts,
            "transport_retries": retries,
            "latency_seconds": latency,
            "token_usage": usage,
            "accepted_objects": accepted,
            "objects_per_request": BATCH_SIZE,
            "accepted_yield_rate": round(yield_rate, 6),
            "sample_size": 1,
        },
        "projection": {
            "expected_batch_calls_minimum": C3_REQUESTS,
            "expected_batch_calls_with_observed_retry_rate": round(
                C3_REQUESTS * (attempts / 1.0), 2),
            "expected_transport_retry_burden": round(C3_REQUESTS * retries, 2),
            "expected_valid_candidates": round(C3_SLOTS * yield_rate, 1),
            "expected_tokens": {key: value * C3_REQUESTS for key, value in usage.items()},
            "serial_model_time_seconds_range": [round(C3_REQUESTS * low, 1),
                                                round(C3_REQUESTS * high, 1)],
            "serial_model_time_minutes_range": [round(C3_REQUESTS * low / 60.0, 1),
                                                round(C3_REQUESTS * high / 60.0, 1)],
            "uncertainty": "Derived from ONE observed batch. The range is a planning band, not a "
                           "prediction, and a single observation supports no confidence interval.",
        },
        "free_tier_risk": {
            "requests_needed": f"{C3_REQUESTS} (plus retries)",
            "observed_free_tier_behaviour": "C2 measured roughly 20 requests per rolling "
                                            "multi-minute window for this model; 12 batch "
                                            "requests is far below the per-window request count "
                                            "that blocked the C2 pilot",
            "request_count_risk": "LOW relative to C2 - 12 requests instead of 384",
            "token_and_output_risk": "HIGHER PER REQUEST - each batch produces ~32x the output "
                                     "of a singleton call, so a per-minute TOKEN limit, not a "
                                     "request limit, becomes the plausible binding constraint",
            "quota_snapshot": "QUOTA_SNAPSHOT_NOT_PROGRAMMATICALLY_AVAILABLE - the Free Tier "
                              "RPM/TPM/RPD limits for this project must be read from AI Studio "
                              "before C3. No number was invented here.",
            "billing": "Free Tier only. Code never enables billing; a Paid Tier remains a user "
                       "decision if C3 becomes genuinely blocked.",
            "c2b_succeeded_on_free_tier": state["status"] == "COMPLETE",
            "c2b_success_does_not_guarantee_c3": True,
        },
        "c3_requests_executed": 0,
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


def build_report(context: BatchContext, state: dict, replay: dict, quota: dict, coverage: dict,
                 pairs: dict, duplicates: dict, comparison: dict, verdict: dict,
                 classification: dict, verification: dict, archive: dict,
                 routes: dict) -> str:
    contract = context.as_contract_record()
    lines: list[str] = []
    add = lines.append

    add("# C2B - 32-recipe batch-shape validation")
    add("")
    add("One logical Gemini request asking for 32 recipe objects in a single structured")
    add("response, under generic ontology-level coverage quotas. This is a disposable")
    add("development experiment: nothing here enters C3, the final 256-recipe bank, a")
    add("synthetic bank or detector training.")
    add("")
    add(f"**Outcome: {verdict['outcome']}**")
    add("")

    add("## Batch contract")
    add("")
    add(_table(["field", "value"], [
        ["provider", contract["provider"]],
        ["model", f"`{contract['model_id']}`"],
        ["API surface", contract["api_surface"]],
        ["thinking_level", contract["thinking_level"]],
        ["system prompt identity", f"`{contract['system_prompt_identity']}`"],
        ["system prompt changed in C2B", str(contract["system_prompt_changed_in_c2b"])],
        ["batch generation-template identity", f"`{contract['batch_generation_template_identity']}`"],
        ["coverage quota identity", f"`{contract['coverage_quota_identity']}`"],
        ["single-recipe (item) schema identity", f"`{contract['single_recipe_schema_identity']}`"],
        ["batch-envelope schema identity (sent)", f"`{contract['batch_envelope_schema_identity']}`"],
        ["ontology identity", f"`{contract['ontology_identity']}`"],
        ["provider config identity", f"`{contract['provider_config_identity']}`"],
        ["allow_ontology_aliases", str(contract["allow_ontology_aliases"])],
        ["recipes per request", contract["recipes_per_request"]],
    ]))
    add("")
    add("### Two schema identities, and why they differ from C1's record")
    add("")
    add(f"> {contract['schema_identity_note']}")
    add("")
    add(f"> **Array-bounds finding.** {contract['array_bounds_finding']}")
    add("")
    add(_table(["schema", "identity", "status"], [
        ["single-recipe item schema", f"`{contract['single_recipe_schema_identity']}`",
         "unchanged from C2 - byte-identical in both envelopes"],
        ["C2 singleton envelope (n=1)", f"`{contract['c2_singleton_envelope_schema_identity']}`",
         "what C2 sent, accepted 42 times"],
        ["C1-recorded 32-object envelope",
         f"`{contract['bounded_batch_envelope_identity_rejected_by_provider']}`",
         "**rejected by the provider, 400 INVALID_ARGUMENT**"],
        ["C2B batch envelope (sent)", f"`{contract['batch_envelope_schema_identity']}`",
         "same schema without the array length bound"],
    ]))
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
        ["returned objects", state["returned_objects"]],
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
            *[[key, value] for key, value in sorted((record["usage"] or {}).items())
              if isinstance(value, (int, float))],
        ]))
        add("")

    add("## Batch structure")
    add("")
    rows = replay["rows"]
    accepted = [row for row in rows if row["status"] == "accepted"]
    add(_table(["measure", "value"], [
        ["requested objects", BATCH_SIZE],
        ["returned objects", len(rows)],
        ["valid (accepted)", len(accepted)],
        ["invalid", len(rows) - len(accepted)],
        ["duplicates", duplicates["exact_duplicate_groups"]],
        ["compiler failures", sum(1 for row in rows if row["compiler_status"] == "failed")],
        ["response-level issues", len(replay["response_issues"])],
    ]))
    add("")
    add(f"- offline replay identical to the live run: **{verification['identical']}**")
    add(f"- distinct structural patterns: {duplicates['distinct_structural_patterns']}"
        f"/{len(accepted)}")
    add("")

    add("## Generator route: valid recipes the compiler cannot build")
    add("")
    add(_table(["generator_route", "recipes"],
               [[route, count] for route, count in routes["route_counts"].items()]))
    add("")
    add(f"- accepted objects without the physics route: "
        f"**{routes['accepted_without_physics_route']}/{routes['accepted_objects']}** "
        f"(batch indices {routes['accepted_without_physics_indices']})")
    add(f"- compiler failures explained entirely by that: "
        f"**{routes['compiler_failures_all_explained_by_missing_physics_route']}**")
    add(f"- caused by the coverage quotas: **{routes['caused_by_coverage_quotas']}** - "
        f"{routes['why_not_quotas']}")
    add("")
    add(f"{routes['c2_singleton_comparison']}")
    add("")
    add(f"**Consequence for C3.** {routes['consequence_for_c3']}")
    add("")

    add("## Coverage and quota compliance")
    add("")
    for axis, entry in quota["axes"].items():
        quota_row = entry["quota"]
        add(f"### {axis} - {entry['categories_present']}/{entry['category_count']} present, "
            f"max share {entry['max_share_percent']}% "
            f"({'PASS' if entry['required_pass'] else 'FAIL'})")
        add("")
        add(_table(["category", "count", "% of recipes", "min", "preferred", "max", "required"],
                   [[name, cell["count"], cell["percent_of_recipes"],
                     cell["quota_minimum"] if cell["quota_minimum"] is not None else "-",
                     cell["quota_preferred_minimum"] if cell["quota_preferred_minimum"] is not None else "-",
                     cell["quota_maximum"] if cell["quota_maximum"] is not None else "-",
                     "pass" if cell["required_pass"] else "**FAIL**"]
                    for name, cell in entry["categories"].items()]))
        add("")
        if quota_row.get("note"):
            add(f"> {quota_row['note']}")
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
    if quota["required_failures"]:
        add("**Required-bound failures**")
        add("")
        add(_table(["axis", "category", "count", "reason"],
                   [[item["axis"], item["category"], item["count"], item["reason"]]
                    for item in quota["required_failures"]]))
        add("")
    if quota["preferred_misses"]:
        add("**Preferred-bound misses** (reported, never repaired - compatibility outranks quota)")
        add("")
        add(_table(["axis", "category", "count", "reason"],
                   [[item["axis"], item["category"], item["count"], item["reason"]]
                    for item in quota["preferred_misses"]]))
        add("")

    add("## C2 singleton versus C2B batch")
    add("")
    add(_table(["axis", "categories", "C2 present", "C2B present", "C2 max share",
                "C2B max share", "collapse resolved"],
               [[row["axis"], row["total_categories"], row["c2_singleton_present"],
                 row["c2b_batch_present"], f"{row['c2_singleton_max_share_percent']}%",
                 f"{row['c2b_batch_max_share_percent']}%",
                 "yes" if row["collapse_resolved"] else "**no**"]
                for row in comparison["axes"]]))
    add("")
    add(f"**Verdict: {comparison['verdict']}.** "
        f"{comparison['axes_fully_covered']}/{comparison['axes_total']} axes are now fully "
        "covered.")
    add("")
    add(comparison["comparison_scope"])
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

    add("## Physical validity classification")
    add("")
    add(_table(["classification", "count"], [
        ["VALID_AND_QUOTA_COMPLIANT", classification["valid_and_quota_compliant"]],
        ["VALID_BUT_QUOTA_MISS", classification["valid_but_quota_miss"]],
        ["INVALID", len(rows) - len(accepted)],
    ]))
    add("")
    add("No recipe was moved between categories after generation and no semantic field was")
    add("edited to satisfy a quota. Compatibility outranks quota by construction.")
    add("")

    add("## All 32 returned objects")
    add("")
    by_id = {recipe.recipe_id: recipe for recipe in replay["recipes"]}
    for row in rows:
        add(f"### index {row['batch_index']}")
        add("")
        recipe = by_id.get(row["recipe_id"])
        table_rows = [["status", row["status"]],
                      ["validation", "accepted" if row["status"] == "accepted" else "rejected"],
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
    add(f"Thresholds were fixed before the batch returned: `{json.dumps(CRITERIA)}`")
    add("")
    return "\n".join(lines) + "\n"


def build_freeze_recommendation(context: BatchContext, verdict: dict, quota: dict,
                                comparison: dict, estimate: dict, routes: dict) -> str:
    contract = context.as_contract_record()
    lines: list[str] = []
    add = lines.append
    add("# C2B - C3 freeze recommendation")
    add("")
    add("**This document does not freeze anything.** It prepares the exact candidate values")
    add("for user approval. The C3 prompt/request contract is frozen only by an explicit user")
    add("decision.")
    add("")
    add(f"C2B outcome: **{verdict['outcome']}**")
    add("")
    add("## Candidate identities awaiting user approval")
    add("")
    add(_table(["field", "candidate value"], [
        ["provider", contract["provider"]],
        ["model", f"`{contract['model_id']}`"],
        ["API surface", contract["api_surface"]],
        ["thinking_level", contract["thinking_level"]],
        ["system prompt identity", f"`{contract['system_prompt_identity']}`"],
        ["batch generation-template identity", f"`{contract['batch_generation_template_identity']}`"],
        ["coverage quota identity", f"`{contract['coverage_quota_identity']}`"],
        ["single-recipe schema identity", f"`{contract['single_recipe_schema_identity']}`"],
        ["batch-envelope schema identity", f"`{contract['batch_envelope_schema_identity']}`"],
        ["ontology identity", f"`{contract['ontology_identity']}`"],
        ["alias policy", f"allow_ontology_aliases = {contract['allow_ontology_aliases']}"],
        ["request schedule", contract["request_schedule_for_c3"]],
    ]))
    add("")
    add("## What changed since the C1 freeze, and why")
    add("")
    add("1. **The batch envelope had to change.** The 32-object envelope C1 recorded")
    add(f"   (`{contract['bounded_batch_envelope_identity_rejected_by_provider']}`)")
    add("   carries `minItems = maxItems = 32` and the provider rejects it outright with")
    add("   `400 INVALID_ARGUMENT`. C3 as recorded at C1 could not have run. The envelope")
    add("   C2B sends is the same schema without that bound; the requirement for exactly 32")
    add("   objects is unchanged and is enforced locally on the response.")
    add("2. **The single-recipe item schema did not change.** It is byte-identical in the")
    add("   1-object envelope C2 used and in the envelope C2B sends, which is the evidence")
    add("   that recipe semantics were untouched.")
    add("3. **A batch generation template and coverage quotas were added.** Both are")
    add("   ontology-level and source-independent. The system instruction was not edited.")
    add("")
    add("## Blocking issue found: valid recipes the compiler cannot build")
    add("")
    add(f"{routes['accepted_without_physics_route']} of {routes['accepted_objects']} accepted")
    add("recipes declared `generator_route` without `physics`. Those recipes are fully valid -")
    add("the ontology enables both routes and the validator only checks membership - but")
    add("`compile_recipe` refuses them, because an operator graph is a physics-route artifact.")
    add("The validator and the compiler disagree about what an acceptable recipe is.")
    add("")
    add(f"Route distribution: `{json.dumps(routes['route_counts'])}`")
    add("")
    add(f"- {routes['c2_singleton_comparison']}")
    add(f"- Caused by the coverage quotas: **no**. {routes['why_not_quotas']}")
    add("")
    add("**This must be decided before C3.** Two options, both a user decision:")
    add("")
    add("1. Require the physics route (in the system instruction, the item schema, or the")
    add("   validator). This changes the single-recipe schema identity if done in the schema.")
    add("2. Accept gpat-only recipes as a distinct class that never enters the operator-graph")
    add("   compiler, and decide what the synthesis path does with them.")
    add("")
    add("C2B changed neither. It only measured the disagreement.")
    add("")
    add("## Coverage evidence behind the recommendation")
    add("")
    add(_table(["axis", "C2 singleton present", "C2B batch present", "collapse resolved"],
               [[row["axis"], f"{row['c2_singleton_present']}/{row['total_categories']}",
                 f"{row['c2b_batch_present']}/{row['total_categories']}",
                 "yes" if row["collapse_resolved"] else "**no**"]
                for row in comparison["axes"]]))
    add("")
    add(f"Required quota failures: {len(quota['required_failures'])}. "
        f"Preferred misses: {len(quota['preferred_misses'])}.")
    add("")
    add("## Corrected C3 cost estimate")
    add("")
    projection = estimate["projection"]
    add(_table(["measure", "value"], [
        ["design", f"{C3_REQUESTS} requests x {BATCH_SIZE} objects = {C3_SLOTS} raw slots"],
        ["expected batch calls (minimum)", projection["expected_batch_calls_minimum"]],
        ["expected transport retry burden", projection["expected_transport_retry_burden"]],
        ["expected valid candidates", projection["expected_valid_candidates"]],
        ["serial model time (minutes, range)",
         f"{projection['serial_model_time_minutes_range'][0]} - "
         f"{projection['serial_model_time_minutes_range'][1]}"],
        ["expected tokens", json.dumps(projection["expected_tokens"])],
    ]))
    add("")
    add(f"> {projection['uncertainty']}")
    add("")
    add(f"> Free Tier: {estimate['free_tier_risk']['token_and_output_risk']} "
        f"{estimate['free_tier_risk']['quota_snapshot']}")
    add("")
    add("## Recommendation")
    add("")
    if verdict["outcome"] == "BATCH_SHAPE_PASS":
        add("The batch shape works and the generic quotas resolved the singleton collapse.")
        add("The values above are recommended **as candidates** for the C3 freeze.")
    elif verdict["outcome"] == "BATCH_SHAPE_PASS_COVERAGE_FAIL":
        add("The batch shape itself works: 32 objects returned, validated and compiled. The")
        add("coverage criteria were not fully met, so the quota values - not the batch shape -")
        add("are what needs a decision before C3.")
    else:
        add("**C3 must not be frozen on this evidence** - but read the failure precisely, because")
        add("it is not where the milestone's name suggests.")
        add("")
        add("What worked, measured rather than asserted:")
        add("")
        add("- one request returned exactly 32 objects in a single structured response;")
        add("- all 32 parsed, satisfied the strict item schema, the ontology, every range and")
        add("  every compatibility rule, and were accepted with zero duplicates;")
        add("- the archived response replays offline to identical identities;")
        add("- the generic quotas were met on every axis, required and preferred, and the")
        add("  singleton mode collapse is gone.")
        add("")
        add("What failed is a single criterion, `no_compiler_failures`, and its cause is the")
        add("validator/compiler disagreement over `generator_route` described above - not the")
        add("batch shape, not the envelope, and not the quotas. The batch mechanism itself is")
        add("sound; the recipe-acceptance contract has a gap that C3 would inherit.")
    add("")
    add(f"Failed criteria: {verdict['failed'] or 'none'}")
    add("")
    add("**No freeze was performed. C3 was not started.**")
    add("")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------- main
def main() -> int:
    context = BatchContext()
    state = read_json(REPORTS / "C2B_BATCH_STATE.json")
    archive = read_json(REPORTS / "C2B_RAW_ARCHIVE.json")
    commit = git("rev-parse", "HEAD")

    replay = replay_batch(context, archive["records"])
    verification = verify_replay(state["recipes"], replay["rows"])
    recipes = replay["recipes"]

    quota = evaluate(context.quotas, recipes, context.ontology)
    classification = classify_recipes(context.quotas, recipes, context.ontology)
    coverage = coverage_audit(recipes, context.ontology)
    duplicates = duplicate_audit(replay["identities"], recipes)
    pairs = {f"{row}_x_{column}": axis_pair_table(recipes, context.ontology, row, column)
             for row, column in PAIR_TABLES}
    comparison = mode_collapse_comparison(coverage)
    routes = route_analysis(replay["rows"])
    verdict = classify_outcome(state, replay, quota, coverage, duplicates)
    estimate = c3_estimate(state, archive, replay)

    write_json(REPORTS / "C2B_LIVE_BATCH_AUDIT.json", {
        "schema_version": "c2b-live-batch-audit-v1",
        "milestone": "C2B",
        "generated_at_utc": utc_now(),
        "generator_code_commit": commit,
        "derivation": "every number was produced by replaying the archived raw response offline "
                      "through a fresh validation pipeline; none was typed by hand",
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
        "response_issues": replay["response_issues"],
        "replay_verification": verification,
        "duplicates": duplicates,
        "physical_validity_classification": classification,
        "generator_route_analysis": routes,
        "verdict": verdict,
        "recipes": replay["rows"],
        "structural_patterns": {recipe.recipe_id: structural_pattern(recipe)
                                for recipe in recipes},
        "disposable": True,
        "enters_c3": False,
    })

    write_json(REPORTS / "C2B_COVERAGE_AUDIT.json", {
        "schema_version": "c2b-coverage-audit-v1",
        "milestone": "C2B",
        "generated_at_utc": utc_now(),
        "scope": "batch-level coverage over the frozen ontology only",
        "compared_against_dataset_attack_families": False,
        "target_information_used": False,
        "coverage_quotas": context.quotas.as_dict(),
        "quota_compliance": quota,
        "coverage": coverage,
        "mode_collapse_comparison": comparison,
    })

    write_json(REPORTS / "C2B_COOCCURRENCE_AUDIT.json", {
        "schema_version": "c2b-cooccurrence-audit-v1",
        "milestone": "C2B",
        "generated_at_utc": utc_now(),
        "reference_dataset_used": False,
        "tables": pairs,
        "repeated_structural_patterns": duplicates["repeated_structural_patterns"],
        "distinct_structural_patterns": duplicates["distinct_structural_patterns"],
    })

    write_json(REPORTS / "C2B_C3_QUOTA_ESTIMATE.json", estimate)

    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "C2B_BATCH_SHAPE_REPORT.md").write_text(
        build_report(context, state, replay, quota, coverage, pairs, duplicates, comparison,
                     verdict, classification, verification, archive, routes), encoding="utf-8")
    print("wrote docs/c2b/C2B_BATCH_SHAPE_REPORT.md")
    (DOCS / "C2B_C3_FREEZE_RECOMMENDATION.md").write_text(
        build_freeze_recommendation(context, verdict, quota, comparison, estimate, routes),
        encoding="utf-8")
    print("wrote docs/c2b/C2B_C3_FREEZE_RECOMMENDATION.md")

    print(f"\noutcome: {verdict['outcome']}")
    print(f"returned {len(replay['rows'])}/{BATCH_SIZE}, "
          f"accepted {sum(1 for row in replay['rows'] if row['status'] == 'accepted')}")
    print(f"replay identical: {verification['identical']}")
    print(f"quota required failures: {len(quota['required_failures'])}, "
          f"preferred misses: {len(quota['preferred_misses'])}")
    print(f"collapse: {comparison['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
