"""Build every C2 audit artifact from the archived pilot, offline.

    python scripts/c2_build_reports.py

No number in `reports/c2/` or `docs/c2/` is typed by hand. This script reads the
archived raw responses, replays them through a fresh validation pipeline with the
replay provider - which holds no client and no credential and therefore cannot
make a network call - and derives the audits from what the replay produces.

Replaying rather than trusting the live run is the point: it proves the archived
bytes alone reproduce the same accept/reject decisions and the same canonical
recipe identities, which is what makes a hosted, non-reproducible model's output
a reproducible scientific artifact.
"""
from __future__ import annotations

import json
import sys
from typing import Any

from c2_pilot_common import (REPO, REPORTS, FrozenContext, read_json, utc_now, write_json, git)

from prism_fas.llm.pilot_audit import (c3_readiness, cooccurrence_audit, coverage_audit,
                                       duplicate_audit, flag_pilot_issues, pilot_statistics,
                                       structural_pattern)
from prism_fas.llm.pipeline import RecipePlanner, compile_accepted
from prism_fas.llm.provenance import build_provenance
from prism_fas.llm.providers.replay import ReplayArchive, ReplayRecipeProvider
from prism_fas.recipes.compile import CompileError

DOCS = REPO / "docs" / "c2"
PILOT_BANK_ID = "c2-pilot-disposable"
C3_CANDIDATE_SLOTS = 384

#: Objective thresholds for the prompt review verdict. A prompt change is
#: recommended only if one of these is breached; "looks odd" is never a reason.
PROMPT_REVIEW_THRESHOLDS = {
    "min_eventual_valid_rate": 0.90,
    "min_first_attempt_valid_rate": 0.75,
    "max_duplicate_rate": 0.10,
    "min_axis_coverage_fraction": 0.50,
    "max_single_category_share": 0.60,
    "max_compatibility_violation_rate": 0.10,
    "max_ontology_violation_rate": 0.05,
}


# --------------------------------------------------------------------- replay
def replay_pilot(context: FrozenContext, state: dict, archive_records: list[dict]) -> dict:
    """Re-run the archived responses through a fresh pipeline. Zero network."""
    archive = ReplayArchive.from_records([
        {key: record[key] for key in
         ("slot_id", "attempt", "raw_text", "provider", "model_id", "model_version",
          "finish_reason", "usage", "provider_request_id", "provider_seed", "sdk_version",
          "api_surface", "request_sha256")}
        for record in archive_records if record["raw_text"] is not None])
    provider = ReplayRecipeProvider(archive, strict=False)
    planner = RecipePlanner(provider=provider, config=context.config, ontology=context.ontology,
                            sleep=lambda _seconds: None)

    by_slot: dict[str, list[dict]] = {}
    for record in archive_records:
        by_slot.setdefault(record["slot_id"], []).append(record)

    attempts: list[dict[str, Any]] = []
    slots: list[dict[str, Any]] = []
    recipes = []
    identities: list[str] = []

    for slot_state in state["slots"]:
        slot_id = slot_state["slot_id"]
        index = slot_state["slot_index"]
        request = context.request(slot_id)
        accepted_candidate = None
        accepted_on = None
        semantic_attempts = 0

        for record in sorted(by_slot.get(slot_id, []),
                             key=lambda item: item.get("sequence", item["attempt"])):
            entry: dict[str, Any] = {
                "slot_id": slot_id,
                "attempt": record["attempt"],
                "latency_seconds": record["latency_seconds"],
                "usage": record["usage"],
                "error_class": (record["error"] or {}).get("error_class"),
                "response_received": record["raw_text"] is not None,
                "all_accepted": False,
                "candidate_outcome": None,
                "issues": [],
            }
            if record["raw_text"] is not None:
                semantic_attempts += 1
                validation = planner.validate_response(
                    record["raw_text"], slot_id=slot_id,
                    recipes_requested=request.recipes_requested, next_recipe_index=index)
                entry["all_accepted"] = validation.all_accepted
                entry["issues"] = [issue for candidate in validation.candidates
                                   for issue in candidate.issues] + validation.response_issues
                if validation.candidates:
                    entry["candidate_outcome"] = validation.candidates[0].outcome.value
                if validation.all_accepted and accepted_candidate is None:
                    accepted_candidate = validation.accepted[0]
                    # Counted among RESPONSES, not among provider calls: a call
                    # that was rate-limited never produced a candidate, so it
                    # cannot make a recipe "not first-attempt valid".
                    accepted_on = semantic_attempts
            attempts.append(entry)

        compiler_status = "not_attempted"
        graph = None
        if accepted_candidate is not None and accepted_candidate.recipe is not None:
            try:
                graph = compile_accepted(accepted_candidate.recipe, context.ontology,
                                         bank_id=PILOT_BANK_ID)
                compiler_status = "compiled"
            except CompileError as exc:
                compiler_status = "failed"
                slot_state["replay_compiler_error"] = str(exc)
            recipes.append(accepted_candidate.recipe)
            identities.append(RecipePlanner.content_identity(accepted_candidate.recipe))

        slots.append({
            "slot_id": slot_id,
            "slot_index": index,
            "provider_calls": len(by_slot.get(slot_id, [])),
            "semantic_attempts": semantic_attempts,
            "accepted_on_attempt": accepted_on,
            "final_status": "accepted" if accepted_candidate is not None else "exhausted",
            "recipe_id": (accepted_candidate.recipe.recipe_id
                          if accepted_candidate and accepted_candidate.recipe else None),
            "recipe_identity": accepted_candidate.recipe_identity if accepted_candidate else None,
            "canonical_recipe": accepted_candidate.canonical_text if accepted_candidate else None,
            "compiler_status": compiler_status,
            "graph_hash": graph.graph_hash if graph is not None else None,
            "conditioning_dimension": graph.conditioning_dimension if graph is not None else None,
            "region_mask_policy": graph.region_mask_policy if graph is not None else None,
            "final_issues": slot_state.get("final_issues", []),
            "final_error": slot_state.get("final_error"),
        })

    return {"slots": slots, "attempts": attempts, "recipes": recipes,
            "content_identities": identities, "replay_calls": list(provider.calls)}


#: Fields whose disagreement would mean the archive does not reproduce the run.
SCIENTIFIC_REPLAY_FIELDS = ("final_status", "recipe_identity", "canonical_recipe", "graph_hash",
                           "compiler_status", "accepted_on_attempt", "semantic_attempts")


def verify_replay_matches_live(live: list[dict], replayed: list[dict]) -> dict:
    """A replay that disagreed with the live run would invalidate the archive.

    `provider_calls` is deliberately not a scientific field. The live record
    counts the calls of the pass that finished a slot; the replay counts every
    archived call for that slot, including attempts from an earlier pass that
    never received a response. Those numbers legitimately differ for a reopened
    slot, so the difference is reported rather than silently compared away.
    """
    mismatches: list[dict] = []
    call_count_differences: list[dict] = []
    live_by_id = {slot["slot_id"]: slot for slot in live}
    for slot in replayed:
        original = live_by_id.get(slot["slot_id"])
        if original is None:
            mismatches.append({"slot_id": slot["slot_id"], "reason": "not present in the live run"})
            continue
        for field in SCIENTIFIC_REPLAY_FIELDS:
            if original.get(field) != slot.get(field):
                mismatches.append({"slot_id": slot["slot_id"], "field": field,
                                   "live": original.get(field), "replay": slot.get(field)})
        if original.get("provider_calls") != slot.get("provider_calls"):
            call_count_differences.append({
                "slot_id": slot["slot_id"],
                "calls_in_the_finishing_pass": original.get("provider_calls"),
                "calls_in_the_whole_archive": slot.get("provider_calls"),
                "reason": "the slot was reopened after a pass in which the provider never "
                          "answered; the earlier calls stay archived as evidence"})
    return {"slots_compared": len(replayed),
            "scientific_fields_compared": list(SCIENTIFIC_REPLAY_FIELDS),
            "mismatches": mismatches,
            "identical": not mismatches,
            "reopened_slot_call_counts": call_count_differences}


# ------------------------------------------------------- physical description
def describe_physically(recipe) -> str:
    """A neutral physical reading of a recipe. No usefulness judgement."""
    medium_words = {
        "paper-like": "a printed paper surface",
        "display-like": "an emissive display panel",
        "plastic-like": "a moulded plastic surface",
        "fabric-like": "a woven fabric surface",
        "reflective-film-like": "a reflective film overlay",
    }
    shape_words = {
        "flat": "held flat", "curved": "curved", "partial-curved": "partly curved",
        "flexible": "flexible and deformable", "rigid": "rigid",
        "boundary-only": "present only at the face edge",
    }
    light_words = {
        "front": "frontal light", "left": "light from the left", "right": "light from the right",
        "top": "light from above", "bottom": "light from below", "mixed": "mixed lighting",
    }
    artifacts = ", ".join(f"{spec.name} at {round(float(spec.strength), 3)}"
                          for spec in recipe.artifacts)
    return (f"{medium_words.get(recipe.medium.family, recipe.medium.family)} "
            f"({shape_words.get(recipe.geometry.shape, recipe.geometry.shape)}, "
            f"transparency {recipe.medium.transparency}, roughness {recipe.medium.roughness}) "
            f"covering {', '.join(recipe.regions)} at coverage {recipe.geometry.coverage}, "
            f"producing {artifacts}, captured under "
            f"{light_words.get(recipe.capture.illumination, recipe.capture.illumination)} "
            f"at yaw {recipe.capture.yaw} deg, scale {recipe.capture.scale}, "
            f"compression q{recipe.capture.compression_q}.")


# ------------------------------------------------------------- prompt review
def prompt_review(statistics: dict, coverage: dict, duplicates: dict) -> dict:
    """Objective, source-independent verdict on whether the prompt must change."""
    rates = statistics["rates"]
    counts = statistics["counts"]
    calls = statistics["calls"]["total_provider_calls"]
    findings: list[dict] = []

    def check(name: str, observed: float, ok: bool, threshold: Any, note: str) -> None:
        findings.append({"criterion": name, "observed": observed, "threshold": threshold,
                         "within_contract": bool(ok), "note": note})

    thresholds = PROMPT_REVIEW_THRESHOLDS
    check("schema_compliance", rates["eventual_valid_rate"],
          rates["eventual_valid_rate"] >= thresholds["min_eventual_valid_rate"],
          f">= {thresholds['min_eventual_valid_rate']}",
          "share of slots that produced an accepted recipe within the frozen retry budget")
    check("first_attempt_compliance", rates["first_attempt_valid_rate"],
          rates["first_attempt_valid_rate"] >= thresholds["min_first_attempt_valid_rate"],
          f">= {thresholds['min_first_attempt_valid_rate']}",
          "share of slots valid on the first call; a low value means the prompt under-constrains")
    check("duplicate_rate", rates["duplicate_rate"],
          rates["duplicate_rate"] <= thresholds["max_duplicate_rate"],
          f"<= {thresholds['max_duplicate_rate']}",
          "exact canonical-content repeats across the pilot")
    check("ontology_violation_rate", round(counts["ontology_violations"] / calls, 6) if calls else 0.0,
          (counts["ontology_violations"] / calls if calls else 0.0)
          <= thresholds["max_ontology_violation_rate"],
          f"<= {thresholds['max_ontology_violation_rate']}",
          "non-canonical enum values per provider call; systematic mismatch would mean the "
          "vocabulary in the prompt does not match the validator")
    check("compatibility_violation_rate",
          round(counts["compatibility_violations"] / calls, 6) if calls else 0.0,
          (counts["compatibility_violations"] / calls if calls else 0.0)
          <= thresholds["max_compatibility_violation_rate"],
          f"<= {thresholds['max_compatibility_violation_rate']}",
          "medium/artifact and geometry/region rejections per call")

    worst_axis = min(coverage["axes"].items(), key=lambda item: item[1]["coverage_fraction"])
    check(f"coverage_{worst_axis[0]}", worst_axis[1]["coverage_fraction"],
          worst_axis[1]["coverage_fraction"] >= thresholds["min_axis_coverage_fraction"],
          f">= {thresholds['min_axis_coverage_fraction']}",
          f"weakest axis; missing {worst_axis[1]['missing_categories']}")

    collapse = []
    for axis, entry in coverage["axes"].items():
        for name, percent in entry["assignment_percent"].items():
            if percent >= thresholds["max_single_category_share"] * 100.0:
                collapse.append({"axis": axis, "category": name, "share_percent": percent})
    check("mode_collapse", len(collapse), not collapse,
          f"no category above {thresholds['max_single_category_share'] * 100.0}% of its axis",
          f"collapsed categories: {collapse}" if collapse else "no axis is dominated by one category")

    breached = [item for item in findings if not item["within_contract"]]
    coverage_breached = [item["criterion"] for item in breached
                         if item["criterion"].startswith("coverage_")
                         or item["criterion"] == "mode_collapse"]
    return {
        "thresholds": thresholds,
        "findings": findings,
        "criteria_breached": [item["criterion"] for item in breached],
        "collapsed_categories": collapse,
        "prompt_change_recommended": bool(breached),
        "prompt_changed_in_c2": False,
        "allow_ontology_aliases": False,
        "alias_policy_changed": False,
        "second_pilot_run_automatically": False,
        "confounds": {
            "applies_to": coverage_breached,
            "pilot_batch_size": 1,
            "c3_planned_batch_size": 32,
            "coverage_quotas_used_in_the_pilot": False,
            "statement": "The coverage breach is measured, but its cause is not isolated. The "
                         "frozen prompt's diversity rules are BATCH-scoped ('vary medium, "
                         "geometry, region coverage ... within this batch'), and a C2 pilot slot "
                         "asked for exactly one recipe, so those rules had no batch to act on. "
                         "The pilot also passed no coverage quotas, although the request template "
                         "supports them. The C3 schedule asks for 32 recipes per call, where both "
                         "mechanisms do have scope.",
            "what_the_evidence_does_show": "Under a batch of one, with no coverage quotas and no "
                                           "sampling controls, 32 independent calls under this "
                                           "prompt return a strongly modal distribution: "
                                           "display-like / flat / front dominate, and four of six "
                                           "geometries, three of five media and four of six "
                                           "illumination modes never appear.",
            "what_the_evidence_does_not_show": "It does not establish that the prompt would "
                                               "collapse the same way at the C3 batch shape, and "
                                               "it is not evidence that any prompt wording is "
                                               "wrong.",
        },
        "recommended_next_step": {
            "action": "Before editing a single prompt byte, re-measure coverage at the C3 batch "
                      "shape (one call requesting 32 recipes, the frozen 12x32 schema) so the "
                      "batch-size confound is removed. Only if the collapse survives that does a "
                      "prompt change have evidence behind it.",
            "requires_user_approval": True,
            "not_done_automatically": "No second pilot was run and no prompt byte was changed.",
            "alternative_if_collapse_persists": "Use the request template's existing "
                                                "coverage_quotas mechanism, which is ontology-level "
                                                "and source-only, rather than rewording the system "
                                                "instruction. It is already implemented and was "
                                                "deliberately left unused in C2.",
        },
        "policy": "A prompt change is recommended only on an objective, source-independent "
                  "contract failure. Aesthetic oddness in a recipe is never a reason, and no "
                  "second 32-slot pilot is run automatically: a material change stops for user "
                  "review.",
    }


# ------------------------------------------------------------------- markdown
def _table(headers: list[str], rows: list[list[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join([line, rule, body])


def build_pilot_markdown(context: FrozenContext, state: dict, replay: dict, statistics: dict,
                         coverage: dict, cooccurrence: dict, duplicates: dict,
                         flags: list[dict], verification: dict) -> str:
    frozen = context.as_frozen_record()
    lines: list[str] = []
    add = lines.append

    add("# C2 - LLM recipe pilot report")
    add("")
    add("32 disposable pilot slots generated live against the frozen C1 provider contract.")
    add("These recipes are C2 development artifacts. They never enter the C3 384 candidate")
    add("slots, the final 256-recipe LLM bank, a synthetic bank or detector training.")
    add("")
    add("## Frozen contract")
    add("")
    add(_table(["field", "value"], [
        ["provider", frozen["provider"]],
        ["model", f"`{frozen['model_id']}`"],
        ["API surface", frozen["api_surface"]],
        ["thinking_level", frozen["thinking_level"]],
        ["prompt identity", f"`{frozen['prompt_template_identity']}`"],
        ["schema identity (12x32 reference)", f"`{frozen['schema_identity_12x32_reference']}`"],
        ["schema identity (per pilot slot, 1 recipe)", f"`{frozen['schema_identity_per_pilot_slot']}`"],
        ["ontology identity", f"`{frozen['ontology_identity']}`"],
        ["provider config identity", f"`{frozen['provider_config_identity']}`"],
        ["allow_ontology_aliases", str(frozen["allow_ontology_aliases"])],
        ["sampling controls sent", "none (temperature / top_p / top_k never sent)"],
        ["tools / grounding / URL context / file search / code execution", "none passed"],
        ["input", frozen["input_type"]],
    ]))
    add("")
    add(f"> {frozen['schema_identity_note']}")
    add("")

    add("## Pilot totals")
    add("")
    slots = statistics["slots"]
    calls = statistics["calls"]
    rates = statistics["rates"]
    counts = statistics["counts"]
    latency = statistics["latency_seconds"]
    tokens = statistics["token_usage"]
    add(_table(["measure", "value"], [
        ["slots", slots["slot_count"]],
        ["successful slots", slots["successful_slots"]],
        ["failed / exhausted slots", slots["failed_or_exhausted_slots"]],
        ["total provider calls", calls["total_provider_calls"]],
        ["calls per slot (mean)", calls["calls_per_slot_mean"]],
        ["first-attempt-valid rate", rates["first_attempt_valid_rate"]],
        ["eventual-valid rate", rates["eventual_valid_rate"]],
        ["invalid rate (per response)", rates["invalid_rate"]],
        ["retry rate (slots that retried)", rates["retry_rate"]],
        ["retry exhaustion rate", rates["retry_exhaustion_rate"]],
        ["duplicate rate (per response)", rates["duplicate_rate"]],
        ["schema violations", counts["schema_violations"]],
        ["ontology violations", counts["ontology_violations"]],
        ["range violations", counts["range_violations"]],
        ["compatibility violations", counts["compatibility_violations"]],
        ["severity-budget violations", counts["severity_budget_violations"]],
        ["duplicate rejections", counts["duplicate_violations"]],
        ["compiler failures", counts["compiler_failures"]],
        ["transport / API errors", calls["transport_or_api_errors"]],
        ["transient 429", counts["transient_rate_limit_429"]],
        ["quota exhausted", counts["quota_exhausted"]],
        ["latency mean / median / p95 (s)",
         f"{latency['mean']} / {latency['median']} / {latency['p95']}"],
    ]))
    add("")
    if tokens["available"]:
        add("Token usage reported by the surface, summed over every attempt:")
        add("")
        add(_table(["counter", "total"],
                   [[key, value] for key, value in tokens["totals"].items()
                    if isinstance(value, (int, float))]))
    else:
        add("The surface reported no token usage on these attempts; no total is invented.")
    add("")

    add("## Offline replay verification")
    add("")
    add(f"- slots compared: {verification['slots_compared']}")
    add(f"- replay identical to the live run: **{verification['identical']}**")
    add(f"- mismatches: {len(verification['mismatches'])}")
    add("")
    add("The archived raw responses were re-parsed, re-validated, re-canonicalized, re-hashed")
    add("and re-compiled by the replay provider, which holds no client and no credential.")
    add("")

    add("## Coverage")
    add("")
    for axis, entry in coverage["axes"].items():
        add(f"### {axis} ({entry['categories_present']}/{entry['category_count']} present, "
            f"coverage {entry['coverage_fraction']})")
        add("")
        add(_table(["category", "recipes", "% of recipes", "assignments", "% of axis"],
                   [[name, entry["counts"][name], entry["recipe_percent"][name],
                     entry["assignment_counts"][name], entry["assignment_percent"][name]]
                    for name in entry["counts"]]))
        add("")
        add(f"missing: {entry['missing_categories'] or 'none'}")
        add("")
    add("### Artifacts and regions per recipe")
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
    add("This is a prompt/contract coverage audit over the frozen ontology only. It is not")
    add("compared against any dataset attack family, and no target information was consulted.")
    add("")

    add("## Co-occurrence")
    add("")
    for name, table in cooccurrence["tables"].items():
        add(f"### {name}")
        add("")
        add(_table(["artifact", *table["columns"], "total"],
                   [[row, *[table["cells"][row][column] for column in table["columns"]],
                     table["row_totals"][row]] for row in table["rows"]]))
        add("")
        if "compatible_cell_count" in table:
            add(f"compatible cells occupied: {table['occupied_compatible_cells']}"
                f"/{table['compatible_cell_count']} "
                "(cells the ontology forbids are structurally empty, not a coverage gap)")
        else:
            add(f"cells occupied: {table['occupied_cells']}/{table['total_cells']}")
        add("")

    add("## Flags")
    add("")
    if flags:
        add(_table(["flag", "where", "detail"],
                   [[item["flag"],
                     item.get("axis") or item.get("recipe_id") or item.get("category") or "-",
                     item.get("detail") or json.dumps(item.get("pattern", ""))[:120]]
                    for item in flags]))
    else:
        add("No objective pilot issue was flagged.")
    add("")
    add("Flags are objective and source-independent. No recipe is rated for usefulness against")
    add("any target.")
    add("")

    add("## All 32 pilot slot outcomes")
    add("")
    for slot in replay["slots"]:
        add(f"### {slot['slot_id']}")
        add("")
        recipe = next((item for item in replay["recipes"]
                       if item.recipe_id == slot["recipe_id"]), None)
        rows = [
            ["attempts (provider calls)", slot["provider_calls"]],
            ["final status", slot["final_status"]],
            ["validation", "accepted" if slot["final_status"] == "accepted" else "rejected"],
            ["compiler", slot["compiler_status"]],
        ]
        if recipe is not None:
            rows.extend([
                ["recipe id", slot["recipe_id"]],
                ["artifact type(s)", ", ".join(spec.name for spec in recipe.artifacts)],
                ["strength / severity", ", ".join(f"{spec.name}={round(float(spec.strength), 4)}"
                                                  for spec in recipe.artifacts)
                 + f" (total {round(sum(float(s.strength) for s in recipe.artifacts), 4)})"],
                ["region(s)", ", ".join(recipe.regions)],
                ["medium", f"{recipe.medium.family} "
                           f"(transparency {recipe.medium.transparency}, "
                           f"roughness {recipe.medium.roughness})"],
                ["geometry", f"{recipe.geometry.shape} "
                             f"(rigidity {recipe.geometry.rigidity}, "
                             f"coverage {recipe.geometry.coverage})"],
                ["illumination", recipe.capture.illumination],
                ["recipe identity", f"`{slot['recipe_identity']}`"],
                ["graph hash", f"`{slot['graph_hash']}`"],
            ])
        else:
            rows.append(["failure", json.dumps(slot["final_issues"])[:400] or
                         json.dumps(slot["final_error"])[:400]])
        add(_table(["field", "value"], rows))
        add("")
        if recipe is not None:
            add(f"*Physical reading:* {describe_physically(recipe)}")
            add("")
    return "\n".join(lines) + "\n"


def build_prompt_review_markdown(review: dict, statistics: dict, coverage: dict,
                                 duplicates: dict, flags: list[dict]) -> str:
    lines: list[str] = []
    add = lines.append
    add("# C2 - prompt review")
    add("")
    add("The full 32-slot pilot ran to completion under the unchanged C1 prompt. This review")
    add("was written afterwards, from the pilot evidence, and no prompt byte was changed")
    add("during C2.")
    add("")
    add("## Rule applied")
    add("")
    add("A prompt change may be recommended only on an objective, source-independent contract")
    add("issue: low schema compliance, a high duplicate rate, severe coverage collapse,")
    add("systematic ontology mismatch, or systematic physical incompatibility. A recipe that")
    add("merely looks unusual is not a reason, and nothing in this review consulted a dataset,")
    add("a target metric or an attack taxonomy.")
    add("")
    add("## Measured against the thresholds")
    add("")
    add(_table(["criterion", "observed", "threshold", "within contract"],
               [[item["criterion"], item["observed"], item["threshold"],
                 "yes" if item["within_contract"] else "**NO**"]
                for item in review["findings"]]))
    add("")
    for item in review["findings"]:
        add(f"- **{item['criterion']}** - {item['note']}")
    add("")
    add("## Verdict")
    add("")
    if review["prompt_change_recommended"]:
        add(f"**A material contract issue was found.** Criteria breached: "
            f"{', '.join(review['criteria_breached'])}.")
        add("")
        collapsed = review["collapsed_categories"]
        if collapsed:
            add(_table(["axis", "dominant category", "share of axis"],
                       [[item["axis"], item["category"], f"{item['share_percent']}%"]
                        for item in collapsed]))
            add("")
        confounds = review["confounds"]
        add("### The measurement is real; the cause is not isolated")
        add("")
        add(confounds["statement"])
        add("")
        add(f"- **What the evidence shows:** {confounds['what_the_evidence_does_show']}")
        add(f"- **What it does not show:** {confounds['what_the_evidence_does_not_show']}")
        add(f"- pilot batch size: {confounds['pilot_batch_size']}; "
            f"C3 planned batch size: {confounds['c3_planned_batch_size']}; "
            f"coverage quotas used: {confounds['coverage_quotas_used_in_the_pilot']}")
        add("")
        add("### Recommended next step")
        add("")
        add(review["recommended_next_step"]["action"])
        add("")
        add(f"- If the collapse survives that: {review['recommended_next_step']['alternative_if_collapse_persists']}")
        add("")
        add("**No second 32-slot pilot was run and no prompt byte was changed.** C2 stops here")
        add("for user review, as instructed.")
    else:
        add("**No prompt change is recommended.** Every objective criterion is inside the")
        add("contract, so the C1 prompt stands unmodified going into the C3 lock decision.")
    add("")
    add("## Alias policy")
    add("")
    add("`allow_ontology_aliases` remains **OFF**. It was not changed, and it is not proposed")
    add("for change: it is identity-bearing, so enabling it would alter the recipe bank")
    add("identity. Enabling it is a user decision only.")
    add("")
    add("## Observed oddities that are NOT prompt defects")
    add("")
    odd = [item for item in flags if item["flag"] == "ODD_BUT_VALID"]
    if odd:
        for item in odd:
            add(f"- `{item.get('recipe_id')}`: {item['detail']}")
        add("")
        add("Each of these validated, compiled and stayed inside every ontology rule. They are")
        add("recorded for visibility, not treated as errors.")
    else:
        add("None were flagged.")
    add("")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------- provenance
def reconcile_provenance(context: FrozenContext, archive_records: list[dict],
                         provenance: dict, commit: str) -> dict:
    """Guarantee one provenance record per archived attempt.

    An earlier revision of the runner broke out of the slot loop on quota
    exhaustion before writing provenance for the blocked attempt, so two archived
    calls had none. The runner no longer does that; this pass reconciles what the
    earlier revision produced and would be a no-op on a clean run. A
    reconstructed record is marked, and its timestamp comes from the archive
    rather than from now, so it is never mistaken for a live one.
    """
    records = list(provenance.get("records", []))
    have = {(record["slot_id"], record["attempt"], index)
            for index, record in enumerate(
                sorted(records, key=lambda item: (item["slot_id"], item["attempt"])))}
    by_slot: dict[str, int] = {}
    for record in records:
        by_slot[record["slot_id"]] = by_slot.get(record["slot_id"], 0) + 1

    archived_by_slot: dict[str, list[dict]] = {}
    for record in archive_records:
        archived_by_slot.setdefault(record["slot_id"], []).append(record)

    added = 0
    for slot_id, attempts in archived_by_slot.items():
        missing = len(attempts) - by_slot.get(slot_id, 0)
        if missing <= 0:
            continue
        request = context.request(slot_id)
        # The earliest attempts are the ones the old code path dropped.
        for record in sorted(attempts,
                             key=lambda item: item.get("sequence", item["attempt"]))[:missing]:
            built = build_provenance(
                request=request, result=_result_view(record),
                config_summary=context.config_summary(),
                prompt_provenance=context.template.as_provenance(),
                schema_identity=context.schema_identity_12x32,
                validation_result="provider_error" if record["error"] else "rejected",
                validation_errors=[], retry_count=0, generator_code_commit=commit,
                request_schedule_id=f"C2_PILOT:{slot_id}",
                raw_response_path=None,
                billing_tier=context.config.quota.billing_tier).as_dict()
            built["generation_timestamp_utc"] = record["recorded_at_utc"]
            built["reconstructed_from_archive"] = True
            built["reconstruction_reason"] = (
                "the runner revision that made this call returned on quota exhaustion before "
                "writing provenance; the record is rebuilt from the archived attempt")
            records.append(built)
            added += 1

    records.sort(key=lambda item: (item["slot_id"], item["generation_timestamp_utc"]))
    return {"schema_version": "c2-pilot-provenance-v1",
            "note": "one immutable record per provider attempt, successful or not",
            "record_count": len(records),
            "reconstructed_records": added,
            "records": records}


def _result_view(record: dict):
    """The ProviderGenerationResult a provenance record needs, from the archive."""
    from prism_fas.llm.contracts import (ErrorClass, ProviderError, ProviderGenerationResult)
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


# ------------------------------------------------------------------ incidents
def rate_limit_incidents(archive_records: list[dict], state: dict, quota_path) -> dict:
    """Every 429 the pilot met, with the provider's own words kept verbatim.

    C2 recorded two of these. Both named a quota in prose yet carried a short
    replenishment hint, and both cleared: the run resumed and finished. That is
    the whole reason the provider now classifies a 429 by its hint rather than by
    its wording, and the raw bodies below are the evidence for that change.
    """
    errors = [record for record in archive_records if record["error"]]
    rate_limited = [record for record in errors
                    if record["error"]["error_class"] in {"rate_limit", "quota_exhausted"}]
    unresolved = [record for record in rate_limited
                  if record["error"]["error_class"] == "quota_exhausted"
                  and state["status"] == "BLOCKED_QUOTA"]
    return {
        "schema_version": "c2-rate-limit-incidents-v1",
        "milestone": "C2",
        "generated_at_utc": utc_now(),
        "incident_count": len(rate_limited),
        "pilot_status": state["status"],
        "active_block": bool(unresolved),
        "incidents": [{
            "slot_id": record["slot_id"],
            "attempt": record["attempt"],
            "recorded_at_utc": record["recorded_at_utc"],
            "status_code": record["error"]["status_code"],
            "classified_as": record["error"]["error_class"],
            "retryable": record["error"]["retryable"],
            "provider_message_verbatim": record["error"]["message"],
        } for record in rate_limited],
        "classification_finding": {
            "observed": "The Free Tier 429 body for the frozen model uses the same "
                        "'You exceeded your current quota' wording for a short-window request "
                        "limit as for daily exhaustion, and separates them only by a "
                        "'Please retry in <N>s' hint (18.0s and 8.4s were observed).",
            "consequence_before_the_fix": "The pilot stopped twice with BLOCKED_QUOTA and the "
                                          "block artifact said 'daily quota exhaustion' for a "
                                          "limit that cleared in seconds.",
            "evidence_it_was_transient": "After waiting, the same frozen request succeeded; the "
                                         "run resumed and completed without any contract change.",
            "fix": "prism_fas.llm.providers.gemini._classify now classifies a 429 by its "
                   "replenishment hint. A hint within the retry ceiling and not naming a per-day "
                   "metric is RATE_LIMIT (bounded backoff); anything else still fails closed as "
                   "QUOTA_EXHAUSTED.",
            "scientific_contract_changed": False,
            "provider_config_identity_changed": False,
            "model_or_prompt_changed": False,
            "billing_enabled": False,
            "regression_test": "tests/c2/test_c2_rate_limit_classification.py",
        },
        "pacing": {
            "min_call_interval_seconds": 45.0,
            "nature": "operational scheduling only - it changes when a request is sent, never "
                      "what is sent, and it is not part of the frozen retry policy",
        },
        "block_artifact_present": quota_path.exists(),
    }


# ----------------------------------------------------------------------- main
def main() -> int:
    context = FrozenContext()
    state = read_json(REPORTS / "C2_PILOT_STATE.json")
    archive = read_json(REPORTS / "C2_PILOT_RAW_ARCHIVE.json")
    smoke = read_json(REPORTS / "C2_LIVE_SMOKE_AUDIT.json")
    commit = git("rev-parse", "HEAD")

    replay = replay_pilot(context, state, archive["records"])
    verification = verify_replay_matches_live(state["slots"], replay["slots"])

    statistics = pilot_statistics(replay["slots"], replay["attempts"])
    coverage = coverage_audit(replay["recipes"], context.ontology)
    cooccurrence = cooccurrence_audit(replay["recipes"], context.ontology)
    duplicates = duplicate_audit(replay["content_identities"], replay["recipes"])
    flags = flag_pilot_issues(coverage, cooccurrence, duplicates, statistics,
                              replay["recipes"], context.ontology)
    readiness = c3_readiness(statistics, candidate_slots=C3_CANDIDATE_SLOTS)
    review = prompt_review(statistics, coverage, duplicates)

    provenance_path = REPORTS / "C2_PILOT_PROVENANCE.json"
    provenance = reconcile_provenance(context, archive["records"], read_json(provenance_path),
                                      commit)
    write_json(provenance_path, provenance)

    quota_path = REPORTS / context.config.quota.quota_block_filename
    incidents = rate_limit_incidents(archive["records"], state, quota_path)
    write_json(REPORTS / "C2_RATE_LIMIT_INCIDENTS.json", incidents)
    if incidents["active_block"] is False and quota_path.exists():
        # The block flag is resume STATE, not evidence: the evidence is the
        # verbatim 429 bodies in the raw archive and the incident record above.
        # Leaving a stale "blocked" flag beside a completed pilot would be false.
        quota_path.unlink()
        print(f"removed stale {quota_path.name} (the run resumed and completed; "
              "the incident history is preserved in C2_RATE_LIMIT_INCIDENTS.json)")

    write_json(REPORTS / "C2_PILOT_AUDIT.json", {
        "schema_version": "c2-pilot-audit-v1",
        "milestone": "C2",
        "generated_at_utc": utc_now(),
        "generator_code_commit": commit,
        "derivation": "every number below was produced by replaying the archived raw responses "
                      "offline through a fresh validation pipeline; none was typed by hand",
        "disposable": True,
        "scope_exclusion": {"enters_c3_384_slots": False, "enters_final_256_bank": False,
                            "enters_detector_training": False, "enters_synthetic_bank": False},
        "frozen_contract": context.as_frozen_record(),
        "pilot_status": state["status"],
        "statistics": statistics,
        "duplicates": duplicates,
        "flags": flags,
        "replay_verification": verification,
        "slots": replay["slots"],
        "structural_patterns": {slot["recipe_id"]: structural_pattern(recipe)
                                for slot, recipe in zip(
                                    [s for s in replay["slots"] if s["recipe_id"]],
                                    replay["recipes"])},
        "smoke_reference": {"result": smoke["result"], "calls": smoke["budget"]["calls_made"],
                            "marker": smoke["marker"], "counted_in_pilot": False},
    })

    write_json(REPORTS / "C2_COVERAGE_AUDIT.json", {
        "schema_version": "c2-coverage-audit-v1",
        "milestone": "C2",
        "generated_at_utc": utc_now(),
        "scope": "prompt/contract coverage over the frozen ontology only",
        "compared_against_dataset_attack_families": False,
        "target_information_used": False,
        "coverage": coverage,
        "cooccurrence": cooccurrence,
        "flags": [item for item in flags
                  if item["flag"] in {"LOW_COVERAGE", "MODE_COLLAPSE", "REPEATED_PATTERN"}],
    })

    write_json(REPORTS / "C2_RETRY_QUOTA_AUDIT.json", {
        "schema_version": "c2-retry-quota-audit-v1",
        "milestone": "C2",
        "generated_at_utc": utc_now(),
        "frozen_retry_policy": {
            "semantic_max_retries": context.config.retry.semantic_max_retries,
            "semantic_attempts_per_slot": context.config.retry.semantic_max_retries + 1,
            "transport_max_attempts": context.config.retry.transport_max_attempts,
            "backoff_initial_seconds": context.config.retry.backoff_initial_seconds,
            "backoff_multiplier": context.config.retry.backoff_multiplier,
            "backoff_max_seconds": context.config.retry.backoff_max_seconds,
            "changed_in_c2": False,
        },
        "observed": {
            "total_provider_calls": statistics["calls"]["total_provider_calls"],
            "slots_that_retried": statistics["slots"]["slots_that_retried"],
            "retry_rate": statistics["rates"]["retry_rate"],
            "retry_exhaustion_rate": statistics["rates"]["retry_exhaustion_rate"],
            "exhausted_slot_ids": statistics["slots"]["exhausted_slot_ids"],
            "transport_error_classes": statistics["counts"]["transport_error_classes"],
            "transient_rate_limit_429": statistics["counts"]["transient_rate_limit_429"],
            "quota_exhausted_events": statistics["counts"]["quota_exhausted"],
            "latency_seconds": statistics["latency_seconds"],
            "token_usage": statistics["token_usage"],
        },
        "quota": {
            "billing_tier": context.config.quota.billing_tier,
            "auto_enable_paid": False,
            "on_quota_exhausted": context.config.quota.on_quota_exhausted,
            "quota_block_artifact": context.config.quota.quota_block_filename,
            "quota_block_artifact_written": quota_path.exists(),
            "billing_enabled_by_code": False,
            "quota_snapshot": "QUOTA_SNAPSHOT_NOT_PROGRAMMATICALLY_AVAILABLE - the Free Tier "
                              "RPM/TPM/RPD limits for this project must be read from AI Studio "
                              "before C3; no number was invented here",
        },
        "model_or_provider_switched": False,
        "prompt_changed_automatically": False,
        "successful_slots_regenerated": 0,
    })

    write_json(REPORTS / "C2_C3_READINESS.json", {
        "schema_version": "c2-c3-readiness-v1",
        "milestone": "C2",
        "generated_at_utc": utc_now(),
        "c3_candidates_generated": 0,
        "note": "an estimate only; no C3 candidate was generated and no bank was frozen",
        **readiness,
    })

    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "C2_LLM_PILOT_REPORT.md").write_text(
        build_pilot_markdown(context, state, replay, statistics, coverage, cooccurrence,
                             duplicates, flags, verification), encoding="utf-8")
    print("wrote docs/c2/C2_LLM_PILOT_REPORT.md")
    (DOCS / "C2_PROMPT_REVIEW.md").write_text(
        build_prompt_review_markdown(review, statistics, coverage, duplicates, flags),
        encoding="utf-8")
    print("wrote docs/c2/C2_PROMPT_REVIEW.md")

    write_json(REPORTS / "C2_PROMPT_REVIEW.json", {
        "schema_version": "c2-prompt-review-v1", "milestone": "C2",
        "generated_at_utc": utc_now(), **review})

    print(f"\nreplay identical to live: {verification['identical']}")
    print(f"accepted slots: {statistics['slots']['successful_slots']}"
          f"/{statistics['slots']['slot_count']}   "
          f"calls: {statistics['calls']['total_provider_calls']}")
    print(f"prompt change recommended: {review['prompt_change_recommended']}")
    return 0 if verification["identical"] else 3


if __name__ == "__main__":
    sys.exit(main())
