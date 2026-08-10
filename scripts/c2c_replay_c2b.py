"""Replay the archived C2B batch under the new route policy. Offline, no network.

    python scripts/c2c_replay_c2b.py

C2B accepted 32 recipes and then failed to compile 10 of them. This script feeds
the very same archived bytes through the C2C pipeline and shows where those 10
now stop: at `SCIENTIFIC_ROUTE_POLICY`, before canonicalization, before duplicate
registration and before the compiler is ever called.

Nothing is altered. The 10 recipes are replayed exactly as the provider wrote
them; the point is that the pipeline now refuses them rather than the compiler
discovering the problem afterwards.

The counts are reported as observed. They are not assumed to be 22/10.
"""
from __future__ import annotations

import sys

from c2c_common import (BATCH_SIZE, C2B_BANK_ID, C2B_REPORTS, REPORTS, RouteContext, git,
                        read_json, utc_now, write_json)

from prism_fas.llm.pipeline import CandidateOutcome, RecipePlanner
from prism_fas.llm.providers.replay import ReplayArchive, ReplayRecipeProvider
from prism_fas.llm.route_policy import ROUTE_POLICY_VIOLATION
from prism_fas.recipes.compile import CompileError, compile_recipe

REPLAY_KEYS = ("slot_id", "attempt", "raw_text", "provider", "model_id", "model_version",
               "finish_reason", "usage", "provider_request_id", "provider_seed", "sdk_version",
               "api_surface", "request_sha256")


def c2b_request(context: RouteContext):
    """Rebuild the request C2B actually sent: pre-amendment prompt, no policy.

    The replay archive refuses to serve a response across a changed request
    identity, which is the point - the archived bytes belong to the C2B request,
    and only the VALIDATION is being re-run under the new policy.
    """
    from prism_fas.llm.contracts import GenerationRequest
    from prism_fas.llm.prompt import build_generation_prompt

    return GenerationRequest(
        slot_id="C2B_BATCH_000",
        system_instruction=context.template_before.system_instruction,
        input_text=build_generation_prompt(
            context.template_before, recipes_requested=BATCH_SIZE,
            coverage_quotas=context.quotas.prompt_block(context.ontology)),
        response_json_schema=context.batch_schema,
        model_id=context.config.model_id,
        thinking_level=context.config.thinking_level,
        response_mime_type=context.config.response_mime_type,
        max_output_tokens=context.config.max_output_tokens,
        recipes_requested=BATCH_SIZE,
        ontology_identity=context.ontology.sha256,
        prompt_template_identity=context.template_before.identity(),
        provider_config_identity=context.provider_config_identity,
        metadata={"phase": "c2b_batch_shape"},
    )


def main() -> int:
    context = RouteContext()
    archive_payload = read_json(C2B_REPORTS / "C2B_RAW_ARCHIVE.json")
    served = [record for record in archive_payload["records"] if record["raw_text"] is not None]
    if not served:
        print("no archived C2B response to replay")
        return 3

    archive = ReplayArchive.from_records([{key: record[key] for key in REPLAY_KEYS}
                                          for record in served])
    request = c2b_request(context)

    def run(policy) -> dict:
        provider = ReplayRecipeProvider(archive, strict=False)
        planner = RecipePlanner(provider=provider, config=context.config,
                                ontology=context.ontology, sleep=lambda _s: None,
                                route_policy=policy)
        result = provider.generate(request, attempt=served[-1]["attempt"])
        validation = planner.validate_response(result.raw_text or "", slot_id="C2B_BATCH_000",
                                               recipes_requested=BATCH_SIZE,
                                               next_recipe_index=0)
        rows = []
        for candidate in validation.candidates:
            row = {
                "batch_index": candidate.index,
                "outcome": candidate.outcome.value,
                "recipe_id": candidate.recipe.recipe_id if candidate.recipe else None,
                "generator_route": (list(candidate.recipe.generator_route)
                                    if candidate.recipe else None),
                "canonical_identity": candidate.recipe_identity,
                "issues": candidate.issues,
                "compiler_status": "not_attempted",
                "compiler_error": None,
            }
            # The compiler is only ever offered an ACCEPTED candidate, which is
            # exactly the property under test.
            if candidate.accepted and candidate.recipe is not None:
                try:
                    graph = compile_recipe(candidate.recipe, context.ontology,
                                           bank_id=C2B_BANK_ID)
                    row["compiler_status"] = "compiled"
                    row["graph_hash"] = graph.graph_hash
                except CompileError as exc:
                    row["compiler_status"] = "failed"
                    row["compiler_error"] = str(exc)
            rows.append(row)
        return {"rows": rows, "response_issues": validation.response_issues}

    before = run(None)
    after = run(context.route_policy)

    def summarise(result: dict) -> dict:
        rows = result["rows"]
        accepted = [row for row in rows if row["outcome"] == CandidateOutcome.ACCEPTED.value]
        return {
            "returned_objects": len(rows),
            "accepted": len(accepted),
            "rejected": len(rows) - len(accepted),
            "rejected_by_route_policy": sum(
                1 for row in rows
                if row["outcome"] == CandidateOutcome.REJECTED_ROUTE_POLICY.value),
            "compiler_attempted": sum(1 for row in rows
                                      if row["compiler_status"] != "not_attempted"),
            "compiler_compiled": sum(1 for row in rows if row["compiler_status"] == "compiled"),
            "compiler_failed": sum(1 for row in rows if row["compiler_status"] == "failed"),
        }

    before_summary = summarise(before)
    after_summary = summarise(after)

    route_rejected = [row for row in after["rows"]
                      if row["outcome"] == CandidateOutcome.REJECTED_ROUTE_POLICY.value]
    unchanged = all(
        b["generator_route"] == a["generator_route"]
        for b, a in zip(before["rows"], after["rows"]))

    write_json(REPORTS / "C2C_C2B_REPLAY_AUDIT.json", {
        "schema_version": "c2c-c2b-replay-audit-v1",
        "milestone": "C2C",
        "generated_at_utc": utc_now(),
        "generator_code_commit": git("rev-parse", "HEAD"),
        "purpose": "Replay the archived C2B 32-object response under the new scientific route "
                   "policy, to prove the validator/compiler mismatch is now caught BEFORE "
                   "compilation.",
        "network_calls": 0,
        "source_archive": "reports/c2b/C2B_RAW_ARCHIVE.json",
        "raw_response_sha256": served[-1]["raw_response_sha256"],
        "c2b_artifacts_modified": False,

        "route_policy": context.route_policy.as_dict(),

        "without_route_policy_as_c2b_ran_it": before_summary,
        "with_route_policy": after_summary,

        "observed_counts": {
            "accepted_and_compilable_under_policy": after_summary["compiler_compiled"],
            "rejected_as_route_policy_violation": after_summary["rejected_by_route_policy"],
            "compiler_failures_after_policy": after_summary["compiler_failed"],
            "note": "counts are reported as observed, not assumed",
        },

        "route_violating_recipes": [
            {"batch_index": row["batch_index"],
             "generator_route": row["generator_route"],
             "rejection_code": ROUTE_POLICY_VIOLATION,
             "reasons": [issue["reason"] for issue in row["issues"]],
             "compiler_ever_called": row["compiler_status"] != "not_attempted"}
            for row in route_rejected],

        "no_recipe_was_altered": unchanged,
        "silent_repairs_performed": 0,
        "rows_without_policy": before["rows"],
        "rows_with_policy": after["rows"],
    })

    print(f"\nC2B archive replayed offline under the C2C route policy")
    print(f"  as C2B ran it : accepted {before_summary['accepted']}, "
          f"compiler failures {before_summary['compiler_failed']}")
    print(f"  under policy  : accepted {after_summary['accepted']}, "
          f"route-policy rejections {after_summary['rejected_by_route_policy']}, "
          f"compiler failures {after_summary['compiler_failed']}")
    print(f"  no recipe altered: {unchanged}")
    return 0 if after_summary["compiler_failed"] == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
