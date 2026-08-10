"""The archived C2C batch must reproduce itself offline, and every accepted
recipe must compile.

The acceptance rule C2C exists to establish is narrow and absolute: an accepted
scientific recipe is a compilable one. These tests pin it against the real
archived batch rather than a fixture.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from prism_fas.llm.coverage_quotas import evaluate
from prism_fas.llm.pilot_audit import coverage_audit, duplicate_audit
from prism_fas.llm.pipeline import RecipePlanner
from prism_fas.llm.providers.replay import ReplayArchive, ReplayRecipeProvider
from prism_fas.llm.route_policy import audit as route_audit
from prism_fas.recipes.canonical import canonical_json, recipe_hash
from prism_fas.recipes.compile import compile_recipe
from prism_fas.recipes.validate import validate_payload

from c2c_constants import (BATCH_SIZE, C2C_BANK_ID, REQUIRED_ROUTE, ROUTE_POLICY_IDENTITY,
                           FROZEN_BATCH_ENVELOPE_IDENTITY, FROZEN_ITEM_SCHEMA_IDENTITY,
                           C2C_SYSTEM_PROMPT_IDENTITY)

REPLAY_KEYS = ("slot_id", "attempt", "raw_text", "provider", "model_id", "model_version",
               "finish_reason", "usage", "provider_request_id", "provider_seed", "sdk_version",
               "api_surface", "request_sha256")


def accepted_recipes(c2c_state, ontology):
    recipes = []
    for row in c2c_state["recipes"]:
        if row["status"] != "accepted":
            continue
        recipe, issues = validate_payload(json.loads(row["canonical_recipe"]), ontology,
                                          canonicalize=False)
        assert recipe is not None and issues == []
        recipes.append(recipe)
    return recipes


# ------------------------------------------------------------------ the batch
def test_exactly_one_logical_semantic_batch_was_executed(c2c_state, c2c_archive):
    assert c2c_state["logical_batch_id"] == "C2C_BATCH_000"
    assert c2c_state["logical_batches_executed"] == 1
    assert c2c_state["second_batch_issued"] is False
    assert c2c_state["provider_attempts"] == 1
    served = [record for record in c2c_archive["records"] if record["raw_text"] is not None]
    assert len(served) == 1, "more than one semantic response was archived"


def test_the_batch_returned_exactly_32_objects(c2c_state, c2c_archive):
    assert c2c_state["requested_objects"] == BATCH_SIZE
    assert c2c_state["returned_objects"] == BATCH_SIZE
    assert len(c2c_state["recipes"]) == BATCH_SIZE
    served = [record for record in c2c_archive["records"] if record["raw_text"] is not None]
    assert len(json.loads(served[-1]["raw_text"])["recipes"]) == BATCH_SIZE


def test_the_archived_response_hash_matches_its_bytes(c2c_archive):
    for record in c2c_archive["records"]:
        if record["raw_text"] is None:
            assert record["error"] is not None
            continue
        digest = hashlib.sha256(record["raw_text"].encode("utf-8")).hexdigest()
        assert record["raw_response_sha256"] == digest


def test_the_request_carried_the_route_policy_identity(c2c_archive):
    for record in c2c_archive["records"]:
        assert record["route_policy_identity"] == ROUTE_POLICY_IDENTITY


def test_every_returned_object_was_accepted_and_declares_the_contract_route(c2c_state):
    rows = c2c_state["recipes"]
    assert all(row["status"] == "accepted" for row in rows)
    assert all(row["generator_route"] == REQUIRED_ROUTE for row in rows)
    assert c2c_state["route_policy_rejections"] == 0


def test_zero_compiler_failures_among_accepted_recipes(c2c_state):
    """The C2C acceptance rule. Not weakened."""
    accepted = [row for row in c2c_state["recipes"] if row["status"] == "accepted"]
    failures = [row for row in accepted if row["compiler_status"] == "failed"]
    assert failures == [], f"accepted recipes that do not compile: {failures}"
    assert c2c_state["compiler_failures"] == 0
    assert c2c_state["compiled_objects"] == len(accepted)


def test_the_batch_replays_offline_to_identical_verdicts(c2c_archive, c2c_state, config,
                                                         ontology, route_policy, make_request):
    served = [record for record in c2c_archive["records"] if record["raw_text"] is not None]
    archive = ReplayArchive.from_records([{key: record[key] for key in REPLAY_KEYS}
                                          for record in served])
    request = make_request(policy=route_policy)
    outcomes = []
    for _pass in range(2):
        provider = ReplayRecipeProvider(archive, strict=False)
        planner = RecipePlanner(provider=provider, config=config, ontology=ontology,
                                sleep=lambda _s: None, route_policy=route_policy)
        result = provider.generate(request, attempt=served[-1]["attempt"])
        assert result.raw_text == served[-1]["raw_text"]
        validation = planner.validate_response(result.raw_text, slot_id="C2C_BATCH_000",
                                               recipes_requested=BATCH_SIZE)
        outcomes.append([(c.index, c.outcome.value, c.recipe_identity)
                         for c in validation.candidates])
        assert provider.describe()["network"] is False
    assert outcomes[0] == outcomes[1], "replaying the same bytes gave a different verdict"
    live = [(row["batch_index"], row["status"], row["canonical_identity"])
            for row in c2c_state["recipes"]]
    assert outcomes[0] == live, "the replay disagreed with the live run"


def test_every_accepted_recipe_revalidates_and_recompiles(c2c_state, ontology):
    rows = {row["recipe_id"]: row for row in c2c_state["recipes"] if row["recipe_id"]}
    recipes = accepted_recipes(c2c_state, ontology)
    assert len(recipes) == BATCH_SIZE
    for recipe in recipes:
        row = rows[recipe.recipe_id]
        assert recipe_hash(recipe) == row["canonical_identity"]
        assert canonical_json(recipe) == row["canonical_recipe"]
        first = compile_recipe(recipe, ontology, bank_id=C2C_BANK_ID)
        second = compile_recipe(recipe, ontology, bank_id=C2C_BANK_ID)
        assert first.graph_hash == second.graph_hash == row["graph_hash"]
        assert first.conditioning_dimension == 41
        assert first.region_mask_policy["policy"] == "parsing_first_geometry_fallback"


def test_the_batch_carries_no_duplicates(c2c_state, ontology):
    recipes = accepted_recipes(c2c_state, ontology)
    identities = [RecipePlanner.content_identity(recipe) for recipe in recipes]
    result = duplicate_audit(identities, recipes)
    assert result["exact_duplicate_groups"] == 0
    assert result["exact_duplicate_rate"] == 0.0


def test_the_route_audit_reports_full_compliance(c2c_state, ontology, route_policy):
    result = route_audit(route_policy, accepted_recipes(c2c_state, ontology))
    assert result["all_compliant"] is True
    assert result["violating_count"] == 0
    assert result["route_counts"] == {"physics+gpat": BATCH_SIZE}
    assert result["silent_repairs_performed"] == 0
    assert result["gpat_only_class_created"] is False


# ----------------------------------------------------------------- coverage
def test_coverage_is_deterministic_and_matches_the_artifact(c2c_state, ontology, c2c_coverage):
    recipes = accepted_recipes(c2c_state, ontology)
    fresh = coverage_audit(recipes, ontology)
    assert json.dumps(fresh, sort_keys=True) == json.dumps(c2c_coverage["coverage"],
                                                           sort_keys=True)


def test_quota_compliance_is_deterministic_and_matches_the_artifact(c2c_state, ontology,
                                                                    quotas, c2c_coverage):
    recipes = accepted_recipes(c2c_state, ontology)
    fresh = evaluate(quotas, recipes, ontology)
    assert json.dumps(fresh, sort_keys=True) == json.dumps(c2c_coverage["quota_compliance"],
                                                           sort_keys=True)


def test_every_axis_is_fully_covered(c2c_coverage):
    for axis, entry in c2c_coverage["quota_compliance"]["axes"].items():
        assert entry["categories_missing"] == [], f"{axis} is missing {entry['categories_missing']}"
        assert entry["required_pass"], f"{axis} failed a required quota bound"
        assert entry["max_share_percent"] <= 60.0, f"{axis} shows severe mode collapse"


def test_the_route_fix_did_not_damage_coverage(c2c_coverage):
    comparison = c2c_coverage["c2b_versus_c2c"]
    assert comparison["route_fix_damaged_coverage"] is False
    assert comparison["damaged_axes"] == []


def test_the_quota_values_were_not_changed_by_c2c(c2c_coverage):
    assert c2c_coverage["quota_values_changed_in_c2c"] is False
    assert (c2c_coverage["coverage_quotas"]["quota_identity"]
            == "89c3468436803c4d6187c716048117a4f4f02681c38d83c3885ce5ddbdb1ddd5")


# ------------------------------------------------------------------ contract
def test_the_contract_recorded_in_the_batch_is_the_frozen_one(c2c_state):
    contract = c2c_state["batch_contract"]
    assert contract["model_id"] == "gemini-3.6-flash"
    assert contract["api_surface"] == "interactions"
    assert contract["thinking_level"] == "medium"
    assert contract["allow_ontology_aliases"] is False
    assert contract["single_recipe_schema_identity"] == FROZEN_ITEM_SCHEMA_IDENTITY
    assert contract["batch_envelope_schema_identity"] == FROZEN_BATCH_ENVELOPE_IDENTITY
    assert contract["route_policy_identity"] == ROUTE_POLICY_IDENTITY
    assert contract["system_prompt_identity"] == C2C_SYSTEM_PROMPT_IDENTITY
    assert contract["single_recipe_schema_changed_in_c2c"] is False
    assert contract["coverage_quotas_changed_in_c2c"] is False


def test_the_c2b_replay_evidence_is_preserved_and_unaltered(c2b_archive):
    """C2B artifacts are read only; C2C must not have rewritten them."""
    from pathlib import Path

    replay_path = (Path(__file__).resolve().parents[2] / "reports" / "c2c"
                   / "C2C_C2B_REPLAY_AUDIT.json")
    if not replay_path.exists():
        pytest.skip("C2C_C2B_REPLAY_AUDIT.json missing; run scripts/c2c_replay_c2b.py")
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    assert replay["c2b_artifacts_modified"] is False
    assert replay["no_recipe_was_altered"] is True
    assert replay["silent_repairs_performed"] == 0
    assert replay["with_route_policy"]["compiler_failed"] == 0
    assert replay["without_route_policy_as_c2b_ran_it"]["compiler_failed"] == 10
    assert replay["network_calls"] == 0


def test_the_c2c_recipes_are_marked_disposable(c2c_state):
    assert c2c_state["disposable"] is True
    assert c2c_state["enters_c3"] is False
    assert c2c_state["enters_final_bank"] is False
