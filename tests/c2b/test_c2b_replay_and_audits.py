"""The archived C2B batch must reproduce itself offline, and every audit over it
must be deterministic.

A coverage number that moved between runs could not be cited in a freeze
decision, so determinism is checked directly rather than assumed.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from prism_fas.llm.coverage_quotas import (QuotaSpec, axis_values, classify_recipes, evaluate,
                                           parse_quota_spec)
from prism_fas.llm.pilot_audit import axis_pair_table, coverage_audit, duplicate_audit
from prism_fas.llm.pipeline import RecipePlanner
from prism_fas.llm.providers.replay import ReplayArchive, ReplayRecipeProvider
from prism_fas.recipes.canonical import canonical_json, recipe_hash
from prism_fas.recipes.compile import compile_recipe
from prism_fas.recipes.validate import validate_payload

from c2b_constants import BATCH_SIZE, C2B_BANK_ID

REPLAY_KEYS = ("slot_id", "attempt", "raw_text", "provider", "model_id", "model_version",
               "finish_reason", "usage", "provider_request_id", "provider_seed", "sdk_version",
               "api_surface", "request_sha256")


def accepted_recipes(batch_state, ontology):
    recipes = []
    for row in batch_state["recipes"]:
        if row["status"] != "accepted":
            continue
        recipe, issues = validate_payload(json.loads(row["canonical_recipe"]), ontology,
                                          canonicalize=False)
        assert recipe is not None and issues == []
        recipes.append(recipe)
    return recipes


# ------------------------------------------------------------------- replay
def test_exactly_one_logical_batch_was_executed(batch_state):
    assert batch_state["logical_batch_id"] == "C2B_BATCH_000"
    assert batch_state["logical_batches_executed"] == 1
    assert batch_state["second_batch_issued"] is False


def test_the_archive_carries_every_attempt_with_its_hash(batch_archive):
    assert batch_archive["record_count"] == len(batch_archive["records"])
    for record in batch_archive["records"]:
        if record["raw_text"] is None:
            assert record["error"] is not None
            continue
        digest = hashlib.sha256(record["raw_text"].encode("utf-8")).hexdigest()
        assert record["raw_response_sha256"] == digest


def test_the_raw_batch_replays_offline_and_deterministically(batch_archive, batch_state,
                                                             config, ontology,
                                                             make_batch_request):
    served = [record for record in batch_archive["records"] if record["raw_text"] is not None]
    if not served:
        pytest.skip("no archived response")
    archive = ReplayArchive.from_records([{key: record[key] for key in REPLAY_KEYS}
                                          for record in served])
    request = make_batch_request()

    outcomes = []
    for _pass in range(2):
        provider = ReplayRecipeProvider(archive, strict=False)
        planner = RecipePlanner(provider=provider, config=config, ontology=ontology,
                                sleep=lambda _seconds: None)
        result = provider.generate(request, attempt=served[-1]["attempt"])
        assert result.raw_text == served[-1]["raw_text"]
        validation = planner.validate_response(result.raw_text, slot_id="C2B_BATCH_000",
                                               recipes_requested=BATCH_SIZE)
        outcomes.append([(candidate.index, candidate.outcome.value, candidate.recipe_identity)
                         for candidate in validation.candidates])
    assert outcomes[0] == outcomes[1], "replaying the same bytes gave a different verdict"
    live = [(row["batch_index"], row["status"], row["canonical_identity"])
            for row in batch_state["recipes"]]
    assert outcomes[0] == live, "the replay disagreed with the live run"


def test_replay_performs_zero_network_calls(batch_archive, config, ontology, make_batch_request):
    served = [record for record in batch_archive["records"] if record["raw_text"] is not None]
    if not served:
        pytest.skip("no archived response")
    provider = ReplayRecipeProvider(
        ReplayArchive.from_records([{key: record[key] for key in REPLAY_KEYS}
                                    for record in served]), strict=False)
    provider.generate(make_batch_request(), attempt=served[-1]["attempt"])
    assert provider.describe()["network"] is False
    assert not any("client" in name.lower() or "key" in name.lower() for name in vars(provider))


def test_every_accepted_recipe_revalidates_independently(batch_state, ontology):
    recipes = accepted_recipes(batch_state, ontology)
    assert recipes, "the batch accepted no recipe"
    rows = {row["recipe_id"]: row for row in batch_state["recipes"] if row["recipe_id"]}
    for recipe in recipes:
        assert recipe_hash(recipe) == rows[recipe.recipe_id]["canonical_identity"]
        assert canonical_json(recipe) == rows[recipe.recipe_id]["canonical_recipe"]


def test_the_compiler_replays_deterministically(batch_state, ontology):
    compiled = [row for row in batch_state["recipes"] if row["compiler_status"] == "compiled"]
    assert compiled, "no accepted recipe reached the compiler"
    for row in compiled:
        recipe, issues = validate_payload(json.loads(row["canonical_recipe"]), ontology,
                                          canonicalize=False)
        assert recipe is not None and issues == []
        first = compile_recipe(recipe, ontology, bank_id=C2B_BANK_ID)
        second = compile_recipe(recipe, ontology, bank_id=C2B_BANK_ID)
        assert first.graph_hash == second.graph_hash == row["graph_hash"]
        assert first.conditioning_dimension == 41
        assert first.region_mask_policy["policy"] == "parsing_first_geometry_fallback"


def test_a_recipe_without_the_physics_route_is_valid_but_cannot_compile(batch_state, ontology):
    """The C2B finding, pinned as a test rather than left as prose.

    The validator and the compiler disagree: a gpat-only recipe passes every
    semantic rule and then has no operator graph to build.
    """
    failed = [row for row in batch_state["recipes"]
              if row["status"] == "accepted" and row["compiler_status"] == "failed"]
    if not failed:
        pytest.skip("this batch produced no gpat-only recipe")
    from prism_fas.recipes.compile import CompileError

    for row in failed:
        payload = json.loads(row["canonical_recipe"])
        assert "physics" not in payload["generator_route"]
        recipe, issues = validate_payload(payload, ontology, canonicalize=False)
        assert recipe is not None and issues == [], "the recipe is semantically valid"
        with pytest.raises(CompileError, match="physics route is required"):
            compile_recipe(recipe, ontology, bank_id=C2B_BANK_ID)


# -------------------------------------------------------------------- audits
def test_quota_evaluation_is_deterministic(batch_state, ontology, quotas):
    recipes = accepted_recipes(batch_state, ontology)
    first = evaluate(quotas, recipes, ontology)
    second = evaluate(quotas, recipes, ontology)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_quota_pass_fail_is_deterministic_and_matches_the_artifact(batch_state, ontology,
                                                                   quotas, coverage_artifact):
    recipes = accepted_recipes(batch_state, ontology)
    fresh = evaluate(quotas, recipes, ontology)
    committed = coverage_artifact["quota_compliance"]
    assert fresh["required_pass"] == committed["required_pass"]
    assert fresh["preferred_pass"] == committed["preferred_pass"]
    assert json.dumps(fresh, sort_keys=True) == json.dumps(committed, sort_keys=True)


def test_batch_coverage_is_deterministic_and_matches_the_artifact(batch_state, ontology,
                                                                  coverage_artifact):
    recipes = accepted_recipes(batch_state, ontology)
    fresh = coverage_audit(recipes, ontology)
    assert json.dumps(fresh, sort_keys=True) == json.dumps(coverage_artifact["coverage"],
                                                           sort_keys=True)


def test_cooccurrence_tables_are_deterministic(batch_state, ontology):
    recipes = accepted_recipes(batch_state, ontology)
    for row, column in (("artifacts", "media"), ("media", "geometry"),
                        ("geometry", "illumination")):
        first = axis_pair_table(recipes, ontology, row, column)
        second = axis_pair_table(recipes, ontology, row, column)
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_duplicate_detection_over_the_batch_is_deterministic(batch_state, ontology):
    recipes = accepted_recipes(batch_state, ontology)
    identities = [RecipePlanner.content_identity(recipe) for recipe in recipes]
    first = duplicate_audit(identities, recipes)
    second = duplicate_audit(identities, recipes)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    # The pipeline rejects a duplicate before acceptance, so an accepted batch
    # can carry no exact repeat.
    assert first["exact_duplicate_groups"] == 0


def test_quota_classification_never_moves_a_recipe(batch_state, ontology, quotas):
    recipes = accepted_recipes(batch_state, ontology)
    before = [axis_values(recipe) for recipe in recipes]
    result = classify_recipes(quotas, recipes, ontology)
    after = [axis_values(recipe) for recipe in recipes]
    assert before == after, "classification mutated a recipe"
    assert (result["valid_and_quota_compliant"] + result["valid_but_quota_miss"]
            == len(recipes))


def test_a_quota_spec_that_cannot_be_satisfied_is_refused(ontology):
    """5 media x 8 minimum = 40 recipes, which a batch of 32 cannot hold."""
    from prism_fas.llm.coverage_quotas import QuotaError

    spec = parse_quota_spec({
        "schema_version": "c2b-coverage-quota-v1", "batch_size": 32,
        "axes": [{"axis": "media", "require_all": True, "min_per_category": 8}]})
    with pytest.raises(QuotaError, match="unsatisfiable"):
        spec.validate_against(ontology)


def test_a_quota_naming_something_outside_the_ontology_is_refused(ontology):
    from prism_fas.llm.coverage_quotas import QuotaError

    spec = parse_quota_spec({
        "schema_version": "c2b-coverage-quota-v1", "batch_size": 32,
        "axes": [{"axis": "attack_family", "require_all": True}]})
    with pytest.raises(QuotaError, match="unknown quota axis"):
        spec.validate_against(ontology)


def test_an_unknown_quota_key_is_rejected_rather_than_ignored():
    from prism_fas.llm.coverage_quotas import QuotaError

    with pytest.raises(QuotaError, match="unknown axis quota keys"):
        parse_quota_spec({"schema_version": "c2b-coverage-quota-v1", "batch_size": 32,
                          "axes": [{"axis": "media", "min_per_recipe": 4}]})


def test_the_request_identity_changes_when_the_quotas_change(make_batch_request, quotas,
                                                             ontology):
    baseline = make_batch_request()
    tightened = QuotaSpec(
        batch_size=quotas.batch_size,
        axes=tuple(quota if quota.axis != "media"
                   else type(quota)(**{**quota.as_dict(), "min_per_category": 5})
                   for quota in quotas.axes),
        schema_version=quotas.schema_version, label=quotas.label,
        diversity_rules=quotas.diversity_rules)
    assert tightened.quota_identity != quotas.quota_identity
    changed = make_batch_request(quota_spec=tightened)
    assert changed.request_sha256 != baseline.request_sha256
    assert changed.input_text != baseline.input_text
