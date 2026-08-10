"""The batch envelope must demand exactly 32 objects, and the item schema must
not have moved.

The count is enforced on the RESPONSE, by the validator, not by the request-side
schema. That distinction is the whole reason C2B could drop the array length
bound the provider refuses without changing what "a valid batch" means.
"""
from __future__ import annotations

import json

import pytest

from prism_fas.llm.json_schema import (candidate_json_schema, candidate_object_schema,
                                       json_schema_identity)
from prism_fas.llm.pipeline import RecipePlanner

from c2b_constants import BATCH_SIZE, FROZEN_ITEM_SCHEMA_IDENTITY, FROZEN_ONTOLOGY_IDENTITY


def valid_candidate(seed: int = 1) -> dict:
    return {
        "schema_version": "1.1",
        "medium": {"family": "display-like", "roughness": 0.2, "transparency": 0.05},
        "geometry": {"shape": "flat", "rigidity": 0.9, "coverage": 0.6},
        "regions": ["left_eye", "right_eye"],
        "artifacts": [{"name": "pixel_grid", "strength": 0.3}],
        "capture": {"yaw": 5.0, "illumination": "front", "compression_q": 80,
                    "scale": 1.0, "motion": 0.05, "defocus": 0.1},
        "forbidden_shortcuts": [],
        "generator_route": ["physics"],
        "seed": seed,
    }


def envelope(count: int) -> str:
    return json.dumps({"recipes": [valid_candidate(1000 + index) for index in range(count)]})


def planner(config, ontology) -> RecipePlanner:
    from prism_fas.llm.providers.mock import MockRecipeProvider
    return RecipePlanner(provider=MockRecipeProvider(), config=config, ontology=ontology,
                         sleep=lambda _seconds: None)


def test_exactly_32_objects_is_accepted(config, ontology):
    validation = planner(config, ontology).validate_response(
        envelope(BATCH_SIZE), slot_id="C2B_BATCH_000", recipes_requested=BATCH_SIZE)
    assert validation.response_issues == []
    assert len(validation.candidates) == BATCH_SIZE
    assert validation.all_accepted


@pytest.mark.parametrize("count", [31, 33])
def test_a_batch_that_is_not_exactly_32_is_rejected(config, ontology, count):
    validation = planner(config, ontology).validate_response(
        envelope(count), slot_id="C2B_BATCH_000", recipes_requested=BATCH_SIZE)
    assert validation.response_issues, f"{count} objects were not rejected"
    assert any("expected exactly 32" in issue["reason"] for issue in validation.response_issues)
    assert not validation.all_accepted


def test_the_count_is_enforced_even_without_the_request_side_array_bound(config, ontology):
    """The provider refuses `maxItems: 32`, so the request cannot carry it. The
    requirement therefore has to hold on the response, and it does."""
    sent = candidate_json_schema(ontology, recipes_requested=BATCH_SIZE, array_bounds=False)
    assert "minItems" not in sent["properties"]["recipes"]
    assert "maxItems" not in sent["properties"]["recipes"]
    validation = planner(config, ontology).validate_response(
        envelope(30), slot_id="C2B_BATCH_000", recipes_requested=BATCH_SIZE)
    assert not validation.all_accepted
    assert any("expected exactly 32" in issue["reason"] for issue in validation.response_issues)


def test_the_item_schema_is_identical_in_every_envelope(ontology):
    item = candidate_object_schema(ontology)
    for size in (1, 2, BATCH_SIZE):
        for bounds in (True, False):
            envelope_schema = candidate_json_schema(ontology, recipes_requested=size,
                                                    array_bounds=bounds)
            assert envelope_schema["properties"]["recipes"]["items"] == item, (
                f"the item schema moved inside the {size}-object envelope (bounds={bounds})")


def test_the_single_recipe_schema_identity_is_unchanged(ontology):
    """C2B may change the envelope. It may not change recipe semantics."""
    assert json_schema_identity(candidate_object_schema(ontology)) == FROZEN_ITEM_SCHEMA_IDENTITY


def test_the_single_recipe_ontology_identity_is_unchanged(ontology):
    assert ontology.sha256 == FROZEN_ONTOLOGY_IDENTITY


def test_dropping_the_array_bound_changes_only_the_envelope_identity(ontology):
    bounded = candidate_json_schema(ontology, recipes_requested=BATCH_SIZE, array_bounds=True)
    unbounded = candidate_json_schema(ontology, recipes_requested=BATCH_SIZE, array_bounds=False)
    assert json_schema_identity(bounded) != json_schema_identity(unbounded)
    assert bounded["properties"]["recipes"]["items"] == unbounded["properties"]["recipes"]["items"]
    # The only difference is the two bound keys.
    assert (set(bounded["properties"]["recipes"]) - set(unbounded["properties"]["recipes"])
            == {"minItems", "maxItems"})


def test_the_envelope_still_forbids_unknown_top_level_keys(config, ontology):
    payload = json.dumps({"recipes": [valid_candidate()], "notes": "extra"})
    validation = planner(config, ontology).validate_response(
        payload, slot_id="C2B_BATCH_000", recipes_requested=1)
    assert any("unknown top-level keys" in issue["reason"]
               for issue in validation.response_issues)


def test_a_model_supplied_recipe_id_is_rejected_not_stripped(config, ontology):
    payload = json.dumps({"recipes": [{**valid_candidate(), "recipe_id": "R-000123"}]})
    validation = planner(config, ontology).validate_response(
        payload, slot_id="C2B_BATCH_000", recipes_requested=1)
    assert validation.candidates[0].outcome.value == "rejected_system_owned_field"


def test_duplicate_objects_inside_the_same_batch_are_detected(config, ontology):
    """Two identical recipes in one response: the second is a duplicate."""
    payload = json.dumps({"recipes": [valid_candidate(7), valid_candidate(7)]})
    validation = planner(config, ontology).validate_response(
        payload, slot_id="C2B_BATCH_000", recipes_requested=2)
    outcomes = [candidate.outcome.value for candidate in validation.candidates]
    assert outcomes[0] == "accepted"
    assert outcomes[1] == "rejected_duplicate"
    assert not validation.all_accepted
