"""The scientific route contract, enforced before anything can be accepted.

C2B produced 32 valid recipes of which 10 could not be compiled, purely because
they omitted the physics route. These tests pin the repair: exactly
`["physics", "gpat"]` is accepted, everything else is rejected as a route-policy
violation, and no route is ever silently added.
"""
from __future__ import annotations

import json

import pytest

from prism_fas.llm.pipeline import CandidateOutcome, RecipePlanner
from prism_fas.llm.route_policy import (ROUTE_POLICY_STAGE, ROUTE_POLICY_VIOLATION, RoutePolicy,
                                        RoutePolicyError, parse_route_policy)
from prism_fas.recipes.compile import compile_recipe

from c2c_constants import (C2C_BANK_ID, REQUIRED_ROUTE, ROUTE_POLICY_IDENTITY,
                           FROZEN_ITEM_SCHEMA_IDENTITY, FROZEN_ONTOLOGY_IDENTITY)


# ------------------------------------------------------------------- the rule
def test_the_only_valid_scientific_route_is_physics_then_gpat(planner, candidate, envelope):
    validation = planner.validate_response(
        envelope(candidate(["physics", "gpat"])), slot_id="C2C_BATCH_000", recipes_requested=1)
    assert validation.all_accepted
    assert validation.candidates[0].outcome is CandidateOutcome.ACCEPTED


@pytest.mark.parametrize("route, why", [
    (["physics"], "physics-only"),
    (["gpat"], "gpat-only"),
    ([], "empty"),
    (["gpat", "physics"], "wrong order"),
    (["physics", "physics"], "duplicate route value"),
])
def test_every_other_route_declaration_is_rejected(planner, candidate, envelope, route, why):
    validation = planner.validate_response(
        envelope(candidate(route)), slot_id="C2C_BATCH_000", recipes_requested=1)
    assert not validation.all_accepted, f"{why} was accepted"
    result = validation.candidates[0]
    # An empty or duplicated route is refused by the inherited typed schema
    # before the policy sees it; either rejection is a rejection.
    assert result.outcome in (CandidateOutcome.REJECTED_ROUTE_POLICY,
                              CandidateOutcome.REJECTED_SCHEMA), why
    assert result.recipe_identity is None or result.outcome is not CandidateOutcome.ACCEPTED


def test_an_unknown_route_value_is_rejected(planner, candidate, envelope):
    validation = planner.validate_response(
        envelope(candidate(["physics", "diffusion"])), slot_id="C2C_BATCH_000",
        recipes_requested=1)
    assert not validation.all_accepted
    assert validation.candidates[0].outcome is CandidateOutcome.REJECTED_SCHEMA


def test_a_route_violation_is_reported_with_its_own_stage_and_code(planner, candidate, envelope):
    validation = planner.validate_response(
        envelope(candidate(["gpat"])), slot_id="C2C_BATCH_000", recipes_requested=1)
    issues = validation.candidates[0].issues
    assert issues
    assert all(issue["stage"] == ROUTE_POLICY_STAGE for issue in issues)
    assert all(issue["code"] == ROUTE_POLICY_VIOLATION for issue in issues)
    assert all(issue["field"] == "generator_route" for issue in issues)


def test_a_route_invalid_recipe_never_reaches_accepted_state(planner, candidate, envelope):
    for route in (["physics"], ["gpat"]):
        validation = planner.validate_response(
            envelope(candidate(route)), slot_id="C2C_BATCH_000", recipes_requested=1)
        assert validation.accepted == []
        assert validation.candidates[0].canonical_text is None
        assert validation.candidates[0].recipe_identity is None


def test_a_route_invalid_recipe_is_never_registered_as_seen(planner, candidate, envelope):
    """A rejected candidate must not occupy the duplicate registry, or a later
    valid recipe with the same content would be wrongly called a duplicate."""
    planner.validate_response(envelope(candidate(["gpat"])), slot_id="C2C_BATCH_000",
                             recipes_requested=1)
    assert planner.seen_identities == {}


def test_no_route_is_silently_repaired(planner, candidate, envelope):
    """The rejected recipe is recorded exactly as the provider wrote it."""
    payload = candidate(["gpat"])
    validation = planner.validate_response(envelope(payload), slot_id="C2C_BATCH_000",
                                           recipes_requested=1)
    result = validation.candidates[0]
    assert result.recipe is not None
    assert list(result.recipe.generator_route) == ["gpat"], "the route was mutated"
    assert result.outcome is CandidateOutcome.REJECTED_ROUTE_POLICY


def test_every_accepted_recipe_is_physics_compilable(planner, candidate, envelope, ontology):
    """The point of the whole contract: accepted implies compilable."""
    payloads = [candidate(["physics", "gpat"], seed=1000 + index) for index in range(5)]
    validation = planner.validate_response(envelope(*payloads), slot_id="C2C_BATCH_000",
                                           recipes_requested=len(payloads))
    assert validation.all_accepted
    for result in validation.accepted:
        graph = compile_recipe(result.recipe, ontology, bank_id=C2C_BANK_ID)
        assert graph.graph_hash
        assert graph.conditioning_dimension == 41


def test_without_the_policy_a_gpat_only_recipe_is_accepted_and_then_fails_to_compile(
        planner_without_route_policy, candidate, envelope, ontology):
    """The C2B defect, reproduced, so the repair is demonstrably the difference."""
    from prism_fas.recipes.compile import CompileError

    validation = planner_without_route_policy.validate_response(
        envelope(candidate(["gpat"])), slot_id="C2B_BATCH_000", recipes_requested=1)
    assert validation.all_accepted, "pre-C2C behaviour accepted this recipe"
    with pytest.raises(CompileError, match="physics route is required"):
        compile_recipe(validation.accepted[0].recipe, ontology, bank_id=C2C_BANK_ID)


# ------------------------------------------------------------------- identity
def test_the_route_policy_is_identity_bearing(route_policy):
    assert route_policy.route_policy_identity == ROUTE_POLICY_IDENTITY
    assert list(route_policy.allowed_scientific_generator_route) == REQUIRED_ROUTE
    assert route_policy.require_exact_order is True
    assert route_policy.allow_gpat_only_class is False
    assert route_policy.silent_repair_permitted is False


def test_the_policy_identity_is_stable_and_canonical(route_policy):
    assert route_policy.route_policy_identity == RoutePolicy().route_policy_identity
    assert json.loads(route_policy.canonical_text())["allowed_scientific_generator_route"] == \
        REQUIRED_ROUTE


def test_the_route_policy_identity_enters_the_generation_request_identity(make_request,
                                                                         route_policy):
    with_policy = make_request(policy=route_policy)
    assert with_policy.identity_material()["route_policy_identity"] == ROUTE_POLICY_IDENTITY
    without = make_request(policy=None)
    assert "route_policy_identity" not in without.identity_material()
    assert with_policy.request_sha256 != without.request_sha256


def test_changing_the_route_policy_changes_the_request_identity(make_request, route_policy):
    baseline = make_request(policy=route_policy)
    altered = RoutePolicy(allowed_scientific_generator_route=("gpat", "physics"))
    assert altered.route_policy_identity != route_policy.route_policy_identity
    assert make_request(policy=altered).request_sha256 != baseline.request_sha256


def test_a_policy_naming_a_route_the_ontology_lacks_is_refused(ontology):
    with pytest.raises(RoutePolicyError, match="does not enable"):
        RoutePolicy(allowed_scientific_generator_route=("physics", "diffusion")
                    ).validate_against(ontology)


def test_a_gpat_only_accepted_class_cannot_be_enabled(ontology):
    with pytest.raises(RoutePolicyError, match="no GPAT-only scientific class"):
        RoutePolicy(allow_gpat_only_class=True).validate_against(ontology)


def test_silent_repair_cannot_be_enabled(ontology):
    with pytest.raises(RoutePolicyError, match="silent route repair is never permitted"):
        RoutePolicy(silent_repair_permitted=True).validate_against(ontology)


def test_an_unknown_policy_key_is_rejected_rather_than_ignored():
    with pytest.raises(RoutePolicyError, match="unknown route policy keys"):
        parse_route_policy({"version": "prism_c_route_policy_v1", "allow_physics_only": True})


# ---------------------------------------------------- what C2C did NOT change
def test_the_item_schema_identity_is_unchanged_by_the_route_repair(ontology):
    """The rule lives in the validation layer, so recipe semantics did not move."""
    from prism_fas.llm.json_schema import candidate_object_schema, json_schema_identity

    assert json_schema_identity(candidate_object_schema(ontology)) == FROZEN_ITEM_SCHEMA_IDENTITY


def test_the_ontology_identity_is_unchanged(ontology):
    assert ontology.sha256 == FROZEN_ONTOLOGY_IDENTITY


def test_the_ontology_still_enables_both_routes_individually(ontology):
    """The ontology is untouched: the restriction is a scientific policy on top
    of it, not a narrowing of the vocabulary."""
    assert set(ontology.routes) == {"physics", "gpat"}
