"""C1 groups C (compatibility), D (canonicalization), E (duplicates) and
K (compiler compatibility).

The load-bearing claim: a candidate that this contract accepts is a candidate the
inherited compiler can execute. If that were untrue, C3 could freeze a bank that
fails at synthesis time.
"""
from __future__ import annotations

import json

import pytest

from prism_fas.llm.pipeline import (
    PIPELINE_STAGES,
    CandidateOutcome,
    RecipePlanner,
    assign_recipe_id,
    compile_accepted,
)
from prism_fas.recipes.canonical import canonical_json, recipe_hash
from prism_fas.recipes.compile import CONDITIONING_DIM, COMPILER_VERSION

from c1_helpers import envelope


def _accept_one(planner: RecipePlanner, candidate: dict, slot_id: str = "s"):
    result = planner.validate_response(envelope(candidate), slot_id=slot_id, recipes_requested=1)
    assert result.all_accepted, result.as_dict()
    return result.candidates[0]


# ===================== C. COMPATIBILITY ===================================

def test_incompatible_medium_artifact_combination_is_rejected(planner, ontology, valid_candidate):
    """Find an artifact this medium genuinely cannot produce, then prove the
    ontology stage refuses it."""
    family = valid_candidate["medium"]["family"]
    allowed = set(ontology.artifacts_for_medium(family))
    forbidden = [name for name in ontology.artifacts if name not in allowed]
    assert forbidden, f"ontology {ontology.version} allows every artifact for {family!r}"
    band = ontology.strength_range(forbidden[0])
    valid_candidate["artifacts"] = [{"name": forbidden[0],
                                     "strength": round((band.minimum + band.maximum) / 2, 4)}]
    result = planner.validate_response(envelope(valid_candidate), slot_id="s", recipes_requested=1)
    assert result.candidates[0].outcome is CandidateOutcome.REJECTED_ONTOLOGY
    assert any(issue["stage"] == "medium_artifact" for issue in result.candidates[0].issues)


def test_incompatible_geometry_region_combination_is_rejected(planner, ontology, valid_candidate):
    """Pick a geometry that genuinely restricts its regions, rather than skipping
    on the permissive one the fixture happens to use."""
    restricted = [(shape, sorted(set(ontology.regions) - set(ontology.regions_for_geometry(shape))))
                  for shape in ontology.geometry_shapes]
    candidates = [(shape, missing) for shape, missing in restricted if missing]
    assert candidates, f"ontology {ontology.version} has no region-restricting geometry to test"
    shape, missing = candidates[0]
    valid_candidate["geometry"] = {**valid_candidate["geometry"], "shape": shape}
    valid_candidate["regions"] = [missing[0]]
    result = planner.validate_response(envelope(valid_candidate), slot_id="s", recipes_requested=1)
    assert result.candidates[0].outcome is CandidateOutcome.REJECTED_ONTOLOGY
    assert any(issue["stage"] == "geometry_region" for issue in result.candidates[0].issues)


def test_a_valid_combination_reaches_the_compiler(planner, ontology, valid_candidate):
    candidate = _accept_one(planner, valid_candidate)
    graph = compile_accepted(candidate.recipe, ontology, bank_id="c1-fixture")
    assert graph.graph_hash


def test_pipeline_stages_are_declared_in_order():
    assert PIPELINE_STAGES[0] == "json_parsing"
    assert PIPELINE_STAGES[-1] == "duplicate_detection"
    assert "ontology_membership" in PIPELINE_STAGES
    assert "compatibility_checks" in PIPELINE_STAGES
    assert len(set(PIPELINE_STAGES)) == len(PIPELINE_STAGES)


# ===================== D. CANONICALIZATION ================================

def test_canonical_serialization_is_deterministic(planner, valid_candidate):
    candidate = _accept_one(planner, valid_candidate)
    assert canonical_json(candidate.recipe) == candidate.canonical_text
    assert canonical_json(candidate.recipe) == canonical_json(candidate.recipe)


def test_canonical_identity_is_stable_across_a_round_trip(planner, valid_candidate):
    candidate = _accept_one(planner, valid_candidate)
    reparsed = json.loads(candidate.canonical_text)
    assert reparsed["recipe_id"] == candidate.recipe.recipe_id
    assert recipe_hash(candidate.recipe) == candidate.recipe_identity


def test_key_order_in_the_response_does_not_change_the_identity(planner, config, ontology,
                                                                valid_candidate):
    """The model may emit keys in any order; the identity must not depend on it."""
    first = _accept_one(planner, valid_candidate, slot_id="a")
    shuffled = dict(reversed(list(valid_candidate.items())))
    other = RecipePlanner(provider=planner._provider, config=config, ontology=ontology)
    second = _accept_one(other, shuffled, slot_id="b")
    assert first.recipe_identity == second.recipe_identity


def test_equivalent_float_spellings_canonicalize_together(planner, config, ontology,
                                                          valid_candidate):
    """0.5864 and 0.58640000 are the same number and must hash the same."""
    first = _accept_one(planner, valid_candidate, slot_id="a")
    respelled = json.loads(json.dumps(valid_candidate))
    respelled["artifacts"][0]["strength"] = 0.58640000
    other = RecipePlanner(provider=planner._provider, config=config, ontology=ontology)
    second = _accept_one(other, respelled, slot_id="b")
    assert first.recipe_identity == second.recipe_identity


def test_a_meaningful_difference_is_not_canonicalized_away(planner, valid_candidate):
    first = _accept_one(planner, valid_candidate, slot_id="a")
    changed = json.loads(json.dumps(valid_candidate))
    changed["artifacts"][0]["strength"] = round(changed["artifacts"][0]["strength"] - 0.01, 4)
    result = planner.validate_response(envelope(changed), slot_id="b", recipes_requested=1)
    assert result.all_accepted
    assert result.candidates[0].recipe_identity != first.recipe_identity


def test_recipe_ids_are_assigned_positionally_and_deterministically():
    assert assign_recipe_id(0) == "R-000000"
    assert assign_recipe_id(255) == "R-000255"
    with pytest.raises(ValueError):
        assign_recipe_id(-1)
    with pytest.raises(ValueError):
        assign_recipe_id(1_000_000)


# ===================== E. DUPLICATES ======================================

def test_exact_duplicate_is_rejected(planner, valid_candidate):
    _accept_one(planner, valid_candidate, slot_id="a")
    result = planner.validate_response(envelope(valid_candidate), slot_id="b", recipes_requested=1)
    assert result.candidates[0].outcome is CandidateOutcome.REJECTED_DUPLICATE
    assert "slot 'a'" in result.candidates[0].issues[0]["reason"]


def test_duplicate_within_a_single_response_is_rejected(planner, valid_candidate):
    result = planner.validate_response(envelope(valid_candidate, valid_candidate),
                                       slot_id="s", recipes_requested=2)
    assert result.candidates[0].outcome is CandidateOutcome.ACCEPTED
    assert result.candidates[1].outcome is CandidateOutcome.REJECTED_DUPLICATE


def test_duplicate_detection_ignores_the_positional_recipe_id(planner, valid_candidate):
    """Two identical recipes land at different indices and therefore get
    different ids. They must still collide."""
    first = planner.validate_response(envelope(valid_candidate), slot_id="a",
                                      recipes_requested=1, next_recipe_index=0)
    second = planner.validate_response(envelope(valid_candidate), slot_id="b",
                                       recipes_requested=1, next_recipe_index=7)
    assert first.candidates[0].outcome is CandidateOutcome.ACCEPTED
    assert second.candidates[0].outcome is CandidateOutcome.REJECTED_DUPLICATE


def test_key_reordering_does_not_defeat_duplicate_detection(planner, valid_candidate):
    _accept_one(planner, valid_candidate, slot_id="a")
    shuffled = dict(reversed(list(valid_candidate.items())))
    result = planner.validate_response(envelope(shuffled), slot_id="b", recipes_requested=1)
    assert result.candidates[0].outcome is CandidateOutcome.REJECTED_DUPLICATE


def test_a_genuinely_different_recipe_is_not_a_duplicate(planner, valid_candidate,
                                                         second_valid_candidate):
    result = planner.validate_response(envelope(valid_candidate, second_valid_candidate),
                                       slot_id="s", recipes_requested=2)
    assert result.all_accepted
    identities = {item.recipe_identity for item in result.candidates}
    assert len(identities) == 2


def test_system_owned_fields_are_refused_not_stripped(planner, valid_candidate):
    """Silently dropping a model-supplied recipe_id would be a repair. It is a
    rejection instead, with a specific reason."""
    valid_candidate["recipe_id"] = "R-123456"
    result = planner.validate_response(envelope(valid_candidate), slot_id="s", recipes_requested=1)
    candidate = result.candidates[0]
    assert candidate.outcome is CandidateOutcome.REJECTED_SYSTEM_OWNED_FIELD
    assert "recipe_id" in candidate.issues[0]["reason"]


# ===================== K. COMPILER COMPATIBILITY ==========================

def test_accepted_fixture_compiles_to_the_full_contract(planner, ontology, valid_candidate):
    candidate = _accept_one(planner, valid_candidate)
    graph = compile_accepted(candidate.recipe, ontology, bank_id="c1-fixture")

    assert graph.compiler_version == COMPILER_VERSION
    assert graph.recipe_id == candidate.recipe.recipe_id
    assert graph.recipe_hash == candidate.recipe_identity
    assert graph.ontology_sha256 == ontology.sha256

    # operator graph
    assert len(graph.nodes) == len(candidate.recipe.artifacts)
    assert {node.operator_name for node in graph.nodes} == {
        spec.name for spec in candidate.recipe.artifacts}

    # mask policy
    assert graph.region_mask_policy["regions"] == list(candidate.recipe.regions)
    assert graph.region_mask_policy["policy"]

    # 41-D conditioning
    assert graph.conditioning_dimension == CONDITIONING_DIM == 41
    assert len(graph.conditioning_vector) == 41
    assert len(graph.conditioning_feature_names) == 41


def test_compilation_is_deterministic(planner, ontology, valid_candidate):
    candidate = _accept_one(planner, valid_candidate)
    first = compile_accepted(candidate.recipe, ontology, bank_id="c1-fixture")
    second = compile_accepted(candidate.recipe, ontology, bank_id="c1-fixture")
    assert first.graph_hash == second.graph_hash


def test_every_accepted_batch_member_compiles(planner, ontology, valid_candidate,
                                              second_valid_candidate):
    result = planner.validate_response(envelope(valid_candidate, second_valid_candidate),
                                       slot_id="s", recipes_requested=2)
    assert result.all_accepted
    for candidate in result.candidates:
        graph = compile_accepted(candidate.recipe, ontology, bank_id="c1-fixture")
        assert graph.graph_hash
        assert len(graph.conditioning_vector) == 41


def test_a_rejected_candidate_never_reaches_the_compiler(planner, valid_candidate):
    valid_candidate["artifacts"] = [{"name": "rainbow_glow", "strength": 0.3}]
    result = planner.validate_response(envelope(valid_candidate), slot_id="s", recipes_requested=1)
    assert result.accepted == []
    assert result.candidates[0].recipe is None
