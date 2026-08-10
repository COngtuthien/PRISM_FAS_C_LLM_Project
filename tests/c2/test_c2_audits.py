"""The C2 audits must be deterministic and must match the committed artifacts.

An audit that changed between runs could not be cited, and an audit artifact that
drifted from the archive it claims to describe would be decoration rather than
evidence. Both properties are checked here against the real archived pilot.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism_fas.llm.pilot_audit import (cooccurrence_audit, coverage_audit, duplicate_audit,
                                       flag_pilot_issues, latency_stats, percentile,
                                       structural_pattern)
from prism_fas.llm.pipeline import RecipePlanner
from prism_fas.recipes.validate import validate_payload

COVERAGE_AUDIT_PATH = (Path(__file__).resolve().parents[2] / "reports" / "c2"
                       / "C2_COVERAGE_AUDIT.json")


def accepted_recipes(pilot_state, ontology):
    recipes = []
    for slot in pilot_state["slots"]:
        if slot["final_status"] != "accepted":
            continue
        recipe, issues = validate_payload(json.loads(slot["canonical_recipe"]), ontology,
                                          canonicalize=False)
        assert recipe is not None and issues == []
        recipes.append(recipe)
    return recipes


def test_coverage_audit_is_deterministic(pilot_state, ontology):
    recipes = accepted_recipes(pilot_state, ontology)
    first = coverage_audit(recipes, ontology)
    second = coverage_audit(recipes, ontology)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_coverage_axes_cover_the_frozen_vocabulary(pilot_state, ontology):
    audit = coverage_audit(accepted_recipes(pilot_state, ontology), ontology)
    assert list(audit["axes"]["artifacts"]["counts"]) == list(ontology.artifacts)
    assert list(audit["axes"]["regions"]["counts"]) == list(ontology.regions)
    assert list(audit["axes"]["media"]["counts"]) == list(ontology.media)
    assert list(audit["axes"]["geometry"]["counts"]) == list(ontology.geometry_shapes)
    assert list(audit["axes"]["illumination"]["counts"]) == list(ontology.illumination)


def test_coverage_counts_agree_with_the_recipes(pilot_state, ontology):
    recipes = accepted_recipes(pilot_state, ontology)
    audit = coverage_audit(recipes, ontology)
    assert audit["recipe_count"] == len(recipes)
    # A single-valued axis must account for exactly one assignment per recipe.
    for axis in ("media", "geometry", "illumination"):
        assert audit["axes"][axis]["assignment_total"] == len(recipes)
    assert audit["axes"]["artifacts"]["assignment_total"] == sum(
        len(recipe.artifacts) for recipe in recipes)
    assert audit["axes"]["regions"]["assignment_total"] == sum(
        len(recipe.regions) for recipe in recipes)


def test_cooccurrence_audit_is_deterministic(pilot_state, ontology):
    recipes = accepted_recipes(pilot_state, ontology)
    first = cooccurrence_audit(recipes, ontology)
    second = cooccurrence_audit(recipes, ontology)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_cooccurrence_marks_cells_the_ontology_forbids(pilot_state, ontology):
    audit = cooccurrence_audit(accepted_recipes(pilot_state, ontology), ontology)
    table = audit["tables"]["artifact_x_medium"]
    # paper-like cannot emit an emissive pixel grid; the cell must be marked
    # incompatible and must be empty.
    assert table["compatible_cells"]["pixel_grid"]["paper-like"] is False
    assert table["cells"]["pixel_grid"]["paper-like"] == 0
    for artifact in ontology.artifacts:
        for medium in ontology.media:
            if not table["compatible_cells"][artifact][medium]:
                assert table["cells"][artifact][medium] == 0, (
                    f"{artifact} appeared under an incompatible medium {medium}")


def test_duplicate_audit_is_deterministic_and_uses_content_identity(pilot_state, ontology):
    recipes = accepted_recipes(pilot_state, ontology)
    identities = [RecipePlanner.content_identity(recipe) for recipe in recipes]
    first = duplicate_audit(identities, recipes)
    second = duplicate_audit(identities, recipes)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["accepted_recipe_count"] == len(recipes)
    # The pipeline rejects a duplicate before acceptance, so an accepted bank
    # must carry no exact repeat at all.
    assert first["exact_duplicate_groups"] == 0
    assert first["exact_duplicate_rate"] == 0.0


def test_content_identity_ignores_the_positional_recipe_id(pilot_state, ontology):
    recipes = accepted_recipes(pilot_state, ontology)
    assert recipes, "no accepted recipe"
    original = recipes[0]
    renumbered = original.model_copy(update={"recipe_id": "R-999999"})
    assert RecipePlanner.content_identity(original) == RecipePlanner.content_identity(renumbered)


def test_structural_pattern_ignores_continuous_values(pilot_state, ontology):
    recipes = accepted_recipes(pilot_state, ontology)
    original = recipes[0]
    nudged = original.model_copy(update={"seed": (original.seed + 1) % 2_147_483_647})
    assert structural_pattern(original) == structural_pattern(nudged)


def test_flags_are_deterministic(pilot_state, ontology):
    recipes = accepted_recipes(pilot_state, ontology)
    coverage = coverage_audit(recipes, ontology)
    cooccurrence = cooccurrence_audit(recipes, ontology)
    identities = [RecipePlanner.content_identity(recipe) for recipe in recipes]
    duplicates = duplicate_audit(identities, recipes)
    statistics = {"counts": {"compatibility_violations": 0}}
    first = flag_pilot_issues(coverage, cooccurrence, duplicates, statistics, recipes, ontology)
    second = flag_pilot_issues(coverage, cooccurrence, duplicates, statistics, recipes, ontology)
    assert first == second


def test_percentile_and_latency_stats_are_exact():
    assert percentile([1.0], 0.95) == 1.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentile([], 0.5) is None
    stats = latency_stats([1.0, 2.0, 3.0])
    assert stats["n"] == 3 and stats["mean"] == 2.0 and stats["median"] == 2.0
    assert latency_stats([])["n"] == 0


def test_the_committed_coverage_artifact_matches_a_fresh_recomputation(pilot_state, ontology):
    """The artifact in reports/c2/ must still describe the archive it cites."""
    committed_path = COVERAGE_AUDIT_PATH
    if not committed_path.exists():
        pytest.skip(f"{committed_path.name} missing; run: python scripts/c2_build_reports.py")
    committed = json.loads(committed_path.read_text(encoding="utf-8"))
    recipes = accepted_recipes(pilot_state, ontology)
    fresh = coverage_audit(recipes, ontology)
    assert json.dumps(committed["coverage"], sort_keys=True) == json.dumps(fresh, sort_keys=True)
    fresh_cooccurrence = cooccurrence_audit(recipes, ontology)
    assert (json.dumps(committed["cooccurrence"], sort_keys=True)
            == json.dumps(fresh_cooccurrence, sort_keys=True))


def test_the_committed_pilot_audit_agrees_with_the_state(pilot_audit, pilot_state, ontology):
    recipes = accepted_recipes(pilot_state, ontology)
    assert pilot_audit["statistics"]["slots"]["successful_slots"] == len(recipes)
    assert pilot_audit["statistics"]["slots"]["slot_count"] == len(pilot_state["slots"])
    assert pilot_audit["replay_verification"]["identical"] is True
