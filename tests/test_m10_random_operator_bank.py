"""The A02 control artifact: the random-operator recipe bank.

These tests exist to answer one question — is this a REAL control, or the
structured bank under a different name? The decisive assertion is
`off_manifold_any > 0`: a uniform draw over the full operator vocabulary must
actually produce compositions the structured ontology would have refused, or the
ablation measures nothing and hypothesis H4 is untestable.

They also pin what must NOT change between the two arms (operator set, parameter
bands, severity budget, recipe count, route mix, schema) and prove the artifact is
deterministic, immutable and source-only.
"""
from __future__ import annotations
import json
from pathlib import Path
import pytest

from prism_fas.recipes.bank import load_bank
from prism_fas.recipes.canonical import recipe_hash
from prism_fas.recipes.compile import compile_recipes
from prism_fas.recipes.generate import generate_recipes
from prism_fas.recipes.ontology import load_ontology
from prism_fas.synthesis.random_operator_bank import (DEFAULT_BANK_ID, DEFAULT_BANK_SEED,
                                                      DEFAULT_RECIPE_COUNT, RANDOM_BANK_SCHEMA_VERSION,
                                                      RANDOM_POLICY_VERSION, RandomOperatorBankError,
                                                      build_random_operator_bank, composition_report,
                                                      generate_random_recipes,
                                                      validate_random_operator_bank)

PROJECT = Path(__file__).resolve().parents[1]
M7_BANK = PROJECT / "assets" / "recipe_banks" / "prism_recipe_bank_m7_v1"
ONTOLOGY = M7_BANK / "ontology.yaml"
# The A02 recipe bank is a deterministic function of the ontology and the seed, so
# its identity is a PIN. `modal_m10_a02.py` asserts the same value in-container.
EXPECTED_RECIPE_BANK_IDENTITY = "9351d08ac824cc67021445d1bb59bd9dc14ef7eb3dfa606414500d8fac49603f"


@pytest.fixture(scope="module")
def ontology():
    return load_ontology(ONTOLOGY)


@pytest.fixture(scope="module")
def recipes(ontology):
    return generate_random_recipes(ontology, count=DEFAULT_RECIPE_COUNT,
                                   bank_id=DEFAULT_BANK_ID, bank_seed=DEFAULT_BANK_SEED)


# ============================================================================
# IS IT A REAL CONTROL?
# ============================================================================

def test_the_random_bank_actually_leaves_the_structured_manifold(recipes, ontology):
    """The decisive test. If a uniform draw never violates the medium/artifact or
    geometry/region compatibility rules, this bank IS the structured bank and H4
    cannot be answered."""
    report = composition_report(recipes, ontology)
    assert report["off_manifold_any"] > 0
    assert report["off_manifold_medium_artifact"] > 0
    assert report["off_manifold_geometry_region"] > 0
    # Measured on the frozen ontology and seed, so a drift in either is visible.
    assert report["off_manifold_any"] == 70
    assert report["off_manifold_medium_artifact"] == 62
    assert report["off_manifold_geometry_region"] == 16


def test_the_builder_refuses_to_ship_a_null_control(monkeypatch, tmp_path):
    """If the composition ever stopped leaving the manifold, the build must FAIL
    rather than quietly produce a control that controls for nothing."""
    import prism_fas.synthesis.random_operator_bank as module
    real = module.composition_report

    def structured_looking(recipes, ontology):
        return {**real(recipes, ontology), "off_manifold_any": 0,
                "off_manifold_medium_artifact": 0, "off_manifold_geometry_region": 0}

    monkeypatch.setattr(module, "composition_report", structured_looking)
    with pytest.raises(RandomOperatorBankError, match="structured bank under another name"):
        build_random_operator_bank(tmp_path / "bank", ONTOLOGY, dry_run=True)


def test_the_random_recipes_are_not_the_structured_recipes(recipes, ontology):
    """Not a shuffle of recipe ids over the structured payloads: the canonical
    content differs, which is what makes the DATA differ rather than the label."""
    structured = generate_recipes(ontology, count=DEFAULT_RECIPE_COUNT,
                                  bank_id="prism_recipe_bank_m7_v1", bank_seed=DEFAULT_BANK_SEED)
    random_hashes = {recipe_hash(recipe) for recipe in recipes}
    structured_hashes = {recipe_hash(recipe) for recipe in structured}
    assert not (random_hashes & structured_hashes)
    # The ids are the SAME sequence; only the content behind them differs. A test
    # that compared ids would pass for a shuffle, which is why it compares content.
    assert [r.recipe_id for r in recipes] == [s.recipe_id for s in structured]


def test_the_random_bank_is_not_the_frozen_m7_bank(recipes):
    frozen = load_bank(M7_BANK)
    frozen_hashes = {recipe_hash(recipe) for recipe in frozen["recipes"]}
    assert not ({recipe_hash(recipe) for recipe in recipes} & frozen_hashes)


# ============================================================================
# WHAT MUST BE HELD CONSTANT BETWEEN THE TWO ARMS
# ============================================================================

def test_the_operator_vocabulary_and_bands_are_unchanged(recipes, ontology):
    for recipe in recipes:
        for spec in recipe.artifacts:
            assert spec.name in ontology.artifacts
            band = ontology.strength_range(spec.name)
            assert band.minimum - 1e-9 <= spec.strength <= band.maximum + 1e-9
        for name in recipe.regions:
            assert name in ontology.regions


def test_the_shared_severity_budget_is_respected(recipes, ontology):
    """The budget is a SAFETY limit both arms obey, not a composition rule."""
    budget = float(ontology.limits["max_total_artifact_strength"])
    for recipe in recipes:
        assert sum(spec.strength for spec in recipe.artifacts) <= budget + 1e-9


def test_the_recipe_count_and_route_mix_match_the_structured_arm(recipes, ontology):
    """Equal sample count and equal route mix: H4 compares composition, not budget
    and not the A03 route dimension."""
    structured = generate_recipes(ontology, count=DEFAULT_RECIPE_COUNT,
                                  bank_id="prism_recipe_bank_m7_v1", bank_seed=DEFAULT_BANK_SEED)
    assert len(recipes) == len(structured) == DEFAULT_RECIPE_COUNT
    routes = lambda items: sorted("+".join(item.generator_route) for item in items)
    assert routes(recipes) == routes(structured)


def test_every_recipe_is_compilable_and_declares_physics(recipes, ontology):
    assert all("physics" in recipe.generator_route for recipe in recipes)
    graphs = compile_recipes(recipes, ontology, bank_id=DEFAULT_BANK_ID)
    assert len({graph.graph_hash for graph in graphs}) == len(recipes)


def test_the_forbidden_shortcut_guards_are_carried_unchanged(recipes, ontology):
    for recipe in recipes:
        assert recipe.forbidden_shortcuts == list(ontology.forbidden_shortcuts)


# ============================================================================
# DETERMINISM, IDENTITY AND IMMUTABILITY
# ============================================================================

def test_the_bank_is_deterministic_and_identity_bearing():
    first = build_random_operator_bank(Path("unused"), ONTOLOGY, dry_run=True)
    second = build_random_operator_bank(Path("unused"), ONTOLOGY, dry_run=True)
    assert first["bank_content_identity_sha256"] == second["bank_content_identity_sha256"]
    assert first["bank_content_identity_sha256"] == EXPECTED_RECIPE_BANK_IDENTITY


def test_a_different_seed_gives_a_different_identity():
    other = build_random_operator_bank(Path("unused"), ONTOLOGY, bank_seed=20260807, dry_run=True)
    assert other["bank_content_identity_sha256"] != EXPECTED_RECIPE_BANK_IDENTITY


def test_writing_then_revalidating_reproduces_every_hash(tmp_path):
    root = tmp_path / DEFAULT_BANK_ID
    created = build_random_operator_bank(root, ONTOLOGY)
    assert created["status"] == "created"
    validation = validate_random_operator_bank(root)
    assert validation["passed"], validation["errors"]
    assert validation["bank_content_identity_sha256"] == EXPECTED_RECIPE_BANK_IDENTITY
    assert validation["composition_policy"] == RANDOM_POLICY_VERSION
    assert validation["source_only"] is True


def test_rebuilding_over_an_identical_bank_is_a_no_op(tmp_path):
    root = tmp_path / DEFAULT_BANK_ID
    build_random_operator_bank(root, ONTOLOGY)
    again = build_random_operator_bank(root, ONTOLOGY)
    assert again["status"] == "reused" and again["written"] == []


def test_the_artifact_is_immutable(tmp_path):
    """A destination holding a different lock is never overwritten."""
    root = tmp_path / DEFAULT_BANK_ID
    build_random_operator_bank(root, ONTOLOGY)
    with pytest.raises(RandomOperatorBankError, match="immutable"):
        build_random_operator_bank(root, ONTOLOGY, bank_seed=20260807)


def test_a_tampered_file_fails_validation(tmp_path):
    root = tmp_path / DEFAULT_BANK_ID
    build_random_operator_bank(root, ONTOLOGY)
    lines = (root / "recipes.jsonl").read_text(encoding="utf-8").splitlines()
    (root / "recipes.jsonl").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    result = validate_random_operator_bank(root)
    assert result["passed"] is False and result["errors"]


def test_the_lock_declares_its_own_schema_and_purpose(tmp_path):
    root = tmp_path / DEFAULT_BANK_ID
    build_random_operator_bank(root, ONTOLOGY)
    lock = json.loads((root / "BANK_LOCK.json").read_text(encoding="utf-8"))
    assert lock["bank_schema_version"] == RANDOM_BANK_SCHEMA_VERSION
    assert lock["composition_policy"] == RANDOM_POLICY_VERSION
    assert lock["bank_id"] == DEFAULT_BANK_ID
    assert lock["status"] == "frozen"
    assert lock["source_only"] is True
    assert lock["target_paths"] == 0 and lock["target_taxonomy"] == 0
    # It must be impossible to mistake this lock for an M7 bank lock.
    assert lock["bank_id"] != "prism_recipe_bank_m7_v1"
    assert lock["bank_content_identity_sha256"] != \
        json.loads((M7_BANK / "BANK_LOCK.json").read_text(encoding="utf-8"))["bank_content_identity_sha256"]


def test_the_written_bank_loads_through_the_ordinary_bank_reader(tmp_path):
    """The M8 pipeline consumes it unchanged; that is what keeps the detector and
    the generator identical between the two arms."""
    root = tmp_path / DEFAULT_BANK_ID
    build_random_operator_bank(root, ONTOLOGY)
    bank = load_bank(root)
    assert bank["bank_id"] == DEFAULT_BANK_ID
    assert len(bank["recipes"]) == DEFAULT_RECIPE_COUNT


# ============================================================================
# SOURCE-ONLY
# ============================================================================

def test_the_bank_names_no_target_anything(tmp_path):
    root = tmp_path / DEFAULT_BANK_ID
    build_random_operator_bank(root, ONTOLOGY)
    forbidden = ("siw", "target_test", "spoof_type", "attack_family")
    for name in ("recipes.jsonl", "generator.json", "coverage.json", "validation.json",
                 "BANK_LOCK.json"):
        text = (root / name).read_text(encoding="utf-8").lower()
        for needle in forbidden:
            assert needle not in text, f"{name} mentions {needle!r}"


def test_the_module_reads_nothing_but_the_ontology():
    """Source-only by construction: no dataset, no package, no image, no label.

    Scanned over the CODE with every string literal and comment removed. A blanket
    text scan flags the docstring that EXPLAINS the rule as a violation of it —
    the same mistake DECISIONS.md has now recorded three times.
    """
    import io, tokenize
    from prism_fas.synthesis import random_operator_bank
    path = Path(random_operator_bank.__file__)
    kept = [token.string for token in tokenize.generate_tokens(io.StringIO(
        path.read_text(encoding="utf-8")).readline)
        if token.type not in (tokenize.STRING, tokenize.COMMENT)]
    code = " ".join(kept)
    for needle in ("PACKAGE_LOCK", "source_train", "source_dev", "target_test", "siw_mv2",
                   "CanonicalPackageDataset", "load_manifest"):
        assert needle not in code, f"the builder's CODE references {needle!r}"
