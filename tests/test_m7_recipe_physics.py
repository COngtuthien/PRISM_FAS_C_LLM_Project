"""M7 focused tests: recipe schema/ontology/validation, compiler, conditioning,
frozen bank, region masks, physics operators and engine, plus the contracts of
the real source-only audit.

No network access and no model/dataset download. The real 64-preview integration
results are asserted from the ignored report artifacts produced by the explicit
`synthesis physics-audit` command.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pytest

from prism_fas.recipes.audit import bank_audit, coverage_table, diversity_audit
from prism_fas.recipes.bank import BANK_FILES, BankError, build_bank, load_bank, validate_bank
from prism_fas.recipes.canonical import canonical_json, recipe_description, recipe_hash
from prism_fas.recipes.compile import COMPILER_VERSION, CompileError, compile_recipe, derive_seed, local_rng
from prism_fas.recipes.conditioning import (CONDITIONING_DIM, CONDITIONING_VERSION, ConditioningError,
                                            conditioning_vector, decode_conditioning, feature_names,
                                            feature_names_sha256, normalize_compression, normalize_scale, normalize_yaw)
from prism_fas.recipes.generate import GENERATOR_EXTERNAL_LLM_INVOKED, build_recipe, render_prompt
from prism_fas.recipes.ontology import load_ontology
from prism_fas.recipes.schema import RECIPE_SCHEMA_VERSION, RecipeSchemaError, parse_recipe
from prism_fas.recipes.validate import validate_payload, validate_recipe, validate_recipes
from prism_fas.synthesis.contracts import MaskBuildError, SynthesisError
from prism_fas.synthesis.masks import MIN_PARSING_PIXELS, PARSING_LABELS, REGION_ORDER, RegionMaskBuilder
from prism_fas.synthesis.operators import (OPERATOR_APPLICATION_ORDER, OPERATOR_CLASSES, OPERATOR_NAMES,
                                           build_operator)
from prism_fas.synthesis.physics import PhysicsEngine

ROOT = Path(__file__).parents[1]
ONTOLOGY_PATH = ROOT / "configs" / "recipes" / "ontology_m7.yaml"
BANK_CONFIG = ROOT / "configs" / "recipes" / "bank_m7.yaml"
PHYSICS_CONFIG = ROOT / "configs" / "synthesis" / "physics_m7.yaml"
BANK_ROOT = ROOT / "assets" / "recipe_banks" / "prism_recipe_bank_m7_v1"
REPORTS = ROOT / "reports" / "m7"
ONTOLOGY = load_ontology(ONTOLOGY_PATH)
BANK_ID = "prism_recipe_bank_m7_v1"


def valid_payload(**overrides):
    payload = {
        "recipe_id": "R-000042",
        "medium": {"family": "display-like", "transparency": 0.0, "roughness": 0.25},
        "geometry": {"shape": "partial-curved", "rigidity": 0.4, "coverage": 0.22},
        "regions": ["right_eye", "right_cheek"],
        "artifacts": [{"name": "specular_reflection", "strength": 0.32},
                      {"name": "texture_smoothing", "strength": 0.18},
                      {"name": "boundary_inconsistency", "strength": 0.12}],
        "capture": {"yaw": 15.0, "illumination": "right", "compression_q": 82, "scale": 1.0,
                    "motion": 0.1, "defocus": 0.2},
        "forbidden_shortcuts": ["always_moire", "always_halftone"],
        "generator_route": ["physics", "gpat"],
        "seed": 7319,
        "schema_version": "1.1"}
    payload.update(overrides)
    return payload


def fixture_arrays(height=64, width=64):
    """Tiny synthetic parsing/landmark fixture with every LaPa class present."""
    parsing = np.zeros((height, width), dtype=np.uint8)
    parsing[10:56, 12:52] = PARSING_LABELS["skin"]
    parsing[4:12, 12:52] = PARSING_LABELS["hair"]
    parsing[20:24, 16:26] = PARSING_LABELS["left_eyebrow"]
    parsing[20:24, 38:48] = PARSING_LABELS["right_eyebrow"]
    parsing[25:31, 17:27] = PARSING_LABELS["left_eye"]
    parsing[25:31, 37:47] = PARSING_LABELS["right_eye"]
    parsing[30:40, 28:36] = PARSING_LABELS["nose"]
    parsing[44:47, 24:40] = PARSING_LABELS["upper_lip"]
    parsing[47:49, 26:38] = PARSING_LABELS["inner_mouth"]
    parsing[49:52, 24:40] = PARSING_LABELS["lower_lip"]
    landmarks = np.asarray([[22.0, 28.0], [42.0, 28.0], [32.0, 35.0], [26.0, 48.0], [38.0, 48.0]], dtype=np.float32)
    bbox = np.asarray([12.0, 10.0, 52.0, 56.0], dtype=np.float32)
    return parsing, landmarks, bbox


def fixture_image(height=64, width=64, seed=7):
    rng = np.random.Generator(np.random.PCG64(seed))
    return np.clip(rng.random((3, height, width)).astype(np.float32) * 0.6 + 0.2, 0.0, 1.0).astype(np.float32)


def full_mask(height=64, width=64):
    return np.ones((1, height, width), dtype=np.float32)


CAPTURE = {"yaw": 10.0, "illumination": "left", "compression_q": 80, "scale": 1.0, "motion": 0.3, "defocus": 0.1}


# --- 1-12 recipe schema, ontology and validation -----------------------------

def test_valid_recipe_parses_and_validates():
    recipe = parse_recipe(valid_payload())
    assert recipe.schema_version == RECIPE_SCHEMA_VERSION and recipe.recipe_id == "R-000042"
    assert validate_recipe(recipe, ONTOLOGY) == []


def test_unknown_field_is_rejected():
    with pytest.raises(RecipeSchemaError, match="extra_forbidden|Extra inputs"):
        parse_recipe(valid_payload(attack_family="print"))


@pytest.mark.parametrize("recipe_id", ["R-42", "r-000042", "R-0000042", "X-000042", ""])
def test_invalid_recipe_id_is_rejected(recipe_id):
    with pytest.raises(RecipeSchemaError):
        parse_recipe(valid_payload(recipe_id=recipe_id))


@pytest.mark.parametrize("version", ["1.0", "1.2", "v1.1", ""])
def test_invalid_schema_version_is_rejected(version):
    with pytest.raises(RecipeSchemaError):
        parse_recipe(valid_payload(schema_version=version))


@pytest.mark.parametrize("payload", [
    {"medium": {"family": "hologram-like", "transparency": 0.0, "roughness": 0.2}},
    {"medium": {"family": "display-like", "transparency": 1.5, "roughness": 0.2}},
    {"geometry": {"shape": "spherical", "rigidity": 0.4, "coverage": 0.2}},
    {"capture": {"yaw": 15.0, "illumination": "side", "compression_q": 82, "scale": 1.0, "motion": 0.1, "defocus": 0.2}},
    {"seed": -1},
    {"generator_route": ["diffusion"]},
    {"forbidden_shortcuts": ["always_blur"]},
    {"artifacts": [{"name": "sparkle", "strength": 0.2}]},
])
def test_enum_and_range_violations_are_rejected(payload):
    with pytest.raises(RecipeSchemaError):
        parse_recipe(valid_payload(**payload))


def test_duplicate_region_is_rejected():
    with pytest.raises(RecipeSchemaError, match="duplicate regions"):
        parse_recipe(valid_payload(regions=["nose", "nose"]))


def test_duplicate_artifact_is_rejected():
    with pytest.raises(RecipeSchemaError, match="duplicate artifact"):
        parse_recipe(valid_payload(artifacts=[{"name": "blur", "strength": 0.2}, {"name": "blur", "strength": 0.1}]))


def test_severity_budget_is_enforced():
    recipe = parse_recipe(valid_payload(artifacts=[{"name": "specular_reflection", "strength": 0.6},
                                                   {"name": "texture_smoothing", "strength": 0.6}]))
    issues = validate_recipe(recipe, ONTOLOGY)
    assert [issue.stage for issue in issues] == ["severity"]
    assert "exceeds budget 1.0" in issues[0].reason and issues[0].recipe_id == "R-000042"


def test_medium_artifact_compatibility_is_enforced():
    # an emissive pixel grid cannot come from printed paper
    recipe = parse_recipe(valid_payload(medium={"family": "paper-like", "transparency": 0.0, "roughness": 0.4},
                                        artifacts=[{"name": "pixel_grid", "strength": 0.2}]))
    issues = validate_recipe(recipe, ONTOLOGY)
    assert any(issue.stage == "medium_artifact" and issue.field == "artifacts" for issue in issues)


def test_boundary_only_geometry_compatibility_is_enforced():
    recipe = parse_recipe(valid_payload(geometry={"shape": "boundary-only", "rigidity": 0.5, "coverage": 0.3},
                                        regions=["nose"]))
    issues = validate_recipe(recipe, ONTOLOGY)
    assert any(issue.stage == "geometry_region" and "cannot cover" in issue.reason for issue in issues)
    ok = parse_recipe(valid_payload(geometry={"shape": "boundary-only", "rigidity": 0.5, "coverage": 0.3},
                                    regions=["face_boundary", "context"]))
    assert not [issue for issue in validate_recipe(ok, ONTOLOGY) if issue.stage == "geometry_region"]


def test_artifact_strength_outside_the_operator_safe_band_is_rejected():
    recipe = parse_recipe(valid_payload(artifacts=[{"name": "color_shift", "strength": 0.9}]))
    issues = validate_recipe(recipe, ONTOLOGY)
    assert any(issue.stage == "strength_range" for issue in issues)


def test_undocumented_artifact_parameter_is_rejected_not_dropped():
    recipe = parse_recipe(valid_payload(artifacts=[{"name": "blur", "strength": 0.2, "parameters": {"mystery": 1.0}}]))
    issues = validate_recipe(recipe, ONTOLOGY)
    assert any("not documented in the ontology" in issue.reason for issue in issues)
    assert recipe.artifacts[0].parameters == {"mystery": 1.0}  # kept, so the error can name it


def test_target_or_private_tokens_in_a_recipe_are_rejected():
    ontology = ONTOLOGY
    assert ontology.leakage_hits('{"note":"siw_mv2 replay"}')
    assert ontology.leakage_hits('{"path":"D:/Dataset/casia"}')
    assert ontology.leakage_hits('{"root":"/home/user/data/work"}')
    assert ontology.leakage_hits(canonical_json(parse_recipe(valid_payload()))) == []


def test_physics_route_requires_implemented_operators():
    recipe = parse_recipe(valid_payload(artifacts=[{"name": "blur", "strength": 0.2}]))
    issues = validate_recipe(recipe, ONTOLOGY, operators=("halftone",))
    assert any(issue.stage == "route" and "missing" in issue.reason for issue in issues)


def test_forbidden_shortcut_consistency():
    recipe = parse_recipe(valid_payload(medium={"family": "display-like", "transparency": 0.0, "roughness": 0.2},
                                        artifacts=[{"name": "moire", "strength": 0.3}],
                                        forbidden_shortcuts=["always_moire"]))
    issues = validate_recipe(recipe, ONTOLOGY)
    assert any(issue.stage == "shortcut" and "only artifact" in issue.reason for issue in issues)


def test_canonical_json_and_hash_are_deterministic():
    first, second = parse_recipe(valid_payload()), parse_recipe(valid_payload())
    assert canonical_json(first) == canonical_json(second)
    assert recipe_hash(first) == recipe_hash(second)
    assert canonical_json(first) == canonical_json(parse_recipe(json.loads(canonical_json(first))))
    assert recipe_hash(parse_recipe(valid_payload(seed=7320))) != recipe_hash(first)


def test_alias_canonicalization_accepts_free_text_then_still_validates():
    payload = valid_payload(regions=["eyes"], artifacts=[{"name": "reflection", "strength": 0.3}],
                            capture={"yaw": 15.0, "illumination": "side", "compression_q": 82, "scale": 1.0,
                                     "motion": 0.1, "defocus": 0.2})
    recipe, issues = validate_payload(payload, ONTOLOGY, canonicalize=True)
    assert recipe is not None and issues == []
    assert recipe.regions == ["left_eye", "right_eye"]
    assert recipe.artifacts[0].name == "specular_reflection" and recipe.capture.illumination == "mixed"


# --- 13-20 compiler and conditioning ----------------------------------------

def test_compiled_graph_is_deterministic():
    recipe = parse_recipe(valid_payload())
    first = compile_recipe(recipe, ONTOLOGY, bank_id=BANK_ID)
    second = compile_recipe(recipe, ONTOLOGY, bank_id=BANK_ID)
    assert first.canonical_json() == second.canonical_json() and first.graph_hash == second.graph_hash
    assert first.compiler_version == COMPILER_VERSION


def test_graph_changes_when_the_recipe_changes():
    base = compile_recipe(parse_recipe(valid_payload()), ONTOLOGY, bank_id=BANK_ID)
    for change in ({"seed": 7320}, {"geometry": {"shape": "flat", "rigidity": 0.4, "coverage": 0.22}},
                   {"regions": ["nose"]}):
        assert compile_recipe(parse_recipe(valid_payload(**change)), ONTOLOGY, bank_id=BANK_ID).graph_hash != base.graph_hash
    assert compile_recipe(parse_recipe(valid_payload()), ONTOLOGY, bank_id="other_bank").graph_hash != base.graph_hash


def test_operator_order_is_stable_and_physically_fixed():
    recipe = parse_recipe(valid_payload(artifacts=[{"name": "blur", "strength": 0.2},
                                                   {"name": "specular_reflection", "strength": 0.2},
                                                   {"name": "texture_smoothing", "strength": 0.2}]))
    graph = compile_recipe(recipe, ONTOLOGY, bank_id=BANK_ID)
    assert graph.operator_names() == ("texture_smoothing", "specular_reflection", "blur")
    assert [node.node_index for node in graph.nodes] == [0, 1, 2]
    assert [node.input_key for node in graph.nodes] == ["image", "image@1", "image@2"]
    assert [node.output_key for node in graph.nodes] == ["image@1", "image@2", "image@3"]
    config = PHYSICS_CONFIG.read_text(encoding="utf-8")
    assert [line.strip("- ").strip() for line in config.splitlines()
            if line.startswith("  - ")][:8] == list(OPERATOR_APPLICATION_ORDER)


def test_operator_seeds_are_deterministic_and_never_use_global_rng():
    graph = compile_recipe(parse_recipe(valid_payload()), ONTOLOGY, bank_id=BANK_ID)
    assert derive_seed(BANK_ID, "R-000042", 7319, "", 0) == graph.nodes[0].seed
    assert graph.node_seed(graph.nodes[0], "sample-a") != graph.node_seed(graph.nodes[0], "sample-b")
    assert graph.node_seed(graph.nodes[0], "sample-a") == graph.node_seed(graph.nodes[0], "sample-a")
    np.random.seed(1234)
    first = local_rng(graph.nodes[0].seed).random(4)
    np.random.seed(4321)
    assert np.array_equal(first, local_rng(graph.nodes[0].seed).random(4))
    source = (ROOT / "src" / "prism_fas" / "synthesis").rglob("*.py")
    for path in source:
        text = path.read_text(encoding="utf-8")
        assert "np.random.seed" not in text and "np.random.rand(" not in text and "random.random()" not in text


def test_conditioning_vector_shape_and_dtype():
    vector = conditioning_vector(parse_recipe(valid_payload()), ONTOLOGY)
    assert vector.shape == (CONDITIONING_DIM,) == (41,) and vector.dtype == np.float32
    assert np.isfinite(vector).all() and CONDITIONING_VERSION == "recipe_conditioning_v1"


def test_conditioning_feature_names_and_order_are_stable():
    names = feature_names(ONTOLOGY)
    assert len(names) == 41 and names[0] == "medium=paper-like" and names[-1] == "route=gpat"
    assert names[5] == "geometry=flat" and names[11] == "region=left_eye" and names[20] == "artifact_strength=halftone"
    assert names[28] == "illumination=front" and names[34] == "capture.yaw_normalized"
    assert feature_names_sha256(ONTOLOGY) == feature_names_sha256(load_ontology(ONTOLOGY_PATH))


def test_conditioning_values_are_normalized_correctly():
    recipe = parse_recipe(valid_payload())
    vector = conditioning_vector(recipe, ONTOLOGY)
    assert vector[1] == 1.0 and vector[0] == 0.0                      # display-like one-hot
    assert vector[7] == 1.0                                            # partial-curved
    assert vector[12] == 1.0 and vector[17] == 1.0                     # right_eye, right_cheek
    assert pytest.approx(float(vector[23]), abs=1e-6) == 0.32          # specular_reflection strength
    assert pytest.approx(float(vector[34]), abs=1e-6) == 15.0 / 45.0
    assert pytest.approx(float(vector[35]), abs=1e-6) == (82 - 30) / 70.0
    assert pytest.approx(float(vector[36]), abs=1e-6) == 2 * (1.0 - 0.75) / 0.5 - 1.0
    assert vector[39] == 1.0 and vector[40] == 1.0                     # physics + gpat
    assert normalize_yaw(90.0, ONTOLOGY) == 1.0 and normalize_yaw(-90.0, ONTOLOGY) == -1.0
    assert normalize_compression(30, ONTOLOGY) == 0.0 and normalize_scale(1.25, ONTOLOGY) == 1.0
    decoded = decode_conditioning(vector, ONTOLOGY)
    assert decoded["dimension"] == 41 and "medium=display-like" in decoded["active"]["medium"]


def test_unknown_category_fails_rather_than_shifting_indices():
    from dataclasses import replace
    shifted = replace(ONTOLOGY, media=("paper-like", "display-like"))
    with pytest.raises(ConditioningError, match="conditioning features, expected 41"):
        feature_names(shifted)
    renamed = replace(ONTOLOGY, media=("paper-like", "screen", "plastic-like", "fabric-like", "reflective-film-like"))
    with pytest.raises(ConditioningError, match="unknown medium category"):
        conditioning_vector(parse_recipe(valid_payload()), renamed)


# --- 21-29 frozen bank -------------------------------------------------------

def test_bank_holds_exactly_128_recipes_with_stable_ids():
    bank = load_bank(BANK_ROOT)
    recipes = bank["recipes"]
    assert len(recipes) == 128 and int(bank["lock"]["recipe_count"]) == 128
    ids = [recipe.recipe_id for recipe in recipes]
    assert ids == [f"R-{index:06d}" for index in range(1, 129)] and len(set(ids)) == 128
    assert int(bank["lock"]["bank_seed"]) == 20260806 and bank["lock"]["status"] == "frozen"


def test_bank_has_no_duplicate_recipe_or_graph_hashes():
    bank = load_bank(BANK_ROOT)
    digests = [recipe_hash(recipe) for recipe in bank["recipes"]]
    assert len(set(digests)) == 128
    graphs = [compile_recipe(recipe, bank["ontology"], bank_id=bank["bank_id"]) for recipe in bank["recipes"]]
    assert len({graph.graph_hash for graph in graphs}) == 128


def test_every_bank_recipe_validates_and_compiles():
    bank = load_bank(BANK_ROOT)
    report = validate_recipes(bank["recipes"], bank["ontology"])
    assert report["passed"] and report["issue_count"] == 0
    for recipe in bank["recipes"]:
        graph = compile_recipe(recipe, bank["ontology"], bank_id=bank["bank_id"])
        assert graph.nodes and graph.conditioning_dimension == 41


def test_bank_meets_required_categorical_coverage_and_diversity():
    bank = load_bank(BANK_ROOT)
    audit = bank_audit(bank["recipes"], bank["ontology"])
    assert audit["passed"] and audit["coverage"]["required_all_met"]
    required = audit["coverage"]["required"]
    assert all(required.values()) and required["physics_route_in_every_recipe"]
    assert audit["diversity"]["method"] == "offline_tfidf_cosine_v1"
    assert audit["diversity"]["external_text_model"] is False and audit["diversity"]["network_access"] is False
    assert audit["diversity"]["text_tfidf"]["max_cosine"] <= 0.98
    assert audit["diversity"]["unique_recipe_hashes"] == 128


def test_bank_lock_hashes_validate_against_the_files_on_disk():
    report = validate_bank(BANK_ROOT)
    assert report["passed"] and report["errors"] == []
    lock = json.loads((BANK_ROOT / "BANK_LOCK.json").read_text(encoding="utf-8"))
    assert lock["bank_content_identity_sha256"] == report["bank_content_identity_sha256"]
    assert set(lock["file_sha256"]) == {name for name in BANK_FILES if name != "BANK_LOCK.json"}
    assert lock["conditioning"] == {"version": "recipe_conditioning_v1", "dimension": 41,
                                    "feature_names_sha256": feature_names_sha256(ONTOLOGY)}


def test_bank_lock_carries_no_timestamp_or_machine_path():
    text = (BANK_ROOT / "BANK_LOCK.json").read_text(encoding="utf-8")
    for marker in ("created_at", "build_seconds", "timestamp", "D:/", "C:\\", "/home/", "/Users/"):
        assert marker not in text
    for name in BANK_FILES:
        assert "\r\n" not in (BANK_ROOT / name).read_text(encoding="utf-8", newline="")


def test_rebuilding_the_frozen_bank_is_byte_identical_and_writes_nothing():
    before = {name: (BANK_ROOT / name).read_bytes() for name in BANK_FILES}
    result = build_bank(BANK_ROOT, ONTOLOGY_PATH, BANK_CONFIG)
    assert result["status"] == "reused" and result["written"] == []
    assert {name: (BANK_ROOT / name).read_bytes() for name in BANK_FILES} == before


def test_a_destination_with_a_different_lock_is_never_overwritten(tmp_path):
    for name in BANK_FILES:
        (tmp_path / name).write_text((BANK_ROOT / name).read_text(encoding="utf-8"), encoding="utf-8", newline="")
    lock = json.loads((tmp_path / "BANK_LOCK.json").read_text(encoding="utf-8"))
    lock["bank_content_identity_sha256"] = "0" * 64
    (tmp_path / "BANK_LOCK.json").write_text(json.dumps(lock, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="")
    with pytest.raises(BankError, match="banks are immutable"):
        build_bank(tmp_path, ONTOLOGY_PATH, BANK_CONFIG)
    assert json.loads((tmp_path / "BANK_LOCK.json").read_text(encoding="utf-8"))["bank_content_identity_sha256"] == "0" * 64


def test_bank_declares_no_external_llm_or_network_dependency():
    generator = json.loads((BANK_ROOT / "generator.json").read_text(encoding="utf-8"))
    assert generator["provider"] == "deterministic_local"
    assert generator["model_id"] == "deterministic-source-only-recipe-generator"
    assert generator["revision"] == "m7-v1"
    assert generator["external_llm_invoked"] is False and GENERATOR_EXTERNAL_LLM_INVOKED is False
    assert generator["network_access"] is False and generator["credential_used"] is False
    source = (ROOT / "src" / "prism_fas" / "recipes").rglob("*.py")
    for path in source:
        text = path.read_text(encoding="utf-8")
        for marker in ("requests", "urllib.request", "httpx", "openai", "anthropic", "api_key", "socket"):
            assert marker not in text, f"{path.name} references {marker}"


def test_prompt_is_a_frozen_contract_and_leaks_nothing():
    prompt = (BANK_ROOT / "prompt.txt").read_text(encoding="utf-8")
    assert prompt == render_prompt(load_ontology(BANK_ROOT / "ontology.yaml"))
    assert ONTOLOGY.leakage_hits(prompt) == []
    assert "schema_version" in prompt and ONTOLOGY.sha256 in prompt


def test_bank_recipes_carry_only_canonical_values():
    bank = load_bank(BANK_ROOT)
    for recipe in bank["recipes"]:
        assert recipe.medium.family in ONTOLOGY.media and recipe.geometry.shape in ONTOLOGY.geometry_shapes
        assert all(name in ONTOLOGY.regions for name in recipe.regions)
        assert all(spec.name in ONTOLOGY.artifacts for spec in recipe.artifacts)
        assert recipe.capture.illumination in ONTOLOGY.illumination
        assert "physics" in recipe.generator_route
        assert ONTOLOGY.leakage_hits(canonical_json(recipe)) == []


# --- 30-34 region masks ------------------------------------------------------

def _builder(**overrides):
    parsing, landmarks, bbox = fixture_arrays()
    kwargs = {"height": 64, "width": 64, "parsing": parsing, "landmarks": landmarks, "bbox": bbox, "crop_box": None}
    kwargs.update(overrides)
    return RegionMaskBuilder(**kwargs)


def test_all_nine_region_masks_are_non_empty_on_the_fixture():
    builder = _builder()
    for name in REGION_ORDER:
        mask, source = builder.region(name)
        assert mask.any(), f"{name} is empty"
        assert source


def test_parsing_first_then_geometry_fallback():
    builder = _builder()
    assert builder.region("left_eye")[1] == "parsing"
    parsing, _, _ = fixture_arrays()
    parsing[parsing == PARSING_LABELS["left_eye"]] = PARSING_LABELS["skin"]      # eye class disappears
    fallback = _builder(parsing=parsing)
    mask, source = fallback.region("left_eye")
    assert source == "landmark_geometry" and mask.any()
    without = _builder(parsing=None)
    assert without.region("nose")[1] == "landmark_geometry"
    assert without.region("face_boundary")[1].startswith("bbox_geometry")
    tiny = fixture_arrays()[0].copy()
    tiny[tiny == PARSING_LABELS["right_eye"]] = PARSING_LABELS["skin"]
    tiny[26, 40:40 + MIN_PARSING_PIXELS - 1] = PARSING_LABELS["right_eye"]        # below the usable threshold
    assert _builder(parsing=tiny).region("right_eye")[1] == "landmark_geometry"


def test_boundary_and_context_masks_obey_the_crop_bounds():
    builder = _builder()
    face = builder.face_mask()[0]
    boundary = builder.region("face_boundary")[0]
    context = builder.region("context")[0]
    assert boundary.shape == (64, 64) and context.shape == (64, 64)
    assert not (context & face).any(), "context must sit outside the face mask"
    assert context.any() and boundary.any()
    assert int(context.sum()) < 64 * 64


def test_requested_coverage_is_deterministic_and_measured_on_the_region_union():
    builder = _builder()
    first = builder.build(["nose", "mouth"], geometry_shape="flat", coverage=0.4, seed=99)
    second = builder.build(["nose", "mouth"], geometry_shape="flat", coverage=0.4, seed=99)
    assert first.mask_hash == second.mask_hash
    assert abs(first.achieved_coverage - 0.4) <= 0.05
    assert first.support_pixels() <= first.requested_pixels()
    other = builder.build(["nose", "mouth"], geometry_shape="flat", coverage=0.4, seed=100)
    assert other.mask_hash != first.mask_hash
    whole = builder.build(["nose", "mouth"], geometry_shape="flat", coverage=1.0, seed=99)
    assert whole.achieved_coverage == 1.0 and whole.support_pixels() == whole.requested_pixels()


def test_mask_values_are_exactly_zero_or_one_and_never_silently_empty():
    result = _builder().build(["left_eye"], geometry_shape="flat", coverage=1.0, seed=5)
    for mask in [result.requested_region_mask, result.operator_support_mask, *result.per_region_masks.values()]:
        assert mask.dtype == np.float32 and mask.shape[0] == 1
        assert set(np.unique(mask).tolist()) <= {0.0, 1.0}
    with pytest.raises(MaskBuildError):
        _builder().build([], geometry_shape="flat", coverage=1.0, seed=5)
    with pytest.raises(MaskBuildError, match="unknown canonical region"):
        _builder().build(["chin"], geometry_shape="flat", coverage=1.0, seed=5)


def test_landmarks_are_mapped_through_the_crop_box():
    parsing, landmarks, bbox = fixture_arrays()
    scaled = RegionMaskBuilder(height=64, width=64, parsing=parsing, landmarks=landmarks * 2.0 + 100.0,
                               bbox=bbox * 2.0 + 100.0, crop_box=np.asarray([100.0, 100.0, 228.0, 228.0], np.float32))
    direct = _builder()
    assert scaled.landmark("nose") == pytest.approx(direct.landmark("nose"), abs=1e-4)
    assert scaled.face_box() == pytest.approx(direct.face_box(), abs=1e-4)


# --- 35-42 physics operators and engine --------------------------------------

@pytest.mark.parametrize("name", sorted(OPERATOR_CLASSES))
def test_operator_is_deterministic_finite_and_in_range(name):
    image, mask = fixture_image(), full_mask()
    operator = build_operator(name)
    parameters = {"medium_roughness": 0.3, "medium_transparency": 0.1, "capture_motion": 0.3,
                  "capture_defocus": 0.1, "capture_scale": 1.0, "capture_yaw": 10.0}
    first = operator.apply(image, mask, 0.4, CAPTURE, local_rng(11), dict(parameters), seed=11)
    second = operator.apply(image, mask, 0.4, CAPTURE, local_rng(11), dict(parameters), seed=11)
    assert np.array_equal(first.image, second.image)
    assert first.image.dtype == np.float32 and np.isfinite(first.image).all()
    assert first.image.min() >= 0.0 and first.image.max() <= 1.0
    assert first.trace["operator"] == name and first.operator_seed == 11
    assert not np.array_equal(first.image, image), f"{name} produced no change at strength 0.4"


@pytest.mark.parametrize("name", sorted(OPERATOR_CLASSES))
def test_operator_never_alters_pixels_outside_its_declared_support(name):
    image = fixture_image()
    mask = np.zeros((1, 64, 64), dtype=np.float32)
    mask[0, 20:40, 20:40] = 1.0
    result = build_operator(name).apply(image, mask, 0.35, CAPTURE, local_rng(5),
                                        {"medium_roughness": 0.2, "capture_motion": 0.2, "capture_defocus": 0.4}, seed=5)
    support = np.asarray(result.actual_support_mask)[0].astype(bool)
    outside = ~support
    assert float(np.abs(result.image - image)[:, outside].max()) == 0.0
    assert float(np.asarray(result.strength_map)[0][outside].max()) == 0.0
    if name == "boundary_inconsistency":
        assert support.sum() > 0 and not np.array_equal(support, mask[0].astype(bool))


@pytest.mark.parametrize("name", sorted(OPERATOR_CLASSES))
def test_zero_strength_operator_returns_the_input_unchanged(name):
    image, mask = fixture_image(), full_mask()
    result = build_operator(name).apply(image, mask, 0.0, CAPTURE, local_rng(3),
                                        {"medium_roughness": 0.5, "capture_motion": 0.5, "capture_defocus": 0.5}, seed=3)
    assert np.array_equal(result.image, image)
    assert float(np.asarray(result.strength_map).max()) == 0.0
    assert np.isfinite(result.image).all()


@pytest.mark.parametrize("name", sorted(OPERATOR_CLASSES))
def test_a_different_seed_changes_the_procedural_output(name):
    image, mask = fixture_image(), full_mask()
    parameters = {"medium_roughness": 0.3, "capture_motion": 0.4, "capture_defocus": 0.1, "capture_yaw": 5.0}
    first = build_operator(name).apply(image, mask, 0.4, CAPTURE, local_rng(21), dict(parameters), seed=21)
    second = build_operator(name).apply(image, mask, 0.4, CAPTURE, local_rng(22), dict(parameters), seed=22)
    assert not np.array_equal(first.image, second.image), f"{name} ignored its local seed"


def test_all_eight_operators_exist_and_are_ordered():
    assert len(OPERATOR_NAMES) == 8 and set(OPERATOR_NAMES) == set(OPERATOR_APPLICATION_ORDER)
    assert set(OPERATOR_NAMES) == {"halftone", "pixel_grid", "moire", "specular_reflection", "texture_smoothing",
                                   "color_shift", "boundary_inconsistency", "blur"}


def test_operator_rejects_an_invalid_image_or_mask():
    operator = build_operator("blur")
    with pytest.raises(SynthesisError, match="dtype"):
        operator.apply(fixture_image().astype(np.float64), full_mask(), 0.2, CAPTURE, local_rng(1), {})
    with pytest.raises(SynthesisError, match="outside"):
        operator.apply(fixture_image() + 2.0, full_mask(), 0.2, CAPTURE, local_rng(1), {})
    with pytest.raises(SynthesisError, match="exactly 0 or 1"):
        operator.apply(fixture_image(), full_mask() * 0.5, 0.2, CAPTURE, local_rng(1), {})


def _engine_result(recipe_payload=None, seed_sample="sample-a"):
    parsing, landmarks, bbox = fixture_arrays()
    recipe = parse_recipe(recipe_payload or valid_payload())
    graph = compile_recipe(recipe, ONTOLOGY, bank_id=BANK_ID)
    return PhysicsEngine().apply(fixture_image(), parsing, landmarks, bbox, graph, seed_sample), graph


def test_engine_outside_the_exact_mask_error_is_exactly_zero():
    result, _ = _engine_result()
    image = fixture_image()
    outside = np.asarray(result.exact_edit_mask)[0] < 0.5
    assert float(np.abs(result.synthetic_image - image)[:, outside].max()) == 0.0
    assert result.trace["outside_mask_max_abs_error"] == 0.0
    assert result.changed_pixels() > 0 and result.trace["changed_pixels"] > 0


def test_engine_strength_map_is_zero_outside_the_mask_and_in_range():
    result, _ = _engine_result()
    strength = np.asarray(result.artifact_strength_map)
    outside = np.asarray(result.exact_edit_mask)[0] < 0.5
    assert float(strength[0][outside].max()) == 0.0
    assert strength.dtype == np.float32 and strength.min() >= 0.0 and strength.max() <= 1.0
    assert float(strength.max()) > 0.0


def test_engine_output_contract_and_trace_carry_the_hashes():
    result, graph = _engine_result()
    assert result.synthetic_image.shape == (3, 64, 64) and result.synthetic_image.dtype == np.float32
    assert set(np.unique(np.asarray(result.exact_edit_mask)).tolist()) <= {0.0, 1.0}
    assert result.recipe_id == "R-000042" and result.graph_hash == graph.graph_hash
    assert result.recipe_hash == graph.recipe_hash and result.sample_id == "sample-a"
    assert set(result.output_hashes) == {"input_image_sha256", "synthetic_image_sha256", "exact_edit_mask_sha256",
                                         "requested_region_mask_sha256", "artifact_strength_map_sha256"}
    assert [entry["operator"] for entry in result.trace["operators"]] == list(graph.operator_names())
    assert all(entry["operator_seed"] for entry in result.trace["operators"])
    assert set(result.per_operator_support_masks) == set(graph.operator_names())


def test_engine_is_deterministic_and_sample_dependent():
    first, _ = _engine_result()
    second, _ = _engine_result()
    assert np.array_equal(first.synthetic_image, second.synthetic_image)
    other, _ = _engine_result(seed_sample="sample-b")
    assert other.output_hashes["synthetic_image_sha256"] != first.output_hashes["synthetic_image_sha256"]


def test_engine_requires_the_physics_route():
    with pytest.raises(CompileError, match="physics route is required"):
        compile_recipe(parse_recipe(valid_payload(generator_route=["gpat"])), ONTOLOGY, bank_id=BANK_ID)


# --- 43-48 real audit contracts (ignored report artifacts) -------------------

def _report(name):
    path = REPORTS / name
    if not path.is_file():
        pytest.skip(f"{name} missing; run: python scripts/m7_physics_audit.py")
    return json.loads(path.read_text(encoding="utf-8"))


def test_real_preview_used_source_train_live_samples_only():
    physics = _report("physics_audit.json")
    isolation = _report("source_isolation_audit.json")
    assert physics["checks"]["only_source_train_live"] and physics["checks"]["no_forbidden_split"]
    assert isolation["splits_used"] == ["source_train"] and isolation["labels_used"] == ["live"]
    assert isolation["manifests_opened"] == ["manifests/source_train.parquet"]


def test_real_audit_used_no_source_dev_and_no_target_data():
    physics = _report("physics_audit.json")
    isolation = _report("source_isolation_audit.json")
    assert physics["source_dev_inputs"] == 0 and physics["target_inputs"] == 0
    assert isolation["source_dev_inputs"] == 0 and isolation["target_inputs"] == 0
    assert isolation["target_metadata_fields"] == [] and isolation["forbidden_token_hits"] == {}
    assert isolation["passed"]


def test_real_audit_meets_the_64_preview_contract():
    physics = _report("physics_audit.json")
    assert physics["preview_rows"] == 64 and physics["unique_samples"] == 32
    assert physics["input_samples_per_dataset"] == {"casia_fasd": 16, "msu_mfsd": 16}
    assert physics["outside_mask_max_abs_error"] == 0.0
    assert len(physics["operators_exercised"]) == 8 and len(physics["regions_exercised"]) == 9
    assert len(physics["media_exercised"]) == 5 and len(physics["geometry_exercised"]) == 6
    assert len(physics["illumination_exercised"]) == 6
    assert physics["empty_mask_samples"] == [] and physics["unchanged_previews"] == []
    assert physics["device"] == "cpu" and physics["modal_used"] is False and physics["gpu_used"] is False
    assert physics["passed"]
    rows = [json.loads(line) for line in (REPORTS / "preview_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 64
    assert all(row["outside_mask_max_abs_error"] == 0.0 and row["exact_edit_pixels"] > 0 for row in rows)
    assert all(row["changed_pixels"] > 0 and row["source_crop_sha256"] and row["source_prior_sha256"] for row in rows)


def test_real_determinism_rerun_has_zero_mismatch():
    determinism = _report("determinism_audit.json")
    assert determinism["mismatch_count"] == 0 and determinism["mismatches"] == []
    for key in ("sample_ids_identical", "recipe_ids_identical", "graph_hashes_identical", "image_hashes_identical",
                "mask_hashes_identical", "strength_map_hashes_identical", "primary_vs_run_a"):
        assert determinism[key] is True, key
    assert determinism["seed_sensitivity"]["passed"] and determinism["seed_sensitivity"]["image_hash_changed"]
    assert determinism["frozen_bank_rebuild"]["status"] == "reused"
    assert determinism["frozen_bank_rebuild"]["lock_unchanged"] and determinism["frozen_bank_rebuild"]["files_written"] == []
    assert determinism["passed"]


def test_real_audit_left_the_package_identity_unchanged():
    lock = json.loads((ROOT / "data" / "processed" / "prism_data_v1_m3b" / "PACKAGE_LOCK.json").read_text(encoding="utf-8")) \
        if (ROOT / "data" / "processed" / "prism_data_v1_m3b" / "PACKAGE_LOCK.json").is_file() else None
    if lock is None: pytest.skip("M3B package not present on this machine")
    assert lock["content_identity_sha256"] == "b1cf29b69a165ed5d9e074fc8127c17fbf057723edf9e272048ec3a564eb9dc6"
    assert lock["status"] == "validated" and lock["package_id"] == "prism_data_v1_m3b"
    bank_report = _report("recipe_bank_audit.json")
    assert bank_report["passed"] and bank_report["recipe_count"] == 128
    compile_report = _report("compile_audit.json")
    assert compile_report["compiled"] == 128 and compile_report["unique_graph_hashes"] == 128
    assert compile_report["conditioning_dimension"] == 41


def test_cli_dry_run_writes_nothing(tmp_path):
    from prism_fas.synthesis.audit import run_audit
    package = ROOT / "data" / "processed" / "prism_data_v1_m3b"
    if not package.is_dir(): pytest.skip("M3B package not present on this machine")
    result = run_audit(package, BANK_ROOT, PHYSICS_CONFIG, tmp_path / "out", dry_run=True)
    assert result["status"] == "dry_run" and result["written"] == []
    assert result["planned_pairs"] == 64 and result["selected_samples"] == 32
    assert not (tmp_path / "out").exists()
    plan = build_bank(tmp_path / "bank", ONTOLOGY_PATH, BANK_CONFIG, dry_run=True)
    assert plan["status"] == "dry_run" and plan["written"] == [] and not (tmp_path / "bank").exists()


def test_recipe_description_and_coverage_helpers_are_pure():
    bank = load_bank(BANK_ROOT)
    description = recipe_description(bank["recipes"][0])
    assert description == recipe_description(bank["recipes"][0])
    assert ONTOLOGY.leakage_hits(description) == []
    table = coverage_table(bank["recipes"], bank["ontology"])
    assert sum(table["media"].values()) == 128 and table["required_all_met"]
    assert diversity_audit(bank["recipes"], bank["ontology"])["passed"]
