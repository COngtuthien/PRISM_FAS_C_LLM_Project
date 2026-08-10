"""C1 groups A (config) and B (schema).

The contract these tests defend: a typo cannot silently change scientific
behaviour, and nothing the model emits is accepted unless it satisfies the
strict schema, the ontology, the ranges and the compatibility rules.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from prism_fas.llm.config import (
    FORBIDDEN_SAMPLING_FIELDS,
    LLMConfigError,
    LLMProviderConfig,
    parse_llm_config,
    provider_config_identity,
)
from prism_fas.llm.json_schema import candidate_json_schema, json_schema_identity
from prism_fas.llm.pipeline import CandidateOutcome

from c1_helpers import envelope


def _base_config() -> dict[str, Any]:
    return {"provider": "gemini", "model_id": "gemini-3.6-flash"}


# ===================== A. CONFIG ==========================================

def test_valid_config_is_accepted(config: LLMProviderConfig):
    assert config.provider == "gemini"
    assert config.model_id == "gemini-3.6-flash"
    assert config.thinking_level == "medium"
    assert config.response_mime_type == "application/json"
    assert config.api_surface == "interactions"
    assert config.api_key_env == "GEMINI_API_KEY"


def test_unknown_config_field_is_rejected():
    with pytest.raises(LLMConfigError):
        parse_llm_config({**_base_config(), "temperature": 0.7})


def test_unknown_nested_config_field_is_rejected():
    with pytest.raises(LLMConfigError):
        parse_llm_config({**_base_config(), "retry": {"semantic_max_retries": 2, "jitter": True}})


def test_unsupported_provider_is_rejected():
    with pytest.raises(LLMConfigError):
        parse_llm_config({**_base_config(), "provider": "openai"})


def test_unsupported_api_surface_is_rejected():
    with pytest.raises(LLMConfigError):
        parse_llm_config({**_base_config(), "api_surface": "generate_content"})


def test_unsupported_thinking_level_is_rejected():
    with pytest.raises(LLMConfigError):
        parse_llm_config({**_base_config(), "thinking_level": "ultra"})


def test_unsupported_response_mime_type_is_rejected():
    with pytest.raises(LLMConfigError):
        parse_llm_config({**_base_config(), "response_mime_type": "text/plain"})


@pytest.mark.parametrize("field", ["tools_enabled", "grounding_enabled", "url_context_enabled",
                                   "code_execution_enabled", "file_search_enabled",
                                   "image_input_enabled", "audio_input_enabled",
                                   "video_input_enabled", "store_interaction"])
def test_leakage_capabilities_cannot_be_switched_on(field: str):
    """Each is typed Literal[False]: enabling one is a config error, not a flag."""
    with pytest.raises(LLMConfigError):
        parse_llm_config({**_base_config(), field: True})


def test_billing_can_never_be_auto_enabled():
    with pytest.raises(LLMConfigError):
        parse_llm_config({**_base_config(), "quota": {"auto_enable_paid": True}})


def test_quota_exhaustion_action_is_fixed():
    with pytest.raises(LLMConfigError):
        parse_llm_config({**_base_config(), "quota": {"on_quota_exhausted": "switch_provider"}})


def test_retry_budget_is_bounded():
    with pytest.raises(LLMConfigError):
        parse_llm_config({**_base_config(), "retry": {"semantic_max_retries": 99}})


def test_api_key_env_must_name_a_variable_not_carry_a_key():
    with pytest.raises(LLMConfigError):
        parse_llm_config({**_base_config(), "api_key_env": "AIza" + "x" * 35})
    with pytest.raises(LLMConfigError):
        parse_llm_config({**_base_config(), "api_key_env": "gemini_api_key"})


def test_config_identity_changes_with_scientific_fields_only(config: LLMProviderConfig):
    baseline = provider_config_identity(config)
    # Operational: must NOT move the identity.
    operational = config.model_copy(update={"request_timeout_seconds": 300.0})
    assert provider_config_identity(operational) == baseline
    # Scientific: MUST move the identity.
    for update in ({"model_id": "gemini-3.5-flash"}, {"thinking_level": "high"},
                   {"allow_ontology_aliases": True}, {"max_output_tokens": 4096}):
        assert provider_config_identity(config.model_copy(update=update)) != baseline, update


def test_forbidden_sampling_fields_are_named_and_absent_from_the_config(config: LLMProviderConfig):
    assert set(FORBIDDEN_SAMPLING_FIELDS) == {"temperature", "top_p", "top_k"}
    for field in FORBIDDEN_SAMPLING_FIELDS:
        assert field not in type(config).model_fields
        assert field not in config.public_summary()


def test_config_summary_never_carries_a_credential(config: LLMProviderConfig):
    blob = json.dumps(config.public_summary())
    assert "AIza" not in blob
    # It names the variable; it must not contain a value for it.
    assert '"api_key_env": "GEMINI_API_KEY"' in blob
    assert "GEMINI_API_KEY=" not in blob


# ===================== B. SCHEMA ==========================================

def test_valid_recipe_passes(planner, valid_candidate):
    result = planner.validate_response(envelope(valid_candidate), slot_id="s", recipes_requested=1)
    assert result.all_accepted
    assert result.candidates[0].outcome is CandidateOutcome.ACCEPTED
    assert result.candidates[0].recipe_identity


def test_malformed_json_is_rejected(planner):
    result = planner.validate_response("Sure! Here is the recipe: {", slot_id="s",
                                       recipes_requested=1)
    assert not result.all_accepted
    assert result.response_issues[0]["stage"] == "json_parsing"
    assert result.candidates == []


def test_markdown_fenced_json_is_rejected(planner, valid_candidate):
    """Prose-wrapped JSON is not the scientific path: it is a contract failure."""
    fenced = "```json\n" + envelope(valid_candidate) + "\n```"
    result = planner.validate_response(fenced, slot_id="s", recipes_requested=1)
    assert not result.all_accepted
    assert result.response_issues[0]["stage"] == "json_parsing"


def test_missing_required_key_is_rejected(planner, valid_candidate):
    del valid_candidate["capture"]
    result = planner.validate_response(envelope(valid_candidate), slot_id="s", recipes_requested=1)
    assert result.candidates[0].outcome is CandidateOutcome.REJECTED_SCHEMA


def test_extra_unknown_key_is_rejected(planner, valid_candidate):
    valid_candidate["confidence"] = 0.9
    result = planner.validate_response(envelope(valid_candidate), slot_id="s", recipes_requested=1)
    assert result.candidates[0].outcome is CandidateOutcome.REJECTED_SCHEMA


def test_unknown_top_level_envelope_key_is_rejected(planner, valid_candidate):
    payload = json.dumps({"recipes": [valid_candidate], "notes": "hope this helps"})
    result = planner.validate_response(payload, slot_id="s", recipes_requested=1)
    assert not result.all_accepted
    assert any(issue["stage"] == "envelope_schema" for issue in result.response_issues)


def test_wrong_recipe_count_is_rejected(planner, valid_candidate, second_valid_candidate):
    result = planner.validate_response(envelope(valid_candidate, second_valid_candidate),
                                       slot_id="s", recipes_requested=1)
    assert not result.all_accepted
    assert any("expected exactly 1" in issue["reason"] for issue in result.response_issues)


@pytest.mark.parametrize("mutation,expected", [
    ({"artifacts": [{"name": "rainbow_glow", "strength": 0.3}]}, "unknown artifact"),
    ({"regions": ["third_eye"]}, "unknown region"),
    ({"medium": {"family": "hologram-like", "roughness": 0.5, "transparency": 0.1}}, "unknown medium"),
    ({"geometry": {"shape": "fractal", "rigidity": 0.4, "coverage": 0.3}}, "unknown geometry"),
    ({"capture": {"compression_q": 42, "defocus": 0.7, "illumination": "backlit",
                  "motion": 0.4, "scale": 0.8, "yaw": 30.5}}, "unknown illumination"),
    ({"generator_route": ["diffusion"]}, "unknown route"),
    ({"schema_version": "2.0"}, "wrong schema version"),
])
def test_unknown_enum_values_are_rejected(planner, valid_candidate, mutation, expected):
    valid_candidate.update(mutation)
    result = planner.validate_response(envelope(valid_candidate), slot_id="s", recipes_requested=1)
    assert result.candidates[0].outcome is CandidateOutcome.REJECTED_SCHEMA, expected


def test_severity_out_of_range_is_rejected(planner, valid_candidate):
    valid_candidate["artifacts"] = [{"name": "halftone", "strength": 1.7}]
    result = planner.validate_response(envelope(valid_candidate), slot_id="s", recipes_requested=1)
    assert result.candidates[0].outcome is CandidateOutcome.REJECTED_SCHEMA


def test_severity_outside_the_operator_safe_band_is_rejected(planner, ontology, valid_candidate):
    """Inside [0,1] so the structural schema passes, but outside the ontology's
    operator-specific safe band, so the ontology stage must catch it."""
    band = ontology.strength_range("halftone")
    outside = band.maximum + (1.0 - band.maximum) / 2 if band.maximum < 1.0 else band.minimum / 2
    valid_candidate["artifacts"] = [{"name": "halftone", "strength": round(outside, 4)}]
    result = planner.validate_response(envelope(valid_candidate), slot_id="s", recipes_requested=1)
    assert result.candidates[0].outcome in (CandidateOutcome.REJECTED_ONTOLOGY,
                                            CandidateOutcome.REJECTED_SCHEMA)


def test_total_severity_budget_is_enforced(planner, valid_candidate):
    valid_candidate["artifacts"] = [{"name": "halftone", "strength": 0.6},
                                    {"name": "color_shift", "strength": 0.6}]
    result = planner.validate_response(envelope(valid_candidate), slot_id="s", recipes_requested=1)
    assert result.candidates[0].outcome in (CandidateOutcome.REJECTED_ONTOLOGY,
                                            CandidateOutcome.REJECTED_SCHEMA)


def test_too_many_regions_is_rejected(planner, valid_candidate):
    valid_candidate["regions"] = ["left_eye", "right_eye", "nose", "mouth"]
    result = planner.validate_response(envelope(valid_candidate), slot_id="s", recipes_requested=1)
    assert result.candidates[0].outcome is CandidateOutcome.REJECTED_SCHEMA


def test_duplicate_region_within_a_recipe_is_rejected(planner, valid_candidate):
    valid_candidate["regions"] = ["left_eye", "left_eye"]
    result = planner.validate_response(envelope(valid_candidate), slot_id="s", recipes_requested=1)
    assert result.candidates[0].outcome is CandidateOutcome.REJECTED_SCHEMA


def test_regions_out_of_canonical_order_are_rejected(planner, valid_candidate):
    valid_candidate["regions"] = ["right_eye", "left_eye"]
    result = planner.validate_response(envelope(valid_candidate), slot_id="s", recipes_requested=1)
    assert result.candidates[0].outcome is CandidateOutcome.REJECTED_SCHEMA


def test_seed_out_of_range_is_rejected(planner, valid_candidate):
    valid_candidate["seed"] = -1
    result = planner.validate_response(envelope(valid_candidate), slot_id="s", recipes_requested=1)
    assert result.candidates[0].outcome is CandidateOutcome.REJECTED_SCHEMA


def test_candidate_that_is_not_an_object_is_rejected(planner):
    result = planner.validate_response(json.dumps({"recipes": ["a recipe"]}),
                                       slot_id="s", recipes_requested=1)
    assert result.candidates[0].outcome is CandidateOutcome.REJECTED_ENVELOPE


# --- the request-side JSON Schema ----------------------------------------

def test_json_schema_is_built_from_the_ontology(ontology):
    schema = candidate_json_schema(ontology, recipes_requested=32)
    item = schema["properties"]["recipes"]["items"]
    assert schema["properties"]["recipes"]["minItems"] == 32
    assert schema["properties"]["recipes"]["maxItems"] == 32
    assert item["properties"]["medium"]["properties"]["family"]["enum"] == list(ontology.media)
    assert item["properties"]["regions"]["items"]["enum"] == list(ontology.regions)
    assert item["properties"]["artifacts"]["items"]["properties"]["name"]["enum"] == list(ontology.artifacts)
    assert item["additionalProperties"] is False


def test_json_schema_never_asks_the_model_for_system_owned_fields(ontology):
    schema = candidate_json_schema(ontology, recipes_requested=1)
    properties = schema["properties"]["recipes"]["items"]["properties"]
    for owned in ("recipe_id", "recipe_hash", "provenance", "validation"):
        assert owned not in properties


def test_json_schema_identity_is_stable_and_sensitive(ontology):
    first = json_schema_identity(candidate_json_schema(ontology, recipes_requested=32))
    again = json_schema_identity(candidate_json_schema(ontology, recipes_requested=32))
    other = json_schema_identity(candidate_json_schema(ontology, recipes_requested=31))
    assert first == again
    assert first != other
