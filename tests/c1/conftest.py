"""Shared fixtures for the C1 contract tests.

Every test here is offline. No fixture constructs a real provider client, reads a
credential, or opens a socket. `no_network` is autouse, so a test that tried to
reach the internet would fail loudly rather than quietly costing money.
"""
from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import pytest

from prism_fas.llm.config import LLMProviderConfig, load_llm_config
from prism_fas.llm.contracts import GenerationRequest
from prism_fas.llm.json_schema import candidate_json_schema, json_schema_identity
from prism_fas.llm.pipeline import RecipePlanner
from prism_fas.llm.prompt import build_generation_prompt, load_prompt_template
from prism_fas.llm.providers import MockRecipeProvider
from prism_fas.recipes.ontology import Ontology, load_ontology

REPO = Path(__file__).resolve().parents[2]
ONTOLOGY_PATH = REPO / "configs" / "recipes" / "ontology_m7.yaml"
LLM_CONFIG_PATH = REPO / "configs" / "version_c" / "llm" / "c1_gemini_provider.yaml"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make an accidental outbound connection impossible for the whole suite."""

    def blocked(*args: Any, **kwargs: Any):
        raise AssertionError("a C1 unit test attempted a network connection; "
                            "the offline contract forbids it")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture(autouse=True)
def no_ambient_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's real key must not change how the suite behaves."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


@pytest.fixture(scope="session")
def ontology() -> Ontology:
    return load_ontology(ONTOLOGY_PATH)


@pytest.fixture(scope="session")
def config() -> LLMProviderConfig:
    return load_llm_config(LLM_CONFIG_PATH)


@pytest.fixture
def planner(config: LLMProviderConfig, ontology: Ontology) -> RecipePlanner:
    return RecipePlanner(provider=MockRecipeProvider(), config=config, ontology=ontology,
                         sleep=lambda _seconds: None)


@pytest.fixture
def valid_candidate() -> dict[str, Any]:
    """A candidate that satisfies schema, ontology, ranges and compatibility.

    Deliberately shaped like the frozen Version-B bank so the C1 contract is
    proven against a recipe form the inherited compiler already accepts.
    """
    return {
        "schema_version": "1.1",
        "medium": {"family": "paper-like", "roughness": 0.6701, "transparency": 0.4417},
        "geometry": {"coverage": 0.4873, "rigidity": 0.2734, "shape": "flat"},
        "regions": ["left_eye"],
        "artifacts": [{"name": "halftone", "strength": 0.5864}],
        "capture": {"compression_q": 42, "defocus": 0.757, "illumination": "front",
                    "motion": 0.466, "scale": 0.808, "yaw": 30.5},
        "forbidden_shortcuts": ["always_moire"],
        "generator_route": ["physics", "gpat"],
        "seed": 717714041,
    }


@pytest.fixture
def second_valid_candidate() -> dict[str, Any]:
    """A different valid recipe, for batch and duplicate tests."""
    return {
        "schema_version": "1.1",
        "medium": {"family": "display-like", "roughness": 0.25, "transparency": 0.0},
        "geometry": {"coverage": 0.22, "rigidity": 0.4, "shape": "partial-curved"},
        "regions": ["right_eye", "right_cheek"],
        "artifacts": [{"name": "specular_reflection", "strength": 0.32},
                      {"name": "texture_smoothing", "strength": 0.18}],
        "capture": {"compression_q": 82, "defocus": 0.1, "illumination": "left",
                    "motion": 0.05, "scale": 1.0, "yaw": 15.0},
        "forbidden_shortcuts": [],
        "generator_route": ["physics"],
        "seed": 7319,
    }


@pytest.fixture
def make_request(ontology: Ontology, config: LLMProviderConfig):
    """Builds a well-formed, target-free request."""

    def _make(slot_id: str = "c1-slot-0001", recipes_requested: int = 1,
              **overrides: Any) -> GenerationRequest:
        template = load_prompt_template(ontology)
        schema = candidate_json_schema(ontology, recipes_requested=recipes_requested)
        fields: dict[str, Any] = {
            "slot_id": slot_id,
            "system_instruction": template.system_instruction,
            "input_text": build_generation_prompt(template, recipes_requested=recipes_requested),
            "response_json_schema": schema,
            "model_id": config.model_id,
            "thinking_level": config.thinking_level,
            "response_mime_type": config.response_mime_type,
            "max_output_tokens": config.max_output_tokens,
            "recipes_requested": recipes_requested,
            "ontology_identity": ontology.sha256,
            "prompt_template_identity": template.identity(),
            "provider_config_identity": json_schema_identity(schema),
        }
        fields.update(overrides)
        return GenerationRequest(**fields)

    return _make
