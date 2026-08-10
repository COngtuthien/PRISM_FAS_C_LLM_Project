"""Shared fixtures for the C2C route-contract tests.

Offline only. No fixture constructs a provider client, reads a credential or
opens a socket.
"""
from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import pytest

from prism_fas.llm.config import LLMProviderConfig, load_llm_config, provider_config_identity
from prism_fas.llm.contracts import GenerationRequest
from prism_fas.llm.coverage_quotas import QuotaSpec, load_quota_spec
from prism_fas.llm.json_schema import candidate_json_schema
from prism_fas.llm.pipeline import RecipePlanner
from prism_fas.llm.prompt import build_generation_prompt, load_prompt_template
from prism_fas.llm.providers.mock import MockRecipeProvider
from prism_fas.llm.route_policy import RoutePolicy, load_route_policy
from prism_fas.recipes.ontology import Ontology, load_ontology

REPO = Path(__file__).resolve().parents[2]
ONTOLOGY_PATH = REPO / "configs" / "recipes" / "ontology_m7.yaml"
LLM_CONFIG_PATH = REPO / "configs" / "version_c" / "llm" / "c1_gemini_provider.yaml"
QUOTA_PATH = REPO / "configs" / "version_c" / "llm" / "c2b_coverage_quotas.yaml"
ROUTE_POLICY_PATH = REPO / "configs" / "version_c" / "llm" / "c2c_route_policy.yaml"
REPORTS = REPO / "reports" / "c2c"
C2B_REPORTS = REPO / "reports" / "c2b"

BATCH_SIZE = 32


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: Any, **kwargs: Any):
        raise AssertionError("a C2C test attempted a network connection; the offline "
                             "contract forbids it")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture(autouse=True)
def no_ambient_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def ontology() -> Ontology:
    return load_ontology(ONTOLOGY_PATH)


@pytest.fixture(scope="session")
def config() -> LLMProviderConfig:
    return load_llm_config(LLM_CONFIG_PATH)


@pytest.fixture(scope="session")
def quotas() -> QuotaSpec:
    return load_quota_spec(QUOTA_PATH)


@pytest.fixture(scope="session")
def route_policy() -> RoutePolicy:
    return load_route_policy(ROUTE_POLICY_PATH)


@pytest.fixture
def planner(config: LLMProviderConfig, ontology: Ontology, route_policy: RoutePolicy):
    """A planner that enforces the scientific route contract."""
    return RecipePlanner(provider=MockRecipeProvider(), config=config, ontology=ontology,
                         sleep=lambda _seconds: None, route_policy=route_policy)


@pytest.fixture
def planner_without_route_policy(config: LLMProviderConfig, ontology: Ontology):
    """The pre-C2C behaviour, kept so the contrast is testable."""
    return RecipePlanner(provider=MockRecipeProvider(), config=config, ontology=ontology,
                         sleep=lambda _seconds: None)


@pytest.fixture
def candidate():
    """A recipe that satisfies schema, ontology, ranges and compatibility.

    `generator_route` is supplied by the caller so route rules can be varied
    without touching anything else.
    """

    def _make(generator_route: list[str] | None = None, seed: int = 4242) -> dict:
        payload = {
            "schema_version": "1.1",
            "medium": {"family": "display-like", "roughness": 0.2, "transparency": 0.05},
            "geometry": {"shape": "flat", "rigidity": 0.9, "coverage": 0.6},
            "regions": ["left_eye", "right_eye"],
            "artifacts": [{"name": "pixel_grid", "strength": 0.3}],
            "capture": {"yaw": 5.0, "illumination": "front", "compression_q": 80,
                        "scale": 1.0, "motion": 0.05, "defocus": 0.1},
            "forbidden_shortcuts": [],
            "seed": seed,
        }
        if generator_route is not None:
            payload["generator_route"] = generator_route
        return payload

    return _make


@pytest.fixture
def envelope():
    def _make(*candidates: dict) -> str:
        return json.dumps({"recipes": list(candidates)})

    return _make


@pytest.fixture
def make_request(ontology: Ontology, config: LLMProviderConfig, quotas: QuotaSpec):
    """The C2C batch request, rebuilt from the committed contract."""

    def _make(policy: RoutePolicy | None = None, quota_spec: QuotaSpec | None = None,
              slot_id: str = "C2C_BATCH_000") -> GenerationRequest:
        spec = quota_spec if quota_spec is not None else quotas
        template = load_prompt_template(ontology, policy)
        schema = candidate_json_schema(ontology, recipes_requested=BATCH_SIZE,
                                       array_bounds=False)
        return GenerationRequest(
            slot_id=slot_id,
            system_instruction=template.system_instruction,
            input_text=build_generation_prompt(
                template, recipes_requested=BATCH_SIZE,
                coverage_quotas=spec.prompt_block(ontology)),
            response_json_schema=schema,
            model_id=config.model_id,
            thinking_level=config.thinking_level,
            response_mime_type=config.response_mime_type,
            max_output_tokens=config.max_output_tokens,
            recipes_requested=BATCH_SIZE,
            ontology_identity=ontology.sha256,
            prompt_template_identity=template.identity(),
            provider_config_identity=provider_config_identity(config),
            route_policy_identity=policy.route_policy_identity if policy else "",
            metadata={"phase": "c2c_route_contract"},
        )

    return _make


@pytest.fixture(scope="session")
def c2b_archive() -> dict[str, Any]:
    path = C2B_REPORTS / "C2B_RAW_ARCHIVE.json"
    if not path.exists():
        pytest.skip(f"{path.name} missing; C2B must have run")
    return _load(path)


@pytest.fixture(scope="session")
def c2c_state() -> dict[str, Any]:
    path = REPORTS / "C2C_BATCH_STATE.json"
    if not path.exists():
        pytest.skip(f"{path.name} missing; run: python scripts/c2c_run_batch.py")
    return _load(path)


@pytest.fixture(scope="session")
def c2c_archive() -> dict[str, Any]:
    path = REPORTS / "C2C_RAW_ARCHIVE.json"
    if not path.exists():
        pytest.skip(f"{path.name} missing; run: python scripts/c2c_run_batch.py")
    return _load(path)


@pytest.fixture(scope="session")
def c2c_coverage() -> dict[str, Any]:
    path = REPORTS / "C2C_COVERAGE_AUDIT.json"
    if not path.exists():
        pytest.skip(f"{path.name} missing; run: python scripts/c2c_build_reports.py")
    return _load(path)
