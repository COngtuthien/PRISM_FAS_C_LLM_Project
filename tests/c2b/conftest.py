"""Shared fixtures for the C2B batch-shape tests.

Offline only. Every test reads the archived live batch committed under
`reports/c2b/`; none constructs a provider client, reads a credential or opens a
socket.
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
from prism_fas.llm.json_schema import candidate_json_schema, json_schema_identity
from prism_fas.llm.prompt import build_generation_prompt, load_prompt_template
from prism_fas.recipes.ontology import Ontology, load_ontology

REPO = Path(__file__).resolve().parents[2]
ONTOLOGY_PATH = REPO / "configs" / "recipes" / "ontology_m7.yaml"
LLM_CONFIG_PATH = REPO / "configs" / "version_c" / "llm" / "c1_gemini_provider.yaml"
QUOTA_PATH = REPO / "configs" / "version_c" / "llm" / "c2b_coverage_quotas.yaml"
REPORTS = REPO / "reports" / "c2b"

BATCH_ARCHIVE_PATH = REPORTS / "C2B_RAW_ARCHIVE.json"
BATCH_STATE_PATH = REPORTS / "C2B_BATCH_STATE.json"
BATCH_AUDIT_PATH = REPORTS / "C2B_LIVE_BATCH_AUDIT.json"
COVERAGE_AUDIT_PATH = REPORTS / "C2B_COVERAGE_AUDIT.json"

BATCH_SIZE = 32
C2B_BANK_ID = "c2b-batch-disposable"

#: The single-recipe item schema identity C2 ran under. C2B must not move it.
FROZEN_ITEM_SCHEMA_IDENTITY = "1e3f050e129a0ee1305bf8af98e9b4e015373c54ff130763be49c82da56e3579"
FROZEN_ONTOLOGY_IDENTITY = "90694441c2ef1477ca8f6c4dd724a4997a3e166cbf5a067d52c101892f952bbd"
FROZEN_SYSTEM_PROMPT_IDENTITY = "d95e46fcef4e3ec54a3405f75526cb60f3966c2820934a5f6224fc979277038f"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: Any, **kwargs: Any):
        raise AssertionError("a C2B test attempted a network connection; the offline "
                             "replay contract forbids it")

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
def batch_archive() -> dict[str, Any]:
    if not BATCH_ARCHIVE_PATH.exists():
        pytest.skip(f"{BATCH_ARCHIVE_PATH.name} missing; run: python scripts/c2b_run_batch.py")
    return _load(BATCH_ARCHIVE_PATH)


@pytest.fixture(scope="session")
def batch_state() -> dict[str, Any]:
    if not BATCH_STATE_PATH.exists():
        pytest.skip(f"{BATCH_STATE_PATH.name} missing; run: python scripts/c2b_run_batch.py")
    return _load(BATCH_STATE_PATH)


@pytest.fixture(scope="session")
def batch_audit() -> dict[str, Any]:
    if not BATCH_AUDIT_PATH.exists():
        pytest.skip(f"{BATCH_AUDIT_PATH.name} missing; run: python scripts/c2b_build_reports.py")
    return _load(BATCH_AUDIT_PATH)


@pytest.fixture(scope="session")
def coverage_artifact() -> dict[str, Any]:
    if not COVERAGE_AUDIT_PATH.exists():
        pytest.skip(f"{COVERAGE_AUDIT_PATH.name} missing; run: python scripts/c2b_build_reports.py")
    return _load(COVERAGE_AUDIT_PATH)


@pytest.fixture
def make_batch_request(ontology: Ontology, config: LLMProviderConfig, quotas: QuotaSpec):
    """The C2B batch request, rebuilt from the committed contract."""

    def _make(quota_spec: QuotaSpec | None = None,
              array_bounds: bool = False) -> GenerationRequest:
        spec = quota_spec if quota_spec is not None else quotas
        template = load_prompt_template(ontology)
        schema = candidate_json_schema(ontology, recipes_requested=BATCH_SIZE,
                                       array_bounds=array_bounds)
        return GenerationRequest(
            slot_id="C2B_BATCH_000",
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
            metadata={"phase": "c2b_batch_shape", "quota_identity": spec.quota_identity},
        )

    return _make


@pytest.fixture
def schema_identity_of():
    def _identity(schema: dict[str, Any]) -> str:
        return json_schema_identity(schema)

    return _identity
