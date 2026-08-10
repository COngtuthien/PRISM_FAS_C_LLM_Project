"""Shared fixtures for the C2 pilot tests.

Every test here is offline and reads the archived live responses committed under
`reports/c2/`. No fixture constructs a provider client, reads a credential or
opens a socket: the whole point of C2's archive is that the science is
reproducible without calling the model again.
"""
from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import pytest

from prism_fas.llm.config import LLMProviderConfig, load_llm_config, provider_config_identity
from prism_fas.llm.contracts import GenerationRequest
from prism_fas.llm.json_schema import candidate_json_schema
from prism_fas.llm.prompt import build_generation_prompt, load_prompt_template
from prism_fas.recipes.ontology import Ontology, load_ontology

REPO = Path(__file__).resolve().parents[2]
ONTOLOGY_PATH = REPO / "configs" / "recipes" / "ontology_m7.yaml"
LLM_CONFIG_PATH = REPO / "configs" / "version_c" / "llm" / "c1_gemini_provider.yaml"
REPORTS = REPO / "reports" / "c2"

PILOT_ARCHIVE_PATH = REPORTS / "C2_PILOT_RAW_ARCHIVE.json"
PILOT_STATE_PATH = REPORTS / "C2_PILOT_STATE.json"
PILOT_AUDIT_PATH = REPORTS / "C2_PILOT_AUDIT.json"
SMOKE_ARCHIVE_PATH = REPORTS / "C2_SMOKE_RAW_ARCHIVE.json"
SMOKE_AUDIT_PATH = REPORTS / "C2_LIVE_SMOKE_AUDIT.json"

#: The pilot bank id used when the archived recipes were compiled.
PILOT_BANK_ID = "c2-pilot-disposable"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """An accidental outbound connection fails loudly instead of costing money."""

    def blocked(*args: Any, **kwargs: Any):
        raise AssertionError("a C2 test attempted a network connection; the offline "
                             "replay contract forbids it")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture(autouse=True)
def no_ambient_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real key on the developer's machine must not change the suite."""
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
def pilot_archive() -> dict[str, Any]:
    if not PILOT_ARCHIVE_PATH.exists():
        pytest.skip(f"{PILOT_ARCHIVE_PATH.name} missing; run: python scripts/c2_run_pilot.py")
    return _load(PILOT_ARCHIVE_PATH)


@pytest.fixture(scope="session")
def pilot_state() -> dict[str, Any]:
    if not PILOT_STATE_PATH.exists():
        pytest.skip(f"{PILOT_STATE_PATH.name} missing; run: python scripts/c2_run_pilot.py")
    return _load(PILOT_STATE_PATH)


@pytest.fixture(scope="session")
def pilot_audit() -> dict[str, Any]:
    if not PILOT_AUDIT_PATH.exists():
        pytest.skip(f"{PILOT_AUDIT_PATH.name} missing; run: python scripts/c2_build_reports.py")
    return _load(PILOT_AUDIT_PATH)


@pytest.fixture(scope="session")
def smoke_archive() -> dict[str, Any]:
    if not SMOKE_ARCHIVE_PATH.exists():
        pytest.skip(f"{SMOKE_ARCHIVE_PATH.name} missing; run: python scripts/c2_live_smoke.py")
    return _load(SMOKE_ARCHIVE_PATH)


@pytest.fixture(scope="session")
def smoke_audit() -> dict[str, Any]:
    if not SMOKE_AUDIT_PATH.exists():
        pytest.skip(f"{SMOKE_AUDIT_PATH.name} missing; run: python scripts/c2_live_smoke.py")
    return _load(SMOKE_AUDIT_PATH)


@pytest.fixture
def make_request(ontology: Ontology, config: LLMProviderConfig):
    """The same frozen request shape the pilot used, rebuilt from the contract."""

    def _make(slot_id: str, recipes_requested: int = 1) -> GenerationRequest:
        template = load_prompt_template(ontology)
        schema = candidate_json_schema(ontology, recipes_requested=recipes_requested)
        return GenerationRequest(
            slot_id=slot_id,
            system_instruction=template.system_instruction,
            input_text=build_generation_prompt(template, recipes_requested=recipes_requested),
            response_json_schema=schema,
            model_id=config.model_id,
            thinking_level=config.thinking_level,
            response_mime_type=config.response_mime_type,
            max_output_tokens=config.max_output_tokens,
            recipes_requested=recipes_requested,
            ontology_identity=ontology.sha256,
            prompt_template_identity=template.identity(),
            # The live pilot bound the real provider-config identity into the
            # request, and the replay archive refuses to serve a response across
            # a different request identity, so this must match exactly.
            provider_config_identity=provider_config_identity(config),
            metadata={"phase": "c2_pilot", "disposable": True, "enters_c3": False},
        )

    return _make
