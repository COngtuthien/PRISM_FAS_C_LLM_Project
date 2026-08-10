"""C1 group F: security.

Two properties, both enforced structurally rather than by convention:

* nothing that could constitute target adaptation reaches a provider;
* no credential reaches a log, an exception, a report or a provenance record.
"""
from __future__ import annotations

import inspect
import json
import re

import pytest

from prism_fas.llm import providers as providers_pkg
from prism_fas.llm.config import LLMProviderConfig
from prism_fas.llm.contracts import GenerationRequest
from prism_fas.llm.firewall import (
    REDACTION,
    RequestFirewallError,
    assert_request_is_target_free,
    redact_secrets,
    scan_for_target_leakage,
)
from prism_fas.llm.prompt import build_generation_prompt, load_prompt_template
from prism_fas.llm.providers import GeminiRecipeProvider, MockRecipeProvider
from prism_fas.llm.providers.gemini import API_SURFACE

from c1_helpers import envelope

FAKE_KEY = "AIza" + "0123456789abcdefghijklmnopqrstuvwxy"   # 39 chars, never real
# Assembled from parts so no literal credential-shaped token or PEM header ever
# appears in a committed file, which lets the repository scan below stay strict.
FAKE_PEM = "-----" + "BEGIN PRIVATE KEY" + "-----\nMIIEvQIBADAN"


# ===================== target firewall ====================================

@pytest.mark.parametrize("text", [
    "Generate recipes for the SiW-Mv2 target set.",
    "Focus on attacks that Version B failed to detect.",
    "Optimize target ACER for the held-out benchmark.",
    "CASIA-FASD and MSU-MFSD are the source corpora.",
    "The test set contains silicone masks and paper glasses.",
    "Previous version had a high APCER on transparent masks.",
    "Improve held-out performance on the evaluation split.",
    "Here is the attack_family distribution of the target.",
])
def test_target_shaped_prompts_are_refused(text: str):
    violations = scan_for_target_leakage({"input_text": text})
    assert violations, f"firewall did not catch: {text!r}"
    with pytest.raises(RequestFirewallError):
        assert_request_is_target_free({"input_text": text})


@pytest.mark.parametrize("payload", [
    {"image": "data:image/png;base64,iVBORw0KGgo="},
    {"inline_data": {"mime_type": "image/jpeg", "data": "..."}},
    {"parts": [{"text": "hi"}]},
    {"input": "look at frames/subject_04/crop_0001.png"},
    {"labels": [0, 1, 1]},
    {"ground_truth": "spoof"},
    {"target_metrics": {"acer": 0.36}},
])
def test_media_and_label_payloads_are_refused(payload: dict):
    assert scan_for_target_leakage(payload)
    with pytest.raises(RequestFirewallError):
        assert_request_is_target_free(payload)


def test_the_real_prompt_passes_the_firewall(ontology):
    """The contract is useless if its own prompt trips it."""
    template = load_prompt_template(ontology)
    body = build_generation_prompt(template, recipes_requested=32)
    assert_request_is_target_free({"system_instruction": template.system_instruction,
                                   "input_text": body,
                                   "response_json_schema": {}})


def test_ontology_vocabulary_does_not_trip_the_firewall(ontology):
    """'context' is a semantic region and 'blur' an artifact; neither may be
    mistaken for target content."""
    payload = {"vocabulary": list(ontology.regions) + list(ontology.artifacts)
               + list(ontology.media) + list(ontology.geometry_shapes)}
    assert scan_for_target_leakage(payload) == []


def test_a_coverage_quota_cannot_smuggle_target_content(ontology):
    template = load_prompt_template(ontology)
    with pytest.raises(RequestFirewallError):
        build_generation_prompt(template, recipes_requested=4,
                                coverage_quotas={"note": "match the SiW attack families"})


def test_every_provider_runs_the_firewall_including_the_mock(make_request):
    """A leak must not be able to hide behind a test double."""
    provider = MockRecipeProvider()
    poisoned = make_request(input_text="Generate recipes that beat the target benchmark.")
    with pytest.raises(RequestFirewallError):
        provider.generate(poisoned)
    assert provider.calls == []          # refused before the call was recorded


def test_firewall_runs_before_any_gemini_client_is_built(config, make_request):
    provider = GeminiRecipeProvider(config)
    poisoned = make_request(input_text="Report the APCER of the held-out target.")
    with pytest.raises(RequestFirewallError):
        provider.generate(poisoned)


# ===================== no image path ======================================

def test_gemini_provider_has_no_media_input_path(config, make_request):
    """`input` is a plain string and nothing attaches media to it."""
    provider = GeminiRecipeProvider(config)
    kwargs = provider.build_call_kwargs(make_request())
    assert isinstance(kwargs["input"], str)
    for forbidden in ("image", "images", "inline_data", "file_data", "parts",
                      "media", "video", "audio", "contents"):
        assert forbidden not in kwargs


def test_gemini_provider_never_passes_tools(config, make_request):
    """Omitting `tools` disables function calling, Search grounding, URL context,
    code execution and file search together."""
    provider = GeminiRecipeProvider(config)
    kwargs = provider.build_call_kwargs(make_request())
    assert "tools" not in kwargs
    assert kwargs["store"] is False
    assert kwargs["stream"] is False


def test_gemini_provider_never_sends_deprecated_sampling_controls(config, make_request):
    provider = GeminiRecipeProvider(config)
    kwargs = provider.build_call_kwargs(make_request())
    generation_config = kwargs["generation_config"]
    for field in ("temperature", "top_p", "top_k"):
        assert field not in generation_config
        assert field not in kwargs
    assert generation_config["thinking_level"] == "medium"


def test_gemini_provider_uses_the_verified_structured_output_shape(config, make_request):
    provider = GeminiRecipeProvider(config)
    kwargs = provider.build_call_kwargs(make_request())
    response_format = kwargs["response_format"]
    assert response_format["type"] == "text"
    assert response_format["mime_type"] == "application/json"
    assert response_format["schema"]["properties"]["recipes"]["items"]["additionalProperties"] is False
    assert API_SURFACE == "interactions"


def test_no_provider_module_mentions_an_image_api():
    """A grep-level guard: the media surface must stay unreferenced."""
    for module in (providers_pkg.gemini, providers_pkg.mock, providers_pkg.replay):
        source = inspect.getsource(module)
        for banned in ("Part.from_bytes", "upload_file", "inline_data", "from_uri",
                       "image_bytes", "PIL.Image"):
            assert banned not in source, f"{module.__name__} references {banned}"


# ===================== secret handling ====================================

@pytest.mark.parametrize("text,marker", [
    (f"key={FAKE_KEY}", "AIza"),
    ("Authorization: Bearer ya29.abcdefghijklmnop", "ya29."),
    ('{"api_key": "sk-0123456789abcdefghij"}', "sk-"),
    (FAKE_PEM, "PRIVATE KEY"),
])
def test_credentials_are_redacted(text: str, marker: str):
    cleaned = redact_secrets(text)
    assert marker not in cleaned or REDACTION in cleaned
    assert FAKE_KEY not in cleaned


def test_provider_never_reads_a_key_into_a_returned_field(config, monkeypatch, make_request):
    monkeypatch.setenv(config.api_key_env, FAKE_KEY)
    provider = GeminiRecipeProvider(config)
    assert provider.api_key_present is True          # boolean only
    described = json.dumps(provider.describe())
    assert FAKE_KEY not in described
    assert "AIza" not in described
    kwargs = json.dumps(provider.build_call_kwargs(make_request()))
    assert FAKE_KEY not in kwargs


def test_missing_key_is_an_auth_error_not_a_crash(config, make_request):
    provider = GeminiRecipeProvider(config)
    assert provider.api_key_present is False
    result = provider.generate(make_request())
    assert result.error is not None
    assert result.error.error_class.value == "auth"
    assert not result.error.retryable
    assert config.api_key_env in str(result.error)      # names the variable
    assert FAKE_KEY not in str(result.error)


def test_error_messages_are_redacted(config, monkeypatch, make_request):
    """A provider exception that echoes the request must not leak the key."""
    monkeypatch.setenv(config.api_key_env, FAKE_KEY)

    class LeakyClient:
        class interactions:  # noqa: N801
            @staticmethod
            def create(**kwargs):
                raise RuntimeError(f"invalid request with key={FAKE_KEY}")

    provider = GeminiRecipeProvider(config, client=LeakyClient())
    result = provider.generate(make_request())
    assert result.error is not None
    assert FAKE_KEY not in str(result.error)
    assert REDACTION in str(result.error)


def test_committed_c1_files_contain_no_credential():
    """A last-line guard over the files this milestone adds to Git."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    targets = [repo / "configs" / "version_c" / "llm" / "c1_gemini_provider.yaml",
               repo / ".env.example"]
    targets += sorted((repo / "src" / "prism_fas" / "llm").rglob("*.py"))
    targets += sorted((repo / "tests" / "c1").glob("*.py"))
    for path in targets:
        text = path.read_text(encoding="utf-8")
        # The fake key is built at runtime from two halves in this module, so a
        # literal 39-character AIza token anywhere is a real finding.
        assert not re.search(r"AIza[0-9A-Za-z_\-]{35}", text), f"credential-shaped token in {path}"
        # A real PEM header, not the detector's regex for one: the literal form
        # allows only capitals and spaces between the dashes.
        assert not re.search(r"-----BEGIN [A-Z ]+-----", text), f"private key material in {path}"


def test_env_example_declares_the_variable_without_a_value():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    lines = (repo / ".env.example").read_text(encoding="utf-8").splitlines()
    assignments = [line for line in lines if "=" in line and not line.lstrip().startswith("#")]
    assert any(line.strip() == "GEMINI_API_KEY=" for line in assignments)
    for line in assignments:
        assert line.partition("=")[2] == "", f"{line!r} carries a value"
