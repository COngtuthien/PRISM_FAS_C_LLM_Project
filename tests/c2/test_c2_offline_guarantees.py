"""Ordinary pytest must never call Gemini, and never spend money.

C1 proved this for a suite with no archive. C2 has archived live responses and a
credential that may now exist in the developer's environment, so the guarantee
needs restating against the stronger conditions.
"""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import pytest

from prism_fas.llm.providers import gemini as gemini_module
from prism_fas.llm.providers.gemini import GeminiRecipeProvider

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "reports" / "c2"
DOCS = REPO / "docs" / "c2"
SCRIPTS = REPO / "scripts"


def test_the_ambient_credential_is_removed_for_every_test():
    assert not os.environ.get("GEMINI_API_KEY")
    assert not os.environ.get("GOOGLE_API_KEY")


def test_the_vendor_sdk_import_is_lazy_and_confined_to_client_construction():
    """The SDK may only be imported where a client is built.

    A module-level import would load the vendor SDK for every consumer of the
    package, and would make "did a test call Gemini?" depend on import order
    rather than on structure. The import lives inside `_ensure_client`, which is
    the single place a client can come into existence.
    """
    source = inspect.getsource(gemini_module)
    assert "from google import genai" in source
    for line in source.splitlines():
        if "import genai" in line or "import google" in line:
            assert line.startswith("        "), (
                f"the vendor SDK is imported outside a function body: {line!r}")
    body = inspect.getsource(GeminiRecipeProvider._ensure_client)
    assert "from google import genai" in body
    assert source.count("genai.Client(") == 1


def test_constructing_the_provider_makes_no_call_and_reads_no_key(config):
    provider = GeminiRecipeProvider(config)
    assert provider.api_key_present is False
    described = provider.describe()
    assert described["api_key_present"] is False
    assert described["tools_passed"] is False
    assert described["media_input_supported"] is False
    assert described["forbidden_sampling_fields_sent"] == []
    # `api_key_env` (the variable NAME) and `api_key_present` (a boolean) are
    # both allowed; a credential VALUE is not.
    assert described["api_key_env"] == config.api_key_env
    assert set(described) & {"api_key", "apiKey", "key", "token", "credential"} == set()
    assert "AIza" not in json.dumps(described)


def test_the_request_shape_sends_no_sampling_control_and_no_media(config, make_request):
    kwargs = GeminiRecipeProvider(config).build_call_kwargs(make_request("pilot_000"))
    assert "tools" not in kwargs
    assert isinstance(kwargs["input"], str)
    assert kwargs["store"] is False
    assert kwargs["stream"] is False
    for forbidden in ("temperature", "top_p", "top_k"):
        assert forbidden not in kwargs["generation_config"]
        assert forbidden not in kwargs
    for forbidden in ("image", "images", "inline_data", "file_data", "media", "video", "audio",
                      "parts", "attachments"):
        assert forbidden not in kwargs


def test_a_missing_credential_is_an_auth_error_not_a_crash(config, make_request):
    result = GeminiRecipeProvider(config).generate(make_request("pilot_000"), attempt=1)
    assert result.error is not None
    assert result.error.error_class.value == "auth"
    assert result.raw_text is None
    assert config.api_key_env in str(result.error)


def test_no_c2_artifact_contains_anything_credential_shaped():
    """Every committed C2 artifact is scanned, not just the ones we expect."""
    suspicious = []
    for path in sorted(list(REPORTS.rglob("*.json")) + list(DOCS.rglob("*.md"))):
        if "raw_responses" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        if "aiza" in lowered or "ya29." in lowered or "-----begin" in lowered:
            suspicious.append(str(path))
        for marker in ('"api_key"', "'api_key'", '"apikey"', '"authorization"'):
            if marker in lowered:
                suspicious.append(f"{path} ({marker})")
    assert suspicious == [], f"credential-shaped content in {suspicious}"


def test_the_c2_scripts_never_read_the_key_into_a_returned_value():
    """The key may be read only where it is handed to the SDK client."""
    for name in ("c2_pilot_common.py", "c2_live_smoke.py", "c2_run_pilot.py",
                 "c2_build_reports.py"):
        source = (SCRIPTS / name).read_text(encoding="utf-8")
        assert "AIza" not in source
        # `api_key_env` names the variable; reading its VALUE belongs to the
        # provider alone.
        assert "os.environ[" not in source
        for banned in ("GEMINI_API_KEY\"]", "GEMINI_API_KEY']"):
            assert banned not in source


def test_the_gemini_provider_has_no_media_code_path():
    source = inspect.getsource(gemini_module)
    for banned in ("Part.from_bytes", "upload_file", "inline_data", "from_uri", "image_bytes",
                   "PIL.Image"):
        assert banned not in source


def test_the_pilot_recipes_are_marked_disposable_everywhere():
    """A C2 pilot recipe must never look like a C3 or bank artifact."""
    for name in ("C2_PILOT_STATE.json", "C2_PILOT_RAW_ARCHIVE.json", "C2_PILOT_AUDIT.json"):
        path = REPORTS / name
        if not path.exists():
            pytest.skip(f"{name} missing; run the C2 pilot scripts")
        payload = json.loads(path.read_text(encoding="utf-8"))
        text = json.dumps(payload)
        assert "bank_lock" not in text.lower()
        assert "BANK_LOCK" not in text
