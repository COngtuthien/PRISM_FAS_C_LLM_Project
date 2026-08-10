"""C2C stays source-independent, and pytest never calls Gemini.

The route amendment is engineering / spec reconciliation. It must be provable
that no dataset, metric, attack family or target result reached the prompt.
"""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import pytest

from prism_fas.llm.firewall import scan_for_target_leakage
from prism_fas.llm.prompt import load_prompt_template
from prism_fas.llm.providers import gemini as gemini_module
from prism_fas.llm.providers.gemini import GeminiRecipeProvider

from c2c_constants import C2B_SYSTEM_PROMPT_IDENTITY, C2C_SYSTEM_PROMPT_IDENTITY

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "reports" / "c2c"
DOCS = REPO / "docs" / "c2c"
SCRIPTS = REPO / "scripts"
ROUTE_POLICY_FILE = REPO / "configs" / "version_c" / "llm" / "c2c_route_policy.yaml"


def test_no_target_field_enters_the_c2c_request(make_request, route_policy):
    request = make_request(policy=route_policy)
    violations = scan_for_target_leakage({
        "system_instruction": request.system_instruction,
        "input_text": request.input_text,
        "response_json_schema": request.response_json_schema,
        "metadata": request.metadata,
    })
    assert violations == [], f"the C2C request leaked: {violations}"


def test_the_route_amendment_itself_is_target_free(ontology, route_policy):
    """Only the added block, examined on its own."""
    before = load_prompt_template(ontology).system_instruction
    after = load_prompt_template(ontology, route_policy).system_instruction
    added = after.replace(before.rstrip(), "") if before.rstrip() in after else after
    assert scan_for_target_leakage(added) == []
    for token in ("siw", "casia", "msu", "acer", "apcer", "bpcer", "hter", "attack_family",
                  "target", "eer", "auc"):
        assert token not in added.lower(), f"the route amendment mentions {token!r}"


def test_the_route_policy_file_carries_no_corpus_vocabulary():
    body = "\n".join(line for line in ROUTE_POLICY_FILE.read_text(encoding="utf-8").splitlines()
                     if not line.lstrip().startswith("#"))
    for token in ("siw", "casia", "msu", "acer", "apcer", "bpcer", "attack_family"):
        assert token not in body.lower()


def test_the_prompt_amendment_is_exactly_the_route_block(ontology, route_policy):
    """Nothing else in the system instruction moved."""
    before = load_prompt_template(ontology).system_instruction
    after = load_prompt_template(ontology, route_policy).system_instruction
    assert load_prompt_template(ontology).identity() == C2B_SYSTEM_PROMPT_IDENTITY
    assert load_prompt_template(ontology, route_policy).identity() == C2C_SYSTEM_PROMPT_IDENTITY
    # Every line of the old instruction survives, in order, in the new one.
    old_lines = before.splitlines()
    new_lines = after.splitlines()
    assert [line for line in old_lines if line in new_lines] == old_lines
    assert len(new_lines) > len(old_lines)
    # The generation template is untouched.
    assert (load_prompt_template(ontology).generation_template
            == load_prompt_template(ontology, route_policy).generation_template)


def test_the_ambient_credential_is_removed_for_every_test():
    assert not os.environ.get("GEMINI_API_KEY")
    assert not os.environ.get("GOOGLE_API_KEY")


def test_a_missing_credential_is_an_auth_error_not_a_call(config, make_request, route_policy):
    result = GeminiRecipeProvider(config).generate(make_request(policy=route_policy), attempt=1)
    assert result.error is not None
    assert result.error.error_class.value == "auth"
    assert result.raw_text is None


def test_the_request_shape_sends_no_sampling_control_and_no_media(config, make_request,
                                                                  route_policy):
    kwargs = GeminiRecipeProvider(config).build_call_kwargs(make_request(policy=route_policy))
    assert "tools" not in kwargs
    assert isinstance(kwargs["input"], str)
    assert kwargs["store"] is False and kwargs["stream"] is False
    for forbidden in ("temperature", "top_p", "top_k"):
        assert forbidden not in kwargs and forbidden not in kwargs["generation_config"]
    for forbidden in ("image", "images", "inline_data", "file_data", "media", "video", "audio",
                      "parts", "attachments"):
        assert forbidden not in kwargs


def test_the_gemini_provider_still_has_no_media_code_path():
    source = inspect.getsource(gemini_module)
    for banned in ("Part.from_bytes", "upload_file", "inline_data", "from_uri", "image_bytes",
                   "PIL.Image"):
        assert banned not in source


def test_no_c2c_artifact_contains_anything_credential_shaped():
    suspicious = []
    for path in sorted(list(REPORTS.rglob("*.json")) + list(DOCS.rglob("*.md"))):
        lowered = path.read_text(encoding="utf-8").lower()
        if "aiza" in lowered or "ya29." in lowered or "-----begin" in lowered:
            suspicious.append(str(path))
    assert suspicious == []


def test_the_c2c_scripts_never_read_the_key_into_a_returned_value():
    for name in ("c2c_common.py", "c2c_run_batch.py", "c2c_build_reports.py",
                 "c2c_replay_c2b.py"):
        source = (SCRIPTS / name).read_text(encoding="utf-8")
        assert "AIza" not in source
        assert "os.environ[" not in source


def test_the_runner_cannot_issue_a_second_semantic_batch():
    source = (SCRIPTS / "c2c_run_batch.py").read_text(encoding="utf-8")
    assert "already_completed" in source
    assert "semantic_response_received" in source
    code = "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith("#"))
    assert ".generate_slot(" not in code, "semantic retry would re-roll the batch"


def test_the_c2b_replay_script_makes_no_network_call():
    source = (SCRIPTS / "c2c_replay_c2b.py").read_text(encoding="utf-8")
    assert "GeminiRecipeProvider" not in source
    assert "ReplayRecipeProvider" in source
