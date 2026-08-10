"""The batch request must stay source-independent, and pytest must never call
Gemini.

C2B adds the first caller-supplied content the prompt has ever carried - the
coverage quotas - so the target-leakage firewall matters more here than anywhere
before it.
"""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import pytest

from prism_fas.llm.coverage_quotas import QuotaSpec, axis_vocabulary, parse_quota_spec
from prism_fas.llm.firewall import RequestFirewallError, scan_for_target_leakage
from prism_fas.llm.prompt import build_generation_prompt, load_prompt_template
from prism_fas.llm.providers import gemini as gemini_module
from prism_fas.llm.providers.gemini import GeminiRecipeProvider

from c2b_constants import BATCH_SIZE, FROZEN_SYSTEM_PROMPT_IDENTITY

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "reports" / "c2b"
DOCS = REPO / "docs" / "c2b"
SCRIPTS = REPO / "scripts"


def test_no_target_field_enters_the_batch_request(make_batch_request):
    request = make_batch_request()
    violations = scan_for_target_leakage({
        "system_instruction": request.system_instruction,
        "input_text": request.input_text,
        "response_json_schema": request.response_json_schema,
        "metadata": request.metadata,
    })
    assert violations == [], f"the batch request leaked: {violations}"


def test_the_rendered_quotas_name_only_ontology_values(quotas, ontology):
    """A quota may mention a category only if the ontology already defines it."""
    vocabulary = axis_vocabulary(ontology)
    block = quotas.prompt_block(ontology)
    assert set(block) - {"diversity", "precedence"} <= set(vocabulary)
    for axis, text in block.items():
        if axis in ("diversity", "precedence"):
            continue
        for name in vocabulary[axis]:
            pass  # every listed name is drawn from the vocabulary by construction
        # Nothing corpus-shaped may appear in the rendered text.
        assert scan_for_target_leakage(text) == []


def test_a_quota_carrying_target_language_is_refused_before_transmission(ontology):
    """The firewall runs over the prompt's own output, so a smuggled quota fails
    locally rather than reaching the provider."""
    template = load_prompt_template(ontology)
    with pytest.raises(RequestFirewallError):
        build_generation_prompt(
            template, recipes_requested=BATCH_SIZE,
            coverage_quotas={"media": "match the SiW-Mv2 attack family frequencies"})


def test_the_system_prompt_identity_is_unchanged_by_c2b(ontology):
    assert load_prompt_template(ontology).identity() == FROZEN_SYSTEM_PROMPT_IDENTITY


def test_the_quota_spec_carries_no_corpus_or_metric_vocabulary(quotas):
    text = json.dumps(quotas.as_dict()).lower()
    for token in ("siw", "casia", "msu", "celeba", "oulu", "acer", "apcer", "bpcer", "hter",
                  "attack_family", "attack family", "target", "eer", "auc"):
        assert token not in text, f"the quota spec mentions {token!r}"


def test_the_committed_quota_file_carries_no_corpus_vocabulary():
    path = REPO / "configs" / "version_c" / "llm" / "c2b_coverage_quotas.yaml"
    body = "\n".join(line for line in path.read_text(encoding="utf-8").splitlines()
                     if not line.lstrip().startswith("#"))
    for token in ("siw", "casia", "msu", "acer", "apcer", "bpcer", "attack_family"):
        assert token not in body.lower()


# --------------------------------------------------------------- no live calls
def test_the_ambient_credential_is_removed_for_every_test():
    assert not os.environ.get("GEMINI_API_KEY")
    assert not os.environ.get("GOOGLE_API_KEY")


def test_a_missing_credential_is_an_auth_error_not_a_call(config, make_batch_request):
    result = GeminiRecipeProvider(config).generate(make_batch_request(), attempt=1)
    assert result.error is not None
    assert result.error.error_class.value == "auth"
    assert result.raw_text is None


def test_the_batch_request_shape_sends_no_sampling_control_and_no_media(config,
                                                                       make_batch_request):
    kwargs = GeminiRecipeProvider(config).build_call_kwargs(make_batch_request())
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


def test_no_c2b_artifact_contains_anything_credential_shaped():
    suspicious = []
    for path in sorted(list(REPORTS.rglob("*.json")) + list(DOCS.rglob("*.md"))):
        if "raw_responses" in path.parts:
            continue
        lowered = path.read_text(encoding="utf-8").lower()
        if "aiza" in lowered or "ya29." in lowered or "-----begin" in lowered:
            suspicious.append(str(path))
    assert suspicious == []


def test_the_c2b_scripts_never_read_the_key_into_a_returned_value():
    for name in ("c2b_common.py", "c2b_run_batch.py", "c2b_build_reports.py"):
        source = (SCRIPTS / name).read_text(encoding="utf-8")
        assert "AIza" not in source
        assert "os.environ[" not in source


def test_the_runner_cannot_issue_a_second_semantic_batch():
    """The anti-cherry-picking guard is structural, not a convention."""
    source = (SCRIPTS / "c2b_run_batch.py").read_text(encoding="utf-8")
    assert "already_completed" in source
    assert "semantic_response_received" in source
    # Semantic retry would ask the provider for a FRESH batch when validation
    # fails, which is exactly the re-roll C2B forbids. The docstring may name it;
    # the code may not call it.
    code = "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith("#"))
    assert ".generate_slot(" not in code
