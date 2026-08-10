"""C1 groups G (provider), H (retry/quota), I (replay) and J (provenance)."""
from __future__ import annotations

import json

import pytest

from prism_fas.llm.config import LLMProviderConfig
from prism_fas.llm.contracts import (
    NON_RETRYABLE_ERROR_CLASSES,
    RETRYABLE_ERROR_CLASSES,
    ErrorClass,
    ProviderError,
)
from prism_fas.llm.pipeline import CandidateOutcome, QuotaBlocked, RecipePlanner
from prism_fas.llm.provenance import (
    REQUIRED_PROVENANCE_FIELDS,
    GenerationProvenance,
    ProvenanceError,
    build_provenance,
    identity_chain,
)
from prism_fas.llm.providers import MockRecipeProvider, ReplayRecipeProvider
from prism_fas.llm.providers.mock import (
    ScriptedResponse,
    auth_error,
    model_unavailable_error,
    quota_exhausted_error,
    rate_limit_error,
    transport_error,
    unsupported_config_error,
)
from prism_fas.llm.providers.replay import ReplayArchive, ReplayEntry

from c1_helpers import envelope


def _planner(config: LLMProviderConfig, ontology, script: list[ScriptedResponse]) -> RecipePlanner:
    return RecipePlanner(provider=MockRecipeProvider(script), config=config, ontology=ontology,
                         sleep=lambda _seconds: None)


# ===================== G. PROVIDER ========================================

def test_mock_valid_output_is_accepted(config, ontology, valid_candidate, make_request):
    planner = _planner(config, ontology, [ScriptedResponse(raw_text=envelope(valid_candidate))])
    result, validation, attempts = planner.generate_slot(make_request(), recipes_requested=1)
    assert result.ok and validation is not None and validation.all_accepted
    assert attempts == 1


def test_invalid_provider_response_is_recorded_not_repaired(config, ontology, make_request):
    planner = _planner(config, ontology, [ScriptedResponse(raw_text="not json")] * 3)
    result, validation, attempts = planner.generate_slot(make_request(), recipes_requested=1)
    assert result.ok                                   # transport succeeded
    assert validation is not None and not validation.all_accepted
    assert validation.response_issues[0]["stage"] == "json_parsing"
    assert attempts == 1 + config.retry.semantic_max_retries


@pytest.mark.parametrize("error,expected", [
    (transport_error(), ErrorClass.TRANSPORT),
    (rate_limit_error(), ErrorClass.RATE_LIMIT),
    (quota_exhausted_error(), ErrorClass.QUOTA_EXHAUSTED),
    (auth_error(), ErrorClass.AUTH),
    (model_unavailable_error("gemini-9.9-flash"), ErrorClass.MODEL_UNAVAILABLE),
    (unsupported_config_error("thinking_level"), ErrorClass.UNSUPPORTED_CONFIG),
])
def test_error_classes_are_assigned(error: ProviderError, expected: ErrorClass):
    assert error.error_class is expected
    assert error.retryable is (expected in RETRYABLE_ERROR_CLASSES)


def test_error_class_partition_is_total_and_disjoint():
    assert RETRYABLE_ERROR_CLASSES.isdisjoint(NON_RETRYABLE_ERROR_CLASSES)
    assert RETRYABLE_ERROR_CLASSES | NON_RETRYABLE_ERROR_CLASSES == set(ErrorClass)


@pytest.mark.parametrize("error", [auth_error(), model_unavailable_error("x"),
                                   unsupported_config_error("y")])
def test_non_retryable_errors_stop_immediately(config, ontology, make_request, error):
    planner = _planner(config, ontology, [ScriptedResponse(error=error),
                                          ScriptedResponse(raw_text="{}")])
    result, validation, attempts = planner.generate_slot(make_request(), recipes_requested=1)
    assert attempts == 1
    assert result.error is not None and result.error.error_class is error.error_class
    assert validation is None
    assert planner._provider.remaining == 1        # the second script entry was never used


def test_gemini_provider_classifies_a_quota_error_apart_from_a_rate_limit(config, make_request,
                                                                          monkeypatch):
    from prism_fas.llm.providers.gemini import _classify

    monkeypatch.setenv(config.api_key_env, "AIza" + "z" * 35)
    rate = _classify(RuntimeError("429 rate_limit_exceeded: too many requests"))
    quota = _classify(RuntimeError("429 quota_exceeded: daily quota exhausted"))
    assert rate.error_class is ErrorClass.RATE_LIMIT and rate.retryable
    assert quota.error_class is ErrorClass.QUOTA_EXHAUSTED and not quota.retryable


def test_gemini_contract_violation_when_no_text_is_returned(config, make_request, monkeypatch):
    monkeypatch.setenv(config.api_key_env, "AIza" + "y" * 35)

    class EmptyClient:
        class interactions:  # noqa: N801
            @staticmethod
            def create(**kwargs):
                return object()          # no output_text

    from prism_fas.llm.providers import GeminiRecipeProvider
    provider = GeminiRecipeProvider(config, client=EmptyClient())
    result = provider.generate(make_request())
    assert result.error is not None
    assert result.error.error_class is ErrorClass.CONTRACT_VIOLATION


# ===================== H. RETRY / QUOTA ===================================

def test_retry_is_bounded_by_the_semantic_budget(config, ontology, make_request):
    script = [ScriptedResponse(raw_text="{}")] * 10
    planner = _planner(config, ontology, script)
    _result, _validation, attempts = planner.generate_slot(make_request(), recipes_requested=1)
    assert attempts == 1 + config.retry.semantic_max_retries == 3


def test_retry_counter_is_exact_and_a_later_success_stops_it(config, ontology, valid_candidate,
                                                             make_request):
    planner = _planner(config, ontology, [
        ScriptedResponse(raw_text="not json"),
        ScriptedResponse(raw_text=envelope(valid_candidate)),
        ScriptedResponse(raw_text="never reached"),
    ])
    _result, validation, attempts = planner.generate_slot(make_request(), recipes_requested=1)
    assert attempts == 2
    assert validation is not None and validation.all_accepted
    assert planner._provider.remaining == 1


def test_transport_failures_do_not_consume_the_semantic_budget(config, ontology, valid_candidate,
                                                               make_request):
    """A flaky connection must not burn the two semantic retries."""
    planner = _planner(config, ontology, [
        ScriptedResponse(error=transport_error()),
        ScriptedResponse(error=rate_limit_error()),
        ScriptedResponse(raw_text="not json"),
        ScriptedResponse(raw_text="still not json"),
        ScriptedResponse(raw_text=envelope(valid_candidate)),
    ])
    _result, validation, attempts = planner.generate_slot(make_request(), recipes_requested=1)
    assert validation is not None and validation.all_accepted
    assert attempts == 3                      # three semantic attempts, transport retried free


def test_transport_retries_are_bounded_too(config, ontology, make_request):
    planner = _planner(config, ontology, [ScriptedResponse(error=transport_error())] * 20)
    result, _validation, _attempts = planner.generate_slot(make_request(), recipes_requested=1)
    assert result.error is not None and result.error.error_class is ErrorClass.TRANSPORT
    used = len(planner._provider.calls)
    assert used == config.retry.transport_max_attempts


def test_backoff_grows_and_is_capped(planner, config):
    delays = [planner.backoff_seconds(attempt) for attempt in range(1, 12)]
    assert delays[0] == config.retry.backoff_initial_seconds
    assert all(later >= earlier for earlier, later in zip(delays, delays[1:]))
    assert max(delays) <= config.retry.backoff_max_seconds


def test_quota_exhaustion_stops_the_run_and_preserves_completed_slots(config, ontology,
                                                                      make_request):
    planner = _planner(config, ontology, [ScriptedResponse(error=quota_exhausted_error())])
    planner.mark_slot_completed("slot-0001")
    planner.mark_slot_completed("slot-0002")
    with pytest.raises(QuotaBlocked) as caught:
        planner.generate_slot(make_request(slot_id="slot-0003"), recipes_requested=1,
                              pending_slot_ids=["slot-0003", "slot-0004"])
    state = caught.value.state.as_dict()
    assert state["blocked"] is True
    assert state["completed_slot_ids"] == ["slot-0001", "slot-0002"]
    assert state["pending_slot_ids"] == ["slot-0003", "slot-0004"]
    assert state["auto_enable_paid"] is False
    assert "never regenerate a completed slot" in state["resume_policy"]
    assert "Code never enables billing" in state["user_decision_required"]


def test_quota_state_serializes_without_a_credential(config, ontology, make_request):
    planner = _planner(config, ontology, [ScriptedResponse(error=quota_exhausted_error())])
    with pytest.raises(QuotaBlocked) as caught:
        planner.generate_slot(make_request(), recipes_requested=1)
    blob = json.dumps(caught.value.state.as_dict())
    assert "AIza" not in blob and "api_key" not in blob.lower()


def test_a_rate_limit_is_retried_rather_than_treated_as_a_quota_stop(config, ontology,
                                                                     valid_candidate, make_request):
    planner = _planner(config, ontology, [
        ScriptedResponse(error=rate_limit_error(retry_after_seconds=0.0)),
        ScriptedResponse(raw_text=envelope(valid_candidate)),
    ])
    _result, validation, _attempts = planner.generate_slot(make_request(), recipes_requested=1)
    assert validation is not None and validation.all_accepted
    assert planner.quota.blocked is False


# ===================== I. REPLAY ==========================================

def _archive(slot_id: str, raw_text: str, request_sha256: str | None = None) -> ReplayArchive:
    return ReplayArchive([ReplayEntry(slot_id=slot_id, attempt=1, raw_text=raw_text,
                                      provider="gemini", model_id="gemini-3.6-flash",
                                      sdk_version="2.17.0", api_surface="interactions",
                                      request_sha256=request_sha256)])


def test_replay_reproduces_the_identical_canonical_recipe(config, ontology, valid_candidate,
                                                          make_request):
    raw = envelope(valid_candidate)
    request = make_request()
    archive = _archive(request.slot_id, raw)

    first = RecipePlanner(provider=ReplayRecipeProvider(archive), config=config, ontology=ontology)
    result_a, validation_a, _ = first.generate_slot(request, recipes_requested=1)
    second = RecipePlanner(provider=ReplayRecipeProvider(archive), config=config, ontology=ontology)
    result_b, validation_b, _ = second.generate_slot(request, recipes_requested=1)

    assert validation_a is not None and validation_b is not None
    assert validation_a.all_accepted and validation_b.all_accepted
    assert result_a.raw_response_sha256 == result_b.raw_response_sha256
    canonical_a = validation_a.candidates[0].canonical_text
    canonical_b = validation_b.candidates[0].canonical_text
    assert canonical_a == canonical_b                       # byte-identical
    assert validation_a.candidates[0].recipe_identity == validation_b.candidates[0].recipe_identity


def test_replay_makes_no_network_call_and_holds_no_client(config, ontology, valid_candidate,
                                                          make_request):
    request = make_request()
    provider = ReplayRecipeProvider(_archive(request.slot_id, envelope(valid_candidate)))
    assert provider.describe()["network"] is False
    assert not hasattr(provider, "_client")
    result = provider.generate(request)
    assert result.ok


def test_replay_refuses_a_slot_it_never_archived(config, valid_candidate, make_request):
    provider = ReplayRecipeProvider(_archive("slot-a", envelope(valid_candidate)))
    with pytest.raises(KeyError):
        provider.generate(make_request(slot_id="slot-b"))


def test_replay_refuses_to_cross_request_identities(config, valid_candidate, make_request):
    """An archived response produced under a different prompt, schema, model or
    ontology must not be replayed under the current one."""
    request = make_request()
    archive = _archive(request.slot_id, envelope(valid_candidate),
                       request_sha256="0" * 64)
    provider = ReplayRecipeProvider(archive)
    with pytest.raises(ValueError, match="refusing to replay across identities"):
        provider.generate(request)


def test_replay_accepts_a_matching_request_identity(config, valid_candidate, make_request):
    request = make_request()
    provider = ReplayRecipeProvider(_archive(request.slot_id, envelope(valid_candidate),
                                             request_sha256=request.request_sha256))
    assert provider.generate(request).ok


def test_replay_archive_identity_is_stable(valid_candidate):
    archive = _archive("slot-a", envelope(valid_candidate))
    same = _archive("slot-a", envelope(valid_candidate))
    assert archive.identity() == same.identity()


def test_replay_archive_rejects_duplicate_entries(valid_candidate):
    entry = ReplayEntry(slot_id="s", attempt=1, raw_text=envelope(valid_candidate))
    with pytest.raises(ValueError, match="duplicate archived response"):
        ReplayArchive([entry, entry])


# ===================== J. PROVENANCE ======================================

def _provenance(config, ontology, valid_candidate, make_request, **overrides):
    planner = _planner(config, ontology, [ScriptedResponse(raw_text=envelope(valid_candidate))])
    request = make_request()
    result, validation, _attempts = planner.generate_slot(request, recipes_requested=1)
    template_provenance = {"system_prompt_sha256": "a" * 64, "request_template_sha256": "b" * 64,
                           "prompt_template_identity": "c" * 64}
    kwargs = dict(request=request, result=result,
                  config_summary={**config.public_summary(), "recipe_schema_version": "1.1"},
                  prompt_provenance=template_provenance, schema_identity="d" * 64,
                  validation_result="accepted",
                  parsed_recipe_sha256=[validation.candidates[0].recipe_identity],
                  retry_count=0, generator_code_commit="deadbeef",
                  request_schedule_id="c1-probe", billing_tier="free")
    kwargs.update(overrides)
    return build_provenance(**kwargs)


def test_every_required_provenance_field_is_present(config, ontology, valid_candidate,
                                                    make_request):
    record = _provenance(config, ontology, valid_candidate, make_request).as_dict()
    missing = [name for name in REQUIRED_PROVENANCE_FIELDS if name not in record]
    assert missing == []


def test_a_provenance_record_missing_a_required_field_fails():
    with pytest.raises(ProvenanceError, match="missing required fields"):
        GenerationProvenance({"provider": "gemini"})


def test_provenance_refuses_a_credential_field():
    record = {name: None for name in REQUIRED_PROVENANCE_FIELDS}
    record["api_key"] = "AIza" + "q" * 35
    with pytest.raises(ProvenanceError, match="forbidden credential fields"):
        GenerationProvenance(record)


def test_provenance_redacts_credential_shaped_strings(config, ontology, valid_candidate,
                                                      make_request):
    record = _provenance(config, ontology, valid_candidate, make_request,
                         generator_code_commit="build with key=AIza" + "w" * 35).as_dict()
    assert "AIza" not in json.dumps(record)


def test_provenance_records_the_sdk_and_thinking_configuration(config, ontology, valid_candidate,
                                                               make_request):
    record = _provenance(config, ontology, valid_candidate, make_request).as_dict()
    assert record["thinking_level"] == "medium"
    assert record["max_output_tokens"] == config.max_output_tokens
    assert record["billing_tier"] == "free"
    assert record["raw_response_sha256"]
    assert record["parsed_recipe_sha256"]


def test_request_identity_changes_when_any_frozen_input_changes(make_request):
    baseline = make_request().request_sha256
    assert make_request(model_id="gemini-3.5-flash").request_sha256 != baseline
    assert make_request(thinking_level="high").request_sha256 != baseline
    assert make_request(system_instruction="different role text").request_sha256 != baseline
    assert make_request(ontology_identity="0" * 64).request_sha256 != baseline
    assert make_request(prompt_template_identity="1" * 64).request_sha256 != baseline
    assert make_request(response_json_schema={"type": "object"}).request_sha256 != baseline
    assert make_request().request_sha256 == baseline          # and is stable


def test_identity_chain_is_ordered_and_sensitive():
    base = dict(ontology_identity="a" * 64, schema_identity="b" * 64,
                prompt_template_identity="c" * 64, provider_config_identity="d" * 64)
    chain = identity_chain(**base)
    assert chain["chain_order"][0] == "ontology_identity"
    assert chain["chain_order"][-1] == "canonical_recipe_identity"
    assert chain["resolved_links"] == 4
    changed = identity_chain(**{**base, "prompt_template_identity": "e" * 64})
    assert changed["chain_identity"] != chain["chain_identity"]
    extended = identity_chain(**base, raw_response_identity="f" * 64)
    assert extended["chain_identity"] != chain["chain_identity"]
    assert extended["resolved_links"] == 5
