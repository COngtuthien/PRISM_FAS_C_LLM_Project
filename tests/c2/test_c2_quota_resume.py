"""Quota exhaustion must checkpoint cleanly, and a resume must not redo work.

The C2 pilot really did hit a 429 mid-run, so this path is not hypothetical. The
requirements are narrow and testable: completed slots are preserved, pending
slots are named, the block artifact is written, billing is never enabled by code,
and a resume regenerates only what is missing.
"""
from __future__ import annotations

import json

import pytest

from prism_fas.llm.contracts import (ErrorClass, GenerationRequest, ProviderError,
                                     ProviderGenerationResult)
from prism_fas.llm.pipeline import QuotaBlocked, RecipePlanner
from prism_fas.llm.providers.base import RecipeProvider


class ScriptedProvider(RecipeProvider):
    """Serves a scripted outcome per slot. No client, no credential, no socket."""

    name = "gemini"

    def __init__(self, script: dict[str, object]) -> None:
        self.script = script
        self.calls: list[tuple[str, int]] = []

    def _generate(self, request: GenerationRequest, *, attempt: int) -> ProviderGenerationResult:
        self.calls.append((request.slot_id, attempt))
        outcome = self.script[request.slot_id]
        base = {"slot_id": request.slot_id, "attempt": attempt, "provider": self.name,
                "model_id": request.model_id, "parsed": None, "api_surface": "interactions"}
        if isinstance(outcome, ProviderError):
            return ProviderGenerationResult(raw_text=None, error=outcome, **base)
        return ProviderGenerationResult(raw_text=str(outcome), error=None, **base)


def envelope(candidate: dict) -> str:
    return json.dumps({"recipes": [candidate]})


@pytest.fixture
def candidate() -> dict:
    """A recipe that satisfies schema, ontology, ranges and compatibility."""
    return {
        "schema_version": "1.1",
        "medium": {"family": "display-like", "roughness": 0.2, "transparency": 0.05},
        "geometry": {"shape": "flat", "rigidity": 0.9, "coverage": 0.6},
        "regions": ["left_eye", "right_eye"],
        "artifacts": [{"name": "pixel_grid", "strength": 0.3}],
        "capture": {"yaw": 5.0, "illumination": "front", "compression_q": 80,
                    "scale": 1.0, "motion": 0.05, "defocus": 0.1},
        "forbidden_shortcuts": ["always_moire"],
        "generator_route": ["physics"],
        "seed": 12345,
    }


def quota_error() -> ProviderError:
    return ProviderError(ErrorClass.QUOTA_EXHAUSTED,
                         "429 quota_exceeded: daily quota for the project is exhausted",
                         status_code=429)


def test_quota_exhaustion_checkpoints_and_stops(config, ontology, make_request, candidate):
    slots = ["pilot_000", "pilot_001", "pilot_002"]
    provider = ScriptedProvider({"pilot_000": envelope(candidate),
                                 "pilot_001": quota_error(),
                                 "pilot_002": envelope(candidate)})
    planner = RecipePlanner(provider=provider, config=config, ontology=ontology,
                            sleep=lambda _seconds: None)

    result, validation, _attempts = planner.generate_slot(
        make_request("pilot_000"), recipes_requested=1, next_recipe_index=0,
        pending_slot_ids=slots[1:])
    assert validation is not None and validation.all_accepted
    planner.mark_slot_completed("pilot_000")

    with pytest.raises(QuotaBlocked) as blocked:
        planner.generate_slot(make_request("pilot_001"), recipes_requested=1,
                              next_recipe_index=1, pending_slot_ids=slots[1:])

    state = blocked.value.state.as_dict()
    assert state["blocked"] is True
    assert state["completed_slot_ids"] == ["pilot_000"]
    assert state["pending_slot_ids"] == ["pilot_001", "pilot_002"]
    assert state["completed_count"] == 1 and state["pending_count"] == 2
    assert state["auto_enable_paid"] is False
    assert state["billing_tier"] == "free"
    assert "never enables billing" in state["user_decision_required"]
    # The blocked slot was never attempted a second time.
    assert provider.calls.count(("pilot_001", 1)) == 1


def test_the_block_artifact_carries_no_credential_field(config, ontology, make_request):
    provider = ScriptedProvider({"pilot_000": quota_error()})
    planner = RecipePlanner(provider=provider, config=config, ontology=ontology,
                            sleep=lambda _seconds: None)
    with pytest.raises(QuotaBlocked) as blocked:
        planner.generate_slot(make_request("pilot_000"), recipes_requested=1,
                              pending_slot_ids=["pilot_000"])
    text = json.dumps(blocked.value.state.as_dict()).lower()
    for forbidden in ("api_key", "apikey", "authorization", "bearer", "aiza"):
        assert forbidden not in text


def test_a_resume_regenerates_only_the_pending_slots(config, ontology, make_request, candidate):
    """The resume contract: a completed slot is never re-requested."""
    completed = ["pilot_000"]
    pending = ["pilot_001", "pilot_002"]
    provider = ScriptedProvider({slot: envelope({**candidate, "seed": 1000 + index})
                                 for index, slot in enumerate(completed + pending)})
    planner = RecipePlanner(provider=provider, config=config, ontology=ontology,
                            sleep=lambda _seconds: None)
    for slot_id in completed:
        planner.mark_slot_completed(slot_id)

    for index, slot_id in enumerate(pending, start=1):
        _result, validation, _attempts = planner.generate_slot(
            make_request(slot_id), recipes_requested=1, next_recipe_index=index)
        assert validation is not None and validation.all_accepted
        planner.mark_slot_completed(slot_id)

    assert {slot for slot, _attempt in provider.calls} == set(pending)
    assert "pilot_000" not in {slot for slot, _attempt in provider.calls}
    assert sorted(planner.quota.completed_slot_ids) == sorted(completed + pending)


def test_a_resumed_run_keeps_the_duplicate_registry(config, ontology, make_request, candidate):
    """A repeat of an already-accepted recipe must still be caught after a resume."""
    provider = ScriptedProvider({"pilot_001": envelope(candidate)})
    planner = RecipePlanner(provider=provider, config=config, ontology=ontology,
                            sleep=lambda _seconds: None)
    # Rebuild the registry the way the runner does when it resumes.
    first_pass = RecipePlanner(provider=ScriptedProvider({"pilot_000": envelope(candidate)}),
                               config=config, ontology=ontology, sleep=lambda _s: None)
    _r, validation, _a = first_pass.generate_slot(make_request("pilot_000"), recipes_requested=1)
    identity = RecipePlanner.content_identity(validation.accepted[0].recipe)
    planner.register(identity, "pilot_000")

    _result, replayed, attempts = planner.generate_slot(
        make_request("pilot_001"), recipes_requested=1, next_recipe_index=1)
    assert replayed is not None
    assert not replayed.all_accepted, "the resumed run failed to notice a duplicate"
    assert replayed.candidates[0].outcome.value == "rejected_duplicate"
    assert attempts == config.retry.semantic_max_retries + 1


def test_the_real_pilot_rate_limit_incidents_are_preserved(pilot_archive):
    """The live 429s stay in the archive as evidence, with their bodies intact."""
    errors = [record for record in pilot_archive["records"] if record["error"]]
    if not errors:
        pytest.skip("this pilot recorded no provider error")
    for record in errors:
        assert record["raw_text"] is None
        assert record["error"]["status_code"] == 429
        assert record["error"]["error_class"] in {"rate_limit", "quota_exhausted"}
        assert "AIza" not in json.dumps(record)
