"""The archived C2 pilot must reproduce itself offline, exactly.

A hosted model is not reproducible: asking Gemini the same question twice may
return different text. The reproducible scientific artifact is therefore the
archived response, and these tests are what make that claim checkable - the
archive alone, with no network and no credential, must reproduce every accept
and reject decision, every canonical recipe identity and every operator graph.
"""
from __future__ import annotations

import json

import pytest

from prism_fas.llm.pipeline import RecipePlanner
from prism_fas.llm.providers.replay import ReplayArchive, ReplayRecipeProvider
from prism_fas.recipes.canonical import canonical_json, recipe_hash
from prism_fas.recipes.compile import compile_recipe
from prism_fas.recipes.validate import validate_payload

#: The bank id the pilot compiled under; the graph hash is bound to it.
PILOT_BANK_ID = "c2-pilot-disposable"

REPLAY_KEYS = ("slot_id", "attempt", "raw_text", "provider", "model_id", "model_version",
               "finish_reason", "usage", "provider_request_id", "provider_seed", "sdk_version",
               "api_surface", "request_sha256")


def build_archive(records: list[dict]) -> ReplayArchive:
    return ReplayArchive.from_records([{key: record[key] for key in REPLAY_KEYS}
                                       for record in records if record["raw_text"] is not None])


def replay_slots(archive_records, state, ontology, config, make_request):
    """Re-run every archived attempt through a fresh pipeline, in call order."""
    provider = ReplayRecipeProvider(build_archive(archive_records), strict=False)
    planner = RecipePlanner(provider=provider, config=config, ontology=ontology,
                            sleep=lambda _seconds: None)
    by_slot: dict[str, list[dict]] = {}
    for record in archive_records:
        by_slot.setdefault(record["slot_id"], []).append(record)

    outcome: dict[str, dict] = {}
    for slot_state in state["slots"]:
        slot_id = slot_state["slot_id"]
        request = make_request(slot_id)
        accepted = None
        for record in sorted(by_slot.get(slot_id, []),
                             key=lambda item: item.get("sequence", item["attempt"])):
            if record["raw_text"] is None:
                continue
            served = provider.generate(request, attempt=record["attempt"])
            assert served.raw_text == record["raw_text"], "replay served different bytes"
            validation = planner.validate_response(
                served.raw_text, slot_id=slot_id, recipes_requested=1,
                next_recipe_index=slot_state["slot_index"])
            if validation.all_accepted and accepted is None:
                accepted = validation.accepted[0]
        outcome[slot_id] = {"accepted": accepted,
                            "status": "accepted" if accepted else "exhausted"}
    return outcome, provider


def test_the_archive_holds_one_record_for_every_attempt(pilot_archive, pilot_state):
    """Every provider call is archived, including calls the run later reopened.

    `provider_calls` counts the attempts of the pass that finished a slot. A slot
    that was reopened after receiving no response at all keeps its earlier
    attempts in the archive as evidence, so the archive is a superset.
    """
    calls = sum(slot["provider_calls"] for slot in pilot_state["slots"])
    assert pilot_archive["record_count"] == len(pilot_archive["records"])
    assert pilot_archive["record_count"] >= calls
    keys = [(record["slot_id"], record.get("sequence", record["attempt"]))
            for record in pilot_archive["records"]]
    assert len(keys) == len(set(keys)), "an archived attempt is not uniquely identified"


def test_every_archived_attempt_carries_its_raw_response_hash(pilot_archive):
    import hashlib
    for record in pilot_archive["records"]:
        if record["raw_text"] is None:
            assert record["raw_response_sha256"] is None
            assert record["error"] is not None, "an attempt with no text must carry an error"
            continue
        digest = hashlib.sha256(record["raw_text"].encode("utf-8")).hexdigest()
        assert record["raw_response_sha256"] == digest


def test_archived_live_responses_replay_offline(pilot_archive, pilot_state, ontology, config,
                                                make_request):
    """The headline claim: the archive alone reproduces the live run."""
    outcome, _provider = replay_slots(pilot_archive["records"], pilot_state, ontology, config,
                                      make_request)
    assert outcome, "no slot replayed"
    for slot in pilot_state["slots"]:
        assert outcome[slot["slot_id"]]["status"] == slot["final_status"], (
            f"{slot['slot_id']} replayed to a different verdict")


def test_replay_reproduces_the_canonical_recipe_identity(pilot_archive, pilot_state, ontology,
                                                         config, make_request):
    outcome, _provider = replay_slots(pilot_archive["records"], pilot_state, ontology, config,
                                      make_request)
    for slot in pilot_state["slots"]:
        if slot["final_status"] != "accepted":
            continue
        accepted = outcome[slot["slot_id"]]["accepted"]
        assert accepted is not None
        assert accepted.recipe_identity == slot["recipe_identity"]
        assert accepted.canonical_text == slot["canonical_recipe"]


def test_replay_performs_zero_network_calls(pilot_archive, pilot_state, ontology, config,
                                            make_request):
    """`no_network` is autouse, so reaching a socket would already have failed.

    This test states the property explicitly and also proves the replay provider
    is structurally incapable of a call: it holds no client and no credential.
    """
    outcome, provider = replay_slots(pilot_archive["records"], pilot_state, ontology, config,
                                     make_request)
    assert outcome
    described = provider.describe()
    assert described["network"] is False
    assert not any("client" in name.lower() or "key" in name.lower()
                   for name in vars(provider))


def test_every_accepted_pilot_recipe_revalidates_offline(pilot_state, ontology):
    accepted = [slot for slot in pilot_state["slots"] if slot["final_status"] == "accepted"]
    assert accepted, "the pilot accepted no recipe"
    for slot in accepted:
        payload = json.loads(slot["canonical_recipe"])
        recipe, issues = validate_payload(payload, ontology, canonicalize=False)
        assert recipe is not None, f"{slot['slot_id']} no longer parses"
        assert issues == [], f"{slot['slot_id']} no longer validates: {issues}"
        assert recipe_hash(recipe) == slot["recipe_identity"]
        assert canonical_json(recipe) == slot["canonical_recipe"]


def test_every_accepted_pilot_recipe_recompiles_offline(pilot_state, ontology):
    accepted = [slot for slot in pilot_state["slots"]
                if slot["final_status"] == "accepted" and slot["compiler_status"] == "compiled"]
    assert accepted, "no accepted recipe reached the compiler"
    for slot in accepted:
        payload = json.loads(slot["canonical_recipe"])
        recipe, issues = validate_payload(payload, ontology, canonicalize=False)
        assert recipe is not None and issues == []
        graph = compile_recipe(recipe, ontology, bank_id=PILOT_BANK_ID)
        assert graph.graph_hash == slot["graph_hash"], f"{slot['slot_id']} graph hash drifted"
        assert graph.conditioning_dimension == 41
        assert graph.region_mask_policy["policy"] == "parsing_first_geometry_fallback"


def test_compilation_is_deterministic_across_repeats(pilot_state, ontology):
    slot = next((item for item in pilot_state["slots"] if item["compiler_status"] == "compiled"),
                None)
    if slot is None:
        pytest.skip("no compiled slot in the archive")
    payload = json.loads(slot["canonical_recipe"])
    recipe, _ = validate_payload(payload, ontology, canonicalize=False)
    first = compile_recipe(recipe, ontology, bank_id=PILOT_BANK_ID)
    second = compile_recipe(recipe, ontology, bank_id=PILOT_BANK_ID)
    assert first.graph_hash == second.graph_hash
    assert first.canonical_json() == second.canonical_json()


def test_the_archive_refuses_to_replay_across_a_changed_request_identity(pilot_archive,
                                                                        make_request):
    """A response produced under a different prompt/schema/model must not be
    silently reused: that would mix two scientific identities."""
    served = [record for record in pilot_archive["records"] if record["raw_text"] is not None]
    if not served:
        pytest.skip("no archived response")
    record = dict({key: served[0][key] for key in REPLAY_KEYS})
    record["request_sha256"] = "0" * 64
    provider = ReplayRecipeProvider(ReplayArchive.from_records([record]), strict=False)
    with pytest.raises(ValueError, match="refusing to replay across identities"):
        provider.generate(make_request(record["slot_id"]), attempt=record["attempt"])


def test_the_smoke_response_also_replays(smoke_archive, ontology, config, make_request):
    records = [record for record in smoke_archive["records"] if record["raw_text"] is not None]
    if not records:
        pytest.skip("no archived smoke response")
    provider = ReplayRecipeProvider(build_archive(records), strict=False)
    planner = RecipePlanner(provider=provider, config=config, ontology=ontology,
                            sleep=lambda _seconds: None)
    for record in records:
        request = make_request(record["slot_id"])
        served = provider.generate(request, attempt=record["attempt"])
        assert served.raw_text == record["raw_text"]
        validation = planner.validate_response(served.raw_text, slot_id=record["slot_id"],
                                               recipes_requested=1)
        assert validation.all_accepted, "an archived smoke response no longer validates"
