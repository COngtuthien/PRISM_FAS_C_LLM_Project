"""The C3 BANK_LOCK is the gate before any scientific generation.

A lock is only worth anything if it (a) reproduces the approved identity from the
live code, (b) refuses to exist when the repository has drifted, and (c) cannot be
quietly rewritten. All three are tested here, offline.
"""
from __future__ import annotations

import json

import pytest

from prism_fas.llm.bank_lock import (APPROVED_C3_CONTRACT,
                                     APPROVED_C3_GENERATION_CONTRACT_IDENTITY,
                                     BANK_LOCK_SCHEMA_VERSION, COMPOSITE_COMPONENT_KEYS,
                                     BankLockError, build_lock, canonical_text,
                                     check_against_approval, composite_identity,
                                     derive_components, sha256_text, verify_lock,
                                     write_lock_once)


# --------------------------------------------------------------- the identity
def test_the_composite_reproduces_the_approved_identity_from_live_code(context):
    """Nothing is copied from the approval text; everything is re-derived."""
    components = derive_components(context)
    assert composite_identity(components) == APPROVED_C3_GENERATION_CONTRACT_IDENTITY


def test_every_component_matches_what_the_user_approved(context):
    drift = [check.as_dict() for check in check_against_approval(derive_components(context))
             if not check.matches]
    assert drift == [], f"the repository drifted from the approved contract: {drift}"


def test_the_lock_binds_the_approved_composite(lock):
    assert lock["composite"]["c3_generation_contract_identity"] == \
        APPROVED_C3_GENERATION_CONTRACT_IDENTITY
    assert lock["user_approval"]["approved_composite_identity"] == \
        APPROVED_C3_GENERATION_CONTRACT_IDENTITY
    assert lock["status"] == "FROZEN"
    assert lock["bank_lock_schema_version"] == BANK_LOCK_SCHEMA_VERSION


def test_the_lock_body_hash_is_reproducible(lock):
    body = {key: value for key, value in lock.items() if key != "bank_lock_identity"}
    assert sha256_text(canonical_text(body)) == lock["bank_lock_identity"]


def test_the_recorded_canonical_text_reproduces_the_composite(lock):
    recorded = lock["composite"]["canonical_text"]
    assert sha256_text(recorded) == lock["composite"]["c3_generation_contract_identity"]
    assert json.loads(recorded) == lock["components"]


def test_the_lock_verifies_against_the_live_repository(lock, context):
    result = verify_lock(lock, context)
    assert result["verified"], result["problems"]
    assert result["problems"] == []


def test_the_committed_verification_artifact_agrees(verification):
    assert verification["verified"] is True
    assert verification["problems"] == []
    assert verification["composite_in_file"] == verification["composite_recomputed"]
    assert verification["composite_in_file"] == verification["composite_approved"]
    assert verification["network_calls"] == 0
    assert verification["credential_read"] is False


# ------------------------------------------------------------ drift refusal
@pytest.mark.parametrize("component", [
    "model_id", "system_prompt_identity", "ontology_identity", "route_policy_identity",
    "single_recipe_schema_identity", "batch_envelope_schema_identity",
    "coverage_quota_identity", "allow_ontology_aliases", "thinking_level",
])
def test_changing_any_component_changes_the_composite(context, component):
    """The invalidation rule, tested rather than asserted in prose."""
    components = derive_components(context)
    baseline = composite_identity(components)
    altered = dict(components)
    value = altered[component]
    altered[component] = (not value) if isinstance(value, bool) else f"{value}-changed"
    assert composite_identity(altered) != baseline


def test_a_drifted_repository_cannot_produce_a_lock(context, monkeypatch):
    """Refusal is a precondition, not a field in the output."""
    original = context.as_contract_record()

    def drifted():
        return {**original, "model_id": "gemini-3.5-flash"}

    monkeypatch.setattr(context, "as_contract_record", drifted)
    with pytest.raises(BankLockError, match="drifted from the approved C3 contract"):
        build_lock(context)


def test_a_composite_missing_a_component_is_refused():
    components = {key: "x" for key in COMPOSITE_COMPONENT_KEYS if key != "ontology_identity"}
    with pytest.raises(BankLockError, match="missing"):
        composite_identity(components)


def test_a_composite_with_an_unapproved_component_is_refused():
    components = {key: "x" for key in COMPOSITE_COMPONENT_KEYS}
    components["smuggled_component"] = "x"
    with pytest.raises(BankLockError, match="unapproved keys"):
        composite_identity(components)


# ------------------------------------------------------------- immutability
def test_a_lock_is_never_silently_rewritten(tmp_path, lock):
    path = tmp_path / "C3_BANK_LOCK.json"
    assert write_lock_once(path, lock) == "created"
    assert write_lock_once(path, lock) == "unchanged", "an identical rewrite must be a no-op"

    tampered = {**lock, "bank_lock_identity": "0" * 64}
    with pytest.raises(BankLockError, match="already exists with a different identity"):
        write_lock_once(path, tampered)
    # The file on disk is untouched by the refused write.
    assert json.loads(path.read_text(encoding="utf-8"))["bank_lock_identity"] == \
        lock["bank_lock_identity"]


def test_the_lock_declares_itself_immutable(lock):
    assert lock["immutability"]["rewrite_permitted"] is False


# ------------------------------------------------------- contents and policy
def test_the_lock_freezes_the_route_contract(lock):
    route = lock["route_contract"]
    assert route["required_generator_route"] == ["physics", "gpat"]
    assert route["physics_only_accepted"] is False
    assert route["gpat_only_accepted"] is False
    assert route["gpat_only_class_exists"] is False
    assert route["silent_repair_permitted"] is False
    assert route["route_policy_identity"] == APPROVED_C3_CONTRACT["route_policy_identity"]


def test_the_lock_freezes_the_request_schedule(lock):
    schedule = lock["scientific_request_schedule"]
    assert schedule["requests"] == 12
    assert schedule["objects_per_request"] == 32
    assert schedule["raw_slots"] == 384
    assert schedule["minimum_unique_pool"] == 320
    assert schedule["final_bank"] == 256
    assert "no manual cherry-picking" in schedule["selection"]


def test_the_lock_records_the_free_tier_policy(lock):
    policy = lock["user_approval"]["approved_free_tier_policy"]
    assert policy["billing_tier"] == "free"
    assert policy["auto_enable_paid"] is False
    assert policy["quota_never_changes_the_contract"] is True


def test_the_quota_snapshot_is_honest_rather_than_invented(lock):
    snapshot = lock["quota_snapshot"]
    assert snapshot["availability"] == "NOT_PROGRAMMATICALLY_AVAILABLE"
    for field in ("rpm", "tpm", "rpd"):
        assert snapshot[field] == "NOT_AVAILABLE", f"{field} must not carry an invented number"
    assert snapshot["manual_step_before_c3"]


def test_the_lock_prohibits_training_and_synthesis_during_c3(lock):
    prohibitions = " ".join(lock["prohibitions_during_c3"]).lower()
    for banned in ("gpu training", "gpat training", "synthetic image generation",
                   "detector training", "siw label access", "siw metric use"):
        assert banned in prohibitions


def test_the_lock_contains_no_recipe_and_no_credential(lock):
    text = json.dumps(lock)
    assert "recipes" not in lock
    assert "AIza" not in text
    assert "ya29." not in text
    lowered = text.lower()
    for banned in ('"api_key"', "'api_key'", '"authorization"'):
        assert banned not in lowered


def test_the_lock_records_that_no_pilot_recipe_enters_c3(lock):
    evidence = lock["evidence"]
    assert evidence["pilot_recipes_entering_c3"] == 0
    assert evidence["c2_c2b_c2c_recipes_entering_c3"] == 0
    assert evidence["c2b"]["result"] == "BATCH_SHAPE_FAIL"
    assert evidence["c2b"]["preserved_unchanged"] is True


def test_no_c3_scientific_request_has_been_made():
    """The lock is the gate; passing through it is a separate, later action."""
    from pathlib import Path

    reports = Path(__file__).resolve().parents[2] / "reports" / "c3"
    generated = sorted(path.name for path in reports.glob("*.json")
                       if path.name not in {"C3_BANK_LOCK.json",
                                            "C3_BANK_LOCK_VERIFICATION.json"})
    assert generated == [], f"unexpected C3 generation artifacts present: {generated}"
