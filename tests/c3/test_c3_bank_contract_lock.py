"""The superseding pre-scientific C3 bank-contract lock.

It must reproduce from live code, name what it supersedes, prove zero affected
scientific requests, and never claim that a bank exists.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism_fas.llm.bank_lock import canonical_text, sha256_text
from prism_fas.llm.selection_contract import assemble

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports" / "c3" / "v15_selection_contract"
LOCK_PATH = OUT / "C3_BANK_CONTRACT_LOCK.json"

PRELIMINARY_LOCK_IDENTITY = "7ee96d3abee3f3b579c2dc6fe47ea27ff51ee3c2e956a1ff16b1ca85f5753fba"
GENERATION_CONTRACT_IDENTITY = (
    "884bce03b4f40a4ffbbef30f14c2216a6166a0ee1e8a6f6facb163f8bb3cdd85")
SELECTION_CONTRACT_IDENTITY = (
    "3d4675ba16b39d10f0e888f3c523ea540647544a436ff387bc84f2c17eced070")
BANK_CONTRACT_IDENTITY = (
    "d6105f8de601ae94cb0d46e087a0ebe664b3b5df9d193f5797e306bfe4fe03b8")


@pytest.fixture(scope="session")
def lock_v15():
    if not LOCK_PATH.exists():
        pytest.skip("run scripts/c3_selection_contract_freeze.py")
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def test_the_lock_body_hash_is_reproducible(lock_v15):
    body = {key: value for key, value in lock_v15.items() if key != "bank_lock_identity"}
    assert sha256_text(canonical_text(body)) == lock_v15["bank_lock_identity"]


def test_the_identities_reproduce_from_live_code(ontology, route_policy):
    record = assemble(repo=REPO, ontology=ontology, route_policy=route_policy,
                      generation_contract_identity=GENERATION_CONTRACT_IDENTITY)
    assert record["c3_selection_contract_identity"] == SELECTION_CONTRACT_IDENTITY
    assert record["c3_bank_contract_identity"] == BANK_CONTRACT_IDENTITY


def test_the_lock_binds_all_three_identities(lock_v15):
    identities = lock_v15["contract_identities"]
    assert identities["c3_generation_contract_identity"] == GENERATION_CONTRACT_IDENTITY
    assert identities["c3_selection_contract_identity"] == SELECTION_CONTRACT_IDENTITY
    assert identities["c3_bank_contract_identity"] == BANK_CONTRACT_IDENTITY


def test_the_lock_records_the_supersession_correctly(lock_v15):
    supersedes = lock_v15["supersedes"]
    assert supersedes["supersedes_bank_lock_identity"] == PRELIMINARY_LOCK_IDENTITY
    assert supersedes["reason"] == (
        "SELECTION_CONTRACT_WAS_REQUIRED_BY_GOVERNING_SPEC_BUT_NOT_"
        "IDENTITY_BOUND_BEFORE_C3_GENERATION")
    assert supersedes["scientific_requests_before_supersession"] == 0
    assert supersedes["old_lock_bytes_preserved"] is True


def test_the_lock_status_is_pre_scientific(lock_v15):
    assert lock_v15["status"] == "PRE_SCIENTIFIC_SUPERSEDING_CONTRACT_LOCK"


def test_the_lock_does_not_claim_a_bank_exists(lock_v15):
    state = lock_v15["state_at_freeze"]
    assert state["c3_scientific_logical_requests"] == 0
    assert state["c3_scientific_candidate_slots"] == 0
    assert state["raw_candidate_archives_created"] == 0
    assert state["recipe_banks_selected"] == 0
    assert state["recipe_bank_locks_created"] == 0
    assert state["live_provider_calls_in_this_task"] == 0
    # It must not be mistaken for a completed per-arm recipe bank lock.
    assert "RECIPE_BANK_LOCK" not in lock_v15["bank_lock_schema_version"].upper()
    assert "recipes" not in lock_v15
    assert "selected_shas" not in json.dumps(lock_v15)


def test_the_lock_carries_the_spec_identity(lock_v15):
    assert lock_v15["spec"]["sha256"] == (
        "ad8495f2576607546ff8c3bd4f47991197cbb3802265a599d1808aa1a97066e5")


def test_the_lock_freezes_the_selection_contract(lock_v15):
    contract = lock_v15["selection_contract"]
    assert contract["version"] == "prism_c3_selection_v1"
    assert contract["raw_candidate_slots_per_arm"] == 384
    assert contract["minimum_eligible_pool_per_arm"] == 320
    assert contract["final_bank_size_per_arm"] == 256
    assert contract["quotas"]["medium"]["hard_min"] == 32
    assert contract["eligibility_order"][2] == "scientific_route_policy"
    assert "lexicographically smallest" in contract["stage_5_rule"]


def test_the_lock_freezes_both_control_arm_schedules(lock_v15):
    arms = lock_v15["control_arm_schedules"]
    assert arms["slots_per_arm"] == 384
    assert len(arms["rnd_schedule_identity"]) == 64
    assert len(arms["det_schedule_identity"]) == 64
    assert arms["rnd_schedule_identity"] != arms["det_schedule_identity"]


def test_the_lock_keeps_the_prohibitions_in_force(lock_v15):
    text = " ".join(lock_v15["prohibitions_still_in_force"]).lower()
    for banned in ("live c3 generation", "gpu training", "gpat training", "synthesis",
                   "detector training", "siw access", "manual recipe selection",
                   "validator weakening"):
        assert banned in text


def test_the_supersession_evidence_matches_the_lock():
    path = OUT / "C3_SUPERSESSION_EVIDENCE.json"
    if not path.exists():
        pytest.skip("run scripts/c3_selection_contract_freeze.py")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    assert evidence["old_lock"]["bank_lock_identity"] == PRELIMINARY_LOCK_IDENTITY
    assert evidence["old_lock_edited"] is False
    assert evidence["old_lock_deleted"] is False
    assert evidence["scientific_requests_before_supersession"] == 0
    assert evidence["c3_scientific_request_audit"]["c3_scientific_logical_requests"] == 0


def test_the_historical_preliminary_lock_is_byte_identical():
    """Byte-level, not just body-hash.

    The comparison is made on the LF-NORMALIZED bytes. The lock predates the
    repository's .gitattributes rules, so Git checks it out with CRLF on Windows
    and LF elsewhere; a raw-byte assertion would be testing the platform rather
    than the artifact. Folding CRLF to LF gives the committed blob's hash on
    every platform, which is exactly the claim worth making: nobody has edited
    this file.
    """
    import hashlib

    path = OUT / "C3_PRELIMINARY_LOCK_INTEGRITY.json"
    if not path.exists():
        pytest.skip("run scripts/c3_selection_contract_freeze.py")
    integrity = json.loads(path.read_text(encoding="utf-8"))
    recorded = integrity["before"]["file_sha256_lf_normalized"]

    preliminary = REPO / "reports" / "c3" / "C3_BANK_LOCK.json"
    raw = preliminary.read_bytes()
    normalized = raw.decode("utf-8").replace("\r\n", "\n").encode("utf-8")

    assert hashlib.sha256(normalized).hexdigest() == recorded, (
        "the historical preliminary lock's content changed")
    assert integrity["after"]["file_sha256_lf_normalized"] == recorded
    assert integrity["before"]["body_hash_reproducible"] is True
    assert integrity["edited"] is False


def test_the_selector_determinism_report_records_a_clean_offline_run():
    path = OUT / "C3_SELECTOR_DETERMINISM_REPORT.json"
    if not path.exists():
        pytest.skip("run scripts/c3_selector_determinism_report.py")
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["determinism_holds"] is True
    assert report["scientific_status"] == "NOT_RUN"
    assert report["scientific_eligible"] is False

    attestation = report["offline_attestation"]
    for key in ("live_provider_calls", "network_calls", "gpu_or_modal_jobs",
                "target_label_or_metric_reads", "c3_scientific_logical_requests",
                "c3_scientific_candidate_slots"):
        assert attestation[key] == 0, f"{key} is not zero"

    probes = report["probes"]
    assert probes["repeated_run_identity"]["identical"] is True
    assert probes["input_order_invariance"]["invariant"] is True
    assert probes["stage_5_canonical_tie_break"][
        "stage_5_returned_the_lexicographically_smallest_set"] is True
    for arm in ("RND", "DET"):
        arm_report = probes["control_arm_schedule_reproducibility"][arm]
        assert arm_report["slots"] == 384
        assert arm_report["unique_slot_ids"] == 384
        assert arm_report["reproducible"] is True
        assert arm_report["route_declared_by_every_slot"] is True

    budget = probes["real_320_to_256_budget"]
    assert budget["eligible"] == 320
    assert budget["selected"] == 256
    assert budget["selected_plus_rejected"] == 320
    assert budget["hard_violations"] == []
    assert budget["quota_table_modified"] is False


def test_the_acceptance_artifact_does_not_claim_scientific_completion():
    path = OUT / "C3_SELECTION_CONTRACT_ACCEPTANCE.json"
    if not path.exists():
        pytest.skip("run scripts/c3_selection_contract_freeze.py")
    acceptance = json.loads(path.read_text(encoding="utf-8"))
    assert acceptance["scientific_status"] == "NOT_RUN"
    assert acceptance["scientific_eligible"] is False
    assert acceptance["execution_profile"] == "validate"
    assert acceptance["checks"]["no_candidates_generated"] is True
    assert acceptance["checks"]["no_live_provider_call"] is True
