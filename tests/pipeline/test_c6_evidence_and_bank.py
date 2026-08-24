"""The C7/C8 input contract: what C6 froze, and the bank the detector opens.

Two things are proved here, and both were unproved before this milestone:

* `verify_c6_evidence` is the STRICT verifier C7 and C8 gate on, and it refuses
  every way a C6 closure can be wrong — not merely a missing file. A stage that
  accepted `reports/full/c6` existing would train against a directory.
* `C6MatchedBankReader` turns one arm's frozen bank lock plus the C5 candidate
  bytes into exactly the surface `M9TrainingDataset` consumes, fail-closed on
  membership, on payload hashes and on retained generation failures.

The fixture writes production-shaped artifacts through the modules that produce
the real ones, so a schema change in `bank_lock_payload` or in `CandidateRecord`
breaks these tests rather than sliding past them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from c6_bank_fixture import (ARMS, PACKAGE_IDENTITY, RECIPE_BANK_IDENTITY,  # noqa: E402
                             build_c6_closure, install_c3_bank, write_candidate)
from prism_fas.detector import c6_bank  # noqa: E402
from prism_fas.evaluation import c6_evidence  # noqa: E402
from prism_fas.synthesis import c5_raw_generation as raw  # noqa: E402
from prism_fas.synthesis import c6_matched_bank as selector  # noqa: E402


@pytest.fixture
def closure(tmp_path, monkeypatch):
    """A verifiable closure at a tiny cardinality.

    The two frozen cardinality constants are monkeypatched rather than the
    verifier weakened: the verifier reads `FINAL_BANK_PER_ARM` and `PER_ROUTE`
    from the module that owns them, which is exactly the property that stops it
    drifting into a laxer second opinion.
    """
    per_route = 2
    monkeypatch.setattr(selector, "PER_ROUTE", per_route)
    monkeypatch.setattr(selector, "FINAL_BANK_PER_ARM", per_route * len(selector.ROUTES))
    return build_c6_closure(tmp_path, per_route=per_route), tmp_path


# --- the strict C6 verifier --------------------------------------------------

def test_a_complete_closure_verifies(closure) -> None:
    _built, repo = closure
    evidence = c6_evidence.verify_c6_evidence(repo)

    assert evidence.arms == tuple(sorted(ARMS))
    assert evidence.selected_profile == "NOMINAL"
    for arm in ARMS:
        bank = evidence.bank(arm)
        assert bank.final_bank_size == selector.FINAL_BANK_PER_ARM
        assert set(bank.by_route) == set(selector.ROUTES)
        assert all(count == selector.PER_ROUTE for count in bank.by_route.values())
        assert bank.selector_identity_sha256 == evidence.selector_identity_sha256


def test_the_report_form_does_not_raise(closure) -> None:
    _built, repo = closure
    report = c6_evidence.evidence_report(repo)

    assert report["valid"] is True
    assert report["problems"] == []
    assert report["evidence"]["target_access"] == 0


@pytest.mark.parametrize("mutation,expected", [
    ({"final_bank_size": 3}, "final bank"),
    ({"by_route": {"physics": 1, "gpat": 2}}, "route physics"),
    ({"q_used_for_selection": True}, "q_used_for_selection"),
    ({"usable_for_c7_c8_source_training": False}, "usable_for_c7_c8_source_training"),
    ({"target_access": 1}, "target_access"),
    ({"is_scientific_lock": False}, "is_scientific_lock"),
    ({"fixture_backed": True}, "fixture_backed"),
    ({"selector_identity_sha256": "0" * 64}, "selector identity"),
    ({"quality_threshold_identity": "0" * 64}, "threshold identity"),
    ({"quality_profile": "STRICT"}, "gated under profile"),
    ({"provenance_closure": {"closed": False, "unaccounted": []}}, "closed"),
    ({"provenance_closure": {"closed": True, "unaccounted": ["x"]}}, "unaccounted"),
])
def test_every_bank_lock_mutation_is_refused(closure, mutation, expected) -> None:
    """The verifier is only worth having if it REFUSES, one reason at a time."""
    _built, repo = closure
    path = repo / "reports/full/c6/C6_BANK_LOCK_DET.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(mutation)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(c6_evidence.C6EvidenceError) as caught:
        c6_evidence.verify_c6_evidence(repo)
    assert expected in str(caught.value)


def test_a_missing_arm_lock_is_refused_as_absent(closure) -> None:
    _built, repo = closure
    (repo / "reports/full/c6/C6_BANK_LOCK_LLM.json").unlink()

    with pytest.raises(c6_evidence.C6EvidenceMissing) as caught:
        c6_evidence.verify_c6_evidence(repo)
    assert caught.value.reason_code == "C6_EVIDENCE_ABSENT"


def test_a_profile_disagreement_between_lock_and_banks_is_refused(closure) -> None:
    """The two artifacts must describe ONE C6 run, not two."""
    _built, repo = closure
    path = repo / "reports/full/c6/C6_MATCHED_BANKS.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selected_profile"] = "PERMISSIVE"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(c6_evidence.C6EvidenceError) as caught:
        c6_evidence.verify_c6_evidence(repo)
    assert "different C6 runs" in str(caught.value)


def test_the_verifier_collects_every_problem_not_only_the_first(closure) -> None:
    _built, repo = closure
    path = repo / "reports/full/c6/C6_BANK_LOCK_RND.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update({"q_used_for_selection": True, "target_access": 2,
                    "usable_for_c7_c8_source_training": False})
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(c6_evidence.C6EvidenceError) as caught:
        c6_evidence.verify_c6_evidence(repo)
    assert len(caught.value.problems) >= 3


# --- the detector's view of one arm's bank -----------------------------------

def _open(repo: Path, built: dict, arm: str = "DET"):
    install_c3_bank(repo, arm, built["recipes"])
    evidence = c6_evidence.verify_c6_evidence(repo).bank(arm)
    return c6_bank.open_arm_bank(
        repo, arm=arm, evidence=evidence,
        candidates_root=built["candidates_root"],
        package_identity=PACKAGE_IDENTITY,
        recipe_bank_identity=RECIPE_BANK_IDENTITY)


def test_the_bank_presents_the_surface_the_dataset_consumes(closure) -> None:
    """Exactly the attributes `M9TrainingDataset` and `region_cache` read."""
    built, repo = closure
    bank = _open(repo, built)

    assert len(bank) == selector.FINAL_BANK_PER_ARM
    assert bank.lock["package_identity"] == PACKAGE_IDENTITY
    assert bank.backend == c6_bank.BACKEND
    assert bank.bank_id.startswith(c6_bank.BANK_ID_PREFIX)
    assert len(bank.identity) == 64
    assert len(bank.rows_identity()) == 64
    assert {row["route"] for row in bank.rows} == set(selector.ROUTES)

    sample = bank.sample(0)
    assert sample.image.shape[0] == 3
    assert sample.exact_mask.dtype.name == "bool"
    assert sample.artifact_map.shape[0] == 1
    assert 0.0 <= sample.quality_weight <= 1.0
    assert bank.primary_artifact_family(0) == "blur"
    assert bank.index_of(sample.synthetic_id) == 0


def test_the_bank_id_can_never_satisfy_an_m8_pin(closure) -> None:
    """Namespaced on purpose: an M8 v3 pin and a C6 arm bank are different banks."""
    from prism_fas.detector.synthetic_bank import FROZEN_BANK_ID, FROZEN_BANK_IDENTITY

    built, repo = closure
    bank = _open(repo, built)

    assert bank.bank_id != FROZEN_BANK_ID
    assert bank.identity != FROZEN_BANK_IDENTITY


def test_q_is_carried_as_a_weight_and_never_as_membership(closure) -> None:
    built, repo = closure
    bank = _open(repo, built)

    assert bank.lock["q_used_for_selection"] is False
    assert all(0.0 <= float(row["q"]) <= 1.0 for row in bank.rows)


def test_a_selected_candidate_with_no_c5_record_is_refused(closure) -> None:
    """A matched bank is never silently shortened."""
    built, repo = closure
    install_c3_bank(repo, "DET", built["recipes"])
    victim = built["banks"]["DET"]["selected"][0]["candidate_id"]
    (raw.candidate_dir(built["candidates_root"], "DET", victim)
     / raw.RECORD_NAME).unlink()

    with pytest.raises(c6_bank.C6BankError) as caught:
        _open(repo, built)
    assert "no C5 record" in str(caught.value)


def test_a_retained_generation_failure_is_never_a_training_sample(closure) -> None:
    built, repo = closure
    install_c3_bank(repo, "DET", built["recipes"])
    victim = built["banks"]["DET"]["selected"][0]
    write_candidate(built["candidates_root"], arm="DET",
                    candidate_id=victim["candidate_id"], route=victim["route"],
                    recipe_id=victim["recipe_id"], recipe_ordinal=0, position=0,
                    live_target_sample_id=victim["live_target_sample_id"], seed=1,
                    status=raw.FAILED_GENERATION)

    with pytest.raises(c6_bank.C6BankError) as caught:
        _open(repo, built)
    assert "negative provenance" in str(caught.value)


def test_altered_payload_bytes_are_refused_at_read_time(closure) -> None:
    """The bank C6 froze must be the bank on disk, byte for byte."""
    built, repo = closure
    bank = _open(repo, built)
    row = bank.rows[0]
    (built["candidates_root"] / row["image_relative_path"]).write_bytes(b"not a png")

    with pytest.raises(c6_bank.C6BankError) as caught:
        bank.sample(0)
    assert "does not match the SHA-256" in str(caught.value)


def test_a_candidate_from_another_source_package_is_refused(closure) -> None:
    built, repo = closure
    install_c3_bank(repo, "DET", built["recipes"])
    evidence = c6_evidence.verify_c6_evidence(repo).bank("DET")

    with pytest.raises(c6_bank.C6BankError) as caught:
        c6_bank.open_arm_bank(
            repo, arm="DET", evidence=evidence,
            candidates_root=built["candidates_root"],
            package_identity="0" * 64,
            recipe_bank_identity=RECIPE_BANK_IDENTITY)
    assert "rendered against source package" in str(caught.value)


def test_a_bank_lock_from_a_different_selection_is_refused(closure) -> None:
    """The evidence's selected-set digest is re-checked against the lock's own."""
    built, repo = closure
    install_c3_bank(repo, "DET", built["recipes"])
    payload = json.loads(
        (repo / "reports/full/c6/C6_BANK_LOCK_DET.json").read_text(encoding="utf-8"))

    with pytest.raises(c6_bank.C6BankError) as caught:
        c6_bank.C6MatchedBankReader.open(
            candidates_root=built["candidates_root"], arm="DET", bank_lock=payload,
            recipes=built["recipes"], package_identity=PACKAGE_IDENTITY,
            recipe_bank_identity=RECIPE_BANK_IDENTITY,
            expected_selected_set_sha256="0" * 64)
    assert "expected" in str(caught.value)


def test_the_bank_identity_moves_when_a_payload_moves(closure) -> None:
    """Identity is over the rows INCLUDING their payload hashes, so a re-render
    of one candidate produces a different bank rather than the same one."""
    built, repo = closure
    first = _open(repo, built).identity
    row = built["banks"]["DET"]["selected"][0]
    write_candidate(built["candidates_root"], arm="DET",
                    candidate_id=row["candidate_id"], route=row["route"],
                    recipe_id=row["recipe_id"], recipe_ordinal=0, position=0,
                    live_target_sample_id=row["live_target_sample_id"], seed=999)

    assert _open(repo, built).identity != first
