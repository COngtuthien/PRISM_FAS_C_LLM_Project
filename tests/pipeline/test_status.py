"""The dual status model (v1.5 Appendix L.3) and eligibility (L.2).

The one question these tests protect is "is this milestone scientifically
complete?". Every wrong way to answer yes is tried here: a SMOKE_PASS, a PASS
under the wrong profile, an artifact stamped eligible by a profile that cannot
grant it, and a blocked full run that never produced ancestry.
"""
from __future__ import annotations

import pytest

from prism_fas.pipeline.status import (ENGINEERING_STATUS, RUN_OUTCOME, SCIENTIFIC_STATUS,
                                       DualStatus, StatusError, assert_not_promoted,
                                       evidence_eligibility, scientifically_complete)


def test_the_l3_vocabularies_are_exactly_the_spec_values() -> None:
    assert ENGINEERING_STATUS == ("NOT_TESTED", "RUNNING", "SMOKE_PASS", "SMOKE_FAIL",
                                  "BLOCKED")
    assert SCIENTIFIC_STATUS == ("NOT_RUN", "RUNNING", "PASS", "FAIL", "BLOCKED")


def test_l8_preserves_every_outcome_including_the_negative_ones() -> None:
    for outcome in ("PASS", "FAIL", "DIVERGED", "INTERRUPTED", "BLOCKED"):
        assert outcome in RUN_OUTCOME


@pytest.mark.parametrize("engineering", ["VALIDATE_PASS", "OK", "pass", ""])
def test_an_invented_engineering_value_is_refused(engineering: str) -> None:
    with pytest.raises(StatusError):
        DualStatus(engineering=engineering, scientific="NOT_RUN")


@pytest.mark.parametrize("scientific", ["SMOKE_PASS", "COMPLETE", "pass"])
def test_an_invented_scientific_value_is_refused(scientific: str) -> None:
    with pytest.raises(StatusError):
        DualStatus(engineering="NOT_TESTED", scientific=scientific)


def test_smoke_pass_is_not_scientific_completion() -> None:
    status = DualStatus(engineering="SMOKE_PASS", scientific="NOT_RUN")
    for profile in ("validate", "smoke", "full"):
        assert not scientifically_complete(status, profile=profile)


def test_scientific_pass_outside_the_full_profile_is_not_completion() -> None:
    status = DualStatus(engineering="SMOKE_PASS", scientific="PASS")
    assert not scientifically_complete(status, profile="smoke")
    assert not scientifically_complete(status, profile="validate")


def test_only_a_full_profile_scientific_pass_is_completion() -> None:
    status = DualStatus(engineering="SMOKE_PASS", scientific="PASS")
    assert scientifically_complete(status, profile="full")


def test_both_axes_are_always_serialized_together() -> None:
    payload = DualStatus(engineering="BLOCKED", scientific="BLOCKED").as_dict()
    assert set(payload) == {"engineering_status", "scientific_status"}


def test_an_artifact_missing_either_l2_field_is_refused() -> None:
    with pytest.raises(StatusError, match="must serialize execution_profile"):
        assert_not_promoted({"execution_profile": "smoke"})
    with pytest.raises(StatusError, match="must serialize execution_profile"):
        assert_not_promoted({"scientific_eligible": False})


def test_a_smoke_artifact_claiming_eligibility_is_refused() -> None:
    with pytest.raises(StatusError, match="only the full profile"):
        assert_not_promoted({"execution_profile": "smoke", "scientific_eligible": True})


def test_a_validate_artifact_claiming_eligibility_is_refused() -> None:
    with pytest.raises(StatusError, match="only the full profile"):
        assert_not_promoted({"execution_profile": "validate", "scientific_eligible": True})


def test_a_conforming_artifact_passes() -> None:
    assert_not_promoted({"execution_profile": "smoke", "scientific_eligible": False})
    assert_not_promoted({"execution_profile": "full", "scientific_eligible": True})


@pytest.mark.parametrize("outcome", ["BLOCKED", "FAIL", "DIVERGED", "INTERRUPTED",
                                     "SKIPPED_VALID"])
def test_a_full_run_that_produced_nothing_earns_no_eligibility(outcome: str) -> None:
    """L.2 validates eligibility from the ancestor chain; a blocked run has none."""
    assert evidence_eligibility(profile_eligible=True, outcome=outcome) is False


def test_a_passing_full_run_is_eligible() -> None:
    assert evidence_eligibility(profile_eligible=True, outcome="PASS") is True


def test_a_passing_non_full_run_is_still_not_eligible() -> None:
    assert evidence_eligibility(profile_eligible=False, outcome="PASS") is False
