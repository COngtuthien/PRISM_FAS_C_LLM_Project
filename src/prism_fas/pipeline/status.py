"""The v1.5 dual status model (Appendix L.3) and run outcome vocabulary (L.8).

Version C separates ENGINEERING READINESS from SCIENTIFIC COMPLETION, and the
whole point of the separation is that it cannot be blurred by accident. So the
two axes are closed vocabularies here rather than free strings, and the one
question that matters — "is this milestone scientifically complete?" — is a
function of both the status and the profile that produced it.

L.3 is explicit that a milestone is scientifically complete only when
``scientific_status=PASS`` under the full profile. SMOKE_PASS is an engineering
statement. It is not paper evidence by itself, and `scientifically_complete`
below is the only place allowed to answer the question.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: L.3, engineering axis: whether the implementation path is technically ready
#: under validate/smoke execution.
ENGINEERING_STATUS: tuple[str, ...] = (
    "NOT_TESTED", "RUNNING", "SMOKE_PASS", "SMOKE_FAIL", "BLOCKED")

#: L.3, scientific axis: whether the complete preregistered scientific workload
#: and its hard acceptance criteria have been executed.
SCIENTIFIC_STATUS: tuple[str, ...] = (
    "NOT_RUN", "RUNNING", "PASS", "FAIL", "BLOCKED")

#: L.8 outcome of one atomic run. Every one of these is preserved; losing and
#: failing rows are evidence and are never deleted after a winner is chosen.
RUN_OUTCOME: tuple[str, ...] = (
    "PASS", "FAIL", "DIVERGED", "INTERRUPTED", "BLOCKED", "SKIPPED_VALID")

#: L.5: the orchestrator stops with this rather than asking an unattended agent
#: to invent a method at a gate outside the preregistered envelope.
NEEDS_SCIENTIFIC_DECISION = "NEEDS_SCIENTIFIC_DECISION"


class StatusError(ValueError):
    """A status value outside the closed L.3 / L.8 vocabulary."""


@dataclass(frozen=True)
class DualStatus:
    """One milestone's two statuses, validated on construction.

    Keeping them in a single frozen object means no artifact can serialize one
    axis and silently omit the other, which is exactly how a SMOKE_PASS starts
    reading like a milestone PASS.
    """

    engineering: str
    scientific: str

    def __post_init__(self) -> None:
        if self.engineering not in ENGINEERING_STATUS:
            raise StatusError(
                f"engineering_status {self.engineering!r} is not one of {ENGINEERING_STATUS}")
        if self.scientific not in SCIENTIFIC_STATUS:
            raise StatusError(
                f"scientific_status {self.scientific!r} is not one of {SCIENTIFIC_STATUS}")

    def as_dict(self) -> dict[str, str]:
        return {"engineering_status": self.engineering,
                "scientific_status": self.scientific}


def scientifically_complete(status: DualStatus, *, profile: str) -> bool:
    """L.3: PASS on the scientific axis, under the full profile, and nothing else.

    The profile is a required argument because the status alone cannot answer
    the question — a PASS carried over from a non-full profile would not be
    scientific completion, and there is no reading of L.3 under which it is.
    """
    return profile == "full" and status.scientific == "PASS"


def evidence_eligibility(*, profile_eligible: bool, outcome: str) -> bool:
    """Whether one produced artifact is scientifically eligible.

    The profile is necessary but not sufficient. L.2 requires a full artifact's
    eligibility to be "validated from its complete ancestor identity chain", and
    a run that was BLOCKED, FAILED, DIVERGED or INTERRUPTED has no such chain —
    it produced no validated evidence at all. Deriving eligibility from the
    profile alone would stamp `scientific_eligible=true` on a row recording that
    nothing ran, which is exactly the claim the flag exists to prevent.

    A PASS under the full profile returns True here, and the stage adapter that
    produced it remains responsible for validating its ancestors; this function
    removes the cases where no ancestry could possibly exist.
    """
    return profile_eligible and outcome == "PASS"


def assert_not_promoted(artifact: dict[str, Any]) -> None:
    """Refuse an artifact that claims eligibility its profile cannot grant.

    L.2 requires every smoke artifact to serialize ``execution_profile="smoke"``
    and ``scientific_eligible=false``. This is the structural check: a
    non-full artifact carrying ``scientific_eligible=true`` is a promotion, and
    a promotion is refused at write time rather than caught in review.
    """
    profile = artifact.get("execution_profile")
    eligible = artifact.get("scientific_eligible")
    if profile is None or eligible is None:
        raise StatusError(
            "every artifact must serialize execution_profile and scientific_eligible "
            f"(got profile={profile!r}, eligible={eligible!r})")
    if eligible is True and profile != "full":
        raise StatusError(
            f"artifact claims scientific_eligible=true under profile {profile!r}; "
            "only the full profile can produce scientifically eligible evidence (L.2)")


__all__ = ["ENGINEERING_STATUS", "SCIENTIFIC_STATUS", "RUN_OUTCOME",
           "NEEDS_SCIENTIFIC_DECISION", "StatusError", "DualStatus",
           "scientifically_complete", "evidence_eligibility", "assert_not_promoted"]
