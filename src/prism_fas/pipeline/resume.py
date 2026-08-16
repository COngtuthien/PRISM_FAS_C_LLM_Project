"""Identity-aware resume and the invalidation subtree (v1.5 Appendix L.11).

L.11 draws a sharp line: a completed artifact may be skipped only after its
expected parent identities, config identity, content hash and acceptance state
all validate. "The file exists" is not one of those four, and this module never
treats it as one.

The rule has teeth in both directions.

* A valid frozen C3 raw archive or GPAT checkpoint must NOT be regenerated
  merely because the orchestrator restarted. Re-generating a C3 archive would
  mean live Gemini calls that the contract never authorized.
* If an expected identity changed, the pipeline fails closed and computes the
  affected invalidation subtree. Stale descendants are never silently reused —
  a bank selected under an old selection contract is not evidence about the new
  one, even when the raw candidates are byte-identical (§21.3).

`decide` returns a decision object rather than a bool so the reason is always
recordable, and so a SKIP can be audited after the fact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from prism_fas.pipeline.stages import STAGES, Stage

SKIP = "SKIP_VALID_COMPLETE"
EXECUTE = "EXECUTE"
INVALIDATE = "INVALIDATE_AND_FAIL_CLOSED"


class ResumeError(RuntimeError):
    """A resume decision could not be made safely."""


@dataclass(frozen=True)
class ResumeDecision:
    """What to do with one unit of work, and exactly why."""

    action: str
    reason: str
    unit_id: str
    checks: dict[str, bool] = field(default_factory=dict)
    invalidation_subtree: tuple[str, ...] = ()

    @property
    def may_skip(self) -> bool:
        return self.action == SKIP

    def as_dict(self) -> dict[str, Any]:
        return {"action": self.action, "reason": self.reason, "unit_id": self.unit_id,
                "checks": dict(self.checks),
                "invalidation_subtree": list(self.invalidation_subtree)}


@dataclass(frozen=True)
class CompletedUnit:
    """The recorded completion of one unit, as the index stored it."""

    unit_id: str
    stage_id: str
    parent_identities: Mapping[str, str]
    config_identity: str
    content_hash: str
    acceptance_state: str
    artifact_path: str | None = None


def invalidation_subtree(stage_id: str) -> tuple[str, ...]:
    """Every stage downstream of a stage whose identity changed.

    L.11 requires computing this rather than assuming the damage is local. C0-C13
    are a linear dependency chain in v1.5, so the subtree is the suffix — but it
    is computed from the declared stage order, not hard-coded, so it stays right
    if the order ever changes.
    """
    ordered = [stage.stage_id for stage in STAGES]
    if stage_id not in ordered:
        raise ResumeError(f"cannot compute an invalidation subtree for unknown stage {stage_id!r}")
    return tuple(ordered[ordered.index(stage_id):])


def decide(unit: CompletedUnit, *, expected_parents: Mapping[str, str],
           expected_config_identity: str, expected_content_hash: str | None,
           accepted_states: tuple[str, ...] = ("PASS", "ACCEPTED"),
           repo: Path | None = None) -> ResumeDecision:
    """Apply the four L.11 conditions. All must hold before a unit is skipped."""
    checks: dict[str, bool] = {}

    parents_match = all(unit.parent_identities.get(name) == value
                        for name, value in expected_parents.items())
    checks["parent_identities_validate"] = parents_match

    config_matches = unit.config_identity == expected_config_identity
    checks["config_identity_validates"] = config_matches

    # A missing expected hash means "not yet known", which is not the same as
    # "matches". It is treated as a failure to validate, so the unit re-executes
    # rather than being skipped on an unverified claim.
    content_matches = (expected_content_hash is not None
                       and unit.content_hash == expected_content_hash)
    checks["content_hash_validates"] = content_matches

    accepted = unit.acceptance_state in accepted_states
    checks["acceptance_state_validates"] = accepted

    artifact_present = True
    if repo is not None and unit.artifact_path:
        artifact_present = (repo / unit.artifact_path).exists()
    checks["artifact_present"] = artifact_present

    if all(checks.values()):
        return ResumeDecision(
            SKIP,
            "parent identities, config identity, content hash and acceptance state all "
            "validate; the completed artifact is reused unchanged (L.11)",
            unit.unit_id, checks)

    # A drifted identity is not a reason to redo the work quietly — it means
    # every descendant computed under the old identity is now suspect.
    if not parents_match or not config_matches:
        return ResumeDecision(
            INVALIDATE,
            "an expected identity changed; failing closed and invalidating the affected "
            "subtree rather than reusing stale descendants (L.11)",
            unit.unit_id, checks, invalidation_subtree(unit.stage_id))

    return ResumeDecision(
        EXECUTE,
        "the recorded completion does not validate; the missing or unfinished work is "
        "re-executed (L.11)",
        unit.unit_id, checks)


def protected_from_regeneration(stage: Stage) -> bool:
    """Stages whose completed artifacts a restart must never silently redo.

    L.11 names both cases explicitly: a valid C3 raw archive or recipe bank must
    not trigger live Gemini re-generation, and a valid final GPAT checkpoint must
    not be retrained while its ancestor and search identities still match. The
    cost of getting this wrong is live provider quota and GPU hours spent
    re-deriving evidence that already exists.
    """
    return stage.stage_id in {"C3", "C4"}


__all__ = ["SKIP", "EXECUTE", "INVALIDATE", "ResumeError", "ResumeDecision",
           "CompletedUnit", "invalidation_subtree", "decide",
           "protected_from_regeneration"]
