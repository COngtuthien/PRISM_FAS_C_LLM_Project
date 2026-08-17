"""What a stage is allowed to do, decided once instead of ten times.

Before this module every C4-C13 adapter had a `run_smoke` and inherited a
`run_full` that refused. The two paths were therefore not the same code, and the
rehearsal proved nothing about the scientific run — which is precisely the gap a
real laptop run exposed.

The fix is one workflow per stage, parameterized by an `ExecutionContext`. The
context answers the questions a stage would otherwise answer by reading the
profile and deciding for itself:

* may I substitute a fixture for an absent real input?
* how much may I truncate?
* where do my artifacts go?
* may I write a governing scientific lock?
* what may I do with the target?

A rehearsal and a scientific run then differ in the *values* of those answers,
not in the code that acts on them. Adding a stage step to one automatically adds
it to the other, which is the property that stops the two from drifting apart.

Two invariants are enforced here rather than trusted:

**Only the full profile is scientific.** `for_profile` reads
`profile.scientific_eligible` and nothing else can raise it, so a rehearsal
cannot acquire eligibility by constructing a context by hand.

**A scientific context truncates nothing.** `budget` is `None` under science, and
`limit()` returns the requested cardinality unchanged. A stage that asks "how many
rows may I run?" gets the declared number, so the smoke sampling constant is not
reachable from the scientific path even by mistake.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: What a stage may do with the held-out target.
TARGET_NONE = "none"                    # C4-C9: no target capability whatsoever
TARGET_FIXTURE_ONLY = "fixture_only"    # rehearsal C10-C13: a synthetic stand-in
TARGET_SEALED_REAL = "sealed_real"      # scientific C10-C13, under §19.2 sealing

#: Context names. Deliberately not the profile names: several profiles map onto
#: the rehearsal context, and the distinction that matters to a stage is what it
#: may do, not which config file selected it.
SCIENTIFIC = "scientific"
REHEARSAL = "rehearsal"


class ExecutionContextError(RuntimeError):
    """A context was asked for something its profile does not permit."""


@dataclass(frozen=True)
class ExecutionContext:
    """The policy a stage executes under. Constructed from the profile, not chosen."""

    name: str
    scientific_eligible: bool
    #: May this run stand a fixture in for an absent real input?
    fixtures_permitted: bool
    #: The engineering budget, or None when nothing may be truncated.
    budget: Any
    reports_namespace: str
    runs_namespace: str | None
    #: May this run write a lock that governs later scientific work?
    writes_governing_locks: bool
    target_capability: str
    profile_name: str

    # --- construction ---------------------------------------------------------

    @classmethod
    def for_profile(cls, profile: Any) -> "ExecutionContext":
        """The one place a profile becomes a policy.

        `scientific_eligible` is read from the profile rather than inferred from
        its name, because the profile is what the orchestrator already refuses to
        promote (`assert_not_promoted`), and having two answers to "is this
        scientific?" is how they come to disagree.
        """
        from prism_fas.pipeline.adapters.common import SmokeBudget

        eligible = bool(profile.scientific_eligible)
        return cls(
            name=SCIENTIFIC if eligible else REHEARSAL,
            scientific_eligible=eligible,
            fixtures_permitted=not eligible,
            budget=None if eligible else SmokeBudget.from_profile(profile),
            reports_namespace=profile.reports_namespace,
            runs_namespace=getattr(profile, "runs_namespace", None),
            writes_governing_locks=eligible,
            target_capability=TARGET_SEALED_REAL if eligible else TARGET_FIXTURE_ONLY,
            profile_name=profile.name)

    # --- what a stage asks it -------------------------------------------------

    @property
    def is_scientific(self) -> bool:
        return self.scientific_eligible

    @property
    def is_rehearsal(self) -> bool:
        return not self.scientific_eligible

    @property
    def cardinality_rule(self) -> str:
        return "FULL_DECLARED" if self.scientific_eligible else "SAMPLED_TO_BUDGET"

    def limit(self, declared: int, *, sample: int) -> int:
        """How many units to run: all of them under science, a sample otherwise.

        The scientific branch ignores `sample` entirely. That is the point — a
        stage cannot accidentally carry its rehearsal sampling constant into the
        scientific path, because the scientific path never reads it.
        """
        if self.scientific_eligible:
            return declared
        return max(1, min(declared, sample))

    def budget_or(self, attribute: str, declared: int) -> int:
        """A budgeted dimension: the declared value under science, the budget's otherwise."""
        if self.scientific_eligible or self.budget is None:
            return declared
        return max(1, min(declared, int(getattr(self.budget, attribute, declared))))

    def require_real(self, what: str) -> None:
        """Refuse to fabricate. Called on a path a fixture must never reach."""
        if self.fixtures_permitted:
            return
        raise ExecutionContextError(
            f"{what} must come from the real input under the {self.name} context; "
            "there is no fixture substitute on a scientific path")

    def lock_filename(self, scientific_name: str) -> str:
        """Governing locks keep their name; a rehearsal writes a marked stand-in.

        A rehearsal that wrote `GPAT_CONFIG_LOCK.json` would leave a file whose
        name is the one later stages look for. Renaming it is a second barrier
        behind the namespace: even if a rehearsal artifact were somehow copied
        into a scientific tree, it would not answer to the name.
        """
        if self.writes_governing_locks:
            return scientific_name
        stem, _, extension = scientific_name.rpartition(".")
        return f"{stem or scientific_name}_REHEARSAL.{extension or 'json'}"

    def stamp(self) -> dict[str, Any]:
        """Serialized onto every artifact a stage writes under this context."""
        return {
            "execution_context": self.name,
            "context_scientific_eligible": self.scientific_eligible,
            "context_fixtures_permitted": self.fixtures_permitted,
            "context_cardinality_rule": self.cardinality_rule,
            "context_target_capability": self.target_capability,
            "context_writes_governing_locks": self.writes_governing_locks,
        }


__all__ = ["ExecutionContext", "ExecutionContextError", "SCIENTIFIC", "REHEARSAL",
           "TARGET_NONE", "TARGET_FIXTURE_ONLY", "TARGET_SEALED_REAL"]
