"""Operational compute budget (v1.5 Appendix L.12).

L.12 opens with the sentence that governs this whole module: operational compute
budget is NOT a treatment factor. A budget may bound how much engineering work
runs; it may never become part of the scientific method, and it may never shrink
a scientific budget because credit is running low.

That produces an asymmetry which is deliberate and is enforced here:

* under validate/smoke a budget cap is legitimate, and exhausting it means
  checkpoint and stop cleanly;
* under full there is no cap to exhaust. If the required scientific work cannot
  finish, the status becomes BLOCKED or INTERRUPTED and the work resumes later
  on sufficient compute. Silently running a smaller matrix is not an option the
  code offers.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from prism_fas.pipeline.profiles import ExecutionProfile

EXHAUSTED_ACTION = "CHECKPOINT_AND_STOP_CLEANLY"


class BudgetError(RuntimeError):
    """A budget was applied where the spec forbids one."""


@dataclass
class BudgetLedger:
    """Tracks engineering spend against a profile's declared cap.

    Deliberately dumb: it counts and compares. Every scientific quantity — seed
    family, matrix size, search budget — comes from a frozen contract, and none
    of them is readable from here.
    """

    profile_name: str
    limits: dict[str, Any] = field(default_factory=dict)
    spent: dict[str, float] = field(default_factory=dict)
    started_monotonic: float = field(default_factory=time.monotonic)
    stops: list[str] = field(default_factory=list)

    @property
    def bounded(self) -> bool:
        return bool(self.limits)

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_monotonic

    def spend(self, dimension: str, amount: float = 1.0) -> None:
        self.spent[dimension] = self.spent.get(dimension, 0.0) + amount

    def exceeded(self) -> list[str]:
        """Which caps are now exceeded. Empty when the ledger is unbounded."""
        breached: list[str] = []
        for name, cap in self.limits.items():
            if cap is None or not isinstance(cap, (int, float)):
                continue
            if name == "max_wall_clock_seconds":
                if self.elapsed_seconds() > cap:
                    breached.append(name)
            elif self.spent.get(name, 0.0) > cap:
                breached.append(name)
        return breached

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile_name,
            "bounded": self.bounded,
            "limits": dict(self.limits),
            "spent": dict(self.spent),
            "elapsed_seconds": round(self.elapsed_seconds(), 3),
            "exceeded": self.exceeded(),
            "on_exhaustion": EXHAUSTED_ACTION if self.bounded else
                             "not applicable; the full profile has no operational cap",
            "treatment_factor": False,
        }


def ledger_for(profile: ExecutionProfile) -> BudgetLedger:
    """Build the ledger a profile is entitled to.

    A full profile carrying an engineering budget is refused rather than
    honoured. `load_profile` already rejects such a config, so reaching this
    error means a ledger was constructed from a profile assembled some other
    way — still worth failing on, because the consequence is a quietly shrunken
    scientific matrix.
    """
    if profile.scientific_eligible and profile.engineering_budget:
        raise BudgetError(
            f"profile {profile.name!r} is scientifically eligible and must not carry an "
            "operational budget cap; L.12 forbids shrinking a scientific budget because "
            "remaining credit is low")
    return BudgetLedger(profile_name=profile.name,
                        limits=dict(profile.engineering_budget or {}))


def assert_reduction_allowed(profile: ExecutionProfile, dimension: str) -> None:
    """Refuse a reduction the profile does not authorize.

    L.12 lists exactly what smoke may reduce — samples, steps, epochs and seed
    count — and exactly what it must preserve regardless: code paths, tensor
    semantics, model topology, active loss semantics, the target firewall and
    artifact schemas. Reducing anything else is a scientific change wearing an
    engineering label.
    """
    if not profile.reduction_permitted:
        raise BudgetError(
            f"profile {profile.name!r} does not permit budget reduction of {dimension!r}")
    reducible = tuple(profile.raw.get("reducible_dimensions") or ())
    if dimension not in reducible:
        raise BudgetError(
            f"{dimension!r} is not a reducible dimension under profile {profile.name!r}; "
            f"L.12 permits only {reducible}")


__all__ = ["EXHAUSTED_ACTION", "BudgetError", "BudgetLedger", "ledger_for",
           "assert_reduction_allowed"]
