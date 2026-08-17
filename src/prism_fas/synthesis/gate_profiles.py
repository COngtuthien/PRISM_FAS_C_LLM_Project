"""The common quality-gate profile and matched-bank construction (§11.3, §11.4).

§11.4 makes the gate adaptive and *common*: thresholds are SOURCE_SEARCH, but the
same profile must apply to RND, DET and LLM. The failure it exists to prevent is
the Version-B confound — an arm that looks better because its gate was looser.
So the arm identity is not an input to anything in this module, and the selection
rule cannot express a per-arm threshold at all.

Three preregistered profiles, derived from the inherited NOMINAL by a formula
rather than by judgement:

* higher-is-better lower bound ``a``: ``STRICT = a + 0.10 * (1 - a)``,
  ``PERMISSIVE = max(0, 0.90 * a)``
* nonnegative lower-is-better upper bound ``a``: ``STRICT = 0.90 * a``,
  ``PERMISSIVE = 1.10 * a``

One threshold is deliberately exempt. ``tau_out`` is an exact-equality
range-safety constraint — the residual must not leak outside the support mask at
all — and §11.4 says range/recipe-safe constraints are never relaxed beyond their
frozen legal range. Profiling it would turn a correctness invariant into a knob.

Selection picks the **strictest** profile that yields the full matched
cardinality in *every* arm and passes the mandatory reliability gates. If none
qualifies, C6 FAILS; it never relaxes the gate for the arm that fell short.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "prism-c6-gate-profile-v1"

STRICT, NOMINAL, PERMISSIVE = "STRICT", "NOMINAL", "PERMISSIVE"

#: Strictest first. Selection walks this order and takes the first qualifier,
#: which is what "select the strictest profile that qualifies" means operationally.
PROFILE_ORDER: tuple[str, ...] = (STRICT, NOMINAL, PERMISSIVE)

#: Direction of every gate threshold, read off the frozen hard-gate table in
#: `configs/synthesis/quality_gate_m8.yaml` and §11.1.
HIGHER_IS_BETTER: tuple[str, ...] = ("tau_fd", "tau_id", "tau_parse")
LOWER_IS_BETTER: tuple[str, ...] = ("tau_lm", "tau_fp")
#: Exact-equality range-safety constraint. Never profiled (§11.4).
RANGE_SAFE: tuple[str, ...] = ("tau_out",)

#: §11.3 / §10.4 matched-bank contract. Scientific constants, transcribed.
CANDIDATES_PER_ARM = 2048
FINAL_BANK_PER_ARM = 1024
PHYSICS_PER_ARM = 512
GPAT_PER_ARM = 512
RENDERS_PER_RECIPE = 8
PHYSICS_RENDERS_PER_RECIPE = 4
GPAT_RENDERS_PER_RECIPE = 4

ARMS: tuple[str, ...] = ("RND", "DET", "LLM")


class GateProfileError(ValueError):
    """A profile cannot be derived, or a selection rule was misapplied."""


def _sha(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def derive_profile(nominal: Mapping[str, float], profile: str) -> dict[str, float]:
    """One profile's thresholds, derived from NOMINAL by the §11.4 formula.

    Deriving rather than declaring is the point: a hand-written STRICT table
    could drift from the formula, and the formula is the part the spec froze.
    """
    if profile not in PROFILE_ORDER:
        raise GateProfileError(f"unknown gate profile {profile!r}; expected {PROFILE_ORDER}")
    values: dict[str, float] = {}
    for name, anchor in nominal.items():
        a = float(anchor)
        if name in RANGE_SAFE:
            values[name] = a          # never relaxed, never tightened
        elif profile == NOMINAL:
            values[name] = a
        elif name in HIGHER_IS_BETTER:
            values[name] = (a + 0.10 * (1.0 - a)) if profile == STRICT else max(0.0, 0.90 * a)
        elif name in LOWER_IS_BETTER:
            values[name] = (0.90 * a) if profile == STRICT else (1.10 * a)
        else:
            raise GateProfileError(
                f"threshold {name!r} has no declared direction; §11.4 cannot derive a "
                "profile for a metric whose better-is-higher/lower semantics are unknown")
    return {name: round(value, 12) + 0.0 for name, value in values.items()}


@dataclass(frozen=True)
class GateProfile:
    """One derived profile and the thresholds object the gate consumes."""

    name: str
    thresholds: dict[str, float]
    nominal_source: str = ""

    def as_thresholds(self) -> Any:
        from prism_fas.synthesis.quality_gate import Thresholds

        return Thresholds.from_dict(self.thresholds)

    @property
    def identity(self) -> str:
        return _sha({"profile": self.name, "thresholds": self.thresholds})

    def as_dict(self) -> dict[str, Any]:
        return {"profile": self.name, "thresholds": dict(self.thresholds),
                "threshold_identity": self.identity,
                "nominal_source": self.nominal_source,
                "range_safe_unprofiled": list(RANGE_SAFE)}


def build_profiles(nominal: Mapping[str, float], *,
                   nominal_source: str = "") -> dict[str, GateProfile]:
    """All three preregistered profiles from one inherited NOMINAL set."""
    return {name: GateProfile(name=name, thresholds=derive_profile(nominal, name),
                              nominal_source=nominal_source)
            for name in PROFILE_ORDER}


@dataclass(frozen=True)
class ArmFeasibility:
    """Whether one arm can fill its matched bank under one profile."""

    arm: str
    candidates: int
    accepted_physics: int
    accepted_gpat: int
    required_physics: int = PHYSICS_PER_ARM
    required_gpat: int = GPAT_PER_ARM

    @property
    def accepted_total(self) -> int:
        return self.accepted_physics + self.accepted_gpat

    @property
    def feasible(self) -> bool:
        """Both route quotas must be fillable, not just the total.

        A bank with 900 physics and 124 GPAT accepted samples reaches 1024 and is
        still infeasible: §11.3 fixes the split at 512 + 512, and letting one
        route cover for the other would change what the bank is.
        """
        return (self.accepted_physics >= self.required_physics
                and self.accepted_gpat >= self.required_gpat)

    def as_dict(self) -> dict[str, Any]:
        return {"arm": self.arm, "candidates": self.candidates,
                "accepted_physics": self.accepted_physics,
                "accepted_gpat": self.accepted_gpat,
                "accepted_total": self.accepted_total,
                "required_physics": self.required_physics,
                "required_gpat": self.required_gpat,
                "required_total": self.required_physics + self.required_gpat,
                "feasible": self.feasible,
                "shortfall_physics": max(0, self.required_physics - self.accepted_physics),
                "shortfall_gpat": max(0, self.required_gpat - self.accepted_gpat)}


@dataclass
class ProfileSelection:
    """The §11.4 decision, with every profile's evidence retained."""

    selected: str | None
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    reliability_passed: dict[str, bool] = field(default_factory=dict)
    failure_reason: str = ""

    @property
    def failed(self) -> bool:
        return self.selected is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "selected_profile": self.selected,
            "selection_rule": (
                "the strictest profile that (i) yields at least "
                f"{FINAL_BANK_PER_ARM} accepted samples in EVERY arm from the frozen "
                f"{CANDIDATES_PER_ARM} candidates/arm with exactly {PHYSICS_PER_ARM} "
                f"Physics + {GPAT_PER_ARM} GPAT feasible, and (ii) passes all mandatory "
                "source-only shortcut/reliability gates (§11.4)"),
            "profile_order_strictest_first": list(PROFILE_ORDER),
            "evaluations": list(self.evaluations),
            "reliability_gates": dict(self.reliability_passed),
            "failed": self.failed,
            "failure_reason": self.failure_reason,
            "arm_independence": (
                "the arm identity is not an input to threshold derivation or selection; "
                "one common profile applies to RND, DET and LLM and no arm may be "
                "relaxed independently (§11.4)"),
            "on_failure": ("C6 FAILS rather than relaxing the gate for the arm that fell "
                           "short; rejected candidates remain rejected provenance and are "
                           "not replaceable by extra renders"),
        }


def select_profile(profiles: Mapping[str, GateProfile],
                   feasibility: Mapping[str, Sequence[ArmFeasibility]], *,
                   reliability: Mapping[str, bool] | None = None) -> ProfileSelection:
    """Choose the strictest qualifying profile, or fail.

    `feasibility` maps profile name to one `ArmFeasibility` per arm. Every arm
    must be feasible under a profile for it to qualify — the conjunction is the
    whole mechanism that stops a strong arm from carrying a weak one.
    """
    gates = dict(reliability or {})
    reliability_ok = all(gates.values()) if gates else True
    evaluations: list[dict[str, Any]] = []
    selected: str | None = None

    for name in PROFILE_ORDER:
        if name not in profiles:
            continue
        arms = list(feasibility.get(name, ()))
        missing = sorted(set(ARMS) - {item.arm for item in arms})
        all_feasible = bool(arms) and not missing and all(item.feasible for item in arms)
        qualifies = all_feasible and reliability_ok
        evaluations.append({
            "profile": name,
            "thresholds": profiles[name].thresholds,
            "threshold_identity": profiles[name].identity,
            "arms": [item.as_dict() for item in arms],
            "arms_missing_evidence": missing,
            "every_arm_feasible": all_feasible,
            "reliability_gates_passed": reliability_ok,
            "qualifies": qualifies,
            "selected": qualifies and selected is None,
        })
        if qualifies and selected is None:
            selected = name

    reason = ""
    if selected is None:
        blocked = [row["profile"] for row in evaluations if not row["every_arm_feasible"]]
        reason = (
            "no preregistered profile yields the full matched cardinality in every arm"
            + (f" (infeasible under {blocked})" if blocked else "")
            + ("; mandatory reliability gates also failed" if not reliability_ok else "")
            + ". §11.4: C6 FAILS. The gate is not relaxed for one arm.")
    return ProfileSelection(selected=selected, evaluations=evaluations,
                            reliability_passed=gates, failure_reason=reason)


def matched_bank_plan(accepted: Mapping[str, ArmFeasibility]) -> dict[str, Any]:
    """The matched banks the selected profile makes possible, or the shortfall.

    Deterministic by construction: every arm takes exactly the frozen split, so
    the plan cannot express a bank that is matched in total but not by route.
    """
    rows = {}
    for arm, item in accepted.items():
        rows[arm] = {
            "arm": arm,
            "final_bank_size": FINAL_BANK_PER_ARM if item.feasible else None,
            "physics": PHYSICS_PER_ARM if item.feasible else item.accepted_physics,
            "gpat": GPAT_PER_ARM if item.feasible else item.accepted_gpat,
            "feasible": item.feasible,
            **item.as_dict(),
        }
    matched = all(row["feasible"] for row in rows.values()) and len(rows) == len(ARMS)
    return {
        "schema_version": SCHEMA_VERSION,
        "arms": rows,
        "matched": matched,
        "cardinality_contract": {
            "candidates_per_arm": CANDIDATES_PER_ARM,
            "renders_per_recipe": RENDERS_PER_RECIPE,
            "physics_renders_per_recipe": PHYSICS_RENDERS_PER_RECIPE,
            "gpat_renders_per_recipe": GPAT_RENDERS_PER_RECIPE,
            "final_bank_per_arm": FINAL_BANK_PER_ARM,
            "physics_per_arm": PHYSICS_PER_ARM,
            "gpat_per_arm": GPAT_PER_ARM},
        "selection_basis": ("balanced deterministically over source domain, route, recipe "
                            "coverage and base live IDs (§11.3)"),
        "on_shortfall": ("if an arm cannot satisfy its quota under the frozen render budget "
                         "and the common gate, C6 FAILS (§11.3)"),
    }


__all__ = ["SCHEMA_VERSION", "STRICT", "NOMINAL", "PERMISSIVE", "PROFILE_ORDER",
           "HIGHER_IS_BETTER", "LOWER_IS_BETTER", "RANGE_SAFE", "CANDIDATES_PER_ARM",
           "FINAL_BANK_PER_ARM", "PHYSICS_PER_ARM", "GPAT_PER_ARM", "RENDERS_PER_RECIPE",
           "PHYSICS_RENDERS_PER_RECIPE", "GPAT_RENDERS_PER_RECIPE", "ARMS",
           "GateProfileError", "derive_profile", "GateProfile", "build_profiles",
           "ArmFeasibility", "ProfileSelection", "select_profile", "matched_bank_plan"]
