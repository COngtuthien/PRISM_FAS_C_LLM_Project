"""prism_c3_selection_v1 — the deterministic 384 -> 256 recipe-bank selector.

v1.5 §7.8.3. Selection is an exact deterministic constrained-subset problem over
unique eligible recipes. Human ranking, aesthetic preference, target performance,
SiW taxonomy or frequencies, Version-B target errors, downstream model score,
synthetic quality q and manual "nice recipe" choice are FORBIDDEN INPUTS. The
same policy semantics apply independently to RND, DET and LLM.

The optimization is lexicographic over exact integers:

    Stage 1  hard feasibility: exactly 256, every hard min/max in §7.8.2
    Stage 2  minimize S_pref   = Σ_a max(0, 32 − count_a) + Σ_r max(0, 24 − count_r)
    Stage 3  minimize S_single = Σ_m |5·count_m − 256| + Σ_g |6·count_g − 256|
                                 + Σ_i |6·count_i − 256|
    Stage 4  minimize S_multi  = Σ_a |8·count_a − A_total| + Σ_r |9·count_r − R_total|
    Stage 5  canonical tie-break: among all subsets tied on 1–4, the
             lexicographically smallest 256-element selected SHA-256 set

Two engineering properties make Stage 5 real rather than aspirational:

* The optimum objective values are found first. Stage 5 then performs iterative
  feasibility fixing over candidates in ascending canonical SHA-256, which yields
  the lexicographically smallest selected set by construction and does not depend
  on how the solver happened to traverse its search tree.
* Every count and every objective value is RECOMPUTED from the selected set in
  pure Python integer arithmetic and must equal what the solver claimed. A
  mismatch raises rather than being accepted, so a floating-point artefact inside
  the engine can never become the scientific answer.

A MILP engine (scipy/HiGHS) is used as the engineering solver. The mathematical
policy above is the scientific contract; the engine is replaceable.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence

from prism_fas.recipes.canonical import recipe_hash
from prism_fas.recipes.ontology import Ontology
from prism_fas.recipes.schema import RecipeV11

SELECTION_VERSION = "prism_c3_selection_v1"

RAW_CANDIDATE_SLOTS_PER_ARM = 384
MINIMUM_ELIGIBLE_POOL_PER_ARM = 320
FINAL_BANK_SIZE_PER_ARM = 256

#: §7.8.2, recipe-occurrence counts over the selected bank.
QUOTAS: dict[str, dict[str, int | None]] = {
    "medium":       {"hard_min": 32, "hard_max": 80,  "preferred_min": None},
    "geometry":     {"hard_min": 24, "hard_max": 64,  "preferred_min": None},
    "illumination": {"hard_min": 24, "hard_max": 64,  "preferred_min": None},
    "artifact":     {"hard_min": 8,  "hard_max": 128, "preferred_min": 32},
    "region":       {"hard_min": 8,  "hard_max": 128, "preferred_min": 24},
}

SINGLE_AXES: tuple[str, ...] = ("medium", "geometry", "illumination")
MULTI_AXES: tuple[str, ...] = ("artifact", "region")
AXES: tuple[str, ...] = SINGLE_AXES + MULTI_AXES


class SelectionError(RuntimeError):
    """Selection failed. Never partially applied, never silently relaxed."""


class SelectionInfeasible(SelectionError):
    """No 256-subset satisfies the hard constraints. C3 FAILS for that arm."""


# ------------------------------------------------------------------ categories
def axis_categories(ontology: Ontology) -> dict[str, tuple[str, ...]]:
    """Frozen category ordering, taken from the ontology."""
    return {
        "medium": tuple(ontology.media),
        "geometry": tuple(ontology.geometry_shapes),
        "illumination": tuple(ontology.illumination),
        "artifact": tuple(ontology.artifacts),
        "region": tuple(ontology.regions),
    }


def recipe_axis_values(recipe: RecipeV11) -> dict[str, list[str]]:
    """The categories one recipe occupies. Recipe content only."""
    return {
        "medium": [recipe.medium.family],
        "geometry": [recipe.geometry.shape],
        "illumination": [recipe.capture.illumination],
        "artifact": sorted({spec.name for spec in recipe.artifacts}),
        "region": sorted(set(recipe.regions)),
    }


def counts_for(recipes: Sequence[RecipeV11], ontology: Ontology) -> dict[str, dict[str, int]]:
    categories = axis_categories(ontology)
    counts = {axis: {name: 0 for name in categories[axis]} for axis in AXES}
    for recipe in recipes:
        values = recipe_axis_values(recipe)
        for axis in AXES:
            for value in values[axis]:
                counts[axis][value] += 1
    return counts


# ------------------------------------------------------- exact objective maths
def s_pref(counts: dict[str, dict[str, int]]) -> int:
    """Σ_artifact max(0, 32 − count_a) + Σ_region max(0, 24 − count_r)."""
    total = 0
    for axis in MULTI_AXES:
        preferred = QUOTAS[axis]["preferred_min"]
        if preferred is None:
            continue
        for value in counts[axis].values():
            total += max(0, int(preferred) - int(value))
    return total


def s_single(counts: dict[str, dict[str, int]], bank_size: int = FINAL_BANK_SIZE_PER_ARM) -> int:
    """Σ_m |5·count_m − 256| + Σ_g |6·count_g − 256| + Σ_i |6·count_i − 256|."""
    total = 0
    for axis in SINGLE_AXES:
        k = len(counts[axis])
        for value in counts[axis].values():
            total += abs(k * int(value) - bank_size)
    return total


def s_multi(counts: dict[str, dict[str, int]]) -> int:
    """Σ_a |8·count_a − A_total| + Σ_r |9·count_r − R_total|."""
    total = 0
    for axis in MULTI_AXES:
        k = len(counts[axis])
        axis_total = sum(int(value) for value in counts[axis].values())
        for value in counts[axis].values():
            total += abs(k * int(value) - axis_total)
    return total


def hard_violations(counts: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    for axis in AXES:
        quota = QUOTAS[axis]
        for name, value in counts[axis].items():
            if value < int(quota["hard_min"]):
                problems.append({"axis": axis, "category": name, "count": value,
                                 "bound": "hard_min", "required": quota["hard_min"]})
            if value > int(quota["hard_max"]):
                problems.append({"axis": axis, "category": name, "count": value,
                                 "bound": "hard_max", "required": quota["hard_max"]})
    return problems


# ------------------------------------------------------------------- the model
@dataclass(frozen=True)
class _Model:
    """Column-oriented incidence data. Index order is ascending canonical SHA."""

    shas: list[str]
    incidence: dict[str, dict[str, list[int]]]   # axis -> category -> candidate indices
    multi_load: dict[str, list[int]]             # axis -> per-candidate occurrence count
    categories: dict[str, tuple[str, ...]]

    @property
    def n(self) -> int:
        return len(self.shas)


def _build_model(recipes: Sequence[RecipeV11], ontology: Ontology) -> tuple[_Model,
                                                                            list[RecipeV11]]:
    """Sort candidates by canonical SHA ascending — the only ordering the policy
    permits — and build incidence lists."""
    paired = sorted(((recipe_hash(recipe), recipe) for recipe in recipes),
                    key=lambda item: item[0])
    shas = [sha for sha, _ in paired]
    if len(set(shas)) != len(shas):
        raise SelectionError("the eligible pool contains duplicate canonical SHA-256 values; "
                             "deduplication must run before selection")
    ordered = [recipe for _, recipe in paired]
    categories = axis_categories(ontology)
    incidence = {axis: {name: [] for name in categories[axis]} for axis in AXES}
    multi_load = {axis: [0] * len(ordered) for axis in MULTI_AXES}
    for index, recipe in enumerate(ordered):
        values = recipe_axis_values(recipe)
        for axis in AXES:
            for value in values[axis]:
                incidence[axis][value].append(index)
            if axis in MULTI_AXES:
                multi_load[axis][index] = len(values[axis])
    return _Model(shas, incidence, multi_load, categories), ordered


# ------------------------------------------------------------------ MILP layer
def _solve(model: _Model, *, bank_size: int, objective: str | None,
           fixed: dict[str, int] | None, forced_in: set[int], forced_out: set[int],
           feasibility_only: bool) -> tuple[bool, int | None, list[int] | None]:
    """One MILP solve. `objective` in {None, 'pref', 'single', 'multi'}.

    `fixed` pins already-optimized stage values so a later stage optimizes only
    among earlier optima. Returns (feasible, objective_value, selected_indices).
    """
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SelectionError(
            "prism_c3_selection_v1 needs an exact MILP engine; scipy.optimize.milp "
            f"is unavailable: {exc}") from exc

    n = model.n
    # Auxiliary variables, in a frozen order:
    #   pref slack   : one per multi-axis category with a preferred minimum
    #   single abs   : one per single-axis category
    #   multi abs    : one per multi-axis category
    pref_cols = [(axis, name) for axis in MULTI_AXES
                 if QUOTAS[axis]["preferred_min"] is not None
                 for name in model.categories[axis]]
    single_cols = [(axis, name) for axis in SINGLE_AXES for name in model.categories[axis]]
    multi_cols = [(axis, name) for axis in MULTI_AXES for name in model.categories[axis]]

    off_pref = n
    off_single = off_pref + len(pref_cols)
    off_multi = off_single + len(single_cols)
    total = off_multi + len(multi_cols)

    rows: list[list[float]] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(coeffs: dict[int, float], low: float, high: float) -> None:
        row = [0.0] * total
        for col, value in coeffs.items():
            row[col] = float(value)
        rows.append(row)
        lower.append(low)
        upper.append(high)

    # exactly `bank_size` selected
    add({i: 1.0 for i in range(n)}, bank_size, bank_size)

    # hard quotas
    for axis in AXES:
        quota = QUOTAS[axis]
        for name in model.categories[axis]:
            members = model.incidence[axis][name]
            add({i: 1.0 for i in members}, float(quota["hard_min"]), float(quota["hard_max"]))

    # pref slack: s >= preferred - count  =>  s + count >= preferred
    for position, (axis, name) in enumerate(pref_cols):
        preferred = float(QUOTAS[axis]["preferred_min"])
        coeffs = {i: 1.0 for i in model.incidence[axis][name]}
        coeffs[off_pref + position] = 1.0
        add(coeffs, preferred, np.inf)

    # single-axis absolute value: t >= k*count - B and t >= B - k*count
    for position, (axis, name) in enumerate(single_cols):
        k = float(len(model.categories[axis]))
        col = off_single + position
        members = model.incidence[axis][name]
        first = {i: -k for i in members}
        first[col] = 1.0
        add(first, -float(bank_size), np.inf)          # t - k*count >= -B
        second = {i: k for i in members}
        second[col] = 1.0
        add(second, float(bank_size), np.inf)          # t + k*count >= B

    # multi-axis absolute value: u >= k*count - total and u >= total - k*count
    for position, (axis, name) in enumerate(multi_cols):
        k = float(len(model.categories[axis]))
        col = off_multi + position
        members = set(model.incidence[axis][name])
        load = model.multi_load[axis]
        first: dict[int, float] = {}
        second: dict[int, float] = {}
        for i in range(n):
            a = (k if i in members else 0.0) - float(load[i])
            if a:
                first[i] = -a
                second[i] = a
        first[col] = 1.0
        second[col] = 1.0
        add(first, 0.0, np.inf)
        add(second, 0.0, np.inf)

    # pinned stage objectives
    if fixed:
        if "pref" in fixed:
            add({off_pref + p: 1.0 for p in range(len(pref_cols))},
                float(fixed["pref"]), float(fixed["pref"]))
        if "single" in fixed:
            add({off_single + p: 1.0 for p in range(len(single_cols))},
                float(fixed["single"]), float(fixed["single"]))
        if "multi" in fixed:
            add({off_multi + p: 1.0 for p in range(len(multi_cols))},
                float(fixed["multi"]), float(fixed["multi"]))

    cost = [0.0] * total
    if objective == "pref":
        for p in range(len(pref_cols)):
            cost[off_pref + p] = 1.0
    elif objective == "single":
        for p in range(len(single_cols)):
            cost[off_single + p] = 1.0
    elif objective == "multi":
        for p in range(len(multi_cols)):
            cost[off_multi + p] = 1.0

    lb = [0.0] * total
    ub = [1.0] * n + [np.inf] * (total - n)
    for i in forced_in:
        lb[i] = 1.0
    for i in forced_out:
        ub[i] = 0.0
    integrality = [1] * n + [0] * (total - n)

    result = milp(c=np.array(cost), constraints=LinearConstraint(np.array(rows), lower, upper),
                  bounds=Bounds(np.array(lb), np.array(ub)),
                  integrality=np.array(integrality))
    if not result.success or result.x is None:
        return False, None, None
    picked = sorted(i for i in range(n) if result.x[i] > 0.5)
    value = None if objective is None else int(round(float(result.fun)))
    if feasibility_only:
        return True, None, picked
    return True, value, picked


# --------------------------------------------------------------------- public
@dataclass(frozen=True)
class SelectionResult:
    arm: str
    selected_shas: list[str]
    selected_recipes: list[RecipeV11]
    rejected_shas: list[str]
    s_pref: int
    s_single: int
    s_multi: int
    counts: dict[str, dict[str, int]]
    eligible_count: int
    tie_break_trace: list[dict[str, Any]]

    @property
    def selected_set_identity(self) -> str:
        text = json.dumps(sorted(self.selected_shas), separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "selection_version": SELECTION_VERSION,
            "arm": self.arm,
            "eligible_count": self.eligible_count,
            "selected_count": len(self.selected_shas),
            "rejected_count": len(self.rejected_shas),
            "objective": {"S_pref": self.s_pref, "S_single": self.s_single,
                          "S_multi": self.s_multi},
            "counts": self.counts,
            "hard_violations": hard_violations(self.counts),
            "selected_shas": list(self.selected_shas),
            "rejected_shas": list(self.rejected_shas),
            "selected_set_identity": self.selected_set_identity,
            "tie_break_trace": self.tie_break_trace,
        }


def select(recipes: Sequence[RecipeV11], ontology: Ontology, *, arm: str = "LLM",
           bank_size: int = FINAL_BANK_SIZE_PER_ARM,
           minimum_pool: int = MINIMUM_ELIGIBLE_POOL_PER_ARM,
           enforce_minimum_pool: bool = True) -> SelectionResult:
    """Run prism_c3_selection_v1 over one arm's eligible pool.

    `recipes` must already be the UNIQUE, fully valid and compilable pool from
    `prism_fas.recipes.eligibility`. Selection never revalidates, repairs or
    extends it.
    """
    if enforce_minimum_pool and len(recipes) < minimum_pool:
        raise SelectionInfeasible(
            f"arm {arm!r} has {len(recipes)} unique eligible candidates, below the frozen "
            f"minimum of {minimum_pool}; C3 FAILS for this arm. No extra candidate "
            "generation, validator relaxation or manual replacement is permitted (§7.8.1)")
    if len(recipes) < bank_size:
        raise SelectionInfeasible(
            f"arm {arm!r} has {len(recipes)} eligible candidates, fewer than the required "
            f"bank size {bank_size}")

    model, ordered = _build_model(recipes, ontology)

    # Stage 1 — hard feasibility.
    feasible, _, _ = _solve(model, bank_size=bank_size, objective=None, fixed=None,
                            forced_in=set(), forced_out=set(), feasibility_only=True)
    if not feasible:
        raise SelectionInfeasible(
            f"arm {arm!r}: no {bank_size}-recipe subset satisfies the hard coverage "
            "constraints in §7.8.2; C3 FAILS for this arm. Coverage is never weakened "
            "and compatibility is never relaxed to make it feasible")

    # Stages 2-4 — lexicographic optima.
    _, pref_star, _ = _solve(model, bank_size=bank_size, objective="pref", fixed=None,
                             forced_in=set(), forced_out=set(), feasibility_only=False)
    _, single_star, _ = _solve(model, bank_size=bank_size, objective="single",
                               fixed={"pref": pref_star}, forced_in=set(), forced_out=set(),
                               feasibility_only=False)
    _, multi_star, _ = _solve(model, bank_size=bank_size, objective="multi",
                              fixed={"pref": pref_star, "single": single_star},
                              forced_in=set(), forced_out=set(), feasibility_only=False)
    pinned = {"pref": pref_star, "single": single_star, "multi": multi_star}

    # Stage 5 — canonical tie-break by iterative feasibility fixing over ascending
    # SHA. This is what makes the answer independent of solver traversal.
    forced_in: set[int] = set()
    forced_out: set[int] = set()
    trace: list[dict[str, Any]] = []
    for index in range(model.n):
        if len(forced_in) == bank_size:
            forced_out |= {i for i in range(index, model.n) if i not in forced_in}
            trace.append({"index": index, "decision": "remaining_forced_out",
                          "reason": "bank already complete"})
            break
        remaining_slots = bank_size - len(forced_in)
        undecided = model.n - index - len(forced_in - set(range(index)))
        if remaining_slots > (model.n - index):
            raise SelectionError("internal: not enough remaining candidates")
        feasible, _, _ = _solve(model, bank_size=bank_size, objective=None, fixed=pinned,
                                forced_in=forced_in | {index}, forced_out=forced_out,
                                feasibility_only=True)
        if feasible:
            forced_in.add(index)
            trace.append({"index": index, "sha": model.shas[index], "decision": "include"})
        else:
            forced_out.add(index)
            trace.append({"index": index, "sha": model.shas[index], "decision": "exclude"})

    if len(forced_in) != bank_size:
        raise SelectionError(
            f"canonical tie-break selected {len(forced_in)} recipes, expected {bank_size}")

    selected = sorted(forced_in)
    selected_recipes = [ordered[i] for i in selected]
    selected_shas = [model.shas[i] for i in selected]
    rejected_shas = [model.shas[i] for i in range(model.n) if i not in forced_in]

    # Exact integer verification. The solver is an engine; these are the numbers.
    counts = counts_for(selected_recipes, ontology)
    violations = hard_violations(counts)
    if violations:
        raise SelectionError(f"selected bank violates a hard quota: {violations}")
    exact_pref, exact_single, exact_multi = (s_pref(counts), s_single(counts, bank_size),
                                             s_multi(counts))
    if (exact_pref, exact_single, exact_multi) != (pref_star, single_star, multi_star):
        raise SelectionError(
            "recomputed objective disagrees with the solver: "
            f"exact=({exact_pref},{exact_single},{exact_multi}) "
            f"solver=({pref_star},{single_star},{multi_star})")
    if len(selected_shas) != bank_size or len(set(selected_shas)) != bank_size:
        raise SelectionError("selected bank is not exactly the required size, or repeats a SHA")

    return SelectionResult(arm=arm, selected_shas=selected_shas,
                           selected_recipes=selected_recipes, rejected_shas=rejected_shas,
                           s_pref=exact_pref, s_single=exact_single, s_multi=exact_multi,
                           counts=counts, eligible_count=len(recipes), tie_break_trace=trace)


__all__ = ["SELECTION_VERSION", "RAW_CANDIDATE_SLOTS_PER_ARM",
           "MINIMUM_ELIGIBLE_POOL_PER_ARM", "FINAL_BANK_SIZE_PER_ARM", "QUOTAS",
           "SINGLE_AXES", "MULTI_AXES", "AXES", "SelectionError", "SelectionInfeasible",
           "SelectionResult", "axis_categories", "recipe_axis_values", "counts_for",
           "s_pref", "s_single", "s_multi", "hard_violations", "select"]
