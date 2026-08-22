"""C6_MATCHED_BANK_SELECTOR_V1 — the frozen deterministic matched-bank selector.

§11.3 requires each arm to end with exactly 1024 accepted samples, 512 Physics
and 512 GPAT, "selected deterministically to balance source domain, route,
recipe coverage and base live IDs". It names the four dimensions and no
algorithm; this module is the algorithm, frozen by user decision before any C6
gate outcome was observed and with target_access = 0.

The priority order is itself result-affecting and part of the selector identity:

    0. HARD route cardinality          — 512 Physics and 512 GPAT, never traded
    1. COMMON source-domain exposure   — one quota vector for all three arms
    2. recipe coverage balance         — soft, greedy on lowest current exposure
    3. base-live exposure balance      — soft, below recipe
    4. canonical hash tie-break        — arm-independent, then candidate_id

Two properties are worth stating plainly because they are what make the
comparison fair.

**Quality does not select.** Once a candidate passes the selected common
profile, the gate is binary. `q`, the gate metrics, the margins and the number
of gates passed are all invisible to the ordering. `q` remains the §11.2
training weight and is serialized for that use, but a bank whose membership
depended on it would let a stronger arm assemble a different, better-tuned
training set than a weaker one — the Version-B confound in a new place.

**The target distribution comes from the frozen PLAN, not from what was
accepted.** Acceptance is a treatment outcome. If the desired source-domain
exposure were derived from accepted candidates, an arm whose recipes happen to
fail more often on one dataset would silently redefine what "balanced" means.
So `ideal[d]` is computed from the 1024 planned slots of the frozen C5 schedule,
which is identical across arms by construction.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .c5_source_pair_plan import ARMS, GPAT, PHYSICS

SCHEMA_VERSION = "prism-c6-matched-bank-selector-v1"
SELECTOR_NAME = "C6_MATCHED_BANK_SELECTOR_V1"

#: The tie-hash domain separator. Frozen: changing it changes every tie-break.
TIE_HASH_PREFIX = "PRISM_C6_MATCHED_BANK_SELECTOR_V1"
TIE_HASH_SEPARATOR = "|"
TIE_HASH_ENCODING = "utf-8"

#: The canonical source-domain field. It is the `dataset` column of the
#: finalized M3B `source_train` manifest, surfaced on each frozen C5 plan row as
#: `live_dataset`. Never inferred from a path, a filename or a directory.
SOURCE_DOMAIN_FIELD = "dataset"
SOURCE_DOMAIN_PLAN_FIELD = "live_dataset"

#: §11.3 final cardinality.
FINAL_BANK_PER_ARM = 1024
PER_ROUTE = 512
ROUTES: tuple[str, ...] = (PHYSICS, GPAT)

DIMENSION_PRIORITY: tuple[str, ...] = (
    "hard_route_cardinality",
    "common_source_domain_exposure",
    "recipe_coverage_balance",
    "base_live_exposure_balance",
    "canonical_tie_hash_then_candidate_id",
)


class MatchedBankError(RuntimeError):
    """The matched bank cannot be built under the frozen selector."""


# --- the candidate the selector sees ----------------------------------------

@dataclass(frozen=True)
class SelectableCandidate:
    """One gate-eligible generated candidate, as the selector sees it.

    `q` is carried because the selected bank serializes it for §11.2 training
    weighting. It is deliberately NOT part of `sort_key`, and a test asserts
    that changing it cannot change which candidates are selected.
    """

    candidate_id: str
    arm: str
    route: str
    source_domain: str
    recipe_id: str
    recipe_ordinal: int
    live_target_sample_id: str
    base_position: int
    q: float | None = None

    def tie_hash(self) -> str:
        return canonical_tie_hash(
            route=self.route, source_domain=self.source_domain,
            base_position=self.base_position,
            live_target_sample_id=self.live_target_sample_id)


def canonical_tie_hash(*, route: str, source_domain: str, base_position: int,
                       live_target_sample_id: str) -> str:
    """The arm-independent tie-break.

    Every input comes from the shared frozen C5 schedule, so two arms looking at
    the same base position compute the same hash and break ties the same way.
    The arm, the recipe-generator type, `q`, any gate metric or margin, any
    target information, any runtime path and any timestamp are all deliberately
    absent — including them would make the tie-break either arm-dependent or
    outcome-dependent, and the tie-break is the last thing standing between two
    otherwise identical candidates.
    """
    material = TIE_HASH_SEPARATOR.join((
        TIE_HASH_PREFIX, str(route), str(source_domain), str(int(base_position)),
        str(live_target_sample_id)))
    return hashlib.sha256(material.encode(TIE_HASH_ENCODING)).hexdigest()


def canonical_domains(domains: Iterable[str]) -> list[str]:
    """Canonical ascending domain-ID order. The one ordering rule."""
    return sorted({str(domain) for domain in domains})


# --- stage 1. the planned source-domain target ------------------------------

def planned_domain_counts(plan: Mapping[str, Any], route: str) -> dict[str, int]:
    """How many of this route's 1024 planned slots belong to each source domain.

    Read from the frozen C5 arm plan, which carries the pre-gate schedule. All
    three arms share it, so this vector is arm-independent by construction.
    """
    counts: dict[str, int] = {}
    for row in plan["candidates"]:
        if row["route"] != route:
            continue
        domain = str(row[SOURCE_DOMAIN_PLAN_FIELD])
        counts[domain] = counts.get(domain, 0) + 1
    if not counts:
        raise MatchedBankError(f"the plan holds no {route} slots")
    return {domain: counts[domain] for domain in canonical_domains(counts)}


def ideal_domain_share(planned: Mapping[str, int],
                       total: int = PER_ROUTE) -> dict[str, float]:
    """`ideal[d] = total * planned[d] / sum(planned)` — the pre-gate target."""
    planned_total = sum(planned.values())
    if planned_total <= 0:
        raise MatchedBankError("the planned route schedule is empty")
    return {domain: total * planned[domain] / planned_total
            for domain in canonical_domains(planned)}


def largest_remainder_quota(ideal: Mapping[str, float],
                            total: int = PER_ROUTE) -> dict[str, int]:
    """Integer quotas summing to exactly `total`, by largest remainder.

    Deterministic twice over: the floor is exact, and the remaining slots go to
    the largest fractional parts with canonical domain id ascending as the tie.
    A dict's insertion order cannot influence the result because every ordering
    is imposed here.
    """
    domains = canonical_domains(ideal)
    quota = {domain: int(ideal[domain] // 1) for domain in domains}
    remaining = int(total) - sum(quota.values())
    if remaining < 0:
        raise MatchedBankError("floor of the ideal share already exceeds the total")

    order = sorted(domains,
                   key=lambda domain: (-(ideal[domain] - quota[domain]), domain))
    for domain in order[:remaining]:
        quota[domain] += 1
    if sum(quota.values()) != int(total):
        raise MatchedBankError(
            f"largest-remainder quotas sum to {sum(quota.values())}, not {total}")
    return quota


# --- stage 2. common capacity and deficit redistribution --------------------

def accepted_capacity(candidates: Sequence[SelectableCandidate],
                      route: str) -> dict[str, int]:
    """Accepted candidates per source domain for one arm and one route."""
    counts: dict[str, int] = {}
    for candidate in candidates:
        if candidate.route != route:
            continue
        counts[candidate.source_domain] = counts.get(candidate.source_domain, 0) + 1
    return counts


def common_capacity(by_arm: Mapping[str, Sequence[SelectableCandidate]], route: str,
                    domains: Sequence[str]) -> dict[str, int]:
    """The per-domain capacity ALL arms can satisfy: the elementwise minimum.

    The final quota vector must be identical across arms — that is what makes
    the three banks matched — so the capacity that matters is the weakest arm's,
    domain by domain.
    """
    if set(by_arm) != set(ARMS):
        raise MatchedBankError(f"expected candidates for {ARMS}, got {sorted(by_arm)}")
    per_arm = {arm: accepted_capacity(candidates, route)
               for arm, candidates in by_arm.items()}
    return {domain: min(per_arm[arm].get(domain, 0) for arm in ARMS)
            for domain in canonical_domains(domains)}


@dataclass
class RouteQuota:
    """One route's frozen common source-domain quota, or why there is none."""

    route: str
    planned: dict[str, int]
    ideal: dict[str, float]
    initial_quota: dict[str, int]
    capacity: dict[str, int]
    quota: dict[str, int]
    feasible: bool
    clipped: dict[str, int] = field(default_factory=dict)
    redistribution: list[dict[str, Any]] = field(default_factory=list)
    shortfall: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"route": self.route, "planned": dict(self.planned),
                "ideal": {domain: round(value, 6)
                          for domain, value in self.ideal.items()},
                "initial_quota": dict(self.initial_quota),
                "common_capacity": dict(self.capacity),
                "quota": dict(self.quota), "feasible": self.feasible,
                "clipped": dict(self.clipped),
                "redistribution": list(self.redistribution),
                "shortfall": int(self.shortfall),
                "quota_total": sum(self.quota.values()),
                "identical_across_arms": True}


def resolve_route_quota(route: str, planned: Mapping[str, int],
                        capacity: Mapping[str, int],
                        total: int = PER_ROUTE) -> RouteQuota:
    """Clip the planned quota to what every arm can supply, then refill.

    Clipping first and refilling second is what keeps the result deterministic
    and independent of which domain happened to be short: the deficit is
    redistributed one slot at a time to whichever domain is furthest below its
    frozen planned target, so the outcome is the closest achievable vector to
    the pre-gate ideal rather than an artefact of iteration order.
    """
    domains = canonical_domains(set(planned) | set(capacity))
    ideal = ideal_domain_share(planned, total)
    initial = largest_remainder_quota(ideal, total)
    capacity = {domain: int(capacity.get(domain, 0)) for domain in domains}

    quota = dict(initial)
    clipped: dict[str, int] = {}
    for domain in domains:
        available = capacity.get(domain, 0)
        if quota.get(domain, 0) > available:
            clipped[domain] = quota[domain] - available
            quota[domain] = available

    redistribution: list[dict[str, Any]] = []
    deficit = int(total) - sum(quota.values())
    while deficit > 0:
        eligible = [domain for domain in domains if quota[domain] < capacity[domain]]
        if not eligible:
            break
        # Furthest below its frozen planned target first; canonical id on a tie.
        chosen = min(eligible,
                     key=lambda domain: (-(ideal[domain] - quota[domain]), domain))
        quota[chosen] += 1
        deficit -= 1
        redistribution.append({
            "domain": chosen, "quota_after": quota[chosen],
            "ideal_minus_quota_before": round(ideal[chosen] - (quota[chosen] - 1), 6),
            "capacity": capacity[chosen]})

    feasible = sum(quota.values()) == int(total)
    return RouteQuota(route=route, planned=dict(planned), ideal=ideal,
                      initial_quota=initial, capacity=capacity, quota=quota,
                      feasible=feasible, clipped=clipped,
                      redistribution=redistribution, shortfall=max(0, deficit))


# --- stage 3. the within-quota fill -----------------------------------------

def select_route_bank(candidates: Sequence[SelectableCandidate], *, route: str,
                      quota: Mapping[str, int]) -> list[dict[str, Any]]:
    """Fill one route's 512 slots for one arm, under the common domain quota.

    The ordering key is the whole selector below the quota level:

        (recipe_selected_count, live_selected_count, tie_hash, candidate_id)

    Recipe exposure dominates live exposure, live exposure dominates the hash,
    and the hash dominates the id. Every component is either a count this
    function maintains or a value fixed by the frozen schedule, so no quality
    number can reach the comparison.

    Greedy rather than a global optimisation on purpose: a greedy minimum on
    current exposure reaches the natural 2-per-recipe target whenever capacity
    allows, and when a recipe is short it simply keeps taking from the least
    exposed recipes. That is the "soft target" the decision specifies, and it
    needs no deficit bookkeeping of its own.
    """
    pool = [candidate for candidate in candidates if candidate.route == route]
    remaining = {domain: int(count) for domain, count in quota.items()}
    target = sum(remaining.values())

    recipe_count: dict[int, int] = {}
    live_count: dict[str, int] = {}
    selected: list[dict[str, Any]] = []
    available = list(pool)

    while len(selected) < target:
        eligible = [candidate for candidate in available
                    if remaining.get(candidate.source_domain, 0) > 0]
        if not eligible:
            raise MatchedBankError(
                f"{route}: only {len(selected)} of {target} slots could be filled "
                f"under the common source-domain quota")
        chosen = min(eligible, key=lambda candidate: (
            recipe_count.get(candidate.recipe_ordinal, 0),
            live_count.get(candidate.live_target_sample_id, 0),
            candidate.tie_hash(),
            candidate.candidate_id))

        selected.append({
            "candidate_id": chosen.candidate_id, "arm": chosen.arm,
            "route": chosen.route, "source_domain": chosen.source_domain,
            "recipe_id": chosen.recipe_id, "recipe_ordinal": chosen.recipe_ordinal,
            "live_target_sample_id": chosen.live_target_sample_id,
            "base_position": chosen.base_position,
            "selection_step": len(selected) + 1,
            "recipe_count_before": recipe_count.get(chosen.recipe_ordinal, 0),
            "live_count_before": live_count.get(chosen.live_target_sample_id, 0),
            "source_domain_quota": int(quota.get(chosen.source_domain, 0)),
            "canonical_tie_hash": chosen.tie_hash(),
            # Serialized for §11.2 training weighting. It played no part above.
            "q": chosen.q,
        })
        recipe_count[chosen.recipe_ordinal] = recipe_count.get(chosen.recipe_ordinal, 0) + 1
        live_count[chosen.live_target_sample_id] = (
            live_count.get(chosen.live_target_sample_id, 0) + 1)
        remaining[chosen.source_domain] -= 1
        available.remove(chosen)

    return selected


def exposure_summary(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """What the balance dimensions actually achieved, for the lock to record."""
    recipes: dict[int, int] = {}
    lives: dict[str, int] = {}
    domains: dict[str, int] = {}
    for row in selected:
        recipes[row["recipe_ordinal"]] = recipes.get(row["recipe_ordinal"], 0) + 1
        lives[row["live_target_sample_id"]] = lives.get(row["live_target_sample_id"], 0) + 1
        domains[row["source_domain"]] = domains.get(row["source_domain"], 0) + 1
    counts = sorted(recipes.values())
    return {"selected": len(selected),
            "by_source_domain": {domain: domains[domain]
                                 for domain in canonical_domains(domains)},
            "distinct_recipes": len(recipes),
            "recipe_exposure_min": counts[0] if counts else 0,
            "recipe_exposure_max": counts[-1] if counts else 0,
            "distinct_live_targets": len(lives),
            "live_exposure_max": max(lives.values()) if lives else 0}


# --- the whole selector ------------------------------------------------------

def route_quotas(plans: Mapping[str, Mapping[str, Any]],
                 by_arm: Mapping[str, Sequence[SelectableCandidate]]
                 ) -> dict[str, RouteQuota]:
    """The common quota vector for each route, or the infeasibility."""
    reference = plans[ARMS[0]]
    resolved: dict[str, RouteQuota] = {}
    for route in ROUTES:
        planned = planned_domain_counts(reference, route)
        capacity = common_capacity(by_arm, route, list(planned))
        resolved[route] = resolve_route_quota(route, planned, capacity)
    return resolved


def matched_feasible(quotas: Mapping[str, RouteQuota]) -> bool:
    """A profile is matched-feasible only when BOTH routes resolve a quota."""
    return all(quotas[route].feasible for route in ROUTES) and set(quotas) >= set(ROUTES)


def build_matched_banks(plans: Mapping[str, Mapping[str, Any]],
                        by_arm: Mapping[str, Sequence[SelectableCandidate]]
                        ) -> dict[str, Any]:
    """The three matched banks, or a typed infeasibility.

    Every arm runs the identical algorithm against the identical quota vector.
    What differs between arms is only which candidates their own recipes
    produced and the gate accepted — which is the treatment under test.
    """
    quotas = route_quotas(plans, by_arm)
    if not matched_feasible(quotas):
        return {"schema_version": SCHEMA_VERSION, "selector": SELECTOR_NAME,
                "matched": False,
                "route_quotas": {route: quota.as_dict()
                                 for route, quota in quotas.items()},
                "reason": "COMMON_SOURCE_DOMAIN_QUOTA_INFEASIBLE",
                "banks": {}}

    banks: dict[str, Any] = {}
    for arm in ARMS:
        rows: list[dict[str, Any]] = []
        for route in ROUTES:
            rows.extend(select_route_bank(by_arm[arm], route=route,
                                          quota=quotas[route].quota))
        if len(rows) != FINAL_BANK_PER_ARM:
            raise MatchedBankError(
                f"{arm} bank holds {len(rows)} samples, not {FINAL_BANK_PER_ARM}")
        banks[arm] = {
            "arm": arm, "selected": rows, "size": len(rows),
            "by_route": {route: sum(1 for row in rows if row["route"] == route)
                         for route in ROUTES},
            "exposure": {route: exposure_summary(
                [row for row in rows if row["route"] == route]) for route in ROUTES},
            "selected_set_sha256": selected_set_digest(rows),
        }

    return {"schema_version": SCHEMA_VERSION, "selector": SELECTOR_NAME,
            "matched": True,
            "route_quotas": {route: quota.as_dict() for route, quota in quotas.items()},
            "banks": banks,
            "final_bank_per_arm": FINAL_BANK_PER_ARM,
            "per_route": PER_ROUTE}


def selected_set_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """Identity over WHICH candidates were selected, in selection order."""
    material = "|".join(f"{row['selection_step']}:{row['candidate_id']}"
                        for row in rows)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def selector_identity(*, quality_profile_identity: str, c5_pool_lock_sha256: str,
                      decision_set_sha256: str) -> dict[str, Any]:
    """The frozen selector contract, and what it was applied to.

    The selector's own rules and the three identities it consumed are hashed
    together, so a bank cannot be re-attributed to a different pool, a different
    gate profile or a different set of accept/reject decisions.
    """
    contract = {
        "schema_version": SCHEMA_VERSION,
        "selector_name": SELECTOR_NAME,
        "dimension_priority": list(DIMENSION_PRIORITY),
        "source_domain_field": SOURCE_DOMAIN_FIELD,
        "source_domain_plan_field": SOURCE_DOMAIN_PLAN_FIELD,
        "domain_ordering_rule": "canonical source-domain id ascending",
        "planned_distribution_rule": (
            "ideal[d] = 512 * planned[d] / 1024 over the frozen PRE-GATE C5 "
            "schedule; never over accepted candidates"),
        "largest_remainder_rule": (
            "floor(ideal), then remaining slots by descending fractional part, "
            "ties by canonical domain id ascending; quotas sum to exactly 512"),
        "common_capacity_rule": (
            "capacity[d] = min over RND/DET/LLM of accepted candidates in "
            "(route, d); the final quota vector is identical for all three arms"),
        "deficit_redistribution_rule": (
            "clip quota to capacity, then refill one slot at a time to the "
            "domain maximizing ideal[d] - quota[d] among domains with "
            "quota[d] < capacity[d]; ties by canonical domain id ascending"),
        "recipe_balancing_rule": (
            "soft: greedy minimum recipe_selected_count. 2 per recipe per route "
            "is reached whenever capacity permits; a short recipe is not a "
            "failure and is never replaced"),
        "live_balancing_rule": (
            "soft: greedy minimum live_selected_count, ranked BELOW recipe "
            "exposure; no hard one-candidate-per-live constraint"),
        "tie_hash_rule": {
            "material": f"{TIE_HASH_PREFIX}|route|source_domain|base_position|"
                        f"live_target_sample_id",
            "separator": TIE_HASH_SEPARATOR, "encoding": TIE_HASH_ENCODING,
            "algorithm": "sha256",
            "excluded": ["arm", "recipe_generator_type", "q", "gate_metrics",
                         "acceptance_margin", "target_information",
                         "runtime_path", "timestamp"]},
        "final_tie_break": "candidate_id ascending",
        "final_cardinality": {"per_arm": FINAL_BANK_PER_ARM,
                              "physics": PER_ROUTE, "gpat": PER_ROUTE},
        "quality_ranking_used": False,
        "quality_gate_is_binary_for_selection": True,
    }
    material = "|".join((
        _stable(contract), str(quality_profile_identity), str(c5_pool_lock_sha256),
        str(decision_set_sha256)))
    return {**contract,
            "quality_profile_identity": quality_profile_identity,
            "c5_pool_lock_sha256": c5_pool_lock_sha256,
            "candidate_decision_set_sha256": decision_set_sha256,
            "selector_identity_sha256":
                hashlib.sha256(material.encode("utf-8")).hexdigest()}


def _stable(payload: Any) -> str:
    import json

    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def decision_set_digest(decisions: Sequence[Mapping[str, Any]]) -> str:
    """Identity over every accept/reject decision the gate produced."""
    material = sorted(f"{row['candidate_id']}:{'1' if row['accepted'] else '0'}"
                      for row in decisions)
    return hashlib.sha256("|".join(material).encode("utf-8")).hexdigest()


__all__ = ["SCHEMA_VERSION", "SELECTOR_NAME", "TIE_HASH_PREFIX", "TIE_HASH_SEPARATOR",
           "TIE_HASH_ENCODING", "SOURCE_DOMAIN_FIELD", "SOURCE_DOMAIN_PLAN_FIELD",
           "FINAL_BANK_PER_ARM", "PER_ROUTE", "ROUTES", "DIMENSION_PRIORITY",
           "MatchedBankError", "SelectableCandidate", "canonical_tie_hash",
           "canonical_domains", "planned_domain_counts", "ideal_domain_share",
           "largest_remainder_quota", "accepted_capacity", "common_capacity",
           "RouteQuota", "resolve_route_quota", "select_route_bank",
           "exposure_summary", "route_quotas", "matched_feasible",
           "build_matched_banks", "selected_set_digest", "selector_identity",
           "decision_set_digest"]
