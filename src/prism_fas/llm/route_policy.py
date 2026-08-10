"""The scientific route contract: `generator_route` is frozen, not a free axis.

C2B produced 32 schema-valid recipes of which 10 could not be compiled, because
they declared `generator_route` without `physics`. Those recipes passed every
semantic rule and then had no operator graph to build: an operator graph is a
physics-route artifact. The validator and the compiler disagreed about what an
acceptable recipe is.

Version-C v1.1 resolves that disagreement in favour of the synthesis design, not
the LLM. Every scientific recipe must be executable through BOTH routes, because

* the physics route is mandatory;
* each recipe receives exactly four physics and four GPAT candidate renders;
* the RND / DET / LLM arms must have identical route exposure, so a recipe that
  can only be rendered one way would confound the comparison.

`generator_route` is therefore an execution contract, and the only accepted
declaration is exactly `["physics", "gpat"]`.

Two things this module deliberately does NOT do:

* It never repairs. A recipe declaring `["gpat"]` is rejected and recorded, not
  silently given a physics route. The provider has to declare the contract
  itself, or its output is invalid.
* It never creates a GPAT-only accepted class. Version-C v1.1 has no such class,
  and inventing one would reintroduce the route-exposure confound.

The policy is identity-bearing. `route_policy_identity` enters the generation
request identity and, later, the recipe-bank provenance and the C3 bank lock, so
changing the policy after C3 begins invalidates the bank identity.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from prism_fas.recipes.ontology import Ontology
from prism_fas.recipes.schema import RecipeV11

ROUTE_POLICY_VERSION = "prism_c_route_policy_v1"

#: The single accepted scientific route declaration, in canonical order.
SCIENTIFIC_GENERATOR_ROUTE: tuple[str, ...] = ("physics", "gpat")

#: The rejection reason recorded when a candidate breaches the contract.
ROUTE_POLICY_STAGE = "scientific_route_policy"
ROUTE_POLICY_VIOLATION = "SCIENTIFIC_ROUTE_POLICY_VIOLATION"


class RoutePolicyError(ValueError):
    """The route policy itself is malformed or disagrees with the ontology."""


@dataclass(frozen=True)
class RoutePolicy:
    """The frozen route contract. Immutable, hashable, self-describing."""

    version: str = ROUTE_POLICY_VERSION
    allowed_scientific_generator_route: tuple[str, ...] = SCIENTIFIC_GENERATOR_ROUTE
    require_exact_order: bool = True
    allow_subset: bool = False
    allow_gpat_only_class: bool = False
    silent_repair_permitted: bool = False
    rationale: str = (
        "Physics is mandatory and every scientific recipe receives 4 physics + 4 GPAT "
        "candidate renders, so the RND/DET/LLM arms must share identical route exposure. "
        "A recipe that only one route can render would confound that comparison."
    )

    # --- identity -----------------------------------------------------------
    def identity_material(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "allowed_scientific_generator_route": list(self.allowed_scientific_generator_route),
            "require_exact_order": self.require_exact_order,
            "allow_subset": self.allow_subset,
            "allow_gpat_only_class": self.allow_gpat_only_class,
            "silent_repair_permitted": self.silent_repair_permitted,
        }

    def canonical_text(self) -> str:
        """The canonical UTF-8 representation the identity is taken over."""
        return json.dumps(self.identity_material(), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False)

    @property
    def route_policy_identity(self) -> str:
        return hashlib.sha256(self.canonical_text().encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {**self.identity_material(), "rationale": self.rationale,
                "canonical_text": self.canonical_text(),
                "route_policy_identity": self.route_policy_identity}

    # --- self-check ---------------------------------------------------------
    def validate_against(self, ontology: Ontology) -> None:
        """Fail closed if the policy names a route the ontology does not enable."""
        if not self.allowed_scientific_generator_route:
            raise RoutePolicyError("the route policy must name at least one route")
        unknown = [name for name in self.allowed_scientific_generator_route
                   if name not in ontology.routes]
        if unknown:
            raise RoutePolicyError(
                f"the route policy names routes the ontology does not enable: {unknown}")
        if len(set(self.allowed_scientific_generator_route)) != len(
                self.allowed_scientific_generator_route):
            raise RoutePolicyError("the route policy repeats a route")
        if self.allow_gpat_only_class:
            raise RoutePolicyError(
                "Version-C v1.1 has no GPAT-only scientific class; enabling one would "
                "reintroduce route-exposure confounding")
        if self.silent_repair_permitted:
            raise RoutePolicyError("silent route repair is never permitted")

    # --- enforcement --------------------------------------------------------
    def violations(self, declared: Sequence[str]) -> list[str]:
        """Every way `declared` breaches the contract. Empty means compliant."""
        allowed = list(self.allowed_scientific_generator_route)
        problems: list[str] = []
        values = list(declared)

        if not values:
            problems.append("generator_route is empty; the scientific contract requires "
                            f"exactly {allowed}")
            return problems
        if len(set(values)) != len(values):
            problems.append(f"generator_route repeats a value: {values}")
        unknown = [name for name in values if name not in allowed]
        if unknown:
            problems.append(f"generator_route contains {unknown}, which is not part of the "
                            f"scientific contract {allowed}")
        missing = [name for name in allowed if name not in values]
        if missing:
            problems.append(f"generator_route omits {missing}; every scientific recipe must be "
                            f"executable through BOTH routes, so exactly {allowed} is required")
        if self.require_exact_order and not problems and values != allowed:
            problems.append(f"generator_route must be exactly {allowed} in that order, got "
                            f"{values}")
        return problems

    def check(self, recipe: RecipeV11) -> list[dict[str, Any]]:
        """Route-policy issues for one parsed recipe, in the validator's shape."""
        return [{"recipe_id": recipe.recipe_id,
                 "stage": ROUTE_POLICY_STAGE,
                 "field": "generator_route",
                 "code": ROUTE_POLICY_VIOLATION,
                 "reason": reason}
                for reason in self.violations(recipe.generator_route)]

    def compliant(self, recipe: RecipeV11) -> bool:
        return not self.violations(recipe.generator_route)


#: The frozen Version-C v1.1 policy.
SCIENTIFIC_ROUTE_POLICY = RoutePolicy()


# ---------------------------------------------------------------------- loading
_ALLOWED_KEYS = {"version", "allowed_scientific_generator_route", "require_exact_order",
                 "allow_subset", "allow_gpat_only_class", "silent_repair_permitted",
                 "rationale"}


def parse_route_policy(payload: Any) -> RoutePolicy:
    """Strict parse. An unknown key is rejected rather than ignored."""
    if not isinstance(payload, dict):
        raise RoutePolicyError("route policy must be a mapping")
    unknown = sorted(set(payload) - _ALLOWED_KEYS)
    if unknown:
        raise RoutePolicyError(f"unknown route policy keys {unknown}")
    fields = dict(payload)
    if "allowed_scientific_generator_route" in fields:
        fields["allowed_scientific_generator_route"] = tuple(
            fields["allowed_scientific_generator_route"])
    policy = RoutePolicy(**fields)
    if policy.version != ROUTE_POLICY_VERSION:
        raise RoutePolicyError(f"unsupported route policy version {policy.version!r}")
    return policy


def load_route_policy(path: Path | str) -> RoutePolicy:
    import yaml
    return parse_route_policy(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def audit(policy: RoutePolicy, recipes: Sequence[RecipeV11]) -> dict[str, Any]:
    """Route distribution and compliance over a set of parsed recipes."""
    from collections import Counter

    counts: Counter[str] = Counter()
    compliant: list[str] = []
    violating: list[dict[str, Any]] = []
    for recipe in recipes:
        counts["+".join(recipe.generator_route)] += 1
        problems = policy.violations(recipe.generator_route)
        if problems:
            violating.append({"recipe_id": recipe.recipe_id,
                              "generator_route": list(recipe.generator_route),
                              "reasons": problems})
        else:
            compliant.append(recipe.recipe_id)
    return {
        "route_policy_version": policy.version,
        "route_policy_identity": policy.route_policy_identity,
        "required_route": list(policy.allowed_scientific_generator_route),
        "recipes_examined": len(recipes),
        "route_counts": dict(sorted(counts.items())),
        "compliant_count": len(compliant),
        "violating_count": len(violating),
        "violations": violating,
        "all_compliant": not violating,
        "silent_repairs_performed": 0,
        "gpat_only_class_created": False,
    }


__all__ = ["ROUTE_POLICY_VERSION", "SCIENTIFIC_GENERATOR_ROUTE", "ROUTE_POLICY_STAGE",
           "ROUTE_POLICY_VIOLATION", "RoutePolicy", "RoutePolicyError",
           "SCIENTIFIC_ROUTE_POLICY", "parse_route_policy", "load_route_policy", "audit"]
