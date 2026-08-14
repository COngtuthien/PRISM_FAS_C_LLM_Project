"""The common C3 eligibility gate, applied identically to RND, DET and LLM.

v1.5 §7.8.3 fixes the order:

    item schema -> ontology -> scientific route policy -> numeric/range checks
    -> compatibility -> canonicalization -> deduplication -> compiler
    -> operator graph -> mask policy -> 41-D conditioning

Only UNIQUE candidates that pass the ENTIRE pipeline are eligible. A candidate
that fails at any stage is recorded with the stage that rejected it and is never
repaired, replaced or regenerated: §7.8.1 forbids extra candidate generation,
validator relaxation, manual replacement and target-guided repair once the frozen
384-slot budget is spent.

The gate is arm-blind. It receives candidate payloads and a slot schedule; it has
no parameter through which the arm's identity, provider response order, detector
score, synthetic quality or any target information could change a verdict.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from prism_fas.llm.pipeline import RecipePlanner, assign_recipe_id
from prism_fas.llm.route_policy import RoutePolicy
from prism_fas.recipes.canonical import canonical_json, recipe_hash
from prism_fas.recipes.compile import CompileError, compile_recipe
from prism_fas.recipes.ontology import Ontology
from prism_fas.recipes.schema import RecipeV11
from prism_fas.recipes.validate import validate_payload

#: The exact governing stage order (§7.8.3).
ELIGIBILITY_ORDER: tuple[str, ...] = (
    "item_schema",
    "ontology",
    "scientific_route_policy",
    "numeric_range",
    "compatibility",
    "canonicalization",
    "deduplication",
    "compiler",
    "operator_graph",
    "mask_policy",
    "conditioning_41d",
)

#: The inherited validator reports these stages; map them onto the governing
#: names so a rejection is attributed to the right pipeline position.
_VALIDATOR_STAGE_TO_ELIGIBILITY: dict[str, str] = {
    "schema": "numeric_range",          # ranges, counts and seed bounds
    "canonical": "ontology",            # enum membership under this ontology
    "medium_artifact": "compatibility",
    "geometry_region": "compatibility",
    "strength_range": "numeric_range",
    "severity": "numeric_range",
    "duplicate": "item_schema",         # intra-recipe repeats
    "route": "scientific_route_policy",
    "shortcut": "compatibility",
    "leakage": "item_schema",
    "serialization": "canonicalization",
    "hash": "canonicalization",
}

CONDITIONING_DIM = 41


@dataclass(frozen=True)
class CandidateVerdict:
    """One candidate slot's outcome. Immutable evidence."""

    slot_index: int
    slot_id: str
    eligible: bool
    rejected_at: str | None = None
    reasons: list[str] = field(default_factory=list)
    recipe_id: str | None = None
    canonical_recipe: str | None = None
    canonical_sha256: str | None = None
    content_identity: str | None = None
    graph_hash: str | None = None
    duplicate_of_slot_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "slot_index": self.slot_index,
            "slot_id": self.slot_id,
            "eligible": self.eligible,
            "rejected_at": self.rejected_at,
            "reasons": list(self.reasons),
            "recipe_id": self.recipe_id,
            "canonical_sha256": self.canonical_sha256,
            "content_identity": self.content_identity,
            "graph_hash": self.graph_hash,
            "duplicate_of_slot_id": self.duplicate_of_slot_id,
        }


@dataclass(frozen=True)
class EligiblePool:
    """The unique, fully valid and compilable pool for one arm."""

    arm: str
    raw_slots: int
    verdicts: list[CandidateVerdict]
    recipes: list[RecipeV11]
    minimum_required: int

    @property
    def eligible_count(self) -> int:
        return len(self.recipes)

    @property
    def meets_minimum(self) -> bool:
        return self.eligible_count >= self.minimum_required

    def rejection_histogram(self) -> dict[str, int]:
        counts = {stage: 0 for stage in ELIGIBILITY_ORDER}
        for verdict in self.verdicts:
            if not verdict.eligible and verdict.rejected_at:
                counts[verdict.rejected_at] = counts.get(verdict.rejected_at, 0) + 1
        return counts

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "raw_candidate_slots": self.raw_slots,
            "eligible_count": self.eligible_count,
            "minimum_required": self.minimum_required,
            "meets_minimum": self.meets_minimum,
            "rejection_histogram": self.rejection_histogram(),
            "eligibility_order": list(ELIGIBILITY_ORDER),
            "verdicts": [verdict.as_dict() for verdict in self.verdicts],
        }


class EligibilityError(ValueError):
    """The gate was invoked with a malformed schedule."""


def _stage_of(issues: Sequence[dict[str, Any]]) -> str:
    """Earliest governing stage among the validator's reported issues."""
    positions = []
    for issue in issues:
        mapped = _VALIDATOR_STAGE_TO_ELIGIBILITY.get(str(issue.get("stage", "")), "item_schema")
        positions.append(ELIGIBILITY_ORDER.index(mapped))
    return ELIGIBILITY_ORDER[min(positions)] if positions else "item_schema"


def evaluate_pool(*, arm: str, candidates: Sequence[dict[str, Any] | None],
                  slot_ids: Sequence[str], ontology: Ontology, route_policy: RoutePolicy,
                  bank_id: str, raw_slots: int, minimum_required: int) -> EligiblePool:
    """Run the frozen eligibility pipeline over one arm's candidate slots.

    `candidates[i]` is the raw recipe payload for `slot_ids[i]`, or None when the
    slot produced nothing (an unanswered request, a malformed response element).
    A None slot is a rejection at `item_schema`, never a silent omission: the
    slot budget is fixed and every slot is accounted for.
    """
    if len(candidates) != len(slot_ids):
        raise EligibilityError("candidates and slot_ids must be the same length")
    if len(candidates) != raw_slots:
        raise EligibilityError(
            f"arm {arm!r} must present exactly {raw_slots} raw candidate slots, "
            f"got {len(candidates)}; the budget is frozen and cannot be topped up")

    verdicts: list[CandidateVerdict] = []
    recipes: list[RecipeV11] = []
    seen_content: dict[str, str] = {}

    for index, (payload, slot_id) in enumerate(zip(candidates, slot_ids)):
        if payload is None:
            verdicts.append(CandidateVerdict(index, slot_id, False, "item_schema",
                                             ["slot produced no candidate object"]))
            continue

        # `recipe_id` is assigned by slot position: the system computes it, the
        # generator never supplies it (§7.3).
        prepared = {**payload, "recipe_id": assign_recipe_id(index)}

        # schema + ontology + numeric/range + compatibility, one inherited pass.
        # `canonicalize=False` keeps the alias map OFF, as frozen.
        recipe, issues = validate_payload(prepared, ontology, canonicalize=False)
        if recipe is None:
            verdicts.append(CandidateVerdict(index, slot_id, False, "item_schema",
                                             [issue.reason for issue in issues]))
            continue
        if issues:
            verdicts.append(CandidateVerdict(
                index, slot_id, False, _stage_of([issue.as_dict() for issue in issues]),
                [issue.reason for issue in issues], recipe_id=recipe.recipe_id))
            continue

        # scientific route policy (§7.3.1) — before canonicalization and compiler.
        route_issues = route_policy.check(recipe)
        if route_issues:
            verdicts.append(CandidateVerdict(
                index, slot_id, False, "scientific_route_policy",
                [issue["reason"] for issue in route_issues], recipe_id=recipe.recipe_id))
            continue

        # canonicalization + canonical identity
        text = canonical_json(recipe)
        sha = recipe_hash(recipe)
        content = RecipePlanner.content_identity(recipe)

        # deduplication over semantic content (recipe_id excluded)
        first = seen_content.get(content)
        if first is not None:
            verdicts.append(CandidateVerdict(
                index, slot_id, False, "deduplication",
                [f"canonical content duplicates slot {first!r}"],
                recipe_id=recipe.recipe_id, canonical_sha256=sha,
                content_identity=content, duplicate_of_slot_id=first))
            continue

        # compiler -> operator graph -> mask policy -> 41-D conditioning
        try:
            graph = compile_recipe(recipe, ontology, bank_id=bank_id)
        except CompileError as exc:
            verdicts.append(CandidateVerdict(index, slot_id, False, "compiler", [str(exc)],
                                             recipe_id=recipe.recipe_id,
                                             canonical_sha256=sha, content_identity=content))
            continue
        if not graph.nodes:
            verdicts.append(CandidateVerdict(index, slot_id, False, "operator_graph",
                                             ["compiled graph has no operator node"],
                                             recipe_id=recipe.recipe_id,
                                             canonical_sha256=sha, content_identity=content))
            continue
        if not graph.region_mask_policy.get("policy"):
            verdicts.append(CandidateVerdict(index, slot_id, False, "mask_policy",
                                             ["compiled graph declares no region mask policy"],
                                             recipe_id=recipe.recipe_id,
                                             canonical_sha256=sha, content_identity=content))
            continue
        if graph.conditioning_dimension != CONDITIONING_DIM:
            verdicts.append(CandidateVerdict(
                index, slot_id, False, "conditioning_41d",
                [f"conditioning dimension {graph.conditioning_dimension} != {CONDITIONING_DIM}"],
                recipe_id=recipe.recipe_id, canonical_sha256=sha, content_identity=content))
            continue

        seen_content[content] = slot_id
        recipes.append(recipe)
        verdicts.append(CandidateVerdict(index, slot_id, True, None, [],
                                         recipe_id=recipe.recipe_id, canonical_recipe=text,
                                         canonical_sha256=sha, content_identity=content,
                                         graph_hash=graph.graph_hash))

    return EligiblePool(arm=arm, raw_slots=raw_slots, verdicts=verdicts, recipes=recipes,
                        minimum_required=minimum_required)


__all__ = ["ELIGIBILITY_ORDER", "CONDITIONING_DIM", "CandidateVerdict", "EligiblePool",
           "EligibilityError", "evaluate_pool"]
