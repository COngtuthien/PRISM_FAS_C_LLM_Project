"""Offline response fixtures for exercising the C3 live path without a provider.

The smoke profile has to drive the real generation code — the state machine, the
archiver, retry classification, checkpointing and resume — while being
structurally unable to reach Gemini. That needs responses, and they have to be
good enough to pass the same validator the live path uses, because a fixture
that failed validation would exercise only the error branch.

The recipes come from `prism_fas.recipes.arm_schedules.draft_candidate`, the
deterministic drafter the RND/DET control arms already use. Reusing it means the
fixture cannot drift from the ontology, the ranges or the route contract: if the
ontology changes, the drafts change with it, and a fixture that stopped
validating would be telling us something real.

This is engineering scaffolding, not science. Nothing built here may enter a
scientific bank: the candidates are drafted from a fixture seed offset that no
scientific arm uses, and every artifact produced from them is written under the
smoke namespace and stamped `scientific_eligible=false`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: Offset applied to the fixture's slot naming so a fixture draft can never
#: collide with a real RND or DET scientific slot id.
FIXTURE_SLOT_PREFIX = "SMOKEFIXTURE"


def build_response_text(*, repo: Path, recipes: int, salt: int = 0) -> str:
    """One provider-shaped response carrying exactly `recipes` valid candidates.

    Deterministic in `salt`, so a test can ask for two responses that are both
    valid and mutually distinct — which matters because the planner rejects a
    duplicate canonical identity, and a fixture that returned the same 32
    recipes twice would fail on the second logical request for the wrong reason.
    """
    from prism_fas.recipes.arm_schedules import draft_candidate
    from prism_fas.recipes.ontology import load_ontology

    ontology = load_ontology(repo / "configs/recipes/ontology_m7.yaml")
    items: list[dict[str, Any]] = []
    for index in range(recipes):
        slot = f"{FIXTURE_SLOT_PREFIX}_SLOT_{salt * 1000 + index:06d}"
        items.append(draft_candidate("DET", slot, ontology))
    return json.dumps({"recipes": items}, sort_keys=True, separators=(",", ":"))


def scripted_success(*, repo: Path, recipes: int, count: int, salt_base: int = 0) -> list[Any]:
    """`count` scripted successful responses, each distinct from the others."""
    from prism_fas.llm.providers.mock import ScriptedResponse

    return [ScriptedResponse(raw_text=build_response_text(
        repo=repo, recipes=recipes, salt=salt_base + index)) for index in range(count)]


def smoke_provider(*, repo: Path, recipes: int, count: int) -> Any:
    """A mock provider scripted to satisfy `count` logical requests offline."""
    from prism_fas.llm.providers.mock import MockRecipeProvider

    return MockRecipeProvider(scripted_success(repo=repo, recipes=recipes, count=count),
                              model_id="smoke-fixture-model")


__all__ = ["FIXTURE_SLOT_PREFIX", "build_response_text", "scripted_success", "smoke_provider"]
