"""Forbidden selector inputs, demonstrated with paired fixtures.

§7.8.3 forbids the selector from seeing SiW information, target metadata, target
metrics, Version-B target errors, human aesthetics, downstream detector scores,
synthetic quality q, quality-gate outcomes, manual ranking and provider response
order. `test_c3_selection.py` argues that structurally, by scanning the selector
for identifiers it must not read. This file argues it EXPERIMENTALLY.

The method is paired fixtures: two pools identical in every permitted field and
differing only in a forbidden one. The selected 256-SHA set must be bit-identical
across the pair. If any forbidden value reached the objective, the tie-break or
the ordering, the pair would diverge.

Nothing here reads a dataset. Every "target metric" is an invented number
attached to a synthetic fixture; no SiW file, label or metric is opened, and no
provider is contacted.
"""
from __future__ import annotations

import contextlib
import json

import pytest

from prism_fas.recipes import selection as sel
from prism_fas.recipes.canonical import recipe_hash
from prism_fas.recipes.eligibility import evaluate_pool
from prism_fas.recipes.schema import RecipeV11
from prism_fas.recipes.selection import select

from c3_fixtures import make_payload, make_pool

BANK = 24
POOL = 40

#: Values a compromised caller might try to hand the selector. Each name is one
#: the contract forbids. The numbers are invented, not measured.
FORBIDDEN_METADATA: dict[str, object] = {
    "siw_mv2_attack_family": "print",
    "siw_subject_id": "SiW_0421",
    "target_acer": 0.0731,
    "target_apcer": 0.0512,
    "target_bpcer": 0.0950,
    "version_b_target_error": 0.1184,
    "detector_score": 0.8817,
    "synthetic_quality_q": 0.4210,
    "q": 0.4210,
    "quality_gate_outcome": "PASS",
    "human_preference_rank": 3,
    "manual_ranking": 1,
}

TINY_QUOTAS: dict[str, dict[str, int | None]] = {
    "medium":       {"hard_min": 1, "hard_max": BANK, "preferred_min": None},
    "geometry":     {"hard_min": 1, "hard_max": BANK, "preferred_min": None},
    "illumination": {"hard_min": 1, "hard_max": BANK, "preferred_min": None},
    "artifact":     {"hard_min": 1, "hard_max": BANK * 3, "preferred_min": 6},
    "region":       {"hard_min": 1, "hard_max": BANK * 3, "preferred_min": 4},
}


@contextlib.contextmanager
def frozen_quotas(quotas):
    original = dict(sel.QUOTAS)
    sel.QUOTAS.update(quotas)
    try:
        yield
    finally:
        sel.QUOTAS.clear()
        sel.QUOTAS.update(original)


def taint(recipe: RecipeV11, extra: dict[str, object]) -> RecipeV11:
    """Smuggle forbidden metadata onto a recipe instance.

    The schema is `frozen=True, extra='forbid'`, so ordinary assignment and
    ordinary validation both refuse these fields — which is itself part of the
    guarantee. `object.__setattr__` goes around pydantic entirely, which is the
    point: even an attacker who bypasses the schema cannot move the bank.
    """
    clone = recipe.model_copy()
    object.__setattr__(clone, "__dict__", {**clone.__dict__, **extra})
    return clone


def selected(pool, ontology):
    with frozen_quotas(TINY_QUOTAS):
        return select(pool, ontology, bank_size=BANK, enforce_minimum_pool=False)


# ------------------------------------------------------- structural guarantee
def test_the_recipe_schema_refuses_forbidden_fields_outright():
    """A forbidden field cannot even be validated onto a recipe."""
    assert RecipeV11.model_config["extra"] == "forbid"
    assert RecipeV11.model_config["frozen"] is True


def test_a_payload_carrying_a_forbidden_field_fails_validation(ontology):
    from prism_fas.recipes.validate import validate_payload

    payload = make_payload(ontology, 0)
    payload["recipe_id"] = "R-000000"
    clean, issues = validate_payload(dict(payload), ontology, canonicalize=False)
    assert clean is not None and not issues

    tainted = {**payload, "target_acer": 0.07, "detector_score": 0.9}
    recipe, issues = validate_payload(tainted, ontology, canonicalize=False)
    assert recipe is None or issues, (
        "a payload carrying target metrics was accepted by the item schema")


# --------------------------------------------------- paired-fixture equality
@pytest.mark.parametrize("field, value", sorted(FORBIDDEN_METADATA.items()))
def test_one_forbidden_field_at_a_time_cannot_move_the_bank(field, value, ontology):
    """Pairs differing in exactly one forbidden field select the same bank."""
    clean = make_pool(ontology, POOL, tag="fi")
    tainted = [taint(recipe, {field: value}) for recipe in clean]

    baseline = selected(clean, ontology)
    observed = selected(tainted, ontology)

    assert observed.selected_shas == baseline.selected_shas, (
        f"the forbidden field {field!r} changed the selected bank")
    assert observed.selected_set_identity == baseline.selected_set_identity
    assert (observed.s_pref, observed.s_single, observed.s_multi) == (
        baseline.s_pref, baseline.s_single, baseline.s_multi)


def test_every_forbidden_field_at_once_cannot_move_the_bank(ontology):
    clean = make_pool(ontology, POOL, tag="fi")
    tainted = [taint(recipe, FORBIDDEN_METADATA) for recipe in clean]

    baseline = selected(clean, ontology)
    observed = selected(tainted, ontology)

    assert observed.selected_shas == baseline.selected_shas
    assert observed.selected_set_identity == baseline.selected_set_identity


def test_forbidden_metadata_does_not_enter_the_canonical_identity(ontology):
    """The canonical SHA is computed from schema fields only, so a smuggled
    field cannot change a recipe's identity and therefore cannot reorder the
    stage-5 tie-break."""
    for index, recipe in enumerate(make_pool(ontology, 8, tag="fi")):
        assert recipe_hash(taint(recipe, FORBIDDEN_METADATA)) == recipe_hash(recipe), (
            f"fixture {index}: forbidden metadata leaked into the canonical hash")


def test_a_forbidden_value_correlated_with_quality_cannot_bias_selection(ontology):
    """The adversarial case: scores deliberately correlated with SHA order.

    If the selector consulted `detector_score` at all — preferring high, low or
    anything in between — assigning scores in ascending and then descending SHA
    order would produce different banks. It produces the same one.
    """
    clean = make_pool(ontology, POOL, tag="fi")
    by_sha = sorted(clean, key=recipe_hash)

    ascending = [taint(recipe, {"detector_score": position / len(by_sha),
                                "target_acer": position / len(by_sha)})
                 for position, recipe in enumerate(by_sha)]
    descending = [taint(recipe, {"detector_score": 1.0 - position / len(by_sha),
                                 "target_acer": 1.0 - position / len(by_sha)})
                  for position, recipe in enumerate(by_sha)]

    baseline = selected(clean, ontology)
    high = selected(ascending, ontology)
    low = selected(descending, ontology)

    assert high.selected_shas == baseline.selected_shas
    assert low.selected_shas == baseline.selected_shas


# ------------------------------------------------- provider response ordering
def test_provider_response_arrival_order_does_not_change_the_pool(ontology, route_policy):
    """§7.8: 12 logical requests x 32 objects fill FIXED slot ranges.

    Request k owns slots 32k..32k+31 whatever order the responses come back in,
    so a re-ordered arrival sequence must produce a byte-identical slot array and
    therefore an identical eligible pool. (Reassigning candidates to DIFFERENT
    slots would legitimately change the bank — `recipe_id` is positional and
    enters the canonical hash — which is exactly why arrival order is decoupled
    from slot order.)
    """
    batch_size, batches = 32, 3
    slots = batch_size * batches
    grouped = [[make_payload(ontology, batch * batch_size + offset, tag="po")
                for offset in range(batch_size)] for batch in range(batches)]
    slot_ids = [f"C3_SLOT_{index:03d}" for index in range(slots)]

    def assemble(arrival_order):
        candidates: list[dict | None] = [None] * slots
        for batch in arrival_order:
            for offset, payload in enumerate(grouped[batch]):
                candidates[batch * batch_size + offset] = payload
        return candidates

    in_order = assemble([0, 1, 2])
    shuffled = assemble([2, 0, 1])
    reversed_order = assemble([2, 1, 0])
    assert in_order == shuffled == reversed_order, "slot assignment is arrival-dependent"

    pools = []
    for candidates in (in_order, shuffled, reversed_order):
        pool = evaluate_pool(arm="TEST", candidates=candidates, slot_ids=slot_ids,
                             ontology=ontology, route_policy=route_policy,
                             bank_id="c3-order-test", raw_slots=slots, minimum_required=0)
        pools.append([recipe_hash(recipe) for recipe in pool.recipes])

    assert pools[0] == pools[1] == pools[2]
    assert pools[0], "the fixture pool must be non-empty for this to mean anything"


def test_the_contract_names_every_forbidden_input_tested_here():
    """The frozen contract must prohibit exactly what this file exercises."""
    from pathlib import Path

    import yaml

    path = (Path(__file__).resolve().parents[2] / "configs" / "version_c" / "llm"
            / "c3_selection_contract.yaml")
    listed = " ".join(yaml.safe_load(path.read_text(encoding="utf-8"))
                      ["forbidden_selector_inputs"]).lower()
    for token in ("siw", "target metadata", "target metrics", "detector score",
                  "synthetic quality", "quality-gate", "manual ranking",
                  "provider response order", "filesystem order", "solver traversal"):
        assert token in listed, f"the contract does not forbid {token!r}"


def test_this_module_invents_its_forbidden_values_rather_than_reading_them():
    """The forbidden values above are literals, not measurements.

    Checked on the import graph rather than on the source text: a source scan
    would match this test's own token list. What must be true is that nothing
    here imports a dataset, target-evaluation or label-resolving module.
    """
    import ast
    import inspect
    from pathlib import Path

    tree = ast.parse(Path(inspect.getfile(taint)).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    for module in sorted(imported):
        lowered = module.lower()
        for banned in ("siw", "target_eval", "target_evaluation", "labels", "dataset",
                       "prism_fas.data", "prism_fas.eval"):
            assert banned not in lowered, f"this module imports {module!r}"

    # Every forbidden value is a plain literal defined in this file.
    assert all(isinstance(value, (str, int, float))
               for value in FORBIDDEN_METADATA.values())
    assert json.dumps(FORBIDDEN_METADATA, sort_keys=True)  # serializable literals
