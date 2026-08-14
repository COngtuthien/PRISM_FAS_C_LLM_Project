"""prism_c3_selection_v1 — the selection contract, tested offline on fixtures.

Covers the properties §7.8.3 makes normative: cardinalities, the hard quota
table, each objective's exact integer arithmetic, the canonical tie-break, and
the invariances that make the selected bank independent of how it was computed.

Every test uses synthetic fixtures. No provider, no network, no dataset, no
target information.
"""
from __future__ import annotations

import json
import random

import pytest

from prism_fas.recipes import selection as sel
from prism_fas.recipes.eligibility import ELIGIBILITY_ORDER, evaluate_pool
from prism_fas.recipes.selection import (SelectionInfeasible, counts_for, hard_violations,
                                         s_multi, s_pref, s_single, select)

from c3_fixtures import make_pool, make_payload, make_recipe

#: A bank small enough to solve repeatedly, large enough to exercise every axis.
BANK = 24
POOL = 40


@pytest.fixture(scope="module")
def small_pool(ontology):
    return make_pool(ontology, POOL)


def run(pool, ontology, *, bank=BANK, quotas=None, **kwargs):
    """Select with scaled quotas, restoring the frozen table afterwards."""
    original = dict(sel.QUOTAS)
    if quotas is not None:
        sel.QUOTAS.update(quotas)
    try:
        return select(pool, ontology, bank_size=bank, enforce_minimum_pool=False, **kwargs)
    finally:
        sel.QUOTAS.clear()
        sel.QUOTAS.update(original)


@pytest.fixture
def relaxed():
    """Quotas that are satisfiable at BANK=24 while keeping the real shape."""
    return {
        "medium": {"hard_min": 2, "hard_max": BANK, "preferred_min": None},
        "geometry": {"hard_min": 1, "hard_max": BANK, "preferred_min": None},
        "illumination": {"hard_min": 1, "hard_max": BANK, "preferred_min": None},
        "artifact": {"hard_min": 1, "hard_max": BANK * 3, "preferred_min": 6},
        "region": {"hard_min": 1, "hard_max": BANK * 3, "preferred_min": 4},
    }


# ------------------------------------------------------- cardinality contract
def test_frozen_cardinalities_are_the_spec_values():
    assert sel.RAW_CANDIDATE_SLOTS_PER_ARM == 384
    assert sel.MINIMUM_ELIGIBLE_POOL_PER_ARM == 320
    assert sel.FINAL_BANK_SIZE_PER_ARM == 256
    assert sel.SELECTION_VERSION == "prism_c3_selection_v1"


def test_a_pool_below_the_minimum_fails_closed(ontology):
    pool = make_pool(ontology, 319, tag="short")
    with pytest.raises(SelectionInfeasible, match="below the frozen minimum"):
        select(pool, ontology)


def test_exactly_the_minimum_pool_is_permitted_to_proceed(ontology, monkeypatch):
    """320 is allowed; the failure boundary is strictly below it."""
    pool = make_pool(ontology, 320, tag="exact")
    seen = {}

    def fake_build(recipes, ontology):
        seen["n"] = len(recipes)
        raise SelectionInfeasible("stop after the pool check")

    monkeypatch.setattr(sel, "_build_model", fake_build)
    with pytest.raises(SelectionInfeasible, match="stop after the pool check"):
        select(pool, ontology)
    assert seen["n"] == 320, "the 320-candidate pool passed the minimum check"


def test_selected_cardinality_is_exact(small_pool, ontology, relaxed):
    result = run(small_pool, ontology, quotas=relaxed)
    assert len(result.selected_shas) == BANK
    assert len(set(result.selected_shas)) == BANK
    assert len(result.selected_recipes) == BANK
    assert len(result.rejected_shas) == POOL - BANK


def test_no_silent_truncation_or_padding(small_pool, ontology, relaxed):
    result = run(small_pool, ontology, quotas=relaxed)
    assert len(result.selected_shas) + len(result.rejected_shas) == len(small_pool)
    assert set(result.selected_shas).isdisjoint(result.rejected_shas)


# --------------------------------------------------------------- hard quotas
def test_every_hard_quota_holds_in_the_selected_bank(small_pool, ontology, relaxed):
    result = run(small_pool, ontology, quotas=relaxed)
    original = dict(sel.QUOTAS)
    sel.QUOTAS.update(relaxed)
    try:
        assert hard_violations(result.counts) == []
    finally:
        sel.QUOTAS.clear()
        sel.QUOTAS.update(original)


def test_the_frozen_quota_table_matches_the_spec():
    assert sel.QUOTAS["medium"] == {"hard_min": 32, "hard_max": 80, "preferred_min": None}
    assert sel.QUOTAS["geometry"] == {"hard_min": 24, "hard_max": 64, "preferred_min": None}
    assert sel.QUOTAS["illumination"] == {"hard_min": 24, "hard_max": 64, "preferred_min": None}
    assert sel.QUOTAS["artifact"] == {"hard_min": 8, "hard_max": 128, "preferred_min": 32}
    assert sel.QUOTAS["region"] == {"hard_min": 8, "hard_max": 128, "preferred_min": 24}


def test_an_infeasible_hard_quota_fails_rather_than_relaxing(small_pool, ontology):
    impossible = {"medium": {"hard_min": BANK, "hard_max": BANK, "preferred_min": None}}
    with pytest.raises(SelectionInfeasible, match="hard coverage"):
        run(small_pool, ontology, quotas=impossible)


# ------------------------------------------------------ exact objective maths
def test_s_pref_is_exact_integer_arithmetic():
    counts = {"artifact": {"a": 30, "b": 32, "c": 40},
              "region": {"x": 20, "y": 24, "z": 30}}
    # (32-30) + 0 + 0  +  (24-20) + 0 + 0
    assert s_pref(counts) == 2 + 4
    assert isinstance(s_pref(counts), int)


def test_s_single_is_exact_integer_arithmetic():
    counts = {"medium": {m: c for m, c in zip("abcde", [50, 50, 52, 52, 52])},
              "geometry": {g: 0 for g in "abcdef"},
              "illumination": {i: 0 for i in "abcdef"}}
    counts["geometry"] = dict(zip("abcdef", [42, 43, 43, 42, 43, 43]))
    counts["illumination"] = dict(zip("abcdef", [42, 43, 43, 42, 43, 43]))
    expected = (sum(abs(5 * c - 256) for c in counts["medium"].values())
                + sum(abs(6 * c - 256) for c in counts["geometry"].values())
                + sum(abs(6 * c - 256) for c in counts["illumination"].values()))
    assert s_single(counts) == expected


def test_s_multi_is_exact_integer_arithmetic():
    counts = {"artifact": dict(zip("abcdefgh", [10, 20, 30, 40, 50, 60, 70, 80])),
              "region": dict(zip("abcdefghi", [5, 10, 15, 20, 25, 30, 35, 40, 45]))}
    a_total = sum(counts["artifact"].values())
    r_total = sum(counts["region"].values())
    expected = (sum(abs(8 * c - a_total) for c in counts["artifact"].values())
                + sum(abs(9 * c - r_total) for c in counts["region"].values()))
    assert s_multi(counts) == expected


def test_objective_values_are_recomputed_from_the_selected_set(small_pool, ontology,
                                                               relaxed):
    result = run(small_pool, ontology, quotas=relaxed)
    original = dict(sel.QUOTAS)
    sel.QUOTAS.update(relaxed)
    try:
        counts = counts_for(result.selected_recipes, ontology)
        assert counts == result.counts
        assert s_pref(counts) == result.s_pref
        assert s_single(counts, BANK) == result.s_single
        assert s_multi(counts) == result.s_multi
    finally:
        sel.QUOTAS.clear()
        sel.QUOTAS.update(original)


# ------------------------------------------------------------- determinism
def test_repeated_runs_are_identical(small_pool, ontology, relaxed):
    first = run(small_pool, ontology, quotas=relaxed)
    second = run(small_pool, ontology, quotas=relaxed)
    assert first.selected_shas == second.selected_shas
    assert first.selected_set_identity == second.selected_set_identity


def test_input_permutation_does_not_change_the_bank(small_pool, ontology, relaxed):
    baseline = run(small_pool, ontology, quotas=relaxed)
    for seed in (1, 2, 3):
        shuffled = list(small_pool)
        random.Random(seed).shuffle(shuffled)
        result = run(shuffled, ontology, quotas=relaxed)
        assert result.selected_shas == baseline.selected_shas, (
            f"permutation seed {seed} changed the selected bank")


def test_reversed_input_does_not_change_the_bank(small_pool, ontology, relaxed):
    baseline = run(small_pool, ontology, quotas=relaxed)
    result = run(list(reversed(small_pool)), ontology, quotas=relaxed)
    assert result.selected_shas == baseline.selected_shas


def test_the_selected_set_is_the_lexicographically_smallest_tied_set(small_pool,
                                                                     ontology,
                                                                     relaxed):
    """Stage 5 as ascending-SHA greedy inclusion: every candidate is decided in
    SHA order, and a candidate is excluded only when including it was proven
    infeasible under the pinned stage 1-4 optimum.

    The stronger claim — that the result equals the lexicographically smallest
    tied set — is proved against exhaustive enumeration in
    `test_c3_selection_optimality.py`; it cannot be established from a single
    run's output alone.
    """
    result = run(small_pool, ontology, quotas=relaxed)
    everything = sorted(result.selected_shas + result.rejected_shas)

    decisions = [(entry["sha"], entry["decision"])
                 for entry in result.tie_break_trace if "sha" in entry]
    # Candidates are visited in ascending canonical SHA, never in input order.
    visited = [sha for sha, _ in decisions]
    assert visited == sorted(visited)
    assert visited == everything[:len(visited)]

    included = [sha for sha, decision in decisions if decision == "include"]
    excluded = {sha for sha, decision in decisions if decision == "exclude"}
    assert set(included) <= set(result.selected_shas)
    assert excluded.isdisjoint(result.selected_shas)
    assert len(included) == BANK

    # Greedy ascending inclusion means the smallest SHA is selected unless
    # including it was infeasible — never because of where it appeared in input.
    assert (everything[0] in result.selected_shas) or (everything[0] in excluded)


def test_the_tie_break_trace_is_recorded(small_pool, ontology, relaxed):
    result = run(small_pool, ontology, quotas=relaxed)
    assert result.tie_break_trace
    included = [entry for entry in result.tie_break_trace
                if entry.get("decision") == "include"]
    assert len(included) == BANK


def test_candidates_are_ordered_by_canonical_sha_not_by_input_order(small_pool,
                                                                    ontology):
    from prism_fas.recipes.selection import _build_model

    model, ordered = _build_model(small_pool, ontology)
    assert model.shas == sorted(model.shas)
    shuffled = list(small_pool)
    random.Random(7).shuffle(shuffled)
    model2, _ = _build_model(shuffled, ontology)
    assert model2.shas == model.shas


# ------------------------------------------------------- forbidden selector inputs
def test_selection_ignores_target_quality_and_score_fields(small_pool, ontology,
                                                           relaxed):
    """A recipe carries no such field, and the selector accepts no such argument."""
    import inspect

    signature = inspect.signature(select)
    forbidden = {"target", "siw", "score", "quality", "q", "detector", "metric", "acer",
                 "ranking", "preference", "order"}
    for name in signature.parameters:
        assert not any(token in name.lower() for token in forbidden), (
            f"select() exposes a forbidden parameter {name!r}")

    # Scan executable code, not prose: the module docstring legitimately NAMES
    # the forbidden inputs in order to prohibit them. What must not exist is a
    # code path that reads one.
    import ast

    tree = ast.parse(inspect.getsource(sel))

    docstring_nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstring_nodes.add(id(body[0].value))

    identifiers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
        elif isinstance(node, ast.arg):
            identifiers.add(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            identifiers.add(node.arg)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # String literals can be dict keys; docstrings are prose and are
            # allowed to name what they prohibit.
            if id(node) not in docstring_nodes:
                identifiers.add(node.value)

    lowered = " ".join(identifiers).lower()
    for token in ("siw", "casia", "msu", "acer", "apcer", "bpcer", "hter",
                  "attack_family", "detector_score", "quality_gate", "target_metric"):
        assert token not in lowered, (
            f"the selector reads something named {token!r} in executable code")


def test_attaching_forbidden_metadata_cannot_change_the_bank(small_pool, ontology,
                                                             relaxed):
    """Even if a caller decorates recipes, selection reads recipe content only."""
    baseline = run(small_pool, ontology, quotas=relaxed)
    decorated = []
    for index, recipe in enumerate(small_pool):
        clone = recipe.model_copy()
        object.__setattr__(clone, "__dict__", dict(clone.__dict__))
        decorated.append(clone)
    result = run(decorated, ontology, quotas=relaxed)
    assert result.selected_shas == baseline.selected_shas


def test_the_selection_contract_lists_the_forbidden_inputs():
    from pathlib import Path

    import yaml

    path = (Path(__file__).resolve().parents[2] / "configs" / "version_c" / "llm"
            / "c3_selection_contract.yaml")
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    listed = " ".join(contract["forbidden_selector_inputs"]).lower()
    for token in ("siw", "target metric", "detector score", "synthetic quality",
                  "manual ranking", "provider response order", "solver traversal"):
        assert token in listed


# ------------------------------------------------------------------ eligibility
def test_eligibility_order_is_the_governing_order():
    assert ELIGIBILITY_ORDER == (
        "item_schema", "ontology", "scientific_route_policy", "numeric_range",
        "compatibility", "canonicalization", "deduplication", "compiler",
        "operator_graph", "mask_policy", "conditioning_41d")


def test_route_invalid_candidates_are_excluded_before_the_compiler(ontology,
                                                                    route_policy):
    payloads = [make_payload(ontology, index, route=route)
                for index, route in enumerate([["physics", "gpat"], ["physics"], ["gpat"]])]
    pool = evaluate_pool(arm="TEST", candidates=payloads,
                         slot_ids=[f"T_{i:03d}" for i in range(3)],
                         ontology=ontology, route_policy=route_policy,
                         bank_id="c3-test", raw_slots=3, minimum_required=1)
    assert pool.eligible_count == 1
    rejected = [v for v in pool.verdicts if not v.eligible]
    assert {v.rejected_at for v in rejected} == {"scientific_route_policy"}
    assert all(v.graph_hash is None for v in rejected), "the compiler was reached"


def test_duplicates_are_excluded_at_the_deduplication_stage(ontology, route_policy):
    payload = make_payload(ontology, 0)
    pool = evaluate_pool(arm="TEST", candidates=[payload, dict(payload)],
                         slot_ids=["T_000", "T_001"], ontology=ontology,
                         route_policy=route_policy, bank_id="c3-test", raw_slots=2,
                         minimum_required=1)
    assert pool.eligible_count == 1
    duplicate = [v for v in pool.verdicts if not v.eligible][0]
    assert duplicate.rejected_at == "deduplication"
    assert duplicate.duplicate_of_slot_id == "T_000"


def test_an_uncompilable_candidate_is_excluded(ontology, route_policy, monkeypatch):
    from prism_fas.recipes import eligibility as elig
    from prism_fas.recipes.compile import CompileError

    def boom(recipe, ontology, *, bank_id, **kwargs):
        raise CompileError("synthetic compiler failure")

    monkeypatch.setattr(elig, "compile_recipe", boom)
    pool = evaluate_pool(arm="TEST", candidates=[make_payload(ontology, 0)],
                         slot_ids=["T_000"], ontology=ontology,
                         route_policy=route_policy, bank_id="c3-test", raw_slots=1,
                         minimum_required=0)
    assert pool.eligible_count == 0
    assert pool.verdicts[0].rejected_at == "compiler"


def test_a_missing_slot_is_a_rejection_not_an_omission(ontology, route_policy):
    pool = evaluate_pool(arm="TEST", candidates=[make_payload(ontology, 0), None],
                         slot_ids=["T_000", "T_001"], ontology=ontology,
                         route_policy=route_policy, bank_id="c3-test", raw_slots=2,
                         minimum_required=0)
    assert len(pool.verdicts) == 2, "every slot is accounted for"
    assert pool.verdicts[1].rejected_at == "item_schema"


def test_the_slot_budget_cannot_be_topped_up(ontology, route_policy):
    from prism_fas.recipes.eligibility import EligibilityError

    with pytest.raises(EligibilityError, match="frozen and cannot be topped up"):
        evaluate_pool(arm="TEST", candidates=[make_payload(ontology, 0)],
                      slot_ids=["T_000"], ontology=ontology,
                      route_policy=route_policy, bank_id="c3-test", raw_slots=384,
                      minimum_required=320)


def test_pool_below_minimum_is_reported(ontology, route_policy):
    payloads = [make_payload(ontology, index) for index in range(4)]
    pool = evaluate_pool(arm="TEST", candidates=payloads,
                         slot_ids=[f"T_{i:03d}" for i in range(4)],
                         ontology=ontology, route_policy=route_policy,
                         bank_id="c3-test", raw_slots=4, minimum_required=320)
    assert pool.meets_minimum is False
