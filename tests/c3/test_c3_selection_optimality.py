"""prism_c3_selection_v1 proved optimal, not merely self-consistent.

`test_c3_selection.py` checks the selector's contract surface. This file checks
the harder claim §7.8.3 actually makes: that the returned bank IS the exact
lexicographic optimum and, among the sets tied on stages 1-4, the
lexicographically smallest 256-SHA set.

The proof technique is exhaustive enumeration on instances small enough to
enumerate. Every C(n, k) subset is scored with the shipped exact-integer
functions, the lexicographic optimum is found by brute force, and the selector's
answer must equal it — set for set, not just objective for objective. A MILP that
returned any other tied optimum would fail here.

One test then runs the REAL budget (320 eligible -> 256 selected under the frozen
quota table) to show the contract is satisfiable at scale and that nothing
truncates or pads. It takes about a minute.

Everything is synthetic and offline. No provider, no network, no dataset, no
target information. The fixtures are not C3 scientific candidates and no
selected bank here is a scientific artifact.
"""
from __future__ import annotations

import itertools
import json

import pytest

from prism_fas.recipes import selection as sel
from prism_fas.recipes.canonical import recipe_hash
from prism_fas.recipes.selection import hard_violations, select
from prism_fas.recipes.validate import validate_payload

from c3_fixtures import make_payload, make_pool, make_recipe

#: Quotas that leave stage 1 trivially satisfiable so stages 2-5 do the deciding.
#: A test-harness device only; the shipped table is never modified.
TINY_QUOTAS: dict[str, dict[str, int | None]] = {
    "medium":       {"hard_min": 0, "hard_max": 99, "preferred_min": None},
    "geometry":     {"hard_min": 0, "hard_max": 99, "preferred_min": None},
    "illumination": {"hard_min": 0, "hard_max": 99, "preferred_min": None},
    "artifact":     {"hard_min": 0, "hard_max": 99, "preferred_min": 3},
    "region":       {"hard_min": 0, "hard_max": 99, "preferred_min": 2},
}

#: (tag, pool size, bank size). Small enough that C(n, k) is enumerable.
TINY_INSTANCES = [("bf0", 8, 3), ("bf1", 9, 4), ("bf2", 10, 4), ("bf3", 11, 5)]


def _with_quotas(quotas):
    """Context manager swapping the module quota table and restoring it."""
    import contextlib

    @contextlib.contextmanager
    def _swap():
        original = dict(sel.QUOTAS)
        sel.QUOTAS.update(quotas)
        try:
            yield
        finally:
            sel.QUOTAS.clear()
            sel.QUOTAS.update(original)

    return _swap()


def _brute_force(pool, bank_size, ontology):
    """Exhaustive lexicographic optimum and the smallest tied SHA set.

    Returns (objective_triple, smallest_sorted_sha_list, number_of_tied_sets).
    """
    shas = {index: recipe_hash(recipe) for index, recipe in enumerate(pool)}
    best_key = None
    tied: list[list[str]] = []
    for combo in itertools.combinations(range(len(pool)), bank_size):
        recipes = [pool[index] for index in combo]
        counts = sel.counts_for(recipes, ontology)
        if hard_violations(counts):
            continue
        key = (sel.s_pref(counts), sel.s_single(counts, bank_size), sel.s_multi(counts))
        tag = sorted(shas[index] for index in combo)
        if best_key is None or key < best_key:
            best_key, tied = key, [tag]
        elif key == best_key:
            tied.append(tag)
    if best_key is None:
        return None, None, 0
    return best_key, min(tied), len(tied)


# ------------------------------------------------------ exhaustive optimality
@pytest.mark.parametrize("tag, size, bank", TINY_INSTANCES)
def test_the_selector_returns_the_exact_lexicographic_optimum(tag, size, bank, ontology):
    """Stages 2, 3 and 4: the objective triple equals the brute-force optimum."""
    pool = make_pool(ontology, size, tag=tag)
    with _with_quotas(TINY_QUOTAS):
        result = select(pool, ontology, bank_size=bank, enforce_minimum_pool=False)
        expected, _, _ = _brute_force(pool, bank, ontology)
    assert expected is not None, "the instance must be feasible for the test to mean anything"
    assert (result.s_pref, result.s_single, result.s_multi) == expected


@pytest.mark.parametrize("tag, size, bank", TINY_INSTANCES)
def test_stage_5_returns_the_lexicographically_smallest_tied_set(tag, size, bank, ontology):
    """Stage 5, checked against every tied subset rather than asserted."""
    pool = make_pool(ontology, size, tag=tag)
    with _with_quotas(TINY_QUOTAS):
        result = select(pool, ontology, bank_size=bank, enforce_minimum_pool=False)
        _, smallest, tie_count = _brute_force(pool, bank, ontology)
    assert sorted(result.selected_shas) == smallest, (
        f"{tie_count} subsets tie on stages 1-4; the selector did not return the "
        "lexicographically smallest one")


def _homogeneous_pool(ontology, size, tag="tie"):
    """Candidates identical on every quota axis, distinct in canonical SHA.

    Coverage counts cannot distinguish these, so EVERY equal-sized subset scores
    identically on stages 1-4 and stage 5 alone decides. The recipes still differ
    — seed, yaw, strengths, capture values — so their canonical SHAs differ and
    there is a well-defined lexicographically smallest set.
    """
    medium = ontology.media[0]
    geometry = ontology.geometry_shapes[0]
    illumination = ontology.illumination[0]
    artifacts = [ontology.artifacts_for_medium(medium)[0]]
    regions = [ontology.regions_for_geometry(geometry)[0]]
    return [make_recipe(ontology, index, tag=tag, medium=medium, geometry=geometry,
                        illumination=illumination, artifacts=artifacts, regions=regions)
            for index in range(size)]


def test_stage_5_decides_alone_when_every_subset_ties(ontology):
    """The pure stage-5 case: all C(n, k) subsets tie, so the answer is exactly
    the k smallest canonical SHAs and nothing else."""
    size, bank = 10, 4
    pool = _homogeneous_pool(ontology, size)
    all_shas = sorted(recipe_hash(recipe) for recipe in pool)
    assert len(set(all_shas)) == size, "the fixtures must be distinct recipes"

    with _with_quotas(TINY_QUOTAS):
        result = select(pool, ontology, bank_size=bank, enforce_minimum_pool=False)
        expected_key, smallest, tie_count = _brute_force(pool, bank, ontology)

    assert tie_count == 210, f"every 4-subset of 10 should tie, got {tie_count}"
    assert sorted(result.selected_shas) == all_shas[:bank]
    assert sorted(result.selected_shas) == smallest
    assert (result.s_pref, result.s_single, result.s_multi) == expected_key


def test_the_tie_case_is_immune_to_input_order(ontology):
    """Same degenerate instance, permuted input: stage 5 still gives the k
    smallest SHAs. If input order leaked in, this is where it would show."""
    import random

    size, bank = 10, 4
    pool = _homogeneous_pool(ontology, size)
    expected = sorted(recipe_hash(recipe) for recipe in pool)[:bank]
    with _with_quotas(TINY_QUOTAS):
        for seed in (1, 2, 3, 4, 5):
            shuffled = list(pool)
            random.Random(seed).shuffle(shuffled)
            result = select(shuffled, ontology, bank_size=bank, enforce_minimum_pool=False)
            assert sorted(result.selected_shas) == expected, f"seed {seed} moved the bank"


# ------------------------------------------------------- solver independence
def test_solver_traversal_order_cannot_decide_the_bank(ontology):
    """Stages 2-4 contribute their objective VALUE, never their chosen subset.

    The real `_solve` is wrapped so that every optimizing call returns a mangled
    candidate list alongside the true objective value. If the selector ever used
    a solver-chosen subset, the bank would move. It must not.
    """
    tag, size, bank = TINY_INSTANCES[1]
    pool = make_pool(ontology, size, tag=tag)
    real_solve = sel._solve

    def scrambling_solve(model, **kwargs):
        feasible, value, picked = real_solve(model, **kwargs)
        if picked is not None and not kwargs.get("feasibility_only"):
            picked = list(reversed(picked))
        return feasible, value, picked

    with _with_quotas(TINY_QUOTAS):
        baseline = select(pool, ontology, bank_size=bank, enforce_minimum_pool=False)
        sel._solve = scrambling_solve
        try:
            scrambled = select(pool, ontology, bank_size=bank, enforce_minimum_pool=False)
        finally:
            sel._solve = real_solve

    assert scrambled.selected_shas == baseline.selected_shas
    assert scrambled.selected_set_identity == baseline.selected_set_identity


# ---------------------------------------------------- filesystem-order safety
def test_filesystem_listing_order_does_not_change_the_bank(ontology, tmp_path):
    """Candidates loaded from disk in any directory order select the same bank."""
    tag, size, bank = TINY_INSTANCES[2]
    for index in range(size):
        payload = make_payload(ontology, index, tag=tag)
        payload["recipe_id"] = f"R-{index:06d}"
        # Names deliberately do NOT sort in construction order.
        (tmp_path / f"cand_{(size - index) * 7 % size:02d}_{index}.json").write_text(
            json.dumps(payload), encoding="utf-8")

    def load(paths):
        pool = []
        for path in paths:
            recipe, issues = validate_payload(json.loads(path.read_text(encoding="utf-8")),
                                              ontology, canonicalize=False)
            assert recipe is not None and not issues
            pool.append(recipe)
        return pool

    listed = list(tmp_path.iterdir())
    orders = [sorted(listed), sorted(listed, reverse=True),
              sorted(listed, key=lambda p: p.stat().st_size), listed]

    banks = []
    with _with_quotas(TINY_QUOTAS):
        for paths in orders:
            result = select(load(paths), ontology, bank_size=bank, enforce_minimum_pool=False)
            banks.append(result.selected_shas)
    assert all(bank_shas == banks[0] for bank_shas in banks), (
        "directory listing order changed the selected bank")


# ------------------------------------------------------------- the real budget
@pytest.mark.slow
def test_the_real_320_to_256_budget_selects_exactly_256(ontology):
    """The frozen contract at full scale: 320 eligible -> exactly 256 selected.

    Runs the shipped quota table unmodified. Synthetic fixtures, so the resulting
    bank is a harness artifact and carries no scientific meaning — what is being
    proven is that 256, the hard quotas and the exact arithmetic are jointly
    satisfiable and that nothing truncates or pads.
    """
    pool = make_pool(ontology, sel.MINIMUM_ELIGIBLE_POOL_PER_ARM, tag="full")
    assert len(pool) == 320

    result = select(pool, ontology, arm="FIXTURE")

    assert len(result.selected_shas) == sel.FINAL_BANK_SIZE_PER_ARM == 256
    assert len(set(result.selected_shas)) == 256
    assert len(result.rejected_shas) == 320 - 256
    assert len(result.selected_shas) + len(result.rejected_shas) == 320
    assert hard_violations(result.counts) == []

    # Hard quotas, restated against the shipped table rather than inferred.
    for value in result.counts["medium"].values():
        assert 32 <= value <= 80
    for axis in ("geometry", "illumination"):
        for value in result.counts[axis].values():
            assert 24 <= value <= 64
    for axis in ("artifact", "region"):
        for value in result.counts[axis].values():
            assert 8 <= value <= 128

    # The recomputed objective is what the result carries.
    counts = sel.counts_for(result.selected_recipes, ontology)
    assert counts == result.counts
    assert sel.s_pref(counts) == result.s_pref
    assert sel.s_single(counts, 256) == result.s_single
    assert sel.s_multi(counts) == result.s_multi


# --------------------------------------- the 12x32 generation shape, in C3 scope
@pytest.mark.parametrize("count", [31, 33])
def test_a_c3_batch_that_is_not_exactly_32_still_fails_closed(ontology, count):
    """§7.8: 12 logical requests x exactly 32 objects = 384 raw slots.

    C2B established this on the response side. C3 inherits it, so the rule is
    re-checked here against the C3 slot budget: 31 or 33 objects is a failure,
    never a 383- or 385-slot arm.
    """
    from prism_fas.llm.config import load_llm_config
    from prism_fas.llm.pipeline import RecipePlanner
    from prism_fas.llm.providers.mock import MockRecipeProvider
    from prism_fas.recipes.arm_schedules import SLOTS_PER_ARM

    from test_c3_schedules_and_identity import REPO

    config = load_llm_config(REPO / "configs" / "version_c" / "llm" / "c1_gemini_provider.yaml")
    planner = RecipePlanner(provider=MockRecipeProvider(), config=config, ontology=ontology,
                            sleep=lambda _seconds: None)
    envelope = json.dumps({"recipes": [make_payload(ontology, index, tag="batch")
                                       for index in range(count)]})
    validation = planner.validate_response(envelope, slot_id="C3_BATCH_000",
                                           recipes_requested=32)
    assert validation.response_issues, f"{count} objects were not rejected"
    assert any("expected exactly 32" in issue["reason"]
               for issue in validation.response_issues)
    assert not validation.all_accepted
    assert 12 * 32 == SLOTS_PER_ARM == sel.RAW_CANDIDATE_SLOTS_PER_ARM == 384
