"""C6_MATCHED_BANK_SELECTOR_V1 — the frozen deterministic matched-bank selector.

§11.3 names four balancing dimensions and no algorithm. This file pins the
algorithm that was frozen to fill that gap, and above all pins the two
properties that make the three-arm comparison fair:

* **quality does not select.** Once a candidate passes the common profile the
  gate is binary. `q`, the gate metrics and the margins are invisible to the
  ordering, so a stronger arm cannot assemble a better-tuned training set than a
  weaker one — which would be the Version-B confound in a new place.
* **the target distribution comes from the frozen PLAN.** Acceptance is a
  treatment outcome; letting it define the desired source exposure would let an
  arm redefine "balanced" by failing more often on one dataset.

Everything here runs in memory. No candidate is rendered and no payload is read.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.synthesis import c6_matched_bank as mb  # noqa: E402
from prism_fas.synthesis import c6_scientific as science  # noqa: E402
from prism_fas.synthesis.c5_source_pair_plan import ARMS, GPAT, PHYSICS  # noqa: E402

DOMAINS = ("casia_fasd", "msu_mfsd")
SELECTOR_SOURCE = (REPO / "src" / "prism_fas" / "synthesis" / "c6_matched_bank.py"
                   ).read_text(encoding="utf-8")


def _function_source(name: str, source: str = SELECTOR_SOURCE) -> str:
    tree = ast.parse(source)
    node = next(item for item in ast.walk(tree)
                if isinstance(item, ast.FunctionDef) and item.name == name)
    return ast.get_source_segment(source, node) or ""


# --- a plan and an accepted pool ---------------------------------------------

def _plan(arm: str, *, per_route: int = 1024,
          domain_split: tuple[int, int] = (512, 512)) -> dict[str, Any]:
    """A frozen arm plan: `per_route` slots on each route, split across domains."""
    rows = []
    position = 0
    for route in (PHYSICS, GPAT):
        for index in range(per_route):
            domain = DOMAINS[0] if index < domain_split[0] else DOMAINS[1]
            rows.append({
                "candidate_id": f"c5syn_{arm.lower()}_{position:05d}",
                "arm": arm, "route": route, "position": position,
                "recipe_id": f"r{index % 256}", "recipe_ordinal": index % 256,
                "live_dataset": domain,
                "live_target_sample_id": f"live_{domain}_{index % 320:04d}",
            })
            position += 1
    return {"arm": arm, "candidates": rows, "planned_candidates": len(rows)}


def _plans(**kwargs) -> dict[str, dict[str, Any]]:
    return {arm: _plan(arm, **kwargs) for arm in ARMS}


def _accepted(plans: dict[str, dict[str, Any]], *, keep: int | None = None,
              q_seed: float = 0.5,
              drop: dict[tuple[str, str, str], int] | None = None
              ) -> dict[str, list[mb.SelectableCandidate]]:
    """Turn planned rows into accepted candidates, optionally dropping some.

    `drop` removes N candidates from an (arm, route, domain) cell, which is how
    a capacity shortfall is expressed.
    """
    drop = dict(drop or {})
    accepted: dict[str, list[mb.SelectableCandidate]] = {}
    for arm, plan in plans.items():
        rows: list[mb.SelectableCandidate] = []
        budget = {key: value for key, value in drop.items() if key[0] == arm}
        for index, row in enumerate(plan["candidates"]):
            cell = (arm, row["route"], row["live_dataset"])
            if budget.get(cell, 0) > 0:
                budget[cell] -= 1
                continue
            rows.append(mb.SelectableCandidate(
                candidate_id=row["candidate_id"], arm=arm, route=row["route"],
                source_domain=row["live_dataset"], recipe_id=row["recipe_id"],
                recipe_ordinal=row["recipe_ordinal"],
                live_target_sample_id=row["live_target_sample_id"],
                base_position=row["position"],
                q=round(q_seed + (index % 17) / 100.0, 4)))
        accepted[arm] = rows[:keep] if keep is not None else rows
    return accepted


@pytest.fixture(scope="module")
def built() -> dict[str, Any]:
    plans = _plans()
    return mb.build_matched_banks(plans, _accepted(plans))


# --- 1, 2. the hard cardinality and the common quota -------------------------

def test_every_arm_holds_exactly_512_physics_and_512_gpat(built) -> None:
    assert built["matched"] is True
    for arm in ARMS:
        bank = built["banks"][arm]
        assert bank["size"] == mb.FINAL_BANK_PER_ARM == 1024
        assert bank["by_route"][PHYSICS] == mb.PER_ROUTE == 512
        assert bank["by_route"][GPAT] == 512


def test_the_source_domain_quota_is_identical_across_the_three_arms(built) -> None:
    """One quota vector per route. That is what "matched" means."""
    for route in mb.ROUTES:
        quota = built["route_quotas"][route]["quota"]
        assert built["route_quotas"][route]["identical_across_arms"] is True
        for arm in ARMS:
            exposure = built["banks"][arm]["exposure"][route]["by_source_domain"]
            assert exposure == quota, f"{arm} {route}"


# --- 3-5. the planned target and largest remainder ---------------------------

def test_the_ideal_target_comes_from_the_planned_schedule_not_acceptance() -> None:
    """Acceptance is a treatment outcome and may not redefine desired exposure."""
    plans = _plans(domain_split=(600, 424))
    # Acceptance skewed hard the other way.
    accepted = _accepted(plans, drop={(arm, PHYSICS, DOMAINS[0]): 400 for arm in ARMS})

    quotas = mb.route_quotas(plans, accepted)
    planned = mb.planned_domain_counts(plans["RND"], PHYSICS)

    assert planned == {DOMAINS[0]: 600, DOMAINS[1]: 424}
    assert quotas[PHYSICS].ideal[DOMAINS[0]] == pytest.approx(512 * 600 / 1024)
    source = _function_source("ideal_domain_share")
    assert "accepted" not in source


def test_largest_remainder_quotas_are_deterministic_and_sum_to_the_total() -> None:
    ideal = {"a": 170.4, "b": 170.3, "c": 171.3}

    quota = mb.largest_remainder_quota(ideal, total=512)

    # floors 170 + 170 + 171 = 511; the single remaining slot goes to the
    # largest fractional part, which is `a` at 0.4.
    assert sum(quota.values()) == 512
    assert quota == {"a": 171, "b": 170, "c": 171}
    assert mb.largest_remainder_quota(ideal, total=512) == quota


def test_a_fractional_tie_is_broken_by_canonical_domain_id() -> None:
    ideal = {"zulu": 170.5, "alpha": 170.5, "mike": 171.0}

    quota = mb.largest_remainder_quota(ideal, total=512)

    assert quota["alpha"] == 171 and quota["zulu"] == 170, (
        "the ascending domain id wins the tie")
    assert sum(quota.values()) == 512


def test_permuting_the_domain_order_cannot_change_the_quotas() -> None:
    forward = {"casia_fasd": 260.0, "msu_mfsd": 252.0}
    reversed_order = {"msu_mfsd": 252.0, "casia_fasd": 260.0}

    assert mb.largest_remainder_quota(forward) == mb.largest_remainder_quota(reversed_order)


# --- 6-11. common capacity, clipping and redistribution ----------------------

def test_common_capacity_is_the_minimum_across_arms() -> None:
    plans = _plans()
    accepted = _accepted(plans, drop={("DET", PHYSICS, DOMAINS[0]): 100,
                                      ("LLM", PHYSICS, DOMAINS[0]): 40})

    capacity = mb.common_capacity(accepted, PHYSICS, DOMAINS)

    assert capacity[DOMAINS[0]] == 512 - 100, "the weakest arm sets the capacity"
    assert capacity[DOMAINS[1]] == 512


def test_a_quota_over_capacity_is_clipped_and_the_deficit_refilled() -> None:
    planned = {DOMAINS[0]: 512, DOMAINS[1]: 512}
    capacity = {DOMAINS[0]: 200, DOMAINS[1]: 500}

    quota = mb.resolve_route_quota(PHYSICS, planned, capacity)

    assert quota.initial_quota == {DOMAINS[0]: 256, DOMAINS[1]: 256}
    assert quota.clipped == {DOMAINS[0]: 56}
    assert quota.quota == {DOMAINS[0]: 200, DOMAINS[1]: 312}
    assert sum(quota.quota.values()) == 512
    assert quota.feasible is True


def test_redistribution_fills_the_domain_furthest_below_its_planned_target() -> None:
    planned = {"a": 512, "b": 256, "c": 256}
    capacity = {"a": 10, "b": 400, "c": 400}

    quota = mb.resolve_route_quota(PHYSICS, planned, capacity)
    steps = quota.redistribution

    assert quota.quota["a"] == 10
    assert sum(quota.quota.values()) == 512
    # `b` and `c` share an ideal of 128 each and start at 128; every refill goes
    # to whichever is currently furthest below, so they stay within one of each
    # other and the ascending id breaks the tie.
    assert abs(quota.quota["b"] - quota.quota["c"]) <= 1
    assert steps and all(step["domain"] in ("b", "c") for step in steps)


def test_a_redistribution_tie_uses_the_canonical_domain_id() -> None:
    planned = {"alpha": 256, "zulu": 256, "short": 512}
    capacity = {"alpha": 400, "zulu": 400, "short": 0}

    quota = mb.resolve_route_quota(PHYSICS, planned, capacity)
    first = quota.redistribution[0]

    assert quota.quota["short"] == 0
    assert first["domain"] == "alpha", "ascending id wins an exact tie"


def test_insufficient_common_capacity_makes_the_profile_infeasible() -> None:
    planned = {DOMAINS[0]: 512, DOMAINS[1]: 512}
    capacity = {DOMAINS[0]: 100, DOMAINS[1]: 100}

    quota = mb.resolve_route_quota(PHYSICS, planned, capacity)

    assert quota.feasible is False
    assert quota.shortfall == 512 - 200
    assert mb.matched_feasible({PHYSICS: quota, GPAT: quota}) is False


def test_no_per_arm_quota_relaxation_exists() -> None:
    """One vector for three arms; there is no per-arm quota to relax."""
    source = _function_source("build_matched_banks")

    assert "quotas[route].quota" in source
    assert "for arm in ARMS" in source
    # The quota is computed once, outside the per-arm loop.
    assert source.index("route_quotas(plans") < source.index("for arm in ARMS")


def test_an_arm_short_on_one_route_makes_the_profile_infeasible() -> None:
    plans = _plans()
    accepted = _accepted(plans, drop={("DET", PHYSICS, DOMAINS[0]): 300,
                                      ("DET", PHYSICS, DOMAINS[1]): 300})

    outcome = mb.build_matched_banks(plans, accepted)

    assert outcome["matched"] is False
    assert outcome["reason"] == "COMMON_SOURCE_DOMAIN_QUOTA_INFEASIBLE"
    assert outcome["banks"] == {}


# --- 12-16. the within-quota ordering ----------------------------------------

def _candidate(**kwargs) -> mb.SelectableCandidate:
    base = dict(candidate_id="c0", arm="RND", route=PHYSICS,
                source_domain=DOMAINS[0], recipe_id="r0", recipe_ordinal=0,
                live_target_sample_id="live_0", base_position=0, q=0.5)
    return mb.SelectableCandidate(**{**base, **kwargs})


def test_recipe_exposure_outranks_live_exposure() -> None:
    """Two candidates, one from an unused recipe on a busy live sample."""
    pool = [
        _candidate(candidate_id="busy_recipe", recipe_ordinal=0,
                   live_target_sample_id="live_fresh", base_position=1),
        _candidate(candidate_id="fresh_recipe", recipe_ordinal=1,
                   live_target_sample_id="live_busy", base_position=2),
        _candidate(candidate_id="seed", recipe_ordinal=0,
                   live_target_sample_id="live_busy", base_position=3),
    ]
    selected = mb.select_route_bank(pool, route=PHYSICS, quota={DOMAINS[0]: 3})

    # Step 1 takes a recipe-0 candidate; step 2 must switch to recipe 1 even
    # though its live sample is the busier one.
    assert selected[1]["recipe_ordinal"] == 1
    assert selected[1]["candidate_id"] == "fresh_recipe"


def test_live_exposure_outranks_the_tie_hash() -> None:
    """One recipe, three candidates, two live samples, two slots.

    Recipe exposure cannot separate them — every candidate is recipe 0 — so
    whichever candidate step 1 takes, step 2 must move to the other live sample
    rather than following the hash back to the one already used.
    """
    pool = [
        _candidate(candidate_id="a1", recipe_ordinal=0,
                   live_target_sample_id="live_a", base_position=1),
        _candidate(candidate_id="a2", recipe_ordinal=0,
                   live_target_sample_id="live_a", base_position=2),
        _candidate(candidate_id="b1", recipe_ordinal=0,
                   live_target_sample_id="live_b", base_position=3),
    ]
    selected = mb.select_route_bank(pool, route=PHYSICS, quota={DOMAINS[0]: 2})

    assert {row["live_target_sample_id"] for row in selected} == {"live_a", "live_b"}
    assert selected[1]["live_count_before"] == 0


def test_the_tie_hash_excludes_the_arm() -> None:
    """Arm-independent by construction: all three arms share the base schedule."""
    left = _candidate(arm="RND").tie_hash()
    right = _candidate(arm="LLM", candidate_id="different").tie_hash()

    assert left == right


@pytest.mark.parametrize("field,value", [("q", 0.99), ("q", None)])
def test_the_tie_hash_excludes_quality(field, value) -> None:
    assert _candidate(**{field: value}).tie_hash() == _candidate().tie_hash()


def test_the_tie_hash_material_is_explicit_and_frozen() -> None:
    import hashlib

    expected = hashlib.sha256(
        "PRISM_C6_MATCHED_BANK_SELECTOR_V1|physics|casia_fasd|7|live_7".encode("utf-8")
    ).hexdigest()

    assert mb.canonical_tie_hash(route="physics", source_domain="casia_fasd",
                                 base_position=7,
                                 live_target_sample_id="live_7") == expected
    assert mb.TIE_HASH_ENCODING == "utf-8" and mb.TIE_HASH_SEPARATOR == "|"

    source = _function_source("canonical_tie_hash")
    for forbidden in ("arm", "q", "metric", "margin", "timestamp", "path"):
        assert f"self.{forbidden}" not in source


def test_candidate_id_is_only_the_final_tie_break() -> None:
    """Same recipe, same live, same hash inputs — only the id can separate them."""
    pool = [_candidate(candidate_id="zzz", base_position=5),
            _candidate(candidate_id="aaa", base_position=5)]

    selected = mb.select_route_bank(pool, route=PHYSICS, quota={DOMAINS[0]: 1})

    assert selected[0]["candidate_id"] == "aaa"


# --- 17-20. determinism, and that quality cannot reach the ordering ----------

def test_changing_q_cannot_change_which_candidates_are_selected() -> None:
    plans = _plans()
    baseline = mb.build_matched_banks(plans, _accepted(plans, q_seed=0.5))
    reweighted = mb.build_matched_banks(plans, _accepted(plans, q_seed=0.9))

    for arm in ARMS:
        assert (baseline["banks"][arm]["selected_set_sha256"]
                == reweighted["banks"][arm]["selected_set_sha256"]), arm


def test_changing_gate_metric_values_cannot_change_the_selection() -> None:
    """The selector never receives a metric at all — the gate is binary."""
    import dataclasses

    fields = {field.name for field in dataclasses.fields(mb.SelectableCandidate)}

    for forbidden in ("face_detection_score", "identity_cosine", "landmark_nme",
                      "parsing_dice", "fingerprint_score", "gate_margin",
                      "failed_gates", "accepted"):
        assert forbidden not in fields, forbidden
    assert "q" in fields, "q is carried for §11.2 training weighting"

    # The ordering key itself: whatever it reads, `q` is not among it.
    ordering = next(
        node for node in ast.walk(ast.parse(_function_source("select_route_bank")))
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "min")
    key = ast.unparse(next(word.value for word in ordering.keywords
                           if word.arg == "key"))

    assert "recipe_count" in key and "live_count" in key
    assert "tie_hash" in key and "candidate_id" in key
    assert ".q" not in key, "quality may not enter the ordering key"


def test_repeated_execution_selects_the_identical_candidates() -> None:
    plans = _plans()
    first = mb.build_matched_banks(plans, _accepted(plans))
    second = mb.build_matched_banks(plans, _accepted(plans))

    for arm in ARMS:
        assert ([row["candidate_id"] for row in first["banks"][arm]["selected"]]
                == [row["candidate_id"] for row in second["banks"][arm]["selected"]])


def test_permuting_the_input_order_selects_the_identical_candidates() -> None:
    plans = _plans()
    forward = _accepted(plans)
    shuffled = {arm: list(reversed(rows)) for arm, rows in _accepted(plans).items()}

    assert (mb.build_matched_banks(plans, forward)["banks"]["RND"]["selected_set_sha256"]
            == mb.build_matched_banks(plans, shuffled)["banks"]["RND"]["selected_set_sha256"])


# --- 21-23. what the balance dimensions achieve ------------------------------

def test_recipe_exposure_is_two_per_route_when_capacity_permits(built) -> None:
    """256 recipes, 512 slots — the natural balanced target."""
    for arm in ARMS:
        for route in mb.ROUTES:
            exposure = built["banks"][arm]["exposure"][route]
            assert exposure["distinct_recipes"] == 256, f"{arm} {route}"
            assert exposure["recipe_exposure_min"] == 2
            assert exposure["recipe_exposure_max"] == 2


def test_a_short_recipe_is_filled_from_the_least_exposed_recipes() -> None:
    """No failure, no replacement, no manual deficit bookkeeping."""
    plans = _plans()
    accepted = _accepted(plans)
    # Recipe 0 keeps a single Physics candidate in every arm.
    for arm, rows in accepted.items():
        physics_zero = [row for row in rows
                        if row.route == PHYSICS and row.recipe_ordinal == 0]
        for row in physics_zero[1:]:
            rows.remove(row)

    outcome = mb.build_matched_banks(plans, accepted)
    exposure = outcome["banks"]["RND"]["exposure"][PHYSICS]

    assert outcome["matched"] is True, "a short recipe is not a failure"
    assert exposure["selected"] == 512
    assert exposure["recipe_exposure_min"] == 1
    assert exposure["recipe_exposure_max"] == 3, (
        "the missing exposure went to the least exposed recipes")


def test_base_live_concentration_is_minimized(built) -> None:
    for arm in ARMS:
        for route in mb.ROUTES:
            exposure = built["banks"][arm]["exposure"][route]
            assert exposure["distinct_live_targets"] >= 320
            assert exposure["live_exposure_max"] <= 2, f"{arm} {route}"


# --- 24-25. provenance and the firewall --------------------------------------

def test_the_provenance_set_closes_over_every_planned_slot() -> None:
    pool_ids = [f"c{index}" for index in range(10)]
    failures = ["f0", "f1"]
    decisions = ([{"candidate_id": item, "accepted": True} for item in pool_ids[:8]]
                 + [{"candidate_id": item, "accepted": False} for item in pool_ids[8:]])

    closure = science.provenance_closure(pool_ids, failures, decisions, pool_ids[:5])

    assert closure["selected"] == 5
    assert closure["accepted_not_selected"] == 3
    assert closure["rejected"] == 2
    assert closure["semantic_failed"] == 2
    assert closure["closed"] is True
    assert closure["unaccounted"] == []


def test_a_selected_candidate_that_was_never_accepted_breaks_the_closure() -> None:
    decisions = [{"candidate_id": "c0", "accepted": False}]

    closure = science.provenance_closure(["c0"], [], decisions, ["c0"])

    assert closure["closed"] is False
    assert closure["selected_outside_accepted"] == ["c0"]


def test_the_selector_resolves_no_target_artifact() -> None:
    for forbidden in ("siw", "SiW", "target_test", "label_live_spoof", "_real_target"):
        assert forbidden not in SELECTOR_SOURCE, forbidden


def test_the_source_domain_field_is_the_canonical_manifest_column() -> None:
    """Never inferred from a path, a filename or a directory name."""
    assert mb.SOURCE_DOMAIN_FIELD == "dataset"
    assert mb.SOURCE_DOMAIN_PLAN_FIELD == "live_dataset"
    for forbidden in ("relative_path", "image_relative_path", "os.path", "basename",
                      "parent.name"):
        assert forbidden not in SELECTOR_SOURCE, forbidden


def test_the_selector_identity_binds_its_rules_and_its_inputs() -> None:
    contract = mb.selector_identity(quality_profile_identity="p" * 64,
                                    c5_pool_lock_sha256="a" * 64,
                                    decision_set_sha256="b" * 64)

    assert contract["selector_name"] == "C6_MATCHED_BANK_SELECTOR_V1"
    assert contract["dimension_priority"] == list(mb.DIMENSION_PRIORITY)
    assert contract["quality_ranking_used"] is False
    assert contract["final_cardinality"] == {"per_arm": 1024, "physics": 512,
                                             "gpat": 512}
    for key in ("planned_distribution_rule", "largest_remainder_rule",
                "common_capacity_rule", "deficit_redistribution_rule",
                "recipe_balancing_rule", "live_balancing_rule", "tie_hash_rule",
                "final_tie_break", "source_domain_field"):
        assert contract[key], key
    # A different pool or profile is a different selector application.
    other = mb.selector_identity(quality_profile_identity="q" * 64,
                                 c5_pool_lock_sha256="a" * 64,
                                 decision_set_sha256="b" * 64)
    assert other["selector_identity_sha256"] != contract["selector_identity_sha256"]


def test_the_dimension_priority_is_frozen_in_order() -> None:
    assert mb.DIMENSION_PRIORITY == (
        "hard_route_cardinality",
        "common_source_domain_exposure",
        "recipe_coverage_balance",
        "base_live_exposure_balance",
        "canonical_tie_hash_then_candidate_id")
