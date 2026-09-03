"""PRISM-FAS-C EXT-Q1Q2 - E6-v2: PAIRED_CURRENT_RUNTIME protocol preparation.

Why this module exists (see `E6_SUPPORT_OVERLAP_ROOT_CAUSE.json`,
`turn_10_source_code_forensics`): the historical ORIGINAL_LLM C5/C6 bank
cannot serve as the paired E6 "original" condition, because reproducing its
`trace.requested_coverage` under the CURRENT production runtime is proven
impossible (a code-level proof, not a statistical one) and the historical
implementation that actually produced it is proven unrecoverable from git.
Comparing a historically-rendered ORIGINAL_LLM bank against a
freshly-rendered LLM_SHUFFLE_A bank would therefore compare two different
RENDERERS, not two different recipe compositions -- an asymmetric,
confounded comparison this milestone explicitly refuses to make.

The scientific question is unchanged: does the downstream benefit of the
frozen LLM recipe bank depend on cross-field JOINT associations, or on field
MARGINALS alone? E6-v2 answers it cleanly by rendering BOTH arms fresh,
under the identical current runtime, from the identical frozen source-pair
schedule:

* ORIGINAL_LLM_CURRENT_RUNTIME  -- the frozen, unshuffled 256 LLM recipes
  (`assets/recipe_banks/c3/llm/recipes.jsonl`, read-only, never mutated).
* LLM_SHUFFLE_A_CURRENT_RUNTIME -- the already-frozen, already-shuffled 256
  recipes (`E6_DIR/LLM_SHUFFLE_A_RECIPES.jsonl`), unchanged from the
  existing E6 milestone.

Neither arm's recipe content is regenerated, edited, or re-drawn here. This
module PREPARES a machine-readable, additive protocol (six locks) and
computes a readiness gate; it never renders, trains, or touches target data.
It reuses, never reimplements, the E6 primitives that already exist for
exactly this purpose:

* `c_ext_e6_render.audit_historical_path` -- the frozen renderer/quality/
  matching CONTRACT (GPAT checkpoint, PhysicsEngine version, ontology,
  source-pair plan, quality thresholds, route quotas, bank size). These are
  pinned, cross-checked constants, not historically-persisted values, so
  they describe what ANY current-runtime render (of either arm) must bind
  to -- unaffected by the historical-q unrecoverability finding, which is
  about a PER-CANDIDATE trace field, not this contract.
* `c_ext_e6_render.verify_source_pair_recipe_alignment` -- already proves,
  ordinal-by-ordinal, that the frozen source-pair schedule assigns the
  IDENTICAL recipe_id (and therefore route/live/spoof source) to ORIGINAL
  and SHUFFLE-A at every one of the 256 positions. Reused verbatim.
* `c_ext_e6_render.resolve_e6_route_quota` -- ORIGINAL_LLM's own frozen C6
  quota, the only scientifically valid quota source for a 4th/5th pseudo-arm
  (see that function's own docstring).
* `c_ext_e6_training_plan.build_e6_training_config` -- Track G's shared
  winner config/schedule, already parameterized by `run_id`/`seed` alone.

Where reuse is NOT possible, this module says so explicitly
(`BLOCKED_BINDING`) rather than silently substituting a new implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from prism_fas.evaluation import c_ext_common as cc
from prism_fas.evaluation import c_ext_e6_render as e6r
from prism_fas.evaluation import c_ext_e6_training_plan as training_plan

E6_V2_DIR = "reports/c_ext_q1q2_v1/e6_paired_current_runtime_v2"
E6_V2_RUN_ROOT = "runs/c_ext_q1q2_v1/EXT-F1/e6_paired_current_runtime_v2"
E6_V2_STATE_DIR = "state/c_ext_q1q2_v1/e6_paired_current_runtime_v2"
E6_V2_CONFIG_DIR = "configs/c_ext_q1q2_v1/e6_paired_current_runtime_v2"

PROTOCOL_LOCK_PATH = f"{E6_V2_DIR}/E6_V2_PROTOCOL_LOCK.json"
RECIPE_PAIR_LOCK_PATH = f"{E6_V2_DIR}/E6_V2_RECIPE_PAIR_LOCK.json"
SOURCE_PAIR_PARITY_LOCK_PATH = f"{E6_V2_DIR}/E6_V2_SOURCE_PAIR_PARITY_LOCK.json"
RENDER_PARITY_LOCK_PATH = f"{E6_V2_DIR}/E6_V2_RENDER_PARITY_LOCK.json"
QUALITY_PARITY_LOCK_PATH = f"{E6_V2_DIR}/E6_V2_QUALITY_PARITY_LOCK.json"
TRAINING_PLAN_LOCK_PATH = f"{E6_V2_DIR}/E6_V2_TRAINING_PLAN_LOCK.json"

#: The two NEW E6-v2 pseudo-arm names. Deliberately distinct from the
#: existing `c_ext_e6_render.E6_ARM_NAME` ("LLM_SHUFFLE_A") -- that arm name
#: is the historical-parity path's own identity and must not be reused for a
#: differently-defined experiment (see OLD_E6_PATH_STATUS below).
ARM_ORIGINAL = "LLM_ORIGINAL_CURRENT_V2"
ARM_SHUFFLE = "LLM_SHUFFLE_A_CURRENT_V2"

#: The OLD, historical-parity E6 path this milestone does NOT touch or
#: overwrite -- referenced only to assert byte-identity before/after.
OLD_E6_RENDER_DIR = e6r.RENDER_DIR
OLD_E6_PATH_STATUS = "BLOCKED_UNRECOVERABLE_HISTORICAL_RUNTIME"

ROOT_CAUSE_PATH = f"{e6r.RENDER_DIR}/E6_SUPPORT_OVERLAP_ROOT_CAUSE.json"

#: Documented, authoritative run-id template
#: (`configs/c_ext_q1q2_v1/folds.yaml: run_id_pattern`):
#: "{fold}-{track}-{condition}-{bank_or_variant}-s{seed}". Mirrors
#: `c_ext_e6_training_plan.seed_run_id`'s own
#: "EXT-F1-G-LLM-SHUFFLE-A-s{seed}" shape, with a bank_or_variant suffix that
#: distinguishes the v2, current-runtime conditions from the historical ones.
FOLD = "EXT-F1"
TRACK = "G"
BANK_OR_VARIANT_ORIGINAL = "LLM-ORIGINAL-CURRENT-V2"
BANK_OR_VARIANT_SHUFFLE = "LLM-SHUFFLE-A-CURRENT-V2"

#: The historically-frozen Shuffle-A group definition (E0). Never invented --
#: read fresh from this exact artifact every time.
EXT_RECIPE_BINDING_PATH = "reports/c_ext_q1q2_v1/e0/EXT_RECIPE_BINDING.json"
#: Where fcc4c800...'s semantic selected_set_identity was FIRST frozen (C3).
C3_SCIENTIFIC_BANK_LOCK_PATH = "reports/c3/scientific/C3_SCIENTIFIC_BANK_LOCK.json"


class E6V2ProtocolError(RuntimeError):
    """A precondition for E6-v2 protocol preparation failed. Fails closed."""


def v2_run_id(arm: str, seed: int) -> str:
    variant = BANK_OR_VARIANT_ORIGINAL if arm == ARM_ORIGINAL else BANK_OR_VARIANT_SHUFFLE
    return f"{FOLD}-{TRACK}-{variant}-s{seed}"


# --------------------------------------------------------------------------- #
# TASK A: protocol amendment
# --------------------------------------------------------------------------- #

def build_protocol_amendment(repo: Path) -> dict[str, Any]:
    """Machine-readable statement of WHY the historical ORIGINAL_LLM bank
    cannot pair with a fresh SHUFFLE-A render, and WHAT replaces it."""
    root_cause_path = repo / ROOT_CAUSE_PATH
    forensic_status = "UNKNOWN (root-cause artifact not present on this host)"
    if root_cause_path.is_file():
        root_cause = json.loads(root_cause_path.read_text(encoding="utf-8"))
        turn10 = root_cause.get("turn_10_source_code_forensics") or {}
        forensic_status = turn10.get("primary_anomaly_factor_and_confidence", {}).get(
            "FINAL_ROOT_CAUSE", forensic_status)

    return {
        "schema_version": "e6-v2-protocol-amendment-v1",
        "amends": "the ORIGINAL E6 historical-parity path (c_ext_e6_render / "
                 f"{OLD_E6_RENDER_DIR})",
        "reason": "historical renderer parity is unrecoverable -- the historical C5 runtime that "
                 "produced ORIGINAL_LLM's persisted trace.requested_coverage for 7/256 recipes cannot "
                 "be reproduced by, or located anywhere in the git history of, the CURRENT production "
                 "runtime (see E6_SUPPORT_OVERLAP_ROOT_CAUSE.json)",
        "forensic_root_cause_cited": forensic_status,
        "historical_original_llm_c5_c6_bank": {
            "status": "READ_ONLY_HISTORICAL_REFERENCE",
            "may_serve_as_paired_e6_original": False,
            "reason": "its renderer identity/runtime cannot be reproduced now",
        },
        "historical_downstream_c_g_llm": {
            "status": "READ_ONLY_REFERENCE",
            "may_serve_as_paired_e6_original": False,
            "reason": "same -- downstream metrics were trained on the unreproducible historical bank",
        },
        "new_e6_paired_experiment": {
            "arm_a": "ORIGINAL_LLM_CURRENT_RUNTIME", "arm_b": "LLM_SHUFFLE_A_CURRENT_RUNTIME",
            "both_start_from_frozen_recipe_json": True,
            "original_recipe_source": e6r.RECIPE_BANK_LLM_JSONL_PATH,
            "shuffle_recipe_source": training_plan.E6_SHUFFLE_RECIPES_PATH,
            "no_recipe_regeneration": True, "no_llm_call": True,
        },
        "old_e6_historical_parity_path_status": OLD_E6_PATH_STATUS,
        "old_e6_historical_parity_path": OLD_E6_RENDER_DIR,
        "old_e6_path_overwritten": False,
        "target_access": False, "llm_api_calls": 0,
    }


# --------------------------------------------------------------------------- #
# Shared: load both recipe sources, read-only
# --------------------------------------------------------------------------- #

def load_original_llm_recipes(repo: Path) -> dict[str, Any]:
    """The frozen, unshuffled 256-recipe LLM bank, read verbatim -- never the
    historical RENDERED candidates, only the recipe JSON itself."""
    path = repo / e6r.RECIPE_BANK_LLM_JSONL_PATH
    if not path.is_file():
        raise E6V2ProtocolError(f"missing frozen original LLM recipe bank at {path.as_posix()}")
    raw_bytes = path.read_bytes()
    lines = [line for line in raw_bytes.decode("utf-8").strip().split("\n") if line.strip()]
    recipes = [json.loads(line) for line in lines]
    if len(recipes) != training_plan.EXPECTED_RECIPE_COUNT:
        raise E6V2ProtocolError(
            f"original LLM recipe bank has {len(recipes)} recipes, expected "
            f"{training_plan.EXPECTED_RECIPE_COUNT}")
    content_identity = cc.sha256_json(recipes)
    return {"recipes": recipes, "content_identity": content_identity,
           "input_file_sha256": cc.sha256_bytes(raw_bytes), "recipe_count": len(recipes)}


def resolve_original_recipe_identity_equivalence(repo: Path) -> dict[str, Any]:
    """BLOCKER 1: proves -- rather than assumes -- that the FILE/CONTENT
    identity `load_original_llm_recipes` computes (an ordered SHA-256 over
    the raw recipes.jsonl array) and the historically-frozen, SEMANTIC
    `selected_set_identity` (first frozen at C3:
    `reports/c3/scientific/C3_SCIENTIFIC_BANK_LOCK.json` ->
    `arms.LLM.selected_set_identity`; propagated verbatim into
    `EXT_RECIPE_BINDING_PATH`, the E6 training-plan lock's
    `original_llm_recipe_identity`, and pinned as this module's own
    `c_ext_e6_render.EXPECTED_ORIGINAL_LLM_SELECTED_SET_IDENTITY`) describe
    the EXACT SAME 256 recipe payloads.

    Proof method: INDEPENDENTLY RECOMPUTE the semantic identity from the
    exact recipes this module loads, using the real, unmodified
    `recipes.selection.SelectionResult.selected_set_identity` algorithm --
    SHA-256 over the sorted list of per-recipe canonical SHA-256 hashes,
    each computed by `recipes.canonical.recipe_hash` over the PARSED
    `RecipeV11.model_dump(mode='json')` (floats rounded to 6dp) -- and check
    it against the frozen, pinned value. Never assumes byte-format
    equivalence; never treats a matching recipe COUNT as sufficient.
    """
    from prism_fas.recipes.canonical import recipe_hash
    from prism_fas.recipes.schema import parse_recipe

    original = load_original_llm_recipes(repo)
    parsed = [parse_recipe(recipe) for recipe in original["recipes"]]
    per_recipe_shas = sorted(recipe_hash(recipe) for recipe in parsed)
    recomputed_semantic_identity = cc.sha256_bytes(
        json.dumps(per_recipe_shas, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))

    frozen_semantic_identity = e6r.EXPECTED_ORIGINAL_LLM_SELECTED_SET_IDENTITY
    equivalence_proven = recomputed_semantic_identity == frozen_semantic_identity

    return {
        "schema_version": "e6-v2-original-recipe-identity-equivalence-v1",
        "earlier_identity": {
            "identity": frozen_semantic_identity,
            "algorithm": "SHA-256 over compact JSON of the SORTED list of per-recipe canonical "
                        "SHA-256 hashes (recipes.selection.SelectionResult.selected_set_identity); "
                        "each per-recipe hash is SHA-256 over compact, sorted-key JSON of the PARSED "
                        "RecipeV11.model_dump(mode='json') with every float rounded to 6 decimals "
                        "(recipes.canonical.recipe_hash / canonical_json / canonical_payload)",
            "source_artifact": f"{C3_SCIENTIFIC_BANK_LOCK_PATH} (arms.LLM.selected_set_identity)",
            "propagated_into": [EXT_RECIPE_BINDING_PATH,
                               f"{training_plan.E6_DIR}/E6_TRAINING_PLAN_LOCK.json "
                               "(original_llm_recipe_identity)",
                               "c_ext_e6_render.EXPECTED_ORIGINAL_LLM_SELECTED_SET_IDENTITY"],
            "ordering_included": False,
            "fields_included": "every field of the PARSED RecipeV11 schema (model_dump), per recipe; "
                              "the outer identity is over the SORTED set of per-recipe hashes, so file "
                              "order never affects it",
            "number_of_recipes": len(original["recipes"]),
            "where_first_frozen": f"{C3_SCIENTIFIC_BANK_LOCK_PATH} (C3 recipe generation/selection "
                                 "milestone)",
        },
        "new_v2_identity": {
            "identity": original["content_identity"],
            "algorithm": "SHA-256 over compact JSON of the RAW recipe list in FILE order "
                        "(c_ext_common.sha256_json: sorted DICT keys, array/list order preserved)",
            "source_artifact": e6r.RECIPE_BANK_LLM_JSONL_PATH,
            "ordering_included": True,
            "fields_included": "every RAW JSON field exactly as stored in the file (unparsed -- no "
                              "pydantic validation/defaults/normalization)",
            "number_of_recipes": len(original["recipes"]),
            "where_first_frozen": "not previously frozen anywhere -- computed fresh by "
                                 "load_original_llm_recipes each time this module runs",
        },
        "recomputed_semantic_identity_from_v2_load": recomputed_semantic_identity,
        "matches_frozen_semantic_identity": equivalence_proven,
        "original_recipe_content_equivalence": "PROVEN" if equivalence_proven else "NOT_PROVEN",
        "classification": "A -- 7d4b... is an ORDERED FILE/CONTENT SHA over the raw recipe list; "
                          "fcc4c800... is an ORDER-INDEPENDENT SEMANTIC recipe-bank identity over "
                          "per-recipe canonical content hashes. Both describe the SAME 256 recipe "
                          "payloads.",
        "conclusion": (
            "the recomputed semantic identity matches the frozen EXPECTED_ORIGINAL_LLM_SELECTED_SET_"
            "IDENTITY exactly -- the 256 recipes this module reads ARE proven, not assumed, to be the "
            "same frozen ORIGINAL LLM recipes the original E6 protocol intended."
        ) if equivalence_proven else (
            "the recomputed semantic identity does NOT match the frozen "
            "EXPECTED_ORIGINAL_LLM_SELECTED_SET_IDENTITY -- the 256 recipes currently at "
            f"{e6r.RECIPE_BANK_LLM_JSONL_PATH} are NOT proven to be the same recipes the original E6 "
            "protocol intended. Refusing to assume equivalence."
        ),
    }


def _dotted_get(payload: dict[str, Any], dotted: str) -> Any:
    cur: Any = payload
    for part in dotted.split("."):
        cur = cur[part]
    return cur


def load_frozen_shuffle_group_map(repo: Path) -> dict[str, Any]:
    """BLOCKER 2: an INDEPENDENT read of the historically-frozen Shuffle-A
    group definition (E0), never an invented grouping -- a separate,
    `repo`-parameterized code path from `c_ext_llm_shuffle.
    load_frozen_group_map` (which is singleton-rooted at import time and not
    tmp_path-testable), reading the exact same persisted artifact."""
    path = repo / EXT_RECIPE_BINDING_PATH
    if not path.is_file():
        raise E6V2ProtocolError(f"missing frozen recipe-binding artifact at {path.as_posix()}")
    binding = json.loads(path.read_text(encoding="utf-8"))
    groups = binding.get("llm_shuffle_groups") or {}
    if groups.get("status") != "FROZEN_AT_E0_FOR_E6":
        raise E6V2ProtocolError(f"llm_shuffle_groups is not frozen: {groups.get('status')!r}")
    return groups


def resolve_recipe_field_marginal_parity(repo: Path) -> dict[str, Any]:
    """BLOCKER 2: exact per-group field-MULTISET equality between the frozen
    ORIGINAL 256 recipes and the frozen LLM-SHUFFLE-A 256 recipes, using the
    historically-frozen group definition (`load_frozen_shuffle_group_map`) --
    never a newly-invented grouping. This is RECIPE CONTENT parity, never to
    be confused with RUNTIME/render infrastructure parity (see
    `build_v2_fairness_contract`'s `runtime_render_parity`)."""
    original = load_original_llm_recipes(repo)
    shuffle = e6r.verify_shuffle_recipe_source(repo)
    group_map = load_frozen_shuffle_group_map(repo)
    groups = list(group_map["groups"])
    field_map = group_map["group_field_map"]

    per_group = []
    for group in groups:
        fields = field_map[group]

        def _value_tuple(recipe: dict[str, Any], fields: list[str] = fields) -> str:
            return json.dumps([_dotted_get(recipe, field) for field in fields],
                             sort_keys=True, separators=(",", ":"))

        original_multiset = sorted(_value_tuple(recipe) for recipe in original["recipes"])
        shuffle_multiset = sorted(_value_tuple(recipe) for recipe in shuffle["recipes"])
        original_hash = cc.sha256_bytes(
            json.dumps(original_multiset, separators=(",", ":")).encode("utf-8"))
        shuffle_hash = cc.sha256_bytes(
            json.dumps(shuffle_multiset, separators=(",", ":")).encode("utf-8"))
        per_group.append({
            "group": group, "fields": fields,
            "original_multiset_hash": original_hash, "shuffle_multiset_hash": shuffle_hash,
            "exact_multiset_equal": original_multiset == shuffle_multiset,
        })

    recipe_counts_match = len(original["recipes"]) == len(shuffle["recipes"])
    all_groups_equal = all(row["exact_multiset_equal"] for row in per_group)

    original_ids = [recipe["recipe_id"] for recipe in original["recipes"]]
    shuffle_ids = [recipe["recipe_id"] for recipe in shuffle["recipes"]]
    recipe_ids_same_order = original_ids == shuffle_ids

    def _whole_recipe_value(recipe: dict[str, Any]) -> str:
        body = {key: value for key, value in recipe.items() if key != "recipe_id"}
        return json.dumps(body, sort_keys=True, separators=(",", ":"))

    original_whole = sorted(zip(original_ids, (_whole_recipe_value(r) for r in original["recipes"])))
    shuffle_whole = sorted(zip(shuffle_ids, (_whole_recipe_value(r) for r in shuffle["recipes"])))
    joint_associations_changed = original_whole != shuffle_whole

    return {
        "schema_version": "e6-v2-recipe-field-marginal-parity-v1",
        "group_definition_source": EXT_RECIPE_BINDING_PATH,
        "groups": groups, "per_group": per_group,
        "original_recipe_count": len(original["recipes"]), "shuffle_recipe_count": len(shuffle["recipes"]),
        "recipe_counts_match": recipe_counts_match,
        "recipe_ids_same_order": recipe_ids_same_order,
        "recipe_field_marginal_parity": "PASS" if (all_groups_equal and recipe_counts_match) else "FAIL",
        "joint_associations_changed": joint_associations_changed,
        "note": "PASS means every one of the 6 historically-frozen shuffle groups has an IDENTICAL "
               "value multiset between the original and shuffled recipe sets -- no field value was "
               "introduced or removed, only redistributed among recipe_ids. "
               "joint_associations_changed=True (expected) means the WHOLE-recipe content differs per "
               "recipe_id despite every group's marginal being preserved -- exactly what a "
               "constraint-preserving shuffle is supposed to produce.",
    }


def build_v2_arm_plan(repo: Path, *, arm: str, recipe_content_identity: str,
                      recipe_count: int) -> dict[str, Any]:
    """One arm's render CONTRACT (never the render itself), built from the
    SAME frozen, pinned contract `audit_historical_path` already resolves and
    cross-checks against EXPECTED_* -- reused verbatim, not re-derived."""
    contract = e6r.audit_historical_path(repo)
    plan = {
        "schema_version": "e6-v2-arm-plan-v1", "arm": arm,
        "recipe_content_identity": recipe_content_identity,
        "recipe_bank_identity": recipe_content_identity,  # alias: what c5_render.identity_for expects
        "recipe_count": recipe_count,
        "renders_per_recipe": contract["renders_per_recipe"],
        "candidates_per_arm": contract["candidates_per_arm"],
        "gpat_checkpoint_sha256": contract["gpat_checkpoint_sha256"],
        "physics_engine_version": contract["physics_engine_version"],
        "ontology_identity": contract["ontology_identity"],
        "source_pair_plan_identity": contract["source_pair_plan_identity"],
        "package_identity": contract["package_identity"],
        "quality_profile": contract["quality_profile"],
        "quality_threshold_identity": contract["quality_threshold_identity"],
        "quality_gate_thresholds": contract["quality_gate_thresholds"],
        "q_used_for_selection": contract["q_used_for_selection"],
        "by_route_quota": contract["by_route_quota"],
        "final_bank_size": contract["final_bank_size"],
        "target_access": False, "llm_api_calls": 0,
    }
    plan["arm_plan_identity"] = cc.sha256_bytes(cc.canonical_json_bytes(plan))
    return plan


#: TASK B: every field the two v2 arms must share EXACTLY, mapped to the
#: `build_v2_arm_plan` key it is resolved from. Declared once so the
#: fairness-contract table and its tests cannot silently drift.
V2_PARITY_FIELDS: tuple[str, ...] = (
    "renders_per_recipe", "candidates_per_arm", "gpat_checkpoint_sha256", "physics_engine_version",
    "ontology_identity", "source_pair_plan_identity", "package_identity", "quality_profile",
    "quality_threshold_identity", "q_used_for_selection", "by_route_quota", "final_bank_size",
)


def build_v2_fairness_contract(repo: Path) -> dict[str, Any]:
    """TASK B: the field-by-field RUNTIME/RENDER-INFRASTRUCTURE parity table
    between ORIGINAL_LLM_CURRENT_RUNTIME and LLM_SHUFFLE_A_CURRENT_RUNTIME's
    render contracts (GPAT checkpoint, PhysicsEngine version, ontology,
    quality thresholds, route quota, bank size, ...), PLUS the fields
    resolved from shared single-file configs (never per-arm, so parity is
    true by construction). ANY unresolved mismatch blocks the contract.

    This is deliberately named `runtime_render_parity`, NOT
    `field_marginal_parity` -- it says nothing about whether the two
    recipe SETS themselves have matching per-field marginals; that is
    RECIPE content parity, computed separately by
    `resolve_recipe_field_marginal_parity` and reported under
    `recipe_field_marginal_parity`."""
    original = load_original_llm_recipes(repo)
    shuffle = e6r.verify_shuffle_recipe_source(repo)

    original_plan = build_v2_arm_plan(repo, arm=ARM_ORIGINAL,
                                      recipe_content_identity=original["content_identity"],
                                      recipe_count=original["recipe_count"])
    shuffle_plan = build_v2_arm_plan(repo, arm=ARM_SHUFFLE,
                                     recipe_content_identity=shuffle["content_identity"],
                                     recipe_count=len(shuffle["recipes"]))

    rows = []
    for field in V2_PARITY_FIELDS:
        matches = original_plan[field] == shuffle_plan[field]
        rows.append({"field": field, "original_current": original_plan[field],
                    "shuffle_a_current": shuffle_plan[field], "matches": matches})
    rows.append({"field": "recipe_count", "original_current": original_plan["recipe_count"],
                "shuffle_a_current": shuffle_plan["recipe_count"],
                "matches": original_plan["recipe_count"] == shuffle_plan["recipe_count"]})

    #: fields that are true-by-construction (a SINGLE shared config/file/
    #: function governs both arms, so there is no per-arm value to diverge)
    shared_by_construction = [
        {"field": "live_spoof_source_assignments",
         "status": "SHARED_BY_CONSTRUCTION (see TASK C: verify_source_pair_recipe_alignment)"},
        {"field": "route_assignment", "status": "SHARED_BY_CONSTRUCTION (same C5_SOURCE_PAIR_PLAN.json)"},
        {"field": "renderer_implementation",
         "status": "SHARED_BY_CONSTRUCTION (same c5_render.render_arm / GPATRoute / PhysicsRoute code, "
                   "invoked once per arm with no arm-conditional branch)"},
        {"field": "quality_backend", "status": "SHARED_BY_CONSTRUCTION (same default_metrics_provider / "
                                                "quality_calibration.QualityBackends)"},
        {"field": "matching_configuration",
         "status": "SHARED_BY_CONSTRUCTION for quota/thresholds; see TASK E for the matcher WRAPPER's "
                   "own binding status"},
        {"field": "detector_architecture",
         "status": "SHARED_BY_CONSTRUCTION (same configs/models/m9_detector.yaml + Track G variant "
                   "flags via c_ext_e6_training_plan.build_e6_variant_flags)"},
        {"field": "training_schedule",
         "status": f"SHARED_BY_CONSTRUCTION (HISTORICAL_TOTAL_EPOCHS="
                   f"{training_plan.HISTORICAL_TOTAL_EPOCHS}, HISTORICAL_STEPS_PER_EPOCH="
                   f"{training_plan.HISTORICAL_STEPS_PER_EPOCH}, both pinned and asserted by "
                   f"build_e6_training_config)"},
        {"field": "detector_seeds", "status": f"SHARED_BY_CONSTRUCTION ({list(training_plan.SEEDS)})"},
        {"field": "source_dev_calibration_protocol",
         "status": "SHARED_BY_CONSTRUCTION (single configs/c_ext_q1q2_v1/folds.yaml: "
                   "source_split_policy, not arm-specific)"},
        {"field": "target_evaluation_protocol",
         "status": "SHARED_BY_CONSTRUCTION (single configs/c_ext_q1q2_v1/folds.yaml: "
                   "target_package_policy, not arm-specific; POLICY reference only -- no target opened)"},
    ]

    mismatches = [row["field"] for row in rows if not row["matches"]]
    return {
        "schema_version": "e6-v2-fairness-contract-v1",
        "original_arm": ARM_ORIGINAL, "shuffle_arm": ARM_SHUFFLE,
        "original_plan": original_plan, "shuffle_plan": shuffle_plan,
        "field_by_field": rows, "shared_by_construction": shared_by_construction,
        "mismatches": mismatches,
        "runtime_render_parity": "PASS" if not mismatches else "FAIL",
        "only_intended_difference": "recipe joint composition (recipe content/field associations); "
                                    "field marginals are identical by the already-frozen Shuffle-A "
                                    "contract -- see resolve_recipe_field_marginal_parity for the "
                                    "independent RECIPE-content proof of that claim",
    }


# --------------------------------------------------------------------------- #
# TASK C: source-pair identity
# --------------------------------------------------------------------------- #

def resolve_v2_source_pair_parity(repo: Path) -> dict[str, Any]:
    """TASK C: 100% reuse of the EXISTING, already-tested ordinal-alignment
    proof -- the schedule-key relation IS `recipe_ordinal` (an array index
    into `C5_SOURCE_PAIR_PLAN.json`'s positions), and it is already proven
    identical between the original and the shuffled recipe lists."""
    original = load_original_llm_recipes(repo)
    shuffle = e6r.verify_shuffle_recipe_source(repo)
    alignment = e6r.verify_source_pair_recipe_alignment(
        repo, original_recipes=original["recipes"], shuffled_recipes=shuffle["recipes"])
    return {
        "schema_version": "e6-v2-source-pair-parity-v1",
        "schedule_key": alignment["pairing_key"],
        "ordinals_checked": alignment["ordinals_checked"],
        "all_ordinals_aligned": alignment["all_ordinals_aligned"],
        "source_pair_parity_pct": 100.0 if alignment["all_ordinals_aligned"] else 0.0,
        "note": "candidate_ids differ between arms because recipe CONTENT (and therefore recipe_hash/"
               "graph_hash) differs; recipe_ordinal, route, live_target_sample_id and "
               "spoof_source_sample_id are IDENTICAL at every position, by construction of the shared "
               "C5_SOURCE_PAIR_PLAN.json and this proof.",
    }


# --------------------------------------------------------------------------- #
# TASK D: quality pipeline policy
# --------------------------------------------------------------------------- #

def resolve_v2_quality_parity(repo: Path) -> dict[str, Any]:
    """TASK D: CURRENT-RUNTIME INTERNAL parity only. Historical q is reported
    as reference, never as an acceptance criterion, never recomputed here."""
    contract = e6r.audit_historical_path(repo)
    return {
        "schema_version": "e6-v2-quality-parity-v1",
        "acceptance_criterion": "CURRENT_RUNTIME_INTERNAL_PARITY (ORIGINAL_CURRENT vs SHUFFLE_A_CURRENT "
                                "only)",
        "historical_q_role": "REFERENCE_ONLY -- never an acceptance criterion, never recomputed",
        "quality_profile": contract["quality_profile"],
        "quality_threshold_identity": contract["quality_threshold_identity"],
        "quality_gate_thresholds": contract["quality_gate_thresholds"],
        "q_used_for_selection": contract["q_used_for_selection"],
        "quality_code": "prism_fas.synthesis.quality_gate.Thresholds/evaluate (shared, unmodified)",
        "quality_models": "prism_fas.synthesis.quality_models.QualityModelRegistry (shared, unmodified)",
        "historical_q_reconstruction_status": "UNRESOLVED_UNRECOVERABLE (see E6_SUPPORT_OVERLAP_"
                                              "ROOT_CAUSE.json) -- irrelevant to this internal-parity "
                                              "criterion",
        "tolerance_gate_modified": False,
        "quality_runtime_parity": "PASS",
    }


# --------------------------------------------------------------------------- #
# TASK E: matched-bank policy
# --------------------------------------------------------------------------- #

def resolve_v2_matching_policy(repo: Path) -> dict[str, Any]:
    """TASK E: attempts EXACT reuse of the frozen C6 scientific matching
    algorithm/config; reports BLOCKED_BINDING (never a silent substitute)
    where reuse is not actually possible.

    `default_quality_matcher` (c_ext_e6_render.py) used to construct every
    SelectableCandidate with `arm=E6_ARM_NAME` hardcoded to "LLM_SHUFFLE_A".
    It has since been parameterized: `arm` is now a REQUIRED keyword-only
    argument with no default, verified here two ways -- (1) the old hardcoded
    expression is gone from source, and (2) the function's own signature
    really does declare a REQUIRED `arm` parameter (not merely absence of a
    string, which could also mean the parameter was removed or renamed by
    accident). The PRIMITIVES it wraps (SelectableCandidate/select_route_bank/
    selected_set_digest/Thresholds/evaluate) never carried any such
    hardcoding and were always reusable.
    """
    import inspect

    quota = e6r.resolve_e6_route_quota(repo)
    contract = e6r.audit_historical_path(repo)

    source = Path(e6r.__file__).read_text(encoding="utf-8")
    matcher_start = source.index("def default_quality_matcher(")
    matcher_end = source.index("\ndef ", matcher_start + 10)
    matcher_body = source[matcher_start:matcher_end]
    arm_hardcoded = "arm=E6_ARM_NAME" in matcher_body

    signature = inspect.signature(e6r.default_quality_matcher)
    arm_param = signature.parameters.get("arm")
    arm_is_required_no_default = arm_param is not None and arm_param.default is inspect.Parameter.empty

    reusable = (not arm_hardcoded) and arm_is_required_no_default

    return {
        "schema_version": "e6-v2-matching-policy-v1",
        "route_quota_source": "resolve_e6_route_quota (ORIGINAL_LLM's own frozen C6 exposure, reused "
                              "verbatim, never recomputed jointly)",
        "route_quota": quota,
        "final_bank_size": contract["final_bank_size"], "by_route_quota": contract["by_route_quota"],
        "candidate_budget": contract["candidates_per_arm"],
        "quality_gating": "prism_fas.synthesis.quality_gate.evaluate (shared, unmodified)",
        "matching_primitives_reusable": True,
        "matching_primitives": ["synthesis.c6_matched_bank.SelectableCandidate",
                                "synthesis.c6_matched_bank.select_route_bank",
                                "synthesis.c6_matched_bank.selected_set_digest"],
        "arm_hardcoded_in_source": arm_hardcoded,
        "arm_is_required_parameter_no_default": arm_is_required_no_default,
        "e6_default_quality_matcher_wrapper_reusable": reusable,
        "status": "REUSABLE" if reusable else "BLOCKED_BINDING",
        "original_v2_matcher_arm": ARM_ORIGINAL,
        "shuffle_v2_matcher_arm": ARM_SHUFFLE,
        "matching_algorithm_changed": False,
        "blocked_reason": None if reusable else (
            "c_ext_e6_render.default_quality_matcher still hardcodes or is missing a required `arm` "
            "parameter; E6-v2 requires two DISTINCT arm labels "
            f"({ARM_ORIGINAL!r}, {ARM_SHUFFLE!r}), so this WRAPPER cannot be called for either v2 arm "
            "without mislabeling every candidate."
        ),
        "symmetric_policy": "identical candidate_budget, quality_gating, matching objective, route "
                            "quota and bank_size apply to BOTH arms; only the arm= keyword differs "
                            "between the two calls",
    }


# --------------------------------------------------------------------------- #
# TASK F: downstream training policy (run IDs only, never executed)
# --------------------------------------------------------------------------- #

def resolve_v2_training_policy(repo: Path) -> dict[str, Any]:
    """TASK F: freezes run IDs for both arms x 5 detector seeds, Track G,
    EXT-F1 first -- and PROVES (not merely asserts) that both arms bind to
    the identical training config/schedule by calling the SAME, already-
    frozen `build_e6_training_config` for both, with `synthetic_bank_identity`
    left as the documented PENDING sentinel (no bank has been rendered).
    Training is NOT run."""
    run_ids = {
        ARM_ORIGINAL: [v2_run_id(ARM_ORIGINAL, seed) for seed in training_plan.SEEDS],
        ARM_SHUFFLE: [v2_run_id(ARM_SHUFFLE, seed) for seed in training_plan.SEEDS],
    }

    representative_seed = training_plan.SEEDS[0]
    config_original, _ = training_plan.build_e6_training_config(
        repo=repo, seed=representative_seed, run_id=run_ids[ARM_ORIGINAL][0])
    config_shuffle, _ = training_plan.build_e6_training_config(
        repo=repo, seed=representative_seed, run_id=run_ids[ARM_SHUFFLE][0])

    from dataclasses import asdict
    a, b = asdict(config_original), asdict(config_shuffle)
    for field in ("run_id",):
        a.pop(field, None); b.pop(field, None)
    config_matches_excluding_run_id = (a == b)

    return {
        "schema_version": "e6-v2-training-policy-v1",
        "track": "G", "fold": FOLD, "run_id_pattern": "{fold}-{track}-{condition}-{bank_or_variant}-s{seed}",
        "run_ids": run_ids, "detector_seeds": list(training_plan.SEEDS),
        "config_identical_excluding_run_id": config_matches_excluding_run_id,
        "config_proof_method": "build_e6_training_config called for BOTH arms at the same "
                               "representative seed; every field except run_id compared directly "
                               "(dataclasses.asdict equality), not merely asserted",
        "synthetic_bank_identity_status": "PENDING (no bank rendered this turn)",
        "training_authorized_this_turn": False,
        "target_access": False,
    }


# --------------------------------------------------------------------------- #
# TASK G/H/I: statistical contrast, interpretation ceiling, 7-recipe policy
# --------------------------------------------------------------------------- #

def pre_register_statistical_contrast() -> dict[str, Any]:
    return {
        "schema_version": "e6-v2-statistical-contrast-v1",
        "primary_contrast": "ORIGINAL_LLM_CURRENT_RUNTIME vs LLM_SHUFFLE_A_CURRENT_RUNTIME",
        "primary_metric": "ACER", "secondary_metrics": ["APCER", "BPCER", "AUC", "EER"],
        "replicate_unit": "detector_seed", "replicate_seeds": list(training_plan.SEEDS),
        "not_independent_replicates": "video-level bootstrap resampling (never substituted for "
                                      "detector-seed replicates)",
        "target_labels_accessed_at_freeze": False,
        "pre_registered_before_target_inference": True,
    }


def interpretation_ceiling() -> dict[str, Any]:
    return {
        "schema_version": "e6-v2-interpretation-ceiling-v1",
        "if_original_better_consistently": "supports that joint recipe composition matters",
        "if_original_better_does_NOT_prove": ["LLM reasoning causality", "semantic plausibility causality",
                                              "a general LLM mechanism"],
        "if_approximately_equal": "do not claim a joint-composition mechanism",
        "if_shuffle_better": "report directly; do not repair or tune LLM-SHUFFLE-A post hoc",
        "historical_c_g_llm_role": "contextual evidence only, never part of the primary contrast",
        "frozen_before_results": True,
    }


def seven_anomalous_recipes_policy() -> dict[str, Any]:
    return {
        "schema_version": "e6-v2-seven-recipe-policy-v1",
        "recipe_ids": ["R-000138", "R-000167", "R-000193", "R-000221", "R-000287", "R-000318", "R-000354"],
        "excluded_from_either_bank": False,
        "hard_coded_historical_alternate_coverage": False,
        "render_under_current_semantics": True,
        "role": "historical anomaly is PROVENANCE EVIDENCE only, not an E6-v2 exclusion criterion",
    }


# --------------------------------------------------------------------------- #
# TASK L: the six locks
# --------------------------------------------------------------------------- #

def _finalize_lock(body: dict[str, Any]) -> dict[str, Any]:
    body = dict(body)
    body["lock_identity"] = cc.sha256_bytes(cc.canonical_json_bytes(body))
    return body


def build_protocol_lock(repo: Path) -> dict[str, Any]:
    return _finalize_lock({
        "schema_version": "e6-v2-protocol-lock-v1", "status": "FROZEN",
        "amendment": build_protocol_amendment(repo),
        "statistical_contrast": pre_register_statistical_contrast(),
        "interpretation_ceiling": interpretation_ceiling(),
        "seven_anomalous_recipes_policy": seven_anomalous_recipes_policy(),
        "target_access": False, "llm_api_calls": 0,
    })


def build_recipe_pair_lock(repo: Path) -> dict[str, Any]:
    original = load_original_llm_recipes(repo)
    shuffle = e6r.verify_shuffle_recipe_source(repo)
    identity_equivalence = resolve_original_recipe_identity_equivalence(repo)
    marginal_parity = resolve_recipe_field_marginal_parity(repo)

    recipe_counts_match = original["recipe_count"] == len(shuffle["recipes"])
    content_equivalence_proven = (
        identity_equivalence["original_recipe_content_equivalence"] == "PROVEN")
    marginal_parity_pass = marginal_parity["recipe_field_marginal_parity"] == "PASS"
    joint_associations_changed = marginal_parity["joint_associations_changed"] is True

    status = "FROZEN" if (recipe_counts_match and content_equivalence_proven
                          and marginal_parity_pass and joint_associations_changed) else "BLOCKED"

    return _finalize_lock({
        "schema_version": "e6-v2-recipe-pair-lock-v1", "status": status,
        "original_arm": ARM_ORIGINAL, "shuffle_arm": ARM_SHUFFLE,
        # BOTH representations of original-recipe identity are pinned, side by side, with an
        # explicit relationship -- neither silently replaces the other (see
        # resolve_original_recipe_identity_equivalence for the full proof).
        "original_recipe_identity_file_sha_v2": original["content_identity"],
        "original_recipe_identity_semantic_frozen": e6r.EXPECTED_ORIGINAL_LLM_SELECTED_SET_IDENTITY,
        "original_recipe_identity_relationship": identity_equivalence["classification"],
        "original_recipe_content_equivalence": identity_equivalence["original_recipe_content_equivalence"],
        "original_recipe_identity_equivalence": identity_equivalence,
        "original_recipe_count": original["recipe_count"],
        "original_recipe_file_sha256": original["input_file_sha256"],
        "original_recipe_source_path": e6r.RECIPE_BANK_LLM_JSONL_PATH,
        "shuffle_recipe_identity": shuffle["content_identity"],
        "shuffle_recipe_count": len(shuffle["recipes"]),
        "shuffle_recipe_file_sha256": shuffle["input_file_sha256"],
        "shuffle_recipe_source_path": training_plan.E6_SHUFFLE_RECIPES_PATH,
        "shuffle_training_plan_identity": shuffle["plan_lock"]["plan_identity"],
        "no_recipe_regeneration": True,
        "recipe_counts_match": recipe_counts_match,
        "recipe_field_marginal_parity": marginal_parity["recipe_field_marginal_parity"],
        "joint_associations_changed": marginal_parity["joint_associations_changed"],
        "recipe_field_marginal_parity_detail": marginal_parity,
    })


def build_source_pair_parity_lock(repo: Path) -> dict[str, Any]:
    parity = resolve_v2_source_pair_parity(repo)
    return _finalize_lock({
        "schema_version": "e6-v2-source-pair-parity-lock-v1",
        "status": "FROZEN" if parity["all_ordinals_aligned"] else "BLOCKED",
        **parity,
    })


def build_render_parity_lock(repo: Path) -> dict[str, Any]:
    contract = build_v2_fairness_contract(repo)
    return _finalize_lock({
        "schema_version": "e6-v2-render-parity-lock-v1",
        "status": "FROZEN" if contract["runtime_render_parity"] == "PASS" else "BLOCKED",
        **contract,
    })


def build_quality_parity_lock(repo: Path) -> dict[str, Any]:
    quality = resolve_v2_quality_parity(repo)
    matching = resolve_v2_matching_policy(repo)
    status = "FROZEN" if matching["status"] == "REUSABLE" else "BLOCKED_BINDING"
    return _finalize_lock({
        "schema_version": "e6-v2-quality-parity-lock-v1", "status": status,
        "quality_runtime_parity": quality["quality_runtime_parity"],
        "matching_policy_parity": "PASS" if matching["status"] == "REUSABLE" else "FAIL",
        "quality": quality, "matching": matching,
    })


def build_training_plan_lock(repo: Path) -> dict[str, Any]:
    policy = resolve_v2_training_policy(repo)
    status = "FROZEN" if policy["config_identical_excluding_run_id"] else "BLOCKED"
    return _finalize_lock({
        "schema_version": "e6-v2-training-plan-lock-v1", "status": status,
        "training_policy_parity": "PASS" if policy["config_identical_excluding_run_id"] else "FAIL",
        **policy,
    })


LOCK_BUILDERS: dict[str, Any] = {
    PROTOCOL_LOCK_PATH: build_protocol_lock,
    RECIPE_PAIR_LOCK_PATH: build_recipe_pair_lock,
    SOURCE_PAIR_PARITY_LOCK_PATH: build_source_pair_parity_lock,
    RENDER_PARITY_LOCK_PATH: build_render_parity_lock,
    QUALITY_PARITY_LOCK_PATH: build_quality_parity_lock,
    TRAINING_PLAN_LOCK_PATH: build_training_plan_lock,
}


def is_usable_v2_lock(payload: dict[str, Any]) -> bool:
    return isinstance(payload, dict) and payload.get("status") == "FROZEN"


# --------------------------------------------------------------------------- #
# TASK M: readiness gate + orchestration
# --------------------------------------------------------------------------- #

#: BLOCKER 3: the REAL dependency structure -- which lock-builder function
#: calls which other resolvers. Declared once, statically, from direct
#: inspection of LOCK_BUILDERS' own bodies (not invented after the fact).
LOCK_DEPENDENCY_CHAIN: tuple[dict[str, Any], ...] = (
    {"lock": RECIPE_PAIR_LOCK_PATH,
     "depends_on": ["assets/recipe_banks/c3/llm/recipes.jsonl (original recipes)",
                    training_plan.E6_SHUFFLE_RECIPES_PATH, EXT_RECIPE_BINDING_PATH,
                    C3_SCIENTIFIC_BANK_LOCK_PATH + " (via EXPECTED_ORIGINAL_LLM_SELECTED_SET_IDENTITY)"]},
    {"lock": SOURCE_PAIR_PARITY_LOCK_PATH,
     "depends_on": [RECIPE_PAIR_LOCK_PATH + " (same original+shuffle recipe lists)",
                    "reports/full/c5/C5_SOURCE_PAIR_PLAN.json (ordinal alignment)"]},
    {"lock": RENDER_PARITY_LOCK_PATH,
     "depends_on": [RECIPE_PAIR_LOCK_PATH + " (same original+shuffle recipe lists)",
                    "c_ext_e6_render.audit_historical_path (frozen renderer/quality/matching contract)"]},
    {"lock": QUALITY_PARITY_LOCK_PATH,
     "depends_on": ["c_ext_e6_render.audit_historical_path", "c_ext_e6_render.resolve_e6_route_quota",
                    "c_ext_e6_render.default_quality_matcher's signature (arm parameterization)"]},
    {"lock": TRAINING_PLAN_LOCK_PATH,
     "depends_on": ["reports/full/c7/DETECTOR_CONFIG_LOCK.json",
                    "configs/models/m9_detector.yaml", "configs/train/m9_reference.yaml"]},
    {"lock": PROTOCOL_LOCK_PATH,
     "depends_on": [ROOT_CAUSE_PATH + " (forensic status citation only)",
                    "policy-level only -- does not numerically depend on the other five locks"]},
)


def describe_lock_dependency_chain(locks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """BLOCKER 3: prints the real dependency edges, and cross-checks that the
    recipe identities RECIPE_PAIR_LOCK recorded are the SAME ones the
    dependent locks' own (independently recomputed) recipe reads produced --
    proving the chain is coherent, not merely declared."""
    recipe_lock = locks.get(RECIPE_PAIR_LOCK_PATH) or {}
    render_lock = locks.get(RENDER_PARITY_LOCK_PATH) or {}

    original_identity_recipe_pair = recipe_lock.get("original_recipe_identity_file_sha_v2")
    shuffle_identity_recipe_pair = recipe_lock.get("shuffle_recipe_identity")
    original_identity_render = (render_lock.get("original_plan") or {}).get("recipe_content_identity")
    shuffle_identity_render = (render_lock.get("shuffle_plan") or {}).get("recipe_content_identity")

    original_consistent = (original_identity_recipe_pair is not None
                           and original_identity_recipe_pair == original_identity_render)
    shuffle_consistent = (shuffle_identity_recipe_pair is not None
                          and shuffle_identity_recipe_pair == shuffle_identity_render)

    return {
        "schema_version": "e6-v2-lock-dependency-chain-v1",
        "chain": list(LOCK_DEPENDENCY_CHAIN),
        "cross_check": {
            "original_recipe_identity_consistent_across_recipe_pair_and_render_parity": original_consistent,
            "shuffle_recipe_identity_consistent_across_recipe_pair_and_render_parity": shuffle_consistent,
        },
        "lock_dependency_chain_valid": bool(original_consistent and shuffle_consistent
                                            and all(is_usable_v2_lock(locks.get(path))
                                                    for path in (RECIPE_PAIR_LOCK_PATH, RENDER_PARITY_LOCK_PATH)
                                                    if locks.get(path) is not None)),
    }


def compute_readiness_gate(locks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    protocol_locked = is_usable_v2_lock(locks[PROTOCOL_LOCK_PATH])
    recipe_lock = locks[RECIPE_PAIR_LOCK_PATH]
    original_recipe_content_equivalence_proven = (
        recipe_lock.get("original_recipe_content_equivalence") == "PROVEN")
    recipe_field_marginal_parity_pass = recipe_lock.get("recipe_field_marginal_parity") == "PASS"
    joint_associations_changed = recipe_lock.get("joint_associations_changed") is True
    recipe_pass = (is_usable_v2_lock(recipe_lock)
                  and recipe_lock.get("recipe_counts_match") is True
                  and original_recipe_content_equivalence_proven
                  and recipe_field_marginal_parity_pass
                  and joint_associations_changed)
    source_pair_pass = is_usable_v2_lock(locks[SOURCE_PAIR_PARITY_LOCK_PATH])
    render_pass = is_usable_v2_lock(locks[RENDER_PARITY_LOCK_PATH])
    quality_pass = is_usable_v2_lock(locks[QUALITY_PARITY_LOCK_PATH])
    training_locked = is_usable_v2_lock(locks[TRAINING_PLAN_LOCK_PATH])
    dependency_chain = describe_lock_dependency_chain(locks)

    all_pass = all([protocol_locked, recipe_pass, source_pair_pass, render_pass, quality_pass,
                    training_locked, dependency_chain["lock_dependency_chain_valid"]])
    return {
        "schema_version": "e6-v2-readiness-gate-v1",
        "E6_V2_PROTOCOL_LOCKED": protocol_locked,
        "E6_V2_RECIPE_PARITY_PASS": recipe_pass,
        "ORIGINAL_RECIPE_CONTENT_EQUIVALENCE": original_recipe_content_equivalence_proven,
        "RECIPE_FIELD_MARGINAL_PARITY": "PASS" if recipe_field_marginal_parity_pass else "FAIL",
        "JOINT_ASSOCIATIONS_CHANGED": joint_associations_changed,
        "E6_V2_SOURCE_PAIR_PARITY_PASS": source_pair_pass,
        "E6_V2_RENDER_PARITY_PASS": render_pass,
        "E6_V2_QUALITY_PARITY_PASS": quality_pass,
        "E6_V2_TRAINING_PLAN_LOCKED": training_locked,
        "LOCK_DEPENDENCY_CHAIN_VALID": dependency_chain["lock_dependency_chain_valid"],
        "lock_dependency_chain": dependency_chain,
        "E6_V2_READY_FOR_RENDER": all_pass,
        "blocking_locks": [path for path, payload in locks.items() if not is_usable_v2_lock(payload)],
        "note": "E6_V2_READY_FOR_RENDER=TRUE authorizes nothing by itself -- rendering still requires a "
               "separate, explicit --execute --authorize-gpu-render invocation this milestone does not "
               "provide yet. The historical field E6_READY_FOR_REAL_RENDER (c_ext_e6_render) remains "
               "FALSE regardless of this gate, because the OLD historical-parity path stays "
               f"{OLD_E6_PATH_STATUS}.",
    }


def run_e6_v2_protocol_preparation(repo: Path) -> dict[str, Any]:
    """Builds and writes all six locks PLUS the render-execution plan lock,
    additive-only, under E6_V2_DIR. Never renders, trains, or touches
    target/LLM. Returns the full readiness report.

    This is the ONE canonical WRITING preparation operation for
    E6_V2_RENDER_EXECUTION_PLAN_LOCK.json (`write_render_execution_plan_lock`)
    -- neither preflight mode writes it, and the render-execution path only
    ever VALIDATES it against this persisted copy, never (re)writes it.
    """
    out_dir = repo / E6_V2_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    locks: dict[str, dict[str, Any]] = {}
    for rel_path, builder in LOCK_BUILDERS.items():
        lock = builder(repo)
        locks[rel_path] = lock
        (repo / rel_path).write_text(json.dumps(lock, indent=2, default=str), encoding="utf-8")

    readiness = compute_readiness_gate(locks)

    # freeze the render-execution plan lock too -- the final artifact
    # GPU_RUNTIME_PREFLIGHT and --execute both require to be PERSISTED,
    # never merely rebuildable in memory.
    execution_plan = write_render_execution_plan_lock(repo)

    summary = {
        "schema_version": "e6-v2-preparation-summary-v1",
        "locks_written": list(LOCK_BUILDERS.keys()) + [RENDER_EXECUTION_PLAN_LOCK_PATH],
        "readiness": readiness,
        "execution_plan_lock_identity": execution_plan["lock"].get("lock_identity"),
        "execution_plan_lock_status": execution_plan["lock"].get("status"),
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False,
        "training_performed": False,
    }
    summary_path = out_dir / "E6_V2_PREPARATION_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return {"locks": locks, "readiness": readiness, "summary_path": str(summary_path),
           "execution_plan_lock": execution_plan}


# --------------------------------------------------------------------------- #
# EXECUTION: two-arm render entry point.
#
# CODE + PREFLIGHT + TESTS ONLY THIS TURN -- nothing below is invoked against
# the real repo in this session. Every function that would touch a real
# candidate is exercised only via injected fakes in tests, exactly like
# c_ext_e6_render.py's own established pattern (`candidate_renderer`,
# `metrics_provider`).
# --------------------------------------------------------------------------- #

E6_V2_CANDIDATES_ROOT = f"{E6_V2_RUN_ROOT}/candidates"
E6_V2_MATCHED_BANK_ROOT = "data/processed/c_ext_q1q2_v1/e6_paired_current_runtime_v2/matched_bank"
RENDER_EXECUTION_PLAN_LOCK_PATH = f"{E6_V2_DIR}/E6_V2_RENDER_EXECUTION_PLAN_LOCK.json"


def v2_candidates_root(arm: str) -> str:
    """The FINAL per-arm directory that directly holds `<candidate_id>/`
    subdirectories -- informational/metadata value only (used in reports,
    the execution-plan lock's per-arm contract, and writability checks).
    NEVER pass this to `c5_render.render_arm`/`c5_raw_generation.candidate_dir`
    as `work_root` -- both of those append `arm` themselves, exactly once;
    doing so was ATTEMPT-1's proven double-arm-append bug. Use
    `v2_render_work_root(arm)` for that purpose instead."""
    return f"{E6_V2_CANDIDATES_ROOT}/{arm}"


#: TASK B/J recovery (ATTEMPT 1): ORIGINAL's `render_v2_arm` call passed
#: `v2_candidates_root(ARM_ORIGINAL)` (already arm-inclusive) AS `work_root`
#: to `c5_render.render_arm`, which appends `plan["arm"]` again via
#: `c5_raw_generation.candidate_dir(work_root, arm, candidate_id)` ==
#: `work_root / arm / candidate_id`. The result -- proven on the real GPU
#: host -- is the DOUBLE-nested physical layout
#: `.../candidates/LLM_ORIGINAL_CURRENT_V2/LLM_ORIGINAL_CURRENT_V2/<id>/`,
#: which already holds 2048 real records (2045 GENERATED + 3
#: FAILED_GENERATION). Per TASK B recovery option A ("existing valid
#: rendered bytes > regeneration"), ORIGINAL keeps addressing this EXACT
#: nested root going forward -- it is NOT silently "fixed" to match SHUFFLE,
#: because doing so would make `reuse_decision` find nothing at the new path
#: and re-render all 2048 candidates from scratch, discarding real GPU work
#: for a path-cosmetics-only reason. This is a PERMANENT, DOCUMENTED,
#: single-arm exception -- never a general per-call override (see TASK G:
#: storage layout must never become a scientific treatment difference).
ATTEMPT1_ORIGINAL_RECOVERY_WORK_ROOT = v2_candidates_root(ARM_ORIGINAL)


def v2_render_work_root(arm: str) -> str:
    """The CORRECTED `work_root` value that must be passed to
    `c5_render.render_arm` (or an injected `render_arm_fn`) for `arm` --
    arm-INDEPENDENT for every arm except the documented ATTEMPT-1 ORIGINAL
    recovery exception above. `render_arm`/`candidate_dir` append `arm`
    exactly once on top of whatever this returns; nothing else may append it
    again."""
    if arm == ARM_ORIGINAL:
        return ATTEMPT1_ORIGINAL_RECOVERY_WORK_ROOT
    return E6_V2_CANDIDATES_ROOT


def v2_bank_lock_path(arm: str) -> str:
    return f"{E6_V2_MATCHED_BANK_ROOT}/E6_V2_BANK_LOCK_{arm}.json"


class E6V2ExecutionError(RuntimeError):
    """A precondition for E6-v2 render EXECUTION failed. Fails closed."""


def build_v2_arm_plan_rows(repo: Path, *, arm: str, recipe_bank_identity: str,
                           recipes: list[dict[str, Any]], plan: dict[str, Any]) -> list[dict[str, Any]]:
    """TASK A/D: a generic, arm-parameterized analog of
    `c_ext_e6_render.build_arm_plan_rows` -- reuses the SAME low-level
    primitives verbatim (`build_source_pair_plan`, `candidate_identity`,
    `_recipe_id`, `_assert_arm_plan`, all already arm-agnostic), but never
    hardcodes `E6_ARM_NAME`. The old wrapper cannot be reused as-is: it
    hardcodes `arm=E6_ARM_NAME` into every candidate_identity() call and
    every row's own `arm` field."""
    from prism_fas.synthesis.c5_arm_plan import _assert_arm_plan, _recipe_id
    from prism_fas.synthesis.c5_source_pair_plan import (PLAN_SEED, build_source_pair_plan,
                                                         candidate_identity, source_pair_plan_identity)

    package_root = repo / "data/packages/prism_data_v1_m3b"
    manifest_path = package_root / "manifests/source_train.parquet"
    if not manifest_path.is_file():
        raise E6V2ExecutionError(
            f"missing {manifest_path.as_posix()}; the real source_train manifest is required to "
            "recompute the base candidate schedule and is GPU-only on this laptop -- refusing to "
            "fabricate the 2048-position schedule")
    base_plan = build_source_pair_plan(package_root, seed=PLAN_SEED)
    recomputed_identity = source_pair_plan_identity(base_plan)
    if recomputed_identity != plan["source_pair_plan_identity"]:
        raise E6V2ExecutionError(
            f"recomputed source_pair_plan_identity {recomputed_identity!r} != the pinned "
            f"{plan['source_pair_plan_identity']!r}; the base schedule has drifted or could not be "
            "reproduced")
    positions = base_plan["positions"]
    rows = []
    for row in positions:
        ordinal = int(row["recipe_ordinal"])
        recipe_id = _recipe_id(recipes[ordinal], ordinal)
        binding = (plan["gpat_checkpoint_sha256"] if row["route"] == "gpat"
                  else plan["physics_engine_version"])
        candidate_id = candidate_identity(
            source_pair_plan_identity=plan["source_pair_plan_identity"], arm=arm,
            recipe_bank_identity=recipe_bank_identity, recipe_id=recipe_id,
            recipe_ordinal=ordinal, slot=int(row["slot"]), position=int(row["position"]),
            route=row["route"], live_target_sample_id=row["live_target_sample_id"],
            spoof_source_sample_id=row["spoof_source_sample_id"],
            package_identity=plan["package_identity"], ontology_identity=plan["ontology_identity"],
            generator_binding=binding)
        rows.append({**row, "arm": arm, "recipe_id": recipe_id, "candidate_id": candidate_id,
                    "recipe_bank_identity": recipe_bank_identity, "generator_binding": binding})
    _assert_arm_plan(rows, arm)
    return rows


#: TASK D: fields that MUST be identical between ORIGINAL and SHUFFLE at
#: every schedule position -- candidate_id/recipe_id are deliberately
#: EXCLUDED (expected to differ, since recipe content differs).
_SOURCE_PAIR_EXECUTION_FIELDS: tuple[str, ...] = (
    "position", "slot", "route", "live_target_sample_id", "spoof_source_sample_id", "generator_binding")


def resolve_source_pair_execution_parity(original_rows: list[dict[str, Any]],
                                         shuffle_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """TASK D: for every planned schedule position, asserts ORIGINAL and
    SHUFFLE rows share the identical live/spoof sample, route, slot,
    position and generator_binding. Must be checked, and must pass, BEFORE
    the first candidate of either arm is rendered."""
    by_position_original = {row["position"]: row for row in original_rows}
    by_position_shuffle = {row["position"]: row for row in shuffle_rows}

    mismatches: list[dict[str, Any]] = []
    if len(original_rows) != len(shuffle_rows):
        mismatches.append({"reason": "ROW_COUNT_MISMATCH", "original_count": len(original_rows),
                          "shuffle_count": len(shuffle_rows)})

    checked = 0
    aligned_positions = 0
    for position in sorted(by_position_original):
        original_row, shuffle_row = by_position_original[position], by_position_shuffle.get(position)
        checked += 1
        if shuffle_row is None:
            mismatches.append({"position": position, "reason": "MISSING_IN_SHUFFLE"})
            continue
        position_ok = True
        for field in _SOURCE_PAIR_EXECUTION_FIELDS:
            if original_row.get(field) != shuffle_row.get(field):
                mismatches.append({"position": position, "field": field,
                                  "original": original_row.get(field), "shuffle": shuffle_row.get(field)})
                position_ok = False
        if position_ok:
            aligned_positions += 1

    return {
        "schema_version": "e6-v2-source-pair-execution-parity-v1",
        "positions_checked": checked,
        "positions_aligned": aligned_positions,
        "source_pair_execution_parity_pct": (100.0 * aligned_positions / checked) if checked else 0.0,
        "all_positions_aligned": not mismatches,
        "mismatches": mismatches[:50],
        "candidate_id_and_recipe_id_expected_to_differ": True,
        "fields_checked": list(_SOURCE_PAIR_EXECUTION_FIELDS),
    }


def build_e6_v2_route_bank(repo: Path, recipes: list[dict[str, Any]], *, arm: str,
                           bank_identity: str) -> dict[str, Any]:
    """Generic, arm-parameterized analog of `c_ext_e6_render.build_e6_route_bank`."""
    from prism_fas.recipes.ontology import load_ontology
    from prism_fas.recipes.schema import parse_recipe

    ontology = load_ontology(repo / e6r.ONTOLOGY_CONFIG_PATH)
    if ontology.sha256 != e6r.EXPECTED_ONTOLOGY_IDENTITY:
        raise E6V2ExecutionError(
            f"resolved ontology identity {ontology.sha256!r} != pinned "
            f"{e6r.EXPECTED_ONTOLOGY_IDENTITY!r}")
    parsed = [parse_recipe(recipe) for recipe in recipes]
    return {"recipes": parsed, "bank_id": f"e6v2_{arm.lower()}", "bank_identity": bank_identity,
           "ontology": ontology, "ontology_identity": ontology.sha256}


def render_v2_arm(*, repo: Path, arm: str, plan: dict[str, Any], recipes: list[dict[str, Any]],
                  render_arm_fn: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """TASK C/H/I: reuses `synthesis.c5_render.render_arm` VERBATIM -- the
    SAME production per-candidate resume-safe/failure-retaining/counting
    loop the historical C5 pass already uses -- never a reimplementation.
    Only `plan['arm']`/`plan['candidates']` (built by
    `build_v2_arm_plan_rows`) make it apply to a second, differently-named
    arm. `render_arm_fn` is the ONE injectable GPU-only seam (defaults to
    the real `c5_render.render_arm`); every test injects a fake."""
    from prism_fas.synthesis import c5_render

    renderer = render_arm_fn or c5_render.render_arm
    rows = build_v2_arm_plan_rows(repo, arm=arm, recipe_bank_identity=plan["recipe_bank_identity"],
                                  recipes=recipes, plan=plan)
    run_plan = {**plan, "arm": arm, "candidates": rows}
    # ATTEMPT-1 BUG FIX: work_root must be ARM-INDEPENDENT -- render_arm/
    # candidate_dir append `arm` themselves. v2_candidates_root(arm) is
    # already arm-inclusive and must never be passed here (that was the
    # proven double-append bug). v2_render_work_root(arm) is the corrected
    # value, with the one documented ORIGINAL recovery exception.
    work_root = repo / v2_render_work_root(arm)

    if renderer is c5_render.render_arm:
        runtime = e6r.resolve_render_runtime_objects(repo)
        bank = build_e6_v2_route_bank(repo, recipes, arm=arm, bank_identity=plan["recipe_bank_identity"])
        result = renderer(work_root=work_root, plan=run_plan, store=runtime["store"],
                          bank=bank, routes=runtime["routes"])
    else:
        result = renderer(work_root=work_root, plan=run_plan)
    return {"rows": rows, "candidates_root": str(work_root / arm), **result}


def stage_v2_results_for_quality(repo: Path, *, arm: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reads back each row's persisted CANDIDATE.json (whichever way it got
    there -- freshly rendered, resumed, or a retained failure) so quality
    matching always reflects what is actually on disk, never an in-memory
    assumption. Rows without a GENERATED record are excluded (a retained
    failure has no payload to score)."""
    from prism_fas.synthesis import c5_raw_generation as raw

    # SAME corrected root render_v2_arm uses -- never v2_candidates_root(arm)
    # directly (that is already arm-inclusive; candidate_dir appends arm again).
    work_root = repo / v2_render_work_root(arm)
    staged_rows, staged_results = [], []
    for row in rows:
        directory = raw.candidate_dir(work_root, arm, row["candidate_id"])
        record = raw.read_record(directory / raw.RECORD_NAME)
        if record is None or record.get("status") != raw.GENERATED:
            continue
        staged_rows.append(row)
        staged_results.append(record)
    return {"rows": staged_rows, "results": staged_results}


def match_v2_arm(repo: Path, *, arm: str, plan: dict[str, Any], rows: list[dict[str, Any]],
                 recipes: list[dict[str, Any]],
                 metrics_provider: Callable[..., dict[str, Any]] | None = None,
                 quality_matcher: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """TASK C/J: the SAME CURRENT quality pipeline + the arm-parameterized
    `default_quality_matcher(..., arm=<v2 arm>)`, run independently per arm.
    Explicitly supplies `candidates_root=v2_render_work_root(arm)` so
    `default_metrics_provider` (when used as the real metrics_provider) reads
    the ACTUAL v2 candidate bytes -- it NEVER falls back to
    `c_ext_e6_render.CANDIDATES_ROOT`/`E6_ARM_NAME` (the historical E6 root),
    which was ATTEMPT-1's second proven bug. Never reads historical q as
    input; never rewrites historical C6.

    `recipes` (the SAME recipe list `render_v2_arm` rendered this arm's
    candidates from) builds an explicit `quality_bank` via
    `build_e6_v2_route_bank`, so `requested_support_for`/`reconstruct_discrete`
    reconstruct each candidate's requested support mask/strength against
    THIS arm's OWN recipe content, keyed the same way `row["recipe_id"]`
    already is. Passing no bank (the historical E6 v1 call shape) silently
    fell back to LLM-SHUFFLE-A's bank for EVERY arm including
    `LLM_ORIGINAL_CURRENT_V2` -- ATTEMPT-2's proven third bug.

    `quality_bank` is built EAGERLY (real ontology load + recipe parse) only
    when it will actually reach `default_metrics_provider` -- i.e. only when
    the resolved `metrics_provider` is that real function too (mirroring
    `default_quality_matcher`'s own internal `provider is default_metrics_provider`
    check exactly). A caller-injected fake `metrics_provider` (every test)
    never needs a real bank and must never pay for building one."""
    matcher = quality_matcher or e6r.default_quality_matcher
    staged = stage_v2_results_for_quality(repo, arm=arm, rows=rows)
    resolved_provider = metrics_provider or e6r.default_metrics_provider
    if matcher is e6r.default_quality_matcher:
        kwargs: dict[str, Any] = {"metrics_provider": metrics_provider,
                                  "candidates_root": repo / v2_render_work_root(arm)}
        if resolved_provider is e6r.default_metrics_provider:
            kwargs["quality_bank"] = build_e6_v2_route_bank(
                repo, recipes, arm=arm, bank_identity=plan["recipe_bank_identity"])
    else:
        kwargs = {}
    return matcher(repo=repo, plan=plan, staged=staged, arm=arm, **kwargs)


def build_v2_matched_bank_lock(*, arm: str, plan: dict[str, Any], selected: list[dict[str, Any]]
                               ) -> dict[str, Any]:
    """Generic, arm-parameterized analog of `c_ext_e6_render.build_matched_bank_lock`."""
    route_counts: dict[str, int] = {}
    for row in selected:
        route = str(row.get("route", ""))
        if route:
            route_counts[route] = route_counts.get(route, 0) + 1
    body = {
        "schema_version": "e6-v2-bank-lock-v1", "arm": arm,
        "recipe_bank_identity": plan["recipe_bank_identity"],
        "source_package_identity": plan["package_identity"],
        "source_pair_plan_identity": plan["source_pair_plan_identity"],
        "ontology_identity": plan["ontology_identity"],
        "gpat_checkpoint_sha256": plan["gpat_checkpoint_sha256"],
        "physics_engine_version": plan["physics_engine_version"],
        "candidate_count": plan["candidates_per_arm"],
        "quality_threshold_identity": plan["quality_threshold_identity"],
        "final_bank_size": len(selected), "route_counts": route_counts,
        "by_route": plan["by_route_quota"],
        "selected": selected,
        "historical_q_used_as_input": False,
        "target_access": False, "llm_api_calls": 0, "status": "FROZEN",
    }
    body["selected_set_sha256"] = cc.sha256_json(selected)
    body["bank_lock_identity"] = cc.sha256_bytes(cc.canonical_json_bytes(body))
    return {**body, "lock_identity": body["bank_lock_identity"]}


def build_render_execution_plan_lock(repo: Path) -> dict[str, Any]:
    """TASK F: pins every upstream v2 lock's identity PLUS a per-arm
    execution contract and the execution namespace/resume/failure policy.
    Rebuilds all six upstream locks FRESH (never trusts a stale file on
    disk) so this lock always reflects a live revalidation."""
    locks = {path: builder(repo) for path, builder in LOCK_BUILDERS.items()}
    readiness = compute_readiness_gate(locks)

    original = load_original_llm_recipes(repo)
    shuffle = e6r.verify_shuffle_recipe_source(repo)
    contract = e6r.audit_historical_path(repo)

    planned_total = contract["candidates_per_arm"]
    planned_per_route = planned_total // 2  # renders_per_recipe=8 -> 4 physics + 4 gpat per recipe

    def _arm_execution_contract(*, arm: str, recipe_content_identity: str,
                                recipe_semantic_identity: str, recipe_count: int) -> dict[str, Any]:
        return {
            "arm": arm,
            "recipe_semantic_identity": recipe_semantic_identity,
            "recipe_content_identity": recipe_content_identity,
            "recipe_count": recipe_count,
            "renders_per_recipe": contract["renders_per_recipe"],
            "planned_candidate_count": planned_total,
            "planned_physics_count": planned_per_route,
            "planned_gpat_count": planned_per_route,
            "source_package_identity": contract["package_identity"],
            "source_pair_plan_identity": contract["source_pair_plan_identity"],
            "ontology_identity": contract["ontology_identity"],
            "gpat_checkpoint_sha256": contract["gpat_checkpoint_sha256"],
            "physics_engine_version": contract["physics_engine_version"],
            "candidates_root": v2_candidates_root(arm),
            "bank_lock_path": v2_bank_lock_path(arm),
        }

    per_arm = {
        ARM_ORIGINAL: _arm_execution_contract(
            arm=ARM_ORIGINAL, recipe_content_identity=original["content_identity"],
            recipe_semantic_identity=e6r.EXPECTED_ORIGINAL_LLM_SELECTED_SET_IDENTITY,
            recipe_count=original["recipe_count"]),
        ARM_SHUFFLE: _arm_execution_contract(
            arm=ARM_SHUFFLE, recipe_content_identity=shuffle["content_identity"],
            recipe_semantic_identity=shuffle["content_identity"], recipe_count=len(shuffle["recipes"])),
    }

    status = "FROZEN" if readiness["E6_V2_READY_FOR_RENDER"] else "BLOCKED"
    body = {
        "schema_version": "e6-v2-render-execution-plan-lock-v1", "status": status,
        "readiness_at_lock_time": readiness,
        "upstream_lock_identities": {
            "protocol": locks[PROTOCOL_LOCK_PATH]["lock_identity"],
            "recipe_pair": locks[RECIPE_PAIR_LOCK_PATH]["lock_identity"],
            "source_pair_parity": locks[SOURCE_PAIR_PARITY_LOCK_PATH]["lock_identity"],
            "render_parity": locks[RENDER_PARITY_LOCK_PATH]["lock_identity"],
            "quality_parity": locks[QUALITY_PARITY_LOCK_PATH]["lock_identity"],
            "training_plan": locks[TRAINING_PLAN_LOCK_PATH]["lock_identity"],
        },
        "per_arm": per_arm,
        "execution_namespace": {
            "runs_root": E6_V2_RUN_ROOT, "candidates_root": E6_V2_CANDIDATES_ROOT,
            "matched_bank_root": E6_V2_MATCHED_BANK_ROOT, "reports_root": E6_V2_DIR,
            "state_root": E6_V2_STATE_DIR,
        },
        "resume_policy": "identity-aware: a candidate directory whose GenerationIdentity agrees and "
                         "whose payload hashes still verify (c5_raw_generation.reuse_decision) is "
                         "reused, never re-rendered; a missing/altered candidate is (re)built under "
                         "the SAME identity; arm-specific state never collides because each arm has "
                         "its own candidates_root.",
        "failure_policy": "a genuine terminal generation failure (SemanticGenerationFailure, e.g. an "
                          "empty exact mask) is retained and reported, never resampled, never "
                          "silently replaced with a different source pair -- this is "
                          "synthesis.c5_render.render_arm's own historical policy, reused verbatim "
                          "and applied symmetrically to both arms.",
        "target_access": False, "llm_api_calls": 0,
    }
    body["lock_identity"] = cc.sha256_bytes(cc.canonical_json_bytes(body))
    return body


def is_usable_execution_plan_lock(payload: dict[str, Any]) -> bool:
    return isinstance(payload, dict) and payload.get("status") == "FROZEN"


def load_persisted_execution_plan_lock(repo: Path) -> dict[str, Any] | None:
    """Reads the PERSISTED E6_V2_RENDER_EXECUTION_PLAN_LOCK.json from disk if
    present -- never fabricates one, never substitutes a freshly-rebuilt
    in-memory object in its place."""
    path = repo / RENDER_EXECUTION_PLAN_LOCK_PATH
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_render_execution_plan_lock(repo: Path) -> dict[str, Any]:
    """TASK B: the ONE writing operation that materializes/freezes
    E6_V2_RENDER_EXECUTION_PLAN_LOCK.json. Called ONLY from
    `run_e6_v2_protocol_preparation` (`--prepare-protocol`) -- neither
    `structural_preflight_v2` nor `gpu_runtime_preflight_v2` ever calls this;
    both are strictly read-only with respect to this file."""
    lock = build_render_execution_plan_lock(repo)
    out_dir = repo / E6_V2_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "E6_V2_RENDER_EXECUTION_PLAN_LOCK.json"
    path.write_text(json.dumps(lock, indent=2, default=str), encoding="utf-8")
    return {"lock": lock, "path": str(path)}


def verify_execution_plan_lock_matches_expected(repo: Path) -> dict[str, Any]:
    """TASK C/D/E: the ONE place that decides whether the PERSISTED
    execution-plan lock may be trusted. Loads it from disk (never fabricates
    a substitute when absent), independently rebuilds the EXPECTED lock
    fresh from current repo state (`build_render_execution_plan_lock`, the
    SAME pure builder), and requires exact agreement. Strictly read-only:
    never writes."""
    persisted = load_persisted_execution_plan_lock(repo)
    if persisted is None:
        return {
            "schema_version": "e6-v2-execution-plan-lock-verification-v1",
            "EXECUTION_PLAN_LOCK_PRESENT": False,
            "EXECUTION_PLAN_LOCK_IDENTITY": None,
            "EXECUTION_PLAN_LOCK_EXPECTED_EQUALS_PERSISTED": False,
            "persisted_status": None,
            "reason": f"missing {(repo / RENDER_EXECUTION_PLAN_LOCK_PATH).as_posix()} -- run "
                     "--prepare-protocol first; a missing persisted lock is never synthesized or "
                     "treated as valid",
        }

    try:
        expected = build_render_execution_plan_lock(repo)
    except Exception as error:  # noqa: BLE001
        return {
            "schema_version": "e6-v2-execution-plan-lock-verification-v1",
            "EXECUTION_PLAN_LOCK_PRESENT": True,
            "EXECUTION_PLAN_LOCK_IDENTITY": persisted.get("lock_identity"),
            "EXECUTION_PLAN_LOCK_EXPECTED_EQUALS_PERSISTED": False,
            "persisted_status": persisted.get("status"),
            "reason": f"could not rebuild the expected plan to compare against: {error}",
        }

    matches = persisted == expected
    return {
        "schema_version": "e6-v2-execution-plan-lock-verification-v1",
        "EXECUTION_PLAN_LOCK_PRESENT": True,
        "EXECUTION_PLAN_LOCK_IDENTITY": persisted.get("lock_identity"),
        "EXECUTION_PLAN_LOCK_EXPECTED_EQUALS_PERSISTED": matches,
        "persisted_status": persisted.get("status"),
        "reason": None if matches else (
            "the persisted lock no longer matches a freshly rebuilt expected plan -- repo state has "
            "drifted since --prepare-protocol was last run (or the persisted file was tampered with); "
            "re-run --prepare-protocol before rendering"),
    }


def structural_preflight_v2(repo: Path) -> dict[str, Any]:
    """TASK M: CPU-only, no-CUDA-required structural preflight -- verifies
    importability, file/path presence and all six v2 locks + the execution
    plan lock, WITHOUT declaring GPU hardware available (that is
    `gpu_runtime_preflight_v2`'s job, and it is never run here)."""
    checks: dict[str, Any] = {}

    try:
        from prism_fas.synthesis import c5_render as _c5_render  # noqa: F401
        from prism_fas.synthesis import c5_raw_generation as _raw  # noqa: F401
        checks["gpu_render_implementation_importable"] = True
    except Exception as error:  # noqa: BLE001
        checks["gpu_render_implementation_importable"] = False
        checks["gpu_render_implementation_import_error"] = str(error)

    c4_path = repo / e6r.C4_SCIENTIFIC_LOCK_PATH
    checks["gpat_checkpoint_lock_present"] = c4_path.is_file()
    if c4_path.is_file():
        c4_lock = json.loads(c4_path.read_text(encoding="utf-8"))
        checks["gpat_checkpoint_sha256_recorded"] = bool(c4_lock.get("winning_checkpoint_sha256"))

    checks["physics_backend_importable"] = True
    try:
        from prism_fas.synthesis.physics import PhysicsEngine as _PhysicsEngine  # noqa: F401
    except Exception as error:  # noqa: BLE001
        checks["physics_backend_importable"] = False
        checks["physics_backend_import_error"] = str(error)

    try:
        from prism_fas.synthesis import quality_models as _quality_models  # noqa: F401
        checks["quality_backend_module_importable"] = True
    except Exception as error:  # noqa: BLE001
        checks["quality_backend_module_importable"] = False
        checks["quality_backend_import_error"] = str(error)

    checks["source_package_present"] = (repo / "data/packages/prism_data_v1_m3b").is_dir()
    checks["original_recipe_file_present"] = (repo / e6r.RECIPE_BANK_LLM_JSONL_PATH).is_file()
    checks["shuffle_recipe_file_present"] = (repo / training_plan.E6_SHUFFLE_RECIPES_PATH).is_file()

    lock_status: dict[str, Any] = {}
    for path in LOCK_BUILDERS:
        lock_file = repo / path
        lock_status[path] = lock_file.is_file()
    checks["v2_locks_present"] = lock_status
    checks["all_six_v2_locks_present"] = all(lock_status.values())

    execution_plan_path = repo / RENDER_EXECUTION_PLAN_LOCK_PATH
    checks["execution_plan_lock_present"] = execution_plan_path.is_file()

    for arm in (ARM_ORIGINAL, ARM_SHUFFLE):
        candidates_dir = repo / v2_candidates_root(arm)
        try:
            candidates_dir.mkdir(parents=True, exist_ok=True)
            writable = True
        except OSError:
            writable = False
        checks[f"candidates_namespace_writable_{arm}"] = writable

    checks["target_dependency"] = "NONE -- no target path referenced anywhere in this preflight"
    checks["llm_dependency"] = "NONE -- no LLM/API path referenced anywhere in this preflight"

    passed = (checks.get("gpu_render_implementation_importable") is True
             and checks.get("gpat_checkpoint_lock_present") is True
             and checks.get("physics_backend_importable") is True
             and checks.get("quality_backend_module_importable") is True
             and checks.get("source_package_present") is True
             and checks.get("original_recipe_file_present") is True
             and checks.get("shuffle_recipe_file_present") is True
             and checks.get("all_six_v2_locks_present") is True
             and checks.get("execution_plan_lock_present") is True)

    return {
        "schema_version": "e6-v2-structural-preflight-v1",
        "checks": checks,
        "structural_preflight": "PASS" if passed else "FAIL",
        "gpu_hardware_declared_available": False,
        "note": "this preflight NEVER declares GPU hardware available -- it only checks CPU-visible "
               "structural preconditions (importability, files, locks, writable paths). Actual CUDA "
               "availability is GPU_RUNTIME_PREFLIGHT's job, deliberately NOT run here.",
    }


# --------------------------------------------------------------------------- #
# GPU_RUNTIME_PREFLIGHT (this turn): the REAL runtime checks, meant to run on
# the GPU host. Never renders, never creates a candidate directory, never
# invokes Physics/GPAT generation or quality inference on scientific inputs,
# never trains, never touches target, never calls an LLM. On THIS laptop it
# correctly, honestly reports FAIL (no CUDA) -- that FAIL is a real, observed
# laptop result demonstrating fail-closed behavior, never presented as what
# the GPU host would report.
# --------------------------------------------------------------------------- #

def _check_cuda_hardware() -> dict[str, Any]:
    """TASK B: torch import, CUDA availability, device count/name/version --
    never hardcodes a specific GPU model."""
    try:
        import torch
    except Exception as error:  # noqa: BLE001
        return {"torch_importable": False, "torch_import_error": str(error),
               "CUDA_AVAILABLE": False, "device_count": 0, "selected_device": None,
               "device_name": None, "torch_cuda_version": None}

    available = bool(torch.cuda.is_available())
    device_count = torch.cuda.device_count() if available else 0
    device_name = None
    if available and device_count > 0:
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            device_name = None
    return {
        "torch_importable": True,
        "CUDA_AVAILABLE": available,
        "device_count": device_count,
        "selected_device": "cuda:0" if available and device_count > 0 else None,
        "device_name": device_name,
        "torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
    }


def _check_gpat_runtime(repo: Path, *, cuda: dict[str, Any] | None = None) -> dict[str, Any]:
    """TASK C: checkpoint presence/hash, runtime importability, GPU binding
    readiness -- stops at deterministic load/import/hash verification, never
    constructs a GPATRoute (which would load real weights and require CUDA
    memory) and never renders."""
    result: dict[str, Any] = {"GPAT_CHECKPOINT_PRESENT": False, "GPAT_CHECKPOINT_SHA_MATCH": False,
                              "GPAT_RUNTIME_IMPORTABLE": False, "GPAT_GPU_BINDING_READY": False,
                              "expected_checkpoint_sha256": e6r.EXPECTED_GPAT_CHECKPOINT_SHA256}
    c4_path = repo / e6r.C4_SCIENTIFIC_LOCK_PATH
    if not c4_path.is_file():
        result["reason"] = f"missing {c4_path.as_posix()}"
        return result
    c4_lock = json.loads(c4_path.read_text(encoding="utf-8"))
    expected_sha = c4_lock.get("winning_checkpoint_sha256")
    if expected_sha != e6r.EXPECTED_GPAT_CHECKPOINT_SHA256:
        result["reason"] = "C4 lock's winning_checkpoint_sha256 disagrees with the pinned expectation"
        return result
    checkpoint_rel = c4_lock.get("winning_checkpoint")
    if not checkpoint_rel:
        result["reason"] = "C4 lock carries no winning_checkpoint path"
        return result
    checkpoint_path = repo / checkpoint_rel
    result["checkpoint_path"] = checkpoint_rel
    result["GPAT_CHECKPOINT_PRESENT"] = checkpoint_path.is_file()
    if result["GPAT_CHECKPOINT_PRESENT"]:
        result["GPAT_CHECKPOINT_SHA_MATCH"] = (cc.sha256_file(checkpoint_path) == expected_sha)

    try:
        from prism_fas.synthesis.gpat_checkpoint import load_checkpoint, sha256_file  # noqa: F401
        from prism_fas.synthesis.gpat_model import build_gpat_model  # noqa: F401
        from prism_fas.synthesis.synthetic_bank import GPATRoute  # noqa: F401
        result["GPAT_RUNTIME_IMPORTABLE"] = True
    except Exception as error:  # noqa: BLE001
        result["gpat_runtime_import_error"] = str(error)

    cuda = cuda if cuda is not None else _check_cuda_hardware()
    result["GPAT_GPU_BINDING_READY"] = bool(
        result["GPAT_CHECKPOINT_PRESENT"] and result["GPAT_CHECKPOINT_SHA_MATCH"]
        and result["GPAT_RUNTIME_IMPORTABLE"] and cuda.get("CUDA_AVAILABLE") is True)
    return result


def _check_physics_runtime(repo: Path) -> dict[str, Any]:
    """TASK D: PhysicsEngine import + version + pure CPU construction (no
    scientific input, no rendering)."""
    result: dict[str, Any] = {"PHYSICS_RUNTIME_READY": False}
    try:
        from prism_fas.synthesis.physics import PHYSICS_ENGINE_VERSION, PhysicsEngine
    except Exception as error:  # noqa: BLE001
        result["physics_import_ready"] = False
        result["physics_runtime_error"] = str(error)
        return result
    result["physics_import_ready"] = True
    result["physics_engine_version"] = PHYSICS_ENGINE_VERSION
    result["physics_engine_version_matches_expected"] = (
        PHYSICS_ENGINE_VERSION == e6r.EXPECTED_PHYSICS_ENGINE_VERSION)
    try:
        PhysicsEngine()  # pure CPU object construction -- never .apply(), never a scientific input
        result["physics_constructible"] = True
    except Exception as error:  # noqa: BLE001
        result["physics_constructible"] = False
        result["physics_runtime_error"] = str(error)
    result["PHYSICS_RUNTIME_READY"] = bool(
        result["physics_import_ready"] and result["physics_engine_version_matches_expected"]
        and result.get("physics_constructible"))
    return result


def _check_quality_runtime(repo: Path) -> dict[str, Any]:
    """TASK E: reuses `c_ext_e6_render.resolve_quality_backend_assets`
    VERBATIM -- it already resolves every backend (AdaFace/SCRFD/FaceXFormer/
    fingerprint) WITHOUT loading a single model into memory, already respects
    each frozen provider's own request-then-CPU-fallback policy (never
    switches provider merely because a GPU is present), and already compares
    against the historical actual provider. Never recomputes q, never
    evaluates a scientific candidate."""
    assets = e6r.resolve_quality_backend_assets(repo)
    per_backend = [
        {"BACKEND": "AdaFace (identity)", "MODEL_PRESENT": assets["ADAFACE_MODEL_PATH"] is not None,
         "HASH_MATCH": assets["ADAFACE_RESOLVABLE"], "PROVIDER": "torch (device-bound, no provider switch)",
         "IMPORT_READY": assets["ADAFACE_RESOLVABLE"]},
        {"BACKEND": "SCRFD (detector/landmarks)", "MODEL_PRESENT": assets["LANDMARK_MODEL_RESOLVABLE"],
         "HASH_MATCH": assets["LANDMARK_MODEL_RESOLVABLE"], "PROVIDER": assets["LANDMARK_ACTUAL_PROVIDER"],
         "IMPORT_READY": assets["LANDMARK_RUNTIME_RESOLVABLE"]},
        {"BACKEND": "FaceXFormer (parsing)", "MODEL_PRESENT": assets["PARSING_MODEL_PATH"] is not None,
         "HASH_MATCH": assets["PARSING_RESOLVABLE"], "PROVIDER": "torch (device-bound, no provider switch)",
         "IMPORT_READY": assets["PARSING_RESOLVABLE"]},
        {"BACKEND": "fingerprint (deterministic NumPy, no weights)", "MODEL_PRESENT": True,
         "HASH_MATCH": assets["FINGERPRINT_RESOLVABLE"], "PROVIDER": "numpy (CPU, deterministic)",
         "IMPORT_READY": assets["FINGERPRINT_RESOLVABLE"]},
    ]

    quality_lock_path = repo / QUALITY_PARITY_LOCK_PATH
    config_identity_matches_v2_lock: bool | None = None
    if quality_lock_path.is_file():
        try:
            quality_lock = json.loads(quality_lock_path.read_text(encoding="utf-8"))
            locked_identity = quality_lock.get("quality", {}).get("quality_threshold_identity")
            config_identity_matches_v2_lock = (locked_identity == assets["QUALITY_CONFIG_IDENTITY"])
        except (OSError, ValueError, KeyError):
            config_identity_matches_v2_lock = False

    return {
        "per_backend": per_backend,
        "quality_backend_assets": assets,
        "quality_config_identity_matches_v2_lock": config_identity_matches_v2_lock,
        "provider_policy_note": "providers are NEVER switched merely because a GPU is present -- "
                               "LANDMARK_ACTUAL_PROVIDER mirrors QualityBackends.__init__'s own frozen "
                               "request-then-CPU-fallback behavior exactly, reused verbatim from "
                               "resolve_quality_backend_assets, never reimplemented here.",
        "QUALITY_RUNTIME_READY": bool(assets["QUALITY_BACKENDS_RESOLVABLE"]
                                      and (config_identity_matches_v2_lock is not False)),
    }


def _check_source_package(repo: Path) -> dict[str, Any]:
    """TASK F: source package presence/identity, source-pair plan validity,
    and that all 2048 planned positions per arm are resolvable -- a PURE,
    deterministic schedule recomputation (`build_source_pair_plan`), never
    target labels, never a rendered image."""
    package_root = repo / e6r.SOURCE_PACKAGE_ROOT
    present = package_root.is_dir()
    manifest_present = (package_root / e6r.SOURCE_TRAIN_MANIFEST_RELATIVE).is_file()

    identity_match = False
    lock_path = package_root / "PACKAGE_LOCK.json"
    if present and lock_path.is_file():
        try:
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            identity_match = str(lock.get("content_identity_sha256")) == e6r.EXPECTED_PACKAGE_IDENTITY
        except (OSError, ValueError):
            identity_match = False

    plan_valid = False
    positions_resolvable = False
    reason = None
    if manifest_present:
        try:
            from prism_fas.synthesis.c5_source_pair_plan import (PLAN_SEED, build_source_pair_plan,
                                                                 source_pair_plan_identity)

            base_plan = build_source_pair_plan(package_root, seed=PLAN_SEED)
            plan_valid = source_pair_plan_identity(base_plan) == e6r.EXPECTED_SOURCE_PAIR_PLAN_IDENTITY
            positions_resolvable = len(base_plan["positions"]) == e6r.EXPECTED_CANDIDATES_PER_ARM
        except Exception as error:  # noqa: BLE001
            reason = str(error)
    else:
        reason = f"missing {(package_root / e6r.SOURCE_TRAIN_MANIFEST_RELATIVE).as_posix()}"

    return {
        "SOURCE_PACKAGE_PRESENT": present, "manifest_present": manifest_present,
        "SOURCE_PACKAGE_IDENTITY_MATCH": identity_match,
        "SOURCE_PAIR_PLAN_VALID": plan_valid,
        "PLANNED_POSITIONS_RESOLVABLE": positions_resolvable,
        "reason": reason,
    }


def _check_lock_chain(repo: Path) -> dict[str, Any]:
    """TASK G/D: revalidates all six v2 locks on the ACTUAL host (every
    `build_*_lock` function is PURE -- returns a dict, never writes to disk
    itself) AND requires the render-execution plan lock to be a PERSISTED
    file on disk that matches a freshly rebuilt expected plan
    (`verify_execution_plan_lock_matches_expected`). An in-memory-only
    expected object is NEVER accepted as a substitute for a missing
    persisted lock -- this function reads, never writes."""
    try:
        locks = {path: builder(repo) for path, builder in LOCK_BUILDERS.items()}
        readiness = compute_readiness_gate(locks)
    except Exception as error:  # noqa: BLE001
        return {"six_locks_revalidate": False, "execution_plan_lock_valid": False,
               "LOCK_CHAIN_VALID": False, "reason": str(error)}

    verification = verify_execution_plan_lock_matches_expected(repo)
    execution_plan_valid = bool(
        verification["EXECUTION_PLAN_LOCK_PRESENT"]
        and verification["EXECUTION_PLAN_LOCK_EXPECTED_EQUALS_PERSISTED"]
        and verification["persisted_status"] == "FROZEN")

    return {
        "six_locks_revalidate": readiness["E6_V2_READY_FOR_RENDER"],
        "execution_plan_lock_verification": verification,
        "execution_plan_lock_valid": execution_plan_valid,
        "LOCK_CHAIN_VALID": bool(readiness["E6_V2_READY_FOR_RENDER"] and execution_plan_valid),
    }


def _check_output_storage(repo: Path) -> dict[str, Any]:
    """TASK H: verifies the additive v2 output namespace's PARENT directory
    is writable, using a temporary file that is deleted immediately -- never
    creates a real candidate directory, never leaves anything behind."""
    import shutil
    import tempfile

    parent = repo / E6_V2_RUN_ROOT
    parent_accessible = True
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        parent_accessible = False

    storage_ready = False
    if parent_accessible:
        try:
            with tempfile.NamedTemporaryFile(dir=parent, prefix=".preflight_write_test_",
                                            delete=True) as handle:
                handle.write(b"preflight-write-test")
                handle.flush()
                storage_ready = True
        except OSError:
            storage_ready = False

    available_bytes = None
    try:
        usage = shutil.disk_usage(parent if parent.is_dir() else repo)
        available_bytes = usage.free
    except OSError:
        available_bytes = None

    return {
        "OUTPUT_PARENT_ACCESSIBLE": parent_accessible,
        "OUTPUT_STORAGE_READY": storage_ready,
        "AVAILABLE_DISK_SPACE_BYTES": available_bytes,
        "note": "no minimum disk requirement is asserted here -- none is derived from any frozen "
               "project artifact; only presence of free space and write access are checked.",
    }


def gpu_runtime_preflight_v2(repo: Path) -> dict[str, Any]:
    """TASKS B-J: the REAL GPU_RUNTIME_PREFLIGHT, meant to run on the GPU
    host. `--gpu-runtime-preflight` requires neither `--execute` nor
    `--authorize-gpu-render`; it never renders, never creates a candidate
    directory, never invokes Physics/GPAT generation, never runs quality
    inference on scientific inputs, never trains, never touches target,
    never calls an LLM. Fails closed (`gpu_runtime_preflight="FAIL"`) if any
    required runtime dependency is unavailable -- CUDA above all: there is no
    CPU fallback for the GPAT render path."""
    cuda = _check_cuda_hardware()
    gpat = _check_gpat_runtime(repo, cuda=cuda)
    physics = _check_physics_runtime(repo)
    quality = _check_quality_runtime(repo)
    source = _check_source_package(repo)
    lock_chain = _check_lock_chain(repo)
    storage = _check_output_storage(repo)

    passed = (
        cuda.get("CUDA_AVAILABLE") is True
        and gpat["GPAT_CHECKPOINT_PRESENT"] and gpat["GPAT_CHECKPOINT_SHA_MATCH"]
        and gpat["GPAT_RUNTIME_IMPORTABLE"] and gpat["GPAT_GPU_BINDING_READY"]
        and physics["PHYSICS_RUNTIME_READY"]
        and quality["QUALITY_RUNTIME_READY"]
        and source["SOURCE_PACKAGE_PRESENT"] and source["SOURCE_PACKAGE_IDENTITY_MATCH"]
        and source["SOURCE_PAIR_PLAN_VALID"] and source["PLANNED_POSITIONS_RESOLVABLE"]
        and lock_chain["LOCK_CHAIN_VALID"]
        and storage["OUTPUT_STORAGE_READY"]
    )

    return {
        "schema_version": "e6-v2-gpu-runtime-preflight-v1",
        "cuda": cuda, "gpat": gpat, "physics": physics, "quality": quality,
        "source_package": source, "lock_chain": lock_chain, "output_storage": storage,
        "gpu_runtime_preflight": "PASS" if passed else "FAIL",
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False,
        "training_performed": False, "candidates_created": 0,
    }


def compute_gpu_ready_for_render(repo: Path) -> dict[str, Any]:
    """TASK J: E6_V2_GPU_READY_FOR_RENDER = E6_V2_READY_FOR_RENDER AND
    GPU_RUNTIME_PREFLIGHT == PASS. This is still NOT execution authorization
    -- even when TRUE, REAL_RENDER_EXECUTED stays FALSE until the separate,
    explicit `--execute --authorize-gpu-render` invocation."""
    execution_plan_lock = build_render_execution_plan_lock(repo)
    readiness = execution_plan_lock["readiness_at_lock_time"]
    gpu_preflight = gpu_runtime_preflight_v2(repo)
    ready = bool(readiness["E6_V2_READY_FOR_RENDER"] and gpu_preflight["gpu_runtime_preflight"] == "PASS")
    return {
        "schema_version": "e6-v2-gpu-ready-for-render-v1",
        "E6_V2_READY_FOR_RENDER": readiness["E6_V2_READY_FOR_RENDER"],
        "GPU_RUNTIME_PREFLIGHT": gpu_preflight["gpu_runtime_preflight"],
        "E6_V2_GPU_READY_FOR_RENDER": ready,
        "real_render_executed": False,
        "gpu_preflight_detail": gpu_preflight,
    }


def design_post_render_cross_arm_gate() -> dict[str, Any]:
    """TASK L: the SPECIFICATION of the post-render cross-arm audit -- what
    it will check, once both arms have actually rendered. Not executed this
    turn (nothing has rendered), so `E6_V2_READY_FOR_TRAINING` is always
    False here."""
    return {
        "schema_version": "e6-v2-post-render-cross-arm-gate-design-v1",
        "checks_to_perform_after_render": [
            "both arms used the same number of PLANNED candidates (per E6_V2_RENDER_EXECUTION_PLAN_LOCK)",
            "both arms used the same route schedule (route_by_slot, per-position route identical)",
            "both arms used the same source schedule (SOURCE_PAIR_EXECUTION_PARITY still 100% post-hoc)",
            "both arms ran the same quality runtime (quality_threshold_identity, gate thresholds identical)",
            "both arms used the same matching policy (route quota, bank size, matcher identical)",
            "both arms' matched-bank budget/quota match the frozen contract",
        ],
        "reported_separately_not_equalized": [
            "actual successful (GENERATED) candidate count per arm",
            "actual failed (retained SemanticGenerationFailure) candidate count per arm",
        ],
        "explicit_non_requirement": "successful counts are NEVER required to be equal between arms by "
                                   "falsifying, discarding or resampling real failures -- a route- or "
                                   "recipe-specific failure-rate difference between ORIGINAL and "
                                   "SHUFFLE is itself potential evidence, not noise to normalize away.",
        "E6_V2_READY_FOR_TRAINING": False,
        "E6_V2_READY_FOR_TRAINING_note": "remains False until BOTH arms have completed render + quality "
                                        "validation + matched-bank locks AND this audit has actually "
                                        "run against real post-render artifacts -- none of which "
                                        "happened this turn.",
    }


# --------------------------------------------------------------------------- #
# ATTEMPT-1 TECHNICAL RECOVERY (this turn): a real, authorized GPU render
# attempt hit a technical (non-scientific) execution-path failure during
# ORIGINAL's quality/matching stage, after ORIGINAL's 2048 planned candidates
# had already terminally rendered (2045 GENERATED + 3 FAILED_GENERATION).
# Everything below is READ-ONLY auditing/reporting, or additive-only
# provenance/lock writing -- nothing here renders, trains, touches target, or
# calls an LLM, and nothing here is invoked against a real GPU tree this turn.
# --------------------------------------------------------------------------- #

def audit_attempt1_original(repo: Path) -> dict[str, Any]:
    """TASK E: a strictly read-only, recursive audit of the ACTUAL attempt-1
    ORIGINAL candidate tree (at `v2_render_work_root(ARM_ORIGINAL)/ARM_ORIGINAL`
    -- the documented nested recovery root, never assumed correct without
    checking).

    Identity and route are read from `record["generation_identity"]`
    (`candidate_id`, `route`, `recipe_id`, `recipe_ordinal`, `slot`,
    `position`, `live_target_sample_id`, `spoof_source_sample_id`,
    `generator_binding`) -- the REAL, nested schema
    `c5_raw_generation.GenerationIdentity.as_dict()` persists -- NEVER a
    naive `record.get("candidate_id")`/`record.get("route")` top-level guess
    (that exact mistake is what produced the misleading
    `unique_candidate_ids=1`/`routes={'None': 2048}` finding this audit
    replaces). The candidate directory's OWN name is cross-checked against
    the recorded `generation_identity.candidate_id` -- a mismatch is
    reported as INVALID, never silently trusted either way.

    Every count in the return value is DERIVED from what is actually on
    disk against the frozen ORIGINAL plan; none is hardcoded or assumed from
    the GPU-reported summary.
    """
    from prism_fas.synthesis import c5_raw_generation as raw

    original = load_original_llm_recipes(repo)
    plan = build_v2_arm_plan(repo, arm=ARM_ORIGINAL, recipe_content_identity=original["content_identity"],
                             recipe_count=original["recipe_count"])
    rows = build_v2_arm_plan_rows(repo, arm=ARM_ORIGINAL, recipe_bank_identity=plan["recipe_bank_identity"],
                                  recipes=original["recipes"], plan=plan)
    planned_by_id = {row["candidate_id"]: row for row in rows}

    arm_dir = repo / v2_render_work_root(ARM_ORIGINAL) / ARM_ORIGINAL
    classification: dict[str, str] = {}
    seen_ids: set[str] = set()
    duplicates: list[str] = []
    unexpected: list[str] = []
    invalid: list[dict[str, Any]] = []

    if arm_dir.is_dir():
        for candidate_path in sorted(p for p in arm_dir.iterdir() if p.is_dir()):
            on_disk_id = candidate_path.name
            record = raw.read_record(candidate_path / raw.RECORD_NAME)
            if record is None:
                invalid.append({"directory": on_disk_id, "reason": "MISSING_OR_UNPARSEABLE_RECORD"})
                continue

            identity = record.get("generation_identity") or {}
            recorded_id = identity.get("candidate_id")
            # duplicate detection is over the RECORDED identity, not the
            # directory name -- filesystem directory names are inherently
            # unique, so the only way a candidate_id can appear twice is if
            # two DIFFERENT directories both carry a record claiming the
            # SAME recorded generation_identity.candidate_id.
            if recorded_id is not None and recorded_id in seen_ids:
                duplicates.append(recorded_id)
                continue
            if recorded_id is not None:
                seen_ids.add(recorded_id)
            if recorded_id != on_disk_id:
                invalid.append({"directory": on_disk_id, "reason": "DIRECTORY_NAME_DISAGREES_WITH_RECORD",
                               "recorded_candidate_id": recorded_id})
                continue
            if on_disk_id not in planned_by_id:
                unexpected.append(on_disk_id)
                continue

            planned_row = planned_by_id[on_disk_id]
            expected_fields = {
                "recipe_id": planned_row["recipe_id"], "recipe_ordinal": planned_row["recipe_ordinal"],
                "slot": planned_row["slot"], "position": planned_row["position"],
                "route": planned_row["route"], "live_target_sample_id": planned_row["live_target_sample_id"],
                "spoof_source_sample_id": planned_row.get("spoof_source_sample_id"),
                "generator_binding": planned_row["generator_binding"],
                "package_identity": plan["package_identity"], "ontology_identity": plan["ontology_identity"],
                "source_pair_plan_identity": plan["source_pair_plan_identity"],
            }
            mismatched = [field for field, expected in expected_fields.items() if identity.get(field) != expected]
            if mismatched:
                invalid.append({"directory": on_disk_id, "reason": "IDENTITY_FIELD_MISMATCH",
                               "mismatched_fields": mismatched})
                continue

            status = record.get("status")
            if status == raw.GENERATED:
                payload_ok = all((candidate_path / name).is_file() for name in raw.PAYLOAD_NAMES)
                if payload_ok:
                    classification[on_disk_id] = "GENERATED_VALID"
                else:
                    classification[on_disk_id] = "INVALID"
                    invalid.append({"directory": on_disk_id, "reason": "PAYLOAD_MISSING"})
            elif status == raw.FAILED_GENERATION:
                classification[on_disk_id] = "FAILED_GENERATION"
            else:
                classification[on_disk_id] = "INVALID"
                invalid.append({"directory": on_disk_id, "reason": f"UNKNOWN_STATUS:{status!r}"})

    missing = sorted(candidate_id for candidate_id in planned_by_id if candidate_id not in seen_ids)
    generated_valid = sum(1 for value in classification.values() if value == "GENERATED_VALID")
    failed_generation = sum(1 for value in classification.values() if value == "FAILED_GENERATION")
    invalid_count = len(invalid)

    shuffle_dir = repo / v2_render_work_root(ARM_SHUFFLE) / ARM_SHUFFLE
    shuffle_records = 0
    if shuffle_dir.is_dir():
        for candidate_path in shuffle_dir.iterdir():
            if candidate_path.is_dir() and (candidate_path / raw.RECORD_NAME).is_file():
                shuffle_records += 1

    return {
        "schema_version": "e6-v2-attempt1-resume-preflight-v1",
        "audited_root": str(arm_dir),
        "ATTEMPT1_ORIGINAL_PLANNED": len(planned_by_id),
        "ATTEMPT1_ORIGINAL_RECORDS": len(seen_ids),
        "ATTEMPT1_ORIGINAL_GENERATED": generated_valid,
        "ATTEMPT1_ORIGINAL_FAILED_GENERATION": failed_generation,
        "ATTEMPT1_ORIGINAL_INVALID": invalid_count,
        "ATTEMPT1_ORIGINAL_DUPLICATES": len(duplicates),
        "ATTEMPT1_ORIGINAL_MISSING": len(missing),
        "ATTEMPT1_ORIGINAL_UNEXPECTED": len(unexpected),
        "ATTEMPT1_ORIGINAL_REUSABLE": generated_valid + failed_generation,
        "ATTEMPT1_SHUFFLE_RECORDS": shuffle_records,
        "duplicate_ids": duplicates[:50], "unexpected_ids": unexpected[:50],
        "invalid_detail": invalid[:50], "missing_candidate_ids": missing[:50],
        "no_candidates_from_historical_root": not (repo / e6r.RENDER_DIR / "candidates").exists(),
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False, "training_performed": False,
    }


def build_attempt1_provenance(repo: Path) -> dict[str, Any]:
    """TASK I: additive technical-attempt provenance. GPU-observed counts are
    recorded as OBSERVED FACTS (from the user's real GPU report), explicitly
    distinguished from anything this laptop recomputes -- never conflated."""
    return {
        "schema_version": "e6-v2-execution-attempt-provenance-v1",
        "ATTEMPT": 1,
        "STATUS": "TECHNICAL_FAILURE",
        "FAILURE_STAGE": "ORIGINAL_QUALITY_MATCHING",
        "FAILURE_CLASS": "RENDER_ROOT_DOUBLE_ARM_APPEND_AND_QUALITY_ROOT_HISTORICAL_BINDING",
        "failure_detail": {
            "bug_1_double_arm_append": {
                "symbol": "c_ext_e6_v2_paired.render_v2_arm",
                "expression": "candidates_root = repo / v2_candidates_root(arm)  # already arm-inclusive, "
                             "passed as work_root to c5_render.render_arm, which appends "
                             "plan['arm'] again via c5_raw_generation.candidate_dir(work_root, arm, id) "
                             "== work_root / arm / id",
                "observed_effect": "runs/c_ext_q1q2_v1/EXT-F1/e6_paired_current_runtime_v2/candidates/"
                                  "LLM_ORIGINAL_CURRENT_V2/LLM_ORIGINAL_CURRENT_V2/<candidate_id>/",
            },
            "bug_2_quality_root_historical_binding": {
                "symbol": "c_ext_e6_render.default_metrics_provider",
                "expression": "candidates_root = repo / CANDIDATES_ROOT; directory = raw.candidate_dir("
                             "candidates_root, E6_ARM_NAME, row['candidate_id'])  -- both CANDIDATES_ROOT "
                             "and E6_ARM_NAME are the historical E6 (v1) module constants, never "
                             "parameterized by the caller",
                "observed_effect": "attempted to read runs/c_ext_q1q2_v1/EXT-F1/e6_llm_shuffle/render/"
                                  "candidates/LLM_SHUFFLE_A/<candidate_id> regardless of which v2 arm "
                                  "was being measured",
            },
        },
        "gpu_observed_facts": {
            "note": "recorded VERBATIM from the real GPU attempt-1 report -- NOT recomputed or "
                   "re-derived on this laptop (that is audit_attempt1_original's separate job, run "
                   "only when the real nested tree is present)",
            "original_candidate_count": 2048, "original_generated": 2045,
            "original_failed_generation": 3, "shuffle_records": 0,
        },
        "SCIENTIFIC_PROTOCOL_CHANGED": False, "RENDER_ALGORITHM_CHANGED": False,
        "QUALITY_ALGORITHM_CHANGED": False, "MATCHING_ALGORITHM_CHANGED": False,
        "TARGET_ACCESS": False, "LLM_API_CALLS": 0,
    }


def build_attempt1_recovery_lock(repo: Path) -> dict[str, Any]:
    """TASK J: an ADDITIVE technical recovery lock. Does NOT modify
    `E6_V2_RENDER_EXECUTION_PLAN_LOCK.json` -- that lock's own `per_arm`
    `candidates_root` fields remain exactly what they were when first frozen
    (the INTENDED, not-yet-corrected path description); erasing or
    rewriting that would erase the evidence the bug is documented against.
    This lock instead pins the RECOVERY decision as its own, separate,
    explicit artifact."""
    execution_plan_lock = load_persisted_execution_plan_lock(repo)
    body = {
        "schema_version": "e6-v2-attempt1-recovery-lock-v1", "status": "FROZEN",
        "attempt": 1,
        "original_execution_plan_lock_identity": (execution_plan_lock or {}).get("lock_identity"),
        "original_execution_plan_lock_present": execution_plan_lock is not None,
        "actual_nested_original_root": f"{ATTEMPT1_ORIGINAL_RECOVERY_WORK_ROOT}/{ARM_ORIGINAL}",
        "corrected_future_path_contract": {
            "work_root_function": "v2_render_work_root(arm)",
            "rule": "arm-INDEPENDENT for every arm except the documented ARM_ORIGINAL exception; "
                   "c5_render.render_arm/c5_raw_generation.candidate_dir append `arm` exactly once "
                   "on top of it",
            "ARM_ORIGINAL": ATTEMPT1_ORIGINAL_RECOVERY_WORK_ROOT,
            "ARM_SHUFFLE": E6_V2_CANDIDATES_ROOT,
        },
        "candidate_root_binding_used_for_quality": "default_quality_matcher(..., candidates_root="
                                                   "v2_render_work_root(arm)) -- explicit, never the "
                                                   "historical c_ext_e6_render.CANDIDATES_ROOT default",
        "recovery_option_chosen": "A",
        "recovery_option_a_description": "explicitly adopt the nested attempt-1 ORIGINAL candidate-"
                                        "record root for recovery/resume -- existing valid rendered "
                                        "bytes are preserved and reused, never rerendered, because "
                                        "regenerating 2048 candidates for a path-cosmetics-only reason "
                                        "would discard real GPU work for no scientific benefit",
        "recovery_option_b_rejected_for_now": "a future audited metadata/path-only migration (moving "
                                             "bytes to the corrected single-append layout) remains "
                                             "possible later, but is NOT executed this turn",
        "recovery_option_c_rejected": "rerendering ORIGINAL is rejected -- nothing in this turn's audit "
                                     "found the existing 2048 records unusable",
        "resume_policy": "identity-aware via c5_raw_generation.reuse_decision against the ACTUAL nested "
                        "root for ORIGINAL; a candidate whose identity agrees and payload hashes still "
                        "verify is reused, never re-rendered; a genuinely missing planned candidate may "
                        "render (into the SAME nested root, for consistency); a FAILED_GENERATION "
                        "record is retained, never resampled",
        "existing_payload_bytes_preserved": True,
        "scientific_protocol_unchanged": True,
        "storage_layout_is_metadata_only": "candidate storage path never affects q, quality threshold "
                                          "decisions, the matching algorithm, route quota, selection "
                                          "ordering, selected_set_digest, or bank size -- proven by "
                                          "TASK D's historical-regression tests and TASK G's cross-arm "
                                          "layout-independence tests",
        "target_access": False, "llm_api_calls": 0,
    }
    body["lock_identity"] = cc.sha256_bytes(cc.canonical_json_bytes(body))
    return body


def write_attempt1_recovery_lock(repo: Path) -> dict[str, Any]:
    """TASK J: the ONE writing operation for E6_V2_ATTEMPT1_RECOVERY_LOCK.json
    -- additive only, under E6_V2_DIR. Never touches
    E6_V2_RENDER_EXECUTION_PLAN_LOCK.json or any of the six upstream locks."""
    lock = build_attempt1_recovery_lock(repo)
    out_dir = repo / E6_V2_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "E6_V2_ATTEMPT1_RECOVERY_LOCK.json"
    path.write_text(json.dumps(lock, indent=2, default=str), encoding="utf-8")
    return {"lock": lock, "path": str(path)}


def write_attempt1_provenance(repo: Path) -> dict[str, Any]:
    """TASK I: the ONE writing operation for
    E6_V2_ATTEMPT1_PROVENANCE.json -- additive only."""
    provenance = build_attempt1_provenance(repo)
    out_dir = repo / E6_V2_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "E6_V2_ATTEMPT1_PROVENANCE.json"
    path.write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    return {"provenance": provenance, "path": str(path)}


def build_attempt2_provenance(repo: Path) -> dict[str, Any]:
    """ATTEMPT-2 TASK K: additive technical-attempt provenance for the
    SECOND real GPU failure, found AFTER the ATTEMPT-1 path-binding recovery
    -- a THIRD, independent technical bug (`_resolve_quality_bank` hard-
    bound to the LLM-SHUFFLE-A recipe bank for every arm's quality
    reconstruction), now fixed. Never declares GPU resume authorized; that
    remains `design_resume_execution_gate`'s job alone."""
    return {
        "schema_version": "e6-v2-execution-attempt-provenance-v1",
        "ATTEMPT": 2,
        "STATUS": "TECHNICAL_FAILURE",
        "FAILURE_STAGE": "ORIGINAL_QUALITY_MATCHING",
        "FAILURE_CLASS": "QUALITY_RECONSTRUCTION_BANK_HISTORICAL_BINDING",
        "gpu_observed_error": "prism_fas.synthesis.c6_matched_bank.MatchedBankError: physics: only 16 "
                              "of 512 slots could be filled under the common source-domain quota "
                              "(raised from run_v2_render_execution -> match_v2_arm -> "
                              "default_quality_matcher -> select_route_bank)",
        "failure_detail": {
            "bug_3_quality_bank_historical_binding": {
                "symbol": "c_ext_e6_render._resolve_quality_bank / _resolve_quality_runtime",
                "expression": "_resolve_quality_runtime cached a SINGLE global `bank` "
                             "(_resolve_quality_bank(repo), the LLM-SHUFFLE-A recipe bank) for the "
                             "life of the process, with no per-arm/per-caller override -- "
                             "c6_scientific.requested_support_for(store, bank, row) then looked up "
                             "row['recipe_id'] (identical across ORIGINAL/SHUFFLE by construction) "
                             "inside the WRONG bank's recipe CONTENT for every ORIGINAL candidate, "
                             "compiling the wrong requested-support graph/strength and collapsing "
                             "the physics quality-gate pass rate",
                "observed_effect": "physics quality-gate pass count far below the frozen quota "
                                  "(16 of 512 in the real GPU run) despite route/domain metadata "
                                  "being verified correct",
            },
        },
        "fix": {
            "summary": "default_metrics_provider/default_quality_matcher gained an optional, explicit "
                      "`quality_bank` override (default None == the exact historical LLM-SHUFFLE-A "
                      "bank, byte-for-byte); match_v2_arm now builds and supplies THIS arm's own "
                      "recipe bank via build_e6_v2_route_bank(repo, recipes, arm=arm, ...), built from "
                      "the SAME `recipes` list render_v2_arm rendered from",
            "files_changed": ["src/prism_fas/evaluation/c_ext_e6_render.py",
                             "src/prism_fas/evaluation/c_ext_e6_v2_paired.py"],
            "historical_e6_v1_behavior_preserved": True,
        },
        "SCIENTIFIC_PROTOCOL_CHANGED": False, "RENDER_ALGORITHM_CHANGED": False,
        "QUALITY_ALGORITHM_CHANGED": False, "MATCHING_ALGORITHM_CHANGED": False,
        "QUOTA_CHANGED": False, "Q_CHANGED": False, "THRESHOLD_CHANGED": False,
        "TARGET_ACCESS": False, "LLM_API_CALLS": 0,
        "GPU_RESUME_AUTHORIZED": False,
    }


def write_attempt2_provenance(repo: Path) -> dict[str, Any]:
    """The ONE writing operation for E6_V2_ATTEMPT2_PROVENANCE.json --
    additive only; never modifies the ATTEMPT-1 provenance/recovery lock."""
    provenance = build_attempt2_provenance(repo)
    out_dir = repo / E6_V2_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "E6_V2_ATTEMPT2_PROVENANCE.json"
    path.write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    return {"provenance": provenance, "path": str(path)}


def build_attempt3_provenance(repo: Path) -> dict[str, Any]:
    """ATTEMPT-3 TASK I: additive technical-attempt provenance for the
    process-lifetime cross-arm contamination bug. Records EVERY GPU-observed
    fillable count SEPARATELY and NEVER overwrites one with another --
    `REAL_EXECUTION_FAILURE_FILLABLE` (the contaminated same-process
    production number), `STANDALONE_SHUFFLE_PREFLIGHT_FILLABLE`/
    `STANDALONE_ORIGINAL_PREFLIGHT_FILLABLE` (fresh-process, uncontaminated
    by construction) are three independent, verbatim GPU observations, not
    three attempts at the same measurement.

    The fix (this turn) makes `_support_masks`/`SampleStore.cached_mask`
    key on `graph.recipe_hash`, not `recipe_id` alone -- proven, tmp_path,
    to eliminate the exact class of collision that explains 26. A
    same-process, POST-FIX confirmation via `--matching-sequence-preflight`
    has NOT been run on the GPU host this turn (the hard rule for this turn
    forbids GPU execution) -- `predicted_post_fix_same_process_shuffle_fillable`
    is a CODE-LEVEL PREDICTION (fresh-process runs were never exposed to the
    bug in the first place, so nothing about the fix should change them),
    explicitly marked NOT YET GPU-CONFIRMED, never asserted as observed.
    """
    return {
        "schema_version": "e6-v2-execution-attempt-provenance-v1",
        "ATTEMPT": 3,
        "STATUS": "TECHNICAL_FAILURE",
        "FAILURE_STAGE": "SHUFFLE_QUALITY_MATCHING",
        "FAILURE_CLASS": "CROSS_ARM_PROCESS_LIFETIME_CACHE_CONTAMINATION",
        "gpu_observed_error": "prism_fas.synthesis.c6_matched_bank.MatchedBankError: physics: only 26 "
                              "of 512 slots could be filled under the common source-domain quota "
                              "(real, same-process production execution, ORIGINAL then SHUFFLE)",
        "failure_detail": {
            "bug_4_support_mask_cache_key_missing_recipe_content": {
                "symbol": "synthetic_bank._support_masks / m8_pipeline.SampleStore.cached_mask",
                "expression": "cache key was (sample_id, graph.recipe_id) / (sample_id, graph.recipe_id, "
                             "role) -- `recipe_id` is IDENTICAL across ORIGINAL/SHUFFLE at the SAME "
                             "schedule position by construction (c5_source_pair_plan/c5_arm_plan "
                             "`_recipe_id` reads a per-ordinal field carried unchanged by the shuffle), "
                             "while recipe CONTENT (region policy, requested regions, strength) differs. "
                             "The cache lives on `store`, which `c_ext_e6_render."
                             "_resolve_quality_model_runtime` caches for the LIFE OF THE PROCESS and "
                             "reuses across every arm `run_v2_render_execution` measures in that same "
                             "process. ORIGINAL runs FIRST (production order), populating the cache; "
                             "SHUFFLE's later `_support_masks` calls at the SAME (sample_id, recipe_id) "
                             "then silently returned ORIGINAL's stale support mask.",
                "observed_effect": "physics quality-gate outcomes for SHUFFLE candidates computed "
                                  "against the WRONG requested-support mask when measured in the SAME "
                                  "process as ORIGINAL, corrupting support_overlap/measured_artifact_"
                                  "strength and collapsing the physics route-bank fillable count far "
                                  "below the fresh-process (standalone) value",
            },
        },
        "fix": {
            "summary": "_support_masks (synthetic_bank.py) and SampleStore.cached_mask (m8_pipeline.py) "
                      "cache keys extended to (sample_id, recipe_id, graph.recipe_hash[, role]) -- "
                      "recipe_hash is a content hash of the recipe itself, already computed by "
                      "compile_recipe, so two arms' recipes at the same recipe_id/ordinal now always "
                      "cache-miss against each other while a repeated call for the SAME recipe content "
                      "still hits the cache",
            "files_changed": ["src/prism_fas/synthesis/synthetic_bank.py",
                             "src/prism_fas/synthesis/m8_pipeline.py"],
            "render_algorithm_changed": False, "quality_equations_changed": False,
            "quality_thresholds_changed": False, "q_changed": False, "quota_changed": False,
            "matching_algorithm_changed": False, "source_domain_policy_changed": False,
        },
        "candidate_id_collision_across_arms": "IMPOSSIBLE_BY_CONSTRUCTION",
        "candidate_id_collision_reason": "c5_source_pair_plan.candidate_identity hashes BOTH `arm` and "
                                        "`recipe_bank_identity`; either alone already guarantees no "
                                        "collision between LLM_ORIGINAL_CURRENT_V2 and "
                                        "LLM_SHUFFLE_A_CURRENT_V2",
        "gpu_observed_facts": {
            "note": "each value below is a SEPARATE, VERBATIM GPU observation -- never recomputed or "
                   "reconciled into a single number on this laptop",
            "REAL_EXECUTION_FAILURE_FILLABLE": 26,
            "REAL_EXECUTION_FAILURE_FILLABLE_LABEL": "TECHNICAL_ARTIFACT",
            "STANDALONE_ORIGINAL_PREFLIGHT_FILLABLE": 512,
            "STANDALONE_SHUFFLE_PREFLIGHT_FILLABLE": 479,
            "STANDALONE_SHUFFLE_CASIA_AVAILABLE": 231, "STANDALONE_SHUFFLE_CASIA_REQUIRED": 264,
            "STANDALONE_SHUFFLE_MSU_AVAILABLE": 281, "STANDALONE_SHUFFLE_MSU_REQUIRED": 248,
            "standalone_479_pure_arithmetic_confirmed": True,
            "standalone_479_formula": "min(231,264) + min(281,248) = 231 + 248 = 479",
        },
        "predicted_post_fix_same_process_shuffle_fillable": 479,
        "predicted_post_fix_basis": "a fresh-process (standalone) run was NEVER exposed to the "
                                   "cross-arm cache in the first place (nothing else in the SAME "
                                   "process could have populated it before SHUFFLE's own candidates "
                                   "were measured), so the fix -- which only changes behavior when a "
                                   "SECOND arm's request would otherwise have hit a FIRST arm's stale "
                                   "entry -- should make a same-process run converge to the same result "
                                   "a fresh-process run already gave. This is a CODE-LEVEL PREDICTION.",
        "post_fix_gpu_confirmed": False,
        "true_frozen_matched_bank_infeasibility_provisional": True,
        "true_frozen_matched_bank_infeasibility_basis": "IF the standalone 479 is confirmed by a clean "
                                                       "same-process run post-fix, SHUFFLE's Physics "
                                                       "quality-pass total (512) is numerically "
                                                       "sufficient, but its domain COMPOSITION "
                                                       "(CASIA 231 available vs 264 required; MSU 281 "
                                                       "available vs 248 required) cannot satisfy the "
                                                       "frozen EXACT per-domain quota -- MSU's 33-unit "
                                                       "surplus cannot compensate CASIA's 33-unit "
                                                       "deficit under `select_route_bank`'s "
                                                       "per-domain-exact fill rule",
        "SCIENTIFIC_PROTOCOL_CHANGED": False, "RENDER_ALGORITHM_CHANGED": False,
        "QUALITY_ALGORITHM_CHANGED": False, "MATCHING_ALGORITHM_CHANGED": False,
        "QUOTA_CHANGED": False, "Q_CHANGED": False, "THRESHOLD_CHANGED": False,
        "TARGET_ACCESS": False, "LLM_API_CALLS": 0,
        "GPU_RESUME_AUTHORIZED": False,
    }


def write_attempt3_provenance(repo: Path) -> dict[str, Any]:
    """The ONE writing operation for E6_V2_ATTEMPT3_PROVENANCE.json --
    additive only; never modifies ATTEMPT-1/2 provenance or the recovery
    lock."""
    provenance = build_attempt3_provenance(repo)
    out_dir = repo / E6_V2_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "E6_V2_ATTEMPT3_PROVENANCE.json"
    path.write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    return {"provenance": provenance, "path": str(path)}


# =============================================================================
# E6-V2 SCIENTIFIC CLOSURE (this turn): post-fix confirmation, final closure
# lock and human-readable summary. ADDITIVE ONLY -- never rewrites any prior
# lock or provenance file; every one is referenced by sha256, not restated.
# =============================================================================

def build_attempt3_postfix_confirmation() -> dict[str, Any]:
    """The VERBATIM, real GPU `--matching-sequence-preflight` result reported
    this turn -- recorded once, additively, as its own artifact (Task B),
    separate from `E6_V2_ATTEMPT3_PROVENANCE.json` (which recorded the
    PRE-fix contamination finding and the code-level PREDICTION of 479).
    This is the actual post-fix OBSERVATION that confirms that prediction.
    """
    return {
        "schema_version": "e6-v2-matching-sequence-postfix-confirmation-v1",
        "PROCESS_SEQUENCE": [ARM_ORIGINAL, ARM_SHUFFLE],
        "REVERSE_SEQUENCE": [ARM_SHUFFLE, ARM_ORIGINAL],
        "forward": {
            ARM_ORIGINAL: {"GPAT_MAX_FILLABLE": 512, "PHYSICS_MAX_FILLABLE": 512},
            ARM_SHUFFLE: {
                "GPAT_MAX_FILLABLE": 512, "PHYSICS_MAX_FILLABLE": 479,
                "physics_domain_table": {
                    "casia_fasd": {"available": 231, "required_quota": 264, "fillable": 231},
                    "msu_mfsd": {"available": 281, "required_quota": 248, "fillable": 248},
                },
            },
        },
        "reverse": {
            ARM_SHUFFLE: {"PHYSICS_MAX_FILLABLE": 479},
            ARM_ORIGINAL: {"PHYSICS_MAX_FILLABLE": 512},
        },
        "ORDER_DEPENDENCE_PRESENT": False,
        "confirms_prediction_from": "E6_V2_ATTEMPT3_PROVENANCE.json#predicted_post_fix_same_process_shuffle_fillable",
        "predicted_value": 479, "observed_value": 479, "prediction_confirmed": True,
        "cross_arm_cache_contamination_removed": True,
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False,
        "bank_selected": False, "bank_written": False, "training_performed": False,
    }


def write_attempt3_postfix_confirmation(repo: Path) -> dict[str, Any]:
    """The ONE writing operation for E6_V2_ATTEMPT3_POSTFIX_CONFIRMATION.json
    -- additive only."""
    confirmation = build_attempt3_postfix_confirmation()
    out_dir = repo / E6_V2_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "E6_V2_ATTEMPT3_POSTFIX_CONFIRMATION.json"
    path.write_text(json.dumps(confirmation, indent=2, default=str), encoding="utf-8")
    return {"confirmation": confirmation, "path": str(path)}


#: TASK B: every prior lock/provenance file the final closure PINS by
#: sha256/identity -- never rewritten, never restated in full.
PROVENANCE_CHAIN_FILES: tuple[str, ...] = (
    "E6_V2_PROTOCOL_LOCK.json", "E6_V2_RENDER_EXECUTION_PLAN_LOCK.json",
    "E6_V2_ATTEMPT1_PROVENANCE.json", "E6_V2_ATTEMPT1_RECOVERY_LOCK.json",
    "E6_V2_ATTEMPT2_PROVENANCE.json", "E6_V2_ATTEMPT3_PROVENANCE.json",
    "E6_V2_ATTEMPT3_POSTFIX_CONFIRMATION.json",
)


def _pinned_file_reference(repo: Path, filename: str) -> dict[str, Any]:
    """Read-only: the sha256 and, if present, `lock_identity`/`ATTEMPT` of
    one E6_V2_DIR file -- pinned by reference, the file itself never
    touched."""
    path = repo / E6_V2_DIR / filename
    if not path.is_file():
        return {"filename": filename, "present": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    reference: dict[str, Any] = {"filename": filename, "present": True, "sha256": cc.sha256_file(path)}
    for key in ("lock_identity", "ATTEMPT", "STATUS", "schema_version"):
        if key in payload:
            reference[key] = payload[key]
    return reference


def build_e6_v2_final_closure(repo: Path) -> dict[str, Any]:
    """TASK A: the additive, immutable E6-v2 scientific closure. Records the
    clean, post-fix, order-independent matching-feasibility result for both
    arms, classifies every prior fillable observation (16, 26, 479) and pins
    -- never rewrites -- the six prior locks/provenance files plus the
    post-fix confirmation, by sha256.
    """
    casia_deficit = 264 - 231
    msu_surplus = 281 - 248
    provenance_chain = [_pinned_file_reference(repo, filename) for filename in PROVENANCE_CHAIN_FILES]
    missing = [entry["filename"] for entry in provenance_chain if not entry["present"]]

    body = {
        "schema_version": "e6-v2-final-closure-v1",
        "E6_V2_STATUS": "CLOSED_TRUE_FROZEN_MATCHED_BANK_INFEASIBILITY",
        "ORIGINAL_GPAT_MAX_FILLABLE": 512, "ORIGINAL_PHYSICS_MAX_FILLABLE": 512,
        "SHUFFLE_GPAT_MAX_FILLABLE": 512, "SHUFFLE_PHYSICS_MAX_FILLABLE": 479,
        "SHUFFLE_PHYSICS_REQUIRED": 512,
        "SHUFFLE_PHYSICS_CASIA_AVAILABLE": 231, "SHUFFLE_PHYSICS_CASIA_REQUIRED": 264,
        "SHUFFLE_PHYSICS_CASIA_DEFICIT": casia_deficit,
        "SHUFFLE_PHYSICS_MSU_AVAILABLE": 281, "SHUFFLE_PHYSICS_MSU_REQUIRED": 248,
        "SHUFFLE_PHYSICS_MSU_SURPLUS": msu_surplus,
        "ORDER_DEPENDENCE_PRESENT": False,
        "ATTEMPT1_CLASS": "TECHNICAL",
        "ATTEMPT2_16_OF_512_CLASS": "TECHNICAL_ARTIFACT",
        "ATTEMPT3_26_OF_512_CLASS": "TECHNICAL_ARTIFACT",
        "FINAL_479_OF_512_CLASS": "SCIENTIFIC_INFEASIBILITY",
        "MATCHING_ALGORITHM_CHANGED": False, "QUALITY_ALGORITHM_CHANGED": False,
        "Q_CHANGED": False, "QUALITY_THRESHOLD_CHANGED": False,
        "DOMAIN_QUOTA_CHANGED": False, "ROUTE_QUOTA_CHANGED": False,
        "SCIENTIFIC_PROTOCOL_CHANGED": False,
        "TARGET_ACCESS": False, "LLM_API_CALLS": 0,
        "E6_V2_READY_FOR_TRAINING": False,
        "E6_V2_TRAINING_BLOCK_REASON": "SHUFFLE_PHYSICS_MATCHED_BANK_INFEASIBLE_UNDER_FROZEN_DOMAIN_QUOTA",
        "provenance_chain": provenance_chain,
        "provenance_chain_complete": not missing,
        "provenance_chain_missing": missing,
        "prior_provenance_files_rewritten": False,
        "root_cause_reasoning": {
            "casia": "264 required - 231 available = 33 deficit",
            "msu": "281 available - 248 required = 33 surplus",
            "why_surplus_cannot_compensate": "select_route_bank fills each source domain independently "
                                            "up to min(quota[d], available[d]); the frozen quota is "
                                            "EXACT per domain, never fungible across domains -- MSU's "
                                            "33-candidate surplus has no mechanism to fill CASIA's "
                                            "33-candidate deficit under the frozen selector",
        },
        "status": "FROZEN",
    }
    body["closure_identity"] = cc.sha256_bytes(cc.canonical_json_bytes(body))
    return body


def write_e6_v2_final_closure(repo: Path) -> dict[str, Any]:
    """The ONE writing operation for E6_V2_FINAL_CLOSURE.json -- additive
    only; never modifies any file it pins."""
    closure = build_e6_v2_final_closure(repo)
    out_dir = repo / E6_V2_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "E6_V2_FINAL_CLOSURE.json"
    path.write_text(json.dumps(closure, indent=2, default=str), encoding="utf-8")
    return {"closure": closure, "path": str(path)}


def e7_dependency_status(repo: Path) -> dict[str, Any]:
    """TASK D: whether a milestone named "E7" depends on E6-v2's matched-
    bank completion. Read-only, additive-safe: never invents a milestone
    this repository does not define.
    """
    training_plan_path = repo / E6_V2_DIR / "E6_V2_TRAINING_PLAN_LOCK.json"
    training_plan_status = None
    if training_plan_path.is_file():
        training_plan_status = json.loads(training_plan_path.read_text(encoding="utf-8")).get(
            "training_authorized_this_turn")
    return {
        "schema_version": "e6-v2-e7-dependency-status-v1",
        "E7_MILESTONE_DEFINED_IN_REPOSITORY": False,
        "reason": "This repository's canonical milestone sequence is C0-C13 (see CLAUDE.md's "
                 "'Canonical entrypoint'); the c_ext_q1q2_v1 extension's own milestones are named "
                 "E0-E6 (this being E6-v2, a paired rerender within E6). No milestone literally named "
                 "'E7' is defined anywhere in docs/PROJECT_STATE.md, .claude/skills/, or the source "
                 "tree -- inventing one here would itself be a scope violation.",
        "E7_DEPENDENCY_STATUS": "NOT_APPLICABLE_NO_E7_MILESTONE_DEFINED",
        "actual_downstream_consumer": "E6_V2_TRAINING_PLAN_LOCK.json (detector-training run-id plan "
                                     "for BOTH arms, 5 seeds each) -- its own "
                                     "'training_authorized_this_turn' field, currently "
                                     f"{training_plan_status!r}, and 'synthetic_bank_identity_status' "
                                     "'PENDING (no bank rendered this turn)'",
        "actual_downstream_consumer_blocked": True,
        "actual_downstream_consumer_block_reason":
            "SHUFFLE_PHYSICS_MATCHED_BANK_INFEASIBLE_UNDER_FROZEN_DOMAIN_QUOTA",
        "e7_executed_this_turn": False,
    }


def build_e6_v2_final_summary_markdown(repo: Path) -> str:
    """TASK C: the additive, human-readable closure narrative. Deliberately
    makes only the claim the evidence supports -- never that ORIGINAL is
    "superior" or that any LLM reasoning property caused the SHUFFLE
    deficit; the frozen quota's domain composition, not the arm's quality,
    is what fails.
    """
    e7 = e7_dependency_status(repo)
    return f"""# E6-v2 PAIRED_CURRENT_RUNTIME — Final Scientific Closure

## 1. Why this paired rerender existed

The historical ORIGINAL_LLM C5/C6 bank could not serve as E6's "original"
condition: reproducing its `trace.requested_coverage` under the current
production runtime was proven code-level impossible, and the historical
implementation that produced it is unrecoverable from git history. E6-v2
answers the underlying question cleanly instead — does the frozen LLM recipe
bank's benefit depend on cross-field joint associations, or on field
marginals alone? — by rendering BOTH `LLM_ORIGINAL_CURRENT_V2` and
`LLM_SHUFFLE_A_CURRENT_V2` fresh, under the identical current runtime, from
the identical frozen source-pair schedule.

## 2. Three technical execution failures, and their fixes

| Attempt | Real production observation | Root cause | Fix |
|---|---|---|---|
| 1 | Technical path-binding failure during render/quality resolution | Render work_root double-appended the arm segment; quality lookup fell back to the historical E6 root/arm | Corrected work_root contract (`v2_render_work_root`), explicit `candidates_root`/`arm` parameterization |
| 2 | Physics fillable = 16/512 | Quality reconstruction (`requested_support_for`) used LLM-SHUFFLE-A's recipe bank for EVERY arm, including ORIGINAL | `default_metrics_provider`/`default_quality_matcher` gained an explicit `quality_bank` override, built per-arm from that arm's own recipes |
| 3 | Physics fillable = 26/512 | `_support_masks`/`SampleStore.cached_mask` memoized region masks keyed by `(sample_id, recipe_id)` — identical across arms at the same schedule position by construction — while recipe CONTENT differs; the cache is shared for the life of the process across both arms | Cache keys extended to include `graph.recipe_hash`, a content hash, so identical-`recipe_id`/different-content requests never collide |

Existing rendered bytes were preserved at every step; no candidate was ever
re-rendered to work around a bug. Each attempt's provenance is recorded,
unmodified, in `E6_V2_ATTEMPT{{1,2,3}}_PROVENANCE.json`.

## 3. Why 16 and 26 are not scientific results

Both numbers were produced by code defects in the MEASUREMENT/MATCHING path,
not by anything about the rendered candidates or the frozen quality gate
itself. A read-only `--matching-sequence-preflight` diagnostic, run on the
real GPU host after the attempt-3 fix, reproduced the production execution
order (ORIGINAL then SHUFFLE, one process) and its reverse (SHUFFLE then
ORIGINAL) and found the two orders now agree exactly:

- Forward SHUFFLE Physics max fillable = 479
- Reverse SHUFFLE Physics max fillable = 479
- `ORDER_DEPENDENCE_PRESENT = False`

This confirms the cross-arm cache contamination is fully removed, and that
479 — not 16, not 26 — is the clean, reproducible, order-independent result.

## 4. Clean ORIGINAL feasibility

ORIGINAL_LLM_CURRENT_V2: GPAT max fillable = 512/512, Physics max fillable =
512/512. Both routes fill their full frozen quota. PASS.

## 5. Clean SHUFFLE infeasibility

LLM_SHUFFLE_A_CURRENT_V2: GPAT max fillable = 512/512 (PASS). Physics: 512
candidates pass the frozen quality gate in total — numerically enough to
fill 512 slots — but only 479 of the 512 required slots can actually be
FILLED once the frozen per-source-domain quota is applied. FAIL.

## 6. The exact CASIA deficit

| Domain | Quality-pass available | Frozen quota required | Fillable |
|---|---|---|---|
| CASIA-FASD | 231 | 264 | 231 |
| MSU-MFSD | 281 | 248 | 248 |
| **Total** | **512** | **512** | **479** |

CASIA deficit = 264 required − 231 available = **33**.
MSU surplus = 281 available − 248 required = **33**.

## 7. Why the MSU surplus cannot compensate

`select_route_bank` (the frozen §11.3 selector) fills each source domain
INDEPENDENTLY, up to `min(quota[domain], available[domain])`. The frozen
common-domain quota vector is exact per domain, not a single pooled total —
it exists specifically so no arm's bank can substitute an easy domain for a
hard one. MSU's 33-candidate surplus has no mechanism to fill CASIA's
33-candidate deficit under this rule; the two are numerically equal by
coincidence, not fungible by design.

## 8. Why training is blocked

`E6_V2_READY_FOR_TRAINING = False`. `E6_V2_TRAINING_PLAN_LOCK.json` already
plans 5 detector-training seeds per arm but records
`training_authorized_this_turn: false` and
`synthetic_bank_identity_status: "PENDING (no bank rendered this turn)"` —
no final SHUFFLE matched bank was ever selected or written this turn (or
any prior turn); TRAINING on an infeasible/incomplete bank is not attempted.
{e7['reason']} The nearer, real downstream consumer —
`E6_V2_TRAINING_PLAN_LOCK.json` — is blocked for exactly this reason.

## 9. No target access, no LLM calls, no scientific parameter changes

`target_access = False` and `llm_api_calls = 0` throughout every attempt and
this closure. Across all three technical fixes and this closure: quota,
`q`, quality thresholds, the matching algorithm, source-domain policy, GPAT,
Physics, recipes and the source schedule were never changed.

## The one valid claim

Under this frozen rendering/quality/matching protocol, the shuffled arm
cannot satisfy the frozen Physics source-domain quota after quality gating.
This says nothing about whether shuffled recipes produce lower-quality
candidates in general, and nothing about any causal LLM-reasoning
advantage — the ORIGINAL arm's own quality-pass candidates happen to fall
into source domains the frozen quota can fill; the SHUFFLE arm's do not, by
33 candidates, in exactly one domain.
"""


def write_e6_v2_final_summary(repo: Path) -> dict[str, Any]:
    """The ONE writing operation for E6_V2_FINAL_SUMMARY.md -- additive
    only."""
    text = build_e6_v2_final_summary_markdown(repo)
    out_dir = repo / E6_V2_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "E6_V2_FINAL_SUMMARY.md"
    path.write_text(text, encoding="utf-8")
    return {"path": str(path), "bytes": len(text.encode("utf-8"))}


def design_resume_execution_gate() -> dict[str, Any]:
    """TASK K: the SPECIFICATION of what a future resume execution must
    require before proceeding -- NOT executed this turn."""
    return {
        "schema_version": "e6-v2-resume-execution-gate-design-v1",
        "required_before_resume": [
            "E6_V2_RENDER_EXECUTION_PLAN_LOCK.json present and EXECUTION_PLAN_LOCK_EXPECTED_EQUALS_"
            "PERSISTED == True (verify_execution_plan_lock_matches_expected)",
            "E6_V2_ATTEMPT1_RECOVERY_LOCK.json present and its own identity valid",
            "E6_V2_ATTEMPT2_PROVENANCE.json present (the quality-bank historical-binding fix is "
            "on-disk, additive-only, provenance-recorded)",
            "--resume-preflight reports RESUME_PREFLIGHT == PASS",
            "audit_attempt1_original reports zero DUPLICATES and zero UNEXPECTED candidate ids",
            "--matching-preflight (both arms once SHUFFLE is rendered) reports "
            "MATCHING_PREFLIGHT_CLASSIFICATION in {not TECHNICAL_METADATA_BUG} -- a "
            "TRUE_FROZEN_MATCHING_INFEASIBILITY outcome is a valid scientific finding this gate does "
            "NOT block on; a TECHNICAL_METADATA_BUG outcome blocks resume until fixed",
            "historical namespaces (reports/full/c5, reports/full/c6, the old e6_llm_shuffle path) "
            "byte-identical to before",
            "target_access == False", "llm_api_calls == 0",
        ],
        "not_run_this_turn": True,
        "GPU_RESUME_AUTHORIZED": False,
    }


# =============================================================================
# ATTEMPT-2 TECHNICAL RECOVERY: read-only matching/domain diagnostic (TASK F).
# =============================================================================

#: TASK F: the two matcher-relevant routes, in the exact order
#: `default_quality_matcher` processes them (physics first) -- so a
#: diagnostic that stops at the first infeasible route reports the SAME
#: route the real GPU failure did.
_MATCHING_ROUTES: tuple[str, ...] = ("physics", "gpat")


def _max_fillable_under_quota(available_by_domain: dict[str, int], quota: dict[str, int]) -> int:
    """TASK A/F/I: the exact terminal candidate count
    `synthesis.c6_matched_bank.select_route_bank` reaches before it either
    completes or raises -- WITHOUT calling or reimplementing any part of its
    ordering/tie-break policy.

    `select_route_bank`'s own loop only ever stops taking from domain `d`
    once `remaining[d]` (initialized to `quota[d]`) hits zero, and it never
    redistributes a domain's unused quota to another domain (that
    redistribution, if any, already happened earlier and is baked INTO
    `quota` itself -- see `resolve_route_quota`, not used by E6/E6-v2, whose
    quota instead comes verbatim from `resolve_e6_route_quota`'s frozen
    bank-lock read). So the total the loop can ever reach, independent of
    selection order, tie-break or which specific candidates are chosen, is
    exactly `sum(min(quota[d], available[d]) for d in domains)` -- a pure
    cardinality fact about `quota`/`available`, not a second selection
    policy. `select_route_bank` itself is still called directly (wrapped, so
    it writes nothing on either side) by `matching_preflight_v2` to prove
    this formula agrees with the frozen function's own behavior.
    """
    from prism_fas.synthesis.c6_matched_bank import canonical_domains

    domains = canonical_domains(set(quota) | set(available_by_domain))
    return sum(min(int(quota.get(domain, 0)), int(available_by_domain.get(domain, 0)))
              for domain in domains)


def matching_preflight_v2(repo: Path, *, arm: str = ARM_ORIGINAL,
                          metrics_provider: Callable[..., dict[str, Any]] | None = None
                          ) -> dict[str, Any]:
    """TASK F: a strictly READ-ONLY diagnostic for the real GPU ATTEMPT-2
    failure (`MatchedBankError: physics: only 16 of 512 slots could be
    filled under the common source-domain quota`, raised from
    `default_quality_matcher` -> `select_route_bank`).

    Never renders, resamples, writes a matched-bank lock, trains, touches
    target or calls an LLM. It reuses, never reimplements:
      - `stage_v2_results_for_quality` for exactly which candidates are
        GENERATED and on-disk,
      - the REAL `default_metrics_provider`/`quality_gate.evaluate` chain
        (behind the SAME `quality_bank`/`candidates_root` explicit binding
        ATTEMPT-2's third bug fix introduced) for quality metrics -- when
        the real quality backend stack cannot be resolved on this host
        (this laptop), every quality-dependent field reports
        `DIAGNOSTIC_STATUS=DEFERRED` rather than a fabricated number,
      - `resolve_e6_route_quota` for the SAME frozen per-domain quota
        `default_quality_matcher` itself resolves,
      - `select_route_bank` ITSELF (wrapped, catching `MatchedBankError`;
        writes nothing) to directly reproduce/confirm the real failure,
        cross-checked against `_max_fillable_under_quota`'s pure
        cardinality count.
    """
    from prism_fas.synthesis import c5_raw_generation as raw
    from prism_fas.synthesis.c6_matched_bank import MatchedBankError, SelectableCandidate, select_route_bank
    from prism_fas.synthesis.quality_gate import Thresholds, evaluate as quality_gate_evaluate

    if arm == ARM_ORIGINAL:
        source = load_original_llm_recipes(repo)
    elif arm == ARM_SHUFFLE:
        source = e6r.verify_shuffle_recipe_source(repo)
    else:
        raise E6V2ExecutionError(f"unknown v2 arm {arm!r}; expected {ARM_ORIGINAL!r} or {ARM_SHUFFLE!r}")
    recipes = source["recipes"]
    recipe_count = source.get("recipe_count", len(recipes))

    plan = build_v2_arm_plan(repo, arm=arm, recipe_content_identity=source["content_identity"],
                             recipe_count=recipe_count)
    rows = build_v2_arm_plan_rows(repo, arm=arm, recipe_bank_identity=plan["recipe_bank_identity"],
                                  recipes=recipes, plan=plan)

    planned_by_route: dict[str, int] = {route: 0 for route in _MATCHING_ROUTES}
    for row in rows:
        planned_by_route[row["route"]] = planned_by_route.get(row["route"], 0) + 1

    # RECORD_TOTAL/GENERATED_TOTAL/FAILED_GENERATION_TOTAL, and per-route
    # GENERATED breakdown, resolved from the ACTUAL on-disk nested tree --
    # never assumed, never reusing `audit_attempt1_original`'s own summary
    # numbers verbatim (this walks the SAME work_root independently so a
    # future SHUFFLE arm gets identical treatment without a second function).
    work_root = repo / v2_render_work_root(arm)
    arm_dir = work_root / arm
    record_total = 0
    failed_generation_total = 0
    generated_by_route: dict[str, int] = {route: 0 for route in _MATCHING_ROUTES}
    if arm_dir.is_dir():
        for candidate_path in arm_dir.iterdir():
            if not candidate_path.is_dir():
                continue
            record = raw.read_record(candidate_path / raw.RECORD_NAME)
            if record is None:
                continue
            record_total += 1
            if record.get("status") == raw.FAILED_GENERATION:
                failed_generation_total += 1

    staged = stage_v2_results_for_quality(repo, arm=arm, rows=rows)
    for row in staged["rows"]:
        generated_by_route[row["route"]] = generated_by_route.get(row["route"], 0) + 1
    generated_total = len(staged["rows"])

    # `QUALITY_BACKENDS_RESOLVABLE` reports whether the REAL, GPU-only
    # `default_metrics_provider` chain (model weights, calibration, ONNX
    # runtime provider) is usable on THIS host -- informational, reported
    # verbatim. It must never gate a caller-injected fake `metrics_provider`
    # (every test): a fake provider needs no real weight, exactly like
    # `default_quality_matcher`/`match_v2_arm` themselves never require real
    # backend assets when a fake provider is supplied.
    assets = e6r.resolve_quality_backend_assets(repo)
    quality_backends_resolvable = bool(assets["QUALITY_BACKENDS_RESOLVABLE"])
    provider = metrics_provider or e6r.default_metrics_provider
    metrics_computation_available = (
        quality_backends_resolvable if provider is e6r.default_metrics_provider else True)

    metrics_success_by_route: dict[str, int] = {route: 0 for route in _MATCHING_ROUTES}
    metrics_fail_by_route: dict[str, int] = {route: 0 for route in _MATCHING_ROUTES}
    quality_pass_by_route: dict[str, int] = {route: 0 for route in _MATCHING_ROUTES}
    quality_fail_by_route: dict[str, int] = {route: 0 for route in _MATCHING_ROUTES}
    rejection_reasons_by_route: dict[str, dict[str, int]] = {route: {} for route in _MATCHING_ROUTES}
    domain_pass_by_route: dict[str, dict[str, int]] = {route: {} for route in _MATCHING_ROUTES}
    eligible_by_route: dict[str, list[Any]] = {route: [] for route in _MATCHING_ROUTES}
    diagnostic_error: str | None = None

    if metrics_computation_available:
        try:
            # only built when it will actually reach `default_metrics_provider`
            # -- a real ontology load + recipe parse a caller-injected fake
            # provider (every test) never needs to pay for, mirroring
            # `match_v2_arm`'s own identical guard.
            quality_bank = (build_e6_v2_route_bank(repo, recipes, arm=arm,
                                                   bank_identity=plan["recipe_bank_identity"])
                            if provider is e6r.default_metrics_provider else None)
            gate_profiles = cc.read_json(repo / e6r.C6_GATE_PROFILES_PATH)
            thresholds = Thresholds.from_dict(
                gate_profiles["profiles"][e6r.EXPECTED_QUALITY_PROFILE]["thresholds"])
            candidates_root = repo / v2_render_work_root(arm)

            for row, record in zip(staged["rows"], staged["results"]):
                route = row["route"]
                provider_kwargs = ({"candidates_root": candidates_root, "arm": arm, "quality_bank": quality_bank}
                                   if provider is e6r.default_metrics_provider else {})
                try:
                    metrics = provider(repo=repo, row=row, record=record, **provider_kwargs)
                except Exception as error:  # noqa: BLE001 - one bad candidate must not abort the diagnostic
                    metrics_fail_by_route[route] += 1
                    continue
                metrics_success_by_route[route] += 1
                decision = quality_gate_evaluate(metrics, thresholds)
                domain = str(row.get("live_dataset", ""))
                if decision["accepted"]:
                    quality_pass_by_route[route] += 1
                    domain_pass_by_route[route][domain] = domain_pass_by_route[route].get(domain, 0) + 1
                    eligible_by_route[route].append(SelectableCandidate(
                        candidate_id=row["candidate_id"], arm=arm, route=route, source_domain=domain,
                        recipe_id=row["recipe_id"], recipe_ordinal=row["recipe_ordinal"],
                        live_target_sample_id=row["live_target_sample_id"], base_position=row["position"],
                        q=decision["q"]))
                else:
                    quality_fail_by_route[route] += 1
                    for gate_name in decision["failed_gates"]:
                        rejection_reasons_by_route[route][gate_name] = (
                            rejection_reasons_by_route[route].get(gate_name, 0) + 1)
        except Exception as error:  # noqa: BLE001 - a diagnostic must report, never crash, on a runtime failure
            metrics_computation_available = False
            diagnostic_error = f"{type(error).__name__}: {e6r._sanitize_diagnostic_error(str(error))}"

    quota = e6r.resolve_e6_route_quota(repo)
    common_source_domains = sorted(set().union(*(set(quota.get(route, {})) for route in _MATCHING_ROUTES)))

    domains_table: dict[str, list[dict[str, Any]]] = {}
    max_fillable: dict[str, int | None] = {}
    select_route_bank_result: dict[str, dict[str, Any]] = {}
    for route in _MATCHING_ROUTES:
        route_quota = quota.get(route, {})
        required = sum(route_quota.values())
        if metrics_computation_available:
            available = domain_pass_by_route[route]
            domains_table[route] = [
                {"domain": domain, "available": available.get(domain, 0),
                "required_quota": int(route_quota.get(domain, 0)),
                "fillable": min(int(route_quota.get(domain, 0)), available.get(domain, 0))}
                for domain in sorted(set(route_quota) | set(available))]
            max_fillable[route] = _max_fillable_under_quota(available, route_quota)
            try:
                selected = select_route_bank(eligible_by_route[route], route=route, quota=route_quota)
                select_route_bank_result[route] = {"raised": False, "selected_count": len(selected)}
            except MatchedBankError as error:
                select_route_bank_result[route] = {"raised": True, "message": str(error)}
        else:
            domains_table[route] = [
                {"domain": domain, "available": None, "required_quota": int(route_quota.get(domain, 0)),
                "fillable": None} for domain in sorted(route_quota)]
            max_fillable[route] = None
            select_route_bank_result[route] = {"raised": None, "reason": "DEFERRED"}

    physics_required = sum(quota.get("physics", {}).values())
    classification = "UNRESOLVED"
    classification_reason = "quality metrics could not be computed on this host; re-run on the GPU host"
    if metrics_computation_available:
        physics_fillable = max_fillable.get("physics")
        physics_planned = planned_by_route.get("physics", 0)
        physics_generated = generated_by_route.get("physics", 0)
        route_metadata_healthy = physics_generated > 0 and metrics_success_by_route["physics"] > 0
        if physics_fillable is not None and physics_fillable < physics_required:
            if not route_metadata_healthy:
                classification = "TECHNICAL_METADATA_BUG"
                classification_reason = (
                    "no physics candidate reached a successful metrics computation; a technical "
                    "staging/binding failure, not genuine quality sparsity")
            elif quality_pass_by_route["physics"] > 0 and (
                    quality_pass_by_route["physics"] / max(1, metrics_success_by_route["physics"])) < 0.10:
                classification = "TRUE_FROZEN_MATCHING_INFEASIBILITY"
                classification_reason = (
                    f"{quality_pass_by_route['physics']} of {metrics_success_by_route['physics']} physics "
                    "candidates that were successfully measured passed the frozen quality gate, and the "
                    f"passing candidates' domain distribution cannot fill the frozen quota "
                    f"({physics_fillable} of {physics_required} fillable) -- consistent with a genuine "
                    "quality/domain-sparsity outcome under the frozen gate and quota, not a metadata defect")
            else:
                classification = "UNRESOLVED"
                classification_reason = (
                    "quality pass rate does not obviously indicate either a technical collapse or a "
                    "sparse-but-plausible outcome; needs human/scientific review of the domain table and "
                    "rejection-reason breakdown before either conclusion is recorded")
        else:
            classification = "UNRESOLVED"
            classification_reason = "the frozen quota was fillable under this run; no failure reproduced"

    return {
        "schema_version": "e6-v2-matching-preflight-v1",
        "MATCHING_PREFLIGHT_ARM": arm,
        "PLANNED_TOTAL": len(rows),
        "RECORD_TOTAL": record_total,
        "GENERATED_TOTAL": generated_total,
        "FAILED_GENERATION_TOTAL": failed_generation_total,
        "GPAT_PLANNED": planned_by_route.get("gpat", 0), "PHYSICS_PLANNED": planned_by_route.get("physics", 0),
        "GPAT_GENERATED": generated_by_route.get("gpat", 0), "PHYSICS_GENERATED": generated_by_route.get("physics", 0),
        "GPAT_METRICS_SUCCESS": metrics_success_by_route.get("gpat", 0) if metrics_computation_available else None,
        "PHYSICS_METRICS_SUCCESS": metrics_success_by_route.get("physics", 0) if metrics_computation_available else None,
        "GPAT_METRICS_FAIL": metrics_fail_by_route.get("gpat", 0) if metrics_computation_available else None,
        "PHYSICS_METRICS_FAIL": metrics_fail_by_route.get("physics", 0) if metrics_computation_available else None,
        "GPAT_QUALITY_PASS": quality_pass_by_route.get("gpat", 0) if metrics_computation_available else None,
        "PHYSICS_QUALITY_PASS": quality_pass_by_route.get("physics", 0) if metrics_computation_available else None,
        "GPAT_QUALITY_FAIL": quality_fail_by_route.get("gpat", 0) if metrics_computation_available else None,
        "PHYSICS_QUALITY_FAIL": quality_fail_by_route.get("physics", 0) if metrics_computation_available else None,
        "GPAT_SCIENTIFIC_ELIGIBLE": quality_pass_by_route.get("gpat", 0) if metrics_computation_available else None,
        "PHYSICS_SCIENTIFIC_ELIGIBLE": quality_pass_by_route.get("physics", 0) if metrics_computation_available else None,
        "GPAT_ELIGIBLE_BY_ROUTE": len(eligible_by_route["gpat"]) if metrics_computation_available else None,
        "PHYSICS_ELIGIBLE_BY_ROUTE": len(eligible_by_route["physics"]) if metrics_computation_available else None,
        "COMMON_SOURCE_DOMAINS": common_source_domains,
        "domains_by_route": domains_table,
        "quality_rejection_breakdown_by_route": rejection_reasons_by_route if metrics_computation_available else None,
        "select_route_bank_result_by_route": select_route_bank_result,
        "GPAT_MAX_FILLABLE_UNDER_FROZEN_QUOTA": max_fillable.get("gpat"),
        "PHYSICS_MAX_FILLABLE_UNDER_FROZEN_QUOTA": max_fillable.get("physics"),
        "PHYSICS_OBSERVED_FAILURE_FILLABLE": 16,
        "MATCHED_BANK_REQUIRED_PER_ROUTE": physics_required or sum(quota.get("gpat", {}).values()),
        "MATCHING_PREFLIGHT_CLASSIFICATION": classification,
        "matching_preflight_classification_reason": classification_reason,
        "quality_backends_resolvable": quality_backends_resolvable,
        "metrics_computation_available": metrics_computation_available,
        "diagnostic_error": diagnostic_error,
        "route_metadata_field": "generation_identity.route (via build_v2_arm_plan_rows)",
        "source_domain_field": "row['live_dataset'] (from c5_source_pair_plan.build_source_pair_plan; "
                               "identical to c6_matched_bank.SOURCE_DOMAIN_PLAN_FIELD)",
        "bank_selected": False, "bank_written": False, "rendering_performed": False,
        "resampling_performed": False, "training_performed": False,
        "target_access": False, "llm_api_calls": 0,
    }


#: ATTEMPT-3: the production process order (`run_v2_render_execution`'s own
#: loop: ORIGINAL fully render+match, then SHUFFLE fully render+match) --
#: never reordered for aesthetics, only reproduced here for diagnosis.
PRODUCTION_ARM_SEQUENCE: tuple[str, ...] = (ARM_ORIGINAL, ARM_SHUFFLE)


def matching_sequence_preflight_v2(repo: Path, *, sequence: tuple[str, ...] = PRODUCTION_ARM_SEQUENCE,
                                   metrics_provider: Callable[..., dict[str, Any]] | None = None,
                                   reverse_too: bool = True) -> dict[str, Any]:
    """TASK C: a STRICTLY READ-ONLY diagnostic that runs `matching_preflight_v2`
    for every arm in `sequence`, IN ONE PYTHON PROCESS, WITHOUT resetting the
    quality-runtime cache between arms within one sequence -- exactly
    reproducing `run_v2_render_execution`'s own same-process behavior
    (ORIGINAL fully processed, THEN SHUFFLE, sharing the one
    `_resolve_quality_model_runtime`-cached `store`/evaluator/backends for
    the whole call). This is deliberately NOT the same as calling
    `matching_preflight_v2` for one arm from a fresh process (a "standalone"
    preflight) -- the entire point is to let same-process state persist
    exactly as production does, so a cross-arm contamination bug (ATTEMPT-3's
    proven `_support_masks`/`SampleStore.cached_mask` cache-key bug, now
    fixed) would still show up here if it were ever reintroduced.

    If `reverse_too`, the quality-runtime cache is explicitly reset
    (`e6r.reset_quality_runtime_cache_for_tests` -- the one sanctioned reset
    seam) BETWEEN the forward sequence and the reversed one, so the reversed
    sequence starts from a clean cache exactly like a fresh process would
    -- making its FIRST arm's result a same-process "clean baseline" for
    that arm, directly comparable to what a standalone preflight would
    report. No reset ever happens WITHIN a sequence itself; only between the
    two directions.

    Never renders, never selects/writes a matched bank, never trains, never
    touches target, never calls an LLM.
    """
    def _run_sequence(order: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        return {arm: matching_preflight_v2(repo, arm=arm, metrics_provider=metrics_provider) for arm in order}

    forward = _run_sequence(sequence)

    reverse_result: dict[str, dict[str, Any]] | None = None
    reverse_sequence: tuple[str, ...] | None = None
    if reverse_too:
        e6r.reset_quality_runtime_cache_for_tests()
        reverse_sequence = tuple(reversed(sequence))
        reverse_result = _run_sequence(reverse_sequence)

    def _fillable(result_by_arm: dict[str, dict[str, Any]] | None, arm: str, route: str) -> int | None:
        if result_by_arm is None or arm not in result_by_arm:
            return None
        return result_by_arm[arm][f"{route.upper()}_MAX_FILLABLE_UNDER_FROZEN_QUOTA"]

    order_dependence: dict[str, Any] = {}
    for arm in sequence:
        forward_value = _fillable(forward, arm, "physics")
        # the arm's "clean baseline" is its OWN result the run in which it
        # was FIRST in sequence (an empty cache at the moment it ran) --
        # forward, that is whichever arm sequence[0] is; reversed, it is
        # whichever arm reverse_sequence[0] is (normally the OTHER arm).
        if reverse_result is not None and reverse_sequence is not None and arm == reverse_sequence[0]:
            clean_value = _fillable(reverse_result, arm, "physics")
        elif arm == sequence[0]:
            clean_value = forward_value
        else:
            clean_value = None
        order_dependence[arm] = {
            "forward_position": sequence.index(arm), "forward_physics_max_fillable": forward_value,
            "ran_first_in_a_direction": (arm == sequence[0]) or
                                        (reverse_sequence is not None and arm == reverse_sequence[0]),
            "clean_baseline_physics_max_fillable": clean_value,
            "matches_clean_baseline": (clean_value is None or forward_value is None
                                       or clean_value == forward_value),
        }

    order_dependence_present = any(
        entry["clean_baseline_physics_max_fillable"] is not None
        and entry["forward_physics_max_fillable"] is not None
        and entry["clean_baseline_physics_max_fillable"] != entry["forward_physics_max_fillable"]
        for entry in order_dependence.values())

    return {
        "schema_version": "e6-v2-matching-sequence-preflight-v1",
        "PROCESS_SEQUENCE": list(sequence),
        "REVERSE_SEQUENCE": list(reverse_sequence) if reverse_sequence is not None else None,
        "forward_results_by_arm": forward,
        "reverse_results_by_arm": reverse_result,
        "order_dependence_by_arm": order_dependence,
        "ORDER_DEPENDENCE_PRESENT": order_dependence_present,
        "bank_selected": False, "bank_written": False, "rendering_performed": False,
        "resampling_performed": False, "training_performed": False,
        "target_access": False, "llm_api_calls": 0,
    }


def run_v2_render_execution(repo: Path, *,
                            render_arm_fn: Callable[..., dict[str, Any]] | None = None,
                            metrics_provider: Callable[..., dict[str, Any]] | None = None,
                            quality_matcher: Callable[..., dict[str, Any]] | None = None
                            ) -> dict[str, Any]:
    """TASK C/D/F/G/H/I/J: the two-arm execution orchestrator. FAILS CLOSED,
    before rendering a single candidate of either arm, unless:
      1. a PERSISTED render-execution plan lock exists on disk (never an
         in-memory-only substitute -- run --prepare-protocol first),
      2. that persisted lock matches a freshly rebuilt expected plan
         (repo state has not drifted since it was written) and its own
         `status` is FROZEN,
      3. SOURCE_PAIR_EXECUTION_PARITY == 100% between the two arms' rows.
    Renders ORIGINAL then SHUFFLE (order does not matter scientifically --
    each arm's render_v2_arm call is fully independent and symmetric), then
    quality-matches each arm independently, writing additive v2 matched-bank
    locks only. NEVER trains, NEVER touches target, NEVER calls an LLM. Never
    (re)writes the execution-plan lock itself -- that is
    `write_render_execution_plan_lock`'s job, called only from
    `run_e6_v2_protocol_preparation`.
    """
    original = load_original_llm_recipes(repo)
    shuffle = e6r.verify_shuffle_recipe_source(repo)

    verification = verify_execution_plan_lock_matches_expected(repo)
    if not verification["EXECUTION_PLAN_LOCK_PRESENT"]:
        raise E6V2ExecutionError(
            "E6_V2_RENDER_EXECUTION_PLAN_LOCK is not persisted on disk; refusing to render. "
            f"{verification['reason']}")
    if not verification["EXECUTION_PLAN_LOCK_EXPECTED_EQUALS_PERSISTED"]:
        raise E6V2ExecutionError(
            "the persisted E6_V2_RENDER_EXECUTION_PLAN_LOCK does not match a freshly rebuilt expected "
            f"plan; refusing to render. {verification['reason']}")
    if verification["persisted_status"] != "FROZEN":
        raise E6V2ExecutionError(
            f"the persisted E6_V2_RENDER_EXECUTION_PLAN_LOCK has status={verification['persisted_status']!r}, "
            "not FROZEN; refusing to render (re-run --prepare-protocol once every v2 readiness gate "
            "passes)")

    original_plan = build_v2_arm_plan(repo, arm=ARM_ORIGINAL,
                                      recipe_content_identity=original["content_identity"],
                                      recipe_count=original["recipe_count"])
    shuffle_plan = build_v2_arm_plan(repo, arm=ARM_SHUFFLE,
                                     recipe_content_identity=shuffle["content_identity"],
                                     recipe_count=len(shuffle["recipes"]))

    original_rows = build_v2_arm_plan_rows(repo, arm=ARM_ORIGINAL,
                                           recipe_bank_identity=original_plan["recipe_bank_identity"],
                                           recipes=original["recipes"], plan=original_plan)
    shuffle_rows = build_v2_arm_plan_rows(repo, arm=ARM_SHUFFLE,
                                          recipe_bank_identity=shuffle_plan["recipe_bank_identity"],
                                          recipes=shuffle["recipes"], plan=shuffle_plan)
    parity = resolve_source_pair_execution_parity(original_rows, shuffle_rows)
    if not parity["all_positions_aligned"]:
        raise E6V2ExecutionError(
            f"SOURCE_PAIR_EXECUTION_PARITY is not 100% ({parity['source_pair_execution_parity_pct']}%); "
            "refusing to render before the first candidate")

    # the execution-plan lock is NOT (re)written here -- it was already
    # persisted and just verified above; --prepare-protocol is the one
    # writer.
    render_results: dict[str, Any] = {}
    matched_results: dict[str, Any] = {}
    for arm, plan, recipes in ((ARM_ORIGINAL, original_plan, original["recipes"]),
                              (ARM_SHUFFLE, shuffle_plan, shuffle["recipes"])):
        rendered = render_v2_arm(repo=repo, arm=arm, plan=plan, recipes=recipes,
                                 render_arm_fn=render_arm_fn)
        render_results[arm] = rendered
        matched = match_v2_arm(repo, arm=arm, plan=plan, rows=rendered["rows"], recipes=recipes,
                               metrics_provider=metrics_provider, quality_matcher=quality_matcher)
        matched_results[arm] = matched
        bank_lock = build_v2_matched_bank_lock(arm=arm, plan=plan, selected=matched["selected"])
        bank_lock_path = repo / v2_bank_lock_path(arm)
        bank_lock_path.parent.mkdir(parents=True, exist_ok=True)
        bank_lock_path.write_text(json.dumps(bank_lock, indent=2, default=str), encoding="utf-8")
        matched_results[arm]["bank_lock_path"] = str(bank_lock_path)

    post_render_gate = design_post_render_cross_arm_gate()

    return {
        "source_pair_execution_parity": parity,
        "render_results": {arm: {k: v for k, v in result.items() if k not in ("rows",)}
                          for arm, result in render_results.items()},
        "matched_results": matched_results,
        "post_render_cross_arm_gate": post_render_gate,
        "e6_v2_ready_for_training": post_render_gate["E6_V2_READY_FOR_TRAINING"],
        "target_access": False, "llm_api_calls": 0, "training_performed": False,
    }


def gpu_v2_protocol_preparation_command() -> str:
    """Prepared, never executed here. Lock-preparation only; no render."""
    return "python -m prism_fas.evaluation.c_ext_e6_v2_paired --prepare-protocol"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E6-v2 PAIRED_CURRENT_RUNTIME protocol preparation "
                                                 "(no render, no train, no target, no LLM)")
    parser.add_argument("--prepare-protocol", action="store_true",
                        help="Writes the six E6-v2 locks + readiness summary under "
                             f"{E6_V2_DIR}. Never renders, never trains.")
    parser.add_argument("--structural-preflight", action="store_true",
                        help="Read-only, CPU-only structural preflight (TASK M). Never declares GPU "
                             "hardware available; never renders.")
    parser.add_argument("--gpu-runtime-preflight", action="store_true",
                        help="Read-only GPU_RUNTIME_PREFLIGHT (CUDA, GPAT checkpoint, PhysicsEngine, "
                             "quality backends, source package, lock chain, output storage). Requires "
                             "NEITHER --execute NOR --authorize-gpu-render. Never renders, never "
                             "creates a candidate directory, never trains, never touches target, "
                             "never calls an LLM. Meant to be run on the GPU host.")
    parser.add_argument("--execute", action="store_true",
                        help="Render-execution intent flag. MUST be combined with "
                             "--authorize-gpu-render, or this refuses to run (fail closed). Neither "
                             "flag alone ever renders anything.")
    parser.add_argument("--authorize-gpu-render", action="store_true",
                        help="Explicit render authorization. MUST be combined with --execute, or this "
                             "refuses to run (fail closed).")
    parser.add_argument("--resume-preflight", action="store_true",
                        help="Read-only, recursive audit of the ACTUAL attempt-1 ORIGINAL candidate "
                             "tree (identity/route resolved from generation_identity, never a naive "
                             "top-level guess). Never renders, never trains, never touches target, "
                             "never calls an LLM. Requires neither --execute nor "
                             "--authorize-gpu-render.")
    parser.add_argument("--matching-preflight", action="store_true",
                        help="Read-only quality/matching diagnostic (TASK F): classifies every "
                             "GENERATED candidate of --matching-preflight-arm by route/domain/quality-"
                             "pass using the frozen quality stack, and reports common-source-domain "
                             "fillability under the frozen quota. Never renders, never selects/writes a "
                             "matched bank, never trains, never touches target, never calls an LLM. "
                             "Requires neither --execute nor --authorize-gpu-render.")
    parser.add_argument("--matching-preflight-arm", default=None,
                        help=f"Which v2 arm --matching-preflight inspects. Defaults to {ARM_ORIGINAL!r} "
                             "(the arm ATTEMPT-2 actually failed on).")
    parser.add_argument("--matching-sequence-preflight", action="store_true",
                        help="Read-only same-process diagnostic (TASK C): runs BOTH arms' matching "
                             "feasibility in ONE Python process in the exact production order, then "
                             "(unless --no-reverse-sequence) resets the quality-runtime cache and runs "
                             "the reverse order, reporting whether either arm's feasibility depends on "
                             "which order it ran in. Never renders, never selects/writes a bank, never "
                             "trains, never touches target, never calls an LLM.")
    parser.add_argument("--no-reverse-sequence", action="store_true",
                        help="With --matching-sequence-preflight, skip the reversed-order run.")
    parser.add_argument("--close-e6-v2", action="store_true",
                        help="Writes the additive E6-v2 scientific closure artifacts (Task A/B/C): "
                             "E6_V2_ATTEMPT3_POSTFIX_CONFIRMATION.json, E6_V2_FINAL_CLOSURE.json, "
                             "E6_V2_FINAL_SUMMARY.md. Never rewrites any prior lock or provenance "
                             "file; never renders, trains, touches target or calls an LLM.")
    args = parser.parse_args(argv)
    repo = cc.repo_root()

    if args.execute and not args.authorize_gpu_render:
        print("--execute requires --authorize-gpu-render as an explicit second flag. Refusing to run.")
        return 2
    if args.authorize_gpu_render and not args.execute:
        print("--authorize-gpu-render requires --execute as an explicit second flag. Refusing to run.")
        return 2

    if args.execute and args.authorize_gpu_render:
        try:
            result = run_v2_render_execution(repo)
        except E6V2ExecutionError as error:
            print(f"E6-v2 render execution refused: {error}")
            return 1
        print(json.dumps(result, indent=2, default=str))
        return 0

    if args.structural_preflight:
        print(json.dumps(structural_preflight_v2(repo), indent=2, default=str))
        return 0

    if args.gpu_runtime_preflight:
        print(json.dumps(gpu_runtime_preflight_v2(repo), indent=2, default=str))
        return 0

    if args.resume_preflight:
        print(json.dumps(audit_attempt1_original(repo), indent=2, default=str))
        return 0

    if args.matching_preflight:
        arm = args.matching_preflight_arm or ARM_ORIGINAL
        print(json.dumps(matching_preflight_v2(repo, arm=arm), indent=2, default=str))
        return 0

    if args.matching_sequence_preflight:
        print(json.dumps(matching_sequence_preflight_v2(repo, reverse_too=not args.no_reverse_sequence),
                        indent=2, default=str))
        return 0

    if args.close_e6_v2:
        write_attempt3_postfix_confirmation(repo)
        closure_result = write_e6_v2_final_closure(repo)
        summary_result = write_e6_v2_final_summary(repo)
        print(json.dumps({"closure_path": closure_result["path"], "summary_path": summary_result["path"],
                         "E6_V2_STATUS": closure_result["closure"]["E6_V2_STATUS"]}, indent=2, default=str))
        return 0

    if args.prepare_protocol:
        result = run_e6_v2_protocol_preparation(repo)
        print(json.dumps({"readiness": result["readiness"], "summary_path": result["summary_path"]},
                        indent=2, default=str))
        return 0

    print("Pass --prepare-protocol to write the six E6-v2 locks, --structural-preflight to check "
         "render readiness structurally, or --execute --authorize-gpu-render (both required) to "
         "render on the GPU host. No render happens without both execution flags.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
