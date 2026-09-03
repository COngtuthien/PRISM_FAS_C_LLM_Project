"""E6 - EXT-Q1Q2 LLM-SHUFFLE-A preparation (CPU only, no LLM, no GPU, no target).

Implements the constraint-preserving shuffle frozen at E0
(``reports/c_ext_q1q2_v1/e0/EXT_RECIPE_BINDING.json["llm_shuffle_groups"]``):
for each of the six frozen field-groups (medium, geometry, illumination,
region, artifact_family_and_parameters, severity_group), attempt up to
``10 * N`` (N=256) random pairwise swaps of that group's fields between two
recipes; accept a swap only if BOTH resulting recipes still validate against
the frozen ontology and schema. A rejected swap is simply not applied - the
working bank is fully valid at every step by induction from a valid start.

This is PREPARE-ONLY (spec section 13 step 2, EXT-H4 "LLM-original vs
LLM-SHUFFLE"): it reads the frozen, already-selected LLM bank
(``assets/recipe_banks/c3/llm/recipes.jsonl``) and writes a NEW,
extension-owned recipe/config/audit artifact under
``reports/c_ext_q1q2_v1/e6_llm_shuffle/``. It never writes to
``assets/recipe_banks/**`` (protected, frozen), never opens any target
artifact, never calls an LLM, and never synthesizes an image or trains a
detector.

Because every accepted transformation is a swap of two recipes' ALREADY-
PRESENT values for one group, the per-group field marginals are preserved
EXACTLY (a swap moves an existing pair of values around, never mints a new
one) - this is a structural guarantee, verified explicitly at the end rather
than merely asserted.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prism_fas.evaluation.c_ext_common import (  # noqa: E402
    EXT_ID, repo_root, read_json, read_jsonl, sha256_json, write_json_atomic,
    write_text_atomic,
)

ROOT = repo_root()
E6_DIR = "reports/c_ext_q1q2_v1/e6_llm_shuffle"
SEED = 20260911  # LLM-SHUFFLE-A, frozen at E0 (EXT_SEED_REGISTRY.json)
SWAP_ATTEMPTS_PER_N = 10


def _get_path(payload: dict, dotted: str) -> Any:
    cur: Any = payload
    for part in dotted.split("."):
        cur = cur[part]
    return cur


def _set_path(payload: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = payload
    for part in parts[:-1]:
        cur = cur[part]
    cur[parts[-1]] = value


def load_frozen_group_map() -> dict:
    binding = read_json(ROOT / "reports/c_ext_q1q2_v1/e0/EXT_RECIPE_BINDING.json")
    groups = binding["llm_shuffle_groups"]
    if groups["status"] != "FROZEN_AT_E0_FOR_E6":
        raise ValueError(f"llm_shuffle_groups is not frozen: {groups['status']!r}")
    return groups


def _clone(payload: dict) -> dict:
    return json.loads(json.dumps(payload))


def swap_group(a: dict, b: dict, fields: list[str]) -> tuple[dict, dict]:
    """Return NEW (a', b') dict copies with `fields` exchanged between a and b."""
    a2, b2 = _clone(a), _clone(b)
    for dotted in fields:
        va, vb = _get_path(a, dotted), _get_path(b, dotted)
        _set_path(a2, dotted, vb)
        _set_path(b2, dotted, va)
    return a2, b2


def validate_pair(a: dict, b: dict, ontology) -> bool:
    from prism_fas.recipes.validate import validate_payload

    _, issues_a = validate_payload(a, ontology)
    if issues_a:
        return False
    _, issues_b = validate_payload(b, ontology)
    return not issues_b


def field_multiset(bank: list[dict], dotted: str) -> list:
    out = []
    for rec in bank:
        try:
            out.append(json.dumps(_get_path(rec, dotted), sort_keys=True))
        except (KeyError, TypeError):
            out.append(None)
    return sorted(out, key=str)


def run_shuffle(bank: list[dict], group_map: dict, *, seed: int) -> dict:
    from prism_fas.recipes.ontology import load_ontology as load_pydantic_ontology

    ontology = load_pydantic_ontology(ROOT / "configs/recipes/ontology_m7.yaml")
    n = len(bank)
    working = [_clone(rec) for rec in bank]
    rng = random.Random(seed)

    per_group_stats = []
    for group in group_map["groups"]:
        fields = group_map["group_field_map"][group]
        attempts = SWAP_ATTEMPTS_PER_N * n
        accepted = 0
        for _ in range(attempts):
            i, j = rng.sample(range(n), 2)
            cand_i, cand_j = swap_group(working[i], working[j], fields)
            if validate_pair(cand_i, cand_j, ontology):
                working[i], working[j] = cand_i, cand_j
                accepted += 1
        per_group_stats.append({
            "group": group, "fields": fields, "swap_attempts": attempts,
            "accepted": accepted, "rejected": attempts - accepted,
            "acceptance_rate": accepted / attempts if attempts else None,
        })

    # ---- exact-marginal assertion: per-field multiset unchanged for every
    # touched field (structural, but verified rather than only asserted) ----
    marginal_checks = []
    all_fields = sorted({f for group in group_map["groups"]
                         for f in group_map["group_field_map"][group]})
    for dotted in all_fields:
        before = field_multiset(bank, dotted)
        after = field_multiset(working, dotted)
        marginal_checks.append({"field": dotted, "exact_marginal_preserved": before == after})

    # ---- joint cross-field structure changed: how many recipes differ from
    # their own original position on at least one shuffled field ----
    changed_recipes = sum(
        1 for orig, new in zip(bank, working)
        if any(_get_path(orig, f) != _get_path(new, f) for f in all_fields)
    )

    # ---- bank-level re-validation (schema, canonical, duplicate ids/hashes,
    # forbidden-shortcut policy) on the FINAL shuffled bank ----
    from prism_fas.recipes.schema import parse_recipe
    from prism_fas.recipes.validate import validate_recipes

    parsed = [parse_recipe(rec) for rec in working]
    bank_validation = validate_recipes(parsed, ontology)

    return {
        "n": n,
        "seed": seed,
        "groups_order": group_map["groups"],
        "per_group_stats": per_group_stats,
        "total_swap_attempts": sum(g["swap_attempts"] for g in per_group_stats),
        "total_accepted": sum(g["accepted"] for g in per_group_stats),
        "exact_marginal_assertion": {
            "all_fields_checked": all_fields,
            "all_preserved": all(c["exact_marginal_preserved"] for c in marginal_checks),
            "per_field": marginal_checks,
        },
        "joint_structure_changed": {
            "recipes_differing_from_original": changed_recipes,
            "recipes_total": n,
            "fraction_changed": changed_recipes / n if n else None,
        },
        "bank_level_validation": {
            "passed": bank_validation["passed"],
            "issue_count": bank_validation["issue_count"],
            "duplicate_recipe_ids": bank_validation["duplicate_recipe_ids"],
            "unique_recipe_hashes": bank_validation["unique_recipe_hashes"],
        },
        "working_bank": working,
        "ontology_identity": ontology.sha256,
    }


def main() -> int:
    group_map = load_frozen_group_map()
    bank = read_jsonl(ROOT / "assets/recipe_banks/c3/llm/recipes.jsonl")
    if len(bank) != 256:
        raise ValueError(f"expected N=256 selected LLM recipes, found {len(bank)}")

    result = run_shuffle(bank, group_map, seed=SEED)

    n_ok = len(result["working_bank"]) == 256
    every_valid = result["bank_level_validation"]["passed"]
    marginals_ok = result["exact_marginal_assertion"]["all_preserved"]
    joint_changed = result["joint_structure_changed"]["recipes_differing_from_original"] > 0
    no_dupes = not result["bank_level_validation"]["duplicate_recipe_ids"]
    unique_ok = result["bank_level_validation"]["unique_recipe_hashes"] == 256

    blocked_reasons = []
    if not n_ok:
        blocked_reasons.append(f"expected N=256, got {len(result['working_bank'])}")
    if not every_valid:
        blocked_reasons.append(
            f"{result['bank_level_validation']['issue_count']} bank-level validation issues")
    if not marginals_ok:
        blocked_reasons.append("exact-marginal assertion failed for at least one field")
    if not joint_changed:
        blocked_reasons.append("joint cross-field structure did not change (0 recipes differ)")
    if not no_dupes:
        blocked_reasons.append(f"duplicate recipe_ids: {result['bank_level_validation']['duplicate_recipe_ids']}")
    if not unique_ok:
        blocked_reasons.append("unique_recipe_hashes != 256")

    status = "PREPARED" if not blocked_reasons else "BLOCKED"

    shuffled_bank_identity = sha256_json(result["working_bank"])
    out_recipes_rel = f"{E6_DIR}/LLM_SHUFFLE_A_RECIPES.jsonl"
    write_text_atomic(
        out_recipes_rel,
        "\n".join(json.dumps(rec, sort_keys=True, separators=(",", ":"))
                  for rec in result["working_bank"]) + "\n",
    )

    audit = {
        "schema_version": "ext-q1q2-e6-llm-shuffle-a-v1",
        "extension_id": EXT_ID,
        "milestone": "E6",
        "hypothesis": "EXT-H4: LLM-original vs LLM-SHUFFLE-A",
        "variant": "LLM-SHUFFLE-A",
        "status": "E6_PREPARATION",
        "e6_preparation_status": status,
        "blocked_reasons": blocked_reasons,
        "gpu_used": False,
        "llm_api_calls": 0,
        "target_labels_accessed": False,
        "no_image_synthesis": True,
        "no_detector_training": True,
        "seed": SEED,
        "swap_attempts_per_group_formula": "10 * N, N=256 -> 2560 attempts/group",
        "groups_order": result["groups_order"],
        "group_field_map": group_map["group_field_map"],
        "per_group_stats": result["per_group_stats"],
        "total_swap_attempts": result["total_swap_attempts"],
        "total_accepted": result["total_accepted"],
        "exact_marginal_assertion": result["exact_marginal_assertion"],
        "joint_structure_changed": result["joint_structure_changed"],
        "bank_level_validation": result["bank_level_validation"],
        "n": result["n"],
        "source_bank": "assets/recipe_banks/c3/llm/recipes.jsonl",
        "source_bank_sha256": read_json(ROOT / "reports/c3/scientific/C3_SCIENTIFIC_BANK_LOCK.json")
        ["arms"]["LLM"]["selected_set_identity"],
        "ontology_identity": result["ontology_identity"],
        "shuffled_bank_identity": shuffled_bank_identity,
        "output_recipes": out_recipes_rel,
        "deterministic_identity_note": (
            "shuffled_bank_identity is a SHA-256 over the canonical JSON of the working "
            "bank after all group swaps; re-running this module with the same seed, same "
            "source bank and same frozen group map reproduces this identity exactly "
            "(random.Random(seed).sample draws are the only source of randomness, and "
            "every draw is deterministic given the seed)."
        ),
        "scientific_interpretation_boundary": (
            "This artifact is PREPARE-ONLY. No image was synthesized and no detector was "
            "trained from it. It becomes scientifically meaningful only once GPU-E "
            "(E6 LLM-SHUFFLE training) executes it under the frozen protocol."
        ),
    }
    write_json_atomic(f"{E6_DIR}/E6_LLM_SHUFFLE_A.json", audit)

    print(json.dumps({
        "e6_preparation_status": status,
        "blocked_reasons": blocked_reasons,
        "n": result["n"],
        "total_swap_attempts": result["total_swap_attempts"],
        "total_accepted": result["total_accepted"],
        "fraction_recipes_changed": round(
            result["joint_structure_changed"]["fraction_changed"], 4),
        "shuffled_bank_identity": shuffled_bank_identity,
    }, indent=2))
    return 0 if status == "PREPARED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
