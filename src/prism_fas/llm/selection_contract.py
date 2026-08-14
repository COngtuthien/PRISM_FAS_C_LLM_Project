"""C3_SELECTION_CONTRACT_IDENTITY and C3_BANK_CONTRACT_IDENTITY.

v1.5 §7.8.4:

    C3_GENERATION_CONTRACT_IDENTITY binds provider/model/API+SDK, thinking config,
    prompt bytes, request-template bytes, coverage-quota bytes, item schema, batch
    envelope, ontology, route policy, alias policy, retry policy and the 12x32
    schedule.

    C3_SELECTION_CONTRACT_IDENTITY separately binds prism_c3_selection_v1, all
    candidate-pool cardinalities, eligibility stages, hard/soft quotas, objective
    formulas and order, category ordering, canonical recipe identity and the final
    tie-break.

    C3_BANK_CONTRACT_IDENTITY = SHA256(canonical_json({generation_contract_identity,
                                                       selection_contract_identity}))

§21.3: a changed generation contract invalidates the affected raw archive and all
descendants; a changed selection contract invalidates selected-bank descendants
even when the raw candidates are unchanged. That is why the two are separate
identities rather than one.

Everything here is derived from live code and configuration — the selector
constants, the eligibility order, the ontology, the route policy and the arm
schedules each compute their own identity from what is actually on disk. Nothing
is transcribed from prose.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from prism_fas.llm.bank_lock import canonical_text, sha256_text
from prism_fas.llm.route_policy import RoutePolicy
from prism_fas.recipes import selection as sel
from prism_fas.recipes.arm_schedules import ArmSchedule, build_schedule
from prism_fas.recipes.eligibility import CONDITIONING_DIM, ELIGIBILITY_ORDER
from prism_fas.recipes.ontology import Ontology

SELECTION_CONTRACT_SCHEMA_VERSION = "c3-selection-contract-v1"

#: Source files whose bytes define the selector's behaviour. Their hashes enter
#: the identity, so a change to the implementation invalidates the contract.
IMPLEMENTATION_FILES: tuple[str, ...] = (
    "src/prism_fas/recipes/selection.py",
    "src/prism_fas/recipes/eligibility.py",
    "src/prism_fas/recipes/arm_schedules.py",
)

CONFIG_FILES: tuple[str, ...] = (
    "configs/version_c/llm/c3_selection_contract.yaml",
)


class SelectionContractError(RuntimeError):
    """The selection contract cannot be assembled."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_selection_contract(*, repo: Path, ontology: Ontology,
                             route_policy: RoutePolicy) -> dict[str, Any]:
    """Assemble the selection contract from live code and configuration."""
    categories = sel.axis_categories(ontology)
    rnd: ArmSchedule = build_schedule("RND", ontology)
    det: ArmSchedule = build_schedule("DET", ontology)

    implementation = {}
    for rel in IMPLEMENTATION_FILES + CONFIG_FILES:
        path = repo / rel
        if not path.exists():
            raise SelectionContractError(f"selection contract input missing: {rel}")
        implementation[rel] = file_sha256(path)

    material: dict[str, Any] = {
        "selection_version": sel.SELECTION_VERSION,

        # candidate-pool cardinalities (§7.8, §7.8.1)
        "raw_candidate_slots_per_arm": sel.RAW_CANDIDATE_SLOTS_PER_ARM,
        "minimum_eligible_pool_per_arm": sel.MINIMUM_ELIGIBLE_POOL_PER_ARM,
        "final_bank_size_per_arm": sel.FINAL_BANK_SIZE_PER_ARM,
        "arms": ["RND", "DET", "LLM"],

        # eligibility stages, in the governing order (§7.8.3)
        "eligibility_order": list(ELIGIBILITY_ORDER),
        "conditioning_dimension": CONDITIONING_DIM,

        # category ordering — the objectives iterate these
        "category_order": {axis: list(categories[axis]) for axis in sel.AXES},
        "single_valued_axes": list(sel.SINGLE_AXES),
        "multi_valued_axes": list(sel.MULTI_AXES),

        # hard and soft quotas (§7.8.2)
        "quotas": {axis: dict(sel.QUOTAS[axis]) for axis in sel.AXES},

        # objective formulas and their strict order (§7.8.3)
        "objective_order": ["hard_feasibility", "preferred_shortfall", "single_value_balance",
                            "multi_label_balance", "canonical_tie_break"],
        "stage_2_formula": "S_pref = sum_artifact max(0, 32 - count_a) "
                           "+ sum_region max(0, 24 - count_r)",
        "stage_3_formula": "S_single = sum_medium |5*count_m - 256| "
                           "+ sum_geometry |6*count_g - 256| "
                           "+ sum_illumination |6*count_i - 256|",
        "stage_4_formula": "S_multi = sum_artifact |8*count_a - A_total| "
                           "+ sum_region |9*count_r - R_total|",
        "stage_5_rule": "order eligible candidates by canonical recipe SHA-256 ascending and "
                        "choose the lexicographically smallest 256-element selected SHA set "
                        "among all subsets tied on stages 1-4",
        "arithmetic": "exact integer; every objective and count is recomputed from the "
                      "selected set in pure Python and must equal the solver's value",

        # canonical recipe identity rules (§7.8.3)
        "canonical_recipe_identity": {
            "method": "prism_fas.recipes.canonical.recipe_hash",
            "float_decimals": 6,
            "json": "sort_keys=True, separators=(',',':'), ensure_ascii=False",
            "algorithm": "SHA-256 over the canonical UTF-8 text",
            "deduplication_identity":
                "prism_fas.llm.pipeline.RecipePlanner.content_identity "
                "(positional recipe_id excluded)",
        },

        # control-arm schedules (§7.8.5)
        "rnd_schedule_identity": rnd.schedule_identity,
        "det_schedule_identity": det.schedule_identity,

        # the route contract the eligibility gate enforces (§7.3.1)
        "route_policy_identity": route_policy.route_policy_identity,
        "required_generator_route": list(route_policy.allowed_scientific_generator_route),

        # ontology the categories and ranges come from
        "ontology_identity": ontology.sha256,

        # implementation and config bytes
        "implementation_identities": implementation,

        # forbidden inputs, recorded so the prohibition is part of the identity
        "forbidden_selector_inputs": [
            "SiW-Mv2 information", "target metadata", "target metrics",
            "Version-B target error analysis", "human aesthetic preference",
            "manual ranking", "downstream detector score", "synthetic quality q",
            "quality-gate outcome", "provider response order",
            "input file order", "filesystem order", "thread scheduling",
            "solver traversal order",
        ],
    }
    return material


def selection_contract_identity(material: dict[str, Any]) -> str:
    return sha256_text(canonical_text(material))


def bank_contract_identity(*, generation_contract_identity: str,
                           selection_contract_identity: str) -> str:
    """C3_BANK_CONTRACT_IDENTITY, exactly as §7.8.4 defines it."""
    payload = {
        "generation_contract_identity": generation_contract_identity,
        "selection_contract_identity": selection_contract_identity,
    }
    return sha256_text(canonical_text(payload))


def assemble(*, repo: Path, ontology: Ontology, route_policy: RoutePolicy,
             generation_contract_identity: str) -> dict[str, Any]:
    """The full selection + bank contract record."""
    material = build_selection_contract(repo=repo, ontology=ontology,
                                        route_policy=route_policy)
    selection_identity = selection_contract_identity(material)
    bank_identity = bank_contract_identity(
        generation_contract_identity=generation_contract_identity,
        selection_contract_identity=selection_identity)
    return {
        "schema_version": SELECTION_CONTRACT_SCHEMA_VERSION,
        "material": material,
        "canonical_text": canonical_text(material),
        "c3_selection_contract_identity": selection_identity,
        "c3_generation_contract_identity": generation_contract_identity,
        "c3_bank_contract_identity": bank_identity,
        "bank_contract_definition":
            "SHA256(canonical_json({generation_contract_identity, "
            "selection_contract_identity})) per v1.5 §7.8.4",
        "invalidation_rule":
            "a changed generation contract invalidates the affected raw archive and all "
            "descendants; a changed selection contract invalidates selected-bank descendants "
            "even if the raw candidates are unchanged (§21.3)",
    }


__all__ = ["SELECTION_CONTRACT_SCHEMA_VERSION", "IMPLEMENTATION_FILES", "CONFIG_FILES",
           "SelectionContractError", "file_sha256", "build_selection_contract",
           "selection_contract_identity", "bank_contract_identity", "assemble"]
