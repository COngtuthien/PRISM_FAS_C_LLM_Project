"""Build the three C3 scientific arms, run eligibility, select and freeze.

    python scripts/c3_scientific_arms.py --stage eligibility
    python scripts/c3_scientific_arms.py --stage freeze

Offline. This script makes NO provider call: the LLM arm is reconstructed from
the raw archives that live generation already wrote, and RND/DET come from their
frozen offline schedules. It computes nothing scientific of its own — the
eligibility pipeline, the selector and the identities all come from the frozen
modules.

The LLM arm's 384 slots are read back out of `reports/c3/live/raw_responses/`
rather than held in memory from the generating run, so the pool that reaches
selection is provably the pool that was archived.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.llm.bank_lock import canonical_text, sha256_text  # noqa: E402
from prism_fas.llm.route_policy import load_route_policy  # noqa: E402
from prism_fas.pipeline.adapters.c3_live import LiveGenerationState  # noqa: E402
from prism_fas.pipeline.state import atomic_write_json  # noqa: E402
from prism_fas.recipes.arm_schedules import build_schedule, draft_schedule, slot_ids  # noqa: E402
from prism_fas.recipes.canonical import canonical_json  # noqa: E402
from prism_fas.recipes.eligibility import ELIGIBILITY_ORDER, evaluate_pool  # noqa: E402
from prism_fas.recipes.ontology import load_ontology  # noqa: E402
from prism_fas.recipes.selection import (FINAL_BANK_SIZE_PER_ARM,  # noqa: E402
                                         MINIMUM_ELIGIBLE_POOL_PER_ARM,
                                         RAW_CANDIDATE_SLOTS_PER_ARM, select)

LIVE_DIR = REPO / "reports" / "c3" / "live"
OUT = REPO / "reports" / "c3" / "scientific"
BANKS = REPO / "assets" / "recipe_banks" / "c3"
ARMS = ("LLM", "RND", "DET")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def llm_slots() -> tuple[list[str], list[dict[str, Any] | None]]:
    """The 384 LLM slots, read back from the archived responses.

    Slot order is the frozen plan order, and every slot is accounted for: a
    response that yielded fewer than 32 objects would leave `None` entries,
    which the eligibility pipeline rejects at `item_schema` rather than
    silently shortening the pool.
    """
    state = LiveGenerationState.load(LIVE_DIR / "C3_LIVE_GENERATION_STATE.json")
    ids: list[str] = []
    payloads: list[dict[str, Any] | None] = []

    for record in state.requests:
        if not record.complete:
            raise SystemExit(
                f"{record.logical_request_id} is {record.status}; the LLM arm is "
                "incomplete and cannot enter eligibility")
        archive = LIVE_DIR / "raw_responses" / f"{record.logical_request_id}.json"
        stored = json.loads(archive.read_text(encoding="utf-8"))
        recipes = json.loads(stored["raw_response"])["recipes"]
        for offset in range(record.slot_count):
            ids.append(f"LLM_SLOT_{record.slot_start + offset:03d}")
            payloads.append(recipes[offset] if offset < len(recipes) else None)

    if len(ids) != RAW_CANDIDATE_SLOTS_PER_ARM:
        raise SystemExit(f"LLM arm produced {len(ids)} slots, expected "
                         f"{RAW_CANDIDATE_SLOTS_PER_ARM}")
    return ids, payloads


def control_slots(arm: str, ontology: Any) -> tuple[list[str], list[dict[str, Any]]]:
    """RND or DET, from the frozen offline schedule. No provider call."""
    drafted = list(draft_schedule(arm, ontology))
    ids = [slot for slot, _payload in drafted]
    payloads = [payload for _slot, payload in drafted]
    if ids != slot_ids(arm, RAW_CANDIDATE_SLOTS_PER_ARM):
        raise SystemExit(f"{arm} slot ids do not match the frozen schedule")
    return ids, payloads


def build_pools(ontology: Any, route_policy: Any) -> dict[str, Any]:
    pools: dict[str, Any] = {}
    for arm in ARMS:
        ids, payloads = (llm_slots() if arm == "LLM"
                         else control_slots(arm, ontology))
        pools[arm] = evaluate_pool(
            arm=arm, candidates=payloads, slot_ids=ids, ontology=ontology,
            route_policy=route_policy, bank_id=f"c3_{arm.lower()}",
            raw_slots=RAW_CANDIDATE_SLOTS_PER_ARM,
            minimum_required=MINIMUM_ELIGIBLE_POOL_PER_ARM)
    return pools


def rejection_breakdown(pool: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for verdict in pool.verdicts:
        if verdict.eligible:
            continue
        counts[verdict.rejected_at or "unknown"] = (
            counts.get(verdict.rejected_at or "unknown", 0) + 1)
    return dict(sorted(counts.items()))


def eligibility_report(pools: dict[str, Any]) -> dict[str, Any]:
    arms = {}
    for arm, pool in pools.items():
        arms[arm] = {
            "raw_slots": pool.raw_slots,
            "eligible": len(pool.recipes),
            "minimum_required": pool.minimum_required,
            "meets_minimum": len(pool.recipes) >= pool.minimum_required,
            "rejected": pool.raw_slots - len(pool.recipes),
            "rejected_by_stage": rejection_breakdown(pool),
        }
    return {
        "schema_version": "c3-scientific-eligibility-v1",
        "generated_at_utc": utc_now(),
        "execution_profile": "full",
        "scientific_eligible": True,
        "eligibility_order": list(ELIGIBILITY_ORDER),
        "arms": arms,
        "all_arms_meet_minimum": all(item["meets_minimum"] for item in arms.values()),
    }


def fairness_report(pools: dict[str, Any], ontology: Any, route_policy: Any) -> dict[str, Any]:
    """Prove the three arms ran the same contract, differing only in generator."""
    shared = {
        "raw_slots_per_arm": RAW_CANDIDATE_SLOTS_PER_ARM,
        "minimum_eligible_pool": MINIMUM_ELIGIBLE_POOL_PER_ARM,
        "final_bank_size": FINAL_BANK_SIZE_PER_ARM,
        "ontology_identity": ontology.sha256,
        "route_policy_identity": route_policy.route_policy_identity,
        "required_generator_route": list(route_policy.allowed_scientific_generator_route),
        "eligibility_order": list(ELIGIBILITY_ORDER),
        "selector": "prism_c3_selection_v1",
    }
    per_arm = {arm: {"raw_slots": pool.raw_slots,
                     "minimum_required": pool.minimum_required}
               for arm, pool in pools.items()}
    identical = (len({pool.raw_slots for pool in pools.values()}) == 1
                 and len({pool.minimum_required for pool in pools.values()}) == 1)
    return {
        "schema_version": "c3-fairness-audit-v1",
        "generated_at_utc": utc_now(),
        "shared_contract": shared,
        "per_arm": per_arm,
        "all_arms_share_the_contract": identical,
        "generator_identity_intentionally_differs": True,
        "generators": {
            "LLM": "frozen 12x32 live Gemini schedule",
            "RND": build_schedule("RND", ontology).schedule_identity,
            "DET": build_schedule("DET", ontology).schedule_identity,
        },
        "target_information_used": False,
    }


def run_eligibility() -> int:
    ontology = load_ontology(REPO / "configs" / "recipes" / "ontology_m7.yaml")
    route_policy = load_route_policy(
        REPO / "configs" / "version_c" / "llm" / "c2c_route_policy.yaml")
    route_policy.validate_against(ontology)

    pools = build_pools(ontology, route_policy)
    report = eligibility_report(pools)
    atomic_write_json(OUT / "C3_ELIGIBILITY.json", report)
    atomic_write_json(OUT / "C3_FAIRNESS_AUDIT.json",
                      fairness_report(pools, ontology, route_policy))

    for arm, item in report["arms"].items():
        flag = "OK " if item["meets_minimum"] else "FAIL"
        print(f"{flag} {arm:4} raw={item['raw_slots']} eligible={item['eligible']} "
              f"(min {item['minimum_required']})  rejected={item['rejected']} "
              f"{item['rejected_by_stage']}")
    print("all arms meet minimum:", report["all_arms_meet_minimum"])
    return 0 if report["all_arms_meet_minimum"] else 1


def run_freeze() -> int:
    ontology = load_ontology(REPO / "configs" / "recipes" / "ontology_m7.yaml")
    route_policy = load_route_policy(
        REPO / "configs" / "version_c" / "llm" / "c2c_route_policy.yaml")
    route_policy.validate_against(ontology)

    pools = build_pools(ontology, route_policy)
    for arm, pool in pools.items():
        if len(pool.recipes) < pool.minimum_required:
            print(f"C3_{arm}_POOL_BELOW_{pool.minimum_required}: "
                  f"{len(pool.recipes)} eligible")
            return 1

    lock = json.loads((REPO / "reports/c3/C3_BANK_LOCK.json").read_text(encoding="utf-8"))
    contract = json.loads(
        (REPO / "reports/c3/v15_selection_contract/C3_BANK_CONTRACT_LOCK.json")
        .read_text(encoding="utf-8"))["contract_identities"]
    quota_sha = sha256_text(
        (REPO / "reports/c3/live/C3_QUOTA_SNAPSHOT.json").read_text(encoding="utf-8"))

    banks: dict[str, Any] = {}
    for arm in ARMS:
        pool = pools[arm]
        result = select(pool.recipes, ontology)
        if len(result.selected_shas) != FINAL_BANK_SIZE_PER_ARM:
            print(f"C3_{arm}_SELECTED_{len(result.selected_shas)}_EXPECTED_"
                  f"{FINAL_BANK_SIZE_PER_ARM}")
            return 1

        eligible_ids = [verdict.canonical_sha256 for verdict in pool.verdicts
                        if verdict.eligible]
        selected_ids = sorted(result.selected_shas)
        payload = {
            "schema_version": "c3-scientific-bank-v1",
            "generated_at_utc": utc_now(),
            "execution_profile": "full",
            "scientific_eligible": True,
            "arm": arm,
            "raw_slots": pool.raw_slots,
            "eligible_pool_size": len(pool.recipes),
            "minimum_eligible_pool": pool.minimum_required,
            "selected": len(result.selected_shas),
            "eligible_pool_identity": sha256_text(canonical_text(sorted(eligible_ids))),
            "selected_set_identity": result.selected_set_identity,
            "selected_recipe_identities": selected_ids,
            "selection_audit": {
                "s_pref": result.s_pref, "s_single": result.s_single,
                "s_multi": result.s_multi, "counts": result.counts,
                "eligible_count": result.eligible_count,
                "tie_break_trace": result.tie_break_trace,
                "rejected_count": len(result.rejected_shas),
            },
            "generator": ("frozen 12x32 live Gemini schedule" if arm == "LLM"
                          else build_schedule(arm, ontology).schedule_identity),
            "c3_generation_contract_identity":
                lock["composite"]["c3_generation_contract_identity"],
            "c3_selection_contract_identity":
                contract["c3_selection_contract_identity"],
            "c3_bank_contract_identity": contract["c3_bank_contract_identity"],
            "ontology_identity": ontology.sha256,
            "route_policy_identity": route_policy.route_policy_identity,
            "required_generator_route":
                list(route_policy.allowed_scientific_generator_route),
            "eligibility_order": list(ELIGIBILITY_ORDER),
            "selector": "prism_c3_selection_v1",
            "quota_snapshot_sha256": quota_sha if arm == "LLM" else None,
            "spec_sha256": "ad8495f2576607546ff8c3bd4f47991197cbb3802265a599d1808aa1a97066e5",
        }
        # The identity covers the SCIENCE, not the clock. Hashing the whole
        # payload would fold `generated_at_utc` in, so re-freezing the identical
        # bank an hour later would produce a different identity — which defeats
        # the point of an identity you can re-derive and compare.
        payload["bank_identity_material"] = [
            "arm", "raw_slots", "eligible_pool_size", "minimum_eligible_pool", "selected",
            "eligible_pool_identity", "selected_set_identity", "selected_recipe_identities",
            "generator", "c3_generation_contract_identity", "c3_selection_contract_identity",
            "c3_bank_contract_identity", "ontology_identity", "route_policy_identity",
            "required_generator_route", "eligibility_order", "selector", "spec_sha256",
        ]
        payload["bank_identity"] = sha256_text(canonical_text(
            {key: payload[key] for key in payload["bank_identity_material"]}))
        banks[arm] = payload

        atomic_write_json(BANKS / arm.lower() / "C3_BANK.json", payload)
        recipes_path = BANKS / arm.lower() / "recipes.jsonl"
        recipes_path.parent.mkdir(parents=True, exist_ok=True)
        # The frozen canonical form, not an ad-hoc dump: these are the exact
        # bytes the recipe identities were computed over, so anything else would
        # produce a bank file that disagrees with its own hashes.
        # newline="" suppresses Windows CRLF translation. .gitattributes pins
        # assets/recipe_banks/** to LF because a bank's bytes are hashed, so a
        # CRLF file on disk would differ from the LF file Git stores and check
        # out — the same file, two hashes, depending on where you read it.
        with recipes_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("".join(canonical_json(recipe) + "\n"
                                 for recipe in result.selected_recipes))
        print(f"OK  {arm:4} raw={pool.raw_slots} eligible={len(pool.recipes)} "
              f"selected={len(result.selected_shas)} "
              f"bank_identity={payload['bank_identity'][:16]}")

    final = {
        "schema_version": "c3-scientific-bank-lock-v1",
        "project": "PRISM-FAS-C-LLM",
        "milestone": "C3",
        "status": "FROZEN_SCIENTIFIC_BANKS",
        "generated_at_utc": utc_now(),
        "execution_profile": "full",
        "scientific_eligible": True,
        "arms": {arm: {"bank_identity": banks[arm]["bank_identity"],
                       "selected_set_identity": banks[arm]["selected_set_identity"],
                       "eligible_pool_identity": banks[arm]["eligible_pool_identity"],
                       "raw_slots": banks[arm]["raw_slots"],
                       "eligible": banks[arm]["eligible_pool_size"],
                       "selected": banks[arm]["selected"]}
                 for arm in ARMS},
        "c3_generation_contract_identity":
            lock["composite"]["c3_generation_contract_identity"],
        "c3_selection_contract_identity": contract["c3_selection_contract_identity"],
        "c3_bank_contract_identity": contract["c3_bank_contract_identity"],
        "llm_schedule": {"logical_requests": 12, "objects_per_request": 32,
                         "raw_slots": 384, "provider_attempts": 13,
                         "extra_logical_requests": 0},
        "quota_snapshot_sha256": quota_sha,
        "supersedes": {
            "preliminary_bank_lock": "7ee96d3abee3f3b579c2dc6fe47ea27ff51ee3c2e956a1ff16b1ca85f5753fba",
            "pre_scientific_contract_lock":
                "1acdf68f56195f1b568449b545865ae2868d99d480ed6b75b28215178c5e9628",
            "rule": "both are historical evidence and are preserved unchanged; this lock "
                    "records the ACTUAL scientific banks they governed",
        },
        "immutability": {"rewrite_permitted": False},
    }
    # Same rule for the governing lock: identity over the frozen science, with
    # the timestamp recorded beside it rather than inside it.
    final["lock_identity_material"] = [
        "arms", "c3_generation_contract_identity", "c3_selection_contract_identity",
        "c3_bank_contract_identity", "llm_schedule", "quota_snapshot_sha256", "supersedes",
    ]
    final["lock_identity"] = sha256_text(canonical_text(
        {key: final[key] for key in final["lock_identity_material"]}))
    atomic_write_json(OUT / "C3_SCIENTIFIC_BANK_LOCK.json", final)
    print("final lock identity:", final["lock_identity"])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("eligibility", "freeze"), required=True)
    args = parser.parse_args()
    return run_eligibility() if args.stage == "eligibility" else run_freeze()


if __name__ == "__main__":
    raise SystemExit(main())
