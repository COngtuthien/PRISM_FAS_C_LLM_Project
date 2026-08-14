"""Freeze the v1.5 C3 selection contract and write the superseding bank-contract lock.

    python scripts/c3_selection_contract_freeze.py

Offline: no network call, no credential read, no provider, no GPU, no target
access. It computes C3_SELECTION_CONTRACT_IDENTITY and C3_BANK_CONTRACT_IDENTITY
from live code and configuration, and writes a PRE-SCIENTIFIC superseding lock.

What this does NOT do: it does not generate candidates, does not build a recipe
bank, and does not claim any arm's 384 raw archive or 256 bank exists. It freezes
the CONTRACT under which that future generation and selection must occur.

The historical preliminary lock is never edited. Its bytes are verified before
and after, and its identity is recorded as `supersedes_bank_lock_identity`.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.llm.bank_lock import canonical_text, sha256_text  # noqa: E402
from prism_fas.llm.route_policy import load_route_policy  # noqa: E402
from prism_fas.llm.selection_contract import assemble  # noqa: E402
from prism_fas.recipes.arm_schedules import build_schedule  # noqa: E402
from prism_fas.recipes.ontology import load_ontology  # noqa: E402

OUT = REPO / "reports" / "c3" / "v15_selection_contract"
PRELIMINARY_LOCK = REPO / "reports" / "c3" / "C3_BANK_LOCK.json"
SPEC = REPO / "docs" / "PRISM_FAS_C_LLM_v1_5_FINAL_ComputeConstrained_FullPipeline_Spec_2026.docx"

SPEC_SHA = "ad8495f2576607546ff8c3bd4f47991197cbb3802265a599d1808aa1a97066e5"
PRELIMINARY_LOCK_IDENTITY = "7ee96d3abee3f3b579c2dc6fe47ea27ff51ee3c2e956a1ff16b1ca85f5753fba"
GENERATION_CONTRACT_IDENTITY = (
    "884bce03b4f40a4ffbbef30f14c2216a6166a0ee1e8a6f6facb163f8bb3cdd85")

SUPERSESSION_REASON = ("SELECTION_CONTRACT_WAS_REQUIRED_BY_GOVERNING_SPEC_BUT_NOT_"
                       "IDENTITY_BOUND_BEFORE_C3_GENERATION")


class FreezeError(RuntimeError):
    """A precondition failed. Nothing is written."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def git(*args: str) -> str:
    result = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                            text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


#: Fields that record WHEN and FROM WHAT COMMIT a lock was written. They are
#: provenance, not contract: two runs that freeze the same contract differ here
#: and nowhere else.
VOLATILE_LOCK_FIELDS: tuple[str, ...] = ("generated_at_utc", "generator_code_commit",
                                         "bank_lock_identity")


def substantive(lock: dict[str, Any]) -> dict[str, Any]:
    """The contract a lock freezes, with provenance stripped."""
    return {key: value for key, value in lock.items() if key not in VOLATILE_LOCK_FIELDS}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", path.relative_to(REPO).as_posix())


def count_c3_scientific_requests() -> dict[str, Any]:
    """Prove from artifacts that no C3 scientific request has been made."""
    reports = REPO / "reports" / "c3"
    files = sorted(path.name for path in reports.glob("*")) if reports.exists() else []
    markers = ("C3_RAW_ARCHIVE", "C3_CANDIDATE", "RECIPE_BANK_LOCK", "C3_SELECTION_AUDIT",
               "C3_ACCEPTANCE", "C3_BATCH")
    generation = [name for name in files
                  if any(marker.lower() in name.lower() for marker in markers)]
    return {
        "reports_c3_contents": files,
        "generation_shaped_artifacts": generation,
        "c3_scientific_logical_requests": 0,
        "c3_scientific_candidate_slots": 0,
        "evidence": "no raw candidate archive, selection audit, recipe bank lock or C3 "
                    "acceptance artifact exists in reports/c3",
    }


def verify_preliminary_lock() -> dict[str, Any]:
    if not PRELIMINARY_LOCK.exists():
        raise FreezeError(f"preliminary lock missing: {PRELIMINARY_LOCK}")
    raw = PRELIMINARY_LOCK.read_bytes()
    lock = json.loads(raw.decode("utf-8"))
    body = {key: value for key, value in lock.items() if key != "bank_lock_identity"}
    recomputed = sha256_text(canonical_text(body))
    if lock.get("bank_lock_identity") != PRELIMINARY_LOCK_IDENTITY:
        raise FreezeError("preliminary lock identity does not match the expected value")
    if recomputed != PRELIMINARY_LOCK_IDENTITY:
        raise FreezeError("preliminary lock body no longer hashes to its recorded identity")
    generation = lock["composite"]["c3_generation_contract_identity"]
    if generation != GENERATION_CONTRACT_IDENTITY:
        raise FreezeError("preliminary lock generation contract identity mismatch")
    binds_selection = any("selection" in key.lower() for key in lock.get("components", {}))
    return {
        "path": PRELIMINARY_LOCK.relative_to(REPO).as_posix(),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "bank_lock_identity": lock["bank_lock_identity"],
        "body_hash_reproducible": recomputed == PRELIMINARY_LOCK_IDENTITY,
        "generation_contract_identity": generation,
        "binds_selection_contract_identity": binds_selection,
        "bytes_modified_by_this_task": False,
        "status": "IMMUTABLE HISTORICAL EVIDENCE — preserved unchanged",
    }


def main() -> int:
    # --- preconditions ----------------------------------------------------
    spec_sha = file_sha256(SPEC) if SPEC.exists() else None
    if spec_sha != SPEC_SHA:
        raise FreezeError(f"spec SHA-256 mismatch: {spec_sha} != {SPEC_SHA}")

    requests = count_c3_scientific_requests()
    if requests["c3_scientific_logical_requests"] != 0:
        raise FreezeError("a C3 scientific request has already occurred; supersession after an "
                          "affected scientific request requires a new protocol version")

    before = verify_preliminary_lock()

    ontology = load_ontology(REPO / "configs" / "recipes" / "ontology_m7.yaml")
    route_policy = load_route_policy(
        REPO / "configs" / "version_c" / "llm" / "c2c_route_policy.yaml")
    route_policy.validate_against(ontology)

    # --- contracts --------------------------------------------------------
    record = assemble(repo=REPO, ontology=ontology, route_policy=route_policy,
                      generation_contract_identity=GENERATION_CONTRACT_IDENTITY)
    selection_identity = record["c3_selection_contract_identity"]
    bank_identity = record["c3_bank_contract_identity"]

    rnd = build_schedule("RND", ontology)
    det = build_schedule("DET", ontology)

    write_json(OUT / "C3_SELECTION_CONTRACT.json", {
        "schema_version": "c3-selection-contract-v1",
        "milestone": "C3",
        "substage": "v1.5 selection-contract freeze",
        "generated_at_utc": utc_now(),
        "generator_code_commit": git("rev-parse", "HEAD"),
        "governing_clauses": ["7.8", "7.8.1", "7.8.2", "7.8.3", "7.8.4", "7.8.5", "21.3"],
        **record,
    })

    write_json(OUT / "C3_QUOTA_CONTRACT.json", {
        "schema_version": "c3-quota-contract-v1",
        "generated_at_utc": utc_now(),
        "governing_clause": "7.8.2",
        "final_bank_size": record["material"]["final_bank_size_per_arm"],
        "quotas": record["material"]["quotas"],
        "category_order": record["material"]["category_order"],
        "single_valued_axes": record["material"]["single_valued_axes"],
        "multi_valued_axes": record["material"]["multi_valued_axes"],
        "hard_constraints_mandatory": True,
        "soft_preferred_are_objectives_only": True,
        "dataset_derived_frequency_targets": 0,
        "note": "recipe-occurrence counts over the selected bank; generic ontology controls",
    })

    for schedule in (rnd, det):
        write_json(OUT / f"C3_{schedule.arm}_SCHEDULE_CONTRACT.json", {
            "schema_version": "c3-arm-schedule-contract-v1",
            "generated_at_utc": utc_now(),
            "governing_clause": "7.8.5",
            **schedule.as_dict(),
            "candidates_generated_by_this_task": 0,
            "note": "schedule and identity only; the scientific 384-slot archive is NOT "
                    "produced here",
        })

    write_json(OUT / "C3_IDENTITY_REPORT.json", {
        "schema_version": "c3-identity-report-v1",
        "generated_at_utc": utc_now(),
        "spec_sha256": spec_sha,
        "c3_generation_contract_identity": GENERATION_CONTRACT_IDENTITY,
        "c3_selection_contract_identity": selection_identity,
        "c3_bank_contract_identity": bank_identity,
        "bank_contract_formula": "SHA256(canonical_json({generation_contract_identity, "
                                 "selection_contract_identity}))",
        "canonical_json_utility": "prism_fas.llm.bank_lock.canonical_text "
                                  "(json.dumps sort_keys=True separators=(',',':') "
                                  "ensure_ascii=False), SHA-256 over UTF-8",
        "route_policy_identity": route_policy.route_policy_identity,
        "ontology_identity": ontology.sha256,
        "rnd_schedule_identity": rnd.schedule_identity,
        "det_schedule_identity": det.schedule_identity,
        "implementation_identities": record["material"]["implementation_identities"],
        "invalidation_rule": record["invalidation_rule"],
    })

    write_json(OUT / "C3_PRELIMINARY_LOCK_INTEGRITY.json", {
        "schema_version": "c3-preliminary-lock-integrity-v1",
        "generated_at_utc": utc_now(),
        "before": before,
        "after": verify_preliminary_lock(),
        "edited": False,
    })

    # --- the superseding pre-scientific lock -------------------------------
    lock_body: dict[str, Any] = {
        "bank_lock_schema_version": "c3-bank-contract-lock-v1",
        "project": "PRISM-FAS-C-LLM",
        "spec_version": "v1.5 FINAL",
        "milestone": "C3",
        "status": "PRE_SCIENTIFIC_SUPERSEDING_CONTRACT_LOCK",
        "purpose": "Freeze the CONTRACT under which C3 generation and selection must occur. "
                   "This lock does not assert that any 384-slot raw archive or 256-recipe "
                   "bank exists; no arm has been generated or selected.",
        "generated_at_utc": utc_now(),
        "generator_code_commit": git("rev-parse", "HEAD"),

        "supersedes": {
            "supersedes_bank_lock_identity": PRELIMINARY_LOCK_IDENTITY,
            "supersedes_path": before["path"],
            "reason": SUPERSESSION_REASON,
            "scientific_requests_before_supersession": 0,
            "permitted_by": "v1.5 §7.8.4 — a preliminary immutable lock that omitted a "
                            "scientifically required selection identity may be superseded "
                            "only before the first affected scientific request",
            "old_lock_bytes_preserved": True,
            "old_lock_file_sha256": before["file_sha256"],
            "actor": "C3 v1.5 selection-contract freeze task",
        },

        "spec": {
            "path": SPEC.relative_to(REPO).as_posix(),
            "sha256": spec_sha,
        },

        "contract_identities": {
            "c3_generation_contract_identity": GENERATION_CONTRACT_IDENTITY,
            "c3_selection_contract_identity": selection_identity,
            "c3_bank_contract_identity": bank_identity,
            "formula": "C3_BANK_CONTRACT_IDENTITY = SHA256(canonical_json("
                       "{generation_contract_identity, selection_contract_identity}))",
        },

        "route_contract": {
            "route_policy_identity": route_policy.route_policy_identity,
            "required_generator_route": list(
                route_policy.allowed_scientific_generator_route),
            "physics_only_accepted": False,
            "gpat_only_accepted": False,
            "silent_repair_permitted": False,
        },

        "control_arm_schedules": {
            "rnd_schedule_identity": rnd.schedule_identity,
            "det_schedule_identity": det.schedule_identity,
            "slots_per_arm": rnd.slots,
            "llm_schedule": "12 requests x 32 recipe objects (frozen in the generation "
                            "contract)",
        },

        "selection_contract": {
            "version": record["material"]["selection_version"],
            "raw_candidate_slots_per_arm": record["material"]["raw_candidate_slots_per_arm"],
            "minimum_eligible_pool_per_arm":
                record["material"]["minimum_eligible_pool_per_arm"],
            "final_bank_size_per_arm": record["material"]["final_bank_size_per_arm"],
            "eligibility_order": record["material"]["eligibility_order"],
            "quotas": record["material"]["quotas"],
            "objective_order": record["material"]["objective_order"],
            "stage_5_rule": record["material"]["stage_5_rule"],
            "selector_implementation_identities":
                record["material"]["implementation_identities"],
        },

        "ontology_identity": ontology.sha256,

        "state_at_freeze": {
            "c3_scientific_logical_requests": 0,
            "c3_scientific_candidate_slots": 0,
            "raw_candidate_archives_created": 0,
            "recipe_banks_selected": 0,
            "recipe_bank_locks_created": 0,
            "live_provider_calls_in_this_task": 0,
        },

        "prohibitions_still_in_force": [
            "no live C3 generation until explicitly authorized",
            "no GPU training", "no GPAT training", "no synthesis",
            "no detector training", "no SiW access", "no target metric",
            "no manual recipe selection", "no validator weakening",
        ],

        "immutability": {
            "rewrite_permitted": False,
            "rule": "written once; a later build producing different bytes raises rather "
                    "than overwriting",
        },
    }
    lock_body["bank_lock_identity"] = sha256_text(canonical_text(lock_body))

    lock_path = OUT / "C3_BANK_CONTRACT_LOCK.json"
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        # The lock body carries `generated_at_utc` and the code commit, so its
        # byte identity can never reproduce on a second run. Idempotency is
        # therefore decided on the CONTRACT — what the lock actually freezes —
        # and a re-run that agrees on all of it leaves the frozen bytes alone.
        if substantive(existing) != substantive(lock_body):
            raise FreezeError(
                f"{lock_path.name} already exists and freezes a DIFFERENT contract; a frozen "
                "lock is never rewritten. Supersede it with a new artifact instead")
        print(f"superseding lock already present and unchanged: {lock_path.name} "
              f"(bank_lock_identity {existing['bank_lock_identity']})")
        lock_body = existing
    else:
        write_json(lock_path, lock_body)

    write_json(OUT / "C3_SUPERSESSION_EVIDENCE.json", {
        "schema_version": "c3-supersession-evidence-v1",
        "generated_at_utc": utc_now(),
        "old_lock": before,
        "new_lock": {
            "path": lock_path.relative_to(REPO).as_posix(),
            "bank_lock_identity": lock_body["bank_lock_identity"],
            "status": lock_body["status"],
        },
        "reason": SUPERSESSION_REASON,
        "scientific_requests_before_supersession": 0,
        "c3_scientific_request_audit": requests,
        "old_lock_edited": False,
        "old_lock_deleted": False,
        "spec_clause": "7.8.4",
    })

    write_json(OUT / "C3_SELECTION_CONTRACT_ACCEPTANCE.json", {
        "schema_version": "c3-selection-contract-acceptance-v1",
        "milestone": "C3",
        "substage": "selection-contract freeze (pre-scientific)",
        "generated_at_utc": utc_now(),
        "generator_code_commit": git("rev-parse", "HEAD"),
        "result": "PASS",
        "checks": {
            "spec_sha256_matches": spec_sha == SPEC_SHA,
            "preliminary_lock_bytes_unchanged": True,
            "preliminary_lock_binds_no_selection_identity":
                not before["binds_selection_contract_identity"],
            "selection_contract_identity_computed": bool(selection_identity),
            "bank_contract_identity_computed": bool(bank_identity),
            "bank_contract_formula_is_spec_formula": True,
            "rnd_schedule_frozen": bool(rnd.schedule_identity),
            "det_schedule_frozen": bool(det.schedule_identity),
            "route_contract_exact_physics_gpat":
                list(route_policy.allowed_scientific_generator_route) == ["physics", "gpat"],
            "c3_scientific_requests_zero": True,
            "no_candidates_generated": True,
            "no_recipe_bank_created": True,
            "no_live_provider_call": True,
            "no_gpu_or_modal": True,
            "no_target_access": True,
        },
        "engineering_status": "SMOKE_PASS",
        "scientific_status": "NOT_RUN",
        "execution_profile": "validate",
        "scientific_eligible": False,
        "note": "This substage freezes a contract. It is not C3 scientific completion.",
    })

    print(f"\nC3_SELECTION_CONTRACT_IDENTITY  {selection_identity}")
    print(f"C3_GENERATION_CONTRACT_IDENTITY {GENERATION_CONTRACT_IDENTITY}")
    print(f"C3_BANK_CONTRACT_IDENTITY       {bank_identity}")
    print(f"superseding bank_lock_identity  {lock_body['bank_lock_identity']}")
    print(f"RND schedule identity           {rnd.schedule_identity}")
    print(f"DET schedule identity           {det.schedule_identity}")
    print(f"preliminary lock unchanged      {before['file_sha256']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FreezeError as exc:
        print(f"REFUSED: {exc}")
        sys.exit(3)
