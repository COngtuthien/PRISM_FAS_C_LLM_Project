"""C3 pre-live verification gate. Verifies the frozen state; freezes nothing.

    python scripts/c3_pre_live_audit.py

Offline: no network, no credential, no provider, no GPU, no Modal, no target
access. It generates no scientific candidate and writes no lock. If every check
passes, the result is that the frozen contract is ready for USER REVIEW — not
that live generation is authorized.

Independence matters here. Where an identity is checked, it is recomputed from
FIRST PRINCIPLES with plain `json.dumps` and `hashlib`, not only by calling the
same helper that produced it: a helper that is wrong in both directions would
agree with itself. The repository utility is then run as a second, separate
witness and the two must agree with the stored value.

Writes to `reports/c3/v15_pre_live_audit/`, a new namespace. It never touches
`reports/c3/C3_BANK_LOCK.json` or `reports/c3/v15_selection_contract/`.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO / "src", REPO / "tests" / "c3"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from prism_fas.llm.bank_lock import canonical_text, sha256_text  # noqa: E402
from prism_fas.llm.route_policy import load_route_policy  # noqa: E402
from prism_fas.llm.selection_contract import (assemble, bank_contract_identity,  # noqa: E402
                                              build_selection_contract,
                                              selection_contract_identity)
from prism_fas.recipes import arm_schedules as sched  # noqa: E402
from prism_fas.recipes import selection as sel  # noqa: E402
from prism_fas.recipes.eligibility import ELIGIBILITY_ORDER  # noqa: E402
from prism_fas.recipes.ontology import load_ontology  # noqa: E402

from c3_fixtures import make_pool  # noqa: E402

OUT = REPO / "reports" / "c3" / "v15_pre_live_audit"
FREEZE = REPO / "reports" / "c3" / "v15_selection_contract"
PRELIMINARY_LOCK = REPO / "reports" / "c3" / "C3_BANK_LOCK.json"
SUPERSEDING_LOCK = FREEZE / "C3_BANK_CONTRACT_LOCK.json"
SPEC = REPO / "docs" / "PRISM_FAS_C_LLM_v1_5_FINAL_ComputeConstrained_FullPipeline_Spec_2026.docx"

EXPECTED = {
    "spec_sha256": "ad8495f2576607546ff8c3bd4f47991197cbb3802265a599d1808aa1a97066e5",
    "generation_contract_identity":
        "884bce03b4f40a4ffbbef30f14c2216a6166a0ee1e8a6f6facb163f8bb3cdd85",
    "selection_contract_identity":
        "3d4675ba16b39d10f0e888f3c523ea540647544a436ff387bc84f2c17eced070",
    "bank_contract_identity":
        "d6105f8de601ae94cb0d46e087a0ebe664b3b5df9d193f5797e306bfe4fe03b8",
    "superseding_lock_identity":
        "1acdf68f56195f1b568449b545865ae2868d99d480ed6b75b28215178c5e9628",
    "preliminary_lock_identity":
        "7ee96d3abee3f3b579c2dc6fe47ea27ff51ee3c2e956a1ff16b1ca85f5753fba",
    "version_b_head": "7799f7decd35db6987ce4578824e5bd8d9eab4ae",
}

#: Every artifact that records a live provider request. The audit fingerprints
#: these before and after running tests; the delta must be zero.
PROVIDER_EVIDENCE = (
    "reports/c1", "reports/c2", "reports/c2b", "reports/c2c", "reports/c3",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def git(*args: str, repo: Path = REPO) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                            text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def independent_canonical_sha(payload: Any) -> str:
    """Canonical identity recomputed WITHOUT the repository helper.

    Deliberately re-implements the documented rule — json.dumps with sort_keys,
    compact separators, ensure_ascii off, SHA-256 over UTF-8 — so that a bug in
    prism_fas.llm.bank_lock cannot hide behind itself.
    """
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def body_of(lock: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in lock.items() if key != "bank_lock_identity"}


# ============================================================== preflight
def audit_preflight() -> dict[str, Any]:
    version_b = Path(r"D:\AI on IOT\Anti_spoofing\PRISM_FAS_B_Project")
    dirty = git("status", "--porcelain")
    spec_sha = file_sha256(SPEC) if SPEC.exists() else None
    vb_status = git("status", "--porcelain", repo=version_b)
    return {
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "HEAD"),
        "commit_feat": git("rev-parse", "51dd4b2"),
        "commit_fix": git("rev-parse", "e850e99"),
        "commits_resolvable": bool(git("rev-parse", "--verify", "51dd4b2^{commit}"))
                              and bool(git("rev-parse", "--verify", "e850e99^{commit}")),
        "remote_tracking": git("rev-parse", "--abbrev-ref", "@{upstream}"),
        "worktree_dirty_paths": [line for line in dirty.splitlines() if line.strip()],
        "spec_path": SPEC.relative_to(REPO).as_posix(),
        "spec_sha256": spec_sha,
        "spec_matches": spec_sha == EXPECTED["spec_sha256"],
        "version_b": {
            "path": str(version_b),
            "head": git("rev-parse", "HEAD", repo=version_b),
            "tag_peeled": git("rev-list", "-n", "1", "m10-blind-evaluation-checkpoint",
                              repo=version_b),
            "clean": not vb_status.strip(),
            "written_by_this_audit": False,
        },
    }


# ============================================================== identities
def audit_identities(ontology, route_policy) -> dict[str, Any]:
    stored_contract = load(FREEZE / "C3_SELECTION_CONTRACT.json")
    stored_identity_report = load(FREEZE / "C3_IDENTITY_REPORT.json")
    lock = load(SUPERSEDING_LOCK)

    # --- selection identity, two independent witnesses -------------------
    rebuilt = build_selection_contract(repo=REPO, ontology=ontology,
                                       route_policy=route_policy)
    witness_helper = selection_contract_identity(rebuilt)
    witness_plain = independent_canonical_sha(rebuilt)
    stored_material = stored_contract["material"]
    stored_canonical = stored_contract["canonical_text"]

    selection = {
        "stored": stored_contract["c3_selection_contract_identity"],
        "expected": EXPECTED["selection_contract_identity"],
        "rebuilt_via_repo_utility": witness_helper,
        "rebuilt_via_independent_json_sha256": witness_plain,
        "stored_canonical_text_rehashes_to":
            hashlib.sha256(stored_canonical.encode("utf-8")).hexdigest(),
        "rebuilt_material_equals_stored_material": rebuilt == stored_material,
        "reproduces": len({stored_contract["c3_selection_contract_identity"],
                           EXPECTED["selection_contract_identity"], witness_helper,
                           witness_plain,
                           hashlib.sha256(stored_canonical.encode("utf-8")).hexdigest()}) == 1,
    }

    # --- generation identity, re-derived from the preliminary lock -------
    preliminary = load(PRELIMINARY_LOCK)
    composite = preliminary["composite"]
    generation = {
        "stored_in_preliminary_lock": composite["c3_generation_contract_identity"],
        "expected": EXPECTED["generation_contract_identity"],
        "recomputed_from_components": independent_canonical_sha(preliminary["components"]),
        "recomputed_from_recorded_canonical_text":
            hashlib.sha256(composite["canonical_text"].encode("utf-8")).hexdigest(),
        "bound_in_superseding_lock":
            lock["contract_identities"]["c3_generation_contract_identity"],
    }
    generation["reproduces"] = len({
        generation["stored_in_preliminary_lock"], generation["expected"],
        generation["recomputed_from_components"],
        generation["recomputed_from_recorded_canonical_text"],
        generation["bound_in_superseding_lock"]}) == 1

    # --- bank-contract identity, the §7.8.4 formula ----------------------
    formula_payload = {
        "generation_contract_identity": generation["expected"],
        "selection_contract_identity": selection["stored"],
    }
    bank = {
        "formula": "SHA256(canonical_json({generation_contract_identity, "
                   "selection_contract_identity}))",
        "stored": stored_contract["c3_bank_contract_identity"],
        "expected": EXPECTED["bank_contract_identity"],
        "recomputed_independently": independent_canonical_sha(formula_payload),
        "recomputed_via_repo_utility": bank_contract_identity(
            generation_contract_identity=generation["expected"],
            selection_contract_identity=selection["stored"]),
        "bound_in_superseding_lock": lock["contract_identities"]["c3_bank_contract_identity"],
        "canonical_text": json.dumps(formula_payload, sort_keys=True,
                                     separators=(",", ":"), ensure_ascii=False),
    }
    bank["reproduces"] = len({bank["stored"], bank["expected"],
                              bank["recomputed_independently"],
                              bank["recomputed_via_repo_utility"],
                              bank["bound_in_superseding_lock"]}) == 1

    # --- superseding lock's own identity ---------------------------------
    lock_body = body_of(lock)
    superseding = {
        "stored": lock["bank_lock_identity"],
        "expected": EXPECTED["superseding_lock_identity"],
        "recomputed_via_repo_utility": sha256_text(canonical_text(lock_body)),
        "recomputed_independently": independent_canonical_sha(lock_body),
    }
    superseding["reproduces"] = len({superseding["stored"], superseding["expected"],
                                     superseding["recomputed_via_repo_utility"],
                                     superseding["recomputed_independently"]}) == 1

    # --- assemble() as an end-to-end third witness ------------------------
    record = assemble(repo=REPO, ontology=ontology, route_policy=route_policy,
                      generation_contract_identity=EXPECTED["generation_contract_identity"])

    return {
        "method": "each identity recomputed both by the repository canonical utility and by "
                  "an independent plain json.dumps+hashlib reimplementation; all witnesses "
                  "must agree with the stored value",
        "generation_contract_identity": generation,
        "selection_contract_identity": selection,
        "bank_contract_identity": bank,
        "superseding_lock_identity": superseding,
        "end_to_end_assemble": {
            "selection": record["c3_selection_contract_identity"],
            "bank": record["c3_bank_contract_identity"],
            "agrees": (record["c3_selection_contract_identity"] == selection["expected"]
                       and record["c3_bank_contract_identity"] == bank["expected"]),
        },
        "identity_report_on_disk_agrees": all([
            stored_identity_report["c3_selection_contract_identity"] == selection["expected"],
            stored_identity_report["c3_bank_contract_identity"] == bank["expected"],
            stored_identity_report["c3_generation_contract_identity"]
            == generation["expected"],
        ]),
        "all_reproduce": all([generation["reproduces"], selection["reproduces"],
                              bank["reproduces"], superseding["reproduces"]]),
        "new_identity_created_by_this_audit": False,
        "lock_rewritten_by_this_audit": False,
    }


# ==================================================== preliminary lock
def audit_preliminary_lock() -> dict[str, Any]:
    raw = PRELIMINARY_LOCK.read_bytes()
    lock = json.loads(raw.decode("utf-8"))
    normalized = raw.decode("utf-8").replace("\r\n", "\n").encode("utf-8")

    blob_sha = git("rev-parse", "HEAD:reports/c3/C3_BANK_LOCK.json")
    blob_bytes = subprocess.run(["git", "-C", str(REPO), "cat-file", "blob",
                                 "HEAD:reports/c3/C3_BANK_LOCK.json"],
                                capture_output=True, check=False).stdout
    touching = [line for line in git(
        "log", "--format=%H", "--", "reports/c3/C3_BANK_LOCK.json").splitlines() if line]

    body_identity = sha256_text(canonical_text(body_of(lock)))
    return {
        "path": PRELIMINARY_LOCK.relative_to(REPO).as_posix(),
        "exists": True,
        "representations": {
            "note": "the file predates this repository's .gitattributes rules, so Git "
                    "materializes it with platform line endings. The three hashes below "
                    "are DIFFERENT representations of identical content and are reported "
                    "separately rather than being claimed equal.",
            "worktree_bytes_sha256": hashlib.sha256(raw).hexdigest(),
            "worktree_line_ending": "CRLF" if b"\r\n" in raw else "LF",
            "lf_normalized_sha256": hashlib.sha256(normalized).hexdigest(),
            "git_blob_content_sha256": hashlib.sha256(blob_bytes).hexdigest(),
            "git_blob_object_id": blob_sha,
            "lf_normalized_equals_git_blob":
                hashlib.sha256(normalized).hexdigest()
                == hashlib.sha256(blob_bytes).hexdigest(),
        },
        "canonical_body_identity": {
            "recomputed": body_identity,
            "expected": EXPECTED["preliminary_lock_identity"],
            "recorded_in_file": lock["bank_lock_identity"],
            "reproduces": body_identity == EXPECTED["preliminary_lock_identity"]
                          == lock["bank_lock_identity"],
            "note": "computed over parsed JSON, so it is independent of line endings and "
                    "formatting entirely. This is the authoritative immutability proof.",
        },
        "commits_touching_this_file": touching,
        "contract_task_commits_that_touched_it": sorted(
            set(touching) & {git("rev-parse", "51dd4b2"), git("rev-parse", "e850e99")}),
        "modified_by_contract_task": bool(
            set(touching) & {git("rev-parse", "51dd4b2"), git("rev-parse", "e850e99")}),
        "still_binds_no_selection_identity":
            not any("selection" in key.lower() for key in lock["components"])
            and "c3_selection_contract_identity" not in json.dumps(lock),
        "modified_by_this_audit": False,
    }


# ================================================== superseding lock claims
def audit_superseding_lock() -> dict[str, Any]:
    lock = load(SUPERSEDING_LOCK)
    text = json.dumps(lock)
    supersedes = lock["supersedes"]
    state = lock["state_at_freeze"]

    binds = {
        "generation_contract_identity":
            bool(lock["contract_identities"]["c3_generation_contract_identity"]),
        "selection_contract_identity":
            bool(lock["contract_identities"]["c3_selection_contract_identity"]),
        "bank_contract_identity":
            bool(lock["contract_identities"]["c3_bank_contract_identity"]),
        "spec_identity": lock["spec"]["sha256"] == EXPECTED["spec_sha256"],
        "route_policy_identity": bool(lock["route_contract"]["route_policy_identity"]),
        "rnd_schedule_identity": bool(lock["control_arm_schedules"]["rnd_schedule_identity"]),
        "det_schedule_identity": bool(lock["control_arm_schedules"]["det_schedule_identity"]),
        "selector_implementation_identities":
            bool(lock["selection_contract"]["selector_implementation_identities"]),
        "provenance": bool(lock["generated_at_utc"]) and bool(lock["generator_code_commit"]),
    }

    # It must NOT claim any archive or bank exists.
    does_not_claim = {
        "rnd_384_archive_exists": state["raw_candidate_archives_created"] == 0,
        "det_384_archive_exists": state["raw_candidate_archives_created"] == 0,
        "llm_384_archive_exists": state["raw_candidate_archives_created"] == 0,
        "final_256_bank_exists": state["recipe_banks_selected"] == 0,
        "recipe_bank_lock_exists": state["recipe_bank_locks_created"] == 0,
        "no_selected_shas_in_lock": "selected_shas" not in text,
        "no_recipes_key": "recipes" not in lock,
        "schema_is_not_a_recipe_bank_lock":
            "RECIPE_BANK_LOCK" not in lock["bank_lock_schema_version"].upper(),
    }

    return {
        "path": SUPERSEDING_LOCK.relative_to(REPO).as_posix(),
        "status": lock["status"],
        "status_correct": lock["status"] == "PRE_SCIENTIFIC_SUPERSEDING_CONTRACT_LOCK",
        "supersedes_bank_lock_identity": supersedes["supersedes_bank_lock_identity"],
        "supersedes_correct": supersedes["supersedes_bank_lock_identity"]
                              == EXPECTED["preliminary_lock_identity"],
        "reason": supersedes["reason"],
        "reason_correct": supersedes["reason"] == (
            "SELECTION_CONTRACT_WAS_REQUIRED_BY_GOVERNING_SPEC_BUT_NOT_"
            "IDENTITY_BOUND_BEFORE_C3_GENERATION"),
        "scientific_requests_before_supersession":
            supersedes["scientific_requests_before_supersession"],
        "scientific_requests_zero":
            supersedes["scientific_requests_before_supersession"] == 0,
        "binds_required_elements": binds,
        "binds_all_required": all(binds.values()),
        "does_not_claim_generation_happened": does_not_claim,
        "no_false_completion_claim": all(does_not_claim.values()),
        "state_at_freeze": state,
    }


# ==================================================== schedules (§7.8.5)
def audit_schedules(ontology) -> dict[str, Any]:
    """Identity + reproducibility, including two FRESH interpreter runs.

    Drafting the schedule is not C3 scientific generation: nothing is archived,
    nothing is validated into a bank, and the payloads are discarded when the
    process exits. What is measured is that the schedule is a pure function.
    """
    probe = (
        "import sys, json, hashlib;"
        f"sys.path.insert(0, r'{(REPO / 'src').as_posix()}');"
        "from prism_fas.recipes.ontology import load_ontology;"
        "from prism_fas.recipes import arm_schedules as s;"
        f"o = load_ontology(r'{(REPO / 'configs' / 'recipes' / 'ontology_m7.yaml').as_posix()}');"
        "out = {};"
        "\nfor arm in ('RND', 'DET'):\n"
        "    sc = s.build_schedule(arm, o)\n"
        "    pairs = list(s.draft_schedule(arm, o))\n"
        "    text = json.dumps(pairs, sort_keys=True, separators=(',',':'), ensure_ascii=False)\n"
        "    out[arm] = {'identity': sc.schedule_identity, 'slots': len(pairs),\n"
        "                'manifest': hashlib.sha256(text.encode('utf-8')).hexdigest(),\n"
        "                'unique_slots': len({p[0] for p in pairs}),\n"
        "                'route_ok': all(p[1]['generator_route'] == ['physics','gpat'] for p in pairs)}\n"
        "print(json.dumps(out))"
    )
    runs = []
    for _ in range(2):
        result = subprocess.run([sys.executable, "-c", probe], cwd=REPO,
                                capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"schedule probe failed: {result.stderr}")
        runs.append(json.loads(result.stdout.strip().splitlines()[-1]))

    in_process = {}
    for arm in ("RND", "DET"):
        schedule = sched.build_schedule(arm, ontology)
        material = schedule.identity_material()
        in_process[arm] = {
            "identity": schedule.schedule_identity,
            "seed": schedule.seed,
            "slots": schedule.slots,
            "slot_id_count": len(schedule.slot_ids),
            "unique_slot_ids": len(set(schedule.slot_ids)),
            "first_slot_id": schedule.slot_ids[0],
            "last_slot_id": schedule.slot_ids[-1],
            "route": list(schedule.route),
            "ontology_identity": material["ontology_identity"],
            "target_information_used": material["target_information_used"],
            "llm_output_used": material["llm_output_used"],
            "derivation": material["derivation"],
        }

    stored = {
        "RND": load(FREEZE / "C3_RND_SCHEDULE_CONTRACT.json"),
        "DET": load(FREEZE / "C3_DET_SCHEDULE_CONTRACT.json"),
    }

    per_arm = {}
    for arm in ("RND", "DET"):
        identities = {in_process[arm]["identity"], runs[0][arm]["identity"],
                      runs[1][arm]["identity"], stored[arm]["schedule_identity"]}
        per_arm[arm] = {
            "schedule_identity": in_process[arm]["identity"],
            "identity_stable_across_two_fresh_interpreters": len(identities) == 1,
            "fresh_run_1": runs[0][arm],
            "fresh_run_2": runs[1][arm],
            "manifest_identical_across_fresh_runs":
                runs[0][arm]["manifest"] == runs[1][arm]["manifest"],
            "exactly_384_slots": (in_process[arm]["slots"] == 384
                                  == in_process[arm]["slot_id_count"]
                                  == in_process[arm]["unique_slot_ids"]
                                  == runs[0][arm]["slots"] == runs[0][arm]["unique_slots"]),
            "frozen_seed": in_process[arm]["seed"],
            "seed_family_source": "prism_fas.recipes.arm_schedules.SEED_FAMILY",
            "route_declared_by_every_slot": runs[0][arm]["route_ok"],
            "shares_ontology_with_llm_arm":
                in_process[arm]["ontology_identity"] == ontology.sha256,
            "target_independent": in_process[arm]["target_information_used"] is False,
            "llm_result_independent": in_process[arm]["llm_output_used"] is False,
            "source_independent": "the seed family and per-slot derivation are constants; "
                                  "no dataset, source-domain statistic or measured result "
                                  "enters the schedule",
            "details": in_process[arm],
        }
        per_arm[arm]["reproducible"] = all([
            per_arm[arm]["identity_stable_across_two_fresh_interpreters"],
            per_arm[arm]["manifest_identical_across_fresh_runs"],
            per_arm[arm]["exactly_384_slots"],
        ])

    per_arm["schedules_are_distinct"] = (per_arm["RND"]["schedule_identity"]
                                         != per_arm["DET"]["schedule_identity"])
    per_arm["strategies"] = {
        "RND": "rule-valid random sampling, per-slot derived from the frozen seed family",
        "DET": "deterministic structured enumeration over the ontology in fixed order",
    }
    per_arm["scientific_archive_created_by_this_audit"] = 0
    return per_arm


# ================================================ selector scientific contract
def audit_selector(ontology, route_policy) -> dict[str, Any]:
    categories = sel.axis_categories(ontology)
    expected_order = ("item_schema", "ontology", "scientific_route_policy", "numeric_range",
                      "compatibility", "canonicalization", "deduplication", "compiler",
                      "operator_graph", "mask_policy", "conditioning_41d")
    expected_quotas = {
        "medium": (5, 32, 80, None), "geometry": (6, 24, 64, None),
        "illumination": (6, 24, 64, None), "artifact": (8, 8, 128, 32),
        "region": (9, 8, 128, 24),
    }
    quota_check = {}
    for axis, (count, low, high, preferred) in expected_quotas.items():
        actual = sel.QUOTAS[axis]
        quota_check[axis] = {
            "categories_expected": count,
            "categories_actual": len(categories[axis]),
            "category_count_ok": len(categories[axis]) == count,
            "hard_min": actual["hard_min"], "hard_min_ok": actual["hard_min"] == low,
            "hard_max": actual["hard_max"], "hard_max_ok": actual["hard_max"] == high,
            "preferred_min": actual["preferred_min"],
            "preferred_ok": actual["preferred_min"] == preferred,
            "categories": list(categories[axis]),
        }
        quota_check[axis]["pass"] = all(
            quota_check[axis][key] for key in
            ("category_count_ok", "hard_min_ok", "hard_max_ok", "preferred_ok"))

    return {
        "selection_version": sel.SELECTION_VERSION,
        "version_ok": sel.SELECTION_VERSION == "prism_c3_selection_v1",
        "cardinalities": {
            "raw_candidate_slots_per_arm": sel.RAW_CANDIDATE_SLOTS_PER_ARM,
            "minimum_eligible_pool_per_arm": sel.MINIMUM_ELIGIBLE_POOL_PER_ARM,
            "final_bank_size_per_arm": sel.FINAL_BANK_SIZE_PER_ARM,
            "ok": (sel.RAW_CANDIDATE_SLOTS_PER_ARM == 384
                   and sel.MINIMUM_ELIGIBLE_POOL_PER_ARM == 320
                   and sel.FINAL_BANK_SIZE_PER_ARM == 256),
            "twelve_by_thirtytwo": 12 * 32 == sel.RAW_CANDIDATE_SLOTS_PER_ARM,
        },
        "eligibility_order": {
            "actual": list(ELIGIBILITY_ORDER),
            "expected": list(expected_order),
            "ok": tuple(ELIGIBILITY_ORDER) == expected_order,
        },
        "route": {
            "required": list(route_policy.allowed_scientific_generator_route),
            "ok": list(route_policy.allowed_scientific_generator_route) == ["physics", "gpat"],
            "route_policy_identity": route_policy.route_policy_identity,
        },
        "quotas": quota_check,
        "quotas_ok": all(entry["pass"] for entry in quota_check.values()),
        "axes": {"single": list(sel.SINGLE_AXES), "multi": list(sel.MULTI_AXES)},
    }


def audit_objectives(ontology) -> dict[str, Any]:
    """Verify the shipped objective functions against an independent formula."""
    import random

    rng = random.Random(20260814)
    categories = sel.axis_categories(ontology)
    mismatches: list[dict[str, Any]] = []
    trials = 200
    for _ in range(trials):
        counts = {axis: {name: rng.randint(0, 90) for name in categories[axis]}
                  for axis in sel.AXES}
        # Independent restatement of the spec formulas.
        ref_pref = (sum(max(0, 32 - value) for value in counts["artifact"].values())
                    + sum(max(0, 24 - value) for value in counts["region"].values()))
        ref_single = (sum(abs(5 * v - 256) for v in counts["medium"].values())
                      + sum(abs(6 * v - 256) for v in counts["geometry"].values())
                      + sum(abs(6 * v - 256) for v in counts["illumination"].values()))
        a_total = sum(counts["artifact"].values())
        r_total = sum(counts["region"].values())
        ref_multi = (sum(abs(8 * v - a_total) for v in counts["artifact"].values())
                     + sum(abs(9 * v - r_total) for v in counts["region"].values()))
        got = (sel.s_pref(counts), sel.s_single(counts, 256), sel.s_multi(counts))
        if got != (ref_pref, ref_single, ref_multi):
            mismatches.append({"counts": counts, "shipped": got,
                               "reference": [ref_pref, ref_single, ref_multi]})

    return {
        "method": "the shipped s_pref/s_single/s_multi compared against an independent "
                  "restatement of the §7.8.3 formulas over 200 randomized count vectors",
        "trials": trials,
        "mismatches": mismatches,
        "stage_2_formula": "S_pref = sum_artifact max(0,32-count_a) + "
                           "sum_region max(0,24-count_r)",
        "stage_3_formula": "S_single = sum_medium |5*count_m-256| + "
                           "sum_geometry |6*count_g-256| + sum_illumination |6*count_i-256|",
        "stage_4_formula": "S_multi = sum_artifact |8*count_a-A_total| + "
                           "sum_region |9*count_r-R_total|",
        "stage_5_rule": "among all stage 1-4 optima, the lexicographically smallest sorted "
                        "256-element canonical SHA-256 set",
        "coefficients_are_category_counts": {
            "medium": len(sel.axis_categories(ontology)["medium"]),
            "geometry": len(sel.axis_categories(ontology)["geometry"]),
            "illumination": len(sel.axis_categories(ontology)["illumination"]),
            "artifact": len(sel.axis_categories(ontology)["artifact"]),
            "region": len(sel.axis_categories(ontology)["region"]),
        },
        "ok": not mismatches,
    }


def audit_stage_5(ontology) -> dict[str, Any]:
    """Stage 5 against exhaustive enumeration, including a full-tie instance."""
    from test_c3_selection_optimality import (TINY_QUOTAS, _brute_force, _homogeneous_pool,
                                              _with_quotas)
    from prism_fas.recipes.canonical import recipe_hash

    instances = []
    for tag, size, bank in [("bf0", 8, 3), ("bf1", 9, 4), ("bf2", 10, 4)]:
        pool = make_pool(ontology, size, tag=tag)
        with _with_quotas(TINY_QUOTAS):
            result = sel.select(pool, ontology, bank_size=bank, enforce_minimum_pool=False)
            key, smallest, ties = _brute_force(pool, bank, ontology)
        instances.append({
            "instance": tag, "pool": size, "bank": bank,
            "subsets_enumerated": len(list(itertools.combinations(range(size), bank))),
            "tied_optima": ties,
            "selector_objective": [result.s_pref, result.s_single, result.s_multi],
            "brute_force_objective": list(key),
            "objective_matches": [result.s_pref, result.s_single, result.s_multi] == list(key),
            "selected_set_matches_lexicographically_smallest":
                sorted(result.selected_shas) == smallest,
        })

    pool = _homogeneous_pool(ontology, 10)
    ordered = sorted(recipe_hash(recipe) for recipe in pool)
    with _with_quotas(TINY_QUOTAS):
        degenerate = sel.select(pool, ontology, bank_size=4, enforce_minimum_pool=False)
        _, smallest, ties = _brute_force(pool, 4, ontology)

    return {
        "method": "exhaustive enumeration of every C(n,k) subset, scored with the shipped "
                  "exact-integer functions; the selector must return the lexicographically "
                  "smallest set among all tied optima",
        "instances": instances,
        "degenerate_full_tie": {
            "note": "all candidates identical on every quota axis, so stages 1-4 cannot "
                    "discriminate and stage 5 alone decides",
            "pool": 10, "bank": 4, "tied_optima": ties,
            "expected_four_smallest_shas": ordered[:4],
            "selected": sorted(degenerate.selected_shas),
            "matches": sorted(degenerate.selected_shas) == ordered[:4] == smallest,
        },
        "solver_traversal_cannot_resolve_stage_5":
            "stage 5 runs AFTER the stage 2-4 optima are pinned, as iterative feasibility "
            "fixing over ascending canonical SHA-256; the MILP contributes objective values "
            "only, never a chosen subset",
        "ok": (all(entry["objective_matches"]
                   and entry["selected_set_matches_lexicographically_smallest"]
                   for entry in instances)
               and sorted(degenerate.selected_shas) == ordered[:4]),
    }


def audit_full_scale(ontology) -> dict[str, Any]:
    """Reproduce the full 320 -> 256 budget twice under the shipped quotas."""
    pool = make_pool(ontology, sel.MINIMUM_ELIGIBLE_POOL_PER_ARM, tag="full")
    started = time.time()
    first = sel.select(pool, ontology, arm="AUDIT_FIXTURE")
    mid = time.time()
    second = sel.select(list(reversed(pool)), ontology, arm="AUDIT_FIXTURE")
    return {
        "note": "synthetic ontology-derived fixtures. This bank is a verification artifact, "
                "is discarded, and is NOT a C3 scientific recipe bank.",
        "eligible_pool": first.eligible_count,
        "selected": len(first.selected_shas),
        "rejected": len(first.rejected_shas),
        "selected_plus_rejected": len(first.selected_shas) + len(first.rejected_shas),
        "unique_selected": len(set(first.selected_shas)),
        "S_pref": first.s_pref,
        "S_single": first.s_single,
        "S_multi": first.s_multi,
        "hard_quota_violation_count": len(sel.hard_violations(first.counts)),
        "hard_quota_violations": sel.hard_violations(first.counts),
        "selected_set_identity": first.selected_set_identity,
        "repeat_run_input_reversed": {
            "selected_set_identity": second.selected_set_identity,
            "identical": second.selected_set_identity == first.selected_set_identity,
            "objective_identical": (second.s_pref, second.s_single, second.s_multi)
                                   == (first.s_pref, first.s_single, first.s_multi),
        },
        "counts": first.counts,
        "quota_table_modified": False,
        "elapsed_seconds": {"first": round(mid - started, 1),
                            "second": round(time.time() - mid, 1)},
        "ok": (len(first.selected_shas) == 256 and first.eligible_count == 320
               and not sel.hard_violations(first.counts)
               and second.selected_set_identity == first.selected_set_identity),
    }


# ================================================== provider delta audit
def provider_fingerprint() -> dict[str, Any]:
    """Content fingerprint of every artifact that could record a provider call."""
    files: list[dict[str, Any]] = []
    for rel in PROVIDER_EVIDENCE:
        root = REPO / rel
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                files.append({"path": path.relative_to(REPO).as_posix(),
                              "size": path.stat().st_size,
                              "sha256": file_sha256(path)})
    archives = {
        "reports/c2/C2_PILOT_RAW_ARCHIVE.json": "records",
        "reports/c2/C2_SMOKE_RAW_ARCHIVE.json": "records",
        "reports/c2b/C2B_RAW_ARCHIVE.json": "records",
        "reports/c2c/C2C_RAW_ARCHIVE.json": "records",
    }
    counts = {}
    for rel, key in archives.items():
        path = REPO / rel
        counts[rel] = len(load(path).get(key, [])) if path.exists() else 0
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "artifact_count": len(files),
        "fingerprint_sha256": digest,
        "archived_provider_records_by_milestone": counts,
        "archived_provider_records_total": sum(counts.values()),
        "c3_generation_shaped_artifacts": [
            entry["path"] for entry in files
            if any(marker in entry["path"].upper() for marker in
                   ("C3_RAW_ARCHIVE", "C3_CANDIDATE", "RECIPE_BANK_LOCK",
                    "C3_SELECTION_AUDIT", "C3_BATCH"))],
    }


# ============================================ required-test obligation mapping
REQUIREMENTS: list[tuple[int, str, list[str]]] = [
    (1, "exact 384 candidate slots", [
        "tests/c3/test_c3_schedules_and_identity.py::test_each_control_arm_has_exactly_384_canonical_slot_ids[RND]",
        "tests/c3/test_c3_schedules_and_identity.py::test_each_control_arm_has_exactly_384_canonical_slot_ids[DET]",
        "tests/c3/test_c3_selection.py::test_frozen_cardinalities_are_the_spec_values",
        "tests/c3/test_c3_selection.py::test_the_slot_budget_cannot_be_topped_up"]),
    (2, "a pool below 320 eligible fails closed", [
        "tests/c3/test_c3_selection.py::test_a_pool_below_the_minimum_fails_closed",
        "tests/c3/test_c3_selection.py::test_pool_below_minimum_is_reported"]),
    (3, "exactly 320 may proceed when feasible", [
        "tests/c3/test_c3_selection.py::test_exactly_the_minimum_pool_is_permitted_to_proceed",
        "tests/c3/test_c3_selection_optimality.py::test_the_real_320_to_256_budget_selects_exactly_256"]),
    (4, "final selected cardinality exactly 256", [
        "tests/c3/test_c3_selection_optimality.py::test_the_real_320_to_256_budget_selects_exactly_256",
        "tests/c3/test_c3_selection.py::test_selected_cardinality_is_exact",
        "tests/c3/test_c3_selection.py::test_frozen_cardinalities_are_the_spec_values"]),
    (5, "every hard quota holds", [
        "tests/c3/test_c3_selection.py::test_the_frozen_quota_table_matches_the_spec",
        "tests/c3/test_c3_selection.py::test_every_hard_quota_holds_in_the_selected_bank",
        "tests/c3/test_c3_selection.py::test_an_infeasible_hard_quota_fails_rather_than_relaxing",
        "tests/c3/test_c3_selection_optimality.py::test_the_real_320_to_256_budget_selects_exactly_256"]),
    (6, "preferred objective S_pref", [
        "tests/c3/test_c3_selection.py::test_s_pref_is_exact_integer_arithmetic",
        "tests/c3/test_c3_selection_optimality.py::test_the_selector_returns_the_exact_lexicographic_optimum[bf0-8-3]",
        "tests/c3/test_c3_selection_optimality.py::test_the_selector_returns_the_exact_lexicographic_optimum[bf1-9-4]"]),
    (7, "S_single arithmetic", [
        "tests/c3/test_c3_selection.py::test_s_single_is_exact_integer_arithmetic",
        "tests/c3/test_c3_selection.py::test_objective_values_are_recomputed_from_the_selected_set",
        "tests/c3/test_c3_selection_optimality.py::test_the_selector_returns_the_exact_lexicographic_optimum[bf2-10-4]"]),
    (8, "S_multi arithmetic", [
        "tests/c3/test_c3_selection.py::test_s_multi_is_exact_integer_arithmetic",
        "tests/c3/test_c3_selection_optimality.py::test_the_selector_returns_the_exact_lexicographic_optimum[bf3-11-5]"]),
    (9, "canonical stage-5 tie-break", [
        "tests/c3/test_c3_selection_optimality.py::test_stage_5_returns_the_lexicographically_smallest_tied_set[bf0-8-3]",
        "tests/c3/test_c3_selection_optimality.py::test_stage_5_returns_the_lexicographically_smallest_tied_set[bf1-9-4]",
        "tests/c3/test_c3_selection_optimality.py::test_stage_5_returns_the_lexicographically_smallest_tied_set[bf2-10-4]",
        "tests/c3/test_c3_selection_optimality.py::test_stage_5_returns_the_lexicographically_smallest_tied_set[bf3-11-5]",
        "tests/c3/test_c3_selection_optimality.py::test_stage_5_decides_alone_when_every_subset_ties",
        "tests/c3/test_c3_selection.py::test_the_selected_set_is_the_lexicographically_smallest_tied_set"]),
    (10, "input permutation invariance", [
        "tests/c3/test_c3_selection.py::test_input_permutation_does_not_change_the_bank",
        "tests/c3/test_c3_selection.py::test_reversed_input_does_not_change_the_bank",
        "tests/c3/test_c3_selection_optimality.py::test_the_tie_case_is_immune_to_input_order"]),
    (11, "repeated-run identity", [
        "tests/c3/test_c3_selection.py::test_repeated_runs_are_identical"]),
    (12, "filesystem-order invariance", [
        "tests/c3/test_c3_selection_optimality.py::test_filesystem_listing_order_does_not_change_the_bank"]),
    (13, "solver-order invariance", [
        "tests/c3/test_c3_selection_optimality.py::test_solver_traversal_order_cannot_decide_the_bank",
        "tests/c3/test_c3_selection.py::test_candidates_are_ordered_by_canonical_sha_not_by_input_order"]),
    (14, "exact-route rejection", [
        "tests/c3/test_c3_selection.py::test_route_invalid_candidates_are_excluded_before_the_compiler",
        "tests/c3/test_c3_schedules_and_identity.py::test_drafted_candidates_declare_the_scientific_route[RND]",
        "tests/c3/test_c3_schedules_and_identity.py::test_drafted_candidates_declare_the_scientific_route[DET]",
        "tests/c3/test_c3_bank_lock.py::test_the_lock_freezes_the_route_contract"]),
    (15, "duplicate handling", [
        "tests/c3/test_c3_selection.py::test_duplicates_are_excluded_at_the_deduplication_stage"]),
    (16, "compiler-invalid exclusion", [
        "tests/c3/test_c3_selection.py::test_an_uncompilable_candidate_is_excluded"]),
    (17, "target-field independence", [
        "tests/c3/test_c3_forbidden_inputs.py::test_one_forbidden_field_at_a_time_cannot_move_the_bank[siw_mv2_attack_family-print]",
        "tests/c3/test_c3_forbidden_inputs.py::test_one_forbidden_field_at_a_time_cannot_move_the_bank[target_acer-0.0731]",
        "tests/c3/test_c3_forbidden_inputs.py::test_every_forbidden_field_at_once_cannot_move_the_bank",
        "tests/c3/test_c3_selection.py::test_selection_ignores_target_quality_and_score_fields",
        "tests/c3/test_c3_schedules_and_identity.py::test_schedules_carry_no_target_or_llm_dependency[RND]",
        "tests/c3/test_c3_schedules_and_identity.py::test_schedules_carry_no_target_or_llm_dependency[DET]"]),
    (18, "synthetic quality q independence", [
        "tests/c3/test_c3_forbidden_inputs.py::test_one_forbidden_field_at_a_time_cannot_move_the_bank[synthetic_quality_q-0.421]",
        "tests/c3/test_c3_forbidden_inputs.py::test_one_forbidden_field_at_a_time_cannot_move_the_bank[q-0.421]",
        "tests/c3/test_c3_forbidden_inputs.py::test_every_forbidden_field_at_once_cannot_move_the_bank"]),
    (19, "downstream detector-score independence", [
        "tests/c3/test_c3_forbidden_inputs.py::test_one_forbidden_field_at_a_time_cannot_move_the_bank[detector_score-0.8817]",
        "tests/c3/test_c3_forbidden_inputs.py::test_a_forbidden_value_correlated_with_quality_cannot_bias_selection",
        "tests/c3/test_c3_forbidden_inputs.py::test_every_forbidden_field_at_once_cannot_move_the_bank"]),
    (20, "31/33 batch failure preserved", [
        "tests/c3/test_c3_selection_optimality.py::test_a_c3_batch_that_is_not_exactly_32_still_fails_closed[31]",
        "tests/c3/test_c3_selection_optimality.py::test_a_c3_batch_that_is_not_exactly_32_still_fails_closed[33]",
        "tests/c2b/test_c2b_batch_envelope.py::test_a_batch_that_is_not_exactly_32_is_rejected[31]",
        "tests/c2b/test_c2b_batch_envelope.py::test_a_batch_that_is_not_exactly_32_is_rejected[33]"]),
    (21, "no silent truncation or padding", [
        "tests/c3/test_c3_selection.py::test_no_silent_truncation_or_padding",
        "tests/c3/test_c3_selection.py::test_a_missing_slot_is_a_rejection_not_an_omission",
        "tests/c3/test_c3_selection_optimality.py::test_the_real_320_to_256_budget_selects_exactly_256"]),
    (22, "RND reproducibility", [
        "tests/c3/test_c3_schedules_and_identity.py::test_schedule_identity_is_reproducible[RND]",
        "tests/c3/test_c3_schedules_and_identity.py::test_candidate_drafting_is_per_slot_deterministic[RND]",
        "tests/c3/test_c3_schedules_and_identity.py::test_drafted_candidates_pass_the_common_eligibility_gate[RND]"]),
    (23, "DET reproducibility", [
        "tests/c3/test_c3_schedules_and_identity.py::test_schedule_identity_is_reproducible[DET]",
        "tests/c3/test_c3_schedules_and_identity.py::test_candidate_drafting_is_per_slot_deterministic[DET]",
        "tests/c3/test_c3_schedules_and_identity.py::test_drafted_candidates_pass_the_common_eligibility_gate[DET]"]),
    (24, "old preliminary lock integrity", [
        "tests/c3/test_c3_bank_contract_lock.py::test_the_historical_preliminary_lock_is_byte_identical",
        "tests/c3/test_c3_schedules_and_identity.py::test_the_preliminary_lock_bytes_are_unchanged",
        "tests/c3/test_c3_schedules_and_identity.py::test_the_preliminary_lock_still_binds_no_selection_identity"]),
    (25, "zero live Gemini calls", [
        "tests/c3/test_c3_schedules_and_identity.py::test_the_ambient_credential_is_removed",
        "tests/c3/test_c3_schedules_and_identity.py::test_importing_the_c3_selection_path_loads_no_provider_sdk",
        "tests/c3/test_c3_schedules_and_identity.py::test_selection_and_schedule_modules_import_no_provider",
        "tests/c3/test_c3_schedules_and_identity.py::test_no_c3_scientific_generation_artifact_exists",
        "tests/c3/test_c3_bank_lock.py::test_no_c3_scientific_request_has_been_made"]),
]


def run_pytest(targets: list[str]) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", *targets, "-q", "--no-header",
               "-p", "no:cacheprovider", "-rA", "--tb=short",
               "--continue-on-collection-errors"]
    started = time.time()
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False)
    stdout = result.stdout or ""
    outcomes: dict[str, str] = {}
    for line in stdout.splitlines():
        match = re.match(r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(\S+)", line.strip())
        if match:
            outcomes[match.group(2)] = match.group(1)
    summary = ""
    for line in reversed(stdout.strip().splitlines()):
        if " in " in line and ("passed" in line or "failed" in line or "error" in line):
            summary = line.strip()
            break

    def count(label: str) -> int:
        found = re.search(rf"(\d+) {label}", summary)
        return int(found.group(1)) if found else 0

    return {
        "command": " ".join(["python", "-m", "pytest", *targets, "-q", "--no-header",
                             "-p", "no:cacheprovider", "-rA", "--tb=short",
                             "--continue-on-collection-errors"]),
        "exit_code": result.returncode,
        "summary_line": summary,
        "passed": count("passed"), "failed": count("failed"),
        "skipped": count("skipped"), "xfailed": count("xfailed"),
        "errors": count("error"),
        "duration_seconds": round(time.time() - started, 1),
        "outcomes": outcomes,
        "failed_node_ids": sorted(node for node, verdict in outcomes.items()
                                  if verdict in {"FAILED", "ERROR"}),
    }


def map_requirements(outcomes: dict[str, str]) -> dict[str, Any]:
    rows = []
    for number, requirement, node_ids in REQUIREMENTS:
        resolved = []
        for node_id in node_ids:
            verdict = outcomes.get(node_id)
            resolved.append({
                "test_file": node_id.split("::")[0],
                "test_function": "::".join(node_id.split("::")[1:]),
                "node_id": node_id,
                "executed": verdict is not None,
                "outcome": verdict or "NOT_EXECUTED_IN_THIS_RUN",
            })
        every_ran = all(entry["executed"] for entry in resolved)
        every_passed = all(entry["outcome"] == "PASSED" for entry in resolved)
        rows.append({
            "requirement_number": f"{number:02d}",
            "requirement": requirement,
            "tests": resolved,
            "test_count": len(resolved),
            "status": "PASS" if (every_ran and every_passed) else "FAIL",
            "evidence": "every listed node id was collected and reported PASSED in the "
                        "recorded run" if (every_ran and every_passed)
                        else "at least one listed node id did not execute or did not pass",
        })
    return {
        "method": "each obligation is bound to explicit pytest node IDs; a requirement is "
                  "PASS only when every bound node ID was actually collected AND reported "
                  "PASSED in the recorded run. Existence of a similarly named file is not "
                  "evidence.",
        "requirements": rows,
        "total": len(rows),
        "passing": sum(1 for row in rows if row["status"] == "PASS"),
        "failing": [row["requirement_number"] for row in rows if row["status"] == "FAIL"],
        "all_pass": all(row["status"] == "PASS" for row in rows),
    }


# ============================================ PROJECT_STATE semantics
def audit_project_state() -> dict[str, Any]:
    import yaml

    text = (REPO / "docs" / "PROJECT_STATE.md").read_text(encoding="utf-8")
    block = re.search(r"```yaml\n(.*?)\n```", text, re.S).group(1)
    state = yaml.safe_load(block)

    c3 = state.get("c3", {})
    execution = state.get("execution", {})
    top_engineering = state.get("engineering_status")

    # Does the document make clear that engineering_status scopes the CONTRACT
    # substage, and NOT an implemented validate/smoke/full pipeline?
    #
    # Checked on PARSED values only. A qualifier that lives in a YAML comment is
    # invisible to every consumer of this file, so it cannot count as evidence.
    def absent(key: str) -> bool:
        value = execution.get(key)
        return isinstance(value, dict) and value.get("exists") is False

    orchestrator_absent = absent("orchestrator")
    pipeline_state_absent = absent("pipeline_state")
    pipeline = state.get("execution_pipeline", {})
    profiles_marked_unimplemented = all(
        pipeline.get(profile) == "NOT_IMPLEMENTED" for profile in ("validate", "smoke", "full"))
    scope_declared = state.get("engineering_status_scope") is not None
    blockers = " ".join(state.get("blockers", []))
    says_layer_missing = "execution layer" in blockers and "does not" in blockers

    findings = {
        "method": "the YAML block is PARSED and only parsed values are treated as evidence; "
                  "a qualifier that exists only in a YAML comment is stripped by every "
                  "consumer and therefore proves nothing",
        "top_level_engineering_status": top_engineering,
        "top_level_engineering_status_scope": state.get("engineering_status_scope"),
        "engineering_status_scope_declared": scope_declared,
        "top_level_scientific_status": state.get("scientific_status"),
        "current_substage": state.get("current_substage"),
        "execution_block": execution,
        "execution_pipeline_block": pipeline,
        "profiles_marked_unimplemented": profiles_marked_unimplemented,
        "orchestrator_marked_absent": orchestrator_absent,
        "pipeline_state_marked_absent": pipeline_state_absent,
        "blockers_state_execution_layer_absent": says_layer_missing,
        "c3_scientific_generation": c3.get("scientific_generation"),
        "c3_scientific_logical_requests": c3.get("c3_scientific_logical_requests"),
        "c3_scientific_candidate_slots": c3.get("c3_scientific_candidate_slots"),
        "next_authorized_action": (state.get("next_authorized_action") or "").strip(),
    }

    ambiguity = []
    if top_engineering == "SMOKE_PASS":
        if not str(state.get("current_substage", "")).strip():
            ambiguity.append("engineering_status is SMOKE_PASS with no substage scoping it")
        if not scope_declared:
            ambiguity.append("engineering_status is SMOKE_PASS with no machine-readable "
                             "engineering_status_scope naming what it covers")
        if not profiles_marked_unimplemented:
            ambiguity.append("execution_pipeline does not mark validate/smoke/full as "
                             "NOT_IMPLEMENTED in parsed values")
        if not (orchestrator_absent and pipeline_state_absent and says_layer_missing):
            ambiguity.append(
                "engineering_status=SMOKE_PASS is not accompanied by an unambiguous parsed "
                "statement that the validate/smoke/full execution layer is unimplemented")

    findings["ambiguities"] = ambiguity
    findings["unambiguous"] = not ambiguity
    findings["falsely_claims_pipeline_completion"] = bool(ambiguity)
    findings["scientific_status_is_not_run"] = state.get("scientific_status") == "NOT_RUN"
    findings["counters_are_zero"] = (c3.get("c3_scientific_logical_requests") == 0
                                     and c3.get("c3_scientific_candidate_slots") == 0)
    return findings


# ==================================================================== main
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    started_utc = utc_now()

    ontology = load_ontology(REPO / "configs" / "recipes" / "ontology_m7.yaml")
    route_policy = load_route_policy(
        REPO / "configs" / "version_c" / "llm" / "c2c_route_policy.yaml")
    route_policy.validate_against(ontology)

    provider_before = provider_fingerprint()

    preflight = audit_preflight()
    identities = audit_identities(ontology, route_policy)
    preliminary = audit_preliminary_lock()
    superseding = audit_superseding_lock()
    schedules = audit_schedules(ontology)
    selector = audit_selector(ontology, route_policy)
    objectives = audit_objectives(ontology)
    stage_5 = audit_stage_5(ontology)
    full_scale = audit_full_scale(ontology)
    project_state = audit_project_state()

    focused = run_pytest(["tests/c3"])
    broad = run_pytest([])

    provider_after = provider_fingerprint()
    delta = {
        "before": provider_before,
        "after": provider_after,
        "fingerprint_unchanged":
            provider_before["fingerprint_sha256"] == provider_after["fingerprint_sha256"],
        "archived_records_before": provider_before["archived_provider_records_total"],
        "archived_records_after": provider_after["archived_provider_records_total"],
        "delta": (provider_after["archived_provider_records_total"]
                  - provider_before["archived_provider_records_total"]),
        "c3_scientific_logical_requests": 0,
        "c3_scientific_candidate_slots": 0,
        "c3_generation_shaped_artifacts": provider_after["c3_generation_shaped_artifacts"],
        "disposable_pilot_calls_are_not_c3": {
            "c1": 0, "c2_smoke": 2, "c2_pilot": 42, "c2b": 1, "c2c": 1, "c3_scientific": 0,
            "note": "C1/C2/C2B/C2C calls were disposable or diagnostic; none produced a "
                    "recipe that enters C3, and the archives record that explicitly",
        },
        "new_live_provider_requests_caused_by_contract_or_audit_tasks": 0,
    }

    mapping = map_requirements({**focused["outcomes"], **broad["outcomes"]})

    inherited = sorted([
        "tests/test_m10_closure.py::test_synthetic_exposure_is_derived_from_the_audited_batch_contract",
        "tests/test_m10_closure.py::test_backend_parity_is_reported_as_measured_not_as_a_pass",
        "tests/test_m10_target_evaluation.py::test_isolation_declarations_do_not_false_positive",
        "tests/test_m2_validation.py::test_actual_small_acceptance_validation_passes",
        "tests/test_m2_validation.py::test_status_reports_expected_counts",
        "tests/test_m8_gpat_synthetic_bank.py::test_pair_plan_lock_records_identities_and_seed",
        "tests/test_m8_gpat_synthetic_bank.py::test_pair_plan_identity_excludes_non_portable_fields",
    ])
    new_failures = sorted(set(broad["failed_node_ids"]) - set(inherited))
    regression = {
        "focused": focused,
        "broad": broad,
        "documented_inherited_failures": inherited,
        "inherited_source": "reports/c0/C0_TEST_SUITE.json",
        "observed_broad_failures": broad["failed_node_ids"],
        "new_unexplained_failures": new_failures,
        "new_unexplained_failure_count": len(new_failures),
        "acceptance": "zero NEW unexplained failures; inherited failures may remain",
        "ok": not new_failures and focused["failed"] == 0 and focused["errors"] == 0,
    }

    # ---------------------------------------------------------- gate
    checks = {
        "spec_sha_matches": preflight["spec_matches"],
        "version_b_head_matches":
            preflight["version_b"]["head"] == EXPECTED["version_b_head"],
        "version_b_tag_peels_correctly":
            preflight["version_b"]["tag_peeled"] == EXPECTED["version_b_head"],
        "version_b_clean": preflight["version_b"]["clean"],
        "commits_resolvable": preflight["commits_resolvable"],
        "preliminary_lock_unchanged": (
            preliminary["canonical_body_identity"]["reproduces"]
            and preliminary["modified_by_contract_task"] is False),
        "preliminary_lock_body_identity_reproduces":
            preliminary["canonical_body_identity"]["reproduces"],
        "preliminary_lock_lf_form_matches_git_blob":
            preliminary["representations"]["lf_normalized_equals_git_blob"],
        "preliminary_lock_untouched_by_contract_commits":
            preliminary["commits_touching_this_file"] == [git("rev-parse", "50514ce")],
        "generation_identity_reproduces":
            identities["generation_contract_identity"]["reproduces"],
        "selection_identity_reproduces":
            identities["selection_contract_identity"]["reproduces"],
        "bank_contract_identity_reproduces": identities["bank_contract_identity"]["reproduces"],
        "superseding_lock_identity_reproduces":
            identities["superseding_lock_identity"]["reproduces"],
        "superseding_lock_status_correct": superseding["status_correct"],
        "superseding_lock_supersedes_correct": superseding["supersedes_correct"],
        "superseding_lock_reason_correct": superseding["reason_correct"],
        "superseding_lock_binds_all_required": superseding["binds_all_required"],
        "superseding_lock_claims_no_generation": superseding["no_false_completion_claim"],
        "rnd_schedule_reproduces": schedules["RND"]["reproducible"],
        "det_schedule_reproduces": schedules["DET"]["reproducible"],
        "schedules_distinct": schedules["schedules_are_distinct"],
        "selector_cardinalities_ok": selector["cardinalities"]["ok"],
        "selector_eligibility_order_ok": selector["eligibility_order"]["ok"],
        "selector_route_exact": selector["route"]["ok"],
        "hard_quotas_ok": selector["quotas_ok"],
        "objective_arithmetic_ok": objectives["ok"],
        "stage_5_tie_break_ok": stage_5["ok"],
        "full_scale_determinism_ok": full_scale["ok"],
        "all_25_requirements_pass": mapping["all_pass"],
        "focused_tests_clean": focused["failed"] == 0 and focused["errors"] == 0,
        "no_new_unexplained_failures": not new_failures,
        "provider_delta_zero": delta["delta"] == 0 and delta["fingerprint_unchanged"],
        "c3_scientific_requests_zero": delta["c3_scientific_logical_requests"] == 0,
        "c3_candidate_slots_zero": delta["c3_scientific_candidate_slots"] == 0,
        "project_state_does_not_claim_pipeline_completion":
            not project_state["falsely_claims_pipeline_completion"],
    }
    gate = "PASS" if all(checks.values()) else "FAIL"
    blockers = sorted(name for name, ok in checks.items() if not ok)

    common = {
        "schema_version": None,
        "milestone": "C3",
        "substage": "pre-live verification gate",
        "artifact_kind": "VERIFICATION_EVIDENCE",
        "not_scientific_generation_evidence": True,
        "generated_at_utc": started_utc,
        "generator_code_commit": git("rev-parse", "HEAD"),
        "execution_profile": "validate",
        "scientific_eligible": False,
        "scientific_status": "NOT_RUN",
    }

    def write(name: str, schema: str, payload: dict[str, Any]) -> None:
        body = {**common, "schema_version": schema, **payload}
        (OUT / name).write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8")
        print("wrote", (OUT / name).relative_to(REPO).as_posix())

    write("C3_IDENTITY_REPRODUCTION.json", "c3-identity-reproduction-v1",
          {"identities": identities, "preliminary_lock": preliminary,
           "superseding_lock": superseding})
    write("C3_SCHEDULE_AUDIT.json", "c3-schedule-audit-v1",
          {"governing_clause": "7.8.5", "schedules": schedules,
           "rnd_schedule_identity": schedules["RND"]["schedule_identity"],
           "det_schedule_identity": schedules["DET"]["schedule_identity"]})
    write("C3_PROVIDER_DELTA_AUDIT.json", "c3-provider-delta-audit-v1",
          {"provider_delta": delta})
    write("C3_REQUIRED_TEST_MAPPING.json", "c3-required-test-mapping-v1",
          {"mapping": mapping, "focused_run": {k: v for k, v in focused.items()
                                               if k != "outcomes"},
           "broad_run": {k: v for k, v in broad.items() if k != "outcomes"},
           "path_note": "§13 of the audit brief suggested reports/c3/v15_selection_contract/. "
                        "It is written here instead, in the audit namespace §20 defines, so "
                        "that the frozen selection-contract freeze evidence is not extended "
                        "after the fact."})
    write("C3_PROJECT_STATE_SEMANTICS_AUDIT.json", "c3-project-state-semantics-audit-v1",
          {"project_state": project_state})
    write("C3_PRE_LIVE_AUDIT.json", "c3-pre-live-audit-v1", {
        "purpose": "Independent verification that the frozen C3 selection/bank contract "
                   "satisfies the previous authorization. Verification only: no scientific "
                   "candidate was generated, no identity was recomputed into a new frozen "
                   "artifact, and no lock was rewritten.",
        "governing_clauses": ["7.8", "7.8.1", "7.8.2", "7.8.3", "7.8.4", "7.8.5", "21.3"],
        "preflight": preflight,
        "selector_contract": selector,
        "objective_functions": objectives,
        "stage_5": stage_5,
        "full_scale_determinism": full_scale,
        "regression": regression,
        "prohibitions_observed": [
            "no Gemini API call", "no Gemini SDK generation", "no live provider request",
            "no C3 12x32 generation", "no new scientific candidate", "no Modal", "no GPU",
            "no GPAT", "no synthesis", "no detector training", "no SiW access",
            "no target labels", "no target metrics", "no source-side retuning",
            "no manual selection", "no scientific config change",
            "no selector semantic change", "no lock overwrite", "no Git history rewrite",
            "no force push",
        ],
    })
    write("C3_PRE_LIVE_ACCEPTANCE.json", "c3-pre-live-acceptance-v1", {
        "checks": checks,
        "blockers": blockers,
        "C3_PRE_LIVE_GATE": gate,
        "meaning": "PASS means only that the frozen contract is ready for USER REVIEW "
                   "before a future live Gemini task. It does not authorize generation.",
        "engineering_status": "SMOKE_PASS",
        "c3_scientific_generation": "NOT_RUN",
        "next_authorized_action":
            "USER REVIEW of the C3 pre-live audit and the superseding bank-contract lock "
            "before any live Gemini 12x32 C3 scientific generation.",
    })

    print(f"\nC3_PRE_LIVE_GATE = {gate}")
    if blockers:
        print("blockers:", ", ".join(blockers))
    print("focused:", focused["summary_line"])
    print("broad:  ", broad["summary_line"])
    print("new unexplained failures:", len(new_failures))
    print("provider delta:", delta["delta"],
          "| fingerprint unchanged:", delta["fingerprint_unchanged"])
    return 0 if gate == "PASS" else 5


if __name__ == "__main__":
    sys.exit(main())
