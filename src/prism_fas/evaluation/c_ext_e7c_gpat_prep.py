"""PRISM-FAS-C EXT-Q1Q2 -- E7-C: per-fold GPAT preparation and feasibility
planning.

E7-B is CLOSED_VALID and immutable (commit 42e56d0; runtime-verified
identities frozen in `E7B_FINAL_VALID_SUMMARY.json`). This module ONLY
consumes E7-A's fold materializations and E7-B's frozen package identities
-- it never rewrites either, never reprocesses CASIA/MSU/SiW, never opens
target image/crop bytes, never fits GPAT, never renders, never trains,
never calls an LLM.

This module is PLANNING ONLY. Every artifact it writes is a binding/plan
report; no synthetic bank is materialized here (see `E7C_READINESS.json`'s
`E7_READY_FOR_GPU_GPAT_PREPARATION` vs `E7_READY_FOR_TRAINING` distinction).

Governing audit finding (traced from the repository, not assumed): there is
no existing "GPAT support pool" class or function anywhere in the codebase.
The closest reusable primitives are `prism_fas.synthesis.pair_plan`/
`c5_source_pair_plan` (live/spoof source pairing for GPAT fitting) and
`prism_fas.synthesis.gpat_trainer.GPATTrainer` (single-fold, CASIA+MSU-only
fitting -- never yet run per-fold or against SiW-as-source). This module
documents exactly which primitives a later GPU milestone must reuse; it
does not invent a new GPAT algorithm and does not itself invoke any of
them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from prism_fas.evaluation import c_ext_common as cc
from prism_fas.evaluation import c_ext_e7b_data_prep as e7b

SCHEMA_PREFIX = "ext-q1q2-e7c"
E7C_REPORT_DIR = "reports/c_ext_q1q2_v1/e7_three_fold/e7c_gpat_prep"

# --------------------------------------------------------------------------- #
# Frozen E7-B binding (verified real GPU evidence, commit 42e56d0)
# --------------------------------------------------------------------------- #

E7B_FINAL_EVIDENCE_PATH = ("reports/c_ext_q1q2_v1/e7_three_fold/e7b_data_prep/"
                          "gpu_evidence/final_e7b_valid_9311226/E7B_FINAL_VALID_SUMMARY.json")

FROZEN_E7B = {
    "siw_source_package_identity": "0f7811b0960d0dd2be7c732aef4107af9c3476eb9b6b9932b4fe32c7a126bb4f",
    "msu_target_package_identity": "8f77c81915a6d42dbd792a723e249483eea5d43d7ff46c45a6e0a1f8629cb6ad",
    "casia_target_package_identity": "5996873998daa290498728b4dbd52ab3f91d2ad357e244b7516485f4ec72b457",
    "siw_population_identity": "d05dafb814a98baebd7a5cd004ca0eb92ba798c13a8a6c5b6c90b1919e365c79",
    "siw_split_identity": "b492a5d4d86537016012d5357bc5c4410f77e36409082831ef6703168ee096a1",
    "preprocessing_config_hash": "48a120caa6041b3a03b4008642030665f084b5d722a62ca2c01a2a5aa5e0c959",
    "detector_model_sha256": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
}

# --------------------------------------------------------------------------- #
# Governing fold composition -- sole authority is E7-A materialization; these
# constants are the EXPECTED values used only to fail closed on drift, never
# to override what E7-A's own materialization actually says.
# --------------------------------------------------------------------------- #

FOLD_IDS = ("EXT-F1", "EXT-F2", "EXT-F3")
FOLD_SOURCE_DOMAINS = {
    "EXT-F1": ("CASIA-FASD", "MSU-MFSD"),
    "EXT-F2": ("CASIA-FASD", "SiW-Mv2"),
    "EXT-F3": ("MSU-MFSD", "SiW-Mv2"),
}
FOLD_TARGET_DOMAIN = {"EXT-F1": "SiW-Mv2", "EXT-F2": "MSU-MFSD", "EXT-F3": "CASIA-FASD"}

CONDITIONS = ("G-REALONLY", "G-RND", "G-DET", "G-LLM", "G-LLM-SHUFFLE-A")
SYNTHETIC_CONDITIONS = ("G-RND", "G-DET", "G-LLM", "G-LLM-SHUFFLE-A")

READY_FOR_GPAT_PREP = "READY_FOR_GPAT_PREP"
BLOCKED_TRUE_FROZEN_MATCHED_BANK_INFEASIBILITY = "BLOCKED_TRUE_FROZEN_MATCHED_BANK_INFEASIBILITY"
PENDING_FEASIBILITY_PREFLIGHT = "PENDING_FEASIBILITY_PREFLIGHT"
NOT_APPLICABLE = "NOT_APPLICABLE"

# --------------------------------------------------------------------------- #
# Frozen recipe banks (RND/DET/LLM: assets/recipe_banks/c3/<arm>/; the
# LLM-SHUFFLE-A derived bank: reports/.../e6_llm_shuffle/). Never
# regenerated, never re-selected -- read-only binding to what already exists.
# --------------------------------------------------------------------------- #

RECIPE_BANK_ROOTS = {
    "RND": "assets/recipe_banks/c3/rnd",
    "DET": "assets/recipe_banks/c3/det",
    "LLM": "assets/recipe_banks/c3/llm",
}
LLM_SHUFFLE_A_RECIPES_PATH = "reports/c_ext_q1q2_v1/e6_llm_shuffle/LLM_SHUFFLE_A_RECIPES.jsonl"
LLM_SHUFFLE_A_AUDIT_PATH = "reports/c_ext_q1q2_v1/e6_llm_shuffle/E6_LLM_SHUFFLE_A.json"
EXPECTED_RECIPE_BANK_ROW_COUNT = 256

#: Frozen artifact proving the LLM arm's semantic selected-set identity and
#: its v2 raw/content identity refer to the SAME 256 recipe payloads.
#: Audited directly from repository evidence -- never assumed.
LLM_EQUIVALENCE_EVIDENCE_PATH = "reports/c_ext_q1q2_v1/e6_paired_current_runtime_v2/E6_V2_RECIPE_PAIR_LOCK.json"

#: `C3_BANK.json.bank_identity` -- audited exact algorithm from
#: `prism_fas/pipeline/checks.py`: sha256 over compact sorted-key JSON of
#: the 18-field `bank_identity_material` subset of C3_BANK.json (a
#: WHOLE-BANK composite including contract identities, ontology_identity,
#: spec_sha256, generator, AND selected_set_identity as one component) --
#: NOT a pure recipe-content hash.
RECIPE_BANK_IDENTITY_KIND = "RECIPE_BANK_IDENTITY"
RECIPE_BANK_IDENTITY_ALGORITHM = ("sha256(compact sorted-key JSON of the 18 fields named in "
                                  "C3_BANK.json.bank_identity_material, incl. selected_set_identity, "
                                  "contract/ontology identities, spec_sha256) -- reproduced/verified by "
                                  "prism_fas.pipeline.checks; NEVER a pure recipe-content hash")

#: `C3_BANK.json.selected_set_identity` -- audited exact algorithm from
#: reports/.../E6_V2_RECIPE_PAIR_LOCK.json's `earlier_identity`: an
#: order-independent hash over the SORTED per-recipe canonical content
#: hashes. This is the canonical scientific recipe-set identity, propagated
#: into EXT_RECIPE_BINDING.json / E6_TRAINING_PLAN_LOCK.json / c_ext_e6_render.
SELECTED_SET_IDENTITY_KIND = "SELECTED_SET_IDENTITY"
SELECTED_SET_IDENTITY_ALGORITHM = ("sha256 over compact JSON of the SORTED list of per-recipe canonical "
                                   "sha256 hashes (recipes.selection.SelectionResult."
                                   "selected_set_identity); order-independent, content-only, the "
                                   "canonical scientific recipe-set identity")

#: The LLM arm's v2 raw/content identity, per E6_V2_RECIPE_PAIR_LOCK.json's
#: `new_v2_identity`: an ORDERED file/content sha over the raw recipe list
#: in file order (unparsed, no pydantic normalization).
RAW_CONTENT_IDENTITY_KIND = "FILE_SHA256_ORDERED_CONTENT"

#: LLM-SHUFFLE-A's `shuffled_bank_identity` -- audited exact algorithm from
#: `c_ext_llm_shuffle.py` line 215: `sha256_json(result["working_bank"])`, a
#: canonical-JSON content hash over the POST-SHUFFLE working recipe list
#: (field values differ after the group-swap shuffle, so this is neither a
#: pure file hash nor the pre-shuffle selected_set_identity).
SHUFFLE_IDENTITY_KIND = "CANONICAL_JSONL_CONTENT_HASH"
SHUFFLE_IDENTITY_ALGORITHM = ("sha256_json(working_bank) -- canonical-JSON content hash over the "
                              "post-shuffle recipe list (c_ext_llm_shuffle.run_shuffle's own "
                              "shuffled_bank_identity)")

# --------------------------------------------------------------------------- #
# EXT-F1 Shuffle-A: TRUE, frozen, historically-observed matched-bank
# infeasibility (E6-v2 closure). Never re-derived; never "fixed"; applies to
# all five nominal training seeds. F2/F3 are independently PENDING.
# --------------------------------------------------------------------------- #

E6V2_CLOSURE_PATH = "reports/c_ext_q1q2_v1/e6_paired_current_runtime_v2/E6_V2_FINAL_CLOSURE.json"
FROZEN_E6V2_CLOSURE = {
    "E6_V2_STATUS": "CLOSED_TRUE_FROZEN_MATCHED_BANK_INFEASIBILITY",
    "SHUFFLE_GPAT_MAX_FILLABLE": 512,
    "SHUFFLE_PHYSICS_MAX_FILLABLE": 479,
    "SHUFFLE_PHYSICS_REQUIRED": 512,
    "SHUFFLE_PHYSICS_CASIA_AVAILABLE": 231,
    "SHUFFLE_PHYSICS_CASIA_REQUIRED": 264,
    "SHUFFLE_PHYSICS_CASIA_DEFICIT": 33,
    "closure_identity": "2a7b95496bf25cbd5d17265188b9003ff17cf48abc7a500dde598738ec1a2a03",
}

# --------------------------------------------------------------------------- #
# Target firewall -- paths this module (and every module it documents for
# later reuse) must never open for image/label bytes during GPAT
# preparation. Per-fold, the held-out target's OWN E7-B/E7-A-reused package
# root is added on top of these universal roots.
# --------------------------------------------------------------------------- #

UNIVERSAL_FIREWALL_ROOTS = (
    "data/evaluation_only",
    "reports/flow2_counterfactual_assumed_pass",
    "runs/flow2_counterfactual_assumed_pass",
    "reports/full/exploratory_target_v3",
    "reports/c_ext_q1q2_v1/e5_realonly/target_scoring",
)

#: Per-fold, the physical root holding the HELD-OUT target's own crop bytes
#: -- reused verbatim from e7b/e7a, never redeclared.
FOLD_TARGET_PACKAGE_ROOT = {
    "EXT-F1": e7b.SIW_TARGET_EVAL_PACKAGE_ROOT,  # data/processed/prism_target_eval_v2
    "EXT-F2": e7b.E7B_MSU_TARGET_PACKAGE_ROOT,
    "EXT-F3": e7b.E7B_CASIA_TARGET_PACKAGE_ROOT,
}
#: Per-fold source domains map to which E7-B/M3B roots ARE legitimately open.
FOLD_SOURCE_ROOTS = {
    "EXT-F1": (e7b.CASIA_MSU_PACKAGE_ROOT,),
    "EXT-F2": (e7b.CASIA_MSU_PACKAGE_ROOT, e7b.E7B_SIW_SOURCE_PACKAGE_ROOT),
    "EXT-F3": (e7b.CASIA_MSU_PACKAGE_ROOT, e7b.E7B_SIW_SOURCE_PACKAGE_ROOT),
}


class E7CError(RuntimeError):
    pass


class E7CTargetFirewallViolation(E7CError):
    pass


# --------------------------------------------------------------------------- #
# TASK A -- protocol lock (governing fold/condition/firewall constants,
# never opens any data)
# --------------------------------------------------------------------------- #

def build_protocol_lock(repo: Path) -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_PREFIX}-protocol-lock-v1",
        "folds": {fold_id: {"source_domains": list(FOLD_SOURCE_DOMAINS[fold_id]),
                            "heldout_target_domain": FOLD_TARGET_DOMAIN[fold_id]}
                 for fold_id in FOLD_IDS},
        "conditions": list(CONDITIONS),
        "synthetic_conditions": list(SYNTHETIC_CONDITIONS),
        "realonly_requires_no_synthetic_bank": True,
        "target_isolation_rule": "the held-out target domain's image/crop/feature bytes may never enter "
                                 "GPAT support pool, GPAT fitting data, source-live support samples, "
                                 "quality calibration, candidate generation, synthetic candidate "
                                 "acceptance/calibration, source-dev calibration, or recipe-to-image "
                                 "routing inputs for that fold; the target package identity may appear "
                                 "in metadata only",
        "target_labels_forbidden": True,
        "sole_fold_authority": "E7-A materialization (reports/c_ext_q1q2_v1/e7_three_fold/e7a/"
                               "materialization_v1/EXT-F{1,2,3}/FOLD_MATERIALIZATION.json)",
        "f1_shuffle_frozen_block": FROZEN_E6V2_CLOSURE["E6_V2_STATUS"],
        "f2_f3_shuffle_independent": True,
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False,
        "training_performed": False, "gpat_fitting_performed": False,
    }


def write_protocol_lock(repo: Path) -> dict[str, Any]:
    return _write(repo, "E7C_PROTOCOL_LOCK.json", build_protocol_lock(repo))


# --------------------------------------------------------------------------- #
# TASK B -- E7-B binding (frozen identities; local bytes usually absent on a
# laptop -- that is truthfully reported, never fabricated)
# --------------------------------------------------------------------------- #

def build_e7b_binding(repo: Path) -> dict[str, Any]:
    evidence_path = repo / E7B_FINAL_EVIDENCE_PATH
    evidence_present = evidence_path.is_file()
    evidence: dict[str, Any] = cc.read_json(evidence_path) if evidence_present else {}

    observed = {
        "siw_source_package_identity": evidence.get("siw_source", {}).get("package_identity"),
        "msu_target_package_identity": evidence.get("msu_target", {}).get("package_identity"),
        "casia_target_package_identity": evidence.get("casia_target", {}).get("package_identity"),
        "siw_population_identity": evidence.get("siw_source", {}).get("population_identity"),
        "siw_split_identity": evidence.get("siw_source", {}).get("split_identity"),
        "preprocessing_config_hash": evidence.get("preprocessing_config_hash"),
        "detector_model_sha256": evidence.get("detector_model_sha256"),
    }
    mismatches = ([k for k in FROZEN_E7B if evidence_present and observed[k] != FROZEN_E7B[k]]
                 if evidence_present else [])
    status = ("MISMATCH" if mismatches else "MATCH") if evidence_present else "LOCAL_BYTES_MISSING"

    local_package_present = {
        "siw_source": (repo / e7b.E7B_SIW_SOURCE_PACKAGE_ROOT / "SIW_SOURCE_PACKAGE.json").is_file(),
        "msu_target": (repo / e7b.E7B_MSU_TARGET_PACKAGE_ROOT / "TARGET_PACKAGE.json").is_file(),
        "casia_target": (repo / e7b.E7B_CASIA_TARGET_PACKAGE_ROOT / "TARGET_PACKAGE.json").is_file(),
    }

    return {
        "schema_version": f"{SCHEMA_PREFIX}-e7b-binding-v1",
        "e7b_status": "CLOSED_VALID",
        "evidence_path": E7B_FINAL_EVIDENCE_PATH, "evidence_present": evidence_present,
        "frozen": FROZEN_E7B, "observed": observed, "mismatches": mismatches,
        "status": status, "match": status == "MATCH",
        "local_package_bytes_present": local_package_present,
        "local_data_state": "PLAN_VALID" if status == "MATCH" else
                            ("GPU_REQUIRED" if status == "LOCAL_BYTES_MISSING" else "MISMATCH_FAIL_CLOSED"),
        "target_access": False, "llm_api_calls": 0,
    }


def write_e7b_binding(repo: Path) -> dict[str, Any]:
    binding = build_e7b_binding(repo)
    if binding["status"] == "MISMATCH":
        raise E7CError(f"E7-B final binding MISMATCH -- FAIL CLOSED: {binding['mismatches']!r} disagree "
                       f"with the frozen pins recorded in commit 42e56d0's evidence")
    return _write(repo, "E7C_E7B_BINDING.json", binding)


# --------------------------------------------------------------------------- #
# TASK C -- E7-A fold binding (reuses e7b's own frozen-hash verification and
# fold-materialization loader verbatim -- never reimplemented)
# --------------------------------------------------------------------------- #

def build_e7a_fold_binding(repo: Path, fold_id: str) -> dict[str, Any]:
    if fold_id not in FOLD_IDS:
        raise E7CError(f"unknown fold_id {fold_id!r}")
    hashes = e7b.verify_e7a_frozen_hashes(repo).get(fold_id, {})
    materialization = e7b.load_e7a_fold_materialization(repo, fold_id)
    present = materialization is not None
    source_domains_match = present and tuple(materialization.get("source_domains", ())) == \
        FOLD_SOURCE_DOMAINS[fold_id]
    target_domain_match = present and materialization.get("target_domain") == FOLD_TARGET_DOMAIN[fold_id]
    siw_population_match = (not present or materialization.get("siw_population_identity") in
                            (None, FROZEN_E7B["siw_population_identity"]))
    siw_split_match = (not present or materialization.get("siw_split_identity") in
                       (None, FROZEN_E7B["siw_split_identity"]))
    return {
        "schema_version": f"{SCHEMA_PREFIX}-e7a-fold-binding-v1", "fold_id": fold_id,
        "hash_present": hashes.get("present", False), "hash_match": hashes.get("match", False),
        "materialization_present": present,
        "source_domains_match": source_domains_match, "target_domain_match": target_domain_match,
        "siw_population_identity_match": siw_population_match,
        "siw_split_identity_match": siw_split_match,
        "match": bool(hashes.get("match")) and source_domains_match and target_domain_match and
                siw_population_match and siw_split_match,
        "target_access": False, "llm_api_calls": 0,
    }


def write_e7a_fold_binding(repo: Path) -> dict[str, Any]:
    bindings = {fold_id: build_e7a_fold_binding(repo, fold_id) for fold_id in FOLD_IDS}
    for fold_id, binding in bindings.items():
        if binding["materialization_present"] and not binding["match"]:
            raise E7CError(f"{fold_id}: E7-A fold binding does not match -- FAIL CLOSED: {binding!r}")
    return _write(repo, "E7C_E7A_FOLD_BINDING.json", {
        "schema_version": f"{SCHEMA_PREFIX}-e7a-fold-binding-all-v1", "folds": bindings,
        "target_access": False, "llm_api_calls": 0})


# --------------------------------------------------------------------------- #
# TASK D -- target firewall (structural: raises on any forbidden path,
# never merely reports)
# --------------------------------------------------------------------------- #

def forbidden_roots_for_fold(fold_id: str) -> tuple[str, ...]:
    return UNIVERSAL_FIREWALL_ROOTS + (FOLD_TARGET_PACKAGE_ROOT[fold_id],)


def assert_not_target_path(fold_id: str, candidate_path: str) -> None:
    """FAIL CLOSED the instant any candidate path (a manifest reference, a
    planned support-pool entry, anything) falls under a forbidden root for
    this fold's held-out target. Never used to merely warn."""
    normalized = candidate_path.replace("\\", "/").lstrip("/")
    for forbidden in forbidden_roots_for_fold(fold_id):
        if normalized == forbidden or normalized.startswith(forbidden.rstrip("/") + "/"):
            raise E7CTargetFirewallViolation(
                f"{fold_id}: path {candidate_path!r} falls under forbidden root {forbidden!r} -- "
                "this is the held-out target's own domain/label root; FAIL CLOSED, never opened")


def build_target_isolation_report(repo: Path, fold_id: str) -> dict[str, Any]:
    if fold_id not in FOLD_IDS:
        raise E7CError(f"unknown fold_id {fold_id!r}")
    materialization = e7b.load_e7a_fold_materialization(repo, fold_id)
    target_domain = FOLD_TARGET_DOMAIN[fold_id]
    violations: list[str] = []
    checked_refs = 0
    if materialization is not None:
        for ref in (materialization.get("source_train_references", []) +
                   materialization.get("source_dev_references", [])):
            checked_refs += 1
            if ref.get("dataset") == target_domain:
                violations.append(f"source reference {ref.get('sample_id') or ref.get('video_id')} "
                                  f"belongs to the held-out target domain {target_domain!r}")
            path = ref.get("image_relative_path") or ref.get("relative_path")
            if path:
                try:
                    assert_not_target_path(fold_id, path)
                except E7CTargetFirewallViolation as exc:
                    violations.append(str(exc))
    return {
        "schema_version": f"{SCHEMA_PREFIX}-target-isolation-report-v1", "fold_id": fold_id,
        "heldout_target_domain": target_domain,
        "forbidden_roots": list(forbidden_roots_for_fold(fold_id)),
        "source_refs_checked": checked_refs, "violations": violations,
        "target_absent_from_source_refs": not violations,
        "target_image_bytes_opened": False, "target_label_bytes_opened": False,
        "target_package_identity_referenced_as_metadata_only": (
            build_f1_target_metadata_identity(repo)["identity"] if fold_id == "EXT-F1" else
            {"EXT-F2": FROZEN_E7B["msu_target_package_identity"],
            "EXT-F3": FROZEN_E7B["casia_target_package_identity"]}[fold_id]),
        "pass": not violations,
        "target_access": False, "llm_api_calls": 0,
    }


def write_target_isolation_report(repo: Path) -> dict[str, Any]:
    reports = {fold_id: build_target_isolation_report(repo, fold_id) for fold_id in FOLD_IDS}
    for fold_id, report in reports.items():
        if not report["pass"]:
            raise E7CError(f"{fold_id}: target isolation FAILED -- {report['violations']!r}")
    return _write(repo, "E7C_TARGET_ISOLATION_REPORT.json", {
        "schema_version": f"{SCHEMA_PREFIX}-target-isolation-report-all-v1", "folds": reports,
        "target_access": False, "llm_api_calls": 0})


# --------------------------------------------------------------------------- #
# TASK E -- per-fold source binding + source-live pool plan (data-driven
# from E7-A's own frozen references; never re-derives from a fresh scan)
# --------------------------------------------------------------------------- #

def build_f1_target_metadata_identity(repo: Path) -> dict[str, Any]:
    """F1's held-out target is the pre-existing SiW target-eval package
    (`data/processed/prism_target_eval_v2`), NOT an E7-B-built package.
    Reuses `e7b.build_f1_target_reuse_binding` VERBATIM (never
    reimplemented) -- it reads only `PACKAGE_LOCK.json`'s
    `content_identity_sha256`, never any image/crop byte."""
    binding = e7b.build_f1_target_reuse_binding(repo)
    return {"identity": binding["package_identity"], "status": binding["package_identity_status"],
           "source": e7b.SIW_TARGET_EVAL_PACKAGE_ROOT + "/PACKAGE_LOCK.json",
           "resolved_via": "e7b.build_f1_target_reuse_binding (reused verbatim)"}


def build_fold_source_binding(repo: Path, fold_id: str) -> dict[str, Any]:
    if fold_id not in FOLD_IDS:
        raise E7CError(f"unknown fold_id {fold_id!r}")
    materialization = e7b.load_e7a_fold_materialization(repo, fold_id)
    if materialization is None:
        return {"schema_version": f"{SCHEMA_PREFIX}-fold-source-binding-v1", "fold_id": fold_id,
               "status": "MATERIALIZATION_MISSING", "target_access": False, "llm_api_calls": 0}

    train_refs = materialization["source_train_references"]
    dev_refs = materialization["source_dev_references"]
    all_refs = train_refs + dev_refs
    datasets_present = sorted({ref["dataset"] for ref in all_refs})
    target_domain = FOLD_TARGET_DOMAIN[fold_id]

    if fold_id == "EXT-F1":
        f1_target = build_f1_target_metadata_identity(repo)
        target_package_identity = f1_target["identity"]
        target_identity_note = ("F1's target is the pre-existing SiW target-eval package, reused "
                                "verbatim via e7b.build_f1_target_reuse_binding; identity status="
                                f"{f1_target['status']}; metadata only, never opened for bytes")
    else:
        target_package_identity = {"EXT-F2": FROZEN_E7B["msu_target_package_identity"],
                                   "EXT-F3": FROZEN_E7B["casia_target_package_identity"]}[fold_id]
        target_identity_note = "frozen E7-B target package identity; metadata only, never opened for bytes"

    return {
        "schema_version": f"{SCHEMA_PREFIX}-fold-source-binding-v1", "fold_id": fold_id,
        "source_domains": list(FOLD_SOURCE_DOMAINS[fold_id]), "heldout_target_domain": target_domain,
        "datasets_present_in_source_refs": datasets_present,
        "heldout_target_absent_from_source_refs": target_domain not in datasets_present,
        "source_train_row_count": len(train_refs), "source_dev_row_count": len(dev_refs),
        "source_package_identities": {
            "m3b_package_identity": materialization.get("m3b_package_identity"),
            "siw_source_package_identity": (FROZEN_E7B["siw_source_package_identity"]
                                            if "SiW-Mv2" in FOLD_SOURCE_DOMAINS[fold_id] else None),
        },
        "target_package_identity": target_package_identity,
        "target_package_identity_note": target_identity_note,
        "target_reference_kind": materialization.get("target_reference", {}).get("kind"),
        "target_image_crop_bytes_opened": False,
        "fold_identity": materialization.get("fold_identity"),
        "status": "RESOLVED", "target_access": False, "llm_api_calls": 0,
    }


def write_fold_source_binding(repo: Path) -> dict[str, Any]:
    bindings = {fold_id: build_fold_source_binding(repo, fold_id) for fold_id in FOLD_IDS}
    for fold_id, binding in bindings.items():
        if binding["status"] == "RESOLVED":
            if tuple(binding["source_domains"]) != FOLD_SOURCE_DOMAINS[fold_id]:
                raise E7CError(f"{fold_id}: source domains do not match the governing fold plan")
            if not binding["heldout_target_absent_from_source_refs"]:
                raise E7CError(f"{fold_id}: held-out target domain found among source references -- "
                               "FAIL CLOSED")
    return _write(repo, "E7C_FOLD_SOURCE_BINDING.json", {
        "schema_version": f"{SCHEMA_PREFIX}-fold-source-binding-all-v1", "folds": bindings,
        "target_access": False, "llm_api_calls": 0})


#: The whole-package SiW E7-B aggregate (GPU-verified, commit 42e56d0). This
#: is a PACKAGE-LEVEL total, never a per-fold crop-row count -- F2 and F3
#: share the identical SiW train/dev video assignment, so it applies
#: identically to both, but it is NEVER divided/inferred down to an exact
#: live-train/live-dev crop-row split without the real package join.
FROZEN_SIW_PACKAGE_AGGREGATE = {"planned_frame_count": 6800, "successful_crop_count": 6776,
                                "failure_count": 24}


def build_source_live_pool_plan(repo: Path, fold_id: str) -> dict[str, Any]:
    """TWO-LEVEL plan, unit-separated by construction -- never summed across
    granularities:

    (A) VIDEO-LEVEL frozen SiW split (siw_video_*_count /
        siw_live_video_*_count) -- exact, resolved locally from E7-A's own
        frozen references; a video reference is NOT a crop/support row.

    (B) CROP-LEVEL M3B support rows (m3b_crop_*_rows /
        m3b_live_crop_*_rows) -- exact, resolved locally (M3B references
        already ARE successful crops).

    SiW CROP-level support rows (what GPATRoute/SampleStore actually
    consume) are NEVER inferred by multiplying video counts by 4: the 24
    no_face failures have a real, unknown-until-joined distribution across
    train/dev/live/spoof. Those fields are explicitly
    UNRESOLVED_GPU_REQUIRED on this laptop.
    """
    if fold_id not in FOLD_IDS:
        raise E7CError(f"unknown fold_id {fold_id!r}")
    materialization = e7b.load_e7a_fold_materialization(repo, fold_id)
    if materialization is None:
        return {"schema_version": f"{SCHEMA_PREFIX}-source-live-pool-plan-v1", "fold_id": fold_id,
               "status": "MATERIALIZATION_MISSING", "target_access": False, "llm_api_calls": 0}

    train_refs = materialization["source_train_references"]
    dev_refs = materialization["source_dev_references"]
    m3b_train = [r for r in train_refs if r["dataset"] != "SiW-Mv2"]
    m3b_dev = [r for r in dev_refs if r["dataset"] != "SiW-Mv2"]
    siw_train = [r for r in train_refs if r["dataset"] == "SiW-Mv2"]
    siw_dev = [r for r in dev_refs if r["dataset"] == "SiW-Mv2"]
    siw_in_fold = "SiW-Mv2" in FOLD_SOURCE_DOMAINS[fold_id]

    m3b_plan = {
        "granularity": "CROP_ROW", "datasets": sorted({r["dataset"] for r in m3b_train + m3b_dev}),
        "m3b_crop_train_rows": len(m3b_train), "m3b_crop_dev_rows": len(m3b_dev),
        "m3b_live_crop_train_rows": sum(1 for r in m3b_train if r.get("label_live_spoof") == "live"),
        "m3b_live_crop_dev_rows": sum(1 for r in m3b_dev if r.get("label_live_spoof") == "live"),
        "terminal_preprocessing_failures": 0,  # M3B references only ever enumerate successful crops
        "note": "every M3B reference IS a successful crop by construction",
    }

    siw_plan = None
    if siw_in_fold:
        siw_plan = {
            "video_level": {
                "granularity": "CANONICAL_VIDEO",
                "siw_video_train_count": len(siw_train), "siw_video_dev_count": len(siw_dev),
                "siw_live_video_train_count": sum(1 for r in siw_train
                                                  if r.get("label_live_spoof") == "live"),
                "siw_live_video_dev_count": sum(1 for r in siw_dev
                                                if r.get("label_live_spoof") == "live"),
                "resolved_locally": True,
                "note": "these are E7-A CANONICAL VIDEO reference counts, never crop/support rows",
            },
            "crop_level": {
                "granularity": "CROP_ROW",
                "siw_success_crop_total": FROZEN_SIW_PACKAGE_AGGREGATE["successful_crop_count"],
                "siw_failure_total": FROZEN_SIW_PACKAGE_AGGREGATE["failure_count"],
                "siw_planned_frame_total": FROZEN_SIW_PACKAGE_AGGREGATE["planned_frame_count"],
                "siw_live_crop_train_rows": "GPU_REQUIRED",
                "siw_live_crop_dev_rows": "GPU_REQUIRED",
                "resolved_locally": False,
                "note": "the whole-package aggregate (6776 success / 24 failure) is frozen GPU "
                       "evidence and applies identically to F2/F3 (same shared SiW package/split), "
                       "but the EXACT per-train/dev, per-live/spoof crop-row split is NOT inferred by "
                       "multiplying video counts by 4 -- it requires the real GPU join described in "
                       "gpu_crop_level_join_contract (source_video_id -> E7-A video ref -> "
                       "project_split/label_live_spoof), never executed on this laptop",
            },
            "gpu_crop_level_join_contract": [
                "1. read source_video_id from each E7-B SIW_SOURCE_PACKAGE.json row",
                "2. join to the authoritative E7-A siw_raw_video reference by video_id",
                "3. inherit project_split (train/dev), label_live_spoof (live/spoof), spoof_family",
                "4. keep only rows with status == 'success'",
                "5. for GPAT live support, additionally keep only label_live_spoof == 'live'",
                "6. preserve crop_relative_path and crop_sha256 from the E7-B row verbatim",
                "7. resolve the crop under siw_source_v1/m2_run/<crop_relative_path> "
                "(e7b._e7b_m2_output_root semantics) -- no subject_id needed, never sample a "
                "replacement frame for any of the 24 terminal failures",
            ],
        }

    return {
        "schema_version": f"{SCHEMA_PREFIX}-source-live-pool-plan-v2", "fold_id": fold_id,
        "source_domains": list(FOLD_SOURCE_DOMAINS[fold_id]),
        "m3b_plan": m3b_plan, "siw_plan": siw_plan,
        "SOURCE_VIDEO_SPLIT_PLAN_VALID": True,
        "SOURCE_CROP_SUPPORT_PLAN_VALID": True,
        "SOURCE_CROP_SUPPORT_BYTES_LOCAL": False,
        "SOURCE_CROP_SUPPORT_MATERIALIZATION_STATUS": "GPU_REQUIRED" if siw_in_fold else
                                                       "LOCAL_M3B_ONLY_NO_SIW_JOIN_NEEDED",
        "mixed_unit_total_never_computed": True,
        "no_subject_id_required_for_siw": True,
        "status": "RESOLVED", "target_access": False, "llm_api_calls": 0,
    }


def write_source_live_pool_plan(repo: Path) -> dict[str, Any]:
    plans = {fold_id: build_source_live_pool_plan(repo, fold_id) for fold_id in FOLD_IDS}
    return _write(repo, "E7C_SOURCE_LIVE_POOL_PLAN.json", {
        "schema_version": f"{SCHEMA_PREFIX}-source-live-pool-plan-all-v1", "folds": plans,
        "target_access": False, "llm_api_calls": 0})


# --------------------------------------------------------------------------- #
# TASK F -- recipe bank binding (RND/DET/LLM/LLM-SHUFFLE-A). Fail closed if
# any REQUIRED bank cannot be bound.
# --------------------------------------------------------------------------- #

def build_recipe_bank_binding(repo: Path) -> dict[str, Any]:
    """Every arm's `identity` is now qualified by an explicit
    `observed_binding_identity_kind` -- audited from the ACTUAL code that
    computes it (`prism_fas/pipeline/checks.py` for `bank_identity`;
    `c_ext_llm_shuffle.py` for the shuffle content hash), never assumed to
    be a canonical scientific recipe identity merely because it is present.
    """
    bindings: dict[str, Any] = {}
    for arm, root in RECIPE_BANK_ROOTS.items():
        bank_path = repo / root / "C3_BANK.json"
        recipes_path = repo / root / "recipes.jsonl"
        if bank_path.is_file() and recipes_path.is_file():
            bank = cc.read_json(bank_path)
            row_count = sum(1 for _ in recipes_path.open("r", encoding="utf-8") if _.strip())
            entry = {
                "source_path": str(Path(root) / "recipes.jsonl"), "row_count": row_count,
                "observed_binding_identity": bank.get("bank_identity"),
                "observed_binding_identity_kind": RECIPE_BANK_IDENTITY_KIND,
                "identity_algorithm": RECIPE_BANK_IDENTITY_ALGORITHM,
                "selected_set_identity": bank.get("selected_set_identity"),
                "selected_set_identity_kind": SELECTED_SET_IDENTITY_KIND,
                "selected_set_identity_algorithm": SELECTED_SET_IDENTITY_ALGORITHM,
                "status": ("FROZEN_REUSE" if row_count == EXPECTED_RECIPE_BANK_ROW_COUNT
                          else "UNRESOLVED"),
            }
            if arm == "LLM":
                equivalence_path = repo / LLM_EQUIVALENCE_EVIDENCE_PATH
                equivalence_proven = False
                raw_content_identity = None
                if equivalence_path.is_file():
                    lock = cc.read_json(equivalence_path)
                    raw_content_identity = lock.get("original_recipe_identity_file_sha_v2")
                    semantic = lock.get("original_recipe_identity_semantic_frozen")
                    equivalence_proven = (lock.get("original_recipe_content_equivalence") == "PROVEN"
                                          and semantic == bank.get("selected_set_identity"))
                entry.update({
                    "canonical_selected_set_identity": bank.get("selected_set_identity"),
                    "raw_content_identity": raw_content_identity,
                    "raw_content_identity_kind": RAW_CONTENT_IDENTITY_KIND,
                    "equivalence_proven": equivalence_proven,
                    "equivalence_evidence_path": (LLM_EQUIVALENCE_EVIDENCE_PATH
                                                  if equivalence_path.is_file() else None),
                })
            bindings[arm] = entry
        else:
            bindings[arm] = {"source_path": str(Path(root) / "recipes.jsonl"), "row_count": 0,
                            "observed_binding_identity": None,
                            "observed_binding_identity_kind": None, "status": "UNRESOLVED"}

    shuffle_recipes = repo / LLM_SHUFFLE_A_RECIPES_PATH
    shuffle_audit = repo / LLM_SHUFFLE_A_AUDIT_PATH
    if shuffle_recipes.is_file() and shuffle_audit.is_file():
        audit = cc.read_json(shuffle_audit)
        row_count = sum(1 for _ in shuffle_recipes.open("r", encoding="utf-8") if _.strip())
        bindings["LLM-SHUFFLE-A"] = {
            "source_path": LLM_SHUFFLE_A_RECIPES_PATH, "row_count": row_count,
            "observed_binding_identity": audit.get("shuffled_bank_identity"),
            "observed_binding_identity_kind": SHUFFLE_IDENTITY_KIND,
            "identity_algorithm": SHUFFLE_IDENTITY_ALGORITHM,
            "seed": audit.get("seed"), "source_bank": audit.get("source_bank"),
            "status": "FROZEN_REUSE" if row_count == EXPECTED_RECIPE_BANK_ROW_COUNT else "UNRESOLVED",
        }
    else:
        bindings["LLM-SHUFFLE-A"] = {"source_path": LLM_SHUFFLE_A_RECIPES_PATH, "row_count": 0,
                                     "observed_binding_identity": None,
                                     "observed_binding_identity_kind": None, "status": "UNRESOLVED"}

    all_bound = all(b["status"] == "FROZEN_REUSE" for b in bindings.values())
    return {"schema_version": f"{SCHEMA_PREFIX}-recipe-bank-binding-v2", "bindings": bindings,
           "all_required_banks_bound": all_bound, "target_access": False, "llm_api_calls": 0}


def write_recipe_bank_binding(repo: Path) -> dict[str, Any]:
    binding = build_recipe_bank_binding(repo)
    if not binding["all_required_banks_bound"]:
        unresolved = {k: v for k, v in binding["bindings"].items() if v["status"] != "FROZEN_REUSE"}
        raise E7CError(f"one or more required recipe banks are UNRESOLVED -- FAIL CLOSED: {unresolved!r}")
    return _write(repo, "E7C_RECIPE_BANK_BINDING.json", binding)


# --------------------------------------------------------------------------- #
# TASK G -- GPAT support policy (documents resolved reuse primitives; never
# invokes them; PLAN ONLY)
# --------------------------------------------------------------------------- #

def build_gpat_support_plan(repo: Path, fold_id: str) -> dict[str, Any]:
    if fold_id not in FOLD_IDS:
        raise E7CError(f"unknown fold_id {fold_id!r}")
    return {
        "schema_version": f"{SCHEMA_PREFIX}-gpat-support-plan-v1", "fold_id": fold_id,
        "plan_only": True, "image_bytes_opened": False, "gpat_fitting_performed": False,
        "resolved_primitives": {
            "support_pairing": "prism_fas.synthesis.c5_source_pair_plan.build_source_pair_plan / "
                               "SourceRow (frozen Version-C schedule: 256 recipes/arm x 8 renders, "
                               "ROUTE_BY_SLOT alternates Physics/GPAT, GPAT slots alternate "
                               "same_domain/cross_domain)",
            "legacy_fitting_pairing": "prism_fas.synthesis.pair_plan.build_pair_plan / SourceRow "
                                      "(PAIRS_PER_LIVE=4, 80/20 train/validation split, "
                                      "PAIR_PLAN_SEED=20260806)",
            "crop_loader": "prism_fas.synthesis.m8_pipeline.SampleStore.open/.load (reads "
                           "manifests/source_train.parquet image_relative_path + prior_relative_path "
                           "-- never raw video, never a second loader)",
            "source_live_selection": "row.label == 'live' filter on the source_train parquet "
                                     "label_live_spoof column, exactly as "
                                     "pair_plan.load_source_train_rows / "
                                     "c5_source_pair_plan.load_source_rows already implement",
            "gpat_fitting": "prism_fas.synthesis.gpat_trainer.GPATTrainer.fit "
                            "(single-fold CASIA+MSU-source implementation; NEVER yet run per-fold or "
                            "with SiW-as-source -- UNRESOLVED for cross-domain reuse, see caveats)",
            "candidate_generation": "prism_fas.synthesis.synthetic_bank.GPATRoute.generate "
                                    "(+ PhysicsRoute.generate for the Physics route)",
            "candidate_acceptance": "prism_fas.synthesis.quality_gate.evaluate against "
                                    "prism_fas.synthesis.quality_calibration.FrozenCalibration "
                                    "thresholds (8 HARD_GATES; q is a training weight, never an "
                                    "acceptance criterion)",
            "quality_computation": "prism_fas.synthesis.quality_gate (landmark_nme, parsing_dice, "
                                   "support_overlap, triangular_strength_score) + "
                                   "prism_fas.synthesis.fingerprint (fingerprint_score)",
            "domain_quota": "prism_fas.synthesis.c6_matched_bank (largest_remainder_quota, "
                            "common_capacity, resolve_route_quota) -- frozen, never fungible across "
                            "domains (this is the exact mechanism behind the F1 Shuffle infeasibility)",
            "bank_size": "gate_profiles.FINAL_BANK_PER_ARM=1024 (PHYSICS_PER_ARM=512, "
                         "GPAT_PER_ARM=512), distinct from the C3 384-slot/256-selected recipe "
                         "generation schedule",
            "seed_convention": "PAIR_PLAN_SEED=20260806 reused verbatim for GPAT fitting/pairing "
                               "determinism, matching every other C5/C6/pair-plan module -- no new "
                               "seed invented for E7-C",
        },
        "unresolved_for_cross_domain_reuse": [
            "GPATTrainer has never been parameterized by fold or run with SiW-as-source; per-fold "
            "invocation requires a NEW, explicit scientific decision/implementation before any GPU "
            "fitting -- NOT performed by this milestone",
            "prism_fas.synthesis.m8_pipeline.SourceOnlyAudit hard-bans the literal substring 'siw' "
            "in any opened path, which is INCOMPATIBLE with EXT-F2/F3 (SiW is legitimately source "
            "there) -- this module's own fold-aware firewall (assert_not_target_path) is used "
            "instead; SourceOnlyAudit itself is NOT reused verbatim for F2/F3",
        ],
        "source_support_status": build_source_live_pool_plan(repo, fold_id) if
            e7b.load_e7a_fold_materialization(repo, fold_id) is not None else
            {"status": "MATERIALIZATION_MISSING"},
        "note": "M3B crop-level support rows are resolved locally (M3B references are already crops). "
               "SiW crop-level support rows (this fold's siw_plan.crop_level, if SiW is a source "
               "domain here) are GPU_REQUIRED and follow the gpu_crop_level_join_contract documented "
               "there verbatim -- never inferred, never sampled here.",
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False,
        "training_performed": False,
    }


def write_gpat_support_plan(repo: Path) -> dict[str, Any]:
    plans = {fold_id: build_gpat_support_plan(repo, fold_id) for fold_id in FOLD_IDS}
    return _write(repo, "E7C_GPAT_SUPPORT_PLAN.json", {
        "schema_version": f"{SCHEMA_PREFIX}-gpat-support-plan-all-v1", "folds": plans,
        "target_access": False, "llm_api_calls": 0})


# --------------------------------------------------------------------------- #
# TASK H -- feasibility plan (condition status matrix; F1 Shuffle hard
# block preserved verbatim; F2/F3 remain independently pending)
# --------------------------------------------------------------------------- #

def build_condition_status(fold_id: str, condition: str) -> str:
    if condition == "G-REALONLY":
        return NOT_APPLICABLE
    if condition == "G-LLM-SHUFFLE-A":
        if fold_id == "EXT-F1":
            return BLOCKED_TRUE_FROZEN_MATCHED_BANK_INFEASIBILITY
        return PENDING_FEASIBILITY_PREFLIGHT
    return READY_FOR_GPAT_PREP  # RND / DET / LLM


def build_feasibility_plan(repo: Path, fold_id: str) -> dict[str, Any]:
    if fold_id not in FOLD_IDS:
        raise E7CError(f"unknown fold_id {fold_id!r}")
    conditions = {c: build_condition_status(fold_id, c) for c in CONDITIONS}
    return {
        "schema_version": f"{SCHEMA_PREFIX}-feasibility-plan-v1", "fold_id": fold_id,
        "conditions": conditions,
        "shuffle_basis": FROZEN_E6V2_CLOSURE if fold_id == "EXT-F1" else {
            "reason": "EXT-F1's Physics infeasibility is a property of EXT-F1's OWN quality-pass "
                     f"source-domain composition; {fold_id} draws from a different source-domain "
                     "pair and must be independently rendered/measured/matched before any "
                     "feasibility conclusion is drawn",
            "ext_f1_result_does_not_predetermine_this_fold": True,
        },
        "forbidden_rescue_actions": ["lower bank size 512", "change source-domain quotas",
                                     "relax quality thresholds", "modify q", "resample candidates",
                                     "rerender solely to obtain a passing bank", "change matching policy"],
        "no_parameter_changes_permitted": True,
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False,
        "training_performed": False, "gpat_fitting_performed": False,
    }


def write_feasibility_plan(repo: Path) -> dict[str, Any]:
    plans = {fold_id: build_feasibility_plan(repo, fold_id) for fold_id in FOLD_IDS}
    if plans["EXT-F1"]["conditions"]["G-LLM-SHUFFLE-A"] != BLOCKED_TRUE_FROZEN_MATCHED_BANK_INFEASIBILITY:
        raise E7CError("EXT-F1/G-LLM-SHUFFLE-A must remain BLOCKED_TRUE_FROZEN_MATCHED_BANK_"
                       "INFEASIBILITY -- refusing to write a plan that silently changed this")
    for fold_id in ("EXT-F2", "EXT-F3"):
        if plans[fold_id]["conditions"]["G-LLM-SHUFFLE-A"] != PENDING_FEASIBILITY_PREFLIGHT:
            raise E7CError(f"{fold_id}/G-LLM-SHUFFLE-A must remain PENDING_FEASIBILITY_PREFLIGHT -- "
                           "never falsely blocked or passed without independent evidence")
    return _write(repo, "E7C_FEASIBILITY_PLAN.json", {
        "schema_version": f"{SCHEMA_PREFIX}-feasibility-plan-all-v1", "folds": plans,
        "target_access": False, "llm_api_calls": 0})


# --------------------------------------------------------------------------- #
# TASK I -- execution plan (aggregate rollup; still plan-only)
# --------------------------------------------------------------------------- #

def build_execution_plan(repo: Path) -> dict[str, Any]:
    source_bindings = {fold_id: build_fold_source_binding(repo, fold_id) for fold_id in FOLD_IDS}
    feasibility = {fold_id: build_feasibility_plan(repo, fold_id) for fold_id in FOLD_IDS}
    isolation = {fold_id: build_target_isolation_report(repo, fold_id) for fold_id in FOLD_IDS}
    recipe_binding = build_recipe_bank_binding(repo)

    per_fold = {}
    for fold_id in FOLD_IDS:
        per_fold[fold_id] = {
            "source_ready": source_bindings[fold_id]["status"] == "RESOLVED" and
                            source_bindings[fold_id]["heldout_target_absent_from_source_refs"],
            "target_excluded": isolation[fold_id]["pass"],
            "conditions": feasibility[fold_id]["conditions"],
        }

    return {
        "schema_version": f"{SCHEMA_PREFIX}-execution-plan-v1", "per_fold": per_fold,
        "recipe_banks_bound": recipe_binding["all_required_banks_bound"],
        "next_gpu_stages": ["per-fold GPAT support pool materialization (image bytes)",
                            "per-fold GPAT fitting", "per-fold candidate generation",
                            "F2/F3 Shuffle-A independent feasibility rendering",
                            "matched-bank quota resolution", "package/bank integrity locks"],
        "e7_ready_for_gpu_gpat_preparation": all(v["source_ready"] and v["target_excluded"]
                                                 for v in per_fold.values()) and
                                             recipe_binding["all_required_banks_bound"],
        "e7_ready_for_training": False,
        "e7_ready_for_training_reason": "per-fold GPAT support is not yet materialized, no synthetic "
                                        "candidate bank exists, F2/F3 Shuffle feasibility is unresolved, "
                                        "and no package/bank integrity lock has run",
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False,
        "training_performed": False, "gpat_fitting_performed": False,
    }


def write_execution_plan(repo: Path) -> dict[str, Any]:
    return _write(repo, "E7C_EXECUTION_PLAN.json", build_execution_plan(repo))


# --------------------------------------------------------------------------- #
# TASK J -- readiness rollup + strict read-only preflight
# --------------------------------------------------------------------------- #

def e7c_preflight(repo: Path) -> dict[str, Any]:
    """`--e7c-preflight`: STRICTLY READ-ONLY. Performs no rendering, no GPAT
    fitting, no candidate generation, no training, no target scoring, no
    LLM calls. Writes nothing."""
    e7b_binding = build_e7b_binding(repo)
    e7a_bindings = {fold_id: build_e7a_fold_binding(repo, fold_id) for fold_id in FOLD_IDS}
    source_bindings = {fold_id: build_fold_source_binding(repo, fold_id) for fold_id in FOLD_IDS}
    isolation = {fold_id: build_target_isolation_report(repo, fold_id) for fold_id in FOLD_IDS}
    recipe_binding = build_recipe_bank_binding(repo)
    feasibility = {fold_id: build_feasibility_plan(repo, fold_id) for fold_id in FOLD_IDS}

    e7a_fold_binding_match = all(b["match"] or not b["materialization_present"]
                                 for b in e7a_bindings.values()) and \
                             all(b["materialization_present"] for b in e7a_bindings.values())

    required = {
        "E7B_FINAL_BINDING_MATCH": e7b_binding["match"],
        "E7A_FOLD_BINDING_MATCH": e7a_fold_binding_match,
        "F1_SOURCE_READY": source_bindings["EXT-F1"]["status"] == "RESOLVED" and
                          source_bindings["EXT-F1"]["heldout_target_absent_from_source_refs"],
        "F2_SOURCE_READY": source_bindings["EXT-F2"]["status"] == "RESOLVED" and
                          source_bindings["EXT-F2"]["heldout_target_absent_from_source_refs"],
        "F3_SOURCE_READY": source_bindings["EXT-F3"]["status"] == "RESOLVED" and
                          source_bindings["EXT-F3"]["heldout_target_absent_from_source_refs"],
        "F1_TARGET_EXCLUDED": isolation["EXT-F1"]["pass"],
        "F2_TARGET_EXCLUDED": isolation["EXT-F2"]["pass"],
        "F3_TARGET_EXCLUDED": isolation["EXT-F3"]["pass"],
        "RND_RECIPE_BANK_BOUND": recipe_binding["bindings"]["RND"]["status"] == "FROZEN_REUSE",
        "DET_RECIPE_BANK_BOUND": recipe_binding["bindings"]["DET"]["status"] == "FROZEN_REUSE",
        "LLM_RECIPE_BANK_BOUND": recipe_binding["bindings"]["LLM"]["status"] == "FROZEN_REUSE",
        "SHUFFLE_RECIPE_BANK_BOUND": recipe_binding["bindings"]["LLM-SHUFFLE-A"]["status"] ==
                                     "FROZEN_REUSE",
    }
    build_pass = all(required.values())

    return {
        "schema_version": f"{SCHEMA_PREFIX}-preflight-v1",
        **required,
        "F1_SHUFFLE_STATUS": feasibility["EXT-F1"]["conditions"]["G-LLM-SHUFFLE-A"],
        "F2_SHUFFLE_STATUS": feasibility["EXT-F2"]["conditions"]["G-LLM-SHUFFLE-A"],
        "F3_SHUFFLE_STATUS": feasibility["EXT-F3"]["conditions"]["G-LLM-SHUFFLE-A"],
        "e7b_local_data_state": e7b_binding["local_data_state"],
        # PLAN/BINDING validity is fully resolvable locally and IS what
        # E7C_PREFLIGHT_PASS reflects; crop-level MATERIALIZATION is a
        # separate, always-false-on-laptop concept -- never conflated.
        "E7C_SOURCE_SUPPORT_PLAN_VALID": required["F1_SOURCE_READY"] and required["F2_SOURCE_READY"]
                                        and required["F3_SOURCE_READY"],
        "E7C_SOURCE_SUPPORT_MATERIALIZED": False,
        "E7C_SIW_CROP_BINDING_REQUIRED_ON_GPU": True,
        "F1_SOURCE_PLAN_READY": required["F1_SOURCE_READY"],
        "F2_SOURCE_PLAN_READY": required["F2_SOURCE_READY"],
        "F3_SOURCE_PLAN_READY": required["F3_SOURCE_READY"],
        "F1_SOURCE_CROP_POOL_MATERIALIZED": False,
        "F2_SOURCE_CROP_POOL_MATERIALIZED": False,
        "F3_SOURCE_CROP_POOL_MATERIALIZED": False,
        "TARGET_LABEL_ACCESS": False, "TARGET_IMAGE_ACCESS": False,
        "RENDERING_PERFORMED": False, "GPAT_FITTING_PERFORMED": False,
        "TRAINING_PERFORMED": False, "LLM_API_CALLS": 0,
        "E7C_PREFLIGHT_PASS": build_pass,
        # Previously omitted from this function's own output (only
        # build_readiness()/build_execution_plan() emitted them) -- a
        # TECHNICAL output-schema gap, not a scientific-result change:
        # E7_READY_FOR_GPU_GPAT_PREPARATION is defined as exactly
        # E7C_PREFLIGHT_PASS (iff every required binding/plan check above
        # passes); E7_READY_FOR_TRAINING is unconditionally False at E7-C.
        "E7_READY_FOR_GPU_GPAT_PREPARATION": build_pass,
        "E7_READY_FOR_TRAINING": False,
        "E7_READY_FOR_TRAINING_REASON": build_execution_plan(repo)["e7_ready_for_training_reason"],
        "readiness_note": "E7C_PREFLIGHT_PASS=True means PLAN/BINDING VALID AND SAFE TO GO TO GPU "
                          "(READY_FOR_GPU_GPAT_PREPARATION), NOT that any crop-level SiW support "
                          "pool is materialized, and NOT READY_FOR_E7_TRAINING -- training readiness "
                          "additionally requires materialized per-fold GPAT support, materialized "
                          "synthetic candidate banks, resolved F2/F3 Shuffle feasibility, and passing "
                          "package/bank integrity locks",
    }


def build_readiness(repo: Path) -> dict[str, Any]:
    preflight = e7c_preflight(repo)
    execution_plan = build_execution_plan(repo)
    return {
        "schema_version": f"{SCHEMA_PREFIX}-readiness-v1",
        "E7C_PREFLIGHT_PASS": preflight["E7C_PREFLIGHT_PASS"],
        "E7_READY_FOR_GPU_GPAT_PREPARATION": execution_plan["e7_ready_for_gpu_gpat_preparation"],
        "E7_READY_FOR_TRAINING": False,
        "E7_READY_FOR_TRAINING_REASON": execution_plan["e7_ready_for_training_reason"],
        "E7C_SOURCE_SUPPORT_PLAN_VALID": preflight["E7C_SOURCE_SUPPORT_PLAN_VALID"],
        "E7C_SOURCE_SUPPORT_MATERIALIZED": preflight["E7C_SOURCE_SUPPORT_MATERIALIZED"],
        "E7C_SIW_CROP_BINDING_REQUIRED_ON_GPU": preflight["E7C_SIW_CROP_BINDING_REQUIRED_ON_GPU"],
        "F1_SOURCE_PLAN_READY": preflight["F1_SOURCE_PLAN_READY"],
        "F2_SOURCE_PLAN_READY": preflight["F2_SOURCE_PLAN_READY"],
        "F3_SOURCE_PLAN_READY": preflight["F3_SOURCE_PLAN_READY"],
        "F1_SOURCE_CROP_POOL_MATERIALIZED": preflight["F1_SOURCE_CROP_POOL_MATERIALIZED"],
        "F2_SOURCE_CROP_POOL_MATERIALIZED": preflight["F2_SOURCE_CROP_POOL_MATERIALIZED"],
        "F3_SOURCE_CROP_POOL_MATERIALIZED": preflight["F3_SOURCE_CROP_POOL_MATERIALIZED"],
        "F1_SHUFFLE_STATUS": preflight["F1_SHUFFLE_STATUS"],
        "F2_SHUFFLE_STATUS": preflight["F2_SHUFFLE_STATUS"],
        "F3_SHUFFLE_STATUS": preflight["F3_SHUFFLE_STATUS"],
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False,
        "training_performed": False, "gpat_fitting_performed": False,
    }


def write_readiness(repo: Path) -> dict[str, Any]:
    return _write(repo, "E7C_READINESS.json", build_readiness(repo))


# --------------------------------------------------------------------------- #
# TASK K -- review-fix audit artifacts (this turn's provenance/granularity
# corrections; additive, never rewrite the artifacts above's OWN history --
# these document WHAT changed and WHY)
# --------------------------------------------------------------------------- #

def build_source_granularity_correction(repo: Path) -> dict[str, Any]:
    plans = {fold_id: build_source_live_pool_plan(repo, fold_id) for fold_id in ("EXT-F2", "EXT-F3")}
    return {
        "schema_version": f"{SCHEMA_PREFIX}-source-granularity-correction-v1",
        "TECHNICAL_ISSUE": "the original E7C_SOURCE_LIVE_POOL_PLAN.json summed SiW E7-A CANONICAL "
                           "VIDEO reference counts together with M3B CROP ROW counts into a single "
                           "source_train_rows/source_live_train_rows total for EXT-F2/EXT-F3 -- a "
                           "mixed-granularity total that is not a valid GPAT support-pool row count "
                           "(GPAT's SampleStore consumes crop/sample rows, never canonical-video refs).",
        "SCIENTIFIC_PROTOCOL_CHANGED": False,
        "SOURCE_GRANULARITY_BUG_FIXED": True,
        "fix": "build_source_live_pool_plan now returns two separate, explicitly-named, "
              "never-summed sections per fold: m3b_plan (CROP_ROW granularity, resolved locally) and "
              "siw_plan.video_level (CANONICAL_VIDEO granularity, resolved locally) + "
              "siw_plan.crop_level (CROP_ROW granularity, GPU_REQUIRED, never inferred by x4).",
        "verified_video_level_counts": {
            fold_id: plans[fold_id]["siw_plan"]["video_level"] for fold_id in plans
        },
        "f3_same_siw_video_split_as_f2": (plans["EXT-F2"]["siw_plan"]["video_level"] ==
                                          plans["EXT-F3"]["siw_plan"]["video_level"]),
        "mixed_unit_totals_removed": True,
        "no_4x_inference_for_live_crop_counts": True,
        "target_access": False, "llm_api_calls": 0,
    }


def write_source_granularity_correction(repo: Path) -> dict[str, Any]:
    return _write(repo, "E7C_SOURCE_GRANULARITY_CORRECTION.json", build_source_granularity_correction(repo))


def build_recipe_identity_provenance(repo: Path) -> dict[str, Any]:
    binding = build_recipe_bank_binding(repo)
    llm = binding["bindings"].get("LLM", {})
    return {
        "schema_version": f"{SCHEMA_PREFIX}-recipe-identity-provenance-v1",
        "TECHNICAL_ISSUE": "the original E7C_RECIPE_BANK_BINDING.json reported each arm's "
                           "C3_BANK.json.bank_identity as an unqualified 'identity', which could be "
                           "mistaken for the canonical scientific recipe-set identity "
                           "(selected_set_identity) -- they are computed by different algorithms over "
                           "different material.",
        "SCIENTIFIC_PROTOCOL_CHANGED": False,
        "per_arm_identity_kinds": {
            arm: {"observed_binding_identity": b.get("observed_binding_identity"),
                 "observed_binding_identity_kind": b.get("observed_binding_identity_kind"),
                 "selected_set_identity": b.get("selected_set_identity")}
            for arm, b in binding["bindings"].items()
        },
        "llm_equivalence": {
            "canonical_selected_set_identity": llm.get("canonical_selected_set_identity"),
            "raw_content_identity": llm.get("raw_content_identity"),
            "equivalence_proven": llm.get("equivalence_proven"),
            "equivalence_evidence_path": llm.get("equivalence_evidence_path"),
        },
        "target_access": False, "llm_api_calls": 0,
    }


def write_recipe_identity_provenance(repo: Path) -> dict[str, Any]:
    return _write(repo, "E7C_RECIPE_IDENTITY_PROVENANCE.json", build_recipe_identity_provenance(repo))


# --------------------------------------------------------------------------- #
# writer plumbing
# --------------------------------------------------------------------------- #

def _write(repo: Path, filename: str, body: dict[str, Any]) -> dict[str, Any]:
    out_dir = repo / E7C_REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    return {"body": body, "path": str(path)}


def prepare_e7c(repo: Path) -> dict[str, Any]:
    """Writes every additive E7-C artifact. Fails closed (raises E7CError,
    writes nothing further) the moment any binding/plan disagrees with its
    frozen authority."""
    results = {
        "protocol_lock": write_protocol_lock(repo),
        "e7b_binding": write_e7b_binding(repo),
        "e7a_fold_binding": write_e7a_fold_binding(repo),
        "fold_source_binding": write_fold_source_binding(repo),
        "source_live_pool_plan": write_source_live_pool_plan(repo),
        "recipe_bank_binding": write_recipe_bank_binding(repo),
        "target_isolation_report": write_target_isolation_report(repo),
        "gpat_support_plan": write_gpat_support_plan(repo),
        "feasibility_plan": write_feasibility_plan(repo),
        "execution_plan": write_execution_plan(repo),
        "readiness": write_readiness(repo),
        "source_granularity_correction": write_source_granularity_correction(repo),
        "recipe_identity_provenance": write_recipe_identity_provenance(repo),
    }
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E7-C per-fold GPAT preparation and feasibility "
                                                 "planning (no render, no GPAT fit, no training, no "
                                                 "target-label/image access, no LLM)")
    parser.add_argument("--e7c-preflight", action="store_true", help="Read-only. Writes nothing.")
    parser.add_argument("--prepare", action="store_true",
                        help="Writes every additive E7-C binding/plan artifact. Fails closed.")
    args = parser.parse_args(argv)
    repo = cc.repo_root()

    if args.e7c_preflight:
        print(json.dumps(e7c_preflight(repo), indent=2, default=str))
        return 0
    if args.prepare:
        try:
            result = prepare_e7c(repo)
        except E7CError as error:
            print(f"E7-C prepare refused: {error}")
            return 1
        print(json.dumps({"readiness": result["readiness"]["body"]}, indent=2, default=str))
        return 0

    print("Pass --e7c-preflight (read-only) or --prepare.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
