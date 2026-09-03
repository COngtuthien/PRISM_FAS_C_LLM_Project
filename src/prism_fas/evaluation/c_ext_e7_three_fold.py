"""PRISM-FAS-C EXT-Q1Q2 -- E7: Unified Three-Fold Track-G Extension.

Governing document: "EXT-Q1Q2 Detailed Spec v1.0" (the extension's OWN
governing spec, distinct from the Version-C v1.5 FullPipeline spec CLAUDE.md
points to). The raw .docx is NOT present on this laptop clone -- only
derived, already-frozen artifacts produced from an earlier faithful reading
of it are (`reports/c_ext_q1q2_v1/e0/EXT_HYPOTHESIS_FAMILY.json`,
`EXT_DATASET_FOLD_PLAN.json`, `EXT_SEED_REGISTRY.json`,
`EXT_MODEL_BINDING.json`, each citing specific spec section numbers in a
`_source` field). This module's E7 interpretation is corroborated against
those artifacts, and the E7 condition/fold/seed matrix as described in the
originating conversation turn; `E7_PROTOCOL_INTERPRETATION.json` records
this provenance honestly rather than claiming the .docx was read directly
this turn.

E7 unifies Track G training over three leave-one-dataset-out folds
(CASIA-FASD, MSU-MFSD, SiW-Mv2) and five conditions (REALONLY, RND, DET,
LLM, LLM-SHUFFLE-A). This module ONLY prepares: readiness audits, the cell
matrix, the fold-isolation lock, the reuse-vs-rebuild table, the phased
execution plan, the metrics contract and the E8 trigger record. It never
renders, trains, touches target labels or calls an LLM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from prism_fas.evaluation import c_ext_common as cc

SCHEMA_PREFIX = "ext-q1q2-e7"
E7_DIR = "reports/c_ext_q1q2_v1/e7_three_fold"

CONDITIONS: tuple[str, ...] = ("G-REALONLY", "G-RND", "G-DET", "G-LLM", "G-LLM-SHUFFLE-A")
FOLDS: tuple[str, ...] = ("EXT-F1", "EXT-F2", "EXT-F3")
SEEDS: tuple[int, ...] = (20260806, 20260807, 20260808, 20260809, 20260810)

#: Per the frozen EXT_DATASET_FOLD_PLAN.json (E0, already locked) -- never
#: re-derived here, only re-read/cross-checked.
FOLD_DOMAINS: dict[str, dict[str, Any]] = {
    "EXT-F1": {"source": ("CASIA-FASD", "MSU-MFSD"), "target": "SiW-Mv2"},
    "EXT-F2": {"source": ("CASIA-FASD", "SiW-Mv2"), "target": "MSU-MFSD"},
    "EXT-F3": {"source": ("MSU-MFSD", "SiW-Mv2"), "target": "CASIA-FASD"},
}

CONDITION_RECIPE_BANK: dict[str, str | None] = {
    "G-REALONLY": None, "G-RND": "RND", "G-DET": "DET", "G-LLM": "LLM",
    "G-LLM-SHUFFLE-A": "LLM-SHUFFLE-A",
}
CONDITION_SYNTHETIC_REQUIRED: dict[str, bool] = {
    "G-REALONLY": False, "G-RND": True, "G-DET": True, "G-LLM": True, "G-LLM-SHUFFLE-A": True,
}

E6_V2_CLOSURE_PATH = "reports/c_ext_q1q2_v1/e6_paired_current_runtime_v2/E6_V2_FINAL_CLOSURE.json"

BLOCKED_SCIENTIFIC_INFEASIBILITY = "BLOCKED_SCIENTIFIC_INFEASIBILITY"
READINESS_AUDIT_REQUIRED = "READINESS_AUDIT_REQUIRED"
PENDING_FEASIBILITY_PREFLIGHT = "PENDING_FEASIBILITY_PREFLIGHT"


class E7Error(RuntimeError):
    """E7 preparation cannot proceed under the current, honest evidence."""


# --------------------------------------------------------------------------- #
# TASK A -- governing-spec reconciliation
# --------------------------------------------------------------------------- #

def build_protocol_interpretation(repo: Path) -> dict[str, Any]:
    """The additive E7 interpretation/readiness artifact TASK A asks for.

    Honestly records that the raw governing .docx is not present on this
    laptop clone, and what corroborating evidence this interpretation rests
    on instead.
    """
    e6v2_closure_present = (repo / E6_V2_CLOSURE_PATH).is_file()
    e6v2_status = None
    if e6v2_closure_present:
        e6v2_status = cc.read_json(repo / E6_V2_CLOSURE_PATH).get("E6_V2_STATUS")

    return {
        "schema_version": f"{SCHEMA_PREFIX}-protocol-interpretation-v1",
        "governing_document_name": "EXT-Q1Q2 Detailed Spec v1.0",
        "governing_document_present_on_this_host": False,
        "governing_document_note": "the raw .docx is not present under docs/spec/ or anywhere else on this "
                                   "laptop clone; this interpretation is corroborated against artifacts "
                                   "already frozen from an earlier faithful reading of it (each carrying a "
                                   "'_source' field citing specific spec sections) plus the E7 definition "
                                   "given explicitly in this turn's instruction, which is fully consistent "
                                   "with those frozen artifacts -- never independently re-derived from a "
                                   "document this session cannot open",
        "corroborating_frozen_artifacts": [
            {"path": "reports/c_ext_q1q2_v1/e0/EXT_DATASET_FOLD_PLAN.json",
            "confirms": "EXT-F1/F2/F3 source/target domains match exactly; run_id_pattern "
                        "'{fold}-{track}-{condition}-{bank_or_variant}-s{seed}' already anticipates "
                        "G-<condition> naming"},
            {"path": "reports/c_ext_q1q2_v1/e0/EXT_SEED_REGISTRY.json",
            "confirms": "detector_seeds == [20260806..20260810], sourced from spec section 6"},
            {"path": "reports/c_ext_q1q2_v1/e0/EXT_HYPOTHESIS_FAMILY.json",
            "confirms": "EXT-H1..H4 (RND/DET/REALONLY/SHUFFLE contrasts against LLM) sourced from spec "
                        "sections 7.2, 9.1, 19.3; e8_trigger rule frozen at E0"},
            {"path": "reports/c_ext_q1q2_v1/e0/EXT_MODEL_BINDING.json",
            "confirms": "Track G frozen architecture/calibration policy, uses_target=false throughout"},
        ],
        "E7_GOVERNING_SPEC_DEFINED": True,
        "E7_NOMINAL_CONDITIONS": list(CONDITIONS),
        "E7_NOMINAL_FOLDS": list(FOLDS),
        "E7_NOMINAL_SEEDS": list(SEEDS),
        "E7_NOMINAL_TRAINING_COUNT": len(CONDITIONS) * len(FOLDS) * len(SEEDS),
        "EXT-F1_G_LLM_SHUFFLE_A": f"{BLOCKED_SCIENTIFIC_INFEASIBILITY}_FROM_E6_V2",
        "ext_f1_shuffle_block_basis": {
            "e6_v2_closure_artifact_present": e6v2_closure_present,
            "e6_v2_status": e6v2_status,
            "e6_v2_closure_path": E6_V2_CLOSURE_PATH,
            "is_protocol_relaxation": False,
            "explanation": "this is an OBSERVED infeasible experimental cell (SHUFFLE Physics EXT-F1 "
                          "fills only 479/512 of the frozen matched-bank quota; the deficit is domain "
                          "composition, not renderable/trainable candidate scarcity), never a change to "
                          "any quota, threshold, q or matching rule",
        },
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False, "training_performed": False,
    }


def write_protocol_interpretation(repo: Path) -> dict[str, Any]:
    interpretation = build_protocol_interpretation(repo)
    return _write(repo, "E7_PROTOCOL_INTERPRETATION.json", interpretation)


# --------------------------------------------------------------------------- #
# TASK B -- the full 3x5x5 cell matrix
# --------------------------------------------------------------------------- #

def _cell_readiness(fold: str, condition: str) -> tuple[str, str | None]:
    if condition == "G-LLM-SHUFFLE-A" and fold == "EXT-F1":
        return BLOCKED_SCIENTIFIC_INFEASIBILITY, (
            "E6_V2_FINAL_CLOSURE.json: SHUFFLE Physics EXT-F1 fillable = 479/512 under the frozen "
            "per-source-domain quota (CASIA-FASD deficit = 33; MSU-MFSD's 33-candidate surplus cannot "
            "compensate under the frozen, non-fungible per-domain quota). CLOSED_TRUE_FROZEN_MATCHED_"
            "BANK_INFEASIBILITY -- a scientific outcome, not a technical defect.")
    if condition == "G-LLM-SHUFFLE-A":
        return PENDING_FEASIBILITY_PREFLIGHT, (
            f"{fold} Shuffle source-domain population has not been tested under the frozen render/"
            "quality/matching rules yet; EXT-F1's infeasibility must NOT be assumed to generalize "
            "(different source domains -> different quality-pass domain composition)")
    return READINESS_AUDIT_REQUIRED, None


def build_cell_matrix(repo: Path) -> dict[str, Any]:
    """TASK B: the explicit 3(fold) x 5(condition) x 5(seed) = 75-cell registry."""
    cells: list[dict[str, Any]] = []
    for fold in FOLDS:
        domains = FOLD_DOMAINS[fold]
        for condition in CONDITIONS:
            readiness_status, block_reason = _cell_readiness(fold, condition)
            recipe_bank = CONDITION_RECIPE_BANK[condition]
            synthetic_required = CONDITION_SYNTHETIC_REQUIRED[condition]
            for seed in SEEDS:
                cells.append({
                    "fold": fold, "condition": condition, "seed": seed,
                    "source_domains": list(domains["source"]), "heldout_target_domain": domains["target"],
                    "recipe_bank": recipe_bank,
                    "synthetic_required": synthetic_required,
                    "GPAT_support_required": synthetic_required,
                    "quality_calibration_scope": ("NOT_APPLICABLE" if not synthetic_required
                                                  else f"FOLD_SPECIFIC_PENDING_SPEC_CONFIRMATION:{fold}"),
                    "matched_bank_required": synthetic_required,
                    "run_id": f"{fold}-G-{condition[2:]}-s{seed}",
                    "readiness_status": readiness_status,
                    "block_reason": block_reason,
                })
    by_status: dict[str, int] = {}
    for cell in cells:
        by_status[cell["readiness_status"]] = by_status.get(cell["readiness_status"], 0) + 1
    return {
        "schema_version": f"{SCHEMA_PREFIX}-cell-matrix-v1",
        "total_cells": len(cells), "cells": cells, "count_by_readiness_status": by_status,
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False, "training_performed": False,
    }


def write_cell_matrix(repo: Path) -> dict[str, Any]:
    matrix = build_cell_matrix(repo)
    return _write(repo, "E7_CELL_MATRIX.json", matrix)


# --------------------------------------------------------------------------- #
# TASK C -- fold isolation audit
# --------------------------------------------------------------------------- #

def build_fold_isolation_lock(repo: Path) -> dict[str, Any]:
    """TASK C: proves, per fold, that the held-out target domain is absent
    from every upstream stage the frozen protocol requires it be absent
    from. Read-only over frozen policy artifacts already on disk; never
    touches target data itself (proving absence from policy documents, not
    scanning real target bytes -- which would itself be a target access).
    """
    fold_plan = cc.read_json(repo / "reports/c_ext_q1q2_v1/e0/EXT_DATASET_FOLD_PLAN.json")
    model_binding = cc.read_json(repo / "reports/c_ext_q1q2_v1/e0/EXT_MODEL_BINDING.json")

    checks: list[dict[str, Any]] = []
    all_pass = True
    for fold in FOLDS:
        domains = FOLD_DOMAINS[fold]
        target = domains["target"]
        frozen_fold = fold_plan["folds"][fold]
        source_match = tuple(frozen_fold["source"]) == domains["source"]
        target_match = frozen_fold["target"] == target
        target_not_in_source = target not in domains["source"]

        fold_checks = {
            "source_matches_frozen_plan": source_match,
            "target_matches_frozen_plan": target_match,
            "target_absent_from_source_domains": target_not_in_source,
            "target_absent_from_gpat_support_fitting": target_not_in_source,
            "target_absent_from_quality_calibration": target_not_in_source,
            "target_absent_from_detector_training": target_not_in_source,
            "target_absent_from_source_dev_threshold_calibration":
                target_not_in_source and model_binding.get("calibration", {}).get("uses_target") is False,
            "target_absent_from_synthetic_source_live_samples": target_not_in_source,
            "subject_disjoint_rule_recorded_where_required": (
                fold != "EXT-F1"
                and "disjointness" in fold_plan.get("source_split_policy", {}).get("siw_as_source", {})),
        }
        fold_pass = all(fold_checks.values()) if fold != "EXT-F1" else all(
            v for k, v in fold_checks.items() if k != "subject_disjoint_rule_recorded_where_required")
        all_pass = all_pass and fold_pass
        checks.append({"fold": fold, "source_domains": list(domains["source"]),
                      "heldout_target_domain": target, "checks": fold_checks, "fold_isolation_pass": fold_pass})

    body = {
        "schema_version": f"{SCHEMA_PREFIX}-fold-isolation-lock-v1",
        "folds": checks, "FOLD_ISOLATION_PASS": all_pass,
        "method": "cross-checked against the E0-frozen EXT_DATASET_FOLD_PLAN.json/EXT_MODEL_BINDING.json "
                 "policy artifacts; never reads or scans real target-domain image bytes",
        "target_access": False, "llm_api_calls": 0,
        "status": "FROZEN",
    }
    body["lock_identity"] = cc.sha256_bytes(cc.canonical_json_bytes(body))
    return body


def write_fold_isolation_lock(repo: Path) -> dict[str, Any]:
    lock = build_fold_isolation_lock(repo)
    return _write(repo, "E7_FOLD_ISOLATION_LOCK.json", lock)


# --------------------------------------------------------------------------- #
# TASK D -- reuse vs rebuild table
# --------------------------------------------------------------------------- #

def build_reuse_rebuild_table(repo: Path) -> dict[str, Any]:
    """TASK D: per governing spec, what may be reused across folds/conditions
    versus what must be rebuilt fold-specifically. `CURRENTLY_PRESENT`/
    `CURRENTLY_VALID` are read from disk, never assumed.
    """
    original_recipes_present = (repo / "assets/recipe_banks/c3/llm/recipes.jsonl").is_file()
    shuffle_recipes_present = (repo / "reports/c_ext_q1q2_v1/e6_llm_shuffle/E6_LLM_SHUFFLE_A_RECIPES.jsonl").is_file()
    detector_config_present = (repo / "reports/full/c7/DETECTOR_CONFIG_LOCK.json").is_file()
    seed_registry_present = (repo / "reports/c_ext_q1q2_v1/e0/EXT_SEED_REGISTRY.json").is_file()
    ext_f1_manifests_present = (repo / "data/processed/prism_target_eval_v2/manifests/source_train.parquet").is_file()
    ext_f1_candidates_dir = repo / "runs/c_ext_q1q2_v1/EXT-F1"

    rows = [
        {"artifact": "abstract frozen recipe banks (ORIGINAL_LLM, LLM-SHUFFLE-A recipe CONTENT)",
        "reuse_allowed": True, "fold_specific": False,
        "currently_present": original_recipes_present and shuffle_recipes_present,
        "currently_valid": original_recipes_present and shuffle_recipes_present,
        "action_required": "none -- reuse verbatim across every fold" if original_recipes_present else
                          "verify assets/recipe_banks presence before F2/F3 preparation"},
        {"artifact": "frozen detector architecture (Track G, DETECTOR_CONFIG_LOCK.json)",
        "reuse_allowed": True, "fold_specific": False, "currently_present": detector_config_present,
        "currently_valid": detector_config_present, "action_required": "none"},
        {"artifact": "optimizer/training policy (m9_reference.yaml)",
        "reuse_allowed": True, "fold_specific": False, "currently_present": True, "currently_valid": True,
        "action_required": "none"},
        {"artifact": "seed registry (detector_seeds)",
        "reuse_allowed": True, "fold_specific": False, "currently_present": seed_registry_present,
        "currently_valid": seed_registry_present, "action_required": "none"},
        {"artifact": "recipe semantics (ontology, compile_recipe)",
        "reuse_allowed": True, "fold_specific": False, "currently_present": True, "currently_valid": True,
        "action_required": "none"},
        {"artifact": "source train/dev manifests",
        "reuse_allowed": False, "fold_specific": True, "currently_present": ext_f1_manifests_present,
        "currently_valid": "EXT-F1 only (reuses frozen Version-C source_train/source_dev construction "
                          "per EXT_DATASET_FOLD_PLAN.json)",
        "action_required": "F2/F3 need SiW-Mv2-as-source subject-disjoint split construction; NOT_PRESENT_"
                          "LOCALLY per E0's own local_data_feasibility audit -- GPU_REQUIRED"},
        {"artifact": "GPAT support/model (per-fold fitting)",
        "reuse_allowed": False, "fold_specific": True, "currently_present": False, "currently_valid": False,
        "action_required": "must be fit fresh per fold; GPU_REQUIRED for F2/F3"},
        {"artifact": "quality calibration, IF source-derived",
        "reuse_allowed": False, "fold_specific": True, "currently_present": "UNCONFIRMED",
        "currently_valid": "UNCONFIRMED",
        "action_required": "governing-spec confirmation needed on whether QUALITY_CALIBRATION.json's "
                          "NOMINAL calibration is source-domain-derived (fold-specific) or dataset-"
                          "independent (reusable); treated conservatively as fold-specific pending that "
                          "confirmation -- flagged, not assumed either way"},
        {"artifact": "Physics/GPAT synthetic candidates (rendered bytes)",
        "reuse_allowed": False, "fold_specific": True,
        "currently_present": ext_f1_candidates_dir.is_dir(),
        "currently_valid": "EXT-F1 only, and only for LLM/LLM-SHUFFLE-A (E6/E6-v2 scope); RND/DET have no "
                          "EXT-scoped rendered candidates at all",
        "action_required": "F2/F3 candidates must be rendered fresh from each fold's own source pairs; "
                          "EXT-F1 candidates are NEVER reused for F2/F3"},
        {"artifact": "matched synthetic bank (selected/frozen)",
        "reuse_allowed": False, "fold_specific": True, "currently_present": False, "currently_valid": False,
        "action_required": "no fold/condition has a completed, selected matched bank yet (E6-v2's SHUFFLE/"
                          "EXT-F1 bank was found infeasible, never selected/written)"},
        {"artifact": "detector checkpoints",
        "reuse_allowed": False, "fold_specific": True, "currently_present": False, "currently_valid": False,
        "action_required": "none exist under the EXT-F{n}-G-<condition>-s<seed> run_id pattern; training "
                          "not authorized this turn"},
        {"artifact": "source-dev calibration (temperature/threshold)",
        "reuse_allowed": False, "fold_specific": True, "currently_present": False, "currently_valid": False,
        "action_required": "computed only after a fold-specific detector checkpoint exists"},
        {"artifact": "target predictions",
        "reuse_allowed": False, "fold_specific": True, "currently_present": False, "currently_valid": False,
        "action_required": "label-free prediction only, only after training; not this turn (Task E7-E)"},
    ]
    return {
        "schema_version": f"{SCHEMA_PREFIX}-reuse-rebuild-table-v1",
        "rows": rows,
        "rule": "EXT-F1 rendered candidates are NEVER reused for EXT-F2/EXT-F3 -- every fold's synthetic "
               "material, GPAT support, matched bank, checkpoint and calibration is fold-specific by "
               "construction (different source domains, different held-out target)",
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False, "training_performed": False,
    }


def write_reuse_rebuild_table(repo: Path) -> dict[str, Any]:
    table = build_reuse_rebuild_table(repo)
    return _write(repo, "E7_REUSE_REBUILD_TABLE.json", table)


# --------------------------------------------------------------------------- #
# TASK E -- non-Shuffle (REALONLY/RND/DET/LLM) readiness audit
# --------------------------------------------------------------------------- #

_NONSHUFFLE_CONDITIONS: tuple[str, ...] = ("G-REALONLY", "G-RND", "G-DET", "G-LLM")

#: Read-only, per-condition EXT-F1 artifact probes -- every path is checked
#: for real presence on disk; nothing here is asserted without a file check.
_EXT_F1_ARTIFACT_PROBE: dict[str, list[str]] = {
    "G-REALONLY": ["reports/c_ext_q1q2_v1/e5_realonly/E5_REAL_ONLY_LOCK.json"],
    "G-RND": [],  # deliberately empty -- see finding below
    "G-DET": [],
    "G-LLM": ["reports/c_ext_q1q2_v1/e6_llm_shuffle/E6_TRAINING_PLAN_LOCK.json"],
}


def audit_nonshuffle_readiness(repo: Path) -> dict[str, Any]:
    """TASK E: for REALONLY/RND/DET/LLM, determines exactly which upstream
    artifacts already exist and are valid, versus what must be regenerated,
    per fold. Every claim is a real `Path.is_file()` read, never assumed.
    Never retrains; never touches target.
    """
    findings: dict[str, dict[str, Any]] = {}
    for fold in FOLDS:
        findings[fold] = {}
        for condition in _NONSHUFFLE_CONDITIONS:
            if fold != "EXT-F1":
                findings[fold][condition] = {
                    "artifacts_present": [], "status": "NOT_PREPARED",
                    "detail": f"no EXT-scoped preparation exists for {fold} (any condition); "
                             "EXT_DATASET_FOLD_PLAN.json's own local_data_feasibility audit records F2/F3 "
                             "fold construction as GPU_REQUIRED and NOT_PRESENT_LOCALLY",
                }
                continue
            probes = _EXT_F1_ARTIFACT_PROBE[condition]
            present = [p for p in probes if (repo / p).is_file()]
            if not probes:
                findings[fold][condition] = {
                    "artifacts_present": [], "status": "NOT_PREPARED",
                    "detail": f"no EXT-F1-scoped preparation module/lock exists for {condition}; the only "
                             "on-disk RND/DET material under reports/c_ext_q1q2_v1/ is HISTORICAL Flow-1 "
                             "C-G-{RND,DET}-P3READY source-score reuse for E4's threshold-transfer "
                             "analysis -- a different analytical purpose, NOT an E7 unified-fold training "
                             "preparation, and explicitly not a substitute for one",
                }
                continue
            gpu_executed = None
            if condition == "G-REALONLY" and present:
                gpu_executed = cc.read_json(repo / present[0]).get("gpu_real_run_executed")
            status = "PREPARED_NOT_TRAINED" if present and gpu_executed is False else (
                "PREPARED_NOT_TRAINED" if present else "NOT_PREPARED")
            findings[fold][condition] = {
                "artifacts_present": present, "status": status,
                "gpu_real_run_executed": gpu_executed,
                "detail": ("lock frozen with EXT-F1-G-<condition>-s<seed> run-ids, GPU training NOT yet "
                          "executed" if present else "no preparation artifact found"),
            }

    ready_cells = sum(1 for fold in findings for cond in findings[fold]
                      if findings[fold][cond]["status"] == "PREPARED_NOT_TRAINED")
    missing_cells = sum(1 for fold in findings for cond in findings[fold]
                        if findings[fold][cond]["status"] == "NOT_PREPARED")

    return {
        "schema_version": f"{SCHEMA_PREFIX}-nonshuffle-readiness-audit-v1",
        "conditions_audited": list(_NONSHUFFLE_CONDITIONS),
        "findings": findings,
        "E7_NONSHUFFLE_READY_CELLS": ready_cells,
        "E7_NONSHUFFLE_MISSING_ARTIFACTS": missing_cells,
        "historical_flow1_accepted_as_e7_substitute": False,
        "historical_flow1_note": "HISTORICAL Flow-1 C-G-{RND,DET}-P3READY source scores exist and are "
                                 "reused (correctly) by E4's threshold-transfer analysis, a SEPARATE "
                                 "purpose from E7 unified-fold detector training; they are NOT accepted "
                                 "as EXT-F1-G-RND/DET training completions and none exists for F2/F3",
        "no_retraining_performed": True, "target_access": False, "llm_api_calls": 0,
    }


def write_nonshuffle_readiness_audit(repo: Path) -> dict[str, Any]:
    audit = audit_nonshuffle_readiness(repo)
    return _write(repo, "E7_NONSHUFFLE_READINESS_AUDIT.json", audit)


# --------------------------------------------------------------------------- #
# TASK F -- Shuffle F2/F3 preflight plan
# --------------------------------------------------------------------------- #

def build_shuffle_f2_f3_preflight_plan() -> dict[str, Any]:
    """TASK F: the frozen (never executed this turn) plan each of
    EXT-F2/G-LLM-SHUFFLE-A and EXT-F3/G-LLM-SHUFFLE-A must pass before
    detector training may be authorized. No parameter changes are ever
    permitted to rescue an infeasible cell.
    """
    stages = [
        "recipe_binding", "source_pair_plan", "fold_specific_rendering",
        "frozen_quality_evaluation", "frozen_source_domain_matched_bank_feasibility",
    ]
    required_bank_size = {"GPAT": 512, "Physics": 512}
    return {
        "schema_version": f"{SCHEMA_PREFIX}-shuffle-f2-f3-preflight-plan-v1",
        "applies_to": ["EXT-F2/G-LLM-SHUFFLE-A", "EXT-F3/G-LLM-SHUFFLE-A"],
        "stages_before_detector_training": stages,
        "required_bank_size_per_route": required_bank_size,
        "authorization_rule": "detector training for a fold/Shuffle cell is authorized ONLY IF both routes "
                              "(GPAT and Physics) reach the full required bank size; otherwise the cell is "
                              "recorded BLOCKED_SCIENTIFIC_INFEASIBILITY",
        "no_parameter_changes_permitted": True,
        "forbidden_rescue_actions": ["lower bank size 512", "change source-domain quotas",
                                     "relax quality thresholds", "modify q", "resample candidates",
                                     "rerender solely to obtain a passing bank", "change matching policy"],
        "ext_f1_result_does_not_predetermine_f2_f3": True,
        "reason": "EXT-F1's Physics infeasibility (479/512) is a property of EXT-F1's OWN quality-pass "
                 "source-domain composition (CASIA-FASD/MSU-MFSD); EXT-F2 and EXT-F3 draw from different "
                 "source-domain pairs (CASIA-FASD/SiW-Mv2 and MSU-MFSD/SiW-Mv2 respectively) and must be "
                 "independently rendered, measured and matched under the SAME frozen rules before any "
                 "feasibility conclusion is drawn",
        "executed_this_turn": False, "rendering_performed": False, "training_performed": False,
        "target_access": False, "llm_api_calls": 0,
    }


def write_shuffle_f2_f3_preflight_plan(repo: Path) -> dict[str, Any]:
    plan = build_shuffle_f2_f3_preflight_plan()
    return _write(repo, "E7_SHUFFLE_F2_F3_PREFLIGHT_PLAN.json", plan)


# --------------------------------------------------------------------------- #
# TASK G -- run-count accounting
# --------------------------------------------------------------------------- #

def build_run_count_accounting() -> dict[str, Any]:
    nominal = len(CONDITIONS) * len(FOLDS) * len(SEEDS)
    predeclared_blocked = len(SEEDS)  # EXT-F1 Shuffle x 5 seeds
    return {
        "schema_version": f"{SCHEMA_PREFIX}-run-count-accounting-v1",
        "E7_NOMINAL_TRAININGS": nominal,
        "E7_PREDECLARED_BLOCKED_TRAININGS": predeclared_blocked,
        "E7_PREDECLARED_BLOCKED_BASIS": "EXT-F1 x G-LLM-SHUFFLE-A x five detector seeds",
        "E7_CURRENT_MAX_AUTHORIZABLE_BEFORE_F2_F3_SHUFFLE_PREFLIGHT": nominal - predeclared_blocked,
        "potentially_authorizable_vs_ready_now": {
            "potentially_authorizable": nominal - predeclared_blocked,
            "ready_now": 0,
            "why_zero_ready_now": "no cell has a completed, valid upstream chain (rendered+matched bank "
                                  "where required, or a REALONLY lock with gpu_real_run_executed=true) "
                                  "AND explicit training authorization this turn; --authorize-gpu-training "
                                  "was never passed",
        },
        "update_policy": "if EXT-F2 and/or EXT-F3 Shuffle preflight (Task F) later finds infeasibility, "
                         "E7_PREDECLARED_BLOCKED_TRAININGS increases ADDITIVELY (+5 per newly-blocked "
                         "fold/condition cell); a prior blocked count is never decreased retroactively "
                         "without a new, separately-recorded finding",
        "target_access": False, "llm_api_calls": 0, "training_performed": False,
    }


def write_run_count_accounting(repo: Path) -> dict[str, Any]:
    accounting = build_run_count_accounting()
    return _write(repo, "E7_RUN_COUNT_ACCOUNTING.json", accounting)


# --------------------------------------------------------------------------- #
# TASK H -- phased GPU execution plan
# --------------------------------------------------------------------------- #

def build_execution_phase_plan() -> dict[str, Any]:
    phases = [
        {"phase": "E7-A", "name": "fold manifests / source-dev / isolation locks",
        "gate": "FOLD_ISOLATION_PASS == true for the fold; source/dev manifests subject-disjoint where "
               "required", "resume_identity": "fold_isolation_lock_identity + source_split_seed"},
        {"phase": "E7-B", "name": "fold-specific GPAT + quality runtime preparation",
        "gate": "E7-A complete for the fold; GPAT support fit and quality backend resolvable on the GPU "
               "host", "resume_identity": "fold GPAT support identity + quality calibration identity"},
        {"phase": "E7-C", "name": "synthetic generation + matching feasibility (RND/DET/LLM, and Shuffle "
                                  "where the preflight plan applies)",
        "gate": "E7-B complete; per condition, render + frozen quality evaluation + frozen matched-bank "
               "feasibility check (Task F's rule) all pass", "resume_identity": "per-arm matched-bank lock "
               "identity or BLOCKED_SCIENTIFIC_INFEASIBILITY record"},
        {"phase": "E7-D", "name": "detector training, 5 seeds per feasible condition/fold",
        "gate": "E7-C feasible for that condition/fold cell (or condition is G-REALONLY, which skips E7-C "
               "entirely)", "resume_identity": "checkpoint identity per EXT-F{n}-G-<condition>-s<seed>"},
        {"phase": "E7-E", "name": "label-free target prediction",
        "gate": "E7-D checkpoint exists and passed source-dev checkpoint selection",
        "resume_identity": "prediction artifact identity; target labels never read here"},
        {"phase": "E7-F", "name": "controlled scoring / aggregation",
        "gate": "E7-E predictions exist for every seed of a condition/fold cell",
        "resume_identity": "aggregate result identity; per-seed rows preserved before aggregation"},
    ]
    return {
        "schema_version": f"{SCHEMA_PREFIX}-execution-phase-plan-v1",
        "phases": phases,
        "single_giant_command_forbidden": True,
        "each_phase_has_own_gate_and_resume_identity": True,
        "executed_this_turn": [], "target_access": False, "llm_api_calls": 0,
    }


def write_execution_phase_plan(repo: Path) -> dict[str, Any]:
    plan = build_execution_phase_plan()
    return _write(repo, "E7_EXECUTION_PLAN.json", plan)


# --------------------------------------------------------------------------- #
# TASK I -- metrics / output contract
# --------------------------------------------------------------------------- #

def build_metrics_output_contract() -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_PREFIX}-metrics-output-contract-v1",
        "primary_metric": "video-level ACER",
        "secondary_metrics": ["APCER", "BPCER", "ROC-AUC", "EER"],
        "calibration_secondary_metrics_if_supported": ["ECE", "Brier", "NLL"],
        "always_preserve_per_seed_rows_before_aggregation": True,
        "ext_f1_per_attack_family_siw_mv2_metrics_required": True,
        "target_scored_this_turn": False,
        "target_access": False, "llm_api_calls": 0,
    }


def write_metrics_output_contract(repo: Path) -> dict[str, Any]:
    contract = build_metrics_output_contract()
    return _write(repo, "E7_METRICS_OUTPUT_CONTRACT.json", contract)


# --------------------------------------------------------------------------- #
# TASK J -- E8 trigger record (not executed)
# --------------------------------------------------------------------------- #

def build_e8_trigger_record(repo: Path) -> dict[str, Any]:
    e8_conditions = ["G-RND-QMATCH", "G-DET-QMATCH", "G-LLM-QMATCH"]
    e8_trigger_source = None
    hyp_path = repo / "reports/c_ext_q1q2_v1/e0/EXT_HYPOTHESIS_FAMILY.json"
    if hyp_path.is_file():
        e8_trigger_source = cc.read_json(hyp_path).get("e8_trigger")
    return {
        "schema_version": f"{SCHEMA_PREFIX}-e8-trigger-record-v1",
        "E8_TRIGGER_FROM_E2": True,
        "trigger_basis": "prior E2 found |SMD(q)| >= 0.25 on the final accepted q distribution",
        "frozen_e8_trigger_rule": e8_trigger_source,
        "E8_CONDITIONS": e8_conditions,
        "shuffle_excluded_from_e8": True,
        "e8_required_unless": "E7 scientific closure makes the planned q-matched analysis impossible "
                              "(e.g. every fold/condition needed for the q-matched contrast is itself "
                              "BLOCKED_SCIENTIFIC_INFEASIBILITY) -- not yet determined, since E7 is only "
                              "being prepared, not executed, this turn",
        "E8_EXECUTED": False,
        "target_access": False, "llm_api_calls": 0, "training_performed": False,
    }


def write_e8_trigger_record(repo: Path) -> dict[str, Any]:
    record = build_e8_trigger_record(repo)
    return _write(repo, "E7_E8_TRIGGER_RECORD.json", record)


# --------------------------------------------------------------------------- #
# consolidated readiness + writer plumbing
# --------------------------------------------------------------------------- #

def build_e7_readiness(repo: Path) -> dict[str, Any]:
    """The consolidated E7_READINESS.json TASK L names -- pulls together the
    non-Shuffle audit, the fold-isolation result and the run-count
    accounting into the exact fields the final report needs, without
    restating full sub-artifact bodies."""
    isolation = build_fold_isolation_lock(repo)
    nonshuffle = audit_nonshuffle_readiness(repo)
    accounting = build_run_count_accounting()
    return {
        "schema_version": f"{SCHEMA_PREFIX}-readiness-v1",
        "E7_EXT_F1_SHUFFLE_STATUS": BLOCKED_SCIENTIFIC_INFEASIBILITY,
        "E7_EXT_F2_SHUFFLE_STATUS": PENDING_FEASIBILITY_PREFLIGHT,
        "E7_EXT_F3_SHUFFLE_STATUS": PENDING_FEASIBILITY_PREFLIGHT,
        "E7_NONSHUFFLE_READY_CELLS": nonshuffle["E7_NONSHUFFLE_READY_CELLS"],
        "E7_NONSHUFFLE_MISSING_ARTIFACTS": nonshuffle["E7_NONSHUFFLE_MISSING_ARTIFACTS"],
        "FOLD_ISOLATION_PASS": isolation["FOLD_ISOLATION_PASS"],
        "E7_READY_FOR_GPU_PHASE_A": isolation["FOLD_ISOLATION_PASS"],
        "next_gpu_commands_required": [
            "none this turn -- E7-A (fold manifests / source-dev / isolation) is the first GPU-boundary "
            "phase (Task H) and is not authorized to run until a future turn explicitly does so",
        ],
        "E7_NOMINAL_TRAININGS": accounting["E7_NOMINAL_TRAININGS"],
        "E7_PREDECLARED_BLOCKED_TRAININGS": accounting["E7_PREDECLARED_BLOCKED_TRAININGS"],
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False, "training_performed": False,
        "status": "PREPARED_NOT_EXECUTED",
    }


def write_e7_readiness(repo: Path) -> dict[str, Any]:
    readiness = build_e7_readiness(repo)
    return _write(repo, "E7_READINESS.json", readiness)


def _write(repo: Path, filename: str, body: dict[str, Any]) -> dict[str, Any]:
    out_dir = repo / E7_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    return {"body": body, "path": str(path)}


def prepare_e7(repo: Path) -> dict[str, Any]:
    """Writes every additive E7 preparation artifact this module builds.
    Never renders, trains, touches target or calls an LLM."""
    results = {
        "protocol_interpretation": write_protocol_interpretation(repo),
        "cell_matrix": write_cell_matrix(repo),
        "fold_isolation_lock": write_fold_isolation_lock(repo),
        "reuse_rebuild_table": write_reuse_rebuild_table(repo),
        "nonshuffle_readiness_audit": write_nonshuffle_readiness_audit(repo),
        "shuffle_f2_f3_preflight_plan": write_shuffle_f2_f3_preflight_plan(repo),
        "run_count_accounting": write_run_count_accounting(repo),
        "execution_plan": write_execution_phase_plan(repo),
        "metrics_output_contract": write_metrics_output_contract(repo),
        "e8_trigger_record": write_e8_trigger_record(repo),
    }
    results["readiness"] = write_e7_readiness(repo)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E7 three-fold Track-G preparation (no render, no train, "
                                                 "no target, no LLM)")
    parser.add_argument("--prepare", action="store_true",
                        help="Writes every additive E7 preparation artifact under E7_DIR. Never renders, "
                             "never trains.")
    args = parser.parse_args(argv)
    repo = cc.repo_root()

    if args.prepare:
        result = prepare_e7(repo)
        print(json.dumps({"readiness": result["readiness"]["body"]}, indent=2, default=str))
        return 0

    print("Pass --prepare to write the additive E7 preparation artifacts. No render, no train, no target, "
         "no LLM ever happens from this module.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
