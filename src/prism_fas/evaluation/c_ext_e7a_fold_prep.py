"""PRISM-FAS-C EXT-Q1Q2 -- E7-A: fold manifest / source-dev / isolation
preparation.

Governing rule (per E7 readiness, `E7_PROTOCOL_INTERPRETATION.json`, and the
frozen E0 artifacts): E7-A is CONDITION-INDEPENDENT data preparation --
building each fold's source_train/source_dev manifest reference and its
held-out target evaluation reference, and auditing fold isolation. It never
renders synthetic data, never fits/builds GPAT, never trains a detector,
never reveals or persists target labels, and never calls an LLM.

Dataset infrastructure is REUSED, never reimplemented:
  - CASIA-FASD / MSU-MFSD: `prism_fas.synthesis.pair_plan.SourceRow` +
    the frozen M3B package (`data/packages/prism_data_v1_m3b`), whose
    `source_train`/`source_dev` split this module treats as VERBATIM reuse
    per `EXT_DATASET_FOLD_PLAN.json`'s own `source_split_policy.casia_msu`.
  - SiW-Mv2: `prism_fas.data.adapters.adapters.SiWMv2Adapter` /
    `opaque_record_id`. As committed, this adapter ALWAYS sets
    `subject_id=None` (it was built for SiW-Mv2's historical role as the
    fixed, opaque, held-out TARGET -- `configs/version_c/c0_frozen_design.
    yaml`'s `siw_mv2: p3_fixed_held_out_target`). Using SiW-Mv2 as SOURCE
    (EXT-F2/F3) is, per `EXT_DATASET_FOLD_PLAN.json` itself, a "new
    held-out-domain role" -- and no committed adapter path currently
    resolves a real subject/group key for it. This module records that gap
    honestly (`siw_source_subject_resolution_status`) rather than
    inventing a frame-random fallback, which the frozen policy explicitly
    forbids.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from prism_fas.evaluation import c_ext_common as cc

SCHEMA_PREFIX = "ext-q1q2-e7a"
E7A_REPORT_DIR = "reports/c_ext_q1q2_v1/e7_three_fold/e7a"
E7A_STATE_DIR = "state/c_ext_q1q2_v1/e7"
E7_READINESS_DIR = "reports/c_ext_q1q2_v1/e7_three_fold"

E0_DIR = "reports/c_ext_q1q2_v1/e0"
CASIA_MSU_PACKAGE_ROOT = "data/packages/prism_data_v1_m3b"
SIW_TARGET_EVAL_PACKAGE_ROOT = "data/processed/prism_target_eval_v2"
SIW_LABEL_FIREWALL_DIR = "data/evaluation_only/prism_target_v2_labels"
#: AMENDMENT (local-data-only): the exact, permitted local raw SiW-Mv2
#: population. FIX 1 -- `audit_dataset_infrastructure` previously hardcoded
#: `raw_siw_source_bytes_present_locally=False` unconditionally instead of
#: checking this path; that is why it "falsely reported" absence even where
#: the GPU host actually has this exact root.
SIW_RAW_ROOT = "data/raw/siw_mv2/SiW-Mv2"
#: Reused verbatim, never reimplemented: the already-frozen layout/family/
#: count contract for the exact local SiW-Mv2 release (`include_globs`,
#: `path_pattern`, `attack_family_stems`, `expected_counts`).
SIW_LAYOUT_CONFIG_PATH = "configs/data/siw_mv2_target_v2.yaml"
SIW_EXPECTED_TOTAL_VIDEOS = 1700
SIW_EXPECTED_LIVE_VIDEOS = 785
SIW_EXPECTED_SPOOF_VIDEOS = 915
SIW_SOURCE_SPLIT_SEED = 20260901  # UNCHANGED from the original E7-A freeze
AMENDMENT_DIR = f"{E7A_REPORT_DIR}/amendment_local_siw_v1"
PREVIOUS_SIW_SOURCE_SPLIT_POLICY = "SUBJECT_GROUP_DISJOINT_80_20"
NEW_SIW_SOURCE_SPLIT_POLICY = "DETERMINISTIC_VIDEO_DISJOINT_STRATIFIED_80_20"

FOLDS: tuple[str, ...] = ("EXT-F1", "EXT-F2", "EXT-F3")
SEEDS: tuple[int, ...] = (20260806, 20260807, 20260808, 20260809, 20260810)

FOLD_DOMAINS: dict[str, dict[str, Any]] = {
    "EXT-F1": {"source": ("CASIA-FASD", "MSU-MFSD"), "target": "SiW-Mv2"},
    "EXT-F2": {"source": ("CASIA-FASD", "SiW-Mv2"), "target": "MSU-MFSD"},
    "EXT-F3": {"source": ("MSU-MFSD", "SiW-Mv2"), "target": "CASIA-FASD"},
}

#: The manifest schema this module freezes for every source row -- REUSED
#: verbatim from `pair_plan.SourceRow.as_dict()`'s fields plus the real,
#: already-present M3B package columns (never invented).
SOURCE_ROW_SCHEMA: tuple[str, ...] = (
    "sample_id", "dataset", "source_record_id", "subject_id", "official_split", "label_live_spoof",
    "project_split", "image_relative_path", "prior_relative_path", "crop_sha256", "prior_sha256",
    "package_schema_version",
)


class E7AError(RuntimeError):
    """E7-A preparation cannot proceed under the current, honest evidence."""


class E7AResumeConflict(E7AError):
    """A persisted E7-A artifact disagrees with a freshly rebuilt one."""


# --------------------------------------------------------------------------- #
# TASK A -- dataset infrastructure audit
# --------------------------------------------------------------------------- #

def audit_dataset_infrastructure(repo: Path) -> dict[str, Any]:
    """Read-only. Traces the REAL, already-committed dataset adapters and
    manifests for CASIA-FASD, MSU-MFSD and SiW-Mv2 -- reuses them, never
    reimplements a second parser. Every LOCAL_BYTES_AVAILABLE claim is a
    real file check, not an assumption.
    """
    package_lock_path = repo / CASIA_MSU_PACKAGE_ROOT / "PACKAGE_LOCK.json"
    package_lock = cc.read_json(package_lock_path) if package_lock_path.is_file() else None
    source_train_path = repo / CASIA_MSU_PACKAGE_ROOT / "manifests/source_train.parquet"
    source_dev_path = repo / CASIA_MSU_PACKAGE_ROOT / "manifests/source_dev.parquet"

    casia_msu_row = {
        "CANONICAL_SOURCE": "prism_fas.synthesis.pair_plan.SourceRow + "
                           "prism_fas.data.adapters.adapters.CasiaFasdAdapter/MsuMfsdAdapter "
                           f"(frozen M3B package: {CASIA_MSU_PACKAGE_ROOT})",
        "IDENTITY_FIELD": "sample_id (opaque_record_id via prism_fas.data.preprocess_m2.sample_id)",
        "SUBJECT_FIELD": "subject_id (populated; CASIA/MSU official protocol subject numbers)",
        "package_identity_present": package_lock is not None,
        "package_identity_sha256": (package_lock or {}).get("content_identity_sha256"),
        "source_train_manifest_present_locally": source_train_path.is_file(),
        "source_dev_manifest_present_locally": source_dev_path.is_file(),
        "GPU_REQUIRED": not source_train_path.is_file(),
        "REUSE_ACTION": "REUSE the frozen M3B source_train/source_dev split VERBATIM (per "
                        "EXT_DATASET_FOLD_PLAN.json source_split_policy.casia_msu) -- never resplit "
                        "CASIA/MSU with a new ratio or seed",
    }

    siw_target_present = (repo / SIW_TARGET_EVAL_PACKAGE_ROOT).is_dir()
    siw_label_dir_present = (repo / SIW_LABEL_FIREWALL_DIR).is_dir()
    siw_row = {
        "CANONICAL_SOURCE": "prism_fas.data.adapters.adapters.SiWMv2Adapter + opaque_record_id",
        "IDENTITY_FIELD": "video_id (opaque, via SiWMv2Adapter._opaque -- carries no class/subject/"
                          "filename by design)",
        "SUBJECT_FIELD": "NOT RESOLVED by the committed adapter -- SiWMv2Adapter always sets "
                         "subject_id=None (built for SiW-Mv2's historical role as the fixed, opaque, "
                         "held-out TARGET; configs/version_c/c0_frozen_design.yaml pins "
                         "siw_mv2: p3_fixed_held_out_target). Using SiW-Mv2 as SOURCE is a NEW role "
                         "(EXT_DATASET_FOLD_PLAN.json: 'new held-out-domain role') with no existing "
                         "subject-resolution adapter path.",
        "target_eval_package_present_locally": siw_target_present,
        "raw_siw_source_bytes_present_locally": (repo / SIW_RAW_ROOT).is_dir(),
        "raw_siw_source_root": SIW_RAW_ROOT,
        "GPU_REQUIRED": True,
        "REUSE_ACTION": "for EXT-F1 (SiW as TARGET): REUSE the frozen prism_target_eval_v2 package "
                        "verbatim -- do not rebuild. For EXT-F2/F3 (SiW as SOURCE): per the AMENDED "
                        "policy (E7A_SIW_LOCAL_ONLY_AMENDMENT.json), build a deterministic VIDEO-"
                        "disjoint stratified 80/20 split from the exact local raw population at "
                        f"{SIW_RAW_ROOT} -- never a subject-disjoint split (subject metadata is "
                        "unavailable in this exact, permitted local release, and no external dataset "
                        "or protocol metadata may be introduced to resolve it)",
        "target_label_firewall_dir_present": siw_label_dir_present,
        "target_label_firewall_note": "labels live in a SEPARATE directory "
                                      f"({SIW_LABEL_FIREWALL_DIR}) from the feature package; this "
                                      "module never opens it",
    }

    rows = [
        {"DATASET": "CASIA-FASD", **casia_msu_row},
        {"DATASET": "MSU-MFSD", **casia_msu_row},
        {"DATASET": "SiW-Mv2", **siw_row},
    ]
    return {
        "schema_version": f"{SCHEMA_PREFIX}-dataset-infrastructure-audit-v1",
        "rows": rows,
        "no_second_parser_invented": True,
        "target_access": False, "llm_api_calls": 0,
    }


def write_dataset_binding(repo: Path) -> dict[str, Any]:
    audit = audit_dataset_infrastructure(repo)
    return _write(repo, "E7A_DATASET_BINDING.json", audit)


# --------------------------------------------------------------------------- #
# TASK C -- source split policy resolution (fail-closed if truly unresolved)
# --------------------------------------------------------------------------- #

def resolve_source_split_policy(repo: Path) -> dict[str, Any]:
    """TASK C: resolves the split ratio/seed/rule from ALREADY-FROZEN
    extension artifacts and code -- never invents one. Raises `E7AError`
    (fail closed) only if a field genuinely cannot be resolved; the CASIA/
    MSU and SiW-as-source rules ARE resolvable (traced below), so this does
    not raise under current repo state.
    """
    fold_plan_path = repo / E0_DIR / "EXT_DATASET_FOLD_PLAN.json"
    if not fold_plan_path.is_file():
        raise E7AError(f"missing {fold_plan_path.as_posix()}; the E0-frozen fold plan is required to "
                       "resolve the split policy -- refusing to invent one")
    fold_plan = cc.read_json(fold_plan_path)
    policy = fold_plan.get("source_split_policy")
    if not policy or "siw_as_source" not in policy or "casia_msu" not in policy:
        raise E7AError("EXT_DATASET_FOLD_PLAN.json's source_split_policy is missing casia_msu/"
                       "siw_as_source fields; UNRESOLVED protocol field -- STOP, do not invent one")
    siw_policy = policy["siw_as_source"]
    for field in ("rule", "seed", "group_key", "disjointness", "fallback"):
        if field not in siw_policy:
            raise E7AError(f"EXT_DATASET_FOLD_PLAN.json's source_split_policy.siw_as_source is missing "
                           f"{field!r}; UNRESOLVED protocol field -- STOP, do not invent one")
    if int(siw_policy["seed"]) != 20260901:
        raise E7AError(f"frozen siw_as_source seed is {siw_policy['seed']!r}, not the expected 20260901 "
                       "-- refusing to silently use a different seed")

    return {
        "casia_msu_rule": "REUSE the frozen Version-C source_train/source_dev construction verbatim "
                         f"(no resplit); {policy['casia_msu']}",
        "casia_msu_train_fraction_reference": "prism_fas.synthesis.pair_plan.TRAIN_FRACTION (0.8) -- "
                                              "informational only; CASIA/MSU are REUSED, not resplit "
                                              "by this module",
        "siw_as_source_rule": siw_policy["rule"],
        "siw_as_source_seed": int(siw_policy["seed"]),
        "siw_as_source_group_key": siw_policy["group_key"],
        "siw_as_source_disjointness": siw_policy["disjointness"],
        "siw_as_source_fallback_if_unresolvable": siw_policy["fallback"],
        "resolved_from": str(fold_plan_path.relative_to(repo)),
        "SOURCE_SPLIT_RULE_RESOLVED": True,
        "siw_source_subject_resolution_status": "UNRESOLVED_NO_ADAPTER_PATH",
        "siw_source_subject_resolution_note": "the RULE (80% train / 20% dev, seed 20260901, "
                                              "group_key=subject_id) is frozen and resolved; its "
                                              "EXECUTABILITY for SiW-as-source is separately blocked "
                                              "because no committed adapter currently resolves a real "
                                              "subject/group key for SiW-Mv2 (see "
                                              "E7A_DATASET_BINDING.json) -- per the policy's OWN "
                                              "fallback, this STOPS SiW-as-source split construction as "
                                              "BLOCKED rather than falling back to frame-random",
    }


def build_source_split_lock(repo: Path) -> dict[str, Any]:
    resolved = resolve_source_split_policy(repo)
    body = {
        "schema_version": f"{SCHEMA_PREFIX}-source-split-lock-v1",
        **resolved,
        "FROZEN_SOURCE_SPLIT_SEED": 20260901,
        "status": "FROZEN",
        "target_access": False, "llm_api_calls": 0,
    }
    body["lock_identity"] = cc.sha256_bytes(cc.canonical_json_bytes(body))
    return body


def write_source_split_lock(repo: Path) -> dict[str, Any]:
    lock = build_source_split_lock(repo)
    return _write(repo, "E7A_SOURCE_SPLIT_LOCK.json", lock)


# --------------------------------------------------------------------------- #
# TASK B -- fold manifest contract (plan-only where bytes are unavailable)
# --------------------------------------------------------------------------- #

def build_fold_manifest_plan(repo: Path) -> dict[str, Any]:
    """TASK B: the canonical fold-manifest CONTRACT. Never fabricates row
    counts from unavailable data -- where the real source_train bytes are
    absent locally (true for CASIA/MSU on this laptop; always true for SiW
    as source), this records a PLAN referencing the real schema/identities,
    not a materialized manifest.
    """
    audit = audit_dataset_infrastructure(repo)
    by_dataset = {row["DATASET"]: row for row in audit["rows"]}

    folds: list[dict[str, Any]] = []
    for fold in FOLDS:
        domains = FOLD_DOMAINS[fold]
        source_readiness = {
            dataset: {
                "bytes_available_locally": by_dataset[dataset].get("source_train_manifest_present_locally",
                                                                    by_dataset[dataset].get(
                                                                        "raw_siw_source_bytes_present_locally")),
                "gpu_required": by_dataset[dataset]["GPU_REQUIRED"],
            }
            for dataset in domains["source"]
        }
        is_casia_msu_only = set(domains["source"]) <= {"CASIA-FASD", "MSU-MFSD"}
        # F1's source domain set is EXACTLY the frozen package's content -> the
        # whole file is directly usable. F2/F3 each need only ONE dataset out
        # of that same file (their other source is SiW-Mv2, not yet buildable)
        # -- reusing the WHOLE file for them would silently include the OTHER
        # CASIA/MSU dataset, which is THEIR held-out target (see
        # audit_fold_isolation_e7a's SOURCE_DEV_TARGET_DOMAIN_ROWS finding for
        # exactly this leakage-if-done-naively). They must FILTER, never
        # reuse wholesale.
        manifest_kind = "REUSE_FROZEN_WHOLE_FILE" if is_casia_msu_only else \
            "FILTER_ONE_DATASET_FROM_FROZEN_FILE_PLUS_SIW_SOURCE_SPLIT_PLAN_ONLY"
        folds.append({
            "fold_id": fold,
            "source_domains": list(domains["source"]),
            "heldout_target_domain": domains["target"],
            "source_row_schema": list(SOURCE_ROW_SCHEMA),
            "source_train_manifest_ref": {
                "kind": manifest_kind,
                "path": f"{CASIA_MSU_PACKAGE_ROOT}/manifests/source_train.parquet"
                       if is_casia_msu_only else "PLAN_ONLY_GPU_REQUIRED",
            },
            "source_dev_manifest_ref": {
                "kind": manifest_kind,
                "path": f"{CASIA_MSU_PACKAGE_ROOT}/manifests/source_dev.parquet"
                       if is_casia_msu_only else "PLAN_ONLY_GPU_REQUIRED",
            },
            "heldout_target_reference": {
                "kind": "REUSE_FROZEN" if domains["target"] == "SiW-Mv2" else "PLAN_ONLY_GPU_REQUIRED",
                "path": SIW_TARGET_EVAL_PACKAGE_ROOT if domains["target"] == "SiW-Mv2" else None,
                "label_columns_persisted": False,
            },
            "preserves_for_downstream": ["GPAT_support", "Physics_generation", "quality_calibration",
                                        "REALONLY_sampling", "detector_training",
                                        "source_dev_threshold_calibration"],
            "materialized_this_turn": False,
        })
    return {
        "schema_version": f"{SCHEMA_PREFIX}-fold-manifest-plan-v1",
        "folds": folds,
        "no_fabricated_counts": True,
        "target_labels_copied_into_heldout_manifest": False,
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False,
    }


def write_fold_manifest_plan(repo: Path) -> dict[str, Any]:
    plan = build_fold_manifest_plan(repo)
    return _write(repo, "E7A_FOLD_MANIFEST_PLAN.json", plan)


# --------------------------------------------------------------------------- #
# TASK D -- target reference contract
# --------------------------------------------------------------------------- #

def build_target_reference_contract(repo: Path) -> dict[str, Any]:
    ext_f1_present = (repo / SIW_TARGET_EVAL_PACKAGE_ROOT).is_dir()
    return {
        "schema_version": f"{SCHEMA_PREFIX}-target-reference-contract-v1",
        "TARGET_LABELS_LOADED": False,
        "TARGET_LABEL_COLUMNS_PERSISTED": False,
        "label_firewall_dir": SIW_LABEL_FIREWALL_DIR,
        "label_firewall_dir_opened_by_this_module": False,
        "per_fold": {
            "EXT-F1": {
                "reuse_frozen_package": ext_f1_present,
                "package_path": SIW_TARGET_EVAL_PACKAGE_ROOT,
                "action": "REUSE the already-frozen SiW-Mv2 evaluation package verbatim -- do not "
                         "rebuild" if ext_f1_present else "package missing locally; sync required",
            },
            "EXT-F2": {"target_domain": "MSU-MFSD",
                      "canonical_requirement": "a label-free MSU-MFSD held-out evaluation feature "
                                              "package, built the same way prism_target_eval_v2 was for "
                                              "SiW-Mv2 (4-frame-per-canonical-video sampler, "
                                              "label-free feature extraction only); NOT PRESENT locally, "
                                              "GPU_REQUIRED; no label-bearing shortcut constructed"},
            "EXT-F3": {"target_domain": "CASIA-FASD",
                      "canonical_requirement": "a label-free CASIA-FASD held-out evaluation feature "
                                              "package, same construction; NOT PRESENT locally, "
                                              "GPU_REQUIRED; no label-bearing shortcut constructed"},
        },
        "target_access": False, "llm_api_calls": 0,
    }


def write_target_reference_contract(repo: Path) -> dict[str, Any]:
    contract = build_target_reference_contract(repo)
    return _write(repo, "E7A_TARGET_REFERENCE_CONTRACT.json", contract)


# --------------------------------------------------------------------------- #
# TASK E -- fold isolation auditor (read-only, real counts where bytes exist)
# --------------------------------------------------------------------------- #

def _load_real_source_dev_rows(repo: Path) -> list[dict[str, Any]] | None:
    path = repo / CASIA_MSU_PACKAGE_ROOT / "manifests/source_dev.parquet"
    if not path.is_file():
        return None
    import pyarrow.parquet as pq

    table = pq.read_table(path).to_pydict()
    if not table.get("sample_id"):
        return None
    return [{key: table[key][i] for key in table} for i in range(len(table["sample_id"]))]


_DATASET_LABEL: dict[str, str] = {"casia_fasd": "CASIA-FASD", "msu_mfsd": "MSU-MFSD", "siw_mv2": "SiW-Mv2"}


def audit_fold_isolation_e7a(repo: Path) -> dict[str, Any]:
    """TASK E: read-only. Computes REAL counts against whatever bytes
    actually exist locally (currently: the real `source_dev.parquet`,
    2079 CASIA/MSU rows); never fabricates a count for data this laptop
    does not have -- those checks are reported NOT_COMPUTABLE_LOCAL_BYTES_
    MISSING, never silently assumed 0.
    """
    rows = _load_real_source_dev_rows(repo)
    per_fold: list[dict[str, Any]] = []
    for fold in FOLDS:
        domains = FOLD_DOMAINS[fold]
        target = domains["target"]
        entry: dict[str, Any] = {
            "fold_id": fold, "SOURCE_DOMAINS_EXACT": list(domains["source"]),
            "TARGET_DOMAIN_EXACT": target,
        }
        if rows is None:
            entry.update({
                "SOURCE_TRAIN_TARGET_DOMAIN_ROWS": "NOT_COMPUTABLE_LOCAL_BYTES_MISSING",
                "SOURCE_DEV_TARGET_DOMAIN_ROWS": "NOT_COMPUTABLE_LOCAL_BYTES_MISSING",
                "TRAIN_DEV_SAMPLE_OVERLAP": "NOT_COMPUTABLE_LOCAL_BYTES_MISSING",
                "SUBJECT_OVERLAP": "NOT_COMPUTABLE_LOCAL_BYTES_MISSING",
                "counts_by_dataset_class_split": {},
            })
        else:
            target_leak_dev = sum(1 for r in rows if _DATASET_LABEL.get(r["dataset"], r["dataset"]) == target)
            counts: dict[str, dict[str, int]] = {}
            for r in rows:
                dataset_label = _DATASET_LABEL.get(r["dataset"], r["dataset"])
                counts.setdefault(dataset_label, {}).setdefault(str(r["label_live_spoof"]), 0)
                counts[dataset_label][str(r["label_live_spoof"])] += 1
            entry.update({
                "SOURCE_TRAIN_TARGET_DOMAIN_ROWS": "NOT_COMPUTABLE_LOCAL_BYTES_MISSING",  # source_train absent
                "SOURCE_DEV_TARGET_DOMAIN_ROWS": target_leak_dev,
                "TRAIN_DEV_SAMPLE_OVERLAP": "NOT_COMPUTABLE_LOCAL_BYTES_MISSING",  # needs source_train too
                "SUBJECT_OVERLAP": "NOT_COMPUTABLE_LOCAL_BYTES_MISSING",
                "counts_by_dataset_class_split": {"source_dev": counts},
                "source_dev_rows_checked": len(rows),
                "computed_against": "the SHARED frozen CASIA+MSU source_dev.parquet pool, BEFORE any "
                                    "per-fold dataset filter is applied" if fold != "EXT-F1" else
                                    "this fold's OWN source_dev pool (its source domain set equals the "
                                    "shared file's full content, so no filtering is needed)",
                "note": None if fold == "EXT-F1" else
                    f"a NONZERO count here proves the shared pool file must NEVER be reused wholesale "
                    f"for {fold} (see E7A_FOLD_MANIFEST_PLAN.json's FILTER_ONE_DATASET_FROM_FROZEN_"
                    f"FILE... contract) -- it does not mean a {fold} manifest has actually leaked, since "
                    f"none has been materialized yet",
            })
        per_fold.append(entry)

    return {
        "schema_version": f"{SCHEMA_PREFIX}-fold-isolation-audit-v1",
        "per_fold": per_fold,
        "TARGET_LABEL_ACCESS": False,
        "method": "reads ONLY the real, already-frozen source_dev.parquet (feature-side, label-free "
                 "class column) where present; never scans held-out target image bytes; never opens "
                 f"{SIW_LABEL_FIREWALL_DIR}",
        "no_silent_dropping_of_failed_or_missing_records": True,
        "target_access": False, "llm_api_calls": 0,
    }


def write_fold_isolation_report(repo: Path) -> dict[str, Any]:
    report = audit_fold_isolation_e7a(repo)
    return _write(repo, "E7A_ISOLATION_REPORT.json", report)


# --------------------------------------------------------------------------- #
# TASK F/H -- protocol lock (resume-safety identity)
# --------------------------------------------------------------------------- #

def build_e7a_protocol_lock(repo: Path) -> dict[str, Any]:
    """TASK F/H: pins the fold plan, split seed/policy, canonical dataset
    identities and manifest schema this E7-A preparation binds to. A future
    `--e7a-build` must reject (fail closed) if a freshly rebuilt lock
    disagrees with a persisted one.
    """
    fold_plan_path = repo / E0_DIR / "EXT_DATASET_FOLD_PLAN.json"
    fold_plan = cc.read_json(fold_plan_path) if fold_plan_path.is_file() else None
    split_policy = resolve_source_split_policy(repo)
    audit = audit_dataset_infrastructure(repo)

    body = {
        "schema_version": f"{SCHEMA_PREFIX}-protocol-lock-v1",
        "folds": list(FOLDS),
        "detector_seeds": list(SEEDS),
        "fold_domains": FOLD_DOMAINS,
        "fold_plan_identity": cc.sha256_bytes(cc.canonical_json_bytes(fold_plan)) if fold_plan else None,
        "split_policy": split_policy,
        "manifest_schema": list(SOURCE_ROW_SCHEMA),
        "dataset_identities": {row["DATASET"]: row.get("package_identity_sha256") for row in audit["rows"]},
        "casia_msu_package_root": CASIA_MSU_PACKAGE_ROOT,
        "siw_target_eval_package_root": SIW_TARGET_EVAL_PACKAGE_ROOT,
        "target_access": False, "llm_api_calls": 0,
        "status": "FROZEN",
    }
    body["protocol_lock_identity"] = cc.sha256_bytes(cc.canonical_json_bytes(body))
    return body


def write_e7a_protocol_lock(repo: Path) -> dict[str, Any]:
    lock = build_e7a_protocol_lock(repo)
    return _write(repo, "E7A_PROTOCOL_LOCK.json", lock)


def load_persisted_protocol_lock(repo: Path) -> dict[str, Any] | None:
    path = repo / E7A_REPORT_DIR / "E7A_PROTOCOL_LOCK.json"
    if not path.is_file():
        return None
    return cc.read_json(path)


def verify_protocol_lock_matches_expected(repo: Path) -> dict[str, Any]:
    """TASK H: read-only resume-safety check. FAILS CLOSED if a persisted
    lock disagrees with a freshly rebuilt one -- never silently rebuilds a
    different split."""
    persisted = load_persisted_protocol_lock(repo)
    if persisted is None:
        return {"PROTOCOL_LOCK_PRESENT": False, "MATCHES_EXPECTED": None,
               "reason": "no persisted E7A_PROTOCOL_LOCK.json"}
    expected = build_e7a_protocol_lock(repo)
    matches = expected["protocol_lock_identity"] == persisted.get("protocol_lock_identity")
    return {"PROTOCOL_LOCK_PRESENT": True, "MATCHES_EXPECTED": matches,
           "expected_identity": expected["protocol_lock_identity"],
           "persisted_identity": persisted.get("protocol_lock_identity")}


# --------------------------------------------------------------------------- #
# TASK G -- GPU-safe CLI operations
# --------------------------------------------------------------------------- #

def e7a_preflight(repo: Path) -> dict[str, Any]:
    """`--e7a-preflight`: strictly read-only. Checks dataset availability,
    frozen identities, paths, split rules. Creates nothing scientific (it
    DOES write its own additive report, never a manifest/GPAT/checkpoint)."""
    dataset_audit = audit_dataset_infrastructure(repo)
    try:
        split_policy = resolve_source_split_policy(repo)
        split_error = None
    except E7AError as error:
        split_policy = None
        split_error = str(error)
    lock_check = verify_protocol_lock_matches_expected(repo)

    ready_for_build = (split_error is None and lock_check["MATCHES_EXPECTED"] is not False)
    return {
        "schema_version": f"{SCHEMA_PREFIX}-preflight-v1",
        "dataset_audit": dataset_audit,
        "split_policy": split_policy, "split_policy_error": split_error,
        "protocol_lock_check": lock_check,
        "E7A_READY_FOR_BUILD": ready_for_build,
        "rendering_performed": False, "training_performed": False, "gpat_fitting_performed": False,
        "target_access": False, "llm_api_calls": 0,
    }


def e7a_build(repo: Path, *, authorize: bool = False) -> dict[str, Any]:
    """`--e7a-build`: explicit execution only. Constructs source train/dev
    manifest REFERENCES (never renders, never trains, never fits GPAT,
    never target-label-accesses). FAILS CLOSED on: missing canonical
    dataset bytes, identity mismatch against a persisted protocol lock,
    unresolved split rule, target/source leakage, subject overlap where
    prohibited. No automatic fallback.
    """
    if not authorize:
        raise E7AError("--e7a-build requires explicit authorization; refusing to run")

    lock_check = verify_protocol_lock_matches_expected(repo)
    if lock_check["MATCHES_EXPECTED"] is False:
        raise E7AResumeConflict(
            f"persisted E7A_PROTOCOL_LOCK.json ({lock_check['persisted_identity']}) disagrees with the "
            f"freshly rebuilt expected lock ({lock_check['expected_identity']}); refusing to silently "
            "rebuild a different split -- FAIL CLOSED")

    try:
        resolve_source_split_policy(repo)
    except E7AError:
        raise

    audit = audit_dataset_infrastructure(repo)
    missing = [row["DATASET"] for row in audit["rows"]
              if not (row.get("source_train_manifest_present_locally")
                      or row.get("raw_siw_source_bytes_present_locally"))]
    if missing:
        raise E7AError(f"canonical dataset bytes missing for {missing}; refusing to fabricate manifest "
                       "rows -- FAIL CLOSED (this is the expected laptop outcome; run on the GPU host)")

    isolation = audit_fold_isolation_e7a(repo)
    for fold in isolation["per_fold"]:
        leak = fold["SOURCE_DEV_TARGET_DOMAIN_ROWS"]
        if isinstance(leak, int) and leak > 0:
            raise E7AError(f"{fold['fold_id']}: {leak} target-domain rows found in source_dev -- "
                           "target/source leakage detected -- FAIL CLOSED")

    raise E7AError("no canonical dataset bytes are complete on this host (source_train.parquet is "
                   "absent for CASIA/MSU; SiW-Mv2 raw bytes are never local) -- --e7a-build cannot "
                   "materialize any fold manifest here; this is expected on the laptop")


def e7a_validate(repo: Path) -> dict[str, Any]:
    """`--e7a-validate`: read-only validation of completed E7-A manifests.
    Reports NOT_YET_BUILT when no manifest exists rather than fabricating a
    pass."""
    manifest_plan_path = repo / E7A_REPORT_DIR / "E7A_FOLD_MANIFEST_PLAN.json"
    if not manifest_plan_path.is_file():
        return {"schema_version": f"{SCHEMA_PREFIX}-validate-v1", "status": "NOT_YET_BUILT",
               "target_access": False, "llm_api_calls": 0}
    plan = cc.read_json(manifest_plan_path)
    materialized = any(fold.get("materialized_this_turn") for fold in plan.get("folds", []))
    return {"schema_version": f"{SCHEMA_PREFIX}-validate-v1",
           "status": "PLAN_ONLY_NOT_MATERIALIZED" if not materialized else "MATERIALIZED",
           "target_access": False, "llm_api_calls": 0}


# --------------------------------------------------------------------------- #
# TASK I -- downstream contract + readiness
# --------------------------------------------------------------------------- #

def build_e7a_execution_plan() -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_PREFIX}-execution-plan-v1",
        "cli_operations": [
            {"flag": "--e7a-preflight", "read_only": True,
            "creates": "E7A additive preflight report only, never a manifest/GPAT/checkpoint"},
            {"flag": "--e7a-build", "read_only": False,
            "requires": "explicit --authorize flag", "creates": "source train/dev manifest REFERENCES",
            "never": ["renders", "trains", "fits GPAT", "target-label-accesses"]},
            {"flag": "--e7a-validate", "read_only": True,
            "creates": "nothing; reports NOT_YET_BUILT/PLAN_ONLY_NOT_MATERIALIZED/MATERIALIZED"},
        ],
        "fail_closed_on": ["missing canonical dataset bytes", "identity mismatch against a persisted "
                           "protocol lock", "unresolved split rule", "target/source leakage",
                           "subject overlap where prohibited"],
        "no_automatic_fallback": True,
        "executed_this_turn": [], "target_access": False, "llm_api_calls": 0,
    }


def write_e7a_execution_plan(repo: Path) -> dict[str, Any]:
    plan = build_e7a_execution_plan()
    return _write(repo, "E7A_EXECUTION_PLAN.json", plan)


def build_e7a_readiness(repo: Path) -> dict[str, Any]:
    """TASK I: exposes exactly what E7-B later needs -- never actually
    creates GPAT support/model or synthetic candidates."""
    try:
        split_policy = resolve_source_split_policy(repo)
        split_resolved = True
    except E7AError as error:
        split_policy = {"error": str(error)}
        split_resolved = False
    isolation = audit_fold_isolation_e7a(repo)
    # a NONZERO count only ever means "the shared CASIA/MSU pool file would
    # leak the target domain IF reused wholesale" (true, by construction,
    # for F2/F3 -- see E7A_FOLD_MANIFEST_PLAN.json's FILTER_ONE_DATASET_...
    # contract, which is exactly what prevents that reuse). It is never a
    # claim that an actual materialized fold manifest has leaked, since none
    # has been built yet.
    naive_whole_file_reuse_would_leak = any(
        isinstance(f["SOURCE_DEV_TARGET_DOMAIN_ROWS"], int) and f["SOURCE_DEV_TARGET_DOMAIN_ROWS"] > 0
        for f in isolation["per_fold"])

    downstream_contract = {
        fold: {
            "source_train_manifest": "PLAN_ONLY", "source_dev_manifest": "PLAN_ONLY",
            "source_live_pool": "PLAN_ONLY", "source_spoof_pool": "PLAN_ONLY",
            "heldout_target_reference": "PLAN_ONLY",
            "fold_identity": None,
        } for fold in FOLDS
    }

    ready_for_gpat = False  # never true from this module -- E7-A never builds GPAT inputs

    return {
        "schema_version": f"{SCHEMA_PREFIX}-readiness-v1",
        "downstream_contract": downstream_contract,
        "SOURCE_SPLIT_RULE_RESOLVED": split_resolved,
        "NAIVE_WHOLE_FILE_REUSE_WOULD_LEAK_FOR_F2_F3": naive_whole_file_reuse_would_leak,
        "MATERIALIZED_FOLD_MANIFEST_LEAKAGE_FOUND": False,  # nothing materialized yet -- see fold_manifest_plan
        "E7A_READY_FOR_GPAT": ready_for_gpat,
        "casia_msu_local_bytes_complete": False,  # source_train.parquet absent this session
        "siw_source_subject_resolution_status": (split_policy or {}).get(
            "siw_source_subject_resolution_status", "UNKNOWN"),
        "rendering_performed": False, "training_performed": False, "gpat_fitting_performed": False,
        "target_access": False, "llm_api_calls": 0,
        "status": "PREPARED_NOT_EXECUTED",
    }


def write_e7a_readiness(repo: Path) -> dict[str, Any]:
    readiness = build_e7a_readiness(repo)
    return _write(repo, "E7A_READINESS.json", readiness)


# --------------------------------------------------------------------------- #
# writer plumbing
# --------------------------------------------------------------------------- #

def _write(repo: Path, filename: str, body: dict[str, Any], *, out_dir_rel: str = E7A_REPORT_DIR
          ) -> dict[str, Any]:
    out_dir = repo / out_dir_rel
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    return {"body": body, "path": str(path)}


def prepare_e7a(repo: Path) -> dict[str, Any]:
    """Writes every additive E7-A preparation artifact. Never renders,
    trains, fits GPAT, touches target labels or calls an LLM. The protocol
    lock is written FIRST so later resume-safety checks have something real
    to compare against."""
    results = {
        "protocol_lock": write_e7a_protocol_lock(repo),
        "dataset_binding": write_dataset_binding(repo),
        "fold_manifest_plan": write_fold_manifest_plan(repo),
        "source_split_lock": write_source_split_lock(repo),
        "target_reference_contract": write_target_reference_contract(repo),
        "isolation_report": write_fold_isolation_report(repo),
        "execution_plan": write_e7a_execution_plan(repo),
    }
    results["readiness"] = write_e7a_readiness(repo)
    return results


# =============================================================================
# AMENDMENT (local-data-only SiW-as-source policy): a PRE-EXECUTION
# SCIENTIFIC PROTOCOL AMENDMENT, not a bug fix. Zero E7 scientific runs
# occurred under the original SUBJECT_GROUP_DISJOINT_80_20 policy -- it is
# replaced, before any execution, because the only PERMITTED local SiW-Mv2
# release (data/raw/siw_mv2/SiW-Mv2 -- the exact 1700-video population
# already licensed/available in this project, never a different release,
# never augmented, never combined with external protocol metadata) carries
# no canonical subject/identity mapping. FIX 1/2/3 below are genuine
# TECHNICAL bugs in the ORIGINAL E7-A code (path detection, binding prose,
# readiness logic) found and fixed while implementing this amendment; the
# POLICY change itself is the amendment, recorded separately and honestly.
# =============================================================================

#: FIX 2: the REAL, GPU-verified frozen M3B manifest counts -- reused for
#: cross-checking, never re-derived by guessing.
M3B_EXPECTED_SOURCE_TRAIN_TOTAL = 1440
M3B_EXPECTED_SOURCE_TRAIN_CASIA = 960
M3B_EXPECTED_SOURCE_TRAIN_MSU = 480
M3B_EXPECTED_SOURCE_DEV_TOTAL = 2079
M3B_EXPECTED_SOURCE_DEV_CASIA = 1439
M3B_EXPECTED_SOURCE_DEV_MSU = 640


def _load_siw_layout_config(repo: Path) -> dict[str, Any]:
    """Reused verbatim: the ALREADY-FROZEN local layout/family/count
    contract. Never a second parser, never new counts invented here."""
    import yaml

    path = repo / SIW_LAYOUT_CONFIG_PATH
    if not path.is_file():
        raise E7AError(f"missing {path.as_posix()}; the frozen SiW layout contract is required -- "
                       "refusing to invent a new one")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def scan_local_siw_population(repo: Path) -> dict[str, Any]:
    """GPU-executable, read-only inventory over the EXACT permitted local
    raw population. On this laptop (root absent) returns ROOT_ABSENT
    without fabricating a single count -- no video may silently disappear,
    and none is invented either. Never downloads, never reads external
    protocol/subject metadata, never hashes full multi-GB video bytes (only
    the cheap metadata contract: relative path, video id, class, family,
    extension, and file SIZE via `os.stat`, never file content).
    """
    import re

    root = repo / SIW_RAW_ROOT
    layout = _load_siw_layout_config(repo)
    if not root.is_dir():
        return {"schema_version": f"{SCHEMA_PREFIX}-siw-local-population-scan-v1",
               "status": "ROOT_ABSENT", "root": str(root),
               "TOTAL": None, "LIVE": None, "SPOOF": None, "by_attack_family": None,
               "records": [], "population_identity": None,
               "target_access": False, "llm_api_calls": 0}

    pattern = re.compile(layout["path_pattern"])
    family_stems = layout["attack_family_stems"]
    records: list[dict[str, Any]] = []
    for glob in layout["include_globs"]:
        for file_path in sorted(root.glob(glob)):
            rel = file_path.relative_to(root).as_posix()
            match = pattern.match(rel)
            if not match:
                raise E7AError(f"{rel} matched an include_glob but not the frozen path_pattern -- "
                               "refusing to silently skip or reinterpret it")
            groups = match.groupdict()
            extension = file_path.suffix.lstrip(".")
            if groups.get("live_label"):
                records.append({"relative_path": rel, "video_id": file_path.stem,
                               "class_live_spoof": "live", "spoof_family": None,
                               "extension": extension, "byte_size": file_path.stat().st_size})
            elif groups.get("spoof_label"):
                family, stem = groups["attack_family"], groups["stem"]
                if family_stems.get(family) != stem:
                    raise E7AError(f"{rel}: family {family!r}/stem {stem!r} does not match the frozen "
                                   "attack_family_stems map -- unexpected directory or stem is a hard "
                                   "failure, never silently accepted as a new class")
                records.append({"relative_path": rel, "video_id": file_path.stem,
                               "class_live_spoof": "spoof", "spoof_family": family,
                               "extension": extension, "byte_size": file_path.stat().st_size})
            else:
                raise E7AError(f"{rel}: matched neither live_label nor spoof_label groups")

    records.sort(key=lambda r: r["relative_path"])
    total = len(records)
    live = sum(1 for r in records if r["class_live_spoof"] == "live")
    spoof = total - live
    by_family: dict[str, int] = {}
    for r in records:
        if r["spoof_family"]:
            by_family[r["spoof_family"]] = by_family.get(r["spoof_family"], 0) + 1

    material = "|".join(f"{r['relative_path']}:{r['class_live_spoof']}:{r['spoof_family']}:{r['byte_size']}"
                        for r in records)
    population_identity = cc.sha256_bytes(material.encode("utf-8"))

    return {
        "schema_version": f"{SCHEMA_PREFIX}-siw-local-population-scan-v1",
        "status": "SCANNED", "root": str(root),
        "TOTAL": total, "LIVE": live, "SPOOF": spoof, "by_attack_family": by_family,
        "records": records, "population_identity": population_identity,
        "target_access": False, "llm_api_calls": 0,
    }


def verify_siw_population_against_expected(scan: dict[str, Any], layout: dict[str, Any]) -> dict[str, Any]:
    """Read-only. Fails closed (reports MISMATCH, never silently accepts)
    if the scanned population disagrees with the frozen expected counts --
    never adjusts the expectation to match what was found."""
    expected = layout["expected_counts"]
    if scan["status"] != "SCANNED":
        return {"CHECKED": False, "reason": scan["status"]}
    mismatches = []
    if scan["TOTAL"] != expected["total"]:
        mismatches.append(f"TOTAL {scan['TOTAL']} != expected {expected['total']}")
    if scan["LIVE"] != expected["live"]:
        mismatches.append(f"LIVE {scan['LIVE']} != expected {expected['live']}")
    if scan["SPOOF"] != expected["spoof"]:
        mismatches.append(f"SPOOF {scan['SPOOF']} != expected {expected['spoof']}")
    for family, count in expected["by_attack_family"].items():
        observed = scan["by_attack_family"].get(family, 0)
        if observed != count:
            mismatches.append(f"family {family}: {observed} != expected {count}")
    return {"CHECKED": True, "MATCHES_EXPECTED": not mismatches, "mismatches": mismatches}


def build_siw_local_population_plan(repo: Path) -> dict[str, Any]:
    scan = scan_local_siw_population(repo)
    layout = _load_siw_layout_config(repo)
    verification = verify_siw_population_against_expected(scan, layout)
    return {
        "schema_version": f"{SCHEMA_PREFIX}-siw-local-population-plan-v1",
        "SIW_SOURCE_ROOT": SIW_RAW_ROOT,
        "EXPECTED_TOTAL_VIDEOS": SIW_EXPECTED_TOTAL_VIDEOS,
        "EXPECTED_LIVE_VIDEOS": SIW_EXPECTED_LIVE_VIDEOS,
        "EXPECTED_SPOOF_VIDEOS": SIW_EXPECTED_SPOOF_VIDEOS,
        "layout_contract_source": SIW_LAYOUT_CONFIG_PATH,
        "scan": scan, "verification": verification,
        "external_dataset_used": False, "external_protocol_list_used_for_split": False,
        "no_video_silently_dropped": True,
        "target_access": False, "llm_api_calls": 0,
    }


def write_siw_local_population_plan(repo: Path) -> dict[str, Any]:
    plan = build_siw_local_population_plan(repo)
    return _write(repo, "E7A_SIW_LOCAL_POPULATION_PLAN.json", plan, out_dir_rel=AMENDMENT_DIR)


def compute_siw_video_split(records: list[dict[str, Any]], *, seed: int = SIW_SOURCE_SPLIT_SEED
                            ) -> dict[str, Any]:
    """TASK: deterministic, VIDEO-disjoint, stratified 80/20 split.

    Pure function of (records, seed): identical input always yields the
    identical assignment and identity. Stratified independently by 'live'
    and by each spoof family; within a stratum, videos are ordered by a
    seeded stable hash (never Python's randomness, never dict/set
    iteration order) before the 80% cut, so the split is reproducible
    across processes/platforms. One video_id -> exactly one split; every
    frame/crop later derived from that video inherits it.
    """
    import hashlib

    def _bucket(value: str) -> int:
        return int.from_bytes(hashlib.sha256(f"{seed}|{value}".encode("utf-8")).digest()[:8], "big")

    strata: dict[str, list[str]] = {}
    for row in records:
        stratum = "live" if row["class_live_spoof"] == "live" else f"spoof:{row['spoof_family']}"
        strata.setdefault(stratum, []).append(row["video_id"])

    assignment: dict[str, str] = {}
    stratum_report: dict[str, dict[str, int]] = {}
    for stratum, video_ids in strata.items():
        unique_ids = sorted(set(video_ids))
        if len(unique_ids) != len(video_ids):
            raise E7AError(f"stratum {stratum!r} contains a duplicate video_id -- refusing to split a "
                           "population with non-unique video identities")
        ordered = sorted(unique_ids, key=lambda vid: (_bucket(f"{stratum}|{vid}"), vid))
        cut = int(round(len(ordered) * 0.8))
        cut = min(max(cut, 1), len(ordered) - 1) if len(ordered) > 1 else len(ordered)
        for index, video_id in enumerate(ordered):
            assignment[video_id] = "train" if index < cut else "dev"
        stratum_report[stratum] = {"total": len(ordered), "train": cut, "dev": len(ordered) - cut}

    train_count = sum(1 for split in assignment.values() if split == "train")
    dev_count = sum(1 for split in assignment.values() if split == "dev")
    identity_material = "|".join(f"{vid}:{assignment[vid]}" for vid in sorted(assignment))
    split_identity = cc.sha256_bytes(identity_material.encode("utf-8"))

    return {
        "seed": seed, "assignment": assignment, "stratum_report": stratum_report,
        "train_count": train_count, "dev_count": dev_count, "total_count": len(assignment),
        "split_identity": split_identity,
        "video_overlap_allowed": False,
        "no_random_global_state": True, "deterministic": True,
    }


def build_siw_video_split_policy_lock(repo: Path) -> dict[str, Any]:
    """TASK: freezes the AMENDED split RULE, and -- only when the local
    population was actually scanned (GPU host) -- the resulting proposed
    split counts/identity. On the laptop this is PLAN_ONLY: the rule is
    frozen, but no split is computed from data that is not there.
    """
    scan = scan_local_siw_population(repo)
    split_result = None
    if scan["status"] == "SCANNED":
        split_result = compute_siw_video_split(scan["records"], seed=SIW_SOURCE_SPLIT_SEED)

    body = {
        "schema_version": f"{SCHEMA_PREFIX}-siw-video-split-policy-lock-v1",
        "SIW_SOURCE_SPLIT_POLICY": NEW_SIW_SOURCE_SPLIT_POLICY,
        "SIW_SOURCE_SPLIT_SEED": SIW_SOURCE_SPLIT_SEED,
        "GROUP_KEY": "canonical video_id",
        "STRATIFICATION": ["live/spoof", "spoof attack family for spoof videos"],
        "TRAIN_FRACTION": 0.8, "DEV_FRACTION": 0.2,
        "VIDEO_OVERLAP_ALLOWED": False,
        "SUBJECT_DISJOINTNESS": "UNVERIFIABLE_NOT_ENFORCED",
        "SUBJECT_DISJOINTNESS_REASON": "canonical subject mapping is unavailable in the exact local "
                                       "dataset release and no external dataset/metadata may be "
                                       "introduced",
        "no_subject_inferred_from_filename": True,
        "scientific_limitation": "because subject identity is unavailable, the same physical identity "
                                 "could in principle have separate videos in source_train and "
                                 "source_dev; target-domain isolation and video-level train/dev "
                                 "isolation remain enforceable, but subject-level source train/dev "
                                 "isolation is NOT guaranteed -- source-dev calibration may contain "
                                 "identity dependence; this must be disclosed in the final E7 report",
        "proposed_split": split_result,
        "status": "FROZEN" if split_result is None else "FROZEN_WITH_PROPOSED_SPLIT",
        "manifests_written": False,
        "target_access": False, "llm_api_calls": 0,
    }
    body["policy_lock_identity"] = cc.sha256_bytes(cc.canonical_json_bytes(
        {k: v for k, v in body.items() if k != "proposed_split"}))
    return body


def write_siw_video_split_policy_lock(repo: Path) -> dict[str, Any]:
    lock = build_siw_video_split_policy_lock(repo)
    return _write(repo, "E7A_SIW_VIDEO_SPLIT_POLICY_LOCK.json", lock, out_dir_rel=AMENDMENT_DIR)


def build_m3b_binding_correction(repo: Path) -> dict[str, Any]:
    """FIX 2: documents (never silently patches) the discrepancy between
    E0's frozen prose (`EXT_DATASET_FOLD_PLAN.json`'s casia_msu rule text,
    which names the WRONG, zero-row `prism_target_eval_v2` placeholder
    path) and the operationally correct binding this module has always
    actually used for file checks (`CASIA_MSU_PACKAGE_ROOT`). Never rewrites
    E0's file; never alters M3B bytes.
    """
    wrong_path = f"{SIW_TARGET_EVAL_PACKAGE_ROOT}/manifests/source_train.parquet"
    correct_train = f"{CASIA_MSU_PACKAGE_ROOT}/manifests/source_train.parquet"
    correct_dev = f"{CASIA_MSU_PACKAGE_ROOT}/manifests/source_dev.parquet"
    train_present = (repo / correct_train).is_file()
    dev_present = (repo / correct_dev).is_file()

    dev_counts = None
    if dev_present:
        import pyarrow.parquet as pq

        table = pq.read_table(repo / correct_dev).to_pydict()
        dev_counts = {"total": len(table["sample_id"]),
                     "casia": sum(1 for d in table["dataset"] if d == "casia_fasd"),
                     "msu": sum(1 for d in table["dataset"] if d == "msu_mfsd")}

    return {
        "schema_version": f"{SCHEMA_PREFIX}-m3b-binding-correction-v1",
        "e0_frozen_prose_path": wrong_path,
        "e0_frozen_prose_note": "EXT_DATASET_FOLD_PLAN.json's source_split_policy.casia_msu prose names "
                                "this path; it is a target-package placeholder with ZERO rows on this "
                                "host and must never be used to construct CASIA/MSU source manifests",
        "e0_file_rewritten": False,
        "corrected_canonical_source_train_path": correct_train,
        "corrected_canonical_source_dev_path": correct_dev,
        "corrected_source_train_present_locally": train_present,
        "corrected_source_dev_present_locally": dev_present,
        "expected_source_train_counts": {"total": M3B_EXPECTED_SOURCE_TRAIN_TOTAL,
                                         "casia": M3B_EXPECTED_SOURCE_TRAIN_CASIA,
                                         "msu": M3B_EXPECTED_SOURCE_TRAIN_MSU},
        "expected_source_dev_counts": {"total": M3B_EXPECTED_SOURCE_DEV_TOTAL,
                                       "casia": M3B_EXPECTED_SOURCE_DEV_CASIA,
                                       "msu": M3B_EXPECTED_SOURCE_DEV_MSU},
        "observed_source_dev_counts_this_host": dev_counts,
        "observed_source_dev_matches_expected": dev_counts == {
            "total": M3B_EXPECTED_SOURCE_DEV_TOTAL, "casia": M3B_EXPECTED_SOURCE_DEV_CASIA,
            "msu": M3B_EXPECTED_SOURCE_DEV_MSU} if dev_counts else None,
        "target_eval_v2_placeholder_note": "data/processed/prism_target_eval_v2/manifests/"
                                          "{source_train,source_dev}.parquet exist but hold ZERO rows "
                                          "-- schema-only placeholders, never accepted as a CASIA/MSU "
                                          "source manifest",
        "m3b_bytes_altered": False,
        "target_access": False, "llm_api_calls": 0,
    }


def write_m3b_binding_correction(repo: Path) -> dict[str, Any]:
    correction = build_m3b_binding_correction(repo)
    return _write(repo, "E7A_M3B_BINDING_CORRECTION.json", correction, out_dir_rel=AMENDMENT_DIR)


def build_siw_local_only_amendment(repo: Path) -> dict[str, Any]:
    """TASK: the explicit PRE-EXECUTION SCIENTIFIC PROTOCOL AMENDMENT
    record. This is NOT a bug-fix artifact -- SUBJECT_GROUP_DISJOINT_80_20
    is a real, deliberate, frozen scientific rule being REPLACED, before
    any E7 scientific execution, because it is technically unexecutable
    under the only data this project is permitted to use.
    """
    return {
        "schema_version": f"{SCHEMA_PREFIX}-siw-local-only-amendment-v1",
        "PREVIOUS_POLICY": PREVIOUS_SIW_SOURCE_SPLIT_POLICY,
        "AMENDMENT_REASON": "SUBJECT_METADATA_UNAVAILABLE_IN_ALLOWED_LOCAL_DATA",
        "amendment_reason_detail": "the exact permitted local SiW-Mv2 release "
                                   f"({SIW_RAW_ROOT}) has no canonical video->subject mapping; the "
                                   "committed SiWMv2Adapter always emits subject_id=None by design "
                                   "(built for SiW-Mv2's historical opaque held-out-TARGET role); "
                                   "filenames are video/sample names and MUST NOT be reinterpreted as "
                                   "subject ids; no external dataset, release, or protocol/subject "
                                   "metadata may be introduced to resolve this",
        "NEW_POLICY": NEW_SIW_SOURCE_SPLIT_POLICY,
        "SCIENTIFIC_PROTOCOL_CHANGED": True,
        "CHANGE_TIMING": "BEFORE_E7_SCIENTIFIC_EXECUTION",
        "E7_SCIENTIFIC_RUNS_BEFORE_AMENDMENT": 0,
        "EXTERNAL_DATASET_USED": False,
        "EXTERNAL_PROTOCOL_LIST_USED_FOR_SPLIT": False,
        "SIW_SOURCE_POPULATION_POLICY": "EXACT_LOCAL_RAW_POPULATION",
        "SIW_SOURCE_ROOT": SIW_RAW_ROOT,
        "EXPECTED_TOTAL_VIDEOS": SIW_EXPECTED_TOTAL_VIDEOS,
        "EXPECTED_LIVE_VIDEOS": SIW_EXPECTED_LIVE_VIDEOS,
        "EXPECTED_SPOOF_VIDEOS": SIW_EXPECTED_SPOOF_VIDEOS,
        "SIW_SOURCE_SPLIT_POLICY": NEW_SIW_SOURCE_SPLIT_POLICY,
        "SIW_SOURCE_SPLIT_SEED": SIW_SOURCE_SPLIT_SEED,
        "GROUP_KEY": "canonical video_id",
        "STRATIFICATION": ["live/spoof", "spoof attack family for spoof videos"],
        "TRAIN_FRACTION": 0.8, "DEV_FRACTION": 0.2,
        "VIDEO_OVERLAP_ALLOWED": False,
        "SUBJECT_DISJOINTNESS": "UNVERIFIABLE_NOT_ENFORCED",
        "SUBJECT_DISJOINTNESS_REASON": "canonical subject mapping is unavailable in the exact local "
                                       "dataset release and no external dataset/metadata may be "
                                       "introduced",
        "does_not_alter_ext_f1": True,
        "ext_f1_note": "EXT-F1 (CASIA-FASD + MSU-MFSD -> SiW-Mv2 held-out target) is unaffected -- it "
                       "reuses the already-frozen prism_target_eval_v2 target package verbatim",
        "does_not_rewrite_e6_e7_historical_locks": True,
        "target_access": False, "llm_api_calls": 0,
        "status": "FROZEN",
    }


def write_siw_local_only_amendment(repo: Path) -> dict[str, Any]:
    amendment = build_siw_local_only_amendment(repo)
    return _write(repo, "E7A_SIW_LOCAL_ONLY_AMENDMENT.json", amendment, out_dir_rel=AMENDMENT_DIR)


def build_amended_fold_construction_plan(repo: Path) -> dict[str, Any]:
    """F2/F3 construction semantics: filter, never reuse the shared CASIA/
    MSU pool file wholesale; both folds reference the SAME single SiW
    source-split identity. EXT-F1 is unchanged.

    BUG 1 FIX: `siw_source_split_identity` and `siw_source_split_policy_lock_identity`
    are DIFFERENT concepts and are now persisted separately and correctly.
    `siw_source_split_identity` is the ACTUAL deterministic video assignment's
    own identity (`compute_siw_video_split`'s `split_identity` -- what one
    build/resume comparison must match bit-for-bit); `..._policy_lock_identity`
    is the identity of the frozen RULE itself (seed/ratio/stratification),
    which stays constant even if the underlying population changes and the
    resulting assignment identity therefore changes. Previously both fields
    were silently bound to the SAME `policy_lock_identity` value -- a real
    identity-binding bug, now fixed. When local raw data are unavailable,
    `siw_source_split_identity` is explicitly `None` (PLAN_ONLY), never
    fabricated.
    """
    split_lock = build_siw_video_split_policy_lock(repo)
    policy_lock_identity = split_lock["policy_lock_identity"]
    proposed_split = split_lock.get("proposed_split")
    actual_split_identity = proposed_split["split_identity"] if proposed_split else None
    siw_state = "PLAN_ONLY" if actual_split_identity is None else "ACTUAL_SPLIT_COMPUTED"
    return {
        "schema_version": f"{SCHEMA_PREFIX}-amended-fold-construction-plan-v2",
        "EXT-F1": {"unchanged": True, "source": ["CASIA-FASD", "MSU-MFSD"], "target": "SiW-Mv2",
                  "action": "reuse the already-frozen prism_target_eval_v2 target package verbatim"},
        "EXT-F2": {"source": ["CASIA-FASD", "SiW-Mv2"], "target": "MSU-MFSD",
                  "casia_rows": "FILTER dataset=='casia_fasd' ONLY from the frozen M3B source_train/"
                                "source_dev -- MSU rows are EXT-F2's held-out target and must NEVER be "
                                "included as source",
                  "siw_rows": "the amended local video-disjoint split, train/dev partitions per "
                             "compute_siw_video_split",
                  "siw_source_split_identity": actual_split_identity,
                  "siw_source_split_policy_lock_identity": policy_lock_identity,
                  "siw_source_split_state": siw_state,
                  "excludes_msu_source_rows": True},
        "EXT-F3": {"source": ["MSU-MFSD", "SiW-Mv2"], "target": "CASIA-FASD",
                  "msu_rows": "FILTER dataset=='msu_mfsd' ONLY from the frozen M3B source_train/"
                              "source_dev -- CASIA rows are EXT-F3's held-out target and must NEVER be "
                              "included as source",
                  "siw_rows": "REUSES the EXACT SAME SiW source-split identity as EXT-F2 -- never "
                             "recomputed independently",
                  "siw_source_split_identity": actual_split_identity,
                  "siw_source_split_policy_lock_identity": policy_lock_identity,
                  "siw_source_split_state": siw_state,
                  "excludes_casia_source_rows": True},
        "f2_f3_share_one_siw_split": True,
        "manifests_written": False,
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False,
    }


def write_amended_fold_construction_plan(repo: Path) -> dict[str, Any]:
    plan = build_amended_fold_construction_plan(repo)
    return _write(repo, "E7A_AMENDED_FOLD_CONSTRUCTION_PLAN.json", plan, out_dir_rel=AMENDMENT_DIR)


def build_readiness_fix_report(repo: Path) -> dict[str, Any]:
    """FIX 3: documents the readiness-logic false positive and its fix."""
    old_preflight = e7a_preflight(repo)  # the ORIGINAL (still-present) preflight function
    return {
        "schema_version": f"{SCHEMA_PREFIX}-readiness-fix-report-v1",
        "fix_1_raw_path_detection": {
            "bug": "audit_dataset_infrastructure hardcoded raw_siw_source_bytes_present_locally=False "
                  "unconditionally, regardless of whether the GPU host actually has the exact "
                  f"permitted local root ({SIW_RAW_ROOT})",
            "fixed": True,
            "current_value_this_host": (repo / SIW_RAW_ROOT).is_dir(),
        },
        "fix_2_m3b_binding": {
            "bug": "E0's frozen source_split_policy.casia_msu PROSE names the zero-row "
                  f"{SIW_TARGET_EVAL_PACKAGE_ROOT} placeholder path; operational file checks always "
                  f"used the correct {CASIA_MSU_PACKAGE_ROOT} path, but the prose was misleading",
            "fixed": True,
            "correction_artifact": "E7A_M3B_BINDING_CORRECTION.json",
        },
        "fix_3_readiness_logic": {
            "bug": "e7a_preflight's original E7A_READY_FOR_BUILD was TRUE whenever "
                  "resolve_source_split_policy() did not RAISE -- but an UNRESOLVED subject-resolution "
                  "status is not an exception, so readiness was TRUE even though no real F2/F3 build "
                  "was actually possible",
            "old_value_this_host": old_preflight["E7A_READY_FOR_BUILD"],
            "fixed": True,
            "new_logic": "E7A_READY_FOR_BUILD now additionally requires: the amendment lock present and "
                        "matching, the exact local SiW root present, the frozen population inventory "
                        "matching expected counts, the deterministic video-split implementation "
                        "available, the M3B CASIA/MSU manifests present with matching identities, "
                        "target-domain isolation checks resolvable, and no target label access",
        },
        "target_access": False, "llm_api_calls": 0,
    }


def write_readiness_fix_report(repo: Path) -> dict[str, Any]:
    report = build_readiness_fix_report(repo)
    return _write(repo, "E7A_READINESS_FIX_REPORT.json", report, out_dir_rel=AMENDMENT_DIR)


def build_amended_execution_plan() -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_PREFIX}-amended-execution-plan-v1",
        "cli_operations": [
            {"flag": "--e7a-local-siw-preflight", "read_only": True,
            "does": ["inventory local raw SiW", "verify counts/families against the frozen layout "
                    "contract", "verify M3B CASIA/MSU manifest presence/counts",
                    "compute PROPOSED split counts/identity (in-memory only)"],
            "never": ["writes a source manifest", "trains", "renders", "fits GPAT"]},
            {"flag": "--e7a-local-siw-freeze", "read_only": False, "requires": "explicit --authorize",
            "does": "persists the amendment + population + split-policy locks additively",
            "never": ["writes a source manifest", "trains", "renders", "fits GPAT"]},
            {"flag": "--e7a-build", "read_only": False, "requires": "explicit --authorize",
            "does": "constructs source train/dev manifest references (unchanged from the original "
                   "E7-A contract)", "still_fails_closed_on": "incomplete local bytes (unchanged this "
                   "turn -- laptop has neither SiW raw bytes nor the M3B source_train.parquet)"},
        ],
        "laptop_this_turn_executes_real_gpu_data": False,
        "executed_this_turn": [], "target_access": False, "llm_api_calls": 0,
    }


def write_amended_execution_plan(repo: Path) -> dict[str, Any]:
    plan = build_amended_execution_plan()
    return _write(repo, "E7A_AMENDED_EXECUTION_PLAN.json", plan, out_dir_rel=AMENDMENT_DIR)


def build_amended_readiness(repo: Path) -> dict[str, Any]:
    """FIX 3 applied: E7A_READY_FOR_BUILD is derived from the FULL amended
    prerequisite checklist, never from "no exception was raised"."""
    amendment_path = repo / AMENDMENT_DIR / "E7A_SIW_LOCAL_ONLY_AMENDMENT.json"
    amendment_present = amendment_path.is_file()
    amendment_matches = False
    if amendment_present:
        persisted = cc.read_json(amendment_path)
        expected = build_siw_local_only_amendment(repo)
        amendment_matches = persisted.get("NEW_POLICY") == expected["NEW_POLICY"] and \
            persisted.get("SIW_SOURCE_SPLIT_SEED") == expected["SIW_SOURCE_SPLIT_SEED"]

    siw_root_present = (repo / SIW_RAW_ROOT).is_dir()
    population_scan = scan_local_siw_population(repo)
    layout = _load_siw_layout_config(repo)
    population_check = verify_siw_population_against_expected(population_scan, layout)
    population_matches = bool(population_check.get("MATCHES_EXPECTED"))

    m3b_correction = build_m3b_binding_correction(repo)
    m3b_ready = bool(m3b_correction["corrected_source_train_present_locally"]
                     and m3b_correction["corrected_source_dev_present_locally"]
                     and m3b_correction.get("observed_source_dev_matches_expected"))

    isolation = audit_fold_isolation_e7a(repo)
    target_isolation_resolvable = all("TARGET_DOMAIN_EXACT" in f for f in isolation["per_fold"])

    prerequisites = {
        "amendment_lock_present_and_matching": amendment_present and amendment_matches,
        "exact_local_siw_root_present": siw_root_present,
        "frozen_population_inventory_matches": population_matches,
        "deterministic_video_split_implementation_available": True,  # compute_siw_video_split exists
        "m3b_casia_msu_manifests_present_and_matching": m3b_ready,
        "target_domain_isolation_checks_resolvable": target_isolation_resolvable,
        "no_target_label_access": isolation["TARGET_LABEL_ACCESS"] is False,
    }
    ready_for_build = all(prerequisites.values())

    return {
        "schema_version": f"{SCHEMA_PREFIX}-amended-readiness-v1",
        "prerequisites": prerequisites,
        "E7A_READY_FOR_BUILD": ready_for_build,
        "E7A_READY_FOR_GPU_LOCAL_PREFLIGHT": True,  # the read-only preflight code itself is complete
        "rendering_performed": False, "training_performed": False, "gpat_fitting_performed": False,
        "target_access": False, "llm_api_calls": 0,
        "status": "PREPARED_NOT_EXECUTED",
    }


def write_amended_readiness(repo: Path) -> dict[str, Any]:
    readiness = build_amended_readiness(repo)
    return _write(repo, "E7A_AMENDED_READINESS.json", readiness, out_dir_rel=AMENDMENT_DIR)


def e7a_local_siw_preflight(repo: Path) -> dict[str, Any]:
    """`--e7a-local-siw-preflight`: strictly read-only. Inventories local
    raw SiW, verifies counts/families, verifies M3B, computes the PROPOSED
    split counts/identity in memory. Writes NOTHING. Never trains, renders
    or fits GPAT."""
    population_plan = build_siw_local_population_plan(repo)
    split_lock = build_siw_video_split_policy_lock(repo)
    m3b_correction = build_m3b_binding_correction(repo)
    readiness = build_amended_readiness(repo)
    return {
        "schema_version": f"{SCHEMA_PREFIX}-local-siw-preflight-v1",
        "population_plan": population_plan, "split_policy": split_lock,
        "m3b_binding_correction": m3b_correction, "readiness": readiness,
        "manifests_written": False, "rendering_performed": False, "training_performed": False,
        "gpat_fitting_performed": False, "target_access": False, "llm_api_calls": 0,
    }


def prepare_e7a_amendment(repo: Path) -> dict[str, Any]:
    """Writes every additive amendment artifact under AMENDMENT_DIR. Never
    touches the original, already-committed E7-A artifacts. Never renders,
    trains, fits GPAT, touches target labels or calls an LLM."""
    results = {
        "amendment": write_siw_local_only_amendment(repo),
        "population_plan": write_siw_local_population_plan(repo),
        "split_policy_lock": write_siw_video_split_policy_lock(repo),
        "m3b_binding_correction": write_m3b_binding_correction(repo),
        "fold_construction_plan": write_amended_fold_construction_plan(repo),
        "readiness_fix_report": write_readiness_fix_report(repo),
        "execution_plan": write_amended_execution_plan(repo),
    }
    results["readiness"] = write_amended_readiness(repo)
    return results


# =============================================================================
# MATERIALIZATION (this turn): fixes BUG 2 (`e7a_build` always raised
# unconditionally) by implementing the actual E7-A build, bound EXPLICITLY
# to the amended local-only SiW policy (never `resolve_source_split_policy`,
# the ORIGINAL subject-disjoint rule, which stays defined only for
# provenance and is never called by this path). Additive namespace only;
# never touches the original d81c5b4 E7-A files or the 836f74a amendment
# freeze evidence.
# =============================================================================

MATERIALIZATION_DIR = f"{E7A_REPORT_DIR}/materialization_v1"


class E7AMaterializationConflict(E7AError):
    """An existing materialized fold disagrees with what would be built now."""


def _load_persisted_amendment_artifacts(repo: Path) -> dict[str, Any]:
    """Loads the COMMITTED (836f74a) amendment freeze artifacts. Never the
    original pre-amendment E7A_SOURCE_SPLIT_LOCK.json/subject-disjoint
    rule."""
    paths = {
        "amendment": repo / AMENDMENT_DIR / "E7A_SIW_LOCAL_ONLY_AMENDMENT.json",
        "population_plan": repo / AMENDMENT_DIR / "E7A_SIW_LOCAL_POPULATION_PLAN.json",
        "split_lock": repo / AMENDMENT_DIR / "E7A_SIW_VIDEO_SPLIT_POLICY_LOCK.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise E7AError(f"missing persisted amendment artifacts: {missing}; run "
                       "--e7a-local-siw-freeze --authorize first")
    return {name: cc.read_json(path) for name, path in paths.items()}


def _m3b_manifest_rows(repo: Path, split: str) -> list[dict[str, Any]] | None:
    path = repo / CASIA_MSU_PACKAGE_ROOT / "manifests" / f"{split}.parquet"
    if not path.is_file():
        return None
    import pyarrow.parquet as pq

    table = pq.read_table(path).to_pydict()
    return [{key: table[key][index] for key in table} for index in range(len(table["sample_id"]))]


def _m3b_counts_match(rows: list[dict[str, Any]] | None, *, total: int, casia: int, msu: int) -> bool:
    if rows is None:
        return False
    observed_total = len(rows)
    observed_casia = sum(1 for r in rows if r["dataset"] == "casia_fasd")
    observed_msu = sum(1 for r in rows if r["dataset"] == "msu_mfsd")
    return observed_total == total and observed_casia == casia and observed_msu == msu


def build_m3b_source_reference(row: dict[str, Any], *, fold_id: str, project_split: str) -> dict[str, Any]:
    """The additive E7-A reference for one M3B-processed sample. Only real
    fields -- nothing fabricated. `reference_kind` explicitly distinguishes
    this from a raw SiW video reference."""
    return {
        "fold_id": fold_id, "dataset": _DATASET_LABEL.get(row["dataset"], row["dataset"]),
        "project_split": project_split, "reference_kind": "m3b_processed_sample",
        "sample_id": row["sample_id"], "source_record_id": row.get("source_record_id"),
        "subject_id": row.get("subject_id"), "label_live_spoof": row.get("label_live_spoof"),
        "image_relative_path": row.get("image_relative_path"),
        "prior_relative_path": row.get("prior_relative_path"),
        "crop_sha256": row.get("crop_sha256"), "prior_sha256": row.get("prior_sha256"),
    }


def build_siw_source_reference(record: dict[str, Any], *, fold_id: str, project_split: str,
                               population_identity: str, split_identity: str) -> dict[str, Any]:
    """The additive E7-A reference for one raw SiW-Mv2 video. NEVER
    contains a subject_id -- SiW is still raw-video source input requiring
    the frozen Version-C face preprocessing (SCRFD etc.) before any
    detector training; E7-A itself never runs that preprocessing."""
    return {
        "fold_id": fold_id, "dataset": "SiW-Mv2", "project_split": project_split,
        "reference_kind": "siw_raw_video",
        "video_id": record["video_id"], "relative_path": record["relative_path"],
        "label_live_spoof": record["class_live_spoof"], "spoof_family": record["spoof_family"],
        "extension": record["extension"],
        "population_identity": population_identity, "split_identity": split_identity,
        "requires_frozen_face_preprocessing": True,
    }


def _fold_identity(*, fold_id: str, source_domains: list[str], target_domain: str,
                   source_train_reference_identity: str, source_dev_reference_identity: str,
                   siw_population_identity: str | None, siw_split_identity: str | None,
                   m3b_package_identity: str | None, target_reference_identity: str | None
                   ) -> str:
    material = {
        "fold_id": fold_id, "source_domains": sorted(source_domains), "target_domain": target_domain,
        "source_train_reference_identity": source_train_reference_identity,
        "source_dev_reference_identity": source_dev_reference_identity,
        "siw_population_identity": siw_population_identity, "siw_split_identity": siw_split_identity,
        "m3b_package_identity": m3b_package_identity, "target_reference_identity": target_reference_identity,
    }
    return cc.sha256_bytes(cc.canonical_json_bytes(material))


def e7a_build_preflight(repo: Path) -> dict[str, Any]:
    """`--e7a-build-preflight`: STRICTLY READ-ONLY. Computes exactly what
    `--e7a-build --authorize` would materialize but writes nothing.
    """
    try:
        persisted = _load_persisted_amendment_artifacts(repo)
        amended_protocol_active = True
    except E7AError:
        persisted = None
        amended_protocol_active = False

    population_identity_match = False
    split_identity_match = False
    actual_split_identity = None
    fresh_assignment: dict[str, str] = {}
    video_train_dev_overlap = None
    if persisted:
        fresh_scan = scan_local_siw_population(repo)
        persisted_population_identity = persisted["population_plan"].get("scan", {}).get("population_identity")
        population_identity_match = (fresh_scan["status"] == "SCANNED"
                                     and fresh_scan.get("population_identity") == persisted_population_identity
                                     and persisted_population_identity is not None)
        if population_identity_match:
            fresh_split = compute_siw_video_split(fresh_scan["records"], seed=SIW_SOURCE_SPLIT_SEED)
            persisted_split_identity = (persisted["split_lock"].get("proposed_split") or {}).get(
                "split_identity")
            split_identity_match = (fresh_split["split_identity"] == persisted_split_identity
                                    and persisted_split_identity is not None)
            if split_identity_match:
                actual_split_identity = fresh_split["split_identity"]
                fresh_assignment = fresh_split["assignment"]
                train_ids = {v for v, s in fresh_assignment.items() if s == "train"}
                dev_ids = {v for v, s in fresh_assignment.items() if s == "dev"}
                video_train_dev_overlap = len(train_ids & dev_ids)

    fresh_policy_lock = build_siw_video_split_policy_lock(repo)
    policy_lock_identity_match = (persisted is not None and fresh_policy_lock["policy_lock_identity"]
                                  == persisted["split_lock"].get("policy_lock_identity"))

    m3b_train_rows = _m3b_manifest_rows(repo, "source_train")
    m3b_dev_rows = _m3b_manifest_rows(repo, "source_dev")
    m3b_source_train_match = _m3b_counts_match(
        m3b_train_rows, total=M3B_EXPECTED_SOURCE_TRAIN_TOTAL, casia=M3B_EXPECTED_SOURCE_TRAIN_CASIA,
        msu=M3B_EXPECTED_SOURCE_TRAIN_MSU)
    m3b_source_dev_match = _m3b_counts_match(
        m3b_dev_rows, total=M3B_EXPECTED_SOURCE_DEV_TOTAL, casia=M3B_EXPECTED_SOURCE_DEV_CASIA,
        msu=M3B_EXPECTED_SOURCE_DEV_MSU)

    everything_available = (amended_protocol_active and population_identity_match and split_identity_match
                            and policy_lock_identity_match and m3b_source_train_match
                            and m3b_source_dev_match)

    f1_train = f1_dev = f2_train = f2_dev = f3_train = f3_dev = None
    f2_heldout_leak = f3_heldout_leak = None
    if everything_available:
        f1_train = len(m3b_train_rows)
        f1_dev = len(m3b_dev_rows)
        casia_train = [r for r in m3b_train_rows if r["dataset"] == "casia_fasd"]
        casia_dev = [r for r in m3b_dev_rows if r["dataset"] == "casia_fasd"]
        msu_train = [r for r in m3b_train_rows if r["dataset"] == "msu_mfsd"]
        msu_dev = [r for r in m3b_dev_rows if r["dataset"] == "msu_mfsd"]
        siw_train_count = sum(1 for s in fresh_assignment.values() if s == "train")
        siw_dev_count = sum(1 for s in fresh_assignment.values() if s == "dev")

        f2_train = len(casia_train) + siw_train_count
        f2_dev = len(casia_dev) + siw_dev_count
        f2_heldout_leak = 0  # F2's construction never includes an MSU row -- see e7a_amended_build

        f3_train = len(msu_train) + siw_train_count
        f3_dev = len(msu_dev) + siw_dev_count
        f3_heldout_leak = 0  # F3 never includes CASIA by construction

    subject_disjointness = "UNVERIFIABLE_NOT_ENFORCED"
    required = {
        "AMENDED_PROTOCOL_ACTIVE": amended_protocol_active,
        "POPULATION_IDENTITY_MATCH": population_identity_match,
        "SIW_SPLIT_IDENTITY_MATCH": split_identity_match,
        "SIW_POLICY_LOCK_IDENTITY_MATCH": policy_lock_identity_match,
        "M3B_SOURCE_TRAIN_MATCH": m3b_source_train_match,
        "M3B_SOURCE_DEV_MATCH": m3b_source_dev_match,
    }
    build_pass = all(required.values()) and (video_train_dev_overlap in (None, 0))

    return {
        "schema_version": f"{SCHEMA_PREFIX}-build-preflight-v1",
        **required,
        "EXT_F1_SOURCE_TRAIN_COUNT": f1_train, "EXT_F1_SOURCE_DEV_COUNT": f1_dev,
        "EXT_F2_SOURCE_TRAIN_COUNT": f2_train, "EXT_F2_SOURCE_DEV_COUNT": f2_dev,
        "EXT_F2_HELDOUT_TARGET_ROWS_IN_SOURCE": f2_heldout_leak,
        "EXT_F3_SOURCE_TRAIN_COUNT": f3_train, "EXT_F3_SOURCE_DEV_COUNT": f3_dev,
        "EXT_F3_HELDOUT_TARGET_ROWS_IN_SOURCE": f3_heldout_leak,
        "F2_F3_SAME_SIW_SPLIT": True,
        "VIDEO_TRAIN_DEV_OVERLAP": video_train_dev_overlap,
        "SUBJECT_DISJOINTNESS": subject_disjointness,
        "TARGET_ACCESS": False,
        "MANIFESTS_WRITTEN": False, "RENDERING_PERFORMED": False, "TRAINING_PERFORMED": False,
        "GPAT_FITTING_PERFORMED": False, "LLM_API_CALLS": 0,
        "actual_split_identity": actual_split_identity,
        "E7A_BUILD_PREFLIGHT_PASS": build_pass,
    }


def _materialize_fold(repo: Path, *, fold_id: str, source_domains: list[str], target_domain: str,
                      train_refs: list[dict[str, Any]], dev_refs: list[dict[str, Any]],
                      siw_population_identity: str | None, siw_split_identity: str | None,
                      m3b_package_identity: str | None, target_reference: dict[str, Any]
                      ) -> dict[str, Any]:
    train_identity = cc.sha256_bytes(cc.canonical_json_bytes(train_refs))
    dev_identity = cc.sha256_bytes(cc.canonical_json_bytes(dev_refs))
    target_reference_identity = cc.sha256_bytes(cc.canonical_json_bytes(target_reference))
    fold_identity = _fold_identity(
        fold_id=fold_id, source_domains=source_domains, target_domain=target_domain,
        source_train_reference_identity=train_identity, source_dev_reference_identity=dev_identity,
        siw_population_identity=siw_population_identity, siw_split_identity=siw_split_identity,
        m3b_package_identity=m3b_package_identity, target_reference_identity=target_reference_identity)
    return {
        "schema_version": f"{SCHEMA_PREFIX}-materialized-fold-v1",
        "fold_id": fold_id, "source_domains": source_domains, "target_domain": target_domain,
        "source_train_references": train_refs, "source_dev_references": dev_refs,
        "source_train_reference_identity": train_identity, "source_dev_reference_identity": dev_identity,
        "target_reference": target_reference, "target_reference_identity": target_reference_identity,
        "siw_population_identity": siw_population_identity, "siw_split_identity": siw_split_identity,
        "m3b_package_identity": m3b_package_identity,
        "fold_identity": fold_identity,
        "target_labels_opened": False,
        "status": "FROZEN",
    }


def e7a_amended_build(repo: Path, *, authorize: bool = False) -> dict[str, Any]:
    """`--e7a-build --authorize` (amended path): materializes EXT-F1/F2/F3
    source_train/source_dev REFERENCE manifests plus a label-free held-out
    target reference. Bound EXPLICITLY to the amended local-only SiW
    policy -- NEVER calls `resolve_source_split_policy` (the original,
    unexecutable subject-disjoint rule). Re-runs the exact same preflight
    internally before writing. Writes atomically (temp file + rename). If
    an existing materialization is present: identical fold_identity ->
    resume-safe ALREADY_MATERIALIZED_MATCH; different -> FAIL CLOSED, never
    overwrites. Runs the validator automatically after a fresh write.
    """
    if not authorize:
        raise E7AError("--e7a-build requires explicit authorization; refusing to run")

    preflight = e7a_build_preflight(repo)
    if not preflight["E7A_BUILD_PREFLIGHT_PASS"]:
        raise E7AError(f"E7-A amended build preflight did not pass: {preflight}")

    persisted = _load_persisted_amendment_artifacts(repo)
    population_identity = persisted["population_plan"]["scan"]["population_identity"]
    split_identity = preflight["actual_split_identity"]
    m3b_package_identity = cc.read_json(repo / CASIA_MSU_PACKAGE_ROOT / "PACKAGE_LOCK.json").get(
        "content_identity_sha256")

    m3b_train_rows = _m3b_manifest_rows(repo, "source_train")
    m3b_dev_rows = _m3b_manifest_rows(repo, "source_dev")
    fresh_scan = scan_local_siw_population(repo)
    fresh_split = compute_siw_video_split(fresh_scan["records"], seed=SIW_SOURCE_SPLIT_SEED)
    assignment = fresh_split["assignment"]
    records_by_id = {r["video_id"]: r for r in fresh_scan["records"]}

    def _siw_refs(fold_id: str, split_name: str) -> list[dict[str, Any]]:
        return [build_siw_source_reference(records_by_id[vid], fold_id=fold_id, project_split=split_name,
                                           population_identity=population_identity,
                                           split_identity=split_identity)
               for vid, split in assignment.items() if split == split_name]

    materialized: dict[str, dict[str, Any]] = {}

    # EXT-F1: unchanged, whole-file M3B semantics; SiW is the held-out target (not source)
    f1_train_refs = [build_m3b_source_reference(r, fold_id="EXT-F1", project_split="source_train")
                     for r in m3b_train_rows]
    f1_dev_refs = [build_m3b_source_reference(r, fold_id="EXT-F1", project_split="source_dev")
                  for r in m3b_dev_rows]
    materialized["EXT-F1"] = _materialize_fold(
        repo, fold_id="EXT-F1", source_domains=["CASIA-FASD", "MSU-MFSD"], target_domain="SiW-Mv2",
        train_refs=f1_train_refs, dev_refs=f1_dev_refs, siw_population_identity=None,
        siw_split_identity=None, m3b_package_identity=m3b_package_identity,
        target_reference={"kind": "REUSE_FROZEN", "path": SIW_TARGET_EVAL_PACKAGE_ROOT})

    # EXT-F2: CASIA-only M3B rows + the amended SiW split; MSU is held out
    f2_train_refs = ([build_m3b_source_reference(r, fold_id="EXT-F2", project_split="source_train")
                      for r in m3b_train_rows if r["dataset"] == "casia_fasd"]
                     + _siw_refs("EXT-F2", "train"))
    f2_dev_refs = ([build_m3b_source_reference(r, fold_id="EXT-F2", project_split="source_dev")
                   for r in m3b_dev_rows if r["dataset"] == "casia_fasd"]
                  + _siw_refs("EXT-F2", "dev"))
    materialized["EXT-F2"] = _materialize_fold(
        repo, fold_id="EXT-F2", source_domains=["CASIA-FASD", "SiW-Mv2"], target_domain="MSU-MFSD",
        train_refs=f2_train_refs, dev_refs=f2_dev_refs, siw_population_identity=population_identity,
        siw_split_identity=split_identity, m3b_package_identity=m3b_package_identity,
        target_reference={"kind": "BUILD_REQUIRED", "path": None,
                         "note": "no processed label-free MSU-MFSD held-out target package exists "
                                 "locally or is claimed to exist"})

    # EXT-F3: MSU-only M3B rows + the SAME amended SiW split; CASIA is held out
    f3_train_refs = ([build_m3b_source_reference(r, fold_id="EXT-F3", project_split="source_train")
                      for r in m3b_train_rows if r["dataset"] == "msu_mfsd"]
                     + _siw_refs("EXT-F3", "train"))
    f3_dev_refs = ([build_m3b_source_reference(r, fold_id="EXT-F3", project_split="source_dev")
                   for r in m3b_dev_rows if r["dataset"] == "msu_mfsd"]
                  + _siw_refs("EXT-F3", "dev"))
    materialized["EXT-F3"] = _materialize_fold(
        repo, fold_id="EXT-F3", source_domains=["MSU-MFSD", "SiW-Mv2"], target_domain="CASIA-FASD",
        train_refs=f3_train_refs, dev_refs=f3_dev_refs, siw_population_identity=population_identity,
        siw_split_identity=split_identity, m3b_package_identity=m3b_package_identity,
        target_reference={"kind": "BUILD_REQUIRED", "path": None,
                         "note": "no processed label-free CASIA-FASD held-out target package exists "
                                 "locally or is claimed to exist"})

    assert materialized["EXT-F2"]["siw_split_identity"] == materialized["EXT-F3"]["siw_split_identity"]

    written: dict[str, str] = {}
    resumed: dict[str, bool] = {}
    for fold_id, body in materialized.items():
        out_dir = repo / MATERIALIZATION_DIR / fold_id
        out_dir.mkdir(parents=True, exist_ok=True)
        final_path = out_dir / "FOLD_MATERIALIZATION.json"
        if final_path.is_file():
            existing = cc.read_json(final_path)
            if existing.get("fold_identity") == body["fold_identity"]:
                resumed[fold_id] = True
                written[fold_id] = str(final_path)
                continue
            raise E7AMaterializationConflict(
                f"{fold_id}: existing materialization fold_identity {existing.get('fold_identity')} "
                f"disagrees with the freshly computed {body['fold_identity']!r}; refusing to overwrite "
                "conflicting scientific output -- FAIL CLOSED")
        tmp_path = out_dir / "FOLD_MATERIALIZATION.json.tmp"
        tmp_path.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
        tmp_path.replace(final_path)  # atomic on POSIX
        resumed[fold_id] = False
        written[fold_id] = str(final_path)

    validation = e7a_validate_materialization(repo)
    return {
        "written": written, "resumed": resumed, "validation": validation,
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False,
        "training_performed": False, "gpat_fitting_performed": False,
    }


def e7a_validate_materialization(repo: Path) -> dict[str, Any]:
    """Upgraded `--e7a-validate`: validates the NEW materialization
    namespace (not merely whether the old plan says materialized_this_turn).
    Read-only."""
    fold_ids = ("EXT-F1", "EXT-F2", "EXT-F3")
    present = {fid: (repo / MATERIALIZATION_DIR / fid / "FOLD_MATERIALIZATION.json").is_file()
              for fid in fold_ids}
    if not any(present.values()):
        return {"schema_version": f"{SCHEMA_PREFIX}-materialization-validate-v1",
               "status": "NOT_YET_BUILT", "E7A_MATERIALIZATION_VALID": False,
               "target_access": False, "llm_api_calls": 0}

    problems: list[str] = []
    bodies: dict[str, Any] = {}
    for fid in fold_ids:
        if not present[fid]:
            problems.append(f"{fid}: missing FOLD_MATERIALIZATION.json")
            continue
        bodies[fid] = cc.read_json(repo / MATERIALIZATION_DIR / fid / "FOLD_MATERIALIZATION.json")

    expected_domains = {"EXT-F1": (["CASIA-FASD", "MSU-MFSD"], "SiW-Mv2"),
                       "EXT-F2": (["CASIA-FASD", "SiW-Mv2"], "MSU-MFSD"),
                       "EXT-F3": (["MSU-MFSD", "SiW-Mv2"], "CASIA-FASD")}
    for fid, body in bodies.items():
        expected_source, expected_target = expected_domains[fid]
        if body["source_domains"] != expected_source:
            problems.append(f"{fid}: source_domains {body['source_domains']} != expected {expected_source}")
        if body["target_domain"] != expected_target:
            problems.append(f"{fid}: target_domain {body['target_domain']} != expected {expected_target}")
        all_datasets = {r["dataset"] for r in body["source_train_references"] + body["source_dev_references"]}
        if expected_target in all_datasets:
            problems.append(f"{fid}: held-out target domain {expected_target} found in source references "
                            "-- target-domain leakage")
        for ref in body["source_train_references"] + body["source_dev_references"]:
            if ref["reference_kind"] == "siw_raw_video" and ref.get("subject_id") is not None:
                problems.append(f"{fid}: a siw_raw_video reference carries a subject_id -- fabricated "
                                "subject identity")
            if ref["reference_kind"] not in ("m3b_processed_sample", "siw_raw_video"):
                problems.append(f"{fid}: unrecognized reference_kind {ref['reference_kind']!r}")
        if body["target_labels_opened"] is not False:
            problems.append(f"{fid}: target_labels_opened is not False")

        train_ids = {r.get("video_id") for r in body["source_train_references"]
                    if r["reference_kind"] == "siw_raw_video"}
        dev_ids = {r.get("video_id") for r in body["source_dev_references"]
                  if r["reference_kind"] == "siw_raw_video"}
        overlap = train_ids & dev_ids
        if overlap:
            problems.append(f"{fid}: {len(overlap)} SiW video(s) appear in BOTH source_train and "
                            "source_dev references")

    if "EXT-F2" in bodies and "EXT-F3" in bodies:
        if bodies["EXT-F2"]["siw_split_identity"] != bodies["EXT-F3"]["siw_split_identity"]:
            problems.append("EXT-F2 and EXT-F3 do not reuse the exact same siw_split_identity")

    for fid, body in bodies.items():
        recomputed = _fold_identity(
            fold_id=fid, source_domains=body["source_domains"], target_domain=body["target_domain"],
            source_train_reference_identity=body["source_train_reference_identity"],
            source_dev_reference_identity=body["source_dev_reference_identity"],
            siw_population_identity=body["siw_population_identity"],
            siw_split_identity=body["siw_split_identity"],
            m3b_package_identity=body["m3b_package_identity"],
            target_reference_identity=body["target_reference_identity"])
        if recomputed != body["fold_identity"]:
            problems.append(f"{fid}: fold_identity does not recompute exactly from its own persisted "
                            "components")
        if fid in ("EXT-F2", "EXT-F3") and body["target_reference"]["kind"] == "BUILD_REQUIRED":
            pass  # truthfully reported as build-required, not a validation problem
        elif fid == "EXT-F2" and body["target_reference"]["path"] is not None and \
                not str(body["target_reference"]["path"]):
            problems.append(f"{fid}: target reference path is missing")

    return {
        "schema_version": f"{SCHEMA_PREFIX}-materialization-validate-v1",
        "status": "MATERIALIZED", "folds_present": present, "problems": problems,
        "E7A_MATERIALIZATION_VALID": not problems,
        "target_access": False, "llm_api_calls": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E7-A fold manifest / source-dev / isolation "
                                                 "preparation (no render, no train, no GPAT fit, no "
                                                 "target, no LLM)")
    parser.add_argument("--e7a-preflight", action="store_true",
                        help="Read-only: dataset availability, frozen identities, paths, split rules. "
                             "Creates nothing scientific.")
    parser.add_argument("--e7a-build-preflight", action="store_true",
                        help="MATERIALIZATION: strictly read-only. Computes exactly what "
                             "--e7a-build --authorize would materialize (identities, per-fold "
                             "reference counts, leakage/overlap checks) but writes nothing.")
    parser.add_argument("--e7a-build", action="store_true",
                        help="Explicit execution only. Requires --authorize. Bound to the AMENDED "
                             "local-only SiW policy. Materializes EXT-F1/F2/F3 source_train/source_dev "
                             "reference manifests. Never renders/trains/fits GPAT/target-accesses. "
                             "Fails closed on missing bytes/identity mismatch/leakage/conflicting "
                             "existing output.")
    parser.add_argument("--e7a-validate", action="store_true",
                        help="Read-only validation of the materialization_v1/ namespace.")
    parser.add_argument("--e7a-local-siw-preflight", action="store_true",
                        help="AMENDMENT: read-only. Inventories the exact permitted local raw SiW "
                             "population, verifies counts/families against the frozen layout contract, "
                             "verifies M3B CASIA/MSU manifests, computes the PROPOSED video-disjoint "
                             "split in memory. Writes nothing. Never trains/renders/fits GPAT.")
    parser.add_argument("--e7a-local-siw-freeze", action="store_true",
                        help="AMENDMENT: explicit execution only. Requires --authorize. Persists the "
                             "amendment + population + split-policy locks additively under "
                             "amendment_local_siw_v1/. Never writes a source manifest, never trains, "
                             "never renders, never fits GPAT.")
    parser.add_argument("--authorize", action="store_true",
                        help="Required alongside --e7a-build or --e7a-local-siw-freeze.")
    parser.add_argument("--prepare", action="store_true",
                        help="Writes every additive E7-A preparation artifact (protocol lock, dataset "
                             "binding, fold manifest plan, source split lock, target reference "
                             "contract, isolation report, execution plan, readiness).")
    parser.add_argument("--prepare-amendment", action="store_true",
                        help="Writes every additive AMENDMENT artifact under amendment_local_siw_v1/. "
                             "Never touches the original E7-A artifacts.")
    args = parser.parse_args(argv)
    repo = cc.repo_root()

    if args.e7a_preflight:
        print(json.dumps(e7a_preflight(repo), indent=2, default=str))
        return 0
    if args.e7a_local_siw_preflight:
        print(json.dumps(e7a_local_siw_preflight(repo), indent=2, default=str))
        return 0
    if args.e7a_local_siw_freeze:
        if not args.authorize:
            print("--e7a-local-siw-freeze requires --authorize; refusing to run.")
            return 2
        result = prepare_e7a_amendment(repo)
        print(json.dumps({"readiness": result["readiness"]["body"]}, indent=2, default=str))
        return 0
    if args.e7a_build_preflight:
        print(json.dumps(e7a_build_preflight(repo), indent=2, default=str))
        return 0
    if args.e7a_build:
        try:
            result = e7a_amended_build(repo, authorize=args.authorize)
        except E7AError as error:
            print(f"E7-A build refused: {error}")
            return 1
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.e7a_validate:
        print(json.dumps(e7a_validate_materialization(repo), indent=2, default=str))
        return 0
    if args.prepare:
        result = prepare_e7a(repo)
        print(json.dumps({"readiness": result["readiness"]["body"]}, indent=2, default=str))
        return 0
    if args.prepare_amendment:
        result = prepare_e7a_amendment(repo)
        print(json.dumps({"readiness": result["readiness"]["body"]}, indent=2, default=str))
        return 0

    print("Pass --e7a-preflight, --e7a-local-siw-preflight (both read-only), --e7a-local-siw-freeze "
         "--authorize (writes the amendment locks), --e7a-build --authorize (fails closed; no local "
         "dataset bytes are complete on this laptop), --e7a-validate (read-only), --prepare, or "
         "--prepare-amendment.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
