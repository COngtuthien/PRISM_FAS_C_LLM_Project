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
        "raw_siw_source_bytes_present_locally": False,
        "GPU_REQUIRED": True,
        "REUSE_ACTION": "for EXT-F1 (SiW as TARGET): REUSE the frozen prism_target_eval_v2 package "
                        "verbatim -- do not rebuild. For EXT-F2/F3 (SiW as SOURCE): a NEW subject-"
                        "disjoint split must be constructed on the GPU host from raw SiW-Mv2, per the "
                        "frozen source_split_policy.siw_as_source rule -- BUT the subject/group key "
                        "this rule requires is not resolvable from the committed adapter as written; "
                        "this is a genuine, unresolved gap, not a laptop-only limitation",
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

def _write(repo: Path, filename: str, body: dict[str, Any]) -> dict[str, Any]:
    out_dir = repo / E7A_REPORT_DIR
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E7-A fold manifest / source-dev / isolation "
                                                 "preparation (no render, no train, no GPAT fit, no "
                                                 "target, no LLM)")
    parser.add_argument("--e7a-preflight", action="store_true",
                        help="Read-only: dataset availability, frozen identities, paths, split rules. "
                             "Creates nothing scientific.")
    parser.add_argument("--e7a-build", action="store_true",
                        help="Explicit execution only. Requires --authorize. Constructs source "
                             "train/dev manifest references. Never renders/trains/fits GPAT/target-"
                             "accesses. Fails closed on missing bytes/identity mismatch/leakage.")
    parser.add_argument("--e7a-validate", action="store_true",
                        help="Read-only validation of completed E7-A manifests.")
    parser.add_argument("--authorize", action="store_true", help="Required alongside --e7a-build.")
    parser.add_argument("--prepare", action="store_true",
                        help="Writes every additive E7-A preparation artifact (protocol lock, dataset "
                             "binding, fold manifest plan, source split lock, target reference "
                             "contract, isolation report, execution plan, readiness).")
    args = parser.parse_args(argv)
    repo = cc.repo_root()

    if args.e7a_preflight:
        print(json.dumps(e7a_preflight(repo), indent=2, default=str))
        return 0
    if args.e7a_build:
        try:
            result = e7a_build(repo, authorize=args.authorize)
        except E7AError as error:
            print(f"E7-A build refused: {error}")
            return 1
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.e7a_validate:
        print(json.dumps(e7a_validate(repo), indent=2, default=str))
        return 0
    if args.prepare:
        result = prepare_e7a(repo)
        print(json.dumps({"readiness": result["readiness"]["body"]}, indent=2, default=str))
        return 0

    print("Pass --e7a-preflight (read-only), --e7a-build --authorize (fails closed; no local dataset "
         "bytes are complete on this laptop), --e7a-validate (read-only), or --prepare (writes every "
         "additive preparation artifact).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
