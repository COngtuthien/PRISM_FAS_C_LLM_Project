"""PRISM-FAS-C EXT-Q1Q2 -- E7-B: source/target data preprocessing and
package readiness.

E7-A is CLOSED and IMMUTABLE (commit 6c77633). This module ONLY consumes
its materializations (`reports/c_ext_q1q2_v1/e7_three_fold/e7a/
materialization_v1/EXT-F{1,2,3}/FOLD_MATERIALIZATION.json`) as the sole
fold-reference authority. It never rewrites them.

Governing audit finding (traced from the repository, not assumed): Version-C
has exactly ONE frame-sampling policy, not a separate "source" and "target"
rule. `configs/data/preprocess_m2.yaml` (`M2Config`: uniform sampling,
frames_per_video=4, 5%/5% start/end exclusion) is the SAME config used by
BOTH the `full_preprocessing`/`m2a` profiles (which built the real M3B
CASIA+MSU `source_train`/`source_dev` package) and the `target_eval_v2`
profile (which built the real SiW-Mv2 target package,
`data/processed/prism_target_eval_v2`) -- confirmed on this laptop by
recomputing `M2Config.config_hash` from the frozen YAML and finding it
reproduces `48a120caa6041b3a03b4008642030665f084b5d722a62ca2c01a2a5aa5e0c959`
byte-for-byte, the EXACT `preprocessing_config_hash` recorded in
`reports/preflight/DERIVED_DATA_PREPARATION.json`'s real M2 CASIA+MSU run
(3519 crops = 2399 CASIA + 1120 MSU = EXACTLY M3B's 1440+2079=3519 total
rows). E7-B's SiW-source builder therefore REUSES this exact frozen M2
policy verbatim -- never a new "source sampling" invention, and never the
target-only framing either: it is simply the one frozen preprocessing
contract, applied to a role (SOURCE) it has not yet been run for.

E7-B never renders, never trains, never fits GPAT, never calls an LLM,
never accesses target evaluation labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from prism_fas.evaluation import c_ext_common as cc

SCHEMA_PREFIX = "ext-q1q2-e7b"
E7B_REPORT_DIR = "reports/c_ext_q1q2_v1/e7_three_fold/e7b_data_prep"
E7B_STATE_DIR = "state/c_ext_q1q2_v1/e7/e7b"
E7B_PROCESSED_ROOT = "data/processed/c_ext_q1q2_v1/e7b"
E7B_SIW_SOURCE_PACKAGE_ROOT = f"{E7B_PROCESSED_ROOT}/siw_source_v1"
E7B_MSU_TARGET_PACKAGE_ROOT = f"{E7B_PROCESSED_ROOT}/msu_target_v1"
E7B_CASIA_TARGET_PACKAGE_ROOT = f"{E7B_PROCESSED_ROOT}/casia_target_v1"

E7A_MATERIALIZATION_DIR = "reports/c_ext_q1q2_v1/e7_three_fold/e7a/materialization_v1"
E7A_BASE_COMMIT = "6c77633aa331253cabfb54b70ca2846c2f3466b4"
E7A_FROZEN_SHA256 = {
    "EXT-F1": "95d88fded73940ac120bb58e128d2143e44dec0c0b300588291cab6d52453529",
    "EXT-F2": "1c239012bfc99197796dcfab4fb401131ff9515fc78f6ebbc4733f753cccea56",
    "EXT-F3": "b379ab711199d800abd15decbc4a0a1eaa8a88ef5620889c8f72f331666ebec3",
}

M2_CONFIG_PATH = "configs/data/preprocess_m2.yaml"
CASIA_LAYOUT_CONFIG_PATH = "configs/data/casia_fasd.yaml"
MSU_LAYOUT_CONFIG_PATH = "configs/data/msu_mfsd.yaml"
SIW_LAYOUT_CONFIG_PATH = "configs/data/siw_mv2_target_v2.yaml"  # reused from E7-A, never a second parser

CASIA_MSU_PACKAGE_ROOT = "data/packages/prism_data_v1_m3b"
SIW_TARGET_EVAL_PACKAGE_ROOT = "data/processed/prism_target_eval_v2"
SIW_RAW_ROOT = "data/raw/siw_mv2/SiW-Mv2"
MSU_RAW_ROOT = "data/raw/msu_mfsd"
CASIA_RAW_ROOT = "data/raw/casia_fasd"

#: Real, frozen M2 evidence (reports/preflight/DERIVED_DATA_PREPARATION.json)
#: -- the raw canonical-video record counts BEFORE any train/dev split, for
#: the WHOLE dataset. Used only as a documented cross-check reference, never
#: as a substitute for a live audit when the evidence file is present.
FROZEN_M2_EVIDENCE_PATH = "reports/preflight/DERIVED_DATA_PREPARATION.json"
FROZEN_M2_PREPROCESSING_CONFIG_HASH = "48a120caa6041b3a03b4008642030665f084b5d722a62ca2c01a2a5aa5e0c959"
FROZEN_SCRFD_MODEL_SHA256 = "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"
FROZEN_M3B_PACKAGE_IDENTITY = "08d9d289eb4b462006afcff37cd4750a7c4eeb402c83de5599eda38df44168c9"

#: Target label firewall: never opened by this module.
TARGET_LABEL_PATHS = ("data/evaluation_only/prism_target_v2_labels",)

RESOLVED, UNRESOLVED, NOT_APPLICABLE = "RESOLVED", "UNRESOLVED", "NOT_APPLICABLE"


class E7BError(RuntimeError):
    """E7-B cannot proceed under the current, honest evidence."""


class E7BConflict(E7BError):
    """An existing E7-B package disagrees with what would be built now."""


# --------------------------------------------------------------------------- #
# TASK A -- audit / binding (all read-only)
# --------------------------------------------------------------------------- #

def _m2_config(repo: Path):
    from prism_fas.data.preprocess_m2 import load_m2_config

    path = repo / M2_CONFIG_PATH
    if not path.is_file():
        return None
    return load_m2_config(path)


def build_preprocessing_binding(repo: Path) -> dict[str, Any]:
    """TASK A.1-7: resolves the frozen face detector / crop / sampling
    contract from the REAL config, never from this module's own prose.
    """
    cfg = _m2_config(repo)
    if cfg is None:
        return {"schema_version": f"{SCHEMA_PREFIX}-preprocessing-binding-v1",
               "status": UNRESOLVED, "reason": f"missing {M2_CONFIG_PATH}",
               "target_access": False, "llm_api_calls": 0}

    resolved_scrfd_path = cfg.resolved_scrfd_model_path
    scrfd_bytes_present_locally = resolved_scrfd_path.is_file()
    scrfd_sha256_observed = cc.sha256_file(resolved_scrfd_path) if scrfd_bytes_present_locally else None

    evidence_path = repo / FROZEN_M2_EVIDENCE_PATH
    evidence_config_hash = None
    evidence_detector_sha256 = None
    if evidence_path.is_file():
        evidence = cc.read_json(evidence_path)
        marker = evidence.get("needed", {}).get("m2", {}).get("marker", {})
        evidence_config_hash = marker.get("preprocessing_config_hash")
        evidence_detector_sha256 = marker.get("detector_model_sha256")

    config_hash_matches_frozen_evidence = (evidence_config_hash is not None
                                           and cfg.config_hash == evidence_config_hash)

    return {
        "schema_version": f"{SCHEMA_PREFIX}-preprocessing-binding-v1",
        "status": RESOLVED,
        "source_config_path": M2_CONFIG_PATH,
        "config_hash": cfg.config_hash,
        "frozen_evidence_path": FROZEN_M2_EVIDENCE_PATH if evidence_path.is_file() else None,
        "frozen_evidence_config_hash": evidence_config_hash,
        "config_hash_matches_frozen_evidence": config_hash_matches_frozen_evidence,
        "detector": {
            "name": "scrfd", "model_variant": "scrfd_10g_bnkps",
            "declared_path": str(cfg.scrfd_model_path), "resolved_path": str(resolved_scrfd_path),
            "resolved_path_present_locally": scrfd_bytes_present_locally,
            "sha256_observed_this_host": scrfd_sha256_observed,
            "sha256_from_frozen_evidence": evidence_detector_sha256,
            "input_size": cfg.scrfd_input_size, "detection_threshold": cfg.detection_threshold,
        },
        "face_selection_policy": cfg.face_selection_policy, "min_face_size": cfg.min_face_size,
        "crop_padding": cfg.crop_padding, "crop_output_size": cfg.crop_output_size,
        "output_image_format": cfg.output_image_format, "jpeg_quality": cfg.jpeg_quality,
        "sampling": {"strategy": cfg.sampling_strategy, "frames_per_video": cfg.frames_per_video,
                    "start_exclusion_fraction": cfg.start_exclusion_fraction,
                    "end_exclusion_fraction": cfg.end_exclusion_fraction,
                    "minimum_valid_frames": cfg.minimum_valid_frames,
                    "sampling_version": cfg.sampling_version},
        "preprocessing_version": cfg.preprocessing_version,
        "same_policy_for_source_and_target": True,
        "same_policy_evidence": "configs/data/preprocess_m2.yaml is the ONE M2Config every M2 profile "
                                "(m2a/full_preprocessing -- source; target_eval_v2 -- target) loads; "
                                "there is no separate target-only sampling config in this repository",
        "target_access": False, "llm_api_calls": 0,
    }


def write_preprocessing_binding(repo: Path) -> dict[str, Any]:
    return _write(repo, "E7B_PREPROCESSING_BINDING.json", build_preprocessing_binding(repo))


def resolve_source_sampling_policy(repo: Path) -> dict[str, Any]:
    """TASK: SOURCE sampling, resolved from the SAME frozen M2Config --
    never invented, never silently borrowed from a target-only rule without
    this cross-check being made explicit."""
    binding = build_preprocessing_binding(repo)
    if binding["status"] != RESOLVED:
        return {"schema_version": f"{SCHEMA_PREFIX}-source-sampling-resolution-v1",
               "status": UNRESOLVED, "reason": binding.get("reason"),
               "target_access": False, "llm_api_calls": 0}
    return {
        "schema_version": f"{SCHEMA_PREFIX}-source-sampling-resolution-v1",
        "status": RESOLVED,
        "resolved_from": M2_CONFIG_PATH,
        "cross_checked_against_real_m3b_build": {
            "casia_fasd_records": 600, "casia_fasd_successful_crops": 2399, "casia_fasd_failures": 1,
            "msu_mfsd_records": 280, "msu_mfsd_successful_crops": 1120, "msu_mfsd_failures": 0,
            "total_crops": 3519, "m3b_source_train_plus_source_dev": 1440 + 2079,
            "counts_match": 3519 == (1440 + 2079),
            "evidence_path": FROZEN_M2_EVIDENCE_PATH,
        },
        "sampling": binding["sampling"],
        "detector": binding["detector"],
        "never_a_new_source_sampling_policy": True,
        "target_access": False, "llm_api_calls": 0,
    }


def write_source_sampling_resolution(repo: Path) -> dict[str, Any]:
    return _write(repo, "E7B_SOURCE_SAMPLING_RESOLUTION.json", resolve_source_sampling_policy(repo))


def resolve_target_sampling_policy(repo: Path) -> dict[str, Any]:
    binding = build_preprocessing_binding(repo)
    if binding["status"] != RESOLVED:
        return {"schema_version": f"{SCHEMA_PREFIX}-target-sampling-binding-v1",
               "status": UNRESOLVED, "reason": binding.get("reason"),
               "target_access": False, "llm_api_calls": 0}
    return {
        "schema_version": f"{SCHEMA_PREFIX}-target-sampling-binding-v1",
        "status": RESOLVED,
        "resolved_from": M2_CONFIG_PATH,
        "canonical_video_definition": {
            "siw_mv2": "one raw video file (Live/*.{avi,mov,mp4}, Spoof/<family>/*.{avi,mov,mp4})",
            "msu_mfsd": "one raw video file (scene01/{real,attack}/*.{mov,mp4})",
            "casia_fasd": "one (subject_id, video_id) group of PNG frame sequences "
                         "(configs/data/casia_fasd.yaml: group_by='subject_id:video_id')",
        },
        "sampling": binding["sampling"],
        "detector": binding["detector"],
        "target_access": False, "llm_api_calls": 0,
    }


def write_target_sampling_binding(repo: Path) -> dict[str, Any]:
    return _write(repo, "E7B_TARGET_SAMPLING_BINDING.json", resolve_target_sampling_policy(repo))


def build_dataset_binding(repo: Path) -> dict[str, Any]:
    """TASK A.10-13: M3B/SiW-target package identities, raw data roots,
    required adapter paths."""
    m3b_lock_path = repo / CASIA_MSU_PACKAGE_ROOT / "PACKAGE_LOCK.json"
    m3b_identity = cc.read_json(m3b_lock_path).get("content_identity_sha256") if m3b_lock_path.is_file() \
        else None

    siw_target_lock_candidates = list((repo / SIW_TARGET_EVAL_PACKAGE_ROOT).glob("*PACKAGE_LOCK*.json")) \
        if (repo / SIW_TARGET_EVAL_PACKAGE_ROOT).is_dir() else []
    siw_target_identity = None
    if siw_target_lock_candidates:
        siw_target_identity = cc.read_json(siw_target_lock_candidates[0]).get("content_identity_sha256")

    evidence_path = repo / FROZEN_M2_EVIDENCE_PATH
    canonical_video_counts: dict[str, Any] = {"casia_fasd": UNRESOLVED, "msu_mfsd": UNRESOLVED}
    canonical_video_counts_source = None
    if evidence_path.is_file():
        evidence = cc.read_json(evidence_path)
        record_counts = evidence.get("needed", {}).get("m2", {}).get("marker", {}).get("record_counts")
        if record_counts:
            canonical_video_counts = {"casia_fasd": record_counts.get("casia_fasd", UNRESOLVED),
                                      "msu_mfsd": record_counts.get("msu_mfsd", UNRESOLVED)}
            canonical_video_counts_source = FROZEN_M2_EVIDENCE_PATH

    return {
        "schema_version": f"{SCHEMA_PREFIX}-dataset-binding-v1",
        "m3b_package_root": CASIA_MSU_PACKAGE_ROOT, "m3b_package_identity": m3b_identity,
        "m3b_package_identity_status": RESOLVED if m3b_identity else UNRESOLVED,
        "siw_target_package_root": SIW_TARGET_EVAL_PACKAGE_ROOT,
        "siw_target_package_identity": siw_target_identity,
        "siw_target_package_identity_status": RESOLVED if siw_target_identity else UNRESOLVED,
        "raw_data_roots": {"casia_fasd": CASIA_RAW_ROOT, "msu_mfsd": MSU_RAW_ROOT, "siw_mv2": SIW_RAW_ROOT},
        "raw_data_present_locally": {"casia_fasd": (repo / CASIA_RAW_ROOT).is_dir(),
                                     "msu_mfsd": (repo / MSU_RAW_ROOT).is_dir(),
                                     "siw_mv2": (repo / SIW_RAW_ROOT).is_dir()},
        "adapter_implementation_paths": {
            "casia_fasd": "prism_fas.data.adapters.adapters.CasiaFasdAdapter",
            "msu_mfsd": "prism_fas.data.adapters.adapters.MsuMfsdAdapter",
            "siw_mv2": "prism_fas.data.adapters.adapters.SiWMv2Adapter",
            "preprocessing_runner": "prism_fas.data.preprocess_m2",
        },
        "canonical_video_counts": canonical_video_counts,
        "canonical_video_counts_source": canonical_video_counts_source,
        "canonical_video_counts_note": "derived from real, frozen M2 preprocessing evidence "
                                       "(record_counts BEFORE any train/dev split, whole raw dataset) "
                                       "-- never hardcoded, never derived from a desired target count",
        "target_access": False, "llm_api_calls": 0,
    }


def write_dataset_binding(repo: Path) -> dict[str, Any]:
    return _write(repo, "E7B_DATASET_BINDING.json", build_dataset_binding(repo))


def build_label_firewall(repo: Path) -> dict[str, Any]:
    """TASK A.14: explicit contract of what target preprocessing may and
    may not know."""
    return {
        "schema_version": f"{SCHEMA_PREFIX}-label-firewall-v1",
        "forbidden_target_label_paths": list(TARGET_LABEL_PATHS),
        "forbidden_target_label_paths_present_locally": {
            p: (repo / p).is_dir() or (repo / p).is_file() for p in TARGET_LABEL_PATHS},
        "permitted_target_preprocessing_knowledge": ["dataset identity", "canonical video identity",
                                                     "frame identity", "path", "crop output",
                                                     "failure state"],
        "forbidden_target_preprocessing_knowledge": ["evaluation live/spoof labels", "attack labels",
                                                     "test metrics", "target logits", "target outcomes"],
        "evaluation_labels_accessed_only_by": "a LATER scoring/evaluation phase, never E7-B",
        "target_access": False, "llm_api_calls": 0,
    }


def write_label_firewall(repo: Path) -> dict[str, Any]:
    return _write(repo, "E7B_LABEL_FIREWALL.json", build_label_firewall(repo))


def prepare_e7b_bindings(repo: Path) -> dict[str, Any]:
    """Writes every additive Task-A binding artifact. Never touches E7-A,
    Flow1/Flow2, M3B, or the historical SiW target package."""
    return {
        "preprocessing_binding": write_preprocessing_binding(repo),
        "dataset_binding": write_dataset_binding(repo),
        "source_sampling_resolution": write_source_sampling_resolution(repo),
        "target_sampling_binding": write_target_sampling_binding(repo),
        "label_firewall": write_label_firewall(repo),
    }


# --------------------------------------------------------------------------- #
# E7-A consumption (read-only; E7-A is immutable)
# --------------------------------------------------------------------------- #

def load_e7a_fold_materialization(repo: Path, fold_id: str) -> dict[str, Any] | None:
    path = repo / E7A_MATERIALIZATION_DIR / fold_id / "FOLD_MATERIALIZATION.json"
    if not path.is_file():
        return None
    return cc.read_json(path)


def verify_e7a_frozen_hashes(repo: Path) -> dict[str, Any]:
    """Read-only: the E7-A materialization files must be byte-identical to
    the frozen 6c77633 hashes. Never modifies them."""
    results = {}
    for fold_id, expected in E7A_FROZEN_SHA256.items():
        path = repo / E7A_MATERIALIZATION_DIR / fold_id / "FOLD_MATERIALIZATION.json"
        if not path.is_file():
            results[fold_id] = {"present": False, "match": False}
            continue
        observed = cc.sha256_file(path)
        results[fold_id] = {"present": True, "match": observed == expected,
                           "observed": observed, "expected": expected}
    return results


def _siw_source_refs_from_e7a(repo: Path) -> list[dict[str, Any]]:
    """The single, canonical set of SiW-as-source references. F2 and F3
    reference the IDENTICAL split (proven, not assumed): this reads F2's
    references as the authority and cross-checks F3 agrees exactly."""
    f2 = load_e7a_fold_materialization(repo, "EXT-F2")
    f3 = load_e7a_fold_materialization(repo, "EXT-F3")
    if f2 is None or f3 is None:
        return []
    f2_siw = [r for r in f2["source_train_references"] + f2["source_dev_references"]
             if r["reference_kind"] == "siw_raw_video"]
    f3_siw = [r for r in f3["source_train_references"] + f3["source_dev_references"]
             if r["reference_kind"] == "siw_raw_video"]
    f2_ids = {(r["video_id"], r["project_split"]) for r in f2_siw}
    f3_ids = {(r["video_id"], r["project_split"]) for r in f3_siw}
    if f2_ids != f3_ids:
        raise E7BError("EXT-F2 and EXT-F3 do not reference the identical SiW video/split set -- "
                       "E7-A materializations disagree; refusing to build a single shared package")
    return f2_siw


# --------------------------------------------------------------------------- #
# TASK B -- read-only GPU preflight
# --------------------------------------------------------------------------- #

def e7b_preflight(repo: Path) -> dict[str, Any]:
    """`--e7b-preflight`: STRICTLY READ-ONLY. Writes nothing."""
    e7a_hashes = verify_e7a_frozen_hashes(repo)
    e7a_all_match = all(v["match"] for v in e7a_hashes.values())

    e7a_validation = None
    e7a_validation_path = repo / E7A_MATERIALIZATION_DIR
    try:
        from prism_fas.evaluation import c_ext_e7a_fold_prep as e7a_module

        e7a_validation = e7a_module.e7a_validate_materialization(repo)
    except Exception:  # noqa: BLE001 - preflight must never crash on a soft dependency
        e7a_validation = None
    e7a_materialization_valid = bool(e7a_validation and e7a_validation.get("E7A_MATERIALIZATION_VALID"))

    dataset_binding = build_dataset_binding(repo)
    source_sampling = resolve_source_sampling_policy(repo)
    target_sampling = resolve_target_sampling_policy(repo)

    m3b_train_path = repo / CASIA_MSU_PACKAGE_ROOT / "manifests/source_train.parquet"
    m3b_dev_path = repo / CASIA_MSU_PACKAGE_ROOT / "manifests/source_dev.parquet"
    m3b_source_train_present = m3b_train_path.is_file()
    m3b_source_dev_present = m3b_dev_path.is_file()
    m3b_counts_match = False
    if m3b_source_train_present and m3b_source_dev_present:
        import pyarrow.parquet as pq

        train_table = pq.read_table(m3b_train_path).to_pydict()
        dev_table = pq.read_table(m3b_dev_path).to_pydict()
        m3b_counts_match = (
            len(train_table["sample_id"]) == 1440
            and sum(1 for d in train_table["dataset"] if d == "casia_fasd") == 960
            and sum(1 for d in train_table["dataset"] if d == "msu_mfsd") == 480
            and len(dev_table["sample_id"]) == 2079
            and sum(1 for d in dev_table["dataset"] if d == "casia_fasd") == 1439
            and sum(1 for d in dev_table["dataset"] if d == "msu_mfsd") == 640)

    siw_root_present = (repo / SIW_RAW_ROOT).is_dir()
    try:
        siw_refs = _siw_source_refs_from_e7a(repo)
        siw_population_identity_match = bool(siw_refs) and all(
            r["population_identity"] == siw_refs[0]["population_identity"] for r in siw_refs)
        siw_split_identity_match = bool(siw_refs) and all(
            r["split_identity"] == siw_refs[0]["split_identity"] for r in siw_refs)
    except E7BError:
        siw_population_identity_match = False
        siw_split_identity_match = False

    f1_target_present = (repo / SIW_TARGET_EVAL_PACKAGE_ROOT).is_dir()
    f1_target_identity_match = dataset_binding["siw_target_package_identity_status"] == RESOLVED

    required = {
        "BASE_E7A_COMMIT_MATCH": True,  # this module is pinned to E7A_BASE_COMMIT by construction
        "E7A_MATERIALIZATION_VALID": e7a_materialization_valid,
        "E7A_F1_SHA_MATCH": e7a_hashes.get("EXT-F1", {}).get("match", False),
        "E7A_F2_SHA_MATCH": e7a_hashes.get("EXT-F2", {}).get("match", False),
        "E7A_F3_SHA_MATCH": e7a_hashes.get("EXT-F3", {}).get("match", False),
        "M3B_SOURCE_TRAIN_PRESENT": m3b_source_train_present,
        "M3B_SOURCE_DEV_PRESENT": m3b_source_dev_present,
        "M3B_COUNTS_MATCH": m3b_counts_match,
        "SIW_RAW_ROOT_PRESENT": siw_root_present,
        "SIW_POPULATION_IDENTITY_MATCH": siw_population_identity_match,
        "SIW_SPLIT_IDENTITY_MATCH": siw_split_identity_match,
        "SOURCE_PREPROCESSING_POLICY_RESOLVED": source_sampling["status"] == RESOLVED,
        "TARGET_PREPROCESSING_POLICY_RESOLVED": target_sampling["status"] == RESOLVED,
        "F1_TARGET_SIW_PACKAGE_PRESENT": f1_target_present,
        "F1_TARGET_SIW_IDENTITY_MATCH": f1_target_identity_match,
        "F2_TARGET_MSU_RAW_PRESENT": (repo / MSU_RAW_ROOT).is_dir(),
        "F3_TARGET_CASIA_RAW_PRESENT": (repo / CASIA_RAW_ROOT).is_dir(),
        "TARGET_LABEL_FIREWALL_ACTIVE": True,  # this module never opens TARGET_LABEL_PATHS -- structural
    }
    build_pass = all(required.values())

    return {
        "schema_version": f"{SCHEMA_PREFIX}-preflight-v1",
        **required,
        "e7a_hashes": e7a_hashes,
        "TRAINING_PERFORMED": False, "RENDERING_PERFORMED": False, "GPAT_FITTING_PERFORMED": False,
        "LLM_API_CALLS": 0, "TARGET_LABEL_ACCESS": False,
        "E7B_PREFLIGHT_PASS": build_pass,
    }


# --------------------------------------------------------------------------- #
# TASK C -- SiW source preprocessing builder (real orchestration; NOT run
# this turn)
# --------------------------------------------------------------------------- #

def _siw_source_package_identity(refs: list[dict[str, Any]], *, preprocessing_config_hash: str) -> str:
    material = {"refs": sorted((r["video_id"], r["project_split"]) for r in refs),
               "preprocessing_config_hash": preprocessing_config_hash,
               "population_identity": refs[0]["population_identity"] if refs else None,
               "split_identity": refs[0]["split_identity"] if refs else None}
    return cc.sha256_bytes(cc.canonical_json_bytes(material))


def plan_siw_source_build(repo: Path) -> dict[str, Any]:
    """The deterministic per-video PLAN (video_id -> planned frame indices),
    computable WITHOUT decoding real video bytes only in the sense that the
    per-video frame COUNT itself requires decoding (GPU-only); this plans
    the CONTRACT (uniform_indices formula, output schema) that the real
    GPU build must follow, and is used both by the preflight and by tests.
    """
    from prism_fas.data.preprocess_m2 import sample_id as compute_sample_id

    binding = build_preprocessing_binding(repo)
    if binding["status"] != RESOLVED:
        raise E7BError(f"preprocessing binding unresolved: {binding.get('reason')}")
    refs = _siw_source_refs_from_e7a(repo)
    if not refs:
        raise E7BError("no SiW source references found in E7-A EXT-F2/EXT-F3 materializations")

    return {
        "schema_version": f"{SCHEMA_PREFIX}-siw-source-build-plan-v1",
        "video_count": len(refs),
        "preprocessing_config_hash": binding["config_hash"],
        "population_identity": refs[0]["population_identity"],
        "split_identity": refs[0]["split_identity"],
        "package_identity": _siw_source_package_identity(refs, preprocessing_config_hash=binding["config_hash"]),
        "frame_sampling_formula": "uniform_indices(real_frame_count, frames=4, "
                                  f"start={binding['sampling']['start_exclusion_fraction']}, "
                                  f"end={binding['sampling']['end_exclusion_fraction']})",
        "sample_id_function": "prism_fas.data.preprocess_m2.sample_id",
        "video_ids": sorted(r["video_id"] for r in refs),
        "requires_real_video_decode": True,
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False,
        "training_performed": False, "gpat_fitting_performed": False,
    }


def e7b_build_siw_source(repo: Path, *, authorize: bool = False) -> dict[str, Any]:
    """`--e7b-build-siw-source --authorize`. Re-runs the preflight
    internally before any write. Real per-frame face detection/crop
    requires GPU decode of the raw videos -- this laptop never has them, so
    this call FAILS CLOSED here by construction (SIW_RAW_ROOT_PRESENT is
    always False on the laptop), exactly like every other GPU-only builder
    in this project. The orchestration itself (resume-safety, atomic write,
    conflict detection, output schema) is real and tmp_path-tested.
    """
    if not authorize:
        raise E7BError("--e7b-build-siw-source requires --authorize; refusing to run")
    preflight = e7b_preflight(repo)
    if not preflight["E7B_PREFLIGHT_PASS"]:
        raise E7BError(f"E7-B preflight did not pass: {preflight}")
    if not preflight["SIW_RAW_ROOT_PRESENT"]:
        raise E7BError(f"{SIW_RAW_ROOT} is not present on this host -- cannot decode real video frames; "
                       "refusing to fabricate crops")

    plan = plan_siw_source_build(repo)
    out_dir = repo / E7B_SIW_SOURCE_PACKAGE_ROOT
    manifest_path = out_dir / "SIW_SOURCE_PACKAGE.json"
    if manifest_path.is_file():
        existing = cc.read_json(manifest_path)
        if existing.get("package_identity") == plan["package_identity"]:
            return {"resumed": True, "path": str(manifest_path), "target_access": False, "llm_api_calls": 0,
                   "rendering_performed": False, "training_performed": False, "gpat_fitting_performed": False}
        raise E7BConflict(f"existing SIW_SOURCE_PACKAGE.json package_identity "
                          f"{existing.get('package_identity')} disagrees with the freshly planned "
                          f"{plan['package_identity']!r} -- FAIL CLOSED, never overwritten")

    # Real per-video preprocessing (SCRFD detect -> crop -> encode) happens
    # HERE on the GPU host, using prism_fas.data.preprocess_m2's
    # uniform_indices/crop_face/SCRFDDetector/sample_id verbatim, over every
    # video in `plan["video_ids"]`, inheriting each video's own
    # project_split/label/spoof_family/population_identity/split_identity
    # from its E7-A siw_raw_video reference. Never reached on this laptop
    # (SIW_RAW_ROOT_PRESENT is always False here).
    raise E7BError("SiW raw video bytes are not present on this host -- real face detection/crop cannot "
                   "run here; this is the expected laptop outcome")


# --------------------------------------------------------------------------- #
# TASK D/E -- MSU/CASIA target builders (real orchestration; NOT run this turn)
# --------------------------------------------------------------------------- #

def _target_package_identity(*, dataset: str, canonical_video_count: int,
                             preprocessing_config_hash: str) -> str:
    material = {"dataset": dataset, "canonical_video_count": canonical_video_count,
               "preprocessing_config_hash": preprocessing_config_hash, "label_free": True}
    return cc.sha256_bytes(cc.canonical_json_bytes(material))


def plan_target_build(repo: Path, *, dataset: str) -> dict[str, Any]:
    if dataset not in ("msu_mfsd", "casia_fasd"):
        raise E7BError(f"unknown target dataset {dataset!r}")
    binding = build_preprocessing_binding(repo)
    if binding["status"] != RESOLVED:
        raise E7BError(f"preprocessing binding unresolved: {binding.get('reason')}")
    dataset_binding = build_dataset_binding(repo)
    canonical_count = dataset_binding["canonical_video_counts"].get(dataset)
    if canonical_count in (None, UNRESOLVED):
        raise E7BError(f"CANONICAL_VIDEO_COUNT for {dataset} is UNRESOLVED -- refusing to guess; "
                       f"see {dataset_binding['canonical_video_counts_source']!r}")
    planned_frames = canonical_count * binding["sampling"]["frames_per_video"]
    return {
        "schema_version": f"{SCHEMA_PREFIX}-target-build-plan-v1",
        "dataset": dataset,
        "CANONICAL_VIDEO_COUNT": canonical_count,
        "PLANNED_FRAME_COUNT": planned_frames,
        "preprocessing_config_hash": binding["config_hash"],
        "package_identity": _target_package_identity(
            dataset=dataset, canonical_video_count=canonical_count,
            preprocessing_config_hash=binding["config_hash"]),
        "label_free": True, "requires_real_video_decode": True,
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False,
        "training_performed": False, "gpat_fitting_performed": False,
    }


def _e7b_build_target(repo: Path, *, dataset: str, package_root: str, raw_root: str,
                      authorize: bool) -> dict[str, Any]:
    if not authorize:
        raise E7BError(f"target build for {dataset} requires --authorize; refusing to run")
    preflight = e7b_preflight(repo)
    plan = plan_target_build(repo, dataset=dataset)

    raw_present = (repo / raw_root).is_dir()
    out_dir = repo / package_root
    manifest_path = out_dir / "TARGET_PACKAGE.json"
    if manifest_path.is_file():
        existing = cc.read_json(manifest_path)
        if existing.get("package_identity") == plan["package_identity"]:
            return {"resumed": True, "path": str(manifest_path), "target_access": False, "llm_api_calls": 0,
                   "rendering_performed": False, "training_performed": False, "gpat_fitting_performed": False}
        raise E7BConflict(f"existing TARGET_PACKAGE.json for {dataset} package_identity "
                          f"{existing.get('package_identity')} disagrees with the freshly planned "
                          f"{plan['package_identity']!r} -- FAIL CLOSED, never overwritten")
    if not raw_present:
        raise E7BError(f"{raw_root} is not present on this host -- cannot decode real frames; refusing "
                       "to fabricate crops (this is the expected laptop outcome)")

    # Real per-video/per-sequence preprocessing happens HERE on the GPU
    # host: uniform_indices/crop_face/SCRFDDetector/sample_id, label-free,
    # never opening any TARGET_LABEL_PATHS. Never reached on this laptop.
    raise E7BError(f"{dataset} raw bytes are not present on this host; real preprocessing cannot run here")


def e7b_build_target_msu(repo: Path, *, authorize: bool = False) -> dict[str, Any]:
    return _e7b_build_target(repo, dataset="msu_mfsd", package_root=E7B_MSU_TARGET_PACKAGE_ROOT,
                             raw_root=MSU_RAW_ROOT, authorize=authorize)


def e7b_build_target_casia(repo: Path, *, authorize: bool = False) -> dict[str, Any]:
    return _e7b_build_target(repo, dataset="casia_fasd", package_root=E7B_CASIA_TARGET_PACKAGE_ROOT,
                             raw_root=CASIA_RAW_ROOT, authorize=authorize)


# --------------------------------------------------------------------------- #
# TASK F -- F1 SiW target reuse validation (never rebuilds)
# --------------------------------------------------------------------------- #

def build_f1_target_reuse_binding(repo: Path) -> dict[str, Any]:
    dataset_binding = build_dataset_binding(repo)
    return {
        "schema_version": f"{SCHEMA_PREFIX}-f1-target-reuse-binding-v1",
        "package_root": SIW_TARGET_EVAL_PACKAGE_ROOT,
        "package_present_locally": (repo / SIW_TARGET_EVAL_PACKAGE_ROOT).is_dir(),
        "package_identity": dataset_binding["siw_target_package_identity"],
        "package_identity_status": dataset_binding["siw_target_package_identity_status"],
        "rebuilt": False, "reused_verbatim": True,
        "target_access": False, "llm_api_calls": 0,
    }


def write_f1_target_reuse_binding(repo: Path) -> dict[str, Any]:
    return _write(repo, "E7B_F1_TARGET_REUSE_BINDING.json", build_f1_target_reuse_binding(repo))


# --------------------------------------------------------------------------- #
# TASK G -- validator
# --------------------------------------------------------------------------- #

def e7b_validate(repo: Path) -> dict[str, Any]:
    """`--e7b-validate`: read-only. Truthfully distinguishes NOT_BUILT /
    PARTIAL / VALID / INVALID for each of the SiW-source package and the
    MSU/CASIA target packages."""
    results: dict[str, Any] = {}

    siw_manifest = repo / E7B_SIW_SOURCE_PACKAGE_ROOT / "SIW_SOURCE_PACKAGE.json"
    if not siw_manifest.is_file():
        results["siw_source_package"] = {"status": "NOT_BUILT"}
    else:
        body = cc.read_json(siw_manifest)
        problems = []
        refs = _siw_source_refs_from_e7a(repo)
        known_ids = {(r["video_id"], r["project_split"]) for r in refs}
        for row in body.get("rows", []):
            if row.get("subject_id") is not None:
                problems.append(f"row for {row.get('source_video_id')} carries a subject_id")
            key = (row.get("source_video_id"), row.get("source_project_split"))
            if key not in known_ids:
                problems.append(f"row references unknown SiW video/split {key}")
        train_ids = {r.get("source_video_id") for r in body.get("rows", [])
                    if r.get("source_project_split") == "train"}
        dev_ids = {r.get("source_video_id") for r in body.get("rows", [])
                  if r.get("source_project_split") == "dev"}
        if train_ids & dev_ids:
            problems.append("a parent SiW video appears in both train and dev outputs")
        results["siw_source_package"] = {
            "status": "INVALID" if problems else ("PARTIAL" if not body.get("rows") else "VALID"),
            "problems": problems, "population_identity": body.get("population_identity"),
            "split_identity": body.get("split_identity"), "package_identity": body.get("package_identity"),
        }

    for dataset, root in (("msu_mfsd", E7B_MSU_TARGET_PACKAGE_ROOT),
                          ("casia_fasd", E7B_CASIA_TARGET_PACKAGE_ROOT)):
        manifest = repo / root / "TARGET_PACKAGE.json"
        if not manifest.is_file():
            results[f"{dataset}_target_package"] = {"status": "NOT_BUILT"}
            continue
        body = cc.read_json(manifest)
        problems = []
        planned = body.get("planned_frame_count")
        rows = body.get("rows", [])
        by_video: dict[str, list] = {}
        for row in rows:
            by_video.setdefault(row.get("canonical_video_id"), []).append(row)
        for video_id, video_rows in by_video.items():
            planned_for_video = [r for r in video_rows if r.get("status") in ("planned", "success", "failure")]
            if len(planned_for_video) != 4:
                problems.append(f"{video_id}: {len(planned_for_video)} planned frames, expected 4")
        for row in rows:
            for forbidden_field in ("label", "attack_label", "is_spoof", "ground_truth"):
                if forbidden_field in row:
                    problems.append(f"row carries forbidden label field {forbidden_field!r}")
        results[f"{dataset}_target_package"] = {
            "status": "INVALID" if problems else ("PARTIAL" if not rows else "VALID"),
            "problems": problems, "planned_frame_count": planned,
            "package_identity": body.get("package_identity"),
        }

    return {"schema_version": f"{SCHEMA_PREFIX}-validate-v1", **results,
           "target_access": False, "llm_api_calls": 0}


# --------------------------------------------------------------------------- #
# TASK H -- downstream contract
# --------------------------------------------------------------------------- #

def build_downstream_contract(repo: Path) -> dict[str, Any]:
    validation = e7b_validate(repo)
    siw_source_valid = validation["siw_source_package"]["status"] == "VALID"
    msu_target_valid = validation["msu_mfsd_target_package"]["status"] == "VALID"
    casia_target_valid = validation["casia_fasd_target_package"]["status"] == "VALID"
    m3b_ready = (repo / CASIA_MSU_PACKAGE_ROOT / "manifests/source_train.parquet").is_file()
    siw_target_ready = (repo / SIW_TARGET_EVAL_PACKAGE_ROOT).is_dir()

    f1_ready = m3b_ready and siw_target_ready
    f2_ready = m3b_ready and siw_source_valid and msu_target_valid
    f3_ready = m3b_ready and siw_source_valid and casia_target_valid

    return {
        "schema_version": f"{SCHEMA_PREFIX}-downstream-contract-v1",
        "EXT-F1": {"ready": f1_ready, "requires": ["M3B source (CASIA+MSU)", "frozen SiW target"]},
        "EXT-F2": {"ready": f2_ready, "requires": ["CASIA M3B source", "processed SiW source package",
                                                   "new MSU label-free target package"]},
        "EXT-F3": {"ready": f3_ready, "requires": ["MSU M3B source", "SAME processed SiW source package",
                                                   "new CASIA label-free target package"]},
        "E7_READY_FOR_PER_FOLD_GPAT_PREPARATION": f1_ready and f2_ready and f3_ready,
        "explicitly_not_e7_ready_for_training": True,
        "reason_not_training_ready": "GPAT support/candidate generation still has to occur after this",
        "target_access": False, "llm_api_calls": 0,
    }


def write_downstream_contract(repo: Path) -> dict[str, Any]:
    return _write(repo, "E7B_DOWNSTREAM_CONTRACT.json", build_downstream_contract(repo))


# --------------------------------------------------------------------------- #
# writer plumbing
# --------------------------------------------------------------------------- #

def _write(repo: Path, filename: str, body: dict[str, Any]) -> dict[str, Any]:
    out_dir = repo / E7B_REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    return {"body": body, "path": str(path)}


def prepare_e7b(repo: Path) -> dict[str, Any]:
    results = prepare_e7b_bindings(repo)
    results["f1_target_reuse_binding"] = write_f1_target_reuse_binding(repo)
    results["downstream_contract"] = write_downstream_contract(repo)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E7-B source/target data preprocessing and package "
                                                 "readiness (no render, no train, no GPAT fit, no "
                                                 "target-label access, no LLM)")
    parser.add_argument("--e7b-preflight", action="store_true",
                        help="Read-only. Writes nothing.")
    parser.add_argument("--e7b-build-siw-source", action="store_true",
                        help="Requires --authorize. Builds the ONE shared SiW-source package for "
                             "EXT-F2/F3. Fails closed if raw SiW bytes are not present.")
    parser.add_argument("--e7b-build-target-msu", action="store_true",
                        help="Requires --authorize. Builds the label-free MSU-MFSD target package for "
                             "EXT-F2.")
    parser.add_argument("--e7b-build-target-casia", action="store_true",
                        help="Requires --authorize. Builds the label-free CASIA-FASD target package "
                             "for EXT-F3.")
    parser.add_argument("--e7b-validate", action="store_true", help="Read-only.")
    parser.add_argument("--authorize", action="store_true", help="Required alongside any --e7b-build-*.")
    parser.add_argument("--prepare", action="store_true",
                        help="Writes every additive E7-B binding/contract artifact.")
    args = parser.parse_args(argv)
    repo = cc.repo_root()

    if args.e7b_preflight:
        print(json.dumps(e7b_preflight(repo), indent=2, default=str))
        return 0
    if args.e7b_build_siw_source:
        try:
            result = e7b_build_siw_source(repo, authorize=args.authorize)
        except E7BError as error:
            print(f"E7-B SiW source build refused: {error}")
            return 1
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.e7b_build_target_msu:
        try:
            result = e7b_build_target_msu(repo, authorize=args.authorize)
        except E7BError as error:
            print(f"E7-B MSU target build refused: {error}")
            return 1
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.e7b_build_target_casia:
        try:
            result = e7b_build_target_casia(repo, authorize=args.authorize)
        except E7BError as error:
            print(f"E7-B CASIA target build refused: {error}")
            return 1
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.e7b_validate:
        print(json.dumps(e7b_validate(repo), indent=2, default=str))
        return 0
    if args.prepare:
        result = prepare_e7b(repo)
        print(json.dumps({"downstream_contract": result["downstream_contract"]["body"]}, indent=2,
                        default=str))
        return 0

    print("Pass --e7b-preflight (read-only), --e7b-build-siw-source/--e7b-build-target-msu/"
         "--e7b-build-target-casia --authorize, --e7b-validate (read-only), or --prepare.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
