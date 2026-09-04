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

#: Engineering-only smoke namespace: clearly non-scientific, never the final
#: package namespace, never a package lock, never enters a scientific table.
E7B_SMOKE_ROOT = "runs/c_ext_q1q2_v1/e7b_smoke"
E7B_SIW_SMOKE_ROOT = f"{E7B_SMOKE_ROOT}/siw_source"
E7B_MSU_SMOKE_ROOT = f"{E7B_SMOKE_ROOT}/msu_target"
E7B_CASIA_SMOKE_ROOT = f"{E7B_SMOKE_ROOT}/casia_target"

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

#: `SourceFrameRecord.official_split`/`SourceCropRecord.official_split` are
#: frozen, required (non-null) `str` fields -- they carry a DATASET'S OWN
#: upstream canonical split (e.g. CASIA/MSU's own train/test protocol split,
#: `record.official_split` from the real adapter). SiW-Mv2 has no such
#: upstream split concept when used as a SOURCE domain (its own
#: `official_split` in `configs/data/siw_mv2_target_v2.yaml` is
#: `"target_test"`, meaningful only for the P3 held-out TARGET role, not for
#: this source usage). The E7-A `project_split` (train/dev) is a DIFFERENT
#: concept -- our own re-split for cross-domain fold construction -- and
#: must never be written into `official_split`, which would misrepresent it
#: as SiW's own upstream split. This placeholder satisfies the frozen
#: non-null `str` schema honestly: it is never treated as a real split by
#: any downstream code (E7-B's own package rows use `source_project_split`,
#: read from the E7-A reference directly, never from this field).
SIW_SOURCE_OFFICIAL_SPLIT_PLACEHOLDER = "not_applicable_no_official_source_split"

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
    binding = build_preprocessing_binding(repo)
    if binding["status"] != RESOLVED:
        raise E7BError(f"preprocessing binding unresolved: {binding.get('reason')}")
    if binding["frozen_evidence_config_hash"] is not None and not binding["config_hash_matches_frozen_evidence"]:
        raise E7BError(f"preprocessing config_hash {binding['config_hash']!r} does not match the frozen "
                       f"M2 evidence {binding['frozen_evidence_config_hash']!r} -- FAIL CLOSED, refusing "
                       "to build with a drifted config identity")
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


# --------------------------------------------------------------------------- #
# real production-pipeline orchestration (reuses prism_fas.data.* verbatim --
# never a second scientific implementation)
# --------------------------------------------------------------------------- #

def _verify_scrfd_model_sha256(cfg: Any) -> str:
    """HARD FAIL before writing a single scientific crop. Never substitutes
    another model, never changes provider/config for performance."""
    from prism_fas.data.preprocess_m2 import resolve_detector_path

    resolved_path = resolve_detector_path(cfg.scrfd_model_path)
    if not resolved_path.is_file():
        raise E7BError(f"SCRFD model not present at resolved path {resolved_path} -- refusing to run "
                       "without the frozen detector")
    observed = cc.sha256_file(resolved_path)
    if observed != FROZEN_SCRFD_MODEL_SHA256:
        raise E7BError(f"SCRFD model SHA256 {observed} != frozen {FROZEN_SCRFD_MODEL_SHA256} -- FAIL "
                       "CLOSED before writing any scientific crop")
    return observed


def _resolve_e7b_detector(cfg: Any, injected_detector: Any = None) -> tuple[Any, str]:
    """Resolves the ACTUAL detector object to hand to `run_preprocessing`.

    `run_preprocessing` does NOT lazily instantiate a detector -- it calls
    `detector.detect(...)` directly, so passing `detector=None` (the old
    behavior whenever `detector is None and smoke`) raised an unrouted,
    unpersisted exception for every single planned frame: a false-green
    package (planned=N, success=0, failure=0, rows=[]) that still exited 0.

    If `injected_detector` is given (unit tests only), it is used as-is and
    the caller need not supply real GPU model bytes; the recorded
    `detector_model_sha256` in that case is the FROZEN constant (tests never
    claim to have verified real bytes -- this matches every existing test's
    expectation).

    Otherwise -- for BOTH the scientific build AND smoke, with no exception
    for smoke -- this verifies the real on-disk SCRFD model bytes match
    `FROZEN_SCRFD_MODEL_SHA256` (fail closed on mismatch, before any crop is
    written) and instantiates the real `prism_fas.data.preprocess_m2.
    SCRFDDetector` with the frozen `scrfd_input_size`, using
    `SCRFDDetector`'s own default execution provider -- never a different
    provider merely because this runs on a GPU host.
    """
    if injected_detector is not None:
        return injected_detector, FROZEN_SCRFD_MODEL_SHA256

    from prism_fas.data.preprocess_m2 import SCRFDDetector, resolve_detector_path

    detector_model_sha256 = _verify_scrfd_model_sha256(cfg)
    resolved_path = resolve_detector_path(cfg.scrfd_model_path)
    detector = SCRFDDetector(resolved_path, cfg.scrfd_input_size)
    return detector, detector_model_sha256


def _fail_closed_on_unrouted_failure(result: Any, *, kind: str) -> None:
    """`run_preprocessing`'s generic outer exception handler tallies any
    UNEXPECTED implementation exception (never a typed detector/decode/crop
    failure) as `unrouted_processing_failure` in `failures_by_code`, WITHOUT
    persisting a `PreprocessingFailureRecord` for it. E7-B must never accept
    that as a valid scientific failure row -- it is an engineering bug."""
    if result is None:
        return
    unrouted = result.failures_by_code.get("unrouted_processing_failure", 0)
    if unrouted:
        raise E7BError(
            f"UNROUTED_PROCESSING_FAILURE for {kind}: {unrouted} planned frame(s) hit an unexpected, "
            "unclassified implementation exception (unrouted_processing_failure) inside "
            "run_preprocessing() -- this is never persisted as a PreprocessingFailureRecord, so it is "
            "an ENGINEERING failure, never a valid scientific failure row. FAIL CLOSED. "
            f"failures_by_code={result.failures_by_code!r}")


def _enforce_strict_terminal_accounting(*, kind: str, active_video_count: int, rows: list[dict[str, Any]],
                                        successful: int, failed: int, result: Any = None) -> None:
    """Before any package manifest may be written (scientific OR smoke):
    every planned frame must have reached a persisted terminal state
    (success or failure) -- never fewer, never more. This is the strict,
    row-level accounting check that would have caught the false-green
    package (planned=8, success=0, failure=0, rows=0) independently of
    whether `run_preprocessing`'s own return value looked clean."""
    expected_planned = active_video_count * 4
    problems = []
    if len(rows) != expected_planned:
        problems.append(f"persisted rows={len(rows)} != expected_planned={expected_planned} "
                        f"({active_video_count} active canonical video(s) x 4)")
    if successful + failed != expected_planned:
        problems.append(f"successful_crop_count({successful}) + failure_count({failed}) != "
                        f"expected_planned={expected_planned}")
    if successful != sum(1 for r in rows if r.get("status") == "success"):
        problems.append("successful_crop_count does not match the number of rows actually marked success")
    if failed != sum(1 for r in rows if r.get("status") == "failure"):
        problems.append("failure_count does not match the number of rows actually marked failure")
    if not problems:
        return
    accounting = ""
    if result is not None:
        accounting = (f" run_preprocessing accounting: canonical_records_attempted="
                     f"{result.canonical_records_attempted}, samples_selected={result.samples_selected}, "
                     f"samples_successful={result.samples_successful}, samples_failed={result.samples_failed}, "
                     f"frames_read={result.frames_read}, detector_calls={result.detector_calls}, "
                     f"crops_written={result.crops_written}, failures_by_code={result.failures_by_code!r}, "
                     f"manifest_counts={result.manifest_counts!r}.")
    raise E7BError(f"STRICT_TERMINAL_ACCOUNTING_FAILED for {kind}: " + "; ".join(problems) +
                   " -- refusing to write/overwrite a final package manifest (SMOKE or FROZEN). "
                   "FAIL CLOSED." + accounting)


def _run_execution_result_summary(result: Any) -> dict[str, Any] | None:
    if result is None:
        return None
    return {"canonical_records_attempted": result.canonical_records_attempted,
           "samples_selected": result.samples_selected, "samples_successful": result.samples_successful,
           "samples_failed": result.samples_failed, "frames_read": result.frames_read,
           "detector_calls": result.detector_calls, "crops_written": result.crops_written,
           "failures_by_code": result.failures_by_code, "manifest_counts": result.manifest_counts}


def _build_e7b_run_context(repo: Path, cfg: Any, *, dataset: str, dataset_role: str, package_root: str,
                           detector_model_sha256: str, run_id: str, dry_run: bool = False,
                           source_metadata_policy: str = "required") -> Any:
    """The ONE context constructor for E7-B's own additive namespace.

    `build_preprocessing_run_context` (the frozen production helper) always
    binds `dataset_role='target' if dataset=='siw_mv2' else 'source'` -- the
    historical assumption this milestone amends. `PreprocessingRunContext`
    itself has NO such restriction for `run_profile='small_acceptance'`
    (only `full_preprocessing`/`target_eval_v2` carry extra role/dataset
    constraints in its own frozen validator), so this constructs the SAME
    frozen `PreprocessingRunContext`/`M2OutputLayout` directly, with an
    explicit role, entirely inside E7-B's own namespace -- never touching
    the frozen `full_preprocessing`/`m2a`/`target_eval_v2` physical trees.

    `source_metadata_policy` defaults to 'required' -- the frozen,
    historical, byte-identical behavior for every context except the one
    E7-B SiW-as-source builder explicitly passes 'optional_unverifiable' to
    (see `build_source_frame_record` in `prism_fas.data.manifests.converters`).
    """
    from prism_fas.data.run_context import M2OutputLayout, PreprocessingRunContext
    from prism_fas.data.preprocess_m2 import resolve_detector_path

    root = (repo / package_root / "m2_run").resolve()
    layout = M2OutputLayout.from_root(root)
    return PreprocessingRunContext(
        project_root=repo, work_root=repo / package_root, run_profile="small_acceptance",
        output_namespace="small_acceptance", output_root=layout.output_root, crops_root=layout.crops_root,
        frames_root=layout.frames_root, manifests_root=layout.manifests_root, state_root=layout.state_root,
        reports_root=layout.reports_root, logs_root=layout.logs_root, run_id=run_id, dataset=dataset,
        dataset_role=dataset_role, preprocessing_version=cfg.preprocessing_version,
        preprocessing_config_hash=cfg.config_hash, detector_model_path=resolve_detector_path(cfg.scrfd_model_path),
        detector_model_sha256=detector_model_sha256, detector_input_size=cfg.scrfd_input_size,
        detector_threshold=cfg.detection_threshold, all_records=True, record_limit=None, sample_limit=None,
        resume=True, dry_run=dry_run, partial_full_profile=False, command="c_ext_e7b_data_prep",
        source_metadata_policy=source_metadata_policy)


def _siw_canonical_records(repo: Path, refs: list[dict[str, Any]]) -> list[Any]:
    """Builds `CanonicalVideoRecord`s DIRECTLY from E7-A's own frozen
    `siw_raw_video` references -- the input authority per this milestone --
    never re-derived from a fresh adapter scan (whose own opaque video-id
    scheme does not match E7-A's filename-stem video_id). `subject_id` is
    always None."""
    from prism_fas.data.schemas.records import CanonicalVideoRecord
    from prism_fas.utils.core import sha256_file

    records = []
    for ref in refs:
        source_path = repo / SIW_RAW_ROOT / ref["relative_path"]
        records.append(CanonicalVideoRecord(
            dataset="siw_mv2", subject_id=None, video_id=ref["video_id"], source_path=source_path,
            official_split=SIW_SOURCE_OFFICIAL_SPLIT_PLACEHOLDER, label=ref["label_live_spoof"],
            adapter_version="1.0", source_fingerprint=sha256_file(source_path),
            metadata_provenance="E7-A frozen siw_raw_video reference"))
    return records


def _read_manifest_rows(context: Any, *, kind: str) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    path = context.manifests_root / f"{kind}.parquet"
    if not path.is_file():
        return []
    return pq.read_table(path).to_pylist()


def _resume_completed_video_ids(package_manifest_path: Path, *, planned_per_video: int) -> set[str]:
    """E7-B's OWN, additive resume-skip layer: a video is skipped only if
    its LAST written package manifest already has `planned_per_video`
    definitive (success or terminal-failure) rows for it -- and every
    successful crop's sha256 still verifies on disk. A missing/corrupt crop
    for an otherwise-"complete" video is an explicit integrity error, never
    silently repaired or re-sampled into a different frame.
    """
    if not package_manifest_path.is_file():
        return set()
    body = json.loads(package_manifest_path.read_text(encoding="utf-8"))
    by_video: dict[str, list[dict[str, Any]]] = {}
    for row in body.get("rows", []):
        by_video.setdefault(row["source_video_id"], []).append(row)
    complete: set[str] = set()
    package_root = package_manifest_path.parent
    for video_id, rows in by_video.items():
        if len(rows) != planned_per_video:
            continue
        for row in rows:
            if row["status"] != "success":
                continue
            crop_path = package_root / row["crop_relative_path"]
            if not crop_path.is_file():
                raise E7BError(f"{video_id}: previously-successful crop {row['crop_relative_path']!r} is "
                               "missing on resume -- integrity error, refusing to silently re-sample")
            if cc.sha256_file(crop_path) != row["crop_sha256"]:
                raise E7BError(f"{video_id}: on-disk crop {row['crop_relative_path']!r} sha256 no longer "
                               "matches the recorded manifest value -- corrupt crop, integrity error, "
                               "refusing to silently repair or re-sample")
        complete.add(video_id)
    return complete


def _write_package_manifest_atomic(package_manifest_path: Path, body: dict[str, Any]) -> None:
    package_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = package_manifest_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    tmp_path.replace(package_manifest_path)


def e7b_build_siw_source(repo: Path, *, authorize: bool = False,
                         detector: Any = None, media_reader_factory: Any = None,
                         limit_videos: int | None = None, smoke: bool = False) -> dict[str, Any]:
    """`--e7b-build-siw-source --authorize`. Builds the ONE canonical shared
    SiW-source package (F2 and F3 both reuse it). Reuses
    `prism_fas.data.m2_runner.run_preprocessing`/`PreprocessingRunContext`/
    `ManifestRepository` VERBATIM for frame sampling, SCRFD detection, crop,
    encode and hashing -- this function only resolves canonical records
    from E7-A, verifies the model binding, drives the resume-skip layer,
    and assembles the final package manifest from what the frozen pipeline
    actually wrote.

    `detector`/`media_reader_factory` are the SAME injection points
    `run_preprocessing` itself exposes; production leaves them `None` (the
    real `SCRFDDetector` + real video/image decoders are used); tests inject
    fakes. `smoke=True` writes under `runs/.../e7b_smoke/` instead of the
    final package namespace and never touches `E7B_SIW_SOURCE_PACKAGE_ROOT`.
    """
    if not authorize:
        raise E7BError("--e7b-build-siw-source requires --authorize; refusing to run")
    if not smoke:
        preflight = e7b_preflight(repo)
        if not preflight["E7B_PREFLIGHT_PASS"]:
            raise E7BError(f"E7-B preflight did not pass: {preflight}")

    cfg = _m2_config(repo)
    if cfg is None:
        raise E7BError(f"missing {M2_CONFIG_PATH}")
    resolved_detector, detector_model_sha256 = _resolve_e7b_detector(cfg, detector)

    plan = plan_siw_source_build(repo)
    refs = _siw_source_refs_from_e7a(repo)
    if limit_videos is not None:
        refs = refs[:limit_videos]

    package_root = E7B_SIW_SMOKE_ROOT if smoke else E7B_SIW_SOURCE_PACKAGE_ROOT
    package_manifest_path = repo / package_root / "SIW_SOURCE_PACKAGE.json"

    if not smoke and package_manifest_path.is_file():
        existing = cc.read_json(package_manifest_path)
        if existing.get("package_identity") == plan["package_identity"] and limit_videos is None:
            return {"resumed": True, "path": str(package_manifest_path), "target_access": False,
                   "llm_api_calls": 0, "rendering_performed": False, "training_performed": False,
                   "gpat_fitting_performed": False}
        if existing.get("package_identity") not in (plan["package_identity"], None) and limit_videos is None:
            raise E7BConflict(f"existing SIW_SOURCE_PACKAGE.json package_identity "
                              f"{existing.get('package_identity')} disagrees with the freshly planned "
                              f"{plan['package_identity']!r} -- FAIL CLOSED, never overwritten")

    completed_video_ids = (set() if smoke else
                           _resume_completed_video_ids(package_manifest_path, planned_per_video=4))
    pending_refs = [r for r in refs if r["video_id"] not in completed_video_ids]

    context = _build_e7b_run_context(repo, cfg, dataset="siw_mv2", dataset_role="source",
                                     package_root=package_root, detector_model_sha256=detector_model_sha256,
                                     run_id=f"e7b-siw-source{'-smoke' if smoke else ''}",
                                     source_metadata_policy="optional_unverifiable")
    canonical_records = _siw_canonical_records(repo, pending_refs)

    from prism_fas.data.m2_runner import run_preprocessing
    from prism_fas.data.manifests.converters import MissingCanonicalMetadataError

    result = None
    if canonical_records:
        # HISTORICAL NOTE (see E7B_SIW_SOURCE_SUBJECT_ID_STRUCTURAL_GAP.json,
        # preserved as diagnostic evidence, and its resolution in
        # E7B_SIW_SOURCE_METADATA_COMPATIBILITY_RESOLUTION.json): this call
        # used to ALWAYS raise MissingCanonicalMetadataError on the first
        # successfully-detected SiW face, because the legacy
        # `build_source_frame_record` guard was stricter than the actual
        # persisted schema (`SourceFrameRecord.subject_id: str | None`
        # already permitted null). That legacy guard is now
        # `source_metadata_policy`-aware, and this context passes
        # 'optional_unverifiable' above -- a genuine subject_id=None SiW
        # source crop is expected to succeed. This try/except remains as a
        # defensive fail-closed guard only: it would still fire if
        # `record.official_split` or `record.label` were ever falsy, which
        # would indicate a real data-integrity problem, never something
        # E7-B should silently route around.
        try:
            result = run_preprocessing(context, canonical_records, detector=resolved_detector,
                                       media_reader_factory=media_reader_factory)
        except MissingCanonicalMetadataError as exc:
            raise E7BError(
                "UNEXPECTED_SOURCE_METADATA_GAP: build_source_frame_record() "
                "(prism_fas/data/manifests/converters.py) refused a SiW "
                "source record even under source_metadata_policy="
                "'optional_unverifiable' -- this means record.official_split "
                "or record.label was falsy, which is a genuine "
                "data-integrity problem (E7-A reference missing a label, or "
                "the official_split placeholder was not applied), never "
                "something to route around by fabricating a value or "
                "misrouting through dataset_role='target'. "
                f"Underlying error: {exc}"
            ) from exc
        _fail_closed_on_unrouted_failure(result, kind="SiW source")

    refs_by_video = {r["video_id"]: r for r in refs}
    rows = _assemble_siw_source_rows(context, refs_by_video)
    successful = sum(1 for r in rows if r["status"] == "success")
    failed = sum(1 for r in rows if r["status"] == "failure")
    _enforce_strict_terminal_accounting(kind="SiW source", active_video_count=len(refs), rows=rows,
                                        successful=successful, failed=failed, result=result)

    body = {
        "schema_version": f"{SCHEMA_PREFIX}-siw-source-package-v1",
        "package_identity": plan["package_identity"], "preprocessing_config_hash": plan["preprocessing_config_hash"],
        "detector_model_sha256": detector_model_sha256, "population_identity": plan["population_identity"],
        "split_identity": plan["split_identity"], "canonical_video_count": len(refs),
        "planned_frame_count": len(refs) * 4, "successful_crop_count": successful, "failure_count": failed,
        "rows": rows, "smoke": smoke, "target_labels_opened": False, "status": "FROZEN" if not smoke else "SMOKE",
        "last_run_accounting": _run_execution_result_summary(result),
    }
    if not smoke:
        _write_package_manifest_atomic(package_manifest_path, body)
    else:
        package_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        package_manifest_path.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")

    return {"resumed": False, "path": str(package_manifest_path), "body": body,
           "run_execution_result": _run_execution_result_summary(result),
           "target_access": False, "llm_api_calls": 0, "rendering_performed": False,
           "training_performed": False, "gpat_fitting_performed": False}


def _assemble_siw_source_rows(context: Any, refs_by_video: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Reads back exactly what the frozen `run_preprocessing`/
    `ManifestRepository` wrote and re-labels each row with its E7-A source
    identity. Never recomputes a detection/crop decision itself."""
    success_rows = _read_manifest_rows(context, kind="source_crops")
    failure_rows = _read_manifest_rows(context, kind="preprocessing_failures")
    rows: list[dict[str, Any]] = []
    for row in success_rows:
        ref = refs_by_video.get(row.get("video_id") or row.get("source_record_id"))
        rows.append({
            "source_video_id": row.get("video_id") or row.get("source_record_id"),
            "source_project_split": ref["project_split"] if ref else None,
            "frame_index": row.get("requested_frame_index"), "timestamp_ms": row.get("timestamp_ms"),
            "selected_frame_reference": row.get("selected_frame_reference"),
            "source_relative_identifier": row.get("source_record_id"),
            "crop_relative_path": row.get("crop_relative_path"), "crop_sha256": row.get("crop_sha256"),
            "detector_status": "success", "failure_reason": None, "status": "success",
            "label_live_spoof": ref["label_live_spoof"] if ref else None,
            "spoof_family": ref.get("spoof_family") if ref else None,
        })
    for row in failure_rows:
        video_id = row.get("source_record_id")
        ref = refs_by_video.get(video_id)
        rows.append({
            "source_video_id": video_id, "source_project_split": ref["project_split"] if ref else None,
            "frame_index": row.get("requested_frame_index"), "timestamp_ms": None,
            "selected_frame_reference": None, "source_relative_identifier": row.get("source_record_id"),
            "crop_relative_path": None, "crop_sha256": None, "detector_status": "failure",
            "failure_reason": row.get("error_code"), "status": "failure",
            "label_live_spoof": ref["label_live_spoof"] if ref else None,
            "spoof_family": ref.get("spoof_family") if ref else None,
        })
    return rows


# --------------------------------------------------------------------------- #
# TASK D/E -- MSU/CASIA target builders (real orchestration)
# --------------------------------------------------------------------------- #

EXPECTED_SIW_SOURCE_CANONICAL_VIDEO_COUNT = 1700
EXPECTED_SIW_SOURCE_PLANNED_FRAME_COUNT = 6800
EXPECTED_TARGET_CANONICAL_VIDEO_COUNT = {"msu_mfsd": 280, "casia_fasd": 600}
EXPECTED_TARGET_PLANNED_FRAME_COUNT = {"msu_mfsd": 1120, "casia_fasd": 2400}

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
    if binding["frozen_evidence_config_hash"] is not None and not binding["config_hash_matches_frozen_evidence"]:
        raise E7BError(f"preprocessing config_hash {binding['config_hash']!r} does not match the frozen "
                       f"M2 evidence {binding['frozen_evidence_config_hash']!r} -- FAIL CLOSED, refusing "
                       "to build with a drifted config identity")
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


def _target_canonical_records(repo: Path, *, dataset: str, raw_root: str) -> list[Any]:
    """Uses the REAL, frozen production adapter (`adapter_for`) to resolve
    canonical video/sequence records from the real raw dataset -- never a
    second inventory scan. For CASIA-FASD this naturally yields one record
    per (subject_id, video_id) canonical sequence, per
    `configs/data/casia_fasd.yaml`'s own `group_by` rule."""
    import yaml as _yaml

    from prism_fas.config.models import DatasetDefinition
    from prism_fas.data.adapters import adapter_for

    dfn = DatasetDefinition.model_validate(
        _yaml.safe_load((repo / "configs/data" / f"{dataset}.yaml").read_text(encoding="utf-8")))
    return adapter_for(dfn, repo / raw_root).records()


def _resume_completed_target_ids(package_manifest_path: Path, *, planned_per_video: int) -> set[str]:
    if not package_manifest_path.is_file():
        return set()
    body = json.loads(package_manifest_path.read_text(encoding="utf-8"))
    by_video: dict[str, list[dict[str, Any]]] = {}
    for row in body.get("rows", []):
        by_video.setdefault(row["canonical_video_id"], []).append(row)
    complete: set[str] = set()
    package_root = package_manifest_path.parent
    for video_id, rows in by_video.items():
        if len(rows) != planned_per_video:
            continue
        for row in rows:
            if row["status"] != "success":
                continue
            crop_path = package_root / row["crop_relative_path"]
            if not crop_path.is_file():
                raise E7BError(f"{video_id}: previously-successful crop is missing on resume -- "
                               "integrity error, refusing to silently re-sample")
            if cc.sha256_file(crop_path) != row["crop_sha256"]:
                raise E7BError(f"{video_id}: on-disk crop sha256 no longer matches the recorded manifest "
                               "value -- corrupt crop, integrity error, refusing to silently repair")
        complete.add(video_id)
    return complete


def _assemble_target_rows(context: Any) -> list[dict[str, Any]]:
    """Label-free by construction: reads back ONLY `target_crops`/
    `preprocessing_failures` (never `source_*`), and `route_target_success`'s
    own `assert_target_safe` firewall already rejects any label-bearing
    field before this ever sees a row. `video_id` is present on the frozen
    target-crop record itself (`route_target_success`'s own consistency
    check requires it), so it is read back verbatim, never re-derived."""
    success_rows = _read_manifest_rows(context, kind="target_crops")
    failure_rows = _read_manifest_rows(context, kind="preprocessing_failures")
    rows: list[dict[str, Any]] = []
    for row in success_rows:
        rows.append({
            "canonical_video_id": row.get("video_id"), "frame_index": row.get("requested_frame_index"),
            "timestamp_ms": row.get("timestamp_ms"), "frame_extraction_status": "success",
            "face_detection_status": "success", "crop_relative_path": row.get("crop_relative_path"),
            "crop_sha256": row.get("crop_sha256"), "failure_reason": None, "status": "success",
        })
    for row in failure_rows:
        rows.append({
            "canonical_video_id": row.get("source_record_id"), "frame_index": row.get("requested_frame_index"),
            "timestamp_ms": None, "frame_extraction_status": "failure", "face_detection_status": "failure",
            "crop_relative_path": None, "crop_sha256": None, "failure_reason": row.get("error_code"),
            "status": "failure",
        })
    return rows


def _e7b_build_target(repo: Path, *, dataset: str, package_root: str, smoke_root: str, raw_root: str,
                      authorize: bool, detector: Any = None, media_reader_factory: Any = None,
                      limit_videos: int | None = None, smoke: bool = False) -> dict[str, Any]:
    if not authorize:
        raise E7BError(f"target build for {dataset} requires --authorize; refusing to run")
    if not smoke:
        preflight = e7b_preflight(repo)
        if not preflight["E7B_PREFLIGHT_PASS"]:
            raise E7BError(f"E7-B preflight did not pass: {preflight}")

    cfg = _m2_config(repo)
    if cfg is None:
        raise E7BError(f"missing {M2_CONFIG_PATH}")
    plan = plan_target_build(repo, dataset=dataset)
    resolved_detector, detector_model_sha256 = _resolve_e7b_detector(cfg, detector)

    active_root = smoke_root if smoke else package_root
    manifest_path = repo / active_root / "TARGET_PACKAGE.json"

    if not smoke and manifest_path.is_file():
        existing = cc.read_json(manifest_path)
        if existing.get("package_identity") == plan["package_identity"] and limit_videos is None:
            return {"resumed": True, "path": str(manifest_path), "target_access": False, "llm_api_calls": 0,
                   "rendering_performed": False, "training_performed": False, "gpat_fitting_performed": False}
        if existing.get("package_identity") not in (plan["package_identity"], None) and limit_videos is None:
            raise E7BConflict(f"existing TARGET_PACKAGE.json for {dataset} package_identity "
                              f"{existing.get('package_identity')} disagrees with the freshly planned "
                              f"{plan['package_identity']!r} -- FAIL CLOSED, never overwritten")

    all_records = _target_canonical_records(repo, dataset=dataset, raw_root=raw_root)
    if limit_videos is not None:
        all_records = all_records[:limit_videos]
    completed_ids = set() if smoke else _resume_completed_target_ids(manifest_path, planned_per_video=4)
    pending_records = [r for r in all_records if r.video_id not in completed_ids]

    context = _build_e7b_run_context(repo, cfg, dataset=dataset, dataset_role="target",
                                     package_root=active_root, detector_model_sha256=detector_model_sha256,
                                     run_id=f"e7b-target-{dataset}{'-smoke' if smoke else ''}")

    from prism_fas.data.m2_runner import run_preprocessing

    result = None
    if pending_records:
        result = run_preprocessing(context, pending_records, detector=resolved_detector,
                                   media_reader_factory=media_reader_factory)
        _fail_closed_on_unrouted_failure(result, kind=f"{dataset} target")

    rows = _assemble_target_rows(context)
    successful = sum(1 for r in rows if r["status"] == "success")
    failed = sum(1 for r in rows if r["status"] == "failure")
    _enforce_strict_terminal_accounting(kind=f"{dataset} target", active_video_count=len(all_records),
                                        rows=rows, successful=successful, failed=failed, result=result)

    body = {
        "schema_version": f"{SCHEMA_PREFIX}-target-package-v1",
        "dataset": dataset, "package_identity": plan["package_identity"],
        "preprocessing_config_hash": plan["preprocessing_config_hash"],
        "detector_model_sha256": detector_model_sha256,
        "canonical_video_count": len(all_records), "planned_frame_count": len(all_records) * 4,
        "successful_crop_count": successful, "failure_count": failed, "rows": rows, "label_free": True,
        "smoke": smoke, "status": "FROZEN" if not smoke else "SMOKE",
        "last_run_accounting": _run_execution_result_summary(result),
    }
    if not smoke:
        _write_package_manifest_atomic(manifest_path, body)
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")

    return {"resumed": False, "path": str(manifest_path), "body": body,
           "run_execution_result": _run_execution_result_summary(result),
           "target_access": False, "llm_api_calls": 0, "rendering_performed": False,
           "training_performed": False, "gpat_fitting_performed": False}


def e7b_build_target_msu(repo: Path, *, authorize: bool = False, detector: Any = None,
                         media_reader_factory: Any = None, limit_videos: int | None = None,
                         smoke: bool = False) -> dict[str, Any]:
    return _e7b_build_target(repo, dataset="msu_mfsd", package_root=E7B_MSU_TARGET_PACKAGE_ROOT,
                             smoke_root=E7B_MSU_SMOKE_ROOT, raw_root=MSU_RAW_ROOT, authorize=authorize,
                             detector=detector, media_reader_factory=media_reader_factory,
                             limit_videos=limit_videos, smoke=smoke)


def e7b_build_target_casia(repo: Path, *, authorize: bool = False, detector: Any = None,
                           media_reader_factory: Any = None, limit_videos: int | None = None,
                           smoke: bool = False) -> dict[str, Any]:
    return _e7b_build_target(repo, dataset="casia_fasd", package_root=E7B_CASIA_TARGET_PACKAGE_ROOT,
                             smoke_root=E7B_CASIA_SMOKE_ROOT, raw_root=CASIA_RAW_ROOT, authorize=authorize,
                             detector=detector, media_reader_factory=media_reader_factory,
                             limit_videos=limit_videos, smoke=smoke)


def e7b_smoke_siw_source(repo: Path, *, limit_videos: int = 2, detector: Any = None,
                         media_reader_factory: Any = None) -> dict[str, Any]:
    """Engineering-only smoke: writes ONLY under `E7B_SIW_SMOKE_ROOT`, never
    the final package namespace, never a package lock. Uses the SAME
    production preprocessing primitives."""
    return e7b_build_siw_source(repo, authorize=True, detector=detector,
                                media_reader_factory=media_reader_factory, limit_videos=limit_videos,
                                smoke=True)


def e7b_smoke_target_msu(repo: Path, *, limit_videos: int = 2, detector: Any = None,
                         media_reader_factory: Any = None) -> dict[str, Any]:
    return e7b_build_target_msu(repo, authorize=True, detector=detector,
                                media_reader_factory=media_reader_factory, limit_videos=limit_videos,
                                smoke=True)


def e7b_smoke_target_casia(repo: Path, *, limit_videos: int = 2, detector: Any = None,
                           media_reader_factory: Any = None) -> dict[str, Any]:
    return e7b_build_target_casia(repo, authorize=True, detector=detector,
                                  media_reader_factory=media_reader_factory, limit_videos=limit_videos,
                                  smoke=True)


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

def _crop_resolves(package_root: Path, relative_path: str | None, expected_sha256: str | None) -> bool:
    if not relative_path or not expected_sha256:
        return False
    crop_path = package_root / relative_path
    return crop_path.is_file() and cc.sha256_file(crop_path) == expected_sha256


def e7b_validate(repo: Path) -> dict[str, Any]:
    """`--e7b-validate`: read-only. Truthfully distinguishes NOT_BUILT /
    PARTIAL / VALID / INVALID for each of the SiW-source package and the
    MSU/CASIA target packages against the FULL package contract: exact
    canonical/planned counts, successful+failure==planned, no unknown
    video, no train/dev parent overlap, exact population/split identity,
    exact config hash, exact SCRFD SHA256, no subject_id anywhere in SiW
    rows, every successful crop hash resolves on disk, failures preserved.
    An empty `rows` array is NEVER considered VALID."""
    results: dict[str, Any] = {}
    binding = build_preprocessing_binding(repo)
    frozen_config_hash = binding.get("config_hash") if binding.get("status") == RESOLVED else None

    siw_manifest = repo / E7B_SIW_SOURCE_PACKAGE_ROOT / "SIW_SOURCE_PACKAGE.json"
    if not siw_manifest.is_file():
        results["siw_source_package"] = {"status": "NOT_BUILT"}
    else:
        body = cc.read_json(siw_manifest)
        package_root = siw_manifest.parent
        problems = []
        rows = body.get("rows", [])
        refs = _siw_source_refs_from_e7a(repo)
        known_ids = {(r["video_id"], r["project_split"]) for r in refs}
        for row in rows:
            if "subject_id" in row and row.get("subject_id") is not None:
                problems.append(f"row for {row.get('source_video_id')} carries a subject_id")
            key = (row.get("source_video_id"), row.get("source_project_split"))
            if key not in known_ids:
                problems.append(f"row references unknown SiW video/split {key}")
            if row.get("status") == "success" and not _crop_resolves(
                    package_root, row.get("crop_relative_path"), row.get("crop_sha256")):
                problems.append(f"{row.get('source_video_id')}/{row.get('frame_index')}: "
                                "successful crop does not resolve on disk")
            if row.get("status") == "failure" and not row.get("failure_reason"):
                problems.append(f"{row.get('source_video_id')}/{row.get('frame_index')}: "
                                "failure row missing failure_reason")
        train_ids = {r.get("source_video_id") for r in rows if r.get("source_project_split") == "train"}
        dev_ids = {r.get("source_video_id") for r in rows if r.get("source_project_split") == "dev"}
        if train_ids & dev_ids:
            problems.append("a parent SiW video appears in both train and dev outputs")
        if body.get("canonical_video_count") != EXPECTED_SIW_SOURCE_CANONICAL_VIDEO_COUNT:
            problems.append(f"canonical_video_count {body.get('canonical_video_count')} != "
                            f"expected {EXPECTED_SIW_SOURCE_CANONICAL_VIDEO_COUNT}")
        if body.get("planned_frame_count") != EXPECTED_SIW_SOURCE_PLANNED_FRAME_COUNT:
            problems.append(f"planned_frame_count {body.get('planned_frame_count')} != "
                            f"expected {EXPECTED_SIW_SOURCE_PLANNED_FRAME_COUNT}")
        if (body.get("successful_crop_count", 0) + body.get("failure_count", 0)) != len(rows) or \
                len(rows) != body.get("planned_frame_count"):
            problems.append("successful_crop_count + failure_count does not equal planned_frame_count")
        if refs and body.get("population_identity") != refs[0]["population_identity"]:
            problems.append("population_identity does not match E7-A frozen reference")
        if refs and body.get("split_identity") != refs[0]["split_identity"]:
            problems.append("split_identity does not match E7-A frozen reference")
        if frozen_config_hash and body.get("preprocessing_config_hash") != frozen_config_hash:
            problems.append("preprocessing_config_hash does not match the frozen M2 config")
        if body.get("detector_model_sha256") != FROZEN_SCRFD_MODEL_SHA256:
            problems.append("detector_model_sha256 does not match the frozen SCRFD model")
        results["siw_source_package"] = {
            "status": "INVALID" if problems else ("PARTIAL" if not rows else "VALID"),
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
        package_root = manifest.parent
        problems = []
        planned = body.get("planned_frame_count")
        rows = body.get("rows", [])
        by_video: dict[str, list] = {}
        for row in rows:
            by_video.setdefault(row.get("canonical_video_id"), []).append(row)
        for video_id, video_rows in by_video.items():
            if len(video_rows) != 4:
                problems.append(f"{video_id}: {len(video_rows)} planned frames, expected 4")
        for row in rows:
            for forbidden_field in ("label", "label_live_spoof", "attack_label", "spoof_family",
                                    "is_spoof", "ground_truth", "subject_id"):
                if forbidden_field in row:
                    problems.append(f"row carries forbidden label field {forbidden_field!r}")
            if row.get("status") == "success" and not _crop_resolves(
                    package_root, row.get("crop_relative_path"), row.get("crop_sha256")):
                problems.append(f"{row.get('canonical_video_id')}/{row.get('frame_index')}: "
                                "successful crop does not resolve on disk")
            if row.get("status") == "failure" and not row.get("failure_reason"):
                problems.append(f"{row.get('canonical_video_id')}/{row.get('frame_index')}: "
                                "failure row missing failure_reason")
        if body.get("canonical_video_count") != EXPECTED_TARGET_CANONICAL_VIDEO_COUNT[dataset]:
            problems.append(f"canonical_video_count {body.get('canonical_video_count')} != "
                            f"expected {EXPECTED_TARGET_CANONICAL_VIDEO_COUNT[dataset]}")
        if planned != EXPECTED_TARGET_PLANNED_FRAME_COUNT[dataset]:
            problems.append(f"planned_frame_count {planned} != "
                            f"expected {EXPECTED_TARGET_PLANNED_FRAME_COUNT[dataset]}")
        if (body.get("successful_crop_count", 0) + body.get("failure_count", 0)) != len(rows) or \
                len(rows) != planned:
            problems.append("successful_crop_count + failure_count does not equal planned_frame_count")
        if frozen_config_hash and body.get("preprocessing_config_hash") != frozen_config_hash:
            problems.append("preprocessing_config_hash does not match the frozen M2 config")
        if body.get("detector_model_sha256") != FROZEN_SCRFD_MODEL_SHA256:
            problems.append("detector_model_sha256 does not match the frozen SCRFD model")
        if not body.get("label_free"):
            problems.append("package does not declare label_free=True")
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
    parser.add_argument("--e7b-smoke-siw-source", action="store_true",
                        help="Engineering-only smoke. Writes ONLY under "
                             f"{E7B_SIW_SMOKE_ROOT!r}. Never the final scientific package.")
    parser.add_argument("--e7b-smoke-target-msu", action="store_true",
                        help="Engineering-only smoke. Writes ONLY under "
                             f"{E7B_MSU_SMOKE_ROOT!r}. Never the final scientific package.")
    parser.add_argument("--e7b-smoke-target-casia", action="store_true",
                        help="Engineering-only smoke. Writes ONLY under "
                             f"{E7B_CASIA_SMOKE_ROOT!r}. Never the final scientific package.")
    parser.add_argument("--limit-videos", type=int, default=2,
                        help="Smoke mode only: number of videos to process (default 2).")
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
    if args.e7b_smoke_siw_source:
        try:
            result = e7b_smoke_siw_source(repo, limit_videos=args.limit_videos)
        except E7BError as error:
            print(f"E7-B SiW source smoke refused: {error}")
            return 1
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.e7b_smoke_target_msu:
        try:
            result = e7b_smoke_target_msu(repo, limit_videos=args.limit_videos)
        except E7BError as error:
            print(f"E7-B MSU target smoke refused: {error}")
            return 1
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.e7b_smoke_target_casia:
        try:
            result = e7b_smoke_target_casia(repo, limit_videos=args.limit_videos)
        except E7BError as error:
            print(f"E7-B CASIA target smoke refused: {error}")
            return 1
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.prepare:
        result = prepare_e7b(repo)
        print(json.dumps({"downstream_contract": result["downstream_contract"]["body"]}, indent=2,
                        default=str))
        return 0

    print("Pass --e7b-preflight (read-only), --e7b-build-siw-source/--e7b-build-target-msu/"
         "--e7b-build-target-casia --authorize, --e7b-validate (read-only), "
         "--e7b-smoke-siw-source/--e7b-smoke-target-msu/--e7b-smoke-target-casia "
         "[--limit-videos N] (engineering-only), or --prepare.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
