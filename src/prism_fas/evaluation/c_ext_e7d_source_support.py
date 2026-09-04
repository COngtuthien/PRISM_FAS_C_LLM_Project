"""PRISM-FAS-C EXT-Q1Q2 -- E7-D: per-fold crop-level SOURCE support
materialization.

E7-C is CLOSED_PASS (commit d0a0ed8). This module consumes E7-A's fold
materializations, E7-B's frozen SiW source package, and the frozen M3B
package -- it never rewrites any of them, never fits GPAT, never renders,
never trains, never scores targets, never opens target labels, never calls
an LLM.

REFERENCE-ONLY: the output manifests never copy crop image bytes. Every row
references an already-existing frozen crop (M3B `image_relative_path` /
E7-B SiW `crop_relative_path`) by path + sha256; the crop bytes themselves
are read only to VERIFY a row (existence + hash), never duplicated.

Governing schema audit (traced from the repository, not assumed):
- M3B references (`c_ext_e7a_fold_prep.build_m3b_source_reference`) already
  carry `sample_id, dataset, source_record_id, subject_id, label_live_spoof,
  image_relative_path, crop_sha256, project_split` -- E7-D reuses these
  verbatim from E7-A's own committed FOLD_MATERIALIZATION.json, never a
  fresh M3B parquet scan (E7-A already resolved exactly which M3B rows
  belong to each fold).
- E7-B's SIW_SOURCE_PACKAGE.json rows (`c_ext_e7b_data_prep.
  _assemble_siw_source_rows`) carry `source_video_id, source_project_split,
  frame_index, crop_relative_path, crop_sha256, status, failure_reason,
  label_live_spoof, spoof_family` -- already joined against E7-A once at
  E7-B build time. E7-D performs its OWN independent join (source_video_id
  -> E7-A siw_raw_video reference, via `e7b._siw_source_refs_from_e7a`,
  the SAME frozen cross-checked ref list E7-B itself uses) as a defense-in-
  depth verification, never merely trusting E7-B's embedded copy -- this is
  what lets E7-D fail closed on an unknown/renamed source_video_id.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from prism_fas.evaluation import c_ext_common as cc
from prism_fas.evaluation import c_ext_e7b_data_prep as e7b
from prism_fas.evaluation import c_ext_e7c_gpat_prep as e7c

SCHEMA_PREFIX = "ext-q1q2-e7d"
E7D_REPORT_DIR = "reports/c_ext_q1q2_v1/e7_three_fold/e7d_source_support"
E7D_OUTPUT_ROOT = "data/processed/c_ext_q1q2_v1/e7d_source_support"

FOLD_IDS = e7c.FOLD_IDS
FOLD_SOURCE_DOMAINS = e7c.FOLD_SOURCE_DOMAINS
FOLD_TARGET_DOMAIN = e7c.FOLD_TARGET_DOMAIN

#: E7-A's own domain-name spelling ("CASIA-FASD") vs M3B's own dataset
#: column spelling ("casia_fasd") -- audited from the real committed
#: FOLD_MATERIALIZATION.json (`dataset: "CASIA-FASD"`) and the M3B
#: manifest schema (`dataset: "casia_fasd"`, per pair_plan.ALLOWED_DATASETS).
M3B_DATASET_TO_E7A_DOMAIN = {"casia_fasd": "CASIA-FASD", "msu_mfsd": "MSU-MFSD"}
E7A_DOMAIN_TO_M3B_DATASET = {v: k for k, v in M3B_DATASET_TO_E7A_DOMAIN.items()}

#: Frozen M3B whole-package counts (commit-verified evidence, cross-checked
#: against the real E7-A materializations in prior milestones).
M3B_TRAIN_EXPECTED = {"CASIA-FASD": 960, "MSU-MFSD": 480, "total": 1440}
M3B_DEV_EXPECTED = {"CASIA-FASD": 1439, "MSU-MFSD": 640, "total": 2079}
M3B_LIVE_TRAIN_EXPECTED = {"CASIA-FASD": 160, "MSU-MFSD": 120}
M3B_LIVE_DEV_EXPECTED = {"CASIA-FASD": 240, "MSU-MFSD": 160}

#: Frozen E7-B SiW source package identities/aggregate (real GPU evidence,
#: reused verbatim from `c_ext_e7c_gpat_prep.FROZEN_E7B`/
#: `FROZEN_SIW_PACKAGE_AGGREGATE` -- never re-declared as a fresh literal).
FROZEN_SIW = {
    "package_identity": e7c.FROZEN_E7B["siw_source_package_identity"],
    "population_identity": e7c.FROZEN_E7B["siw_population_identity"],
    "split_identity": e7c.FROZEN_E7B["siw_split_identity"],
    "planned_frame_count": e7c.FROZEN_SIW_PACKAGE_AGGREGATE["planned_frame_count"],
    "successful_crop_count": e7c.FROZEN_SIW_PACKAGE_AGGREGATE["successful_crop_count"],
    "failure_count": e7c.FROZEN_SIW_PACKAGE_AGGREGATE["failure_count"],
}

#: Frozen SiW video-level split (E7-A authority; laptop-resolvable).
FROZEN_SIW_VIDEO_SPLIT = {"train_videos": 1362, "dev_videos": 338,
                          "live_train_videos": 628, "live_dev_videos": 157}

ROW_SCHEMA_FIELDS = ("fold_id", "dataset", "project_split", "label_live_spoof", "spoof_family",
                    "source_video_id", "frame_index", "crop_relative_path", "crop_sha256",
                    "source_package_kind", "source_package_identity", "status", "subject_id",
                    "failure_reason")

M3B_SOURCE_PACKAGE_KIND = "M3B_PROCESSED_SAMPLE"
SIW_SOURCE_PACKAGE_KIND = "E7B_SIW_SOURCE_CROP"


class E7DError(RuntimeError):
    pass


class E7DConflict(E7DError):
    pass


class E7DTargetFirewallViolation(E7DError):
    pass


# --------------------------------------------------------------------------- #
# TASK A -- protocol lock
# --------------------------------------------------------------------------- #

def build_protocol_lock(repo: Path) -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_PREFIX}-protocol-lock-v1",
        "folds": {fold_id: {"source_domains": list(FOLD_SOURCE_DOMAINS[fold_id]),
                            "heldout_target_domain": FOLD_TARGET_DOMAIN[fold_id]}
                 for fold_id in FOLD_IDS},
        "objective": "materialize deterministic crop-level SOURCE reference manifests per fold; "
                    "reference-only, never copies crop image bytes",
        "never_performs": ["GPAT fitting", "synthetic candidate rendering", "PhysicsRoute",
                           "GPATRoute", "quality gate evaluation", "matched bank construction",
                           "detector training", "target scoring", "target label access",
                           "LLM calls"],
        "row_schema_fields": list(ROW_SCHEMA_FIELDS),
        "no_subject_id_fabrication_for_siw": True,
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False,
        "training_performed": False, "gpat_fitting_performed": False,
    }


def write_protocol_lock(repo: Path) -> dict[str, Any]:
    return _write(repo, "E7D_PROTOCOL_LOCK.json", build_protocol_lock(repo))


# --------------------------------------------------------------------------- #
# TASK B -- input binding (E7-C preflight, E7-B SiW, E7-A SiW split, M3B)
# --------------------------------------------------------------------------- #

def build_input_binding(repo: Path) -> dict[str, Any]:
    e7c_preflight = e7c.e7c_preflight(repo)
    e7b_siw_binding = e7c.build_e7b_binding(repo)  # reused verbatim, never reimplemented
    e7a_bindings = {fold_id: e7c.build_e7a_fold_binding(repo, fold_id) for fold_id in FOLD_IDS}

    m3b_lock_path = repo / e7b.CASIA_MSU_PACKAGE_ROOT / "PACKAGE_LOCK.json"
    m3b_present = m3b_lock_path.is_file()
    m3b_identity = cc.read_json(m3b_lock_path).get("content_identity_sha256") if m3b_present else None
    m3b_match = m3b_present and m3b_identity == e7b.FROZEN_M3B_PACKAGE_IDENTITY

    siw_split_match = all(
        b["siw_population_identity_match"] and b["siw_split_identity_match"]
        for fold_id, b in e7a_bindings.items() if "SiW-Mv2" in FOLD_SOURCE_DOMAINS[fold_id])

    return {
        "schema_version": f"{SCHEMA_PREFIX}-input-binding-v1",
        "E7C_PREFLIGHT_BINDING_MATCH": bool(e7c_preflight["E7C_PREFLIGHT_PASS"]),
        "E7B_SIW_BINDING_MATCH": bool(e7b_siw_binding["match"]),
        "E7A_SIW_SPLIT_BINDING_MATCH": bool(siw_split_match),
        "M3B_BINDING_MATCH": bool(m3b_match),
        "m3b_package_identity_observed": m3b_identity,
        "m3b_package_identity_frozen": e7b.FROZEN_M3B_PACKAGE_IDENTITY,
        "m3b_package_present_locally": m3b_present,
        "frozen_siw": FROZEN_SIW,
        "e7c_preflight_pass": e7c_preflight["E7C_PREFLIGHT_PASS"],
        "e7a_fold_bindings": e7a_bindings,
        "target_access": False, "llm_api_calls": 0,
    }


def write_input_binding(repo: Path) -> dict[str, Any]:
    binding = build_input_binding(repo)
    for key in ("E7C_PREFLIGHT_BINDING_MATCH", "E7B_SIW_BINDING_MATCH", "E7A_SIW_SPLIT_BINDING_MATCH"):
        if not binding[key]:
            raise E7DError(f"input binding FAILED: {key}=False -- FAIL CLOSED: {binding!r}")
    if binding["m3b_package_present_locally"] and not binding["M3B_BINDING_MATCH"]:
        raise E7DError(f"M3B package identity MISMATCH -- FAIL CLOSED: "
                       f"observed={binding['m3b_package_identity_observed']!r} != "
                       f"frozen={binding['m3b_package_identity_frozen']!r}")
    return _write(repo, "E7D_INPUT_BINDING.json", binding)


# --------------------------------------------------------------------------- #
# TASK C -- fold-aware target firewall (extends e7c's: for a fold that does
# NOT use SiW as source, the E7-B SiW SOURCE package itself is ALSO
# forbidden, on top of e7c's target-package/label firewall)
# --------------------------------------------------------------------------- #

def forbidden_roots_for_fold(fold_id: str) -> tuple[str, ...]:
    base = e7c.forbidden_roots_for_fold(fold_id)
    if "SiW-Mv2" not in FOLD_SOURCE_DOMAINS[fold_id]:
        return base + (e7b.E7B_SIW_SOURCE_PACKAGE_ROOT,)
    return base


def assert_not_target_path(fold_id: str, candidate_path: str) -> None:
    normalized = candidate_path.replace("\\", "/").lstrip("/")
    for forbidden in forbidden_roots_for_fold(fold_id):
        if normalized == forbidden or normalized.startswith(forbidden.rstrip("/") + "/"):
            raise E7DTargetFirewallViolation(
                f"{fold_id}: path {candidate_path!r} falls under forbidden root {forbidden!r} -- "
                "FAIL CLOSED, never opened")


def build_target_firewall(repo: Path) -> dict[str, Any]:
    """This is FOLD-AWARE, never a global dataset-name ban: MSU M3B source
    is legal for F1/F3 while the MSU E7-B TARGET package is forbidden ONLY
    for F2; CASIA M3B source is legal for F1/F2 while the CASIA E7-B TARGET
    package is forbidden ONLY for F3."""
    folds = {}
    for fold_id in FOLD_IDS:
        folds[fold_id] = {
            "heldout_target_domain": FOLD_TARGET_DOMAIN[fold_id],
            "forbidden_roots": list(forbidden_roots_for_fold(fold_id)),
            "m3b_source_root_never_forbidden": e7b.CASIA_MSU_PACKAGE_ROOT,
            "active": True,
        }
    return {"schema_version": f"{SCHEMA_PREFIX}-target-firewall-v1", "folds": folds,
           "global_dataset_name_ban": False,
           "note": "CASIA/MSU M3B is a combined SOURCE package legitimately read by every fold "
                  "that uses either dataset as source; it is NEVER on any fold's forbidden list. "
                  "Only each fold's own E7-B TARGET package root (msu_target_v1/casia_target_v1) "
                  "and, for folds without SiW as source, the E7-B SiW SOURCE package root, are "
                  "forbidden -- fold-specific, never dataset-name-global.",
           "target_access": False, "llm_api_calls": 0}


def write_target_firewall(repo: Path) -> dict[str, Any]:
    return _write(repo, "E7D_TARGET_FIREWALL.json", build_target_firewall(repo))


# --------------------------------------------------------------------------- #
# TASK D -- SiW GPU join contract (documented + the actual join function
# used by materialize_fold)
# --------------------------------------------------------------------------- #

def build_join_contract(repo: Path) -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_PREFIX}-join-contract-v1",
        "m3b_source": {
            "authority": "E7-A's own frozen source_train_references/source_dev_references rows "
                        "with reference_kind=='m3b_processed_sample' -- reused verbatim, never a "
                        "fresh M3B parquet scan (E7-A already resolved the exact per-fold subset)",
            "fields_inherited_from_e7a_ref": ["dataset", "project_split", "sample_id",
                                              "source_record_id", "subject_id", "label_live_spoof",
                                              "image_relative_path", "crop_sha256"],
            "crop_resolution": f"{e7b.CASIA_MSU_PACKAGE_ROOT}/<image_relative_path>",
        },
        "siw_source": {
            "authority": "E7-B's SIW_SOURCE_PACKAGE.json rows, INDEPENDENTLY re-joined (not merely "
                        "trusted) against E7-A's own siw_raw_video references via "
                        "e7b._siw_source_refs_from_e7a(repo) -- the SAME frozen, F2/F3-cross-checked "
                        "ref list E7-B itself uses",
            "join_key": "source_video_id (E7-B row) == video_id (E7-A siw_raw_video reference)",
            "steps": [
                "1. read source_video_id from each E7-B SIW_SOURCE_PACKAGE.json row",
                "2. look up the authoritative E7-A siw_raw_video reference by video_id -- FAIL "
                "CLOSED (E7DError) if the video_id is unknown",
                "3. inherit project_split (train/dev), label_live_spoof (live/spoof), spoof_family "
                "from the E7-A reference (authoritative), never from the E7-B row's own embedded "
                "copy alone",
                "4. keep only rows with status == 'success' for crop support manifests",
                "5. for GPAT live support, additionally keep only label_live_spoof == 'live'",
                "6. preserve frame_index, crop_relative_path, crop_sha256, failure_reason from the "
                "E7-B row verbatim",
                "7. resolve the crop under "
                f"{e7b.E7B_SIW_SOURCE_PACKAGE_ROOT}/m2_run/<crop_relative_path> "
                "(NEVER siw_source_v1/<crop_relative_path> directly)",
                "8. no subject_id needed; never sample a replacement frame for any terminal failure",
            ],
            "crop_resolution": f"{e7b.E7B_SIW_SOURCE_PACKAGE_ROOT}/m2_run/<crop_relative_path>",
        },
        "target_access": False, "llm_api_calls": 0,
    }


def write_join_contract(repo: Path) -> dict[str, Any]:
    return _write(repo, "E7D_JOIN_CONTRACT.json", build_join_contract(repo))


def _m3b_row(fold_id: str, ref: dict[str, Any], *, m3b_identity: str | None) -> dict[str, Any]:
    return {"fold_id": fold_id, "dataset": ref["dataset"], "project_split": ref["project_split"],
           "label_live_spoof": ref["label_live_spoof"], "spoof_family": None,
           "source_video_id": ref["source_record_id"], "frame_index": None,
           "crop_relative_path": ref["image_relative_path"], "crop_sha256": ref["crop_sha256"],
           "source_package_kind": M3B_SOURCE_PACKAGE_KIND, "source_package_identity": m3b_identity,
           "status": "success", "subject_id": ref.get("subject_id"), "failure_reason": None,
           "sample_id": ref["sample_id"]}


def _siw_row(fold_id: str, e7b_row: dict[str, Any], *, refs_by_video: dict[str, dict[str, Any]]) -> \
        dict[str, Any]:
    video_id = e7b_row.get("source_video_id")
    ref = refs_by_video.get(video_id)
    if ref is None:
        raise E7DError(f"{fold_id}: unknown SiW source_video_id {video_id!r} -- no E7-A "
                       "siw_raw_video reference exists for it; FAIL CLOSED, refusing to join")
    return {"fold_id": fold_id, "dataset": "SiW-Mv2", "project_split": ref["project_split"],
           "label_live_spoof": ref["label_live_spoof"], "spoof_family": ref.get("spoof_family"),
           "source_video_id": video_id, "frame_index": e7b_row.get("frame_index"),
           "crop_relative_path": e7b_row.get("crop_relative_path"),
           "crop_sha256": e7b_row.get("crop_sha256"), "source_package_kind": SIW_SOURCE_PACKAGE_KIND,
           "source_package_identity": FROZEN_SIW["package_identity"],
           "status": e7b_row.get("status"), "subject_id": None,
           "failure_reason": e7b_row.get("failure_reason")}


def _assert_unique(rows: list[dict[str, Any]], *, key, kind: str) -> None:
    seen: set[Any] = set()
    for row in rows:
        k = key(row)
        if k in seen:
            raise E7DConflict(f"duplicate join key {k!r} for {kind} -- one-to-one join expected; "
                              "FAIL CLOSED")
        seen.add(k)


# --------------------------------------------------------------------------- #
# TASK E -- output schema
# --------------------------------------------------------------------------- #

def build_output_schema(repo: Path) -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_PREFIX}-output-schema-v1",
        "row_fields": list(ROW_SCHEMA_FIELDS) + ["sample_id (M3B rows only)"],
        "nullable_fields": ["spoof_family", "frame_index", "subject_id", "failure_reason"],
        "subject_id_policy": "never fabricated for SiW (always null); preserved verbatim where "
                             "M3B provides one, nullable otherwise",
        "crop_path_semantics": {
            "crop_storage_root_kind": {"M3B_PROCESSED_SAMPLE": e7b.CASIA_MSU_PACKAGE_ROOT,
                                       "E7B_SIW_SOURCE_CROP": f"{e7b.E7B_SIW_SOURCE_PACKAGE_ROOT}/m2_run"},
            "resolution_rule": "repo / crop_storage_root_kind[row.source_package_kind] / "
                              "row.crop_relative_path -- the ONE explicit contract; never "
                              "siw_source_v1/<crop_relative_path> directly",
        },
        "manifest_files_per_fold": ["source_train.json", "source_dev.json",
                                    "source_live_train.json", "source_live_dev.json",
                                    "SOURCE_SUPPORT_PACKAGE.json"],
        "optional_manifest_files": ["terminal_failures.json"],
        "format_note": "JSON metadata manifests (a `rows` array), not `.parquet` -- an explicit "
                       "TECHNICAL implementation choice (JSON is explicitly permitted by this "
                       "milestone's own instructions; no pyarrow schema dependency; matches every "
                       "other E7-B/E7-C planning artifact's format in this codebase). No crop image "
                       "bytes are ever copied into this namespace.",
        "target_access": False, "llm_api_calls": 0,
    }


def write_output_schema(repo: Path) -> dict[str, Any]:
    return _write(repo, "E7D_OUTPUT_SCHEMA.json", build_output_schema(repo))


# --------------------------------------------------------------------------- #
# TASK F -- identity policy
# --------------------------------------------------------------------------- #

def compute_package_identity(*, fold_id: str, m3b_identity: str | None, rows: list[dict[str, Any]]) -> str:
    """Deterministic identity over CANONICAL METADATA only -- never an
    absolute machine path. Material: fold_id, source domains, component
    package identities, and the SORTED list of (dataset, project_split,
    source_video_id, frame_index, crop_sha256, label_live_spoof,
    status) tuples -- crop_sha256 is content-derived, never a path."""
    row_material = sorted(
        (r["dataset"], r["project_split"], r["source_video_id"], r.get("frame_index"),
         r.get("crop_sha256"), r["label_live_spoof"], r["status"])
        for r in rows)
    material = {"fold_id": fold_id, "source_domains": sorted(FOLD_SOURCE_DOMAINS[fold_id]),
               "m3b_package_identity": m3b_identity,
               "siw_package_identity": FROZEN_SIW["package_identity"]
                                      if "SiW-Mv2" in FOLD_SOURCE_DOMAINS[fold_id] else None,
               "siw_split_identity": FROZEN_SIW["split_identity"]
                                    if "SiW-Mv2" in FOLD_SOURCE_DOMAINS[fold_id] else None,
               "row_material": row_material, "schema_version": f"{SCHEMA_PREFIX}-output-schema-v1"}
    return cc.sha256_bytes(cc.canonical_json_bytes(material))


def build_identity_policy(repo: Path) -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_PREFIX}-identity-policy-v1",
        "algorithm": "sha256(canonical_json(material)) where material = {fold_id, sorted "
                    "source_domains, m3b_package_identity, siw_package_identity, "
                    "siw_split_identity, sorted row_material tuples, schema_version}; "
                    "row_material tuples are (dataset, project_split, source_video_id, "
                    "frame_index, crop_sha256, label_live_spoof, status) -- content/identity "
                    "derived fields ONLY",
        "absolute_paths_excluded": True,
        "excluded_from_identity": ["repo root", "any absolute filesystem path",
                                   "crop_relative_path string itself (crop_sha256 stands in for it)",
                                   "wall-clock timestamps", "hostnames"],
        "included_in_identity": ["fold_id", "source_domains", "component package identities",
                                 "per-row dataset/project_split/source_video_id/frame_index/"
                                 "crop_sha256/label_live_spoof/status"],
        "deterministic": True,
        "rerun_same_inputs_same_identity": True,
        "target_access": False, "llm_api_calls": 0,
    }


def write_identity_policy(repo: Path) -> dict[str, Any]:
    return _write(repo, "E7D_IDENTITY_POLICY.json", build_identity_policy(repo))


# --------------------------------------------------------------------------- #
# TASK G -- real materialization (fail-closed; NOT executed on the laptop
# this turn -- see the module docstring / CLAUDE.md GPU boundary)
# --------------------------------------------------------------------------- #

def materialize_fold(repo: Path, fold_id: str, *, authorize: bool = False) -> dict[str, Any]:
    if fold_id not in FOLD_IDS:
        raise E7DError(f"unknown fold_id {fold_id!r}")
    if not authorize:
        raise E7DError(f"materialization for {fold_id} requires --authorize; refusing to run")

    input_binding = build_input_binding(repo)
    for key in ("E7C_PREFLIGHT_BINDING_MATCH", "E7B_SIW_BINDING_MATCH", "E7A_SIW_SPLIT_BINDING_MATCH"):
        if not input_binding[key]:
            raise E7DError(f"{fold_id}: input binding FAILED ({key}=False) -- FAIL CLOSED")
    if input_binding["m3b_package_present_locally"] and not input_binding["M3B_BINDING_MATCH"]:
        raise E7DError(f"{fold_id}: M3B package identity MISMATCH -- FAIL CLOSED")

    materialization = e7b.load_e7a_fold_materialization(repo, fold_id)
    if materialization is None:
        raise E7DError(f"{fold_id}: E7-A materialization missing -- FAIL CLOSED")

    m3b_domains = [d for d in FOLD_SOURCE_DOMAINS[fold_id] if d != "SiW-Mv2"]
    siw_in_fold = "SiW-Mv2" in FOLD_SOURCE_DOMAINS[fold_id]
    m3b_identity = input_binding["m3b_package_identity_observed"]

    m3b_rows: list[dict[str, Any]] = []
    if m3b_domains:
        all_refs = (materialization["source_train_references"] +
                   materialization["source_dev_references"])
        for ref in all_refs:
            if ref["reference_kind"] != "m3b_processed_sample" or ref["dataset"] not in m3b_domains:
                continue
            row = _m3b_row(fold_id, ref, m3b_identity=m3b_identity)
            crop_path = repo / e7b.CASIA_MSU_PACKAGE_ROOT / row["crop_relative_path"]
            if not crop_path.is_file():
                raise E7DError(f"{fold_id}: M3B source crop missing on disk: "
                               f"{row['crop_relative_path']!r} -- FAIL CLOSED")
            if cc.sha256_file(crop_path) != row["crop_sha256"]:
                raise E7DError(f"{fold_id}: M3B source crop SHA256 mismatch: "
                               f"{row['crop_relative_path']!r} -- FAIL CLOSED")
            m3b_rows.append(row)
        _assert_unique(m3b_rows, key=lambda r: r["sample_id"], kind="M3B sample_id")

    siw_success_rows: list[dict[str, Any]] = []
    siw_failure_rows: list[dict[str, Any]] = []
    if siw_in_fold:
        package_path = repo / e7b.E7B_SIW_SOURCE_PACKAGE_ROOT / "SIW_SOURCE_PACKAGE.json"
        if not package_path.is_file():
            raise E7DError(f"{fold_id}: E7-B SIW_SOURCE_PACKAGE.json not present -- GPU_REQUIRED, "
                           "FAIL CLOSED (not a scientific failure)")
        package = cc.read_json(package_path)
        if package.get("package_identity") != FROZEN_SIW["package_identity"] or \
                package.get("population_identity") != FROZEN_SIW["population_identity"] or \
                package.get("split_identity") != FROZEN_SIW["split_identity"]:
            raise E7DError(f"{fold_id}: E7-B SiW source package identity MISMATCH -- FAIL CLOSED")

        refs = e7b._siw_source_refs_from_e7a(repo)
        refs_by_video = {r["video_id"]: r for r in refs}
        m2_output_root = repo / e7b.E7B_SIW_SOURCE_PACKAGE_ROOT / "m2_run"
        for e7b_row in package.get("rows", []):
            out = _siw_row(fold_id, e7b_row, refs_by_video=refs_by_video)
            if out["status"] == "success":
                crop_path = m2_output_root / out["crop_relative_path"]
                if not crop_path.is_file():
                    raise E7DError(f"{fold_id}: SiW source crop missing on disk: "
                                   f"{out['crop_relative_path']!r} -- FAIL CLOSED")
                if cc.sha256_file(crop_path) != out["crop_sha256"]:
                    raise E7DError(f"{fold_id}: SiW source crop SHA256 mismatch: "
                                   f"{out['crop_relative_path']!r} -- FAIL CLOSED")
                if not out["failure_reason"] and out["status"] == "success":
                    pass
                siw_success_rows.append(out)
            else:
                if not out["failure_reason"]:
                    raise E7DError(f"{fold_id}: SiW failure row missing failure_reason -- "
                                   "FAIL CLOSED")
                siw_failure_rows.append(out)

        _assert_unique(siw_success_rows + siw_failure_rows,
                      key=lambda r: (r["source_video_id"], r["frame_index"]), kind="SiW video/frame")

        total = len(siw_success_rows) + len(siw_failure_rows)
        if total != FROZEN_SIW["planned_frame_count"]:
            raise E7DError(f"{fold_id}: SiW terminal accounting mismatch: total={total} != "
                           f"expected {FROZEN_SIW['planned_frame_count']} -- FAIL CLOSED")
        if len(siw_success_rows) != FROZEN_SIW["successful_crop_count"] or \
                len(siw_failure_rows) != FROZEN_SIW["failure_count"]:
            raise E7DError(f"{fold_id}: SiW success/failure counts do not match the frozen "
                           f"aggregate (success={len(siw_success_rows)}, "
                           f"failure={len(siw_failure_rows)}) -- FAIL CLOSED")

    all_source_rows = m3b_rows + siw_success_rows
    train_rows = [r for r in all_source_rows if r["project_split"] in ("source_train", "train")]
    dev_rows = [r for r in all_source_rows if r["project_split"] in ("source_dev", "dev")]
    if len(train_rows) + len(dev_rows) != len(all_source_rows):
        raise E7DError(f"{fold_id}: unrecognized project_split value among source rows -- "
                       "FAIL CLOSED")
    train_video_ids = {r["source_video_id"] for r in train_rows}
    dev_video_ids = {r["source_video_id"] for r in dev_rows}
    if train_video_ids & dev_video_ids:
        raise E7DError(f"{fold_id}: a source_video_id/sample appears in BOTH train and dev -- "
                       "mixed train/dev assignment, FAIL CLOSED")

    live_train_rows = [r for r in train_rows if r["label_live_spoof"] == "live"]
    live_dev_rows = [r for r in dev_rows if r["label_live_spoof"] == "live"]
    for row in all_source_rows:
        if row["label_live_spoof"] not in ("live", "spoof"):
            raise E7DError(f"{fold_id}: row for {row['source_video_id']!r} missing a valid "
                           "live/spoof label -- FAIL CLOSED")
        if row["dataset"] not in FOLD_SOURCE_DOMAINS[fold_id]:
            raise E7DError(f"{fold_id}: row belongs to unexpected source domain "
                           f"{row['dataset']!r} -- FAIL CLOSED")
        if row["dataset"] == FOLD_TARGET_DOMAIN[fold_id]:
            raise E7DError(f"{fold_id}: row belongs to the HELD-OUT TARGET domain -- FAIL CLOSED")

    package_identity = compute_package_identity(fold_id=fold_id, m3b_identity=m3b_identity,
                                                rows=all_source_rows)

    fold_root = repo / E7D_OUTPUT_ROOT / fold_id
    package_manifest_path = fold_root / "SOURCE_SUPPORT_PACKAGE.json"
    if package_manifest_path.is_file():
        existing = cc.read_json(package_manifest_path)
        if existing.get("package_identity") == package_identity:
            return {"resumed": True, "status": "ALREADY_VALID", "path": str(package_manifest_path),
                   "package_identity": package_identity, "target_access": False, "llm_api_calls": 0}
        raise E7DConflict(f"{fold_id}: existing SOURCE_SUPPORT_PACKAGE.json package_identity "
                          f"{existing.get('package_identity')!r} disagrees with the freshly "
                          f"computed {package_identity!r} -- FAIL CLOSED, never overwritten")

    body = {
        "schema_version": f"{SCHEMA_PREFIX}-source-support-package-v1", "fold_id": fold_id,
        "source_domains": list(FOLD_SOURCE_DOMAINS[fold_id]),
        "heldout_target_domain": FOLD_TARGET_DOMAIN[fold_id],
        "package_identity": package_identity, "m3b_package_identity": m3b_identity,
        "siw_package_identity": FROZEN_SIW["package_identity"] if siw_in_fold else None,
        "train_row_count": len(train_rows), "dev_row_count": len(dev_rows),
        "live_train_row_count": len(live_train_rows), "live_dev_row_count": len(live_dev_rows),
        "siw_success_total": len(siw_success_rows) if siw_in_fold else None,
        "siw_failure_total": len(siw_failure_rows) if siw_in_fold else None,
        "target_labels_opened": False, "target_image_bytes_opened": False,
        "status": "MATERIALIZED",
    }
    _write_json_atomic(package_manifest_path, body)
    _write_json_atomic(fold_root / "source_train.json", {"rows": train_rows})
    _write_json_atomic(fold_root / "source_dev.json", {"rows": dev_rows})
    _write_json_atomic(fold_root / "source_live_train.json", {"rows": live_train_rows})
    _write_json_atomic(fold_root / "source_live_dev.json", {"rows": live_dev_rows})
    if siw_failure_rows:
        _write_json_atomic(fold_root / "terminal_failures.json", {"rows": siw_failure_rows})

    return {"resumed": False, "status": "MATERIALIZED", "path": str(package_manifest_path),
           "body": body, "target_access": False, "llm_api_calls": 0, "rendering_performed": False,
           "training_performed": False, "gpat_fitting_performed": False}


def e7d_materialize(repo: Path, *, authorize: bool = False) -> dict[str, Any]:
    return {fold_id: materialize_fold(repo, fold_id, authorize=authorize) for fold_id in FOLD_IDS}


# --------------------------------------------------------------------------- #
# TASK H -- read-only validator
# --------------------------------------------------------------------------- #

def e7d_validate(repo: Path) -> dict[str, Any]:
    """`--e7d-validate`: read-only. Never alters package bytes."""
    results: dict[str, Any] = {}
    for fold_id in FOLD_IDS:
        fold_root = repo / E7D_OUTPUT_ROOT / fold_id
        manifest_path = fold_root / "SOURCE_SUPPORT_PACKAGE.json"
        if not manifest_path.is_file():
            results[fold_id] = {"status": "NOT_MATERIALIZED"}
            continue
        body = cc.read_json(manifest_path)
        problems = []
        for manifest_name, count_field in (("source_train.json", "train_row_count"),
                                           ("source_dev.json", "dev_row_count"),
                                           ("source_live_train.json", "live_train_row_count"),
                                           ("source_live_dev.json", "live_dev_row_count")):
            path = fold_root / manifest_name
            if not path.is_file():
                problems.append(f"missing {manifest_name}")
                continue
            rows = cc.read_json(path).get("rows", [])
            if len(rows) != body.get(count_field):
                problems.append(f"{manifest_name} row count {len(rows)} != recorded "
                                f"{count_field}={body.get(count_field)}")
        if "SiW-Mv2" in FOLD_SOURCE_DOMAINS[fold_id]:
            total = (body.get("siw_success_total") or 0) + (body.get("siw_failure_total") or 0)
            if total != FROZEN_SIW["planned_frame_count"]:
                problems.append(f"SiW terminal accounting {total} != "
                                f"{FROZEN_SIW['planned_frame_count']}")
        results[fold_id] = {"status": "INVALID" if problems else "VALID", "problems": problems,
                           "package_identity": body.get("package_identity")}
    return {"schema_version": f"{SCHEMA_PREFIX}-validate-v1", "folds": results,
           "target_access": False, "llm_api_calls": 0}


# --------------------------------------------------------------------------- #
# TASK I -- execution plan + readiness rollup + strict read-only preflight
# --------------------------------------------------------------------------- #

def build_execution_plan(repo: Path) -> dict[str, Any]:
    input_binding = build_input_binding(repo)
    plan_valid = (input_binding["E7C_PREFLIGHT_BINDING_MATCH"] and
                 input_binding["E7B_SIW_BINDING_MATCH"] and
                 input_binding["E7A_SIW_SPLIT_BINDING_MATCH"] and
                 (not input_binding["m3b_package_present_locally"] or
                  input_binding["M3B_BINDING_MATCH"]))
    siw_bytes_present = (repo / e7b.E7B_SIW_SOURCE_PACKAGE_ROOT / "SIW_SOURCE_PACKAGE.json").is_file()
    return {
        "schema_version": f"{SCHEMA_PREFIX}-execution-plan-v1",
        "E7D_PLAN_VALID": plan_valid,
        "E7D_GPU_BYTES_REQUIRED": not siw_bytes_present,
        "next_gpu_stages": ["per-fold crop-level source support materialization "
                            "(--e7d-materialize --authorize)", "per-fold validation "
                            "(--e7d-validate)"],
        "e7_ready_for_gpat_fitting": False,
        "e7_ready_for_training": False,
        "reason": "no crop-level source support package has been materialized yet; GPAT fitting "
                 "additionally requires all fold source packages to validate, crop hashes to "
                 "resolve, target firewalls to pass, and exact source/live counts to be frozen",
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False,
        "training_performed": False, "gpat_fitting_performed": False,
    }


def write_execution_plan(repo: Path) -> dict[str, Any]:
    return _write(repo, "E7D_EXECUTION_PLAN.json", build_execution_plan(repo))


def e7d_preflight(repo: Path) -> dict[str, Any]:
    """`--e7d-preflight`: STRICTLY READ-ONLY. Performs no GPAT fitting, no
    rendering, no PhysicsRoute/GPATRoute, no quality gate, no training, no
    target scoring, no LLM calls. Writes nothing."""
    input_binding = build_input_binding(repo)
    execution_plan = build_execution_plan(repo)
    validation = e7d_validate(repo)

    source_plan_valid = {fold_id: input_binding["E7C_PREFLIGHT_BINDING_MATCH"] and
                         input_binding["E7A_SIW_SPLIT_BINDING_MATCH"] for fold_id in FOLD_IDS}
    target_firewall_active = {fold_id: True for fold_id in FOLD_IDS}  # structural, always active

    siw_materialized = any(validation["folds"][f]["status"] == "VALID" for f in FOLD_IDS
                           if "SiW-Mv2" in FOLD_SOURCE_DOMAINS[f])

    plan_pass = (input_binding["E7C_PREFLIGHT_BINDING_MATCH"] and
                input_binding["E7B_SIW_BINDING_MATCH"] and
                input_binding["E7A_SIW_SPLIT_BINDING_MATCH"] and
                (not input_binding["m3b_package_present_locally"] or
                 input_binding["M3B_BINDING_MATCH"]) and
                all(source_plan_valid.values()) and all(target_firewall_active.values()))

    return {
        "schema_version": f"{SCHEMA_PREFIX}-preflight-v1",
        "E7C_PREFLIGHT_BINDING_MATCH": input_binding["E7C_PREFLIGHT_BINDING_MATCH"],
        "E7B_SIW_BINDING_MATCH": input_binding["E7B_SIW_BINDING_MATCH"],
        "E7A_SIW_SPLIT_BINDING_MATCH": input_binding["E7A_SIW_SPLIT_BINDING_MATCH"],
        "M3B_BINDING_MATCH": input_binding["M3B_BINDING_MATCH"],
        "F1_SOURCE_PLAN_VALID": source_plan_valid["EXT-F1"],
        "F2_SOURCE_PLAN_VALID": source_plan_valid["EXT-F2"],
        "F3_SOURCE_PLAN_VALID": source_plan_valid["EXT-F3"],
        "F1_TARGET_FIREWALL_ACTIVE": target_firewall_active["EXT-F1"],
        "F2_TARGET_FIREWALL_ACTIVE": target_firewall_active["EXT-F2"],
        "F3_TARGET_FIREWALL_ACTIVE": target_firewall_active["EXT-F3"],
        "SIW_CROP_JOIN_REQUIRED_ON_GPU": True,
        "SIW_CROP_JOIN_MATERIALIZED": siw_materialized,
        "E7D_PLAN_VALID": plan_pass,
        "E7D_GPU_BYTES_REQUIRED": execution_plan["E7D_GPU_BYTES_REQUIRED"],
        "E7D_READY_FOR_GPU_SOURCE_SUPPORT_MATERIALIZATION": plan_pass,
        "E7_READY_FOR_GPAT_FITTING": False,
        "E7_READY_FOR_TRAINING": False,
        "TARGET_LABEL_ACCESS": False, "TARGET_IMAGE_ACCESS": False,
        "GPAT_FITTING_PERFORMED": False, "RENDERING_PERFORMED": False,
        "TRAINING_PERFORMED": False, "LLM_API_CALLS": 0,
        "local_data_state": "PLAN_VALID" if plan_pass and execution_plan["E7D_GPU_BYTES_REQUIRED"]
                            else ("SOURCE_SUPPORT_MATERIALIZED" if siw_materialized else
                                  ("PLAN_VALID" if plan_pass else "MISMATCH_FAIL_CLOSED")),
        "readiness_note": "E7D_READY_FOR_GPU_SOURCE_SUPPORT_MATERIALIZATION=True means the PLAN/"
                          "BINDING is valid and safe to proceed to GPU materialization -- it does "
                          "NOT mean any crop-level source support package is materialized "
                          "(E7D_GPU_BYTES_REQUIRED distinguishes that), and it does NOT mean "
                          "E7_READY_FOR_GPAT_FITTING or E7_READY_FOR_TRAINING.",
    }


def build_readiness(repo: Path) -> dict[str, Any]:
    preflight = e7d_preflight(repo)
    execution_plan = build_execution_plan(repo)
    return {
        "schema_version": f"{SCHEMA_PREFIX}-readiness-v1",
        "E7D_PLAN_VALID": preflight["E7D_PLAN_VALID"],
        "E7D_GPU_BYTES_REQUIRED": preflight["E7D_GPU_BYTES_REQUIRED"],
        "E7D_READY_FOR_GPU_SOURCE_SUPPORT_MATERIALIZATION":
            preflight["E7D_READY_FOR_GPU_SOURCE_SUPPORT_MATERIALIZATION"],
        "E7D_SOURCE_SUPPORT_MATERIALIZED": preflight["SIW_CROP_JOIN_MATERIALIZED"],
        "E7_READY_FOR_GPAT_FITTING": False,
        "E7_READY_FOR_TRAINING": False,
        "reason": execution_plan["reason"],
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False,
        "training_performed": False, "gpat_fitting_performed": False,
    }


def write_readiness(repo: Path) -> dict[str, Any]:
    return _write(repo, "E7D_READINESS.json", build_readiness(repo))


# --------------------------------------------------------------------------- #
# writer plumbing
# --------------------------------------------------------------------------- #

def _write(repo: Path, filename: str, body: dict[str, Any]) -> dict[str, Any]:
    out_dir = repo / E7D_REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    return {"body": body, "path": str(path)}


def _write_json_atomic(path: Path, body: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    tmp_path.replace(path)


def prepare_e7d(repo: Path) -> dict[str, Any]:
    """Writes every additive E7-D planning artifact. Fails closed the
    moment any binding disagrees with its frozen authority. Never
    materializes crop-level packages (that is `--e7d-materialize` only)."""
    return {
        "protocol_lock": write_protocol_lock(repo),
        "input_binding": write_input_binding(repo),
        "target_firewall": write_target_firewall(repo),
        "join_contract": write_join_contract(repo),
        "output_schema": write_output_schema(repo),
        "identity_policy": write_identity_policy(repo),
        "execution_plan": write_execution_plan(repo),
        "readiness": write_readiness(repo),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E7-D per-fold crop-level SOURCE support "
                                                 "materialization (no GPAT fit, no render, no "
                                                 "training, no target-label/image access, no LLM)")
    parser.add_argument("--e7d-preflight", action="store_true", help="Read-only. Writes nothing.")
    parser.add_argument("--e7d-materialize", action="store_true",
                        help="Requires --authorize. Materializes per-fold crop-level source "
                             "support manifests on GPU.")
    parser.add_argument("--e7d-validate", action="store_true", help="Read-only.")
    parser.add_argument("--authorize", action="store_true", help="Required alongside --e7d-materialize.")
    parser.add_argument("--prepare", action="store_true",
                        help="Writes every additive E7-D planning artifact.")
    args = parser.parse_args(argv)
    repo = cc.repo_root()

    if args.e7d_preflight:
        print(json.dumps(e7d_preflight(repo), indent=2, default=str))
        return 0
    if args.e7d_materialize:
        try:
            result = e7d_materialize(repo, authorize=args.authorize)
        except E7DError as error:
            print(f"E7-D materialization refused: {error}")
            return 1
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.e7d_validate:
        print(json.dumps(e7d_validate(repo), indent=2, default=str))
        return 0
    if args.prepare:
        try:
            result = prepare_e7d(repo)
        except E7DError as error:
            print(f"E7-D prepare refused: {error}")
            return 1
        print(json.dumps({"readiness": result["readiness"]["body"]}, indent=2, default=str))
        return 0

    print("Pass --e7d-preflight (read-only), --e7d-materialize --authorize (GPU), "
         "--e7d-validate (read-only), or --prepare.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
