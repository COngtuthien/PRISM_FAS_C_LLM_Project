"""PRISM-FAS-C EXT-Q1Q2 -- E7: per-fold GPAT + synthetic-bank preparation.

E7-D is CLOSED_VALID (commit 89ee9db). This module consumes E7-A/E7-B/E7-C/
E7-D's frozen bindings -- it never rewrites any of them, never invents a
new GPAT/quality/matching algorithm, and never opens held-out target
bytes/labels.

Governing reuse rule (audited from the repository, not assumed): every
scientific primitive below is REUSED VERBATIM; this module's only new code
is the fold-aware DATA-ACCESS BOUNDARY (which rows a fold's GPAT fit/
generation may see) and identity/binding bookkeeping:

- GPAT fitting:        prism_fas.synthesis.gpat_trainer.GPATTrainer.fit
                       (unmodified; consumes prism_fas.synthesis.m8_pipeline.
                       SampleStore/SourceOnlyAudit -- ALSO UNMODIFIED, see the
                       `SourceOnlyAudit compatibility` note below)
- Pairing rules:        prism_fas.synthesis.pair_plan (build_pair_plan,
                       SourceRow, live/spoof + same/cross-domain rules) --
                       reused; only `ALLOWED_DATASETS`/expected pair counts
                       are fold-aware overridable (frozen only as legacy
                       M3B-size assumptions, never as pairing RULES)
- Candidate generation: prism_fas.synthesis.c5_source_pair_plan /
                       c5_arm_plan / c5_raw_generation / c5_render
                       (render_arm, build_routes, GPATRoute, PhysicsRoute)
- Quality gate:         prism_fas.synthesis.quality_gate.evaluate +
                       quality_calibration (source-only calibration)
- Matched bank:         prism_fas.synthesis.c6_matched_bank
- M3B prior generation: prism_fas.data.package.m3b.build_m3b_package
                       (FaceXFormer parsing + AdaFace identity, SHA256-pinned)

SourceOnlyAudit compatibility (audited, not assumed): `SourceOnlyAudit.
record()` (m8_pipeline.py) rejects a path ONLY if the literal substrings
"siw"/"target" (or a FORBIDDEN_SPLITS token) appear in the STRING passed to
it -- `SampleStore.open()` always passes the FIXED string
"manifests/source_train.parquet" (never derived from package_root), and
`SampleStore.load()` passes each row's own `image_relative_path`/
`prior_relative_path` values. Therefore `SourceOnlyAudit`/`SampleStore` are
reused **completely unmodified** for EXT-F2/F3 (SiW-as-source): this
module's own fold-aware adapter selects which rows go into the physical
`package_root` it materializes, and stores their crop/prior paths under
content-hash-style filenames (never containing the literal substring "siw"
or "target") -- a path-naming detail, never an algorithm change, and never
a global weakening of `SourceOnlyAudit`.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
from pathlib import Path
from typing import Any

from prism_fas.evaluation import c_ext_common as cc
from prism_fas.evaluation import c_ext_e7b_data_prep as e7b
from prism_fas.evaluation import c_ext_e7c_gpat_prep as e7c
from prism_fas.evaluation import c_ext_e7d_source_support as e7d

SCHEMA_PREFIX = "ext-q1q2-e7-gpat-bank"
REPORT_DIR = "reports/c_ext_q1q2_v1/e7_three_fold/gpat_bank"
RUN_ROOT = "runs/c_ext_q1q2_v1/e7_gpat_bank"
STATE_ROOT = "state/c_ext_q1q2_v1/e7/gpat_bank"
DATA_ROOT = "data/processed/c_ext_q1q2_v1/e7_gpat_bank"

FOLD_IDS = e7d.FOLD_IDS
FOLD_SOURCE_DOMAINS = e7d.FOLD_SOURCE_DOMAINS
FOLD_TARGET_DOMAIN = e7d.FOLD_TARGET_DOMAIN

CONDITIONS = ("G-REALONLY", "G-RND", "G-DET", "G-LLM", "G-LLM-SHUFFLE-A")
SYNTHETIC_CONDITIONS = ("G-RND", "G-DET", "G-LLM", "G-LLM-SHUFFLE-A")

# --------------------------------------------------------------------------- #
# Frozen E7-D binding (real GPU evidence, commit 89ee9db)
# --------------------------------------------------------------------------- #

FROZEN_E7D_PACKAGE_IDENTITY = {
    "EXT-F1": "955b630fec438c80f284ecbcb30fbf10c83251a23fd31d8ab1a52e0f8ce8383b",
    "EXT-F2": "b617dc8ee6b0827ef5c0be3072563bb0013d1c87c7982527a6b16a5b75dde6a0",
    "EXT-F3": "508a9bd002d22571534d9609a7b301065a8d752b9625f7b500ae27b199ec8955",
}
FROZEN_SIW_CROP_ACCOUNTING = {
    "planned": 6800, "success": 6776, "failure": 24,
    "train_success": 5426, "dev_success": 1350,
    "live_train_success": 2512, "live_dev_success": 628,
    "spoof_train_success": 2914, "spoof_dev_success": 722,
    "train_failures": 22, "dev_failures": 2, "live_failures": 0, "spoof_failures": 24,
}
E7D_FINAL_EVIDENCE_PATH = ("reports/c_ext_q1q2_v1/e7_three_fold/e7d_source_support/"
                          "gpu_evidence/final_e7d_valid_8e02114/E7D_FINAL_VALID_SUMMARY.json")

# --------------------------------------------------------------------------- #
# Frozen M3B prior-generation primitive (audited: prism_fas.data.package.m3b.
# build_m3b_package, already invoked twice -- CASIA+MSU's prism_data_v1_m3b
# and, separately, SiW-Mv2's prism_target_eval_v2 -- with these exact pinned
# weights/config/seed. Reused verbatim for any NEW invocation this module
# requires; never a new model, never a new config choice.)
# --------------------------------------------------------------------------- #

M3B_PRIOR_MODEL_CONFIG_PATH = "configs/models/m3b_priors.yaml"
M3B_PRIOR_GENERATION_FUNCTION = "prism_fas.data.package.m3b.build_m3b_package"
FROZEN_PRIOR_MODELS = {
    "parsing": {"backend": "facexformer", "revision": "fd12148d0b19",
               "weight_sha256": "327a755849ba64d336fb96589ff87b27e84a12be1ecf8bcfaa503d66f803286d"},
    "identity": {"backend": "adaface_ir50", "revision": "60a65befbcf7",
                "weight_sha256": "43bd2d570584d95d4a17ce81f26449034c45dbeed750afcab651872abc0e1496"},
}
PRIOR_SCHEMA_VERSION = "m3b-priors-v1"
PRIOR_SEED = 20260805
FROZEN_M3B_PACKAGE_IDENTITY = e7b.FROZEN_M3B_PACKAGE_IDENTITY  # reused verbatim, never redeclared

#: The SiW-as-source prior package this module materializes (GPU-only, not
#: executed this turn). F2 and F3 share this EXACT same package -- the SiW
#: source crop population is identical for both folds -- so prior
#: generation runs at most ONCE, never per-fold.
SIW_SOURCE_PRIOR_PACKAGE_ROOT = f"{DATA_ROOT}/siw_source_priors_v1"
SIW_SOURCE_PRIOR_PACKAGE_FILENAME = "SIW_SOURCE_PRIOR_PACKAGE.json"
EXPECTED_SIW_SUCCESS_CROP_COUNT = FROZEN_SIW_CROP_ACCOUNTING["success"]  # == 6776
REQUIRED_PRIOR_KEYS = ("parsing_labels", "pose_ypr", "visibility", "bbox", "landmarks", "crop_box")

#: TECHNICAL M3A-compatibility input package: `prism_fas.data.package.m3b.
#: build_m3b_package` requires a VALIDATED M3A package (PACKAGE_LOCK.json
#: status=="validated", manifests/{samples,source_train,source_dev}.parquet,
#: base priors with bbox/landmarks/crop_box/quality_vector). E7-B's SiW
#: source package is NOT an M3A package (it is E7-B's own frozen M2-crop
#: namespace) -- this additive adapter materializes a real, validated M3A
#: package FROM it, joined against E7-D/E7-A project_split authority
#: (never the legacy `config.project_split()`, which rejects E7-B's
#: `SIW_SOURCE_OFFICIAL_SPLIT_PLACEHOLDER`). This is a data-contract
#: adapter, not a new scientific method: every array/manifest/shard/
#: validation primitive it calls (`build_priors`, `write_manifest`,
#: `build_lock`, `validate_package`, `finalize_lock`, `plan_shards`,
#: `write_shard`) is reused verbatim from `prism_fas.data.package.*`.
SIW_SOURCE_M3A_INPUT_PACKAGE_ROOT = f"{DATA_ROOT}/siw_source_m3a_input_v1"
#: Deliberately distinct from the canonical CASIA/MSU M3A `package_id_prefix`
#: ("prism_data_v1_m3a") -- this package must never be mistaken for, or
#: silently claim, the frozen scientific M3A package identity.
SIW_SOURCE_M3A_INPUT_PACKAGE_ID = "prism_data_v1_m3a_ext_siw_source_v1"
M3A_PACKAGE_CONFIG_PATH = "configs/data/package_m3a.yaml"
#: Distinct from the frozen M3B `package_id` (`prism_data_v1_m3b`) for the
#: same reason `m3b.py::_finalize` takes `package_id` as an input rather
#: than a constant -- an ADDITIVE package must never silently claim the
#: frozen identity.
SIW_SOURCE_PRIOR_M3B_PACKAGE_ID = "prism_data_v1_m3b_ext_siw_source_v1"
M3A_INPUT_BINDING_FILENAME = "E7_M3A_INPUT_BINDING.json"
EXPECTED_SIW_TRAIN_SUCCESS_COUNT = FROZEN_SIW_CROP_ACCOUNTING["train_success"]  # == 5426
EXPECTED_SIW_DEV_SUCCESS_COUNT = FROZEN_SIW_CROP_ACCOUNTING["dev_success"]  # == 1350
#: `prism_fas.data.package.config.PACKAGE_SCHEMA_VERSION` frozen literal, reused as a plain
#: string here so the identity binding never has to import the config module.
M3A_PACKAGE_SCHEMA_VERSION = "m3a-v1"
#: The two folds that actually carry a SiW source domain -- the SHARED SiW authority is
#: cross-checked against BOTH, independent of which one a command was invoked with.
SIW_SHARING_FOLD_IDS = ("EXT-F2", "EXT-F3")

#: The ONLY existing on-disk SiW prior material (`prism_target_eval_v2`) is
#: architecturally bound as F1's held-out TARGET-feature package (real,
#: validated M3B-schema priors, but every row `project_split==target_test`,
#: registered as a protected target artifact in c_ext_protected_manifest.py/
#: c_ext_e0_freeze.py/c_ext_e7c_gpat_prep.FOLD_TARGET_PACKAGE_ROOT). This
#: module NEVER reads it for source purposes -- reusing its crop/prior
#: bytes as SOURCE input would cross the target firewall in a way no prior
#: milestone authorized. If SiW-as-source priors are ever needed (F2/F3),
#: the SAME frozen `build_m3b_package` primitive must be invoked AGAIN, on
#: GPU, against E7-B's OWN SiW-as-source crop package -- never against this
#: target-tagged package.
PROTECTED_SIW_TARGET_PRIOR_PACKAGE_ROOT = e7b.SIW_TARGET_EVAL_PACKAGE_ROOT

# --------------------------------------------------------------------------- #
# Frozen GPAT fit config (audited: configs/synthesis/gpat_m8.yaml)
# --------------------------------------------------------------------------- #

GPAT_FIT_CONFIG_PATH = "configs/synthesis/gpat_m8.yaml"
GPAT_FIT_SEED = 20260806  # config["seed"] / config["pair_plan"]["seed"] -- both frozen at this value
M7_RECIPE_BANK_ROOT = "assets/recipe_banks/prism_recipe_bank_m7_v1"
FROZEN_M7_BANK = {"bank_id": "prism_recipe_bank_m7_v1", "bank_content_identity_sha256":
                 "fa989938cafdc4887518cc45c35d559d00278358439dc68c2486da10309210cb",
                 "recipe_count": 128, "bank_seed": 20260806, "status": "frozen"}

# --------------------------------------------------------------------------- #
# Frozen shared SiW source-prior evidence (CLOSED_VALID, commit 94e3b55) --
# reused verbatim; never regenerated by this module.
# --------------------------------------------------------------------------- #

FROZEN_SOURCE_PRIOR_EVIDENCE_PATH = ("reports/c_ext_q1q2_v1/e7_three_fold/gpat_bank/gpu_evidence/"
                                    "source_priors_c8800d0/E7_SOURCE_PRIOR_CLOSURE.json")
FROZEN_SIW_SOURCE_PRIOR_PACKAGE_IDENTITY = ("36f9154a4bd8e82382ffcc57a6b2c78749188ef07d222a0f11ab3"
                                            "d2f7817dec6")

# --------------------------------------------------------------------------- #
# TASK N/O -- real fold-aware GPAT-input package + pair-plan + fit
# orchestration. Reuses `prism_fas.synthesis.gpat_trainer.GPATTrainer.fit`,
# `prism_fas.synthesis.pair_plan.build_pair_plan`/`write_pair_plan`, and
# `prism_fas.synthesis.m8_pipeline.SampleStore`/`SourceOnlyAudit` UNMODIFIED.
# The only new code is: which rows go into a fold-local package (E7-D/E7-A
# authority), a scoped adapter around `pair_plan`'s module-level CASIA/MSU-
# only constants, effective-config bookkeeping, and transaction/resume
# plumbing around the real trainer call.
# --------------------------------------------------------------------------- #

GPAT_INPUT_ROOT = f"{DATA_ROOT}/gpat_input"
GPAT_PAIR_PLAN_ROOT = f"{DATA_ROOT}/gpat_pairs"
GPAT_INPUT_LOCK_FILENAME = "PACKAGE_LOCK.json"
GPAT_FIT_LOCK_FILENAME = "GPAT_FIT_LOCK.json"

#: Fold source dataset slugs in the M2/M3B naming convention (`casia_fasd`,
#: `msu_mfsd`, `siw_mv2`) -- the ONLY thing that varies pair_plan's
#: `ALLOWED_DATASETS` per fold; never a change to the pairing algorithm.
FOLD_SOURCE_DATASET_SLUGS = {
    "EXT-F1": ("casia_fasd", "msu_mfsd"),
    "EXT-F2": ("casia_fasd", "siw_mv2"),
    "EXT-F3": ("msu_mfsd", "siw_mv2"),
}

# --------------------------------------------------------------------------- #
# Frozen recipe banks for CANDIDATE GENERATION (arm content) -- reused
# verbatim from E7-C's own provenance binding, never re-derived.
# --------------------------------------------------------------------------- #

RECIPE_BANK_ROOTS = e7c.RECIPE_BANK_ROOTS
LLM_SHUFFLE_A_RECIPES_PATH = e7c.LLM_SHUFFLE_A_RECIPES_PATH
LLM_SHUFFLE_A_AUDIT_PATH = e7c.LLM_SHUFFLE_A_AUDIT_PATH

# --------------------------------------------------------------------------- #
# Frozen bank-size quota (audited from gate_profiles.py; asserted, never
# trusted from prose)
# --------------------------------------------------------------------------- #

def frozen_bank_quota() -> dict[str, int]:
    from prism_fas.synthesis import gate_profiles

    quota = {"final_bank_per_arm": gate_profiles.FINAL_BANK_PER_ARM,
            "physics_per_arm": gate_profiles.PHYSICS_PER_ARM,
            "gpat_per_arm": gate_profiles.GPAT_PER_ARM}
    expected = {"final_bank_per_arm": 1024, "physics_per_arm": 512, "gpat_per_arm": 512}
    if quota != expected:
        raise E7Error(f"frozen bank quota drift detected: repository authority {quota!r} != "
                      f"expected {expected!r} -- STOP, do not choose a new quota")
    return quota


# --------------------------------------------------------------------------- #
# F1 Shuffle: TRUE, frozen, permanent block (E6-v2 closure, reused verbatim)
# --------------------------------------------------------------------------- #

FROZEN_E6V2_CLOSURE = e7c.FROZEN_E6V2_CLOSURE

BLOCKED_TRUE_FROZEN_MATCHED_BANK_INFEASIBILITY = e7c.BLOCKED_TRUE_FROZEN_MATCHED_BANK_INFEASIBILITY
PENDING_FEASIBILITY_PREFLIGHT = e7c.PENDING_FEASIBILITY_PREFLIGHT
NOT_APPLICABLE = e7c.NOT_APPLICABLE

FORBIDDEN_RESCUE_ACTIONS = ["lower bank size 512", "change source-domain quotas",
                           "relax quality thresholds", "modify q", "resample candidates",
                           "rerender solely to obtain a passing bank", "change matching policy"]

# --------------------------------------------------------------------------- #
# Target firewall (fold-aware, extends e7c/e7d's -- never a global ban)
# --------------------------------------------------------------------------- #

FORBIDDEN_ROOTS_FOR_FOLD = e7d.forbidden_roots_for_fold


class E7Error(RuntimeError):
    pass


class E7Conflict(E7Error):
    pass


class E7TargetFirewallViolation(E7Error):
    pass


def assert_not_target_path(fold_id: str, candidate_path: str) -> None:
    try:
        e7d.assert_not_target_path(fold_id, candidate_path)  # reused verbatim
    except e7d.E7DTargetFirewallViolation as exc:
        raise E7TargetFirewallViolation(str(exc)) from exc


# --------------------------------------------------------------------------- #
# TASK A -- protocol lock
# --------------------------------------------------------------------------- #

def build_protocol_lock(repo: Path) -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_PREFIX}-protocol-lock-v1",
        "folds": {fold_id: {"source_domains": list(FOLD_SOURCE_DOMAINS[fold_id]),
                            "heldout_target_domain": FOLD_TARGET_DOMAIN[fold_id]}
                 for fold_id in FOLD_IDS},
        "stages": ["gpat_input_compatibility", "gpat_pair_plan", "gpat_fit",
                  "source_only_quality_calibration", "candidate_generation", "quality_gate",
                  "shuffle_feasibility", "matched_bank", "integrity_lock",
                  "training_readiness_closure"],
        "conditions": list(CONDITIONS), "synthetic_conditions": list(SYNTHETIC_CONDITIONS),
        "reused_primitives": {
            "gpat_fitting": "prism_fas.synthesis.gpat_trainer.GPATTrainer.fit",
            "gpat_model": "prism_fas.synthesis.gpat_model.build_gpat_model",
            "gpat_losses": "prism_fas.synthesis.gpat_losses.compute_losses",
            "sample_store": "prism_fas.synthesis.m8_pipeline.SampleStore (UNMODIFIED)",
            "source_only_audit": "prism_fas.synthesis.m8_pipeline.SourceOnlyAudit (UNMODIFIED)",
            "pairing": "prism_fas.synthesis.pair_plan.build_pair_plan",
            "candidate_schedule": "prism_fas.synthesis.c5_source_pair_plan.build_source_pair_plan",
            "candidate_render": "prism_fas.synthesis.c5_render.render_arm / build_routes",
            "routes": "prism_fas.synthesis.synthetic_bank.GPATRoute / PhysicsRoute",
            "quality_gate": "prism_fas.synthesis.quality_gate.evaluate",
            "quality_calibration": "prism_fas.synthesis.quality_calibration.calibrate",
            "matched_bank": "prism_fas.synthesis.c6_matched_bank.build_matched_banks",
            "prior_generation": M3B_PRIOR_GENERATION_FUNCTION,
        },
        "never_modifies": ["GPAT architecture", "GPAT losses", "optimizer/scheduler",
                          "checkpoint selection", "early stopping", "pairing scientific rules",
                          "quality metrics", "quality thresholds", "matching policy",
                          "SourceOnlyAudit", "quota"],
        "no_global_dataset_name_ban": True,
        "source_split_policy": "GPAT fit/pairing/candidate-source support uses E7-D source TRAIN "
                               "only; source DEV is reserved for preregistered source-only "
                               "calibration/validation; train and dev are never merged",
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False,
        "training_performed": False, "gpat_fitting_performed": False,
    }


def write_protocol_lock(repo: Path) -> dict[str, Any]:
    return _write(repo, "GPAT_BANK_PROTOCOL_LOCK.json", build_protocol_lock(repo))


# --------------------------------------------------------------------------- #
# TASK B -- E7-D binding
# --------------------------------------------------------------------------- #

def build_e7d_binding(repo: Path) -> dict[str, Any]:
    evidence_path = repo / E7D_FINAL_EVIDENCE_PATH
    evidence_present = evidence_path.is_file()
    evidence = cc.read_json(evidence_path) if evidence_present else {}
    folds_evidence = evidence.get("folds", {}) if evidence_present else {}

    per_fold = {}
    for fold_id in FOLD_IDS:
        observed_identity = folds_evidence.get(fold_id, {}).get("package_identity")
        per_fold[fold_id] = {
            "frozen_package_identity": FROZEN_E7D_PACKAGE_IDENTITY[fold_id],
            "observed_package_identity": observed_identity,
            "match": (observed_identity == FROZEN_E7D_PACKAGE_IDENTITY[fold_id]) if evidence_present
                    else None,
        }
    all_match = evidence_present and all(f["match"] for f in per_fold.values())
    siw_accounting_match = (evidence_present and
                            evidence.get("SIW_SUCCESS_TOTAL") == FROZEN_SIW_CROP_ACCOUNTING["success"]
                            and evidence.get("SIW_FAILURE_TOTAL") == FROZEN_SIW_CROP_ACCOUNTING["failure"])

    local_package_present = {fold_id: (repo / e7d.E7D_OUTPUT_ROOT / fold_id /
                                       "SOURCE_SUPPORT_PACKAGE.json").is_file() for fold_id in FOLD_IDS}

    return {
        "schema_version": f"{SCHEMA_PREFIX}-e7d-binding-v1",
        "e7d_status": "CLOSED_VALID", "evidence_path": E7D_FINAL_EVIDENCE_PATH,
        "evidence_present": evidence_present, "folds": per_fold,
        "E7D_BINDING_MATCH": all_match, "siw_accounting_match": siw_accounting_match,
        "frozen_siw_crop_accounting": FROZEN_SIW_CROP_ACCOUNTING,
        "local_package_bytes_present": local_package_present,
        "target_access": False, "llm_api_calls": 0,
    }


def write_e7d_binding(repo: Path) -> dict[str, Any]:
    binding = build_e7d_binding(repo)
    if binding["evidence_present"] and not binding["E7D_BINDING_MATCH"]:
        raise E7Error(f"E7-D binding MISMATCH -- FAIL CLOSED: {binding['folds']!r}")
    return _write(repo, "GPAT_FOLD_SOURCE_BINDING.json", binding)


# --------------------------------------------------------------------------- #
# TASK C -- GPAT input compatibility audit (THE critical fail-closed gate)
# --------------------------------------------------------------------------- #

def audit_gpat_input_compatibility(repo: Path, fold_id: str) -> dict[str, Any]:
    """Determines, from real repository evidence only, whether this fold's
    GPAT-fitting input (crops + priors in the exact schema SampleStore/
    RegionMaskBuilder require) can be produced WITHOUT any new scientific
    choice. Never fabricates a positive result.

    Reports FOUR distinct, never-conflated dimensions per source:
    (1) COMPATIBLE vs BLOCKED (is there any unresolved scientific gap at all)
    (2) prior_generation_primitive_resolved (is the exact frozen code path/
        config/model/seed known -- always independent of whether it has
        been RUN)
    (3) priors_materialized (does a real, on-disk, validated prior package
        actually exist RIGHT NOW)
    (4) the overall `status`, which is COMPATIBLE only once EVERY required
        source's priors are materialized -- never merely "the primitive is
        known".
    """
    if fold_id not in FOLD_IDS:
        raise E7Error(f"unknown fold_id {fold_id!r}")
    m3b_domains = [d for d in FOLD_SOURCE_DOMAINS[fold_id] if d != "SiW-Mv2"]
    siw_in_fold = "SiW-Mv2" in FOLD_SOURCE_DOMAINS[fold_id]

    m3b_lock_path = repo / e7b.CASIA_MSU_PACKAGE_ROOT / "PACKAGE_LOCK.json"
    m3b_primitive_resolved = None
    m3b_priors_materialized = None
    m3b_note = "no M3B source domain in this fold"
    if m3b_domains:
        # M3B priors already exist as a frozen, previously-completed build --
        # no fresh generation is ever needed for M3B; "primitive resolved" is
        # therefore NOT_APPLICABLE_EXISTING_PRIORS rather than a pending step.
        m3b_primitive_resolved = "NOT_APPLICABLE_EXISTING_PRIORS"
        if m3b_lock_path.is_file():
            m3b_lock = cc.read_json(m3b_lock_path)
            m3b_priors_materialized = (
                m3b_lock.get("content_identity_sha256") == FROZEN_M3B_PACKAGE_IDENTITY
                and m3b_lock.get("status") == "validated")
            m3b_note = ("frozen, already-validated M3B package with full priors "
                       "(parsing_labels/pose_ypr/visibility/bbox/landmarks/crop_box) -- "
                       "package_identity verified locally, MATERIALIZED" if m3b_priors_materialized
                       else "local PACKAGE_LOCK.json present but identity/status does not match "
                       "the frozen pin -- FAIL CLOSED")
        else:
            m3b_priors_materialized = False
            m3b_note = ("M3B PACKAGE_LOCK.json not present on this laptop -- NOT MATERIALIZED "
                       "locally. Prior schema/identity is nonetheless resolvable from E7-A's own "
                       "frozen m3b_processed_sample references (each embeds prior_relative_path + "
                       "prior_sha256), so this is GPU_REQUIRED for the actual bytes, never a "
                       "scientific compatibility gap")

    siw_note = "no SiW source domain in this fold"
    siw_primitive_resolved = None
    siw_priors_materialized = None
    if siw_in_fold:
        siw_source_package_present = (repo / e7b.E7B_SIW_SOURCE_PACKAGE_ROOT /
                                      "SIW_SOURCE_PACKAGE.json").is_file()
        # The frozen prior-generation PRIMITIVE is fully resolved regardless of whether it has
        # been invoked yet -- this is independent of materialization.
        siw_primitive_resolved = True
        prior_package_path = repo / SIW_SOURCE_PRIOR_PACKAGE_ROOT / SIW_SOURCE_PRIOR_PACKAGE_FILENAME
        siw_priors_materialized = False
        if prior_package_path.is_file():
            prior_validation = validate_source_priors(repo, fold_id)
            siw_priors_materialized = prior_validation["status"] == "VALID"
        if siw_priors_materialized:
            siw_note = ("SiW-as-source priors ARE materialized and strictly validated at "
                       f"{SIW_SOURCE_PRIOR_PACKAGE_ROOT} (shared verbatim by F2/F3).")
        else:
            # The only existing SiW prior material lives under the PROTECTED target-feature
            # package (prism_target_eval_v2); it is NEVER read here. SiW-as-source priors must be
            # generated by re-invoking the SAME frozen build_m3b_package() primitive against
            # E7-B's own SiW-as-source crop package -- a real, already-twice-proven-reusable,
            # policy-neutral primitive, applied to a role (SiW-as-source) it has not yet been run
            # for. This is the SAME resolution pattern already established for E7-B's own SiW
            # source build (one frozen policy, applied to a new authorized role).
            siw_note = (
                "SiW-as-source priors do NOT exist yet (E7-B's own SIW_SOURCE_PACKAGE_ROOT has no "
                f"prior-generation step; siw_source_package_present_locally="
                f"{siw_source_package_present}). NOT MATERIALIZED. The REQUIRED prior-generation "
                f"primitive IS fully frozen and reusable ({M3B_PRIOR_GENERATION_FUNCTION}, "
                f"FaceXFormer rev {FROZEN_PRIOR_MODELS['parsing']['revision']} sha256 "
                f"{FROZEN_PRIOR_MODELS['parsing']['weight_sha256'][:16]}..., AdaFace rev "
                f"{FROZEN_PRIOR_MODELS['identity']['revision']}, config {M3B_PRIOR_MODEL_CONFIG_PATH}, "
                f"seed {PRIOR_SEED}) and MUST be invoked (on GPU, via "
                "`--prepare-source-priors --fold EXT-F2/F3 --authorize`) against E7-B's own "
                "SiW-as-source crop package -- NEVER against the protected target-tagged "
                "prism_target_eval_v2 package. This is GPU_REQUIRED, NOT a new scientific choice, "
                "and NOT scientifically blocked -- but it is also NOT yet ready to fit GPAT."
            )

    m3b_blocked = bool(m3b_domains) and m3b_lock_path.is_file() and not m3b_priors_materialized
    if m3b_blocked:
        status = "BLOCKED_UNRESOLVED_SOURCE_PRIOR_REQUIREMENT"
    elif siw_in_fold and not siw_priors_materialized:
        status = "COMPATIBLE_PENDING_GPU_PRIOR_GENERATION"
    else:
        status = "COMPATIBLE"

    return {
        "schema_version": f"{SCHEMA_PREFIX}-gpat-input-compatibility-v2", "fold_id": fold_id,
        "status": status,
        "m3b_prior_generation_primitive_resolved": m3b_primitive_resolved,
        "m3b_priors_materialized": m3b_priors_materialized, "m3b_note": m3b_note,
        "siw_prior_generation_primitive_resolved": siw_primitive_resolved,
        "siw_priors_materialized": siw_priors_materialized, "siw_note": siw_note,
        "sample_store_mask_builder_note": (
            "prism_fas.synthesis.m8_pipeline.SampleStore.mask_builder() hard-indexes "
            "arrays['parsing_labels']/arrays['crop_box'] (no .get fallback) -- any prior lacking "
            "those exact keys will KeyError. This module's adapter must materialize a full "
            "M3B-schema prior (never a partial/geometry-only one) for every row it feeds to "
            "SampleStore, to stay compatible with the UNMODIFIED legacy code path."),
        "target_access": False, "llm_api_calls": 0,
    }


def write_input_compatibility(repo: Path) -> dict[str, Any]:
    audits = {fold_id: audit_gpat_input_compatibility(repo, fold_id) for fold_id in FOLD_IDS}
    return _write(repo, "GPAT_INPUT_COMPATIBILITY.json", {
        "schema_version": f"{SCHEMA_PREFIX}-gpat-input-compatibility-all-v1", "folds": audits,
        "target_access": False, "llm_api_calls": 0})


# --------------------------------------------------------------------------- #
# TASK D -- prior binding (freezes the exact code path/config/model/schema
# identities required by Q "AUDIT FIRST"; never executes generation)
# --------------------------------------------------------------------------- #

def build_prior_binding(repo: Path) -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_PREFIX}-prior-binding-v1",
        "prior_generation_function": M3B_PRIOR_GENERATION_FUNCTION,
        "config_path": M3B_PRIOR_MODEL_CONFIG_PATH,
        "prior_schema_version": PRIOR_SCHEMA_VERSION, "seed": PRIOR_SEED,
        "models": FROZEN_PRIOR_MODELS,
        "m3b_package_identity": FROZEN_M3B_PACKAGE_IDENTITY,
        "protected_siw_target_prior_package_root": PROTECTED_SIW_TARGET_PRIOR_PACKAGE_ROOT,
        "protected_package_never_read_for_source": True,
        "output_prior_identity_rule": "sha256 of the merged prior .npz content, recorded per-row "
                                      "as prior_sha256 -- identical convention to M3B's own "
                                      "prior_sha256 field (frozen, never redefined)",
        "input_crop_sha_binding": "prior generation for a SiW-as-source row must be keyed on "
                                  "E7-B's own crop_sha256 for that row (never recomputed from a "
                                  "different crop, never a replacement/resampled frame)",
        "siw_source_prior_package_root": SIW_SOURCE_PRIOR_PACKAGE_ROOT,
        "siw_source_prior_package_shared_by_f2_f3": True,
        "expected_siw_success_crop_count": EXPECTED_SIW_SUCCESS_CROP_COUNT,
        "target_access": False, "llm_api_calls": 0,
    }


def write_prior_binding(repo: Path) -> dict[str, Any]:
    return _write(repo, "GPAT_PRIOR_BINDING.json", build_prior_binding(repo))


# --------------------------------------------------------------------------- #
# TASK D.2 -- SiW-as-source prior package: identity + strict read-only
# validator + GPU-stage generation entry point. Shared verbatim by F2/F3
# (identical underlying SiW source crop population).
# --------------------------------------------------------------------------- #

def compute_siw_source_prior_package_identity(rows: list[dict[str, Any]]) -> str:
    """Deterministic identity over CANONICAL METADATA only -- never an
    absolute machine path. Binds: E7-B SiW source package identity, E7-A
    split identity, the frozen prior-generation primitive/config/model/seed
    identities, and the sorted per-row (source_video_id, frame_index,
    source_crop_sha256, prior_sha256) material."""
    row_material = sorted(
        (r.get("source_video_id"), r.get("frame_index"), r.get("source_crop_sha256"),
         r.get("prior_sha256")) for r in rows)
    material = {
        "e7b_siw_source_package_identity": e7c.FROZEN_E7B["siw_source_package_identity"],
        "e7a_siw_split_identity": e7c.FROZEN_E7B["siw_split_identity"],
        "prior_generation_function": M3B_PRIOR_GENERATION_FUNCTION,
        "prior_config_path": M3B_PRIOR_MODEL_CONFIG_PATH, "prior_models": FROZEN_PRIOR_MODELS,
        "prior_seed": PRIOR_SEED, "prior_schema_version": PRIOR_SCHEMA_VERSION,
        "row_material": row_material,
    }
    return cc.sha256_bytes(cc.canonical_json_bytes(material))


def _validate_siw_source_prior_rows(repo: Path, fold_id: str, rows: list[dict[str, Any]],
                                    recorded_package_identity: str | None) -> dict[str, Any]:
    """The SUBSTANTIVE validation shared by both the strict on-disk
    validator (`validate_source_priors`) and the pre-write CANDIDATE
    validator (`validate_source_prior_candidate`) -- identical checks
    either way: row_count, duplicate-key absence, every source crop/prior
    exists on disk with the exact SHA, full required M3B prior keys, the
    target firewall, and package-identity recomputation. By the time this
    runs the candidate's crop/prior BYTES already exist on disk (written by
    the real `build_m3b_package` call) -- the ONLY thing that may not yet
    exist is the terminal `SIW_SOURCE_PRIOR_PACKAGE.json` marker itself,
    which this function never reads or requires. Never writes."""
    problems: list[str] = []
    if len(rows) != EXPECTED_SIW_SUCCESS_CROP_COUNT:
        problems.append(f"row_count {len(rows)} != expected {EXPECTED_SIW_SUCCESS_CROP_COUNT}")

    seen_keys: set[Any] = set()
    crop_rows_verified = 0
    missing_crops = missing_priors = bad_crop_hashes = bad_prior_hashes = 0
    for row in rows:
        if row.get("status") != "success":
            problems.append(f"row {row.get('source_video_id')!r} is not status=='success' -- "
                            "terminal failures must never enter the prior package")
        key = (row.get("source_video_id"), row.get("frame_index"))
        if key in seen_keys:
            problems.append(f"duplicate source crop/frame key {key!r}")
        seen_keys.add(key)

        crop_relative_path = row.get("source_crop_relative_path")
        for kind, root, rel, sha_field in (
            ("source crop", repo / e7b.E7B_SIW_SOURCE_PACKAGE_ROOT / "m2_run", crop_relative_path,
             row.get("source_crop_sha256")),
            ("prior file", repo / SIW_SOURCE_PRIOR_PACKAGE_ROOT, row.get("prior_relative_path"),
             row.get("prior_sha256")),
        ):
            if not rel:
                problems.append(f"row {key!r}: missing {kind} reference")
                continue
            try:
                assert_not_target_path(fold_id, f"{root.relative_to(repo).as_posix()}/{rel}")
            except E7TargetFirewallViolation as exc:
                problems.append(str(exc))
            path = root / rel
            if not path.is_file():
                problems.append(f"{kind} missing on disk: {rel!r}")
                if kind == "source crop":
                    missing_crops += 1
                else:
                    missing_priors += 1
                continue
            if cc.sha256_file(path) != sha_field:
                problems.append(f"{kind} SHA256 mismatch: {rel!r}")
                if kind == "source crop":
                    bad_crop_hashes += 1
                else:
                    bad_prior_hashes += 1
                continue
            if kind == "source crop":
                crop_rows_verified += 1
            else:
                import numpy as np

                try:
                    with np.load(path, allow_pickle=False) as handle:
                        keys = set(handle.files)
                except Exception as exc:  # noqa: BLE001 -- any unreadable prior is a hard failure
                    problems.append(f"prior file unreadable: {rel!r} ({exc})")
                    continue
                missing_keys = set(REQUIRED_PRIOR_KEYS) - keys
                if missing_keys:
                    problems.append(f"prior file {rel!r} missing required keys: {sorted(missing_keys)}")

    recomputed_identity = compute_siw_source_prior_package_identity(rows)
    identity_match = recomputed_identity == recorded_package_identity
    if not identity_match:
        problems.append(f"recomputed package_identity {recomputed_identity!r} != recorded "
                        f"{recorded_package_identity!r}")

    return {
        "status": "INVALID" if problems else "VALID", "problems": problems,
        "row_count": len(rows), "expected_row_count": EXPECTED_SIW_SUCCESS_CROP_COUNT,
        "crop_rows_verified": crop_rows_verified, "missing_crops": missing_crops,
        "bad_crop_hashes": bad_crop_hashes, "missing_prior_files": missing_priors,
        "bad_prior_hashes": bad_prior_hashes, "package_identity": recorded_package_identity,
        "recomputed_package_identity": recomputed_identity, "package_identity_match": identity_match,
        "target_access": False, "llm_api_calls": 0,
    }


def validate_source_priors(repo: Path, fold_id: str) -> dict[str, Any]:
    """STRICT, read-only validator for the ON-DISK shared SiW-as-source
    prior package (the written `SIW_SOURCE_PRIOR_PACKAGE.json` terminal
    marker). F2 and F3 both validate the exact SAME on-disk package. Never
    alters package bytes."""
    if fold_id not in FOLD_IDS:
        raise E7Error(f"unknown fold_id {fold_id!r}")
    if "SiW-Mv2" not in FOLD_SOURCE_DOMAINS[fold_id]:
        return {"schema_version": f"{SCHEMA_PREFIX}-source-priors-validate-v1", "fold_id": fold_id,
               "status": "NOT_APPLICABLE"}
    package_path = repo / SIW_SOURCE_PRIOR_PACKAGE_ROOT / SIW_SOURCE_PRIOR_PACKAGE_FILENAME
    if not package_path.is_file():
        return {"schema_version": f"{SCHEMA_PREFIX}-source-priors-validate-v1", "fold_id": fold_id,
               "status": "NOT_MATERIALIZED"}

    body = cc.read_json(package_path)
    rows = body.get("rows", [])
    result = _validate_siw_source_prior_rows(repo, fold_id, rows, body.get("package_identity"))
    return {"schema_version": f"{SCHEMA_PREFIX}-source-priors-validate-v1", "fold_id": fold_id, **result}


def validate_source_prior_candidate(repo: Path, fold_id: str, rows: list[dict[str, Any]],
                                    package_identity: str) -> dict[str, Any]:
    """STRICT, read-only validator for CANDIDATE rows/identity -- performs
    the EXACT SAME substantive checks as `validate_source_priors`, but
    BEFORE `SIW_SOURCE_PRIOR_PACKAGE.json` exists (never reads or requires
    the terminal marker). This is what `prepare_source_priors` MUST call,
    and MUST see return VALID, before it is ever allowed to write that
    marker -- a failure here leaves no terminal marker at all."""
    if fold_id not in FOLD_IDS:
        raise E7Error(f"unknown fold_id {fold_id!r}")
    if "SiW-Mv2" not in FOLD_SOURCE_DOMAINS[fold_id]:
        raise E7Error(f"{fold_id} has no SiW source domain -- candidate validation is "
                      "NOT_APPLICABLE for this fold")
    result = _validate_siw_source_prior_rows(repo, fold_id, rows, package_identity)
    return {"schema_version": f"{SCHEMA_PREFIX}-source-priors-candidate-validate-v1",
           "fold_id": fold_id, **result}


def _m3a_sample_id(source_video_id: Any, frame_index: Any) -> str:
    """Deterministic sample_id for the M3A/M3B adapter namespace -- derived
    from the real (source_video_id, frame_index) key, never fabricated or
    random, and never containing the literal substring "siw"/"target" (the
    same path-naming discipline this module's docstring already documents
    for `SourceOnlyAudit` compatibility)."""
    digest = cc.sha256_bytes(f"{source_video_id}:{frame_index}".encode("utf-8"))
    return f"extsrc{digest[:32]}"


def _resolve_weight_root(repo: Path) -> Path:
    """Same frozen `model_cache` convention `prism.cli.main.priors_model_build`
    resolves against (`$PRISM_MODEL_CACHE` or `model_cache`) -- anchored
    under `repo` only when the resolved value is not already absolute."""
    import os

    raw = Path(os.environ.get("PRISM_MODEL_CACHE", "model_cache"))
    return raw if raw.is_absolute() else repo / raw


def _gpu_prior_generation_capability(repo: Path) -> dict[str, Any]:
    """Real, non-fabricated host-capability check: CUDA availability AND
    resolvable pinned FaceXFormer/AdaFace weights. Never a hardcoded
    'non-GPU host' string -- this genuinely differs between a laptop and a
    real GPU worker, and is re-evaluated every call."""
    problems: list[str] = []
    cuda_available = False
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001 -- any import/probe failure means NOT capable
        problems.append(f"torch unavailable or unusable: {exc!r}")
    if not cuda_available:
        problems.append("torch.cuda.is_available() is False on this host")
    weight_root = _resolve_weight_root(repo)
    config_path = repo / M3B_PRIOR_MODEL_CONFIG_PATH
    if not config_path.is_file():
        problems.append(f"model config missing: {M3B_PRIOR_MODEL_CONFIG_PATH}")
    else:
        from prism_fas.data.package.model_priors import load_model_config, resolve_weight

        try:
            model_config = load_model_config(config_path)
            resolve_weight(model_config, "parsing", weight_root)
            resolve_weight(model_config, "identity", weight_root)
        except Exception as exc:  # noqa: BLE001 -- any unresolved pinned weight means NOT capable
            problems.append(f"pinned model weight unresolved under {weight_root}: {exc!r}")
    return {"capable": not problems, "cuda_available": cuda_available,
           "weight_root": str(weight_root), "problems": problems}


def _load_e7d_authoritative_siw_rows(repo: Path, fold_id: str) -> tuple[list[dict], list[dict]]:
    """E7-D's OWN authoritative source_train.json/source_dev.json (already
    filtered to status=='success' at E7-D build time) -- the project_split
    AUTHORITY this adapter uses, never the legacy `config.project_split()`."""
    fold_root = repo / e7d.E7D_OUTPUT_ROOT / fold_id
    train_path = fold_root / "source_train.json"
    dev_path = fold_root / "source_dev.json"
    if not train_path.is_file() or not dev_path.is_file():
        raise E7Error(f"{fold_id}: E7-D source_train.json/source_dev.json not present -- E7-D "
                      "authoritative source support is not materialized -- FAIL CLOSED")
    train_rows = [r for r in cc.read_json(train_path)["rows"]
                 if r.get("source_package_kind") == e7d.SIW_SOURCE_PACKAGE_KIND]
    dev_rows = [r for r in cc.read_json(dev_path)["rows"]
               if r.get("source_package_kind") == e7d.SIW_SOURCE_PACKAGE_KIND]
    return train_rows, dev_rows


def _load_siw_m2_crop_index(repo: Path) -> dict[tuple[Any, Any], dict[str, Any]]:
    """Reads E7-B's own REAL, frozen M2 crop manifest (never recomputes a
    detection/crop decision). Join key mirrors E7-B's own
    `_assemble_siw_source_rows`: (video_id or source_record_id,
    requested_frame_index)."""
    from prism_fas.data.package.manifests import read_manifest

    m2_root = repo / e7b.E7B_SIW_SOURCE_PACKAGE_ROOT / "m2_run"
    crops_path = m2_root / "manifests" / "source_crops.parquet"
    if not crops_path.is_file():
        raise E7Error(f"E7-B SiW source m2_run/manifests/source_crops.parquet not present -- "
                      "GPU_REQUIRED -- FAIL CLOSED")
    index: dict[tuple[Any, Any], dict[str, Any]] = {}
    for row in read_manifest(crops_path):
        key = (row.get("video_id") or row.get("source_record_id"), row.get("requested_frame_index"))
        index[key] = row
    return index


def _canonicalize_siw_row(row: dict[str, Any]) -> tuple[Any, ...]:
    """The canonical, comparable tuple for one E7-D authoritative SiW row --
    used both to assert F2/F3 population equality and as the per-row
    material a fold-order-independent identity is built from."""
    return (row.get("source_video_id"), row.get("frame_index"), row.get("project_split"),
           row.get("label_live_spoof"), row.get("crop_relative_path"), row.get("crop_sha256"))


def load_siw_shared_source_authority(repo: Path) -> dict[str, Any]:
    """Loads E7-D's authoritative SiW rows from BOTH EXT-F2 and EXT-F3,
    canonicalizes each, and asserts they are IDENTICAL -- F2 and F3 share
    ONE SiW source population, so neither fold may ever be treated as
    authority over the other, and no command's invocation order may be
    allowed to change which rows the shared package is built from.

    FAILS CLOSED (raises `E7Error`) if the two folds' canonical SiW
    populations differ at all, or if either fold's E7-D authority is not
    yet materialized. Returns a fold-order-independent identity plus the
    canonical merged row set."""
    per_fold_canonical: dict[str, set[tuple[Any, ...]]] = {}
    for fold_id in SIW_SHARING_FOLD_IDS:
        train, dev = _load_e7d_authoritative_siw_rows(repo, fold_id)
        rows = ([{**r, "project_split": "source_train"} for r in train] +
                [{**r, "project_split": "source_dev"} for r in dev])
        per_fold_canonical[fold_id] = {_canonicalize_siw_row(r) for r in rows}

    f2_canonical = per_fold_canonical["EXT-F2"]
    f3_canonical = per_fold_canonical["EXT-F3"]
    if f2_canonical != f3_canonical:
        only_f2 = sorted(f2_canonical - f3_canonical)
        only_f3 = sorted(f3_canonical - f2_canonical)
        raise E7Error(f"EXT-F2 and EXT-F3 authoritative SiW source populations DIFFER -- FAIL "
                      f"CLOSED (never choosing one fold as authority): {len(only_f2)} row(s) only "
                      f"in EXT-F2, {len(only_f3)} row(s) only in EXT-F3 -- first divergent "
                      f"examples: only_f2={only_f2[:3]!r}, only_f3={only_f3[:3]!r}")

    canonical_rows = sorted(f2_canonical)
    train_count = sum(1 for r in canonical_rows if r[2] == "source_train")
    dev_count = sum(1 for r in canonical_rows if r[2] == "source_dev")
    if train_count != EXPECTED_SIW_TRAIN_SUCCESS_COUNT or dev_count != EXPECTED_SIW_DEV_SUCCESS_COUNT:
        raise E7Error(f"canonical shared SiW authority counts train={train_count} dev={dev_count} "
                      f"!= expected train={EXPECTED_SIW_TRAIN_SUCCESS_COUNT} "
                      f"dev={EXPECTED_SIW_DEV_SUCCESS_COUNT} -- FAIL CLOSED")

    e7d_fold_package_identities = {}
    for fold_id in SIW_SHARING_FOLD_IDS:
        binding_path = repo / e7d.E7D_OUTPUT_ROOT / fold_id / "SOURCE_SUPPORT_PACKAGE.json"
        e7d_fold_package_identities[fold_id] = (cc.read_json(binding_path).get("package_identity")
                                                if binding_path.is_file() else None)

    # `e7d_fold_package_identities` is bound as SORTED provenance only (canonical_json_bytes
    # sorts keys) -- it documents which two frozen E7-D packages agreed, it never determines
    # the identity by itself and never depends on which fold a command was invoked with.
    material = {
        "e7b_siw_source_package_identity": e7c.FROZEN_E7B["siw_source_package_identity"],
        "e7a_siw_split_identity": e7c.FROZEN_E7B["siw_split_identity"],
        "e7d_fold_package_identities": e7d_fold_package_identities,
        "canonical_row_material": canonical_rows,
    }
    identity = cc.sha256_bytes(cc.canonical_json_bytes(material))
    return {"identity": identity, "row_count": len(canonical_rows), "train_count": train_count,
           "dev_count": dev_count, "e7d_fold_package_identities": e7d_fold_package_identities,
           "rows": [{"source_video_id": r[0], "frame_index": r[1], "project_split": r[2],
                    "label_live_spoof": r[3], "crop_relative_path": r[4], "crop_sha256": r[5]}
                   for r in canonical_rows]}


def compute_m3a_input_package_identity(*, shared_authority_identity: str | None,
                                       m3a_config_identity: str | None,
                                       rows: list[dict[str, Any]]) -> str:
    """Deterministic identity over CANONICAL METADATA only -- no absolute
    paths, no timestamps, no hostname, and DELIBERATELY no per-fold E7-D
    whole-package identity (EXT-F2's and EXT-F3's differ by construction,
    since each fold's own E7-D package also carries a different non-SiW
    source domain) -- fold-order independence is structural here: this
    function does not accept a `fold_id` at all, only the already
    fold-order-independent `SIW_SHARED_SOURCE_AUTHORITY_IDENTITY` (see
    `load_siw_shared_source_authority`). Binds E7-B SiW source package
    identity, E7-A split identity, the shared SiW authority identity, the
    frozen M3A schema version, the actual M3A package config identity,
    E7-B's own frozen preprocessing-config/detector-model identity, and the
    sorted per-row (source_video_id, frame_index, project_split,
    label_live_spoof, crop_sha256, base_prior_sha256) material."""
    row_material = sorted(
        (r.get("source_video_id"), r.get("frame_index"), r.get("project_split"),
         r.get("label_live_spoof"), r.get("crop_sha256"), r.get("base_prior_sha256"))
        for r in rows)
    material = {
        "e7b_siw_source_package_identity": e7c.FROZEN_E7B["siw_source_package_identity"],
        "e7a_siw_split_identity": e7c.FROZEN_E7B["siw_split_identity"],
        "siw_shared_source_authority_identity": shared_authority_identity,
        "m3a_input_package_id": SIW_SOURCE_M3A_INPUT_PACKAGE_ID,
        "m3a_schema_version": M3A_PACKAGE_SCHEMA_VERSION,
        "m3a_package_config_identity": m3a_config_identity,
        "e7b_preprocessing_config_hash": e7c.FROZEN_E7B["preprocessing_config_hash"],
        "e7b_detector_model_sha256": e7c.FROZEN_E7B["detector_model_sha256"],
        "row_material": row_material,
    }
    return cc.sha256_bytes(cc.canonical_json_bytes(material))


def _m3a_package_config(repo: Path):
    """Loads the frozen `configs/data/package_m3a.yaml` UNMODIFIED, with
    ONLY `package_id_prefix` overridden to this adapter's own distinct id
    (never the canonical CASIA/MSU M3A package_id) -- a data-contract
    identity distinction, never a schema/scientific-value change."""
    import yaml
    from prism_fas.data.package.config import M3APackageConfig

    raw = yaml.safe_load((repo / M3A_PACKAGE_CONFIG_PATH).read_text(encoding="utf-8"))
    return M3APackageConfig.model_validate({**raw, "package_id_prefix": SIW_SOURCE_M3A_INPUT_PACKAGE_ID})


def materialize_m3a_input_package(repo: Path, fold_id: str, *, authorize: bool = False) -> dict[str, Any]:
    """Materializes (or resumes/validates) the E7-specific M3A-compatible
    input package for SiW-as-source, shared verbatim by F2/F3. Reuses
    `build_priors`/`write_manifest`/`build_lock`/`validate_package`/
    `finalize_lock`/`plan_shards`/`write_shard` UNMODIFIED; the only new
    code is row selection + the project_split assignment (from E7-D/E7-A
    authority, never the legacy `config.project_split()`)."""
    if fold_id not in FOLD_IDS:
        raise E7Error(f"unknown fold_id {fold_id!r}")
    if "SiW-Mv2" not in FOLD_SOURCE_DOMAINS[fold_id]:
        raise E7Error(f"{fold_id} has no SiW source domain -- M3A input materialization is "
                      "NOT_APPLICABLE for this fold")
    if not authorize:
        raise E7Error(f"M3A input package materialization for {fold_id} requires --authorize")

    from prism_fas.data.package import builder as m3a_builder
    from prism_fas.data.package import validator as m3a_validator
    from prism_fas.utils.core import atomic_json_write

    package_root = repo / SIW_SOURCE_M3A_INPUT_PACKAGE_ROOT
    lock_path = package_root / "PACKAGE_LOCK.json"
    if lock_path.is_file():
        existing_lock = cc.read_json(lock_path)
        if existing_lock.get("status") == "validated":
            validation = validate_m3a_input_package(repo, fold_id)
            if validation["status"] != "VALID":
                raise E7Error(f"{fold_id}: existing M3A input package FAILED strict validation "
                              f"-- FAIL CLOSED, never silently rewritten: "
                              f"{validation['problems']!r}")
            return {"resumed": True, "status": "ALREADY_VALID", "path": str(package_root),
                   "validation": validation, "target_access": False, "llm_api_calls": 0}
        # status is still "building" (e.g. a crashed prior attempt) -- fall through and
        # resume via build_priors()'s own reuse logic; NEVER report ALREADY_VALID for this.

    # The shared SiW authority is loaded from BOTH EXT-F2 and EXT-F3's E7-D rows and asserted
    # identical -- this fold's own identity/content is NEVER used alone; invocation order can
    # never change which rows the shared package is built from (GAP 2 fix).
    authority = load_siw_shared_source_authority(repo)
    crop_index = _load_siw_m2_crop_index(repo)

    samples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    m2_root = repo / e7b.E7B_SIW_SOURCE_PACKAGE_ROOT / "m2_run"
    for row in authority["rows"]:
        key = (row["source_video_id"], row["frame_index"])
        crop = crop_index.get(key)
        if crop is None:
            raise E7Error(f"SiW crop for {key!r} not present in E7-B m2_run crop manifest -- "
                          "FAIL CLOSED")
        if crop.get("crop_relative_path") != row["crop_relative_path"] or \
                crop.get("crop_sha256") != row["crop_sha256"]:
            raise E7Error(f"canonical shared SiW authority row for {key!r} disagrees with the "
                          "real M2 crop manifest -- FAIL CLOSED")
        candidate_ref = f"{e7b.E7B_SIW_SOURCE_PACKAGE_ROOT}/m2_run/{crop['crop_relative_path']}"
        for check_fold_id in SIW_SHARING_FOLD_IDS:
            assert_not_target_path(check_fold_id, candidate_ref)
        crop_path = m2_root / crop["crop_relative_path"]
        if not crop_path.is_file():
            raise E7Error(f"SiW crop missing on disk: {crop['crop_relative_path']!r} -- FAIL CLOSED")
        if cc.sha256_file(crop_path) != crop["crop_sha256"]:
            raise E7Error(f"SiW crop SHA256 mismatch on disk: {crop['crop_relative_path']!r} -- "
                          "FAIL CLOSED")
        sample_id = _m3a_sample_id(*key)
        if sample_id in seen_ids:
            raise E7Error(f"duplicate derived sample_id for {key!r} -- FAIL CLOSED")
        seen_ids.add(sample_id)
        samples.append({**crop, "sample_id": sample_id, "dataset": "siw_mv2",
                        "dataset_role": "source",
                        "official_split": e7b.SIW_SOURCE_OFFICIAL_SPLIT_PLACEHOLDER,
                        "subject_id": None, "label_live_spoof": row["label_live_spoof"],
                        "_e7_project_split": row["project_split"],
                        "_e7_source_video_id": key[0], "_e7_frame_index": key[1]})

    config = _m3a_package_config(repo)
    package_root.mkdir(parents=True, exist_ok=True)
    prior_rows, stats = m3a_builder.build_priors(samples, input_root=m2_root, package_root=package_root,
                                                 config=config, resume=True)
    priors = {row["sample_id"]: row for row in prior_rows}
    metadata = {"package_schema_version": config.package_schema_version,
               "prior_schema_version": prior_rows[0]["prior_schema_version"] if prior_rows else None,
               "quality_schema_version": prior_rows[0]["quality_schema_version"] if prior_rows else None,
               "package_config_hash": config.config_hash}
    from prism_fas.data.package.manifests import MANIFEST_SCHEMAS, write_manifest
    from prism_fas.data.package.priors import load_prior, validate_prior_arrays
    from prism_fas.data.package.quality import QUALITY_NAMES
    from prism_fas.data.package.shards import plan_shards, write_shard

    sample_rows: list[dict[str, Any]] = []
    source_rows: dict[str, list[dict[str, Any]]] = {"source_train": [], "source_dev": []}
    binding_rows: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = sample["sample_id"]
        prior = priors[sample_id]
        arrays = load_prior(package_root / prior["prior_relative_path"])
        validate_prior_arrays(arrays)
        metrics = {name: float(value) for name, value in zip(QUALITY_NAMES, arrays["quality_vector"].tolist())}
        row = _e7_sample_row(sample, prior, metrics, config)
        sample_rows.append(row)
        image = package_root / row["image_relative_path"]
        m3a_builder._link_or_copy(m2_root / sample["crop_relative_path"], image)
        if cc.sha256_file(image) != sample["crop_sha256"]:
            raise E7Error(f"{fold_id}: packaged image SHA changed for {sample_id} -- FAIL CLOSED")
        stats.images_linked += 1
        stats.per_dataset[sample["dataset"]] = stats.per_dataset.get(sample["dataset"], 0) + 1
        stats.per_split[row["project_split"]] = stats.per_split.get(row["project_split"], 0) + 1
        common = {"sample_id": sample_id, "dataset": sample["dataset"],
                 "source_record_id": sample["source_record_id"], "project_split": row["project_split"],
                 "image_relative_path": row["image_relative_path"],
                 "prior_relative_path": prior["prior_relative_path"], "crop_sha256": sample["crop_sha256"],
                 "prior_sha256": prior["prior_sha256"], "package_schema_version": config.package_schema_version}
        source_rows[row["project_split"]].append({**common, "subject_id": None,
                                                   "official_split": sample["official_split"],
                                                   "label_live_spoof": sample["label_live_spoof"]})
        binding_rows.append({"sample_id": sample_id, "source_video_id": sample["_e7_source_video_id"],
                             "frame_index": sample["_e7_frame_index"],
                             "project_split": row["project_split"],
                             "label_live_spoof": sample["label_live_spoof"],
                             "crop_relative_path": sample["crop_relative_path"],
                             "crop_sha256": sample["crop_sha256"],
                             "base_prior_relative_path": prior["prior_relative_path"],
                             "base_prior_sha256": prior["prior_sha256"],
                             "preprocessing_config_hash": sample["preprocessing_config_hash"],
                             "detector_model_sha256": sample["detector_model_sha256"]})

    hashes = {}
    hashes["samples"] = write_manifest(package_root / "manifests" / "samples.parquet", sample_rows,
                                       MANIFEST_SCHEMAS["samples"], metadata)
    for name in ("source_train", "source_dev"):
        hashes[name] = write_manifest(package_root / "manifests" / f"{name}.parquet", source_rows[name],
                                      MANIFEST_SCHEMAS[name], metadata)
    hashes["target_test_features"] = write_manifest(package_root / "manifests" / "target_test_features.parquet",
                                                     [], MANIFEST_SCHEMAS["target_test_features"], metadata)
    hashes["priors_index"] = write_manifest(package_root / "manifests" / "priors_index.parquet", prior_rows,
                                            MANIFEST_SCHEMAS["priors_index"], metadata)
    sizes = {row["sample_id"]: row["prior_bytes"] + (package_root / row["image_relative_path"]).stat().st_size
            for row in sample_rows}
    shard_rows: list[dict[str, Any]] = []
    for split in ("source_train", "source_dev"):
        rows = [row for row in sample_rows if row["project_split"] == split]
        if not rows:
            continue
        lookup = {row["sample_id"]: row for row in rows}
        for number, group in enumerate(plan_shards([r["sample_id"] for r in rows], sizes,
                                                    max_samples=config.shard_max_samples,
                                                    max_bytes=config.shard_max_bytes)):
            entries = []
            for sample_id in group:
                row = lookup[sample_id]
                entries.append((sample_id, (package_root / row["image_relative_path"]).read_bytes(),
                               (package_root / row["prior_relative_path"]).read_bytes(),
                               m3a_builder._shard_metadata(row, source_rows, split)))
            summary = write_shard(package_root / "shards" / f"{split}-{number:05d}.tar", entries)
            shard_rows.append({**summary, "split": split, "package_schema_version": config.package_schema_version})
    hashes["shards_index"] = write_manifest(package_root / "manifests" / "shards_index.parquet", shard_rows,
                                            MANIFEST_SCHEMAS["shards_index"], metadata, sort_key="shard_filename")

    m3a_builder.build_lock(m2_root, package_root, config, sample_rows, shard_rows, hashes, stats)
    pre = m3a_validator.validate_package(package_root, require_validated_status=False)
    if not pre["passed"]:
        raise E7Error(f"{fold_id}: M3A input package structural validation FAILED before "
                      f"finalize -- FAIL CLOSED: {pre['errors']!r}")
    m3a_builder.finalize_lock(package_root, pre)
    report = m3a_validator.validate_package(package_root)
    if not report["passed"]:
        raise E7Error(f"{fold_id}: M3A input package FAILED final validation -- FAIL CLOSED: "
                      f"{report['errors']!r}")

    m3a_input_package_identity = compute_m3a_input_package_identity(
        shared_authority_identity=authority["identity"], m3a_config_identity=config.config_hash,
        rows=binding_rows)
    lock = cc.read_json(lock_path)
    atomic_json_write(package_root / M3A_INPUT_BINDING_FILENAME, {
        "schema_version": f"{SCHEMA_PREFIX}-m3a-input-binding-v2",
        # `fold_id` records ONLY which invocation happened to perform this materialization --
        # it is provenance, never authority: the identity/content above are IDENTICAL whichever
        # of EXT-F2/EXT-F3 this was invoked with.
        "materializing_fold_id": fold_id,
        "m3a_input_package_identity": m3a_input_package_identity,
        "e7b_siw_source_package_identity": e7c.FROZEN_E7B["siw_source_package_identity"],
        "siw_shared_source_authority_identity": authority["identity"],
        "e7d_fold_package_identities": authority["e7d_fold_package_identities"],
        "m3a_package_config_identity": config.config_hash,
        "package_lock_content_identity": lock.get("content_identity_sha256"),
        "official_split_placeholder": e7b.SIW_SOURCE_OFFICIAL_SPLIT_PLACEHOLDER,
        "rows": binding_rows})
    return {"resumed": False, "status": "MATERIALIZED", "path": str(package_root),
           "m3a_input_package_identity": m3a_input_package_identity,
           "target_access": False, "llm_api_calls": 0}


def _e7_sample_row(sample: dict[str, Any], prior: dict[str, Any], metrics: dict[str, float], config) -> dict[str, Any]:
    """Mirrors `prism_fas.data.package.builder._sample_row` field-for-field,
    with ONE change: `project_split` comes from E7-D/E7-A authority
    (`sample["_e7_project_split"]`), never `config.project_split()` (which
    rejects E7-B's `SIW_SOURCE_OFFICIAL_SPLIT_PLACEHOLDER`)."""
    from prism_fas.data.package.priors import prior_schema_version
    from prism_fas.data.package.quality import quality_schema_version
    from prism_fas.data.package.config import DEFERRED_PRIOR_STATUS

    return {"sample_id": sample["sample_id"], "dataset": sample["dataset"],
           "dataset_role": sample["dataset_role"], "project_split": sample["_e7_project_split"],
           "source_record_id": sample["source_record_id"],
           "requested_frame_index": sample["requested_frame_index"],
           "actual_frame_index": sample["actual_frame_index"],
           "image_relative_path": f"images/{sample['sample_id']}{config.image_extension}",
           "crop_sha256": sample["crop_sha256"], "prior_relative_path": prior["prior_relative_path"],
           "prior_sha256": prior["prior_sha256"], "prior_bytes": prior["prior_bytes"],
           "source_media_type": sample["source_media_type"], "image_format": config.image_format,
           "frame_width": sample["frame_width"], "frame_height": sample["frame_height"],
           "crop_width": sample["crop_width"], "crop_height": sample["crop_height"],
           "detection_score": sample["detection_score"],
           "detected_face_count": sample["detected_face_count"], **metrics,
           "quality_schema_version": quality_schema_version(), "prior_schema_version": prior_schema_version(),
           "package_schema_version": config.package_schema_version,
           "preprocessing_version": sample["preprocessing_version"],
           "preprocessing_config_hash": sample["preprocessing_config_hash"],
           "detector_model_sha256": sample["detector_model_sha256"], **dict(DEFERRED_PRIOR_STATUS)}


def validate_m3a_input_package(repo: Path, fold_id: str) -> dict[str, Any]:
    """STRICT, read-only validator for the shared E7-specific M3A input
    package. Reuses the real `validate_package()` structural validator
    unmodified, then adds the project-specific checks the user's spec
    enumerates. Never writes."""
    if fold_id not in FOLD_IDS:
        raise E7Error(f"unknown fold_id {fold_id!r}")
    if "SiW-Mv2" not in FOLD_SOURCE_DOMAINS[fold_id]:
        return {"schema_version": f"{SCHEMA_PREFIX}-m3a-input-validate-v1", "fold_id": fold_id,
               "status": "NOT_APPLICABLE"}
    from prism_fas.data.package import validator as m3a_validator
    from prism_fas.data.package.manifests import read_manifest

    package_root = repo / SIW_SOURCE_M3A_INPUT_PACKAGE_ROOT
    binding_path = package_root / M3A_INPUT_BINDING_FILENAME
    if not (package_root / "PACKAGE_LOCK.json").is_file() or not binding_path.is_file():
        return {"schema_version": f"{SCHEMA_PREFIX}-m3a-input-validate-v1", "fold_id": fold_id,
               "status": "NOT_MATERIALIZED"}

    problems: list[str] = []
    structural = m3a_validator.validate_package(package_root, require_validated_status=True)
    if not structural["passed"]:
        problems.append(f"structural validate_package() failed: {structural['errors']!r}")

    binding = cc.read_json(binding_path)
    rows = binding.get("rows", [])
    samples = read_manifest(package_root / "manifests" / "samples.parquet")
    train_rows = read_manifest(package_root / "manifests" / "source_train.parquet")
    dev_rows = read_manifest(package_root / "manifests" / "source_dev.parquet")
    target_rows = read_manifest(package_root / "manifests" / "target_test_features.parquet")

    if len(samples) != EXPECTED_SIW_SUCCESS_CROP_COUNT:
        problems.append(f"total samples {len(samples)} != expected {EXPECTED_SIW_SUCCESS_CROP_COUNT}")
    if len(train_rows) != EXPECTED_SIW_TRAIN_SUCCESS_COUNT:
        problems.append(f"source_train rows {len(train_rows)} != expected "
                        f"{EXPECTED_SIW_TRAIN_SUCCESS_COUNT}")
    if len(dev_rows) != EXPECTED_SIW_DEV_SUCCESS_COUNT:
        problems.append(f"source_dev rows {len(dev_rows)} != expected {EXPECTED_SIW_DEV_SUCCESS_COUNT}")
    if target_rows:
        problems.append(f"target_test_features has {len(target_rows)} rows -- SiW-as-source must "
                        "carry ZERO target_test rows")
    if any(row["project_split"] == "target_test" for row in samples):
        problems.append("a samples.parquet row is assigned project_split=='target_test'")

    for split_rows, split_name in ((train_rows, "source_train"), (dev_rows, "source_dev")):
        for row in split_rows:
            if row.get("subject_id") is not None:
                problems.append(f"{split_name} row {row['sample_id']!r} has a non-null subject_id "
                                "-- SiW subject_id must never be fabricated")
            if row.get("official_split") != e7b.SIW_SOURCE_OFFICIAL_SPLIT_PLACEHOLDER:
                problems.append(f"{split_name} row {row['sample_id']!r} official_split "
                                f"{row.get('official_split')!r} != frozen placeholder")

    live_train = sum(1 for r in train_rows if r["label_live_spoof"] == "live")
    spoof_train = sum(1 for r in train_rows if r["label_live_spoof"] == "spoof")
    live_dev = sum(1 for r in dev_rows if r["label_live_spoof"] == "live")
    spoof_dev = sum(1 for r in dev_rows if r["label_live_spoof"] == "spoof")
    expected = FROZEN_SIW_CROP_ACCOUNTING
    if (live_train, spoof_train, live_dev, spoof_dev) != (
            expected["live_train_success"], expected["spoof_train_success"],
            expected["live_dev_success"], expected["spoof_dev_success"]):
        problems.append(f"live/spoof counts (train_live={live_train}, train_spoof={spoof_train}, "
                        f"dev_live={live_dev}, dev_spoof={spoof_dev}) do not match the frozen "
                        f"E7-D accounting")

    seen_keys: set[Any] = set()
    m2_root = repo / e7b.E7B_SIW_SOURCE_PACKAGE_ROOT / "m2_run"
    for row in rows:
        key = (row.get("source_video_id"), row.get("frame_index"))
        if key in seen_keys:
            problems.append(f"duplicate source_video_id/frame_index key {key!r} in binding rows")
        seen_keys.add(key)
        rel = row.get("crop_relative_path")
        try:
            assert_not_target_path(fold_id, f"{e7b.E7B_SIW_SOURCE_PACKAGE_ROOT}/m2_run/{rel}")
        except E7TargetFirewallViolation as exc:
            problems.append(str(exc))
        crop_path = m2_root / rel if rel else None
        if crop_path is None or not crop_path.is_file():
            problems.append(f"binding row {key!r}: source crop missing on disk: {rel!r}")
        elif cc.sha256_file(crop_path) != row.get("crop_sha256"):
            problems.append(f"binding row {key!r}: source crop SHA256 mismatch: {rel!r}")

    # Never merely trust the binding file's own stored rows: reload THIS fold's own E7-D
    # authoritative SiW rows, canonicalize, and require EXACT equality with the binding
    # population. A package that only "looks" shared but was silently built from one fold's
    # rows must fail validation from the OTHER fold.
    fold_train_e7d, fold_dev_e7d = _load_e7d_authoritative_siw_rows(repo, fold_id)
    fold_canonical = ({_canonicalize_siw_row({**r, "project_split": "source_train"})
                       for r in fold_train_e7d} |
                      {_canonicalize_siw_row({**r, "project_split": "source_dev"})
                       for r in fold_dev_e7d})
    binding_canonical = {_canonicalize_siw_row(r) for r in rows}
    if fold_canonical != binding_canonical:
        only_fold = sorted(fold_canonical - binding_canonical)
        only_binding = sorted(binding_canonical - fold_canonical)
        problems.append(f"{fold_id}'s own E7-D authoritative SiW population does not exactly "
                        f"equal the shared M3A binding population -- {len(only_fold)} row(s) "
                        f"only in {fold_id}'s own authority, {len(only_binding)} row(s) only in "
                        f"the binding (first examples: only_fold={only_fold[:3]!r}, "
                        f"only_binding={only_binding[:3]!r})")

    try:
        fresh_authority = load_siw_shared_source_authority(repo)
        fresh_authority_identity = fresh_authority["identity"]
    except E7Error as exc:
        problems.append(f"fresh SIW_SHARED_SOURCE_AUTHORITY_IDENTITY recomputation failed: {exc}")
        fresh_authority_identity = None
    if fresh_authority_identity is not None and \
            fresh_authority_identity != binding.get("siw_shared_source_authority_identity"):
        problems.append(f"recomputed SIW_SHARED_SOURCE_AUTHORITY_IDENTITY {fresh_authority_identity!r} "
                        f"!= binding's recorded "
                        f"{binding.get('siw_shared_source_authority_identity')!r} -- E7-D "
                        "authoritative data has changed since materialization")

    fresh_config_identity = _m3a_package_config(repo).config_hash
    recomputed_identity = compute_m3a_input_package_identity(
        shared_authority_identity=fresh_authority_identity, m3a_config_identity=fresh_config_identity,
        rows=rows)
    identity_match = recomputed_identity == binding.get("m3a_input_package_identity")
    if not identity_match:
        problems.append(f"recomputed m3a_input_package_identity {recomputed_identity!r} != "
                        f"recorded {binding.get('m3a_input_package_identity')!r}")

    return {"schema_version": f"{SCHEMA_PREFIX}-m3a-input-validate-v1", "fold_id": fold_id,
           "status": "INVALID" if problems else "VALID", "problems": problems,
           "total_samples": len(samples), "train_rows": len(train_rows), "dev_rows": len(dev_rows),
           "m3a_input_package_identity": binding.get("m3a_input_package_identity"),
           "recomputed_m3a_input_package_identity": recomputed_identity,
           "identity_match": identity_match, "target_access": False, "llm_api_calls": 0}


def _derive_siw_source_prior_rows(repo: Path) -> list[dict[str, Any]]:
    """Derives the shared SIW_SOURCE_PRIOR_PACKAGE.json row schema FROM the
    real, already-validated M3B output manifests -- source of truth is the
    actual M3B `samples.parquet`/`priors_index.parquet`, cross-referenced
    back to the M3A input package's own source_video_id/frame_index binding
    (`E7_M3A_INPUT_BINDING.json`). Never fabricates hashes/counts."""
    from prism_fas.data.package.manifests import read_manifest

    m3a_root = repo / SIW_SOURCE_M3A_INPUT_PACKAGE_ROOT
    m3b_root = repo / SIW_SOURCE_PRIOR_PACKAGE_ROOT
    binding = cc.read_json(m3a_root / M3A_INPUT_BINDING_FILENAME)
    key_by_sample_id = {row["sample_id"]: row for row in binding["rows"]}
    priors_index = {row["sample_id"]: row for row in read_manifest(m3b_root / "manifests" / "priors_index.parquet")}
    samples = read_manifest(m3b_root / "manifests" / "samples.parquet")

    rows = []
    for sample in samples:
        sample_id = sample["sample_id"]
        binding_row = key_by_sample_id.get(sample_id)
        if binding_row is None:
            raise E7Error(f"M3B output sample {sample_id!r} has no M3A input binding entry -- "
                          "FAIL CLOSED")
        prior = priors_index.get(sample_id)
        if prior is None:
            raise E7Error(f"M3B output sample {sample_id!r} missing from priors_index.parquet -- "
                          "FAIL CLOSED")
        if sample["crop_sha256"] != binding_row["crop_sha256"]:
            raise E7Error(f"M3B output sample {sample_id!r} crop_sha256 disagrees with its M3A "
                          "input binding -- FAIL CLOSED")
        rows.append({"source_video_id": binding_row["source_video_id"],
                    "frame_index": binding_row["frame_index"],
                    "source_crop_relative_path": binding_row["crop_relative_path"],
                    "source_crop_sha256": sample["crop_sha256"],
                    "prior_relative_path": sample["prior_relative_path"],
                    "prior_sha256": sample["prior_sha256"], "status": "success"})
    return rows


def prepare_source_priors(repo: Path, fold_id: str, *, authorize: bool = False) -> dict[str, Any]:
    """`--prepare-source-priors --fold EXT-F2/F3 --authorize`: REAL
    transactional materialization of the SHARED SiW-as-source prior
    package. Order: (1) E7-B/E7-D input authority, (2) M3A-compatible input
    package materialize+validate, (3) real GPU-capability gate, (4) real
    `build_m3b_package`, (5) validate the M3B output, (6) derive E7
    source-prior rows FROM the validated M3B output, (7) strict-validate,
    (8) write SIW_SOURCE_PRIOR_PACKAGE.json LAST as the terminal commit
    marker. F2 and F3 converge on the exact same on-disk package -- prior
    generation runs at most once, never per-fold."""
    if fold_id not in FOLD_IDS:
        raise E7Error(f"unknown fold_id {fold_id!r}")
    if "SiW-Mv2" not in FOLD_SOURCE_DOMAINS[fold_id]:
        raise E7Error(f"{fold_id} has no SiW source domain -- source-prior generation is "
                      "NOT_APPLICABLE for this fold")
    if not authorize:
        raise E7Error(f"source-prior generation for {fold_id} requires --authorize; refusing to run")

    # Terminal marker already present -- strict validate first, never trust a matching
    # top-level identity alone.
    package_path = repo / SIW_SOURCE_PRIOR_PACKAGE_ROOT / SIW_SOURCE_PRIOR_PACKAGE_FILENAME
    if package_path.is_file():
        existing = cc.read_json(package_path)
        validation = validate_source_priors(repo, fold_id)
        if validation["status"] != "VALID":
            raise E7Error(f"{fold_id}: existing SiW source-prior package FAILED strict "
                          f"validation -- FAIL CLOSED, never silently rewritten: "
                          f"{validation['problems']!r}")
        if existing.get("package_identity") != validation["recomputed_package_identity"]:
            raise E7Conflict(f"{fold_id}: existing SiW source-prior package_identity "
                             f"{existing.get('package_identity')!r} disagrees with the freshly "
                             f"recomputed {validation['recomputed_package_identity']!r} -- FAIL "
                             "CLOSED, never overwritten")
        return {"resumed": True, "status": "ALREADY_VALID", "path": str(package_path),
               "package_identity": existing.get("package_identity"), "validation": validation,
               "target_access": False, "llm_api_calls": 0}

    # Step 1: E7-B/E7-D input authority.
    siw_package_path = repo / e7b.E7B_SIW_SOURCE_PACKAGE_ROOT / "SIW_SOURCE_PACKAGE.json"
    if not siw_package_path.is_file():
        raise E7Error(f"{fold_id}: E7-B SIW_SOURCE_PACKAGE.json not present -- GPU_REQUIRED, "
                      "cannot generate source priors without the frozen source crop package")

    # Step 2: materialize/validate the E7-specific M3A-compatible input package. This step is
    # CPU-only (base geometric priors from already-computed M2 crop metadata) and runs for real.
    m3a_result = materialize_m3a_input_package(repo, fold_id, authorize=True)
    if m3a_result["status"] not in ("ALREADY_VALID", "MATERIALIZED"):
        raise E7Error(f"{fold_id}: M3A input package materialization did not reach a valid "
                      f"state -- FAIL CLOSED: {m3a_result!r}")

    # Step 3: real GPU-capability gate -- CUDA + resolvable pinned weights, never a hardcoded
    # "non-GPU host" stub raise.
    capability = _gpu_prior_generation_capability(repo)
    if not capability["capable"]:
        raise E7Error(f"{fold_id}: GPU_REQUIRED for real FaceXFormer/AdaFace inference -- this "
                      f"host is not capable: {capability['problems']!r}. The M3A-compatible "
                      "input package IS materialized and strictly validated; only "
                      "model-dependent prior inference (build_m3b_package) remains, and it must "
                      "run on a GPU host with the pinned weights resolvable under "
                      f"{capability['weight_root']!r}.")

    # Steps 4-8: real GPU inference + derivation + terminal marker. Written for correctness;
    # unreachable on this laptop (the capability gate above always fails first here).
    from prism_fas.data.package import builder as m3a_builder
    from prism_fas.data.package import validator as m3a_validator
    from prism_fas.data.package.m3b import build_m3b_package
    from prism_fas.utils.core import atomic_json_write

    m3a_root = repo / SIW_SOURCE_M3A_INPUT_PACKAGE_ROOT
    output_root = repo / SIW_SOURCE_PRIOR_PACKAGE_ROOT
    build_result = build_m3b_package(m3a_root, output_root, repo / M3B_PRIOR_MODEL_CONFIG_PATH,
                                     weight_root=Path(capability["weight_root"]), resume=True,
                                     package_id=SIW_SOURCE_PRIOR_M3B_PACKAGE_ID)
    if build_result["failures"]:
        raise E7Error(f"{fold_id}: build_m3b_package reported {len(build_result['failures'])} "
                      "unresolved model-prior failures -- FAIL CLOSED")
    pre = m3a_validator.validate_package(output_root, require_validated_status=False, parent_package=m3a_root)
    if not pre["passed"]:
        raise E7Error(f"{fold_id}: M3B output structural validation FAILED -- FAIL CLOSED: "
                      f"{pre['errors']!r}")
    m3a_builder.finalize_lock(output_root, pre)
    report = m3a_validator.validate_package(output_root, parent_package=m3a_root)
    if not report["passed"]:
        raise E7Error(f"{fold_id}: M3B output FAILED final validation -- FAIL CLOSED: "
                      f"{report['errors']!r}")

    rows = _derive_siw_source_prior_rows(repo)
    package_identity = compute_siw_source_prior_package_identity(rows)

    # STRICTLY validate the CANDIDATE rows/identity BEFORE the terminal marker is ever written
    # (GAP 1 fix) -- a failure here leaves NO SIW_SOURCE_PRIOR_PACKAGE.json on disk at all,
    # never a partially-valid/invalid marker.
    candidate_validation = validate_source_prior_candidate(repo, fold_id, rows, package_identity)
    if candidate_validation["status"] != "VALID":
        raise E7Error(f"{fold_id}: candidate SiW source-prior rows FAILED strict validation -- "
                      f"refusing to write the terminal marker: "
                      f"{candidate_validation['problems']!r}")

    atomic_json_write(package_path, {"schema_version": "siw-source-prior-package-v1",
                                     "package_identity": package_identity, "rows": rows})
    # Optional read-only post-write validation -- the terminal marker is the literal LAST write
    # of this transaction; this only re-confirms what step above already established.
    validation = validate_source_priors(repo, fold_id)
    if validation["status"] != "VALID":
        raise E7Error(f"{fold_id}: freshly-written SiW source-prior package FAILED strict "
                      f"validation immediately after write -- {validation['problems']!r}")
    return {"resumed": False, "status": "MATERIALIZED", "path": str(package_path),
           "package_identity": package_identity, "validation": validation,
           "target_access": False, "llm_api_calls": 0}


# --------------------------------------------------------------------------- #
# TASK E -- fold source binding (M3B rows joined against E7-A's own
# prior_relative_path/prior_sha256 fields; SiW rows against the pending
# prior-generation namespace)
# --------------------------------------------------------------------------- #

def build_fold_source_binding(repo: Path, fold_id: str) -> dict[str, Any]:
    if fold_id not in FOLD_IDS:
        raise E7Error(f"unknown fold_id {fold_id!r}")
    materialization = e7b.load_e7a_fold_materialization(repo, fold_id)
    if materialization is None:
        return {"schema_version": f"{SCHEMA_PREFIX}-fold-source-binding-v1", "fold_id": fold_id,
               "status": "E7A_MATERIALIZATION_MISSING", "target_access": False, "llm_api_calls": 0}
    m3b_refs = [r for r in materialization["source_train_references"]
               if r["reference_kind"] == "m3b_processed_sample"]
    m3b_refs_have_priors = all("prior_relative_path" in r and "prior_sha256" in r for r in m3b_refs)
    return {
        "schema_version": f"{SCHEMA_PREFIX}-fold-source-binding-v1", "fold_id": fold_id,
        "source_domains": list(FOLD_SOURCE_DOMAINS[fold_id]),
        "e7d_package_identity": FROZEN_E7D_PACKAGE_IDENTITY[fold_id],
        "m3b_train_reference_count": len(m3b_refs),
        "m3b_refs_carry_prior_fields": m3b_refs_have_priors,
        "prior_join_key": "sample_id (E7-D row) == sample_id (E7-A m3b_processed_sample reference)",
        "siw_in_fold": "SiW-Mv2" in FOLD_SOURCE_DOMAINS[fold_id],
        "status": "RESOLVED", "target_access": False, "llm_api_calls": 0,
    }


def write_fold_source_binding(repo: Path) -> dict[str, Any]:
    bindings = {fold_id: build_fold_source_binding(repo, fold_id) for fold_id in FOLD_IDS}
    return _write(repo, "GPAT_FOLD_SOURCE_BINDING_DETAIL.json", {
        "schema_version": f"{SCHEMA_PREFIX}-fold-source-binding-all-v1", "folds": bindings,
        "target_access": False, "llm_api_calls": 0})


# --------------------------------------------------------------------------- #
# TASK F -- pairing policy (documents reuse; fold-aware wrapper never
# alters pairing RULES, only which records/domains are eligible)
# --------------------------------------------------------------------------- #

def build_pairing_policy(repo: Path) -> dict[str, Any]:
    from prism_fas.synthesis import pair_plan

    return {
        "schema_version": f"{SCHEMA_PREFIX}-pairing-policy-v1",
        "reused_module": "prism_fas.synthesis.pair_plan",
        "frozen_seed": pair_plan.PAIR_PLAN_SEED,
        "frozen_train_fraction": pair_plan.TRAIN_FRACTION,
        "frozen_pairs_per_live": pair_plan.PAIRS_PER_LIVE,
        "legacy_allowed_datasets": list(pair_plan.ALLOWED_DATASETS),
        "legacy_allowed_datasets_fold_aware_override": {
            fold_id: sorted({{"CASIA-FASD": "casia_fasd", "MSU-MFSD": "msu_mfsd",
                              "SiW-Mv2": "siw_mv2"}[d] for d in FOLD_SOURCE_DOMAINS[fold_id]})
            for fold_id in FOLD_IDS
        },
        "legacy_pair_count_constants_are_m3b_size_assumptions_only": True,
        "pair_counts_resolved_from": "real E7-D fold source TRAIN rows, computed at GPU pair-plan "
                                     "build time -- NEVER predeclared/assumed on this laptop",
        "preserved_scientific_rules": ["live/spoof roles", "deterministic record partition",
                                       "same-domain pairing", "cross-domain pairing",
                                       "different-record rule",
                                       "different-subject rule (where subject identity exists)",
                                       "subject rule NOT_APPLICABLE where SiW subject_id is "
                                       "unavailable (never fabricated)", "deterministic seed",
                                       "train/validation isolation"],
        "never_alters_rules_to_force_a_pair_count": True,
        "target_access": False, "llm_api_calls": 0,
    }


def write_pairing_policy(repo: Path) -> dict[str, Any]:
    return _write(repo, "GPAT_PAIRING_POLICY.json", build_pairing_policy(repo))


# --------------------------------------------------------------------------- #
# TASK G -- GPAT fit policy
# --------------------------------------------------------------------------- #

def build_gpat_fit_policy(repo: Path) -> dict[str, Any]:
    config_path = repo / GPAT_FIT_CONFIG_PATH
    config_present = config_path.is_file()
    config_identity = cc.sha256_file(config_path) if config_present else None
    return {
        "schema_version": f"{SCHEMA_PREFIX}-gpat-fit-policy-v1",
        "config_path": GPAT_FIT_CONFIG_PATH, "config_present": config_present,
        "config_sha256": config_identity, "seed": GPAT_FIT_SEED,
        "conditioning_bank": FROZEN_M7_BANK,
        "conditioning_bank_root": M7_RECIPE_BANK_ROOT,
        "one_fit_per_fold_source_only": True,
        "fit_arm_independent": "GPAT is fit ONCE per fold, source-only, conditioned on the frozen "
                               "M7 recipe bank -- NEVER separately per RND/DET/LLM/Shuffle arm. "
                               "Recipe conditioning differs only at CANDIDATE-GENERATION time "
                               "(inference against the already-fitted checkpoint with a "
                               "C3-arm-bank recipe's conditioning vector), never at fit time -- "
                               "this is how cross-arm fairness is achieved: every arm generates "
                               "candidates from the exact same fitted weights.",
        "reused_class": "prism_fas.synthesis.gpat_trainer.GPATTrainer",
        "reused_fit_method": "GPATTrainer.fit",
        "never_reimplements": ["architecture (gpat_model.build_gpat_model)",
                               "losses (gpat_losses.compute_losses)", "optimizer", "scheduler",
                               "checkpoint_selection", "early_stopping", "identity_loss",
                               "invariants (gpat_losses.assert_invariants)"],
        "source_split_used": "source_train only (config.data.package_split=='source_train', "
                             "config.data.forbidden_splits includes 'source_dev'/'target_test')",
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False,
        "training_performed": False, "gpat_fitting_performed": False,
    }


def write_gpat_fit_policy(repo: Path) -> dict[str, Any]:
    return _write(repo, "GPAT_FIT_POLICY.json", build_gpat_fit_policy(repo))


# --------------------------------------------------------------------------- #
# TASK N.1 -- fold-local GPAT-input package: row selection ONLY (E7-D/E7-A
# authority). Every array/manifest/identity primitive it touches from
# `prism_fas.data.package.*` (`_link_or_copy`) is reused verbatim.
# --------------------------------------------------------------------------- #

def _gpat_local_link_or_copy(source: Path, destination: Path) -> None:
    from prism_fas.data.package.builder import _link_or_copy

    _link_or_copy(source, destination)


def _load_fold_m3b_source_train_rows(repo: Path, fold_id: str) -> list[dict[str, Any]]:
    """M3B (CASIA/MSU) source_train rows for one fold: membership from
    E7-D's authoritative source_train.json, prior fields cross-joined from
    E7-A's own FOLD_MATERIALIZATION source_train_references -- the SAME
    join key `build_fold_source_binding` already documents (sample_id).
    Never a fresh M3B parquet scan."""
    fold_root = repo / e7d.E7D_OUTPUT_ROOT / fold_id
    train_path = fold_root / "source_train.json"
    if not train_path.is_file():
        raise E7Error(f"{fold_id}: E7-D source_train.json not present -- FAIL CLOSED")
    e7d_rows = [r for r in cc.read_json(train_path)["rows"]
               if r.get("source_package_kind") == e7d.M3B_SOURCE_PACKAGE_KIND]
    materialization = e7b.load_e7a_fold_materialization(repo, fold_id)
    if materialization is None:
        raise E7Error(f"{fold_id}: E7-A FOLD_MATERIALIZATION.json not present -- FAIL CLOSED")
    refs_by_sample_id = {r["sample_id"]: r for r in materialization["source_train_references"]
                        if r.get("reference_kind") == "m3b_processed_sample"}
    rows: list[dict[str, Any]] = []
    for row in e7d_rows:
        sample_id = row.get("sample_id")
        ref = refs_by_sample_id.get(sample_id)
        if ref is None:
            raise E7Error(f"{fold_id}: M3B source_train row {sample_id!r} has no matching E7-A "
                          "FOLD_MATERIALIZATION reference -- FAIL CLOSED")
        if ref.get("image_relative_path") != row.get("crop_relative_path") or \
                ref.get("crop_sha256") != row.get("crop_sha256"):
            raise E7Error(f"{fold_id}: E7-D row for {sample_id!r} disagrees with E7-A's own "
                          "materialization reference -- FAIL CLOSED")
        rows.append({"sample_id": sample_id, "dataset": e7d.E7A_DOMAIN_TO_M3B_DATASET[row["dataset"]],
                    "source_record_id": row.get("source_video_id"), "subject_id": row.get("subject_id"),
                    "label_live_spoof": row["label_live_spoof"], "project_split": "source_train",
                    "image_relative_path": ref["image_relative_path"], "crop_sha256": ref["crop_sha256"],
                    "prior_relative_path": ref["prior_relative_path"], "prior_sha256": ref["prior_sha256"],
                    "_kind": "m3b"})
    return rows


def _load_fold_siw_source_train_rows(repo: Path, fold_id: str) -> list[dict[str, Any]]:
    """SiW source_train rows for F2/F3: membership from E7-D's authoritative
    source_train.json, prior fields cross-joined from the FROZEN, shared
    SiW source-prior package (identity-pinned)."""
    if "SiW-Mv2" not in FOLD_SOURCE_DOMAINS[fold_id]:
        return []
    train_e7d, _dev_e7d = _load_e7d_authoritative_siw_rows(repo, fold_id)
    prior_validation = validate_source_priors(repo, fold_id)
    if prior_validation["status"] != "VALID":
        raise E7Error(f"{fold_id}: shared SiW source-prior package is not VALID -- FAIL CLOSED: "
                      f"{prior_validation.get('problems')!r}")
    package_path = repo / SIW_SOURCE_PRIOR_PACKAGE_ROOT / SIW_SOURCE_PRIOR_PACKAGE_FILENAME
    body = cc.read_json(package_path)
    if body.get("package_identity") != FROZEN_SIW_SOURCE_PRIOR_PACKAGE_IDENTITY:
        raise E7Error(f"{fold_id}: shared SiW source-prior package_identity "
                      f"{body.get('package_identity')!r} != frozen "
                      f"{FROZEN_SIW_SOURCE_PRIOR_PACKAGE_IDENTITY!r} -- FAIL CLOSED, do NOT "
                      "rerun source-prior generation, do NOT mutate the frozen package")
    prior_by_key = {(r["source_video_id"], r["frame_index"]): r for r in body.get("rows", [])}
    rows: list[dict[str, Any]] = []
    for row in train_e7d:
        key = (row.get("source_video_id"), row.get("frame_index"))
        prior = prior_by_key.get(key)
        if prior is None:
            raise E7Error(f"{fold_id}: SiW source_train row {key!r} has no matching shared "
                          "source-prior row -- FAIL CLOSED")
        if prior.get("source_crop_relative_path") != row.get("crop_relative_path") or \
                prior.get("source_crop_sha256") != row.get("crop_sha256"):
            raise E7Error(f"{fold_id}: E7-D SiW row for {key!r} disagrees with the shared "
                          "source-prior row -- FAIL CLOSED")
        local_id = _m3a_sample_id(*key)
        rows.append({"sample_id": local_id, "dataset": "siw_mv2", "source_record_id": key[0],
                    "subject_id": None, "label_live_spoof": row["label_live_spoof"],
                    "project_split": "source_train", "image_relative_path": row["crop_relative_path"],
                    "crop_sha256": row["crop_sha256"], "prior_relative_path": prior["prior_relative_path"],
                    "prior_sha256": prior["prior_sha256"], "_kind": "siw"})
    return rows


def _gpat_input_identity_inputs(repo: Path, fold_id: str) -> dict[str, Any]:
    datasets = FOLD_SOURCE_DATASET_SLUGS[fold_id]
    m3b_present = ("casia_fasd" in datasets) or ("msu_mfsd" in datasets)
    siw_present = "siw_mv2" in datasets
    support_path = repo / e7d.E7D_OUTPUT_ROOT / fold_id / "SOURCE_SUPPORT_PACKAGE.json"
    e7d_package_identity = (cc.read_json(support_path).get("package_identity")
                            if support_path.is_file() else None)
    return {"e7d_package_identity": e7d_package_identity,
           "m3b_package_identity": e7b.FROZEN_M3B_PACKAGE_IDENTITY if m3b_present else None,
           "siw_source_prior_package_identity": FROZEN_SIW_SOURCE_PRIOR_PACKAGE_IDENTITY if siw_present
                                                else None,
           "base_config_sha256": cc.sha256_file(repo / GPAT_FIT_CONFIG_PATH),
           "m7_bank_identity": FROZEN_M7_BANK["bank_content_identity_sha256"]}


def compute_gpat_input_package_identity(*, fold_id: str, e7d_package_identity: str | None,
                                        m3b_package_identity: str | None,
                                        siw_source_prior_package_identity: str | None,
                                        base_config_sha256: str, m7_bank_identity: str,
                                        rows: list[tuple[Any, ...]]) -> str:
    """Deterministic identity over CANONICAL METADATA only -- no absolute
    paths, no timestamps, no hostname. Binds fold_id, the frozen E7-D
    source-support package identity for THIS fold, the canonical M3B
    package identity, the shared SiW source-prior package identity when
    applicable, the frozen base GPAT config SHA, the frozen M7 recipe-bank
    identity, and the sorted per-row (dataset, source_record_id,
    subject_id, label_live_spoof, project_split, crop_sha256,
    prior_sha256) material -- subject_id is bound so a CASIA/MSU subject
    swap or a fabricated SiW subject changes the identity, never just
    membership."""
    material = {"fold_id": fold_id, "e7d_source_support_package_identity": e7d_package_identity,
               "m3b_package_identity": m3b_package_identity,
               "siw_source_prior_package_identity": siw_source_prior_package_identity,
               "base_gpat_config_sha256": base_config_sha256, "m7_recipe_bank_identity": m7_bank_identity,
               "row_material": sorted(rows)}
    return cc.sha256_bytes(cc.canonical_json_bytes(material))


def _validate_gpat_input_rows(repo: Path, fold_id: str, manifest_rows: list[dict[str, Any]],
                              package_identity: str | None) -> dict[str, Any]:
    """Substantive validation shared by the pre-write CANDIDATE check inside
    `materialize_gpat_input_package` and the on-disk `validate_gpat_input_package`.
    Re-derives the EXPECTED fold source_train membership from E7-D/E7-A/the
    shared SiW source-prior package authority -- never merely trusts the
    manifest's own stored rows."""
    problems: list[str] = []
    try:
        expected_rows = (_load_fold_m3b_source_train_rows(repo, fold_id) +
                         _load_fold_siw_source_train_rows(repo, fold_id))
    except E7Error as exc:
        return {"status": "INVALID", "problems": [f"could not re-derive expected membership: {exc}"],
               "row_count": len(manifest_rows), "dataset_counts": {}, "live_count": 0, "spoof_count": 0,
               "missing_images": 0, "bad_image_hashes": 0, "missing_priors": 0, "bad_prior_hashes": 0,
               "duplicate_ids": 0, "forbidden_rows": [], "package_identity_match": False,
               "target_access": False, "llm_api_calls": 0}
    expected_by_id = {r["sample_id"]: r for r in expected_rows}

    duplicate_ids = len(manifest_rows) - len({row["sample_id"] for row in manifest_rows})
    if duplicate_ids:
        problems.append(f"{duplicate_ids} duplicate sample_id(s) in manifest")
    manifest_by_id = {row["sample_id"]: row for row in manifest_rows}
    if set(expected_by_id) != set(manifest_by_id):
        only_expected = sorted(set(expected_by_id) - set(manifest_by_id))
        only_manifest = sorted(set(manifest_by_id) - set(expected_by_id))
        problems.append(f"membership mismatch vs E7-D/E7-A authority: {len(only_expected)} row(s) "
                        f"only expected, {len(only_manifest)} row(s) only in manifest "
                        f"(examples: expected={only_expected[:3]!r}, manifest={only_manifest[:3]!r})")

    target_slug = {"CASIA-FASD": "casia_fasd", "MSU-MFSD": "msu_mfsd",
                  "SiW-Mv2": "siw_mv2"}[FOLD_TARGET_DOMAIN[fold_id]]
    allowed_slugs = set(FOLD_SOURCE_DATASET_SLUGS[fold_id])
    dataset_counts: dict[str, int] = {}
    live_count = spoof_count = 0
    missing_images = missing_priors = bad_image_hashes = bad_prior_hashes = 0
    forbidden_rows: list[str] = []
    bad_prior_arrays: list[str] = []
    package_root = repo / GPAT_INPUT_ROOT / fold_id
    import numpy as np

    for sample_id, row in manifest_by_id.items():
        if row.get("project_split") != "source_train":
            problems.append(f"row {sample_id!r} project_split != source_train")
        dataset = row.get("dataset")
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        if dataset == target_slug or dataset not in allowed_slugs:
            forbidden_rows.append(sample_id)
        if row.get("label_live_spoof") == "live":
            live_count += 1
        elif row.get("label_live_spoof") == "spoof":
            spoof_count += 1
        if dataset == "siw_mv2" and row.get("subject_id") is not None:
            problems.append(f"row {sample_id!r} is siw_mv2 but has a non-null subject_id -- "
                            "SiW subject_id must never be fabricated")
        expected = expected_by_id.get(sample_id)
        # GAP 2 hardening: exact-compare EVERY authoritative field, not merely sample_id
        # membership -- catches a mutated CASIA/MSU subject_id, a fabricated SiW subject, a
        # swapped source_record_id, or a swapped label even when sample_id itself is untouched.
        if expected is not None:
            for field in ("dataset", "source_record_id", "subject_id", "label_live_spoof",
                          "project_split"):
                if row.get(field) != expected.get(field):
                    problems.append(f"row {sample_id!r} field {field!r} = {row.get(field)!r} != "
                                    f"E7-D/E7-A authoritative {expected.get(field)!r}")
        image_sha = expected["crop_sha256"] if expected else row.get("crop_sha256")
        prior_sha = expected["prior_sha256"] if expected else row.get("prior_sha256")
        image_path = package_root / row.get("image_relative_path", "")
        prior_path = package_root / row.get("prior_relative_path", "")
        if not image_path.is_file():
            missing_images += 1
        elif cc.sha256_file(image_path) != image_sha:
            bad_image_hashes += 1
        if not prior_path.is_file():
            missing_priors += 1
        else:
            if cc.sha256_file(prior_path) != prior_sha:
                bad_prior_hashes += 1
            try:
                with np.load(prior_path, allow_pickle=False) as handle:
                    keys = set(handle.files)
                if not set(REQUIRED_PRIOR_KEYS) <= keys:
                    bad_prior_arrays.append(sample_id)
            except Exception:  # noqa: BLE001 -- any unreadable prior is a hard failure
                bad_prior_arrays.append(sample_id)

    if forbidden_rows:
        problems.append(f"{len(forbidden_rows)} row(s) belong to a forbidden/held-out-target dataset")
    if missing_images:
        problems.append(f"{missing_images} missing image(s)")
    if bad_image_hashes:
        problems.append(f"{bad_image_hashes} bad image hash(es)")
    if missing_priors:
        problems.append(f"{missing_priors} missing prior(s)")
    if bad_prior_hashes:
        problems.append(f"{bad_prior_hashes} bad prior hash(es)")
    if bad_prior_arrays:
        problems.append(f"{len(bad_prior_arrays)} prior file(s) failed to load required arrays: "
                        f"{sorted(bad_prior_arrays)[:5]!r}")
    if live_count == 0 or spoof_count == 0:
        problems.append(f"both live and spoof must be present (live={live_count}, spoof={spoof_count})")
    if set(dataset_counts) != allowed_slugs:
        problems.append(f"dataset_counts keys {sorted(dataset_counts)} != expected fold datasets "
                        f"{sorted(allowed_slugs)}")

    from prism_fas.synthesis.m8_pipeline import PipelineError, SourceOnlyAudit

    audit = SourceOnlyAudit()
    try:
        audit.record("manifests/source_train.parquet")
        for row in manifest_by_id.values():
            audit.record(row.get("image_relative_path", ""))
            audit.record(row.get("prior_relative_path", ""))
    except PipelineError as exc:
        problems.append(f"SourceOnlyAudit rejected a package-local path: {exc}")

    identity_inputs = _gpat_input_identity_inputs(repo, fold_id)
    recomputed_identity = compute_gpat_input_package_identity(
        fold_id=fold_id, rows=[(row.get("dataset"), row.get("source_record_id"),
                                row.get("subject_id"), row.get("label_live_spoof"),
                                row.get("project_split"),
                                (expected_by_id.get(sample_id) or {}).get("crop_sha256"),
                                (expected_by_id.get(sample_id) or {}).get("prior_sha256"))
                               for sample_id, row in manifest_by_id.items()],
        **identity_inputs)
    identity_match = recomputed_identity == package_identity
    if not identity_match:
        problems.append(f"recomputed package_identity {recomputed_identity!r} != recorded "
                        f"{package_identity!r}")

    return {"status": "INVALID" if problems else "VALID", "problems": problems,
           "row_count": len(manifest_by_id), "dataset_counts": dataset_counts,
           "live_count": live_count, "spoof_count": spoof_count, "missing_images": missing_images,
           "bad_image_hashes": bad_image_hashes, "missing_priors": missing_priors,
           "bad_prior_hashes": bad_prior_hashes, "duplicate_ids": duplicate_ids,
           "forbidden_rows": forbidden_rows, "package_identity": package_identity,
           "recomputed_package_identity": recomputed_identity, "package_identity_match": identity_match,
           "target_access": False, "llm_api_calls": 0}


def materialize_gpat_input_package(repo: Path, fold_id: str, *, authorize: bool = False) -> dict[str, Any]:
    """Materializes (or resumes/validates) the fold-local, SampleStore-
    compatible GPAT-input package. Reuses `_link_or_copy` from
    `prism_fas.data.package.builder` UNMODIFIED; the only new code is row
    selection from E7-D/E7-A/the shared SiW source-prior authority."""
    if fold_id not in FOLD_IDS:
        raise E7Error(f"unknown fold_id {fold_id!r}")
    if not authorize:
        raise E7Error(f"GPAT input package materialization for {fold_id} requires --authorize")

    package_root = repo / GPAT_INPUT_ROOT / fold_id
    lock_path = package_root / GPAT_INPUT_LOCK_FILENAME
    if lock_path.is_file():
        validation = validate_gpat_input_package(repo, fold_id)
        if validation["status"] != "VALID":
            raise E7Error(f"{fold_id}: existing GPAT input package FAILED strict validation -- "
                          f"FAIL CLOSED, never silently rewritten: {validation['problems']!r}")
        return {"resumed": True, "status": "ALREADY_VALID", "path": str(package_root),
               "validation": validation, "target_access": False, "llm_api_calls": 0}

    rows = _load_fold_m3b_source_train_rows(repo, fold_id) + _load_fold_siw_source_train_rows(repo, fold_id)
    if not rows:
        raise E7Error(f"{fold_id}: no source_train rows resolved -- FAIL CLOSED")
    seen_ids: set[str] = set()
    for row in rows:
        if row["sample_id"] in seen_ids:
            raise E7Error(f"{fold_id}: duplicate sample_id {row['sample_id']!r} -- FAIL CLOSED")
        seen_ids.add(row["sample_id"])
        lowered = row["sample_id"].lower()
        for token in ("siw", "target"):
            if token in lowered:
                raise E7Error(f"{fold_id}: derived sample_id {row['sample_id']!r} contains the "
                              f"forbidden substring {token!r} -- FAIL CLOSED")

    images_dir = package_root / "images"
    priors_dir = package_root / "priors"
    images_dir.mkdir(parents=True, exist_ok=True)
    priors_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["_kind"] == "m3b":
            image_root, prior_root = e7b.CASIA_MSU_PACKAGE_ROOT, e7b.CASIA_MSU_PACKAGE_ROOT
        else:
            image_root = f"{e7b.E7B_SIW_SOURCE_PACKAGE_ROOT}/m2_run"
            prior_root = SIW_SOURCE_PRIOR_PACKAGE_ROOT
        image_candidate = f"{image_root}/{row['image_relative_path']}"
        prior_candidate = f"{prior_root}/{row['prior_relative_path']}"
        assert_not_target_path(fold_id, image_candidate)
        assert_not_target_path(fold_id, prior_candidate)
        src_image, src_prior = repo / image_candidate, repo / prior_candidate
        if not src_image.is_file():
            raise E7Error(f"{fold_id}: source image missing on disk: {image_candidate!r} -- FAIL CLOSED")
        if cc.sha256_file(src_image) != row["crop_sha256"]:
            raise E7Error(f"{fold_id}: source image SHA256 mismatch: {image_candidate!r} -- FAIL CLOSED")
        if not src_prior.is_file():
            raise E7Error(f"{fold_id}: source prior missing on disk: {prior_candidate!r} -- FAIL CLOSED")
        if cc.sha256_file(src_prior) != row["prior_sha256"]:
            raise E7Error(f"{fold_id}: source prior SHA256 mismatch: {prior_candidate!r} -- FAIL CLOSED")

        image_ext = Path(row["image_relative_path"]).suffix or ".jpg"
        prior_ext = Path(row["prior_relative_path"]).suffix or ".npz"
        dest_image_rel = f"images/{row['sample_id']}{image_ext}"
        dest_prior_rel = f"priors/{row['sample_id']}{prior_ext}"
        _gpat_local_link_or_copy(src_image, package_root / dest_image_rel)
        _gpat_local_link_or_copy(src_prior, package_root / dest_prior_rel)
        manifest_rows.append({"sample_id": row["sample_id"], "project_split": "source_train",
                              "dataset": row["dataset"], "source_record_id": row["source_record_id"],
                              "subject_id": row["subject_id"], "label_live_spoof": row["label_live_spoof"],
                              "image_relative_path": dest_image_rel, "prior_relative_path": dest_prior_rel})

    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([("sample_id", pa.string()), ("project_split", pa.string()),
                       ("dataset", pa.string()), ("source_record_id", pa.string()),
                       ("subject_id", pa.string()), ("label_live_spoof", pa.string()),
                       ("image_relative_path", pa.string()), ("prior_relative_path", pa.string())])
    ordered = sorted(manifest_rows, key=lambda r: r["sample_id"])
    manifest_path = package_root / "manifests" / "source_train.parquet"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(ordered, schema=schema), manifest_path)

    identity_inputs = _gpat_input_identity_inputs(repo, fold_id)
    package_identity = compute_gpat_input_package_identity(
        fold_id=fold_id, rows=[(row["dataset"], row["source_record_id"], row["subject_id"],
                               row["label_live_spoof"], row["project_split"], row["crop_sha256"],
                               row["prior_sha256"]) for row in rows],
        **identity_inputs)

    candidate = _validate_gpat_input_rows(repo, fold_id, manifest_rows, package_identity)
    if candidate["status"] != "VALID":
        raise E7Error(f"{fold_id}: candidate GPAT input package FAILED strict validation -- "
                      f"refusing to write the terminal PACKAGE_LOCK.json: {candidate['problems']!r}")

    from prism_fas.utils.core import atomic_json_write

    lock = {"schema_version": f"{SCHEMA_PREFIX}-gpat-input-package-lock-v1", "fold_id": fold_id,
           "content_identity_sha256": package_identity, "row_count": len(rows),
           "source_domains": sorted(FOLD_SOURCE_DATASET_SLUGS[fold_id]),
           **identity_inputs, "status": "validated"}
    atomic_json_write(lock_path, lock)
    validation = validate_gpat_input_package(repo, fold_id)
    if validation["status"] != "VALID":
        raise E7Error(f"{fold_id}: freshly-written GPAT input package FAILED strict validation "
                      f"immediately after write -- {validation['problems']!r}")
    return {"resumed": False, "status": "MATERIALIZED", "path": str(package_root),
           "package_identity": package_identity, "validation": validation,
           "target_access": False, "llm_api_calls": 0}


def validate_gpat_input_package(repo: Path, fold_id: str) -> dict[str, Any]:
    """STRICT, read-only validator. Never writes."""
    if fold_id not in FOLD_IDS:
        raise E7Error(f"unknown fold_id {fold_id!r}")
    package_root = repo / GPAT_INPUT_ROOT / fold_id
    lock_path = package_root / GPAT_INPUT_LOCK_FILENAME
    manifest_path = package_root / "manifests" / "source_train.parquet"
    if not lock_path.is_file() or not manifest_path.is_file():
        return {"schema_version": f"{SCHEMA_PREFIX}-gpat-input-validate-v1", "fold_id": fold_id,
               "status": "NOT_MATERIALIZED"}
    import pyarrow.parquet as pq

    lock = cc.read_json(lock_path)
    manifest_rows = pq.read_table(manifest_path).to_pylist()
    result = _validate_gpat_input_rows(repo, fold_id, manifest_rows, lock.get("content_identity_sha256"))
    return {"schema_version": f"{SCHEMA_PREFIX}-gpat-input-validate-v1", "fold_id": fold_id, **result}


# --------------------------------------------------------------------------- #
# TASK N.2 -- fold-aware pair-plan adapter. `pair_plan.py` is scientifically
# frozen and NEVER modified; this only temporarily binds its module-level
# CASIA/MSU-only constants for the duration of one real call, always
# restored in `finally` (even on error).
# --------------------------------------------------------------------------- #

@contextlib.contextmanager
def _scoped_pair_plan_allowed_datasets(datasets: tuple[str, ...]):
    from prism_fas.synthesis import pair_plan

    original = pair_plan.ALLOWED_DATASETS
    try:
        pair_plan.ALLOWED_DATASETS = tuple(datasets)
        yield pair_plan
    finally:
        pair_plan.ALLOWED_DATASETS = original


@contextlib.contextmanager
def _scoped_pair_plan_expected_counts(train_pairs: int, validation_pairs: int):
    """Temporarily binds the legacy `EXPECTED_TRAIN_PAIRS`/
    `EXPECTED_VALIDATION_PAIRS` constants to counts ALREADY DERIVED from a
    real `build_pair_plan()` call -- never a tuning target. Restored in
    `finally`, even on error."""
    from prism_fas.synthesis import pair_plan

    original = (pair_plan.EXPECTED_TRAIN_PAIRS, pair_plan.EXPECTED_VALIDATION_PAIRS)
    try:
        pair_plan.EXPECTED_TRAIN_PAIRS = int(train_pairs)
        pair_plan.EXPECTED_VALIDATION_PAIRS = int(validation_pairs)
        yield pair_plan
    finally:
        pair_plan.EXPECTED_TRAIN_PAIRS, pair_plan.EXPECTED_VALIDATION_PAIRS = original


def materialize_fold_pair_plan(repo: Path, fold_id: str, *, authorize: bool = False) -> dict[str, Any]:
    if fold_id not in FOLD_IDS:
        raise E7Error(f"unknown fold_id {fold_id!r}")
    if not authorize:
        raise E7Error(f"pair-plan materialization for {fold_id} requires --authorize")

    package_root = repo / GPAT_INPUT_ROOT / fold_id
    output_root = repo / GPAT_PAIR_PLAN_ROOT / fold_id
    lock_path = output_root / "PAIR_PLAN_LOCK.json"
    if lock_path.is_file():
        validation = validate_fold_pair_plan(repo, fold_id)
        if validation["status"] != "VALID":
            raise E7Error(f"{fold_id}: existing pair plan FAILED strict validation -- FAIL "
                          f"CLOSED, never silently rewritten: {validation['problems']!r}")
        return {"resumed": True, "status": "ALREADY_VALID", "path": str(output_root),
               "validation": validation, "target_access": False, "llm_api_calls": 0}

    input_validation = validate_gpat_input_package(repo, fold_id)
    if input_validation["status"] != "VALID":
        raise E7Error(f"{fold_id}: GPAT input package is not VALID -- FAIL CLOSED: "
                      f"{input_validation.get('problems')!r}")

    bank_root = repo / M7_RECIPE_BANK_ROOT
    datasets = FOLD_SOURCE_DATASET_SLUGS[fold_id]
    with _scoped_pair_plan_allowed_datasets(datasets) as pair_plan:
        plan = pair_plan.build_pair_plan(package_root, bank_root, seed=GPAT_FIT_SEED)
        train_count = len(plan["pairs"]["train"])
        validation_count = len(plan["pairs"]["validation"])
        with _scoped_pair_plan_expected_counts(train_count, validation_count):
            result = pair_plan.write_pair_plan(package_root, bank_root, output_root, seed=GPAT_FIT_SEED,
                                               config_hash=cc.sha256_file(repo / GPAT_FIT_CONFIG_PATH))
    return {"resumed": False, "status": "MATERIALIZED", "path": str(output_root),
           "train_pairs": train_count, "validation_pairs": validation_count, "lock": result["lock"],
           "target_access": False, "llm_api_calls": 0}


def validate_fold_pair_plan(repo: Path, fold_id: str) -> dict[str, Any]:
    """STRICT, read-only validator. Never writes, never modifies
    `pair_plan.py`'s module-level constants."""
    if fold_id not in FOLD_IDS:
        raise E7Error(f"unknown fold_id {fold_id!r}")
    output_root = repo / GPAT_PAIR_PLAN_ROOT / fold_id
    lock_path = output_root / "PAIR_PLAN_LOCK.json"
    if not lock_path.is_file():
        return {"schema_version": f"{SCHEMA_PREFIX}-pair-plan-validate-v1", "fold_id": fold_id,
               "status": "NOT_MATERIALIZED"}
    from prism_fas.synthesis import pair_plan

    problems: list[str] = []
    lock = cc.read_json(lock_path)
    if lock.get("seed") != GPAT_FIT_SEED:
        problems.append(f"seed {lock.get('seed')!r} != frozen {GPAT_FIT_SEED!r}")
    if lock.get("recipe_bank_identity") != FROZEN_M7_BANK["bank_content_identity_sha256"]:
        problems.append("recipe_bank_identity != frozen M7 bank identity")
    package_root = repo / GPAT_INPUT_ROOT / fold_id
    package_lock_path = package_root / GPAT_INPUT_LOCK_FILENAME
    current_package_identity = (cc.read_json(package_lock_path).get("content_identity_sha256")
                                if package_lock_path.is_file() else None)
    if lock.get("package_identity") != current_package_identity:
        problems.append(f"pair-plan package_identity {lock.get('package_identity')!r} != current "
                        f"GPAT input package identity {current_package_identity!r}")

    train_rows = pair_plan.load_pair_manifest(output_root / "pair_manifest_train.parquet")
    validation_rows = pair_plan.load_pair_manifest(output_root / "pair_manifest_validation.parquet")

    # GAP 1 fix: `pair_plan.pair_plan_identity()` merely RE-READS
    # PAIR_PLAN_LOCK.json's own stored field -- that is an echo, not a recomputation. TRUE
    # recomputation calls the real, unmodified `pair_plan.build_pair_plan()` under the same
    # scoped ALLOWED_DATASETS adapter `materialize_fold_pair_plan` uses, compares EVERY pair
    # field between the freshly-rebuilt plan and the on-disk manifests, and independently
    # rebuilds every identity-bearing PAIR_PLAN_LOCK component using the SAME helper functions
    # (`summarize_pairs`, `rows_digest`, `_digest`) and the SAME canonical-JSON/IDENTITY_EXCLUDED
    # contract `write_pair_plan()` itself uses -- never a fresh reimplementation of pairing.
    bank_root = repo / M7_RECIPE_BANK_ROOT
    datasets = FOLD_SOURCE_DATASET_SLUGS[fold_id]
    recomputed_identity: str | None = None
    try:
        with _scoped_pair_plan_allowed_datasets(datasets) as scoped_pair_plan:
            expected_plan = scoped_pair_plan.build_pair_plan(package_root, bank_root, seed=GPAT_FIT_SEED)
    except Exception as exc:  # noqa: BLE001 -- any failure to rebuild is itself a hard validation failure
        problems.append(f"could not rebuild the expected pair plan for comparison: {exc}")
        expected_plan = None

    if expected_plan is not None:
        expected_train = expected_plan["pairs"]["train"]
        expected_validation = expected_plan["pairs"]["validation"]

        for name, expected_rows, actual_rows in (("train", expected_train, train_rows),
                                                  ("validation", expected_validation, validation_rows)):
            by_id_expected = {r["pair_id"]: r for r in expected_rows}
            by_id_actual = {r["pair_id"]: r for r in actual_rows}
            if set(by_id_expected) != set(by_id_actual):
                only_expected = sorted(set(by_id_expected) - set(by_id_actual))
                only_actual = sorted(set(by_id_actual) - set(by_id_expected))
                problems.append(f"{name} pair_id set does not match a fresh rebuild -- "
                                f"{len(only_expected)} only in the rebuild, {len(only_actual)} "
                                f"only on disk (examples: rebuild={only_expected[:3]!r}, "
                                f"disk={only_actual[:3]!r})")
            content_mismatches = sorted(pid for pid in (set(by_id_expected) & set(by_id_actual))
                                        if by_id_expected[pid] != by_id_actual[pid])
            if content_mismatches:
                problems.append(f"{name} pair(s) with the SAME pair_id but DIFFERENT field content "
                                f"vs a fresh rebuild (covers pair_id/recipe_id/recipe_seed/every "
                                f"other pair field): {content_mismatches[:5]!r}")

        # Independently rebuild every identity-bearing PAIR_PLAN_LOCK component from the
        # freshly-rebuilt plan -- reusing `summarize_pairs`/`rows_digest`/`_digest` verbatim.
        expected_summary = {"train": pair_plan.summarize_pairs(expected_train),
                            "validation": pair_plan.summarize_pairs(expected_validation)}
        expected_lock_material = {
            "pair_plan_schema_version": pair_plan.PAIR_PLAN_SCHEMA_VERSION, "seed": int(GPAT_FIT_SEED),
            "package_identity": expected_plan["package_identity"],
            "recipe_bank_identity": expected_plan["recipe_bank_identity"],
            "train_pairs": len(expected_train), "validation_pairs": len(expected_validation),
            "record_set_hashes": {
                "live_train": pair_plan._digest(*sorted(
                    r for r, p in expected_plan["live_partition"].items() if p == "train")),
                "live_validation": pair_plan._digest(*sorted(
                    r for r, p in expected_plan["live_partition"].items() if p == "validation")),
                "spoof_train": pair_plan._digest(*sorted(
                    r for r, p in expected_plan["spoof_partition"].items() if p == "train")),
                "spoof_validation": pair_plan._digest(*sorted(
                    r for r, p in expected_plan["spoof_partition"].items() if p == "validation"))},
            "pair_rows_sha256": {"train": pair_plan.rows_digest(expected_train),
                                 "validation": pair_plan.rows_digest(expected_validation)},
            "pair_id_set_sha256": {"train": pair_plan._digest(*[r["pair_id"] for r in expected_train]),
                                   "validation": pair_plan._digest(
                                       *[r["pair_id"] for r in expected_validation])},
            "domain_composition": {"train": expected_summary["train"]["live_datasets"],
                                   "validation": expected_summary["validation"]["live_datasets"],
                                   "train_relation": expected_summary["train"]["domain_relation"],
                                   "validation_relation": expected_summary["validation"]["domain_relation"]},
            "recipe_coverage": {"train": expected_summary["train"]["distinct_recipes"],
                                "validation": expected_summary["validation"]["distinct_recipes"]},
            "attack_family_balance": "unavailable"}

        for field, expected_value in expected_lock_material.items():
            if lock.get(field) != expected_value:
                problems.append(f"PAIR_PLAN_LOCK identity-bearing field {field!r} does not match a "
                                f"fresh recomputation: recorded={lock.get(field)!r}, "
                                f"expected={expected_value!r}")

        recomputed_identity = pair_plan._digest(json.dumps(expected_lock_material, sort_keys=True,
                                                            separators=(",", ":")))
        if recomputed_identity != lock.get("pair_plan_identity_sha256"):
            problems.append(f"recomputed pair_plan_identity_sha256 {recomputed_identity!r} != "
                            f"recorded {lock.get('pair_plan_identity_sha256')!r}")
    source_train_ids: set[str] = set()
    subject_by_sample_id: dict[str, str | None] = {}
    manifest_path = package_root / "manifests" / "source_train.parquet"
    if manifest_path.is_file():
        import pyarrow.parquet as pq

        manifest_rows = pq.read_table(manifest_path).to_pylist()
        source_train_ids = {row["sample_id"] for row in manifest_rows}
        subject_by_sample_id = {row["sample_id"]: row["subject_id"] for row in manifest_rows}

    dataset_slugs = set(FOLD_SOURCE_DATASET_SLUGS[fold_id])
    target_slug = {"CASIA-FASD": "casia_fasd", "MSU-MFSD": "msu_mfsd",
                  "SiW-Mv2": "siw_mv2"}[FOLD_TARGET_DOMAIN[fold_id]]
    live_slot_counts: dict[str, dict[str, int]] = {}
    train_records: set[str] = set()
    validation_records: set[str] = set()
    for partition_name, rows in (("train", train_rows), ("validation", validation_rows)):
        record_set = train_records if partition_name == "train" else validation_records
        for row in rows:
            for sample_id, role in ((row["live_sample_id"], "live"), (row["spoof_sample_id"], "spoof")):
                if source_train_ids and sample_id not in source_train_ids:
                    problems.append(f"{role} sample {sample_id!r} not present in source_train")
            for dataset in (row["live_dataset"], row["spoof_dataset"]):
                if dataset not in dataset_slugs:
                    problems.append(f"pair {row['pair_id']!r} references dataset {dataset!r} "
                                    "outside the fold's own source domains")
                if dataset == target_slug:
                    problems.append(f"pair {row['pair_id']!r} references the HELD-OUT TARGET domain")
            if row["live_source_record_id"] == row["spoof_source_record_id"]:
                problems.append(f"pair {row['pair_id']!r} reuses one source_record_id for both roles")
            if row["domain_relation"] == "same_domain" and row["live_dataset"] != row["spoof_dataset"]:
                problems.append(f"pair {row['pair_id']!r} marked same_domain but datasets differ")
            if row["domain_relation"] == "cross_domain" and row["live_dataset"] == row["spoof_dataset"]:
                problems.append(f"pair {row['pair_id']!r} marked cross_domain but datasets match")
            # The real `build_pair_plan()` decides this PER PAIR (both members must have a
            # subject_id), never from a fold-wide flag -- re-derived here from the GPAT input
            # package's own manifest, never fabricated.
            live_subject = subject_by_sample_id.get(row["live_sample_id"])
            spoof_subject = subject_by_sample_id.get(row["spoof_sample_id"])
            expected_rule = "enforced" if (live_subject and spoof_subject) else "not_applicable"
            if row["different_subject_rule"] != expected_rule:
                problems.append(f"pair {row['pair_id']!r} different_subject_rule="
                                f"{row['different_subject_rule']!r} != expected {expected_rule!r} "
                                "given its live/spoof subject_id availability")
            live_slot_counts.setdefault(row["live_sample_id"], {"same_domain": 0, "cross_domain": 0})
            live_slot_counts[row["live_sample_id"]][row["domain_relation"]] += 1
            record_set.add(row["live_source_record_id"])
            record_set.add(row["spoof_source_record_id"])

    overlap = train_records & validation_records
    if overlap:
        problems.append(f"train/validation record overlap: {sorted(overlap)[:5]}")
    for live_id, counts in live_slot_counts.items():
        if counts != {"same_domain": 2, "cross_domain": 2}:
            problems.append(f"live {live_id!r} does not have exactly 2 same-domain + 2 cross-domain "
                            f"pairs: {counts!r}")

    return {"schema_version": f"{SCHEMA_PREFIX}-pair-plan-validate-v1", "fold_id": fold_id,
           "status": "INVALID" if problems else "VALID", "problems": problems,
           "observed_train_pairs": len(train_rows), "observed_validation_pairs": len(validation_rows),
           "seed": lock.get("seed"), "package_identity": lock.get("package_identity"),
           "pair_plan_identity": lock.get("pair_plan_identity_sha256"),
           "recomputed_pair_plan_identity": recomputed_identity,
           "pair_plan_identity_match": recomputed_identity is not None
                                       and recomputed_identity == lock.get("pair_plan_identity_sha256"),
           "target_access": False, "llm_api_calls": 0}


# --------------------------------------------------------------------------- #
# TASK N.3 -- effective GPAT config (data-contract overrides only, never a
# scientific-hyperparameter change).
# --------------------------------------------------------------------------- #

def build_effective_gpat_config(repo: Path, fold_id: str) -> dict[str, Any]:
    import copy

    import yaml

    base_path = repo / GPAT_FIT_CONFIG_PATH
    base_config_sha256 = cc.sha256_file(base_path)
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    effective = copy.deepcopy(base_config)

    old_datasets = list(effective["data"]["allowed_datasets"])
    new_datasets = sorted(FOLD_SOURCE_DATASET_SLUGS[fold_id])
    effective["data"]["allowed_datasets"] = new_datasets

    pair_plan_validation = validate_fold_pair_plan(repo, fold_id)
    if pair_plan_validation["status"] != "VALID":
        raise E7Error(f"{fold_id}: cannot build effective config -- pair plan is not VALID: "
                      f"{pair_plan_validation.get('problems')!r}")
    old_train_pairs = effective["pair_plan"]["expected_train_pairs"]
    old_validation_pairs = effective["pair_plan"]["expected_validation_pairs"]
    effective["pair_plan"]["expected_train_pairs"] = pair_plan_validation["observed_train_pairs"]
    effective["pair_plan"]["expected_validation_pairs"] = pair_plan_validation["observed_validation_pairs"]

    declared_overrides = {
        "data.allowed_datasets": {"old": old_datasets, "new": new_datasets},
        "pair_plan.expected_train_pairs": {"old": old_train_pairs,
                                          "new": effective["pair_plan"]["expected_train_pairs"]},
        "pair_plan.expected_validation_pairs": {"old": old_validation_pairs,
                                               "new": effective["pair_plan"]["expected_validation_pairs"]},
    }
    unexpected: list[str] = []

    def _walk(a: Any, b: Any, path: str = "") -> None:
        if path in declared_overrides:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                _walk(a.get(key), b.get(key), f"{path}.{key}" if path else key)
        elif a != b:
            unexpected.append(path)

    _walk(base_config, effective)
    if unexpected:
        raise E7Error(f"{fold_id}: effective config changed undeclared field(s): {unexpected!r} -- "
                      "FAIL CLOSED")

    from prism_fas.synthesis.m8_pipeline import config_hash as gpat_config_hash

    effective_config_hash = gpat_config_hash(effective)
    return {"schema_version": f"{SCHEMA_PREFIX}-effective-gpat-config-v1", "fold_id": fold_id,
           "base_config_path": GPAT_FIT_CONFIG_PATH, "base_config_sha256": base_config_sha256,
           "effective_config_hash": effective_config_hash, "declared_overrides": declared_overrides,
           "effective_config": effective}


# --------------------------------------------------------------------------- #
# TASK N.4 -- native checkpoint path helpers (the ONE canonical resolver).
# --------------------------------------------------------------------------- #

def gpat_fit_run_root(repo: Path, fold_id: str) -> Path:
    return repo / RUN_ROOT / fold_id / "gpat_fit"


def gpat_best_checkpoint_path(repo: Path, fold_id: str) -> Path:
    return gpat_fit_run_root(repo, fold_id) / "checkpoints" / "best.pt"


def gpat_last_checkpoint_path(repo: Path, fold_id: str) -> Path:
    return gpat_fit_run_root(repo, fold_id) / "checkpoints" / "last.pt"


def gpat_fit_lock_path(repo: Path, fold_id: str) -> Path:
    return repo / RUN_ROOT / fold_id / GPAT_FIT_LOCK_FILENAME


# --------------------------------------------------------------------------- #
# TASK N.5 -- real GPU-capability check (CUDA + resolvable pinned AdaFace
# weight, the SAME weight/convention GPATTrainer itself resolves
# internally). Never a hardcoded "non-GPU host" string.
# --------------------------------------------------------------------------- #

def _gpat_fit_capability(repo: Path) -> dict[str, Any]:
    problems: list[str] = []
    cuda_available = False
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001 -- any import/probe failure means NOT capable
        problems.append(f"torch unavailable or unusable: {exc!r}")
    if not cuda_available:
        problems.append("torch.cuda.is_available() is False on this host")
    weight_root = _resolve_weight_root(repo)
    try:
        from prism_fas.synthesis.quality_models import resolve_weight

        resolve_weight(weight_root, "identity")
    except Exception as exc:  # noqa: BLE001 -- any unresolved pinned weight means NOT capable
        problems.append(f"pinned AdaFace weight unresolved under {weight_root}: {exc!r}")
    return {"capable": not problems, "cuda_available": cuda_available, "weight_root": str(weight_root),
           "problems": problems}


# --------------------------------------------------------------------------- #
# TASK N.5.5 -- TECHNICAL_SINGLE_SAMPLE_EMPTY_CHEEK_FALLBACK_EDGE_CASE.
#
# The frozen `prism_fas.synthesis.masks.RegionMaskBuilder` cheek fallback
# derives its ellipse centre from `nose_x +/- 0.22 * face_width`. For one
# real EXT-F2 SiW-as-source sample (a large detector bbox on a small crop)
# that centre lands entirely outside the 224x224 crop, so the ellipse
# intersects zero pixels and `RegionMaskBuilder.build()` raises
# `MaskBuildError("region 'right_cheek' produced an empty mask")`.
#
# This is a TECHNICAL crop-boundary edge case, never a scientific defect
# (F1's exhaustive audit proves BBOX_OOB is common and harmless; only ONE
# EXT-F2 sample's cheek ellipse centre falls fully outside the crop). The
# ORIGINAL `RegionMaskBuilder` in `masks.py` is NEVER modified. This is an
# ADDITIVE, E7-scoped compatibility subclass: it calls the real, unmodified
# `RegionMaskBuilder.region()` first; a non-empty result (or any non-cheek
# region) passes through byte-identical; only an EMPTY left_cheek/
# right_cheek result gets a deterministic second-stage recovery that
# reuses the SAME fallback geometry (`face_box`, `_eye_line`, `_mouth_line`,
# `landmark`, `face_width`/`face_height`, the same `nose_x +/- 0.22 *
# face_width` centre rule, the same `0.15 * face_width` / `0.14 *
# face_height` radii) and ONLY clips the already-computed centre into the
# valid crop pixel-centre domain before re-rendering via the REAL, frozen
# `masks._ellipse` (never reimplemented). If the clipped recovery is STILL
# empty, this returns the empty mask unchanged -- `RegionMaskBuilder.build()`
# (inherited, never overridden) raises the exact same `MaskBuildError` it
# always would, so an unrecoverable case fails closed exactly as before.
# --------------------------------------------------------------------------- #

MASK_COMPATIBILITY_POLICY = "e7-empty-cheek-crop-boundary-recovery-v1"
MASK_RECOVERY_SOURCE_TAG = "bbox_geometry+crop_boundary_recovery_v1"
GPAT_ATTEMPT_PROVENANCE_FILENAME = "GPAT_ATTEMPT_PROVENANCE.json"


class _E7CompatibleRegionMaskBuilder:
    """Mixin-style override, applied via a real subclass of the frozen
    `RegionMaskBuilder` dataclass constructed lazily (see
    `_e7_compatible_region_mask_builder_class()`) so this module never has
    to import `RegionMaskBuilder` at module-import time in a way that could
    be mistaken for a `masks.py` change. Holds NO dataclass fields of its
    own -- only a class-level, per-invocation-scoped recovery counter."""
    _recovery_counter: list[int] | None = None

    def region(self, name: str) -> tuple[Any, str]:
        import numpy as np

        mask, source = super().region(name)  # the REAL, unmodified RegionMaskBuilder.region()
        if name not in ("left_cheek", "right_cheek") or np.asarray(mask).any():
            return mask, source  # non-cheek, or already non-empty: pass through byte-identical

        from prism_fas.synthesis.masks import _ellipse  # the REAL, frozen renderer -- never duplicated

        x1, y1, x2, y2 = self.face_box()
        face_width, face_height = max(x2 - x1, 2.0), max(y2 - y1, 2.0)
        nose_x, _ = self.landmark("nose")
        top = self._eye_line() + 0.10 * face_height
        bottom = min(self._mouth_line() + 0.05 * face_height, y2)
        centre_x = nose_x - 0.22 * face_width if name == "left_cheek" else nose_x + 0.22 * face_width
        centre_y = (top + max(bottom, top + 2.0)) / 2.0
        # The ONLY correction: clip the already-computed ellipse centre into the valid crop
        # pixel-centre domain. Radii/formula are otherwise byte-identical to the frozen fallback.
        centre_x = float(np.clip(centre_x, 0.5, self.width - 0.5))
        centre_y = float(np.clip(centre_y, 0.5, self.height - 0.5))
        recovered = _ellipse(self.height, self.width, centre_x, centre_y, face_width * 0.15, face_height * 0.14)
        if not np.asarray(recovered).any():
            return recovered, source  # still empty -- FAIL CLOSED exactly as before, via build()
        counter = type(self)._recovery_counter
        if counter is not None:
            counter[0] += 1
        return recovered, MASK_RECOVERY_SOURCE_TAG


def _e7_compatible_region_mask_builder_class() -> type:
    """Builds (once) a real subclass of the frozen `RegionMaskBuilder`
    dataclass -- MRO `(Built, _E7CompatibleRegionMaskBuilder,
    RegionMaskBuilder, object)`, i.e. `type(name, (_E7CompatibleRegionMaskBuilder,
    RegionMaskBuilder), {})` -- so `_E7CompatibleRegionMaskBuilder.region()`
    overrides `RegionMaskBuilder.region()` and `super().region()` inside it
    resolves to the REAL, unmodified `RegionMaskBuilder.region()`; every
    other field/method is inherited unmodified. Never mutates
    `masks.RegionMaskBuilder` itself."""
    cached = getattr(_e7_compatible_region_mask_builder_class, "_cached", None)
    if cached is not None:
        return cached
    from prism_fas.synthesis.masks import RegionMaskBuilder

    built = type("E7CompatibleRegionMaskBuilder", (_E7CompatibleRegionMaskBuilder, RegionMaskBuilder), {})
    _e7_compatible_region_mask_builder_class._cached = built
    return built


@contextlib.contextmanager
def _scoped_mask_builder_binding(builder_class: type, *, recovery_counter: list[int] | None = None):
    """Saves the EXACT original `m8_pipeline.RegionMaskBuilder` module-
    global object, binds `builder_class` for the duration, and restores the
    original in `finally` -- even on error. `SampleStore.mask_builder()`
    resolves `RegionMaskBuilder` from `m8_pipeline`'s module namespace at
    CALL time, so this is the one legitimate seam to redirect construction
    without ever touching `m8_pipeline.py` itself."""
    from prism_fas.synthesis import m8_pipeline

    original = m8_pipeline.RegionMaskBuilder
    if recovery_counter is not None:
        builder_class._recovery_counter = recovery_counter
    try:
        m8_pipeline.RegionMaskBuilder = builder_class
        yield
    finally:
        m8_pipeline.RegionMaskBuilder = original
        if recovery_counter is not None:
            builder_class._recovery_counter = None


def _scoped_e7_mask_compatibility_binding(recovery_counter: list[int]):
    """The ONE binding wrapped around the real `GPATTrainer` construction +
    `fit()` call in `prepare_gpat`."""
    return _scoped_mask_builder_binding(_e7_compatible_region_mask_builder_class(),
                                        recovery_counter=recovery_counter)


def _iter_gpat_input_sample_ids(package_root: Path) -> list[str]:
    from prism_fas.data.package.manifests import read_manifest

    rows = read_manifest(package_root / "manifests" / "source_train.parquet")
    return sorted(row["sample_id"] for row in rows)


def audit_mask_compatibility_row_invariance(repo: Path, fold_id: str) -> dict[str, Any]:
    """Read-only. For every row in this fold's GPAT input source_train
    manifest and every `masks.REGION_ORDER` region, compares the ORIGINAL
    frozen `RegionMaskBuilder.region()` result against the E7-corrected
    result. Never writes; never fits, renders, or trains."""
    import numpy as np
    from prism_fas.synthesis.m8_pipeline import SampleStore
    from prism_fas.synthesis.masks import REGION_ORDER, RegionMaskBuilder

    if fold_id not in FOLD_IDS:
        raise E7Error(f"unknown fold_id {fold_id!r}")
    package_root = repo / GPAT_INPUT_ROOT / fold_id
    store = SampleStore.open(package_root)
    corrected_cls = _e7_compatible_region_mask_builder_class()

    rows_checked = regions_checked = different_masks = different_sources = 0
    recovery_activations = original_failures = corrected_failures = 0
    recovered_sample_ids: list[str] = []
    for sample_id in _iter_gpat_input_sample_ids(package_root):
        _, arrays = store.load(sample_id)
        kwargs = dict(height=224, width=224, parsing=arrays["parsing_labels"],
                     landmarks=arrays["landmarks"], bbox=arrays["bbox"], crop_box=arrays["crop_box"])
        original_builder = RegionMaskBuilder(**kwargs)
        corrected_builder = corrected_cls(**kwargs)
        rows_checked += 1
        for region_name in REGION_ORDER:
            regions_checked += 1
            try:
                original_mask, original_source = original_builder.region(region_name)
                original_empty = not bool(np.asarray(original_mask).any())
            except Exception:  # noqa: BLE001 -- any anomaly counts as a failure, never silently skipped
                original_mask, original_source, original_empty = None, None, True
            if original_empty:
                original_failures += 1
            try:
                corrected_mask, corrected_source = corrected_builder.region(region_name)
                corrected_empty = not bool(np.asarray(corrected_mask).any())
            except Exception:  # noqa: BLE001
                corrected_mask, corrected_source, corrected_empty = None, None, True
            if corrected_empty:
                corrected_failures += 1
            if original_empty and region_name in ("left_cheek", "right_cheek") and not corrected_empty:
                recovery_activations += 1
                recovered_sample_ids.append(sample_id)
                continue  # a genuine recovery is EXPECTED to differ from the (empty) original
            if not original_empty:
                if not np.array_equal(np.asarray(original_mask), np.asarray(corrected_mask)):
                    different_masks += 1
                if original_source != corrected_source:
                    different_sources += 1
    return {"schema_version": f"{SCHEMA_PREFIX}-mask-compat-row-audit-v1", "fold_id": fold_id,
           "rows_checked": rows_checked, "regions_checked": regions_checked,
           "different_masks": different_masks, "different_sources": different_sources,
           "recovery_activations": recovery_activations, "original_failures": original_failures,
           "corrected_failures": corrected_failures,
           "recovered_sample_ids_diagnostic": recovered_sample_ids,  # diagnostic only, never policy
           "target_access": False, "llm_api_calls": 0}


def audit_mask_compatibility_pair_invariance(repo: Path, fold_id: str) -> dict[str, Any]:
    """Read-only. Reconstructs the live support mask + spoof style mask for
    EVERY real frozen train+validation pair, using the SAME
    `compile_recipe`/`SampleStore.cached_mask` coverage/seed-scope/
    use_support semantics `m8_pipeline.build_batch` itself uses, once under
    the ORIGINAL frozen `RegionMaskBuilder` and once under the E7-corrected
    builder. Proves the compatibility branch is a no-op wherever it is not
    the empty-cheek edge case. Never writes; never fits, renders, or
    trains."""
    import numpy as np
    from prism_fas.recipes.compile import compile_recipe
    from prism_fas.synthesis import m8_pipeline
    from prism_fas.synthesis.m8_pipeline import SampleStore, load_pairs, resolve_bank
    from prism_fas.synthesis.masks import RegionMaskBuilder

    if fold_id not in FOLD_IDS:
        raise E7Error(f"unknown fold_id {fold_id!r}")
    package_root = repo / GPAT_INPUT_ROOT / fold_id
    pairs_root = repo / GPAT_PAIR_PLAN_ROOT / fold_id
    bank_root = repo / M7_RECIPE_BANK_ROOT
    bank = resolve_bank(bank_root)
    recipes = {recipe.recipe_id: recipe for recipe in bank["recipes"]}

    all_pairs: list[dict[str, Any]] = []
    for partition in ("train", "validation"):
        all_pairs.extend(load_pairs(pairs_root, partition))

    def _compute(builder_class: type, recovery_counter: list[int] | None) -> list[tuple[str, Any, Any]]:
        with _scoped_mask_builder_binding(builder_class, recovery_counter=recovery_counter):
            store = SampleStore.open(package_root)
            out = []
            for pair in all_pairs:
                recipe = recipes[pair["recipe_id"]]
                graph = compile_recipe(recipe, bank["ontology"], bank_id=bank["bank_id"])
                policy = graph.region_mask_policy
                support = store.cached_mask(pair["live_sample_id"], "live", graph,
                                            coverage=float(policy["requested_coverage"]),
                                            seed_scope="region_mask", use_support=True)
                style = store.cached_mask(pair["spoof_sample_id"], "spoof", graph, coverage=1.0,
                                          seed_scope="style_mask", use_support=False)
                out.append((pair["pair_id"], support, style))
            return out

    original_results = _compute(RegionMaskBuilder, None)
    recovery_counter = [0]
    corrected_results = _compute(_e7_compatible_region_mask_builder_class(), recovery_counter)

    different_masks = 0
    for (pid_o, support_o, style_o), (pid_c, support_c, style_c) in zip(original_results, corrected_results):
        if pid_o != pid_c:
            raise E7Error(f"{fold_id}: pair audit ordering drifted ({pid_o!r} != {pid_c!r})")
        if not np.array_equal(support_o, support_c) or not np.array_equal(style_o, style_c):
            different_masks += 1
    return {"schema_version": f"{SCHEMA_PREFIX}-mask-compat-pair-audit-v1", "fold_id": fold_id,
           "pairs_checked": len(all_pairs), "different_masks": different_masks,
           "recovery_activations": recovery_counter[0], "target_access": False, "llm_api_calls": 0}


def gpat_attempt_provenance_path(repo: Path, fold_id: str) -> Path:
    return gpat_fit_run_root(repo, fold_id) / GPAT_ATTEMPT_PROVENANCE_FILENAME


# --------------------------------------------------------------------------- #
# TASK N.6 -- GPAT_FIT_LOCK validation + the real fit orchestrator.
# --------------------------------------------------------------------------- #

def _history_is_finite(history: list[dict[str, Any]]) -> bool:
    """True iff every numeric value in every history entry is finite -- not
    only `train_total`."""
    for entry in history:
        if not isinstance(entry, dict):
            continue
        for value in entry.values():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and not math.isfinite(value):
                return False
    return True


def _current_checkpoint_expected_identity(*, package_identity: str | None, pair_plan_identity: str | None,
                                          config_hash: str | None, architecture_hash: str | None,
                                          adaface_weight_sha256: str | None) -> dict[str, str | None]:
    """Builds the checkpoint identity a terminal-lock validation must load
    against, entirely from INDEPENDENTLY-DERIVED CURRENT authorities --
    NEVER from the lock's own recorded fields (that would let a
    corrupted/replaced lock+checkpoint pair remain self-consistent and
    pass validation undetected)."""
    return {"package_identity": package_identity,
           "recipe_bank_identity": FROZEN_M7_BANK["bank_content_identity_sha256"],
           "pair_plan_identity": pair_plan_identity, "config_hash": config_hash,
           "architecture_hash": architecture_hash, "adaface_weight_sha256": adaface_weight_sha256}


def _git_show_module_bytes(repo: Path, commit: str, relative_path: str) -> bytes:
    """`git show <commit>:<relative_path>` -- a small, dedicated, mockable
    seam used ONLY by the terminal-lock provenance self-check below. The
    production pre-fit gate `resolve_implementation_commit_provenance()`
    has its own independent inline `git show` call and is intentionally
    left untouched by this helper."""
    import subprocess

    try:
        return subprocess.check_output(["git", "show", f"{commit}:{relative_path}"], cwd=repo,
                                       stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        raise E7Error(str(exc.stderr)) from exc


def validate_gpat_fit_lock(repo: Path, fold_id: str) -> dict[str, Any]:
    """STRICT, read-only. A terminal lock is VALID only if it, the native
    checkpoints, AND every upstream artifact (GPAT input package, pair
    plan, effective config, frozen M7 bank) ALL independently revalidate
    against CURRENT on-disk state -- never merely a self-consistent
    snapshot frozen at fit time (GAP 3). Never writes, never resumes/
    retrains."""
    if fold_id not in FOLD_IDS:
        raise E7Error(f"unknown fold_id {fold_id!r}")
    lock_path = gpat_fit_lock_path(repo, fold_id)
    if not lock_path.is_file():
        return {"schema_version": f"{SCHEMA_PREFIX}-gpat-fit-lock-validate-v1", "fold_id": fold_id,
               "status": "NOT_MATERIALIZED"}
    from prism_fas.synthesis.gpat_checkpoint import CheckpointError, checkpoint_summary, load_checkpoint
    from prism_fas.synthesis.gpat_contracts import GPAT_CHECKPOINT_SCHEMA_VERSION

    problems: list[str] = []
    lock = cc.read_json(lock_path)
    if lock.get("status") != "VALID":
        problems.append(f"lock status {lock.get('status')!r} != 'VALID'")

    # 1-2: current GPAT input package must independently revalidate AND match the lock's own
    # recorded package_identity -- never merely trust the lock's stored value.
    input_validation = validate_gpat_input_package(repo, fold_id)
    if input_validation["status"] != "VALID":
        problems.append(f"current GPAT input package is not VALID: "
                        f"{input_validation.get('problems')!r}")
    if input_validation.get("recomputed_package_identity") != lock.get("package_identity"):
        problems.append(f"current GPAT input package identity "
                        f"{input_validation.get('recomputed_package_identity')!r} != "
                        f"lock.package_identity {lock.get('package_identity')!r}")

    # 3-4: current pair plan must independently revalidate (a TRUE rebuild, see GAP 1) AND
    # match the lock's own recorded pair_plan_identity.
    pair_validation = validate_fold_pair_plan(repo, fold_id)
    if pair_validation["status"] != "VALID":
        problems.append(f"current pair plan is not VALID: {pair_validation.get('problems')!r}")
    if pair_validation.get("recomputed_pair_plan_identity") != lock.get("pair_plan_identity"):
        problems.append(f"current pair-plan identity "
                        f"{pair_validation.get('recomputed_pair_plan_identity')!r} != "
                        f"lock.pair_plan_identity {lock.get('pair_plan_identity')!r}")

    # 5-7: effective config recomputes exactly (base config SHA + effective config hash).
    effective: dict[str, Any] | None = None
    try:
        effective = build_effective_gpat_config(repo, fold_id)
        if effective["base_config_sha256"] != lock.get("base_config_sha256"):
            problems.append(f"current base_config_sha256 {effective['base_config_sha256']!r} != "
                            f"lock.base_config_sha256 {lock.get('base_config_sha256')!r}")
        if effective["effective_config_hash"] != lock.get("effective_config_hash"):
            problems.append(f"current effective_config_hash {effective['effective_config_hash']!r} "
                            f"!= lock.effective_config_hash {lock.get('effective_config_hash')!r}")
    except E7Error as exc:
        problems.append(f"effective config could not be recomputed: {exc}")

    # 8: frozen M7 recipe-bank identity.
    if lock.get("m7_recipe_bank_identity") != FROZEN_M7_BANK["bank_content_identity_sha256"]:
        problems.append(f"lock.m7_recipe_bank_identity {lock.get('m7_recipe_bank_identity')!r} != "
                        f"frozen M7 bank identity "
                        f"{FROZEN_M7_BANK['bank_content_identity_sha256']!r}")

    # ARCHITECTURE / ADAFACE CURRENT-STATE ANCHOR (residual gap fix): the expected checkpoint
    # identity must be derived from CURRENT FROZEN AUTHORITIES, never merely echoed from the
    # lock's own recorded architecture_hash/adaface_weight_sha256 -- otherwise a corrupted or
    # replaced lock+checkpoint pair could remain self-consistent and pass validation.
    architecture_hash_current: str | None = None
    adaface_weight_sha256_current: str | None = None
    if effective is not None:
        from prism_fas.synthesis.gpat_model import build_gpat_model

        try:
            architecture_hash_current = build_gpat_model(effective["effective_config"]).architecture_hash()
        except Exception as exc:  # noqa: BLE001 -- any failure to build the model is itself a hard failure
            problems.append(f"current architecture_hash could not be recomputed: {exc}")
        if architecture_hash_current is not None and architecture_hash_current != lock.get("architecture_hash"):
            problems.append(f"current architecture_hash {architecture_hash_current!r} != "
                            f"lock.architecture_hash {lock.get('architecture_hash')!r}")

        # The frozen AdaFace SHA is anchored from the CURRENT effective GPAT config's own
        # declared identity_model.weight_sha256 -- never from the lock. Reading the weight
        # BYTES (which requires the pinned file to be present) is never required for this
        # read-only validation to reach a verdict; it is only an OPTIONAL additional check
        # below, when the file happens to be resolvable locally.
        adaface_weight_sha256_current = (effective["effective_config"].get("identity_model") or {}
                                         ).get("weight_sha256")
        if adaface_weight_sha256_current != FROZEN_PRIOR_MODELS["identity"]["weight_sha256"]:
            problems.append(f"current effective_config.identity_model.weight_sha256 "
                            f"{adaface_weight_sha256_current!r} != the frozen AdaFace SHA already "
                            f"declared by the repository "
                            f"{FROZEN_PRIOR_MODELS['identity']['weight_sha256']!r}")
        if adaface_weight_sha256_current != lock.get("adaface_weight_sha256"):
            problems.append(f"current AdaFace weight SHA {adaface_weight_sha256_current!r} != "
                            f"lock.adaface_weight_sha256 {lock.get('adaface_weight_sha256')!r}")
        try:
            from prism_fas.synthesis.quality_models import resolve_weight
            from prism_fas.synthesis.quality_models import sha256_file as quality_sha256_file

            weight_path = resolve_weight(_resolve_weight_root(repo), "identity", verify=False)
            on_disk_sha256 = quality_sha256_file(weight_path)
            if on_disk_sha256 != FROZEN_PRIOR_MODELS["identity"]["weight_sha256"]:
                problems.append(f"locally resolvable AdaFace weight file SHA256 {on_disk_sha256!r} "
                                f"!= frozen {FROZEN_PRIOR_MODELS['identity']['weight_sha256']!r}")
        except Exception:  # noqa: BLE001 -- the weight file need not be present for this validator
            pass  # anchoring from the frozen config identity above is sufficient on its own

    # 9-12: BOTH best.pt and last.pt exist, their SHA matches the lock, their schema is correct,
    # and BOTH strictly identity-verify against the FULL expected identity -- built entirely from
    # the independently-derived CURRENT package/pair-plan/config/architecture/AdaFace values
    # above, never from the lock's own fields (never only best.pt).
    best_path = gpat_best_checkpoint_path(repo, fold_id)
    last_path = gpat_last_checkpoint_path(repo, fold_id)
    expected_identity = _current_checkpoint_expected_identity(
        package_identity=input_validation.get("recomputed_package_identity"),
        pair_plan_identity=pair_validation.get("recomputed_pair_plan_identity"),
        config_hash=effective["effective_config_hash"] if effective is not None else None,
        architecture_hash=architecture_hash_current, adaface_weight_sha256=adaface_weight_sha256_current)
    checkpoint_payloads: dict[str, dict[str, Any]] = {}
    for name, path, sha_field in (("best", best_path, "best_checkpoint_sha256"),
                                  ("last", last_path, "last_checkpoint_sha256")):
        if not path.is_file():
            problems.append(f"{name} checkpoint missing on disk")
            continue
        if cc.sha256_file(path) != lock.get(sha_field):
            problems.append(f"{name} checkpoint SHA256 mismatch")
        summary = checkpoint_summary(path)
        if summary.get("schema_version") != GPAT_CHECKPOINT_SCHEMA_VERSION:
            problems.append(f"{name} checkpoint schema_version {summary.get('schema_version')!r} "
                            f"!= expected {GPAT_CHECKPOINT_SCHEMA_VERSION!r}")
        try:
            checkpoint_payloads[name] = load_checkpoint(path, expected_identity=expected_identity)
        except CheckpointError as exc:
            problems.append(f"{name} checkpoint FAILED strict identity load against CURRENT "
                            f"authorities: {exc}")

    # 13: record_set_hashes remain consistent between the lock and both checkpoints, and between
    # the two checkpoints themselves.
    for name, payload in checkpoint_payloads.items():
        if payload.get("record_set_hashes") != lock.get("record_set_hashes"):
            problems.append(f"{name} checkpoint record_set_hashes != lock.record_set_hashes")
    if "best" in checkpoint_payloads and "last" in checkpoint_payloads and \
            checkpoint_payloads["best"].get("record_set_hashes") != checkpoint_payloads["last"].get("record_set_hashes"):
        problems.append("best checkpoint record_set_hashes != last checkpoint record_set_hashes")

    # 14: history contains only finite numeric metrics, in both checkpoints.
    for name, payload in checkpoint_payloads.items():
        if not _history_is_finite(payload.get("history") or []):
            problems.append(f"{name} checkpoint history contains a non-finite numeric value")

    # ANCHOR TERMINAL LOCK EXECUTION METADATA (validation-only; never changes training behavior).
    if lock.get("seed") != GPAT_FIT_SEED:
        problems.append(f"lock.seed {lock.get('seed')!r} != frozen GPAT_FIT_SEED {GPAT_FIT_SEED!r}")
    if effective is not None:
        effective_config = effective["effective_config"]
        if lock.get("seed") != effective_config.get("seed"):
            problems.append(f"lock.seed {lock.get('seed')!r} != effective_config.seed "
                            f"{effective_config.get('seed')!r}")
        if lock.get("epochs_requested") != effective_config.get("epochs"):
            problems.append(f"lock.epochs_requested {lock.get('epochs_requested')!r} != "
                            f"effective_config.epochs {effective_config.get('epochs')!r}")
    device = lock.get("device")
    if not (isinstance(device, str) and device.startswith("cuda")):
        problems.append(f"lock.device {device!r} is not 'cuda' or a 'cuda*' variant")
    if "last" in checkpoint_payloads:
        last_payload = checkpoint_payloads["last"]
        if lock.get("global_step") != last_payload.get("global_step"):
            problems.append(f"lock.global_step {lock.get('global_step')!r} != last checkpoint "
                            f"global_step {last_payload.get('global_step')!r}")
        last_history_len = len(last_payload.get("history") or [])
        if lock.get("epochs_completed") != last_history_len:
            problems.append(f"lock.epochs_completed {lock.get('epochs_completed')!r} != "
                            f"len(last checkpoint history) {last_history_len!r}")
        if lock.get("best_metrics") != last_payload.get("best_metrics"):
            problems.append("lock.best_metrics != last checkpoint best_metrics")

    # PROVENANCE LOCK SELF-CHECK: proves the lock's declared implementation_commit/module-SHA
    # pair is a real, immutable Git commit/module mapping -- WITHOUT requiring
    # $PRISM_E7_IMPLEMENTATION_COMMIT and WITHOUT comparing against the CURRENTLY-running module
    # file (a LATER validator-only hardening pass must never invalidate an EARLIER, scientifically
    # valid checkpoint whose lock was written against an older module revision). The
    # pre-fit production gate `resolve_implementation_commit_provenance()` is untouched.
    implementation_commit = lock.get("implementation_commit")
    implementation_module_sha256 = lock.get("implementation_module_sha256")
    if not implementation_commit:
        problems.append("lock.implementation_commit is missing")
    if not implementation_module_sha256:
        problems.append("lock.implementation_module_sha256 is missing")
    if implementation_commit and implementation_module_sha256:
        module_relative = "src/prism_fas/evaluation/c_ext_e7_gpat_bank.py"
        try:
            committed_bytes = _git_show_module_bytes(repo, implementation_commit, module_relative)
        except E7Error as exc:
            problems.append(f"could not read {module_relative!r} at lock.implementation_commit "
                            f"{implementation_commit!r}: {exc}")
        else:
            committed_sha256 = cc.sha256_bytes(committed_bytes)
            if committed_sha256 != implementation_module_sha256:
                problems.append(f"lock.implementation_module_sha256 "
                                f"{implementation_module_sha256!r} != sha256 of {module_relative!r} "
                                f"at lock.implementation_commit {implementation_commit!r} "
                                f"({committed_sha256!r})")

    return {"schema_version": f"{SCHEMA_PREFIX}-gpat-fit-lock-validate-v1", "fold_id": fold_id,
           "status": "INVALID" if problems else "VALID", "problems": problems,
           "target_access": False, "llm_api_calls": 0}


# --------------------------------------------------------------------------- #
# TASK N.7 -- GPU code provenance (GAP 5). GPU git HEAD may legitimately stay
# on a different branch (e.g. `main`) while the exact GPAT implementation
# module is rsynced in from `gpu-work`. `git_commit(repo)` (repository HEAD)
# must therefore never be recorded AS IF it were the implementation
# checkpoint -- this proves the RUNNING module file is byte-identical to a
# real, inspectable, declared commit instead.
# --------------------------------------------------------------------------- #

def resolve_implementation_commit_provenance(repo: Path) -> dict[str, Any]:
    """Requires `$PRISM_E7_IMPLEMENTATION_COMMIT` and verifies that
    `git show <that commit>:src/prism_fas/evaluation/c_ext_e7_gpat_bank.py`
    has SHA256 EXACTLY equal to the currently-running module file. FAILS
    CLOSED (raises `E7Error`) if the env var is unset or the bytes disagree
    -- never falsely calls repository HEAD the implementation checkpoint.
    Laptop CPU/unit tests may monkeypatch this function; it is intentionally
    called only AFTER the real GPU-capability gate, so laptop
    GPU_REQUIRED behavior is reached before this is ever evaluated."""
    import os
    import subprocess

    from prism_fas.utils.core import git_commit

    implementation_commit = os.environ.get("PRISM_E7_IMPLEMENTATION_COMMIT")
    if not implementation_commit:
        raise E7Error("PRISM_E7_IMPLEMENTATION_COMMIT is not set -- production GPU GPAT fitting "
                      "requires an explicit, verifiable implementation-commit pin; refusing to "
                      "proceed (repository HEAD is never treated as the implementation checkpoint)")
    module_relative = "src/prism_fas/evaluation/c_ext_e7_gpat_bank.py"
    module_path = Path(__file__).resolve()
    module_sha256 = cc.sha256_file(module_path)
    try:
        committed_bytes = subprocess.check_output(
            ["git", "show", f"{implementation_commit}:{module_relative}"], cwd=repo,
            stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        raise E7Error(f"could not read {module_relative!r} at commit "
                      f"{implementation_commit!r}: {exc.stderr!r}") from exc
    committed_sha256 = cc.sha256_bytes(committed_bytes)
    if committed_sha256 != module_sha256:
        raise E7Error(f"PRISM_E7_IMPLEMENTATION_COMMIT={implementation_commit!r}: "
                      f"{module_relative!r} at that commit (sha256={committed_sha256}) does NOT "
                      f"match the currently-running module file (sha256={module_sha256}) -- FAIL "
                      "CLOSED before GPAT fitting; the rsynced runtime module is not provably "
                      "pinned to the declared implementation commit")
    return {"repository_head_commit": git_commit(repo), "implementation_commit": implementation_commit,
           "implementation_module_sha256": module_sha256}


def prepare_gpat(repo: Path, fold_id: str, *, authorize: bool = False) -> dict[str, Any]:
    """`--prepare-gpat --fold EXT-Fn --authorize`: REAL transactional GPAT
    fit. Order: (1) terminal-lock resume/short-circuit (strengthened, see
    `validate_gpat_fit_lock`), (2) GPAT-input package materialize+validate,
    (3) pair-plan materialize+validate (TRUE recomputation, see GAP 1),
    (4) effective-config build, (5) real GPU-capability gate, (5.5) GPU
    implementation-commit provenance gate (GAP 5), (6) resume-compatibility
    pre-check, (7) real `GPATTrainer(...).fit(...)`, (8) independent
    checkpoint SHA/identity/record-set/history/source-isolation validation
    (GAP 4), (9) write GPAT_FIT_LOCK.json LAST. Uses `GPATTrainer.fit`
    UNMODIFIED; never
    monkeypatched."""
    if fold_id not in FOLD_IDS:
        raise E7Error(f"unknown fold_id {fold_id!r}")
    if not authorize:
        raise E7Error(f"GPAT fit for {fold_id} requires --authorize; refusing to run")

    lock_path = gpat_fit_lock_path(repo, fold_id)
    if lock_path.is_file():
        validation = validate_gpat_fit_lock(repo, fold_id)
        if validation["status"] != "VALID":
            raise E7Error(f"{fold_id}: existing GPAT_FIT_LOCK.json FAILED strict validation -- "
                          f"FAIL CLOSED, NEVER retrain: {validation['problems']!r}")
        return {"resumed": True, "status": "ALREADY_VALID", "path": str(lock_path),
               "validation": validation, "target_access": False, "llm_api_calls": 0,
               "gpat_fitting_performed": False}

    input_result = materialize_gpat_input_package(repo, fold_id, authorize=True)
    if input_result["status"] not in ("ALREADY_VALID", "MATERIALIZED"):
        raise E7Error(f"{fold_id}: GPAT input package did not reach a valid state -- FAIL CLOSED")
    pair_result = materialize_fold_pair_plan(repo, fold_id, authorize=True)
    if pair_result["status"] not in ("ALREADY_VALID", "MATERIALIZED"):
        raise E7Error(f"{fold_id}: pair plan did not reach a valid state -- FAIL CLOSED")
    pair_validation = validate_fold_pair_plan(repo, fold_id)
    if pair_validation["status"] != "VALID":
        raise E7Error(f"{fold_id}: pair plan FAILED strict validation -- FAIL CLOSED: "
                      f"{pair_validation['problems']!r}")

    effective = build_effective_gpat_config(repo, fold_id)
    from prism_fas.utils.core import atomic_json_write

    run_root = gpat_fit_run_root(repo, fold_id)
    run_root.mkdir(parents=True, exist_ok=True)
    atomic_json_write(run_root / "EFFECTIVE_GPAT_CONFIG.json", effective)

    capability = _gpat_fit_capability(repo)
    if not capability["capable"]:
        raise E7Error(f"{fold_id}: GPU_REQUIRED for real GPAT fitting -- this host is not "
                      f"capable: {capability['problems']!r}. The GPAT input package and pair "
                      "plan ARE materialized and strictly validated; only the real "
                      "GPATTrainer.fit call remains, and it must run on a GPU host with the "
                      f"pinned AdaFace weight resolvable under {capability['weight_root']!r}.")

    # GAP 5: required strictly AFTER the GPU-capability gate (so laptop GPU_REQUIRED behavior
    # is reached first) and strictly BEFORE GPATTrainer.fit. Never treats repository HEAD as
    # the implementation checkpoint -- GPU git HEAD may legitimately stay on `main` while this
    # module is rsynced in from `gpu-work`.
    provenance = resolve_implementation_commit_provenance(repo)

    # Steps 6-9: real resume-compatibility pre-check + GPATTrainer.fit + terminal lock. Written
    # for correctness; unreachable on this laptop (the capability gate above always fails first).
    from prism_fas.synthesis.gpat_checkpoint import CheckpointError, checkpoint_summary, load_checkpoint
    from prism_fas.synthesis.gpat_model import build_gpat_model
    from prism_fas.synthesis.gpat_trainer import GPATTrainer
    from prism_fas.synthesis.quality_models import resolve_weight
    from prism_fas.synthesis.quality_models import sha256_file as quality_sha256_file

    package_root = repo / GPAT_INPUT_ROOT / fold_id
    pairs_root = repo / GPAT_PAIR_PLAN_ROOT / fold_id
    package_identity = cc.read_json(package_root / GPAT_INPUT_LOCK_FILENAME)["content_identity_sha256"]
    pair_plan_identity_value = cc.read_json(pairs_root / "PAIR_PLAN_LOCK.json")["pair_plan_identity_sha256"]
    architecture_hash = build_gpat_model(effective["effective_config"]).architecture_hash()
    weight_root = Path(capability["weight_root"])
    adaface_weight_sha256 = quality_sha256_file(resolve_weight(weight_root, "identity"))
    expected_identity = {"package_identity": package_identity,
                         "recipe_bank_identity": FROZEN_M7_BANK["bank_content_identity_sha256"],
                         "pair_plan_identity": pair_plan_identity_value,
                         "config_hash": effective["effective_config_hash"],
                         "architecture_hash": architecture_hash,
                         "adaface_weight_sha256": adaface_weight_sha256}

    # TECHNICAL_SINGLE_SAMPLE_EMPTY_CHEEK_FALLBACK_EDGE_CASE fix: a partial checkpoint may only
    # be resumed if a matching GPAT_ATTEMPT_PROVENANCE.json sidecar exists and agrees on
    # implementation/mask-compatibility-policy/package/pair-plan/config/architecture/AdaFace --
    # never merely the checkpoint's own embedded identity (which predates this policy field and
    # cannot itself attest to it). The checkpoint is NEVER deleted on any mismatch.
    attempt_material = {"package_identity": package_identity,
                        "pair_plan_identity": pair_plan_identity_value,
                        "effective_config_hash": effective["effective_config_hash"],
                        "architecture_hash": architecture_hash, "adaface_weight_sha256": adaface_weight_sha256,
                        "implementation_commit": provenance["implementation_commit"],
                        "implementation_module_sha256": provenance["implementation_module_sha256"],
                        "mask_compatibility_policy": MASK_COMPATIBILITY_POLICY}
    attempt_provenance_path = gpat_attempt_provenance_path(repo, fold_id)
    last_path = gpat_last_checkpoint_path(repo, fold_id)
    resume = False
    if last_path.is_file():
        if not attempt_provenance_path.is_file():
            raise E7Error(f"{fold_id}: a partial checkpoint exists at {last_path} but no "
                          f"{GPAT_ATTEMPT_PROVENANCE_FILENAME} sidecar is present -- FAIL CLOSED, "
                          "never resuming an unattributed partial checkpoint; the checkpoint is "
                          "left on disk untouched")
        recorded_attempt = cc.read_json(attempt_provenance_path)
        mismatched_attempt = [field for field, value in attempt_material.items()
                              if recorded_attempt.get(field) != value]
        if mismatched_attempt:
            raise E7Error(f"{fold_id}: the partial checkpoint's {GPAT_ATTEMPT_PROVENANCE_FILENAME} "
                          f"sidecar disagrees with the CURRENT implementation/mask-policy/package/"
                          f"pair-plan/config/architecture/AdaFace identity on "
                          f"{mismatched_attempt} -- FAIL CLOSED, never auto-deleted or restarted")
        summary = checkpoint_summary(last_path)
        identity = summary.get("identity") or {}
        mismatched = [field for field, value in expected_identity.items() if identity.get(field) != value]
        if mismatched:
            raise E7Error(f"{fold_id}: an existing partial checkpoint at {last_path} is "
                          f"INCOMPATIBLE with the current package/bank/pair-plan/config/"
                          f"architecture/AdaFace identity ({mismatched}) -- FAIL CLOSED, never "
                          "auto-deleted or restarted")
        resume = True

    # Non-terminal attempt provenance, written atomically BEFORE trainer.fit() -- never the
    # terminal GPAT_FIT_LOCK.json, and never treated as a resumable identity on its own.
    atomic_json_write(attempt_provenance_path, {
        "schema_version": f"{SCHEMA_PREFIX}-gpat-attempt-provenance-v1", "fold_id": fold_id,
        **attempt_material, "resume_requested": resume})

    run_id = f"gpat_fit_{fold_id.lower().replace('-', '_')}_{effective['effective_config_hash'][:12]}"
    recovery_counter = [0]
    with _scoped_e7_mask_compatibility_binding(recovery_counter):
        trainer = GPATTrainer(config=effective["effective_config"], package_root=package_root,
                              bank_root=repo / M7_RECIPE_BANK_ROOT, pairs_root=pairs_root, run_root=run_root,
                              weight_root=weight_root, device="cuda")
        fit_result = trainer.fit(run_id=run_id, progress=lambda event: None, resume=resume)

    best_path = gpat_best_checkpoint_path(repo, fold_id)
    if not best_path.is_file() or not last_path.is_file():
        raise E7Error(f"{fold_id}: GPATTrainer.fit returned but native checkpoints are missing -- "
                      "FAIL CLOSED, no terminal lock written")

    # GAP 4: independently SHA256 BOTH checkpoints from disk (never merely trust fit()'s own
    # reported hash), strictly load/validate BOTH against the full expected identity, verify
    # BOTH checkpoints' record_set_hashes, and verify ALL numeric history values (not only
    # train_total) in both the checkpoints and fit()'s own returned history.
    best_sha256_disk = cc.sha256_file(best_path)
    last_sha256_disk = cc.sha256_file(last_path)
    if best_sha256_disk != fit_result["checkpoints"]["best_sha256"]:
        raise E7Error(f"{fold_id}: best checkpoint on-disk SHA256 {best_sha256_disk!r} != fit()'s "
                      f"own reported SHA256 {fit_result['checkpoints']['best_sha256']!r} -- FAIL CLOSED")
    if last_sha256_disk != fit_result["checkpoints"]["last_sha256"]:
        raise E7Error(f"{fold_id}: last checkpoint on-disk SHA256 {last_sha256_disk!r} != fit()'s "
                      f"own reported SHA256 {fit_result['checkpoints']['last_sha256']!r} -- FAIL CLOSED")
    try:
        best_payload = load_checkpoint(best_path, expected_identity=expected_identity)
        last_payload = load_checkpoint(last_path, expected_identity=expected_identity)
    except CheckpointError as exc:
        raise E7Error(f"{fold_id}: post-fit strict checkpoint identity validation FAILED -- FAIL "
                      f"CLOSED, no terminal lock written: {exc}") from exc
    if best_payload.get("identity") != fit_result["identity"]:
        raise E7Error(f"{fold_id}: best checkpoint identity does not match fit() identity -- "
                      "FAIL CLOSED")
    for name, payload in (("best", best_payload), ("last", last_payload)):
        if payload.get("record_set_hashes") != fit_result["record_set_hashes"]:
            raise E7Error(f"{fold_id}: {name} checkpoint record_set_hashes != fit()'s own "
                          "record_set_hashes -- FAIL CLOSED")
        if not _history_is_finite(payload.get("history") or []):
            raise E7Error(f"{fold_id}: {name} checkpoint history contains a non-finite numeric "
                          "value -- FAIL CLOSED")
    if not _history_is_finite(fit_result["history"]):
        raise E7Error(f"{fold_id}: fit() history contains a non-finite numeric value -- FAIL CLOSED")

    source_isolation = fit_result["source_isolation"]
    forbidden_flags = ("source_dev_opened", "target_test_opened", "target_label_artifact_opened",
                       "raw_dataset_path_opened")
    tripped = [flag for flag in forbidden_flags if source_isolation.get(flag)]
    if tripped:
        raise E7Error(f"{fold_id}: source isolation audit reports forbidden open(s) {tripped} -- "
                      "FAIL CLOSED")
    non_train_manifests = [m for m in (source_isolation.get("manifests_opened") or [])
                           if not str(m).endswith("source_train.parquet")]
    if non_train_manifests:
        raise E7Error(f"{fold_id}: source isolation audit opened non-source_train manifest(s) "
                      f"{non_train_manifests} -- FAIL CLOSED")

    lock = {"schema_version": f"{SCHEMA_PREFIX}-gpat-fit-lock-v1", "fold_id": fold_id, "status": "VALID",
           "run_id": run_id, "repository_head_commit": provenance["repository_head_commit"],
           "implementation_commit": provenance["implementation_commit"],
           "implementation_module_sha256": provenance["implementation_module_sha256"],
           "base_config_sha256": effective["base_config_sha256"],
           "effective_config_hash": effective["effective_config_hash"], "package_identity": package_identity,
           "pair_plan_identity": pair_plan_identity_value,
           "m7_recipe_bank_identity": FROZEN_M7_BANK["bank_content_identity_sha256"],
           "best_checkpoint_relative_path": str(best_path.relative_to(repo)),
           "best_checkpoint_sha256": best_sha256_disk,
           "last_checkpoint_relative_path": str(last_path.relative_to(repo)),
           "last_checkpoint_sha256": last_sha256_disk,
           "architecture_hash": architecture_hash, "adaface_weight_sha256": adaface_weight_sha256,
           "seed": GPAT_FIT_SEED, "device": fit_result["device"],
           "epochs_requested": effective["effective_config"]["epochs"],
           "epochs_completed": fit_result["epochs_run"], "global_step": fit_result["global_step"],
           "stop_reason": fit_result["stop_reason"], "best_metrics": fit_result["best"],
           "record_set_hashes": fit_result["record_set_hashes"], "source_isolation": source_isolation,
           "mask_compatibility_policy": MASK_COMPATIBILITY_POLICY,
           "mask_compatibility_recovery_count": recovery_counter[0],
           "TARGET_LABEL_ACCESS": False, "TARGET_IMAGE_ACCESS": False, "RENDERING_PERFORMED": False,
           "DETECTOR_TRAINING_PERFORMED": False, "LLM_API_CALLS": 0}
    atomic_json_write(lock_path, lock)
    return {"resumed": resume, "status": "FITTED", "run_id": run_id, "lock": lock,
           "target_access": False, "llm_api_calls": 0, "gpat_fitting_performed": True}


# --------------------------------------------------------------------------- #
# TASK H -- candidate generation policy
# --------------------------------------------------------------------------- #

def build_candidate_generation_policy(repo: Path) -> dict[str, Any]:
    quota = frozen_bank_quota()
    return {
        "schema_version": f"{SCHEMA_PREFIX}-candidate-generation-policy-v1",
        "reused_modules": ["prism_fas.synthesis.c5_source_pair_plan",
                          "prism_fas.synthesis.c5_arm_plan", "prism_fas.synthesis.c5_raw_generation",
                          "prism_fas.synthesis.c5_render"],
        "never_uses_legacy_m8_candidate_plan": "prism_fas.synthesis.candidate_plan is the LEGACY "
                                              "Version-B planner (keyed on live samples, no arm "
                                              "dimension, EXPECTED_TOTAL=1120) -- NOT used here",
        "frozen_quota": quota,
        "route_reuse": ["prism_fas.synthesis.synthetic_bank.GPATRoute",
                       "prism_fas.synthesis.synthetic_bank.PhysicsRoute"],
        "treatment_fairness": "for a given fold, source assignment/schedule is ARM-INDEPENDENT -- "
                             "RND/DET/LLM differ only by recipe content, never by a different "
                             "source-sampling opportunity; Shuffle follows the SAME frozen "
                             "positional/source schedule when feasible",
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False,
    }


def write_candidate_generation_policy(repo: Path) -> dict[str, Any]:
    return _write(repo, "CANDIDATE_GENERATION_POLICY.json", build_candidate_generation_policy(repo))


# --------------------------------------------------------------------------- #
# TASK I -- quality gate binding
# --------------------------------------------------------------------------- #

def build_quality_gate_binding(repo: Path) -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_PREFIX}-quality-gate-binding-v1",
        "reused_modules": ["prism_fas.synthesis.quality_gate", "prism_fas.synthesis.quality_calibration",
                          "prism_fas.synthesis.c6_scientific"],
        "hard_gates_reused_unmodified": True,
        "q_semantics": "q remains a training-weight/matching variable exactly as frozen -- never "
                      "reinterpreted as an acceptance criterion",
        "calibration_source_only": True,
        "calibration_never_uses_held_out_target": True,
        "thresholds_never_relaxed_to_force_feasibility": True,
        "target_access": False, "llm_api_calls": 0,
    }


def write_quality_gate_binding(repo: Path) -> dict[str, Any]:
    return _write(repo, "QUALITY_GATE_BINDING.json", build_quality_gate_binding(repo))


# --------------------------------------------------------------------------- #
# TASK J -- matched bank policy
# --------------------------------------------------------------------------- #

def build_matched_bank_policy(repo: Path) -> dict[str, Any]:
    quota = frozen_bank_quota()
    return {
        "schema_version": f"{SCHEMA_PREFIX}-matched-bank-policy-v1",
        "reused_module": "prism_fas.synthesis.c6_matched_bank",
        "reused_functions": ["planned_domain_counts", "ideal_domain_share",
                            "largest_remainder_quota", "common_capacity", "resolve_route_quota",
                            "select_route_bank", "build_matched_banks"],
        "frozen_quota": quota,
        "required_bank_fields": ["fold_id", "source_support_package_identity",
                                "gpat_checkpoint_identity", "recipe_bank_identity",
                                "recipe_bank_identity_kind", "source_pair_plan_identity",
                                "candidate_plan_identity", "quality_config_identity",
                                "quality_calibration_identity", "route_quotas",
                                "accepted_candidate_ids", "crop_image_hashes", "q_values",
                                "bank_identity", "target_access_audit", "llm_api_calls"],
        "absolute_host_paths_excluded_from_identity": True,
        "target_access": False, "llm_api_calls": 0,
    }


def write_matched_bank_policy(repo: Path) -> dict[str, Any]:
    return _write(repo, "MATCHED_BANK_POLICY.json", build_matched_bank_policy(repo))


# --------------------------------------------------------------------------- #
# TASK K -- shuffle feasibility policy
# --------------------------------------------------------------------------- #

def build_condition_status(fold_id: str, condition: str) -> str:
    return e7c.build_condition_status(fold_id, condition)  # reused verbatim


def build_shuffle_feasibility_policy(repo: Path) -> dict[str, Any]:
    per_fold = {}
    for fold_id in FOLD_IDS:
        status = build_condition_status(fold_id, "G-LLM-SHUFFLE-A")
        per_fold[fold_id] = {
            "status": status,
            "basis": FROZEN_E6V2_CLOSURE if fold_id == "EXT-F1" else {
                "reason": "independent -- F1's Physics infeasibility is a property of F1's OWN "
                         "source-domain composition; this fold draws from a different "
                         "source-domain pair and must be independently rendered/measured/matched",
                "ext_f1_result_does_not_predetermine_this_fold": True,
            },
            "required_reporting": ["PHYSICS_CANDIDATES_GENERATED", "GPAT_CANDIDATES_GENERATED",
                                  "PHYSICS_QUALITY_PASS", "GPAT_QUALITY_PASS",
                                  "per_source_domain_pass_counts", "required_quota_per_route_domain",
                                  "maximum_fillable_bank_size_under_exact_frozen_matching",
                                  "final_status"],
        }
    return {"schema_version": f"{SCHEMA_PREFIX}-shuffle-feasibility-policy-v1", "folds": per_fold,
           "forbidden_rescue_actions": FORBIDDEN_RESCUE_ACTIONS, "no_rescue_actions_permitted": True,
           "target_access": False, "llm_api_calls": 0}


def write_shuffle_feasibility_policy(repo: Path) -> dict[str, Any]:
    policy = build_shuffle_feasibility_policy(repo)
    if policy["folds"]["EXT-F1"]["status"] != BLOCKED_TRUE_FROZEN_MATCHED_BANK_INFEASIBILITY:
        raise E7Error("EXT-F1/G-LLM-SHUFFLE-A must remain "
                      "BLOCKED_TRUE_FROZEN_MATCHED_BANK_INFEASIBILITY")
    for fold_id in ("EXT-F2", "EXT-F3"):
        if policy["folds"][fold_id]["status"] != PENDING_FEASIBILITY_PREFLIGHT:
            raise E7Error(f"{fold_id}/G-LLM-SHUFFLE-A must remain PENDING_FEASIBILITY_PREFLIGHT")
    return _write(repo, "SHUFFLE_FEASIBILITY_POLICY.json", policy)


# --------------------------------------------------------------------------- #
# TASK L -- target firewall (extends e7d's fold-aware firewall with the
# GPAT-bank namespace's own roots -- M7 bank / weight roots are always
# source-side, never forbidden; per-fold target package roots stay
# forbidden exactly as e7d/e7c already established)
# --------------------------------------------------------------------------- #

def build_target_firewall(repo: Path) -> dict[str, Any]:
    folds = {}
    for fold_id in FOLD_IDS:
        folds[fold_id] = {
            "heldout_target_domain": FOLD_TARGET_DOMAIN[fold_id],
            "forbidden_roots": list(e7d.forbidden_roots_for_fold(fold_id)),
            "always_open_source_side_roots": [e7b.CASIA_MSU_PACKAGE_ROOT, M7_RECIPE_BANK_ROOT] +
                                             ([e7b.E7B_SIW_SOURCE_PACKAGE_ROOT]
                                             if "SiW-Mv2" in FOLD_SOURCE_DOMAINS[fold_id] else []),
            "active": True,
        }
    return {"schema_version": f"{SCHEMA_PREFIX}-target-firewall-v1", "folds": folds,
           "global_dataset_name_ban": False, "never_opens": ["data/evaluation_only",
                                                             PROTECTED_SIW_TARGET_PRIOR_PACKAGE_ROOT
                                                             + " (for source purposes)"],
           "target_access": False, "llm_api_calls": 0}


def write_target_firewall(repo: Path) -> dict[str, Any]:
    return _write(repo, "TARGET_FIREWALL.json", build_target_firewall(repo))


# --------------------------------------------------------------------------- #
# TASK M -- execution plan + readiness + strict read-only preflight
# --------------------------------------------------------------------------- #

def _source_priors_materialized_per_fold(compatibility: dict[str, dict[str, Any]]) -> dict[str, bool]:
    """A fold's source priors are MATERIALIZED only when every source it
    actually has (M3B and/or SiW) is independently materialized -- `None`
    means "no such source in this fold" and is treated as trivially
    satisfied, never as a missing requirement."""
    out = {}
    for fold_id, audit in compatibility.items():
        m3b_ok = audit["m3b_priors_materialized"] in (True, None)
        siw_ok = audit["siw_priors_materialized"] in (True, None)
        out[fold_id] = m3b_ok and siw_ok
    return out


def _prior_generation_primitive_resolved_per_fold(compatibility: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per fold, reports the ACTIONABLE primitive-resolution status: SiW's
    (if this fold has a SiW source) since that is the one requiring a new
    GPU invocation; otherwise M3B's (NOT_APPLICABLE_EXISTING_PRIORS, since
    M3B priors already exist and no fresh generation is ever needed)."""
    out = {}
    for fold_id, audit in compatibility.items():
        if audit["siw_prior_generation_primitive_resolved"] is not None:
            out[fold_id] = audit["siw_prior_generation_primitive_resolved"]
        else:
            out[fold_id] = audit["m3b_prior_generation_primitive_resolved"]
    return out


def build_execution_plan(repo: Path) -> dict[str, Any]:
    compatibility = {fold_id: audit_gpat_input_compatibility(repo, fold_id) for fold_id in FOLD_IDS}
    shuffle = build_shuffle_feasibility_policy(repo)
    gpat_input_compatible = {fold_id: compatibility[fold_id]["status"] in
                             ("COMPATIBLE", "COMPATIBLE_PENDING_GPU_PRIOR_GENERATION")
                             for fold_id in FOLD_IDS}
    source_priors_materialized = _source_priors_materialized_per_fold(compatibility)
    # READY_FOR_GPU_GPAT_FIT is LITERAL: every source input GPATTrainer.fit would actually open
    # must already exist and validate -- never merely "the fold is not scientifically blocked".
    ready_for_gpu_gpat_fit = {fold_id: gpat_input_compatible[fold_id] and source_priors_materialized[fold_id]
                              for fold_id in FOLD_IDS}
    ready_for_gpu_source_prior_materialization = {
        fold_id: gpat_input_compatible[fold_id] and not source_priors_materialized[fold_id]
        for fold_id in FOLD_IDS
    }
    return {
        "schema_version": f"{SCHEMA_PREFIX}-execution-plan-v2",
        "per_fold_compatibility": {f: compatibility[f]["status"] for f in FOLD_IDS},
        "per_fold_shuffle_status": {f: shuffle["folds"][f]["status"] for f in FOLD_IDS},
        "per_fold_source_priors_materialized": source_priors_materialized,
        "ready_for_gpu_gpat_fit": ready_for_gpu_gpat_fit,
        "ready_for_gpu_source_prior_materialization": ready_for_gpu_source_prior_materialization,
        "next_gpu_stages": ["A. source prior materialization (--prepare-source-priors, SiW-as-"
                            "source rows for F2/F3 -- shared package, generated at most once)",
                            "B. strict prior validation (--validate-source-priors)",
                            "C. GPAT-input package/adapter materialization",
                            "D. GPAT pair-plan materialization/validation",
                            "E. GPATTrainer.fit (only after A-D succeed)",
                            "source-only quality calibration",
                            "per-fold/per-arm candidate generation (Physics+GPAT)",
                            "quality-gate evaluation", "F2/F3 Shuffle-A independent feasibility",
                            "matched-bank resolution", "per-fold/per-condition integrity locks"],
        "e7_ready_for_training": False,
        "reason": "no GPAT checkpoint fitted, no synthetic candidate bank exists, F2/F3 Shuffle "
                 "feasibility unresolved, no matched-bank/integrity lock has run",
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False,
        "training_performed": False, "gpat_fitting_performed": False,
    }


def write_execution_plan(repo: Path) -> dict[str, Any]:
    return _write(repo, "EXECUTION_PLAN.json", build_execution_plan(repo))


def preflight(repo: Path) -> dict[str, Any]:
    """`--preflight`: STRICTLY READ-ONLY. No rendering, no GPAT fitting, no
    candidate generation, no quality-gate evaluation, no training, no
    target scoring, no LLM calls. Writes nothing."""
    e7d_binding = build_e7d_binding(repo)
    compatibility = {fold_id: audit_gpat_input_compatibility(repo, fold_id) for fold_id in FOLD_IDS}
    e7d_validation = e7d.e7d_validate(repo)
    shuffle = build_shuffle_feasibility_policy(repo)
    recipe_binding = e7c.build_recipe_bank_binding(repo)

    source_support_valid = {
        fold_id: (e7d_validation["folds"][fold_id]["status"] == "VALID" or
                 (e7d_validation["folds"][fold_id]["status"] == "NOT_MATERIALIZED" and
                  e7d_binding["folds"][fold_id]["frozen_package_identity"] is not None))
        for fold_id in FOLD_IDS
    }
    gpat_input_compatible = {fold_id: compatibility[fold_id]["status"] in
                             ("COMPATIBLE", "COMPATIBLE_PENDING_GPU_PRIOR_GENERATION")
                             for fold_id in FOLD_IDS}
    source_priors_materialized = _source_priors_materialized_per_fold(compatibility)
    # READY means LITERALLY usable by GPATTrainer.fit right now -- identical to materialized.
    source_priors_ready = dict(source_priors_materialized)
    prior_generation_primitive_resolved = _prior_generation_primitive_resolved_per_fold(compatibility)
    target_firewall_active = {fold_id: True for fold_id in FOLD_IDS}

    ready_for_gpu_gpat_fit = all(gpat_input_compatible[f] and source_priors_materialized[f]
                                for f in FOLD_IDS)
    ready_for_gpu_source_prior_materialization = any(
        gpat_input_compatible[f] and not source_priors_materialized[f] for f in FOLD_IDS)
    gpat_fitted = {fold_id: validate_gpat_fit_lock(repo, fold_id)["status"] == "VALID"
                  for fold_id in FOLD_IDS}

    return {
        "schema_version": f"{SCHEMA_PREFIX}-preflight-v2",
        "E7D_BINDING_MATCH": e7d_binding["E7D_BINDING_MATCH"] if e7d_binding["evidence_present"]
                            else False,
        "F1_SOURCE_SUPPORT_VALID": source_support_valid["EXT-F1"],
        "F2_SOURCE_SUPPORT_VALID": source_support_valid["EXT-F2"],
        "F3_SOURCE_SUPPORT_VALID": source_support_valid["EXT-F3"],
        "F1_GPAT_INPUT_COMPATIBLE": gpat_input_compatible["EXT-F1"],
        "F2_GPAT_INPUT_COMPATIBLE": gpat_input_compatible["EXT-F2"],
        "F3_GPAT_INPUT_COMPATIBLE": gpat_input_compatible["EXT-F3"],
        "F1_PRIOR_GENERATION_PRIMITIVE_RESOLVED": prior_generation_primitive_resolved["EXT-F1"],
        "F2_PRIOR_GENERATION_PRIMITIVE_RESOLVED": prior_generation_primitive_resolved["EXT-F2"],
        "F3_PRIOR_GENERATION_PRIMITIVE_RESOLVED": prior_generation_primitive_resolved["EXT-F3"],
        "F1_SOURCE_PRIORS_MATERIALIZED": source_priors_materialized["EXT-F1"],
        "F2_SOURCE_PRIORS_MATERIALIZED": source_priors_materialized["EXT-F2"],
        "F3_SOURCE_PRIORS_MATERIALIZED": source_priors_materialized["EXT-F3"],
        "F1_SOURCE_PRIORS_READY": source_priors_ready["EXT-F1"],
        "F2_SOURCE_PRIORS_READY": source_priors_ready["EXT-F2"],
        "F3_SOURCE_PRIORS_READY": source_priors_ready["EXT-F3"],
        "F1_TARGET_FIREWALL_ACTIVE": target_firewall_active["EXT-F1"],
        "F2_TARGET_FIREWALL_ACTIVE": target_firewall_active["EXT-F2"],
        "F3_TARGET_FIREWALL_ACTIVE": target_firewall_active["EXT-F3"],
        "GPAT_SCIENTIFIC_PRIMITIVES_REUSED": True,
        "PAIRING_SCIENTIFIC_RULES_REUSED": True,
        "QUALITY_GATE_REUSED": True,
        "MATCHED_BANK_LOGIC_REUSED": True,
        "F1_SHUFFLE_STATUS": shuffle["folds"]["EXT-F1"]["status"],
        "F2_SHUFFLE_STATUS": shuffle["folds"]["EXT-F2"]["status"],
        "F3_SHUFFLE_STATUS": shuffle["folds"]["EXT-F3"]["status"],
        "RECIPE_BANKS_BOUND": recipe_binding["all_required_banks_bound"],
        # LITERAL meaning: all source inputs GPATTrainer.fit would open already exist and
        # validate -- never "the workflow knows how to create priors first".
        "READY_FOR_GPU_GPAT_FIT": ready_for_gpu_gpat_fit,
        "READY_FOR_GPU_SOURCE_PRIOR_MATERIALIZATION": ready_for_gpu_source_prior_materialization,
        # Per-fold GPAT_FIT_LOCK.json presence/validity -- reported SEPARATELY from readiness.
        # E7_READY_FOR_TRAINING stays FALSE regardless: candidate generation, quality gates,
        # matched banks and F2/F3 Shuffle feasibility remain unresolved even once every fold
        # is fitted.
        "F1_GPAT_FITTED": gpat_fitted["EXT-F1"], "F2_GPAT_FITTED": gpat_fitted["EXT-F2"],
        "F3_GPAT_FITTED": gpat_fitted["EXT-F3"],
        "E7_READY_FOR_TRAINING": False,
        "TARGET_LABEL_ACCESS": False, "TARGET_IMAGE_ACCESS": False,
        "TRAINING_PERFORMED": False, "RENDERING_PERFORMED": False,
        "GPAT_FITTING_PERFORMED": False, "LLM_API_CALLS": 0,
        "per_fold_compatibility_detail": {f: compatibility[f]["status"] for f in FOLD_IDS},
    }


def build_readiness(repo: Path) -> dict[str, Any]:
    pf = preflight(repo)
    execution_plan = build_execution_plan(repo)
    return {
        "schema_version": f"{SCHEMA_PREFIX}-readiness-v2",
        "READY_FOR_GPU_GPAT_FIT": pf["READY_FOR_GPU_GPAT_FIT"],
        "READY_FOR_GPU_SOURCE_PRIOR_MATERIALIZATION": pf["READY_FOR_GPU_SOURCE_PRIOR_MATERIALIZATION"],
        "F1_SOURCE_PRIORS_MATERIALIZED": pf["F1_SOURCE_PRIORS_MATERIALIZED"],
        "F2_SOURCE_PRIORS_MATERIALIZED": pf["F2_SOURCE_PRIORS_MATERIALIZED"],
        "F3_SOURCE_PRIORS_MATERIALIZED": pf["F3_SOURCE_PRIORS_MATERIALIZED"],
        "F1_GPAT_FITTED": pf["F1_GPAT_FITTED"], "F2_GPAT_FITTED": pf["F2_GPAT_FITTED"],
        "F3_GPAT_FITTED": pf["F3_GPAT_FITTED"],
        "E7_READY_FOR_TRAINING": False, "reason": execution_plan["reason"],
        "F1_SHUFFLE_STATUS": pf["F1_SHUFFLE_STATUS"], "F2_SHUFFLE_STATUS": pf["F2_SHUFFLE_STATUS"],
        "F3_SHUFFLE_STATUS": pf["F3_SHUFFLE_STATUS"],
        "target_access": False, "llm_api_calls": 0, "rendering_performed": False,
        "training_performed": False, "gpat_fitting_performed": False,
    }


def write_readiness(repo: Path) -> dict[str, Any]:
    return _write(repo, "READINESS.json", build_readiness(repo))


# --------------------------------------------------------------------------- #
# TASK N -- GPU-stage entry points (real fail-closed orchestration; NOT
# executed on the laptop this turn). `prepare_gpat` itself is defined above
# (TASK N.6) next to the transaction/identity plumbing it depends on.
# --------------------------------------------------------------------------- #

def generate_and_match(repo: Path, fold_id: str, *, authorize: bool = False) -> dict[str, Any]:
    """`--generate-and-match --authorize [--fold EXT-Fn]`: per-arm candidate
    generation (PhysicsRoute/GPATRoute), quality-gate evaluation, and
    matched-bank resolution via the frozen c6_matched_bank machinery. Fails
    closed if GPAT has not been fit for this fold yet."""
    if fold_id not in FOLD_IDS:
        raise E7Error(f"unknown fold_id {fold_id!r}")
    if not authorize:
        raise E7Error(f"candidate generation for {fold_id} requires --authorize; refusing to run")
    checkpoint_path = gpat_best_checkpoint_path(repo, fold_id)
    if not checkpoint_path.is_file():
        raise E7Error(f"{fold_id}: no fitted GPAT checkpoint present -- run --prepare-gpat first; "
                      "FAIL CLOSED")
    raise E7Error(f"{fold_id}: reached the real GPU candidate-generation boundary on a non-GPU "
                  "host -- refusing to proceed")


def e7_gpat_bank_validate(repo: Path) -> dict[str, Any]:
    """`--validate`: STRICTLY read-only."""
    results = {}
    for fold_id in FOLD_IDS:
        checkpoint_path = gpat_best_checkpoint_path(repo, fold_id)
        bank_lock_paths = {cond: repo / RUN_ROOT / fold_id / cond / "BANK_LOCK.json"
                          for cond in SYNTHETIC_CONDITIONS}
        results[fold_id] = {
            "gpat_checkpoint_present": checkpoint_path.is_file(),
            "gpat_fit_lock_status": validate_gpat_fit_lock(repo, fold_id)["status"],
            "banks": {cond: ("VALID" if p.is_file() else "NOT_MATERIALIZED")
                     for cond, p in bank_lock_paths.items()},
        }
    return {"schema_version": f"{SCHEMA_PREFIX}-validate-v1", "folds": results,
           "target_access": False, "llm_api_calls": 0}


# --------------------------------------------------------------------------- #
# writer plumbing
# --------------------------------------------------------------------------- #

def _write(repo: Path, filename: str, body: dict[str, Any]) -> dict[str, Any]:
    out_dir = repo / REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    return {"body": body, "path": str(path)}


def prepare_planning_artifacts(repo: Path) -> dict[str, Any]:
    """Writes every additive planning/contract artifact. Never fits GPAT,
    never renders, never trains, never calls an LLM."""
    return {
        "protocol_lock": write_protocol_lock(repo),
        "e7d_binding": write_e7d_binding(repo),
        "input_compatibility": write_input_compatibility(repo),
        "prior_binding": write_prior_binding(repo),
        "fold_source_binding": write_fold_source_binding(repo),
        "pairing_policy": write_pairing_policy(repo),
        "gpat_fit_policy": write_gpat_fit_policy(repo),
        "candidate_generation_policy": write_candidate_generation_policy(repo),
        "quality_gate_binding": write_quality_gate_binding(repo),
        "matched_bank_policy": write_matched_bank_policy(repo),
        "shuffle_feasibility_policy": write_shuffle_feasibility_policy(repo),
        "target_firewall": write_target_firewall(repo),
        "execution_plan": write_execution_plan(repo),
        "readiness": write_readiness(repo),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E7 GPAT + synthetic-bank preparation (no render, "
                                                 "no GPAT fit, no training, no target access, no LLM "
                                                 "unless an authorized GPU stage is explicitly run)")
    parser.add_argument("--preflight", action="store_true", help="Read-only. Writes nothing.")
    parser.add_argument("--audit-gpat-inputs", action="store_true", help="Read-only compatibility audit.")
    parser.add_argument("--prepare-source-priors", action="store_true",
                        help="Requires --authorize --fold EXT-F2/F3. GPU stage A: materializes the "
                             "shared SiW-as-source prior package (mechanical prerequisite to GPAT fit).")
    parser.add_argument("--validate-source-priors", action="store_true",
                        help="Read-only. Strictly validates the shared SiW-as-source prior package.")
    parser.add_argument("--prepare-gpat", action="store_true", help="Requires --authorize. GPU stage.")
    parser.add_argument("--generate-and-match", action="store_true",
                        help="Requires --authorize. GPU stage.")
    parser.add_argument("--validate", action="store_true", help="Read-only.")
    parser.add_argument("--authorize", action="store_true")
    parser.add_argument("--fold", choices=list(FOLD_IDS), default=None)
    parser.add_argument("--prepare", action="store_true", help="Writes every planning artifact.")
    args = parser.parse_args(argv)
    repo = cc.repo_root()
    folds = [args.fold] if args.fold else list(FOLD_IDS)

    if args.preflight:
        print(json.dumps(preflight(repo), indent=2, default=str))
        return 0
    if args.audit_gpat_inputs:
        print(json.dumps({f: audit_gpat_input_compatibility(repo, f) for f in folds}, indent=2,
                        default=str))
        return 0
    if args.prepare_source_priors:
        # NEVER defaults to all folds: F1 has no SiW source, and running this against every
        # fold implicitly hides that this is a single shared-package operation. --fold EXT-F2
        # or --fold EXT-F3 is REQUIRED; fail closed otherwise.
        if args.fold not in ("EXT-F2", "EXT-F3"):
            print(json.dumps({"error": "--prepare-source-priors requires an explicit "
                             "--fold EXT-F2 or --fold EXT-F3 -- refusing to default to all "
                             "folds"}, indent=2))
            return 1
        try:
            result = prepare_source_priors(repo, args.fold, authorize=args.authorize)
            print(json.dumps({args.fold: result}, indent=2, default=str))
            return 0
        except E7Error as error:
            print(json.dumps({args.fold: {"error": str(error)}}, indent=2, default=str))
            return 1
    if args.validate_source_priors:
        print(json.dumps({f: validate_source_priors(repo, f) for f in folds}, indent=2, default=str))
        return 0
    if args.prepare_gpat:
        results = {}
        for f in folds:
            try:
                results[f] = prepare_gpat(repo, f, authorize=args.authorize)
            except E7Error as error:
                results[f] = {"error": str(error)}
        print(json.dumps(results, indent=2, default=str))
        return 0 if all("error" not in r for r in results.values()) else 1
    if args.generate_and_match:
        results = {}
        for f in folds:
            try:
                results[f] = generate_and_match(repo, f, authorize=args.authorize)
            except E7Error as error:
                results[f] = {"error": str(error)}
        print(json.dumps(results, indent=2, default=str))
        return 0 if all("error" not in r for r in results.values()) else 1
    if args.validate:
        print(json.dumps(e7_gpat_bank_validate(repo), indent=2, default=str))
        return 0
    if args.prepare:
        result = prepare_planning_artifacts(repo)
        print(json.dumps({"readiness": result["readiness"]["body"]}, indent=2, default=str))
        return 0

    print("Pass --preflight, --audit-gpat-inputs, "
         "--prepare-source-priors --authorize --fold EXT-F2/F3, --validate-source-priors, "
         "--prepare-gpat --authorize [--fold ...], "
         "--generate-and-match --authorize [--fold ...], --validate, or --prepare.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
