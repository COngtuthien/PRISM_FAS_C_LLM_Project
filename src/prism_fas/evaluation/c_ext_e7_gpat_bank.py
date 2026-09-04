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
import json
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


def validate_source_priors(repo: Path, fold_id: str) -> dict[str, Any]:
    """STRICT, read-only validator for the shared SiW-as-source prior
    package. F2 and F3 both validate the exact SAME on-disk package. Never
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

    problems: list[str] = []
    body = cc.read_json(package_path)
    rows = body.get("rows", [])
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
    identity_match = recomputed_identity == body.get("package_identity")
    if not identity_match:
        problems.append(f"recomputed package_identity {recomputed_identity!r} != recorded "
                        f"{body.get('package_identity')!r}")

    return {
        "schema_version": f"{SCHEMA_PREFIX}-source-priors-validate-v1", "fold_id": fold_id,
        "status": "INVALID" if problems else "VALID", "problems": problems,
        "row_count": len(rows), "expected_row_count": EXPECTED_SIW_SUCCESS_CROP_COUNT,
        "crop_rows_verified": crop_rows_verified, "missing_crops": missing_crops,
        "bad_crop_hashes": bad_crop_hashes, "missing_prior_files": missing_priors,
        "bad_prior_hashes": bad_prior_hashes, "package_identity": body.get("package_identity"),
        "recomputed_package_identity": recomputed_identity, "package_identity_match": identity_match,
        "target_access": False, "llm_api_calls": 0,
    }


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


def compute_m3a_input_package_identity(fold_id: str, *, e7d_package_identity: str | None,
                                       rows: list[dict[str, Any]]) -> str:
    """Deterministic identity over CANONICAL METADATA only -- no absolute
    paths, no timestamps, no hostname. Binds E7-B SiW source package
    identity, E7-D source support / E7-A split identity, the frozen M3A
    package config identity, and the sorted per-row (source_video_id,
    frame_index, project_split, crop_sha256, preprocessing_config_hash,
    detector_model_sha256) material."""
    row_material = sorted(
        (r.get("source_video_id"), r.get("frame_index"), r.get("project_split"),
         r.get("crop_sha256"), r.get("preprocessing_config_hash"), r.get("detector_model_sha256"))
        for r in rows)
    material = {
        "e7b_siw_source_package_identity": e7c.FROZEN_E7B["siw_source_package_identity"],
        "e7d_source_support_package_identity": e7d_package_identity,
        "e7a_siw_split_identity": e7c.FROZEN_E7B["siw_split_identity"],
        "m3a_input_package_id": SIW_SOURCE_M3A_INPUT_PACKAGE_ID,
        "m3a_package_config_path": M3A_PACKAGE_CONFIG_PATH,
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

    siw_binding_path = repo / e7d.E7D_OUTPUT_ROOT / fold_id / "SOURCE_SUPPORT_PACKAGE.json"
    e7d_package_identity = (cc.read_json(siw_binding_path).get("package_identity")
                            if siw_binding_path.is_file() else None)

    train_e7d, dev_e7d = _load_e7d_authoritative_siw_rows(repo, fold_id)
    if len(train_e7d) != EXPECTED_SIW_TRAIN_SUCCESS_COUNT or len(dev_e7d) != EXPECTED_SIW_DEV_SUCCESS_COUNT:
        raise E7Error(f"{fold_id}: E7-D authoritative SiW row counts train={len(train_e7d)} "
                      f"dev={len(dev_e7d)} != expected train={EXPECTED_SIW_TRAIN_SUCCESS_COUNT} "
                      f"dev={EXPECTED_SIW_DEV_SUCCESS_COUNT} -- FAIL CLOSED")
    crop_index = _load_siw_m2_crop_index(repo)

    samples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    m2_root = repo / e7b.E7B_SIW_SOURCE_PACKAGE_ROOT / "m2_run"
    for project_split, e7d_rows in (("source_train", train_e7d), ("source_dev", dev_e7d)):
        for row in e7d_rows:
            key = (row.get("source_video_id"), row.get("frame_index"))
            crop = crop_index.get(key)
            if crop is None:
                raise E7Error(f"{fold_id}: SiW crop for {key!r} not present in E7-B m2_run crop "
                              "manifest -- FAIL CLOSED")
            if crop.get("crop_relative_path") != row.get("crop_relative_path") or \
                    crop.get("crop_sha256") != row.get("crop_sha256"):
                raise E7Error(f"{fold_id}: E7-D authoritative row for {key!r} disagrees with the "
                              "real M2 crop manifest -- FAIL CLOSED")
            candidate_ref = f"{e7b.E7B_SIW_SOURCE_PACKAGE_ROOT}/m2_run/{crop['crop_relative_path']}"
            assert_not_target_path(fold_id, candidate_ref)
            crop_path = m2_root / crop["crop_relative_path"]
            if not crop_path.is_file():
                raise E7Error(f"{fold_id}: SiW crop missing on disk: "
                              f"{crop['crop_relative_path']!r} -- FAIL CLOSED")
            if cc.sha256_file(crop_path) != crop["crop_sha256"]:
                raise E7Error(f"{fold_id}: SiW crop SHA256 mismatch on disk: "
                              f"{crop['crop_relative_path']!r} -- FAIL CLOSED")
            sample_id = _m3a_sample_id(*key)
            if sample_id in seen_ids:
                raise E7Error(f"{fold_id}: duplicate derived sample_id for {key!r} -- FAIL CLOSED")
            seen_ids.add(sample_id)
            samples.append({**crop, "sample_id": sample_id, "dataset": "siw_mv2",
                            "dataset_role": "source",
                            "official_split": e7b.SIW_SOURCE_OFFICIAL_SPLIT_PLACEHOLDER,
                            "subject_id": None, "label_live_spoof": row.get("label_live_spoof"),
                            "_e7_project_split": project_split,
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
                             "crop_relative_path": sample["crop_relative_path"],
                             "crop_sha256": sample["crop_sha256"],
                             "preprocessing_config_hash": sample["preprocessing_config_hash"],
                             "detector_model_sha256": sample["detector_model_sha256"],
                             "project_split": row["project_split"]})

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
        fold_id, e7d_package_identity=e7d_package_identity, rows=binding_rows)
    lock = cc.read_json(lock_path)
    atomic_json_write(package_root / M3A_INPUT_BINDING_FILENAME, {
        "schema_version": f"{SCHEMA_PREFIX}-m3a-input-binding-v1", "fold_id": fold_id,
        "m3a_input_package_identity": m3a_input_package_identity,
        "e7b_siw_source_package_identity": e7c.FROZEN_E7B["siw_source_package_identity"],
        "e7d_source_support_package_identity": e7d_package_identity,
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

    recomputed_identity = compute_m3a_input_package_identity(
        fold_id, e7d_package_identity=binding.get("e7d_source_support_package_identity"), rows=rows)
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
    atomic_json_write(package_path, {"schema_version": "siw-source-prior-package-v1",
                                     "package_identity": package_identity, "rows": rows})
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
# executed on the laptop this turn)
# --------------------------------------------------------------------------- #

def prepare_gpat(repo: Path, fold_id: str, *, authorize: bool = False) -> dict[str, Any]:
    """`--prepare-gpat --authorize [--fold EXT-Fn]`: materializes the
    fold-aware SampleStore-compatible package_root (crops+priors joined
    from E7-D+E7-A / freshly-generated SiW priors) and runs
    GPATTrainer.fit() UNMODIFIED. Fails closed on any unresolved
    compatibility, identity mismatch, or missing input."""
    if fold_id not in FOLD_IDS:
        raise E7Error(f"unknown fold_id {fold_id!r}")
    if not authorize:
        raise E7Error(f"GPAT preparation for {fold_id} requires --authorize; refusing to run")

    pf = preflight(repo)
    if not pf["READY_FOR_GPU_GPAT_FIT"]:
        raise E7Error(f"{fold_id}: preflight not ready for GPU GPAT fit -- FAIL CLOSED: {pf!r}")
    compatibility = audit_gpat_input_compatibility(repo, fold_id)
    if compatibility["status"] not in ("COMPATIBLE", "COMPATIBLE_PENDING_GPU_PRIOR_GENERATION"):
        raise E7Error(f"{fold_id}: E7_GPAT_INPUT_COMPATIBILITY={compatibility['status']} -- "
                      "FAIL CLOSED, refusing to fit GPAT")
    if compatibility["status"] == "COMPATIBLE_PENDING_GPU_PRIOR_GENERATION":
        raise E7Error(f"{fold_id}: SiW-as-source priors have not been generated yet on GPU "
                      "(build_m3b_package against E7-B's own SiW-as-source package) -- "
                      "GPU_REQUIRED, refusing to fit GPAT until priors exist")

    e7d_validation = e7d.validate_fold(repo, fold_id)
    if e7d_validation["status"] != "VALID":
        raise E7Error(f"{fold_id}: E7-D source support package is not VALID -- FAIL CLOSED: "
                      f"{e7d_validation.get('problems')!r}")

    # Real GPU execution (crop/prior parquet materialization + GPATTrainer.fit) is intentionally
    # NOT invoked here -- reaching this point on the laptop this turn is impossible (compatibility
    # is never both COMPATIBLE and locally-materializable without real GPU bytes); documented for
    # the real GPU host: the next call is `_materialize_gpat_package_root(repo, fold_id)` followed
    # by `GPATTrainer(config=..., package_root=..., bank_root=M7_RECIPE_BANK_ROOT, ...).fit(...)`.
    raise E7Error(f"{fold_id}: reached the real GPU GPAT-fit boundary on a non-GPU host -- "
                  "refusing to proceed (this call path is never exercised on the laptop)")


def generate_and_match(repo: Path, fold_id: str, *, authorize: bool = False) -> dict[str, Any]:
    """`--generate-and-match --authorize [--fold EXT-Fn]`: per-arm candidate
    generation (PhysicsRoute/GPATRoute), quality-gate evaluation, and
    matched-bank resolution via the frozen c6_matched_bank machinery. Fails
    closed if GPAT has not been fit for this fold yet."""
    if fold_id not in FOLD_IDS:
        raise E7Error(f"unknown fold_id {fold_id!r}")
    if not authorize:
        raise E7Error(f"candidate generation for {fold_id} requires --authorize; refusing to run")
    checkpoint_path = repo / RUN_ROOT / fold_id / "gpat_checkpoint" / "best.pt"
    if not checkpoint_path.is_file():
        raise E7Error(f"{fold_id}: no fitted GPAT checkpoint present -- run --prepare-gpat first; "
                      "FAIL CLOSED")
    raise E7Error(f"{fold_id}: reached the real GPU candidate-generation boundary on a non-GPU "
                  "host -- refusing to proceed")


def e7_gpat_bank_validate(repo: Path) -> dict[str, Any]:
    """`--validate`: STRICTLY read-only."""
    results = {}
    for fold_id in FOLD_IDS:
        checkpoint_path = repo / RUN_ROOT / fold_id / "gpat_checkpoint" / "best.pt"
        bank_lock_paths = {cond: repo / RUN_ROOT / fold_id / cond / "BANK_LOCK.json"
                          for cond in SYNTHETIC_CONDITIONS}
        results[fold_id] = {
            "gpat_checkpoint_present": checkpoint_path.is_file(),
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
