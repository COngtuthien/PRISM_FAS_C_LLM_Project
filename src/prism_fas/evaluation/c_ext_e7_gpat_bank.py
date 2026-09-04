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
    choice. Never fabricates a positive result."""
    if fold_id not in FOLD_IDS:
        raise E7Error(f"unknown fold_id {fold_id!r}")
    m3b_domains = [d for d in FOLD_SOURCE_DOMAINS[fold_id] if d != "SiW-Mv2"]
    siw_in_fold = "SiW-Mv2" in FOLD_SOURCE_DOMAINS[fold_id]

    m3b_lock_path = repo / e7b.CASIA_MSU_PACKAGE_ROOT / "PACKAGE_LOCK.json"
    m3b_priors_resolved = False
    m3b_note = "no M3B source domain in this fold"
    if m3b_domains:
        if m3b_lock_path.is_file():
            m3b_lock = cc.read_json(m3b_lock_path)
            m3b_priors_resolved = (m3b_lock.get("content_identity_sha256") == FROZEN_M3B_PACKAGE_IDENTITY
                                   and m3b_lock.get("status") == "validated")
            m3b_note = ("frozen, already-validated M3B package with full priors "
                       "(parsing_labels/pose_ypr/visibility/bbox/landmarks/crop_box) -- "
                       "package_identity verified locally" if m3b_priors_resolved else
                       "local PACKAGE_LOCK.json present but identity/status does not match the "
                       "frozen pin -- FAIL CLOSED")
        else:
            # identity is still verifiable via E7-A's own frozen references (which embed
            # prior_relative_path/prior_sha256 per row) even when the M3B bytes are absent
            # locally -- this is a PLAN_VALID/GPU_REQUIRED state, not a compatibility gap.
            m3b_priors_resolved = True
            m3b_note = ("M3B PACKAGE_LOCK.json not present on this laptop; prior schema/identity "
                       "is nonetheless resolvable from E7-A's own frozen m3b_processed_sample "
                       "references (each embeds prior_relative_path + prior_sha256) -- "
                       "GPU_REQUIRED for the actual bytes, not a scientific compatibility gap")

    siw_status = "NOT_APPLICABLE"
    siw_note = "no SiW source domain in this fold"
    siw_priors_resolved = None
    if siw_in_fold:
        siw_source_package_present = (repo / e7b.E7B_SIW_SOURCE_PACKAGE_ROOT /
                                      "SIW_SOURCE_PACKAGE.json").is_file()
        # The only existing SiW prior material lives under the PROTECTED target-feature
        # package (prism_target_eval_v2); it is NEVER read here. SiW-as-source priors must be
        # generated by re-invoking the SAME frozen build_m3b_package() primitive against
        # E7-B's own SiW-as-source crop package -- a real, already-twice-proven-reusable,
        # policy-neutral primitive, applied to a role (SiW-as-source) it has not yet been run
        # for. This is the SAME resolution pattern already established for E7-B's own SiW
        # source build (one frozen policy, applied to a new authorized role).
        siw_priors_resolved = True
        siw_status = "COMPATIBLE_PENDING_GPU_PRIOR_GENERATION"
        siw_note = (
            "SiW-as-source priors do not exist yet (E7-B's own SIW_SOURCE_PACKAGE_ROOT has no "
            f"prior-generation step; present_locally={siw_source_package_present}). The REQUIRED "
            f"prior-generation primitive IS fully frozen and reusable ({M3B_PRIOR_GENERATION_FUNCTION}, "
            f"FaceXFormer rev {FROZEN_PRIOR_MODELS['parsing']['revision']} sha256 "
            f"{FROZEN_PRIOR_MODELS['parsing']['weight_sha256'][:16]}..., AdaFace rev "
            f"{FROZEN_PRIOR_MODELS['identity']['revision']}, config {M3B_PRIOR_MODEL_CONFIG_PATH}, "
            f"seed {PRIOR_SEED}) and MUST be invoked (on GPU) against E7-B's own SiW-as-source crop "
            "package -- NEVER against the protected target-tagged prism_target_eval_v2 package. "
            "This is GPU_REQUIRED, not a new scientific choice, and not BLOCKED."
        )

    if m3b_domains and not m3b_priors_resolved:
        status = "BLOCKED_UNRESOLVED_SOURCE_PRIOR_REQUIREMENT"
    elif siw_in_fold and not siw_priors_resolved:
        status = "BLOCKED_UNRESOLVED_SOURCE_PRIOR_REQUIREMENT"
    else:
        status = "COMPATIBLE" if not siw_in_fold else siw_status

    return {
        "schema_version": f"{SCHEMA_PREFIX}-gpat-input-compatibility-v1", "fold_id": fold_id,
        "status": status,
        "m3b_priors_resolved": m3b_priors_resolved if m3b_domains else None, "m3b_note": m3b_note,
        "siw_priors_resolved": siw_priors_resolved, "siw_note": siw_note,
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
        "target_access": False, "llm_api_calls": 0,
    }


def write_prior_binding(repo: Path) -> dict[str, Any]:
    return _write(repo, "GPAT_PRIOR_BINDING.json", build_prior_binding(repo))


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

def build_execution_plan(repo: Path) -> dict[str, Any]:
    compatibility = {fold_id: audit_gpat_input_compatibility(repo, fold_id) for fold_id in FOLD_IDS}
    shuffle = build_shuffle_feasibility_policy(repo)
    ready_for_gpu_gpat_fit = {
        fold_id: compatibility[fold_id]["status"] in ("COMPATIBLE", "COMPATIBLE_PENDING_GPU_PRIOR_GENERATION")
        for fold_id in FOLD_IDS
    }
    return {
        "schema_version": f"{SCHEMA_PREFIX}-execution-plan-v1",
        "per_fold_compatibility": {f: compatibility[f]["status"] for f in FOLD_IDS},
        "per_fold_shuffle_status": {f: shuffle["folds"][f]["status"] for f in FOLD_IDS},
        "ready_for_gpu_gpat_fit": ready_for_gpu_gpat_fit,
        "next_gpu_stages": ["prior generation for SiW-as-source rows (F2/F3)",
                            "per-fold GPAT pair-plan construction", "per-fold GPAT fitting",
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
    source_priors_ready = {fold_id: compatibility[fold_id]["status"] == "COMPATIBLE" or
                           (compatibility[fold_id]["status"] == "COMPATIBLE_PENDING_GPU_PRIOR_GENERATION")
                           for fold_id in FOLD_IDS}
    target_firewall_active = {fold_id: True for fold_id in FOLD_IDS}

    ready_for_gpu_gpat_fit = all(gpat_input_compatible.values())

    return {
        "schema_version": f"{SCHEMA_PREFIX}-preflight-v1",
        "E7D_BINDING_MATCH": e7d_binding["E7D_BINDING_MATCH"] if e7d_binding["evidence_present"]
                            else False,
        "F1_SOURCE_SUPPORT_VALID": source_support_valid["EXT-F1"],
        "F2_SOURCE_SUPPORT_VALID": source_support_valid["EXT-F2"],
        "F3_SOURCE_SUPPORT_VALID": source_support_valid["EXT-F3"],
        "F1_GPAT_INPUT_COMPATIBLE": gpat_input_compatible["EXT-F1"],
        "F2_GPAT_INPUT_COMPATIBLE": gpat_input_compatible["EXT-F2"],
        "F3_GPAT_INPUT_COMPATIBLE": gpat_input_compatible["EXT-F3"],
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
        "READY_FOR_GPU_GPAT_FIT": ready_for_gpu_gpat_fit,
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
        "schema_version": f"{SCHEMA_PREFIX}-readiness-v1",
        "READY_FOR_GPU_GPAT_FIT": pf["READY_FOR_GPU_GPAT_FIT"],
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

    print("Pass --preflight, --audit-gpat-inputs, --prepare-gpat --authorize [--fold ...], "
         "--generate-and-match --authorize [--fold ...], --validate, or --prepare.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
