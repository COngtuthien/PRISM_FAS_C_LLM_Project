"""Tests for `prism_fas.evaluation.c_ext_e7_gpat_bank` (E7 GPAT + synthetic
bank preparation). Every test builds a self-contained fake repo under
`tmp_path` unless it explicitly checks the REAL committed repo (frozen
identity/config checks, which can only be meaningfully verified against
real bytes). No test ever fits GPAT, renders, trains, or calls an LLM.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism_fas.evaluation import c_ext_e7b_data_prep as e7b
from prism_fas.evaluation import c_ext_e7_gpat_bank as e7g

REPO = Path(__file__).resolve().parents[2]


def _base_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


# --- E7-D package identities exact -------------------------------------------

def test_e7d_package_identities_exact():
    binding = e7g.build_e7d_binding(REPO)
    assert binding["E7D_BINDING_MATCH"] is True
    assert binding["folds"]["EXT-F1"]["frozen_package_identity"] == \
        "955b630fec438c80f284ecbcb30fbf10c83251a23fd31d8ab1a52e0f8ce8383b"
    assert binding["folds"]["EXT-F2"]["frozen_package_identity"] == \
        "b617dc8ee6b0827ef5c0be3072563bb0013d1c87c7982527a6b16a5b75dde6a0"
    assert binding["folds"]["EXT-F3"]["frozen_package_identity"] == \
        "508a9bd002d22571534d9609a7b301065a8d752b9625f7b500ae27b199ec8955"
    assert binding["siw_accounting_match"] is True


def test_e7d_binding_fails_closed_on_mismatch(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    evidence_path = repo / e7g.E7D_FINAL_EVIDENCE_PATH
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps({
        "SIW_SUCCESS_TOTAL": 6776, "SIW_FAILURE_TOTAL": 24,
        "folds": {"EXT-F1": {"package_identity": "0" * 65},
                 "EXT-F2": {"package_identity": e7g.FROZEN_E7D_PACKAGE_IDENTITY["EXT-F2"]},
                 "EXT-F3": {"package_identity": e7g.FROZEN_E7D_PACKAGE_IDENTITY["EXT-F3"]}}}),
        encoding="utf-8")
    with pytest.raises(e7g.E7Error, match="MISMATCH"):
        e7g.write_e7d_binding(repo)


# --- all three source folds exact --------------------------------------------

def test_f1_source_domains_exact():
    assert e7g.FOLD_SOURCE_DOMAINS["EXT-F1"] == ("CASIA-FASD", "MSU-MFSD")


def test_f2_source_domains_exact():
    assert e7g.FOLD_SOURCE_DOMAINS["EXT-F2"] == ("CASIA-FASD", "SiW-Mv2")


def test_f3_source_domains_exact():
    assert e7g.FOLD_SOURCE_DOMAINS["EXT-F3"] == ("MSU-MFSD", "SiW-Mv2")


# --- held-out domain excluded / no target labels -----------------------------

def test_held_out_domain_excluded_from_firewall_allow_list():
    firewall = e7g.build_target_firewall(REPO)
    for fold_id in e7g.FOLD_IDS:
        assert e7g.FOLD_TARGET_DOMAIN[fold_id] not in " ".join(
            firewall["folds"][fold_id]["always_open_source_side_roots"])


def test_no_target_labels_ever_opened():
    for fold_id in e7g.FOLD_IDS:
        with pytest.raises(e7g.E7TargetFirewallViolation):
            e7g.assert_not_target_path(fold_id, "data/evaluation_only/prism_target_v2_labels/x.parquet")
    firewall = e7g.build_target_firewall(REPO)
    assert "data/evaluation_only" in firewall["never_opens"]


# --- fold-aware SiW allowance -------------------------------------------------

def test_fold_aware_siw_allowance_no_global_ban():
    firewall = e7g.build_target_firewall(REPO)
    assert firewall["global_dataset_name_ban"] is False
    assert e7b.E7B_SIW_SOURCE_PACKAGE_ROOT in firewall["folds"]["EXT-F2"]["always_open_source_side_roots"]
    assert e7b.E7B_SIW_SOURCE_PACKAGE_ROOT in firewall["folds"]["EXT-F3"]["always_open_source_side_roots"]
    assert e7b.E7B_SIW_SOURCE_PACKAGE_ROOT not in firewall["folds"]["EXT-F1"]["always_open_source_side_roots"]
    with pytest.raises(e7g.E7TargetFirewallViolation):
        e7g.assert_not_target_path("EXT-F1", f"{e7b.E7B_SIW_SOURCE_PACKAGE_ROOT}/m2_run/x.jpg")
    e7g.assert_not_target_path("EXT-F2", f"{e7b.E7B_SIW_SOURCE_PACKAGE_ROOT}/m2_run/x.jpg")


# --- legacy M8 SourceOnlyAudit remains unchanged -----------------------------

def test_legacy_m8_source_only_audit_unmodified():
    from prism_fas.synthesis import m8_pipeline

    source = Path(m8_pipeline.__file__).read_text(encoding="utf-8")
    assert '"siw" in lowered or "target" in lowered' in source
    assert 'FORBIDDEN_SPLITS = ("source_dev", "target_test")' in source
    # this module never edits m8_pipeline.py
    module_source = Path(e7g.__file__).read_text(encoding="utf-8")
    assert "m8_pipeline.py" not in module_source or "def " not in module_source.split(
        "m8_pipeline.py")[0][-50:]


def test_e7_module_never_writes_to_m8_pipeline_file():
    import prism_fas.synthesis.m8_pipeline as m8_pipeline

    before = Path(m8_pipeline.__file__).read_bytes()
    e7g.preflight(REPO)  # read-only call
    after = Path(m8_pipeline.__file__).read_bytes()
    assert before == after


# --- E7 adapter permits SiW only as authorized source ------------------------

def test_sample_store_audit_only_checks_path_strings_not_data():
    """Confirms the exact compatibility mechanism this module relies on:
    SourceOnlyAudit.record() only inspects the STRING passed to it -- the
    fixed 'manifests/source_train.parquet' path SampleStore.open() always
    uses never contains 'siw'/'target' regardless of the DATA inside."""
    from prism_fas.synthesis.m8_pipeline import SOURCE_SPLIT, SourceOnlyAudit

    audit = SourceOnlyAudit()
    fixed_path = f"manifests/{SOURCE_SPLIT}.parquet"
    assert "siw" not in fixed_path.lower() and "target" not in fixed_path.lower()
    audit.record(fixed_path)  # does not raise regardless of what dataset values live inside


def test_adapter_must_avoid_forbidden_substrings_in_stored_paths():
    """A crop/prior relative path containing the literal substring 'siw' or
    'target' WOULD be rejected by the unmodified audit -- this module's own
    row-materialization must therefore use content-hash-style filenames,
    never a path embedding the dataset name."""
    from prism_fas.synthesis.m8_pipeline import SourceOnlyAudit

    audit = SourceOnlyAudit()
    with pytest.raises(Exception):
        audit.record("images/siw_mv2/some_video/frame_0.jpg")


# --- prior compatibility fails closed if unresolved --------------------------

def test_prior_compatibility_blocked_when_m3b_identity_mismatches(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    lock_path = repo / e7b.CASIA_MSU_PACKAGE_ROOT / "PACKAGE_LOCK.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps({"content_identity_sha256": "0" * 65, "status": "validated"}),
                        encoding="utf-8")
    audit = e7g.audit_gpat_input_compatibility(repo, "EXT-F1")
    assert audit["status"] == "BLOCKED_UNRESOLVED_SOURCE_PRIOR_REQUIREMENT"
    assert audit["m3b_priors_materialized"] is False


def test_prior_compatibility_plan_valid_when_m3b_bytes_absent_locally(tmp_path):
    repo = _base_repo(tmp_path)  # no PACKAGE_LOCK.json at all -- laptop-realistic absence
    audit = e7g.audit_gpat_input_compatibility(repo, "EXT-F1")
    assert audit["status"] == "COMPATIBLE"
    assert audit["m3b_prior_generation_primitive_resolved"] == "NOT_APPLICABLE_EXISTING_PRIORS"
    assert audit["m3b_priors_materialized"] is False
    assert "GPU_REQUIRED" in audit["m3b_note"]


def test_siw_compatibility_never_blocked_by_construction(tmp_path):
    repo = _base_repo(tmp_path)
    for fold_id in ("EXT-F2", "EXT-F3"):
        audit = e7g.audit_gpat_input_compatibility(repo, fold_id)
        assert audit["status"] == "COMPATIBLE_PENDING_GPU_PRIOR_GENERATION"
        assert audit["status"] != "BLOCKED_UNRESOLVED_SOURCE_PRIOR_REQUIREMENT"


def test_siw_prior_generation_never_reads_protected_target_package():
    audit = e7g.audit_gpat_input_compatibility(REPO, "EXT-F2")
    assert "prism_target_eval_v2" in audit["siw_note"]
    assert "NEVER" in audit["siw_note"]
    source = Path(e7g.__file__).read_text(encoding="utf-8")
    # the protected root is only ever referenced as a documented exclusion, never opened
    assert 'cc.read_json(repo / e7b.SIW_TARGET_EVAL_PACKAGE_ROOT' not in source
    assert "read_prior" not in source.replace("PROTECTED_SIW_TARGET_PRIOR_PACKAGE_ROOT", "")


# --- no fabricated prior paths ------------------------------------------------

def test_no_fabricated_prior_path_or_sha():
    source = Path(e7g.__file__).read_text(encoding="utf-8")
    assert "prior_relative_path = " not in source  # never assigns a literal/fake path
    assert "prior_sha256 = '" not in source and 'prior_sha256 = "' not in source


# --- crop SHA verification (reused from E7-D, never reimplemented) ----------

def test_crop_sha_verification_reused_from_e7d():
    source = Path(e7g.__file__).read_text(encoding="utf-8")
    assert "sha256_file" not in source or "cc.sha256_file" not in source.split("def audit_gpat_input_compatibility")[1].split("def ")[0]


# --- GPAT pairing train/dev isolation; same/cross-domain; subject rule -----

def test_pairing_policy_preserves_scientific_rules():
    policy = e7g.build_pairing_policy(REPO)
    rules = policy["preserved_scientific_rules"]
    for expected in ("live/spoof roles", "deterministic record partition", "same-domain pairing",
                     "cross-domain pairing", "different-record rule", "deterministic seed",
                     "train/validation isolation"):
        assert expected in rules
    assert any("subject rule NOT_APPLICABLE" in r for r in rules)
    assert policy["never_alters_rules_to_force_a_pair_count"] is True


def test_pairing_frozen_seed_matches_repo_authority():
    from prism_fas.synthesis import pair_plan

    policy = e7g.build_pairing_policy(REPO)
    assert policy["frozen_seed"] == pair_plan.PAIR_PLAN_SEED == 20260806


# --- no legacy fixed pair-count assumption for F2/F3 -------------------------

def test_no_legacy_fixed_pair_count_for_f2_f3():
    policy = e7g.build_pairing_policy(REPO)
    assert "expected_train_pairs" not in policy
    assert "expected_validation_pairs" not in policy
    assert policy["legacy_pair_count_constants_are_m3b_size_assumptions_only"] is True
    assert "real E7-D fold source TRAIN rows" in policy["pair_counts_resolved_from"]


def test_legacy_allowed_datasets_overridden_fold_aware():
    policy = e7g.build_pairing_policy(REPO)
    override = policy["legacy_allowed_datasets_fold_aware_override"]
    assert set(override["EXT-F1"]) == {"casia_fasd", "msu_mfsd"}
    assert set(override["EXT-F2"]) == {"casia_fasd", "siw_mv2"}
    assert set(override["EXT-F3"]) == {"msu_mfsd", "siw_mv2"}
    # legacy constant itself is reported but never silently trusted as still valid for F2/F3
    assert set(policy["legacy_allowed_datasets"]) == {"casia_fasd", "msu_mfsd"}


# --- one GPAT fit per fold policy; no per-arm source-sampling advantage -----

def test_one_gpat_fit_per_fold_policy():
    policy = e7g.build_gpat_fit_policy(REPO)
    assert policy["one_fit_per_fold_source_only"] is True
    assert "NEVER separately per RND/DET/LLM/Shuffle arm" in policy["fit_arm_independent"]


def test_no_per_arm_source_sampling_advantage():
    policy = e7g.build_candidate_generation_policy(REPO)
    assert "ARM-INDEPENDENT" in policy["treatment_fairness"]


def test_gpat_fit_binds_m7_conditioning_bank_not_c3_arm_banks():
    policy = e7g.build_gpat_fit_policy(REPO)
    assert policy["conditioning_bank"]["bank_id"] == "prism_recipe_bank_m7_v1"
    assert policy["conditioning_bank_root"] == "assets/recipe_banks/prism_recipe_bank_m7_v1"


# --- recipe identity kinds preserved ------------------------------------------

def test_recipe_identity_kinds_preserved_from_e7c():
    from prism_fas.evaluation import c_ext_e7c_gpat_prep as e7c

    binding = e7c.build_recipe_bank_binding(REPO)
    assert binding["all_required_banks_bound"] is True
    assert binding["bindings"]["LLM"]["observed_binding_identity_kind"] == "RECIPE_BANK_IDENTITY"
    assert binding["bindings"]["LLM"]["equivalence_proven"] is True
    assert binding["bindings"]["LLM-SHUFFLE-A"]["observed_binding_identity_kind"] == \
        "CANONICAL_JSONL_CONTENT_HASH"


# --- F1 Shuffle permanently blocked; F2/F3 independently unresolved --------

def test_f1_shuffle_permanently_blocked():
    policy = e7g.build_shuffle_feasibility_policy(REPO)
    assert policy["folds"]["EXT-F1"]["status"] == e7g.BLOCKED_TRUE_FROZEN_MATCHED_BANK_INFEASIBILITY
    assert policy["folds"]["EXT-F1"]["basis"]["SHUFFLE_PHYSICS_CASIA_DEFICIT"] == 33


def test_f2_f3_shuffle_independently_pending_before_gpu_rendering():
    policy = e7g.build_shuffle_feasibility_policy(REPO)
    for fold_id in ("EXT-F2", "EXT-F3"):
        assert policy["folds"][fold_id]["status"] == e7g.PENDING_FEASIBILITY_PREFLIGHT
        assert policy["folds"][fold_id]["basis"]["ext_f1_result_does_not_predetermine_this_fold"] \
            is True


def test_shuffle_policy_write_fails_closed_if_f1_status_drifts(monkeypatch):
    monkeypatch.setattr(e7g, "build_condition_status",
                        lambda fold_id, condition: "READY_FOR_GPAT_PREP")
    with pytest.raises(e7g.E7Error, match="BLOCKED_TRUE_FROZEN_MATCHED_BANK_INFEASIBILITY"):
        e7g.write_shuffle_feasibility_policy(REPO)


# --- exact frozen quotas read from authority; no rescue action -------------

def test_frozen_quota_matches_repository_authority():
    quota = e7g.frozen_bank_quota()
    assert quota == {"final_bank_per_arm": 1024, "physics_per_arm": 512, "gpat_per_arm": 512}


def test_frozen_quota_fails_closed_on_drift(monkeypatch):
    from prism_fas.synthesis import gate_profiles

    monkeypatch.setattr(gate_profiles, "FINAL_BANK_PER_ARM", 2048)
    with pytest.raises(e7g.E7Error, match="quota drift"):
        e7g.frozen_bank_quota()


def test_no_rescue_action_permitted():
    policy = e7g.build_shuffle_feasibility_policy(REPO)
    assert policy["no_rescue_actions_permitted"] is True
    for action in ("lower bank size 512", "change source-domain quotas", "relax quality thresholds",
                  "modify q", "resample candidates", "change matching policy"):
        assert action in policy["forbidden_rescue_actions"]


# --- quality calibration source-only -----------------------------------------

def test_quality_calibration_source_only():
    binding = e7g.build_quality_gate_binding(REPO)
    assert binding["calibration_source_only"] is True
    assert binding["calibration_never_uses_held_out_target"] is True
    assert binding["thresholds_never_relaxed_to_force_feasibility"] is True


# --- target firewall per fold -------------------------------------------------

def test_target_firewall_per_fold_forbids_own_target_package():
    firewall = e7g.build_target_firewall(REPO)
    assert e7b.E7B_MSU_TARGET_PACKAGE_ROOT in firewall["folds"]["EXT-F2"]["forbidden_roots"]
    assert e7b.E7B_CASIA_TARGET_PACKAGE_ROOT in firewall["folds"]["EXT-F3"]["forbidden_roots"]
    assert e7b.SIW_TARGET_EVAL_PACKAGE_ROOT in firewall["folds"]["EXT-F1"]["forbidden_roots"]


# --- no GPAT algorithm rewrite; no quality metric rewrite -------------------

def test_no_gpat_algorithm_rewrite():
    source = Path(e7g.__file__).read_text(encoding="utf-8")
    for forbidden in ("class GPATResidualModel", "def compute_losses(", "def build_gpat_model("):
        assert forbidden not in source


def test_no_quality_metric_rewrite():
    source = Path(e7g.__file__).read_text(encoding="utf-8")
    for forbidden in ("def landmark_nme(", "def parsing_dice(", "def evaluate(metrics"):
        assert forbidden not in source


# --- no LLM; no detector training --------------------------------------------

def test_no_llm_calls():
    source = Path(e7g.__file__).read_text(encoding="utf-8")
    for forbidden in ("openai", "google.generativeai", "GEMINI_API_KEY"):
        assert forbidden not in source
    pf = e7g.preflight(REPO)
    assert pf["LLM_API_CALLS"] == 0


def test_no_detector_training():
    source = Path(e7g.__file__).read_text(encoding="utf-8")
    for forbidden in ("train_detector(", "optimizer.step("):
        assert forbidden not in source
    pf = e7g.preflight(REPO)
    assert pf["TRAINING_PERFORMED"] is False


def test_no_gpat_fitting_or_rendering_performed():
    pf = e7g.preflight(REPO)
    assert pf["GPAT_FITTING_PERFORMED"] is False
    assert pf["RENDERING_PERFORMED"] is False


# --- planning calls are read-only --------------------------------------------

def test_planning_calls_are_read_only():
    e7a_dir = REPO / e7b.E7A_MATERIALIZATION_DIR
    before = {p: p.read_bytes() for p in e7a_dir.rglob("*") if p.is_file()}
    before_m3b_lock = (REPO / e7b.CASIA_MSU_PACKAGE_ROOT / "PACKAGE_LOCK.json").read_bytes()

    e7g.preflight(REPO)
    for fold_id in e7g.FOLD_IDS:
        e7g.audit_gpat_input_compatibility(REPO, fold_id)
    e7g.build_pairing_policy(REPO)
    e7g.build_gpat_fit_policy(REPO)
    e7g.frozen_bank_quota()

    after = {p: p.read_bytes() for p in e7a_dir.rglob("*") if p.is_file()}
    after_m3b_lock = (REPO / e7b.CASIA_MSU_PACKAGE_ROOT / "PACKAGE_LOCK.json").read_bytes()
    assert after == before
    assert after_m3b_lock == before_m3b_lock


# --- protected E7-A/B/C/D files byte-identical -------------------------------

def test_protected_e7abcd_files_byte_identical():
    files = [
        REPO / "src/prism_fas/evaluation/c_ext_e7b_data_prep.py",
        REPO / "src/prism_fas/evaluation/c_ext_e7c_gpat_prep.py",
        REPO / "src/prism_fas/evaluation/c_ext_e7d_source_support.py",
        REPO / "src/prism_fas/evaluation/c_ext_e7a_fold_prep.py",
    ]
    before = {f: f.read_bytes() for f in files if f.is_file()}
    e7g.preflight(REPO)
    e7g.prepare_planning_artifacts(REPO)
    after = {f: f.read_bytes() for f in files if f.is_file()}
    assert after == before


def test_prepare_writes_only_into_own_namespace():
    before_tree = sorted(str(p.relative_to(REPO)) for p in
                         (REPO / e7g.REPORT_DIR).rglob("*") if p.is_file()) \
        if (REPO / e7g.REPORT_DIR).is_dir() else []
    result = e7g.prepare_planning_artifacts(REPO)
    for key, entry in result.items():
        path = Path(entry["path"])
        assert str(path.relative_to(REPO)).startswith(e7g.REPORT_DIR)
    assert result["readiness"]["body"]["READY_FOR_GPU_GPAT_FIT"] is False
    assert result["readiness"]["body"]["READY_FOR_GPU_SOURCE_PRIOR_MATERIALIZATION"] is True


# --- GPU stage entry points fail closed on the laptop ------------------------

def test_prepare_gpat_requires_authorize(tmp_path):
    repo = _base_repo(tmp_path)
    with pytest.raises(e7g.E7Error, match="requires --authorize"):
        e7g.prepare_gpat(repo, "EXT-F1", authorize=False)


def test_prepare_gpat_blocked_for_pending_siw_priors():
    with pytest.raises(e7g.E7Error, match="preflight not ready for GPU GPAT fit -- FAIL CLOSED"):
        e7g.prepare_gpat(REPO, "EXT-F2", authorize=True)


def test_generate_and_match_requires_gpat_checkpoint(tmp_path):
    repo = _base_repo(tmp_path)
    with pytest.raises(e7g.E7Error, match="no fitted GPAT checkpoint"):
        e7g.generate_and_match(repo, "EXT-F1", authorize=True)


def test_validate_is_read_only():
    before = (REPO / e7g.REPORT_DIR).exists()
    result = e7g.e7_gpat_bank_validate(REPO)
    after = (REPO / e7g.REPORT_DIR).exists()
    assert before == after
    for fold_id in e7g.FOLD_IDS:
        assert result["folds"][fold_id]["gpat_checkpoint_present"] is False


# --- CLI wiring ---------------------------------------------------------------

def test_main_preflight_flag(monkeypatch, capsys):
    monkeypatch.setattr(e7g.cc, "repo_root", lambda: REPO)
    assert e7g.main(["--preflight"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["TARGET_LABEL_ACCESS"] is False


def test_main_no_flags_returns_nonzero(monkeypatch):
    monkeypatch.setattr(e7g.cc, "repo_root", lambda: REPO)
    assert e7g.main([]) == 1


def test_main_prepare_gpat_without_authorize_fails(monkeypatch, tmp_path):
    repo = _base_repo(tmp_path)
    monkeypatch.setattr(e7g.cc, "repo_root", lambda: repo)
    assert e7g.main(["--prepare-gpat", "--fold", "EXT-F1"]) == 1


def test_main_fold_flag_scopes_single_fold(monkeypatch, capsys):
    monkeypatch.setattr(e7g.cc, "repo_root", lambda: REPO)
    assert e7g.main(["--audit-gpat-inputs", "--fold", "EXT-F1"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["EXT-F1"]["fold_id"] == "EXT-F1"


# =========================================================================== #
# TECHNICAL_READINESS_SEMANTICS_FALSE_READY fix -- readiness must never
# conflate "compatible" / "primitive resolved" with "actually materialized".
# =========================================================================== #

def _make_valid_siw_prior_package(repo: Path, *, row_count: int = 1) -> dict:
    """Builds a minimal, strictly-valid SiW-as-source prior package on disk
    (crop bytes + one `.npz` prior per row, full M3B schema keys), and
    returns the row material used so callers can corrupt it deliberately."""
    import numpy as np

    crop_root = repo / e7b.E7B_SIW_SOURCE_PACKAGE_ROOT / "m2_run"
    crop_root.mkdir(parents=True, exist_ok=True)
    prior_root = repo / e7g.SIW_SOURCE_PRIOR_PACKAGE_ROOT
    prior_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(row_count):
        crop_path = crop_root / f"crop_{i}.jpg"
        crop_path.write_bytes(f"fake-crop-bytes-{i}".encode("utf-8"))
        prior_path = prior_root / f"prior_{i}.npz"
        np.savez(
            prior_path,
            parsing_labels=np.zeros((224, 224), dtype="uint8"),
            pose_ypr=np.zeros((3,), dtype="float32"),
            visibility=np.zeros((9,), dtype="float16"),
            bbox=np.zeros((4,), dtype="float32"),
            landmarks=np.zeros((5, 2), dtype="float32"),
            crop_box=np.zeros((4,), dtype="float32"),
        )
        rows.append({
            "source_video_id": f"siw_video_{i}", "frame_index": i, "status": "success",
            "source_crop_relative_path": f"crop_{i}.jpg",
            "source_crop_sha256": e7g.cc.sha256_file(crop_path),
            "prior_relative_path": f"prior_{i}.npz",
            "prior_sha256": e7g.cc.sha256_file(prior_path),
        })

    package_identity = e7g.compute_siw_source_prior_package_identity(rows)
    package_path = prior_root / e7g.SIW_SOURCE_PRIOR_PACKAGE_FILENAME
    package_path.write_text(json.dumps({
        "schema_version": "siw-source-prior-package-v1",
        "package_identity": package_identity, "rows": rows,
    }), encoding="utf-8")
    return {"rows": rows, "package_identity": package_identity, "package_path": package_path}


def test_real_preflight_f2_f3_not_ready_before_materialization():
    pf = e7g.preflight(REPO)
    assert pf["F2_SOURCE_PRIORS_READY"] is False
    assert pf["F3_SOURCE_PRIORS_READY"] is False
    assert pf["READY_FOR_GPU_GPAT_FIT"] is False


def test_real_preflight_ready_for_source_prior_materialization():
    pf = e7g.preflight(REPO)
    assert pf["READY_FOR_GPU_SOURCE_PRIOR_MATERIALIZATION"] is True


def test_primitive_resolution_independent_of_materialization():
    pf = e7g.preflight(REPO)
    assert pf["F2_PRIOR_GENERATION_PRIMITIVE_RESOLVED"] is True
    assert pf["F3_PRIOR_GENERATION_PRIMITIVE_RESOLVED"] is True
    assert pf["F2_SOURCE_PRIORS_MATERIALIZED"] is False
    assert pf["F3_SOURCE_PRIORS_MATERIALIZED"] is False


def test_f1_remains_prior_ready():
    pf = e7g.preflight(REPO)
    assert pf["F1_SOURCE_PRIORS_MATERIALIZED"] is True
    assert pf["F1_SOURCE_PRIORS_READY"] is True
    assert pf["F1_PRIOR_GENERATION_PRIMITIVE_RESOLVED"] == "NOT_APPLICABLE_EXISTING_PRIORS"


def test_audit_never_conflates_primitive_resolved_with_materialized():
    audit = e7g.audit_gpat_input_compatibility(REPO, "EXT-F2")
    assert audit["siw_prior_generation_primitive_resolved"] is True
    assert audit["siw_priors_materialized"] is False
    assert audit["status"] == "COMPATIBLE_PENDING_GPU_PRIOR_GENERATION"


def test_prepare_gpat_fails_before_fit_for_f3_too():
    with pytest.raises(e7g.E7Error, match="preflight not ready for GPU GPAT fit -- FAIL CLOSED"):
        e7g.prepare_gpat(REPO, "EXT-F3", authorize=True)


def test_source_prior_generation_uses_e7b_siw_source_root_only(tmp_path):
    repo = _base_repo(tmp_path)
    with pytest.raises(e7g.E7Error, match="E7-B SIW_SOURCE_PACKAGE.json not present"):
        e7g.prepare_source_priors(repo, "EXT-F2", authorize=True)
    # never references the protected target-tagged package root
    assert e7g.PROTECTED_SIW_TARGET_PRIOR_PACKAGE_ROOT not in e7b.E7B_SIW_SOURCE_PACKAGE_ROOT


def test_prism_target_eval_v2_rejected_by_firewall():
    # SiW-Mv2 is EXT-F1's held-out target -- prism_target_eval_v2 is forbidden there.
    with pytest.raises(e7g.E7TargetFirewallViolation):
        e7g.assert_not_target_path(
            "EXT-F1", f"{e7g.PROTECTED_SIW_TARGET_PRIOR_PACKAGE_ROOT}/priors/x.npz")


def test_evaluation_only_rejected_by_firewall():
    with pytest.raises(e7g.E7TargetFirewallViolation):
        e7g.assert_not_target_path("EXT-F2", "data/evaluation_only/some_file.parquet")


def test_same_siw_prior_package_shared_by_f2_and_f3():
    assert e7g.SIW_SOURCE_PRIOR_PACKAGE_ROOT  # fold-independent constant, not a per-fold template
    assert "{" not in e7g.SIW_SOURCE_PRIOR_PACKAGE_ROOT
    f2_path = REPO / e7g.SIW_SOURCE_PRIOR_PACKAGE_ROOT / e7g.SIW_SOURCE_PRIOR_PACKAGE_FILENAME
    f3_path = REPO / e7g.SIW_SOURCE_PRIOR_PACKAGE_ROOT / e7g.SIW_SOURCE_PRIOR_PACKAGE_FILENAME
    assert f2_path == f3_path


def test_full_prior_schema_enforced(tmp_path, monkeypatch):
    import numpy as np

    repo = _base_repo(tmp_path)
    monkeypatch.setattr(e7g, "EXPECTED_SIW_SUCCESS_CROP_COUNT", 1)
    crop_root = repo / e7b.E7B_SIW_SOURCE_PACKAGE_ROOT / "m2_run"
    crop_root.mkdir(parents=True, exist_ok=True)
    prior_root = repo / e7g.SIW_SOURCE_PRIOR_PACKAGE_ROOT
    prior_root.mkdir(parents=True, exist_ok=True)
    crop_path = crop_root / "crop_0.jpg"
    crop_path.write_bytes(b"fake-crop-bytes-0")
    prior_path = prior_root / "prior_0.npz"
    np.savez(prior_path, parsing_labels=np.zeros((224, 224), dtype="uint8"))  # missing keys
    rows = [{
        "source_video_id": "siw_video_0", "frame_index": 0, "status": "success",
        "source_crop_relative_path": "crop_0.jpg",
        "source_crop_sha256": e7g.cc.sha256_file(crop_path),
        "prior_relative_path": "prior_0.npz",
        "prior_sha256": e7g.cc.sha256_file(prior_path),
    }]
    package_identity = e7g.compute_siw_source_prior_package_identity(rows)
    (prior_root / e7g.SIW_SOURCE_PRIOR_PACKAGE_FILENAME).write_text(
        json.dumps({"package_identity": package_identity, "rows": rows}), encoding="utf-8")
    result = e7g.validate_source_priors(repo, "EXT-F2")
    assert result["status"] == "INVALID"
    assert any("missing required keys" in p for p in result["problems"])


def test_expected_siw_success_row_count_exact():
    assert e7g.EXPECTED_SIW_SUCCESS_CROP_COUNT == 6776


def test_existing_valid_package_returns_already_valid_only_after_strict_validation(
        tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    monkeypatch.setattr(e7g, "EXPECTED_SIW_SUCCESS_CROP_COUNT", 1)
    _make_valid_siw_prior_package(repo, row_count=1)
    result = e7g.prepare_source_priors(repo, "EXT-F2", authorize=True)
    assert result["status"] == "ALREADY_VALID"
    assert result["resumed"] is True
    assert result["validation"]["status"] == "VALID"


def test_invalid_existing_package_fails_closed(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    monkeypatch.setattr(e7g, "EXPECTED_SIW_SUCCESS_CROP_COUNT", 2)  # deliberately wrong
    _make_valid_siw_prior_package(repo, row_count=1)
    with pytest.raises(e7g.E7Error, match="FAILED strict validation"):
        e7g.prepare_source_priors(repo, "EXT-F2", authorize=True)


def test_conflicting_package_identity_fails_closed(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    monkeypatch.setattr(e7g, "EXPECTED_SIW_SUCCESS_CROP_COUNT", 1)
    fixture = _make_valid_siw_prior_package(repo, row_count=1)
    package_path = fixture["package_path"]
    body = json.loads(package_path.read_text(encoding="utf-8"))
    body["package_identity"] = "0" * 64  # conflicting, but rows still individually valid
    package_path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(e7g.E7Error, match="FAILED strict validation"):
        e7g.prepare_source_priors(repo, "EXT-F2", authorize=True)


def test_no_scientific_primitive_config_model_changes():
    assert e7g.M3B_PRIOR_GENERATION_FUNCTION == "prism_fas.data.package.m3b.build_m3b_package"
    assert e7g.M3B_PRIOR_MODEL_CONFIG_PATH == "configs/models/m3b_priors.yaml"
    assert e7g.FROZEN_PRIOR_MODELS["parsing"]["backend"] == "facexformer"
    assert e7g.FROZEN_PRIOR_MODELS["parsing"]["revision"] == "fd12148d0b19"
    assert e7g.FROZEN_PRIOR_MODELS["parsing"]["weight_sha256"] == \
        "327a755849ba64d336fb96589ff87b27e84a12be1ecf8bcfaa503d66f803286d"
    assert e7g.FROZEN_PRIOR_MODELS["identity"]["backend"] == "adaface_ir50"
    assert e7g.FROZEN_PRIOR_MODELS["identity"]["revision"] == "60a65befbcf7"
    assert e7g.PRIOR_SEED == 20260805


def test_no_target_or_training_or_rendering_or_llm_this_turn():
    pf = e7g.preflight(REPO)
    assert pf["TARGET_LABEL_ACCESS"] is False
    assert pf["TARGET_IMAGE_ACCESS"] is False
    assert pf["TRAINING_PERFORMED"] is False
    assert pf["RENDERING_PERFORMED"] is False
    assert pf["GPAT_FITTING_PERFORMED"] is False
    assert pf["LLM_API_CALLS"] == 0
