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
from prism_fas.evaluation import c_ext_e7d_source_support as e7d
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


# =========================================================================== #
# TECHNICAL_GPU_SOURCE_PRIOR_EXECUTION_AND_M3A_ADAPTER_GAP fix -- real M3A
# input-package adapter + real (gated) build_m3b_package invocation.
# =========================================================================== #

_TRAIN_FIXTURE = [
    {"video_id": "siwv1", "frame_index": 0, "label_live_spoof": "live",
     "crop_relative_path": "crops/x/siwv1_0.jpg"},
    {"video_id": "siwv2", "frame_index": 0, "label_live_spoof": "spoof",
     "crop_relative_path": "crops/x/siwv2_0.jpg"},
]
_DEV_FIXTURE = [
    {"video_id": "siwv3", "frame_index": 0, "label_live_spoof": "live",
     "crop_relative_path": "crops/x/siwv3_0.jpg"},
]


def _write_siw_m2_fixture(repo: Path, rows: list[dict], *, corrupt_sha_for: str | None = None) -> None:
    """Writes real, decodable crop JPEG bytes plus a real M2
    source_crops.parquet at E7-B's own SiW source m2_run root -- the
    schema `build_priors`/`prior_payload` genuinely require."""
    import cv2
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    m2_root = repo / e7b.E7B_SIW_SOURCE_PACKAGE_ROOT / "m2_run"
    crop_rows = []
    for index, r in enumerate(rows):
        crop_path = m2_root / r["crop_relative_path"]
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        image = (index * 37 % 256) * np.ones((64, 64, 3), dtype="uint8")
        cv2.imwrite(str(crop_path), image)
        r["crop_sha256"] = e7g.cc.sha256_file(crop_path)
        crop_rows.append({
            "sample_id": f"m2_{r['video_id']}_{r['frame_index']}", "dataset": "siw_mv2",
            "video_id": r["video_id"], "source_record_id": r["video_id"],
            "source_media_type": "video", "requested_frame_index": r["frame_index"],
            "actual_frame_index": r["frame_index"], "timestamp_ms": 0.0,
            "frame_width": 640, "frame_height": 480,
            "bbox_x1": 10.0, "bbox_y1": 10.0, "bbox_x2": 50.0, "bbox_y2": 50.0,
            "detection_score": 0.99, "detected_face_count": 1,
            "crop_x1": 0, "crop_y1": 0, "crop_x2": 64, "crop_y2": 64,
            "requested_crop_padding": 0.25, "effective_crop_padding": 0.25,
            "crop_width": 64, "crop_height": 64,
            "landmark_0_x": 20.0, "landmark_0_y": 20.0, "landmark_1_x": 30.0, "landmark_1_y": 20.0,
            "landmark_2_x": 25.0, "landmark_2_y": 30.0, "landmark_3_x": 20.0, "landmark_3_y": 40.0,
            "landmark_4_x": 30.0, "landmark_4_y": 40.0,
            "crop_relative_path": r["crop_relative_path"],
            "crop_sha256": ("0" * 64) if r["video_id"] == corrupt_sha_for else r["crop_sha256"],
            "detector_name": "scrfd", "detector_model_sha256": "d" * 64,
            "detector_provider": "CPUExecutionProvider", "detector_input_size": 640,
            "detector_threshold": 0.5, "preprocessing_version": "v1",
            "preprocessing_config_hash": "c" * 64, "status": "success",
        })
    manifests_dir = m2_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(crop_rows), manifests_dir / "source_crops.parquet")
    (repo / e7b.E7B_SIW_SOURCE_PACKAGE_ROOT).mkdir(parents=True, exist_ok=True)
    (repo / e7b.E7B_SIW_SOURCE_PACKAGE_ROOT / "SIW_SOURCE_PACKAGE.json").write_text("{}", encoding="utf-8")
    # The frozen M3A package config is a real, small, read-only file -- copied verbatim into
    # the fake repo rather than re-derived, so tests exercise the SAME config bytes.
    config_dest = repo / e7g.M3A_PACKAGE_CONFIG_PATH
    config_dest.parent.mkdir(parents=True, exist_ok=True)
    config_dest.write_text((REPO / e7g.M3A_PACKAGE_CONFIG_PATH).read_text(encoding="utf-8"),
                           encoding="utf-8")


def _e7d_siw_row(fold_id: str, r: dict, *, status: str = "success") -> dict:
    return {"fold_id": fold_id, "dataset": "SiW-Mv2", "project_split": "source_train",
           "label_live_spoof": r["label_live_spoof"], "spoof_family": None,
           "source_video_id": r["video_id"], "frame_index": r["frame_index"],
           "crop_relative_path": r.get("crop_relative_path"), "crop_sha256": r.get("crop_sha256"),
           "source_package_kind": e7d.SIW_SOURCE_PACKAGE_KIND,
           "source_package_identity": e7g.e7c.FROZEN_E7B["siw_source_package_identity"],
           "status": status, "subject_id": None,
           "failure_reason": None if status == "success" else "detector_failed"}


def _write_e7d_siw_fixture(repo: Path, fold_id: str, train_rows: list[dict], dev_rows: list[dict],
                           *, crop_path_override: str | None = None) -> None:
    fold_root = repo / e7d.E7D_OUTPUT_ROOT / fold_id
    fold_root.mkdir(parents=True, exist_ok=True)
    train_e7d = [_e7d_siw_row(fold_id, r) for r in train_rows]
    dev_e7d = [_e7d_siw_row(fold_id, r) for r in dev_rows]
    if crop_path_override is not None and train_e7d:
        train_e7d[0]["crop_relative_path"] = crop_path_override
    (fold_root / "source_train.json").write_text(json.dumps({"rows": train_e7d}), encoding="utf-8")
    (fold_root / "source_dev.json").write_text(json.dumps({"rows": dev_e7d}), encoding="utf-8")


def _patch_expected_counts(monkeypatch, *, train: int, dev: int, total: int,
                           live_train: int, spoof_train: int, live_dev: int, spoof_dev: int) -> None:
    monkeypatch.setattr(e7g, "EXPECTED_SIW_TRAIN_SUCCESS_COUNT", train)
    monkeypatch.setattr(e7g, "EXPECTED_SIW_DEV_SUCCESS_COUNT", dev)
    monkeypatch.setattr(e7g, "EXPECTED_SIW_SUCCESS_CROP_COUNT", total)
    monkeypatch.setattr(e7g, "FROZEN_SIW_CROP_ACCOUNTING", {
        **e7g.FROZEN_SIW_CROP_ACCOUNTING, "live_train_success": live_train,
        "spoof_train_success": spoof_train, "live_dev_success": live_dev, "spoof_dev_success": spoof_dev})


def _build_full_fixture(tmp_path: Path, monkeypatch) -> Path:
    """The SHARED SiW authority requires BOTH EXT-F2's and EXT-F3's E7-D
    rows to exist and agree -- writes the IDENTICAL canonical population to
    both folds, matching the real invariant (F2 and F3 share one SiW
    source population)."""
    repo = _base_repo(tmp_path)
    train = [dict(r) for r in _TRAIN_FIXTURE]
    dev = [dict(r) for r in _DEV_FIXTURE]
    _write_siw_m2_fixture(repo, train + dev)
    for fold_id in ("EXT-F2", "EXT-F3"):
        _write_e7d_siw_fixture(repo, fold_id, [dict(r) for r in train], [dict(r) for r in dev])
    _patch_expected_counts(monkeypatch, train=2, dev=1, total=3, live_train=1, spoof_train=1,
                           live_dev=1, spoof_dev=0)
    return repo


def test_m3a_input_materialization_not_a_stub(tmp_path, monkeypatch):
    repo = _build_full_fixture(tmp_path, monkeypatch)
    result = e7g.materialize_m3a_input_package(repo, "EXT-F2", authorize=True)
    assert result["status"] == "MATERIALIZED"
    assert (repo / e7g.SIW_SOURCE_M3A_INPUT_PACKAGE_ROOT / "PACKAGE_LOCK.json").is_file()


def test_m3a_project_split_authority_is_e7d_e7a(tmp_path, monkeypatch):
    repo = _build_full_fixture(tmp_path, monkeypatch)
    e7g.materialize_m3a_input_package(repo, "EXT-F2", authorize=True)
    from prism_fas.data.package.manifests import read_manifest

    root = repo / e7g.SIW_SOURCE_M3A_INPUT_PACKAGE_ROOT
    train_ids = {r["sample_id"] for r in read_manifest(root / "manifests" / "source_train.parquet")}
    dev_ids = {r["sample_id"] for r in read_manifest(root / "manifests" / "source_dev.parquet")}
    samples = {r["sample_id"]: r for r in read_manifest(root / "manifests" / "samples.parquet")}
    assert all(samples[sid]["project_split"] == "source_train" for sid in train_ids)
    assert all(samples[sid]["project_split"] == "source_dev" for sid in dev_ids)
    assert len(train_ids) == 2 and len(dev_ids) == 1


def test_m3a_official_split_placeholder_preserved(tmp_path, monkeypatch):
    repo = _build_full_fixture(tmp_path, monkeypatch)
    e7g.materialize_m3a_input_package(repo, "EXT-F2", authorize=True)
    from prism_fas.data.package.manifests import read_manifest

    root = repo / e7g.SIW_SOURCE_M3A_INPUT_PACKAGE_ROOT
    for name in ("source_train", "source_dev"):
        for row in read_manifest(root / "manifests" / f"{name}.parquet"):
            assert row["official_split"] == e7b.SIW_SOURCE_OFFICIAL_SPLIT_PLACEHOLDER


def test_no_legacy_project_split_misuse(tmp_path, monkeypatch):
    from prism_fas.data.package.config import project_split as legacy_project_split

    with pytest.raises(ValueError):
        legacy_project_split("source", e7b.SIW_SOURCE_OFFICIAL_SPLIT_PLACEHOLDER)
    # ...and yet the real adapter succeeds, proving it never calls the legacy function.
    repo = _build_full_fixture(tmp_path, monkeypatch)
    result = e7g.materialize_m3a_input_package(repo, "EXT-F2", authorize=True)
    assert result["status"] == "MATERIALIZED"


def test_m3a_counts_exact(tmp_path, monkeypatch):
    repo = _build_full_fixture(tmp_path, monkeypatch)
    e7g.materialize_m3a_input_package(repo, "EXT-F2", authorize=True)
    validation = e7g.validate_m3a_input_package(repo, "EXT-F2")
    assert validation["status"] == "VALID"
    assert validation["total_samples"] == 3
    assert validation["train_rows"] == 2
    assert validation["dev_rows"] == 1


def test_m3a_live_spoof_counts_exact(tmp_path, monkeypatch):
    repo = _build_full_fixture(tmp_path, monkeypatch)
    e7g.materialize_m3a_input_package(repo, "EXT-F2", authorize=True)
    from prism_fas.data.package.manifests import read_manifest

    root = repo / e7g.SIW_SOURCE_M3A_INPUT_PACKAGE_ROOT
    train_rows = read_manifest(root / "manifests" / "source_train.parquet")
    dev_rows = read_manifest(root / "manifests" / "source_dev.parquet")
    assert sum(1 for r in train_rows if r["label_live_spoof"] == "live") == 1
    assert sum(1 for r in train_rows if r["label_live_spoof"] == "spoof") == 1
    assert sum(1 for r in dev_rows if r["label_live_spoof"] == "live") == 1
    assert sum(1 for r in dev_rows if r["label_live_spoof"] == "spoof") == 0


def test_terminal_failures_never_enter_m3a_package(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    train = [dict(r) for r in _TRAIN_FIXTURE]
    dev = [dict(r) for r in _DEV_FIXTURE]
    _write_siw_m2_fixture(repo, train + dev)
    for fold_id in ("EXT-F2", "EXT-F3"):
        fold_root = repo / e7d.E7D_OUTPUT_ROOT / fold_id
        fold_root.mkdir(parents=True, exist_ok=True)
        train_e7d = [_e7d_siw_row(fold_id, r) for r in train]
        dev_e7d = [_e7d_siw_row(fold_id, r) for r in dev]
        # E7-D's OWN source_train.json/source_dev.json never contain a failure row -- terminal
        # failures live only in terminal_failures.json, which this adapter never reads.
        (fold_root / "source_train.json").write_text(json.dumps({"rows": train_e7d}), encoding="utf-8")
        (fold_root / "source_dev.json").write_text(json.dumps({"rows": dev_e7d}), encoding="utf-8")
        (fold_root / "terminal_failures.json").write_text(
            json.dumps({"rows": [_e7d_siw_row(fold_id, {"video_id": "siwfail", "frame_index": 0,
                                                         "label_live_spoof": "spoof"}, status="failure")]}),
            encoding="utf-8")
    _patch_expected_counts(monkeypatch, train=2, dev=1, total=3, live_train=1, spoof_train=1,
                           live_dev=1, spoof_dev=0)
    result = e7g.materialize_m3a_input_package(repo, "EXT-F2", authorize=True)
    assert result["status"] == "MATERIALIZED"
    validation = e7g.validate_m3a_input_package(repo, "EXT-F2")
    assert validation["total_samples"] == 3  # the failure row never entered


def test_m3a_no_subject_id_fabrication(tmp_path, monkeypatch):
    repo = _build_full_fixture(tmp_path, monkeypatch)
    e7g.materialize_m3a_input_package(repo, "EXT-F2", authorize=True)
    from prism_fas.data.package.manifests import read_manifest

    root = repo / e7g.SIW_SOURCE_M3A_INPUT_PACKAGE_ROOT
    for name in ("source_train", "source_dev"):
        for row in read_manifest(root / "manifests" / f"{name}.parquet"):
            assert row["subject_id"] is None


def test_m3a_crop_sha_verified(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    train = [dict(r) for r in _TRAIN_FIXTURE]
    dev = [dict(r) for r in _DEV_FIXTURE]
    _write_siw_m2_fixture(repo, train + dev, corrupt_sha_for="siwv1")
    for fold_id in ("EXT-F2", "EXT-F3"):
        _write_e7d_siw_fixture(repo, fold_id, [dict(r) for r in train], [dict(r) for r in dev])
    _patch_expected_counts(monkeypatch, train=2, dev=1, total=3, live_train=1, spoof_train=1,
                           live_dev=1, spoof_dev=0)
    with pytest.raises(e7g.E7Error, match="disagrees with the real M2 crop manifest"):
        e7g.materialize_m3a_input_package(repo, "EXT-F2", authorize=True)


def test_m3a_full_base_prior_schema(tmp_path, monkeypatch):
    repo = _build_full_fixture(tmp_path, monkeypatch)
    e7g.materialize_m3a_input_package(repo, "EXT-F2", authorize=True)
    from prism_fas.data.package.manifests import read_manifest
    from prism_fas.data.package.priors import PRIOR_ARRAYS, load_prior, validate_prior_arrays

    root = repo / e7g.SIW_SOURCE_M3A_INPUT_PACKAGE_ROOT
    for row in read_manifest(root / "manifests" / "samples.parquet"):
        arrays = load_prior(root / row["prior_relative_path"])
        validate_prior_arrays(arrays)  # never raises for a well-formed base prior
        assert set(PRIOR_ARRAYS) <= set(arrays)


def test_m3a_target_package_never_used(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    train = [dict(r) for r in _TRAIN_FIXTURE]
    dev = [dict(r) for r in _DEV_FIXTURE]
    _write_siw_m2_fixture(repo, train + dev)
    override = f"../../../{e7g.PROTECTED_SIW_TARGET_PRIOR_PACKAGE_ROOT}/x.jpg"
    for fold_id in ("EXT-F2", "EXT-F3"):
        # Both folds must agree (even on the malicious override) so the shared-authority
        # equality check doesn't mask this as a mere population mismatch.
        _write_e7d_siw_fixture(repo, fold_id, [dict(r) for r in train], [dict(r) for r in dev],
                               crop_path_override=override)
    _patch_expected_counts(monkeypatch, train=2, dev=1, total=3, live_train=1, spoof_train=1,
                           live_dev=1, spoof_dev=0)
    with pytest.raises(e7g.E7Error):
        e7g.materialize_m3a_input_package(repo, "EXT-F2", authorize=True)


def test_m3a_evaluation_only_never_opened(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    train = [dict(r) for r in _TRAIN_FIXTURE]
    dev = [dict(r) for r in _DEV_FIXTURE]
    _write_siw_m2_fixture(repo, train + dev)
    override = "../../../../data/evaluation_only/x.jpg"
    for fold_id in ("EXT-F2", "EXT-F3"):
        _write_e7d_siw_fixture(repo, fold_id, [dict(r) for r in train], [dict(r) for r in dev],
                               crop_path_override=override)
    _patch_expected_counts(monkeypatch, train=2, dev=1, total=3, live_train=1, spoof_train=1,
                           live_dev=1, spoof_dev=0)
    with pytest.raises(e7g.E7Error):
        e7g.materialize_m3a_input_package(repo, "EXT-F2", authorize=True)


def test_m3a_validator_is_read_only(tmp_path, monkeypatch):
    repo = _build_full_fixture(tmp_path, monkeypatch)
    e7g.materialize_m3a_input_package(repo, "EXT-F2", authorize=True)
    root = repo / e7g.SIW_SOURCE_M3A_INPUT_PACKAGE_ROOT
    before = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
    e7g.validate_m3a_input_package(repo, "EXT-F2")
    after = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
    assert before == after


def test_m3a_existing_valid_package_already_valid(tmp_path, monkeypatch):
    repo = _build_full_fixture(tmp_path, monkeypatch)
    first = e7g.materialize_m3a_input_package(repo, "EXT-F2", authorize=True)
    assert first["status"] == "MATERIALIZED"
    lock_path = repo / e7g.SIW_SOURCE_M3A_INPUT_PACKAGE_ROOT / "PACKAGE_LOCK.json"
    mtime_before = lock_path.stat().st_mtime_ns
    second = e7g.materialize_m3a_input_package(repo, "EXT-F2", authorize=True)
    assert second["status"] == "ALREADY_VALID"
    assert second["resumed"] is True
    assert lock_path.stat().st_mtime_ns == mtime_before


def test_m3a_interrupted_partial_never_already_valid(tmp_path, monkeypatch):
    repo = _build_full_fixture(tmp_path, monkeypatch)
    e7g.materialize_m3a_input_package(repo, "EXT-F2", authorize=True)
    lock_path = repo / e7g.SIW_SOURCE_M3A_INPUT_PACKAGE_ROOT / "PACKAGE_LOCK.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["status"] = "building"  # simulate a crash before finalize_lock ran
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    result = e7g.materialize_m3a_input_package(repo, "EXT-F2", authorize=True)
    assert result["status"] != "ALREADY_VALID"


def test_m3a_invalid_existing_package_fails_closed(tmp_path, monkeypatch):
    repo = _build_full_fixture(tmp_path, monkeypatch)
    e7g.materialize_m3a_input_package(repo, "EXT-F2", authorize=True)
    binding_path = repo / e7g.SIW_SOURCE_M3A_INPUT_PACKAGE_ROOT / e7g.M3A_INPUT_BINDING_FILENAME
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["m3a_input_package_identity"] = "0" * 64
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises(e7g.E7Error, match="FAILED strict validation"):
        e7g.materialize_m3a_input_package(repo, "EXT-F2", authorize=True)


def test_m3a_f1_not_applicable(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    with pytest.raises(e7g.E7Error, match="NOT_APPLICABLE"):
        e7g.materialize_m3a_input_package(repo, "EXT-F1", authorize=True)


# --- outer prepare_source_priors transaction (GPU boundary real, not a stub) -------------

def test_prepare_source_priors_not_unconditional_stub(tmp_path, monkeypatch):
    repo = _build_full_fixture(tmp_path, monkeypatch)
    with pytest.raises(e7g.E7Error, match="GPU_REQUIRED"):
        e7g.prepare_source_priors(repo, "EXT-F2", authorize=True)
    # by the time it fails, the M3A input package IS materialized and strictly valid --
    # proof this is no longer an unconditional stub raise.
    assert (repo / e7g.SIW_SOURCE_M3A_INPUT_PACKAGE_ROOT / "PACKAGE_LOCK.json").is_file()
    assert e7g.validate_m3a_input_package(repo, "EXT-F2")["status"] == "VALID"


def test_gpu_capability_check_is_real_not_hardcoded(tmp_path):
    repo = _base_repo(tmp_path)
    capability = e7g._gpu_prior_generation_capability(repo)
    assert capability["capable"] is False
    assert capability["cuda_available"] is False
    assert any("cuda" in p.lower() for p in capability["problems"])
    assert not any("non-gpu host" in p.lower() for p in capability["problems"])


def test_build_m3b_package_invoked_only_after_m3a_validation(tmp_path, monkeypatch):
    repo = _build_full_fixture(tmp_path, monkeypatch)
    calls: list[dict] = []

    def fake_build_m3b_package(input_package, output_package, model_config, *, weight_root, resume,
                               package_id):
        calls.append({"input_package": input_package, "output_package": output_package,
                      "model_config": model_config, "weight_root": weight_root, "resume": resume,
                      "package_id": package_id})
        raise RuntimeError("intentionally stop right after the spy records the call")

    monkeypatch.setattr(e7g, "_gpu_prior_generation_capability",
                        lambda repo: {"capable": True, "cuda_available": True,
                                     "weight_root": str(repo / "model_cache"), "problems": []})
    monkeypatch.setattr("prism_fas.data.package.m3b.build_m3b_package", fake_build_m3b_package)
    with pytest.raises(RuntimeError, match="intentionally stop"):
        e7g.prepare_source_priors(repo, "EXT-F2", authorize=True)
    assert len(calls) == 1
    # M3A input package was VALID before build_m3b_package was ever called.
    assert e7g.validate_m3a_input_package(repo, "EXT-F2")["status"] == "VALID"
    assert calls[0]["input_package"] == repo / e7g.SIW_SOURCE_M3A_INPUT_PACKAGE_ROOT
    assert calls[0]["output_package"] == repo / e7g.SIW_SOURCE_PRIOR_PACKAGE_ROOT


def test_frozen_model_config_seed_passed_exactly_to_build_m3b_package(tmp_path, monkeypatch):
    repo = _build_full_fixture(tmp_path, monkeypatch)
    calls: list[dict] = []

    def fake_build_m3b_package(input_package, output_package, model_config, *, weight_root, resume,
                               package_id):
        calls.append({"model_config": model_config, "package_id": package_id})
        raise RuntimeError("stop")

    monkeypatch.setattr(e7g, "_gpu_prior_generation_capability",
                        lambda repo: {"capable": True, "cuda_available": True,
                                     "weight_root": str(repo / "model_cache"), "problems": []})
    monkeypatch.setattr("prism_fas.data.package.m3b.build_m3b_package", fake_build_m3b_package)
    with pytest.raises(RuntimeError):
        e7g.prepare_source_priors(repo, "EXT-F2", authorize=True)
    assert calls[0]["model_config"] == repo / e7g.M3B_PRIOR_MODEL_CONFIG_PATH
    assert calls[0]["package_id"] == e7g.SIW_SOURCE_PRIOR_M3B_PACKAGE_ID
    assert calls[0]["package_id"] != e7b.FROZEN_M3B_PACKAGE_IDENTITY  # never claims the frozen id


def test_siw_source_prior_package_written_last(tmp_path, monkeypatch):
    repo = _build_full_fixture(tmp_path, monkeypatch)
    order: list[str] = []
    output_root = repo / e7g.SIW_SOURCE_PRIOR_PACKAGE_ROOT

    def fake_build_m3b_package(input_package, output_package, model_config, *, weight_root, resume,
                               package_id):
        order.append("build_m3b_package")
        assert not (output_root / e7g.SIW_SOURCE_PRIOR_PACKAGE_FILENAME).exists()
        output_package.mkdir(parents=True, exist_ok=True)
        return {"failures": [], "lock": {"content_identity_sha256": "x"}}

    from prism_fas.data.package.validator import validate_package as _orig_validate_package
    from prism_fas.data.package.builder import finalize_lock as _orig_finalize_lock

    def fake_validate_package(package_root, **kwargs):
        if Path(package_root) != output_root:
            return _orig_validate_package(package_root, **kwargs)  # the M3A package's own call
        order.append("validate_package")
        assert not (output_root / e7g.SIW_SOURCE_PRIOR_PACKAGE_FILENAME).exists()
        return {"passed": True, "errors": [], "checks": [], "counts": {}, "package_id": "x",
               "target_isolation": {"passed": True}}

    def fake_finalize_lock(package_root, report):
        if Path(package_root) != output_root:
            return _orig_finalize_lock(package_root, report)  # the M3A package's own call
        order.append("finalize_lock")
        assert not (output_root / e7g.SIW_SOURCE_PRIOR_PACKAGE_FILENAME).exists()
        return {"status": "validated"}

    def fake_derive_rows(repo_arg):
        order.append("derive_rows")
        return []

    def fake_write_source_prior(rows):
        order.append("compute_identity")
        return "y" * 64

    def fake_candidate_validate(repo_arg, fold_id, rows, package_identity):
        order.append("candidate_validate")
        assert not (output_root / e7g.SIW_SOURCE_PRIOR_PACKAGE_FILENAME).exists()
        return {"status": "VALID", "problems": []}

    monkeypatch.setattr(e7g, "_gpu_prior_generation_capability",
                        lambda repo: {"capable": True, "cuda_available": True,
                                     "weight_root": str(repo / "model_cache"), "problems": []})
    monkeypatch.setattr("prism_fas.data.package.m3b.build_m3b_package", fake_build_m3b_package)
    monkeypatch.setattr("prism_fas.data.package.validator.validate_package", fake_validate_package)
    monkeypatch.setattr("prism_fas.data.package.builder.finalize_lock", fake_finalize_lock)
    monkeypatch.setattr(e7g, "_derive_siw_source_prior_rows", fake_derive_rows)
    monkeypatch.setattr(e7g, "compute_siw_source_prior_package_identity", fake_write_source_prior)
    monkeypatch.setattr(e7g, "validate_source_prior_candidate", fake_candidate_validate)
    monkeypatch.setattr(e7g, "validate_source_priors",
                        lambda repo, fold_id: {"status": "VALID", "recomputed_package_identity": "y" * 64})

    result = e7g.prepare_source_priors(repo, "EXT-F2", authorize=True)
    assert result["status"] == "MATERIALIZED"
    # build_m3b_package, THEN pre-finalize validate, THEN finalize_lock, THEN the post-finalize
    # validate (mirroring cli/main.py's own pre/finalize_lock/report pattern), THEN the shared
    # prior rows are derived and the CANDIDATE is strictly validated, and only THEN is the
    # terminal marker ever written.
    assert order == ["build_m3b_package", "validate_package", "finalize_lock", "validate_package",
                     "derive_rows", "compute_identity", "candidate_validate"]
    assert (output_root / e7g.SIW_SOURCE_PRIOR_PACKAGE_FILENAME).is_file()


def test_f3_after_f2_valid_no_second_model_inference(tmp_path, monkeypatch):
    repo = _build_full_fixture(tmp_path, monkeypatch)
    # F3 needs its own E7-D authoritative rows (shares the same underlying SiW population).
    _write_e7d_siw_fixture(repo, "EXT-F3", [dict(r) for r in _TRAIN_FIXTURE], [dict(r) for r in _DEV_FIXTURE])
    output_root = repo / e7g.SIW_SOURCE_PRIOR_PACKAGE_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / e7g.SIW_SOURCE_PRIOR_PACKAGE_FILENAME).write_text(
        json.dumps({"schema_version": "siw-source-prior-package-v1", "package_identity": "z" * 64,
                   "rows": []}), encoding="utf-8")
    monkeypatch.setattr(e7g, "validate_source_priors",
                        lambda repo, fold_id: {"status": "VALID", "recomputed_package_identity": "z" * 64})
    calls: list[str] = []
    monkeypatch.setattr("prism_fas.data.package.m3b.build_m3b_package",
                        lambda *a, **k: calls.append("called"))
    result = e7g.prepare_source_priors(repo, "EXT-F3", authorize=True)
    assert result["status"] == "ALREADY_VALID"
    assert calls == []  # build_m3b_package (model inference) never called for F3


def test_invalid_conflicting_existing_prior_package_fails_closed(tmp_path, monkeypatch):
    repo = _build_full_fixture(tmp_path, monkeypatch)
    output_root = repo / e7g.SIW_SOURCE_PRIOR_PACKAGE_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / e7g.SIW_SOURCE_PRIOR_PACKAGE_FILENAME).write_text(
        json.dumps({"schema_version": "siw-source-prior-package-v1", "package_identity": "bad",
                   "rows": [{"source_video_id": "v", "frame_index": 0, "status": "success"}]}),
        encoding="utf-8")
    with pytest.raises(e7g.E7Error, match="FAILED strict"):
        e7g.prepare_source_priors(repo, "EXT-F2", authorize=True)


def test_prepare_source_priors_no_gpat_render_train_llm(tmp_path, monkeypatch):
    repo = _build_full_fixture(tmp_path, monkeypatch)
    try:
        e7g.prepare_source_priors(repo, "EXT-F2", authorize=True)
    except e7g.E7Error:
        pass
    pf = e7g.preflight(repo)
    assert pf["GPAT_FITTING_PERFORMED"] is False
    assert pf["RENDERING_PERFORMED"] is False
    assert pf["TRAINING_PERFORMED"] is False
    assert pf["LLM_API_CALLS"] == 0


def test_synthesis_and_package_primitives_unchanged():
    import subprocess

    for relative in ("src/prism_fas/data/package/m3b.py", "src/prism_fas/data/package/builder.py",
                     "src/prism_fas/data/package/priors.py", "src/prism_fas/data/package/model_priors.py",
                     "src/prism_fas/data/package/manifests.py", "src/prism_fas/data/package/quality.py",
                     "src/prism_fas/data/package/validator.py", "src/prism_fas/data/package/config.py",
                     "src/prism_fas/data/package/shards.py"):
        committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=REPO, check=True,
                                   capture_output=True, text=True).stdout
        on_disk = (REPO / relative).read_text(encoding="utf-8")
        assert committed == on_disk, f"{relative} differs from HEAD -- must remain unmodified"


def test_e7abcd_modules_unchanged():
    import subprocess

    for relative in ("src/prism_fas/evaluation/c_ext_e7a_fold_prep.py",
                     "src/prism_fas/evaluation/c_ext_e7b_data_prep.py",
                     "src/prism_fas/evaluation/c_ext_e7c_gpat_prep.py",
                     "src/prism_fas/evaluation/c_ext_e7d_source_support.py"):
        committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=REPO, check=True,
                                   capture_output=True, text=True).stdout
        on_disk = (REPO / relative).read_text(encoding="utf-8")
        assert committed == on_disk, f"{relative} differs from HEAD -- must remain unmodified"


# --- CLI: explicit --fold required for --prepare-source-priors ---------------------------

def test_cli_prepare_source_priors_requires_explicit_fold(monkeypatch, tmp_path):
    repo = _base_repo(tmp_path)
    monkeypatch.setattr(e7g.cc, "repo_root", lambda: repo)
    assert e7g.main(["--prepare-source-priors", "--authorize"]) == 1


def test_cli_prepare_source_priors_rejects_f1(monkeypatch, tmp_path):
    repo = _base_repo(tmp_path)
    monkeypatch.setattr(e7g.cc, "repo_root", lambda: repo)
    assert e7g.main(["--prepare-source-priors", "--authorize", "--fold", "EXT-F1"]) == 1


def test_cli_prepare_source_priors_accepts_f2(monkeypatch, tmp_path, capsys):
    repo = _base_repo(tmp_path)
    monkeypatch.setattr(e7g.cc, "repo_root", lambda: repo)
    assert e7g.main(["--prepare-source-priors", "--authorize", "--fold", "EXT-F2"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert "EXT-F2" in out


# =========================================================================== #
# TECHNICAL_SHARED_PRIOR_IDENTITY_AND_TERMINAL_MARKER_GAP fix.
# =========================================================================== #

# --- GAP 2: shared SiW authority is fold-order independent -------------------------------

def test_f2_f3_populations_must_match_exactly(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    train = [dict(r) for r in _TRAIN_FIXTURE]
    dev = [dict(r) for r in _DEV_FIXTURE]
    _write_siw_m2_fixture(repo, train + dev)
    _write_e7d_siw_fixture(repo, "EXT-F2", [dict(r) for r in train], [dict(r) for r in dev])
    diverged_train = [dict(r) for r in train]
    diverged_train[0]["label_live_spoof"] = "spoof" if diverged_train[0]["label_live_spoof"] == "live" else "live"
    _write_e7d_siw_fixture(repo, "EXT-F3", diverged_train, [dict(r) for r in dev])
    _patch_expected_counts(monkeypatch, train=2, dev=1, total=3, live_train=1, spoof_train=1,
                           live_dev=1, spoof_dev=0)
    with pytest.raises(e7g.E7Error, match="DIFFER"):
        e7g.load_siw_shared_source_authority(repo)


def test_shared_authority_counts_exact(tmp_path, monkeypatch):
    repo = _build_full_fixture(tmp_path, monkeypatch)
    authority = e7g.load_siw_shared_source_authority(repo)
    assert authority["train_count"] == 2
    assert authority["dev_count"] == 1
    assert authority["row_count"] == 3


def test_shared_authority_provenance_binds_both_fold_identities(tmp_path, monkeypatch):
    repo = _build_full_fixture(tmp_path, monkeypatch)
    authority = e7g.load_siw_shared_source_authority(repo)
    assert set(authority["e7d_fold_package_identities"]) == {"EXT-F2", "EXT-F3"}


def test_m3a_identity_fold_order_independent(tmp_path, monkeypatch):
    tmp_a = tmp_path / "a"
    tmp_a.mkdir()
    tmp_b = tmp_path / "b"
    tmp_b.mkdir()
    repo_f2_first = _build_full_fixture(tmp_a, monkeypatch)
    result_f2 = e7g.materialize_m3a_input_package(repo_f2_first, "EXT-F2", authorize=True)

    repo_f3_first = _build_full_fixture(tmp_b, monkeypatch)
    result_f3 = e7g.materialize_m3a_input_package(repo_f3_first, "EXT-F3", authorize=True)

    assert result_f2["m3a_input_package_identity"] == result_f3["m3a_input_package_identity"]


def test_f2_materialize_first_then_f3_validates(tmp_path, monkeypatch):
    repo = _build_full_fixture(tmp_path, monkeypatch)
    e7g.materialize_m3a_input_package(repo, "EXT-F2", authorize=True)
    validation = e7g.validate_m3a_input_package(repo, "EXT-F3")
    assert validation["status"] == "VALID"


def test_f3_materialize_first_then_f2_validates(tmp_path, monkeypatch):
    repo = _build_full_fixture(tmp_path, monkeypatch)
    e7g.materialize_m3a_input_package(repo, "EXT-F3", authorize=True)
    validation = e7g.validate_m3a_input_package(repo, "EXT-F2")
    assert validation["status"] == "VALID"


def test_validate_m3a_rejects_when_current_fold_authority_diverges_from_binding(tmp_path, monkeypatch):
    repo = _build_full_fixture(tmp_path, monkeypatch)
    e7g.materialize_m3a_input_package(repo, "EXT-F2", authorize=True)
    # E7-D data for EXT-F3 changes AFTER materialization -- validating from EXT-F3 must now
    # detect the divergence rather than merely trusting the binding file's own stored rows.
    fold_root = repo / e7d.E7D_OUTPUT_ROOT / "EXT-F3"
    body = json.loads((fold_root / "source_train.json").read_text(encoding="utf-8"))
    body["rows"][0]["label_live_spoof"] = ("spoof" if body["rows"][0]["label_live_spoof"] == "live"
                                           else "live")
    (fold_root / "source_train.json").write_text(json.dumps(body), encoding="utf-8")
    validation = e7g.validate_m3a_input_package(repo, "EXT-F3")
    assert validation["status"] == "INVALID"
    assert any("does not exactly equal" in p or "DIFFER" in p for p in validation["problems"])


def test_m3a_identity_binds_label_live_spoof():
    rows_a = [{"source_video_id": "v", "frame_index": 0, "project_split": "source_train",
              "label_live_spoof": "live", "crop_sha256": "c" * 64, "base_prior_sha256": "p" * 64}]
    rows_b = [{**rows_a[0], "label_live_spoof": "spoof"}]
    identity_a = e7g.compute_m3a_input_package_identity(
        shared_authority_identity="s" * 64, m3a_config_identity="cfg", rows=rows_a)
    identity_b = e7g.compute_m3a_input_package_identity(
        shared_authority_identity="s" * 64, m3a_config_identity="cfg", rows=rows_b)
    assert identity_a != identity_b


def test_m3a_identity_binds_base_prior_sha():
    rows_a = [{"source_video_id": "v", "frame_index": 0, "project_split": "source_train",
              "label_live_spoof": "live", "crop_sha256": "c" * 64, "base_prior_sha256": "p" * 64}]
    rows_b = [{**rows_a[0], "base_prior_sha256": "q" * 64}]
    identity_a = e7g.compute_m3a_input_package_identity(
        shared_authority_identity="s" * 64, m3a_config_identity="cfg", rows=rows_a)
    identity_b = e7g.compute_m3a_input_package_identity(
        shared_authority_identity="s" * 64, m3a_config_identity="cfg", rows=rows_b)
    assert identity_a != identity_b


def test_m3a_identity_binds_config_hash():
    rows = [{"source_video_id": "v", "frame_index": 0, "project_split": "source_train",
            "label_live_spoof": "live", "crop_sha256": "c" * 64, "base_prior_sha256": "p" * 64}]
    identity_a = e7g.compute_m3a_input_package_identity(
        shared_authority_identity="s" * 64, m3a_config_identity="cfg-1", rows=rows)
    identity_b = e7g.compute_m3a_input_package_identity(
        shared_authority_identity="s" * 64, m3a_config_identity="cfg-2", rows=rows)
    assert identity_a != identity_b


def test_m3a_identity_binds_shared_authority_identity():
    rows = [{"source_video_id": "v", "frame_index": 0, "project_split": "source_train",
            "label_live_spoof": "live", "crop_sha256": "c" * 64, "base_prior_sha256": "p" * 64}]
    identity_a = e7g.compute_m3a_input_package_identity(
        shared_authority_identity="s" * 64, m3a_config_identity="cfg", rows=rows)
    identity_b = e7g.compute_m3a_input_package_identity(
        shared_authority_identity="t" * 64, m3a_config_identity="cfg", rows=rows)
    assert identity_a != identity_b


def test_m3a_binding_rows_carry_required_fields(tmp_path, monkeypatch):
    repo = _build_full_fixture(tmp_path, monkeypatch)
    e7g.materialize_m3a_input_package(repo, "EXT-F2", authorize=True)
    binding_path = repo / e7g.SIW_SOURCE_M3A_INPUT_PACKAGE_ROOT / e7g.M3A_INPUT_BINDING_FILENAME
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    required = {"sample_id", "source_video_id", "frame_index", "project_split", "label_live_spoof",
               "crop_relative_path", "crop_sha256", "base_prior_relative_path", "base_prior_sha256",
               "preprocessing_config_hash", "detector_model_sha256"}
    for row in binding["rows"]:
        assert required <= set(row)


def test_m3a_binding_never_fabricates_fields(tmp_path, monkeypatch):
    repo = _build_full_fixture(tmp_path, monkeypatch)
    e7g.materialize_m3a_input_package(repo, "EXT-F2", authorize=True)
    binding_path = repo / e7g.SIW_SOURCE_M3A_INPUT_PACKAGE_ROOT / e7g.M3A_INPUT_BINDING_FILENAME
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    from prism_fas.data.package.manifests import read_manifest

    root = repo / e7g.SIW_SOURCE_M3A_INPUT_PACKAGE_ROOT
    priors_index = {r["sample_id"]: r for r in read_manifest(root / "manifests" / "priors_index.parquet")}
    for row in binding["rows"]:
        prior = priors_index[row["sample_id"]]
        assert row["base_prior_sha256"] == prior["prior_sha256"]
        assert row["base_prior_relative_path"] == prior["prior_relative_path"]


# --- GAP 1: candidate validated BEFORE the terminal marker is ever written ----------------

def _make_candidate_prior_material(repo: Path, *, row_count: int = 1,
                                   corrupt_crop_sha: bool = False, corrupt_prior_sha: bool = False,
                                   missing_prior_key: bool = False,
                                   missing_prior_file: bool = False) -> tuple[list[dict], str]:
    """Real crop bytes + real `.npz` prior files on disk (full M3B schema
    unless deliberately corrupted), WITHOUT ever writing
    SIW_SOURCE_PRIOR_PACKAGE.json -- the exact pre-write state
    `validate_source_prior_candidate` must operate against."""
    import numpy as np

    crop_root = repo / e7b.E7B_SIW_SOURCE_PACKAGE_ROOT / "m2_run"
    crop_root.mkdir(parents=True, exist_ok=True)
    prior_root = repo / e7g.SIW_SOURCE_PRIOR_PACKAGE_ROOT
    prior_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(row_count):
        crop_path = crop_root / f"crop_{i}.jpg"
        crop_path.write_bytes(f"fake-crop-bytes-{i}".encode("utf-8"))
        real_crop_sha = e7g.cc.sha256_file(crop_path)
        prior_path = prior_root / f"prior_{i}.npz"
        if not missing_prior_file:
            arrays = dict(parsing_labels=np.zeros((224, 224), dtype="uint8"),
                         pose_ypr=np.zeros((3,), dtype="float32"),
                         visibility=np.zeros((9,), dtype="float16"),
                         bbox=np.zeros((4,), dtype="float32"),
                         landmarks=np.zeros((5, 2), dtype="float32"),
                         crop_box=np.zeros((4,), dtype="float32"))
            if missing_prior_key:
                del arrays["parsing_labels"]
            np.savez(prior_path, **arrays)
        real_prior_sha = e7g.cc.sha256_file(prior_path) if prior_path.is_file() else "0" * 64
        rows.append({
            "source_video_id": f"siw_video_{i}", "frame_index": i, "status": "success",
            "source_crop_relative_path": f"crop_{i}.jpg",
            "source_crop_sha256": ("0" * 64) if corrupt_crop_sha else real_crop_sha,
            "prior_relative_path": f"prior_{i}.npz",
            "prior_sha256": ("0" * 64) if corrupt_prior_sha else real_prior_sha,
        })
    package_identity = e7g.compute_siw_source_prior_package_identity(rows)
    return rows, package_identity


def test_candidate_validation_before_marker_write(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    monkeypatch.setattr(e7g, "EXPECTED_SIW_SUCCESS_CROP_COUNT", 1)
    rows, package_identity = _make_candidate_prior_material(repo, row_count=1)
    package_path = repo / e7g.SIW_SOURCE_PRIOR_PACKAGE_ROOT / e7g.SIW_SOURCE_PRIOR_PACKAGE_FILENAME
    assert not package_path.exists()
    result = e7g.validate_source_prior_candidate(repo, "EXT-F2", rows, package_identity)
    assert result["status"] == "VALID"
    assert not package_path.exists()  # candidate validation itself never writes


def test_corrupt_or_missing_prior_before_commit_leaves_no_marker(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    monkeypatch.setattr(e7g, "EXPECTED_SIW_SUCCESS_CROP_COUNT", 1)
    rows, package_identity = _make_candidate_prior_material(repo, row_count=1, missing_prior_file=True)
    result = e7g.validate_source_prior_candidate(repo, "EXT-F2", rows, package_identity)
    assert result["status"] == "INVALID"
    package_path = repo / e7g.SIW_SOURCE_PRIOR_PACKAGE_ROOT / e7g.SIW_SOURCE_PRIOR_PACKAGE_FILENAME
    assert not package_path.exists()


def test_bad_crop_sha_before_commit_leaves_no_marker(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    monkeypatch.setattr(e7g, "EXPECTED_SIW_SUCCESS_CROP_COUNT", 1)
    rows, package_identity = _make_candidate_prior_material(repo, row_count=1, corrupt_crop_sha=True)
    result = e7g.validate_source_prior_candidate(repo, "EXT-F2", rows, package_identity)
    assert result["status"] == "INVALID"
    assert result["bad_crop_hashes"] == 1
    package_path = repo / e7g.SIW_SOURCE_PRIOR_PACKAGE_ROOT / e7g.SIW_SOURCE_PRIOR_PACKAGE_FILENAME
    assert not package_path.exists()


def test_bad_prior_sha_before_commit_leaves_no_marker(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    monkeypatch.setattr(e7g, "EXPECTED_SIW_SUCCESS_CROP_COUNT", 1)
    rows, package_identity = _make_candidate_prior_material(repo, row_count=1, corrupt_prior_sha=True)
    result = e7g.validate_source_prior_candidate(repo, "EXT-F2", rows, package_identity)
    assert result["status"] == "INVALID"
    assert result["bad_prior_hashes"] == 1
    package_path = repo / e7g.SIW_SOURCE_PRIOR_PACKAGE_ROOT / e7g.SIW_SOURCE_PRIOR_PACKAGE_FILENAME
    assert not package_path.exists()


def test_missing_required_prior_key_before_commit_leaves_no_marker(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    monkeypatch.setattr(e7g, "EXPECTED_SIW_SUCCESS_CROP_COUNT", 1)
    rows, package_identity = _make_candidate_prior_material(repo, row_count=1, missing_prior_key=True)
    result = e7g.validate_source_prior_candidate(repo, "EXT-F2", rows, package_identity)
    assert result["status"] == "INVALID"
    assert any("missing required keys" in p for p in result["problems"])
    package_path = repo / e7g.SIW_SOURCE_PRIOR_PACKAGE_ROOT / e7g.SIW_SOURCE_PRIOR_PACKAGE_FILENAME
    assert not package_path.exists()


def test_identity_mismatch_before_commit_leaves_no_marker(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    monkeypatch.setattr(e7g, "EXPECTED_SIW_SUCCESS_CROP_COUNT", 1)
    rows, _real_identity = _make_candidate_prior_material(repo, row_count=1)
    result = e7g.validate_source_prior_candidate(repo, "EXT-F2", rows, "0" * 64)  # wrong identity
    assert result["status"] == "INVALID"
    assert result["package_identity_match"] is False
    package_path = repo / e7g.SIW_SOURCE_PRIOR_PACKAGE_ROOT / e7g.SIW_SOURCE_PRIOR_PACKAGE_FILENAME
    assert not package_path.exists()


def test_successful_candidate_validation_marker_is_final_write(tmp_path, monkeypatch):
    repo = _build_full_fixture(tmp_path, monkeypatch)
    package_path = repo / e7g.SIW_SOURCE_PRIOR_PACKAGE_ROOT / e7g.SIW_SOURCE_PRIOR_PACKAGE_FILENAME

    def fake_build_m3b_package(input_package, output_package, model_config, *, weight_root, resume,
                               package_id):
        output_package.mkdir(parents=True, exist_ok=True)
        return {"failures": [], "lock": {"content_identity_sha256": "x"}}

    def fake_validate_package(pr, **kwargs):
        return {"passed": True, "errors": [], "checks": [], "counts": {}, "package_id": "x",
               "target_isolation": {"passed": True}}

    def fake_finalize_lock(pr, report):
        return {"status": "validated"}

    fake_rows, fake_identity = ([{"source_video_id": "v", "frame_index": 0, "status": "success",
                                 "source_crop_relative_path": "x.jpg", "source_crop_sha256": "a" * 64,
                                 "prior_relative_path": "p.npz", "prior_sha256": "b" * 64}], "z" * 64)

    monkeypatch.setattr(e7g, "_gpu_prior_generation_capability",
                        lambda repo: {"capable": True, "cuda_available": True,
                                     "weight_root": str(repo / "model_cache"), "problems": []})
    monkeypatch.setattr("prism_fas.data.package.m3b.build_m3b_package", fake_build_m3b_package)
    monkeypatch.setattr("prism_fas.data.package.validator.validate_package", fake_validate_package)
    monkeypatch.setattr("prism_fas.data.package.builder.finalize_lock", fake_finalize_lock)
    monkeypatch.setattr(e7g, "_derive_siw_source_prior_rows", lambda repo_arg: fake_rows)
    monkeypatch.setattr(e7g, "compute_siw_source_prior_package_identity", lambda rows: fake_identity)
    monkeypatch.setattr(e7g, "validate_source_prior_candidate",
                        lambda repo, fold_id, rows, identity: {"status": "VALID", "problems": []})
    # The optional post-write validation step re-reads the just-written marker; its own
    # correctness is covered by the dedicated validate_source_priors tests, so it is mocked
    # here to isolate what this test actually checks: write ordering and write content.
    monkeypatch.setattr(e7g, "validate_source_priors",
                        lambda repo, fold_id: {"status": "VALID", "recomputed_package_identity": fake_identity})

    assert not package_path.exists()
    result = e7g.prepare_source_priors(repo, "EXT-F2", authorize=True)
    assert result["status"] == "MATERIALIZED"
    assert package_path.is_file()
    body = json.loads(package_path.read_text(encoding="utf-8"))
    assert body["package_identity"] == fake_identity
    assert body["rows"] == fake_rows
    assert result["validation"]["status"] == "VALID"
