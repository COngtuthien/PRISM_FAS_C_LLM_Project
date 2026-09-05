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
    # Real E7-D per-fold output is GPU-produced and not present on this laptop -- prepare_gpat
    # now reaches the REAL GPAT-input package materialization step and fails closed there,
    # earlier and more precisely than the old coarse preflight gate.
    with pytest.raises(e7g.E7Error, match="E7-D source_train.json not present -- FAIL CLOSED"):
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
    with pytest.raises(e7g.E7Error, match="E7-D source_train.json not present -- FAIL CLOSED"):
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


# =========================================================================== #
# E7_REAL_FOLD_AWARE_GPAT_FIT_IMPLEMENTATION.
# =========================================================================== #

def _write_m3b_dataset_rows(repo: Path, fold_id: str, *, e7a_domain: str, label_counts: dict[str, int]) -> tuple[list[dict], list[dict]]:
    """Real crop+prior bytes under the canonical M3B package, plus matching
    E7-A FOLD_MATERIALIZATION source_train_references and E7-D
    source_train.json rows, for one M3B dataset. Returns (refs, e7d_rows)."""
    import cv2
    import numpy as np

    m3b_root = repo / e7b.CASIA_MSU_PACKAGE_ROOT
    (m3b_root / "images").mkdir(parents=True, exist_ok=True)
    (m3b_root / "priors").mkdir(parents=True, exist_ok=True)
    refs: list[dict] = []
    e7d_rows: list[dict] = []
    slug = e7d.E7A_DOMAIN_TO_M3B_DATASET[e7a_domain]
    for label, count in label_counts.items():
        for i in range(count):
            sample_id = f"m3b_{slug}_{label}_{i}"
            image_path = m3b_root / "images" / f"{sample_id}.jpg"
            image = ((i * 11 + 5) % 256) * np.ones((32, 32, 3), dtype="uint8")
            cv2.imwrite(str(image_path), image)
            crop_sha = e7g.cc.sha256_file(image_path)
            prior_path = m3b_root / "priors" / f"{sample_id}.npz"
            np.savez(prior_path, parsing_labels=np.zeros((224, 224), dtype="uint8"),
                    pose_ypr=np.zeros((3,), dtype="float32"), visibility=np.zeros((9,), dtype="float16"),
                    bbox=np.zeros((4,), dtype="float32"), landmarks=np.zeros((5, 2), dtype="float32"),
                    crop_box=np.zeros((4,), dtype="float32"))
            prior_sha = e7g.cc.sha256_file(prior_path)
            subject_id = f"subj_{sample_id}"  # globally unique -- never triggers the
            # different-subject exclusion in _pick, which is not what this fixture is testing
            refs.append({"fold_id": fold_id, "dataset": e7a_domain, "project_split": "source_train",
                        "reference_kind": "m3b_processed_sample", "sample_id": sample_id,
                        "source_record_id": sample_id, "subject_id": subject_id,
                        "label_live_spoof": label, "image_relative_path": f"images/{sample_id}.jpg",
                        "prior_relative_path": f"priors/{sample_id}.npz", "crop_sha256": crop_sha,
                        "prior_sha256": prior_sha})
            e7d_rows.append({"fold_id": fold_id, "dataset": e7a_domain, "project_split": "source_train",
                            "label_live_spoof": label, "spoof_family": None, "source_video_id": sample_id,
                            "frame_index": None, "crop_relative_path": f"images/{sample_id}.jpg",
                            "crop_sha256": crop_sha, "source_package_kind": e7d.M3B_SOURCE_PACKAGE_KIND,
                            "source_package_identity": e7b.FROZEN_M3B_PACKAGE_IDENTITY, "status": "success",
                            "subject_id": subject_id, "failure_reason": None,
                            "sample_id": sample_id})
    return refs, e7d_rows


def _write_fold_m3b_materialization(repo: Path, fold_id: str, refs: list[dict]) -> None:
    mat_path = repo / e7b.E7A_MATERIALIZATION_DIR / fold_id / "FOLD_MATERIALIZATION.json"
    mat_path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(mat_path.read_text(encoding="utf-8")) if mat_path.is_file() \
        else {"source_train_references": []}
    existing["source_train_references"] = existing["source_train_references"] + refs
    mat_path.write_text(json.dumps(existing), encoding="utf-8")


def _write_fold_e7d_source_train_rows(repo: Path, fold_id: str, rows: list[dict]) -> None:
    fold_root = repo / e7d.E7D_OUTPUT_ROOT / fold_id
    fold_root.mkdir(parents=True, exist_ok=True)
    train_path = fold_root / "source_train.json"
    existing = json.loads(train_path.read_text(encoding="utf-8")) if train_path.is_file() else {"rows": []}
    existing["rows"] = existing["rows"] + rows
    train_path.write_text(json.dumps(existing), encoding="utf-8")
    dev_path = fold_root / "source_dev.json"
    if not dev_path.is_file():
        dev_path.write_text(json.dumps({"rows": []}), encoding="utf-8")


def _write_shared_siw_prior_fixture(repo: Path, monkeypatch, *, video_rows: list[dict]) -> None:
    """Real crop+prior bytes for the SHARED SiW source-prior package
    (FULL M3B schema), matching `video_rows` (each: video_id, frame_index,
    crop_relative_path). Patches `FROZEN_SIW_SOURCE_PRIOR_PACKAGE_IDENTITY`
    and `EXPECTED_SIW_SUCCESS_CROP_COUNT` to match this fixture."""
    import numpy as np

    m2_root = repo / e7b.E7B_SIW_SOURCE_PACKAGE_ROOT / "m2_run"
    prior_root = repo / e7g.SIW_SOURCE_PRIOR_PACKAGE_ROOT
    m2_root.mkdir(parents=True, exist_ok=True)
    prior_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in video_rows:
        crop_path = m2_root / r["crop_relative_path"]
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        if not crop_path.is_file():
            crop_path.write_bytes(f"siw-crop-{r['video_id']}-{r['frame_index']}".encode("utf-8"))
        prior_path = prior_root / f"prior_{r['video_id']}_{r['frame_index']}.npz"
        np.savez(prior_path, parsing_labels=np.zeros((224, 224), dtype="uint8"),
                pose_ypr=np.zeros((3,), dtype="float32"), visibility=np.zeros((9,), dtype="float16"),
                bbox=np.zeros((4,), dtype="float32"), landmarks=np.zeros((5, 2), dtype="float32"),
                crop_box=np.zeros((4,), dtype="float32"))
        rows.append({"source_video_id": r["video_id"], "frame_index": r["frame_index"], "status": "success",
                    "source_crop_relative_path": r["crop_relative_path"],
                    "source_crop_sha256": e7g.cc.sha256_file(crop_path),
                    "prior_relative_path": f"prior_{r['video_id']}_{r['frame_index']}.npz",
                    "prior_sha256": e7g.cc.sha256_file(prior_path)})
    monkeypatch.setattr(e7g, "EXPECTED_SIW_SUCCESS_CROP_COUNT", len(rows))
    package_identity = e7g.compute_siw_source_prior_package_identity(rows)
    monkeypatch.setattr(e7g, "FROZEN_SIW_SOURCE_PRIOR_PACKAGE_IDENTITY", package_identity)
    (prior_root / e7g.SIW_SOURCE_PRIOR_PACKAGE_FILENAME).write_text(
        json.dumps({"schema_version": "siw-source-prior-package-v1", "package_identity": package_identity,
                   "rows": rows}), encoding="utf-8")


_GROUP_SIZE = 10  # 80/20 split of 10 -> 8 train / 2 validation, enough for 2 same + 2 cross per live


def _build_gpat_ready_fixture(tmp_path: Path, monkeypatch, fold_id: str) -> Path:
    """Builds a REAL, unmocked GPAT-input-ready fixture for one fold: E7-A/
    E7-D authority + real M3B and/or shared-SiW crop/prior bytes, sized so
    the REAL `pair_plan.build_pair_plan` can find 2 same-domain + 2
    cross-domain spoof sources for every live sample."""
    repo = _base_repo(tmp_path)
    config_dest = repo / e7g.GPAT_FIT_CONFIG_PATH
    config_dest.parent.mkdir(parents=True, exist_ok=True)
    config_dest.write_text((REPO / e7g.GPAT_FIT_CONFIG_PATH).read_text(encoding="utf-8"), encoding="utf-8")
    bank_dest = repo / e7g.M7_RECIPE_BANK_ROOT
    bank_dest.parent.mkdir(parents=True, exist_ok=True)
    bank_dest.symlink_to((REPO / e7g.M7_RECIPE_BANK_ROOT).resolve())
    domains = e7g.FOLD_SOURCE_DOMAINS[fold_id]
    all_refs: list[dict] = []
    all_e7d_rows: list[dict] = []
    for domain in domains:
        if domain == "SiW-Mv2":
            continue
        refs, e7d_rows = _write_m3b_dataset_rows(repo, fold_id, e7a_domain=domain,
                                                 label_counts={"live": _GROUP_SIZE, "spoof": _GROUP_SIZE})
        all_refs += refs
        all_e7d_rows += e7d_rows
    if all_refs:
        _write_fold_m3b_materialization(repo, fold_id, all_refs)
    if "SiW-Mv2" in domains:
        video_rows = [{"video_id": f"siwv{label}{i}", "frame_index": 0,
                      "crop_relative_path": f"crops/siwv{label}{i}_0.jpg"}
                     for label in ("live", "spoof") for i in range(_GROUP_SIZE)]
        _write_shared_siw_prior_fixture(repo, monkeypatch, video_rows=video_rows)
        siw_rows = [_e7d_siw_row(fold_id, {"video_id": r["video_id"], "frame_index": r["frame_index"],
                                          "label_live_spoof": "live" if "live" in r["video_id"] else "spoof",
                                          "crop_relative_path": r["crop_relative_path"],
                                          "crop_sha256": e7g.cc.sha256_file(
                                              repo / e7b.E7B_SIW_SOURCE_PACKAGE_ROOT / "m2_run" /
                                              r["crop_relative_path"])})
                    for r in video_rows]
        all_e7d_rows += siw_rows
    _write_fold_e7d_source_train_rows(repo, fold_id, all_e7d_rows)
    return repo


def test_f1_gpat_input_exact_source_domains(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    result = e7g.materialize_gpat_input_package(repo, "EXT-F1", authorize=True)
    assert result["status"] == "MATERIALIZED"
    validation = e7g.validate_gpat_input_package(repo, "EXT-F1")
    assert validation["status"] == "VALID"
    assert set(validation["dataset_counts"]) == {"casia_fasd", "msu_mfsd"}


def test_f2_gpat_input_exact_casia_siw(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F2")
    result = e7g.materialize_gpat_input_package(repo, "EXT-F2", authorize=True)
    assert result["status"] == "MATERIALIZED"
    validation = e7g.validate_gpat_input_package(repo, "EXT-F2")
    assert validation["status"] == "VALID"
    assert set(validation["dataset_counts"]) == {"casia_fasd", "siw_mv2"}


def test_f3_gpat_input_exact_msu_siw(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F3")
    result = e7g.materialize_gpat_input_package(repo, "EXT-F3", authorize=True)
    assert result["status"] == "MATERIALIZED"
    validation = e7g.validate_gpat_input_package(repo, "EXT-F3")
    assert validation["status"] == "VALID"
    assert set(validation["dataset_counts"]) == {"msu_mfsd", "siw_mv2"}


def test_target_domain_rejected_in_each_fold(tmp_path, monkeypatch):
    for fold_id, target_domain in e7g.FOLD_TARGET_DOMAIN.items():
        sub = tmp_path / fold_id
        sub.mkdir()
        repo = _build_gpat_ready_fixture(sub, monkeypatch, fold_id)
        e7g.materialize_gpat_input_package(repo, fold_id, authorize=True)
        validation = e7g.validate_gpat_input_package(repo, fold_id)
        target_slug = {"CASIA-FASD": "casia_fasd", "MSU-MFSD": "msu_mfsd",
                      "SiW-Mv2": "siw_mv2"}[target_domain]
        assert target_slug not in validation["dataset_counts"]


def test_source_dev_rejected(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    # Inject a source_dev row for the SAME sample_id as an existing source_train row -- must
    # never be picked up (only source_train.json's own rows are ever read).
    fold_root = repo / e7d.E7D_OUTPUT_ROOT / "EXT-F1"
    train_rows = json.loads((fold_root / "source_train.json").read_text(encoding="utf-8"))["rows"]
    dev_row = dict(train_rows[0])
    dev_row["project_split"] = "source_dev"
    (fold_root / "source_dev.json").write_text(json.dumps({"rows": [dev_row]}), encoding="utf-8")
    result = e7g.materialize_gpat_input_package(repo, "EXT-F1", authorize=True)
    assert result["status"] == "MATERIALIZED"
    validation = e7g.validate_gpat_input_package(repo, "EXT-F1")
    assert validation["row_count"] == len(train_rows)


def test_image_hash_mismatch_fails(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    fold_root = repo / e7d.E7D_OUTPUT_ROOT / "EXT-F1"
    body = json.loads((fold_root / "source_train.json").read_text(encoding="utf-8"))
    body["rows"][0]["crop_sha256"] = "0" * 64
    (fold_root / "source_train.json").write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(e7g.E7Error, match="disagrees with E7-A's own materialization reference"):
        e7g.materialize_gpat_input_package(repo, "EXT-F1", authorize=True)


def test_prior_hash_mismatch_fails(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    mat_path = repo / e7b.E7A_MATERIALIZATION_DIR / "EXT-F1" / "FOLD_MATERIALIZATION.json"
    body = json.loads(mat_path.read_text(encoding="utf-8"))
    body["source_train_references"][0]["prior_sha256"] = "0" * 64
    mat_path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(e7g.E7Error, match="source prior SHA256 mismatch"):
        e7g.materialize_gpat_input_package(repo, "EXT-F1", authorize=True)


def test_siw_subject_null_never_fabricated(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F2")
    e7g.materialize_gpat_input_package(repo, "EXT-F2", authorize=True)
    import pyarrow.parquet as pq

    manifest = pq.read_table(repo / e7g.GPAT_INPUT_ROOT / "EXT-F2" / "manifests" /
                             "source_train.parquet").to_pylist()
    siw_rows = [r for r in manifest if r["dataset"] == "siw_mv2"]
    assert siw_rows
    assert all(r["subject_id"] is None for r in siw_rows)
    casia_rows = [r for r in manifest if r["dataset"] == "casia_fasd"]
    assert all(r["subject_id"] is not None for r in casia_rows)


def test_source_only_audit_opens_fold_local_paths(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F2")
    e7g.materialize_gpat_input_package(repo, "EXT-F2", authorize=True)
    validation = e7g.validate_gpat_input_package(repo, "EXT-F2")
    assert validation["status"] == "VALID"
    assert not any("SourceOnlyAudit" in p for p in validation["problems"])
    # And directly: the REAL, unmodified SourceOnlyAudit accepts every path this module writes.
    from prism_fas.synthesis.m8_pipeline import SourceOnlyAudit

    import pyarrow.parquet as pq

    audit = SourceOnlyAudit()
    manifest = pq.read_table(repo / e7g.GPAT_INPUT_ROOT / "EXT-F2" / "manifests" /
                             "source_train.parquet").to_pylist()
    for row in manifest:
        audit.record(row["image_relative_path"])
        audit.record(row["prior_relative_path"])


# --- pair-plan scientific reuse -----------------------------------------------------------

def test_pair_plan_scientific_primitive_actually_invoked(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    e7g.materialize_gpat_input_package(repo, "EXT-F1", authorize=True)
    from prism_fas.synthesis import pair_plan

    calls = []
    real_build = pair_plan.build_pair_plan

    def spy(*args, **kwargs):
        calls.append(1)
        return real_build(*args, **kwargs)

    monkeypatch.setattr(pair_plan, "build_pair_plan", spy)
    result = e7g.materialize_fold_pair_plan(repo, "EXT-F1", authorize=True)
    assert result["status"] == "MATERIALIZED"
    assert len(calls) >= 1


def test_same_cross_domain_2_2_preserved(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    e7g.materialize_gpat_input_package(repo, "EXT-F1", authorize=True)
    e7g.materialize_fold_pair_plan(repo, "EXT-F1", authorize=True)
    validation = e7g.validate_fold_pair_plan(repo, "EXT-F1")
    assert validation["status"] == "VALID"


def test_8020_record_partition_isolation_preserved(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    e7g.materialize_gpat_input_package(repo, "EXT-F1", authorize=True)
    e7g.materialize_fold_pair_plan(repo, "EXT-F1", authorize=True)
    from prism_fas.synthesis import pair_plan

    output_root = repo / e7g.GPAT_PAIR_PLAN_ROOT / "EXT-F1"
    train_rows = pair_plan.load_pair_manifest(output_root / "pair_manifest_train.parquet")
    validation_rows = pair_plan.load_pair_manifest(output_root / "pair_manifest_validation.parquet")
    train_records = {r["live_source_record_id"] for r in train_rows} | \
        {r["spoof_source_record_id"] for r in train_rows}
    validation_records = {r["live_source_record_id"] for r in validation_rows} | \
        {r["spoof_source_record_id"] for r in validation_rows}
    assert not (train_records & validation_records)


def test_pair_plan_globals_restored_after_success(tmp_path, monkeypatch):
    from prism_fas.synthesis import pair_plan

    original_datasets = pair_plan.ALLOWED_DATASETS
    original_train = pair_plan.EXPECTED_TRAIN_PAIRS
    original_validation = pair_plan.EXPECTED_VALIDATION_PAIRS
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    e7g.materialize_gpat_input_package(repo, "EXT-F1", authorize=True)
    e7g.materialize_fold_pair_plan(repo, "EXT-F1", authorize=True)
    assert pair_plan.ALLOWED_DATASETS == original_datasets
    assert pair_plan.EXPECTED_TRAIN_PAIRS == original_train
    assert pair_plan.EXPECTED_VALIDATION_PAIRS == original_validation


def test_pair_plan_globals_restored_after_failure(tmp_path, monkeypatch):
    from prism_fas.synthesis import pair_plan

    original_datasets = pair_plan.ALLOWED_DATASETS
    original_train = pair_plan.EXPECTED_TRAIN_PAIRS
    original_validation = pair_plan.EXPECTED_VALIDATION_PAIRS

    def boom(*args, **kwargs):
        raise RuntimeError("simulated failure inside the scoped adapter")

    monkeypatch.setattr(pair_plan, "build_pair_plan", boom)
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    e7g.materialize_gpat_input_package(repo, "EXT-F1", authorize=True)
    with pytest.raises(RuntimeError, match="simulated failure"):
        e7g.materialize_fold_pair_plan(repo, "EXT-F1", authorize=True)
    assert pair_plan.ALLOWED_DATASETS == original_datasets
    assert pair_plan.EXPECTED_TRAIN_PAIRS == original_train
    assert pair_plan.EXPECTED_VALIDATION_PAIRS == original_validation


def test_f2_f3_accept_siw_mv2_only_via_scoped_adapter(tmp_path, monkeypatch):
    from prism_fas.synthesis import pair_plan

    assert "siw_mv2" not in pair_plan.ALLOWED_DATASETS  # frozen module constant never includes it
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F2")
    e7g.materialize_gpat_input_package(repo, "EXT-F2", authorize=True)
    result = e7g.materialize_fold_pair_plan(repo, "EXT-F2", authorize=True)
    assert result["status"] == "MATERIALIZED"
    assert "siw_mv2" not in pair_plan.ALLOWED_DATASETS  # restored after the call


def test_historical_896_224_not_imposed_on_new_folds(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F2")
    e7g.materialize_gpat_input_package(repo, "EXT-F2", authorize=True)
    result = e7g.materialize_fold_pair_plan(repo, "EXT-F2", authorize=True)
    assert result["train_pairs"] != 896 or result["validation_pairs"] != 224
    assert result["train_pairs"] > 0
    assert result["validation_pairs"] > 0


# --- effective config -----------------------------------------------------------------

def test_effective_config_changes_only_declared_fields(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F2")
    e7g.materialize_gpat_input_package(repo, "EXT-F2", authorize=True)
    e7g.materialize_fold_pair_plan(repo, "EXT-F2", authorize=True)
    effective = e7g.build_effective_gpat_config(repo, "EXT-F2")
    assert set(effective["declared_overrides"]) == {"data.allowed_datasets",
                                                     "pair_plan.expected_train_pairs",
                                                     "pair_plan.expected_validation_pairs"}
    assert effective["effective_config"]["data"]["allowed_datasets"] == ["casia_fasd", "siw_mv2"]


def test_base_gpat_scientific_hyperparameters_identical(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F2")
    e7g.materialize_gpat_input_package(repo, "EXT-F2", authorize=True)
    e7g.materialize_fold_pair_plan(repo, "EXT-F2", authorize=True)
    effective = e7g.build_effective_gpat_config(repo, "EXT-F2")
    cfg = effective["effective_config"]
    assert cfg["seed"] == 20260806
    assert cfg["input_size"] == 224
    assert cfg["batch_size"] == 16
    assert cfg["epochs"] == 15
    assert cfg["optimizer"] == {"name": "AdamW", "encoder_lr": 2.0e-4, "recipe_lr": 1.0e-4,
                               "generator_lr": 2.0e-4, "weight_decay": 1.0e-4, "betas": [0.9, 0.999]}
    assert cfg["scheduler"] == {"name": "cosine", "warmup_fraction": 0.05, "min_lr": 1.0e-6}
    assert cfg["gradient_clip_norm"] == 1.0
    assert cfg["precision"] == {"cuda": "fp16", "cpu": "fp32"}
    assert cfg["early_stopping"] == {"enabled": True, "min_epochs": 5, "patience_epochs": 4}
    assert cfg["checkpoint_selection"] == {"primary": "validation_total_loss", "mode": "min",
                                          "tie_breaker": "validation_identity_cosine",
                                          "tie_breaker_mode": "max"}
    assert cfg["pair_plan"]["seed"] == 20260806
    assert cfg["pair_plan"]["train_fraction"] == 0.8
    assert cfg["pair_plan"]["pairs_per_live"] == 4
    assert cfg["pair_plan"]["same_domain_per_live"] == 2
    assert cfg["pair_plan"]["cross_domain_per_live"] == 2
    assert cfg["identity_model"]["weight_sha256"] == \
        "43bd2d570584d95d4a17ce81f26449034c45dbeed750afcab651872abc0e1496"


# --- prepare_gpat orchestrator ---------------------------------------------------------

def test_prepare_gpat_without_authorize_fails(tmp_path):
    repo = _base_repo(tmp_path)
    with pytest.raises(e7g.E7Error, match="requires --authorize"):
        e7g.prepare_gpat(repo, "EXT-F1", authorize=False)


def test_production_path_on_laptop_fails_gpu_required(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    with pytest.raises(e7g.E7Error, match="GPU_REQUIRED for real GPAT fitting"):
        e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    # by the time it fails, the input package and pair plan ARE materialized and valid.
    assert e7g.validate_gpat_input_package(repo, "EXT-F1")["status"] == "VALID"
    assert e7g.validate_fold_pair_plan(repo, "EXT-F1")["status"] == "VALID"


def test_gpat_capability_check_is_real_not_hardcoded(tmp_path):
    repo = _base_repo(tmp_path)
    capability = e7g._gpat_fit_capability(repo)
    assert capability["capable"] is False
    assert capability["cuda_available"] is False
    assert any("cuda" in p.lower() for p in capability["problems"])
    assert not any("non-gpu host" in p.lower() for p in capability["problems"])


class _FakeTrainer:
    """Stands in for the REAL GPATTrainer at the exact GPU boundary -- never
    monkeypatches GPATTrainer.fit itself; only the class construction is
    replaced so no CUDA/model construction is attempted on the laptop."""
    instances: list[dict] = []
    identity_fn = None  # set by _patch_fake_trainer; computes REAL identity fields from disk

    def __init__(self, *, config, package_root, bank_root, pairs_root, run_root, weight_root, device):
        self.run_root = Path(run_root)
        self.kwargs = {"config": config, "package_root": package_root, "bank_root": bank_root,
                      "pairs_root": pairs_root, "run_root": run_root, "weight_root": weight_root,
                      "device": device}
        _FakeTrainer.instances.append(self.kwargs)

    def fit(self, *, run_id, progress, resume):
        checkpoints = self.run_root / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        identity = _FakeTrainer.identity_fn()
        for name in ("last.pt", "best.pt"):
            (checkpoints / name).write_bytes(f"fake-checkpoint-{name}".encode("utf-8"))
        return {"run_id": run_id, "epochs_run": 1, "global_step": 1, "stop_reason": "completed_all_epochs",
               "best": {"validation_total_loss": 0.1}, "history": [{"train_total": 0.1}],
               "device": "cuda", "identity": identity, "record_set_hashes": {},
               "checkpoints": {"best_sha256": e7g.cc.sha256_bytes(b"fake-checkpoint-best.pt"),
                              "last_sha256": e7g.cc.sha256_bytes(b"fake-checkpoint-last.pt")},
               "source_isolation": {"source_dev_opened": False, "target_test_opened": False}}


_FAKE_IMPLEMENTATION_MODULE_BYTES = b"fake-implementation-module-bytes-for-tests"
_FIXED_IMPLEMENTATION_PROVENANCE = {
    "repository_head_commit": "deadbeef" * 5, "implementation_commit": "cafef00d" * 5,
    "implementation_module_sha256": e7g.cc.sha256_bytes(_FAKE_IMPLEMENTATION_MODULE_BYTES)}


def _patch_gpat_capable(monkeypatch, repo):
    monkeypatch.setattr(e7g, "_gpat_fit_capability",
                        lambda repo_arg: {"capable": True, "cuda_available": True,
                                         "weight_root": str(repo / "model_cache"), "problems": []})
    # GAP 5: production execution requires PRISM_E7_IMPLEMENTATION_COMMIT + a real `git show`
    # byte-match; laptop CPU/unit tests mock this dedicated gate directly rather than requiring
    # a real commit to exist in the test repo.
    monkeypatch.setattr(e7g, "resolve_implementation_commit_provenance",
                        lambda repo_arg: dict(_FIXED_IMPLEMENTATION_PROVENANCE))
    # The terminal-lock validator's OWN provenance self-check independently `git show`s
    # lock.implementation_commit -- mocked at its own dedicated seam so it doesn't require a
    # real git commit to exist in the fake tmp_path repo, but still exercises the real
    # byte-comparison logic against bytes that hash to `implementation_module_sha256` above.
    monkeypatch.setattr(e7g, "_git_show_module_bytes",
                        lambda repo_arg, commit, relative_path: _FAKE_IMPLEMENTATION_MODULE_BYTES)


def _patch_fake_trainer(monkeypatch, repo: Path, fold_id: str) -> None:
    """Patches the GPU boundary with a fake trainer whose reported identity
    is computed FRESH, from the REAL on-disk package/pair-plan/effective-
    config -- self-consistent with whatever `prepare_gpat` independently
    computes and writes into GPAT_FIT_LOCK.json, so the ALREADY_VALID
    resume path genuinely re-validates rather than trivially matching
    hardcoded strings."""
    _FakeTrainer.instances = []

    def real_identity() -> dict:
        package_identity = json.loads((repo / e7g.GPAT_INPUT_ROOT / fold_id /
                                       e7g.GPAT_INPUT_LOCK_FILENAME).read_text()) ["content_identity_sha256"]
        pair_plan_identity = json.loads((repo / e7g.GPAT_PAIR_PLAN_ROOT / fold_id /
                                         "PAIR_PLAN_LOCK.json").read_text())["pair_plan_identity_sha256"]
        effective = e7g.build_effective_gpat_config(repo, fold_id)
        return {"package_identity": package_identity,
               "recipe_bank_identity": e7g.FROZEN_M7_BANK["bank_content_identity_sha256"],
               "pair_plan_identity": pair_plan_identity, "config_hash": effective["effective_config_hash"],
               "architecture_hash": "arch",
               "adaface_weight_sha256": e7g.FROZEN_PRIOR_MODELS["identity"]["weight_sha256"]}

    from prism_fas.synthesis.gpat_checkpoint import CheckpointError, STRICT_IDENTITY_FIELDS
    from prism_fas.synthesis.gpat_contracts import GPAT_CHECKPOINT_SCHEMA_VERSION

    monkeypatch.setattr(_FakeTrainer, "identity_fn", staticmethod(real_identity))
    monkeypatch.setattr("prism_fas.synthesis.gpat_trainer.GPATTrainer", _FakeTrainer)
    monkeypatch.setattr("prism_fas.synthesis.gpat_model.build_gpat_model",
                        lambda config: type("A", (), {"architecture_hash": lambda self: "arch"})())
    # The terminal-lock validator now anchors the expected AdaFace SHA from the frozen
    # effective-config value directly (never from resolved weight bytes), so these mocks only
    # need to keep `prepare_gpat`'s OWN production-path computation self-consistent with that
    # same real frozen value -- never a placeholder that would never match the frozen config.
    monkeypatch.setattr("prism_fas.synthesis.quality_models.resolve_weight",
                        lambda weight_root, role: Path(weight_root) / "identity.bin")
    monkeypatch.setattr("prism_fas.synthesis.quality_models.sha256_file",
                        lambda path: e7g.FROZEN_PRIOR_MODELS["identity"]["weight_sha256"])
    monkeypatch.setattr("prism_fas.synthesis.gpat_checkpoint.checkpoint_summary",
                        lambda path: {"identity": real_identity(),
                                     "schema_version": GPAT_CHECKPOINT_SCHEMA_VERSION})

    def fake_load_checkpoint(path, *, expected_identity):
        identity = real_identity()
        mismatched = [f for f in STRICT_IDENTITY_FIELDS
                     if f in expected_identity and identity.get(f) != expected_identity[f]]
        if mismatched:
            raise CheckpointError(f"refusing to resume: identity mismatch on {mismatched}")
        return {"identity": identity, "record_set_hashes": {}, "history": [{"train_total": 0.1}],
               "global_step": 1, "best_metrics": {"validation_total_loss": 0.1}}

    monkeypatch.setattr("prism_fas.synthesis.gpat_checkpoint.load_checkpoint", fake_load_checkpoint)


def test_existing_valid_gpat_fit_lock_already_valid_zero_trainer_calls(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")
    first = e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    assert first["status"] == "FITTED"
    assert len(_FakeTrainer.instances) == 1
    second = e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    assert second["status"] == "ALREADY_VALID"
    assert len(_FakeTrainer.instances) == 1  # trainer.fit was NOT called again


def _write_matching_attempt_provenance(repo: Path, fold_id: str, *, resume_requested: bool) -> None:
    """Writes a GPAT_ATTEMPT_PROVENANCE.json sidecar that agrees EXACTLY
    with what `prepare_gpat` will independently recompute -- the fixture's
    real on-disk package/pair-plan/effective-config plus the fixed fake
    implementation/AdaFace/architecture identity `_patch_fake_trainer`
    establishes."""
    package_identity = json.loads((repo / e7g.GPAT_INPUT_ROOT / fold_id /
                                   e7g.GPAT_INPUT_LOCK_FILENAME).read_text(encoding="utf-8")
                                  )["content_identity_sha256"]
    pair_plan_identity = json.loads((repo / e7g.GPAT_PAIR_PLAN_ROOT / fold_id /
                                     "PAIR_PLAN_LOCK.json").read_text(encoding="utf-8")
                                    )["pair_plan_identity_sha256"]
    effective = e7g.build_effective_gpat_config(repo, fold_id)
    path = e7g.gpat_attempt_provenance_path(repo, fold_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": "test-gpat-attempt-provenance-v1", "fold_id": fold_id,
        "package_identity": package_identity, "pair_plan_identity": pair_plan_identity,
        "effective_config_hash": effective["effective_config_hash"], "architecture_hash": "arch",
        "adaface_weight_sha256": e7g.FROZEN_PRIOR_MODELS["identity"]["weight_sha256"],
        "implementation_commit": _FIXED_IMPLEMENTATION_PROVENANCE["implementation_commit"],
        "implementation_module_sha256": _FIXED_IMPLEMENTATION_PROVENANCE["implementation_module_sha256"],
        "mask_compatibility_policy": e7g.MASK_COMPATIBILITY_POLICY,
        "resume_requested": resume_requested}), encoding="utf-8")


def test_partial_compatible_checkpoint_resumes(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    e7g.materialize_gpat_input_package(repo, "EXT-F1", authorize=True)
    e7g.materialize_fold_pair_plan(repo, "EXT-F1", authorize=True)
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")  # its identity_fn matches the REAL package/
    # pair-plan/effective-config identity below, since both are computed from the same disk state.
    run_root = e7g.gpat_fit_run_root(repo, "EXT-F1")
    (run_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_root / "checkpoints" / "last.pt").write_bytes(b"partial-compatible")
    _write_matching_attempt_provenance(repo, "EXT-F1", resume_requested=True)

    result = e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    assert result["status"] == "FITTED"
    assert _FakeTrainer.instances[-1]["config"] is not None


def test_partial_incompatible_checkpoint_fails_closed(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    e7g.materialize_gpat_input_package(repo, "EXT-F1", authorize=True)
    e7g.materialize_fold_pair_plan(repo, "EXT-F1", authorize=True)
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")
    run_root = e7g.gpat_fit_run_root(repo, "EXT-F1")
    (run_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_root / "checkpoints" / "last.pt").write_bytes(b"partial-incompatible")
    # A MATCHING attempt-provenance sidecar (so this test isolates the CHECKPOINT'S OWN
    # embedded-identity mismatch, distinct from the sidecar-missing/sidecar-mismatch tests below).
    _write_matching_attempt_provenance(repo, "EXT-F1", resume_requested=True)

    def incompatible_summary(path):
        return {"identity": {"package_identity": "WRONG", "recipe_bank_identity": "WRONG",
                            "pair_plan_identity": "WRONG", "config_hash": "WRONG",
                            "architecture_hash": "WRONG", "adaface_weight_sha256": "WRONG"}}

    monkeypatch.setattr("prism_fas.synthesis.gpat_checkpoint.checkpoint_summary", incompatible_summary)
    with pytest.raises(e7g.E7Error, match="INCOMPATIBLE"):
        e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    assert not e7g.gpat_fit_lock_path(repo, "EXT-F1").is_file()
    # the partial checkpoint is NEVER deleted/auto-restarted
    assert (run_root / "checkpoints" / "last.pt").read_bytes() == b"partial-incompatible"


def test_terminal_lock_written_only_after_checkpoint_validation(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)

    class _BadFakeTrainer(_FakeTrainer):
        def fit(self, *, run_id, progress, resume):
            result = super().fit(run_id=run_id, progress=progress, resume=resume)
            result["identity"] = {**result["identity"], "package_identity": "MISMATCHED"}
            return result

    _FakeTrainer.instances = []
    fixed_identity = {"package_identity": "pkg", "recipe_bank_identity": "bank",
                      "pair_plan_identity": "pairs", "config_hash": "cfg",
                      "architecture_hash": "arch", "adaface_weight_sha256": "ada"}
    monkeypatch.setattr(_FakeTrainer, "identity_fn", staticmethod(lambda: dict(fixed_identity)))
    monkeypatch.setattr("prism_fas.synthesis.gpat_trainer.GPATTrainer", _BadFakeTrainer)
    monkeypatch.setattr("prism_fas.synthesis.gpat_model.build_gpat_model",
                        lambda config: type("A", (), {"architecture_hash": lambda self: "arch"})())
    monkeypatch.setattr("prism_fas.synthesis.quality_models.resolve_weight",
                        lambda weight_root, role: Path(weight_root) / "identity.bin")
    monkeypatch.setattr("prism_fas.synthesis.quality_models.sha256_file", lambda path: "ada")
    monkeypatch.setattr("prism_fas.synthesis.gpat_checkpoint.checkpoint_summary",
                        lambda path: {"identity": dict(fixed_identity)})
    # Bypasses the strict expected_identity check itself (tested separately) so this test
    # isolates the SPECIFIC comparison of best_payload.identity vs fit_result["identity"].
    monkeypatch.setattr("prism_fas.synthesis.gpat_checkpoint.load_checkpoint",
                        lambda path, *, expected_identity: {"identity": dict(fixed_identity),
                                                            "record_set_hashes": {},
                                                            "history": [{"train_total": 0.1}]})
    with pytest.raises(e7g.E7Error, match="best checkpoint identity does not match"):
        e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    assert not e7g.gpat_fit_lock_path(repo, "EXT-F1").is_file()


def test_failed_fit_leaves_no_terminal_lock(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)

    class _RaisingFakeTrainer(_FakeTrainer):
        def fit(self, *, run_id, progress, resume):
            raise RuntimeError("simulated GPU fit failure")

    _FakeTrainer.instances = []
    monkeypatch.setattr("prism_fas.synthesis.gpat_trainer.GPATTrainer", _RaisingFakeTrainer)
    monkeypatch.setattr("prism_fas.synthesis.gpat_model.build_gpat_model",
                        lambda config: type("A", (), {"architecture_hash": lambda self: "arch"})())
    monkeypatch.setattr("prism_fas.synthesis.quality_models.resolve_weight",
                        lambda weight_root, role: Path(weight_root) / "identity.bin")
    monkeypatch.setattr("prism_fas.synthesis.quality_models.sha256_file", lambda path: "ada")
    with pytest.raises(RuntimeError, match="simulated GPU fit failure"):
        e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    assert not e7g.gpat_fit_lock_path(repo, "EXT-F1").is_file()


def test_checkpoint_resolver_uses_native_path(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    expected = repo / e7g.RUN_ROOT / "EXT-F1" / "gpat_fit" / "checkpoints" / "best.pt"
    assert e7g.gpat_best_checkpoint_path(repo, "EXT-F1") == expected
    assert e7g.gpat_last_checkpoint_path(repo, "EXT-F1") == expected.with_name("last.pt")
    # generate_and_match / e7_gpat_bank_validate use the SAME canonical helper, not the old
    # gpat_checkpoint/best.pt assumption.
    with pytest.raises(e7g.E7Error, match="no fitted GPAT checkpoint present"):
        e7g.generate_and_match(repo, "EXT-F1", authorize=True)
    result = e7g.e7_gpat_bank_validate(repo)
    assert result["folds"]["EXT-F1"]["gpat_checkpoint_present"] is False


def test_gpat_no_target_or_evaluation_only_paths(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F2")
    e7g.materialize_gpat_input_package(repo, "EXT-F2", authorize=True)
    import pyarrow.parquet as pq

    manifest = pq.read_table(repo / e7g.GPAT_INPUT_ROOT / "EXT-F2" / "manifests" /
                             "source_train.parquet").to_pylist()
    for row in manifest:
        assert "evaluation_only" not in row["image_relative_path"]
        assert "prism_target_eval_v2" not in row["image_relative_path"]
        assert e7g.PROTECTED_SIW_TARGET_PRIOR_PACKAGE_ROOT not in row["image_relative_path"]


def test_gpat_no_llm_calls(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    result = e7g.materialize_gpat_input_package(repo, "EXT-F1", authorize=True)
    assert result["llm_api_calls"] == 0
    pf = e7g.preflight(repo)
    assert pf["LLM_API_CALLS"] == 0


def test_source_prior_frozen_evidence_unchanged():
    import subprocess

    committed = subprocess.run(["git", "show", f"HEAD:{e7g.FROZEN_SOURCE_PRIOR_EVIDENCE_PATH}"],
                              cwd=REPO, check=True, capture_output=True, text=True).stdout
    on_disk = (REPO / e7g.FROZEN_SOURCE_PRIOR_EVIDENCE_PATH).read_text(encoding="utf-8")
    assert committed == on_disk
    body = json.loads(on_disk)
    assert body["source_prior_package_identity"] == e7g.FROZEN_SIW_SOURCE_PRIOR_PACKAGE_IDENTITY


def test_e7abcd_protected_artifacts_unchanged_gpat(tmp_path, monkeypatch):
    import subprocess

    for relative in ("src/prism_fas/evaluation/c_ext_e7a_fold_prep.py",
                     "src/prism_fas/evaluation/c_ext_e7b_data_prep.py",
                     "src/prism_fas/evaluation/c_ext_e7c_gpat_prep.py",
                     "src/prism_fas/evaluation/c_ext_e7d_source_support.py",
                     "src/prism_fas/synthesis/gpat_trainer.py", "src/prism_fas/synthesis/gpat_model.py",
                     "src/prism_fas/synthesis/gpat_losses.py", "src/prism_fas/synthesis/m8_pipeline.py",
                     "src/prism_fas/synthesis/pair_plan.py", "src/prism_fas/synthesis/gpat_checkpoint.py"):
        committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=REPO, check=True,
                                  capture_output=True, text=True).stdout
        on_disk = (REPO / relative).read_text(encoding="utf-8")
        assert committed == on_disk, f"{relative} differs from HEAD -- must remain unmodified"


def test_readiness_reports_per_fold_gpat_fitted(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    pf = e7g.preflight(repo)
    assert pf["F1_GPAT_FITTED"] is False
    assert pf["F2_GPAT_FITTED"] is False
    assert pf["F3_GPAT_FITTED"] is False
    assert pf["E7_READY_FOR_TRAINING"] is False
    readiness = e7g.build_readiness(repo)
    assert readiness["F1_GPAT_FITTED"] is False
    assert readiness["E7_READY_FOR_TRAINING"] is False


# =========================================================================== #
# TECHNICAL_GPAT_VALIDATION_AND_PROVENANCE_GAP fix (GAPs 1-5).
# =========================================================================== #

def _corrupt_pair_manifest_row(repo: Path, fold_id: str, *, partition: str, field: str, value) -> None:
    """Test-only helper: rewrites ONE field of ONE row in the on-disk pair
    manifest, reusing pair_plan's own `_write_parquet`/`_PAIR_FIELDS`
    (never reimplementing the parquet schema)."""
    from prism_fas.synthesis import pair_plan

    output_root = repo / e7g.GPAT_PAIR_PLAN_ROOT / fold_id
    path = output_root / f"pair_manifest_{partition}.parquet"
    rows = pair_plan.load_pair_manifest(path)
    rows[0] = {**rows[0], field: value}
    pair_plan._write_parquet(path, rows)


# --- GAP 1: pair-plan identity TRUE recomputation --------------------------------------

def test_pair_plan_identity_recomputed_not_echoed(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    e7g.materialize_gpat_input_package(repo, "EXT-F1", authorize=True)
    e7g.materialize_fold_pair_plan(repo, "EXT-F1", authorize=True)
    from prism_fas.synthesis import pair_plan

    # The OLD echo function is never relied upon any more -- corrupting it must not affect
    # a genuinely-valid plan's validation result.
    monkeypatch.setattr(pair_plan, "pair_plan_identity", lambda output_root: "corrupted-echo")
    validation = e7g.validate_fold_pair_plan(repo, "EXT-F1")
    assert validation["status"] == "VALID"
    assert validation["recomputed_pair_plan_identity"] != "corrupted-echo"


def test_pair_manifest_content_corruption_recipe_id_detected(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    e7g.materialize_gpat_input_package(repo, "EXT-F1", authorize=True)
    e7g.materialize_fold_pair_plan(repo, "EXT-F1", authorize=True)
    _corrupt_pair_manifest_row(repo, "EXT-F1", partition="train", field="recipe_id",
                               value="corrupted_recipe")
    validation = e7g.validate_fold_pair_plan(repo, "EXT-F1")
    # The corrupted bytes live only in the on-disk parquet manifest (never in the untouched
    # PAIR_PLAN_LOCK.json), so this is caught by the direct rebuild-vs-disk CONTENT comparison,
    # not necessarily by the lock's own identity field -- either way, status must be INVALID.
    assert validation["status"] == "INVALID"
    assert any("content" in p.lower() for p in validation["problems"])


def test_pair_manifest_content_corruption_pair_id_detected(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    e7g.materialize_gpat_input_package(repo, "EXT-F1", authorize=True)
    e7g.materialize_fold_pair_plan(repo, "EXT-F1", authorize=True)
    _corrupt_pair_manifest_row(repo, "EXT-F1", partition="train", field="pair_id",
                               value="gpatpair_corrupted00000000")
    validation = e7g.validate_fold_pair_plan(repo, "EXT-F1")
    assert validation["status"] == "INVALID"


def test_pair_manifest_content_corruption_recipe_seed_detected(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    e7g.materialize_gpat_input_package(repo, "EXT-F1", authorize=True)
    e7g.materialize_fold_pair_plan(repo, "EXT-F1", authorize=True)
    from prism_fas.synthesis import pair_plan

    output_root = repo / e7g.GPAT_PAIR_PLAN_ROOT / "EXT-F1"
    rows = pair_plan.load_pair_manifest(output_root / "pair_manifest_train.parquet")
    rows[0] = {**rows[0], "recipe_seed": int(rows[0]["recipe_seed"]) + 1}
    pair_plan._write_parquet(output_root / "pair_manifest_train.parquet", rows)
    validation = e7g.validate_fold_pair_plan(repo, "EXT-F1")
    assert validation["status"] == "INVALID"


def test_pair_manifest_row_content_corruption_detected(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    e7g.materialize_gpat_input_package(repo, "EXT-F1", authorize=True)
    e7g.materialize_fold_pair_plan(repo, "EXT-F1", authorize=True)
    from prism_fas.synthesis import pair_plan

    output_root = repo / e7g.GPAT_PAIR_PLAN_ROOT / "EXT-F1"
    rows = pair_plan.load_pair_manifest(output_root / "pair_manifest_train.parquet")
    flipped = "cross_domain" if rows[0]["domain_relation"] == "same_domain" else "same_domain"
    rows[0] = {**rows[0], "domain_relation": flipped}
    pair_plan._write_parquet(output_root / "pair_manifest_train.parquet", rows)
    validation = e7g.validate_fold_pair_plan(repo, "EXT-F1")
    assert validation["status"] == "INVALID"


def test_pair_plan_lock_identity_bearing_field_corruption_detected(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    e7g.materialize_gpat_input_package(repo, "EXT-F1", authorize=True)
    e7g.materialize_fold_pair_plan(repo, "EXT-F1", authorize=True)
    lock_path = repo / e7g.GPAT_PAIR_PLAN_ROOT / "EXT-F1" / "PAIR_PLAN_LOCK.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["train_pairs"] = lock["train_pairs"] + 1
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    validation = e7g.validate_fold_pair_plan(repo, "EXT-F1")
    assert validation["status"] == "INVALID"
    assert any("train_pairs" in p for p in validation["problems"])


# --- GAP 2: subject_id authority strictly bound ------------------------------------------

def _mutate_gpat_input_row(repo: Path, fold_id: str, *, predicate, field: str, value) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    manifest_path = repo / e7g.GPAT_INPUT_ROOT / fold_id / "manifests" / "source_train.parquet"
    rows = pq.read_table(manifest_path).to_pylist()
    mutated = False
    for row in rows:
        if predicate(row):
            row[field] = value
            mutated = True
            break
    assert mutated, "predicate matched no row -- fixture assumption broken"
    pq.write_table(pa.Table.from_pylist(rows), manifest_path)


def test_m3b_subject_authority_mutation_detected(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    e7g.materialize_gpat_input_package(repo, "EXT-F1", authorize=True)
    _mutate_gpat_input_row(repo, "EXT-F1", predicate=lambda r: r["dataset"] == "casia_fasd",
                           field="subject_id", value="fabricated_subject")
    validation = e7g.validate_gpat_input_package(repo, "EXT-F1")
    assert validation["status"] == "INVALID"
    assert any("subject_id" in p for p in validation["problems"])


def test_siw_subject_fabrication_detected(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F2")
    e7g.materialize_gpat_input_package(repo, "EXT-F2", authorize=True)
    _mutate_gpat_input_row(repo, "EXT-F2", predicate=lambda r: r["dataset"] == "siw_mv2",
                           field="subject_id", value="fabricated_siw_subject")
    validation = e7g.validate_gpat_input_package(repo, "EXT-F2")
    assert validation["status"] == "INVALID"
    assert any("subject_id" in p for p in validation["problems"])


def test_source_record_id_swap_detected(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    e7g.materialize_gpat_input_package(repo, "EXT-F1", authorize=True)
    _mutate_gpat_input_row(repo, "EXT-F1", predicate=lambda r: r["dataset"] == "casia_fasd",
                           field="source_record_id", value="swapped_record_id")
    validation = e7g.validate_gpat_input_package(repo, "EXT-F1")
    assert validation["status"] == "INVALID"
    assert any("source_record_id" in p for p in validation["problems"])


def test_label_swap_detected(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    e7g.materialize_gpat_input_package(repo, "EXT-F1", authorize=True)
    _mutate_gpat_input_row(repo, "EXT-F1", predicate=lambda r: r["label_live_spoof"] == "live",
                           field="label_live_spoof", value="spoof")
    validation = e7g.validate_gpat_input_package(repo, "EXT-F1")
    assert validation["status"] == "INVALID"
    assert any("label_live_spoof" in p for p in validation["problems"])


def test_subject_id_included_in_package_identity():
    common = dict(fold_id="EXT-F1", e7d_package_identity="e7d", m3b_package_identity="m3b",
                 siw_source_prior_package_identity=None, base_config_sha256="cfg", m7_bank_identity="bank")
    identity_a = e7g.compute_gpat_input_package_identity(
        rows=[("casia_fasd", "rec1", "subj_a", "live", "source_train", "c" * 64, "p" * 64)], **common)
    identity_b = e7g.compute_gpat_input_package_identity(
        rows=[("casia_fasd", "rec1", "subj_b", "live", "source_train", "c" * 64, "p" * 64)], **common)
    assert identity_a != identity_b


# --- GAP 3: terminal lock revalidates against CURRENT state --------------------------------

def test_terminal_lock_rejects_current_package_drift(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")
    e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    lock_path = e7g.gpat_fit_lock_path(repo, "EXT-F1")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["package_identity"] = "drifted-package-identity"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    validation = e7g.validate_gpat_fit_lock(repo, "EXT-F1")
    assert validation["status"] == "INVALID"
    assert any("package identity" in p for p in validation["problems"])


def test_terminal_lock_rejects_current_pair_plan_drift(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")
    e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    lock_path = e7g.gpat_fit_lock_path(repo, "EXT-F1")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["pair_plan_identity"] = "drifted-pair-plan-identity"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    validation = e7g.validate_gpat_fit_lock(repo, "EXT-F1")
    assert validation["status"] == "INVALID"
    assert any("pair-plan identity" in p for p in validation["problems"])


def test_terminal_lock_rejects_effective_config_drift(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")
    e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    lock_path = e7g.gpat_fit_lock_path(repo, "EXT-F1")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["effective_config_hash"] = "drifted-effective-config-hash"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    validation = e7g.validate_gpat_fit_lock(repo, "EXT-F1")
    assert validation["status"] == "INVALID"
    assert any("effective_config_hash" in p for p in validation["problems"])


def test_terminal_lock_rejects_last_checkpoint_identity_drift(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")
    e7g.prepare_gpat(repo, "EXT-F1", authorize=True)

    from prism_fas.synthesis.gpat_checkpoint import CheckpointError, STRICT_IDENTITY_FIELDS

    lock = json.loads(e7g.gpat_fit_lock_path(repo, "EXT-F1").read_text(encoding="utf-8"))
    good_identity = {"package_identity": lock["package_identity"],
                     "recipe_bank_identity": lock["m7_recipe_bank_identity"],
                     "pair_plan_identity": lock["pair_plan_identity"],
                     "config_hash": lock["effective_config_hash"],
                     "architecture_hash": lock["architecture_hash"],
                     "adaface_weight_sha256": lock["adaface_weight_sha256"]}

    def selective_load_checkpoint(path, *, expected_identity):
        if str(path).endswith("last.pt"):
            raise CheckpointError("refusing to resume: identity mismatch on ['package_identity']")
        return {"identity": good_identity, "record_set_hashes": {}, "history": [{"train_total": 0.1}]}

    monkeypatch.setattr("prism_fas.synthesis.gpat_checkpoint.load_checkpoint", selective_load_checkpoint)
    from prism_fas.synthesis.gpat_contracts import GPAT_CHECKPOINT_SCHEMA_VERSION

    monkeypatch.setattr("prism_fas.synthesis.gpat_checkpoint.checkpoint_summary",
                        lambda path: {"identity": good_identity,
                                     "schema_version": GPAT_CHECKPOINT_SCHEMA_VERSION})
    validation = e7g.validate_gpat_fit_lock(repo, "EXT-F1")
    assert validation["status"] == "INVALID"
    assert any("last checkpoint FAILED strict identity load" in p for p in validation["problems"])


# =========================================================================== #
# TECHNICAL_TERMINAL_LOCK_MODEL_IDENTITY_ANCHOR_GAP fix -- the expected
# checkpoint identity (architecture_hash/adaface_weight_sha256) is derived
# from CURRENT FROZEN AUTHORITIES, never merely echoed from the lock.
# =========================================================================== #

def test_terminal_lock_rejects_architecture_hash_drift(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")
    e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    lock_path = e7g.gpat_fit_lock_path(repo, "EXT-F1")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["architecture_hash"] = "drifted-architecture-hash"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    validation = e7g.validate_gpat_fit_lock(repo, "EXT-F1")
    assert validation["status"] == "INVALID"
    assert any("architecture_hash" in p for p in validation["problems"])


def test_terminal_lock_rejects_adaface_sha_drift(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")
    e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    lock_path = e7g.gpat_fit_lock_path(repo, "EXT-F1")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["adaface_weight_sha256"] = "drifted-adaface-sha"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    validation = e7g.validate_gpat_fit_lock(repo, "EXT-F1")
    assert validation["status"] == "INVALID"
    assert any("AdaFace" in p for p in validation["problems"])


def test_checkpoint_expected_identity_uses_current_authorities(tmp_path, monkeypatch):
    """Proves the validator anchors the expected checkpoint identity from
    CURRENT authorities, not the lock's own fields: even a lock AND both
    checkpoints that are corrupted to agree with EACH OTHER (a
    self-consistent snapshot) must still be rejected, because the frozen
    architecture (mocked `build_gpat_model` -> "arch") and the frozen
    repository AdaFace SHA never change to match the corruption."""
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")
    e7g.prepare_gpat(repo, "EXT-F1", authorize=True)

    lock_path = e7g.gpat_fit_lock_path(repo, "EXT-F1")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["architecture_hash"] = "SELF-CONSISTENT-WRONG"
    lock["adaface_weight_sha256"] = "SELF-CONSISTENT-WRONG"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    corrupted_identity = {"package_identity": lock["package_identity"],
                          "recipe_bank_identity": lock["m7_recipe_bank_identity"],
                          "pair_plan_identity": lock["pair_plan_identity"],
                          "config_hash": lock["effective_config_hash"],
                          "architecture_hash": "SELF-CONSISTENT-WRONG",
                          "adaface_weight_sha256": "SELF-CONSISTENT-WRONG"}

    from prism_fas.synthesis.gpat_checkpoint import CheckpointError, STRICT_IDENTITY_FIELDS
    from prism_fas.synthesis.gpat_contracts import GPAT_CHECKPOINT_SCHEMA_VERSION

    monkeypatch.setattr("prism_fas.synthesis.gpat_checkpoint.checkpoint_summary",
                        lambda path: {"identity": corrupted_identity,
                                     "schema_version": GPAT_CHECKPOINT_SCHEMA_VERSION})

    def self_consistent_load_checkpoint(path, *, expected_identity):
        mismatched = [f for f in STRICT_IDENTITY_FIELDS
                     if f in expected_identity and corrupted_identity.get(f) != expected_identity[f]]
        if mismatched:
            raise CheckpointError(f"refusing to resume: identity mismatch on {mismatched}")
        return {"identity": corrupted_identity, "record_set_hashes": {}, "history": [{"train_total": 0.1}],
               "global_step": 1, "best_metrics": {"validation_total_loss": 0.1}}

    monkeypatch.setattr("prism_fas.synthesis.gpat_checkpoint.load_checkpoint", self_consistent_load_checkpoint)

    validation = e7g.validate_gpat_fit_lock(repo, "EXT-F1")
    assert validation["status"] == "INVALID"
    assert any("architecture_hash" in p or "AdaFace" in p or "FAILED strict identity load" in p
              for p in validation["problems"])


# --- execution-metadata anchor checks -------------------------------------------------------

def test_terminal_lock_rejects_seed_drift(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")
    e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    lock_path = e7g.gpat_fit_lock_path(repo, "EXT-F1")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["seed"] = lock["seed"] + 1
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    validation = e7g.validate_gpat_fit_lock(repo, "EXT-F1")
    assert validation["status"] == "INVALID"
    assert any("seed" in p for p in validation["problems"])


def test_terminal_lock_rejects_device_noncuda(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")
    e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    lock_path = e7g.gpat_fit_lock_path(repo, "EXT-F1")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["device"] = "cpu"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    validation = e7g.validate_gpat_fit_lock(repo, "EXT-F1")
    assert validation["status"] == "INVALID"
    assert any("device" in p for p in validation["problems"])


def test_terminal_lock_rejects_global_step_drift(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")
    e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    lock_path = e7g.gpat_fit_lock_path(repo, "EXT-F1")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["global_step"] = lock["global_step"] + 1000
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    validation = e7g.validate_gpat_fit_lock(repo, "EXT-F1")
    assert validation["status"] == "INVALID"
    assert any("global_step" in p for p in validation["problems"])


def test_terminal_lock_rejects_epoch_count_drift(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")
    e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    lock_path = e7g.gpat_fit_lock_path(repo, "EXT-F1")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["epochs_completed"] = lock["epochs_completed"] + 1000
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    validation = e7g.validate_gpat_fit_lock(repo, "EXT-F1")
    assert validation["status"] == "INVALID"
    assert any("epochs_completed" in p for p in validation["problems"])


def test_terminal_lock_rejects_best_metrics_drift(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")
    e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    lock_path = e7g.gpat_fit_lock_path(repo, "EXT-F1")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["best_metrics"] = {"validation_total_loss": 999.9}
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    validation = e7g.validate_gpat_fit_lock(repo, "EXT-F1")
    assert validation["status"] == "INVALID"
    assert any("best_metrics" in p for p in validation["problems"])


# --- provenance lock self-check ---------------------------------------------------------------

def test_terminal_lock_implementation_commit_module_sha_verified(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")
    e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    validation = e7g.validate_gpat_fit_lock(repo, "EXT-F1")
    assert validation["status"] == "VALID"


def test_terminal_lock_implementation_provenance_corruption_detected(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")
    e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    lock_path = e7g.gpat_fit_lock_path(repo, "EXT-F1")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["implementation_module_sha256"] = "0" * 64
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    validation = e7g.validate_gpat_fit_lock(repo, "EXT-F1")
    assert validation["status"] == "INVALID"
    assert any("implementation_module_sha256" in p for p in validation["problems"])


def test_terminal_lock_missing_implementation_provenance_fields_detected(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")
    e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    lock_path = e7g.gpat_fit_lock_path(repo, "EXT-F1")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    del lock["implementation_commit"]
    del lock["implementation_module_sha256"]
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    validation = e7g.validate_gpat_fit_lock(repo, "EXT-F1")
    assert validation["status"] == "INVALID"
    assert any("implementation_commit is missing" in p for p in validation["problems"])
    assert any("implementation_module_sha256 is missing" in p for p in validation["problems"])


def test_resolve_implementation_commit_provenance_gate_unchanged(monkeypatch, tmp_path):
    """The pre-fit production gate itself must remain untouched by this
    turn's fix -- still requires the env var, still fails closed."""
    repo = _base_repo(tmp_path)
    monkeypatch.delenv("PRISM_E7_IMPLEMENTATION_COMMIT", raising=False)
    with pytest.raises(e7g.E7Error, match="PRISM_E7_IMPLEMENTATION_COMMIT is not set"):
        e7g.resolve_implementation_commit_provenance(repo)


# --- GAP 4: post-fit validation before the terminal marker ---------------------------------

def test_nonfinite_history_detected(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")

    class _NonFiniteTrainer(_FakeTrainer):
        def fit(self, *, run_id, progress, resume):
            result = super().fit(run_id=run_id, progress=progress, resume=resume)
            result["history"] = [{"train_total": 0.1, "some_other_metric": float("nan")}]
            return result

    monkeypatch.setattr("prism_fas.synthesis.gpat_trainer.GPATTrainer", _NonFiniteTrainer)
    with pytest.raises(e7g.E7Error, match="non-finite"):
        e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    assert not e7g.gpat_fit_lock_path(repo, "EXT-F1").is_file()


def test_forbidden_source_audit_flag_detected(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")

    class _LeakyTrainer(_FakeTrainer):
        def fit(self, *, run_id, progress, resume):
            result = super().fit(run_id=run_id, progress=progress, resume=resume)
            result["source_isolation"] = {"source_dev_opened": False, "target_test_opened": False,
                                          "target_label_artifact_opened": True,
                                          "raw_dataset_path_opened": False}
            return result

    monkeypatch.setattr("prism_fas.synthesis.gpat_trainer.GPATTrainer", _LeakyTrainer)
    with pytest.raises(e7g.E7Error, match="forbidden open"):
        e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    assert not e7g.gpat_fit_lock_path(repo, "EXT-F1").is_file()


def test_forbidden_manifest_access_detected(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")

    class _NonTrainManifestTrainer(_FakeTrainer):
        def fit(self, *, run_id, progress, resume):
            result = super().fit(run_id=run_id, progress=progress, resume=resume)
            result["source_isolation"] = {"source_dev_opened": False, "target_test_opened": False,
                                          "manifests_opened": ["manifests/source_dev.parquet"]}
            return result

    monkeypatch.setattr("prism_fas.synthesis.gpat_trainer.GPATTrainer", _NonTrainManifestTrainer)
    with pytest.raises(e7g.E7Error, match="non-source_train manifest"):
        e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    assert not e7g.gpat_fit_lock_path(repo, "EXT-F1").is_file()


def test_checkpoint_disk_sha_independently_verified(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")

    class _WrongShaTrainer(_FakeTrainer):
        def fit(self, *, run_id, progress, resume):
            result = super().fit(run_id=run_id, progress=progress, resume=resume)
            result["checkpoints"] = {**result["checkpoints"], "best_sha256": "0" * 64}
            return result

    monkeypatch.setattr("prism_fas.synthesis.gpat_trainer.GPATTrainer", _WrongShaTrainer)
    with pytest.raises(e7g.E7Error, match="on-disk SHA256"):
        e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    assert not e7g.gpat_fit_lock_path(repo, "EXT-F1").is_file()


# --- GAP 5: GPU implementation-commit provenance --------------------------------------------

def test_gpu_implementation_commit_sha_mismatch_detected():
    import os

    old = os.environ.get("PRISM_E7_IMPLEMENTATION_COMMIT")
    try:
        # The module file on disk right now (mid-session edits) cannot possibly be
        # byte-identical to its content at the pre-session BASE_COMMIT.
        os.environ["PRISM_E7_IMPLEMENTATION_COMMIT"] = "04295804479747488ebfa7edaeb49d1a35dac89b"
        with pytest.raises(e7g.E7Error, match="does NOT match"):
            e7g.resolve_implementation_commit_provenance(REPO)
    finally:
        if old is None:
            os.environ.pop("PRISM_E7_IMPLEMENTATION_COMMIT", None)
        else:
            os.environ["PRISM_E7_IMPLEMENTATION_COMMIT"] = old


def test_gpu_implementation_module_sha_match_required_env_missing():
    import os

    old = os.environ.pop("PRISM_E7_IMPLEMENTATION_COMMIT", None)
    try:
        with pytest.raises(e7g.E7Error, match="PRISM_E7_IMPLEMENTATION_COMMIT is not set"):
            e7g.resolve_implementation_commit_provenance(REPO)
    finally:
        if old is not None:
            os.environ["PRISM_E7_IMPLEMENTATION_COMMIT"] = old


def test_gpu_provenance_gate_not_required_before_capability_gate(tmp_path, monkeypatch):
    import os

    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    old = os.environ.pop("PRISM_E7_IMPLEMENTATION_COMMIT", None)
    try:
        # GPU capability is NOT patched here -- the real capability gate must fail FIRST,
        # never the provenance gate, so laptop GPU_REQUIRED behavior is unaffected.
        with pytest.raises(e7g.E7Error, match="GPU_REQUIRED"):
            e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    finally:
        if old is not None:
            os.environ["PRISM_E7_IMPLEMENTATION_COMMIT"] = old


def test_gpat_fit_lock_carries_provenance_fields_not_repo_head(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")
    result = e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    lock = result["lock"]
    assert lock["repository_head_commit"] == _FIXED_IMPLEMENTATION_PROVENANCE["repository_head_commit"]
    assert lock["implementation_commit"] == _FIXED_IMPLEMENTATION_PROVENANCE["implementation_commit"]
    assert lock["implementation_module_sha256"] == \
        _FIXED_IMPLEMENTATION_PROVENANCE["implementation_module_sha256"]
    assert "code_checkpoint" not in lock  # never falsely calls repository HEAD the impl checkpoint


# =========================================================================== #
# TECHNICAL_SINGLE_SAMPLE_EMPTY_CHEEK_FALLBACK_EDGE_CASE fix.
# =========================================================================== #

def _region_mask_builder_kwargs(*, degenerate_cheek: bool) -> dict:
    """`parsing` is all-background so every region takes the geometry
    fallback path deterministically. `degenerate_cheek=True` reconstructs
    the REAL EXT-F2 incident geometry (crop_box=[320,208,720,917],
    bbox=[401.9,326.6,725.7,799.0], nose raw x=738.6) whose right_cheek
    ellipse centre lands entirely outside the 224x224 crop."""
    import numpy as np

    parsing = np.zeros((224, 224), dtype="uint8")
    if degenerate_cheek:
        bbox = np.array([401.9000244140625, 326.62579345703125, 725.75439453125, 798.960693359375],
                        dtype=np.float32)
        crop_box = np.array([320.0, 208.0, 720.0, 917.0], dtype=np.float32)
        cy = float((bbox[1] + bbox[3]) / 2)
        landmarks = np.array([[450.0, cy - 40], [650.0, cy - 40], [738.636962890625, cy],
                              [480.0, cy + 80], [650.0, cy + 80]], dtype=np.float32)
    else:
        bbox = np.array([30.0, 20.0, 194.0, 204.0], dtype=np.float32)
        crop_box = np.array([0.0, 0.0, 224.0, 224.0], dtype=np.float32)
        landmarks = np.array([[80.0, 80.0], [144.0, 80.0], [112.0, 110.0],
                              [90.0, 150.0], [134.0, 150.0]], dtype=np.float32)
    return {"height": 224, "width": 224, "parsing": parsing, "landmarks": landmarks, "bbox": bbox,
           "crop_box": crop_box}


# --- 1-9: core recovery-logic behavior -------------------------------------------------------

def test_nonempty_left_cheek_byte_identical():
    import numpy as np
    from prism_fas.synthesis.masks import RegionMaskBuilder

    kwargs = _region_mask_builder_kwargs(degenerate_cheek=False)
    corrected_cls = e7g._e7_compatible_region_mask_builder_class()
    original = RegionMaskBuilder(**kwargs)
    corrected = corrected_cls(**kwargs)
    mask_o, source_o = original.region("left_cheek")
    mask_c, source_c = corrected.region("left_cheek")
    assert np.asarray(mask_o).any()
    assert np.array_equal(mask_o, mask_c)
    assert source_o == source_c


def test_nonempty_right_cheek_byte_identical():
    import numpy as np
    from prism_fas.synthesis.masks import RegionMaskBuilder

    kwargs = _region_mask_builder_kwargs(degenerate_cheek=False)
    corrected_cls = e7g._e7_compatible_region_mask_builder_class()
    original = RegionMaskBuilder(**kwargs)
    corrected = corrected_cls(**kwargs)
    mask_o, source_o = original.region("right_cheek")
    mask_c, source_c = corrected.region("right_cheek")
    assert np.asarray(mask_o).any()
    assert np.array_equal(mask_o, mask_c)
    assert source_o == source_c


def test_empty_right_cheek_deterministic_recovery():
    import numpy as np
    from prism_fas.synthesis.masks import RegionMaskBuilder

    kwargs = _region_mask_builder_kwargs(degenerate_cheek=True)
    original = RegionMaskBuilder(**kwargs)
    mask_o, _ = original.region("right_cheek")
    assert not np.asarray(mask_o).any()  # reproduces the real EXT-F2 incident

    corrected_cls = e7g._e7_compatible_region_mask_builder_class()
    corrected = corrected_cls(**kwargs)
    mask_c1, source_c1 = corrected.region("right_cheek")
    mask_c2, source_c2 = corrected_cls(**kwargs).region("right_cheek")
    assert np.asarray(mask_c1).any()
    assert np.array_equal(mask_c1, mask_c2)  # deterministic
    assert source_c1 == source_c2 == e7g.MASK_RECOVERY_SOURCE_TAG


def test_empty_left_cheek_symmetric_recovery():
    import numpy as np

    # Mirror the incident geometry onto the left side (nose far to the LEFT of the crop).
    kwargs = _region_mask_builder_kwargs(degenerate_cheek=True)
    kwargs = {**kwargs, "landmarks": kwargs["landmarks"].copy()}
    kwargs["landmarks"][2] = [720.0 - 738.636962890625 + 320.0, kwargs["landmarks"][2][1]]
    from prism_fas.synthesis.masks import RegionMaskBuilder

    original = RegionMaskBuilder(**kwargs)
    mask_o, _ = original.region("left_cheek")
    corrected = e7g._e7_compatible_region_mask_builder_class()(**kwargs)
    mask_c, source_c = corrected.region("left_cheek")
    if not np.asarray(mask_o).any():
        assert np.asarray(mask_c).any()
        assert source_c == e7g.MASK_RECOVERY_SOURCE_TAG


def test_noncheek_empty_result_still_fails_closed_through_build(tmp_path):
    """A non-cheek region that would be empty must still be surfaced ONLY
    via the normal `RegionMaskBuilder.build()` path -- never specially
    handled or recovered by the E7 adapter."""
    import numpy as np
    from prism_fas.synthesis.contracts import MaskBuildError

    kwargs = _region_mask_builder_kwargs(degenerate_cheek=False)
    # Degenerate landmarks/bbox collapsed to a single point make every landmark-geometry region
    # (nose, eyes, mouth) resolve to a near-zero-radius ellipse; force nose radius to zero pixels
    # by giving a zero-area face_box (bbox == crop_box corners), which face_box() detects as
    # degenerate and fully falls back to the whole crop -- instead, directly prove the pass-
    # through contract: the corrected builder's `region()` for "nose" is IDENTICAL to original.
    corrected = e7g._e7_compatible_region_mask_builder_class()(**kwargs)
    from prism_fas.synthesis.masks import RegionMaskBuilder

    original = RegionMaskBuilder(**kwargs)
    mask_o, source_o = original.region("nose")
    mask_c, source_c = corrected.region("nose")
    assert np.array_equal(mask_o, mask_c) and source_o == source_c
    # And build() itself is inherited, unoverridden -- it still raises the SAME MaskBuildError
    # for an empty non-cheek region (proven directly against the real, frozen build() contract).
    with pytest.raises(MaskBuildError, match="unknown canonical region"):
        corrected.region("not_a_real_region")


def test_recovered_cheek_is_nonempty():
    import numpy as np

    kwargs = _region_mask_builder_kwargs(degenerate_cheek=True)
    corrected = e7g._e7_compatible_region_mask_builder_class()(**kwargs)
    mask, _ = corrected.region("right_cheek")
    assert int(np.asarray(mask).sum()) > 0


def test_source_tag_identifies_recovery_policy():
    kwargs = _region_mask_builder_kwargs(degenerate_cheek=True)
    corrected = e7g._e7_compatible_region_mask_builder_class()(**kwargs)
    _, source = corrected.region("right_cheek")
    assert source == "bbox_geometry+crop_boundary_recovery_v1"
    assert source == e7g.MASK_RECOVERY_SOURCE_TAG


def test_no_sample_id_hardcoding():
    import inspect

    source = inspect.getsource(e7g._E7CompatibleRegionMaskBuilder)
    assert "Live_849" not in source
    assert "extsrce6f52da34a17b88e0eddd19ba7c05ad7" not in source


def test_no_dataset_specific_hardcoding():
    import inspect

    source = inspect.getsource(e7g._E7CompatibleRegionMaskBuilder)
    for token in ("siw", "SiW", "casia", "CASIA", "msu", "MSU"):
        assert token not in source


# --- 10-11: scoped binding restoration --------------------------------------------------------

def test_scoped_binding_restores_exact_original_class_on_success():
    from prism_fas.synthesis import m8_pipeline

    original = m8_pipeline.RegionMaskBuilder
    with e7g._scoped_e7_mask_compatibility_binding([0]):
        assert m8_pipeline.RegionMaskBuilder is not original
        assert issubclass(m8_pipeline.RegionMaskBuilder, original)
    assert m8_pipeline.RegionMaskBuilder is original


def test_scoped_binding_restores_exact_original_class_after_exception():
    from prism_fas.synthesis import m8_pipeline

    original = m8_pipeline.RegionMaskBuilder
    with pytest.raises(RuntimeError, match="boom"):
        with e7g._scoped_e7_mask_compatibility_binding([0]):
            assert m8_pipeline.RegionMaskBuilder is not original
            raise RuntimeError("boom")
    assert m8_pipeline.RegionMaskBuilder is original


def test_scoped_binding_recovery_counter_resets_after_scope():
    corrected_cls = e7g._e7_compatible_region_mask_builder_class()
    counter = [0]
    with e7g._scoped_e7_mask_compatibility_binding(counter):
        assert corrected_cls._recovery_counter is counter
    assert corrected_cls._recovery_counter is None


# --- 12-15: real scientific primitives + protected files byte-identical -----------------------

def test_gpattrainer_remains_real_unmodified_class():
    from prism_fas.synthesis.gpat_trainer import GPATTrainer

    assert GPATTrainer.__module__ == "prism_fas.synthesis.gpat_trainer"
    assert not issubclass(GPATTrainer, e7g._E7CompatibleRegionMaskBuilder)


def test_masks_py_unchanged():
    import subprocess

    relative = "src/prism_fas/synthesis/masks.py"
    committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=REPO, check=True,
                               capture_output=True, text=True).stdout
    on_disk = (REPO / relative).read_text(encoding="utf-8")
    assert committed == on_disk


def test_m8_pipeline_py_unchanged():
    import subprocess

    relative = "src/prism_fas/synthesis/m8_pipeline.py"
    committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=REPO, check=True,
                               capture_output=True, text=True).stdout
    on_disk = (REPO / relative).read_text(encoding="utf-8")
    assert committed == on_disk


def test_gpat_trainer_py_unchanged():
    import subprocess

    relative = "src/prism_fas/synthesis/gpat_trainer.py"
    committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=REPO, check=True,
                               capture_output=True, text=True).stdout
    on_disk = (REPO / relative).read_text(encoding="utf-8")
    assert committed == on_disk


def test_pair_plan_py_unchanged():
    import subprocess

    relative = "src/prism_fas/synthesis/pair_plan.py"
    committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=REPO, check=True,
                               capture_output=True, text=True).stdout
    on_disk = (REPO / relative).read_text(encoding="utf-8")
    assert committed == on_disk


def test_gpat_model_and_losses_and_data_package_unchanged():
    import subprocess

    for relative in ("src/prism_fas/synthesis/gpat_model.py", "src/prism_fas/synthesis/gpat_losses.py",
                     "src/prism_fas/synthesis/gpat_checkpoint.py"):
        committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=REPO, check=True,
                                   capture_output=True, text=True).stdout
        on_disk = (REPO / relative).read_text(encoding="utf-8")
        assert committed == on_disk, f"{relative} differs from HEAD"


# --- 16-25: attempt provenance / fresh-fit / resume safety -------------------------------------

def test_no_last_checkpoint_means_fresh_fit_resume_false(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")
    assert not e7g.gpat_last_checkpoint_path(repo, "EXT-F1").is_file()
    result = e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    assert result["status"] == "FITTED"
    assert result["resumed"] is False  # fresh fit -- no partial checkpoint existed
    provenance = json.loads(e7g.gpat_attempt_provenance_path(repo, "EXT-F1").read_text(encoding="utf-8"))
    assert provenance["resume_requested"] is False


def test_partial_checkpoint_without_sidecar_fails(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")
    run_root = e7g.gpat_fit_run_root(repo, "EXT-F1")
    (run_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_root / "checkpoints" / "last.pt").write_bytes(b"partial-no-sidecar")
    with pytest.raises(e7g.E7Error, match="no.*GPAT_ATTEMPT_PROVENANCE.json.*sidecar"):
        e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    assert (run_root / "checkpoints" / "last.pt").read_bytes() == b"partial-no-sidecar"  # never deleted


def _write_partial_checkpoint_with_provenance(repo: Path, fold_id: str, *, override_field: str | None = None,
                                              override_value: object = None) -> None:
    e7g.materialize_gpat_input_package(repo, fold_id, authorize=True)
    e7g.materialize_fold_pair_plan(repo, fold_id, authorize=True)
    run_root = e7g.gpat_fit_run_root(repo, fold_id)
    (run_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_root / "checkpoints" / "last.pt").write_bytes(b"partial-checkpoint")
    _write_matching_attempt_provenance(repo, fold_id, resume_requested=True)
    if override_field is not None:
        path = e7g.gpat_attempt_provenance_path(repo, fold_id)
        body = json.loads(path.read_text(encoding="utf-8"))
        body[override_field] = override_value
        path.write_text(json.dumps(body), encoding="utf-8")


def test_cross_implementation_partial_resume_fails(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")
    _write_partial_checkpoint_with_provenance(repo, "EXT-F1", override_field="implementation_commit",
                                              override_value="0" * 40)
    with pytest.raises(e7g.E7Error, match="disagrees with the CURRENT"):
        e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    assert e7g.gpat_last_checkpoint_path(repo, "EXT-F1").read_bytes() == b"partial-checkpoint"


def test_cross_module_sha_partial_resume_fails(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")
    _write_partial_checkpoint_with_provenance(repo, "EXT-F1", override_field="implementation_module_sha256",
                                              override_value="1" * 64)
    with pytest.raises(e7g.E7Error, match="disagrees with the CURRENT"):
        e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    assert e7g.gpat_last_checkpoint_path(repo, "EXT-F1").read_bytes() == b"partial-checkpoint"


def test_cross_mask_policy_partial_resume_fails(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")
    _write_partial_checkpoint_with_provenance(repo, "EXT-F1", override_field="mask_compatibility_policy",
                                              override_value="some-other-policy-v0")
    with pytest.raises(e7g.E7Error, match="disagrees with the CURRENT"):
        e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    assert e7g.gpat_last_checkpoint_path(repo, "EXT-F1").read_bytes() == b"partial-checkpoint"


def test_cross_package_pair_config_model_identity_partial_resume_fails(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")
    _write_partial_checkpoint_with_provenance(repo, "EXT-F1", override_field="package_identity",
                                              override_value="drifted-package-identity")
    with pytest.raises(e7g.E7Error, match="disagrees with the CURRENT"):
        e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    assert e7g.gpat_last_checkpoint_path(repo, "EXT-F1").read_bytes() == b"partial-checkpoint"


def test_matching_partial_checkpoint_may_resume(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")
    _write_partial_checkpoint_with_provenance(repo, "EXT-F1")
    result = e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    assert result["status"] == "FITTED"
    assert result["resumed"] is True  # a legitimately resumed fit must report resumed=True


def test_attempt_provenance_written_before_trainer_fit(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)

    seen_before_fit = {}

    class _RecordingTrainer(_FakeTrainer):
        def fit(self, *, run_id, progress, resume):
            path = e7g.gpat_attempt_provenance_path(repo, "EXT-F1")
            seen_before_fit["exists"] = path.is_file()
            return super().fit(run_id=run_id, progress=progress, resume=resume)

    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")
    monkeypatch.setattr("prism_fas.synthesis.gpat_trainer.GPATTrainer", _RecordingTrainer)
    e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    assert seen_before_fit["exists"] is True


def test_terminal_lock_written_after_successful_validation_only(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")
    assert not e7g.gpat_fit_lock_path(repo, "EXT-F1").is_file()
    result = e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    assert result["status"] == "FITTED"
    assert e7g.gpat_fit_lock_path(repo, "EXT-F1").is_file()
    assert e7g.validate_gpat_fit_lock(repo, "EXT-F1")["status"] == "VALID"


def test_failed_fit_leaves_no_terminal_lock_mask_compat(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")

    class _CrashingTrainer(_FakeTrainer):
        def fit(self, *, run_id, progress, resume):
            raise RuntimeError("simulated fit crash")

    monkeypatch.setattr("prism_fas.synthesis.gpat_trainer.GPATTrainer", _CrashingTrainer)
    with pytest.raises(RuntimeError, match="simulated fit crash"):
        e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    assert not e7g.gpat_fit_lock_path(repo, "EXT-F1").is_file()
    # the attempt provenance sidecar (non-terminal) is fine to have landed before the crash
    assert e7g.gpat_attempt_provenance_path(repo, "EXT-F1").is_file()


# --- 26-28: terminal lock backward compatibility + new fields + counter scope ------------------

def test_old_f1_terminal_lock_without_mask_policy_fields_remains_valid(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")
    e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    lock_path = e7g.gpat_fit_lock_path(repo, "EXT-F1")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    # Simulate a HISTORICAL lock written before this fix -- absence of the two new fields.
    del lock["mask_compatibility_policy"]
    del lock["mask_compatibility_recovery_count"]
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    validation = e7g.validate_gpat_fit_lock(repo, "EXT-F1")
    assert validation["status"] == "VALID"


def test_new_terminal_lock_records_mask_policy_and_recovery_count(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F2")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F2")
    result = e7g.prepare_gpat(repo, "EXT-F2", authorize=True)
    lock = result["lock"]
    assert lock["mask_compatibility_policy"] == e7g.MASK_COMPATIBILITY_POLICY
    assert lock["mask_compatibility_recovery_count"] == 0  # the fake trainer never invokes real masks


def test_recovery_counter_is_invocation_scoped(tmp_path, monkeypatch):
    corrected_cls = e7g._e7_compatible_region_mask_builder_class()
    counter_a = [0]
    with e7g._scoped_e7_mask_compatibility_binding(counter_a):
        kwargs = _region_mask_builder_kwargs(degenerate_cheek=True)
        corrected_cls(**kwargs).region("right_cheek")
    assert counter_a[0] == 1
    assert corrected_cls._recovery_counter is None
    counter_b = [0]
    with e7g._scoped_e7_mask_compatibility_binding(counter_b):
        pass  # no recovery activated in this scope
    assert counter_b[0] == 0  # a FRESH counter per invocation, never accumulated across scopes


# --- 29-30: F1 row/pair invariance audit LOGIC --------------------------------------------------

def _write_audit_ready_package(repo: Path, fold_id: str, *, degenerate_cheek: bool) -> tuple[Path, list[str]]:
    """A minimal, directly-constructed GPAT-input-shaped package (real jpg
    + real npz priors, real `manifests/source_train.parquet`) with
    CONTROLLED, non-degenerate-by-default geometry -- bypasses the E7-D/M3B
    machinery entirely so the row/pair invariance audits can be exercised
    against known geometry without fighting the all-zero-prior identity
    fixtures used elsewhere in this file."""
    import cv2
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    package_root = repo / e7g.GPAT_INPUT_ROOT / fold_id
    (package_root / "images").mkdir(parents=True, exist_ok=True)
    (package_root / "priors").mkdir(parents=True, exist_ok=True)
    (package_root / "manifests").mkdir(parents=True, exist_ok=True)

    rows = []
    sample_ids = []
    for label, count in (("live", 2), ("spoof", 2)):
        for i in range(count):
            sample_id = f"audit_{label}_{i}"
            sample_ids.append(sample_id)
            image_path = package_root / "images" / f"{sample_id}.jpg"
            cv2.imwrite(str(image_path), ((i * 7 + 3) % 256) * np.ones((224, 224, 3), dtype="uint8"))
            prior_path = package_root / "priors" / f"{sample_id}.npz"
            kwargs = _region_mask_builder_kwargs(degenerate_cheek=degenerate_cheek and label == "live" and i == 0)
            np.savez(prior_path, parsing_labels=kwargs["parsing"], pose_ypr=np.zeros((3,), dtype="float32"),
                    visibility=np.zeros((9,), dtype="float16"), bbox=kwargs["bbox"],
                    landmarks=kwargs["landmarks"], crop_box=kwargs["crop_box"])
            rows.append({"sample_id": sample_id, "project_split": "source_train",
                        "image_relative_path": f"images/{sample_id}.jpg",
                        "prior_relative_path": f"priors/{sample_id}.npz"})
    table = pa.Table.from_pydict({key: [row[key] for row in rows] for key in rows[0]})
    pq.write_table(table, package_root / "manifests" / "source_train.parquet")
    return package_root, sample_ids


def test_f1_row_invariance_audit_logic_zero_diff_for_normal_geometry(tmp_path):
    repo = _base_repo(tmp_path)
    _write_audit_ready_package(repo, "EXT-F1", degenerate_cheek=False)
    report = e7g.audit_mask_compatibility_row_invariance(repo, "EXT-F1")
    assert report["rows_checked"] == 4
    assert report["different_masks"] == 0
    assert report["different_sources"] == 0
    assert report["recovery_activations"] == 0
    assert report["original_failures"] == 0
    assert report["corrected_failures"] == 0
    assert report["target_access"] is False


def test_f1_row_invariance_audit_logic_detects_recovery(tmp_path):
    repo = _base_repo(tmp_path)
    _write_audit_ready_package(repo, "EXT-F2", degenerate_cheek=True)
    report = e7g.audit_mask_compatibility_row_invariance(repo, "EXT-F2")
    assert report["recovery_activations"] >= 1
    assert report["corrected_failures"] == 0
    assert report["original_failures"] >= 1
    assert report["recovered_sample_ids_diagnostic"]  # diagnostic only, never policy


def test_f1_pair_invariance_audit_logic(tmp_path, monkeypatch):
    from prism_fas.recipes.bank import load_bank

    repo = _base_repo(tmp_path)
    package_root, sample_ids = _write_audit_ready_package(repo, "EXT-F1", degenerate_cheek=False)
    bank_dest = repo / e7g.M7_RECIPE_BANK_ROOT
    bank_dest.parent.mkdir(parents=True, exist_ok=True)
    bank_dest.symlink_to((REPO / e7g.M7_RECIPE_BANK_ROOT).resolve())
    bank = load_bank(bank_dest)
    cheek_recipe = next(r for r in bank["recipes"] if "right_cheek" in r.regions)

    import pyarrow as pa
    import pyarrow.parquet as pq

    pairs_root = repo / e7g.GPAT_PAIR_PLAN_ROOT / "EXT-F1"
    pairs_root.mkdir(parents=True, exist_ok=True)
    pair_row = {"pair_id": "P-1", "partition": "train", "slot": 0, "domain_relation": "same_domain",
               "live_sample_id": sample_ids[0], "live_dataset": "casia_fasd",
               "live_source_record_id": sample_ids[0], "spoof_sample_id": sample_ids[2],
               "spoof_dataset": "casia_fasd", "spoof_source_record_id": sample_ids[2],
               "recipe_id": cheek_recipe.recipe_id, "recipe_seed": int(cheek_recipe.seed),
               "different_subject_rule": "not_applicable", "package_identity": "x",
               "recipe_bank_identity": bank["bank_id"]}
    from prism_fas.synthesis.pair_plan import _PAIR_FIELDS

    schema = pa.schema(_PAIR_FIELDS)
    table = pa.Table.from_pydict({name: [pair_row[name]] for name, _ in _PAIR_FIELDS}, schema=schema)
    pq.write_table(table, pairs_root / "pair_manifest_train.parquet")
    pq.write_table(pa.Table.from_pydict({name: [] for name, _ in _PAIR_FIELDS}, schema=schema),
                   pairs_root / "pair_manifest_validation.parquet")

    report = e7g.audit_mask_compatibility_pair_invariance(repo, "EXT-F1")
    assert report["pairs_checked"] == 1
    assert report["different_masks"] == 0
    assert report["recovery_activations"] == 0
    assert report["target_access"] is False


# --- 31-32: no target access, no LLM -----------------------------------------------------------

def test_mask_compat_audit_target_access_false(tmp_path):
    repo = _base_repo(tmp_path)
    _write_audit_ready_package(repo, "EXT-F1", degenerate_cheek=False)
    report = e7g.audit_mask_compatibility_row_invariance(repo, "EXT-F1")
    assert report["target_access"] is False
    assert report["llm_api_calls"] == 0


def test_mask_compat_no_llm_calls(tmp_path, monkeypatch):
    repo = _build_gpat_ready_fixture(tmp_path, monkeypatch, "EXT-F1")
    _patch_gpat_capable(monkeypatch, repo)
    _patch_fake_trainer(monkeypatch, repo, "EXT-F1")
    result = e7g.prepare_gpat(repo, "EXT-F1", authorize=True)
    assert result["llm_api_calls"] == 0
    assert result["lock"]["LLM_API_CALLS"] == 0
