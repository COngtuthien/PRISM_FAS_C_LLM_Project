"""Tests for `prism_fas.evaluation.c_ext_e7c_gpat_prep` (E7-C per-fold GPAT
preparation and feasibility planning). Every test builds a self-contained
fake repo under `tmp_path` unless it explicitly checks the REAL committed
repo (frozen-hash / bank-identity checks, which can only be meaningfully
verified against real bytes). No test ever renders, fits GPAT, trains, or
calls an LLM.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism_fas.evaluation import c_ext_e7b_data_prep as e7b
from prism_fas.evaluation import c_ext_e7c_gpat_prep as e7c

REPO = Path(__file__).resolve().parents[2]


def _base_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


def _m3b_ref(dataset: str, label: str, sample_id: str = "s1",
            project_split: str = "source_train") -> dict:
    return {"dataset": dataset, "project_split": project_split, "reference_kind": "m3b_processed_sample",
           "sample_id": sample_id, "source_record_id": f"rec-{sample_id}", "subject_id": "1",
           "label_live_spoof": label, "image_relative_path": f"images/{sample_id}.jpg",
           "prior_relative_path": f"priors/{sample_id}.npz", "crop_sha256": "a" * 64,
           "prior_sha256": "b" * 64}


def _siw_ref(video_id: str, project_split: str, label: str, *, population_identity: str = "pop-1",
            split_identity: str = "split-1", family=None) -> dict:
    return {"dataset": "SiW-Mv2", "project_split": project_split, "reference_kind": "siw_raw_video",
           "video_id": video_id, "relative_path": f"{video_id}.mov", "label_live_spoof": label,
           "spoof_family": family, "extension": "mov", "population_identity": population_identity,
           "split_identity": split_identity, "requires_frozen_face_preprocessing": True}


def _write_materialization(repo: Path, fold_id: str, *, source_domains, target_domain,
                           train_refs: list[dict], dev_refs: list[dict] | None = None,
                           siw_population_identity=None, siw_split_identity=None,
                           target_reference=None) -> None:
    body = {
        "schema_version": "ext-q1q2-e7a-materialized-fold-v1", "fold_id": fold_id,
        "source_domains": list(source_domains), "target_domain": target_domain,
        "source_train_references": train_refs, "source_dev_references": dev_refs or [],
        "target_reference": target_reference or {"kind": "BUILD_REQUIRED", "path": None},
        "siw_population_identity": siw_population_identity, "siw_split_identity": siw_split_identity,
        "m3b_package_identity": "fake-m3b-identity", "fold_identity": "fake-fold-identity",
        "target_labels_opened": False, "status": "FROZEN",
    }
    out_dir = repo / e7b.E7A_MATERIALIZATION_DIR / fold_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "FOLD_MATERIALIZATION.json").write_text(json.dumps(body), encoding="utf-8")


def _write_e7b_evidence(repo: Path, **overrides) -> None:
    F = e7c.FROZEN_E7B
    body = {
        "runtime_commit": "test-fixture",
        "siw_source": {"status": "VALID",
                      "package_identity": overrides.get("siw_identity", F["siw_source_package_identity"]),
                      "population_identity": overrides.get("population_identity",
                                                            F["siw_population_identity"]),
                      "split_identity": overrides.get("split_identity", F["siw_split_identity"])},
        "msu_target": {"status": "VALID",
                      "package_identity": overrides.get("msu_identity", F["msu_target_package_identity"])},
        "casia_target": {"status": "VALID",
                        "package_identity": overrides.get("casia_identity",
                                                          F["casia_target_package_identity"])},
        "preprocessing_config_hash": overrides.get("config_hash", F["preprocessing_config_hash"]),
        "detector_model_sha256": overrides.get("detector_sha", F["detector_model_sha256"]),
    }
    path = repo / e7c.E7B_FINAL_EVIDENCE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body), encoding="utf-8")


def _write_recipe_banks(repo: Path, *, row_count: int = 256, skip: set[str] | None = None) -> None:
    skip = skip or set()
    for arm in ("rnd", "det", "llm"):
        if arm in skip:
            continue
        root = repo / f"assets/recipe_banks/c3/{arm}"
        root.mkdir(parents=True, exist_ok=True)
        (root / "C3_BANK.json").write_text(json.dumps({"bank_identity": f"{arm}-identity"}),
                                           encoding="utf-8")
        (root / "recipes.jsonl").write_text(
            "\n".join(json.dumps({"id": i}) for i in range(row_count)) + "\n", encoding="utf-8")
    if "shuffle" not in skip:
        shuffle_dir = repo / "reports/c_ext_q1q2_v1/e6_llm_shuffle"
        shuffle_dir.mkdir(parents=True, exist_ok=True)
        (shuffle_dir / "LLM_SHUFFLE_A_RECIPES.jsonl").write_text(
            "\n".join(json.dumps({"id": i}) for i in range(row_count)) + "\n", encoding="utf-8")
        (shuffle_dir / "E6_LLM_SHUFFLE_A.json").write_text(json.dumps({
            "shuffled_bank_identity": "shuffle-identity", "seed": 20260911,
            "source_bank": "assets/recipe_banks/c3/llm/recipes.jsonl"}), encoding="utf-8")


def _f1_fixture(tmp_path: Path) -> Path:
    repo = _base_repo(tmp_path)
    train = [_m3b_ref("CASIA-FASD", "live", "c1"), _m3b_ref("CASIA-FASD", "spoof", "c2"),
            _m3b_ref("MSU-MFSD", "live", "m1"), _m3b_ref("MSU-MFSD", "spoof", "m2")]
    dev = [_m3b_ref("CASIA-FASD", "live", "c3", project_split="source_dev")]
    _write_materialization(repo, "EXT-F1", source_domains=("CASIA-FASD", "MSU-MFSD"),
                           target_domain="SiW-Mv2", train_refs=train, dev_refs=dev,
                           target_reference={"kind": "REUSE_FROZEN",
                                            "path": "data/processed/prism_target_eval_v2"})
    return repo


def _f2_fixture(tmp_path: Path) -> Path:
    repo = _base_repo(tmp_path)
    train = [_m3b_ref("CASIA-FASD", "live", "c1"), _m3b_ref("CASIA-FASD", "spoof", "c2"),
            _siw_ref("Live_0", "train", "live")]
    dev = [_siw_ref("Live_1", "dev", "live")]
    _write_materialization(repo, "EXT-F2", source_domains=("CASIA-FASD", "SiW-Mv2"),
                           target_domain="MSU-MFSD", train_refs=train, dev_refs=dev,
                           siw_population_identity=e7c.FROZEN_E7B["siw_population_identity"],
                           siw_split_identity=e7c.FROZEN_E7B["siw_split_identity"])
    return repo


def _f3_fixture(tmp_path: Path) -> Path:
    repo = _base_repo(tmp_path)
    train = [_m3b_ref("MSU-MFSD", "live", "m1"), _m3b_ref("MSU-MFSD", "spoof", "m2"),
            _siw_ref("Live_0", "train", "live")]
    dev = [_siw_ref("Live_1", "dev", "live")]
    _write_materialization(repo, "EXT-F3", source_domains=("MSU-MFSD", "SiW-Mv2"),
                           target_domain="CASIA-FASD", train_refs=train, dev_refs=dev,
                           siw_population_identity=e7c.FROZEN_E7B["siw_population_identity"],
                           siw_split_identity=e7c.FROZEN_E7B["siw_split_identity"])
    return repo


def _full_fixture(tmp_path: Path) -> Path:
    repo = _f1_fixture(tmp_path)
    # merge F2/F3 materializations into the same repo
    train2 = [_m3b_ref("CASIA-FASD", "live", "c1"), _siw_ref("Live_0", "train", "live")]
    dev2 = [_siw_ref("Live_1", "dev", "live")]
    _write_materialization(repo, "EXT-F2", source_domains=("CASIA-FASD", "SiW-Mv2"),
                           target_domain="MSU-MFSD", train_refs=train2, dev_refs=dev2,
                           siw_population_identity=e7c.FROZEN_E7B["siw_population_identity"],
                           siw_split_identity=e7c.FROZEN_E7B["siw_split_identity"])
    train3 = [_m3b_ref("MSU-MFSD", "live", "m1"), _siw_ref("Live_0", "train", "live")]
    dev3 = [_siw_ref("Live_1", "dev", "live")]
    _write_materialization(repo, "EXT-F3", source_domains=("MSU-MFSD", "SiW-Mv2"),
                           target_domain="CASIA-FASD", train_refs=train3, dev_refs=dev3,
                           siw_population_identity=e7c.FROZEN_E7B["siw_population_identity"],
                           siw_split_identity=e7c.FROZEN_E7B["siw_split_identity"])
    _write_e7b_evidence(repo)
    _write_recipe_banks(repo)
    return repo


# --- 1: E7-B identity binding exact ------------------------------------------

def test_e7b_binding_matches_when_identities_agree(tmp_path):
    repo = _base_repo(tmp_path)
    _write_e7b_evidence(repo)
    binding = e7c.build_e7b_binding(repo)
    assert binding["status"] == "MATCH"
    assert binding["match"] is True
    assert binding["local_data_state"] == "PLAN_VALID"


def test_e7b_binding_local_bytes_missing_when_no_evidence(tmp_path):
    repo = _base_repo(tmp_path)
    binding = e7c.build_e7b_binding(repo)
    assert binding["status"] == "LOCAL_BYTES_MISSING"
    assert binding["match"] is False


# --- 2: E7-A materialization identity binding exact (against the REAL repo) -

def test_e7a_fold_binding_matches_real_repo():
    for fold_id in e7c.FOLD_IDS:
        binding = e7c.build_e7a_fold_binding(REPO, fold_id)
        assert binding["hash_present"] is True
        assert binding["hash_match"] is True
        assert binding["match"] is True


# --- 3/4/5: fold source domains exact ----------------------------------------

def test_f1_source_domains_casia_msu_only(tmp_path):
    repo = _f1_fixture(tmp_path)
    binding = e7c.build_fold_source_binding(repo, "EXT-F1")
    assert set(binding["datasets_present_in_source_refs"]) == {"CASIA-FASD", "MSU-MFSD"}
    assert binding["source_domains"] == ["CASIA-FASD", "MSU-MFSD"]


def test_f2_source_domains_casia_siw_only(tmp_path):
    repo = _f2_fixture(tmp_path)
    binding = e7c.build_fold_source_binding(repo, "EXT-F2")
    assert set(binding["datasets_present_in_source_refs"]) == {"CASIA-FASD", "SiW-Mv2"}


def test_f3_source_domains_msu_siw_only(tmp_path):
    repo = _f3_fixture(tmp_path)
    binding = e7c.build_fold_source_binding(repo, "EXT-F3")
    assert set(binding["datasets_present_in_source_refs"]) == {"MSU-MFSD", "SiW-Mv2"}


# --- 6: held-out target absent from source support refs ---------------------

def test_heldout_target_absent_from_source_refs(tmp_path):
    repo = _f1_fixture(tmp_path)
    binding = e7c.build_fold_source_binding(repo, "EXT-F1")
    assert binding["heldout_target_absent_from_source_refs"] is True


def test_heldout_target_present_detected_and_fails_closed(tmp_path):
    repo = _base_repo(tmp_path)
    train = [_m3b_ref("CASIA-FASD", "live", "c1"), _m3b_ref("MSU-MFSD", "live", "m1"),
            _siw_ref("Sneaky_0", "train", "live")]  # SiW leaked into F1's source refs
    _write_materialization(repo, "EXT-F1", source_domains=("CASIA-FASD", "MSU-MFSD"),
                           target_domain="SiW-Mv2", train_refs=train)
    binding = e7c.build_fold_source_binding(repo, "EXT-F1")
    assert binding["heldout_target_absent_from_source_refs"] is False
    with pytest.raises(e7c.E7CError, match="held-out target domain"):
        e7c.write_fold_source_binding(repo)


# --- 7: target label paths rejected ------------------------------------------

def test_target_label_path_rejected():
    with pytest.raises(e7c.E7CTargetFirewallViolation):
        e7c.assert_not_target_path("EXT-F1", "data/evaluation_only/prism_target_v2_labels/x.parquet")


# --- 8: target image/crop package paths rejected as support input -----------

def test_f1_target_image_package_path_rejected():
    with pytest.raises(e7c.E7CTargetFirewallViolation):
        e7c.assert_not_target_path("EXT-F1", "data/processed/prism_target_eval_v2/frames/x.jpg")


def test_f2_target_image_package_path_rejected():
    with pytest.raises(e7c.E7CTargetFirewallViolation):
        e7c.assert_not_target_path("EXT-F2", f"{e7b.E7B_MSU_TARGET_PACKAGE_ROOT}/crops/x.jpg")


def test_f3_target_image_package_path_rejected():
    with pytest.raises(e7c.E7CTargetFirewallViolation):
        e7c.assert_not_target_path("EXT-F3", f"{e7b.E7B_CASIA_TARGET_PACKAGE_ROOT}/crops/x.jpg")


def test_source_paths_never_rejected_for_their_own_fold():
    # SiW is legitimately source for F2/F3 -- must NOT be firewalled there
    e7c.assert_not_target_path("EXT-F2", f"{e7b.E7B_SIW_SOURCE_PACKAGE_ROOT}/crops/x.jpg")
    e7c.assert_not_target_path("EXT-F3", f"{e7b.E7B_SIW_SOURCE_PACKAGE_ROOT}/crops/x.jpg")
    e7c.assert_not_target_path("EXT-F1", f"{e7b.CASIA_MSU_PACKAGE_ROOT}/images/x.jpg")


# --- 9/10: M3B / E7-B SiW source reused, never rewritten ---------------------

def test_m3b_and_e7b_siw_source_never_written_by_this_module():
    """Every write in this module goes through `_write()`, which always
    targets `E7C_REPORT_DIR` -- the module never contains a literal path
    string for M3B or any E7-B package root as a write target (it only
    ever imports those roots from `c_ext_e7b_data_prep` for READING)."""
    source = Path(e7c.__file__).read_text(encoding="utf-8")
    for forbidden_literal in ('"data/packages/prism_data_v1_m3b"', '"data/processed/c_ext_q1q2_v1/e7b'):
        assert forbidden_literal not in source
    assert "write_text" not in source or "_write(" in source  # the only writer is _write()


def test_e7c_prepare_leaves_m3b_and_e7b_untouched(tmp_path):
    repo = _full_fixture(tmp_path)
    m3b_marker = repo / e7b.CASIA_MSU_PACKAGE_ROOT / "PACKAGE_LOCK.json"
    m3b_marker.parent.mkdir(parents=True, exist_ok=True)
    m3b_marker.write_text('{"untouched": true}', encoding="utf-8")
    before = m3b_marker.read_bytes()
    try:
        e7c.prepare_e7c(repo)
    except e7c.E7CError:
        pass  # fixture may not satisfy every fail-closed check; irrelevant to this test
    assert m3b_marker.read_bytes() == before


# --- 11: source-live filtering exact -----------------------------------------

def test_source_live_pool_filtering_exact(tmp_path):
    repo = _f1_fixture(tmp_path)
    plan = e7c.build_source_live_pool_plan(repo, "EXT-F1")
    assert plan["m3b_plan"]["m3b_crop_train_rows"] == 4
    assert plan["m3b_plan"]["m3b_live_crop_train_rows"] == 2  # c1, m1 live; c2, m2 spoof
    assert plan["m3b_plan"]["m3b_crop_dev_rows"] == 1
    assert plan["m3b_plan"]["m3b_live_crop_dev_rows"] == 1
    assert plan["siw_plan"] is None  # F1 has no SiW in its source domains


# --- 12: no subject-id requirement for SiW -----------------------------------

def test_no_subject_id_required_for_siw(tmp_path):
    repo = _f2_fixture(tmp_path)
    plan = e7c.build_source_live_pool_plan(repo, "EXT-F2")
    assert plan["no_subject_id_required_for_siw"] is True
    materialization = e7b.load_e7a_fold_materialization(repo, "EXT-F2")
    siw_refs = [r for r in materialization["source_train_references"] +
               materialization["source_dev_references"] if r["dataset"] == "SiW-Mv2"]
    assert siw_refs
    assert all("subject_id" not in r for r in siw_refs)


# --- 13/14/15/16: recipe bank bindings ---------------------------------------

def test_rnd_recipe_binding(tmp_path):
    repo = _base_repo(tmp_path)
    _write_recipe_banks(repo)
    binding = e7c.build_recipe_bank_binding(repo)
    assert binding["bindings"]["RND"]["status"] == "FROZEN_REUSE"
    assert binding["bindings"]["RND"]["row_count"] == 256


def test_det_recipe_binding(tmp_path):
    repo = _base_repo(tmp_path)
    _write_recipe_banks(repo)
    binding = e7c.build_recipe_bank_binding(repo)
    assert binding["bindings"]["DET"]["status"] == "FROZEN_REUSE"


def test_llm_recipe_binding(tmp_path):
    repo = _base_repo(tmp_path)
    _write_recipe_banks(repo)
    binding = e7c.build_recipe_bank_binding(repo)
    assert binding["bindings"]["LLM"]["status"] == "FROZEN_REUSE"


def test_shuffle_recipe_binding(tmp_path):
    repo = _base_repo(tmp_path)
    _write_recipe_banks(repo)
    binding = e7c.build_recipe_bank_binding(repo)
    assert binding["bindings"]["LLM-SHUFFLE-A"]["status"] == "FROZEN_REUSE"
    assert binding["bindings"]["LLM-SHUFFLE-A"]["row_count"] == 256


def test_recipe_bank_bindings_match_real_committed_banks():
    binding = e7c.build_recipe_bank_binding(REPO)
    assert binding["all_required_banks_bound"] is True
    assert binding["bindings"]["RND"]["observed_binding_identity"] == \
        "07db567c2b432a9239b01d02bac80b95211baafd7f7047ddbad3af43a7ee1136"
    assert binding["bindings"]["LLM-SHUFFLE-A"]["observed_binding_identity"] == \
        "a9fe9897cf78d03a57c871b461dc2fef15282deacddd767d032495eaf02533cf"


# --- 12/13/14/15/16: identity_kind audit -------------------------------------

def test_every_arm_exposes_identity_kind(tmp_path):
    repo = _base_repo(tmp_path)
    _write_recipe_banks(repo)
    binding = e7c.build_recipe_bank_binding(repo)
    for arm in ("RND", "DET", "LLM", "LLM-SHUFFLE-A"):
        assert binding["bindings"][arm]["observed_binding_identity_kind"] is not None


def test_llm_observed_binding_identity_not_mislabeled_canonical(tmp_path):
    binding = e7c.build_recipe_bank_binding(REPO)
    llm = binding["bindings"]["LLM"]
    assert llm["observed_binding_identity_kind"] == "RECIPE_BANK_IDENTITY"
    assert llm["observed_binding_identity"] != llm["canonical_selected_set_identity"]
    assert llm["observed_binding_identity"] == \
        "f225df13ad49eafb90fa9eb903d4dc85efec79c390ec42243a077c80f5d6cb59"
    assert llm["canonical_selected_set_identity"] == \
        "fcc4c8005c0699c903909ab19bcc87800b73a2fc2d28d1a6eab73bcbd8a8f326"


def test_frozen_llm_equivalence_evidence_resolved_from_repository():
    binding = e7c.build_recipe_bank_binding(REPO)
    llm = binding["bindings"]["LLM"]
    assert llm["equivalence_proven"] is True
    assert llm["raw_content_identity"] == \
        "7d4b56421c8e98928db980605befb6419b1851c59e64e09ae05649266326e1d0"
    assert llm["equivalence_evidence_path"] == \
        "reports/c_ext_q1q2_v1/e6_paired_current_runtime_v2/E6_V2_RECIPE_PAIR_LOCK.json"
    assert (REPO / llm["equivalence_evidence_path"]).is_file()


def test_shuffle_identity_unchanged_and_kind_reported():
    binding = e7c.build_recipe_bank_binding(REPO)
    shuffle = binding["bindings"]["LLM-SHUFFLE-A"]
    assert shuffle["observed_binding_identity"] == \
        "a9fe9897cf78d03a57c871b461dc2fef15282deacddd767d032495eaf02533cf"
    assert shuffle["observed_binding_identity_kind"] == "CANONICAL_JSONL_CONTENT_HASH"


def test_rnd_det_identity_kinds_are_recipe_bank_identity():
    binding = e7c.build_recipe_bank_binding(REPO)
    assert binding["bindings"]["RND"]["observed_binding_identity_kind"] == "RECIPE_BANK_IDENTITY"
    assert binding["bindings"]["DET"]["observed_binding_identity_kind"] == "RECIPE_BANK_IDENTITY"


# --- F1 target metadata identity ---------------------------------------------

def test_f1_target_metadata_identity_resolved_from_real_repo():
    f1_target = e7c.build_f1_target_metadata_identity(REPO)
    assert f1_target["status"] == "RESOLVED"
    assert f1_target["identity"] == "c3a29e695ad08c4b31e01533f1d12374f4e30c51f0167c6622cf8168792e48a8"


def test_f1_target_metadata_identity_appears_in_isolation_report_and_source_binding():
    isolation = e7c.build_target_isolation_report(REPO, "EXT-F1")
    assert isolation["target_package_identity_referenced_as_metadata_only"] == \
        "c3a29e695ad08c4b31e01533f1d12374f4e30c51f0167c6622cf8168792e48a8"
    assert isolation["target_image_bytes_opened"] is False
    source_binding = e7c.build_fold_source_binding(REPO, "EXT-F1")
    assert source_binding["target_package_identity"] == \
        "c3a29e695ad08c4b31e01533f1d12374f4e30c51f0167c6622cf8168792e48a8"
    assert source_binding["target_image_crop_bytes_opened"] is False


# --- video vs crop granularity separation ------------------------------------

def test_siw_video_counts_never_labelled_crop_rows(tmp_path):
    """Item 1: SiW video counts must never appear under a *_rows/*_crop_*
    key -- only under the explicit siw_video_*/siw_live_video_* names."""
    repo = _f2_fixture(tmp_path)
    plan = e7c.build_source_live_pool_plan(repo, "EXT-F2")
    video_level = plan["siw_plan"]["video_level"]
    for key in video_level:
        if key.startswith("siw_"):
            assert "crop" not in key


def test_f2_f3_do_not_expose_mixed_unit_aggregate(tmp_path):
    """Item 2: no single field sums M3B crop rows and SiW video refs."""
    repo = _f2_fixture(tmp_path)
    plan = e7c.build_source_live_pool_plan(repo, "EXT-F2")
    assert "source_train_rows" not in plan
    assert "source_live_train_rows" not in plan
    assert plan["mixed_unit_total_never_computed"] is True


def test_siw_crop_level_pool_is_gpu_required(tmp_path):
    """Item 3."""
    repo = _f2_fixture(tmp_path)
    plan = e7c.build_source_live_pool_plan(repo, "EXT-F2")
    assert plan["siw_plan"]["crop_level"]["resolved_locally"] is False
    assert plan["SOURCE_CROP_SUPPORT_MATERIALIZATION_STATUS"] == "GPU_REQUIRED"


def test_no_4x_inference_used_for_live_crop_counts(tmp_path):
    """Item 4: live crop counts are the string 'GPU_REQUIRED', never an
    integer computed as video_count * 4."""
    repo = _f2_fixture(tmp_path)
    plan = e7c.build_source_live_pool_plan(repo, "EXT-F2")
    assert plan["siw_plan"]["crop_level"]["siw_live_crop_train_rows"] == "GPU_REQUIRED"
    assert plan["siw_plan"]["crop_level"]["siw_live_crop_dev_rows"] == "GPU_REQUIRED"


def test_exact_crop_level_counts_require_e7b_package_materialization(tmp_path):
    """Item 5."""
    repo = _f2_fixture(tmp_path)
    plan = e7c.build_source_live_pool_plan(repo, "EXT-F2")
    assert plan["SOURCE_CROP_SUPPORT_BYTES_LOCAL"] is False
    assert plan["siw_plan"]["crop_level"]["siw_success_crop_total"] == 6776
    assert plan["siw_plan"]["crop_level"]["siw_failure_total"] == 24


def test_future_gpu_join_uses_source_video_id_to_e7a_video_ref(tmp_path):
    """Item 6."""
    repo = _f2_fixture(tmp_path)
    plan = e7c.build_source_live_pool_plan(repo, "EXT-F2")
    contract = plan["siw_plan"]["gpu_crop_level_join_contract"]
    joined = " ".join(contract)
    assert "source_video_id" in joined and "E7-A" in joined


def test_only_successful_siw_crops_enter_crop_pool(tmp_path):
    """Item 7."""
    repo = _f2_fixture(tmp_path)
    plan = e7c.build_source_live_pool_plan(repo, "EXT-F2")
    contract = plan["siw_plan"]["gpu_crop_level_join_contract"]
    assert any("status == 'success'" in step for step in contract)


def test_only_live_successful_crops_enter_gpat_support(tmp_path):
    """Item 8."""
    repo = _f2_fixture(tmp_path)
    plan = e7c.build_source_live_pool_plan(repo, "EXT-F2")
    contract = plan["siw_plan"]["gpu_crop_level_join_contract"]
    assert any("live" in step and "GPAT" in step for step in contract)


def test_no_replacement_for_terminal_failures(tmp_path):
    """Item 9."""
    repo = _f2_fixture(tmp_path)
    plan = e7c.build_source_live_pool_plan(repo, "EXT-F2")
    contract = plan["siw_plan"]["gpu_crop_level_join_contract"]
    assert any("never sample a replacement" in step for step in contract)


def test_preflight_pass_does_not_imply_materialization():
    """Item 10."""
    preflight = e7c.e7c_preflight(REPO)
    assert preflight["E7C_PREFLIGHT_PASS"] is True
    assert preflight["E7C_SOURCE_SUPPORT_MATERIALIZED"] is False
    for fold in ("F1", "F2", "F3"):
        assert preflight[f"{fold}_SOURCE_CROP_POOL_MATERIALIZED"] is False


def test_ready_for_training_remains_false():
    """Item 11."""
    readiness = e7c.build_readiness(REPO)
    assert readiness["E7_READY_FOR_TRAINING"] is False


def test_target_bytes_and_labels_still_unopened():
    """Items 17/18."""
    preflight = e7c.e7c_preflight(REPO)
    assert preflight["TARGET_IMAGE_ACCESS"] is False
    assert preflight["TARGET_LABEL_ACCESS"] is False


def test_no_gpat_render_train_llm_this_turn():
    """Item 19."""
    preflight = e7c.e7c_preflight(REPO)
    assert preflight["GPAT_FITTING_PERFORMED"] is False
    assert preflight["RENDERING_PERFORMED"] is False
    assert preflight["TRAINING_PERFORMED"] is False
    assert preflight["LLM_API_CALLS"] == 0


# --- review-fix audit artifacts ----------------------------------------------

def test_source_granularity_correction_artifact_real_repo():
    correction = e7c.build_source_granularity_correction(REPO)
    assert correction["SOURCE_GRANULARITY_BUG_FIXED"] is True
    assert correction["SCIENTIFIC_PROTOCOL_CHANGED"] is False
    assert correction["verified_video_level_counts"]["EXT-F2"]["siw_video_train_count"] == 1362
    assert correction["verified_video_level_counts"]["EXT-F2"]["siw_video_dev_count"] == 338
    assert correction["verified_video_level_counts"]["EXT-F2"]["siw_live_video_train_count"] == 628
    assert correction["verified_video_level_counts"]["EXT-F2"]["siw_live_video_dev_count"] == 157
    assert correction["f3_same_siw_video_split_as_f2"] is True


def test_recipe_identity_provenance_artifact_real_repo():
    provenance = e7c.build_recipe_identity_provenance(REPO)
    assert provenance["SCIENTIFIC_PROTOCOL_CHANGED"] is False
    assert provenance["llm_equivalence"]["equivalence_proven"] is True


# --- 17/18/19/20: no LLM / no renderer / no GPAT fitting / no training ------

def test_no_llm_calls():
    source = Path(e7c.__file__).read_text(encoding="utf-8")
    for forbidden in ("openai", "google.generativeai", "GEMINI_API_KEY", "GeminiClient("):
        assert forbidden not in source


def test_no_renderer_calls():
    source = Path(e7c.__file__).read_text(encoding="utf-8")
    for forbidden in ("render_arm(", "render_one(", "GPATRoute(", "PhysicsRoute("):
        assert forbidden not in source


def test_no_gpat_fitting_in_preflight(tmp_path):
    repo = _base_repo(tmp_path)
    preflight = e7c.e7c_preflight(repo)
    assert preflight["GPAT_FITTING_PERFORMED"] is False
    source = Path(e7c.__file__).read_text(encoding="utf-8")
    assert "GPATTrainer(" not in source
    assert ".fit(" not in source


def test_no_training_performed(tmp_path):
    repo = _base_repo(tmp_path)
    preflight = e7c.e7c_preflight(repo)
    assert preflight["TRAINING_PERFORMED"] is False
    assert preflight["RENDERING_PERFORMED"] is False


# --- 21: F1 Shuffle hard-block preserved -------------------------------------

def test_f1_shuffle_hard_block_preserved():
    assert e7c.build_condition_status("EXT-F1", "G-LLM-SHUFFLE-A") == \
        e7c.BLOCKED_TRUE_FROZEN_MATCHED_BANK_INFEASIBILITY


def test_f1_shuffle_block_survives_write(tmp_path):
    repo = _base_repo(tmp_path)
    plan = e7c.build_feasibility_plan(repo, "EXT-F1")
    assert plan["conditions"]["G-LLM-SHUFFLE-A"] == e7c.BLOCKED_TRUE_FROZEN_MATCHED_BANK_INFEASIBILITY
    assert plan["shuffle_basis"]["SHUFFLE_PHYSICS_CASIA_DEFICIT"] == 33


# --- 22: F2/F3 Shuffle remain pending, never falsely blocked/pass -----------

def test_f2_f3_shuffle_remain_pending_not_inherited_from_f1():
    assert e7c.build_condition_status("EXT-F2", "G-LLM-SHUFFLE-A") == e7c.PENDING_FEASIBILITY_PREFLIGHT
    assert e7c.build_condition_status("EXT-F3", "G-LLM-SHUFFLE-A") == e7c.PENDING_FEASIBILITY_PREFLIGHT


def test_f2_f3_shuffle_not_falsely_blocked_or_passed(tmp_path):
    repo = _base_repo(tmp_path)
    for fold_id in ("EXT-F2", "EXT-F3"):
        plan = e7c.build_feasibility_plan(repo, fold_id)
        status = plan["conditions"]["G-LLM-SHUFFLE-A"]
        assert status not in (e7c.BLOCKED_TRUE_FROZEN_MATCHED_BANK_INFEASIBILITY, "READY_FOR_GPAT_PREP")
        assert status == e7c.PENDING_FEASIBILITY_PREFLIGHT
        assert plan["shuffle_basis"]["ext_f1_result_does_not_predetermine_this_fold"] is True


# --- 23: target isolation fail-closed ----------------------------------------

def test_target_isolation_passes_for_clean_fixture(tmp_path):
    repo = _f1_fixture(tmp_path)
    report = e7c.build_target_isolation_report(repo, "EXT-F1")
    assert report["pass"] is True
    assert report["violations"] == []


def test_target_isolation_fails_closed_on_violation(tmp_path):
    repo = _base_repo(tmp_path)
    train = [_m3b_ref("CASIA-FASD", "live", "c1"), _siw_ref("Sneaky_0", "train", "live")]
    _write_materialization(repo, "EXT-F1", source_domains=("CASIA-FASD", "MSU-MFSD"),
                           target_domain="SiW-Mv2", train_refs=train)
    report = e7c.build_target_isolation_report(repo, "EXT-F1")
    assert report["pass"] is False
    assert report["violations"]
    with pytest.raises(e7c.E7CError, match="target isolation FAILED"):
        e7c.write_target_isolation_report(repo)


# --- 24: mismatched E7-B package identity fails closed -----------------------

def test_mismatched_e7b_package_identity_fails_closed(tmp_path):
    repo = _base_repo(tmp_path)
    _write_e7b_evidence(repo, siw_identity="0" * 65)
    binding = e7c.build_e7b_binding(repo)
    assert binding["status"] == "MISMATCH"
    assert "siw_source_package_identity" in binding["mismatches"]
    with pytest.raises(e7c.E7CError, match="MISMATCH"):
        e7c.write_e7b_binding(repo)


# --- 25: mismatched SiW split identity fails closed --------------------------

def test_mismatched_siw_split_identity_fails_closed(tmp_path):
    repo = _base_repo(tmp_path)
    train = [_siw_ref("Live_0", "train", "live", split_identity="drifted-split-identity")]
    _write_materialization(repo, "EXT-F2", source_domains=("CASIA-FASD", "SiW-Mv2"),
                           target_domain="MSU-MFSD", train_refs=train,
                           siw_population_identity=e7c.FROZEN_E7B["siw_population_identity"],
                           siw_split_identity="drifted-split-identity")
    binding = e7c.build_e7a_fold_binding(repo, "EXT-F2")
    assert binding["siw_split_identity_match"] is False
    assert binding["match"] is False


# --- 26: missing recipe bank fails closed ------------------------------------

def test_missing_recipe_bank_fails_closed(tmp_path):
    repo = _base_repo(tmp_path)
    _write_recipe_banks(repo, skip={"det"})
    binding = e7c.build_recipe_bank_binding(repo)
    assert binding["bindings"]["DET"]["status"] == "UNRESOLVED"
    assert binding["all_required_banks_bound"] is False
    with pytest.raises(e7c.E7CError, match="UNRESOLVED"):
        e7c.write_recipe_bank_binding(repo)


def test_missing_shuffle_bank_fails_closed(tmp_path):
    repo = _base_repo(tmp_path)
    _write_recipe_banks(repo, skip={"shuffle"})
    binding = e7c.build_recipe_bank_binding(repo)
    assert binding["bindings"]["LLM-SHUFFLE-A"]["status"] == "UNRESOLVED"
    with pytest.raises(e7c.E7CError, match="UNRESOLVED"):
        e7c.write_recipe_bank_binding(repo)


# --- 27: protected E7-B files untouched --------------------------------------

def test_protected_e7b_module_and_artifacts_untouched():
    e7b_module_path = REPO / "src/prism_fas/evaluation/c_ext_e7b_data_prep.py"
    before = e7b_module_path.read_bytes()
    e7c.e7c_preflight(REPO)  # read-only call against the real repo
    assert e7b_module_path.read_bytes() == before

    e7b_evidence_path = REPO / e7c.E7B_FINAL_EVIDENCE_PATH
    before_evidence = e7b_evidence_path.read_bytes()
    e7c.build_e7b_binding(REPO)
    assert e7b_evidence_path.read_bytes() == before_evidence


# --- 28: E7-C output additive only -------------------------------------------

def test_e7c_prepare_writes_only_into_own_namespace(tmp_path):
    repo = _full_fixture(tmp_path)
    before_tree = sorted(str(p.relative_to(repo)) for p in repo.rglob("*") if p.is_file())
    try:
        e7c.prepare_e7c(repo)
    except e7c.E7CError:
        pass  # a fixture-level fail-closed error is fine -- what matters is no stray writes happened
    after_tree = sorted(str(p.relative_to(repo)) for p in repo.rglob("*") if p.is_file())
    new_files = sorted(set(after_tree) - set(before_tree))
    assert all(f.startswith(e7c.E7C_REPORT_DIR) for f in new_files)


def test_e7c_prepare_succeeds_end_to_end_on_full_fixture(tmp_path):
    repo = _full_fixture(tmp_path)
    # this fixture's E7-A materializations are synthetic, so the frozen-hash
    # check will never match real committed bytes -- prepare_e7c is expected
    # to fail closed at write_e7a_fold_binding, never silently succeed with
    # fabricated hash agreement
    with pytest.raises(e7c.E7CError):
        e7c.prepare_e7c(repo)


# --- extra: CLI wiring --------------------------------------------------------

def test_main_preflight_flag(monkeypatch, capsys):
    monkeypatch.setattr(e7c.cc, "repo_root", lambda: REPO)
    assert e7c.main(["--e7c-preflight"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["E7C_PREFLIGHT_PASS"] is True
    assert out["TARGET_LABEL_ACCESS"] is False
    assert out["TARGET_IMAGE_ACCESS"] is False


def test_main_no_flags_returns_nonzero(monkeypatch):
    monkeypatch.setattr(e7c.cc, "repo_root", lambda: REPO)
    assert e7c.main([]) == 1


def test_real_repo_preflight_passes_cleanly():
    preflight = e7c.e7c_preflight(REPO)
    assert preflight["E7C_PREFLIGHT_PASS"] is True
    assert preflight["F1_SHUFFLE_STATUS"] == e7c.BLOCKED_TRUE_FROZEN_MATCHED_BANK_INFEASIBILITY
    assert preflight["F2_SHUFFLE_STATUS"] == e7c.PENDING_FEASIBILITY_PREFLIGHT
    assert preflight["F3_SHUFFLE_STATUS"] == e7c.PENDING_FEASIBILITY_PREFLIGHT
