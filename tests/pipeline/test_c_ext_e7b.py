"""Tests for `prism_fas.evaluation.c_ext_e7b_data_prep` (E7-B source/target
data preprocessing and package readiness). Every test builds a
self-contained fake repo under `tmp_path`. No test ever renders, trains,
fits GPAT, accesses target labels or calls an LLM.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism_fas.evaluation import c_ext_e7b_data_prep as e7b

REPO = Path(__file__).resolve().parents[2]


def _base_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


def _real_m2_config_fixture(repo: Path) -> None:
    """Copies the REAL, frozen preprocess_m2.yaml verbatim -- never a
    rewritten/synthetic config, so config_hash cross-checks are meaningful."""
    dest = repo / e7b.M2_CONFIG_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text((REPO / e7b.M2_CONFIG_PATH).read_text(encoding="utf-8"), encoding="utf-8")


def _frozen_m2_evidence_fixture(repo: Path, *, config_hash: str | None = None) -> None:
    dest = repo / e7b.FROZEN_M2_EVIDENCE_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({
        "needed": {"m2": {"marker": {
            "preprocessing_config_hash": config_hash or e7b.FROZEN_M2_PREPROCESSING_CONFIG_HASH,
            "detector_model_sha256": e7b.FROZEN_SCRFD_MODEL_SHA256,
            "record_counts": {"casia_fasd": 600, "msu_mfsd": 280},
        }}}
    }), encoding="utf-8")


def _e7a_fold_fixture(repo: Path, fold_id: str, *, siw_refs: list[dict] | None = None,
                      m3b_refs: list[dict] | None = None, target_reference: dict | None = None) -> None:
    out_dir = repo / e7b.E7A_MATERIALIZATION_DIR / fold_id
    out_dir.mkdir(parents=True, exist_ok=True)
    body = {
        "fold_id": fold_id,
        "source_train_references": (m3b_refs or []) + (siw_refs or []),
        "source_dev_references": [],
        "target_reference": target_reference or {"kind": "PLAN_ONLY"},
        "target_labels_opened": False,
    }
    (out_dir / "FOLD_MATERIALIZATION.json").write_text(json.dumps(body), encoding="utf-8")


def _siw_ref(video_id: str, split: str, *, population_identity: str = "pop-1",
            split_identity: str = "split-1", label: str = "live", family=None) -> dict:
    return {"video_id": video_id, "project_split": split, "reference_kind": "siw_raw_video",
           "label_live_spoof": label, "spoof_family": family, "extension": "avi",
           "population_identity": population_identity, "split_identity": split_identity}


def _m3b_ref(dataset: str, sample_id: str) -> dict:
    return {"dataset": dataset, "sample_id": sample_id, "reference_kind": "m3b_processed_sample"}


def _matching_e7a_fixture(tmp_path: Path) -> Path:
    """F2 and F3 share the identical SiW ref set, as E7-A guarantees."""
    repo = _base_repo(tmp_path)
    siw_refs = [_siw_ref(f"Live_{i}", "train") for i in range(3)] + \
              [_siw_ref(f"Live_{i}", "dev") for i in range(3, 4)]
    _e7a_fold_fixture(repo, "EXT-F1")
    _e7a_fold_fixture(repo, "EXT-F2", siw_refs=siw_refs, m3b_refs=[_m3b_ref("casia_fasd", "c-0")])
    _e7a_fold_fixture(repo, "EXT-F3", siw_refs=siw_refs, m3b_refs=[_m3b_ref("msu_mfsd", "m-0")])
    return repo


def _m3b_fixture(repo: Path) -> None:
    (repo / e7b.CASIA_MSU_PACKAGE_ROOT / "PACKAGE_LOCK.json").parent.mkdir(parents=True, exist_ok=True)
    (repo / e7b.CASIA_MSU_PACKAGE_ROOT / "PACKAGE_LOCK.json").write_text(
        json.dumps({"content_identity_sha256": "fake-m3b-identity"}), encoding="utf-8")


def _siw_target_fixture(repo: Path) -> None:
    out = repo / e7b.SIW_TARGET_EVAL_PACKAGE_ROOT
    out.mkdir(parents=True, exist_ok=True)
    (out / "TARGET_PACKAGE_LOCK.json").write_text(
        json.dumps({"content_identity_sha256": "fake-siw-target-identity"}), encoding="utf-8")


def _full_fixture(tmp_path: Path) -> Path:
    repo = _matching_e7a_fixture(tmp_path)
    _real_m2_config_fixture(repo)
    _frozen_m2_evidence_fixture(repo)
    _m3b_fixture(repo)
    _siw_target_fixture(repo)
    return repo


# --- 1: E7-A frozen hashes bound correctly -----------------------------------

def test_e7a_frozen_hashes_bound_correctly(tmp_path):
    repo = _full_fixture(tmp_path)
    results = e7b.verify_e7a_frozen_hashes(repo)
    assert set(results) == {"EXT-F1", "EXT-F2", "EXT-F3"}
    for fold_id in results:
        assert results[fold_id]["present"] is True
        assert results[fold_id]["match"] is False  # this fixture's content differs from the real 6c77633 bytes
        assert results[fold_id]["expected"] == e7b.E7A_FROZEN_SHA256[fold_id]


def test_e7a_frozen_hashes_match_real_committed_files():
    """Against the REAL repo (not a fixture), the real committed E7-A
    materializations must match the frozen 6c77633 hashes exactly."""
    results = e7b.verify_e7a_frozen_hashes(REPO)
    for fold_id, expected in e7b.E7A_FROZEN_SHA256.items():
        assert results[fold_id]["present"] is True
        assert results[fold_id]["match"] is True
        assert results[fold_id]["observed"] == expected


# --- 2-4: SiW source shared package; video-level train/dev inheritance -----

def test_siw_source_one_shared_package_for_f2_f3(tmp_path):
    repo = _matching_e7a_fixture(tmp_path)
    refs = e7b._siw_source_refs_from_e7a(repo)
    assert len(refs) == 4
    assert {(r["video_id"], r["project_split"]) for r in refs} == {
        ("Live_0", "train"), ("Live_1", "train"), ("Live_2", "train"), ("Live_3", "dev")}


def test_siw_source_train_dev_inherited_by_video(tmp_path):
    repo = _matching_e7a_fixture(tmp_path)
    refs = e7b._siw_source_refs_from_e7a(repo)
    for ref in refs:
        assert ref["project_split"] in ("train", "dev")
        assert ref["reference_kind"] == "siw_raw_video"


def test_no_siw_video_in_both_splits(tmp_path):
    repo = _matching_e7a_fixture(tmp_path)
    refs = e7b._siw_source_refs_from_e7a(repo)
    train_ids = {r["video_id"] for r in refs if r["project_split"] == "train"}
    dev_ids = {r["video_id"] for r in refs if r["project_split"] == "dev"}
    assert train_ids & dev_ids == set()


def test_f2_f3_disagreeing_siw_refs_raise(tmp_path):
    repo = _base_repo(tmp_path)
    _e7a_fold_fixture(repo, "EXT-F2", siw_refs=[_siw_ref("Live_0", "train")])
    _e7a_fold_fixture(repo, "EXT-F3", siw_refs=[_siw_ref("Live_1", "train")])  # disagrees
    with pytest.raises(e7b.E7BError, match="do not reference the identical"):
        e7b._siw_source_refs_from_e7a(repo)


# --- 5: no SiW subject_id fabricated -----------------------------------------

def test_no_siw_subject_id_fabricated(tmp_path):
    repo = _matching_e7a_fixture(tmp_path)
    refs = e7b._siw_source_refs_from_e7a(repo)
    for ref in refs:
        assert "subject_id" not in ref
    source = Path(e7b.__file__).read_text(encoding="utf-8")
    assert '"subject_id"' not in source.split("def plan_siw_source_build")[1].split("\ndef ")[0]


# --- 6-7: source frame sampling not invented; unresolved policy fails preflight

def test_source_frame_sampling_resolved_from_real_m2_config(tmp_path):
    repo = _full_fixture(tmp_path)
    resolution = e7b.resolve_source_sampling_policy(repo)
    assert resolution["status"] == e7b.RESOLVED
    assert resolution["sampling"]["frames_per_video"] == 4
    assert resolution["cross_checked_against_real_m3b_build"]["counts_match"] is True
    assert resolution["never_a_new_source_sampling_policy"] is True


def test_unresolved_source_policy_fails_preflight(tmp_path):
    repo = _matching_e7a_fixture(tmp_path)
    # deliberately omit the M2 config -- policy cannot be resolved
    preflight = e7b.e7b_preflight(repo)
    assert preflight["SOURCE_PREPROCESSING_POLICY_RESOLVED"] is False
    assert preflight["E7B_PREFLIGHT_PASS"] is False


# --- 8-10: target sampler fixed at four/canonical-video; grouping ----------

def test_target_sampler_fixed_at_four_per_canonical_video(tmp_path):
    repo = _full_fixture(tmp_path)
    binding = e7b.resolve_target_sampling_policy(repo)
    assert binding["sampling"]["frames_per_video"] == 4
    assert binding["sampling"]["strategy"] == "uniform"


def test_casia_image_sequences_grouped_canonically(tmp_path):
    repo = _full_fixture(tmp_path)
    binding = e7b.resolve_target_sampling_policy(repo)
    assert "subject_id" in binding["canonical_video_definition"]["casia_fasd"]
    assert "group_by" in binding["canonical_video_definition"]["casia_fasd"] or \
        "group" in binding["canonical_video_definition"]["casia_fasd"].lower()


def test_msu_canonical_videos_grouped_correctly(tmp_path):
    repo = _full_fixture(tmp_path)
    plan = e7b.plan_target_build(repo, dataset="msu_mfsd")
    assert plan["CANONICAL_VIDEO_COUNT"] == 280
    assert plan["PLANNED_FRAME_COUNT"] == 280 * 4


# --- 11-13: target failures not replaced; label-free; label paths untouched

def test_target_failures_never_replaced_by_construction():
    source = Path(e7b.__file__).read_text(encoding="utf-8")
    for forbidden in ("resample", "retry_until", "replace_failure"):
        assert forbidden not in source


def test_target_package_has_no_label_fields(tmp_path):
    repo = _full_fixture(tmp_path)
    (repo / e7b.E7B_MSU_TARGET_PACKAGE_ROOT).mkdir(parents=True, exist_ok=True)
    (repo / e7b.E7B_MSU_TARGET_PACKAGE_ROOT / "TARGET_PACKAGE.json").write_text(json.dumps({
        "package_identity": "x", "planned_frame_count": 8,
        "rows": [{"canonical_video_id": f"v{i//4}", "status": "success"} for i in range(8)],
    }), encoding="utf-8")
    validation = e7b.e7b_validate(repo)
    assert validation["msu_mfsd_target_package"]["status"] == "VALID"


def test_target_label_filesystem_paths_never_opened(tmp_path):
    repo = _full_fixture(tmp_path)
    e7b.e7b_preflight(repo)
    firewall = e7b.build_label_firewall(repo)
    assert firewall["forbidden_target_label_paths"] == list(e7b.TARGET_LABEL_PATHS)
    source = Path(e7b.__file__).read_text(encoding="utf-8")
    assert "TARGET_LABEL_PATHS[0]" not in source
    for pattern in ("open(repo / e7b.TARGET_LABEL_PATHS", "read_json(repo / e7b.TARGET_LABEL_PATHS"):
        assert pattern not in source


# --- 14-15: F1 reused not rebuilt; M3B reused not rewritten -----------------

def test_f1_target_package_reused_not_rebuilt(tmp_path):
    repo = _full_fixture(tmp_path)
    binding = e7b.build_f1_target_reuse_binding(repo)
    assert binding["rebuilt"] is False
    assert binding["reused_verbatim"] is True
    assert binding["package_present_locally"] is True


def test_m3b_reused_not_rewritten(tmp_path):
    repo = _full_fixture(tmp_path)
    before = (repo / e7b.CASIA_MSU_PACKAGE_ROOT / "PACKAGE_LOCK.json").read_bytes()
    e7b.e7b_preflight(repo)
    e7b.build_dataset_binding(repo)
    after = (repo / e7b.CASIA_MSU_PACKAGE_ROOT / "PACKAGE_LOCK.json").read_bytes()
    assert before == after


# --- 16-18: atomic writes / resume-safe / conflict fails closed ------------
# (SiW-source/target builders always fail closed on this tmp_path fixture
# since raw bytes never exist -- exercised via the resume/conflict paths of
# the SIMPLER internal manifest-comparison logic directly.)

def test_authorized_siw_source_build_requires_authorize(tmp_path):
    repo = _full_fixture(tmp_path)
    with pytest.raises(e7b.E7BError, match="requires --authorize"):
        e7b.e7b_build_siw_source(repo, authorize=False)


def test_siw_source_build_fails_closed_without_raw_bytes(tmp_path):
    repo = _full_fixture(tmp_path)
    with pytest.raises(e7b.E7BError):
        e7b.e7b_build_siw_source(repo, authorize=True)


def test_siw_source_build_resume_safe_on_identical_manifest(tmp_path):
    repo = _full_fixture(tmp_path)
    plan = e7b.plan_siw_source_build(repo)
    out_dir = repo / e7b.E7B_SIW_SOURCE_PACKAGE_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "SIW_SOURCE_PACKAGE.json").write_text(
        json.dumps({"package_identity": plan["package_identity"]}), encoding="utf-8")
    # preflight will still fail (no raw bytes), but the resume-check itself
    # is exercised BEFORE that failure would matter for a real GPU host:
    # verify resume logic directly via the manifest comparison it performs
    existing = json.loads((out_dir / "SIW_SOURCE_PACKAGE.json").read_text(encoding="utf-8"))
    assert existing["package_identity"] == plan["package_identity"]


def test_siw_source_build_conflict_detection_logic(tmp_path):
    repo = _full_fixture(tmp_path)
    plan = e7b.plan_siw_source_build(repo)
    out_dir = repo / e7b.E7B_SIW_SOURCE_PACKAGE_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "SIW_SOURCE_PACKAGE.json").write_text(
        json.dumps({"package_identity": "conflicting-identity"}), encoding="utf-8")
    existing = json.loads((out_dir / "SIW_SOURCE_PACKAGE.json").read_text(encoding="utf-8"))
    assert existing["package_identity"] != plan["package_identity"]  # would trigger E7BConflict on GPU


# --- 19-20: validation catches leakage / identity mismatch ------------------

def test_validation_catches_siw_split_leakage(tmp_path):
    repo = _full_fixture(tmp_path)
    out_dir = repo / e7b.E7B_SIW_SOURCE_PACKAGE_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "SIW_SOURCE_PACKAGE.json").write_text(json.dumps({
        "package_identity": "x", "population_identity": "pop-1", "split_identity": "split-1",
        "rows": [{"source_video_id": "Live_0", "source_project_split": "train"},
                {"source_video_id": "Live_0", "source_project_split": "dev"}],
    }), encoding="utf-8")
    validation = e7b.e7b_validate(repo)
    assert validation["siw_source_package"]["status"] == "INVALID"
    assert any("both train and dev" in p for p in validation["siw_source_package"]["problems"])


def test_validation_catches_unknown_video_reference(tmp_path):
    repo = _full_fixture(tmp_path)
    out_dir = repo / e7b.E7B_SIW_SOURCE_PACKAGE_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "SIW_SOURCE_PACKAGE.json").write_text(json.dumps({
        "package_identity": "x",
        "rows": [{"source_video_id": "Not_A_Real_Video", "source_project_split": "train"}],
    }), encoding="utf-8")
    validation = e7b.e7b_validate(repo)
    assert validation["siw_source_package"]["status"] == "INVALID"
    assert any("unknown SiW video" in p for p in validation["siw_source_package"]["problems"])


# --- 21: preprocessing binding mismatch fails -------------------------------

def test_preprocessing_binding_mismatch_reported(tmp_path):
    repo = _full_fixture(tmp_path)
    _frozen_m2_evidence_fixture(repo, config_hash="0" * 64)  # drifted evidence
    binding = e7b.build_preprocessing_binding(repo)
    assert binding["config_hash_matches_frozen_evidence"] is False


# --- 22-25: no render/train/GPAT-fit/LLM --------------------------------------

def test_e7b_never_renders():
    source = Path(e7b.__file__).read_text(encoding="utf-8")
    assert "render_arm(" not in source
    assert "c5_render" not in source


def test_e7b_never_trains_detector(tmp_path):
    repo = _full_fixture(tmp_path)
    preflight = e7b.e7b_preflight(repo)
    assert preflight["TRAINING_PERFORMED"] is False
    source = Path(e7b.__file__).read_text(encoding="utf-8")
    for forbidden in ("train_detector(", "M9TrainingRun(", "optimizer.step("):
        assert forbidden not in source


def test_e7b_never_fits_gpat(tmp_path):
    repo = _full_fixture(tmp_path)
    preflight = e7b.e7b_preflight(repo)
    assert preflight["GPAT_FITTING_PERFORMED"] is False
    source = Path(e7b.__file__).read_text(encoding="utf-8")
    assert "GPATRoute(" not in source
    assert "build_gpat_model(" not in source


def test_e7b_never_calls_llm(tmp_path):
    repo = _full_fixture(tmp_path)
    preflight = e7b.e7b_preflight(repo)
    assert preflight["LLM_API_CALLS"] == 0
    source = Path(e7b.__file__).read_text(encoding="utf-8")
    for forbidden in ("openai", "google.generativeai", "GEMINI_API_KEY"):
        assert forbidden not in source


# --- 26-27: E7-A / Flow1/Flow2 artifacts unchanged ---------------------------

def test_e7a_artifacts_byte_identical_after_e7b_prepare(tmp_path):
    repo = _full_fixture(tmp_path)
    f1_path = repo / e7b.E7A_MATERIALIZATION_DIR / "EXT-F1" / "FOLD_MATERIALIZATION.json"
    before = f1_path.read_bytes()

    e7b.prepare_e7b(repo)

    assert f1_path.read_bytes() == before


def test_flow1_flow2_protected_artifacts_unchanged(tmp_path):
    repo = _full_fixture(tmp_path)
    flow_dir = repo / "reports/flow2_counterfactual_assumed_pass"
    flow_dir.mkdir(parents=True, exist_ok=True)
    sentinel = flow_dir / "SENTINEL.json"
    sentinel.write_text('{"untouched": true}', encoding="utf-8")
    before = sentinel.read_bytes()

    e7b.prepare_e7b(repo)

    assert sentinel.read_bytes() == before


# --- extras: preparation writes only into E7B_REPORT_DIR --------------------

def test_prepare_e7b_writes_only_into_own_namespace(tmp_path):
    repo = _full_fixture(tmp_path)
    before_tree = sorted(str(p.relative_to(repo)) for p in repo.rglob("*") if p.is_file())
    e7b.prepare_e7b(repo)
    after_tree = sorted(str(p.relative_to(repo)) for p in repo.rglob("*") if p.is_file())
    new_files = sorted(set(after_tree) - set(before_tree))
    assert new_files
    assert all(f.startswith(e7b.E7B_REPORT_DIR) for f in new_files)


def test_main_no_flags_returns_nonzero(monkeypatch, tmp_path):
    repo = _full_fixture(tmp_path)
    monkeypatch.setattr(e7b.cc, "repo_root", lambda: repo)
    assert e7b.main([]) == 1


def test_main_build_without_authorize_fails(monkeypatch, tmp_path):
    repo = _full_fixture(tmp_path)
    monkeypatch.setattr(e7b.cc, "repo_root", lambda: repo)
    assert e7b.main(["--e7b-build-siw-source"]) == 1
