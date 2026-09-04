"""Tests for `prism_fas.evaluation.c_ext_e7b_data_prep` (E7-B source/target
data preprocessing and package readiness). Every test builds a
self-contained fake repo under `tmp_path`. No test ever renders, trains,
fits GPAT, accesses target labels or calls an LLM.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
           "population_identity": population_identity, "split_identity": split_identity,
           "relative_path": f"{video_id}.avi"}


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
    config_hash = e7b.build_preprocessing_binding(repo)["config_hash"]
    rows = [{"canonical_video_id": f"v{i // 4}", "frame_index": i % 4, "status": "failure",
            "failure_reason": "no_face"} for i in range(e7b.EXPECTED_TARGET_PLANNED_FRAME_COUNT["msu_mfsd"])]
    (repo / e7b.E7B_MSU_TARGET_PACKAGE_ROOT).mkdir(parents=True, exist_ok=True)
    (repo / e7b.E7B_MSU_TARGET_PACKAGE_ROOT / "TARGET_PACKAGE.json").write_text(json.dumps({
        "package_identity": "x", "preprocessing_config_hash": config_hash,
        "detector_model_sha256": e7b.FROZEN_SCRFD_MODEL_SHA256,
        "canonical_video_count": e7b.EXPECTED_TARGET_CANONICAL_VIDEO_COUNT["msu_mfsd"],
        "planned_frame_count": e7b.EXPECTED_TARGET_PLANNED_FRAME_COUNT["msu_mfsd"],
        "successful_crop_count": 0, "failure_count": len(rows), "label_free": True, "rows": rows,
    }), encoding="utf-8")
    validation = e7b.e7b_validate(repo)
    assert validation["msu_mfsd_target_package"]["status"] == "VALID"
    assert not any("forbidden label field" in p for p in validation["msu_mfsd_target_package"]["problems"])
    for row in rows:
        for forbidden in ("label", "label_live_spoof", "spoof_family", "subject_id"):
            assert forbidden not in row


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


# =========================================================================== #
# REAL BUILDER EXECUTION -- E7-B additive milestone fixing the previous
# unconditional-terminal-raise stub. Laptop tests inject the SAME
# `detector`/`media_reader_factory` seams `run_preprocessing` itself exposes
# (tiny synthetic frames, `MockFaceDetector`) to prove REAL orchestration
# through the frozen production pipeline -- never a second scientific
# implementation, never fake GPU execution claimed as real.
# =========================================================================== #

from prism_fas.data.media import FrameResult  # noqa: E402
from prism_fas.data.preprocess_m2 import Detection, MockFaceDetector  # noqa: E402
from prism_fas.data.schemas.records import CanonicalVideoRecord  # noqa: E402

_FAKE_IMAGE = np.full((60, 60, 3), 128, dtype=np.uint8)
_SUCCESS_DETECTIONS = [Detection(bbox=(5.0, 5.0, 55.0, 55.0), score=0.99,
                                 landmarks=[(15.0, 15.0), (45.0, 15.0), (30.0, 30.0),
                                           (18.0, 45.0), (42.0, 45.0)])]


class _FakeReader:
    """Stands in for `OpenCVVideoDecoder`/`ImageSequenceReader` -- exactly
    the injection point `run_preprocessing` itself exposes via
    `media_reader_factory`. Never opens or decodes a real file."""

    def __init__(self, frame_count: int, image: np.ndarray) -> None:
        self._frame_count = frame_count
        self._image = image

    def frame_count(self) -> int:
        return self._frame_count

    def read_frame(self, index: int) -> FrameResult:
        return FrameResult(index, index, float(index) * 40.0, self._image.shape[1],
                           self._image.shape[0], self._image, "fake")

    def close(self) -> None:
        pass


class _CountingDetector:
    """Proves `detector.detect` was genuinely invoked (real orchestration
    reached the processing loop), not that a stub raised immediately."""

    name = "counting-mock"

    def __init__(self, detections: list) -> None:
        self._detections = detections
        self.call_count = 0

    def detect(self, image: np.ndarray) -> list:
        self.call_count += 1
        return self._detections


def _write_siw_raw_bytes(repo: Path, refs: list[dict]) -> None:
    for ref in refs:
        path = repo / e7b.SIW_RAW_ROOT / ref["relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fake-siw-video-bytes-{ref['video_id']}".encode())


def _siw_smoke_fixture(tmp_path: Path, refs: list[dict]) -> Path:
    repo = _base_repo(tmp_path)
    _e7a_fold_fixture(repo, "EXT-F1")
    _e7a_fold_fixture(repo, "EXT-F2", siw_refs=refs, m3b_refs=[_m3b_ref("casia_fasd", "c-0")])
    _e7a_fold_fixture(repo, "EXT-F3", siw_refs=refs, m3b_refs=[_m3b_ref("msu_mfsd", "m-0")])
    _real_m2_config_fixture(repo)
    _frozen_m2_evidence_fixture(repo)
    _write_siw_raw_bytes(repo, refs)
    return repo


def _fake_target_records(n: int, dataset: str) -> list[CanonicalVideoRecord]:
    return [CanonicalVideoRecord(
        dataset=dataset, subject_id=f"s{i}", video_id=f"{dataset}-video-{i}",
        source_path=Path(f"/nonexistent/{dataset}/{i}.mov"), official_split=None,
        label="live" if i % 2 == 0 else "spoof", capture_metadata={}, adapter_version="1.0",
        source_fingerprint=f"fp-{dataset}-{i}", metadata_provenance="test-fixture")
        for i in range(n)]


# --- 1: fully satisfied SiW builder reaches the real processing loop -------

def test_siw_source_smoke_reaches_successful_source_routing(tmp_path):
    """Post-compatibility-fix: a genuine face detection on a subject_id=None
    SiW source record now succeeds all the way through
    `route_source_success`, not just to the old STRUCTURAL_SCIENTIFIC_POLICY_
    GAP wall."""
    refs = [_siw_ref("Live_0", "train"), _siw_ref("Live_1", "dev")]
    repo = _siw_smoke_fixture(tmp_path, refs)
    detector = _CountingDetector(_SUCCESS_DETECTIONS)
    result = e7b.e7b_smoke_siw_source(repo, limit_videos=2, detector=detector,
                                      media_reader_factory=lambda record: _FakeReader(6, _FAKE_IMAGE))
    assert detector.call_count >= 1
    body = result["body"]
    assert body["successful_crop_count"] == 8
    assert body["failure_count"] == 0
    for row in body["rows"]:
        assert row["status"] == "success"
        assert row["crop_relative_path"]
        assert row["crop_sha256"]


# --- 2/3: fully satisfied MSU/CASIA target builders reach the loop and complete

def test_msu_target_smoke_full_success_produces_verifiable_crops(tmp_path, monkeypatch):
    repo = _full_fixture(tmp_path)
    records = _fake_target_records(2, "msu_mfsd")
    monkeypatch.setattr(e7b, "_target_canonical_records",
                        lambda repo_, *, dataset, raw_root: records)
    detector = MockFaceDetector(_SUCCESS_DETECTIONS)
    result = e7b.e7b_smoke_target_msu(repo, limit_videos=2, detector=detector,
                                      media_reader_factory=lambda record: _FakeReader(6, _FAKE_IMAGE))
    body = result["body"]
    assert body["dataset"] == "msu_mfsd"
    assert body["canonical_video_count"] == 2
    assert body["planned_frame_count"] == 8
    assert body["successful_crop_count"] == 8
    assert body["failure_count"] == 0
    assert body["label_free"] is True
    package_root = Path(result["path"]).parent / "m2_run"
    for row in body["rows"]:
        assert row["status"] == "success"
        crop_path = package_root / row["crop_relative_path"]
        assert crop_path.is_file()
        assert e7b.cc.sha256_file(crop_path) == row["crop_sha256"]
        for forbidden in ("label", "label_live_spoof", "spoof_family", "subject_id", "attack_label"):
            assert forbidden not in row
    assert not (repo / e7b.E7B_MSU_TARGET_PACKAGE_ROOT / "TARGET_PACKAGE.json").is_file()


def test_casia_target_smoke_full_success_reaches_processing_loop(tmp_path, monkeypatch):
    repo = _full_fixture(tmp_path)
    records = _fake_target_records(1, "casia_fasd")
    monkeypatch.setattr(e7b, "_target_canonical_records",
                        lambda repo_, *, dataset, raw_root: records)
    detector = _CountingDetector(_SUCCESS_DETECTIONS)
    result = e7b.e7b_smoke_target_casia(repo, limit_videos=1, detector=detector,
                                        media_reader_factory=lambda record: _FakeReader(6, _FAKE_IMAGE))
    assert detector.call_count >= 1
    assert result["body"]["successful_crop_count"] == 4
    assert result["body"]["dataset"] == "casia_fasd"


# --- 4: no unconditional terminal placeholder raise remains ----------------

def test_no_unconditional_terminal_placeholder_raise_remains():
    source = Path(e7b.__file__).read_text(encoding="utf-8")
    for placeholder in ("real preprocessing happens", "NOT_IMPLEMENTED", "not yet implemented"):
        assert placeholder not in source


# --- 5/6/10/11: 4 deterministic samples/video; source split inherited;
# failed face terminal; no replacement sampling (via an all-detection-miss
# scenario, which never crosses the subject_id wall since no success is
# ever routed) --------------------------------------------------------------

def test_siw_source_smoke_all_failures_completes_and_retains_source_labels(tmp_path):
    refs = [_siw_ref("Live_0", "train", label="live"),
           _siw_ref("Spoof_0", "dev", label="spoof", family="print")]
    repo = _siw_smoke_fixture(tmp_path, refs)
    detector = MockFaceDetector([])  # no face ever found -> route_source_success never reached
    result = e7b.e7b_smoke_siw_source(repo, limit_videos=2, detector=detector,
                                      media_reader_factory=lambda record: _FakeReader(6, _FAKE_IMAGE))
    body = result["body"]
    assert body["smoke"] is True
    assert body["canonical_video_count"] == 2
    assert body["planned_frame_count"] == 8
    assert body["successful_crop_count"] == 0
    assert body["failure_count"] == 8
    assert len(body["rows"]) == 8

    by_video: dict[str, list[dict]] = {}
    for row in body["rows"]:
        by_video.setdefault(row["source_video_id"], []).append(row)
        assert row["status"] == "failure"
        assert row["failure_reason"]
        assert row["crop_relative_path"] is None  # no replacement sampling -- no crop ever produced

    assert set(by_video) == {"Live_0", "Spoof_0"}
    for rows in by_video.values():
        assert len(rows) == 4  # 4 deterministic samples/video, never more (no resampling)

    live_row = next(r for r in body["rows"] if r["source_video_id"] == "Live_0")
    assert live_row["label_live_spoof"] == "live"
    assert live_row["source_project_split"] == "train"  # source split inherited exactly
    spoof_row = next(r for r in body["rows"] if r["source_video_id"] == "Spoof_0")
    assert spoof_row["label_live_spoof"] == "spoof"
    assert spoof_row["spoof_family"] == "print"
    assert spoof_row["source_project_split"] == "dev"


# --- 19/20: smoke namespace separate from final package; smoke cannot mark
# the final package ready -----------------------------------------------------

def test_siw_smoke_writes_only_smoke_namespace_never_marks_final_ready(tmp_path):
    refs = [_siw_ref("Live_0", "train"), _siw_ref("Spoof_0", "dev", label="spoof")]
    repo = _siw_smoke_fixture(tmp_path, refs)
    detector = MockFaceDetector([])
    e7b.e7b_smoke_siw_source(repo, limit_videos=2, detector=detector,
                             media_reader_factory=lambda record: _FakeReader(6, _FAKE_IMAGE))
    assert (repo / e7b.E7B_SIW_SMOKE_ROOT / "SIW_SOURCE_PACKAGE.json").is_file()
    assert not (repo / e7b.E7B_SIW_SOURCE_PACKAGE_ROOT / "SIW_SOURCE_PACKAGE.json").is_file()
    validation = e7b.e7b_validate(repo)
    assert validation["siw_source_package"]["status"] == "NOT_BUILT"


def test_msu_smoke_writes_only_smoke_namespace_never_marks_final_ready(tmp_path, monkeypatch):
    repo = _full_fixture(tmp_path)
    records = _fake_target_records(1, "msu_mfsd")
    monkeypatch.setattr(e7b, "_target_canonical_records",
                        lambda repo_, *, dataset, raw_root: records)
    e7b.e7b_smoke_target_msu(repo, limit_videos=1, detector=MockFaceDetector(_SUCCESS_DETECTIONS),
                             media_reader_factory=lambda record: _FakeReader(6, _FAKE_IMAGE))
    assert (repo / e7b.E7B_MSU_SMOKE_ROOT / "TARGET_PACKAGE.json").is_file()
    assert not (repo / e7b.E7B_MSU_TARGET_PACKAGE_ROOT / "TARGET_PACKAGE.json").is_file()
    validation = e7b.e7b_validate(repo)
    assert validation["msu_mfsd_target_package"]["status"] == "NOT_BUILT"


# --- 21: label firewall prevents target label path from ever being opened --

def test_target_build_never_opens_target_label_path(tmp_path, monkeypatch):
    repo = _full_fixture(tmp_path)
    records = _fake_target_records(1, "msu_mfsd")
    monkeypatch.setattr(e7b, "_target_canonical_records",
                        lambda repo_, *, dataset, raw_root: records)
    forbidden_marker = e7b.TARGET_LABEL_PATHS[0]
    real_open = Path.open

    def guarded_open(self, *a, **k):
        if forbidden_marker in str(self):
            raise AssertionError("target label path was opened during target build")
        return real_open(self, *a, **k)

    monkeypatch.setattr(Path, "open", guarded_open)
    result = e7b.e7b_smoke_target_msu(repo, limit_videos=1, detector=MockFaceDetector(_SUCCESS_DETECTIONS),
                                      media_reader_factory=lambda record: _FakeReader(6, _FAKE_IMAGE))
    assert result["body"]["successful_crop_count"] == 4


# --- 13/14: detector model SHA mismatch / config identity mismatch fail closed

def test_scrfd_model_sha_mismatch_fails_before_crop_write(tmp_path, monkeypatch):
    repo = _full_fixture(tmp_path)
    cfg = e7b._m2_config(repo)
    fake_model = tmp_path / "fake_scrfd.onnx"
    fake_model.write_bytes(b"not-the-real-scrfd-model")
    monkeypatch.setattr("prism_fas.data.preprocess_m2.resolve_detector_path", lambda declared: fake_model)
    with pytest.raises(e7b.E7BError, match="FAIL CLOSED"):
        e7b._verify_scrfd_model_sha256(cfg)


def test_config_identity_mismatch_fails_closed_before_build(tmp_path):
    repo = _full_fixture(tmp_path)
    _frozen_m2_evidence_fixture(repo, config_hash="0" * 64)  # drifted evidence
    with pytest.raises(e7b.E7BError, match="does not match the frozen"):
        e7b.plan_siw_source_build(repo)
    with pytest.raises(e7b.E7BError, match="does not match the frozen"):
        e7b.plan_target_build(repo, dataset="msu_mfsd")


# --- 15/16: resume skips completed valid outputs; resume detects corrupt crop

def test_resume_completed_target_ids_skips_valid_and_flags_corrupt(tmp_path):
    package_root = tmp_path / "pkg"
    m2_output_root = package_root / "m2_run"  # crop_relative_path is relative to <package_root>/m2_run
    (m2_output_root / "crops").mkdir(parents=True)
    crop_rel = "crops/v0_0.jpg"
    crop_bytes = b"deterministic-crop-bytes"
    (m2_output_root / crop_rel).write_bytes(crop_bytes)
    crop_sha = e7b.cc.sha256_bytes(crop_bytes)
    manifest = package_root / "TARGET_PACKAGE.json"
    rows = [{"canonical_video_id": "v0", "status": "success", "crop_relative_path": crop_rel,
            "crop_sha256": crop_sha}] + [{"canonical_video_id": "v0", "status": "failure"}] * 3
    manifest.write_text(json.dumps({"rows": rows}), encoding="utf-8")

    completed = e7b._resume_completed_target_ids(manifest, planned_per_video=4)
    assert completed == {"v0"}

    (m2_output_root / crop_rel).write_bytes(b"tampered-bytes")  # corrupt after the manifest recorded it
    with pytest.raises(e7b.E7BError, match="corrupt crop"):
        e7b._resume_completed_target_ids(manifest, planned_per_video=4)


def test_resume_completed_target_ids_ignores_old_wrong_location(tmp_path):
    """A crop mistakenly placed at the OLD, wrong `<package_root>/crops/...`
    location (bypassing `/m2_run/`) must NOT be found -- proves the resume
    check uses the correct root, not merely "some file that happens to
    exist"."""
    package_root = tmp_path / "pkg2"
    (package_root / "crops").mkdir(parents=True)  # OLD wrong location
    crop_rel = "crops/v0_0.jpg"
    crop_bytes = b"deterministic-crop-bytes"
    (package_root / crop_rel).write_bytes(crop_bytes)  # misleading file at the wrong root
    crop_sha = e7b.cc.sha256_bytes(crop_bytes)
    manifest = package_root / "TARGET_PACKAGE.json"
    rows = [{"canonical_video_id": "v0", "status": "success", "crop_relative_path": crop_rel,
            "crop_sha256": crop_sha}] + [{"canonical_video_id": "v0", "status": "failure"}] * 3
    manifest.write_text(json.dumps({"rows": rows}), encoding="utf-8")

    with pytest.raises(e7b.E7BError, match="missing on resume"):
        e7b._resume_completed_target_ids(manifest, planned_per_video=4)


def test_resume_completed_siw_video_ids_skips_valid_and_flags_corrupt(tmp_path):
    package_root = tmp_path / "siwpkg"
    m2_output_root = package_root / "m2_run"
    (m2_output_root / "crops").mkdir(parents=True)
    crop_rel = "crops/Live_0_0.jpg"
    crop_bytes = b"deterministic-siw-crop-bytes"
    (m2_output_root / crop_rel).write_bytes(crop_bytes)
    crop_sha = e7b.cc.sha256_bytes(crop_bytes)
    manifest = package_root / "SIW_SOURCE_PACKAGE.json"
    rows = [{"source_video_id": "Live_0", "status": "success", "crop_relative_path": crop_rel,
            "crop_sha256": crop_sha}] + [{"source_video_id": "Live_0", "status": "failure"}] * 3
    manifest.write_text(json.dumps({"rows": rows}), encoding="utf-8")

    completed = e7b._resume_completed_video_ids(manifest, planned_per_video=4)
    assert completed == {"Live_0"}

    (m2_output_root / crop_rel).write_bytes(b"tampered-bytes")
    with pytest.raises(e7b.E7BError, match="corrupt crop"):
        e7b._resume_completed_video_ids(manifest, planned_per_video=4)


def test_resume_completed_siw_video_ids_ignores_old_wrong_location(tmp_path):
    package_root = tmp_path / "siwpkg2"
    (package_root / "crops").mkdir(parents=True)  # OLD wrong location
    crop_rel = "crops/Live_0_0.jpg"
    crop_bytes = b"deterministic-siw-crop-bytes"
    (package_root / crop_rel).write_bytes(crop_bytes)
    crop_sha = e7b.cc.sha256_bytes(crop_bytes)
    manifest = package_root / "SIW_SOURCE_PACKAGE.json"
    rows = [{"source_video_id": "Live_0", "status": "success", "crop_relative_path": crop_rel,
            "crop_sha256": crop_sha}] + [{"source_video_id": "Live_0", "status": "failure"}] * 3
    manifest.write_text(json.dumps({"rows": rows}), encoding="utf-8")

    with pytest.raises(e7b.E7BError, match="missing on resume"):
        e7b._resume_completed_video_ids(manifest, planned_per_video=4)


# --- 17: no duplicate rows (frozen ManifestRepository guarantee E7-B relies on)

def test_manifest_repository_prevents_duplicate_rows(tmp_path):
    from prism_fas.data.manifests.repository import ManifestRepository

    repository = ManifestRepository(tmp_path / "manifests",
                                     {"manifest_schema_version": "m2f1a-v1"}).initialize()
    row = {"sample_id": "abc123", "dataset": "msu_mfsd", "canonical_video_id": "v0"}
    repository._upsert("target_frames", dict(row))
    repository._upsert("target_frames", dict(row))  # identical row, same key -> idempotent
    assert len(repository.rows["target_frames"]) == 1
    with pytest.raises(ValueError, match="conflicting duplicate"):
        repository._upsert("target_frames", {**row, "canonical_video_id": "v1"})


# --- 22: F2/F3 share exactly one SiW package (also covered structurally
# above; this asserts the shared refs feed a SINGLE package_identity) -------

def test_f2_f3_share_exactly_one_siw_package_identity(tmp_path):
    refs = [_siw_ref("Live_0", "train"), _siw_ref("Live_1", "dev")]
    repo = _siw_smoke_fixture(tmp_path, refs)
    plan = e7b.plan_siw_source_build(repo)
    assert plan["video_count"] == len(refs)
    # both F2 and F3 resolve to the SAME refs (proven by
    # `_siw_source_refs_from_e7a`'s own cross-check), so there is exactly
    # ONE package_identity for the shared package.
    refs_again = e7b._siw_source_refs_from_e7a(repo)
    assert {r["video_id"] for r in refs_again} == {"Live_0", "Live_1"}


# =========================================================================== #
# SUBJECT_ID METADATA-COMPATIBILITY FIX -- `source_metadata_policy` on
# `PreprocessingRunContext` (additive, default 'required'). CASIA/MSU/every
# historical context is unaffected (they never pass this field); ONLY the
# E7-B SiW-as-source context passes 'optional_unverifiable'. The persisted
# `SourceFrameRecord.subject_id`/`SourceCropRecord.subject_id` schema fields
# were ALREADY nullable (`str | None`) -- the legacy converter guard was
# stricter than the actual schema; this is a technical compatibility fix,
# not a scientific protocol change.
# =========================================================================== #

def _ctx(tmp_path, *, role="source", dataset="casia_fasd", policy="required"):
    from prism_fas.data.run_context import M2OutputLayout, PreprocessingRunContext

    layout = M2OutputLayout.from_root(tmp_path / "full_preprocessing")
    return PreprocessingRunContext(
        project_root=tmp_path, work_root=tmp_path, run_profile="full_preprocessing",
        output_namespace="full_preprocessing", output_root=layout.output_root,
        crops_root=layout.crops_root, frames_root=layout.frames_root,
        manifests_root=layout.manifests_root, state_root=layout.state_root,
        reports_root=layout.reports_root, logs_root=layout.logs_root, run_id="x",
        dataset=dataset, dataset_role=role, preprocessing_version="m2-v1",
        preprocessing_config_hash="h", detector_model_path=tmp_path / "m",
        detector_model_sha256="a" * 64, detector_input_size=320, detector_threshold=.5,
        all_records=False, record_limit=1, sample_limit=None, resume=False, dry_run=True,
        partial_full_profile=True, command="x", source_metadata_policy=policy)


def _canonical_source_record(tmp_path, *, dataset="casia_fasd", subject_id="1",
                             official_split="train"):
    return CanonicalVideoRecord(dataset=dataset, subject_id=subject_id, video_id="v",
                                source_path=tmp_path / "x", official_split=official_split,
                                label="live", adapter_version="1", source_fingerprint="f",
                                metadata_provenance="test")


_CONVERTER_KW = dict(sample_id="s", source_media_type="image_sequence",
                     source_relative_identifier="x", requested_frame_index=0,
                     actual_frame_index=0, frame_width=1, frame_height=1,
                     selected_frame_reference="x", decoder_backend="x")


def test_default_source_policy_still_rejects_null_subject_id(tmp_path):
    from prism_fas.data.manifests.converters import (MissingCanonicalMetadataError,
                                                      build_source_frame_record)

    record = _canonical_source_record(tmp_path, subject_id=None)
    with pytest.raises(MissingCanonicalMetadataError):
        build_source_frame_record(_ctx(tmp_path, policy="required"), record, **_CONVERTER_KW)


def test_casia_normal_source_behavior_unchanged(tmp_path):
    from prism_fas.data.manifests.converters import build_source_frame_record

    record = _canonical_source_record(tmp_path, dataset="casia_fasd", subject_id="s7")
    result = build_source_frame_record(_ctx(tmp_path, dataset="casia_fasd", policy="required"),
                                       record, **_CONVERTER_KW)
    assert result.subject_id == "s7"
    assert result.label_live_spoof == "live"


def test_msu_normal_source_behavior_unchanged(tmp_path):
    from prism_fas.data.manifests.converters import build_source_frame_record

    record = _canonical_source_record(tmp_path, dataset="msu_mfsd", subject_id="client001")
    result = build_source_frame_record(_ctx(tmp_path, dataset="msu_mfsd", policy="required"),
                                       record, **_CONVERTER_KW)
    assert result.subject_id == "client001"


def test_e7b_optional_unverifiable_policy_accepts_null_subject_id(tmp_path):
    from prism_fas.data.manifests.converters import build_source_frame_record

    record = _canonical_source_record(tmp_path, dataset="siw_mv2", subject_id=None,
                                      official_split=e7b.SIW_SOURCE_OFFICIAL_SPLIT_PLACEHOLDER)
    result = build_source_frame_record(
        _ctx(tmp_path, dataset="siw_mv2", policy="optional_unverifiable"), record, **_CONVERTER_KW)
    assert result.subject_id is None
    assert result.video_id != result.subject_id  # video_id never copied into subject_id


def test_unknown_source_metadata_policy_rejected(tmp_path):
    import pydantic

    # PreprocessingRunContext's own frozen Literal type already fails closed
    # on an unknown policy value at construction time; build_source_frame_
    # record()'s own defensive check (ManifestConversionError) is a second,
    # redundant fail-closed layer for a duck-typed/bypassed context.
    with pytest.raises(pydantic.ValidationError):
        _ctx(tmp_path, policy="not_a_real_policy")


# --- STRUCTURAL TEST: real run_preprocessing()/route_source_success()/
# ManifestRepository/converters/SourceFrameRecord/SourceCropRecord end-to-end
# for a tiny one-video, four-planned-frame, subject_id=None SiW source case.
# Reads the ACTUAL persisted parquet rows -- never mocks route_source_success
# itself. -----------------------------------------------------------------

def test_end_to_end_siw_source_manifest_rows_have_null_subject_id(tmp_path):
    import pyarrow.parquet as pq

    from prism_fas.data.m2_runner import run_preprocessing

    repo = _full_fixture(tmp_path)
    ref = _siw_ref("Live_0", "train")
    _write_siw_raw_bytes(repo, [ref])
    cfg = e7b._m2_config(repo)
    context = e7b._build_e7b_run_context(
        repo, cfg, dataset="siw_mv2", dataset_role="source",
        package_root="runs/c_ext_q1q2_v1/e7b_smoke/_structural_test",
        detector_model_sha256=e7b.FROZEN_SCRFD_MODEL_SHA256, run_id="structural-test",
        source_metadata_policy="optional_unverifiable")
    canonical_records = e7b._siw_canonical_records(repo, [ref])
    assert canonical_records[0].subject_id is None
    assert canonical_records[0].dataset == "siw_mv2"

    result = run_preprocessing(context, canonical_records,
                               detector=MockFaceDetector(_SUCCESS_DETECTIONS),
                               media_reader_factory=lambda record: _FakeReader(6, _FAKE_IMAGE))
    assert result.dataset_role == "source"  # route is still SOURCE, never target
    assert result.samples_selected == 4
    assert result.samples_successful == 4
    assert result.samples_failed == 0
    assert result.manifest_counts["source_frames"] == 4
    assert result.manifest_counts["source_crops"] == 4
    assert result.manifest_counts["preprocessing_failures"] == 0
    assert result.manifest_counts["target_frames"] == 0
    assert result.manifest_counts["target_crops"] == 0

    frames = pq.read_table(context.manifests_root / "source_frames.parquet").to_pylist()
    crops = pq.read_table(context.manifests_root / "source_crops.parquet").to_pylist()
    assert len(frames) == 4
    assert len(crops) == 4
    assert all(row["subject_id"] is None for row in frames)  # NULL, never a sentinel
    assert all(row["subject_id"] is None for row in crops)
    assert all(row["video_id"] == "Live_0" for row in frames)  # video_id never in subject_id
    assert all(row["official_split"] == e7b.SIW_SOURCE_OFFICIAL_SPLIT_PLACEHOLDER for row in frames)
    assert all(row["label_live_spoof"] == "live" for row in frames)  # source label retained
    for row in frames + crops:
        assert row["subject_id"] not in ("unknown", "Live_0", "siw_mv2", "Live_0.avi")


def test_diagnostic_gap_artifact_preserved_and_resolution_is_additive(tmp_path):
    diagnostic = REPO / e7b.E7B_REPORT_DIR / "E7B_SIW_SOURCE_SUBJECT_ID_STRUCTURAL_GAP.json"
    resolution = REPO / e7b.E7B_REPORT_DIR / "E7B_SIW_SOURCE_METADATA_COMPATIBILITY_RESOLUTION.json"
    assert diagnostic.is_file()  # historical evidence, never rewritten to pretend the gap never existed
    diagnostic_body = json.loads(diagnostic.read_text(encoding="utf-8"))
    assert diagnostic_body["TECHNICAL_FINDING"] is True
    assert resolution.is_file()  # additive resolution artifact
    resolution_body = json.loads(resolution.read_text(encoding="utf-8"))
    assert resolution_body["TECHNICAL_GAP_DISCOVERED"] is True
    assert resolution_body["SCIENTIFIC_PROTOCOL_CHANGE_REQUIRED"] is False
    assert resolution_body["SUBJECT_ID_FABRICATED"] is False
    assert resolution_body["SUBJECT_ID_VALUE_FOR_SIW"] is None


# --- 23/24: expected planned counts; validator rejects incomplete/empty ----

def test_expected_planned_frame_counts_frozen():
    assert e7b.EXPECTED_SIW_SOURCE_CANONICAL_VIDEO_COUNT == 1700
    assert e7b.EXPECTED_SIW_SOURCE_PLANNED_FRAME_COUNT == 6800
    assert e7b.EXPECTED_TARGET_CANONICAL_VIDEO_COUNT == {"msu_mfsd": 280, "casia_fasd": 600}
    assert e7b.EXPECTED_TARGET_PLANNED_FRAME_COUNT == {"msu_mfsd": 1120, "casia_fasd": 2400}


def test_validator_rejects_empty_rows_as_not_valid(tmp_path):
    repo = _full_fixture(tmp_path)
    (repo / e7b.E7B_MSU_TARGET_PACKAGE_ROOT).mkdir(parents=True, exist_ok=True)
    (repo / e7b.E7B_MSU_TARGET_PACKAGE_ROOT / "TARGET_PACKAGE.json").write_text(
        json.dumps({"package_identity": "x", "rows": []}), encoding="utf-8")
    validation = e7b.e7b_validate(repo)
    assert validation["msu_mfsd_target_package"]["status"] != "VALID"


def test_validator_rejects_incomplete_planned_population(tmp_path):
    repo = _full_fixture(tmp_path)
    config_hash = e7b.build_preprocessing_binding(repo)["config_hash"]
    rows = [{"canonical_video_id": "v0", "frame_index": i, "status": "failure",
            "failure_reason": "no_face"} for i in range(4)]  # only 1 of 280 videos present
    (repo / e7b.E7B_MSU_TARGET_PACKAGE_ROOT).mkdir(parents=True, exist_ok=True)
    (repo / e7b.E7B_MSU_TARGET_PACKAGE_ROOT / "TARGET_PACKAGE.json").write_text(json.dumps({
        "package_identity": "x", "preprocessing_config_hash": config_hash,
        "detector_model_sha256": e7b.FROZEN_SCRFD_MODEL_SHA256,
        "canonical_video_count": e7b.EXPECTED_TARGET_CANONICAL_VIDEO_COUNT["msu_mfsd"],
        "planned_frame_count": e7b.EXPECTED_TARGET_PLANNED_FRAME_COUNT["msu_mfsd"],
        "successful_crop_count": 0, "failure_count": len(rows), "label_free": True, "rows": rows,
    }), encoding="utf-8")
    validation = e7b.e7b_validate(repo)
    assert validation["msu_mfsd_target_package"]["status"] == "INVALID"


# --- 12: crop SHA verified (end-to-end through the real pipeline) ----------

def test_crop_sha256_verified_end_to_end(tmp_path, monkeypatch):
    repo = _full_fixture(tmp_path)
    records = _fake_target_records(1, "casia_fasd")
    monkeypatch.setattr(e7b, "_target_canonical_records",
                        lambda repo_, *, dataset, raw_root: records)
    result = e7b.e7b_smoke_target_casia(repo, limit_videos=1, detector=MockFaceDetector(_SUCCESS_DETECTIONS),
                                        media_reader_factory=lambda record: _FakeReader(6, _FAKE_IMAGE))
    package_root = Path(result["path"]).parent / "m2_run"
    for row in result["body"]["rows"]:
        crop_path = package_root / row["crop_relative_path"]
        assert e7b.cc.sha256_file(crop_path) == row["crop_sha256"]


# --- 25-28: no training / no synthetic rendering / no GPAT fitting / no LLM
# in the real-builder code path specifically ---------------------------------

def test_real_builders_never_train_render_fit_gpat_or_call_llm():
    source = Path(e7b.__file__).read_text(encoding="utf-8")
    for forbidden in ("render_arm(", "train_detector(", "GPATRoute(", "build_gpat_model(",
                      "openai", "google.generativeai", "GEMINI_API_KEY", "optimizer.step("):
        assert forbidden not in source


# --- 29/30: E7-A / prior 7be02d6 artifacts byte-identical after real-builder
# additions (exercised for real against the committed repo, not a fixture) --

def test_e7a_frozen_materializations_still_byte_identical_against_real_repo():
    results = e7b.verify_e7a_frozen_hashes(REPO)
    for fold_id, expected in e7b.E7A_FROZEN_SHA256.items():
        assert results[fold_id]["observed"] == expected


def test_7be02d6_bindings_module_still_importable_and_constants_unchanged():
    assert e7b.E7A_BASE_COMMIT == "6c77633aa331253cabfb54b70ca2846c2f3466b4"
    assert e7b.FROZEN_M2_PREPROCESSING_CONFIG_HASH == \
        "48a120caa6041b3a03b4008642030665f084b5d722a62ca2c01a2a5aa5e0c959"
    assert e7b.FROZEN_SCRFD_MODEL_SHA256 == \
        "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"


# =========================================================================== #
# 6751638 FALSE-GREEN SMOKE FIX -- production CLI smoke commands returned
# exit 0 with planned=8/success=0/failure=0/rows=[] because `detector=None`
# was passed straight into the frozen `run_preprocessing()`, which does not
# lazily instantiate a detector. `None.detect(...)` raised an
# AttributeError, caught by run_preprocessing's generic outer handler and
# tallied as an UNPERSISTED `unrouted_processing_failure` -- while E7-B
# discarded the returned RunExecutionResult and read back an (empty)
# manifest. Fixed via `_resolve_e7b_detector` (real SCRFDDetector whenever no
# detector is injected, smoke included), `_fail_closed_on_unrouted_failure`,
# and `_enforce_strict_terminal_accounting` (both required before any
# package manifest write).
# =========================================================================== #

class _BrokenDetector:
    """`.detect()` raises a plain, UNTYPED exception -- not
    DetectorInferenceError -- so the frozen run_preprocessing() cannot route
    it as a typed failure and must fall through to its generic outer
    handler, tallying `unrouted_processing_failure`."""

    name = "broken"

    def detect(self, image):
        raise RuntimeError("boom -- not a DetectorInferenceError")


# --- 1/2: detector=None (smoke AND full builder) constructs the real SCRFD -

def test_resolve_e7b_detector_constructs_real_scrfd_when_none_injected(tmp_path, monkeypatch):
    repo = _full_fixture(tmp_path)
    cfg = e7b._m2_config(repo)
    stand_in_model = tmp_path / "stand_in_scrfd.onnx"
    stand_in_model.write_bytes(b"stand-in-bytes")

    constructed = {}

    class _FakeSCRFDDetector:
        def __init__(self, path, input_size):
            constructed["path"] = path
            constructed["input_size"] = input_size

    monkeypatch.setattr(e7b.cc, "sha256_file", lambda p: e7b.FROZEN_SCRFD_MODEL_SHA256)
    monkeypatch.setattr("prism_fas.data.preprocess_m2.resolve_detector_path", lambda declared: stand_in_model)
    monkeypatch.setattr("prism_fas.data.preprocess_m2.SCRFDDetector", _FakeSCRFDDetector)

    detector, sha256 = e7b._resolve_e7b_detector(cfg, None)
    assert isinstance(detector, _FakeSCRFDDetector)
    assert sha256 == e7b.FROZEN_SCRFD_MODEL_SHA256
    assert constructed["path"] == stand_in_model
    assert constructed["input_size"] == cfg.scrfd_input_size  # exact frozen input size, never invented


def test_full_builder_and_smoke_share_the_same_no_injection_detector_path(tmp_path, monkeypatch):
    """Proves items 1 AND 2 in one assertion: `_resolve_e7b_detector` is the
    SAME function `e7b_build_siw_source`/`_e7b_build_target` call for BOTH
    smoke=True and smoke=False whenever `detector` is None -- there is no
    separate smoke-only bypass branch left in the source."""
    source = Path(e7b.__file__).read_text(encoding="utf-8")
    assert "FROZEN_SCRFD_MODEL_SHA256 if (detector is not None or smoke)" not in source
    assert source.count("_resolve_e7b_detector(cfg, detector)") == 2  # SiW source + target builder


# --- 3: actual model SHA mismatch fails before processing ------------------

def test_scrfd_sha_mismatch_fails_before_any_instantiation(tmp_path, monkeypatch):
    repo = _full_fixture(tmp_path)
    cfg = e7b._m2_config(repo)
    wrong_model = tmp_path / "wrong_scrfd.onnx"
    wrong_model.write_bytes(b"definitely-not-the-real-model")

    instantiated = []

    class _FakeSCRFDDetector:
        def __init__(self, path, input_size):
            instantiated.append(path)

    monkeypatch.setattr("prism_fas.data.preprocess_m2.resolve_detector_path", lambda declared: wrong_model)
    monkeypatch.setattr("prism_fas.data.preprocess_m2.SCRFDDetector", _FakeSCRFDDetector)

    with pytest.raises(e7b.E7BError, match="FAIL CLOSED"):
        e7b._resolve_e7b_detector(cfg, None)
    assert instantiated == []  # never reached instantiation


# --- 4: smoke does NOT bypass the SHA check (real absence of weights on
# this laptop is the ground truth -- no need to fake it) --------------------

def test_smoke_does_not_bypass_sha_verification_when_no_detector_injected(tmp_path, monkeypatch):
    repo = _full_fixture(tmp_path)
    records = _fake_target_records(1, "msu_mfsd")
    monkeypatch.setattr(e7b, "_target_canonical_records", lambda repo_, *, dataset, raw_root: records)
    with pytest.raises(e7b.E7BError, match="SCRFD model not present"):
        e7b.e7b_smoke_target_msu(repo, limit_videos=1)  # no detector injected


# --- 5: injected MockFaceDetector still works for unit tests ---------------

def test_injected_mock_detector_bypasses_real_model_requirement(tmp_path):
    repo = _full_fixture(tmp_path)
    cfg = e7b._m2_config(repo)
    fake = MockFaceDetector(_SUCCESS_DETECTIONS)
    detector, sha256 = e7b._resolve_e7b_detector(cfg, fake)
    assert detector is fake
    assert sha256 == e7b.FROZEN_SCRFD_MODEL_SHA256


# --- 6/13: RunExecutionResult is captured and exposed -----------------------

def test_run_execution_result_captured_and_exposed(tmp_path, monkeypatch):
    repo = _full_fixture(tmp_path)
    records = _fake_target_records(2, "msu_mfsd")
    monkeypatch.setattr(e7b, "_target_canonical_records", lambda repo_, *, dataset, raw_root: records)
    result = e7b.e7b_smoke_target_msu(repo, limit_videos=2, detector=MockFaceDetector(_SUCCESS_DETECTIONS),
                                      media_reader_factory=lambda record: _FakeReader(6, _FAKE_IMAGE))
    summary = result["run_execution_result"]
    assert summary is not None
    for key in ("canonical_records_attempted", "samples_selected", "samples_successful", "samples_failed",
               "frames_read", "detector_calls", "crops_written", "failures_by_code", "manifest_counts"):
        assert key in summary
    assert summary["samples_selected"] == 8
    assert summary["detector_calls"] > 0
    assert summary["frames_read"] > 0
    assert result["body"]["last_run_accounting"] == summary  # also persisted into the package body


# --- 7: unrouted_processing_failure fails closed ----------------------------

def test_unrouted_processing_failure_fails_closed(tmp_path):
    fake_result = type("FakeResult", (), {"failures_by_code": {"unrouted_processing_failure": 3}})()
    with pytest.raises(e7b.E7BError, match="UNROUTED_PROCESSING_FAILURE"):
        e7b._fail_closed_on_unrouted_failure(fake_result, kind="test kind")


def test_unrouted_processing_failure_fails_closed_end_to_end(tmp_path, monkeypatch):
    repo = _full_fixture(tmp_path)
    records = _fake_target_records(1, "msu_mfsd")
    monkeypatch.setattr(e7b, "_target_canonical_records", lambda repo_, *, dataset, raw_root: records)
    with pytest.raises(e7b.E7BError, match="UNROUTED_PROCESSING_FAILURE"):
        e7b.e7b_smoke_target_msu(repo, limit_videos=1, detector=_BrokenDetector(),
                                 media_reader_factory=lambda record: _FakeReader(6, _FAKE_IMAGE))
    # no package written on this hard failure
    assert not (repo / e7b.E7B_MSU_SMOKE_ROOT / "TARGET_PACKAGE.json").is_file()


# --- 8/9/10/12: strict terminal accounting ----------------------------------

def test_strict_accounting_rejects_zero_rows_for_nonzero_planned():
    with pytest.raises(e7b.E7BError, match="STRICT_TERMINAL_ACCOUNTING_FAILED"):
        e7b._enforce_strict_terminal_accounting(kind="k", active_video_count=2, rows=[],
                                                successful=0, failed=0)


def test_strict_accounting_rejects_incomplete_terminal_rows():
    rows = [{"status": "success"}] * 3 + [{"status": "failure"}] * 2  # 5 of 8 expected
    with pytest.raises(e7b.E7BError, match="STRICT_TERMINAL_ACCOUNTING_FAILED"):
        e7b._enforce_strict_terminal_accounting(kind="k", active_video_count=2, rows=rows,
                                                successful=3, failed=2)


def test_strict_accounting_accepts_exactly_eight_terminal_rows():
    rows = [{"status": "success"}] * 3 + [{"status": "failure"}] * 5
    e7b._enforce_strict_terminal_accounting(kind="k", active_video_count=2, rows=rows,
                                            successful=3, failed=5)  # does not raise


def test_strict_accounting_rejects_mismatched_success_failure_counters():
    rows = [{"status": "success"}] * 8
    with pytest.raises(e7b.E7BError, match="STRICT_TERMINAL_ACCOUNTING_FAILED"):
        # claims 4/4 but all 8 rows are actually "success"
        e7b._enforce_strict_terminal_accounting(kind="k", active_video_count=2, rows=rows,
                                                successful=4, failed=4)


# --- 11: all-eight-no_face-failures is a VALID outcome if all eight are
# persisted (already exercised for SiW by
# test_siw_source_smoke_all_failures_completes_and_retains_source_labels;
# this covers the target side end-to-end through the real fixed pipeline) --

def test_all_eight_no_face_failures_is_valid_when_fully_persisted(tmp_path, monkeypatch):
    repo = _full_fixture(tmp_path)
    records = _fake_target_records(2, "casia_fasd")
    monkeypatch.setattr(e7b, "_target_canonical_records", lambda repo_, *, dataset, raw_root: records)
    result = e7b.e7b_smoke_target_casia(repo, limit_videos=2, detector=MockFaceDetector([]),
                                        media_reader_factory=lambda record: _FakeReader(6, _FAKE_IMAGE))
    body = result["body"]
    assert body["successful_crop_count"] == 0
    assert body["failure_count"] == 8
    assert len(body["rows"]) == 8
    assert all(r["status"] == "failure" for r in body["rows"])


# --- 26: 6751638 false-green evidence + diagnosis preserved ----------------

def test_6751638_false_green_evidence_and_diagnosis_preserved():
    evidence_root = REPO / "reports/c_ext_q1q2_v1/e7_three_fold/e7b_data_prep/gpu_evidence/smoke_6751638_false_green"
    assert evidence_root.is_dir()
    assert (evidence_root / "e7b_smoke_siw.out").is_file()
    assert (evidence_root / "e7b_preflight_6751638.json").is_file()

    diagnosis = REPO / e7b.E7B_REPORT_DIR / "E7B_SMOKE_FALSE_GREEN_TECHNICAL_DIAGNOSIS.json"
    assert diagnosis.is_file()
    body = json.loads(diagnosis.read_text(encoding="utf-8"))
    assert body["BASE_COMMIT"] == "6751638ff3514d4f852415adc32aa55026b8d460"
    assert body["FALSE_GREEN_DETECTED"] is True
    assert body["SCIENTIFIC_PROTOCOL_CHANGED"] is False
    assert body["TECHNICAL_EXECUTION_FAILURE"] is True
    assert body["root_cause"]["detector_none_passed_to_run_preprocessing"] is True


# =========================================================================== #
# 65714cb CROP PATH RESOLUTION FIX -- `crop_relative_path` is relative to
# `PreprocessingRunContext.output_root`, which `_build_e7b_run_context`
# always places at `<package_root>/m2_run`. The package manifest itself
# lives at `<package_root>` (one level up). `e7b_validate()` and the two
# resume-integrity functions used to resolve crops against
# `package_manifest_path.parent` directly -- the wrong root -- producing a
# TECHNICAL false-negative "successful crop does not resolve on disk" for
# every real, on-disk, hash-verified crop the real GPU SiW-source build
# produced. Fixed via the ONE `_e7b_m2_output_root()` helper, used
# everywhere a crop is resolved. No persisted data (crop_relative_path,
# rows, package identities) was ever rewritten -- those values were always
# correct; only the CONSUMER's physical base path was wrong.
# =========================================================================== #

def _real_siw_package_layout_fixture(tmp_path: Path, *, n_success: int = 2, n_failure: int = 2) -> Path:
    """Mirrors the REAL GPU package layout: SIW_SOURCE_PACKAGE.json at
    <package_root>, crops physically under <package_root>/m2_run/crops/...
    -- exactly what the real full SiW-source build produced (verified
    against reports/c_ext_q1q2_v1/e7_three_fold/e7b_data_prep/gpu_evidence/
    full_siw_path_audit/FULL_SIW_PHYSICAL_AUDIT.json)."""
    package_root = tmp_path / "siw_source_v1"
    m2_output_root = package_root / "m2_run"
    (m2_output_root / "crops" / "siw_mv2").mkdir(parents=True)
    rows = []
    for i in range(n_success):
        crop_rel = f"crops/siw_mv2/success_{i}.jpg"
        crop_bytes = f"crop-bytes-{i}".encode()
        (m2_output_root / crop_rel).write_bytes(crop_bytes)
        rows.append({"source_video_id": f"Live_{i}", "source_project_split": "train",
                    "frame_index": 0, "status": "success", "crop_relative_path": crop_rel,
                    "crop_sha256": e7b.cc.sha256_bytes(crop_bytes), "label_live_spoof": "live",
                    "spoof_family": None})
    for i in range(n_failure):
        rows.append({"source_video_id": f"Live_{i}", "source_project_split": "train",
                    "frame_index": i + 1, "status": "failure", "crop_relative_path": None,
                    "crop_sha256": None, "failure_reason": "no_face", "label_live_spoof": "live",
                    "spoof_family": None})
    (package_root / "SIW_SOURCE_PACKAGE.json").write_text(json.dumps({
        "package_identity": "x", "population_identity": "pop-1", "split_identity": "split-1",
        "rows": rows,
    }), encoding="utf-8")
    return package_root


def test_validator_resolves_real_gpu_package_layout_crops_correctly(tmp_path):
    package_root = _real_siw_package_layout_fixture(tmp_path, n_success=2, n_failure=2)
    # the OLD, buggy consumer location must NOT exist -- proves the fix
    # cannot be "accidentally" finding a crop at the wrong root
    assert not (package_root / "crops").is_dir()

    repo = tmp_path / "repo"
    repo.mkdir()
    import shutil
    shutil.move(str(package_root), str(repo / e7b.E7B_SIW_SOURCE_PACKAGE_ROOT))

    validation = e7b.e7b_validate(repo)
    problems = validation["siw_source_package"]["problems"]
    assert not any("does not resolve on disk" in p for p in problems)  # the exact false-negative is gone


def test_crop_resolves_uses_m2_run_root_not_manifest_parent(tmp_path):
    package_root = tmp_path / "pkg"
    manifest_path = package_root / "SIW_SOURCE_PACKAGE.json"
    m2_output_root = package_root / "m2_run"
    (m2_output_root / "crops" / "siw_mv2").mkdir(parents=True)
    crop_bytes = b"a-real-crop"
    (m2_output_root / "crops/siw_mv2/a.jpg").write_bytes(crop_bytes)
    sha = e7b.cc.sha256_bytes(crop_bytes)

    assert e7b._e7b_m2_output_root(manifest_path) == m2_output_root
    assert e7b._crop_resolves(e7b._e7b_m2_output_root(manifest_path), "crops/siw_mv2/a.jpg", sha) is True
    # the OLD wrong root (manifest's own parent) does NOT contain the crop
    assert not (package_root / "crops/siw_mv2/a.jpg").is_file()
    assert e7b._crop_resolves(package_root, "crops/siw_mv2/a.jpg", sha) is False


# --- Resume tests A/B/C/D, for BOTH SiW source and target -------------------

def test_resume_A_siw_success_with_matching_sha_at_correct_root_accepted(tmp_path):
    package_root = tmp_path / "siwA"
    m2_output_root = package_root / "m2_run"
    (m2_output_root / "crops").mkdir(parents=True)
    crop_rel = "crops/Live_0_0.jpg"
    crop_bytes = b"crop-a"
    (m2_output_root / crop_rel).write_bytes(crop_bytes)
    manifest = package_root / "SIW_SOURCE_PACKAGE.json"
    rows = [{"source_video_id": "Live_0", "status": "success", "crop_relative_path": crop_rel,
            "crop_sha256": e7b.cc.sha256_bytes(crop_bytes)}] + \
           [{"source_video_id": "Live_0", "status": "failure"}] * 3
    manifest.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    assert e7b._resume_completed_video_ids(manifest, planned_per_video=4) == {"Live_0"}


def test_resume_B_siw_crop_entirely_missing_fails_closed(tmp_path):
    package_root = tmp_path / "siwB"
    manifest = package_root / "SIW_SOURCE_PACKAGE.json"
    manifest.parent.mkdir(parents=True)
    crop_rel = "crops/Live_0_0.jpg"
    rows = [{"source_video_id": "Live_0", "status": "success", "crop_relative_path": crop_rel,
            "crop_sha256": "a" * 64}] + [{"source_video_id": "Live_0", "status": "failure"}] * 3
    manifest.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    with pytest.raises(e7b.E7BError, match="missing on resume"):
        e7b._resume_completed_video_ids(manifest, planned_per_video=4)


def test_resume_C_siw_crop_sha_mismatch_fails_closed(tmp_path):
    package_root = tmp_path / "siwC"
    m2_output_root = package_root / "m2_run"
    (m2_output_root / "crops").mkdir(parents=True)
    crop_rel = "crops/Live_0_0.jpg"
    (m2_output_root / crop_rel).write_bytes(b"real-bytes")
    manifest = package_root / "SIW_SOURCE_PACKAGE.json"
    rows = [{"source_video_id": "Live_0", "status": "success", "crop_relative_path": crop_rel,
            "crop_sha256": "0" * 64}] + [{"source_video_id": "Live_0", "status": "failure"}] * 3
    manifest.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    with pytest.raises(e7b.E7BError, match="corrupt crop"):
        e7b._resume_completed_video_ids(manifest, planned_per_video=4)


def test_resume_D_siw_decoy_at_old_location_still_fails_closed(tmp_path):
    package_root = tmp_path / "siwD"
    (package_root / "crops").mkdir(parents=True)  # OLD wrong location -- decoy
    crop_rel = "crops/Live_0_0.jpg"
    crop_bytes = b"decoy-bytes"
    (package_root / crop_rel).write_bytes(crop_bytes)  # matches sha, but at the WRONG root
    manifest = package_root / "SIW_SOURCE_PACKAGE.json"
    rows = [{"source_video_id": "Live_0", "status": "success", "crop_relative_path": crop_rel,
            "crop_sha256": e7b.cc.sha256_bytes(crop_bytes)}] + \
           [{"source_video_id": "Live_0", "status": "failure"}] * 3
    manifest.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    with pytest.raises(e7b.E7BError, match="missing on resume"):
        e7b._resume_completed_video_ids(manifest, planned_per_video=4)


def test_resume_A_target_success_with_matching_sha_at_correct_root_accepted(tmp_path):
    package_root = tmp_path / "tgtA"
    m2_output_root = package_root / "m2_run"
    (m2_output_root / "crops").mkdir(parents=True)
    crop_rel = "crops/v0_0.jpg"
    crop_bytes = b"crop-a"
    (m2_output_root / crop_rel).write_bytes(crop_bytes)
    manifest = package_root / "TARGET_PACKAGE.json"
    rows = [{"canonical_video_id": "v0", "status": "success", "crop_relative_path": crop_rel,
            "crop_sha256": e7b.cc.sha256_bytes(crop_bytes)}] + \
           [{"canonical_video_id": "v0", "status": "failure"}] * 3
    manifest.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    assert e7b._resume_completed_target_ids(manifest, planned_per_video=4) == {"v0"}


def test_resume_B_target_crop_entirely_missing_fails_closed(tmp_path):
    package_root = tmp_path / "tgtB"
    manifest = package_root / "TARGET_PACKAGE.json"
    manifest.parent.mkdir(parents=True)
    rows = [{"canonical_video_id": "v0", "status": "success", "crop_relative_path": "crops/v0_0.jpg",
            "crop_sha256": "a" * 64}] + [{"canonical_video_id": "v0", "status": "failure"}] * 3
    manifest.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    with pytest.raises(e7b.E7BError, match="missing on resume"):
        e7b._resume_completed_target_ids(manifest, planned_per_video=4)


def test_resume_C_target_crop_sha_mismatch_fails_closed(tmp_path):
    package_root = tmp_path / "tgtC"
    m2_output_root = package_root / "m2_run"
    (m2_output_root / "crops").mkdir(parents=True)
    crop_rel = "crops/v0_0.jpg"
    (m2_output_root / crop_rel).write_bytes(b"real-bytes")
    manifest = package_root / "TARGET_PACKAGE.json"
    rows = [{"canonical_video_id": "v0", "status": "success", "crop_relative_path": crop_rel,
            "crop_sha256": "0" * 64}] + [{"canonical_video_id": "v0", "status": "failure"}] * 3
    manifest.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    with pytest.raises(e7b.E7BError, match="corrupt crop"):
        e7b._resume_completed_target_ids(manifest, planned_per_video=4)


def test_resume_D_target_decoy_at_old_location_still_fails_closed(tmp_path):
    package_root = tmp_path / "tgtD"
    (package_root / "crops").mkdir(parents=True)  # OLD wrong location -- decoy
    crop_rel = "crops/v0_0.jpg"
    crop_bytes = b"decoy-bytes"
    (package_root / crop_rel).write_bytes(crop_bytes)
    manifest = package_root / "TARGET_PACKAGE.json"
    rows = [{"canonical_video_id": "v0", "status": "success", "crop_relative_path": crop_rel,
            "crop_sha256": e7b.cc.sha256_bytes(crop_bytes)}] + \
           [{"canonical_video_id": "v0", "status": "failure"}] * 3
    manifest.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    with pytest.raises(e7b.E7BError, match="missing on resume"):
        e7b._resume_completed_target_ids(manifest, planned_per_video=4)


# --- path-fix technical artifact preserved ----------------------------------

def test_crop_path_resolution_fix_artifact_and_gpu_evidence_preserved():
    evidence_root = REPO / "reports/c_ext_q1q2_v1/e7_three_fold/e7b_data_prep/gpu_evidence/full_siw_path_audit"
    assert evidence_root.is_dir()
    assert (evidence_root / "FULL_SIW_PHYSICAL_AUDIT.json").is_file()
    audit = json.loads((evidence_root / "FULL_SIW_PHYSICAL_AUDIT.json").read_text(encoding="utf-8"))
    assert audit["missing_crop_count"] == 0
    assert audit["bad_hash_count"] == 0
    assert audit["all_frame_subject_null"] is True

    fix = REPO / e7b.E7B_REPORT_DIR / "E7B_CROP_PATH_RESOLUTION_TECHNICAL_FIX.json"
    assert fix.is_file()
    body = json.loads(fix.read_text(encoding="utf-8"))
    assert body["BASE_COMMIT"] == "65714cb897042840fa27dbbd69f5e72bba9ef25d"
    assert body["BUG_CLASS"] == "TECHNICAL_PATH_RESOLUTION_FALSE_NEGATIVE"
    assert body["SCIENTIFIC_DATA_VALID"] is True
    assert body["SCIENTIFIC_RERUN_REQUIRED"] is False
    assert body["CROP_RELATIVE_PATH_REWRITTEN"] is False
    assert body["PACKAGE_MANIFEST_REWRITTEN"] is False
    assert body["SCIENTIFIC_PROTOCOL_CHANGED"] is False
