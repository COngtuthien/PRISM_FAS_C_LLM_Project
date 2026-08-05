import numpy as np
import pytest

from prism_fas.data.m2_validation import validate_full_profile, validate_m2
from prism_fas.data.manifests.repository import ManifestRepository
from prism_fas.data.manifests.routing import route_source_success, route_target_success
from prism_fas.data.output import hash_crop_artifact, write_crop_image
from prism_fas.data.run_context import M2OutputLayout, PreprocessingRunContext
from prism_fas.data.schemas.records import CanonicalVideoRecord, TargetInferenceRecord

DETECTOR_SHA = "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"


class _Paths:
    def __init__(self, tmp_path): self.work_root = tmp_path; self.project_root = tmp_path


class _Config:
    preprocessing_version = "m2-v1"; config_hash = "a" * 64; scrfd_input_size = 320
    detection_threshold = .5; crop_output_size = 224; output_image_format = "jpg"
    def __init__(self, model_path): self.scrfd_model_path = model_path


def _build(tmp_path, *, dataset="casia_fasd", role="source", model_bytes=b"synthetic-model"):
    model = tmp_path / "scrfd.onnx"; model.write_bytes(model_bytes)
    layout = M2OutputLayout.from_root(tmp_path / "full_preprocessing_v2")
    context = PreprocessingRunContext(project_root=tmp_path, work_root=tmp_path, run_profile="full_preprocessing", output_namespace="full_preprocessing", output_root=layout.output_root, crops_root=layout.crops_root, frames_root=layout.frames_root, manifests_root=layout.manifests_root, state_root=layout.state_root, reports_root=layout.reports_root, logs_root=layout.logs_root, run_id="v2", dataset=dataset, dataset_role=role, preprocessing_version="m2-v1", preprocessing_config_hash="a" * 64, detector_model_path=model, detector_model_sha256=_sha(model), detector_input_size=320, detector_threshold=.5, all_records=False, record_limit=1, sample_limit=None, resume=False, dry_run=False, partial_full_profile=False, command="test")
    repository = ManifestRepository(context.manifests_root, {"manifest_schema_version": "m2f1a-v1", "preprocessing_config_hash": context.preprocessing_config_hash, "detector_model_sha256": DETECTOR_SHA}).initialize()
    return context, repository, _Config(model)


def _sha(path):
    from prism_fas.utils.core import sha256_file
    return sha256_file(path)


def _add_sample(context, repository, *, role, sample, dataset, video_id):
    relative = f"crops/{dataset}/{sample}.jpg"
    artifact = write_crop_image(np.full((224, 224, 3), 120, dtype=np.uint8), context.output_root / relative)
    common = dict(repository=repository, context=context, sample_id=sample, requested_frame_index=0, actual_frame_index=1, source_media_type="video_file", timestamp_ms=None, frame_width=64, frame_height=64, decoder_backend="opencv", bbox=[10., 10., 50., 50.], landmarks=[(20., 22.)] * 5, detection_score=.9, detected_face_count=1, crop_box=[0, 0, 60, 60], crop_width=224, crop_height=224, crop_relative_path=relative, crop_sha256=hash_crop_artifact(artifact))
    if role == "source":
        route_source_success(canonical_record=CanonicalVideoRecord(dataset=dataset, subject_id="s1", video_id=video_id, source_path=context.project_root / "x.mp4", official_split="train", label="live", adapter_version="v1", source_fingerprint="c" * 64, metadata_provenance="test"), selected_frame_reference=f"{video_id}#frame=1", **common)
    else:
        route_target_success(canonical_record=TargetInferenceRecord(dataset=dataset, video_id=video_id, source_path=context.project_root / "x.mov", official_split="target_test", adapter_version="v1", source_fingerprint="d" * 64, metadata_provenance="test"), selected_frame_reference=f"target/{sample}#frame=1", **common)
    return context.output_root / relative


def test_valid_full_profile_run_passes_without_legacy_artifacts(tmp_path, monkeypatch):
    context, repository, cfg = _build(tmp_path)
    monkeypatch.setattr("prism_fas.data.m2_validation.sha256_file", lambda p: DETECTOR_SHA if p == cfg.scrfd_model_path else _sha(p))
    _add_sample(context, repository, role="source", sample="src1", dataset="casia_fasd", video_id="train_s1v1")
    target_context = context.model_copy(update={"dataset": "siw_mv2", "dataset_role": "target"})
    _add_sample(target_context, repository, role="target", sample="tgt1", dataset="siw_mv2", video_id="siw_c267caa1e8b4aaf2")
    repository.flush()

    report = validate_full_profile(_Paths(tmp_path), cfg, context.output_root)
    assert report["passed"] is True, report["errors"]
    assert report["profile"] == "full_preprocessing"
    assert report["manifests"] == {"source_frames": 1, "source_crops": 1, "target_frames": 1, "target_crops": 1, "preprocessing_failures": 0}
    assert report["crop_integrity"]["passed"] is True and report["crops_on_disk"] == 2
    assert report["target_isolation"]["passed"] is True
    # the same tree fails small-acceptance validation, which is intentionally unchanged
    assert validate_m2(_Paths(tmp_path), cfg, context.output_root)["passed"] is False


@pytest.mark.parametrize("damage", ["missing_crop", "sha_mismatch", "orphan_crop", "temporary_artifact"])
def test_crop_integrity_failures_are_detected(tmp_path, monkeypatch, damage):
    context, repository, cfg = _build(tmp_path)
    monkeypatch.setattr("prism_fas.data.m2_validation.sha256_file", lambda p: DETECTOR_SHA if p == cfg.scrfd_model_path else _sha(p))
    artifact = _add_sample(context, repository, role="source", sample="src1", dataset="casia_fasd", video_id="train_s1v1")
    repository.flush()
    if damage == "missing_crop": artifact.unlink()
    elif damage == "sha_mismatch": artifact.write_bytes(artifact.read_bytes() + b"tampered")
    elif damage == "orphan_crop": (artifact.parent / "orphan.jpg").write_bytes(b"orphan")
    else: (artifact.parent / "src1.jpg.abcd.tmp").write_bytes(b"partial")

    report = validate_full_profile(_Paths(tmp_path), cfg, context.output_root)
    assert report["passed"] is False
    assert {e["check_id"] for e in report["errors"]} & {"crop.missing", "crop.sha", "crop.orphans", "crop.temporary", "crop.readable"}


def test_private_target_tokens_fail_validation(tmp_path, monkeypatch):
    context, repository, cfg = _build(tmp_path, dataset="siw_mv2", role="target")
    monkeypatch.setattr("prism_fas.data.m2_validation.sha256_file", lambda p: DETECTOR_SHA if p == cfg.scrfd_model_path else _sha(p))
    _add_sample(context, repository, role="target", sample="tgt1", dataset="siw_mv2", video_id="Live_477")
    repository.flush()
    report = validate_full_profile(_Paths(tmp_path), cfg, context.output_root)
    assert report["passed"] is False
    assert "live" in report["target_isolation"]["token_matches"]
    assert any(e["check_id"] == "target.tokens" for e in report["errors"])


def test_detector_and_config_metadata_mismatch_is_detected(tmp_path, monkeypatch):
    context, repository, cfg = _build(tmp_path)
    monkeypatch.setattr("prism_fas.data.m2_validation.sha256_file", lambda p: ("f" * 64) if p == cfg.scrfd_model_path else _sha(p))
    _add_sample(context, repository, role="source", sample="src1", dataset="casia_fasd", video_id="train_s1v1")
    repository.flush()
    report = validate_full_profile(_Paths(tmp_path), cfg, context.output_root)
    assert report["passed"] is False
    assert any(e["check_id"] == "detector.hash" for e in report["errors"])
