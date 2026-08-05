import pytest
from unittest.mock import Mock

import prism_fas.data.manifests.routing as routing
from prism_fas.data.manifests.converters import InvalidMediaTypeError, build_source_crop_record, build_source_frame_record, build_target_crop_record, build_target_frame_record
from prism_fas.data.manifests.repository import ManifestRepository
from prism_fas.data.manifests.routing import SourceRoutingConsistencyError, TargetRoutingConsistencyError, route_source_success, route_target_success
from prism_fas.data.run_context import M2OutputLayout, PreprocessingRunContext
from prism_fas.data.schemas.records import CanonicalVideoRecord

GEOMETRY = dict(bbox=[10., 10., 50., 50.], landmarks=[(20., 22.)] * 5, crop_box=[0, 0, 60, 60], crop_width=224, crop_height=224, detection_score=.9, detected_face_count=1, crop_sha256="a" * 64)


def _context(tmp_path, dataset, role):
    layout = M2OutputLayout.from_root(tmp_path / "full_preprocessing")
    return PreprocessingRunContext(project_root=tmp_path, work_root=tmp_path, run_profile="full_preprocessing", output_namespace="full_preprocessing", output_root=layout.output_root, crops_root=layout.crops_root, frames_root=layout.frames_root, manifests_root=layout.manifests_root, state_root=layout.state_root, reports_root=layout.reports_root, logs_root=layout.logs_root, run_id="media-type", dataset=dataset, dataset_role=role, preprocessing_version="m2-v1", preprocessing_config_hash="c" * 64, detector_model_path=tmp_path / "model.onnx", detector_model_sha256="d" * 64, detector_input_size=320, detector_threshold=.5, all_records=False, record_limit=1, sample_limit=None, resume=False, dry_run=False, partial_full_profile=False, command="test")


def _casia(tmp_path):
    return CanonicalVideoRecord(dataset="casia_fasd", subject_id="s1", video_id="s001v001", source_path=tmp_path / "s001v001f001.png", official_split="train", label="live", adapter_version="casia-v1", source_fingerprint="e" * 64, metadata_provenance="synthetic-test")


def _msu(tmp_path):
    return CanonicalVideoRecord(dataset="msu_mfsd", subject_id="s2", video_id="real_client005", source_path=tmp_path / "real_client005.mp4", official_split="train", label="live", adapter_version="msu-v1", source_fingerprint="e" * 64, metadata_provenance="synthetic-test")


def _siw(tmp_path):
    return CanonicalVideoRecord(dataset="siw_mv2", subject_id="subject_007", video_id="v0007", source_path=tmp_path / "raw" / "subject_007" / "replay_attack.mp4", official_split="target_test", label="spoof", capture_metadata={"attack_type": "replay"}, adapter_version="siw-v1", source_fingerprint="e" * 64, metadata_provenance="synthetic-test")


def _repository(context):
    return ManifestRepository(context.manifests_root, {"manifest_schema_version": "m2f1a-v1"}).initialize()


def _route_source(context, record, repository, media_type):
    return route_source_success(repository=repository, context=context, canonical_record=record, sample_id="sid-source", requested_frame_index=3, actual_frame_index=4, source_media_type=media_type, timestamp_ms=None, frame_width=640, frame_height=480, decoder_backend="opencv", selected_frame_reference=f"{record.video_id}#frame=4", crop_relative_path="crops/x.jpg", **GEOMETRY)


def _route_target(context, record, repository, media_type):
    return route_target_success(repository=repository, context=context, canonical_record=record, sample_id="sid-target", requested_frame_index=3, actual_frame_index=4, source_media_type=media_type, timestamp_ms=None, frame_width=640, frame_height=480, decoder_backend="opencv", selected_frame_reference=f"siw_mv2/{record.video_id}#frame=4", crop_relative_path="crops/y.jpg", **GEOMETRY)


def test_casia_image_sequence_frame_and_crop_agree(tmp_path):
    context, record = _context(tmp_path, "casia_fasd", "source"), _casia(tmp_path)
    repository = _repository(context)
    result = _route_source(context, record, repository, "image_sequence")
    assert result.frame_record.source_media_type == result.crop_record.source_media_type == "image_sequence"
    repository.flush(); repository.load()
    assert repository.rows["source_frames"][0]["source_media_type"] == "image_sequence"
    assert repository.rows["source_crops"][0]["source_media_type"] == "image_sequence"
    assert repository.counts() == {"source_frames": 1, "source_crops": 1, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 0}


def test_msu_video_frame_and_crop_are_video_file_in_real_parquet(tmp_path):
    context, record = _context(tmp_path, "msu_mfsd", "source"), _msu(tmp_path)
    repository = _repository(context)
    result = _route_source(context, record, repository, "video_file")
    assert result.frame_record.source_media_type == result.crop_record.source_media_type == "video_file"
    repository.flush(); repository.load()
    frame, crop = repository.rows["source_frames"][0], repository.rows["source_crops"][0]
    assert frame["source_media_type"] == "video_file"
    assert crop["source_media_type"] == "video_file", "video source crop must not be recorded as an image sequence"
    assert frame["dataset"] == crop["dataset"] == "msu_mfsd" and frame["sample_id"] == crop["sample_id"] == "sid-source"
    assert repository.counts() == {"source_frames": 1, "source_crops": 1, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 0}


def test_siw_target_video_frame_and_crop_are_video_file_in_real_parquet(tmp_path):
    context, record = _context(tmp_path, "siw_mv2", "target"), _siw(tmp_path)
    repository = _repository(context)
    result = _route_target(context, record, repository, "video_file")
    assert result.frame_record.source_media_type == result.crop_record.source_media_type == "video_file"
    repository.flush(); repository.load()
    frame, crop = repository.rows["target_frames"][0], repository.rows["target_crops"][0]
    assert frame["source_media_type"] == crop["source_media_type"] == "video_file"
    assert repository.counts() == {"source_frames": 0, "source_crops": 0, "target_frames": 1, "target_crops": 1, "preprocessing_failures": 0}
    # privacy must survive the media-type fix
    for row in (frame, crop):
        for token in ("spoof", "live", "replay", "attack", "subject_007", "subject"):
            assert token not in str(row).lower()
        for name in ("label", "label_live_spoof", "subject_id", "attack_type", "capture_metadata", "source_path"):
            assert name not in row


@pytest.mark.parametrize("builder,kwargs", [
    (build_source_frame_record, dict(sample_id="s", source_relative_identifier="x", requested_frame_index=0, actual_frame_index=0, frame_width=10, frame_height=10, selected_frame_reference="x#0", decoder_backend="opencv")),
    (build_source_crop_record, dict(sample_id="s", requested=0, actual=0, timestamp=None, width=10, height=10, bbox=[1., 1., 5., 5.], landmarks=[(1., 1.)] * 5, score=.9, count=1, box=[0, 0, 6, 6], padding=.2, cw=6, ch=6, path="c.jpg", sha="a" * 64)),
])
@pytest.mark.parametrize("media_type", ["unknown_media", "single_image", "jpg", "", None, 1])
def test_invalid_media_type_is_rejected_without_silent_default(tmp_path, builder, kwargs, media_type):
    context, record = _context(tmp_path, "casia_fasd", "source"), _casia(tmp_path)
    with pytest.raises(InvalidMediaTypeError):
        builder(context, record, source_media_type=media_type, **kwargs)


def test_missing_media_type_is_rejected_for_every_builder(tmp_path):
    source_context, target_context = _context(tmp_path, "casia_fasd", "source"), _context(tmp_path, "siw_mv2", "target")
    frame_kwargs = dict(sample_id="s", source_relative_identifier="x", requested_frame_index=0, actual_frame_index=0, frame_width=10, frame_height=10, selected_frame_reference="x#0", decoder_backend="opencv")
    crop_kwargs = dict(sample_id="s", requested=0, actual=0, timestamp=None, width=10, height=10, bbox=[1., 1., 5., 5.], landmarks=[(1., 1.)] * 5, score=.9, count=1, box=[0, 0, 6, 6], padding=.2, cw=6, ch=6, path="c.jpg", sha="a" * 64)
    for builder, context, record, kwargs in ((build_source_frame_record, source_context, _casia(tmp_path), frame_kwargs), (build_source_crop_record, source_context, _casia(tmp_path), crop_kwargs), (build_target_frame_record, target_context, _siw(tmp_path), frame_kwargs), (build_target_crop_record, target_context, _siw(tmp_path), crop_kwargs)):
        with pytest.raises(InvalidMediaTypeError):
            builder(context, record, **kwargs)


def test_invalid_media_type_does_not_mutate_repository(tmp_path):
    context, record = _context(tmp_path, "msu_mfsd", "source"), _msu(tmp_path)
    repository = _repository(context)
    with pytest.raises(InvalidMediaTypeError):
        _route_source(context, record, repository, "unknown_media")
    assert repository.counts() == {"source_frames": 0, "source_crops": 0, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 0}


def test_source_frame_crop_media_type_mismatch_is_rejected_before_upsert(tmp_path, monkeypatch):
    context, record = _context(tmp_path, "msu_mfsd", "source"), _msu(tmp_path)
    crop = build_source_crop_record(context, record, sample_id="sid-source", source_media_type="image_sequence", requested=3, actual=4, timestamp=None, width=640, height=480, bbox=[10., 10., 50., 50.], landmarks=[(20., 22.)] * 5, score=.9, count=1, box=[0, 0, 60, 60], padding=.25, cw=224, ch=224, path="crops/x.jpg", sha="a" * 64)
    monkeypatch.setattr(routing, "build_source_crop_record", lambda *a, **k: crop)
    repository = Mock()
    with pytest.raises(SourceRoutingConsistencyError, match="source_media_type"):
        _route_source(context, record, repository, "video_file")
    repository.upsert_source_success.assert_not_called()


def test_target_frame_crop_media_type_mismatch_is_rejected_before_upsert(tmp_path, monkeypatch):
    context, record = _context(tmp_path, "siw_mv2", "target"), _siw(tmp_path)
    crop = build_target_crop_record(context, record, sample_id="sid-target", source_media_type="image_sequence", requested=3, actual=4, timestamp=None, width=640, height=480, bbox=[10., 10., 50., 50.], landmarks=[(20., 22.)] * 5, score=.9, count=1, box=[0, 0, 60, 60], padding=.25, cw=224, ch=224, path="crops/y.jpg", sha="a" * 64)
    monkeypatch.setattr(routing, "build_target_crop_record", lambda *a, **k: crop)
    repository = Mock()
    with pytest.raises(TargetRoutingConsistencyError, match="source_media_type"):
        _route_target(context, record, repository, "video_file")
    repository.upsert_target_success.assert_not_called()


def test_crop_media_type_is_independent_of_crop_output_format(tmp_path):
    # The crop artifact is always a JPEG; that must never drive source_media_type.
    context, record = _context(tmp_path, "msu_mfsd", "source"), _msu(tmp_path)
    crop = build_source_crop_record(context, record, sample_id="s", source_media_type="video_file", requested=0, actual=0, timestamp=None, width=10, height=10, bbox=[1., 1., 5., 5.], landmarks=[(1., 1.)] * 5, score=.9, count=1, box=[0, 0, 6, 6], padding=.2, cw=6, ch=6, path="crops/a.jpg", sha="a" * 64)
    assert crop.crop_relative_path.endswith(".jpg") and crop.source_media_type == "video_file"


def test_runner_passes_real_media_type_for_image_sequence_and_video(tmp_path, monkeypatch):
    import numpy as np
    import prism_fas.data.m2_runner as m2_runner
    from prism_fas.data.media.readers import FrameResult
    from prism_fas.data.preprocess_m2 import Detection

    class Reader:
        def frame_count(self): return 1
        def read_frame(self, index): return FrameResult(index, index, None, 64, 64, np.zeros((64, 64, 3), dtype=np.uint8), "synthetic-reader")
        def close(self): pass
    detector = type("Detector", (), {"detect": lambda self, image: [Detection(bbox=(12, 12, 52, 52), score=.91, landmarks=[(20, 22)] * 5)]})()

    for dataset, record, expected in (("casia_fasd", _casia(tmp_path / "a"), "image_sequence"), ("msu_mfsd", _msu(tmp_path / "b"), "video_file")):
        root = tmp_path / dataset
        context = _context(root, dataset, "source")
        repository = ManifestRepository(context.manifests_root, {"manifest_schema_version": "m2f1a-v1"})
        m2_runner.run_preprocessing(context, [record], detector=detector, media_reader_factory=lambda _: Reader(), repository_factory=lambda *_: repository)
        repository.load()
        assert repository.rows["source_frames"][0]["source_media_type"] == expected
        assert repository.rows["source_crops"][0]["source_media_type"] == expected
