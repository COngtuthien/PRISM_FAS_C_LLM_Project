import hashlib
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import prism_fas.data.m2_runner as m2_runner
from prism_fas.data.media.readers import FrameResult
from prism_fas.data.preprocess_m2 import Detection, MockFaceDetector, sample_id
from prism_fas.data.run_context import M2OutputLayout, PreprocessingRunContext
from prism_fas.data.schemas.records import CanonicalVideoRecord, TargetInferenceRecord


def test_run_preprocessing_routes_one_synthetic_source_before_success_counters(tmp_path, monkeypatch):
    layout = M2OutputLayout.from_root(tmp_path / "full_preprocessing")
    context = PreprocessingRunContext(
        project_root=tmp_path,
        work_root=tmp_path,
        run_profile="full_preprocessing",
        output_namespace="full_preprocessing",
        output_root=layout.output_root,
        crops_root=layout.crops_root,
        frames_root=layout.frames_root,
        manifests_root=layout.manifests_root,
        state_root=layout.state_root,
        reports_root=layout.reports_root,
        logs_root=layout.logs_root,
        run_id="synthetic-source",
        dataset="casia_fasd",
        dataset_role="source",
        preprocessing_version="m2-v1",
        preprocessing_config_hash="c" * 64,
        detector_model_path=tmp_path / "synthetic.onnx",
        detector_model_sha256="d" * 64,
        detector_input_size=320,
        detector_threshold=0.5,
        all_records=False,
        record_limit=1,
        sample_limit=1,
        resume=False,
        dry_run=False,
        partial_full_profile=False,
        command="test",
    )
    source_record = CanonicalVideoRecord(
        dataset="casia_fasd",
        subject_id="subject-1",
        video_id="s001v001",
        source_path=tmp_path / "s001v001f001.png",
        official_split="train",
        label="live",
        adapter_version="casia-v1",
        source_fingerprint="e" * 64,
        metadata_provenance="synthetic-test",
    )
    frame = np.full((64, 64, 3), 127, dtype=np.uint8)
    detection = Detection(
        bbox=(12.0, 12.0, 52.0, 52.0),
        score=0.91,
        landmarks=[(20.0, 22.0), (44.0, 22.0), (32.0, 32.0), (24.0, 43.0), (40.0, 43.0)],
    )
    events, routing_calls = [], []

    class Reader:
        def frame_count(self):
            return 1

        def read_frame(self, requested_index):
            assert requested_index == 0
            return FrameResult(0, 17, 680.0, 64, 64, frame, "synthetic-reader")

        def close(self):
            events.append("close")

    class Detector:
        def detect(self, image):
            assert image is frame
            return [detection]

    class Repository:
        def initialize(self):
            return self

        def flush(self):
            events.append("flush")

        def counts(self):
            return {"source_frames": 1, "source_crops": 1, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 0}

    repository = Repository()

    def route_probe(**kwargs):
        crop_path = context.output_root / kwargs["crop_relative_path"]
        assert crop_path.is_file()
        assert hashlib.sha256(crop_path.read_bytes()).hexdigest() == kwargs["crop_sha256"]
        events.append("route")
        routing_calls.append(kwargs)
        return SimpleNamespace(sample_id=kwargs["sample_id"])

    monkeypatch.setattr(m2_runner, "route_source_success", route_probe)
    result = m2_runner.run_preprocessing(
        context=context,
        canonical_records=[source_record],
        detector=Detector(),
        media_reader_factory=lambda record: Reader(),
        repository_factory=lambda manifests_root, metadata: repository,
    )

    assert len(routing_calls) == 1
    routed = routing_calls[0]
    assert routed["repository"] is repository
    assert routed["context"] is context
    assert routed["canonical_record"] is source_record
    assert routed["sample_id"] == sample_id("casia_fasd", "s001v001", 0, "casia-v1", "uniform-v1", "m2-v1")
    assert routed["requested_frame_index"] == 0
    assert routed["actual_frame_index"] == 17
    assert len(routed["bbox"]) == 4 and all(np.isfinite(value) for value in routed["bbox"])
    assert len(routed["landmarks"]) == 5
    assert routed["detection_score"] == 0.91
    assert routed["detected_face_count"] == 1
    relative_crop_path = Path(routed["crop_relative_path"])
    assert not relative_crop_path.is_absolute() and ".." not in relative_crop_path.parts
    crop_path = context.output_root / relative_crop_path
    assert crop_path.is_file()
    assert re.fullmatch(r"[0-9a-f]{64}", routed["crop_sha256"])
    assert hashlib.sha256(crop_path.read_bytes()).hexdigest() == routed["crop_sha256"]
    assert events.index("route") < events.index("flush")
    assert result.samples_selected == result.samples_successful == result.crops_written == 1
    assert result.samples_failed == 0
    assert result.frames_read == result.detector_calls == 1


def test_run_preprocessing_propagates_source_routing_exception_without_success_counts(tmp_path, monkeypatch):
    layout = M2OutputLayout.from_root(tmp_path / "full_preprocessing")
    context = PreprocessingRunContext(project_root=tmp_path, work_root=tmp_path, run_profile="full_preprocessing", output_namespace="full_preprocessing", output_root=layout.output_root, crops_root=layout.crops_root, frames_root=layout.frames_root, manifests_root=layout.manifests_root, state_root=layout.state_root, reports_root=layout.reports_root, logs_root=layout.logs_root, run_id="routing-error", dataset="casia_fasd", dataset_role="source", preprocessing_version="m2-v1", preprocessing_config_hash="c" * 64, detector_model_path=tmp_path / "synthetic.onnx", detector_model_sha256="d" * 64, detector_input_size=320, detector_threshold=0.5, all_records=False, record_limit=1, sample_limit=1, resume=False, dry_run=False, partial_full_profile=False, command="test")
    record = CanonicalVideoRecord(dataset="casia_fasd", subject_id="subject-1", video_id="s001v001", source_path=tmp_path / "s001v001f001.png", official_split="train", label="live", adapter_version="casia-v1", source_fingerprint="e" * 64, metadata_provenance="synthetic-test")
    frame = np.full((64, 64, 3), 127, dtype=np.uint8)
    calls = []

    class Reader:
        def frame_count(self): return 1
        def read_frame(self, index): return FrameResult(index, 17, None, 64, 64, frame, "synthetic-reader")
        def close(self): pass

    class Repository:
        def initialize(self): return self
        def flush(self): raise AssertionError("routing failure must not flush a completed result")
        def counts(self): raise AssertionError("routing failure must not return a result")

    def routing_failure(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("synthetic source routing failure")

    monkeypatch.setattr(m2_runner, "route_source_success", routing_failure)
    with pytest.raises(RuntimeError, match="^synthetic source routing failure$"):
        m2_runner.run_preprocessing(context, [record], detector=MockFaceDetector([Detection(bbox=(12, 12, 52, 52), score=.91, landmarks=[(20, 22)] * 5)]), media_reader_factory=lambda _: Reader(), repository_factory=lambda *_: Repository())
    assert len(calls) == 1
    assert list(context.crops_root.rglob("*.jpg"))


def test_run_preprocessing_rejects_missing_source_repository_before_routing(tmp_path, monkeypatch):
    layout = M2OutputLayout.from_root(tmp_path / "full_preprocessing")
    context = PreprocessingRunContext(project_root=tmp_path, work_root=tmp_path, run_profile="full_preprocessing", output_namespace="full_preprocessing", output_root=layout.output_root, crops_root=layout.crops_root, frames_root=layout.frames_root, manifests_root=layout.manifests_root, state_root=layout.state_root, reports_root=layout.reports_root, logs_root=layout.logs_root, run_id="missing-repository", dataset="casia_fasd", dataset_role="source", preprocessing_version="m2-v1", preprocessing_config_hash="c" * 64, detector_model_path=tmp_path / "synthetic.onnx", detector_model_sha256="d" * 64, detector_input_size=320, detector_threshold=.5, all_records=False, record_limit=1, sample_limit=1, resume=False, dry_run=False, partial_full_profile=False, command="test")
    record = CanonicalVideoRecord(dataset="casia_fasd", subject_id="subject-1", video_id="s001v001", source_path=tmp_path / "s001v001f001.png", official_split="train", label="live", adapter_version="casia-v1", source_fingerprint="e" * 64, metadata_provenance="synthetic-test")
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    routed = []

    class Reader:
        def frame_count(self): return 1
        def read_frame(self, index): return FrameResult(index, index, None, 64, 64, frame, "synthetic-reader")
        def close(self): pass

    class UninitializedRepository:
        def initialize(self): return None

    monkeypatch.setattr(m2_runner, "route_source_success", lambda **kwargs: routed.append(kwargs))
    with pytest.raises(RuntimeError, match="^source routing requires an initialized manifest repository$"):
        m2_runner.run_preprocessing(context, [record], detector=MockFaceDetector([Detection(bbox=(12, 12, 52, 52), score=.91, landmarks=[(20, 22)] * 5)]), media_reader_factory=lambda _: Reader(), repository_factory=lambda *_: UninitializedRepository())
    assert not routed


def test_run_preprocessing_target_role_never_calls_source_routing(tmp_path, monkeypatch):
    layout = M2OutputLayout.from_root(tmp_path / "full_preprocessing")
    context = PreprocessingRunContext(project_root=tmp_path, work_root=tmp_path, run_profile="full_preprocessing", output_namespace="full_preprocessing", output_root=layout.output_root, crops_root=layout.crops_root, frames_root=layout.frames_root, manifests_root=layout.manifests_root, state_root=layout.state_root, reports_root=layout.reports_root, logs_root=layout.logs_root, run_id="target-guard", dataset="siw_mv2", dataset_role="target", preprocessing_version="m2-v1", preprocessing_config_hash="c" * 64, detector_model_path=tmp_path / "synthetic.onnx", detector_model_sha256="d" * 64, detector_input_size=320, detector_threshold=.5, all_records=False, record_limit=1, sample_limit=1, resume=False, dry_run=False, partial_full_profile=False, command="test")
    record = TargetInferenceRecord(dataset="siw_mv2", video_id="target-1", source_path=tmp_path / "target.mp4", official_split="test", adapter_version="siw-v1", source_fingerprint="f" * 64, metadata_provenance="synthetic-test")
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    class Reader:
        def frame_count(self): return 1
        def read_frame(self, index): return FrameResult(index, index, None, 64, 64, frame, "synthetic-reader")
        def close(self): pass

    class Repository:
        def initialize(self): return self
        def flush(self): pass
        def upsert_target_success(self, frame_row, crop_row): pass
        def counts(self): return {"source_frames": 0, "source_crops": 0, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 0}

    def source_routing_must_not_run(**kwargs):
        raise AssertionError("target record entered source routing")

    monkeypatch.setattr(m2_runner, "route_source_success", source_routing_must_not_run)
    result = m2_runner.run_preprocessing(context, [record], detector=MockFaceDetector([Detection(bbox=(12, 12, 52, 52), score=.91, landmarks=[(20, 22)] * 5)]), media_reader_factory=lambda _: Reader(), repository_factory=lambda *_: Repository())
    assert result.samples_successful == result.crops_written == 1
    assert result.samples_failed == 0
