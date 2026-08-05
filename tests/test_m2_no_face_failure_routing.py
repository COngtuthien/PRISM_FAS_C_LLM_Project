import numpy as np
import pytest

import prism_fas.data.m2_runner as m2_runner
from prism_fas.data.manifests.repository import ManifestRepository
from prism_fas.data.media.readers import FrameResult
from prism_fas.data.preprocess_m2 import Detection
from prism_fas.data.run_context import M2OutputLayout, PreprocessingRunContext
from prism_fas.data.schemas.records import CanonicalVideoRecord


COUNT_KEYS = ("source_frames", "source_crops", "target_frames", "target_crops", "preprocessing_failures")


def _context_record(tmp_path):
    layout = M2OutputLayout.from_root(tmp_path / "full_preprocessing")
    context = PreprocessingRunContext(project_root=tmp_path, work_root=tmp_path, run_profile="full_preprocessing", output_namespace="full_preprocessing", output_root=layout.output_root, crops_root=layout.crops_root, frames_root=layout.frames_root, manifests_root=layout.manifests_root, state_root=layout.state_root, reports_root=layout.reports_root, logs_root=layout.logs_root, run_id="no-face", dataset="casia_fasd", dataset_role="source", preprocessing_version="m2-v1", preprocessing_config_hash="a" * 64, detector_model_path=tmp_path / "model.onnx", detector_model_sha256="b" * 64, detector_input_size=320, detector_threshold=.5, all_records=False, record_limit=1, sample_limit=None, resume=False, dry_run=False, partial_full_profile=False, command="test")
    record = CanonicalVideoRecord(dataset="casia_fasd", subject_id="subject-1", video_id="s001v001", source_path=tmp_path / "s001v001f001.png", official_split="train", label="live", adapter_version="casia-v1", source_fingerprint="c" * 64, metadata_provenance="synthetic-test")
    return context, record


def _reader(frame_count=1):
    class Reader:
        def frame_count(self): return frame_count
        def read_frame(self, index): return FrameResult(index, index + 17, None, 64, 64, np.zeros((64, 64, 3), dtype=np.uint8), "synthetic-reader")
        def close(self): pass
    return Reader()


def _metadata(context):
    return {"manifest_schema_version": "m2f1a-v1", "preprocessing_version": context.preprocessing_version, "preprocessing_config_hash": context.preprocessing_config_hash, "detector_model_sha256": context.detector_model_sha256, "detector_input_size": str(context.detector_input_size), "detector_threshold": str(context.detector_threshold)}


def test_source_no_face_routes_failure_before_failure_counters(tmp_path, monkeypatch):
    context, record = _context_record(tmp_path)
    calls, events = [], []

    class Repository:
        def initialize(self): return self
        def flush(self): events.append("flush")
        def counts(self): return {"source_frames": 0, "source_crops": 0, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 1}

    class Detector:
        def detect(self, image): return []

    repository = Repository()
    def route_probe(**kwargs): events.append("route_failure"); calls.append(kwargs)

    monkeypatch.setattr(m2_runner, "route_preprocessing_failure", route_probe)
    result = m2_runner.run_preprocessing(context, [record], detector=Detector(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: repository)
    assert len(calls) == 1
    routed = calls[0]
    assert routed["repository"] is repository and routed["context"] is context and routed["canonical_record"] is record
    assert routed["sample_id"] and routed["requested_frame_index"] == 0 and routed["actual_frame_index"] == 17
    assert routed["stage"] == "detector" and routed["error_code"] == "no_face" and routed["recoverable"] is True
    assert ":\\" not in routed["error_message"] and "/" not in routed["error_message"]
    assert events[0] == "route_failure"
    assert result.samples_selected == result.samples_failed == result.frames_read == result.detector_calls == 1
    assert result.samples_successful == result.crops_written == 0 and result.failures_by_code == {"no_face": 1}


def test_no_face_failure_helper_exception_propagates_without_false_counter(tmp_path, monkeypatch):
    context, record = _context_record(tmp_path)

    class Repository:
        def initialize(self): return self
        def flush(self): raise AssertionError("failure helper exception must stop before flush")
        def counts(self): raise AssertionError("failure helper exception must stop before counts")

    monkeypatch.setattr(m2_runner, "route_preprocessing_failure", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic no-face routing error")))
    with pytest.raises(RuntimeError, match="^synthetic no-face routing error$"):
        m2_runner.run_preprocessing(context, [record], detector=type("Detector", (), {"detect": lambda self, image: []})(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: Repository())


def test_no_face_missing_repository_raises_before_helper(tmp_path, monkeypatch):
    context, record = _context_record(tmp_path)
    routed = []

    class UninitializedRepository:
        def initialize(self): return None

    monkeypatch.setattr(m2_runner, "route_preprocessing_failure", lambda **kwargs: routed.append(kwargs))
    with pytest.raises(RuntimeError, match="^failure routing requires an initialized manifest repository$"):
        m2_runner.run_preprocessing(context, [record], detector=type("Detector", (), {"detect": lambda self, image: []})(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: UninitializedRepository())
    assert not routed


def test_real_repository_no_face_creates_one_failure_row_and_no_success_rows(tmp_path):
    context, record = _context_record(tmp_path)
    repository = ManifestRepository(context.manifests_root, _metadata(context))
    result = m2_runner.run_preprocessing(context, [record], detector=type("Detector", (), {"detect": lambda self, image: []})(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: repository)
    assert result.manifest_counts == {"source_frames": 0, "source_crops": 0, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 1}
    repository.load()
    assert len(repository.rows["preprocessing_failures"]) == 1


def test_successfully_routed_no_face_continues_to_later_source_success(tmp_path):
    context, record = _context_record(tmp_path)
    repository = ManifestRepository(context.manifests_root, _metadata(context))

    class Detector:
        def __init__(self): self.calls = 0
        def detect(self, image):
            self.calls += 1
            return [] if self.calls == 1 else [Detection(bbox=(12, 12, 52, 52), score=.91, landmarks=[(20, 22)] * 5)]

    result = m2_runner.run_preprocessing(context, [record], detector=Detector(), media_reader_factory=lambda _: _reader(2), repository_factory=lambda *_: repository)
    assert (result.samples_selected, result.samples_successful, result.samples_failed, result.crops_written) == (2, 1, 1, 1)
    assert result.failures_by_code == {"no_face": 1}
    assert result.manifest_counts == {"source_frames": 1, "source_crops": 1, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 1}
