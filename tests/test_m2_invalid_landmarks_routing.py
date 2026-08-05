import numpy as np
import pytest

import prism_fas.data.m2_runner as m2_runner
from prism_fas.data.manifests.repository import ManifestRepository
from prism_fas.data.media.readers import FrameResult
from prism_fas.data.preprocess_m2 import Detection, DetectorInferenceError, InvalidLandmarksError, validate_landmarks
from prism_fas.data.run_context import M2OutputLayout, PreprocessingRunContext
from prism_fas.data.schemas.records import CanonicalVideoRecord


def _context_record(tmp_path):
    layout = M2OutputLayout.from_root(tmp_path / "full_preprocessing")
    context = PreprocessingRunContext(project_root=tmp_path, work_root=tmp_path, run_profile="full_preprocessing", output_namespace="full_preprocessing", output_root=layout.output_root, crops_root=layout.crops_root, frames_root=layout.frames_root, manifests_root=layout.manifests_root, state_root=layout.state_root, reports_root=layout.reports_root, logs_root=layout.logs_root, run_id="invalid-landmarks", dataset="casia_fasd", dataset_role="source", preprocessing_version="m2-v1", preprocessing_config_hash="a" * 64, detector_model_path=tmp_path / "model.onnx", detector_model_sha256="b" * 64, detector_input_size=320, detector_threshold=.5, all_records=False, record_limit=1, sample_limit=None, resume=False, dry_run=False, partial_full_profile=False, command="test")
    record = CanonicalVideoRecord(dataset="casia_fasd", subject_id="subject-1", video_id="s001v001", source_path=tmp_path / "s001v001f001.png", official_split="train", label="live", adapter_version="casia-v1", source_fingerprint="c" * 64, metadata_provenance="synthetic-test")
    return context, record


def _metadata(context):
    return {"manifest_schema_version": "m2f1a-v1", "preprocessing_version": context.preprocessing_version, "preprocessing_config_hash": context.preprocessing_config_hash, "detector_model_sha256": context.detector_model_sha256, "detector_input_size": str(context.detector_input_size), "detector_threshold": str(context.detector_threshold)}


def _reader(count=1):
    class Reader:
        def frame_count(self): return count
        def read_frame(self, index): return FrameResult(index, index + 5, None, 64, 64, np.zeros((64, 64, 3), dtype=np.uint8), "synthetic-reader")
        def close(self): pass
    return Reader()


@pytest.mark.parametrize("landmarks", [[(1, 1)] * 4, [(1, 1)] * 4 + [(1,)], [(float("nan"), 1)] * 5, [(float("inf"), 1)] * 5, [("not-a-number", 1)] * 5])
def test_landmark_variants_raise_typed_normalization_error(landmarks):
    with pytest.raises(InvalidLandmarksError): validate_landmarks(landmarks)


def test_source_invalid_landmarks_routes_failure_before_counter_or_crop(tmp_path, monkeypatch):
    context, record, calls = *_context_record(tmp_path), []
    class Repository:
        def initialize(self): return self
        def flush(self): pass
        def counts(self): return {"source_frames": 0, "source_crops": 0, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 1}
    class Detector:
        def detect(self, image): return [Detection(bbox=(12, 12, 52, 52), score=.91, landmarks=[(float("nan"), 22)] * 5)]
    repository = Repository()
    monkeypatch.setattr(m2_runner, "route_preprocessing_failure", lambda **kwargs: calls.append(kwargs))
    result = m2_runner.run_preprocessing(context, [record], detector=Detector(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: repository)
    assert len(calls) == 1
    routed = calls[0]
    assert routed["repository"] is repository and routed["context"] is context and routed["canonical_record"] is record
    assert routed["requested_frame_index"] == 0 and routed["actual_frame_index"] == 5
    assert routed["stage"] == "geometry" and routed["error_code"] == "invalid_landmarks" and routed["recoverable"] is True
    assert ":\\" not in routed["error_message"] and "nan" not in routed["error_message"].lower()
    assert (result.samples_selected, result.samples_successful, result.samples_failed, result.frames_read, result.detector_calls, result.crops_written) == (1, 0, 1, 1, 1, 0)
    assert result.failures_by_code == {"invalid_landmarks": 1}
    assert not list(context.crops_root.rglob("*"))


def test_invalid_landmarks_helper_exception_and_missing_repository_are_fatal(tmp_path, monkeypatch):
    context, record = _context_record(tmp_path)
    class Detector:
        def detect(self, image): return [Detection(bbox=(12, 12, 52, 52), score=.91, landmarks=[(float("nan"), 22)] * 5)]
    class Repository:
        def initialize(self): return self
        def flush(self): raise AssertionError("must not flush after routing exception")
        def counts(self): raise AssertionError("must not count after routing exception")
    monkeypatch.setattr(m2_runner, "route_preprocessing_failure", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic landmarks routing error")))
    with pytest.raises(RuntimeError, match="^synthetic landmarks routing error$"):
        m2_runner.run_preprocessing(context, [record], detector=Detector(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: Repository())
    class MissingRepository:
        def initialize(self): return None
    routed = []
    monkeypatch.setattr(m2_runner, "route_preprocessing_failure", lambda **kwargs: routed.append(kwargs))
    with pytest.raises(RuntimeError, match="^failure routing requires an initialized manifest repository$"):
        m2_runner.run_preprocessing(context, [record], detector=Detector(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: MissingRepository())
    assert not routed


def test_real_repository_invalid_landmarks_persists_one_failure_row(tmp_path):
    context, record = _context_record(tmp_path)
    repository = ManifestRepository(context.manifests_root, _metadata(context))
    class Detector:
        def detect(self, image): return [Detection(bbox=(12, 12, 52, 52), score=.91, landmarks=[(float("inf"), 22)] * 5)]
    result = m2_runner.run_preprocessing(context, [record], detector=Detector(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: repository)
    assert result.manifest_counts == {"source_frames": 0, "source_crops": 0, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 1}
    repository.load()
    assert repository.rows["preprocessing_failures"][0]["error_code"] == "invalid_landmarks"


def test_routed_invalid_landmarks_continues_to_valid_source_success(tmp_path):
    context, record = _context_record(tmp_path)
    repository = ManifestRepository(context.manifests_root, _metadata(context))
    class Detector:
        def __init__(self): self.calls = 0
        def detect(self, image):
            self.calls += 1
            landmarks = [(float("nan"), 22)] * 5 if self.calls == 1 else [(20, 22)] * 5
            return [Detection(bbox=(12, 12, 52, 52), score=.91, landmarks=landmarks)]
    result = m2_runner.run_preprocessing(context, [record], detector=Detector(), media_reader_factory=lambda _: _reader(2), repository_factory=lambda *_: repository)
    assert (result.samples_selected, result.samples_successful, result.samples_failed, result.crops_written, result.frames_read, result.detector_calls) == (2, 1, 1, 1, 2, 2)
    assert result.failures_by_code == {"invalid_landmarks": 1}
    assert result.manifest_counts == {"source_frames": 1, "source_crops": 1, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 1}


def test_invalid_bbox_takes_precedence_over_invalid_landmarks(tmp_path, monkeypatch):
    context, record, routed = *_context_record(tmp_path), []
    class Repository:
        def initialize(self): return self
        def flush(self): pass
        def counts(self): return {"source_frames": 0, "source_crops": 0, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 1}
    class Detector:
        def detect(self, image): return [Detection(bbox=(30, 10, 10, 40), score=.91, landmarks=[(float("nan"), 22)] * 5)]
    monkeypatch.setattr(m2_runner, "route_preprocessing_failure", lambda **kwargs: routed.append(kwargs))
    m2_runner.run_preprocessing(context, [record], detector=Detector(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: Repository())
    assert [call["error_code"] for call in routed] == ["invalid_bbox"]


@pytest.mark.parametrize("detector", [type("NoFace", (), {"detect": lambda self, image: []})(), type("Inference", (), {"detect": lambda self, image: (_ for _ in ()).throw(DetectorInferenceError("broken"))})(), type("Generic", (), {"detect": lambda self, image: (_ for _ in ()).throw(ValueError("unrelated"))})()])
def test_non_landmark_conditions_do_not_route_invalid_landmarks(tmp_path, monkeypatch, detector):
    context, record, routed = *_context_record(tmp_path), []
    class Repository:
        def initialize(self): return self
        def flush(self): pass
        def counts(self): return {"source_frames": 0, "source_crops": 0, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 0}
    monkeypatch.setattr(m2_runner, "route_preprocessing_failure", lambda **kwargs: routed.append(kwargs))
    result = m2_runner.run_preprocessing(context, [record], detector=detector, media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: Repository())
    assert not any(call["error_code"] == "invalid_landmarks" for call in routed)
    assert result.failures_by_code.get("invalid_landmarks", 0) == 0
