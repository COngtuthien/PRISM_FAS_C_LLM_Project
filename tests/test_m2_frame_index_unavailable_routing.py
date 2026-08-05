import numpy as np
import pytest

import prism_fas.data.m2_runner as m2_runner
from prism_fas.data.manifests.repository import ManifestRepository
from prism_fas.data.media.readers import FrameResult
from prism_fas.data.preprocess_m2 import Detection
from prism_fas.data.run_context import M2OutputLayout, PreprocessingRunContext
from prism_fas.data.schemas.records import CanonicalVideoRecord


def _context_record(tmp_path):
    layout = M2OutputLayout.from_root(tmp_path / "full_preprocessing")
    context = PreprocessingRunContext(project_root=tmp_path, work_root=tmp_path, run_profile="full_preprocessing", output_namespace="full_preprocessing", output_root=layout.output_root, crops_root=layout.crops_root, frames_root=layout.frames_root, manifests_root=layout.manifests_root, state_root=layout.state_root, reports_root=layout.reports_root, logs_root=layout.logs_root, run_id="frame-unavailable", dataset="casia_fasd", dataset_role="source", preprocessing_version="m2-v1", preprocessing_config_hash="a" * 64, detector_model_path=tmp_path / "model.onnx", detector_model_sha256="b" * 64, detector_input_size=320, detector_threshold=.5, all_records=False, record_limit=1, sample_limit=None, resume=False, dry_run=False, partial_full_profile=False, command="test")
    record = CanonicalVideoRecord(dataset="casia_fasd", subject_id="subject-1", video_id="s001v001", source_path=tmp_path / "s001v001f001.png", official_split="train", label="live", adapter_version="casia-v1", source_fingerprint="c" * 64, metadata_provenance="synthetic-test")
    return context, record


def _metadata(context):
    return {"manifest_schema_version": "m2f1a-v1", "preprocessing_version": context.preprocessing_version, "preprocessing_config_hash": context.preprocessing_config_hash, "detector_model_sha256": context.detector_model_sha256, "detector_input_size": str(context.detector_input_size), "detector_threshold": str(context.detector_threshold)}


def test_source_unavailable_index_routes_failure_before_failure_counter(tmp_path, monkeypatch):
    context, record, calls = *_context_record(tmp_path), []

    class Repository:
        def initialize(self): return self
        def flush(self): pass
        def counts(self): return {"source_frames": 0, "source_crops": 0, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 1}
    class Reader:
        def frame_count(self): return 1
        def read_frame(self, index): raise IndexError(index)
        def close(self): pass

    repository = Repository()
    monkeypatch.setattr(m2_runner, "route_preprocessing_failure", lambda **kwargs: calls.append(kwargs))
    result = m2_runner.run_preprocessing(context, [record], detector=type("Detector", (), {"detect": lambda self, image: (_ for _ in ()).throw(AssertionError("detector must not run"))})(), media_reader_factory=lambda _: Reader(), repository_factory=lambda *_: repository)
    assert len(calls) == 1
    routed = calls[0]
    assert routed["repository"] is repository and routed["context"] is context and routed["canonical_record"] is record
    assert routed["sample_id"] and routed["requested_frame_index"] == 0 and routed["actual_frame_index"] is None
    assert routed["stage"] == "media_read" and routed["error_code"] == "frame_index_unavailable" and routed["recoverable"] is True
    assert ":\\" not in routed["error_message"] and "/" not in routed["error_message"]
    assert (result.samples_selected, result.samples_successful, result.samples_failed, result.frames_read, result.detector_calls, result.crops_written) == (1, 0, 1, 0, 0, 0)
    assert result.failures_by_code == {"frame_index_unavailable": 1}


def test_unavailable_index_helper_exception_propagates_without_false_failure(tmp_path, monkeypatch):
    context, record = _context_record(tmp_path)
    class Repository:
        def initialize(self): return self
        def flush(self): raise AssertionError("must not flush after routing exception")
        def counts(self): raise AssertionError("must not count after routing exception")
    class Reader:
        def frame_count(self): return 1
        def read_frame(self, index): raise IndexError(index)
        def close(self): pass
    monkeypatch.setattr(m2_runner, "route_preprocessing_failure", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic unavailable routing error")))
    with pytest.raises(RuntimeError, match="^synthetic unavailable routing error$"):
        m2_runner.run_preprocessing(context, [record], detector=object(), media_reader_factory=lambda _: Reader(), repository_factory=lambda *_: Repository())


def test_unavailable_index_missing_repository_raises_before_helper(tmp_path, monkeypatch):
    context, record, routed = *_context_record(tmp_path), []
    class Repository:
        def initialize(self): return None
    class Reader:
        def frame_count(self): return 1
        def read_frame(self, index): raise IndexError(index)
        def close(self): pass
    monkeypatch.setattr(m2_runner, "route_preprocessing_failure", lambda **kwargs: routed.append(kwargs))
    with pytest.raises(RuntimeError, match="^failure routing requires an initialized manifest repository$"):
        m2_runner.run_preprocessing(context, [record], detector=object(), media_reader_factory=lambda _: Reader(), repository_factory=lambda *_: Repository())
    assert not routed


def test_real_repository_unavailable_index_persists_one_failure_row(tmp_path):
    context, record = _context_record(tmp_path)
    repository = ManifestRepository(context.manifests_root, _metadata(context))
    class Reader:
        def frame_count(self): return 1
        def read_frame(self, index): raise IndexError(index)
        def close(self): pass
    result = m2_runner.run_preprocessing(context, [record], detector=object(), media_reader_factory=lambda _: Reader(), repository_factory=lambda *_: repository)
    assert result.manifest_counts == {"source_frames": 0, "source_crops": 0, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 1}
    repository.load()
    assert repository.rows["preprocessing_failures"][0]["error_code"] == "frame_index_unavailable"


def test_routed_unavailable_index_continues_to_next_source_success(tmp_path):
    context, record = _context_record(tmp_path)
    repository = ManifestRepository(context.manifests_root, _metadata(context))
    class Reader:
        def frame_count(self): return 2
        def read_frame(self, index):
            if index == 0: raise IndexError(index)
            return FrameResult(index, 18, None, 64, 64, np.zeros((64, 64, 3), dtype=np.uint8), "synthetic-reader")
        def close(self): pass
    class Detector:
        def detect(self, image): return [Detection(bbox=(12, 12, 52, 52), score=.91, landmarks=[(20, 22)] * 5)]
    result = m2_runner.run_preprocessing(context, [record], detector=Detector(), media_reader_factory=lambda _: Reader(), repository_factory=lambda *_: repository)
    assert (result.samples_selected, result.samples_successful, result.samples_failed, result.crops_written, result.frames_read, result.detector_calls) == (2, 1, 1, 1, 1, 1)
    assert result.failures_by_code == {"frame_index_unavailable": 1}
    assert result.manifest_counts == {"source_frames": 1, "source_crops": 1, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 1}
