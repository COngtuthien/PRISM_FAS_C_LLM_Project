from pathlib import Path

import numpy as np
import pytest

import prism_fas.data.m2_runner as m2_runner
from prism_fas.data.media.readers import FrameResult
from prism_fas.data.preprocess_m2 import Detection
from prism_fas.data.run_context import M2OutputLayout, PreprocessingRunContext
from prism_fas.data.schemas.records import CanonicalVideoRecord


COUNT_KEYS = ("source_frames", "source_crops", "target_frames", "target_crops", "preprocessing_failures")


def _context(tmp_path):
    layout = M2OutputLayout.from_root(tmp_path / "full_preprocessing")
    return PreprocessingRunContext(project_root=tmp_path, work_root=tmp_path, run_profile="full_preprocessing", output_namespace="full_preprocessing", output_root=layout.output_root, crops_root=layout.crops_root, frames_root=layout.frames_root, manifests_root=layout.manifests_root, state_root=layout.state_root, reports_root=layout.reports_root, logs_root=layout.logs_root, run_id="counts-test", dataset="casia_fasd", dataset_role="source", preprocessing_version="m2-v1", preprocessing_config_hash="a" * 64, detector_model_path=tmp_path / "model.onnx", detector_model_sha256="b" * 64, detector_input_size=320, detector_threshold=.5, all_records=False, record_limit=1, sample_limit=1, resume=False, dry_run=False, partial_full_profile=False, command="test")


def _record(tmp_path):
    return CanonicalVideoRecord(dataset="casia_fasd", subject_id="subject-1", video_id="s001v001", source_path=tmp_path / "s001v001f001.png", official_split="train", label="live", adapter_version="casia-v1", source_fingerprint="c" * 64, metadata_provenance="synthetic-test")


def _reader(events):
    class Reader:
        def frame_count(self): return 1
        def read_frame(self, index):
            events.append("media_read")
            return FrameResult(index, 3, None, 64, 64, np.zeros((64, 64, 3), dtype=np.uint8), "synthetic-reader")
        def close(self): pass
    return Reader()


def _detector(events):
    class Detector:
        def detect(self, image):
            events.append("detect")
            return [Detection(bbox=(12, 12, 52, 52), score=.91, landmarks=[(20, 22)] * 5)]
    return Detector()


def test_counts_follow_final_flush_and_are_returned_as_strict_int_mapping(tmp_path, monkeypatch):
    context, events, routed = _context(tmp_path), [], []
    raw_counts = {"source_frames": np.int64(1), "source_crops": np.int64(1), "target_frames": np.int64(0), "target_crops": np.int64(0), "preprocessing_failures": np.int64(0)}

    class Repository:
        def initialize(self): events.append("initialize"); return self
        def load(self): events.append("load"); return self
        def flush(self): events.append("flush")
        def counts(self): events.append("counts"); return raw_counts

    repository = Repository()
    def factory(root, metadata): events.append("factory"); return repository
    def route_probe(**kwargs): events.append("route_source"); routed.append(kwargs)

    monkeypatch.setattr(m2_runner, "route_source_success", route_probe)
    result = m2_runner.run_preprocessing(context, [_record(tmp_path)], detector=_detector(events), media_reader_factory=lambda _: _reader(events), repository_factory=factory)
    assert events == ["factory", "initialize", "load", "media_read", "detect", "route_source", "flush", "flush", "counts"]
    assert routed[0]["repository"] is repository
    assert tuple(result.manifest_counts) == COUNT_KEYS
    assert result.manifest_counts == {key: int(value) for key, value in raw_counts.items()}
    assert all(type(value) is int for value in result.manifest_counts.values())


def test_zero_record_run_returns_existing_repository_counts(tmp_path):
    context, events = _context(tmp_path), []
    existing = {"source_frames": 3, "source_crops": 3, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 0}

    class Repository:
        def initialize(self): events.append("initialize"); return self
        def load(self): events.append("load"); return self
        def flush(self): events.append("flush_final")
        def counts(self): events.append("counts"); return existing

    def factory(root, metadata): events.append("factory"); return Repository()

    result = m2_runner.run_preprocessing(context, [], detector=object(), repository_factory=factory)
    assert events == ["factory", "initialize", "load", "flush_final", "counts"]
    assert result.manifest_counts == existing
    assert result.samples_selected == result.samples_successful == result.crops_written == 0


def test_exact_duplicate_execution_keeps_persisted_counts_distinct_from_execution_counters(tmp_path, monkeypatch):
    context = _context(tmp_path)
    counts = {"source_frames": 1, "source_crops": 1, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 0}

    class Repository:
        def initialize(self): return self
        def load(self): return self
        def flush(self): pass
        def counts(self): return counts

    monkeypatch.setattr(m2_runner, "route_source_success", lambda **kwargs: None)
    result = m2_runner.run_preprocessing(context, [_record(tmp_path)], detector=_detector([]), media_reader_factory=lambda _: _reader([]), repository_factory=lambda *_: Repository())
    assert result.samples_successful == result.crops_written == 1
    assert result.manifest_counts["source_frames"] == result.manifest_counts["source_crops"] == 1


@pytest.mark.parametrize("failure_stage", ["record_flush", "final_flush"])
def test_flush_failures_do_not_call_counts(tmp_path, monkeypatch, failure_stage):
    context, events = _context(tmp_path), []

    class Repository:
        def initialize(self): events.append("initialize"); return self
        def load(self): events.append("load"); return self
        def flush(self):
            events.append("flush")
            if failure_stage == "record_flush" or failure_stage == "final_flush": raise RuntimeError(f"synthetic {failure_stage} failure")
        def counts(self): events.append("counts"); return {key: 0 for key in COUNT_KEYS}

    monkeypatch.setattr(m2_runner, "route_source_success", lambda **kwargs: events.append("route_source"))
    records = [_record(tmp_path)] if failure_stage == "record_flush" else []
    with pytest.raises(RuntimeError, match=f"synthetic {failure_stage} failure"):
        m2_runner.run_preprocessing(context, records, detector=_detector(events), media_reader_factory=lambda _: _reader(events), repository_factory=lambda *_: Repository())
    assert "counts" not in events


def test_counts_exception_propagates_without_completed_result(tmp_path):
    context, events = _context(tmp_path), []

    class Repository:
        def initialize(self): events.append("initialize"); return self
        def load(self): events.append("load"); return self
        def flush(self): events.append("flush_final")
        def counts(self): events.append("counts"); raise RuntimeError("synthetic counts failure")

    with pytest.raises(RuntimeError, match="^synthetic counts failure$"):
        m2_runner.run_preprocessing(context, [], detector=object(), repository_factory=lambda *_: Repository())
    assert events == ["initialize", "load", "flush_final", "counts"]
