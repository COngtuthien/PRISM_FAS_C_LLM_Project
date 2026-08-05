import ast
from pathlib import Path

import numpy as np
import pytest

import prism_fas.data.m2_runner as m2_runner
from prism_fas.data.media.readers import FrameResult
from prism_fas.data.preprocess_m2 import Detection
from prism_fas.data.run_context import M2OutputLayout, PreprocessingRunContext
from prism_fas.data.schemas.records import CanonicalVideoRecord


def _context(tmp_path):
    layout = M2OutputLayout.from_root(tmp_path / "full_preprocessing")
    return PreprocessingRunContext(project_root=tmp_path, work_root=tmp_path, run_profile="full_preprocessing", output_namespace="full_preprocessing", output_root=layout.output_root, crops_root=layout.crops_root, frames_root=layout.frames_root, manifests_root=layout.manifests_root, state_root=layout.state_root, reports_root=layout.reports_root, logs_root=layout.logs_root, run_id="flush-test", dataset="casia_fasd", dataset_role="source", preprocessing_version="m2-v1", preprocessing_config_hash="a" * 64, detector_model_path=tmp_path / "model.onnx", detector_model_sha256="b" * 64, detector_input_size=320, detector_threshold=.5, all_records=False, record_limit=2, sample_limit=1, resume=False, dry_run=False, partial_full_profile=False, command="test")


def _record(tmp_path, video_id):
    return CanonicalVideoRecord(dataset="casia_fasd", subject_id="subject-1", video_id=video_id, source_path=tmp_path / f"{video_id}f001.png", official_split="train", label="live", adapter_version="casia-v1", source_fingerprint="c" * 64, metadata_provenance="synthetic-test")


def _reader(events, actual_index=1):
    class Reader:
        def frame_count(self): return 1
        def read_frame(self, index):
            events.append("media_read")
            return FrameResult(index, actual_index, None, 64, 64, np.zeros((64, 64, 3), dtype=np.uint8), "synthetic-reader")
        def close(self): pass
    return Reader()


def _detector(events):
    class Detector:
        def detect(self, image):
            events.append("detect")
            return [Detection(bbox=(12, 12, 52, 52), score=.91, landmarks=[(20, 22)] * 5)]
    return Detector()


def test_one_record_flushes_after_route_and_once_before_return(tmp_path, monkeypatch):
    context, events, routed = _context(tmp_path), [], []

    class Repository:
        def initialize(self): events.append("initialize"); return self
        def load(self): events.append("load"); return self
        def flush(self): events.append(f"flush_{events.count('flush_1') + events.count('flush_2') + 1}")
        def counts(self): return {"source_frames": 1, "source_crops": 1, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 0}

    repository = Repository()
    def factory(root, metadata): events.append("factory"); return repository
    def route_probe(**kwargs): events.append("route_source"); routed.append(kwargs)

    monkeypatch.setattr(m2_runner, "route_source_success", route_probe)
    m2_runner.run_preprocessing(context, [_record(tmp_path, "s001v001")], detector=_detector(events), media_reader_factory=lambda _: _reader(events), repository_factory=factory)
    assert events == ["factory", "initialize", "load", "media_read", "detect", "route_source", "flush_1", "flush_2"]
    assert len(routed) == 1 and routed[0]["repository"] is repository


def test_two_records_flush_each_record_before_the_next_route_then_finally(tmp_path, monkeypatch):
    context, events = _context(tmp_path), []

    class Repository:
        def initialize(self): events.append("initialize"); return self
        def load(self): events.append("load"); return self
        def flush(self): events.append("flush")
        def counts(self): return {"source_frames": 2, "source_crops": 2, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 0}

    def factory(root, metadata): events.append("factory"); return Repository()
    def route_probe(**kwargs): events.append(f"route_{kwargs['canonical_record'].video_id}")

    monkeypatch.setattr(m2_runner, "route_source_success", route_probe)
    m2_runner.run_preprocessing(context, [_record(tmp_path, "s001v001"), _record(tmp_path, "s002v001")], detector=_detector(events), media_reader_factory=lambda _: _reader(events), repository_factory=factory)
    assert events == ["factory", "initialize", "load", "media_read", "detect", "route_s001v001", "flush", "media_read", "detect", "route_s002v001", "flush", "flush"]


def test_zero_records_performs_only_final_flush(tmp_path):
    context, events = _context(tmp_path), []

    class Repository:
        def initialize(self): events.append("initialize"); return self
        def load(self): events.append("load"); return self
        def flush(self): events.append("flush_final")
        def counts(self): return {"source_frames": 0, "source_crops": 0, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 0}

    def factory(root, metadata): events.append("factory"); return Repository()

    result = m2_runner.run_preprocessing(context, [], detector=object(), repository_factory=factory)
    assert events == ["factory", "initialize", "load", "flush_final"]
    assert result.samples_selected == result.samples_successful == result.crops_written == 0


def test_record_flush_exception_propagates_and_stops_next_record(tmp_path, monkeypatch):
    context, events = _context(tmp_path), []

    class Repository:
        def initialize(self): events.append("initialize"); return self
        def load(self): events.append("load"); return self
        def flush(self): events.append("flush_record"); raise RuntimeError("synthetic record flush failure")

    def factory(root, metadata): events.append("factory"); return Repository()
    def route_probe(**kwargs): events.append(f"route_{kwargs['canonical_record'].video_id}")

    monkeypatch.setattr(m2_runner, "route_source_success", route_probe)
    with pytest.raises(RuntimeError, match="^synthetic record flush failure$"):
        m2_runner.run_preprocessing(context, [_record(tmp_path, "s001v001"), _record(tmp_path, "s002v001")], detector=_detector(events), media_reader_factory=lambda _: _reader(events), repository_factory=factory)
    assert events == ["factory", "initialize", "load", "media_read", "detect", "route_s001v001", "flush_record"]


def test_final_flush_exception_propagates_without_completed_result(tmp_path):
    context, events = _context(tmp_path), []

    class Repository:
        def initialize(self): events.append("initialize"); return self
        def load(self): events.append("load"); return self
        def flush(self): events.append("flush_final"); raise RuntimeError("synthetic final flush failure")

    def factory(root, metadata): events.append("factory"); return Repository()

    with pytest.raises(RuntimeError, match="^synthetic final flush failure$"):
        m2_runner.run_preprocessing(context, [], detector=object(), repository_factory=factory)
    assert events == ["factory", "initialize", "load", "flush_final"]


def test_context_aware_runner_does_not_call_m2a_migration():
    tree = ast.parse(Path(m2_runner.__file__).read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_preprocessing")
    calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id for node in ast.walk(function) if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))}
    strings = {node.value for node in ast.walk(function) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert "migrate_m2a" not in calls
    assert not any("m2a" in value.lower() for value in strings)
