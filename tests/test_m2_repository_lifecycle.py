import ast
from pathlib import Path

import numpy as np
import pytest

import prism_fas.data.m2_runner as m2_runner
from prism_fas.data.media.readers import FrameResult
from prism_fas.data.preprocess_m2 import Detection, MockFaceDetector
from prism_fas.data.run_context import M2OutputLayout, PreprocessingRunContext
from prism_fas.data.schemas.records import CanonicalVideoRecord


def test_full_profile_repository_lifecycle_precedes_media_and_routes_same_instance(tmp_path, monkeypatch):
    layout = M2OutputLayout.from_root(tmp_path / "full_preprocessing")
    context = PreprocessingRunContext(project_root=tmp_path, work_root=tmp_path, run_profile="full_preprocessing", output_namespace="full_preprocessing", output_root=layout.output_root, crops_root=layout.crops_root, frames_root=layout.frames_root, manifests_root=layout.manifests_root, state_root=layout.state_root, reports_root=layout.reports_root, logs_root=layout.logs_root, run_id="lifecycle", dataset="casia_fasd", dataset_role="source", preprocessing_version="m2-v1", preprocessing_config_hash="a" * 64, detector_model_path=tmp_path / "model.onnx", detector_model_sha256="b" * 64, detector_input_size=320, detector_threshold=.5, all_records=False, record_limit=1, sample_limit=1, resume=False, dry_run=False, partial_full_profile=False, command="test")
    record = CanonicalVideoRecord(dataset="casia_fasd", subject_id="subject-1", video_id="s001v001", source_path=tmp_path / "s001v001f001.png", official_split="train", label="live", adapter_version="casia-v1", source_fingerprint="c" * 64, metadata_provenance="synthetic-test")
    events, factory_calls, routed = [], [], []

    class Repository:
        def initialize(self): events.append("initialize"); return self
        def load(self): events.append("load"); return self
        def flush(self): pass
        def counts(self): return {"source_frames": 1, "source_crops": 1, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 0}

    repository = Repository()

    class Reader:
        def frame_count(self): return 1
        def read_frame(self, index):
            events.append("media_read")
            return FrameResult(index, 9, None, 64, 64, np.zeros((64, 64, 3), dtype=np.uint8), "synthetic-reader")
        def close(self): pass

    class Detector:
        def detect(self, image):
            events.append("detect")
            return [Detection(bbox=(12, 12, 52, 52), score=.91, landmarks=[(20, 22)] * 5)]

    def factory(root, metadata):
        events.append("factory")
        factory_calls.append((root, metadata))
        return repository

    def route_probe(**kwargs):
        events.append("route_source")
        routed.append(kwargs)

    monkeypatch.setattr(m2_runner, "route_source_success", route_probe)
    m2_runner.run_preprocessing(context, [record], detector=Detector(), media_reader_factory=lambda _: Reader(), repository_factory=factory)

    assert len(factory_calls) == 1
    root, metadata = factory_calls[0]
    assert root == context.manifests_root
    assert metadata["output_root"] == str(context.output_root)
    assert metadata["run_profile"] == context.run_profile
    assert metadata["preprocessing_config_hash"] == context.preprocessing_config_hash
    assert metadata["detector_model_sha256"] == context.detector_model_sha256
    assert metadata["detector_input_size"] == str(context.detector_input_size)
    assert metadata["detector_threshold"] == str(context.detector_threshold)
    assert events[:6] == ["factory", "initialize", "load", "media_read", "detect", "route_source"]
    assert len(routed) == 1 and routed[0]["repository"] is repository


@pytest.mark.parametrize("failing_stage", ["initialize", "load"])
def test_repository_lifecycle_exceptions_stop_media_before_routing(tmp_path, monkeypatch, failing_stage):
    layout = M2OutputLayout.from_root(tmp_path / "full_preprocessing")
    context = PreprocessingRunContext(project_root=tmp_path, work_root=tmp_path, run_profile="full_preprocessing", output_namespace="full_preprocessing", output_root=layout.output_root, crops_root=layout.crops_root, frames_root=layout.frames_root, manifests_root=layout.manifests_root, state_root=layout.state_root, reports_root=layout.reports_root, logs_root=layout.logs_root, run_id="lifecycle-error", dataset="casia_fasd", dataset_role="source", preprocessing_version="m2-v1", preprocessing_config_hash="a" * 64, detector_model_path=tmp_path / "model.onnx", detector_model_sha256="b" * 64, detector_input_size=320, detector_threshold=.5, all_records=False, record_limit=1, sample_limit=1, resume=False, dry_run=False, partial_full_profile=False, command="test")
    record = CanonicalVideoRecord(dataset="casia_fasd", subject_id="subject-1", video_id="s001v001", source_path=tmp_path / "s001v001f001.png", official_split="train", label="live", adapter_version="casia-v1", source_fingerprint="c" * 64, metadata_provenance="synthetic-test")
    events = []

    class Repository:
        def initialize(self):
            events.append("initialize")
            if failing_stage == "initialize": raise RuntimeError("synthetic initialize failure")
            return self
        def load(self):
            events.append("load")
            if failing_stage == "load": raise RuntimeError("synthetic load failure")
            return self

    def factory(root, metadata): events.append("factory"); return Repository()
    def unexpected_reader(record): events.append("reader"); raise AssertionError("media processing started")
    def unexpected_route(**kwargs): events.append("route"); raise AssertionError("source routing started")

    monkeypatch.setattr(m2_runner, "route_source_success", unexpected_route)
    with pytest.raises(RuntimeError, match=f"synthetic {failing_stage} failure"):
        m2_runner.run_preprocessing(context, [record], detector=object(), media_reader_factory=unexpected_reader, repository_factory=factory)
    assert events == (["factory", "initialize"] if failing_stage == "initialize" else ["factory", "initialize", "load"])


def test_context_aware_runner_has_no_m2a_migration_call():
    runner_path = Path(m2_runner.__file__)
    tree = ast.parse(runner_path.read_text(encoding="utf-8"), filename=str(runner_path))
    run_preprocessing = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_preprocessing")
    names = {node.id for node in ast.walk(run_preprocessing) if isinstance(node, ast.Name)}
    strings = {node.value for node in ast.walk(run_preprocessing) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert "migrate_m2a" not in names
    assert not any("m2a" in value.lower() for value in strings)
