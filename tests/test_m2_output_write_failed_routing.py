import os

import numpy as np
import pytest

import prism_fas.data.m2_runner as m2_runner
import prism_fas.data.output.writers as writers
from prism_fas.data.manifests.repository import ManifestRepository
from prism_fas.data.media.readers import FrameResult
from prism_fas.data.output import OutputWriteError, write_crop_image
from prism_fas.data.preprocess_m2 import CropProcessingError, Detection
from prism_fas.data.run_context import M2OutputLayout, PreprocessingRunContext
from prism_fas.data.schemas.records import CanonicalVideoRecord


def _context_record(tmp_path):
    layout = M2OutputLayout.from_root(tmp_path / "full_preprocessing")
    context = PreprocessingRunContext(project_root=tmp_path, work_root=tmp_path, run_profile="full_preprocessing", output_namespace="full_preprocessing", output_root=layout.output_root, crops_root=layout.crops_root, frames_root=layout.frames_root, manifests_root=layout.manifests_root, state_root=layout.state_root, reports_root=layout.reports_root, logs_root=layout.logs_root, run_id="output-write-failed", dataset="casia_fasd", dataset_role="source", preprocessing_version="m2-v1", preprocessing_config_hash="a" * 64, detector_model_path=tmp_path / "model.onnx", detector_model_sha256="b" * 64, detector_input_size=320, detector_threshold=.5, all_records=False, record_limit=1, sample_limit=None, resume=False, dry_run=False, partial_full_profile=False, command="test")
    record = CanonicalVideoRecord(dataset="casia_fasd", subject_id="subject-1", video_id="s001v001", source_path=tmp_path / "s001v001f001.png", official_split="train", label="live", adapter_version="casia-v1", source_fingerprint="c" * 64, metadata_provenance="synthetic-test")
    return context, record


def _metadata(context):
    return {"manifest_schema_version": "m2f1a-v1", "preprocessing_version": context.preprocessing_version, "preprocessing_config_hash": context.preprocessing_config_hash, "detector_model_sha256": context.detector_model_sha256, "detector_input_size": str(context.detector_input_size), "detector_threshold": str(context.detector_threshold)}


def _reader(count=1):
    class Reader:
        def frame_count(self): return count
        def read_frame(self, index): return FrameResult(index, index + 3, None, 64, 64, np.zeros((64, 64, 3), dtype=np.uint8), "synthetic-reader")
        def close(self): pass
    return Reader()


def _detector(detections=None):
    faces = detections if detections is not None else [Detection(bbox=(12, 12, 52, 52), score=.91, landmarks=[(20, 22)] * 5)]
    return type("Detector", (), {"detect": lambda self, image: faces})()


class _CountingRepository:
    def __init__(self, failures=0): self.failures = failures
    def initialize(self): return self
    def flush(self): pass
    def counts(self): return {"source_frames": 0, "source_crops": 0, "target_frames": 0, "target_crops": 0, "preprocessing_failures": self.failures}


def _failing_writer(calls, error=None):
    def writer(image, target, **kwargs):
        calls.append(target)
        raise error if error is not None else OutputWriteError()
    return writer


def _artifacts(context):
    return sorted(p.name for p in context.crops_root.rglob("*") if p.is_file())


# --- writer boundary -------------------------------------------------------

def test_write_crop_image_materializes_valid_crop(tmp_path):
    target = write_crop_image(np.zeros((224, 224, 3), dtype=np.uint8), tmp_path / "crops" / "casia_fasd" / "sample.jpg")
    assert target.exists() and target.stat().st_size > 0
    assert not list(target.parent.glob("*.tmp*"))


@pytest.mark.parametrize("failure", ["backend_false", "permission_error", "encoder_error", "zero_byte_output"])
def test_writer_boundary_converts_backend_failures_to_typed_error(tmp_path, monkeypatch, failure):
    def imwrite(path, image, params=None):
        if failure == "backend_false": return False
        if failure == "permission_error": raise PermissionError("synthetic permission denied")
        if failure == "encoder_error": raise writers.cv2.error("synthetic encoder failure")
        open(path, "wb").close()
        return True
    monkeypatch.setattr(writers.cv2, "imwrite", imwrite)
    target = tmp_path / "crops" / "sample.jpg"
    with pytest.raises(OutputWriteError) as excinfo:
        write_crop_image(np.zeros((224, 224, 3), dtype=np.uint8), target)
    assert excinfo.value.backend == writers.CROP_WRITER_BACKEND
    assert not target.exists() and not list(target.parent.glob("*.tmp*"))


def test_writer_cleans_up_temporary_file_when_atomic_replace_fails(tmp_path, monkeypatch):
    target = tmp_path / "crops" / "sample.jpg"
    monkeypatch.setattr(writers.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("synthetic replace failure")))
    with pytest.raises(OutputWriteError):
        write_crop_image(np.zeros((224, 224, 3), dtype=np.uint8), target)
    assert not target.exists() and not list(target.parent.glob("*"))


def test_writer_removes_empty_final_artifact_reported_as_success(tmp_path, monkeypatch):
    target = tmp_path / "crops" / "sample.jpg"
    real_replace = os.replace
    def replace(src, dst):
        real_replace(src, dst)
        open(dst, "wb").close()
    monkeypatch.setattr(writers.os, "replace", replace)
    with pytest.raises(OutputWriteError):
        write_crop_image(np.zeros((224, 224, 3), dtype=np.uint8), target)
    assert not target.exists() and not list(target.parent.glob("*"))


# --- runner routing --------------------------------------------------------

def test_typed_output_write_error_routes_failure_once_with_expected_arguments(tmp_path, monkeypatch):
    context, record, routed, written = *_context_record(tmp_path), [], []
    repository = _CountingRepository(failures=1)
    monkeypatch.setattr(m2_runner, "write_crop_image", _failing_writer(written))
    monkeypatch.setattr(m2_runner, "route_preprocessing_failure", lambda **kwargs: routed.append(kwargs))
    result = m2_runner.run_preprocessing(context, [record], detector=_detector(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: repository)
    assert len(written) == 1 and len(routed) == 1
    call = routed[0]
    assert call["repository"] is repository and call["context"] is context and call["canonical_record"] is record
    assert call["requested_frame_index"] == 0 and call["actual_frame_index"] == 3
    assert call["stage"] == "output_write" and call["error_code"] == "output_write_failed" and call["recoverable"] is True
    assert call["backend"] == writers.CROP_WRITER_BACKEND
    assert call["error_message"] == "face crop output could not be written"
    assert ":\\" not in call["error_message"] and str(tmp_path) not in call["error_message"]
    assert (result.samples_selected, result.samples_successful, result.samples_failed, result.frames_read, result.detector_calls, result.crops_written) == (1, 0, 1, 1, 1, 0)
    assert result.failures_by_code == {"output_write_failed": 1}


def test_backend_write_failure_routes_output_write_failed_without_artifact(tmp_path, monkeypatch):
    context, record = _context_record(tmp_path)
    repository = ManifestRepository(context.manifests_root, _metadata(context))
    monkeypatch.setattr(writers.cv2, "imwrite", lambda *args, **kwargs: False)
    result = m2_runner.run_preprocessing(context, [record], detector=_detector(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: repository)
    assert result.failures_by_code == {"output_write_failed": 1}
    assert (result.samples_successful, result.crops_written) == (0, 0)
    assert _artifacts(context) == []
    assert result.manifest_counts == {"source_frames": 0, "source_crops": 0, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 1}


def test_real_repository_persists_exactly_one_output_write_failure_row(tmp_path, monkeypatch):
    context, record = _context_record(tmp_path)
    repository = ManifestRepository(context.manifests_root, _metadata(context))
    monkeypatch.setattr(m2_runner, "write_crop_image", _failing_writer([]))
    result = m2_runner.run_preprocessing(context, [record], detector=_detector(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: repository)
    assert result.manifest_counts == {"source_frames": 0, "source_crops": 0, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 1}
    repository.load()
    rows = repository.rows["preprocessing_failures"]
    assert len(rows) == 1
    assert rows[0]["error_code"] == "output_write_failed" and rows[0]["stage"] == "output_write"
    assert rows[0]["recoverable"] is True and rows[0]["sample_id"]
    assert str(tmp_path) not in rows[0]["sanitized_error_message"]


def test_output_write_failure_helper_exception_propagates_without_counters(tmp_path, monkeypatch):
    context, record = _context_record(tmp_path)
    class Repository:
        def initialize(self): return self
        def flush(self): raise AssertionError("must not flush after routing exception")
        def counts(self): raise AssertionError("must not count after routing exception")
    monkeypatch.setattr(m2_runner, "write_crop_image", _failing_writer([]))
    monkeypatch.setattr(m2_runner, "route_preprocessing_failure", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic write routing error")))
    with pytest.raises(RuntimeError, match="^synthetic write routing error$"):
        m2_runner.run_preprocessing(context, [record], detector=_detector(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: Repository())


def test_missing_repository_raises_before_failure_helper(tmp_path, monkeypatch):
    context, record, routed = *_context_record(tmp_path), []
    class MissingRepository:
        def initialize(self): return None
    monkeypatch.setattr(m2_runner, "write_crop_image", _failing_writer([]))
    monkeypatch.setattr(m2_runner, "route_preprocessing_failure", lambda **kwargs: routed.append(kwargs))
    with pytest.raises(RuntimeError, match="^failure routing requires an initialized manifest repository$"):
        m2_runner.run_preprocessing(context, [record], detector=_detector(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: MissingRepository())
    assert not routed


def test_routed_output_write_failure_continues_to_next_sample(tmp_path, monkeypatch):
    context, record = _context_record(tmp_path)
    repository = ManifestRepository(context.manifests_root, _metadata(context))
    original, calls = m2_runner.write_crop_image, []
    def write_once(image, target, **kwargs):
        calls.append(target)
        if len(calls) == 1: raise OutputWriteError()
        return original(image, target, **kwargs)
    monkeypatch.setattr(m2_runner, "write_crop_image", write_once)
    result = m2_runner.run_preprocessing(context, [record], detector=_detector(), media_reader_factory=lambda _: _reader(2), repository_factory=lambda *_: repository)
    assert (result.samples_selected, result.samples_successful, result.samples_failed, result.frames_read, result.detector_calls, result.crops_written) == (2, 1, 1, 2, 2, 1)
    assert result.failures_by_code == {"output_write_failed": 1}
    assert result.manifest_counts == {"source_frames": 1, "source_crops": 1, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 1}
    assert _artifacts(context) == [calls[1].name] and calls[0].name not in _artifacts(context)


def test_hash_and_source_success_are_not_reached_after_write_failure(tmp_path, monkeypatch):
    context, record, hashed, successes = *_context_record(tmp_path), [], []
    monkeypatch.setattr(m2_runner, "write_crop_image", _failing_writer([]))
    monkeypatch.setattr(m2_runner, "hash_crop_artifact", lambda path: hashed.append(path) or "d" * 64)
    monkeypatch.setattr(m2_runner, "route_source_success", lambda **kwargs: successes.append(kwargs))
    monkeypatch.setattr(m2_runner, "route_preprocessing_failure", lambda **kwargs: None)
    result = m2_runner.run_preprocessing(context, [record], detector=_detector(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: _CountingRepository(failures=1))
    assert hashed == [] and successes == []
    assert result.failures_by_code == {"output_write_failed": 1}


def test_hash_failure_after_successful_write_is_not_output_write_failed(tmp_path, monkeypatch):
    context, record, routed = *_context_record(tmp_path), []
    monkeypatch.setattr(m2_runner, "hash_crop_artifact", lambda path: (_ for _ in ()).throw(RuntimeError("synthetic hash failure")))
    monkeypatch.setattr(m2_runner, "route_preprocessing_failure", lambda **kwargs: routed.append(kwargs))
    with pytest.raises(RuntimeError, match="^synthetic hash failure$"):
        m2_runner.run_preprocessing(context, [record], detector=_detector(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: _CountingRepository())
    assert not any(call["error_code"] == "output_write_failed" for call in routed)


@pytest.mark.parametrize("failing", ["route_source_success", "repository_counts"])
def test_generic_errors_outside_writer_boundary_are_not_output_write_failed(tmp_path, monkeypatch, failing):
    context, record, routed = *_context_record(tmp_path), []
    class Repository(_CountingRepository):
        def counts(self):
            if failing == "repository_counts": raise ValueError("synthetic repository failure")
            return super().counts()
    if failing == "route_source_success":
        monkeypatch.setattr(m2_runner, "route_source_success", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic routing failure")))
    monkeypatch.setattr(m2_runner, "route_preprocessing_failure", lambda **kwargs: routed.append(kwargs))
    with pytest.raises((RuntimeError, ValueError)):
        m2_runner.run_preprocessing(context, [record], detector=_detector(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: Repository())
    assert not any(call["error_code"] == "output_write_failed" for call in routed)


# --- precedence ------------------------------------------------------------

@pytest.mark.parametrize("scenario,expected", [("invalid_bbox", "invalid_bbox"), ("invalid_landmarks", "invalid_landmarks"), ("crop_failed", "crop_failed")])
def test_geometry_and_crop_failures_take_precedence_over_write_failures(tmp_path, monkeypatch, scenario, expected):
    context, record, routed, written = *_context_record(tmp_path), [], []
    detections = {"invalid_bbox": [Detection(bbox=(30, 10, 10, 40), score=.91, landmarks=[(20, 22)] * 5)],
                  "invalid_landmarks": [Detection(bbox=(12, 12, 52, 52), score=.91, landmarks=[(float("nan"), 22)] * 5)],
                  "crop_failed": [Detection(bbox=(12, 12, 52, 52), score=.91, landmarks=[(20, 22)] * 5)]}[scenario]
    if scenario == "crop_failed":
        monkeypatch.setattr(m2_runner, "crop_face", lambda *args: (_ for _ in ()).throw(CropProcessingError("synthetic crop failure")))
    monkeypatch.setattr(m2_runner, "write_crop_image", _failing_writer(written))
    monkeypatch.setattr(m2_runner, "route_preprocessing_failure", lambda **kwargs: routed.append(kwargs))
    result = m2_runner.run_preprocessing(context, [record], detector=_detector(detections), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: _CountingRepository(failures=1))
    assert written == []
    assert [call["error_code"] for call in routed] == [expected]
    assert result.failures_by_code == {expected: 1}
    assert result.failures_by_code.get("output_write_failed", 0) == 0


def test_only_output_write_failed_code_is_introduced(tmp_path, monkeypatch):
    context, record = _context_record(tmp_path)
    monkeypatch.setattr(m2_runner, "write_crop_image", _failing_writer([]))
    result = m2_runner.run_preprocessing(context, [record], detector=_detector(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: ManifestRepository(context.manifests_root, _metadata(context)))
    assert set(result.failures_by_code) == {"output_write_failed"}
