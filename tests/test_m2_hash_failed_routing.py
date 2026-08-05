from pathlib import Path

import numpy as np
import pytest

import prism_fas.data.m2_runner as m2_runner
import prism_fas.data.output.hashing as hashing
from prism_fas.data.manifests.repository import ManifestRepository
from prism_fas.data.media.readers import FrameResult
from prism_fas.data.output import CROP_HASH_BACKEND, HashComputationError, OutputWriteError, hash_crop_artifact, validate_digest, write_crop_image
from prism_fas.data.preprocess_m2 import CropProcessingError, Detection
from prism_fas.data.run_context import M2OutputLayout, PreprocessingRunContext
from prism_fas.data.schemas.records import CanonicalVideoRecord
from prism_fas.utils.core import sha256_file


def _context_record(tmp_path):
    layout = M2OutputLayout.from_root(tmp_path / "full_preprocessing")
    context = PreprocessingRunContext(project_root=tmp_path, work_root=tmp_path, run_profile="full_preprocessing", output_namespace="full_preprocessing", output_root=layout.output_root, crops_root=layout.crops_root, frames_root=layout.frames_root, manifests_root=layout.manifests_root, state_root=layout.state_root, reports_root=layout.reports_root, logs_root=layout.logs_root, run_id="hash-failed", dataset="casia_fasd", dataset_role="source", preprocessing_version="m2-v1", preprocessing_config_hash="a" * 64, detector_model_path=tmp_path / "model.onnx", detector_model_sha256="b" * 64, detector_input_size=320, detector_threshold=.5, all_records=False, record_limit=1, sample_limit=None, resume=False, dry_run=False, partial_full_profile=False, command="test")
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


def _failing_hash(calls=None, error=None):
    def hasher(path):
        if calls is not None: calls.append(path)
        raise error if error is not None else HashComputationError()
    return hasher


def _artifacts(context):
    return sorted(p.name for p in context.crops_root.rglob("*") if p.is_file())


# --- typed hash helper -----------------------------------------------------

def test_hash_crop_artifact_returns_valid_lowercase_sha256(tmp_path):
    artifact = write_crop_image(np.zeros((224, 224, 3), dtype=np.uint8), tmp_path / "crops" / "sample.jpg")
    digest = hash_crop_artifact(artifact)
    assert digest == sha256_file(artifact)
    assert len(digest) == 64 and digest == digest.lower() and all(c in "0123456789abcdef" for c in digest)


def test_hash_crop_artifact_missing_file_is_typed_and_safe(tmp_path):
    missing = tmp_path / "crops" / "absent.jpg"
    with pytest.raises(HashComputationError) as excinfo:
        hash_crop_artifact(missing)
    assert excinfo.value.backend == CROP_HASH_BACKEND == "sha256"
    assert str(excinfo.value) == "face crop hash could not be computed"
    assert str(tmp_path) not in str(excinfo.value) and ":\\" not in str(excinfo.value)


@pytest.mark.parametrize("error", [OSError("synthetic read failure"), PermissionError("synthetic permission denied"), ValueError("synthetic digest backend failure"), TypeError("synthetic backend type failure")])
def test_hash_backend_failures_become_typed_hash_errors(tmp_path, monkeypatch, error):
    artifact = write_crop_image(np.zeros((224, 224, 3), dtype=np.uint8), tmp_path / "crops" / "sample.jpg")
    monkeypatch.setattr(hashing, "sha256_file", lambda path: (_ for _ in ()).throw(error))
    with pytest.raises(HashComputationError):
        hash_crop_artifact(artifact)


@pytest.mark.parametrize("digest", ["", "abc", "d" * 63, "d" * 65, "z" * 64, "d" * 62 + "!!", None, 12345, b"\xff" * 64])
def test_malformed_digests_are_rejected(tmp_path, monkeypatch, digest):
    artifact = write_crop_image(np.zeros((224, 224, 3), dtype=np.uint8), tmp_path / "crops" / "sample.jpg")
    monkeypatch.setattr(hashing, "sha256_file", lambda path: digest)
    with pytest.raises(HashComputationError):
        hash_crop_artifact(artifact)
    with pytest.raises(HashComputationError):
        validate_digest(digest)


def test_valid_digest_is_normalized_to_lowercase(tmp_path, monkeypatch):
    artifact = write_crop_image(np.zeros((224, 224, 3), dtype=np.uint8), tmp_path / "crops" / "sample.jpg")
    monkeypatch.setattr(hashing, "sha256_file", lambda path: ("A1" * 32) + "")
    assert hash_crop_artifact(artifact) == "a1" * 32
    assert validate_digest((b"B2" * 32)) == "b2" * 32


def test_unrelated_backend_exception_is_not_wrapped_as_hash_error(tmp_path, monkeypatch):
    artifact = write_crop_image(np.zeros((224, 224, 3), dtype=np.uint8), tmp_path / "crops" / "sample.jpg")
    monkeypatch.setattr(hashing, "sha256_file", lambda path: (_ for _ in ()).throw(RuntimeError("synthetic unrelated failure")))
    with pytest.raises(RuntimeError, match="^synthetic unrelated failure$") as excinfo:
        hash_crop_artifact(artifact)
    assert not isinstance(excinfo.value, HashComputationError)


# --- runner routing --------------------------------------------------------

def test_typed_hash_error_routes_failure_once_with_expected_arguments(tmp_path, monkeypatch):
    context, record, routed = *_context_record(tmp_path), []
    repository = _CountingRepository(failures=1)
    monkeypatch.setattr(m2_runner, "hash_crop_artifact", _failing_hash())
    monkeypatch.setattr(m2_runner, "route_preprocessing_failure", lambda **kwargs: routed.append(kwargs))
    result = m2_runner.run_preprocessing(context, [record], detector=_detector(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: repository)
    assert len(routed) == 1
    call = routed[0]
    assert call["repository"] is repository and call["context"] is context and call["canonical_record"] is record
    assert call["requested_frame_index"] == 0 and call["actual_frame_index"] == 3
    assert call["stage"] == "hash" and call["error_code"] == "hash_failed" and call["recoverable"] is True
    assert call["backend"] == "sha256"
    assert call["error_message"] == "face crop hash could not be computed"
    assert str(tmp_path) not in call["error_message"] and ":\\" not in call["error_message"]
    assert (result.samples_selected, result.samples_successful, result.samples_failed, result.frames_read, result.detector_calls, result.crops_written) == (1, 0, 1, 1, 1, 0)
    assert result.failures_by_code == {"hash_failed": 1}


def test_hash_failure_cleans_up_written_crop_artifact(tmp_path, monkeypatch):
    context, record, hashed = *_context_record(tmp_path), []
    repository = ManifestRepository(context.manifests_root, _metadata(context))
    monkeypatch.setattr(m2_runner, "hash_crop_artifact", _failing_hash(hashed))
    result = m2_runner.run_preprocessing(context, [record], detector=_detector(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: repository)
    assert len(hashed) == 1 and Path(hashed[0]).suffix == ".jpg"
    assert _artifacts(context) == []
    assert result.failures_by_code == {"hash_failed": 1} and result.crops_written == 0
    assert result.manifest_counts == {"source_frames": 0, "source_crops": 0, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 1}


def test_real_repository_persists_exactly_one_hash_failure_row(tmp_path, monkeypatch):
    context, record = _context_record(tmp_path)
    repository = ManifestRepository(context.manifests_root, _metadata(context))
    monkeypatch.setattr(m2_runner, "hash_crop_artifact", _failing_hash())
    m2_runner.run_preprocessing(context, [record], detector=_detector(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: repository)
    repository.load()
    rows = repository.rows["preprocessing_failures"]
    assert len(rows) == 1 and not repository.rows["source_frames"] and not repository.rows["source_crops"]
    row = rows[0]
    assert row["sample_id"] and row["dataset"] == "casia_fasd" and row["source_record_id"] == "s001v001"
    assert row["requested_frame_index"] == 0 and row["actual_frame_index"] == 3
    assert row["stage"] == "hash" and row["error_code"] == "hash_failed" and row["recoverable"] is True
    assert row["backend"] == "sha256" and row["sanitized_error_message"] == "face crop hash could not be computed"
    assert row["preprocessing_config_hash"] == context.preprocessing_config_hash
    assert row["detector_model_sha256"] == context.detector_model_sha256
    assert str(tmp_path) not in row["sanitized_error_message"]


def test_hash_failure_helper_exception_propagates_and_still_cleans_artifact(tmp_path, monkeypatch):
    context, record = _context_record(tmp_path)
    class Repository:
        def initialize(self): return self
        def flush(self): raise AssertionError("must not flush after routing exception")
        def counts(self): raise AssertionError("must not count after routing exception")
    monkeypatch.setattr(m2_runner, "hash_crop_artifact", _failing_hash())
    monkeypatch.setattr(m2_runner, "route_preprocessing_failure", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic hash routing error")))
    with pytest.raises(RuntimeError, match="^synthetic hash routing error$"):
        m2_runner.run_preprocessing(context, [record], detector=_detector(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: Repository())
    assert _artifacts(context) == []


def test_missing_repository_raises_before_hash_failure_helper(tmp_path, monkeypatch):
    context, record, routed = *_context_record(tmp_path), []
    class MissingRepository:
        def initialize(self): return None
    monkeypatch.setattr(m2_runner, "hash_crop_artifact", _failing_hash())
    monkeypatch.setattr(m2_runner, "route_preprocessing_failure", lambda **kwargs: routed.append(kwargs))
    with pytest.raises(RuntimeError, match="^failure routing requires an initialized manifest repository$"):
        m2_runner.run_preprocessing(context, [record], detector=_detector(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: MissingRepository())
    assert not routed and _artifacts(context) == []


def test_cleanup_failure_does_not_mask_hash_failure_routing(tmp_path, monkeypatch):
    context, record, routed = *_context_record(tmp_path), []
    monkeypatch.setattr(m2_runner, "hash_crop_artifact", _failing_hash())
    monkeypatch.setattr(Path, "unlink", lambda self, **kwargs: (_ for _ in ()).throw(PermissionError("synthetic cleanup failure")))
    monkeypatch.setattr(m2_runner, "route_preprocessing_failure", lambda **kwargs: routed.append(kwargs))
    result = m2_runner.run_preprocessing(context, [record], detector=_detector(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: _CountingRepository(failures=1))
    assert [call["error_code"] for call in routed] == ["hash_failed"]
    assert result.failures_by_code == {"hash_failed": 1}


def test_source_success_is_not_routed_after_hash_failure(tmp_path, monkeypatch):
    context, record, successes = *_context_record(tmp_path), []
    monkeypatch.setattr(m2_runner, "hash_crop_artifact", _failing_hash())
    monkeypatch.setattr(m2_runner, "route_source_success", lambda **kwargs: successes.append(kwargs))
    monkeypatch.setattr(m2_runner, "route_preprocessing_failure", lambda **kwargs: None)
    result = m2_runner.run_preprocessing(context, [record], detector=_detector(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: _CountingRepository(failures=1))
    assert successes == []
    assert (result.samples_successful, result.crops_written) == (0, 0)


def test_routed_hash_failure_continues_to_next_sample(tmp_path, monkeypatch):
    context, record = _context_record(tmp_path)
    repository = ManifestRepository(context.manifests_root, _metadata(context))
    original, calls = m2_runner.hash_crop_artifact, []
    def hash_once(path):
        calls.append(Path(path))
        if len(calls) == 1: raise HashComputationError()
        return original(path)
    monkeypatch.setattr(m2_runner, "hash_crop_artifact", hash_once)
    result = m2_runner.run_preprocessing(context, [record], detector=_detector(), media_reader_factory=lambda _: _reader(2), repository_factory=lambda *_: repository)
    assert (result.samples_selected, result.samples_successful, result.samples_failed, result.frames_read, result.detector_calls, result.crops_written) == (2, 1, 1, 2, 2, 1)
    assert result.failures_by_code == {"hash_failed": 1}
    assert result.manifest_counts == {"source_frames": 1, "source_crops": 1, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 1}
    assert _artifacts(context) == [calls[1].name] and calls[0].name != calls[1].name
    assert not list(context.crops_root.rglob("*.tmp*"))


def test_valid_hash_still_reaches_source_success_with_digest(tmp_path):
    context, record = _context_record(tmp_path)
    repository = ManifestRepository(context.manifests_root, _metadata(context))
    result = m2_runner.run_preprocessing(context, [record], detector=_detector(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: repository)
    assert (result.samples_successful, result.samples_failed, result.crops_written) == (1, 0, 1)
    assert result.failures_by_code == {}
    assert result.manifest_counts == {"source_frames": 1, "source_crops": 1, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 0}
    repository.load()
    artifact = next(p for p in context.crops_root.rglob("*.jpg"))
    assert repository.rows["source_crops"][0]["crop_sha256"] == sha256_file(artifact)


# --- precedence and classification safety ----------------------------------

def test_output_write_failure_keeps_precedence_and_skips_hash(tmp_path, monkeypatch):
    context, record, routed, hashed = *_context_record(tmp_path), [], []
    monkeypatch.setattr(m2_runner, "write_crop_image", lambda image, target, **kwargs: (_ for _ in ()).throw(OutputWriteError()))
    monkeypatch.setattr(m2_runner, "hash_crop_artifact", _failing_hash(hashed))
    monkeypatch.setattr(m2_runner, "route_preprocessing_failure", lambda **kwargs: routed.append(kwargs))
    result = m2_runner.run_preprocessing(context, [record], detector=_detector(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: _CountingRepository(failures=1))
    assert hashed == [] and [call["error_code"] for call in routed] == ["output_write_failed"]
    assert result.failures_by_code == {"output_write_failed": 1} and "hash_failed" not in result.failures_by_code


def test_crop_failure_keeps_precedence_over_write_and_hash(tmp_path, monkeypatch):
    context, record, routed, written, hashed = *_context_record(tmp_path), [], [], []
    monkeypatch.setattr(m2_runner, "crop_face", lambda *args: (_ for _ in ()).throw(CropProcessingError("synthetic crop failure")))
    monkeypatch.setattr(m2_runner, "write_crop_image", lambda image, target, **kwargs: written.append(target))
    monkeypatch.setattr(m2_runner, "hash_crop_artifact", _failing_hash(hashed))
    monkeypatch.setattr(m2_runner, "route_preprocessing_failure", lambda **kwargs: routed.append(kwargs))
    result = m2_runner.run_preprocessing(context, [record], detector=_detector(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: _CountingRepository(failures=1))
    assert written == [] and hashed == []
    assert [call["error_code"] for call in routed] == ["crop_failed"]
    assert result.failures_by_code == {"crop_failed": 1}


def test_successful_write_with_hash_failure_routes_hash_failed_only(tmp_path, monkeypatch):
    context, record, routed = *_context_record(tmp_path), []
    monkeypatch.setattr(m2_runner, "hash_crop_artifact", _failing_hash())
    monkeypatch.setattr(m2_runner, "route_preprocessing_failure", lambda **kwargs: routed.append(kwargs))
    result = m2_runner.run_preprocessing(context, [record], detector=_detector(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: _CountingRepository(failures=1))
    assert [call["error_code"] for call in routed] == ["hash_failed"]
    assert result.failures_by_code.get("output_write_failed", 0) == 0
    assert set(result.failures_by_code) == {"hash_failed"}


def test_source_routing_error_after_valid_hash_is_not_hash_failed(tmp_path, monkeypatch):
    context, record, routed = *_context_record(tmp_path), []
    monkeypatch.setattr(m2_runner, "route_source_success", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic source routing error")))
    monkeypatch.setattr(m2_runner, "route_preprocessing_failure", lambda **kwargs: routed.append(kwargs))
    with pytest.raises(RuntimeError, match="^synthetic source routing error$"):
        m2_runner.run_preprocessing(context, [record], detector=_detector(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: _CountingRepository())
    assert not any(call["error_code"] == "hash_failed" for call in routed)


@pytest.mark.parametrize("failing", ["upsert_source_success", "flush", "counts"])
def test_repository_errors_are_not_hash_failed(tmp_path, monkeypatch, failing):
    context, record, routed = *_context_record(tmp_path), []
    class Repository(_CountingRepository):
        def upsert_source_success(self, frame, crop):
            if failing == "upsert_source_success": raise RuntimeError("synthetic repository upsert failure")
        def upsert_failure(self, row): raise AssertionError("must not persist a failure row")
        def flush(self):
            if failing == "flush": raise RuntimeError("synthetic repository flush failure")
        def counts(self):
            if failing == "counts": raise RuntimeError("synthetic repository counts failure")
            return super().counts()
    monkeypatch.setattr(m2_runner, "route_preprocessing_failure", lambda **kwargs: routed.append(kwargs))
    with pytest.raises(RuntimeError):
        m2_runner.run_preprocessing(context, [record], detector=_detector(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: Repository())
    assert not any(call["error_code"] == "hash_failed" for call in routed)


def test_generic_error_outside_hash_helper_is_not_hash_failed(tmp_path, monkeypatch):
    context, record, routed = *_context_record(tmp_path), []
    monkeypatch.setattr(m2_runner, "hash_crop_artifact", lambda path: (_ for _ in ()).throw(RuntimeError("synthetic unrelated failure")))
    monkeypatch.setattr(m2_runner, "route_preprocessing_failure", lambda **kwargs: routed.append(kwargs))
    with pytest.raises(RuntimeError, match="^synthetic unrelated failure$") as excinfo:
        m2_runner.run_preprocessing(context, [record], detector=_detector(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: _CountingRepository())
    assert type(excinfo.value) is RuntimeError and not isinstance(excinfo.value, HashComputationError)
    assert not routed


def test_only_hash_failed_code_is_introduced(tmp_path, monkeypatch):
    context, record = _context_record(tmp_path)
    monkeypatch.setattr(m2_runner, "hash_crop_artifact", _failing_hash())
    result = m2_runner.run_preprocessing(context, [record], detector=_detector(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: ManifestRepository(context.manifests_root, _metadata(context)))
    assert set(result.failures_by_code) == {"hash_failed"}
    assert _artifacts(context) == [] and not list(context.crops_root.rglob("*.tmp*"))
