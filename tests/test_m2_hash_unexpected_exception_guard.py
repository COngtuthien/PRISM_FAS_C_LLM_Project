from pathlib import Path

import numpy as np
import pytest

import prism_fas.data.m2_runner as m2_runner
from prism_fas.data.manifests.repository import ManifestRepository
from prism_fas.data.media.readers import FrameResult
from prism_fas.data.output import HashComputationError
from prism_fas.data.run_context import M2OutputLayout, PreprocessingRunContext
from prism_fas.data.preprocess_m2 import Detection
from prism_fas.data.schemas.records import CanonicalVideoRecord
from prism_fas.utils.core import sha256_file


class UnexpectedHashError(Exception):
    """Synthetic unexpected failure raised from the hash boundary."""


def _context_record(tmp_path):
    layout = M2OutputLayout.from_root(tmp_path / "full_preprocessing")
    context = PreprocessingRunContext(project_root=tmp_path, work_root=tmp_path, run_profile="full_preprocessing", output_namespace="full_preprocessing", output_root=layout.output_root, crops_root=layout.crops_root, frames_root=layout.frames_root, manifests_root=layout.manifests_root, state_root=layout.state_root, reports_root=layout.reports_root, logs_root=layout.logs_root, run_id="hash-guard", dataset="casia_fasd", dataset_role="source", preprocessing_version="m2-v1", preprocessing_config_hash="a" * 64, detector_model_path=tmp_path / "model.onnx", detector_model_sha256="b" * 64, detector_input_size=320, detector_threshold=.5, all_records=False, record_limit=1, sample_limit=None, resume=False, dry_run=False, partial_full_profile=False, command="test")
    record = CanonicalVideoRecord(dataset="casia_fasd", subject_id="subject-1", video_id="s001v001", source_path=tmp_path / "s001v001f001.png", official_split="train", label="live", adapter_version="casia-v1", source_fingerprint="c" * 64, metadata_provenance="synthetic-test")
    return context, record


def _metadata(context):
    return {"manifest_schema_version": "m2f1a-v1", "preprocessing_version": context.preprocessing_version, "preprocessing_config_hash": context.preprocessing_config_hash, "detector_model_sha256": context.detector_model_sha256, "detector_input_size": str(context.detector_input_size), "detector_threshold": str(context.detector_threshold)}


class _SpyReader:
    def __init__(self, count=1): self.count = count; self.reads = []; self.closed = 0
    def frame_count(self): return self.count
    def read_frame(self, index):
        self.reads.append(index)
        return FrameResult(index, index + 3, None, 64, 64, np.zeros((64, 64, 3), dtype=np.uint8), "synthetic-reader")
    def close(self): self.closed += 1


class _SpyDetector:
    def __init__(self): self.calls = []
    def detect(self, image):
        self.calls.append(image.shape)
        return [Detection(bbox=(12, 12, 52, 52), score=.91, landmarks=[(20, 22)] * 5)]


class _SpyRepository:
    def __init__(self, failures=0):
        self.failures = failures; self.events = []
    def initialize(self): self.events.append("initialize"); return self
    def load(self): self.events.append("load"); return self
    def upsert_source_success(self, frame, crop): self.events.append("upsert_source_success")
    def upsert_failure(self, row): self.events.append("upsert_failure")
    def flush(self): self.events.append("flush")
    def counts(self):
        self.events.append("counts")
        return {"source_frames": 0, "source_crops": 0, "target_frames": 0, "target_crops": 0, "preprocessing_failures": self.failures}


def _raise(error):
    def hasher(path): raise error
    return hasher


def _artifacts(context):
    return sorted(p.name for p in context.crops_root.rglob("*") if p.is_file())


def _run(context, records, *, repository, detector=None, reader=None, monkeypatch=None, hasher=None, routed=None, successes=None):
    if hasher is not None: monkeypatch.setattr(m2_runner, "hash_crop_artifact", hasher)
    if routed is not None: monkeypatch.setattr(m2_runner, "route_preprocessing_failure", lambda **kwargs: routed.append(kwargs))
    if successes is not None: monkeypatch.setattr(m2_runner, "route_source_success", lambda **kwargs: successes.append(kwargs))
    return m2_runner.run_preprocessing(context, records, detector=detector or _SpyDetector(), media_reader_factory=lambda _: reader or _SpyReader(), repository_factory=lambda *_: repository)


# --- exception identity ----------------------------------------------------

@pytest.mark.parametrize("error", [RuntimeError("synthetic unexpected hash runtime error"), ValueError("synthetic unexpected hash value error"), AssertionError("synthetic unexpected hash assertion error"), UnexpectedHashError("synthetic unexpected hash custom error")])
def test_unexpected_hash_exception_propagates_with_exact_identity(tmp_path, monkeypatch, error):
    context, record, routed, successes = *_context_record(tmp_path), [], []
    repository = _SpyRepository()
    with pytest.raises(type(error)) as excinfo:
        _run(context, [record], repository=repository, monkeypatch=monkeypatch, hasher=_raise(error), routed=routed, successes=successes)
    assert type(excinfo.value) is type(error)
    assert str(excinfo.value) == str(error)
    assert not isinstance(excinfo.value, HashComputationError)
    assert excinfo.value.__cause__ is None and excinfo.value.__context__ is None
    assert routed == [] and successes == []


def test_unexpected_hash_exception_is_never_classified_as_a_failure_code(tmp_path, monkeypatch):
    context, record, routed, successes = *_context_record(tmp_path), [], []
    repository = _SpyRepository()
    with pytest.raises(RuntimeError, match="^synthetic unexpected hash runtime error$"):
        _run(context, [record], repository=repository, monkeypatch=monkeypatch, hasher=_raise(RuntimeError("synthetic unexpected hash runtime error")), routed=routed, successes=successes)
    # No routing, no counter finalization: the outer handler must not have
    # swallowed the exception into unrouted_processing_failure.
    assert routed == [] and successes == []
    assert repository.events == ["initialize", "load"]
    assert "counts" not in repository.events and "flush" not in repository.events
    assert "upsert_failure" not in repository.events and "upsert_source_success" not in repository.events


def test_unexpected_hash_exception_writes_no_manifest_row(tmp_path, monkeypatch):
    context, record = _context_record(tmp_path)
    repository = ManifestRepository(context.manifests_root, _metadata(context))
    monkeypatch.setattr(m2_runner, "hash_crop_artifact", _raise(RuntimeError("synthetic unexpected hash runtime error")))
    with pytest.raises(RuntimeError, match="^synthetic unexpected hash runtime error$"):
        m2_runner.run_preprocessing(context, [record], detector=_SpyDetector(), media_reader_factory=lambda _: _SpyReader(), repository_factory=lambda *_: repository)
    repository.load()
    assert repository.counts() == {"source_frames": 0, "source_crops": 0, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 0}
    assert not any(repository.rows[name] for name in repository.rows)


# --- cleanup ---------------------------------------------------------------

def test_unexpected_hash_exception_cleans_up_orphan_crop_artifact(tmp_path, monkeypatch):
    context, record = _context_record(tmp_path)
    with pytest.raises(UnexpectedHashError):
        _run(context, [record], repository=_SpyRepository(), monkeypatch=monkeypatch, hasher=_raise(UnexpectedHashError("synthetic unexpected hash custom error")))
    assert _artifacts(context) == []
    assert not list(context.crops_root.rglob("*.tmp*"))


def test_cleanup_failure_does_not_mask_unexpected_hash_exception(tmp_path, monkeypatch):
    context, record = _context_record(tmp_path)
    monkeypatch.setattr(Path, "unlink", lambda self, **kwargs: (_ for _ in ()).throw(PermissionError("synthetic cleanup failure")))
    with pytest.raises(RuntimeError, match="^synthetic unexpected hash runtime error$") as excinfo:
        _run(context, [record], repository=_SpyRepository(), monkeypatch=monkeypatch, hasher=_raise(RuntimeError("synthetic unexpected hash runtime error")))
    assert type(excinfo.value) is RuntimeError and not isinstance(excinfo.value, PermissionError)


def test_unexpected_hash_exception_does_not_touch_other_sample_artifacts(tmp_path, monkeypatch):
    context, record = _context_record(tmp_path)
    original, calls = m2_runner.hash_crop_artifact, []
    def hash_then_fail(path):
        calls.append(Path(path))
        if len(calls) == 1: return original(path)
        raise RuntimeError("synthetic unexpected hash runtime error")
    monkeypatch.setattr(m2_runner, "hash_crop_artifact", hash_then_fail)
    repository = ManifestRepository(context.manifests_root, _metadata(context))
    with pytest.raises(RuntimeError, match="^synthetic unexpected hash runtime error$"):
        m2_runner.run_preprocessing(context, [record], detector=_SpyDetector(), media_reader_factory=lambda _: _SpyReader(2), repository_factory=lambda *_: repository)
    assert _artifacts(context) == [calls[0].name]
    assert context.crops_root.is_dir() and calls[1].name not in _artifacts(context)


# --- no continuation -------------------------------------------------------

def test_next_sample_is_not_processed_after_unexpected_hash_exception(tmp_path, monkeypatch):
    context, record, routed, successes = *_context_record(tmp_path), [], []
    reader, detector, repository = _SpyReader(2), _SpyDetector(), _SpyRepository()
    with pytest.raises(RuntimeError, match="^synthetic unexpected hash runtime error$"):
        _run(context, [record], repository=repository, detector=detector, reader=reader, monkeypatch=monkeypatch, hasher=_raise(RuntimeError("synthetic unexpected hash runtime error")), routed=routed, successes=successes)
    assert reader.reads == [0] and len(detector.calls) == 1 and reader.closed == 0
    assert routed == [] and successes == []
    assert _artifacts(context) == []


def test_second_record_is_not_attempted_after_unexpected_hash_exception(tmp_path, monkeypatch):
    context, record = _context_record(tmp_path)
    readers = []
    def factory(_):
        readers.append(_SpyReader())
        return readers[-1]
    monkeypatch.setattr(m2_runner, "hash_crop_artifact", _raise(RuntimeError("synthetic unexpected hash runtime error")))
    with pytest.raises(RuntimeError, match="^synthetic unexpected hash runtime error$"):
        m2_runner.run_preprocessing(context, [record, record], detector=_SpyDetector(), media_reader_factory=factory, repository_factory=lambda *_: _SpyRepository())
    assert len(readers) == 1


def test_missing_repository_does_not_change_unexpected_exception_classification(tmp_path, monkeypatch):
    context, record = _context_record(tmp_path)
    class MissingRepository:
        def initialize(self): return None
    monkeypatch.setattr(m2_runner, "hash_crop_artifact", _raise(RuntimeError("synthetic unexpected hash runtime error")))
    with pytest.raises(RuntimeError, match="^synthetic unexpected hash runtime error$"):
        m2_runner.run_preprocessing(context, [record], detector=_SpyDetector(), media_reader_factory=lambda _: _SpyReader(), repository_factory=lambda *_: MissingRepository())
    monkeypatch.setattr(m2_runner, "hash_crop_artifact", _raise(HashComputationError()))
    with pytest.raises(RuntimeError, match="^failure routing requires an initialized manifest repository$"):
        m2_runner.run_preprocessing(context, [record], detector=_SpyDetector(), media_reader_factory=lambda _: _SpyReader(), repository_factory=lambda *_: MissingRepository())
    assert _artifacts(context) == []


# --- unchanged behaviors ---------------------------------------------------

def test_typed_hash_error_still_routes_hash_failed_and_continues(tmp_path, monkeypatch):
    context, record = _context_record(tmp_path)
    repository = ManifestRepository(context.manifests_root, _metadata(context))
    original, calls = m2_runner.hash_crop_artifact, []
    def hash_once(path):
        calls.append(Path(path))
        if len(calls) == 1: raise HashComputationError()
        return original(path)
    monkeypatch.setattr(m2_runner, "hash_crop_artifact", hash_once)
    result = m2_runner.run_preprocessing(context, [record], detector=_SpyDetector(), media_reader_factory=lambda _: _SpyReader(2), repository_factory=lambda *_: repository)
    assert (result.samples_selected, result.samples_successful, result.samples_failed, result.crops_written) == (2, 1, 1, 1)
    assert result.failures_by_code == {"hash_failed": 1}
    assert result.manifest_counts == {"source_frames": 1, "source_crops": 1, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 1}
    assert _artifacts(context) == [calls[1].name]


def test_valid_hash_still_routes_source_success(tmp_path):
    context, record = _context_record(tmp_path)
    repository = ManifestRepository(context.manifests_root, _metadata(context))
    result = m2_runner.run_preprocessing(context, [record], detector=_SpyDetector(), media_reader_factory=lambda _: _SpyReader(), repository_factory=lambda *_: repository)
    assert (result.samples_successful, result.samples_failed, result.crops_written) == (1, 0, 1)
    assert result.failures_by_code == {}
    assert result.manifest_counts == {"source_frames": 1, "source_crops": 1, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 0}
    repository.load()
    artifact = next(p for p in context.crops_root.rglob("*.jpg"))
    digest = repository.rows["source_crops"][0]["crop_sha256"]
    assert len(digest) == 64 and digest == sha256_file(artifact)


def test_source_routing_and_repository_errors_keep_current_contract(tmp_path, monkeypatch):
    context, record, routed = *_context_record(tmp_path), []
    monkeypatch.setattr(m2_runner, "route_preprocessing_failure", lambda **kwargs: routed.append(kwargs))
    monkeypatch.setattr(m2_runner, "route_source_success", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic source routing error")))
    with pytest.raises(RuntimeError, match="^synthetic source routing error$"):
        m2_runner.run_preprocessing(context, [record], detector=_SpyDetector(), media_reader_factory=lambda _: _SpyReader(), repository_factory=lambda *_: _SpyRepository())
    monkeypatch.undo()
    monkeypatch.setattr(m2_runner, "route_preprocessing_failure", lambda **kwargs: routed.append(kwargs))
    class Repository(_SpyRepository):
        def counts(self): raise RuntimeError("synthetic repository counts failure")
    with pytest.raises(RuntimeError, match="^synthetic repository counts failure$"):
        m2_runner.run_preprocessing(context, [record], detector=_SpyDetector(), media_reader_factory=lambda _: _SpyReader(), repository_factory=lambda *_: Repository())
    assert routed == []


def test_unrelated_stage_errors_still_use_the_outer_unrouted_contract(tmp_path, monkeypatch):
    context, record = _context_record(tmp_path)
    monkeypatch.setattr(m2_runner, "crop_face", lambda *args: (_ for _ in ()).throw(ValueError("synthetic unrelated crop failure")))
    result = m2_runner.run_preprocessing(context, [record], detector=_SpyDetector(), media_reader_factory=lambda _: _SpyReader(), repository_factory=lambda *_: _SpyRepository())
    assert result.failures_by_code == {"unrouted_processing_failure": 1}
    assert (result.samples_failed, result.samples_successful) == (1, 0)
