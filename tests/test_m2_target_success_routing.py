import json

import numpy as np
import pytest

import prism_fas.data.m2_runner as m2_runner
from prism_fas.data.manifests.leakage import FORBIDDEN, find_target_leakage
from prism_fas.data.manifests.repository import ManifestRepository
from prism_fas.data.media.readers import FrameResult
from prism_fas.data.preprocess_m2 import Detection
from prism_fas.data.run_context import M2OutputLayout, PreprocessingRunContext
from prism_fas.data.schemas.records import CanonicalVideoRecord
from prism_fas.utils.core import sha256_file

PRIVATE_TOKENS = ("spoof", "live", "replay", "screen_attack", "attack", "taxonomy", "subject_007", "subject", "private_session_3", "session")


def _context(tmp_path, dataset="siw_mv2", role="target"):
    layout = M2OutputLayout.from_root(tmp_path / "full_preprocessing")
    return PreprocessingRunContext(project_root=tmp_path, work_root=tmp_path, run_profile="full_preprocessing", output_namespace="full_preprocessing", output_root=layout.output_root, crops_root=layout.crops_root, frames_root=layout.frames_root, manifests_root=layout.manifests_root, state_root=layout.state_root, reports_root=layout.reports_root, logs_root=layout.logs_root, run_id="target-success", dataset=dataset, dataset_role=role, preprocessing_version="m2-v1", preprocessing_config_hash="a" * 64, detector_model_path=tmp_path / "model.onnx", detector_model_sha256="b" * 64, detector_input_size=320, detector_threshold=.5, all_records=False, record_limit=1, sample_limit=None, resume=False, dry_run=False, partial_full_profile=False, command="test")


def _target_record(tmp_path):
    # Deliberately private canonical metadata: none of it may reach the manifests.
    return CanonicalVideoRecord(dataset="siw_mv2", subject_id="subject_007", video_id="v0007", source_path=tmp_path / "raw" / "siw_mv2" / "subject_007" / "replay_screen_attack_spoof.mp4", official_split="target_test", label="spoof", capture_metadata={"attack_type": "replay", "taxonomy": "screen_attack", "session_id": "private_session_3"}, adapter_version="siw-v1", source_fingerprint="e" * 64, metadata_provenance="PRIVATE evaluator-only YAML path_pattern")


def _source_record(tmp_path):
    return CanonicalVideoRecord(dataset="casia_fasd", subject_id="subject-1", video_id="s001v001", source_path=tmp_path / "s001v001f001.png", official_split="train", label="live", adapter_version="casia-v1", source_fingerprint="c" * 64, metadata_provenance="synthetic-test")


def _metadata(context):
    return {"manifest_schema_version": "m2f1a-v1", "preprocessing_version": context.preprocessing_version, "preprocessing_config_hash": context.preprocessing_config_hash, "detector_model_sha256": context.detector_model_sha256}


def _reader(count=1):
    class Reader:
        def frame_count(self): return count
        def read_frame(self, index): return FrameResult(index, index + 3, 40.0 * index, 64, 64, np.zeros((64, 64, 3), dtype=np.uint8), "opencv")
        def close(self): pass
    return Reader()


def _detector():
    return type("Detector", (), {"detect": lambda self, image: [Detection(bbox=(12, 12, 52, 52), score=.91, landmarks=[(20, 22), (30, 22), (25, 30), (20, 38), (30, 38)])]})()


def _run(context, records, repository, frames=1):
    return m2_runner.run_preprocessing(context, records, detector=_detector(), media_reader_factory=lambda _: _reader(frames), repository_factory=lambda *_: repository)


class _SpyRepository:
    def initialize(self): return self
    def flush(self): pass
    def counts(self): return {"source_frames": 0, "source_crops": 0, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 0}


def test_target_success_routes_once_with_privacy_safe_arguments(tmp_path, monkeypatch):
    context, record, targets, sources = _context(tmp_path), _target_record(tmp_path), [], []
    repository = _SpyRepository()
    monkeypatch.setattr(m2_runner, "route_target_success", lambda **kwargs: targets.append(kwargs))
    monkeypatch.setattr(m2_runner, "route_source_success", lambda **kwargs: sources.append(kwargs))
    result = _run(context, [record], repository)

    assert len(targets) == 1 and sources == []
    call = targets[0]
    assert call["repository"] is repository and call["context"] is context and call["canonical_record"] is record
    assert call["requested_frame_index"] == 0 and call["actual_frame_index"] == 3
    assert call["source_media_type"] == "video_file"
    assert call["frame_width"] == 64 and call["frame_height"] == 64 and call["decoder_backend"] == "opencv"
    assert call["timestamp_ms"] == 0.0 and call["detected_face_count"] == 1 and call["detection_score"] == .91
    assert len(call["landmarks"]) == 5 and all(np.isfinite(v) for point in call["landmarks"] for v in point)
    assert call["crop_sha256"] == sha256_file(context.output_root / call["crop_relative_path"])
    assert call["sample_id"] and call["selected_frame_reference"] == f"target/{call['sample_id']}#frame=3"

    # the call site must not hand any private metadata to the target helper
    payload = json.dumps({k: str(v) for k, v in call.items() if k != "canonical_record"}).lower()
    for token in PRIVATE_TOKENS:
        assert token not in payload, f"private token {token!r} reached the target helper"
    assert str(tmp_path).lower() not in payload and "replay_screen_attack_spoof.mp4" not in payload
    assert not (set(call) & {"source_path", "label", "subject_id", "capture_metadata", "metadata_provenance"})

    assert (result.samples_selected, result.samples_successful, result.samples_failed) == (1, 1, 0)
    assert (result.frames_read, result.detector_calls, result.crops_written) == (1, 1, 1)
    assert result.failures_by_code == {}


def test_real_repository_persists_target_rows_and_no_source_rows(tmp_path):
    context, record = _context(tmp_path), _target_record(tmp_path)
    repository = ManifestRepository(context.manifests_root, _metadata(context))
    result = _run(context, [record], repository)

    assert result.manifest_counts == {"source_frames": 0, "source_crops": 0, "target_frames": 1, "target_crops": 1, "preprocessing_failures": 0}
    repository.load()
    frame, crop = repository.rows["target_frames"][0], repository.rows["target_crops"][0]
    assert repository.rows["source_frames"] == [] and repository.rows["source_crops"] == [] and repository.rows["preprocessing_failures"] == []
    assert frame["sample_id"] == crop["sample_id"] and frame["dataset"] == crop["dataset"] == "siw_mv2"
    assert frame["requested_frame_index"] == crop["requested_frame_index"] == 0
    assert frame["actual_frame_index"] == crop["actual_frame_index"] == 3
    assert frame["source_media_type"] == crop["source_media_type"] == "video_file"
    assert frame["source_relative_identifier"] == f"target/{frame['sample_id']}"

    artifact = context.output_root / crop["crop_relative_path"]
    assert artifact.is_file() and crop["crop_sha256"] == sha256_file(artifact)
    assert artifact.name == f"{crop['sample_id']}.jpg" and not list(context.crops_root.rglob("*.tmp*"))

    for row in (frame, crop):
        assert not (set(row) & FORBIDDEN) and find_target_leakage(row) == []
        serialized = json.dumps(row, default=str).lower()
        for token in PRIVATE_TOKENS:
            assert token not in serialized
        assert "replay_screen_attack_spoof.mp4" not in serialized and str(tmp_path).lower() not in serialized


def test_two_target_samples_produce_two_manifest_pairs(tmp_path):
    context, record = _context(tmp_path), _target_record(tmp_path)
    repository = ManifestRepository(context.manifests_root, _metadata(context))
    result = _run(context, [record], repository, frames=2)

    assert (result.samples_selected, result.samples_successful, result.samples_failed) == (2, 2, 0)
    assert (result.frames_read, result.detector_calls, result.crops_written) == (2, 2, 2)
    assert result.manifest_counts == {"source_frames": 0, "source_crops": 0, "target_frames": 2, "target_crops": 2, "preprocessing_failures": 0}
    repository.load()
    artifacts = sorted(p.name for p in context.crops_root.rglob("*.jpg"))
    assert len(artifacts) == 2 and artifacts == sorted(f"{row['sample_id']}.jpg" for row in repository.rows["target_crops"])


def test_target_routing_exception_propagates_without_counters_or_next_sample(tmp_path, monkeypatch):
    context, record, reads = _context(tmp_path), _target_record(tmp_path), []
    class Repository:
        def initialize(self): return self
        def flush(self): raise AssertionError("must not flush after routing exception")
        def counts(self): raise AssertionError("must not count after routing exception")
    class Reader:
        def frame_count(self): return 2
        def read_frame(self, index):
            reads.append(index)
            return FrameResult(index, index, None, 64, 64, np.zeros((64, 64, 3), dtype=np.uint8), "opencv")
        def close(self): raise AssertionError("must not close after routing exception")
    monkeypatch.setattr(m2_runner, "route_target_success", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic target routing error")))
    with pytest.raises(RuntimeError, match="^synthetic target routing error$"):
        m2_runner.run_preprocessing(context, [record], detector=_detector(), media_reader_factory=lambda _: Reader(), repository_factory=lambda *_: Repository())
    assert reads == [0]


def test_missing_repository_blocks_target_routing(tmp_path, monkeypatch):
    context, record, targets = _context(tmp_path), _target_record(tmp_path), []
    class MissingRepository:
        def initialize(self): return None
    monkeypatch.setattr(m2_runner, "route_target_success", lambda **kwargs: targets.append(kwargs))
    with pytest.raises(RuntimeError, match="^target routing requires an initialized manifest repository$"):
        m2_runner.run_preprocessing(context, [record], detector=_detector(), media_reader_factory=lambda _: _reader(), repository_factory=lambda *_: MissingRepository())
    assert not targets


def test_source_role_regression_still_routes_source_only(tmp_path, monkeypatch):
    context, record, targets = _context(tmp_path, dataset="casia_fasd", role="source"), _source_record(tmp_path), []
    repository = ManifestRepository(context.manifests_root, _metadata(context))
    monkeypatch.setattr(m2_runner, "route_target_success", lambda **kwargs: targets.append(kwargs))
    result = _run(context, [record], repository)

    assert targets == []
    assert (result.samples_successful, result.crops_written) == (1, 1)
    assert result.manifest_counts == {"source_frames": 1, "source_crops": 1, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 0}
    repository.load()
    assert repository.rows["source_frames"][0]["source_media_type"] == "image_sequence"
    assert repository.rows["target_frames"] == [] and repository.rows["target_crops"] == []
