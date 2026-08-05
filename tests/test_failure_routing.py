import pytest
from unittest.mock import Mock

import prism_fas.data.manifests.routing as routing
from prism_fas.data.manifests.converters import ManifestConversionError, build_preprocessing_failure_record
from prism_fas.data.manifests.repository import ManifestRepository
from prism_fas.data.manifests.routing import FailureRoutingConsistencyError, route_preprocessing_failure
from prism_fas.data.run_context import M2OutputLayout, PreprocessingRunContext
from prism_fas.data.schemas.records import CanonicalVideoRecord


def _context_and_record(tmp_path):
    layout = M2OutputLayout.from_root(tmp_path / "full_preprocessing")
    context = PreprocessingRunContext(project_root=tmp_path, work_root=tmp_path, run_profile="full_preprocessing", output_namespace="full_preprocessing", output_root=layout.output_root, crops_root=layout.crops_root, frames_root=layout.frames_root, manifests_root=layout.manifests_root, state_root=layout.state_root, reports_root=layout.reports_root, logs_root=layout.logs_root, run_id="failure-routing", dataset="casia_fasd", dataset_role="source", preprocessing_version="m2-v1", preprocessing_config_hash="a" * 64, detector_model_path=tmp_path / "model.onnx", detector_model_sha256="b" * 64, detector_input_size=320, detector_threshold=.5, all_records=False, record_limit=1, sample_limit=1, resume=False, dry_run=False, partial_full_profile=False, command="test")
    record = CanonicalVideoRecord(dataset="casia_fasd", subject_id="subject-1", video_id="s001v001", source_path=tmp_path / "s001v001f001.png", official_split="train", label="live", adapter_version="casia-v1", source_fingerprint="c" * 64, metadata_provenance="synthetic-test")
    return context, record


def _route_kwargs(context, record, **updates):
    values = dict(repository=Mock(), context=context, canonical_record=record, sample_id="failure-sample-1", requested_frame_index=0, actual_frame_index=17, stage="detector", error_code="no_face", error_message="synthetic no face", backend="synthetic-reader", recoverable=True, warning_codes=("synthetic",))
    values.update(updates)
    return values


def _repository(context):
    return ManifestRepository(context.manifests_root, {"manifest_schema_version": "m2f1a-v1", "preprocessing_version": context.preprocessing_version, "preprocessing_config_hash": context.preprocessing_config_hash, "detector_model_sha256": context.detector_model_sha256, "detector_input_size": str(context.detector_input_size), "detector_threshold": str(context.detector_threshold)}).initialize()


def test_route_preprocessing_failure_persists_one_real_parquet_row(tmp_path):
    context, record = _context_and_record(tmp_path)
    repository = _repository(context)

    result = route_preprocessing_failure(repository=repository, context=context, canonical_record=record, sample_id="failure-sample-1", requested_frame_index=0, actual_frame_index=17, stage="detector", error_code="no_face", error_message="synthetic no face\n", backend="synthetic-reader", recoverable=True)
    repository.flush()
    repository.load()

    assert result.sample_id == "failure-sample-1" and result.failure_count == 1 and result.routed
    assert repository.counts() == {"source_frames": 0, "source_crops": 0, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 1}
    row = repository.rows["preprocessing_failures"][0]
    assert row["sample_id"] == "failure-sample-1"
    assert row["dataset"] == "casia_fasd" and row["source_record_id"] == "s001v001"
    assert row["error_code"] == "no_face" and row["stage"] == "detector" and row["recoverable"] is True
    assert row["sanitized_error_message"] == "synthetic no face "
    assert row["preprocessing_config_hash"] == context.preprocessing_config_hash
    assert row["detector_model_sha256"] == context.detector_model_sha256


def test_failure_converter_is_called_before_upsert_with_all_routing_inputs(tmp_path, monkeypatch):
    context, record = _context_and_record(tmp_path)
    failure = build_preprocessing_failure_record(context, record, sample_id="failure-sample-1", requested_frame_index=0, actual_frame_index=17, stage="detector", error_code="no_face", message="synthetic no face", backend="synthetic-reader", recoverable=True, warning_codes=["synthetic"])
    events, repository = [], Mock()
    repository.counts.return_value = {"preprocessing_failures": 1}

    def converter(actual_context, actual_record, **kwargs):
        events.append("converter")
        assert actual_context is context and actual_record is record
        assert kwargs == {"sample_id": "failure-sample-1", "requested_frame_index": 0, "actual_frame_index": 17, "stage": "detector", "error_code": "no_face", "message": "synthetic no face", "backend": "synthetic-reader", "recoverable": True, "warning_codes": ["synthetic"]}
        return failure

    monkeypatch.setattr(routing, "build_preprocessing_failure_record", converter)
    result = route_preprocessing_failure(**_route_kwargs(context, record, repository=repository))
    repository.upsert_failure.assert_called_once_with(failure.model_dump(mode="json"))
    assert events == ["converter"] and result.routed and result.failure_count == 1


def test_failure_converter_exception_does_not_mutate_repository(tmp_path, monkeypatch):
    context, record = _context_and_record(tmp_path)
    repository = Mock()
    monkeypatch.setattr(routing, "build_preprocessing_failure_record", Mock(side_effect=ManifestConversionError("synthetic failure conversion error")))
    with pytest.raises(ManifestConversionError, match="^synthetic failure conversion error$"):
        route_preprocessing_failure(**_route_kwargs(context, record, repository=repository))
    repository.upsert_failure.assert_not_called()


@pytest.mark.parametrize(("field", "value"), [("sample_id", "different-sample"), ("dataset", "other_dataset"), ("source_record_id", "different-source")])
def test_failure_identity_mismatches_reject_before_repository_mutation(tmp_path, monkeypatch, field, value):
    context, record = _context_and_record(tmp_path)
    valid = build_preprocessing_failure_record(context, record, sample_id="failure-sample-1", requested_frame_index=0, actual_frame_index=17, stage="detector", error_code="no_face", message="synthetic no face", backend="synthetic-reader", recoverable=True)
    repository = Mock()
    monkeypatch.setattr(routing, "build_preprocessing_failure_record", Mock(return_value=valid.model_copy(update={field: value})))
    with pytest.raises(FailureRoutingConsistencyError, match=field):
        route_preprocessing_failure(**_route_kwargs(context, record, repository=repository))
    repository.upsert_failure.assert_not_called()


def test_failure_upsert_exception_propagates_without_routed_result(tmp_path, monkeypatch):
    context, record = _context_and_record(tmp_path)
    valid = build_preprocessing_failure_record(context, record, sample_id="failure-sample-1", requested_frame_index=0, actual_frame_index=17, stage="detector", error_code="no_face", message="synthetic no face", backend="synthetic-reader", recoverable=True)
    repository = Mock()
    repository.upsert_failure.side_effect = RuntimeError("synthetic failure upsert error")
    converter = Mock(return_value=valid)
    monkeypatch.setattr(routing, "build_preprocessing_failure_record", converter)
    with pytest.raises(RuntimeError, match="^synthetic failure upsert error$"):
        route_preprocessing_failure(**_route_kwargs(context, record, repository=repository))
    converter.assert_called_once()
    repository.upsert_failure.assert_called_once_with(valid.model_dump(mode="json"))


def test_exact_failure_duplicate_is_idempotent_in_real_parquet(tmp_path):
    context, record = _context_and_record(tmp_path)
    repository = _repository(context)
    first = route_preprocessing_failure(**_route_kwargs(context, record, repository=repository))
    repository.flush(); repository.load()
    second = route_preprocessing_failure(**_route_kwargs(context, record, repository=repository))
    repository.flush(); repository.load()
    assert first.routed and second.routed and second.failure_count == 1
    assert repository.counts() == {"source_frames": 0, "source_crops": 0, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 1}


def test_conflicting_failure_duplicate_is_rejected_without_overwriting_parquet_row(tmp_path):
    context, record = _context_and_record(tmp_path)
    repository = _repository(context)
    route_preprocessing_failure(**_route_kwargs(context, record, repository=repository))
    repository.flush(); repository.load()
    with pytest.raises(ValueError, match="conflicting duplicate sample_id"):
        route_preprocessing_failure(**_route_kwargs(context, record, repository=repository, error_message="changed immutable payload"))
    repository.flush(); repository.load()
    assert repository.counts()["preprocessing_failures"] == 1
    assert repository.rows["preprocessing_failures"][0]["sanitized_error_message"] == "synthetic no face"


def test_routed_failure_persists_sanitized_message(tmp_path):
    context, record = _context_and_record(tmp_path)
    repository = _repository(context)
    route_preprocessing_failure(**_route_kwargs(context, record, repository=repository, error_message="detector failed\nC:\\model-cache\\attack_private_metadata=target_secret"))
    repository.flush(); repository.load()
    message = repository.rows["preprocessing_failures"][0]["sanitized_error_message"]
    assert "\n" not in message and "\r" not in message
    assert "C:\\" not in message and "model-cache" not in message
    assert "attack_private_metadata" not in message and "target_secret" not in message
    assert "detector failed" in message and "[redacted-path]" in message
