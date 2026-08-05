import os

import pyarrow.parquet as pq
import pytest

import prism_fas.data.manifests.parquet_writer as writer
from prism_fas.data.manifests.parquet_writer import write_parquet_atomic
from prism_fas.data.manifests.schemas import PreprocessingFailureRecord


def _row(sample_id="s1"):
    return {"dataset": "casia_fasd", "source_record_id": "train_s1v1", "source_relative_identifier": "source", "sample_id": sample_id, "requested_frame_index": 0, "actual_frame_index": 0, "stage": "detector", "error_code": "no_face", "sanitized_error_message": "no valid face detected", "timestamp": "", "preprocessing_config_hash": "a" * 64, "detector_model_sha256": "b" * 64, "backend": "detector", "recoverable": True, "warning_codes": []}


def test_transient_sharing_violation_is_retried_not_propagated(tmp_path, monkeypatch):
    """Regression: a lingering reader handle on Windows made os.replace raise
    PermissionError mid-run and aborted the whole dataset."""
    path = tmp_path / "preprocessing_failures.parquet"
    real_replace, attempts = os.replace, []

    def flaky_replace(src, dst):
        attempts.append(dst)
        if len(attempts) <= 3: raise PermissionError(5, "Access is denied")
        real_replace(src, dst)
    monkeypatch.setattr(writer.os, "replace", flaky_replace)
    monkeypatch.setattr(writer.time, "sleep", lambda _: None)

    summary = write_parquet_atomic(path, [_row()], PreprocessingFailureRecord, {"manifest_schema_version": "m2f1a-v1"}, True)
    assert len(attempts) == 4 and summary["rows"] == 1
    assert pq.read_table(path).num_rows == 1


def test_persistent_permission_error_still_fails(tmp_path, monkeypatch):
    path = tmp_path / "preprocessing_failures.parquet"
    monkeypatch.setattr(writer.os, "replace", lambda src, dst: (_ for _ in ()).throw(PermissionError(5, "Access is denied")))
    monkeypatch.setattr(writer.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError):
        write_parquet_atomic(path, [_row()], PreprocessingFailureRecord, {"manifest_schema_version": "m2f1a-v1"}, True)
    assert not list(tmp_path.glob("*.tmp"))
