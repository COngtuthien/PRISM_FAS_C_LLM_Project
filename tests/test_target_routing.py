import json

import pyarrow.parquet as pq

from prism_fas.data.manifests.leakage import FORBIDDEN, find_target_leakage
from prism_fas.data.manifests.repository import ManifestRepository
from prism_fas.data.manifests.routing import TargetRoutingResult, route_target_success
from prism_fas.data.manifests.schemas import TargetCropRecord, TargetFrameRecord
from prism_fas.data.preprocess_m2 import sample_id as make_sample_id
from prism_fas.data.run_context import M2OutputLayout, PreprocessingRunContext
from prism_fas.data.schemas.records import CanonicalVideoRecord

# Private tokens deliberately injected into the canonical target record; none of
# them may reach a persisted target manifest row.
PRIVATE_TOKENS = ("spoof", "live", "replay", "screen_attack", "screen", "attack", "taxonomy", "subject_007", "subject", "private_session_3", "session", "identity", "genuine", "protocol_a")


def test_real_target_routing_persists_isolated_target_rows(tmp_path):
    layout = M2OutputLayout.from_root(tmp_path / "full_preprocessing")
    context = PreprocessingRunContext(project_root=tmp_path, work_root=tmp_path, run_profile="full_preprocessing", output_namespace="full_preprocessing", output_root=layout.output_root, crops_root=layout.crops_root, frames_root=layout.frames_root, manifests_root=layout.manifests_root, state_root=layout.state_root, reports_root=layout.reports_root, logs_root=layout.logs_root, run_id="target-routing", dataset="siw_mv2", dataset_role="target", preprocessing_version="m2-v1", preprocessing_config_hash="a" * 64, detector_model_path=tmp_path / "model.onnx", detector_model_sha256="b" * 64, detector_input_size=320, detector_threshold=.5, all_records=False, record_limit=1, sample_limit=None, resume=False, dry_run=False, partial_full_profile=False, command="test")
    raw = tmp_path / "raw" / "siw_mv2" / "subject_007" / "replay_screen_attack_spoof.mp4"
    record = CanonicalVideoRecord(dataset="siw_mv2", subject_id="subject_007", video_id="v0007", source_path=raw, official_split="target_test", label="spoof", capture_metadata={"attack_type": "replay", "taxonomy": "screen_attack", "session_id": "private_session_3"}, adapter_version="siw-v1", protocol_version="protocol_a", source_fingerprint="e" * 64, metadata_provenance="PRIVATE evaluator-only YAML path_pattern")

    sid = make_sample_id("siw_mv2", record.video_id, 7, record.adapter_version, "uniform-v1", context.preprocessing_version)
    assert sid == make_sample_id("siw_mv2", record.video_id, 7, record.adapter_version, "uniform-v1", context.preprocessing_version)
    crop_relative_path = f"crops/siw_mv2/{sid}.jpg"
    artifact = context.output_root / crop_relative_path
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"synthetic-crop-bytes")

    repository = ManifestRepository(context.manifests_root, {"manifest_schema_version": "m2f1a-v1", "preprocessing_version": context.preprocessing_version, "preprocessing_config_hash": context.preprocessing_config_hash, "detector_model_sha256": context.detector_model_sha256}).initialize()
    result = route_target_success(repository=repository, context=context, canonical_record=record, sample_id=sid, requested_frame_index=6, actual_frame_index=7, timestamp_ms=240.0, frame_width=1280, frame_height=720, decoder_backend="opencv-video", source_media_type="video_file", selected_frame_reference=f"siw_mv2/{record.video_id}#frame=7", bbox=(100.0, 120.0, 300.0, 340.0), landmarks=[(150.0, 180.0), (250.0, 180.0), (200.0, 240.0), (160.0, 290.0), (240.0, 290.0)], detection_score=.93, detected_face_count=1, crop_box=(50, 70, 350, 390), requested_crop_padding=.25, effective_crop_padding=.25, crop_width=224, crop_height=224, crop_relative_path=crop_relative_path, crop_sha256="f" * 64)

    # --- result contract ---
    assert isinstance(result, TargetRoutingResult) and result.routed is True
    assert result.sample_id == sid
    assert isinstance(result.frame_record, TargetFrameRecord) and isinstance(result.crop_record, TargetCropRecord)
    assert result.frame_record.source_media_type == result.crop_record.source_media_type == "video_file"
    assert (result.target_frame_count, result.target_crop_count) == (1, 1)

    repository.flush()
    repository.load()
    assert repository.counts() == {"source_frames": 0, "source_crops": 0, "target_frames": 1, "target_crops": 1, "preprocessing_failures": 0}
    frame = repository.rows["target_frames"][0]
    crop = repository.rows["target_crops"][0]

    # --- identity and linkage ---
    assert frame["sample_id"] == crop["sample_id"] == sid
    assert frame["dataset"] == crop["dataset"] == "siw_mv2"
    assert frame["video_id"] == crop["video_id"] == "v0007"
    assert frame["source_record_id"] == crop["source_record_id"] == "v0007"
    assert frame["requested_frame_index"] == crop["requested_frame_index"] == 6
    assert frame["actual_frame_index"] == crop["actual_frame_index"] == 7
    assert frame["timestamp_ms"] == crop["timestamp_ms"] == 240.0
    assert frame["preprocessing_config_hash"] == crop["preprocessing_config_hash"] == "a" * 64
    assert crop["detector_model_sha256"] == "b" * 64
    assert frame["preprocessing_version"] == crop["preprocessing_version"] == "m2-v1"

    # --- target frame fields ---
    assert frame["source_media_type"] == crop["source_media_type"] == "video_file"
    assert frame["source_relative_identifier"] == "siw_mv2/v0007"
    assert frame["selected_frame_reference"] == "siw_mv2/v0007#frame=7"
    assert (frame["frame_width"], frame["frame_height"]) == (1280, 720)
    assert frame["decoder_backend"] == "opencv-video" and frame["status"] == "success"
    assert frame["source_fingerprint"] == "e" * 64 and frame["adapter_version"] == "siw-v1"
    assert frame["materialized_frame_relative_path"] is None and frame["warning_codes"] == []

    # --- target crop fields ---
    assert (crop["bbox_x1"], crop["bbox_y1"], crop["bbox_x2"], crop["bbox_y2"]) == (100.0, 120.0, 300.0, 340.0)
    assert (crop["landmark_0_x"], crop["landmark_0_y"]) == (150.0, 180.0)
    assert (crop["landmark_4_x"], crop["landmark_4_y"]) == (240.0, 290.0)
    assert (crop["crop_x1"], crop["crop_y1"], crop["crop_x2"], crop["crop_y2"]) == (50, 70, 350, 390)
    assert (crop["crop_width"], crop["crop_height"]) == (224, 224)
    assert crop["detection_score"] == .93 and crop["detected_face_count"] == 1
    assert crop["crop_relative_path"] == crop_relative_path and crop["crop_sha256"] == "f" * 64
    assert (context.output_root / crop["crop_relative_path"]).is_file()

    # --- path safety ---
    for row in (frame, crop):
        for value in row.values():
            text = str(value)
            assert not text.startswith(("C:", "D:", "/")) and ":\\" not in text and ".." not in text
    assert str(raw) not in json.dumps(frame, default=str) + json.dumps(crop, default=str)
    assert "replay_screen_attack_spoof.mp4" not in json.dumps(frame, default=str) + json.dumps(crop, default=str)

    # --- privacy: field names and persisted values ---
    for row in (frame, crop):
        assert not (set(row) & FORBIDDEN)
        assert find_target_leakage(row) == []
        for name in ("label", "label_live_spoof", "is_live", "is_spoof", "attack", "attack_type", "attack_subtype", "spoof_medium", "taxonomy", "subject_id", "person_id", "identity", "session_id", "protocol_version", "official_split", "capture_metadata", "metadata_provenance", "source_path"):
            assert name not in row
        serialized = json.dumps(row, default=str).lower()
        for token in PRIVATE_TOKENS:
            assert token not in serialized, f"private token {token!r} leaked into {serialized}"

    # --- privacy: persisted parquet schema (names + key/value metadata) ---
    for name in ("target_frames", "target_crops"):
        schema = pq.read_schema(context.manifests_root / f"{name}.parquet")
        assert find_target_leakage(schema) == []
        assert not (set(schema.names) & FORBIDDEN)

    # --- no source, failure or m2a side effects ---
    assert repository.rows["source_frames"] == [] and repository.rows["source_crops"] == []
    assert repository.rows["preprocessing_failures"] == []
    assert not list(tmp_path.rglob("*.jsonl"))
    assert not any("m2a" in p.parts for p in tmp_path.rglob("*"))
