from __future__ import annotations

from math import isfinite
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

class ManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

class _FrameBase(ManifestModel):
    sample_id: str; dataset: str; video_id: str; source_record_id: str
    source_media_type: str; source_relative_identifier: str
    requested_frame_index: int; actual_frame_index: int; timestamp_ms: float | None
    frame_width: int; frame_height: int; selected_frame_reference: str
    materialized_frame_relative_path: str | None; source_fingerprint: str
    frame_fingerprint: str | None; decoder_backend: str; adapter_version: str
    sampling_version: str; preprocessing_version: str; preprocessing_config_hash: str
    status: str; warning_codes: list[str] = Field(default_factory=list)

class SourceFrameRecord(_FrameBase):
    subject_id: str | None; official_split: str; label_live_spoof: Literal["live", "spoof"]

class TargetFrameRecord(_FrameBase):
    pass

class _CropBase(ManifestModel):
    sample_id: str; dataset: str; video_id: str; source_record_id: str
    source_media_type: str; requested_frame_index: int; actual_frame_index: int
    timestamp_ms: float | None; frame_width: int; frame_height: int
    bbox_x1: float; bbox_y1: float; bbox_x2: float; bbox_y2: float
    landmark_0_x: float; landmark_0_y: float; landmark_1_x: float; landmark_1_y: float
    landmark_2_x: float; landmark_2_y: float; landmark_3_x: float; landmark_3_y: float
    landmark_4_x: float; landmark_4_y: float; detection_score: float; detected_face_count: int
    crop_x1: int; crop_y1: int; crop_x2: int; crop_y2: int; requested_crop_padding: float
    effective_crop_padding: float; crop_width: int; crop_height: int; crop_relative_path: str
    crop_sha256: str; detector_name: str; detector_model_sha256: str; detector_provider: str
    detector_input_size: int; detector_threshold: float; preprocessing_version: str
    preprocessing_config_hash: str; status: str
    @model_validator(mode="after")
    def valid_geometry(self):
        values = [self.bbox_x1,self.bbox_y1,self.bbox_x2,self.bbox_y2,
                  self.landmark_0_x,self.landmark_0_y,self.landmark_1_x,self.landmark_1_y,
                  self.landmark_2_x,self.landmark_2_y,self.landmark_3_x,self.landmark_3_y,
                  self.landmark_4_x,self.landmark_4_y]
        if not all(isfinite(v) for v in values) or self.bbox_x2 <= self.bbox_x1 or self.bbox_y2 <= self.bbox_y1:
            raise ValueError("invalid_bbox_or_landmarks")
        return self

class SourceCropRecord(_CropBase):
    subject_id: str | None; official_split: str; label_live_spoof: Literal["live", "spoof"]

class TargetCropRecord(_CropBase):
    pass

class PreprocessingFailureRecord(ManifestModel):
    dataset: str; source_record_id: str; source_relative_identifier: str; sample_id: str | None
    requested_frame_index: int | None; actual_frame_index: int | None; stage: str
    error_code: Literal["source_missing","unsupported_media","metadata_read_failed","decode_failed","frame_index_unavailable","unreadable_image","duplicate_frame_index","output_shape_invalid","detector_failed","no_face","invalid_bbox","invalid_landmarks","crop_failed","output_write_failed","hash_failed","config_mismatch","state_conflict","target_leakage_detected"]
    sanitized_error_message: str; timestamp: str; preprocessing_config_hash: str
    detector_model_sha256: str; backend: str; recoverable: bool; warning_codes: list[str] = Field(default_factory=list)

MODELS = {"source_frames": SourceFrameRecord, "source_crops": SourceCropRecord,
          "target_frames": TargetFrameRecord, "target_crops": TargetCropRecord,
          "preprocessing_failures": PreprocessingFailureRecord}
