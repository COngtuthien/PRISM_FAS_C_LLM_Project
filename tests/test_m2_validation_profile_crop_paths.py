import numpy as np
import pytest

from prism_fas.data.m2_validation import validate_m2
from prism_fas.data.manifests.repository import ManifestRepository
from prism_fas.data.manifests.routing import route_source_success
from prism_fas.data.output import write_crop_image, hash_crop_artifact
from prism_fas.data.run_context import M2OutputLayout, PreprocessingRunContext
from prism_fas.data.schemas.records import CanonicalVideoRecord


class _Paths:
    def __init__(self, tmp_path): self.work_root = tmp_path; self.project_root = tmp_path


@pytest.mark.parametrize("layout_kind", ["profile_relative", "legacy_m2a"])
def test_crop_integrity_resolves_both_profile_and_legacy_layouts(tmp_path, layout_kind):
    """Regression: validation resolved crops only under a hardcoded m2a/ prefix."""
    root = tmp_path / "8f1e" / ("full_preprocessing" if layout_kind == "profile_relative" else "")
    layout = M2OutputLayout.from_root(root)
    context = PreprocessingRunContext(project_root=tmp_path, work_root=tmp_path, run_profile="full_preprocessing", output_namespace="full_preprocessing", output_root=layout.output_root, crops_root=layout.crops_root, frames_root=layout.frames_root, manifests_root=layout.manifests_root, state_root=layout.state_root, reports_root=layout.reports_root, logs_root=layout.logs_root, run_id="validate", dataset="casia_fasd", dataset_role="source", preprocessing_version="m2-v1", preprocessing_config_hash="a" * 64, detector_model_path=tmp_path / "m.onnx", detector_model_sha256="b" * 64, detector_input_size=320, detector_threshold=.5, all_records=False, record_limit=1, sample_limit=None, resume=False, dry_run=False, partial_full_profile=False, command="test")
    record = CanonicalVideoRecord(dataset="casia_fasd", subject_id="s1", video_id="v1", source_path=tmp_path / "v1f1.png", official_split="train", label="live", adapter_version="v1", source_fingerprint="c" * 64, metadata_provenance="test")

    relative = "crops/casia_fasd/sample.jpg"
    artifact = write_crop_image(np.full((224, 224, 3), 128, dtype=np.uint8), (root / "m2a" / relative) if layout_kind == "legacy_m2a" else (context.output_root / relative))
    repository = ManifestRepository(context.manifests_root, {"manifest_schema_version": "m2f1a-v1"}).initialize()
    route_source_success(repository=repository, context=context, canonical_record=record, sample_id="sample", requested_frame_index=0, actual_frame_index=0, source_media_type="image_sequence", timestamp_ms=None, frame_width=64, frame_height=64, decoder_backend="opencv", selected_frame_reference="v1#frame=0", bbox=[10., 10., 50., 50.], landmarks=[(20., 22.)] * 5, detection_score=.9, detected_face_count=1, crop_box=[0, 0, 60, 60], crop_width=224, crop_height=224, crop_relative_path=relative, crop_sha256=hash_crop_artifact(artifact))
    repository.flush()

    class Config:
        preprocessing_version = "m2-v1"; config_hash = "a" * 64; scrfd_model_path = context.detector_model_path
        scrfd_input_size = 320; detection_threshold = .5
    context.detector_model_path.write_bytes(b"synthetic-model")
    report = validate_m2(_Paths(tmp_path), Config(), root)
    crop_checks = [c for c in report["checks"] if c["check_id"].startswith("crop.")]
    assert crop_checks and all(c["passed"] for c in crop_checks), crop_checks
    assert report["crop_integrity"]["passed"] is True
