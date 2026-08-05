import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from prism_fas.data.manifests.leakage import FORBIDDEN, find_target_leakage
from prism_fas.data.manifests.repository import ManifestRepository
from prism_fas.data.manifests.routing import route_source_success, route_target_success
from prism_fas.data.output import hash_crop_artifact, write_crop_image
from prism_fas.data.package import (QUALITY_NAMES, TargetIsolationError, build_package, compute_quality,
                                    finalize_lock, load_package_config, project_split, select_split_manifest,
                                    validate_package)
from prism_fas.data.package.manifests import read_manifest
from prism_fas.data.package.priors import load_prior, validate_prior_arrays
from prism_fas.data.package.quality import QualityMetricError
from prism_fas.data.run_context import M2OutputLayout, PreprocessingRunContext
from prism_fas.data.schemas.records import CanonicalVideoRecord, TargetInferenceRecord

CONFIG = load_package_config(Path(__file__).parents[1] / "configs" / "data" / "package_m3a.yaml")
PRIVATE_TOKENS = ("live", "spoof", "attack", "taxonomy", "subject", "session", "replay", ".mov")


def _crop(seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)


def _m2_run(tmp_path):
    """Build a small but real M2 output tree (routing + writer + hashes)."""
    layout = M2OutputLayout.from_root(tmp_path / "full_preprocessing_v2")
    def context(dataset, role):
        return PreprocessingRunContext(project_root=tmp_path, work_root=tmp_path, run_profile="full_preprocessing", output_namespace="full_preprocessing", output_root=layout.output_root, crops_root=layout.crops_root, frames_root=layout.frames_root, manifests_root=layout.manifests_root, state_root=layout.state_root, reports_root=layout.reports_root, logs_root=layout.logs_root, run_id="m3a", dataset=dataset, dataset_role=role, preprocessing_version="m2-v1", preprocessing_config_hash="a" * 64, detector_model_path=tmp_path / "m.onnx", detector_model_sha256="b" * 64, detector_input_size=320, detector_threshold=.5, all_records=False, record_limit=1, sample_limit=None, resume=False, dry_run=False, partial_full_profile=False, command="test")
    repository = ManifestRepository(layout.manifests_root, {"manifest_schema_version": "m2f1a-v1", "preprocessing_config_hash": "a" * 64, "detector_model_sha256": "b" * 64}).initialize()
    plan = [("casia_fasd", "source", "train", "live", "src-train-1"), ("casia_fasd", "source", "train", "spoof", "src-train-2"),
            ("msu_mfsd", "source", "test", "live", "src-dev-1"), ("casia_fasd", "source", "test", "spoof", "src-dev-2"),
            ("siw_mv2", "target", None, None, "tgt-1"), ("siw_mv2", "target", None, None, "tgt-2")]
    for index, (dataset, role, split, label, sample) in enumerate(plan):
        relative = f"crops/{dataset}/{sample}.jpg"
        artifact = write_crop_image(_crop(index), layout.output_root / relative)
        common = dict(repository=repository, context=context(dataset, role), sample_id=sample, requested_frame_index=index,
                      actual_frame_index=index + 1, source_media_type="image_sequence" if dataset == "casia_fasd" else "video_file",
                      timestamp_ms=None, frame_width=640, frame_height=480, decoder_backend="opencv",
                      bbox=[10., 10., 110., 130.], landmarks=[(20., 22.), (60., 22.), (40., 60.), (25., 90.), (55., 90.)],
                      detection_score=.9, detected_face_count=1, crop_box=[0, 0, 120, 140], crop_width=224, crop_height=224,
                      crop_relative_path=relative, crop_sha256=hash_crop_artifact(artifact))
        if role == "source":
            route_source_success(canonical_record=CanonicalVideoRecord(dataset=dataset, subject_id=f"s{index}", video_id=f"{split}_v{index}", source_path=tmp_path / "x.mp4", official_split=split, label=label, adapter_version="1.0", source_fingerprint="c" * 64, metadata_provenance="test"), selected_frame_reference=f"{split}_v{index}#frame={index+1}", **common)
        else:
            route_target_success(canonical_record=TargetInferenceRecord(dataset=dataset, video_id=f"siw_{index:016x}", source_path=tmp_path / "Live_477.mov", official_split="target_test", adapter_version="1.0", source_fingerprint="d" * 64, metadata_provenance="test"), selected_frame_reference=f"target/{sample}#frame={index+1}", **common)
    repository.flush()
    return layout.output_root


def _package(tmp_path, name="pkg", **kwargs):
    input_root = _m2_run(tmp_path)
    root = tmp_path / name
    result = build_package(input_root, root, CONFIG, **kwargs)
    pre = validate_package(root, require_validated_status=False)
    if pre["passed"]: finalize_lock(root, pre)
    return input_root, root, result, validate_package(root)


# --- quality metrics -------------------------------------------------------

def test_quality_metrics_are_deterministic_and_finite():
    image = _crop(7)
    first = compute_quality(image, bbox=(10., 10., 110., 130.), frame_width=640, frame_height=480)
    second = compute_quality(image.copy(), bbox=(10., 10., 110., 130.), frame_width=640, frame_height=480)
    assert first == second and sorted(first) == sorted(QUALITY_NAMES)
    assert all(np.isfinite(v) for v in first.values())
    assert first["face_size_ratio"] == pytest.approx((100. * 120.) / (640. * 480.))
    assert 0. <= first["brightness_mean"] <= 1. and 0. <= first["saturation_mean"] <= 1.


def test_invalid_frame_area_fails_loudly():
    with pytest.raises(QualityMetricError):
        compute_quality(_crop(1), bbox=(10., 10., 110., 130.), frame_width=0, frame_height=0)


# --- priors ----------------------------------------------------------------

def test_prior_npz_schema_and_no_pickle(tmp_path):
    _, root, _, _ = _package(tmp_path)
    for path in sorted((root / "priors").glob("*.npz")):
        arrays = load_prior(path)          # np.load(..., allow_pickle=False)
        validate_prior_arrays(arrays)
        assert arrays["bbox"].dtype == np.float32 and arrays["landmarks"].shape == (5, 2)
        assert arrays["quality_vector"].shape == (len(QUALITY_NAMES),)
        assert list(arrays["quality_names"]) == list(QUALITY_NAMES)
        blob = path.read_bytes().lower()
        for token in (b"c:\\", b"d:\\", b"/users/", b"live_477", b"spoof"):
            assert token not in blob


def test_prior_resume_is_idempotent_and_atomic(tmp_path):
    input_root, root, first, _ = _package(tmp_path)
    digests = {p.name: p.read_bytes() for p in sorted((root / "priors").glob("*.npz"))}
    second = build_package(input_root, root, CONFIG, resume=True)
    assert second["stats"].priors_reused == first["stats"].samples and second["stats"].priors_built == 0
    assert {p.name: p.read_bytes() for p in sorted((root / "priors").glob("*.npz"))} == digests
    assert second["hashes"] == first["hashes"]
    assert second["lock"]["content_identity_sha256"] == first["lock"]["content_identity_sha256"]
    assert not list((root / "priors").glob("*.tmp"))


# --- splits and target isolation -------------------------------------------

def test_split_mapping_from_official_metadata():
    assert project_split("source", "train") == "source_train"
    assert project_split("source", "test") == "source_dev"
    assert project_split("target", None) == "target_test"
    with pytest.raises(ValueError): project_split("source", "unknown")


def test_target_rows_are_label_free(tmp_path):
    _, root, _, report = _package(tmp_path)
    target = read_manifest(root / "manifests" / "target_test_features.parquet")
    assert target and not (set(target[0]) & FORBIDDEN) and find_target_leakage(target) == []
    samples = [r for r in read_manifest(root / "manifests" / "samples.parquet") if r["project_split"] == "target_test"]
    assert find_target_leakage(samples) == []
    for row in target + samples:
        blob = json.dumps(row, default=str).lower()
        for token in PRIVATE_TOKENS:
            assert token not in blob
    assert report["target_isolation"]["passed"] is True


def test_training_selector_rejects_target_split(tmp_path):
    _, root, _, _ = _package(tmp_path)
    assert select_split_manifest(root, "source_train", mode="training")
    assert select_split_manifest(root, "source_dev", mode="training")
    with pytest.raises(TargetIsolationError, match="cannot request target_test"):
        select_split_manifest(root, "target_test", mode="training")
    assert select_split_manifest(root, "target_test", mode="inference")


# --- shards ----------------------------------------------------------------

def test_shards_are_deterministic_triplets_isolated_by_split(tmp_path):
    input_root, root, result, _ = _package(tmp_path)
    shards = read_manifest(root / "manifests" / "shards_index.parquet")
    assert sum(row["row_count"] for row in shards) == result["stats"].samples
    splits = {row["shard_filename"]: row["split"] for row in shards}
    for shard in sorted((root / "shards").glob("*.tar")):
        with tarfile.open(shard) as archive: names = archive.getnames()
        assert names == sorted(names, key=lambda n: (n.rsplit(".", 1)[0], [".jpg", ".npz", ".json"].index("." + n.rsplit(".", 1)[1])))
        assert len(names) == len(set(names))
        stems = {n.rsplit(".", 1)[0] for n in names}
        for stem in stems:
            assert {n for n in names if n.startswith(stem + ".")} == {f"{stem}.jpg", f"{stem}.npz", f"{stem}.json"}
        assert not any(n.startswith("/") or ".." in Path(n).parts or ":" in n for n in names)
        expected = {r["sample_id"] for r in read_manifest(root / "manifests" / "samples.parquet") if r["project_split"] == splits[shard.name]}
        assert stems == expected
    # rebuilding produces byte-identical shards
    before = {p.name: p.read_bytes() for p in sorted((root / "shards").glob("*.tar"))}
    build_package(input_root, root, CONFIG, resume=True)
    assert {p.name: p.read_bytes() for p in sorted((root / "shards").glob("*.tar"))} == before


def test_target_shard_json_has_no_labels(tmp_path):
    _, root, _, _ = _package(tmp_path)
    with tarfile.open(root / "shards" / "target_test-00000.tar") as archive:
        payloads = [json.loads(archive.extractfile(n).read()) for n in archive.getnames() if n.endswith(".json")]
    assert payloads
    for payload in payloads:
        assert "label_live_spoof" not in payload and "official_split" not in payload
        assert not (set(payload) & FORBIDDEN)
        for token in PRIVATE_TOKENS:
            assert token not in json.dumps(payload).lower()
    with tarfile.open(root / "shards" / "source_train-00000.tar") as archive:
        source = [json.loads(archive.extractfile(n).read()) for n in archive.getnames() if n.endswith(".json")]
    assert all(payload["label_live_spoof"] in {"live", "spoof"} for payload in source)


# --- lock and validator ----------------------------------------------------

def test_package_lock_and_validation_pass(tmp_path):
    _, root, result, report = _package(tmp_path)
    lock = json.loads((root / "PACKAGE_LOCK.json").read_text())
    assert lock["status"] == "validated" and lock["package_validation"]["status"] == "passed"
    assert lock["total_samples"] == result["stats"].samples and lock["content_identity_sha256"]
    assert lock["deferred_priors"] == {"parsing_status": "not_computed", "pose_status": "not_computed", "visibility_status": "not_computed", "identity_status": "not_computed"}
    assert report["passed"] is True and report["errors"] == []
    assert report["counts"]["samples"] == result["stats"].samples
    blob = json.dumps(lock).lower()
    for token in ("c:\\", "d:\\", "live_477"):
        assert token not in blob


@pytest.mark.parametrize("damage", ["missing_image", "missing_prior", "image_sha", "prior_sha", "shard_sha", "orphan"])
def test_validator_detects_damage(tmp_path, damage):
    _, root, _, _ = _package(tmp_path)
    samples = read_manifest(root / "manifests" / "samples.parquet")
    image = root / samples[0]["image_relative_path"]; prior = root / samples[0]["prior_relative_path"]
    if damage == "missing_image": image.unlink()
    elif damage == "missing_prior": prior.unlink()
    elif damage == "image_sha": image.write_bytes(image.read_bytes() + b"tamper")
    elif damage == "prior_sha": prior.write_bytes(prior.read_bytes() + b"tamper")
    elif damage == "shard_sha":
        shard = next((root / "shards").glob("*.tar")); shard.write_bytes(shard.read_bytes() + b"tamper")
    else: (root / "images" / "orphan.jpg").write_bytes(b"orphan")
    report = validate_package(root)
    assert report["passed"] is False and report["errors"]


def test_validator_rejects_unvalidated_status(tmp_path):
    input_root = _m2_run(tmp_path)
    root = tmp_path / "pending"
    build_package(input_root, root, CONFIG)          # leaves status=building
    report = validate_package(root)
    assert any(check["check_id"] == "lock.status" and not check["passed"] for check in report["checks"])
