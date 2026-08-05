import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from prism_fas.data.package.model_priors import (VISIBILITY_REGIONS, ModelPriorError, ModelWeightError, compute_visibility,
                                                 load_model_config, region_masks, resolve_weight, validate_identity,
                                                 validate_parsing, validate_pose)

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "models" / "m3b_priors.yaml"
CONFIG = load_model_config(CONFIG_PATH)


def _parsing(seed=0):
    """Synthetic LaPa-style parsing mask with every required region present."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[20:200, 40:184] = 1          # skin
    mask[10:60, :] = 10               # hair
    mask[70:85, 60:90] = 4            # left eye
    mask[70:85, 134:164] = 5          # right eye
    mask[95:135, 100:124] = 6         # nose
    mask[150:160, 95:130] = 7         # upper lip
    mask[160:168, 95:130] = 8         # inner mouth
    mask[168:178, 95:130] = 9         # lower lip
    mask[60:68, 60:90] = 2            # left brow
    mask[60:68, 134:164] = 3          # right brow
    return mask


# --- config / pins ---------------------------------------------------------

def test_model_registry_pins_revisions_and_hashes():
    for section in ("parsing", "identity"):
        spec = CONFIG[section]
        assert spec["revision"] and len(spec["weight_sha256"]) == 64
        assert not spec["weight_relative_path"].startswith(("/", "C:", "D:"))
    assert CONFIG["pose"]["convention"] == "yaw_pitch_roll_radians"
    assert CONFIG["identity"]["embedding_dim"] == 512
    assert tuple(CONFIG["visibility"]["region_order"]) == VISIBILITY_REGIONS
    assert "D:" not in CONFIG_PATH.read_text(encoding="utf-8")


def test_missing_or_mismatched_weight_is_rejected(tmp_path):
    with pytest.raises(ModelWeightError, match="weight is missing"):
        resolve_weight(CONFIG, "parsing", tmp_path)
    target = tmp_path / CONFIG["parsing"]["weight_relative_path"]
    target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(b"wrong weights")
    with pytest.raises(ModelWeightError, match="SHA-256 mismatch"):
        resolve_weight(CONFIG, "parsing", tmp_path)


# --- prior array contracts -------------------------------------------------

def test_parsing_validation_shape_dtype_and_degenerate_rejection():
    parsing = validate_parsing(_parsing())
    assert parsing.shape == (224, 224) and parsing.dtype == np.uint8
    with pytest.raises(ModelPriorError, match="\\[224,224\\]"):
        validate_parsing(np.zeros((112, 112), dtype=np.uint8))
    with pytest.raises(ModelPriorError, match="degenerate"):
        validate_parsing(np.zeros((224, 224), dtype=np.uint8))
    with pytest.raises(ModelPriorError, match="exceeds"):
        validate_parsing(np.full((224, 224), 40, dtype=np.uint8))


@pytest.mark.parametrize("pose", [np.array([np.nan, 0, 0]), np.array([np.inf, 0, 0]), np.zeros(2)])
def test_invalid_pose_is_rejected(pose):
    with pytest.raises(ModelPriorError):
        validate_pose(pose)


def test_pose_is_three_finite_float32():
    pose = validate_pose(np.array([0.1, -0.2, 0.05], dtype=np.float64))
    assert pose.shape == (3,) and pose.dtype == np.float32 and np.isfinite(pose).all()


# --- visibility ------------------------------------------------------------

def test_visibility_shape_range_order_and_not_constant():
    visibility = compute_visibility(_parsing(), np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert visibility.shape == (9,) and visibility.dtype == np.float16
    values = visibility.astype(np.float32)
    assert np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()
    assert len(set(np.round(values, 3).tolist())) > 1, "visibility must not be a constant vector"
    masks = region_masks(_parsing())
    assert list(masks) == list(VISIBILITY_REGIONS)


def test_visibility_reacts_to_yaw_self_occlusion():
    parsing = _parsing()
    frontal = compute_visibility(parsing, np.array([0.0, 0.0, 0.0], np.float32)).astype(np.float32)
    turned = compute_visibility(parsing, np.array([1.2, 0.0, 0.0], np.float32)).astype(np.float32)
    left = VISIBILITY_REGIONS.index("left_cheek")
    assert turned[left] < frontal[left]
    assert compute_visibility(parsing, np.array([0.0, 0.0, 0.0], np.float32)).tolist() == frontal.tolist()


# --- identity --------------------------------------------------------------

def test_identity_embedding_validation():
    vector = np.random.default_rng(0).normal(size=512).astype(np.float32)
    vector /= np.linalg.norm(vector)
    embedding = validate_identity(vector)
    assert embedding.shape == (512,) and embedding.dtype == np.float16
    with pytest.raises(ModelPriorError):
        validate_identity(np.zeros(256, dtype=np.float32))
    with pytest.raises(ModelPriorError, match="norm"):
        validate_identity(np.zeros(512, dtype=np.float32))
    with pytest.raises(ModelPriorError):
        validate_identity(np.full(512, np.nan, dtype=np.float32))


def test_identity_selection_contract_is_source_train_live_only():
    from prism_fas.data.package.m3b import _identity_applicable
    labels = {"a": "live", "b": "spoof", "c": "live"}
    rows = [{"sample_id": "a", "project_split": "source_train"}, {"sample_id": "b", "project_split": "source_train"},
            {"sample_id": "c", "project_split": "source_dev"}, {"sample_id": "t", "project_split": "target_test"}]
    applicable = [row["sample_id"] for row in rows if _identity_applicable(row, labels, CONFIG)]
    assert applicable == ["a"]


# --- real smoke package ----------------------------------------------------

SMOKE = Path(__file__).parents[1] / "data" / "processed" / "prism_data_v1_m3b_smoke"
smoke_required = pytest.mark.skipif(not (SMOKE / "PACKAGE_LOCK.json").is_file(), reason="M3B smoke package not built")


@smoke_required
def test_smoke_package_priors_and_isolation():
    from prism_fas.data.package.manifests import read_manifest
    from prism_fas.data.package.priors import load_prior
    lock = json.loads((SMOKE / "PACKAGE_LOCK.json").read_text())
    assert lock["status"] == "validated" and lock["parent_package_id"] == "prism_data_v1_m3a"
    samples = read_manifest(SMOKE / "manifests" / "samples.parquet")
    identity = 0
    for row in samples:
        arrays = load_prior(SMOKE / row["prior_relative_path"])      # allow_pickle=False
        assert arrays["parsing_labels"].shape == (224, 224) and arrays["parsing_labels"].dtype == np.uint8
        assert len(np.unique(arrays["parsing_labels"])) > 1
        assert arrays["pose_ypr"].shape == (3,) and np.isfinite(arrays["pose_ypr"]).all()
        visibility = arrays["visibility"].astype(np.float32)
        assert visibility.shape == (9,) and (visibility >= 0).all() and (visibility <= 1).all()
        if "identity_embedding" in arrays:
            identity += 1
            assert row["project_split"] == "source_train"
            assert 0.5 <= float(np.linalg.norm(arrays["identity_embedding"].astype(np.float32))) <= 1.5
        if row["project_split"] == "target_test":
            assert "identity_embedding" not in arrays
    assert identity == lock["prior_counts"]["identity_computed"] > 0


@smoke_required
def test_smoke_package_validates_against_parent():
    from prism_fas.data.package.validator import validate_package
    parent = Path(__file__).parents[1] / "data" / "processed" / "prism_data_v1_m3a"
    report = validate_package(SMOKE, parent_package=parent)
    assert report["passed"] is True, report["errors"]
    assert report["parent_package_id"] == "prism_data_v1_m3a"
    assert report["target_isolation"]["passed"] is True
    assert report["prior_counts"]["parsing"] == report["counts"]["samples"]


@smoke_required
def test_validator_detects_missing_parsing_and_wrong_identity(tmp_path):
    import shutil
    from prism_fas.data.package.manifests import read_manifest
    from prism_fas.data.package.priors import load_prior, serialize_prior, write_prior_atomic
    from prism_fas.data.package.validator import validate_package
    root = tmp_path / "damaged"; shutil.copytree(SMOKE, root)
    row = read_manifest(root / "manifests" / "samples.parquet")[0]
    arrays = load_prior(root / row["prior_relative_path"])
    write_prior_atomic(root / row["prior_relative_path"], serialize_prior({k: v for k, v in arrays.items() if k != "parsing_labels"}))
    report = validate_package(root, require_validated_status=False)
    assert report["passed"] is False
    assert any(check["check_id"].startswith("m3b.") and not check["passed"] for check in report["checks"])


def test_package_content_identity_excludes_wall_clock_fields():
    """Regression: rebuilding identical artifacts changed content_identity_sha256
    because the promoted lock hashed the wall-clock build_seconds field."""
    from prism_fas.data.package.builder import IDENTITY_EXCLUDED_FIELDS
    from prism_fas.utils.core import stable_json_hash
    assert {"created_at", "git_commit", "build_seconds", "environment", "content_identity_sha256"} <= IDENTITY_EXCLUDED_FIELDS
    base = {"package_id": "p", "total_samples": 3, "manifest_sha256": {"samples": "a" * 64}}
    first = {**base, "created_at": "2026-08-05T10:00:00Z", "build_seconds": 2371.0, "git_commit": "aaa",
             "environment": {"device": "cpu", "torch": "2.13.0+cpu"}}
    second = {**base, "created_at": "2026-08-06T22:31:00Z", "build_seconds": 157.7, "git_commit": "bbb",
              "environment": {"device": "cuda", "torch": "2.13.0+cu121"}}
    identity = lambda lock: stable_json_hash({k: v for k, v in lock.items() if k not in IDENTITY_EXCLUDED_FIELDS})
    assert identity(first) == identity(second)
    assert identity({**base, "total_samples": 4}) != identity(first)
