"""POST_FAILURE_EXPLORATORY_TARGET_V3 — E1 execution TECHNICAL FIX
(`E1_TECHNICAL_TARGET_LOADER_POLICY_FAILURE`).

A real GPU `--predict` attempt (code commit
`89929ab109b994ab0ab31688087f3a4befc33bb1`, execution identity
`c08186944c8f39ff`) loaded checkpoint weights, passed the recipe_arm/
variant-capability fix, then failed inside `target_batches()`'s
`open_package()` call with `PackageContractError: package id
'prism_target_eval_v2' != expected 'prism_data_v1_m3b'`. No target batch
was yielded, no target sample was read, no prediction was produced, no
label was opened.

Root cause: `construct_row_trainer` correctly reconstructs the SOURCE
trainer, so `trainer.loader_config` still declares the SOURCE package
policy (`configs/data/loader_m4.yaml`: `expected_package_id:
prism_data_v1_m3b`). V3 passed that SOURCE-bound config straight into
`target_batches`, which opens the TARGET feature package through the same
typed policy — so `open_package()` correctly rejected the valid target
package `prism_target_eval_v2`.

This file proves the corrected `build_verified_target_loader_config`
(rebinds ONLY the two package-policy fields, preserves every other field,
never weakens validation) and `resolve_frozen_target_package_reference`
(the target package id/identity come SOLELY from the already-frozen
`PREDICTION_PLAN_BINDING.json`, fail-closed on any inconsistency, before
any target sample is read).

**FIXTURE / ENGINEERING ONLY.** Loader-config construction is pure
Pydantic validation; the one on-disk fixture package built here is
synthetic bytes in `tmp_path`, opened only through `open_package` (a
manifest-level structural check — never a real image, never a real
checkpoint, never a real target label). No test in this file runs real
target prediction, binds a prediction plan against real data, or opens a
target label.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.data.loader.config import INFERENCE_SPLIT, load_loader_config  # noqa: E402
from prism_fas.data.loader.package_index import PackageContractError, open_package  # noqa: E402
from prism_fas.data.package.manifests import TARGET_FEATURES_SCHEMA, write_manifest  # noqa: E402
from prism_fas.evaluation import post_failure_exploratory_target_v3 as v3  # noqa: E402

from test_post_failure_exploratory_target_v3 import _install_v3_binding_fixtures  # noqa: E402

V3_PROTOCOL_PATH = REPO / "configs/evaluation/post_failure_exploratory_target_v3.yaml"
LOADER_M4_PATH = REPO / "configs/data/loader_m4.yaml"
V3_IDENTITY = "a2b54f8844a2a36540e62470c2f5f30de52fbf509a37f03feb7f6d769d5c702c"
TARGET_PACKAGE_ID = "prism_target_eval_v2"
TARGET_CONTENT_IDENTITY = "c3a29e695ad08c4b31e01533f1d12374f4e30c51f0167c6622cf8168792e48a8"
FROZEN_BINDING_IDENTITY = "d6e4f4c45c375c1d8fca6257023436aff16bd5332ca911ed4c61ab1cd193223c"


def _load_source_config():
    """The REAL, frozen source loader config — read-only, never modified."""
    return load_loader_config(LOADER_M4_PATH)


def _build_fixture_target_package(root: Path, *, package_id: str = TARGET_PACKAGE_ID,
                                   content_identity: str = TARGET_CONTENT_IDENTITY,
                                   status: str = "validated") -> Path:
    row = {"sample_id": "s1", "dataset": "siw_mv2", "source_record_id": "vid-1",
          "project_split": "target_test", "image_relative_path": "images/s1.jpg",
          "prior_relative_path": "priors/s1.npz", "crop_sha256": "a" * 64, "prior_sha256": "b" * 64,
          "source_media_type": "video", "frame_width": 224, "frame_height": 224,
          "crop_width": 224, "crop_height": 224, "detection_score": 0.9, "detected_face_count": 1,
          "blur_laplacian_variance": 1.0, "brightness_mean": 1.0, "brightness_std": 1.0,
          "contrast_michelson": 1.0, "saturation_mean": 1.0, "face_size_ratio": 1.0,
          "package_schema_version": "target-eval-v1"}
    write_manifest(root / "manifests" / "target_test_features.parquet", [row], TARGET_FEATURES_SCHEMA, {})
    lock = {"package_id": package_id, "status": status, "content_identity_sha256": content_identity}
    (root / "PACKAGE_LOCK.json").write_text(json.dumps(lock), encoding="utf-8")
    return root


# ==============================================================================
# A/B/C. Package-policy fields swapped; everything else byte-identical;
#        the source object itself is untouched
# ==============================================================================

def test_source_config_produces_target_config_with_swapped_package_policy() -> None:
    source = _load_source_config()
    assert source.package.expected_package_id == "prism_data_v1_m3b"
    assert source.package.expected_content_identity_sha256 is None

    target = v3.build_verified_target_loader_config(
        source, target_package_id=TARGET_PACKAGE_ID, target_content_identity=TARGET_CONTENT_IDENTITY)

    assert target.package.expected_package_id == TARGET_PACKAGE_ID
    assert target.package.expected_content_identity_sha256 == TARGET_CONTENT_IDENTITY


def test_source_loader_config_object_remains_unchanged() -> None:
    source = _load_source_config()
    before = source.model_dump(mode="python")
    v3.build_verified_target_loader_config(
        source, target_package_id=TARGET_PACKAGE_ID, target_content_identity=TARGET_CONTENT_IDENTITY)
    after = source.model_dump(mode="python")
    assert before == after
    assert source.package.expected_package_id == "prism_data_v1_m3b"
    assert source.package.expected_content_identity_sha256 is None


def test_every_other_field_remains_exactly_identical() -> None:
    source = _load_source_config()
    target = v3.build_verified_target_loader_config(
        source, target_package_id=TARGET_PACKAGE_ID, target_content_identity=TARGET_CONTENT_IDENTITY)

    assert target is not source
    assert target.loader_schema_version == source.loader_schema_version
    assert target.image == source.image
    assert target.label_mapping == source.label_mapping
    assert target.sampler == source.sampler
    assert target.backends == source.backends
    assert target.dataloader == source.dataloader
    assert target.package.require_validated_status == source.package.require_validated_status
    assert target.package.integrity_verification == source.package.integrity_verification


# ==============================================================================
# D. Target package status/integrity policy remains enforced
# ==============================================================================

def test_non_validated_status_still_fails_closed_with_adapted_config(tmp_path) -> None:
    source = _load_source_config()
    target_cfg = v3.build_verified_target_loader_config(
        source, target_package_id=TARGET_PACKAGE_ID, target_content_identity=TARGET_CONTENT_IDENTITY)
    root = _build_fixture_target_package(tmp_path / TARGET_PACKAGE_ID, status="building")
    with pytest.raises(PackageContractError, match="expected 'validated'"):
        open_package(root, INFERENCE_SPLIT, target_cfg, mode="inference")


# ==============================================================================
# E/F/G. Opens with matching id/identity; wrong id or identity fails closed
# ==============================================================================

def test_fixture_target_package_opens_with_the_adapted_config(tmp_path) -> None:
    source = _load_source_config()
    target_cfg = v3.build_verified_target_loader_config(
        source, target_package_id=TARGET_PACKAGE_ID, target_content_identity=TARGET_CONTENT_IDENTITY)
    root = _build_fixture_target_package(tmp_path / TARGET_PACKAGE_ID)
    index = open_package(root, INFERENCE_SPLIT, target_cfg, mode="inference")
    assert index.package_id == TARGET_PACKAGE_ID
    assert index.content_identity == TARGET_CONTENT_IDENTITY
    assert len(index) == 1


def test_wrong_target_package_id_fails_closed(tmp_path) -> None:
    source = _load_source_config()
    root = _build_fixture_target_package(tmp_path / TARGET_PACKAGE_ID)
    wrong_cfg = v3.build_verified_target_loader_config(
        source, target_package_id="some_other_package_id", target_content_identity=TARGET_CONTENT_IDENTITY)
    with pytest.raises(PackageContractError, match="package id"):
        open_package(root, INFERENCE_SPLIT, wrong_cfg, mode="inference")


def test_wrong_target_content_identity_fails_closed(tmp_path) -> None:
    source = _load_source_config()
    root = _build_fixture_target_package(tmp_path / TARGET_PACKAGE_ID)
    wrong_cfg = v3.build_verified_target_loader_config(
        source, target_package_id=TARGET_PACKAGE_ID, target_content_identity="0" * 64)
    with pytest.raises(PackageContractError, match="content identity"):
        open_package(root, INFERENCE_SPLIT, wrong_cfg, mode="inference")


def test_empty_target_package_id_fails_closed_without_building_a_config() -> None:
    source = _load_source_config()
    with pytest.raises(v3.ExploratoryTargetV3Error, match="target_package_id"):
        v3.build_verified_target_loader_config(
            source, target_package_id="", target_content_identity=TARGET_CONTENT_IDENTITY)


def test_empty_target_content_identity_fails_closed_without_building_a_config() -> None:
    source = _load_source_config()
    with pytest.raises(v3.ExploratoryTargetV3Error, match="target_content_identity"):
        v3.build_verified_target_loader_config(
            source, target_package_id=TARGET_PACKAGE_ID, target_content_identity="")


# ==============================================================================
# resolve_frozen_target_package_reference — sourced SOLELY from the frozen
# binding, fail-closed on any inconsistency
# ==============================================================================

def _valid_frozen_binding(*, row_ids=("ROW-0", "ROW-1")) -> dict[str, Any]:
    return {
        "target_feature_package_identity": TARGET_CONTENT_IDENTITY,
        "target_feature_package": {"package_id": TARGET_PACKAGE_ID,
                                   "computed_identity": TARGET_CONTENT_IDENTITY,
                                   "expected_identity": TARGET_CONTENT_IDENTITY},
        "rows": {row_id: {"target_feature_package_identity": TARGET_CONTENT_IDENTITY} for row_id in row_ids},
    }


def test_resolve_frozen_target_package_reference_reads_the_binding() -> None:
    package_id, content_identity = v3.resolve_frozen_target_package_reference(_valid_frozen_binding())
    assert package_id == TARGET_PACKAGE_ID
    assert content_identity == TARGET_CONTENT_IDENTITY


def test_empty_package_id_in_binding_fails_closed() -> None:
    binding = _valid_frozen_binding()
    binding["target_feature_package"]["package_id"] = ""
    with pytest.raises(v3.ExploratoryTargetV3Error, match="package_id"):
        v3.resolve_frozen_target_package_reference(binding)


def test_empty_top_level_identity_in_binding_fails_closed() -> None:
    binding = _valid_frozen_binding()
    binding["target_feature_package_identity"] = ""
    with pytest.raises(v3.ExploratoryTargetV3Error, match="target_feature_package_identity"):
        v3.resolve_frozen_target_package_reference(binding)


def test_nested_computed_identity_disagreement_fails_closed() -> None:
    binding = _valid_frozen_binding()
    binding["target_feature_package"]["computed_identity"] = "f" * 64
    with pytest.raises(v3.ExploratoryTargetV3Error, match="computed_identity"):
        v3.resolve_frozen_target_package_reference(binding)


def test_nested_expected_identity_disagreement_fails_closed() -> None:
    binding = _valid_frozen_binding()
    binding["target_feature_package"]["expected_identity"] = "f" * 64
    with pytest.raises(v3.ExploratoryTargetV3Error, match="expected_identity"):
        v3.resolve_frozen_target_package_reference(binding)


def test_a_disagreeing_row_identity_fails_closed() -> None:
    binding = _valid_frozen_binding()
    binding["rows"]["ROW-1"]["target_feature_package_identity"] = "f" * 64
    with pytest.raises(v3.ExploratoryTargetV3Error, match=r"row\(s\).*ROW-1"):
        v3.resolve_frozen_target_package_reference(binding)


# ==============================================================================
# H/I/J/K. _predict()/predict_one_row_to_staging wiring
# ==============================================================================

def test_predict_source_resolves_target_package_id_from_the_frozen_binding() -> None:
    source = inspect.getsource(v3._predict)
    assert "resolve_frozen_target_package_reference(frozen_binding)" in source
    assert '"prism_target_eval_v2"' not in source
    resolve_index = source.index("resolve_frozen_target_package_reference(frozen_binding)")
    call_index = source.index("predict_one_row_to_staging(")
    assert resolve_index < call_index
    assert "target_package_id=target_package_id" in source


def test_predict_one_row_to_staging_passes_the_adapted_loader_not_the_trainer_loader_config() -> None:
    source = inspect.getsource(v3.predict_one_row_to_staging)
    assert "target_batches(Path(package_root), target_loader_config" in source
    assert "target_batches(Path(package_root), trainer.loader_config" not in source
    assert "build_verified_target_loader_config(" in source


def test_capability_resolution_still_precedes_loader_build_and_target_batches() -> None:
    source = inspect.getsource(v3.predict_one_row_to_staging)
    capability_index = source.index("resolve_verified_row_capabilities(binding")
    loader_index = source.index("build_verified_target_loader_config(")
    batches_index = source.index("batches = target_batches(")
    assert capability_index < loader_index < batches_index


def test_row_lock_uses_the_passed_target_package_id_not_a_hardcoded_literal() -> None:
    source = inspect.getsource(v3.predict_one_row_to_staging)
    assert 'target_package_id="prism_target_eval_v2"' not in source
    assert "target_package_id=target_package_id" in source


def test_predict_one_row_to_staging_requires_an_explicit_target_package_id_argument() -> None:
    signature = inspect.signature(v3.predict_one_row_to_staging)
    assert "target_package_id" in signature.parameters
    assert signature.parameters["target_package_id"].kind == inspect.Parameter.KEYWORD_ONLY


def test_predict_stops_before_any_row_prediction_when_binding_package_metadata_is_inconsistent(
        monkeypatch, tmp_path) -> None:
    """`resolve_frozen_target_package_reference` is called and its failure
    is respected BEFORE the row-prediction loop: `verify_binding_unchanged`
    (an earlier, independent defense layer) already refuses any on-disk
    tamper that would make this reachable through real file mutation, so
    this test verifies the WIRING directly — a poisoned resolver still
    blocks `_predict` with zero calls to `predict_one_row_to_staging`,
    exactly as it would if the frozen binding's own per-row identity
    consistency were ever violated."""
    _install_v3_binding_fixtures(monkeypatch)
    exit_code, _ = v3._bind_prediction_plan(tmp_path)
    assert exit_code == v3.EXIT_PASS

    calls: list[str] = []
    monkeypatch.setattr(v3, "predict_one_row_to_staging",
                        lambda *a, **k: calls.append("called") or (_ for _ in ()).throw(
                            AssertionError("must never be called")))
    monkeypatch.setattr(v3, "current_code_commit", lambda repo: "c" * 40)

    def _poisoned_resolver(frozen_binding):
        raise v3.ExploratoryTargetV3Error("simulated frozen-binding package metadata inconsistency")

    monkeypatch.setattr(v3, "resolve_frozen_target_package_reference", _poisoned_resolver)

    exit_code, payload = v3._predict(tmp_path)
    assert exit_code == v3.EXIT_BLOCKED
    assert calls == []
    assert "simulated frozen-binding package metadata inconsistency" in payload.get("error", "")


# ==============================================================================
# L. The recipe_arm regression stays passing (imported, not re-implemented)
# ==============================================================================

def test_recipe_arm_capability_resolution_is_untouched_by_this_fix() -> None:
    source = inspect.getsource(v3.resolve_verified_row_capabilities)
    assert "KNOWN_NON_VARIANT_ROW_METADATA_FLAGS" in source
    assert "trainer_variant" in source


# ==============================================================================
# M. build_prediction_plan_binding is untouched (its recomputed identity is
#    therefore unaffected by this fix)
# ==============================================================================

def test_build_prediction_plan_binding_source_never_references_the_new_loader_machinery() -> None:
    """This function is not modified by this fix. Its source referencing
    none of the new symbols is a structural guarantee that its output — and
    therefore the frozen prediction-plan binding identity
    d6e4f4c45c375c1d8fca6257023436aff16bd5332ca911ed4c61ab1cd193223c — is
    computed exactly as before; recomputing that identity for real requires
    the actual GPU-side checkpoint/calibration inputs, not available on this
    laptop."""
    source = inspect.getsource(v3.build_prediction_plan_binding)
    for forbidden in ("resolve_frozen_target_package_reference", "build_verified_target_loader_config",
                     "target_package_id", "LoaderConfig"):
        assert forbidden not in source


def test_build_prediction_plan_binding_still_constructs_deterministically(monkeypatch, tmp_path) -> None:
    _install_v3_binding_fixtures(monkeypatch)
    first = v3.build_prediction_plan_binding(tmp_path)
    second = v3.build_prediction_plan_binding(tmp_path)
    assert first == second
    assert first["prediction_plan_binding_identity"]


# ==============================================================================
# N/O. V3 protocol identity unchanged; V1/V2 and frozen configs untouched
# ==============================================================================

def test_v3_protocol_identity_unchanged() -> None:
    payload = yaml.safe_load(V3_PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert v3.protocol_identity(payload) == V3_IDENTITY


def test_v1_v2_and_frozen_configs_untouched() -> None:
    import subprocess

    result = subprocess.run(["git", "diff", "--stat", "HEAD", "--",
                            "configs/evaluation/post_failure_exploratory_target_v1.yaml",
                            "configs/evaluation/post_failure_exploratory_target_v2.yaml",
                            "configs/evaluation/post_failure_exploratory_target_v3.yaml",
                            "configs/data/loader_m4.yaml",
                            "src/prism_fas/evaluation/post_failure_exploratory_target.py",
                            "src/prism_fas/evaluation/post_failure_exploratory_target_scorer.py",
                            "src/prism_fas/evaluation/post_failure_exploratory_target_v2.py",
                            "src/prism_fas/evaluation/post_failure_exploratory_target_v2_scorer.py",
                            "src/prism_fas/data/loader/config.py",
                            "src/prism_fas/data/loader/package_index.py",
                            "src/prism_fas/data/loader/loose_dataset.py",
                            "src/prism_fas/evaluation/target_prediction.py",
                            "src/prism_fas/evaluation/source_matrix.py",
                            "src/prism_fas/detector/variant.py"],
                            cwd=str(REPO), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""
