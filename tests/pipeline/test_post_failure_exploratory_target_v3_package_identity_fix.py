"""POST_FAILURE_EXPLORATORY_TARGET_V3 — target feature package verification
SEMANTIC BUG FIX.

V1's `compute_target_feature_package_identity` (and, transitively, V2's
`verify_target_feature_package_required`, which V3 called) hashed the
WHOLE FILE TREE under the target feature package root and compared that
digest against the frozen pin
`c3a29e695ad08c4b31e01533f1d12374f4e30c51f0167c6622cf8168792e48a8`. That pin
is not a whole-tree digest at all — it is
`PACKAGE_LOCK.json["content_identity_sha256"]`, a stable JSON hash of the
lock's own declared metadata body with five volatile fields excluded
(`created_at`, `git_commit`, `content_identity_sha256`, `build_seconds`,
`environment`) — exactly the algorithm Version B's `modal_m10.py::m10_verify_target_features`
and `data.package.builder.finalize_lock` use. This file proves the
corrected V3-only verifiers (`verify_locked_target_feature_package`,
`verify_target_feature_package_expected_v3`,
`verify_target_feature_package_required_v3`) implement THAT algorithm, fail
closed on every tamper scenario, and that `build_prediction_plan_binding`
now uses them directly instead of V1's/V2's defective whole-tree check.

**FIXTURE / ENGINEERING ONLY.** Every test runs against `tmp_path`
fixtures or pure functions — never a real target feature package, never a
real target label. No test in this file binds a prediction plan for real
or runs `--predict`.
"""
from __future__ import annotations

import hashlib
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

from prism_fas.evaluation import post_failure_exploratory_target as v1  # noqa: E402
from prism_fas.evaluation import post_failure_exploratory_target_v2 as v2  # noqa: E402
from prism_fas.evaluation import post_failure_exploratory_target_v3 as v3  # noqa: E402

from test_post_failure_exploratory_target_v3 import (  # noqa: E402
    _install_v3_binding_fixtures, _real_v3_protocol,
)

V3_PROTOCOL_PATH = REPO / "configs/evaluation/post_failure_exploratory_target_v3.yaml"
V3_IDENTITY = "a2b54f8844a2a36540e62470c2f5f30de52fbf509a37f03feb7f6d769d5c702c"
EXPECTED_TARGET_FEATURE_IDENTITY = "c3a29e695ad08c4b31e01533f1d12374f4e30c51f0167c6622cf8168792e48a8"
EXPECTED_TARGET_PACKAGE_ID = "prism_target_eval_v2"


def _stub_validate_package(monkeypatch, *, passed: bool = True, errors=None) -> None:
    """`validate_package` (general M3A/M3B structural validator) has its own
    test suite; here it is reused, never duplicated, so tests targeting the
    NEW content-identity logic stub it to isolate that logic."""
    monkeypatch.setattr("prism_fas.data.package.validator.validate_package",
                        lambda root, **kwargs: {"passed": passed, "errors": errors or []})


def _build_valid_package(tmp_path: Path, *, package_id: str = EXPECTED_TARGET_PACKAGE_ID
                         ) -> tuple[Path, dict[str, Any]]:
    """A minimal, REAL, on-disk locked package: a real PACKAGE_LOCK.json
    whose `content_identity_sha256` is computed by the exact historical
    algorithm, real (empty) manifests satisfying `assert_features_label_free`/
    `assert_no_target_identity` for real, and one real shard file whose
    hash/size the lock declares."""
    from prism_fas.data.package.manifests import SAMPLES_SCHEMA, TARGET_FEATURES_SCHEMA, write_manifest

    root = tmp_path / "prism_target_eval_v2"
    metadata = {"package_schema_version": "target-eval-v1"}
    features_sha = write_manifest(root / "manifests" / "target_test_features.parquet", [],
                                  TARGET_FEATURES_SCHEMA, metadata)
    samples_sha = write_manifest(root / "manifests" / "samples.parquet", [], SAMPLES_SCHEMA, metadata)

    shard_bytes = b"fake-shard-bytes-for-a-fixture-package"
    shard_path = root / "shards" / "target_test-00000.tar"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_bytes(shard_bytes)

    body = {"package_id": package_id, "status": "validated",
           "manifest_sha256": {"target_test_features": features_sha, "samples": samples_sha},
           "shards": [{"shard_filename": "target_test-00000.tar",
                      "sha256": hashlib.sha256(shard_bytes).hexdigest(),
                      "byte_size": len(shard_bytes), "row_count": 0}],
           "total_samples": 0, "package_schema_version": "target-eval-v1",
           "created_at": "2026-01-01T00:00:00Z", "git_commit": "d" * 40,
           "build_seconds": 12.3, "environment": {"host": "gpu-fixture"}}
    content_identity = v3.recompute_package_lock_content_identity(body)
    lock = {**body, "content_identity_sha256": content_identity}
    (root / "PACKAGE_LOCK.json").write_text(json.dumps(lock), encoding="utf-8")
    return root, lock


# ==============================================================================
# The historical algorithm, reused verbatim
# ==============================================================================

def test_recompute_excludes_exactly_the_five_historical_volatile_fields() -> None:
    assert v3.V3_PACKAGE_LOCK_IDENTITY_EXCLUDED_FIELDS == (
        "created_at", "git_commit", "content_identity_sha256", "build_seconds", "environment")


def test_recompute_is_deterministic_and_field_order_independent() -> None:
    body = {"package_id": "prism_target_eval_v2", "status": "validated", "a": 1, "b": [1, 2, 3]}
    reordered = {"b": [1, 2, 3], "status": "validated", "a": 1, "package_id": "prism_target_eval_v2"}
    assert v3.recompute_package_lock_content_identity(body) == v3.recompute_package_lock_content_identity(reordered)


# ==============================================================================
# Valid locked package passes
# ==============================================================================

def test_valid_locked_package_passes(monkeypatch, tmp_path) -> None:
    root, lock = _build_valid_package(tmp_path)
    _stub_validate_package(monkeypatch)
    result = v3.verify_locked_target_feature_package(
        root, expected_package_id=EXPECTED_TARGET_PACKAGE_ID,
        expected_content_identity=lock["content_identity_sha256"])
    assert result["present_on_this_host"] is True
    assert result["verified"] is True
    assert result["computed_identity"] == lock["content_identity_sha256"]
    assert result["identity_self_consistent"] is True
    assert result["identity_matches_pin"] is True
    assert result["label_free"]["labels_present"] is False
    assert result["no_target_identity"]["target_identity_embeddings"] == 0


def test_absent_package_root_returns_not_present_without_raising(tmp_path) -> None:
    result = v3.verify_locked_target_feature_package(
        tmp_path / "does-not-exist", expected_package_id=EXPECTED_TARGET_PACKAGE_ID,
        expected_content_identity=EXPECTED_TARGET_FEATURE_IDENTITY)
    assert result == {"present_on_this_host": False, "verified": False,
                      "expected_identity": EXPECTED_TARGET_FEATURE_IDENTITY, "computed_identity": None,
                      "reason": "NOT_PRESENT_ON_THIS_HOST"}


# ==============================================================================
# Expected pin mismatch fails closed
# ==============================================================================

def test_expected_pin_mismatch_fails_closed(monkeypatch, tmp_path) -> None:
    root, lock = _build_valid_package(tmp_path)
    _stub_validate_package(monkeypatch)
    with pytest.raises(v3.ExploratoryTargetV3Error, match="content identity mismatch"):
        v3.verify_locked_target_feature_package(
            root, expected_package_id=EXPECTED_TARGET_PACKAGE_ID, expected_content_identity="0" * 64)


# ==============================================================================
# Lock self-identity tampering fails closed
# ==============================================================================

def test_lock_self_identity_tampering_fails_closed(monkeypatch, tmp_path) -> None:
    root, lock = _build_valid_package(tmp_path)
    _stub_validate_package(monkeypatch)
    tampered = {**lock, "content_identity_sha256": "f" * 64}
    (root / "PACKAGE_LOCK.json").write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(v3.ExploratoryTargetV3Error, match="does not hash to its own recorded"):
        v3.verify_locked_target_feature_package(
            root, expected_package_id=EXPECTED_TARGET_PACKAGE_ID,
            expected_content_identity=lock["content_identity_sha256"])


# ==============================================================================
# Manifest tampering fails closed
# ==============================================================================

def test_manifest_tampering_fails_closed(monkeypatch, tmp_path) -> None:
    root, lock = _build_valid_package(tmp_path)
    _stub_validate_package(monkeypatch)
    (root / "manifests" / "samples.parquet").write_bytes(b"TAMPERED-MANIFEST-BYTES")
    with pytest.raises(v3.ExploratoryTargetV3Error, match="manifest integrity failed"):
        v3.verify_locked_target_feature_package(
            root, expected_package_id=EXPECTED_TARGET_PACKAGE_ID,
            expected_content_identity=lock["content_identity_sha256"])


# ==============================================================================
# Shard tampering fails closed
# ==============================================================================

def test_shard_tampering_fails_closed(monkeypatch, tmp_path) -> None:
    root, lock = _build_valid_package(tmp_path)
    _stub_validate_package(monkeypatch)
    (root / "shards" / "target_test-00000.tar").write_bytes(b"TAMPERED-SHARD-BYTES")
    with pytest.raises(v3.ExploratoryTargetV3Error, match="shard integrity failed"):
        v3.verify_locked_target_feature_package(
            root, expected_package_id=EXPECTED_TARGET_PACKAGE_ID,
            expected_content_identity=lock["content_identity_sha256"])


def test_missing_shard_fails_closed(monkeypatch, tmp_path) -> None:
    root, lock = _build_valid_package(tmp_path)
    _stub_validate_package(monkeypatch)
    (root / "shards" / "target_test-00000.tar").unlink()
    with pytest.raises(v3.ExploratoryTargetV3Error, match="shard integrity failed"):
        v3.verify_locked_target_feature_package(
            root, expected_package_id=EXPECTED_TARGET_PACKAGE_ID,
            expected_content_identity=lock["content_identity_sha256"])


# ==============================================================================
# Package-id mismatch fails closed
# ==============================================================================

def test_package_id_mismatch_fails_closed(monkeypatch, tmp_path) -> None:
    root, lock = _build_valid_package(tmp_path, package_id="some_other_package")
    _stub_validate_package(monkeypatch)
    with pytest.raises(v3.ExploratoryTargetV3Error, match="package_id"):
        v3.verify_locked_target_feature_package(
            root, expected_package_id=EXPECTED_TARGET_PACKAGE_ID,
            expected_content_identity=lock["content_identity_sha256"])


def test_non_validated_status_fails_closed(monkeypatch, tmp_path) -> None:
    root, lock = _build_valid_package(tmp_path)
    _stub_validate_package(monkeypatch)
    body = {k: v for k, v in lock.items() if k not in ("content_identity_sha256",)}
    body["status"] = "building"
    body["content_identity_sha256"] = v3.recompute_package_lock_content_identity(body)
    (root / "PACKAGE_LOCK.json").write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(v3.ExploratoryTargetV3Error, match="not 'validated'"):
        v3.verify_locked_target_feature_package(
            root, expected_package_id=EXPECTED_TARGET_PACKAGE_ID,
            expected_content_identity=body["content_identity_sha256"])


def test_missing_package_lock_fails_closed(tmp_path) -> None:
    root = tmp_path / "present_but_empty"
    root.mkdir()
    with pytest.raises(v3.ExploratoryTargetV3Error, match="PACKAGE_LOCK.json"):
        v3.verify_locked_target_feature_package(
            root, expected_package_id=EXPECTED_TARGET_PACKAGE_ID,
            expected_content_identity=EXPECTED_TARGET_FEATURE_IDENTITY)


def test_structural_validation_failure_fails_closed(monkeypatch, tmp_path) -> None:
    root, lock = _build_valid_package(tmp_path)
    _stub_validate_package(monkeypatch, passed=False, errors=[{"check_id": "fixture.failure"}])
    with pytest.raises(v3.ExploratoryTargetV3Error, match="structural validation"):
        v3.verify_locked_target_feature_package(
            root, expected_package_id=EXPECTED_TARGET_PACKAGE_ID,
            expected_content_identity=lock["content_identity_sha256"])


# ==============================================================================
# V3 uses the LOCK CONTENT identity, never a whole-tree digest — the core fix
# ==============================================================================

def test_computed_identity_is_the_lock_content_identity_not_a_whole_tree_hash(monkeypatch, tmp_path) -> None:
    root, lock = _build_valid_package(tmp_path)
    _stub_validate_package(monkeypatch)
    result = v3.verify_locked_target_feature_package(
        root, expected_package_id=EXPECTED_TARGET_PACKAGE_ID,
        expected_content_identity=lock["content_identity_sha256"])
    whole_tree_hash = v1.compute_target_feature_package_identity(root)
    assert result["computed_identity"] == lock["content_identity_sha256"]
    assert result["computed_identity"] != whole_tree_hash


def test_a_byte_change_that_leaves_the_lock_body_untouched_does_not_change_the_verified_identity(
        monkeypatch, tmp_path) -> None:
    """A whole-tree hash would change if ANY file byte changes. The real
    contract must not: touching a byte that is not referenced by the lock's
    manifest/shard hashes (e.g. adding an unrelated file) leaves the LOCK
    content identity — and therefore `computed_identity` — unchanged."""
    root, lock = _build_valid_package(tmp_path)
    _stub_validate_package(monkeypatch)
    before = v3.verify_locked_target_feature_package(
        root, expected_package_id=EXPECTED_TARGET_PACKAGE_ID,
        expected_content_identity=lock["content_identity_sha256"])
    (root / "an_unrelated_scratch_file.txt").write_text("not part of any manifest or shard", encoding="utf-8")
    after = v3.verify_locked_target_feature_package(
        root, expected_package_id=EXPECTED_TARGET_PACKAGE_ID,
        expected_content_identity=lock["content_identity_sha256"])
    assert before["computed_identity"] == after["computed_identity"]


# ==============================================================================
# The protocol wrapper resolves ONLY the frozen V3 declaration
# ==============================================================================

def test_protocol_wrapper_resolves_root_and_pin_from_the_frozen_protocol(tmp_path) -> None:
    """No package is built on disk at the protocol's declared root, so this
    proves — without any risk of a real fixture accidentally matching the
    frozen pin — that the wrapper resolves BOTH the target feature root AND
    the expected content identity from the frozen V3 protocol itself."""
    protocol = _real_v3_protocol()
    result = v3.verify_target_feature_package_expected_v3(tmp_path, protocol)
    assert result["present_on_this_host"] is False
    assert result["verified"] is False
    assert result["expected_identity"] == EXPECTED_TARGET_FEATURE_IDENTITY
    assert result["reason"] == "NOT_PRESENT_ON_THIS_HOST"


def test_required_v3_raises_when_not_present_and_verified(tmp_path) -> None:
    protocol = _real_v3_protocol()
    with pytest.raises(v3.ExploratoryTargetV3Error, match="not present-and-verified"):
        v3.verify_target_feature_package_required_v3(tmp_path, protocol)


def test_v3_package_verifiers_never_reference_target_label_root() -> None:
    for func in (v3.verify_locked_target_feature_package, v3.verify_target_feature_package_expected_v3,
                v3.verify_target_feature_package_required_v3):
        source = inspect.getsource(func)
        assert '"target_label_root"' not in source
        assert "load_evaluation_labels" not in source
        assert "siw_target_labels" not in source


# ==============================================================================
# build_prediction_plan_binding uses the NEW verifier, never V2's defective one
# ==============================================================================

def test_build_prediction_plan_binding_source_never_calls_v2_verifier() -> None:
    source = inspect.getsource(v3.build_prediction_plan_binding)
    assert "v2.verify_target_feature_package_required" not in source
    assert "verify_target_feature_package_required_v3" in source


def test_preflight_source_never_calls_v1_verifier() -> None:
    source = inspect.getsource(v3._preflight)
    assert "v1.verify_target_feature_package_expected" not in source
    assert "verify_target_feature_package_expected_v3" in source


def test_build_prediction_plan_binding_never_actually_invokes_v2_defective_verifier(monkeypatch, tmp_path) -> None:
    def _must_not_be_called(repo, protocol):
        raise AssertionError("V2's defective whole-tree verifier must never be called from V3")

    monkeypatch.setattr(v2, "verify_target_feature_package_required", _must_not_be_called)
    _install_v3_binding_fixtures(monkeypatch)
    binding = v3.build_prediction_plan_binding(tmp_path)
    assert binding["target_feature_package_identity"]


# ==============================================================================
# V1 and V2 files are not modified; frozen V3 YAML/identity unchanged
# ==============================================================================

def test_v1_and_v2_modules_and_all_configs_not_modified() -> None:
    import subprocess

    result = subprocess.run(["git", "diff", "--stat", "HEAD", "--",
                            "configs/evaluation/post_failure_exploratory_target_v1.yaml",
                            "configs/evaluation/post_failure_exploratory_target_v2.yaml",
                            "configs/evaluation/post_failure_exploratory_target_v3.yaml",
                            "src/prism_fas/evaluation/post_failure_exploratory_target.py",
                            "src/prism_fas/evaluation/post_failure_exploratory_target_scorer.py",
                            "src/prism_fas/evaluation/post_failure_exploratory_target_v2.py",
                            "src/prism_fas/evaluation/post_failure_exploratory_target_v2_scorer.py"],
                            cwd=str(REPO), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_v3_protocol_identity_unchanged() -> None:
    assert v3.protocol_identity(_real_v3_protocol()) == V3_IDENTITY


def test_v1_compute_target_feature_package_identity_itself_unchanged() -> None:
    """V1's (defective, but historically-frozen-as-evidence) whole-tree
    verifier is left byte-for-byte alone — this task fixes only V3's OWN
    verification path, never V1's function body."""
    source = inspect.getsource(v1.compute_target_feature_package_identity)
    assert "rglob" in source and "sha256" in source
