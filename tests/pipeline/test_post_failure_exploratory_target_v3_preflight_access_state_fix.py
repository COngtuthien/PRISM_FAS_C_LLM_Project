"""POST_FAILURE_EXPLORATORY_TARGET_V3 — `_preflight()` access-state
PROVENANCE/REPORTING BUG FIX.

`_preflight()` initialized `report["access_state"] = _access_state()` (all
false) once at the top and never updated it after actually calling
`verify_target_feature_package_expected_v3(...)` — so a REAL GPU preflight
that opened and checked the target feature package (successfully or not)
still reported an all-false access state, contradicting the frozen V3
protocol's own semantics ("after GPU feature-package identity verification:
feature identity true, prediction features false, labels false" — and no
V3 artifact may claim an all-false access state after a real target feature
read).

This file proves the corrected, conservative semantics:
  - package root absent, nothing read  -> feature_identity=false, count=0
  - package present, verified          -> feature_identity=true,  count=1
  - package present, verification FAILS CLOSED (mismatch/tamper/etc.)
    -> feature_identity=true, count=1 — a real read/attempt still happened,
       even though `_preflight` remains BLOCKED
  - prediction-feature access and label access are NEVER claimed by
    `_preflight` under any of the above
  - `checkpoint_weights_loaded` and `images_forwarded` stay `false` always

**FIXTURE / ENGINEERING ONLY.** Every test runs against `tmp_path`
fixtures or a real-but-synthetic on-disk package — never a real GPU
checkpoint, never real target images, never a real target label. No test
here binds a prediction plan or runs `--predict`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.evaluation import post_failure_exploratory_target as v1  # noqa: E402
from prism_fas.evaluation import post_failure_exploratory_target_v2 as v2  # noqa: E402
from prism_fas.evaluation import post_failure_exploratory_target_v3 as v3  # noqa: E402

from test_post_failure_exploratory_target_v3 import (  # noqa: E402
    FAKE_ROWS, _fake_binding_for, _real_v3_protocol,
)
from test_post_failure_exploratory_target_v3_package_identity_fix import (  # noqa: E402
    _build_valid_package, _stub_validate_package,
)

V3_PROTOCOL_PATH = REPO / "configs/evaluation/post_failure_exploratory_target_v3.yaml"
V3_IDENTITY = "a2b54f8844a2a36540e62470c2f5f30de52fbf509a37f03feb7f6d769d5c702c"


def _install_preflight_fixtures(monkeypatch, *, protocol: dict[str, Any] | None = None,
                                rows=FAKE_ROWS) -> None:
    """Mirrors `_install_v3_binding_fixtures` from the base test file, but
    deliberately does NOT monkeypatch the target feature package verifier —
    that is exactly the real code path under test here."""
    monkeypatch.setattr(v3, "load_protocol", lambda repo: protocol if protocol is not None else _real_v3_protocol())
    monkeypatch.setattr(v1, "resolve_target_matrix", lambda repo: list(rows))
    monkeypatch.setattr(v1, "target_matrix_identity",
                        lambda rows_arg: "matrix-" + "".join(sorted(r.row_id for r in rows_arg)))
    monkeypatch.setattr(v1, "bind_c8_matrix_identity", lambda repo: {"c8_matrix_identity": "c8" + "0" * 62})
    monkeypatch.setattr(v1, "verify_target_label_root_sealed",
                        lambda repo, protocol: {"stage": "G7", "label_root_permission": "deny",
                                                "label_root_declared": True, "label_root_exists": False,
                                                "target_labels_opened": False})

    def _resolve_all_row_bindings_v2(repo, rows_arg):
        return {row.row_id: _fake_binding_for(row) for row in rows_arg}

    monkeypatch.setattr(v2, "resolve_all_row_bindings_v2", _resolve_all_row_bindings_v2)


def _protocol_with_package_pin(pin: str) -> dict[str, Any]:
    """An in-memory copy of the REAL frozen protocol with only
    `target_feature_package.expected_identity` overridden — never touches
    the YAML file on disk, used only so a synthetic fixture package can
    verify successfully against a pin it can actually match."""
    protocol = dict(_real_v3_protocol())
    protocol["target_feature_package"] = {**protocol["target_feature_package"], "expected_identity": pin}
    return protocol


# ==============================================================================
# 1. Absent package root -> all false / count 0
# ==============================================================================

def test_absent_package_root_reports_all_false_access_state(monkeypatch, tmp_path) -> None:
    _install_preflight_fixtures(monkeypatch)
    # No package built anywhere under tmp_path: the declared target_feature_root
    # does not exist.
    exit_code, payload = v3._preflight(tmp_path)
    assert payload["target_feature_package"]["present_on_this_host"] is False
    assert payload["access_state"] == {
        "target_feature_identity_accessed": False, "target_prediction_features_accessed": False,
        "target_labels_accessed": False, "target_feature_access_count": 0, "target_label_access_count": 0}
    assert exit_code == v3.EXIT_BLOCKED
    assert payload["checkpoint_weights_loaded"] is False
    assert payload["images_forwarded"] is False


# ==============================================================================
# 2. Present + verified -> feature_identity true / count 1
# ==============================================================================

def test_present_and_verified_package_reports_feature_identity_accessed(monkeypatch, tmp_path) -> None:
    root, lock = _build_valid_package(tmp_path / "data/processed")
    protocol = _protocol_with_package_pin(lock["content_identity_sha256"])
    _install_preflight_fixtures(monkeypatch, protocol=protocol)
    _stub_validate_package(monkeypatch)

    exit_code, payload = v3._preflight(tmp_path)

    assert payload["target_feature_package"]["present_on_this_host"] is True
    assert payload["target_feature_package"]["verified"] is True
    assert payload["target_feature_package"]["computed_identity"] == lock["content_identity_sha256"]
    assert payload["access_state"] == {
        "target_feature_identity_accessed": True, "target_prediction_features_accessed": False,
        "target_labels_accessed": False, "target_feature_access_count": 1, "target_label_access_count": 0}
    assert payload["checkpoint_weights_loaded"] is False
    assert payload["images_forwarded"] is False


# ==============================================================================
# 3. Present + verification FAILS CLOSED -> still feature_identity true /
#    count 1, while preflight remains BLOCKED
# ==============================================================================

def test_present_package_that_fails_verification_still_reports_feature_identity_accessed(
        monkeypatch, tmp_path) -> None:
    # A real, on-disk package IS present at the declared root, but the REAL,
    # UNMODIFIED frozen protocol's pin (c3a29e...) can never match a
    # synthetic fixture's own computed identity -> verification fails
    # closed with a content-identity mismatch.
    _build_valid_package(tmp_path / "data/processed")
    _install_preflight_fixtures(monkeypatch)   # real, unmodified frozen protocol
    _stub_validate_package(monkeypatch)

    exit_code, payload = v3._preflight(tmp_path)

    assert exit_code == v3.EXIT_BLOCKED
    assert payload["target_feature_package"]["verified"] is False
    assert "error" in payload["target_feature_package"]
    assert payload["access_state"] == {
        "target_feature_identity_accessed": True, "target_prediction_features_accessed": False,
        "target_labels_accessed": False, "target_feature_access_count": 1, "target_label_access_count": 0}
    assert payload["ready_for_bind"] is False
    assert payload["checkpoint_weights_loaded"] is False
    assert payload["images_forwarded"] is False


def test_present_package_with_wrong_package_id_still_reports_feature_identity_accessed(
        monkeypatch, tmp_path) -> None:
    """A different fail-closed reason (package_id mismatch, not a content
    identity mismatch) — the same access-state truth must hold regardless
    of WHICH check inside the verifier fails."""
    root, lock = _build_valid_package(tmp_path / "data/processed", package_id="some_other_package")
    protocol = _protocol_with_package_pin(lock["content_identity_sha256"])
    _install_preflight_fixtures(monkeypatch, protocol=protocol)
    _stub_validate_package(monkeypatch)

    exit_code, payload = v3._preflight(tmp_path)

    assert exit_code == v3.EXIT_BLOCKED
    assert payload["target_feature_package"]["verified"] is False
    assert payload["access_state"]["target_feature_identity_accessed"] is True
    assert payload["access_state"]["target_feature_access_count"] == 1


# ==============================================================================
# Never claims prediction-feature or label access; the frozen YAML is
# never mutated; the protocol identity is unchanged
# ==============================================================================

def test_preflight_never_claims_prediction_feature_or_label_access(monkeypatch, tmp_path) -> None:
    for build in (lambda: None,
                 lambda: _build_valid_package(tmp_path / "data/processed")):
        build()
        _install_preflight_fixtures(monkeypatch)
        _stub_validate_package(monkeypatch)
        _, payload = v3._preflight(tmp_path)
        assert payload["access_state"]["target_prediction_features_accessed"] is False
        assert payload["access_state"]["target_labels_accessed"] is False
        assert payload["access_state"]["target_label_access_count"] == 0


def test_preflight_source_never_mutates_the_frozen_yaml_on_disk(monkeypatch, tmp_path) -> None:
    _install_preflight_fixtures(monkeypatch)
    before = V3_PROTOCOL_PATH.read_bytes()
    v3._preflight(tmp_path)
    after = V3_PROTOCOL_PATH.read_bytes()
    assert before == after


def test_v3_protocol_identity_unchanged() -> None:
    payload = yaml.safe_load(V3_PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert v3.protocol_identity(payload) == V3_IDENTITY


def test_v1_and_v2_configs_and_modules_not_modified() -> None:
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


# ==============================================================================
# build_prediction_plan_binding stays correct: feature identity access is
# recorded ONLY after a genuinely successful package verification
# ==============================================================================

def test_build_prediction_plan_binding_still_only_records_access_on_success(monkeypatch, tmp_path) -> None:
    root, lock = _build_valid_package(tmp_path / "data/processed")
    protocol = _protocol_with_package_pin(lock["content_identity_sha256"])
    _install_preflight_fixtures(monkeypatch, protocol=protocol)
    _stub_validate_package(monkeypatch)

    binding = v3.build_prediction_plan_binding(tmp_path)
    assert binding["access_state"] == {
        "target_feature_identity_accessed": True, "target_prediction_features_accessed": False,
        "target_labels_accessed": False, "target_feature_access_count": 1, "target_label_access_count": 0}


def test_build_prediction_plan_binding_raises_rather_than_report_partial_access_on_failure(
        monkeypatch, tmp_path) -> None:
    import pytest

    _build_valid_package(tmp_path / "data/processed")
    _install_preflight_fixtures(monkeypatch)   # real, unmodified pin -> guaranteed mismatch
    _stub_validate_package(monkeypatch)

    with pytest.raises(v3.ExploratoryTargetV3Error):
        v3.build_prediction_plan_binding(tmp_path)
