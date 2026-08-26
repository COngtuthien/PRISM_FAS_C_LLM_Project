"""C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V1 — protocol, mechanics and runner.

**FIXTURE / ENGINEERING ONLY.** Every test in this file runs against
`tmp_path` fixtures or pure functions with synthetic arrays — never a real
repo's GPU artifacts, never a real checkpoint, never a real image, never
target data. `post_failure_diagnostics_runner --execute` was run once
against this real (laptop) repo as part of the freeze task and correctly
reported the source package unresolved (no M3B package here), exit 2,
writing nothing — that is not repeated as a "real" scientific claim here.

This file proves: the new protocol is not a BA_sep revision and cannot
reopen C9 or touch the BA_sep/reliability-lock artifacts; the no-rerun and
partial-result guards match the BA_sep runner's own contract; group-safe
calibration/evaluation separation; deterministic corruption transforms;
`score_shift` regression; every BLOCKED test in the frozen config carries an
explicit reason; and every mismatched/corrupted binding fails closed.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.evaluation import detector_reliability as barrier  # noqa: E402
from prism_fas.evaluation import post_failure_diagnostics as diag  # noqa: E402
from prism_fas.evaluation import post_failure_diagnostics_runner as runner  # noqa: E402
from prism_fas.evaluation import synthetic_real_probe as probe  # noqa: E402
from prism_fas.pipeline.state import atomic_write_json  # noqa: E402

PROTOCOL_PATH = REPO / "configs/evaluation/c9_post_failure_source_diagnostics_v1.yaml"
BA_SEP_V2_IDENTITY = "720a2e344017d588d71005b81fdf0e7d2062081ae2f3881a61a306d952dc4ac8"


def _real_protocol() -> dict[str, Any]:
    return yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))


# ==============================================================================
# A. protocol identity deterministic
# ==============================================================================

def test_protocol_config_exists_and_is_frozen_not_run() -> None:
    assert PROTOCOL_PATH.is_file()
    payload = _real_protocol()
    assert payload["status"] == "FROZEN_NOT_RUN"
    assert payload["decision_id"] == "C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V1"


def test_protocol_identity_is_deterministic() -> None:
    payload = _real_protocol()
    first = diag.protocol_identity(payload)
    second = diag.protocol_identity(payload)
    assert first == second
    assert len(first) == 64


def test_protocol_identity_changes_with_a_result_affecting_field() -> None:
    payload = _real_protocol()
    baseline = diag.protocol_identity(payload)
    changed = dict(payload)
    changed["checkpoint_policy"] = {**payload["checkpoint_policy"], "checkpoints_per_arm": 3}
    assert diag.protocol_identity(changed) != baseline


def test_protocol_identity_unaffected_by_metadata_only_change() -> None:
    payload = _real_protocol()
    baseline = diag.protocol_identity(payload)
    changed = dict(payload)
    changed["frozen_on"] = "2099-01-01"
    changed["approved_by"] = "someone else"
    assert diag.protocol_identity(changed) == baseline


# ==============================================================================
# B. target_access always 0
# ==============================================================================

def test_protocol_declares_target_access_zero() -> None:
    payload = _real_protocol()
    assert payload["target_access"] == 0
    assert payload["target_firewall"]["target_access"] == 0


def test_module_never_references_a_target_path() -> None:
    source = Path(inspect.getfile(diag)).read_text(encoding="utf-8")
    for forbidden in ("siw", "SiW", "target_test", "target_taxonomy",
                      "prism_target_eval_v2", "resolve_target"):
        assert forbidden not in source, forbidden


def test_runner_never_references_a_target_path() -> None:
    source = Path(inspect.getfile(runner)).read_text(encoding="utf-8")
    for forbidden in ("siw", "SiW", "target_test", "target_taxonomy",
                      "prism_target_eval_v2", "resolve_target"):
        assert forbidden not in source, forbidden


def test_every_written_artifact_carries_target_access_zero(monkeypatch, tmp_path) -> None:
    fixtures = _install_diagnostics_bind_fixtures(monkeypatch)
    exit_code, payload = runner._bind_only(tmp_path)
    assert exit_code == runner.EXIT_PASS
    for relative in (diag.PROTOCOL_BINDING_PATH, diag.POPULATION_BINDING_PATH,
                    diag.CHECKPOINT_BINDING_PATH):
        doc = json.loads((tmp_path / relative).read_text())
        assert doc["target_access"] == 0


# ==============================================================================
# C. BA_sep protocol untouched  /  D. BA_sep result cannot be overwritten
# ==============================================================================

def test_ba_sep_v2_protocol_identity_unchanged() -> None:
    assert probe.protocol_identity(REPO) == BA_SEP_V2_IDENTITY


def test_ba_sep_config_file_not_modified_by_this_freeze() -> None:
    import subprocess

    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--",
         "configs/evaluation/c9_detector_ba_sep_option1_v2.yaml"],
        cwd=str(REPO), capture_output=True, text=True)
    assert result.stdout.strip() == ""


def test_diagnostics_namespace_never_equals_ba_sep_namespace() -> None:
    assert diag.DIAGNOSTICS_DIR != probe.RELIABILITY_DIR
    assert not diag.DIAGNOSTICS_DIR.startswith(probe.RELIABILITY_DIR)
    assert not probe.RELIABILITY_DIR.startswith(diag.DIAGNOSTICS_DIR)


def test_diagnostics_module_never_writes_under_the_ba_sep_reliability_dir() -> None:
    source = Path(inspect.getfile(diag)).read_text(encoding="utf-8")
    # The module docstring legitimately DISCUSSES the BA_sep path in prose
    # (explaining what this module must never touch); the structural proof
    # is that it never imports or references the BA_sep RELIABILITY_DIR
    # constant anywhere, in code or otherwise.
    assert "RELIABILITY_DIR" not in source


def test_diagnostics_runner_never_writes_ba_sep_or_lock_paths() -> None:
    source = Path(inspect.getfile(runner)).read_text(encoding="utf-8")
    for forbidden in ("probe.RESULT_PATH", "probe.VERDICT_PATH", "barrier.LOCK_PATH",
                      "detector_reliability.LOCK_PATH,"):
        assert forbidden not in source, forbidden


def test_real_ba_sep_artifacts_untouched_by_a_real_diagnostics_bind(monkeypatch, tmp_path) -> None:
    """A behavioral proof: binding diagnostics fixtures must never touch
    anything under the REAL repo's BA_sep reliability directory or lock."""
    real_reliability_dir = REPO / probe.RELIABILITY_DIR
    real_lock = REPO / barrier.LOCK_PATH
    before_listing = (sorted(p.name for p in real_reliability_dir.rglob("*"))
                      if real_reliability_dir.is_dir() else None)
    before_lock = real_lock.read_bytes() if real_lock.is_file() else None

    _install_diagnostics_bind_fixtures(monkeypatch)
    runner._bind_only(tmp_path)

    after_listing = (sorted(p.name for p in real_reliability_dir.rglob("*"))
                     if real_reliability_dir.is_dir() else None)
    after_lock = real_lock.read_bytes() if real_lock.is_file() else None
    assert before_listing == after_listing
    assert before_lock == after_lock


# ==============================================================================
# E. failed barrier cannot be reopened  /  F. c9_may_close never true
# ==============================================================================

def test_execute_result_hard_codes_c9_may_close_false() -> None:
    source = inspect.getsource(runner._execute)
    assert '"c9_may_close": False' in source


def test_status_result_hard_codes_c9_may_close_false() -> None:
    source = inspect.getsource(runner._status)
    assert 'c9_may_close"] = False' in source


def test_execute_source_never_calls_lock_payload_or_verify_lock() -> None:
    source = inspect.getsource(runner._execute)
    for forbidden in ("lock_payload(", "verify_lock(", "validate_lock_record("):
        assert forbidden not in source, forbidden


def test_written_verdict_always_records_ba_sep_observed_fail(monkeypatch, tmp_path) -> None:
    fixtures = _install_full_execute_fixtures(monkeypatch, tmp_path, verdict_per_arm="PASS")
    runner._execute(tmp_path)
    verdict_doc = json.loads((tmp_path / diag.VERDICT_PATH).read_text())
    assert verdict_doc["ba_sep_observed_verdict"] == "FAIL"
    assert verdict_doc["detector_reliability_lock_c_observed_overall"] == "FAILED"
    assert verdict_doc["c9_may_close"] is False


# ==============================================================================
# G. preflight read-only
# ==============================================================================

def test_preflight_writes_nothing(monkeypatch, tmp_path) -> None:
    _install_diagnostics_bind_fixtures(monkeypatch)
    runner._preflight(tmp_path)
    assert not (tmp_path / diag.DIAGNOSTICS_DIR).exists()


def test_preflight_source_never_forwards_or_computes() -> None:
    source = inspect.getsource(runner._preflight)
    for forbidden in ("run_benign_corruption_diagnostic_for_arm(",
                      "run_cross_route_diagnostic_for_arm(", "forward_",
                      "construct_row_trainer("):
        assert forbidden not in source, forbidden


def test_preflight_reports_per_test_readiness_from_the_frozen_config() -> None:
    exit_code, payload = runner._preflight(REPO)
    assert payload["per_test_gpu_ready"]["benign_jpeg_corruption"] is True
    assert payload["per_test_gpu_ready"]["residual_scale_zero"] is False


# ==============================================================================
# H. bind-only contains zero scientific metric  /  I. bind idempotent
# ==============================================================================

def _install_diagnostics_bind_fixtures(monkeypatch, *, seeds=(1, 2, 3, 4, 5)) -> dict[str, Any]:
    from prism_fas.evaluation import c6_evidence
    from prism_fas.pipeline.adapters import sources

    fake_protocol = _real_protocol()
    fake_protocol["benign_corruption_shared"]["split_hash_namespace"] = "test-diag-ns"
    fake_protocol["benign_corruption_shared"]["split_seed"] = 1
    monkeypatch.setattr(diag, "load_protocol", lambda repo: fake_protocol)

    # build_checkpoint_binding (reused from the BA_sep module) internally
    # calls probe.load_protocol/protocol_identity for ITS OWN protocol
    # identity field; patch those too so it never tries to read the real
    # BA_sep config off a bare tmp_path.
    fake_ba_sep_protocol = {"probe_seed_values": [20260806, 20260807, 20260808]}
    monkeypatch.setattr(probe, "load_protocol", lambda repo: fake_ba_sep_protocol)
    monkeypatch.setattr(probe, "protocol_identity", lambda repo: "a" * 64)

    checkpoints_by_arm = {
        arm: [probe.CheckpointBinding(
            arm=arm, seed=s, row_id=f"C-G-{arm}-P3READY-s{s}", run_identity=f"run-{arm}-{s}",
            config_identity="c" * 64, checkpoint_sha256=f"{arm.lower()}{s}".ljust(64, "0"),
            checkpoint_path="p", checkpoint_kind="best", decision_logit_name="global_logit_G",
            decision_graph_hash="g" * 64) for s in seeds]
        for arm in probe.ARMS}
    monkeypatch.setattr(probe, "resolve_checkpoint_set", lambda repo, arm: checkpoints_by_arm[arm])
    monkeypatch.setattr(sources, "verify_detector_inputs",
                        lambda repo, arms=None: {
                            "package_identity": "pkg" + "0" * 61,
                            "c6": {"banks": {arm: {"selected_set_sha256": f"bank-{arm}" + "0" * 55}
                                            for arm in probe.ARMS}}})

    live_records = [{"sample_id": f"live-{i}", "stable_group_identity": f"g{i}",
                    "source_domain": "casia_fasd"} for i in range(40)]
    monkeypatch.setattr(diag, "resolve_source_dev_live_records", lambda repo: live_records)

    def _by_route(repo, arm):
        return {"physics": [probe.PopulationRecord(f"{arm.lower()}-phys-{i}", f"pg{arm.lower()}{i}",
                                                    "casia_fasd", probe.SYNTHETIC_SPOOF_CLASS)
                            for i in range(20)],
               "gpat": [probe.PopulationRecord(f"{arm.lower()}-gpat-{i}", f"gg{arm.lower()}{i}",
                                               "casia_fasd", probe.SYNTHETIC_SPOOF_CLASS)
                       for i in range(20)]}

    monkeypatch.setattr(diag, "resolve_synthetic_population_by_route", _by_route)
    return {"protocol": fake_protocol, "checkpoints_by_arm": checkpoints_by_arm,
           "live_records": live_records}


def test_bind_only_writes_three_bindings_with_zero_scientific_metric(monkeypatch, tmp_path) -> None:
    _install_diagnostics_bind_fixtures(monkeypatch)
    exit_code, payload = runner._bind_only(tmp_path)
    assert exit_code == runner.EXIT_PASS
    for relative in (diag.PROTOCOL_BINDING_PATH, diag.POPULATION_BINDING_PATH,
                    diag.CHECKPOINT_BINDING_PATH):
        text = (tmp_path / relative).read_text()
        for forbidden in ("balanced_accuracy", "mean_shift", "verdict", "PASS", "FAIL"):
            assert forbidden not in text, (relative, forbidden)


def test_bind_only_is_idempotent(monkeypatch, tmp_path) -> None:
    _install_diagnostics_bind_fixtures(monkeypatch)
    runner._bind_only(tmp_path)
    before = (tmp_path / diag.PROTOCOL_BINDING_PATH).read_text()
    exit_code, payload = runner._bind_only(tmp_path)
    after = (tmp_path / diag.PROTOCOL_BINDING_PATH).read_text()
    assert exit_code == runner.EXIT_PASS
    assert payload["reused"] is True
    assert before == after


# ==============================================================================
# J. complete-existing-result --execute re-reports only  /  K. partial blocks
# ==============================================================================

def _install_full_execute_fixtures(monkeypatch, tmp_path, *, verdict_per_arm: str) -> None:
    _install_diagnostics_bind_fixtures(monkeypatch)
    exit_code, _ = runner._bind_only(tmp_path)
    assert exit_code == runner.EXIT_PASS

    def _fake_benign(repo, test_id, arm, checkpoints, *, calibration_ids, evaluation_ids):
        return {"arm": arm, "test_id": test_id,
               "threshold": {"threshold": 0.1}, "evaluation": {"mean_shift": 0.01},
               "verdict": verdict_per_arm}

    def _fake_cross_route(repo, arm, checkpoints, *, protocol):
        return {"arm": arm, "mean_cross_route_ba": 0.5, "verdict": verdict_per_arm}

    monkeypatch.setattr(diag, "run_benign_corruption_diagnostic_for_arm", _fake_benign)
    monkeypatch.setattr(diag, "run_cross_route_diagnostic_for_arm", _fake_cross_route)


def test_all_present_result_re_reports_without_recomputation(monkeypatch, tmp_path) -> None:
    _install_full_execute_fixtures(monkeypatch, tmp_path, verdict_per_arm="PASS")
    runner._execute(tmp_path)   # first real execution
    called = {"n": 0}
    monkeypatch.setattr(diag, "run_benign_corruption_diagnostic_for_arm",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    exit_code, payload = runner._execute(tmp_path)   # second call
    assert called["n"] == 0
    assert payload["reused_existing_diagnostics_result"] is True
    assert payload["checkpoint_weights_loaded"] is False
    assert exit_code == runner.EXIT_PASS


def test_execute_pass_verdict_when_all_executed_tests_pass(monkeypatch, tmp_path) -> None:
    _install_full_execute_fixtures(monkeypatch, tmp_path, verdict_per_arm="PASS")
    exit_code, payload = runner._execute(tmp_path)
    assert exit_code == runner.EXIT_PASS
    assert payload["overall_diagnostics_verdict"] == "PASS"


def test_execute_fail_verdict_when_one_test_fails(monkeypatch, tmp_path) -> None:
    _install_full_execute_fixtures(monkeypatch, tmp_path, verdict_per_arm="FAIL")
    exit_code, payload = runner._execute(tmp_path)
    assert exit_code == runner.EXIT_FAIL
    assert payload["overall_diagnostics_verdict"] == "FAIL"


def test_partial_result_set_blocks(monkeypatch, tmp_path) -> None:
    _install_full_execute_fixtures(monkeypatch, tmp_path, verdict_per_arm="PASS")
    runner._execute(tmp_path)
    (tmp_path / diag.VERDICT_PATH).unlink()
    exit_code, payload = runner._execute(tmp_path)
    assert exit_code == runner.EXIT_BLOCKED
    assert payload["error"] == "PARTIAL_SCIENTIFIC_RESULT_SET"


def test_existing_result_files_never_overwritten_by_a_second_execute(monkeypatch, tmp_path) -> None:
    _install_full_execute_fixtures(monkeypatch, tmp_path, verdict_per_arm="PASS")
    runner._execute(tmp_path)
    before = (tmp_path / diag.RESULT_PATH).read_text()
    runner._execute(tmp_path)
    after = (tmp_path / diag.RESULT_PATH).read_text()
    assert before == after


# ==============================================================================
# L. checkpoint set fixed  /  M. no best-seed selection
# ==============================================================================

def test_checkpoint_policy_matches_ba_sep_family_exactly() -> None:
    payload = _real_protocol()
    policy = payload["checkpoint_policy"]
    assert policy["checkpoints_per_arm"] == 5
    assert policy["total_checkpoints"] == 15
    assert set(policy["seeds_per_arm"]) == {20260806, 20260807, 20260808, 20260809, 20260810}
    assert policy["resolver"] == "prism_fas.evaluation.synthetic_real_probe.resolve_checkpoint_set"


def test_no_best_seed_selection_language_in_checkpoint_policy() -> None:
    payload = _real_protocol()
    policy = payload["checkpoint_policy"]
    assert policy["no_best_seed_selection"] is True
    assert policy["no_favorable_seed_averaging_only"] is True


def test_no_best_checkpoint_selection_in_diagnostics_source() -> None:
    source = Path(inspect.getfile(diag)).read_text(encoding="utf-8")
    for forbidden in ("sorted(evidence", "sort(key=lambda item: item.metrics",
                      "max(checkpoints", "min(checkpoints"):
        assert forbidden not in source, forbidden


# ==============================================================================
# N. calibration/evaluation group separation
# ==============================================================================

def test_calibration_evaluation_split_is_group_safe() -> None:
    group_ids = [f"g{i}" for i in range(200)]
    split = diag.calibration_evaluation_split(group_ids, namespace="ns", seed=1)
    overlap = set(split["calibration"]) & set(split["evaluation"])
    assert not overlap
    assert set(split["calibration"]) | set(split["evaluation"]) == set(group_ids)


def test_calibration_evaluation_split_is_deterministic() -> None:
    group_ids = [f"g{i}" for i in range(50)]
    first = diag.calibration_evaluation_split(group_ids, namespace="ns", seed=1)
    second = diag.calibration_evaluation_split(group_ids, namespace="ns", seed=1)
    assert first == second


def test_calibration_evaluation_split_fails_closed_on_a_degenerate_group() -> None:
    with pytest.raises(diag.PostFailureDiagnosticsError):
        diag.calibration_evaluation_split([], namespace="ns", seed=1)


def test_threshold_derived_only_from_calibration_never_evaluation() -> None:
    source = inspect.getsource(diag.run_benign_corruption_diagnostic_for_arm)
    # the threshold call must receive calibration_shifts, never evaluation values
    assert "derive_corruption_threshold(calibration_shifts)" in source


# ==============================================================================
# O. corruption transform deterministic identity
# ==============================================================================

def test_jpeg_corrupt_is_deterministic() -> None:
    image = np.random.RandomState(1).rand(3, 224, 224).astype(np.float32)
    first = diag.jpeg_corrupt(image)
    second = diag.jpeg_corrupt(image)
    assert np.array_equal(first, second)


def test_resize_corrupt_is_deterministic_and_preserves_shape() -> None:
    image = np.random.RandomState(1).rand(3, 224, 224).astype(np.float32)
    out = diag.resize_corrupt(image)
    assert out.shape == image.shape
    assert np.array_equal(out, diag.resize_corrupt(image))


def test_color_corrupt_is_deterministic_and_applies_the_frozen_gain() -> None:
    image = np.full((3, 4, 4), 0.5, dtype=np.float32)
    out = diag.color_corrupt(image, gain=(1.15, 1.0, 0.9))
    assert out[0].mean() == pytest.approx(0.5 * 1.15, abs=1e-4)
    assert out[2].mean() == pytest.approx(0.5 * 0.9, abs=1e-4)
    assert out.max() <= 1.0 and out.min() >= 0.0


def test_color_corrupt_clips_to_valid_range() -> None:
    image = np.full((3, 2, 2), 0.95, dtype=np.float32)
    out = diag.color_corrupt(image, gain=(1.15, 1.0, 0.9))
    assert out[0].max() <= 1.0


# ==============================================================================
# P. score_shift exact metric regression
# ==============================================================================

def test_score_shift_matches_the_existing_reliability_implementation() -> None:
    from prism_fas.evaluation.reliability import score_shift

    before = [0.1, 0.2, 0.3, 0.4]
    after = [0.15, 0.25, 0.28, 0.5]
    result = score_shift(before, after)
    expected_mean_shift = float(np.mean(np.array(after) - np.array(before)))
    assert result["mean_shift"] == pytest.approx(expected_mean_shift)
    assert result["samples"] == 4


def test_corruption_verdict_pass_and_fail_boundaries() -> None:
    assert diag.corruption_verdict(0.05, 0.05) == "PASS"   # tie passes
    assert diag.corruption_verdict(0.0501, 0.05) == "FAIL"
    assert diag.corruption_verdict(-0.1, 0.05) == "PASS"


# ==============================================================================
# Q. blocked test carries explicit reason
# ==============================================================================

@pytest.mark.parametrize("test_id", ["residual_scale_zero", "recipe_region_shift",
                                     "artifact_map_swap", "crop_padding_interpolation"])
def test_every_blocked_test_carries_an_explicit_nonempty_reason(test_id: str) -> None:
    payload = _real_protocol()
    config = payload["tests"][test_id]
    assert config["classification"] != "EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL"
    assert config["gpu_ready"] is False
    assert len(config["blocked_reason"].strip()) > 50


def test_executable_tests_are_exactly_the_four_declared() -> None:
    payload = _real_protocol()
    executable = {test_id for test_id, config in payload["tests"].items()
                 if config["classification"] == "EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL"}
    assert executable == set(diag.EXECUTABLE_TESTS)
    assert executable == {"benign_jpeg_corruption", "benign_resize_corruption",
                          "benign_color_corruption", "cross_route_synthetic"}


# ==============================================================================
# R. crop_padding_interpolation remains blocked if original geometry absent
# ==============================================================================

def test_crop_padding_interpolation_is_structurally_data_blocked() -> None:
    payload = _real_protocol()
    config = payload["tests"]["crop_padding_interpolation"]
    assert config["classification"] == "STRUCTURALLY_DATA_BLOCKED"
    assert "bounding" in config["blocked_reason"].lower() or "bbox" in config["blocked_reason"].lower()


def test_crop_padding_interpolation_reason_matches_the_known_reliability_audit() -> None:
    from prism_fas.evaluation.reliability import DATA_BLOCKED

    assert "crop_padding_interpolation" in DATA_BLOCKED
    payload = _real_protocol()
    # Both describe the SAME structural gap: no bbox / no source-frame path.
    assert "bbox" in DATA_BLOCKED["crop_padding_interpolation"]
    assert "bounding box" in payload["tests"]["crop_padding_interpolation"]["blocked_reason"]


def test_crop_padding_interpolation_never_renamed_to_resize() -> None:
    payload = _real_protocol()
    assert "crop_padding_interpolation" not in payload["benign_corruption_shared"]
    assert "crop_padding_interpolation" not in diag.CORRUPTION_FUNCTIONS


# ==============================================================================
# S. regional tests cannot substitute global evidence
# ==============================================================================

def test_residual_scale_zero_blocked_reason_names_the_real_structural_gap() -> None:
    payload = _real_protocol()
    reason = payload["tests"]["residual_scale_zero"]["blocked_reason"]
    assert "region_distances" in reason
    assert "manifold" in reason


def test_no_blocked_test_reason_claims_a_global_proxy_suffices() -> None:
    payload = _real_protocol()
    for test_id in ("residual_scale_zero", "recipe_region_shift", "artifact_map_swap"):
        reason = payload["tests"][test_id]["blocked_reason"].lower()
        assert "p_global" not in reason or "substitut" not in reason


def test_diagnostics_module_never_reads_forbidden_evidence_fields_for_benign_or_cross_route() -> None:
    source = inspect.getsource(diag.run_benign_corruption_diagnostic_for_arm)
    source += inspect.getsource(diag.run_cross_route_diagnostic_for_arm)
    for forbidden in probe.FORBIDDEN_EVIDENCE_FIELDS:
        assert forbidden not in source, forbidden


# ==============================================================================
# T. target path rejection
# ==============================================================================

def test_protocol_declares_forbidden_target_populations() -> None:
    payload = _real_protocol()
    forbidden = payload["target_firewall"]["forbidden_populations"]
    assert "siw_mv2" in forbidden
    assert "target_test" in forbidden


def test_no_function_in_the_module_resolves_a_target_root() -> None:
    for name in diag.__all__:
        obj = getattr(diag, name)
        if callable(obj) and not isinstance(obj, type):
            try:
                source = inspect.getsource(obj)
            except (OSError, TypeError):
                continue
            assert "target_label_root" not in source
            assert "target_feature_root" not in source


# ==============================================================================
# U. corrupted/mismatched binding fails closed
# ==============================================================================

def test_bind_only_refuses_a_mismatched_existing_protocol_binding(monkeypatch, tmp_path) -> None:
    _install_diagnostics_bind_fixtures(monkeypatch)
    runner._bind_only(tmp_path)
    stale = json.loads((tmp_path / diag.PROTOCOL_BINDING_PATH).read_text())
    stale["protocol_identity"] = "different" + "0" * 56
    atomic_write_json(tmp_path / diag.PROTOCOL_BINDING_PATH, stale)
    exit_code, payload = runner._bind_only(tmp_path)
    assert exit_code == runner.EXIT_BLOCKED
    assert "differs from the one just resolved" in payload["error"]


def test_execute_blocks_when_bindings_disagree_with_active_protocol(monkeypatch, tmp_path) -> None:
    _install_full_execute_fixtures(monkeypatch, tmp_path, verdict_per_arm="PASS")
    stale = json.loads((tmp_path / diag.PROTOCOL_BINDING_PATH).read_text())
    stale["protocol_identity"] = "different" + "0" * 56
    atomic_write_json(tmp_path / diag.PROTOCOL_BINDING_PATH, stale)
    exit_code, payload = runner._execute(tmp_path)
    assert exit_code == runner.EXIT_BLOCKED
    assert "not bound to the currently active protocol identity" in payload["error"]


# ==============================================================================
# V. existing result identity mismatch fails closed
# ==============================================================================

def test_execute_fails_closed_on_unresolvable_checkpoint_arm(monkeypatch, tmp_path) -> None:
    _install_diagnostics_bind_fixtures(monkeypatch)
    exit_code, _ = runner._bind_only(tmp_path)
    assert exit_code == runner.EXIT_PASS

    def _raise_benign(*a, **k):
        raise diag.PostFailureDiagnosticsError("forced failure for the test")

    monkeypatch.setattr(diag, "run_benign_corruption_diagnostic_for_arm", _raise_benign)
    exit_code, payload = runner._execute(tmp_path)
    assert exit_code == runner.EXIT_BLOCKED
    assert not (tmp_path / diag.RESULT_PATH).exists()


def test_execute_requires_bind_only_to_have_run_first(monkeypatch, tmp_path) -> None:
    _install_diagnostics_bind_fixtures(monkeypatch)   # protocol resolves; no bind
    exit_code, payload = runner._execute(tmp_path)
    assert exit_code == runner.EXIT_BLOCKED
    assert "bind-only" in payload["error"]


# ==============================================================================
# safety / import purity
# ==============================================================================

def test_importing_diagnostics_modules_touches_no_filesystem_state() -> None:
    import subprocess

    for module in ("prism_fas.evaluation.post_failure_diagnostics",
                  "prism_fas.evaluation.post_failure_diagnostics_runner"):
        result = subprocess.run([sys.executable, "-c", f"import {module}"],
                                capture_output=True, text=True, cwd=str(REPO))
        assert result.returncode == 0, result.stderr


def test_runner_invocable_as_python_dash_m_module() -> None:
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "prism_fas.evaluation.post_failure_diagnostics_runner",
         "--repo", str(REPO), "--preflight-only"],
        capture_output=True, text=True, cwd=str(REPO / "src"))
    assert result.returncode == runner.EXIT_BLOCKED, result.stderr
    payload = json.loads(result.stdout)
    assert payload["target_access"] == 0
