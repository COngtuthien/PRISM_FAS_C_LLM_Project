"""Tests for src/prism_fas/evaluation/c_ext_e8_target_evaluation.py (E8 G7).

Pure, read-only-over-synthetic-fixtures tests. No GPU, no real target
features/labels, no real checkpoints. The calibration authority lock is
injected (mocked) rather than built for real; staging/promotion/lockset
logic is exercised against fully synthetic per-run results, never a real
``target_batches``/``predict_target`` call.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from prism_fas.evaluation import c_ext_e8_calibration_authority as auth  # noqa: E402
from prism_fas.evaluation import c_ext_e8_target_evaluation as g7  # noqa: E402
from prism_fas.evaluation import c_ext_e8_training_runner as runner  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def _fake_calibration_lock() -> dict[str, Any]:
    rows = []
    for spec in runner.all_scientific_run_specs():
        rows.append({
            "run_id": spec.run_id, "arm": spec.arm, "seed": spec.seed,
            "authoritative_calibration_kind": "HISTORICAL_VALID",
            "best_checkpoint_path": f"runs/fake/{spec.run_id}/checkpoints/best.pt",
            "best_checkpoint_sha256": f"ckpt-sha-{spec.run_id}",
            "calibration_file_sha256": f"calib-sha-{spec.run_id}",
            "calibration_hash": f"calib-hash-{spec.run_id}",
            "temperature": 0.2, "selected_threshold": 0.5,
        })
    return {"lock_identity": "fake-calibration-authority-identity", "rows": rows,
           "entry_count": 15, "corrected_entry_count": 1}


# --------------------------------------------------------------------------- #
# Plan building
# --------------------------------------------------------------------------- #

_FAST_PACKAGE_CHECK = {"present_on_this_host": True, "verified": True}


def _patched_plan(**kwargs):
    """Builds the plan with the (slow, real) target-feature-package
    verification replaced by a fast stub -- these tests exercise plan
    LOGIC, not package verification (covered separately, real, below)."""
    with mock.patch.object(auth, "require_valid_calibration_authority_for_g7",
                           lambda repo: _fake_calibration_lock()), \
         mock.patch.object(g7, "verify_target_feature_package", lambda repo: dict(_FAST_PACKAGE_CHECK)):
        return g7.build_target_prediction_plan(REPO, **kwargs)


def test_plan_has_exactly_15_runs():
    plan = _patched_plan(code_commit="deadbeef")
    assert plan["run_count"] == 15
    assert len(plan["runs"]) == 15


def test_plan_5_rnd_5_det_5_llm():
    plan = _patched_plan(code_commit="deadbeef")
    counts: dict[str, int] = {}
    for entry in plan["runs"]:
        counts[entry["arm"]] = counts.get(entry["arm"], 0) + 1
    assert counts == {"RND": 5, "DET": 5, "LLM": 5}


def test_plan_binds_calibration_authority_identity_and_unknown_threshold():
    plan = _patched_plan(code_commit="deadbeef")
    assert plan["calibration_authority_identity"] == "fake-calibration-authority-identity"
    assert plan["unknown_threshold"] is None
    assert plan["target_package_id"] == "prism_target_eval_v2"
    assert plan["target_feature_package_identity"] == g7.TARGET_FEATURE_PACKAGE_IDENTITY


def test_plan_refuses_incomplete_calibration_authority():
    bad_lock = _fake_calibration_lock()
    bad_lock["rows"] = bad_lock["rows"][:14]  # drop one run
    with mock.patch.object(auth, "require_valid_calibration_authority_for_g7", lambda repo: bad_lock):
        with pytest.raises(g7.E8TargetEvaluationError, match="exactly the 15"):
            g7.build_target_prediction_plan(REPO)


def test_plan_identity_deterministic():
    plan_1 = _patched_plan(code_commit="deadbeef")
    plan_2 = _patched_plan(code_commit="deadbeef")
    assert plan_1["plan_identity"] == plan_2["plan_identity"]


# --------------------------------------------------------------------------- #
# Firewall: G7 cannot resolve labels
# --------------------------------------------------------------------------- #

def test_g7_cannot_read_target_labels():
    firewall = g7.build_target_firewall(REPO)
    report = firewall.report()
    assert report["g7_can_read_target_labels"] is False
    assert report["g8_can_read_target_labels"] is True


def test_g7_target_label_root_denied_structurally():
    firewall = g7.build_target_firewall(REPO)
    audit = firewall.assert_cannot_resolve_labels("G7")
    assert audit["label_root_permission"] == "deny"
    assert audit["target_labels_opened"] is False


def test_preflight_never_reads_target_labels():
    result = g7.preflight_e8_target_prediction(REPO)
    assert result["target_labels_accessed"] is False
    assert result["target_features_opened"] is False
    assert result["unknown_threshold"] is None


def test_preflight_never_calls_target_batches_or_predict_target():
    fn_source = Path(g7.__file__).read_text(encoding="utf-8")
    # isolate just the preflight function body
    start = fn_source.index("def preflight_e8_target_prediction")
    end = fn_source.index("\ndef ", start + 1)
    body = fn_source[start:end]
    assert "target_batches(" not in body
    assert "predict_target(" not in body


# --------------------------------------------------------------------------- #
# Predictions structurally contain no forbidden columns (reused validator)
# --------------------------------------------------------------------------- #

def test_prediction_schema_forbids_label_columns():
    from prism_fas.evaluation.target_prediction import FORBIDDEN_PREDICTION_COLUMNS, validate_predictions

    row = {"sample_id": "s1", "video_id": "v1", "frame_id": 0, "p_global": 0.5, "s_region": None,
          "p_prompt": None, "s_final": 0.5, "decision_score": 0.5, "confidence": 0.5, "decision": "live",
          "top_region_ids": [], "region_distances": [], "checkpoint_hash": "a", "calibration_hash": "b",
          "inference_config_hash": "c", "region_status": "not_applicable", "prompt_status": "not_applicable",
          "variant": "E8_QMATCH_RND"}
    validate_predictions([row])  # must not raise
    leaked = dict(row)
    leaked["label"] = 0
    with pytest.raises(Exception):
        validate_predictions([leaked])
    assert "label" in FORBIDDEN_PREDICTION_COLUMNS
    assert "attack_family" in FORBIDDEN_PREDICTION_COLUMNS
    assert "taxonomy" in FORBIDDEN_PREDICTION_COLUMNS


# --------------------------------------------------------------------------- #
# Target package identity mismatch fails closed
# --------------------------------------------------------------------------- #

def test_target_package_identity_mismatch_fails_closed(tmp_path):
    package_root = tmp_path / "fake_target_package"
    package_root.mkdir()
    (package_root / "PACKAGE_LOCK.json").write_text(json.dumps({
        "package_id": "prism_target_eval_v2", "status": "validated",
        "content_identity_sha256": "0" * 64,
    }), encoding="utf-8")
    with mock.patch.object(g7, "TARGET_FEATURE_ROOT", "fake_target_package"):
        with pytest.raises(Exception):
            g7.verify_target_feature_package(tmp_path)


# --------------------------------------------------------------------------- #
# Lockset: exactly 15 entries, 5/5/5, exact seed sets, no engineering smoke
# --------------------------------------------------------------------------- #

def _fake_manifest() -> dict[str, Any]:
    runs = []
    for spec in runner.all_scientific_run_specs():
        runs.append({
            "run_id": spec.run_id, "arm": spec.arm, "seed": spec.seed,
            "best_checkpoint_sha256": f"ckpt-{spec.run_id}", "calibration_sha256": f"calib-{spec.run_id}",
            "calibration_hash": f"hash-{spec.run_id}",
            "target_feature_package_identity": g7.TARGET_FEATURE_PACKAGE_IDENTITY,
            "prediction_output_path": g7.prediction_output_path(spec.run_id),
            "prediction_file_sha256": f"predfile-{spec.run_id}",
            "prediction_lock_identity": f"predlock-{spec.run_id}",
            "row_count": 6800, "video_count": 1700,
            "no_label_audit": {"labels_present": False, "checked": True},
        })
    return {"schema_version": "e8-target-prediction-manifest-v1", "plan_identity": "fake-plan",
           "calibration_authority_identity": "fake-calibration-authority-identity",
           "run_count": len(runs), "runs": runs}


def test_lockset_exactly_15_entries_5_5_5():
    lockset = g7.build_target_prediction_lockset(_fake_manifest())
    assert lockset["run_count"] == 15
    assert len(lockset["entries"]) == 15
    counts: dict[str, int] = {}
    for entry in lockset["entries"]:
        counts[entry["arm"]] = counts.get(entry["arm"], 0) + 1
    assert counts == {"RND": 5, "DET": 5, "LLM": 5}


def test_lockset_exact_seed_sets_per_arm():
    lockset = g7.build_target_prediction_lockset(_fake_manifest())
    by_arm: dict[str, list[int]] = {}
    for entry in lockset["entries"]:
        by_arm.setdefault(entry["arm"], []).append(entry["seed"])
    for arm, seeds in by_arm.items():
        assert sorted(seeds) == sorted(runner.SEEDS)


def test_lockset_status_frozen_and_no_target_labels():
    lockset = g7.build_target_prediction_lockset(_fake_manifest())
    assert lockset["status"] == "FROZEN"
    assert lockset["target_labels_accessed"] is False
    assert lockset["no_engineering_smoke_entries"] is True


def test_lockset_refuses_extra_or_missing_run():
    manifest = _fake_manifest()
    manifest["runs"] = manifest["runs"][:14]  # missing one
    manifest["run_count"] = 14
    with pytest.raises(g7.E8TargetEvaluationError, match="expected exactly"):
        g7.build_target_prediction_lockset(manifest)


def test_lockset_refuses_wrong_seed_set():
    manifest = _fake_manifest()
    manifest["runs"][0]["seed"] = 99999999  # not a frozen seed
    with pytest.raises(g7.E8TargetEvaluationError, match="seed set"):
        g7.build_target_prediction_lockset(manifest)


def test_is_usable_lockset_roundtrip():
    lockset = g7.build_target_prediction_lockset(_fake_manifest())
    assert g7.is_usable_lockset(lockset) is True
    tampered = dict(lockset)
    tampered["run_count"] = 14
    assert g7.is_usable_lockset(tampered) is False


def test_require_valid_lockset_for_g8_refuses_when_missing(tmp_path):
    with pytest.raises(g7.E8TargetEvaluationError, match="no valid, complete 15-row"):
        g7.require_valid_lockset_for_g8(tmp_path)


# --------------------------------------------------------------------------- #
# Staging / promotion crash-safety (synthetic; no real inference)
# --------------------------------------------------------------------------- #

def _fake_run_result(staging_root: Path, run_id: str) -> dict[str, Any]:
    from prism_fas.evaluation.target_prediction import PREDICTION_LOCK_FILE

    run_dir = staging_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "target_predictions.parquet").write_bytes(f"fake-parquet-{run_id}".encode())
    lock = {"prediction_lock_identity": f"lockid-{run_id}", "row_count": 10, "video_count": 3}
    (run_dir / PREDICTION_LOCK_FILE).write_text(json.dumps(lock), encoding="utf-8")
    import prism_fas.evaluation.c_ext_common as cc
    return {"run_id": run_id, "staging_dir": str(run_dir),
           "prediction_file_sha256": cc.sha256_file(run_dir / "target_predictions.parquet"),
           "row_count": 10, "lock": lock}


def test_promote_staged_runs_moves_all_runs(tmp_path):
    repo = tmp_path
    staging_root = repo / g7.STAGING_ROOT_RELATIVE
    run_ids = [f"run_{i}" for i in range(3)]
    run_results = {rid: _fake_run_result(staging_root, rid) for rid in run_ids}
    g7.promote_staged_runs(repo, staging_root=staging_root, plan_identity="plan-x", run_results=run_results)
    for rid in run_ids:
        final_dir = repo / g7.PREDICTION_RUN_ROOT / rid
        assert final_dir.is_dir()
        assert (final_dir / "target_predictions.parquet").is_file()
        assert not (final_dir / "STAGING_MARKER.json").exists()


def test_promotion_is_idempotent_on_rerun(tmp_path):
    repo = tmp_path
    staging_root = repo / g7.STAGING_ROOT_RELATIVE
    run_ids = ["run_a", "run_b"]
    run_results = {rid: _fake_run_result(staging_root, rid) for rid in run_ids}
    g7.promote_staged_runs(repo, staging_root=staging_root, plan_identity="plan-x", run_results=run_results)
    # re-running promotion (simulating a resumed/rerun collision) must validate
    # in place, never re-stage or duplicate, and never raise
    g7.promote_staged_runs(repo, staging_root=staging_root, plan_identity="plan-x", run_results=run_results)
    for rid in run_ids:
        assert (repo / g7.PREDICTION_RUN_ROOT / rid / "target_predictions.parquet").is_file()


def test_promotion_refuses_mismatched_run_set(tmp_path):
    repo = tmp_path
    staging_root = repo / g7.STAGING_ROOT_RELATIVE
    run_results = {"run_a": _fake_run_result(staging_root, "run_a")}
    g7.promote_staged_runs(repo, staging_root=staging_root, plan_identity="plan-x", run_results=run_results)
    different = {"run_z": _fake_run_result(staging_root, "run_z")}
    with pytest.raises(g7.E8TargetEvaluationError, match="does not match"):
        g7.promote_staged_runs(repo, staging_root=staging_root, plan_identity="plan-x", run_results=different)


def test_promotion_detects_tampered_staged_artifact(tmp_path):
    repo = tmp_path
    staging_root = repo / g7.STAGING_ROOT_RELATIVE
    run_results = {"run_a": _fake_run_result(staging_root, "run_a")}
    # corrupt the staged prediction file after building the result record
    (staging_root / "run_a" / "target_predictions.parquet").write_bytes(b"tampered")
    with pytest.raises(g7.E8TargetEvaluationError, match="hash disagrees"):
        g7.promote_staged_runs(repo, staging_root=staging_root, plan_identity="plan-x", run_results=run_results)


# --------------------------------------------------------------------------- #
# Structural: no target label access anywhere in this module's source
# --------------------------------------------------------------------------- #

def test_module_never_hardcodes_a_label_read():
    source = Path(g7.__file__).read_text(encoding="utf-8")
    assert "read_parquet" not in source or "label" not in source.lower().split("read_parquet")[1][:200]
    assert "pq.read_table" not in source
    assert "openai" not in source.lower()
    assert "gemini" not in source.lower()
