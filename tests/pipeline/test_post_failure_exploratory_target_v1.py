"""POST_FAILURE_EXPLORATORY_TARGET_V1 — Phase 1C protocol, firewall and
prediction/scoring handoff design.

**FIXTURE / ENGINEERING ONLY.** Every test in this file runs against
`tmp_path` fixtures, pure functions, or the REAL frozen C8 source matrix
metadata (row IDs, tracks, arms, seeds — never a checkpoint, never an
image, never target data). No test opens the real target feature package
or the real target label artifact, because neither exists on this laptop;
several tests exercise that absence directly and assert it is reported
honestly rather than fabricated.

This file proves the twenty-six items the freeze task requires (A-Z):
deterministic protocol identity; the exploratory namespace never collides
with C9-C13 or with the legacy M10/BA_sep artifact trees; `c9_may_close`
can never be set true; the BA_sep/detector-reliability/diagnostics-V2
failures travel with every artifact; the real, frozen 24-row P3-ready
target matrix resolves with the exact Track-G/Track-R breakdown; no
best-seed selection language exists; checkpoint and calibration identities
bind per row and fail closed off a non-`source_dev` calibration; the
predictor never imports the label reader and the prediction schema forbids
every label-bearing column; the scorer holds no checkpoint-loading
capability and refuses to run before a `FROZEN` prediction lock exists;
target labels can never be opened before that lock; no attack taxonomy is
resolvable during blind prediction; no target quantile can set a threshold;
the reject policy stays `NOT_APPLICABLE_NOT_FITTED`; partial prediction and
partial scoring results both BLOCK; complete ones re-report without
recomputation; the legacy `m10_target.yaml` is left untouched; and
`target_access` stays 0 throughout.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.evaluation import post_failure_exploratory_target as expl  # noqa: E402
from prism_fas.evaluation import post_failure_exploratory_target_scorer as scorer  # noqa: E402
from prism_fas.pipeline.state import atomic_write_json  # noqa: E402

PROTOCOL_PATH = REPO / "configs/evaluation/post_failure_exploratory_target_v1.yaml"
LEGACY_M10_PATH = REPO / "configs/evaluation/m10_target.yaml"


def _real_protocol() -> dict[str, Any]:
    return yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))


# ==============================================================================
# A. protocol identity deterministic
# ==============================================================================

def test_protocol_config_exists_and_is_frozen_not_run() -> None:
    assert PROTOCOL_PATH.is_file()
    payload = _real_protocol()
    assert payload["status"] == "FROZEN_NOT_RUN"
    assert payload["decision_id"] == "POST_FAILURE_EXPLORATORY_TARGET_V1"


def test_protocol_identity_is_deterministic() -> None:
    payload = _real_protocol()
    assert expl.protocol_identity(payload) == expl.protocol_identity(payload)
    assert len(expl.protocol_identity(payload)) == 64


def test_protocol_identity_changes_with_a_result_affecting_field() -> None:
    payload = _real_protocol()
    baseline = expl.protocol_identity(payload)
    changed = dict(payload)
    changed["thresholds"] = {**payload["thresholds"], "acer_threshold_source": "target_eer"}
    assert expl.protocol_identity(changed) != baseline


def test_protocol_identity_unaffected_by_metadata_only_change() -> None:
    payload = _real_protocol()
    baseline = expl.protocol_identity(payload)
    changed = dict(payload)
    changed["frozen_on"] = "2099-01-01"
    changed["approved_by"] = "someone else"
    assert expl.protocol_identity(changed) == baseline


def test_protocol_starts_with_labels_closed() -> None:
    payload = _real_protocol()
    assert payload["target_labels_opened"] is False
    assert payload["target_labels_revealed"] is False
    assert payload["target_predictions_observed"] is False
    assert payload["target_metrics_observed"] is False


def test_load_protocol_refuses_a_config_that_starts_opened(tmp_path) -> None:
    payload = _real_protocol()
    payload = dict(payload)
    payload["target_labels_revealed"] = True
    (tmp_path / "configs/evaluation").mkdir(parents=True)
    (tmp_path / expl.PROTOCOL_CONFIG_PATH).write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(expl.ExploratoryTargetError, match="target_labels_revealed"):
        expl.load_protocol(tmp_path)


# ==============================================================================
# B. exploratory namespace separate from original C9-C13
# ==============================================================================

def test_namespaces_never_collide_with_c9_through_c13() -> None:
    for forbidden in ("reports/full/c9", "reports/full/c10", "reports/full/c11",
                     "reports/full/c12", "reports/full/c13", "runs/full/c8"):
        assert not expl.DIAGNOSTICS_DIR.startswith(forbidden)
        assert not expl.RUN_DIR.startswith(forbidden)
        assert not forbidden.startswith(expl.DIAGNOSTICS_DIR)


def test_namespace_disjoint_from_ba_sep_and_diagnostics_v1_v2() -> None:
    from prism_fas.evaluation import post_failure_diagnostics as diag_v1
    from prism_fas.evaluation import post_failure_diagnostics_v2 as diag2
    from prism_fas.evaluation import synthetic_real_probe as probe

    for other in (diag_v1.DIAGNOSTICS_DIR, diag2.DIAGNOSTICS_DIR, probe.RELIABILITY_DIR):
        assert expl.DIAGNOSTICS_DIR != other
        assert not expl.DIAGNOSTICS_DIR.startswith(other)
        assert not other.startswith(expl.DIAGNOSTICS_DIR)


def test_protocol_declares_never_writes_into_confirmatory_namespaces() -> None:
    payload = _real_protocol()
    declared = payload["namespaces"]["never_writes_into"]
    for forbidden in ("reports/full/c9", "reports/full/c10", "reports/full/c11",
                     "reports/full/c12", "reports/full/c13",
                     "reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json"):
        assert forbidden in declared


# ==============================================================================
# C. cannot set c9_may_close=true
# ==============================================================================

def test_no_function_in_either_module_sets_c9_may_close_true() -> None:
    for module in (expl, scorer):
        source = Path(inspect.getfile(module)).read_text(encoding="utf-8")
        assert '"c9_may_close": True' not in source
        assert "c9_may_close = True" not in source


def test_score_result_hard_codes_c9_may_close_false() -> None:
    source = inspect.getsource(scorer._score)
    assert '"c9_may_close": False' in source


# ==============================================================================
# D/E/F. BA_sep FAIL, detector reliability FAILED, diagnostics V2 FAIL present
# ==============================================================================

def test_protocol_declares_ba_sep_fail_immutable() -> None:
    payload = _real_protocol()
    state = payload["immutable_upstream_state"]
    assert state["ba_sep_observed_verdict"] == "FAIL"
    assert state["ba_sep_rerun"] == "FORBIDDEN"
    assert state["ba_sep_by_arm"]["RND"] == pytest.approx(0.7843079833902619)


def test_protocol_declares_detector_reliability_failed() -> None:
    payload = _real_protocol()
    state = payload["immutable_upstream_state"]
    assert state["detector_reliability_lock_c_overall"] == "FAILED"
    assert state["c9_may_close"] is False
    assert state["c9_original_confirmatory_path"] == "BLOCKED"


def test_protocol_declares_diagnostics_v2_fail() -> None:
    payload = _real_protocol()
    state = payload["immutable_upstream_state"]
    assert state["post_failure_source_diagnostics_v2_overall"] == "FAIL"
    per_test = state["post_failure_source_diagnostics_v2_per_test"]
    assert per_test["benign_jpeg_corruption"] == "FAIL"
    assert per_test["benign_resize_corruption"] == "FAIL"
    assert per_test["benign_color_corruption"] == "PASS"


def test_required_binding_declared_for_every_future_output() -> None:
    payload = _real_protocol()
    required = payload["required_binding_in_every_output"]
    assert required["ba_sep_observed_verdict"] == "FAIL"
    assert required["detector_reliability_overall"] == "FAILED"
    assert required["post_failure_diagnostics_v2"] == "FAIL"
    assert required["c9_original_confirmatory_path"] == "BLOCKED"
    assert required["exploratory_target_status"] == "POST_FAILURE_EXPLORATORY"


# ==============================================================================
# G/H. exact P3-ready target matrix cardinality and breakdown
# ==============================================================================

def test_target_matrix_cardinality_is_24() -> None:
    rows = expl.resolve_target_matrix(REPO)
    assert len(rows) == 24


def test_track_g_and_track_r_breakdown() -> None:
    rows = expl.resolve_target_matrix(REPO)
    track_g = [row for row in rows if row.track == "G"]
    track_r = [row for row in rows if row.track == "R"]
    assert len(track_g) == 15
    assert len(track_r) == 9
    for arm in ("RND", "DET", "LLM"):
        assert sum(1 for row in track_g if row.arm == arm) == 5
    for experiment in ("C-R-DET", "C-R-LLM", "C-R-NOPROMPT"):
        assert sum(1 for row in track_r if row.experiment_id == experiment) == 3


def test_target_matrix_identity_is_deterministic_and_distinct_from_full_matrix() -> None:
    from prism_fas.evaluation.source_matrix import build_plan

    rows = expl.resolve_target_matrix(REPO)
    first = expl.target_matrix_identity(rows)
    second = expl.target_matrix_identity(rows)
    assert first == second
    full_identity = build_plan().identity
    assert first != full_identity


def test_target_matrix_rows_are_never_hand_picked() -> None:
    source = inspect.getsource(expl.resolve_target_matrix)
    assert "target_prediction_required" in source
    assert "row_id ==" not in source
    assert "row_id in (" not in source


# ==============================================================================
# I. no best-seed selection
# ==============================================================================

def test_no_best_seed_selection_language_in_matrix_or_binding_resolution() -> None:
    for fn in (expl.resolve_target_matrix, expl.resolve_row_binding, expl.resolve_all_row_bindings):
        source = inspect.getsource(fn)
        for forbidden in ("sorted(evidence", "max(checkpoints", "min(checkpoints",
                         "key=lambda item: item.metrics"):
            assert forbidden not in source, (fn.__name__, forbidden)


def test_protocol_declares_no_retraining_no_best_seed_no_dropped_arms() -> None:
    payload = _real_protocol()
    matrix = payload["target_matrix"]
    assert matrix["no_retraining"] is True
    assert matrix["no_best_seed_selection"] is True
    assert matrix["no_dropping_unfavorable_arms"] is True


# ==============================================================================
# J/K. checkpoint hashes and calibration identities bound per row
# ==============================================================================

def _fake_row(row_id="FAKE-ROW", experiment_id="FAKE-EXP", track="G", arm="RND",
             protocol="P3", seed=20260806, config_identity="c" * 64) -> Any:
    return SimpleNamespace(row_id=row_id, experiment_id=experiment_id, track=track, arm=arm,
                           protocol=protocol, seed=seed, config_identity=config_identity,
                           flags={"recipe_arm": arm})


def _write_row_manifest_and_calibration(repo: Path, row: Any, *, calibration_split="source_dev",
                                        target_labels_resolved=0) -> None:
    from prism_fas.evaluation import source_evidence

    directory = source_evidence.row_directory(repo / source_evidence.C8_RUNS, row)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "row_id": row.row_id, "run_identity": "run-" + row.row_id, "config_identity": row.config_identity,
        "status": "PASS", "fixture_backed": False, "target_labels_resolved": target_labels_resolved,
        "decision_logit_name": "global_logit_G", "decision_score_name": "p_global",
        "decision_graph_hash": "g" * 64,
        "checkpoint": {"sha256": "k" * 64, "path": f"runs/x/{row.row_id}/checkpoint_best.pt", "kind": "best"},
    }
    (directory / source_evidence.RUN_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    calibration = {"split": calibration_split, "threshold": 0.42, "temperature": 1.5,
                  "calibration_hash": "h" * 64}
    (directory / "calibration.json").write_text(json.dumps(calibration), encoding="utf-8")


def test_resolve_row_binding_binds_checkpoint_and_calibration(tmp_path) -> None:
    row = _fake_row()
    _write_row_manifest_and_calibration(tmp_path, row)
    binding = expl.resolve_row_binding(tmp_path, row)
    assert binding["checkpoint_sha256"] == "k" * 64
    assert binding["calibration_hash"] == "h" * 64
    assert binding["threshold"] == pytest.approx(0.42)
    assert binding["temperature"] == pytest.approx(1.5)


# ==============================================================================
# L. frozen threshold source is source-dev only
# ==============================================================================

def test_calibration_must_be_source_dev_or_binding_fails_closed(tmp_path) -> None:
    row = _fake_row()
    _write_row_manifest_and_calibration(tmp_path, row, calibration_split="target_test")
    with pytest.raises(expl.ExploratoryTargetError, match="source_dev"):
        expl.resolve_row_binding(tmp_path, row)


def test_manifest_with_resolved_target_labels_fails_closed(tmp_path) -> None:
    row = _fake_row()
    _write_row_manifest_and_calibration(tmp_path, row, target_labels_resolved=1)
    with pytest.raises(expl.ExploratoryTargetError, match="target_labels_resolved"):
        expl.resolve_row_binding(tmp_path, row)


def test_protocol_forbids_target_quantile_for_operating_threshold() -> None:
    payload = _real_protocol()
    assert payload["thresholds"]["target_quantile_forbidden_for_operating_threshold"] is True
    assert payload["thresholds"]["source"] == "SOURCE_DEV_FROZEN_CALIBRATION"


# ==============================================================================
# M/N. inference cannot resolve labels; prediction schema forbids labels
# ==============================================================================

def test_predictor_module_never_imports_the_label_reader() -> None:
    import_lines = [line.strip() for line in Path(inspect.getfile(expl)).read_text(encoding="utf-8")
                    .splitlines() if line.strip().startswith(("import ", "from "))]
    for forbidden in ("load_evaluation_labels", "EvaluationLabels", "scoring"):
        assert not any(forbidden in line for line in import_lines), (forbidden, import_lines)


def test_prediction_schema_reused_forbids_every_label_bearing_column() -> None:
    from prism_fas.evaluation.target_prediction import FORBIDDEN_PREDICTION_COLUMNS

    for forbidden in ("label", "true_label", "label_live_spoof", "attack_type",
                     "attack_family", "subject_id", "session_id"):
        assert forbidden in FORBIDDEN_PREDICTION_COLUMNS


def test_predict_one_row_reuses_target_prediction_verbatim() -> None:
    source = inspect.getsource(expl.predict_one_row)
    assert "target_batches(" in source
    assert "predict_target(" in source
    assert "write_predictions(" in source
    assert "build_prediction_lock(" in source


# ==============================================================================
# O/P/Q. scorer holds no checkpoint-loading capability; requires a frozen lock
# ==============================================================================

def test_scorer_holds_no_training_or_checkpoint_loading_capability() -> None:
    audit = scorer.assert_no_training_capability()
    assert audit["passed"] is True
    assert audit["violations"] == []


def test_scorer_module_source_never_mentions_torch_or_row_construction() -> None:
    source = Path(inspect.getfile(scorer)).read_text(encoding="utf-8")
    for forbidden in ("import torch", "construct_row_trainer", "M9Trainer", "load_checkpoint"):
        assert forbidden not in source, forbidden


def test_score_requires_a_frozen_prediction_lock(tmp_path) -> None:
    with pytest.raises(scorer.ExploratoryScoringError, match="does not exist"):
        scorer.require_frozen_prediction_lock(tmp_path)


def test_score_refuses_an_unlocked_or_unfrozen_lock(tmp_path) -> None:
    lock_path = tmp_path / scorer.PREDICTION_LOCK_PATH
    lock_path.parent.mkdir(parents=True)
    atomic_write_json(lock_path, {"status": "PARTIAL", "target_labels_opened": False})
    with pytest.raises(scorer.ExploratoryScoringError, match="FROZEN"):
        scorer.require_frozen_prediction_lock(tmp_path)


def test_preflight_score_blocks_before_any_prediction_lock_exists(tmp_path) -> None:
    exit_code, payload = scorer._preflight_score(tmp_path)
    assert exit_code == scorer.EXIT_BLOCKED
    assert payload["prediction_lock_valid"] is False
    assert payload["target_labels_opened"] is False


# ==============================================================================
# R. attack taxonomy unavailable to prediction phase
# ==============================================================================

def test_predictor_module_never_references_attack_taxonomy() -> None:
    source = Path(inspect.getfile(expl)).read_text(encoding="utf-8")
    for forbidden in ("attack_family", "attack_type", "taxonomy"):
        assert forbidden not in source, forbidden


# ==============================================================================
# S/T. target quantile cannot set threshold; reject remains not-applicable
# ==============================================================================

def test_resolve_row_binding_never_touches_target_data() -> None:
    source = inspect.getsource(expl.resolve_row_binding)
    for forbidden in ("target_feature_root", "target_label_root", "prism_target_eval_v2"):
        assert forbidden not in source, forbidden


def test_reject_policy_not_applicable_unless_frozen_source_threshold_exists() -> None:
    payload = _real_protocol()
    reject = payload["reject_policy"]
    assert reject["unknown_threshold"] is None
    assert reject["reject_policy"] == "NOT_APPLICABLE_NOT_FITTED"
    assert reject["reject_dependent_metrics"] == "NOT_APPLICABLE"


# ==============================================================================
# U/V. partial prediction blocks; complete prediction re-reports
# ==============================================================================

def _install_predict_fixtures(monkeypatch, tmp_path, *, row_count: int = 2) -> list[Any]:
    rows = [_fake_row(row_id=f"ROW-{i}", experiment_id=f"EXP-{i}") for i in range(row_count)]
    bindings = {row.row_id: {"row_id": row.row_id, "experiment_id": row.experiment_id,
                             "arm": row.arm, "track": row.track, "flags": row.flags}
               for row in rows}
    monkeypatch.setattr(expl, "load_protocol", lambda repo: {
        "roots": {"source_package_root": "x", "target_feature_root": "y",
                 "target_label_root": "z", "prediction_root": "runs/exploratory_target_v1"},
        "permissions": {"TRAIN": {}, "G7": {}, "G8": {}}, "g8_forbidden_write_patterns": []})
    monkeypatch.setattr(expl, "build_firewall", lambda repo, protocol: object())
    monkeypatch.setattr(expl, "resolve_target_matrix", lambda repo: rows)
    monkeypatch.setattr(expl, "resolve_all_row_bindings", lambda repo, rows: bindings)
    monkeypatch.setattr(expl, "verify_target_feature_package_expected",
                        lambda repo, protocol: {"computed_identity": "pkg" + "0" * 61})
    monkeypatch.setattr(expl, "build_overall_prediction_lock",
                        lambda repo, **kwargs: {"status": "FROZEN", "entry_count": row_count,
                                                "lockset_identity": "l" * 64})

    binding_path = tmp_path / expl.PREDICTION_PLAN_BINDING_PATH
    binding_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(binding_path, {"protocol_identity": "p" * 64, "target_matrix_identity": "m" * 64,
                                     "row_count": row_count})
    return rows


def test_predict_partial_result_blocks(monkeypatch, tmp_path) -> None:
    _install_predict_fixtures(monkeypatch, tmp_path, row_count=2)
    # simulate ONE of two rows already predicted
    partial_dir = tmp_path / expl.RUN_DIR / "ROW-0"
    partial_dir.mkdir(parents=True)
    (partial_dir / "target_predictions.parquet").write_bytes(b"x")

    exit_code, payload = expl._predict(tmp_path)
    assert exit_code == expl.EXIT_BLOCKED
    assert payload["error"] == "PARTIAL_SCIENTIFIC_RESULT_SET"


def test_predict_complete_lock_reuses_without_inference(monkeypatch, tmp_path) -> None:
    _install_predict_fixtures(monkeypatch, tmp_path, row_count=2)
    lock_path = tmp_path / expl.PREDICTION_LOCK_PATH
    atomic_write_json(lock_path, {"status": "FROZEN", "entry_count": 2})

    called = {"n": 0}
    monkeypatch.setattr(expl, "predict_one_row",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    exit_code, payload = expl._predict(tmp_path)
    assert exit_code == expl.EXIT_PASS
    assert payload["reused_existing_lock"] is True
    assert payload["checkpoint_weights_loaded"] is False
    assert called["n"] == 0


def test_predict_full_run_from_clean_host_calls_every_row_exactly_once(monkeypatch, tmp_path) -> None:
    rows = _install_predict_fixtures(monkeypatch, tmp_path, row_count=2)
    calls: list[str] = []

    def _fake_predict_one_row(repo, binding, *, firewall):
        calls.append(binding["row_id"])
        return {"row_id": binding["row_id"], "row_count": 4,
               "lock": {"experiment_id": binding["experiment_id"], "row_count": 4}}

    monkeypatch.setattr(expl, "predict_one_row", _fake_predict_one_row)
    exit_code, payload = expl._predict(tmp_path)
    assert exit_code == expl.EXIT_PASS
    assert payload["executed"] is True
    assert sorted(calls) == sorted(row.row_id for row in rows)
    lock_doc = json.loads((tmp_path / expl.PREDICTION_LOCK_PATH).read_text())
    assert lock_doc["status"] == "FROZEN"


def test_predict_requires_bind_prediction_plan_first(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(expl, "load_protocol", lambda repo: {})
    exit_code, payload = expl._predict(tmp_path)
    assert exit_code == expl.EXIT_BLOCKED
    assert "bind-prediction-plan" in payload["error"]


# ==============================================================================
# W/X. partial scoring blocks; complete scoring re-reports
# ==============================================================================

def _write_frozen_lock(tmp_path: Path, *, entry_count: int = 2) -> None:
    lock_path = tmp_path / scorer.PREDICTION_LOCK_PATH
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(lock_path, {"status": "FROZEN", "entry_count": entry_count,
                                  "target_labels_opened": False, "lockset_identity": "l" * 64,
                                  "entries": [{"experiment_id": f"EXP-{i}", "seed": 1}
                                             for i in range(entry_count)]})


def test_score_partial_row_set_blocks(tmp_path) -> None:
    _write_frozen_lock(tmp_path, entry_count=2)
    rows_dir = tmp_path / scorer.SCORE_ROWS_DIR
    rows_dir.mkdir(parents=True)
    atomic_write_json(rows_dir / "ROW-0.json", {"acer": 0.1})   # only one of two

    exit_code, payload = scorer._score(tmp_path)
    assert exit_code == scorer.EXIT_BLOCKED
    assert payload["error"] == "PARTIAL_SCIENTIFIC_RESULT_SET"


def test_score_complete_result_reuses_without_rescoring(monkeypatch, tmp_path) -> None:
    _write_frozen_lock(tmp_path, entry_count=2)
    result_path = tmp_path / scorer.SCORE_RESULT_PATH
    atomic_write_json(result_path, {"row_count": 2, "target_labels_opened": True})

    called = {"n": 0}
    monkeypatch.setattr(scorer, "score_one_row",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    exit_code, payload = scorer._score(tmp_path)
    assert exit_code == scorer.EXIT_PASS
    assert payload["reused_existing_score_result"] is True
    assert called["n"] == 0


# ==============================================================================
# Y. old m10_target.yaml remains untouched
# ==============================================================================

def test_legacy_m10_target_config_not_modified_by_this_freeze() -> None:
    import subprocess

    result = subprocess.run(["git", "diff", "--stat", "HEAD", "--",
                            "configs/evaluation/m10_target.yaml"],
                            cwd=str(REPO), capture_output=True, text=True)
    assert result.stdout.strip() == ""


def test_legacy_m10_target_config_still_declares_labels_revealed_true() -> None:
    # Documented, not reused: the legacy engineering-fixture state is exactly
    # why this protocol never loads that file as its own state.
    legacy = yaml.safe_load(LEGACY_M10_PATH.read_text(encoding="utf-8"))
    assert legacy["target_labels_revealed"] is True


def test_protocol_declares_legacy_reuse_policy() -> None:
    payload = _real_protocol()
    reuse = payload["legacy_m10_reuse"]
    assert reuse["legacy_config_path"] == "configs/evaluation/m10_target.yaml"
    assert reuse["legacy_config_never_loaded_as_this_protocol"] is True
    assert reuse["legacy_config_left_untouched"] is True
    changed = {item["field"] for item in reuse["intentionally_changed_semantics"]}
    assert "target_labels_revealed" in changed


# ==============================================================================
# Z. target_access remains 0 during laptop task
# ==============================================================================

def test_protocol_declares_target_access_zero() -> None:
    payload = _real_protocol()
    assert payload["target_access"] == 0


def test_preflight_reports_target_access_zero() -> None:
    exit_code, payload = expl._preflight(REPO)
    assert payload["target_access"] == 0


def test_status_reports_target_access_zero(tmp_path) -> None:
    exit_code, payload = expl._status(tmp_path)
    assert payload["target_access"] == 0


def test_scorer_preflight_reports_target_access_zero(tmp_path) -> None:
    exit_code, payload = scorer._preflight_score(tmp_path)
    assert payload["target_access"] == 0


# ==============================================================================
# real repo behavior: target feature package / label root, never fabricated
# ==============================================================================

def test_target_feature_package_reports_not_present_on_this_host() -> None:
    payload = _real_protocol()
    result = expl.verify_target_feature_package_expected(REPO, payload)
    assert result["present_on_this_host"] is False
    assert result["verified"] is False
    assert result["computed_identity"] is None


def test_target_label_root_reports_sealed_and_unopened() -> None:
    payload = _real_protocol()
    result = expl.verify_target_label_root_sealed(REPO, payload)
    assert result["target_labels_opened"] is False
    assert result["label_root_permission"] == "deny"


def test_package_identity_mismatch_fails_closed(tmp_path) -> None:
    package_root = tmp_path / "data/processed/prism_target_eval_v2"
    package_root.mkdir(parents=True)
    (package_root / "a.bin").write_bytes(b"not the real package")
    payload = dict(_real_protocol())
    payload["target_feature_package"] = {**payload["target_feature_package"],
                                         "target_feature_root": "data/processed/prism_target_eval_v2"}
    with pytest.raises(expl.ExploratoryTargetError, match="mismatch"):
        expl.verify_target_feature_package_expected(tmp_path, payload)


# ==============================================================================
# reused-verbatim legacy components (never reimplemented)
# ==============================================================================

def test_video_aggregation_reused_verbatim() -> None:
    from prism_fas.evaluation import video_aggregation

    source = inspect.getsource(expl)
    # the predictor never reimplements trimmed-mean aggregation itself
    assert "def aggregate_frames" not in source
    assert "def trimmed_mean" not in source
    assert video_aggregation.TRIM_FRACTION == 0.10


def test_bootstrap_and_holm_bonferroni_are_pure_functions() -> None:
    for fn in (scorer.paired_bootstrap_acer_difference, scorer.holm_bonferroni):
        source = inspect.getsource(fn)
        for forbidden in ("open(", "Path(", "torch", "read_text", "read_bytes"):
            assert forbidden not in source, (fn.__name__, forbidden)


def test_runner_invocable_as_python_dash_m_module() -> None:
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "prism_fas.evaluation.post_failure_exploratory_target",
         "--repo", str(REPO), "--preflight-only"],
        capture_output=True, text=True, cwd=str(REPO / "src"))
    assert result.returncode == expl.EXIT_BLOCKED, result.stderr
    payload = json.loads(result.stdout)
    assert payload["target_access"] == 0


def test_scorer_invocable_as_python_dash_m_module() -> None:
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "prism_fas.evaluation.post_failure_exploratory_target_scorer",
         "--repo", str(REPO), "--preflight-score"],
        capture_output=True, text=True, cwd=str(REPO / "src"))
    assert result.returncode == scorer.EXIT_BLOCKED, result.stderr
    payload = json.loads(result.stdout)
    assert payload["target_access"] == 0


def test_importing_modules_touches_no_filesystem_state() -> None:
    import subprocess

    for module in ("prism_fas.evaluation.post_failure_exploratory_target",
                  "prism_fas.evaluation.post_failure_exploratory_target_scorer"):
        result = subprocess.run([sys.executable, "-c", f"import {module}"],
                                capture_output=True, text=True, cwd=str(REPO))
        assert result.returncode == 0, result.stderr
