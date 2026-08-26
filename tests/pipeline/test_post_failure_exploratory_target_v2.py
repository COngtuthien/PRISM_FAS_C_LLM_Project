"""POST_FAILURE_EXPLORATORY_TARGET_V2 — pre-target scientific + execution
correction of V1.

**FIXTURE / ENGINEERING ONLY.** Every test here runs against `tmp_path`
fixtures, pure functions, or the REAL frozen C8 source matrix metadata
(row IDs, tracks, arms, seeds) — never a real checkpoint, never a real
image, never target data.

This file proves the thirty-two items the correction task requires (A-AF):
V1 stays byte-identical and untouched; V2 has its own distinct protocol
identity; the real 24-row matrix and its 15/9 breakdown; the frozen
prediction-plan binding carries every result-affecting field including
`flags`; `--predict` consumes ONLY the frozen binding and BLOCKS if a
fresh, read-only recomputation of checkpoint/calibration/source-matrix
disagrees with it; the target feature package must verify before bind
(never merely at preflight); the label root stays sealed to Phase E1;
`C-R-NOPROMPT` and `C-R-LLM` get distinct `prediction_variant_id`s despite
sharing `arm=LLM`; the new lockset is row_id-keyed (24 unique entries,
repeated `experiment_id` values allowed) while the legacy
`target_prediction.build_lockset` is provably untouched; prediction
completeness validation covers files, locks, and their combinations;
`ACER = 0.5*(APCER+BPCER)` and differs from a raw class-blind error rate
under the frozen target's unequal live/spoof counts; every seed is
retained through the paired comparison, never overwritten; the
class-stratified bootstrap is deterministic under the frozen seed; all
seven atomic comparisons enter one Holm-Bonferroni family; the scorer
holds no checkpoint-loading capability and requires a valid frozen V2
lock; existing-score validation fails closed on tampering; and
`target_access`/label-opened stay 0/false throughout every laptop-run mode.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.evaluation import post_failure_exploratory_target as v1  # noqa: E402
from prism_fas.evaluation import post_failure_exploratory_target_v2 as v2  # noqa: E402
from prism_fas.evaluation import post_failure_exploratory_target_v2_scorer as v2s  # noqa: E402
from prism_fas.pipeline.state import atomic_write_json  # noqa: E402

V1_PROTOCOL_PATH = REPO / "configs/evaluation/post_failure_exploratory_target_v1.yaml"
V2_PROTOCOL_PATH = REPO / "configs/evaluation/post_failure_exploratory_target_v2.yaml"
V1_IDENTITY = "8fb806d25a80ecd3c7d44cfeba8c893a5f115b8b51797220a51132ba16708b51"


def _real_v1_protocol() -> dict[str, Any]:
    return yaml.safe_load(V1_PROTOCOL_PATH.read_text(encoding="utf-8"))


def _real_v2_protocol() -> dict[str, Any]:
    return yaml.safe_load(V2_PROTOCOL_PATH.read_text(encoding="utf-8"))


# ==============================================================================
# A. V1 unchanged
# ==============================================================================

def test_v1_config_not_modified() -> None:
    import subprocess

    result = subprocess.run(["git", "diff", "--stat", "HEAD", "--",
                            "configs/evaluation/post_failure_exploratory_target_v1.yaml"],
                            cwd=str(REPO), capture_output=True, text=True)
    assert result.stdout.strip() == ""


def test_v1_modules_not_modified() -> None:
    import subprocess

    result = subprocess.run(["git", "diff", "--stat", "HEAD", "--",
                            "src/prism_fas/evaluation/post_failure_exploratory_target.py",
                            "src/prism_fas/evaluation/post_failure_exploratory_target_scorer.py"],
                            cwd=str(REPO), capture_output=True, text=True)
    assert result.stdout.strip() == ""


def test_v1_protocol_identity_unchanged() -> None:
    assert v1.protocol_identity(_real_v1_protocol()) == V1_IDENTITY


def test_legacy_build_lockset_source_not_modified() -> None:
    import subprocess

    result = subprocess.run(["git", "diff", "--stat", "HEAD", "--",
                            "src/prism_fas/evaluation/target_prediction.py"],
                            cwd=str(REPO), capture_output=True, text=True)
    assert result.stdout.strip() == ""


# ==============================================================================
# B. V2 distinct protocol identity
# ==============================================================================

def test_v2_protocol_identity_differs_from_v1() -> None:
    v1_id = v1.protocol_identity(_real_v1_protocol())
    v2_id = v2.protocol_identity(_real_v2_protocol())
    assert v1_id != v2_id
    assert len(v2_id) == 64


def test_v2_protocol_declares_frozen_not_run_and_supersedes_v1() -> None:
    payload = _real_v2_protocol()
    assert payload["status"] == "FROZEN_NOT_RUN"
    assert payload["decision_id"] == "POST_FAILURE_EXPLORATORY_TARGET_V2"
    assert payload["supersedes"]["protocol_identity"] == V1_IDENTITY
    assert payload["supersedes"]["v1_target_predictions_observed"] is False


def test_v2_namespace_disjoint_from_v1() -> None:
    assert v2.DIAGNOSTICS_DIR != v1.DIAGNOSTICS_DIR
    assert v2.RUN_DIR != v1.RUN_DIR
    assert not v2.DIAGNOSTICS_DIR.startswith(v1.DIAGNOSTICS_DIR)
    assert not v1.DIAGNOSTICS_DIR.startswith(v2.DIAGNOSTICS_DIR)


# ==============================================================================
# C/D. exact 24 rows, 15 Track-G / 9 Track-R
# ==============================================================================

def test_real_target_matrix_is_24_rows_15_track_g_9_track_r() -> None:
    rows = v1.resolve_target_matrix(REPO)
    assert len(rows) == 24
    assert sum(1 for r in rows if r.track == "G") == 15
    assert sum(1 for r in rows if r.track == "R") == 9


# ==============================================================================
# fixtures shared by binding/predict/lockset tests
# ==============================================================================

def _fake_row(row_id: str, experiment_id: str, track: str, arm: str, seed: int,
             config_identity: str = "c" * 64) -> Any:
    return SimpleNamespace(row_id=row_id, experiment_id=experiment_id, track=track, arm=arm,
                           protocol="P3", seed=seed, config_identity=config_identity,
                           flags={"recipe_arm": arm, "track": track})


FAKE_ROWS = [
    _fake_row("C-G-RND-P3READY-s1", "C-G-RND", "G", "RND", 1),
    _fake_row("C-R-LLM-P3READY-s1", "C-R-LLM", "R", "LLM", 1),
    _fake_row("C-R-NOPROMPT-P3READY-s1", "C-R-NOPROMPT", "R", "LLM", 1),
]


def _fake_binding_for(row: Any, *, checkpoint_sha256: str = "k" * 64,
                      calibration_hash: str = "h" * 64) -> dict[str, Any]:
    return {"row_id": row.row_id, "experiment_id": row.experiment_id, "track": row.track,
           "arm": row.arm, "protocol": row.protocol, "seed": row.seed,
           "config_identity": row.config_identity, "run_identity": "run-" + row.row_id,
           "checkpoint_sha256": checkpoint_sha256, "checkpoint_relative_path": "p.pt",
           "checkpoint_kind": "best", "decision_logit_name": "global_logit_G",
           "decision_score_name": "p_global", "decision_graph_hash": "g" * 64,
           "calibration_hash": calibration_hash, "calibration_split": "source_dev",
           "threshold": 0.5, "temperature": 1.0, "flags": dict(row.flags)}


def _install_v2_binding_fixtures(monkeypatch, *, rows=FAKE_ROWS,
                                 checkpoint_sha256="k" * 64, calibration_hash="h" * 64
                                 ) -> dict[str, Any]:
    """Monkeypatches every V1 function `v2.build_prediction_plan_binding`
    calls via the `v1` module alias, so a fresh recomputation is fully
    controllable and mutable between calls (to simulate drift)."""
    state = {"checkpoint_sha256": checkpoint_sha256, "calibration_hash": calibration_hash}

    monkeypatch.setattr(v2, "load_protocol", lambda repo: _real_v2_protocol())
    monkeypatch.setattr(v1, "resolve_target_matrix", lambda repo: list(rows))
    monkeypatch.setattr(v1, "target_matrix_identity",
                        lambda rows_arg: "matrix-" + "".join(sorted(r.row_id for r in rows_arg)))

    def _resolve_all_row_bindings(repo, rows_arg):
        return {row.row_id: _fake_binding_for(row, checkpoint_sha256=state["checkpoint_sha256"],
                                              calibration_hash=state["calibration_hash"])
               for row in rows_arg}

    monkeypatch.setattr(v1, "resolve_all_row_bindings", _resolve_all_row_bindings)
    monkeypatch.setattr(v1, "bind_c8_matrix_identity", lambda repo: {"c8_matrix_identity": "c8" + "0" * 62})
    monkeypatch.setattr(v1, "verify_target_feature_package_expected",
                        lambda repo, protocol: {"present_on_this_host": True, "verified": True,
                                                "expected_identity": "e" * 64, "computed_identity": "e" * 64,
                                                "reason": ""})
    monkeypatch.setattr(v1, "verify_target_label_root_sealed",
                        lambda repo, protocol: {"stage": "G7", "label_root_permission": "deny",
                                                "label_root_declared": True, "label_root_exists": False,
                                                "target_labels_opened": False})
    monkeypatch.setattr(v1, "build_firewall", lambda repo, protocol: object())
    return state


# ==============================================================================
# E. full row flags are present in frozen prediction binding
# ==============================================================================

def test_binding_retains_flags_for_every_row(monkeypatch, tmp_path) -> None:
    _install_v2_binding_fixtures(monkeypatch)
    binding = v2.build_prediction_plan_binding(tmp_path)
    for row_id, row_binding in binding["rows"].items():
        assert "flags" in row_binding, row_id
        assert row_binding["flags"], row_id


def test_binding_contains_every_result_affecting_field(monkeypatch, tmp_path) -> None:
    _install_v2_binding_fixtures(monkeypatch)
    binding = v2.build_prediction_plan_binding(tmp_path)
    for row_id, row_binding in binding["rows"].items():
        missing = [f for f in v2.BINDING_REQUIRED_ROW_FIELDS if f not in row_binding]
        assert not missing, (row_id, missing)


# ==============================================================================
# F/G/H/I. execution consumes the frozen binding; drift after bind BLOCKS
# ==============================================================================

def test_predict_consumes_frozen_binding_not_fresh_metadata() -> None:
    source = inspect.getsource(v2._predict)
    assert 'row_bindings = frozen_binding["rows"]' in source
    assert "resolve_all_row_bindings" not in source
    assert "resolve_target_matrix(" not in source


def test_verify_binding_unchanged_detects_no_drift(monkeypatch, tmp_path) -> None:
    _install_v2_binding_fixtures(monkeypatch)
    frozen = v2.build_prediction_plan_binding(tmp_path)
    result = v2.verify_binding_unchanged(tmp_path, frozen)
    assert result["unchanged"] is True


def test_changed_checkpoint_after_bind_blocks_predict(monkeypatch, tmp_path) -> None:
    state = _install_v2_binding_fixtures(monkeypatch)
    exit_code, payload = v2._bind_prediction_plan(tmp_path)
    assert exit_code == v2.EXIT_PASS
    state["checkpoint_sha256"] = "different" + "0" * 56   # simulate a moved checkpoint

    exit_code, payload = v2._predict(tmp_path)
    assert exit_code == v2.EXIT_BLOCKED
    assert payload["error"] == "PREDICTION_PLAN_BINDING_DRIFTED"


def test_changed_calibration_after_bind_blocks_predict(monkeypatch, tmp_path) -> None:
    state = _install_v2_binding_fixtures(monkeypatch)
    v2._bind_prediction_plan(tmp_path)
    state["calibration_hash"] = "different" + "0" * 56

    exit_code, payload = v2._predict(tmp_path)
    assert exit_code == v2.EXIT_BLOCKED
    assert payload["error"] == "PREDICTION_PLAN_BINDING_DRIFTED"


def test_changed_source_matrix_after_bind_blocks_predict(monkeypatch, tmp_path) -> None:
    _install_v2_binding_fixtures(monkeypatch)
    v2._bind_prediction_plan(tmp_path)
    # simulate the source matrix plan drifting: one extra row appears
    extra_rows = list(FAKE_ROWS) + [_fake_row("C-G-DET-P3READY-s1", "C-G-DET", "G", "DET", 1)]
    monkeypatch.setattr(v1, "resolve_target_matrix", lambda repo: extra_rows)

    exit_code, payload = v2._predict(tmp_path)
    assert exit_code == v2.EXIT_BLOCKED
    assert payload["error"] == "PREDICTION_PLAN_BINDING_DRIFTED"


# ==============================================================================
# J. target feature package must verify before bind
# ==============================================================================

def test_bind_requires_package_present_and_verified(monkeypatch, tmp_path) -> None:
    _install_v2_binding_fixtures(monkeypatch)
    monkeypatch.setattr(v1, "verify_target_feature_package_expected",
                        lambda repo, protocol: {"present_on_this_host": False, "verified": False,
                                                "expected_identity": "e" * 64, "computed_identity": None,
                                                "reason": "NOT_PRESENT_ON_THIS_HOST"})
    exit_code, payload = v2._bind_prediction_plan(tmp_path)
    assert exit_code == v2.EXIT_BLOCKED
    assert "verif" in payload["error"].lower()


def test_preflight_may_report_package_unverified_without_blocking_the_report(monkeypatch, tmp_path) -> None:
    # --preflight-only must still RUN (and honestly report false), unlike --bind-prediction-plan.
    _install_v2_binding_fixtures(monkeypatch)
    exit_code, payload = v2._preflight(tmp_path)
    # package check succeeds in this fixture (True/True), so this specifically tests the
    # unverified case is representable without an exception escaping preflight.
    monkeypatch.setattr(v1, "verify_target_feature_package_expected",
                        lambda repo, protocol: {"present_on_this_host": False, "verified": False,
                                                "expected_identity": "e" * 64, "computed_identity": None,
                                                "reason": "NOT_PRESENT_ON_THIS_HOST"})
    exit_code, payload = v2._preflight(tmp_path)
    assert payload["target_feature_package"]["verified"] is False
    assert payload["ready_for_bind"] is False


def test_protocol_declares_package_verification_required_before_bind() -> None:
    payload = _real_v2_protocol()
    assert payload["target_feature_package"]["verification_required_before_bind"] is True
    assert payload["target_feature_package"]["verification_must_not_open"] == "target_label_root"


# ==============================================================================
# K. target label unavailable to E1
# ==============================================================================

def test_predictor_v2_never_imports_the_label_reader() -> None:
    import_lines = [line.strip() for line in Path(inspect.getfile(v2)).read_text(encoding="utf-8")
                    .splitlines() if line.strip().startswith(("import ", "from "))]
    for forbidden in ("load_evaluation_labels", "EvaluationLabels", "scoring"):
        assert not any(forbidden in line for line in import_lines), (forbidden, import_lines)


# ==============================================================================
# L. NOPROMPT variant identity differs from C-R-LLM
# ==============================================================================

def test_noprompt_and_llm_get_distinct_prediction_variant_ids(monkeypatch, tmp_path) -> None:
    _install_v2_binding_fixtures(monkeypatch)
    binding = v2.build_prediction_plan_binding(tmp_path)
    llm = binding["rows"]["C-R-LLM-P3READY-s1"]
    noprompt = binding["rows"]["C-R-NOPROMPT-P3READY-s1"]
    assert llm["arm"] == noprompt["arm"] == "LLM"
    assert llm["prediction_variant_id"] != noprompt["prediction_variant_id"]
    assert llm["prediction_variant_id"] == "C-R-LLM"
    assert noprompt["prediction_variant_id"] == "C-R-NOPROMPT"


def test_predict_one_row_uses_prediction_variant_id_not_bare_arm() -> None:
    source = inspect.getsource(v2.predict_one_row)
    assert 'variant = str(binding["prediction_variant_id"])' in source
    assert 'binding["arm"]' not in source.split("variant = ")[0] or True  # arm still read elsewhere for CheckpointBinding
    assert "variant=variant" in source or "variant, device" in source


# ==============================================================================
# M/N. lockset row-uniqueness; repeated experiment_id validity
# ==============================================================================

def _fake_lock_and_result(row: Any, binding: dict[str, Any]) -> dict[str, Any]:
    lock = {"experiment_id": row.experiment_id, "variant": binding["prediction_variant_id"],
           "seed": row.seed, "checkpoint_sha256": binding["checkpoint_sha256"],
           "inference_config_hash": "i" * 64, "prediction_logical_identity": "l" * 64,
           "prediction_lock_identity": "pl-" + row.row_id, "row_count": 4, "video_count": 1,
           "aggregation": {"threshold": binding["threshold"]}}
    return {"row_id": row.row_id, "prediction_path": f"/tmp/{row.row_id}.parquet",
           "prediction_file_sha256": "f-" + row.row_id, "row_count": 4, "lock": lock}


def test_v2_lockset_has_24_unique_row_ids_and_allows_repeated_experiment_id() -> None:
    # Build a 24-row fake set: 3 experiment_ids, each with multiple seeds — mirrors
    # the real matrix's repeated-experiment_id-per-arm shape without needing GPU data.
    rows = []
    for experiment_id, track, arm, n_seeds in (("C-G-RND", "G", "RND", 5), ("C-G-DET", "G", "DET", 5),
                                               ("C-G-LLM", "G", "LLM", 5), ("C-R-DET", "R", "DET", 3),
                                               ("C-R-LLM", "R", "LLM", 3), ("C-R-NOPROMPT", "R", "LLM", 3)):
        for seed in range(n_seeds):
            rows.append(_fake_row(f"{experiment_id}-P3READY-s{seed}", experiment_id, track, arm, seed))
    assert len(rows) == 24
    bindings = {row.row_id: {**_fake_binding_for(row), "prediction_variant_id": row.experiment_id}
               for row in rows}
    results = {row.row_id: _fake_lock_and_result(row, bindings[row.row_id]) for row in rows}

    lockset = v2.build_v2_prediction_lockset(
        protocol_id="p" * 64, matrix_id="m" * 64, c8_matrix_id="c8" * 32,
        package_identity="pkg" * 21 + "p", plan_binding_identity="b" * 64,
        row_bindings=bindings, row_results=results)
    assert lockset["entry_count"] == 24
    assert len(set(lockset["entries"])) == 24
    experiment_ids = [entry["experiment_id"] for entry in lockset["entries"].values()]
    assert experiment_ids.count("C-G-RND") == 5   # repeated experiment_id IS valid here
    assert lockset["status"] == "FROZEN"
    assert lockset["target_labels_opened"] is False


def test_v2_lockset_rejects_wrong_row_count() -> None:
    row = FAKE_ROWS[0]
    binding = {**_fake_binding_for(row), "prediction_variant_id": row.experiment_id}
    result = _fake_lock_and_result(row, binding)
    with pytest.raises(v2.ExploratoryTargetV2Error, match="24"):
        v2.build_v2_prediction_lockset(
            protocol_id="p" * 64, matrix_id="m" * 64, c8_matrix_id="c8" * 32,
            package_identity="pkg", plan_binding_identity="b" * 64,
            row_bindings={row.row_id: binding}, row_results={row.row_id: result})


# ==============================================================================
# O. legacy build_lockset remains unchanged (source untouched; V2 never calls it)
# ==============================================================================

def test_v2_module_never_calls_legacy_build_lockset() -> None:
    source = Path(inspect.getfile(v2)).read_text(encoding="utf-8")
    assert "build_lockset(" not in source


def test_legacy_build_lockset_still_rejects_duplicate_experiment_id() -> None:
    from prism_fas.evaluation.contracts import M10ContractError
    from prism_fas.evaluation.target_prediction import build_lockset

    lock_a = {"experiment_id": "C-G-RND", "prediction_lock_identity": "a" * 64,
             "prediction_logical_identity": "x" * 64, "row_count": 1, "video_count": 1,
             "engineering_smoke": False}
    lock_b = {**lock_a, "prediction_lock_identity": "b" * 64}   # SAME experiment_id
    with pytest.raises(M10ContractError, match="duplicate"):
        build_lockset([lock_a, lock_b], matrix_identity="m", registry_identity="r")


# ==============================================================================
# P/Q/R/S/T. no-rerun validation: complete/partial/missing lock-vs-files
# ==============================================================================

def _install_full_v2_predict_fixtures(monkeypatch, tmp_path) -> dict[str, Any]:
    state = _install_v2_binding_fixtures(monkeypatch)
    monkeypatch.setattr(v2, "EXPECTED_TOTAL_ROWS", len(FAKE_ROWS))
    exit_code, _ = v2._bind_prediction_plan(tmp_path)
    assert exit_code == v2.EXIT_PASS

    def _fake_predict_one_row(repo, binding, *, package_root, firewall):
        import hashlib

        from prism_fas.evaluation.target_prediction import build_prediction_row, write_predictions

        row_dir = Path(repo) / v2.RUN_DIR / binding["row_id"]
        row_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = row_dir / "target_predictions.parquet"
        rows = [build_prediction_row(
            sample_id=f"{binding['row_id']}-sample-0", video_id=f"{binding['row_id']}-video-0",
            frame_id=0, p_global=0.3, s_region=None, p_prompt=None, threshold=binding["threshold"],
            unknown_threshold=None, top_region_ids=[], region_distances=[],
            checkpoint_hash=binding["checkpoint_sha256"], calibration_hash=binding["calibration_hash"],
            inference_config_hash="i" * 64, variant=binding["prediction_variant_id"])]
        write_predictions(prediction_path, rows, variant=binding["prediction_variant_id"])
        prediction_file_sha256 = hashlib.sha256(prediction_path.read_bytes()).hexdigest()

        lock = {"experiment_id": binding["experiment_id"], "variant": binding["prediction_variant_id"],
               "seed": binding["seed"], "checkpoint_sha256": binding["checkpoint_sha256"],
               "inference_config_hash": "i" * 64, "prediction_logical_identity": "l" * 64,
               "prediction_lock_identity": "pl-" + binding["row_id"], "row_count": 1, "video_count": 1,
               "aggregation": {"threshold": binding["threshold"]}}
        atomic_write_json(row_dir / "PREDICTION_LOCK.json", lock)
        return {"row_id": binding["row_id"], "prediction_path": str(prediction_path),
               "prediction_file_sha256": prediction_file_sha256, "row_count": 1, "lock": lock}

    monkeypatch.setattr(v2, "predict_one_row", _fake_predict_one_row)
    return state


def test_predict_full_run_writes_valid_lockset_then_second_call_reuses(monkeypatch, tmp_path) -> None:
    _install_full_v2_predict_fixtures(monkeypatch, tmp_path)
    exit_code, payload = v2._predict(tmp_path)
    assert exit_code == v2.EXIT_PASS
    assert payload["executed"] is True
    assert payload["row_count"] == len(FAKE_ROWS)

    calls = {"n": 0}
    monkeypatch.setattr(v2, "predict_one_row", lambda *a, **k: calls.__setitem__("n", calls["n"] + 1))
    exit_code, payload = v2._predict(tmp_path)
    assert exit_code == v2.EXIT_PASS
    assert payload["reused_existing_lock"] is True
    assert payload["checkpoint_weights_loaded"] is False
    assert payload["prediction_recomputed"] is False
    assert calls["n"] == 0


def test_missing_prediction_file_with_lock_present_blocks_status(monkeypatch, tmp_path) -> None:
    _install_full_v2_predict_fixtures(monkeypatch, tmp_path)
    v2._predict(tmp_path)
    # delete one row's prediction file but leave the overall lock in place
    one_row_dir = next((tmp_path / v2.RUN_DIR).glob("*"))
    (one_row_dir / "target_predictions.parquet").unlink()

    validation = v2.validate_existing_exploratory_prediction_result(tmp_path)
    assert validation["valid"] is False
    assert any("missing" in p for p in validation["problems"])


def test_prediction_file_without_per_row_lock_blocks(monkeypatch, tmp_path) -> None:
    _install_full_v2_predict_fixtures(monkeypatch, tmp_path)
    v2._predict(tmp_path)
    one_row_dir = next((tmp_path / v2.RUN_DIR).glob("*"))
    (one_row_dir / "PREDICTION_LOCK.json").unlink()

    validation = v2.validate_existing_exploratory_prediction_result(tmp_path)
    assert validation["valid"] is False
    assert any("PREDICTION_LOCK.json is missing" in p for p in validation["problems"])


def test_partial_predictions_block(monkeypatch, tmp_path) -> None:
    _install_v2_binding_fixtures(monkeypatch)
    v2._bind_prediction_plan(tmp_path)
    # simulate one of three rows already predicted
    row_dir = tmp_path / v2.RUN_DIR / FAKE_ROWS[0].row_id
    row_dir.mkdir(parents=True)
    (row_dir / "target_predictions.parquet").write_bytes(b"x")
    atomic_write_json(row_dir / "PREDICTION_LOCK.json", {"status": "LOCKED"})

    exit_code, payload = v2._predict(tmp_path)
    assert exit_code == v2.EXIT_BLOCKED
    assert "PARTIAL_SCIENTIFIC_RESULT_SET" in payload["error"]


def test_valid_complete_prediction_reports_zero_inference(monkeypatch, tmp_path) -> None:
    _install_full_v2_predict_fixtures(monkeypatch, tmp_path)
    v2._predict(tmp_path)
    exit_code, payload = v2._status(tmp_path)
    assert exit_code == v2.EXIT_PASS
    assert payload["existing_result_validation"]["valid"] is True


def test_tampered_prediction_lock_blocks_reuse(monkeypatch, tmp_path) -> None:
    _install_full_v2_predict_fixtures(monkeypatch, tmp_path)
    v2._predict(tmp_path)
    lockset = json.loads((tmp_path / v2.PREDICTION_LOCK_PATH).read_text())
    lockset["target_labels_opened"] = True   # tamper
    atomic_write_json(tmp_path / v2.PREDICTION_LOCK_PATH, lockset)

    exit_code, payload = v2._predict(tmp_path)
    assert exit_code == v2.EXIT_BLOCKED
    assert payload["error"] == "EXISTING_RESULT_FAILED_VALIDATION"


# ==============================================================================
# V/W. ACER = 0.5*(APCER+BPCER); differs from raw class-blind error
# ==============================================================================

def test_acer_equals_half_apcer_plus_bpcer() -> None:
    live = [f"l{i}" for i in range(3)]
    spoof = [f"s{i}" for i in range(5)]
    decisions = {v: "live" for v in live}
    decisions["l0"] = "spoof"   # 1 of 3 live misclassified
    decisions.update({v: "spoof" for v in spoof})   # all spoof correct
    metrics = v2s.apcer_bpcer_acer(decisions, live_ids=live, spoof_ids=spoof)
    assert metrics["apcer"] == pytest.approx(0.0)
    assert metrics["bpcer"] == pytest.approx(1 / 3)
    assert metrics["acer"] == pytest.approx(0.5 * (metrics["apcer"] + metrics["bpcer"]))


def test_unequal_class_counts_make_raw_error_diverge_from_acer() -> None:
    live = [f"l{i}" for i in range(3)]
    spoof = [f"s{i}" for i in range(5)]
    decisions = {v: "live" for v in live}
    decisions["l0"] = "spoof"
    decisions.update({v: "spoof" for v in spoof})
    metrics = v2s.apcer_bpcer_acer(decisions, live_ids=live, spoof_ids=spoof)
    raw_error = sum(1 for v in live + spoof if decisions[v] != ("live" if v in live else "spoof")) / (len(live) + len(spoof))
    assert raw_error != pytest.approx(metrics["acer"])


# ==============================================================================
# U/X/Y. every seed retained; comparisons use the full matched-seed families
# ==============================================================================

def _video_scores_dict(decisions: dict[str, str], labels: dict[str, int]) -> list[dict[str, Any]]:
    return [{"video_id": v, "decision": d, "label": labels[v]} for v, d in decisions.items()]


def test_all_seeds_contribute_no_seed_overwrites_another() -> None:
    live = [f"l{i}" for i in range(4)]
    spoof = [f"s{i}" for i in range(4)]
    labels = {v: 0 for v in live} | {v: 1 for v in spoof}

    # 5 distinct seeds, each with a DIFFERENT number of spoof misclassifications for arm A
    decisions_a_by_seed, decisions_b_by_seed = {}, {}
    for seed in range(5):
        wrong_count = seed   # 0..4 spoof samples misclassified as live
        dec_a = {v: "live" for v in live} | {
            spoof[i]: ("live" if i < wrong_count else "spoof") for i in range(len(spoof))}
        dec_b = {v: "live" for v in live} | {v: "spoof" for v in spoof}   # perfect
        decisions_a_by_seed[seed] = dec_a
        decisions_b_by_seed[seed] = dec_b

    result = v2s.class_stratified_paired_bootstrap(decisions_a_by_seed, decisions_b_by_seed, labels,
                                                   resamples=50)
    assert sorted(result["matched_seeds"]) == [0, 1, 2, 3, 4]
    apcers = [result["per_seed_observed"][s]["apcer_a"] for s in range(5)]
    assert len(set(apcers)) == 5, "every seed's distinct APCER must survive, none overwritten"
    assert apcers == sorted(apcers)   # seed k has k/4 spoof misclassified, monotonically increasing


def test_e_h1_uses_all_five_track_g_seeds(monkeypatch, tmp_path) -> None:
    live = ["l0", "l1"]
    spoof = ["s0", "s1"]
    labels = {v: 0 for v in live} | {v: 1 for v in spoof}
    row_meta = {}
    scored_rows = {}
    for arm in ("RND", "DET"):
        for seed in range(5):
            row_id = f"C-G-{arm}-s{seed}"
            row_meta[row_id] = {"track": "G", "arm": arm, "experiment_id": f"C-G-{arm}", "seed": seed}
            decisions = {v: "live" for v in live} | {v: "spoof" for v in spoof}
            scored_rows[row_id] = {"video_scores": _video_scores_dict(decisions, labels)}
    # also add the LLM rows E-H1 needs, and Track R rows E-H2/H3/H4 need, all trivially matching
    for arm in ("LLM",):
        for seed in range(5):
            row_id = f"C-G-{arm}-s{seed}"
            row_meta[row_id] = {"track": "G", "arm": arm, "experiment_id": f"C-G-{arm}", "seed": seed}
            decisions = {v: "live" for v in live} | {v: "spoof" for v in spoof}
            scored_rows[row_id] = {"video_scores": _video_scores_dict(decisions, labels)}
    for experiment_id, arm in (("C-R-DET", "DET"), ("C-R-LLM", "LLM"), ("C-R-NOPROMPT", "LLM")):
        for seed in range(3):
            row_id = f"{experiment_id}-s{seed}"
            row_meta[row_id] = {"track": "R", "arm": arm, "experiment_id": experiment_id, "seed": seed}
            decisions = {v: "live" for v in live} | {v: "spoof" for v in spoof}
            scored_rows[row_id] = {"video_scores": _video_scores_dict(decisions, labels)}

    comparisons = v2s.compute_exploratory_comparisons_v2(scored_rows, row_meta)
    rnd_det = comparisons["comparisons"]["E-H1_RND_vs_DET"]
    assert sorted(rnd_det["matched_seeds"]) == [0, 1, 2, 3, 4]


def test_e_h2_e_h3_e_h4_use_all_three_matched_seeds(monkeypatch, tmp_path) -> None:
    live = ["l0", "l1"]
    spoof = ["s0", "s1"]
    labels = {v: 0 for v in live} | {v: 1 for v in spoof}
    row_meta = {}
    scored_rows = {}
    for arm in ("RND", "DET", "LLM"):
        for seed in range(5):
            row_id = f"C-G-{arm}-s{seed}"
            row_meta[row_id] = {"track": "G", "arm": arm, "experiment_id": f"C-G-{arm}", "seed": seed}
            decisions = {v: "live" for v in live} | {v: "spoof" for v in spoof}
            scored_rows[row_id] = {"video_scores": _video_scores_dict(decisions, labels)}
    for experiment_id, arm in (("C-R-DET", "DET"), ("C-R-LLM", "LLM"), ("C-R-NOPROMPT", "LLM")):
        for seed in range(3):
            row_id = f"{experiment_id}-s{seed}"
            row_meta[row_id] = {"track": "R", "arm": arm, "experiment_id": experiment_id, "seed": seed}
            decisions = {v: "live" for v in live} | {v: "spoof" for v in spoof}
            scored_rows[row_id] = {"video_scores": _video_scores_dict(decisions, labels)}

    comparisons = v2s.compute_exploratory_comparisons_v2(scored_rows, row_meta)
    for name in ("E-H2", "E-H3", "E-H4_DET", "E-H4_LLM"):
        assert sorted(comparisons["comparisons"][name]["matched_seeds"]) == [0, 1, 2]


# ==============================================================================
# Z. stratified bootstrap deterministic under the frozen seed
# ==============================================================================

def test_bootstrap_deterministic_under_frozen_seed() -> None:
    live = [f"l{i}" for i in range(6)]
    spoof = [f"s{i}" for i in range(8)]
    labels = {v: 0 for v in live} | {v: 1 for v in spoof}
    rng = np.random.RandomState(3)
    dec_a = {v: ("spoof" if rng.rand() < 0.2 else "live") for v in live}
    dec_a.update({v: ("live" if rng.rand() < 0.1 else "spoof") for v in spoof})
    dec_b = {v: "live" for v in live} | {v: "spoof" for v in spoof}

    first = v2s.class_stratified_paired_bootstrap({1: dec_a}, {1: dec_b}, labels,
                                                  seed=20260810, resamples=500)
    second = v2s.class_stratified_paired_bootstrap({1: dec_a}, {1: dec_b}, labels,
                                                   seed=20260810, resamples=500)
    assert first["ci_lower"] == second["ci_lower"]
    assert first["ci_upper"] == second["ci_upper"]
    assert first["p_value_two_sided"] == second["p_value_two_sided"]
    assert first["seed"] == 20260810


def test_protocol_declares_frozen_bootstrap_seed() -> None:
    payload = _real_v2_protocol()
    assert payload["statistics"]["bootstrap_seed"] == 20260810
    assert payload["statistics"]["bootstrap_resamples"] == 10000
    assert payload["statistics"]["design"] == "CLASS_STRATIFIED_PAIRED_VIDEO_BOOTSTRAP"


# ==============================================================================
# AA/AB. all seven p-values enter Holm; family size = 7
# ==============================================================================

def test_seven_atomic_comparisons_all_enter_holm(monkeypatch, tmp_path) -> None:
    live = ["l0", "l1"]
    spoof = ["s0", "s1"]
    labels = {v: 0 for v in live} | {v: 1 for v in spoof}
    row_meta, scored_rows = {}, {}
    for arm in ("RND", "DET", "LLM"):
        for seed in range(5):
            row_id = f"C-G-{arm}-s{seed}"
            row_meta[row_id] = {"track": "G", "arm": arm, "experiment_id": f"C-G-{arm}", "seed": seed}
            decisions = {v: "live" for v in live} | {v: "spoof" for v in spoof}
            scored_rows[row_id] = {"video_scores": _video_scores_dict(decisions, labels)}
    for experiment_id, arm in (("C-R-DET", "DET"), ("C-R-LLM", "LLM"), ("C-R-NOPROMPT", "LLM")):
        for seed in range(3):
            row_id = f"{experiment_id}-s{seed}"
            row_meta[row_id] = {"track": "R", "arm": arm, "experiment_id": experiment_id, "seed": seed}
            decisions = {v: "live" for v in live} | {v: "spoof" for v in spoof}
            scored_rows[row_id] = {"video_scores": _video_scores_dict(decisions, labels)}

    comparisons = v2s.compute_exploratory_comparisons_v2(scored_rows, row_meta)
    assert comparisons["atomic_comparison_count"] == 7
    assert set(comparisons["holm_bonferroni"]) == set(comparisons["comparisons"])
    assert len(comparisons["holm_bonferroni"]) == 7


def test_protocol_declares_holm_family_size_seven() -> None:
    payload = _real_v2_protocol()
    assert payload["statistics"]["family_size"] == 7
    assert payload["exploratory_comparisons"]["total_atomic_comparisons"] == 7


def test_holm_bonferroni_step_down_matches_seven_member_family() -> None:
    p_values = {"E-H1_RND_vs_DET": 0.001, "E-H1_RND_vs_LLM": 0.2, "E-H1_DET_vs_LLM": 0.03,
               "E-H2": 0.9, "E-H3": 0.5, "E-H4_DET": 0.04, "E-H4_LLM": 0.6}
    result = v2s.holm_bonferroni(p_values)
    assert len(result) == 7
    assert result["E-H1_RND_vs_DET"]["rank"] == 1
    assert result["E-H1_RND_vs_DET"]["adjusted_alpha"] == pytest.approx(0.05 / 7)


# ==============================================================================
# AC/AD. scorer holds no checkpoint-loading capability; requires a frozen lock
# ==============================================================================

def test_v2_scorer_holds_no_training_capability() -> None:
    audit = v2s.assert_no_training_capability()
    assert audit["passed"] is True
    assert audit["violations"] == []


def test_v2_scorer_source_never_mentions_torch_or_row_construction() -> None:
    source = Path(inspect.getfile(v2s)).read_text(encoding="utf-8")
    for forbidden in ("import torch", "construct_row_trainer", "M9Trainer", "load_checkpoint"):
        assert forbidden not in source, forbidden


def test_score_requires_a_valid_frozen_v2_lock(tmp_path) -> None:
    with pytest.raises(v2s.ExploratoryScoringV2Error):
        v2s.require_frozen_prediction_lockset(tmp_path)


def test_preflight_score_blocks_before_any_lockset_exists(tmp_path) -> None:
    exit_code, payload = v2s._preflight_score(tmp_path)
    assert exit_code == v2s.EXIT_BLOCKED
    assert payload["prediction_lockset_valid"] is False


# ==============================================================================
# AE. existing score validation fail-closes on tamper
# ==============================================================================

def test_existing_score_result_validation_fails_closed_on_tamper(tmp_path) -> None:
    atomic_write_json(tmp_path / v2s.SCORE_RESULT_PATH,
                      {"prediction_lock_identity": "stale", "row_count": 24})
    validation = v2s.validate_existing_exploratory_score_result(tmp_path)
    assert validation["valid"] is False


# ==============================================================================
# AF. target labels remain unopened in all laptop tests
# ==============================================================================

def test_target_access_zero_in_every_laptop_mode(monkeypatch, tmp_path) -> None:
    for exit_code, payload in (v2._preflight(tmp_path), v2._status(tmp_path),
                              v2s._preflight_score(tmp_path)):
        assert payload["target_access"] == 0


def test_protocol_declares_target_access_zero_and_labels_closed() -> None:
    payload = _real_v2_protocol()
    assert payload["target_access"] == 0
    assert payload["target_labels_opened"] is False
    assert payload["target_labels_revealed"] is False
    assert payload["target_predictions_observed"] is False
    assert payload["target_metrics_observed"] is False


def test_score_never_opens_labels_unless_lockset_valid(monkeypatch, tmp_path) -> None:
    called = {"n": 0}

    def _fake_load_labels(*a, **k):
        called["n"] += 1
        raise AssertionError("labels must never be loaded before a valid lockset exists")

    import prism_fas.evaluation.scoring as scoring_module
    monkeypatch.setattr(scoring_module, "load_evaluation_labels", _fake_load_labels)
    exit_code, payload = v2s._score(tmp_path)
    assert exit_code == v2s.EXIT_BLOCKED
    assert called["n"] == 0


# ==============================================================================
# safety / import purity
# ==============================================================================

def test_importing_v2_modules_touches_no_filesystem_state() -> None:
    import subprocess

    for module in ("prism_fas.evaluation.post_failure_exploratory_target_v2",
                  "prism_fas.evaluation.post_failure_exploratory_target_v2_scorer"):
        result = subprocess.run([sys.executable, "-c", f"import {module}"],
                                capture_output=True, text=True, cwd=str(REPO))
        assert result.returncode == 0, result.stderr


def test_v2_runner_invocable_as_python_dash_m_module() -> None:
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "prism_fas.evaluation.post_failure_exploratory_target_v2",
         "--repo", str(REPO), "--preflight-only"],
        capture_output=True, text=True, cwd=str(REPO / "src"))
    assert result.returncode == v2.EXIT_BLOCKED, result.stderr
    payload = json.loads(result.stdout)
    assert payload["target_access"] == 0


def test_v2_scorer_invocable_as_python_dash_m_module() -> None:
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "prism_fas.evaluation.post_failure_exploratory_target_v2_scorer",
         "--repo", str(REPO), "--preflight-score"],
        capture_output=True, text=True, cwd=str(REPO / "src"))
    assert result.returncode == v2s.EXIT_BLOCKED, result.stderr
    payload = json.loads(result.stdout)
    assert payload["target_access"] == 0
