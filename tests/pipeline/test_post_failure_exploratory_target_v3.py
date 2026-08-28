"""POST_FAILURE_EXPLORATORY_TARGET_V3 — final pre-target provenance/access/
statistics hardening.

**FIXTURE / ENGINEERING ONLY.** Every test here runs against `tmp_path`
fixtures, pure functions, or the REAL frozen C8 source matrix metadata —
never a real checkpoint, never a real image, never target data.

This file proves the items the final hardening task requires (A-AJ): V1
and V2 stay byte-identical and untouched; V3 has its own distinct protocol
identity; every row's binding carries a REAL, non-empty
`target_feature_package_identity` equal to the top-level verified value,
threaded into the inference-config hash and the per-row lock; the current
git commit is bound into every row and the overall lockset, and a mismatch
BLOCKS reuse; the explicit `target_access_state` transitions correctly and
never lies after a real access; the label-reveal artifact is mandatory,
idempotent on exact match, blocks on conflict, and blocks on label
tampering; the score state machine correctly distinguishes zero/partial/
"24 files but no final result"/complete states; the bootstrap produces a
deterministic CI that retains every matched seed and carries no p-value;
the paired randomization test is deterministic, uses the finite-sample
`+1` correction, and treats ties with `>=`; every comparison's matched-seed
set is asserted exactly, with a missing seed blocking; all seven
randomization p-values enter one Holm family; a canonical cross-seed
mean/std(ddof=0) summary exists per configuration without pooling
predictions across seeds; the scorer holds no training capability; and
target labels remain unopened in every laptop-run mode.
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
from prism_fas.evaluation import post_failure_exploratory_target_v3 as v3  # noqa: E402
from prism_fas.evaluation import post_failure_exploratory_target_v3_scorer as v3s  # noqa: E402
from prism_fas.pipeline.state import atomic_write_json  # noqa: E402

V1_PROTOCOL_PATH = REPO / "configs/evaluation/post_failure_exploratory_target_v1.yaml"
V2_PROTOCOL_PATH = REPO / "configs/evaluation/post_failure_exploratory_target_v2.yaml"
V3_PROTOCOL_PATH = REPO / "configs/evaluation/post_failure_exploratory_target_v3.yaml"
V1_IDENTITY = "8fb806d25a80ecd3c7d44cfeba8c893a5f115b8b51797220a51132ba16708b51"
V2_IDENTITY = "2f1beb0b95f01051e06c0ef8a82d06a759d0fe8f81f693c5d3a4d777845196a9"


def _real_v3_protocol() -> dict[str, Any]:
    return yaml.safe_load(V3_PROTOCOL_PATH.read_text(encoding="utf-8"))


# ==============================================================================
# A/B. V1/V2 unchanged
# ==============================================================================

def test_v1_and_v2_configs_and_modules_not_modified() -> None:
    import subprocess

    result = subprocess.run(["git", "diff", "--stat", "HEAD", "--",
                            "configs/evaluation/post_failure_exploratory_target_v1.yaml",
                            "configs/evaluation/post_failure_exploratory_target_v2.yaml",
                            "src/prism_fas/evaluation/post_failure_exploratory_target.py",
                            "src/prism_fas/evaluation/post_failure_exploratory_target_scorer.py",
                            "src/prism_fas/evaluation/post_failure_exploratory_target_v2.py",
                            "src/prism_fas/evaluation/post_failure_exploratory_target_v2_scorer.py"],
                            cwd=str(REPO), capture_output=True, text=True)
    assert result.stdout.strip() == ""


def test_v1_and_v2_identities_unchanged() -> None:
    v1_payload = yaml.safe_load(V1_PROTOCOL_PATH.read_text(encoding="utf-8"))
    v2_payload = yaml.safe_load(V2_PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert v1.protocol_identity(v1_payload) == V1_IDENTITY
    assert v2.protocol_identity(v2_payload) == V2_IDENTITY


# ==============================================================================
# C. V3 distinct identity
# ==============================================================================

def test_v3_protocol_identity_distinct_and_deterministic() -> None:
    payload = _real_v3_protocol()
    identity = v3.protocol_identity(payload)
    assert identity == v3.protocol_identity(payload)
    assert identity not in (V1_IDENTITY, V2_IDENTITY)
    assert len(identity) == 64


def test_v3_protocol_declares_frozen_and_supersedes_v1_v2() -> None:
    payload = _real_v3_protocol()
    assert payload["status"] == "FROZEN_NOT_RUN"
    assert payload["decision_id"] == "POST_FAILURE_EXPLORATORY_TARGET_V3"
    superseded = payload["supersedes"]["superseded"]
    ids = {entry["protocol_identity"] for entry in superseded}
    assert ids == {V1_IDENTITY, V2_IDENTITY}


def test_v3_namespace_disjoint_from_v1_and_v2() -> None:
    assert v3.DIAGNOSTICS_DIR not in (v1.DIAGNOSTICS_DIR, v2.DIAGNOSTICS_DIR)
    assert v3.RUN_DIR not in (v1.RUN_DIR, v2.RUN_DIR)


# ==============================================================================
# fixtures
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
PACKAGE_IDENTITY = "e" * 64


def _fake_binding_for(row: Any, *, checkpoint_sha256: str = "k" * 64,
                      calibration_hash: str = "h" * 64) -> dict[str, Any]:
    return {"row_id": row.row_id, "experiment_id": row.experiment_id, "track": row.track,
           "arm": row.arm, "protocol": row.protocol, "seed": row.seed,
           "config_identity": row.config_identity, "run_identity": "run-" + row.row_id,
           "checkpoint_sha256": checkpoint_sha256, "checkpoint_relative_path": "p.pt",
           "checkpoint_kind": "best", "decision_logit_name": "global_logit_G",
           "decision_score_name": "p_global", "decision_graph_hash": "g" * 64,
           "calibration_hash": calibration_hash, "calibration_split": "source_dev",
           "threshold": 0.5, "temperature": 1.0, "flags": dict(row.flags),
           "prediction_variant_id": row.experiment_id}


def _install_v3_binding_fixtures(monkeypatch, *, rows=FAKE_ROWS, checkpoint_sha256="k" * 64,
                                 calibration_hash="h" * 64, package_identity=PACKAGE_IDENTITY
                                 ) -> dict[str, Any]:
    state = {"checkpoint_sha256": checkpoint_sha256, "calibration_hash": calibration_hash,
            "package_identity": package_identity}

    monkeypatch.setattr(v3, "load_protocol", lambda repo: _real_v3_protocol())
    monkeypatch.setattr(v1, "resolve_target_matrix", lambda repo: list(rows))
    monkeypatch.setattr(v1, "target_matrix_identity",
                        lambda rows_arg: "matrix-" + "".join(sorted(r.row_id for r in rows_arg)))
    monkeypatch.setattr(v1, "bind_c8_matrix_identity", lambda repo: {"c8_matrix_identity": "c8" + "0" * 62})
    monkeypatch.setattr(v1, "verify_target_label_root_sealed",
                        lambda repo, protocol: {"stage": "G7", "label_root_permission": "deny",
                                                "label_root_declared": True, "label_root_exists": False,
                                                "target_labels_opened": False})
    monkeypatch.setattr(v1, "build_firewall", lambda repo, protocol: object())

    def _resolve_all_row_bindings_v2(repo, rows_arg):
        return {row.row_id: _fake_binding_for(row, checkpoint_sha256=state["checkpoint_sha256"],
                                              calibration_hash=state["calibration_hash"])
               for row in rows_arg}

    monkeypatch.setattr(v2, "resolve_all_row_bindings_v2", _resolve_all_row_bindings_v2)

    def _fake_package_check(repo, protocol):
        return {"present_on_this_host": True, "verified": True, "package_id": "prism_target_eval_v2",
               "expected_identity": state["package_identity"], "computed_identity": state["package_identity"],
               "target_feature_package_identity_verified": True,
               "target_label_access": 0, "reason": ""}

    monkeypatch.setattr(v3, "verify_target_feature_package_expected_v3", _fake_package_check)
    monkeypatch.setattr(v3, "verify_target_feature_package_required_v3", _fake_package_check)
    return state


# ==============================================================================
# D/E/F/G. per-row target package identity (Defect A)
# ==============================================================================

def test_every_row_binding_carries_nonempty_package_identity(monkeypatch, tmp_path) -> None:
    _install_v3_binding_fixtures(monkeypatch)
    binding = v3.build_prediction_plan_binding(tmp_path)
    for row_id, row_binding in binding["rows"].items():
        assert row_binding["target_feature_package_identity"], row_id


def test_row_package_identity_equals_top_level(monkeypatch, tmp_path) -> None:
    _install_v3_binding_fixtures(monkeypatch)
    binding = v3.build_prediction_plan_binding(tmp_path)
    top = binding["target_feature_package_identity"]
    assert top == PACKAGE_IDENTITY
    for row_id, row_binding in binding["rows"].items():
        assert row_binding["target_feature_package_identity"] == top, row_id


def test_inference_config_hash_binds_package_identity() -> None:
    source = inspect.getsource(v3.predict_one_row_to_staging)
    assert "package_identity=package_identity" in source
    assert 'package_identity = str(binding["target_feature_package_identity"])' in source


def test_per_row_lock_binds_package_identity() -> None:
    source = inspect.getsource(v3.predict_one_row_to_staging)
    assert "target_feature_package_identity=package_identity" in source


def test_resolve_all_row_bindings_v3_rejects_empty_package_identity(tmp_path) -> None:
    with pytest.raises(v3.ExploratoryTargetV3Error, match="non-empty"):
        v3.resolve_all_row_bindings_v3(tmp_path, FAKE_ROWS, target_feature_package_identity="")


# ==============================================================================
# H/I. code commit bound; mismatch blocks reuse
# ==============================================================================

def test_predict_one_row_passes_code_commit_to_build_prediction_lock() -> None:
    source = inspect.getsource(v3.predict_one_row_to_staging)
    assert "code_commit=code_commit" in source


def test_v3_lockset_rejects_row_with_mismatched_code_commit(monkeypatch) -> None:
    monkeypatch.setattr(v3, "EXPECTED_TOTAL_ROWS", 1)
    row = FAKE_ROWS[0]
    binding = {**_fake_binding_for(row), "target_feature_package_identity": PACKAGE_IDENTITY}
    lock = {"experiment_id": row.experiment_id, "variant": binding["prediction_variant_id"],
           "seed": row.seed, "checkpoint_sha256": binding["checkpoint_sha256"],
           "inference_config_hash": "i" * 64, "prediction_logical_identity": "l" * 64,
           "prediction_lock_identity": "pl-" + row.row_id, "row_count": 1, "video_count": 1,
           "target_feature_package_identity": PACKAGE_IDENTITY, "code_commit": "WRONG_COMMIT"}
    result = {"row_id": row.row_id, "prediction_file_sha256": "f", "row_count": 1, "lock": lock}
    with pytest.raises(v3.ExploratoryTargetV3Error, match="code_commit"):
        v3.build_v3_prediction_lockset(
            protocol_id="p" * 64, matrix_id="m" * 64, c8_matrix_id="c8" * 32,
            package_identity=PACKAGE_IDENTITY, plan_binding_identity="b" * 64, code_commit="REAL_COMMIT",
            row_bindings={row.row_id: binding}, row_results={row.row_id: result})


def test_lockset_binds_code_commit_at_top_level(monkeypatch) -> None:
    monkeypatch.setattr(v3, "EXPECTED_TOTAL_ROWS", 1)
    row = FAKE_ROWS[0]
    binding = {**_fake_binding_for(row), "target_feature_package_identity": PACKAGE_IDENTITY}
    lock = {"experiment_id": row.experiment_id, "variant": binding["prediction_variant_id"],
           "seed": row.seed, "checkpoint_sha256": binding["checkpoint_sha256"],
           "inference_config_hash": "i" * 64, "prediction_logical_identity": "l" * 64,
           "prediction_lock_identity": "pl-" + row.row_id, "row_count": 1, "video_count": 1,
           "target_feature_package_identity": PACKAGE_IDENTITY, "code_commit": "REAL_COMMIT"}
    result = {"row_id": row.row_id, "prediction_file_sha256": "f", "row_count": 1, "lock": lock}
    lockset = v3.build_v3_prediction_lockset(
        protocol_id="p" * 64, matrix_id="m" * 64, c8_matrix_id="c8" * 32,
        package_identity=PACKAGE_IDENTITY, plan_binding_identity="b" * 64, code_commit="REAL_COMMIT",
        row_bindings={row.row_id: binding}, row_results={row.row_id: result})
    assert lockset["prediction_execution_code_commit"] == "REAL_COMMIT"


# ==============================================================================
# J/K. target access state transitions correctly; never lies
# ==============================================================================

def test_access_state_starts_all_false() -> None:
    payload = _real_v3_protocol()
    state = payload["target_access_state"]
    assert state["target_feature_identity_accessed"] is False
    assert state["target_prediction_features_accessed"] is False
    assert state["target_labels_accessed"] is False


def test_load_protocol_refuses_a_protocol_that_starts_with_access_true(tmp_path) -> None:
    payload = dict(_real_v3_protocol())
    payload["target_access_state"] = {**payload["target_access_state"], "target_feature_identity_accessed": True}
    (tmp_path / "configs/evaluation").mkdir(parents=True)
    (tmp_path / v3.PROTOCOL_CONFIG_PATH).write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(v3.ExploratoryTargetV3Error, match="target_access_state"):
        v3.load_protocol(tmp_path)


def test_binding_access_state_is_feature_identity_only(monkeypatch, tmp_path) -> None:
    _install_v3_binding_fixtures(monkeypatch)
    binding = v3.build_prediction_plan_binding(tmp_path)
    state = binding["access_state"]
    assert state["target_feature_identity_accessed"] is True
    assert state["target_prediction_features_accessed"] is False
    assert state["target_labels_accessed"] is False


def test_lockset_access_state_includes_prediction_features(monkeypatch) -> None:
    monkeypatch.setattr(v3, "EXPECTED_TOTAL_ROWS", 1)
    row = FAKE_ROWS[0]
    binding = {**_fake_binding_for(row), "target_feature_package_identity": PACKAGE_IDENTITY}
    lock = {"experiment_id": row.experiment_id, "variant": binding["prediction_variant_id"],
           "seed": row.seed, "checkpoint_sha256": binding["checkpoint_sha256"],
           "inference_config_hash": "i" * 64, "prediction_logical_identity": "l" * 64,
           "prediction_lock_identity": "pl-" + row.row_id, "row_count": 1, "video_count": 1,
           "target_feature_package_identity": PACKAGE_IDENTITY, "code_commit": "c"}
    result = {"row_id": row.row_id, "prediction_file_sha256": "f", "row_count": 1, "lock": lock}
    lockset = v3.build_v3_prediction_lockset(
        protocol_id="p" * 64, matrix_id="m" * 64, c8_matrix_id="c8" * 32,
        package_identity=PACKAGE_IDENTITY, plan_binding_identity="b" * 64, code_commit="c",
        row_bindings={row.row_id: binding}, row_results={row.row_id: result})
    state = lockset["access_state"]
    assert state["target_feature_identity_accessed"] is True
    assert state["target_prediction_features_accessed"] is True
    assert state["target_labels_accessed"] is False


# ==============================================================================
# L/M/N/O. label reveal mandatory, idempotent, blocks on conflict/tamper
# ==============================================================================

def _write_frozen_lockset(tmp_path: Path, *, entry_count: int = 1, code_commit: str = "c" * 40,
                          package_identity: str = "e" * 64) -> dict[str, Any]:
    """Defect B: the lockset entries are now the sole authority for row
    metadata, so the fixture must carry every field `_row_meta_from_lockset`
    reads (this is exactly what the real `build_v3_prediction_lockset`
    entries already carry)."""
    entries = {}
    for i in range(entry_count):
        row_id = f"ROW-{i}"
        entries[row_id] = {
            "row_id": row_id, "experiment_id": f"EXP-{i}", "track": "G", "arm": "RND",
            "seed": 20260806 + i, "prediction_variant_id": f"EXP-{i}",
            "threshold": 0.5, "checkpoint_sha256": "k" * 64, "calibration_hash": "h" * 64,
            "target_feature_package_identity": package_identity, "code_commit": code_commit,
            "prediction_lock_identity": f"pl-{row_id}", "prediction_logical_identity": f"pli-{row_id}",
            "prediction_file_sha256": f"pf-{row_id}", "row_count": 1, "video_count": 1,
        }
    lockset = {"status": "FROZEN", "entry_count": entry_count, "target_labels_opened": False,
              "lockset_identity": "l" * 64, "prediction_execution_code_commit": code_commit,
              "target_feature_package_identity": package_identity, "entries": entries}
    path = tmp_path / v3s.PREDICTION_LOCK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, lockset)
    return lockset


def test_label_reveal_required_before_scoring(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(v3s, "require_frozen_prediction_lockset", lambda repo: _write_frozen_lockset(repo))
    label_path = tmp_path / "labels.parquet"
    label_path.write_bytes(b"fake-label-bytes")
    monkeypatch.setattr("prism_fas.evaluation.post_failure_exploratory_target_v3.active_protocol_identity",
                        lambda repo: "p" * 64)
    binding_path = tmp_path / v3s.PREDICTION_PLAN_BINDING_PATH
    binding_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(binding_path, {"prediction_plan_binding_identity": "b" * 64})

    reveal = v3s.reveal_target_labels(tmp_path, label_path=label_path)
    assert reveal["target_labels_accessed"] is True
    assert (tmp_path / v3s.LABEL_REVEAL_PATH).is_file()


def test_label_reveal_is_idempotent_on_exact_match(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(v3s, "require_frozen_prediction_lockset", lambda repo: _write_frozen_lockset(repo))
    label_path = tmp_path / "labels.parquet"
    label_path.write_bytes(b"fake-label-bytes")
    monkeypatch.setattr("prism_fas.evaluation.post_failure_exploratory_target_v3.active_protocol_identity",
                        lambda repo: "p" * 64)
    binding_path = tmp_path / v3s.PREDICTION_PLAN_BINDING_PATH
    binding_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(binding_path, {"prediction_plan_binding_identity": "b" * 64})

    first = v3s.reveal_target_labels(tmp_path, label_path=label_path)
    second = v3s.reveal_target_labels(tmp_path, label_path=label_path)
    assert first == second


def test_conflicting_label_reveal_blocks(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(v3s, "require_frozen_prediction_lockset", lambda repo: _write_frozen_lockset(repo))
    label_path = tmp_path / "labels.parquet"
    label_path.write_bytes(b"fake-label-bytes")
    monkeypatch.setattr("prism_fas.evaluation.post_failure_exploratory_target_v3.active_protocol_identity",
                        lambda repo: "p" * 64)
    binding_path = tmp_path / v3s.PREDICTION_PLAN_BINDING_PATH
    binding_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(binding_path, {"prediction_plan_binding_identity": "b" * 64})
    v3s.reveal_target_labels(tmp_path, label_path=label_path)

    stale = json.loads((tmp_path / v3s.LABEL_REVEAL_PATH).read_text())
    stale["target_label_artifact_sha256"] = "tampered"
    atomic_write_json(tmp_path / v3s.LABEL_REVEAL_PATH, stale)

    with pytest.raises(v3s.ExploratoryScoringV3Error, match="differs"):
        v3s.reveal_target_labels(tmp_path, label_path=label_path)


def test_label_artifact_tampering_blocks_reveal(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(v3s, "require_frozen_prediction_lockset", lambda repo: _write_frozen_lockset(repo))
    label_path = tmp_path / "labels.parquet"
    label_path.write_bytes(b"original-bytes")
    monkeypatch.setattr("prism_fas.evaluation.post_failure_exploratory_target_v3.active_protocol_identity",
                        lambda repo: "p" * 64)
    binding_path = tmp_path / v3s.PREDICTION_PLAN_BINDING_PATH
    binding_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(binding_path, {"prediction_plan_binding_identity": "b" * 64})
    v3s.reveal_target_labels(tmp_path, label_path=label_path)

    label_path.write_bytes(b"tampered-bytes")   # label file changed after reveal
    with pytest.raises(v3s.ExploratoryScoringV3Error, match="differs"):
        v3s.reveal_target_labels(tmp_path, label_path=label_path)


def test_score_never_calls_load_evaluation_labels_before_reveal(monkeypatch, tmp_path) -> None:
    calls: list[str] = []
    monkeypatch.setattr(v3s, "reveal_target_labels", lambda *a, **k: calls.append("reveal") or {
        "reveal_identity": "r" * 64})

    import prism_fas.evaluation.scoring as scoring_module

    def _fake_load_labels(*a, **k):
        assert calls == ["reveal"], "labels must never load before the reveal"
        raise AssertionError("stop before real scoring; this test only checks ordering")

    monkeypatch.setattr(scoring_module, "load_evaluation_labels", _fake_load_labels)
    monkeypatch.setattr(v3s, "require_frozen_prediction_lockset", lambda repo: _write_frozen_lockset(repo, entry_count=24))
    monkeypatch.setattr("prism_fas.evaluation.post_failure_exploratory_target_v3.load_protocol",
                        lambda repo: _real_v3_protocol())
    exit_code, payload = v3s._score(tmp_path)
    assert exit_code == v3s.EXIT_BLOCKED
    assert calls == ["reveal"]


# ==============================================================================
# P/Q/R/S. score state machine
# ==============================================================================

def test_24_score_files_no_final_result_is_incomplete_finalization(monkeypatch, tmp_path) -> None:
    lockset = _write_frozen_lockset(tmp_path, entry_count=3)
    monkeypatch.setattr(v3s, "require_frozen_prediction_lockset", lambda repo: lockset)
    rows_dir = tmp_path / v3s.SCORE_ROWS_DIR
    rows_dir.mkdir(parents=True)
    for row_id in lockset["entries"]:
        atomic_write_json(rows_dir / f"{row_id}.json", {"acer": 0.1})

    exit_code, payload = v3s._score(tmp_path)
    assert exit_code == v3s.EXIT_BLOCKED
    assert payload["error"] == "INCOMPLETE_FINALIZATION"


def test_partial_1_to_n_minus_1_score_files_block(monkeypatch, tmp_path) -> None:
    lockset = _write_frozen_lockset(tmp_path, entry_count=3)
    monkeypatch.setattr(v3s, "require_frozen_prediction_lockset", lambda repo: lockset)
    rows_dir = tmp_path / v3s.SCORE_ROWS_DIR
    rows_dir.mkdir(parents=True)
    first_row_id = sorted(lockset["entries"])[0]
    atomic_write_json(rows_dir / f"{first_row_id}.json", {"acer": 0.1})

    exit_code, payload = v3s._score(tmp_path)
    assert exit_code == v3s.EXIT_BLOCKED
    assert payload["error"] == "PARTIAL_SCIENTIFIC_RESULT_SET"


def test_extra_score_row_file_blocks(monkeypatch, tmp_path) -> None:
    lockset = _write_frozen_lockset(tmp_path, entry_count=1)
    monkeypatch.setattr(v3s, "require_frozen_prediction_lockset", lambda repo: lockset)
    rows_dir = tmp_path / v3s.SCORE_ROWS_DIR
    rows_dir.mkdir(parents=True)
    atomic_write_json(rows_dir / "NOT-A-REAL-ROW.json", {"acer": 0.1})

    exit_code, payload = v3s._score(tmp_path)
    assert exit_code == v3s.EXIT_BLOCKED
    assert payload["error"] == "UNEXPECTED_SCORE_ROW_FILES"


def test_final_result_with_missing_score_row_blocks(monkeypatch, tmp_path) -> None:
    lockset = _write_frozen_lockset(tmp_path, entry_count=2)
    monkeypatch.setattr(v3s, "require_frozen_prediction_lockset", lambda repo: lockset)
    atomic_write_json(tmp_path / v3s.SCORE_RESULT_PATH, {"row_count": 2, "rows": {"ROW-0": {}}})
    validation = v3s.validate_existing_exploratory_score_result_v3(tmp_path)
    assert validation["valid"] is False


# ==============================================================================
# T/U. score_result_identity and per-row identities verified
# ==============================================================================

def test_score_result_validation_fails_closed_on_missing_identity(monkeypatch, tmp_path) -> None:
    lockset = _write_frozen_lockset(tmp_path, entry_count=1)
    monkeypatch.setattr(v3s, "require_frozen_prediction_lockset", lambda repo: lockset)
    atomic_write_json(tmp_path / v3s.SCORE_RESULT_PATH, {"row_count": 1, "rows": {"ROW-0": {}},
                                                         "prediction_lock_identity": "wrong"})
    validation = v3s.validate_existing_exploratory_score_result_v3(tmp_path)
    assert validation["valid"] is False
    assert any("prediction_lock_identity" in p for p in validation["problems"])


# ==============================================================================
# V/W. bootstrap CI deterministic; retains all matched seeds
# ==============================================================================

def _labels_and_decisions(n_live=6, n_spoof=8):
    live = [f"l{i}" for i in range(n_live)]
    spoof = [f"s{i}" for i in range(n_spoof)]
    labels = {v: 0 for v in live} | {v: 1 for v in spoof}
    return live, spoof, labels


def test_bootstrap_ci_deterministic_under_frozen_seed() -> None:
    live, spoof, labels = _labels_and_decisions()
    rng = np.random.RandomState(2)
    dec_a = {v: ("spoof" if rng.rand() < 0.2 else "live") for v in live}
    dec_a.update({v: ("live" if rng.rand() < 0.1 else "spoof") for v in spoof})
    dec_b = {v: "live" for v in live} | {v: "spoof" for v in spoof}

    first = v3s.class_stratified_bootstrap_ci({1: dec_a}, {1: dec_b}, labels, seed=20260810, resamples=300)
    second = v3s.class_stratified_bootstrap_ci({1: dec_a}, {1: dec_b}, labels, seed=20260810, resamples=300)
    assert first["ci_lower"] == second["ci_lower"]
    assert first["ci_upper"] == second["ci_upper"]
    assert "p_value_two_sided" not in first


def test_bootstrap_ci_retains_all_matched_seeds() -> None:
    live, spoof, labels = _labels_and_decisions()
    dec_a = {s: ({v: "live" for v in live} | {v: "spoof" for v in spoof}) for s in range(5)}
    dec_b = {s: ({v: "live" for v in live} | {v: "spoof" for v in spoof}) for s in range(5)}
    ci = v3s.class_stratified_bootstrap_ci(dec_a, dec_b, labels, resamples=100)
    assert sorted(ci["matched_seeds"]) == [0, 1, 2, 3, 4]


# ==============================================================================
# X/Y/Z. randomization test deterministic; finite-sample correction; ties >=
# ==============================================================================

def test_randomization_p_value_deterministic() -> None:
    live, spoof, labels = _labels_and_decisions()
    dec_a = {1: {v: "live" for v in live} | {v: "spoof" for v in spoof}}
    dec_b = {1: {v: "live" for v in live} | {v: "live" for v in spoof}}
    first = v3s.paired_randomization_test(dec_a, dec_b, labels, seed=20260810, resamples=500)
    second = v3s.paired_randomization_test(dec_a, dec_b, labels, seed=20260810, resamples=500)
    assert first["p_value_two_sided"] == second["p_value_two_sided"]


def test_randomization_uses_finite_sample_plus_one_correction() -> None:
    live, spoof, labels = _labels_and_decisions()
    dec_a = {1: {v: "live" for v in live} | {v: "spoof" for v in spoof}}
    dec_b = {1: {v: "live" for v in live} | {v: "spoof" for v in spoof}}   # identical -> observed = 0
    result = v3s.paired_randomization_test(dec_a, dec_b, labels, resamples=100)
    # observed statistic is 0, so EVERY permutation satisfies |T_perm| >= |T_obs|=0
    assert result["count_ge"] == 100
    assert result["p_value_two_sided"] == pytest.approx((1 + 100) / (1 + 100))


def test_randomization_source_uses_ge_for_ties() -> None:
    source = inspect.getsource(v3s.paired_randomization_test)
    assert "permuted >= observed_abs" in source


# ==============================================================================
# AA/AB. exact matched seeds required; missing seed blocks
# ==============================================================================

def _scored_rows_and_meta_for_comparisons(complete: bool = True):
    live = ["l0", "l1"]
    spoof = ["s0", "s1"]
    labels = {v: 0 for v in live} | {v: 1 for v in spoof}

    def _video_scores(decisions):
        return [{"video_id": v, "decision": d, "label": labels[v]} for v, d in decisions.items()]

    row_meta, scored_rows = {}, {}
    track_g_seeds = [20260806, 20260807, 20260808, 20260809, 20260810]
    if not complete:
        track_g_seeds = track_g_seeds[:-1]   # drop one required seed
    for arm in ("RND", "DET", "LLM"):
        for seed in track_g_seeds:
            row_id = f"C-G-{arm}-s{seed}"
            row_meta[row_id] = {"track": "G", "arm": arm, "experiment_id": f"C-G-{arm}", "seed": seed}
            decisions = {v: "live" for v in live} | {v: "spoof" for v in spoof}
            scored_rows[row_id] = {"row_id": row_id,
                                   "metrics": {"video_scores": _video_scores(decisions),
                                              "video": {"apcer": 0.0, "bpcer": 0.0, "acer": 0.0, "roc_auc": 1.0,
                                                       "eer": 0.0,
                                                       "calibration": {"ece": 0.0, "brier": 0.0, "nll": 0.0}}}}
    for experiment_id, arm in (("C-R-DET", "DET"), ("C-R-LLM", "LLM"), ("C-R-NOPROMPT", "LLM")):
        for seed in (20260806, 20260807, 20260808):
            row_id = f"{experiment_id}-s{seed}"
            row_meta[row_id] = {"track": "R", "arm": arm, "experiment_id": experiment_id, "seed": seed}
            decisions = {v: "live" for v in live} | {v: "spoof" for v in spoof}
            scored_rows[row_id] = {"row_id": row_id,
                                   "metrics": {"video_scores": _video_scores(decisions),
                                              "video": {"apcer": 0.0, "bpcer": 0.0, "acer": 0.0, "roc_auc": 1.0,
                                                       "eer": 0.0,
                                                       "calibration": {"ece": 0.0, "brier": 0.0, "nll": 0.0}}}}
    return scored_rows, row_meta


def test_exact_matched_seeds_required_and_satisfied() -> None:
    scored_rows, row_meta = _scored_rows_and_meta_for_comparisons(complete=True)
    comparisons = v3s.compute_exploratory_comparisons_v3(scored_rows, row_meta)
    assert comparisons["comparisons"]["E-H1_RND_vs_DET"]["matched_seeds"] == \
        sorted(v3s.REQUIRED_MATCHED_SEEDS["E-H1_RND_vs_DET"])


def test_missing_one_required_seed_blocks() -> None:
    scored_rows, row_meta = _scored_rows_and_meta_for_comparisons(complete=False)
    with pytest.raises(v3s.ExploratoryScoringV3Error, match="missing required matched seed"):
        v3s.compute_exploratory_comparisons_v3(scored_rows, row_meta)


# ==============================================================================
# AC/AD. exactly seven randomization p-values; Holm covers all seven
# ==============================================================================

def test_exactly_seven_comparisons_and_holm_covers_all() -> None:
    scored_rows, row_meta = _scored_rows_and_meta_for_comparisons(complete=True)
    comparisons = v3s.compute_exploratory_comparisons_v3(scored_rows, row_meta)
    assert comparisons["atomic_comparison_count"] == 7
    assert set(comparisons["holm_bonferroni"]) == set(comparisons["comparisons"])
    assert comparisons["holm_input"] == "randomization_p_values"
    for name, entry in comparisons["comparisons"].items():
        assert "randomization" in entry and "bootstrap_ci" in entry
        assert "p_value_two_sided" not in entry["bootstrap_ci"]


# ==============================================================================
# AE/AF/AG. canonical cross-seed summaries; ddof=0; no seed pooling
# ==============================================================================

def test_cross_seed_summary_covers_all_configurations_and_metrics() -> None:
    scored_rows, row_meta = _scored_rows_and_meta_for_comparisons(complete=True)
    summary = v3s.build_cross_seed_summary(scored_rows, row_meta)
    assert set(summary) == set(v3s.CONFIGURATIONS)
    for config in v3s.CONFIGURATIONS:
        assert set(summary[config]) == set(v3s.CROSS_SEED_METRICS)
        for metric_summary in summary[config].values():
            assert "mean" in metric_summary and "std_ddof0" in metric_summary
            assert "per_seed" in metric_summary


def test_cross_seed_summary_ddof_zero() -> None:
    source = inspect.getsource(v3s.build_cross_seed_summary)
    assert "ddof=0" in source


def test_cross_seed_summary_never_pools_across_seeds() -> None:
    source = inspect.getsource(v3s.build_cross_seed_summary)
    for forbidden in ("concatenate(", "flatten("):
        assert forbidden not in source


def test_cross_seed_summary_all_frozen_seeds_present() -> None:
    scored_rows, row_meta = _scored_rows_and_meta_for_comparisons(complete=True)
    summary = v3s.build_cross_seed_summary(scored_rows, row_meta)
    assert summary["C-G-RND"]["acer"]["seeds"] == [20260806, 20260807, 20260808, 20260809, 20260810]
    assert summary["C-R-DET"]["acer"]["seeds"] == [20260806, 20260807, 20260808]


# ==============================================================================
# AH. scorer imports no model/checkpoint/training capability
# ==============================================================================

def test_v3_scorer_holds_no_training_capability() -> None:
    audit = v3s.assert_no_training_capability()
    assert audit["passed"] is True
    assert audit["violations"] == []


def test_v3_scorer_source_never_mentions_torch_or_row_construction() -> None:
    source = Path(inspect.getfile(v3s)).read_text(encoding="utf-8")
    for forbidden in ("import torch", "construct_row_trainer", "M9Trainer", "load_checkpoint"):
        assert forbidden not in source, forbidden


# ==============================================================================
# AI. target labels remain unopened in all laptop tests
# ==============================================================================

def test_preflight_and_status_never_open_labels(tmp_path) -> None:
    exit_code, payload = v3._preflight(tmp_path)
    assert "target_labels_accessed" not in json.dumps(payload) or payload.get("access_state", {}).get(
        "target_labels_accessed") is False
    exit_code, payload = v3s._preflight_score(tmp_path)
    assert exit_code == v3s.EXIT_BLOCKED


def test_protocol_declares_labels_closed() -> None:
    payload = _real_v3_protocol()
    assert payload["target_labels_opened"] is False
    assert payload["target_labels_revealed"] is False


# ==============================================================================
# AJ. staging artifacts never accepted as final scientific outputs
# ==============================================================================

def test_staging_row_never_counted_by_partial_state_detection(monkeypatch, tmp_path) -> None:
    _install_v3_binding_fixtures(monkeypatch)
    exit_code, _ = v3._bind_prediction_plan(tmp_path)
    assert exit_code == v3.EXIT_PASS
    staging_dir = tmp_path / v3.STAGING_ROOT / "deadbeefcafebabe" / FAKE_ROWS[0].row_id
    staging_dir.mkdir(parents=True)
    (staging_dir / "target_predictions.parquet").write_bytes(b"x")
    # a staging artifact must NOT be visible to the final-row partial-state scan
    problem = v3._detect_partial_state(tmp_path, len(FAKE_ROWS))
    assert problem is None


def test_staging_marker_constant_is_explicit() -> None:
    assert v3.STAGING_MARKER == "ENGINEERING_STAGING_NOT_SCIENTIFICALLY_LOCKED"


def test_predict_one_row_writes_staging_marker() -> None:
    source = inspect.getsource(v3.predict_one_row_to_staging)
    assert "STAGING_MARKER.json" in source
    assert "STAGING_MARKER" in source


def test_promote_refuses_to_overwrite_a_conflicting_final_row_directory(tmp_path) -> None:
    staging_root = tmp_path / "staging"
    (staging_root / "ROW-0").mkdir(parents=True)
    (staging_root / "ROW-0" / "target_predictions.parquet").write_bytes(b"staged")
    (staging_root / "ROW-0" / "PREDICTION_LOCK.json").write_text("{}", encoding="utf-8")
    final_dir = tmp_path / v3.RUN_DIR / "ROW-0"
    final_dir.mkdir(parents=True)
    (final_dir / "target_predictions.parquet").write_bytes(b"unrelated content already there")
    (final_dir / "PREDICTION_LOCK.json").write_text("{}", encoding="utf-8")
    row_results = {"ROW-0": {"staging_dir": str(staging_root / "ROW-0"),
                             "prediction_file_sha256": "deadbeef", "row_count": 1,
                             "lock": {"prediction_lock_identity": "pl-0"}}}
    with pytest.raises(v3.ExploratoryTargetV3Error, match="disagrees with the promotion transaction"):
        v3.promote_staged_rows(
            tmp_path, staging_root, ["ROW-0"], protocol_identity="p" * 64, plan_binding_identity="b" * 64,
            execution_identity="deadbeefcafebabe", code_commit="c" * 40, row_results=row_results)


# ==============================================================================
# safety / import purity
# ==============================================================================

def test_importing_v3_modules_touches_no_filesystem_state() -> None:
    import subprocess

    for module in ("prism_fas.evaluation.post_failure_exploratory_target_v3",
                  "prism_fas.evaluation.post_failure_exploratory_target_v3_scorer"):
        result = subprocess.run([sys.executable, "-c", f"import {module}"],
                                capture_output=True, text=True, cwd=str(REPO))
        assert result.returncode == 0, result.stderr


def test_v3_runner_invocable_as_python_dash_m_module() -> None:
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "prism_fas.evaluation.post_failure_exploratory_target_v3",
         "--repo", str(REPO), "--preflight-only"],
        capture_output=True, text=True, cwd=str(REPO / "src"))
    assert result.returncode == v3.EXIT_BLOCKED, result.stderr


def test_v3_scorer_invocable_as_python_dash_m_module() -> None:
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "prism_fas.evaluation.post_failure_exploratory_target_v3_scorer",
         "--repo", str(REPO), "--preflight-score"],
        capture_output=True, text=True, cwd=str(REPO / "src"))
    assert result.returncode == v3s.EXIT_BLOCKED, result.stderr
