"""POST_FAILURE_EXPLORATORY_TARGET_V3 — IMPLEMENTATION RECONCILIATION.

**NO NEW PROTOCOL VERSION.** `configs/evaluation/post_failure_exploratory_target_v3.yaml`
is untouched and its protocol identity (`a2b54f8844a2a36540e62470c2f5f30de52fbf509a37f03feb7f6d769d5c702c`)
is unchanged — only the IMPLEMENTATION of the already-frozen V3 protocol is
corrected here. This file proves the eight defect corrections found by a
pre-target audit of the already-frozen implementation:

  A. E1 promotion is now a crash-recoverable transaction (`PREDICTION_PROMOTION_TRANSACTION_*.json`).
  B. Row metadata comes solely from the frozen prediction lockset — `_score`
     never calls `resolve_target_matrix`.
  C. The label reveal binds two DISTINCT commits — the E1 inference commit
     (from the lockset) and a freshly-read E2 reveal commit.
  D. `target_access_state` is explicit and correct on every real artifact
     (binding/lockset/reveal/score result) and never lies on reuse.
  E. The score-result validator re-hashes the CURRENT label artifact and
     detects tampering, without ever touching label bytes before a reveal exists.
  F. Every per-row score artifact is wrapped in a self-hashing identity
     envelope, and the validator checks exactly 24 files, no extras, and
     per-file integrity.
  G. The final score result binds complete provenance (scoring commit,
     label artifact hash, feature package identity, all 24 row identities).
  H. E2 scoring is now crash-recoverable via a disposable staging namespace
     and a `SCORE_PROMOTION_TRANSACTION_*.json` manifest.

**FIXTURE / ENGINEERING ONLY.** Every test here runs against `tmp_path`
fixtures and monkeypatched functions — never a real checkpoint, never a
real image, never target data. No test in this file accesses a target
feature, a target prediction, or a target label.
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
from prism_fas.evaluation import post_failure_exploratory_target_v3_scorer as v3s  # noqa: E402
from prism_fas.evaluation.contracts import stable_identity  # noqa: E402
from prism_fas.pipeline.state import atomic_write_json  # noqa: E402

from test_post_failure_exploratory_target_v3 import (  # noqa: E402
    FAKE_ROWS, PACKAGE_IDENTITY, _install_v3_binding_fixtures,
    _scored_rows_and_meta_for_comparisons, _write_frozen_lockset,
)

V3_PROTOCOL_PATH = REPO / "configs/evaluation/post_failure_exploratory_target_v3.yaml"
V3_IDENTITY = "a2b54f8844a2a36540e62470c2f5f30de52fbf509a37f03feb7f6d769d5c702c"
STARTING_HEAD = "c02113bb7a4296fc61860acd2ca41df06f347d31"


def _real_v3_protocol() -> dict[str, Any]:
    return yaml.safe_load(V3_PROTOCOL_PATH.read_text(encoding="utf-8"))


# ==============================================================================
# A. NO NEW PROTOCOL VERSION — the frozen V3 config is byte-identical
# ==============================================================================

def test_v3_config_untouched_since_the_starting_head() -> None:
    import subprocess

    result = subprocess.run(
        ["git", "diff", STARTING_HEAD, "--", "configs/evaluation/post_failure_exploratory_target_v3.yaml"],
        cwd=str(REPO), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_v3_protocol_identity_unchanged_after_reconciliation() -> None:
    assert v3.protocol_identity(_real_v3_protocol()) == V3_IDENTITY


# ==============================================================================
# Defect A — E1 promotion is a crash-recoverable transaction
# ==============================================================================

def _stage_row(staging_root: Path, row_id: str, *, package_identity: str, code_commit: str,
              content: bytes) -> dict[str, Any]:
    row_dir = staging_root / row_id
    row_dir.mkdir(parents=True, exist_ok=True)
    (row_dir / "target_predictions.parquet").write_bytes(content)
    lock = {"experiment_id": row_id, "variant": row_id, "seed": 1, "checkpoint_sha256": "k" * 64,
           "calibration_hash": "h" * 64, "inference_config_hash": "i" * 64,
           "prediction_logical_identity": f"l-{row_id}", "prediction_lock_identity": f"pl-{row_id}",
           "row_count": 1, "video_count": 1, "target_feature_package_identity": package_identity,
           "code_commit": code_commit}
    (row_dir / "PREDICTION_LOCK.json").write_text(json.dumps(lock), encoding="utf-8")
    prediction_file_sha256 = hashlib.sha256((row_dir / "target_predictions.parquet").read_bytes()).hexdigest()
    return {"row_id": row_id, "staging_dir": str(row_dir),
           "prediction_file_sha256": prediction_file_sha256, "row_count": 1, "lock": lock}


def test_build_promotion_transaction_binds_identities_and_hashes(tmp_path) -> None:
    staging_root = tmp_path / "staging"
    row_results = {"ROW-0": _stage_row(staging_root, "ROW-0", package_identity=PACKAGE_IDENTITY,
                                       code_commit="c" * 40, content=b"row-0")}
    transaction = v3.build_promotion_transaction(
        protocol_identity="p" * 64, plan_binding_identity="b" * 64, execution_id="deadbeefcafebabe",
        code_commit="c" * 40, row_results=row_results)
    assert transaction["state"] == "READY_TO_PROMOTE"
    assert transaction["row_ids"] == ["ROW-0"]
    assert transaction["staged_artifacts"]["ROW-0"]["prediction_file_sha256"] == \
        row_results["ROW-0"]["prediction_file_sha256"]
    recomputed = stable_identity({k: v for k, v in transaction.items()
                                 if k not in ("transaction_identity", "state")})
    assert recomputed == transaction["transaction_identity"]


def test_promote_staged_rows_writes_transaction_before_moving_anything(tmp_path) -> None:
    staging_root = tmp_path / "staging"
    row_results = {"ROW-0": _stage_row(staging_root, "ROW-0", package_identity=PACKAGE_IDENTITY,
                                       code_commit="c" * 40, content=b"row-0")}
    transaction = v3.promote_staged_rows(
        tmp_path, staging_root, ["ROW-0"], protocol_identity="p" * 64, plan_binding_identity="b" * 64,
        execution_identity="deadbeefcafebabe", code_commit="c" * 40, row_results=row_results)
    assert transaction["state"] == "COMPLETE"
    assert (tmp_path / v3.RUN_DIR / "ROW-0" / "target_predictions.parquet").is_file()
    transaction_path = v3._promotion_transaction_path(tmp_path, "deadbeefcafebabe")
    assert transaction_path.is_file()
    assert json.loads(transaction_path.read_text())["state"] == "COMPLETE"


def test_promote_staged_rows_rejects_a_row_set_mismatch_against_an_existing_transaction(tmp_path) -> None:
    staging_root = tmp_path / "staging"
    row_results = {"ROW-0": _stage_row(staging_root, "ROW-0", package_identity=PACKAGE_IDENTITY,
                                       code_commit="c" * 40, content=b"row-0")}
    v3.promote_staged_rows(tmp_path, staging_root, ["ROW-0"], protocol_identity="p" * 64,
                           plan_binding_identity="b" * 64, execution_identity="deadbeefcafebabe",
                           code_commit="c" * 40, row_results=row_results)
    with pytest.raises(v3.ExploratoryTargetV3Error, match="does not match the requested row set"):
        v3.promote_staged_rows(tmp_path, staging_root, ["ROW-0", "ROW-1"], protocol_identity="p" * 64,
                               plan_binding_identity="b" * 64, execution_identity="deadbeefcafebabe",
                               code_commit="c" * 40, row_results=row_results)


def test_promote_staged_rows_detects_a_tampered_staged_artifact(tmp_path) -> None:
    staging_root = tmp_path / "staging"
    row_results = {"ROW-0": _stage_row(staging_root, "ROW-0", package_identity=PACKAGE_IDENTITY,
                                       code_commit="c" * 40, content=b"row-0")}
    transaction = v3.build_promotion_transaction(
        protocol_identity="p" * 64, plan_binding_identity="b" * 64, execution_id="deadbeefcafebabe",
        code_commit="c" * 40, row_results=row_results)
    atomic_write_json(v3._promotion_transaction_path(tmp_path, "deadbeefcafebabe"), transaction)
    (staging_root / "ROW-0" / "target_predictions.parquet").write_bytes(b"TAMPERED")
    with pytest.raises(v3.ExploratoryTargetV3Error, match="disagrees with the promotion transaction"):
        v3.promote_staged_rows(tmp_path, staging_root, ["ROW-0"], protocol_identity="p" * 64,
                               plan_binding_identity="b" * 64, execution_identity="deadbeefcafebabe",
                               code_commit="c" * 40, row_results=row_results)


def test_promote_staged_rows_is_idempotent_reusing_the_existing_transaction(tmp_path) -> None:
    staging_root = tmp_path / "staging"
    row_results = {"ROW-0": _stage_row(staging_root, "ROW-0", package_identity=PACKAGE_IDENTITY,
                                       code_commit="c" * 40, content=b"row-0")}
    first = v3.promote_staged_rows(tmp_path, staging_root, ["ROW-0"], protocol_identity="p" * 64,
                                   plan_binding_identity="b" * 64, execution_identity="deadbeefcafebabe",
                                   code_commit="c" * 40, row_results=row_results)
    second = v3.promote_staged_rows(tmp_path, staging_root, ["ROW-0"], protocol_identity="p" * 64,
                                    plan_binding_identity="b" * 64, execution_identity="deadbeefcafebabe",
                                    code_commit="c" * 40, row_results=row_results)
    assert first["transaction_identity"] == second["transaction_identity"]
    assert second["state"] == "COMPLETE"


@pytest.mark.parametrize("crash_after", [1, 2])
def test_predict_recovers_from_crash_after_partial_promotion_with_zero_inference(
        monkeypatch, tmp_path, crash_after) -> None:
    """Defect A: a crash after `crash_after` of the (fake, small) row set
    has been PROMOTED must recover to the identical final lock on the next
    `--predict`, with zero model inference performed during recovery."""
    monkeypatch.setattr(v3, "EXPECTED_TOTAL_ROWS", len(FAKE_ROWS))
    _install_v3_binding_fixtures(monkeypatch)
    monkeypatch.setattr(v3, "current_code_commit", lambda repo: "c" * 40)

    inference_calls: list[str] = []

    def _fake_predict_one_row(repo, binding, *, package_root, firewall, staging_root, code_commit,
                              target_package_id):
        inference_calls.append(binding["row_id"])
        row_dir = Path(staging_root) / binding["row_id"]
        row_dir.mkdir(parents=True, exist_ok=True)
        (row_dir / "target_predictions.parquet").write_bytes(f"pred-{binding['row_id']}".encode())
        lock = {"experiment_id": binding["experiment_id"], "variant": binding["prediction_variant_id"],
               "seed": binding["seed"], "checkpoint_sha256": binding["checkpoint_sha256"],
               "calibration_hash": binding["calibration_hash"], "inference_config_hash": "i" * 64,
               "prediction_logical_identity": "l-" + binding["row_id"],
               "prediction_lock_identity": "pl-" + binding["row_id"], "row_count": 1, "video_count": 1,
               "target_feature_package_identity": binding["target_feature_package_identity"],
               "code_commit": code_commit}
        (row_dir / "PREDICTION_LOCK.json").write_text(json.dumps(lock), encoding="utf-8")
        prediction_file_sha256 = hashlib.sha256(
            (row_dir / "target_predictions.parquet").read_bytes()).hexdigest()
        return {"row_id": binding["row_id"], "staging_dir": str(row_dir),
               "prediction_file_sha256": prediction_file_sha256, "row_count": 1, "lock": lock}

    monkeypatch.setattr(v3, "predict_one_row_to_staging", _fake_predict_one_row)

    exit_code, _ = v3._bind_prediction_plan(tmp_path)
    assert exit_code == v3.EXIT_PASS

    real_promote = v3.promote_staged_rows

    def _crash_after_n(repo, staging_root, row_ids, **kwargs):
        transaction = v3.build_promotion_transaction(
            protocol_identity=kwargs["protocol_identity"], plan_binding_identity=kwargs["plan_binding_identity"],
            execution_id=kwargs["execution_identity"], code_commit=kwargs["code_commit"],
            row_results=kwargs["row_results"])
        atomic_write_json(v3._promotion_transaction_path(repo, kwargs["execution_identity"]), transaction)
        for row_id in sorted(row_ids)[:crash_after]:
            final_dir = Path(repo) / v3.RUN_DIR / row_id
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            (Path(staging_root) / row_id).rename(final_dir)
        raise RuntimeError(f"SIMULATED_CRASH_AFTER_{crash_after}_ROWS_PROMOTED")

    monkeypatch.setattr(v3, "promote_staged_rows", _crash_after_n)
    exit_code, payload = v3._predict(tmp_path)
    assert exit_code == v3.EXIT_BLOCKED
    assert len(inference_calls) == len(FAKE_ROWS)
    assert not (tmp_path / v3.PREDICTION_LOCK_PATH).is_file()

    monkeypatch.setattr(v3, "promote_staged_rows", real_promote)
    inference_calls.clear()
    exit_code, payload = v3._predict(tmp_path)
    assert exit_code == v3.EXIT_PASS
    assert inference_calls == []
    assert payload["recovered_from_promotion_transaction"] is True
    assert payload["model_inference_performed"] is False
    assert payload["row_count"] == len(FAKE_ROWS)

    lock = json.loads((tmp_path / v3.PREDICTION_LOCK_PATH).read_text())
    assert lock["entry_count"] == len(FAKE_ROWS)
    transaction = json.loads(v3._promotion_transaction_path(
        tmp_path, v3.execution_identity(
            plan_binding_identity=lock["prediction_plan_binding_identity"], code_commit="c" * 40)).read_text())
    assert transaction["state"] == "COMPLETE"


# ==============================================================================
# Defect B — row metadata comes solely from the frozen lockset
# ==============================================================================

def test_row_meta_from_lockset_matches_entries() -> None:
    lockset = {"entries": {"ROW-0": {"experiment_id": "C-G-RND", "track": "G", "arm": "RND", "seed": 20260806,
                                    "prediction_variant_id": "C-G-RND", "threshold": 0.5,
                                    "checkpoint_sha256": "k" * 64, "calibration_hash": "h" * 64,
                                    "prediction_lock_identity": "pl-0", "prediction_logical_identity": "pli-0"}}}
    row_meta = v3s._row_meta_from_lockset(lockset)
    assert row_meta["ROW-0"]["experiment_id"] == "C-G-RND"
    assert row_meta["ROW-0"]["prediction_lock_identity"] == "pl-0"


def test_score_source_never_calls_resolve_target_matrix() -> None:
    source = inspect.getsource(v3s._score)
    assert "resolve_target_matrix" not in source
    assert "resolve_all_row_bindings_v3" not in source


def test_score_row_meta_immune_to_a_mutated_source_matrix(monkeypatch, tmp_path) -> None:
    """Even if the source matrix resolver were mutated/corrupted after the
    lockset was frozen, E2 metadata must stay byte-identical because it is
    never consulted."""
    def _boom(repo):
        raise AssertionError("resolve_target_matrix must never be called from _score")

    monkeypatch.setattr(v1, "resolve_target_matrix", _boom)
    lockset = _write_frozen_lockset(tmp_path, entry_count=1)
    monkeypatch.setattr(v3s, "require_frozen_prediction_lockset", lambda repo: lockset)
    # No reveal/label wiring: this should fail on the (mocked-away) label
    # path, never on the poisoned resolve_target_matrix.
    exit_code, payload = v3s._score(tmp_path)
    assert exit_code == v3s.EXIT_BLOCKED
    assert "resolve_target_matrix" not in str(payload.get("error", ""))


# ==============================================================================
# Defect C — two distinct code commits bound into the label reveal
# ==============================================================================

def test_build_label_reveal_binds_two_distinct_commits() -> None:
    reveal = v3s.build_label_reveal(
        protocol_id="p" * 64, plan_binding_identity="b" * 64, prediction_lock_identity="l" * 64,
        label_relative_path="labels.parquet", label_sha256="s" * 64,
        prediction_execution_code_commit="1" * 40, first_authorized_reveal_code_commit="2" * 40)
    assert reveal["prediction_execution_code_commit"] == "1" * 40
    assert reveal["first_authorized_reveal_code_commit"] == "2" * 40
    assert reveal["prediction_execution_code_commit"] != reveal["first_authorized_reveal_code_commit"]


def test_reveal_target_labels_reads_a_fresh_commit_not_the_prediction_commit(monkeypatch, tmp_path) -> None:
    lockset = _write_frozen_lockset(tmp_path, code_commit="PREDICT_COMMIT" + "0" * 26)
    monkeypatch.setattr(v3s, "require_frozen_prediction_lockset", lambda repo: lockset)
    monkeypatch.setattr(v3s, "_current_scorer_git_commit", lambda repo: "REVEAL_COMMIT" + "0" * 27)
    monkeypatch.setattr("prism_fas.evaluation.post_failure_exploratory_target_v3.active_protocol_identity",
                        lambda repo: "p" * 64)
    label_path = tmp_path / "labels.parquet"
    label_path.write_bytes(b"fake-label-bytes")
    binding_path = tmp_path / v3s.PREDICTION_PLAN_BINDING_PATH
    binding_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(binding_path, {"prediction_plan_binding_identity": "b" * 64})

    reveal = v3s.reveal_target_labels(tmp_path, label_path=label_path)
    assert reveal["prediction_execution_code_commit"] == "PREDICT_COMMIT" + "0" * 26
    assert reveal["first_authorized_reveal_code_commit"] == "REVEAL_COMMIT" + "0" * 27


def test_current_scorer_git_commit_helper_does_not_import_detector_checkpoint() -> None:
    source = inspect.getsource(v3s._current_scorer_git_commit)
    assert "import" not in source or "subprocess" in source
    assert "from prism_fas.detector" not in source
    audit = v3s.assert_no_training_capability()
    assert "prism_fas.detector.checkpoint" not in audit["module_level_imports"]


def test_scorer_still_holds_no_training_capability_after_reconciliation() -> None:
    audit = v3s.assert_no_training_capability()
    assert audit["violations"] == []


# ==============================================================================
# Defect D — explicit, correct target_access_state on every real artifact
# ==============================================================================

def test_label_reveal_carries_full_access_state() -> None:
    reveal = v3s.build_label_reveal(
        protocol_id="p" * 64, plan_binding_identity="b" * 64, prediction_lock_identity="l" * 64,
        label_relative_path="labels.parquet", label_sha256="s" * 64,
        prediction_execution_code_commit="1" * 40, first_authorized_reveal_code_commit="2" * 40)
    assert reveal["access_state"] == {"target_feature_identity_accessed": True,
                                      "target_prediction_features_accessed": True,
                                      "target_labels_accessed": True, "target_feature_access_count": 1,
                                      "target_label_access_count": 1}


def test_access_state_counts_never_increment_on_reveal_reuse(monkeypatch, tmp_path) -> None:
    lockset = _write_frozen_lockset(tmp_path)
    monkeypatch.setattr(v3s, "require_frozen_prediction_lockset", lambda repo: lockset)
    monkeypatch.setattr("prism_fas.evaluation.post_failure_exploratory_target_v3.active_protocol_identity",
                        lambda repo: "p" * 64)
    label_path = tmp_path / "labels.parquet"
    label_path.write_bytes(b"fake-label-bytes")
    binding_path = tmp_path / v3s.PREDICTION_PLAN_BINDING_PATH
    binding_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(binding_path, {"prediction_plan_binding_identity": "b" * 64})

    first = v3s.reveal_target_labels(tmp_path, label_path=label_path)
    second = v3s.reveal_target_labels(tmp_path, label_path=label_path)
    assert first["access_state"]["target_label_access_count"] == 1
    assert second["access_state"]["target_label_access_count"] == 1


# ==============================================================================
# Defect E — label-artifact tamper detection lives in the validator only
# ==============================================================================

def test_validator_never_touches_label_bytes_before_a_reveal_exists(monkeypatch, tmp_path) -> None:
    lockset = _write_frozen_lockset(tmp_path)
    monkeypatch.setattr(v3s, "require_frozen_prediction_lockset", lambda repo: lockset)
    atomic_write_json(tmp_path / v3s.SCORE_RESULT_PATH, {"row_count": 1})
    # No TARGET_LABEL_REVEAL.json and no label file anywhere on disk.
    validation = v3s.validate_existing_exploratory_score_result_v3(tmp_path)
    assert validation["valid"] is False
    assert any("no TARGET_LABEL_REVEAL.json" in p for p in validation["problems"])


def test_validator_detects_label_artifact_tampering_after_reveal(monkeypatch, tmp_path) -> None:
    lockset = _write_frozen_lockset(tmp_path)
    monkeypatch.setattr(v3s, "require_frozen_prediction_lockset", lambda repo: lockset)
    label_path = tmp_path / "labels.parquet"
    label_path.write_bytes(b"original-bytes")
    reveal = {"target_label_artifact_relative_path": "labels.parquet",
             "target_label_artifact_sha256": v3s.compute_label_artifact_sha256(label_path),
             "reveal_identity": ""}
    reveal["reveal_identity"] = stable_identity({k: v for k, v in reveal.items() if k != "reveal_identity"})
    atomic_write_json(tmp_path / v3s.LABEL_REVEAL_PATH, reveal)
    atomic_write_json(tmp_path / v3s.SCORE_RESULT_PATH, {"row_count": 1, "label_reveal_identity":
                                                         reveal["reveal_identity"]})

    label_path.write_bytes(b"TAMPERED-bytes-after-reveal")
    validation = v3s.validate_existing_exploratory_score_result_v3(tmp_path)
    assert validation["valid"] is False
    assert any("tampering" in p for p in validation["problems"])


# ==============================================================================
# Defect F — per-row score artifacts are self-hashing identity envelopes
# ==============================================================================

def test_score_one_row_wraps_metrics_with_an_identity_envelope(monkeypatch, tmp_path) -> None:
    row_meta_entry = {"prediction_lock_identity": "pl-0", "prediction_logical_identity": "pli-0",
                      "checkpoint_sha256": "k" * 64, "calibration_hash": "h" * 64, "seed": 20260806,
                      "track": "G", "arm": "RND", "prediction_variant_id": "C-G-RND"}
    run_root = tmp_path / "run"
    row_dir = run_root / "ROW-0"
    row_dir.mkdir(parents=True)
    (row_dir / "PREDICTION_LOCK.json").write_text(json.dumps({"aggregation": {"threshold": 0.5}}),
                                                  encoding="utf-8")

    monkeypatch.setattr("prism_fas.evaluation.target_prediction.read_predictions", lambda path: [])
    monkeypatch.setattr("prism_fas.evaluation.scoring.score",
                        lambda **kwargs: {"video": {"acer": 0.1}, "video_scores": []})

    body = v3s.score_one_row(tmp_path, "ROW-0", labels={}, run_root=run_root, row_meta_entry=row_meta_entry)
    for field in ("row_id", "prediction_lock_identity", "prediction_logical_identity", "checkpoint_sha256",
                 "calibration_hash", "seed", "track", "arm", "prediction_variant_id", "metrics",
                 "score_artifact_identity"):
        assert field in body
    recomputed = stable_identity({k: v for k, v in body.items() if k != "score_artifact_identity"})
    assert recomputed == body["score_artifact_identity"]


def _minimal_score_result_and_row_file(tmp_path, lockset) -> tuple[dict[str, Any], Path]:
    row_id = next(iter(lockset["entries"]))
    entry = lockset["entries"][row_id]
    body = {"row_id": row_id, "prediction_lock_identity": entry["prediction_lock_identity"],
           "prediction_logical_identity": entry["prediction_logical_identity"],
           "checkpoint_sha256": entry["checkpoint_sha256"], "calibration_hash": entry["calibration_hash"],
           "seed": entry["seed"], "track": entry["track"], "arm": entry["arm"],
           "prediction_variant_id": entry["prediction_variant_id"], "metrics": {"video": {"acer": 0.0}}}
    body["score_artifact_identity"] = stable_identity({k: v for k, v in body.items()
                                                       if k != "score_artifact_identity"})
    rows_dir = tmp_path / v3s.SCORE_ROWS_DIR
    rows_dir.mkdir(parents=True, exist_ok=True)
    row_path = rows_dir / f"{row_id}.json"
    atomic_write_json(row_path, body)
    return body, row_path


def test_validator_detects_a_tampered_per_row_score_file(monkeypatch, tmp_path) -> None:
    lockset = _write_frozen_lockset(tmp_path, entry_count=1)
    monkeypatch.setattr(v3s, "require_frozen_prediction_lockset", lambda repo: lockset)
    body, row_path = _minimal_score_result_and_row_file(tmp_path, lockset)
    row_id = body["row_id"]
    result = {"row_count": 1, "rows": {row_id: body},
             "per_row_score_artifacts": {row_id: {"score_artifact_identity": body["score_artifact_identity"],
                                                  "score_file_sha256": hashlib.sha256(
                                                      row_path.read_bytes()).hexdigest()}}}
    atomic_write_json(tmp_path / v3s.SCORE_RESULT_PATH, result)

    row_path.write_text(row_path.read_text() + " ", encoding="utf-8")   # tamper
    validation = v3s.validate_existing_exploratory_score_result_v3(tmp_path)
    assert any("tampering" in p for p in validation["problems"])


def test_validator_detects_extra_and_missing_score_files(monkeypatch, tmp_path) -> None:
    lockset = _write_frozen_lockset(tmp_path, entry_count=1)
    monkeypatch.setattr(v3s, "require_frozen_prediction_lockset", lambda repo: lockset)
    row_id = next(iter(lockset["entries"]))
    rows_dir = tmp_path / v3s.SCORE_ROWS_DIR
    rows_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(rows_dir / "NOT-A-ROW.json", {"row_id": "NOT-A-ROW"})
    atomic_write_json(tmp_path / v3s.SCORE_RESULT_PATH, {"row_count": 1, "rows": {}})

    validation = v3s.validate_existing_exploratory_score_result_v3(tmp_path)
    assert any("unexpected per-row score files" in p for p in validation["problems"])
    assert any(f"missing per-row score files: ['{row_id}']" in p for p in validation["problems"])


# ==============================================================================
# Defect G — the final score result binds complete provenance
# ==============================================================================

def test_final_score_result_validator_requires_scoring_execution_code_commit(monkeypatch, tmp_path) -> None:
    lockset = _write_frozen_lockset(tmp_path, entry_count=0)
    lockset["entries"] = {}
    lockset["entry_count"] = 0
    monkeypatch.setattr(v3s, "require_frozen_prediction_lockset", lambda repo: lockset)
    atomic_write_json(tmp_path / v3s.SCORE_RESULT_PATH, {"row_count": 0, "rows": {},
                                                         "prediction_lock_identity": lockset["lockset_identity"]})
    validation = v3s.validate_existing_exploratory_score_result_v3(tmp_path)
    assert any("scoring_execution_code_commit" in p for p in validation["problems"])
    assert any("target_feature_package_identity" in p for p in validation["problems"])


# ==============================================================================
# Defect H — E2 scoring is crash-recoverable
# ==============================================================================

def _full_24_row_lockset(*, code_commit: str = "c" * 40, package_identity: str = "e" * 64) -> dict[str, Any]:
    _, row_meta = _scored_rows_and_meta_for_comparisons(complete=True)
    entries = {}
    for row_id, meta in row_meta.items():
        entries[row_id] = {"row_id": row_id, "experiment_id": meta["experiment_id"], "track": meta["track"],
                           "arm": meta["arm"], "seed": meta["seed"],
                           "prediction_variant_id": meta["experiment_id"], "threshold": 0.5,
                           "checkpoint_sha256": "k" * 64, "calibration_hash": "h" * 64,
                           "target_feature_package_identity": package_identity, "code_commit": code_commit,
                           "prediction_lock_identity": f"pl-{row_id}", "prediction_logical_identity": f"pli-{row_id}",
                           "prediction_file_sha256": f"pf-{row_id}", "row_count": 1, "video_count": 1}
    return {"status": "FROZEN", "entry_count": len(entries), "target_labels_opened": False,
           "lockset_identity": "l" * 64, "prediction_execution_code_commit": code_commit,
           "target_feature_package_identity": package_identity, "entries": entries}


def test_score_recovers_from_crash_after_partial_promotion_with_zero_rescoring_and_no_relabel(
        monkeypatch, tmp_path) -> None:
    fixture_scored_rows, _ = _scored_rows_and_meta_for_comparisons(complete=True)
    lockset = _full_24_row_lockset()
    monkeypatch.setattr(v3s, "require_frozen_prediction_lockset", lambda repo: lockset)
    monkeypatch.setattr(v3s, "_current_scorer_git_commit", lambda repo: "d" * 40)
    monkeypatch.setattr("prism_fas.evaluation.post_failure_exploratory_target_v3.load_protocol",
                        lambda repo: _real_v3_protocol())
    monkeypatch.setattr(v3s, "_build_firewall", lambda repo, protocol: object())

    import prism_fas.evaluation.scoring as scoring_module

    label_load_calls: list[int] = []
    monkeypatch.setattr(scoring_module, "load_evaluation_labels",
                        lambda *a, **k: label_load_calls.append(1) or {})

    reveal_calls: list[int] = []

    def _fake_reveal(repo, *, label_path):
        reveal_calls.append(1)
        body = {"reveal_schema_version": "post-failure-exploratory-target-v3-label-reveal-v2",
               "protocol_identity": "p" * 64, "prediction_plan_binding_identity": "b" * 64,
               "target_prediction_lock_identity": lockset["lockset_identity"],
               "target_label_artifact_relative_path": "fake/labels.parquet",
               "target_label_artifact_sha256": "f" * 64,
               "prediction_execution_code_commit": lockset["prediction_execution_code_commit"],
               "first_authorized_reveal_code_commit": "d" * 40, "target_labels_accessed": True,
               "one_way": True, "may_be_reset": False, "reason": "POST_FAILURE_EXPLORATORY_E2_SCORING",
               "access_state": {"target_feature_identity_accessed": True,
                                "target_prediction_features_accessed": True, "target_labels_accessed": True,
                                "target_feature_access_count": 1, "target_label_access_count": 1},
               "ba_sep_observed_verdict": "FAIL", "detector_reliability_overall": "FAILED",
               "post_failure_diagnostics_v2": "FAIL", "c9_original_confirmatory_path": "BLOCKED"}
        body["reveal_identity"] = stable_identity(body)
        atomic_write_json(Path(repo) / v3s.LABEL_REVEAL_PATH, body)
        return body

    monkeypatch.setattr(v3s, "reveal_target_labels", _fake_reveal)

    score_calls: list[str] = []

    def _fake_score_one_row(repo, row_id, *, labels, run_root, row_meta_entry):
        score_calls.append(row_id)
        body = dict(fixture_scored_rows[row_id])
        body.update({"row_id": row_id, "prediction_lock_identity": row_meta_entry["prediction_lock_identity"],
                    "prediction_logical_identity": row_meta_entry["prediction_logical_identity"],
                    "checkpoint_sha256": row_meta_entry["checkpoint_sha256"],
                    "calibration_hash": row_meta_entry["calibration_hash"],
                    "seed": int(row_meta_entry["seed"]), "track": row_meta_entry["track"],
                    "arm": row_meta_entry["arm"],
                    "prediction_variant_id": row_meta_entry["prediction_variant_id"]})
        body["score_artifact_identity"] = stable_identity(
            {k: v for k, v in body.items() if k != "score_artifact_identity"})
        return body

    monkeypatch.setattr(v3s, "score_one_row", _fake_score_one_row)

    def _crash_after_one(repo, staging_root, row_ids, **kwargs):
        staged_hashes = kwargs["staged_hashes"]
        body = {"schema_version": v3s.SCORE_PROMOTION_TRANSACTION_SCHEMA_VERSION,
               "prediction_lock_identity": kwargs["prediction_lock_identity"],
               "label_reveal_identity": kwargs["label_reveal_identity"],
               "scoring_execution_code_commit": kwargs["scoring_execution_code_commit"],
               "execution_identity": kwargs["execution_identity"], "row_ids": sorted(row_ids),
               "staged_artifacts": {rid: dict(staged_hashes[rid]) for rid in sorted(row_ids)}}
        body["transaction_identity"] = stable_identity(body)
        body["state"] = "READY_TO_PROMOTE"
        atomic_write_json(v3s._score_promotion_transaction_path(repo, kwargs["execution_identity"]), body)
        first_row_id = sorted(row_ids)[0]
        final_path = Path(repo) / v3s.SCORE_ROWS_DIR / f"{first_row_id}.json"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        (Path(staging_root) / f"{first_row_id}.json").rename(final_path)
        raise RuntimeError("SIMULATED_CRASH_AFTER_ONE_SCORE_ROW_PROMOTED")

    real_promote = v3s.promote_staged_score_rows
    monkeypatch.setattr(v3s, "promote_staged_score_rows", _crash_after_one)

    exit_code, payload = v3s._score(tmp_path)
    assert exit_code == v3s.EXIT_BLOCKED
    assert len(score_calls) == 24
    assert len(reveal_calls) == 1
    assert len(label_load_calls) == 1
    assert not (tmp_path / v3s.SCORE_RESULT_PATH).is_file()

    monkeypatch.setattr(v3s, "promote_staged_score_rows", real_promote)
    score_calls.clear()
    label_load_calls.clear()
    reveal_calls.clear()

    exit_code, payload = v3s._score(tmp_path)
    assert exit_code == v3s.EXIT_PASS
    assert score_calls == []
    assert label_load_calls == []
    assert reveal_calls == []
    assert payload["recovered_from_score_promotion_transaction"] is True
    assert payload["row_count"] == 24

    result = json.loads((tmp_path / v3s.SCORE_RESULT_PATH).read_text())
    for field in ("scoring_execution_code_commit", "target_label_artifact_sha256",
                 "target_feature_package_identity", "per_row_score_artifacts", "access_state",
                 "cross_seed_summary", "exploratory_comparisons"):
        assert field in result, field
    assert len(result["per_row_score_artifacts"]) == 24
    assert result["access_state"] == {"target_feature_identity_accessed": True,
                                      "target_prediction_features_accessed": True,
                                      "target_labels_accessed": True, "target_feature_access_count": 1,
                                      "target_label_access_count": 1}
    assert result["ba_sep_observed_verdict"] == "FAIL"
    assert result["c9_may_close"] is False
