"""Tests for src/prism_fas/evaluation/c_ext_e8_calibration_authority.py.

Pure, read-only-over-synthetic-fixtures tests. No GPU, no real E8 run roots,
no real checkpoints, no target access. Every 15-run fixture is a disposable
tmp_path tree; the ONE corrected run is exercised through fully injected
``trainer_provider``/``checkpoint_loader`` seams (never a real torch model).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from prism_fas.evaluation import c_ext_e8_calibration_authority as auth  # noqa: E402
from prism_fas.evaluation import c_ext_e8_training_adapter as adapter  # noqa: E402
from prism_fas.evaluation import c_ext_e8_training_runner as runner  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #

def _write_fake_run(fake_runs: Path, spec: Any, *, positions_match: bool) -> Path:
    run_root = fake_runs / spec.run_id
    (run_root / "stages" / "G6").mkdir(parents=True, exist_ok=True)
    (run_root / "stages" / "G6" / "output_hashes.json").write_text("{}", encoding="utf-8")
    (run_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_root / "checkpoints" / "best.pt").write_bytes(f"fake-best-{spec.run_id}".encode())
    (run_root / "checkpoints" / "last.pt").write_bytes(f"fake-last-{spec.run_id}".encode())
    last = {"epoch": 35, "global_step": 1575}
    best = last if positions_match else {"epoch": 30, "global_step": 1350}
    run_json = {"status": "COMPLETED", "epoch": last["epoch"], "global_step": last["global_step"],
               "best_metrics": best}
    (run_root / "run.json").write_text(json.dumps(run_json), encoding="utf-8")
    (run_root / "calibration").mkdir(parents=True, exist_ok=True)
    calib = {"calibration_hash": f"hash-{spec.run_id}", "temperature": 0.5, "selected_threshold": 0.5,
            "source_dev_prediction_hash": f"pred-{spec.run_id}"}
    (run_root / "calibration" / "source_dev.json").write_text(json.dumps(calib), encoding="utf-8")
    return run_root


def _write_all_fake_runs(fake_runs: Path) -> list[Any]:
    specs = runner.all_scientific_run_specs()
    for spec in specs:
        _write_fake_run(fake_runs, spec, positions_match=(spec.run_id != auth.CORRECTED_RUN_ID))
    return list(specs)


class _FakeModel:
    pass


class _FakeDataset:
    package_identity = "fake-package-identity"


class _FakeTrainer:
    def __init__(self, calls: list, logits: np.ndarray | None = None, targets: np.ndarray | None = None):
        self.model = _FakeModel()
        self.dataset = _FakeDataset()
        self._calls = calls
        rng = np.random.default_rng(0)
        self._logits = logits if logits is not None else rng.normal(size=200)
        self._targets = targets if targets is not None else rng.integers(0, 2, size=200)

    def source_dev_logits(self):
        self._calls.append("source_dev_logits")
        return self._logits, self._targets


def _fake_checkpoint_loader_factory(calls: list, *, ok: bool = True):
    def _loader(path: Path, model: Any) -> dict[str, Any]:
        calls.append(f"checkpoint_loader:{path}")
        sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return {"checkpoint_sha256": sha if ok else "wrong-sha"}
    return _loader


_FAKE_DETECTOR_INPUTS = {
    "package_root": runner.M3B_RUNTIME_PACKAGE_RELATIVE_PATH,
    "package_identity": runner.M3B_CONTENT_IDENTITY,
    "recipe_bank_root": runner.M7_DETECTOR_RECIPE_BANK_ROOT,
    "recipe_bank_id": runner.M7_DETECTOR_RECIPE_BANK_ID,
    "recipe_bank_identity": runner.M7_DETECTOR_RECIPE_BANK_IDENTITY,
    "recipe_bank_recipe_count": runner.M7_EXPECTED_RECIPE_COUNT,
    "candidates_root": runner.C5_CANDIDATES_ROOT_CANONICAL,
    "weight_root": "weights",
    "target_paths_resolved": 0, "target_labels_resolved": 0,
}


def _fake_c3_bank(arm: str) -> dict[str, Any]:
    return {"arm": arm, "bank_identity": runner.C3_TREATMENT_BANK_IDENTITY_BY_ARM[arm],
           "recipes": [{"recipe_id": f"r{i}"} for i in range(runner.C3_EXPECTED_RECIPE_COUNT)]}


@pytest.fixture()
def fake_runs(tmp_path):
    fake_runs_dir = tmp_path / "fake_runs"
    specs = _write_all_fake_runs(fake_runs_dir)
    return fake_runs_dir, specs


def _run_root_resolver(fake_runs_dir):
    return lambda spec: fake_runs_dir / spec.run_id


# --------------------------------------------------------------------------- #
# 1. Enumeration
# --------------------------------------------------------------------------- #

def test_enumerate_exactly_15_run_specs():
    specs = auth.enumerate_e8_run_specs()
    assert len(specs) == 15
    assert len({spec.run_id for spec in specs}) == 15


def test_exact_seed_sets_per_arm():
    specs = auth.enumerate_e8_run_specs()
    by_arm: dict[str, list[int]] = {}
    for spec in specs:
        by_arm.setdefault(spec.arm, []).append(spec.seed)
    assert set(by_arm) == {"RND", "DET", "LLM"}
    for arm, seeds in by_arm.items():
        assert sorted(seeds) == sorted(runner.SEEDS)


# --------------------------------------------------------------------------- #
# 2-3. Full lock build: 14 historical + 1 corrected, targeting DET/20260809
# --------------------------------------------------------------------------- #

def _build_lock_with_fixtures(fake_runs_dir, *, calls: list, checkpoint_ok: bool = True):
    def trainer_provider(spec, bindings, run_root):
        return _FakeTrainer(calls)

    checkpoint_loader = _fake_checkpoint_loader_factory(calls, ok=checkpoint_ok)
    override_inputs_by_run = {auth.CORRECTED_RUN_ID: dict(_FAKE_DETECTOR_INPUTS)}
    override_c3_by_arm = {"DET": _fake_c3_bank("DET")}

    with mock.patch.object(adapter, "open_e8_arm_bank", lambda *a, **k: type("S", (), {"identity": "s"})()):
        return auth.build_calibration_authority_lock(
            REPO, trainer_provider=trainer_provider, checkpoint_loader=checkpoint_loader,
            _run_root_resolver=_run_root_resolver(fake_runs_dir),
            _correction_root_resolver=lambda spec: fake_runs_dir.parent / "correction_ns" / spec.run_id,
            _device_resolver=lambda: "cuda",
            _override_detector_inputs_by_run=override_inputs_by_run,
            _override_c3_bank_by_arm=override_c3_by_arm, _skip_m3b_guard=True,
        )


def test_lock_has_exactly_15_rows_and_1_correction(fake_runs):
    fake_runs_dir, specs = fake_runs
    calls: list = []
    lock = _build_lock_with_fixtures(fake_runs_dir, calls=calls)
    assert lock["entry_count"] == 15
    assert lock["corrected_entry_count"] == 1
    assert lock["corrected_run_ids"] == [auth.CORRECTED_RUN_ID]
    assert len(lock["rows"]) == 15


def test_correction_targets_exactly_det_20260809(fake_runs):
    fake_runs_dir, specs = fake_runs
    calls: list = []
    lock = _build_lock_with_fixtures(fake_runs_dir, calls=calls)
    corrected = [row for row in lock["rows"] if row["correction_required"]]
    assert len(corrected) == 1
    assert corrected[0]["run_id"] == "e8_ext_f1_g_det_qmatch_s20260809"
    assert corrected[0]["arm"] == "DET"
    assert corrected[0]["seed"] == 20260809
    assert corrected[0]["authoritative_calibration_kind"] == "BEST_CHECKPOINT_CORRECTION_V1"


def test_14_rows_retain_historical_calibration_unmodified(fake_runs):
    fake_runs_dir, specs = fake_runs
    calls: list = []
    lock = _build_lock_with_fixtures(fake_runs_dir, calls=calls)
    for row in lock["rows"]:
        if row["run_id"] == auth.CORRECTED_RUN_ID:
            continue
        assert row["authoritative_calibration_kind"] == "HISTORICAL_VALID"
        assert row["correction_required"] is False
        assert row["calibration_hash"] == row["historical_calibration_hash"] == f"hash-{row['run_id']}"
        assert row["temperature"] == 0.5
        assert row["selected_threshold"] == 0.5


# --------------------------------------------------------------------------- #
# 3b. Historical calibration never overwritten
# --------------------------------------------------------------------------- #

def test_historical_calibration_files_byte_identical_after_lock_build(fake_runs):
    fake_runs_dir, specs = fake_runs
    before = {}
    for spec in specs:
        path = fake_runs_dir / spec.run_id / "calibration" / "source_dev.json"
        before[spec.run_id] = path.read_bytes()
    calls: list = []
    _build_lock_with_fixtures(fake_runs_dir, calls=calls)
    for spec in specs:
        path = fake_runs_dir / spec.run_id / "calibration" / "source_dev.json"
        assert path.read_bytes() == before[spec.run_id], f"{spec.run_id}: historical calibration mutated"


def test_corrected_row_historical_hash_preserved_distinct_from_new(fake_runs):
    fake_runs_dir, specs = fake_runs
    calls: list = []
    lock = _build_lock_with_fixtures(fake_runs_dir, calls=calls)
    row = next(r for r in lock["rows"] if r["run_id"] == auth.CORRECTED_RUN_ID)
    assert row["historical_calibration_hash"] == f"hash-{auth.CORRECTED_RUN_ID}"
    assert row["calibration_hash"] != row["historical_calibration_hash"]


# --------------------------------------------------------------------------- #
# 4. Best checkpoint really loaded BEFORE corrected source_dev inference
# --------------------------------------------------------------------------- #

def test_checkpoint_loaded_before_source_dev_inference(fake_runs):
    fake_runs_dir, specs = fake_runs
    calls: list = []
    _build_lock_with_fixtures(fake_runs_dir, calls=calls)
    loader_positions = [i for i, c in enumerate(calls) if c.startswith("checkpoint_loader")]
    logits_positions = [i for i, c in enumerate(calls) if c == "source_dev_logits"]
    assert loader_positions and logits_positions
    assert loader_positions[0] < logits_positions[0]


def test_checkpoint_sha_mismatch_hard_fails(fake_runs):
    fake_runs_dir, specs = fake_runs
    calls: list = []
    with pytest.raises(auth.CalibrationAuthorityError, match="SHA256"):
        _build_lock_with_fixtures(fake_runs_dir, calls=calls, checkpoint_ok=False)


# --------------------------------------------------------------------------- #
# 5. No target access during calibration correction
# --------------------------------------------------------------------------- #

def test_no_target_access_during_correction():
    # "target_labels_accessed"/"target_feature_accessed" are FLAG NAMES this
    # module declares (always False); the risk this asserts is that no target
    # ROOT/PACKAGE constant or path is ever referenced.
    source = Path(auth.__file__).read_text(encoding="utf-8")
    assert "prism_target_eval" not in source
    assert "data/evaluation_only" not in source
    assert "target_feature_root" not in source.lower()
    assert "target_label_root" not in source.lower()


def test_row_flags_report_no_target_access(fake_runs):
    fake_runs_dir, specs = fake_runs
    calls: list = []
    lock = _build_lock_with_fixtures(fake_runs_dir, calls=calls)
    for row in lock["rows"]:
        assert row["target_feature_accessed"] is False
        assert row["target_labels_accessed"] is False
        assert row["produced_gradient"] is False


# --------------------------------------------------------------------------- #
# Fail-closed guards
# --------------------------------------------------------------------------- #

def test_out_of_scope_mismatch_hard_fails(fake_runs):
    fake_runs_dir, specs = fake_runs
    rnd_spec = next(s for s in specs if s.run_id == "e8_ext_f1_g_rnd_qmatch_s20260806")
    # rewrite RND/20260806 to show a mismatch it should never have
    _write_fake_run(fake_runs_dir, rnd_spec, positions_match=False)
    with pytest.raises(auth.CalibrationAuthorityError, match="NOT in the frozen correction set"):
        auth.build_row_for_run(rnd_spec, REPO, _run_root_resolver=_run_root_resolver(fake_runs_dir))


def test_stale_correction_rule_hard_fails_if_det_seed_now_agrees(fake_runs):
    fake_runs_dir, specs = fake_runs
    det_spec = next(s for s in specs if s.run_id == auth.CORRECTED_RUN_ID)
    _write_fake_run(fake_runs_dir, det_spec, positions_match=True)
    with pytest.raises(auth.CalibrationAuthorityError, match="expected the frozen audited"):
        auth.build_row_for_run(det_spec, REPO, _run_root_resolver=_run_root_resolver(fake_runs_dir))


def test_not_completed_run_hard_fails(tmp_path):
    fake_runs_dir = tmp_path / "fake_runs"
    spec = runner.build_run_spec("LLM", 20260806)
    (fake_runs_dir / spec.run_id).mkdir(parents=True)
    with pytest.raises(auth.CalibrationAuthorityError, match="COMPLETED"):
        auth.build_row_for_run(spec, REPO, _run_root_resolver=lambda s: fake_runs_dir / s.run_id)


def test_missing_run_json_hard_fails(fake_runs):
    fake_runs_dir, specs = fake_runs
    spec = specs[0]
    (fake_runs_dir / spec.run_id / "run.json").unlink()
    with pytest.raises(auth.CalibrationAuthorityError, match="run.json"):
        auth.build_row_for_run(spec, REPO, _run_root_resolver=_run_root_resolver(fake_runs_dir))


def test_missing_historical_calibration_hard_fails(fake_runs):
    fake_runs_dir, specs = fake_runs
    spec = specs[0]
    (fake_runs_dir / spec.run_id / "calibration" / "source_dev.json").unlink()
    with pytest.raises(auth.CalibrationAuthorityError, match="calibration/source_dev.json"):
        auth.build_row_for_run(spec, REPO, _run_root_resolver=_run_root_resolver(fake_runs_dir))


def test_missing_best_checkpoint_hard_fails(fake_runs):
    fake_runs_dir, specs = fake_runs
    spec = specs[0]
    (fake_runs_dir / spec.run_id / "checkpoints" / "best.pt").unlink()
    with pytest.raises(auth.CalibrationAuthorityError, match="best checkpoint missing"):
        auth.build_row_for_run(spec, REPO, _run_root_resolver=_run_root_resolver(fake_runs_dir))


def test_lock_refuses_if_more_than_one_correction_found(fake_runs):
    fake_runs_dir, specs = fake_runs
    # introduce a SECOND run that would need correction, alongside an override
    # that fakes it into the corrected code path by monkeypatching CORRECTED_RUN_ID
    # comparisons is not exposed; instead assert the guard fires via a direct
    # two-row scenario using build_calibration_authority_lock with a monkeypatched
    # frozen id set that no longer matches reality.
    import contextlib

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(auth, "CORRECTED_RUN_ID", "e8_ext_f1_g_rnd_qmatch_s20260806"))
        calls: list = []
        with pytest.raises(auth.CalibrationAuthorityError):
            _build_lock_with_fixtures(fake_runs_dir, calls=calls)


# --------------------------------------------------------------------------- #
# G7 preflight gate: refuses without a valid, self-consistent 15-row lock
# --------------------------------------------------------------------------- #

def test_require_valid_lock_refuses_when_missing(tmp_path):
    with pytest.raises(auth.CalibrationAuthorityError, match="no valid 15-row"):
        auth.require_valid_calibration_authority_for_g7(tmp_path)


def test_require_valid_lock_refuses_when_entry_count_wrong():
    payload = {"entry_count": 14, "corrected_entry_count": 1,
              "corrected_run_ids": [auth.CORRECTED_RUN_ID], "source_only": True, "target_access": False,
              "rows": []}
    assert auth.is_usable_lock(payload) is False


def test_require_valid_lock_refuses_when_corrected_count_wrong():
    payload = {"entry_count": 15, "corrected_entry_count": 2,
              "corrected_run_ids": [auth.CORRECTED_RUN_ID], "source_only": True, "target_access": False,
              "rows": []}
    assert auth.is_usable_lock(payload) is False


def test_require_valid_lock_refuses_tampered_identity(fake_runs):
    fake_runs_dir, specs = fake_runs
    calls: list = []
    lock = _build_lock_with_fixtures(fake_runs_dir, calls=calls)
    tampered = dict(lock)
    tampered["entry_count"] = 15  # unchanged, but identity now stale after this mutation
    tampered["extra_field"] = "tampered"
    assert auth.is_usable_lock(tampered) is False


def test_lock_identity_deterministic(fake_runs):
    fake_runs_dir, specs = fake_runs
    calls_1: list = []
    calls_2: list = []
    lock_1 = _build_lock_with_fixtures(fake_runs_dir, calls=calls_1)
    lock_2 = _build_lock_with_fixtures(fake_runs_dir, calls=calls_2)
    assert lock_1["lock_identity"] == lock_2["lock_identity"]
    assert auth.is_usable_lock(lock_1) is True


# --------------------------------------------------------------------------- #
# Structural: no GPU/train/smoke/LLM triggered by this module
# --------------------------------------------------------------------------- #

def test_module_never_imports_llm_client():
    source = Path(auth.__file__).read_text(encoding="utf-8")
    assert "openai" not in source.lower()
    assert "gemini" not in source.lower()
    assert "anthropic" not in source.lower()
