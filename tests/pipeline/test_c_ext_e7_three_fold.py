"""Tests for `prism_fas.evaluation.c_ext_e7_three_fold` (E7 unified
three-fold Track-G extension preparation). Every test builds a
self-contained fake repo under `tmp_path`. No test ever renders, trains,
touches target or calls an LLM -- the module itself cannot: every build
function is pure/read-only over frozen policy JSON, and no CLI path this
module exposes accepts a render/train/target/LLM authorization flag.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism_fas.evaluation import c_ext_e7_three_fold as e7


def _base_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


def _e0_fixture(repo: Path) -> None:
    e0 = repo / "reports/c_ext_q1q2_v1/e0"
    e0.mkdir(parents=True, exist_ok=True)
    (e0 / "EXT_DATASET_FOLD_PLAN.json").write_text(json.dumps({
        "folds": {
            "EXT-F1": {"source": ["CASIA-FASD", "MSU-MFSD"], "target": "SiW-Mv2"},
            "EXT-F2": {"source": ["CASIA-FASD", "SiW-Mv2"], "target": "MSU-MFSD"},
            "EXT-F3": {"source": ["MSU-MFSD", "SiW-Mv2"], "target": "CASIA-FASD"},
        },
        "source_split_policy": {
            "siw_as_source": {"disjointness": "no video/frame from one subject/group in both train and dev",
                             "rule": "deterministic subject/group-disjoint split, 80/20"},
        },
    }), encoding="utf-8")
    (e0 / "EXT_MODEL_BINDING.json").write_text(json.dumps({
        "calibration": {"split": "source_dev only", "uses_target": False},
    }), encoding="utf-8")
    (e0 / "EXT_HYPOTHESIS_FAMILY.json").write_text(json.dumps({
        "e8_trigger": {"frozen_at": "E0", "rule": "|SMD(q)| >= 0.25 between LLM and RND OR LLM and DET"},
    }), encoding="utf-8")
    (e0 / "EXT_SEED_REGISTRY.json").write_text(json.dumps({
        "detector_seeds": [20260806, 20260807, 20260808, 20260809, 20260810],
    }), encoding="utf-8")


def _e6v2_closure_fixture(repo: Path, *, status: str = "CLOSED_TRUE_FROZEN_MATCHED_BANK_INFEASIBILITY") -> None:
    out = repo / e7.E6_V2_CLOSURE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"E6_V2_STATUS": status}), encoding="utf-8")


def _e5_realonly_fixture(repo: Path, *, gpu_real_run_executed: bool = False) -> None:
    out = repo / "reports/c_ext_q1q2_v1/e5_realonly/E5_REAL_ONLY_LOCK.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"fold": "EXT-F1", "gpu_real_run_executed": gpu_real_run_executed}),
                   encoding="utf-8")


def _e6_llm_training_plan_fixture(repo: Path) -> None:
    out = repo / "reports/c_ext_q1q2_v1/e6_llm_shuffle/E6_TRAINING_PLAN_LOCK.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"milestone": "E6"}), encoding="utf-8")


def _full_fixture(tmp_path: Path) -> Path:
    repo = _base_repo(tmp_path)
    _e0_fixture(repo)
    _e6v2_closure_fixture(repo)
    _e5_realonly_fixture(repo)
    _e6_llm_training_plan_fixture(repo)
    return repo


# --- 1: governing E7 exists --------------------------------------------------

def test_governing_e7_exists(tmp_path):
    repo = _full_fixture(tmp_path)
    interpretation = e7.build_protocol_interpretation(repo)
    assert interpretation["E7_GOVERNING_SPEC_DEFINED"] is True
    assert interpretation["governing_document_name"] == "EXT-Q1Q2 Detailed Spec v1.0"


# --- 2-4: exact matrix dimensions -------------------------------------------

def test_three_folds_exactly_correct():
    assert e7.FOLDS == ("EXT-F1", "EXT-F2", "EXT-F3")


def test_five_conditions_exactly_correct():
    assert e7.CONDITIONS == ("G-REALONLY", "G-RND", "G-DET", "G-LLM", "G-LLM-SHUFFLE-A")


def test_five_detector_seeds_exactly_correct():
    assert e7.SEEDS == (20260806, 20260807, 20260808, 20260809, 20260810)


# --- 5: nominal matrix = 75 --------------------------------------------------

def test_nominal_matrix_is_75(tmp_path):
    repo = _full_fixture(tmp_path)
    matrix = e7.build_cell_matrix(repo)
    assert matrix["total_cells"] == 75
    assert len(matrix["cells"]) == 75
    interpretation = e7.build_protocol_interpretation(repo)
    assert interpretation["E7_NOMINAL_TRAINING_COUNT"] == 75


# --- 6-7: EXT-F1 Shuffle blocked, scientific reason -------------------------

def test_ext_f1_shuffle_five_trainings_blocked(tmp_path):
    repo = _full_fixture(tmp_path)
    matrix = e7.build_cell_matrix(repo)
    f1_shuffle = [c for c in matrix["cells"] if c["fold"] == "EXT-F1" and c["condition"] == "G-LLM-SHUFFLE-A"]
    assert len(f1_shuffle) == 5
    assert all(c["readiness_status"] == e7.BLOCKED_SCIENTIFIC_INFEASIBILITY for c in f1_shuffle)


def test_block_reason_is_scientific_infeasibility(tmp_path):
    repo = _full_fixture(tmp_path)
    matrix = e7.build_cell_matrix(repo)
    cell = next(c for c in matrix["cells"] if c["fold"] == "EXT-F1" and c["condition"] == "G-LLM-SHUFFLE-A")
    assert cell["block_reason"] is not None
    assert "479" in cell["block_reason"] or "CLOSED_TRUE_FROZEN_MATCHED_BANK_INFEASIBILITY" in cell["block_reason"] \
        or "quota" in cell["block_reason"].lower()
    assert "TECHNICAL_ARTIFACT" not in cell["block_reason"]
    assert "not a technical defect" in cell["block_reason"]


# --- 8-9: no quota/threshold relaxation -------------------------------------

def test_no_quota_relaxation(tmp_path):
    repo = _full_fixture(tmp_path)
    plan = e7.build_shuffle_f2_f3_preflight_plan()
    assert "lower bank size 512" in plan["forbidden_rescue_actions"]
    assert "change source-domain quotas" in plan["forbidden_rescue_actions"]
    assert plan["no_parameter_changes_permitted"] is True
    assert plan["required_bank_size_per_route"] == {"GPAT": 512, "Physics": 512}


def test_no_threshold_relaxation(tmp_path):
    plan = e7.build_shuffle_f2_f3_preflight_plan()
    assert "relax quality thresholds" in plan["forbidden_rescue_actions"]
    assert "modify q" in plan["forbidden_rescue_actions"]
    source = Path(e7.__file__).read_text(encoding="utf-8")
    for forbidden in ("tau_fd =", "tau_id =", "SUPPORT_OVERLAP_MIN ="):
        assert forbidden not in source


# --- 10-11: F2/F3 Shuffle not pre-labelled infeasible; need feasibility gate

def test_f2_f3_shuffle_not_pre_labelled_infeasible(tmp_path):
    repo = _full_fixture(tmp_path)
    matrix = e7.build_cell_matrix(repo)
    for fold in ("EXT-F2", "EXT-F3"):
        cells = [c for c in matrix["cells"] if c["fold"] == fold and c["condition"] == "G-LLM-SHUFFLE-A"]
        assert len(cells) == 5
        assert all(c["readiness_status"] == e7.PENDING_FEASIBILITY_PREFLIGHT for c in cells)
        assert all(c["readiness_status"] != e7.BLOCKED_SCIENTIFIC_INFEASIBILITY for c in cells)


def test_f2_f3_shuffle_require_feasibility_gate(tmp_path):
    plan = e7.build_shuffle_f2_f3_preflight_plan()
    assert plan["applies_to"] == ["EXT-F2/G-LLM-SHUFFLE-A", "EXT-F3/G-LLM-SHUFFLE-A"]
    assert plan["stages_before_detector_training"] == [
        "recipe_binding", "source_pair_plan", "fold_specific_rendering",
        "frozen_quality_evaluation", "frozen_source_domain_matched_bank_feasibility"]
    assert plan["ext_f1_result_does_not_predetermine_f2_f3"] is True
    assert plan["executed_this_turn"] is False


# --- 12-15: fold isolation ---------------------------------------------------

def test_fold_target_excluded_from_gpat_support(tmp_path):
    repo = _full_fixture(tmp_path)
    lock = e7.build_fold_isolation_lock(repo)
    for fold in lock["folds"]:
        assert fold["checks"]["target_absent_from_gpat_support_fitting"] is True


def test_target_excluded_from_quality_calibration(tmp_path):
    repo = _full_fixture(tmp_path)
    lock = e7.build_fold_isolation_lock(repo)
    for fold in lock["folds"]:
        assert fold["checks"]["target_absent_from_quality_calibration"] is True


def test_target_excluded_from_detector_train(tmp_path):
    repo = _full_fixture(tmp_path)
    lock = e7.build_fold_isolation_lock(repo)
    for fold in lock["folds"]:
        assert fold["checks"]["target_absent_from_detector_training"] is True


def test_target_excluded_from_source_dev_calibration(tmp_path):
    repo = _full_fixture(tmp_path)
    lock = e7.build_fold_isolation_lock(repo)
    for fold in lock["folds"]:
        assert fold["checks"]["target_absent_from_source_dev_threshold_calibration"] is True
    assert lock["FOLD_ISOLATION_PASS"] is True


# --- 16-17: reuse vs rebuild -------------------------------------------------

def test_abstract_recipes_reusable(tmp_path):
    repo = _full_fixture(tmp_path)
    table = e7.build_reuse_rebuild_table(repo)
    row = next(r for r in table["rows"] if r["artifact"].startswith("abstract frozen recipe banks"))
    assert row["reuse_allowed"] is True
    assert row["fold_specific"] is False


def test_rendered_candidates_not_reusable_across_folds(tmp_path):
    repo = _full_fixture(tmp_path)
    table = e7.build_reuse_rebuild_table(repo)
    row = next(r for r in table["rows"] if "Physics/GPAT synthetic candidates" in r["artifact"])
    assert row["reuse_allowed"] is False
    assert row["fold_specific"] is True
    assert "NEVER reused for EXT-F2/EXT-F3" in table["rule"]


# --- 18: historical Flow1 not accepted as replacement ------------------------

def test_historical_flow1_not_accepted_as_e7_replacement(tmp_path):
    repo = _full_fixture(tmp_path)
    audit = e7.audit_nonshuffle_readiness(repo)
    assert audit["historical_flow1_accepted_as_e7_substitute"] is False
    # RND/DET have no EXT-scoped preparation -- historical Flow-1 reuse for
    # E4 threshold-transfer must NOT be silently treated as an E7 completion
    assert audit["findings"]["EXT-F1"]["G-RND"]["status"] == "NOT_PREPARED"
    assert audit["findings"]["EXT-F1"]["G-DET"]["status"] == "NOT_PREPARED"


# --- 19-20: no target, no LLM ------------------------------------------------

def test_target_access_false_everywhere(tmp_path):
    repo = _full_fixture(tmp_path)
    for builder in (lambda: e7.build_protocol_interpretation(repo), lambda: e7.build_cell_matrix(repo),
                   lambda: e7.build_fold_isolation_lock(repo), lambda: e7.build_reuse_rebuild_table(repo),
                   lambda: e7.audit_nonshuffle_readiness(repo),
                   e7.build_shuffle_f2_f3_preflight_plan, e7.build_run_count_accounting,
                   e7.build_execution_phase_plan, e7.build_metrics_output_contract,
                   lambda: e7.build_e8_trigger_record(repo)):
        body = builder()
        assert body["target_access"] is False


def test_llm_api_calls_zero_everywhere(tmp_path):
    repo = _full_fixture(tmp_path)
    for builder in (lambda: e7.build_protocol_interpretation(repo), lambda: e7.build_cell_matrix(repo),
                   lambda: e7.build_fold_isolation_lock(repo), lambda: e7.build_reuse_rebuild_table(repo),
                   lambda: e7.audit_nonshuffle_readiness(repo),
                   e7.build_shuffle_f2_f3_preflight_plan, e7.build_run_count_accounting,
                   e7.build_execution_phase_plan, e7.build_metrics_output_contract,
                   lambda: e7.build_e8_trigger_record(repo)):
        body = builder()
        assert body["llm_api_calls"] == 0


# --- 21: no GPU execution ----------------------------------------------------

def test_no_gpu_execution_this_turn(tmp_path):
    repo = _full_fixture(tmp_path)
    plan = e7.build_execution_phase_plan()
    assert plan["executed_this_turn"] == []
    readiness = e7.build_e7_readiness(repo)
    assert readiness["rendering_performed"] is False
    assert readiness["training_performed"] is False
    assert readiness["status"] == "PREPARED_NOT_EXECUTED"
    source = Path(e7.__file__).read_text(encoding="utf-8")
    for forbidden in ("add_argument(\"--authorize-gpu-training\"", "add_argument(\"--authorize-gpu-render\"",
                     "add_argument(\"--execute\"", "render_arm(", "train_detector(", "M9TrainingRun("):
        assert forbidden not in source


def test_main_has_no_execute_flag():
    """The module's own CLI surface has exactly one flag (--prepare) and no
    render/train/execute authorization path at all."""
    import argparse

    source = Path(e7.__file__).read_text(encoding="utf-8")
    assert source.count("add_argument") == 1
    assert '"--prepare"' in source


# --- 22: protected artifacts unchanged --------------------------------------

def test_prepare_e7_never_writes_outside_e7_dir(tmp_path):
    repo = _full_fixture(tmp_path)
    (repo / "reports/full/c6").mkdir(parents=True, exist_ok=True)
    sentinel = repo / "reports/full/c6/SENTINEL.json"
    sentinel.write_text('{"untouched": true}', encoding="utf-8")
    before = sentinel.read_bytes()
    before_tree = sorted(str(p.relative_to(repo)) for p in repo.rglob("*") if p.is_file())

    e7.prepare_e7(repo)

    assert sentinel.read_bytes() == before
    after_tree = sorted(str(p.relative_to(repo)) for p in repo.rglob("*") if p.is_file())
    new_files = sorted(set(after_tree) - set(before_tree))
    assert new_files, "prepare_e7 should have written something"
    assert all(f.startswith(e7.E7_DIR) for f in new_files)


def test_prepare_e7_writes_all_expected_artifacts(tmp_path):
    repo = _full_fixture(tmp_path)
    result = e7.prepare_e7(repo)
    expected = {"E7_PROTOCOL_INTERPRETATION.json", "E7_CELL_MATRIX.json", "E7_FOLD_ISOLATION_LOCK.json",
               "E7_REUSE_REBUILD_TABLE.json", "E7_NONSHUFFLE_READINESS_AUDIT.json",
               "E7_SHUFFLE_F2_F3_PREFLIGHT_PLAN.json", "E7_RUN_COUNT_ACCOUNTING.json",
               "E7_EXECUTION_PLAN.json", "E7_METRICS_OUTPUT_CONTRACT.json", "E7_E8_TRIGGER_RECORD.json",
               "E7_READINESS.json"}
    written = {Path(v["path"]).name for v in result.values()}
    assert written == expected
    for name in expected:
        assert (repo / e7.E7_DIR / name).is_file()


# --- extra: run-count accounting, E8 trigger, isolation-lock identity ------

def test_run_count_accounting_matches_final_report_fields():
    accounting = e7.build_run_count_accounting()
    assert accounting["E7_NOMINAL_TRAININGS"] == 75
    assert accounting["E7_PREDECLARED_BLOCKED_TRAININGS"] == 5
    assert accounting["E7_CURRENT_MAX_AUTHORIZABLE_BEFORE_F2_F3_SHUFFLE_PREFLIGHT"] == 70
    assert accounting["potentially_authorizable_vs_ready_now"]["ready_now"] == 0


def test_e8_trigger_record_excludes_shuffle(tmp_path):
    repo = _full_fixture(tmp_path)
    record = e7.build_e8_trigger_record(repo)
    assert record["E8_TRIGGER_FROM_E2"] is True
    assert record["E8_CONDITIONS"] == ["G-RND-QMATCH", "G-DET-QMATCH", "G-LLM-QMATCH"]
    assert record["shuffle_excluded_from_e8"] is True
    assert not any("SHUFFLE" in c for c in record["E8_CONDITIONS"])
    assert record["E8_EXECUTED"] is False


def test_fold_isolation_lock_has_deterministic_identity(tmp_path):
    repo = _full_fixture(tmp_path)
    first = e7.build_fold_isolation_lock(repo)
    second = e7.build_fold_isolation_lock(repo)
    assert first["lock_identity"] == second["lock_identity"]
    assert len(first["lock_identity"]) == 64
