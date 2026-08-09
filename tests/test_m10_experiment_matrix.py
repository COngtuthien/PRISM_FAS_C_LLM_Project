"""M10 matrix, replication policy and registry.

Every assertion here is about a DECLARED contract, not about an observed result:
these tests must pass before any matrix row is launched and before any target label
exists.
"""
from __future__ import annotations
import copy
from pathlib import Path
import pytest
from prism_fas.evaluation import experiment_matrix as matrix
from prism_fas.evaluation.contracts import M10ContractError
from prism_fas.evaluation.experiment_registry import ExperimentRegistry, source_matrix_lock

CONFIG = Path("configs/experiments/m10_matrix.yaml")


@pytest.fixture(scope="module")
def config() -> dict:
    return matrix.load_matrix_config(CONFIG)


@pytest.fixture(scope="module")
def plan() -> dict:
    return matrix.build_plan(CONFIG)


# --- Table 59 --------------------------------------------------------------
def test_all_nine_table_59_baselines_are_present(plan):
    families = {row["family"] for row in plan["rows"] if row["category"] == "baseline"}
    assert families == {f"B0{index}" for index in range(9)}


@pytest.mark.parametrize("baseline,expected", [
    ("B00", {"local_branch": "convnext", "global_branch": "off", "fusion": "single_logit",
             "region": "off", "manifold": "off", "synthetic": "none", "prompt": "off",
             "sampler": "domain_class_balanced"}),
    ("B01", {"local_branch": "off", "global_branch": "siglip2_frozen", "fusion": "single_logit",
             "region": "off", "manifold": "off", "synthetic": "none", "prompt": "off"}),
    ("B02", {"fusion": "simple_concat", "region": "off", "manifold": "off", "synthetic": "none",
             "prompt": "off"}),
    ("B03", {"fusion": "simple_concat", "synthetic": "physics_only", "manifold": "off",
             "region": "off"}),
    ("B04", {"fusion": "simple_concat", "synthetic": "gpat_only", "manifold": "off",
             "recipe_conditioning": "off"}),
    ("B05", {"fusion": "simple_concat", "manifold": "global_center", "prototype_k": 1,
             "region": "off", "synthetic": "none"}),
    ("B06", {"region": "on", "manifold": "global_center", "prototype_k": 1, "synthetic": "none",
             "prompt": "off"}),
    ("B07", {"region": "on", "manifold": "multi_prototype", "prototype_k": 4, "synthetic": "none",
             "prompt": "off"}),
    ("B08", {"region": "on", "manifold": "multi_prototype", "prototype_k": 4,
             "synthetic": "bank_physics_gpat", "prompt": "frozen_prompt",
             "quality_weighting": "q_weighted", "outlier_loss": "mask_aware"}),
])
def test_baseline_semantics_match_table_59(plan, baseline, expected):
    row = next(row for row in plan["rows"] if row["family"] == baseline)
    for key, value in expected.items():
        assert row["flags"][key] == value, f"{baseline}.{key}"


def test_b08_is_the_predeclared_full_method_at_k4(plan):
    rows = [row for row in plan["rows"] if row["family"] == "B08"]
    assert len(rows) == 3 and {row["flags"]["prototype_k"] for row in rows} == {4}
    assert all(row["replication_role"] == "spec_mandated" for row in rows)


def test_b08_first_seed_binds_the_m9_reference_run(plan):
    reused = [row for row in plan["rows"] if row["reuses_m9_reference"]]
    assert [row["experiment_id"] for row in reused] == ["B08-s20260806"]
    assert reused[0]["reference_run_id"] == "m9_reference_seed20260806"


# --- Table 60 --------------------------------------------------------------
def test_all_ten_table_60_ablation_families_are_present(plan):
    families = {row["family"] for row in plan["rows"] if row["category"] == "ablation"}
    assert families == {"data_balance", "recipe", "synthetic_route", "quality_weighting", "region",
                        "prototype_k", "outlier", "prompt", "backend", "frame_count"}


def test_prototype_k_ablation_covers_1_2_4_6(plan):
    variants = {row["flags"]["prototype_k"] for row in plan["rows"] if row["family"] == "prototype_k"}
    assert variants == {1, 2, 6}          # K=4 is the parent B08, never duplicated
    assert 4 in {row["flags"]["prototype_k"] for row in plan["rows"] if row["family"] == "B08"}


def test_frame_count_ablation_is_blocked_and_still_in_the_matrix(plan):
    rows = [row for row in plan["rows"] if row["family"] == "frame_count"]
    assert len(rows) == 3
    assert {row["variant"] for row in rows} == {"f16", "f32", "f48_64"}
    for row in rows:
        assert row["status"] == "BLOCKED"
        assert "frozen frame plan" in row["blocked_reason"]
        assert row["target_prediction_required"] is False
        assert row["required_stages"] == []


def test_backend_family_carries_a_bounded_parity_row_and_a_blocked_full_row(plan):
    rows = {row["variant"]: row for row in plan["rows"] if row["family"] == "backend"}
    assert rows["pc_bounded_parity"]["replication_role"] == "parity"
    assert rows["pc_bounded_parity"]["protocol"] == "bounded_step_parity"
    assert rows["pc_full_training"]["status"] == "BLOCKED"
    assert "no local CUDA device" in rows["pc_full_training"]["blocked_reason"]


# --- replication policy ----------------------------------------------------
def test_seed_list_is_exactly_the_three_canonical_seeds(config, plan):
    assert config["seeds"]["canonical"] == [20260806, 20260807, 20260808]
    used = {row["seed"] for row in plan["rows"] if row["seed"] is not None}
    assert used <= {20260806, 20260807, 20260808}


def test_b00_and_b08_carry_three_seeds(plan):
    for family in ("B00", "B08"):
        seeds = sorted(row["seed"] for row in plan["rows"] if row["family"] == family)
        assert seeds == [20260806, 20260807, 20260808], family


def test_hypothesis_critical_rows_carry_three_seeds(plan):
    critical: dict[str, list[int]] = {}
    for row in plan["rows"]:
        if row["replication_role"] == "hypothesis_critical":
            critical.setdefault(row["experiment_id"].rsplit("-s", 1)[0], []).append(row["seed"])
    assert critical, "the hypothesis-critical set must not be empty"
    for name, seeds in critical.items():
        assert sorted(seeds) == [20260806, 20260807, 20260808], name


def test_diagnostic_rows_are_single_seed_and_cannot_carry_a_claim(config, plan):
    for row in plan["rows"]:
        if row["replication_role"] == "diagnostic":
            assert row["seed"] == 20260806
    assert config["replication_roles"]["diagnostic"]["statistical_claim_allowed"] is False
    assert config["replication_roles"]["parity"]["statistical_claim_allowed"] is False


def test_declared_hypothesis_family_is_exactly_h1_to_h6(config, plan):
    assert set(config["hypotheses"]) == {"H1", "H2", "H3", "H4", "H5", "H6"}
    checked = matrix.validate_hypotheses(matrix.plan_matrix(config), config)
    assert checked["H2"]["treatment"] == "B07" and checked["H2"]["control"] == "B06"
    assert checked["H6"]["kind"] == "parity_not_superiority"


# --- determinism and identity ----------------------------------------------
def test_planning_twice_produces_an_identical_matrix_identity():
    first, second = matrix.build_plan(CONFIG), matrix.build_plan(CONFIG)
    agreement = matrix.plans_agree(first, second)
    assert agreement["identical"] and agreement["identity_matches"]
    assert agreement["mismatched_rows"] == []


def test_experiment_ids_are_unique(plan):
    identifiers = [row["experiment_id"] for row in plan["rows"]]
    assert len(identifiers) == len(set(identifiers))


def test_matrix_identity_excludes_execution_attributes(config):
    rows = matrix.plan_matrix(config)
    baseline = matrix.matrix_identity(rows)
    for row in rows:
        assert "protocol" not in row.canonical()
        assert "reference_run_id" not in row.canonical()
    assert matrix.matrix_identity(matrix.plan_matrix(config)) == baseline


def test_parity_row_shares_the_scientific_config_hash_of_its_parent(plan):
    parity = next(row for row in plan["rows"] if row["variant"] == "pc_bounded_parity")
    parent = next(row for row in plan["rows"] if row["experiment_id"] == "B08-s20260806")
    assert parity["scientific_config_hash"] == parent["scientific_config_hash"]
    assert parity["backend"] == "local" and parent["backend"] == "modal"


def test_every_row_binds_the_frozen_inputs(plan):
    for row in plan["rows"]:
        assert row["source_package_identity"].startswith("b1cf29b6")
        assert row["m8_bank_identity"].startswith("e84c78cd")
        assert row["source_selection_rule"]["uses_target"] is False


# --- contradictions are refused, not repaired -------------------------------
def test_a_contradictory_flag_set_is_refused(config):
    broken = copy.deepcopy(config)
    broken["baselines"][0]["flags"]["manifold"] = "multi_prototype"   # with prototype_k 0
    with pytest.raises(M10ContractError, match="contradictory flags"):
        matrix.plan_matrix(broken)


def test_a_blocked_row_without_a_reason_is_refused(config):
    broken = copy.deepcopy(config)
    for block in broken["ablations"]:
        if block["id"] == "A10":
            block["variants"][0].pop("blocked_reason")
    with pytest.raises(M10ContractError, match="must carry its reason"):
        matrix.plan_matrix(broken)


def test_a_seed_outside_the_canonical_list_cannot_be_requested(config):
    broken = copy.deepcopy(config)
    broken["replication_roles"]["spec_mandated"]["seeds"] = 4
    with pytest.raises(M10ContractError, match="only 3 are declared"):
        matrix.plan_matrix(broken)


# --- registry ---------------------------------------------------------------
def test_registry_materializes_every_planned_row(tmp_path, plan):
    registry = ExperimentRegistry.from_plan(tmp_path, plan)
    assert len(registry.records) == plan["summary"]["logical_rows"]
    assert len(registry.by_status("BLOCKED")) == plan["summary"]["blocked_rows"]


def test_registry_refuses_a_duplicate_launch(tmp_path, plan):
    registry = ExperimentRegistry.from_plan(tmp_path, plan)
    registry.claim("B00-s20260806", run_id="run-a", backend="modal")
    with pytest.raises(M10ContractError, match="duplicate launch"):
        registry.claim("B00-s20260806", run_id="run-b", backend="modal")


def test_registry_refuses_to_launch_a_blocked_row(tmp_path, plan):
    registry = ExperimentRegistry.from_plan(tmp_path, plan)
    with pytest.raises(M10ContractError, match="BLOCKED"):
        registry.claim("A10-frame_count-f16", run_id="run", backend="modal")


def test_a_failed_row_is_kept_with_its_failure_record(tmp_path, plan):
    registry = ExperimentRegistry.from_plan(tmp_path, plan)
    registry.claim("B01-s20260806", run_id="run", backend="modal")
    registry.fail("B01-s20260806", stage="G5", error="CUDA OOM at step 12")
    registry.write()
    reloaded = ExperimentRegistry.load(tmp_path)
    record = reloaded.get("B01-s20260806")
    assert record.status == "FAILED"
    assert record.failure["error"] == "CUDA OOM at step 12"
    assert record.failure["stage"] == "G5"
    assert "B01-s20260806" in reloaded.summary()["failed"]


def test_a_completed_row_must_name_its_selected_checkpoint(tmp_path, plan):
    registry = ExperimentRegistry.from_plan(tmp_path, plan)
    registry.claim("B02-s20260806", run_id="run", backend="modal")
    with pytest.raises(M10ContractError, match="must name its selected checkpoint"):
        registry.update("B02-s20260806", status="COMPLETED")


def test_registry_identity_ignores_run_labels_and_wall_clock(tmp_path, plan):
    first = ExperimentRegistry.from_plan(tmp_path, plan)
    second = ExperimentRegistry.from_plan(tmp_path / "other", plan)
    second.claim("B00-s20260807", run_id="a-different-run-id", backend="local")
    second.update("B00-s20260807", status="PLANNED", run_id="another", compute={"seconds": 12})
    assert first.identity() == second.identity()


def test_source_matrix_lock_refuses_a_completed_row_without_a_calibration(tmp_path, plan):
    registry = ExperimentRegistry.from_plan(tmp_path, plan)
    registry.claim("B00-s20260806", run_id="run", backend="modal")
    registry.update("B00-s20260806", status="COMPLETED", best_checkpoint_sha256="a" * 64)
    with pytest.raises(M10ContractError, match="no frozen"):
        source_matrix_lock(registry, plan, require_terminal=False)


def test_source_matrix_lock_records_that_selection_never_used_target(tmp_path, plan):
    registry = ExperimentRegistry.from_plan(tmp_path, plan)
    registry.claim("B00-s20260806", run_id="run", backend="modal")
    registry.update("B00-s20260806", status="COMPLETED", best_checkpoint_sha256="a" * 64,
                    source_calibration_sha256="b" * 64, calibration_hash="c" * 64)
    lock = source_matrix_lock(registry, plan, require_terminal=False)
    assert lock["selection_used_target"] is False and lock["target_labels_opened"] is False


def test_source_matrix_lock_refuses_to_freeze_while_a_row_is_in_flight(tmp_path, plan):
    """The real lock is taken only when every row is terminal. Freezing early would
    record a source side that can still change after the target is opened."""
    registry = ExperimentRegistry.from_plan(tmp_path, plan)
    with pytest.raises(M10ContractError, match="not terminal"):
        source_matrix_lock(registry, plan)


def test_source_matrix_lock_keeps_every_logical_row_including_failures(tmp_path, plan):
    """A failed row is kept with its failure record, a blocked row with its reason,
    and the lock covers all 42 logical rows — not just the ones that worked."""
    from prism_fas.evaluation.experiment_registry import validate_source_matrix_lock
    registry = ExperimentRegistry.from_plan(tmp_path, plan)
    for record in registry.ordered():
        if record.status == "BLOCKED": continue
        registry.claim(record.experiment_id, run_id=record.experiment_id, backend=record.backend)
        if record.experiment_id == "B01-s20260806":
            registry.fail(record.experiment_id, stage="G5", error="deliberate test failure")
        elif record.replication_role == "parity":
            # A parity row trains nothing to completion and selects no checkpoint, so
            # it is COMPLETED by naming the parity evidence it produced.
            registry.update(record.experiment_id, status="COMPLETED",
                            parity_identity="p" * 64, parity_passed=True,
                            parity_checks={"batch_identity_exact": True})
        else:
            registry.update(record.experiment_id, status="COMPLETED",
                            best_checkpoint_sha256="a" * 64, source_calibration_sha256="b" * 64,
                            calibration_hash="c" * 64, source_dev_metrics={"source_dev/acer": 0.1})
    lock = source_matrix_lock(registry, plan, code_lineage={"head": "d" * 40},
                              reference_binding={"binding_accepted": True})
    assert lock["logical_rows"] == 42
    assert lock["blocked_rows"] == 4
    assert lock["failed_rows"] == ["B01-s20260806"]
    failed = next(e for e in lock["entries"] if e["experiment_id"] == "B01-s20260806")
    assert failed["failure"]["stage"] == "G5" and failed["failure"]["error"]
    # A parity row is locked as parity evidence, never as a second training result.
    parity = next(e for e in lock["entries"] if e["replication_role"] == "parity")
    assert parity["kind"] == "parity_not_superiority" and parity["parity_identity"]
    assert "best_checkpoint_sha256" not in parity
    assert all(e.get("blocked_reason") for e in lock["entries"] if e["status"] == "BLOCKED")
    # And it validates, twice, reproducing its own identity both times.
    first = validate_source_matrix_lock(lock, plan)
    second = validate_source_matrix_lock(lock, plan)
    assert first["passed"] and second["passed"], first["checks"]
    assert first["source_matrix_lock_identity"] == second["source_matrix_lock_identity"]
