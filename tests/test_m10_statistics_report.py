"""M10 statistics (deterministic paired video bootstrap, Holm-Bonferroni) and the
report assembler.

The report tests assert the property that matters while `target_labels_revealed`
is false: a report assembled with no G8 result contains no target number anywhere.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
from prism_fas.evaluation import bootstrap, reliability, report
from prism_fas.evaluation import experiment_matrix as matrix
from prism_fas.evaluation.contracts import M10ContractError
from prism_fas.evaluation.experiment_registry import ExperimentRegistry

CONFIG = Path("configs/experiments/m10_matrix.yaml")


@pytest.fixture(scope="module")
def plan() -> dict:
    return matrix.build_plan(CONFIG)


def _population(count: int = 40) -> tuple[list[str], list[int]]:
    ids = [f"siw_{index:016x}" for index in range(count)]
    labels = [index % 2 for index in range(count)]
    return ids, labels


# --- the frozen settings ----------------------------------------------------
def test_bootstrap_settings_match_the_frozen_contract():
    settings = bootstrap.BootstrapSettings().validate()
    assert settings.unit == "video"
    assert settings.resamples == 10000 and settings.seed == 20260810
    assert settings.confidence_level == 0.95
    assert settings.statistic == "video_acer_difference"


def test_a_non_video_bootstrap_unit_is_refused():
    with pytest.raises(M10ContractError, match="frozen at 'video'"):
        bootstrap.BootstrapSettings(unit="frame").validate()


# --- determinism ------------------------------------------------------------
def test_the_resample_plan_is_deterministic():
    ids, _ = _population()
    agreement = bootstrap.plans_agree(ids, bootstrap.BootstrapSettings(resamples=200))
    assert agreement["identical"]


def test_a_different_video_set_gets_a_different_plan():
    ids, _ = _population()
    settings = bootstrap.BootstrapSettings(resamples=200)
    first = bootstrap.build_plan(ids, settings)
    second = bootstrap.build_plan(ids[:-1], settings)
    assert first["bootstrap_plan_identity"] != second["bootstrap_plan_identity"]


def test_the_plan_seed_is_derived_structurally_not_from_the_raw_integer():
    ids, _ = _population()
    settings = bootstrap.BootstrapSettings(resamples=50)
    assert bootstrap.seed_material(ids, settings) != str(settings.seed)
    assert bootstrap.seed_material(ids, settings) == bootstrap.seed_material(list(reversed(ids)), settings)


def test_duplicate_video_ids_are_refused():
    with pytest.raises(M10ContractError, match="duplicate video ids"):
        bootstrap.build_plan(["a", "a", "b"], bootstrap.BootstrapSettings(resamples=10))


# --- the paired interval ----------------------------------------------------
def test_paired_bootstrap_reproduces_exactly_and_detects_a_real_difference():
    ids, labels = _population()
    truth = np.asarray(labels)
    # `better` separates the classes; `worse` is close to chance.
    better = np.where(truth == 1, 0.9, 0.1)
    worse = np.where(truth == 1, 0.55, 0.45)
    worse[:8] = 1.0 - worse[:8]                       # eight confident mistakes
    settings = bootstrap.BootstrapSettings(resamples=300)
    first = bootstrap.paired_bootstrap(video_ids=ids, scores_a=better, scores_b=worse, labels=labels,
                                       threshold=0.5, settings=settings)
    second = bootstrap.paired_bootstrap(video_ids=ids, scores_a=better, scores_b=worse, labels=labels,
                                        threshold=0.5, settings=settings)
    assert first == second
    assert first["paired"] is True and first["n_units"] == len(ids)
    assert first["observed_delta"] < 0                  # a lower ACER is better
    assert first["ci_high"] < 0 and first["significant_at_alpha"] is True


def test_identical_models_produce_an_interval_that_contains_zero():
    ids, labels = _population()
    scores = np.where(np.asarray(labels) == 1, 0.8, 0.2)
    result = bootstrap.paired_bootstrap(video_ids=ids, scores_a=scores, scores_b=scores,
                                        labels=labels, threshold=0.5,
                                        settings=bootstrap.BootstrapSettings(resamples=200))
    assert result["observed_delta"] == pytest.approx(0.0)
    assert result["ci_low"] <= 0.0 <= result["ci_high"]
    assert result["significant_at_alpha"] is False and result["p_value"] == pytest.approx(1.0)


def test_misaligned_inputs_are_refused():
    ids, labels = _population(10)
    with pytest.raises(M10ContractError, match="must align"):
        bootstrap.paired_bootstrap(video_ids=ids, scores_a=[0.5] * 9, scores_b=[0.5] * 10,
                                   labels=labels, threshold=0.5)


# --- single-seed refusal ----------------------------------------------------
def test_a_comparison_against_a_single_seed_row_is_refused_not_downgraded():
    with pytest.raises(M10ContractError, match="single-seed row"):
        bootstrap.refuse_single_seed_comparison("H9", ["spec_mandated", "diagnostic"])
    bootstrap.refuse_single_seed_comparison("H1", ["spec_mandated", "hypothesis_critical"])


def test_the_declared_family_refuses_and_reports_a_single_seed_comparison(plan):
    ids, labels = _population(20)
    scores = np.where(np.asarray(labels) == 1, 0.8, 0.2)
    scored = {name: {"video_ids": ids, "scores": list(scores), "labels": labels}
              for name in ("B08", "A01-data_balance-naive_concat", "B06", "B07",
                           "A07-outlier-image_level", "A02-recipe-random_operators",
                           "A04-quality_weighting-hard_gate_only")}
    roles = {name: "hypothesis_critical" for name in scored}
    roles["B08"] = "spec_mandated"
    roles["A04-quality_weighting-hard_gate_only"] = "diagnostic"      # forced, to prove the refusal
    result = bootstrap.run_declared_family(hypotheses=plan["hypotheses"], scored=scored, roles=roles,
                                           threshold=0.5,
                                           settings=bootstrap.BootstrapSettings(resamples=100))
    assert result["comparisons"]["H5"]["status"] == "refused"
    assert "single-seed" in result["comparisons"]["H5"]["reason"]
    assert result["comparisons"]["H6"]["status"] == "parity_not_superiority"
    assert result["comparisons"]["H1"]["status"] == "computed"
    assert result["family_size"] == 6


def test_an_unavailable_side_is_reported_not_skipped(plan):
    result = bootstrap.run_declared_family(hypotheses=plan["hypotheses"], scored={}, roles={},
                                           threshold=0.5)
    assert all(entry["status"] in ("unavailable", "parity_not_superiority")
               for entry in result["comparisons"].values())
    assert result["multiple_comparison"]["rejected_null"] == []


# --- Holm-Bonferroni --------------------------------------------------------
def test_holm_bonferroni_known_values():
    result = bootstrap.holm_bonferroni({"H1": 0.01, "H2": 0.02, "H3": 0.03, "H4": 0.04, "H5": 0.05})
    assert result["family_size"] == 5
    assert result["adjusted"]["H1"] == pytest.approx(0.05)
    assert result["adjusted"]["H2"] == pytest.approx(0.08)
    assert result["adjusted"]["H3"] == pytest.approx(0.09)
    assert result["adjusted"]["H4"] == pytest.approx(0.09)
    assert result["adjusted"]["H5"] == pytest.approx(0.09)
    assert result["rejected_null"] == ["H1"]


def test_holm_bonferroni_is_monotone_and_never_below_the_raw_value():
    raw = {"a": 0.001, "b": 0.2, "c": 0.9}
    result = bootstrap.holm_bonferroni(raw)
    assert all(result["adjusted"][name] >= raw[name] for name in raw)


# --- reliability ------------------------------------------------------------
def test_every_declared_reliability_test_is_present():
    tests = {test.test_id for test in reliability.declared_tests()}
    assert tests == {"synthetic_vs_real_spoof_probe", "benign_jpeg_corruption",
                     "benign_resize_corruption", "benign_color_corruption", "residual_scale_zero",
                     "recipe_region_shift", "artifact_map_swap", "crop_padding_interpolation",
                     "cross_route_synthetic", "benign_glasses_makeup_lowlight"}


def test_the_benign_attribute_test_is_blocked_with_its_reason():
    blocked = [test for test in reliability.declared_tests() if test.status == "BLOCKED"]
    assert [test.test_id for test in blocked] == ["benign_glasses_makeup_lowlight"]
    assert "labelled SPOOF presentations" in blocked[0].blocked_reason


def test_a_blocked_reliability_test_cannot_be_given_a_result():
    blocked = next(test for test in reliability.declared_tests() if test.status == "BLOCKED")
    with pytest.raises(M10ContractError, match="cannot be given a result"):
        reliability.evaluate(blocked, result={"mean_shift": 0.0}, passed=True)


def test_a_failed_reliability_test_stays_in_the_report():
    tests = list(reliability.declared_tests())
    tests[1] = reliability.evaluate(tests[1], result={"mean_shift": 0.4}, passed=False)
    payload = reliability.build_report(tests)
    assert payload["failed"] == ["benign_jpeg_corruption"]
    assert payload["by_status"]["FAILED"] == 1
    assert payload["uses_target_labels"] is False


def test_score_shift_is_a_paired_measurement():
    result = reliability.score_shift([0.1, 0.2, 0.3], [0.2, 0.2, 0.1])
    assert result["mean_shift"] == pytest.approx((0.1 + 0.0 - 0.2) / 3)
    assert result["fraction_increased"] == pytest.approx(1 / 3)


# --- report -----------------------------------------------------------------
@pytest.fixture()
def registry(tmp_path, plan) -> ExperimentRegistry:
    built = ExperimentRegistry.from_plan(tmp_path, plan)
    built.claim("B01-s20260806", run_id="run-1", backend="modal")
    built.fail("B01-s20260806", stage="G5", error="CUDA OOM at step 12")
    built.claim("B00-s20260806", run_id="run-0", backend="modal")
    built.update("B00-s20260806", status="COMPLETED", best_checkpoint_sha256="a" * 64,
                 source_calibration_sha256="b" * 64, calibration_hash="c" * 64,
                 source_dev_metrics={"source_dev/acer": 0.0623})
    return built


def test_the_report_has_every_declared_section(plan, registry):
    assembled = report.assemble(plan=plan, registry=registry)
    assert list(assembled["sections"]) == list(report.REPORT_SECTIONS)


def test_an_unscored_report_contains_no_target_value(plan, registry):
    assembled = report.assemble(plan=plan, registry=registry)
    audit = report.audit_no_fabricated_target_values(assembled)
    assert audit["fabricated_target_values"] == 0 and audit["sections_verified"] == 7
    assert assembled["target_labels_revealed"] is False
    assert "SiW-Mv2 performance" in assembled["not_claimed"]


def test_a_target_score_cannot_exist_while_labels_are_unrevealed(plan, registry):
    with pytest.raises(M10ContractError, match="target_labels_revealed is false"):
        report.assemble(plan=plan, registry=registry,
                        scores={"B00-s20260806": {"video": {}, "frame": {}}},
                        target_labels_revealed=False)


def test_failed_and_blocked_rows_are_reported_not_dropped(plan, registry):
    assembled = report.assemble(plan=plan, registry=registry)
    negatives = assembled["sections"]["negative_and_blocked"]
    assert negatives["counts"]["failed"] == 1
    assert negatives["failed_experiments"][0]["experiment_id"] == "B01-s20260806"
    assert negatives["counts"]["blocked"] == plan["summary"]["blocked_rows"]
    baseline_rows = {row["experiment_id"]: row for row in assembled["sections"]["baseline_table"]["rows"]}
    assert baseline_rows["B01-s20260806"]["status"] == "FAILED"
    assert baseline_rows["B01-s20260806"]["target"]["status"] == "not_yet_scored"
    assert "CUDA OOM" in baseline_rows["B01-s20260806"]["target"]["reason"]


def test_a_blocked_ablation_row_explains_itself_in_the_report(plan, registry):
    assembled = report.assemble(plan=plan, registry=registry)
    rows = {row["experiment_id"]: row for row in assembled["sections"]["ablations"]["rows"]}
    blocked = rows["A10-frame_count-f16"]
    assert blocked["status"] == "BLOCKED"
    assert "frozen frame plan" in blocked["target"]["reason"]


def test_single_seed_rows_are_marked_descriptive(plan, registry):
    assembled = report.assemble(plan=plan, registry=registry)
    rows = {row["experiment_id"]: row for row in assembled["sections"]["ablations"]["rows"]}
    assert rows["A06-prototype_k-k1-s20260806"]["single_seed_descriptive"] is True
    assert rows["A01-data_balance-naive_concat-s20260806"]["single_seed_descriptive"] is False


def test_the_report_carries_the_source_selection_rule_and_never_a_target_selection(plan, registry):
    assembled = report.assemble(plan=plan, registry=registry)
    section = assembled["sections"]["source_selection_and_calibration"]
    assert section["selection_used_target"] is False
    assert section["rule"]["selection_metric"] == "source_dev/acer"
    assert section["selected"][0]["experiment_id"] == "B00-s20260806"


def test_the_markdown_rendering_prints_absences(plan, registry):
    assembled = report.assemble(plan=plan, registry=registry)
    text = report.render_markdown(assembled)
    assert "target_labels_revealed: **False**" in text
    assert "not_yet_scored" in text
    for name in report.REPORT_SECTIONS:
        assert name.replace("_", " ") in text
