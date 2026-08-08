"""M10 firewall, G7 prediction schema, PREDICTION_LOCK, aggregation, metrics and G8.

No test here opens a real SiW-Mv2 label. The label artifact in these tests is a
synthetic fixture written into `tmp_path`, which is exactly what
`docs/M10_TARGET_EVALUATION_CONTRACT.md` requires while
`target_labels_revealed: false`.
"""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from prism_fas.evaluation import metrics as m10_metrics
from prism_fas.evaluation import scoring
from prism_fas.evaluation import target_prediction as g7
from prism_fas.evaluation import video_aggregation as agg
from prism_fas.evaluation.contracts import (M10ContractError, PredictionLockError, ScoringRefusal,
                                            TargetLabelFirewallViolation, is_not_applicable)
from prism_fas.evaluation.firewall import FirewallConfig, TargetLabelFirewall, load_firewall_config

EVALUATION_CONFIG = Path("configs/evaluation/m10_target.yaml")


@pytest.fixture()
def firewall(tmp_path) -> TargetLabelFirewall:
    payload = yaml.safe_load(EVALUATION_CONFIG.read_text(encoding="utf-8"))
    payload["roots"]["source_package_root"] = str(tmp_path / "source")
    payload["roots"]["target_feature_root"] = str(tmp_path / "features")
    payload["roots"]["target_label_root"] = str(tmp_path / "labels")
    payload["roots"]["prediction_root"] = str(tmp_path / "runs")
    for name in ("source", "features", "labels", "runs"): (tmp_path / name).mkdir()
    path = tmp_path / "evaluation.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return TargetLabelFirewall(load_firewall_config(path), project_root=tmp_path)


# --- firewall ---------------------------------------------------------------
def test_train_cannot_resolve_target_features_or_labels(firewall, tmp_path):
    with pytest.raises(TargetLabelFirewallViolation, match="target_label_root"):
        firewall.check_read("TRAIN", tmp_path / "labels" / "siw_target_labels.parquet")
    with pytest.raises(TargetLabelFirewallViolation, match="target_feature_root"):
        firewall.check_read("TRAIN", tmp_path / "features" / "manifests")


def test_g7_reads_features_and_cannot_resolve_labels(firewall, tmp_path):
    assert firewall.check_read("G7", tmp_path / "features" / "images").exists() is False
    with pytest.raises(TargetLabelFirewallViolation, match="target_label_root"):
        firewall.check_read("G7", tmp_path / "labels" / "siw_target_labels.parquet")
    evidence = firewall.assert_cannot_resolve_labels("G7")
    assert evidence["label_root_permission"] == "deny" and evidence["target_labels_opened"] is False


def test_g8_reads_labels_but_cannot_write_model_state(firewall, tmp_path):
    firewall.check_read("G8", tmp_path / "labels" / "siw_target_labels.parquet")
    for name in ("best.pt", "last.pth", "model.safetensors"):
        with pytest.raises(TargetLabelFirewallViolation, match="model state"):
            firewall.check_write("G8", tmp_path / "runs" / name)
    with pytest.raises(TargetLabelFirewallViolation, match="model state"):
        firewall.check_write("G8", tmp_path / "runs" / "calibration" / "source_dev.json")


def test_a_relative_or_dotdot_path_cannot_slip_past_the_firewall(firewall, tmp_path):
    sneaky = tmp_path / "runs" / ".." / "labels" / "siw_target_labels.parquet"
    with pytest.raises(TargetLabelFirewallViolation):
        firewall.check_read("G7", sneaky)


def test_isolation_declarations_do_not_false_positive(firewall):
    """The M8/M9 lesson: the proof of isolation must not read as a leak."""
    payload = yaml.safe_load(EVALUATION_CONFIG.read_text(encoding="utf-8"))
    assert payload["target_labels_revealed"] is False
    assert "Replay" in payload["isolation"]["attack_taxonomy_forbidden_outside_g8"]
    assert firewall.assert_no_attack_taxonomy(payload, where="m10_target.yaml")["violations"] == 0
    assert firewall.assert_no_target_paths(payload, where="m10_target.yaml")["violations"] == 0


def test_an_attack_family_outside_a_declaration_is_a_violation(firewall):
    with pytest.raises(TargetLabelFirewallViolation, match="attack taxonomy"):
        firewall.assert_no_attack_taxonomy({"model": {"train_on": "Replay"}}, where="fixture")


def test_a_firewall_config_that_lets_train_read_labels_is_refused(firewall):
    broken = copy.deepcopy(firewall.config.permissions)
    broken["TRAIN"]["target_label_root"] = "read"
    with pytest.raises(M10ContractError, match="TRAIN may never read target labels"):
        FirewallConfig(roots=firewall.config.roots, permissions=broken).validate()


# --- prediction schema ------------------------------------------------------
def _row(index: int, *, p_global: float, s_region=None, p_prompt=None, threshold=0.5,
         variant="B08") -> dict:
    return g7.build_prediction_row(
        sample_id=f"sample_{index:03d}", video_id=f"siw_{index:016x}", frame_id=index,
        p_global=p_global, s_region=s_region, p_prompt=p_prompt, threshold=threshold,
        unknown_threshold=None,
        top_region_ids=[0, 1] if s_region is not None else [],
        region_distances=[0.1] * 9 if s_region is not None else [],
        checkpoint_hash="c" * 64, calibration_hash="d" * 64, inference_config_hash="e" * 64,
        variant=variant)


def test_full_variant_rows_satisfy_the_table_57_schema():
    rows = [_row(index, p_global=0.1 * index, s_region=0.2, p_prompt=0.1) for index in range(1, 6)]
    summary = g7.validate_predictions(rows)
    assert summary["rows"] == 5 and summary["labels_present"] is False
    assert summary["region_status"] == ["computed"] and summary["prompt_status"] == ["computed"]


def test_a_baseline_without_regions_writes_null_not_zero():
    row = _row(1, p_global=0.7)
    assert row["s_region"] is None and row["p_prompt"] is None
    assert row["region_status"] == "not_applicable" and row["prompt_status"] == "not_applicable"
    assert row["top_region_ids"] == [] and row["region_distances"] == []
    # The Table 34 fusion with the absent factors omitted collapses to p_global.
    assert row["s_final"] == pytest.approx(0.7)
    g7.validate_predictions([row])


def test_a_zero_masquerading_as_an_absent_component_is_refused():
    row = _row(1, p_global=0.7)
    row["s_region"] = 0.0                       # still declared not_applicable
    with pytest.raises(M10ContractError, match="an absent term is null, never 0.0"):
        g7.validate_predictions([row])


def test_the_fusion_matches_table_34_when_every_factor_exists():
    row = _row(1, p_global=0.2, s_region=0.5, p_prompt=0.25)
    assert row["s_final"] == pytest.approx(1 - (1 - 0.2) * (1 - 0.5) * (1 - 0.25))


def test_a_prediction_row_may_never_carry_a_label():
    row = _row(1, p_global=0.7)
    for column in ("label", "true_label", "attack_family", "subject_id"):
        polluted = {**row, column: "spoof"}
        with pytest.raises(M10ContractError, match="forbidden columns"):
            g7.validate_predictions([polluted])


def test_non_finite_and_out_of_range_scores_are_refused():
    row = _row(1, p_global=0.7)
    for value in (math.nan, math.inf, 1.5, -0.2):
        with pytest.raises(M10ContractError):
            g7.validate_predictions([{**row, "s_final": value}])


def test_duplicate_sample_ids_are_refused():
    row = _row(1, p_global=0.7)
    with pytest.raises(M10ContractError, match="duplicate sample_id"):
        g7.validate_predictions([row, dict(row)])


def test_a_target_inference_batch_cannot_carry_an_attack_mask():
    with pytest.raises(M10ContractError, match="attack region mask"):
        g7.TargetInferenceBatch(image=None, region_priors=None, visibility=None, is_synthetic=None,
                                sample_ids=(), video_ids=(), frame_ids=(), attack_region_mask=[1])


def test_a_structurally_zero_prompt_term_is_written_as_null_not_zero():
    """Measured in the G7 smoke, not assumed.

    `PromptHead.applicability` is `is_synthetic AND attacked region AND visible`. A
    target sample is never synthetic and carries no attack mask, so no region is
    applicable and the head returns an exact structural zero. G7 must read the
    model's own applicability mask and write null, never record that constant as a
    measurement.
    """
    import torch

    class _Output:
        def __init__(self, batch: int) -> None:
            self.p_global = torch.full((batch,), 0.4)
            self.global_logit = torch.zeros((batch, 1))
            self.s_region = torch.full((batch,), 0.5)
            self.p_prompt_spoof = torch.zeros(batch)          # the structural zero
            self.region_valid = torch.ones((batch, 9), dtype=torch.bool)
            self.aux = {"normalized_distances": torch.full((batch, 9), 0.3),
                        "prompt_applicable": torch.zeros((batch, 9))}

    class _Model:
        def eval(self): return self
        def __call__(self, batch): return _Output(len(batch.sample_ids))

    batch = g7.TargetInferenceBatch(
        image=torch.zeros((2, 3, 224, 224)), region_priors=torch.zeros((2, 9, 56, 56)),
        visibility=torch.ones((2, 9)), is_synthetic=torch.zeros(2, dtype=torch.bool),
        sample_ids=("a", "b"), video_ids=("v", "v"), frame_ids=(0, 1))
    rows = g7.predict_target(_Model(), [batch],
                             capabilities=g7.VariantCapabilities(has_region=True, has_prompt=True),
                             threshold=0.5, unknown_threshold=None, temperature=None,
                             checkpoint_hash="c" * 64, calibration_hash="d" * 64,
                             inference_config_hash="e" * 64, variant="B08")
    assert all(row["p_prompt"] is None for row in rows)
    assert all(row["prompt_status"] == "not_applicable" for row in rows)
    assert all(row["region_status"] == "computed" for row in rows)
    # The fusion therefore reduces to the two factors that exist.
    assert rows[0]["s_final"] == pytest.approx(1 - (1 - 0.4) * (1 - 0.5))


def test_variant_capabilities_read_the_flags():
    assert g7.VariantCapabilities.from_flags(
        {"region": "on", "manifold": "multi_prototype", "prompt": "frozen_prompt"}) == \
        g7.VariantCapabilities(has_region=True, has_prompt=True)
    assert g7.VariantCapabilities.from_flags(
        {"region": "off", "manifold": "off", "prompt": "off"}) == \
        g7.VariantCapabilities(has_region=False, has_prompt=False)


# --- PREDICTION_LOCK --------------------------------------------------------
@pytest.fixture()
def locked() -> tuple[list[dict], dict]:
    rows = [_row(index, p_global=0.05 + 0.09 * index, s_region=0.2, p_prompt=0.1)
            for index in range(1, 7)]
    lock = g7.build_prediction_lock(
        experiment_id="B08-s20260806", variant="B08", seed=20260806, rows=rows,
        checkpoint_sha256="c" * 64, source_calibration_sha256="f" * 64, calibration_hash="d" * 64,
        inference_config_hash="e" * 64, target_feature_package_identity="p" * 64,
        target_package_id="prism_target_v2", threshold=0.5, unknown_threshold=None)
    return rows, lock


def test_prediction_lock_binds_the_frozen_artifacts(locked):
    rows, lock = locked
    assert lock["row_count"] == 6 and lock["video_count"] == 6
    assert lock["aggregation"]["trim"] == 0.10 and lock["aggregation"]["video_confidence"] == "median"
    assert lock["target_labels_opened"] is False and lock["status"] == "LOCKED"
    assert g7.validate_prediction_lock(lock, rows)["passed"]


def test_lock_identity_is_logical_and_survives_a_row_reorder(locked):
    rows, lock = locked
    assert g7.prediction_logical_identity(list(reversed(rows))) == lock["prediction_logical_identity"]


@pytest.mark.parametrize("field,value", [
    ("expected_checkpoint_sha256", "0" * 64),
    ("expected_calibration_sha256", "0" * 64),
    ("expected_calibration_hash", "0" * 64),
    ("expected_inference_config_hash", "0" * 64),
    ("expected_package_identity", "0" * 64)])
def test_a_frozen_artifact_mismatch_is_refused(locked, field, value):
    rows, lock = locked
    with pytest.raises(PredictionLockError, match="mismatch"):
        g7.validate_prediction_lock(lock, rows, **{field: value})


def test_an_unlocked_prediction_is_refused(locked):
    rows, lock = locked
    unlocked = {**lock, "status": "DRAFT"}
    with pytest.raises(PredictionLockError, match="unlocked"):
        g7.validate_prediction_lock(unlocked, rows)


def test_a_tampered_lock_does_not_hash_to_its_own_identity(locked):
    rows, lock = locked
    with pytest.raises(PredictionLockError, match="own identity"):
        g7.validate_prediction_lock({**lock, "row_count": 99}, rows)


def test_a_changed_prediction_file_no_longer_reproduces_the_locked_identity(locked):
    rows, lock = locked
    edited = [dict(row) for row in rows]
    edited[0]["s_final"] = 0.999
    with pytest.raises(PredictionLockError, match="locked logical identity"):
        g7.validate_prediction_lock(lock, edited)


def test_predictions_round_trip_through_parquet_without_a_label(tmp_path, locked):
    rows, lock = locked
    path = g7.write_predictions(tmp_path / "target_predictions.parquet", rows, variant="B08")
    reloaded = g7.read_predictions(path)
    assert g7.prediction_logical_identity(reloaded) == lock["prediction_logical_identity"]
    columns = set(pq.read_table(path).column_names)
    assert not (columns & g7.FORBIDDEN_PREDICTION_COLUMNS)


def test_a_lockset_refuses_duplicate_experiment_ids(locked):
    _, lock = locked
    with pytest.raises(M10ContractError, match="duplicate experiment ids"):
        g7.build_lockset([lock, lock], matrix_identity="m", registry_identity="r")


# --- aggregation ------------------------------------------------------------
def test_trimmed_mean_at_four_frames_is_the_plain_mean():
    from prism_fas.train.video_aggregation import trimmed_mean
    value, trim = trimmed_mean([0.1, 0.2, 0.3, 0.4], 0.10)
    assert trim == 0 and value == pytest.approx(0.25)


def test_trimmed_mean_drops_ten_percent_from_each_end_at_twenty_frames():
    from prism_fas.train.video_aggregation import trimmed_mean
    value, trim = trimmed_mean([float(index) for index in range(20)], 0.10)
    assert trim == 2 and value == pytest.approx(9.5)      # mean of 2..17


def test_video_confidence_is_the_median_and_grouping_is_deterministic():
    rows = [{"video_id": "v1", "sample_id": f"s{index}", "s_final": score, "confidence": confidence}
            for index, (score, confidence) in enumerate([(0.9, 0.9), (0.1, 0.55), (0.5, 0.6)])]
    rows += [{"video_id": "v0", "sample_id": "s9", "s_final": 0.2, "confidence": 0.8}]
    aggregates = agg.aggregate_frames(rows, threshold=0.5)
    assert [row["video_id"] for row in aggregates] == ["v0", "v1"]
    assert aggregates[1]["video_confidence"] == pytest.approx(0.6)
    assert aggregates[1]["video_score"] == pytest.approx(0.5)
    assert aggregates[1]["decision"] == "spoof" and aggregates[0]["decision"] == "live"
    assert agg.aggregate_frames(list(reversed(rows)), threshold=0.5) == aggregates


def test_nothing_is_rejected_while_the_unknown_threshold_is_unfitted():
    assert agg.threshold_and_reject(0.9, 0.01, threshold=0.5, unknown_threshold=None) == "spoof"
    assert agg.threshold_and_reject(0.9, 0.01, threshold=0.5, unknown_threshold=0.6) == "reject"


# --- metrics: known-answer fixtures -----------------------------------------
SCORES = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05]
LABELS = [1, 1, 0, 1, 0, 0]


def test_apcer_bpcer_acer_known_values():
    result = m10_metrics.core_metrics(SCORES, LABELS, threshold=0.5)
    assert result["apcer"] == pytest.approx(1 / 3)
    assert result["bpcer"] == pytest.approx(1 / 3)
    assert result["acer"] == pytest.approx(1 / 3)
    assert result["confusion"] == {"tp_spoof": 2, "fn_spoof": 1, "fp_live": 1, "tn_live": 2,
                                   "total_spoof": 3, "total_live": 3}


def test_roc_auc_known_value():
    assert m10_metrics.core_metrics(SCORES, LABELS, threshold=0.5)["roc_auc"] == pytest.approx(8 / 9)


def test_eer_and_hter_on_a_separable_population():
    result = m10_metrics.core_metrics([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0], threshold=0.5)
    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["eer"] == pytest.approx(0.0)
    assert result["hter"] == pytest.approx(0.0)
    assert result["threshold_source"] == "source_dev_frozen"
    assert result["hter_threshold_source"] == "target_eer"


def test_brier_and_nll_known_values():
    result = m10_metrics.calibration_metrics(SCORES, LABELS)
    assert result["brier"] == pytest.approx(0.19875)
    assert result["nll"] == pytest.approx(0.5497614, abs=1e-6)


def test_ece_known_values():
    assert m10_metrics.calibration_metrics([0.5, 0.5], [1, 0])["ece"] == pytest.approx(0.0)
    assert m10_metrics.calibration_metrics([1.0, 1.0], [0, 0])["ece"] == pytest.approx(1.0)


def test_a_live_only_population_makes_apcer_not_applicable():
    """Exactly the frozen 785-live-only package. BPCER stays computable."""
    result = m10_metrics.core_metrics([0.1, 0.2, 0.6], [0, 0, 0], threshold=0.5)
    for name in ("apcer", "acer", "hter", "roc_auc", "eer"):
        assert is_not_applicable(result[name]), name
        assert "zero attack presentations" in result[name]["reason"]
    assert result["bpcer"] == pytest.approx(1 / 3)


def test_risk_coverage_needs_no_reject_threshold():
    result = m10_metrics.risk_coverage(SCORES, LABELS, [0.9, 0.8, 0.55, 0.7, 0.9, 0.95],
                                       threshold=0.5, grid=[0.5, 1.0])
    assert [point["coverage"] for point in result["points"]] == [0.5, 1.0]
    assert result["points"][-1]["risk"] == pytest.approx(2 / 6)


def test_open_set_metrics_are_declared_not_applicable():
    result = m10_metrics.open_set_metrics()
    assert is_not_applicable(result) and "fabricate a label" in result["reason"]


def test_attack_wise_apcer_is_post_hoc_and_per_family():
    result = m10_metrics.attack_wise_apcer([0.9, 0.2, 0.8, 0.1], [1, 1, 1, 0],
                                           ["Replay", "Replay", "Paper", "live"], threshold=0.5)
    assert result["used_in_tuning"] is False and result["computed"] == "post_hoc_only"
    assert result["by_family"]["Replay"]["apcer"] == pytest.approx(0.5)
    assert result["by_family"]["Paper"]["apcer"] == pytest.approx(0.0)


def test_region_wise_is_not_applicable_without_a_regional_branch():
    result = m10_metrics.region_wise_summary([_row(1, p_global=0.7)], region_order=["a", "b"])
    assert is_not_applicable(result) and "no regional branch" in result["reason"]


def test_seed_summary_marks_a_single_seed_descriptive():
    assert m10_metrics.summarize_seeds([0.2])["single_seed_descriptive"] is True
    summary = m10_metrics.summarize_seeds([0.2, 0.3, 0.4])
    assert summary["mean"] == pytest.approx(0.3) and summary["single_seed_descriptive"] is False


# --- G8 ---------------------------------------------------------------------
def _labels(tmp_path: Path, *, families: bool = True, count: int = 6) -> Path:
    rows = []
    for index in range(1, count + 1):
        spoof = index % 2 == 1
        row = {"video_id": f"siw_{index:016x}", "label": "spoof" if spoof else "live"}
        if families: row["attack_family"] = "Replay" if spoof else "live"
        rows.append(row)
    path = tmp_path / "siw_target_labels.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def test_g8_has_no_training_capability():
    report = scoring.isolation_report()
    assert report["static_import_audit"]["passed"]
    assert report["static_import_audit"]["forbidden_imports"] == []
    assert report["static_import_audit"]["import_closure"]["passed"]
    assert report["g8_can_save_checkpoint"] is False and report["g8_reads_labels"] is True


def test_g8_scores_a_locked_prediction_against_evaluation_only_labels(tmp_path, firewall, locked):
    rows, lock = locked
    labels_path = tmp_path / "labels" / "siw_target_labels.parquet"
    pq.write_table(pq.read_table(_labels(tmp_path)), labels_path)
    labels = scoring.load_evaluation_labels(labels_path, firewall=firewall, stage="G8")
    result = scoring.score(predictions=rows, lock=lock, labels=labels, threshold=0.5,
                           region_order=[f"r{index}" for index in range(9)])
    assert result["video"]["population"] == {"total": 6, "live": 3, "spoof": 3}
    assert result["wrote_checkpoint"] is False and result["modified_calibration"] is False
    assert result["target_labels_opened"] is True
    assert result["attack_wise"]["by_family"]["Replay"]["attacks"] == 3


def test_only_g8_may_open_the_label_artifact(tmp_path, firewall):
    labels_path = tmp_path / "labels" / "siw_target_labels.parquet"
    pq.write_table(pq.read_table(_labels(tmp_path)), labels_path)
    with pytest.raises(ScoringRefusal, match="may not open"):
        scoring.load_evaluation_labels(labels_path, firewall=firewall, stage="G7")


def test_g8_refuses_an_unlocked_prediction(tmp_path, firewall, locked):
    rows, lock = locked
    labels_path = tmp_path / "labels" / "siw_target_labels.parquet"
    pq.write_table(pq.read_table(_labels(tmp_path)), labels_path)
    labels = scoring.load_evaluation_labels(labels_path, firewall=firewall, stage="G8")
    with pytest.raises(PredictionLockError):
        scoring.score(predictions=rows, lock={**lock, "status": "DRAFT"}, labels=labels, threshold=0.5)


def test_g8_refuses_a_checkpoint_mismatch(tmp_path, firewall, locked):
    rows, lock = locked
    labels_path = tmp_path / "labels" / "siw_target_labels.parquet"
    pq.write_table(pq.read_table(_labels(tmp_path)), labels_path)
    labels = scoring.load_evaluation_labels(labels_path, firewall=firewall, stage="G8")
    with pytest.raises(PredictionLockError, match="mismatch"):
        scoring.score(predictions=rows, lock=lock, labels=labels, threshold=0.5,
                      expected={"checkpoint_sha256": "0" * 64})


def test_g8_refuses_a_partially_joined_population(tmp_path, firewall, locked):
    rows, lock = locked
    labels_path = tmp_path / "labels" / "siw_target_labels.parquet"
    pq.write_table(pq.read_table(_labels(tmp_path, count=3)), labels_path)
    labels = scoring.load_evaluation_labels(labels_path, firewall=firewall, stage="G8")
    with pytest.raises(ScoringRefusal, match="no label"):
        scoring.score(predictions=rows, lock=lock, labels=labels, threshold=0.5)


def test_g8_refuses_to_score_an_engineering_smoke_as_science(tmp_path, firewall, locked):
    rows, _ = locked
    lock = g7.build_prediction_lock(
        experiment_id="smoke", variant="B08", seed=None, rows=rows, checkpoint_sha256="c" * 64,
        source_calibration_sha256="f" * 64, calibration_hash="d" * 64,
        inference_config_hash="e" * 64, target_feature_package_identity="p" * 64,
        target_package_id="prism_target_v2", threshold=0.5, unknown_threshold=None,
        engineering_smoke=True)
    labels_path = tmp_path / "labels" / "siw_target_labels.parquet"
    pq.write_table(pq.read_table(_labels(tmp_path)), labels_path)
    labels = scoring.load_evaluation_labels(labels_path, firewall=firewall, stage="G8")
    assert lock["scientific_use"] is False
    with pytest.raises(ScoringRefusal, match="engineering smoke"):
        scoring.score(predictions=rows, lock=lock, labels=labels, threshold=0.5)


def test_missing_optional_attack_metadata_is_reported_not_invented(tmp_path, firewall, locked):
    rows, lock = locked
    labels_path = tmp_path / "labels" / "siw_target_labels.parquet"
    pq.write_table(pq.read_table(_labels(tmp_path, families=False)), labels_path)
    labels = scoring.load_evaluation_labels(labels_path, firewall=firewall, stage="G8")
    result = scoring.score(predictions=rows, lock=lock, labels=labels, threshold=0.5)
    assert is_not_applicable(result["attack_wise"])
    assert "optional metadata" in result["attack_wise"]["reason"]


def test_g8_cannot_write_a_checkpoint_through_its_report_writer(tmp_path, firewall, locked):
    _, lock = locked
    with pytest.raises(TargetLabelFirewallViolation, match="model state"):
        scoring.write_scoring_report(tmp_path / "runs" / "best.pt", lock, firewall=firewall)


def test_the_target_label_reveal_waits_for_a_frozen_lockset(locked):
    _, lock = locked
    lockset = g7.build_lockset([lock], matrix_identity="m", registry_identity="r")
    with pytest.raises(ScoringRefusal, match="FROZEN"):
        scoring.target_label_reveal(lockset={**lockset, "status": "DRAFT"}, authorized_by="tester")
    reveal = scoring.target_label_reveal(lockset=lockset, authorized_by="tester")
    assert reveal["target_labels_revealed"] is True and reveal["may_be_reset"] is False
