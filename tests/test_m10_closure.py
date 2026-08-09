"""M10 closure: G8 isolation, the source-side binding of a prediction, the lockset,
the pre-reveal audit, seed aggregation and the disclosures.

No test here opens a real SiW-Mv2 label. Every artifact is a fixture in `tmp_path`,
which is what `docs/M10_TARGET_EVALUATION_CONTRACT.md` requires while
`target_labels_revealed: false`.
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path
import pytest
from prism_fas.evaluation import bootstrap, closure, disclosures, report, scoring
from prism_fas.evaluation import target_prediction as g7
from prism_fas.evaluation.contracts import M10ContractError, PredictionLockError


def _row(index: int, *, p_global: float, s_region=None) -> dict:
    return g7.build_prediction_row(
        sample_id=f"sample_{index:03d}", video_id=f"siw_{index:016x}", frame_id=index,
        p_global=p_global, s_region=s_region, p_prompt=None, threshold=0.5, unknown_threshold=None,
        top_region_ids=[0, 1] if s_region is not None else [],
        region_distances=[0.1] * 9 if s_region is not None else [],
        checkpoint_hash="c" * 64, calibration_hash="d" * 64, inference_config_hash="e" * 64,
        variant="B08")


@pytest.fixture()
def rows() -> list[dict]:
    return [_row(index, p_global=0.05 + 0.09 * index, s_region=0.2) for index in range(1, 7)]


def _lock(rows, name: str, *, source_identity: str = "m" * 64, seed: int = 20260806) -> dict:
    return g7.build_prediction_lock(
        experiment_id=name, variant=name.rsplit("-s", 1)[0], seed=seed, rows=rows,
        checkpoint_sha256="c" * 64, source_calibration_sha256="f" * 64, calibration_hash="d" * 64,
        inference_config_hash="e" * 64, target_feature_package_identity="p" * 64,
        target_package_id="prism_target_v2", threshold=0.5, unknown_threshold=None,
        scientific_config_hash="s" * 64, source_matrix_lock_identity=source_identity)


# --- G8 really holds no training runtime ------------------------------------
def test_the_mirrored_region_order_equals_the_detector_s():
    """G8 names regions without importing the detector package, which imports torch.

    The mirror is only safe while the two agree, so a drift is a failing test rather
    than a silently relabelled region histogram.
    """
    from prism_fas.detector.contracts import REGION_ORDER as DETECTOR_ORDER
    from prism_fas.evaluation.contracts import REGION_ORDER as EVALUATION_ORDER
    assert tuple(EVALUATION_ORDER) == tuple(DETECTOR_ORDER)


def test_importing_g8_pulls_in_no_training_runtime():
    """`g8_imports_torch: false` must be true of a FRESH interpreter.

    Asserting it in this process would prove nothing: the test session has already
    imported torch for other milestones.
    """
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; import prism_fas.evaluation.scoring; import prism_fas.evaluation.closure; "
         "print('torch' in sys.modules)"],
        capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False", result.stdout


def test_the_import_audit_resolves_relative_imports_and_walks_transitively():
    audit = scoring.import_closure_audit()
    assert audit["passed"] and not audit["failures"]
    # A relative `from .metrics import ...` must resolve, or whole subtrees would be
    # dropped and the audit would pass vacuously.
    assert "prism_fas.evaluation.metrics" in audit["modules_audited"]
    # And the walk must leave the evaluation package: this is the edge that used to
    # drag torch in through `prism_fas.train.__init__`.
    assert "prism_fas.train.metrics" in audit["modules_audited"]
    assert audit["module_count"] >= 10


def test_a_relative_import_is_not_mistaken_for_a_third_party_module(tmp_path):
    module = tmp_path / "probe.py"
    module.write_text("from .metrics import roc_auc\nfrom ..train import x\n", encoding="utf-8")
    audit = scoring.static_import_audit(module, dotted="prism_fas.evaluation.probe")
    assert "prism_fas.evaluation.metrics" in audit["module_level_imports"]
    assert "prism_fas.train" in audit["module_level_imports"]


# --- the prediction binds the frozen source-side decision --------------------
def test_prediction_lock_binds_the_source_side_decision(rows):
    lock = _lock(rows, "B08-s20260806")
    assert lock["scientific_config_hash"] == "s" * 64
    assert lock["source_matrix_lock_identity"] == "m" * 64
    assert g7.validate_prediction_lock(
        lock, rows, expected_scientific_config_hash="s" * 64,
        expected_source_matrix_lock_identity="m" * 64)["passed"]
    with pytest.raises(PredictionLockError, match="scientific_config_hash"):
        g7.validate_prediction_lock(lock, rows, expected_scientific_config_hash="0" * 64)
    with pytest.raises(PredictionLockError, match="source_matrix_lock_identity"):
        g7.validate_prediction_lock(lock, rows, expected_source_matrix_lock_identity="0" * 64)


def test_lockset_refuses_a_mixed_or_smoked_set(rows):
    good = _lock(rows, "B08-s20260806")
    other = _lock(rows, "B00-s20260806", source_identity="0" * 64)
    with pytest.raises(M10ContractError, match="same SOURCE_MATRIX_LOCK identity"):
        g7.build_lockset([good, other], matrix_identity="m", registry_identity="r",
                         source_matrix_lock_identity="m" * 64)
    smoke = {**_lock(rows, "B00-s20260806"), "engineering_smoke": True}
    with pytest.raises(M10ContractError, match="engineering-smoke"):
        g7.build_lockset([good, smoke], matrix_identity="m", registry_identity="r")


def test_lockset_records_blocked_rows_and_rows_that_produce_no_prediction(rows):
    lockset = g7.build_lockset(
        [_lock(rows, "B08-s20260806")], matrix_identity="m", registry_identity="r",
        source_matrix_lock_identity="m" * 64, target_feature_package_identity="p" * 64,
        row_statuses={"B08-s20260806": "COMPLETED", "A10-frame_count-f16": "BLOCKED"},
        blocked_rows=[{"experiment_id": "A10-frame_count-f16", "blocked_reason": "declared data"}],
        rows_without_prediction=[{"experiment_id": "A09-backend-pc_bounded_parity-s20260806",
                                  "reason": "a parity result selects no checkpoint"}])
    assert lockset["status"] == "FROZEN" and lockset["frame_rows_total"] == len(rows)
    assert lockset["blocked_rows"][0]["reason"] == "declared data"
    assert lockset["rows_without_prediction"][0]["experiment_id"].startswith("A09-")


def _source_lock() -> dict:
    return {"source_matrix_lock_identity": "m" * 64, "m10_matrix_identity": "x" * 64,
            "logical_rows": 3, "executable_rows": 2, "blocked_rows": 1, "failed_rows": [],
            "target_labels_opened": False, "selection_used_target": False,
            "entries": [
                {"experiment_id": "B08-s20260806", "status": "COMPLETED",
                 "target_prediction_required": True, "best_checkpoint_sha256": "c" * 64,
                 "source_calibration_sha256": "f" * 64, "calibration_hash": "d" * 64,
                 "scientific_config_hash": "s" * 64},
                {"experiment_id": "A09-backend-pc_bounded_parity-s20260806", "status": "COMPLETED",
                 "target_prediction_required": False},
                {"experiment_id": "A10-frame_count-f16", "status": "BLOCKED",
                 "blocked_reason": "declared data not available at the frozen frame plan"}]}


def test_eligible_rows_and_absent_predictions_are_both_explicit():
    source = _source_lock()
    assert [row["experiment_id"] for row in closure.eligible_rows(source)] == ["B08-s20260806"]
    absent = {row["experiment_id"]: row for row in closure.rows_without_prediction(source)}
    assert set(absent) == {"A09-backend-pc_bounded_parity-s20260806", "A10-frame_count-f16"}
    # An absent prediction must say WHY; it is never merely missing.
    assert all(row["reason"] for row in absent.values())
    assert "parity" in absent["A09-backend-pc_bounded_parity-s20260806"]["reason"]


# --- the hard stop ----------------------------------------------------------
def test_pre_reveal_audit_fails_when_a_row_is_missing(tmp_path, rows):
    lockset = g7.build_lockset([_lock(rows, "B08-s20260806")], matrix_identity="x" * 64,
                               registry_identity="r", source_matrix_lock_identity="m" * 64,
                               target_feature_package_identity="p" * 64)
    audit = closure.pre_reveal_audit(
        lockset=lockset, source_matrix_lock=_source_lock(),
        matrix_plan={"m10_matrix_identity": "x" * 64},
        prediction_root=tmp_path / "absent", reports_root=tmp_path,
        package_identity="p" * 64, expected_rows=6, expected_videos=6,
        evaluation_config={"target_labels_revealed": False})
    assert audit["passed"] is False
    assert audit["checks"]["every_eligible_row_is_locked"] is False
    assert audit["missing_rows"] == ["B08-s20260806"]


def test_pre_reveal_audit_refuses_once_a_reveal_exists(tmp_path, rows):
    (tmp_path / closure.REVEAL_FILE).write_text("{}", encoding="utf-8")
    audit = closure.pre_reveal_audit(
        lockset=g7.build_lockset([_lock(rows, "B08-s20260806")], matrix_identity="x" * 64,
                                 registry_identity="r", source_matrix_lock_identity="m" * 64),
        source_matrix_lock=_source_lock(), matrix_plan={"m10_matrix_identity": "x" * 64},
        prediction_root=tmp_path, reports_root=tmp_path, package_identity="p" * 64,
        expected_rows=6, expected_videos=6, evaluation_config={"target_labels_revealed": False})
    assert audit["checks"]["labels_not_yet_revealed"] is False


def test_pre_reveal_audit_passes_on_a_complete_frozen_set(tmp_path, rows):
    row_root = tmp_path / "g7" / "B08-s20260806"
    row_root.mkdir(parents=True)
    g7.write_predictions(row_root / g7.PREDICTION_FILE, rows, variant="B08")
    lock = _lock(rows, "B08-s20260806")
    g7.write_prediction_lock(row_root / g7.PREDICTION_LOCK_FILE, lock)
    lockset = g7.build_lockset([lock], matrix_identity="x" * 64, registry_identity="r",
                               source_matrix_lock_identity="m" * 64,
                               target_feature_package_identity="p" * 64)
    audit = closure.pre_reveal_audit(
        lockset=lockset, source_matrix_lock=_source_lock(),
        matrix_plan={"m10_matrix_identity": "x" * 64}, prediction_root=tmp_path / "g7",
        reports_root=tmp_path, package_identity="p" * 64, expected_rows=6, expected_videos=6,
        evaluation_config={"target_labels_revealed": False})
    assert audit["passed"] is True, audit["checks"]
    assert audit["per_row"][0]["checks"]["frame_rows_complete"] is True
    assert audit["target_labels_opened"] is False


def test_a_prediction_with_the_wrong_checkpoint_fails_row_validation(tmp_path, rows):
    row_root = tmp_path / "g7" / "B08-s20260806"
    row_root.mkdir(parents=True)
    g7.write_predictions(row_root / g7.PREDICTION_FILE, rows, variant="B08")
    entry = dict(_source_lock()["entries"][0], best_checkpoint_sha256="0" * 64)
    with pytest.raises(PredictionLockError, match="checkpoint_sha256"):
        closure.validate_row_prediction(
            entry=entry, prediction_path=row_root / g7.PREDICTION_FILE,
            lock=_lock(rows, "B08-s20260806"), package_identity="p" * 64,
            source_matrix_lock_identity="m" * 64, expected_rows=6, expected_videos=6)


# --- seeds and the statistics input -----------------------------------------
def _scored(name: str, seed: int, offset: float) -> dict:
    return {"experiment_id": name, "variant": name.rsplit("-s", 1)[0], "seed": seed,
            "threshold": 0.5 + offset / 10.0,
            "video": {"acer": 0.1 + offset, "apcer": 0.1, "bpcer": 0.1, "hter": 0.1,
                      "roc_auc": 0.9, "eer": 0.1,
                      "calibration": {"ece": 0.01, "brier": 0.02, "nll": 0.3}},
            "frame": {"acer": 0.2 + offset, "apcer": 0.2, "bpcer": 0.2, "hter": 0.2,
                      "roc_auc": 0.8, "eer": 0.2},
            "video_scores": [{"video_id": "siw_a", "video_score": 0.2 + offset, "label": 0},
                             {"video_id": "siw_b", "video_score": 0.8 + offset, "label": 1}]}


def test_seed_summary_reports_spread_and_refuses_to_invent_it():
    scorings = {f"B08-s2026080{index}": _scored(f"B08-s2026080{index}", 20260806 + index, 0.01 * index)
                for index in range(3)}
    scorings["B01-s20260806"] = _scored("B01-s20260806", 20260806, 0.05)
    summary = closure.seed_summary(scorings, roles={"B08": "spec_mandated", "B01": "diagnostic"})
    assert summary["B08"]["n_seeds"] == 3 and summary["B08"]["may_carry_statistical_claim"] is True
    assert summary["B08"]["video"]["acer"]["std"] > 0.0
    single = summary["B01"]
    assert single["single_seed_descriptive"] is True
    assert single["may_carry_statistical_claim"] is False
    # A one-seed row still reports its value; it just carries no claim and no spread.
    assert single["video"]["acer"]["std"] == 0.0 and single["video"]["acer"]["n_seeds"] == 1


def test_statistics_input_averages_over_seeds_and_carries_the_row_threshold():
    scorings = {f"B08-s2026080{index}": _scored(f"B08-s2026080{index}", 20260806 + index, 0.01 * index)
                for index in range(3)}
    block = closure.statistics_input(
        scorings, thresholds={name: float(payload["threshold"]) for name, payload in scorings.items()})
    assert block["B08"]["video_ids"] == ["siw_a", "siw_b"] and block["B08"]["n_seeds"] == 3
    assert block["B08"]["scores"][0] == pytest.approx(0.2 + (0.0 + 0.01 + 0.02) / 3)
    assert block["B08"]["threshold"] == pytest.approx(0.5 + (0.0 + 0.001 + 0.002) / 3)
    assert len(block["B08"]["per_seed_thresholds"]) == 3


# --- each side at its own frozen threshold ----------------------------------
def test_the_paired_bootstrap_scores_each_side_at_its_own_frozen_threshold():
    """Two models never share a `source_dev` threshold, so one threshold for both
    would report neither at the operating point it would actually deploy."""
    videos = [f"siw_{index:016x}" for index in range(12)]
    labels = [index % 2 for index in range(12)]
    treatment = [0.9 if label else 0.1 for label in labels]
    control = [0.6 if label else 0.4 for label in labels]
    settings = bootstrap.BootstrapSettings(resamples=200)
    shared = bootstrap.paired_bootstrap(video_ids=videos, scores_a=treatment, scores_b=control,
                                        labels=labels, threshold=0.5, settings=settings)
    split = bootstrap.paired_bootstrap(video_ids=videos, scores_a=treatment, scores_b=control,
                                       labels=labels, threshold=0.5, threshold_a=0.5,
                                       threshold_b=0.95, settings=settings)
    assert shared["threshold_treatment"] == 0.5 and shared["threshold_control"] == 0.5
    assert split["threshold_control"] == 0.95
    # At 0.95 the control accepts every attack as live, so the difference moves.
    assert split["observed_delta"] != shared["observed_delta"]
    # The resample plan depends on the video set, not on the thresholds.
    assert split["bootstrap_plan_identity"] == shared["bootstrap_plan_identity"]


# --- disclosures ------------------------------------------------------------
def test_synthetic_exposure_is_derived_from_the_audited_batch_contract():
    """Section 28: for a `synthetic: none` row the report must print 0, not the
    stale 871 the run artifact displays."""
    reconciliation = disclosures.synthetic_exposure_reconciliation(Path("reports/m10"))
    assert reconciliation["training_affected"] is False
    per_row = reconciliation["per_row"]
    assert per_row, "the implementability audit should carry every executable row"
    for name, block in per_row.items():
        if block["declared_synthetic_flag"] == "none":
            assert block["synthetic_per_batch"] == 0, name
            assert block["synthetic_loss_terms_active"] == [], name
    affected = reconciliation["blast_radius"]["rows_carrying_the_stale_field"]
    assert all(per_row[name]["synthetic_per_batch"] == 0 for name in affected)
    assert reconciliation["blast_radius"]["artifacts_rewritten"].startswith("none")


def test_backend_parity_is_reported_as_measured_not_as_a_pass():
    parity = disclosures.backend_parity_record(Path("reports/m10/A09_BACKEND_PARITY.json"))
    assert parity["kind"] == "parity_not_superiority"
    assert parity["in_holm_bonferroni_family"] is False
    assert parity["tolerance_widened"] is False
    # The failing check is named, not summarised away.
    assert parity["failed_checks"] == ["global_logits"]
    assert parity["checks_passed"] < parity["checks_total"]
    assert parity["measured"]["mean_tolerance_exceeded_by_percent"] > 0


def test_the_a02_disclosure_states_the_pool_difference_and_the_ood_conditioning():
    a02 = disclosures.a02_disclosure(Path("reports/m10"))
    assert a02["conditioning_is_out_of_distribution"] is True
    assert a02["pools"]["random_operator_accepted"] == 838
    assert a02["pools"]["structured_accepted"] == 871
    assert a02["pools"]["pool_size_difference_percent"] == pytest.approx(3.79, abs=0.02)
    assert a02["training_exposure_is_equal"]["g5_optimizer_steps"] == 1350


# --- the HTML rendering never invents a number ------------------------------
def test_html_renders_not_applicable_rather_than_a_blank_or_a_zero():
    from prism_fas.evaluation.contracts import not_applicable
    payload = {"report_identity": "r" * 64, "target_labels_revealed": True,
               "sections": {name: {"status": "not_yet_scored", "reason": "x"}
                            for name in report.REPORT_SECTIONS}}
    payload["sections"]["reproducibility"] = {"m10_matrix_identity": "x" * 64,
                                              "config_identity_sha256": "y" * 64,
                                              "registry_identity": "z" * 64}
    payload["sections"]["negative_and_blocked"] = {"counts": {"failed": 0}}
    summary = {"per_seed": {"B08-s20260806": {"threshold": 0.5,
                                              "video": {"acer": 0.1, "apcer": None,
                                                        "bpcer": not_applicable("no attacks"),
                                                        "hter": 0.2, "roc_auc": 0.9, "eer": 0.1}}},
               "by_row": {}, "not_claimed": ["state-of-the-art"]}
    html = report.render_html(payload, summary)
    assert "not_applicable" in html and "&mdash;" in html
    assert "0.00000" not in html.split("negative results")[0]
    assert "state-of-the-art" in html


# --- contract 2b: the frozen threshold belongs to p_global -------------------
def test_the_decision_follows_p_global_not_the_fusion():
    """The regression that motivated schema v2.

    With a regional term near 1.0 the fusion saturates, so deciding on `s_final`
    against a threshold fitted on `p_global` calls every sample spoof. Measured on
    source_dev this was BPCER 1.0 / ACER 0.5.
    """
    row = _row(1, p_global=0.30, s_region=0.96)
    assert row["decision_score"] == pytest.approx(0.30)
    assert row["s_final"] == pytest.approx(1.0 - 0.70 * 0.04)      # 0.972, the fusion
    assert row["decision"] == "live"                               # not "spoof"
    assert row["confidence"] == pytest.approx(0.70)                # from the decision score


def test_decision_score_equals_the_fusion_when_there_is_no_regional_branch():
    """B00-B05 are unaffected by the revision, by construction rather than by luck."""
    row = _row(1, p_global=0.42, s_region=None)
    assert row["decision_score"] == pytest.approx(row["s_final"]) == pytest.approx(0.42)
    assert row["region_status"] == "not_applicable" and row["s_region"] is None


def test_scoring_reports_the_fused_evidence_without_giving_it_an_acer(tmp_path):
    """The fused score keeps its threshold-free metrics and is never given an ACER
    at a threshold that does not belong to it."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from prism_fas.evaluation.firewall import TargetLabelFirewall, load_firewall_config
    import yaml
    payload = yaml.safe_load(Path("configs/evaluation/m10_target.yaml").read_text(encoding="utf-8"))
    for name in ("source", "features", "labels", "runs"): (tmp_path / name).mkdir()
    payload["roots"].update({"source_package_root": str(tmp_path / "source"),
                             "target_feature_root": str(tmp_path / "features"),
                             "target_label_root": str(tmp_path / "labels"),
                             "prediction_root": str(tmp_path / "runs")})
    config_path = tmp_path / "evaluation.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    firewall = TargetLabelFirewall(load_firewall_config(config_path), project_root=tmp_path)

    rows = [_row(index, p_global=0.10 + 0.15 * index, s_region=0.95) for index in range(1, 7)]
    lock = _lock(rows, "B08-s20260806")
    labels_path = tmp_path / "labels" / "siw_target_labels.parquet"
    pq.write_table(pa.table({"video_id": [row["video_id"] for row in rows],
                             "label": [index % 2 for index in range(len(rows))],
                             "attack_family": ["Replay" if index % 2 else "live"
                                               for index in range(len(rows))]}), labels_path)
    labels = scoring.load_evaluation_labels(labels_path, firewall=firewall, stage="G8")
    result = scoring.score(predictions=rows, lock=lock, labels=labels, threshold=0.5)
    fused = result["fused_evidence"]
    assert fused["threshold_free_only"] is True
    assert "roc_auc" in fused["video"] and "eer" in fused["video"]
    assert "acer" not in fused["video"] and "apcer" not in fused["video"]
    # And the headline metrics are on the decision quantity.
    assert "calibrated p_global" in result["decision_score_definition"]


def test_the_revealed_flag_cannot_be_set_without_the_reveal_record(tmp_path):
    """`target_labels_revealed` is one-way, and `true` is earned, not typed.

    The flag lives in a YAML file, so the loader — not the file — is what makes the
    transition one-way: `true` is accepted only when the artifact recording the
    first authorized read exists beside it.
    """
    import yaml
    payload = yaml.safe_load(Path("configs/evaluation/m10_target.yaml").read_text(encoding="utf-8"))
    config = tmp_path / "evaluation.yaml"

    payload["target_labels_revealed"] = True
    config.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(M10ContractError, match="one-way"):
        g7.load_evaluation_config(config, reveal_path=tmp_path / "absent.json")
    (tmp_path / "reveal.json").write_text("{}", encoding="utf-8")
    assert g7.load_evaluation_config(config, reveal_path=tmp_path / "reveal.json")

    payload["target_labels_revealed"] = False
    config.write_text(yaml.safe_dump(payload), encoding="utf-8")
    assert g7.load_evaluation_config(config, reveal_path=tmp_path / "absent.json")

    payload["target_labels_revealed"] = "yes"
    config.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(M10ContractError, match="target_labels_revealed: false"):
        g7.load_evaluation_config(config, reveal_path=tmp_path / "reveal.json")
