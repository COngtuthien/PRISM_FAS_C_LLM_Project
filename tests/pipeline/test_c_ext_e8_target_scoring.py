"""Tests for src/prism_fas/evaluation/c_ext_e8_target_scoring.py (E8 G8 +
aggregation/closure).

Pure, synthetic-fixture tests. No GPU, no torch model, no real target
predictions/labels. Per-run "scored" results are built directly in the exact
shape ``prism_fas.evaluation.scoring.score()`` returns, so the aggregation/
paired-comparison/closure logic is exercised without needing real target
features or a real label artifact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from prism_fas.evaluation import c_ext_e8_calibration_authority as auth  # noqa: E402
from prism_fas.evaluation import c_ext_e8_target_evaluation as g7  # noqa: E402
from prism_fas.evaluation import c_ext_e8_target_scoring as g8  # noqa: E402
from prism_fas.evaluation import c_ext_e8_training_runner as runner  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# No-training-capability self-audit
# --------------------------------------------------------------------------- #

def test_g8_scorer_has_no_training_capability():
    audit = g8.assert_g8_scorer_has_no_training_capability()
    assert audit["passed"] is True
    assert audit["holds_model"] is False
    assert audit["holds_optimizer"] is False
    assert audit["can_save_checkpoint"] is False
    assert audit["forbidden_imports"] == []


def test_g8_scorer_module_never_imports_torch_at_module_level():
    source = Path(g8.__file__).read_text(encoding="utf-8")
    top_level = source.split("\ndef ")[0].split("\nclass ")[0]
    assert "import torch" not in top_level


# --------------------------------------------------------------------------- #
# Target label use record (Part C)
# --------------------------------------------------------------------------- #

def test_label_use_record_bound_to_real_historical_reveal():
    lockset = {"lockset_identity": "fake-lockset-identity"}
    record = g8.build_target_label_use_record(REPO, lockset=lockset, scoring_code_commit="deadbeef")
    assert record["status"] == "BOUND"
    assert record["is_first_ever_blind_reveal"] is False
    assert record["e8_uses_target_labels_in_training_or_g7"] is False
    assert record["e8_prediction_lockset_identity"] == "fake-lockset-identity"
    assert record["historical_reveal_artifact_path"] == g8.HISTORICAL_REVEAL_ARTIFACT_RELATIVE_PATH
    assert len(record["historical_reveal_artifact_sha256"]) == 64
    assert record["target_label_artifact_sha256"]  # copied verbatim, never recomputed here


def test_label_use_record_never_opens_label_parquet_itself():
    source = Path(g8.__file__).read_text(encoding="utf-8")
    start = source.index("def build_target_label_use_record")
    end = source.index("\ndef ", start + 1)
    body = source[start:end]
    assert "read_table" not in body
    assert "siw_target_labels" not in body


def test_label_use_record_fails_closed_on_missing_reveal_artifact(tmp_path):
    (tmp_path / "reports").mkdir()
    lockset = {"lockset_identity": "fake-id"}
    record = g8.build_target_label_use_record(
        tmp_path, lockset=lockset, reveal_artifact_relative_path="reports/does_not_exist.json")
    assert record["status"] == "PROVENANCE_GAP"
    assert record["is_first_ever_blind_reveal"] is False


# --------------------------------------------------------------------------- #
# Synthetic per-run "scored" fixtures
# --------------------------------------------------------------------------- #

def _make_video_population(n: int = 40, seed: int = 0) -> tuple[list[str], list[int]]:
    rng = np.random.default_rng(seed)
    video_ids = [f"siw_{i:016x}" for i in range(n)]
    labels = [0] * (n // 2) + [1] * (n - n // 2)
    return video_ids, labels


_VIDEO_IDS, _LABELS = _make_video_population()


def _fake_scored_run(arm: str, seed: int, *, bias: float = 0.0, threshold: float = 0.5,
                     rng_seed: int = 1) -> dict[str, Any]:
    rng = np.random.default_rng(rng_seed)
    scores = []
    for label in _LABELS:
        base = rng.normal(0.3 if label == 0 else 0.7, 0.12) + bias
        scores.append(float(np.clip(base, 0.0, 1.0)))
    preds = np.asarray(scores)
    lbls = np.asarray(_LABELS)
    tp = int(((preds >= threshold) & (lbls == 1)).sum())
    fn = int(((preds < threshold) & (lbls == 1)).sum())
    fp = int(((preds >= threshold) & (lbls == 0)).sum())
    tn = int(((preds < threshold) & (lbls == 0)).sum())
    apcer = fn / (tp + fn) if (tp + fn) else 0.0
    bpcer = fp / (fp + tn) if (fp + tn) else 0.0
    acer = (apcer + bpcer) / 2.0
    video_scores = [{"video_id": vid, "video_score": score, "video_fused_score": score,
                     "video_confidence": max(score, 1 - score),
                     "decision": "spoof" if score >= threshold else "live", "label": label}
                    for vid, score, label in zip(_VIDEO_IDS, scores, _LABELS)]
    return {"video": {"apcer": apcer, "bpcer": bpcer, "acer": acer, "roc_auc": 0.85, "eer": 0.15,
                      "calibration": {"ece": 0.05, "brier": 0.15, "nll": 0.4}},
           "threshold": threshold, "video_scores": video_scores}


_ARM_SEED_OFFSET = {"RND": 0, "DET": 100, "LLM": 200}


def _fake_results_and_lockset(bias_by_arm: dict[str, float]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    entries = []
    for arm, bias in bias_by_arm.items():
        for i, seed in enumerate(runner.SEEDS):
            run_id = f"e8_ext_f1_g_{arm.lower()}_qmatch_s{seed}"
            results[run_id] = _fake_scored_run(arm, seed, bias=bias, rng_seed=i + _ARM_SEED_OFFSET[arm])
            entries.append({"run_id": run_id, "arm": arm, "seed": seed})
    return results, {"entries": entries, "lockset_identity": "fake-lockset"}


# --------------------------------------------------------------------------- #
# Per-run table / aggregation (Part E)
# --------------------------------------------------------------------------- #

def test_per_run_table_has_15_rows_all_required_metrics():
    results, lockset = _fake_results_and_lockset({"RND": 0.0, "DET": 0.0, "LLM": 0.0})
    rows = g8.build_per_run_table(results, lockset)
    assert len(rows) == 15
    for row in rows:
        for metric in g8.REQUIRED_METRICS:
            assert metric in row


def test_arm_summary_n5_mean_std_ddof0():
    results, lockset = _fake_results_and_lockset({"RND": 0.0, "DET": 0.02, "LLM": -0.02})
    rows = g8.build_per_run_table(results, lockset)
    summary = g8.build_arm_summary(rows)
    assert set(summary) == {"RND", "DET", "LLM"}
    for arm, entry in summary.items():
        assert entry["n"] == 5
        acer_values = [row["acer"] for row in rows if row["arm"] == arm]
        assert entry["acer_mean"] == pytest.approx(float(np.mean(acer_values)))
        assert entry["acer_std"] == pytest.approx(float(np.std(acer_values, ddof=0)))


def test_arm_summary_refuses_wrong_seed_count():
    results, lockset = _fake_results_and_lockset({"RND": 0.0, "DET": 0.0, "LLM": 0.0})
    rows = g8.build_per_run_table(results, lockset)
    with pytest.raises(g8.E8TargetScoringError, match="expected exactly 5"):
        g8.aggregate_arm(rows[:4], "RND")


def test_no_extra_or_missing_run_accepted():
    results, lockset = _fake_results_and_lockset({"RND": 0.0, "DET": 0.0, "LLM": 0.0})
    lockset["entries"].append({"run_id": "not_a_real_run", "arm": "RND", "seed": 20260806})
    with pytest.raises(KeyError):
        g8.build_per_run_table(results, lockset)


# --------------------------------------------------------------------------- #
# Paired same-seed comparisons (reuses bootstrap.paired_bootstrap)
# --------------------------------------------------------------------------- #

def test_paired_comparisons_cover_all_three_pairs_and_five_seeds():
    results, lockset = _fake_results_and_lockset({"RND": 0.0, "DET": 0.0, "LLM": -0.15})
    rows = g8.build_per_run_table(results, lockset)
    from prism_fas.evaluation.bootstrap import BootstrapSettings

    settings = BootstrapSettings(resamples=200)  # small, deterministic, fast for tests
    comparisons = g8.paired_same_seed_comparisons(results, rows, settings=settings)
    assert set(comparisons) == {"RND_vs_DET", "RND_vs_LLM", "DET_vs_LLM"}
    for name, entry in comparisons.items():
        assert len(entry["per_seed"]) == 5
        assert "mean_paired_acer_delta" in entry
        for seed_row in entry["per_seed"]:
            assert seed_row["seed"] in runner.SEEDS


def test_descriptive_seed_summary_is_purely_descriptive_no_inference():
    results, lockset = _fake_results_and_lockset({"RND": 0.0, "DET": 0.0, "LLM": -0.3})
    rows = g8.build_per_run_table(results, lockset)
    from prism_fas.evaluation.bootstrap import BootstrapSettings

    comparisons = g8.paired_same_seed_comparisons(results, rows, settings=BootstrapSettings(resamples=300))
    summary = g8.build_descriptive_seed_summary(comparisons)
    assert summary["inferential_status"] == "NOT_CLAIMED"
    assert summary["statistical_significance_claimed"] is False
    assert summary["multiple_comparison_correction"] == "NOT_APPLICABLE"
    assert set(summary["comparisons"]) == {"RND_vs_DET", "RND_vs_LLM", "DET_vs_LLM"}
    for entry in summary["comparisons"].values():
        assert entry["n"] == 5
        # no inferential fields exist anywhere in the descriptive summary
        for forbidden in ("p_value", "sign_flip_p_value", "holm_adjusted_p_value",
                          "significant_at_alpha", "ci_95_low", "ci_95_high"):
            assert forbidden not in entry


def test_descriptive_seed_summary_uses_all_five_seeds_never_one():
    results, lockset = _fake_results_and_lockset({"RND": 0.0, "DET": 0.0, "LLM": -0.3})
    rows = g8.build_per_run_table(results, lockset)
    from prism_fas.evaluation.bootstrap import BootstrapSettings

    comparisons = g8.paired_same_seed_comparisons(results, rows, settings=BootstrapSettings(resamples=200))
    summary = g8.build_descriptive_seed_summary(comparisons)
    for name, entry in summary["comparisons"].items():
        assert entry["n"] == 5
        assert len(entry["deltas"]) == 5
        assert sorted(entry["seeds"]) == sorted(runner.SEEDS)
        source_seeds = [row["seed"] for row in comparisons[name]["per_seed"]]
        assert sorted(source_seeds) == sorted(runner.SEEDS)


def test_descriptive_seed_summary_reports_mean_std_ddof0_and_direction_counts():
    results, lockset = _fake_results_and_lockset({"RND": 0.0, "DET": 0.0, "LLM": -0.3})
    rows = g8.build_per_run_table(results, lockset)
    from prism_fas.evaluation.bootstrap import BootstrapSettings

    comparisons = g8.paired_same_seed_comparisons(results, rows, settings=BootstrapSettings(resamples=200))
    summary = g8.build_descriptive_seed_summary(comparisons)
    for name, entry in summary["comparisons"].items():
        deltas = entry["deltas"]
        assert entry["mean_delta"] == pytest.approx(float(np.mean(deltas)))
        assert entry["std_delta_ddof0"] == pytest.approx(float(np.std(deltas, ddof=0)))
        assert entry["count_positive"] + entry["count_negative"] + entry["count_zero"] == 5


def test_descriptive_seed_summary_refuses_partial_seed_set():
    results, lockset = _fake_results_and_lockset({"RND": 0.0, "DET": 0.0, "LLM": 0.0})
    rows = g8.build_per_run_table(results, lockset)
    from prism_fas.evaluation.bootstrap import BootstrapSettings

    comparisons = g8.paired_same_seed_comparisons(results, rows, settings=BootstrapSettings(resamples=100))
    tampered = dict(comparisons)
    key = next(iter(tampered))
    tampered[key] = dict(tampered[key])
    tampered[key]["per_seed"] = tampered[key]["per_seed"][:4]  # drop one seed
    with pytest.raises(g8.E8TargetScoringError, match="frozen detector seeds"):
        g8.build_descriptive_seed_summary(tampered)


def test_paired_same_seed_comparisons_refuses_partial_arm_seed_set():
    """paired_same_seed_comparisons itself must never silently intersect a
    partial seed set for either arm in a pair."""
    results, lockset = _fake_results_and_lockset({"RND": 0.0, "DET": 0.0, "LLM": 0.0})
    rows = g8.build_per_run_table(results, lockset)
    truncated_rows = [row for row in rows if not (row["arm"] == "DET" and row["seed"] == runner.SEEDS[0])]
    with pytest.raises(g8.E8TargetScoringError, match="frozen detector seeds"):
        g8.paired_same_seed_comparisons(results, truncated_rows)


def test_diagnostic_per_seed_bootstrap_never_used_as_family_p_value():
    results, lockset = _fake_results_and_lockset({"RND": 0.0, "DET": 0.0, "LLM": -0.2})
    rows = g8.build_per_run_table(results, lockset)
    from prism_fas.evaluation.bootstrap import BootstrapSettings

    comparisons = g8.paired_same_seed_comparisons(results, rows, settings=BootstrapSettings(resamples=100))
    for entry in comparisons.values():
        assert "family_p_value" not in entry
        for seed_row in entry["per_seed"]:
            assert "diagnostic_video_bootstrap_p_value" in seed_row
            assert "p_value" not in seed_row  # renamed to make its (non-authoritative) role explicit


# --------------------------------------------------------------------------- #
# Closure (Part E)
# --------------------------------------------------------------------------- #

def test_closure_no_llm_superiority_claim_without_significance():
    results, lockset = _fake_results_and_lockset({"RND": 0.0, "DET": 0.0, "LLM": 0.0})
    rows = g8.build_per_run_table(results, lockset)
    from prism_fas.evaluation.bootstrap import BootstrapSettings

    comparisons = g8.paired_same_seed_comparisons(results, rows, settings=BootstrapSettings(resamples=200))
    summary = g8.build_descriptive_seed_summary(comparisons)
    arm_summary = g8.build_arm_summary(rows)
    closure = g8.build_closure(arm_summary, summary)
    assert closure["llm_superiority_supported"] is False


def test_closure_llm_superiority_never_true_even_when_llm_has_best_mean():
    """Whichever arm descriptively has the lowest mean ACER, LLM superiority
    must NOT be claimed unless it demonstrably IS that arm and even then only
    descriptively -- there is no valid inferential authority to promote a
    descriptive rank into a superiority claim."""
    results, lockset = _fake_results_and_lockset({"RND": 0.0, "DET": 0.0, "LLM": -0.5})
    rows = g8.build_per_run_table(results, lockset)
    from prism_fas.evaluation.bootstrap import BootstrapSettings

    comparisons = g8.paired_same_seed_comparisons(results, rows, settings=BootstrapSettings(resamples=100))
    summary = g8.build_descriptive_seed_summary(comparisons)
    arm_summary = g8.build_arm_summary(rows)
    closure = g8.build_closure(arm_summary, summary)
    # regardless of which arm descriptively ranks best here, superiority is never claimed
    assert closure["llm_superiority_supported"] is False
    assert closure["inferential_status"] == "NOT_CLAIMED"


def test_closure_pre_qmatch_comparison_marked_unavailable():
    results, lockset = _fake_results_and_lockset({"RND": 0.0, "DET": 0.0, "LLM": 0.0})
    rows = g8.build_per_run_table(results, lockset)
    from prism_fas.evaluation.bootstrap import BootstrapSettings

    comparisons = g8.paired_same_seed_comparisons(results, rows, settings=BootstrapSettings(resamples=200))
    summary = g8.build_descriptive_seed_summary(comparisons)
    arm_summary = g8.build_arm_summary(rows)
    closure = g8.build_closure(arm_summary, summary)
    assert closure["pre_qmatch_frozen_f1_comparison"]["available"] is False
    assert closure["ranking_change_vs_pre_qmatch"] == "UNAVAILABLE"
    assert closure["controls_additional_f2_f3_work"] is False


def test_closure_answers_the_required_question_explicitly():
    results, lockset = _fake_results_and_lockset({"RND": 0.0, "DET": 0.0, "LLM": 0.0})
    rows = g8.build_per_run_table(results, lockset)
    from prism_fas.evaluation.bootstrap import BootstrapSettings

    comparisons = g8.paired_same_seed_comparisons(results, rows, settings=BootstrapSettings(resamples=200))
    summary = g8.build_descriptive_seed_summary(comparisons)
    arm_summary = g8.build_arm_summary(rows)
    closure = g8.build_closure(arm_summary, summary)
    assert closure["question"] == "After q matching, do RND/DET/LLM performance differences remain?"
    assert isinstance(closure["descriptive_differences_remain"], bool)
    assert isinstance(closure["statistical_significance_claimed"], bool)


def test_closure_states_inferential_status_not_claimed_explicitly():
    results, lockset = _fake_results_and_lockset({"RND": 0.0, "DET": 0.02, "LLM": -0.02})
    rows = g8.build_per_run_table(results, lockset)
    from prism_fas.evaluation.bootstrap import BootstrapSettings

    comparisons = g8.paired_same_seed_comparisons(results, rows, settings=BootstrapSettings(resamples=100))
    summary = g8.build_descriptive_seed_summary(comparisons)
    arm_summary = g8.build_arm_summary(rows)
    closure = g8.build_closure(arm_summary, summary)
    assert closure["inferential_status"] == "NOT_CLAIMED"
    assert closure["statistical_significance_claimed"] is False
    assert closure["multiple_comparison_correction"] == "NOT_APPLICABLE"
    # must never phrase this as "no significant difference" -- no test was run
    assert "no significant difference" not in closure["answer"].lower()
    assert "not claimed" in closure["answer"].lower()


def test_closure_never_computes_a_p_value_or_holm_correction():
    results, lockset = _fake_results_and_lockset({"RND": 0.0, "DET": 0.0, "LLM": 0.0})
    rows = g8.build_per_run_table(results, lockset)
    from prism_fas.evaluation.bootstrap import BootstrapSettings

    comparisons = g8.paired_same_seed_comparisons(results, rows, settings=BootstrapSettings(resamples=100))
    summary = g8.build_descriptive_seed_summary(comparisons)
    arm_summary = g8.build_arm_summary(rows)
    closure = g8.build_closure(arm_summary, summary)
    for forbidden in ("p_value", "sign_flip", "holm", "significant_pairs"):
        assert forbidden not in json.dumps(closure).lower()


# --------------------------------------------------------------------------- #
# G8 write policy: never model state, always via the firewall
# --------------------------------------------------------------------------- #

def test_g8_write_refuses_model_state_patterns(tmp_path):
    firewall = _fake_firewall(tmp_path)
    for bad in ("checkpoints/best.pt", "calibration/source_dev.json", "model.safetensors"):
        with pytest.raises(Exception):
            firewall.check_write("G8", tmp_path / bad)


def _fake_firewall(repo: Path):
    from prism_fas.evaluation.firewall import FirewallConfig, TargetLabelFirewall

    config = FirewallConfig(
        roots={"source_package_root": repo / "src", "target_feature_root": repo / "tf",
              "target_label_root": repo / "tl", "prediction_root": repo / "pred"},
        permissions={
            "TRAIN": {"source_package_root": "read", "target_feature_root": "deny",
                     "target_label_root": "deny", "prediction_root": "deny"},
            "G7": {"source_package_root": "deny", "target_feature_root": "read",
                  "target_label_root": "deny", "prediction_root": "write"},
            "G8": {"source_package_root": "deny", "target_feature_root": "deny",
                  "target_label_root": "read", "prediction_root": "read"},
        }).validate()
    return TargetLabelFirewall(config=config, project_root=repo)


def test_write_all_evidence_produces_all_required_files(tmp_path):
    results, lockset = _fake_results_and_lockset({"RND": 0.0, "DET": 0.0, "LLM": 0.0})
    rows = g8.build_per_run_table(results, lockset)
    from prism_fas.evaluation.bootstrap import BootstrapSettings

    comparisons = g8.paired_same_seed_comparisons(results, rows, settings=BootstrapSettings(resamples=200))
    descriptive_summary = g8.build_descriptive_seed_summary(comparisons)
    summary = g8.build_arm_summary(rows)
    closure = g8.build_closure(summary, descriptive_summary)
    label_record = {"status": "PROVENANCE_GAP", "reason": "test fixture"}
    firewall = _fake_firewall(tmp_path)

    written = g8.write_all_evidence(
        tmp_path, per_run_rows=rows, arm_summary=summary, paired_comparisons=comparisons,
        descriptive_summary=descriptive_summary, closure=closure, label_use_record=label_record,
        firewall=firewall)

    for key in ("score_result", "per_run_csv", "arm_summary_csv", "paired_comparisons", "closure",
               "label_use_record", "evidence_sha256", "summary_md"):
        assert key in written
        candidate = Path(written[key])
        resolved = candidate if candidate.is_absolute() else tmp_path / candidate
        assert resolved.is_file(), f"{key}: {resolved} missing"

    # scoring never modifies model/optimizer/calibration/threshold/predictions
    score_result = json.loads((tmp_path / g8.SCORE_RESULT_PATH).read_text(encoding="utf-8"))
    assert "per_run" in score_result and len(score_result["per_run"]) == 15


def test_write_all_evidence_never_writes_model_state(tmp_path):
    source = Path(g8.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "torch.save" not in source
    assert ".pt\"" not in source and "'.pt'" not in source


# --------------------------------------------------------------------------- #
# Structural: no LLM API calls anywhere
# --------------------------------------------------------------------------- #

def test_module_never_calls_llm_api():
    source = Path(g8.__file__).read_text(encoding="utf-8")
    assert "openai" not in source.lower()
    assert "gemini" not in source.lower()
    assert "anthropic" not in source.lower()


# =========================================================================== #
# E8-EVAL-V1.1 -- G8 operator CLI + strict execution ordering
# =========================================================================== #

def test_g8_preflight_reads_zero_target_labels():
    result = g8.preflight_e8_g8_scoring(REPO)
    assert result["target_labels_accessed"] is False
    assert result["scoring_performed"] is False
    assert result["no_training_capability"] is True


def test_g8_preflight_never_calls_load_evaluation_labels():
    source = Path(g8.__file__).read_text(encoding="utf-8")
    start = source.index("def preflight_e8_g8_scoring")
    end = source.index("\ndef ", start + 1)
    body = source[start:end]
    assert "load_evaluation_labels" not in body
    assert "score_all_e8_runs(" not in body


def test_cli_score_requires_explicit_authorization(capsys):
    exit_code = g8.main(["--score"])
    assert exit_code == 2
    assert "authorize-target-label-scoring" in capsys.readouterr().out


def test_cli_preflight_exits_zero(capsys):
    with mock.patch.object(g8.cc, "repo_root", lambda: REPO):
        exit_code = g8.main(["--preflight"])
    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["target_labels_accessed"] is False


def _seed_matched_calibration_lock():
    rows = [{"run_id": s.run_id, "selected_threshold": 0.5} for s in runner.all_scientific_run_specs()]
    return {"lock_identity": "fake-auth", "rows": rows}


def _seed_matched_lockset():
    entries = [{"run_id": s.run_id, "arm": s.arm, "seed": s.seed} for s in runner.all_scientific_run_specs()]
    return {"entries": entries, "lockset_identity": "fake-lockset", "code_commit": "d" * 40}


def test_run_e8_g8_scoring_validates_lockset_before_label_read():
    calls: list = []

    def tracking_labels_loader(*, repo, firewall):
        calls.append("labels_loaded")
        return object()

    def tracking_score_all(repo, *, labels, firewall=None):
        calls.append("scored")
        results = {}
        for spec in runner.all_scientific_run_specs():
            video_scores = [{"video_id": f"v{i}", "video_score": 0.3 if i % 2 == 0 else 0.7,
                             "video_fused_score": 0.3, "video_confidence": 0.7,
                             "decision": "live" if i % 2 == 0 else "spoof", "label": i % 2}
                            for i in range(20)]
            results[spec.run_id] = {"video": {"apcer": 0.1, "bpcer": 0.1, "acer": 0.1, "roc_auc": 0.9,
                                              "eer": 0.1, "calibration": {"ece": 0.05, "brier": 0.1, "nll": 0.3}},
                                    "threshold": 0.5, "video_scores": video_scores}
        return results

    with mock.patch.object(g7, "require_valid_lockset_for_g8", lambda root=None: (
            calls.append("lockset_validated") or _seed_matched_lockset())), \
         mock.patch.object(auth, "load_calibration_authority_lock_if_usable", lambda root=None: (
            calls.append("calibration_authority_validated") or _seed_matched_calibration_lock())), \
         mock.patch.object(g8, "build_target_label_use_record", lambda repo, *, lockset, scoring_code_commit="": (
            calls.append("reveal_validated") or {"status": "BOUND"})), \
         mock.patch.object(g8, "score_all_e8_runs", tracking_score_all), \
         mock.patch.object(g8, "write_all_evidence", lambda *a, **k: (calls.append("evidence_written"), {})[1]):
        from prism_fas.evaluation.bootstrap import BootstrapSettings
        g8.run_e8_g8_scoring(REPO, _labels_loader=tracking_labels_loader,
                             bootstrap_settings=BootstrapSettings(resamples=50))

    assert calls == ["lockset_validated", "calibration_authority_validated", "reveal_validated",
                     "labels_loaded", "scored", "evidence_written"]


def test_run_e8_g8_scoring_blocks_incomplete_lockset_before_label_read():
    calls: list = []

    def failing_lockset(root=None):
        raise g7.E8TargetEvaluationError("fixture: incomplete lockset")

    def forbidden_labels_loader(*, repo, firewall):
        calls.append("labels_loaded")
        raise AssertionError("labels must never be read when the lockset is incomplete")

    with mock.patch.object(g7, "require_valid_lockset_for_g8", failing_lockset):
        with pytest.raises(g7.E8TargetEvaluationError, match="incomplete lockset"):
            g8.run_e8_g8_scoring(REPO, _labels_loader=forbidden_labels_loader)
    assert "labels_loaded" not in calls


def test_run_e8_g8_scoring_blocks_missing_calibration_authority_before_label_read():
    calls: list = []

    def forbidden_labels_loader(*, repo, firewall):
        calls.append("labels_loaded")
        raise AssertionError("labels must never be read when calibration authority is missing")

    with mock.patch.object(g7, "require_valid_lockset_for_g8", lambda root=None: _seed_matched_lockset()), \
         mock.patch.object(auth, "load_calibration_authority_lock_if_usable", lambda root=None: None):
        with pytest.raises(g8.E8TargetScoringError, match="no valid calibration authority lock"):
            g8.run_e8_g8_scoring(REPO, _labels_loader=forbidden_labels_loader)
    assert "labels_loaded" not in calls


def test_run_e8_g8_scoring_blocks_on_missing_reveal_provenance_before_label_read():
    calls: list = []

    def forbidden_labels_loader(*, repo, firewall):
        calls.append("labels_loaded")
        raise AssertionError("labels must never be read when reveal provenance is not BOUND")

    with mock.patch.object(g7, "require_valid_lockset_for_g8", lambda root=None: _seed_matched_lockset()), \
         mock.patch.object(auth, "load_calibration_authority_lock_if_usable",
                           lambda root=None: _seed_matched_calibration_lock()), \
         mock.patch.object(g8, "build_target_label_use_record", lambda repo, *, lockset, scoring_code_commit="":
                           {"status": "PROVENANCE_GAP", "reason": "fixture"}):
        with pytest.raises(g8.E8TargetScoringError, match="refusing to open target labels"):
            g8.run_e8_g8_scoring(REPO, _labels_loader=forbidden_labels_loader)
    assert "labels_loaded" not in calls


def test_missing_prediction_row_blocks_before_label_read():
    """An incomplete/extra prediction row is caught at lockset validation
    (g7.require_valid_lockset_for_g8 / is_usable_lockset), long before any
    label could be read."""
    incomplete_lockset = _seed_matched_lockset()
    incomplete_lockset["entries"] = incomplete_lockset["entries"][:14]
    incomplete_lockset["run_count"] = 14
    assert g7.is_usable_lockset(incomplete_lockset) is False


# =========================================================================== #
# E8-EVAL-V1.1 correction -- zero production dependency on the untracked
# c_ext_e5_synthetic_comparison.py (absent from git at the frozen base
# commit; present only as a local, untracked file in some worktrees).
# =========================================================================== #

def test_g8_module_never_imports_untracked_e5_synthetic_comparison():
    """Structural: no import statement anywhere in the G8 module names the
    untracked module, under any alias."""
    source = Path(g8.__file__).read_text(encoding="utf-8")
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            assert "c_ext_e5_synthetic_comparison" not in stripped, f"forbidden import: {stripped!r}"


def test_g8_import_closure_never_reaches_untracked_e5_synthetic_comparison():
    """Behavioral: the module's own transitive first-party import-closure
    audit (used to prove no-training-capability) must never resolve to the
    untracked module either."""
    audit = g8.assert_g8_scorer_has_no_training_capability()
    audited = audit["import_closure"]["modules_audited"]
    assert not any("c_ext_e5_synthetic_comparison" in name for name in audited)


def test_g8_has_no_reference_to_untracked_module_functions():
    """The specific functions named in the audit (sign_flip_p_value,
    paired_t_confidence_interval) must not appear as live production
    identifiers -- only, at most, inside documentation strings explaining
    why they were removed."""
    import ast

    tree = ast.parse(Path(g8.__file__).read_text(encoding="utf-8"))
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name)
    assert "sign_flip_p_value" not in imported_names
    assert "paired_t_confidence_interval" not in imported_names
    assert "seed_level_holm_bonferroni" not in imported_names
    assert not hasattr(g8, "sign_flip_p_value")
    assert not hasattr(g8, "paired_t_confidence_interval")
    assert not hasattr(g8, "build_seed_level_significance")


def test_e8_eval_v1_1_suite_succeeds_without_the_untracked_module_importable():
    """Proves the module can be freshly imported (as a clean GPU worktree
    would) with c_ext_e5_synthetic_comparison made deliberately unimportable
    -- simulating its absence from a clean worktree."""
    import builtins
    import importlib

    real_import = builtins.__import__

    def _blocking_import(name, *args, **kwargs):
        if name == "prism_fas.evaluation.c_ext_e5_synthetic_comparison" or name.endswith(
                ".c_ext_e5_synthetic_comparison"):
            raise ImportError(f"simulated clean worktree: {name} does not exist")
        return real_import(name, *args, **kwargs)

    with mock.patch.object(builtins, "__import__", _blocking_import):
        reloaded = importlib.reload(g8)
        audit = reloaded.assert_g8_scorer_has_no_training_capability()
        assert audit["passed"] is True
    importlib.reload(g8)  # restore normal state for any tests that run after this one
