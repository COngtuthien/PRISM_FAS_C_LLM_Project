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

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

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


def test_holm_correction_reuses_existing_machinery():
    results, lockset = _fake_results_and_lockset({"RND": 0.0, "DET": 0.0, "LLM": -0.3})
    rows = g8.build_per_run_table(results, lockset)
    from prism_fas.evaluation.bootstrap import BootstrapSettings

    comparisons = g8.paired_same_seed_comparisons(results, rows, settings=BootstrapSettings(resamples=300))
    holm = g8.build_holm_correction(comparisons)
    assert holm["policy"] == "holm_bonferroni"
    assert set(holm["raw"]) == {"RND_vs_DET", "RND_vs_LLM", "DET_vs_LLM"}
    assert isinstance(holm["rejected_null"], list)


# --------------------------------------------------------------------------- #
# Closure (Part E)
# --------------------------------------------------------------------------- #

def test_closure_no_llm_superiority_claim_without_significance():
    results, lockset = _fake_results_and_lockset({"RND": 0.0, "DET": 0.0, "LLM": 0.0})
    rows = g8.build_per_run_table(results, lockset)
    from prism_fas.evaluation.bootstrap import BootstrapSettings

    comparisons = g8.paired_same_seed_comparisons(results, rows, settings=BootstrapSettings(resamples=200))
    holm = g8.build_holm_correction(comparisons)
    summary = g8.build_arm_summary(rows)
    closure = g8.build_closure(summary, holm)
    assert closure["llm_superiority_supported"] is False


def test_closure_pre_qmatch_comparison_marked_unavailable():
    results, lockset = _fake_results_and_lockset({"RND": 0.0, "DET": 0.0, "LLM": 0.0})
    rows = g8.build_per_run_table(results, lockset)
    from prism_fas.evaluation.bootstrap import BootstrapSettings

    comparisons = g8.paired_same_seed_comparisons(results, rows, settings=BootstrapSettings(resamples=200))
    holm = g8.build_holm_correction(comparisons)
    summary = g8.build_arm_summary(rows)
    closure = g8.build_closure(summary, holm)
    assert closure["pre_qmatch_frozen_f1_comparison"]["available"] is False
    assert closure["ranking_change_vs_pre_qmatch"] == "UNAVAILABLE"
    assert closure["controls_additional_f2_f3_work"] is False


def test_closure_answers_the_required_question_explicitly():
    results, lockset = _fake_results_and_lockset({"RND": 0.0, "DET": 0.0, "LLM": 0.0})
    rows = g8.build_per_run_table(results, lockset)
    from prism_fas.evaluation.bootstrap import BootstrapSettings

    comparisons = g8.paired_same_seed_comparisons(results, rows, settings=BootstrapSettings(resamples=200))
    holm = g8.build_holm_correction(comparisons)
    summary = g8.build_arm_summary(rows)
    closure = g8.build_closure(summary, holm)
    assert closure["question"] == "After q matching, do RND/DET/LLM performance differences remain?"
    assert isinstance(closure["differences_remain"], bool)


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
    holm = g8.build_holm_correction(comparisons)
    summary = g8.build_arm_summary(rows)
    closure = g8.build_closure(summary, holm)
    label_record = {"status": "PROVENANCE_GAP", "reason": "test fixture"}
    firewall = _fake_firewall(tmp_path)

    written = g8.write_all_evidence(
        tmp_path, per_run_rows=rows, arm_summary=summary, paired_comparisons=comparisons,
        holm_result=holm, closure=closure, label_use_record=label_record, firewall=firewall)

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
