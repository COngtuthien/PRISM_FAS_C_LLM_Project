import json
from pathlib import Path

import numpy as np
import pytest
import torch

from prism_fas.train.calibration import CalibrationError, apply_temperature, fit_temperature
from prism_fas.train.checkpoint import CheckpointContractError, load_checkpoint, save_checkpoint
from prism_fas.train.config import load_b00_config
from prism_fas.train.inference import FORBIDDEN_TARGET_COLUMNS, TARGET_PREDICTION_SCHEMA, SOURCE_PREDICTION_SCHEMA
from prism_fas.train.losses import LossContractError, b00_binary_cross_entropy
from prism_fas.train.metrics import (MetricInputError, apcer_bpcer_acer, equal_error_rate, expected_calibration_error,
                                     negative_log_likelihood, roc_auc, select_min_acer_threshold)
from prism_fas.train.models.b00_convnext import B00ConvNeXtBinaryClassifier, B00Output
from prism_fas.train.report import TARGET_ISOLATION_STATEMENT, build_report, write_complete
from prism_fas.train.seed import restore_rng_state, rng_state, seed_everything
from prism_fas.train.video_aggregation import aggregate_videos, trimmed_mean

CONFIG = load_b00_config(Path(__file__).parents[1] / "configs" / "train" / "b00_local.yaml")


class _TinyBackbone(torch.nn.Module):
    """Stand-in feature extractor so unit tests never download weights."""
    num_features = 8
    def __init__(self): super().__init__(); self.layer = torch.nn.Conv2d(3, 8, 3, stride=2)
    def forward(self, x): return self.layer(x).mean(dim=(2, 3))


def _tiny_model():
    model = B00ConvNeXtBinaryClassifier.__new__(B00ConvNeXtBinaryClassifier)
    torch.nn.Module.__init__(model)
    model.model_name = "tiny-test"; model.backbone = _TinyBackbone(); model.feature_dim = 8
    model.dropout = torch.nn.Identity(); model.head = torch.nn.Linear(8, 1)
    model.register_buffer("norm_mean", torch.zeros(1, 3, 1, 1)); model.register_buffer("norm_std", torch.ones(1, 3, 1, 1))
    return model


# --- config ----------------------------------------------------------------

def test_config_pins_model_and_label_convention():
    assert CONFIG.model.model_name == "convnextv2_atto.fcmae_ft_in1k" and CONFIG.model.source == "timm"
    assert len(CONFIG.model.weight_sha256) == 64 and CONFIG.model.pretrained
    assert CONFIG.label_mapping == {"live": 0, "spoof": 1}
    assert CONFIG.checkpoint_selection.primary == "source_dev_roc_auc" and CONFIG.checkpoint_selection.mode == "max"
    assert CONFIG.calibration.reject_policy == "disabled_for_b00"
    assert "D:" not in (Path(__file__).parents[1] / "configs" / "train" / "b00_local.yaml").read_text(encoding="utf-8")


# --- model -----------------------------------------------------------------

def test_forward_output_shape_and_probability_convention():
    model = _tiny_model()
    output = model(torch.rand(4, 3, 32, 32), return_features=True)
    assert isinstance(output, B00Output) and output.spoof_logit.shape == (4,)
    assert torch.isfinite(output.spoof_logit).all() and output.feature_embedding.shape == (4, 8)
    with torch.no_grad(): model.head.bias.fill_(5.0)
    high = torch.sigmoid(model(torch.rand(2, 3, 32, 32)).spoof_logit)
    assert (high > 0.9).all(), "sigmoid(logit) must be the SPOOF probability"


def test_parameter_groups_assign_distinct_learning_rates():
    groups = _tiny_model().parameter_groups(backbone_lr=1e-4, head_lr=5e-4, weight_decay=0.05)
    assert [g["name"] for g in groups] == ["backbone", "head"]
    assert groups[0]["lr"] == 1e-4 and groups[1]["lr"] == 5e-4 and all(g["weight_decay"] == 0.05 for g in groups)


def test_normalization_applied_once():
    model = _tiny_model()
    model.norm_mean.fill_(0.5); model.norm_std.fill_(0.25)
    normalized = model.normalize(torch.full((1, 3, 4, 4), 0.75))
    assert torch.allclose(normalized, torch.ones_like(normalized))


# --- loss ------------------------------------------------------------------

def test_bce_known_value_and_reduction():
    logit = torch.zeros(4); target = torch.tensor([0, 1, 0, 1])
    assert float(b00_binary_cross_entropy(logit, target)) == pytest.approx(np.log(2), abs=1e-6)
    confident = torch.tensor([-6.0, 6.0]); labels = torch.tensor([0, 1])
    assert float(b00_binary_cross_entropy(confident, labels)) < float(b00_binary_cross_entropy(-confident, labels))
    assert float(b00_binary_cross_entropy(confident, labels)) >= 0.0


def test_bce_backward_is_finite_and_rejects_bad_inputs():
    model = _tiny_model()
    output = model(torch.rand(4, 3, 32, 32))
    loss = b00_binary_cross_entropy(output.spoof_logit, torch.tensor([0, 1, 1, 0]))
    loss.backward()
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    with pytest.raises(LossContractError, match="0 \\(live\\) and 1 \\(spoof\\)"):
        b00_binary_cross_entropy(torch.zeros(2), torch.tensor([0, 2]))
    with pytest.raises(LossContractError):
        b00_binary_cross_entropy(torch.zeros(2, 1), torch.tensor([0, 1]))


def test_target_batch_cannot_reach_the_loss():
    from prism_fas.data.loader.collate import collate_target_batch
    from prism_fas.data.loader.contracts import TargetIsolationViolation
    class _Fake:  # not a CanonicalTargetSample
        sample_id = "x"
    with pytest.raises(TargetIsolationViolation):
        collate_target_batch([_Fake()])


# --- metrics ---------------------------------------------------------------

def test_apcer_bpcer_acer_known_fixture():
    probabilities = np.array([0.9, 0.4, 0.6, 0.1]); targets = np.array([1, 1, 0, 0])
    row = apcer_bpcer_acer(probabilities, targets, 0.5)
    assert row["apcer"] == pytest.approx(0.5) and row["bpcer"] == pytest.approx(0.5) and row["acer"] == pytest.approx(0.5)
    perfect = apcer_bpcer_acer(np.array([0.9, 0.8, 0.2, 0.1]), targets, 0.5)
    assert perfect["acer"] == 0.0 and perfect["tp_spoof"] == 2 and perfect["tn_live"] == 2


def test_roc_auc_and_eer_fixture():
    targets = np.array([1, 1, 0, 0])
    assert roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), targets) == pytest.approx(1.0)
    assert roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), targets) == pytest.approx(0.0)
    assert equal_error_rate(np.array([0.9, 0.8, 0.2, 0.1]), targets)["eer"] == pytest.approx(0.0)
    with pytest.raises(MetricInputError):
        roc_auc(np.array([0.5, 0.5]), np.array([1, 1]))


def test_threshold_selection_is_deterministic_with_documented_tie_break():
    probabilities = np.array([0.2, 0.8, 0.2, 0.8]); targets = np.array([0, 1, 0, 1])
    first = select_min_acer_threshold(probabilities, targets)
    second = select_min_acer_threshold(probabilities, targets)
    assert first["selected"]["threshold"] == second["selected"]["threshold"]
    assert first["selected"]["acer"] == 0.0 and first["criterion"] == "min_acer"
    assert first["tie_break"] == ["min_acer", "min_apcer", "min_threshold"]


# --- calibration -----------------------------------------------------------

def test_temperature_is_positive_and_improves_nll():
    rng = np.random.default_rng(3)
    targets = rng.integers(0, 2, 300)
    logits = (targets * 2 - 1) * rng.normal(4.0, 1.0, 300)      # over-confident
    temperature = fit_temperature(logits, targets)
    assert temperature > 0
    before = negative_log_likelihood(1 / (1 + np.exp(-logits)), targets)
    after = negative_log_likelihood(apply_temperature(logits, temperature), targets)
    assert after <= before + 1e-6
    assert fit_temperature(logits, targets) == pytest.approx(temperature, rel=1e-9)
    with pytest.raises(CalibrationError):
        fit_temperature(np.array([]), np.array([]))


# --- aggregation -----------------------------------------------------------

def test_video_aggregation_is_deterministic():
    assert trimmed_mean([0.1, 0.2, 0.8, 0.9]) == (0.5, 0)         # 4 frames -> no trim
    assert trimmed_mean(list(np.linspace(0, 1, 20)))[1] == 2
    rows = [{"sample_id": f"s{i}", "source_record_id": "v1", "p_spoof_calibrated": p, "confidence": 0.7}
            for i, p in enumerate([0.9, 0.8, 0.7, 0.6])]
    videos = aggregate_videos(rows, threshold=0.5)
    assert len(videos) == 1 and videos[0]["frames"] == 4 and videos[0]["decision"] == "spoof"
    assert aggregate_videos(rows, 0.5) == videos


# --- checkpoint ------------------------------------------------------------

def _payload():
    return {"model_state": {"w": torch.zeros(2)}, "optimizer_state": {}, "epoch": 3, "global_step": 42,
            "config_hash": "cfg", "package_content_identity": "pkg", "model_name": "m", "label_mapping": {"live": 0, "spoof": 1},
            "rng_state": rng_state(), "sampler_state": {"epoch": 3}}


def test_checkpoint_round_trip_and_atomic_write(tmp_path):
    path = tmp_path / "last.pt"
    digest = save_checkpoint(path, _payload())
    assert path.is_file() and len(digest) == 64 and not list(tmp_path.glob("*.tmp"))
    payload = load_checkpoint(path, config_hash="cfg", package_identity="pkg", model_name="m",
                              label_mapping={"live": 0, "spoof": 1})
    assert payload["epoch"] == 3 and payload["global_step"] == 42 and payload["sampler_state"]["epoch"] == 3


@pytest.mark.parametrize("kwargs,message", [({"config_hash": "other"}, "config hash"),
                                            ({"package_identity": "other"}, "package identity"),
                                            ({"model_name": "other"}, "model name"),
                                            ({"label_mapping": {"live": 1, "spoof": 0}}, "label mapping")])
def test_resume_blocked_on_mismatch(tmp_path, kwargs, message):
    path = tmp_path / "last.pt"; save_checkpoint(path, _payload())
    with pytest.raises(CheckpointContractError, match=message):
        load_checkpoint(path, **kwargs)


def test_rng_state_round_trip():
    seed_everything(42); state = rng_state(); first = torch.rand(3)
    restore_rng_state(state)
    assert torch.allclose(torch.rand(3), first)


# --- prediction schemas and report ----------------------------------------

def test_target_prediction_schema_has_no_label_or_private_columns():
    names = set(TARGET_PREDICTION_SCHEMA.names)
    assert not (names & FORBIDDEN_TARGET_COLUMNS)
    assert "true_target" not in names and "true_label" not in names
    assert {"sample_id", "source_record_id", "p_spoof_calibrated", "decision", "checkpoint_hash",
            "calibration_hash", "inference_config_hash"} <= names
    assert {"true_label", "true_target"} <= set(SOURCE_PREDICTION_SCHEMA.names)


def test_report_states_target_isolation_and_completes_only_with_artifacts(tmp_path):
    run_root = tmp_path / "run"
    for name in ("reports", "predictions", "calibration"): (run_root / name).mkdir(parents=True)
    target_rows = [{"sample_id": "t1", "source_record_id": "siw_a", "p_spoof_calibrated": 0.4, "confidence": 0.6, "decision": "live"}]
    summary = build_report(run_root, run_info={"run_id": "run", "device": "cpu", "source_dev_count": 2},
                           training={"epochs_run": 1, "global_step": 45, "stopped_reason": "x", "history": [],
                                     "selection_rule": "roc", "best": {}, "elapsed_seconds": 1.0},
                           calibration={"temperature": 1.0, "before": {}, "after": {}, "selected_threshold": 0.5,
                                        "threshold_selection": {}, "calibration_hash": "h", "reject_policy": "disabled_for_b00"},
                           source_metrics={"acer": 0.1}, source_video=[], target_rows=target_rows,
                           target_video=[{"video_score": 0.4, "decision": "live"}], environment={"device": "cpu"})
    assert summary["target"]["labels_accessed"] is False and summary["target"]["metrics_reported"] is False
    assert summary["target_isolation_statement"] == TARGET_ISOLATION_STATEMENT
    html = (run_root / "reports" / "report.html").read_text(encoding="utf-8")
    assert "Target labels were not accessed" in html and "B00 local baseline" in html
    with pytest.raises(FileNotFoundError, match="missing artifacts"):
        write_complete(run_root, required=[run_root / "checkpoints" / "best.pt"], payload={})


# --- real smoke run contract ----------------------------------------------

SMOKE = Path(__file__).parents[1] / "runs" / "b00_smoke_seed42"
smoke_required = pytest.mark.skipif(not (SMOKE / "COMPLETE.json").is_file(), reason="B00 smoke run not present")


@smoke_required
def test_real_smoke_run_contract():
    from prism_fas.data.package.manifests import read_manifest
    complete = json.loads((SMOKE / "COMPLETE.json").read_text(encoding="utf-8"))
    assert complete["status"] == "complete" and complete["target_labels_accessed"] is False
    metrics = [json.loads(line) for line in (SMOKE / "logs" / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert metrics and all(np.isfinite(row["train_loss"]) and np.isfinite(row["grad_norm"]) for row in metrics)
    assert metrics[0]["batch_composition"] == {"casia_fasd/live": 8, "casia_fasd/spoof": 8, "msu_mfsd/live": 8, "msu_mfsd/spoof": 8}
    target = read_manifest(SMOKE / "predictions" / "siw_mv2.parquet")
    assert target and not (FORBIDDEN_TARGET_COLUMNS & set(target[0]))
    calibration = json.loads((SMOKE / "calibration" / "source_dev.json").read_text(encoding="utf-8"))
    assert calibration["temperature"] > 0 and calibration["reject_policy"] == "disabled_for_b00"
    assert (SMOKE / "checkpoints" / "best.pt").is_file() and not list((SMOKE / "checkpoints").glob("*.tmp"))
