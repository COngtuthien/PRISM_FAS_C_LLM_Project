"""M8 focused tests for the stages implemented so far: source-only pair plan,
differentiable Haar DWT, GPAT model contract and the GPAT losses.

No network access and no model/dataset download. Real integration results are
asserted from the ignored report artifacts produced by the explicit commands.

Stages still to come (see reports/m8/M8_HANDOFF.md): quality calibration/gate,
candidate plan, synthetic bank, shards, validation, export and the Modal runs.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from prism_fas.synthesis.dwt import (BAND_ORDER, DWT_CONVENTION, DWT_RECONSTRUCTION_TOLERANCE_FP32, DWTError,
                                     dwt_bands, haar_dwt2, haar_idwt2, idwt_bands, pack_high, reconstruction_error,
                                     unpack_high)
from prism_fas.synthesis.gpat_contracts import (FORBIDDEN_OUTPUT_FIELDS, GPATBatch, GPATContractError,
                                                LL_INVARIANT_TOLERANCE, check_mask)
from prism_fas.synthesis.gpat_losses import (DEFAULT_WEIGHTS, GPATLossError, artifact_map_loss, compute_losses,
                                             identity_loss, loss_manifest, masked_channel_stats, residual_loss,
                                             strength_loss, style_loss, total_variation_loss)
from prism_fas.synthesis.gpat_model import build_gpat_model, downsample_mask
from prism_fas.synthesis.pair_plan import (ALLOWED_DATASETS, EXPECTED_TRAIN_PAIRS, EXPECTED_VALIDATION_PAIRS,
                                           PAIR_PLAN_SEED, PairPlanError, SOURCE_SPLIT, load_pair_manifest,
                                           summarize_pairs)
from prism_fas.synthesis.quality_models import PINNED, QualityModelError, resolve_weight

ROOT = Path(__file__).parents[1]
REPORTS = ROOT / "reports" / "m8"
PAIRS = REPORTS / "pairs"
CONFIG = ROOT / "configs" / "synthesis" / "gpat_m8.yaml"


def _pairs(name):
    path = PAIRS / name
    if not path.is_file():
        pytest.skip("pair plan missing; run: prism synthesis build-pair-plan")
    return load_pair_manifest(path)


def _report(name):
    path = REPORTS / name
    if not path.is_file():
        pytest.skip(f"{name} missing; run: python scripts/m8_local_smoke.py")
    return json.loads(path.read_text(encoding="utf-8"))


def _masks(batch=2, size=224):
    target = torch.zeros(batch, 1, size, size); target[:, :, 60:160, 60:160] = 1.0
    style = torch.zeros(batch, 1, size, size); style[:, :, 50:170, 50:170] = 1.0
    return target, style


def _batch(batch=2):
    target, style = _masks(batch)
    torch.manual_seed(3)
    return GPATBatch(live_image=torch.rand(batch, 3, 224, 224), source_spoof_image=torch.rand(batch, 3, 224, 224),
                     recipe_conditioning=torch.rand(batch, 41), target_support_mask=target, source_style_mask=style,
                     recipe_strength=torch.full((batch,), 0.35),
                     live_identity_embedding=torch.nn.functional.normalize(torch.rand(batch, 512), dim=1)).validate()


# --- pair plan ---------------------------------------------------------------

def test_pair_plan_counts_are_896_and_224():
    train, validation = _pairs("pair_manifest_train.parquet"), _pairs("pair_manifest_validation.parquet")
    assert len(train) == EXPECTED_TRAIN_PAIRS == 896
    assert len(validation) == EXPECTED_VALIDATION_PAIRS == 224
    assert len(train) + len(validation) == 1120


def test_pair_plan_uses_source_train_live_and_spoof_only():
    rows = _pairs("pair_manifest_train.parquet") + _pairs("pair_manifest_validation.parquet")
    summary = json.loads((PAIRS / "pair_plan_summary.json").read_text(encoding="utf-8"))
    assert summary["source_dev_opened"] is False and summary["target_test_opened"] is False
    assert summary["manifests_opened"] == [f"manifests/{SOURCE_SPLIT}.parquet"]
    for row in rows:
        assert row["live_dataset"] in ALLOWED_DATASETS and row["spoof_dataset"] in ALLOWED_DATASETS


def test_pair_plan_train_validation_record_isolation():
    train, validation = _pairs("pair_manifest_train.parquet"), _pairs("pair_manifest_validation.parquet")
    for key in ("live_source_record_id", "spoof_source_record_id"):
        assert not ({row[key] for row in train} & {row[key] for row in validation}), key


def test_pair_plan_domain_balance_and_relations():
    for name in ("pair_manifest_train.parquet", "pair_manifest_validation.parquet"):
        summary = summarize_pairs(_pairs(name))
        assert summary["domain_relation"]["same_domain"] == summary["domain_relation"]["cross_domain"]
        assert set(summary["live_datasets"]) == set(ALLOWED_DATASETS)
        assert summary["pairs"] == summary["distinct_live_samples"] * 4


def test_pair_plan_never_reuses_a_record_for_both_roles():
    for name in ("pair_manifest_train.parquet", "pair_manifest_validation.parquet"):
        for row in _pairs(name):
            assert row["live_source_record_id"] != row["spoof_source_record_id"]
            assert row["different_subject_rule"] in ("enforced", "not_applicable")


def test_pair_ids_are_deterministic_and_unique():
    rows = _pairs("pair_manifest_train.parquet") + _pairs("pair_manifest_validation.parquet")
    ids = [row["pair_id"] for row in rows]
    assert len(set(ids)) == len(ids)
    assert all(row["pair_id"].startswith("gpatpair_") and len(row["pair_id"]) == len("gpatpair_") + 20 for row in rows)


def test_pair_plan_lock_records_identities_and_seed():
    lock = json.loads((PAIRS / "PAIR_PLAN_LOCK.json").read_text(encoding="utf-8"))
    assert lock["seed"] == PAIR_PLAN_SEED == 20260806
    assert lock["train_pairs"] == 896 and lock["validation_pairs"] == 224
    assert lock["recipe_bank_identity"] == "fa989938cafdc4887518cc45c35d559d00278358439dc68c2486da10309210cb"
    assert lock["package_identity"].endswith("9dc6")
    assert set(lock["record_set_hashes"]) == {"live_train", "live_validation", "spoof_train", "spoof_validation"}
    assert lock["attack_family_balance"] == "unavailable"
    assert len(lock["pair_plan_identity_sha256"]) == 64


def test_pair_manifests_carry_no_raw_path_or_target_token():
    text = json.dumps(_pairs("pair_manifest_train.parquet") + _pairs("pair_manifest_validation.parquet"))
    for token in ("siw", "target_test", "source_dev", ".jpg", ".npz", "images/", "priors/", "D:/", "/home/"):
        assert token not in text, token


# --- DWT ---------------------------------------------------------------------

def test_dwt_band_shapes_and_order():
    ll, lh, hl, hh = haar_dwt2(torch.rand(2, 3, 224, 224))
    for band in (ll, lh, hl, hh): assert tuple(band.shape) == (2, 3, 112, 112)
    high = pack_high(lh, hl, hh)
    assert tuple(high.shape) == (2, 9, 112, 112) and BAND_ORDER == ("LH", "HL", "HH")
    again = unpack_high(high)
    assert all(torch.equal(a, b) for a, b in zip(again, (lh, hl, hh)))
    assert DWT_CONVENTION == "orthonormal_haar_v1"


def test_dwt_idwt_reconstruction_is_within_tolerance():
    torch.manual_seed(11)
    for shape in ((1, 3, 8, 8), (2, 3, 224, 224), (3, 9, 112, 112)):
        error = reconstruction_error(torch.rand(*shape))
        assert error <= DWT_RECONSTRUCTION_TOLERANCE_FP32, (shape, error)


def test_dwt_rejects_odd_or_wrong_rank_inputs():
    with pytest.raises(DWTError, match="even"): haar_dwt2(torch.rand(1, 3, 7, 8))
    with pytest.raises(DWTError, match=r"\[B,C,H,W\]"): haar_dwt2(torch.rand(3, 8, 8))
    ll = torch.rand(1, 3, 4, 4)
    with pytest.raises(DWTError, match="shape"): haar_idwt2(ll, ll, ll, torch.rand(1, 3, 8, 8))


def test_dwt_is_differentiable_with_finite_gradients():
    image = torch.rand(2, 3, 32, 32, requires_grad=True)
    ll, high = dwt_bands(image)
    idwt_bands(ll, high).square().sum().backward()
    assert image.grad is not None and bool(torch.isfinite(image.grad).all())


def test_dwt_is_orthonormal_and_energy_preserving():
    torch.manual_seed(5)
    image = torch.rand(2, 3, 64, 64)
    ll, high = dwt_bands(image)
    energy = float((ll ** 2).sum() + (high ** 2).sum())
    assert energy == pytest.approx(float((image ** 2).sum()), rel=1e-5)


# --- GPAT model --------------------------------------------------------------

def test_gpat_forward_shapes_and_bounded_delta():
    model = build_gpat_model({"model": {}})
    batch = _batch()
    output = model.forward_batch(batch)
    assert tuple(output.synthetic_image.shape) == (2, 3, 224, 224)
    assert tuple(output.delta_high.shape) == (2, 9, 112, 112)
    assert tuple(output.artifact_map.shape) == (2, 1, 224, 224)
    assert tuple(output.artifact_latent.shape) == (2, 128) and tuple(output.recipe_latent.shape) == (2, 64)
    assert float(output.delta_high.detach().abs().max()) <= model.max_high_frequency_delta + 1e-6
    amap = output.artifact_map.detach()
    assert float(amap.min()) >= 0.0 and float(amap.max()) <= 1.0


def test_gpat_ll_is_hard_locked_and_delta_ll_does_not_exist():
    model = build_gpat_model({"model": {}})
    batch = _batch()
    output = model.forward_batch(batch)
    assert output.ll_invariant_error() <= LL_INVARIANT_TOLERANCE
    assert torch.equal(output.live_bands["LL"], dwt_bands(batch.live_image)[0])
    for name in FORBIDDEN_OUTPUT_FIELDS:
        assert not hasattr(output, name)
    assert model.architecture_payload()["delta_ll_enabled"] is False
    source = (ROOT / "src" / "prism_fas" / "synthesis" / "gpat_model.py").read_text(encoding="utf-8")
    assert "delta_ll_head" not in source and "delta_LL" not in source


def test_gpat_output_outside_the_support_mask_is_bit_identical():
    model = build_gpat_model({"model": {}})
    batch = _batch()
    output = model.forward_batch(batch)
    outside = batch.target_support_mask < 0.5
    difference = (output.synthetic_image - batch.live_image).detach().abs()
    assert float(difference.masked_select(outside.expand_as(difference)).max()) == 0.0
    assert output.outside_mask_error(batch.live_image) == 0.0
    output.validate(batch.live_image)


def test_gpat_is_deterministic_and_conditioning_sensitive():
    torch.manual_seed(0); model = build_gpat_model({"model": {}}).eval()
    batch = _batch()
    with torch.no_grad():
        first = model.forward_batch(batch).synthetic_image
        second = model.forward_batch(batch).synthetic_image
        altered = model(batch.live_image, batch.source_spoof_image, torch.rand(2, 41),
                        batch.target_support_mask, batch.source_style_mask).synthetic_image
    assert torch.equal(first, second)
    assert not torch.equal(first, altered)


def test_gpat_rejects_an_empty_or_non_binary_mask():
    model = build_gpat_model({"model": {}})
    batch = _batch()
    with pytest.raises(GPATContractError, match="zero support"):
        model(batch.live_image, batch.source_spoof_image, batch.recipe_conditioning,
              torch.zeros(2, 1, 224, 224), batch.source_style_mask)
    with pytest.raises(GPATContractError, match="binary"):
        check_mask(torch.full((2, 1, 224, 224), 0.5), "m")


def test_downsampled_mask_is_binary_and_preserves_thin_regions():
    mask = torch.zeros(1, 1, 224, 224); mask[:, :, 100:101, 40:180] = 1.0
    half = downsample_mask(mask)
    assert tuple(half.shape) == (1, 1, 112, 112)
    assert set(torch.unique(half).tolist()) <= {0.0, 1.0} and float(half.sum()) > 0


def test_gpat_amp_forward_and_backward_are_finite_on_cpu():
    model = build_gpat_model({"model": {}})
    batch = _batch()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model.forward_batch(batch)
    loss = output.delta_high.float().abs().mean()
    loss.backward()
    assert bool(torch.isfinite(loss))
    assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)


def test_architecture_hash_changes_with_architecture():
    base = build_gpat_model({"model": {}})
    assert base.architecture_hash() == build_gpat_model({"model": {}}).architecture_hash()
    assert base.architecture_hash() != build_gpat_model({"model": {"film_blocks": 3}}).architecture_hash()
    assert base.parameter_count() > 0


# --- losses ------------------------------------------------------------------

def test_masked_channel_stats_matches_a_hand_computed_fixture():
    bands = torch.zeros(1, 2, 4, 4); bands[0, 0, :2, :2] = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
    mask = torch.zeros(1, 1, 4, 4); mask[0, 0, :2, :2] = 1.0
    mean, std = masked_channel_stats(bands, mask)
    assert float(mean[0, 0]) == pytest.approx(4.0)
    assert float(std[0, 0]) == pytest.approx(((9 + 1 + 1 + 9) / 4) ** 0.5, abs=1e-4)
    assert float(mean[0, 1]) == pytest.approx(0.0)


def test_style_loss_is_zero_for_identical_masked_content():
    torch.manual_seed(1)
    image = torch.rand(1, 3, 32, 32)
    mask = torch.ones(1, 1, 32, 32)
    assert float(style_loss(image, image, mask, mask)) == pytest.approx(0.0, abs=1e-6)


def test_identity_loss_and_stopgrad_on_the_live_embedding():
    generated = torch.nn.functional.normalize(torch.rand(2, 512), dim=1).requires_grad_(True)
    live = torch.nn.functional.normalize(torch.rand(2, 512), dim=1).requires_grad_(True)
    loss = identity_loss(generated, live)
    loss.backward()
    assert live.grad is None and generated.grad is not None
    assert float(identity_loss(generated.detach(), generated.detach())) == pytest.approx(0.0, abs=1e-6)


def test_map_strength_residual_and_tv_fixtures():
    mask = torch.zeros(1, 1, 4, 4); mask[0, 0, :2, :2] = 1.0
    amap = mask * 0.4
    assert float(artifact_map_loss(amap, mask, torch.tensor([0.4]))) == pytest.approx(0.0, abs=1e-7)
    assert float(strength_loss(amap, mask, torch.tensor([0.4]))) == pytest.approx(0.0, abs=1e-7)
    assert float(strength_loss(amap, mask, torch.tensor([0.1]))) == pytest.approx(0.3, abs=1e-6)
    assert float(residual_loss(torch.full((1, 9, 4, 4), -0.2))) == pytest.approx(0.2, abs=1e-7)
    assert float(total_variation_loss(torch.zeros(1, 9, 4, 4), torch.ones(1, 9, 4, 4))) == pytest.approx(0.0)


def test_zero_mask_and_empty_style_are_rejected_not_zero_weighted():
    with pytest.raises(GPATLossError, match="no valid mask pixels"):
        masked_channel_stats(torch.rand(1, 9, 8, 8), torch.zeros(1, 1, 8, 8))
    with pytest.raises(GPATLossError, match="empty support"):
        strength_loss(torch.rand(1, 1, 4, 4), torch.zeros(1, 1, 4, 4), torch.tensor([0.3]))


def test_total_loss_matches_the_declared_weighted_sum():
    model = build_gpat_model({"model": {}})
    batch = _batch()
    output = model.forward_batch(batch)
    embedding = torch.nn.functional.normalize(torch.rand(2, 512), dim=1)
    result = compute_losses(output, batch, embedding)
    expected = sum(DEFAULT_WEIGHTS[name] * float(value.detach()) for name, value in result.components.items())
    assert float(result.total) == pytest.approx(expected, rel=1e-6)
    assert result.weights == DEFAULT_WEIGHTS == {"style": 1.0, "identity": 0.5, "map": 0.5,
                                                 "strength": 0.25, "total_variation": 0.02, "residual": 0.01}
    assert set(result.components) == set(DEFAULT_WEIGHTS)
    manifest = loss_manifest()
    assert manifest["adversarial"] is False and manifest["detector_classification"] is False


def test_backward_through_the_total_loss_is_finite():
    model = build_gpat_model({"model": {}})
    batch = _batch()
    output = model.forward_batch(batch)
    result = compute_losses(output, batch, torch.nn.functional.normalize(torch.rand(2, 512), dim=1))
    result.total.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
    assert all(bool(torch.isfinite(v)) for v in result.components.values())


# --- pinned quality models ---------------------------------------------------

def test_pinned_quality_model_shas_are_frozen_in_code():
    assert PINNED["parsing"]["sha256"] == "327a755849ba64d336fb96589ff87b27e84a12be1ecf8bcfaa503d66f803286d"
    assert PINNED["parsing"]["revision"] == "fd12148d0b19"
    assert PINNED["identity"]["sha256"] == "43bd2d570584d95d4a17ce81f26449034c45dbeed750afcab651872abc0e1496"
    assert PINNED["identity"]["revision"] == "60a65befbcf7"
    assert PINNED["detector"]["sha256"] == "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"
    assert PINNED["detector"]["revision"] is None and "no upstream revision" in PINNED["detector"]["revision_note"]


def test_missing_pinned_weight_raises_and_tests_never_download(tmp_path):
    with pytest.raises(QualityModelError, match="missing"):
        resolve_weight(tmp_path, "identity")
    source = (ROOT / "src" / "prism_fas" / "synthesis" / "quality_models.py").read_text(encoding="utf-8")
    for marker in ("hf_hub_download", "snapshot_download", "requests", "urllib.request"):
        assert marker not in source


# --- real local smoke report -------------------------------------------------

def test_local_cpu_smoke_report_contract():
    smoke = _report("local_gpat_smoke.json")
    assert smoke["passed"] and smoke["device"] == "cpu" and smoke["precision"] == "fp32"
    assert smoke["modal_used"] is False and smoke["gpu_used"] is False and smoke["ssh_used"] is False
    assert smoke["dwt"]["passed"] and smoke["dwt"]["reconstruction_max_abs_error"] <= 1e-6
    assert smoke["invariants"]["ll_passed"] and smoke["invariants"]["outside_mask_exactly_zero"]
    assert smoke["invariants"]["outside_mask_max_abs_error"] == 0.0
    assert smoke["backward"]["finite"] and smoke["backward"]["parameters_with_gradient"] > 0
    assert smoke["backward"]["identity_model_received_gradients"] is False
    assert smoke["model"]["delta_ll_enabled"] is False
    assert smoke["recipe_bank_identity"] == "fa989938cafdc4887518cc45c35d559d00278358439dc68c2486da10309210cb"
    assert smoke["package_identity"].endswith("9dc6")
    isolation = smoke["source_isolation"]
    assert isolation["source_dev_opened"] is False and isolation["target_test_opened"] is False
    assert isolation["target_label_artifact_opened"] is False and isolation["raw_dataset_path_opened"] is False
    assert isolation["manifests_opened"] == [f"manifests/{SOURCE_SPLIT}.parquet"]
    assert set(isolation["path_prefixes"]) <= {"manifests", "images", "priors"}
    verified = smoke["quality_models"]["models"]["identity"]
    assert verified["sha256_matches_pin"] and verified["revision"] == "60a65befbcf7"


def test_source_only_audit_refuses_forbidden_artifacts():
    from prism_fas.synthesis.m8_pipeline import PipelineError, SourceOnlyAudit
    audit = SourceOnlyAudit()
    for bad in ("manifests/source_dev.parquet", "manifests/target_test_features.parquet", "shards/target_test-00000.tar"):
        with pytest.raises(PipelineError):
            audit.record(bad)
    assert audit.record("manifests/source_train.parquet") == "manifests/source_train.parquet"


def test_gpat_config_is_source_only_and_carries_no_absolute_path():
    from prism_fas.synthesis.m8_pipeline import load_gpat_config
    config = load_gpat_config(CONFIG)
    assert config["seed"] == 20260806 and config["batch_size"] == 16 and config["epochs"] == 15
    assert config["data"]["package_split"] == "source_train"
    assert config["data"]["forbidden_splits"] == ["source_dev", "target_test"]
    assert config["loss"] == {"style": 1.0, "identity": 0.5, "map": 0.5, "strength": 0.25,
                              "total_variation": 0.02, "residual": 0.01}
    assert config["model"]["max_high_frequency_delta"] == 0.15
    assert config["identity_model"]["trainable"] is False
    text = CONFIG.read_text(encoding="utf-8")
    for marker in ("D:/", "C:\\", "/home/", "/Users/"):
        assert marker not in text


# --- checkpoint strictness ---------------------------------------------------

def _identity(**overrides):
    base = {"package_identity": "p" * 64, "recipe_bank_identity": "b" * 64, "pair_plan_identity": "q" * 64,
            "config_hash": "c" * 64, "architecture_hash": "a" * 64, "adaface_weight_sha256": "d" * 64}
    base.update(overrides)
    return base


def _save(tmp_path, model, identity):
    from prism_fas.synthesis.gpat_checkpoint import save_checkpoint
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    return save_checkpoint(tmp_path / "last.pt", model=model, optimizer=optimizer, scheduler=None, scaler=None,
                           epoch=3, global_step=42, best_metrics={"validation_total_loss": 0.5},
                           identity=identity, history=[{"epoch": 3}],
                           record_set_hashes={"train_live_source_record_id": "r" * 64}, git_commit="deadbeef")


def test_checkpoint_round_trip_restores_step_and_rng(tmp_path):
    from prism_fas.synthesis.gpat_checkpoint import apply_checkpoint, load_checkpoint
    model = build_gpat_model({"model": {}})
    digest = _save(tmp_path, model, _identity())
    assert len(digest) == 64
    payload = load_checkpoint(tmp_path / "last.pt", expected_identity=_identity())
    restored = build_gpat_model({"model": {}})
    state = apply_checkpoint(payload, model=restored, optimizer=None)
    assert state["epoch"] == 3 and state["global_step"] == 42
    assert payload["rng_state"]["torch_cpu"] is not None
    for (name, a), (_, b) in zip(model.state_dict().items(), restored.state_dict().items()):
        assert torch.equal(a, b), name


@pytest.mark.parametrize("field", ["package_identity", "recipe_bank_identity", "pair_plan_identity",
                                   "config_hash", "architecture_hash", "adaface_weight_sha256"])
def test_resume_rejects_every_identity_mismatch(tmp_path, field):
    from prism_fas.synthesis.gpat_checkpoint import CheckpointError, load_checkpoint
    _save(tmp_path, build_gpat_model({"model": {}}), _identity())
    with pytest.raises(CheckpointError, match="identity mismatch"):
        load_checkpoint(tmp_path / "last.pt", expected_identity=_identity(**{field: "z" * 64}))
    assert load_checkpoint(tmp_path / "last.pt", expected_identity=_identity())["global_step"] == 42


def test_batch_slices_are_deterministic_and_cover_every_pair():
    from prism_fas.synthesis.gpat_trainer import batch_slices
    first = batch_slices(100, 16, seed=20260806, epoch=0, shuffle=True)
    assert first == batch_slices(100, 16, seed=20260806, epoch=0, shuffle=True)
    assert first != batch_slices(100, 16, seed=20260806, epoch=1, shuffle=True)
    assert sorted(index for group in first for index in group) == list(range(100))


# --- real Modal artifacts ----------------------------------------------------

def test_modal_gpat_smoke_report_contract():
    smoke = _report("modal_gpat_smoke.json")
    assert smoke["gpu"]["gpu_name"] == "NVIDIA L4" and smoke["cuda_available"] is True
    assert smoke["device"] == "cuda" and smoke["amp"] is True
    assert smoke["steps_first"] == 5 and smoke["steps_after_resume"] >= 6
    assert smoke["resume_continued"] is True and smoke["resumed_from_step"] == 5
    assert smoke["losses_finite"] and len(smoke["losses"]) == 6
    assert smoke["ll_invariant_max"] <= 1e-5 and smoke["outside_mask_max"] == 0.0
    assert smoke["package_identity"].endswith("9dc6")
    assert smoke["recipe_bank_identity"] == "fa989938cafdc4887518cc45c35d559d00278358439dc68c2486da10309210cb"
    assert smoke["source_isolation"]["source_dev_opened"] is False
    assert smoke["source_isolation"]["target_test_opened"] is False


def test_real_full_gpat_training_report_contract():
    run = _report("gpat_training.json")
    assert run["run_id"] == "gpat_m8_seed20260806" and run["device"] == "cuda" and run["amp"] is True
    assert run["gpu"]["gpu_name"] == "NVIDIA L4"
    assert run["train_pairs"] == 896 and run["validation_pairs"] == 224
    assert run["epochs_run"] >= 5 and run["epochs_run"] <= run["epochs_configured"] == 15
    assert run["stop_reason"].startswith("early_stopped") or run["stop_reason"] == "completed_all_epochs"
    assert run["checkpoints"]["best_sha256"] and run["checkpoints"]["last_sha256"]
    best = run["best"]
    assert best["validation_total_loss"] < float("inf") and best["epoch"] >= 0
    for entry in run["history"]:
        for key in ("train_total", "validation_total_loss", "validation_identity_cosine"):
            assert entry[key] == entry[key], f"NaN in {key}"        # NaN != NaN
    assert run["best"]["validation_total_loss"] == min(e["validation_total_loss"] for e in run["history"])
    assert run["identity"]["pair_plan_identity"] == json.loads(
        (PAIRS / "PAIR_PLAN_LOCK.json").read_text(encoding="utf-8"))["pair_plan_identity_sha256"]
    assert run["source_isolation"]["manifests_opened"] == ["manifests/source_train.parquet"]
    assert run["loss_manifest"]["adversarial"] is False


def test_pair_plan_identity_excludes_non_portable_fields():
    from prism_fas.synthesis.pair_plan import IDENTITY_EXCLUDED_FIELDS
    lock = json.loads((PAIRS / "PAIR_PLAN_LOCK.json").read_text(encoding="utf-8"))
    assert "config_hash" in IDENTITY_EXCLUDED_FIELDS
    # parquet bytes differ between pyarrow writer versions, so they cannot be
    # part of a portable identity; the logical rows are hashed instead.
    assert "pair_manifest_sha256" in IDENTITY_EXCLUDED_FIELDS
    assert set(lock["pair_rows_sha256"]) == {"train", "validation"}
    assert lock["identity_excluded_fields"] == list(IDENTITY_EXCLUDED_FIELDS)


def test_modal_wrapper_does_not_leak_into_core_modules():
    import ast
    for path in (ROOT / "src" / "prism_fas" / "synthesis").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            assert not any(name.split(".")[0] == "modal" for name in names), f"{path.name} imports modal"
    wrapper = (ROOT / "modal_m8.py").read_text(encoding="utf-8")
    assert "prism-fas-b-m8" in wrapper and '"L4"' in wrapper
    for banned in ("H100", "H200", "B200"):
        assert banned not in wrapper


# --- fingerprint -------------------------------------------------------------

def test_fingerprint_vector_is_24_dimensional_and_finite():
    from prism_fas.synthesis.fingerprint import FEATURE_NAMES, FINGERPRINT_DIM, fingerprint_features
    rng = np.random.Generator(np.random.PCG64(4))
    vector = fingerprint_features(rng.random((3, 64, 64)).astype(np.float32))
    assert FINGERPRINT_DIM == 24 and vector.shape == (24,) and vector.dtype == np.float32
    assert bool(np.isfinite(vector).all()) and len(FEATURE_NAMES) == 24
    assert FEATURE_NAMES[:9] == tuple(f"high_abs_mean_{i}" for i in range(9))
    assert FEATURE_NAMES[18:] == ("edge_energy_r", "edge_energy_g", "edge_energy_b",
                                  "laplacian_variance", "mean_saturation", "channel_balance_magnitude")


def test_fingerprint_is_deterministic_and_rejects_bad_shapes():
    from prism_fas.synthesis.fingerprint import FingerprintError, fingerprint_features
    image = np.random.Generator(np.random.PCG64(9)).random((3, 32, 32)).astype(np.float32)
    assert np.array_equal(fingerprint_features(image), fingerprint_features(image))
    with pytest.raises(FingerprintError):
        fingerprint_features(np.zeros((4, 32, 32), dtype=np.float32))


def test_robust_reference_uses_median_and_mad_with_epsilon_floor():
    from prism_fas.synthesis.fingerprint import MAD_EPSILON, MAD_SCALE, robust_reference, score_against
    values = np.tile(np.arange(24, dtype=np.float64), (11, 1))
    values[0] += 1000.0
    reference = robust_reference(values)
    assert reference["median"] == [float(v) for v in np.arange(24)]
    assert all(scale >= MAD_EPSILON for scale in reference["scale"])
    assert MAD_SCALE == 1.4826
    assert score_against(np.arange(24, dtype=np.float64), reference) == pytest.approx(0.0, abs=1e-9)


def test_fingerprint_score_picks_the_closest_domain():
    from prism_fas.synthesis.fingerprint import fingerprint_score, robust_reference
    near = robust_reference(np.tile(np.zeros(24), (5, 1)) + np.arange(5)[:, None] * 1e-3)
    far = robust_reference(np.tile(np.full(24, 50.0), (5, 1)) + np.arange(5)[:, None] * 1e-3)
    probe = np.zeros(24)
    assert fingerprint_score(probe, {"casia_fasd": near, "msu_mfsd": far}) == pytest.approx(
        min(fingerprint_score(probe, {"casia_fasd": near}), fingerprint_score(probe, {"msu_mfsd": far})))


def test_leave_one_record_out_excludes_the_own_record():
    from prism_fas.synthesis.fingerprint import fit_fingerprint_reference, leave_one_record_out_scores
    rng = np.random.Generator(np.random.PCG64(1))
    vectors = rng.normal(size=(24, 24))
    domains = ["casia_fasd"] * 12 + ["msu_mfsd"] * 12
    records = [f"r{index // 3}" for index in range(24)]
    scores = leave_one_record_out_scores(vectors, domains, records)
    assert len(scores) == 24 and all(np.isfinite(scores))
    fitted = fit_fingerprint_reference(vectors, domains, records)
    assert fitted["dimension"] == 24 and fitted["tau_fp_percentile"] == 99.0
    assert fitted["trainable_probe"] is False and fitted["used_generated_candidates"] is False
    assert fitted["used_source_dev"] is False and fitted["used_target"] is False
    assert fitted["tau_fp"] == pytest.approx(float(np.percentile(scores, 99.0)))


# --- quality gate ------------------------------------------------------------

def _thresholds(**overrides):
    from prism_fas.synthesis.quality_gate import Thresholds
    base = {"tau_fd": 0.5, "tau_id": 0.99, "tau_lm": 0.05, "tau_parse": 0.8, "tau_out": 0.0, "tau_fp": 5.0}
    base.update(overrides)
    return Thresholds(**base)


def _metrics(**overrides):
    base = {"face_detection_score": 0.9, "identity_cosine": 0.999, "landmark_nme": 0.01,
            "outside_mask_parsing_dice": 0.95, "outside_mask_max_error": 0.0,
            "measured_artifact_strength": 0.3, "requested_artifact_strength": 0.3,
            "fingerprint_score": 1.0, "support_overlap": 0.99}
    base.update(overrides)
    return base


def test_dynamic_strength_bounds():
    from prism_fas.synthesis.quality_gate import strength_bounds
    assert strength_bounds(0.4) == (pytest.approx(0.1), pytest.approx(0.5))
    assert strength_bounds(0.02) == (pytest.approx(0.01), pytest.approx(0.035))
    assert strength_bounds(0.8)[1] == pytest.approx(0.50)


def test_all_hard_gates_must_pass_for_acceptance():
    from prism_fas.synthesis.quality_gate import HARD_GATES, evaluate
    good = evaluate(_metrics(), _thresholds())
    assert good["accepted"] and good["failed_gates"] == [] and set(good["gates"]) == set(HARD_GATES)
    failing = {
        "face_detection": {"face_detection_score": 0.4}, "identity": {"identity_cosine": 0.5},
        "landmark": {"landmark_nme": 0.9}, "parsing_dice": {"outside_mask_parsing_dice": 0.1},
        "outside_mask": {"outside_mask_max_error": 1e-9}, "artifact_strength": {"measured_artifact_strength": 0.9},
        "fingerprint": {"fingerprint_score": 99.0}, "support_overlap": {"support_overlap": 0.5}}
    for gate, override in failing.items():
        result = evaluate(_metrics(**override), _thresholds())
        assert not result["accepted"] and gate in result["failed_gates"], gate


def test_q_formula_range_and_that_it_is_not_a_label():
    from prism_fas.synthesis.quality_gate import QUALITY_COMPONENTS, RECIPE_MATCH, evaluate
    result = evaluate(_metrics(), _thresholds())
    components = [result["quality_components"][name] for name in QUALITY_COMPONENTS]
    expected = float(np.exp(np.mean([np.log(max(value, 1e-6)) for value in components])))
    assert result["q"] == pytest.approx(expected, rel=1e-5)
    assert 0.0 <= result["q"] <= 1.0 and np.isfinite(result["q"])
    assert result["recipe_match"] == RECIPE_MATCH == "not_applicable"
    # A candidate sitting just above every threshold earns a low q but is still
    # accepted: q is a sample weight, not a rejection criterion.
    weak = evaluate(_metrics(identity_cosine=0.9901, outside_mask_parsing_dice=0.801), _thresholds())
    assert weak["accepted"] and weak["failed_gates"] == []
    assert weak["q"] < result["q"] / 2.0


def test_q_strength_is_triangular_around_the_requested_value():
    from prism_fas.synthesis.quality_gate import triangular_strength_score
    assert triangular_strength_score(0.3, 0.3) == pytest.approx(1.0)
    assert triangular_strength_score(0.075, 0.3) == pytest.approx(0.0, abs=1e-9)
    assert triangular_strength_score(0.525, 0.3) == 0.0
    assert 0.0 < triangular_strength_score(0.2, 0.3) < 1.0


def test_landmark_nme_and_parsing_dice_fixtures():
    from prism_fas.synthesis.quality_gate import landmark_nme, parsing_dice, support_overlap
    reference = np.asarray([[0.0, 0.0], [10.0, 0.0], [5.0, 5.0], [2.0, 9.0], [8.0, 9.0]], dtype=np.float32)
    assert landmark_nme(reference, reference) == pytest.approx(0.0)
    shifted = reference + np.asarray([1.0, 0.0], dtype=np.float32)
    assert landmark_nme(shifted, reference) == pytest.approx(1.0 / 10.0)
    a = np.zeros((8, 8), dtype=np.uint8); b = np.zeros((8, 8), dtype=np.uint8)
    mask = np.ones((8, 8), dtype=bool)
    assert parsing_dice(a, b, mask) == pytest.approx(1.0)
    b[:4] = 1
    assert 0.0 <= parsing_dice(a, b, mask) < 1.0
    exact = np.zeros((8, 8), dtype=bool); exact[2:6, 2:6] = True
    assert support_overlap(exact, exact) == pytest.approx(1.0)
    assert support_overlap(np.zeros((8, 8), dtype=bool), exact) == 0.0


def test_thresholds_hash_is_stable_and_round_trips():
    from prism_fas.synthesis.quality_gate import Thresholds
    first = _thresholds()
    assert first.sha256() == _thresholds().sha256() and len(first.sha256()) == 64
    assert first.sha256() != _thresholds(tau_id=0.98).sha256()
    assert Thresholds.from_dict(first.as_dict()).sha256() == first.sha256()


# --- benign perturbations ----------------------------------------------------

def test_benign_perturbations_are_deterministic_and_mild():
    from prism_fas.synthesis.quality_calibration import BENIGN_NOISE_STD, BENIGN_VARIANTS, benign_variant
    image = np.full((3, 32, 32), 0.5, dtype=np.float32)
    assert len(BENIGN_VARIANTS) == 4 and BENIGN_NOISE_STD <= 0.002
    for variant in BENIGN_VARIANTS:
        first = benign_variant(image, variant, sample_id="s1", noise_std=BENIGN_NOISE_STD)
        assert np.array_equal(first, benign_variant(image, variant, sample_id="s1", noise_std=BENIGN_NOISE_STD))
        assert not np.array_equal(first, benign_variant(image, variant, sample_id="s2", noise_std=BENIGN_NOISE_STD))
        assert float(np.abs(first - image).max()) < 0.05
        assert first.dtype == np.float32 and first.min() >= 0.0 and first.max() <= 1.0
    assert {variant["name"] for variant in BENIGN_VARIANTS} == {
        "brightness_098", "brightness_102", "contrast_098", "contrast_102"}


def test_calibration_config_is_source_train_only():
    from prism_fas.synthesis.quality_calibration import load_quality_config
    config = load_quality_config(ROOT / "configs" / "synthesis" / "quality_gate_m8.yaml")
    assert config["calibration_population"]["split"] == "source_train"
    assert config["calibration_population"]["live_samples"] == 280
    assert config["calibration_population"]["spoof_samples"] == 1160
    assert config["benign"]["jpeg_q95"] is False
    assert config["thresholds"]["tau_fd"]["fitted"] is False
    assert config["quality_weight"]["is_label"] is False
    assert config["recipe_match"]["status"] == "not_applicable"
    assert config["fingerprint"]["trainable_probe"] is False


# --- real calibration artifact ----------------------------------------------

def test_real_quality_calibration_report_contract():
    calibration = _report("quality_calibration.json")
    assert calibration["populations"]["split"] == "source_train"
    assert calibration["populations"]["live"] == 280 and calibration["populations"]["spoof"] == 1160
    assert calibration["populations"]["benign"] == 1120
    assert calibration["populations"]["live_per_dataset"] == {"casia_fasd": 160, "msu_mfsd": 120}
    assert calibration["populations"]["spoof_per_dataset"] == {"casia_fasd": 800, "msu_mfsd": 360}
    thresholds = calibration["thresholds"]
    assert thresholds["tau_fd"] == 0.5 and thresholds["tau_out"] == 0.0
    assert 0.0 < thresholds["tau_id"] <= 1.0 and thresholds["tau_lm"] > 0.0
    assert 0.0 < thresholds["tau_parse"] <= 1.0 and thresholds["tau_fp"] > 0.0
    assert calibration["fingerprint"]["dimension"] == 24
    assert calibration["fingerprint"]["samples"] == 1160
    assert calibration["fingerprint"]["used_generated_candidates"] is False
    assert calibration["used_source_dev"] is False and calibration["used_target"] is False
    assert calibration["benign_policy"]["jpeg_q95_used"] is False
    assert calibration["benign_policy"]["uses_m7_spoof_operators"] is False
    assert calibration["source_isolation"]["manifests_opened"] == ["manifests/source_train.parquet"]
    assert calibration["source_isolation"]["source_dev_opened"] is False
    assert calibration["source_isolation"]["target_test_opened"] is False
    for role in ("identity", "parsing", "detector"):
        assert calibration["quality_models"]["models"][role]["sha256_matches_pin"] is True
    assert len(calibration["threshold_sha256"]) == 64 and len(calibration["calibration_config_sha256"]) == 64


# --- candidate plan ----------------------------------------------------------

GPAT_BEST_SHA = "2047cdb513767010cfdf368c6f53a3664922451c56e1e837ec59cb96918a5b63"
CANDIDATE_PLAN_IDENTITY = "b167c169dcb92426c0dc2ee96a80eb69f4645fbf887360a1b67abfc8890f40b8"
THRESHOLD_SHA = "4798a392243c85f89b37a14dc51958637d4ae177756bf88f693804f065c4c297"
FINGERPRINT_REFERENCE_SHA = "c5c09cfa26819e125eafb4640eec6ab02eec5419ae6a83bad9a293ae4c4ebb39"
PACKAGE_IDENTITY = "b1cf29b69a165ed5d9e074fc8127c17fbf057723edf9e272048ec3a564eb9dc6"
TAU_ID_V2_THRESHOLD_SHA = "a3f20e5e46641deeac0f1110f6783869ed88ee01786baa8d006bf5ca8d159754"
TAU_ID_V2 = 0.547440037939055


def _plan():
    from prism_fas.synthesis.candidate_plan import load_candidate_plan
    path = REPORTS / "candidate_plan.parquet"
    if not path.is_file(): pytest.skip("candidate plan missing; run the plan-bank command")
    return load_candidate_plan(path)


def test_candidate_plan_has_exactly_1120_rows_split_560_560():
    from prism_fas.synthesis.candidate_plan import EXPECTED_PER_ROUTE, EXPECTED_TOTAL
    rows = _plan()
    assert len(rows) == EXPECTED_TOTAL == 1120
    assert sum(1 for row in rows if row["route"] == "physics") == EXPECTED_PER_ROUTE == 560
    assert sum(1 for row in rows if row["route"] == "gpat") == 560
    assert len({row["live_target_sample_id"] for row in rows}) == 280


def test_candidate_ids_are_unique_deterministic_and_carry_no_private_token():
    from prism_fas.synthesis.candidate_plan import candidate_id
    rows = _plan()
    ids = [row["synthetic_id"] for row in rows]
    assert len(set(ids)) == 1120
    assert all(value.startswith("syn_") and len(value) == len("syn_") + 24 for value in ids)
    text = json.dumps(rows)
    for token in ("siw", "target_test", "source_dev", ".jpg", ".npz", "images/", "priors/", "D:/", "subject"):
        assert token not in text, token
    row = rows[0]
    assert candidate_id(package_identity=row["package_identity"], bank_identity=row["recipe_bank_identity"],
                        route=row["route"], live_sample_id=row["live_target_sample_id"],
                        spoof_sample_id=row["spoof_source_sample_id"], recipe_id=row["recipe_id"],
                        seed=row["candidate_seed"], generator_binding=row["generator_binding"]) == row["synthetic_id"]


def test_candidate_id_binds_the_exact_gpat_checkpoint_sha():
    from prism_fas.synthesis.candidate_plan import candidate_id
    rows = _plan()
    gpat = [row for row in rows if row["route"] == "gpat"]
    physics = [row for row in rows if row["route"] == "physics"]
    assert {row["gpat_checkpoint_sha256"] for row in gpat} == {GPAT_BEST_SHA}
    assert {row["generator_binding"] for row in gpat} == {GPAT_BEST_SHA}
    assert {row["physics_engine_version"] for row in physics} == {"m7-physics-v1"}
    assert all(row["gpat_checkpoint_sha256"] is None for row in physics)
    assert all(row["physics_engine_version"] is None for row in gpat)
    row = gpat[0]
    altered = candidate_id(package_identity=row["package_identity"], bank_identity=row["recipe_bank_identity"],
                           route=row["route"], live_sample_id=row["live_target_sample_id"],
                           spoof_sample_id=row["spoof_source_sample_id"], recipe_id=row["recipe_id"],
                           seed=row["candidate_seed"], generator_binding="0" * 64)
    assert altered != row["synthetic_id"]


def test_gpat_candidates_use_one_same_and_one_cross_domain_source():
    rows = [row for row in _plan() if row["route"] == "gpat"]
    assert sum(1 for row in rows if row["domain_relation"] == "same_domain") == 280
    assert sum(1 for row in rows if row["domain_relation"] == "cross_domain") == 280
    by_live: dict = {}
    for row in rows: by_live.setdefault(row["live_target_sample_id"], set()).add(row["domain_relation"])
    assert all(value == {"same_domain", "cross_domain"} for value in by_live.values())
    for row in rows:
        assert row["spoof_source_record_id"] != row["live_target_record_id"]
        assert (row["spoof_source_dataset"] == row["live_target_dataset"]) == (row["domain_relation"] == "same_domain")


def test_physics_candidates_carry_no_spoof_source():
    for row in [row for row in _plan() if row["route"] == "physics"]:
        assert row["spoof_source_sample_id"] is None and row["spoof_source_dataset"] is None
        assert row["domain_relation"] == "not_applicable"


def test_candidate_plan_lock_is_deterministic_and_identity_excludes_parquet_bytes():
    from prism_fas.synthesis.candidate_plan import IDENTITY_EXCLUDED_FIELDS, rows_digest
    path = REPORTS / "CANDIDATE_PLAN_LOCK.json"
    if not path.is_file(): pytest.skip("candidate plan lock missing")
    lock = json.loads(path.read_text(encoding="utf-8"))
    assert lock["candidate_count"] == 1120 and lock["seed"] == 20260806
    assert lock["route_counts"] == {"gpat": 560, "physics": 560}
    assert lock["gpat_checkpoint_sha256"] == GPAT_BEST_SHA
    assert lock["package_identity"].endswith("9dc6")
    assert lock["recipe_bank_identity"] == "fa989938cafdc4887518cc45c35d559d00278358439dc68c2486da10309210cb"
    assert lock["candidate_rows_sha256"] == rows_digest(_plan())
    assert "candidate_manifest_sha256" in IDENTITY_EXCLUDED_FIELDS
    assert lock["identity_excluded_fields"] == list(IDENTITY_EXCLUDED_FIELDS)
    assert len(lock["candidate_plan_identity_sha256"]) == 64


def test_m7_physics_config_was_not_modified_by_m8():
    text = (ROOT / "configs" / "synthesis" / "physics_m7.yaml").read_text(encoding="utf-8")
    assert "recipes_per_sample: 2" in text
    assert "candidate_recipes_per_live" not in text
    bank = (ROOT / "configs" / "synthesis" / "synthetic_bank_m8.yaml").read_text(encoding="utf-8")
    assert "candidate_recipes_per_live" in bank


# --- M8 discrete uint8 output ------------------------------------------------
from prism_fas.synthesis.synthetic_bank import (ARTIFACT_MAP_KEY, BANK_LOCK_SCHEMA_VERSION,  # noqa: E402
                                                CANDIDATE_RECORD_SCHEMA_VERSION, DISCRETE_CONVENTION,
                                                MANIFEST_SCHEMAS, SyntheticBankError, _sanitize,
                                                applied_strength_map, assemble_bank, check_operational_minimums,
                                                decode_npz, decode_png, encode_npz, encode_png,
                                                finalize_discrete, from_uint8, load_manifest,
                                                record_is_reusable, to_uint8)
from prism_fas.synthesis.synthetic_shards import (MEMBER_SUFFIXES, TAR_MTIME, build_shard_bytes,  # noqa: E402
                                                  build_shards, index_digest, load_shards_index,
                                                  shard_member_metadata, validate_shards, write_shards_index)

SIZE = 224


def _live(seed=7):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 256, size=(3, SIZE, SIZE)).astype(np.float32) / 255.0).astype(np.float32)


def _support(top=40, bottom=100, left=30, right=90):
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def test_uint8_conversion_is_round_half_up_and_clipped():
    values = np.zeros((3, 2, 2), dtype=np.float32)
    values[0] = np.asarray([[0.0, 1.0], [0.5 / 255.0, 1.5 / 255.0]], dtype=np.float32)
    out = to_uint8(values)
    assert out.dtype == np.uint8 and out.shape == (2, 2, 3)
    assert out[0, 0, 0] == 0 and out[0, 1, 0] == 255
    # an exact .5 rounds up, never to-even
    assert out[1, 0, 0] == 1 and out[1, 1, 0] == 2
    assert DISCRETE_CONVENTION == "round_half_up_clip_0_255"


def test_uint8_round_trip_is_exact_for_package_style_floats():
    original = _live()
    assert np.array_equal(to_uint8(from_uint8(to_uint8(original))), to_uint8(original))
    assert np.allclose(from_uint8(to_uint8(original)), original, atol=1e-6)


def test_finalize_composites_outside_the_requested_support():
    original, support = _live(), _support()
    generated = np.clip(original + 0.2, 0.0, 1.0).astype(np.float32)     # changed everywhere
    result = finalize_discrete(generated, original, support, np.full((1, SIZE, SIZE), 0.3, np.float32))
    outside = ~support
    assert np.array_equal(result.image_uint8[outside], result.original_uint8[outside])
    assert result.outside_mask_max_error == 0


def test_exact_mask_is_the_actual_changed_pixels_not_the_request():
    original, support = _live(), _support()
    generated = original.copy()
    generated[:, 50:60, 40:50] = np.clip(generated[:, 50:60, 40:50] + 0.5, 0, 1)   # a strict sub-region
    result = finalize_discrete(generated, original, support, np.full((1, SIZE, SIZE), 0.3, np.float32))
    assert 0 < result.exact_mask_pixels < int(support.sum())
    assert int((result.exact_edit_mask & ~support).sum()) == 0
    changed = np.any(result.image_uint8 != result.original_uint8, axis=2)
    assert np.array_equal(result.exact_edit_mask, changed)


def test_sub_quantization_change_yields_an_empty_exact_mask():
    """An artifact too weak to survive uint8 rounding is never silently counted."""
    from prism_fas.synthesis.quality_gate import support_overlap
    original, support = _live(), _support()
    generated = (original + np.float32(1.0 / 4096.0)).astype(np.float32)
    result = finalize_discrete(generated, original, support, np.zeros((1, SIZE, SIZE), np.float32))
    assert result.exact_mask_pixels == 0
    assert support_overlap(result.exact_edit_mask, support) == 0.0
    assert result.outside_mask_max_error == 0


def test_saved_png_reverifies_and_masks_hold_only_0_and_255():
    original, support = _live(), _support()
    generated = np.clip(original + 0.25, 0, 1).astype(np.float32)
    result = finalize_discrete(generated, original, support, np.full((1, SIZE, SIZE), 0.4, np.float32))
    decoded = decode_png(result.image_png)
    assert decoded.shape == (SIZE, SIZE, 3) and np.array_equal(decoded, result.image_uint8)
    mask = decode_png(result.mask_png)
    assert mask.shape == (SIZE, SIZE) and set(np.unique(mask).tolist()) <= {0, 255}
    assert np.array_equal(mask == 255, result.exact_edit_mask)


def test_artifact_map_is_float16_and_exactly_zero_outside_the_exact_mask():
    original, support = _live(), _support()
    generated = np.clip(original + 0.25, 0, 1).astype(np.float32)
    result = finalize_discrete(generated, original, support, np.full((1, SIZE, SIZE), 0.37, np.float32))
    array = decode_npz(result.artifact_map_npz, ARTIFACT_MAP_KEY)
    assert array.dtype == np.float16 and array.shape == (1, SIZE, SIZE)
    values = np.asarray(array, dtype=np.float32)
    assert np.isfinite(values).all() and values.min() >= 0.0 and values.max() <= 1.0
    assert float(np.abs(values[0][~result.exact_edit_mask]).max()) == 0.0


def test_npz_bytes_are_deterministic_and_load_without_pickle():
    array = np.linspace(0, 1, SIZE * SIZE, dtype=np.float16).reshape(1, SIZE, SIZE)
    first, second = encode_npz({ARTIFACT_MAP_KEY: array}), encode_npz({ARTIFACT_MAP_KEY: array})
    assert first == second                      # no wall clock stamped into the zip entries
    assert np.array_equal(decode_npz(first), array)


def test_finalize_rejects_a_non_finite_generated_image():
    original, support = _live(), _support()
    broken = original.copy(); broken[0, 0, 0] = np.nan
    with pytest.raises(SyntheticBankError):
        finalize_discrete(broken, original, support, np.zeros((1, SIZE, SIZE), np.float32))


def test_failure_reasons_are_sanitized_of_absolute_paths():
    assert "D:" not in _sanitize(r"failed reading D:\AI on IOT\dataset\x.png")
    assert "<path>" in _sanitize(r"failed reading D:\AI on IOT\dataset\x.png")
    assert "/home/" not in _sanitize("failed reading /home/user/secret/x.png")


def test_physics_artifact_map_is_in_requested_strength_units():
    """M7's preview map is peak-normalized; the M8 map averages the applied
    per-operator strengths so masked_mean == mean(strengths) == a_recipe."""
    class _Node:
        def __init__(self, name, strength): self.operator_name, self.strength = name, strength
    class _Graph:
        recipe_id = "rec"
        nodes = [_Node("blur", 0.2), _Node("halftone", 0.4)]
    class _Result:
        per_operator_support_masks = {"blur": np.ones((1, 8, 8), np.float32),
                                      "halftone": np.ones((1, 8, 8), np.float32)}
    values = applied_strength_map(_Graph(), _Result())
    assert values.shape == (1, 8, 8)
    assert pytest.approx(float(values.mean()), abs=1e-6) == 0.3


# --- M8 bank fixture ---------------------------------------------------------
_THRESHOLDS_FIXTURE = {"tau_fd": 0.5, "tau_id": 0.99, "tau_lm": 0.01, "tau_parse": 0.8,
                       "tau_out": 0.0, "tau_fp": 5.0}
_FIXTURE_MINIMUMS = {"candidates": 12, "accepted_total": 4, "accepted_physics": 2, "accepted_gpat": 2,
                     "accepted_live_casia": 2, "accepted_live_msu": 2, "require_all_artifact_types": 1,
                     "require_all_regions": 1, "require_same_and_cross_domain_gpat": True}


class _StubCalibration:
    quality_models = {"identity": {"sha256": "a" * 64}}


class _StubAudit:
    @staticmethod
    def report():
        return {"source_train_opened": True, "source_dev_opened": False, "target_test_opened": False,
                "target_label_artifact_opened": False, "raw_dataset_path_opened": False,
                "manifests_opened": ["manifests/source_train.parquet"]}


class _StubGenerator:
    """Duck-typed stand-in for `SyntheticBankGenerator` during assembly.

    Assembly only reads identity, config and already-written terminal records, so a
    stub keeps these tests hermetic: no package, no weights, no network.
    """
    def __init__(self, work_root, plan_rows, identity, calibration_path, bank_config):
        self.work_root, self.plan_rows, self._identity = work_root, plan_rows, identity
        self.calibration_path, self.bank_config = calibration_path, bank_config
        self.audit, self.calibration = _StubAudit(), _StubCalibration()
        self.gpat_architecture_hash = "c" * 64
        self.expected_pair_plan_identity = "d" * 64
        self.bank_id_prefix = None
        self.calibration_files: dict = {}

    def identity(self): return dict(self._identity)


def _fixture_identity(**overrides):
    identity = {"package_identity": "p" * 64, "recipe_bank_identity": "r" * 64,
                "candidate_plan_identity": "n" * 64, "threshold_sha256": "t" * 64,
                "fingerprint_reference_sha256": "f" * 64, "calibration_sha256": "l" * 64,
                "generation_config_sha256": "g" * 64, "gpat_checkpoint_sha256": "k" * 64,
                "physics_engine_version": "m7-physics-v1",
                "generator_version": "m8-synthetic-generator-v1",
                "discrete_convention": DISCRETE_CONVENTION}
    identity.update(overrides)
    return identity


def _fixture_plan_row(index, identity):
    route = "physics" if index % 2 == 0 else "gpat"
    dataset = "casia_fasd" if index % 4 < 2 else "msu_mfsd"
    return {"synthetic_id": f"syn_{index:024d}", "route": route, "slot": index % 2,
            "live_target_sample_id": f"live_{index}", "live_target_dataset": dataset,
            "live_target_record_id": f"rec_{index}",
            "spoof_source_sample_id": None if route == "physics" else f"spoof_{index}",
            "spoof_source_dataset": None if route == "physics" else "msu_mfsd",
            "spoof_source_record_id": None if route == "physics" else f"srec_{index}",
            "domain_relation": "not_applicable" if route == "physics" else
                               ("same_domain" if dataset == "msu_mfsd" else "cross_domain"),
            "recipe_id": f"rcp_{index % 3}", "recipe_seed": index, "candidate_seed": 20260806,
            "generator_binding": identity["physics_engine_version"] if route == "physics"
                                 else identity["gpat_checkpoint_sha256"],
            "gpat_checkpoint_sha256": None if route == "physics" else identity["gpat_checkpoint_sha256"],
            "physics_engine_version": "m7-physics-v1" if route == "physics" else None,
            "package_identity": identity["package_identity"],
            "recipe_bank_identity": identity["recipe_bank_identity"]}


def _fixture_record(row, identity, work_root, *, state):
    from prism_fas.utils.core import atomic_json_write
    from prism_fas.synthesis.synthetic_bank import _relative_paths, _sample_metadata
    index = int(row["synthetic_id"].removeprefix("syn_"))
    original, support = _live(index + 1), _support()
    generated = np.clip(original + 0.25, 0, 1).astype(np.float32)
    discrete = finalize_discrete(generated, original, support, np.full((1, SIZE, SIZE), 0.3, np.float32))
    failed = [] if state == "accepted" else ["identity"]
    record = {"schema_version": CANDIDATE_RECORD_SCHEMA_VERSION, "synthetic_id": row["synthetic_id"],
              "route": row["route"], "terminal_state": state, "identity": identity,
              "plan": {name: row[name] for name in
                       ("synthetic_id", "route", "slot", "live_target_sample_id", "live_target_dataset",
                        "spoof_source_sample_id", "spoof_source_dataset", "domain_relation", "recipe_id",
                        "candidate_seed", "generator_binding", "gpat_checkpoint_sha256",
                        "physics_engine_version")},
              "recipe": {"recipe_id": row["recipe_id"], "recipe_hash": "h" * 64, "graph_hash": "j" * 64,
                         "artifact_types": ["halftone"], "regions": ["left_eye"],
                         "requested_artifact_strength": 0.3},
              "geometry": {"exact_mask_pixels": discrete.exact_mask_pixels,
                           "requested_support_pixels": discrete.requested_support_pixels,
                           "requested_region_pixels": discrete.requested_support_pixels,
                           "requested_coverage": 1.0, "achieved_coverage": 1.0},
              "quality": {"accepted": state == "accepted", "failed_gates": failed,
                          "gates": {"identity": state == "accepted"},
                          "quality_components": {name: 0.5 for name in
                                                 ("q_fd", "q_id", "q_lm", "q_parse", "q_strength",
                                                  "q_fp", "q_support")},
                          "q": 0.5, "recipe_match": "not_applicable",
                          "threshold_hash": identity["threshold_sha256"],
                          "metrics": {"face_detection_score": 0.9, "identity_cosine": 0.999,
                                      "landmark_nme": 0.001, "outside_mask_parsing_dice": 0.95,
                                      "outside_mask_max_error": 0.0, "measured_artifact_strength": 0.3,
                                      "requested_artifact_strength": 0.3, "fingerprint_score": 1.0,
                                      "support_overlap": 1.0}},
              "generation_trace": {"engine_version": "m7-physics-v1"}}
    if state == "failed_generation":
        record["failure"] = {"stage": "generate", "exception_type": "SyntheticBankError", "reason": "fixture"}
        for key in ("quality", "recipe", "geometry"): record.pop(key)
    elif state == "accepted":
        paths = _relative_paths(row["synthetic_id"])
        record["outputs"] = {**paths, "image_sha256": discrete.image_sha256,
                             "mask_sha256": discrete.mask_sha256,
                             "artifact_map_sha256": discrete.artifact_map_sha256}
        for relative, payload in ((paths["image_relative_path"], discrete.image_png),
                                  (paths["mask_relative_path"], discrete.mask_png),
                                  (paths["artifact_map_relative_path"], discrete.artifact_map_npz)):
            target = work_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        atomic_json_write(work_root / "metadata" / f"{row['synthetic_id']}.json", _sample_metadata(record))
    atomic_json_write(work_root / "records" / f"{row['synthetic_id']}.json", record)
    return record


def _fixture_pairs(root):
    import pyarrow as pa, pyarrow.parquet as pq
    root.mkdir(parents=True, exist_ok=True)
    for name in ("pair_manifest_train.parquet", "pair_manifest_validation.parquet"):
        pq.write_table(pa.table({"pair_id": ["gpatpair_0"]}), root / name, compression="none")
    return root


def _fixture_calibration(path):
    from prism_fas.synthesis.quality_gate import Thresholds
    from prism_fas.utils.core import atomic_json_write
    thresholds = Thresholds.from_dict(_THRESHOLDS_FIXTURE)
    atomic_json_write(path, {"thresholds": thresholds.as_dict(), "threshold_sha256": thresholds.sha256(),
                             "fingerprint": {"references": {}, "reference_sha256": "f" * 64},
                             "used_source_dev": False, "used_target": False,
                             "used_generated_candidates": False,
                             "source_isolation": {"source_dev_opened": False, "target_test_opened": False},
                             "quality_models": {"identity": {"sha256": "a" * 64}}})
    return path


def _build_fixture_bank(tmp_path, *, count=12, identity=None, states=None):
    """A schema-valid 12-candidate bank: 6 physics / 6 gpat, mixed terminal states."""
    from prism_fas.synthesis.quality_gate import Thresholds
    identity = {**(identity or _fixture_identity()),
                "threshold_sha256": Thresholds.from_dict(_THRESHOLDS_FIXTURE).sha256()}
    work = Path(tmp_path) / "work"
    work.mkdir(parents=True, exist_ok=True)
    plan_rows = [_fixture_plan_row(index, identity) for index in range(count)]
    states = states or (["accepted"] * 8 + ["rejected"] * 3 + ["failed_generation"])
    records = [_fixture_record(row, identity, work, state=state) for row, state in zip(plan_rows, states)]
    config = {"seed": 20260806, "operational_minimums": {**_FIXTURE_MINIMUMS, "candidates": count},
              "shards": {"max_samples_per_shard": 3}}
    generator = _StubGenerator(work, plan_rows, identity,
                               _fixture_calibration(Path(tmp_path) / "quality_gate.json"), config)
    assembled = assemble_bank(generator, records, pairs_root=_fixture_pairs(Path(tmp_path) / "pairs"))
    return generator, records, assembled


def _validate_fixture(bank_root, **kwargs):
    from prism_fas.synthesis.synthetic_validation import validate_bank
    return validate_bank(Path(bank_root), expected_candidates=12, **kwargs)


# --- bank assembly and BANK_LOCK ---------------------------------------------
def test_bank_layout_holds_every_required_path(tmp_path):
    _, _, assembled = _build_fixture_bank(tmp_path)
    root = Path(assembled["bank_root"])
    assert root.name == assembled["bank_id"] and root.name.startswith("prism_synthetic_bank_m8_v1_")
    for relative in ("images", "artifact_maps", "masks", "metadata", "manifests", "calibration", "shards",
                     "manifests/candidate_manifest.parquet", "manifests/manifest.parquet",
                     "manifests/rejected.parquet", "manifests/failures.parquet",
                     "manifests/pair_manifest_train.parquet", "manifests/pair_manifest_validation.parquet",
                     "calibration/quality_gate.json", "shards_index.parquet", "quality_summary.json",
                     "generation_summary.json", "BANK_LOCK.json"):
        assert (root / relative).exists(), relative


def test_terminal_accounting_covers_every_planned_candidate(tmp_path):
    _, _, assembled = _build_fixture_bank(tmp_path)
    lock = assembled["lock"]
    assert lock["accepted_count"] + lock["rejected_count"] + lock["failed_count"] == lock["candidate_count"] == 12
    assert lock["status"] == "validated" and lock["bank_lock_schema_version"] == BANK_LOCK_SCHEMA_VERSION


def test_accepted_manifest_carries_every_required_field(tmp_path):
    _, _, assembled = _build_fixture_bank(tmp_path)
    rows = load_manifest(Path(assembled["bank_root"]) / "manifests" / "manifest.parquet",
                         MANIFEST_SCHEMAS["manifest"])
    assert rows
    for name in ("synthetic_id", "route", "live_target_sample_id", "spoof_source_sample_id",
                 "live_target_dataset", "recipe_id", "recipe_hash", "graph_hash", "gpat_checkpoint_sha256",
                 "image_relative_path", "image_sha256", "mask_relative_path", "mask_sha256",
                 "artifact_map_relative_path", "artifact_map_sha256", "q", "identity_cosine",
                 "calibration_hash", "exact_mask_pixels", "requested_coverage", "achieved_coverage",
                 "package_identity", "recipe_bank_identity", "candidate_plan_identity",
                 "generation_config_hash"):
        assert name in rows[0], name
    assert all(row["recipe_match"] == "not_applicable" for row in rows)
    assert all(not str(row["image_relative_path"]).startswith("/") for row in rows)


def test_rejected_rows_name_failed_gates_and_failures_carry_no_path(tmp_path):
    _, _, assembled = _build_fixture_bank(tmp_path)
    root = Path(assembled["bank_root"])
    rejected = load_manifest(root / "manifests" / "rejected.parquet", MANIFEST_SCHEMAS["rejected"])
    assert rejected and all(row["failed_gates"] for row in rejected)
    failures = load_manifest(root / "manifests" / "failures.parquet", MANIFEST_SCHEMAS["failures"])
    assert failures and all(row["failed_stage"] and row["exception_type"] for row in failures)
    assert all("/" not in str(row["reason"]) and "\\" not in str(row["reason"]) for row in failures)


def test_rejected_image_binaries_are_not_preserved(tmp_path):
    _, records, assembled = _build_fixture_bank(tmp_path)
    root = Path(assembled["bank_root"])
    rejected = [record for record in records if record["terminal_state"] == "rejected"]
    assert rejected
    for record in rejected:
        assert not (root / "images" / f"{record['synthetic_id']}.png").exists()


def test_operational_minimums_are_enforced(tmp_path):
    _, _, assembled = _build_fixture_bank(tmp_path)
    assert assembled["operational_minimums"]["passed"]
    accepted = load_manifest(Path(assembled["bank_root"]) / "manifests" / "manifest.parquet",
                             MANIFEST_SCHEMAS["manifest"])
    strict = check_operational_minimums(accepted, 12, {**_FIXTURE_MINIMUMS, "accepted_total": 999})
    assert not strict["passed"] and strict["checks"]["accepted_total"] is False
    short = check_operational_minimums(accepted, 11, dict(_FIXTURE_MINIMUMS))
    assert short["checks"]["candidates"] is False


def test_bank_lock_identity_excludes_timestamps_and_is_reproducible(tmp_path):
    import hashlib
    _, _, assembled = _build_fixture_bank(tmp_path)
    lock = assembled["lock"]
    excluded = lock["identity_excluded_fields"]
    assert "created_at" in excluded and "bank_id" in excluded
    recomputed = hashlib.sha256(json.dumps({key: value for key, value in lock.items() if key not in excluded},
                                           sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert recomputed == lock["bank_content_identity_sha256"]
    assert lock["bank_id"].endswith(lock["bank_content_identity_sha256"][:12])
    text = json.dumps(lock)
    assert "/tmp" not in text and "C:\\" not in text and "modal" not in text.lower()


@pytest.mark.parametrize("field", ["gpat_checkpoint_sha256", "calibration_sha256", "candidate_plan_identity",
                                   "generation_config_sha256", "fingerprint_reference_sha256"])
def test_bank_lock_changes_when_a_bound_identity_changes(tmp_path, field):
    _, _, base = _build_fixture_bank(tmp_path / "base")
    _, _, changed = _build_fixture_bank(tmp_path / "changed", identity=_fixture_identity(**{field: "z" * 64}))
    assert changed["lock"]["bank_content_identity_sha256"] != base["lock"]["bank_content_identity_sha256"]
    assert changed["bank_id"] != base["bank_id"]


def test_reassembly_reuses_the_same_bank_id(tmp_path):
    generator, records, first = _build_fixture_bank(tmp_path)
    second = assemble_bank(generator, records, pairs_root=tmp_path / "pairs")
    assert second["status"] == "reused" and second["bank_id"] == first["bank_id"]
    assert second["lock"]["bank_content_identity_sha256"] == first["lock"]["bank_content_identity_sha256"]


# --- deterministic shards -----------------------------------------------------
def test_shard_bytes_are_deterministic_and_metadata_normalized(tmp_path):
    _, _, assembled = _build_fixture_bank(tmp_path)
    root = Path(assembled["bank_root"])
    index = load_shards_index(root / "shards_index.parquet")
    assert len(index) == 3                      # 8 accepted at 3 per shard
    ids = sorted(row["synthetic_id"] for row in
                 load_manifest(root / "manifests" / "manifest.parquet", MANIFEST_SCHEMAS["manifest"]))
    assert build_shard_bytes(root, ids[:3]) == build_shard_bytes(root, ids[:3])
    for member in shard_member_metadata(root / "shards" / index[0]["shard_name"]):
        assert member["mtime"] == TAR_MTIME and member["uid"] == 0 and member["gid"] == 0
        assert member["uname"] == "" and member["gname"] == "" and member["mode"] == 0o644
    for row in index:
        assert row["first_synthetic_id"] <= row["last_synthetic_id"]
        assert row["physics_count"] + row["gpat_count"] == row["row_count"]
        assert row["live_casia_fasd_count"] + row["live_msu_mfsd_count"] == row["row_count"]


def test_shard_members_and_loose_shard_parity(tmp_path):
    from prism_fas.synthesis.synthetic_shards import read_shard_members
    _, _, assembled = _build_fixture_bank(tmp_path)
    root = Path(assembled["bank_root"])
    accepted = load_manifest(root / "manifests" / "manifest.parquet", MANIFEST_SCHEMAS["manifest"])
    index = load_shards_index(root / "shards_index.parquet")
    report = validate_shards(root, index, accepted)
    assert report["passed"] and report["errors"] == [] and report["covers_every_accepted_row"]
    members = read_shard_members(root / "shards" / index[0]["shard_name"])
    assert all(any(name.endswith(suffix) for suffix in MEMBER_SUFFIXES) for name in members)
    assert len(members) == index[0]["row_count"] * len(MEMBER_SUFFIXES)
    assert index_digest(index) == assembled["lock"]["shards_index_sha256"]


def test_shard_rebuild_is_byte_identical(tmp_path):
    _, _, assembled = _build_fixture_bank(tmp_path)
    root = Path(assembled["bank_root"])
    accepted = load_manifest(root / "manifests" / "manifest.parquet", MANIFEST_SCHEMAS["manifest"])
    before = {row["shard_name"]: row["sha256"] for row in load_shards_index(root / "shards_index.parquet")}
    rebuilt = build_shards(root, accepted, max_samples=3)
    write_shards_index(root, rebuilt)
    assert {row["shard_name"]: row["sha256"] for row in rebuilt} == before


def test_shard_parity_detects_a_diverged_loose_file(tmp_path):
    _, _, assembled = _build_fixture_bank(tmp_path)
    root = Path(assembled["bank_root"])
    accepted = load_manifest(root / "manifests" / "manifest.parquet", MANIFEST_SCHEMAS["manifest"])
    index = load_shards_index(root / "shards_index.parquet")
    target = root / accepted[0]["image_relative_path"]
    target.write_bytes(encode_png(np.zeros((SIZE, SIZE, 3), np.uint8)))
    report = validate_shards(root, index, accepted)
    assert not report["passed"] and report["error_count"] > 0


# --- resume and reuse ---------------------------------------------------------
def test_candidate_reuse_requires_every_identity_field(tmp_path):
    _, records, _ = _build_fixture_bank(tmp_path)
    work = tmp_path / "work"
    accepted = next(record for record in records if record["terminal_state"] == "accepted")
    usable, reason = record_is_reusable(accepted, accepted["identity"], work)
    assert usable and reason == "reused"
    for field in ("gpat_checkpoint_sha256", "threshold_sha256", "candidate_plan_identity",
                  "generation_config_sha256", "package_identity", "recipe_bank_identity"):
        usable, reason = record_is_reusable(accepted, {**accepted["identity"], field: "z" * 64}, work)
        assert not usable and reason == f"identity:{field}"


def test_candidate_reuse_rejects_a_corrupted_or_missing_output(tmp_path):
    _, records, _ = _build_fixture_bank(tmp_path)
    work = tmp_path / "work"
    accepted = next(record for record in records if record["terminal_state"] == "accepted")
    image = work / accepted["outputs"]["image_relative_path"]
    image.write_bytes(image.read_bytes()[:64])
    usable, reason = record_is_reusable(accepted, accepted["identity"], work)
    assert not usable and reason.startswith("hash:")
    image.unlink()
    usable, reason = record_is_reusable(accepted, accepted["identity"], work)
    assert not usable and reason.startswith("missing:")


def test_rejected_and_failed_records_are_reusable_without_binaries(tmp_path):
    _, records, _ = _build_fixture_bank(tmp_path)
    for state in ("rejected", "failed_generation"):
        record = next(item for item in records if item["terminal_state"] == state)
        usable, _ = record_is_reusable(record, record["identity"], tmp_path / "work")
        assert usable


def test_reuse_rejects_a_record_without_a_terminal_state(tmp_path):
    _, records, _ = _build_fixture_bank(tmp_path)
    record = dict(records[0]); record["terminal_state"] = "in_progress"
    usable, reason = record_is_reusable(record, record["identity"], tmp_path / "work")
    assert not usable and reason == "terminal_state"


# --- full validation ----------------------------------------------------------
def test_full_validation_passes_on_a_valid_bank(tmp_path):
    _, _, assembled = _build_fixture_bank(tmp_path)
    report = _validate_fixture(assembled["bank_root"])
    assert report["passed"], report["errors"]
    for name in ("lock_status_validated", "bank_content_identity_reproducible", "terminal_accounting",
                 "no_duplicate_synthetic_ids", "accepted_files_exist", "accepted_hashes_match",
                 "images_decode_rgb_224", "masks_binary_0_255", "artifact_maps_load_without_pickle",
                 "artifact_maps_finite_in_range", "artifact_maps_zero_outside_exact_mask",
                 "exact_mask_pixel_counts_match", "every_accepted_row_passes_every_hard_gate",
                 "every_rejected_row_names_a_failed_gate", "recipe_match_not_applicable",
                 "no_target_or_private_fields", "source_only_isolation_evidence",
                 "calibration_declares_no_source_dev_or_target", "operational_minimums",
                 "shards_validate", "shard_hashes_match_lock", "shards_index_digest",
                 "bank_directory_matches_bank_id", "candidate_manifest_covers_terminals"):
        assert report["checks"][name] is True, name
    assert report["counts"] == {"candidates": 12, "accepted": 8, "rejected": 3, "failed_generation": 1}


def test_validation_detects_a_tampered_accepted_image(tmp_path):
    _, _, assembled = _build_fixture_bank(tmp_path)
    root = Path(assembled["bank_root"])
    sorted((root / "images").glob("*.png"))[0].write_bytes(encode_png(np.zeros((SIZE, SIZE, 3), np.uint8)))
    report = _validate_fixture(root)
    assert not report["passed"] and report["checks"]["accepted_hashes_match"] is False


def test_validation_detects_a_tampered_lock(tmp_path):
    _, _, assembled = _build_fixture_bank(tmp_path)
    root = Path(assembled["bank_root"])
    lock = json.loads((root / "BANK_LOCK.json").read_text(encoding="utf-8"))
    lock["accepted_count"] = lock["accepted_count"] + 1
    (root / "BANK_LOCK.json").write_text(json.dumps(lock), encoding="utf-8")
    report = _validate_fixture(root)
    assert not report["passed"] and report["checks"]["bank_content_identity_reproducible"] is False


def test_validation_detects_an_artifact_map_outside_the_exact_mask(tmp_path):
    _, _, assembled = _build_fixture_bank(tmp_path)
    root = Path(assembled["bank_root"])
    target = sorted((root / "artifact_maps").glob("*.npz"))[0]
    target.write_bytes(encode_npz({ARTIFACT_MAP_KEY: np.full((1, SIZE, SIZE), 0.5, np.float16)}))
    report = _validate_fixture(root)
    assert not report["passed"]
    assert report["payload_report"]["map_outside_errors"] > 0 or report["checks"]["accepted_hashes_match"] is False


# --- export, freeze and import ------------------------------------------------
def test_export_archive_is_deterministic_and_round_trips(tmp_path):
    from prism_fas.synthesis.synthetic_export import build_archive_bytes, export_archive, extract_archive
    from prism_fas.synthesis.synthetic_validation import compare_banks
    _, _, assembled = _build_fixture_bank(tmp_path)
    root = Path(assembled["bank_root"])
    assert build_archive_bytes(root) == build_archive_bytes(root)
    first = export_archive(root, tmp_path / "exports")
    assert first["status"] == "created" and first["archive_is_identity_bearing"] is False
    again = export_archive(root, tmp_path / "exports")
    assert again["status"] == "reused" and again["archive_sha256"] == first["archive_sha256"]
    extraction = extract_archive(tmp_path / "exports" / f"{assembled['bank_id']}.tar", tmp_path / "down")
    assert extraction["status"] == "extracted" and extraction["bank_id"] == assembled["bank_id"]
    assert compare_banks(root, Path(extraction["bank_root"]))["identical"]
    local = _validate_fixture(extraction["bank_root"])
    assert local["passed"], local["errors"]


def test_export_refuses_an_unvalidated_bank(tmp_path):
    from prism_fas.synthesis.synthetic_export import ExportError, export_archive
    _, _, assembled = _build_fixture_bank(tmp_path)
    lock_path = Path(assembled["bank_root"]) / "BANK_LOCK.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8")); lock["status"] = "draft"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(ExportError):
        export_archive(Path(assembled["bank_root"]), tmp_path / "exports")


def test_import_refuses_a_different_bank_at_the_same_path(tmp_path):
    from prism_fas.synthesis.synthetic_export import ExportError, export_archive, extract_archive
    _, _, first = _build_fixture_bank(tmp_path / "a")
    export_archive(Path(first["bank_root"]), tmp_path / "exports")
    extract_archive(tmp_path / "exports" / f"{first['bank_id']}.tar", tmp_path / "down")
    assert extract_archive(tmp_path / "exports" / f"{first['bank_id']}.tar",
                           tmp_path / "down")["status"] == "already_present"
    lock = tmp_path / "down" / first["bank_id"] / "BANK_LOCK.json"
    payload = json.loads(lock.read_text(encoding="utf-8"))
    payload["bank_content_identity_sha256"] = "0" * 64
    lock.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ExportError):
        extract_archive(tmp_path / "exports" / f"{first['bank_id']}.tar", tmp_path / "down")


def test_freeze_reuses_an_identical_bank_and_refuses_a_different_one(tmp_path):
    from prism_fas.synthesis.synthetic_export import ExportError, freeze_bank
    _, _, assembled = _build_fixture_bank(tmp_path)
    root = Path(assembled["bank_root"])
    assert freeze_bank(root, tmp_path / "frozen")["status"] == "frozen"
    assert freeze_bank(root, tmp_path / "frozen")["status"] == "reused"
    lock = tmp_path / "frozen" / assembled["bank_id"] / "BANK_LOCK.json"
    payload = json.loads(lock.read_text(encoding="utf-8"))
    payload["bank_content_identity_sha256"] = "0" * 64
    lock.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ExportError):
        freeze_bank(root, tmp_path / "frozen")


# --- CLI and module hygiene ----------------------------------------------------
def test_every_m8_write_command_exists_and_supports_dry_run():
    text = (ROOT / "src" / "prism_fas" / "cli" / "main.py").read_text(encoding="utf-8")
    for command in ("generation-pilot", "generate-bank", "build-shards", "export-bank",
                    "validate-bank", "validate-downloaded-bank"):
        assert f'@synthesis_app.command("{command}")' in text, command
    body = text[text.index('@synthesis_app.command("generation-pilot")'):]
    assert body.count("'--dry-run'") >= 5
    assert "--resume/--no-resume" in body and "'--limit'" in body


def test_synthetic_bank_modules_never_import_modal():
    import ast
    for name in ("synthetic_bank", "synthetic_shards", "synthetic_validation", "synthetic_export",
                 "m8_pipeline"):
        path = ROOT / "src" / "prism_fas" / "synthesis" / f"{name}.py"
        assert path.is_file(), name
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            imported = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                        else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            assert not any(item.split(".")[0] == "modal" for item in imported), name


def test_synthetic_bank_config_declares_the_frozen_minimums():
    import yaml
    config = yaml.safe_load((ROOT / "configs" / "synthesis" / "synthetic_bank_m8.yaml").read_text(encoding="utf-8"))
    assert config["operational_minimums"] == {"candidates": 1120, "accepted_total": 400,
                                              "accepted_physics": 200, "accepted_gpat": 100,
                                              "accepted_live_casia": 100, "accepted_live_msu": 100,
                                              "require_all_artifact_types": 8, "require_all_regions": 9,
                                              "require_same_and_cross_domain_gpat": True}
    assert config["shards"]["max_samples_per_shard"] == 500 and config["shards"]["compression"] == "none"
    assert config["remote"]["export_is_identity_bearing"] is False


def test_downloaded_bank_validator_script_is_source_only():
    """The forbidden split names may appear in the module docstring's isolation
    statement, but never in the executable code."""
    import ast
    path = ROOT / "scripts" / "m8_validate_downloaded_bank.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))]
    code = ast.unparse(tree)
    assert "source_dev" not in code and "target_test" not in code
    text = path.read_text(encoding="utf-8")
    assert "--expected-identity" in text and "--dry-run" in text and "--expected-archive-sha256" in text


# --- real report contracts ------------------------------------------------------
def test_real_generation_pilot_report_contract():
    report = _report("generation_pilot.json")
    assert report["passed"] is True and report["planned"] == 32
    assert sum(report["terminal_counts"].values()) == 32
    assert report["coverage"]["artifact_type_count"] == 8 and report["coverage"]["region_count"] == 9
    buckets = report["coverage"]["bucket_counts"]
    assert buckets["route:physics"] == 16 and buckets["route:gpat"] == 16
    assert buckets["relation:same_domain"] == 8 and buckets["relation:cross_domain"] == 8
    assert buckets["live:casia_fasd"] == 16 and buckets["live:msu_mfsd"] == 16
    assert report["payload_errors"] == [] and all(report["checks"].values())
    assert report["identity"]["gpat_checkpoint_sha256"] == GPAT_BEST_SHA
    assert report["identity"]["discrete_convention"] == DISCRETE_CONVENTION
    isolation = report["source_isolation"]
    assert isolation["source_train_opened"] is True and isolation["source_dev_opened"] is False
    assert isolation["target_test_opened"] is False and isolation["raw_dataset_path_opened"] is False
    assert isolation["manifests_opened"] == ["manifests/source_train.parquet"]
    assert report["gpu"]["gpu_name"] == "NVIDIA L4"


def test_real_generation_pilot_determinism_report_contract():
    report = _report("generation_pilot_determinism.json")
    assert report["passed"] is True and report["identical"] is True
    assert report["candidates"] == 32 and report["compared"] == 32
    assert report["mismatch_count"] == 0 and report["mismatches"] == []


def test_bank_lock_status_is_not_unconditional(tmp_path):
    """A bank that misses a pre-declared minimum is retained and labelled, never
    relabelled `validated`."""
    _, records, _ = _build_fixture_bank(tmp_path / "ok")
    assert json.loads((Path(_build_fixture_bank(tmp_path / "ok2")[2]["bank_root"]) /
                       "BANK_LOCK.json").read_text(encoding="utf-8"))["status"] == "validated"
    work = tmp_path / "strict" / "work"
    work.mkdir(parents=True, exist_ok=True)
    identity = records[0]["identity"]
    plan_rows = [_fixture_plan_row(index, identity) for index in range(12)]
    states = ["rejected"] * 12                       # nothing accepted -> minimums cannot pass
    fresh = [_fixture_record(row, identity, work, state=state) for row, state in zip(plan_rows, states)]
    config = {"seed": 20260806, "operational_minimums": dict(_FIXTURE_MINIMUMS),
              "shards": {"max_samples_per_shard": 3}}
    generator = _StubGenerator(work, plan_rows, identity,
                               _fixture_calibration(tmp_path / "strict" / "quality_gate.json"), config)
    assembled = assemble_bank(generator, fresh, pairs_root=_fixture_pairs(tmp_path / "strict" / "pairs"))
    assert assembled["operational_minimums"]["passed"] is False
    assert assembled["lock"]["status"] == "operational_minimums_failed"
    report = _validate_fixture(assembled["bank_root"])
    assert report["passed"] is False and report["checks"]["lock_status_validated"] is False


def test_export_refuses_a_failed_run_unless_explicitly_allowed(tmp_path):
    from prism_fas.synthesis.synthetic_export import ExportError, export_archive, freeze_bank
    _, _, assembled = _build_fixture_bank(tmp_path)
    lock_path = Path(assembled["bank_root"]) / "BANK_LOCK.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["status"] = "operational_minimums_failed"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(ExportError):
        export_archive(Path(assembled["bank_root"]), tmp_path / "exports")
    moved = export_archive(Path(assembled["bank_root"]), tmp_path / "exports", require_validated=False)
    assert moved["status"] == "created" and moved["bank_status"] == "operational_minimums_failed"
    # freezing under the immutable versioned path is never allowed for a failed run
    with pytest.raises(ExportError):
        freeze_bank(Path(assembled["bank_root"]), tmp_path / "frozen")


# --- real full-generation report contracts --------------------------------------
def test_real_resume_audit_report_contract():
    report = _report("resume_audit.json")
    assert report["passed"] is True
    phases = {phase["phase"]: phase for phase in report["phases"]}
    assert set(phases) == {"interrupted", "resumed", "completed_rerun"}
    assert phases["interrupted"]["status"] == "interrupted" and phases["interrupted"]["rebuilt"] == 96
    assert phases["resumed"]["status"] == "completed" and phases["resumed"]["examined"] == 1120
    assert phases["resumed"]["reused"] > 0
    rerun = phases["completed_rerun"]
    assert rerun["examined"] == 1120 and rerun["reused"] == 1120 and rerun["rebuilt"] == 0
    probe = report["corruption_probe"]
    assert probe["performed"] is True and probe["detected_as_unusable"] is True
    assert probe["reason"].startswith("hash:") and probe["rebuilt_bytes_identical"] is True
    assert probe["terminal_state_after_rebuild"] == "accepted"
    assert sum(report["terminal_counts"].values()) == 1120
    assert report["terminal_counts"].get("failed_generation", 0) == 0
    for name in ("all_candidates_examined", "completed_rerun_rebuilt_nothing",
                 "completed_rerun_reused_everything", "no_duplicate_candidate_ids",
                 "terminal_accounting", "resume_reused_valid_candidates",
                 "corrupted_candidate_detected", "corrupted_candidate_rebuilt"):
        assert report["checks"][name] is True, name
    isolation = report["source_isolation"]
    assert isolation["source_dev_opened"] is False and isolation["target_test_opened"] is False


def test_real_determinism_audit_report_contract():
    report = _report("determinism_audit.json")
    assert report["passed"] is True and report["identical"] is True
    assert report["candidates"] == 32 and report["compared"] == 32
    assert report["mismatch_count"] == 0 and report["mismatches"] == []
    assert report["identity"]["gpat_checkpoint_sha256"] == GPAT_BEST_SHA


def test_real_synthetic_bank_validation_report_contract():
    """Every structural invariant holds. The ONLY failing check is the
    pre-declared operational minimums, which are never relaxed to make it pass."""
    report = _report("synthetic_bank_validation.json")
    counts = report["counts"]
    assert counts["candidates"] == 1120
    assert counts["accepted"] + counts["rejected"] + counts["failed_generation"] == 1120
    assert counts["failed_generation"] == 0
    payload = report["payload_report"]
    assert payload["checked"] == counts["accepted"]
    for name in ("missing_files", "hash_mismatches", "image_shape_errors", "mask_value_errors",
                 "npz_errors", "map_range_errors", "map_outside_errors", "mask_pixel_mismatches",
                 "outside_mask_errors"):
        assert payload[name] == 0, name
    assert payload["outside_mask_checked"] == counts["accepted"]
    assert report["shard_report"]["passed"] is True
    assert report["shard_report"]["covers_every_accepted_row"] is True
    for name in ("terminal_accounting", "no_duplicate_synthetic_ids", "accepted_files_exist",
                 "accepted_hashes_match", "images_decode_rgb_224", "masks_binary_0_255",
                 "artifact_maps_load_without_pickle", "artifact_maps_zero_outside_exact_mask",
                 "saved_outside_mask_error_exactly_zero", "every_accepted_row_passes_every_hard_gate",
                 "every_rejected_row_names_a_failed_gate", "recipe_match_not_applicable",
                 "no_target_or_private_fields", "source_only_isolation_evidence",
                 "shards_validate", "shard_hashes_match_lock", "candidate_count",
                 "source_package_unchanged", "recipe_bank_unchanged", "gpat_checkpoint_hash_matches"):
        assert report["checks"][name] is True, name
    assert report["coverage"]["artifact_type_count"] == 8 and report["coverage"]["region_count"] == 9
    identities = report["parent_identities"]
    assert identities["candidate_plan_identity"] == CANDIDATE_PLAN_IDENTITY
    assert identities["gpat_checkpoint_sha256"] == GPAT_BEST_SHA
    assert identities["threshold_sha256"] == THRESHOLD_SHA
    assert identities["fingerprint_reference_sha256"] == FINGERPRINT_REFERENCE_SHA
    assert report["leak_scan"]["hits"] == {}
    assert report["leak_scan"]["source_isolation_clean"] is True


def test_real_run_missed_only_the_pre_declared_operational_minimums():
    """The honest record of the run: the frozen gate is strict, and the response is
    to document it, never to lower a threshold or resample a rejected candidate."""
    report = _report("synthetic_bank_validation.json")
    minimums = report["operational_minimums"]
    failing = sorted(name for name, passed in minimums["checks"].items() if not passed)
    assert failing == ["accepted_physics", "accepted_total"]
    assert minimums["passed"] is False
    assert [name for name in report["checks"] if report["checks"][name] is False] != []
    declared = minimums["declared_minimums"]
    assert declared["accepted_total"] == 400 and declared["accepted_physics"] == 200
    observed = minimums["observed"]
    assert observed["accepted_total"] < declared["accepted_total"]
    assert observed["accepted_route_counts"]["physics"] < declared["accepted_physics"]
    assert observed["accepted_route_counts"]["gpat"] >= declared["accepted_gpat"]
    assert set(observed["accepted_gpat_domain_relations"]) == {"same_domain", "cross_domain"}


def test_frozen_thresholds_were_not_touched_by_generation():
    calibration = _report("quality_calibration.json")
    assert calibration["thresholds"] == {"tau_fd": 0.5, "tau_id": 0.9995203357934952,
                                         "tau_lm": 0.002135227532959269,
                                         "tau_parse": 0.8747814437904173, "tau_out": 0.0,
                                         "tau_fp": 5.687657785453908}
    assert calibration["threshold_sha256"] == THRESHOLD_SHA
    assert calibration["fingerprint"]["reference_sha256"] == FINGERPRINT_REFERENCE_SHA
    assert calibration["used_generated_candidates"] is False
    validation = _report("synthetic_bank_validation.json")
    assert validation["parent_identities"]["threshold_sha256"] == THRESHOLD_SHA


def test_real_local_downloaded_bank_validation_report_contract():
    """The bank that came back over the single transport archive is byte-for-byte
    the bank the container built."""
    report = _report("local_downloaded_bank_validation.json")
    transport = report["transport"]
    assert transport["archive_sha256_matches"] is True
    assert transport["archive_bytes"] > 0 and transport["archive_name"].endswith(".tar")
    assert transport["extraction"]["bank_id"] == report["bank_id"]
    assert report["local_identity_equals_remote"] is True
    assert report["bank_content_identity_sha256"] == report["expected_identity"]
    assert report["local_bank_root_name"] == report["bank_id"]
    counts = report["counts"]
    assert counts["candidates"] == 1120
    assert counts["accepted"] + counts["rejected"] + counts["failed_generation"] == 1120
    payload = report["payload_report"]
    for name in ("missing_files", "hash_mismatches", "image_shape_errors", "mask_value_errors",
                 "npz_errors", "map_range_errors", "map_outside_errors", "mask_pixel_mismatches",
                 "outside_mask_errors"):
        assert payload[name] == 0, name
    assert report["shard_report"]["passed"] is True
    assert report["leak_scan"]["hits"] == {} and report["leak_scan"]["source_isolation_clean"] is True
    for name in ("bank_content_identity_reproducible", "accepted_hashes_match", "shards_validate",
                 "source_package_unchanged", "recipe_bank_unchanged",
                 "saved_outside_mask_error_exactly_zero"):
        assert report["checks"][name] is True, name
    # The downloaded copy fails exactly the two checks the remote build failed:
    # the pre-declared minimums and the status they force. Nothing else.
    assert sorted(name for name, ok in report["checks"].items() if not ok) == \
        ["lock_status_validated", "operational_minimums"]


def test_real_bank_lock_records_the_failed_run_honestly():
    lock = _report("BANK_LOCK_remote.json")
    assert lock["status"] == "operational_minimums_failed"
    assert lock["operational_minimums_passed"] is False
    assert lock["candidate_count"] == 1120
    assert lock["accepted_count"] + lock["rejected_count"] + lock["failed_count"] == 1120
    assert lock["failed_count"] == 0
    assert lock["gpat_checkpoint_sha256"] == GPAT_BEST_SHA
    assert lock["candidate_plan_identity"] == CANDIDATE_PLAN_IDENTITY
    assert lock["threshold_sha256"] == THRESHOLD_SHA
    assert lock["fingerprint_reference_sha256"] == FINGERPRINT_REFERENCE_SHA
    assert lock["physics_engine_version"] == "m7-physics-v1"
    assert lock["discrete_convention"] == DISCRETE_CONVENTION
    assert lock["bank_id"].endswith(lock["bank_content_identity_sha256"][:12])
    assert lock["accepted_coverage"]["artifact_type_count"] == 8
    assert lock["accepted_coverage"]["region_count"] == 9
    assert sorted(lock["accepted_domain_relations"]) == ["cross_domain", "same_domain"]
    text = json.dumps(lock)
    assert "/tmp" not in text and "C:\\" not in text and "/vol/" not in text
    assert "created_at" in lock["identity_excluded_fields"]


def test_real_export_report_marks_the_archive_as_non_identity_bearing():
    report = _report("export.json")
    assert report["archive_is_identity_bearing"] is False
    assert report["archive_excluded_from_bank_lock"] is True
    assert report["bank_status"] == "operational_minimums_failed"
    assert report["archive_relative_name"] == f"{report['bank_id']}.tar"
    assert len(report["archive_sha256"]) == 64 and report["archive_bytes"] > 0
    # 391 accepted x 4 members + the 12 top-level bank files
    assert report["member_count"] == 391 * 4 + 12


# --- M8 identity calibration v2 ------------------------------------------------
from prism_fas.synthesis.identity_calibration import (  # noqa: E402
    COSINE_TOLERANCE, GENUINE_FIELDS, IDENTITY_CALIBRATION_VERSION, IMPOSTOR_FIELDS,
    IdentityCalibrationError, build_genuine_pairs, build_impostor_pairs, build_lock,
    compare_calibrations, group_by_identity, identity_key, identity_key_hash, load_identity_config,
    normalize_subject_id, pairs_digest, preprocessing_contract_identity, write_pairs_parquet)
from prism_fas.synthesis.pair_plan import SourceRow  # noqa: E402

V2_CONFIG = ROOT / "configs" / "synthesis" / "quality_gate_m8_v2.yaml"
TAU_ID_V1 = 0.9995203357934952


def _row(sample, dataset, record, subject):
    return SourceRow(sample_id=sample, dataset=dataset, source_record_id=record,
                     subject_id=subject, label="live", project_split="source_train")


def _population(samples_per_record=2, records=2, identities=3):
    """A miniature but structurally faithful source_train live population."""
    rows = []
    for dataset in ("casia_fasd", "msu_mfsd"):
        for subject in range(identities):
            for record in range(records):
                for frame in range(samples_per_record):
                    rows.append(_row(f"{dataset}_s{subject}_r{record}_f{frame}", dataset,
                                     f"{dataset}_rec{subject}_{record}", str(subject)))
    return sorted(rows, key=lambda row: row.sample_id)


def test_identity_key_is_dataset_prefixed_and_normalized():
    assert identity_key("casia_fasd", " 5 ") == "casia_fasd::5"
    assert identity_key("msu_mfsd", "A") == "msu_mfsd::a"
    # the same numeric subject id in two datasets is two different people
    assert identity_key("casia_fasd", "5") != identity_key("msu_mfsd", "5")
    assert identity_key_hash("casia_fasd::5") != identity_key_hash("msu_mfsd::5")
    with pytest.raises(IdentityCalibrationError): normalize_subject_id("")
    with pytest.raises(IdentityCalibrationError): normalize_subject_id(None)
    with pytest.raises(IdentityCalibrationError): identity_key("siw_mv2", "1")


def test_genuine_pairs_are_same_identity_and_cross_record():
    rows = _population()
    groups = group_by_identity(rows)
    pairs = build_genuine_pairs(rows)
    assert pairs
    keyed = {identity_key_hash(key): key for key in groups}
    for pair in pairs:
        key = keyed[pair["identity_key_hash"]]
        dataset, subject = key.split("::", 1)
        assert pair["dataset"] == dataset
        assert pair["source_record_id_a"] != pair["source_record_id_b"]
        assert pair["sample_id_a"] != pair["sample_id_b"]
        assert pair["sample_id_a"] < pair["sample_id_b"]
        for sample in (pair["sample_id_a"], pair["sample_id_b"]):
            row = next(item for item in rows if item.sample_id == sample)
            assert identity_key(row.dataset, row.subject_id) == key


def test_genuine_pairs_reject_same_record_frames():
    """Two frames of one canonical record are one observation, not two."""
    rows = [_row("a0", "casia_fasd", "rec0", "1"), _row("a1", "casia_fasd", "rec0", "1"),
            _row("a2", "casia_fasd", "rec0", "1")]
    assert build_genuine_pairs(rows) == []
    rows.append(_row("a3", "casia_fasd", "rec1", "1"))
    pairs = build_genuine_pairs(rows)
    assert len(pairs) == 3 and all(pair["source_record_id_a"] != pair["source_record_id_b"] for pair in pairs)


def test_genuine_cap_is_per_identity():
    rows = _population(samples_per_record=6, records=2, identities=1)
    capped = build_genuine_pairs(rows, maximum_per_identity=5)
    counts = {}
    for pair in capped: counts[pair["identity_key_hash"]] = counts.get(pair["identity_key_hash"], 0) + 1
    assert counts and all(value <= 5 for value in counts.values())
    assert len(build_genuine_pairs(rows, maximum_per_identity=0)) == 0
    uncapped = build_genuine_pairs(rows, maximum_per_identity=1000)
    assert len(uncapped) > len(capped)


def test_impostor_pairs_are_different_identity_and_same_dataset():
    rows = _population()
    pairs = build_impostor_pairs(rows, maximum_total=10000)
    assert pairs
    by_sample = {row.sample_id: row for row in rows}
    for pair in pairs:
        left, right = by_sample[pair["sample_id_a"]], by_sample[pair["sample_id_b"]]
        assert left.dataset == right.dataset == pair["dataset"]
        assert identity_key(left.dataset, left.subject_id) != identity_key(right.dataset, right.subject_id)
        assert pair["identity_key_hash_a"] != pair["identity_key_hash_b"]
        assert pair["sample_id_a"] < pair["sample_id_b"]


def test_impostor_pairs_never_cross_datasets():
    """A CASIA/MSU pair differs in capture domain as well as identity, which would
    make the impostor problem artificially easy."""
    rows = _population()
    by_sample = {row.sample_id: row for row in rows}
    for pair in build_impostor_pairs(rows, maximum_total=10000):
        assert by_sample[pair["sample_id_a"]].dataset == by_sample[pair["sample_id_b"]].dataset


def test_pairs_are_unique_and_deterministic():
    rows = _population()
    for builder, fields in ((build_genuine_pairs, GENUINE_FIELDS), (build_impostor_pairs, IMPOSTOR_FIELDS)):
        first, second = builder(rows), builder(rows)
        assert pairs_digest(first, fields) == pairs_digest(second, fields)
        assert [row["pair_id"] for row in first] == [row["pair_id"] for row in second]
        assert len({row["pair_id"] for row in first}) == len(first)
        assert len({(row["dataset"], row["sample_id_a"], row["sample_id_b"]) for row in first}) == len(first)
        # a different input order must not change the plan
        assert pairs_digest(builder(list(reversed(rows))), fields) == pairs_digest(first, fields)


def test_impostor_selection_is_balanced_across_datasets_and_identities():
    rows = _population(samples_per_record=3, records=2, identities=4)
    pairs = build_impostor_pairs(rows, maximum_total=200)
    per_dataset = {}
    for pair in pairs: per_dataset[pair["dataset"]] = per_dataset.get(pair["dataset"], 0) + 1
    assert set(per_dataset) == {"casia_fasd", "msu_mfsd"}
    assert abs(per_dataset["casia_fasd"] - per_dataset["msu_mfsd"]) <= 1
    for dataset in per_dataset:
        participation = {}
        for pair in [row for row in pairs if row["dataset"] == dataset]:
            for side in ("identity_key_hash_a", "identity_key_hash_b"):
                participation[pair[side]] = participation.get(pair[side], 0) + 1
        values = sorted(participation.values())
        # round-robin over participation keeps no identity more than 2x the least
        assert values[-1] <= 2 * values[0]


def test_pair_plan_identity_is_logical_not_parquet_bytes(tmp_path):
    rows = _population()
    pairs = build_genuine_pairs(rows)
    logical = pairs_digest(pairs, GENUINE_FIELDS)
    written = write_pairs_parquet(tmp_path / "a.parquet", pairs, GENUINE_FIELDS,
                                  {row["pair_id"]: 0.5 for row in pairs})
    assert written == logical
    # a different cosine payload changes the parquet BYTES but not the plan identity
    other = write_pairs_parquet(tmp_path / "b.parquet", pairs, GENUINE_FIELDS,
                                {row["pair_id"]: 0.9 for row in pairs})
    assert other == logical
    assert (tmp_path / "a.parquet").read_bytes() != (tmp_path / "b.parquet").read_bytes()
    assert write_pairs_parquet(tmp_path / "c.parquet", pairs, GENUINE_FIELDS) == logical


def test_pair_rows_carry_no_raw_path_or_plaintext_subject():
    rows = _population()
    for pairs, fields in ((build_genuine_pairs(rows), GENUINE_FIELDS),
                          (build_impostor_pairs(rows, maximum_total=500), IMPOSTOR_FIELDS)):
        text = json.dumps([{name: row[name] for name in fields} for row in pairs])
        assert ".png" not in text and ".jpg" not in text and "/" not in text.replace("\\/", "")
        assert "subject_id" not in text
        for row in pairs:
            for name in fields:
                if name.startswith("identity_key_hash"): assert len(row[name]) == 64


def test_v2_config_is_source_train_live_only_and_declares_the_rule():
    payload = load_identity_config(V2_CONFIG)
    block = payload["identity_calibration"]
    assert block["version"] == IDENTITY_CALIBRATION_VERSION and block["seed"] == 20260806
    assert block["maximum_genuine_pairs_per_identity"] == 20
    assert block["maximum_impostor_pairs_total"] == 20000
    assert block["genuine_percentile"] == 1.0 and block["impostor_percentile"] == 99.9
    assert block["require_different_source_record"] is True
    assert block["require_same_dataset_for_impostor"] is True
    assert block["require_source_train_live"] is True
    assert payload["threshold_rule"]["tau_id_v2"] == "max(tau_genuine, tau_impostor)"
    assert payload["threshold_rule"]["tuned_on_candidate_acceptance"] is False
    assert payload["calibration_population"]["uses_generated_candidates"] is False
    # the operational minimums are carried over untouched
    assert payload["operational_minimums"] == {"candidates": 1120, "accepted_total": 400,
                                               "accepted_physics": 200, "accepted_gpat": 100,
                                               "accepted_live_casia": 100, "accepted_live_msu": 100,
                                               "require_all_artifact_types": 8, "require_all_regions": 9,
                                               "require_same_and_cross_domain_gpat": True}
    inherited = payload["inherited_from_v1"]
    assert inherited["tau_id_v1"] == TAU_ID_V1
    assert inherited["threshold_sha256_v1"] == THRESHOLD_SHA
    assert inherited["fingerprint_reference_sha256"] == FINGERPRINT_REFERENCE_SHA


def test_v1_quality_gate_config_was_not_modified_by_v2():
    text = (ROOT / "configs" / "synthesis" / "quality_gate_m8.yaml").read_text(encoding="utf-8")
    assert "identity_calibration" not in text
    assert "m8-quality-gate-v1" in text
    assert "benign_identity_cosine_percentile" in text


def test_v2_config_rejects_a_forbidden_split_at_any_depth(tmp_path):
    import yaml
    payload = yaml.safe_load(V2_CONFIG.read_text(encoding="utf-8"))
    payload["calibration_population"]["extra"] = {"nested": {"split": "source_dev"}}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(IdentityCalibrationError):
        load_identity_config(path)


def test_adaface_preprocessing_contract_is_the_pinned_production_wrapper():
    from prism_fas.synthesis.quality_models import PINNED
    identity = preprocessing_contract_identity()
    assert len(identity) == 64
    assert identity == preprocessing_contract_identity()          # pure function of the pin
    spec = PINNED["identity"]
    assert spec["revision"] == "60a65befbcf7"
    assert spec["sha256"] == "43bd2d570584d95d4a17ce81f26449034c45dbeed750afcab651872abc0e1496"
    assert spec["input_size"] == 112 and spec["input_color"] == "bgr" and spec["embedding_dim"] == 512
    # v2 must not introduce an alternate resize/alignment/normalization path: it
    # embeds only through the registry's pinned wrapper.
    source = (ROOT / "src" / "prism_fas" / "synthesis" / "identity_calibration.py").read_text(encoding="utf-8")
    assert "registry.adaface(" in source
    for alternate in ("cv2.resize", "cv2.warpAffine", "F.interpolate", "torch.nn.functional.interpolate",
                      "Resize(", "Normalize("):
        assert alternate not in source, alternate


def test_calibration_never_reads_a_candidate_source_dev_or_target():
    """The forbidden split names appear only in the guard that REJECTS them, so
    the test asserts what the module can open, not which words it contains."""
    import ast
    from prism_fas.synthesis.identity_calibration import SOURCE_SPLIT
    path = ROOT / "src" / "prism_fas" / "synthesis" / "identity_calibration.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert SOURCE_SPLIT == "source_train"
    literals = [node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    # no literal may name a readable artifact of a forbidden split or a candidate
    for text in literals:
        for banned in ("source_dev.parquet", "target_test.parquet", "candidate_plan", "records/",
                       "synthetic_id", "images/", "shards"):
            assert banned not in text, f"{banned!r} in {text!r}"
    # calibration imports nothing from the generation side and never imports modal
    for node in ast.walk(tree):
        names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                 else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
        for name in names:
            assert name.split(".")[0] != "modal"
            assert "synthetic_bank" not in name and "candidate_plan" not in name


def test_threshold_rule_is_the_max_and_percentiles_are_exact():
    from prism_fas.synthesis.identity_calibration import _distribution
    genuine = [0.60, 0.70, 0.80, 0.90]
    impostor = [0.10, 0.20, 0.30, 0.95]
    tau_genuine = float(np.percentile(genuine, 1.0))
    tau_impostor = float(np.percentile(impostor, 99.9))
    assert max(tau_genuine, tau_impostor) == tau_impostor        # impostor floor binds here
    assert _distribution(genuine)["count"] == 4
    assert _distribution([])["count"] if _distribution([]) else True
    report = _report("quality_calibration_v2.json")
    assert report["tau_genuine"] == float(np.percentile(
        [row for row in _v2_cosines("identity_genuine_pairs_v2.parquet")], 1.0))
    assert report["tau_impostor"] == float(np.percentile(
        [row for row in _v2_cosines("identity_impostor_pairs_v2.parquet")], 99.9))
    assert report["tau_id_v2"] == max(report["tau_genuine"], report["tau_impostor"])
    assert report["thresholds"]["tau_id"] == report["tau_id_v2"]


def _v2_cosines(name):
    import pyarrow.parquet as pq
    path = REPORTS / name
    if not path.is_file(): pytest.skip(f"{name} missing; run: prism synthesis identity-calibrate-v2")
    return list(pq.read_table(path).to_pydict()["adaface_cosine"])


def test_calibration_lock_is_deterministic_and_excludes_runtime_metadata():
    lock = _report("IDENTITY_CALIBRATION_V2_LOCK.json")
    calibration = _report("quality_calibration_v2.json")
    payload = load_identity_config(V2_CONFIG)
    rebuilt = build_lock({**calibration, "_genuine_pairs": [], "_impostor_pairs": []},
                         package_identity=lock["package_identity"], config=payload)
    assert rebuilt["calibration_content_identity_sha256"] == lock["calibration_content_identity_sha256"]
    for name in ("created_at", "device_report", "parquet_byte_hashes"):
        assert name in lock["identity_excluded_fields"]
    text = json.dumps(lock)
    assert "/vol/" not in text and "C:\\" not in text and "modal" not in text.lower()
    assert "/tmp" not in text


def test_calibration_comparison_detects_a_drifted_cosine():
    left = {"genuine_pair_plan_identity_sha256": "a", "impostor_pair_plan_identity_sha256": "b",
            "tau_genuine": 0.5, "tau_impostor": 0.4, "tau_id_v2": 0.5, "threshold_sha256": "c",
            "_genuine_scores": {"g": 0.5}, "_impostor_scores": {"i": 0.4}}
    assert compare_calibrations(left, dict(left))["identical"]
    drifted = {**left, "_genuine_scores": {"g": 0.5 + 10 * COSINE_TOLERANCE}}
    result = compare_calibrations(left, drifted)
    assert not result["identical"] and result["mismatch_count"] == 1
    within = {**left, "_genuine_scores": {"g": 0.5 + COSINE_TOLERANCE / 10}}
    assert compare_calibrations(left, within)["identical"]
    assert not compare_calibrations(left, {**left, "tau_id_v2": 0.6})["identical"]


# --- real v2 calibration report contracts ---------------------------------------
def test_real_identity_structure_v2_report_contract():
    report = _report("identity_structure_v2.json")
    assert report["supports_cross_record_identity_calibration"] is True
    assert all(report["requirements"].values())
    assert report["live_count"] == 280
    assert report["live_count_by_dataset"] == {"casia_fasd": 160, "msu_mfsd": 120}
    assert report["identity_count"] == 35
    assert report["identity_count_by_dataset"] == {"casia_fasd": 20, "msu_mfsd": 15}
    assert report["identities_with_at_least_two_source_records"] == 35
    assert report["potential_genuine_pairs"] == 560
    assert report["skipped_samples_without_subject_id"] == 0 and report["skipped_identities"] == []
    assert report["cross_dataset_subject_ids_are_different_people"] is True
    isolation = report["source_isolation"]
    assert isolation["manifests_opened"] == ["manifests/source_train.parquet"]
    assert isolation["source_dev_opened"] is False and isolation["target_test_opened"] is False
    assert isolation["raw_dataset_path_opened"] is False


def test_real_v2_calibration_report_contract():
    report = _report("quality_calibration_v2.json")
    assert report["calibration_version"] == IDENTITY_CALIBRATION_VERSION and report["seed"] == 20260806
    assert report["populations"]["genuine_pairs"] == 560
    assert report["populations"]["impostor_pairs"] == 13440
    assert report["populations"]["impostor_pairs_by_dataset"] == {"casia_fasd": 6720, "msu_mfsd": 6720}
    assert report["genuine_percentile"] == 1.0 and report["impostor_percentile"] == 99.9
    assert report["threshold_rule"] == "tau_id_v2 = max(tau_genuine, tau_impostor)"
    assert report["tau_id_v2"] == max(report["tau_genuine"], report["tau_impostor"])
    assert report["used_generated_candidates"] is False
    assert report["used_source_dev"] is False and report["used_target"] is False
    assert report["v1_tau_id_informational_only"] == TAU_ID_V1
    # every non-identity threshold is carried over from v1 untouched
    assert report["unchanged_from_v1"] == {"tau_fd": 0.5, "tau_lm": 0.002135227532959269,
                                           "tau_parse": 0.8747814437904173, "tau_out": 0.0,
                                           "tau_fp": 5.687657785453908}
    assert report["thresholds"]["tau_fp"] == 5.687657785453908
    assert report["embedding_cache"]["cached_by_absolute_path"] is False
    assert report["embedding_cache"]["forward_passes"] == 280
    assert report["device_report"]["gpu_name"] == "NVIDIA L4"
    models = report["quality_models"]["models"]["identity"]
    assert models["revision"] == "60a65befbcf7" and models["sha256_matches_pin"] is True
    assert 0.0 <= report["impostor_false_match_rate_at_tau"] <= 0.001
    assert report["genuine_acceptance_rate_at_tau"] >= 0.98


def test_real_v2_calibration_determinism_report_contract():
    report = _report("identity_calibration_v2_determinism.json")
    assert report["passed"] is True and report["identical"] is True
    assert report["mismatch_count"] == 0 and report["mismatches"] == []
    assert report["runs"] == 2 and report["cosine_tolerance"] == COSINE_TOLERANCE


def test_real_v2_lock_binds_every_declared_identity():
    lock = _report("IDENTITY_CALIBRATION_V2_LOCK.json")
    for name in ("identity_calibration_lock_schema_version", "calibration_version", "seed",
                 "package_identity", "source_population_sha256", "adaface_revision",
                 "adaface_weight_sha256", "preprocessing_contract_identity_sha256", "config_sha256",
                 "genuine_pair_plan_identity_sha256", "impostor_pair_plan_identity_sha256",
                 "genuine_pairs", "impostor_pairs", "tau_genuine", "tau_impostor", "tau_id_v2",
                 "unchanged_from_v1", "thresholds", "threshold_sha256", "source_isolation",
                 "calibration_content_identity_sha256"):
        assert name in lock, name
    assert lock["package_identity"] == PACKAGE_IDENTITY
    assert lock["adaface_weight_sha256"] == "43bd2d570584d95d4a17ce81f26449034c45dbeed750afcab651872abc0e1496"
    assert lock["tau_id_v2"] == max(lock["tau_genuine"], lock["tau_impostor"])
    assert lock["threshold_sha256"] != THRESHOLD_SHA          # tau_id changed, so the set changed
    assert lock["thresholds"]["tau_id"] == lock["tau_id_v2"]
    assert lock["used_generated_candidates"] is False


def test_v2_identity_gate_is_more_permissive_and_that_is_reported_not_praised():
    """A factual comparison only. v2 is a different calibration population, not a
    better one, and the change must not be justified by acceptance counts."""
    report = _report("quality_calibration_v2.json")
    assert report["tau_id_v2"] < report["v1_tau_id_informational_only"]
    assert report["distributions_are_well_separated"] in (True, False)
    assert isinstance(report["separation_note"], str) and report["separation_note"]
    assert report["genuine_fraction_at_or_below_tau_impostor"] >= 0.0


# --- real v2 re-evaluation report contracts --------------------------------------
def test_real_candidate_plan_v2_reuse_decision_contract():
    """The candidate plan is reused because calibration is not part of candidate
    identity -- established with hash evidence, not by reading."""
    report = _report("candidate_plan_v2_reuse_decision.json")
    assert report["candidate_id_binds_calibration"] is False
    assert report["candidate_plan_lock_fields_binding_calibration"] == []
    assert report["candidate_plan_module_calibration_mentions"] == []
    assert report["rebuilt_rows_match_frozen_plan"] is True
    assert report["candidate_plan_identity_matches_expected"] is True
    assert report["frozen_candidate_plan_identity_sha256"] == CANDIDATE_PLAN_IDENTITY
    assert report["candidate_count"] == 1120 and report["unique_candidate_ids"] == 1120
    assert report["candidate_id_is_deterministic"] is True
    assert report["candidate_id_changes_with_generator_binding"] is True
    # calibration IS bound, but only where a decision change belongs
    assert report["generation_config_sha256_changes_with_calibration"] is True
    assert report["decision"].startswith("reuse the frozen candidate-plan identity")


def test_real_v2_pilot_report_contract():
    report = _report("v2_pilot.json")
    assert report["passed"] is True and report["planned"] == 32
    assert sum(report["terminal_counts"].values()) == 32
    assert report["payload_errors"] == [] and all(report["checks"].values())
    assert report["coverage"]["artifact_type_count"] == 8 and report["coverage"]["region_count"] == 9
    assert report["identity"]["gpat_checkpoint_sha256"] == GPAT_BEST_SHA
    # the v2 gate is bound, and it is not the v1 gate
    assert report["identity"]["threshold_sha256"] == TAU_ID_V2_THRESHOLD_SHA
    assert report["identity"]["threshold_sha256"] != THRESHOLD_SHA
    assert report["identity"]["fingerprint_reference_sha256"] == FINGERPRINT_REFERENCE_SHA
    comparison = report["payload_comparison_against_v1"]
    assert comparison["passed"] is True and comparison["differing"] == 0
    assert comparison["identical"] == comparison["compared"] > 0
    isolation = report["source_isolation"]
    assert isolation["source_dev_opened"] is False and isolation["target_test_opened"] is False


def test_real_v2_pilot_determinism_report_contract():
    report = _report("v2_pilot_determinism.json")
    assert report["passed"] is True and report["identical"] is True
    assert report["candidates"] == 32 and report["compared"] == 32
    assert report["mismatch_count"] == 0 and report["mismatches"] == []


def test_real_v2_resume_audit_report_contract():
    report = _report("v2_resume_audit.json")
    assert report["passed"] is True
    phases = {phase["phase"]: phase for phase in report["phases"]}
    assert set(phases) == {"interrupted", "resumed", "completed_rerun"}
    rerun = phases["completed_rerun"]
    assert rerun["examined"] == 1120 and rerun["reused"] == 1120 and rerun["rebuilt"] == 0
    probe = report["corruption_probe"]
    assert probe["detected_as_unusable"] is True and probe["rebuilt_bytes_identical"] is True
    assert probe["reason"].startswith("hash:")
    assert sum(report["terminal_counts"].values()) == 1120
    assert report["terminal_counts"].get("failed_generation", 0) == 0
    assert all(report["checks"].values())


def test_real_v2_determinism_audit_report_contract():
    report = _report("v2_determinism_audit.json")
    assert report["passed"] is True and report["identical"] is True
    assert report["candidates"] == 32 and report["mismatch_count"] == 0


def test_real_v1_v2_decision_comparison_contract():
    """A factual decision diff. More accepted candidates is not evidence of a
    better bank and says nothing about detector or target performance."""
    report = _report("v1_v2_decision_comparison.json")
    assert report["compared"] == 1120 and report["missing_v1_record"] == 0
    total = (report["unchanged_decisions"] + report["rejected_to_accepted"]
             + report["accepted_to_rejected"])
    assert total == 1120
    # relaxing only the identity gate can never reject something v1 accepted
    assert report["accepted_to_rejected"] == 0
    assert report["rejected_to_accepted"] > 0
    for route in ("physics", "gpat"):
        assert report["per_route"][route]["accepted_to_rejected"] == 0
    # the identity gate no longer appears among the v2 rejection reasons
    assert not any(name.endswith(":identity") for name in report["v2_failed_gate_counts_by_route"])
    assert "not evidence of a better bank" in report["interpretation_note"]


def test_real_v2_bank_validation_report_contract():
    """Every structural invariant holds under v2. The failing checks are the
    pre-declared operational minimums and the lock status they force."""
    report = _report("v2_synthetic_bank_validation.json")
    counts = report["counts"]
    assert counts["candidates"] == 1120
    assert counts["accepted"] + counts["rejected"] + counts["failed_generation"] == 1120
    assert counts["failed_generation"] == 0
    payload = report["payload_report"]
    assert payload["checked"] == counts["accepted"]
    for name in ("missing_files", "hash_mismatches", "image_shape_errors", "mask_value_errors",
                 "npz_errors", "map_range_errors", "map_outside_errors", "mask_pixel_mismatches",
                 "outside_mask_errors"):
        assert payload[name] == 0, name
    assert report["shard_report"]["passed"] is True
    for name in ("terminal_accounting", "no_duplicate_synthetic_ids", "accepted_files_exist",
                 "accepted_hashes_match", "images_decode_rgb_224", "masks_binary_0_255",
                 "artifact_maps_load_without_pickle", "artifact_maps_zero_outside_exact_mask",
                 "saved_outside_mask_error_exactly_zero", "every_accepted_row_passes_every_hard_gate",
                 "every_rejected_row_names_a_failed_gate", "no_target_or_private_fields",
                 "source_only_isolation_evidence", "shards_validate", "candidate_count",
                 "source_package_unchanged", "recipe_bank_unchanged", "gpat_checkpoint_hash_matches"):
        assert report["checks"][name] is True, name
    assert report["coverage"]["artifact_type_count"] == 8 and report["coverage"]["region_count"] == 9
    identities = report["parent_identities"]
    assert identities["candidate_plan_identity"] == CANDIDATE_PLAN_IDENTITY
    assert identities["gpat_checkpoint_sha256"] == GPAT_BEST_SHA
    assert identities["threshold_sha256"] == TAU_ID_V2_THRESHOLD_SHA
    assert identities["fingerprint_reference_sha256"] == FINGERPRINT_REFERENCE_SHA
    assert report["leak_scan"]["hits"] == {}


def test_real_v2_run_missed_only_the_physics_minimum():
    """The honest record: v2 lifted physics acceptance but still misses its
    pre-declared minimum, and no threshold was moved to close the gap."""
    report = _report("v2_synthetic_bank_validation.json")
    minimums = report["operational_minimums"]
    failing = sorted(name for name, passed in minimums["checks"].items() if not passed)
    assert failing == ["accepted_physics"]
    assert minimums["passed"] is False
    declared = minimums["declared_minimums"]
    assert declared["accepted_physics"] == 200 and declared["accepted_total"] == 400
    observed = minimums["observed"]
    assert observed["accepted_route_counts"]["physics"] < declared["accepted_physics"]
    assert observed["accepted_total"] >= declared["accepted_total"]
    assert observed["accepted_route_counts"]["gpat"] >= declared["accepted_gpat"]
    assert observed["accepted_live_target_datasets"]["casia_fasd"] >= declared["accepted_live_casia"]
    assert observed["accepted_live_target_datasets"]["msu_mfsd"] >= declared["accepted_live_msu"]
    assert set(observed["accepted_gpat_domain_relations"]) == {"same_domain", "cross_domain"}


def test_v1_artifacts_were_not_modified_by_v2():
    """v1 is a retained, closed experimental run."""
    v1 = _report("quality_calibration.json")
    assert v1["thresholds"]["tau_id"] == TAU_ID_V1
    assert v1["threshold_sha256"] == THRESHOLD_SHA
    lock = _report("BANK_LOCK_remote.json")
    assert lock["status"] == "operational_minimums_failed"
    assert lock["bank_id"].startswith("prism_synthetic_bank_m8_v1_")
    assert lock["accepted_count"] == 391 and lock["rejected_count"] == 729
    assert lock["threshold_sha256"] == THRESHOLD_SHA
    export = _report("export.json")
    assert export["archive_sha256"] == "9b3a48f220ad9cadd1387a2a4adaffba5d48ea1c72c6fb7f66adef745fb24676"


def test_v2_bank_id_prefix_is_versioned(tmp_path):
    """A bank built under a different calibration policy must not carry the v1
    name."""
    from prism_fas.synthesis.synthetic_bank import BANK_ID_PREFIX
    assert BANK_ID_PREFIX == "prism_synthetic_bank_m8_v1"
    generator, records, default = _build_fixture_bank(tmp_path / "default")
    assert default["bank_id"].startswith("prism_synthetic_bank_m8_v1_")
    generator2, records2, _ = _build_fixture_bank(tmp_path / "v2")
    generator2.bank_id_prefix = "prism_synthetic_bank_m8_v2"
    versioned = assemble_bank(generator2, records2, pairs_root=tmp_path / "v2" / "pairs",
                              build_root=tmp_path / "v2" / "_build2")
    assert versioned["bank_id"].startswith("prism_synthetic_bank_m8_v2_")
    # the prefix is a name, not part of the content identity
    assert versioned["lock"]["bank_content_identity_sha256"] == default["lock"]["bank_content_identity_sha256"]
    text = (ROOT / "modal_m8.py").read_text(encoding="utf-8")
    assert 'bank_id_prefix="prism_synthetic_bank_m8_v2"' in text
