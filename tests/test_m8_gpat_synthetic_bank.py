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
    difference = (output.synthetic_image - batch.live_image).abs()
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
    expected = sum(DEFAULT_WEIGHTS[name] * float(value) for name, value in result.components.items())
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
