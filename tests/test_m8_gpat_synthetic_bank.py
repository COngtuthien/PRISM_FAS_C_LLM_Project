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
