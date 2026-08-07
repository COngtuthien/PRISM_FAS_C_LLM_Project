"""Focused M9 tests: heads, detector, fusion, manifold, losses, data, checkpoint.

Every scientific formula is checked against a hand-computed toy tensor, and every
isolation rule the milestone declares has a test that proves the rule FAILS CLOSED
rather than merely that the happy path works.

These tests never touch the real datasets, the real bank payloads or the network:
the fixtures are tiny synthetic structures with the real shapes and the real
contracts. The real-data evidence lives in the CPU/L4 smokes and the acceptance
report, not here.
"""
from __future__ import annotations
import hashlib, io, json, math, tokenize
from pathlib import Path
import numpy as np
import pytest
import torch

from prism_fas.detector.checkpoint import (M9_CHECKPOINT_SCHEMA_VERSION, STAGE_ORDER,
                                           M9CheckpointError, RunIdentity, StageLineage,
                                           StageTransitionError, apply_checkpoint,
                                           check_stage_transition, check_status_transition,
                                           checkpoint_summary, load_checkpoint, save_checkpoint)
from prism_fas.detector.contracts import (LIVE, REGION_COUNT, REGION_ORDER, SPOOF, DetectorBatch,
                                          DetectorContractError, ModelOutput)
from prism_fas.detector.dataset import (DatasetError, TrainingItem, assert_source_only,
                                        batch_composition, collate_items, domain_composition)
from prism_fas.detector.heads import (GlobalHead, PromptHead, RecipeTextCache, TextCacheError,
                                      cache_identity, content_identity, read_recipe_text_cache,
                                      write_recipe_text_cache)
from prism_fas.detector.losses import (DEFAULT_WEIGHTS, LOSS_NAMES, clean_loss, classification_loss,
                                       compute_losses, consistency_loss, mil_loss, outlier_loss,
                                       prompt_loss, real_manifold_loss, risk_loss,
                                       weighted_local_loss)
from prism_fas.detector.manifold import (DEFAULT_COVARIANCE_EPSILON, ManifoldError, PrototypeState,
                                         RealManifold, deterministic_kmeans, initialize_prototypes,
                                         read_prototypes_npz, write_prototypes_npz)
from prism_fas.detector.pretrained import CONVNEXT_PIN, SIGLIP2_PIN, SIGLIP2_TOKENIZATION
from prism_fas.detector.prism_detector import DetectorConfig, PRISMDetector
from prism_fas.detector.region_cache import PRIOR_STORAGE_SIZE, _downsample
from prism_fas.detector.regions import attack_region_mask, build_region_priors
from prism_fas.detector.sampler import (BatchContract, BatchPlan, M9BatchSampler, SamplerError,
                                        stream_seed)
from prism_fas.detector.synthetic_bank import (FROZEN_BANK_ID, FROZEN_BANK_IDENTITY,
                                               SyntheticBankAccessError, SyntheticBankReader)
from prism_fas.detector.trainer import (SYNTHETIC_TERMS, batch_contract_for, enabled_terms,
                                        M9TrainingConfig)

def code_text(path: Path) -> str:
    """Source with every string literal and comment removed.

    A prose ban belongs in review, not in a test: these tests assert that the CODE
    does not do something, so docstrings that NAME the forbidden thing (to explain
    why it is forbidden) must not trip them.
    """
    source = Path(path).read_text(encoding="utf-8")
    kept: list[str] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in (tokenize.STRING, tokenize.COMMENT): continue
        kept.append(token.string)
    return " ".join(kept)


def detector_modules() -> list[Path]:
    import prism_fas.detector as package
    return sorted(Path(package.__file__).resolve().parent.glob("*.py"))


SEED = 20260806
TEXT_DIM = 16
REGION_DIM = 8
N_PROMPT = 5


# --- fixtures ---------------------------------------------------------------

def _text_matrix(count: int = N_PROMPT, dim: int = TEXT_DIM) -> torch.Tensor:
    generator = torch.Generator().manual_seed(SEED)
    return torch.nn.functional.normalize(torch.randn(count, dim, generator=generator), dim=-1)


def _batch(*, batch_size: int = 4, synthetic: int = 2, prior_size: int = PRIOR_STORAGE_SIZE,
           visibility: float = 1.0) -> DetectorBatch:
    """A structurally real batch: right shapes, right dtypes, right invariants."""
    generator = torch.Generator().manual_seed(SEED)
    is_synthetic = torch.zeros(batch_size, dtype=torch.bool)
    is_synthetic[batch_size - synthetic:] = True
    real_count = batch_size - synthetic
    label = torch.zeros(batch_size, dtype=torch.long)
    label[real_count // 2:real_count] = SPOOF
    label[is_synthetic] = SPOOF
    attack = torch.zeros(batch_size, REGION_COUNT)
    attack[is_synthetic, 0] = 1.0
    attack[is_synthetic, 3] = 1.0
    return DetectorBatch(
        image=torch.rand(batch_size, 3, 224, 224, generator=generator),
        label=label, dataset_id=torch.arange(batch_size) % 2, is_synthetic=is_synthetic,
        region_priors=torch.rand(batch_size, REGION_COUNT, prior_size, prior_size, generator=generator),
        visibility=torch.full((batch_size, REGION_COUNT), float(visibility)),
        sample_ids=tuple(f"s{index}" for index in range(batch_size)),
        datasets=("casia_fasd", "msu_mfsd"),
        artifact_map=torch.rand(batch_size, 1, 224, 224, generator=generator) * is_synthetic.view(-1, 1, 1, 1),
        attack_region_mask=attack,
        quality_weight=torch.where(is_synthetic, torch.full((batch_size,), 0.8), torch.zeros(batch_size)),
        recipe_index=torch.where(is_synthetic, torch.ones(batch_size, dtype=torch.long),
                                 torch.full((batch_size,), -1, dtype=torch.long)),
        artifact_family=tuple("moire" if bool(flag) else "" for flag in is_synthetic)).validate()


def _output(batch: DetectorBatch, *, distances: torch.Tensor | None = None,
            region_dim: int = REGION_DIM) -> ModelOutput:
    size = batch.batch_size
    generator = torch.Generator().manual_seed(SEED + 1)
    d = distances if distances is not None else torch.rand(size, REGION_COUNT, generator=generator)
    p_global = torch.full((size,), 0.4)
    s_region = torch.full((size,), 0.25)
    p_prompt = torch.zeros(size)
    embeddings = torch.randn(size, REGION_COUNT, region_dim, generator=generator)
    # The detector always exposes the PromptHead projection by name; `L_prompt`
    # reads it from there because raw `z_r` does not live in the text space.
    projected = torch.nn.functional.normalize(
        torch.randn(size, REGION_COUNT, TEXT_DIM, generator=generator), dim=-1)
    return ModelOutput(
        global_logit=torch.zeros(size, 1), local_logits=torch.zeros(size, 49),
        region_embeddings=embeddings,
        region_distances=d, region_valid=batch.visibility >= 0.30,
        prompt_logits=torch.zeros(size, N_PROMPT), p_global=p_global, s_region=s_region,
        p_prompt_spoof=p_prompt,
        s_final=1.0 - (1.0 - p_global) * (1.0 - s_region) * (1.0 - p_prompt),
        aux={"prompt_region_embeddings": projected}).validate()


def _detector(**overrides) -> PRISMDetector:
    """The trainable half of the detector: no SigLIP2 tower, no network, no weights."""
    config = DetectorConfig(region_dim=REGION_DIM, region_attention_heads=2,
                            local_pretrained=False, prototype_k=2, **overrides)
    return PRISMDetector(config, text_embeddings=_text_matrix(), text_cache_identity="toy")


# ============================================================================
# HEADS
# ============================================================================

def test_global_head_maps_the_global_embedding_to_one_logit():
    head = GlobalHead(12)
    out = head(torch.randn(5, 12))
    assert out.shape == (5, 1)
    with pytest.raises(DetectorContractError):
        head(torch.randn(5, 11))


def test_prompt_head_shapes_and_bounded_evidence():
    head = PromptHead(REGION_DIM, _text_matrix())
    embeddings = torch.randn(3, REGION_COUNT, REGION_DIM)
    applicable = torch.zeros(3, REGION_COUNT, dtype=torch.bool)
    applicable[1, 2] = True
    out = head(embeddings, applicable)
    assert out["region_prompt_logits"].shape == (3, REGION_COUNT, N_PROMPT)
    assert out["prompt_logits"].shape == (3, N_PROMPT)
    assert out["prompt_region_embeddings"].shape == (3, REGION_COUNT, TEXT_DIM)
    spoof = out["p_prompt_spoof"].detach()
    assert float(spoof.min()) >= 0.0 and float(spoof.max()) <= 1.0
    # Rows with no applicable region contribute exactly nothing.
    assert float(spoof[0]) == 0.0 and float(spoof[2]) == 0.0
    assert float(out["prompt_logits"][0].detach().abs().sum()) == 0.0


def test_prompt_head_projection_is_l2_normalized():
    head = PromptHead(REGION_DIM, _text_matrix())
    projected = head.project_regions(torch.randn(4, REGION_COUNT, REGION_DIM) * 7.0)
    norms = projected.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert torch.allclose(head.text_embeddings.norm(dim=-1), torch.ones(N_PROMPT), atol=1e-5)


def test_prompt_head_applicability_is_synthetic_and_attacked_and_visible():
    is_synthetic = torch.tensor([False, True, True])
    attack = torch.zeros(3, REGION_COUNT); attack[1, 0] = 1.0; attack[2, 0] = 1.0; attack[0, 0] = 1.0
    valid = torch.ones(3, REGION_COUNT, dtype=torch.bool); valid[2, 0] = False
    mask = PromptHead.applicability(is_synthetic, attack, valid)
    assert not bool(mask[0].any())        # real sample: never applicable
    assert bool(mask[1, 0])               # synthetic + attacked + visible
    assert not bool(mask[2, 0])           # attacked but not visible
    assert not bool(mask[1, 1])           # visible but not attacked
    assert not PromptHead.applicability(is_synthetic, None, valid).any()


def test_prompt_head_needs_no_text_encoder_at_inference():
    """The frozen text matrix is a buffer; the fast path makes no network call."""
    head = PromptHead(REGION_DIM, _text_matrix())
    assert "text_embeddings" in dict(head.named_buffers())
    assert not any("text" in name for name, _ in head.named_parameters())
    with torch.inference_mode():
        head(torch.randn(2, REGION_COUNT, REGION_DIM), torch.ones(2, REGION_COUNT, dtype=torch.bool))


def test_recipe_text_cache_identity_binds_the_declared_inputs_only(tmp_path):
    embeddings = torch.nn.functional.normalize(torch.randn(3, TEXT_DIM), dim=-1).numpy()
    binding = {"schema_version": "m9-recipe-text-cache-v1", "recipe_bank_identity_sha256": "bank",
               "model_id": SIGLIP2_PIN["model_id"], "model_revision": SIGLIP2_PIN["revision"],
               "tokenization": dict(sorted(SIGLIP2_TOKENIZATION.items())), "embedding_dim": TEXT_DIM,
               "normalization": "l2",
               "content_identity_sha256": content_identity(embeddings, ("a", "b", "c"))}
    cache = RecipeTextCache(recipe_ids=("a", "b", "c"), embeddings=embeddings,
                            identity=cache_identity(binding), binding=binding).validate()
    path = tmp_path / "cache.npz"
    write_recipe_text_cache(path, cache)
    reloaded = read_recipe_text_cache(path, expected_identity=cache.identity)
    assert reloaded.identity == cache.identity
    assert reloaded.recipe_ids == cache.recipe_ids
    np.testing.assert_allclose(reloaded.embeddings, cache.embeddings)
    # The identity must NOT depend on a path, a machine or a clock.
    for forbidden in ("path", "machine", "hostname", "timestamp", "created_at", "user"):
        assert not any(forbidden in key for key in binding)
    with pytest.raises(TextCacheError):
        read_recipe_text_cache(path, expected_identity="0" * 64)


def test_recipe_text_cache_rejects_content_that_does_not_match_its_binding(tmp_path):
    embeddings = torch.nn.functional.normalize(torch.randn(2, TEXT_DIM), dim=-1).numpy()
    binding = {"content_identity_sha256": content_identity(embeddings, ("a", "b"))}
    cache = RecipeTextCache(recipe_ids=("a", "b"), embeddings=embeddings,
                            identity=cache_identity(binding), binding=binding)
    path = tmp_path / "cache.npz"
    write_recipe_text_cache(path, cache)
    from prism_fas.detector.npz_io import read_arrays_npz, write_arrays_npz
    arrays = read_arrays_npz(path)
    arrays["embeddings"] = np.roll(arrays["embeddings"], 1, axis=0)
    write_arrays_npz(path, arrays)
    with pytest.raises(TextCacheError):
        read_recipe_text_cache(path)


def test_m8_recipe_match_placeholder_is_never_a_prompt_target():
    """M8 recorded `recipe_match = not_applicable` because no PromptHead existed.
    Nothing in the M9 detector may read it."""
    for path in detector_modules():
        assert "recipe_match" not in code_text(path), \
            f"{path.name} reads the M8 recipe_match placeholder"


# ============================================================================
# DETECTOR AND FUSION
# ============================================================================

def test_detector_region_order_matches_m3b_and_m7():
    from prism_fas.data.package.model_priors import VISIBILITY_REGIONS
    from prism_fas.synthesis.masks import REGION_ORDER as M7_ORDER
    assert REGION_ORDER == tuple(VISIBILITY_REGIONS) == tuple(M7_ORDER)
    assert _detector().config.region_order == REGION_ORDER


def test_detector_emits_a_typed_output_with_every_declared_component():
    model = _detector()
    batch = _batch()
    model.attach_global_tower(_FakeTower(model.global_dim, model.global_patch_tokens))
    out = model(batch)
    assert isinstance(out, ModelOutput) and not isinstance(out, tuple)
    assert out.global_logit.shape == (4, 1)
    assert out.region_embeddings.shape == (4, REGION_COUNT, REGION_DIM)
    assert out.region_distances.shape == (4, REGION_COUNT)
    assert out.region_valid.shape == (4, REGION_COUNT)
    assert out.local_logits.shape[0] == 4
    assert out.prompt_logits.shape == (4, N_PROMPT)
    for name in ("p_global", "s_region", "p_prompt_spoof", "s_final"):
        assert getattr(out, name).shape == (4,)
    assert {"entropy", "global_local_disagreement"} <= set(out.confidence_features)
    for name in ("global_embedding", "local_tokens", "normalized_distances",
                 "prompt_region_embeddings", "prompt_applicable", "distance_scale"):
        assert name in out.aux


class _FakeTower(torch.nn.Module):
    """Stands in for the frozen SigLIP2 vision tower: same output contract, no
    weights, no download. The real tower is exercised by the CPU and L4 smokes."""

    def __init__(self, dim: int, tokens: int):
        super().__init__()
        self.dim, self.tokens = int(dim), int(tokens)
        self.scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, pixel_values: torch.Tensor):
        size = pixel_values.shape[0]
        pooled = pixel_values.mean(dim=(1, 2, 3), keepdim=False).view(size, 1).expand(size, self.dim)
        tokens = pooled.unsqueeze(1).expand(size, self.tokens, self.dim)
        return type("Out", (), {"last_hidden_state": tokens.contiguous(),
                                "pooler_output": pooled.contiguous()})()


def test_detector_uses_soft_priors_and_never_crops_nine_images():
    """Spec section 9.1: the prior is a soft mask / query initialization."""
    import prism_fas.detector.prism_detector as module
    source = code_text(Path(module.__file__))
    for forbidden in ("crop", "F.crop", "torchvision"):
        assert forbidden not in source
    model = _detector()
    model.attach_global_tower(_FakeTower(model.global_dim, model.global_patch_tokens))
    batch = _batch()
    out = model(batch)
    # The image enters the local backbone exactly once, whatever the region count.
    assert out.aux["local_tokens"].shape[0] == batch.batch_size


def test_visibility_masks_invalid_regions_out_of_the_region_score():
    model = _detector()
    distances = torch.tensor([[10.0] + [0.0] * (REGION_COUNT - 1)])
    all_valid = torch.ones(1, REGION_COUNT, dtype=torch.bool)
    only_invalid_first = all_valid.clone(); only_invalid_first[0, 0] = False
    high = model.region_score(distances, all_valid)
    low = model.region_score(distances, only_invalid_first)
    assert float(high) > float(low)
    none_valid = torch.zeros(1, REGION_COUNT, dtype=torch.bool)
    assert float(model.region_score(distances, none_valid)) == 0.0


def test_fusion_matches_a_hand_computed_table_34_value():
    p_global = torch.tensor([0.5, 0.2])
    s_region = torch.tensor([0.25, 0.0])
    p_prompt = torch.tensor([0.0, 0.5])
    fused = PRISMDetector.fuse(p_global, s_region, p_prompt)
    # 1 - (1-0.5)(1-0.25)(1-0.0)  = 1 - 0.375  = 0.625
    # 1 - (1-0.2)(1-0.0)(1-0.5)   = 1 - 0.4    = 0.6
    assert float(fused[0]) == pytest.approx(0.625, abs=1e-6)
    assert float(fused[1]) == pytest.approx(0.6, abs=1e-6)


def test_topk_mean_is_the_mean_of_the_two_largest_valid_normalized_distances():
    model = _detector()
    model.set_distance_scale(torch.ones(REGION_COUNT), freeze=True)
    distances = torch.zeros(1, REGION_COUNT)
    distances[0, 0], distances[0, 1], distances[0, 2] = 4.0, 1.0, 2.0
    valid = torch.ones(1, REGION_COUNT, dtype=torch.bool)
    # normalize(d) = 1 - exp(-d); top-2 are d = 4 and d = 2.
    expected = ((1 - math.exp(-4.0)) + (1 - math.exp(-2.0))) / 2.0
    assert float(model.region_score(distances, valid)) == pytest.approx(expected, abs=1e-6)


def test_confidence_features_are_entropy_and_global_local_disagreement():
    p_global, s_region = torch.tensor([0.5]), torch.tensor([0.1])
    s_final = torch.tensor([0.5])
    features = PRISMDetector.confidence(p_global, s_region, s_final)
    assert float(features["entropy"][0]) == pytest.approx(1.0, abs=1e-5)     # maximal at p = 0.5
    assert float(features["global_local_disagreement"][0]) == pytest.approx(0.4, abs=1e-6)


def test_frozen_global_tower_is_not_in_the_trainable_state():
    model = _detector()
    tower = _FakeTower(model.global_dim, model.global_patch_tokens)
    model.attach_global_tower(tower)
    assert not any(key.startswith("_global") or "global_tower" in key for key in model.state_dict())
    groups = model.parameter_groups(backbone_lr=1e-5, head_lr=1e-4, weight_decay=0.05)
    assert [group["name"] for group in groups] == ["backbone", "heads"]
    owned = {id(parameter) for parameter in model.parameters()}
    assert id(tower.scale) not in owned


def test_architecture_identity_changes_when_a_declared_choice_changes():
    base = _detector().architecture_identity()
    assert _detector().architecture_identity() == base
    assert _detector(visibility_threshold=0.5).architecture_identity() != base
    assert _detector(distance_scale_convention="sum").architecture_identity() != base


# ============================================================================
# MANIFOLD
# ============================================================================

def test_manifold_holds_k_equals_four_prototypes_per_region_by_default():
    manifold = RealManifold(REGION_DIM)
    assert manifold.k == 4 and manifold.regions == REGION_COUNT
    assert manifold.centers.shape == (REGION_COUNT, 4, REGION_DIM)


def test_deterministic_kmeans_repeats_exactly():
    points = np.random.default_rng(SEED).normal(size=(40, 5))
    first = deterministic_kmeans(points, 3, seed=SEED)
    second = deterministic_kmeans(points, 3, seed=SEED)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_prototype_initialization_floors_the_variance_at_epsilon():
    embeddings = np.zeros((10, REGION_COUNT, 3))
    valid = np.ones((10, REGION_COUNT), dtype=bool)
    state = initialize_prototypes(embeddings, valid, k=2, epsilon=DEFAULT_COVARIANCE_EPSILON)
    assert float(state.variances.min()) >= DEFAULT_COVARIANCE_EPSILON - 1e-12


def test_prototype_initialization_refuses_a_region_with_fewer_than_k_valid_samples():
    embeddings = np.random.default_rng(SEED).normal(size=(6, REGION_COUNT, 3))
    valid = np.ones((6, REGION_COUNT), dtype=bool)
    valid[:, 2] = False
    valid[0, 2] = True
    with pytest.raises(ManifoldError, match="fewer than k"):
        initialize_prototypes(embeddings, valid, k=4)


def _initialized_manifold(dim: int = 3) -> RealManifold:
    manifold = RealManifold(dim, k=2)
    embeddings = np.random.default_rng(SEED).normal(size=(12, REGION_COUNT, dim))
    manifold.load_state(initialize_prototypes(embeddings, np.ones((12, REGION_COUNT), dtype=bool), k=2))
    return manifold


def test_only_live_non_synthetic_visible_samples_move_a_prototype():
    manifold = _initialized_manifold()
    before = manifold.centers.clone()
    embeddings = torch.randn(4, REGION_COUNT, 3) * 5.0
    valid = torch.ones(4, REGION_COUNT, dtype=torch.bool)

    # A real SPOOF sample must change nothing.
    manifold.update(embeddings, is_live=torch.zeros(4, dtype=torch.bool),
                    is_synthetic=torch.zeros(4, dtype=torch.bool), region_valid=valid)
    assert torch.equal(manifold.centers, before)

    # A SYNTHETIC sample must change nothing, even labelled live.
    manifold.update(embeddings, is_live=torch.ones(4, dtype=torch.bool),
                    is_synthetic=torch.ones(4, dtype=torch.bool), region_valid=valid)
    assert torch.equal(manifold.centers, before)

    # An INVALID region must change nothing.
    manifold.update(embeddings, is_live=torch.ones(4, dtype=torch.bool),
                    is_synthetic=torch.zeros(4, dtype=torch.bool),
                    region_valid=torch.zeros(4, REGION_COUNT, dtype=torch.bool))
    assert torch.equal(manifold.centers, before)

    # A live, real, visible sample does move it.
    manifold.update(embeddings, is_live=torch.ones(4, dtype=torch.bool),
                    is_synthetic=torch.zeros(4, dtype=torch.bool), region_valid=valid)
    assert not torch.equal(manifold.centers, before)


def test_prototype_update_never_carries_a_gradient():
    manifold = _initialized_manifold()
    embeddings = torch.randn(3, REGION_COUNT, 3, requires_grad=True)
    manifold.update(embeddings, is_live=torch.ones(3, dtype=torch.bool),
                    is_synthetic=torch.zeros(3, dtype=torch.bool),
                    region_valid=torch.ones(3, REGION_COUNT, dtype=torch.bool))
    assert not manifold.centers.requires_grad and manifold.centers.grad is None
    assert embeddings.grad is None


def test_distance_is_the_declared_diagonal_mahalanobis_per_dimension():
    manifold = RealManifold(2, k=1)
    manifold.load_state(PrototypeState(centers=np.zeros((REGION_COUNT, 1, 2)),
                                       variances=np.ones((REGION_COUNT, 1, 2)),
                                       counts=np.ones((REGION_COUNT, 1), dtype=np.int64),
                                       valid=np.ones((REGION_COUNT, 1), dtype=bool), epsilon=1e-4))
    embeddings = torch.zeros(1, REGION_COUNT, 2)
    embeddings[0, 0] = torch.tensor([3.0, 4.0])
    # (9 + 16) / D = 25 / 2 = 12.5 under the declared per-dimension convention.
    assert float(manifold.distance(embeddings)[0, 0]) == pytest.approx(12.5, abs=1e-6)
    raw = RealManifold(2, k=1, distance_scale_convention="sum")
    raw.load_state(manifold.export_state())
    assert float(raw.distance(embeddings)[0, 0]) == pytest.approx(25.0, abs=1e-6)


def test_uninitialized_manifold_contributes_zero_rather_than_noise():
    manifold = RealManifold(3)
    embeddings = torch.randn(2, REGION_COUNT, 3)
    assert float(manifold.distance(embeddings).abs().sum()) == 0.0
    assert float(manifold.soft_min(embeddings).abs().sum()) == 0.0


def test_prototypes_npz_roundtrips_without_pickle(tmp_path):
    state = _initialized_manifold().export_state()
    path = tmp_path / "prototypes.npz"
    export = write_prototypes_npz(path, state, config_hash="cfg", population_identity="pop",
                                  feature_identity="feat")
    with np.load(path, allow_pickle=False) as handle:
        assert "centers" in handle
    reloaded, meta = read_prototypes_npz(path)
    np.testing.assert_allclose(reloaded.centers, state.centers)
    np.testing.assert_allclose(reloaded.variances, state.variances)
    assert meta["prototype_identity_sha256"] == export["prototype_identity_sha256"]
    assert tuple(reloaded.region_names) == REGION_ORDER
    # Deterministic bytes: a second export of the same state is byte-identical.
    second = tmp_path / "prototypes2.npz"
    assert write_prototypes_npz(second, state, config_hash="cfg", population_identity="pop",
                                feature_identity="feat")["file_sha256"] == export["file_sha256"]


def test_prototype_identity_depends_on_the_population_and_the_config():
    state = _initialized_manifold().export_state()
    base = state.identity(config_hash="cfg", population_identity="pop")
    assert state.identity(config_hash="cfg", population_identity="pop") == base
    assert state.identity(config_hash="other", population_identity="pop") != base
    assert state.identity(config_hash="cfg", population_identity="other") != base


# ============================================================================
# LOSSES
# ============================================================================

def test_classification_loss_matches_a_hand_computed_bce():
    logit = torch.tensor([[0.0], [2.0]])
    label = torch.tensor([0, 1])
    selector = torch.ones(2, dtype=torch.bool)
    expected = (math.log(2.0) + math.log(1 + math.exp(-2.0))) / 2.0
    assert float(classification_loss(logit, label, selector)) == pytest.approx(expected, abs=1e-6)


def test_outlier_loss_matches_the_hand_computed_hinge():
    distance = torch.tensor([[1.0, 5.0, 0.0]])
    attack = torch.tensor([[1.0, 1.0, 0.0]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    # max(0, 3-1) + max(0, 3-5) = 2 + 0 = 2, mean over 1 synthetic sample.
    value = outlier_loss(distance, attack, torch.ones(1, dtype=torch.bool), valid, margin=3.0)
    assert float(value) == pytest.approx(2.0, abs=1e-6)


def test_clean_loss_matches_the_hand_computed_cap():
    distance = torch.tensor([[1.0, 5.0, 2.0]])
    attack = torch.tensor([[1.0, 0.0, 0.0]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    # (1-m) * min(d, 3) over the two clean regions = min(5,3) + min(2,3) = 3 + 2 = 5.
    value = clean_loss(distance, attack, torch.ones(1, dtype=torch.bool), valid, clean_cap=3.0)
    assert float(value) == pytest.approx(5.0, abs=1e-6)


def test_real_manifold_loss_sums_over_valid_regions_only():
    soft_min = torch.tensor([[1.0, 2.0, 3.0]])
    valid = torch.tensor([[True, False, True]])
    value = real_manifold_loss(soft_min, torch.ones(1, dtype=torch.bool), valid)
    assert float(value) == pytest.approx(4.0, abs=1e-6)


def test_consistency_loss_matches_the_hand_computed_symmetric_l1():
    p_global = torch.tensor([0.7]); s_region = torch.tensor([0.2])
    assert float(consistency_loss(p_global, s_region)) == pytest.approx(1.0, abs=1e-6)


def test_mil_loss_uses_logsumexp_pooling():
    logits = torch.tensor([[0.0, 0.0]])
    label = torch.tensor([1])
    pooled = math.log(2.0)
    expected = math.log(1 + math.exp(-pooled))
    value = mil_loss(logits, label, torch.ones(1, dtype=torch.bool), temperature=1.0)
    assert float(value) == pytest.approx(expected, abs=1e-6)


def test_risk_loss_is_a_finite_zero_with_fewer_than_two_groups():
    per_sample = torch.tensor([1.0, 2.0, 3.0])
    one_group = torch.zeros(3, dtype=torch.long)
    assert float(risk_loss(per_sample, one_group, torch.full((3,), -1, dtype=torch.long))) == 0.0
    two_groups = torch.tensor([0, 0, 1])
    # group means 1.5 and 3.0 -> population variance of [1.5, 3.0] = 0.5625
    value = risk_loss(per_sample, two_groups, torch.full((3,), -1, dtype=torch.long))
    assert float(value) == pytest.approx(0.5625, abs=1e-6)


def test_local_loss_is_a_finite_zero_when_the_artifact_map_is_empty():
    logits = torch.zeros(1, 4)
    empty = torch.zeros(1, 1, 224, 224)
    value = weighted_local_loss(logits, empty, torch.ones(1, dtype=torch.bool))
    assert float(value) == 0.0 and math.isfinite(float(value))


def test_every_loss_is_a_finite_zero_when_nothing_is_applicable():
    batch = _batch(synthetic=0)
    out = _output(batch)
    manifold = RealManifold(REGION_DIM)
    result = compute_losses(out, batch, manifold, text_embeddings=_text_matrix())
    for name in ("L_cls_syn", "L_local", "L_out", "L_clean", "L_prompt"):
        assert float(result.terms[name]) == 0.0, name
    assert math.isfinite(float(result.total))
    assert set(result.terms) == set(LOSS_NAMES)


def test_prompt_loss_ignores_real_samples_and_invisible_regions():
    embeddings = torch.nn.functional.normalize(torch.randn(2, REGION_COUNT, TEXT_DIM), dim=-1)
    attack = torch.zeros(2, REGION_COUNT); attack[:, 0] = 1.0
    valid = torch.ones(2, REGION_COUNT, dtype=torch.bool)
    real_only = prompt_loss(embeddings, _text_matrix(), torch.tensor([-1, -1]), attack,
                            torch.zeros(2, dtype=torch.bool), valid)
    assert float(real_only) == 0.0
    invisible = prompt_loss(embeddings, _text_matrix(), torch.tensor([0, 1]), attack,
                            torch.ones(2, dtype=torch.bool), torch.zeros(2, REGION_COUNT, dtype=torch.bool))
    assert float(invisible) == 0.0


def test_q_scales_only_the_declared_synthetic_bracket():
    """Changing q must change the synthetic contribution and nothing else."""
    batch = _batch()
    out = _output(batch)
    manifold = _initialized_manifold(dim=REGION_DIM)
    kwargs = {"text_embeddings": _text_matrix()}
    low = compute_losses(out, batch, manifold, **kwargs)
    high_q = DetectorBatch(**{**vars(batch), "quality_weight": torch.where(
        batch.is_synthetic, torch.full((batch.batch_size,), 1.0), torch.zeros(batch.batch_size))})
    high = compute_losses(out, high_q, manifold, **kwargs)

    unchanged = ("L_cls_real", "L_real", "L_clean", "L_MIL", "L_prompt", "L_cons", "L_risk",
                 "L_cls_syn", "L_local", "L_out")
    for name in unchanged:
        assert float(low.terms[name]) == pytest.approx(float(high.terms[name]), abs=1e-9), name
    assert torch.equal(batch.label, high_q.label)

    lam = DEFAULT_WEIGHTS
    bracket = (float(low.terms["L_cls_syn"]) + lam["lambda_local"] * float(low.terms["L_local"])
               + lam["lambda_out"] * float(low.terms["L_out"]))
    delta = lam["lambda_syn"] * (1.0 - 0.8) * bracket
    assert float(high.total) - float(low.total) == pytest.approx(delta, abs=1e-5)


def test_q_is_never_turned_into_a_label():
    for path in detector_modules():
        source = code_text(path).replace(" ", "")
        for forbidden in ("label=q", "label=quality_weight", "class_target=q",
                          "label=batch.quality_weight", "y=q"):
            assert forbidden not in source, f"{path.name} derives a label from q"


def test_loss_result_logs_every_term_independently_of_the_total():
    batch = _batch()
    out = _output(batch)
    result = compute_losses(out, batch, RealManifold(REGION_DIM), text_embeddings=_text_matrix())
    metrics = result.metrics()
    for name in LOSS_NAMES:
        assert name in metrics
    assert "L_total" in metrics
    assert any(key.startswith("applicable/") for key in metrics)


# ============================================================================
# DATA: BANK, DATASET, SAMPLER
# ============================================================================

def _write_bank(root: Path, *, identity: str = FROZEN_BANK_IDENTITY,
                bank_id: str = FROZEN_BANK_ID, status: str = "validated",
                accepted: int = 0) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifests").mkdir(exist_ok=True)
    (root / "BANK_LOCK.json").write_text(json.dumps({
        "bank_id": bank_id, "status": status, "bank_content_identity_sha256": identity,
        "accepted_count": accepted, "package_identity": "pkg", "recipe_bank_identity": "rec"}),
        encoding="utf-8")
    from prism_fas.synthesis.synthetic_bank import MANIFEST_SCHEMAS, write_manifest
    write_manifest(root / "manifests" / "manifest.parquet", [], MANIFEST_SCHEMAS["manifest"])
    return root


def test_bank_reader_rejects_a_mismatched_content_identity(tmp_path):
    root = _write_bank(tmp_path / "other", identity="a" * 64)
    with pytest.raises(SyntheticBankAccessError, match="content identity"):
        SyntheticBankReader.open(root)


def test_bank_reader_rejects_a_bank_that_is_not_validated(tmp_path):
    root = _write_bank(tmp_path / "pending", status="operational_minimums_failed")
    with pytest.raises(SyntheticBankAccessError, match="status"):
        SyntheticBankReader.open(root)


def test_bank_reader_rejects_the_v1_and_v2_bank_ids(tmp_path):
    for name in ("prism_synthetic_bank_m8_v1_ef6a76ed46f0", "prism_synthetic_bank_m8_v2_deadbeef0000"):
        root = _write_bank(tmp_path / name, bank_id=name)
        with pytest.raises(SyntheticBankAccessError, match="bank id"):
            SyntheticBankReader.open(root)


def test_bank_reader_rejects_a_lock_whose_accepted_count_disagrees(tmp_path):
    root = _write_bank(tmp_path / "counts", accepted=871)
    with pytest.raises(SyntheticBankAccessError, match="accepted"):
        SyntheticBankReader.open(root)


def test_bank_reader_only_ever_opens_the_accepted_manifest():
    import prism_fas.detector.synthetic_bank as module
    path = Path(module.__file__)
    code = code_text(path)
    # Only the accepted manifest is reachable from code; the strings the reader
    # actually opens live in the raw source.
    assert "rejected.parquet" not in code and "candidate_manifest" not in code
    assert '"manifests" / "manifest.parquet"' in path.read_text(encoding="utf-8")


def test_dataset_refuses_to_open_the_target_split():
    from prism_fas.detector.contracts import TargetIsolationViolation
    for split in ("target_test", "target_test_features", "siw_target"):
        with pytest.raises(TargetIsolationViolation):
            assert_source_only(split)
    assert assert_source_only("source_train") == "source_train"
    assert assert_source_only("source_dev") == "source_dev"


def test_batch_contract_rejects_a_composition_the_spec_forbids():
    BatchContract().validate()
    with pytest.raises(SamplerError):
        BatchContract(real_live=0).validate()                      # every batch needs real live
    with pytest.raises(SamplerError):
        BatchContract(real_live=8, real_spoof=8, synthetic=16).validate()   # synthetic > 25 %
    with pytest.raises(SamplerError):
        BatchContract(real_live=11, real_spoof=13).validate()       # not domain-divisible
    with pytest.raises(SamplerError):
        BatchContract(synthetic=0).validate()                       # mixed training needs synthetic
    BatchContract(real_live=16, real_spoof=16, synthetic=0, phase="real_only").validate()


def _sampler(**overrides) -> M9BatchSampler:
    live = {"casia_fasd": list(range(0, 40)), "msu_mfsd": list(range(40, 70))}
    spoof = {"casia_fasd": list(range(70, 170)), "msu_mfsd": list(range(170, 220))}
    routes = {"physics": list(range(0, 30)), "gpat": list(range(30, 65))}
    kwargs = {"real_live": live, "real_spoof": spoof, "synthetic_routes": routes,
              "contract": BatchContract(), "seed": SEED, "steps_per_epoch": 6, "identity": "pkg"}
    kwargs.update(overrides)
    return M9BatchSampler(**kwargs)


def test_sampler_produces_the_exact_12_12_8_composition_every_batch():
    for plan in _sampler().epoch_plans(0):
        assert len(plan.real_live) == 12 and len(plan.real_spoof) == 12 and len(plan.synthetic) == 8
        assert plan.size == 32


def test_sampler_balances_the_declared_domains_on_both_real_partitions():
    sampler = _sampler()
    live_pool = {index: name for name, values in sampler.real_live_pools.items() for index in values}
    spoof_pool = {index: name for name, values in sampler.real_spoof_pools.items() for index in values}
    for plan in sampler.epoch_plans(0):
        live_counts = {name: 0 for name in ("casia_fasd", "msu_mfsd")}
        for index in plan.real_live: live_counts[live_pool[index]] += 1
        spoof_counts = {name: 0 for name in ("casia_fasd", "msu_mfsd")}
        for index in plan.real_spoof: spoof_counts[spoof_pool[index]] += 1
        assert live_counts == {"casia_fasd": 6, "msu_mfsd": 6}
        assert spoof_counts == {"casia_fasd": 6, "msu_mfsd": 6}


def test_sampler_keeps_both_synthetic_routes_present():
    sampler = _sampler()
    routes = {index: name for name, values in sampler.synthetic_pools.items() for index in values}
    for plan in sampler.epoch_plans(0):
        assert {routes[index] for index in plan.synthetic} == {"physics", "gpat"}


def test_sampler_replays_deterministically_and_changes_with_the_epoch():
    first, second = _sampler(), _sampler()
    assert first.fingerprint(0) == second.fingerprint(0)
    assert first.fingerprint(0) != first.fingerprint(1)
    assert _sampler(seed=SEED + 1).fingerprint(0) != first.fingerprint(0)


def test_sampler_does_not_depend_on_pythonhashseed():
    """Seeds come from SHA-256, never from Python's randomized `hash()`."""
    import prism_fas.detector.sampler as module
    source = code_text(Path(module.__file__))
    assert " hash (" not in source and "= hash" not in source
    assert stream_seed("a", 1) == stream_seed("a", 1)
    assert stream_seed("a", 1) == int.from_bytes(hashlib.sha256(b"a|1").digest()[:8], "big")


def test_sampler_resume_continues_the_same_sequence():
    sampler = _sampler()
    full = sampler.epoch_plans(0)
    resumed = list(sampler.iter_epoch(0, start_step=3))
    assert [plan.real_live for plan in resumed] == [plan.real_live for plan in full[3:]]
    assert [plan.synthetic for plan in resumed] == [plan.synthetic for plan in full[3:]]


def test_gradient_accumulation_preserves_the_exact_effective_composition():
    plan = _sampler(contract=BatchContract(accumulation_steps=4)).epoch_plans(0)[0]
    micro = plan.microbatches(4)
    assert len(micro) == 4
    for part in micro:
        assert len(part.real_live) == 3 and len(part.real_spoof) == 3 and len(part.synthetic) == 2
    assert sum(len(part.real_live) for part in micro) == 12
    assert sum(len(part.synthetic) for part in micro) == 8
    with pytest.raises(SamplerError):
        BatchContract(accumulation_steps=3).validate()


def test_warmup_stage_uses_a_real_only_batch_and_g5_uses_12_12_8():
    config = M9TrainingConfig()
    for stage in ("G1", "G2"):
        contract = batch_contract_for(stage, config).validate()
        assert contract.synthetic == 0 and contract.batch_size == 32 and contract.phase == "real_only"
    mixed = batch_contract_for("G5", config).validate()
    assert (mixed.real_live, mixed.real_spoof, mixed.synthetic) == (12, 12, 8)


def test_collate_rejects_a_real_row_that_carries_an_attack_mask():
    item = TrainingItem(sample_id="r", kind="real_live", image=np.zeros((3, 224, 224), np.float32),
                        label=LIVE, dataset="casia_fasd",
                        region_priors=np.zeros((REGION_COUNT, 8, 8), np.float32),
                        visibility=np.ones(REGION_COUNT, np.float32), is_synthetic=False,
                        attack_region_mask=np.ones(REGION_COUNT, np.float32))
    with pytest.raises(DetectorContractError, match="attack region mask"):
        collate_items([item])


def test_collate_reports_the_declared_composition():
    batch = _batch(batch_size=4, synthetic=2)
    assert batch_composition(batch) == {"real_live": 1, "real_spoof": 1, "synthetic_spoof": 2}
    assert batch_composition(_batch(batch_size=8, synthetic=2)) == {
        "real_live": 3, "real_spoof": 3, "synthetic_spoof": 2}
    assert set(domain_composition(batch)) <= {"real_live", "real_spoof", "synthetic_spoof"}


def test_region_prior_downsampling_is_an_exact_area_reduction():
    priors = np.random.default_rng(SEED).random((REGION_COUNT, 224, 224)).astype(np.float32)
    small = _downsample(priors, PRIOR_STORAGE_SIZE)
    assert small.shape == (REGION_COUNT, PRIOR_STORAGE_SIZE, PRIOR_STORAGE_SIZE)
    # 224 -> 56 -> 14 must equal 224 -> 14 for area averaging.
    direct = priors.reshape(REGION_COUNT, 14, 16, 14, 16).mean(axis=(2, 4))
    staged = small.reshape(REGION_COUNT, 14, 4, 14, 4).mean(axis=(2, 4))
    np.testing.assert_allclose(direct, staged, atol=1e-5)


def test_attack_region_mask_comes_from_the_exact_mask_not_from_q():
    priors = np.zeros((REGION_COUNT, 8, 8), dtype=np.float32)
    priors[0, :4, :] = 1.0
    priors[1, 4:, :] = 1.0
    exact = np.zeros((8, 8), dtype=bool)
    exact[:4, :] = True
    mask = attack_region_mask(exact, priors)
    assert float(mask[0]) == 1.0 and float(mask[1]) == 0.0
    assert float(attack_region_mask(np.zeros((8, 8), dtype=bool), priors).sum()) == 0.0


# ============================================================================
# CHECKPOINT AND STAGES
# ============================================================================

def _identity(**overrides) -> RunIdentity:
    base = {name: f"{name}-sha" for name in
            ("source_package_identity", "m8_bank_identity", "architecture_identity",
             "siglip2_identity", "recipe_text_cache_identity", "config_hash",
             "loss_contract_hash", "batch_contract_hash", "dataset_contract_identity")}
    base.update(overrides)
    return RunIdentity(**base)


def _save(tmp_path: Path, model: PRISMDetector, *, identity: RunIdentity, stage: str = "G1",
          epoch: int = 1, step: int = 7) -> Path:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    path = tmp_path / "last.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, scheduler=None, scaler=None,
                    epoch=epoch, global_step=step, stage=stage, identity=identity,
                    sampler_state={"epoch": epoch, "step": 3, "seed": SEED},
                    prototype_identity="proto-sha", best_metrics={"source_dev/acer": 0.2},
                    stage_lineage=[{"stage": "G1", "status": "RUNNING"}], git_sha="deadbeef")
    return path


def test_checkpoint_roundtrips_every_declared_field(tmp_path):
    model = _detector()
    path = _save(tmp_path, model, identity=_identity())
    summary = checkpoint_summary(path)
    assert summary["schema_version"] == M9_CHECKPOINT_SCHEMA_VERSION
    assert summary["stage"] == "G1" and summary["global_step"] == 7
    assert summary["prototype_identity"] == "proto-sha"
    assert summary["best_metrics"] == {"source_dev/acer": 0.2}
    assert summary["rng_state_present"] and summary["git_commit"] == "deadbeef"
    assert summary["ema_enabled"] is False
    assert any(key.startswith("global_head.") for key in summary["global_head_keys"])
    assert any(key.startswith("prompt_head.") for key in summary["prompt_head_keys"])
    assert summary["sampler_state"]["step"] == 3


@pytest.mark.parametrize("field", [
    "source_package_identity", "m8_bank_identity", "architecture_identity", "siglip2_identity",
    "recipe_text_cache_identity", "config_hash", "loss_contract_hash", "batch_contract_hash",
    "dataset_contract_identity"])
def test_strict_resume_rejects_every_mismatched_scientific_identity(tmp_path, field):
    model = _detector()
    path = _save(tmp_path, model, identity=_identity())
    with pytest.raises(M9CheckpointError, match="identity mismatch"):
        load_checkpoint(path, expected_identity=_identity(**{field: "different"}))
    # The matching identity still loads, so the guard is specific, not blanket.
    load_checkpoint(path, expected_identity=_identity())


def test_strict_resume_rejects_a_wrong_schema_version(tmp_path):
    model = _detector()
    path = _save(tmp_path, model, identity=_identity())
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["schema_version"] = "m9-detector-checkpoint-v0"
    torch.save(payload, path)
    with pytest.raises(M9CheckpointError, match="schema"):
        load_checkpoint(path, expected_identity=_identity())


def test_strict_resume_rejects_a_stage_mismatch(tmp_path):
    model = _detector()
    path = _save(tmp_path, model, identity=_identity(), stage="G5")
    with pytest.raises(M9CheckpointError, match="stage"):
        load_checkpoint(path, expected_identity=_identity(), expected_stage="G2")
    load_checkpoint(path, expected_identity=_identity(), expected_stage="G5")


def test_strict_resume_rejects_a_checkpoint_missing_identity_fields(tmp_path):
    model = _detector()
    path = _save(tmp_path, model, identity=_identity())
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["identity"].pop("siglip2_identity")
    torch.save(payload, path)
    with pytest.raises(M9CheckpointError, match="missing identity"):
        load_checkpoint(path, expected_identity=_identity())


def test_there_is_no_strict_false_fallback_for_a_scientific_checkpoint():
    import prism_fas.detector.checkpoint as module
    source = code_text(Path(module.__file__)).replace(" ", "")
    assert "strict=False" not in source
    assert "strict=True" in source


def test_resume_restores_rng_sampler_and_prototype_state(tmp_path):
    model = _detector()
    manifold_state = _initialized_manifold(dim=REGION_DIM)
    model.manifold.load_state(manifold_state.export_state())
    path = _save(tmp_path, model, identity=_identity(), stage="G5", epoch=4, step=180)
    fresh = _detector()
    assert not bool(fresh.manifold.initialized)
    torch.manual_seed(0)
    payload = load_checkpoint(path, expected_identity=_identity())
    restored = apply_checkpoint(payload, model=fresh)
    assert restored["global_step"] == 180 and restored["epoch"] == 4 and restored["stage"] == "G5"
    assert restored["sampler_state"]["step"] == 3
    assert bool(fresh.manifold.initialized)
    torch.testing.assert_close(fresh.manifold.centers, model.manifold.centers)
    torch.testing.assert_close(fresh.manifold.variances, model.manifold.variances)


def test_m9_stage_flow_is_g1_g2_g5_g6_and_refuses_anything_else():
    assert STAGE_ORDER == ("G1", "G2", "G5", "G6")
    assert check_stage_transition(None, "G1") == "G1"
    assert check_stage_transition("G1", "G2") == "G2"
    assert check_stage_transition("G2", "G5") == "G5"
    assert check_stage_transition("G5", "G6") == "G6"
    with pytest.raises(StageTransitionError): check_stage_transition(None, "G5")
    with pytest.raises(StageTransitionError): check_stage_transition("G1", "G5")   # no skipping
    with pytest.raises(StageTransitionError): check_stage_transition("G5", "G2")   # no going back
    for stage in ("G0", "G3", "G4", "G7", "G8"):
        with pytest.raises(StageTransitionError): check_stage_transition("G5", stage)


def test_run_status_machine_matches_table_40():
    assert check_status_transition("PENDING", "RUNNING") == "RUNNING"
    assert check_status_transition("RUNNING", "INTERRUPTED") == "INTERRUPTED"
    assert check_status_transition("INTERRUPTED", "RESUMING") == "RESUMING"
    assert check_status_transition("RESUMING", "RUNNING") == "RUNNING"
    assert check_status_transition("FAILED", "RESUMING", recoverable=True) == "RESUMING"
    with pytest.raises(StageTransitionError):
        check_status_transition("FAILED", "RESUMING")               # unrecoverable
    with pytest.raises(StageTransitionError):
        check_status_transition("COMPLETED", "RUNNING")


def test_stage_lineage_records_input_and_output_hashes():
    lineage = StageLineage()
    lineage.enter("G1", input_hashes={"package": "pkg"})
    lineage.complete("G1", output_hashes={"checkpoint": "sha"})
    lineage.enter("G2", input_hashes={"live_population": "pop"})
    assert lineage.current == "G2"
    assert lineage.payload()[0]["status"] == "COMPLETED"
    assert lineage.payload()[0]["output_hashes"] == {"checkpoint": "sha"}
    with pytest.raises(StageTransitionError):
        lineage.complete("G5", output_hashes={})
    assert StageLineage.from_payload(lineage.payload()).identity() == lineage.identity()


# ============================================================================
# STAGE LOSS ENABLEMENT AND PINS
# ============================================================================

def test_g1_switches_off_the_manifold_and_synthetic_terms():
    g1 = enabled_terms("G1")
    for name in ("L_real", "L_out", "L_clean", *SYNTHETIC_TERMS): assert g1[name] is False, name
    for name in ("L_cls_real", "L_MIL", "L_cons", "L_risk"): assert g1[name] is True, name
    g2 = enabled_terms("G2")
    assert g2["L_real"] is True and g2["L_cls_syn"] is False
    assert all(enabled_terms("G5").values())


def test_g6_calls_the_source_calibration_with_a_signature_that_actually_binds():
    """A stage that only runs at the very end of a multi-hour job must not fail on
    a keyword name. This binds G6's real call against the real signature."""
    import inspect
    from prism_fas.train.calibration import calibrate_source_dev
    signature = inspect.signature(calibrate_source_dev)
    signature.bind(np.zeros(4), np.zeros(4, dtype=np.int64), checkpoint_sha="sha",
                   package_identity="pkg", prediction_hash="pred", run_root=Path("."))
    required = {name for name, parameter in signature.parameters.items()
                if parameter.default is inspect.Parameter.empty
                and parameter.kind is inspect.Parameter.KEYWORD_ONLY}
    import prism_fas.detector.trainer as module
    body = Path(module.__file__).read_text(encoding="utf-8").split("def run_g6")[1].split("def ")[0]
    for name in required:
        assert f"{name}=" in body, f"run_g6 does not pass the required {name!r}"


def test_g6_source_calibration_runs_on_source_dev_style_inputs(tmp_path):
    from prism_fas.train.calibration import calibrate_source_dev
    rng = np.random.default_rng(SEED)
    targets = np.concatenate([np.zeros(40, dtype=np.int64), np.ones(60, dtype=np.int64)])
    logits = rng.normal(loc=targets * 1.5, scale=1.0)
    record = calibrate_source_dev(logits, targets, checkpoint_sha="sha", package_identity="pkg",
                                  prediction_hash="pred", run_root=tmp_path)
    assert 0.0 <= record["selected_threshold"] <= 1.0
    assert math.isfinite(record["temperature"]) and record["temperature"] > 0
    assert (tmp_path / "calibration" / "source_dev.json").is_file()
    assert "target" not in json.dumps(record).lower()


def test_the_pins_record_a_real_revision_and_never_latest():
    assert SIGLIP2_PIN["revision"] not in ("main", "latest", "", None)
    assert len(SIGLIP2_PIN["revision"]) == 40
    for name, spec in SIGLIP2_PIN["files"].items():
        assert len(spec["sha256"]) == 64 and spec["bytes"] > 0
    assert len(CONVNEXT_PIN["weight_sha256"]) == 64
    assert int(SIGLIP2_TOKENIZATION["max_length"]) == int(SIGLIP2_PIN["text_max_position_embeddings"])


def test_the_detector_package_never_imports_modal():
    for path in detector_modules():
        assert "import modal" not in code_text(path), f"{path.name} imports modal"


def test_the_frozen_configs_pin_the_declared_identities_and_no_target_path():
    from prism_fas.detector.config import (ConfigError, assert_no_superseded_bank,
                                           assert_no_target_paths, load_m9_configs)
    root = Path(__file__).resolve().parents[1]
    configs = load_m9_configs(root / "configs/models/m9_detector.yaml",
                              root / "configs/train/m9_reference.yaml")
    model = configs["model_payload"]
    assert model["synthetic_bank"]["expected_content_identity_sha256"] == FROZEN_BANK_IDENTITY
    assert model["synthetic_bank"]["bank_id"] == FROZEN_BANK_ID
    assert model["backbones"]["global"]["revision"] == SIGLIP2_PIN["revision"]
    assert configs["batch_contract"].payload()["ratios"]["synthetic"] == 0.25
    assert configs["training_config"].total_epochs == 35
    assert configs["training_config"].ema_enabled is False
    with pytest.raises(ConfigError):
        assert_no_target_paths({"data": "/vol/data/packages/x/target_test/foo"})
    with pytest.raises(ConfigError):
        assert_no_superseded_bank({"bank": "prism_synthetic_bank_m8_v1_ef6a76ed46f0"})
