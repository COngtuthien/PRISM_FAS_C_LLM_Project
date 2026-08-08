"""The PRISM regional CNN-VLM detector, and every M10 baseline as a CONFIGURATION of it.

Spec section 9.2 (Table 32), implemented literally for the full method:

    F_local              = ConvNeXtV2(x_rgb, optional_highpass)
    T_global, z_global   = SigLIP2.image_encoder(x_rgb)

    q_r = RegionQuery(parsing_r, landmarks_r, learnable_token_r)
    z_r = CrossAttention(q_r, K=F_local, V=F_local) + RegionPool(T_global, parsing_r)

    p_global = GlobalHead(z_global)
    d_r      = RealManifold.distance(z_r)
    p_prompt = PromptHead(z_r, frozen_recipe_text_embeddings)

and spec section 9.4 (Table 34) for the image-level fusion:

    s_region = TopKMean(normalize(d_r), k=2)
    s_final  = 1 - (1 - p_global) * (1 - s_region) * (1 - p_prompt_spoof)
    # Unknown/reject also considers entropy and global-local disagreement.

The region prior is a SOFT MASK, never nine hard crops (spec section 9.1): the
image is encoded once and the nine regions are read out by prior-biased attention
and prior-weighted pooling. Visibility gates every regional term.

The global SigLIP2 branch is FROZEN (Table 5: "SigLIP2 Base P16-224 frozen, train
fusion/heads"); it is deliberately kept out of the trainable state and out of the
checkpoint, and is bound instead by its verified SHA identity.

## The M10 switches

`DetectorConfig.variant` is a `ResolvedExperimentVariant`. It decides what is
INSTANTIATED, not merely what is used:

    local_branch=off      the ConvNeXt tower, its projection and its token head are
                          not created, so no unused trainable parameter reaches the
                          optimizer and no local loss has anything to compute
    global_branch=off     the frozen SigLIP2 tower is never attached, never run and
                          never pooled; the architecture identity records its absence
    fusion                single_logit  -> one branch, one linear classifier
                          simple_concat -> both branch summaries concatenated,
                                           projected, one binary classifier; the
                                           score is the classifier's, NOT a noisy-or
                          prism_noisy_or-> the Table 34 fusion, unchanged
    region=off            no RegionQuery, no cross-attention, no RegionPool, no
                          regional prototype state, no PromptHead
    manifold              off | global_center (ONE site over a pooled image
                          embedding, K=1) | multi_prototype (the nine regional
                          manifolds, K from prototype_k)
    prompt                off | frozen_prompt | adapter (a zero-initialized residual
                          bottleneck on the CACHED text matrix; the text tower is
                          still never loaded and never fine-tuned)

**Absent means absent.** A component the variant does not declare is `None` in the
`ModelOutput`, not a zero tensor, so nothing downstream can mistake a missing term
for a measured one.

**The reference configuration keeps its M9 identity.** `DetectorConfig.payload()`
carries the variant's architecture DELTA against the frozen B08 reference, which is
empty for B08 itself — so the M9 reference checkpoint stays loadable and every other
variant gets a different architecture identity. Two different architectures never
share a hash; that is the property that matters.
"""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import torch
from torch import nn
from .contracts import REGION_COUNT, REGION_ORDER, DetectorContractError, ModelOutput, architecture_identity
from .heads import DEFAULT_PROMPT_TEMPERATURE, GlobalHead, PromptHead
from .manifold import (DEFAULT_COVARIANCE_EPSILON, DEFAULT_DISTANCE_SCALE_CONVENTION, DEFAULT_K,
                       DEFAULT_TAU_PROTOTYPE, DEFAULT_UPDATE_DECAY, RealManifold)
from .pretrained import CONVNEXT_PIN, SIGLIP2_PIN
from .regions import RegionCrossAttention, RegionPool, RegionQuery, resize_priors
from .variant import ResolvedExperimentVariant

DETECTOR_SCHEMA_VERSION = "m9-prism-detector-v1"
# Spec section 9.4 fixes k = 2. This is not a default.
FUSION_TOP_K = 2
# SPEC_UNDERSPECIFIED defaults, all listed in docs/M9_DETECTOR_CONTRACT.md.
DEFAULT_REGION_DIM = 256
DEFAULT_ATTENTION_HEADS = 4
DEFAULT_VISIBILITY_THRESHOLD = 0.30
DEFAULT_DISTANCE_SCALE = 1.0
LOG2 = 0.6931471805599453


@dataclass(frozen=True)
class DetectorConfig:
    """Every architectural choice, spec-fixed or declared default, in one place.

    Anything here that the spec does not fix is marked in
    `docs/M9_DETECTOR_CONTRACT.md` as SPEC_UNDERSPECIFIED; nothing here may be
    presented as a spec requirement just because it lives in config.
    """
    region_dim: int = DEFAULT_REGION_DIM
    region_attention_heads: int = DEFAULT_ATTENTION_HEADS
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD
    optional_highpass: bool = False
    local_model_name: str = CONVNEXT_PIN["timm_name"]
    local_pretrained: bool = True
    global_model_id: str = SIGLIP2_PIN["model_id"]
    global_revision: str = SIGLIP2_PIN["revision"]
    prototype_k: int = DEFAULT_K
    covariance_epsilon: float = DEFAULT_COVARIANCE_EPSILON
    tau_prototype: float = DEFAULT_TAU_PROTOTYPE
    prototype_decay: float = DEFAULT_UPDATE_DECAY
    # SPEC_UNDERSPECIFIED: see the `manifold` module docstring and DECISIONS.md.
    distance_scale_convention: str = DEFAULT_DISTANCE_SCALE_CONVENTION
    prompt_temperature: float = DEFAULT_PROMPT_TEMPERATURE
    fusion_top_k: int = FUSION_TOP_K
    global_head_dropout: float = 0.0
    region_order: tuple[str, ...] = REGION_ORDER
    # The M10 switch set. Defaults to the frozen B08 reference, so an M9 call site
    # that never mentions a variant builds exactly the M9 reference detector.
    variant: ResolvedExperimentVariant = field(default_factory=ResolvedExperimentVariant.reference)

    def __post_init__(self) -> None:
        self.variant.validate()
        # `prototype_k` appears both here and in the M10 flag set. `DetectorConfig`
        # is the single authority and the variant follows it, so the two can never
        # disagree at run time; `detector_config_from` sets this field FROM the
        # variant, so for a real matrix row they are the same number by construction
        # and this alignment is a no-op. `manifold=off` carries K=0 and is left alone.
        if self.variant.has_manifold and int(self.prototype_k) != int(self.variant.prototype_k):
            object.__setattr__(self, "variant", self.variant.with_flags(prototype_k=int(self.prototype_k)))

    def payload(self) -> dict[str, Any]:
        body = {"schema_version": DETECTOR_SCHEMA_VERSION, "region_order": list(self.region_order),
                "region_count": REGION_COUNT, "region_dim": self.region_dim,
                "region_attention_heads": self.region_attention_heads,
                "visibility_threshold": self.visibility_threshold,
                "optional_highpass": self.optional_highpass,
                "local_model_name": self.local_model_name,
                "local_weight_sha256": CONVNEXT_PIN["weight_sha256"],
                "global_model_id": self.global_model_id, "global_revision": self.global_revision,
                "prototype_k": self.prototype_k, "covariance_epsilon": self.covariance_epsilon,
                "tau_prototype": self.tau_prototype, "prototype_decay": self.prototype_decay,
                "distance_scale_convention": self.distance_scale_convention,
                "prompt_temperature": self.prompt_temperature, "fusion_top_k": self.fusion_top_k,
                "global_head_dropout": self.global_head_dropout,
                "distance_normalizer": "1-exp(-d/d_scale)", "fusion": "noisy_or_table_34"}
        # The DELTA against the frozen B08 reference, absent when there is none. The
        # M9 reference configuration therefore keeps the architecture identity its
        # checkpoints were written with, while every ablation that changes the tensor
        # graph gets a different one — which is the property "no silent
        # compatibility" actually requires.
        delta = architecture_delta(self.variant)
        if delta: body["variant_architecture"] = delta
        return body

    def identity(self) -> str:
        return architecture_identity(self.payload())


def architecture_delta(variant: ResolvedExperimentVariant) -> dict[str, Any]:
    """The architecture switches on which `variant` differs from the B08 reference."""
    reference = ResolvedExperimentVariant.reference().architecture_payload()
    return {key: value for key, value in variant.architecture_payload().items()
            if reference.get(key) != value}


class _Detached:
    """Holds the frozen global tower WITHOUT registering it as a submodule.

    The tower is 375M frozen parameters. Registering it would put half a gigabyte
    of unchanging weights into every checkpoint and let a `strict=True` load
    silently accept a different SigLIP2 build; instead the pin's SHA identity is
    what the checkpoint guards.
    """
    __slots__ = ("module",)

    def __init__(self, module: Any) -> None: self.module = module


class TextAdapter(nn.Module):
    """The `prompt: adapter` variant. SPEC_UNDERSPECIFIED, frozen in DECISIONS.md.

    Spec section 2.2 forbids fine-tuning the text encoder from the start and allows
    "chỉ LoRA/adapter sau ablation frozen backbone". This is that adapter, and it is
    deliberately the smallest thing that satisfies the sentence:

        location   a residual bottleneck on the CACHED frozen recipe text matrix.
                   The text tower is still never loaded, never run and never
                   fine-tuned, at training or at inference.
        shape      text_dim -> rank -> text_dim, rank = 32
        trainable  the LayerNorm, the down projection and the up projection
        init       up projection ZEROED, so at step 0 the adapter is an exact no-op
                   and `prompt=adapter` starts numerically identical to
                   `prompt=frozen_prompt`. The ablation measures what the adapter
                   LEARNS, not a different initialization.
        optimizer  the heads group (LR 1e-4), like every other head
        text cache unchanged, still bound by its identity; the adapted matrix is
                   re-L2-normalized so the InfoNCE geometry is preserved

    No target text, no target taxonomy and no target label reaches this module.
    """
    RANK = 32

    def __init__(self, text_dim: int, *, rank: int = RANK):
        super().__init__()
        self.text_dim, self.rank = int(text_dim), int(rank)
        self.norm = nn.LayerNorm(self.text_dim)
        self.down = nn.Linear(self.text_dim, self.rank)
        self.up = nn.Linear(self.rank, self.text_dim, bias=False)
        nn.init.trunc_normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)

    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        residual = self.up(torch.nn.functional.gelu(self.down(self.norm(text_embeddings))))
        return torch.nn.functional.normalize(text_embeddings + residual, dim=-1, eps=1e-6)

    def extra_repr(self) -> str: return f"text_dim={self.text_dim}, rank={self.rank}, zero_init_up=True"


class PRISMDetector(nn.Module):
    """Table 32 architecture with the Table 34 fusion, emitting a typed
    `ModelOutput` — never an opaque tuple.

    Every M10 baseline and ablation is this class under a different
    `config.variant`. There is no `if experiment_id == ...` anywhere: the module
    asks the variant what it HAS.
    """

    def __init__(self, config: DetectorConfig, *, text_embeddings: torch.Tensor,
                 global_tower: Any = None, local_weight_file: str | Path | None = None,
                 text_cache_identity: str = ""):
        super().__init__()
        self.config = config
        self.variant = config.variant.validate()
        self.text_cache_identity = str(text_cache_identity)
        dim = int(config.region_dim)
        variant = self.variant

        # --- local branch ---------------------------------------------------
        # `local_branch=off` creates NOTHING: no tower, no projection, no token head.
        # An unused trainable branch left in the optimizer would make B01 quietly
        # more expensive than it is and would put dead weights in its checkpoint.
        self.local_backbone = None
        self.local_channels = 0
        self.local_reduction = 0
        self.highpass_enabled = bool(config.optional_highpass) and variant.has_local
        self.highpass_stem = None
        if variant.has_local:
            import timm
            overlay = {"file": str(local_weight_file)} if local_weight_file else None
            self.local_backbone = timm.create_model(
                config.local_model_name, pretrained=bool(config.local_pretrained), features_only=True,
                out_indices=(3,), **({"pretrained_cfg_overlay": overlay} if overlay else {}))
            self.local_channels = int(self.local_backbone.feature_info.channels()[-1])
            self.local_reduction = int(self.local_backbone.feature_info.reduction()[-1])
            # `optional_highpass` (spec section 9.2). Default OFF. When enabled the
            # detail map enters as a fourth stem channel whose weights start at zero,
            # so switching it on is a no-op at step 0 rather than a different model.
            if self.highpass_enabled:
                self.highpass_stem = nn.Conv2d(1, 3, kernel_size=3, padding=1, bias=False)
                nn.init.zeros_(self.highpass_stem.weight)
            self.local_projection = nn.Sequential(nn.Linear(self.local_channels, dim), nn.LayerNorm(dim))
            self.local_head = nn.Linear(dim, 1)
            nn.init.zeros_(self.local_head.bias)
            nn.init.trunc_normal_(self.local_head.weight, std=0.01)

        # --- global branch --------------------------------------------------
        self.global_dim = int(SIGLIP2_PIN["vision_hidden_size"])
        self.global_patch_tokens = int(SIGLIP2_PIN["vision_patch_tokens"])
        self._global = _Detached(global_tower if variant.has_global else None)

        # --- semantic region path -------------------------------------------
        if variant.has_region_path:
            self.region_query = RegionQuery(dim)
            self.region_attention = RegionCrossAttention(dim, heads=int(config.region_attention_heads))
            self.region_pool = RegionPool(self.global_dim, dim)
            self.region_norm = nn.LayerNorm(dim)

        # --- classifier -----------------------------------------------------
        # The Table 34 path keeps GlobalHead exactly as M9 froze it. The two simpler
        # fusions do NOT reuse it: their score is a classifier over the branch
        # summaries, which is a different computational graph, not a renamed one.
        if variant.fusion == "prism_noisy_or":
            self.global_head = GlobalHead(self.global_dim, dropout=float(config.global_head_dropout))
        else:
            if variant.has_global:
                self.global_projection = nn.Sequential(nn.Linear(self.global_dim, dim), nn.LayerNorm(dim))
            if variant.fusion == "simple_concat":
                # CNN + ViT feature fusion, then ONE binary classifier. No region,
                # no manifold and no prompt semantics enter here.
                self.fusion_projection = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.LayerNorm(dim))
            self.fusion_classifier = nn.Linear(dim, 1)
            nn.init.zeros_(self.fusion_classifier.bias)
            nn.init.trunc_normal_(self.fusion_classifier.weight, std=0.01)

        # --- manifold --------------------------------------------------------
        # `manifold=off` instantiates no manifold at all: no prototype state, no
        # distances, no manifold losses and no G2 stage.
        self.manifold = None
        self.manifold_slots = int(variant.manifold_slots)
        if variant.has_manifold:
            self.manifold = RealManifold(dim, k=int(variant.prototype_k),
                                         epsilon=float(config.covariance_epsilon),
                                         tau_prototype=float(config.tau_prototype),
                                         decay=float(config.prototype_decay),
                                         regions=self.manifold_slots,
                                         distance_scale_convention=str(config.distance_scale_convention))

        # --- prompt ----------------------------------------------------------
        self.prompt_head = None
        self.text_adapter = None
        if variant.has_prompt:
            self.prompt_head = PromptHead(dim, text_embeddings, temperature=float(config.prompt_temperature),
                                          cache_identity_sha256=self.text_cache_identity)
            if variant.prompt_adapter_trainable:
                self.text_adapter = TextAdapter(self.prompt_head.text_dim)

        # `normalize(d_r)` scale: the median live distance per site, set once at
        # the G2 transition and frozen from then on (SPEC_UNDERSPECIFIED, see
        # docs/M9_DETECTOR_CONTRACT.md section 7).
        if variant.has_manifold:
            self.register_buffer("distance_scale", torch.full((self.manifold_slots,), float(DEFAULT_DISTANCE_SCALE)))
            self.register_buffer("distance_scale_frozen", torch.zeros((), dtype=torch.bool))
        # SigLIP2 input contract: [0,1] RGB -> (x-0.5)/0.5.
        self.register_buffer("global_mean", torch.tensor(SIGLIP2_PIN["input_contract"]["normalization_mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("global_std", torch.tensor(SIGLIP2_PIN["input_contract"]["normalization_std"]).view(1, 3, 1, 1), persistent=False)
        # ConvNeXt input contract, applied exactly once.
        self.register_buffer("local_mean", torch.tensor(CONVNEXT_PIN["input_contract"]["normalization_mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("local_std", torch.tensor(CONVNEXT_PIN["input_contract"]["normalization_std"]).view(1, 3, 1, 1), persistent=False)

    # --- frozen tower plumbing ---------------------------------------------
    @property
    def global_tower(self) -> Any: return self._global.module

    def attach_global_tower(self, tower: Any) -> "PRISMDetector":
        if not self.variant.has_global:
            raise DetectorContractError("this variant declares global_branch=off; the SigLIP2 tower "
                                        "must not be attached")
        self._global.module = tower
        return self

    def to(self, *args: Any, **kwargs: Any) -> "PRISMDetector":
        moved = super().to(*args, **kwargs)
        if self._global.module is not None: self._global.module.to(*args, **kwargs)
        return moved

    def train(self, mode: bool = True) -> "PRISMDetector":
        # The frozen global tower never leaves eval mode, even inside training.
        out = super().train(mode)
        if self._global.module is not None: self._global.module.eval()
        return out

    # --- pieces -------------------------------------------------------------
    def _highpass(self, image: torch.Tensor) -> torch.Tensor:
        """Grey-level detail: image minus its 3x3 box blur. Computed on the fly,
        never stored, never part of the package."""
        grey = image.mean(dim=1, keepdim=True)
        blurred = torch.nn.functional.avg_pool2d(grey, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        return grey - blurred

    def local_features(self, image: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """`F_local` as `[B,P,D]` tokens on the stage-4 grid."""
        if self.local_backbone is None:
            raise DetectorContractError("this variant declares local_branch=off; there are no local features")
        normalized = (image - self.local_mean) / self.local_std
        if self.highpass_stem is not None:
            normalized = normalized + self.highpass_stem(self._highpass(image))
        feature_map = self.local_backbone(normalized)[-1]                      # [B,C,h,w]
        batch, channels, height, width = feature_map.shape
        tokens = feature_map.flatten(2).transpose(1, 2)                        # [B,P,C]
        return self.local_projection(tokens), (height, width)

    def global_features(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`T_global, z_global` from the FROZEN SigLIP2 image encoder."""
        tower = self._global.module
        if tower is None: raise DetectorContractError("the frozen SigLIP2 global tower is not attached")
        normalized = (image - self.global_mean) / self.global_std
        with torch.no_grad():
            output = tower(pixel_values=normalized.to(next(tower.parameters()).dtype))
        tokens = output.last_hidden_state.to(image.dtype)
        pooled = output.pooler_output.to(image.dtype)
        if tokens.shape[1] != self.global_patch_tokens:
            raise DetectorContractError(f"SigLIP2 returned {tokens.shape[1]} patch tokens, "
                                        f"expected {self.global_patch_tokens}")
        return tokens, pooled

    def region_embeddings(self, local_tokens: torch.Tensor, local_grid: tuple[int, int],
                          global_tokens: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        """`z_r = CrossAttention(q_r, K=F_local, V=F_local) + RegionPool(T_global, parsing_r)`."""
        local_priors = resize_priors(priors, local_grid).flatten(2)             # [B,R,P_local]
        side = int(round(float(global_tokens.shape[1]) ** 0.5))
        if side * side != global_tokens.shape[1]:
            raise DetectorContractError("the global patch grid is not square")
        global_priors = resize_priors(priors, (side, side)).flatten(2)          # [B,R,P_global]
        queries = self.region_query(resize_priors(priors, local_grid))
        attended = self.region_attention(queries, local_tokens, local_priors)
        pooled = self.region_pool(global_tokens, global_priors)
        return self.region_norm(attended + pooled)

    def text_matrix(self) -> torch.Tensor | None:
        """The recipe text matrix the PromptHead scores against.

        `frozen_prompt` returns the cached constant unchanged. `adapter` returns the
        adapted matrix — which at step 0 IS the cached constant, because the adapter's
        up projection is zero-initialized.
        """
        if self.prompt_head is None: return None
        base = self.prompt_head.text_embeddings
        return base if self.text_adapter is None else self.text_adapter(base)

    # --- fusion (Table 34) --------------------------------------------------
    def normalize_distance(self, distances: torch.Tensor) -> torch.Tensor:
        """`normalize(d_r)`: a monotone map of the spec's squared Mahalanobis
        distance into [0,1] (SPEC_UNDERSPECIFIED, section 7 of the contract)."""
        scale = self.distance_scale.clamp_min(1e-6).to(distances.dtype).unsqueeze(0)
        return 1.0 - torch.exp(-distances.clamp_min(0.0) / scale)

    def region_score(self, distances: torch.Tensor, region_valid: torch.Tensor) -> torch.Tensor:
        """`s_region = TopKMean(normalize(d_r), k=2)` over VALID sites only.

        Invalid sites are removed before the top-k; with fewer than k valid the
        mean is over what is valid; with none valid the score is exactly 0. A single
        global center has one site, so the top-k degenerates to that one value —
        which is the same formula, not a second one.
        """
        normalized = self.normalize_distance(distances)
        valid = region_valid.bool()
        k = int(self.config.fusion_top_k)
        very_negative = torch.finfo(normalized.dtype).min
        candidate = normalized.masked_fill(~valid, very_negative)
        top_values, top_index = candidate.topk(k=min(k, normalized.shape[1]), dim=1)
        selected_valid = torch.gather(valid, 1, top_index)
        counts = selected_valid.sum(dim=1)
        summed = (top_values * selected_valid.to(normalized.dtype)).sum(dim=1)
        return torch.where(counts > 0, summed / counts.clamp_min(1).to(normalized.dtype),
                           torch.zeros_like(summed))

    @staticmethod
    def fuse(p_global: torch.Tensor, s_region: torch.Tensor | None,
             p_prompt_spoof: torch.Tensor | None) -> torch.Tensor:
        """`s_final = 1 - (1 - p_global)(1 - s_region)(1 - p_prompt_spoof)`.

        An absent factor is OMITTED from the product, never replaced by 0.0 — the
        same formula with fewer factors, which is why the prediction schema writes
        `null` for a term the variant does not have.
        """
        complement = 1.0 - p_global
        if s_region is not None: complement = complement * (1.0 - s_region)
        if p_prompt_spoof is not None: complement = complement * (1.0 - p_prompt_spoof)
        return 1.0 - complement

    @staticmethod
    def confidence(p_global: torch.Tensor, s_region: torch.Tensor | None,
                   s_final: torch.Tensor) -> dict[str, torch.Tensor]:
        """The two reject signals spec section 9.4 names: predictive entropy and
        global-local disagreement.

        Disagreement needs two evidence terms. A variant with one score has nothing
        to disagree with, so the signal is absent rather than a constant zero that
        would read as perfect agreement.
        """
        probability = s_final.clamp(1e-6, 1.0 - 1e-6)
        entropy = -(probability * probability.log() + (1 - probability) * (1 - probability).log()) / LOG2
        features = {"entropy": entropy}
        if s_region is not None:
            features["global_local_disagreement"] = (p_global - s_region).abs()
        return features

    # --- distance scale -----------------------------------------------------
    @torch.no_grad()
    def set_distance_scale(self, values: torch.Tensor, *, freeze: bool = True) -> None:
        if self.manifold is None:
            raise DetectorContractError("this variant declares manifold=off; there is no distance scale")
        if values.shape != (self.manifold_slots,):
            raise DetectorContractError(f"distance_scale must be [{self.manifold_slots}]")
        self.distance_scale.copy_(values.detach().to(self.distance_scale.dtype).clamp_min(1e-6))
        self.distance_scale_frozen.fill_(bool(freeze))

    # --- manifold input -----------------------------------------------------
    def manifold_input(self, region_embeddings: torch.Tensor | None, image_embedding: torch.Tensor | None,
                       region_valid: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """The `[B,M,D]` embeddings the manifold measures, and their `[B,M]` validity.

        `multi_prototype` measures the nine regional embeddings, exactly as M9 does.
        `global_center` measures ONE pooled image embedding: the visibility-weighted
        mean of the valid region embeddings when the region path exists, and the
        fused image embedding when it does not. Either way it is one site, so B06
        and B07 differ in the manifold and in nothing else.
        """
        if self.variant.manifold_scope == "regional":
            if region_embeddings is None or region_valid is None:
                raise DetectorContractError("a regional manifold needs the region path")
            return region_embeddings, region_valid
        if region_embeddings is not None and region_valid is not None:
            weight = region_valid.to(region_embeddings.dtype).unsqueeze(-1)      # [B,R,1]
            count = weight.sum(dim=1).clamp_min(1.0)                             # [B,1]
            pooled = (region_embeddings * weight).sum(dim=1) / count             # [B,D]
            valid = region_valid.any(dim=1, keepdim=True)                        # [B,1]
            return pooled.unsqueeze(1), valid
        if image_embedding is None:
            raise DetectorContractError("a global center needs a pooled image embedding")
        return (image_embedding.unsqueeze(1),
                torch.ones(image_embedding.shape[0], 1, dtype=torch.bool, device=image_embedding.device))

    # --- forward ------------------------------------------------------------
    def forward(self, batch: Any) -> ModelOutput:
        variant = self.variant
        image, priors = batch.image, batch.region_priors
        if image.dim() != 4 or image.shape[1] != 3:
            raise DetectorContractError(f"detector input must be [B,3,H,W], got {tuple(image.shape)}")
        if priors.shape[1] != REGION_COUNT:
            raise DetectorContractError(f"region priors must carry {REGION_COUNT} regions in the frozen order")
        visibility_valid = batch.visibility >= float(self.config.visibility_threshold)   # [B,R]

        local_tokens, local_grid = (self.local_features(image) if variant.has_local else (None, None))
        global_tokens, global_pooled = (self.global_features(image) if variant.has_global else (None, None))

        embeddings = (self.region_embeddings(local_tokens, local_grid, global_tokens, priors)
                      if variant.has_region_path else None)

        # The pooled per-branch summaries the two simpler fusions classify. Never
        # built for the Table 34 path, which classifies `z_global` directly.
        image_embedding: torch.Tensor | None = None
        if variant.fusion != "prism_noisy_or":
            parts: list[torch.Tensor] = []
            if variant.has_local: parts.append(local_tokens.mean(dim=1))
            if variant.has_global: parts.append(self.global_projection(global_pooled))
            image_embedding = (self.fusion_projection(torch.cat(parts, dim=-1))
                               if variant.fusion == "simple_concat" else parts[0])
            global_logit = self.fusion_classifier(image_embedding)
        else:
            global_logit = self.global_head(global_pooled)

        local_logits = self.local_head(local_tokens).squeeze(-1) if variant.has_local else None

        distances = manifold_valid = None
        if self.manifold is not None:
            manifold_embeddings, manifold_valid = self.manifold_input(embeddings, image_embedding, visibility_valid)
            distances = self.manifold.distance(manifold_embeddings)
        else:
            manifold_embeddings = None

        prompt: dict[str, torch.Tensor] | None = None
        if self.prompt_head is not None:
            prompt = self.prompt_head(embeddings, PromptHead.applicability(
                batch.is_synthetic, getattr(batch, "attack_region_mask", None), visibility_valid),
                text_embeddings=self.text_matrix())

        p_global = torch.sigmoid(global_logit).squeeze(-1)
        s_region = (self.region_score(distances, manifold_valid) if variant.fuses_region_evidence else None)
        p_prompt_spoof = (prompt["p_prompt_spoof"] if (prompt is not None and variant.fuses_prompt_evidence)
                          else None)
        # Only the Table 34 path fuses. A concat or single-logit baseline's score IS
        # its classifier's probability; running it through a one-factor noisy-or
        # would be the same number, but writing it that way would suggest a fusion
        # that does not exist.
        s_final = (self.fuse(p_global, s_region, p_prompt_spoof)
                   if variant.fusion == "prism_noisy_or" else p_global)

        aux: dict[str, torch.Tensor] = {}
        if global_pooled is not None: aux["global_embedding"] = global_pooled
        if local_tokens is not None: aux["local_tokens"] = local_tokens
        if image_embedding is not None: aux["image_embedding"] = image_embedding
        if manifold_embeddings is not None: aux["manifold_embeddings"] = manifold_embeddings
        if distances is not None:
            aux["normalized_distances"] = self.normalize_distance(distances)
            aux["distance_scale"] = self.distance_scale.detach().clone()
        if prompt is not None:
            aux["region_prompt_logits"] = prompt["region_prompt_logits"]
            aux["prompt_region_embeddings"] = prompt["prompt_region_embeddings"]
            aux["prompt_applicable"] = prompt["prompt_applicable"]
        return ModelOutput(
            global_logit=global_logit, local_logits=local_logits, region_embeddings=embeddings,
            region_distances=distances, region_valid=manifold_valid,
            prompt_logits=(prompt["prompt_logits"] if prompt is not None else None),
            p_global=p_global, s_region=s_region, p_prompt_spoof=p_prompt_spoof, s_final=s_final,
            confidence_features=self.confidence(p_global, s_region, s_final),
            region_visibility_valid=visibility_valid,
            # Every fusion component and auxiliary feature the declared losses and
            # the M10 ablations need, exposed by name rather than re-derived.
            aux=aux).validate()

    # --- optimization -------------------------------------------------------
    def parameter_groups(self, *, backbone_lr: float, head_lr: float, weight_decay: float) -> list[dict[str, Any]]:
        """Spec section 10.3: backbone LR 1e-5, heads/manifold LR 1e-4.

        The frozen SigLIP2 tower contributes no parameters at all; the manifold
        lives in buffers, so it contributes none either. A group with no parameters
        is OMITTED rather than passed to AdamW empty — B01 has no trainable
        backbone, and an empty group would misreport it as having one.
        """
        backbone = list(self.local_backbone.parameters()) if self.local_backbone is not None else []
        if self.highpass_stem is not None: backbone += list(self.highpass_stem.parameters())
        backbone_ids = {id(parameter) for parameter in backbone}
        heads = [parameter for parameter in self.parameters() if id(parameter) not in backbone_ids]
        groups = [{"params": backbone, "lr": float(backbone_lr), "weight_decay": float(weight_decay), "name": "backbone"},
                  {"params": heads, "lr": float(head_lr), "weight_decay": float(weight_decay), "name": "heads"}]
        return [group for group in groups if group["params"]]

    def parameter_counts(self) -> dict[str, int]:
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        frozen_tower = (sum(parameter.numel() for parameter in self._global.module.parameters())
                        if self._global.module is not None else 0)
        local = (sum(parameter.numel() for parameter in self.local_backbone.parameters())
                 if self.local_backbone is not None else 0)
        manifold_elements = (int(self.manifold.centers.numel() + self.manifold.variances.numel())
                             if self.manifold is not None else 0)
        return {"trainable": trainable,
                "total_registered": sum(parameter.numel() for parameter in self.parameters()),
                "local_backbone": local,
                "frozen_global_tower": frozen_tower,
                "manifold_buffer_elements": manifold_elements}

    def architecture_identity(self) -> str:
        payload = {**self.config.payload(), "local_channels": self.local_channels,
                   "local_reduction": self.local_reduction, "global_dim": self.global_dim,
                   "global_patch_tokens": self.global_patch_tokens,
                   "n_prompt": (self.prompt_head.n_prompt if self.prompt_head is not None else 0),
                   "text_dim": (self.prompt_head.text_dim if self.prompt_head is not None else 0),
                   "text_cache_identity": self.text_cache_identity}
        return architecture_identity(payload)


def build_detector(config: DetectorConfig, *, text_embeddings: torch.Tensor, siglip: Any = None,
                   local_weight_file: str | Path | None = None, text_cache_identity: str = "",
                   device: str = "cpu") -> PRISMDetector:
    """Assemble the detector with the frozen SigLIP2 vision tower attached.

    `siglip` is a `SigLIP2Artifacts` whose SHA-256s were already verified; passing
    `None` builds the trainable half only, which is what the toy tests use. A variant
    with `global_branch=off` never loads the tower at all, even when one is offered:
    a frozen branch that is not part of the declared architecture must not be present
    "just to satisfy shapes".
    """
    tower = None
    if siglip is not None and config.variant.has_global:
        tower = siglip.load_model(device=device).vision_model
        tower.eval()
        for parameter in tower.parameters(): parameter.requires_grad_(False)
    model = PRISMDetector(config, text_embeddings=text_embeddings, global_tower=tower,
                          local_weight_file=local_weight_file, text_cache_identity=text_cache_identity)
    return model.to(device)


def detector_summary(model: PRISMDetector) -> dict[str, Any]:
    return {"detector_schema_version": DETECTOR_SCHEMA_VERSION,
            "architecture_identity_sha256": model.architecture_identity(),
            "config": model.config.payload(), "parameter_counts": model.parameter_counts(),
            "text_cache_identity_sha256": model.text_cache_identity,
            "region_order": list(REGION_ORDER),
            "variant": model.variant.payload(),
            "trainable_modules": sorted(name for name, _ in model.named_children())}
