"""The M9 PRISM regional CNN-VLM detector.

Spec section 9.2 (Table 32), implemented literally:

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

    def payload(self) -> dict[str, Any]:
        return {"schema_version": DETECTOR_SCHEMA_VERSION, "region_order": list(self.region_order),
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

    def identity(self) -> str:
        return architecture_identity(self.payload())


class _Detached:
    """Holds the frozen global tower WITHOUT registering it as a submodule.

    The tower is 375M frozen parameters. Registering it would put half a gigabyte
    of unchanging weights into every checkpoint and let a `strict=True` load
    silently accept a different SigLIP2 build; instead the pin's SHA identity is
    what the checkpoint guards.
    """
    __slots__ = ("module",)

    def __init__(self, module: Any) -> None: self.module = module


class PRISMDetector(nn.Module):
    """Table 32 architecture with the Table 34 fusion, emitting a typed
    `ModelOutput` — never an opaque tuple."""

    def __init__(self, config: DetectorConfig, *, text_embeddings: torch.Tensor,
                 global_tower: Any = None, local_weight_file: str | Path | None = None,
                 text_cache_identity: str = ""):
        super().__init__()
        self.config = config
        self.text_cache_identity = str(text_cache_identity)
        dim = int(config.region_dim)

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
        self.highpass_enabled = bool(config.optional_highpass)
        self.highpass_stem = nn.Conv2d(1, 3, kernel_size=3, padding=1, bias=False) if self.highpass_enabled else None
        if self.highpass_stem is not None: nn.init.zeros_(self.highpass_stem.weight)

        self.local_projection = nn.Sequential(nn.Linear(self.local_channels, dim), nn.LayerNorm(dim))
        self.local_head = nn.Linear(dim, 1)
        nn.init.zeros_(self.local_head.bias)
        nn.init.trunc_normal_(self.local_head.weight, std=0.01)

        self.global_dim = int(SIGLIP2_PIN["vision_hidden_size"])
        self.global_patch_tokens = int(SIGLIP2_PIN["vision_patch_tokens"])
        self._global = _Detached(global_tower)

        self.region_query = RegionQuery(dim)
        self.region_attention = RegionCrossAttention(dim, heads=int(config.region_attention_heads))
        self.region_pool = RegionPool(self.global_dim, dim)
        self.region_norm = nn.LayerNorm(dim)

        self.global_head = GlobalHead(self.global_dim, dropout=float(config.global_head_dropout))
        self.manifold = RealManifold(dim, k=int(config.prototype_k), epsilon=float(config.covariance_epsilon),
                                     tau_prototype=float(config.tau_prototype), decay=float(config.prototype_decay),
                                     distance_scale_convention=str(config.distance_scale_convention))
        self.prompt_head = PromptHead(dim, text_embeddings, temperature=float(config.prompt_temperature),
                                      cache_identity_sha256=self.text_cache_identity)

        # `normalize(d_r)` scale: the median live distance per region, set once at
        # the G2 transition and frozen from then on (SPEC_UNDERSPECIFIED, see
        # docs/M9_DETECTOR_CONTRACT.md section 7).
        self.register_buffer("distance_scale", torch.full((REGION_COUNT,), float(DEFAULT_DISTANCE_SCALE)))
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

    def local_features(self, image: torch.Tensor) -> torch.Tensor:
        """`F_local` as `[B,P,D]` tokens on the stage-4 grid."""
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

    # --- fusion (Table 34) --------------------------------------------------
    def normalize_distance(self, distances: torch.Tensor) -> torch.Tensor:
        """`normalize(d_r)`: a monotone map of the spec's squared Mahalanobis
        distance into [0,1] (SPEC_UNDERSPECIFIED, section 7 of the contract)."""
        scale = self.distance_scale.clamp_min(1e-6).to(distances.dtype).unsqueeze(0)
        return 1.0 - torch.exp(-distances.clamp_min(0.0) / scale)

    def region_score(self, distances: torch.Tensor, region_valid: torch.Tensor) -> torch.Tensor:
        """`s_region = TopKMean(normalize(d_r), k=2)` over VALID regions only.

        Invalid regions are removed before the top-k; with fewer than k valid the
        mean is over what is valid; with none valid the score is exactly 0.
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
    def fuse(p_global: torch.Tensor, s_region: torch.Tensor, p_prompt_spoof: torch.Tensor) -> torch.Tensor:
        """`s_final = 1 - (1 - p_global)(1 - s_region)(1 - p_prompt_spoof)`."""
        return 1.0 - (1.0 - p_global) * (1.0 - s_region) * (1.0 - p_prompt_spoof)

    @staticmethod
    def confidence(p_global: torch.Tensor, s_region: torch.Tensor, s_final: torch.Tensor) -> dict[str, torch.Tensor]:
        """The two reject signals spec section 9.4 names: predictive entropy and
        global-local disagreement."""
        probability = s_final.clamp(1e-6, 1.0 - 1e-6)
        entropy = -(probability * probability.log() + (1 - probability) * (1 - probability).log()) / LOG2
        return {"entropy": entropy, "global_local_disagreement": (p_global - s_region).abs()}

    # --- distance scale -----------------------------------------------------
    @torch.no_grad()
    def set_distance_scale(self, values: torch.Tensor, *, freeze: bool = True) -> None:
        if values.shape != (REGION_COUNT,): raise DetectorContractError("distance_scale must be [R]")
        self.distance_scale.copy_(values.detach().to(self.distance_scale.dtype).clamp_min(1e-6))
        self.distance_scale_frozen.fill_(bool(freeze))

    # --- forward ------------------------------------------------------------
    def forward(self, batch: Any) -> ModelOutput:
        image, priors = batch.image, batch.region_priors
        if image.dim() != 4 or image.shape[1] != 3:
            raise DetectorContractError(f"detector input must be [B,3,H,W], got {tuple(image.shape)}")
        if priors.shape[1] != REGION_COUNT:
            raise DetectorContractError(f"region priors must carry {REGION_COUNT} regions in the frozen order")
        region_valid = batch.visibility >= float(self.config.visibility_threshold)

        local_tokens, local_grid = self.local_features(image)
        global_tokens, global_pooled = self.global_features(image)
        embeddings = self.region_embeddings(local_tokens, local_grid, global_tokens, priors)

        global_logit = self.global_head(global_pooled)
        local_logits = self.local_head(local_tokens).squeeze(-1)
        distances = self.manifold.distance(embeddings)
        prompt = self.prompt_head(embeddings, PromptHead.applicability(
            batch.is_synthetic, getattr(batch, "attack_region_mask", None), region_valid))

        p_global = torch.sigmoid(global_logit).squeeze(-1)
        s_region = self.region_score(distances, region_valid)
        p_prompt_spoof = prompt["p_prompt_spoof"]
        s_final = self.fuse(p_global, s_region, p_prompt_spoof)
        return ModelOutput(
            global_logit=global_logit, local_logits=local_logits, region_embeddings=embeddings,
            region_distances=distances, region_valid=region_valid,
            prompt_logits=prompt["prompt_logits"], p_global=p_global, s_region=s_region,
            p_prompt_spoof=p_prompt_spoof, s_final=s_final,
            confidence_features=self.confidence(p_global, s_region, s_final),
            # Every fusion component and auxiliary feature the declared losses and
            # the M10 ablations need, exposed by name rather than re-derived.
            aux={"global_embedding": global_pooled, "local_tokens": local_tokens,
                 "normalized_distances": self.normalize_distance(distances),
                 "region_prompt_logits": prompt["region_prompt_logits"],
                 "prompt_region_embeddings": prompt["prompt_region_embeddings"],
                 "prompt_applicable": prompt["prompt_applicable"],
                 "distance_scale": self.distance_scale.detach().clone()}).validate()

    # --- optimization -------------------------------------------------------
    def parameter_groups(self, *, backbone_lr: float, head_lr: float, weight_decay: float) -> list[dict[str, Any]]:
        """Spec section 10.3: backbone LR 1e-5, heads/manifold LR 1e-4.

        The frozen SigLIP2 tower contributes no parameters at all; the manifold
        lives in buffers, so it contributes none either.
        """
        backbone = list(self.local_backbone.parameters())
        if self.highpass_stem is not None: backbone += list(self.highpass_stem.parameters())
        backbone_ids = {id(parameter) for parameter in backbone}
        heads = [parameter for parameter in self.parameters() if id(parameter) not in backbone_ids]
        return [{"params": backbone, "lr": float(backbone_lr), "weight_decay": float(weight_decay), "name": "backbone"},
                {"params": heads, "lr": float(head_lr), "weight_decay": float(weight_decay), "name": "heads"}]

    def parameter_counts(self) -> dict[str, int]:
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        frozen_tower = (sum(parameter.numel() for parameter in self._global.module.parameters())
                        if self._global.module is not None else 0)
        return {"trainable": trainable,
                "total_registered": sum(parameter.numel() for parameter in self.parameters()),
                "local_backbone": sum(parameter.numel() for parameter in self.local_backbone.parameters()),
                "frozen_global_tower": frozen_tower,
                "manifold_buffer_elements": int(self.manifold.centers.numel() + self.manifold.variances.numel())}

    def architecture_identity(self) -> str:
        payload = {**self.config.payload(), "local_channels": self.local_channels,
                   "local_reduction": self.local_reduction, "global_dim": self.global_dim,
                   "global_patch_tokens": self.global_patch_tokens,
                   "n_prompt": self.prompt_head.n_prompt, "text_dim": self.prompt_head.text_dim,
                   "text_cache_identity": self.text_cache_identity}
        return architecture_identity(payload)


def build_detector(config: DetectorConfig, *, text_embeddings: torch.Tensor, siglip: Any = None,
                   local_weight_file: str | Path | None = None, text_cache_identity: str = "",
                   device: str = "cpu") -> PRISMDetector:
    """Assemble the detector with the frozen SigLIP2 vision tower attached.

    `siglip` is a `SigLIP2Artifacts` whose SHA-256s were already verified; passing
    `None` builds the trainable half only, which is what the toy tests use.
    """
    tower = None
    if siglip is not None:
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
            "region_order": list(REGION_ORDER)}
