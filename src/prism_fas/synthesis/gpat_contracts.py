from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Any
import torch

GPAT_SCHEMA_VERSION = "m8-gpat-v1"
GPAT_CHECKPOINT_SCHEMA_VERSION = "m8-gpat-ckpt-v1"
INPUT_SIZE = 224
BAND_SIZE = 112
HIGH_CHANNELS = 9
CONDITIONING_DIM = 41
LL_INVARIANT_TOLERANCE = 1.0e-5
OUTSIDE_MASK_TOLERANCE = 0.0
# ΔLL is structurally absent: no field, head or output key exists for it.
FORBIDDEN_OUTPUT_FIELDS = ("delta_LL", "delta_ll", "ll_residual", "low_frequency_delta")


class GPATContractError(ValueError):
    """A GPAT tensor or batch violates the frozen M8 contract."""


def check_image(tensor: torch.Tensor, name: str, *, size: int = INPUT_SIZE) -> torch.Tensor:
    if tensor.dim() != 4 or tensor.shape[1] != 3: raise GPATContractError(f"{name} must be [B,3,H,W], got {tuple(tensor.shape)}")
    if tensor.shape[2] != size or tensor.shape[3] != size:
        raise GPATContractError(f"{name} must be {size}x{size}, got {tuple(tensor.shape[2:])}")
    return tensor


def check_mask(tensor: torch.Tensor, name: str, *, size: int = INPUT_SIZE) -> torch.Tensor:
    if tensor.dim() != 4 or tensor.shape[1] != 1: raise GPATContractError(f"{name} must be [B,1,H,W], got {tuple(tensor.shape)}")
    if tensor.shape[2] != size or tensor.shape[3] != size:
        raise GPATContractError(f"{name} must be {size}x{size}, got {tuple(tensor.shape[2:])}")
    unique = torch.unique(tensor.detach())
    if unique.numel() and not bool(((unique == 0) | (unique == 1)).all()):
        raise GPATContractError(f"{name} must be binary 0/1")
    if float(tensor.sum().item()) <= 0.0:
        raise GPATContractError(f"{name} has zero support; the pair must be failed, not silently zero-weighted")
    return tensor


def check_conditioning(tensor: torch.Tensor, name: str = "recipe_conditioning") -> torch.Tensor:
    if tensor.dim() != 2 or tensor.shape[1] != CONDITIONING_DIM:
        raise GPATContractError(f"{name} must be [B,{CONDITIONING_DIM}], got {tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all()): raise GPATContractError(f"{name} is not finite")
    return tensor


@dataclass
class GPATBatch:
    """One materialized GPAT training batch. Source-only by construction."""
    live_image: torch.Tensor
    source_spoof_image: torch.Tensor
    recipe_conditioning: torch.Tensor
    target_support_mask: torch.Tensor
    source_style_mask: torch.Tensor
    recipe_strength: torch.Tensor          # [B] mean requested artifact strength
    live_identity_embedding: torch.Tensor  # [B,512] cached, stop-gradient
    pair_ids: tuple[str, ...] = ()
    live_sample_ids: tuple[str, ...] = ()
    spoof_sample_ids: tuple[str, ...] = ()
    recipe_ids: tuple[str, ...] = ()

    def validate(self) -> "GPATBatch":
        check_image(self.live_image, "live_image")
        check_image(self.source_spoof_image, "source_spoof_image")
        check_conditioning(self.recipe_conditioning)
        check_mask(self.target_support_mask, "target_support_mask")
        check_mask(self.source_style_mask, "source_style_mask")
        if self.recipe_strength.dim() != 1: raise GPATContractError("recipe_strength must be [B]")
        if self.live_identity_embedding.dim() != 2 or self.live_identity_embedding.shape[1] != 512:
            raise GPATContractError("live_identity_embedding must be [B,512]")
        return self

    def to(self, device: torch.device | str) -> "GPATBatch":
        moved = {}
        for item in fields(self):
            value = getattr(self, item.name)
            moved[item.name] = value.to(device) if isinstance(value, torch.Tensor) else value
        return GPATBatch(**moved)

    @property
    def batch_size(self) -> int: return int(self.live_image.shape[0])


@dataclass
class GPATOutput:
    """Model output. There is deliberately no ΔLL field: low-frequency geometry
    is hard-locked by the architecture, not by a loss."""
    synthetic_image: torch.Tensor          # [B,3,224,224] in [0,1]
    delta_high: torch.Tensor               # [B,9,112,112] bounded
    artifact_map: torch.Tensor             # [B,1,224,224] in [0,1]
    artifact_map_half: torch.Tensor        # [B,1,112,112] in [0,1]
    target_support_mask: torch.Tensor
    source_style_mask: torch.Tensor
    live_bands: dict[str, torch.Tensor]    # {"LL","high"}
    generated_bands: dict[str, torch.Tensor]
    artifact_latent: torch.Tensor          # [B,128]
    recipe_latent: torch.Tensor            # [B,64]
    pre_composite_image: torch.Tensor
    trace: dict[str, Any] = field(default_factory=dict)

    def ll_invariant_error(self) -> float:
        return float((self.generated_bands["LL"] - self.live_bands["LL"]).abs().max().item())

    def outside_mask_error(self, live_image: torch.Tensor) -> float:
        outside = (self.target_support_mask < 0.5)
        if not bool(outside.any()): return 0.0
        difference = (self.synthetic_image - live_image).abs()
        return float(difference.masked_select(outside.expand_as(difference)).max().item())

    def validate(self, live_image: torch.Tensor) -> "GPATOutput":
        check_image(self.synthetic_image, "synthetic_image")
        if not bool(torch.isfinite(self.synthetic_image).all()): raise GPATContractError("synthetic_image is not finite")
        detached = self.synthetic_image.detach()
        low, high = float(detached.min()), float(detached.max())
        if low < -1e-6 or high > 1.0 + 1e-6: raise GPATContractError(f"synthetic_image outside [0,1]: [{low},{high}]")
        error = self.ll_invariant_error()
        if error > LL_INVARIANT_TOLERANCE:
            raise GPATContractError(f"LL invariant violated: {error} > {LL_INVARIANT_TOLERANCE}")
        outside = self.outside_mask_error(live_image)
        if outside > OUTSIDE_MASK_TOLERANCE:
            raise GPATContractError(f"outside-support difference {outside} is not exactly zero")
        for name in FORBIDDEN_OUTPUT_FIELDS:
            if hasattr(self, name): raise GPATContractError(f"GPATOutput must not expose {name}")
        return self
