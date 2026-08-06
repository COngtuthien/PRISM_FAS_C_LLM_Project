from __future__ import annotations
import hashlib
from typing import Any
import torch
from torch import nn
from .dwt import DWT_CONVENTION, dwt_bands, idwt_bands
from .gpat_contracts import (BAND_SIZE, CONDITIONING_DIM, GPATBatch, GPATOutput, HIGH_CHANNELS, INPUT_SIZE,
                             check_conditioning, check_image, check_mask)

GPAT_ARCHITECTURE_VERSION = "m8-gpat-arch-v1"
DEFAULT_MAX_HIGH_FREQUENCY_DELTA = 0.15


def downsample_mask(mask: torch.Tensor) -> torch.Tensor:
    """Binary 2x2 max-pool to the band resolution, re-binarized.

    Max-pool (not average) so a thin region never vanishes at half resolution.
    """
    half = torch.nn.functional.max_pool2d(mask, kernel_size=2, stride=2)
    return (half > 0.5).to(mask.dtype)


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)


class ArtifactEncoder(nn.Module):
    """Encodes the spoof source's artifact appearance into z_a [B,128]."""
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        channels = (3, 32, 64, 128, 256)
        blocks: list[nn.Module] = []
        for index in range(4):
            blocks += [nn.Conv2d(channels[index], channels[index + 1], 3, stride=2, padding=1),
                       _group_norm(channels[index + 1]), nn.SiLU()]
        self.features = nn.Sequential(*blocks)
        self.project = nn.Linear(channels[-1], latent_dim)

    def forward(self, image: torch.Tensor, style_mask: torch.Tensor) -> torch.Tensor:
        features = self.features(image)
        mask = torch.nn.functional.interpolate(style_mask.to(features.dtype), size=features.shape[-2:], mode="nearest")
        weight = mask.sum(dim=(1, 2, 3), keepdim=True)
        # masked global average pool; an all-zero mask cannot reach here (the
        # batch contract rejects it), but the guard keeps the graph finite.
        pooled = torch.where(weight > 0, (features * mask).sum(dim=(2, 3), keepdim=True) / weight.clamp_min(1.0),
                             features.mean(dim=(2, 3), keepdim=True))
        return self.project(pooled.flatten(1))


class RecipeEncoder(nn.Module):
    """41-d conditioning vector -> z_recipe [B,64]."""
    def __init__(self, latent_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(CONDITIONING_DIM, hidden), nn.SiLU(), nn.Linear(hidden, latent_dim))

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        return self.net(conditioning)


class FiLMResidualBlock(nn.Module):
    def __init__(self, channels: int, conditioning_dim: int):
        super().__init__()
        self.norm_in, self.norm_mid = _group_norm(channels), _group_norm(channels)
        self.conv_in = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv_out = nn.Conv2d(channels, channels, 3, padding=1)
        self.film = nn.Sequential(nn.Linear(conditioning_dim, 128), nn.SiLU(), nn.Linear(128, channels * 2))

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(conditioning).chunk(2, dim=1)
        h = self.conv_in(self.norm_in(torch.nn.functional.silu(x)))
        h = h * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        h = self.conv_out(torch.nn.functional.silu(self.norm_mid(h)))
        return x + h


class GPATResidualModel(nn.Module):
    """Recipe-conditioned geometry-preserving artifact transfer.

    The generator only ever produces a bounded **high-frequency** residual. LL is
    passed through untouched, so low-frequency geometry is hard-locked by the
    architecture and there is no ΔLL parameter, head or output field to disable.
    No discriminator and no GAN loss.
    """
    def __init__(self, *, artifact_latent_dim: int = 128, recipe_latent_dim: int = 64, base_channels: int = 64,
                 film_blocks: int = 4, max_high_frequency_delta: float = DEFAULT_MAX_HIGH_FREQUENCY_DELTA):
        super().__init__()
        self.artifact_latent_dim, self.recipe_latent_dim = int(artifact_latent_dim), int(recipe_latent_dim)
        self.base_channels, self.film_blocks = int(base_channels), int(film_blocks)
        self.max_high_frequency_delta = float(max_high_frequency_delta)
        self.artifact_encoder = ArtifactEncoder(self.artifact_latent_dim)
        self.recipe_encoder = RecipeEncoder(self.recipe_latent_dim)
        conditioning_dim = self.artifact_latent_dim + self.recipe_latent_dim
        # 3 LL + 9 live high-frequency + 1 downsampled support = 13 channels
        self.stem = nn.Conv2d(3 + HIGH_CHANNELS + 1, self.base_channels, 3, padding=1)
        self.blocks = nn.ModuleList([FiLMResidualBlock(self.base_channels, conditioning_dim) for _ in range(self.film_blocks)])
        self.delta_head = nn.Conv2d(self.base_channels, HIGH_CHANNELS, 3, padding=1)
        self.artifact_map_head = nn.Conv2d(self.base_channels, 1, 3, padding=1)

    # --- identity -----------------------------------------------------------
    def architecture_payload(self) -> dict[str, Any]:
        return {"architecture_version": GPAT_ARCHITECTURE_VERSION, "dwt_convention": DWT_CONVENTION,
                "input_size": INPUT_SIZE, "band_size": BAND_SIZE, "high_channels": HIGH_CHANNELS,
                "conditioning_dim": CONDITIONING_DIM, "artifact_latent_dim": self.artifact_latent_dim,
                "recipe_latent_dim": self.recipe_latent_dim, "base_channels": self.base_channels,
                "film_blocks": self.film_blocks, "max_high_frequency_delta": self.max_high_frequency_delta,
                "delta_ll_enabled": False,
                "parameter_shapes": {name: list(tensor.shape) for name, tensor in sorted(self.state_dict().items())}}

    def architecture_hash(self) -> str:
        import json
        raw = json.dumps(self.architecture_payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))

    def parameter_groups(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        optimizer = config["optimizer"]
        return [{"params": list(self.artifact_encoder.parameters()), "lr": float(optimizer["encoder_lr"]), "name": "artifact_encoder"},
                {"params": list(self.recipe_encoder.parameters()), "lr": float(optimizer["recipe_lr"]), "name": "recipe_encoder"},
                {"params": list(self.stem.parameters()) + list(self.blocks.parameters())
                           + list(self.delta_head.parameters()) + list(self.artifact_map_head.parameters()),
                 "lr": float(optimizer["generator_lr"]), "name": "generator"}]

    # --- forward ------------------------------------------------------------
    def forward(self, live_image: torch.Tensor, source_spoof_image: torch.Tensor, recipe_conditioning: torch.Tensor,
                target_support_mask: torch.Tensor, source_style_mask: torch.Tensor) -> GPATOutput:
        check_image(live_image, "live_image"); check_image(source_spoof_image, "source_spoof_image")
        check_conditioning(recipe_conditioning)
        check_mask(target_support_mask, "target_support_mask"); check_mask(source_style_mask, "source_style_mask")
        ll_live, high_live = dwt_bands(live_image)
        support_half = downsample_mask(target_support_mask).to(high_live.dtype)
        z_artifact = self.artifact_encoder(source_spoof_image, source_style_mask)
        z_recipe = self.recipe_encoder(recipe_conditioning)
        conditioning = torch.cat((z_artifact, z_recipe), dim=1)
        hidden = self.stem(torch.cat((ll_live, high_live, support_half), dim=1))
        for block in self.blocks: hidden = block(hidden, conditioning)
        artifact_map_half = torch.sigmoid(self.artifact_map_head(hidden)) * support_half
        delta_high = torch.tanh(self.delta_head(hidden)) * artifact_map_half * self.max_high_frequency_delta
        high_out = high_live + delta_high
        # LL is passed through unchanged: the low-frequency band is hard-locked.
        generated = idwt_bands(ll_live, high_out)
        pre_composite = generated
        generated = torch.where(target_support_mask > 0.5, generated, live_image)
        generated = generated.clamp(0.0, 1.0)
        # clamp can only move pixels inside the support (outside pixels already
        # equal the live image, which is in range), but re-compose to be exact.
        generated = torch.where(target_support_mask > 0.5, generated, live_image)
        artifact_map = torch.nn.functional.interpolate(artifact_map_half, size=(INPUT_SIZE, INPUT_SIZE),
                                                       mode="nearest") * target_support_mask
        ll_generated, high_generated = dwt_bands(pre_composite)
        return GPATOutput(
            synthetic_image=generated, delta_high=delta_high, artifact_map=artifact_map,
            artifact_map_half=artifact_map_half, target_support_mask=target_support_mask,
            source_style_mask=source_style_mask,
            live_bands={"LL": ll_live, "high": high_live},
            generated_bands={"LL": ll_generated, "high": high_generated},
            artifact_latent=z_artifact, recipe_latent=z_recipe, pre_composite_image=pre_composite,
            trace={"architecture_version": GPAT_ARCHITECTURE_VERSION, "dwt_convention": DWT_CONVENTION,
                   "max_high_frequency_delta": self.max_high_frequency_delta,
                   "support_pixels": int(target_support_mask.sum().item()),
                   "style_pixels": int(source_style_mask.sum().item()),
                   "delta_high_abs_max": float(delta_high.detach().abs().max().item()),
                   "delta_ll_enabled": False})

    def forward_batch(self, batch: GPATBatch) -> GPATOutput:
        return self(batch.live_image, batch.source_spoof_image, batch.recipe_conditioning,
                    batch.target_support_mask, batch.source_style_mask)


def build_gpat_model(config: dict[str, Any]) -> GPATResidualModel:
    model = config.get("model", {})
    return GPATResidualModel(artifact_latent_dim=int(model.get("artifact_latent_dim", 128)),
                             recipe_latent_dim=int(model.get("recipe_latent_dim", 64)),
                             base_channels=int(model.get("base_channels", 64)),
                             film_blocks=int(model.get("film_blocks", 4)),
                             max_high_frequency_delta=float(model.get("max_high_frequency_delta",
                                                                      DEFAULT_MAX_HIGH_FREQUENCY_DELTA)))
