from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from .dwt import dwt_bands
from .gpat_contracts import GPATBatch, GPATContractError, GPATOutput
from .gpat_model import downsample_mask

LOSS_SCHEMA_VERSION = "m8-gpat-loss-v1"
DEFAULT_WEIGHTS = {"style": 1.0, "identity": 0.5, "map": 0.5, "strength": 0.25,
                   "total_variation": 0.02, "residual": 0.01}
EPS = 1.0e-6


class GPATLossError(ValueError):
    """A loss term could not be computed under the declared contract."""


def masked_channel_stats(bands: torch.Tensor, mask_half: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel masked mean and standard deviation at band resolution.

    A sample with no valid mask pixel raises: the contract forbids silently
    turning an empty mask into a zero loss.
    """
    if mask_half.shape[-2:] != bands.shape[-2:]:
        raise GPATLossError(f"mask {tuple(mask_half.shape[-2:])} does not match bands {tuple(bands.shape[-2:])}")
    mask = mask_half.to(bands.dtype)
    count = mask.sum(dim=(2, 3), keepdim=True)
    if bool((count <= 0).any()):
        raise GPATLossError("a sample has no valid mask pixels; the pair must be failed, not zero-weighted")
    mean = (bands * mask).sum(dim=(2, 3), keepdim=True) / count
    variance = (((bands - mean) ** 2) * mask).sum(dim=(2, 3), keepdim=True) / count
    return mean.squeeze(-1).squeeze(-1), variance.clamp_min(0.0).add(EPS).sqrt().squeeze(-1).squeeze(-1)


def style_loss(generated: torch.Tensor, spoof: torch.Tensor, target_mask: torch.Tensor,
               style_mask: torch.Tensor) -> torch.Tensor:
    """L1 between the masked first and second moments of the 9 high-frequency
    channels of the generated image and of the real spoof source."""
    _, high_generated = dwt_bands(generated)
    _, high_spoof = dwt_bands(spoof)
    mean_generated, std_generated = masked_channel_stats(high_generated, downsample_mask(target_mask))
    mean_spoof, std_spoof = masked_channel_stats(high_spoof, downsample_mask(style_mask))
    return (mean_generated - mean_spoof).abs().mean() + (std_generated - std_spoof).abs().mean()


def identity_loss(generated_embedding: torch.Tensor, live_embedding: torch.Tensor) -> torch.Tensor:
    """1 - cosine to the cached live embedding. AdaFace itself is frozen and the
    live embedding is detached, so no gradient reaches the identity model."""
    reference = live_embedding.detach()
    cosine = torch.nn.functional.cosine_similarity(generated_embedding.float(), reference.float(), dim=1, eps=EPS)
    return (1.0 - cosine).mean()


def artifact_map_loss(artifact_map: torch.Tensor, target_mask: torch.Tensor,
                      recipe_strength: torch.Tensor) -> torch.Tensor:
    target = target_mask.to(artifact_map.dtype) * recipe_strength.to(artifact_map.dtype).view(-1, 1, 1, 1)
    return ((artifact_map - target) ** 2).mean()


def strength_loss(artifact_map: torch.Tensor, target_mask: torch.Tensor,
                  recipe_strength: torch.Tensor) -> torch.Tensor:
    mask = target_mask.to(artifact_map.dtype)
    count = mask.sum(dim=(1, 2, 3))
    if bool((count <= 0).any()): raise GPATLossError("artifact-strength loss received an empty support mask")
    measured = (artifact_map * mask).sum(dim=(1, 2, 3)) / count
    return (measured - recipe_strength.to(measured.dtype)).abs().mean()


def residual_loss(delta_high: torch.Tensor) -> torch.Tensor:
    return delta_high.abs().mean()


def total_variation_loss(delta_high: torch.Tensor, mask_half: torch.Tensor) -> torch.Tensor:
    masked = delta_high * mask_half.to(delta_high.dtype)
    horizontal = (masked[:, :, :, 1:] - masked[:, :, :, :-1]).abs().mean()
    vertical = (masked[:, :, 1:, :] - masked[:, :, :-1, :]).abs().mean()
    return horizontal + vertical


@dataclass
class GPATLossResult:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    weights: dict[str, float]
    metrics: dict[str, float]

    def detached(self) -> dict[str, float]:
        out = {"total": float(self.total.detach().item())}
        out.update({name: float(value.detach().item()) for name, value in self.components.items()})
        out.update(self.metrics)
        return out


def compute_losses(output: GPATOutput, batch: GPATBatch, generated_embedding: torch.Tensor,
                   weights: dict[str, float] | None = None) -> GPATLossResult:
    """The exact M8 GPAT objective. No adversarial and no classification term.

    `L_LL` and the outside-support condition are asserted invariants, not soft
    losses: LL is hard-locked by the architecture and the composite is exact.
    """
    resolved = {**DEFAULT_WEIGHTS, **(weights or {})}
    components = {
        "style": style_loss(output.synthetic_image, batch.source_spoof_image,
                            batch.target_support_mask, batch.source_style_mask),
        "identity": identity_loss(generated_embedding, batch.live_identity_embedding),
        "map": artifact_map_loss(output.artifact_map, batch.target_support_mask, batch.recipe_strength),
        "strength": strength_loss(output.artifact_map, batch.target_support_mask, batch.recipe_strength),
        "total_variation": total_variation_loss(output.delta_high, downsample_mask(batch.target_support_mask)),
        "residual": residual_loss(output.delta_high)}
    for name, value in components.items():
        if value.dim() != 0: raise GPATLossError(f"loss component {name} is not a scalar")
        if not bool(torch.isfinite(value)): raise GPATLossError(f"loss component {name} is not finite")
    total = sum(float(resolved[name]) * value for name, value in components.items())
    if not bool(torch.isfinite(total)): raise GPATLossError("total loss is not finite")
    mask = batch.target_support_mask
    measured = float(((output.artifact_map * mask).sum() / mask.sum().clamp_min(1.0)).detach().item())
    with torch.no_grad():
        cosine = torch.nn.functional.cosine_similarity(generated_embedding.float(),
                                                       batch.live_identity_embedding.float(), dim=1, eps=EPS)
    metrics = {"ll_invariant_max_abs_error": output.ll_invariant_error(),
               "outside_mask_max_abs_error": output.outside_mask_error(batch.live_image),
               "identity_cosine": float(cosine.mean().item()),
               "measured_artifact_strength": measured,
               "requested_artifact_strength": float(batch.recipe_strength.mean().item()),
               "delta_high_abs_max": float(output.delta_high.detach().abs().max().item())}
    return GPATLossResult(total=total, components=components, weights=resolved, metrics=metrics)


def assert_invariants(result: GPATLossResult, *, ll_tolerance: float, outside_tolerance: float = 0.0) -> None:
    if result.metrics["ll_invariant_max_abs_error"] > ll_tolerance:
        raise GPATContractError(f"LL invariant violated: {result.metrics['ll_invariant_max_abs_error']} > {ll_tolerance}")
    if result.metrics["outside_mask_max_abs_error"] > outside_tolerance:
        raise GPATContractError(f"outside-support difference {result.metrics['outside_mask_max_abs_error']} "
                                f"exceeds {outside_tolerance}")


def loss_manifest(weights: dict[str, float] | None = None) -> dict[str, Any]:
    resolved = {**DEFAULT_WEIGHTS, **(weights or {})}
    return {"loss_schema_version": LOSS_SCHEMA_VERSION, "weights": resolved,
            "formulas": {
                "style": "mean|mu_high(G,M_t)-mu_high(S,M_s)| + mean|std_high(G,M_t)-std_high(S,M_s)|",
                "identity": "mean(1 - cos(e_G, stopgrad(e_L)))",
                "map": "mean((A_full - M_t*a_recipe)^2)",
                "strength": "mean|masked_mean(A_full,M_t) - a_recipe|",
                "total_variation": "TV(delta_high * M_half)",
                "residual": "mean|delta_high|"},
            "asserted_invariants": {"L_LL": "max|LL_generated - LL_live| <= tolerance (architectural hard lock)",
                                    "L_outside": "max|G - L| outside M_t == 0 (exact composite)"},
            "adversarial": False, "detector_classification": False}
