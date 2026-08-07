"""M9 region priors and the RegionQuery / CrossAttention / RegionPool stack.

Spec section 9.1: the region prior is a SOFT MASK / QUERY INITIALIZATION, never nine
hard-cropped images. Spec section 9.2 (Table 32):

    q_r = RegionQuery(parsing_r, landmarks_r, learnable_token_r)
    z_r = CrossAttention(q_r, K=F_local, V=F_local) + RegionPool(T_global, parsing_r)

Priors are built from the frozen M3B parsing/landmark priors through the exact M7
`RegionMaskBuilder`, so M7, M8 and M9 all place a region in the same pixels.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from .contracts import REGION_COUNT, REGION_ORDER, DetectorContractError

# A soft prior is the region mask blurred so a query attends to a neighbourhood
# rather than a hard boundary. The mask itself is the frozen M7 geometry.
PRIOR_BLUR_SIGMA = 2.0


def build_region_priors(parsing_labels: np.ndarray, landmarks: np.ndarray, bbox: np.ndarray,
                        crop_box: np.ndarray, *, height: int = 224, width: int = 224) -> np.ndarray:
    """Nine soft region priors in `[R,H,W]`, from the exact M7 mask builder.

    Reuses `synthesis.masks.RegionMaskBuilder` rather than re-deriving geometry, so
    the M9 region for a sample is pixel-identical to the M7/M8 region for the same
    sample.
    """
    from prism_fas.synthesis.masks import RegionMaskBuilder
    builder = RegionMaskBuilder(height=height, width=width, parsing=np.asarray(parsing_labels),
                                landmarks=np.asarray(landmarks, dtype=np.float32),
                                bbox=np.asarray(bbox, dtype=np.float32),
                                crop_box=np.asarray(crop_box, dtype=np.float32))
    priors = np.zeros((REGION_COUNT, height, width), dtype=np.float32)
    for index, name in enumerate(REGION_ORDER):
        mask, _ = builder.region(name)
        priors[index] = np.asarray(mask, dtype=np.float32)
    return _soft(priors)


def _soft(priors: np.ndarray) -> np.ndarray:
    """Separable Gaussian smoothing with reflect padding, deterministic in numpy."""
    radius = int(np.ceil(3.0 * PRIOR_BLUR_SIGMA))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets ** 2) / (2.0 * PRIOR_BLUR_SIGMA ** 2))
    kernel = (kernel / kernel.sum()).astype(np.float32)
    out = priors
    for axis in (1, 2):
        padding = [(0, 0)] * out.ndim
        padding[axis] = (radius, radius)
        padded = np.pad(out, padding, mode="reflect")
        accumulated = np.zeros_like(out)
        for index, weight in enumerate(kernel):
            window = [slice(None)] * out.ndim
            window[axis] = slice(index, index + out.shape[axis])
            accumulated = accumulated + np.float32(weight) * padded[tuple(window)]
        out = accumulated
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def resize_priors(priors: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """`[B,R,H,W]` -> `[B,R,h,w]` on the feature grid. Area interpolation keeps the
    prior's mass rather than sampling it, which matters for small regions."""
    if priors.dim() != 4: raise DetectorContractError("region priors must be [B,R,H,W]")
    if tuple(priors.shape[-2:]) == tuple(size): return priors
    return torch.nn.functional.interpolate(priors, size=size, mode="area")


def attack_region_mask(exact_mask: np.ndarray, priors: np.ndarray, *, overlap: float = 0.05) -> np.ndarray:
    """`m_r` for spec section 10.1: region `r` is attacked when the M8 exact edit
    mask covers at least `overlap` of the region's prior mass.

    Derived from the frozen M8 mask, never from `q` and never from a label.
    """
    changed = np.asarray(exact_mask).astype(bool)
    mass = priors.reshape(REGION_COUNT, -1).sum(axis=1)
    hit = (priors.reshape(REGION_COUNT, -1) * changed.reshape(1, -1)).sum(axis=1)
    ratio = np.divide(hit, np.maximum(mass, 1e-6))
    return (ratio >= float(overlap)).astype(np.float32)


class RegionQuery(nn.Module):
    """`q_r = RegionQuery(parsing_r, landmarks_r, learnable_token_r)`.

    One learnable token per region, conditioned on that region's own soft prior
    statistics, so the nine queries start apart instead of collapsing to one.
    """

    def __init__(self, dim: int, *, regions: int = REGION_COUNT):
        super().__init__()
        self.dim, self.regions = int(dim), int(regions)
        self.token = nn.Parameter(torch.randn(regions, dim) * 0.02)
        self.prior_projection = nn.Linear(4, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, priors: torch.Tensor) -> torch.Tensor:
        """priors `[B,R,h,w]` -> queries `[B,R,D]`."""
        batch, regions, height, width = priors.shape
        if regions != self.regions: raise DetectorContractError("region count mismatch in RegionQuery")
        flat = priors.reshape(batch, regions, -1)
        total = flat.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        rows = torch.linspace(0.0, 1.0, height, device=priors.device, dtype=priors.dtype)
        columns = torch.linspace(0.0, 1.0, width, device=priors.device, dtype=priors.dtype)
        grid_y = rows[:, None].expand(height, width).reshape(1, 1, -1)
        grid_x = columns[None, :].expand(height, width).reshape(1, 1, -1)
        # Mass, centroid and spread: a compact, differentiable description of where
        # the region is, without cropping the image.
        centre_y = (flat * grid_y).sum(dim=-1, keepdim=True) / total
        centre_x = (flat * grid_x).sum(dim=-1, keepdim=True) / total
        spread = ((flat * (grid_y - centre_y) ** 2).sum(dim=-1, keepdim=True) / total).sqrt()
        statistics = torch.cat([total / float(height * width), centre_y, centre_x, spread], dim=-1)
        return self.norm(self.token.unsqueeze(0) + self.prior_projection(statistics))


class RegionCrossAttention(nn.Module):
    """`CrossAttention(q_r, K=F_local, V=F_local)`, with the region prior as an
    additive attention bias so a query is anchored to its own region."""

    def __init__(self, dim: int, *, heads: int = 4):
        super().__init__()
        if dim % heads: raise DetectorContractError("region dim must divide the head count")
        self.dim, self.heads, self.head_dim = int(dim), int(heads), int(dim) // int(heads)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.project = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, queries: torch.Tensor, features: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        """queries `[B,R,D]`, features `[B,P,D]`, priors `[B,R,P]` -> `[B,R,D]`."""
        batch, regions, _ = queries.shape
        tokens = features.shape[1]
        shape = lambda value, count: value.reshape(batch, count, self.heads, self.head_dim).transpose(1, 2)
        q = shape(self.to_q(queries), regions)
        k = shape(self.to_k(features), tokens)
        v = shape(self.to_v(features), tokens)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # log-prior bias: inside the region the bias is ~0, far outside it is very
        # negative, so attention cannot wander to an unrelated region.
        scores = scores + torch.log(priors.clamp_min(1e-4)).unsqueeze(1)
        attended = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(batch, regions, self.dim)
        return self.norm(queries + self.project(attended))


class RegionPool(nn.Module):
    """`RegionPool(T_global, parsing_r)`: prior-weighted mean of the frozen global
    patch tokens, projected into the region embedding space."""

    def __init__(self, global_dim: int, dim: int):
        super().__init__()
        self.project = nn.Linear(global_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        """tokens `[B,P,Dg]`, priors `[B,R,P]` -> `[B,R,D]`."""
        weights = priors / priors.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return self.norm(self.project(weights @ tokens))
