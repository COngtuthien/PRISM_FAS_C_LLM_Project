from __future__ import annotations
import torch

DWT_CONVENTION = "orthonormal_haar_v1"
DWT_RECONSTRUCTION_TOLERANCE_FP32 = 1.0e-6
BAND_ORDER = ("LH", "HL", "HH")


class DWTError(ValueError):
    """A tensor does not satisfy the Haar DWT shape contract."""


def _check(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.dim() != 4: raise DWTError(f"{name} must be [B,C,H,W], got {tuple(tensor.shape)}")
    if tensor.shape[-1] % 2 or tensor.shape[-2] % 2:
        raise DWTError(f"{name} spatial dims must be even, got {tuple(tensor.shape[-2:])}")
    return tensor


def haar_dwt2(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Orthonormal separable Haar DWT.

    Pure PyTorch and differentiable: no PyWavelets call ever runs inside a model
    forward. Works on CPU and CUDA, under fp32 and AMP autocast.

        LL = (x00 + x01 + x10 + x11) / 2
        LH = (x00 + x01 - x10 - x11) / 2
        HL = (x00 - x01 + x10 - x11) / 2
        HH = (x00 - x01 - x10 + x11) / 2
    """
    _check(image, "image")
    x00 = image[..., 0::2, 0::2]
    x01 = image[..., 0::2, 1::2]
    x10 = image[..., 1::2, 0::2]
    x11 = image[..., 1::2, 1::2]
    half = x00.new_tensor(0.5)
    ll = (x00 + x01 + x10 + x11) * half
    lh = (x00 + x01 - x10 - x11) * half
    hl = (x00 - x01 + x10 - x11) * half
    hh = (x00 - x01 - x10 + x11) * half
    return ll, lh, hl, hh


def haar_idwt2(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
    """Exact inverse of `haar_dwt2` (the transform is its own transpose)."""
    for name, band in (("LL", ll), ("LH", lh), ("HL", hl), ("HH", hh)):
        if band.dim() != 4: raise DWTError(f"{name} must be [B,C,H,W], got {tuple(band.shape)}")
        if band.shape != ll.shape: raise DWTError(f"{name} shape {tuple(band.shape)} != LL {tuple(ll.shape)}")
    half = ll.new_tensor(0.5)
    x00 = (ll + lh + hl + hh) * half
    x01 = (ll + lh - hl - hh) * half
    x10 = (ll - lh + hl - hh) * half
    x11 = (ll - lh - hl + hh) * half
    batch, channels, height, width = ll.shape
    out = ll.new_empty((batch, channels, height * 2, width * 2))
    out[..., 0::2, 0::2] = x00
    out[..., 0::2, 1::2] = x01
    out[..., 1::2, 0::2] = x10
    out[..., 1::2, 1::2] = x11
    return out


def pack_high(lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
    """[B,9,H/2,W/2] in the fixed order LH, HL, HH."""
    return torch.cat((lh, hl, hh), dim=1)


def unpack_high(high: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if high.shape[1] % 3: raise DWTError(f"high band channel count {high.shape[1]} is not divisible by 3")
    step = high.shape[1] // 3
    return high[:, :step], high[:, step:2 * step], high[:, 2 * step:]


def dwt_bands(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience: returns (LL [B,3,H/2,W/2], high [B,9,H/2,W/2])."""
    ll, lh, hl, hh = haar_dwt2(image)
    return ll, pack_high(lh, hl, hh)


def idwt_bands(ll: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    lh, hl, hh = unpack_high(high)
    return haar_idwt2(ll, lh, hl, hh)


def reconstruction_error(image: torch.Tensor) -> float:
    ll, high = dwt_bands(image)
    return float((idwt_bands(ll, high) - image).abs().max().item())
