from __future__ import annotations
import torch, torch.nn.functional as F

class LossContractError(ValueError):
    """The loss received inputs outside the B00 contract."""
def b00_binary_cross_entropy(spoof_logit:torch.Tensor,target:torch.Tensor)->torch.Tensor:
    """B00's only loss: mean binary cross-entropy with logits.

    L_i = -[y_i log(sigmoid(z_i)) + (1-y_i) log(1-sigmoid(z_i))], reduced by mean.
    No class weighting, pos_weight, focal, label smoothing or auxiliary terms:
    the M4 sampler already balances every batch across domain and class.
    """
    if spoof_logit.ndim!=1: raise LossContractError(f"spoof_logit must be [B], got {tuple(spoof_logit.shape)}")
    if target.shape!=spoof_logit.shape: raise LossContractError("target shape must match spoof_logit")
    values=torch.unique(target)
    if bool(((values!=0)&(values!=1)).any()): raise LossContractError("target must contain only 0 (live) and 1 (spoof)")
    if not torch.isfinite(spoof_logit).all(): raise LossContractError("spoof_logit contains non-finite values")
    return F.binary_cross_entropy_with_logits(spoof_logit,target.float(),reduction="mean")
