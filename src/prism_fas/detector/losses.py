"""M9 losses. Spec section 10.1 (Table 35), implemented exactly as written.

    L_cls    = CE(p_global, y_image)
    L_local  = weighted_BCE(local_token_logits, artifact_map)
    L_MIL    = CE(LogSumExpMIL(token_logits), y_image)
    L_real   = mean_live sum_r SoftMin_k d_rk
    L_out    = mean_syn  sum_r m_r * max(0, margin_r - d_r)
    L_clean  = mean_syn  sum_r (1-m_r) * min(d_r, clean_cap)
    L_prompt = InfoNCE(z_attack_region, text_embedding(recipe))
    L_cons   = |p_global - stopgrad(s_region)| + |s_region - stopgrad(p_global)|
    L_risk   = Var(domain_risk) + Var(artifact_family_risk)

    L_total = L_cls_real + lambda_syn * q * (L_cls_syn + lambda_local*L_local + lambda_out*L_out)
            + lambda_M*L_real + lambda_clean*L_clean + lambda_MIL*L_MIL
            + lambda_P*L_prompt + lambda_cons*L_cons + lambda_risk*L_risk

Read `L_total` literally: `q` weights ONLY the synthetic bracket. It does not scale
`L_real`, `L_clean`, `L_MIL`, `L_prompt`, `L_cons` or `L_risk`, and never touches
`L_cls_real`. `q` is a weight and never a target (spec section 8, Table 30).

Every term divides by its count of APPLICABLE entries and returns a finite 0 when
that count is 0. Invalid-visibility regions are removed before the denominator is
formed, never zero-filled inside it.
"""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, field
from typing import Any
import torch
from .contracts import LIVE, REGION_COUNT, SPOOF, DetectorContractError

LOSS_SCHEMA_VERSION = "m9-losses-v1"
# Spec section 10.4 (Table 38). Initial values; M9 never tunes them.
DEFAULT_WEIGHTS: dict[str, float] = {
    "lambda_syn": 0.50, "lambda_M": 1.00, "lambda_out": 1.00, "lambda_clean": 0.25,
    "lambda_local": 1.00, "lambda_MIL": 0.50, "lambda_P": 0.20, "lambda_cons": 0.05,
    "lambda_risk": 0.10, "margin_out": 3.0}
# SPEC_UNDERSPECIFIED, see docs/M9_TRAINING_CONTRACT.md section 1.
DEFAULT_CLEAN_CAP = 3.0
DEFAULT_MIL_TEMPERATURE = 1.0
DEFAULT_PROMPT_TEMPERATURE = 0.07
LOSS_NAMES = ("L_cls_real", "L_cls_syn", "L_local", "L_MIL", "L_real", "L_out",
              "L_clean", "L_prompt", "L_cons", "L_risk")


class LossError(ValueError):
    """A loss was given tensors that do not satisfy its declared contract."""


def _zero(reference: torch.Tensor) -> torch.Tensor:
    """A finite, differentiable zero on the right device/dtype."""
    return reference.sum() * 0.0


def _mean_or_zero(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over applicable entries only. Zero-filling inside the denominator would
    silently shrink a term in proportion to how much was inapplicable."""
    keep = mask.bool()
    count = keep.sum()
    if int(count) == 0: return _zero(values)
    return (values * keep.to(values.dtype)).sum() / count.to(values.dtype)


def classification_loss(global_logit: torch.Tensor, label: torch.Tensor,
                        selector: torch.Tensor) -> torch.Tensor:
    """`CE(p_global, y_image)` restricted to `selector`.

    The head is a single logit (the M5 convention, live=0/spoof=1), so cross entropy
    over two classes is BCE-with-logits.
    """
    logit = global_logit.reshape(-1)
    target = label.to(logit.dtype).reshape(-1)
    per_sample = torch.nn.functional.binary_cross_entropy_with_logits(logit, target, reduction="none")
    return _mean_or_zero(per_sample, selector)


def weighted_local_loss(local_logits: torch.Tensor, artifact_map: torch.Tensor,
                        selector: torch.Tensor) -> torch.Tensor:
    """`L_local = weighted_BCE(local_token_logits, artifact_map)`.

    The artifact map is pooled onto the token grid and used both as the BCE target
    and as the per-token weight, so a sparse map is not swamped by background.
    """
    keep = selector.bool()
    if int(keep.sum()) == 0: return _zero(local_logits)
    logits = local_logits[keep]
    tokens = logits.shape[1]
    side = int(round(tokens ** 0.5))
    if side * side != tokens: raise LossError(f"local token count {tokens} is not square")
    target = torch.nn.functional.adaptive_avg_pool2d(artifact_map[keep], (side, side)).reshape(-1, tokens)
    target = target.clamp(0.0, 1.0).to(logits.dtype)
    per_token = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weight = target
    total = weight.sum()
    if float(total) <= 0.0:
        # A synthetic sample whose map is entirely zero contributes ~0, which is the
        # behaviour the required test asserts.
        return per_token.mean() * 0.0
    return (per_token * weight).sum() / total


def mil_loss(local_logits: torch.Tensor, label: torch.Tensor, selector: torch.Tensor, *,
             temperature: float = DEFAULT_MIL_TEMPERATURE) -> torch.Tensor:
    """`L_MIL = CE(LogSumExpMIL(token_logits), y_image)`."""
    pooled = temperature * torch.logsumexp(local_logits / temperature, dim=1)
    per_sample = torch.nn.functional.binary_cross_entropy_with_logits(
        pooled, label.to(pooled.dtype), reduction="none")
    return _mean_or_zero(per_sample, selector)


def real_manifold_loss(soft_min_distance: torch.Tensor, is_live: torch.Tensor,
                       region_valid: torch.Tensor) -> torch.Tensor:
    """`L_real = mean_live sum_r SoftMin_k d_rk`, valid regions only."""
    keep = is_live.bool()
    if int(keep.sum()) == 0: return _zero(soft_min_distance)
    valid = region_valid[keep].to(soft_min_distance.dtype)
    per_sample = (soft_min_distance[keep] * valid).sum(dim=1)
    return per_sample.mean()


def outlier_loss(distance: torch.Tensor, attack_mask: torch.Tensor, is_synthetic: torch.Tensor,
                 region_valid: torch.Tensor, *, margin: float) -> torch.Tensor:
    """`L_out = mean_syn sum_r m_r * max(0, margin_r - d_r)`, valid regions only."""
    keep = is_synthetic.bool()
    if int(keep.sum()) == 0: return _zero(distance)
    applicable = attack_mask[keep].to(distance.dtype) * region_valid[keep].to(distance.dtype)
    hinge = torch.clamp(float(margin) - distance[keep], min=0.0)
    return (hinge * applicable).sum(dim=1).mean()


def clean_loss(distance: torch.Tensor, attack_mask: torch.Tensor, is_synthetic: torch.Tensor,
               region_valid: torch.Tensor, *, clean_cap: float) -> torch.Tensor:
    """`L_clean = mean_syn sum_r (1-m_r) * min(d_r, clean_cap)`, valid regions only.

    This is what keeps a partial attack's untouched regions from being dragged
    toward "spoof" (spec objective O4).
    """
    keep = is_synthetic.bool()
    if int(keep.sum()) == 0: return _zero(distance)
    applicable = (1.0 - attack_mask[keep].to(distance.dtype)) * region_valid[keep].to(distance.dtype)
    capped = torch.clamp(distance[keep], max=float(clean_cap))
    return (capped * applicable).sum(dim=1).mean()


def prompt_loss(region_embeddings: torch.Tensor, text_embeddings: torch.Tensor,
                recipe_index: torch.Tensor, attack_mask: torch.Tensor, is_synthetic: torch.Tensor,
                region_valid: torch.Tensor, *, temperature: float = DEFAULT_PROMPT_TEMPERATURE) -> torch.Tensor:
    """`L_prompt = InfoNCE(z_attack_region, text_embedding(recipe))`.

    Applies to synthetic samples on ATTACKED regions with valid visibility only. A
    real sample carries no recipe, so it contributes a masked finite zero.
    """
    keep = is_synthetic.bool() & (recipe_index >= 0)
    if int(keep.sum()) == 0: return _zero(region_embeddings)
    embeddings = region_embeddings[keep]
    applicable = (attack_mask[keep] * region_valid[keep].to(attack_mask.dtype)).bool()
    if int(applicable.sum()) == 0: return _zero(region_embeddings)
    normalized = torch.nn.functional.normalize(embeddings, dim=-1, eps=1e-6)
    text = torch.nn.functional.normalize(text_embeddings.to(normalized.dtype), dim=-1, eps=1e-6)
    logits = normalized @ text.transpose(0, 1) / float(temperature)     # [n,R,N_prompt]
    targets = recipe_index[keep].reshape(-1, 1).expand(-1, logits.shape[1])
    per_region = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
    return _mean_or_zero(per_region, applicable.reshape(-1))


def consistency_loss(p_global: torch.Tensor, s_region: torch.Tensor) -> torch.Tensor:
    """`L_cons = |p_global - sg(s_region)| + |s_region - sg(p_global)|`."""
    return ((p_global - s_region.detach()).abs() + (s_region - p_global.detach()).abs()).mean()


def risk_loss(per_sample_loss: torch.Tensor, domain_ids: torch.Tensor,
              family_ids: torch.Tensor) -> torch.Tensor:
    """`L_risk = Var(domain_risk) + Var(artifact_family_risk)`.

    A group risk is that group's mean classification loss. Variance over fewer than
    two groups is undefined, so the term is a finite zero there.
    """
    def group_variance(ids: torch.Tensor) -> torch.Tensor:
        valid = ids >= 0
        if int(valid.sum()) == 0: return _zero(per_sample_loss)
        present = torch.unique(ids[valid])
        if present.numel() < 2: return _zero(per_sample_loss)
        means = torch.stack([per_sample_loss[valid][ids[valid] == group].mean() for group in present])
        return means.var(unbiased=False)
    return group_variance(domain_ids) + group_variance(family_ids)


@dataclass(frozen=True)
class LossResult:
    total: torch.Tensor
    terms: dict[str, torch.Tensor]
    applicable: dict[str, int]
    weights: dict[str, float]

    def metrics(self) -> dict[str, float]:
        """Per-term values logged independently of the total, so a legitimately zero
        term is visible rather than inferred."""
        out = {name: float(value.detach()) for name, value in self.terms.items()}
        out["L_total"] = float(self.total.detach())
        out.update({f"applicable/{name}": float(count) for name, count in self.applicable.items()})
        return out


def loss_contract_identity(weights: dict[str, float], extra: dict[str, Any] | None = None) -> str:
    """Hash of the loss contract, bound into every checkpoint."""
    payload = {"schema_version": LOSS_SCHEMA_VERSION, "names": list(LOSS_NAMES),
               "weights": {key: float(value) for key, value in sorted(weights.items())},
               "extra": dict(sorted((extra or {}).items()))}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     default=str).encode("utf-8")).hexdigest()


def compute_losses(output: Any, batch: Any, manifold: Any, *, weights: dict[str, float] | None = None,
                   text_embeddings: torch.Tensor | None = None, clean_cap: float = DEFAULT_CLEAN_CAP,
                   mil_temperature: float = DEFAULT_MIL_TEMPERATURE,
                   prompt_temperature: float = DEFAULT_PROMPT_TEMPERATURE,
                   enabled: dict[str, bool] | None = None) -> LossResult:
    """Assemble `L_total` exactly as spec section 10.1 writes it."""
    lam = {**DEFAULT_WEIGHTS, **(weights or {})}
    on = {name: True for name in LOSS_NAMES}
    on.update(enabled or {})
    synthetic = batch.is_synthetic.bool()
    real = ~synthetic
    live = (batch.label == LIVE) & real
    valid = output.region_valid
    zero = _zero(output.global_logit)

    quality = batch.quality_weight
    if quality is None: quality = torch.zeros_like(output.p_global)
    # The mean q over the synthetic rows of this batch: `q` multiplies the whole
    # synthetic bracket in the declared total.
    q_bar = quality[synthetic].mean() if bool(synthetic.any()) else zero

    attack = batch.attack_region_mask
    if attack is None: attack = torch.zeros_like(valid, dtype=output.region_distances.dtype)

    terms: dict[str, torch.Tensor] = {}
    terms["L_cls_real"] = classification_loss(output.global_logit, batch.label, real) if on["L_cls_real"] else zero
    terms["L_cls_syn"] = classification_loss(output.global_logit, batch.label, synthetic) if on["L_cls_syn"] else zero
    terms["L_local"] = (weighted_local_loss(output.local_logits, batch.artifact_map, synthetic)
                        if on["L_local"] and batch.artifact_map is not None else zero)
    terms["L_MIL"] = mil_loss(output.local_logits, batch.label,
                              torch.ones_like(batch.label, dtype=torch.bool),
                              temperature=mil_temperature) if on["L_MIL"] else zero
    soft_min = manifold.soft_min(output.region_embeddings)
    terms["L_real"] = real_manifold_loss(soft_min, live, valid) if on["L_real"] else zero
    terms["L_out"] = (outlier_loss(output.region_distances, attack, synthetic, valid,
                                   margin=float(lam["margin_out"])) if on["L_out"] else zero)
    terms["L_clean"] = (clean_loss(output.region_distances, attack, synthetic, valid,
                                   clean_cap=float(clean_cap)) if on["L_clean"] else zero)
    terms["L_prompt"] = (prompt_loss(output.region_embeddings, text_embeddings, batch.recipe_index,
                                     attack, synthetic, valid, temperature=prompt_temperature)
                         if on["L_prompt"] and text_embeddings is not None and batch.recipe_index is not None
                         else zero)
    terms["L_cons"] = consistency_loss(output.p_global, output.s_region) if on["L_cons"] else zero
    per_sample = torch.nn.functional.binary_cross_entropy_with_logits(
        output.global_logit.reshape(-1), batch.label.to(output.global_logit.dtype), reduction="none")
    families = _family_ids(batch)
    terms["L_risk"] = risk_loss(per_sample, batch.dataset_id, families) if on["L_risk"] else zero

    total = (terms["L_cls_real"]
             + lam["lambda_syn"] * q_bar * (terms["L_cls_syn"]
                                            + lam["lambda_local"] * terms["L_local"]
                                            + lam["lambda_out"] * terms["L_out"])
             + lam["lambda_M"] * terms["L_real"]
             + lam["lambda_clean"] * terms["L_clean"]
             + lam["lambda_MIL"] * terms["L_MIL"]
             + lam["lambda_P"] * terms["L_prompt"]
             + lam["lambda_cons"] * terms["L_cons"]
             + lam["lambda_risk"] * terms["L_risk"])
    if not torch.isfinite(total): raise LossError("L_total is not finite")

    applicable = {
        "real": int(real.sum()), "synthetic": int(synthetic.sum()), "live": int(live.sum()),
        "valid_regions": int(valid.sum()),
        "attacked_regions": int((attack.bool() & valid).sum()),
        "clean_regions": int(((~attack.bool()) & valid & synthetic.unsqueeze(1)).sum())}
    return LossResult(total=total, terms=terms, applicable=applicable,
                      weights={**{k: float(v) for k, v in lam.items()},
                               "clean_cap": float(clean_cap), "q_bar": float(q_bar.detach())})


def _family_ids(batch: Any) -> torch.Tensor:
    """Artifact-family group ids; -1 where a sample has no family (real samples)."""
    names = list(getattr(batch, "artifact_family", ()) or [])
    if not names: return torch.full_like(batch.dataset_id, -1)
    ordered = sorted({name for name in names if name})
    lookup = {name: index for index, name in enumerate(ordered)}
    return torch.tensor([lookup.get(name, -1) for name in names], device=batch.dataset_id.device,
                        dtype=batch.dataset_id.dtype)
