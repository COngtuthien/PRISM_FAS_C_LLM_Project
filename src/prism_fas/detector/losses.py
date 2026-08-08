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


def image_level_outlier_loss(distance: torch.Tensor, is_synthetic: torch.Tensor,
                             region_valid: torch.Tensor, *, margin: float) -> torch.Tensor:
    """The Table 60 `image-level` outlier control: `mean_syn max(0, margin - d_image)`.

    This is NOT `outlier_loss` under another name, and the difference is the whole
    point of hypothesis H3. The mask-aware term reads `m_r` and pushes ONLY the
    regions the recipe actually attacked, leaving the clean regions to `L_clean`.
    This term never reads `m_r` at all: it collapses the per-site distances to one
    image-level value and pushes the WHOLE sample away from the real manifold, so it
    cannot distinguish an attacked region from an untouched one. That is exactly the
    question "Vùng sạch có được bảo toàn?" asks, and it is why `L_clean` is
    structurally inactive under this variant rather than merely down-weighted.

    The image-level distance is the mean over valid sites, so a variant with one
    global center and a variant with nine regional sites aggregate the same way.
    """
    keep = is_synthetic.bool()
    if int(keep.sum()) == 0: return _zero(distance)
    valid = region_valid[keep].to(distance.dtype)
    count = valid.sum(dim=1).clamp_min(1.0)
    image_distance = (distance[keep] * valid).sum(dim=1) / count
    return torch.clamp(float(margin) - image_distance, min=0.0).mean()


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
    # Which terms the variant and the stage actually computed. A term reported as
    # 0.0 with `active=False` was never evaluated; one with `active=True` was
    # measured and is legitimately zero. Without this the two are indistinguishable
    # in `metrics.jsonl`.
    active: dict[str, bool] = field(default_factory=lambda: {name: True for name in LOSS_NAMES})

    def metrics(self) -> dict[str, float]:
        """Per-term values logged independently of the total, so a legitimately zero
        term is visible rather than inferred."""
        out = {name: float(value.detach()) for name, value in self.terms.items()}
        out["L_total"] = float(self.total.detach())
        out.update({f"applicable/{name}": float(count) for name, count in self.applicable.items()})
        out.update({f"active/{name}": float(bool(value)) for name, value in self.active.items()})
        return out


def loss_contract_identity(weights: dict[str, float], extra: dict[str, Any] | None = None) -> str:
    """Hash of the loss contract, bound into every checkpoint.

    `extra` carries the variant's loss-relevant delta against the B08 reference; it
    is EMPTY for the reference itself, so the M9 reference checkpoints keep the
    contract hash they were written with while every ablation that changes the loss
    graph gets a different one.
    """
    payload = {"schema_version": LOSS_SCHEMA_VERSION, "names": list(LOSS_NAMES),
               "weights": {key: float(value) for key, value in sorted(weights.items())},
               "extra": dict(sorted((extra or {}).items()))}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     default=str).encode("utf-8")).hexdigest()


def loss_graph_delta(variant: Any) -> dict[str, Any]:
    """The loss-graph switches on which `variant` differs from the B08 reference."""
    from .variant import ResolvedExperimentVariant
    reference = ResolvedExperimentVariant.reference()
    body: dict[str, Any] = {}
    if variant.outlier_loss != reference.outlier_loss: body["outlier_loss"] = variant.outlier_loss
    if variant.quality_weighting != reference.quality_weighting:
        body["quality_weighting"] = variant.quality_weighting
    active, reference_active = variant.active_loss_terms(), reference.active_loss_terms()
    changed = {name: value for name, value in active.items() if reference_active[name] != value}
    if changed: body["active_loss_terms"] = changed
    return body


def compute_losses(output: Any, batch: Any, manifold: Any, *, weights: dict[str, float] | None = None,
                   text_embeddings: torch.Tensor | None = None, clean_cap: float = DEFAULT_CLEAN_CAP,
                   mil_temperature: float = DEFAULT_MIL_TEMPERATURE,
                   prompt_temperature: float = DEFAULT_PROMPT_TEMPERATURE,
                   enabled: dict[str, bool] | None = None,
                   variant: Any = None) -> LossResult:
    """Assemble `L_total` exactly as spec section 10.1 writes it, for the variant's
    ACTIVE capabilities.

    `enabled` narrows the graph to what a stage may compute; `variant` narrows it to
    what the configuration HAS. An inactive term is never computed — it is not
    evaluated and then multiplied by zero, because a constant that reaches the log
    reads as a measurement. Its value is reported as an exact structural zero and its
    `applicable` count as 0, so the two are distinguishable.
    """
    from .variant import ResolvedExperimentVariant
    if variant is None: variant = ResolvedExperimentVariant.reference()
    lam = {**DEFAULT_WEIGHTS, **(weights or {})}
    on = dict(variant.active_loss_terms())
    on.update(enabled or {})
    synthetic = batch.is_synthetic.bool()
    real = ~synthetic
    live = (batch.label == LIVE) & real
    valid = output.region_valid                     # [B,M] manifold-site validity, or None
    zero = _zero(output.global_logit)

    # `q` weights ONLY the synthetic bracket, and only when the variant declares
    # soft quality weighting. `hard_gate_only` reuses the SAME accepted M8 samples
    # and gives every one of them weight 1.0 — it does not rebuild the gate, does
    # not alter acceptance and never turns `q` into a label. `off` has no synthetic
    # bracket at all, so the factor is irrelevant and fixed at 1.0.
    quality = batch.quality_weight
    if quality is None: quality = torch.zeros_like(output.p_global)
    if variant.uses_quality_weight:
        q_bar = quality[synthetic].mean() if bool(synthetic.any()) else zero
    else:
        q_bar = zero + 1.0

    attack = batch.attack_region_mask               # [B,R] over the NINE semantic regions
    terms: dict[str, torch.Tensor] = {}
    terms["L_cls_real"] = classification_loss(output.global_logit, batch.label, real) if on["L_cls_real"] else zero
    terms["L_cls_syn"] = classification_loss(output.global_logit, batch.label, synthetic) if on["L_cls_syn"] else zero
    terms["L_local"] = (weighted_local_loss(output.local_logits, batch.artifact_map, synthetic)
                        if on["L_local"] and output.local_logits is not None
                        and batch.artifact_map is not None else zero)
    terms["L_MIL"] = (mil_loss(output.local_logits, batch.label,
                               torch.ones_like(batch.label, dtype=torch.bool),
                               temperature=mil_temperature)
                      if on["L_MIL"] and output.local_logits is not None else zero)
    # For the regional manifold the sites ARE the region embeddings; the detector
    # exposes the pooled form by name for a single global center. Falling back keeps
    # a hand-built `ModelOutput` (the toy tests) working without a second code path.
    manifold_embeddings = (output.aux or {}).get("manifold_embeddings", output.region_embeddings)
    # `L_prompt` and the attacked/clean counts are defined over the NINE semantic
    # regions, which is `region_valid` exactly when the manifold is regional.
    declared_visibility = getattr(output, "region_visibility_valid", None)
    region_visibility = declared_visibility if declared_visibility is not None else output.region_valid
    terms["L_real"] = (real_manifold_loss(manifold.soft_min(manifold_embeddings), live, valid)
                       if on["L_real"] and manifold is not None and manifold_embeddings is not None else zero)
    # mask-aware reads `m_r`; image-level deliberately does not. Two different
    # graphs, selected by the declared flag, never one renamed as the other.
    if on["L_out"] and output.region_distances is not None:
        if variant.outlier_loss == "mask_aware":
            terms["L_out"] = outlier_loss(output.region_distances, attack, synthetic, valid,
                                          margin=float(lam["margin_out"]))
        else:
            terms["L_out"] = image_level_outlier_loss(output.region_distances, synthetic, valid,
                                                      margin=float(lam["margin_out"]))
    else:
        terms["L_out"] = zero
    terms["L_clean"] = (clean_loss(output.region_distances, attack, synthetic, valid, clean_cap=float(clean_cap))
                        if on["L_clean"] and output.region_distances is not None and attack is not None else zero)
    # `z_attack_region` is the PromptHead's projection of `z_r` into the frozen text
    # space (Table 32); the raw `z_r` lives in the region dim and cannot be dotted
    # against a text embedding at all. The detector exposes the projection by name.
    prompt_embeddings = (output.aux or {}).get("prompt_region_embeddings")
    terms["L_prompt"] = (prompt_loss(prompt_embeddings, text_embeddings, batch.recipe_index,
                                     attack, synthetic, region_visibility,
                                     temperature=prompt_temperature)
                         if on["L_prompt"] and prompt_embeddings is not None and text_embeddings is not None
                         and batch.recipe_index is not None and attack is not None else zero)
    terms["L_cons"] = (consistency_loss(output.p_global, output.s_region)
                       if on["L_cons"] and output.s_region is not None else zero)
    per_sample = torch.nn.functional.binary_cross_entropy_with_logits(
        output.global_logit.reshape(-1), batch.label.to(output.global_logit.dtype), reduction="none")
    # A variant that does not consume recipe identity has no artifact-family group,
    # so the family half of `L_risk` is absent rather than grouped by an empty string.
    families = (_family_ids(batch) if variant.consumes_recipe_identity
                else torch.full_like(batch.dataset_id, -1))
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

    attacked = attack.bool() if attack is not None else None
    applicable = {
        "real": int(real.sum()), "synthetic": int(synthetic.sum()), "live": int(live.sum()),
        "valid_regions": int(valid.sum()) if valid is not None else 0,
        "attacked_regions": int((attacked & region_visibility).sum())
        if attacked is not None and region_visibility is not None else 0,
        "clean_regions": int(((~attacked) & region_visibility & synthetic.unsqueeze(1)).sum())
        if attacked is not None and region_visibility is not None else 0}
    return LossResult(total=total, terms=terms, applicable=applicable,
                      weights={**{k: float(v) for k, v in lam.items()},
                               "clean_cap": float(clean_cap), "q_bar": float(q_bar.detach())},
                      active=dict(on))


def _family_ids(batch: Any) -> torch.Tensor:
    """Artifact-family group ids; -1 where a sample has no family (real samples)."""
    names = list(getattr(batch, "artifact_family", ()) or [])
    if not names: return torch.full_like(batch.dataset_id, -1)
    ordered = sorted({name for name in names if name})
    lookup = {name: index for index, name in enumerate(ordered)}
    return torch.tensor([lookup.get(name, -1) for name in names], device=batch.dataset_id.device,
                        dtype=batch.dataset_id.dtype)
