"""M9 typed contracts: regions, batch and model output.

Every shape and name here traces to `docs/spec_snapshot.md`; see
`docs/M9_DETECTOR_CONTRACT.md` for the mapping. Nothing in this package may import
modal (spec section 12.1), and no field may depend on target metadata.
"""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, field
from typing import Any
import torch

M9_SCHEMA_VERSION = "m9-detector-v1"
# Spec section 9.1 region set. Identical to the M3B visibility order
# (`data.package.model_priors.VISIBILITY_REGIONS`) and the M7 mask order
# (`synthesis.masks.REGION_ORDER`), so the three never drift.
REGION_ORDER = ("left_eye", "right_eye", "nose", "mouth", "forehead",
                "left_cheek", "right_cheek", "face_boundary", "context")
REGION_COUNT = len(REGION_ORDER)
# M5 label convention, unchanged: live=0, spoof=1.
LIVE, SPOOF = 0, 1
SAMPLE_KINDS = ("real_live", "real_spoof", "synthetic_spoof")


class DetectorContractError(ValueError):
    """A tensor or batch does not satisfy the declared M9 contract."""


class TargetIsolationViolation(PermissionError):
    """M9 was asked for target data it must never open."""


def _check(tensor: torch.Tensor, name: str, shape: tuple[int | None, ...]) -> None:
    if tensor.dim() != len(shape):
        raise DetectorContractError(f"{name} has {tensor.dim()} dims, expected {len(shape)}")
    for axis, expected in enumerate(shape):
        if expected is not None and tensor.shape[axis] != expected:
            raise DetectorContractError(f"{name} axis {axis} is {tensor.shape[axis]}, expected {expected}")
    if not torch.isfinite(tensor).all():
        raise DetectorContractError(f"{name} contains non-finite values")


@dataclass(frozen=True)
class DetectorBatch:
    """One training batch. Spec section 14.1 (Table 52) plus the M9 fields the
    declared losses actually need.

    `quality_weight` is the M8 `q`. It is a WEIGHT, never a target: nothing in this
    package may derive a label from it (spec section 8, Table 30).
    """
    image: torch.Tensor                      # [B,3,224,224] float in [0,1]
    label: torch.Tensor                      # [B] long, live=0 spoof=1
    dataset_id: torch.Tensor                 # [B] long, index into `datasets`
    is_synthetic: torch.Tensor               # [B] bool
    region_priors: torch.Tensor              # [B,R,Ht,Wt] float soft masks in [0,1]
    visibility: torch.Tensor                 # [B,R] float in [0,1]
    sample_ids: tuple[str, ...]
    datasets: tuple[str, ...]
    artifact_map: torch.Tensor | None = None       # [B,1,224,224], synthetic only
    attack_region_mask: torch.Tensor | None = None # [B,R] float 0/1, synthetic only
    quality_weight: torch.Tensor | None = None     # [B] float in [0,1], synthetic only
    recipe_index: torch.Tensor | None = None       # [B] long, -1 when not applicable
    artifact_family: tuple[str, ...] = ()

    def validate(self) -> "DetectorBatch":
        batch = self.image.shape[0]
        _check(self.image, "image", (batch, 3, 224, 224))
        if self.label.shape != (batch,): raise DetectorContractError("label must be [B]")
        if set(self.label.unique().tolist()) - {LIVE, SPOOF}:
            raise DetectorContractError("label must be 0 (live) or 1 (spoof)")
        _check(self.region_priors, "region_priors", (batch, REGION_COUNT, None, None))
        _check(self.visibility, "visibility", (batch, REGION_COUNT))
        if self.is_synthetic.shape != (batch,): raise DetectorContractError("is_synthetic must be [B]")
        if len(self.sample_ids) != batch: raise DetectorContractError("sample_ids must align with the batch")
        synthetic = self.is_synthetic.bool()
        if synthetic.any():
            if self.quality_weight is None: raise DetectorContractError("synthetic rows require quality_weight")
            _check(self.quality_weight, "quality_weight", (batch,))
            if float(self.quality_weight[synthetic].min()) < 0.0 or float(self.quality_weight[synthetic].max()) > 1.0:
                raise DetectorContractError("quality_weight must lie in [0,1]")
            if self.attack_region_mask is not None:
                _check(self.attack_region_mask, "attack_region_mask", (batch, REGION_COUNT))
            # A synthetic sample is a declared synthetic spoof; the label never comes from q.
            if int(self.label[synthetic].min()) != SPOOF:
                raise DetectorContractError("every synthetic sample must carry the declared spoof label")
        if (~synthetic).any() and self.attack_region_mask is not None:
            if float(self.attack_region_mask[~synthetic].abs().sum()) != 0.0:
                raise DetectorContractError("a real sample cannot carry an attack region mask")
        return self

    @property
    def batch_size(self) -> int: return int(self.image.shape[0])

    def kinds(self) -> list[str]:
        return ["synthetic_spoof" if bool(s) else ("real_spoof" if int(y) == SPOOF else "real_live")
                for s, y in zip(self.is_synthetic.tolist(), self.label.tolist())]

    def to(self, device: str) -> "DetectorBatch":
        move = lambda value: None if value is None else value.to(device)
        return DetectorBatch(
            image=self.image.to(device), label=self.label.to(device), dataset_id=self.dataset_id.to(device),
            is_synthetic=self.is_synthetic.to(device), region_priors=self.region_priors.to(device),
            visibility=self.visibility.to(device), sample_ids=self.sample_ids, datasets=self.datasets,
            artifact_map=move(self.artifact_map), attack_region_mask=move(self.attack_region_mask),
            quality_weight=move(self.quality_weight), recipe_index=move(self.recipe_index),
            artifact_family=self.artifact_family)


@dataclass(frozen=True)
class ModelOutput:
    """Spec section 14.2 (Table 53), plus the named fusion components section 9.4
    needs and the visibility mask every regional loss needs.

    Typed on purpose: the spec's contract is a named structure, and M10 ablations
    must be able to read each evidence term without re-deriving it.
    """
    global_logit: torch.Tensor               # [B,1]
    local_logits: torch.Tensor               # [B,P]
    region_embeddings: torch.Tensor          # [B,R,D]
    region_distances: torch.Tensor           # [B,R]
    region_valid: torch.Tensor               # [B,R] bool, from visibility
    prompt_logits: torch.Tensor | None       # [B,N_prompt] or None
    p_global: torch.Tensor                   # [B] sigmoid(global_logit)
    s_region: torch.Tensor                   # [B] TopKMean(normalize(d_r), k=2)
    p_prompt_spoof: torch.Tensor             # [B] bounded prompt evidence in [0,1]
    s_final: torch.Tensor                    # [B] fused score
    confidence_features: dict[str, torch.Tensor] = field(default_factory=dict)
    aux: dict[str, torch.Tensor] = field(default_factory=dict)

    def validate(self) -> "ModelOutput":
        batch = self.global_logit.shape[0]
        _check(self.global_logit, "global_logit", (batch, 1))
        _check(self.local_logits, "local_logits", (batch, None))
        _check(self.region_embeddings, "region_embeddings", (batch, REGION_COUNT, None))
        _check(self.region_distances, "region_distances", (batch, REGION_COUNT))
        for name in ("p_global", "s_region", "p_prompt_spoof", "s_final"):
            value = getattr(self, name)
            _check(value, name, (batch,))
            if float(value.min()) < -1e-6 or float(value.max()) > 1.0 + 1e-6:
                raise DetectorContractError(f"{name} must lie in [0,1]")
        if self.region_valid.shape != (batch, REGION_COUNT):
            raise DetectorContractError("region_valid must be [B,R]")
        if float(self.region_distances.min()) < 0.0:
            raise DetectorContractError("a squared Mahalanobis distance cannot be negative")
        return self


def architecture_identity(payload: dict[str, Any]) -> str:
    """Hash of the declared architecture contract. Bound into every checkpoint so a
    silent architecture change cannot be resumed into."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     default=str).encode("utf-8")).hexdigest()
