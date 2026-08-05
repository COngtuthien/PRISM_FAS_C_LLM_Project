from __future__ import annotations
import numpy as np
from .contracts import CanonicalSourceSample, CanonicalTargetSample, FORBIDDEN_TARGET_FIELDS, TargetIsolationViolation

def _stack(samples,attribute:str,dtype):
    import torch
    return torch.from_numpy(np.stack([getattr(sample.geometry,attribute) for sample in samples]).astype(dtype))
def collate_source_batch(samples:list[CanonicalSourceSample])->dict:
    """Collate labelled source samples. Rejects any target sample."""
    import torch
    if not samples: raise ValueError("cannot collate an empty batch")
    foreign=[sample.sample_id for sample in samples if not isinstance(sample,CanonicalSourceSample)]
    if foreign: raise TargetIsolationViolation(f"non-source samples in a source batch: {foreign[:5]}")
    identity=np.zeros((len(samples),512),dtype=np.float32); available=np.zeros(len(samples),dtype=bool)
    for position,sample in enumerate(samples):
        if sample.identity_available:
            identity[position]=sample.identity_embedding; available[position]=True
    return {"image":torch.from_numpy(np.stack([sample.image for sample in samples])),
            "target":torch.tensor([sample.class_target for sample in samples],dtype=torch.int64),
            "sample_id":[sample.sample_id for sample in samples],
            "dataset":[sample.dataset for sample in samples],
            "domain":[sample.dataset for sample in samples],
            "source_record_id":[sample.source_record_id for sample in samples],
            "label":[sample.label for sample in samples],
            "bbox":_stack(samples,"bbox",np.float32),"landmarks":_stack(samples,"landmarks",np.float32),
            "crop_box":_stack(samples,"crop_box",np.float32),
            "parsing":_stack(samples,"parsing_labels",np.int64),
            "pose":_stack(samples,"pose_ypr",np.float32),"visibility":_stack(samples,"visibility",np.float32),
            "quality":_stack(samples,"quality_vector",np.float32),
            "detection_score":torch.tensor([s.geometry.detection_score for s in samples],dtype=torch.float32),
            # Zero rows are placeholders only; they are meaningless unless the
            # matching identity_available flag is true.
            "identity_embedding":torch.from_numpy(identity),
            "identity_available":torch.from_numpy(available),
            "crop_sha256":[sample.crop_sha256 for sample in samples],
            "prior_sha256":[sample.prior_sha256 for sample in samples]}
def collate_target_batch(samples:list[CanonicalTargetSample])->dict:
    """Collate inference-only target samples. Never emits labels or identity."""
    import torch
    if not samples: raise ValueError("cannot collate an empty batch")
    foreign=[getattr(sample,"sample_id","?") for sample in samples if not isinstance(sample,CanonicalTargetSample)]
    if foreign: raise TargetIsolationViolation(f"non-target samples in a target batch: {foreign[:5]}")
    for sample in samples:
        leaked=sorted(FORBIDDEN_TARGET_FIELDS & set(vars(sample)))
        if leaked: raise TargetIsolationViolation(f"target sample exposes forbidden fields: {leaked}")
    batch={"image":torch.from_numpy(np.stack([sample.image for sample in samples])),
           "sample_id":[sample.sample_id for sample in samples],
           "dataset":[sample.dataset for sample in samples],
           "source_record_id":[sample.source_record_id for sample in samples],
           "bbox":_stack(samples,"bbox",np.float32),"landmarks":_stack(samples,"landmarks",np.float32),
           "crop_box":_stack(samples,"crop_box",np.float32),
           "parsing":_stack(samples,"parsing_labels",np.int64),
           "pose":_stack(samples,"pose_ypr",np.float32),"visibility":_stack(samples,"visibility",np.float32),
           "quality":_stack(samples,"quality_vector",np.float32),
           "detection_score":torch.tensor([s.geometry.detection_score for s in samples],dtype=torch.float32),
           "identity_available":torch.zeros(len(samples),dtype=torch.bool),
           "crop_sha256":[sample.crop_sha256 for sample in samples],
           "prior_sha256":[sample.prior_sha256 for sample in samples]}
    leaked=sorted(FORBIDDEN_TARGET_FIELDS & set(batch))
    if leaked: raise TargetIsolationViolation(f"target batch exposes forbidden keys: {leaked}")
    return batch
