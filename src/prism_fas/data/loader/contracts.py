from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np

class TargetIsolationViolation(PermissionError):
    """A target sample or batch was asked for source-only training data."""
class SampleContractError(ValueError):
    """A loaded sample does not satisfy the canonical contract."""
# Fields that must never appear on a target sample or target batch.
FORBIDDEN_TARGET_FIELDS=frozenset({"label","label_live_spoof","target","class_target","subject_id","official_split",
    "identity_embedding","attack_type","taxonomy","session_id","source_path","crop_relative_path_absolute"})
@dataclass(frozen=True)
class CanonicalGeometry:
    """Technical geometry/quality features shared by both roles."""
    bbox:np.ndarray; landmarks:np.ndarray; crop_box:np.ndarray
    parsing_labels:np.ndarray; pose_ypr:np.ndarray; visibility:np.ndarray
    quality_vector:np.ndarray; quality_names:tuple[str,...]
    detection_score:float; detected_face_count:int
    def validate(self)->None:
        checks=(("bbox",(4,),np.float32),("landmarks",(5,2),np.float32),("crop_box",(4,),np.float32),
                ("parsing_labels",(224,224),np.uint8),("pose_ypr",(3,),np.float32),
                ("visibility",(9,),np.float16),("quality_vector",(6,),np.float32))
        for name,shape,dtype in checks:
            value=getattr(self,name)
            if tuple(value.shape)!=shape: raise SampleContractError(f"{name} shape {tuple(value.shape)} != {shape}")
            if value.dtype!=np.dtype(dtype): raise SampleContractError(f"{name} dtype {value.dtype} != {np.dtype(dtype)}")
        for name in ("bbox","landmarks","crop_box","pose_ypr","quality_vector"):
            if not np.isfinite(getattr(self,name)).all(): raise SampleContractError(f"{name} contains non-finite values")
        visibility=self.visibility.astype(np.float32)
        if not (np.isfinite(visibility).all() and (visibility>=0).all() and (visibility<=1).all()):
            raise SampleContractError("visibility outside [0,1]")
@dataclass(frozen=True)
class CanonicalSourceSample:
    """A labelled source sample. Only source splits may produce this."""
    sample_id:str; dataset:str; project_split:str; source_record_id:str
    requested_frame_index:int; actual_frame_index:int
    label:str; class_target:int
    image:np.ndarray; geometry:CanonicalGeometry
    identity_embedding:np.ndarray|None; identity_available:bool
    crop_sha256:str; prior_sha256:str; source_media_type:str
    def validate(self)->None:
        _validate_image(self.image); self.geometry.validate()
        if not self.label or self.class_target is None: raise SampleContractError("source sample must carry a label and class target")
        if self.project_split.startswith("target"): raise TargetIsolationViolation("target split cannot produce a source sample")
        if self.identity_available and (self.identity_embedding is None or self.identity_embedding.shape!=(512,)):
            raise SampleContractError("identity_available implies a 512-d embedding")
        if not self.identity_available and self.identity_embedding is not None:
            raise SampleContractError("identity_embedding present while identity_available is false")
@dataclass(frozen=True)
class CanonicalTargetSample:
    """An inference-only target sample: technical features, never a label."""
    sample_id:str; dataset:str; project_split:str; source_record_id:str
    image:np.ndarray; geometry:CanonicalGeometry
    crop_sha256:str; prior_sha256:str; source_media_type:str
    identity_available:bool=field(default=False,init=False)
    def validate(self)->None:
        _validate_image(self.image); self.geometry.validate()
        if self.project_split!="target_test": raise SampleContractError("target sample must belong to target_test")
        leaked=sorted(FORBIDDEN_TARGET_FIELDS & set(vars(self)))
        if leaked: raise TargetIsolationViolation(f"target sample exposes forbidden fields: {leaked}")
        if self.identity_available: raise TargetIsolationViolation("target sample cannot claim identity availability")
def _validate_image(image:np.ndarray)->None:
    if image.shape!=(3,224,224): raise SampleContractError(f"image shape {image.shape} != (3,224,224)")
    if image.dtype!=np.float32: raise SampleContractError(f"image dtype {image.dtype} != float32")
    if not np.isfinite(image).all() or image.min()<0. or image.max()>1.:
        raise SampleContractError("image values outside [0,1]")
def source_fields()->tuple[str,...]:
    return ("sample_id","dataset","project_split","source_record_id","requested_frame_index","actual_frame_index",
            "label","class_target","image","geometry","identity_embedding","identity_available",
            "crop_sha256","prior_sha256","source_media_type")
def target_fields()->tuple[str,...]:
    return ("sample_id","dataset","project_split","source_record_id","image","geometry",
            "crop_sha256","prior_sha256","source_media_type","identity_available")
