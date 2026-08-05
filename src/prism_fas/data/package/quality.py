from __future__ import annotations
import cv2, numpy as np
from .config import QUALITY_SCHEMA_VERSION

class QualityMetricError(ValueError):
    """A crop produced a non-finite or otherwise unusable quality metric."""
# Fixed order: the quality vector is positional in the prior NPZ.
QUALITY_NAMES=("blur_laplacian_variance","brightness_mean","brightness_std","contrast_michelson","saturation_mean","face_size_ratio")
def compute_quality(image:np.ndarray,*,bbox,frame_width:int,frame_height:int)->dict[str,float]:
    """Deterministic per-crop quality metrics.

    blur_laplacian_variance : variance of cv2.Laplacian(gray, CV_64F)
    brightness_mean/std     : grayscale mean / population std, normalized to [0,1]
    contrast_michelson      : (max-min)/(max+min+1e-8) on normalized grayscale
    saturation_mean         : mean HSV saturation, normalized to [0,1]
    face_size_ratio         : detector bbox area divided by original frame area
    """
    if image is None or image.ndim!=3 or image.shape[2]!=3: raise QualityMetricError("crop is not a 3-channel image")
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY); normalized=gray.astype(np.float64)/255.
    x1,y1,x2,y2=(float(v) for v in bbox); area=max(0.,(x2-x1))*max(0.,(y2-y1)); frame_area=float(frame_width)*float(frame_height)
    values={"blur_laplacian_variance":float(cv2.Laplacian(gray,cv2.CV_64F).var()),
            "brightness_mean":float(normalized.mean()),
            "brightness_std":float(normalized.std()),
            "contrast_michelson":float((normalized.max()-normalized.min())/(normalized.max()+normalized.min()+1e-8)),
            "saturation_mean":float(cv2.cvtColor(image,cv2.COLOR_BGR2HSV)[:,:,1].astype(np.float64).mean()/255.),
            "face_size_ratio":float(area/frame_area) if frame_area>0 else float("nan")}
    invalid=sorted(name for name,value in values.items() if not np.isfinite(value))
    if invalid: raise QualityMetricError(f"non-finite quality metrics: {invalid}")
    return values
def quality_vector(values:dict[str,float])->np.ndarray:
    missing=[name for name in QUALITY_NAMES if name not in values]
    if missing: raise QualityMetricError(f"missing quality metrics: {missing}")
    return np.asarray([values[name] for name in QUALITY_NAMES],dtype=np.float32)
def quality_schema_version()->str: return QUALITY_SCHEMA_VERSION
