from __future__ import annotations
import hashlib, io, os, tempfile
from pathlib import Path
import cv2, numpy as np
from .config import DEFERRED_PRIOR_STATUS, PRIOR_SCHEMA_VERSION
from .quality import QUALITY_NAMES, compute_quality, quality_vector

class PriorBuildError(RuntimeError):
    """A prior could not be produced or a stored prior is incompatible."""
PRIOR_ARRAYS={"bbox":((4,),np.float32),"landmarks":((5,2),np.float32),"crop_box":((4,),np.float32),
              "quality_vector":((len(QUALITY_NAMES),),np.float32),"detection_score":((),np.float32),
              "detected_face_count":((),np.int32),"frame_width":((),np.int32),"frame_height":((),np.int32),
              "crop_width":((),np.int32),"crop_height":((),np.int32)}
def _bbox(row): return np.asarray([row["bbox_x1"],row["bbox_y1"],row["bbox_x2"],row["bbox_y2"]],dtype=np.float32)
def _landmarks(row): return np.asarray([[row[f"landmark_{i}_x"],row[f"landmark_{i}_y"]] for i in range(5)],dtype=np.float32)
def _crop_box(row): return np.asarray([row["crop_x1"],row["crop_y1"],row["crop_x2"],row["crop_y2"]],dtype=np.float32)
def prior_payload(row:dict,image:np.ndarray)->tuple[dict[str,np.ndarray],dict[str,float]]:
    """Build the array payload for one sample. No paths, names or labels."""
    metrics=compute_quality(image,bbox=(row["bbox_x1"],row["bbox_y1"],row["bbox_x2"],row["bbox_y2"]),frame_width=row["frame_width"],frame_height=row["frame_height"])
    payload={"bbox":_bbox(row),"landmarks":_landmarks(row),"crop_box":_crop_box(row),
             "quality_vector":quality_vector(metrics),"quality_names":np.asarray(QUALITY_NAMES,dtype="U32"),
             "detection_score":np.float32(row["detection_score"]),"detected_face_count":np.int32(row["detected_face_count"]),
             "frame_width":np.int32(row["frame_width"]),"frame_height":np.int32(row["frame_height"]),
             "crop_width":np.int32(row["crop_width"]),"crop_height":np.int32(row["crop_height"])}
    return payload,metrics
def serialize_prior(payload:dict[str,np.ndarray])->bytes:
    """Deterministic uncompressed NPZ bytes with a stable field order."""
    buffer=io.BytesIO(); np.savez(buffer,**{name:payload[name] for name in sorted(payload)}); return buffer.getvalue()
def write_prior_atomic(path:Path,data:bytes)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix=path.name+".",suffix=".tmp",dir=path.parent); os.close(fd)
    try:
        Path(temporary).write_bytes(data)
        for attempt in range(12):
            try: os.replace(temporary,path); break
            except PermissionError:
                if attempt==11: raise
                import gc, time; gc.collect(); time.sleep(.15)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
def load_prior(path:Path)->dict[str,np.ndarray]:
    with np.load(path,allow_pickle=False) as handle: return {name:handle[name] for name in handle.files}
def validate_prior_arrays(arrays:dict[str,np.ndarray])->None:
    for name,(shape,dtype) in PRIOR_ARRAYS.items():
        if name not in arrays: raise PriorBuildError(f"prior missing array: {name}")
        value=arrays[name]
        if tuple(value.shape)!=shape: raise PriorBuildError(f"prior {name} shape {tuple(value.shape)} != {shape}")
        if value.dtype!=np.dtype(dtype): raise PriorBuildError(f"prior {name} dtype {value.dtype} != {np.dtype(dtype)}")
    if list(arrays["quality_names"])!=list(QUALITY_NAMES): raise PriorBuildError("prior quality_names mismatch")
    if not np.isfinite(arrays["quality_vector"]).all(): raise PriorBuildError("prior quality_vector is not finite")
def prior_status()->dict[str,str]: return dict(DEFERRED_PRIOR_STATUS)
def prior_schema_version()->str: return PRIOR_SCHEMA_VERSION
def sha256_bytes(data:bytes)->str: return hashlib.sha256(data).hexdigest()
def read_crop(path:Path)->np.ndarray:
    image=cv2.imread(str(path))
    if image is None: raise PriorBuildError("crop could not be decoded")
    return image
