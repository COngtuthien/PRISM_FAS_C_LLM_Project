from __future__ import annotations
import io
from pathlib import Path
import numpy as np
from .config import ImagePolicy
from .contracts import CanonicalGeometry, SampleContractError

def decode_image(data:bytes,policy:ImagePolicy)->np.ndarray:
    """Deterministically decode frozen package JPEG bytes to CHW float32 RGB in [0,1].

    No augmentation, no backbone-specific normalization: those belong downstream.
    """
    if policy.decoder=="opencv":
        import cv2
        array=cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_COLOR)
        if array is None: raise SampleContractError("package image could not be decoded")
        rgb=cv2.cvtColor(array,cv2.COLOR_BGR2RGB)
    else:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as handle: rgb=np.asarray(handle.convert("RGB"))
    height,width=policy.size
    if rgb.shape[:2]!=(height,width): raise SampleContractError(f"package image is {rgb.shape[:2]}, expected {(height,width)}")
    image=rgb.astype(np.float32)/255.
    return np.ascontiguousarray(image.transpose(2,0,1)) if policy.channels_first else image
def read_image(path:Path,policy:ImagePolicy)->np.ndarray:
    return decode_image(Path(path).read_bytes(),policy)
def geometry_from_arrays(arrays:dict[str,np.ndarray])->CanonicalGeometry:
    """Build the canonical geometry contract from a loaded M3B prior NPZ."""
    missing=[name for name in ("bbox","landmarks","crop_box","parsing_labels","pose_ypr","visibility","quality_vector","quality_names") if name not in arrays]
    if missing: raise SampleContractError(f"prior is missing arrays: {missing}")
    geometry=CanonicalGeometry(bbox=arrays["bbox"],landmarks=arrays["landmarks"],crop_box=arrays["crop_box"],
        parsing_labels=arrays["parsing_labels"],pose_ypr=arrays["pose_ypr"],visibility=arrays["visibility"],
        quality_vector=arrays["quality_vector"],quality_names=tuple(str(name) for name in arrays["quality_names"]),
        detection_score=float(arrays["detection_score"]),detected_face_count=int(arrays["detected_face_count"]))
    geometry.validate(); return geometry
def to_tensors(geometry:CanonicalGeometry)->dict:
    """Torch-facing dtypes: parsing becomes int64, everything else float32."""
    import torch
    return {"bbox":torch.from_numpy(geometry.bbox.astype(np.float32)),
            "landmarks":torch.from_numpy(geometry.landmarks.astype(np.float32)),
            "crop_box":torch.from_numpy(geometry.crop_box.astype(np.float32)),
            "parsing":torch.from_numpy(geometry.parsing_labels.astype(np.int64)),
            "pose":torch.from_numpy(geometry.pose_ypr.astype(np.float32)),
            "visibility":torch.from_numpy(geometry.visibility.astype(np.float32)),
            "quality":torch.from_numpy(geometry.quality_vector.astype(np.float32))}
