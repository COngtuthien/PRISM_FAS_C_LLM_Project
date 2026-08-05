from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
import cv2, numpy as np
class UnreadableImageError(ValueError):
    pass
class VideoDecodeError(ValueError):
    pass
@dataclass(frozen=True)
class FrameResult:
    requested_frame_index:int; actual_frame_index:int; timestamp_ms:float|None; width:int; height:int; image:np.ndarray; backend:str; warnings:tuple[str,...]=()
class ImageSequenceReader:
    def __init__(self,anchor:Path):
        m=re.match(r'(?P<prefix>[bf]?s\d+v[A-Za-z0-9_]+)f\d+\.png$',anchor.name)
        if not m: raise ValueError(f"CASIA sequence anchor is invalid: {anchor}")
        self.paths=sorted(anchor.parent.glob(m['prefix']+'f*.png'),key=lambda p:int(re.search(r'f(\d+)\.png$',p.name)[1]))
        if not self.paths: raise FileNotFoundError(anchor)
    def frame_count(self)->int:return len(self.paths)
    def inspect_metadata(self)->dict:return {"frame_count":len(self.paths),"backend":"opencv-image-sequence"}
    def read_frame(self,index:int)->FrameResult:
        p=self.paths[index]; im=cv2.imread(str(p));
        if im is None: raise UnreadableImageError(f"unreadable_image: {p}")
        return FrameResult(index,index,None,im.shape[1],im.shape[0],im,"opencv-image-sequence")
    def close(self)->None:pass
class OpenCVVideoDecoder:
    def __init__(self,path:Path):self.path=path;self.cap=cv2.VideoCapture(str(path))
    def inspect_metadata(self)->dict:
        if not self.cap.isOpened():raise ValueError(f"metadata_read_failed: {self.path}")
        fps=self.cap.get(cv2.CAP_PROP_FPS); n=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)); return {"frame_count":n,"fps":fps,"width":int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),"height":int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),"backend":"opencv"}
    def frame_count(self)->int:return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    def read_frame(self,index:int)->FrameResult:
        if index<0 or index>=self.frame_count():raise IndexError(index)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,im=self.cap.read()
        if not ok or im is None: raise VideoDecodeError(f"decode_failed: {self.path} index={index}")
        return FrameResult(index,index,index*1000/self.cap.get(cv2.CAP_PROP_FPS) if self.cap.get(cv2.CAP_PROP_FPS)>0 else None,im.shape[1],im.shape[0],im,"opencv")
    def close(self)->None:self.cap.release()
