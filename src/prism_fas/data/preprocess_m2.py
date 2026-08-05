from __future__ import annotations
import hashlib, re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Any
import cv2, numpy as np, yaml
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from prism_fas.utils.core import atomic_json_write, sha256_file, stable_json_hash, utc_timestamp

class M2Config(BaseModel):
    model_config=ConfigDict(extra="forbid")
    preprocessing_version:str; sampling_version:str; sampling_strategy:Literal["uniform"]; frames_per_video:int=Field(gt=0); start_exclusion_fraction:float=0; end_exclusion_fraction:float=0; minimum_valid_frames:int=1
    image_extensions:list[str]; video_decoder_priority:list[str]; scrfd_model_path:Path; scrfd_input_size:int; detection_threshold:float; detector:dict[str,Any]=Field(default_factory=dict); face_selection_policy:Literal["largest_valid_face"]; min_face_size:int; crop_padding:float; crop_output_size:int; output_image_format:Literal["jpg","png"]; jpeg_quality:int; materialize_frames:bool; overwrite_policy:str; resume_policy:str; failure_policy:str; hash_policy:str; worker_count:int; device:str
    @property
    def config_hash(self)->str:return stable_json_hash(self.model_dump(mode="json"))
def load_m2_config(path:Path)->M2Config:
    return M2Config.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
def uniform_indices(count:int, frames:int, start:float=0,end:float=0)->list[int]:
    if count<=0:return []
    lo=min(count-1,max(0,int(count*start))); hi=max(lo,count-1-int(count*end)); n=min(frames,hi-lo+1)
    return sorted(set(round(lo+(hi-lo)*i/max(1,n-1)) for i in range(n)))
def natural_paths(paths:list[Path])->list[Path]:
    return sorted(paths,key=lambda p:[int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)",p.name)])
def sample_id(dataset:str,video_id:str,index:int,adapter_version:str,sampling_version:str,preprocessing_version:str)->str:
    return hashlib.sha256(f"{dataset}|{video_id}|{index}|{adapter_version}|{sampling_version}|{preprocessing_version}".encode()).hexdigest()[:24]
class SourceFrameRecord(BaseModel):
    model_config=ConfigDict(extra="forbid"); sample_id:str; dataset:str; subject_id:str|None; video_id:str; official_split:str|None; label:Literal["live","spoof"]; media_type:str; source_relative_identifier:str; frame_index:int; timestamp_ms:float|None; selected_frame_reference:str; source_fingerprint:str; adapter_version:str; sampling_version:str; preprocessing_version:str; preprocessing_config_hash:str; status:str
class TargetFrameRecord(BaseModel):
    model_config=ConfigDict(extra="forbid"); sample_id:str; dataset:Literal["siw_mv2"]; video_id:str; media_type:str; frame_index:int; timestamp_ms:float|None; selected_frame_reference:str; source_fingerprint:str; adapter_version:str; sampling_version:str; preprocessing_version:str; preprocessing_config_hash:str; status:str
class Detection(BaseModel): bbox:tuple[float,float,float,float]; score:float; landmarks:list[tuple[float,float]]=[]
class DetectorInferenceError(RuntimeError):
    pass
class InvalidBoundingBoxError(ValueError):
    pass
class InvalidLandmarksError(ValueError):
    pass
class CropProcessingError(RuntimeError):
    pass
class FaceDetector(Protocol):
    name:str
    def detect(self,image:np.ndarray)->list[Detection]:...
class MockFaceDetector:
    name="mock"
    def __init__(self,detections:list[Detection]):self.detections=detections
    def detect(self,image:np.ndarray)->list[Detection]:return self.detections
class SCRFDDetector:
    name="scrfd"
    def __init__(self,model_path:Path,input_size:int=640,provider:str="CPUExecutionProvider"):
        import onnxruntime as ort
        self.model_path=model_path; self.model_sha256=sha256_file(model_path); self.provider=provider; self.input_size=input_size
        self.session=ort.InferenceSession(str(model_path),providers=[provider]); self.input_name=self.session.get_inputs()[0].name
    def detect(self,image:np.ndarray)->list[Detection]:
        h,w=image.shape[:2]; scale=min(self.input_size/w,self.input_size/h); resized=cv2.resize(image,(round(w*scale),round(h*scale))); canvas=np.zeros((self.input_size,self.input_size,3),np.uint8); canvas[:resized.shape[0],:resized.shape[1]]=resized
        blob=cv2.dnn.blobFromImage(canvas,1/128,(self.input_size,self.input_size),(127.5,127.5,127.5),swapRB=True)
        try: outputs=self.session.run(None,{self.input_name:blob})
        except Exception as exc: raise DetectorInferenceError('face detector inference failed') from exc
        found=[]
        for stride,score,boxes,kps in zip((8,16,32),outputs[:3],outputs[3:6],outputs[6:9]):
            grid=np.stack(np.mgrid[0:self.input_size//stride,0:self.input_size//stride][::-1],axis=-1).reshape(-1,2)*stride; centers=np.repeat(grid,2,axis=0)
            for i,value in enumerate(score.reshape(-1)):
                if value<.5: continue
                x,y=centers[i]; d=boxes[i]*stride; box=((x-d[0])/scale,(y-d[1])/scale,(x+d[2])/scale,(y+d[3])/scale); points=[((x+kps[i][j]*stride)/scale,(y+kps[i][j+1]*stride)/scale) for j in range(0,10,2)]; found.append(Detection(bbox=box,score=float(value),landmarks=points))
        return found
def select_largest(detections:list[Detection],threshold:float,min_size:int)->Detection|None:
    candidates=[d for d in detections if d.score>=threshold]
    def valid_bbox(d):
        x1,y1,x2,y2=d.bbox
        return all(np.isfinite(value) for value in d.bbox) and x2>x1 and y2>y1
    valid=[d for d in candidates if valid_bbox(d) and min(d.bbox[2]-d.bbox[0],d.bbox[3]-d.bbox[1])>=min_size]
    if not valid and any(not valid_bbox(d) for d in candidates):raise InvalidBoundingBoxError('detected face bounding box is invalid')
    return sorted(valid,key=lambda d:(-((d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1])),-d.score,d.bbox))[0] if valid else None
def crop_face(image:np.ndarray,detection:Detection,padding:float,size:int)->tuple[np.ndarray,tuple[int,int,int,int]]:
    h,w=image.shape[:2]; x1,y1,x2,y2=detection.bbox
    if not all(np.isfinite(value) for value in detection.bbox) or x2<=x1 or y2<=y1:raise InvalidBoundingBoxError('detected face bounding box is invalid')
    dx=(x2-x1)*padding; dy=(y2-y1)*padding; box=(max(0,int(x1-dx)),max(0,int(y1-dy)),min(w,int(x2+dx)),min(h,int(y2+dy)))
    if box[2]<=box[0] or box[3]<=box[1]:raise InvalidBoundingBoxError('detected face bounding box is invalid')
    crop=image[box[1]:box[3],box[0]:box[2]]
    if crop.size==0 or crop.ndim!=3 or crop.shape[2]!=3:raise CropProcessingError('face crop could not be created')
    try: output=cv2.resize(crop,(size,size))
    except cv2.error as exc:raise CropProcessingError('face crop could not be created') from exc
    if output.size==0 or output.shape!=(size,size,3):raise CropProcessingError('face crop could not be created')
    return output,box
def validate_landmarks(landmarks)->None:
    if len(landmarks)!=5:raise InvalidLandmarksError('detected face landmarks are invalid')
    try: values=[float(value) for point in landmarks for value in point]
    except (TypeError,ValueError):raise InvalidLandmarksError('detected face landmarks are invalid')
    if len(values)!=10 or not all(np.isfinite(value) for value in values):raise InvalidLandmarksError('detected face landmarks are invalid')
def media_type(path:Path)->str:return "video_file" if path.suffix.lower() in {".mov",".mp4",".avi"} else "single_image" if path.suffix.lower() in {".png",".jpg",".jpeg"} else "unsupported_media"
