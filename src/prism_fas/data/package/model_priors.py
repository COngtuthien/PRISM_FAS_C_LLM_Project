from __future__ import annotations
import hashlib, importlib.util, os, sys, urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
import cv2, numpy as np, yaml
from prism_fas.utils.core import sha256_file

VISIBILITY_REGIONS=("left_eye","right_eye","nose","mouth","forehead","left_cheek","right_cheek","face_boundary","context")
class ModelPriorError(RuntimeError):
    """A model-dependent prior could not be produced."""
class ModelWeightError(ModelPriorError):
    """A pinned model weight is missing or its SHA-256 does not match."""
def load_model_config(path:Path)->dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
def resolve_weight(config:dict,section:str,weight_root:Path)->Path:
    """Resolve and verify a pinned weight file against its recorded SHA-256."""
    spec=config[section]; path=Path(weight_root)/spec["weight_relative_path"]
    if not path.is_file(): raise ModelWeightError(f"{section} weight is missing: {spec['weight_relative_path']} (repo {spec['repo_id']} rev {spec['revision']})")
    digest=sha256_file(path)
    if digest!=spec["weight_sha256"]: raise ModelWeightError(f"{section} weight SHA-256 mismatch for {spec['weight_relative_path']}")
    return path
def _load_module(name:str,path:Path):
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module; spec.loader.exec_module(module); return module
def fetch_backend_code(cache_root:Path,files:dict[str,str])->Path:
    """Download pinned third-party model code into the ignored model cache.

    Vendoring the upstream code into the repository would redistribute it; the
    cache mirrors it instead, pinned by URL and verified on every load.
    """
    root=Path(cache_root); root.mkdir(parents=True,exist_ok=True)
    for relative,url in files.items():
        target=root/relative
        if target.is_file(): continue
        target.parent.mkdir(parents=True,exist_ok=True)
        with urllib.request.urlopen(url,timeout=60) as response: target.write_bytes(response.read())
    return root
FACEXFORMER_FILES={"network/__init__.py":"https://raw.githubusercontent.com/Kartik-3004/facexformer/main/network/__init__.py",
    "network/models/__init__.py":"https://raw.githubusercontent.com/Kartik-3004/facexformer/main/network/models/__init__.py",
    "network/models/facexformer.py":"https://raw.githubusercontent.com/Kartik-3004/facexformer/main/network/models/facexformer.py",
    "network/models/transformer.py":"https://raw.githubusercontent.com/Kartik-3004/facexformer/main/network/models/transformer.py"}
ADAFACE_FILES={"adaface_net.py":"https://raw.githubusercontent.com/mk-minchul/AdaFace/master/net.py"}
class GeometryBackend(Protocol):
    def infer(self,images:list[np.ndarray])->list[dict[str,Any]]:...
@dataclass
class FaceXFormerBackend:
    """Official FaceXFormer: parsing labels and head pose in one model."""
    weight_path:Path; code_root:Path; device:str="cpu"; parsing_task:int=0; pose_task:int=2; num_classes:int=11
    def __post_init__(self):
        import torch
        sys.path.insert(0,str(self.code_root))
        _load_module("network",self.code_root/"network"/"__init__.py") if False else None
        from network import FaceXFormer  # noqa: E402  (pinned upstream code from the model cache)
        self.torch=torch; self.model=FaceXFormer()
        checkpoint=torch.load(self.weight_path,map_location="cpu",weights_only=False)
        self.model.load_state_dict(checkpoint["state_dict_backbone"],strict=True)
        self.model.eval().to(self.device)
        self.mean=np.asarray([0.485,0.456,0.406],np.float32); self.std=np.asarray([0.229,0.224,0.225],np.float32)
    def _tensor(self,image:np.ndarray):
        if image.shape[:2]!=(224,224): image=cv2.resize(image,(224,224),interpolation=cv2.INTER_CUBIC)
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)/255.
        return self.torch.from_numpy(((rgb-self.mean)/self.std).transpose(2,0,1))
    def _labels(self,batch:int):
        zeros=self.torch.zeros
        return {"segmentation":zeros([batch,224,224]),"lnm_seg":zeros([batch,5,2]),"landmark":zeros([batch,68,2]),
                "headpose":zeros([batch,3]),"attribute":zeros([batch,40]),"a_g_e":zeros([batch,3]),"visibility":zeros([batch,29])}
    def infer(self,images:list[np.ndarray])->list[dict[str,Any]]:
        torch=self.torch; tensors=[self._tensor(image) for image in images]
        stacked=torch.stack(tensors+tensors).to(self.device)
        tasks=torch.tensor([self.parsing_task]*len(images)+[self.pose_task]*len(images)).to(self.device)
        labels={k:v.to(self.device) for k,v in self._labels(len(stacked)).items()}
        with torch.inference_mode(): _,pose,_,_,_,_,_,segmentation=self.model(stacked,labels,tasks)
        parsing=segmentation.softmax(dim=1).argmax(dim=1).to("cpu").numpy().astype(np.uint8)
        poses=pose.to("cpu").numpy().astype(np.float32)
        if len(parsing)!=len(images) or len(poses)!=len(images): raise ModelPriorError("backend returned an unexpected batch size")
        return [{"parsing_labels":parsing[i],"pose_ypr":poses[i]} for i in range(len(images))]
@dataclass
class AdaFaceBackend:
    """Official AdaFace IR-50 identity embeddings (512-d, L2 normalized)."""
    weight_path:Path; code_root:Path; device:str="cpu"; architecture:str="ir_50"; embedding_dim:int=512; input_size:int=112
    def __post_init__(self):
        import torch
        net=_load_module("adaface_net",Path(self.code_root)/"adaface_net.py")
        self.torch=torch; self.model=net.build_model(self.architecture)
        checkpoint=torch.load(self.weight_path,map_location="cpu",weights_only=False)
        state={key[4:]:value for key,value in checkpoint.items() if key.startswith("net.")}
        self.model.load_state_dict(state,strict=True); self.model.eval().to(self.device)
    def _tensor(self,image:np.ndarray):
        if image.shape[:2]!=(self.input_size,self.input_size): image=cv2.resize(image,(self.input_size,)*2,interpolation=cv2.INTER_CUBIC)
        return self.torch.from_numpy(((image.astype(np.float32)/255.-0.5)/0.5).transpose(2,0,1))
    def embed(self,images:list[np.ndarray])->np.ndarray:
        torch=self.torch; stacked=torch.stack([self._tensor(image) for image in images]).to(self.device)
        with torch.inference_mode(): features,_=self.model(stacked)
        vectors=features.to("cpu").numpy().astype(np.float32)
        if vectors.shape[1]!=self.embedding_dim: raise ModelPriorError("identity embedding has an unexpected dimension")
        return vectors
# Parsing class groups (LaPa 11-class ordering used by FaceXFormer).
REGION_CLASSES={"left_eye":(4,),"right_eye":(5,),"nose":(6,),"mouth":(7,8,9),
                "forehead":(1,),"left_cheek":(1,),"right_cheek":(1,),"face_boundary":(1,2,3,4,5,6,7,8,9),"context":(0,10)}
def region_masks(parsing:np.ndarray)->dict[str,np.ndarray]:
    """Spatial region masks combining parsing classes with crop geometry."""
    height,width=parsing.shape; face=np.isin(parsing,REGION_CLASSES["face_boundary"])
    rows=np.arange(height)[:,None]*np.ones((1,width)); columns=np.ones((height,1))*np.arange(width)[None,:]
    skin=parsing==1
    return {"left_eye":parsing==4,"right_eye":parsing==5,"nose":parsing==6,"mouth":np.isin(parsing,(7,8,9)),
            "forehead":skin&(rows<height*0.35),"left_cheek":skin&(columns<width*0.45)&(rows>=height*0.35),
            "right_cheek":skin&(columns>width*0.55)&(rows>=height*0.35),"face_boundary":face,
            "context":np.isin(parsing,REGION_CLASSES["context"])}
def compute_visibility(parsing:np.ndarray,pose_ypr:np.ndarray,*,yaw_scale:float=np.pi/2)->np.ndarray:
    """Per-region visibility in [0,1] derived from parsing area and pose yaw.

    Region presence is the parsing-area fraction relative to a reference share
    of the detected face; the cheeks are additionally attenuated by yaw-driven
    self-occlusion, so the vector is never a constant.
    """
    masks=region_masks(parsing); face=float(masks["face_boundary"].sum()); total=float(parsing.size)
    yaw=float(pose_ypr[0]); occlusion=float(np.clip(abs(yaw)/max(yaw_scale,1e-6),0.,1.))
    reference={"left_eye":0.01,"right_eye":0.01,"nose":0.05,"mouth":0.03,"forehead":0.10,
               "left_cheek":0.08,"right_cheek":0.08,"face_boundary":0.35,"context":0.10}
    values=[]
    for region in VISIBILITY_REGIONS:
        area=float(masks[region].sum()); denominator=face if region!="context" else total
        share=area/denominator if denominator>0 else 0.
        value=float(np.clip(share/reference[region],0.,1.))
        if region=="left_cheek" and yaw>0: value*=(1.-occlusion)
        if region=="right_cheek" and yaw<0: value*=(1.-occlusion)
        values.append(value)
    vector=np.asarray(values,dtype=np.float16)
    if not np.isfinite(vector.astype(np.float32)).all(): raise ModelPriorError("visibility vector is not finite")
    return vector
def validate_pose(pose:np.ndarray)->np.ndarray:
    pose=np.asarray(pose,dtype=np.float32).reshape(-1)
    if pose.shape!=(3,) or not np.isfinite(pose).all(): raise ModelPriorError("pose_ypr must be three finite values")
    return pose
def validate_parsing(parsing:np.ndarray,num_classes:int=11)->np.ndarray:
    parsing=np.asarray(parsing,dtype=np.uint8)
    if parsing.shape!=(224,224): raise ModelPriorError(f"parsing_labels must be [224,224], got {parsing.shape}")
    if int(parsing.max(initial=0))>=num_classes: raise ModelPriorError("parsing label exceeds the configured class count")
    if len(np.unique(parsing))<2: raise ModelPriorError("parsing mask is degenerate (single class)")
    return parsing
def validate_identity(embedding:np.ndarray,dimension:int=512)->np.ndarray:
    vector=np.asarray(embedding,dtype=np.float32).reshape(-1)
    if vector.shape!=(dimension,) or not np.isfinite(vector).all(): raise ModelPriorError("identity embedding is invalid")
    norm=float(np.linalg.norm(vector))
    if not (0.5<=norm<=1.5): raise ModelPriorError(f"identity embedding L2 norm out of range: {norm:.4f}")
    return vector.astype(np.float16)
def environment_fingerprint(device:str)->dict[str,str]:
    import platform
    try:
        import torch; torch_version=torch.__version__; cuda=str(torch.version.cuda); available=str(torch.cuda.is_available())
    except Exception: torch_version=cuda=available="unavailable"
    return {"python":platform.python_version(),"torch":torch_version,"cuda":cuda,"cuda_available":available,
            "device":device,"opencv":cv2.__version__,"numpy":np.__version__}
