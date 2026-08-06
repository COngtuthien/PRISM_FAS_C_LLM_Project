from __future__ import annotations
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, ConfigDict, Field

CLOUD_SCHEMA_VERSION="m6-modal-v1"
FORBIDDEN_UPLOAD_MARKERS=("Dataset","data/work","casia-fasd","MSU-MFSD","SiW-Mv2","paths.local.yaml","model_cache")
class VolumeSpec(BaseModel):
    model_config=ConfigDict(extra="forbid")
    name:str; mount:str; mode:Literal["read_only","read_write"]
class CloudConfig(BaseModel):
    model_config=ConfigDict(extra="forbid",protected_namespaces=())
    cloud_schema_version:str; app_name:str; volumes:dict[str,VolumeSpec]; remote_paths:dict[str,str]
    package:dict; model:dict; gpu:dict; timeouts_seconds:dict[str,int]; image:dict; parity:dict; train_smoke:dict
    def volume(self,key:str)->VolumeSpec: return self.volumes[key]
    def resolve_gpu(self,requested:str|None)->str:
        gpu=requested or self.gpu["default"]
        if gpu not in self.gpu["allow_list"]:
            raise ValueError(f"gpu {gpu!r} is not in the allow-list {self.gpu['allow_list']}")
        return gpu
def load_cloud_config(path:Path)->CloudConfig:
    return CloudConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
def assert_upload_is_safe(local_path:Path)->Path:
    """Refuse to upload raw datasets, work directories or local secrets."""
    text=str(Path(local_path).as_posix())
    for marker in FORBIDDEN_UPLOAD_MARKERS:
        if marker.lower() in text.lower():
            raise PermissionError(f"refusing to upload path containing {marker!r}")
    return Path(local_path)
def assert_remote_path_is_safe(remote:str)->str:
    """Remote paths must be absolute volume paths without traversal or drive letters."""
    if not remote.startswith("/"): raise ValueError(f"remote path must be absolute: {remote!r}")
    if ".." in remote.split("/") or ":" in remote: raise ValueError(f"unsafe remote path: {remote!r}")
    return remote
def assert_no_absolute_local_paths(payload:dict)->dict:
    """Portable run metadata must not carry local Windows/Unix absolute paths."""
    import json, re
    text=json.dumps(payload,default=str)
    if re.search(r"[A-Za-z]:[\\/]|/home/|/Users/",text):
        raise ValueError("portable metadata contains an absolute local path")
    return payload
def redact(value:str)->str:
    """Redact anything token-shaped before it reaches a report or log."""
    if not value: return value
    return value[:3]+"…redacted…" if value.startswith(("ak-","as-","st-")) else value
