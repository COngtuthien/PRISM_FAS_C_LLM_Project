from __future__ import annotations
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, ConfigDict, Field
from prism_fas.utils.core import stable_json_hash

TRAIN_SCHEMA_VERSION="m5-b00-v1"
class ModelSpec(BaseModel):
    model_config=ConfigDict(extra="forbid",protected_namespaces=())
    model_name:str; source:Literal["timm"]; revision:str; weight_sha256:str; pretrained:bool
    image_size:int; normalization_mean:tuple[float,float,float]; normalization_std:tuple[float,float,float]
    dropout:float=0.; feature_dim:int|None=None
class OptimizerSpec(BaseModel):
    model_config=ConfigDict(extra="forbid")
    name:Literal["AdamW"]; backbone_lr:float; head_lr:float; weight_decay:float; betas:tuple[float,float]
class SchedulerSpec(BaseModel):
    model_config=ConfigDict(extra="forbid")
    name:Literal["cosine"]; warmup_epochs:int; min_lr:float
class SelectionSpec(BaseModel):
    model_config=ConfigDict(extra="forbid")
    primary:str; mode:Literal["max","min"]; tie_breaker:str; tie_breaker_mode:Literal["max","min"]
class EarlyStoppingSpec(BaseModel):
    model_config=ConfigDict(extra="forbid")
    enabled:bool; patience_epochs:int; min_epochs:int
class CalibrationSpec(BaseModel):
    model_config=ConfigDict(extra="forbid")
    temperature_scaling:bool; threshold_criterion:Literal["min_acer"]; reject_policy:Literal["disabled_for_b00"]
class B00Config(BaseModel):
    model_config=ConfigDict(extra="forbid")
    experiment_id:str; stage:str; model:ModelSpec; label_mapping:dict[str,int]
    seed:int; epochs:int=Field(gt=0); batch_size:int=Field(gt=0); steps_per_epoch:int=Field(gt=0)
    optimizer:OptimizerSpec; scheduler:SchedulerSpec; gradient_clip_norm:float
    augmentation:dict[str,float]; precision:dict[str,str]; checkpoint_selection:SelectionSpec
    early_stopping:EarlyStoppingSpec; calibration:CalibrationSpec; video_aggregation:dict[str,object]
    num_workers:dict[str,int]; package:dict[str,str]
    @property
    def config_hash(self)->str: return stable_json_hash(self.model_dump(mode="json"))
def load_b00_config(path:Path)->B00Config:
    return B00Config.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
