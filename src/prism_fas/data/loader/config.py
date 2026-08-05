from __future__ import annotations
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, ConfigDict, Field

LOADER_SCHEMA_VERSION="m4-loader-v1"
TRAINING_SPLIT="source_train"; VALIDATION_SPLIT="source_dev"; INFERENCE_SPLIT="target_test"
class PackagePolicy(BaseModel):
    model_config=ConfigDict(extra="forbid")
    expected_package_id:str; expected_content_identity_sha256:str|None=None
    require_validated_status:bool=True; integrity_verification:Literal["on_open","per_sample_sha","none"]="on_open"
class ImagePolicy(BaseModel):
    model_config=ConfigDict(extra="forbid")
    decoder:Literal["opencv","pillow"]="opencv"; color:Literal["rgb"]="rgb"; size:tuple[int,int]=(224,224)
    dtype:Literal["float32"]="float32"; value_range:tuple[float,float]=(0.,1.); channels_first:bool=True
class SamplerPolicy(BaseModel):
    model_config=ConfigDict(extra="forbid")
    batch_size:int=Field(gt=0); steps_per_epoch_policy:str; steps_per_epoch:int|None=None; seed:int
    drop_last:bool=True; balance:Literal["exact_domain_class"]="exact_domain_class"
    pool_key:list[str]; avoid_same_record_in_batch:bool=True
    minority_policy:Literal["reshuffle_cycle"]="reshuffle_cycle"
class BackendPolicy(BaseModel):
    model_config=ConfigDict(extra="forbid")
    default:Literal["loose","shard"]="loose"; shard_shuffle:bool=False
class DataLoaderPolicy(BaseModel):
    model_config=ConfigDict(extra="forbid")
    num_workers:int=0; prefetch_factor:int|None=None; persistent_workers:bool=False; pin_memory:bool=False
class LoaderConfig(BaseModel):
    model_config=ConfigDict(extra="forbid")
    loader_schema_version:str; package:PackagePolicy; image:ImagePolicy
    label_mapping:dict[str,int]; sampler:SamplerPolicy; backends:BackendPolicy; dataloader:DataLoaderPolicy
    def label_to_index(self,label:str)->int:
        if label not in self.label_mapping: raise ValueError(f"label outside the configured vocabulary: {label!r}")
        return self.label_mapping[label]
def load_loader_config(path:Path)->LoaderConfig:
    return LoaderConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
