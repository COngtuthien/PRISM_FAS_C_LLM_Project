from __future__ import annotations
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, model_validator

class M2OutputLayout(BaseModel):
    model_config=ConfigDict(extra='forbid')
    output_root:Path; crops_root:Path; frames_root:Path; manifests_root:Path; state_root:Path; reports_root:Path; logs_root:Path
    @classmethod
    def from_root(cls,root:Path)->'M2OutputLayout':
        root=root.resolve();return cls(output_root=root,crops_root=root/'crops',frames_root=root/'frames',manifests_root=root/'manifests',state_root=root/'state',reports_root=root/'reports',logs_root=root/'logs')
class PreprocessingRunContext(BaseModel):
    model_config=ConfigDict(extra='forbid',arbitrary_types_allowed=True)
    project_root:Path;work_root:Path;run_profile:Literal['small_acceptance','full_preprocessing'];output_namespace:str;output_root:Path;crops_root:Path;frames_root:Path;manifests_root:Path;state_root:Path;reports_root:Path;logs_root:Path;run_id:str;dataset:str;dataset_role:Literal['source','target'];preprocessing_version:str;preprocessing_config_hash:str;detector_model_path:Path;detector_model_sha256:str;detector_input_size:int;detector_threshold:float;all_records:bool;record_limit:int|None;sample_limit:int|None;resume:bool;dry_run:bool;partial_full_profile:bool;command:str
    @model_validator(mode='after')
    def safe(self):
        root=self.output_root.resolve()
        if self.run_profile=='full_preprocessing' and self.output_namespace!='full_preprocessing':raise ValueError('full profile namespace mismatch')
        if self.run_profile=='full_preprocessing' and 'm2a' in root.parts:raise ValueError('full profile cannot use small namespace')
        for child in [self.crops_root,self.frames_root,self.manifests_root,self.state_root]:
            if not str(child.resolve()).startswith(str(root)):raise ValueError('layout path escapes profile root')
        return self
