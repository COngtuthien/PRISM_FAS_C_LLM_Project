from __future__ import annotations
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, ConfigDict
class PreprocessingRunProfile(BaseModel):
 model_config=ConfigDict(extra='forbid')
 name:Literal['small_acceptance','full_preprocessing'];output_namespace:str;record_selection_mode:Literal['limited','all_records'];default_record_limit:int|None;require_explicit_confirmation:bool;validation_profile:Literal['small_acceptance','full_preprocessing'];preserve_existing_outputs:bool;expected_datasets:list[str];description:str;profile_version:str
def load_profiles(path:Path)->dict[str,PreprocessingRunProfile]:
 return {k:PreprocessingRunProfile.model_validate(v) for k,v in yaml.safe_load(path.read_text()).items()}
def profile_root(work:Path,version:str,config_hash:str,profile:PreprocessingRunProfile,override:Path|None=None)->Path:
 root=(override or work/'m2'/version/config_hash/profile.output_namespace).resolve()
 base=(work/'m2'/version/config_hash).resolve()
 if not str(root).startswith(str(base)) or '..' in root.parts:raise ValueError('unsafe output root')
 return root
