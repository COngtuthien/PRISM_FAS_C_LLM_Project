from __future__ import annotations
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, ConfigDict, Field
from prism_fas.utils.core import stable_json_hash

PACKAGE_SCHEMA_VERSION="m3a-v1"
PRIOR_SCHEMA_VERSION="m3a-priors-v1"
QUALITY_SCHEMA_VERSION="m3a-quality-v1"
# Model-dependent priors are deliberately deferred; M3A records their absence
# explicitly instead of writing zero-filled placeholders that look valid.
DEFERRED_PRIOR_STATUS={"parsing_status":"not_computed","pose_status":"not_computed","visibility_status":"not_computed","identity_status":"not_computed"}
SPLITS=("source_train","source_dev","target_test")
OFFICIAL_SPLIT_TO_PROJECT_SPLIT={"train":"source_train","test":"source_dev","dev":"source_dev"}

class M3APackageConfig(BaseModel):
    model_config=ConfigDict(extra="forbid")
    package_schema_version:str; prior_schema_version:str; quality_schema_version:str
    package_id_prefix:str; shard_max_samples:int=Field(gt=0); shard_max_bytes:int=Field(gt=0)
    quality_metrics:list[str]; image_policy:Literal["preserve_m2_bytes"]; image_format:Literal["jpeg"]
    image_extension:str; target_isolation_policy:str; resume_policy:str
    @property
    def config_hash(self)->str: return stable_json_hash(self.model_dump(mode="json"))
def load_package_config(path:Path)->M3APackageConfig:
    return M3APackageConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
def project_split(dataset_role:str,official_split:str|None)->str:
    """Map M2 role/official split onto the canonical project split."""
    if dataset_role=="target": return "target_test"
    key=(official_split or "").strip().lower()
    if key not in OFFICIAL_SPLIT_TO_PROJECT_SPLIT: raise ValueError(f"unmapped official split: {official_split!r}")
    return OFFICIAL_SPLIT_TO_PROJECT_SPLIT[key]
