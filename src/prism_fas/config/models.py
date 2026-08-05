from __future__ import annotations
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

class StrictModel(BaseModel): model_config = ConfigDict(extra="forbid")
class RawDatasets(StrictModel): casia_fasd: Path; msu_mfsd: Path; siw_mv2: Path
class PathsConfig(StrictModel):
    workspace_root: Path; project_root: Path; raw_datasets: RawDatasets; model_cache: Path
    work_root: Path; processed_root: Path; package_root: Path; runs_root: Path; reports_root: Path
class DatasetDefinition(StrictModel):
    dataset: Literal["casia_fasd", "msu_mfsd", "siw_mv2"]; adapter_version: str; root_relative: Path = Path(".")
    record_kind: str | None = None
    include_globs: list[str] = Field(default_factory=list); path_pattern: str = ""
    labels: dict[str, str] = Field(default_factory=dict); splits: dict[str, str] = Field(default_factory=dict)
    protocol_files: dict[str, Path] = Field(default_factory=dict); filename_patterns: dict[str, str] = Field(default_factory=dict); provenance: dict[str, dict] = Field(default_factory=dict)
    group_by: str | None = None; official_split: str | None = None; private_evaluation_fields: list[str] = Field(default_factory=list); blocked_reason: str | None = None
class RawAuditConfig(StrictModel): duplicate_hash_limit_bytes: int = Field(default=5_000_000, ge=1)
class RuntimeConfig(StrictModel): log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
class ReproducibilityMetadata(StrictModel): seed: int = Field(default=42, ge=0); deterministic: bool = True
class ResolvedConfig(StrictModel): paths: PathsConfig; raw_audit: RawAuditConfig = RawAuditConfig(); runtime: RuntimeConfig = RuntimeConfig(); reproducibility: ReproducibilityMetadata = ReproducibilityMetadata()
def load_paths(path: Path) -> PathsConfig:
    if not path.is_file(): raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open(encoding="utf-8") as handle: return PathsConfig.model_validate(yaml.safe_load(handle))
def resolve_config(path: Path) -> ResolvedConfig: return ResolvedConfig(paths=load_paths(path))
