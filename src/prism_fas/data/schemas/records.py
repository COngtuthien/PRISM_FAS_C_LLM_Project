from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict
class CanonicalVideoRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset: str; subject_id: str | None; video_id: str; source_path: Path; official_split: str | None
    label: Literal["live", "spoof"] | None = None; capture_metadata: dict[str, Any] = {}
    adapter_version: str; protocol_version: str | None = None; source_fingerprint: str; metadata_provenance: str
class TargetInferenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset: Literal["siw_mv2"]; video_id: str; source_path: Path; official_split: str; adapter_version: str; source_fingerprint: str; metadata_provenance: str
