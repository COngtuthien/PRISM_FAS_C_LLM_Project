from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from prism_fas.data.package.manifests import read_manifest
from .config import INFERENCE_SPLIT, LoaderConfig, TRAINING_SPLIT, VALIDATION_SPLIT
from .contracts import TargetIsolationViolation

class PackageContractError(ValueError):
    """The package on disk does not satisfy the loader's package policy."""
SPLIT_MANIFEST={TRAINING_SPLIT:"source_train",VALIDATION_SPLIT:"source_dev",INFERENCE_SPLIT:"target_test_features"}
MODE_SPLITS={"training":(TRAINING_SPLIT,),"validation":(VALIDATION_SPLIT,),"inference":(INFERENCE_SPLIT,)}
@dataclass(frozen=True)
class PackageIndex:
    """Validated, read-only view of an M3B package.

    The lock/identity checks run once here, never per sample access.
    """
    root:Path; package_id:str; content_identity:str; parent_package_id:str
    split:str; rows:tuple[dict,...]; label_mapping:dict[str,int]
    @property
    def sample_ids(self)->tuple[str,...]: return tuple(row["sample_id"] for row in self.rows)
    def __len__(self)->int: return len(self.rows)
def open_package(root:Path,split:str,config:LoaderConfig,*,mode:str)->PackageIndex:
    """Open a package split under a mode, enforcing target isolation up front."""
    root=Path(root)
    if mode not in MODE_SPLITS: raise ValueError(f"unknown loader mode: {mode!r}")
    if split not in SPLIT_MANIFEST: raise ValueError(f"unknown split: {split!r}")
    if split not in MODE_SPLITS[mode]:
        if split==INFERENCE_SPLIT:
            raise TargetIsolationViolation(f"target isolation violation: split {split!r} cannot be opened in {mode!r} mode")
        raise PackageContractError(f"split {split!r} is not permitted in {mode!r} mode")
    lock_path=root/"PACKAGE_LOCK.json"
    if not lock_path.is_file(): raise PackageContractError(f"PACKAGE_LOCK.json missing under {root.name}")
    lock=json.loads(lock_path.read_text(encoding="utf-8"))
    policy=config.package
    if policy.require_validated_status and lock.get("status")!="validated":
        raise PackageContractError(f"package status is {lock.get('status')!r}, expected 'validated'")
    if lock.get("package_id")!=policy.expected_package_id:
        raise PackageContractError(f"package id {lock.get('package_id')!r} != expected {policy.expected_package_id!r}")
    if policy.expected_content_identity_sha256 and lock.get("content_identity_sha256")!=policy.expected_content_identity_sha256:
        raise PackageContractError("package content identity does not match the configured pin")
    manifest=root/"manifests"/f"{SPLIT_MANIFEST[split]}.parquet"
    if not manifest.is_file(): raise PackageContractError(f"missing split manifest for {split!r}")
    rows=sorted(read_manifest(manifest),key=lambda row:row["sample_id"])
    if not rows: raise PackageContractError(f"split {split!r} is empty")
    if len({row["sample_id"] for row in rows})!=len(rows): raise PackageContractError(f"duplicate sample ids in {split!r}")
    if split!=INFERENCE_SPLIT:
        missing=[row["sample_id"] for row in rows if not row.get("label_live_spoof")]
        if missing: raise PackageContractError(f"source split {split!r} has {len(missing)} rows without a label")
        unknown=sorted({row["label_live_spoof"] for row in rows}-set(config.label_mapping))
        if unknown: raise PackageContractError(f"labels outside the configured vocabulary: {unknown}")
        datasets={row["dataset"] for row in rows}
        if "siw_mv2" in datasets: raise TargetIsolationViolation("target dataset found inside a source split")
    else:
        leaked=sorted({key for row in rows for key in row} & {"label_live_spoof","subject_id","official_split"})
        if leaked: raise TargetIsolationViolation(f"target manifest exposes forbidden fields: {leaked}")
    for row in rows:
        for key in ("image_relative_path","prior_relative_path"):
            value=row[key]
            if value.startswith(("/","\\")) or ".." in Path(value).parts or ":" in value:
                raise PackageContractError(f"unsafe relative path in manifest: {key}")
    return PackageIndex(root=root,package_id=lock["package_id"],content_identity=lock["content_identity_sha256"],
                        parent_package_id=lock.get("parent_package_id",""),split=split,rows=tuple(rows),
                        label_mapping=dict(config.label_mapping))
def package_summary(root:Path)->dict:
    lock=json.loads((Path(root)/"PACKAGE_LOCK.json").read_text(encoding="utf-8"))
    return {"package_id":lock["package_id"],"content_identity_sha256":lock["content_identity_sha256"],
            "parent_package_id":lock.get("parent_package_id"),"status":lock["status"],
            "per_split_counts":lock.get("per_split_counts"),"total_samples":lock.get("total_samples")}
