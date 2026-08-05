from __future__ import annotations
import io, json, tarfile
from pathlib import Path
from typing import Iterator
import numpy as np
from prism_fas.data.package.manifests import read_manifest
from .config import INFERENCE_SPLIT, LoaderConfig
from .contracts import CanonicalSourceSample, CanonicalTargetSample, SampleContractError, TargetIsolationViolation
from .package_index import PackageIndex, open_package
from .transforms import decode_image, geometry_from_arrays

MEMBER_SUFFIXES=(".jpg",".npz",".json")
class CanonicalShardDataset:
    """Iterable dataset streaming <id>.jpg/.npz/.json triplets from package tars.

    One triplet is held in memory at a time; shards are never extracted to disk.
    """
    def __init__(self,package_root:Path,split:str,config:LoaderConfig,*,mode:str,index:PackageIndex|None=None):
        self.config=config; self.mode=mode
        self.index=index or open_package(package_root,split,config,mode=mode)
        self.split=self.index.split; self.root=self.index.root; self.is_target=self.split==INFERENCE_SPLIT
        rows=read_manifest(self.root/"manifests"/"shards_index.parquet")
        self.shards=tuple(sorted((row for row in rows if row["split"]==self.split),key=lambda row:row["shard_filename"]))
        if not self.shards: raise SampleContractError(f"no shards for split {self.split!r}")
        foreign=[row["shard_filename"] for row in rows if row["split"]!=self.split and row["shard_filename"] in {s["shard_filename"] for s in self.shards}]
        if foreign: raise TargetIsolationViolation(f"shard appears under multiple splits: {foreign}")
        self._by_id={row["sample_id"]:row for row in self.index.rows}
    def __len__(self)->int: return sum(row["row_count"] for row in self.shards)
    def __iter__(self)->Iterator:
        for shard in self.shards:
            path=self.root/"shards"/shard["shard_filename"]
            with tarfile.open(path,"r") as archive:
                pending:dict[str,dict[str,bytes]]={}
                seen=set()
                for info in archive:
                    if not info.isfile(): raise SampleContractError(f"non-regular tar member: {info.name}")
                    if info.name.startswith(("/","\\")) or ".." in Path(info.name).parts or ":" in info.name:
                        raise SampleContractError(f"unsafe tar member path: {info.name}")
                    if info.name in seen: raise SampleContractError(f"duplicate tar member: {info.name}")
                    seen.add(info.name)
                    stem,_,suffix=info.name.rpartition(".")
                    handle=archive.extractfile(info)
                    if handle is None: raise SampleContractError(f"unreadable tar member: {info.name}")
                    pending.setdefault(stem,{})["."+suffix]=handle.read()
                    entry=pending[stem]
                    if set(entry)=={*MEMBER_SUFFIXES}:
                        yield self._materialize(stem,entry); pending.pop(stem)
                if pending:
                    raise SampleContractError(f"incomplete shard triplets in {shard['shard_filename']}: {sorted(pending)[:5]}")
    def _materialize(self,sample_id:str,entry:dict[str,bytes]):
        row=self._by_id.get(sample_id)
        if row is None: raise SampleContractError(f"shard sample {sample_id} is not in split manifest {self.split!r}")
        metadata=json.loads(entry[".json"].decode("utf-8"))
        if metadata.get("project_split")!=self.split:
            raise TargetIsolationViolation(f"shard member {sample_id} declares split {metadata.get('project_split')!r}")
        image=decode_image(entry[".jpg"],self.config.image)
        with np.load(io.BytesIO(entry[".npz"]),allow_pickle=False) as handle:
            arrays={name:handle[name] for name in handle.files}
        geometry=geometry_from_arrays(arrays)
        if self.is_target:
            if "identity_embedding" in arrays: raise SampleContractError(f"target shard prior carries an identity embedding: {sample_id}")
            if any(key in metadata for key in ("label_live_spoof","official_split","subject_id")):
                raise TargetIsolationViolation(f"target shard JSON exposes forbidden fields: {sample_id}")
            sample=CanonicalTargetSample(sample_id=sample_id,dataset=row["dataset"],project_split=self.split,
                source_record_id=row["source_record_id"],image=image,geometry=geometry,
                crop_sha256=row["crop_sha256"],prior_sha256=row["prior_sha256"],
                source_media_type=metadata.get("source_media_type",""))
        else:
            embedding=arrays.get("identity_embedding"); available=embedding is not None
            sample=CanonicalSourceSample(sample_id=sample_id,dataset=row["dataset"],project_split=self.split,
                source_record_id=row["source_record_id"],requested_frame_index=int(metadata.get("requested_frame_index",0)),
                actual_frame_index=int(metadata.get("actual_frame_index",0)),label=row["label_live_spoof"],
                class_target=self.config.label_to_index(row["label_live_spoof"]),image=image,geometry=geometry,
                identity_embedding=embedding.astype(np.float32) if available else None,identity_available=bool(available),
                crop_sha256=row["crop_sha256"],prior_sha256=row["prior_sha256"],
                source_media_type=metadata.get("source_media_type",""))
        sample.validate(); return sample
