from __future__ import annotations
import hashlib, io, json, os, tarfile, tempfile
from pathlib import Path
from typing import Iterable
from prism_fas.data.manifests.parquet_writer import _replace_with_retry

MEMBER_SUFFIXES=(".jpg",".npz",".json")
def _member(name:str,data:bytes)->tarfile.TarInfo:
    # Fixed metadata keeps shard bytes reproducible across machines and runs.
    info=tarfile.TarInfo(name); info.size=len(data); info.mtime=0; info.mode=0o644
    info.uid=info.gid=0; info.uname=info.gname=""; info.type=tarfile.REGTYPE
    return info
def _safe(name:str)->str:
    if name.startswith("/") or name.startswith("\\") or ".." in Path(name).parts or ":" in name:
        raise ValueError(f"unsafe tar member path: {name!r}")
    return name
def write_shard(path:Path,entries:Iterable[tuple[str,bytes,bytes,dict]])->dict:
    """Write one uncompressed, deterministically ordered tar shard.

    Each entry contributes exactly <sample_id>.jpg, .npz and .json in that order.
    """
    ordered=sorted(entries,key=lambda entry:entry[0]); path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix=path.name+".",suffix=".tmp",dir=path.parent); os.close(fd)
    seen=set()
    try:
        with tarfile.open(temporary,"w",format=tarfile.PAX_FORMAT) as archive:
            for sample_id,image,prior,metadata in ordered:
                payload=json.dumps(metadata,sort_keys=True,separators=(",",":")).encode("utf-8")
                for suffix,data in ((".jpg",image),(".npz",prior),(".json",payload)):
                    name=_safe(f"{sample_id}{suffix}")
                    if name in seen: raise ValueError(f"duplicate tar member: {name}")
                    seen.add(name); archive.addfile(_member(name,data),io.BytesIO(data))
        _replace_with_retry(temporary,path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    return {"shard_filename":path.name,"first_sample_id":ordered[0][0],"last_sample_id":ordered[-1][0],
            "row_count":len(ordered),"byte_size":path.stat().st_size,"sha256":digest}
def plan_shards(sample_ids:list[str],sizes:dict[str,int],*,max_samples:int,max_bytes:int)->list[list[str]]:
    """Group sorted sample IDs into shards honouring the sample/byte limits."""
    groups:list[list[str]]=[]; current:list[str]=[]; total=0
    for sample_id in sorted(sample_ids):
        size=sizes.get(sample_id,0)
        if current and (len(current)>=max_samples or total+size>max_bytes): groups.append(current); current=[]; total=0
        current.append(sample_id); total+=size
    if current: groups.append(current)
    return groups
def read_shard_members(path:Path)->dict[str,set[str]]:
    members:dict[str,set[str]]={}
    with tarfile.open(path,"r") as archive:
        for info in archive.getmembers():
            if not info.isfile(): raise ValueError(f"non-regular tar member: {info.name}")
            _safe(info.name); stem,suffix=info.name.rsplit(".",1)
            members.setdefault(stem,set()).add("."+suffix)
    return members
