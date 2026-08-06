from __future__ import annotations
import hashlib, io, json, tarfile
from pathlib import Path
import numpy as np

VALIDATION_PROFILE="remote_parity"
EXPECTED_SPLITS={"source_train":1440,"source_dev":2079,"target_test":3140}
FORBIDDEN_TARGET_KEYS=frozenset({"label","label_live_spoof","true_label","true_target","attack_type","taxonomy",
    "subject_id","session_id","identity_embedding","official_split","source_path","crop_relative_path"})
M3B_ARRAYS={"parsing_labels":((224,224),"uint8"),"pose_ypr":((3,),"float32"),"visibility":((9,),"float16"),
            "bbox":((4,),"float32"),"landmarks":((5,2),"float32"),"quality_vector":((6,),"float32")}
def _sha256(path:Path,chunk:int=8*1024*1024)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(chunk),b""): digest.update(block)
    return digest.hexdigest()
def verify_remote_package(package_root:Path,*,expected_identity:str,expected_parent:str|None=None,
                          per_shard_samples:int=4,per_split_samples:int=12)->dict:
    """Shard-first remote verification of an uploaded M3B package.

    This is NOT the full local loose-file validator. It proves the bytes the
    remote trainer actually reads (9 tar shards + manifests + lock) arrived
    intact, then deterministically opens real triplets from every shard. Loose
    per-file hashing of 13k small files over a network volume is intentionally
    avoided; the local package validator remains unchanged and authoritative
    for the loose tree.
    """
    root=Path(package_root); checks=[];errors=[]
    def check(name:str,expected,actual,category:str="package")->bool:
        ok=expected==actual
        checks.append({"check_id":name,"category":category,"expected":str(expected)[:160],
                       "actual":str(actual)[:160],"passed":ok})
        if not ok: errors.append(name)
        return ok
    lock_path=root/"PACKAGE_LOCK.json"
    check("lock.exists",True,lock_path.is_file())
    if not lock_path.is_file():
        return {"validation_profile":VALIDATION_PROFILE,"passed":False,"errors":errors,"checks":checks}
    lock=json.loads(lock_path.read_text(encoding="utf-8"))
    check("lock.package_id","prism_data_v1_m3b",lock.get("package_id"))
    check("lock.status","validated",lock.get("status"))
    check("lock.content_identity",expected_identity,lock.get("content_identity_sha256"))
    if expected_parent: check("lock.parent_identity",expected_parent,lock.get("parent_content_identity_sha256"))
    check("lock.parent_package_id","prism_data_v1_m3a",lock.get("parent_package_id"))
    manifest_hashes=lock.get("manifest_sha256") or {}
    for name,digest in sorted(manifest_hashes.items()):
        path=root/"manifests"/f"{name}.parquet"
        check(f"manifest.exists.{name}",True,path.is_file(),"manifests")
        if path.is_file(): check(f"manifest.sha.{name}",digest,_sha256(path),"manifests")
    check("splits.counts",EXPECTED_SPLITS,lock.get("per_split_counts"),"counts")
    shards=lock.get("shards") or []
    check("shards.count",9,len(shards),"shards")
    split_rows={};shard_members={}
    for entry in shards:
        path=root/"shards"/entry["shard_filename"]
        if not check(f"shard.exists.{entry['shard_filename']}",True,path.is_file(),"shards"): continue
        check(f"shard.sha.{entry['shard_filename']}",entry["sha256"],_sha256(path),"shards")
        check(f"shard.size.{entry['shard_filename']}",entry["byte_size"],path.stat().st_size,"shards")
        split_rows[entry["split"]]=split_rows.get(entry["split"],0)+entry["row_count"]
    check("shards.total_rows",6659,sum(entry["row_count"] for entry in shards),"counts")
    check("shards.split_rows",EXPECTED_SPLITS,split_rows,"counts")
    sampled={"source_train":0,"source_dev":0,"target_test":0}; triplet_failures=[];decode_failures=[];leak_failures=[]
    for entry in sorted(shards,key=lambda row:row["shard_filename"]):
        path=root/"shards"/entry["shard_filename"]
        if not path.is_file(): continue
        split=entry["split"]; taken=0; grouped={}
        with tarfile.open(path,"r") as archive:
            for info in archive:
                if not info.isfile(): continue
                if info.name.startswith(("/","\\")) or ".." in Path(info.name).parts:
                    triplet_failures.append(info.name); continue
                stem,_,suffix=info.name.rpartition(".")
                handle=archive.extractfile(info)
                grouped.setdefault(stem,{})["."+suffix]=handle.read() if handle else b""
                if set(grouped[stem])=={".jpg",".npz",".json"}:
                    payload=grouped.pop(stem)
                    if taken>=per_shard_samples and sampled[split]>=per_split_samples: break
                    taken+=1; sampled[split]+=1
                    _inspect(stem,payload,split,decode_failures,leak_failures)
                    shard_members[entry["shard_filename"]]=taken
        if taken<min(per_shard_samples,entry["row_count"]):
            triplet_failures.append(f"{entry['shard_filename']}:only {taken} complete triplets sampled")
    check("triplets.sampled_per_shard",True,all(count>=1 for count in shard_members.values()),"triplets")
    for split,minimum in (("source_train",per_split_samples),("source_dev",per_split_samples),("target_test",per_split_samples)):
        check(f"triplets.sampled.{split}",True,sampled[split]>=minimum,"triplets")
    check("triplets.complete",[],triplet_failures[:5],"triplets")
    check("triplets.decode",[],decode_failures[:5],"triplets")
    check("target.isolation",[],leak_failures[:5],"target_isolation")
    passed=not errors
    return {"validation_profile":VALIDATION_PROFILE,"passed":passed,"errors":errors,"checks":checks,
            "checks_passed":sum(1 for row in checks if row["passed"]),"checks_total":len(checks),
            "package_id":lock.get("package_id"),"content_identity":lock.get("content_identity_sha256"),
            "parent_package_id":lock.get("parent_package_id"),"per_split_counts":lock.get("per_split_counts"),
            "shards":[{"shard_filename":entry["shard_filename"],"split":entry["split"],"row_count":entry["row_count"],
                       "byte_size":entry["byte_size"],"sha256":entry["sha256"]} for entry in shards],
            "sampled_triplets":sampled,"sampled_per_shard":shard_members,
            "note":"shard-first remote transfer-integrity verification; the full loose-file validator runs locally"}
def _inspect(sample_id:str,payload:dict[str,bytes],split:str,decode_failures:list,leak_failures:list)->None:
    import cv2
    image=cv2.imdecode(np.frombuffer(payload[".jpg"],dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None or image.shape!=(224,224,3): decode_failures.append(f"{sample_id}:image")
    try:
        with np.load(io.BytesIO(payload[".npz"]),allow_pickle=False) as handle:
            arrays={name:handle[name] for name in handle.files}
    except Exception: decode_failures.append(f"{sample_id}:npz"); return
    for name,(shape,dtype) in M3B_ARRAYS.items():
        value=arrays.get(name)
        if value is None or tuple(value.shape)!=shape or value.dtype!=np.dtype(dtype):
            decode_failures.append(f"{sample_id}:{name}")
    metadata=json.loads(payload[".json"].decode("utf-8"))
    if split=="target_test":
        leaked=sorted(FORBIDDEN_TARGET_KEYS & set(metadata))
        if leaked or "identity_embedding" in arrays: leak_failures.append(f"{sample_id}:{leaked or 'identity_embedding'}")
