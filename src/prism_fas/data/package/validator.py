from __future__ import annotations
import json, re
from pathlib import Path
from typing import Any
from prism_fas.data.manifests.leakage import FORBIDDEN, find_target_leakage
from prism_fas.utils.core import sha256_file
from .config import SPLITS
from .manifests import read_manifest
from .priors import PRIOR_ARRAYS, load_prior, validate_prior_arrays
from .quality import QUALITY_NAMES
from .shards import read_shard_members

PRIVATE_TARGET_TOKENS=("live","spoof","attack","taxonomy","subject","session","genuine","replay","mask","paper","makeup",".mov",".avi",".mp4")
def _check(checks:list,ident:str,category:str,description:str,expected:Any,actual:Any,severity:str="error")->bool:
    ok=expected==actual; checks.append({"check_id":ident,"category":category,"description":description,
        "expected":str(expected)[:200],"actual":str(actual)[:200],"passed":ok,"severity":severity}); return ok
M3B_ARRAYS={"parsing_labels":((224,224),"uint8"),"pose_ypr":((3,),"float32"),"visibility":((9,),"float16")}
def _validate_model_priors(root:Path,lock:dict,samples:list,manifests:dict,checks:list,parent_package:Path|None)->dict:
    """M3B-specific checks: parsing/pose/visibility everywhere, identity only where applicable."""
    import numpy as np
    index={row["sample_id"]:row for row in manifests.get("priors_index",[])}
    labels={}
    for name in ("source_train","source_dev"):
        for row in manifests.get(name,[]): labels[row["sample_id"]]=row["label_live_spoof"]
    counts={"parsing":0,"pose":0,"visibility":0,"identity":0,"identity_not_applicable":0}
    bad_shape=[];bad_pose=[];bad_visibility=[];bad_identity=[];unexpected_identity=[];missing_identity=[]
    applicable=set()
    for row in samples:
        sample_id=row["sample_id"]
        if row["project_split"]=="source_train" and labels.get(sample_id)=="live": applicable.add(sample_id)
        try: arrays=load_prior(root/row["prior_relative_path"])
        except Exception: bad_shape.append(sample_id); continue
        for name,(shape,dtype) in M3B_ARRAYS.items():
            value=arrays.get(name)
            if value is None or tuple(value.shape)!=shape or value.dtype!=np.dtype(dtype): bad_shape.append(sample_id); break
        else:
            counts["parsing"]+=1
            pose=arrays["pose_ypr"]
            if not np.isfinite(pose).all(): bad_pose.append(sample_id)
            else: counts["pose"]+=1
            visibility=arrays["visibility"].astype(np.float32)
            if not (np.isfinite(visibility).all() and (visibility>=0).all() and (visibility<=1).all()): bad_visibility.append(sample_id)
            else: counts["visibility"]+=1
            if int(arrays["parsing_labels"].max(initial=0))>=lock["models"]["parsing"]["num_classes"]: bad_shape.append(sample_id)
        has_identity="identity_embedding" in arrays
        if has_identity:
            vector=arrays["identity_embedding"].astype(np.float32)
            if sample_id not in applicable: unexpected_identity.append(sample_id)
            elif vector.shape!=(lock["models"]["identity"]["embedding_dim"],) or not np.isfinite(vector).all() or not (0.5<=float(np.linalg.norm(vector))<=1.5): bad_identity.append(sample_id)
            else: counts["identity"]+=1
        else:
            counts["identity_not_applicable"]+=1
            if sample_id in applicable: missing_identity.append(sample_id)
        entry=index.get(sample_id)
        if entry is not None:
            expected="computed" if has_identity else "not_applicable"
            if entry["identity_status"]!=expected: bad_identity.append(sample_id)
    total=len(samples)
    _check(checks,"m3b.parsing","model_priors","parsing prior for every sample",total,counts["parsing"])
    _check(checks,"m3b.pose","model_priors","pose prior for every sample",total,counts["pose"])
    _check(checks,"m3b.visibility","model_priors","visibility prior for every sample",total,counts["visibility"])
    _check(checks,"m3b.arrays","model_priors","prior arrays have expected shape/dtype",[],sorted(set(bad_shape))[:20])
    _check(checks,"m3b.pose_finite","model_priors","pose values finite",[],bad_pose[:20])
    _check(checks,"m3b.visibility_range","model_priors","visibility within [0,1]",[],bad_visibility[:20])
    _check(checks,"m3b.identity_count","model_priors","identity embeddings match applicable samples",len(applicable),counts["identity"])
    _check(checks,"m3b.identity_scope","model_priors","identity only on source_train live",[],sorted(set(unexpected_identity))[:20])
    _check(checks,"m3b.identity_missing","model_priors","every applicable sample has an identity embedding",[],missing_identity[:20])
    _check(checks,"m3b.identity_valid","model_priors","identity embeddings valid shape/norm",[],sorted(set(bad_identity))[:20])
    target_identity=[row["sample_id"] for row in samples if row["project_split"]=="target_test" and "identity_embedding" in load_prior(root/row["prior_relative_path"])] if total<=64 else []
    _check(checks,"m3b.target_identity","target_isolation","no identity embedding on target samples",[],target_identity)
    revisions={row["parsing_model_sha256"] for row in manifests.get("priors_index",[])}|{lock["models"]["parsing"]["weight_sha256"]}
    _check(checks,"m3b.model_pins","model_priors","parsing model SHA consistent across priors",1,len(revisions))
    failures=manifests.get("model_prior_failures",[])
    _check(checks,"m3b.failures","model_priors","no unresolved model prior failures",0,len(failures))
    if parent_package is not None:
        parent=json.loads((Path(parent_package)/"PACKAGE_LOCK.json").read_text(encoding="utf-8"))
        _check(checks,"m3b.parent_identity","lock","parent content identity matches",parent["content_identity_sha256"],lock.get("parent_content_identity_sha256"))
        _check(checks,"m3b.parent_id","lock","parent package id matches",parent["package_id"],lock.get("parent_package_id"))
        parent_images={row["sample_id"]:row["crop_sha256"] for row in read_manifest(Path(parent_package)/"manifests"/"samples.parquet")}
        changed=[row["sample_id"] for row in samples if parent_images.get(row["sample_id"])!=row["crop_sha256"]]
        _check(checks,"m3b.image_sha_stable","lock","image SHA unchanged from parent package",[],changed[:20])
    return {"prior_counts":counts,"applicable_identity":len(applicable),
            "model_revisions":{key:value.get("revision") for key,value in lock.get("models",{}).items()}}
def validate_package(package_root:Path,*,require_validated_status:bool=True,parent_package:Path|None=None)->dict:
    """Structural, integrity and target-isolation validation of an M3A package.

    The build runs this once with require_validated_status=False (the lock is
    still `building` at that point), promotes the lock, then validates again in
    the default mode where `validated` is required.
    """
    root=Path(package_root); checks:list[dict]=[]; counts:dict[str,int]={}
    lock_path=root/"PACKAGE_LOCK.json"
    if not lock_path.is_file():
        _check(checks,"lock.exists","lock","PACKAGE_LOCK.json exists",True,False)
        return {"passed":False,"errors":[c for c in checks if not c["passed"]],"checks":checks,"counts":counts,"package_id":None,"target_isolation":{"passed":False}}
    try: lock=json.loads(lock_path.read_text(encoding="utf-8")); _check(checks,"lock.parses","lock","PACKAGE_LOCK parses",True,True)
    except json.JSONDecodeError as exc:
        _check(checks,"lock.parses","lock","PACKAGE_LOCK parses",True,False)
        return {"passed":False,"errors":[c for c in checks if not c["passed"]],"checks":checks,"counts":counts,"package_id":None,"target_isolation":{"passed":False}}
    if require_validated_status: _check(checks,"lock.status","lock","status is validated","validated",lock.get("status"))
    else: _check(checks,"lock.status","lock","status is building or validated",True,lock.get("status") in {"building","validated"})
    manifests={name:read_manifest(root/"manifests"/f"{name}.parquet") for name in ("samples","source_train","source_dev","target_test_features","priors_index","shards_index","model_prior_failures") if (root/"manifests"/f"{name}.parquet").is_file()}
    for name in ("samples","source_train","source_dev","target_test_features","priors_index","shards_index"):
        _check(checks,f"manifest.{name}","manifests","manifest present",True,name in manifests)
    if "samples" not in manifests:
        return {"passed":False,"errors":[c for c in checks if not c["passed"]],"checks":checks,"counts":counts,"package_id":lock.get("package_id"),"target_isolation":{"passed":False}}
    samples=manifests["samples"]; counts["samples"]=len(samples)
    versions={row["package_schema_version"] for row in samples}|{lock.get("package_schema_version")}
    _check(checks,"schema.versions","schema","package schema versions agree",1,len(versions))
    _check(checks,"count.samples","counts","lock total matches samples manifest",lock.get("total_samples"),len(samples))
    ids=[row["sample_id"] for row in samples]
    _check(checks,"identity.duplicates","identity","unique sample IDs",len(ids),len(set(ids)))
    split_ids={"source_train":{r["sample_id"] for r in manifests.get("source_train",[])},
               "source_dev":{r["sample_id"] for r in manifests.get("source_dev",[])},
               "target_test":{r["sample_id"] for r in manifests.get("target_test_features",[])}}
    for split in SPLITS: counts[split]=len(split_ids[split])
    pairs=[("source_train","source_dev"),("source_train","target_test"),("source_dev","target_test")]
    _check(checks,"splits.disjoint","splits","split manifests are disjoint",[],sorted({s for a,b in pairs for s in split_ids[a]&split_ids[b]}))
    union=set().union(*split_ids.values())
    _check(checks,"splits.union","splits","split union equals samples manifest",sorted(set(ids)),sorted(union))
    for split in SPLITS:
        declared={row["sample_id"] for row in samples if row["project_split"]==split}
        _check(checks,f"splits.assignment.{split}","splits","samples project_split matches manifest",sorted(declared),sorted(split_ids[split]))
    missing_images=[];missing_priors=[];image_sha=[];prior_sha=[];bad_npz=[];bad_metrics=[]
    referenced=set()
    for row in samples:
        image=root/row["image_relative_path"]; prior=root/row["prior_relative_path"]
        referenced.add(image.resolve()); referenced.add(prior.resolve())
        for relative in (row["image_relative_path"],row["prior_relative_path"]):
            if relative.startswith("/") or ".." in Path(relative).parts or re.match(r"^[A-Za-z]:",relative): bad_npz.append(row["sample_id"])
        if not image.is_file(): missing_images.append(row["sample_id"])
        elif sha256_file(image)!=row["crop_sha256"]: image_sha.append(row["sample_id"])
        if not prior.is_file(): missing_priors.append(row["sample_id"]); continue
        if sha256_file(prior)!=row["prior_sha256"]: prior_sha.append(row["sample_id"])
        try:
            arrays=load_prior(prior); validate_prior_arrays(arrays)
            if not all(isinstance(row[name],float) and row[name]==row[name] and abs(row[name])!=float("inf") for name in QUALITY_NAMES): bad_metrics.append(row["sample_id"])
        except Exception: bad_npz.append(row["sample_id"])
    _check(checks,"artifacts.images","artifacts","every image exists",[],missing_images[:20])
    _check(checks,"artifacts.priors","artifacts","every prior exists",[],missing_priors[:20])
    _check(checks,"artifacts.image_sha","artifacts","image SHA matches manifest",[],image_sha[:20])
    _check(checks,"artifacts.prior_sha","artifacts","prior SHA matches manifest",[],prior_sha[:20])
    _check(checks,"artifacts.npz","artifacts","priors load with allow_pickle=False and expected shapes",[],bad_npz[:20])
    _check(checks,"artifacts.metrics","artifacts","quality metrics are finite",[],bad_metrics[:20])
    on_disk={p.resolve() for folder in ("images","priors") for p in (root/folder).rglob("*") if p.is_file()}
    _check(checks,"artifacts.orphans","artifacts","no orphan images or priors",[],sorted(p.name for p in on_disk-referenced)[:20])
    counts["images"]=len([p for p in (root/"images").rglob("*") if p.is_file()]) if (root/"images").is_dir() else 0
    counts["priors"]=len([p for p in (root/"priors").rglob("*") if p.is_file()]) if (root/"priors").is_dir() else 0
    shard_rows=manifests.get("shards_index",[]); shard_total=0; shard_problems=[];mixed=[];incomplete=[];json_entries=0
    by_split={split:set() for split in SPLITS}
    for row in shard_rows:
        path=root/"shards"/row["shard_filename"]
        if not path.is_file(): shard_problems.append(row["shard_filename"]); continue
        if sha256_file(path)!=row["sha256"] or path.stat().st_size!=row["byte_size"]: shard_problems.append(row["shard_filename"]); continue
        try: members=read_shard_members(path)
        except ValueError: shard_problems.append(row["shard_filename"]); continue
        if len(members)!=row["row_count"]: shard_problems.append(row["shard_filename"])
        shard_total+=row["row_count"]
        for stem,suffixes in members.items():
            if suffixes!={".jpg",".npz",".json"}: incomplete.append(stem)
            else: json_entries+=1
            if stem not in split_ids[row["split"]]: mixed.append(stem)
            by_split[row["split"]].add(stem)
    _check(checks,"shards.files","shards","shard files exist with matching SHA/size/rows",[],shard_problems[:20])
    _check(checks,"shards.rows","shards","shard row counts reconcile with samples",len(samples),shard_total)
    _check(checks,"shards.triplets","shards","every shard member is a jpg+npz+json triplet",[],incomplete[:20])
    _check(checks,"shards.split_isolation","shards","no split mixed within a shard",[],mixed[:20])
    for split in SPLITS: _check(checks,f"shards.membership.{split}","shards","shard IDs equal split manifest IDs",sorted(split_ids[split]),sorted(by_split[split]))
    counts["shards"]=len(shard_rows); counts["shard_json_entries"]=json_entries
    target_rows=manifests.get("target_test_features",[])
    leaks=find_target_leakage(target_rows)
    forbidden_fields=sorted({key for row in target_rows for key in row} & FORBIDDEN)
    tokens=sorted({token for row in target_rows for value in row.values() for token in PRIVATE_TARGET_TOKENS if token in str(value).lower()})
    sample_target_tokens=sorted({token for row in samples if row["project_split"]=="target_test" for value in row.values() for token in PRIVATE_TARGET_TOKENS if token in str(value).lower()})
    _check(checks,"target.leakage","target_isolation","no forbidden target metadata",[],leaks)
    _check(checks,"target.fields","target_isolation","no forbidden target field names",[],forbidden_fields)
    _check(checks,"target.tokens","target_isolation","no private target tokens in values",[],tokens+sample_target_tokens)
    _check(checks,"target.samples_manifest","target_isolation","samples manifest target rows pass leakage scan",[],find_target_leakage([r for r in samples if r["project_split"]=="target_test"]))
    raw_paths=sorted({str(value) for row in samples for value in row.values() if isinstance(value,str) and (re.match(r"^[A-Za-z]:[\\/]",value) or value.startswith("/") or "\\" in value)})
    _check(checks,"paths.raw","paths","no raw or absolute paths in manifests",[],raw_paths[:10])
    for name,digest in (lock.get("manifest_sha256") or {}).items():
        path=root/"manifests"/f"{name}.parquet"
        _check(checks,f"lock.hash.{name}","lock","lock manifest hash matches artifact",digest,sha256_file(path) if path.is_file() else None)
    legacy=sorted(str(p.relative_to(root)) for p in list(root.rglob("*.jsonl"))+list(root.rglob("m2a")))
    _check(checks,"artifacts.legacy","artifacts","no M2A/legacy JSONL artifacts",[],legacy)
    model_priors={}
    if lock.get("package_schema_version","").startswith("m3b"):
        model_priors=_validate_model_priors(root,lock,samples,manifests,checks,parent_package)
    passed=all(c["passed"] for c in checks)
    return {"passed":passed,"errors":[c for c in checks if not c["passed"]],"checks":checks,"counts":counts,
            "package_id":lock.get("package_id"),"parent_package_id":lock.get("parent_package_id"),
            "prior_counts":model_priors.get("prior_counts",{}),"model_revisions":model_priors.get("model_revisions",{}),
            "target_isolation":{"passed":not (leaks or forbidden_fields or tokens or sample_target_tokens),"matches":leaks,"token_matches":tokens+sample_target_tokens}}
def validate_source_m2_hashes(package_root:Path,input_root:Path)->bool:
    """Confirm the frozen M2 input still matches the hashes recorded in the lock."""
    lock=json.loads((Path(package_root)/"PACKAGE_LOCK.json").read_text(encoding="utf-8"))
    manifests=Path(input_root)/"manifests"
    return all(sha256_file(manifests/f"{name}.parquet")==digest for name,digest in lock["source_m2_manifest_sha256"].items())
