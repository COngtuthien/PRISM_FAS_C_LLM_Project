from __future__ import annotations
import io, json, os, shutil, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
import cv2, numpy as np, pyarrow as pa
from prism_fas.utils.core import atomic_json_write, git_commit, sha256_file, stable_json_hash
from .config import SPLITS
from .manifests import MANIFEST_SCHEMAS, read_manifest, write_manifest
from .model_priors import (AdaFaceBackend, ADAFACE_FILES, FACEXFORMER_FILES, FaceXFormerBackend, ModelPriorError,
                           VISIBILITY_REGIONS, compute_visibility, environment_fingerprint, fetch_backend_code,
                           load_model_config, resolve_weight, validate_identity, validate_parsing, validate_pose)
from .priors import load_prior, sha256_bytes, write_prior_atomic
from .quality import QUALITY_NAMES
from .shards import plan_shards, write_shard

M3B_PACKAGE_SCHEMA_VERSION="m3b-v1"
M3B_PACKAGE_ID="prism_data_v1_m3b"
_STATUS_COLUMNS=[("parsing_status",pa.string()),("parsing_backend",pa.string()),("parsing_model_revision",pa.string()),
    ("parsing_model_sha256",pa.string()),("pose_status",pa.string()),("pose_backend",pa.string()),("pose_convention",pa.string()),
    ("visibility_status",pa.string()),("visibility_schema_version",pa.string()),("identity_status",pa.string()),
    ("identity_backend",pa.string()),("identity_model_revision",pa.string()),("identity_model_sha256",pa.string())]
PRIORS_INDEX_M3B_SCHEMA=pa.schema([("sample_id",pa.string()),("prior_relative_path",pa.string()),("prior_sha256",pa.string()),
    ("prior_bytes",pa.int64()),("prior_schema_version",pa.string()),("quality_schema_version",pa.string()),*_STATUS_COLUMNS,
    ("source_crop_sha256",pa.string()),("preprocessing_config_hash",pa.string()),("detector_model_sha256",pa.string())])
FAILURES_SCHEMA=pa.schema([("sample_id",pa.string()),("stage",pa.string()),("error_code",pa.string()),
    ("sanitized_error_message",pa.string()),("backend",pa.string()),("recoverable",pa.bool_()),
    ("prior_schema_version",pa.string()),("parsing_model_sha256",pa.string()),("identity_model_sha256",pa.string())])
def _device(policy:str,override:str|None)->str:
    if override: return override
    try:
        import torch; return "cuda" if (policy.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    except Exception: return "cpu"
def _identity_applicable(row:dict,labels:dict[str,str],config:dict)->bool:
    return row["project_split"]==config["identity"]["applicable_project_split"] and labels.get(row["sample_id"])==config["identity"]["applicable_label"]
def _serialize(base:dict[str,np.ndarray],extra:dict[str,np.ndarray])->bytes:
    payload={**base,**extra}; buffer=io.BytesIO()
    np.savez(buffer,**{name:payload[name] for name in sorted(payload)}); return buffer.getvalue()
def build_m3b_package(input_package:Path,output_package:Path,model_config:Path,*,weight_root:Path,resume:bool=True,
                      limit_samples:int|None=None,split:str|None=None,device:str|None=None,batch_size:int|None=None,
                      dry_run:bool=False,progress:Callable[[dict],None]|None=None)->dict:
    """Build the M3B package: model-dependent priors over an immutable M3A package."""
    started=time.time()
    input_package=Path(input_package); output_package=Path(output_package)
    config=load_model_config(model_config)
    parent=json.loads((input_package/"PACKAGE_LOCK.json").read_text(encoding="utf-8"))
    if parent.get("status")!="validated": raise ModelPriorError("parent M3A package is not validated")
    samples=read_manifest(input_package/"manifests"/"samples.parquet")
    labels={}
    for name in ("source_train","source_dev"):
        for row in read_manifest(input_package/"manifests"/f"{name}.parquet"): labels[row["sample_id"]]=row["label_live_spoof"]
    if split: samples=[row for row in samples if row["project_split"]==split]
    samples=sorted(samples,key=lambda row:row["sample_id"])
    if limit_samples: samples=_stratified(samples,labels,config,limit_samples)
    applicable=[row["sample_id"] for row in samples if _identity_applicable(row,labels,config)]
    resolved_device=_device(config["runtime"]["device_policy"],device)
    plan={"samples":len(samples),"identity_applicable":len(applicable),"device":resolved_device,
          "parsing_backend":config["parsing"]["backend"],"identity_backend":config["identity"]["backend"],
          "output_package":str(output_package),"parent_package_id":parent["package_id"],"dry_run":dry_run}
    if dry_run: return {"plan":plan,"dry_run":True}
    parsing_weight=resolve_weight(config,"parsing",weight_root); identity_weight=resolve_weight(config,"identity",weight_root)
    code_root=Path(weight_root)/"code"
    fetch_backend_code(code_root/"facexformer",FACEXFORMER_FILES); fetch_backend_code(code_root/"adaface",ADAFACE_FILES)
    import torch
    torch.manual_seed(int(config["runtime"]["seed"])); torch.use_deterministic_algorithms(False)
    geometry=FaceXFormerBackend(weight_path=parsing_weight,code_root=code_root/"facexformer",device=resolved_device,
        parsing_task=config["parsing"]["task_token"],pose_task=config["pose"]["task_token"],num_classes=config["parsing"]["num_classes"])
    identity_backend=AdaFaceBackend(weight_path=identity_weight,code_root=code_root/"adaface",device=resolved_device,
        architecture=config["identity"]["architecture"],embedding_dim=config["identity"]["embedding_dim"],input_size=config["identity"]["input_size"])
    output_package.mkdir(parents=True,exist_ok=True)
    existing={row["sample_id"]:row for row in read_manifest(output_package/"manifests"/"priors_index.parquet")} if (resume and (output_package/"manifests"/"priors_index.parquet").is_file()) else {}
    status_common={"parsing_backend":config["parsing"]["backend"],"parsing_model_revision":config["parsing"]["revision"],
        "parsing_model_sha256":config["parsing"]["weight_sha256"],"pose_backend":config["pose"]["backend"],
        "pose_convention":config["pose"]["convention"],"visibility_schema_version":config["visibility_schema_version"],
        "identity_backend":config["identity"]["backend"],"identity_model_revision":config["identity"]["revision"],
        "identity_model_sha256":config["identity"]["weight_sha256"]}
    prior_rows=[];failures=[];counts={"parsing":0,"pose":0,"visibility":0,"identity":0,"reused":0,"rebuilt":0}
    batch=int(batch_size or config["runtime"]["batch_size"])
    reusable={row["sample_id"] for row in samples if _reusable(row,existing,output_package,config)}
    todo=[row for row in samples if row["sample_id"] not in reusable]
    for row in (row for row in samples if row["sample_id"] in reusable):
        prior_rows.append(existing[row["sample_id"]]); counts["reused"]+=1
        for key in ("parsing","pose","visibility"): counts[key]+=1
        if existing[row["sample_id"]]["identity_status"]=="computed": counts["identity"]+=1
    for start in range(0,len(todo),batch):
        chunk=todo[start:start+batch]
        images=[_read_image(input_package/row["image_relative_path"]) for row in chunk]
        try: geometry_results=geometry.infer(images)
        except Exception as exc:
            for row in chunk: failures.append(_failure(row,"parsing","model_inference_failed",config,str(type(exc).__name__)))
            continue
        identity_positions=[index for index,row in enumerate(chunk) if _identity_applicable(row,labels,config)]
        embeddings={}
        if identity_positions:
            try:
                vectors=identity_backend.embed([images[index] for index in identity_positions])
                embeddings={chunk[index]["sample_id"]:vectors[position] for position,index in enumerate(identity_positions)}
            except Exception as exc:
                for index in identity_positions: failures.append(_failure(chunk[index],"identity","model_inference_failed",config,str(type(exc).__name__)))
        for row,result in zip(chunk,geometry_results):
            sample_id=row["sample_id"]
            try:
                parsing=validate_parsing(result["parsing_labels"],config["parsing"]["num_classes"])
                pose=validate_pose(result["pose_ypr"])
                visibility=compute_visibility(parsing,pose,yaw_scale=float(config["visibility"]["yaw_occlusion_scale"]))
            except ModelPriorError as exc:
                failures.append(_failure(row,"parsing","invalid_model_output",config,str(exc))); continue
            base=load_prior(input_package/row["prior_relative_path"])
            extra={"parsing_labels":parsing,"pose_ypr":pose.astype(np.float32),"visibility":visibility}
            identity_status="not_applicable"
            if sample_id in embeddings:
                try: extra["identity_embedding"]=validate_identity(embeddings[sample_id],config["identity"]["embedding_dim"]); identity_status="computed"
                except ModelPriorError as exc: failures.append(_failure(row,"identity","invalid_model_output",config,str(exc))); continue
            data=_serialize(base,extra); relative=f"priors/{sample_id}.npz"
            write_prior_atomic(output_package/relative,data)
            counts["parsing"]+=1;counts["pose"]+=1;counts["visibility"]+=1;counts["rebuilt"]+=1
            if identity_status=="computed": counts["identity"]+=1
            prior_rows.append({"sample_id":sample_id,"prior_relative_path":relative,"prior_sha256":sha256_bytes(data),
                "prior_bytes":len(data),"prior_schema_version":config["prior_schema_version"],
                "quality_schema_version":row["quality_schema_version"],"parsing_status":"computed","pose_status":"computed",
                "visibility_status":"computed","identity_status":identity_status,
                "source_crop_sha256":row["crop_sha256"],"preprocessing_config_hash":row["preprocessing_config_hash"],
                "detector_model_sha256":row["detector_model_sha256"],**status_common})
        if progress: progress({"stage":"model_priors","done":min(start+batch,len(todo)),"total":len(todo),
            "parsing":counts["parsing"],"pose":counts["pose"],"visibility":counts["visibility"],"identity":counts["identity"],
            "reused":counts["reused"],"rebuilt":counts["rebuilt"],"failures":len(failures),
            "elapsed_seconds":round(time.time()-started,1),
            "samples_per_second":round(counts["rebuilt"]/max(time.time()-started,1e-6),3)})
    return _finalize(input_package,output_package,config,parent,samples,prior_rows,failures,counts,labels,
                     resolved_device,started,progress)
def _reusable(row:dict,existing:dict,output_package:Path,config:dict)->bool:
    previous=existing.get(row["sample_id"])
    if previous is None: return False
    path=output_package/previous["prior_relative_path"]
    if not path.is_file(): return False
    if previous["source_crop_sha256"]!=row["crop_sha256"]: return False
    if previous["prior_schema_version"]!=config["prior_schema_version"]: return False
    if previous["parsing_model_sha256"]!=config["parsing"]["weight_sha256"]: return False
    if previous["identity_status"]=="computed" and previous["identity_model_sha256"]!=config["identity"]["weight_sha256"]: return False
    try:
        if sha256_bytes(path.read_bytes())!=previous["prior_sha256"]: return False
        arrays=load_prior(path)
        validate_parsing(arrays["parsing_labels"],config["parsing"]["num_classes"]); validate_pose(arrays["pose_ypr"])
    except Exception: return False
    return True
def _failure(row:dict,stage:str,code:str,config:dict,message:str)->dict:
    return {"sample_id":row["sample_id"],"stage":stage,"error_code":code,
            "sanitized_error_message":str(message)[:200].replace("\\","/").split("/")[-1] if ":" in str(message) else str(message)[:200],
            "backend":config[stage if stage in config else "parsing"]["backend"],"recoverable":True,
            "prior_schema_version":config["prior_schema_version"],"parsing_model_sha256":config["parsing"]["weight_sha256"],
            "identity_model_sha256":config["identity"]["weight_sha256"]}
def _read_image(path:Path)->np.ndarray:
    image=cv2.imread(str(path))
    if image is None: raise ModelPriorError("packaged crop could not be decoded")
    return image
def _stratified(samples:list[dict],labels:dict,config:dict,limit:int)->list[dict]:
    """Smoke subset: source_train live/spoof, source_dev and target_test."""
    groups={"train_live":[],"train_spoof":[],"dev":[],"target":[]}
    for row in samples:
        if row["project_split"]=="target_test": groups["target"].append(row)
        elif row["project_split"]=="source_dev": groups["dev"].append(row)
        elif labels.get(row["sample_id"])=="live": groups["train_live"].append(row)
        else: groups["train_spoof"].append(row)
    quota={"train_live":max(8,limit//4),"train_spoof":max(8,limit//4),"dev":max(8,limit//4),"target":max(12,limit//3)}
    chosen=[]
    for name,rows in groups.items():
        datasets={}
        for row in rows: datasets.setdefault(row["dataset"],[]).append(row)
        taken=0;position=0;buckets=[datasets[key] for key in sorted(datasets)]
        while taken<quota[name] and any(position<len(b) for b in buckets):
            for bucket in buckets:
                if taken>=quota[name]: break
                if position<len(bucket): chosen.append(bucket[position]); taken+=1
            position+=1
    return sorted(chosen,key=lambda row:row["sample_id"])
def _finalize(input_package,output_package,config,parent,samples,prior_rows,failures,counts,labels,device,started,progress)->dict:
    priors={row["sample_id"]:row for row in prior_rows}
    metadata={"package_schema_version":M3B_PACKAGE_SCHEMA_VERSION,"prior_schema_version":config["prior_schema_version"],
              "parent_package_id":parent["package_id"]}
    sample_rows=[];source_rows={"source_train":[],"source_dev":[]};target_rows=[]
    for row in samples:
        prior=priors.get(row["sample_id"])
        if prior is None: continue
        image=output_package/row["image_relative_path"]
        image.parent.mkdir(parents=True,exist_ok=True)
        if not image.exists():
            try: os.link(input_package/row["image_relative_path"],image)
            except OSError: shutil.copyfile(input_package/row["image_relative_path"],image)
        if sha256_file(image)!=row["crop_sha256"]: raise ModelPriorError(f"image SHA changed for {row['sample_id']}")
        updated={**row,"package_schema_version":M3B_PACKAGE_SCHEMA_VERSION,"prior_schema_version":config["prior_schema_version"],
                 "prior_sha256":prior["prior_sha256"],"prior_bytes":prior["prior_bytes"],
                 "parsing_status":"computed","pose_status":"computed","visibility_status":"computed",
                 "identity_status":prior["identity_status"]}
        sample_rows.append(updated)
        common={"sample_id":row["sample_id"],"dataset":row["dataset"],"source_record_id":row["source_record_id"],
                "project_split":row["project_split"],"image_relative_path":row["image_relative_path"],
                "prior_relative_path":prior["prior_relative_path"],"crop_sha256":row["crop_sha256"],
                "prior_sha256":prior["prior_sha256"],"package_schema_version":M3B_PACKAGE_SCHEMA_VERSION}
        if row["project_split"]=="target_test":
            target_rows.append({**common,"source_media_type":row["source_media_type"],"frame_width":row["frame_width"],
                "frame_height":row["frame_height"],"crop_width":row["crop_width"],"crop_height":row["crop_height"],
                "detection_score":row["detection_score"],"detected_face_count":row["detected_face_count"],
                **{name:row[name] for name in QUALITY_NAMES}})
        else:
            reference=read_manifest(input_package/"manifests"/f"{row['project_split']}.parquet")
            source_rows[row["project_split"]].append({**common,"subject_id":next((r["subject_id"] for r in reference if r["sample_id"]==row["sample_id"]),None),
                "official_split":next((r["official_split"] for r in reference if r["sample_id"]==row["sample_id"]),None),
                "label_live_spoof":labels.get(row["sample_id"])})
    hashes={}
    hashes["samples"]=write_manifest(output_package/"manifests"/"samples.parquet",sample_rows,MANIFEST_SCHEMAS["samples"],metadata)
    for name in ("source_train","source_dev"):
        hashes[name]=write_manifest(output_package/"manifests"/f"{name}.parquet",source_rows[name],MANIFEST_SCHEMAS[name],metadata)
    hashes["target_test_features"]=write_manifest(output_package/"manifests"/"target_test_features.parquet",target_rows,MANIFEST_SCHEMAS["target_test_features"],metadata)
    hashes["priors_index"]=write_manifest(output_package/"manifests"/"priors_index.parquet",prior_rows,PRIORS_INDEX_M3B_SCHEMA,metadata)
    hashes["model_prior_failures"]=write_manifest(output_package/"manifests"/"model_prior_failures.parquet",failures,FAILURES_SCHEMA,metadata,sort_key="sample_id")
    sizes={row["sample_id"]:priors[row["sample_id"]]["prior_bytes"]+(output_package/row["image_relative_path"]).stat().st_size for row in sample_rows}
    shard_rows=[]
    for split_name in SPLITS:
        rows=[row for row in sample_rows if row["project_split"]==split_name]
        if not rows: continue
        lookup={row["sample_id"]:row for row in rows}
        for number,group in enumerate(plan_shards([r["sample_id"] for r in rows],sizes,max_samples=1000,max_bytes=1610612736)):
            entries=[]
            for sample_id in group:
                row=lookup[sample_id]
                entries.append((sample_id,(output_package/row["image_relative_path"]).read_bytes(),
                                (output_package/row["prior_relative_path"]).read_bytes(),
                                _shard_metadata(row,source_rows,split_name,config)))
            summary=write_shard(output_package/"shards"/f"{split_name}-{number:05d}.tar",entries)
            shard_rows.append({**summary,"split":split_name,"package_schema_version":M3B_PACKAGE_SCHEMA_VERSION})
            if progress: progress({"stage":"shards","done":len(shard_rows),"total":0})
    hashes["shards_index"]=write_manifest(output_package/"manifests"/"shards_index.parquet",shard_rows,MANIFEST_SCHEMAS["shards_index"],metadata,sort_key="shard_filename")
    lock={"package_id":M3B_PACKAGE_ID,"package_schema_version":M3B_PACKAGE_SCHEMA_VERSION,"status":"building",
        "parent_package_id":parent["package_id"],"parent_content_identity_sha256":parent["content_identity_sha256"],
        "parent_manifest_sha256":parent.get("manifest_sha256"),
        "preprocessing_version":parent.get("preprocessing_version"),"preprocessing_config_hash":parent.get("preprocessing_config_hash"),
        "detector_model_sha256":parent.get("detector_model_sha256"),"git_commit":git_commit(Path.cwd()),
        "prior_schema_version":config["prior_schema_version"],"visibility_schema_version":config["visibility_schema_version"],
        "models":{"parsing":{"backend":config["parsing"]["backend"],"revision":config["parsing"]["revision"],"weight_sha256":config["parsing"]["weight_sha256"],"num_classes":config["parsing"]["num_classes"]},
                  "pose":{"backend":config["pose"]["backend"],"revision":config["pose"]["revision"],"convention":config["pose"]["convention"]},
                  "visibility":{"backend":config["visibility"]["backend"],"region_order":list(VISIBILITY_REGIONS)},
                  "identity":{"backend":config["identity"]["backend"],"revision":config["identity"]["revision"],"weight_sha256":config["identity"]["weight_sha256"],"embedding_dim":config["identity"]["embedding_dim"]}},
        "environment":environment_fingerprint(device),
        "manifest_sha256":dict(sorted(hashes.items())),
        "shards":[{"shard_filename":r["shard_filename"],"split":r["split"],"sha256":r["sha256"],"row_count":r["row_count"],"byte_size":r["byte_size"]} for r in sorted(shard_rows,key=lambda r:r["shard_filename"])],
        "total_samples":len(sample_rows),
        "per_split_counts":{name:sum(1 for r in sample_rows if r["project_split"]==name) for name in SPLITS},
        "per_dataset_counts":{name:sum(1 for r in sample_rows if r["dataset"]==name) for name in sorted({r["dataset"] for r in sample_rows})},
        "prior_counts":{"parsing_computed":counts["parsing"],"pose_computed":counts["pose"],"visibility_computed":counts["visibility"],
            "identity_computed":counts["identity"],"identity_not_applicable":len(sample_rows)-counts["identity"]},
        "model_prior_failures":len(failures),
        "target_isolation":{"policy":"feature_only_no_labels_no_identity","status":"pending"},
        "package_validation":{"status":"pending","checks_passed":None,"checks_total":None}}
    lock["content_identity_sha256"]=stable_json_hash({k:v for k,v in lock.items() if k not in {"created_at","git_commit","environment"}})
    lock["created_at"]=datetime.now(timezone.utc).isoformat()
    lock["build_seconds"]=round(time.time()-started,1)
    atomic_json_write(output_package/"PACKAGE_LOCK.json",lock)
    return {"lock":lock,"counts":counts,"failures":failures,"hashes":hashes,"shards":shard_rows,"samples":sample_rows,"device":device}
def _shard_metadata(row:dict,source_rows:dict,split_name:str,config:dict)->dict:
    payload={key:row[key] for key in ("sample_id","dataset","project_split","source_record_id","requested_frame_index",
        "actual_frame_index","crop_sha256","prior_sha256","source_media_type","image_format","frame_width","frame_height",
        "crop_width","crop_height","detection_score","detected_face_count",*QUALITY_NAMES,"quality_schema_version",
        "prior_schema_version","package_schema_version","parsing_status","pose_status","visibility_status","identity_status")}
    payload["pose_convention"]=config["pose"]["convention"]; payload["visibility_regions"]=list(VISIBILITY_REGIONS)
    if split_name!="target_test":
        entry=next(e for e in source_rows[split_name] if e["sample_id"]==row["sample_id"])
        payload["label_live_spoof"]=entry["label_live_spoof"]; payload["official_split"]=entry["official_split"]
    return payload
