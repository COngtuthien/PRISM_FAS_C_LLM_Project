from __future__ import annotations
import json, os, shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from prism_fas.utils.core import atomic_json_write, git_commit, sha256_file, stable_json_hash
from .config import M3APackageConfig, SPLITS, project_split
from .manifests import MANIFEST_SCHEMAS, read_manifest, write_manifest
from .priors import PriorBuildError, load_prior, prior_payload, prior_schema_version, prior_status, read_crop, serialize_prior, sha256_bytes, validate_prior_arrays, write_prior_atomic
from .quality import QUALITY_NAMES, quality_schema_version
from .shards import plan_shards, write_shard

M2_MANIFESTS=("source_frames","source_crops","target_frames","target_crops","preprocessing_failures")
@dataclass
class BuildStats:
    samples:int=0; priors_built:int=0; priors_reused:int=0; images_linked:int=0; shards:int=0
    per_dataset:dict[str,int]=field(default_factory=dict); per_split:dict[str,int]=field(default_factory=dict)
def load_m2_samples(input_root:Path)->list[dict]:
    """Join M2 frame and crop manifests into one successful-sample view."""
    manifests=Path(input_root)/"manifests"; joined=[]
    for role in ("source","target"):
        frames={row["sample_id"]:row for row in read_manifest(manifests/f"{role}_frames.parquet")}
        crops={row["sample_id"]:row for row in read_manifest(manifests/f"{role}_crops.parquet")}
        if set(frames)!=set(crops): raise ValueError(f"{role} frame/crop sample sets differ in M2 input")
        for sample_id in sorted(frames):
            frame,crop=frames[sample_id],crops[sample_id]
            joined.append({**crop,"dataset_role":role,"official_split":frame.get("official_split"),
                           "subject_id":frame.get("subject_id"),"label_live_spoof":frame.get("label_live_spoof"),
                           "source_relative_identifier":frame["source_relative_identifier"]})
    return joined
def _relative(sample_id:str,suffix:str,folder:str)->str: return f"{folder}/{sample_id}{suffix}"
def _link_or_copy(source:Path,destination:Path)->None:
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists(): destination.unlink()
    try: os.link(source,destination)
    except OSError: shutil.copyfile(source,destination)
def build_priors(samples:list[dict],*,input_root:Path,package_root:Path,config:M3APackageConfig,resume:bool=True,progress:Callable[[int,int],None]|None=None)->tuple[list[dict],BuildStats]:
    """Create one NPZ prior per sample; valid priors are reused when resuming."""
    stats=BuildStats(); rows=[]; existing={}
    index=package_root/"manifests"/"priors_index.parquet"
    if resume and index.exists(): existing={row["sample_id"]:row for row in read_manifest(index)}
    for position,sample in enumerate(samples,1):
        sample_id=sample["sample_id"]; relative=_relative(sample_id,".npz","priors"); target=package_root/relative
        crop=Path(input_root)/sample["crop_relative_path"]
        previous=existing.get(sample_id); reused=False
        if resume and previous is not None and target.is_file():
            compatible=(previous["crop_sha256"]==sample["crop_sha256"] and previous["prior_schema_version"]==prior_schema_version()
                        and previous["quality_schema_version"]==quality_schema_version()
                        and previous["preprocessing_config_hash"]==sample["preprocessing_config_hash"]
                        and previous["detector_model_sha256"]==sample["detector_model_sha256"])
            if compatible:
                try:
                    arrays=load_prior(target); validate_prior_arrays(arrays)
                    if sha256_bytes(target.read_bytes())==previous["prior_sha256"]: reused=True
                except (PriorBuildError,ValueError,OSError): reused=False
        if reused:
            rows.append(previous); stats.priors_reused+=1
        else:
            payload,_=prior_payload(sample,read_crop(crop)); data=serialize_prior(payload)
            write_prior_atomic(target,data); stats.priors_built+=1
            rows.append({"sample_id":sample_id,"prior_relative_path":relative,"prior_sha256":sha256_bytes(data),
                         "prior_bytes":len(data),"prior_schema_version":prior_schema_version(),
                         "quality_schema_version":quality_schema_version(),"crop_sha256":sample["crop_sha256"],
                         "preprocessing_version":sample["preprocessing_version"],
                         "preprocessing_config_hash":sample["preprocessing_config_hash"],
                         "detector_model_sha256":sample["detector_model_sha256"],**prior_status()})
        if progress and (position==1 or position%500==0 or position==len(samples)): progress(position,len(samples))
    stats.samples=len(samples)
    return rows,stats
def _sample_row(sample:dict,prior:dict,metrics:dict[str,float],config:M3APackageConfig)->dict:
    return {"sample_id":sample["sample_id"],"dataset":sample["dataset"],"dataset_role":sample["dataset_role"],
            "project_split":project_split(sample["dataset_role"],sample.get("official_split")),
            "source_record_id":sample["source_record_id"],"requested_frame_index":sample["requested_frame_index"],
            "actual_frame_index":sample["actual_frame_index"],
            "image_relative_path":_relative(sample["sample_id"],config.image_extension,"images"),
            "crop_sha256":sample["crop_sha256"],"prior_relative_path":prior["prior_relative_path"],
            "prior_sha256":prior["prior_sha256"],"prior_bytes":prior["prior_bytes"],
            "source_media_type":sample["source_media_type"],"image_format":config.image_format,
            "frame_width":sample["frame_width"],"frame_height":sample["frame_height"],
            "crop_width":sample["crop_width"],"crop_height":sample["crop_height"],
            "detection_score":sample["detection_score"],"detected_face_count":sample["detected_face_count"],
            **metrics,"quality_schema_version":quality_schema_version(),"prior_schema_version":prior_schema_version(),
            "package_schema_version":config.package_schema_version,"preprocessing_version":sample["preprocessing_version"],
            "preprocessing_config_hash":sample["preprocessing_config_hash"],
            "detector_model_sha256":sample["detector_model_sha256"],**prior_status()}
def select_samples(samples:list[dict],limit_per_dataset:int|None)->list[dict]:
    """Deterministic per-dataset subset used by smoke builds.

    Stratified round-robin over the dataset's project splits and records, so a
    smoke package still exercises every split the dataset actually has.
    """
    if limit_per_dataset is None: return samples
    grouped:dict[str,dict[str,list[dict]]]={}
    for sample in sorted(samples,key=lambda s:(s["dataset"],s["source_record_id"],s["sample_id"])):
        split=project_split(sample["dataset_role"],sample.get("official_split"))
        grouped.setdefault(sample["dataset"],{}).setdefault(split,[]).append(sample)
    chosen=[]
    for dataset in sorted(grouped):
        buckets=[grouped[dataset][split] for split in sorted(grouped[dataset])]
        taken=0;position=0
        while taken<limit_per_dataset and any(position<len(bucket) for bucket in buckets):
            for bucket in buckets:
                if taken>=limit_per_dataset: break
                if position<len(bucket): chosen.append(bucket[position]); taken+=1
            position+=1
    return sorted(chosen,key=lambda s:s["sample_id"])
def build_package(input_root:Path,package_root:Path,config:M3APackageConfig,*,resume:bool=True,limit_per_dataset:int|None=None,progress:Callable[[str,int,int],None]|None=None)->dict:
    """Build priors, images, manifests, shards and PACKAGE_LOCK for an M2 run."""
    input_root=Path(input_root); package_root=Path(package_root); package_root.mkdir(parents=True,exist_ok=True)
    samples=select_samples(load_m2_samples(input_root),limit_per_dataset)
    failures_path=Path(input_root)/"manifests"/"preprocessing_failures.parquet"
    m2_failures=[{k:row[k] for k in ("dataset","error_code","stage","recoverable")} for row in read_manifest(failures_path)] if failures_path.is_file() else []
    prior_rows,stats=build_priors(samples,input_root=input_root,package_root=package_root,config=config,resume=resume,
                                  progress=(lambda done,total: progress("priors",done,total)) if progress else None)
    priors={row["sample_id"]:row for row in prior_rows}
    metadata={"package_schema_version":config.package_schema_version,"prior_schema_version":prior_schema_version(),
              "quality_schema_version":quality_schema_version(),"package_config_hash":config.config_hash}
    sample_rows=[];source_rows={"source_train":[],"source_dev":[]};target_rows=[];images={}
    for sample in samples:
        sample_id=sample["sample_id"]; prior=priors[sample_id]
        arrays=load_prior(package_root/prior["prior_relative_path"]); validate_prior_arrays(arrays)
        metrics={name:float(value) for name,value in zip(QUALITY_NAMES,arrays["quality_vector"].tolist())}
        row=_sample_row(sample,prior,metrics,config); sample_rows.append(row)
        image=package_root/row["image_relative_path"]; _link_or_copy(input_root/sample["crop_relative_path"],image)
        if sha256_file(image)!=sample["crop_sha256"]: raise ValueError(f"packaged image SHA changed for {sample_id}")
        stats.images_linked+=1; images[sample_id]=image
        stats.per_dataset[sample["dataset"]]=stats.per_dataset.get(sample["dataset"],0)+1
        stats.per_split[row["project_split"]]=stats.per_split.get(row["project_split"],0)+1
        common={"sample_id":sample_id,"dataset":sample["dataset"],"source_record_id":sample["source_record_id"],
                "project_split":row["project_split"],"image_relative_path":row["image_relative_path"],
                "prior_relative_path":prior["prior_relative_path"],"crop_sha256":sample["crop_sha256"],
                "prior_sha256":prior["prior_sha256"],"package_schema_version":config.package_schema_version}
        if sample["dataset_role"]=="source":
            source_rows[row["project_split"]].append({**common,"subject_id":sample.get("subject_id"),
                "official_split":sample.get("official_split"),"label_live_spoof":sample.get("label_live_spoof")})
        else:
            target_rows.append({**common,"source_media_type":sample["source_media_type"],
                "frame_width":sample["frame_width"],"frame_height":sample["frame_height"],
                "crop_width":sample["crop_width"],"crop_height":sample["crop_height"],
                "detection_score":sample["detection_score"],"detected_face_count":sample["detected_face_count"],**metrics})
    hashes={}
    hashes["samples"]=write_manifest(package_root/"manifests"/"samples.parquet",sample_rows,MANIFEST_SCHEMAS["samples"],metadata)
    hashes["source_train"]=write_manifest(package_root/"manifests"/"source_train.parquet",source_rows["source_train"],MANIFEST_SCHEMAS["source_train"],metadata)
    hashes["source_dev"]=write_manifest(package_root/"manifests"/"source_dev.parquet",source_rows["source_dev"],MANIFEST_SCHEMAS["source_dev"],metadata)
    hashes["target_test_features"]=write_manifest(package_root/"manifests"/"target_test_features.parquet",target_rows,MANIFEST_SCHEMAS["target_test_features"],metadata)
    hashes["priors_index"]=write_manifest(package_root/"manifests"/"priors_index.parquet",prior_rows,MANIFEST_SCHEMAS["priors_index"],metadata)
    by_split={split:[row for row in sample_rows if row["project_split"]==split] for split in SPLITS}
    sizes={row["sample_id"]:row["prior_bytes"]+images[row["sample_id"]].stat().st_size for row in sample_rows}
    shard_rows=[]
    for split in SPLITS:
        ids=[row["sample_id"] for row in by_split[split]]
        if not ids: continue
        lookup={row["sample_id"]:row for row in by_split[split]}
        for number,group in enumerate(plan_shards(ids,sizes,max_samples=config.shard_max_samples,max_bytes=config.shard_max_bytes)):
            entries=[]
            for sample_id in group:
                row=lookup[sample_id]
                entries.append((sample_id,(package_root/row["image_relative_path"]).read_bytes(),
                                (package_root/row["prior_relative_path"]).read_bytes(),_shard_metadata(row,source_rows,split)))
            summary=write_shard(package_root/"shards"/f"{split}-{number:05d}.tar",entries)
            shard_rows.append({**summary,"split":split,"package_schema_version":config.package_schema_version})
            stats.shards+=1
            if progress: progress("shards",stats.shards,0)
    hashes["shards_index"]=write_manifest(package_root/"manifests"/"shards_index.parquet",shard_rows,MANIFEST_SCHEMAS["shards_index"],metadata,sort_key="shard_filename")
    lock=build_lock(input_root,package_root,config,sample_rows,shard_rows,hashes,stats)
    return {"stats":stats,"hashes":hashes,"lock":lock,"shards":shard_rows,"samples":sample_rows,"m2_failures":m2_failures}
def _shard_metadata(row:dict,source_rows:dict,split:str)->dict:
    """Package-safe per-sample JSON. Target entries never carry labels."""
    payload={key:row[key] for key in ("sample_id","dataset","project_split","source_record_id","requested_frame_index",
        "actual_frame_index","crop_sha256","prior_sha256","source_media_type","image_format","frame_width","frame_height",
        "crop_width","crop_height","detection_score","detected_face_count",*QUALITY_NAMES,"quality_schema_version",
        "prior_schema_version","package_schema_version","parsing_status","pose_status","visibility_status","identity_status")}
    if split!="target_test":
        labels={entry["sample_id"]:entry for entry in source_rows[split]}
        entry=labels[row["sample_id"]]; payload["label_live_spoof"]=entry["label_live_spoof"]; payload["official_split"]=entry["official_split"]
    return payload
def build_lock(input_root:Path,package_root:Path,config:M3APackageConfig,sample_rows:list[dict],shard_rows:list[dict],hashes:dict,stats:BuildStats,*,status:str="building")->dict:
    manifests=Path(input_root)/"manifests"
    lock={"package_id":f"{config.package_id_prefix}",
          "package_schema_version":config.package_schema_version,"status":status,
          "source_m2_namespace":Path(input_root).name,
          "source_m2_validation_profile":"full_preprocessing","source_m2_validation_version":"m2f1a-full-v1",
          "source_m2_manifest_sha256":{name:sha256_file(manifests/f"{name}.parquet") for name in M2_MANIFESTS if (manifests/f"{name}.parquet").exists()},
          "source_m2_crop_count":len(sample_rows),
          "preprocessing_version":sample_rows[0]["preprocessing_version"] if sample_rows else None,
          "preprocessing_config_hash":sample_rows[0]["preprocessing_config_hash"] if sample_rows else None,
          "detector_model_sha256":sample_rows[0]["detector_model_sha256"] if sample_rows else None,
          "adapter_versions":{"casia_fasd":"1.0","msu_mfsd":"1.0","siw_mv2":"1.0"},
          "git_commit":git_commit(Path.cwd()),
          "prior_schema_version":prior_schema_version(),"quality_schema_version":quality_schema_version(),
          "prior_config_hash":config.config_hash,
          "manifest_sha256":dict(sorted(hashes.items())),
          "shards":[{"shard_filename":row["shard_filename"],"split":row["split"],"sha256":row["sha256"],"row_count":row["row_count"],"byte_size":row["byte_size"]} for row in sorted(shard_rows,key=lambda r:r["shard_filename"])],
          "total_samples":len(sample_rows),
          "per_dataset_counts":dict(sorted(stats.per_dataset.items())),
          "per_split_counts":dict(sorted(stats.per_split.items())),
          "deferred_priors":prior_status(),
          "target_isolation":{"policy":config.target_isolation_policy,"status":"pending"},
          "package_validation":{"status":"pending","checks_passed":None,"checks_total":None}}
    lock["content_identity_sha256"]=stable_json_hash({k:v for k,v in lock.items() if k not in {"created_at","git_commit"}})
    lock["created_at"]=datetime.now(timezone.utc).isoformat()
    atomic_json_write(package_root/"PACKAGE_LOCK.json",lock)
    return lock
def finalize_lock(package_root:Path,report:dict)->dict:
    """Promote the lock to validated once the package validator has passed."""
    path=Path(package_root)/"PACKAGE_LOCK.json"; lock=json.loads(path.read_text(encoding="utf-8"))
    if not report["passed"]: raise ValueError("cannot mark package validated: validation failed")
    lock["status"]="validated"
    lock["target_isolation"]={**lock["target_isolation"],"status":"passed" if report["target_isolation"]["passed"] else "failed"}
    lock["package_validation"]={"status":"passed","checks_passed":sum(1 for c in report["checks"] if c["passed"]),"checks_total":len(report["checks"])}
    lock["content_identity_sha256"]=stable_json_hash({k:v for k,v in lock.items() if k not in {"created_at","git_commit","content_identity_sha256"}})
    atomic_json_write(path,lock); return lock
