from __future__ import annotations
import hashlib, json, subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import cv2, yaml
import pyarrow.parquet as pq
from prism_fas.config.models import DatasetDefinition, load_paths
from prism_fas.data.adapters import adapter_for
from prism_fas.data.preprocess_m2 import load_m2_config, resolve_detector_path
from prism_fas.data.manifests.leakage import find_target_leakage
from prism_fas.data.manifests.parquet_writer import write_parquet_atomic
from prism_fas.data.manifests.schemas import MODELS
from prism_fas.utils.core import sha256_file, atomic_json_write

DATASETS=("casia_fasd","msu_mfsd","siw_mv2")
def _read(path: Path) -> list[dict[str,Any]]:
    if not path.is_file(): raise FileNotFoundError(path)
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
def _git(root: Path) -> str:
    try: return subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: return "unknown"
def _definition(root: Path, name: str) -> DatasetDefinition:
    return DatasetDefinition.model_validate(yaml.safe_load((root/"configs"/"data"/f"{name}.yaml").read_text(encoding="utf-8")))
def _canonical(paths: Any, root: Path) -> dict[tuple[str,str],Any]:
    result={}
    for name in DATASETS:
        definition=_definition(root,name); records=adapter_for(definition,getattr(paths.raw_datasets,name)).records()
        for r in records: result[(name,r.video_id)]=r
    return result
def _relative(record: Any, raw_root: Path) -> str:
    return record.source_path.relative_to(raw_root).as_posix()
def _verify(rows: dict[str,list[dict[str,Any]]], m2a_root: Path, cfg: Any, report: dict[str,Any]) -> None:
    allids=set(); errors=[]
    expected_model=sha256_file(resolve_detector_path(cfg.scrfd_model_path))
    for dataset,items in rows.items():
        for row in items:
            sid=row.get("sample_id"); prefix=f"{dataset}:{sid}"
            if sid in allids: errors.append(prefix+": duplicate sample_id")
            allids.add(sid)
            crop=m2a_root/row.get("crop_relative_path","")
            if not crop.is_file(): errors.append(prefix+": crop missing")
            elif cv2.imread(str(crop)) is None: errors.append(prefix+": crop unreadable")
            elif sha256_file(crop)!=row.get("crop_sha256"): errors.append(prefix+": crop hash mismatch")
            if row.get("detector_model_sha256") != expected_model: errors.append(prefix+": detector hash mismatch")
            if row.get("detector_input_size") != 320 or row.get("detector_input_size") != cfg.scrfd_input_size: errors.append(prefix+": input size mismatch")
            if float(row.get("detector_threshold",-1)) != .5 or float(row.get("detector_threshold",-1)) != cfg.detection_threshold: errors.append(prefix+": threshold mismatch")
            if row.get("preprocessing_config_hash") != cfg.config_hash: errors.append(prefix+": config hash mismatch")
    report.update({"success_rows":{k:len(v) for k,v in rows.items()},"unique_sample_ids":len(allids),"errors":errors,"crop_existence_hash_verified":not errors,"expected_detector_model_sha256":expected_model})
    if errors: raise ValueError("M2A input verification failed: "+"; ".join(errors))
def _frame(row:dict[str,Any],record:Any,raw_root:Path,cfg:Any,source:bool)->dict[str,Any]:
    common={"sample_id":row["sample_id"],"dataset":row["dataset"],"video_id":row["video_id"],"source_record_id":row["source_record_id"],"source_media_type":row["source_media_type"],"source_relative_identifier":_relative(record,raw_root),"requested_frame_index":row["requested_frame_index"],"actual_frame_index":row["actual_frame_index"],"timestamp_ms":row.get("timestamp_ms"),"frame_width":row["frame_width"],"frame_height":row["frame_height"],"selected_frame_reference":f"{_relative(record,raw_root)}#frame={row['actual_frame_index']}","materialized_frame_relative_path":None,"source_fingerprint":record.source_fingerprint,"frame_fingerprint":None,"decoder_backend":row["decoder_backend"],"adapter_version":record.adapter_version,"sampling_version":cfg.sampling_version,"preprocessing_version":cfg.preprocessing_version,"preprocessing_config_hash":cfg.config_hash,"status":row["status"],"warning_codes":[]}
    if source: common.update(subject_id=record.subject_id,official_split=record.official_split,label_live_spoof=record.label)
    return common
def _crop(row:dict[str,Any],record:Any,cfg:Any,source:bool,m2a_root:Path)->dict[str,Any]:
    bbox=row["bbox"]; lms=row["landmarks"]; box=row["crop_box"]
    if len(bbox)!=4 or len(lms)!=5 or any(len(p)!=2 for p in lms) or len(box)!=4: raise ValueError(f"invalid geometry: {row['sample_id']}")
    image=cv2.imread(str(m2a_root/row["crop_relative_path"])); h,w=image.shape[:2]
    d={"sample_id":row["sample_id"],"dataset":row["dataset"],"video_id":row["video_id"],"source_record_id":row["source_record_id"],"source_media_type":row["source_media_type"],"requested_frame_index":row["requested_frame_index"],"actual_frame_index":row["actual_frame_index"],"timestamp_ms":row.get("timestamp_ms"),"frame_width":row["frame_width"],"frame_height":row["frame_height"],"bbox_x1":bbox[0],"bbox_y1":bbox[1],"bbox_x2":bbox[2],"bbox_y2":bbox[3],"detection_score":row["detection_score"],"detected_face_count":row["detected_face_count"],"crop_x1":box[0],"crop_y1":box[1],"crop_x2":box[2],"crop_y2":box[3],"requested_crop_padding":cfg.crop_padding,"effective_crop_padding":cfg.crop_padding,"crop_width":w,"crop_height":h,"crop_relative_path":row["crop_relative_path"],"crop_sha256":row["crop_sha256"],"detector_name":"scrfd","detector_model_sha256":row["detector_model_sha256"],"detector_provider":"CPUExecutionProvider","detector_input_size":row["detector_input_size"],"detector_threshold":row["detector_threshold"],"preprocessing_version":cfg.preprocessing_version,"preprocessing_config_hash":cfg.config_hash,"status":row["status"]}
    for i,(x,y) in enumerate(lms): d[f"landmark_{i}_x"]=x; d[f"landmark_{i}_y"]=y
    if source: d.update(subject_id=record.subject_id,official_split=record.official_split,label_live_spoof=record.label)
    return d
def migrate_m2a(config_path:Path, preprocess_path:Path, m2a_root:Path, output_root:Path, force:bool=False)->dict[str,Any]:
    paths=load_paths(config_path); cfg=load_m2_config(preprocess_path); project=paths.project_root
    m2a_root=m2a_root.resolve(); output_root=output_root.resolve()
    if output_root.name != "manifests": raise ValueError("output root must be the explicit manifests directory")
    rows={name:_read(m2a_root/"results"/f"{name}.jsonl") for name in DATASETS}; failures=_read(m2a_root/"results"/"failures.jsonl")
    verify={"m2a_root":str(m2a_root),"failure_rows":len(failures)}; _verify(rows,m2a_root,cfg,verify)
    reports=paths.reports_root/"m2b1a"; reports.mkdir(parents=True,exist_ok=True); atomic_json_write(reports/"input_verification.json",verify); (reports/"input_verification.md").write_text("# M2A input verification\n\n```json\n"+json.dumps(verify,indent=2)+"\n```\n",encoding="utf-8")
    canonical=_canonical(paths,project); source_frames=[];source_crops=[];target_frames=[];target_crops=[]
    for dataset,items in rows.items():
        for row in items:
            record=canonical.get((dataset,row["video_id"]))
            if record is None: raise ValueError(f"canonical record missing for {dataset}:{row['video_id']}")
            source=dataset!="siw_mv2"; raw_root=getattr(paths.raw_datasets,dataset)
            (source_frames if source else target_frames).append(_frame(row,record,raw_root,cfg,source))
            (source_crops if source else target_crops).append(_crop(row,record,cfg,source,m2a_root))
    leakage=find_target_leakage(target_frames)+find_target_leakage(target_crops)
    isolation={"passed":not leakage,"violations":leakage,"target_rows":len(target_frames)}; atomic_json_write(reports/"target_isolation_report.json",isolation)
    if leakage: raise ValueError(f"target leakage detected: {leakage}")
    output_root.mkdir(parents=True,exist_ok=True); timestamp=datetime.now(timezone.utc).isoformat(); modelhash=sha256_file(resolve_detector_path(cfg.scrfd_model_path))
    metadata={"manifest_schema_version":"m2b1a-v1","preprocessing_version":cfg.preprocessing_version,"preprocessing_config_hash":cfg.config_hash,"detector_model_sha256":modelhash,"detector_input_size":str(cfg.scrfd_input_size),"detector_threshold":str(cfg.detection_threshold),"git_commit":_git(project),"created_at":timestamp,"dataset_roles":"casia_fasd,msu_mfsd=source;siw_mv2=target"}
    collections={"source_frames":source_frames,"source_crops":source_crops,"target_frames":target_frames,"target_crops":target_crops,"preprocessing_failures":failures}
    output={}
    for name,items in collections.items(): output[name]=write_parquet_atomic(output_root/f"{name}.parquet",items,MODELS[name],metadata,name=="preprocessing_failures")
    expected={"source_frames":24,"source_crops":24,"target_frames":12,"target_crops":12,"preprocessing_failures":0}
    counts={k:v["rows"] for k,v in output.items()}
    if counts != expected: raise ValueError(f"migration data loss/count mismatch: {counts}")
    report={"timestamp":timestamp,"m2a_input_paths":{k:str(m2a_root/"results"/f"{k}.jsonl") for k in DATASETS},"jsonl_row_counts":{**{k:len(v) for k,v in rows.items()},"failures":len(failures)},"input_verification":verify,"source_target_roles":{"casia_fasd":"source","msu_mfsd":"source","siw_mv2":"target"},"outputs":{k:{**v,"path":str(output_root/f"{k}.parquet")} for k,v in output.items()},"config_hash":cfg.config_hash,"detector_hash":modelhash,"git_commit":metadata["git_commit"],"command":"prism data preprocess migrate-m2a"}
    atomic_json_write(reports/"migration_report.json",report); (reports/"migration_report.md").write_text("# M2B1a migration\n\n```json\n"+json.dumps(report,indent=2)+"\n```\n",encoding="utf-8")
    return report
