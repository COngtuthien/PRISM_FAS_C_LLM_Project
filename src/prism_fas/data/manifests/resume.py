from __future__ import annotations
import json, os, secrets, socket, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
import cv2, pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict
from prism_fas.data.manifests.leakage import assert_target_safe
from prism_fas.data.manifests.parquet_writer import write_parquet_atomic
from prism_fas.utils.core import atomic_json_write, sha256_file, utc_timestamp

class CompletedSampleRecord(BaseModel):
    model_config=ConfigDict(extra='forbid')
    sample_id:str; dataset:str; dataset_role:Literal['source','target']; source_record_id:str; video_id:str; requested_frame_index:int; actual_frame_index:int; source_fingerprint:str; frame_fingerprint:str|None; preprocessing_version:str; preprocessing_config_hash:str; detector_model_sha256:str; detector_input_size:int; detector_threshold:float; frame_manifest_name:str; crop_manifest_name:str; frame_manifest_present:bool; crop_manifest_present:bool; crop_relative_path:str; crop_sha256:str; crop_exists:bool; crop_hash_valid:bool; target_isolation_valid:bool; completed_at:str; last_verified_at:str; status:Literal['completed','stale','missing_output','corrupt_output','config_mismatch','source_changed','manifest_missing','target_isolation_failed']
class RunState(BaseModel):
    model_config=ConfigDict(extra='forbid')
    run_id:str; preprocessing_version:str; preprocessing_config_hash:str; detector_model_sha256:str; detector_input_size:int; detector_threshold:float; requested_datasets:list[str]; limit_records:int; limit_samples:int|None; resume_enabled:bool; force_enabled:bool; started_at:str; updated_at:str; finished_at:str|None; status:Literal['initializing','running','interrupted','completed','failed','blocked']; records_selected:int=0; samples_selected:int=0; samples_processed:int=0; samples_skipped:int=0; samples_reprocessed:int=0; samples_failed:int=0; decode_failures:int=0; no_face_failures:int=0; other_failures:int=0; interruption_count:int=0; last_completed_sample_id:str|None=None; git_commit:str; hostname:str; pid:int; command:str
class RunLock(BaseModel):
    model_config=ConfigDict(extra='forbid')
    lock_token:str; pid:int; hostname:str; process_start_time:str; run_started_at:str; command:str; project_root:str; output_root:str; preprocessing_config_hash:str; created_at:str
def state_path(root:Path)->Path:return root/'state'/'run_state.json'
def write_run_state_atomic(path:Path,state:RunState)->None:
    state.updated_at=utc_timestamp(); state.finished_at=state.finished_at if state.status in {'completed','failed','blocked'} else None; atomic_json_write(path,state.model_dump(mode='json')); RunState.model_validate(json.loads(path.read_text()))
def load_run_state(path:Path)->RunState:return RunState.model_validate(json.loads(path.read_text()))
def initialize_run_state(**kwargs)->RunState:
    now=utc_timestamp(); return RunState(run_id=uuid.uuid4().hex,started_at=now,updated_at=now,finished_at=None,status='initializing',hostname=socket.gethostname(),pid=os.getpid(),**kwargs)
def _alive(pid:int)->bool:
    if os.name == 'nt':
        # ``os.kill(pid, 0)`` is not a safe liveness probe on every Windows
        # Python/runtime combination. Query the process handle instead.
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION=0x1000
        handle=ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION,False,pid)
        if not handle:return False
        try:
            code=ctypes.c_ulong(); ctypes.windll.kernel32.GetExitCodeProcess(handle,ctypes.byref(code))
            return code.value==259 # STILL_ACTIVE
        finally: ctypes.windll.kernel32.CloseHandle(handle)
    try: os.kill(pid,0); return True
    except OSError:return False
class Lock:
    def __init__(self,path:Path,lock:RunLock):self.path=path;self.lock=lock
    @classmethod
    def acquire(cls,path:Path,**kwargs):
        path.parent.mkdir(parents=True,exist_ok=True); now=utc_timestamp(); lock=RunLock(lock_token=secrets.token_urlsafe(32),pid=os.getpid(),hostname=socket.gethostname(),process_start_time=now,run_started_at=now,created_at=now,**kwargs)
        try:
            fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL); os.write(fd,json.dumps(lock.model_dump()).encode());os.close(fd);return cls(path,lock)
        except FileExistsError:
            existing=RunLock.model_validate(json.loads(path.read_text()))
            if existing.hostname!=socket.gethostname(): raise RuntimeError('state_conflict: foreign-host lock requires manual review')
            if _alive(existing.pid): raise RuntimeError('state_conflict: active local lock')
            stale=path.with_name(path.name+'.stale.'+utc_timestamp().replace(':','-')); os.replace(path,stale)
            return cls.acquire(path,**kwargs)
    def release(self):
        if self.path.exists() and RunLock.model_validate(json.loads(self.path.read_text())).lock_token==self.lock.lock_token:self.path.unlink()
def _rows(path:Path):return pq.read_table(path).to_pylist()
def build_completed_index(manifest_root:Path,state_root:Path,cfg,model_hash:str,reports:Path)->dict:
    frames={}; crops={}
    for role in ('source','target'):
        for r in _rows(manifest_root/f'{role}_frames.parquet'): frames[r['sample_id']]=(role,r)
        for r in _rows(manifest_root/f'{role}_crops.parquet'): crops[r['sample_id']]=(role,r)
    fail={r.get('sample_id') for r in _rows(manifest_root/'preprocessing_failures.parquet') if r.get('sample_id')}
    now=utc_timestamp(); completed=[]; invalid=[]
    for sid,(role,fr) in frames.items():
        cr=crops.get(sid)
        if not cr or sid in fail: invalid.append(sid);continue
        crop=manifest_root.parent/'m2a'/fr.get('crop_relative_path','') if False else manifest_root.parent/'m2a'/cr[1]['crop_relative_path']
        exists=crop.is_file(); readable=exists and cv2.imread(str(crop)) is not None; valid=readable and sha256_file(crop)==cr[1]['crop_sha256']
        iso=True
        if role=='target':
            try:assert_target_safe(fr);assert_target_safe(cr[1])
            except ValueError:iso=False
        status='completed' if valid and iso else ('target_isolation_failed' if not iso else ('missing_output' if not exists else 'corrupt_output'))
        rec=CompletedSampleRecord(sample_id=sid,dataset=fr['dataset'],dataset_role=role,source_record_id=fr['source_record_id'],video_id=fr['video_id'],requested_frame_index=fr['requested_frame_index'],actual_frame_index=fr['actual_frame_index'],source_fingerprint=fr['source_fingerprint'],frame_fingerprint=fr.get('frame_fingerprint'),preprocessing_version=cfg.preprocessing_version,preprocessing_config_hash=cfg.config_hash,detector_model_sha256=model_hash,detector_input_size=cfg.scrfd_input_size,detector_threshold=cfg.detection_threshold,frame_manifest_name=f'{role}_frames.parquet',crop_manifest_name=f'{role}_crops.parquet',frame_manifest_present=True,crop_manifest_present=True,crop_relative_path=cr[1]['crop_relative_path'],crop_sha256=cr[1]['crop_sha256'],crop_exists=exists,crop_hash_valid=valid,target_isolation_valid=iso,completed_at=now,last_verified_at=now,status=status)
        if status=='completed':completed.append(rec.model_dump(mode='json'))
        else:invalid.append(sid)
    meta={'manifest_schema_version':'m2b1b-v1','preprocessing_version':cfg.preprocessing_version,'preprocessing_config_hash':cfg.config_hash,'detector_model_sha256':model_hash,'detector_input_size':str(cfg.scrfd_input_size),'detector_threshold':str(cfg.detection_threshold),'created_at':now}
    info=write_parquet_atomic(state_root/'state'/'completed_samples.parquet',completed,CompletedSampleRecord,meta)
    report={'completed':len(completed),'by_dataset':{d:sum(r['dataset']==d for r in completed) for d in ['casia_fasd','msu_mfsd','siw_mv2']},'invalid':invalid,'writer':info};reports.mkdir(parents=True,exist_ok=True);atomic_json_write(reports/'completed_index_build.json',report);(reports/'completed_index_build.md').write_text('# Completed index build\n\n'+json.dumps(report,indent=2),encoding='utf-8');return report
def resume_action(row:dict,cfg,model_hash:str,output_root:Path)->str:
    if row['preprocessing_config_hash']!=cfg.config_hash or row['detector_model_sha256']!=model_hash:return 'blocked_config_mismatch'
    p=output_root/'m2a'/row['crop_relative_path']
    if not p.exists():return 'reprocess_missing_crop'
    if cv2.imread(str(p)) is None or sha256_file(p)!=row['crop_sha256']:return 'reprocess_corrupt_crop'
    return 'skip'
