import json, socket
from pathlib import Path
import pytest
from prism_fas.data.manifests.resume import CompletedSampleRecord, Lock

def row(): return dict(sample_id='a',dataset='d',dataset_role='source',source_record_id='v',video_id='v',requested_frame_index=0,actual_frame_index=0,source_fingerprint='f',frame_fingerprint=None,preprocessing_version='p',preprocessing_config_hash='h',detector_model_sha256='m',detector_input_size=320,detector_threshold=.5,frame_manifest_name='f',crop_manifest_name='c',frame_manifest_present=True,crop_manifest_present=True,crop_relative_path='x',crop_sha256='z',crop_exists=True,crop_hash_valid=True,target_isolation_valid=True,completed_at='x',last_verified_at='x',status='completed')
def test_completed_strict():
 assert CompletedSampleRecord.model_validate(row()).sample_id=='a'
 with pytest.raises(Exception): CompletedSampleRecord.model_validate({**row(),'bad':1})
def test_lock_active_and_release(tmp_path):
 p=tmp_path/'run.lock'; kw=dict(command='x',project_root='p',output_root='o',preprocessing_config_hash='h'); lock=Lock.acquire(p,**kw)
 with pytest.raises(RuntimeError): Lock.acquire(p,**kw)
 lock.release(); assert not p.exists()
def test_stale_lock_recovery(tmp_path):
 p=tmp_path/'run.lock'; p.write_text(json.dumps(dict(lock_token='old',pid=99999999,hostname=socket.gethostname(),process_start_time='x',run_started_at='x',command='x',project_root='p',output_root='o',preprocessing_config_hash='h',created_at='x')))
 lock=Lock.acquire(p,command='x',project_root='p',output_root='o',preprocessing_config_hash='h'); assert p.exists(); lock.release()
