from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from .converters import build_preprocessing_failure_record,build_source_frame_record,build_source_crop_record,build_target_frame_record,build_target_crop_record
class SourceRoutingConsistencyError(ValueError): pass
class TargetRoutingConsistencyError(ValueError): pass
class FailureRoutingConsistencyError(ValueError): pass
@dataclass(frozen=True)
class SourceRoutingResult:
 sample_id:str;frame_record:object;crop_record:object;source_frame_count:int;source_crop_count:int;duplicate_collapsed:bool;routed:bool
@dataclass(frozen=True)
class TargetRoutingResult:
 sample_id:str;frame_record:object;crop_record:object;target_frame_count:int;target_crop_count:int;duplicate_collapsed:bool;routed:bool
@dataclass(frozen=True)
class FailureRoutingResult:
 sample_id:str;failure_record:object;failure_count:int;routed:bool
def route_source_success(*,repository,context,canonical_record,sample_id,requested_frame_index,actual_frame_index,timestamp_ms,frame_width,frame_height,decoder_backend,source_media_type,source_fingerprint=None,frame_fingerprint=None,selected_frame_reference='source',materialized_frame_relative_path=None,bbox=(),landmarks=(),detection_score=0.,detected_face_count=1,crop_box=(),requested_crop_padding=.25,effective_crop_padding=.25,crop_width=1,crop_height=1,crop_relative_path='',crop_sha256='',warning_codes=()):
 frame=build_source_frame_record(context,canonical_record,sample_id=sample_id,source_media_type=source_media_type,source_relative_identifier=selected_frame_reference.split('#')[0],requested_frame_index=requested_frame_index,actual_frame_index=actual_frame_index,timestamp_ms=timestamp_ms,frame_width=frame_width,frame_height=frame_height,selected_frame_reference=selected_frame_reference,frame_fingerprint=frame_fingerprint,decoder_backend=decoder_backend,warning_codes=list(warning_codes))
 crop=build_source_crop_record(context,canonical_record,sample_id=sample_id,source_media_type=source_media_type,requested=requested_frame_index,actual=actual_frame_index,timestamp=timestamp_ms,width=frame_width,height=frame_height,bbox=bbox,landmarks=landmarks,score=detection_score,count=detected_face_count,box=crop_box,padding=requested_crop_padding,cw=crop_width,ch=crop_height,path=crop_relative_path,sha=crop_sha256)
 for field in ('sample_id','dataset','source_record_id','requested_frame_index','source_media_type'):
  if getattr(frame,field)!=getattr(crop,field):raise SourceRoutingConsistencyError(f'frame/crop {field} mismatch')
 if frame.source_media_type!=source_media_type:raise SourceRoutingConsistencyError('frame/crop source_media_type mismatch')
 before=repository.counts();repository.upsert_source_success(frame.model_dump(mode='json'),crop.model_dump(mode='json'));after=repository.counts()
 return SourceRoutingResult(sample_id,frame,crop,after['source_frames'],after['source_crops'],after['source_frames']==before['source_frames'],True)
def route_target_success(*,repository,context,canonical_record,sample_id,requested_frame_index,actual_frame_index,timestamp_ms,frame_width,frame_height,decoder_backend,source_media_type,frame_fingerprint=None,selected_frame_reference='target',materialized_frame_relative_path=None,bbox=(),landmarks=(),detection_score=0.,detected_face_count=1,crop_box=(),requested_crop_padding=.25,effective_crop_padding=.25,crop_width=1,crop_height=1,crop_relative_path='',crop_sha256='',warning_codes:Sequence[str]=()):
 # Target manifests carry no canonical private metadata: only the fields the
 # strict target converters accept are forwarded, never the record itself.
 if context.dataset_role!='target':raise TargetRoutingConsistencyError("target routing requires dataset_role='target'")
 frame=build_target_frame_record(context,canonical_record,sample_id=sample_id,source_media_type=source_media_type,source_relative_identifier=selected_frame_reference.split('#')[0],requested_frame_index=requested_frame_index,actual_frame_index=actual_frame_index,timestamp_ms=timestamp_ms,frame_width=frame_width,frame_height=frame_height,selected_frame_reference=selected_frame_reference,frame_fingerprint=frame_fingerprint,decoder_backend=decoder_backend,warning_codes=list(warning_codes))
 crop=build_target_crop_record(context,canonical_record,sample_id=sample_id,source_media_type=source_media_type,requested=requested_frame_index,actual=actual_frame_index,timestamp=timestamp_ms,width=frame_width,height=frame_height,bbox=bbox,landmarks=landmarks,score=detection_score,count=detected_face_count,box=crop_box,padding=requested_crop_padding,cw=crop_width,ch=crop_height,path=crop_relative_path,sha=crop_sha256)
 for field in ('sample_id','dataset','video_id','source_record_id','requested_frame_index','actual_frame_index','source_media_type'):
  if getattr(frame,field)!=getattr(crop,field):raise TargetRoutingConsistencyError(f'frame/crop {field} mismatch')
 for record,name in ((frame,'frame'),(crop,'crop')):
  if record.source_media_type!=source_media_type:raise TargetRoutingConsistencyError(f'target {name} source_media_type mismatch')
  if record.sample_id!=sample_id:raise TargetRoutingConsistencyError(f'target {name} sample_id mismatch')
  if record.dataset!=canonical_record.dataset:raise TargetRoutingConsistencyError(f'target {name} dataset mismatch')
  if record.source_record_id!=canonical_record.video_id:raise TargetRoutingConsistencyError(f'target {name} source_record_id mismatch')
  if record.requested_frame_index!=requested_frame_index:raise TargetRoutingConsistencyError(f'target {name} requested_frame_index mismatch')
  if record.actual_frame_index!=actual_frame_index:raise TargetRoutingConsistencyError(f'target {name} actual_frame_index mismatch')
 if crop.crop_relative_path!=Path(crop_relative_path).as_posix() or crop.crop_sha256!=crop_sha256:raise TargetRoutingConsistencyError('target crop artifact reference mismatch')
 before=repository.counts();repository.upsert_target_success(frame.model_dump(mode='json'),crop.model_dump(mode='json'));after=repository.counts()
 return TargetRoutingResult(sample_id,frame,crop,after['target_frames'],after['target_crops'],after['target_frames']==before['target_frames'],True)
def route_preprocessing_failure(*,repository,context,canonical_record,sample_id,requested_frame_index,actual_frame_index,stage,error_code,error_message,backend,recoverable,warning_codes:Sequence[str]=()):
 failure=build_preprocessing_failure_record(context,canonical_record,sample_id=sample_id,requested_frame_index=requested_frame_index,actual_frame_index=actual_frame_index,stage=stage,error_code=error_code,message=error_message,backend=backend,recoverable=recoverable,warning_codes=list(warning_codes))
 if failure.sample_id!=sample_id:raise FailureRoutingConsistencyError('failure sample_id mismatch')
 if failure.dataset!=canonical_record.dataset:raise FailureRoutingConsistencyError('failure dataset mismatch')
 if failure.source_record_id!=canonical_record.video_id:raise FailureRoutingConsistencyError('failure source_record_id mismatch')
 repository.upsert_failure(failure.model_dump(mode='json'))
 return FailureRoutingResult(sample_id,failure,repository.counts()['preprocessing_failures'],True)
