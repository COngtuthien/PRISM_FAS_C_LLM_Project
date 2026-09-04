from __future__ import annotations
import math,re
from pathlib import Path
from prism_fas.data.manifests.schemas import SourceFrameRecord,SourceCropRecord,TargetFrameRecord,TargetCropRecord,PreprocessingFailureRecord
from prism_fas.data.manifests.leakage import assert_target_safe
class ManifestConversionError(ValueError):pass
class MissingCanonicalMetadataError(ManifestConversionError):pass
class InvalidGeometryError(ManifestConversionError):pass
class InvalidPortablePathError(ManifestConversionError):pass
class TargetLeakageConversionError(ManifestConversionError):pass
class InvalidMediaTypeError(ManifestConversionError):pass
MEDIA_TYPES=('image_sequence','video_file')
def _media(v):
 # Describes the media the frame came from, never the crop output format.
 if v not in MEDIA_TYPES:raise InvalidMediaTypeError(f'source_media_type must be one of {MEDIA_TYPES}')
 return v
def _path(v):
 p=Path(v)
 if p.is_absolute() or '..' in p.parts:raise InvalidPortablePathError(str(v))
 return p.as_posix()
def _geo(b,l,c):
 if len(b)!=4 or len(c)!=4 or len(l)!=5 or any(len(x)!=2 for x in l) or not all(math.isfinite(float(x)) for x in [*b,*c,*sum(([a,b] for a,b in l),[])]):raise InvalidGeometryError('geometry')
 if b[2]<=b[0] or b[3]<=b[1] or c[2]<=c[0] or c[3]<=c[1]:raise InvalidGeometryError('box')
def _crop(context,record,sid=None,requested=None,actual=None,timestamp=None,width=None,height=None,bbox=None,landmarks=None,score=None,count=None,box=None,padding=None,cw=None,ch=None,path=None,sha=None,source=True,source_media_type=None,**aliases):
 sid=aliases.get('sample_id',sid);requested=aliases.get('requested_frame_index',requested);actual=aliases.get('actual_frame_index',actual)
 media=_media(source_media_type);_geo(bbox,landmarks,box)
 if not re.fullmatch('[0-9a-f]{64}',sha):raise ManifestConversionError('sha256')
 d=dict(sample_id=sid,dataset=record.dataset,video_id=record.video_id,source_record_id=record.video_id,source_media_type=media,requested_frame_index=requested,actual_frame_index=actual,timestamp_ms=timestamp,frame_width=width,frame_height=height,bbox_x1=bbox[0],bbox_y1=bbox[1],bbox_x2=bbox[2],bbox_y2=bbox[3],detection_score=score,detected_face_count=count,crop_x1=box[0],crop_y1=box[1],crop_x2=box[2],crop_y2=box[3],requested_crop_padding=padding,effective_crop_padding=padding,crop_width=cw,crop_height=ch,crop_relative_path=_path(path),crop_sha256=sha,detector_name='scrfd',detector_model_sha256=context.detector_model_sha256,detector_provider='CPUExecutionProvider',detector_input_size=context.detector_input_size,detector_threshold=context.detector_threshold,preprocessing_version=context.preprocessing_version,preprocessing_config_hash=context.preprocessing_config_hash,status='success')
 for i,(x,y) in enumerate(landmarks):d[f'landmark_{i}_x']=x;d[f'landmark_{i}_y']=y
 if source:d.update(subject_id=record.subject_id,official_split=record.official_split,label_live_spoof=record.label);return SourceCropRecord.model_validate(d)
 out=TargetCropRecord.model_validate(d);assert_target_safe(out.model_dump());return out
def build_source_frame_record(context,record,**k):
 # `source_metadata_policy` is additive on PreprocessingRunContext (default
 # 'required'): every context that does not explicitly set it behaves exactly
 # as before this field existed. Only 'optional_unverifiable' (set ONLY by
 # the E7-B SiW-as-source context) allows a null subject_id through -- the
 # persisted schema (SourceFrameRecord.subject_id: str | None) already
 # permitted this; the legacy guard here was stricter than the schema.
 policy=getattr(context,'source_metadata_policy','required')
 if policy not in ('required','optional_unverifiable'):raise ManifestConversionError(f'unknown source_metadata_policy {policy!r}')
 subject_id_ok=bool(record.subject_id) or policy=='optional_unverifiable'
 if not subject_id_ok or not record.official_split or not record.label:raise MissingCanonicalMetadataError('source metadata')
 return SourceFrameRecord(sample_id=k['sample_id'],dataset=record.dataset,subject_id=record.subject_id,video_id=record.video_id,official_split=record.official_split,label_live_spoof=record.label,source_media_type=_media(k.get('source_media_type')),source_record_id=record.video_id,source_relative_identifier=_path(k['source_relative_identifier']),requested_frame_index=k['requested_frame_index'],actual_frame_index=k['actual_frame_index'],timestamp_ms=k.get('timestamp_ms'),frame_width=k['frame_width'],frame_height=k['frame_height'],selected_frame_reference=k['selected_frame_reference'],materialized_frame_relative_path=None,source_fingerprint=record.source_fingerprint,frame_fingerprint=k.get('frame_fingerprint'),decoder_backend=k['decoder_backend'],adapter_version=record.adapter_version,sampling_version='uniform-v1',preprocessing_version=context.preprocessing_version,preprocessing_config_hash=context.preprocessing_config_hash,status='success',warning_codes=k.get('warning_codes',[]))
def build_target_frame_record(context,record,**k):
 d=dict(sample_id=k['sample_id'],dataset=record.dataset,video_id=record.video_id,source_record_id=record.video_id,source_media_type=_media(k.get('source_media_type')),source_relative_identifier=_path(k['source_relative_identifier']),requested_frame_index=k['requested_frame_index'],actual_frame_index=k['actual_frame_index'],timestamp_ms=k.get('timestamp_ms'),frame_width=k['frame_width'],frame_height=k['frame_height'],selected_frame_reference=k['selected_frame_reference'],materialized_frame_relative_path=None,source_fingerprint=record.source_fingerprint,frame_fingerprint=k.get('frame_fingerprint'),decoder_backend=k['decoder_backend'],adapter_version=record.adapter_version,sampling_version='uniform-v1',preprocessing_version=context.preprocessing_version,preprocessing_config_hash=context.preprocessing_config_hash,status='success',warning_codes=k.get('warning_codes',[]));out=TargetFrameRecord.model_validate(d);assert_target_safe(out.model_dump());return out
def build_source_crop_record(context,record,**k):return _crop(context,record,source=True,**k)
def build_target_crop_record(context,record,**k):return _crop(context,record,source=False,**k)
def build_preprocessing_failure_record(context,record,**k):
 msg=re.sub(r'[\r\n\x00-\x1f]+',' ',str(k.get('message','')))[:240];msg=re.sub(r'[A-Za-z]:[^ ]+','[redacted-path]',msg)
 return PreprocessingFailureRecord(dataset=record.dataset,source_record_id=record.video_id,source_relative_identifier=_path(k.get('source_relative_identifier','source')),sample_id=k.get('sample_id'),requested_frame_index=k.get('requested_frame_index'),actual_frame_index=k.get('actual_frame_index'),stage=k.get('stage','process'),error_code=k.get('error_code','detector_failed'),sanitized_error_message=msg,timestamp=k.get('timestamp',''),preprocessing_config_hash=context.preprocessing_config_hash,detector_model_sha256=context.detector_model_sha256,backend=k.get('backend','opencv'),recoverable=k.get('recoverable',True),warning_codes=k.get('warning_codes',[]))
