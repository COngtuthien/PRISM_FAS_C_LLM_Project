from __future__ import annotations
import json, hashlib
from pathlib import Path
from numbers import Real
import yaml, cv2
from prism_fas.config.models import DatasetDefinition,load_paths
from prism_fas.data.adapters import adapter_for
from prism_fas.data.media import ImageSequenceReader,OpenCVVideoDecoder,UnreadableImageError,VideoDecodeError
from prism_fas.data.preprocess_m2 import CropProcessingError,DetectorInferenceError,InvalidBoundingBoxError,InvalidLandmarksError,SCRFDDetector,load_m2_config,uniform_indices,sample_id,select_largest,crop_face,validate_landmarks, resolve_detector_path
from prism_fas.data.output import HashComputationError,OutputWriteError,discard_crop_artifact,hash_crop_artifact,write_crop_image
from prism_fas.utils.core import sha256_file,atomic_json_write
from prism_fas.data.manifests.resume import resume_action, Lock, initialize_run_state, write_run_state_atomic
from prism_fas.data.run_context import PreprocessingRunContext
from prism_fas.data.manifests.repository import ManifestRepository
from prism_fas.data.manifests.routing import route_preprocessing_failure,route_source_success,route_target_success
from pydantic import BaseModel, ConfigDict, model_validator
from datetime import datetime, timezone
FORBIDDEN={"label","live","spoof","attack","taxonomy","ground_truth","private_metadata","subject_id"}
MANIFEST_COUNT_KEYS=("source_frames","source_crops","target_frames","target_crops","preprocessing_failures")
class RunExecutionResult(BaseModel):
    model_config=ConfigDict(extra='forbid')
    run_profile:str;run_id:str;dataset:str;dataset_role:str;output_root:Path;canonical_records_total:int;canonical_records_selected:int;canonical_records_attempted:int;samples_selected:int;samples_successful:int;samples_failed:int;frames_read:int;detector_calls:int;crops_written:int;failures_by_code:dict[str,int];manifest_counts:dict[str,int];dry_run:bool;partial:bool;status:str;started_at:str;finished_at:str
    @model_validator(mode='after')
    def accounting(self):
        if self.samples_selected!=self.samples_successful+self.samples_failed or self.crops_written>self.samples_successful:raise ValueError('invalid sample accounting')
        return self
def assert_context_write_layout(context:PreprocessingRunContext)->None:
    roots=[context.crops_root,context.frames_root,context.manifests_root,context.state_root,context.reports_root,context.logs_root]
    if any(not str(p.resolve()).startswith(str(context.output_root.resolve())) for p in roots):raise ValueError('context layout escapes output root')
    if context.run_profile=='full_preprocessing' and any('m2a' in p.resolve().parts for p in roots):raise ValueError('full profile cannot write to m2a')
def normalize_manifest_counts(counts)->dict[str,int]:
    if not isinstance(counts,dict) or set(counts)!=set(MANIFEST_COUNT_KEYS):raise ValueError('invalid manifest count keys')
    normalized={}
    for key in MANIFEST_COUNT_KEYS:
        value=counts[key]
        if isinstance(value,bool) or not isinstance(value,Real) or int(value)!=value or value<0:raise ValueError(f'invalid manifest count for {key}')
        normalized[key]=int(value)
    return normalized
def run_preprocessing(context:PreprocessingRunContext,canonical_records,*,detector=None,media_reader_factory=None,repository_factory=ManifestRepository)->RunExecutionResult:
    assert_context_write_layout(context)
    metadata={'manifest_schema_version':'m2f1a-v1','output_root':str(context.output_root),'run_profile':context.run_profile,'preprocessing_version':context.preprocessing_version,'preprocessing_config_hash':context.preprocessing_config_hash,'detector_model_sha256':context.detector_model_sha256,'detector_input_size':str(context.detector_input_size),'detector_threshold':str(context.detector_threshold)}
    repo=repository_factory(context.manifests_root,metadata).initialize()
    if repo is not None and hasattr(repo,'load'): repo.load()
    now=datetime.now(timezone.utc).isoformat(); selected=len(canonical_records); attempted=0; samples=success=failed=frames=calls=crops=0; failure_codes={}
    # Real media execution is deliberately injected: tests use the existing
    # reader contract while production CLI remains on its legacy path.
    if not context.dry_run:
        for record in canonical_records:
            # One decision drives both the reader and the media type recorded
            # in the manifests, so frame and crop rows can never disagree.
            attempted+=1; source_media_type='image_sequence' if context.dataset=='casia_fasd' else 'video_file'
            reader=(media_reader_factory(record) if media_reader_factory else (ImageSequenceReader(record.source_path) if source_media_type=='image_sequence' else OpenCVVideoDecoder(record.source_path)))
            try: indices=uniform_indices(reader.frame_count(),4,.05,.05)
            except Exception: indices=[]
            for index in indices:
                samples+=1
                source_routing_started=target_routing_started=failure_routing_started=hash_boundary_started=False
                sid=sample_id(context.dataset,record.video_id,index,record.adapter_version,"uniform-v1",context.preprocessing_version)
                try:
                    try: frame=reader.read_frame(index)
                    except IndexError:
                        if context.dataset_role in ('source','target'):
                            failure_routing_started=True
                            if repo is None: raise RuntimeError('failure routing requires an initialized manifest repository')
                            route_preprocessing_failure(repository=repo,context=context,canonical_record=record,source_relative_identifier=context.dataset_role,sample_id=sid,requested_frame_index=index,actual_frame_index=None,stage='media_read',error_code='frame_index_unavailable',error_message='requested frame index is unavailable',backend='media_reader',recoverable=True)
                            failed+=1;failure_codes['frame_index_unavailable']=failure_codes.get('frame_index_unavailable',0)+1;continue
                        raise
                    except UnreadableImageError:
                        if context.dataset_role in ('source',) and context.dataset == 'casia_fasd':
                            failure_routing_started=True
                            if repo is None: raise RuntimeError('failure routing requires an initialized manifest repository')
                            route_preprocessing_failure(repository=repo,context=context,canonical_record=record,source_relative_identifier=context.dataset_role,sample_id=sid,requested_frame_index=index,actual_frame_index=index,stage='media_read',error_code='unreadable_image',error_message='source image could not be read',backend='image_sequence',recoverable=True)
                            failed+=1;failure_codes['unreadable_image']=failure_codes.get('unreadable_image',0)+1;continue
                        raise
                    except VideoDecodeError:
                        if context.dataset_role in ('source','target') and record.source_path.suffix.lower() in {'.mp4','.avi','.mov'}:
                            failure_routing_started=True
                            if repo is None: raise RuntimeError('failure routing requires an initialized manifest repository')
                            route_preprocessing_failure(repository=repo,context=context,canonical_record=record,source_relative_identifier=context.dataset_role,sample_id=sid,requested_frame_index=index,actual_frame_index=None,stage='media_read',error_code='decode_failed',error_message='video frame could not be decoded',backend='video_decoder',recoverable=True)
                            failed+=1;failure_codes['decode_failed']=failure_codes.get('decode_failed',0)+1;continue
                        raise
                    frames+=1; calls+=1
                    try: detections=detector.detect(frame.image)
                    except DetectorInferenceError:
                        if context.dataset_role in ('source','target'):
                            failure_routing_started=True
                            if repo is None: raise RuntimeError('failure routing requires an initialized manifest repository')
                            route_preprocessing_failure(repository=repo,context=context,canonical_record=record,source_relative_identifier=context.dataset_role,sample_id=sid,requested_frame_index=index,actual_frame_index=frame.actual_frame_index,stage='detector',error_code='detector_failed',error_message='face detector inference failed',backend='detector',recoverable=True)
                            failed+=1;failure_codes['detector_failed']=failure_codes.get('detector_failed',0)+1;continue
                        raise
                    try:
                        face=select_largest(detections,context.detector_threshold,16)
                        if face is None and context.dataset_role in ('source','target'):
                            failure_routing_started=True
                            if repo is None: raise RuntimeError('failure routing requires an initialized manifest repository')
                            route_preprocessing_failure(repository=repo,context=context,canonical_record=record,source_relative_identifier=context.dataset_role,sample_id=sid,requested_frame_index=index,actual_frame_index=frame.actual_frame_index,stage='detector',error_code='no_face',error_message='no valid face detected',backend=frame.backend,recoverable=True)
                            failed+=1;failure_codes['no_face']=failure_codes.get('no_face',0)+1;continue
                        if face is None:
                            failed+=1;failure_codes['no_face']=failure_codes.get('no_face',0)+1;continue
                        validate_landmarks(face.landmarks)
                        image,box=crop_face(frame.image,face,.25,224)
                    except InvalidBoundingBoxError:
                        if context.dataset_role in ('source','target'):
                            failure_routing_started=True
                            if repo is None: raise RuntimeError('failure routing requires an initialized manifest repository')
                            route_preprocessing_failure(repository=repo,context=context,canonical_record=record,source_relative_identifier=context.dataset_role,sample_id=sid,requested_frame_index=index,actual_frame_index=frame.actual_frame_index,stage='geometry',error_code='invalid_bbox',error_message='detected face bounding box is invalid',backend='geometry',recoverable=True)
                            failed+=1;failure_codes['invalid_bbox']=failure_codes.get('invalid_bbox',0)+1;continue
                        raise
                    except InvalidLandmarksError:
                        if context.dataset_role in ('source','target'):
                            failure_routing_started=True
                            if repo is None: raise RuntimeError('failure routing requires an initialized manifest repository')
                            route_preprocessing_failure(repository=repo,context=context,canonical_record=record,source_relative_identifier=context.dataset_role,sample_id=sid,requested_frame_index=index,actual_frame_index=frame.actual_frame_index,stage='geometry',error_code='invalid_landmarks',error_message='detected face landmarks are invalid',backend='geometry',recoverable=True)
                            failed+=1;failure_codes['invalid_landmarks']=failure_codes.get('invalid_landmarks',0)+1;continue
                        raise
                    except CropProcessingError:
                        if context.dataset_role in ('source','target'):
                            failure_routing_started=True
                            if repo is None: raise RuntimeError('failure routing requires an initialized manifest repository')
                            route_preprocessing_failure(repository=repo,context=context,canonical_record=record,source_relative_identifier=context.dataset_role,sample_id=sid,requested_frame_index=index,actual_frame_index=frame.actual_frame_index,stage='crop',error_code='crop_failed',error_message='face crop could not be created',backend='crop_preprocessing',recoverable=True)
                            failed+=1;failure_codes['crop_failed']=failure_codes.get('crop_failed',0)+1;continue
                        raise
                    try: target=write_crop_image(image,context.crops_root/context.dataset/f'{sid}.jpg')
                    except OutputWriteError as exc:
                        if context.dataset_role in ('source','target'):
                            failure_routing_started=True
                            if repo is None: raise RuntimeError('failure routing requires an initialized manifest repository')
                            route_preprocessing_failure(repository=repo,context=context,canonical_record=record,source_relative_identifier=context.dataset_role,sample_id=sid,requested_frame_index=index,actual_frame_index=frame.actual_frame_index,stage='output_write',error_code='output_write_failed',error_message='face crop output could not be written',backend=exc.backend,recoverable=True)
                            failed+=1;failure_codes['output_write_failed']=failure_codes.get('output_write_failed',0)+1;continue
                        raise
                    hash_boundary_started=True
                    try: digest=hash_crop_artifact(target)
                    except HashComputationError as exc:
                        hash_boundary_started=False
                        discard_crop_artifact(target)
                        if context.dataset_role in ('source','target'):
                            failure_routing_started=True
                            if repo is None: raise RuntimeError('failure routing requires an initialized manifest repository')
                            route_preprocessing_failure(repository=repo,context=context,canonical_record=record,source_relative_identifier=context.dataset_role,sample_id=sid,requested_frame_index=index,actual_frame_index=frame.actual_frame_index,stage='hash',error_code='hash_failed',error_message='face crop hash could not be computed',backend=exc.backend,recoverable=True)
                            failed+=1;failure_codes['hash_failed']=failure_codes.get('hash_failed',0)+1;continue
                        raise
                    except Exception:
                        # Unexpected hash failures stay unclassified: clean the
                        # orphan crop, then let the marker force a re-raise.
                        discard_crop_artifact(target);raise
                    hash_boundary_started=False
                    if context.dataset_role == 'source':
                        source_routing_started=True
                        if repo is None: raise RuntimeError('source routing requires an initialized manifest repository')
                        route_source_success(repository=repo,context=context,canonical_record=record,sample_id=sid,requested_frame_index=index,actual_frame_index=frame.actual_frame_index,source_media_type=source_media_type,timestamp_ms=frame.timestamp_ms,frame_width=frame.width,frame_height=frame.height,decoder_backend=frame.backend,source_fingerprint=record.source_fingerprint,selected_frame_reference=f'{record.video_id}#frame={frame.actual_frame_index}',bbox=face.bbox,landmarks=face.landmarks,detection_score=face.score,detected_face_count=len(detections),crop_box=box,requested_crop_padding=.25,effective_crop_padding=.25,crop_width=image.shape[1],crop_height=image.shape[0],crop_relative_path=target.relative_to(context.output_root).as_posix(),crop_sha256=digest)
                    elif context.dataset_role == 'target':
                        target_routing_started=True
                        if repo is None: raise RuntimeError('target routing requires an initialized manifest repository')
                        # Target references are built from the deterministic sample
                        # identity only: no raw path, filename or private metadata.
                        route_target_success(repository=repo,context=context,canonical_record=record,sample_id=sid,requested_frame_index=index,actual_frame_index=frame.actual_frame_index,source_media_type=source_media_type,timestamp_ms=frame.timestamp_ms,frame_width=frame.width,frame_height=frame.height,decoder_backend=frame.backend,selected_frame_reference=f'target/{sid}#frame={frame.actual_frame_index}',bbox=face.bbox,landmarks=face.landmarks,detection_score=face.score,detected_face_count=len(detections),crop_box=box,requested_crop_padding=.25,effective_crop_padding=.25,crop_width=image.shape[1],crop_height=image.shape[0],crop_relative_path=target.relative_to(context.output_root).as_posix(),crop_sha256=digest,warning_codes=frame.warnings)
                    else:
                        raise RuntimeError(f'unsupported dataset role: {context.dataset_role}')
                    crops+=1;success+=1
                except Exception as exc:
                    if source_routing_started or target_routing_started or failure_routing_started or hash_boundary_started: raise
                    failed+=1; code='no_face' if str(exc)=='no_face' else 'unrouted_processing_failure';failure_codes[code]=failure_codes.get(code,0)+1
            reader.close()
            repo.flush()
    repo.flush()
    manifest_counts=normalize_manifest_counts(repo.counts())
    end=datetime.now(timezone.utc).isoformat()
    return RunExecutionResult(run_profile=context.run_profile,run_id=context.run_id,dataset=context.dataset,dataset_role=context.dataset_role,output_root=context.output_root,canonical_records_total=selected,canonical_records_selected=selected,canonical_records_attempted=attempted,samples_selected=samples,samples_successful=success,samples_failed=failed,frames_read=frames,detector_calls=calls,crops_written=crops,failures_by_code=failure_codes,manifest_counts=manifest_counts,dry_run=context.dry_run,partial=context.partial_full_profile,status='partial' if context.partial_full_profile else 'completed',started_at=now,finished_at=end)
def run(dataset:str,config_path:Path,preprocess_path:Path,limit_records:int=3,dry_run:bool=False,resume:bool=False,force:bool=False)->dict:
    paths=load_paths(config_path); cfg=load_m2_config(preprocess_path); dfn=DatasetDefinition.model_validate(yaml.safe_load((Path('configs/data')/(dataset+'.yaml')).read_text())); records=adapter_for(dfn,getattr(paths.raw_datasets,dataset)).records()[:limit_records]
    root=paths.work_root/'m2'/cfg.preprocessing_version/cfg.config_hash/'m2a'; crops=root/'crops'/dataset; results=root/'results'; crops.mkdir(parents=True,exist_ok=True);results.mkdir(parents=True,exist_ok=True)
    model_hash=sha256_file(resolve_detector_path(cfg.scrfd_model_path)); state_root=paths.work_root/'m2'/cfg.preprocessing_version/cfg.config_hash; completed={}
    index=state_root/'state'/'completed_samples.parquet'
    if resume and index.exists():
        import pyarrow.parquet as pq
        completed={r['sample_id']:r for r in pq.read_table(index).to_pylist()}
    lock=Lock.acquire(state_root/'state'/'run.lock',command='prism data preprocess run',project_root=str(paths.project_root),output_root=str(root),preprocessing_config_hash=cfg.config_hash)
    state=initialize_run_state(preprocessing_version=cfg.preprocessing_version,preprocessing_config_hash=cfg.config_hash,detector_model_sha256=model_hash,detector_input_size=cfg.scrfd_input_size,detector_threshold=cfg.detection_threshold,requested_datasets=[dataset],limit_records=limit_records,limit_samples=None,resume_enabled=resume,force_enabled=force,git_commit='unknown',command='prism data preprocess run')
    write_run_state_atomic(state_root/'state'/'run_state.json',state); detector=None; stats={"dataset":dataset,"records_attempted":len(records),"frames_selected":0,"frames_decoded":0,"crops_created":0,"failures":0,"samples_selected":0,"samples_processed":0,"samples_skipped":0,"samples_reprocessed":0,"output_root":str(root)}; rows=[];fails=[]
    for r in records:
        reader=ImageSequenceReader(r.source_path) if dataset=='casia_fasd' else OpenCVVideoDecoder(r.source_path)
        try:n=reader.frame_count(); indices=uniform_indices(n,cfg.frames_per_video,cfg.start_exclusion_fraction,cfg.end_exclusion_fraction)
        except Exception as e:fails.append({"dataset":dataset,"source_record_id":r.video_id,"stage":"metadata","error_code":"metadata_read_failed","sanitized_error_message":str(e)});continue
        stats['frames_selected']+=len(indices)
        for idx in indices:
            sid=sample_id(dataset,r.video_id,idx,r.adapter_version,cfg.sampling_version,cfg.preprocessing_version); stats['samples_selected']+=1
            action=resume_action(completed[sid],cfg,model_hash,state_root) if resume and sid in completed else 'process'
            if action=='skip': stats['samples_skipped']+=1; continue
            if action.startswith('blocked'): raise RuntimeError(action)
            try:
                fr=reader.read_frame(idx);stats['frames_decoded']+=1
                if detector is None: detector=SCRFDDetector(resolve_detector_path(cfg.scrfd_model_path),cfg.scrfd_input_size)
                ds=detector.detect(fr.image); selected=select_largest(ds,cfg.detection_threshold,cfg.min_face_size)
                if selected is None: raise ValueError('no_face')
                crop,box=crop_face(fr.image,selected,cfg.crop_padding,cfg.crop_output_size); target=crops/f'{sid}.{cfg.output_image_format}'; tmp=target.with_suffix('.tmp.jpg'); cv2.imwrite(str(tmp),crop,[cv2.IMWRITE_JPEG_QUALITY,cfg.jpeg_quality]);tmp.replace(target)
                rec={"sample_id":sid,"dataset":dataset,"source_record_id":r.video_id,"video_id":r.video_id,"source_media_type":"image_sequence" if dataset=='casia_fasd' else "video_file","requested_frame_index":idx,"actual_frame_index":fr.actual_frame_index,"timestamp_ms":fr.timestamp_ms,"frame_width":fr.width,"frame_height":fr.height,"decoder_backend":fr.backend,"detection_score":selected.score,"detected_face_count":len(ds),"bbox":selected.bbox,"landmarks":selected.landmarks,"crop_box":box,"crop_relative_path":target.relative_to(root).as_posix(),"crop_sha256":sha256_file(target),"detector_model_sha256":detector.model_sha256,"detector_input_size":cfg.scrfd_input_size,"detector_threshold":cfg.detection_threshold,"preprocessing_config_hash":cfg.config_hash,"status":"success"}
                if dataset!='siw_mv2':rec.update(subject_id=r.subject_id,official_split=r.official_split,label_live_spoof=r.label)
                rows.append(rec);stats['crops_created']+=1;stats['samples_reprocessed' if action.startswith('reprocess') else 'samples_processed']+=1
            except Exception as e:fails.append({"dataset":dataset,"source_record_id":r.video_id,"sample_id":sid,"requested_frame_index":idx,"stage":"detect_crop","error_code":"no_face" if str(e)=='no_face' else "decode_failed","sanitized_error_message":str(e),"recoverable":True,"preprocessing_config_hash":cfg.config_hash})
        reader.close()
    for row in rows:
        if dataset=='siw_mv2' and any(key in row for key in FORBIDDEN): raise ValueError('target leakage')
    if rows:
        old=[json.loads(x) for x in (results/f'{dataset}.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()] if (results/f'{dataset}.jsonl').exists() else []
        merged={x['sample_id']:x for x in old};merged.update({x['sample_id']:x for x in rows});(results/f'{dataset}.jsonl').write_text(''.join(json.dumps(x,default=str)+'\n' for x in sorted(merged.values(),key=lambda x:x['sample_id'])),encoding='utf-8')
    stats['failures']=len(fails);state.samples_selected=stats['samples_selected'];state.samples_processed=stats['samples_processed'];state.samples_skipped=stats['samples_skipped'];state.samples_reprocessed=stats['samples_reprocessed'];state.samples_failed=len(fails);state.status='completed';state.finished_at=__import__('prism_fas.utils.core',fromlist=['utc_timestamp']).utc_timestamp();write_run_state_atomic(state_root/'state'/'run_state.json',state);lock.release()
    return stats
