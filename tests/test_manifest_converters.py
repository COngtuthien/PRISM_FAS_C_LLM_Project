from pathlib import Path
import pytest
from prism_fas.data.run_context import PreprocessingRunContext,M2OutputLayout
from prism_fas.data.manifests.converters import *
from prism_fas.data.schemas.records import CanonicalVideoRecord,TargetInferenceRecord

def ctx(tmp_path,role='source'):
 l=M2OutputLayout.from_root(tmp_path/'full_preprocessing');return PreprocessingRunContext(project_root=tmp_path,work_root=tmp_path,run_profile='full_preprocessing',output_namespace='full_preprocessing',output_root=l.output_root,crops_root=l.crops_root,frames_root=l.frames_root,manifests_root=l.manifests_root,state_root=l.state_root,reports_root=l.reports_root,logs_root=l.logs_root,run_id='x',dataset='casia_fasd' if role=='source' else 'siw_mv2',dataset_role=role,preprocessing_version='m2-v1',preprocessing_config_hash='h',detector_model_path=tmp_path/'m',detector_model_sha256='a'*64,detector_input_size=320,detector_threshold=.5,all_records=False,record_limit=1,sample_limit=None,resume=False,dry_run=True,partial_full_profile=True,command='x')
def source(tmp_path):return CanonicalVideoRecord(dataset='casia_fasd',subject_id='1',video_id='v',source_path=tmp_path/'x',official_split='train',label='live',adapter_version='1',source_fingerprint='f',metadata_provenance='test')
def test_source_frame_and_metadata(tmp_path):
 r=build_source_frame_record(ctx(tmp_path),source(tmp_path),sample_id='s',source_media_type='image_sequence',source_relative_identifier='x',requested_frame_index=0,actual_frame_index=0,frame_width=10,frame_height=10,selected_frame_reference='x#0',decoder_backend='opencv');assert r.subject_id=='1' and r.label_live_spoof=='live'
def test_missing_source_metadata_rejected(tmp_path):
 r=source(tmp_path).model_copy(update={'subject_id':None})
 with pytest.raises(MissingCanonicalMetadataError):build_source_frame_record(ctx(tmp_path),r,sample_id='s',source_media_type='image_sequence',source_relative_identifier='x',requested_frame_index=0,actual_frame_index=0,frame_width=1,frame_height=1,selected_frame_reference='x',decoder_backend='x')
def test_source_crop_geometry_path_hash(tmp_path):
 kw=dict(sample_id='s',source_media_type='image_sequence',requested=0,actual=0,timestamp=None,width=10,height=10,bbox=[1,1,5,5],landmarks=[(1,1)]*5,score=.9,count=1,box=[0,0,6,6],padding=.2,cw=6,ch=6,path='crops/a.jpg',sha='a'*64)
 assert build_source_crop_record(ctx(tmp_path),source(tmp_path),**kw).bbox_x2==5
 with pytest.raises(InvalidGeometryError):build_source_crop_record(ctx(tmp_path),source(tmp_path),**{**kw,'bbox':[1,1,1,5]})
 with pytest.raises(InvalidPortablePathError):build_source_crop_record(ctx(tmp_path),source(tmp_path),**{**kw,'path':'../x.jpg'})
def test_target_and_failure(tmp_path):
 t=TargetInferenceRecord(dataset='siw_mv2',video_id='v',source_path=tmp_path/'x',official_split='target_test',adapter_version='1',source_fingerprint='f',metadata_provenance='test')
 out=build_target_frame_record(ctx(tmp_path,'target'),t,sample_id='s',source_media_type='video_file',source_relative_identifier='x',requested_frame_index=0,actual_frame_index=0,frame_width=1,frame_height=1,selected_frame_reference='x',decoder_backend='x');assert 'label' not in out.model_dump()
 fail=build_preprocessing_failure_record(ctx(tmp_path),source(tmp_path),sample_id='s',error_code='no_face',message='D:\\raw\\x\nsecret');assert '[redacted-path]' in fail.sanitized_error_message
def test_crop_rejects_nonfinite_absolute_and_bad_hash(tmp_path):
 kw=dict(sample_id='s',source_media_type='image_sequence',requested=0,actual=0,timestamp=None,width=10,height=10,bbox=[1,1,5,5],landmarks=[(1,1)]*5,score=.9,count=1,box=[0,0,6,6],padding=.2,cw=6,ch=6,path='c.jpg',sha='a'*64)
 with pytest.raises(InvalidGeometryError):build_source_crop_record(ctx(tmp_path),source(tmp_path),**{**kw,'landmarks':[(float('nan'),1)]*5})
 with pytest.raises(InvalidPortablePathError):build_source_crop_record(ctx(tmp_path),source(tmp_path),**{**kw,'path':'C:/bad.jpg'})
 with pytest.raises(ManifestConversionError):build_source_crop_record(ctx(tmp_path),source(tmp_path),**{**kw,'sha':'bad'})
@pytest.mark.parametrize('code',['no_face','detector_failed','decode_failed','frame_index_unavailable','invalid_bbox','invalid_landmarks','crop_failed','output_write_failed','hash_failed','target_leakage_detected'])
def test_failure_codes_are_strict(tmp_path,code):
 assert build_preprocessing_failure_record(ctx(tmp_path),source(tmp_path),sample_id='s',error_code=code,message='x').error_code==code
