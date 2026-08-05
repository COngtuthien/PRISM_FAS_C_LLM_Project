from pathlib import Path
from prism_fas.data.manifests.repository import ManifestRepository
from prism_fas.data.manifests.routing import route_source_success
from test_manifest_converters import ctx,source
import pytest
from unittest.mock import Mock
import prism_fas.data.manifests.routing as routing
from prism_fas.data.manifests.routing import SourceRoutingConsistencyError
from prism_fas.data.manifests.converters import MissingCanonicalMetadataError,InvalidGeometryError
def test_real_source_routing(tmp_path):
 c=ctx(tmp_path);r=ManifestRepository(tmp_path/'full_preprocessing'/'manifests',{'manifest_schema_version':'x'}).initialize()
 out=route_source_success(repository=r,context=c,canonical_record=source(tmp_path),sample_id='s',requested_frame_index=0,actual_frame_index=0,source_media_type='image_sequence',timestamp_ms=None,frame_width=10,frame_height=10,decoder_backend='opencv',selected_frame_reference='source#0',bbox=[1,1,5,5],landmarks=[(1,1)]*5,detection_score=.9,detected_face_count=1,crop_box=[0,0,6,6],crop_width=6,crop_height=6,crop_relative_path='crops/c.jpg',crop_sha256='a'*64)
 r.flush();assert out.routed and r.counts()['source_frames']==1 and r.counts()['target_frames']==0
def test_duplicate_and_second_source_are_retained(tmp_path):
 c=ctx(tmp_path);r=ManifestRepository(tmp_path/'full_preprocessing'/'manifests',{'manifest_schema_version':'x'}).initialize();base=dict(repository=r,context=c,canonical_record=source(tmp_path),requested_frame_index=0,actual_frame_index=0,source_media_type='image_sequence',timestamp_ms=None,frame_width=10,frame_height=10,decoder_backend='opencv',selected_frame_reference='source#0',bbox=[1,1,5,5],landmarks=[(1,1)]*5,detection_score=.9,detected_face_count=1,crop_box=[0,0,6,6],crop_width=6,crop_height=6,crop_relative_path='crops/c.jpg',crop_sha256='a'*64)
 route_source_success(sample_id='s',**base);route_source_success(sample_id='s',**base);assert r.counts()['source_frames']==1
 route_source_success(sample_id='s2',**{**base,'requested_frame_index':1,'actual_frame_index':1,'crop_relative_path':'crops/d.jpg'});r.flush();assert r.counts()['source_frames']==2 and r.counts()['source_crops']==2
@pytest.mark.parametrize('field,value',[('sample_id','other'),('dataset','msu_mfsd'),('source_record_id','other-video'),('requested_frame_index',9)])
def test_pair_identity_mismatches_do_not_upsert(tmp_path,monkeypatch,field,value):
 c=ctx(tmp_path);record=source(tmp_path);frame=routing.build_source_frame_record(c,record,sample_id='s',source_media_type='image_sequence',source_relative_identifier='source',requested_frame_index=0,actual_frame_index=0,frame_width=10,frame_height=10,selected_frame_reference='source#0',decoder_backend='opencv')
 crop=routing.build_source_crop_record(c,record,sample_id='s',source_media_type='image_sequence',requested=0,actual=0,timestamp=None,width=10,height=10,bbox=[1,1,5,5],landmarks=[(1,1)]*5,score=.9,count=1,box=[0,0,6,6],padding=.25,cw=6,ch=6,path='c.jpg',sha='a'*64).model_copy(update={field:value})
 monkeypatch.setattr(routing,'build_source_frame_record',lambda *a,**k:frame);monkeypatch.setattr(routing,'build_source_crop_record',lambda *a,**k:crop)
 repo=Mock();repo.counts.return_value={'source_frames':0,'source_crops':0}
 with pytest.raises(SourceRoutingConsistencyError,match=field):route_source_success(repository=repo,context=c,canonical_record=record,sample_id='s',requested_frame_index=0,actual_frame_index=0,source_media_type='image_sequence',timestamp_ms=None,frame_width=10,frame_height=10,decoder_backend='opencv',bbox=[1,1,5,5],landmarks=[(1,1)]*5,crop_box=[0,0,6,6],crop_relative_path='c.jpg',crop_sha256='a'*64)
 repo.upsert_source_success.assert_not_called()
def test_frame_converter_exception_stops_routing(tmp_path,monkeypatch):
 monkeypatch.setattr(routing,'build_source_frame_record',Mock(side_effect=MissingCanonicalMetadataError('synthetic missing source metadata')));crop=Mock();monkeypatch.setattr(routing,'build_source_crop_record',crop);repo=Mock()
 with pytest.raises(MissingCanonicalMetadataError):route_source_success(repository=repo,context=ctx(tmp_path),canonical_record=source(tmp_path),sample_id='s',requested_frame_index=0,actual_frame_index=0,source_media_type='image_sequence',timestamp_ms=None,frame_width=1,frame_height=1,decoder_backend='x')
 crop.assert_not_called();repo.upsert_source_success.assert_not_called()
def test_crop_converter_exception_stops_upsert(tmp_path,monkeypatch):
 c=ctx(tmp_path);rec=source(tmp_path);frame=routing.build_source_frame_record(c,rec,sample_id='s',source_media_type='image_sequence',source_relative_identifier='x',requested_frame_index=0,actual_frame_index=0,frame_width=1,frame_height=1,selected_frame_reference='x',decoder_backend='x');f=Mock(return_value=frame);g=Mock(side_effect=InvalidGeometryError('synthetic invalid crop geometry'));monkeypatch.setattr(routing,'build_source_frame_record',f);monkeypatch.setattr(routing,'build_source_crop_record',g);repo=Mock()
 with pytest.raises(InvalidGeometryError):route_source_success(repository=repo,context=c,canonical_record=rec,sample_id='s',requested_frame_index=0,actual_frame_index=0,source_media_type='image_sequence',timestamp_ms=None,frame_width=1,frame_height=1,decoder_backend='x')
 assert f.call_count==g.call_count==1;repo.upsert_source_success.assert_not_called()
def test_upsert_exception_propagates_after_converters(tmp_path,monkeypatch):
 c=ctx(tmp_path);rec=source(tmp_path);frame=routing.build_source_frame_record(c,rec,sample_id='s',source_media_type='image_sequence',source_relative_identifier='x',requested_frame_index=0,actual_frame_index=0,frame_width=10,frame_height=10,selected_frame_reference='x',decoder_backend='x');crop=routing.build_source_crop_record(c,rec,sample_id='s',source_media_type='image_sequence',requested=0,actual=0,timestamp=None,width=10,height=10,bbox=[1,1,5,5],landmarks=[(1,1)]*5,score=.9,count=1,box=[0,0,6,6],padding=.2,cw=6,ch=6,path='c.jpg',sha='a'*64);monkeypatch.setattr(routing,'build_source_frame_record',Mock(return_value=frame));monkeypatch.setattr(routing,'build_source_crop_record',Mock(return_value=crop));repo=Mock();repo.upsert_source_success.side_effect=RuntimeError('synthetic repository upsert failure')
 with pytest.raises(RuntimeError):route_source_success(repository=repo,context=c,canonical_record=rec,sample_id='s',requested_frame_index=0,actual_frame_index=0,source_media_type='image_sequence',timestamp_ms=None,frame_width=10,frame_height=10,decoder_backend='x')
 repo.upsert_source_success.assert_called_once()
def test_conflicting_duplicate_is_rejected_and_persisted_row_retained(tmp_path):
 c=ctx(tmp_path);r=ManifestRepository(tmp_path/'full_preprocessing'/'manifests',{'manifest_schema_version':'x'}).initialize();base=dict(repository=r,context=c,canonical_record=source(tmp_path),sample_id='s',requested_frame_index=0,actual_frame_index=0,source_media_type='image_sequence',timestamp_ms=None,frame_width=10,frame_height=10,decoder_backend='opencv',selected_frame_reference='source#0',bbox=[1,1,5,5],landmarks=[(1,1)]*5,detection_score=.9,detected_face_count=1,crop_box=[0,0,6,6],crop_width=6,crop_height=6,crop_relative_path='crops/c.jpg',crop_sha256='a'*64)
 route_source_success(**base);r.flush();r.initialize()
 with pytest.raises(ValueError,match='conflicting duplicate'):route_source_success(**{**base,'crop_sha256':'b'*64})
 r.flush();r.initialize();assert r.counts()=={'source_frames':1,'source_crops':1,'target_frames':0,'target_crops':0,'preprocessing_failures':0}
