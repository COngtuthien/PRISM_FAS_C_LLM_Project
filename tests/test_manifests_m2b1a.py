from pathlib import Path
import pyarrow.parquet as pq
import pytest
from prism_fas.data.manifests.schemas import SourceFrameRecord, SourceCropRecord, TargetFrameRecord, PreprocessingFailureRecord
from prism_fas.data.manifests.leakage import find_target_leakage
from prism_fas.data.manifests.parquet_writer import write_parquet_atomic

def frame(): return dict(sample_id="a",dataset="casia_fasd",video_id="v",source_record_id="v",source_media_type="video_file",source_relative_identifier="x",requested_frame_index=0,actual_frame_index=0,timestamp_ms=None,frame_width=10,frame_height=10,selected_frame_reference="x#frame=0",materialized_frame_relative_path=None,source_fingerprint="f",frame_fingerprint=None,decoder_backend="opencv",adapter_version="v1",sampling_version="s",preprocessing_version="p",preprocessing_config_hash="h",status="success",warning_codes=[])
def crop():
    d=dict(sample_id="a",dataset="casia_fasd",video_id="v",source_record_id="v",source_media_type="video_file",requested_frame_index=0,actual_frame_index=0,timestamp_ms=None,frame_width=10,frame_height=10,bbox_x1=0.,bbox_y1=0.,bbox_x2=5.,bbox_y2=5.,detection_score=.8,detected_face_count=1,crop_x1=0,crop_y1=0,crop_x2=5,crop_y2=5,requested_crop_padding=.2,effective_crop_padding=.2,crop_width=5,crop_height=5,crop_relative_path="x.jpg",crop_sha256="h",detector_name="scrfd",detector_model_sha256="h",detector_provider="CPUExecutionProvider",detector_input_size=320,detector_threshold=.5,preprocessing_version="p",preprocessing_config_hash="h",status="success")
    for i in range(5): d[f"landmark_{i}_x"]=float(i);d[f"landmark_{i}_y"]=float(i)
    return d
def test_source_frame_strict_and_valid():
    d=frame(); d.update(subject_id="1",official_split="train",label_live_spoof="live"); assert SourceFrameRecord.model_validate(d).sample_id=="a"
    d["unexpected"]=1
    with pytest.raises(Exception): SourceFrameRecord.model_validate(d)
def test_crop_geometry_and_target_isolation():
    d=crop();d.update(subject_id="1",official_split="train",label_live_spoof="spoof"); assert SourceCropRecord.model_validate(d).bbox_x2==5
    d["bbox_x2"]=0
    with pytest.raises(Exception): SourceCropRecord.model_validate(d)
    assert find_target_leakage({"nested":{"attack_type":"print"}})
    assert find_target_leakage('{"label_live_spoof":"live"}')
def test_target_rejects_source_fields():
    d=frame();d["label_live_spoof"]="live"
    with pytest.raises(Exception): TargetFrameRecord.model_validate(d)
def test_writer_atomic_sort_dedup_conflict_and_empty(tmp_path:Path):
    one=frame(); one.update(subject_id="1",official_split="train",label_live_spoof="live")
    two={**one,"sample_id":"b"}; target=tmp_path/"out.parquet"
    info=write_parquet_atomic(target,[two,one,one],SourceFrameRecord,{"x":"y"}); table=pq.read_table(target)
    assert info["duplicates_collapsed"]==1 and table.column("sample_id").to_pylist()==["a","b"] and table.schema.metadata[b"x"]==b"y"
    with pytest.raises(ValueError): write_parquet_atomic(target,[one,{**one,"status":"other"}],SourceFrameRecord,{})
    empty=tmp_path/"empty.parquet"; assert write_parquet_atomic(empty,[],PreprocessingFailureRecord,{})["rows"]==0
