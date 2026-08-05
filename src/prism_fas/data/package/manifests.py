from __future__ import annotations
import os, tempfile
from pathlib import Path
from typing import Any
import pyarrow as pa, pyarrow.parquet as pq
from prism_fas.data.manifests.parquet_writer import _replace_with_retry
from prism_fas.utils.core import sha256_file

_QUALITY=[("blur_laplacian_variance",pa.float64()),("brightness_mean",pa.float64()),("brightness_std",pa.float64()),
          ("contrast_michelson",pa.float64()),("saturation_mean",pa.float64()),("face_size_ratio",pa.float64())]
_STATUS=[("parsing_status",pa.string()),("pose_status",pa.string()),("visibility_status",pa.string()),("identity_status",pa.string())]
# samples.parquet carries no label/subject column at all: target rows must pass
# the existing target-leakage validator, which rejects those keys outright.
SAMPLES_SCHEMA=pa.schema([("sample_id",pa.string()),("dataset",pa.string()),("dataset_role",pa.string()),("project_split",pa.string()),
    ("source_record_id",pa.string()),("requested_frame_index",pa.int64()),("actual_frame_index",pa.int64()),
    ("image_relative_path",pa.string()),("crop_sha256",pa.string()),("prior_relative_path",pa.string()),("prior_sha256",pa.string()),
    ("prior_bytes",pa.int64()),("source_media_type",pa.string()),("image_format",pa.string()),
    ("frame_width",pa.int64()),("frame_height",pa.int64()),("crop_width",pa.int64()),("crop_height",pa.int64()),
    ("detection_score",pa.float64()),("detected_face_count",pa.int64()),*_QUALITY,
    ("quality_schema_version",pa.string()),("prior_schema_version",pa.string()),("package_schema_version",pa.string()),
    ("preprocessing_version",pa.string()),("preprocessing_config_hash",pa.string()),("detector_model_sha256",pa.string()),*_STATUS])
SOURCE_SPLIT_SCHEMA=pa.schema([("sample_id",pa.string()),("dataset",pa.string()),("source_record_id",pa.string()),
    ("subject_id",pa.string()),("official_split",pa.string()),("label_live_spoof",pa.string()),("project_split",pa.string()),
    ("image_relative_path",pa.string()),("prior_relative_path",pa.string()),("crop_sha256",pa.string()),("prior_sha256",pa.string()),
    ("package_schema_version",pa.string())])
TARGET_FEATURES_SCHEMA=pa.schema([("sample_id",pa.string()),("dataset",pa.string()),("source_record_id",pa.string()),
    ("project_split",pa.string()),("image_relative_path",pa.string()),("prior_relative_path",pa.string()),
    ("crop_sha256",pa.string()),("prior_sha256",pa.string()),("source_media_type",pa.string()),
    ("frame_width",pa.int64()),("frame_height",pa.int64()),("crop_width",pa.int64()),("crop_height",pa.int64()),
    ("detection_score",pa.float64()),("detected_face_count",pa.int64()),*_QUALITY,("package_schema_version",pa.string())])
PRIORS_INDEX_SCHEMA=pa.schema([("sample_id",pa.string()),("prior_relative_path",pa.string()),("prior_sha256",pa.string()),
    ("prior_bytes",pa.int64()),("prior_schema_version",pa.string()),("quality_schema_version",pa.string()),
    ("crop_sha256",pa.string()),("preprocessing_version",pa.string()),("preprocessing_config_hash",pa.string()),
    ("detector_model_sha256",pa.string()),*_STATUS])
SHARDS_INDEX_SCHEMA=pa.schema([("shard_filename",pa.string()),("split",pa.string()),("first_sample_id",pa.string()),
    ("last_sample_id",pa.string()),("row_count",pa.int64()),("byte_size",pa.int64()),("sha256",pa.string()),
    ("package_schema_version",pa.string())])
MANIFEST_SCHEMAS={"samples":SAMPLES_SCHEMA,"source_train":SOURCE_SPLIT_SCHEMA,"source_dev":SOURCE_SPLIT_SCHEMA,
                  "target_test_features":TARGET_FEATURES_SCHEMA,"priors_index":PRIORS_INDEX_SCHEMA,"shards_index":SHARDS_INDEX_SCHEMA}
def write_manifest(path:Path,rows:list[dict[str,Any]],schema:pa.Schema,metadata:dict[str,str],*,sort_key:str="sample_id")->str:
    """Deterministic atomic Parquet write; returns the file SHA-256."""
    ordered=sorted(rows,key=lambda row:row[sort_key])
    keys=[key for key in {k for row in ordered for k in row} if key not in schema.names]
    if keys: raise ValueError(f"unexpected manifest columns: {sorted(keys)}")
    table=pa.Table.from_pylist([{name:row.get(name) for name in schema.names} for row in ordered],schema=schema)
    table=table.replace_schema_metadata({str(k).encode():str(v).encode() for k,v in metadata.items()})
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix=path.name+".",suffix=".tmp",dir=path.parent); os.close(fd)
    try: pq.write_table(table,temporary); _replace_with_retry(temporary,path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    read=pq.read_table(path)
    if read.num_rows!=len(ordered): raise RuntimeError(f"manifest read-back mismatch for {path.name}")
    del read
    return sha256_file(path)
def read_manifest(path:Path)->list[dict[str,Any]]:
    table=pq.read_table(path); rows=table.to_pylist(); del table; return rows
