from __future__ import annotations
from pathlib import Path
from typing import Any
import pyarrow.parquet as pq
from prism_fas.data.manifests.schemas import MODELS
from prism_fas.data.manifests.leakage import assert_target_safe
from prism_fas.data.manifests.parquet_writer import write_parquet_atomic

class ManifestRepository:
    """Profile-scoped incremental manifests; no small-run count assumptions."""
    def __init__(self,root:Path,metadata:dict[str,str]):
        self.root=root;self.metadata=metadata;self.rows={name:[] for name in MODELS}
    def initialize(self):
        self.root.mkdir(parents=True,exist_ok=True)
        for name,model in MODELS.items():
            path=self.root/f'{name}.parquet'
            if not path.exists(): write_parquet_atomic(path,[],model,self.metadata,name=='preprocessing_failures')
        return self
    def load(self):
        for name in MODELS:
            self.rows[name]=pq.read_table(self.root/f'{name}.parquet').to_pylist()
        return self
    def _upsert(self,name:str,row:dict[str,Any]):
        key=row.get('sample_id') or '|'.join(str(row.get(x,'')) for x in ('dataset','source_record_id','requested_frame_index','error_code','stage'))
        old={ (r.get('sample_id') or '|'.join(str(r.get(x,'')) for x in ('dataset','source_record_id','requested_frame_index','error_code','stage'))):r for r in self.rows[name]}
        if key in old and old[key] != row:
            raise ValueError(f'conflicting duplicate sample_id: {key}')
        old[key]=row;self.rows[name]=list(old.values())
    def upsert_source_success(self,frame,crop):self._upsert('source_frames',frame);self._upsert('source_crops',crop)
    def upsert_target_success(self,frame,crop):assert_target_safe(frame);assert_target_safe(crop);self._upsert('target_frames',frame);self._upsert('target_crops',crop)
    def upsert_failure(self,row):self._upsert('preprocessing_failures',row)
    def resolve_failure(self,sample_id:str):self.rows['preprocessing_failures']=[r for r in self.rows['preprocessing_failures'] if r.get('sample_id')!=sample_id]
    def flush(self):
        for n,m in MODELS.items():write_parquet_atomic(self.root/f'{n}.parquet',self.rows[n],m,self.metadata,n=='preprocessing_failures')
    def counts(self):return {n:len(v) for n,v in self.rows.items()}
