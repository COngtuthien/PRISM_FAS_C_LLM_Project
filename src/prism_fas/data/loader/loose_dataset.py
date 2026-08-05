from __future__ import annotations
from pathlib import Path
import numpy as np
from prism_fas.data.package.priors import load_prior
from .config import INFERENCE_SPLIT, LoaderConfig
from .contracts import CanonicalSourceSample, CanonicalTargetSample, SampleContractError
from .package_index import PackageIndex, open_package
from .transforms import geometry_from_arrays, read_image

class CanonicalPackageDataset:
    """Map-style dataset over the loose files of a validated M3B package.

    Deterministic index order (sorted by sample_id); never mutates the package.
    """
    def __init__(self,package_root:Path,split:str,config:LoaderConfig,*,mode:str,index:PackageIndex|None=None):
        self.config=config; self.mode=mode
        self.index=index or open_package(package_root,split,config,mode=mode)
        self.split=self.index.split; self.root=self.index.root
        self.is_target=self.split==INFERENCE_SPLIT
    def __len__(self)->int: return len(self.index)
    @property
    def sample_ids(self)->tuple[str,...]: return self.index.sample_ids
    def index_of(self,sample_id:str)->int:
        try: return self.index.sample_ids.index(sample_id)
        except ValueError: raise KeyError(f"sample not present in split {self.split!r}") from None
    def __getitem__(self,position:int):
        row=self.index.rows[position]
        return self._target(row) if self.is_target else self._source(row)
    def _load(self,row:dict):
        image_path=self.root/row["image_relative_path"]; prior_path=self.root/row["prior_relative_path"]
        if not image_path.is_file(): raise SampleContractError(f"package image missing for sample {row['sample_id']}")
        if not prior_path.is_file(): raise SampleContractError(f"package prior missing for sample {row['sample_id']}")
        image=read_image(image_path,self.config.image)
        arrays=load_prior(prior_path)            # np.load(..., allow_pickle=False)
        return image,arrays,geometry_from_arrays(arrays)
    def _source(self,row:dict)->CanonicalSourceSample:
        image,arrays,geometry=self._load(row)
        embedding=arrays.get("identity_embedding")
        available=embedding is not None
        sample=CanonicalSourceSample(sample_id=row["sample_id"],dataset=row["dataset"],project_split=row["project_split"],
            source_record_id=row["source_record_id"],requested_frame_index=int(row.get("requested_frame_index",0)),
            actual_frame_index=int(row.get("actual_frame_index",0)),label=row["label_live_spoof"],
            class_target=self.config.label_to_index(row["label_live_spoof"]),image=image,geometry=geometry,
            identity_embedding=embedding.astype(np.float32) if available else None,identity_available=bool(available),
            crop_sha256=row["crop_sha256"],prior_sha256=row["prior_sha256"],
            source_media_type=row.get("source_media_type",""))
        sample.validate(); return sample
    def _target(self,row:dict)->CanonicalTargetSample:
        image,arrays,geometry=self._load(row)
        if "identity_embedding" in arrays:
            raise SampleContractError(f"target prior unexpectedly carries an identity embedding: {row['sample_id']}")
        sample=CanonicalTargetSample(sample_id=row["sample_id"],dataset=row["dataset"],project_split=row["project_split"],
            source_record_id=row["source_record_id"],image=image,geometry=geometry,crop_sha256=row["crop_sha256"],
            prior_sha256=row["prior_sha256"],source_media_type=row.get("source_media_type",""))
        sample.validate(); return sample
def build_dataset(package_root:Path,split:str,config:LoaderConfig,*,mode:str)->CanonicalPackageDataset:
    return CanonicalPackageDataset(package_root,split,config,mode=mode)
