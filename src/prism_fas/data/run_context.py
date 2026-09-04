from __future__ import annotations
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, model_validator

class M2OutputLayout(BaseModel):
    model_config=ConfigDict(extra='forbid')
    output_root:Path; crops_root:Path; frames_root:Path; manifests_root:Path; state_root:Path; reports_root:Path; logs_root:Path
    @classmethod
    def from_root(cls,root:Path)->'M2OutputLayout':
        root=root.resolve();return cls(output_root=root,crops_root=root/'crops',frames_root=root/'frames',manifests_root=root/'manifests',state_root=root/'state',reports_root=root/'reports',logs_root=root/'logs')
class PreprocessingRunContext(BaseModel):
    model_config=ConfigDict(extra='forbid',arbitrary_types_allowed=True)
    project_root:Path;work_root:Path;run_profile:Literal['small_acceptance','full_preprocessing','target_eval_v2'];output_namespace:str;output_root:Path;crops_root:Path;frames_root:Path;manifests_root:Path;state_root:Path;reports_root:Path;logs_root:Path;run_id:str;dataset:str;dataset_role:Literal['source','target'];preprocessing_version:str;preprocessing_config_hash:str;detector_model_path:Path;detector_model_sha256:str;detector_input_size:int;detector_threshold:float;all_records:bool;record_limit:int|None;sample_limit:int|None;resume:bool;dry_run:bool;partial_full_profile:bool;command:str
    # Additive, default-safe: every historical/CASIA/MSU context is built
    # without passing this, so it defaults to 'required' -- byte-identical
    # behavior to before this field existed. Only an explicitly-constructed
    # context (E7-B's SiW-as-source path) may set 'optional_unverifiable',
    # since the exact permitted local SiW-Mv2 release carries no canonical
    # subject mapping and none may ever be fabricated.
    source_metadata_policy:Literal['required','optional_unverifiable']='required'
    @model_validator(mode='after')
    def safe(self):
        root=self.output_root.resolve()
        if self.run_profile=='full_preprocessing' and self.output_namespace!='full_preprocessing':raise ValueError('full profile namespace mismatch')
        if self.run_profile=='full_preprocessing' and 'm2a' in root.parts:raise ValueError('full profile cannot use small namespace')
        # The M10 target profile is additive: it may write ONLY under its own
        # namespace, so it cannot reach the frozen `full_preprocessing`/`m2a`
        # artifacts that every M8/M9 identity is bound to.
        if self.run_profile=='target_eval_v2':
            if self.output_namespace!='target_eval_v2':raise ValueError('target profile namespace mismatch')
            if {'m2a','full_preprocessing','full_preprocessing_v2'} & set(root.parts):
                raise ValueError('the target evaluation profile cannot write into a frozen M2 namespace')
            if self.dataset!='siw_mv2':raise ValueError('the target evaluation profile is siw_mv2 only')
            if self.dataset_role!='target':raise ValueError('the target evaluation profile is target-role only')
        for child in [self.crops_root,self.frames_root,self.manifests_root,self.state_root]:
            if not str(child.resolve()).startswith(str(root)):raise ValueError('layout path escapes profile root')
        return self

def build_preprocessing_run_context(paths, cfg, profile, dataset, run_id, *, all_records=False, limit_records=None, limit_samples=None, resume=False, dry_run=False, partial=False, root=None):
    """The one constructor for a production preprocessing context.

    It lived in `cli/main.py`, which meant the pre-C4 preparation path could not
    reach it without importing the whole CLI — and so it did not, and drifted
    onto the legacy `m2a` runner instead. It lives beside the context it builds
    now; the CLI imports it from here.
    """
    from prism_fas.data.preprocess_m2 import resolve_detector_path
    from prism_fas.data.run_profiles import profile_root
    from prism_fas.utils.core import sha256_file

    root=root or profile_root(paths.work_root,cfg.preprocessing_version,cfg.config_hash,profile); layout=M2OutputLayout.from_root(root); role='target' if dataset=='siw_mv2' else 'source'
    return PreprocessingRunContext(project_root=paths.project_root,work_root=paths.work_root,run_profile=profile.name,output_namespace=profile.output_namespace,output_root=layout.output_root,crops_root=layout.crops_root,frames_root=layout.frames_root,manifests_root=layout.manifests_root,state_root=layout.state_root,reports_root=layout.reports_root,logs_root=layout.logs_root,run_id=run_id or f'{profile.name}-{dataset}',dataset=dataset,dataset_role=role,preprocessing_version=cfg.preprocessing_version,preprocessing_config_hash=cfg.config_hash,detector_model_path=resolve_detector_path(cfg.scrfd_model_path),detector_model_sha256=sha256_file(resolve_detector_path(cfg.scrfd_model_path)),detector_input_size=cfg.scrfd_input_size,detector_threshold=cfg.detection_threshold,all_records=all_records,record_limit=limit_records,sample_limit=limit_samples,resume=resume,dry_run=dry_run,partial_full_profile=partial,command='prism data preprocess run')
