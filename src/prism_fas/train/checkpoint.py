from __future__ import annotations
import hashlib, os, tempfile
from pathlib import Path

CHECKPOINT_SCHEMA_VERSION="m5-b00-ckpt-v1"
class CheckpointContractError(ValueError):
    """A checkpoint is unreadable or incompatible with the current run."""
def save_checkpoint(path:Path,payload:dict)->str:
    """Atomic checkpoint write: temp file -> verify readable/non-empty -> os.replace."""
    import torch
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    payload={**payload,"schema_version":CHECKPOINT_SCHEMA_VERSION}
    fd,temporary=tempfile.mkstemp(prefix=path.name+".",suffix=".tmp",dir=path.parent); os.close(fd)
    try:
        torch.save(payload,temporary)
        if os.path.getsize(temporary)==0: raise CheckpointContractError("checkpoint temp file is empty")
        torch.load(temporary,map_location="cpu",weights_only=False)
        digest=hashlib.sha256(Path(temporary).read_bytes()).hexdigest()
        for attempt in range(12):
            try: os.replace(temporary,path); break
            except PermissionError:
                if attempt==11: raise
                import gc,time; gc.collect(); time.sleep(.15)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    return digest
def checkpoint_sha256(path:Path)->str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_checkpoint(path:Path,*,config_hash:str|None=None,package_identity:str|None=None,
                    model_name:str|None=None,label_mapping:dict|None=None)->dict:
    """Load a checkpoint, refusing to resume across incompatible runs."""
    import torch
    path=Path(path)
    if not path.is_file(): raise CheckpointContractError(f"checkpoint not found: {path.name}")
    payload=torch.load(path,map_location="cpu",weights_only=False)
    if payload.get("schema_version")!=CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointContractError("checkpoint schema version mismatch")
    for name,expected,actual in (("config hash",config_hash,payload.get("config_hash")),
                                 ("package identity",package_identity,payload.get("package_content_identity")),
                                 ("model name",model_name,payload.get("model_name"))):
        if expected is not None and actual!=expected:
            raise CheckpointContractError(f"resume blocked: {name} mismatch")
    if label_mapping is not None and payload.get("label_mapping")!=label_mapping:
        raise CheckpointContractError("resume blocked: label mapping mismatch")
    return payload
