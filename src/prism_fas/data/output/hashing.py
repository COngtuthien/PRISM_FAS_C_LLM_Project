from __future__ import annotations
import re
from pathlib import Path
from prism_fas.utils.core import sha256_file
CROP_HASH_BACKEND="sha256"
_DIGEST=re.compile(r'[0-9a-f]{64}')
class HashComputationError(RuntimeError):
    def __init__(self,message:str="face crop hash could not be computed",*,backend:str=CROP_HASH_BACKEND):
        super().__init__(message); self.backend=backend
def validate_digest(digest)->str:
    """Normalize and validate a SHA-256 digest for the crop manifest."""
    if isinstance(digest,(bytes,bytearray)):
        try: digest=digest.decode('ascii')
        except UnicodeDecodeError as exc: raise HashComputationError() from exc
    if not isinstance(digest,str): raise HashComputationError()
    normalized=digest.strip().lower()
    if not _DIGEST.fullmatch(normalized): raise HashComputationError()
    return normalized
def hash_crop_artifact(path:Path)->str:
    """Compute the SHA-256 digest of a materialized crop artifact.

    Every read/backend/digest failure inside this boundary becomes
    HashComputationError; unrelated exceptions keep their own contract.
    """
    path=Path(path)
    try:
        if not path.is_file(): raise HashComputationError()
        digest=sha256_file(path)
    except HashComputationError: raise
    except (OSError,ValueError,TypeError) as exc: raise HashComputationError() from exc
    return validate_digest(digest)
