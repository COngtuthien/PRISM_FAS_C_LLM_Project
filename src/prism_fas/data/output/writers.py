from __future__ import annotations
import os
from pathlib import Path
import cv2, numpy as np
CROP_WRITER_BACKEND="opencv"
class OutputWriteError(RuntimeError):
    def __init__(self,message:str="face crop output could not be written",*,backend:str=CROP_WRITER_BACKEND):
        super().__init__(message); self.backend=backend
def _discard(path:Path)->None:
    # Cleanup must never mask the original OutputWriteError.
    try: path.unlink(missing_ok=True)
    except OSError: pass
def discard_crop_artifact(target:Path)->None:
    """Best-effort removal of a crop artifact and its temporary sibling.

    Used when a written crop must not survive a later failure in the same
    sample; never raises, so it cannot mask the originating error.
    """
    target=Path(target); _discard(target); _discard(target.with_suffix('.tmp'+target.suffix))
def write_crop_image(image:np.ndarray,target:Path,*,jpeg_quality:int=95)->Path:
    """Materialize an in-memory crop as an output artifact.

    Every filesystem/encoder failure inside this boundary is converted to
    OutputWriteError; no partial or zero-byte artifact survives a failure.
    """
    target=Path(target); temporary=target.with_suffix('.tmp'+target.suffix); replaced=False
    try:
        try: target.parent.mkdir(parents=True,exist_ok=True)
        except OSError as exc: raise OutputWriteError(backend=CROP_WRITER_BACKEND) from exc
        try: written=cv2.imwrite(str(temporary),image,[cv2.IMWRITE_JPEG_QUALITY,jpeg_quality])
        except (cv2.error,OSError,ValueError,TypeError) as exc: raise OutputWriteError(backend=CROP_WRITER_BACKEND) from exc
        if not written: raise OutputWriteError(backend=CROP_WRITER_BACKEND)
        if not temporary.exists() or temporary.stat().st_size==0: raise OutputWriteError(backend=CROP_WRITER_BACKEND)
        try: os.replace(temporary,target); replaced=True
        except OSError as exc: raise OutputWriteError(backend=CROP_WRITER_BACKEND) from exc
        if not target.exists() or target.stat().st_size==0: raise OutputWriteError(backend=CROP_WRITER_BACKEND)
    except OutputWriteError:
        _discard(temporary)
        if replaced: _discard(target)
        raise
    return target
