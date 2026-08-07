"""Deterministic, pickle-free NPZ archives.

`np.savez_compressed` stamps the local wall clock into every zip entry, so two
byte-identical builds would disagree. Members are written in sorted order with a
fixed member date, exactly as the M8 bank and the M9 prototype export already do.
"""
from __future__ import annotations
import hashlib, io, zipfile
from pathlib import Path
from typing import Any
import numpy as np

FIXED_MEMBER_DATE = (1980, 1, 1, 0, 0, 0)


def encode_arrays(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=FIXED_MEMBER_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload.getvalue())
    return buffer.getvalue()


def write_arrays_npz(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    data = encode_arrays(arrays)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return {"bytes": len(data), "file_sha256": hashlib.sha256(data).hexdigest()}


def read_arrays_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as handle:
        return {name: np.asarray(handle[name]) for name in handle.files}
