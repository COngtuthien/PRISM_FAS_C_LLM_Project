from __future__ import annotations
import gc, json, os, tempfile, time, types
from pathlib import Path
from typing import Any, Type
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel
from typing import get_args, get_origin, Union

def _canonical(row: dict[str,Any]) -> str: return json.dumps(row,sort_keys=True,separators=(",",":"),default=str)
def _arrow_type(annotation: Any) -> pa.DataType:
    origin=get_origin(annotation)
    if origin in (list,): return pa.list_(pa.string())
    if origin is Union or origin is types.UnionType:
        args=[a for a in get_args(annotation) if a is not type(None)]
        return _arrow_type(args[0]) if args else pa.string()
    if annotation is int: return pa.int64()
    if annotation is float: return pa.float64()
    if annotation is bool: return pa.bool_()
    return pa.string()
def _replace_with_retry(tmp: str, path: Path, attempts: int=12, delay: float=.15) -> None:
    """Atomic rename that tolerates transient Windows sharing violations.

    A reader handle on the destination (pyarrow read-back, or an on-access
    virus scanner) makes os.replace raise PermissionError until it is closed.
    """
    for attempt in range(attempts):
        try: os.replace(tmp,path); return
        except PermissionError:
            if attempt==attempts-1: raise
            gc.collect(); time.sleep(delay)
def write_parquet_atomic(path: Path, rows: list[dict[str,Any]], model: Type[BaseModel], metadata: dict[str,str], failure: bool=False) -> dict[str,Any]:
    columns=list(model.model_fields)
    validated=[model.model_validate(row).model_dump(mode="json") for row in rows]
    seen: dict[str,str]={}; unique=[]
    for row in validated:
        key=str(row.get("sample_id")) if row.get("sample_id") is not None else _canonical(row)
        encoded=_canonical(row)
        if key in seen:
            if seen[key] != encoded: raise ValueError(f"conflicting duplicate sample_id: {key}")
            continue
        seen[key]=encoded; unique.append(row)
    unique.sort(key=(lambda r:(r["dataset"],r["source_record_id"],r["requested_frame_index"] if r["requested_frame_index"] is not None else -1,r["error_code"])) if failure else lambda r:r["sample_id"])
    normalized=[{c:r.get(c) for c in columns} for r in unique]
    schema=pa.schema([pa.field(c,_arrow_type(model.model_fields[c].annotation),nullable=True) for c in columns])
    table=pa.Table.from_pylist(normalized,schema=schema)
    table=table.replace_schema_metadata({str(k).encode():str(v).encode() for k,v in metadata.items()})
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+".",suffix=".tmp",dir=path.parent); os.close(fd)
    try:
        pq.write_table(table,tmp); _replace_with_retry(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    read=pq.read_table(path)
    if read.num_rows != len(normalized) or read.column_names != columns: raise RuntimeError("Parquet read-back validation failed")
    summary={"rows":read.num_rows,"columns":read.column_names,"metadata":{k.decode():v.decode() for k,v in (read.schema.metadata or {}).items()},"duplicates_collapsed":len(validated)-len(unique)}
    del read
    return summary
