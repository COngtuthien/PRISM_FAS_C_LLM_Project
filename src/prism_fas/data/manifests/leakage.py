from __future__ import annotations
import json, re
from typing import Any

FORBIDDEN = {"subject_id","label","label_live_spoof","live","spoof","bona_fide","fake","attack","attack_id","attack_type","attack_family","presentation_type","taxonomy","ground_truth","private_metadata","protocol_outcome","evaluation_label","evaluation_split_label"}
def _tokens(value: str) -> set[str]:
    lower=value.lower(); return {lower, *filter(None,re.split(r"[^a-z0-9]+",lower))}
def find_target_leakage(value: Any, location: str="$", sample_id: str | None=None) -> list[dict[str,str|None]]:
    try:
        import pandas as pd
        if isinstance(value, pd.DataFrame):
            found=[]
            for name in list(value.columns)+list(value.index.names): found += find_target_leakage(str(name), f"{location}.column", sample_id)
            return found
    except ImportError: pass
    try:
        import pyarrow as pa
        if isinstance(value, pa.Schema):
            found=[]
            for field in value: found += find_target_leakage(field.name, f"{location}.field", sample_id)
            for k,v in (value.metadata or {}).items(): found += find_target_leakage(k.decode(), f"{location}.metadata_key", sample_id)+find_target_leakage(v.decode(errors="replace"), f"{location}.metadata_value", sample_id)
            return found
    except ImportError: pass
    if isinstance(value, dict):
        found=[]
        for key,item in value.items():
            matches=_tokens(str(key)) & FORBIDDEN
            found += [{"location":f"{location}.{key}","matched_key_token":m,"offending_value_summary":str(item)[:160],"sample_id":sample_id or str(value.get("sample_id",""))} for m in matches]
            found += find_target_leakage(item, f"{location}.{key}", sample_id or value.get("sample_id"))
        return found
    if isinstance(value,(list,tuple)): return [x for i,item in enumerate(value) for x in find_target_leakage(item,f"{location}[{i}]",sample_id)]
    if isinstance(value,str):
        text=value.strip()
        if text[:1] in "[{":
            try: return find_target_leakage(json.loads(text),location,sample_id)
            except json.JSONDecodeError: pass
        # Scalar values may legitimately include opaque source identifiers such as
        # Live_100.  Leakage is a prohibited field/key problem, not a filename
        # substring problem; structured JSON is handled above.
        return []
    return []
def assert_target_safe(value: Any) -> None:
    found=find_target_leakage(value)
    if found: raise ValueError(f"target leakage detected: {found}")
