from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from prism_fas.utils.core import atomic_json_write
from .config import SPLITS
from .manifests import read_manifest
from .quality import QUALITY_NAMES

def _percentiles(values:list[float])->dict[str,float]:
    if not values: return {}
    array=np.asarray(values,dtype=np.float64)
    return {"min":float(array.min()),"p05":float(np.percentile(array,5)),"p50":float(np.percentile(array,50)),
            "p95":float(np.percentile(array,95)),"max":float(array.max()),"mean":float(array.mean())}
def build_audit_report(package_root:Path,*,lock:dict,resume:dict|None=None,m2_failures:list[dict]|None=None)->dict:
    """Concise package audit. Carries no target labels or private metadata."""
    root=Path(package_root); samples=read_manifest(root/"manifests"/"samples.parquet")
    shards=read_manifest(root/"manifests"/"shards_index.parquet")
    per_dataset={};per_split={}
    for row in samples:
        per_dataset[row["dataset"]]=per_dataset.get(row["dataset"],0)+1
        per_split[row["project_split"]]=per_split.get(row["project_split"],0)+1
    label_balance={}
    for split in ("source_train","source_dev"):
        path=root/"manifests"/f"{split}.parquet"
        if not path.is_file(): continue
        rows=read_manifest(path); balance={}
        for row in rows: balance[(row["dataset"],row["label_live_spoof"])]=balance.get((row["dataset"],row["label_live_spoof"]),0)+1
        label_balance[split]={f"{dataset}/{label}":count for (dataset,label),count in sorted(balance.items())}
    quality={name:_percentiles([row[name] for row in samples]) for name in QUALITY_NAMES}
    quality_by_split={split:{name:_percentiles([row[name] for row in samples if row["project_split"]==split]) for name in ("blur_laplacian_variance","brightness_mean")} for split in SPLITS}
    report={"package_id":lock.get("package_id"),"package_schema_version":lock.get("package_schema_version"),
        "status":lock.get("status"),"generated_at":datetime.now(timezone.utc).isoformat(),
        "total_samples":len(samples),"per_dataset_counts":dict(sorted(per_dataset.items())),
        "per_split_counts":dict(sorted(per_split.items())),"source_label_balance":label_balance,
        "quality_summary":quality,"quality_by_split":quality_by_split,
        "shards":{"count":len(shards),"total_bytes":sum(row["byte_size"] for row in shards),
                  "per_split":{split:sum(1 for row in shards if row["split"]==split) for split in SPLITS},
                  "rows":[{"shard_filename":row["shard_filename"],"split":row["split"],"row_count":row["row_count"],"byte_size":row["byte_size"]} for row in shards]},
        "inherited_m2_failures":m2_failures or [],
        "deferred_priors":lock.get("deferred_priors"),
        "target_isolation":lock.get("target_isolation"),
        "package_hashes":lock.get("manifest_sha256"),
        "content_identity_sha256":lock.get("content_identity_sha256"),
        "resume":resume or {}}
    atomic_json_write(root/"audit"/"data_report.json",report)
    (root/"audit"/"data_report.md").write_text(_markdown(report),encoding="utf-8")
    return report
def _markdown(report:dict)->str:
    lines=[f"# M3A package audit — {report['package_id']}","",
           f"- Status: **{report['status']}**",f"- Schema: `{report['package_schema_version']}`",
           f"- Total samples: **{report['total_samples']}**","","## Counts","",
           "| Dataset | Samples |","|---|---|"]
    lines+= [f"| {k} | {v} |" for k,v in report["per_dataset_counts"].items()]
    lines+=["","| Split | Samples |","|---|---|"]+[f"| {k} | {v} |" for k,v in report["per_split_counts"].items()]
    lines+=["","## Source label balance",""]
    for split,balance in report["source_label_balance"].items():
        lines+= [f"- `{split}`: "+", ".join(f"{k}={v}" for k,v in balance.items())]
    lines+=["","## Quality metrics","","| Metric | min | p05 | p50 | p95 | max |","|---|---|---|---|---|---|"]
    for name,stats in report["quality_summary"].items():
        lines.append(f"| {name} | {stats['min']:.4f} | {stats['p05']:.4f} | {stats['p50']:.4f} | {stats['p95']:.4f} | {stats['max']:.4f} |")
    lines+=["","## Shards","",f"- Count: {report['shards']['count']}",f"- Total bytes: {report['shards']['total_bytes']}",
            f"- Per split: {report['shards']['per_split']}","","## Target isolation","",
            f"- {report['target_isolation']}","","## Deferred model-dependent priors","",
            f"- {report['deferred_priors']}","","## Inherited M2 failures","",
            f"- {len(report['inherited_m2_failures'])} failed sample(s) from M2 (not packaged)","",
            "## Resume","",f"- {report['resume']}",""]
    return "\n".join(lines)
