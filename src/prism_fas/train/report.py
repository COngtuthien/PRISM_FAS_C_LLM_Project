from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from prism_fas.utils.core import atomic_json_write

TARGET_ISOLATION_STATEMENT=("Target predictions were generated under the frozen source-only calibration. "
    "Target labels were not accessed in M5, so target accuracy/FAS metrics are not reported.")
LIMITATIONS=["B00 is a plain ConvNeXt V2 binary baseline: it ignores the M3B parsing, pose, visibility and identity priors by design.",
    "One seed only; multi-seed comparison belongs to M10.",
    "No open-set/reject mechanism is trained, so reject_policy is disabled_for_b00.",
    "Target metrics cannot be computed because target labels are deliberately not present in the package."]
def _distribution(values:list[float])->dict:
    if not values: return {}
    array=np.asarray(values,dtype=np.float64)
    return {"count":int(array.size),"min":float(array.min()),"p05":float(np.percentile(array,5)),
            "p25":float(np.percentile(array,25)),"p50":float(np.percentile(array,50)),
            "p75":float(np.percentile(array,75)),"p95":float(np.percentile(array,95)),
            "max":float(array.max()),"mean":float(array.mean())}
def build_report(run_root:Path,*,run_info:dict,training:dict,calibration:dict,source_metrics:dict,
                 source_video:list[dict],target_rows:list[dict],target_video:list[dict],environment:dict)->dict:
    run_root=Path(run_root)
    summary={"run_id":run_root.name,"stage":"M5_B00","generated_at":datetime.now(timezone.utc).isoformat(),
        "identity":{k:run_info.get(k) for k in ("run_id","git_commit","package_id","package_content_identity",
            "config_hash","model_name","model_revision","model_weight_sha256","seed","best_checkpoint_sha256")},
        "environment":environment,"device":run_info.get("device"),
        "data_counts":{"source_train":run_info.get("source_train_count"),"source_dev":run_info.get("source_dev_count"),
                       "target_test":len(target_rows)},
        "training":{"epochs_run":training.get("epochs_run"),"global_step":training.get("global_step"),
                    "stopped_reason":training.get("stopped_reason"),"history":training.get("history"),
                    "selection_rule":training.get("selection_rule"),"best":training.get("best"),
                    "elapsed_seconds":training.get("elapsed_seconds")},
        "calibration":{k:calibration.get(k) for k in ("temperature","before","after","selected_threshold",
            "threshold_selection","calibration_hash","reject_policy")},
        "source_dev_frame_metrics":source_metrics,
        "source_dev_video_metrics":{"videos":len(source_video),
            "decisions":{d:sum(1 for v in source_video if v["decision"]==d) for d in ("live","spoof")},
            "score_distribution":_distribution([v["video_score"] for v in source_video])},
        "target":{"frame_predictions":len(target_rows),"videos":len(target_video),
            "frame_score_distribution":_distribution([r["p_spoof_calibrated"] for r in target_rows]),
            "frame_confidence_distribution":_distribution([r["confidence"] for r in target_rows]),
            "video_score_distribution":_distribution([v["video_score"] for v in target_video]),
            "decisions":{d:sum(1 for r in target_rows if r["decision"]==d) for d in ("live","spoof")},
            "labels_accessed":False,"metrics_reported":False},
        "target_isolation_statement":TARGET_ISOLATION_STATEMENT,
        "known_limitations":LIMITATIONS}
    atomic_json_write(run_root/"reports"/"summary.json",summary)
    (run_root/"reports"/"report.html").write_text(_html(summary),encoding="utf-8")
    return summary
def _table(rows:list[tuple])->str:
    if not rows: return "<p>none</p>"
    head="".join(f"<th>{name}</th>" for name in rows[0])
    body="".join("<tr>"+"".join(f"<td>{value}</td>" for value in row)+"</tr>" for row in rows[1:])
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
def _html(summary:dict)->str:
    history=summary["training"].get("history") or []
    curve=[("epoch","train_loss","source_dev_roc_auc","source_dev_nll")]+[
        (h["epoch"],round(h["train_loss"],5) if h.get("train_loss") is not None else "-",
         round(h["source_dev_roc_auc"],5),round(h["source_dev_nll"],5)) for h in history]
    metrics=summary["source_dev_frame_metrics"]
    metric_rows=[("metric","value")]+[(k,round(v,6) if isinstance(v,float) else v) for k,v in sorted(metrics.items())]
    return f"""<!doctype html><meta charset="utf-8"><title>B00 report — {summary['run_id']}</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;max-width:60rem}}table{{border-collapse:collapse;margin:.5rem 0}}
td,th{{border:1px solid #ccc;padding:.3rem .6rem;font-size:.9rem}}code{{background:#f4f4f4;padding:.1rem .3rem}}
h2{{margin-top:1.6rem;border-bottom:1px solid #ddd}}</style>
<h1>B00 local baseline — {summary['run_id']}</h1>
<p><b>Stage</b> {summary['stage']} · <b>device</b> {summary['device']} · generated {summary['generated_at']}</p>
<h2>1. Run identity and hashes</h2>{_table([("field","value")]+[(k,f"<code>{v}</code>") for k,v in summary['identity'].items()])}
<h2>2. Environment</h2>{_table([("key","value")]+list(summary['environment'].items()))}
<h2>3. Data counts</h2>{_table([("split","samples")]+list(summary['data_counts'].items()))}
<h2>4. Training curve</h2>{_table(curve)}
<p>Stopped: <b>{summary['training']['stopped_reason']}</b> · steps {summary['training']['global_step']} ·
elapsed {summary['training']['elapsed_seconds']}s</p>
<h2>5. Best checkpoint selection</h2><p>Rule: <code>{summary['training']['selection_rule']}</code></p>
{_table([("field","value")]+list((summary['training'].get('best') or {}).items()))}
<h2>6. Source-dev calibration</h2>{_table([("field","value")]+[(k,json.dumps(v) if isinstance(v,dict) else v) for k,v in summary['calibration'].items()])}
<h2>7. Source-dev frame metrics</h2>{_table(metric_rows)}
<h2>8. Source-dev video metrics</h2>{_table([("field","value")]+[(k,json.dumps(v)) for k,v in summary['source_dev_video_metrics'].items()])}
<h2>9. Target predictions (blind)</h2>{_table([("field","value")]+[(k,json.dumps(v)) for k,v in summary['target'].items()])}
<h2>10. Target isolation</h2><p><b>{summary['target_isolation_statement']}</b></p>
<h2>11. Known limitations</h2><ul>{''.join(f'<li>{item}</li>' for item in summary['known_limitations'])}</ul>
"""
def write_complete(run_root:Path,*,required:list[Path],payload:dict)->dict:
    """COMPLETE.json is written only after every required artifact validates."""
    run_root=Path(run_root)
    missing=[str(Path(path).relative_to(run_root)) for path in required if not Path(path).is_file() or Path(path).stat().st_size==0]
    if missing: raise FileNotFoundError(f"cannot complete run, missing artifacts: {missing}")
    record={**payload,"status":"complete","completed_at":datetime.now(timezone.utc).isoformat(),
            "validated_artifacts":[str(Path(path).relative_to(run_root)).replace("\\","/") for path in required]}
    atomic_json_write(run_root/"COMPLETE.json",record)
    return record
