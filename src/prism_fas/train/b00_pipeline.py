from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from prism_fas.data.loader import load_loader_config, open_package, package_summary
from prism_fas.utils.core import atomic_json_write
from .calibration import calibrate_source_dev
from .checkpoint import checkpoint_sha256
from .config import B00Config
from .inference import predict_source_dev, predict_target
from .metrics import summarize
from .report import build_report, write_complete
from .trainer import environment_fingerprint, resolve_device, train_b00
from .video_aggregation import aggregate_videos

RUNS_ROOT=Path("runs")
def _run_root(run_id:str)->Path: return RUNS_ROOT/run_id
def _dry_run(package_root:Path,config:B00Config,run_id:str,device:str|None)->dict:
    """Validate config/package and print the plan without loading weights or training."""
    loader_config=load_loader_config(Path("configs/data/loader_m4.yaml"))
    summary=package_summary(package_root)
    index=open_package(package_root,"source_train",loader_config,mode="training")
    resolved=resolve_device(device)
    return {"dry_run":True,"run_root":str(_run_root(run_id)),"package":summary,"device":resolved,
            "model_name":config.model.model_name,"batch_size":config.batch_size,
            "steps_per_epoch":config.steps_per_epoch,"epochs":config.epochs,
            "source_train":len(index),"config_hash":config.config_hash,"weights_loaded":False,"checkpoints_written":False}
def run_b00_train(package_root:Path,config:B00Config,run_id:str,*,device=None,workers:int=0,resume:bool=False,
                  limit_steps=None,dry_run:bool=False,progress=None)->dict:
    if dry_run: return _dry_run(package_root,config,run_id,device)
    result=train_b00(package_root,_run_root(run_id),config,device=device,resume=resume,limit_steps=limit_steps,
                     workers=workers,progress=progress)
    return {k:v for k,v in result.items() if k!="history"} | {"epochs_recorded":len(result["history"])}
def run_b00_calibrate(package_root:Path,run_root:Path,config:B00Config,*,device=None,workers:int=0,limit=None)->dict:
    """Fit temperature and threshold on source_dev only, then persist predictions."""
    run_root=Path(run_root); device=resolve_device(device)
    checkpoint=run_root/"checkpoints"/"best.pt"
    summary=package_summary(package_root)
    raw=predict_source_dev(package_root,run_root,config,device=device,checkpoint=checkpoint,limit=limit,workers=workers)
    record=calibrate_source_dev(raw["logits"],raw["targets"],checkpoint_sha=checkpoint_sha256(checkpoint),
                                package_identity=summary["content_identity_sha256"],
                                prediction_hash=raw["prediction_hash"],run_root=run_root)
    final=predict_source_dev(package_root,run_root,config,device=device,checkpoint=checkpoint,
                             temperature=record["temperature"],threshold=record["selected_threshold"],
                             calibration_hash=record["calibration_hash"],limit=limit,workers=workers)
    videos=aggregate_videos(final["rows"],record["selected_threshold"])
    atomic_json_write(run_root/"predictions"/"source_dev_videos.json",videos)
    metrics=summarize(np.array([row["p_spoof_calibrated"] for row in final["rows"]]),final["targets"],record["selected_threshold"])
    return {"source_dev_rows":len(final["rows"]),"temperature":record["temperature"],
            "selected_threshold":record["selected_threshold"],"before":record["before"],"after":record["after"],
            "metrics":metrics,"videos":len(videos),"calibration_hash":record["calibration_hash"],
            "checkpoint_hash":record["checkpoint_sha256"]}
def run_b00_predict_target(package_root:Path,run_root:Path,config:B00Config,*,device=None,workers:int=0,limit=None)->dict:
    """Blind target inference: frozen checkpoint + frozen source-only calibration."""
    run_root=Path(run_root); device=resolve_device(device)
    record=json.loads((run_root/"calibration"/"source_dev.json").read_text(encoding="utf-8"))
    result=predict_target(package_root,run_root,config,device=device,checkpoint=run_root/"checkpoints"/"best.pt",
                          temperature=record["temperature"],threshold=record["selected_threshold"],
                          calibration_hash=record["calibration_hash"],limit=limit,workers=workers)
    videos=aggregate_videos(result["rows"],record["selected_threshold"])
    atomic_json_write(run_root/"predictions"/"siw_mv2_videos.json",videos)
    return {"target_rows":result["count"],"target_videos":len(videos),"labels_accessed":False,
            "metrics_reported":False,"checkpoint_hash":result["checkpoint_hash"]}
def run_b00_report(run_root:Path)->dict:
    from prism_fas.data.package.manifests import read_manifest
    run_root=Path(run_root)
    run_info=json.loads((run_root/"run.json").read_text(encoding="utf-8"))
    calibration=json.loads((run_root/"calibration"/"source_dev.json").read_text(encoding="utf-8"))
    source_rows=read_manifest(run_root/"predictions"/"source_dev.parquet")
    target_rows=read_manifest(run_root/"predictions"/"siw_mv2.parquet")
    source_videos=json.loads((run_root/"predictions"/"source_dev_videos.json").read_text(encoding="utf-8"))
    target_videos=json.loads((run_root/"predictions"/"siw_mv2_videos.json").read_text(encoding="utf-8"))
    history=[json.loads(line) for line in (run_root/"logs"/"train.log").read_text(encoding="utf-8").splitlines() if line.strip()]
    environment=dict(line.split("=",1) for line in (run_root/"environment.txt").read_text(encoding="utf-8").splitlines() if "=" in line)
    metrics=summarize(np.array([row["p_spoof_calibrated"] for row in source_rows]),
                      np.array([row["true_target"] for row in source_rows]),calibration["selected_threshold"])
    training={"epochs_run":run_info.get("epochs_run"),"global_step":run_info.get("global_step"),
              "stopped_reason":run_info.get("stopped_reason"),"history":history,
              "selection_rule":run_info.get("selection_rule"),
              "best":{"sha256":run_info.get("best_checkpoint_sha256")},"elapsed_seconds":None}
    summary=build_report(run_root,run_info={**run_info,"source_dev_count":len(source_rows)},training=training,
                         calibration=calibration,source_metrics=metrics,source_video=source_videos,
                         target_rows=target_rows,target_video=target_videos,environment=environment)
    required=[run_root/"run.json",run_root/"checkpoints"/"last.pt",run_root/"checkpoints"/"best.pt",
              run_root/"calibration"/"source_dev.json",run_root/"predictions"/"source_dev.parquet",
              run_root/"predictions"/"siw_mv2.parquet",run_root/"reports"/"report.html",run_root/"reports"/"summary.json"]
    complete=write_complete(run_root,required=required,payload={"run_id":run_root.name,"stage":"M5_B00",
        "package_content_identity":run_info.get("package_content_identity"),
        "best_checkpoint_sha256":run_info.get("best_checkpoint_sha256"),
        "calibration_hash":calibration["calibration_hash"],"source_dev_rows":len(source_rows),
        "target_rows":len(target_rows),"target_labels_accessed":False})
    return {"report_html":str(run_root/"reports"/"report.html"),"summary_json":str(run_root/"reports"/"summary.json"),
            "source_dev_rows":len(source_rows),"target_rows":len(target_rows),"source_metrics":metrics,
            "complete":complete["status"]}
def run_b00_smoke(package_root:Path,config:B00Config,run_id:str,*,device=None,workers:int=0,limit_steps:int=5,
                  limit_dev_samples:int=128,limit_target_samples:int=64,dry_run:bool=False)->dict:
    """Real 5-step smoke: train, resume, calibrate, predict, report on subsets."""
    if dry_run: return _dry_run(package_root,config,run_id,device)
    run_root=_run_root(run_id); device=resolve_device(device)
    first=train_b00(package_root,run_root,config,device=device,limit_steps=limit_steps,
                    limit_dev_samples=limit_dev_samples,workers=workers)
    resumed=train_b00(package_root,run_root,config,device=device,resume=True,limit_steps=limit_steps+1,
                      limit_dev_samples=limit_dev_samples,workers=workers)
    calibration=run_b00_calibrate(package_root,run_root,config,device=device,workers=workers,limit=limit_dev_samples)
    target=run_b00_predict_target(package_root,run_root,config,device=device,workers=workers,limit=limit_target_samples)
    report=run_b00_report(run_root)
    return {"run_root":str(run_root),"device":device,"steps_first":first["global_step"],
            "steps_after_resume":resumed["global_step"],"resume_continued":resumed["global_step"]>first["global_step"],
            "best_epoch":first["best"]["epoch"],"calibration":{k:calibration[k] for k in ("temperature","selected_threshold","source_dev_rows")},
            "target_rows":target["target_rows"],"report":report["report_html"],"complete":report["complete"],
            "environment":environment_fingerprint(device)}
