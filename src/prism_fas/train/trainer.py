from __future__ import annotations
import json, math, platform, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
import numpy as np
from prism_fas.data.loader import (BalancedDomainClassBatchSampler, CanonicalPackageDataset, collate_source_batch,
                                   load_loader_config, open_package, package_summary)
from prism_fas.utils.core import atomic_json_write, git_commit
from .checkpoint import CheckpointContractError, checkpoint_sha256, load_checkpoint, save_checkpoint
from .config import B00Config, TRAIN_SCHEMA_VERSION
from .losses import b00_binary_cross_entropy
from .metrics import roc_auc, negative_log_likelihood, summarize
from .models import build_b00_model
from .seed import restore_rng_state, rng_state, seed_everything

RUN_DIRECTORIES=("logs","checkpoints","calibration","predictions","reports","failures")
class TrainingContractError(RuntimeError):
    """A training precondition was violated."""
def resolve_device(requested:str|None)->str:
    import torch
    if requested: return requested
    return "cuda" if torch.cuda.is_available() else "cpu"
def environment_fingerprint(device:str)->dict:
    import torch, torchvision
    info={"python":platform.python_version(),"platform":platform.platform(),"torch":torch.__version__,
          "torchvision":torchvision.__version__,"cuda_available":str(torch.cuda.is_available()),"device":device}
    try:
        import timm; info["timm"]=timm.__version__
    except ImportError: info["timm"]="missing"
    if torch.cuda.is_available():
        info["gpu_name"]=torch.cuda.get_device_name(0)
        info["gpu_memory_gb"]=round(torch.cuda.get_device_properties(0).total_memory/1e9,2)
    return info
def prepare_run(run_root:Path,config:B00Config,package_root:Path,device:str,*,loader_config)->dict:
    run_root=Path(run_root); run_root.mkdir(parents=True,exist_ok=True)
    for name in RUN_DIRECTORIES: (run_root/name).mkdir(exist_ok=True)
    summary=package_summary(package_root)
    (run_root/"resolved_config.yaml").write_text(json.dumps(config.model_dump(mode="json"),indent=1,sort_keys=True),encoding="utf-8")
    (run_root/"environment.txt").write_text("\n".join(f"{k}={v}" for k,v in environment_fingerprint(device).items()),encoding="utf-8")
    atomic_json_write(run_root/"git_state.json",{"commit":git_commit(Path.cwd()),"branch_note":"recorded at run start"})
    atomic_json_write(run_root/"data_lock.json",{"package_id":summary["package_id"],
        "package_content_identity":summary["content_identity_sha256"],"parent_package_id":summary["parent_package_id"],
        "per_split_counts":summary["per_split_counts"],"loader_schema_version":loader_config.loader_schema_version,
        "label_mapping":dict(config.label_mapping)})
    return summary
def _lr_at(step:int,total_steps:int,warmup_steps:int,base_lr:float,min_lr:float)->float:
    if step<warmup_steps: return base_lr*(step+1)/max(warmup_steps,1)
    progress=(step-warmup_steps)/max(total_steps-warmup_steps,1)
    return min_lr+(base_lr-min_lr)*0.5*(1.+math.cos(math.pi*min(progress,1.)))
def evaluate_source_dev(model,dataset,device:str,*,batch_size:int=32,limit:int|None=None,workers:int=0)->dict:
    """Deterministic sequential source_dev pass; never used for gradients."""
    import torch
    from torch.utils.data import DataLoader
    model.eval()
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,collate_fn=collate_source_batch,num_workers=workers)
    logits=[];targets=[];ids=[];records=[];datasets=[];frames=[]
    with torch.inference_mode():
        for batch in loader:
            output=model(batch["image"].to(device))
            logits.append(output.spoof_logit.float().cpu().numpy()); targets.append(batch["target"].numpy())
            ids.extend(batch["sample_id"]); records.extend(batch["source_record_id"]); datasets.extend(batch["dataset"])
            if limit and len(ids)>=limit: break
    logits=np.concatenate(logits); targets=np.concatenate(targets)
    if limit: logits,targets,ids,records,datasets=logits[:limit],targets[:limit],ids[:limit],records[:limit],datasets[:limit]
    probabilities=1./(1.+np.exp(-logits))
    return {"logits":logits,"targets":targets,"sample_id":ids,"source_record_id":records,"dataset":datasets,
            "probabilities":probabilities,"roc_auc":roc_auc(probabilities,targets),
            "nll":negative_log_likelihood(probabilities,targets)}
def train_b00(package_root:Path,run_root:Path,config:B00Config,*,device:str|None=None,resume:bool=False,
              limit_steps:int|None=None,limit_dev_samples:int|None=None,workers:int=0,
              loader_config_path:Path|None=None,progress:Callable[[dict],None]|None=None)->dict:
    """Train B00 on source_train only, selecting the best epoch by source_dev ROC-AUC."""
    import torch
    from torch.utils.data import DataLoader
    device=resolve_device(device); started=time.time()
    loader_config=load_loader_config(loader_config_path or Path("configs/data/loader_m4.yaml"))
    summary=prepare_run(run_root,config,package_root,device,loader_config=loader_config)
    if summary["package_id"]!=config.package["expected_package_id"]:
        raise TrainingContractError("package id does not match the training config")
    seed_everything(config.seed)
    train_index=open_package(package_root,"source_train",loader_config,mode="training")
    train_dataset=CanonicalPackageDataset(package_root,"source_train",loader_config,mode="training",index=train_index)
    dev_dataset=CanonicalPackageDataset(package_root,"source_dev",loader_config,mode="validation")
    sampler=BalancedDomainClassBatchSampler(train_index,loader_config)
    model=build_b00_model(config.model).to(device)
    groups=model.parameter_groups(config.optimizer.backbone_lr,config.optimizer.head_lr,config.optimizer.weight_decay)
    optimizer=torch.optim.AdamW(groups,betas=tuple(config.optimizer.betas))
    use_amp=(device=="cuda" and config.precision.get("cuda")=="amp")
    scaler=torch.amp.GradScaler("cuda",enabled=use_amp)
    total_steps=config.epochs*config.steps_per_epoch; warmup=config.scheduler.warmup_epochs*config.steps_per_epoch
    base_lrs=[group["lr"] for group in optimizer.param_groups]
    start_epoch=0; global_step=0; best={"roc_auc":-1.,"nll":float("inf"),"epoch":-1,"sha256":None}; history=[]
    last_path=Path(run_root)/"checkpoints"/"last.pt"; best_path=Path(run_root)/"checkpoints"/"best.pt"
    if resume and last_path.is_file():
        payload=load_checkpoint(last_path,config_hash=config.config_hash,package_identity=summary["content_identity_sha256"],
                                model_name=config.model.model_name,label_mapping=dict(config.label_mapping))
        model.load_state_dict(payload["model_state"]); optimizer.load_state_dict(payload["optimizer_state"])
        if payload.get("scaler_state") and use_amp: scaler.load_state_dict(payload["scaler_state"])
        restore_rng_state(payload.get("rng_state") or {})
        start_epoch=int(payload["epoch"])+1; global_step=int(payload["global_step"]); best=payload.get("best",best)
        history=payload.get("history",[])
    metrics_path=Path(run_root)/"logs"/"metrics.jsonl"; log_path=Path(run_root)/"logs"/"train.log"
    stopped_reason="completed_all_epochs"
    for epoch in range(start_epoch,config.epochs):
        sampler.set_epoch(epoch); model.train()
        loader=DataLoader(train_dataset,batch_sampler=sampler,collate_fn=collate_source_batch,num_workers=workers)
        epoch_started=time.time(); losses=[]; composition={}
        for position,batch in enumerate(loader):
            if limit_steps is not None and global_step>=limit_steps: break
            for group,base in zip(optimizer.param_groups,base_lrs):
                group["lr"]=_lr_at(global_step,total_steps,warmup,base,config.scheduler.min_lr)
            optimizer.zero_grad(set_to_none=True)
            images=batch["image"].to(device); targets=batch["target"].to(device)
            with torch.autocast("cuda",enabled=use_amp):
                output=model(images); loss=b00_binary_cross_entropy(output.spoof_logit.float(),targets)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            grad_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip_norm))
            scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach())); global_step+=1
            for dataset_name,label in zip(batch["dataset"],batch["label"]):
                composition[f"{dataset_name}/{label}"]=composition.get(f"{dataset_name}/{label}",0)+1
            record={"epoch":epoch,"step":global_step,"train_loss":float(loss.detach()),
                    "lr_backbone":optimizer.param_groups[0]["lr"],"lr_head":optimizer.param_groups[1]["lr"],
                    "grad_norm":grad_norm,"batch_composition":dict(sorted(composition.items())),
                    "samples_per_second":round(config.batch_size*(position+1)/max(time.time()-epoch_started,1e-6),2)}
            with metrics_path.open("a",encoding="utf-8") as handle: handle.write(json.dumps(record)+"\n")
            if progress: progress(record)
        evaluation=evaluate_source_dev(model,dev_dataset,device,batch_size=config.batch_size,limit=limit_dev_samples,workers=workers)
        entry={"epoch":epoch,"train_loss":float(np.mean(losses)) if losses else None,
               "source_dev_roc_auc":evaluation["roc_auc"],"source_dev_nll":evaluation["nll"],
               "source_dev_samples":len(evaluation["sample_id"]),"seconds":round(time.time()-epoch_started,1)}
        history.append(entry)
        with log_path.open("a",encoding="utf-8") as handle: handle.write(json.dumps(entry)+"\n")
        if progress: progress({"stage":"epoch_end",**entry})
        payload={"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),
                 "scaler_state":scaler.state_dict() if use_amp else None,"scheduler_state":{"base_lrs":base_lrs},
                 "epoch":epoch,"global_step":global_step,"best":best,"history":history,"rng_state":rng_state(),
                 "sampler_state":sampler.state(),"config_hash":config.config_hash,
                 "package_content_identity":summary["content_identity_sha256"],"model_name":config.model.model_name,
                 "model_revision":config.model.revision,"model_weight_sha256":config.model.weight_sha256,
                 "label_mapping":dict(config.label_mapping),"git_commit":git_commit(Path.cwd()),
                 "train_schema_version":TRAIN_SCHEMA_VERSION}
        save_checkpoint(last_path,payload)
        improved=(entry["source_dev_roc_auc"]>best["roc_auc"] or
                  (entry["source_dev_roc_auc"]==best["roc_auc"] and entry["source_dev_nll"]<best["nll"]))
        if improved:
            best={"roc_auc":entry["source_dev_roc_auc"],"nll":entry["source_dev_nll"],"epoch":epoch,"sha256":None}
            payload["best"]=best; best["sha256"]=save_checkpoint(best_path,payload)
        if limit_steps is not None and global_step>=limit_steps: stopped_reason="limit_steps_reached"; break
        if config.early_stopping.enabled and epoch+1>=config.early_stopping.min_epochs:
            if epoch-best["epoch"]>=config.early_stopping.patience_epochs:
                stopped_reason=f"early_stopped_patience_{config.early_stopping.patience_epochs}"; break
    result={"run_root":str(run_root),"device":device,"epochs_run":len(history),"global_step":global_step,
            "best":best,"history":history,"stopped_reason":stopped_reason,
            "last_checkpoint_sha256":checkpoint_sha256(last_path) if last_path.is_file() else None,
            "best_checkpoint_sha256":checkpoint_sha256(best_path) if best_path.is_file() else None,
            "package_content_identity":summary["content_identity_sha256"],
            "selection_rule":"max source_dev_roc_auc, tie-break min source_dev_nll",
            "elapsed_seconds":round(time.time()-started,1),"amp":use_amp}
    atomic_json_write(Path(run_root)/"run.json",{"run_id":Path(run_root).name,"stage":config.stage,"status":"trained",
        "backend":"local","device":device,"git_commit":git_commit(Path.cwd()),"package_id":summary["package_id"],
        "package_content_identity":summary["content_identity_sha256"],"config_hash":config.config_hash,
        "model_name":config.model.model_name,"model_revision":config.model.revision,
        "model_weight_sha256":config.model.weight_sha256,"seed":config.seed,
        "started_at":datetime.fromtimestamp(started,timezone.utc).isoformat(),
        "finished_at":datetime.now(timezone.utc).isoformat(),"selection_rule":result["selection_rule"],
        "best_checkpoint_sha256":result["best_checkpoint_sha256"],"last_checkpoint_sha256":result["last_checkpoint_sha256"],
        "epochs_run":result["epochs_run"],"global_step":global_step,"stopped_reason":stopped_reason,
        "train_schema_version":TRAIN_SCHEMA_VERSION})
    return result
