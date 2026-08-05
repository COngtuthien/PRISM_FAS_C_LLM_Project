from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pyarrow as pa
from prism_fas.data.loader import CanonicalPackageDataset, collate_source_batch, collate_target_batch, load_loader_config, package_summary
from prism_fas.data.package.manifests import write_manifest
from prism_fas.utils.core import stable_json_hash
from .calibration import apply_temperature
from .checkpoint import checkpoint_sha256, load_checkpoint
from .config import B00Config
from .models import build_b00_model

SOURCE_PREDICTION_SCHEMA=pa.schema([("sample_id",pa.string()),("source_record_id",pa.string()),("dataset",pa.string()),
    ("actual_frame_index",pa.int64()),("true_label",pa.string()),("true_target",pa.int64()),("spoof_logit",pa.float64()),
    ("p_spoof_raw",pa.float64()),("p_spoof_calibrated",pa.float64()),("confidence",pa.float64()),("decision",pa.string()),
    ("checkpoint_hash",pa.string()),("calibration_hash",pa.string()),("inference_config_hash",pa.string())])
# Target rows deliberately carry no label, target, identity or private field.
TARGET_PREDICTION_SCHEMA=pa.schema([("sample_id",pa.string()),("source_record_id",pa.string()),
    ("actual_frame_index",pa.int64()),("spoof_logit",pa.float64()),("p_spoof_raw",pa.float64()),
    ("p_spoof_calibrated",pa.float64()),("confidence",pa.float64()),("decision",pa.string()),
    ("checkpoint_hash",pa.string()),("calibration_hash",pa.string()),("inference_config_hash",pa.string())])
FORBIDDEN_TARGET_COLUMNS=frozenset({"true_label","true_target","label","label_live_spoof","attack_type","taxonomy",
    "subject_id","session_id","identity_embedding","crop_relative_path","source_path"})
def load_model_for_inference(checkpoint_path:Path,config:B00Config,device:str):
    import torch
    payload=load_checkpoint(checkpoint_path,config_hash=config.config_hash,model_name=config.model.model_name,
                            label_mapping=dict(config.label_mapping))
    model=build_b00_model(config.model).to(device)
    model.load_state_dict(payload["model_state"]); model.eval()
    return model,payload
def _forward(model,dataset,device:str,collate,batch_size:int,limit:int|None,workers:int)->dict:
    import torch
    from torch.utils.data import DataLoader
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,collate_fn=collate,num_workers=workers)
    logits=[];rows=[]
    with torch.inference_mode():
        for batch in loader:
            output=model(batch["image"].to(device))
            logits.append(output.spoof_logit.float().cpu().numpy())
            for position in range(len(batch["sample_id"])):
                row={"sample_id":batch["sample_id"][position],"source_record_id":batch["source_record_id"][position],
                     "dataset":batch["dataset"][position]}
                if "target" in batch: row["true_target"]=int(batch["target"][position]); row["true_label"]=batch["label"][position]
                rows.append(row)
            if limit and len(rows)>=limit: break
    values=np.concatenate(logits) if logits else np.zeros(0)
    if limit: values,rows=values[:limit],rows[:limit]
    return {"logits":values,"rows":rows}
def predict_source_dev(package_root:Path,run_root:Path,config:B00Config,*,device:str,checkpoint:Path,
                       temperature:float|None=None,threshold:float|None=None,calibration_hash:str="",
                       limit:int|None=None,workers:int=0,loader_config_path:Path|None=None)->dict:
    loader_config=load_loader_config(loader_config_path or Path("configs/data/loader_m4.yaml"))
    dataset=CanonicalPackageDataset(package_root,"source_dev",loader_config,mode="validation")
    model,_=load_model_for_inference(checkpoint,config,device)
    result=_forward(model,dataset,device,collate_source_batch,config.batch_size,limit,workers)
    logits=result["logits"]; raw=1./(1.+np.exp(-logits))
    calibrated=apply_temperature(logits,temperature) if temperature else raw
    decision_threshold=threshold if threshold is not None else 0.5
    checkpoint_hash=checkpoint_sha256(checkpoint)
    inference_hash=stable_json_hash({"model":config.model.model_name,"threshold":decision_threshold,
                                     "temperature":temperature,"package":package_summary(package_root)["content_identity_sha256"]})
    rows=[]
    for position,row in enumerate(result["rows"]):
        probability=float(calibrated[position])
        rows.append({**row,"actual_frame_index":0,"spoof_logit":float(logits[position]),"p_spoof_raw":float(raw[position]),
                     "p_spoof_calibrated":probability,"confidence":float(max(probability,1.-probability)),
                     "decision":"spoof" if probability>=decision_threshold else "live",
                     "checkpoint_hash":checkpoint_hash,"calibration_hash":calibration_hash,
                     "inference_config_hash":inference_hash})
    if temperature is not None:
        write_manifest(Path(run_root)/"predictions"/"source_dev.parquet",rows,SOURCE_PREDICTION_SCHEMA,
                       {"stage":"M5_B00","split":"source_dev"})
    return {"rows":rows,"logits":logits,"targets":np.array([row["true_target"] for row in result["rows"]],dtype=np.int64),
            "checkpoint_hash":checkpoint_hash,"inference_config_hash":inference_hash,
            "prediction_hash":hashlib.sha256(json.dumps([[r["sample_id"],round(float(l),6)] for r,l in zip(result["rows"],logits)],sort_keys=True).encode()).hexdigest()}
def predict_target(package_root:Path,run_root:Path,config:B00Config,*,device:str,checkpoint:Path,temperature:float,
                   threshold:float,calibration_hash:str,limit:int|None=None,workers:int=0,
                   loader_config_path:Path|None=None)->dict:
    """Blind target inference under the frozen source-only calibration."""
    loader_config=load_loader_config(loader_config_path or Path("configs/data/loader_m4.yaml"))
    dataset=CanonicalPackageDataset(package_root,"target_test",loader_config,mode="inference")
    model,_=load_model_for_inference(checkpoint,config,device)
    result=_forward(model,dataset,device,collate_target_batch,config.batch_size,limit,workers)
    logits=result["logits"]; raw=1./(1.+np.exp(-logits)); calibrated=apply_temperature(logits,temperature)
    checkpoint_hash=checkpoint_sha256(checkpoint)
    inference_hash=stable_json_hash({"model":config.model.model_name,"threshold":threshold,"temperature":temperature,
                                     "package":package_summary(package_root)["content_identity_sha256"]})
    rows=[]
    for position,row in enumerate(result["rows"]):
        probability=float(calibrated[position])
        rows.append({"sample_id":row["sample_id"],"source_record_id":row["source_record_id"],"actual_frame_index":0,
                     "spoof_logit":float(logits[position]),"p_spoof_raw":float(raw[position]),
                     "p_spoof_calibrated":probability,"confidence":float(max(probability,1.-probability)),
                     "decision":"spoof" if probability>=threshold else "live","checkpoint_hash":checkpoint_hash,
                     "calibration_hash":calibration_hash,"inference_config_hash":inference_hash})
    leaked=sorted(FORBIDDEN_TARGET_COLUMNS & {key for row in rows for key in row})
    if leaked: raise ValueError(f"target predictions expose forbidden columns: {leaked}")
    write_manifest(Path(run_root)/"predictions"/"siw_mv2.parquet",rows,TARGET_PREDICTION_SCHEMA,
                   {"stage":"M5_B00","split":"target_test","labels":"not_accessed"})
    return {"rows":rows,"count":len(rows),"checkpoint_hash":checkpoint_hash,"inference_config_hash":inference_hash}
