from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
from prism_fas.utils.core import atomic_json_write
from .metrics import brier_score, expected_calibration_error, negative_log_likelihood, select_min_acer_threshold, summarize

CALIBRATION_SCHEMA_VERSION="m5-b00-calibration-v1"
class CalibrationError(ValueError):
    """Temperature scaling could not be fitted."""
def fit_temperature(logits:np.ndarray,targets:np.ndarray,*,iterations:int=500,learning_rate:float=0.05)->float:
    """Fit one positive scalar T minimizing source_dev BCE of sigmoid(logit / T).

    T = softplus(raw) + eps keeps the temperature strictly positive. The
    optimization is deterministic: fixed init, fixed step count, no shuffling,
    and it never touches model weights or target data.
    """
    import torch
    values=torch.tensor(np.asarray(logits,dtype=np.float64),dtype=torch.float64)
    labels=torch.tensor(np.asarray(targets,dtype=np.float64),dtype=torch.float64)
    if values.numel()==0: raise CalibrationError("no source_dev logits to calibrate")
    raw=torch.zeros(1,dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.Adam([raw],lr=learning_rate)
    for _ in range(iterations):
        optimizer.zero_grad()
        temperature=torch.nn.functional.softplus(raw)+1e-6
        loss=torch.nn.functional.binary_cross_entropy_with_logits(values/temperature,labels,reduction="mean")
        loss.backward(); optimizer.step()
    temperature=float(torch.nn.functional.softplus(raw.detach()).item()+1e-6)
    if not np.isfinite(temperature) or temperature<=0: raise CalibrationError("fitted temperature is not positive")
    return temperature
def apply_temperature(logits:np.ndarray,temperature:float)->np.ndarray:
    return 1./(1.+np.exp(-np.asarray(logits,dtype=np.float64)/float(temperature)))
def calibrate_source_dev(logits:np.ndarray,targets:np.ndarray,*,checkpoint_sha:str,package_identity:str,
                         prediction_hash:str,run_root:Path)->dict:
    """Fit temperature and select the min-ACER threshold using source_dev only."""
    logits=np.asarray(logits,dtype=np.float64); targets=np.asarray(targets,dtype=np.int64)
    raw=1./(1.+np.exp(-logits))
    temperature=fit_temperature(logits,targets)
    calibrated=apply_temperature(logits,temperature)
    selection=select_min_acer_threshold(calibrated,targets)
    threshold=float(selection["selected"]["threshold"])
    record={"calibration_schema_version":CALIBRATION_SCHEMA_VERSION,"checkpoint_sha256":checkpoint_sha,
            "package_content_identity":package_identity,"source_dev_prediction_hash":prediction_hash,
            "criterion":"temperature_scaling_bce","temperature":temperature,
            "before":{"nll":negative_log_likelihood(raw,targets),"ece":expected_calibration_error(raw,targets),"brier":brier_score(raw,targets)},
            "after":{"nll":negative_log_likelihood(calibrated,targets),"ece":expected_calibration_error(calibrated,targets),"brier":brier_score(calibrated,targets)},
            "threshold_selection":{key:value for key,value in selection.items() if key!="selected"},
            "selected_threshold":threshold,"source_dev_metrics_at_threshold":summarize(calibrated,targets,threshold),
            "reject_policy":"disabled_for_b00"}
    record["calibration_hash"]=hashlib.sha256(json.dumps({k:v for k,v in record.items() if k!="calibration_hash"},sort_keys=True,default=str).encode()).hexdigest()
    atomic_json_write(Path(run_root)/"calibration"/"source_dev.json",record)
    return record
