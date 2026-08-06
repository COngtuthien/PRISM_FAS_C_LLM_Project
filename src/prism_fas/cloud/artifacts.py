from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from prism_fas.data.loader import CanonicalPackageDataset, collate_source_batch, collate_target_batch, load_loader_config, package_summary
from prism_fas.train.calibration import apply_temperature
from prism_fas.train.checkpoint import checkpoint_sha256
from prism_fas.train.config import B00Config
from prism_fas.train.losses import b00_binary_cross_entropy
from prism_fas.train.models import build_b00_model

def fixture_sample_ids(package_root:Path,split:str,mode:str,count:int,loader_config)->list[str]:
    dataset=CanonicalPackageDataset(package_root,split,loader_config,mode=mode)
    return list(dataset.sample_ids)[:count]
def _collect_by_ids(package_root:Path,split:str,mode:str,wanted:list[str],loader_config)->list:
    """Stream the split's shards and return the wanted samples in the given order.

    Only 9 tar files are opened and nothing is extracted to disk, which avoids
    thousands of small-file opens on a network-mounted Volume.
    """
    from prism_fas.data.loader import CanonicalShardDataset
    remaining=set(wanted); found={}
    for sample in CanonicalShardDataset(package_root,split,loader_config,mode=mode):
        if sample.sample_id in remaining:
            found[sample.sample_id]=sample; remaining.discard(sample.sample_id)
            if not remaining: break
    if remaining: raise KeyError(f"shard backend could not locate {len(remaining)} sample(s) in {split}")
    return [found[sample_id] for sample_id in wanted]
def run_parity_forward(package_root:Path,checkpoint:Path,calibration:dict,config:B00Config,*,device:str,
                       source_count:int=32,target_count:int=16,loader_config_path:Path|None=None,
                       backend:str="loose",weight_file:str|None=None)->dict:
    """Deterministic fp32 eval-mode forward over fixed sample IDs.

    Identical code path on local CPU and Modal GPU; the only difference is the
    device, which is what parity is measuring.
    """
    import torch
    loader_config=load_loader_config(loader_config_path or Path("configs/data/loader_m4.yaml"))
    model=build_b00_model(config.model,weight_file=weight_file).to(device)
    payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
    model.load_state_dict(payload["model_state"]); model.eval()
    temperature=float(calibration["temperature"]); threshold=float(calibration["selected_threshold"])
    source_dataset=CanonicalPackageDataset(package_root,"source_dev",loader_config,mode="validation")
    target_dataset=CanonicalPackageDataset(package_root,"target_test",loader_config,mode="inference")
    source_ids=list(source_dataset.sample_ids)[:source_count]
    target_ids=list(target_dataset.sample_ids)[:target_count]
    if backend=="shard":
        source_samples=_collect_by_ids(package_root,"source_dev","validation",source_ids,loader_config)
        target_samples=_collect_by_ids(package_root,"target_test","inference",target_ids,loader_config)
    else:
        source_samples=[source_dataset[index] for index in range(source_count)]
        target_samples=[target_dataset[index] for index in range(target_count)]
    source_batch=collate_source_batch(source_samples); target_batch=collate_target_batch(target_samples)
    with torch.inference_mode():
        source_output=model(source_batch["image"].to(device),return_features=True)
        target_output=model(target_batch["image"].to(device),return_features=True)
    source_logits=source_output.spoof_logit.float().cpu().numpy()
    target_logits=target_output.spoof_logit.float().cpu().numpy()
    features=source_output.feature_embedding.float().cpu().numpy()
    with torch.no_grad():
        loss=float(b00_binary_cross_entropy(torch.tensor(source_logits,dtype=torch.float32),source_batch["target"]))
    source_rows=[]
    for position,sample in enumerate(source_samples):
        calibrated=float(apply_temperature(np.array([source_logits[position]]),temperature)[0])
        source_rows.append({"sample_id":sample.sample_id,"source_record_id":sample.source_record_id,
            "true_label":sample.label,"true_target":int(sample.class_target),"crop_sha256":sample.crop_sha256,
            "prior_sha256":sample.prior_sha256,"spoof_logit":float(source_logits[position]),
            "p_spoof_raw":float(1/(1+np.exp(-source_logits[position]))),"p_spoof_calibrated":calibrated,
            "decision":"spoof" if calibrated>=threshold else "live"})
    target_rows=[]
    for position,sample in enumerate(target_samples):
        calibrated=float(apply_temperature(np.array([target_logits[position]]),temperature)[0])
        target_rows.append({"sample_id":sample.sample_id,"source_record_id":sample.source_record_id,
            "crop_sha256":sample.crop_sha256,"prior_sha256":sample.prior_sha256,
            "spoof_logit":float(target_logits[position]),
            "p_spoof_raw":float(1/(1+np.exp(-target_logits[position]))),"p_spoof_calibrated":calibrated,
            "decision":"spoof" if calibrated>=threshold else "live"})
    return {"device":device,"data_backend":backend,"package_content_identity":package_summary(package_root)["content_identity_sha256"],
            "checkpoint_sha256":checkpoint_sha256(checkpoint),"calibration_hash":calibration["calibration_hash"],
            "temperature":temperature,"threshold":threshold,"batch_bce_loss":loss,
            "source":source_rows,"target":target_rows,
            "feature_stats":{"shape":list(features.shape),"mean":float(features.mean()),"std":float(features.std())},
            "features":features.tolist()}
def save_reference(result:dict,json_path:Path,npz_path:Path)->None:
    features=np.asarray(result.pop("features"),dtype=np.float32)
    Path(json_path).parent.mkdir(parents=True,exist_ok=True)
    Path(json_path).write_text(json.dumps(result,indent=1,sort_keys=True),encoding="utf-8")
    np.savez(npz_path,features=features,
             source_logits=np.array([row["spoof_logit"] for row in result["source"]],dtype=np.float32),
             target_logits=np.array([row["spoof_logit"] for row in result["target"]],dtype=np.float32))
