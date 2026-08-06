"""Build the deterministic local CPU parity reference for M6."""
from __future__ import annotations
import json, sys
from pathlib import Path

def main()->int:
    from prism_fas.cloud.artifacts import run_parity_forward, save_reference
    from prism_fas.cloud.config import load_cloud_config
    from prism_fas.train.config import load_b00_config
    cloud=load_cloud_config(Path("configs/cloud/modal_m6.yaml"))
    config=load_b00_config(Path("configs/train/b00_local.yaml"))
    run=Path("runs/b00_local_seed42")
    calibration=json.loads((run/"calibration"/"source_dev.json").read_text(encoding="utf-8"))
    result=run_parity_forward(Path("data/processed/prism_data_v1_m3b"),run/"checkpoints"/"best.pt",calibration,config,
                              device="cpu",source_count=cloud.parity["source_dev_samples"],
                              target_count=cloud.parity["target_samples"])
    save_reference(result,Path("reports/m6/local_parity_reference.json"),Path("reports/m6/local_parity_reference.npz"))
    print(json.dumps({"device":result["device"],"source":len(result["source"]),"target":len(result["target"]),
                      "loss":result["batch_bce_loss"],"checkpoint_sha256":result["checkpoint_sha256"][:16]+"...",
                      "temperature":result["temperature"],"threshold":result["threshold"]}))
    return 0
if __name__=="__main__": sys.exit(main())
