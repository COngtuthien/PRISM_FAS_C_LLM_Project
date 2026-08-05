from __future__ import annotations
from pathlib import Path
from .config import SPLITS
from .manifests import read_manifest

TRAINING_SPLITS=("source_train","source_dev")
TARGET_SPLITS=("target_test",)
class TargetIsolationError(PermissionError):
    """A training-mode selector attempted to reach the inference-only target split."""
def select_split_manifest(package_root:Path,split:str,*,mode:str="training")->list[dict]:
    """Package-level split selector enforcing target isolation.

    Training mode may only reach source_train/source_dev; target_test is
    inference-only and must be requested explicitly in inference mode.
    """
    if split not in SPLITS: raise ValueError(f"unknown split: {split!r}")
    if mode not in {"training","inference"}: raise ValueError(f"unknown selector mode: {mode!r}")
    if mode=="training" and split in TARGET_SPLITS:
        raise TargetIsolationError("training-mode selector cannot request target_test")
    name={"source_train":"source_train","source_dev":"source_dev","target_test":"target_test_features"}[split]
    return read_manifest(Path(package_root)/"manifests"/f"{name}.parquet")
def available_splits(mode:str="training")->tuple[str,...]:
    return TRAINING_SPLITS if mode=="training" else TARGET_SPLITS
