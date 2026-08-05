from __future__ import annotations
import os, random
import numpy as np

def seed_everything(seed:int)->dict:
    """Seed local RNGs deterministically; never relies on ambient global state."""
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED",str(seed))
    return rng_state()
def rng_state()->dict:
    import torch
    state={"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state()}
    if torch.cuda.is_available(): state["torch_cuda"]=torch.cuda.get_rng_state_all()
    return state
def restore_rng_state(state:dict)->None:
    import torch
    if not state: return
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state["torch_cuda"])
