from __future__ import annotations
import numpy as np

class ParityError(AssertionError):
    """A local/remote parity comparison failed."""
EXACT_SOURCE_FIELDS=("sample_id","source_record_id","true_label","true_target","crop_sha256","prior_sha256")
EXACT_TARGET_FIELDS=("sample_id","source_record_id","crop_sha256","prior_sha256")
FORBIDDEN_TARGET_FIELDS=frozenset({"true_label","true_target","label","label_live_spoof","attack_type","taxonomy",
    "subject_id","session_id","identity_embedding","source_path","crop_relative_path"})
def compare_exact(local:list[dict],remote:list[dict],fields:tuple[str,...])->dict:
    """Metadata that must match bit-for-bit: identity, order, labels, hashes."""
    if len(local)!=len(remote): raise ParityError(f"row count differs: local {len(local)} vs remote {len(remote)}")
    mismatches=[]
    for position,(a,b) in enumerate(zip(local,remote)):
        for field in fields:
            if a.get(field)!=b.get(field):
                mismatches.append({"index":position,"field":field,"local":a.get(field),"remote":b.get(field)})
    if mismatches: raise ParityError(f"exact field mismatch: {mismatches[:5]}")
    return {"rows":len(local),"fields_checked":list(fields),"mismatches":0}
def compare_numeric(local:np.ndarray,remote:np.ndarray,*,name:str,max_abs:float,mean_abs:float|None=None)->dict:
    local=np.asarray(local,dtype=np.float64); remote=np.asarray(remote,dtype=np.float64)
    if local.shape!=remote.shape: raise ParityError(f"{name} shape differs: {local.shape} vs {remote.shape}")
    difference=np.abs(local-remote)
    result={"name":name,"max_abs_diff":float(difference.max()),"mean_abs_diff":float(difference.mean()),
            "tolerance_max":max_abs,"tolerance_mean":mean_abs}
    if result["max_abs_diff"]>max_abs: raise ParityError(f"{name} max abs diff {result['max_abs_diff']:.3e} > {max_abs:.3e}")
    if mean_abs is not None and result["mean_abs_diff"]>mean_abs:
        raise ParityError(f"{name} mean abs diff {result['mean_abs_diff']:.3e} > {mean_abs:.3e}")
    result["passed"]=True
    return result
def compare_features(local:np.ndarray,remote:np.ndarray,*,min_cosine:float)->dict:
    local=np.asarray(local,dtype=np.float64); remote=np.asarray(remote,dtype=np.float64)
    if local.shape!=remote.shape: raise ParityError(f"feature shape differs: {local.shape} vs {remote.shape}")
    numerator=(local*remote).sum(axis=1)
    denominator=np.linalg.norm(local,axis=1)*np.linalg.norm(remote,axis=1)+1e-12
    cosine=numerator/denominator
    result={"mean_cosine":float(cosine.mean()),"min_cosine":float(cosine.min()),"tolerance":min_cosine}
    if result["mean_cosine"]<min_cosine: raise ParityError(f"mean cosine {result['mean_cosine']:.8f} < {min_cosine}")
    result["passed"]=True
    return result
def compare_decisions(local:list[dict],remote:list[dict],*,threshold:float,ambiguous_band:float)->dict:
    """Decisions must agree except for samples within the ambiguity band.

    The threshold is never adjusted to force agreement.
    """
    disagreements=[];ambiguous=[]
    for a,b in zip(local,remote):
        if a["decision"]==b["decision"]: continue
        distance=min(abs(a["p_spoof_calibrated"]-threshold),abs(b["p_spoof_calibrated"]-threshold))
        (ambiguous if distance<=ambiguous_band else disagreements).append(
            {"sample_id":a["sample_id"],"local":a["decision"],"remote":b["decision"],"distance_to_threshold":distance})
    if disagreements: raise ParityError(f"decision disagreement outside the ambiguity band: {disagreements[:5]}")
    return {"rows":len(local),"disagreements":0,"numerically_ambiguous":len(ambiguous),
            "threshold":threshold,"ambiguous_band":ambiguous_band}
def assert_target_isolated(rows:list[dict])->dict:
    leaked=sorted(FORBIDDEN_TARGET_FIELDS & {key for row in rows for key in row})
    if leaked: raise ParityError(f"target parity rows expose forbidden fields: {leaked}")
    return {"rows":len(rows),"forbidden_fields":[],"labels_present":False}
