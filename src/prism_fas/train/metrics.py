from __future__ import annotations
import numpy as np

class MetricInputError(ValueError):
    """Metric inputs are degenerate or malformed; no value is invented."""
def _check(probabilities:np.ndarray,targets:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    probabilities=np.asarray(probabilities,dtype=np.float64).reshape(-1)
    targets=np.asarray(targets,dtype=np.int64).reshape(-1)
    if probabilities.shape!=targets.shape: raise MetricInputError("probabilities and targets must align")
    if probabilities.size==0: raise MetricInputError("empty metric input")
    if not np.isfinite(probabilities).all(): raise MetricInputError("non-finite probabilities")
    if not set(np.unique(targets)).issubset({0,1}): raise MetricInputError("targets must be 0 (live) or 1 (spoof)")
    return probabilities,targets
def confusion_at_threshold(probabilities,targets,threshold:float)->dict:
    """spoof=1 is the positive class; decision is p_spoof >= threshold."""
    probabilities,targets=_check(probabilities,targets)
    predicted=(probabilities>=threshold).astype(np.int64)
    return {"tp_spoof":int(((targets==1)&(predicted==1)).sum()),"fn_spoof":int(((targets==1)&(predicted==0)).sum()),
            "fp_live":int(((targets==0)&(predicted==1)).sum()),"tn_live":int(((targets==0)&(predicted==0)).sum()),
            "total_spoof":int((targets==1).sum()),"total_live":int((targets==0).sum())}
def apcer_bpcer_acer(probabilities,targets,threshold:float)->dict:
    counts=confusion_at_threshold(probabilities,targets,threshold)
    if counts["total_spoof"]==0 or counts["total_live"]==0:
        raise MetricInputError("APCER/BPCER need both live and spoof samples")
    apcer=counts["fn_spoof"]/counts["total_spoof"]      # attack accepted as live
    bpcer=counts["fp_live"]/counts["total_live"]        # genuine rejected as attack
    return {**counts,"threshold":float(threshold),"apcer":apcer,"bpcer":bpcer,"acer":(apcer+bpcer)/2}
def accuracy(probabilities,targets,threshold:float=0.5)->float:
    probabilities,targets=_check(probabilities,targets)
    return float(((probabilities>=threshold).astype(np.int64)==targets).mean())
def balanced_accuracy(probabilities,targets,threshold:float=0.5)->float:
    counts=confusion_at_threshold(probabilities,targets,threshold)
    if counts["total_spoof"]==0 or counts["total_live"]==0: raise MetricInputError("balanced accuracy needs both classes")
    return float(0.5*(counts["tp_spoof"]/counts["total_spoof"]+counts["tn_live"]/counts["total_live"]))
def roc_curve(probabilities,targets)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    probabilities,targets=_check(probabilities,targets)
    if len(np.unique(targets))<2: raise MetricInputError("ROC needs both classes")
    order=np.argsort(-probabilities,kind="mergesort")
    scores=probabilities[order]; labels=targets[order]
    tps=np.cumsum(labels==1); fps=np.cumsum(labels==0)
    distinct=np.r_[np.where(np.diff(scores))[0],scores.size-1]
    tpr=tps[distinct]/max(int((targets==1).sum()),1); fpr=fps[distinct]/max(int((targets==0).sum()),1)
    return np.r_[0.,fpr],np.r_[0.,tpr],np.r_[np.inf,scores[distinct]]
def roc_auc(probabilities,targets)->float:
    fpr,tpr,_=roc_curve(probabilities,targets)
    return float(np.trapezoid(tpr,fpr)) if hasattr(np,"trapezoid") else float(np.trapz(tpr,fpr))
def equal_error_rate(probabilities,targets)->dict:
    fpr,tpr,thresholds=roc_curve(probabilities,targets)
    fnr=1.-tpr; index=int(np.nanargmin(np.abs(fnr-fpr)))
    return {"eer":float((fnr[index]+fpr[index])/2),"threshold":float(thresholds[index])}
def negative_log_likelihood(probabilities,targets,eps:float=1e-12)->float:
    probabilities,targets=_check(probabilities,targets)
    clipped=np.clip(probabilities,eps,1-eps)
    return float(-(targets*np.log(clipped)+(1-targets)*np.log(1-clipped)).mean())
def brier_score(probabilities,targets)->float:
    probabilities,targets=_check(probabilities,targets)
    return float(((probabilities-targets)**2).mean())
def expected_calibration_error(probabilities,targets,bins:int=15)->float:
    probabilities,targets=_check(probabilities,targets)
    edges=np.linspace(0.,1.,bins+1); error=0.
    for low,high in zip(edges[:-1],edges[1:]):
        mask=(probabilities>low)&(probabilities<=high) if low>0 else (probabilities>=low)&(probabilities<=high)
        if not mask.any(): continue
        error+=mask.mean()*abs(targets[mask].mean()-probabilities[mask].mean())
    return float(error)
def select_min_acer_threshold(probabilities,targets)->dict:
    """Deterministic sweep over candidate thresholds derived from the scores.

    Tie-break: lower ACER, then lower APCER (prefer rejecting attacks), then
    the lower threshold.
    """
    probabilities,targets=_check(probabilities,targets)
    candidates=np.unique(np.r_[0.,probabilities,1.,np.clip(probabilities+1e-9,0,1)])
    rows=[apcer_bpcer_acer(probabilities,targets,float(threshold)) for threshold in candidates]
    best=min(rows,key=lambda row:(round(row["acer"],12),round(row["apcer"],12),row["threshold"]))
    return {"selected":best,"candidates_evaluated":len(rows),
            "criterion":"min_acer","tie_break":["min_acer","min_apcer","min_threshold"]}
def summarize(probabilities,targets,threshold:float)->dict:
    return {**apcer_bpcer_acer(probabilities,targets,threshold),
            "accuracy":accuracy(probabilities,targets,threshold),
            "balanced_accuracy":balanced_accuracy(probabilities,targets,threshold),
            "roc_auc":roc_auc(probabilities,targets),**{"eer_"+k:v for k,v in equal_error_rate(probabilities,targets).items()},
            "nll":negative_log_likelihood(probabilities,targets),"brier":brier_score(probabilities,targets),
            "ece":expected_calibration_error(probabilities,targets)}
