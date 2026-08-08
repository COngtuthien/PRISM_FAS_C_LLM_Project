"""M10 metrics.

The proven primitives (APCER/BPCER/ACER, ROC, EER, NLL, Brier, ECE) come from
`prism_fas.train.metrics` and are NOT reimplemented, so M5's known-answer fixtures
and M10's bind the same code. This module adds what target evaluation needs on top:
HTER at its own declared threshold source, risk-coverage, rejection rate,
attack-wise APCER, and one honest way to say "not computable".

Label convention, unchanged from M4/M5: `live = 0`, `spoof = 1`; spoof is the
positive attack class; the decision is `p_spoof >= threshold`.

ACER and HTER are the same functional at the same threshold. They are kept distinct
because their THRESHOLD SOURCE differs — ACER at the frozen source-dev operating
threshold M10 must actually deploy, HTER at the target EER threshold, which is the
conventional cross-domain reporting point. Both sources are recorded beside the
value, so the two can never be confused for independent evidence.
"""
from __future__ import annotations
from typing import Any, Sequence
import numpy as np
from prism_fas.train.metrics import (MetricInputError, apcer_bpcer_acer, brier_score,
                                     confusion_at_threshold, equal_error_rate,
                                     expected_calibration_error, negative_log_likelihood,
                                     roc_auc, roc_curve)
from .contracts import M10ContractError, not_applicable

METRICS_SCHEMA_VERSION = "m10-metrics-v1"
DEFAULT_ECE_BINS = 15
DEFAULT_COVERAGE_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def _arrays(scores: Sequence[float], labels: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    if values.shape != targets.shape: raise M10ContractError("scores and labels must align")
    if values.size == 0: raise M10ContractError("cannot score an empty population")
    if not np.isfinite(values).all(): raise M10ContractError("non-finite scores")
    if set(np.unique(targets).tolist()) - {0, 1}:
        raise M10ContractError("labels must be 0 (live) or 1 (spoof)")
    return values, targets


def population(scores: Sequence[float], labels: Sequence[int]) -> dict[str, int]:
    _, targets = _arrays(scores, labels)
    return {"total": int(targets.size), "live": int((targets == 0).sum()),
            "spoof": int((targets == 1).sum())}


def _degenerate(counts: dict[str, int], metric: str) -> dict[str, Any] | None:
    """The exact situation the frozen 785-live-only package produces."""
    if counts["spoof"] == 0:
        return not_applicable(f"{metric} is undefined with zero attack presentations "
                              f"(live {counts['live']}, spoof 0)", population=counts)
    if counts["live"] == 0:
        return not_applicable(f"{metric} is undefined with zero bona-fide presentations "
                              f"(live 0, spoof {counts['spoof']})", population=counts)
    return None


def core_metrics(scores: Sequence[float], labels: Sequence[int], *, threshold: float,
                 threshold_source: str = "source_dev_frozen") -> dict[str, Any]:
    """APCER/BPCER/ACER at the frozen operating threshold, plus ROC-AUC, EER, HTER."""
    values, targets = _arrays(scores, labels)
    counts = population(values, targets)
    blocked = _degenerate(counts, "APCER/ACER/HTER/ROC-AUC/EER")
    bpcer = (float(confusion_at_threshold(values, targets, threshold)["fp_live"]) / counts["live"]
             if counts["live"] else None)
    if blocked is not None:
        return {"schema_version": METRICS_SCHEMA_VERSION, "population": counts,
                "threshold": float(threshold), "threshold_source": threshold_source,
                "apcer": blocked, "acer": blocked, "hter": blocked, "roc_auc": blocked, "eer": blocked,
                "bpcer": (bpcer if bpcer is not None else blocked),
                "confusion": confusion_at_threshold(values, targets, threshold)}
    at_threshold = apcer_bpcer_acer(values, targets, float(threshold))
    eer = equal_error_rate(values, targets)
    at_eer = apcer_bpcer_acer(values, targets, float(eer["threshold"]))
    return {"schema_version": METRICS_SCHEMA_VERSION, "population": counts,
            "threshold": float(threshold), "threshold_source": threshold_source,
            "apcer": float(at_threshold["apcer"]), "bpcer": float(at_threshold["bpcer"]),
            "acer": float(at_threshold["acer"]), "roc_auc": float(roc_auc(values, targets)),
            "eer": float(eer["eer"]), "eer_threshold": float(eer["threshold"]),
            # HTER = (FAR + FRR)/2 at the EER threshold; FAR = APCER, FRR = BPCER.
            "hter": float((at_eer["apcer"] + at_eer["bpcer"]) / 2.0),
            "hter_threshold_source": "target_eer",
            "confusion": {key: int(value) for key, value in at_threshold.items()
                          if key in ("tp_spoof", "fn_spoof", "fp_live", "tn_live",
                                     "total_spoof", "total_live")}}


def calibration_metrics(scores: Sequence[float], labels: Sequence[int], *,
                        bins: int = DEFAULT_ECE_BINS) -> dict[str, Any]:
    values, targets = _arrays(scores, labels)
    return {"ece": float(expected_calibration_error(values, targets, bins=bins)), "ece_bins": int(bins),
            "brier": float(brier_score(values, targets)),
            "nll": float(negative_log_likelihood(values, targets))}


def risk_coverage(scores: Sequence[float], labels: Sequence[int], confidences: Sequence[float], *,
                  threshold: float, grid: Sequence[float] = DEFAULT_COVERAGE_GRID) -> dict[str, Any]:
    """Error rate on the most-confident fraction. Needs no reject threshold.

    Ties in confidence are broken by descending confidence then ascending index, so
    the curve is deterministic for a given input order.
    """
    values, targets = _arrays(scores, labels)
    confidence = np.asarray(confidences, dtype=np.float64).reshape(-1)
    if confidence.shape != values.shape: raise M10ContractError("confidences must align with scores")
    if not np.isfinite(confidence).all(): raise M10ContractError("non-finite confidences")
    order = np.lexsort((np.arange(values.size), -confidence))
    predicted = (values[order] >= float(threshold)).astype(np.int64)
    errors = (predicted != targets[order]).astype(np.float64)
    points = []
    for coverage in grid:
        count = max(1, int(round(float(coverage) * values.size)))
        points.append({"coverage": float(coverage), "samples": int(count),
                       "risk": float(errors[:count].mean())})
    return {"points": points, "ranking": "confidence_descending", "threshold": float(threshold),
            "area_under_risk_coverage": float(np.trapezoid([p["risk"] for p in points],
                                                           [p["coverage"] for p in points])
                                              if hasattr(np, "trapezoid") else
                                              np.trapz([p["risk"] for p in points],
                                                       [p["coverage"] for p in points]))}


def rejection_metrics(decisions: Sequence[str]) -> dict[str, Any]:
    if not decisions: raise M10ContractError("cannot compute a rejection rate over zero decisions")
    rejected = sum(1 for value in decisions if value == "reject")
    return {"rejection_rate": float(rejected / len(decisions)), "rejected": int(rejected),
            "total": int(len(decisions))}


def attack_wise_apcer(scores: Sequence[float], labels: Sequence[int], families: Sequence[str], *,
                      threshold: float) -> dict[str, Any]:
    """APCER per attack family, computed POST-HOC. Never used in tuning.

    Attack family reaches this function only from the evaluator-only label
    artifact, only inside G8, and only after the prediction file is frozen.
    """
    values, targets = _arrays(scores, labels)
    names = [str(value) for value in families]
    if len(names) != values.size: raise M10ContractError("families must align with scores")
    out: dict[str, Any] = {}
    for family in sorted({name for name, target in zip(names, targets.tolist()) if target == 1}):
        mask = np.array([name == family and target == 1
                         for name, target in zip(names, targets.tolist())], dtype=bool)
        accepted = int((values[mask] < float(threshold)).sum())
        out[family] = {"attacks": int(mask.sum()), "accepted_as_live": accepted,
                       "apcer": float(accepted / int(mask.sum()))}
    if not out:
        return not_applicable("no attack family is present in the scored population")
    return {"threshold": float(threshold), "computed": "post_hoc_only", "used_in_tuning": False,
            "by_family": out}


def region_wise_summary(rows: Sequence[dict[str, Any]], *, region_order: Sequence[str]) -> dict[str, Any]:
    """How often each region was among the top-k evidence regions.

    `not_applicable` for a variant without a regional branch, rather than an
    all-zero histogram that would read as a measurement.
    """
    applicable = [row for row in rows if row.get("region_status") == "computed"]
    if not applicable:
        return not_applicable("this variant has no regional branch, so no region was ever ranked")
    counts = {name: 0 for name in region_order}
    for row in applicable:
        for index in row.get("top_region_ids") or []:
            if 0 <= int(index) < len(region_order): counts[region_order[int(index)]] += 1
    return {"rows_with_regions": len(applicable), "top_region_counts": counts,
            "mean_top_region_distance": float(np.mean([
                float(np.mean(row["region_distances"])) for row in applicable
                if row.get("region_distances")])) if any(row.get("region_distances")
                                                         for row in applicable) else None}


def threshold_table(scores: Sequence[float], labels: Sequence[int], *,
                    thresholds: Sequence[float]) -> list[dict[str, Any]]:
    values, targets = _arrays(scores, labels)
    counts = population(values, targets)
    if counts["spoof"] == 0 or counts["live"] == 0:
        raise M10ContractError("a threshold table needs both bona-fide and attack presentations")
    return [{**{key: float(value) if isinstance(value, float) else value
                for key, value in apcer_bpcer_acer(values, targets, float(threshold)).items()}}
            for threshold in thresholds]


def open_set_metrics() -> dict[str, Any]:
    """Declared `not_applicable` on this protocol, with the reason.

    SiW-Mv2 provides no unknown-class label here; deriving one from the attack
    families would be fabricating a label, which is exactly what this project does
    not do.
    """
    return not_applicable("SiW-Mv2 provides no unknown-class label on this protocol; deriving one "
                          "from the attack families would fabricate a label",
                          metrics=["auroc_unknown", "aupr_unknown", "fpr_at_95tpr"])


def summarize_seeds(values: Sequence[float]) -> dict[str, Any]:
    """Mean ± std over seeds (population std, ddof = 0), per the statistics contract."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0: raise M10ContractError("cannot summarize zero seeds")
    if not np.isfinite(array).all(): raise M10ContractError("non-finite seed values")
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0)), "min": float(array.min()),
            "max": float(array.max()), "n_seeds": int(array.size),
            "single_seed_descriptive": bool(array.size < 3)}


__all__ = ["METRICS_SCHEMA_VERSION", "MetricInputError", "attack_wise_apcer", "calibration_metrics",
           "core_metrics", "open_set_metrics", "population", "region_wise_summary",
           "rejection_metrics", "risk_coverage", "roc_curve", "summarize_seeds", "threshold_table"]
