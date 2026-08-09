"""Frame -> video aggregation, frozen by Table 57 and never changed after scoring.

    video_score      = trimmed_mean(frame_s_final, trim = 0.10)
    video_confidence = median(frame_confidence)
    video_decision   = threshold_and_reject(video_score, video_confidence)

`trimmed_mean` is the deterministic form already frozen in M5 and is imported from
there rather than reimplemented, so the two can never drift: sort ascending,
`trim_count = floor(n * 0.10)`, drop `trim_count` from each end only when
`trim_count > 0 and 2 * trim_count < n`. At the frozen 4 frames per video
`trim_count = 0`, so aggregation reduces to the plain mean of four values — a
consequence of the frozen frame plan, stated rather than hidden.

Aggregation needs no label and is never given one.
"""
from __future__ import annotations
from typing import Any, Iterable, Sequence
import numpy as np
from prism_fas.train.video_aggregation import trimmed_mean
from .contracts import M10ContractError

AGGREGATION_SCHEMA_VERSION = "m10-video-aggregation-v1"
TRIM_FRACTION = 0.10


def threshold_and_reject(video_score: float, video_confidence: float, *, threshold: float,
                         unknown_threshold: float | None = None) -> str:
    """`reject` first, then the live/spoof threshold.

    `unknown_threshold is None` means the source-side unknown/reject exposure fit
    (G6b) has not been performed, so nothing is rejected — the reject rate is 0 by
    construction, not by measurement. A target quantile is never used to invent one.
    """
    if unknown_threshold is not None and float(video_confidence) < float(unknown_threshold):
        return "reject"
    return "spoof" if float(video_score) >= float(threshold) else "live"


def aggregate_frames(rows: Iterable[dict[str, Any]], *, threshold: float,
                     unknown_threshold: float | None = None,
                     trim_fraction: float = TRIM_FRACTION,
                     score_key: str = "decision_score", confidence_key: str = "confidence",
                     fused_key: str = "s_final",
                     video_key: str = "video_id", order_key: str = "sample_id") -> list[dict[str, Any]]:
    """Deterministic grouping: videos in sorted id order, frames in sorted id order.

    `video_score` aggregates the DECISION quantity (`decision_score`, contract
    section 2b), because that is the quantity the frozen threshold belongs to.
    `video_fused_score` aggregates the declared fusion `s_final` under exactly the
    same trimmed mean, and is carried alongside so the fused evidence is still
    reported on the threshold-free metrics. For a variant with no regional branch
    the two are identical by construction.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if video_key not in row: raise M10ContractError(f"a frame row carries no {video_key!r}")
        grouped.setdefault(str(row[video_key]), []).append(row)
    aggregates: list[dict[str, Any]] = []
    for video_id in sorted(grouped):
        frames = sorted(grouped[video_id], key=lambda row: str(row[order_key]))
        scores = [float(frame[score_key]) for frame in frames]
        confidences = [float(frame[confidence_key]) for frame in frames]
        fused = [float(frame.get(fused_key, frame[score_key])) for frame in frames]
        if not np.isfinite(scores).all() or not np.isfinite(confidences).all():
            raise M10ContractError(f"video {video_id}: non-finite frame score or confidence")
        score, trim = trimmed_mean(scores, trim_fraction)
        fused_score, _ = trimmed_mean(fused, trim_fraction)
        confidence = float(np.median(np.asarray(confidences, dtype=np.float64)))
        aggregates.append({"video_id": video_id, "frames": len(frames), "trim_count": trim,
                           "video_score": score, "video_fused_score": fused_score,
                           "video_confidence": confidence,
                           "decision": threshold_and_reject(score, confidence, threshold=threshold,
                                                            unknown_threshold=unknown_threshold)})
    return aggregates


def aggregation_config(threshold: float, unknown_threshold: float | None,
                       trim_fraction: float = TRIM_FRACTION) -> dict[str, Any]:
    """The exact aggregation contract bound into every PREDICTION_LOCK."""
    return {"schema_version": AGGREGATION_SCHEMA_VERSION, "video_score": "trimmed_mean",
            "video_score_key": "decision_score", "video_fused_score_key": "s_final",
            "trim": float(trim_fraction), "video_confidence": "median",
            "video_decision": "threshold_and_reject", "threshold": float(threshold),
            "unknown_threshold": (None if unknown_threshold is None else float(unknown_threshold)),
            "group_by": "video_id", "frame_order": "sample_id"}


def rejection_rate(aggregates: Sequence[dict[str, Any]]) -> float:
    if not aggregates: raise M10ContractError("cannot compute a rejection rate over zero videos")
    return float(sum(1 for row in aggregates if row["decision"] == "reject") / len(aggregates))
