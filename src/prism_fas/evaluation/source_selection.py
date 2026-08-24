"""The §15.4 source-only selection tuples, computed once for C7 and C8.

Both stages minimize a frozen metric tuple lexicographically and neither may own
its own version of it. C7 ranks CONFIGURATIONS by it and C8 ranks CHECKPOINTS
inside a row by it; the quantities are identical, and two implementations would
be two answers to "which checkpoint is best".

Three things the tuple is that the trainer's own epoch metric is not, and all
three are the reason this module exists rather than a convenience wrapper:

* **It is video-level.** §15.4 names `video_ACER` and `video_BPCER`. The
  trainer's `source_dev/acer` is per frame. Frames of one video are not
  independent, so a frame-level ACER is a different number with a different
  variance, and selecting on it would select on it.
* **It is calibrated.** `NLL` and `ECE` are read after temperature scaling
  fitted on source_dev, because that is the score the operating threshold is
  applied to. Reading them off raw sigmoid outputs would rank configurations by
  a score nothing downstream uses.
* **It is equal-weight across domains for P3-ready.** Pooling CASIA-dev and
  MSU-dev and averaging would weight each domain by however many rows it
  contributes. §15.4 says equal weight, so each domain's metric is computed
  separately and then averaged.

Everything numeric is imported from `train.metrics`, `train.calibration` and
`train.video_aggregation`. This module decides WHICH numbers and in WHAT order,
and computes none of them itself.

`source_dev` only. No target split, label or metric is reachable from here.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "prism-source-selection-v1"

#: §15.4 / Table 65: video score is the trimmed mean of the frame scores.
TRIM_FRACTION = 0.10

#: §15.4, P1/P2. Minimized lexicographically.
P1P2_TUPLE: tuple[str, ...] = ("video_ACER", "video_BPCER", "NLL", "ECE", "epoch")

#: §15.4, P3-ready: equal-weight CASIA-dev and MSU-dev.
P3_READY_TUPLE: tuple[str, ...] = (
    "mean_domain_video_ACER", "max_domain_video_ACER", "mean_domain_video_BPCER",
    "mean_domain_NLL", "mean_domain_ECE", "epoch")

TUPLES: dict[str, tuple[str, ...]] = {"P1P2": P1P2_TUPLE, "P3_READY": P3_READY_TUPLE}

#: Which protocol uses which tuple. §19 + §15.4, transcribed rather than inferred.
TUPLE_FOR_PROTOCOL: dict[str, str] = {"P1": "P1P2", "P2": "P1P2", "P3": "P3_READY"}

#: §19: the source domains each protocol trains, selects and calibrates on.
PROTOCOL_DOMAINS: dict[str, tuple[str, ...]] = {
    "P1": ("casia_fasd",),
    "P2": ("msu_mfsd",),
    "P3": ("casia_fasd", "msu_mfsd"),
}

#: §19: the OTHER source domain a P1/P2 row is evaluated on. Diagnostic only —
#: §15.4 selects and calibrates on the protocol's own source_dev, so a number
#: computed here may never enter a selection tuple, a threshold or a temperature.
#: P3's test domain is the held-out target, which is predicted at C11 and is not
#: reachable from this module at all.
CROSS_SOURCE_DOMAINS: dict[str, tuple[str, ...]] = {
    "P1": ("msu_mfsd",),
    "P2": ("casia_fasd",),
    "P3": (),
}


class SourceSelectionError(ValueError):
    """The selection tuple cannot be computed from these predictions."""


def domains_for(protocol: str) -> tuple[str, ...]:
    try:
        return PROTOCOL_DOMAINS[protocol]
    except KeyError:
        raise SourceSelectionError(
            f"unknown protocol {protocol!r}; §19 declares {sorted(PROTOCOL_DOMAINS)}"
        ) from None


def tuple_for(protocol: str) -> tuple[str, ...]:
    return TUPLES[TUPLE_FOR_PROTOCOL[protocol]]


def source_dev_frame_rows(trainer: Any) -> list[dict[str, Any]]:
    """One row per source_dev frame: id, video, domain, label and decision logit.

    The logit is `output.global_logit`, which is the DECISION logit slot whatever
    the variant names it — `fused_logit_R` for a `glr_concat` Track-R row and
    `global_logit_G` otherwise. The name travels with the rows so a caller can
    prove the quantity it calibrated is the quantity it thresholded.
    """
    import torch

    validation = trainer.validation()
    model = trainer.model
    model.eval()
    rows: list[dict[str, Any]] = []
    positions = list(validation.positions)
    size = int(trainer.config.validation_batch_size)
    with torch.no_grad():
        for start in range(0, len(positions), size):
            chunk = positions[start:start + size]
            batch = validation.batch(chunk).to(trainer.device)
            logits = model(batch).global_logit.detach().float().cpu().numpy().reshape(-1)
            labels = batch.label.detach().cpu().numpy().reshape(-1)
            for offset, position in enumerate(chunk):
                rows.append({
                    "sample_id": validation.sample_id_of(position),
                    "source_record_id": validation.record_of(position),
                    "dataset": validation.domain_of[position],
                    "label": int(labels[offset]),
                    "logit": float(logits[offset]),
                })
    if not rows:
        raise SourceSelectionError("source_dev produced no predictions")
    return rows


def video_rows(frames: Iterable[Mapping[str, Any]], *,
               temperature: float) -> list[dict[str, Any]]:
    """Frame rows collapsed to one row per video, by trimmed mean of p_spoof.

    Grouped on `source_record_id` — the safe record identifier the loader already
    carries — so the grouping key is a property of the package rather than of a
    filename convention. A record whose frames disagree about the label is a
    package defect and raises rather than being resolved by majority.
    """
    from prism_fas.train.calibration import apply_temperature
    from prism_fas.train.video_aggregation import trimmed_mean

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in frames:
        grouped.setdefault(str(row["source_record_id"]), []).append(dict(row))

    out: list[dict[str, Any]] = []
    for record_id in sorted(grouped):
        members = sorted(grouped[record_id], key=lambda row: str(row["sample_id"]))
        labels = {int(row["label"]) for row in members}
        if len(labels) != 1:
            raise SourceSelectionError(
                f"source_dev record {record_id!r} carries frames with labels {sorted(labels)}; "
                "a video is live or spoof and a selection metric may not average the two")
        domains = {str(row["dataset"]) for row in members}
        if len(domains) != 1:
            raise SourceSelectionError(
                f"source_dev record {record_id!r} spans domains {sorted(domains)}")
        probabilities = apply_temperature(
            np.asarray([row["logit"] for row in members], dtype=np.float64), temperature)
        score, trim = trimmed_mean([float(value) for value in probabilities], TRIM_FRACTION)
        out.append({"source_record_id": record_id, "dataset": domains.pop(),
                    "label": labels.pop(), "frames": len(members), "trim_count": trim,
                    "video_score": float(score)})
    return out


def _metrics(videos: Sequence[Mapping[str, Any]], *, threshold: float) -> dict[str, float]:
    from prism_fas.train.metrics import (apcer_bpcer_acer, expected_calibration_error,
                                         negative_log_likelihood)

    scores = np.asarray([row["video_score"] for row in videos], dtype=np.float64)
    labels = np.asarray([row["label"] for row in videos], dtype=np.int64)
    if not scores.size:
        raise SourceSelectionError("no videos to score")
    rates = apcer_bpcer_acer(scores, labels, threshold)
    return {
        "video_ACER": float(rates["acer"]),
        "video_APCER": float(rates["apcer"]),
        "video_BPCER": float(rates["bpcer"]),
        "NLL": float(negative_log_likelihood(scores, labels)),
        "ECE": float(expected_calibration_error(scores, labels)),
        "videos": int(scores.size),
    }


def cross_source_domains_for(protocol: str) -> tuple[str, ...]:
    try:
        return CROSS_SOURCE_DOMAINS[protocol]
    except KeyError:
        raise SourceSelectionError(
            f"unknown protocol {protocol!r}; §19 declares {sorted(CROSS_SOURCE_DOMAINS)}"
        ) from None


def evaluate(frames: Sequence[Mapping[str, Any]], *, protocol: str, temperature: float,
             threshold: float, epoch: int,
             decision_logit_name: str, decision_score_name: str,
             domains: Sequence[str] | None = None,
             role: str = "selection") -> dict[str, Any]:
    """Every §15.4 field for one checkpoint, plus the protocol's ranking tuple.

    `threshold` and `temperature` are supplied rather than fitted here: both are
    fitted on the protocol's own source_dev by the caller's calibration step, and
    re-fitting them per evaluation would produce a metric for a threshold the run
    never used.

    `domains` overrides the protocol's own domains, which is how a P1/P2 row gets
    its CROSS-SOURCE diagnostic: the same frozen temperature and threshold applied
    to the other source domain. `role` is carried into the payload so a consumer
    can tell a selection number from a diagnostic one without inferring it from
    the domain list — §15.4 forbids the second becoming the first.
    """
    expected = tuple(domains) if domains is not None else domains_for(protocol)
    if not expected:
        raise SourceSelectionError(
            f"protocol {protocol} declares no domains for role {role!r}")
    videos = video_rows(frames, temperature=temperature)
    present = sorted({str(row["dataset"]) for row in videos})
    unexpected = sorted(set(present) - set(expected))
    if unexpected:
        raise SourceSelectionError(
            f"protocol {protocol} selects on {list(expected)} but the predictions "
            f"contain domain(s) {unexpected}")
    missing = sorted(set(expected) - set(present))
    if missing:
        raise SourceSelectionError(
            f"protocol {protocol} selects on {list(expected)} but the predictions "
            f"contain no rows for {missing}")

    pooled = _metrics(videos, threshold=threshold)
    per_domain = {name: _metrics([row for row in videos if row["dataset"] == name],
                                 threshold=threshold)
                  for name in expected}

    # Equal weight, so a plain mean over the per-domain numbers rather than over
    # the pooled rows. With one domain the two coincide, which is why P1/P2 can
    # read the pooled fields directly.
    def mean(field: str) -> float:
        return float(np.mean([item[field] for item in per_domain.values()]))

    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": protocol,
        "role": role,
        "is_selection_signal": role == "selection",
        "selection_tuple_name": TUPLE_FOR_PROTOCOL[protocol],
        "selection_tuple": list(tuple_for(protocol)),
        "domains": list(expected),
        "epoch": int(epoch),
        "temperature": float(temperature),
        "threshold": float(threshold),
        "decision_logit_name": decision_logit_name,
        "decision_score_name": decision_score_name,
        "video_aggregation": {"rule": "trimmed_mean", "trim_fraction": TRIM_FRACTION,
                              "group_key": "source_record_id"},
        "pooled": pooled,
        "per_domain": per_domain,
        "frames": len(frames),
        "target_paths_resolved": 0,
        "target_labels_resolved": 0,
        **{key: value for key, value in pooled.items() if key != "videos"},
        "mean_domain_video_ACER": mean("video_ACER"),
        "max_domain_video_ACER": float(max(item["video_ACER"] for item in per_domain.values())),
        "mean_domain_video_BPCER": mean("video_BPCER"),
        "mean_domain_NLL": mean("NLL"),
        "mean_domain_ECE": mean("ECE"),
    }
    payload["ranking_tuple"] = {name: payload[name] for name in tuple_for(protocol)}
    payload["identity"] = hashlib.sha256(
        json.dumps({key: payload[key] for key in sorted(payload) if key != "identity"},
                   sort_keys=True, separators=(",", ":"),
                   default=str).encode("utf-8")).hexdigest()
    return payload


def fit_source_dev_calibration(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Temperature and operating threshold, fitted on source_dev frames alone.

    The threshold is chosen on the VIDEO scores the calibrated frames aggregate
    to, not on the frames, because the video score is what a decision is taken
    on. Fitting a frame threshold and applying it to a trimmed mean would apply
    an operating point to a quantity it was never chosen for.
    """
    from prism_fas.train.calibration import fit_temperature
    from prism_fas.train.metrics import select_min_acer_threshold

    logits = np.asarray([row["logit"] for row in frames], dtype=np.float64)
    labels = np.asarray([row["label"] for row in frames], dtype=np.int64)
    temperature = float(fit_temperature(logits, labels))
    videos = video_rows(frames, temperature=temperature)
    scores = np.asarray([row["video_score"] for row in videos], dtype=np.float64)
    video_labels = np.asarray([row["label"] for row in videos], dtype=np.int64)
    selection = select_min_acer_threshold(scores, video_labels)
    return {
        "schema_version": SCHEMA_VERSION,
        "split": "source_dev",
        "temperature": temperature,
        "threshold": float(selection["selected"]["threshold"]),
        "threshold_criterion": selection["criterion"],
        "threshold_tie_break": selection["tie_break"],
        "fitted_on": "source_dev frames (temperature) and source_dev videos (threshold)",
        "frames": len(frames),
        "videos": len(videos),
        "uses_target": False,
    }


__all__ = ["SCHEMA_VERSION", "TRIM_FRACTION", "P1P2_TUPLE", "P3_READY_TUPLE", "TUPLES",
           "TUPLE_FOR_PROTOCOL", "PROTOCOL_DOMAINS", "CROSS_SOURCE_DOMAINS",
           "SourceSelectionError", "domains_for", "cross_source_domains_for",
           "tuple_for", "source_dev_frame_rows", "video_rows",
           "evaluate", "fit_source_dev_calibration"]
