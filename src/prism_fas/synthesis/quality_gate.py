from __future__ import annotations
import hashlib, json
from dataclasses import asdict, dataclass
from typing import Any
import numpy as np

QUALITY_GATE_SCHEMA_VERSION = "m8-quality-gate-v1"
EPS = 1.0e-6
# M9's PromptHead does not exist yet, so no prompt score is invented.
RECIPE_MATCH = "not_applicable"
SUPPORT_OVERLAP_MIN = 0.90
HARD_GATES = ("face_detection", "identity", "landmark", "parsing_dice", "outside_mask",
              "artifact_strength", "fingerprint", "support_overlap")
QUALITY_COMPONENTS = ("q_fd", "q_id", "q_lm", "q_parse", "q_strength", "q_fp", "q_support")


class QualityGateError(ValueError):
    """A quality metric or threshold set is malformed."""


def strength_bounds(requested: float) -> tuple[float, float]:
    """Dynamic bounds around the requested mean recipe artifact strength."""
    a = float(requested)
    return max(0.01, 0.25 * a), min(0.50, 1.75 * a)


def landmark_nme(predicted: np.ndarray, reference: np.ndarray) -> float:
    """Normalized mean error between two 5-point sets, normalized by the
    inter-ocular distance with an epsilon guard."""
    a = np.asarray(predicted, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(reference, dtype=np.float64).reshape(-1, 2)
    if a.shape != b.shape or a.shape[0] < 2: raise QualityGateError("landmark sets must align and hold >= 2 points")
    inter_ocular = float(np.linalg.norm(b[0] - b[1]))
    return float(np.mean(np.linalg.norm(a - b, axis=1)) / max(inter_ocular, EPS))


def parsing_dice(predicted: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    """Mean per-class Dice between two parsing maps, restricted to `mask`.

    Used outside the exact edit mask, where the generated image must still parse
    like the original face.
    """
    a = np.asarray(predicted); b = np.asarray(reference); keep = np.asarray(mask).astype(bool)
    if a.shape != b.shape or a.shape != keep.shape: raise QualityGateError("parsing maps and mask must align")
    if not keep.any(): return 1.0
    classes = sorted(set(np.unique(a[keep]).tolist()) | set(np.unique(b[keep]).tolist()))
    scores = []
    for value in classes:
        left, right = (a == value) & keep, (b == value) & keep
        denominator = int(left.sum()) + int(right.sum())
        if denominator == 0: continue
        scores.append(2.0 * float((left & right).sum()) / denominator)
    return float(np.mean(scores)) if scores else 1.0


def support_overlap(exact_mask: np.ndarray, requested_mask: np.ndarray) -> float:
    exact = np.asarray(exact_mask).astype(bool)
    requested = np.asarray(requested_mask).astype(bool)
    changed = int(exact.sum())
    if changed == 0: return 0.0
    return float(int((exact & requested).sum()) / changed)


def _clip01(value: float) -> float:
    return float(min(max(float(value), 0.0), 1.0))


def triangular_strength_score(measured: float, requested: float) -> float:
    """1 at the requested strength, falling linearly to 0 at each dynamic bound."""
    lower, upper = strength_bounds(requested)
    a = float(requested)
    if not lower <= measured <= upper: return 0.0
    if measured <= a: return _clip01((measured - lower) / max(a - lower, EPS))
    return _clip01((upper - measured) / max(upper - a, EPS))


@dataclass(frozen=True)
class Thresholds:
    tau_fd: float
    tau_id: float
    tau_lm: float
    tau_parse: float
    tau_out: float
    tau_fp: float

    def as_dict(self) -> dict[str, float]: return {key: float(value) for key, value in asdict(self).items()}

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Thresholds":
        missing = [key for key in ("tau_fd", "tau_id", "tau_lm", "tau_parse", "tau_out", "tau_fp") if key not in payload]
        if missing: raise QualityGateError(f"threshold set is missing {missing}")
        return cls(**{key: float(payload[key]) for key in
                      ("tau_fd", "tau_id", "tau_lm", "tau_parse", "tau_out", "tau_fp")})


def evaluate(metrics: dict[str, Any], thresholds: Thresholds) -> dict[str, Any]:
    """Apply every hard gate and compute the quality weight q.

    `q` is a sample weight for M9, never a live/spoof label, and a candidate is
    never rejected for a low `q` when all hard gates pass.
    """
    for key in ("face_detection_score", "identity_cosine", "landmark_nme", "outside_mask_parsing_dice",
                "outside_mask_max_error", "measured_artifact_strength", "requested_artifact_strength",
                "fingerprint_score", "support_overlap"):
        if key not in metrics: raise QualityGateError(f"metric {key!r} is missing")
        value = float(metrics[key])
        if not np.isfinite(value): raise QualityGateError(f"metric {key!r} is not finite")
    requested = float(metrics["requested_artifact_strength"])
    measured = float(metrics["measured_artifact_strength"])
    lower, upper = strength_bounds(requested)
    gates = {
        "face_detection": float(metrics["face_detection_score"]) >= thresholds.tau_fd,
        "identity": float(metrics["identity_cosine"]) >= thresholds.tau_id,
        "landmark": float(metrics["landmark_nme"]) <= thresholds.tau_lm,
        "parsing_dice": float(metrics["outside_mask_parsing_dice"]) >= thresholds.tau_parse,
        "outside_mask": float(metrics["outside_mask_max_error"]) == thresholds.tau_out,
        "artifact_strength": lower <= measured <= upper,
        "fingerprint": float(metrics["fingerprint_score"]) <= thresholds.tau_fp,
        "support_overlap": float(metrics["support_overlap"]) >= SUPPORT_OVERLAP_MIN}
    components = {
        "q_fd": _clip01((float(metrics["face_detection_score"]) - thresholds.tau_fd) / max(1.0 - thresholds.tau_fd, EPS)),
        "q_id": _clip01((float(metrics["identity_cosine"]) - thresholds.tau_id) / max(1.0 - thresholds.tau_id, EPS)),
        "q_lm": _clip01((thresholds.tau_lm - float(metrics["landmark_nme"])) / max(thresholds.tau_lm, EPS)),
        "q_parse": _clip01((float(metrics["outside_mask_parsing_dice"]) - thresholds.tau_parse) / max(1.0 - thresholds.tau_parse, EPS)),
        "q_strength": triangular_strength_score(measured, requested),
        "q_fp": _clip01(1.0 - float(metrics["fingerprint_score"]) / max(thresholds.tau_fp, EPS)),
        "q_support": _clip01(float(metrics["support_overlap"]))}
    logs = [np.log(max(components[name], EPS)) for name in QUALITY_COMPONENTS]
    q = float(np.float32(np.exp(float(np.mean(logs)))))
    if not np.isfinite(q): raise QualityGateError("q is not finite")
    q = float(np.float32(min(max(q, 0.0), 1.0)))
    failed = sorted(name for name, passed in gates.items() if not passed)
    return {"quality_gate_schema_version": QUALITY_GATE_SCHEMA_VERSION,
            "accepted": not failed, "failed_gates": failed, "gates": gates,
            "quality_components": {name: round(float(components[name]), 8) for name in QUALITY_COMPONENTS},
            "q": q, "recipe_match": RECIPE_MATCH,
            "strength_bounds": {"lower": round(lower, 8), "upper": round(upper, 8)},
            "threshold_hash": thresholds.sha256(),
            "metrics": {key: (float(value) if isinstance(value, (int, float, np.floating)) else value)
                        for key, value in metrics.items()}}


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in results if row["accepted"]]
    rejected = [row for row in results if not row["accepted"]]
    reasons: dict[str, int] = {}
    for row in rejected:
        for name in row["failed_gates"]: reasons[name] = reasons.get(name, 0) + 1
    values = [row["q"] for row in accepted]
    return {"evaluated": len(results), "accepted": len(accepted), "rejected": len(rejected),
            "failed_gate_counts": dict(sorted(reasons.items())),
            "q_statistics": ({"min": float(np.min(values)), "median": float(np.median(values)),
                              "mean": float(np.mean(values)), "max": float(np.max(values))}
                             if values else {}),
            "recipe_match": RECIPE_MATCH}
