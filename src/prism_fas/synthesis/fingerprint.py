from __future__ import annotations
import hashlib, json
from typing import Any
import numpy as np

FINGERPRINT_SCHEMA_VERSION = "m8-fingerprint-v1"
FINGERPRINT_DIM = 24
MAD_SCALE = 1.4826            # MAD -> sigma for a normal distribution
MAD_EPSILON = 1.0e-6
TAU_FP_PERCENTILE = 99.0
FEATURE_NAMES = (
    tuple(f"high_abs_mean_{index}" for index in range(9))
    + tuple(f"high_std_{index}" for index in range(9))
    + ("edge_energy_r", "edge_energy_g", "edge_energy_b", "laplacian_variance",
       "mean_saturation", "channel_balance_magnitude"))
assert len(FEATURE_NAMES) == FINGERPRINT_DIM


class FingerprintError(ValueError):
    """A fingerprint feature vector or reference is malformed."""


def _haar_high(image: np.ndarray) -> np.ndarray:
    """The 9 Haar high-frequency channels, same orthonormal convention as dwt.py."""
    x00, x01 = image[:, 0::2, 0::2], image[:, 0::2, 1::2]
    x10, x11 = image[:, 1::2, 0::2], image[:, 1::2, 1::2]
    lh = (x00 + x01 - x10 - x11) * np.float32(0.5)
    hl = (x00 - x01 + x10 - x11) * np.float32(0.5)
    hh = (x00 - x01 - x10 + x11) * np.float32(0.5)
    return np.concatenate((lh, hl, hh), axis=0).astype(np.float32)


def _gradient_energy(channel: np.ndarray) -> float:
    dy = np.abs(channel[1:, :] - channel[:-1, :]).mean() if channel.shape[0] > 1 else 0.0
    dx = np.abs(channel[:, 1:] - channel[:, :-1]).mean() if channel.shape[1] > 1 else 0.0
    return float(0.5 * (dx + dy))


def _laplacian_variance(grey: np.ndarray) -> float:
    padded = np.pad(grey, 1, mode="edge")
    laplacian = (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
                 - 4.0 * padded[1:-1, 1:-1])
    return float(np.var(laplacian))


def fingerprint_features(image: np.ndarray) -> np.ndarray:
    """Fixed 24-dimensional high-frequency descriptor of an RGB [3,H,W] float
    image in [0,1]. Pure NumPy: no trainable probe is involved."""
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 3: raise FingerprintError(f"expected [3,H,W], got {values.shape}")
    high = _haar_high(values)
    grey = (0.299 * values[0] + 0.587 * values[1] + 0.114 * values[2]).astype(np.float32)
    channel_means = values.reshape(3, -1).mean(axis=1)
    vector = np.concatenate((
        np.abs(high).reshape(9, -1).mean(axis=1),
        high.reshape(9, -1).std(axis=1),
        np.asarray([_gradient_energy(values[index]) for index in range(3)], dtype=np.float32),
        np.asarray([_laplacian_variance(grey)], dtype=np.float32),
        np.asarray([float((values.max(axis=0) - values.min(axis=0)).mean())], dtype=np.float32),
        np.asarray([float(np.abs(channel_means - channel_means.mean()).max())], dtype=np.float32),
    )).astype(np.float32)
    if vector.shape != (FINGERPRINT_DIM,): raise FingerprintError(f"fingerprint must be [{FINGERPRINT_DIM}]")
    if not np.isfinite(vector).all(): raise FingerprintError("fingerprint contains non-finite values")
    return vector


def robust_reference(vectors: np.ndarray) -> dict[str, list[float]]:
    """Per-dimension median and MAD (epsilon-floored) — no mean/std, so a few
    outliers cannot move the reference."""
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != FINGERPRINT_DIM:
        raise FingerprintError(f"expected [N,{FINGERPRINT_DIM}], got {values.shape}")
    if not len(values): raise FingerprintError("cannot fit a reference from zero samples")
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    scale = np.maximum(MAD_SCALE * mad, MAD_EPSILON)
    return {"median": [float(value) for value in median], "scale": [float(value) for value in scale],
            "count": int(len(values))}


def score_against(vector: np.ndarray, reference: dict[str, list[float]]) -> float:
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    median = np.asarray(reference["median"], dtype=np.float64)
    scale = np.asarray(reference["scale"], dtype=np.float64)
    z = (values - median) / np.maximum(scale, MAD_EPSILON)
    return float(np.mean(z ** 2))


def fingerprint_score(vector: np.ndarray, references: dict[str, dict[str, list[float]]],
                      *, domains: tuple[str, ...] | None = None) -> float:
    """Mean squared robust z-score to the **closest** valid real-spoof
    source-domain reference."""
    usable = [name for name in (domains or tuple(references)) if name in references]
    if not usable: raise FingerprintError("no valid source-domain reference available")
    return float(min(score_against(vector, references[name]) for name in usable))


def leave_one_record_out_scores(vectors: np.ndarray, domains: list[str], records: list[str]) -> list[float]:
    """Score every real spoof sample against references refitted without its own
    `source_record_id`, so a sample never calibrates against itself."""
    values = np.asarray(vectors, dtype=np.float64)
    if len(values) != len(domains) or len(values) != len(records):
        raise FingerprintError("vectors, domains and records must align")
    scores: list[float] = []
    for position in range(len(values)):
        held = records[position]
        references: dict[str, dict[str, list[float]]] = {}
        for domain in sorted(set(domains)):
            keep = [index for index in range(len(values))
                    if domains[index] == domain and records[index] != held]
            if keep: references[domain] = robust_reference(values[keep])
        scores.append(fingerprint_score(values[position], references))
    return scores


def fit_fingerprint_reference(vectors: np.ndarray, domains: list[str], records: list[str]) -> dict[str, Any]:
    """Fit the source-only reference and derive `tau_fp` from leave-one-record-out
    scores. Generated candidates never participate."""
    values = np.asarray(vectors, dtype=np.float64)
    references = {domain: robust_reference(values[[index for index in range(len(values)) if domains[index] == domain]])
                  for domain in sorted(set(domains))}
    loo = leave_one_record_out_scores(values, domains, records)
    tau = float(np.percentile(loo, TAU_FP_PERCENTILE))
    payload = {"fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION, "dimension": FINGERPRINT_DIM,
               "feature_names": list(FEATURE_NAMES), "references": references,
               "domains": sorted(set(domains)), "samples": int(len(values)),
               "distinct_records": len(set(records)),
               "tau_fp": tau, "tau_fp_percentile": TAU_FP_PERCENTILE,
               "leave_one_record_out": {"count": len(loo), "min": float(np.min(loo)),
                                        "median": float(np.median(loo)), "max": float(np.max(loo)),
                                        "p99": tau},
               "method": "robust_median_mad_closest_domain", "trainable_probe": False,
               "used_generated_candidates": False, "used_source_dev": False, "used_target": False}
    payload["reference_sha256"] = hashlib.sha256(
        json.dumps({"references": references, "tau_fp": tau}, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")).hexdigest()
    return payload
