"""§11.4 NOMINAL threshold inheritance from the frozen Version-B calibration.

§11.4 is a two-branch rule, and only the second branch fits a threshold:

    For each gate metric, NOMINAL uses the unique inherited Version-B threshold
    when semantically compatible. If no semantically compatible inherited
    threshold exists for a metric, derive source-only thresholds from the frozen
    reference distribution ... for higher-is-better use STRICT/NOMINAL/PERMISSIVE
    = 10th/5th/1st percentile; for lower-is-better use 90th/95th/99th.

The first C6 executor fitted every NOMINAL fresh through
`quality_calibration.calibrate`, which takes the second branch unconditionally.
That is wrong here, and not harmlessly: `calibrate` recomputes `tau_id` as the
1st percentile of benign self-similarity (~0.9995), `tau_lm` as p99 benign
(~0.00214) and `tau_parse` as p1 benign (~0.87478) — and those three are exactly
the values Version B itself examined and replaced. Refitting would have
resurrected superseded science.

WHY EVERY METRIC IS COMPATIBLE HERE. Compatibility is not inferred from the
variable name. It is provable by identity: `quality_gate.py`,
`quality_calibration.py`, `quality_models.py`, `synthetic_bank.py` (the
`CandidateEvaluator`), `identity_calibration.py`, `structural_calibration.py` and
`fingerprint.py` are byte-identical between the frozen Version-B tree and this
one; the three pinned models resolve to the same SHA-256s; and Version B
calibrated on the same M3B source package this repository uses
(`b1cf29b6…dc6`). Same measurement code, same models, same population, same
comparator directions, same scale. So all six thresholds are inherited and the
derived branch is currently unused — it stays implemented because a future
metric without an inherited threshold must take it.

THE VERSION-B SUPERSESSION CHAIN, read out of the artifacts themselves:

    v1  reports/m8/quality_calibration.json      — the M8 base calibration
    v2  reports/m8/quality_calibration_v2.json   — `tau_id_v2 = max(tau_genuine,
        tau_impostor)`; demotes v1's value to `v1_tau_id_informational_only`
    v3  reports/m8/quality_calibration_v3.json   — `tau_lm_v3`, `tau_parse_v3`;
        records `tau_lm_v1_superseded` and `tau_parse_v1_superseded`, and carries
        the rest forward as `unchanged_from_v2`

v3 therefore holds the cumulative final six. Every metric has exactly one
authoritative value and none had to be chosen.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "prism-c6-threshold-inheritance-v1"

#: The immutable Version-B tree these values come from.
VERSION_B_COMMIT = "7799f7decd35db6987ce4578824e5bd8d9eab4ae"
VERSION_B_TAG = "m10-blind-evaluation-checkpoint"

#: The unique authoritative final Version-B calibration artifact.
VERSION_B_ARTIFACT = "reports/m8/quality_calibration_v3.json"
VERSION_B_ARTIFACT_SHA256 = (
    "a21cb3e168ab04b1f1fc06b4cc311a12357316e68d2cfcdc6f82395aa08d4c2c")
#: `threshold_sha256` as recorded inside that artifact.
VERSION_B_THRESHOLD_SHA256 = (
    "8fa2648643cd526730497ae2d717e17684dda3ecea361fc84929db07ac03bb19")

#: The M3B source package Version B calibrated on — the one this repository uses.
VERSION_B_PACKAGE_IDENTITY = (
    "b1cf29b69a165ed5d9e074fc8127c17fbf057723edf9e272048ec3a564eb9dc6")

#: Per-metric NOMINAL provenance.
INHERITED_VERSION_B = "INHERITED_VERSION_B"
SOURCE_REFERENCE_DERIVED = "SOURCE_REFERENCE_DERIVED"
FROZEN_RANGE_CONSTRAINT = "FROZEN_RANGE_CONSTRAINT"

#: The frozen Version-B NOMINAL set, transcribed from the artifact above.
#: Verified against the live artifact by `verify_version_b_artifact` whenever the
#: Version-B tree is present; vendored so the GPU host does not need it mounted.
INHERITED_NOMINAL: dict[str, float] = {
    "tau_fd": 0.5,
    "tau_id": 0.547440037939055,
    "tau_lm": 0.00836817528937794,
    "tau_parse": 0.7094826178704915,
    "tau_out": 0.0,
    "tau_fp": 5.687657785453908,
}

#: The metric-by-metric audit, serialized into the C6 artifacts so a reader can
#: check the reasoning rather than take it on trust.
PROVENANCE: dict[str, dict[str, Any]] = {
    "tau_fd": {
        "metric": "face detection score",
        "direction": "higher_is_better", "comparator": ">=",
        "version_b_rule": ("pinned SCRFD production threshold "
                           "(scrfd_source_policy_v1); never fitted, identical in "
                           "v1, v2 and v3"),
        "version_b_value": 0.5,
        "nominal_source": INHERITED_VERSION_B,
        "semantic_compatibility": "YES",
        "compatibility_reason": (
            "the same pinned SCRFD ONNX (sha256 5838f7fe…) at the same input "
            "size, invoked by byte-identical detector code; the threshold is a "
            "property of that model, not of a fitted population"),
    },
    "tau_id": {
        "metric": "identity similarity (AdaFace cosine)",
        "direction": "higher_is_better", "comparator": ">=",
        "version_b_rule": "tau_id_v2 = max(tau_genuine, tau_impostor)",
        "version_b_value": 0.547440037939055,
        "superseded_version_b_value": 0.9995203357934952,
        "superseded_note": ("v1's p1-of-benign-self-similarity was demoted by "
                            "Version B itself to `v1_tau_id_informational_only`; "
                            "refitting would resurrect it"),
        "nominal_source": INHERITED_VERSION_B,
        "semantic_compatibility": "YES",
        "compatibility_reason": (
            "the same AdaFace weight (sha256 43bd2d57…) produces both, and "
            "`quality_gate.evaluate` applies the same `identity_cosine >= tau_id` "
            "comparator to the same candidate-versus-source cosine it did in "
            "Version B; the threshold is the identity-separation boundary in both"),
    },
    "tau_lm": {
        "metric": "landmark NME",
        "direction": "lower_is_better", "comparator": "<=",
        "version_b_rule": "tau_lm_v3 = p99(same-image structural landmark NME)",
        "version_b_value": 0.00836817528937794,
        "superseded_version_b_value": 0.002135227532959269,
        "superseded_note": "recorded in v3 as `tau_lm_v1_superseded`",
        "nominal_source": INHERITED_VERSION_B,
        "semantic_compatibility": "YES",
        "compatibility_reason": (
            "the same SCRFD landmarks and the same NME definition in "
            "byte-identical `CandidateEvaluator` code; v3 calibrated it on "
            "same-image structural transforms, which is the same quantity a "
            "candidate-versus-its-own-source measurement produces"),
    },
    "tau_parse": {
        "metric": "outside-mask parsing consistency (Dice)",
        "direction": "higher_is_better", "comparator": ">=",
        "version_b_rule": "tau_parse_v3 = p01(outside-support parsing Dice)",
        "version_b_value": 0.7094826178704915,
        "superseded_version_b_value": 0.8747814437904173,
        "superseded_note": "recorded in v3 as `tau_parse_v1_superseded`",
        "nominal_source": INHERITED_VERSION_B,
        "semantic_compatibility": "YES",
        "compatibility_reason": (
            "the same FaceXFormer weight (sha256 327a7558…), the same "
            "outside-the-exact-mask region semantics and the same Dice "
            "definition in byte-identical code"),
    },
    "tau_out": {
        "metric": "outside-mask maximum uint8 error",
        "direction": "exact_equality", "comparator": "==",
        "version_b_rule": "structural invariant, exactly 0 in v1, v2 and v3",
        "version_b_value": 0.0,
        "nominal_source": FROZEN_RANGE_CONSTRAINT,
        "semantic_compatibility": "YES",
        "compatibility_reason": (
            "not a fitted threshold at all: `finalize_discrete` guarantees the "
            "region outside the exact mask is byte-identical to the original, so "
            "the gate asserts an invariant. `gate_profiles.RANGE_SAFE` already "
            "excludes it from every profile transform"),
    },
    "tau_fp": {
        "metric": "generator-fingerprint separability",
        "direction": "lower_is_better", "comparator": "<=",
        "version_b_rule": ("fingerprint reference percentile, identical in v1, v2 "
                           "and v3"),
        "version_b_value": 5.687657785453908,
        "nominal_source": INHERITED_VERSION_B,
        "semantic_compatibility": "YES",
        "compatibility_reason": (
            "byte-identical `fingerprint.py` over the same M3B source package, so "
            "the reference distribution and the score are the same construction"),
    },
}


class ThresholdInheritanceError(RuntimeError):
    """The inherited NOMINAL set cannot be established."""


def verify_version_b_artifact(version_b_root: Path) -> dict[str, Any]:
    """Re-verify the vendored values against the live Version-B artifact.

    Read-only, and optional: the GPU host need not have Version B mounted. When
    it is present this proves the transcription rather than trusting it.
    """
    path = Path(version_b_root) / VERSION_B_ARTIFACT
    if not path.is_file():
        return {"available": False, "artifact": VERSION_B_ARTIFACT,
                "reason": "the Version-B tree is not mounted here"}
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    payload = json.loads(raw.decode("utf-8"))
    recorded = {key: float(value) for key, value in payload["thresholds"].items()}
    mismatched = {key: {"vendored": INHERITED_NOMINAL[key], "artifact": recorded.get(key)}
                  for key in INHERITED_NOMINAL
                  if recorded.get(key) != INHERITED_NOMINAL[key]}
    return {"available": True, "artifact": VERSION_B_ARTIFACT,
            "artifact_sha256": digest,
            "artifact_sha256_matches": digest == VERSION_B_ARTIFACT_SHA256,
            "threshold_sha256": payload.get("threshold_sha256"),
            "threshold_sha256_matches":
                payload.get("threshold_sha256") == VERSION_B_THRESHOLD_SHA256,
            "values_match": not mismatched, "mismatched": mismatched,
            "version_b_commit": VERSION_B_COMMIT}


def assemble_nominal(derived: Mapping[str, float] | None = None
                     ) -> tuple[dict[str, float], dict[str, Any]]:
    """The final NOMINAL map, metric by metric, with per-threshold provenance.

    An inherited threshold is never overwritten by a fitted one. `derived` may
    supply a metric that has no compatible inherited threshold; today every
    metric has one, so a derived value for an inherited metric is dropped and
    recorded as ignored rather than silently winning.
    """
    derived = dict(derived or {})
    nominal: dict[str, float] = {}
    provenance: dict[str, Any] = {}
    ignored: dict[str, float] = {}

    for name, entry in PROVENANCE.items():
        source = entry["nominal_source"]
        if source in (INHERITED_VERSION_B, FROZEN_RANGE_CONSTRAINT):
            nominal[name] = float(INHERITED_NOMINAL[name])
            if name in derived and float(derived[name]) != nominal[name]:
                ignored[name] = float(derived[name])
        elif source == SOURCE_REFERENCE_DERIVED:
            if name not in derived:
                raise ThresholdInheritanceError(
                    f"{name} has no compatible inherited threshold and no "
                    "source-reference value was derived for it")
            nominal[name] = float(derived[name])
        else:                                            # pragma: no cover
            raise ThresholdInheritanceError(f"{name}: unknown source {source!r}")
        provenance[name] = {**entry, "nominal": nominal[name]}

    extra = sorted(set(derived) - set(PROVENANCE))
    if extra:
        raise ThresholdInheritanceError(
            f"the calibrator produced thresholds with no inheritance ruling: {extra}")

    return nominal, {
        "schema_version": SCHEMA_VERSION,
        "version_b_commit": VERSION_B_COMMIT, "version_b_tag": VERSION_B_TAG,
        "version_b_artifact": VERSION_B_ARTIFACT,
        "version_b_artifact_sha256": VERSION_B_ARTIFACT_SHA256,
        "version_b_threshold_sha256": VERSION_B_THRESHOLD_SHA256,
        "version_b_package_identity": VERSION_B_PACKAGE_IDENTITY,
        "per_threshold": provenance,
        "inherited": sorted(name for name, item in PROVENANCE.items()
                            if item["nominal_source"] == INHERITED_VERSION_B),
        "frozen_range_constraints": sorted(
            name for name, item in PROVENANCE.items()
            if item["nominal_source"] == FROZEN_RANGE_CONSTRAINT),
        "source_reference_derived": sorted(
            name for name, item in PROVENANCE.items()
            if item["nominal_source"] == SOURCE_REFERENCE_DERIVED),
        "calibrator_values_ignored_because_inherited": ignored,
        "nominal_identity_sha256": nominal_identity(nominal),
        "rule": ("§11.4: NOMINAL uses the unique inherited Version-B threshold "
                 "when semantically compatible; only a metric without one is "
                 "derived from the frozen source reference distribution"),
    }


def nominal_identity(nominal: Mapping[str, float]) -> str:
    """One identity for the assembled NOMINAL set, shared by all three arms."""
    payload = {key: float(value) for key, value in nominal.items()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


__all__ = ["SCHEMA_VERSION", "VERSION_B_COMMIT", "VERSION_B_TAG",
           "VERSION_B_ARTIFACT", "VERSION_B_ARTIFACT_SHA256",
           "VERSION_B_THRESHOLD_SHA256", "VERSION_B_PACKAGE_IDENTITY",
           "INHERITED_VERSION_B", "SOURCE_REFERENCE_DERIVED",
           "FROZEN_RANGE_CONSTRAINT", "INHERITED_NOMINAL", "PROVENANCE",
           "ThresholdInheritanceError", "verify_version_b_artifact",
           "assemble_nominal", "nominal_identity"]
