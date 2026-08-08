"""The M10 reliability / shortcut / causal test framework (spec section 17.3).

Nine of the ten declared tests run on SOURCE data only — they ask whether the
detector is reading the physics it claims to read, and none of them needs a target
label. The tenth has no legitimate population in this project and is recorded
`BLOCKED` with its reason rather than substituted with an invalid one.

Each test declares, in advance:

    what population it runs on, what it measures, what a PASS looks like, and what
    an observed failure would mean.

A test whose data does not exist is `BLOCKED`. A test that runs and fails is a
FAILED test and a reported negative result — never a quietly dropped one.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from .contracts import M10_RELIABILITY_SCHEMA_VERSION, M10ContractError, stable_identity

TEST_STATUSES = ("PLANNED", "PASSED", "FAILED", "BLOCKED")


@dataclass(frozen=True)
class ReliabilityTest:
    """One declared test, with its acceptance rule fixed before it runs."""
    test_id: str
    spec_reference: str
    population: str
    measures: str
    pass_rule: str
    failure_meaning: str
    status: str = "PLANNED"
    blocked_reason: str | None = None
    result: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "ReliabilityTest":
        if self.status not in TEST_STATUSES:
            raise M10ContractError(f"{self.test_id}: unknown status {self.status!r}")
        if self.status == "BLOCKED" and not self.blocked_reason:
            raise M10ContractError(f"{self.test_id}: a BLOCKED test must carry its reason")
        if self.status != "BLOCKED" and self.blocked_reason:
            raise M10ContractError(f"{self.test_id}: only a BLOCKED test may carry a blocked reason")
        if "target" in self.population.lower() and "no target" not in self.population.lower():
            raise M10ContractError(f"{self.test_id}: a reliability test may not run on target labels")
        return self

    def payload(self) -> dict[str, Any]:
        return {"test_id": self.test_id, "spec_reference": self.spec_reference,
                "population": self.population, "measures": self.measures,
                "pass_rule": self.pass_rule, "failure_meaning": self.failure_meaning,
                "status": self.status, "blocked_reason": self.blocked_reason,
                "result": dict(self.result)}


DECLARED_TESTS: tuple[ReliabilityTest, ...] = (
    ReliabilityTest(
        test_id="synthetic_vs_real_spoof_probe", spec_reference="17.3.1",
        population="source_train real spoof vs accepted M8 v3 synthetic spoof; no target",
        measures="balanced accuracy of a probe trained to separate synthetic from real spoof",
        pass_rule="probe balanced accuracy stays below the declared ceiling",
        failure_meaning="a generator fingerprint the detector could shortcut on"),
    ReliabilityTest(
        test_id="benign_jpeg_corruption", spec_reference="17.3.2",
        population="source_dev LIVE only, benign JPEG re-encode; no spoof, no target",
        measures="shift of the spoof score on unchanged bona-fide content",
        pass_rule="no systematic large increase in the spoof score",
        failure_meaning="the detector reads compression artefacts as attack evidence"),
    ReliabilityTest(
        test_id="benign_resize_corruption", spec_reference="17.3.2",
        population="source_dev LIVE only, benign resize; no spoof, no target",
        measures="shift of the spoof score on unchanged bona-fide content",
        pass_rule="no systematic large increase in the spoof score",
        failure_meaning="the detector reads resampling as attack evidence"),
    ReliabilityTest(
        test_id="benign_color_corruption", spec_reference="17.3.2",
        population="source_dev LIVE only, benign colour shift; no spoof, no target",
        measures="shift of the spoof score on unchanged bona-fide content",
        pass_rule="no systematic large increase in the spoof score",
        failure_meaning="the detector reads colour statistics as attack evidence"),
    ReliabilityTest(
        test_id="residual_scale_zero", spec_reference="17.3.3",
        population="source_train live with the GPAT residual scaled to 0; no target",
        measures="spoof score and regional distance at zero residual",
        pass_rule="score approaches live and the regional distance falls",
        failure_meaning="the score does not depend on the artefact it claims to detect"),
    ReliabilityTest(
        test_id="recipe_region_shift", spec_reference="17.3.4",
        population="source_train live under two recipes differing only in region; no target",
        measures="displacement of the anomaly heat-map peak",
        pass_rule="the heat-map moves with the attacked region",
        failure_meaning="the regional evidence is not localized where the attack is"),
    ReliabilityTest(
        test_id="artifact_map_swap", spec_reference="17.3.5",
        population="accepted M8 v3 synthetic samples with artefact maps swapped; no target",
        measures="local supervision performance under a mismatched artefact map",
        pass_rule="local supervision performance drops",
        failure_meaning="the local head ignores the supervision it is given"),
    ReliabilityTest(
        test_id="crop_padding_interpolation", spec_reference="17.3.6",
        population="source_dev under a different crop padding/interpolation; no target",
        measures="score sensitivity to a preprocessing-only change",
        pass_rule="no large systematic shift",
        failure_meaning="a preprocessing shortcut rather than a physical cue"),
    ReliabilityTest(
        test_id="cross_route_synthetic", spec_reference="17.3.7",
        population="train on one synthetic route, evaluate on the other; source only",
        measures="cross-route generalization of the synthetic evidence",
        pass_rule="performance is retained across routes",
        failure_meaning="the detector learned one generator, not the artefact class"),
    ReliabilityTest(
        test_id="benign_glasses_makeup_lowlight", spec_reference="17.3.8",
        population="no legitimate benign population exists in this project",
        measures="BPCER under benign glasses / makeup / low light",
        pass_rule="BPCER does not rise materially on benign attributes",
        failure_meaning="bona-fide users with glasses or makeup are falsely rejected",
        status="BLOCKED",
        blocked_reason=(
            "no legitimate benign glasses/makeup/low-light population exists here. SiW-Mv2's "
            "Makeup_* and Partial_*Glasses videos are labelled SPOOF presentations, so treating "
            "them as benign live stress cases would invert their label; CASIA-FASD and MSU-MFSD "
            "carry no benign attribute metadata. Recorded as blocked rather than substituted "
            "with an invalid population.")),
)


def declared_tests() -> list[ReliabilityTest]:
    return [test.validate() for test in DECLARED_TESTS]


def score_shift(before: Sequence[float], after: Sequence[float]) -> dict[str, Any]:
    """The measurement every benign-corruption test shares.

    Reports the paired mean shift and the fraction that moved toward `spoof`. It
    applies no threshold of its own; the acceptance rule lives in the test.
    """
    left = np.asarray(before, dtype=np.float64).reshape(-1)
    right = np.asarray(after, dtype=np.float64).reshape(-1)
    if left.shape != right.shape: raise M10ContractError("paired corruption vectors must align")
    if left.size == 0: raise M10ContractError("cannot measure a shift over zero samples")
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        raise M10ContractError("non-finite corruption scores")
    delta = right - left
    return {"samples": int(left.size), "mean_before": float(left.mean()),
            "mean_after": float(right.mean()), "mean_shift": float(delta.mean()),
            "median_shift": float(np.median(delta)), "max_shift": float(delta.max()),
            "fraction_increased": float((delta > 0).mean()),
            "p95_shift": float(np.percentile(delta, 95))}


def evaluate(test: ReliabilityTest, *, result: dict[str, Any], passed: bool) -> ReliabilityTest:
    """Record an outcome. A failure stays a FAILED test and a reported result."""
    if test.status == "BLOCKED":
        raise M10ContractError(f"{test.test_id} is BLOCKED and cannot be given a result")
    from dataclasses import replace
    return replace(test, status="PASSED" if passed else "FAILED", result=dict(result)).validate()


def build_report(tests: Sequence[ReliabilityTest]) -> dict[str, Any]:
    resolved = [test.validate() for test in tests]
    counts: dict[str, int] = {}
    for test in resolved: counts[test.status] = counts.get(test.status, 0) + 1
    body = {"reliability_schema_version": M10_RELIABILITY_SCHEMA_VERSION,
            "tests": [test.payload() for test in resolved], "count": len(resolved),
            "by_status": dict(sorted(counts.items())),
            "blocked": [{"test_id": test.test_id, "reason": test.blocked_reason}
                        for test in resolved if test.status == "BLOCKED"],
            "failed": [test.test_id for test in resolved if test.status == "FAILED"],
            "uses_target_labels": False}
    return {**body, "reliability_identity": stable_identity(body)}


def write_report(path: Path, payload: dict[str, Any], *, dry_run: bool = False) -> Path:
    from prism_fas.utils.core import atomic_json_write
    if not dry_run: atomic_json_write(Path(path), payload)
    return Path(path)
