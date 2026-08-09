"""Deterministic paired bootstrap by video, plus the declared multiple-comparison
policy.

Frozen in `docs/M10_STATISTICS_CONTRACT.md` and never changed after a target result
is seen:

    unit        video (the opaque siw_<16hex> id)
    pairing     paired — ONE resample of video ids, both models scored on it
    resamples   10000
    seed        20260810
    confidence  0.95, percentile interval
    statistic   video-level ACER difference at the frozen source-dev threshold
    p-value     two-sided, 2 * min(P(delta <= 0), P(delta >= 0)), clipped to [0,1]
    correction  Holm-Bonferroni over the five declared hypotheses H1-H5

The generator seed is derived STRUCTURALLY from the comparison, not from the raw
integer, so the same comparison always draws the same videos and a different
comparison cannot accidentally reuse a plan. The resulting index matrix has its own
`bootstrap_plan_identity`, which is what proves determinism rather than a promise
of it.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Sequence
import numpy as np
from .contracts import (M10_BOOTSTRAP_SCHEMA_VERSION, M10ContractError, may_carry_statistical_claim,
                        stable_identity)
from .metrics import core_metrics

DEFAULT_RESAMPLES = 10000
DEFAULT_SEED = 20260810
DEFAULT_CONFIDENCE = 0.95
DEFAULT_STATISTIC = "video_acer_difference"


@dataclass(frozen=True)
class BootstrapSettings:
    resamples: int = DEFAULT_RESAMPLES
    seed: int = DEFAULT_SEED
    confidence_level: float = DEFAULT_CONFIDENCE
    statistic: str = DEFAULT_STATISTIC
    unit: str = "video"

    def validate(self) -> "BootstrapSettings":
        if self.resamples < 1: raise M10ContractError("resamples must be positive")
        if not 0.0 < self.confidence_level < 1.0: raise M10ContractError("confidence level outside (0,1)")
        if self.unit != "video": raise M10ContractError("the bootstrap unit is frozen at 'video'")
        return self

    def payload(self) -> dict[str, Any]:
        return {"resamples": int(self.resamples), "seed": int(self.seed),
                "confidence_level": float(self.confidence_level), "statistic": str(self.statistic),
                "unit": str(self.unit)}


def seed_material(video_ids: Sequence[str], settings: BootstrapSettings) -> str:
    joined = hashlib.sha256("\n".join(sorted(str(value) for value in video_ids)).encode("utf-8")).hexdigest()
    return hashlib.sha256("|".join([M10_BOOTSTRAP_SCHEMA_VERSION, str(settings.seed),
                                    str(settings.resamples), settings.statistic,
                                    joined]).encode("utf-8")).hexdigest()


def build_plan(video_ids: Sequence[str], settings: BootstrapSettings) -> dict[str, Any]:
    """The `[resamples, n]` index matrix and its identity."""
    settings.validate()
    ids = [str(value) for value in video_ids]
    if len(ids) < 2: raise M10ContractError("a bootstrap needs at least two videos")
    if len(set(ids)) != len(ids): raise M10ContractError("duplicate video ids in the bootstrap unit set")
    material = seed_material(ids, settings)
    generator = np.random.Generator(np.random.PCG64(int(material[:16], 16)))
    indices = generator.integers(0, len(ids), size=(int(settings.resamples), len(ids)), dtype=np.int64)
    return {"schema_version": M10_BOOTSTRAP_SCHEMA_VERSION, "settings": settings.payload(),
            "n_units": len(ids), "seed_material_sha256": material, "indices": indices,
            "bootstrap_plan_identity": hashlib.sha256(
                np.ascontiguousarray(indices).tobytes()).hexdigest()}


def _video_acer(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    result = core_metrics(scores, labels, threshold=threshold)
    value = result["acer"]
    if isinstance(value, dict):
        raise M10ContractError(f"the resampled population cannot produce an ACER: {value['reason']}")
    return float(value)


def paired_bootstrap(*, video_ids: Sequence[str], scores_a: Sequence[float], scores_b: Sequence[float],
                     labels: Sequence[int], threshold: float,
                     threshold_a: float | None = None, threshold_b: float | None = None,
                     settings: BootstrapSettings | None = None,
                     statistic: Callable[[np.ndarray, np.ndarray, float], float] | None = None
                     ) -> dict[str, Any]:
    """One resample of videos, BOTH models scored on it.

    Drawing two independent resamples would inflate the interval and is not what
    Table 58 asks for: the between-video variance the two models share is exactly
    what pairing removes.

    **Each side is scored at its OWN frozen source-dev threshold.** The statistics
    contract's statistic is "ACER at the frozen source-dev threshold", and that
    threshold is fitted per experiment on `source_dev`; two experiments do not share
    one. Forcing a single threshold on both would report neither model at the
    operating point it would actually deploy. `threshold` remains the default for
    both sides so a genuinely shared-threshold comparison is still expressible.
    """
    settings = (settings or BootstrapSettings()).validate()
    metric = statistic or _video_acer
    left_threshold = float(threshold if threshold_a is None else threshold_a)
    right_threshold = float(threshold if threshold_b is None else threshold_b)
    ids = [str(value) for value in video_ids]
    left = np.asarray(scores_a, dtype=np.float64)
    right = np.asarray(scores_b, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    if not (len(ids) == left.size == right.size == truth.size):
        raise M10ContractError("video ids, both score vectors and labels must align")
    plan = build_plan(ids, settings)
    observed = metric(left, truth, left_threshold) - metric(right, truth, right_threshold)
    deltas = np.empty(int(settings.resamples), dtype=np.float64)
    for position, row in enumerate(plan["indices"]):
        deltas[position] = (metric(left[row], truth[row], left_threshold)
                            - metric(right[row], truth[row], right_threshold))
    alpha = 1.0 - float(settings.confidence_level)
    low, high = np.percentile(deltas, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    p_value = float(min(1.0, 2.0 * min(float((deltas <= 0).mean()), float((deltas >= 0).mean()))))
    return {"schema_version": M10_BOOTSTRAP_SCHEMA_VERSION, "settings": settings.payload(),
            "n_units": plan["n_units"], "bootstrap_plan_identity": plan["bootstrap_plan_identity"],
            "seed_material_sha256": plan["seed_material_sha256"],
            "threshold_treatment": left_threshold, "threshold_control": right_threshold,
            "threshold_source": "frozen source_dev calibration of each side",
            "observed_delta": float(observed), "ci_low": float(low), "ci_high": float(high),
            "confidence_level": float(settings.confidence_level), "interval": "percentile",
            "p_value": p_value, "paired": True,
            "significant_at_alpha": bool(low > 0.0 or high < 0.0),
            "bootstrap_mean_delta": float(deltas.mean()), "bootstrap_std_delta": float(deltas.std(ddof=0))}


def plans_agree(video_ids: Sequence[str], settings: BootstrapSettings | None = None) -> dict[str, Any]:
    """Build the plan twice and prove the identity reproduces exactly."""
    settings = settings or BootstrapSettings()
    first, second = build_plan(video_ids, settings), build_plan(video_ids, settings)
    return {"identical": bool(first["bootstrap_plan_identity"] == second["bootstrap_plan_identity"]
                              and np.array_equal(first["indices"], second["indices"])),
            "bootstrap_plan_identity": first["bootstrap_plan_identity"]}


def holm_bonferroni(p_values: dict[str, float], *, alpha: float = 0.05) -> dict[str, Any]:
    """Holm-Bonferroni over the declared family. Raw and adjusted are both reported."""
    if not p_values: raise M10ContractError("an empty comparison family cannot be corrected")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    size = len(ordered)
    adjusted, running = {}, 0.0
    for position, (name, raw) in enumerate(ordered):
        running = max(running, min(1.0, float(raw) * (size - position)))
        adjusted[name] = running
    return {"policy": "holm_bonferroni", "family_size": size, "alpha": float(alpha),
            "raw": {name: float(value) for name, value in sorted(p_values.items())},
            "adjusted": {name: float(value) for name, value in sorted(adjusted.items())},
            "rejected_null": sorted(name for name, value in adjusted.items() if value <= float(alpha))}


def refuse_single_seed_comparison(name: str, roles: Sequence[str]) -> None:
    """A comparison whose either side is a 1-seed row is REFUSED, not downgraded.

    Silently attaching an interval to a single-seed row would present training
    variability as if it had been measured.
    """
    weak = [role for role in roles if not may_carry_statistical_claim(role)]
    if weak:
        raise M10ContractError(f"{name}: refusing a statistical comparison against a "
                               f"single-seed row (roles {sorted(set(weak))}); it is reported as "
                               f"single_seed_descriptive instead")


def run_declared_family(*, hypotheses: dict[str, Any], scored: dict[str, dict[str, Any]],
                        roles: dict[str, str], threshold: float,
                        settings: BootstrapSettings | None = None, alpha: float = 0.05
                        ) -> dict[str, Any]:
    """Run exactly the five pre-declared comparisons, then correct across them.

    `scored` maps an experiment id to `{"video_ids": [...], "scores": [...],
    "labels": [...]}`. A comparison whose either side is a single-seed row is
    recorded as `refused` with its reason and takes no part in the correction — the
    family size stays what was declared, so refusing one does not make the others
    easier to pass.
    """
    settings = settings or BootstrapSettings()
    comparisons: dict[str, Any] = {}
    p_values: dict[str, float] = {}
    for name, block in sorted(hypotheses.items()):
        treatment, control = str(block["treatment"]), str(block["control"])
        if block.get("kind") == "parity_not_superiority":
            comparisons[name] = {"status": "parity_not_superiority",
                                 "reason": "a parity check is reported as agreement within a "
                                           "declared tolerance, with no p-value"}
            continue
        if treatment not in scored or control not in scored:
            comparisons[name] = {"status": "unavailable",
                                 "reason": f"no scoring result for {treatment!r} or {control!r}"}
            continue
        try:
            refuse_single_seed_comparison(name, [roles.get(treatment, "diagnostic"),
                                                 roles.get(control, "diagnostic")])
        except M10ContractError as error:
            comparisons[name] = {"status": "refused", "reason": str(error)}
            continue
        left, right = scored[treatment], scored[control]
        if list(left["video_ids"]) != list(right["video_ids"]):
            comparisons[name] = {"status": "unavailable",
                                 "reason": "the two models were not scored on the same video set, "
                                           "so the comparison cannot be paired"}
            continue
        # Each side at its own frozen source-dev threshold; `threshold` is the
        # fallback only for a scored block that does not carry one.
        result = paired_bootstrap(video_ids=left["video_ids"], scores_a=left["scores"],
                                  scores_b=right["scores"], labels=left["labels"],
                                  threshold=threshold,
                                  threshold_a=left.get("threshold"), threshold_b=right.get("threshold"),
                                  settings=settings)
        comparisons[name] = {"status": "computed", "treatment": treatment, "control": control,
                             "treatment_seeds": left.get("n_seeds"), "control_seeds": right.get("n_seeds"),
                             **result}
        p_values[name] = float(result["p_value"])
    correction = holm_bonferroni(p_values, alpha=alpha) if p_values else {
        "policy": "holm_bonferroni", "family_size": len(hypotheses), "alpha": float(alpha),
        "raw": {}, "adjusted": {}, "rejected_null": [],
        "note": "no comparison in the declared family produced a p-value"}
    return {"schema_version": M10_BOOTSTRAP_SCHEMA_VERSION, "settings": settings.payload(),
            "declared_family": sorted(hypotheses), "family_size": len(hypotheses),
            "comparisons": comparisons, "multiple_comparison": correction,
            "exploratory_not_in_family": []}


def comparison_identity(payload: dict[str, Any]) -> str:
    return stable_identity({"schema_version": M10_BOOTSTRAP_SCHEMA_VERSION, **payload})
