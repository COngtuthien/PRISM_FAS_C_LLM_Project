"""Bounded, arithmetic-only interpretation of an ALREADY-OBSERVED,
ALREADY-VALIDATED `C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2` result.

This module never recomputes a diagnostic metric, never loads a checkpoint,
never forwards an image, and never opens a target path. Every number it
touches is read from the already-written, already-validated
`DIAGNOSTICS_PER_TEST.json` — the only new arithmetic performed here is:
which acceptance criterion (mean/tail) an arm satisfied or missed, its exact
margin below (or exceedance above) the corresponding `tau`, and a
descending ordering of tail exceedances among arms that failed the tail
criterion. Every emitted statement is a plain-language label, never a new
scientific claim beyond what the recorded numbers and the frozen protocol's
own pass/fail arithmetic already establish.

A BLOCKED test (`NEEDS_SCIENTIFIC_DECISION` / `STRUCTURALLY_MODEL_BLOCKED` /
`STRUCTURALLY_DATA_BLOCKED`) is carried through unchanged and is NEVER
converted into a PASS or FAIL — this module has no code path that assigns a
verdict to a test the runner itself recorded as BLOCKED.
"""
from __future__ import annotations

from typing import Any, Mapping

ARM_ORDER: tuple[str, ...] = ("RND", "DET", "LLM")

#: Human-readable labels for the frozen perturbation each executable test
#: applies — used only for prose, never to select or alter a threshold.
TEST_LABELS: dict[str, str] = {
    "benign_color_corruption": "color-gain",
    "benign_jpeg_corruption": "JPEG re-encode",
    "benign_resize_corruption": "resize (downscale/upscale)",
}

#: Fixed, protocol-level boundary statements — apply regardless of which
#: numbers were observed. Included once per interpretation, not derived
#: from any arithmetic.
GLOBAL_NOT_SUPPORTED: tuple[str, ...] = (
    "a causal claim that any diagnostic result CAUSED the BA_sep FAILURE — "
    "causality has not been established by this protocol",
    "treating a BLOCKED test (NEEDS_SCIENTIFIC_DECISION / "
    "STRUCTURALLY_MODEL_BLOCKED / STRUCTURALLY_DATA_BLOCKED) as an observed "
    "FAIL or as negative evidence",
    "a claim generalizing any PASS or FAIL beyond this protocol's exact "
    "frozen perturbation, threshold and population",
)
GLOBAL_INTERPRETATION: tuple[str, ...] = (
    "The resize/JPEG diagnostic results are consistent with the possibility "
    "that some detectors use preprocessing/resampling/compression-sensitive "
    "cues, which is mechanistically compatible with shortcut sensitivity "
    "already implicated by the BA_sep FAILURE. This is a consistency "
    "statement, not a causal one.",
)


def _criterion(mean_delta_plus: float, tau_mean: float,
              p95_delta_plus: float, tau_tail: float) -> tuple[bool, bool, str | None]:
    mean_ok = float(mean_delta_plus) <= float(tau_mean)
    tail_ok = float(p95_delta_plus) <= float(tau_tail)
    if mean_ok and tail_ok:
        failed = None
    elif not mean_ok and not tail_ok:
        failed = "both"
    elif not tail_ok:
        failed = "tail"
    else:
        failed = "mean"
    return mean_ok, tail_ok, failed


def derive_test_interpretation(test_id: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    """The bounded interpretation of one recorded `per_test` entry. Never
    reads anything except the entry itself — no filesystem, no checkpoint,
    no image."""
    if entry.get("classification") != "EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL":
        return {
            "test_id": test_id,
            "classification": entry.get("classification"),
            "observed": {"status": entry.get("status"),
                        "blocked_reason": entry.get("blocked_reason", "")},
            "interpretation": [
                "BLOCKED is not evidence of PASS or FAIL; the scientific question "
                "remains open pending a normative decision or structural resolution."],
            "not_supported": ["treating this BLOCKED test as a FAIL or as negative evidence"],
        }

    per_arm_in = dict(entry.get("per_arm") or {})
    per_arm_out: dict[str, Any] = {}
    tail_failures: list[tuple[str, float]] = []
    mean_failures: list[str] = []
    for arm in ARM_ORDER:
        arm_entry = per_arm_in.get(arm) or {}
        evaluation = dict(arm_entry.get("evaluation") or {})
        threshold = dict(arm_entry.get("reference_threshold") or {})
        mean_delta_plus = float(evaluation["mean_delta_plus"])
        p95_delta_plus = float(evaluation["p95_delta_plus"])
        tau_mean = float(threshold["tau_mean"])
        tau_tail = float(threshold["tau_tail"])
        mean_ok, tail_ok, failed = _criterion(mean_delta_plus, tau_mean, p95_delta_plus, tau_tail)

        derived: dict[str, Any] = {}
        if mean_ok:
            derived["mean_margin"] = tau_mean - mean_delta_plus
        else:
            derived["mean_exceedance"] = mean_delta_plus - tau_mean
            mean_failures.append(arm)
        if tail_ok:
            derived["tail_margin"] = tau_tail - p95_delta_plus
        else:
            derived["tail_exceedance"] = p95_delta_plus - tau_tail
            tail_failures.append((arm, p95_delta_plus - tau_tail))

        per_arm_out[arm] = {
            "observed": {"mean_delta_plus": mean_delta_plus, "p95_delta_plus": p95_delta_plus,
                        "tau_mean": tau_mean, "tau_tail": tau_tail,
                        "verdict": arm_entry.get("verdict")},
            "derived_arithmetic": derived,
            "criterion_failed": failed,
        }

    interpretation: list[str] = []
    not_supported: list[str] = []
    label = TEST_LABELS.get(test_id, test_id)

    if entry.get("status") == "PASS":
        interpretation.append(
            f"No evidence, under this protocol, of excessive sensitivity to the frozen "
            f"benign {label} perturbation for any arm ({', '.join(ARM_ORDER)}).")
        not_supported.append(
            f"generalizing this PASS to all {label}-adjacent conditions outside this "
            "protocol's exact frozen perturbation and population")
    else:
        failing_arms = [arm for arm in ARM_ORDER if per_arm_out[arm]["criterion_failed"]]
        passing_arms = [arm for arm in ARM_ORDER if not per_arm_out[arm]["criterion_failed"]]
        if passing_arms:
            interpretation.append(
                f"{', '.join(passing_arms)} PASS under the frozen {label} diagnostic; "
                f"{', '.join(failing_arms)} FAIL.")
        if failing_arms and not mean_failures:
            if len(failing_arms) == len(ARM_ORDER):
                interpretation.append(
                    "All arms' mean criterion PASSES; the failure is not a broad average "
                    "score inflation. It is primarily a tail-sensitivity / "
                    "subset-of-samples effect concentrated in the upper-tail (p95) of the "
                    "delta_plus distribution.")
            else:
                interpretation.append(
                    f"{', '.join(failing_arms)} fail ONLY the upper-tail (p95) criterion, "
                    "not the mean criterion — not a broad average score inflation, but a "
                    "high-sensitivity tail effect in a subset of source_dev LIVE samples.")
        if len(tail_failures) >= 2:
            ranked = sorted(tail_failures, key=lambda item: item[1], reverse=True)
            ordering = " > ".join(f"{arm} ({exceedance:.7f})" for arm, exceedance in ranked)
            interpretation.append(
                f"Exact frozen-threshold tail exceedances, most to least: {ordering}. "
                "This is a descriptive comparison only.")
            not_supported.append(
                "a statistical ranking among arms' tail exceedance — no preregistered "
                "paired statistical comparison exists for this quantity")

    return {
        "test_id": test_id, "classification": entry.get("classification"),
        "test_verdict": entry.get("status"),
        "per_arm": per_arm_out,
        "interpretation": interpretation,
        "not_supported": not_supported,
    }


def derive_full_interpretation(per_test: Mapping[str, Any]) -> dict[str, Any]:
    """The complete bounded interpretation of a recorded `per_test` mapping —
    pure function, zero filesystem/model/image access."""
    tests = {test_id: derive_test_interpretation(test_id, entry)
            for test_id, entry in per_test.items()}
    return {"tests": tests, "global_interpretation": list(GLOBAL_INTERPRETATION),
           "global_not_supported": list(GLOBAL_NOT_SUPPORTED)}


__all__ = ["ARM_ORDER", "TEST_LABELS", "GLOBAL_NOT_SUPPORTED", "GLOBAL_INTERPRETATION",
           "derive_test_interpretation", "derive_full_interpretation"]
