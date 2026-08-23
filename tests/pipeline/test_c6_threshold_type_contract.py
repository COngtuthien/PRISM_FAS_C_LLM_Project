"""One representation at the gating boundary, and the production path uses it.

`GateProfile.thresholds` is a `dict[str, float]` — right for hashing, identity
and serialization. `quality_gate.evaluate` reads `thresholds.tau_fd`, so it needs
a `quality_gate.Thresholds`. `GateProfile.as_thresholds()` is the conversion the
class already owns, and the engineering rehearsal already called it.

The scientific adapter did not. It passed the raw dict into `gate_candidates`,
and the GPU run measured every candidate and then died at the first gating call
with `'dict' object has no attribute 'tau_fd'`.

Every existing test missed it for one reason: they built the `Thresholds`
themselves — `Thresholds.from_dict(profiles[name].thresholds)` — doing the
conversion the production code omitted. So this file drives the real
`_check_profile_matched_feasibility` instead, and asserts a real `Thresholds`
reaches `evaluate`.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.pipeline.adapters import AdapterRequest  # noqa: E402
from prism_fas.pipeline.adapters.c6 import C6Adapter  # noqa: E402
from prism_fas.pipeline.profiles import load_profile  # noqa: E402
from prism_fas.synthesis import c6_scientific as science  # noqa: E402
from prism_fas.synthesis import quality_gate  # noqa: E402
from prism_fas.synthesis.c5_source_pair_plan import ARMS, GPAT, PHYSICS  # noqa: E402
from prism_fas.synthesis.c6_threshold_inheritance import INHERITED_NOMINAL  # noqa: E402
from prism_fas.synthesis.gate_profiles import (PROFILE_ORDER,  # noqa: E402
                                               build_profiles)

TAUS = ("tau_fd", "tau_id", "tau_lm", "tau_parse", "tau_out", "tau_fp")


def _raw(**overrides: Any) -> dict[str, Any]:
    base = {"face_detection_score": 0.93, "identity_cosine": 0.88,
            "landmark_nme": 0.004, "outside_mask_parsing_dice": 0.97,
            "outside_mask_max_error": 0.0, "measured_artifact_strength": 0.40,
            "requested_artifact_strength": 0.40, "fingerprint_score": 1.2,
            "support_overlap": 0.99, "reference_detection_score": 0.95,
            "landmark_detected": True}
    base.update(overrides)
    return base


# --- 1, 2, 3. the two representations, and their equality --------------------

def test_gate_profile_keeps_the_exact_frozen_dict_values() -> None:
    profiles = build_profiles(INHERITED_NOMINAL, nominal_source="x")

    assert dict(profiles["NOMINAL"].thresholds) == INHERITED_NOMINAL
    assert isinstance(profiles["NOMINAL"].thresholds, dict)


def test_as_thresholds_returns_the_canonical_object() -> None:
    profile = build_profiles(INHERITED_NOMINAL, nominal_source="x")["NOMINAL"]

    assert isinstance(profile.as_thresholds(), quality_gate.Thresholds)


@pytest.mark.parametrize("name", list(PROFILE_ORDER))
@pytest.mark.parametrize("tau", list(TAUS))
def test_the_two_representations_agree_value_for_value(name, tau) -> None:
    profile = build_profiles(INHERITED_NOMINAL, nominal_source="x")[name]

    assert getattr(profile.as_thresholds(), tau) == profile.thresholds[tau]


def test_the_conversion_changes_no_threshold_value() -> None:
    profiles = build_profiles(INHERITED_NOMINAL, nominal_source="x")

    for name in PROFILE_ORDER:
        assert profiles[name].as_thresholds().as_dict() == profiles[name].thresholds


# --- the typed boundary -------------------------------------------------------

def test_the_gating_boundary_refuses_the_raw_mapping() -> None:
    """The negative regression: a dict is rejected, never reinterpreted."""
    profile = build_profiles(INHERITED_NOMINAL, nominal_source="x")["NOMINAL"]

    with pytest.raises(science.ScientificGateError, match="requires a quality_gate"):
        science.gate_candidates({"c0": _raw()}, profile.thresholds)


def test_the_refusal_names_the_conversion_to_use() -> None:
    profile = build_profiles(INHERITED_NOMINAL, nominal_source="x")["NOMINAL"]

    with pytest.raises(science.ScientificGateError) as raised:
        science.gate_candidates({}, profile.thresholds)

    assert "as_thresholds()" in str(raised.value)
    assert "hashing and serialization only" in str(raised.value)


def test_there_is_no_dict_accepting_compatibility_path() -> None:
    source = inspect.getsource(science.gate_candidates)

    assert "isinstance(thresholds, Thresholds)" in source
    for forbidden in ("from_dict(thresholds)", "if isinstance(thresholds, dict)",
                      "Thresholds(**thresholds)"):
        assert forbidden not in source, forbidden


def test_the_annotation_is_no_longer_any() -> None:
    """`Any` is how a dict reached `evaluate` silently."""
    signature = inspect.signature(science.gate_candidates)

    assert signature.parameters["thresholds"].annotation not in (Any, "Any")


# --- 4-9. the production path ------------------------------------------------

class _NeverCalled:
    """A CandidateEvaluator stand-in that fails if profile gating measures."""

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:   # pragma: no cover
        raise AssertionError("profile gating must not re-measure a candidate")


def _plans() -> dict[str, dict[str, Any]]:
    return {arm: {"arm": arm, "candidates": [{
        "candidate_id": f"c5syn_{arm.lower()}_{index:04d}", "arm": arm,
        "route": PHYSICS if index % 2 == 0 else GPAT, "position": index,
        "recipe_id": f"r{index}", "recipe_ordinal": index,
        "live_dataset": "casia_fasd" if index % 2 == 0 else "msu_mfsd",
        "live_target_sample_id": f"live_{index:04d}"} for index in range(4)]}
        for arm in ARMS}


def _state(**overrides: Any) -> dict[str, Any]:
    plans = _plans()
    profiles = build_profiles(INHERITED_NOMINAL, nominal_source="x")
    metrics = {arm: {row["candidate_id"]: _raw() for row in plans[arm]["candidates"]}
               for arm in ARMS}
    base: dict[str, Any] = {
        "plans": plans, "profiles": profiles, "metrics": metrics,
        "selectable": science.candidate_pool(plans),
        "backends": _NeverCalled()}
    base.update(overrides)
    return base


def _run(tmp_path: Path, **overrides: Any) -> tuple[Any, dict[str, Any]]:
    reports = tmp_path / "reports" / "full" / "c6"
    reports.mkdir(parents=True, exist_ok=True)
    request = AdapterRequest(repo=tmp_path, profile=load_profile("full", repo=REPO))
    state = _state(**overrides)
    result = C6Adapter()._check_profile_matched_feasibility(request, state, reports)
    return result, state


def test_the_production_path_gates_without_an_attribute_error(tmp_path: Path) -> None:
    """The exact call that raised on the GPU host."""
    result, state = _run(tmp_path)

    assert result.mode == "CHECK_PROFILE_MATCHED_FEASIBILITY"
    assert result.status_axes.engineering != "BLOCKED"
    assert len(state["assessments"]) == 3


@pytest.mark.parametrize("profile", list(PROFILE_ORDER))
def test_every_profile_produced_decisions(tmp_path: Path, profile) -> None:
    _, state = _run(tmp_path)
    decisions = state["decisions"][profile]

    assert sorted(decisions) == sorted(ARMS)
    for arm in ARMS:
        assert len(decisions[arm]) == 4
        assert all("accepted" in row for row in decisions[arm])


def test_the_production_path_passes_a_real_thresholds_object(tmp_path: Path,
                                                             monkeypatch) -> None:
    """Intercept the canonical gate and inspect what it actually receives."""
    seen: list[Any] = []
    original = quality_gate.evaluate

    def recording(metrics, thresholds):
        seen.append(thresholds)
        return original(metrics, thresholds)

    monkeypatch.setattr(quality_gate, "evaluate", recording)
    _run(tmp_path)

    assert seen, "the gate really was called"
    assert all(isinstance(item, quality_gate.Thresholds) for item in seen)
    assert not any(isinstance(item, dict) for item in seen)


def test_the_adapter_calls_as_thresholds_not_the_raw_mapping() -> None:
    source = inspect.getsource(C6Adapter._check_profile_matched_feasibility)

    assert "profile.as_thresholds()" in source
    assert "gate_candidates(state[\"metrics\"][arm], profile.thresholds)" not in source


def test_all_three_profiles_gate_the_same_stored_measurements(tmp_path: Path) -> None:
    _, state = _run(tmp_path)
    before = {arm: dict(rows) for arm, rows in state["metrics"].items()}

    for arm in ARMS:
        assert state["metrics"][arm] == before[arm], "measurements were not mutated"
    # One measurement set, three gate passes over it.
    assert len(state["decisions"]) == 3


def test_no_candidate_is_re_measured_during_gating(tmp_path: Path) -> None:
    """`_NeverCalled.evaluate` raises if anything tries."""
    result, _ = _run(tmp_path)

    assert result.status_axes.engineering != "BLOCKED"


def test_no_evaluator_is_constructed_during_profile_gating() -> None:
    source = inspect.getsource(C6Adapter._check_profile_matched_feasibility)

    for forbidden in ("CandidateEvaluator", "evaluate_pool", "FrozenCalibration",
                      "reconstruct_discrete"):
        assert forbidden not in source, forbidden


# --- 10, 11. profiles really can differ, and nothing was retuned -------------

def test_different_profiles_can_yield_different_decisions(tmp_path: Path) -> None:
    a = INHERITED_NOMINAL["tau_id"]
    strict_bound = a + 0.10 * (1.0 - a)
    between = (a + strict_bound) / 2.0
    plans = _plans()
    metrics = {arm: {row["candidate_id"]: _raw(identity_cosine=between)
                     for row in plans[arm]["candidates"]} for arm in ARMS}

    _, state = _run(tmp_path, plans=plans, metrics=metrics,
                    selectable=science.candidate_pool(plans))

    assert all(row["accepted"] is False for row in state["decisions"]["STRICT"]["RND"])
    assert all(row["accepted"] is True for row in state["decisions"]["NOMINAL"]["RND"])
    assert all(row["accepted"] is True
               for row in state["decisions"]["PERMISSIVE"]["RND"])


def test_no_threshold_value_was_changed_by_this_fix() -> None:
    profiles = build_profiles(INHERITED_NOMINAL, nominal_source="x")

    assert dict(profiles["NOMINAL"].thresholds) == {
        "tau_fd": 0.5, "tau_id": 0.547440037939055,
        "tau_lm": 0.00836817528937794, "tau_parse": 0.7094826178704915,
        "tau_out": 0.0, "tau_fp": 5.687657785453908}
    a = INHERITED_NOMINAL["tau_id"]
    assert profiles["STRICT"].thresholds["tau_id"] == a + 0.10 * (1.0 - a)
    assert profiles["PERMISSIVE"].thresholds["tau_id"] == max(0.0, 0.90 * a)


# --- the other profile.thresholds consumers stay on the mapping --------------

def test_identity_and_serialization_still_use_the_raw_mapping() -> None:
    """Not every `profile.thresholds` was wrong — only the gating one."""
    profiles = build_profiles(INHERITED_NOMINAL, nominal_source="x")
    source = (REPO / "src" / "prism_fas" / "pipeline" / "adapters" / "c6.py"
              ).read_text(encoding="utf-8")

    assert "science.threshold_identity(profile.thresholds)" in source
    assert '"thresholds": dict(profile.thresholds)' in source
    assert science.threshold_identity(profiles["NOMINAL"].thresholds)


def test_the_engineering_path_still_converts_as_it_always_did() -> None:
    source = (REPO / "src" / "prism_fas" / "pipeline" / "adapters" / "c6.py"
              ).read_text(encoding="utf-8")

    assert "profiles[NOMINAL].as_thresholds()" in source


# --- 12. the firewall ---------------------------------------------------------

def test_profile_gating_opens_no_target_artifact() -> None:
    source = inspect.getsource(C6Adapter._check_profile_matched_feasibility)

    for forbidden in ("siw", "SiW", "target_test", "label_live_spoof",
                      "source_dev"):
        assert forbidden not in source, forbidden
