"""Measure once, gate three times — and never inherit the evaluator's verdict.

`CandidateEvaluator.evaluate` returns `quality_gate.evaluate(metrics,
self.calibration.thresholds)`. That is a gate ENVELOPE: the raw measurements sit
under `["metrics"]`, next to an acceptance decision already taken under the
calibration's own NOMINAL thresholds. `evaluate_pool` stored the envelope, and
`gate_candidates` then handed it to `quality_gate.evaluate` as if it were the
flat metric map — so the GPU run measured every candidate and then died with
"metric 'face_detection_score' is missing" at CHECK_PROFILE_MATCHED_FEASIBILITY.

The fix is at the C6 boundary. `CandidateEvaluator` is inherited canonical
measurement code and is byte-identical to Version B's, which is the whole basis
of the §11.4 compatibility argument — so it is not touched.

Two properties matter beyond "it does not crash": the stored measurement is
threshold-INDEPENDENT, and the evaluator's embedded NOMINAL verdict never
becomes C6's. Otherwise NOMINAL would hold a privileged position among three
profiles that are supposed to be assessed identically.
"""
from __future__ import annotations

import hashlib
import inspect
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.synthesis import c5_raw_generation as raw  # noqa: E402
from prism_fas.synthesis import c6_scientific as science  # noqa: E402
from prism_fas.synthesis.c6_threshold_inheritance import INHERITED_NOMINAL  # noqa: E402
from prism_fas.synthesis.gate_profiles import build_profiles  # noqa: E402
from prism_fas.synthesis.quality_gate import (QualityGateError,  # noqa: E402
                                              Thresholds, evaluate)
from prism_fas.synthesis.synthetic_bank import CandidateEvaluator  # noqa: E402


def _raw(**overrides: Any) -> dict[str, Any]:
    """A plausible raw metric set, in the shape CandidateEvaluator emits."""
    base = {"face_detection_score": 0.93, "identity_cosine": 0.88,
            "landmark_nme": 0.004, "outside_mask_parsing_dice": 0.97,
            "outside_mask_max_error": 0.0, "measured_artifact_strength": 0.40,
            "requested_artifact_strength": 0.40, "fingerprint_score": 1.2,
            "support_overlap": 0.99, "reference_detection_score": 0.95,
            "landmark_detected": True}
    base.update(overrides)
    return base


def _envelope(**overrides: Any) -> dict[str, Any]:
    """What `CandidateEvaluator.evaluate` really returns."""
    return evaluate(_raw(**overrides), Thresholds.from_dict(INHERITED_NOMINAL))


# --- 1. the producer's real shape --------------------------------------------

def test_the_evaluator_returns_a_gate_envelope_with_nested_metrics() -> None:
    source = inspect.getsource(CandidateEvaluator.evaluate)
    envelope = _envelope()

    assert "return evaluate(metrics, self.calibration.thresholds)" in source
    assert "metrics" in envelope
    assert "face_detection_score" not in envelope, (
        "the raw metrics are nested; that is the whole defect")
    assert "face_detection_score" in envelope["metrics"]


def test_the_envelope_carries_a_decision_taken_under_nominal() -> None:
    envelope = _envelope()

    for field in ("accepted", "failed_gates", "gates", "q", "threshold_hash"):
        assert field in envelope, field


def test_the_evaluator_is_unchanged_from_version_b() -> None:
    """The compatibility argument rests on this file being byte-identical."""
    version_b = REPO.parent / "PRISM_FAS_B_Project"
    relative = "src/prism_fas/synthesis/synthetic_bank.py"
    if not (version_b / relative).is_file():
        pytest.skip("the Version-B tree is not mounted beside this repository")

    assert (hashlib.sha256((REPO / relative).read_bytes()).hexdigest()
            == hashlib.sha256((version_b / relative).read_bytes()).hexdigest())


# --- 2, 3, 4. the boundary stores raw metrics only ---------------------------

def test_the_boundary_unwraps_to_the_raw_metric_map() -> None:
    unwrapped = science.raw_metrics_of(_envelope(), "c0")

    assert set(unwrapped) == set(_raw())
    for name in science.REQUIRED_RAW_METRICS:
        assert name in unwrapped, name


def test_the_unwrapped_map_carries_no_threshold_dependent_field() -> None:
    unwrapped = science.raw_metrics_of(_envelope(), "c0")

    for field in science.THRESHOLD_DEPENDENT_FIELDS:
        assert field not in unwrapped, field


def test_the_required_field_list_matches_what_the_gate_reads() -> None:
    """Taken from the canonical gate, not restated independently."""
    source = inspect.getsource(evaluate)

    for name in science.REQUIRED_RAW_METRICS:
        assert f'"{name}"' in source, name


# --- schema validation, failing closed ---------------------------------------

def test_a_missing_metrics_block_fails_closed() -> None:
    with pytest.raises(science.ScientificGateError, match="no 'metrics' block"):
        science.raw_metrics_of({"accepted": True, "q": 0.5}, "c0")


def test_a_non_mapping_metrics_block_fails_closed() -> None:
    with pytest.raises(science.ScientificGateError, match="not a map"):
        science.raw_metrics_of({"metrics": [1, 2, 3]}, "c0")


@pytest.mark.parametrize("missing", list(science.REQUIRED_RAW_METRICS))
def test_a_missing_required_metric_fails_closed(missing) -> None:
    metrics = _raw()
    metrics.pop(missing)

    with pytest.raises(science.ScientificGateError, match="missing"):
        science.raw_metrics_of({"metrics": metrics}, "c0")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_required_metric_fails_closed(value) -> None:
    with pytest.raises(science.ScientificGateError, match="not finite"):
        science.raw_metrics_of({"metrics": _raw(identity_cosine=value)}, "c0")


def test_an_unrecognized_field_is_refused_rather_than_passed_through() -> None:
    """No silent unwrapping of arbitrary nested objects."""
    with pytest.raises(science.ScientificGateError, match="unrecognized"):
        science.raw_metrics_of({"metrics": _raw(surprise_field=1.0)}, "c0")


def test_the_canonical_diagnostics_are_allowed_through() -> None:
    unwrapped = science.raw_metrics_of(_envelope(), "c0")

    for name in science.DIAGNOSTIC_RAW_METRICS:
        assert name in unwrapped, name


# --- 5, 6, 7, 8. one measurement, three profiles -----------------------------

@pytest.mark.parametrize("profile", ["STRICT", "NOMINAL", "PERMISSIVE"])
def test_gate_candidates_can_apply_every_profile_to_the_raw_metrics(profile) -> None:
    metrics = {"c0": science.raw_metrics_of(_envelope(), "c0")}
    profiles = build_profiles(INHERITED_NOMINAL, nominal_source="x")

    decisions = science.gate_candidates(
        metrics, Thresholds.from_dict(profiles[profile].thresholds))

    assert [row["candidate_id"] for row in decisions] == ["c0"]
    assert isinstance(decisions[0]["accepted"], bool)
    assert decisions[0]["q"] is not None


def test_the_envelope_would_still_crash_the_gate_if_stored_whole() -> None:
    """The exact GPU failure, pinned as the thing that must not recur."""
    with pytest.raises(QualityGateError, match="face_detection_score"):
        science.gate_candidates({"c0": _envelope()},
                                Thresholds.from_dict(INHERITED_NOMINAL))


def test_changing_the_profile_changes_decisions_without_remeasuring() -> None:
    """A candidate on the wrong side of STRICT but inside PERMISSIVE."""
    a = INHERITED_NOMINAL["tau_id"]
    strict_bound = a + 0.10 * (1.0 - a)
    metrics = {"c0": science.raw_metrics_of(
        _envelope(identity_cosine=(a + strict_bound) / 2.0), "c0")}
    profiles = build_profiles(INHERITED_NOMINAL, nominal_source="x")

    verdicts = {name: science.gate_candidates(
        metrics, Thresholds.from_dict(profiles[name].thresholds))[0]["accepted"]
        for name in ("STRICT", "NOMINAL", "PERMISSIVE")}

    assert verdicts["STRICT"] is False
    assert verdicts["NOMINAL"] is True and verdicts["PERMISSIVE"] is True
    # ...and the stored measurement never moved.
    assert metrics["c0"]["identity_cosine"] == (a + strict_bound) / 2.0


def test_the_measurement_layer_is_threshold_independent() -> None:
    """No threshold reaches the measurement code — the docstring may say so."""
    import ast

    source = inspect.getsource(science.evaluate_pool)
    node = ast.parse(source.strip()).body[0]
    body = "\n".join(ast.unparse(item) for item in node.body[1:])

    assert "raw_metrics_of(result" in body
    for forbidden in ("thresholds", "profile", "accepted", "STRICT", "PERMISSIVE"):
        assert forbidden not in body, forbidden


# --- 9, 10. measured once, and never a semantic failure ----------------------

class _Store:
    def load(self, sample_id: str) -> tuple[Any, dict[str, Any]]:
        return object(), {}


class _CountingEvaluator:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def evaluate(self, discrete, *, live_target_sample_id, requested_strength,
                 requested_support):
        self.seen.append(live_target_sample_id)
        return _envelope()


def _pool(tmp_path: Path, statuses: dict[str, str]) -> list[dict[str, Any]]:
    """Write candidate records with the given terminal statuses."""
    import json

    rows = []
    for index, (candidate_id, status) in enumerate(sorted(statuses.items())):
        directory = raw.candidate_dir(tmp_path, "RND", candidate_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / raw.RECORD_NAME).write_text(
            json.dumps({"status": status}), encoding="utf-8")
        rows.append({"candidate_id": candidate_id, "recipe_id": "r0",
                     "live_target_sample_id": f"live_{index:03d}"})
    return rows


def test_the_evaluator_runs_once_per_generated_candidate(tmp_path, monkeypatch) -> None:
    rows = _pool(tmp_path, {"c0": raw.GENERATED, "c1": raw.GENERATED})
    evaluator = _CountingEvaluator()
    monkeypatch.setattr(science, "requested_support_for",
                        lambda store, bank, row: (None, _Graph()))
    monkeypatch.setattr(science, "reconstruct_discrete",
                        lambda directory, original: object())

    metrics = science.evaluate_pool(evaluator, _Store(), {},
                                    candidate_root=tmp_path, arm="RND", rows=rows)

    assert sorted(metrics) == ["c0", "c1"]
    assert len(evaluator.seen) == 2, "measured exactly once each"
    # ...and what was stored is the RAW map, not the envelope.
    for stored in metrics.values():
        assert "face_detection_score" in stored
        assert "metrics" not in stored
        for field in science.THRESHOLD_DEPENDENT_FIELDS:
            assert field not in stored, field


def test_the_pool_it_stores_can_be_gated_by_every_profile(tmp_path, monkeypatch) -> None:
    """End to end: what `evaluate_pool` returns is what `gate_candidates` eats."""
    rows = _pool(tmp_path, {"c0": raw.GENERATED})
    monkeypatch.setattr(science, "requested_support_for",
                        lambda store, bank, row: (None, _Graph()))
    monkeypatch.setattr(science, "reconstruct_discrete",
                        lambda directory, original: object())

    metrics = science.evaluate_pool(_CountingEvaluator(), _Store(), {},
                                    candidate_root=tmp_path, arm="RND", rows=rows)
    profiles = build_profiles(INHERITED_NOMINAL, nominal_source="x")

    for name in ("STRICT", "NOMINAL", "PERMISSIVE"):
        decisions = science.gate_candidates(
            metrics, Thresholds.from_dict(profiles[name].thresholds))
        assert len(decisions) == 1 and "accepted" in decisions[0]


def test_a_semantic_failure_is_never_measured(tmp_path, monkeypatch) -> None:
    rows = _pool(tmp_path, {"c0": raw.GENERATED, "c1": raw.FAILED_GENERATION})
    evaluator = _CountingEvaluator()
    monkeypatch.setattr(science, "requested_support_for",
                        lambda store, bank, row: (None, _Graph()))
    monkeypatch.setattr(science, "reconstruct_discrete",
                        lambda directory, original: object())

    metrics = science.evaluate_pool(evaluator, _Store(), {},
                                    candidate_root=tmp_path, arm="RND", rows=rows)

    assert sorted(metrics) == ["c0"]
    assert len(evaluator.seen) == 1


class _Graph:
    nodes = [type("Node", (), {"strength": 0.4})()]


# --- 11, 12. q, and the firewall ---------------------------------------------

def test_q_comes_only_from_the_profile_gate_and_selects_nothing() -> None:
    unwrapped = science.raw_metrics_of(_envelope(), "c0")
    decisions = science.gate_candidates({"c0": unwrapped},
                                        Thresholds.from_dict(INHERITED_NOMINAL))

    assert "q" not in unwrapped, "the measurement layer carries no q"
    assert decisions[0]["q"] is not None, "the gate produces it"
    # And the selector still cannot see it as an ordering key.
    from prism_fas.synthesis.c6_matched_bank import select_route_bank

    assert ".q" not in inspect.getsource(select_route_bank).split("selected.append")[0]


def test_the_boundary_opens_no_target_or_source_dev() -> None:
    source = inspect.getsource(science.evaluate_pool) + inspect.getsource(
        science.raw_metrics_of)

    for forbidden in ("siw", "SiW", "target_test", "label_live_spoof",
                      "source_dev"):
        assert forbidden not in source, forbidden
