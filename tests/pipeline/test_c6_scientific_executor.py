"""The scientific C6 executor: routing, calibration, profiles, failure.

C6 is where the Version-B confound is designed out, so most of what is checked
here is a negative: that one threshold identity reaches all three arms, that no
fixture can reach the scientific path, that a C5 semantic failure is never
measured, and that "no profile qualifies" ends the stage instead of widening it.

Nothing renders and nothing is gated for real — the calibration and the
evaluator need the source package and the weights, which is why `full` blocks on
this laptop. What is exercised here is the wiring, the ordering and the refusals.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.pipeline.adapters import AdapterRequest  # noqa: E402
from prism_fas.pipeline.adapters import c6 as c6_module  # noqa: E402
from prism_fas.pipeline.adapters.c6 import C6Adapter  # noqa: E402
from prism_fas.pipeline.profiles import load_profile  # noqa: E402
from prism_fas.synthesis import c6_matched_bank as mb  # noqa: E402
from prism_fas.synthesis import c6_scientific as science  # noqa: E402
from prism_fas.synthesis.c5_source_pair_plan import ARMS, GPAT, PHYSICS  # noqa: E402

C6_SOURCE = (REPO / "src" / "prism_fas" / "pipeline" / "adapters" / "c6.py"
             ).read_text(encoding="utf-8")


def _method_code(name: str, source: str = C6_SOURCE) -> str:
    """A method's source with its docstring removed.

    A docstring that NAMES what the method must never call would otherwise fail
    a check that the method never calls it.
    """
    tree = ast.parse(source)
    node = next(item for item in ast.walk(tree)
                if isinstance(item, ast.FunctionDef) and item.name == name)
    body = node.body[1:] if ast.get_docstring(node) else node.body
    return chr(10).join(ast.get_source_segment(source, item) or "" for item in body)


def _method_source(name: str, source: str = C6_SOURCE) -> str:
    tree = ast.parse(source)
    node = next(item for item in ast.walk(tree)
                if isinstance(item, ast.FunctionDef) and item.name == name)
    return ast.get_source_segment(source, node) or ""


SCIENTIFIC_METHODS = ("_scientific_workflow", "_verify_c5_pool",
                      "_build_source_reference", "_fit_nominal_calibration",
                      "_build_common_profiles", "_evaluate_generated_candidates",
                      "_run_bank_level_reliability",
                      "_check_profile_matched_feasibility",
                      "_select_strictest_profile", "_build_matched_banks",
                      "_verify_c6_locks")


# --- 26, 27, 36. the two paths are separate ----------------------------------

def test_the_workflow_branches_on_the_execution_context() -> None:
    source = _method_source("workflow")

    assert "context.is_scientific" in source
    assert "_scientific_workflow" in source and "_engineering_workflow" in source


def test_the_scientific_path_cannot_reach_engineering_nominal() -> None:
    body = chr(10).join(_method_code(name) for name in SCIENTIFIC_METHODS)

    assert "ENGINEERING_NOMINAL" not in body, (
        "a fixture threshold set may never be a scientific gate")


def test_the_scientific_path_cannot_reach_the_gate_metrics_fixture() -> None:
    body = chr(10).join(_method_code(name) for name in SCIENTIFIC_METHODS)

    for forbidden in ("gate_metrics", "SMOKE_CANDIDATES_PER_ARM", "ArmFeasibility("):
        assert forbidden not in body, forbidden


def test_the_engineering_workflow_is_unchanged_and_still_uses_its_fixtures() -> None:
    engineering = _method_source("_engineering_workflow")

    for mode in ("_apply_common_gate", "_profile_selection", "_reliability",
                 "_matched_banks", "_cardinality_refusal"):
        assert mode in engineering, mode
    assert "ENGINEERING_NOMINAL" in C6_SOURCE and "gate_metrics" in C6_SOURCE


def test_the_ten_scientific_substages_run_in_the_declared_order() -> None:
    assert c6_module.SCIENTIFIC_MODES == (
        "VERIFY_C5_POOL", "BUILD_SOURCE_REFERENCE", "FIT_NOMINAL_CALIBRATION",
        "BUILD_COMMON_PROFILES", "EVALUATE_GENERATED_CANDIDATES",
        "CHECK_PROFILE_MATCHED_FEASIBILITY", "SELECT_STRICTEST_PROFILE",
        "BUILD_MATCHED_BANKS", "RUN_BANK_LEVEL_RELIABILITY", "VERIFY_C6_LOCKS")

    source = _method_source("_scientific_workflow")
    order = [source.index(f"self._{mode.lower()}") for mode in c6_module.SCIENTIFIC_MODES]
    assert order == sorted(order)


def test_only_the_lock_verification_claims_scientific_evidence() -> None:
    assert C6_SOURCE.count("scientific_evidence=") == 1
    assert "scientific_evidence=passed" in _method_source("_verify_c6_locks")


# --- 28, 29. the calibration is a C6 output, fitted on source_train ----------

def test_the_calibration_is_not_a_precondition() -> None:
    required = {item.name for item in C6Adapter().required_inputs()}

    assert "quality_calibration" not in required
    assert "c5_synthesis_lock" in required


def test_the_calibration_is_written_by_c6() -> None:
    source = _method_source("_fit_nominal_calibration")

    assert "QUALITY_CALIBRATION.json" in source
    assert "write_artifact" in source
    assert "fit_nominal_calibration" in source
    assert "c6_calibration_is_an_output_not_an_input" in source


def test_the_calibration_is_source_train_only() -> None:
    reference = _method_source("_build_source_reference")

    assert "source_train" in reference
    assert "source_dev_opened=False" in reference
    assert "target_opened=False" in reference
    # C8 is the stage the spec gives source_dev to; C6 must not reach for it.
    # It is named only as an isolation proof or as the rule prose, never opened.
    body = chr(10).join(_method_code(name) for name in SCIENTIFIC_METHODS)
    for forbidden in ('split="source_dev"', "source_dev.parquet", "SOURCE_DEV",
                      'manifests/source_dev'):
        assert forbidden not in body, forbidden
    assert "source_dev_opened" in body, "the isolation is proved, not assumed"


def test_the_fitter_is_the_canonical_calibrator() -> None:
    source = ast.get_source_segment(
        (REPO / "src" / "prism_fas" / "synthesis" / "c6_scientific.py"
         ).read_text(encoding="utf-8"),
        next(node for node in ast.walk(ast.parse(
            (REPO / "src" / "prism_fas" / "synthesis" / "c6_scientific.py"
             ).read_text(encoding="utf-8")))
            if isinstance(node, ast.FunctionDef)
            and node.name == "fit_nominal_calibration")) or ""

    assert "from .quality_calibration import calibrate" in source
    assert "return calibrate(" in source


# --- 30, 31. exactly three profiles, one threshold identity each -------------

def test_exactly_three_preregistered_profiles_are_evaluated() -> None:
    assert science.PROFILE_ORDER == ("STRICT", "NOMINAL", "PERMISSIVE")

    with pytest.raises(science.ScientificGateError, match="exactly"):
        science.select_strictest_profile([_assessment("STRICT", feasible=True)])


def test_one_threshold_identity_is_shared_by_the_three_arms() -> None:
    source = _method_source("_build_common_profiles")

    assert "c6_one_threshold_identity_per_profile" in source
    assert "COMMON across RND/DET/LLM" in source
    # The identity is a function of the thresholds alone, so three arms under
    # one profile cannot produce three different identities.
    thresholds = {"tau_fd": 0.5, "tau_lm": 0.2}
    assert science.threshold_identity(thresholds) == science.threshold_identity(
        dict(reversed(list(thresholds.items()))))


# --- 32, 34. selection, and the failure that has no fallback -----------------

def _assessment(profile: str, *, feasible: bool = True, floor: bool = True,
                quota: bool = True) -> science.MatchedFeasibility:
    """Reliability is deliberately absent: it is not a selection input."""
    return science.MatchedFeasibility(
        profile=profile, arm_route_counts={}, route_quotas={},
        arms_meet_route_floor=floor and feasible,
        common_quota_feasible=quota and feasible)


def test_the_strictest_qualifying_profile_is_selected() -> None:
    decision = science.select_strictest_profile([
        _assessment("STRICT", feasible=False),
        _assessment("NOMINAL", feasible=True),
        _assessment("PERMISSIVE", feasible=True)])

    assert decision.selected == "NOMINAL"
    assert decision.failed is False
    assert [item["profile"] for item in decision.evaluations] == [
        "STRICT", "NOMINAL", "PERMISSIVE"], "every profile is recorded, refusals too"


def test_a_route_floor_shortfall_alone_refuses_a_profile() -> None:
    decision = science.select_strictest_profile([
        _assessment("STRICT", floor=False),
        _assessment("NOMINAL"), _assessment("PERMISSIVE")])

    assert decision.evaluations[0]["arms_meet_route_floor"] is False
    assert decision.evaluations[0]["feasible"] is False
    assert decision.selected == "NOMINAL"


def test_a_common_quota_shortfall_alone_refuses_a_profile() -> None:
    """The test that is stronger than the arm-count test."""
    decision = science.select_strictest_profile([
        _assessment("STRICT", quota=False),
        _assessment("NOMINAL"), _assessment("PERMISSIVE")])

    assert decision.evaluations[0]["arms_meet_route_floor"] is True
    assert decision.evaluations[0]["common_quota_feasible"] is False
    assert decision.evaluations[0]["feasible"] is False


def test_reliability_cannot_refuse_a_profile_at_selection_time() -> None:
    """Option B: the closure gate has no vote in choosing the profile."""
    decision = science.select_strictest_profile([
        _assessment(name) for name in science.PROFILE_ORDER])

    assert decision.selected == "STRICT"
    for evaluation in decision.evaluations:
        assert evaluation["reliability_used_for_selection"] is False
        assert "reliability_passed" not in evaluation


def test_no_qualifying_profile_is_a_scientific_failure_with_no_fallback() -> None:
    decision = science.select_strictest_profile([
        _assessment(name, feasible=False) for name in science.PROFILE_ORDER])

    assert decision.selected is None
    assert decision.failed is True
    payload = decision.as_dict()
    assert payload["c6_scientific_failed"] is True
    assert "C6 FAILS" in payload["on_failure"]
    # The rule names each thing that must NOT happen on failure.
    for refusal in ("never widened", "no arm gets its own threshold",
                    "target distribution is not altered",
                    "selector is not changed", "no candidate is regenerated"):
        assert refusal in payload["on_failure"], refusal


def test_the_adapter_stops_when_no_profile_qualifies() -> None:
    source = _method_source("_select_strictest_profile")

    assert 'state["halt"] = True' in source
    assert "c6_failure_has_no_fallback" in source
    workflow = _method_source("_scientific_workflow")
    assert 'state.get("halt")' in workflow


# --- 33, 35. semantic failures, and what the locks bind ----------------------

def test_a_c5_semantic_failure_is_never_a_candidate_evaluator_input() -> None:
    source = ast.get_source_segment(
        (REPO / "src" / "prism_fas" / "synthesis" / "c6_scientific.py"
         ).read_text(encoding="utf-8"),
        next(node for node in ast.walk(ast.parse(
            (REPO / "src" / "prism_fas" / "synthesis" / "c6_scientific.py"
             ).read_text(encoding="utf-8")))
            if isinstance(node, ast.FunctionDef) and node.name == "evaluate_pool")) or ""

    assert 'record.get("status") != raw.GENERATED' in source
    assert "continue" in source


def test_a_semantic_failure_becomes_neither_an_accept_nor_a_reject() -> None:
    """It holds no decision at all, so it cannot be counted either way."""
    decisions = [{"candidate_id": "generated_0", "accepted": True},
                 {"candidate_id": "generated_1", "accepted": False}]

    closure = science.provenance_closure(
        ["generated_0", "generated_1"], ["failed_0"], decisions, ["generated_0"])

    assert closure["semantic_failed"] == 1
    assert closure["selected"] + closure["accepted_not_selected"] + closure["rejected"] == 2
    assert closure["closed"] is True


def test_the_gate_accepting_an_unknown_candidate_is_refused() -> None:
    with pytest.raises(science.ScientificGateError, match="not a verified"):
        science.eligible_candidates([{"candidate_id": "ghost", "accepted": True}], {})


def test_the_bank_lock_binds_the_selector_identity_and_the_pool() -> None:
    contract = mb.selector_identity(quality_profile_identity="p" * 64,
                                    c5_pool_lock_sha256="a" * 64,
                                    decision_set_sha256="b" * 64)
    bank = {"size": 1024, "by_route": {PHYSICS: 512, GPAT: 512},
            "exposure": {}, "selected_set_sha256": "c" * 64, "selected": []}

    payload = science.bank_lock_payload(
        arm="RND", bank=bank, selector_contract=contract, profile="NOMINAL",
        threshold_identity="p" * 64, c5_pool_lock_sha256="a" * 64,
        provenance={"closed": True})

    assert payload["selector_identity_sha256"] == contract["selector_identity_sha256"]
    assert payload["selector_name"] == "C6_MATCHED_BANK_SELECTOR_V1"
    assert payload["c5_pool_lock_sha256"] == "a" * 64
    assert payload["quality_profile"] == "NOMINAL"
    assert payload["q_used_for_selection"] is False
    assert "TRAINING WEIGHT" in payload["q_purpose"]
    assert payload["final_bank_size"] == 1024
    assert payload["no_target_capability_proof"]["target_labels_resolved"] == 0


def test_the_lock_verification_checks_every_arm() -> None:
    source = _method_source("_verify_c6_locks")

    assert "for arm in selector.ARMS" in source
    assert "selected_set_sha256" in source
    assert "selector_identity_sha256" in source
    assert 'provenance_closure' in source
    assert "q_used_for_selection" in source


# --- the matched-feasibility object ------------------------------------------

def test_matched_feasibility_needs_both_cardinality_conditions() -> None:
    assert _assessment("STRICT").feasible is True
    assert _assessment("STRICT", floor=False).feasible is False
    assert _assessment("STRICT", quota=False).feasible is False
    # ...and reliability is not among them; there is no third input.
    import dataclasses

    fields = {f.name for f in dataclasses.fields(science.MatchedFeasibility)}
    assert "reliability_passed" not in fields


def test_the_arm_count_test_alone_is_no_longer_sufficient() -> None:
    source = _method_source("_check_profile_matched_feasibility")

    assert "c6_matched_feasibility_is_stronger_than_the_route_floor" in source
    assert "additionally_required" in source


# --- 37, 38. the firewall and the frozen repository --------------------------

def test_the_scientific_path_resolves_no_target_artifact() -> None:
    body = chr(10).join(_method_code(name) for name in SCIENTIFIC_METHODS)

    for forbidden in ("siw", "SiW", "target_test.parquet", "label_live_spoof",
                      "_real_target_roots", "resolve_target"):
        assert forbidden not in body, forbidden
    # The stage carries an explicit no-target-capability proof (L.6).
    assert "c6_no_target_capability" in body
    assert "target_labels_resolved=0" in body
    assert "target_roots_mounted=[]" in body


def test_the_full_profile_blocks_on_this_machine(tmp_path: Path) -> None:
    """No scientific C6 runs here: there is no verified C5 pool."""
    request = AdapterRequest(repo=tmp_path, profile=load_profile("full", repo=REPO))
    gate = C6Adapter().full_precondition_gate(request)

    assert gate is not None and gate.status == "BLOCKED"
    assert gate.status_axes.scientific == "BLOCKED"


def test_the_scientific_branch_fails_closed_without_a_c5_pool() -> None:
    request = AdapterRequest(repo=REPO, profile=load_profile("full", repo=REPO))

    results = C6Adapter().workflow(request, request.context)

    assert len(results) == 1, "it stops at the first substage"
    assert results[0].status == "BLOCKED"
    assert results[0].status_axes.scientific == "BLOCKED"


def test_version_b_is_untouched() -> None:
    import subprocess

    version_b = REPO.parent / "PRISM_FAS_B_Project"
    if not (version_b / ".git").exists():
        pytest.skip("Version B is not checked out beside this repository")

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=version_b,
                          capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=version_b,
                           capture_output=True, text=True, check=True).stdout.strip()

    assert head == "7799f7decd35db6987ce4578824e5bd8d9eab4ae"
    assert dirty == ""
