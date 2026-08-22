"""The C6 scientific contract: what is determined, and the one thing that is not.

Phase B of this milestone audited C6 before writing any executor. Two
architecture defects were found:

* C6 required its own `QUALITY_CALIBRATION.json` as a precondition. §11.4 fits
  NOMINAL from the source_train benign population AT C6, so that file is a C6
  OUTPUT. The requirement made C6 depend on itself, and the only way to satisfy
  it would have been to hand-write a fitted scientific threshold.
* C6's `full` profile still runs the engineering workflow — `ENGINEERING_NOMINAL`
  and `gate_metrics` fixture rows — with no scientific branch of its own.

The first is mechanical and is fixed here. The second is not fixed, because the
deterministic matched-bank selector §11.3 requires is not determined by the
frozen spec, the frozen configs or any canonical module. These tests pin the
audit so the gap cannot quietly close by accident.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.pipeline.adapters.c6 import C6Adapter  # noqa: E402

C6_SOURCE = (REPO / "src" / "prism_fas" / "pipeline" / "adapters" / "c6.py"
             ).read_text(encoding="utf-8")


# --- 11. the self-dependency is gone -----------------------------------------

def test_c6_does_not_require_its_own_calibration_output_as_an_input() -> None:
    required = {item.name: item.relative_path
                for item in C6Adapter().required_inputs()}

    assert "quality_calibration" not in required
    assert "reports/full/c6/QUALITY_CALIBRATION.json" not in required.values(), (
        "§11.4 fits NOMINAL at C6; requiring the fitted file first is circular")


def test_the_calibration_is_documented_as_a_c6_output() -> None:
    source = ast.get_source_segment(C6_SOURCE, next(
        node for node in ast.walk(ast.parse(C6_SOURCE))
        if isinstance(node, ast.FunctionDef) and node.name == "required_inputs")) or ""

    assert "QUALITY_CALIBRATION.json` is NOT listed here" in source
    assert "OUTPUT" in source


def test_c6_still_requires_the_verified_c5_pool() -> None:
    required = {item.name for item in C6Adapter().required_inputs()}

    assert "c5_synthesis_lock" in required
    assert "quality_gate_config" in required


# --- 14. only a strictly verified pool reaches C6 -----------------------------

def test_the_c6_precondition_uses_the_shared_pre_gate_conclusion() -> None:
    source = ast.get_source_segment(C6_SOURCE, next(
        node for node in ast.walk(ast.parse(C6_SOURCE))
        if isinstance(node, ast.FunctionDef)
        and node.name == "semantic_preconditions")) or ""

    assert "verify_c5_synthesis_lock" in source
    # The pre-gate conclusion, not the completion one: a complete pool that is
    # already below the 512 floor is valid C5 evidence and still blocks C6.
    assert 'verification["c6_pre_gate_input_ready"]' in source
    assert '"present": verification["c5_scientific_complete"]' not in source


# --- 12, 13. no scientific branch exists, and none is faked ------------------

def test_c6_declares_its_scientific_executor() -> None:
    from tests.pipeline.test_scientific_fixture_leakage import DECLARED_SCIENTIFIC_GAPS

    assert DECLARED_SCIENTIFIC_GAPS["c6"]["scientific_executor"] is True
    assert "def _scientific_workflow" in C6_SOURCE
    assert "C6_MATCHED_BANK_SELECTOR_V1" in DECLARED_SCIENTIFIC_GAPS["c6"]["note"]


def test_c6_claims_scientific_evidence_in_exactly_one_place() -> None:
    """VERIFY_C6_LOCKS, and only when every bank lock verifies."""
    assert C6_SOURCE.count("scientific_evidence=") == 1
    assert "scientific_evidence=passed" in C6_SOURCE


def test_the_engineering_fixtures_are_still_confined_to_the_engineering_path() -> None:
    """`ENGINEERING_NOMINAL` and `gate_metrics` may never feed a scientific gate."""
    assert "from prism_fas.pipeline.adapters.tiny import" in C6_SOURCE
    assert "ENGINEERING_NOMINAL" in C6_SOURCE and "gate_metrics" in C6_SOURCE
    # ...and because there is no scientific branch, `full` reaches the
    # precondition gate and BLOCKS instead of running them.
    assert "def full_precondition_gate" not in C6_SOURCE, (
        "C6 must inherit the shared gate rather than define a weaker one")


def test_the_full_profile_blocks_rather_than_running_the_rehearsal(tmp_path: Path) -> None:
    from prism_fas.pipeline.adapters import AdapterRequest
    from prism_fas.pipeline.profiles import load_profile

    request = AdapterRequest(repo=tmp_path, profile=load_profile("full", repo=REPO))
    gate = C6Adapter().full_precondition_gate(request)

    assert gate is not None and gate.status == "BLOCKED"
    assert gate.status_axes.scientific == "BLOCKED"


# --- what the audit found already determined ---------------------------------

@pytest.mark.parametrize("module,name", [
    ("prism_fas.synthesis.quality_calibration", "calibrate"),
    ("prism_fas.synthesis.gate_profiles", "derive_profile"),
    ("prism_fas.synthesis.gate_profiles", "build_profiles"),
    ("prism_fas.synthesis.gate_profiles", "select_profile"),
    ("prism_fas.synthesis.synthetic_bank", "CandidateEvaluator"),
    ("prism_fas.synthesis.quality_gate", "evaluate"),
])
def test_the_determined_c6_operations_have_canonical_implementations(module, name) -> None:
    """Nothing in this list needs inventing; the executor would import them."""
    import importlib

    assert hasattr(importlib.import_module(module), name), f"{module}.{name}"


def test_the_three_profiles_and_the_frozen_formulas_are_canonical() -> None:
    from prism_fas.synthesis.gate_profiles import (GPAT_PER_ARM, PHYSICS_PER_ARM,
                                                   PROFILE_ORDER, derive_profile)

    assert PROFILE_ORDER == ("STRICT", "NOMINAL", "PERMISSIVE")
    assert PHYSICS_PER_ARM == 512 and GPAT_PER_ARM == 512
    # §11.4: higher-is-better lower bound a -> STRICT = a + 0.10 * (1 - a),
    # PERMISSIVE = max(0, 0.90 * a); lower-is-better a -> 0.90 * a and 1.10 * a.
    assert derive_profile({"tau_fd": 0.5}, "STRICT")["tau_fd"] == pytest.approx(0.55)
    assert derive_profile({"tau_fd": 0.5}, "PERMISSIVE")["tau_fd"] == pytest.approx(0.45)
    assert derive_profile({"tau_lm": 0.5}, "STRICT")["tau_lm"] == pytest.approx(0.45)
    assert derive_profile({"tau_lm": 0.5}, "PERMISSIVE")["tau_lm"] == pytest.approx(0.55)


def test_profile_selection_is_conjunctive_across_arms() -> None:
    """An arm may not qualify a profile on its own, and a strong arm may not
    carry a weak one — the Version-B confound §11.4 exists to prevent."""
    from prism_fas.synthesis.gate_profiles import ARMS, ArmFeasibility

    short = ArmFeasibility("DET", 2048, 400, 512)
    full = ArmFeasibility("LLM", 2048, 512, 512)

    assert short.feasible is False
    assert full.feasible is True
    assert len(ARMS) == 3


# --- 10. the one item that is NOT determined ---------------------------------

def test_the_matched_bank_selector_now_exists_and_is_frozen() -> None:
    """The gap this file recorded is closed by C6_MATCHED_BANK_SELECTOR_V1."""
    from prism_fas.synthesis import c6_matched_bank

    assert c6_matched_bank.SELECTOR_NAME == "C6_MATCHED_BANK_SELECTOR_V1"
    assert callable(c6_matched_bank.build_matched_banks)
    assert len(c6_matched_bank.DIMENSION_PRIORITY) == 5


def test_the_plan_only_counts_the_shape_and_the_selector_picks_the_candidates() -> None:
    """§11.3 names four balancing dimensions and no algorithm.

    `matched_bank_plan` returns COUNTS and a sentence describing what a selector
    would have to do. It never returns which candidates. Nothing else in the
    repository selects a subset of accepted candidates either, and Version B has
    no such selector because Version B is the confounded design this rule exists
    to fix.
    """
    from prism_fas.synthesis import gate_profiles
    from prism_fas.synthesis.gate_profiles import ArmFeasibility, matched_bank_plan

    plan = matched_bank_plan(
        {arm: ArmFeasibility(arm, 2048, 600, 700) for arm in gate_profiles.ARMS})

    assert plan["arms"]["RND"]["physics"] == 512
    assert plan["arms"]["RND"]["gpat"] == 512
    # Counts only. No candidate identity appears anywhere in the plan.
    assert "candidate_id" not in str(plan)
    assert "selected_candidates" not in plan
    assert isinstance(plan["selection_basis"], str), (
        "the balancing rule is a description, not an implementation")


def test_the_selector_decision_is_recorded_as_frozen() -> None:
    state = (REPO / "docs" / "PROJECT_STATE.md").read_text(encoding="utf-8")

    assert "C6_MATCHED_BANK_SELECTOR_V1" in state
    assert "RESOLVED_BY: C6_MATCHED_BANK_SELECTOR_V1" in state


# --- 21-23. the firewall and the frozen repository ---------------------------

def test_c6_resolves_no_target_artifact() -> None:
    for forbidden in ("siw", "SiW", "target_test.parquet", "label_live_spoof",
                      "_real_target_roots", "resolve_target"):
        assert forbidden not in C6_SOURCE, forbidden
    assert "c6_no_target_capability" in C6_SOURCE


def test_c6_does_not_infer_a_source_dev_permission() -> None:
    """§11.4 fits NOMINAL from source_train. C6 must not reach for source_dev on
    its own initiative; C8 is the stage the spec gives source_dev to."""
    for forbidden in ('split="source_dev"', "source_dev.parquet", "SOURCE_DEV",
                      "manifests/source_dev"):
        assert forbidden not in C6_SOURCE, forbidden


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
