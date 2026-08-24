"""No scientific profile may reach fixture machinery, in any adapter.

C4 proved the failure mode is not hypothetical: one workflow, written as an
engineering rehearsal, was executed by `--profile full`. A fixture batch, a
stand-in identity model and a one-step evaluator all ran under a scientific
profile, every check passed, and the stage reported PASS. Nothing was wrong with
the engineering — it was in the wrong place.

So this file is the standing audit rather than a one-off review: it walks every
C4-C13 adapter and requires each fixture producer to be either guarded by
`assert_fixture_permitted` or already inside a branch on `fixtures_permitted` /
`is_scientific`. A new fixture callsite added without a guard fails here.

`DECLARED_SCIENTIFIC_GAPS` records, per stage, what is known to be missing. It is
a ledger, not an excuse: every entry must still be reachable only under a
rehearsal, which the audit below re-checks.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

ADAPTERS = REPO / "src" / "prism_fas" / "pipeline" / "adapters"
STAGES = ("c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11", "c12", "c13")

#: Names that produce a fixture rather than resolve a real frozen input.
#:
#: `audit_batch` and `build_audit_detector` are here because leaving them out is
#: how C8's defect survived review: its row executor called both unconditionally
#: and this audit walked straight past them, so the ledger said "SMOKE_ROWS=2
#: caps the rows" while a scientific context would have run all 42 on fixtures.
#: A producer absent from this tuple is a producer nothing checks.
FIXTURE_PRODUCERS = ("_fixture_batch", "_fixture_roots", "_fixture_rows",
                     "prediction_rows", "evaluation_labels",
                     "audit_batch", "build_audit_detector", "_complete_evidence")

#: Rehearsal-only budget machinery. A scientific context truncates nothing, so a
#: function that reads one of these must be unreachable from the scientific path.
BUDGET_NAMES = ("SmokeBudget", "SMOKE_ROWS", "SMOKE_BATCH", "SMOKE_CANDIDATES_PER_ARM")

#: What each stage still owes before it can produce scientific evidence.
#: `scientific_executor` is True only where a scientific branch exists today.
DECLARED_SCIENTIFIC_GAPS: dict[str, dict[str, object]] = {
    "c4": {"scientific_executor": True,
           "note": "GPATTrainer branch wired in this milestone"},
    "c5": {"scientific_executor": True,
           "note": "_scientific_workflow renders 2048 candidates per arm through "
                   "the frozen C4 checkpoint and the M7 physics engine. The "
                   "rehearsal path is unchanged and still reaches _render_gpat, "
                   "its fixture batch and its randomly initialized generator, "
                   "which is why that path may never be entered under a "
                   "scientific ExecutionContext."},
    "c6": {"scientific_executor": True,
           "note": "_scientific_workflow fits NOMINAL from source_train at C6, "
                   "gates the verified C5 pool under STRICT/NOMINAL/PERMISSIVE and "
                   "builds three matched banks under C6_MATCHED_BANK_SELECTOR_V1. "
                   "The rehearsal path is unchanged and still reaches "
                   "ENGINEERING_NOMINAL and gate_metrics, which is why neither may "
                   "be entered under a scientific ExecutionContext."},
    "c7": {"scientific_executor": True,
           "note": "_scientific_workflow verifies the C6 closure with the canonical "
                   "strict verifier, binds the approved LR interpretation and the "
                   "C7 search decision, trains every trial of the frozen §15.2.2 "
                   "envelope through M9Trainer on the decided arm's C6 matched "
                   "bank, and writes DETECTOR_CONFIG_LOCK.json. The readiness path "
                   "is unchanged and still reaches _fixture_batch, audit_batch and "
                   "build_audit_detector, which is why none of them may be entered "
                   "under a scientific ExecutionContext."},
    "c8": {"scientific_executor": True,
           "note": "_scientific_workflow trains every scheduled row through "
                   "M9Trainer at the frozen C7 configuration, its own C6 matched "
                   "bank and its protocol's source domains. The rehearsal path is "
                   "unchanged and still reaches _run_one (audit_batch + "
                   "build_audit_detector + SmokeBudget) and _failure_preservation's "
                   "constructed failure, which is why both are guarded."},
    "c9": {"scientific_executor": True,
           "note": "_scientific_workflow loads REAL C8 run manifests through "
                   "source_evidence, re-hashes the checkpoints they name and "
                   "freezes SOURCE_MATRIX_LOCK_C over them. The rehearsal path "
                   "still builds constructed evidence to reach the seven refusal "
                   "branches, which a complete real matrix has none of."},
    "c10": {"scientific_executor": False,
            "note": "_fixture_roots builds a synthetic target package; the real "
                    "sealed package is resolved by sources._real_target_roots, "
                    "which is gated by the target firewall."},
    "c11": {"scientific_executor": False,
            "note": "prediction rows come from adapters.tiny under rehearsal."},
    "c12": {"scientific_executor": False,
            "note": "labels are fabricated by adapters.tiny under rehearsal; the "
                    "real scorer is the only component permitted to read them."},
    "c13": {"scientific_executor": False, "note": "closure over upstream evidence."},
}


def _module(stage: str) -> tuple[str, ast.Module]:
    source = (ADAPTERS / f"{stage}.py").read_text(encoding="utf-8")
    return source, ast.parse(source)


def _enclosing_function(tree: ast.Module, node: ast.AST) -> ast.FunctionDef | None:
    for candidate in ast.walk(tree):
        if isinstance(candidate, ast.FunctionDef) and node in ast.walk(candidate):
            return candidate
    return None


def _fixture_calls(tree: ast.Module) -> list[ast.Call]:
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = (node.func.id if isinstance(node.func, ast.Name)
                else node.func.attr if isinstance(node.func, ast.Attribute) else "")
        if name in FIXTURE_PRODUCERS:
            found.append(node)
    return found


# --- the audit ---------------------------------------------------------------

@pytest.mark.parametrize("stage", STAGES)
def test_every_fixture_call_is_guarded_or_branched(stage: str) -> None:
    """A fixture producer must be unreachable from a scientific context.

    Two acceptable proofs, because both really are proofs: an explicit
    `assert_fixture_permitted` in the same function, or the call sitting inside
    a function that already branches on the context.
    """
    source, tree = _module(stage)
    unguarded: list[str] = []

    for call in _fixture_calls(tree):
        function = _enclosing_function(tree, call)
        if function is None:
            continue
        body = ast.get_source_segment(source, function) or ""
        guarded = ("assert_fixture_permitted" in body
                   or "fixtures_permitted" in body
                   or "is_scientific" in body)
        if not guarded:
            name = (call.func.id if isinstance(call.func, ast.Name)
                    else call.func.attr)
            unguarded.append(f"{function.name} -> {name}")

    assert unguarded == [], (
        f"{stage}.py reaches a fixture producer with no context guard: {unguarded}")


@pytest.mark.parametrize("stage", STAGES)
def test_every_stage_is_covered_by_the_gap_ledger(stage: str) -> None:
    assert stage in DECLARED_SCIENTIFIC_GAPS
    entry = DECLARED_SCIENTIFIC_GAPS[stage]
    assert isinstance(entry["scientific_executor"], bool)
    assert entry["note"], f"{stage} has no recorded reason"


def test_the_source_side_stages_declare_a_scientific_executor_today() -> None:
    """Honest ledger. C4-C9 are wired; C10-C13 are not, and both halves are
    asserted so that wiring a stage without updating the ledger fails here."""
    wired = sorted(stage for stage, entry in DECLARED_SCIENTIFIC_GAPS.items()
                   if entry["scientific_executor"])
    unwired = sorted(stage for stage, entry in DECLARED_SCIENTIFIC_GAPS.items()
                     if not entry["scientific_executor"])

    assert wired == ["c4", "c5", "c6", "c7", "c8", "c9"]
    assert unwired == ["c10", "c11", "c12", "c13"]


def test_a_stage_without_a_scientific_executor_claims_no_scientific_evidence() -> None:
    """`scientific_evidence=True` is the only route to a PASS on that axis."""
    for stage, entry in DECLARED_SCIENTIFIC_GAPS.items():
        if entry["scientific_executor"]:
            continue
        source, _ = _module(stage)
        assert "scientific_evidence" not in source, (
            f"{stage}.py claims scientific evidence but has no scientific executor")


def test_the_c5_gpat_render_is_explicitly_guarded() -> None:
    """The stage the audit was asked to look at first."""
    source, tree = _module("c5")
    render = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "_render_gpat")
    body = ast.get_source_segment(source, render) or ""

    assert "assert_fixture_permitted" in body
    assert body.index("assert_fixture_permitted") < body.index("_fixture_batch(")
    assert "trained_checkpoint_used" in source, (
        "the artifact must keep saying the generator was untrained")


def test_the_guard_is_derived_from_the_profile_not_passed_in() -> None:
    from prism_fas.pipeline.adapters.common import assert_fixture_permitted
    from prism_fas.pipeline.execution import ExecutionContext

    execution = (REPO / "src" / "prism_fas" / "pipeline" / "execution.py").read_text(
        encoding="utf-8")
    assert "fixtures_permitted=not eligible" in execution.replace(" ", "").replace(
        "fixtures_permitted=noteligible", "fixtures_permitted=not eligible")

    assert callable(assert_fixture_permitted)
    assert hasattr(ExecutionContext, "for_profile")


def test_no_adapter_hard_codes_a_scientific_pass() -> None:
    """The axis must follow evidence. A literal would be a way around that."""
    for stage in STAGES:
        source, _ = _module(stage)
        assert 'scientific="PASS"' not in source, stage
        assert "DualStatus(" not in source, (
            f"{stage}.py constructs a status directly instead of going through "
            "EngineeringAdapter.result, which is where the axis is derived")

# --- the C8 defect, as standing regressions ----------------------------------

SCIENTIFIC_FORBIDDEN: dict[str, tuple[str, ...]] = {
    "c7": ("_fixture_batch", "audit_batch", "build_audit_detector"),
    "c8": ("_run_one", "_failure_preservation", "audit_batch", "build_audit_detector"),
    "c9": ("_complete_evidence", "_refusals"),
    "c10": ("_build_fixture", "_fixture_roots"),
    "c11": ("prediction_rows",),
    "c12": ("evaluation_labels",),
}


def _scientific_workflow_body(stage: str) -> str:
    source, tree = _module(stage)
    node = next((item for item in ast.walk(tree)
                 if isinstance(item, ast.FunctionDef)
                 and item.name == "_scientific_workflow"), None)
    if node is None:
        return ""
    return ast.get_source_segment(source, node) or ""


@pytest.mark.parametrize("stage", sorted(SCIENTIFIC_FORBIDDEN))
def test_a_scientific_workflow_names_no_fixture_helper(stage: str) -> None:
    """The scientific workflow may not so much as MENTION a fixture helper.

    A name check rather than a call check on purpose: `_scientific_workflow` is
    the entry point, and a helper it calls could reach a fixture two frames
    down. What this pins is the top-level separation the C8 defect lacked —
    there was no `_scientific_workflow` at all, so the fixture executor was the
    only executor.
    """
    body = _scientific_workflow_body(stage)
    if not body:
        pytest.skip(f"{stage} declares no scientific workflow yet")
    named = [name for name in SCIENTIFIC_FORBIDDEN[stage] if name in body]
    assert named == [], (
        f"{stage}._scientific_workflow names fixture helper(s) {named}")


@pytest.mark.parametrize("stage", ("c7", "c8", "c9"))
def test_a_wired_stage_dispatches_on_the_context(stage: str) -> None:
    """One `workflow` that branches, not one that adapts.

    The C4 and C8 defects were the same shape: a single workflow, written as a
    rehearsal, executed by `--profile full`. A stage with a scientific executor
    must choose between two named methods.
    """
    source, tree = _module(stage)
    workflow = next(item for item in ast.walk(tree)
                    if isinstance(item, ast.FunctionDef) and item.name == "workflow")
    body = ast.get_source_segment(source, workflow) or ""

    assert "context.is_scientific" in body, f"{stage}.workflow does not branch"
    assert "_scientific_workflow" in body
    assert "_engineering_workflow" in body


def test_the_c8_fixture_executor_is_guarded_at_its_first_statement() -> None:
    """The exact defect: `_run_one` was reachable from a scientific context.

    Asserting the guard is FIRST, not merely present: a guard after the model is
    built has already imported torch, resolved a variant and allocated an audit
    detector under a scientific profile.
    """
    source, tree = _module("c8")
    for name in ("_run_one", "_failure_preservation"):
        function = next(item for item in ast.walk(tree)
                        if isinstance(item, ast.FunctionDef) and item.name == name)
        statements = [item for item in function.body
                      if not isinstance(item, ast.Expr)
                      or not isinstance(item.value, ast.Constant)]
        first = ast.get_source_segment(source, statements[0]) or ""
        assert "assert_fixture_permitted" in first, (
            f"c8.{name} does not guard on its first statement; it guards on "
            f"{first.splitlines()[0] if first else '<nothing>'}")


@pytest.mark.parametrize("stage", ("c7", "c8", "c9"))
def test_no_scientific_workflow_reads_a_rehearsal_budget(stage: str) -> None:
    """§8: no SMOKE_ROWS, first-N or fixture cardinality may affect a full run."""
    body = _scientific_workflow_body(stage)
    if not body:
        pytest.skip(f"{stage} declares no scientific workflow yet")
    named = [name for name in BUDGET_NAMES if name in body]
    assert named == [], f"{stage}._scientific_workflow reads {named}"


def test_the_context_never_hands_a_budget_to_a_scientific_run() -> None:
    """The structural half of the same rule, at the source."""
    from prism_fas.pipeline.execution import ExecutionContext

    class _Profile:
        name = "full"
        scientific_eligible = True
        may_select_scientific_winner = True
        reports_namespace = "reports/full"
        runs_namespace = "runs/full"
        engineering_budget = {"max_steps_per_epoch": 2, "max_seeds": 1}

    context = ExecutionContext.for_profile(_Profile())
    assert context.budget is None
    assert context.fixtures_permitted is False
    assert context.limit(42, sample=2) == 42
    assert context.budget_or("steps", 45) == 45


def test_the_fixture_guard_refuses_under_a_scientific_context() -> None:
    from prism_fas.pipeline.adapters.common import (FixtureInScientificContext,
                                                    assert_fixture_permitted)
    from prism_fas.pipeline.execution import ExecutionContext

    class _Profile:
        name = "full"
        scientific_eligible = True
        may_select_scientific_winner = True
        reports_namespace = "reports/full"
        runs_namespace = "runs/full"
        engineering_budget = None

    with pytest.raises(FixtureInScientificContext):
        assert_fixture_permitted(ExecutionContext.for_profile(_Profile()), "a fixture")


def test_c7_no_longer_asserts_its_own_scientific_lock_is_absent() -> None:
    """The check that made a legitimate scientific C7 unreachable.

    `reports/full/c7/DETECTOR_CONFIG_LOCK.json` is what C8 declares as a required
    input. An engineering check asserting it does NOT exist meant the full
    profile could never legitimately produce it, and that once produced every
    rehearsal would fail on a file it was right to have written.
    """
    source, _tree = _module("c7")

    assert "c7_no_scientific_config_lock_written" not in source
    assert "c7_rehearsal_writes_no_scientific_config_lock" in source
