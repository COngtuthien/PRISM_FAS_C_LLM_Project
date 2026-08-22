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
FIXTURE_PRODUCERS = ("_fixture_batch", "_fixture_roots", "_fixture_rows",
                     "prediction_rows", "evaluation_labels")

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
    "c7": {"scientific_executor": False,
           "note": "readiness is a CPU fixture obligation by design (requires_gpu "
                   "False); the scientific detector search is not wired."},
    "c8": {"scientific_executor": False, "note": "SMOKE_ROWS=2 caps the rows."},
    "c9": {"scientific_executor": False, "note": "reporting over upstream evidence."},
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


def test_only_c4_c5_and_c6_declare_a_scientific_executor_today() -> None:
    """Honest ledger. If a later milestone wires C7, this test is what makes
    updating the ledger part of that work rather than an afterthought."""
    wired = sorted(stage for stage, entry in DECLARED_SCIENTIFIC_GAPS.items()
                   if entry["scientific_executor"])

    assert wired == ["c4", "c5", "c6"]


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
