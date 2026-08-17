"""The full scientific path exists, and a rehearsal can never be mistaken for it.

Two obligations, and they pull in opposite directions, which is why they are
tested together.

The first is that C4-C13 have a REAL scientific path. Before this suite existed
every one of them inherited a `run_full` that refused with
SCIENTIFIC_PATH_NOT_EXERCISED, so the rehearsal exercised code the scientific run
would never reach. A stage may still block — for missing data, a missing weight,
an incompatible GPU — but never because nobody wrote the code.

The second is that the rehearsal's own artifacts must be structurally incapable
of standing in for scientific ones. Not "we are careful not to", but "the
namespace, the eligibility flag and the lock filename all disagree, and the
resolver reads all three".

Nothing here executes scientific work. The full path is probed through dispatch
and static inspection, which is the whole point: proving the code is reachable
must not cost a GPU.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from prism_fas.pipeline.adapters import AdapterRequest  # noqa: E402
from prism_fas.pipeline.adapters.common import EngineeringAdapter  # noqa: E402
from prism_fas.pipeline.adapters.registry import build_registry  # noqa: E402
from prism_fas.pipeline.execution import ExecutionContext  # noqa: E402
from prism_fas.pipeline.profiles import load_profile  # noqa: E402

SCIENTIFIC_STAGES = ("C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13")

#: Reason codes that mean "the code is missing". None may appear for C4-C13.
IMPLEMENTATION_PLACEHOLDERS = {
    "SCIENTIFIC_PATH_NOT_EXERCISED", "NOT_IMPLEMENTED", "FULL_MODE_PLACEHOLDER",
    "TODO", "SMOKE_ONLY",
}

ADAPTER_DIR = REPO / "src" / "prism_fas" / "pipeline" / "adapters"


@pytest.fixture(scope="module")
def registry() -> dict:
    return build_registry()


# --- the full path is implemented -------------------------------------------

@pytest.mark.parametrize("stage_id", SCIENTIFIC_STAGES)
def test_every_scientific_stage_implements_the_shared_workflow(registry, stage_id) -> None:
    adapter = registry[stage_id]
    assert "workflow" in type(adapter).__dict__, (
        f"{stage_id} does not implement workflow(); it would fall through to the "
        "base class, which raises rather than pretending")


def test_the_base_class_has_no_scientific_placeholder() -> None:
    """`run_full` is gone entirely; there is nothing left to inherit."""
    assert not hasattr(EngineeringAdapter, "run_full")


@pytest.mark.parametrize("stage_id", SCIENTIFIC_STAGES)
def test_no_adapter_defines_a_placeholder_reason_code(stage_id) -> None:
    source = (ADAPTER_DIR / f"{stage_id.lower()}.py").read_text(encoding="utf-8")
    for placeholder in IMPLEMENTATION_PLACEHOLDERS:
        assert placeholder not in source, (
            f"{stage_id} still references {placeholder}")


def _live_string_constants(path: Path) -> set[str]:
    """String literals the code actually evaluates, excluding docstrings.

    A grep would also flag the comment in common.py explaining why the
    placeholder was REMOVED, which is exactly the prose worth keeping.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                docstrings.add(id(first.value))
    return {node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and id(node) not in docstrings}


def test_the_placeholder_code_is_absent_from_the_whole_adapter_package() -> None:
    offenders = {}
    for path in ADAPTER_DIR.glob("*.py"):
        hits = sorted(IMPLEMENTATION_PLACEHOLDERS & _live_string_constants(path))
        if hits:
            offenders[path.name] = hits
    assert offenders == {}, offenders


# --- the context is what separates the two paths -----------------------------

def test_only_the_full_profile_yields_a_scientific_context() -> None:
    for name in ("validate", "smoke", "rehearsal"):
        context = ExecutionContext.for_profile(load_profile(name, repo=REPO))
        assert not context.is_scientific, name
        assert context.fixtures_permitted, name
        assert not context.writes_governing_locks, name

    scientific = ExecutionContext.for_profile(load_profile("full", repo=REPO))
    assert scientific.is_scientific
    assert not scientific.fixtures_permitted
    assert scientific.writes_governing_locks


def test_a_scientific_context_never_truncates() -> None:
    """`limit` must ignore the sample size entirely under science."""
    scientific = ExecutionContext.for_profile(load_profile("full", repo=REPO))
    for declared in (1, 2, 42, 210, 2048):
        assert scientific.limit(declared, sample=2) == declared
        assert scientific.budget_or("samples", declared) == declared
    assert scientific.budget is None


def test_a_rehearsal_context_samples_and_says_so() -> None:
    rehearsal = ExecutionContext.for_profile(load_profile("rehearsal", repo=REPO))
    assert rehearsal.limit(42, sample=2) == 2
    assert rehearsal.cardinality_rule == "SAMPLED_TO_BUDGET"
    # It never samples MORE than the declared count.
    assert rehearsal.limit(1, sample=2) == 1


def test_the_request_derives_its_context_and_cannot_be_handed_a_false_one() -> None:
    """`context` is a property, so a rehearsal request cannot claim eligibility."""
    request = AdapterRequest(repo=REPO, profile=load_profile("rehearsal", repo=REPO))
    assert not request.context.is_scientific
    with pytest.raises(AttributeError):
        request.context = ExecutionContext.for_profile(load_profile("full", repo=REPO))


# --- C8: the full matrix, with no route to the sampling constant -------------

def test_c8_full_schedules_every_declared_row() -> None:
    """§8: the complete frozen matrix, never a sample."""
    from prism_fas.evaluation.source_matrix import build_plan

    plan = build_plan()
    scientific = ExecutionContext.for_profile(load_profile("full", repo=REPO))
    from prism_fas.pipeline.adapters.c8 import SMOKE_ROWS

    assert scientific.limit(len(plan.rows), sample=SMOKE_ROWS) == len(plan.rows)
    assert len(plan.rows) == 42


def test_c8_scientific_cardinality_ignores_every_sampling_mechanism() -> None:
    """Whatever the sample constant were set to, science runs the whole matrix."""
    from prism_fas.evaluation.source_matrix import build_plan

    plan = build_plan()
    scientific = ExecutionContext.for_profile(load_profile("full", repo=REPO))
    for absurd_sample in (0, 1, 2, 7, 41, 10_000):
        assert scientific.limit(len(plan.rows), sample=absurd_sample) == len(plan.rows)


def test_c8_full_does_not_read_the_pending_prefix() -> None:
    source = (ADAPTER_DIR / "c8.py").read_text(encoding="utf-8")
    assert "pending[:SMOKE_ROWS]" not in source
    assert "context.limit(len(plan.rows), sample=SMOKE_ROWS)" in source


def test_the_c8_matrix_keeps_its_declared_seed_and_protocol_identities() -> None:
    from prism_fas.evaluation.source_matrix import SEED_FAMILY, build_plan

    plan = build_plan()
    report = plan.validate()
    assert report["valid"], report["problems"]
    assert report["rows"] == 42
    assert all(row.seed in SEED_FAMILY for row in plan.rows)


# --- rehearsal artifacts cannot be scientific ancestors -----------------------

REHEARSAL_ROOTS = ("reports/rehearsal", "runs/rehearsal",
                   "reports/smoke", "runs/smoke", "reports/validate")


def test_scientific_completion_reads_only_the_full_namespace() -> None:
    """§27: no rehearsal artifact can satisfy a scientific completion check."""
    from prism_fas.pipeline import runner

    source = (REPO / "src" / "prism_fas" / "pipeline" / "runner.py").read_text(
        encoding="utf-8")
    start = source.index("def scientific_completion")
    end = source.index("def first_incomplete_stage")
    body = source[start:end]
    assert '"full"' in body
    for root in ("rehearsal", "smoke", "validate"):
        assert root not in body, (
            f"scientific_completion mentions {root}; completion must be decided "
            "from the full namespace alone")


@pytest.mark.parametrize("ancestor", [
    "C4 checkpoint", "C6 bank", "C7 selected detector config", "C8 run",
    "C9 SOURCE_MATRIX_LOCK_C",
])
def test_named_scientific_ancestors_are_required_from_the_full_namespace(
        registry, ancestor) -> None:
    """Every declared scientific input path is under reports/full or runs/full."""
    inputs = []
    for stage_id in SCIENTIFIC_STAGES:
        inputs.extend(registry[stage_id].required_inputs())
    lineage = [item for item in inputs
               if item.relative_path.startswith(("reports/", "runs/"))]
    assert lineage, "no stage declares an inherited artifact"
    for item in lineage:
        assert item.relative_path.startswith(("reports/full", "runs/full")), (
            f"{item.name} inherits from {item.relative_path}, which is not the "
            "scientific namespace")


def test_no_required_input_can_be_satisfied_by_a_rehearsal_path(registry) -> None:
    for stage_id in SCIENTIFIC_STAGES:
        for item in registry[stage_id].required_inputs():
            for root in REHEARSAL_ROOTS:
                assert not item.relative_path.startswith(root), (
                    f"{stage_id}.{item.name} would accept {root}")


def test_a_rehearsal_names_its_locks_differently() -> None:
    """A second barrier behind the namespace: the filename itself disagrees."""
    rehearsal = ExecutionContext.for_profile(load_profile("rehearsal", repo=REPO))
    scientific = ExecutionContext.for_profile(load_profile("full", repo=REPO))
    for name in ("GPAT_CONFIG_LOCK.json", "SOURCE_MATRIX_LOCK_C.json",
                 "DETECTOR_CONFIG_LOCK.json", "TARGET_PACKAGE_LOCK.json"):
        assert scientific.lock_filename(name) == name
        assert rehearsal.lock_filename(name) != name
        assert "REHEARSAL" in rehearsal.lock_filename(name)


def test_every_rehearsal_artifact_on_disk_declares_itself_ineligible() -> None:
    """Whatever the last rehearsal wrote must say what it is."""
    root = REPO / "reports" / "rehearsal"
    if not root.is_dir():
        pytest.skip("no rehearsal has run in this checkout")
    checked = 0
    for path in root.rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "scientific_eligible" not in payload:
            continue
        checked += 1
        assert payload["scientific_eligible"] is False, path
        assert payload.get("execution_profile") in ("rehearsal", "smoke"), path
    assert checked > 0, "no rehearsal artifact carried an eligibility stamp"


# --- the final report may not aggregate a rehearsal ---------------------------

def test_the_report_generator_is_given_one_namespace_at_a_time() -> None:
    """§26: reports/full must not be able to absorb reports/rehearsal."""
    path = REPO / "src" / "prism_fas" / "reporting" / "__init__.py"
    assert "reports_root" in path.read_text(encoding="utf-8")
    # The roots come from the profile. Nothing the reporting layer EVALUATES may
    # name a second namespace to merge in; the docstring may of course discuss
    # the rehearsal, which is why this reads constants rather than grepping.
    live = _live_string_constants(path)
    for root in ("reports/rehearsal", "runs/rehearsal", "reports/smoke",
                 "rehearsal", "smoke"):
        assert root not in live, (
            f"the reporting layer evaluates the literal {root!r}; it must read "
            "only the namespace it was given")


def test_reporting_marks_a_rehearsal_bundle_as_not_scientifically_eligible() -> None:
    summary = REPO / "reports" / "rehearsal" / "final" / "REPORTING_SUMMARY.json"
    if not summary.is_file():
        pytest.skip("no rehearsal reporting bundle in this checkout")
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["execution_intent"] == "CPU_FULL_REHEARSAL"
    bundle = REPO / "reports" / "rehearsal" / "final" / "FINAL_BUNDLE_MANIFEST.json"
    assert json.loads(bundle.read_text(encoding="utf-8"))["scientific_eligible"] is False
