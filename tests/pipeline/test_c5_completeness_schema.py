"""The completeness contract, and the production method that consumes it.

The C5 stage-ownership correction replaced a single `failed` count with three
that mean different things — `generated`, `semantic_failed` and
`runtime_unresolved`. `_finalize_c5` was updated; `_verify_raw_candidates` was
not, and kept reading `state["failed"]`.

Nothing caught it. Every existing test drove `completeness()` directly or drove
the adapter under a profile that blocks earlier, so the one line that reads the
dictionary was never executed. That is the gap this file closes: it calls
`_verify_raw_candidates` itself, over a pool shaped like the real GPU outcome —
6144 planned, 6082 generated, 62 terminal semantic failures — and asserts the
production checks come out right rather than merely that no exception was raised.

Nothing is rendered, retried or replaced. The pool is written as records on disk
and read back.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.pipeline.adapters import AdapterRequest  # noqa: E402
from prism_fas.pipeline.adapters.c5 import C5Adapter  # noqa: E402
from prism_fas.pipeline.profiles import load_profile  # noqa: E402
from prism_fas.synthesis import c5_raw_generation as raw  # noqa: E402
from prism_fas.synthesis import c5_render as render_module  # noqa: E402
from prism_fas.synthesis.c5_render import SemanticGenerationFailure  # noqa: E402
from prism_fas.synthesis.c5_source_pair_plan import ARMS, GPAT, PHYSICS  # noqa: E402

#: The real first scientific run, per arm: Physics slots that failed.
OBSERVED_PHYSICS_FAILURES = {"DET": 28, "LLM": 14, "RND": 20}
SEMANTIC_REASON = ("the artifact did not survive uint8 quantization and "
                   "finalized to an empty exact mask over 0 requested support pixels")


def _arm_plan(arm: str, per_route: int) -> dict[str, Any]:
    rows = []
    position = 0
    for route in (PHYSICS, GPAT):
        for index in range(per_route):
            rows.append({
                "candidate_id": f"c5syn_{arm.lower()}_{position:05d}", "arm": arm,
                "route": route, "position": position,
                "recipe_id": f"r{index % 256}", "recipe_ordinal": index % 256,
                "slot": (index % 8) + 1, "domain_relation": "same_domain",
                "live_dataset": "casia_fasd" if index % 2 == 0 else "msu_mfsd",
                "live_target_sample_id": f"live_{index:05d}",
                "spoof_source_sample_id": None if route == PHYSICS else f"spoof_{index:05d}",
                "recipe_bank_identity": f"bank-{arm}",
                "generator_binding": "m7-physics-v1" if route == PHYSICS else "c" * 64,
            })
            position += 1
    return {"arm": arm, "arm_plan_identity": f"armplan-{arm}",
            "source_pair_plan_identity": "baseplan", "package_identity": "b" * 64,
            "recipe_bank_identity": f"bank-{arm}", "ontology_identity": "onto",
            "planned_candidates": len(rows), "binds_quality_calibration": False,
            "candidates": rows}


def _build_pool(root: Path, per_route: int,
                physics_failures: dict[str, int]) -> dict[str, dict[str, Any]]:
    """A terminal pool: some Physics slots end as retained semantic failures."""
    import hashlib

    plans = {arm: _arm_plan(arm, per_route) for arm in ARMS}
    for arm, plan in plans.items():
        budget = physics_failures.get(arm, 0)
        for row in plan["candidates"]:
            identity = render_module.identity_for(row, plan)
            directory = raw.candidate_dir(root, arm, identity.candidate_id)
            directory.mkdir(parents=True, exist_ok=True)
            if row["route"] == PHYSICS and budget > 0:
                budget -= 1
                record = raw.failure_record(
                    identity, stage="render_physics",
                    error=SemanticGenerationFailure(SEMANTIC_REASON))
            else:
                hashes = {}
                for index, name in enumerate(raw.PAYLOAD_NAMES):
                    payload = f"{identity.candidate_id}:{index}".encode()
                    (directory / name).write_bytes(payload)
                    hashes[name] = hashlib.sha256(payload).hexdigest()
                record = raw.CandidateRecord(identity=identity, status=raw.GENERATED,
                                             payload_sha256=hashes)
            (directory / raw.RECORD_NAME).write_text(json.dumps(record.as_dict()),
                                                     encoding="utf-8")
    return plans


def _run_verify(root: Path, plans: dict[str, dict[str, Any]]
                ) -> tuple[list[dict[str, Any]], Any, Path]:
    """Drive the production `_verify_raw_candidates` exactly as C5 does.

    `runs` is passed explicitly so the adapter's own `_scientific_work_root`
    resolves to the pool built above — the method is called, not reimplemented.
    """
    request = AdapterRequest(repo=root, profile=load_profile("full", repo=REPO))
    reports = root / "reports" / "full" / "c5"
    reports.mkdir(parents=True, exist_ok=True)
    runs = root / "runs" / "full" / "c5"
    records, result = C5Adapter()._verify_raw_candidates(
        request, plans, {}, reports, runs)
    return records, result, reports


@pytest.fixture(scope="module")
def observed(tmp_path_factory) -> tuple[Path, dict[str, dict[str, Any]], Any]:
    """The real GPU shape: 6144 planned, 6082 generated, 62 semantic failures."""
    root = tmp_path_factory.mktemp("gpu")
    plans = _build_pool(root / "runs" / "full" / "c5" / "scientific" / "candidates",
                        1024, OBSERVED_PHYSICS_FAILURES)
    records, result, reports = _run_verify(root, plans)
    return root, plans, (records, result, reports)


# --- the defect ---------------------------------------------------------------

def test_verify_raw_candidates_survives_the_real_pool_shape(observed) -> None:
    """The exact call that raised `KeyError: 'failed'` on the GPU host."""
    _, _, (records, result, _) = observed

    assert len(records) == 6144
    assert result.mode == "VERIFY_RAW_CANDIDATES"
    assert [item["check_id"] for item in result.checks if not item["ok"]] == []


def test_the_retention_check_reports_the_semantic_failure_count(observed) -> None:
    _, _, (_, result, _) = observed
    retention = next(item for item in result.checks
                     if item["check_id"] == "c5_failures_are_retained_not_replaced")

    detail = retention["detail"]
    assert retention["ok"] is True
    assert detail["semantic_failed"] == 62
    assert detail["generated"] == 6082
    assert detail["runtime_unresolved"] == 0
    assert "62 terminal semantic failure(s) retained" in retention["summary"]
    assert "failed" not in detail, "the collapsed count is not reported either"


def test_the_terminal_check_sees_a_complete_pool(observed) -> None:
    _, _, (_, result, _) = observed
    terminal = next(item for item in result.checks
                    if item["check_id"] == "c5_every_planned_candidate_is_terminal")

    detail = terminal["detail"]
    assert terminal["ok"] is True
    assert detail["planned"] == 6144 and detail["terminal"] == 6144
    assert detail["missing"] == 0


def test_the_audit_artifact_records_the_three_counts(observed) -> None:
    root, _, (_, _, reports) = observed
    payload = json.loads((reports / "C5_RAW_CANDIDATE_AUDIT.json")
                         .read_text(encoding="utf-8"))

    assert payload["generated"] == 6082
    assert payload["semantic_failed"] == 62
    assert payload["runtime_unresolved"] == 0
    assert payload["planned"] == payload["terminal"] == 6144
    assert "failed" not in payload, (
        "the collapsed count is gone; three distinct counts replace it")


def test_no_candidate_was_modified_retried_or_replaced(observed) -> None:
    root, plans, (records, _, _) = observed
    pool = root / "runs" / "full" / "c5" / "scientific" / "candidates"

    on_disk = [path.parent.name for arm in ARMS
               for path in (pool / arm).glob("*/CANDIDATE.json")]
    planned = [row["candidate_id"] for plan in plans.values()
               for row in plan["candidates"]]

    assert sorted(on_disk) == sorted(planned), "the pool is exactly the schedule"
    assert len(on_disk) == 6144, "nothing was added"
    for record in records:
        if record["status"] == raw.FAILED_GENERATION:
            assert record["failure"]["replacement_generated"] is False
            assert record["failure"]["error_type"] == "SemanticGenerationFailure"
            assert record["payload_sha256"] == {}


# --- the schema itself --------------------------------------------------------

def test_completeness_has_no_legacy_failed_key() -> None:
    """Deliberately absent. An alias would hide the next drift instead of
    surfacing it, and `failed` no longer has one unambiguous meaning."""
    state = render_module.completeness({}, [])

    assert "failed" not in state
    assert {"generated", "semantic_failed", "runtime_unresolved"} <= set(state)


def test_the_three_counts_mean_different_things() -> None:
    keys = set(render_module.completeness({}, []))

    for key in ("planned", "terminal", "generated", "semantic_failed",
                "runtime_unresolved", "missing", "missing_candidate_ids",
                "per_arm", "usable_generated_by_arm_and_route",
                "every_planned_candidate_is_terminal",
                "every_planned_candidate_is_usable", "failed_candidate_ids", "rule"):
        assert key in keys, key


def test_no_c5_consumer_reads_a_stale_completeness_key() -> None:
    """The standing guard: every `state[...]` access must name a real field.

    `state` is the completeness result in both consumers, so this catches the
    next rename the same way it would have caught this one.
    """
    import ast

    source = (REPO / "src" / "prism_fas" / "pipeline" / "adapters" / "c5.py"
              ).read_text(encoding="utf-8")
    available = set(render_module.completeness({}, []))

    read: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == "state"
                and isinstance(node.slice, ast.Constant)):
            read.add(node.slice.value)

    assert read, "the consumers really do read the completeness state"
    assert read <= available, f"stale completeness keys: {sorted(read - available)}"


def test_the_render_arm_per_pass_count_is_a_different_field() -> None:
    """`outcome["failed"]` is render_arm's own per-pass counter and stays.

    It answers "how many candidates failed during THIS pass", which is not the
    same question as "how many terminal semantic failures does the pool hold" —
    a resumed pass reports 0 rendered and still inherits every earlier failure.
    """
    outcome_keys = {"arm", "planned", "attempted", "records", "reused", "rendered",
                    "rebuilt", "failed", "record_set_digest", "payload_set_digest",
                    "summary"}
    source = (REPO / "src" / "prism_fas" / "synthesis" / "c5_render.py"
              ).read_text(encoding="utf-8")

    assert '"failed": failed' in source
    assert outcome_keys - set(render_module.completeness({}, [])) >= {"failed"}


def test_raw_summarize_keeps_its_historical_presentation_field() -> None:
    """`summarize` is a separate canonical API and was not renamed."""
    summary = raw.summarize([])

    assert "failed" in summary and "generated" in summary
    assert "semantic_failed" not in summary
