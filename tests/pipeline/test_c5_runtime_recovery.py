"""C5_RUNTIME_RECOVERY_V1: what a failure during rendering costs.

The policy separates three things that had been one:

* a DETERMINISTIC CANDIDATE-SEMANTIC failure — the artifact finalizes to an
  empty exact mask — is terminal. The candidate cannot exist, a rerun would
  reach the same answer, and the arm is short by one forever.
* a PROCESS INTERRUPTION is not an outcome at all. It propagates untouched and
  leaves the candidate unresolved.
* EVERYTHING ELSE — CUDA, OOM, filesystem, codec, an unexpected
  `SyntheticBankError` — says nothing about the candidate. It is recorded as
  operational provenance, the pass aborts, and the next invocation retries the
  identical candidate. That is recovery-ladder L1.

The distinction is worth the machinery because a terminal failure is permanent:
under the C5 completion contract one candidate lost to a CUDA hiccup fails the
whole 2048-candidate arm, with no way back short of deleting evidence.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.synthesis import c5_raw_generation as raw  # noqa: E402
from prism_fas.synthesis import c5_render as render_module  # noqa: E402
from prism_fas.synthesis.c5_render import (RuntimeAttemptFailure,  # noqa: E402
                                           SemanticGenerationFailure)
from prism_fas.synthesis.c5_source_pair_plan import GPAT, PHYSICS  # noqa: E402
from prism_fas.synthesis.synthetic_bank import SyntheticBankError  # noqa: E402

RENDER_SOURCE = (REPO / "src" / "prism_fas" / "synthesis" / "c5_render.py"
                 ).read_text(encoding="utf-8")
SIZE = 224


def _method_source(name: str, source: str = RENDER_SOURCE) -> str:
    tree = ast.parse(source)
    node = next(item for item in ast.walk(tree)
                if isinstance(item, ast.FunctionDef) and item.name == name)
    return ast.get_source_segment(source, node) or ""


# --- a store and a route that can fail in any way we name --------------------

class _Store:
    def load(self, sample_id: str) -> tuple[np.ndarray, dict[str, Any]]:
        rng = np.random.default_rng(abs(hash(sample_id)) % (2 ** 32))
        return rng.random((3, SIZE, SIZE), dtype=np.float32), {}


class _Route:
    """Renders a real, finalizable result unless told to fail in some way."""

    def __init__(self, binding: str, *, raises: dict[str, BaseException] | None = None,
                 empty_on: set[str] | None = None) -> None:
        self.binding = binding
        self.raises = dict(raises or {})
        self.empty_on = empty_on or set()
        self.rendered: list[str] = []

    def generate(self, store: Any, bank: Any, row: dict[str, Any]) -> Any:
        from prism_fas.synthesis.synthetic_bank import RouteOutput

        candidate = row["candidate_id"]
        self.rendered.append(candidate)
        if candidate in self.raises:
            raise self.raises[candidate]
        image, _ = store.load(row["live_target_sample_id"])
        support = np.zeros((SIZE, SIZE), dtype=bool)
        if candidate not in self.empty_on:
            support[20:60, 20:60] = True
        edited = image.copy()
        edited[:, 20:60, 20:60] = np.clip(edited[:, 20:60, 20:60] + 0.5, 0.0, 1.0)
        artifact = np.zeros((1, SIZE, SIZE), dtype=np.float32)
        artifact[0, 20:60, 20:60] = 0.4
        return RouteOutput(image=edited, artifact_map=artifact,
                           requested_support=support,
                           requested_region_pixels=int(support.sum()),
                           requested_coverage=0.1, achieved_coverage=0.1,
                           binding=self.binding, trace={"engine": self.binding})


def _plan(count: int = 6, arm: str = "RND") -> dict[str, Any]:
    return {
        "arm": arm, "arm_plan_identity": f"armplan-{arm}",
        "source_pair_plan_identity": "baseplan", "package_identity": "b" * 64,
        "recipe_bank_identity": f"bank-{arm}", "ontology_identity": "onto",
        "planned_candidates": count, "binds_quality_calibration": False,
        "candidates": [{
            "candidate_id": f"c5syn_{arm.lower()}_{index:04d}", "arm": arm,
            "recipe_id": f"r{index}", "recipe_ordinal": index,
            "slot": (index % 8) + 1, "position": index,
            "route": PHYSICS if index % 2 == 0 else GPAT,
            "domain_relation": "same_domain",
            "live_target_sample_id": f"live_{index:03d}",
            "spoof_source_sample_id": None if index % 2 == 0 else f"spoof_{index:03d}",
            "recipe_bank_identity": f"bank-{arm}",
            "generator_binding": "physics-v1" if index % 2 == 0 else "c" * 64,
        } for index in range(count)],
    }


def _routes(**kwargs) -> dict[str, Any]:
    return {PHYSICS: _Route("physics-v1", **kwargs), GPAT: _Route("c" * 64, **kwargs)}


def _render(work: Path, plan: dict[str, Any], routes: dict[str, Any]) -> dict[str, Any]:
    return render_module.render_arm(work_root=work, plan=plan, store=_Store(),
                                    bank={}, routes=routes)


def _record(work: Path, candidate: str, arm: str = "RND") -> dict[str, Any] | None:
    return raw.read_record(raw.candidate_dir(work, arm, candidate) / raw.RECORD_NAME)


def _attempts(work: Path, candidate: str, arm: str = "RND") -> list[dict[str, Any]]:
    return raw.runtime_attempts(raw.candidate_dir(work, arm, candidate))


# --- 1-3. the deterministic semantic failure is terminal ---------------------

def test_the_empty_exact_mask_raises_the_semantic_type(tmp_path: Path) -> None:
    plan = _plan(count=2)
    doomed = plan["candidates"][0]["candidate_id"]
    route = _Route("physics-v1", empty_on={doomed})

    with pytest.raises(SemanticGenerationFailure, match="empty exact mask"):
        render_module.render_one(_Store(), {}, route, plan["candidates"][0])


def test_a_semantic_failure_writes_a_terminal_candidate_record(tmp_path: Path) -> None:
    plan = _plan()
    doomed = plan["candidates"][0]["candidate_id"]

    outcome = _render(tmp_path, plan, _routes(empty_on={doomed}))
    record = _record(tmp_path, doomed)

    assert record["status"] == raw.FAILED_GENERATION
    assert record["failure"]["error_type"] == "SemanticGenerationFailure"
    assert record["failure"]["deterministic_candidate_semantic"] is True
    assert record["failure"]["replacement_generated"] is False
    assert outcome["failed"] == 1 and outcome["rendered"] == 5
    assert _attempts(tmp_path, doomed) == [], "a semantic failure is not an attempt"


def test_a_semantic_failure_is_retained_across_reruns(tmp_path: Path) -> None:
    plan = _plan()
    doomed = plan["candidates"][0]["candidate_id"]
    _render(tmp_path, plan, _routes(empty_on={doomed}))
    before = _record(tmp_path, doomed)

    routes = _routes()                       # a pass where it WOULD now succeed
    second = _render(tmp_path, plan, routes)

    assert doomed not in routes[PHYSICS].rendered + routes[GPAT].rendered
    assert _record(tmp_path, doomed) == before
    assert second["failed"] == 1 and second["attempted"] == 6


# --- 4-10. every other exception is runtime-incomplete -----------------------

def _cuda_error() -> RuntimeError:
    return RuntimeError("CUDA error: an illegal memory access was encountered")


def _oom_error() -> BaseException:
    """`torch.cuda.OutOfMemoryError` when the import is safe, else its base."""
    import torch

    return getattr(torch.cuda, "OutOfMemoryError", torch.cuda.CudaError if
                   hasattr(torch.cuda, "CudaError") else RuntimeError)(
        "CUDA out of memory. Tried to allocate 2.00 GiB")


RUNTIME_ERRORS = {
    "RuntimeError": RuntimeError("something went wrong mid-render"),
    "cuda": _cuda_error(),
    "OSError": OSError(28, "No space left on device"),
    "codec": ValueError("cannot identify image file"),
    "SyntheticBankError": SyntheticBankError("generated image is not finite"),
}


@pytest.mark.parametrize("name", sorted(RUNTIME_ERRORS))
def test_a_runtime_error_never_becomes_a_candidate_outcome(tmp_path: Path, name) -> None:
    plan = _plan()
    doomed = plan["candidates"][2]["candidate_id"]

    with pytest.raises(RuntimeAttemptFailure) as raised:
        _render(tmp_path, plan, _routes(raises={doomed: RUNTIME_ERRORS[name]}))

    assert _record(tmp_path, doomed) is None, "no CANDIDATE.json was written"
    assert raised.value.candidate_id == doomed
    assert raised.value.error_type == type(RUNTIME_ERRORS[name]).__name__
    assert raised.value.as_dict()["candidate_consumed"] is False


def test_a_torch_cuda_out_of_memory_error_is_runtime_incomplete(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    plan = _plan()
    doomed = plan["candidates"][2]["candidate_id"]

    with pytest.raises(RuntimeAttemptFailure):
        _render(tmp_path, plan, _routes(raises={doomed: _oom_error()}))

    assert _record(tmp_path, doomed) is None
    assert _attempts(tmp_path, doomed)[0]["outcome"] == raw.RUNTIME_ATTEMPT_FAILURE


def test_a_generic_synthetic_bank_error_is_not_treated_as_semantic(tmp_path: Path) -> None:
    """It arises inside the frozen finalizer, which is not proof of determinism."""
    plan = _plan()
    doomed = plan["candidates"][2]["candidate_id"]

    with pytest.raises(RuntimeAttemptFailure):
        _render(tmp_path, plan,
                _routes(raises={doomed: SyntheticBankError("not finite")}))

    record = _record(tmp_path, doomed)
    assert record is None, "a SyntheticBankError must not spend a candidate"
    assert not isinstance(SyntheticBankError("x"), SemanticGenerationFailure)


def test_a_runtime_error_appends_attempt_provenance(tmp_path: Path) -> None:
    plan = _plan()
    doomed = plan["candidates"][2]["candidate_id"]

    with pytest.raises(RuntimeAttemptFailure):
        _render(tmp_path, plan, _routes(raises={doomed: _cuda_error()}))

    attempts = _attempts(tmp_path, doomed)
    assert len(attempts) == 1
    attempt = attempts[0]
    assert attempt["outcome"] == raw.RUNTIME_ATTEMPT_FAILURE == "runtime_incomplete"
    assert attempt["terminal"] is False and attempt["candidate_consumed"] is False
    assert attempt["candidate_id"] == doomed
    assert attempt["arm"] == "RND" and attempt["position"] == 2
    assert attempt["route"] == PHYSICS
    assert attempt["attempt_ordinal"] == 1
    assert attempt["error_type"] == "RuntimeError"
    assert attempt["recorded_at_utc"]
    assert len(attempt["generation_identity_sha256"]) == 64


def test_the_attempt_lives_beside_the_candidate_and_is_not_candidate_json(tmp_path: Path) -> None:
    plan = _plan()
    doomed = plan["candidates"][2]["candidate_id"]

    with pytest.raises(RuntimeAttemptFailure):
        _render(tmp_path, plan, _routes(raises={doomed: _cuda_error()}))

    directory = raw.candidate_dir(tmp_path, "RND", doomed)
    attempt_dir = raw.runtime_attempt_dir(directory)
    assert sorted(item.name for item in attempt_dir.iterdir()) == [
        "RUNTIME_ATTEMPT_0001.json"]
    assert not (directory / raw.RECORD_NAME).exists()

    # It carries no field the candidate layer reads as an outcome, so it can
    # never be mistaken for one however it is loaded.
    attempt = json.loads(
        (attempt_dir / "RUNTIME_ATTEMPT_0001.json").read_text(encoding="utf-8"))
    assert attempt["schema_version"] != raw.SCHEMA_VERSION
    assert "status" not in attempt and "payload_sha256" not in attempt
    assert attempt["outcome"] not in (raw.GENERATED, raw.FAILED_GENERATION)


def test_a_host_path_never_reaches_an_attempt_record(tmp_path: Path) -> None:
    plan = _plan()
    doomed = plan["candidates"][2]["candidate_id"]

    with pytest.raises(RuntimeAttemptFailure):
        _render(tmp_path, plan,
                _routes(raises={doomed: OSError(r"cannot open D:\runs\full\c5\x.png")}))

    reason = _attempts(tmp_path, doomed)[0]["sanitized_reason"]
    assert "[redacted-path]" in reason and "D:\\runs" not in reason


# --- 6, 21. the pass aborts, and nothing after it runs -----------------------

def test_the_pass_aborts_immediately_and_attempts_no_later_candidate(tmp_path: Path) -> None:
    plan = _plan()
    doomed = plan["candidates"][2]["candidate_id"]
    routes = _routes(raises={doomed: _cuda_error()})

    with pytest.raises(RuntimeAttemptFailure):
        _render(tmp_path, plan, routes)

    # The two routes keep separate logs, so "last" is decided by plan position
    # rather than by concatenation order.
    attempted = set(routes[PHYSICS].rendered) | set(routes[GPAT].rendered)
    assert doomed in attempted
    assert attempted == {row["candidate_id"] for row in plan["candidates"][:3]}, (
        "exactly the candidates up to and including the failing one were tried")
    for later in plan["candidates"][3:]:
        assert later["candidate_id"] not in attempted
        assert _record(tmp_path, later["candidate_id"]) is None


def test_no_automatic_retry_happens_inside_one_invocation(tmp_path: Path) -> None:
    plan = _plan()
    doomed = plan["candidates"][2]["candidate_id"]
    routes = _routes(raises={doomed: _cuda_error()})

    with pytest.raises(RuntimeAttemptFailure):
        _render(tmp_path, plan, routes)

    tried = (routes[PHYSICS].rendered + routes[GPAT].rendered).count(doomed)
    assert tried == 1, "the retry is the next train.py run, not a loop in here"
    assert len(_attempts(tmp_path, doomed)) == 1


def test_the_loop_has_the_three_branches_in_the_frozen_order() -> None:
    source = _method_source("render_arm")
    semantic = source.index("except SemanticGenerationFailure")
    interrupt = source.index("except (KeyboardInterrupt, SystemExit)")
    generic = source.index("except Exception")

    assert semantic < interrupt < generic, (
        "a bare `except Exception` placed first would swallow both of the others")
    assert "except BaseException" not in source
    assert "raise RuntimeAttemptFailure" in source


# --- 11-13. interruption is not an outcome -----------------------------------

@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_a_process_interruption_propagates_untouched(tmp_path: Path, interruption) -> None:
    plan = _plan()
    doomed = plan["candidates"][2]["candidate_id"]

    with pytest.raises(interruption):
        _render(tmp_path, plan, _routes(raises={doomed: interruption()}))

    assert _record(tmp_path, doomed) is None, "no terminal record"
    assert _attempts(tmp_path, doomed) == [], "not even a runtime attempt"


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_an_interruption_preserves_the_candidates_already_finished(
        tmp_path: Path, interruption) -> None:
    plan = _plan()
    doomed = plan["candidates"][2]["candidate_id"]

    with pytest.raises(interruption):
        _render(tmp_path, plan, _routes(raises={doomed: interruption()}))

    for done in plan["candidates"][:2]:
        assert _record(tmp_path, done["candidate_id"])["status"] == raw.GENERATED

    resumed = _render(tmp_path, plan, _routes())
    assert resumed["reused"] == 2 and resumed["rendered"] == 4
    assert resumed["failed"] == 0


# --- 14-17. the rerun retries THIS candidate ---------------------------------

def test_the_rerun_retries_the_identical_candidate_id(tmp_path: Path) -> None:
    plan = _plan()
    doomed = plan["candidates"][2]["candidate_id"]
    with pytest.raises(RuntimeAttemptFailure):
        _render(tmp_path, plan, _routes(raises={doomed: _cuda_error()}))

    routes = _routes()
    resumed = _render(tmp_path, plan, routes)

    assert doomed in routes[PHYSICS].rendered + routes[GPAT].rendered
    assert resumed["rendered"] == 4 and resumed["reused"] == 2
    assert resumed["failed"] == 0
    assert _record(tmp_path, doomed)["status"] == raw.GENERATED


def test_the_retried_candidate_keeps_its_exact_generation_identity(tmp_path: Path) -> None:
    plan = _plan()
    doomed = plan["candidates"][2]["candidate_id"]
    expected = render_module.identity_for(plan["candidates"][2], plan)
    with pytest.raises(RuntimeAttemptFailure):
        _render(tmp_path, plan, _routes(raises={doomed: _cuda_error()}))
    recorded_before = _attempts(tmp_path, doomed)[0]["generation_identity_sha256"]

    _render(tmp_path, plan, _routes())
    record = _record(tmp_path, doomed)

    assert record["generation_identity_sha256"] == expected.digest()
    assert record["generation_identity_sha256"] == recorded_before, (
        "the identity that failed and the identity that succeeded are one candidate")


def test_completed_candidates_are_reused_rather_than_re_rendered(tmp_path: Path) -> None:
    plan = _plan()
    doomed = plan["candidates"][4]["candidate_id"]
    with pytest.raises(RuntimeAttemptFailure):
        _render(tmp_path, plan, _routes(raises={doomed: _cuda_error()}))

    routes = _routes()
    resumed = _render(tmp_path, plan, routes)

    for done in plan["candidates"][:4]:
        assert done["candidate_id"] not in routes[PHYSICS].rendered
        assert done["candidate_id"] not in routes[GPAT].rendered
    assert resumed["reused"] == 4


def test_the_successful_retry_produces_the_same_bytes_as_an_undisturbed_render(
        tmp_path: Path) -> None:
    """A runtime fault must not change what the candidate IS."""
    plan = _plan(count=2)
    doomed = plan["candidates"][0]["candidate_id"]

    clean = tmp_path / "clean"
    _render(clean, plan, _routes())
    undisturbed = _record(clean, doomed)

    disturbed = tmp_path / "disturbed"
    with pytest.raises(RuntimeAttemptFailure):
        _render(disturbed, plan, _routes(raises={doomed: _cuda_error()}))
    _render(disturbed, plan, _routes())
    retried = _record(disturbed, doomed)

    assert retried["payload_sha256"] == undisturbed["payload_sha256"]
    assert retried["generation_identity_sha256"] == undisturbed["generation_identity_sha256"]


# --- 18, 19. repeated failure accumulates and changes no identity ------------

def test_repeated_runtime_failures_accumulate_and_stay_non_terminal(tmp_path: Path) -> None:
    plan = _plan()
    doomed = plan["candidates"][2]["candidate_id"]

    for _ in range(3):
        with pytest.raises(RuntimeAttemptFailure):
            _render(tmp_path, plan, _routes(raises={doomed: _cuda_error()}))

    attempts = _attempts(tmp_path, doomed)
    assert [item["attempt_ordinal"] for item in attempts] == [1, 2, 3]
    assert all(item["terminal"] is False for item in attempts)
    assert _record(tmp_path, doomed) is None, (
        "repetition is an L0 diagnostic signal, never a promotion to terminal")


def test_operational_attempt_fields_do_not_touch_the_candidate_identity(
        tmp_path: Path) -> None:
    plan = _plan(count=2)
    doomed = plan["candidates"][0]["candidate_id"]
    expected = render_module.identity_for(plan["candidates"][0], plan).digest()

    for _ in range(2):
        with pytest.raises(RuntimeAttemptFailure):
            _render(tmp_path, plan, _routes(raises={doomed: _cuda_error()}))
    _render(tmp_path, plan, _routes())

    record = _record(tmp_path, doomed)
    assert record["generation_identity_sha256"] == expected
    assert "recorded_at_utc" not in json.dumps(record["generation_identity"])
    assert "attempt_ordinal" not in json.dumps(record["generation_identity"])
    # The count is kept as trace, which is provenance and not identity.
    assert record["trace"]["prior_runtime_attempts"] == 2


# --- 20. orphan payloads are not completion evidence -------------------------

def test_orphan_payloads_without_a_record_are_rebuilt_not_reused(tmp_path: Path) -> None:
    plan = _plan(count=2)
    doomed = plan["candidates"][0]["candidate_id"]
    directory = raw.candidate_dir(tmp_path, "RND", doomed)
    directory.mkdir(parents=True)
    for name in raw.PAYLOAD_NAMES:                       # a half-written candidate
        (directory / name).write_bytes(b"orphan")

    routes = _routes()
    outcome = _render(tmp_path, plan, routes)

    assert doomed in routes[PHYSICS].rendered, "the orphan proved nothing"
    assert outcome["rendered"] == 2
    record = _record(tmp_path, doomed)
    assert record["status"] == raw.GENERATED
    for name in raw.PAYLOAD_NAMES:
        assert (directory / name).read_bytes() != b"orphan"
        assert raw.sha256_file(directory / name) == record["payload_sha256"][name]


def test_an_attempt_record_alone_never_makes_a_candidate_terminal(tmp_path: Path) -> None:
    plan = _plan(count=2)
    doomed = plan["candidates"][0]["candidate_id"]
    identity = render_module.identity_for(plan["candidates"][0], plan)
    directory = raw.candidate_dir(tmp_path, "RND", doomed)
    raw.append_runtime_attempt(directory, identity, stage="render_physics",
                               error=_cuda_error())

    decision = raw.reuse_decision(directory, identity)
    plans = {"RND": plan}
    state = render_module.completeness(
        plans, render_module.collect_records(tmp_path, plans))

    assert decision["reusable"] is False and decision["reason"] == "ABSENT"
    assert state["every_planned_candidate_is_terminal"] is False
    assert state["missing"] == 2


# --- 22-23. a partial run cannot complete C5, and C6 stays blocked -----------

def test_a_runtime_aborted_pass_cannot_produce_a_completion_lock() -> None:
    from prism_fas.pipeline.adapters import c5 as c5_adapter

    source = ast.get_source_segment(
        (REPO / "src" / "prism_fas" / "pipeline" / "adapters" / "c5.py"
         ).read_text(encoding="utf-8"),
        next(node for node in ast.walk(ast.parse(
            (REPO / "src" / "prism_fas" / "pipeline" / "adapters" / "c5.py"
             ).read_text(encoding="utf-8")))
            if isinstance(node, ast.FunctionDef)
            and node.name == "_render_candidates")) or ""

    assert "except render_module.RuntimeAttemptFailure" in source
    assert "return None, self.result(" in source
    assert "C5_RENDER_INCOMPLETE.json" in source
    assert c5_adapter.C5Adapter.SCIENTIFIC_LOCK not in source

    workflow = ast.get_source_segment(
        (REPO / "src" / "prism_fas" / "pipeline" / "adapters" / "c5.py"
         ).read_text(encoding="utf-8"),
        next(node for node in ast.walk(ast.parse(
            (REPO / "src" / "prism_fas" / "pipeline" / "adapters" / "c5.py"
             ).read_text(encoding="utf-8")))
            if isinstance(node, ast.FunctionDef)
            and node.name == "_scientific_workflow")) or ""
    assert "if rendered is None:" in workflow
    assert workflow.index("if rendered is None:") < workflow.index("_finalize_c5")


def test_an_incomplete_pass_leaves_c6_blocked(tmp_path: Path) -> None:
    from prism_fas.pipeline.adapters import AdapterRequest
    from prism_fas.pipeline.adapters.c6 import C6Adapter
    from prism_fas.pipeline.profiles import load_profile

    request = AdapterRequest(repo=tmp_path, profile=load_profile("full", repo=REPO))
    gate = C6Adapter().full_precondition_gate(request)

    assert gate is not None and gate.status == "BLOCKED"
    assert gate.status_axes.scientific == "BLOCKED"
    assert "c5_synthesis_verified" in gate.summary


# --- 24, 25. the firewall and the frozen repository --------------------------

def test_the_recovery_path_resolves_no_target_artifact() -> None:
    body = "\n".join(_method_source(name) for name in
                     ("render_arm", "render_one"))
    for forbidden in ("siw", "SiW", "target_test", "label_live_spoof", "_real_target"):
        assert forbidden not in body, forbidden


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
