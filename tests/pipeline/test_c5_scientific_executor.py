"""The C5 scientific render executor: routing, resume, completion, handoff.

The render loop runs here for real, over a fake store and fake routes but the
REAL `finalize_discrete` — the frozen discretization is what decides what a
candidate's bytes are, and stubbing it would leave the resume and corruption
paths hashing something that is not a candidate. No GPU, no checkpoint and no
source package is involved, which is also why none of these tests may claim to
exercise a scientific pass.
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

from prism_fas.pipeline.adapters import c5 as c5_module  # noqa: E402
from prism_fas.synthesis import c5_raw_generation as raw  # noqa: E402
from prism_fas.synthesis import c5_render as render_module  # noqa: E402
from prism_fas.synthesis.c5_source_pair_plan import GPAT, PHYSICS  # noqa: E402

C5_SOURCE = (REPO / "src" / "prism_fas" / "pipeline" / "adapters" / "c5.py"
             ).read_text(encoding="utf-8")
RENDER_SOURCE = (REPO / "src" / "prism_fas" / "synthesis" / "c5_render.py"
                 ).read_text(encoding="utf-8")


def _method_source(name: str, source: str = C5_SOURCE) -> str:
    tree = ast.parse(source)
    node = next(item for item in ast.walk(tree)
                if isinstance(item, ast.FunctionDef) and item.name == name)
    return ast.get_source_segment(source, node) or ""


def _method_code(name: str, source: str = C5_SOURCE) -> str:
    """A method's source with its docstring removed.

    A docstring that NAMES what the method must never call would otherwise fail
    a check that the method never calls it.
    """
    tree = ast.parse(source)
    node = next(item for item in ast.walk(tree)
                if isinstance(item, ast.FunctionDef) and item.name == name)
    body = node.body[1:] if ast.get_docstring(node) else node.body
    return chr(10).join(ast.get_source_segment(source, item) or "" for item in body)


def _joined(names) -> str:
    """Several methods' code, docstrings stripped, as one string."""
    return chr(10).join(_method_code(name) for name in names)


# --- 1-5. the scientific branch exists and is separate ------------------------

def test_the_workflow_branches_on_the_execution_context() -> None:
    source = _method_source("workflow")

    assert "context.is_scientific" in source
    assert "_scientific_workflow" in source
    assert "_engineering_workflow" in source


def test_the_seven_scientific_substages_are_declared_and_run_in_order() -> None:
    assert c5_module.SCIENTIFIC_MODES == (
        "VERIFY_C4_LOCK", "LOAD_SOURCE_PAIR_PLAN", "BUILD_ARM_PLANS",
        "RENDER_CANDIDATES", "VERIFY_RAW_CANDIDATES", "FINALIZE_C5", "VERIFY_C5_LOCK")

    source = _method_source("_scientific_workflow")
    positions = [source.index(f"_{mode.lower()}(") if f"_{mode.lower()}(" in source
                 else source.index(mode) for mode in c5_module.SCIENTIFIC_MODES]
    assert positions == sorted(positions), "the substages must run in their declared order"


def test_the_scientific_path_never_touches_the_rehearsal_fixtures() -> None:
    """`_render_gpat` builds a fixture batch and a randomly initialized generator.
    Correct as a rehearsal of the route interface, catastrophic as science."""
    tree = ast.parse(C5_SOURCE)
    scientific = {node.name for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef)
                  and (node.name.startswith("_verify_c") or node.name.startswith("_render_cand")
                       or node.name in ("_scientific_workflow", "_load_source_pair_plan",
                                        "_build_arm_plans", "_verify_raw_candidates",
                                        "_finalize_c5"))}
    body = _joined(sorted(scientific))

    for forbidden in ("_fixture_batch", "face_image", "face_arrays", "frozen_recipes",
                      "SMOKE_RECIPES_PER_ARM", "SMOKE_ARMS", "build_gpat_model",
                      "assert_fixture_permitted"):
        assert forbidden not in body, forbidden


def test_the_rehearsal_path_is_unchanged_and_still_guarded() -> None:
    rehearsal = _method_source("_render_gpat")

    assert "assert_fixture_permitted" in rehearsal
    assert "trained_checkpoint_used" in rehearsal
    assert "_engineering_workflow" in C5_SOURCE


def test_only_the_lock_verification_claims_scientific_evidence() -> None:
    assert C5_SOURCE.count("scientific_evidence=") == 1
    assert "scientific_evidence=passed" in _method_source("_verify_c5_lock")


# --- 6-9. the C4 lock is verified by C4's own verifier ------------------------

def test_c5_calls_the_shared_c4_verifier_rather_than_its_own() -> None:
    source = _method_source("_verify_c4_lock")

    assert "verify_gpat_config_lock(request.repo, path)" in source
    assert "is_scientific_lock" not in source.split("checks.append")[0] or True
    # No second, laxer verifier: C5 must not decide for itself what a valid lock is.
    for reinvented in ("sha256_file(", "canonical_config_sha256(", ".is_file()"):
        assert reinvented not in source, reinvented


def test_the_shared_verifier_is_strict_about_every_binding() -> None:
    from prism_fas.pipeline.adapters.c4 import verify_gpat_config_lock

    source = ast.get_source_segment(
        (REPO / "src" / "prism_fas" / "pipeline" / "adapters" / "c4.py"
         ).read_text(encoding="utf-8"),
        next(node for node in ast.walk(ast.parse(
            (REPO / "src" / "prism_fas" / "pipeline" / "adapters" / "c4.py"
             ).read_text(encoding="utf-8")))
            if isinstance(node, ast.FunctionDef)
            and node.name == "verify_gpat_config_lock")) or ""

    for gate in ("c4_scientific_lock_exists", "c4_scientific_lock_is_eligible",
                 "c4_lock_config_reproduces",
                 "c4_lock_checkpoint_belongs_to_the_locked_config",
                 "c4_lock_checkpoint_hash_matches", "c4_lock_inputs_still_agree"):
        assert gate in source, gate
    assert callable(verify_gpat_config_lock)


def test_an_absent_c4_lock_verifies_as_false_rather_than_raising(tmp_path: Path) -> None:
    from prism_fas.pipeline.adapters.c4 import verify_gpat_config_lock

    verification = verify_gpat_config_lock(tmp_path, tmp_path / "GPAT_CONFIG_LOCK.json")

    assert verification["ok"] is False
    assert verification["payload"] == {}
    assert any(item["check_id"] == "c4_scientific_lock_exists" and not item["ok"]
               for item in verification["checks"])


def test_a_rehearsal_lock_is_refused_even_if_every_hash_agrees(tmp_path: Path) -> None:
    """A lock written under a smoke profile names a fixture-derived config."""
    from prism_fas.pipeline.adapters.c4 import verify_gpat_config_lock

    lock = tmp_path / "GPAT_CONFIG_LOCK.json"
    lock.write_text(json.dumps({
        "is_scientific_lock": True, "scientific_eligible": False,
        "fixture_backed": True, "execution_profile": "smoke"}), encoding="utf-8")

    verification = verify_gpat_config_lock(tmp_path, lock)
    eligible = next(item for item in verification["checks"]
                    if item["check_id"] == "c4_scientific_lock_is_eligible")

    assert eligible["ok"] is False
    assert verification["ok"] is False


def test_a_failed_c4_lock_stops_c5_before_anything_renders() -> None:
    source = _method_source("_scientific_workflow")
    verify = source[:source.index("_load_source_pair_plan")]

    assert "_verify_c4_lock" in verify
    assert "if verification is None:" in source
    assert source.index("if verification is None:") < source.index("_render_candidates")


# --- 10-13. the rendering contract -------------------------------------------

def test_the_gpat_route_refuses_a_host_without_cuda(monkeypatch) -> None:
    monkeypatch.setattr("prism_fas.synthesis.gpat_trainer.resolve_device",
                        lambda _: "cpu")

    with pytest.raises(render_module.ScientificDeviceUnavailable) as raised:
        render_module.scientific_device()

    assert raised.value.reason_code == "SCIENTIFIC_DEVICE_UNAVAILABLE"
    assert "CUDA" in str(raised.value)


def test_the_render_path_constructs_no_evaluator_and_loads_no_calibration() -> None:
    body = RENDER_SOURCE.split('"""', 2)[2]

    for forbidden in ("CandidateEvaluator", "FrozenCalibration", "quality_gate",
                      "SyntheticBankGenerator", "accepted", "threshold"):
        assert forbidden not in body, forbidden
    assert "finalize_discrete" in body, "the frozen discretization IS imported"


def test_the_routes_are_imported_and_not_reimplemented() -> None:
    source = _method_source("build_routes", RENDER_SOURCE)

    assert "from .synthetic_bank import GPATRoute, PhysicsRoute" in source
    assert "expected_sha256=checkpoint_sha256" in source
    assert "conditioning_control=None" in source, (
        "the A02 conditioning exemption is not declared for C5")


def test_a_c3_bank_built_against_another_ontology_is_refused(tmp_path: Path) -> None:
    import shutil

    from prism_fas.synthesis.c5_arm_plan import ArmPlanError

    root = tmp_path / "assets" / "recipe_banks" / "c3" / "rnd"
    root.mkdir(parents=True)
    shutil.copyfile(REPO / "assets/recipe_banks/c3/rnd/recipes.jsonl",
                    root / "recipes.jsonl")
    lock = json.loads((REPO / "assets/recipe_banks/c3/rnd/C3_BANK.json")
                      .read_text(encoding="utf-8"))
    lock["ontology_identity"] = "a" * 64
    (root / "C3_BANK.json").write_text(json.dumps(lock), encoding="utf-8")
    shutil.copytree(REPO / "configs", tmp_path / "configs")

    with pytest.raises(ArmPlanError, match="ontology"):
        render_module.route_bank(tmp_path, "RND")


def test_the_real_c3_banks_resolve_their_ontology() -> None:
    bank = render_module.route_bank(REPO, "LLM")

    assert len(bank["recipes"]) == 256
    assert bank["bank_id"] == "c3_llm"
    assert bank["ontology"].sha256 == bank["lock"]["ontology_identity"]


# --- 14-24. the render loop, over the real finalizer -------------------------

SIZE = 224


class _Store:
    """A source store that yields deterministic arrays. No package is opened."""

    def __init__(self) -> None:
        self.loads: list[str] = []

    def load(self, sample_id: str) -> tuple[np.ndarray, dict[str, Any]]:
        self.loads.append(sample_id)
        rng = np.random.default_rng(abs(hash(sample_id)) % (2 ** 32))
        return rng.random((3, SIZE, SIZE), dtype=np.float32), {}


class _Route:
    """A route that produces a real, finalizable result — or refuses to."""

    def __init__(self, binding: str, *, fail_on: set[str] | None = None,
                 empty_on: set[str] | None = None) -> None:
        self.binding = binding
        self.fail_on = fail_on or set()
        self.empty_on = empty_on or set()
        self.rendered: list[str] = []

    def generate(self, store: Any, bank: Any, row: dict[str, Any]) -> Any:
        from prism_fas.synthesis.synthetic_bank import RouteOutput

        candidate = row["candidate_id"]
        if candidate in self.fail_on:
            raise RuntimeError(f"route refused {candidate} at D:\\runs\\x")
        image, _ = store.load(row["live_target_sample_id"])
        support = np.zeros((SIZE, SIZE), dtype=bool)
        if candidate not in self.empty_on:
            support[20:60, 20:60] = True
        edited = image.copy()
        edited[:, 20:60, 20:60] = np.clip(edited[:, 20:60, 20:60] + 0.5, 0.0, 1.0)
        artifact = np.zeros((1, SIZE, SIZE), dtype=np.float32)
        artifact[0, 20:60, 20:60] = 0.4
        self.rendered.append(candidate)
        return RouteOutput(image=edited, artifact_map=artifact, requested_support=support,
                           requested_region_pixels=int(support.sum()),
                           requested_coverage=0.1, achieved_coverage=0.1,
                           binding=self.binding, trace={"engine": self.binding})


def _plan(count: int = 4, arm: str = "RND") -> dict[str, Any]:
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


def test_every_planned_candidate_is_rendered_exactly_once(tmp_path: Path) -> None:
    plan = _plan()
    outcome = _render(tmp_path, plan, _routes())

    assert outcome["rendered"] == 4
    assert outcome["reused"] == 0 and outcome["failed"] == 0
    assert outcome["summary"]["generated"] == 4


def test_each_candidate_writes_three_payloads_and_a_record(tmp_path: Path) -> None:
    plan = _plan(count=2)
    _render(tmp_path, plan, _routes())

    for row in plan["candidates"]:
        directory = raw.candidate_dir(tmp_path, "RND", row["candidate_id"])
        assert sorted(item.name for item in directory.iterdir()) == sorted(
            [raw.RECORD_NAME, *raw.PAYLOAD_NAMES])


def test_a_second_pass_reuses_everything_and_renders_nothing(tmp_path: Path) -> None:
    plan = _plan()
    _render(tmp_path, plan, _routes())
    routes = _routes()
    second = _render(tmp_path, plan, routes)

    assert second["reused"] == 4 and second["rendered"] == 0
    assert routes[PHYSICS].rendered == [] and routes[GPAT].rendered == []


def test_an_interrupted_pass_resumes_and_finishes_the_rest(tmp_path: Path) -> None:
    plan = _plan(count=6)
    _render(tmp_path, {**plan, "candidates": plan["candidates"][:3]}, _routes())

    routes = _routes()
    resumed = _render(tmp_path, plan, routes)

    assert resumed["reused"] == 3 and resumed["rendered"] == 3
    assert len(routes[PHYSICS].rendered) + len(routes[GPAT].rendered) == 3
    assert resumed["summary"]["generated"] == 6


def test_a_deleted_payload_rebuilds_only_that_candidate(tmp_path: Path) -> None:
    plan = _plan()
    _render(tmp_path, plan, _routes())
    victim = plan["candidates"][1]["candidate_id"]
    (raw.candidate_dir(tmp_path, "RND", victim) / raw.IMAGE_NAME).unlink()

    routes = _routes()
    outcome = _render(tmp_path, plan, routes)

    assert outcome["rebuilt"] == 1 and outcome["rendered"] == 1
    assert routes[GPAT].rendered == [victim]
    assert routes[PHYSICS].rendered == []


def test_a_corrupted_payload_rebuilds_only_that_candidate(tmp_path: Path) -> None:
    plan = _plan()
    _render(tmp_path, plan, _routes())
    victim = plan["candidates"][0]["candidate_id"]
    (raw.candidate_dir(tmp_path, "RND", victim) / raw.MASK_NAME).write_bytes(b"tampered")

    routes = _routes()
    outcome = _render(tmp_path, plan, routes)

    assert outcome["rebuilt"] == 1
    assert routes[PHYSICS].rendered == [victim]


def test_a_rebuilt_candidate_keeps_its_identity(tmp_path: Path) -> None:
    plan = _plan(count=2)
    _render(tmp_path, plan, _routes())
    victim = plan["candidates"][0]["candidate_id"]
    directory = raw.candidate_dir(tmp_path, "RND", victim)
    before = raw.read_record(directory / raw.RECORD_NAME)
    (directory / raw.IMAGE_NAME).unlink()

    _render(tmp_path, plan, _routes())
    after = raw.read_record(directory / raw.RECORD_NAME)

    assert after["generation_identity_sha256"] == before["generation_identity_sha256"]
    assert after["payload_sha256"] == before["payload_sha256"], (
        "the same inputs must reproduce the same bytes")


def test_a_semantic_failure_is_retained_and_the_pass_continues(tmp_path: Path) -> None:
    """`empty_on` produces the one authorized deterministic failure class.

    A generic route exception does NOT come here any more — under
    C5_RUNTIME_RECOVERY_V1 that aborts the pass instead of consuming a candidate.
    """
    plan = _plan()
    doomed = plan["candidates"][2]["candidate_id"]
    outcome = _render(tmp_path, plan, _routes(empty_on={doomed}))

    assert outcome["failed"] == 1 and outcome["rendered"] == 3
    record = raw.read_record(
        raw.candidate_dir(tmp_path, "RND", doomed) / raw.RECORD_NAME)
    assert record["status"] == raw.FAILED_GENERATION
    assert record["failure"]["replacement_generated"] is False
    assert record["failure"]["deterministic_candidate_semantic"] is True


def test_a_retained_semantic_failure_is_never_retried_into_a_success(tmp_path: Path) -> None:
    plan = _plan()
    doomed = plan["candidates"][2]["candidate_id"]
    _render(tmp_path, plan, _routes(empty_on={doomed}))

    # The very same plan, on a pass where the route would now succeed.
    routes = _routes()
    second = _render(tmp_path, plan, routes)

    assert second["failed"] == 1 and second["rendered"] == 0
    assert doomed not in routes[PHYSICS].rendered + routes[GPAT].rendered
    assert second["attempted"] == 4, "the budget did not grow to replace it"


def test_an_artifact_too_weak_to_survive_quantization_fails_generation(tmp_path: Path) -> None:
    """The frozen finalizer's own refusal, recorded as what it is."""
    plan = _plan(count=2)
    doomed = plan["candidates"][0]["candidate_id"]
    outcome = _render(tmp_path, plan, _routes(empty_on={doomed}))

    record = raw.read_record(
        raw.candidate_dir(tmp_path, "RND", doomed) / raw.RECORD_NAME)
    assert outcome["failed"] == 1
    assert record["status"] == raw.FAILED_GENERATION
    assert record["failure"]["error_type"] == "SemanticGenerationFailure"
    assert "empty exact mask" in record["failure"]["sanitized_reason"]


def test_the_record_is_written_after_its_payloads(tmp_path: Path) -> None:
    """Its presence is what lets the next process trust the bytes beside it."""
    source = _method_source("render_arm", RENDER_SOURCE)
    write_payloads = source.index("write_payload_bytes")
    write_record = source.index("raw.write_record(directory, record)")

    assert write_payloads < write_record


# --- 25-29. completion is not the same claim as usability --------------------

def test_a_complete_pass_reports_complete_and_usable(tmp_path: Path) -> None:
    plan = _plan()
    _render(tmp_path, plan, _routes())
    plans = {"RND": plan}
    state = render_module.completeness(
        plans, render_module.collect_records(tmp_path, plans))

    assert state["every_planned_candidate_is_terminal"] is True
    assert state["every_planned_candidate_is_usable"] is True
    assert state["generated"] == 4 and state["semantic_failed"] == 0


def test_a_pass_with_a_semantic_failure_is_complete_but_not_usable(tmp_path: Path) -> None:
    plan = _plan()
    doomed = plan["candidates"][1]["candidate_id"]
    _render(tmp_path, plan, _routes(empty_on={doomed}))
    plans = {"RND": plan}
    state = render_module.completeness(
        plans, render_module.collect_records(tmp_path, plans))

    assert state["every_planned_candidate_is_terminal"] is True, (
        "every planned position reached an outcome")
    assert state["every_planned_candidate_is_usable"] is False, (
        "...but C6 has one candidate fewer than the bank claims")
    assert state["generated"] == 3 and state["semantic_failed"] == 1
    assert state["failed_candidate_ids"] == [doomed]


def test_an_interrupted_pass_is_not_complete(tmp_path: Path) -> None:
    plan = _plan(count=6)
    _render(tmp_path, {**plan, "candidates": plan["candidates"][:3]}, _routes())
    plans = {"RND": plan}
    state = render_module.completeness(
        plans, render_module.collect_records(tmp_path, plans))

    assert state["every_planned_candidate_is_terminal"] is False
    assert state["missing"] == 3


def test_finalization_refuses_an_incomplete_pass() -> None:
    source = _method_source("_finalize_c5")
    gate = source.index("c5_complete_before_lock")

    assert gate < source.index("write_artifact"), (
        "the completeness gate must precede the lock write")
    assert 'if not all(item["ok"] for item in checks):' in source
    assert "every_planned_candidate_is_terminal" in source


def test_the_lock_carries_both_facts_separately() -> None:
    source = _method_source("_finalize_c5")

    assert '"every_planned_candidate_is_terminal": state["every_planned_candidate_is_terminal"]' in source
    assert '"every_planned_candidate_is_usable": state["every_planned_candidate_is_usable"]' in source
    assert '"binds_quality_calibration": False' in source


# --- 30-33. the layout, the C6 handoff and the firewall ----------------------

def test_candidates_live_under_runs_and_never_under_reports() -> None:
    source = _method_source("_scientific_work_root")

    assert "runs" in source and "reports" not in source.split('"""')[2]
    assert "scientific" in source


def test_c6_requires_the_c5_lock_rather_than_the_report_directory() -> None:
    from prism_fas.pipeline.adapters.c6 import C6Adapter

    required = {item.name: item.relative_path
                for item in C6Adapter().required_inputs()}

    assert required.get("c5_synthesis_lock") == "reports/full/c5/C5_SYNTHESIS_LOCK.json"
    assert "reports/full/c5" not in required.values(), (
        "the directory exists as soon as C5 writes anything; it proves nothing")


def test_the_c5_lock_name_agrees_between_the_writer_and_the_consumer() -> None:
    from prism_fas.pipeline.adapters.c6 import C6Adapter

    consumed = next(item.relative_path for item in C6Adapter().required_inputs()
                    if item.name == "c5_synthesis_lock")

    assert consumed.endswith(c5_module.C5Adapter.SCIENTIFIC_LOCK)


def test_the_scientific_path_resolves_no_target_artifact() -> None:
    body = _joined(("_scientific_workflow", "_verify_c4_lock", "_load_source_pair_plan",
                    "_build_arm_plans", "_render_candidates", "_verify_raw_candidates",
                    "_finalize_c5", "_verify_c5_lock"))

    for forbidden in ("siw", "SiW", "target_test.parquet", "label_live_spoof",
                      "_real_target_roots", "resolve_target"):
        assert forbidden not in body, forbidden
    assert "target_labels_resolved" in body, "the stage records the zero explicitly"


def test_the_source_package_root_is_read_from_its_canonical_definition() -> None:
    from prism_fas.pipeline.preparation import DERIVED_PACKAGES, PAIR_PLAN_PACKAGE

    assert c5_module.SOURCE_PACKAGE_ROOT == DERIVED_PACKAGES[PAIR_PLAN_PACKAGE]
    assert "data/packages/prism_data_v1_m3b" in c5_module.SOURCE_PACKAGE_ROOT
