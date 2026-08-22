"""The real first scientific C5 outcome, as a regression fixture.

The RTX 5090 run produced, over the frozen 6144-slot schedule:

    terminal 6144, generated 6082, semantic failures 62

    DET  generated 2020  (Physics  996, GPAT 1024)   28 Physics failures
    LLM  generated 2034  (Physics 1010, GPAT 1024)   14 Physics failures
    RND  generated 2028  (Physics 1004, GPAT 1024)   20 Physics failures

Every failure was a Physics render whose artifact did not survive uint8
quantization, and every GPAT render succeeded — 3072 of 3072. Those 62 records
are immutable negative provenance: never retried, never deleted, never replaced,
and never re-paired to a different live sample or recipe.

The first implementation of the C5 verifier would have called this outcome a
failure, because it required `generated == 6144`. That was stronger than the
frozen contract. §10.4 fixes the planned SCHEDULE at 2048 renders per arm; §11.3
puts the authority over the resulting cardinality in C6 — "Nếu một arm không đạt
1024 dưới frozen render budget/gate, C6 FAILS." This file pins the corrected
semantics against the shape that actually happened.

Nothing here re-renders anything. The pool is reconstructed as records on disk.
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

from prism_fas.pipeline.adapters import c5 as c5_module  # noqa: E402
from prism_fas.synthesis import c5_raw_generation as raw  # noqa: E402
from prism_fas.synthesis import c5_render as render_module  # noqa: E402
from prism_fas.synthesis.c5_render import SemanticGenerationFailure  # noqa: E402
from prism_fas.synthesis.c5_source_pair_plan import (ARMS, CANDIDATES_PER_ARM,  # noqa: E402
                                                     GPAT, PHYSICS)

#: The observed run, verbatim. Physics failures per arm.
OBSERVED_PHYSICS_FAILURES = {"DET": 28, "LLM": 14, "RND": 20}
OBSERVED_GENERATED = {"DET": 2020, "LLM": 2034, "RND": 2028}
OBSERVED_PHYSICS_GENERATED = {"DET": 996, "LLM": 1010, "RND": 1004}
EXPECTED_TOTAL = CANDIDATES_PER_ARM * len(ARMS)
SEMANTIC_REASON = ("the artifact did not survive uint8 quantization and "
                   "finalized to an empty exact mask over 0 requested support pixels")


def _arm_plan(arm: str, count: int = CANDIDATES_PER_ARM) -> dict[str, Any]:
    return {
        "arm": arm, "arm_plan_identity": f"armplan-{arm}",
        "source_pair_plan_identity": "baseplan", "package_identity": "b" * 64,
        "recipe_bank_identity": f"bank-{arm}",
        "recipe_bank_root": f"assets/recipe_banks/c3/{arm.lower()}",
        "selected_set_identity": f"selected-{arm}", "ontology_identity": "onto",
        "planned_candidates": count, "binds_quality_calibration": False,
        "candidates": [{
            "candidate_id": f"c5syn_{arm.lower()}_{index:05d}", "arm": arm,
            "recipe_id": f"r{index % 256}", "recipe_ordinal": index % 256,
            "slot": (index % 8) + 1, "position": index,
            "route": PHYSICS if index % 2 == 0 else GPAT,
            "domain_relation": "same_domain",
            "live_target_sample_id": f"live_{index:05d}",
            "spoof_source_sample_id": None if index % 2 == 0 else f"spoof_{index:05d}",
            "recipe_bank_identity": f"bank-{arm}",
            "generator_binding": "m7-physics-v1" if index % 2 == 0 else "c" * 64,
        } for index in range(count)],
    }


def _write_generated(directory: Path, identity: Any) -> None:
    import hashlib

    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for index, name in enumerate(raw.PAYLOAD_NAMES):
        payload = f"{identity.candidate_id}:{index}".encode()
        (directory / name).write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    record = raw.CandidateRecord(identity=identity, status=raw.GENERATED,
                                 payload_sha256=hashes)
    (directory / raw.RECORD_NAME).write_text(json.dumps(record.as_dict()),
                                             encoding="utf-8")


def _write_semantic_failure(directory: Path, identity: Any) -> None:
    """Exactly what `render_arm` writes for the empty-exact-mask class."""
    directory.mkdir(parents=True, exist_ok=True)
    record = raw.failure_record(identity, stage="render_physics",
                                error=SemanticGenerationFailure(SEMANTIC_REASON))
    (directory / raw.RECORD_NAME).write_text(json.dumps(record.as_dict()),
                                             encoding="utf-8")


def _build_pool(root: Path, physics_failures: dict[str, int],
                count: int = CANDIDATES_PER_ARM) -> dict[str, dict[str, Any]]:
    """The observed pool: N Physics slots per arm end as semantic failures.

    `count` shrinks the schedule for the cases where arm SIZE is irrelevant — a
    corrupt payload or a fabricated failure record is a per-candidate property,
    and building 6144 slots to prove one of them would cost minutes of file I/O
    per test for nothing. The cases that genuinely turn on the frozen numbers
    (the observed run, and the 511-vs-512 floor) use the real size.
    """
    plans = {arm: _arm_plan(arm, count) for arm in ARMS}
    for arm, plan in plans.items():
        budget = physics_failures.get(arm, 0)
        for row in plan["candidates"]:
            identity = render_module.identity_for(row, plan)
            directory = raw.candidate_dir(root, arm, identity.candidate_id)
            if row["route"] == PHYSICS and budget > 0:
                _write_semantic_failure(directory, identity)
                budget -= 1
            else:
                _write_generated(directory, identity)
    return plans


@pytest.fixture(scope="module")
def observed(tmp_path_factory) -> tuple[Path, dict[str, dict[str, Any]], dict[str, Any]]:
    root = tmp_path_factory.mktemp("gpu")
    plans = _build_pool(root, OBSERVED_PHYSICS_FAILURES)
    return root, plans, c5_module.verify_c5_candidates(root, plans)


# --- 1, 2, 10. the observed outcome is valid scientific C5 evidence ----------

def test_the_observed_pool_reproduces_the_recorded_counts(observed) -> None:
    _, _, result = observed

    assert result["planned"] == EXPECTED_TOTAL == 6144
    assert result["terminal"] == 6144
    assert result["generated"] == 6082
    assert result["semantic_failed"] == 62
    assert result["runtime_unresolved"] == 0
    assert sum(OBSERVED_PHYSICS_FAILURES.values()) == 62


def test_the_observed_pool_is_scientifically_complete(observed) -> None:
    """The case the old `generated == 6144` rule would have rejected."""
    _, _, result = observed

    assert result["problems"] == []
    assert result["schedule_exact"] is True
    assert result["every_planned_slot_is_terminal"] is True
    assert result["pool_complete"] is True
    assert result["every_planned_slot_generated"] is False, (
        "62 slots produced no payload, and the pool is complete anyway")


def test_the_per_arm_shape_matches_the_gpu_run(observed) -> None:
    _, _, result = observed

    for arm in ARMS:
        counts = result["per_arm"][arm]
        assert counts["terminal"] == CANDIDATES_PER_ARM
        assert counts["generated"] == OBSERVED_GENERATED[arm], arm
        assert counts["semantic_failed"] == OBSERVED_PHYSICS_FAILURES[arm], arm
        assert counts["generated_by_route"][PHYSICS] == OBSERVED_PHYSICS_GENERATED[arm]
        assert counts["generated_by_route"][GPAT] == 1024, arm
        assert counts["semantic_failed_by_route"][GPAT] == 0, (
            f"{arm}: every GPAT render succeeded in the observed run")


def test_all_3072_gpat_renders_survive(observed) -> None:
    _, _, result = observed

    total_gpat = sum(counts["generated_by_route"][GPAT]
                     for counts in result["per_arm"].values())
    assert total_gpat == 3072


def test_the_planned_route_split_is_still_the_frozen_1024_1024(observed) -> None:
    """The failures changed the yield, never the schedule."""
    _, _, result = observed

    for arm in ARMS:
        planned = result["per_arm"][arm]["planned_by_route"]
        assert planned[PHYSICS] == 1024 and planned[GPAT] == 1024, arm
    assert result["expected_planned_per_route"] == 1024


def test_the_observed_pool_is_c6_pre_gate_feasible(observed) -> None:
    """996 / 1010 / 1004 Physics all clear the 512 floor, so C6 may proceed."""
    _, _, result = observed
    floor = c5_module._c6_route_floor()

    assert floor == 512
    for arm in ARMS:
        routes = result["per_arm"][arm]["generated_by_route"]
        assert routes[PHYSICS] >= floor, arm
        assert routes[GPAT] >= floor, arm


# --- 3, 4, 5. what a failure record may and may not contain ------------------

def test_no_failed_candidate_was_replaced(observed) -> None:
    root, plans, result = observed

    identifiers = [row["candidate_id"] for plan in plans.values()
                   for row in plan["candidates"]]
    on_disk = [path.parent.name for arm in ARMS
               for path in (root / arm).glob("*/CANDIDATE.json")]

    assert len(on_disk) == EXPECTED_TOTAL, "no extra candidate exists"
    assert sorted(on_disk) == sorted(identifiers), (
        "the pool is exactly the frozen schedule; nothing was resampled")


def test_a_semantic_failure_record_carries_no_payload(observed) -> None:
    root, plans, _ = observed
    plan = plans["DET"]
    failed = next(row for row in plan["candidates"] if row["route"] == PHYSICS)
    directory = raw.candidate_dir(root, "DET", failed["candidate_id"])
    record = raw.read_record(directory / raw.RECORD_NAME)

    assert record["status"] == raw.FAILED_GENERATION
    assert record["payload_sha256"] == {} and record["payloads"] == []
    assert record["failure"]["error_type"] == "SemanticGenerationFailure"
    assert record["failure"]["replacement_generated"] is False
    assert record["failure"]["deterministic_candidate_semantic"] is True
    for name in raw.PAYLOAD_NAMES:
        assert not (directory / name).exists()


def test_a_generated_record_still_needs_three_byte_valid_payloads(observed) -> None:
    root, plans, _ = observed
    plan = plans["LLM"]
    row = next(item for item in plan["candidates"] if item["route"] == GPAT)
    directory = raw.candidate_dir(root, "LLM", row["candidate_id"])
    record = raw.read_record(directory / raw.RECORD_NAME)

    assert sorted(record["payload_sha256"]) == sorted(raw.PAYLOAD_NAMES)
    for name in raw.PAYLOAD_NAMES:
        assert raw.sha256_file(directory / name) == record["payload_sha256"][name]


# --- contrast A. one runtime-incomplete slot breaks completion ---------------

def test_a_single_runtime_incomplete_slot_blocks_c5(tmp_path: Path) -> None:
    plans = _build_pool(tmp_path, {"DET": 1, "LLM": 0, "RND": 0}, count=8)
    row = plans["RND"]["candidates"][0]
    identity = render_module.identity_for(row, plans["RND"])
    directory = raw.candidate_dir(tmp_path, "RND", identity.candidate_id)
    (directory / raw.RECORD_NAME).unlink()
    for name in raw.PAYLOAD_NAMES:
        (directory / name).unlink(missing_ok=True)
    raw.append_runtime_attempt(directory, identity, stage="render_physics",
                               error=RuntimeError("CUDA error"))

    result = c5_module.verify_c5_candidates(tmp_path, plans)

    assert result["runtime_unresolved"] == 1
    assert result["every_planned_slot_is_terminal"] is False
    assert result["pool_complete"] is False
    assert result["problems"][0]["reason"] == "RUNTIME_UNRESOLVED"
    assert result["terminal"] == 24 - 1


# --- contrast B. complete, but C6 already impossible -------------------------

@pytest.fixture(scope="module")
def starved(tmp_path_factory) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """A full-size pool with DET one Physics candidate below C6's floor."""
    root = tmp_path_factory.mktemp("starved")
    floor = c5_module._c6_route_floor()
    plans = _build_pool(root, {"DET": 1024 - (floor - 1), "LLM": 0, "RND": 0})
    return plans, c5_module.verify_c5_candidates(root, plans)


def test_511_generated_physics_is_complete_but_not_pre_gate_feasible(starved) -> None:
    """One below the floor. C5 still did its job; C6 cannot start."""
    _, result = starved
    floor = c5_module._c6_route_floor()

    assert result["pool_complete"] is True, "every planned slot is still terminal"
    assert result["terminal"] == EXPECTED_TOTAL
    assert result["per_arm"]["DET"]["generated_by_route"][PHYSICS] == floor - 1 == 511
    assert result["per_arm"]["DET"]["generated_by_route"][GPAT] == 1024
    # ...and the pre-gate conclusion is the one that fails.
    starved_arms = [arm for arm in ARMS
                    if result["per_arm"][arm]["generated_by_route"][PHYSICS] < floor]
    assert starved_arms == ["DET"]


# --- contrast C. a corrupted payload breaks completion -----------------------

def test_a_corrupted_generated_payload_blocks_c5(tmp_path: Path) -> None:
    plans = _build_pool(tmp_path, {"DET": 2, "LLM": 0, "RND": 0}, count=8)
    row = plans["LLM"]["candidates"][0]
    identity = render_module.identity_for(row, plans["LLM"])
    directory = raw.candidate_dir(tmp_path, "LLM", identity.candidate_id)
    (directory / raw.IMAGE_NAME).write_bytes(b"tampered")

    result = c5_module.verify_c5_candidates(tmp_path, plans)

    assert result["pool_complete"] is False
    assert result["problems"][0]["reason"] == "PAYLOAD_CHANGED"


# --- contrast D. a failure record that claims a replacement ------------------

def test_a_failure_record_claiming_a_replacement_blocks_c5(tmp_path: Path) -> None:
    plans = _build_pool(tmp_path, {"DET": 2, "LLM": 0, "RND": 0}, count=8)
    row = next(item for item in plans["DET"]["candidates"]
               if item["route"] == PHYSICS)
    directory = raw.candidate_dir(tmp_path, "DET", row["candidate_id"])
    record = raw.read_record(directory / raw.RECORD_NAME)
    record["failure"]["replacement_generated"] = True
    (directory / raw.RECORD_NAME).write_text(json.dumps(record), encoding="utf-8")

    result = c5_module.verify_c5_candidates(tmp_path, plans)

    assert result["pool_complete"] is False
    assert result["problems"][0]["reason"] == "INVALID_FAILURE_RECORD"
    assert result["problems"][0]["replacement_generated"] is True


@pytest.mark.parametrize("mutation,field", [
    ({"error_type": "RuntimeError"}, "error_type"),
    ({"deterministic_candidate_semantic": False}, "deterministic_candidate_semantic"),
])
def test_a_failure_record_that_is_not_the_semantic_class_blocks_c5(
        tmp_path: Path, mutation, field) -> None:
    """Only the proven deterministic class may consume a planned slot."""
    plans = _build_pool(tmp_path, {"DET": 2, "LLM": 0, "RND": 0}, count=8)
    row = next(item for item in plans["DET"]["candidates"]
               if item["route"] == PHYSICS)
    directory = raw.candidate_dir(tmp_path, "DET", row["candidate_id"])
    record = raw.read_record(directory / raw.RECORD_NAME)
    record["failure"].update(mutation)
    (directory / raw.RECORD_NAME).write_text(json.dumps(record), encoding="utf-8")

    result = c5_module.verify_c5_candidates(tmp_path, plans)

    assert result["pool_complete"] is False
    assert result["problems"][0]["reason"] == "INVALID_FAILURE_RECORD"


def test_a_failure_record_that_claims_payloads_blocks_c5(tmp_path: Path) -> None:
    """A fabricated completion: a failure with payload hashes beside it."""
    plans = _build_pool(tmp_path, {"DET": 2, "LLM": 0, "RND": 0}, count=8)
    row = next(item for item in plans["DET"]["candidates"]
               if item["route"] == PHYSICS)
    directory = raw.candidate_dir(tmp_path, "DET", row["candidate_id"])
    record = raw.read_record(directory / raw.RECORD_NAME)
    record["payload_sha256"] = {name: "0" * 64 for name in raw.PAYLOAD_NAMES}
    (directory / raw.RECORD_NAME).write_text(json.dumps(record), encoding="utf-8")

    result = c5_module.verify_c5_candidates(tmp_path, plans)

    assert result["pool_complete"] is False
    assert result["problems"][0]["declared_payloads"] is True


# --- 9. the old predicate no longer governs C5 -------------------------------

def test_the_old_all_generated_condition_no_longer_controls_c5(observed) -> None:
    _, _, result = observed
    source = (REPO / "src" / "prism_fas" / "pipeline" / "adapters" / "c5.py"
              ).read_text(encoding="utf-8")

    assert result["every_planned_slot_generated"] is False
    assert result["pool_complete"] is True
    # The descriptive field survives for older readers but governs nothing.
    assert "every_planned_candidate_is_usable_is_descriptive_only" in source
    assert 'payload.get("every_planned_candidate_is_usable") is True' not in source


# --- the GPU lock written under the old verifier is preserved ----------------

def test_an_existing_lock_is_archived_byte_for_byte_before_being_replaced(
        tmp_path: Path) -> None:
    """The first GPU run's lock is a real measurement of a real render pass.

    Correcting the verifier's contract afterwards must not delete it.
    """
    import hashlib
    from types import SimpleNamespace

    reports = tmp_path / "reports" / "full" / "c5"
    reports.mkdir(parents=True)
    lock = reports / "C5_SYNTHESIS_LOCK.json"
    original = json.dumps({"schema_version": "c5-synthesis-lock-v1",
                           "generated_at_utc": "2026-08-22T10:00:00+00:00",
                           "generated": 6082, "failed": 62,
                           "every_planned_candidate_is_terminal": True,
                           "every_planned_candidate_is_usable": False})
    lock.write_text(original, encoding="utf-8")

    info = c5_module._archive_superseded_lock(SimpleNamespace(repo=tmp_path), lock)

    archived = tmp_path / info["archived_lock"]
    assert archived.read_text(encoding="utf-8") == original
    assert info["archived_lock_sha256"] == hashlib.sha256(
        original.encode("utf-8")).hexdigest()
    assert info["archived_counts"]["generated"] == 6082
    assert info["supersedes_verifier_semantics"] is True
    assert info["candidates_modified"] is False
    assert "stage ownership" in info["reason"]
    assert lock.read_text(encoding="utf-8") == original, "the original is untouched"


def test_archiving_is_a_no_op_when_no_lock_exists(tmp_path: Path) -> None:
    from types import SimpleNamespace

    assert c5_module._archive_superseded_lock(
        SimpleNamespace(repo=tmp_path), tmp_path / "C5_SYNTHESIS_LOCK.json") is None


def test_the_new_lock_binds_the_superseded_one() -> None:
    source = (REPO / "src" / "prism_fas" / "pipeline" / "adapters" / "c5.py"
              ).read_text(encoding="utf-8")

    assert "_archive_superseded_lock(request, reports / self.SCIENTIFIC_LOCK)" in source
    assert 'lock_payload["supersedes"] = superseded' in source
    assert "c5_superseded_lock_archived" in source


def test_the_lock_keeps_all_five_counts_apart() -> None:
    source = (REPO / "src" / "prism_fas" / "pipeline" / "adapters" / "c5.py"
              ).read_text(encoding="utf-8")

    for field in ('"planned":', '"terminal":', '"generated":', '"semantic_failed":',
                  '"runtime_unresolved":', '"usable_generated_by_arm_and_route":',
                  '"c6_pre_gate_route_floor_feasible":'):
        assert field in source, field
    assert '"lock_kind": "scientific_candidate_pool"' in source
