"""The strict C5 lock verification and the C6 handoff it gates.

Three defects are covered here, and each one passed the previous verification:

* a lock could be written and verified with 6143 usable candidates out of 6144;
* payload verification hashed the SHA strings stored in `CANDIDATE.json` rather
  than the files, so a truncated PNG verified;
* the identity check compared the lock's recorded identity against the same
  lock's recorded identity, which is true of any JSON file.

The full 6144-candidate tree is built once for this module. Its payloads are
arbitrary bytes: `reuse_decision` hashes files, and what a real PNG contains
makes no difference to whether its bytes still hash to what was recorded.

That fixture is slow on Windows — minutes, nearly all of it inside `open()` at
about 7ms per freshly written file, which is the on-access virus scanner rather
than anything this code does. It is paid once per session and is the price of
checking the actual 2048-per-arm contract instead of a scaled-down imitation of
it. Everything that is a per-candidate property uses the twelve-candidate bank.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.pipeline.adapters import AdapterRequest  # noqa: E402
from prism_fas.pipeline.adapters import c5 as c5_module  # noqa: E402
from prism_fas.pipeline.adapters.c6 import C6Adapter  # noqa: E402
from prism_fas.pipeline.profiles import load_profile  # noqa: E402
from prism_fas.synthesis import c5_raw_generation as raw  # noqa: E402
from prism_fas.synthesis import c5_render as render_module  # noqa: E402
from prism_fas.synthesis.c5_source_pair_plan import (ARMS, CANDIDATES_PER_ARM,  # noqa: E402
                                                     GPAT, PHYSICS)

C5_SOURCE = (REPO / "src" / "prism_fas" / "pipeline" / "adapters" / "c5.py"
             ).read_text(encoding="utf-8")
EXPECTED_TOTAL = CANDIDATES_PER_ARM * len(ARMS)
CHECKPOINT = "c" * 64
PHYSICS_VERSION = "m7-physics-v1"


def _function_source(name: str, source: str = C5_SOURCE) -> str:
    tree = ast.parse(source)
    node = next(item for item in ast.walk(tree)
                if isinstance(item, ast.FunctionDef) and item.name == name)
    return ast.get_source_segment(source, node) or ""


# --- the fabricated scientific bank ------------------------------------------

def _arm_plan(arm: str, count: int = CANDIDATES_PER_ARM) -> dict[str, Any]:
    return {
        "arm": arm, "arm_plan_identity": f"armplan-{arm}",
        "source_pair_plan_identity": "baseplan", "package_identity": "b" * 64,
        "recipe_bank_identity": f"bank-{arm}",
        "recipe_bank_root": f"assets/recipe_banks/c3/{arm.lower()}",
        "selected_set_identity": f"selected-{arm}", "ontology_identity": "onto",
        "physics_engine_version": PHYSICS_VERSION,
        "gpat_checkpoint_sha256": CHECKPOINT,
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
            "generator_binding": PHYSICS_VERSION if index % 2 == 0 else CHECKPOINT,
        } for index in range(count)],
    }


def _plans(count: int = CANDIDATES_PER_ARM) -> dict[str, dict[str, Any]]:
    return {arm: _arm_plan(arm, count) for arm in ARMS}


def _materialize(root: Path, plans: dict[str, dict[str, Any]], *,
                 fail: set[str] | None = None) -> None:
    """Build a candidate tree the verifier will read.

    The record is written with a plain `write_text` rather than through
    `raw.write_record`. The production writer is atomic — temp file, fsync,
    replace — which is the right cost to pay 6144 times during a real render and
    the wrong one to pay while building a fixture; it is what a separate test
    already covers. The BYTES on disk are identical either way, and the bytes are
    the whole of what verification looks at.
    """
    import hashlib

    fail = fail or set()
    for arm, plan in plans.items():
        for row in plan["candidates"]:
            identity = render_module.identity_for(row, plan)
            directory = raw.candidate_dir(root, arm, identity.candidate_id)
            directory.mkdir(parents=True, exist_ok=True)
            if identity.candidate_id in fail:
                record = raw.failure_record(identity, stage="render",
                                            error=RuntimeError("refused"))
            else:
                hashes = {}
                for index, name in enumerate(raw.PAYLOAD_NAMES):
                    payload = f"{identity.candidate_id}:{index}".encode()
                    (directory / name).write_bytes(payload)
                    hashes[name] = hashlib.sha256(payload).hexdigest()
                record = raw.CandidateRecord(identity=identity, status=raw.GENERATED,
                                             payload_sha256=hashes)
            (directory / raw.RECORD_NAME).write_text(
                json.dumps(record.as_dict()), encoding="utf-8")


@pytest.fixture(scope="module")
def full_bank(tmp_path_factory) -> tuple[Path, dict[str, dict[str, Any]]]:
    """6144 candidates, complete and verifiable. Built once."""
    root = tmp_path_factory.mktemp("bank")
    plans = _plans()
    _materialize(root, plans)
    return root, plans


@pytest.fixture(scope="module")
def full_result(full_bank) -> dict[str, Any]:
    """One verification pass over 6144 candidates, reused by the count tests.

    Re-hashing 18432 files per test would buy nothing: the pass is deterministic
    and every assertion below reads a different field of the same result.
    """
    root, plans = full_bank
    return c5_module.verify_c5_candidates(root, plans)


@pytest.fixture(scope="module")
def small_bank(tmp_path_factory) -> tuple[Path, dict[str, dict[str, Any]]]:
    """Twelve candidates. Corruption is a per-candidate property, so the arm size
    changes nothing about what these tests prove and costs 500x less to build."""
    root = tmp_path_factory.mktemp("small")
    plans = _plans(count=4)
    _materialize(root, plans)
    return root, plans


@pytest.fixture
def repaired(small_bank):
    """Hand a test the shared bank and undo whatever it broke."""
    root, plans = small_bank
    broken: list[tuple[Path, bytes | None]] = []

    def damage(path: Path, content: bytes | None) -> None:
        broken.append((path, path.read_bytes() if path.is_file() else None))
        if content is None:
            path.unlink()
        else:
            path.write_bytes(content)

    yield root, plans, damage
    for path, original in reversed(broken):
        if original is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(original)


def _first_candidate(plans: dict[str, dict[str, Any]]) -> tuple[str, str]:
    plan = plans["RND"]
    return "RND", plan["candidates"][0]["candidate_id"]


# --- 1, 17, 18. the complete bank verifies -----------------------------------

def test_a_full_6144_candidate_bank_verifies(full_result) -> None:
    result = full_result

    assert result["planned"] == EXPECTED_TOTAL == 6144
    assert result["verified"] == EXPECTED_TOTAL
    assert result["problems"] == []
    assert result["all_verified"] is True


def test_exactly_2048_verified_candidates_per_arm_is_enforced(full_result) -> None:
    result = full_result

    assert result["expected_per_arm"] == 2048
    assert {arm: counts["verified"] for arm, counts in result["per_arm"].items()} == {
        "RND": 2048, "DET": 2048, "LLM": 2048}
    assert result["counts_exact"] is True


def test_exactly_1024_physics_and_1024_gpat_per_arm_is_enforced(full_result) -> None:
    result = full_result

    assert result["expected_per_route"] == 1024
    for arm, counts in result["per_arm"].items():
        assert counts[PHYSICS] == 1024, arm
        assert counts[GPAT] == 1024, arm


def test_a_short_arm_fails_the_count_check(tmp_path: Path) -> None:
    """2047 verified candidates is not 2048, however healthy each one is."""
    plans = _plans(count=4)
    _materialize(tmp_path, plans)
    result = c5_module.verify_c5_candidates(tmp_path, plans)

    assert result["problems"] == [], "every candidate present is individually fine"
    assert result["counts_exact"] is False, "...and the arm is still short"
    assert result["all_verified"] is False


# --- 2. one failure is complete and NOT usable -------------------------------

def test_6143_generated_and_one_failed_does_not_verify(full_bank) -> None:
    """The exact case the previous verification let through."""
    root, plans = full_bank
    arm, candidate = _first_candidate(plans)
    directory = raw.candidate_dir(root, arm, candidate)
    kept = {name: (directory / name).read_bytes() for name in raw.PAYLOAD_NAMES}
    record = raw.read_record(directory / raw.RECORD_NAME)
    identity = render_module.identity_for(plans[arm]["candidates"][0], plans[arm])
    raw.write_record(directory, raw.failure_record(
        identity, stage="render_physics", error=RuntimeError("boom")))
    try:
        result = c5_module.verify_c5_candidates(root, plans)

        assert result["verified"] == EXPECTED_TOTAL - 1 == 6143
        assert result["all_verified"] is False
        assert result["problems"][0]["reason"] == "FAILED_GENERATION"
    finally:
        for name, payload in kept.items():
            (directory / name).write_bytes(payload)
        (directory / raw.RECORD_NAME).write_text(json.dumps(record), encoding="utf-8")


def test_a_lock_claiming_terminal_but_not_usable_is_refused(tmp_path: Path) -> None:
    lock = tmp_path / "C5_SYNTHESIS_LOCK.json"
    lock.write_text(json.dumps({
        "is_scientific_lock": True, "scientific_eligible": True,
        "fixture_backed": False, "binds_quality_calibration": False,
        "every_planned_candidate_is_terminal": True,
        "every_planned_candidate_is_usable": False,
        "generated": 6143, "failed": 1}), encoding="utf-8")

    verification = c5_module.verify_c5_synthesis_lock(REPO, lock)
    by_id = {item["check_id"]: item["ok"] for item in verification["checks"]}

    assert by_id["c5_lock_declares_every_candidate_terminal"] is True
    assert by_id["c5_lock_declares_every_candidate_usable"] is False
    assert by_id["c5_lock_counts_are_the_frozen_budget"] is False
    assert verification["lock_valid"] is False


def test_the_two_facts_are_separate_checks_not_one() -> None:
    source = _function_source("verify_c5_synthesis_lock")

    assert "c5_lock_declares_every_candidate_terminal" in source
    assert "c5_lock_declares_every_candidate_usable" in source
    assert source.index("terminal") < source.index("usable")


# --- 5-9. payload BYTES, not recorded hashes ---------------------------------

@pytest.mark.parametrize("payload_name", list(raw.PAYLOAD_NAMES))
def test_a_corrupted_payload_file_fails_verification(repaired, payload_name) -> None:
    root, plans, damage = repaired
    arm, candidate = _first_candidate(plans)
    damage(raw.candidate_dir(root, arm, candidate) / payload_name, b"tampered")

    result = c5_module.verify_c5_candidates(root, plans)

    assert result["all_verified"] is False
    assert result["problems"] == [{"arm": arm, "candidate_id": candidate,
                                   "reason": "PAYLOAD_CHANGED", "payload": payload_name}]


def test_a_missing_payload_file_fails_verification(repaired) -> None:
    root, plans, damage = repaired
    arm, candidate = _first_candidate(plans)
    damage(raw.candidate_dir(root, arm, candidate) / raw.IMAGE_NAME, None)

    result = c5_module.verify_c5_candidates(root, plans)

    planned = sum(len(plan["candidates"]) for plan in plans.values())
    assert result["problems"][0]["reason"] == "PAYLOAD_MISSING"
    assert result["verified"] == planned - 1


def test_a_record_bound_to_another_identity_fails_verification(repaired) -> None:
    root, plans, damage = repaired
    arm, candidate = _first_candidate(plans)
    directory = raw.candidate_dir(root, arm, candidate)
    record = raw.read_record(directory / raw.RECORD_NAME)
    record["generation_identity"]["recipe_id"] = "a different recipe"
    record["generation_identity_sha256"] = "0" * 64
    damage(directory / raw.RECORD_NAME, json.dumps(record).encode())

    result = c5_module.verify_c5_candidates(root, plans)

    assert result["problems"][0]["reason"] == "STALE"


def test_verification_reads_files_rather_than_the_recorded_hashes(repaired) -> None:
    """The defect, stated directly.

    `payload_set_digest` hashes the SHA strings inside CANDIDATE.json. Corrupting
    a PNG leaves those strings untouched, so it agrees with itself — while the
    verifier, which re-reads the bytes, does not.
    """
    root, plans, damage = repaired
    arm, candidate = _first_candidate(plans)
    records = render_module.collect_records(root, plans)
    before = raw.payload_set_digest(records)

    damage(raw.candidate_dir(root, arm, candidate) / raw.IMAGE_NAME, b"tampered")

    after = raw.payload_set_digest(render_module.collect_records(root, plans))
    assert after == before, "hashing the recorded hashes cannot see this"
    assert c5_module.verify_c5_candidates(root, plans)["all_verified"] is False


def test_the_verifier_calls_the_canonical_reuse_decision() -> None:
    source = _function_source("verify_c5_candidates")

    assert "raw.reuse_decision(directory, identity)" in source
    assert "payload_set_digest" not in source, (
        "the recorded-hash digest is not evidence about bytes on disk")


# --- 10-16. current inputs, never the lock's own values ----------------------

def _current(**overrides: Any) -> dict[str, Any]:
    base = {"ok": True, "checks": [], "plans": _plans(count=8),
            "package_identity": "b" * 64,
            "source_pair_plan_identity": "baseplan",
            "gpat_checkpoint_sha256": CHECKPOINT,
            "physics_engine_version": PHYSICS_VERSION}
    base.update(overrides)
    return base


def _lock(**overrides: Any) -> dict[str, Any]:
    base = {"package_identity": "b" * 64, "source_pair_plan_identity": "baseplan",
            "gpat_checkpoint_sha256": CHECKPOINT,
            "physics_engine_version": PHYSICS_VERSION,
            "arms": {arm: {"arm_plan_identity": f"armplan-{arm}",
                           "recipe_bank_identity": f"bank-{arm}",
                           "selected_set_identity": f"selected-{arm}",
                           "ontology_identity": "onto",
                           "planned_candidates": CANDIDATES_PER_ARM}
                     for arm in ARMS}}
    base.update(overrides)
    return base


def _compare(lock: dict[str, Any], current: dict[str, Any]) -> dict[str, bool]:
    return {item["check_id"]: item["ok"]
            for item in c5_module.compare_lock_to_current(lock, current)}


def test_an_agreeing_lock_and_current_input_set_compares_clean() -> None:
    assert all(_compare(_lock(), _current()).values())


def test_a_changed_m3b_package_identity_invalidates_the_lock() -> None:
    result = _compare(_lock(), _current(package_identity="a rebuilt package"))

    assert result["c5_lock_binds_the_current_package"] is False


def test_a_changed_source_pair_plan_identity_invalidates_the_lock() -> None:
    result = _compare(_lock(), _current(source_pair_plan_identity="another schedule"))

    assert result["c5_lock_binds_the_current_schedule"] is False


def test_a_changed_c4_checkpoint_sha_invalidates_the_lock() -> None:
    result = _compare(_lock(), _current(gpat_checkpoint_sha256="d" * 64))

    assert result["c5_lock_binds_the_current_checkpoint"] is False


@pytest.mark.parametrize("arm", list(ARMS))
def test_a_changed_c3_bank_identity_invalidates_that_arm(arm: str) -> None:
    plans = _plans(count=8)
    plans[arm]["recipe_bank_identity"] = "a regenerated bank"

    result = _compare(_lock(), _current(plans=plans))

    assert result[f"c5_lock_arm_{arm.lower()}_binds_current_inputs"] is False
    for other in ARMS:
        if other != arm:
            assert result[f"c5_lock_arm_{other.lower()}_binds_current_inputs"] is True


@pytest.mark.parametrize("field", ["arm_plan_identity", "ontology_identity",
                                   "selected_set_identity"])
def test_a_changed_arm_binding_invalidates_that_arm(field: str) -> None:
    plans = _plans(count=8)
    plans["LLM"][field] = "changed"

    result = _compare(_lock(), _current(plans=plans))

    assert result["c5_lock_arm_llm_binds_current_inputs"] is False


def test_a_changed_physics_engine_version_invalidates_the_lock() -> None:
    result = _compare(_lock(), _current(physics_engine_version="m7-physics-v2"))

    assert result["c5_lock_binds_the_current_physics_engine"] is False


def test_the_planned_count_is_checked_against_the_frozen_constant() -> None:
    lock = _lock()
    for arm in ARMS:
        lock["arms"][arm]["planned_candidates"] = 1120     # the Version-B number

    result = _compare(lock, _current())

    assert not any(result[f"c5_lock_arm_{arm.lower()}_binds_current_inputs"]
                   for arm in ARMS)


def test_every_expected_value_is_recomputed_and_none_is_copied_from_the_lock() -> None:
    """The self-referential defect, ruled out structurally.

    A comparison whose expected side came from `payload` would be true of any
    lock. Every comparison in this function reads `current` on one side.
    """
    tree = ast.parse(_function_source("compare_lock_to_current"))
    comparisons = [node for node in ast.walk(tree) if isinstance(node, ast.Compare)]
    assert comparisons

    for node in comparisons:
        rendered = ast.dump(node)
        if "payload" not in rendered:
            continue
        assert "current" in rendered or "plan" in rendered or "CANDIDATES_PER_ARM" in rendered, (
            f"a lock value is compared against another lock value: {ast.unparse(node)}")


def test_the_current_inputs_are_rebuilt_without_reading_the_lock() -> None:
    source = _function_source("reconstruct_current_c5_inputs")

    for rebuilt in ("verify_gpat_config_lock", "verify_support_inputs",
                    "build_source_pair_plan", "build_all_arm_plans",
                    "PHYSICS_ENGINE_VERSION"):
        assert rebuilt in source, rebuilt
    # C4_SCIENTIFIC_LOCK is an INPUT and is rightly read. The C5 lock is not.
    for forbidden in ("payload", "C5_SYNTHESIS_LOCK", "self.SCIENTIFIC_LOCK",
                      "lock_path"):
        assert forbidden not in source, (
            f"the reconstruction must not read the C5 lock: {forbidden}")
    assert "C4_SCIENTIFIC_LOCK" in source


def test_resume_decision_is_no_longer_the_identity_proof() -> None:
    """It compared the lock's own field against the lock's own field."""
    source = _function_source("_verify_c5_lock")

    assert "resume_decision" not in source
    assert "verify_c5_synthesis_lock(request.repo, path)" in source


# --- 3, 4, 19, 20. the C6 handoff --------------------------------------------

def _request(repo: Path) -> AdapterRequest:
    return AdapterRequest(repo=repo, profile=load_profile("full", repo=REPO))


def test_a_lock_that_merely_exists_does_not_unblock_c6(tmp_path: Path) -> None:
    lock = tmp_path / "reports" / "full" / "c5" / "C5_SYNTHESIS_LOCK.json"
    lock.parent.mkdir(parents=True)
    lock.write_text(json.dumps({"is_scientific_lock": True, "scientific_eligible": True,
                                "fixture_backed": False, "generated": 6144,
                                "failed": 0, "binds_quality_calibration": False,
                                "every_planned_candidate_is_terminal": True,
                                "every_planned_candidate_is_usable": True}),
                    encoding="utf-8")
    adapter = C6Adapter()
    request = _request(tmp_path)

    presence = {item.name: item.resolve(tmp_path)
                for item in adapter.required_inputs()}
    semantic = adapter.semantic_preconditions(request)[0]

    assert presence["c5_synthesis_lock"]["present"] is True, "the file is there"
    assert semantic["present"] is False, "...and it still proves nothing"
    assert semantic["blocking"] is True


def test_an_engineering_c5_artifact_cannot_unblock_scientific_c6(tmp_path: Path) -> None:
    lock = tmp_path / "reports" / "full" / "c5" / "C5_SYNTHESIS_LOCK.json"
    lock.parent.mkdir(parents=True)
    lock.write_text(json.dumps({"is_scientific_lock": False, "fixture_backed": True,
                                "scientific_eligible": False, "generated": 6144,
                                "failed": 0}), encoding="utf-8")

    semantic = C6Adapter().semantic_preconditions(_request(tmp_path))[0]

    assert semantic["blocking"] is True
    assert "c5_scientific_lock_exists" in semantic["failed_checks"]


def test_a_terminal_but_short_lock_blocks_c6(tmp_path: Path, monkeypatch) -> None:
    """C6's gate follows the verifier's verdict, whatever produced it."""
    monkeypatch.setattr(c5_module, "verify_c5_synthesis_lock",
                        lambda repo, path: {
                            "lock_valid": False, "ok": False, "reason": "VERIFICATION_FAILED",
                            "checks": [{"check_id": "c5_lock_declares_every_candidate_usable",
                                        "ok": False}],
                            "payload": {"every_planned_candidate_is_terminal": True,
                                        "every_planned_candidate_is_usable": False,
                                        "lock_kind": "terminal_audit_record",
                                        "usable_as_c6_input": False}})

    semantic = C6Adapter().semantic_preconditions(_request(tmp_path))[0]

    assert semantic["blocking"] is True
    assert semantic["lock_kind"] == "terminal_audit_record"
    assert semantic["usable_as_c6_input"] is False


def test_a_verified_lock_stops_blocking_c6(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(c5_module, "verify_c5_synthesis_lock",
                        lambda repo, path: {"lock_valid": True, "ok": True,
                                            "reason": "VERIFIED", "checks": [],
                                            "payload": {"usable_as_c6_input": True,
                                                        "lock_kind": "scientific_synthesis"}})

    semantic = C6Adapter().semantic_preconditions(_request(tmp_path))[0]

    assert semantic["blocking"] is False
    assert semantic["present"] is True


def test_c5_and_c6_call_the_identical_verifier() -> None:
    from prism_fas.pipeline.adapters.c6 import C6Adapter as adapter

    c6_source = ast.get_source_segment(
        (REPO / "src" / "prism_fas" / "pipeline" / "adapters" / "c6.py"
         ).read_text(encoding="utf-8"),
        next(node for node in ast.walk(ast.parse(
            (REPO / "src" / "prism_fas" / "pipeline" / "adapters" / "c6.py"
             ).read_text(encoding="utf-8")))
            if isinstance(node, ast.FunctionDef)
            and node.name == "semantic_preconditions")) or ""

    assert "from prism_fas.pipeline.adapters.c5 import" in c6_source
    assert "verify_c5_synthesis_lock" in c6_source
    # No second verifier: C6 must not decide for itself what a valid bank is.
    for reinvented in ("reuse_decision", "record_set_digest", "sha256",
                       "every_planned_candidate_is_usable ==", "read_json"):
        assert reinvented not in c6_source, reinvented
    assert "c5_synthesis_verified" in c6_source


def test_an_unsatisfied_semantic_precondition_blocks_like_a_missing_file(tmp_path: Path) -> None:
    gate = C6Adapter().full_precondition_gate(_request(tmp_path))

    assert gate is not None and gate.status == "BLOCKED"
    assert gate.status_axes.scientific == "BLOCKED"
    assert "c5_synthesis_verified" in gate.summary
    assert any(item["name"] == "c5_synthesis_verified"
               for item in gate.detail["missing_inputs"])


def test_the_presence_only_gate_is_no_longer_the_whole_story() -> None:
    from prism_fas.pipeline.adapters.common import EngineeringAdapter

    source = _function_source(
        "full_precondition_gate",
        (REPO / "src" / "prism_fas" / "pipeline" / "adapters" / "common.py"
         ).read_text(encoding="utf-8"))

    assert "self.semantic_preconditions(request)" in source
    assert EngineeringAdapter().semantic_preconditions(None) == [], (
        "stages that need nothing more are unaffected")


# --- 21-23. the firewall and the frozen repository ---------------------------

def test_the_verifier_opens_no_source_dev_and_no_target_artifact() -> None:
    body = "\n".join(_function_source(name) for name in
                     ("verify_c5_synthesis_lock", "reconstruct_current_c5_inputs",
                      "compare_lock_to_current", "verify_c5_candidates"))

    for forbidden in ("source_dev", "target_test", "siw", "SiW", "label_live_spoof",
                      "_real_target"):
        assert forbidden not in body, forbidden


def test_candidate_verification_reads_only_under_the_candidate_root(small_bank) -> None:
    """It resolves directories by identity beneath one root and reads nothing else."""
    root, plans = small_bank
    opened: list[Path] = []
    original = raw.sha256_file

    def recording(path: Path) -> str:
        opened.append(Path(path))
        return original(path)

    raw.sha256_file = recording
    try:
        c5_module.verify_c5_candidates(root, {"RND": _arm_plan("RND", 4)})
    finally:
        raw.sha256_file = original

    assert opened, "it really did read files"
    assert all(str(path).startswith(str(root)) for path in opened)


def test_version_b_is_untouched() -> None:
    version_b = REPO.parent / "PRISM_FAS_B_Project"
    if not (version_b / ".git").exists():
        pytest.skip("Version B is not checked out beside this repository")

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=version_b,
                          capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=version_b,
                           capture_output=True, text=True, check=True).stdout.strip()

    assert head == "7799f7decd35db6987ce4578824e5bd8d9eab4ae"
    assert dirty == "", f"Version B has uncommitted changes: {dirty}"
