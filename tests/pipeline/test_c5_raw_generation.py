"""C5 raw generation: three arm plans, gate-free identity, resume and retention.

The Version-B `SyntheticBankGenerator` couples generation to quality evaluation
and binds three calibration hashes into its generation identity. Version-C splits
those across the frozen stage boundary — C5 renders, C6 gates — so the Version-C
path stops at finalization and its records bind no calibration. The Version-B
class is untouched and a test here holds that.

Nothing renders. The record layer, the resume decision and the plan arithmetic
are what this file exercises, and none of them needs a GPU or a pixel.
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

from prism_fas.synthesis import c5_arm_plan as arm_module  # noqa: E402
from prism_fas.synthesis import c5_raw_generation as raw  # noqa: E402
from prism_fas.synthesis import c5_source_pair_plan as sp  # noqa: E402
from prism_fas.synthesis.c5_arm_plan import (ArmPlanError, build_all_arm_plans,  # noqa: E402
                                             load_arm_bank)

pytest.importorskip("pyarrow")

CHECKPOINT_SHA = "c" * 64
PHYSICS_VERSION = "m7-physics-v1"


def _package(root: Path) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [{"sample_id": f"{label}_{ds}_{i:03d}", "dataset": ds,
             "source_record_id": f"{label}_{ds}_rec{i:03d}",
             "subject_id": f"subj{i:03d}", "project_split": "source_train",
             "label_live_spoof": label}
            for label, n in (("live", 40), ("spoof", 40)) for i in range(n)
            for ds in [("casia_fasd" if i % 2 == 0 else "msu_mfsd")]]
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict(
        {name: [row[name] for row in rows] for name in rows[0]}),
        root / "manifests" / "source_train.parquet")
    for forbidden in ("source_dev.parquet", "target_test_features.parquet"):
        (root / "manifests" / forbidden).write_bytes(b"opening this is a failure")
    (root / "PACKAGE_LOCK.json").write_text(json.dumps({
        "content_identity_sha256": "b" * 64}), encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def base_plan(tmp_path_factory) -> dict:
    return sp.build_source_pair_plan(_package(tmp_path_factory.mktemp("m3b") / "m3b"))


@pytest.fixture(scope="module")
def plans(base_plan: dict) -> dict:
    return build_all_arm_plans(REPO, base_plan,
                               gpat_checkpoint_sha256=CHECKPOINT_SHA,
                               physics_engine_version=PHYSICS_VERSION)


# --- 7-8, 11-15. the three arm plans -----------------------------------------

@pytest.mark.parametrize("arm", ["RND", "DET", "LLM"])
def test_each_c3_treatment_bank_resolves_with_256_recipes(arm: str) -> None:
    bank = load_arm_bank(REPO, arm)

    assert len(bank["recipes"]) == 256
    assert bank["lock"]["arm"] == arm
    assert bank["lock"]["scientific_eligible"] is True
    assert bank["bank_identity"]


def test_the_neutral_m7_bank_is_not_a_treatment_arm() -> None:
    """M7 is what the shared GPAT generator was TRAINED on. Using it as a
    treatment arm would compare the generator against itself."""
    with pytest.raises(ArmPlanError, match="unknown arm"):
        load_arm_bank(REPO, "prism_recipe_bank_m7_v1")
    assert arm_module.NEUTRAL_SUPPORT_BANK not in [
        arm_module.arm_bank_root(REPO, arm).as_posix() for arm in sp.ARMS]


def test_exactly_2048_per_arm_and_6144_globally(plans: dict) -> None:
    assert {arm: plan["planned_candidates"] for arm, plan in plans.items()} == {
        "RND": 2048, "DET": 2048, "LLM": 2048}
    assert arm_module.global_candidate_count(plans) == 6144

    every_id = [row["candidate_id"] for plan in plans.values()
                for row in plan["candidates"]]
    assert len(set(every_id)) == 6144, "candidate ids must be globally unique"


def test_each_arm_splits_1024_physics_and_1024_gpat(plans: dict) -> None:
    for arm, plan in plans.items():
        routes = [row["route"] for row in plan["candidates"]]
        assert routes.count(sp.PHYSICS) == 1024, arm
        assert routes.count(sp.GPAT) == 1024, arm


def test_the_base_schedule_is_identical_across_arms(plans: dict) -> None:
    """The fairness invariant. Asserted by the module and again here."""
    arm_module.assert_arms_share_the_schedule(plans)

    signature = lambda plan: [(row["position"], row["route"], row["domain_relation"],
                               row["live_target_sample_id"],
                               row["spoof_source_sample_id"])
                              for row in plan["candidates"]]
    assert signature(plans["RND"]) == signature(plans["DET"]) == signature(plans["LLM"])
    assert len({plan["source_pair_plan_identity"] for plan in plans.values()}) == 1
    assert len({plan["arm_plan_identity"] for plan in plans.values()}) == 3


def test_a_divergent_arm_schedule_is_refused(plans: dict) -> None:
    tampered = json.loads(json.dumps(plans))
    tampered["DET"]["candidates"][0]["live_target_sample_id"] = "a different live"

    with pytest.raises(ArmPlanError, match="arm-independent"):
        arm_module.assert_arms_share_the_schedule(tampered)


def test_physics_binds_the_engine_and_gpat_binds_the_checkpoint(plans: dict) -> None:
    for plan in plans.values():
        for row in plan["candidates"]:
            if row["route"] == sp.PHYSICS:
                assert row["generator_binding"] == PHYSICS_VERSION
                assert row["generator_binding"] != CHECKPOINT_SHA
            else:
                assert row["generator_binding"] == CHECKPOINT_SHA


def test_a_bank_with_the_wrong_recipe_count_is_refused(tmp_path: Path) -> None:
    import shutil

    fake = tmp_path / "assets" / "recipe_banks" / "c3" / "rnd"
    fake.mkdir(parents=True)
    shutil.copyfile(REPO / "assets/recipe_banks/c3/rnd/C3_BANK.json",
                    fake / "C3_BANK.json")
    (fake / "recipes.jsonl").write_text('{"recipe_id": "only-one"}\n', encoding="utf-8")

    with pytest.raises(ArmPlanError, match="256"):
        load_arm_bank(tmp_path, "RND")


# --- 16-17. the generation identity binds no gate ----------------------------

def _identity(**overrides) -> raw.GenerationIdentity:
    base = dict(candidate_id="c5syn_abc", arm="RND", arm_plan_identity="armplan",
                source_pair_plan_identity="baseplan", package_identity="b" * 64,
                recipe_bank_identity="bank", recipe_id="r0", recipe_ordinal=0,
                slot=1, position=1, route=sp.GPAT,
                live_target_sample_id="live_0", spoof_source_sample_id="spoof_0",
                generator_binding=CHECKPOINT_SHA, ontology_identity="onto")
    return raw.GenerationIdentity(**{**base, **overrides})


def test_no_calibration_field_exists_in_the_generation_identity() -> None:
    import dataclasses

    fields = {field.name for field in dataclasses.fields(raw.GenerationIdentity)}
    for forbidden in ("threshold_sha256", "fingerprint_reference_sha256",
                      "calibration_sha256", "quality_profile", "accepted",
                      "calibration"):
        assert forbidden not in fields, forbidden
    assert forbidden not in json.dumps(_identity().as_dict())


def test_changing_a_c6_threshold_leaves_resume_valid(tmp_path: Path) -> None:
    """There is no calibration input, so a C6 re-calibration cannot invalidate a
    completed C5 payload. That is what breaks the C5 -> C6 -> C5 cycle."""
    identity = _identity()
    directory = raw.candidate_dir(tmp_path, "RND", identity.candidate_id)
    hashes = _write_fake_payloads(directory)
    raw.write_record(directory, raw.CandidateRecord(
        identity=identity, status=raw.GENERATED, payload_sha256=hashes))

    before = raw.reuse_decision(directory, identity)
    # A C6 re-calibration changes nothing this function can see.
    after = raw.reuse_decision(directory, identity)

    assert before["reusable"] is True and after["reusable"] is True
    assert before == after


@pytest.mark.parametrize("field,value", [
    ("package_identity", "another package"),
    ("recipe_bank_identity", "another bank"),
    ("recipe_id", "another recipe"),
    ("live_target_sample_id", "another live"),
    ("spoof_source_sample_id", "another spoof"),
    ("route", sp.PHYSICS),
    ("generator_binding", "another checkpoint"),
    ("arm", "DET"),
    ("source_pair_plan_identity", "another base"),
])
def test_every_generation_relevant_change_moves_the_identity(field, value) -> None:
    assert _identity(**{field: value}).digest() != _identity().digest()


# --- 18-24. records, resume, corruption, failure retention -------------------

def _write_fake_payloads(directory: Path) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for index, name in enumerate(raw.PAYLOAD_NAMES):
        payload = f"payload-{index}".encode()
        (directory / name).write_bytes(payload)
        hashes[name] = raw.sha256_file(directory / name)
    return hashes


def test_a_valid_completed_candidate_is_reused(tmp_path: Path) -> None:
    identity = _identity()
    directory = raw.candidate_dir(tmp_path, "RND", identity.candidate_id)
    raw.write_record(directory, raw.CandidateRecord(
        identity=identity, status=raw.GENERATED,
        payload_sha256=_write_fake_payloads(directory)))

    decision = raw.reuse_decision(directory, identity)
    assert decision["reusable"] is True
    assert decision["reason"] == "REUSABLE"


def test_an_absent_record_is_not_reuse(tmp_path: Path) -> None:
    identity = _identity()
    decision = raw.reuse_decision(
        raw.candidate_dir(tmp_path, "RND", identity.candidate_id), identity)

    assert decision["reusable"] is False
    assert decision["reason"] == "ABSENT"


def test_a_missing_payload_rebuilds_that_exact_candidate(tmp_path: Path) -> None:
    identity = _identity()
    directory = raw.candidate_dir(tmp_path, "RND", identity.candidate_id)
    raw.write_record(directory, raw.CandidateRecord(
        identity=identity, status=raw.GENERATED,
        payload_sha256=_write_fake_payloads(directory)))
    (directory / raw.IMAGE_NAME).unlink()

    decision = raw.reuse_decision(directory, identity)
    assert decision["reusable"] is False
    assert decision["reason"] == "PAYLOAD_MISSING"
    assert decision["candidate_id"] == identity.candidate_id


def test_a_corrupted_payload_rebuilds_that_exact_candidate(tmp_path: Path) -> None:
    identity = _identity()
    directory = raw.candidate_dir(tmp_path, "RND", identity.candidate_id)
    raw.write_record(directory, raw.CandidateRecord(
        identity=identity, status=raw.GENERATED,
        payload_sha256=_write_fake_payloads(directory)))
    (directory / raw.MASK_NAME).write_bytes(b"tampered")

    decision = raw.reuse_decision(directory, identity)
    assert decision["reusable"] is False
    assert decision["reason"] == "PAYLOAD_CHANGED"
    assert decision["payload"] == raw.MASK_NAME


def test_a_record_from_another_identity_is_stale_not_reused(tmp_path: Path) -> None:
    identity = _identity()
    directory = raw.candidate_dir(tmp_path, "RND", identity.candidate_id)
    raw.write_record(directory, raw.CandidateRecord(
        identity=_identity(recipe_id="a different recipe"), status=raw.GENERATED,
        payload_sha256=_write_fake_payloads(directory)))

    decision = raw.reuse_decision(directory, identity)
    assert decision["reusable"] is False
    assert decision["reason"] == "STALE", (
        "a record for another candidate is never called reused")


def test_a_failure_is_retained_and_never_replaced(tmp_path: Path) -> None:
    identity = _identity()
    directory = raw.candidate_dir(tmp_path, "RND", identity.candidate_id)
    directory.mkdir(parents=True)
    record = raw.failure_record(identity, stage="gpat_forward",
                                error=RuntimeError("CUDA out of memory at D:\\runs\\x"))
    raw.write_record(directory, record)

    payload = raw.read_record(directory / raw.RECORD_NAME)
    assert payload["status"] == raw.FAILED_GENERATION
    assert payload["failure"]["replacement_generated"] is False
    assert "2048" in payload["failure"]["rule"]
    assert "[redacted-path]" in payload["failure"]["sanitized_reason"]
    assert "D:\\runs" not in payload["failure"]["sanitized_reason"]

    decision = raw.reuse_decision(directory, identity)
    assert decision["reusable"] is False
    assert decision["reason"] == "FAILED_GENERATION"


def test_an_interrupted_run_keeps_every_completed_candidate(tmp_path: Path) -> None:
    """Half the positions done, the process gone. All of them still reusable."""
    identities = [_identity(candidate_id=f"c5syn_{index:04d}", position=index)
                  for index in range(10)]
    for identity in identities[:5]:
        directory = raw.candidate_dir(tmp_path, "RND", identity.candidate_id)
        raw.write_record(directory, raw.CandidateRecord(
            identity=identity, status=raw.GENERATED,
            payload_sha256=_write_fake_payloads(directory)))

    decisions = [raw.reuse_decision(
        raw.candidate_dir(tmp_path, "RND", identity.candidate_id), identity)
        for identity in identities]

    assert [item["reusable"] for item in decisions] == [True] * 5 + [False] * 5
    assert {item["reason"] for item in decisions[5:]} == {"ABSENT"}


# --- 25-27. what completion may and may not claim ----------------------------

def _records(generated: int, failed: int = 0) -> list[dict[str, Any]]:
    rows = []
    for index in range(generated):
        identity = _identity(candidate_id=f"ok_{index}", position=index,
                             route=sp.PHYSICS if index % 2 == 0 else sp.GPAT)
        rows.append(raw.CandidateRecord(identity=identity, status=raw.GENERATED,
                                        payload_sha256={name: f"h{index}"
                                                        for name in raw.PAYLOAD_NAMES}
                                        ).as_dict())
    for index in range(failed):
        identity = _identity(candidate_id=f"bad_{index}", position=1000 + index)
        rows.append(raw.failure_record(identity, stage="render",
                                       error=RuntimeError("boom")).as_dict())
    return rows


def test_the_summary_separates_generated_from_failed() -> None:
    summary = raw.summarize(_records(generated=6, failed=2))

    assert summary["records"] == 8
    assert summary["generated"] == 6
    assert summary["failed"] == 2
    assert summary["per_arm"]["RND"] == {"generated": 6, "failed": 2,
                                         "physics": 3, "gpat": 3}


def test_a_failed_candidate_reduces_the_usable_count() -> None:
    """C6 needs payloads. A terminal record is not a payload."""
    summary = raw.summarize(_records(generated=2047, failed=1))

    assert summary["records"] == 2048, "every planned position is accounted for"
    assert summary["generated"] == 2047, "but only 2047 are usable by C6"
    assert summary["generated"] < 2048


def test_the_record_set_digest_changes_when_an_outcome_changes() -> None:
    complete = raw.record_set_digest(_records(generated=8))
    with_failure = raw.record_set_digest(_records(generated=7, failed=1))

    assert complete != with_failure


def test_the_payload_digest_covers_only_generated_candidates() -> None:
    assert raw.payload_set_digest(_records(generated=4, failed=2)) == \
           raw.payload_set_digest(_records(generated=4))


def test_a_record_declares_it_binds_no_calibration() -> None:
    record = raw.CandidateRecord(identity=_identity(), status=raw.GENERATED)

    assert record.as_dict()["binds_quality_calibration"] is False


# --- 33. the Version-B contracts are untouched -------------------------------

def test_the_version_b_generator_and_planner_are_not_imported() -> None:
    import ast

    for name in ("c5_raw_generation", "c5_arm_plan", "c5_source_pair_plan"):
        tree = ast.parse((REPO / "src" / "prism_fas" / "synthesis" / f"{name}.py")
                         .read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not any("candidate_plan" in item for item in imported), name
        assert not any("synthetic_bank" in item for item in imported), name


def test_the_version_b_constants_still_hold() -> None:
    from prism_fas.synthesis import candidate_plan

    assert candidate_plan.EXPECTED_TOTAL == 1120
    assert candidate_plan.EXPECTED_PER_ROUTE == 560


def test_no_evaluator_or_calibration_is_reachable_from_the_version_c_path() -> None:
    source = (REPO / "src" / "prism_fas" / "synthesis" / "c5_raw_generation.py"
              ).read_text(encoding="utf-8")
    body = source.split('"""', 2)[2]          # past the module docstring

    for forbidden in ("CandidateEvaluator", "FrozenCalibration", "quality_gate",
                      "evaluate("):
        assert forbidden not in body, forbidden
