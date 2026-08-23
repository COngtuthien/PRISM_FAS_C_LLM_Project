"""The planned schedule partitions exactly, semantic failures included.

`_build_matched_banks` passed `[]` where `provenance_closure` expects
`semantic_failure_ids`. `candidate_pool()` is built from the frozen PLAN, so its
keys include every planned slot — including the ones that ended in a terminal
`SemanticGenerationFailure`. Those slots have no payload, were never measured
and therefore carry no gate decision, so with an empty failure set they landed in
`planned` but never in `covered` and the closure was False. All three bank locks
failed for that reason, on a run whose profile assessment, selection and bank
construction were all valid.

The fix is producer-side, as with the calibration-hash defect before it: the
strict verifier is right to reject inconsistent evidence, so the evidence is made
truthful. The failure ids are re-derived by `verify_c5_candidates` while it
already validates every planned slot — not copied from a lock, and never inferred
from a missing decision, because an absent decision could equally mean a pipeline
defect.

Nothing here may move a scientific result. The invariance tests below pin that.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.pipeline.adapters import c5 as c5_module  # noqa: E402
from prism_fas.pipeline.adapters import AdapterRequest  # noqa: E402
from prism_fas.pipeline.adapters.c6 import C6Adapter  # noqa: E402
from prism_fas.pipeline.profiles import load_profile  # noqa: E402
from prism_fas.synthesis import c5_raw_generation as raw  # noqa: E402
from prism_fas.synthesis import c5_render as render_module  # noqa: E402
from prism_fas.synthesis import c6_matched_bank as selector  # noqa: E402
from prism_fas.synthesis import c6_scientific as science  # noqa: E402
from prism_fas.synthesis.c5_render import SemanticGenerationFailure  # noqa: E402
from prism_fas.synthesis.c5_source_pair_plan import ARMS, GPAT, PHYSICS  # noqa: E402

PLANNED = ["c0", "c1", "c2", "c3", "c4", "c5_failed"]
DECISIONS = ([{"candidate_id": name, "accepted": True}
              for name in ("c0", "c1", "c2", "c3")]
             + [{"candidate_id": "c4", "accepted": False}])
SELECTED = ["c0", "c1", "c2"]


# --- 1-8. the partition contract ---------------------------------------------

def test_an_all_generated_pool_closes() -> None:
    planned = ["c0", "c1", "c2"]
    decisions = [{"candidate_id": name, "accepted": True} for name in planned]

    closure = science.provenance_closure(planned, [], decisions, ["c0"])

    assert closure["closed"] is True
    assert closure["semantic_failed"] == 0


def test_a_pool_with_a_verified_semantic_failure_closes() -> None:
    closure = science.provenance_closure(PLANNED, ["c5_failed"], DECISIONS, SELECTED)

    assert closure["closed"] is True
    assert closure["semantic_failed"] == 1
    assert closure["selected"] == 3
    assert closure["accepted_not_selected"] == 1
    assert closure["rejected"] == 1
    assert closure["covered"] == closure["planned"] == 6


def test_omitting_the_semantic_failure_does_not_close() -> None:
    """The exact defect: `[]` where the failure ids belong."""
    closure = science.provenance_closure(PLANNED, [], DECISIONS, SELECTED)

    assert closure["closed"] is False
    assert closure["unaccounted"] == ["c5_failed"]


def test_a_fabricated_failure_id_does_not_close() -> None:
    closure = science.provenance_closure(PLANNED, ["ghost"], DECISIONS, SELECTED)

    assert closure["closed"] is False
    assert "c5_failed" in closure["unaccounted"]


def test_a_semantic_failure_carrying_a_gate_decision_does_not_close() -> None:
    """It has no payload, so it can never have been measured."""
    closure = science.provenance_closure(PLANNED, ["c5_failed", "c4"], DECISIONS,
                                         SELECTED)

    assert closure["closed"] is False
    assert closure["semantic_failures_carrying_a_gate_decision"] == ["c4"]


def test_a_selected_candidate_outside_accepted_does_not_close() -> None:
    closure = science.provenance_closure(PLANNED, ["c5_failed"], DECISIONS,
                                         ["c0", "c4"])

    assert closure["closed"] is False
    assert closure["selected_outside_accepted"] == ["c4"]


def test_an_unaccounted_planned_candidate_does_not_close() -> None:
    closure = science.provenance_closure(PLANNED + ["c6_lost"], ["c5_failed"],
                                         DECISIONS, SELECTED)

    assert closure["closed"] is False
    assert closure["unaccounted"] == ["c6_lost"]


def test_duplicate_category_membership_does_not_close() -> None:
    closure = science.provenance_closure(PLANNED, ["c5_failed", "c0"], DECISIONS,
                                         SELECTED)

    assert closure["closed"] is False
    assert closure["pairwise_disjoint"] is False
    assert "selected&semantic_failed" in closure["category_overlaps"]


# --- 9, 11-13. the C5 verifier re-derives the ids ----------------------------

def _arm_plan(arm: str, count: int = 8) -> dict[str, Any]:
    return {"arm": arm, "arm_plan_identity": f"armplan-{arm}",
            "source_pair_plan_identity": "baseplan", "package_identity": "b" * 64,
            "recipe_bank_identity": f"bank-{arm}", "ontology_identity": "onto",
            "planned_candidates": count, "binds_quality_calibration": False,
            "candidates": [{
                "candidate_id": f"c5syn_{arm.lower()}_{index:04d}", "arm": arm,
                "route": PHYSICS if index % 2 == 0 else GPAT, "position": index,
                "recipe_id": f"r{index}", "recipe_ordinal": index,
                "slot": (index % 8) + 1, "domain_relation": "same_domain",
                "live_dataset": "casia_fasd" if index % 2 == 0 else "msu_mfsd",
                "live_target_sample_id": f"live_{index:04d}",
                "spoof_source_sample_id": None,
                "recipe_bank_identity": f"bank-{arm}",
                "generator_binding": "m7-physics-v1"} for index in range(count)]}


def _materialize(root: Path, plans: dict[str, dict[str, Any]],
                 failures_per_arm: int) -> dict[str, list[str]]:
    """Write records; the first N Physics slots of each arm fail semantically."""
    import hashlib

    expected: dict[str, list[str]] = {}
    for arm, plan in plans.items():
        budget = failures_per_arm
        for row in plan["candidates"]:
            identity = render_module.identity_for(row, plan)
            directory = raw.candidate_dir(root, arm, identity.candidate_id)
            directory.mkdir(parents=True, exist_ok=True)
            if row["route"] == PHYSICS and budget > 0:
                budget -= 1
                expected.setdefault(arm, []).append(identity.candidate_id)
                record = raw.failure_record(
                    identity, stage="render_physics",
                    error=SemanticGenerationFailure("empty exact mask"))
            else:
                hashes = {}
                for index, name in enumerate(raw.PAYLOAD_NAMES):
                    payload = f"{identity.candidate_id}:{index}".encode()
                    (directory / name).write_bytes(payload)
                    hashes[name] = hashlib.sha256(payload).hexdigest()
                record = raw.CandidateRecord(identity=identity,
                                             status=raw.GENERATED,
                                             payload_sha256=hashes)
            (directory / raw.RECORD_NAME).write_text(
                json.dumps(record.as_dict()), encoding="utf-8")
    return {arm: sorted(ids) for arm, ids in expected.items()}


@pytest.fixture(scope="module")
def verified(tmp_path_factory) -> tuple[dict[str, Any], dict[str, list[str]]]:
    root = tmp_path_factory.mktemp("pool")
    plans = {arm: _arm_plan(arm) for arm in ARMS}
    expected = _materialize(root, plans, failures_per_arm=2)
    return c5_module.verify_c5_candidates(root, plans), expected


def test_the_verifier_returns_the_exact_semantic_failure_ids(verified) -> None:
    result, expected = verified

    assert result["semantic_failed_candidate_ids_by_arm"] == expected
    assert result["semantic_failed"] == 6
    assert len(result["semantic_failed_candidate_ids"]) == 6


@pytest.mark.parametrize("arm", list(ARMS))
def test_each_arm_closes_with_its_own_verified_failures(verified, arm) -> None:
    result, _ = verified
    ids = result["semantic_failed_candidate_ids_by_arm"][arm]
    planned = [f"c5syn_{arm.lower()}_{index:04d}" for index in range(8)]
    generated = [name for name in planned if name not in ids]
    decisions = [{"candidate_id": name, "accepted": index < 4}
                 for index, name in enumerate(generated)]
    selected = [name for index, name in enumerate(generated) if index < 2]

    closure = science.provenance_closure(planned, ids, decisions, selected)

    assert len(ids) == 2
    assert closure["closed"] is True
    assert closure["semantic_failed"] == 2


def test_the_ids_are_sorted_and_stable(verified) -> None:
    result, _ = verified

    for arm, ids in result["semantic_failed_candidate_ids_by_arm"].items():
        assert ids == sorted(ids), arm
    assert result["semantic_failed_candidate_ids"] == sorted(
        result["semantic_failed_candidate_ids"])


def test_an_invalid_failure_record_is_not_counted_as_a_semantic_failure(
        tmp_path: Path) -> None:
    """Only records the verifier validates become provenance ids."""
    plans = {arm: _arm_plan(arm, count=4) for arm in ARMS}
    _materialize(tmp_path, plans, failures_per_arm=1)
    row = next(item for item in plans["RND"]["candidates"]
               if item["route"] == PHYSICS)
    directory = raw.candidate_dir(tmp_path, "RND", row["candidate_id"])
    record = raw.read_record(directory / raw.RECORD_NAME)
    record["failure"]["replacement_generated"] = True
    (directory / raw.RECORD_NAME).write_text(json.dumps(record), encoding="utf-8")

    result = c5_module.verify_c5_candidates(tmp_path, plans)

    assert row["candidate_id"] not in result["semantic_failed_candidate_ids"]
    assert result["problems"][0]["reason"] == "INVALID_FAILURE_RECORD"


# --- 10. C6 consumes only verified ids ---------------------------------------

def test_c6_uses_the_verified_ids_and_not_an_empty_list() -> None:
    source = inspect.getsource(C6Adapter._build_matched_banks)

    assert 'state["semantic_failure_ids_by_arm"].get(arm, [])' in source
    assert "[], state[\"decisions\"][profile][arm]" not in source


def test_c6_takes_the_ids_from_the_strict_c5_verification() -> None:
    source = inspect.getsource(C6Adapter._verify_c5_pool)

    assert '"semantic_failure_ids_by_arm"' in source
    assert "semantic_failed_candidate_ids_by_arm" in source


def test_failure_membership_is_never_inferred_from_a_missing_decision() -> None:
    source = inspect.getsource(science.provenance_closure)

    assert "semantic_failure_ids" in inspect.signature(
        science.provenance_closure).parameters
    # The closure never derives failures from set arithmetic on decisions.
    assert "planned - decided" not in source
    assert "accepted | rejected" in source, "decisions are used only to CHECK"


# --- 14-18. result invariance -------------------------------------------------

def _banks(plans: dict[str, dict[str, Any]], accepted_ids: dict[str, list[str]]
           ) -> dict[str, Any]:
    by_arm = {arm: [selector.SelectableCandidate(
        candidate_id=row["candidate_id"], arm=arm, route=row["route"],
        source_domain=row["live_dataset"], recipe_id=row["recipe_id"],
        recipe_ordinal=row["recipe_ordinal"],
        live_target_sample_id=row["live_target_sample_id"],
        base_position=row["position"], q=0.5)
        for row in plans[arm]["candidates"]
        if row["candidate_id"] in set(accepted_ids[arm])] for arm in ARMS}
    return selector.build_matched_banks(plans, by_arm)


def test_provenance_accounting_changes_no_selected_set() -> None:
    """The whole point: this hotfix may not move a scientific result."""
    planned = PLANNED
    with_failures = science.provenance_closure(planned, ["c5_failed"], DECISIONS,
                                               SELECTED)
    without = science.provenance_closure(planned, [], DECISIONS, SELECTED)

    # Only the accounting fields differ; the selected set is an input, untouched.
    assert with_failures["selected"] == without["selected"] == 3
    assert with_failures["rejected"] == without["rejected"] == 1
    assert with_failures["accepted_not_selected"] == without["accepted_not_selected"]
    assert with_failures["closed"] is True and without["closed"] is False
    assert with_failures["semantic_failed"] == 1 and without["semantic_failed"] == 0


def test_the_closure_is_a_pure_reader_and_selects_nothing() -> None:
    source = inspect.getsource(science.provenance_closure)

    for forbidden in ("select_route_bank", "build_matched_banks", "derive_profile",
                      "gate_candidates", "q"):
        assert f"{forbidden}(" not in source, forbidden


def test_the_selector_identity_does_not_depend_on_provenance() -> None:
    contract = selector.selector_identity(
        quality_profile_identity="p" * 64, c5_pool_lock_sha256="a" * 64,
        decision_set_sha256="b" * 64)
    again = selector.selector_identity(
        quality_profile_identity="p" * 64, c5_pool_lock_sha256="a" * 64,
        decision_set_sha256="b" * 64)

    assert contract["selector_identity_sha256"] == again["selector_identity_sha256"]
    assert "provenance" not in json.dumps(contract)


def test_the_decision_set_digest_ignores_semantic_failures() -> None:
    """Failures hold no decision, so they cannot enter the decision identity."""
    base = selector.decision_set_digest(DECISIONS)

    assert base == selector.decision_set_digest(list(reversed(DECISIONS)))


def test_the_strict_lock_verifier_was_not_weakened() -> None:
    source = inspect.getsource(C6Adapter._verify_c6_locks)

    for requirement in ("selected_set_sha256", "selector_identity_sha256",
                        "final_bank_size", "q_used_for_selection",
                        "provenance_closure"):
        assert requirement in source, requirement
    assert 'provenance_closure", {}).get("closed")' in source


# --- the real production path -------------------------------------------------

def _full_plan(arm: str) -> dict[str, Any]:
    """A frozen 2048-slot arm plan, 1024 per route."""
    rows = []
    position = 0
    for route in (PHYSICS, GPAT):
        for index in range(1024):
            rows.append({
                "candidate_id": f"c5syn_{arm.lower()}_{position:05d}", "arm": arm,
                "route": route, "position": position,
                "recipe_id": f"r{index % 256}", "recipe_ordinal": index % 256,
                "live_dataset": "casia_fasd" if index % 2 == 0 else "msu_mfsd",
                "live_target_sample_id": f"live_{index % 320:04d}"})
            position += 1
    return {"arm": arm, "arm_plan_identity": f"armplan-{arm}",
            "source_pair_plan_identity": "baseplan", "package_identity": "b" * 64,
            "recipe_bank_identity": f"bank-{arm}",
            "recipe_bank_root": f"assets/recipe_banks/c3/{arm.lower()}",
            "selected_set_identity": f"selected-{arm}", "ontology_identity": "onto",
            "planned_candidates": len(rows), "binds_quality_calibration": False,
            "candidates": rows}


@pytest.fixture(scope="module")
def production(tmp_path_factory) -> dict[str, Any]:
    """Drive the real `_build_matched_banks` then `_verify_c6_locks`.

    Every planned slot lands in exactly one class: a handful of Physics slots per
    arm are terminal semantic failures, some generated candidates are rejected,
    and the rest are accepted and available to the selector.
    """
    root = tmp_path_factory.mktemp("prod")
    reports = root / "reports" / "full" / "c6"
    reports.mkdir(parents=True, exist_ok=True)
    plans = {arm: _full_plan(arm) for arm in ARMS}
    failures = {arm: sorted(row["candidate_id"] for row in plans[arm]["candidates"]
                            if row["route"] == PHYSICS)[:5] for arm in ARMS}

    decisions: dict[str, list[dict[str, Any]]] = {}
    accepted: dict[str, list[selector.SelectableCandidate]] = {}
    for arm in ARMS:
        failed = set(failures[arm])
        rows, keep = [], []
        for index, row in enumerate(plans[arm]["candidates"]):
            if row["candidate_id"] in failed:
                continue                       # no payload, so never measured
            ok = index % 50 != 0               # a few rejects, plenty accepted
            rows.append({"candidate_id": row["candidate_id"], "accepted": ok,
                         "q": 0.5, "failed_gates": []})
            if ok:
                keep.append(selector.SelectableCandidate(
                    candidate_id=row["candidate_id"], arm=arm, route=row["route"],
                    source_domain=row["live_dataset"], recipe_id=row["recipe_id"],
                    recipe_ordinal=row["recipe_ordinal"],
                    live_target_sample_id=row["live_target_sample_id"],
                    base_position=row["position"], q=0.5))
        decisions[arm], accepted[arm] = rows, keep

    state: dict[str, Any] = {
        "plans": plans, "decision": science.ProfileDecision(selected="NOMINAL"),
        "accepted": {"NOMINAL": accepted}, "decisions": {"NOMINAL": decisions},
        "threshold_identities": {"NOMINAL": "t" * 64},
        "c5_pool_lock_sha256": "a" * 64,
        "selectable": science.candidate_pool(plans),
        "semantic_failure_ids_by_arm": failures}

    request = AdapterRequest(repo=root, profile=load_profile("full", repo=REPO))
    adapter = C6Adapter()
    build = adapter._build_matched_banks(request, state, reports)
    verify = adapter._verify_c6_locks(request, state, reports)
    return {"build": build, "verify": verify, "state": state, "reports": reports,
            "failures": failures}


def test_the_production_path_builds_three_matched_banks(production) -> None:
    build = production["build"]

    assert build.status_axes.engineering != "BLOCKED", build.summary
    assert [item["check_id"] for item in build.checks if not item["ok"]] == []
    for arm in ARMS:
        bank = production["state"]["banks"]["banks"][arm]
        assert bank["size"] == 1024
        assert bank["by_route"] == {PHYSICS: 512, GPAT: 512}


def test_the_production_bank_locks_verify(production) -> None:
    """The exact substage that failed on the GPU, for all three arms."""
    verify = production["verify"]
    failed = [item["check_id"] for item in verify.checks if not item["ok"]]

    assert failed == [], failed
    for arm in ARMS:
        payload = json.loads((production["reports"] / f"C6_BANK_LOCK_{arm}.json")
                             .read_text(encoding="utf-8"))
        assert payload["provenance_closure"]["closed"] is True


def test_the_production_closure_accounts_for_every_planned_slot(production) -> None:
    for arm in ARMS:
        payload = json.loads((production["reports"] / f"C6_BANK_LOCK_{arm}.json")
                             .read_text(encoding="utf-8"))
        closure = payload["provenance_closure"]

        assert closure["planned"] == 2048
        assert closure["covered"] == 2048
        assert closure["semantic_failed"] == 5
        assert closure["selected"] == 1024
        assert (closure["selected"] + closure["accepted_not_selected"]
                + closure["rejected"] + closure["semantic_failed"]) == 2048
        assert closure["pairwise_disjoint"] is True
        assert closure["unaccounted"] == []


def test_the_production_locks_still_bind_their_identities(production) -> None:
    for arm in ARMS:
        payload = json.loads((production["reports"] / f"C6_BANK_LOCK_{arm}.json")
                             .read_text(encoding="utf-8"))
        bank = production["state"]["banks"]["banks"][arm]

        assert payload["selected_set_sha256"] == bank["selected_set_sha256"]
        assert payload["selector_identity_sha256"] == production["state"][
            "selector_contract"]["selector_identity_sha256"]
        assert payload["final_bank_size"] == 1024
        assert payload["q_used_for_selection"] is False


def test_the_production_banks_are_unchanged_by_the_provenance_fix(production) -> None:
    """Re-run the selector directly: identical selected sets, identity and quotas."""
    outcome = selector.build_matched_banks(
        production["state"]["plans"], production["state"]["accepted"]["NOMINAL"])

    for arm in ARMS:
        assert (outcome["banks"][arm]["selected_set_sha256"]
                == production["state"]["banks"]["banks"][arm]["selected_set_sha256"])
    assert (outcome["route_quotas"][PHYSICS]["quota"]
            == production["state"]["banks"]["route_quotas"][PHYSICS]["quota"])


# --- 19. the firewall ---------------------------------------------------------

def test_provenance_accounting_opens_no_target_artifact() -> None:
    source = (inspect.getsource(science.provenance_closure)
              + inspect.getsource(C6Adapter._build_matched_banks))

    for forbidden in ("siw", "SiW", "target_test", "label_live_spoof",
                      "source_dev"):
        assert forbidden not in source, forbidden


def test_no_c5_candidate_is_touched_by_the_fix() -> None:
    for relative in ("src/prism_fas/synthesis/c6_scientific.py",
                     "src/prism_fas/pipeline/adapters/c6.py"):
        source = (REPO / relative).read_text(encoding="utf-8")
        for forbidden in ("write_payload_bytes", "failure_record(", "render_arm",
                          "write_record("):
            assert forbidden not in source, f"{relative}: {forbidden}"
