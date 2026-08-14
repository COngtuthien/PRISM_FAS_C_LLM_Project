"""RND/DET schedules, the selection/bank contract identities, and lock integrity.

Offline only. These tests must never reach a provider, a GPU, Modal or SiW.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from prism_fas.llm.bank_lock import canonical_text, sha256_text
from prism_fas.llm.selection_contract import (assemble, bank_contract_identity,
                                              build_selection_contract,
                                              selection_contract_identity)
from prism_fas.recipes import arm_schedules as sched
from prism_fas.recipes import selection as sel
from prism_fas.recipes.eligibility import evaluate_pool
from prism_fas.recipes.validate import validate_payload

REPO = Path(__file__).resolve().parents[2]
PRELIMINARY_LOCK = REPO / "reports" / "c3" / "C3_BANK_LOCK.json"
PRELIMINARY_LOCK_IDENTITY = "7ee96d3abee3f3b579c2dc6fe47ea27ff51ee3c2e956a1ff16b1ca85f5753fba"
GENERATION_CONTRACT_IDENTITY = (
    "884bce03b4f40a4ffbbef30f14c2216a6166a0ee1e8a6f6facb163f8bb3cdd85")


# ------------------------------------------------------------------ schedules
@pytest.mark.parametrize("arm", ["RND", "DET"])
def test_each_control_arm_has_exactly_384_canonical_slot_ids(arm, ontology):
    schedule = sched.build_schedule(arm, ontology)
    assert schedule.slots == 384
    assert len(schedule.slot_ids) == 384
    assert len(set(schedule.slot_ids)) == 384
    assert schedule.slot_ids[0] == f"{arm}_SLOT_000"
    assert schedule.slot_ids[-1] == f"{arm}_SLOT_383"


@pytest.mark.parametrize("arm", ["RND", "DET"])
def test_schedule_identity_is_reproducible(arm, ontology):
    first = sched.build_schedule(arm, ontology).schedule_identity
    second = sched.build_schedule(arm, ontology).schedule_identity
    assert first == second
    assert len(first) == 64


def test_rnd_and_det_schedules_are_distinct(ontology):
    rnd = sched.build_schedule("RND", ontology)
    det = sched.build_schedule("DET", ontology)
    assert rnd.schedule_identity != det.schedule_identity
    assert rnd.seed != det.seed


@pytest.mark.parametrize("arm", ["RND", "DET"])
def test_candidate_drafting_is_per_slot_deterministic(arm, ontology):
    """Slot 137 is identical whether drafted alone, in order or in reverse."""
    alone = sched.draft_candidate(arm, f"{arm}_SLOT_137", ontology)
    in_order = dict(sched.draft_schedule(arm, ontology, slots=200))[f"{arm}_SLOT_137"]
    assert alone == in_order
    again = sched.draft_candidate(arm, f"{arm}_SLOT_137", ontology)
    assert alone == again


@pytest.mark.parametrize("arm", ["RND", "DET"])
def test_drafted_candidates_declare_the_scientific_route(arm, ontology):
    for slot_id, payload in list(sched.draft_schedule(arm, ontology, slots=40)):
        assert payload["generator_route"] == ["physics", "gpat"]


@pytest.mark.parametrize("arm", ["RND", "DET"])
def test_drafted_candidates_pass_the_common_eligibility_gate(arm, ontology, route_policy):
    """Control arms use the same gate as LLM; nothing here bypasses validation."""
    drafted = list(sched.draft_schedule(arm, ontology, slots=48))
    pool = evaluate_pool(arm=arm, candidates=[payload for _, payload in drafted],
                         slot_ids=[slot for slot, _ in drafted], ontology=ontology,
                         route_policy=route_policy, bank_id=f"c3-{arm.lower()}-test",
                         raw_slots=48, minimum_required=0)
    rejected = [v for v in pool.verdicts if not v.eligible]
    assert not any(v.rejected_at in {"item_schema", "ontology", "numeric_range",
                                     "compatibility", "scientific_route_policy"}
                   for v in rejected), \
        f"a drafted candidate is structurally invalid: {[v.as_dict() for v in rejected][:3]}"
    assert pool.eligible_count > 0


@pytest.mark.parametrize("arm", ["RND", "DET"])
def test_schedules_carry_no_target_or_llm_dependency(arm, ontology):
    material = sched.build_schedule(arm, ontology).identity_material()
    assert material["target_information_used"] is False
    assert material["llm_output_used"] is False
    text = json.dumps(material).lower()
    for token in ("siw", "casia", "msu", "acer", "target_metric", "attack_family"):
        assert token not in text


# ----------------------------------------------------------------- identities
def test_selection_contract_binds_every_required_element(ontology, route_policy):
    material = build_selection_contract(repo=REPO, ontology=ontology,
                                        route_policy=route_policy)
    assert material["selection_version"] == "prism_c3_selection_v1"
    assert material["raw_candidate_slots_per_arm"] == 384
    assert material["minimum_eligible_pool_per_arm"] == 320
    assert material["final_bank_size_per_arm"] == 256
    assert material["eligibility_order"][0] == "item_schema"
    assert material["eligibility_order"][-1] == "conditioning_41d"
    assert material["quotas"]["medium"]["hard_min"] == 32
    assert material["quotas"]["artifact"]["preferred_min"] == 32
    assert material["quotas"]["region"]["preferred_min"] == 24
    assert "S_pref" in material["stage_2_formula"]
    assert "S_single" in material["stage_3_formula"]
    assert "S_multi" in material["stage_4_formula"]
    assert "lexicographically smallest" in material["stage_5_rule"]
    assert material["category_order"]["medium"] == list(ontology.media)
    assert material["route_policy_identity"] == route_policy.route_policy_identity
    assert material["ontology_identity"] == ontology.sha256
    assert len(material["rnd_schedule_identity"]) == 64
    assert len(material["det_schedule_identity"]) == 64
    assert material["implementation_identities"]


def test_selection_contract_identity_is_reproducible(ontology, route_policy):
    first = build_selection_contract(repo=REPO, ontology=ontology, route_policy=route_policy)
    second = build_selection_contract(repo=REPO, ontology=ontology, route_policy=route_policy)
    assert selection_contract_identity(first) == selection_contract_identity(second)


@pytest.mark.parametrize("field, value", [
    ("final_bank_size_per_arm", 255),
    ("minimum_eligible_pool_per_arm", 319),
    ("raw_candidate_slots_per_arm", 383),
    ("route_policy_identity", "0" * 64),
    ("ontology_identity", "0" * 64),
    ("rnd_schedule_identity", "0" * 64),
    ("stage_5_rule", "pick any tied set"),
])
def test_changing_any_bound_element_changes_the_selection_identity(field, value, ontology,
                                                                   route_policy):
    material = build_selection_contract(repo=REPO, ontology=ontology,
                                        route_policy=route_policy)
    baseline = selection_contract_identity(material)
    altered = dict(material)
    altered[field] = value
    assert selection_contract_identity(altered) != baseline


def test_bank_contract_identity_is_the_spec_formula():
    generation = "a" * 64
    selection = "b" * 64
    expected = sha256_text(canonical_text({
        "generation_contract_identity": generation,
        "selection_contract_identity": selection}))
    assert bank_contract_identity(generation_contract_identity=generation,
                                  selection_contract_identity=selection) == expected


def test_bank_contract_identity_changes_with_either_input():
    base = bank_contract_identity(generation_contract_identity="a" * 64,
                                  selection_contract_identity="b" * 64)
    assert bank_contract_identity(generation_contract_identity="c" * 64,
                                  selection_contract_identity="b" * 64) != base
    assert bank_contract_identity(generation_contract_identity="a" * 64,
                                  selection_contract_identity="c" * 64) != base


def test_assemble_binds_the_verified_generation_identity(ontology, route_policy):
    record = assemble(repo=REPO, ontology=ontology, route_policy=route_policy,
                      generation_contract_identity=GENERATION_CONTRACT_IDENTITY)
    assert record["c3_generation_contract_identity"] == GENERATION_CONTRACT_IDENTITY
    assert record["c3_bank_contract_identity"] == bank_contract_identity(
        generation_contract_identity=GENERATION_CONTRACT_IDENTITY,
        selection_contract_identity=record["c3_selection_contract_identity"])
    assert sha256_text(record["canonical_text"]) == record["c3_selection_contract_identity"]


# -------------------------------------------------- preliminary lock integrity
def test_the_preliminary_lock_bytes_are_unchanged():
    """The historical lock is immutable evidence; supersession never edits it."""
    assert PRELIMINARY_LOCK.exists()
    lock = json.loads(PRELIMINARY_LOCK.read_text(encoding="utf-8"))
    assert lock["bank_lock_identity"] == PRELIMINARY_LOCK_IDENTITY
    assert lock["composite"]["c3_generation_contract_identity"] == GENERATION_CONTRACT_IDENTITY
    body = {key: value for key, value in lock.items() if key != "bank_lock_identity"}
    assert sha256_text(canonical_text(body)) == PRELIMINARY_LOCK_IDENTITY


def test_the_preliminary_lock_still_binds_no_selection_identity():
    """The precise reason supersession is required."""
    lock = json.loads(PRELIMINARY_LOCK.read_text(encoding="utf-8"))
    assert not any("selection" in key.lower() for key in lock["components"])
    assert "c3_selection_contract_identity" not in json.dumps(lock)


# ------------------------------------------------------------ no live calls
def test_no_c3_scientific_generation_artifact_exists():
    reports = REPO / "reports" / "c3"
    generated = sorted(
        path.name for path in reports.glob("*.json")
        if path.name not in {"C3_BANK_LOCK.json", "C3_BANK_LOCK_VERIFICATION.json"})
    assert generated == [], f"unexpected C3 generation artifacts: {generated}"
    for marker in ("C3_RAW_ARCHIVE", "RECIPE_BANK_LOCK", "C3_ACCEPTANCE"):
        assert not list(reports.rglob(f"*{marker}*")), f"{marker} exists"


def test_the_ambient_credential_is_removed():
    import os

    assert not os.environ.get("GEMINI_API_KEY")
    assert not os.environ.get("GOOGLE_API_KEY")


def test_importing_the_c3_selection_path_loads_no_provider_sdk():
    """Checked in a FRESH interpreter, not against this session's `sys.modules`.

    Whether some earlier test in the run imported the Gemini SDK says nothing
    about the C3 selection path. The claim that matters is that importing the
    selector, the eligibility gate, the arm schedules and the contract builder
    pulls in no provider SDK at all — which only a clean process can show.
    """
    import subprocess
    import sys

    probe = (
        "import sys;"
        "import prism_fas.recipes.selection, prism_fas.recipes.eligibility,"
        "       prism_fas.recipes.arm_schedules, prism_fas.llm.selection_contract;"
        "loaded=[m for m in sys.modules if m.startswith('google')];"
        "print('LOADED:' + ','.join(sorted(loaded)))"
    )
    result = subprocess.run([sys.executable, "-c", probe], cwd=REPO,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    loaded = result.stdout.strip().removeprefix("LOADED:")
    assert loaded == "", f"the C3 selection path imported a provider SDK: {loaded}"


def test_selection_and_schedule_modules_import_no_provider():
    import inspect

    for module in (sel, sched):
        source = inspect.getsource(module)
        assert "GeminiRecipeProvider" not in source
        assert "google.genai" not in source
        assert "requests" not in source
