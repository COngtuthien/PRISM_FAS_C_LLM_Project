"""Which candidate-cardinality contract governs Version-C C5, and why.

The tree carries two sets of constants that look like they disagree:

    gate_profiles.py        3 arms x 2048 candidates/arm -> 1024 accepted/arm
    candidate_plan.py       280 live x (2 physics + 2 gpat) = 1120 total

They are not in conflict. They belong to two different experiments, and this
file pins that reading so neither can drift into the other.

`candidate_plan.py` and `configs/synthesis/synthetic_bank_m8.yaml` are the
Version-B inherited M8 synthesis: one bank, no arm dimension anywhere, keyed on
LIVE SAMPLES (`candidate_recipes_per_live`), `bank_id_prefix:
prism_synthetic_bank_m8_v1`. Version-B froze `prism_synthetic_bank_m8_v3_
e84c78cd2a9b` under it.

The Version-C v1.5 specification states the C5 budget in §10.4 and repeats it in
§11.3, §11.4 and both stage rows. It is keyed on RECIPES, and it has three arms.
The numbers 1120, 280 and 560 do not appear in that specification at all.

So `gate_profiles.py` is the transcription of the Version-C contract, and the
shipped C3 banks agree with it: exactly 256 recipes per arm, from 384 raw slots.
"""
from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.synthesis import candidate_plan, gate_profiles  # noqa: E402

SPEC = (REPO / "docs" /
        "PRISM_FAS_C_LLM_v1_5_FINAL_ComputeConstrained_FullPipeline_Spec_2026.docx")


@pytest.fixture(scope="module")
def spec_text() -> str:
    """The frozen specification's own words, read from the shipped .docx."""
    if not SPEC.is_file():                                  # pragma: no cover
        pytest.skip("the frozen specification is not in this checkout")
    import html

    body = zipfile.ZipFile(SPEC).read("word/document.xml").decode("utf-8")
    return html.unescape(re.sub(r"<[^>]+>", "", re.sub(r"</w:p>", "\n", body)))


# --- what the frozen specification actually says -----------------------------

def test_the_spec_fixes_the_render_budget_at_2048_per_arm(spec_text: str) -> None:
    """§10.4, the C5 clause, stated in one sentence."""
    assert ("2048 candidate renders per arm = 256 recipes × 8 renders/recipe"
            in spec_text)
    assert "exactly 4 Physics and 4 GPAT candidates per recipe" in spec_text
    assert "Final accepted bank is exactly 1024/arm = 512 Physics + 512 GPAT" in spec_text


def test_the_spec_repeats_it_in_every_dependent_clause(spec_text: str) -> None:
    """Not one sentence that could be a typo — the same numbers throughout."""
    occurrences = spec_text.count("2048")
    assert occurrences >= 5, occurrences
    assert "1024 accepted/arm from 2048 candidates/arm" in spec_text
    assert ("Generate 2048 candidate renders/arm (256 recipes × 8 renders: "
            "4 Physics + 4 GPAT)" in spec_text)


def test_the_spec_never_mentions_the_version_b_cardinalities(spec_text: str) -> None:
    """The decisive asymmetry: one contract is in the spec and the other is not."""
    for absent in ("1120", "560"):
        assert absent not in spec_text, (
            f"{absent} appears in the Version-C specification; the reconciliation "
            "recorded here would need revisiting")


def test_the_spec_declares_three_arms(spec_text: str) -> None:
    assert "RND/DET/LLM" in spec_text
    assert "384 raw candidate slots per arm" in spec_text
    assert "256 recipes/arm" in spec_text or "256 recipes per arm" in spec_text


def test_the_spec_puts_rendering_in_c5_and_gating_in_c6(spec_text: str) -> None:
    """The stage boundary, which is what makes a C5 quality gate wrong."""
    assert "C5 — Synthesis integration" in spec_text
    assert "Physics/GPAT render for 3 arms" in spec_text
    assert "C6 — Quality + matched banks" in spec_text
    assert "Gate candidates and build exact matched training banks" in spec_text


# --- the Version-C constants transcribe it faithfully ------------------------

def test_gate_profiles_matches_the_spec() -> None:
    assert gate_profiles.ARMS == ("RND", "DET", "LLM")
    assert gate_profiles.CANDIDATES_PER_ARM == 2048
    assert gate_profiles.FINAL_BANK_PER_ARM == 1024
    assert gate_profiles.PHYSICS_PER_ARM == 512
    assert gate_profiles.GPAT_PER_ARM == 512
    assert gate_profiles.RENDERS_PER_RECIPE == 8
    assert gate_profiles.PHYSICS_RENDERS_PER_RECIPE == 4
    assert gate_profiles.GPAT_RENDERS_PER_RECIPE == 4


def test_the_arithmetic_closes() -> None:
    recipes_per_arm = 256
    assert recipes_per_arm * gate_profiles.RENDERS_PER_RECIPE == gate_profiles.CANDIDATES_PER_ARM
    assert (gate_profiles.PHYSICS_RENDERS_PER_RECIPE
            + gate_profiles.GPAT_RENDERS_PER_RECIPE) == gate_profiles.RENDERS_PER_RECIPE
    assert (gate_profiles.PHYSICS_PER_ARM
            + gate_profiles.GPAT_PER_ARM) == gate_profiles.FINAL_BANK_PER_ARM
    assert gate_profiles.FINAL_BANK_PER_ARM < gate_profiles.CANDIDATES_PER_ARM, (
        "the gate must have something to reject"
    )


@pytest.mark.parametrize("arm", ["det", "llm", "rnd"])
def test_the_shipped_c3_banks_carry_exactly_256_recipes(arm: str) -> None:
    """The independent confirmation: the recipe banks C5 renders from."""
    root = REPO / "assets" / "recipe_banks" / "c3" / arm
    recipes = (root / "recipes.jsonl").read_text(encoding="utf-8").splitlines()
    bank = json.loads((root / "C3_BANK.json").read_text(encoding="utf-8"))

    assert len([line for line in recipes if line.strip()]) == 256
    assert bank["raw_slots"] == 384
    assert bank["arm"] == arm.upper()
    assert bank["arm"] in gate_profiles.ARMS


def test_every_arm_has_the_same_cardinality() -> None:
    """No arm-specific budget. §11.3's whole point."""
    counts = {
        arm: len([line for line in
                  (REPO / "assets" / "recipe_banks" / "c3" / arm / "recipes.jsonl")
                  .read_text(encoding="utf-8").splitlines() if line.strip()])
        for arm in ("det", "llm", "rnd")}

    assert set(counts.values()) == {256}, counts


# --- the Version-B contract is a different experiment ------------------------

def test_the_m8_contract_is_keyed_on_live_samples_not_recipes() -> None:
    config = yaml.safe_load(
        (REPO / "configs" / "synthesis" / "synthetic_bank_m8.yaml").read_text(
            encoding="utf-8"))

    assert config["source"]["live_samples"] == 280
    assert config["candidate_recipes_per_live"] == {"physics": 2, "gpat": 2}
    assert config["expected_counts"] == {"physics": 560, "gpat": 560, "total": 1120}
    assert candidate_plan.EXPECTED_PER_ROUTE == 560
    assert candidate_plan.EXPECTED_TOTAL == 1120


def test_the_m8_contract_has_no_arm_dimension_at_all() -> None:
    """This is what makes it a different experiment rather than a disagreement."""
    config_text = (REPO / "configs" / "synthetic_bank_m8.yaml"
                   if (REPO / "configs" / "synthetic_bank_m8.yaml").is_file()
                   else REPO / "configs" / "synthesis" / "synthetic_bank_m8.yaml"
                   ).read_text(encoding="utf-8")
    plan_text = (REPO / "src" / "prism_fas" / "synthesis" / "candidate_plan.py"
                 ).read_text(encoding="utf-8")

    for text, name in ((config_text, "synthetic_bank_m8.yaml"),
                       (plan_text, "candidate_plan.py")):
        assert '"arm"' not in text and "'arm'" not in text, name
        assert "RND" not in text and "DET" not in text, name


def test_the_m8_bank_prefix_is_the_version_b_one() -> None:
    config = yaml.safe_load(
        (REPO / "configs" / "synthesis" / "synthetic_bank_m8.yaml").read_text(
            encoding="utf-8"))

    assert config["bank_id_prefix"] == "prism_synthetic_bank_m8_v1"
    assert "c5" not in config["bank_id_prefix"]


def test_the_two_contracts_do_not_coincide_by_accident() -> None:
    """If they ever became equal the distinction would go unnoticed."""
    version_c_total = gate_profiles.CANDIDATES_PER_ARM * len(gate_profiles.ARMS)

    assert version_c_total == 6144
    assert candidate_plan.EXPECTED_TOTAL == 1120
    assert version_c_total != candidate_plan.EXPECTED_TOTAL


# --- the boundary that blocks a C5 executor ----------------------------------

def test_the_canonical_generator_still_binds_a_calibration_into_its_identity() -> None:
    """Recorded as the blocker, not worked around.

    `SyntheticBankGenerator.identity()` includes `threshold_sha256`,
    `fingerprint_reference_sha256` and `calibration_sha256`. A C5 that used it
    would need a frozen quality calibration BEFORE C6 selects one, and removing
    those fields would change candidate identities rather than preserve them.
    Either way it is a scientific decision, not a refactor.
    """
    source = (REPO / "src" / "prism_fas" / "synthesis" / "synthetic_bank.py"
              ).read_text(encoding="utf-8")
    identity = source[source.index("def identity(self)"):]
    identity = identity[:identity.index("\n    def ", 10)]

    for field in ("threshold_sha256", "fingerprint_reference_sha256",
                  "calibration_sha256"):
        assert field in identity, field
    assert "FrozenCalibration.load" in source, (
        "the generator loads a calibration unconditionally; a generation-only "
        "mode does not exist")


def test_no_c5_scientific_executor_claims_to_exist_yet() -> None:
    """The ledger in the leakage audit must keep telling the truth."""
    from tests.pipeline.test_scientific_fixture_leakage import DECLARED_SCIENTIFIC_GAPS

    assert DECLARED_SCIENTIFIC_GAPS["c5"]["scientific_executor"] is False
    source = (REPO / "src" / "prism_fas" / "pipeline" / "adapters" / "c5.py"
              ).read_text(encoding="utf-8")
    assert "scientific_evidence" not in source
