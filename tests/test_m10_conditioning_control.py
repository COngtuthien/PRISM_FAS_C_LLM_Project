"""The A02 conditioning-bank compatibility control, and the guard it must not weaken.

The feature here is small; the guard around it is the point. These tests exist to
prove that ONE declared pairing of frozen identities is authorized and that
everything else — every ordinary M8/M9/B08/M10 path, every wrong identity, every
attempt to reach the exemption by another route — still FAILS CLOSED.

If any test in this file starts passing for the wrong reason, a learned generator
can be silently paired with a recipe bank it was never trained on.
"""
from __future__ import annotations
import io
import json
import tokenize
from pathlib import Path
import pytest

from prism_fas.synthesis.conditioning_control import (A02_CONDITIONING_RECIPE_BANK,
                                                      CONDITIONING_CONTROL_SCHEMA_VERSION,
                                                      CONTROL_POLICY, EXEMPTED_IDENTITY_FIELD,
                                                      FROZEN_GPAT_CHECKPOINT_SHA,
                                                      FROZEN_SOURCE_PACKAGE,
                                                      GPAT_TRAINED_ON_RECIPE_BANK,
                                                      ConditioningBankControl,
                                                      ConditioningControlError,
                                                      expected_identity_for)
from prism_fas.synthesis.gpat_checkpoint import STRICT_IDENTITY_FIELDS

OTHER = "0" * 64


def control() -> ConditioningBankControl:
    return ConditioningBankControl.for_a02_random_operators(
        conditioning_recipe_bank_identity=A02_CONDITIONING_RECIPE_BANK,
        gpat_checkpoint_sha256=FROZEN_GPAT_CHECKPOINT_SHA,
        source_package_identity=FROZEN_SOURCE_PACKAGE)


def full_identity(recipe_bank: str = A02_CONDITIONING_RECIPE_BANK) -> dict[str, str]:
    return {"package_identity": FROZEN_SOURCE_PACKAGE,
            "recipe_bank_identity": recipe_bank,
            "pair_plan_identity": "p" * 64}


# ============================================================================
# THE ORDINARY PATH STILL FAILS CLOSED
# ============================================================================

def test_no_control_leaves_the_identity_map_completely_unchanged():
    """Every M8/M9/B08 path passes `control=None`, and must keep every field —
    `recipe_bank_identity` above all."""
    identity = full_identity(GPAT_TRAINED_ON_RECIPE_BANK)
    assert expected_identity_for(identity, control=None) == identity
    assert EXEMPTED_IDENTITY_FIELD in expected_identity_for(identity, control=None)


def test_the_recipe_bank_field_is_still_a_strict_gpat_identity_field():
    """The exemption is applied by the CALLER narrowing what it asks to be checked;
    it never removes the field from the checkpoint contract itself."""
    assert EXEMPTED_IDENTITY_FIELD in STRICT_IDENTITY_FIELDS


def test_an_ordinary_mismatch_still_raises_from_the_real_loader(tmp_path):
    """The real `load_checkpoint`, with no control, on a genuine mismatch."""
    import torch
    from prism_fas.synthesis.gpat_checkpoint import CheckpointError, load_checkpoint
    from prism_fas.synthesis.gpat_contracts import GPAT_CHECKPOINT_SCHEMA_VERSION
    stored = {field: ("t" * 64) for field in STRICT_IDENTITY_FIELDS}
    stored["recipe_bank_identity"] = GPAT_TRAINED_ON_RECIPE_BANK
    path = tmp_path / "gpat.pt"
    torch.save({"schema_version": GPAT_CHECKPOINT_SCHEMA_VERSION, "identity": stored,
                "model_state": {}}, path)
    asked = {"recipe_bank_identity": A02_CONDITIONING_RECIPE_BANK}
    with pytest.raises(CheckpointError, match="recipe_bank_identity"):
        load_checkpoint(path, expected_identity=expected_identity_for(asked, control=None))
    # ...and the authorized control is what makes the same load succeed.
    payload = load_checkpoint(path, expected_identity=expected_identity_for(asked, control=control()))
    assert payload["identity"]["recipe_bank_identity"] == GPAT_TRAINED_ON_RECIPE_BANK


# ============================================================================
# ONLY THE DECLARED CONTROL, ONLY THE DECLARED IDENTITIES
# ============================================================================

def test_the_authorized_control_exempts_exactly_one_field():
    narrowed = expected_identity_for(full_identity(), control=control())
    assert EXEMPTED_IDENTITY_FIELD not in narrowed
    assert narrowed == {"package_identity": FROZEN_SOURCE_PACKAGE, "pair_plan_identity": "p" * 64}
    assert control().exempted_fields() == (EXEMPTED_IDENTITY_FIELD,)


@pytest.mark.parametrize("kwargs,reason", [
    ({"conditioning_recipe_bank_identity": OTHER}, "random-operator bank"),
    ({"gpat_checkpoint_sha256": OTHER}, "GPAT checkpoint"),
    ({"source_package_identity": OTHER}, "source package"),
    ({"trained_on_recipe_bank_identity": OTHER}, "M7 structured bank"),
])
def test_a_wrong_identity_is_refused(kwargs, reason):
    """A near-miss is not the control. Every pin is checked."""
    base = {"conditioning_recipe_bank_identity": A02_CONDITIONING_RECIPE_BANK,
            "gpat_checkpoint_sha256": FROZEN_GPAT_CHECKPOINT_SHA,
            "source_package_identity": FROZEN_SOURCE_PACKAGE}
    with pytest.raises(ConditioningControlError, match=reason):
        ConditioningBankControl.for_a02_random_operators(**{**base, **kwargs})


def test_a_control_for_two_identical_banks_is_refused():
    """This control exists for a MISMATCH. Asking for it where the banks agree means
    the caller has misunderstood what it authorizes."""
    with pytest.raises(ConditioningControlError, match="identical banks"):
        ConditioningBankControl(policy=CONTROL_POLICY,
                                trained_on_recipe_bank_identity=GPAT_TRAINED_ON_RECIPE_BANK,
                                conditioning_recipe_bank_identity=GPAT_TRAINED_ON_RECIPE_BANK,
                                gpat_checkpoint_sha256=FROZEN_GPAT_CHECKPOINT_SHA,
                                source_package_identity=FROZEN_SOURCE_PACKAGE).validate()


def test_an_invented_policy_name_is_refused():
    with pytest.raises(ConditioningControlError, match="declared control"):
        ConditioningBankControl(policy="anything_goes",
                                trained_on_recipe_bank_identity=GPAT_TRAINED_ON_RECIPE_BANK,
                                conditioning_recipe_bank_identity=A02_CONDITIONING_RECIPE_BANK,
                                gpat_checkpoint_sha256=FROZEN_GPAT_CHECKPOINT_SHA,
                                source_package_identity=FROZEN_SOURCE_PACKAGE).validate()


def test_the_control_refuses_a_run_that_asks_to_exempt_a_different_bank():
    """Holding a valid control does not authorize exempting some OTHER bank."""
    with pytest.raises(ConditioningControlError, match="authorizes conditioning bank"):
        expected_identity_for(full_identity(OTHER), control=control())


def test_the_control_refuses_a_run_on_a_different_source_package():
    identity = {**full_identity(), "package_identity": OTHER}
    with pytest.raises(ConditioningControlError, match="source package"):
        expected_identity_for(identity, control=control())


# ============================================================================
# B08, M8 AND M9 CANNOT REACH THE EXEMPTION
# ============================================================================

def test_b08_and_every_ordinary_row_pass_no_control(tmp_path):
    """The exemption is not reachable from a variant flag. `recipe_conditioning`
    has three values and none of them is a permission; only the A02 ARTIFACT BUILD
    constructs a control, and B08 uses the structured bank where no control is even
    constructible."""
    from prism_fas.detector.variant import FLAG_VOCABULARY, ResolvedExperimentVariant
    for value in FLAG_VOCABULARY["recipe_conditioning"]:
        assert value in ("structured", "random_operators", "off")
    reference = ResolvedExperimentVariant.reference()
    assert reference.recipe_conditioning == "structured"
    # B08's bank is the structured one, and a control for it is refused outright.
    with pytest.raises(ConditioningControlError):
        ConditioningBankControl.for_a02_random_operators(
            conditioning_recipe_bank_identity=GPAT_TRAINED_ON_RECIPE_BANK,
            gpat_checkpoint_sha256=FROZEN_GPAT_CHECKPOINT_SHA,
            source_package_identity=FROZEN_SOURCE_PACKAGE)


def test_the_generator_defaults_to_no_control():
    """A caller that says nothing gets the full guard. Defaulting the other way is
    how a safety guard quietly stops guarding."""
    import inspect
    from prism_fas.synthesis.m8_pipeline import build_generator
    from prism_fas.synthesis.synthetic_bank import GPATRoute, SyntheticBankGenerator
    assert inspect.signature(build_generator).parameters["conditioning_control"].default is None
    assert inspect.signature(GPATRoute.__init__).parameters["conditioning_control"].default is None
    fields = {field.name: field for field in SyntheticBankGenerator.__dataclass_fields__.values()}
    assert fields["conditioning_control"].default is None


def test_no_module_branches_on_an_experiment_id_to_grant_the_exemption():
    """The exemption is granted by a typed object carrying frozen identities, never
    by recognising a row's name."""
    from prism_fas.synthesis import conditioning_control as module
    source = Path(module.__file__).read_text(encoding="utf-8")
    code = " ".join(token.string for token in
                    tokenize.generate_tokens(io.StringIO(source).readline)
                    if token.type not in (tokenize.STRING, tokenize.COMMENT))
    for needle in ("experiment_id", "A02-recipe", "B08", "startswith"):
        assert needle not in code, f"the control's CODE branches on {needle!r}"


# ============================================================================
# THE CONTROL IS IN THE IDENTITY
# ============================================================================

def test_the_policy_payload_records_what_was_and_was_not_done():
    payload = control().policy_payload()
    assert payload["schema_version"] == CONDITIONING_CONTROL_SCHEMA_VERSION
    assert payload["policy"] == CONTROL_POLICY
    assert payload["exempted_identity_field"] == EXEMPTED_IDENTITY_FIELD
    assert payload["trained_on_recipe_bank_identity"] == GPAT_TRAINED_ON_RECIPE_BANK
    assert payload["conditioning_recipe_bank_identity"] == A02_CONDITIONING_RECIPE_BANK
    assert payload["gpat_checkpoint_sha256"] == FROZEN_GPAT_CHECKPOINT_SHA
    # What must be false, and stay false.
    assert payload["gpat_weights_modified"] is False
    assert payload["gpat_retrained"] is False
    assert payload["quality_gate_relaxed"] is False
    assert payload["other_identity_fields_still_enforced"] is True
    # And the honest caveat travels with the identity, not only in a document.
    assert payload["out_of_training_conditioning_distribution"] is True
    assert "out of its training conditioning distribution" in payload["caveat"]


def test_the_control_has_a_stable_distinct_identity():
    first, second = control().identity(), control().identity()
    assert first == second and len(first) == 64
    # A different pairing would be a different control; the identity binds the pins.
    payload = control().policy_payload()
    assert payload["conditioning_recipe_bank_identity"] != payload["trained_on_recipe_bank_identity"]


def test_the_generator_identity_carries_the_control_only_when_it_is_used():
    """The M8 v3 bank identity must be untouched, and an A02 artifact must carry the
    policy inside its own identity."""
    import inspect
    from prism_fas.synthesis.synthetic_bank import SyntheticBankGenerator
    source = inspect.getsource(SyntheticBankGenerator.identity)
    assert "if self.conditioning_control is not None" in source
    assert "conditioning_control_identity" in source
    assert "gpat_trained_on_recipe_bank_identity" in source


def test_the_frozen_m8_v3_bank_identity_is_not_disturbed():
    """The structured bank was built with no control, so nothing about it moves."""
    lock = Path("data/processed/prism_synthetic_bank_m8_v3_e84c78cd2a9b/BANK_LOCK.json")
    if not lock.is_file(): pytest.skip("the frozen M8 v3 bank is not present in this checkout")
    payload = json.loads(lock.read_text(encoding="utf-8"))
    assert payload["bank_content_identity_sha256"] == \
        "e84c78cd2a9b548244e243de0380998d04bc6770b91caf32ac7be96f489bb542"
    assert "conditioning_control_identity" not in json.dumps(payload)
