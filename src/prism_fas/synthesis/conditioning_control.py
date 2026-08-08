"""The A02 conditioning-bank compatibility control — narrow, typed, identity-bearing.

## The problem this exists to solve, stated exactly

The frozen GPAT checkpoint was trained conditioned on the STRUCTURED M7 recipe
bank, and its checkpoint identity binds that bank. `gpat_checkpoint.load_checkpoint`
lists `recipe_bank_identity` in `STRICT_IDENTITY_FIELDS` and refuses any other
bank. That guard is correct and must stay: pairing a learned generator with a bank
it was not trained for is exactly the silent mistake it prevents.

But Table 60's recipe ablation and hypothesis H4 require a control that holds the
GENERATOR constant and varies only the operator COMPOSITION:

    "Structured recipe bank tot hon random augmentation cung so sample va cung
     detector."

The control therefore has to feed the SAME frozen weights a conditioning vector
drawn from the M10 random-operator bank. Under the ordinary guard that is refused,
and every GPAT candidate of the A02 pilot failed with exactly that message.

## What this module authorizes, and what it deliberately does not

It authorizes ONE pairing, named in advance, between two exact identities:

    trained-on bank   fa989938...  (frozen M7 structured recipe bank)
    conditioning bank 9351d08a...  (frozen M10 random-operator bank)

and nothing else. It is not a flag, not a boolean argument threaded through the
generator, and not reachable by an experiment id. Every other caller — M8, M9, B08
and every ordinary M10 row — keeps the full guard unchanged, including the
`recipe_bank_identity` field, and still FAILS CLOSED on a mismatch.

What it does NOT do:

* it does not modify, retrain or reload the GPAT weights — the checkpoint SHA is
  pinned and re-verified;
* it does not relax any other identity field — package, pair plan, config hash,
  architecture hash and AdaFace weight are all still checked;
* it does not touch the quality gate. Every generated sample still faces the
  frozen M8 v3 thresholds, so an out-of-distribution conditioning vector that
  produces a degenerate image is REJECTED, not silently accepted;
* it does not make the two banks interchangeable anywhere else.

## The honest scientific caveat, recorded in the identity itself

The generator was trained on structured conditioning vectors. A random-operator
vector is OUT OF THE TRAINING CONDITIONING DISTRIBUTION for it. That is a real
limitation of this control and it is bound into the artifact identity through
`policy_payload()`, so any run, bank or report derived from it carries the fact
rather than relying on a reader remembering it.

The alternative designs were rejected for stronger reasons: a physics-only control
would confound H4 with the A03 route dimension, and retraining GPAT on the random
bank would confound H4 with a second generator training — the very thing H4 holds
constant.
"""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from typing import Any

CONDITIONING_CONTROL_SCHEMA_VERSION = "m10-conditioning-bank-control-v1"
CONTROL_POLICY = "a02_random_operator_conditioning"

# The exact frozen identities this control is authorized for. Not configurable:
# a value that is not one of these is not this control.
GPAT_TRAINED_ON_RECIPE_BANK = "fa989938cafdc4887518cc45c35d559d00278358439dc68c2486da10309210cb"
A02_CONDITIONING_RECIPE_BANK = "9351d08ac824cc67021445d1bb59bd9dc14ef7eb3dfa606414500d8fac49603f"
FROZEN_GPAT_CHECKPOINT_SHA = "2047cdb513767010cfdf368c6f53a3664922451c56e1e837ec59cb96918a5b63"
FROZEN_SOURCE_PACKAGE = "b1cf29b69a165ed5d9e074fc8127c17fbf057723edf9e272048ec3a564eb9dc6"
# The identity field this control — and ONLY this control — is allowed to exempt.
EXEMPTED_IDENTITY_FIELD = "recipe_bank_identity"


class ConditioningControlError(RuntimeError):
    """A conditioning-bank compatibility control was requested that is not the one
    declared scientific control, or was requested for the wrong identities."""


@dataclass(frozen=True)
class ConditioningBankControl:
    """An authorization for ONE declared generator/conditioning-bank pairing.

    Construct it only through `for_a02_random_operators`. The bare constructor is
    validated too, so an object that reaches the generator is always one whose
    identities were checked.
    """
    policy: str
    trained_on_recipe_bank_identity: str
    conditioning_recipe_bank_identity: str
    gpat_checkpoint_sha256: str
    source_package_identity: str

    @classmethod
    def for_a02_random_operators(cls, *, conditioning_recipe_bank_identity: str,
                                 gpat_checkpoint_sha256: str,
                                 source_package_identity: str,
                                 trained_on_recipe_bank_identity: str = GPAT_TRAINED_ON_RECIPE_BANK
                                 ) -> "ConditioningBankControl":
        """The one authorized control. Every argument is checked against a pin."""
        return cls(policy=CONTROL_POLICY,
                   trained_on_recipe_bank_identity=str(trained_on_recipe_bank_identity),
                   conditioning_recipe_bank_identity=str(conditioning_recipe_bank_identity),
                   gpat_checkpoint_sha256=str(gpat_checkpoint_sha256),
                   source_package_identity=str(source_package_identity)).validate()

    def validate(self) -> "ConditioningBankControl":
        problems: list[str] = []
        if self.policy != CONTROL_POLICY:
            problems.append(f"policy {self.policy!r} is not the declared control {CONTROL_POLICY!r}")
        if self.trained_on_recipe_bank_identity != GPAT_TRAINED_ON_RECIPE_BANK:
            problems.append("the trained-on recipe bank is not the frozen M7 structured bank")
        if self.conditioning_recipe_bank_identity != A02_CONDITIONING_RECIPE_BANK:
            problems.append("the conditioning recipe bank is not the frozen M10 random-operator bank")
        if self.gpat_checkpoint_sha256 != FROZEN_GPAT_CHECKPOINT_SHA:
            problems.append("the GPAT checkpoint is not the frozen one")
        if self.source_package_identity != FROZEN_SOURCE_PACKAGE:
            problems.append("the source package is not the frozen one")
        if self.trained_on_recipe_bank_identity == self.conditioning_recipe_bank_identity:
            problems.append("this control exists for a MISMATCHED pairing; identical banks need no control")
        if problems:
            raise ConditioningControlError(
                f"refusing an unauthorized conditioning-bank control: {problems}")
        return self

    def exempted_fields(self) -> tuple[str, ...]:
        """Exactly one field, and only after `validate()` has passed."""
        return (EXEMPTED_IDENTITY_FIELD,)

    def policy_payload(self) -> dict[str, Any]:
        """What every artifact and run built under this control must record.

        The out-of-distribution caveat is part of the payload, so it travels with
        the identity instead of living only in a document.
        """
        return {
            "schema_version": CONDITIONING_CONTROL_SCHEMA_VERSION,
            "policy": self.policy,
            "exempted_identity_field": EXEMPTED_IDENTITY_FIELD,
            "trained_on_recipe_bank_identity": self.trained_on_recipe_bank_identity,
            "conditioning_recipe_bank_identity": self.conditioning_recipe_bank_identity,
            "gpat_checkpoint_sha256": self.gpat_checkpoint_sha256,
            "source_package_identity": self.source_package_identity,
            "gpat_weights_modified": False,
            "gpat_retrained": False,
            "quality_gate_relaxed": False,
            "other_identity_fields_still_enforced": True,
            "out_of_training_conditioning_distribution": True,
            "rationale": ("Hypothesis H4 compares a structured recipe bank against random "
                          "augmentation at equal sample count and equal detector, so the control "
                          "must hold the generator constant and vary only the operator "
                          "composition. The frozen GPAT checkpoint binds the structured bank it "
                          "was trained on, so feeding it the random-operator conditioning vector "
                          "requires this one declared exemption. A physics-only control would "
                          "confound H4 with the A03 route dimension; retraining GPAT would "
                          "confound H4 with a second generator training."),
            "caveat": ("The generator was trained on structured conditioning vectors, so a "
                       "random-operator vector is out of its training conditioning distribution. "
                       "The frozen quality gate still measures every output, so a degenerate "
                       "sample is rejected rather than accepted, but the limitation is real and "
                       "is reported with any A02 result.")}

    def identity(self) -> str:
        return hashlib.sha256(json.dumps(self.policy_payload(), sort_keys=True,
                                         separators=(",", ":")).encode("utf-8")).hexdigest()


def expected_identity_for(identity: dict[str, str], *,
                          control: ConditioningBankControl | None) -> dict[str, str]:
    """Narrow a GPAT `expected_identity` map according to an authorized control.

    With no control (every ordinary M8/M9/B08/M10 path) the map is returned
    UNCHANGED, so `recipe_bank_identity` is still checked and a mismatch still
    fails closed. With the one authorized control that single field is dropped —
    and only after the control has validated its own pinned identities, and only
    when the map's own values are the two banks the control names.
    """
    if control is None: return dict(identity)
    control.validate()
    requested = str(identity.get(EXEMPTED_IDENTITY_FIELD, ""))
    if requested and requested != control.conditioning_recipe_bank_identity:
        raise ConditioningControlError(
            f"the control authorizes conditioning bank {control.conditioning_recipe_bank_identity} "
            f"but this run asks to exempt {requested}")
    package = str(identity.get("package_identity", ""))
    if package and package != control.source_package_identity:
        raise ConditioningControlError(
            f"the control is bound to source package {control.source_package_identity} "
            f"but this run uses {package}")
    return {key: value for key, value in identity.items() if key != EXEMPTED_IDENTITY_FIELD}
