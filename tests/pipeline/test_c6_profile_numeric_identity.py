"""§11.4 profile arithmetic, with no quantization step anywhere.

`derive_profile` ended with `round(value, 12) + 0.0`. That arrived in the
engineering-readiness milestone (f8d5a5f, 2026-08-17), when NOMINAL was
fixture-derived and rounding was cosmetic. §11.4 inheritance made it
result-affecting: the authoritative tau_id is 0.547440037939055 and the NOMINAL
profile carried 0.547440037939, so the profile no longer contained the value the
spec requires it to inherit exactly.

§11.4 specifies four formulas and a range-safe exemption. It specifies no
rounding, so there is none. Every expectation below is written as the direct
Python expression rather than as a decimal literal, because a rounded constant
would re-admit the drift through the test.
"""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.synthesis import gate_profiles  # noqa: E402
from prism_fas.synthesis.c6_scientific import threshold_identity  # noqa: E402
from prism_fas.synthesis.c6_threshold_inheritance import (  # noqa: E402
    INHERITED_NOMINAL, VERSION_B_THRESHOLD_SHA256)
from prism_fas.synthesis.gate_profiles import (HIGHER_IS_BETTER,  # noqa: E402
                                               LOWER_IS_BETTER, PROFILE_ORDER,
                                               RANGE_SAFE, derive_profile)
from prism_fas.synthesis.quality_gate import Thresholds  # noqa: E402


# --- 1-4. NOMINAL preserves every inherited value exactly --------------------

def test_nominal_preserves_every_inherited_value_exactly() -> None:
    assert derive_profile(INHERITED_NOMINAL, "NOMINAL") == INHERITED_NOMINAL


def test_tau_id_keeps_its_full_precision() -> None:
    """The metric that made the drift visible."""
    nominal = derive_profile(INHERITED_NOMINAL, "NOMINAL")

    assert nominal["tau_id"] == 0.547440037939055
    assert repr(nominal["tau_id"]) == "0.547440037939055"
    assert nominal["tau_id"] != 0.547440037939


def test_tau_lm_keeps_its_full_precision() -> None:
    nominal = derive_profile(INHERITED_NOMINAL, "NOMINAL")

    assert nominal["tau_lm"] == 0.00836817528937794
    assert repr(nominal["tau_lm"]) == "0.00836817528937794"


def test_tau_parse_and_tau_fp_keep_their_full_precision() -> None:
    nominal = derive_profile(INHERITED_NOMINAL, "NOMINAL")

    assert nominal["tau_parse"] == 0.7094826178704915
    assert nominal["tau_fp"] == 5.687657785453908


@pytest.mark.parametrize("name", sorted(INHERITED_NOMINAL))
def test_every_nominal_threshold_round_trips_bit_for_bit(name) -> None:
    nominal = derive_profile(INHERITED_NOMINAL, "NOMINAL")

    assert nominal[name].hex() == INHERITED_NOMINAL[name].hex(), name


# --- 5-8. the four frozen formulas, as expressions ---------------------------

@pytest.mark.parametrize("name", sorted(HIGHER_IS_BETTER))
def test_higher_is_better_strict_is_the_direct_formula(name) -> None:
    a = INHERITED_NOMINAL[name]

    assert derive_profile(INHERITED_NOMINAL, "STRICT")[name] == a + 0.10 * (1.0 - a)


@pytest.mark.parametrize("name", sorted(HIGHER_IS_BETTER))
def test_higher_is_better_permissive_is_the_direct_formula(name) -> None:
    a = INHERITED_NOMINAL[name]

    assert derive_profile(INHERITED_NOMINAL, "PERMISSIVE")[name] == max(0.0, 0.90 * a)


@pytest.mark.parametrize("name", sorted(LOWER_IS_BETTER))
def test_lower_is_better_strict_is_the_direct_formula(name) -> None:
    a = INHERITED_NOMINAL[name]

    assert derive_profile(INHERITED_NOMINAL, "STRICT")[name] == 0.90 * a


@pytest.mark.parametrize("name", sorted(LOWER_IS_BETTER))
def test_lower_is_better_permissive_is_the_direct_formula(name) -> None:
    a = INHERITED_NOMINAL[name]

    assert derive_profile(INHERITED_NOMINAL, "PERMISSIVE")[name] == 1.10 * a


def test_a_rounded_expectation_would_no_longer_pass() -> None:
    """Proof the formulas are exact rather than merely close."""
    a = INHERITED_NOMINAL["tau_id"]
    strict = derive_profile(INHERITED_NOMINAL, "STRICT")["tau_id"]

    assert strict == a + 0.10 * (1.0 - a)
    assert strict != round(a + 0.10 * (1.0 - a), 12)


# --- 9. the range-safe exemption ---------------------------------------------

@pytest.mark.parametrize("profile", list(PROFILE_ORDER))
def test_range_safe_is_never_profiled(profile) -> None:
    for name in RANGE_SAFE:
        assert derive_profile(INHERITED_NOMINAL, profile)[name] == INHERITED_NOMINAL[name]


def test_permissive_still_clips_at_zero() -> None:
    """The one clip §11.4 does specify stays."""
    assert derive_profile({"tau_id": 0.0}, "PERMISSIVE")["tau_id"] == 0.0


def test_negative_zero_is_canonicalized_without_touching_other_values() -> None:
    """`+ 0.0` is exact for every float except -0.0, which it normalizes."""
    assert repr(derive_profile({"tau_out": -0.0}, "NOMINAL")["tau_out"]) == "0.0"
    for name, value in INHERITED_NOMINAL.items():
        assert (value + 0.0).hex() == value.hex(), name


# --- 10. no quantization step survives anywhere ------------------------------

def test_derive_profile_contains_no_rounding_or_quantization() -> None:
    source = inspect.getsource(derive_profile)
    body = source.split('"""', 2)[2]

    for forbidden in ("round(", "Decimal", "quantize", "nextafter", "format(",
                      ":.1", ":.9", ":.12", "%.1"):
        assert forbidden not in body, forbidden


def test_no_rounding_call_survives_in_the_returned_expression() -> None:
    """Structural, so a rounding helper cannot creep back under a new name."""
    tree = ast.parse(inspect.getsource(gate_profiles))
    node = next(item for item in ast.walk(tree)
                if isinstance(item, ast.FunctionDef) and item.name == "derive_profile")
    calls = {getattr(call.func, "id", getattr(call.func, "attr", ""))
             for call in ast.walk(node) if isinstance(call, ast.Call)}

    assert "round" not in calls
    assert {"float", "max"} & calls, "the formula still uses max() for the clip"


# --- 11, 12. identity is stable and agrees with the calibration --------------

def test_input_dictionary_ordering_cannot_change_the_thresholds() -> None:
    reversed_input = dict(reversed(list(INHERITED_NOMINAL.items())))

    assert (derive_profile(reversed_input, "NOMINAL")
            == derive_profile(INHERITED_NOMINAL, "NOMINAL"))
    for profile in PROFILE_ORDER:
        assert (threshold_identity(derive_profile(reversed_input, profile))
                == threshold_identity(derive_profile(INHERITED_NOMINAL, profile)))


def test_the_nominal_values_identity_now_matches_the_calibration_hash() -> None:
    """Three hashes of the same values, so they agree — legitimately."""
    nominal = derive_profile(INHERITED_NOMINAL, "NOMINAL")

    assert threshold_identity(nominal) == VERSION_B_THRESHOLD_SHA256
    assert Thresholds.from_dict(nominal).sha256() == VERSION_B_THRESHOLD_SHA256


def test_the_gate_profile_identity_stays_a_different_object() -> None:
    """It binds the profile NAME too, so it is not the values identity."""
    profiles = gate_profiles.build_profiles(INHERITED_NOMINAL, nominal_source="x")

    assert profiles["NOMINAL"].identity != threshold_identity(
        profiles["NOMINAL"].thresholds)
    assert len({profiles[name].identity for name in PROFILE_ORDER}) == 3


def test_strict_and_permissive_identities_differ_from_nominal() -> None:
    identities = {profile: threshold_identity(derive_profile(INHERITED_NOMINAL, profile))
                  for profile in PROFILE_ORDER}

    assert len(set(identities.values())) == 3


# --- the C5 pool and the firewall --------------------------------------------

def test_nothing_here_touches_the_candidate_pool_or_the_target() -> None:
    source = (REPO / "src" / "prism_fas" / "synthesis" / "gate_profiles.py"
              ).read_text(encoding="utf-8")

    for forbidden in ("candidate_dir(", "write_record(", "CANDIDATE.json",
                      "siw", "SiW", "target_test", "label_live_spoof"):
        assert forbidden not in source, forbidden


def test_the_frozen_nominal_values_are_unchanged() -> None:
    assert INHERITED_NOMINAL == {
        "tau_fd": 0.5, "tau_id": 0.547440037939055,
        "tau_lm": 0.00836817528937794, "tau_parse": 0.7094826178704915,
        "tau_out": 0.0, "tau_fp": 5.687657785453908}
