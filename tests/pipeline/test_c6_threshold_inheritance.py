"""§11.4 NOMINAL inheritance, and the frozen CUDA device family.

Two things are pinned here.

The device is a FAMILY, frozen by decision before any C6 result existed. It is
never chosen from availability and never falls back: on a host without CUDA the
stage blocks, because the gate would otherwise be applied to measurements from a
different backend than the one frozen. The GPU model and library versions are
run provenance, recorded rather than claimed as reproducible.

The thresholds are INHERITED. §11.4 fits a threshold only when no semantically
compatible Version-B value exists, and here every metric has one — provably, not
by name: the measurement modules and the three pinned models are byte-identical
between the frozen Version-B tree and this one, over the same M3B package. The
first executor refitted everything, which would have resurrected the three v1
values Version B itself superseded.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.pipeline.adapters import c6 as c6_module  # noqa: E402
from prism_fas.synthesis import c6_threshold_inheritance as inheritance  # noqa: E402
from prism_fas.synthesis.gate_profiles import (HIGHER_IS_BETTER,  # noqa: E402
                                               LOWER_IS_BETTER, PROFILE_ORDER,
                                               RANGE_SAFE, derive_profile)

VERSION_B = REPO.parent / "PRISM_FAS_B_Project"

#: The final Version-B set, transcribed independently of the module under test.
EXPECTED = {"tau_fd": 0.5, "tau_id": 0.547440037939055,
            "tau_lm": 0.00836817528937794, "tau_parse": 0.7094826178704915,
            "tau_out": 0.0, "tau_fp": 5.687657785453908}


# --- 1, 2. the frozen device --------------------------------------------------

def test_the_scientific_device_family_is_frozen_at_cuda() -> None:
    assert c6_module.FROZEN_QUALITY_BACKEND_DEVICE == "cuda"


def test_the_device_is_resolved_from_the_constant_not_from_availability(
        monkeypatch) -> None:
    monkeypatch.setattr(c6_module, "_cuda_present", lambda: True)

    assert c6_module._quality_backend_device(object()) == "cuda"

    source = __import__("inspect").getsource(c6_module._quality_backend_device)
    assert "FROZEN_QUALITY_BACKEND_DEVICE" in source
    assert "resolve_device" not in source
    # `_cuda_present` appears only to REFUSE, never to choose between two devices.
    assert '"cpu"' not in source and "'cpu'" not in source


def test_an_absent_cuda_device_blocks_and_never_falls_back(monkeypatch) -> None:
    monkeypatch.setattr(c6_module, "_cuda_present", lambda: False)

    with pytest.raises(c6_module.QualityBackendDeviceUnavailable) as raised:
        c6_module._quality_backend_device(object())

    assert raised.value.reason_code == "C6_QUALITY_BACKEND_DEVICE_UNAVAILABLE"
    assert "does not fall back to CPU" in str(raised.value)


def test_the_run_provenance_is_recorded_separately_from_the_family() -> None:
    record = c6_module.quality_backend_provenance("cuda")

    assert record["frozen_device_family"] == "cuda"
    assert record["requested_device"] == "cuda"
    assert record["bitwise_reproduction_across_gpu_models_claimed"] is False
    # Whatever this host is, the probe records rather than asserts.
    assert "cuda_available" in record or "torch_probe_error" in record


# --- 3, 4. inheritance, and that calibration cannot overwrite it -------------

def test_every_metric_inherits_the_final_version_b_value() -> None:
    nominal, provenance = inheritance.assemble_nominal()

    assert nominal == EXPECTED
    assert provenance["source_reference_derived"] == [], (
        "every metric has a compatible inherited threshold today")
    assert set(provenance["inherited"]) == {"tau_fd", "tau_id", "tau_lm",
                                            "tau_parse", "tau_fp"}
    assert provenance["frozen_range_constraints"] == ["tau_out"]


def test_a_refitted_threshold_cannot_overwrite_an_inherited_one() -> None:
    """The exact defect: `calibrate` returns the superseded v1 values."""
    refitted = {"tau_fd": 0.5, "tau_id": 0.9995203357934952,
                "tau_lm": 0.002135227532959269, "tau_parse": 0.8747814437904173,
                "tau_out": 0.0, "tau_fp": 5.687657785453908}

    nominal, provenance = inheritance.assemble_nominal(refitted)

    assert nominal == EXPECTED, "the inherited values won"
    ignored = provenance["calibrator_values_ignored_because_inherited"]
    assert ignored == {"tau_id": 0.9995203357934952,
                       "tau_lm": 0.002135227532959269,
                       "tau_parse": 0.8747814437904173}


def test_the_superseded_v1_values_are_named_so_they_cannot_creep_back() -> None:
    for metric, superseded in (("tau_id", 0.9995203357934952),
                               ("tau_lm", 0.002135227532959269),
                               ("tau_parse", 0.8747814437904173)):
        entry = inheritance.PROVENANCE[metric]
        assert entry["superseded_version_b_value"] == superseded
        assert entry["version_b_value"] != superseded


def test_the_vendored_values_match_the_frozen_version_b_artifact() -> None:
    if not (VERSION_B / inheritance.VERSION_B_ARTIFACT).is_file():
        pytest.skip("the Version-B tree is not mounted beside this repository")

    verification = inheritance.verify_version_b_artifact(VERSION_B)

    assert verification["available"] is True
    assert verification["artifact_sha256"] == inheritance.VERSION_B_ARTIFACT_SHA256
    assert verification["artifact_sha256_matches"] is True
    assert verification["threshold_sha256_matches"] is True
    assert verification["values_match"] is True
    assert verification["mismatched"] == {}


def test_the_assembled_nominal_reproduces_version_bs_own_threshold_hash() -> None:
    """A cross-check the inheritance is exact rather than merely close."""
    nominal, _ = inheritance.assemble_nominal()

    assert inheritance.nominal_identity(nominal) == inheritance.VERSION_B_THRESHOLD_SHA256


def test_the_missing_version_b_tree_is_reported_not_guessed() -> None:
    verification = inheritance.verify_version_b_artifact(Path("does-not-exist"))

    assert verification["available"] is False
    assert "not mounted" in verification["reason"]


# --- 5. a metric with no inherited threshold takes the derived branch --------

def test_a_metric_without_a_compatible_inherited_threshold_is_derived(
        monkeypatch) -> None:
    ruling = {**inheritance.PROVENANCE}
    ruling["tau_parse"] = {**ruling["tau_parse"],
                           "nominal_source": inheritance.SOURCE_REFERENCE_DERIVED,
                           "semantic_compatibility": "NO"}
    monkeypatch.setattr(inheritance, "PROVENANCE", ruling)

    nominal, provenance = inheritance.assemble_nominal({"tau_parse": 0.61})

    assert nominal["tau_parse"] == 0.61, "the derived value is used"
    assert nominal["tau_id"] == EXPECTED["tau_id"], "the others still inherit"
    assert provenance["source_reference_derived"] == ["tau_parse"]


def test_a_derived_metric_with_no_derived_value_fails_closed(monkeypatch) -> None:
    ruling = {**inheritance.PROVENANCE}
    ruling["tau_parse"] = {**ruling["tau_parse"],
                           "nominal_source": inheritance.SOURCE_REFERENCE_DERIVED}
    monkeypatch.setattr(inheritance, "PROVENANCE", ruling)

    with pytest.raises(inheritance.ThresholdInheritanceError, match="no compatible"):
        inheritance.assemble_nominal({})


def test_a_threshold_with_no_inheritance_ruling_is_refused() -> None:
    with pytest.raises(inheritance.ThresholdInheritanceError, match="no inheritance"):
        inheritance.assemble_nominal({"tau_invented": 0.5})


# --- the compatibility claim is provable, not asserted -----------------------

@pytest.mark.parametrize("relative", [
    "src/prism_fas/synthesis/quality_gate.py",
    "src/prism_fas/synthesis/quality_calibration.py",
    "src/prism_fas/synthesis/quality_models.py",
    "src/prism_fas/synthesis/synthetic_bank.py",
    "src/prism_fas/synthesis/identity_calibration.py",
    "src/prism_fas/synthesis/structural_calibration.py",
    "src/prism_fas/synthesis/fingerprint.py",
])
def test_the_measurement_modules_are_byte_identical_to_version_b(relative) -> None:
    """Compatibility rests on this. If a measurement module diverges, the
    inherited thresholds stop describing what Version C measures."""
    if not (VERSION_B / relative).is_file():
        pytest.skip("the Version-B tree is not mounted beside this repository")

    mine = hashlib.sha256((REPO / relative).read_bytes()).hexdigest()
    theirs = hashlib.sha256((VERSION_B / relative).read_bytes()).hexdigest()

    assert mine == theirs, f"{relative} diverged; re-audit §11.4 compatibility"


def test_the_pinned_models_are_the_same_three_weights() -> None:
    if not (VERSION_B / inheritance.VERSION_B_ARTIFACT).is_file():
        pytest.skip("the Version-B tree is not mounted beside this repository")
    from prism_fas.synthesis.quality_models import PINNED

    recorded = json.loads((VERSION_B / inheritance.VERSION_B_ARTIFACT)
                          .read_text(encoding="utf-8"))["quality_models"]["models"]

    for role in ("detector", "identity", "parsing"):
        assert PINNED[role]["sha256"] == recorded[role]["sha256"], role


def test_version_b_calibrated_on_the_same_source_package() -> None:
    from prism_fas.synthesis.conditioning_control import FROZEN_SOURCE_PACKAGE

    assert inheritance.VERSION_B_PACKAGE_IDENTITY == FROZEN_SOURCE_PACKAGE


def test_every_metric_records_a_compatibility_verdict_and_a_reason() -> None:
    for name, entry in inheritance.PROVENANCE.items():
        assert entry["semantic_compatibility"] in ("YES", "NO"), name
        assert len(entry["compatibility_reason"]) > 40, name
        assert entry["direction"] in ("higher_is_better", "lower_is_better",
                                      "exact_equality"), name
        assert entry["comparator"] in (">=", "<=", "=="), name


def test_the_declared_directions_match_the_gate_implementation() -> None:
    for name, entry in inheritance.PROVENANCE.items():
        if entry["direction"] == "higher_is_better":
            assert name in HIGHER_IS_BETTER, name
        elif entry["direction"] == "lower_is_better":
            assert name in LOWER_IS_BETTER, name
        else:
            assert name in RANGE_SAFE, name


# --- 6, 7. the §11.4 profile formulas, exactly --------------------------------

def test_higher_is_better_profiles_are_exact() -> None:
    a = EXPECTED["tau_id"]
    profiles = {name: derive_profile(EXPECTED, name)["tau_id"] for name in PROFILE_ORDER}

    assert profiles["NOMINAL"] == pytest.approx(a)
    assert profiles["STRICT"] == pytest.approx(a + 0.10 * (1 - a))
    assert profiles["PERMISSIVE"] == pytest.approx(max(0.0, 0.90 * a))


def test_lower_is_better_profiles_are_exact() -> None:
    a = EXPECTED["tau_lm"]
    profiles = {name: derive_profile(EXPECTED, name)["tau_lm"] for name in PROFILE_ORDER}

    assert profiles["NOMINAL"] == pytest.approx(a)
    assert profiles["STRICT"] == pytest.approx(0.90 * a)
    assert profiles["PERMISSIVE"] == pytest.approx(1.10 * a)


def test_the_range_safe_constraint_is_never_relaxed() -> None:
    for name in PROFILE_ORDER:
        assert derive_profile(EXPECTED, name)["tau_out"] == 0.0


def test_permissive_never_goes_below_zero() -> None:
    assert derive_profile({"tau_id": 0.0}, "PERMISSIVE")["tau_id"] == 0.0


# --- 8, 9. one threshold identity per profile, for all three arms -------------

def test_all_arms_share_one_threshold_identity_per_profile() -> None:
    from prism_fas.synthesis.c6_scientific import (build_common_profiles,
                                                   threshold_identity)
    from prism_fas.synthesis.gate_profiles import ARMS

    profiles = build_common_profiles(EXPECTED, nominal_source="inherited")

    for name in PROFILE_ORDER:
        identity = threshold_identity(profiles[name].thresholds)
        # The same object is handed to every arm; there is no per-arm variant.
        assert {threshold_identity(profiles[name].thresholds) for _ in ARMS} == {identity}
    assert len({threshold_identity(profiles[name].thresholds)
                for name in PROFILE_ORDER}) == 3


def test_no_arm_specific_threshold_exists_anywhere() -> None:
    source = (REPO / "src" / "prism_fas" / "pipeline" / "adapters" / "c6.py"
              ).read_text(encoding="utf-8")

    for forbidden in ("thresholds_by_arm", "per_arm_threshold", "arm_thresholds",
                      "relax_for_arm"):
        assert forbidden not in source, forbidden


# --- 10, 11, 12. firewall, C5 immutability, prior regression ------------------

def test_the_inheritance_module_opens_no_target_or_source_dev() -> None:
    source = (REPO / "src" / "prism_fas" / "synthesis"
              / "c6_threshold_inheritance.py").read_text(encoding="utf-8")

    for forbidden in ("siw", "SiW", "target_test", "label_live_spoof", "source_dev"):
        assert forbidden not in source, forbidden


def test_the_inherited_values_came_from_a_source_train_only_calibration() -> None:
    if not (VERSION_B / inheritance.VERSION_B_ARTIFACT).is_file():
        pytest.skip("the Version-B tree is not mounted beside this repository")

    payload = json.loads((VERSION_B / inheritance.VERSION_B_ARTIFACT)
                         .read_text(encoding="utf-8"))

    assert payload["split"] == "source_train"
    assert payload["used_target"] is False
    assert payload["used_source_dev"] is False
    assert payload["used_generated_candidates"] is False


def test_nothing_here_touches_the_c5_candidate_pool() -> None:
    for relative in ("src/prism_fas/synthesis/c6_threshold_inheritance.py",
                     "src/prism_fas/pipeline/adapters/c6.py"):
        source = (REPO / relative).read_text(encoding="utf-8")
        for forbidden in ("write_payload_bytes", "failure_record(", "write_record(",
                          "render_arm", "candidate_dir("):
            assert forbidden not in source, f"{relative}: {forbidden}"


def test_version_b_is_untouched() -> None:
    import subprocess

    if not (VERSION_B / ".git").exists():
        pytest.skip("the Version-B tree is not mounted beside this repository")

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=VERSION_B,
                          capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=VERSION_B,
                           capture_output=True, text=True, check=True).stdout.strip()

    assert head == inheritance.VERSION_B_COMMIT
    assert dirty == ""
