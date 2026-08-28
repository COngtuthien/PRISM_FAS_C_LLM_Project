"""POST_FAILURE_EXPLORATORY_TARGET_V3 — E1 execution TECHNICAL FIX.

A real GPU `--predict` attempt (code commit `04d7d40c07f4fbc4d493dd3b30c5a4bbbc541ed9`)
loaded checkpoint weights successfully, then failed BEFORE `target_batches(...)`
and BEFORE `predict_target(...)` with `VariantError: unknown flags
['recipe_arm']`. Root cause: `predict_one_row_to_staging` called
`VariantCapabilities.from_flags(binding["flags"])`, which resolves a full
`ResolvedExperimentVariant` from the ENTIRE bound flags dict — including
`recipe_arm`, C8 treatment/bank metadata that `source_matrix._track_g_flags`/
`_track_r_flags` deliberately add and that `pipeline.adapters.c8._detector_config_for_row`
(the canonical C8 path `synthetic_real_probe.construct_row_trainer` reuses)
deliberately strips before resolving a variant. `recipe_arm` is not, and
never has been, a `detector.variant.FLAG_KEYS` vocabulary entry.

This file proves the corrected `resolve_verified_row_capabilities`: it
derives capabilities from the variant the canonical trainer reconstruction
already resolved (`trainer.config.variant`), cross-checked against the
variant portion of the frozen binding's own flags — never re-resolving a
variant from the full flag set, and never widening
`ResolvedExperimentVariant.resolve()` to tolerate `recipe_arm` or any other
unknown flag globally.

**FIXTURE / ENGINEERING ONLY.** Every test here is pure Python —
`ResolvedExperimentVariant.resolve()` is cheap, deterministic, and touches
no GPU, no checkpoint, no target feature, no target label. No test in this
file runs real target prediction, binds a prediction plan against real
data, or opens a target label.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.detector.variant import FLAG_KEYS, ResolvedExperimentVariant, VariantError  # noqa: E402
from prism_fas.evaluation import post_failure_exploratory_target_v3 as v3  # noqa: E402
from prism_fas.evaluation.source_matrix import _track_g_flags, _track_r_flags  # noqa: E402
from prism_fas.evaluation.target_prediction import VariantCapabilities  # noqa: E402

from test_post_failure_exploratory_target_v3 import _install_v3_binding_fixtures  # noqa: E402

V3_PROTOCOL_PATH = REPO / "configs/evaluation/post_failure_exploratory_target_v3.yaml"
V3_IDENTITY = "a2b54f8844a2a36540e62470c2f5f30de52fbf509a37f03feb7f6d769d5c702c"


def _binding_for(row_id: str, arm: str, flags: dict) -> dict:
    return {"row_id": row_id, "arm": arm, "flags": dict(flags)}


def _variant_for(flags: dict) -> ResolvedExperimentVariant:
    """The variant a correctly-behaving canonical trainer reconstruction
    would resolve — built the SAME way `pipeline.adapters.c8._detector_config_for_row`
    does: strip `recipe_arm`, then resolve."""
    variant_only = {key: value for key, value in flags.items() if key in FLAG_KEYS}
    return ResolvedExperimentVariant.resolve(variant_only)


# ==============================================================================
# Real frozen rows resolve capabilities successfully
# ==============================================================================

@pytest.mark.parametrize("arm", ["RND", "DET", "LLM"])
def test_track_g_row_resolves_capabilities_successfully(arm) -> None:
    flags = _track_g_flags(arm)
    binding = _binding_for(f"C-G-{arm}-P3READY-s1", arm, flags)
    trainer_variant = _variant_for(flags)
    capabilities = v3.resolve_verified_row_capabilities(binding, trainer_variant=trainer_variant)
    assert isinstance(capabilities, VariantCapabilities)


@pytest.mark.parametrize("arm", ["DET", "LLM"])
def test_track_r_row_resolves_capabilities_successfully(arm) -> None:
    flags = _track_r_flags(arm)
    binding = _binding_for(f"C-R-{arm}-P3READY-s1", arm, flags)
    trainer_variant = _variant_for(flags)
    capabilities = v3.resolve_verified_row_capabilities(binding, trainer_variant=trainer_variant)
    assert isinstance(capabilities, VariantCapabilities)


def test_c_r_noprompt_resolves_with_recipe_arm_llm_and_prompt_off() -> None:
    flags = _track_r_flags("LLM", prompt="off")
    assert flags["recipe_arm"] == "LLM"
    assert flags["prompt"] == "off"
    binding = _binding_for("C-R-NOPROMPT-P3READY-s1", "LLM", flags)
    trainer_variant = _variant_for(flags)
    capabilities = v3.resolve_verified_row_capabilities(binding, trainer_variant=trainer_variant)
    assert isinstance(capabilities, VariantCapabilities)


# ==============================================================================
# Fail-closed scenarios
# ==============================================================================

def test_recipe_arm_disagreeing_with_bound_arm_fails_closed() -> None:
    flags = _track_g_flags("RND")
    binding = _binding_for("C-G-RND-P3READY-s1", "DET", flags)   # bound arm != flags["recipe_arm"]
    trainer_variant = _variant_for(flags)
    with pytest.raises(v3.ExploratoryTargetV3Error, match="recipe_arm"):
        v3.resolve_verified_row_capabilities(binding, trainer_variant=trainer_variant)


def test_missing_required_variant_flag_fails_closed() -> None:
    flags = _track_g_flags("RND")
    del flags["frames_per_video"]
    binding = _binding_for("C-G-RND-P3READY-s1", "RND", flags)
    trainer_variant = _variant_for(_track_g_flags("RND"))
    with pytest.raises(v3.ExploratoryTargetV3Error, match="missing required variant flag"):
        v3.resolve_verified_row_capabilities(binding, trainer_variant=trainer_variant)


def test_trainer_variant_disagreeing_with_frozen_binding_fails_closed() -> None:
    flags = _track_g_flags("RND")
    binding = _binding_for("C-G-RND-P3READY-s1", "RND", flags)
    drifted_variant = _variant_for(_track_r_flags("RND"))   # a different variant entirely
    with pytest.raises(v3.ExploratoryTargetV3Error, match="disagrees with the frozen binding"):
        v3.resolve_verified_row_capabilities(binding, trainer_variant=drifted_variant)


def test_arbitrary_unexpected_non_variant_metadata_is_not_silently_accepted() -> None:
    flags = {**_track_g_flags("RND"), "bogus_extra_metadata_key": "surprise"}
    binding = _binding_for("C-G-RND-P3READY-s1", "RND", flags)
    trainer_variant = _variant_for(flags)
    with pytest.raises(v3.ExploratoryTargetV3Error, match="unexpected non-variant metadata"):
        v3.resolve_verified_row_capabilities(binding, trainer_variant=trainer_variant)


def test_known_non_variant_metadata_flags_is_exactly_recipe_arm() -> None:
    assert v3.KNOWN_NON_VARIANT_ROW_METADATA_FLAGS == frozenset({"recipe_arm"})


# ==============================================================================
# The historical failure, reproduced and proven fixed
# ==============================================================================

def test_from_flags_on_the_full_binding_still_raises_the_historical_variant_error() -> None:
    """Confirms this is a genuine regression test, not a tautology: the OLD
    call shape (`VariantCapabilities.from_flags(binding["flags"])`) still
    raises exactly the observed GPU failure against a real frozen row's
    flags."""
    flags = _track_g_flags("RND")
    with pytest.raises(VariantError, match=r"unknown flags \['recipe_arm'\]"):
        VariantCapabilities.from_flags(flags)


def test_the_new_path_resolves_the_same_row_without_a_variant_error() -> None:
    flags = _track_g_flags("RND")
    binding = _binding_for("C-G-RND-P3READY-s1", "RND", flags)
    trainer_variant = _variant_for(flags)
    capabilities = v3.resolve_verified_row_capabilities(binding, trainer_variant=trainer_variant)
    assert isinstance(capabilities, VariantCapabilities)


def test_predict_one_row_to_staging_no_longer_calls_from_flags_on_the_full_binding() -> None:
    source = inspect.getsource(v3.predict_one_row_to_staging)
    assert 'VariantCapabilities.from_flags(binding["flags"])' not in source
    assert "resolve_verified_row_capabilities" in source


def test_capability_resolution_happens_before_target_batches_and_predict_target() -> None:
    source = inspect.getsource(v3.predict_one_row_to_staging)
    resolution_index = source.index("resolve_verified_row_capabilities(binding")
    assert resolution_index < source.index("batches = target_batches(")
    assert resolution_index < source.index("rows = predict_target(")
    assert resolution_index < source.index("row_staging_dir.mkdir")


# ==============================================================================
# inference_config_hash still sees the FULL flags, including recipe_arm
# ==============================================================================

def test_inference_config_hash_still_receives_the_full_binding_flags() -> None:
    source = inspect.getsource(v3.predict_one_row_to_staging)
    assert 'flags=binding["flags"]' in source


def test_resolve_verified_row_capabilities_never_removes_recipe_arm_from_the_binding() -> None:
    flags = _track_g_flags("RND")
    binding = _binding_for("C-G-RND-P3READY-s1", "RND", flags)
    trainer_variant = _variant_for(flags)
    v3.resolve_verified_row_capabilities(binding, trainer_variant=trainer_variant)
    assert binding["flags"]["recipe_arm"] == "RND"
    assert binding["flags"] == flags


# ==============================================================================
# build_prediction_plan_binding is untouched by this fix
# ==============================================================================

def test_build_prediction_plan_binding_source_never_references_variant_capabilities() -> None:
    source = inspect.getsource(v3.build_prediction_plan_binding)
    assert "VariantCapabilities" not in source
    assert "resolve_verified_row_capabilities" not in source


def test_build_prediction_plan_binding_still_constructs_deterministically(monkeypatch, tmp_path) -> None:
    _install_v3_binding_fixtures(monkeypatch)
    first = v3.build_prediction_plan_binding(tmp_path)
    second = v3.build_prediction_plan_binding(tmp_path)
    assert first == second
    assert first["prediction_plan_binding_identity"]
    for row_id, row_binding in first["rows"].items():
        assert "recipe_arm" not in row_binding.get("flags", {}) or row_binding["flags"].get(
            "recipe_arm") == row_binding["arm"]


# ==============================================================================
# V3 protocol identity unchanged; V1/V2 untouched
# ==============================================================================

def test_v3_protocol_identity_unchanged() -> None:
    payload = yaml.safe_load(V3_PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert v3.protocol_identity(payload) == V3_IDENTITY


def test_v1_and_v2_configs_and_modules_not_modified() -> None:
    import subprocess

    result = subprocess.run(["git", "diff", "--stat", "HEAD", "--",
                            "configs/evaluation/post_failure_exploratory_target_v1.yaml",
                            "configs/evaluation/post_failure_exploratory_target_v2.yaml",
                            "configs/evaluation/post_failure_exploratory_target_v3.yaml",
                            "src/prism_fas/evaluation/post_failure_exploratory_target.py",
                            "src/prism_fas/evaluation/post_failure_exploratory_target_scorer.py",
                            "src/prism_fas/evaluation/post_failure_exploratory_target_v2.py",
                            "src/prism_fas/evaluation/post_failure_exploratory_target_v2_scorer.py",
                            "src/prism_fas/evaluation/source_matrix.py",
                            "src/prism_fas/detector/variant.py"],
                            cwd=str(REPO), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""
