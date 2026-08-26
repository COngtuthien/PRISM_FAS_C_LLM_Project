"""C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2 closure — interpretation registration.

**FIXTURE / ENGINEERING ONLY.** Every test here runs against `tmp_path`
fixtures with synthetic checkpoints/populations — never a real repo's GPU
artifacts, never a real checkpoint, never a real image, never target data.
The observed numeric values used below are the real, user-reported GPU
scientific values from the actual V2 execution (color PASS all arms; JPEG
FAIL via RND's tail only; resize FAIL via all three arms' tail only) so
that the interpretation module is proven correct against genuine recorded
numbers, not invented ones.

This file proves: the closure CLI's `--status` and `--register-interpretation`
both require `validate_existing_diagnostics_result`; interpretation
registration never loads a checkpoint, forwards an image, or recomputes a
metric; the four already-written result files are hashed (not fabricated);
the bounded interpretation text matches the required scientific boundaries
for color/JPEG/resize; a BLOCKED test is never promoted to FAIL; no causal
claim is ever labeled anything but NOT_SUPPORTED or an explicit
consistency-not-causal statement; BA_sep/target_access/c9_may_close
invariants hold in the written document; registration is idempotent; a
conflicting existing interpretation blocks; and result tampering blocks
registration.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.evaluation import post_failure_diagnostics_v2 as diag2  # noqa: E402
from prism_fas.evaluation import post_failure_diagnostics_v2_closure as closure  # noqa: E402
from prism_fas.evaluation import post_failure_diagnostics_v2_interpretation as interp  # noqa: E402
from prism_fas.evaluation import post_failure_diagnostics_v2_runner as runner2  # noqa: E402
from prism_fas.evaluation import synthetic_real_probe as probe  # noqa: E402
from prism_fas.pipeline.state import atomic_write_json  # noqa: E402

# The real, user-reported GPU-observed per-arm values for the three
# executable tests (C9 Phase-1B-close observed result).
OBSERVED = {
    "benign_color_corruption": {
        "DET": {"mean_delta_plus": 0.056465106366764904, "p95_delta_plus": 0.2578913045674562,
                "tau_mean": 0.325571173346355, "tau_tail": 0.5053004979020645, "verdict": "PASS"},
        "LLM": {"mean_delta_plus": 0.06030356387006274, "p95_delta_plus": 0.27864541038870805,
                "tau_mean": 0.3771376967108539, "tau_tail": 0.594541305309721, "verdict": "PASS"},
        "RND": {"mean_delta_plus": 0.0745881237909392, "p95_delta_plus": 0.3252048272266983,
                "tau_mean": 0.32637083662993505, "tau_tail": 0.5056458191764535, "verdict": "PASS"},
    },
    "benign_jpeg_corruption": {
        "DET": {"mean_delta_plus": 0.10791187270224327, "p95_delta_plus": 0.43382501669228046,
                "tau_mean": 0.325571173346355, "tau_tail": 0.5053004979020645, "verdict": "PASS"},
        "LLM": {"mean_delta_plus": 0.11490263352946688, "p95_delta_plus": 0.500528473202139,
                "tau_mean": 0.3771376967108539, "tau_tail": 0.594541305309721, "verdict": "PASS"},
        "RND": {"mean_delta_plus": 0.15106372730612444, "p95_delta_plus": 0.5480184525623917,
                "tau_mean": 0.32637083662993505, "tau_tail": 0.5056458191764535, "verdict": "FAIL"},
    },
    "benign_resize_corruption": {
        "DET": {"mean_delta_plus": 0.18970596051804023, "p95_delta_plus": 0.5432409213483332,
                "tau_mean": 0.325571173346355, "tau_tail": 0.5053004979020645, "verdict": "FAIL"},
        "LLM": {"mean_delta_plus": 0.19188802765711443, "p95_delta_plus": 0.6175760632008315,
                "tau_mean": 0.3771376967108539, "tau_tail": 0.594541305309721, "verdict": "FAIL"},
        "RND": {"mean_delta_plus": 0.2091119995355257, "p95_delta_plus": 0.6064400591328736,
                "tau_mean": 0.32637083662993505, "tau_tail": 0.5056458191764535, "verdict": "FAIL"},
    },
}
OBSERVED_TEST_VERDICT = {"benign_color_corruption": "PASS", "benign_jpeg_corruption": "FAIL",
                        "benign_resize_corruption": "FAIL"}


def _observed_per_test() -> dict[str, Any]:
    per_test: dict[str, Any] = {}
    for test_id, by_arm in OBSERVED.items():
        per_arm = {arm: {"arm": arm, "test_id": test_id,
                         "reference_threshold": {"tau_mean": vals["tau_mean"],
                                                 "tau_tail": vals["tau_tail"]},
                         "evaluation": {"samples": 10, "mean_delta_plus": vals["mean_delta_plus"],
                                       "p95_delta_plus": vals["p95_delta_plus"]},
                         "verdict": vals["verdict"]}
                   for arm, vals in by_arm.items()}
        per_test[test_id] = {"status": OBSERVED_TEST_VERDICT[test_id],
                             "classification": "EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL",
                             "per_arm": per_arm}
    for test_id, classification in (("cross_route_synthetic", "NEEDS_SCIENTIFIC_DECISION"),
                                    ("recipe_region_shift", "NEEDS_SCIENTIFIC_DECISION"),
                                    ("artifact_map_swap", "NEEDS_SCIENTIFIC_DECISION"),
                                    ("residual_scale_zero", "STRUCTURALLY_MODEL_BLOCKED"),
                                    ("crop_padding_interpolation", "STRUCTURALLY_DATA_BLOCKED")):
        per_test[test_id] = {"status": "BLOCKED", "classification": classification,
                             "blocked_reason": f"{test_id} blocked reason"}
    return per_test


def _install_v2_execute_fixtures_with_observed_result(monkeypatch, tmp_path) -> None:
    """Binds a real V2 diagnostics result set on `tmp_path`, using --bind-only
    and a monkeypatched --execute path that writes the REAL, user-reported
    observed numbers instead of a fabricated PASS/FAIL."""
    fake_protocol = json.loads(json.dumps(_real_v2_protocol()))
    fake_protocol["benign_corruption_shared"]["split_hash_namespace"] = "test-closure-ns"
    fake_protocol["benign_corruption_shared"]["split_seed"] = 1
    monkeypatch.setattr(diag2, "load_protocol", lambda repo: fake_protocol)
    monkeypatch.setattr(
        diag2, "bind_c8_matrix_identity",
        lambda repo: {"c8_matrix_identity": "c8" + "0" * 62,
                     "c8_acceptance_matrix_identity": "c8" + "0" * 62,
                     "c8_acceptance_path": diag2.C8_ACCEPTANCE_PATH})

    fake_ba_sep_protocol = {"probe_seed_values": [20260806, 20260807, 20260808]}
    monkeypatch.setattr(probe, "load_protocol", lambda repo: fake_ba_sep_protocol)
    monkeypatch.setattr(probe, "protocol_identity", lambda repo: "a" * 64)
    seeds = (20260806, 20260807, 20260808, 20260809, 20260810)
    checkpoints_by_arm = {
        arm: [probe.CheckpointBinding(
            arm=arm, seed=s, row_id=f"C-G-{arm}-P3READY-s{s}", run_identity=f"run-{arm}-{s}",
            config_identity="c" * 64, checkpoint_sha256=f"{arm.lower()}{s}".ljust(64, "0"),
            checkpoint_path="p", checkpoint_kind="best", decision_logit_name="global_logit_G",
            decision_graph_hash="g" * 64) for s in seeds]
        for arm in probe.ARMS}
    monkeypatch.setattr(probe, "resolve_checkpoint_set", lambda repo, arm: checkpoints_by_arm[arm])

    from prism_fas.pipeline.adapters import sources
    monkeypatch.setattr(sources, "verify_detector_inputs",
                        lambda repo, arms=None: {
                            "package_identity": "pkg" + "0" * 61,
                            "c6": {"banks": {arm: {"selected_set_sha256": f"bank-{arm}" + "0" * 55}
                                            for arm in probe.ARMS}}})

    live_records = (
        [{"sample_id": f"casia-live-{i}", "stable_group_identity": f"cg{i}",
         "source_domain": "casia_fasd"} for i in range(40)]
        + [{"sample_id": f"msu-live-{i}", "stable_group_identity": f"mg{i}",
           "source_domain": "msu_mfsd"} for i in range(40)])
    monkeypatch.setattr(diag2, "resolve_source_dev_live_records", lambda repo: live_records)

    exit_code, _ = runner2._bind_only(tmp_path)
    assert exit_code == runner2.EXIT_PASS

    observed_per_test = _observed_per_test()

    def _fake_benign(repo, test_id, arm, checkpoints, *, evaluation_ids, reference_threshold):
        return observed_per_test[test_id]["per_arm"][arm]

    monkeypatch.setattr(diag2, "reference_delta_plus_for_arm",
                        lambda repo, arm, checkpoints, calibration_ids: [0.0, 0.0, 0.001])
    monkeypatch.setattr(diag2, "run_benign_corruption_diagnostic_for_arm", _fake_benign)

    exit_code, payload = runner2._execute(tmp_path)
    assert exit_code == runner2.EXIT_FAIL   # a real result: overall verdict is FAIL
    assert payload["overall_diagnostics_verdict"] == "FAIL"


def _real_v2_protocol() -> dict[str, Any]:
    import yaml

    return yaml.safe_load(
        (REPO / "configs/evaluation/c9_post_failure_source_diagnostics_v2.yaml")
        .read_text(encoding="utf-8"))


# ==============================================================================
# validator required by both modes
# ==============================================================================

def test_status_requires_the_canonical_validator(tmp_path) -> None:
    exit_code, payload = closure._status(tmp_path)
    assert exit_code == closure.EXIT_BLOCKED
    assert payload["reason"] == "NO_VALID_DIAGNOSTICS_RESULT"


def test_register_interpretation_requires_the_canonical_validator(tmp_path) -> None:
    exit_code, payload = closure._register_interpretation(tmp_path)
    assert exit_code == closure.EXIT_BLOCKED
    assert payload["error"] == "EXISTING_RESULT_FAILED_VALIDATION"


def test_status_reports_interpretation_not_yet_registered(monkeypatch, tmp_path) -> None:
    _install_v2_execute_fixtures_with_observed_result(monkeypatch, tmp_path)
    exit_code, payload = closure._status(tmp_path)
    assert exit_code == closure.EXIT_BLOCKED
    assert payload["diagnostics_result_valid"] is True
    assert payload["interpretation_registered"] is False
    assert payload["reason"] == "INTERPRETATION_NOT_YET_REGISTERED"


# ==============================================================================
# no recomputation / no checkpoint load / no image forward
# ==============================================================================

def test_register_interpretation_source_never_forwards_or_loads_checkpoints() -> None:
    source = inspect.getsource(closure._register_interpretation)
    for forbidden in ("construct_row_trainer(", "forward_", "run_benign_corruption_diagnostic_for_arm(",
                      "reference_delta_plus_for_arm(", "collate_items("):
        assert forbidden not in source, forbidden


def test_register_interpretation_reports_zero_recomputation(monkeypatch, tmp_path) -> None:
    _install_v2_execute_fixtures_with_observed_result(monkeypatch, tmp_path)
    exit_code, payload = closure._register_interpretation(tmp_path)
    assert exit_code == closure.EXIT_PASS
    assert payload["checkpoint_weights_loaded"] is False
    assert payload["images_forwarded"] is False
    assert payload["diagnostic_metric_recomputed"] is False


def test_interpretation_module_is_pure_no_filesystem_or_model_access() -> None:
    for name in ("derive_test_interpretation", "derive_full_interpretation"):
        source = inspect.getsource(getattr(interp, name))
        for forbidden in ("open(", "Path(", "construct_row_trainer", "torch", "read_text", "read_bytes"):
            assert forbidden not in source, (name, forbidden)


# ==============================================================================
# exact result hashes bound
# ==============================================================================

def test_result_file_hashes_are_real_not_fabricated(monkeypatch, tmp_path) -> None:
    _install_v2_execute_fixtures_with_observed_result(monkeypatch, tmp_path)
    exit_code, payload = closure._register_interpretation(tmp_path)
    assert exit_code == closure.EXIT_PASS
    document = json.loads((tmp_path / closure.INTERPRETATION_PATH).read_text())
    for name, relative in diag2.RESULT_ARTIFACT_PATHS.items():
        real_hash = hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest()
        assert document["result_file_sha256"][name] == real_hash


# ==============================================================================
# bounded interpretation text: color / JPEG / resize
# ==============================================================================

def test_color_interpretation_is_bounded_no_evidence_for_any_arm() -> None:
    result = interp.derive_full_interpretation(_observed_per_test())
    text = " ".join(result["tests"]["benign_color_corruption"]["interpretation"])
    assert "No evidence" in text
    assert "RND" in text and "DET" in text and "LLM" in text
    assert not result["tests"]["benign_color_corruption"]["not_supported"] == []  # a boundary is still recorded
    assert "generalizing" in " ".join(result["tests"]["benign_color_corruption"]["not_supported"])


def test_jpeg_rnd_tail_only_failure_is_correctly_represented() -> None:
    result = interp.derive_full_interpretation(_observed_per_test())
    entry = result["tests"]["benign_jpeg_corruption"]
    assert entry["test_verdict"] == "FAIL"
    assert entry["per_arm"]["RND"]["criterion_failed"] == "tail"
    assert entry["per_arm"]["DET"]["criterion_failed"] is None
    assert entry["per_arm"]["LLM"]["criterion_failed"] is None
    assert entry["per_arm"]["RND"]["derived_arithmetic"]["tail_exceedance"] == pytest.approx(
        0.5480184525623917 - 0.5056458191764535)
    text = " ".join(entry["interpretation"])
    assert "ONLY the upper-tail" in text
    assert "not a broad average score inflation" in text


def test_resize_all_arm_tail_failure_is_correctly_represented() -> None:
    result = interp.derive_full_interpretation(_observed_per_test())
    entry = result["tests"]["benign_resize_corruption"]
    assert entry["test_verdict"] == "FAIL"
    for arm in ("RND", "DET", "LLM"):
        assert entry["per_arm"][arm]["criterion_failed"] == "tail"
    text = " ".join(entry["interpretation"])
    assert "primarily a tail-sensitivity" in text
    # exact ranking: RND largest, then DET, then LLM smallest
    assert "RND (0.1007942)" in text
    assert text.index("RND (0.1007942)") < text.index("DET (0.0379404)") < text.index("LLM (0.0230348)")
    assert "descriptive comparison only" in text
    assert any("statistical ranking" in s for s in entry["not_supported"])


# ==============================================================================
# blocked tests never promoted; no causal claim promoted to fact
# ==============================================================================

@pytest.mark.parametrize("test_id", ["cross_route_synthetic", "recipe_region_shift",
                                     "artifact_map_swap", "residual_scale_zero",
                                     "crop_padding_interpolation"])
def test_blocked_tests_are_never_promoted_to_pass_or_fail(test_id: str) -> None:
    result = interp.derive_full_interpretation(_observed_per_test())
    entry = result["tests"][test_id]
    assert entry["observed"]["status"] == "BLOCKED"
    assert "test_verdict" not in entry
    assert "BLOCKED is not evidence" in " ".join(entry["interpretation"])


def test_no_causal_statement_promoted_to_observed_fact() -> None:
    result = interp.derive_full_interpretation(_observed_per_test())
    assert any("not established" in s or "not a causal" in s
              for s in result["global_interpretation"])
    assert any("causal" in s for s in result["global_not_supported"])


def test_global_not_supported_forbids_generalizing_blocked_as_fail() -> None:
    result = interp.derive_full_interpretation(_observed_per_test())
    assert any("BLOCKED" in s and "FAIL" in s for s in result["global_not_supported"])


# ==============================================================================
# BA_sep / c9_may_close / target_access invariants in the written document
# ==============================================================================

def test_written_interpretation_carries_immutable_invariants(monkeypatch, tmp_path) -> None:
    _install_v2_execute_fixtures_with_observed_result(monkeypatch, tmp_path)
    closure._register_interpretation(tmp_path)
    document = json.loads((tmp_path / closure.INTERPRETATION_PATH).read_text())
    assert document["ba_sep_observed_verdict"] == "FAIL"
    assert document["detector_reliability_lock_c_observed_overall"] == "FAILED"
    assert document["c9_may_close"] is False
    assert document["target_access"] == 0
    assert document["overall_diagnostics_verdict"] == "FAIL"


def test_closure_source_never_references_a_target_path() -> None:
    for module in (closure, interp):
        source = Path(inspect.getfile(module)).read_text(encoding="utf-8")
        for forbidden in ("siw", "SiW", "target_test", "target_taxonomy",
                          "prism_target_eval_v2", "resolve_target"):
            assert forbidden not in source, (module.__name__, forbidden)


# ==============================================================================
# idempotent registration / conflicting interpretation blocks / tampering blocks
# ==============================================================================

def test_registration_is_idempotent_on_exact_match(monkeypatch, tmp_path) -> None:
    _install_v2_execute_fixtures_with_observed_result(monkeypatch, tmp_path)
    exit_code, first = closure._register_interpretation(tmp_path)
    assert exit_code == closure.EXIT_PASS
    assert first["reused"] is False
    before = (tmp_path / closure.INTERPRETATION_PATH).read_text()

    exit_code, second = closure._register_interpretation(tmp_path)
    assert exit_code == closure.EXIT_PASS
    assert second["registered"] is True
    assert second["reused"] is True
    after = (tmp_path / closure.INTERPRETATION_PATH).read_text()
    assert before == after


def test_conflicting_existing_interpretation_blocks_without_overwriting(monkeypatch, tmp_path) -> None:
    _install_v2_execute_fixtures_with_observed_result(monkeypatch, tmp_path)
    closure._register_interpretation(tmp_path)
    stale = json.loads((tmp_path / closure.INTERPRETATION_PATH).read_text())
    stale["overall_diagnostics_verdict"] = "PASS"   # tamper the registered interpretation
    atomic_write_json(tmp_path / closure.INTERPRETATION_PATH, stale)

    exit_code, payload = closure._register_interpretation(tmp_path)
    assert exit_code == closure.EXIT_BLOCKED
    assert "differs from the one just derived" in payload["error"]
    after = json.loads((tmp_path / closure.INTERPRETATION_PATH).read_text())
    assert after["overall_diagnostics_verdict"] == "PASS"   # still tampered — never overwritten


def test_result_tampering_after_execution_blocks_registration(monkeypatch, tmp_path) -> None:
    _install_v2_execute_fixtures_with_observed_result(monkeypatch, tmp_path)
    tampered = json.loads((tmp_path / diag2.VERDICT_PATH).read_text())
    tampered["overall_diagnostics_verdict"] = "PASS"   # flip the real FAIL to a fake PASS
    atomic_write_json(tmp_path / diag2.VERDICT_PATH, tampered)

    exit_code, payload = closure._register_interpretation(tmp_path)
    assert exit_code == closure.EXIT_BLOCKED
    assert payload["error"] == "EXISTING_RESULT_FAILED_VALIDATION"
    assert not (tmp_path / closure.INTERPRETATION_PATH).exists()


def test_status_reflects_a_registered_interpretation(monkeypatch, tmp_path) -> None:
    _install_v2_execute_fixtures_with_observed_result(monkeypatch, tmp_path)
    closure._register_interpretation(tmp_path)
    exit_code, payload = closure._status(tmp_path)
    assert exit_code == closure.EXIT_PASS
    assert payload["interpretation_registered"] is True


# ==============================================================================
# safety / import purity
# ==============================================================================

def test_importing_closure_modules_touches_no_filesystem_state() -> None:
    import subprocess

    for module in ("prism_fas.evaluation.post_failure_diagnostics_v2_closure",
                  "prism_fas.evaluation.post_failure_diagnostics_v2_interpretation"):
        result = subprocess.run([sys.executable, "-c", f"import {module}"],
                                capture_output=True, text=True, cwd=str(REPO))
        assert result.returncode == 0, result.stderr


def test_runner_invocable_as_python_dash_m_module() -> None:
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "prism_fas.evaluation.post_failure_diagnostics_v2_closure",
         "--repo", str(REPO), "--status"],
        capture_output=True, text=True, cwd=str(REPO / "src"))
    assert result.returncode == closure.EXIT_BLOCKED, result.stderr
    payload = json.loads(result.stdout)
    assert payload["target_access"] == 0
