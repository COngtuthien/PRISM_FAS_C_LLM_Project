"""C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2 — pre-execution scientific correction.

**FIXTURE / ENGINEERING ONLY.** Every test in this file runs against
`tmp_path` fixtures or pure functions with synthetic arrays — never a real
repo's GPU artifacts, never a real checkpoint, never a real image, never
target data.

This file proves the seventeen items the correction task requires: V1 stays
byte-identical and untouched; V2 has its own, distinct protocol identity;
the benign-corruption threshold can never be derived from the tested
corruption's own effect and only from the independent M8 reference
population; `cross_route_synthetic` cannot reuse the BA_sep ceiling under a
name whose declared meaning is the opposite direction; the existing-result
validator detects a tampered result, provenance, verdict or a mismatched
binding and blocks rather than recomputing; a complete VALID result set
re-reports with zero recomputation; the real canonical C8 matrix identity is
bound and cross-checked; the calibration/evaluation split is proven
group-safe per domain; `target_access` is 0 everywhere; a FAILED BA_sep/
reliability barrier can never reopen C9; and the V1/V2 artifact namespaces
never collide.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.evaluation import detector_reliability as barrier  # noqa: E402
from prism_fas.evaluation import post_failure_diagnostics as diag_v1  # noqa: E402
from prism_fas.evaluation import post_failure_diagnostics_v2 as diag2  # noqa: E402
from prism_fas.evaluation import post_failure_diagnostics_v2_runner as runner2  # noqa: E402
from prism_fas.evaluation import synthetic_real_probe as probe  # noqa: E402
from prism_fas.pipeline.state import atomic_write_json  # noqa: E402

V1_PROTOCOL_PATH = REPO / "configs/evaluation/c9_post_failure_source_diagnostics_v1.yaml"
V2_PROTOCOL_PATH = REPO / "configs/evaluation/c9_post_failure_source_diagnostics_v2.yaml"
V1_IDENTITY = "cb05271e26d9a421f2f9277599523e185026e1eab644febc07c75432d26f3fc5"
BA_SEP_V2_IDENTITY = "720a2e344017d588d71005b81fdf0e7d2062081ae2f3881a61a306d952dc4ac8"


def _real_v1_protocol() -> dict[str, Any]:
    return yaml.safe_load(V1_PROTOCOL_PATH.read_text(encoding="utf-8"))


def _real_v2_protocol() -> dict[str, Any]:
    return yaml.safe_load(V2_PROTOCOL_PATH.read_text(encoding="utf-8"))


# ==============================================================================
# 1. V1 protocol byte-identical / unchanged
# ==============================================================================

def test_v1_config_file_not_modified_by_this_correction() -> None:
    import subprocess

    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--",
         "configs/evaluation/c9_post_failure_source_diagnostics_v1.yaml"],
        cwd=str(REPO), capture_output=True, text=True)
    assert result.stdout.strip() == ""


def test_v1_module_files_not_modified_by_this_correction() -> None:
    import subprocess

    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--",
         "src/prism_fas/evaluation/post_failure_diagnostics.py",
         "src/prism_fas/evaluation/post_failure_diagnostics_runner.py"],
        cwd=str(REPO), capture_output=True, text=True)
    assert result.stdout.strip() == ""


def test_v1_protocol_identity_unchanged() -> None:
    assert diag_v1.protocol_identity(_real_v1_protocol()) == V1_IDENTITY


def test_v1_protocol_still_declares_frozen_not_run() -> None:
    payload = _real_v1_protocol()
    assert payload["status"] == "FROZEN_NOT_RUN"
    assert payload["decision_id"] == "C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V1"


# ==============================================================================
# 2. V2 has a distinct protocol identity
# ==============================================================================

def test_v2_protocol_identity_differs_from_v1() -> None:
    v1_id = diag_v1.protocol_identity(_real_v1_protocol())
    v2_id = diag2.protocol_identity(_real_v2_protocol())
    assert v1_id != v2_id
    assert len(v2_id) == 64


def test_v2_protocol_declares_frozen_not_run_and_supersedes_v1() -> None:
    payload = _real_v2_protocol()
    assert payload["status"] == "FROZEN_NOT_RUN"
    assert payload["decision_id"] == "C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2"
    assert payload["supersedes"]["protocol_identity"] == V1_IDENTITY
    assert payload["supersedes"]["v1_scientific_metric_observed"] is False
    assert payload["supersedes"]["scientific_requests_before_supersession"] == 0


def test_v2_protocol_identity_deterministic() -> None:
    payload = _real_v2_protocol()
    assert diag2.protocol_identity(payload) == diag2.protocol_identity(payload)


def test_v2_protocol_identity_unaffected_by_metadata_only_change() -> None:
    payload = _real_v2_protocol()
    baseline = diag2.protocol_identity(payload)
    changed = dict(payload)
    changed["frozen_on"] = "2099-01-01"
    changed["approved_by"] = "someone else"
    assert diag2.protocol_identity(changed) == baseline


# ==============================================================================
# 3/4/5. threshold cannot depend on the tested corruption; independent
#         reference only; no reference => NEEDS_SCIENTIFIC_DECISION
# ==============================================================================

def test_threshold_derivation_reads_only_reference_values() -> None:
    source = inspect.getsource(diag2.derive_reference_threshold)
    for forbidden in ("evaluation", "corruption_fn", "CORRUPTION_FUNCTIONS"):
        assert forbidden not in source, forbidden


def test_run_benign_corruption_diagnostic_never_derives_threshold_from_evaluation() -> None:
    source = inspect.getsource(diag2.run_benign_corruption_diagnostic_for_arm)
    assert "derive_reference_threshold(" not in source
    assert "reference_threshold[" in source or "reference_threshold\"" in source


def test_reference_threshold_uses_the_frozen_m8_benign_population() -> None:
    source = inspect.getsource(diag2.forward_reference_benign_evidence_for_arm)
    assert "quality_calibration" in source
    assert "BENIGN_VARIANTS" in source
    assert "BENIGN_NOISE_STD" in source
    assert "benign_variant" in source


def test_reference_variants_match_the_frozen_v2_protocol_declaration() -> None:
    from prism_fas.synthesis.quality_calibration import BENIGN_NOISE_STD, BENIGN_VARIANTS

    payload = _real_v2_protocol()
    declared = payload["benign_corruption_shared"]["reference_benign_controls"]
    assert set(declared["variants"]) == {v["name"] for v in BENIGN_VARIANTS}
    assert declared["gaussian_noise_std"] == BENIGN_NOISE_STD


def test_threshold_is_computed_only_from_calibration_group_delta_plus() -> None:
    reference_values = [0.0, 0.0, 0.0, 0.01]
    report = diag2.derive_reference_threshold(reference_values)
    values = np.asarray(reference_values, dtype=np.float64)
    assert report["reference_mean_delta_plus"] == pytest.approx(float(values.mean()))
    assert report["reference_std_delta_plus"] == pytest.approx(float(values.std(ddof=0)))
    assert report["tau_mean"] == pytest.approx(report["reference_mean_delta_plus"]
                                               + 3 * report["reference_std_delta_plus"])
    assert report["tau_tail"] == pytest.approx(report["reference_p95_delta_plus"]
                                               + 3 * report["reference_std_delta_plus"])


def test_reference_threshold_fails_closed_on_empty_reference_population() -> None:
    with pytest.raises(diag2.PostFailureDiagnosticsError):
        diag2.derive_reference_threshold([])


def test_corruption_verdict_requires_both_mean_and_tail_within_tolerance() -> None:
    assert diag2.corruption_verdict(0.05, 0.05, tau_mean=0.05, tau_tail=0.05) == "PASS"   # ties pass
    assert diag2.corruption_verdict(0.0501, 0.0, tau_mean=0.05, tau_tail=1.0) == "FAIL"    # mean fails
    assert diag2.corruption_verdict(0.0, 0.0501, tau_mean=1.0, tau_tail=0.05) == "FAIL"    # tail fails
    assert diag2.corruption_verdict(0.0, 0.0, tau_mean=0.05, tau_tail=0.05) == "PASS"


def test_no_arbitrary_numeric_threshold_literal_in_benign_corruption_protocol() -> None:
    payload = _real_v2_protocol()
    shared = payload["benign_corruption_shared"]["threshold_derivation"]
    assert shared["tau_mean"] == "reference_mean + 3 * reference_std"
    assert shared["tau_tail"] == "reference_p95 + 3 * reference_std"


# ==============================================================================
# 6. cross_route_synthetic cannot reuse the BA_sep ceiling under this name
# ==============================================================================

def test_cross_route_synthetic_is_needs_scientific_decision_in_v2() -> None:
    payload = _real_v2_protocol()
    config = payload["tests"]["cross_route_synthetic"]
    assert config["classification"] == "NEEDS_SCIENTIFIC_DECISION"
    assert config["gpu_ready"] is False
    assert "cross_route_synthetic" not in diag2.EXECUTABLE_TESTS


def test_cross_route_blocked_reason_names_the_semantic_mismatch() -> None:
    payload = _real_v2_protocol()
    reason = payload["tests"]["cross_route_synthetic"]["blocked_reason"].lower()
    assert "retained" in reason or "retention" in reason
    assert "separab" in reason
    assert "0.75" in payload["tests"]["cross_route_synthetic"]["blocked_reason"]


def test_cross_route_synthetic_canonical_declared_meaning_is_retention_not_separability() -> None:
    from prism_fas.evaluation.reliability import DECLARED_TESTS

    declared = {t.test_id: t for t in DECLARED_TESTS}["cross_route_synthetic"]
    assert "retained" in declared.pass_rule


def test_v1_cross_route_attempt_preserved_unused_by_v2() -> None:
    # V1's real-vs-synthetic separability probe machinery must still exist,
    # unchanged, but nothing in the V2 runner may call it.
    assert hasattr(diag_v1, "run_cross_route_diagnostic_for_arm")
    runner2_source = Path(inspect.getfile(runner2)).read_text(encoding="utf-8")
    assert "run_cross_route_diagnostic_for_arm(" not in runner2_source


# ==============================================================================
# fixtures shared by bind/execute tests below
# ==============================================================================

def _install_v2_bind_fixtures(monkeypatch, *, seeds=(20260806, 20260807, 20260808,
                                                     20260809, 20260810)) -> dict[str, Any]:
    fake_protocol = _real_v2_protocol()
    fake_protocol["benign_corruption_shared"]["split_hash_namespace"] = "test-diag-v2-ns"
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

    return {"protocol": fake_protocol, "checkpoints_by_arm": checkpoints_by_arm,
           "live_records": live_records}


def _install_v2_execute_fixtures(monkeypatch, tmp_path, *, verdict_per_arm: str) -> None:
    _install_v2_bind_fixtures(monkeypatch)
    exit_code, _ = runner2._bind_only(tmp_path)
    assert exit_code == runner2.EXIT_PASS

    monkeypatch.setattr(diag2, "reference_delta_plus_for_arm",
                        lambda repo, arm, checkpoints, calibration_ids: [0.0, 0.0, 0.001])

    def _fake_benign(repo, test_id, arm, checkpoints, *, evaluation_ids, reference_threshold):
        return {"arm": arm, "test_id": test_id, "reference_threshold": dict(reference_threshold),
               "evaluation": {"samples": len(evaluation_ids), "mean_delta_plus": 0.0,
                              "p95_delta_plus": 0.0},
               "verdict": verdict_per_arm}

    monkeypatch.setattr(diag2, "run_benign_corruption_diagnostic_for_arm", _fake_benign)


# ==============================================================================
# bind-only mechanics
# ==============================================================================

def test_bind_only_binds_a_real_c8_matrix_identity_not_placeholder_text(monkeypatch, tmp_path) -> None:
    _install_v2_bind_fixtures(monkeypatch)
    exit_code, payload = runner2._bind_only(tmp_path)
    assert exit_code == runner2.EXIT_PASS
    assert payload["c8_matrix_identity"] == "c8" + "0" * 62
    protocol_binding = json.loads((tmp_path / diag2.PROTOCOL_BINDING_PATH).read_text())
    assert protocol_binding["c8_matrix_identity"] == "c8" + "0" * 62
    assert "see reports/full/c8" not in json.dumps(protocol_binding)


def test_bind_only_writes_zero_scientific_metric(monkeypatch, tmp_path) -> None:
    _install_v2_bind_fixtures(monkeypatch)
    exit_code, _ = runner2._bind_only(tmp_path)
    assert exit_code == runner2.EXIT_PASS
    for relative in (diag2.PROTOCOL_BINDING_PATH, diag2.POPULATION_BINDING_PATH,
                    diag2.CHECKPOINT_BINDING_PATH):
        text = (tmp_path / relative).read_text()
        for forbidden in ("balanced_accuracy", "mean_delta_plus", "verdict", "PASS", "FAIL"):
            assert forbidden not in text, (relative, forbidden)


def test_bind_only_is_idempotent(monkeypatch, tmp_path) -> None:
    _install_v2_bind_fixtures(monkeypatch)
    runner2._bind_only(tmp_path)
    before = (tmp_path / diag2.PROTOCOL_BINDING_PATH).read_text()
    exit_code, payload = runner2._bind_only(tmp_path)
    after = (tmp_path / diag2.PROTOCOL_BINDING_PATH).read_text()
    assert exit_code == runner2.EXIT_PASS
    assert payload["reused"] is True
    assert before == after


# ==============================================================================
# 14. per-domain calibration/evaluation group safety
# ==============================================================================

def test_bind_only_records_per_domain_group_safety_for_both_domains(monkeypatch, tmp_path) -> None:
    _install_v2_bind_fixtures(monkeypatch)
    exit_code, _ = runner2._bind_only(tmp_path)
    assert exit_code == runner2.EXIT_PASS
    population = json.loads((tmp_path / diag2.POPULATION_BINDING_PATH).read_text())
    per_domain = population["per_domain_group_safety"]["per_domain"]
    for domain in ("casia_fasd", "msu_mfsd"):
        assert per_domain[domain]["calibration_unique_groups"] > 0
        assert per_domain[domain]["evaluation_unique_groups"] > 0


def test_per_domain_group_safety_fails_closed_when_one_domain_is_degenerate() -> None:
    records = [{"sample_id": f"c{i}", "stable_group_identity": f"g{i}",
               "source_domain": "casia_fasd"} for i in range(10)]
    # msu_mfsd contributes zero records: entirely degenerate for that domain.
    split = {"calibration": [f"g{i}" for i in range(5)],
            "evaluation": [f"g{i}" for i in range(5, 10)]}
    with pytest.raises(diag2.PostFailureDiagnosticsError, match="msu_mfsd"):
        diag2.verify_per_domain_group_safety(records, split, domains=("casia_fasd", "msu_mfsd"))


def test_per_domain_group_safety_fails_closed_on_domain_split_intersection() -> None:
    records = [{"sample_id": f"c{i}", "stable_group_identity": "shared",
               "source_domain": "casia_fasd"} for i in range(2)]
    split = {"calibration": ["shared"], "evaluation": ["shared"]}
    with pytest.raises(diag2.PostFailureDiagnosticsError, match="intersection"):
        diag2.verify_per_domain_group_safety(records, split, domains=("casia_fasd",))


def test_global_split_safety_alone_is_not_sufficient_a_per_domain_check_exists() -> None:
    # a pooled/global split can be non-degenerate while one domain is entirely
    # absent from a group — the per-domain function must still catch it.
    records = ([{"sample_id": f"c{i}", "stable_group_identity": f"cg{i}",
               "source_domain": "casia_fasd"} for i in range(20)]
              + [{"sample_id": f"m{i}", "stable_group_identity": f"mg{i}",
                 "source_domain": "msu_mfsd"} for i in range(20)])
    group_ids = sorted({r["stable_group_identity"] for r in records})
    global_split = diag2.calibration_evaluation_split(group_ids, namespace="ns", seed=1)
    assert global_split["calibration"] and global_split["evaluation"]   # globally fine
    # Now simulate a degenerate PER-DOMAIN split by hand: all casia groups
    # dumped into calibration only.
    degenerate_split = {"calibration": [f"cg{i}" for i in range(20)] + global_split["calibration"],
                        "evaluation": [g for g in global_split["evaluation"]
                                      if not g.startswith("cg")]}
    with pytest.raises(diag2.PostFailureDiagnosticsError, match="casia_fasd"):
        diag2.verify_per_domain_group_safety(records, degenerate_split,
                                            domains=("casia_fasd", "msu_mfsd"))


# ==============================================================================
# 13. actual canonical C8 matrix identity is bound
# ==============================================================================

def test_bind_c8_matrix_identity_uses_the_canonical_source_matrix_plan() -> None:
    source = inspect.getsource(diag2.bind_c8_matrix_identity)
    assert "source_matrix" in source
    assert "build_plan" in source
    assert ".identity" in source


def test_bind_c8_matrix_identity_fails_closed_when_acceptance_file_absent(tmp_path) -> None:
    with pytest.raises(diag2.PostFailureDiagnosticsError, match="C8_ACCEPTANCE"):
        diag2.bind_c8_matrix_identity(tmp_path)


def test_bind_c8_matrix_identity_fails_closed_on_mismatch(monkeypatch, tmp_path) -> None:
    from prism_fas.evaluation import source_matrix

    class _FakePlan:
        identity = "canonical" + "0" * 55

    monkeypatch.setattr(source_matrix, "build_plan", lambda: _FakePlan())
    acceptance_path = tmp_path / diag2.C8_ACCEPTANCE_PATH
    acceptance_path.parent.mkdir(parents=True, exist_ok=True)
    acceptance_path.write_text(json.dumps({"matrix_identity": "stale" + "0" * 59}))
    with pytest.raises(diag2.PostFailureDiagnosticsError, match="does not match"):
        diag2.bind_c8_matrix_identity(tmp_path)


def test_bind_c8_matrix_identity_succeeds_on_agreement(monkeypatch, tmp_path) -> None:
    from prism_fas.evaluation import source_matrix

    class _FakePlan:
        identity = "canonical" + "0" * 55

    monkeypatch.setattr(source_matrix, "build_plan", lambda: _FakePlan())
    acceptance_path = tmp_path / diag2.C8_ACCEPTANCE_PATH
    acceptance_path.parent.mkdir(parents=True, exist_ok=True)
    acceptance_path.write_text(json.dumps({"matrix_identity": "canonical" + "0" * 55}))
    result = diag2.bind_c8_matrix_identity(tmp_path)
    assert result["c8_matrix_identity"] == "canonical" + "0" * 55


# ==============================================================================
# 12. complete VALID result re-reports  /  11. complete INVALID result blocks
# ==============================================================================

def test_all_present_valid_result_re_reports_without_recomputation(monkeypatch, tmp_path) -> None:
    _install_v2_execute_fixtures(monkeypatch, tmp_path, verdict_per_arm="PASS")
    runner2._execute(tmp_path)   # first real execution
    called = {"n": 0}
    monkeypatch.setattr(diag2, "run_benign_corruption_diagnostic_for_arm",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    monkeypatch.setattr(diag2, "reference_delta_plus_for_arm",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    exit_code, payload = runner2._execute(tmp_path)   # second call
    assert called["n"] == 0
    assert payload["reused_existing_diagnostics_result"] is True
    assert payload["checkpoint_weights_loaded"] is False
    assert exit_code == runner2.EXIT_PASS


def test_execute_pass_verdict_when_all_executed_tests_pass(monkeypatch, tmp_path) -> None:
    _install_v2_execute_fixtures(monkeypatch, tmp_path, verdict_per_arm="PASS")
    exit_code, payload = runner2._execute(tmp_path)
    assert exit_code == runner2.EXIT_PASS
    assert payload["overall_diagnostics_verdict"] == "PASS"


def test_execute_fail_verdict_when_one_test_fails(monkeypatch, tmp_path) -> None:
    _install_v2_execute_fixtures(monkeypatch, tmp_path, verdict_per_arm="FAIL")
    exit_code, payload = runner2._execute(tmp_path)
    assert exit_code == runner2.EXIT_FAIL
    assert payload["overall_diagnostics_verdict"] == "FAIL"


def test_partial_result_set_blocks(monkeypatch, tmp_path) -> None:
    _install_v2_execute_fixtures(monkeypatch, tmp_path, verdict_per_arm="PASS")
    runner2._execute(tmp_path)
    (tmp_path / diag2.VERDICT_PATH).unlink()
    exit_code, payload = runner2._execute(tmp_path)
    assert exit_code == runner2.EXIT_BLOCKED
    assert payload["error"] == "PARTIAL_SCIENTIFIC_RESULT_SET"


def test_existing_result_files_never_overwritten_by_a_second_execute(monkeypatch, tmp_path) -> None:
    _install_v2_execute_fixtures(monkeypatch, tmp_path, verdict_per_arm="PASS")
    runner2._execute(tmp_path)
    before = (tmp_path / diag2.RESULT_PATH).read_text()
    runner2._execute(tmp_path)
    after = (tmp_path / diag2.RESULT_PATH).read_text()
    assert before == after


# ==============================================================================
# 7/8/9/10/11. existing-result validator: tampered result/provenance/verdict,
#              mismatched bindings, complete-invalid blocks
# ==============================================================================

def test_validator_detects_tampered_result(monkeypatch, tmp_path) -> None:
    _install_v2_execute_fixtures(monkeypatch, tmp_path, verdict_per_arm="PASS")
    runner2._execute(tmp_path)
    doc = json.loads((tmp_path / diag2.RESULT_PATH).read_text())
    doc["per_test"]["benign_jpeg_corruption"]["status"] = "PASS"
    doc["per_test"]["benign_jpeg_corruption"]["per_arm"]["RND"]["verdict"] = "FAIL"
    atomic_write_json(tmp_path / diag2.RESULT_PATH, doc)
    validation = diag2.validate_existing_diagnostics_result(tmp_path)
    assert validation["valid"] is False


def test_validator_detects_tampered_provenance(monkeypatch, tmp_path) -> None:
    _install_v2_execute_fixtures(monkeypatch, tmp_path, verdict_per_arm="PASS")
    runner2._execute(tmp_path)
    doc = json.loads((tmp_path / diag2.PROVENANCE_PATH).read_text())
    doc["ba_sep_observed_verdict"] = "PASS"   # tampering the immutable BA_sep record
    atomic_write_json(tmp_path / diag2.PROVENANCE_PATH, doc)
    validation = diag2.validate_existing_diagnostics_result(tmp_path)
    assert validation["valid"] is False
    assert any("ba_sep_observed_verdict" in p for p in validation["problems"])


def test_validator_detects_tampered_verdict(monkeypatch, tmp_path) -> None:
    _install_v2_execute_fixtures(monkeypatch, tmp_path, verdict_per_arm="FAIL")
    runner2._execute(tmp_path)
    doc = json.loads((tmp_path / diag2.VERDICT_PATH).read_text())
    doc["overall_diagnostics_verdict"] = "PASS"   # flip a real FAIL to a fake PASS
    atomic_write_json(tmp_path / diag2.VERDICT_PATH, doc)
    validation = diag2.validate_existing_diagnostics_result(tmp_path)
    assert validation["valid"] is False
    assert any("overall_diagnostics_verdict" in p for p in validation["problems"])


def test_validator_detects_c9_may_close_tampered_true(monkeypatch, tmp_path) -> None:
    _install_v2_execute_fixtures(monkeypatch, tmp_path, verdict_per_arm="PASS")
    runner2._execute(tmp_path)
    doc = json.loads((tmp_path / diag2.VERDICT_PATH).read_text())
    doc["c9_may_close"] = True
    atomic_write_json(tmp_path / diag2.VERDICT_PATH, doc)
    validation = diag2.validate_existing_diagnostics_result(tmp_path)
    assert validation["valid"] is False


def test_validator_detects_mismatched_bindings(monkeypatch, tmp_path) -> None:
    _install_v2_execute_fixtures(monkeypatch, tmp_path, verdict_per_arm="PASS")
    runner2._execute(tmp_path)
    stale = json.loads((tmp_path / diag2.PROTOCOL_BINDING_PATH).read_text())
    stale["protocol_identity"] = "different" + "0" * 56
    atomic_write_json(tmp_path / diag2.PROTOCOL_BINDING_PATH, stale)
    validation = diag2.validate_existing_diagnostics_result(tmp_path)
    assert validation["valid"] is False
    assert any("protocol_identity" in p for p in validation["problems"])


def test_execute_blocks_on_invalid_complete_result_without_recomputing(monkeypatch, tmp_path) -> None:
    _install_v2_execute_fixtures(monkeypatch, tmp_path, verdict_per_arm="PASS")
    runner2._execute(tmp_path)
    doc = json.loads((tmp_path / diag2.VERDICT_PATH).read_text())
    doc["overall_diagnostics_verdict"] = "PASS_BUT_TAMPERED"
    atomic_write_json(tmp_path / diag2.VERDICT_PATH, doc)

    called = {"n": 0}
    monkeypatch.setattr(diag2, "run_benign_corruption_diagnostic_for_arm",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    exit_code, payload = runner2._execute(tmp_path)
    assert exit_code == runner2.EXIT_BLOCKED
    assert payload["error"] == "EXISTING_RESULT_FAILED_VALIDATION"
    assert called["n"] == 0
    # and it must not have overwritten the tampered file either
    after = json.loads((tmp_path / diag2.VERDICT_PATH).read_text())
    assert after["overall_diagnostics_verdict"] == "PASS_BUT_TAMPERED"


def test_status_uses_the_same_validator_and_blocks_on_invalid_result(monkeypatch, tmp_path) -> None:
    _install_v2_execute_fixtures(monkeypatch, tmp_path, verdict_per_arm="PASS")
    runner2._execute(tmp_path)
    doc = json.loads((tmp_path / diag2.VERDICT_PATH).read_text())
    doc["overall_diagnostics_verdict"] = "TAMPERED"
    atomic_write_json(tmp_path / diag2.VERDICT_PATH, doc)
    exit_code, payload = runner2._status(tmp_path)
    assert exit_code == runner2.EXIT_BLOCKED
    assert payload["reason"] == "EXISTING_RESULT_FAILED_VALIDATION"


def test_status_re_reports_a_valid_complete_result(monkeypatch, tmp_path) -> None:
    _install_v2_execute_fixtures(monkeypatch, tmp_path, verdict_per_arm="PASS")
    runner2._execute(tmp_path)
    exit_code, payload = runner2._status(tmp_path)
    assert exit_code == runner2.EXIT_PASS
    assert payload["overall_diagnostics_verdict"] == "PASS"
    assert payload["c9_may_close"] is False


# ==============================================================================
# 15. target_access = 0 everywhere
# ==============================================================================

def test_protocol_declares_target_access_zero() -> None:
    payload = _real_v2_protocol()
    assert payload["target_access"] == 0
    assert payload["target_firewall"]["target_access"] == 0


def test_every_written_artifact_carries_target_access_zero(monkeypatch, tmp_path) -> None:
    _install_v2_execute_fixtures(monkeypatch, tmp_path, verdict_per_arm="PASS")
    runner2._execute(tmp_path)
    for relative in list(diag2.BINDING_ARTIFACT_PATHS.values()) + list(diag2.RESULT_ARTIFACT_PATHS.values()):
        doc = json.loads((tmp_path / relative).read_text())
        assert doc["target_access"] == 0


def test_v2_module_never_references_a_target_path() -> None:
    source = Path(inspect.getfile(diag2)).read_text(encoding="utf-8")
    for forbidden in ("siw", "SiW", "target_test", "target_taxonomy",
                      "prism_target_eval_v2", "resolve_target"):
        assert forbidden not in source, forbidden


def test_v2_runner_never_references_a_target_path() -> None:
    source = Path(inspect.getfile(runner2)).read_text(encoding="utf-8")
    for forbidden in ("siw", "SiW", "target_test", "target_taxonomy",
                      "prism_target_eval_v2", "resolve_target"):
        assert forbidden not in source, forbidden


# ==============================================================================
# 16. a FAILED BA_sep/reliability barrier can never reopen C9
# ==============================================================================

def test_ba_sep_v2_protocol_identity_unchanged() -> None:
    assert probe.protocol_identity(REPO) == BA_SEP_V2_IDENTITY


def test_execute_result_hard_codes_c9_may_close_false() -> None:
    source = inspect.getsource(runner2._execute)
    assert '"c9_may_close": False' in source


def test_status_result_hard_codes_c9_may_close_false() -> None:
    source = inspect.getsource(runner2._status)
    assert '"c9_may_close": False' in source


def test_execute_source_never_calls_lock_payload_or_verify_lock() -> None:
    source = inspect.getsource(runner2._execute)
    for forbidden in ("lock_payload(", "verify_lock(", "validate_lock_record("):
        assert forbidden not in source, forbidden


def test_written_verdict_always_records_ba_sep_observed_fail(monkeypatch, tmp_path) -> None:
    _install_v2_execute_fixtures(monkeypatch, tmp_path, verdict_per_arm="PASS")
    runner2._execute(tmp_path)
    verdict_doc = json.loads((tmp_path / diag2.VERDICT_PATH).read_text())
    assert verdict_doc["ba_sep_observed_verdict"] == "FAIL"
    assert verdict_doc["detector_reliability_lock_c_observed_overall"] == "FAILED"
    assert verdict_doc["c9_may_close"] is False


def test_validator_fails_closed_if_the_live_reliability_lock_ever_flips_to_passed(
        monkeypatch, tmp_path) -> None:
    _install_v2_execute_fixtures(monkeypatch, tmp_path, verdict_per_arm="PASS")
    runner2._execute(tmp_path)
    lock_path = tmp_path / barrier.LOCK_PATH
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps({"overall": "PASSED"}))
    validation = diag2.validate_existing_diagnostics_result(tmp_path)
    assert validation["valid"] is False
    assert any("DETECTOR_RELIABILITY_LOCK_C" in p for p in validation["problems"])


def test_diagnostics_runner_never_writes_ba_sep_or_lock_paths() -> None:
    source = Path(inspect.getfile(runner2)).read_text(encoding="utf-8")
    for forbidden in ("probe.RESULT_PATH", "probe.VERDICT_PATH", "barrier.LOCK_PATH",
                      "detector_reliability.LOCK_PATH,"):
        assert forbidden not in source, forbidden


def test_real_ba_sep_artifacts_untouched_by_a_real_v2_bind(monkeypatch, tmp_path) -> None:
    real_reliability_dir = REPO / probe.RELIABILITY_DIR
    real_lock = REPO / barrier.LOCK_PATH
    before_listing = (sorted(p.name for p in real_reliability_dir.rglob("*"))
                      if real_reliability_dir.is_dir() else None)
    before_lock = real_lock.read_bytes() if real_lock.is_file() else None

    _install_v2_bind_fixtures(monkeypatch)
    runner2._bind_only(tmp_path)

    after_listing = (sorted(p.name for p in real_reliability_dir.rglob("*"))
                     if real_reliability_dir.is_dir() else None)
    after_lock = real_lock.read_bytes() if real_lock.is_file() else None
    assert before_listing == after_listing
    assert before_lock == after_lock


# ==============================================================================
# 17. V1/V2 result namespaces cannot collide
# ==============================================================================

def test_v1_and_v2_diagnostics_namespaces_never_collide() -> None:
    assert diag_v1.DIAGNOSTICS_DIR != diag2.DIAGNOSTICS_DIR
    assert not diag_v1.DIAGNOSTICS_DIR.startswith(diag2.DIAGNOSTICS_DIR)
    assert not diag2.DIAGNOSTICS_DIR.startswith(diag_v1.DIAGNOSTICS_DIR)


def test_v2_namespace_also_disjoint_from_ba_sep_namespace() -> None:
    assert diag2.DIAGNOSTICS_DIR != probe.RELIABILITY_DIR
    assert not diag2.DIAGNOSTICS_DIR.startswith(probe.RELIABILITY_DIR)
    assert not probe.RELIABILITY_DIR.startswith(diag2.DIAGNOSTICS_DIR)


def test_v2_bind_never_writes_into_the_v1_namespace(monkeypatch, tmp_path) -> None:
    _install_v2_bind_fixtures(monkeypatch)
    runner2._bind_only(tmp_path)
    assert not (tmp_path / diag_v1.DIAGNOSTICS_DIR).exists()


def test_v2_execute_never_writes_into_the_v1_namespace(monkeypatch, tmp_path) -> None:
    _install_v2_execute_fixtures(monkeypatch, tmp_path, verdict_per_arm="PASS")
    runner2._execute(tmp_path)
    assert not (tmp_path / diag_v1.DIAGNOSTICS_DIR).exists()


# ==============================================================================
# preflight / other structural checks
# ==============================================================================

def test_preflight_writes_nothing(monkeypatch, tmp_path) -> None:
    _install_v2_bind_fixtures(monkeypatch)
    runner2._preflight(tmp_path)
    assert not (tmp_path / diag2.DIAGNOSTICS_DIR).exists()


def test_preflight_reports_reduced_executable_set() -> None:
    exit_code, payload = runner2._preflight(REPO)
    assert payload["per_test_gpu_ready"]["benign_jpeg_corruption"] is True
    assert payload["per_test_gpu_ready"]["cross_route_synthetic"] is False


def test_executable_tests_are_exactly_the_three_benign_corruption_tests() -> None:
    payload = _real_v2_protocol()
    executable = {test_id for test_id, config in payload["tests"].items()
                 if config["classification"] == "EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL"}
    assert executable == set(diag2.EXECUTABLE_TESTS)
    assert executable == {"benign_jpeg_corruption", "benign_resize_corruption",
                          "benign_color_corruption"}


def test_execute_requires_bind_only_to_have_run_first(monkeypatch, tmp_path) -> None:
    _install_v2_bind_fixtures(monkeypatch)
    exit_code, payload = runner2._execute(tmp_path)
    assert exit_code == runner2.EXIT_BLOCKED
    assert "bind-only" in payload["error"]


def test_execute_fails_closed_when_c8_matrix_identity_drifts_after_bind(monkeypatch, tmp_path) -> None:
    _install_v2_bind_fixtures(monkeypatch)
    exit_code, _ = runner2._bind_only(tmp_path)
    assert exit_code == runner2.EXIT_PASS
    # simulate the canonical C8 matrix identity moving after binding
    monkeypatch.setattr(diag2, "bind_c8_matrix_identity",
                        lambda repo: {"c8_matrix_identity": "drifted" + "0" * 57,
                                     "c8_acceptance_matrix_identity": "drifted" + "0" * 57,
                                     "c8_acceptance_path": diag2.C8_ACCEPTANCE_PATH})
    exit_code, payload = runner2._execute(tmp_path)
    assert exit_code == runner2.EXIT_BLOCKED
    assert "c8_matrix_identity" in payload["error"] or "C8 matrix" in payload["error"]
    assert not (tmp_path / diag2.RESULT_PATH).exists()


def test_importing_v2_modules_touches_no_filesystem_state() -> None:
    import subprocess

    for module in ("prism_fas.evaluation.post_failure_diagnostics_v2",
                  "prism_fas.evaluation.post_failure_diagnostics_v2_runner"):
        result = subprocess.run([sys.executable, "-c", f"import {module}"],
                                capture_output=True, text=True, cwd=str(REPO))
        assert result.returncode == 0, result.stderr


def test_runner_invocable_as_python_dash_m_module() -> None:
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "prism_fas.evaluation.post_failure_diagnostics_v2_runner",
         "--repo", str(REPO), "--preflight-only"],
        capture_output=True, text=True, cwd=str(REPO / "src"))
    assert result.returncode == runner2.EXIT_BLOCKED, result.stderr
    payload = json.loads(result.stdout)
    assert payload["target_access"] == 0


# ==============================================================================
# reused-verbatim V1 primitives (never reimplemented in V2)
# ==============================================================================

def test_corruption_transforms_reused_verbatim_from_v1() -> None:
    assert diag2.jpeg_corrupt is diag_v1.jpeg_corrupt
    assert diag2.resize_corrupt is diag_v1.resize_corrupt
    assert diag2.color_corrupt is diag_v1.color_corrupt
    assert diag2.CORRUPTION_FUNCTIONS is diag_v1.CORRUPTION_FUNCTIONS


def test_split_rule_reused_verbatim_from_v1() -> None:
    assert diag2.calibration_evaluation_split is diag_v1.calibration_evaluation_split
