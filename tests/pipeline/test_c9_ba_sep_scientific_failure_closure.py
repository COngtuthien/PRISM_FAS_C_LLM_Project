"""C9 BA_sep Option-1 V2 — scientific FAILURE closure.

The real GPU host completed the FIRST and ONLY authorized real BA_sep
Option-1 V2 execution; the observed scientific hard verdict is FAILED
(RND=0.7843079833902619, DET=0.8514170182841069, LLM=0.7902658339472685,
all over the frozen 0.75 ceiling). This file proves, entirely against
fixtures built in `tmp_path` (never a real repo's artifacts, never a real
checkpoint, never a real image, never target data):

  * `--execute` is scientifically NO-RERUN: a complete existing result set
    is re-reported, never recomputed; a partial set blocks; a clean host
    still reaches the original execution path.
  * `synthetic_real_probe.validate_existing_scientific_result` strictly
    cross-checks a written result set and recomputes (never re-fits) the
    aggregate and the hard verdict from the RECORDED per-seed values.
  * `detector_reliability_runner --register-ba-sep-result` binds ONLY
    `synthetic_vs_real_spoof_probe` from the observed verdict; the barrier's
    overall is FAILED, the other eight required tests stay UNRESOLVED, and
    registration is idempotent (reused on identical content, refused on
    different content).
  * `detector_reliability.verify_lock` still refuses a FAILED barrier —
    unweakened — while the new `validate_lock_record` recognizes it as a
    structurally valid NEGATIVE result.
  * C9 rejects a FAILED reliability record via its own
    `semantic_preconditions`, exactly as it already rejects an absent one.
  * `synthetic_real_probe.c_h4_preconditions` reports the real observed
    values fail C-H4's basic conditions, without fabricating any bootstrap
    result.

Uses the REAL observed per-seed BA_sep values (arithmetic facts anyone can
recompute); uses SELF-CONSISTENT fixture identities throughout (this laptop
cannot reproduce the real GPU run's actual checkpoint/package/bank hashes,
and must not pretend to).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.evaluation import detector_reliability as barrier  # noqa: E402
from prism_fas.evaluation import detector_reliability_runner as reg_runner  # noqa: E402
from prism_fas.evaluation import synthetic_real_probe as probe  # noqa: E402
from prism_fas.evaluation import synthetic_real_probe_runner as runner  # noqa: E402
from prism_fas.pipeline.state import atomic_write_json  # noqa: E402

# --- the real observed per-seed values (arithmetic fact, not a fabrication) ---
OBSERVED_PER_SEED: dict[str, dict[int, float]] = {
    "RND": {20260806: 0.753968253968254, 20260807: 0.8164556962025317, 20260808: 0.7825},
    "DET": {20260806: 0.8333333333333333, 20260807: 0.8734177215189873, 20260808: 0.8474999999999999},
    "LLM": {20260806: 0.7486772486772486, 20260807: 0.819620253164557, 20260808: 0.8025},
}
OBSERVED_BA_SEP: dict[str, float] = {
    arm: sum(values.values()) / 3 for arm, values in OBSERVED_PER_SEED.items()}
PROTOCOL_SEEDS = [20260806, 20260807, 20260808]


def _fixture_checkpoints() -> list[dict[str, Any]]:
    checkpoints = []
    for arm in probe.ARMS:
        for seed in (20260806, 20260807, 20260808, 20260809, 20260810):
            checkpoints.append({
                "arm": arm, "seed": seed, "row_id": f"C-G-{arm}-P3READY-s{seed}",
                "run_identity": f"run-{arm}-{seed}", "config_identity": "c" * 64,
                "checkpoint_relative_path":
                    f"runs/full/c8/P3/C-G-{arm}/cfg/{seed}/checkpoints/best.pt",
                "checkpoint_sha256": f"{arm.lower()}{seed}".ljust(64, "0"),
                "decision_graph_hash": "g" * 64, "decision_logit_name": "global_logit_G"})
    return checkpoints


def write_ba_sep_result_set(repo: Path, *, ba_sep_by_arm: dict[str, float] | None = None,
                            per_seed_by_arm: dict[str, dict[int, float]] | None = None,
                            protocol_id: str = "p" * 64,
                            probe_seed_values: list[int] | None = None,
                            omit: set[str] | None = None,
                            corrupt: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write a fully self-consistent, group of seven BA_sep artifacts to
    `repo`, using the REAL observed per-seed values by default.

    `omit` names files to skip writing (for partial-result-set tests, by
    the SAME keys as `probe.RESULT_ARTIFACT_PATHS` plus `"binding"`/`"plan"`).
    `corrupt` is `{artifact_key: {json_key: new_value}}`, applied to a
    document right before it is written (for negative cross-check tests).
    Returns the dict of `{key: written_document}` (post-corruption) for
    assertions.
    """
    omit = omit or set()
    corrupt = corrupt or {}
    per_seed_by_arm = per_seed_by_arm or OBSERVED_PER_SEED
    if ba_sep_by_arm is None:
        ba_sep_by_arm = {arm: sum(values.values()) / len(values)
                        for arm, values in per_seed_by_arm.items()}
    probe_seed_values = probe_seed_values if probe_seed_values is not None else list(PROTOCOL_SEEDS)

    checkpoints = _fixture_checkpoints()
    binding = {
        "schema_version": "c9-ba-sep-execution-binding-v1", "protocol_identity": protocol_id,
        "source_package_identity": "pkg" + "0" * 61,
        "c6_bank_identities": {arm: f"bank-{arm}" + "0" * 55 for arm in probe.ARMS},
        "checkpoints": checkpoints,
        "checkpoints_per_arm": {arm: 5 for arm in probe.ARMS}, "total_checkpoints": 15,
        "target_access": 0,
    }
    binding["checkpoint_binding_identity_sha256"] = probe.checkpoint_binding_identity(binding)
    binding.update(corrupt.get("binding", {}))

    plan = {
        "schema_version": "c9-ba-sep-population-plan-v1", "protocol_identity": protocol_id,
        "split_hash_namespace": "ns", "probe_seed_values": probe_seed_values,
        "source_domains": ["casia_fasd", "msu_mfsd"], "cells": [], "leakage_audit": {},
        "target_access": 0,
    }
    plan["population_plan_identity_sha256"] = probe.population_plan_identity(plan)
    plan.update(corrupt.get("plan", {}))

    common = {
        "protocol_identity": protocol_id,
        "checkpoint_binding_identity": binding["checkpoint_binding_identity_sha256"],
        "population_plan_identity": plan["population_plan_identity_sha256"],
        "source_package_identity": binding["source_package_identity"],
        "c6_bank_identities": binding["c6_bank_identities"],
        "target_access": 0,
    }
    sha_by_arm = {arm: sorted(item["checkpoint_sha256"] for item in checkpoints
                              if item["arm"] == arm) for arm in probe.ARMS}

    result = {**common, "ba_sep_by_arm": ba_sep_by_arm, "checkpoint_sha256_by_arm": sha_by_arm}
    result.update(corrupt.get("result", {}))
    per_seed_doc = {**common, "per_seed_by_arm": per_seed_by_arm}
    per_seed_doc.update(corrupt.get("per_seed", {}))
    parameters = {**common, "probe_seed_values": probe_seed_values}
    parameters.update(corrupt.get("parameters", {}))
    evidence_manifest = {**common, "seed_details": {}}
    evidence_manifest.update(corrupt.get("evidence_manifest", {}))
    verdict_doc = {**common, "verdict": probe.hard_verdict(ba_sep_by_arm)}
    verdict_doc.update(corrupt.get("verdict", {}))

    docs = {"binding": binding, "plan": plan, "result": result, "per_seed": per_seed_doc,
           "parameters": parameters, "evidence_manifest": evidence_manifest,
           "verdict": verdict_doc}
    paths = {"binding": probe.EXECUTION_BINDING_PATH, "plan": probe.POPULATION_PLAN_PATH,
            "result": probe.RESULT_PATH, "per_seed": probe.PER_SEED_PATH,
            "parameters": probe.PARAMETERS_PATH,
            "evidence_manifest": probe.EVIDENCE_MANIFEST_PATH, "verdict": probe.VERDICT_PATH}
    for key, relative in paths.items():
        if key not in omit:
            atomic_write_json(Path(repo) / relative, docs[key])
    return docs


def _patch_active_protocol(monkeypatch, protocol_id: str = "p" * 64) -> None:
    monkeypatch.setattr(probe, "load_protocol",
                        lambda repo: {"probe_seed_values": list(PROTOCOL_SEEDS)})
    monkeypatch.setattr(probe, "protocol_identity", lambda repo: protocol_id)


# ==============================================================================
# A. result immutability / no-rerun
# ==============================================================================

def test_no_result_files_leaves_the_first_execution_path_reachable(monkeypatch, tmp_path) -> None:
    """A clean host (nothing written) must fall through to the ORIGINAL
    execution path — proven here by the familiar "no binding on disk"
    refusal from that path, not by the no-rerun guard."""
    _patch_active_protocol(monkeypatch)
    exit_code, payload = runner._execute(tmp_path)
    assert exit_code == runner.EXIT_BLOCKED
    assert "bind-only" in payload["error"]


def test_all_five_result_files_present_reports_no_trainer_construction(monkeypatch, tmp_path) -> None:
    _patch_active_protocol(monkeypatch)
    write_ba_sep_result_set(tmp_path, ba_sep_by_arm=OBSERVED_BA_SEP,
                            per_seed_by_arm=OBSERVED_PER_SEED)
    called = {"construct": 0}
    monkeypatch.setattr(probe, "construct_row_trainer",
                        lambda repo, binding: called.__setitem__("construct", called["construct"] + 1))
    exit_code, payload = runner._execute(tmp_path)
    assert called["construct"] == 0
    assert payload["reused_existing_scientific_result"] is True


def test_all_five_result_files_present_no_checkpoint_load(monkeypatch, tmp_path) -> None:
    _patch_active_protocol(monkeypatch)
    write_ba_sep_result_set(tmp_path, ba_sep_by_arm=OBSERVED_BA_SEP,
                            per_seed_by_arm=OBSERVED_PER_SEED)
    exit_code, payload = runner._execute(tmp_path)
    assert payload["checkpoint_weights_loaded"] is False


def test_all_five_result_files_present_no_detector_forward(monkeypatch, tmp_path) -> None:
    _patch_active_protocol(monkeypatch)
    write_ba_sep_result_set(tmp_path, ba_sep_by_arm=OBSERVED_BA_SEP,
                            per_seed_by_arm=OBSERVED_PER_SEED)
    called = {"forward": 0}
    monkeypatch.setattr(
        probe, "forward_evidence_for_records",
        lambda trainer, records: called.__setitem__("forward", called["forward"] + 1))
    exit_code, payload = runner._execute(tmp_path)
    assert called["forward"] == 0
    assert payload["images_forwarded"] is False


def test_all_five_result_files_present_no_probe_fit(monkeypatch, tmp_path) -> None:
    _patch_active_protocol(monkeypatch)
    write_ba_sep_result_set(tmp_path, ba_sep_by_arm=OBSERVED_BA_SEP,
                            per_seed_by_arm=OBSERVED_PER_SEED)
    called = {"fit": 0}
    monkeypatch.setattr(
        probe, "fit_linear_probe",
        lambda *a, **k: called.__setitem__("fit", called["fit"] + 1))
    exit_code, payload = runner._execute(tmp_path)
    assert called["fit"] == 0
    assert payload["probe_fit_executed"] is False


def test_existing_pass_result_re_reports_exit_0(monkeypatch, tmp_path) -> None:
    _patch_active_protocol(monkeypatch)
    passing = {"RND": 0.5, "DET": 0.5, "LLM": 0.5}
    per_seed = {arm: {seed: 0.5 for seed in PROTOCOL_SEEDS} for arm in probe.ARMS}
    write_ba_sep_result_set(tmp_path, ba_sep_by_arm=passing, per_seed_by_arm=per_seed)
    exit_code, payload = runner._execute(tmp_path)
    assert exit_code == runner.EXIT_PASS
    assert payload["verdict"] == "PASS"


def test_existing_fail_result_re_reports_exit_1(monkeypatch, tmp_path) -> None:
    _patch_active_protocol(monkeypatch)
    write_ba_sep_result_set(tmp_path, ba_sep_by_arm=OBSERVED_BA_SEP,
                            per_seed_by_arm=OBSERVED_PER_SEED)
    exit_code, payload = runner._execute(tmp_path)
    assert exit_code == runner.EXIT_FAIL
    assert payload["verdict"] == "FAIL"
    assert payload["ba_sep_by_arm"] == pytest.approx(OBSERVED_BA_SEP)


def test_partial_result_set_blocks_with_exit_2(monkeypatch, tmp_path) -> None:
    _patch_active_protocol(monkeypatch)
    write_ba_sep_result_set(tmp_path, ba_sep_by_arm=OBSERVED_BA_SEP,
                            per_seed_by_arm=OBSERVED_PER_SEED,
                            omit={"verdict"})   # 4/5 result files only
    exit_code, payload = runner._execute(tmp_path)
    assert exit_code == runner.EXIT_BLOCKED
    assert payload["error"] == "PARTIAL_SCIENTIFIC_RESULT_SET"
    assert "verdict" in payload["missing"]


def test_existing_result_files_are_never_overwritten_by_a_second_execute(
        monkeypatch, tmp_path) -> None:
    _patch_active_protocol(monkeypatch)
    write_ba_sep_result_set(tmp_path, ba_sep_by_arm=OBSERVED_BA_SEP,
                            per_seed_by_arm=OBSERVED_PER_SEED)
    before = (tmp_path / probe.RESULT_PATH).read_text()
    runner._execute(tmp_path)
    runner._execute(tmp_path)
    after = (tmp_path / probe.RESULT_PATH).read_text()
    assert before == after


# ==============================================================================
# B. result verifier
# ==============================================================================

def test_verifier_requires_matching_protocol_identity_across_all_artifacts(
        monkeypatch, tmp_path) -> None:
    _patch_active_protocol(monkeypatch)
    write_ba_sep_result_set(tmp_path, ba_sep_by_arm=OBSERVED_BA_SEP,
                            per_seed_by_arm=OBSERVED_PER_SEED,
                            corrupt={"result": {"protocol_identity": "stale" + "0" * 59}})
    result = probe.validate_existing_scientific_result(tmp_path)
    assert result["valid"] is False
    assert any("protocol_identity" in problem for problem in result["problems"])


def test_verifier_requires_matching_checkpoint_binding_identity(monkeypatch, tmp_path) -> None:
    _patch_active_protocol(monkeypatch)
    write_ba_sep_result_set(tmp_path, ba_sep_by_arm=OBSERVED_BA_SEP,
                            per_seed_by_arm=OBSERVED_PER_SEED,
                            corrupt={"per_seed": {"checkpoint_binding_identity": "x" * 64}})
    result = probe.validate_existing_scientific_result(tmp_path)
    assert result["valid"] is False
    assert any("checkpoint_binding_identity" in problem for problem in result["problems"])


def test_verifier_requires_matching_population_plan_identity(monkeypatch, tmp_path) -> None:
    _patch_active_protocol(monkeypatch)
    write_ba_sep_result_set(tmp_path, ba_sep_by_arm=OBSERVED_BA_SEP,
                            per_seed_by_arm=OBSERVED_PER_SEED,
                            corrupt={"parameters": {"population_plan_identity": "y" * 64}})
    result = probe.validate_existing_scientific_result(tmp_path)
    assert result["valid"] is False
    assert any("population_plan_identity" in problem for problem in result["problems"])


def test_verifier_requires_matching_source_package_identity(monkeypatch, tmp_path) -> None:
    _patch_active_protocol(monkeypatch)
    write_ba_sep_result_set(tmp_path, ba_sep_by_arm=OBSERVED_BA_SEP,
                            per_seed_by_arm=OBSERVED_PER_SEED,
                            corrupt={"verdict": {"source_package_identity": "z" * 64}})
    result = probe.validate_existing_scientific_result(tmp_path)
    assert result["valid"] is False
    assert any("source_package_identity" in problem for problem in result["problems"])


def test_verifier_requires_matching_c6_bank_identities(monkeypatch, tmp_path) -> None:
    _patch_active_protocol(monkeypatch)
    write_ba_sep_result_set(tmp_path, ba_sep_by_arm=OBSERVED_BA_SEP,
                            per_seed_by_arm=OBSERVED_PER_SEED,
                            corrupt={"evidence_manifest": {"c6_bank_identities": {"RND": "w" * 64}}})
    result = probe.validate_existing_scientific_result(tmp_path)
    assert result["valid"] is False
    assert any("c6_bank_identities" in problem for problem in result["problems"])


def test_verifier_requires_exactly_fifteen_checkpoint_hashes(monkeypatch, tmp_path) -> None:
    _patch_active_protocol(monkeypatch)
    docs = write_ba_sep_result_set(tmp_path, ba_sep_by_arm=OBSERVED_BA_SEP,
                                   per_seed_by_arm=OBSERVED_PER_SEED)
    # drop one DET checkpoint from the binding after the fact
    binding = json.loads((tmp_path / probe.EXECUTION_BINDING_PATH).read_text())
    binding["checkpoints"] = [c for c in binding["checkpoints"]
                              if not (c["arm"] == "DET" and c["seed"] == 20260810)]
    atomic_write_json(tmp_path / probe.EXECUTION_BINDING_PATH, binding)
    result = probe.validate_existing_scientific_result(tmp_path)
    assert result["valid"] is False
    assert any("exactly" in problem for problem in result["problems"])


def test_verifier_requires_exact_5_5_5_arm_counts(monkeypatch, tmp_path) -> None:
    _patch_active_protocol(monkeypatch)
    write_ba_sep_result_set(tmp_path, ba_sep_by_arm=OBSERVED_BA_SEP,
                            per_seed_by_arm=OBSERVED_PER_SEED)
    result = probe.validate_existing_scientific_result(tmp_path)
    assert result["valid"] is True
    assert result["checkpoints_per_arm"] == {"RND": 5, "DET": 5, "LLM": 5}
    assert result["total_checkpoints"] == 15


def test_verifier_requires_exactly_the_three_frozen_probe_seeds(monkeypatch, tmp_path) -> None:
    _patch_active_protocol(monkeypatch)
    bad_per_seed = dict(OBSERVED_PER_SEED)
    bad_per_seed["RND"] = {20260806: 0.75, 20260807: 0.80, 99999999: 0.78}   # wrong seed
    write_ba_sep_result_set(tmp_path, per_seed_by_arm=bad_per_seed)
    result = probe.validate_existing_scientific_result(tmp_path)
    assert result["valid"] is False
    assert any("seeds" in problem for problem in result["problems"])


def test_verifier_recomputes_the_aggregate_from_recorded_per_seed_values(
        monkeypatch, tmp_path) -> None:
    _patch_active_protocol(monkeypatch)
    write_ba_sep_result_set(tmp_path, ba_sep_by_arm=OBSERVED_BA_SEP,
                            per_seed_by_arm=OBSERVED_PER_SEED)
    result = probe.validate_existing_scientific_result(tmp_path)
    assert result["valid"] is True
    for arm in probe.ARMS:
        assert result["ba_sep_by_arm"][arm] == pytest.approx(OBSERVED_BA_SEP[arm], abs=1e-9)


def test_verifier_rejects_a_recorded_aggregate_that_disagrees_with_its_own_per_seed(
        monkeypatch, tmp_path) -> None:
    _patch_active_protocol(monkeypatch)
    tampered = dict(OBSERVED_BA_SEP)
    tampered["DET"] = 0.1   # does not match the mean of OBSERVED_PER_SEED["DET"]
    write_ba_sep_result_set(tmp_path, ba_sep_by_arm=tampered, per_seed_by_arm=OBSERVED_PER_SEED)
    result = probe.validate_existing_scientific_result(tmp_path)
    assert result["valid"] is False
    assert any("DET" in problem and "recomputed" in problem for problem in result["problems"])


def test_verifier_requires_the_recorded_verdict_equal_the_recomputed_hard_verdict(
        monkeypatch, tmp_path) -> None:
    _patch_active_protocol(monkeypatch)
    write_ba_sep_result_set(tmp_path, ba_sep_by_arm=OBSERVED_BA_SEP,
                            per_seed_by_arm=OBSERVED_PER_SEED,
                            corrupt={"verdict": {"verdict": {"verdict": "PASS"}}})
    result = probe.validate_existing_scientific_result(tmp_path)
    assert result["valid"] is False
    assert any("verdict" in problem.lower() for problem in result["problems"])


def test_verifier_requires_target_access_zero_everywhere(monkeypatch, tmp_path) -> None:
    _patch_active_protocol(monkeypatch)
    write_ba_sep_result_set(tmp_path, ba_sep_by_arm=OBSERVED_BA_SEP,
                            per_seed_by_arm=OBSERVED_PER_SEED,
                            corrupt={"result": {"target_access": 1}})
    result = probe.validate_existing_scientific_result(tmp_path)
    assert result["valid"] is False
    assert any("target_access" in problem for problem in result["problems"])


def test_verifier_valid_on_the_real_observed_failed_result(monkeypatch, tmp_path) -> None:
    """The exact real GPU scenario: a genuinely self-consistent FAILED
    result set, with the real observed per-seed and aggregate values,
    validates cleanly."""
    _patch_active_protocol(monkeypatch)
    write_ba_sep_result_set(tmp_path, ba_sep_by_arm=OBSERVED_BA_SEP,
                            per_seed_by_arm=OBSERVED_PER_SEED)
    result = probe.validate_existing_scientific_result(tmp_path)
    assert result["valid"] is True
    assert result["scientific_verdict"] == "FAIL"
    assert result["target_access"] == 0


def test_verifier_never_constructs_a_trainer_or_fits_a_probe() -> None:
    source = __import__("inspect").getsource(probe.validate_existing_scientific_result)
    for forbidden in ("construct_row_trainer(", "forward_evidence_for_records(",
                      "fit_linear_probe(", "torch.load", "Image.open", "cv2.imread"):
        assert forbidden not in source, forbidden


# ==============================================================================
# C. failed barrier registration
# ==============================================================================

def test_registration_binds_ba_fail_to_synthetic_vs_real_spoof_probe_failed(
        monkeypatch, tmp_path) -> None:
    _patch_active_protocol(monkeypatch)
    write_ba_sep_result_set(tmp_path, ba_sep_by_arm=OBSERVED_BA_SEP,
                            per_seed_by_arm=OBSERVED_PER_SEED)
    exit_code, payload = reg_runner._register(tmp_path)
    assert exit_code == reg_runner.EXIT_FAIL
    assert payload["per_test"]["synthetic_vs_real_spoof_probe"] == barrier.FAILED


def test_registration_leaves_the_other_eight_required_tests_unresolved(
        monkeypatch, tmp_path) -> None:
    _patch_active_protocol(monkeypatch)
    write_ba_sep_result_set(tmp_path, ba_sep_by_arm=OBSERVED_BA_SEP,
                            per_seed_by_arm=OBSERVED_PER_SEED)
    _, payload = reg_runner._register(tmp_path)
    others = [name for name in barrier.REQUIRED_DETECTOR_RELIABILITY_TESTS
             if name != "synthetic_vs_real_spoof_probe"]
    assert all(payload["per_test"][name] == barrier.UNRESOLVED for name in others)


def test_registration_overall_is_failed(monkeypatch, tmp_path) -> None:
    _patch_active_protocol(monkeypatch)
    write_ba_sep_result_set(tmp_path, ba_sep_by_arm=OBSERVED_BA_SEP,
                            per_seed_by_arm=OBSERVED_PER_SEED)
    _, payload = reg_runner._register(tmp_path)
    assert payload["overall"] == barrier.FAILED


def test_registration_c9_may_close_is_false(monkeypatch, tmp_path) -> None:
    _patch_active_protocol(monkeypatch)
    write_ba_sep_result_set(tmp_path, ba_sep_by_arm=OBSERVED_BA_SEP,
                            per_seed_by_arm=OBSERVED_PER_SEED)
    _, payload = reg_runner._register(tmp_path)
    assert payload["c9_may_close"] is False


def test_registration_uses_lock_payload_not_a_second_schema(monkeypatch, tmp_path) -> None:
    _patch_active_protocol(monkeypatch)
    write_ba_sep_result_set(tmp_path, ba_sep_by_arm=OBSERVED_BA_SEP,
                            per_seed_by_arm=OBSERVED_PER_SEED)
    reg_runner._register(tmp_path)
    lock = json.loads((tmp_path / barrier.LOCK_PATH).read_text())
    assert lock["schema_version"] == barrier.SCHEMA_VERSION
    assert lock["stage"] == barrier.STAGE
    assert "identity_sha256" in lock


def test_registration_binds_all_fifteen_checkpoint_row_identities(monkeypatch, tmp_path) -> None:
    _patch_active_protocol(monkeypatch)
    write_ba_sep_result_set(tmp_path, ba_sep_by_arm=OBSERVED_BA_SEP,
                            per_seed_by_arm=OBSERVED_PER_SEED)
    reg_runner._register(tmp_path)
    lock = json.loads((tmp_path / barrier.LOCK_PATH).read_text())
    identities = lock["detector_checkpoint_identities"]
    assert len(identities) == 15
    for arm in probe.ARMS:
        for seed in (20260806, 20260807, 20260808, 20260809, 20260810):
            assert f"C-G-{arm}-P3READY-s{seed}" in identities


def test_repeated_identical_registration_reuses(monkeypatch, tmp_path) -> None:
    _patch_active_protocol(monkeypatch)
    write_ba_sep_result_set(tmp_path, ba_sep_by_arm=OBSERVED_BA_SEP,
                            per_seed_by_arm=OBSERVED_PER_SEED)
    reg_runner._register(tmp_path)
    before = (tmp_path / barrier.LOCK_PATH).read_text()
    exit_code, payload = reg_runner._register(tmp_path)
    after = (tmp_path / barrier.LOCK_PATH).read_text()
    assert payload["reused"] is True
    assert before == after
    assert exit_code == reg_runner.EXIT_FAIL


def test_a_different_existing_lock_blocks_overwrite(monkeypatch, tmp_path) -> None:
    _patch_active_protocol(monkeypatch)
    write_ba_sep_result_set(tmp_path, ba_sep_by_arm=OBSERVED_BA_SEP,
                            per_seed_by_arm=OBSERVED_PER_SEED)
    stale_payload = barrier.lock_payload(
        results={"synthetic_vs_real_spoof_probe": barrier.PASSED},
        probe_protocol_identity="stale" * 16,
        detector_checkpoint_identities={"stale-row": "s" * 64})
    atomic_write_json(tmp_path / barrier.LOCK_PATH, stale_payload)
    before = (tmp_path / barrier.LOCK_PATH).read_text()
    exit_code, payload = reg_runner._register(tmp_path)
    after = (tmp_path / barrier.LOCK_PATH).read_text()
    assert exit_code == reg_runner.EXIT_BLOCKED
    assert payload["reason"] == "EXISTING_LOCK_HAS_DIFFERENT_SCIENTIFIC_CONTENT"
    assert before == after   # never overwritten


def test_registration_never_mutates_the_seven_ba_sep_artifacts(monkeypatch, tmp_path) -> None:
    _patch_active_protocol(monkeypatch)
    write_ba_sep_result_set(tmp_path, ba_sep_by_arm=OBSERVED_BA_SEP,
                            per_seed_by_arm=OBSERVED_PER_SEED)
    snapshots = {}
    for key, relative in (("binding", probe.EXECUTION_BINDING_PATH),
                          ("plan", probe.POPULATION_PLAN_PATH),
                          ("result", probe.RESULT_PATH), ("per_seed", probe.PER_SEED_PATH),
                          ("parameters", probe.PARAMETERS_PATH),
                          ("evidence_manifest", probe.EVIDENCE_MANIFEST_PATH),
                          ("verdict", probe.VERDICT_PATH)):
        snapshots[key] = (tmp_path / relative).read_text()
    reg_runner._register(tmp_path)
    for key, relative in (("binding", probe.EXECUTION_BINDING_PATH),
                          ("plan", probe.POPULATION_PLAN_PATH),
                          ("result", probe.RESULT_PATH), ("per_seed", probe.PER_SEED_PATH),
                          ("parameters", probe.PARAMETERS_PATH),
                          ("evidence_manifest", probe.EVIDENCE_MANIFEST_PATH),
                          ("verdict", probe.VERDICT_PATH)):
        assert (tmp_path / relative).read_text() == snapshots[key], key


def test_register_source_writes_only_the_lock() -> None:
    source = __import__("inspect").getsource(reg_runner._register)
    assert "atomic_write_json(lock_path" in source
    assert source.count("atomic_write_json(") == 1   # writes ONLY the lock


# ==============================================================================
# D. C9 fail closed
# ==============================================================================

def test_verify_lock_on_a_failed_record_returns_valid_false(tmp_path) -> None:
    payload = barrier.lock_payload(
        results={"synthetic_vs_real_spoof_probe": barrier.FAILED},
        probe_protocol_identity="p" * 64,
        detector_checkpoint_identities={f"row{i}": "a" * 64 for i in range(15)},
        ba_sep_by_arm=OBSERVED_BA_SEP)
    atomic_write_json(tmp_path / barrier.LOCK_PATH, payload)
    verification = barrier.verify_lock(tmp_path)
    assert verification["valid"] is False
    assert any("overall" in problem for problem in verification["problems"])


def test_validate_lock_record_recognizes_the_same_record_as_structurally_valid(tmp_path) -> None:
    payload = barrier.lock_payload(
        results={"synthetic_vs_real_spoof_probe": barrier.FAILED},
        probe_protocol_identity="p" * 64,
        detector_checkpoint_identities={f"row{i}": "a" * 64 for i in range(15)},
        ba_sep_by_arm=OBSERVED_BA_SEP)
    atomic_write_json(tmp_path / barrier.LOCK_PATH, payload)
    record = barrier.validate_lock_record(tmp_path)
    assert record["valid"] is True
    assert record["overall"] == barrier.FAILED
    assert record["c9_may_close"] is False


def test_c9_semantic_preconditions_reject_a_failed_reliability_record(tmp_path) -> None:
    from prism_fas.pipeline.adapters import AdapterRequest
    from prism_fas.pipeline.adapters.c9 import C9Adapter
    from prism_fas.pipeline.profiles import load_profile

    payload = barrier.lock_payload(
        results={"synthetic_vs_real_spoof_probe": barrier.FAILED},
        probe_protocol_identity="p" * 64,
        detector_checkpoint_identities={f"row{i}": "a" * 64 for i in range(15)},
        ba_sep_by_arm=OBSERVED_BA_SEP)
    atomic_write_json(tmp_path / barrier.LOCK_PATH, payload)

    request = AdapterRequest(repo=tmp_path, profile=load_profile("full", repo=REPO))
    checks = C9Adapter().semantic_preconditions(request)
    reliability_check = next(c for c in checks if c["name"] == "detector_reliability_resolved")
    assert reliability_check["present"] is False
    assert reliability_check["blocking"] is True


def test_c9_semantic_preconditions_reject_an_absent_reliability_record(tmp_path) -> None:
    from prism_fas.pipeline.adapters import AdapterRequest
    from prism_fas.pipeline.adapters.c9 import C9Adapter
    from prism_fas.pipeline.profiles import load_profile

    request = AdapterRequest(repo=tmp_path, profile=load_profile("full", repo=REPO))
    checks = C9Adapter().semantic_preconditions(request)
    reliability_check = next(c for c in checks if c["name"] == "detector_reliability_resolved")
    assert reliability_check["blocking"] is True


def test_c9_never_reaches_a_target_path_after_a_failed_reliability_record() -> None:
    """`semantic_preconditions` is what blocking gates on; the C9 module's
    own source must never resolve a target path regardless of any
    reliability record's content."""
    source = Path(__import__("inspect").getfile(
        __import__("prism_fas.pipeline.adapters.c9", fromlist=["C9Adapter"]))
    ).read_text(encoding="utf-8")
    for forbidden in ("siw", "SiW", "target_test", "resolve_target"):
        assert forbidden not in source, forbidden


# ==============================================================================
# E. C-H4
# ==============================================================================

def test_c_h4_basic_conditions_fail_on_the_real_observed_values() -> None:
    result = probe.c_h4_preconditions(OBSERVED_BA_SEP)
    assert result["basic_conditions_hold"] is False
    assert result["status"] == "NOT_SUPPORTED_BY_CURRENT_BA_SEP_RESULT"
    assert result["hard_gate_llm_le_ceiling"] is False   # 0.7902... > 0.75


def test_c_h4_llm_does_not_beat_rnd_under_the_observed_values() -> None:
    result = probe.c_h4_preconditions(OBSERVED_BA_SEP)
    assert result["llm_beats_rnd"] is False   # LLM 0.7902... > RND 0.7843...


def test_c_h4_llm_does_beat_det_but_that_alone_is_insufficient() -> None:
    result = probe.c_h4_preconditions(OBSERVED_BA_SEP)
    assert result["llm_beats_det"] is True    # LLM 0.7902... < DET 0.8514...
    assert result["basic_conditions_hold"] is False   # still insufficient overall


def test_c_h4_preconditions_never_fabricates_a_bootstrap_result() -> None:
    result = probe.c_h4_preconditions(OBSERVED_BA_SEP)
    assert result["bootstrap_ci_evaluated"] is False
    assert result["validity_condition_evaluated"] is False
    assert result["recipe_diversity_condition_evaluated"] is False
    source = __import__("inspect").getsource(probe.c_h4_preconditions)
    for forbidden in ("bootstrap(", "np.random", "resample", "confidence_interval"):
        assert forbidden not in source, forbidden


def test_c_h4_preconditions_fails_closed_on_missing_arm() -> None:
    with pytest.raises(probe.SyntheticRealProbeError):
        probe.c_h4_preconditions({"RND": 0.5, "DET": 0.5})


# ==============================================================================
# F. safety / no real GPU access anywhere in this file
# ==============================================================================

def test_every_fixture_write_in_this_module_targets_tmp_path(monkeypatch, tmp_path) -> None:
    """A behavioral proof, not a source grep: writing a fixture result set
    and registering it must never touch anything under the REAL repo's
    reliability directory or lock path."""
    real_lock = REPO / barrier.LOCK_PATH
    real_reliability_dir = REPO / probe.RELIABILITY_DIR
    real_lock_before = real_lock.read_bytes() if real_lock.is_file() else None
    real_reliability_listing_before = (sorted(p.name for p in real_reliability_dir.iterdir())
                                       if real_reliability_dir.is_dir() else None)

    _patch_active_protocol(monkeypatch)
    write_ba_sep_result_set(tmp_path, ba_sep_by_arm=OBSERVED_BA_SEP,
                            per_seed_by_arm=OBSERVED_PER_SEED)
    reg_runner._register(tmp_path)

    real_lock_after = real_lock.read_bytes() if real_lock.is_file() else None
    real_reliability_listing_after = (sorted(p.name for p in real_reliability_dir.iterdir())
                                      if real_reliability_dir.is_dir() else None)
    assert real_lock_before == real_lock_after
    assert real_reliability_listing_before == real_reliability_listing_after
