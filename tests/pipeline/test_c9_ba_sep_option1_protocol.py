"""The shared mechanics in `prism_fas.evaluation.synthetic_real_probe` that
serve BOTH the historical `C9_DETECTOR_BA_SEP_OPTION1_V1` protocol and its
V2 pre-execution correction (see `test_c9_ba_sep_option1_v2_runner.py` for
V2-specific coverage: supersession, group-identity resolution, the
sample/group identity split, the CLI runner and the hard verdict rule).

This file proves the PROTOCOL and its MECHANICS are correct and safe — it
never computes a real BA_sep value, never loads a real detector checkpoint,
never opens a real image, and never touches target data. Every numeric test
here (split, balance, normalization, the linear probe, BA_sep aggregation)
runs against clearly-synthetic fixture arrays, not real scientific evidence.
The one function that would touch real data, `run_scientific_probe`, is
proven NOT to run (it raises `NotImplementedError`) rather than exercised.
`PopulationRecord` fixtures below carry BOTH `sample_identity` and
`stable_group_identity` since the V2 correction split what was one field
into two (`reports/readiness/C9_BA_SEP_OPTION1_V2_PREEXECUTION_CORRECTION.md`).
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.evaluation import detector_reliability as barrier  # noqa: E402
from prism_fas.evaluation import synthetic_real_probe as probe  # noqa: E402


# --- 1, 2. the frozen protocol resolves ---------------------------------------

def test_protocol_contains_every_required_field() -> None:
    protocol = probe.load_protocol(REPO)
    missing = [field for field in barrier.PROBE_PROTOCOL_REQUIRED_FIELDS
               if field not in protocol]
    assert not missing, missing


def test_probe_protocol_status_resolved_with_repo() -> None:
    status = barrier.probe_protocol_status(REPO)
    assert status["resolved"] is True
    assert status["may_execute"] is True
    assert status["reason_code"] is None


def test_probe_protocol_status_still_unresolved_without_repo() -> None:
    """The module-constant, no-repo-context answer is unchanged: no scientific
    value is ever frozen by editing a Python literal in this module."""
    status = barrier.probe_protocol_status()
    assert status["resolved"] is False
    assert barrier.DETECTOR_BA_SEP_PROBE_PROTOCOL is None


def test_protocol_only_marks_the_protocol_resolved_not_any_test_passed() -> None:
    """Protocol resolved != test passed (task rule): resolving the protocol
    must never make verify_lock (or any test's PASSED state) true."""
    status = barrier.probe_protocol_status(REPO)
    assert status["resolved"] is True
    per_test = barrier.barrier_state({})["per_test"]
    assert per_test["synthetic_vs_real_spoof_probe"] == barrier.UNRESOLVED


# --- 3, 4. protocol identity ---------------------------------------------------

def test_protocol_identity_is_deterministic() -> None:
    first = probe.protocol_identity(REPO)
    second = probe.protocol_identity(REPO)
    assert first == second
    assert len(first) == 64


def test_changing_a_result_affecting_field_changes_the_identity() -> None:
    protocol = probe.load_protocol(REPO)
    baseline = barrier.protocol_identity(protocol)

    changed = dict(protocol)
    changed["probe_seed_values"] = [1, 2, 3]
    assert barrier.protocol_identity(changed) != baseline

    changed2 = dict(protocol)
    changed2["ba_ceiling"] = 0.80
    assert barrier.protocol_identity(changed2) != baseline


def test_changing_only_metadata_does_not_change_the_identity() -> None:
    """frozen_on/approved_by/status are provenance, not result-affecting."""
    protocol = probe.load_protocol(REPO)
    baseline = barrier.protocol_identity(protocol)
    changed = dict(protocol)
    changed["frozen_on"] = "2099-01-01"
    changed["approved_by"] = "someone else"
    assert barrier.protocol_identity(changed) == baseline


# --- 5, 6. the evidence vector --------------------------------------------------

def test_evidence_vector_is_exactly_global_logit_g_and_p_global() -> None:
    assert probe.EVIDENCE_FIELDS == ("global_logit_G", "p_global")
    assert probe.EVIDENCE_DIMENSION == 2
    protocol = probe.load_protocol(REPO)
    assert protocol["evidence_vector_definition"]["fields"] == ["global_logit_G", "p_global"]


def test_no_forbidden_evidence_field_is_ever_read() -> None:
    source = inspect.getsource(probe.extract_evidence)
    source += inspect.getsource(probe.forward_checkpoint_evidence)
    for forbidden in probe.FORBIDDEN_EVIDENCE_FIELDS:
        assert forbidden not in source, forbidden


def test_extract_evidence_reads_only_the_two_frozen_attributes() -> None:
    class FakeOutput:
        global_logit = 1.5
        p_global = 0.6
        s_region = "must never be read"
        region_distances = "must never be read"

    vector = probe.extract_evidence(FakeOutput())
    assert vector.shape == (2,)
    assert list(vector) == [1.5, 0.6]


def test_forward_checkpoint_evidence_never_touches_forbidden_fields() -> None:
    """A fake model exposing forbidden attributes must still yield only the
    frozen 2-D vector — no test-only fixture needed beyond a plain object."""
    class FakeModelOutput:
        def __init__(self) -> None:
            self.global_logit = 2.0
            self.p_global = 0.8
            self.s_region = object()          # would raise if ever touched wrongly
            self.region_distances = object()

    class FakeModel:
        def __call__(self, batch: Any) -> FakeModelOutput:
            return FakeModelOutput()

    vector = probe.forward_checkpoint_evidence(FakeModel(), batch=None)
    assert list(vector) == [2.0, 0.8]


# --- 7, 8, 9. checkpoint binding -------------------------------------------------

def test_all_five_p3_track_g_rows_required_per_arm() -> None:
    for arm in probe.ARMS:
        rows = probe.track_g_p3_rows(arm)
        assert len(rows) == probe.CHECKPOINTS_PER_ARM == 5
        assert all(row.track == "G" and row.protocol == "P3" and row.arm == arm
                  for row in rows)
        assert sorted(row.seed for row in rows) == [20260806, 20260807, 20260808,
                                                     20260809, 20260810]


def test_total_checkpoints_is_fifteen() -> None:
    assert probe.TOTAL_CHECKPOINTS == 15 == probe.CHECKPOINTS_PER_ARM * len(probe.ARMS)


def test_resolve_checkpoint_set_refuses_on_this_dev_clone_rather_than_partial(
        tmp_path: Path) -> None:
    """This laptop has no runs/full/c8/; the resolver must refuse (raise),
    never silently return a partial or best-effort set."""
    with pytest.raises(probe.SyntheticRealProbeError):
        probe.resolve_checkpoint_set(tmp_path, "DET")


def test_no_best_checkpoint_selection_exists_in_source() -> None:
    source = inspect.getsource(probe.resolve_checkpoint_set)
    for forbidden in ("best", "lowest", "min(", "max(", "sorted(evidence",
                      "sort(key=lambda item: item.metrics"):
        assert forbidden not in source, forbidden


def test_checkpoint_aggregation_is_arithmetic_mean_over_five() -> None:
    vectors = [np.array([float(i), 0.1 * i]) for i in range(1, 6)]
    averaged = probe.average_checkpoint_evidence(vectors)
    assert np.allclose(averaged, [3.0, 0.3])
    assert "mean(axis=0)" in inspect.getsource(probe.average_checkpoint_evidence)


def test_checkpoint_aggregation_rejects_wrong_arity() -> None:
    with pytest.raises(probe.SyntheticRealProbeError):
        probe.average_checkpoint_evidence([])
    with pytest.raises(probe.SyntheticRealProbeError):
        probe.average_checkpoint_evidence([np.array([1.0, 2.0, 3.0])])


# --- 10. probe seeds -------------------------------------------------------------

def test_probe_seeds_are_exactly_the_three_frozen_values() -> None:
    protocol = probe.load_protocol(REPO)
    assert protocol["probe_seed_values"] == [20260806, 20260807, 20260808]
    assert "NOT a claim that" in protocol["probe_seed_provenance"] or \
        "newly preregistered" in protocol["probe_seed_provenance"]


# --- 11, 12. split ----------------------------------------------------------------

def test_split_is_deterministic() -> None:
    first = probe.split_bucket("ns", 20260806, "casia_fasd", "sample-1")
    second = probe.split_bucket("ns", 20260806, "casia_fasd", "sample-1")
    assert first == second


def test_split_is_group_safe() -> None:
    """The SAME stable identity always lands in the same bucket for a given
    (namespace, seed, domain) — a group can never straddle train/validation."""
    records = [probe.PopulationRecord(sample_identity=f"sample-{i}",
                                      stable_group_identity=f"group-{i}",
                                      source_domain="casia_fasd", label=0)
              for i in range(200)]
    first = probe.assign_splits(records, namespace="ns", probe_seed=20260806)
    second = probe.assign_splits(records, namespace="ns", probe_seed=20260806)
    assert {r.stable_group_identity for r in first["train"]} == \
        {r.stable_group_identity for r in second["train"]}
    all_ids = {r.stable_group_identity for r in records}
    assigned_ids = ({r.stable_group_identity for r in first["train"]} |
                    {r.stable_group_identity for r in first["validation"]})
    assert all_ids == assigned_ids
    overlap = ({r.stable_group_identity for r in first["train"]} &
              {r.stable_group_identity for r in first["validation"]})
    assert not overlap


def test_split_is_common_across_arms() -> None:
    """The same source identity gets the same split bucket regardless of
    which arm's synthetic population it is paired with — the split does not
    take the arm as an input at all."""
    import inspect as _inspect

    params = list(_inspect.signature(probe.split_bucket).parameters)
    assert "arm" not in params


def test_split_fraction_is_approximately_80_20() -> None:
    records = [probe.PopulationRecord(sample_identity=f"s-{i}", stable_group_identity=f"s-{i}",
                                      source_domain="msu_mfsd", label=0) for i in range(2000)]
    buckets = probe.assign_splits(records, namespace="c9-ba-sep-option1-v1", probe_seed=20260807)
    train_fraction = len(buckets["train"]) / len(records)
    assert 0.75 < train_fraction < 0.85


# --- 13, 14. class balance ---------------------------------------------------------

def test_class_balance_is_1_to_1_within_a_cell() -> None:
    real = [probe.PopulationRecord(sample_identity=f"r{i}", stable_group_identity=f"r{i}",
                                   source_domain="casia_fasd", label=0) for i in range(100)]
    synthetic = {arm: [probe.PopulationRecord(sample_identity=f"{arm}-{i}",
                                              stable_group_identity=f"{arm}-{i}",
                                              source_domain="casia_fasd", label=1)
                       for i in range(40 + index * 10)]
                for index, arm in enumerate(probe.ARMS)}
    selected_real, selected_synthetic = probe.balance_classes(
        protocol_id="p", probe_seed=20260806, split="train", source_domain="casia_fasd",
        real_spoof=real, synthetic_by_arm=synthetic)
    n = min(len(real), *(len(v) for v in synthetic.values()))
    assert len(selected_real) == n
    assert all(len(v) == n for v in selected_synthetic.values())


def test_the_same_real_subset_is_shared_across_arms() -> None:
    real = [probe.PopulationRecord(sample_identity=f"r{i}", stable_group_identity=f"r{i}",
                                   source_domain="casia_fasd", label=0) for i in range(50)]
    synthetic = {arm: [probe.PopulationRecord(sample_identity=f"{arm}-{i}",
                                              stable_group_identity=f"{arm}-{i}",
                                              source_domain="casia_fasd", label=1)
                       for i in range(50)] for arm in probe.ARMS}
    selected_real, _ = probe.balance_classes(
        protocol_id="p", probe_seed=20260806, split="train", source_domain="casia_fasd",
        real_spoof=real, synthetic_by_arm=synthetic)
    # Re-running with a DIFFERENT arm's synthetic pool composition unchanged
    # must select the identical real subset (deterministic, not per-arm).
    selected_real_again, _ = probe.balance_classes(
        protocol_id="p", probe_seed=20260806, split="train", source_domain="casia_fasd",
        real_spoof=real, synthetic_by_arm=synthetic)
    assert [r.stable_group_identity for r in selected_real] == \
        [r.stable_group_identity for r in selected_real_again]


def test_balance_has_no_replacement_or_oversampling() -> None:
    source = inspect.getsource(probe.balance_classes)
    for forbidden in ("replace=True", "np.random.choice", "resample", "oversample",
                      "class_weight"):
        assert forbidden not in source, forbidden


# --- 15, 16. normalization ----------------------------------------------------------

def test_normalization_is_fit_on_train_only() -> None:
    train = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    validation = np.array([[100.0, 1000.0]])   # wildly different scale
    normalization = probe.fit_normalization(train)
    # The fitted mean/std reflect ONLY train, not the validation outlier.
    assert np.allclose(normalization.mean, [2.0, 20.0])


def test_validation_statistics_cannot_enter_normalization() -> None:
    source = inspect.getsource(probe.compute_ba_sep_for_seed)
    # fit_normalization is called exactly once, over train_features only.
    assert source.count("fit_normalization(") == 1
    assert "fit_normalization(train_features" in source
    assert "fit_normalization(validation" not in source


# --- 17, 18. the linear probe --------------------------------------------------------

def test_probe_is_zero_initialized_in_source() -> None:
    source = inspect.getsource(probe.fit_linear_probe)
    assert "probe.weight.zero_()" in source
    assert "probe.bias.zero_()" in source


def test_solver_hyperparameters_match_the_frozen_config() -> None:
    assert probe.LBFGS_CONFIG == {
        "lr": 1.0, "max_iter": 200, "max_eval": 250,
        "tolerance_grad": 1e-7, "tolerance_change": 1e-9,
        "history_size": 100, "line_search_fn": "strong_wolfe",
    }
    protocol = probe.load_protocol(REPO)
    solver = protocol["optimizer_or_solver"]
    assert solver["lr"] == probe.LBFGS_CONFIG["lr"]
    assert solver["max_iter"] == probe.LBFGS_CONFIG["max_iter"]
    assert solver["max_eval"] == probe.LBFGS_CONFIG["max_eval"]
    assert solver["tolerance_grad"] == probe.LBFGS_CONFIG["tolerance_grad"]
    assert solver["tolerance_change"] == probe.LBFGS_CONFIG["tolerance_change"]
    assert solver["history_size"] == probe.LBFGS_CONFIG["history_size"]
    assert solver["line_search_fn"] == probe.LBFGS_CONFIG["line_search_fn"]


def test_no_minibatching_or_hyperparameter_search_in_source() -> None:
    source = inspect.getsource(probe.fit_linear_probe)
    for forbidden in ("DataLoader", "batch_size", "GridSearch", "RandomizedSearch",
                      "optuna"):
        assert forbidden not in source, forbidden


def test_probe_fit_is_deterministic_given_the_same_data() -> None:
    rng = np.random.RandomState(1)
    features = rng.normal(size=(30, 2))
    labels = (rng.uniform(size=30) > 0.5).astype(float)
    probe_a = probe.fit_linear_probe(features, labels)
    probe_b = probe.fit_linear_probe(features, labels)
    import torch

    with torch.no_grad():
        assert torch.allclose(probe_a.weight, probe_b.weight)
        assert torch.allclose(probe_a.bias, probe_b.bias)


# --- 19. threshold ------------------------------------------------------------------

def test_classifier_threshold_is_exactly_0_5() -> None:
    assert probe.CLASSIFIER_THRESHOLD == 0.5
    protocol = probe.load_protocol(REPO)
    assert protocol["classifier_threshold"] == 0.5


# --- 20, 21. BA aggregation and ceiling ----------------------------------------------

def test_ba_aggregation_is_arithmetic_mean_over_exactly_three_probe_seeds() -> None:
    result = probe.aggregate_ba_sep({20260806: 0.6, 20260807: 0.7, 20260808: 0.8})
    assert abs(result - 0.7) < 1e-12
    with pytest.raises(probe.SyntheticRealProbeError):
        probe.aggregate_ba_sep({20260806: 0.6, 20260807: 0.7})


def test_ba_ceiling_remains_0_75() -> None:
    assert barrier.BA_SEP_CEILING == 0.75
    protocol = probe.load_protocol(REPO)
    assert protocol["ba_ceiling"] == 0.75


# --- 22. target firewall -------------------------------------------------------------

def test_no_target_path_can_resolve_from_this_module() -> None:
    source = Path(inspect.getfile(probe)).read_text(encoding="utf-8")
    for forbidden in ("siw", "SiW", "target_test", "target_taxonomy",
                      "resolve_target", "_real_target_roots"):
        assert forbidden not in source, forbidden
    protocol = probe.load_protocol(REPO)
    assert protocol["target_firewall"]["target_access"] == 0


def test_preflight_reports_zero_target_access() -> None:
    result = probe.preflight(REPO)
    assert result["target_access"] == 0


# --- 23. preflight never executes the probe -------------------------------------------

def test_preflight_never_fits_a_probe_or_computes_ba() -> None:
    result = probe.preflight(REPO)
    assert result["probe_fit_executed"] is False
    assert result["ba_metric_computed"] is False
    assert result["scientific_artifacts_written"] is False
    assert result["state_modified"] is False
    assert result["detector_reliability_lock_created"] is False


def test_preflight_source_calls_no_execution_function() -> None:
    source = inspect.getsource(probe.preflight)
    for forbidden in ("fit_linear_probe(", "compute_ba_sep_for_seed(",
                      "run_scientific_probe(", "forward_checkpoint_evidence(",
                      "resolve_checkpoint_set(", "resolve_arm_populations("):
        assert forbidden not in source, forbidden


# --- 24. importing the module executes no science --------------------------------------

def test_importing_the_module_touches_no_filesystem_state(tmp_path: Path) -> None:
    import importlib
    import subprocess

    probe_module = "prism_fas.evaluation.synthetic_real_probe"
    result = subprocess.run(
        [sys.executable, "-c", f"import {probe_module}"],
        capture_output=True, text=True, cwd=str(REPO))
    assert result.returncode == 0, result.stderr
    # Nothing under state/ or reports/ changed as a side effect of import.
    for relative in ("state/PIPELINE_STATE.json", "state/MASTER_RUN_INDEX.json"):
        path = REPO / relative
        assert not (path.exists() and path.stat().st_mtime > (path.stat().st_ctime)), (
            "import must not touch state files")
    importlib.import_module(probe_module)  # importable from this test process too


def test_run_scientific_probe_is_not_wired() -> None:
    with pytest.raises(NotImplementedError):
        probe.run_scientific_probe(REPO, "DET")


# --- 25. no C8 evidence is rewritten -----------------------------------------------

def test_module_never_writes_under_runs_or_reports_full_c8() -> None:
    full_source = Path(inspect.getfile(probe)).read_text(encoding="utf-8")
    for forbidden in ("write_artifact(", "atomic_write_json(", "write_text(",
                      "write_bytes(", ".write("):
        assert forbidden not in full_source, forbidden


# --- 26. verify_lock still blocks -----------------------------------------------------

def test_verify_lock_still_blocks_with_the_protocol_resolved(tmp_path: Path) -> None:
    """Resolving the protocol must not, by itself, make any part of
    verify_lock's strict requirement looser."""
    verification = barrier.verify_lock(tmp_path)
    assert verification["valid"] is False
    assert any("absent" in problem or "DETECTOR_RELIABILITY_LOCK_C" in problem
              for problem in verification["problems"])


def test_verify_lock_refuses_a_lock_missing_eight_other_tests(tmp_path: Path) -> None:
    """Even a lock that DOES exist and claims the probe passed must still be
    refused unless every one of the nine required tests is genuinely PASSED —
    this task resolves the protocol for one test only."""
    import json

    results = {"synthetic_vs_real_spoof_probe": barrier.PASSED}
    payload = barrier.lock_payload(
        results=results, probe_protocol_identity=probe.protocol_identity(REPO),
        detector_checkpoint_identities={"stub": "a" * 64})
    payload["overall"] = barrier.PASSED   # tampering to prove verify_lock catches it
    lock_path = tmp_path / "DETECTOR_RELIABILITY_LOCK_C.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(payload), encoding="utf-8")

    verification = barrier.verify_lock(tmp_path, lock_path)
    assert verification["valid"] is False
    unresolved_others = [name for name in barrier.REQUIRED_DETECTOR_RELIABILITY_TESTS
                         if name != "synthetic_vs_real_spoof_probe"]
    assert all(any(name in problem for problem in verification["problems"])
              for name in unresolved_others)


def test_crop_padding_interpolation_remains_unresolved_and_not_reclassified() -> None:
    """This task does not touch the known crop_padding_interpolation gap."""
    assert "crop_padding_interpolation" in barrier.REQUIRED_DETECTOR_RELIABILITY_TESTS
    assert "crop_padding_interpolation" not in barrier.CANONICALLY_BLOCKED_TESTS
