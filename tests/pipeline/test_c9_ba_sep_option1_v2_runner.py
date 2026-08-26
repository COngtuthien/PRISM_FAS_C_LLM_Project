"""C9_DETECTOR_BA_SEP_OPTION1_V2 — the pre-execution group-identity
correction and the scientific runner CLI.

Covers, all against synthetic fixtures and monkeypatched resolvers — never
real C8/C6/source-package data, never a real checkpoint, never an image,
never target data:

  * V1 stays byte-identical and still resolves on its own terms
  * V2 supersession metadata and protocol identity
  * sample_identity vs stable_group_identity: resolution, and the fail-closed
    rules when a source_record_id (real) or live_target_sample_id -> real
    mapping (synthetic) is missing, empty or ambiguous
  * the split (group identity) and the selection order (sample identity)
    are genuinely two different axes, not the same key reused
  * verify_group_safe_split
  * balance_report's group-count reporting
  * the all-arm hard verdict rule
  * the `synthetic_real_probe_runner` CLI's three modes and their fail-closed
    contracts, on a host (this laptop) that has none of the real GPU C8
    artifacts

No test here computes a real BA_sep value, loads a real detector checkpoint,
opens a real image, or reads any target/SiW artifact.
"""
from __future__ import annotations

import importlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.evaluation import detector_reliability as barrier  # noqa: E402
from prism_fas.evaluation import synthetic_real_probe as probe  # noqa: E402
from prism_fas.evaluation import synthetic_real_probe_runner as runner  # noqa: E402

V1_PATH = REPO / "configs/evaluation/c9_detector_ba_sep_option1_v1.yaml"
V2_PATH = REPO / "configs/evaluation/c9_detector_ba_sep_option1_v2.yaml"
V1_IDENTITY = "a6da0ce75ebd92589ea61cba24a85bf8d8144bdbb99f7ec54d31066a66594908"


# ==============================================================================
# A. V1 stays untouched
# ==============================================================================

def test_v1_config_file_still_exists() -> None:
    assert V1_PATH.is_file()


def test_v1_still_declares_its_own_decision_id_and_status() -> None:
    payload = yaml.safe_load(V1_PATH.read_text(encoding="utf-8"))
    assert payload["decision_id"] == "C9_DETECTOR_BA_SEP_OPTION1_V1"
    assert payload["status"] == "FROZEN_NOT_RUN"


def test_v1_protocol_identity_is_unchanged_by_the_v2_correction() -> None:
    payload = yaml.safe_load(V1_PATH.read_text(encoding="utf-8"))
    assert barrier.protocol_identity(payload) == V1_IDENTITY


def test_v1_config_is_no_longer_the_active_protocol_path() -> None:
    assert barrier.PROBE_PROTOCOL_CONFIG_PATH != "configs/evaluation/c9_detector_ba_sep_option1_v1.yaml"


# ==============================================================================
# B. V2 resolves as the active protocol, with correct supersession metadata
# ==============================================================================

def test_active_protocol_path_is_v2() -> None:
    assert barrier.PROBE_PROTOCOL_CONFIG_PATH == "configs/evaluation/c9_detector_ba_sep_option1_v2.yaml"


def test_v2_config_file_exists_and_is_frozen_not_run() -> None:
    assert V2_PATH.is_file()
    payload = yaml.safe_load(V2_PATH.read_text(encoding="utf-8"))
    assert payload["status"] == "FROZEN_NOT_RUN"
    assert payload["decision_id"] == "C9_DETECTOR_BA_SEP_OPTION1_V2"


def test_v2_declares_every_required_field() -> None:
    protocol = probe.load_protocol(REPO)
    missing = [f for f in barrier.PROBE_PROTOCOL_REQUIRED_FIELDS if f not in protocol]
    assert not missing, missing


def test_v2_declares_supersession_of_v1() -> None:
    protocol = probe.load_protocol(REPO)
    assert protocol["supersedes"] == "configs/evaluation/c9_detector_ba_sep_option1_v1.yaml"
    assert protocol["superseded_protocol_identity"] == V1_IDENTITY
    assert protocol["supersession_reason"] == "PRE_EXECUTION_GROUP_IDENTITY_CORRECTION"


def test_v2_declares_no_ba_sep_ever_observed() -> None:
    protocol = probe.load_protocol(REPO)
    assert protocol["no_ba_sep_observed_under_v1"] is True
    assert protocol["no_ba_sep_observed_before_v2_freeze"] is True


def test_v2_protocol_identity_differs_from_v1() -> None:
    assert probe.protocol_identity(REPO) != V1_IDENTITY
    assert len(probe.protocol_identity(REPO)) == 64


def test_v2_evidence_vector_checkpoints_seeds_solver_unchanged_from_v1() -> None:
    """Everything the freeze task says V2 must retain byte-identical."""
    v1 = yaml.safe_load(V1_PATH.read_text(encoding="utf-8"))
    v2 = probe.load_protocol(REPO)
    assert v2["evidence_vector_definition"]["fields"] == v1["evidence_vector_definition"]["fields"]
    assert v2["probe_seed_values"] == v1["probe_seed_values"]
    assert v2["optimizer_or_solver"] == v1["optimizer_or_solver"]
    assert v2["regularization"] == v1["regularization"]
    assert v2["classifier_threshold"] == v1["classifier_threshold"]
    assert v2["ba_ceiling"] == v1["ba_ceiling"]
    assert v2["detector_checkpoint_identity"]["checkpoints_per_arm"] == \
        v1["detector_checkpoint_identity"]["checkpoints_per_arm"]
    assert v2["detector_checkpoint_identity"]["total_checkpoints"] == \
        v1["detector_checkpoint_identity"]["total_checkpoints"]


def test_v2_split_namespace_differs_from_v1() -> None:
    """A result-affecting change: the split hash material now includes a
    different namespace, so V1 and V2 buckets are not required to agree."""
    v1 = yaml.safe_load(V1_PATH.read_text(encoding="utf-8"))
    v2 = probe.load_protocol(REPO)
    assert v2["matched_source_split"]["split_hash_namespace"] != \
        v1["matched_source_split"]["split_hash_namespace"]


def test_v2_target_access_is_zero() -> None:
    protocol = probe.load_protocol(REPO)
    assert protocol["target_access"] == 0
    assert protocol["target_firewall"]["target_access"] == 0


# ==============================================================================
# C. PopulationRecord: sample_identity vs stable_group_identity
# ==============================================================================

def test_population_record_has_both_identities() -> None:
    fields = {f.name for f in __import__("dataclasses").fields(probe.PopulationRecord)}
    assert fields == {"sample_identity", "stable_group_identity", "source_domain", "label"}


def _install_fake_source_train(monkeypatch, rows: list[dict]) -> None:
    import prism_fas.data.loader.config as loader_config_module
    import prism_fas.data.loader.loose_dataset as loose_dataset
    from prism_fas.pipeline.adapters import sources

    class _FakeIndex:
        def __init__(self, rows: list[dict]) -> None:
            self.rows = rows

    class _FakeDataset:
        def __init__(self, package_root, split, loader_config, mode) -> None:
            self.index = _FakeIndex(rows)

    monkeypatch.setattr(loose_dataset, "CanonicalPackageDataset", _FakeDataset)
    monkeypatch.setattr(loader_config_module, "load_loader_config", lambda path: object())
    monkeypatch.setattr(sources, "verify_detector_inputs",
                        lambda repo, arms=None: {
                            "package_root": "pkg", "candidates_root": "cand",
                            "package_identity": "pkgid", "recipe_bank_identity": "recid"})


def _install_fake_bank(monkeypatch, rows: list[dict]) -> None:
    import prism_fas.detector.c6_bank as c6_bank
    from prism_fas.evaluation import c6_evidence

    class _FakeBank:
        def __init__(self, rows: list[dict]) -> None:
            self.rows = rows

    monkeypatch.setattr(c6_bank, "open_arm_bank",
                        lambda repo, *, arm, evidence, candidates_root,
                               package_identity, recipe_bank_identity: _FakeBank(rows))

    class _FakeEvidence:
        def bank(self, arm: str) -> None:
            return None

    monkeypatch.setattr(c6_evidence, "verify_c6_evidence", lambda repo: _FakeEvidence())


_SOURCE_TRAIN_ROWS = [
    {"sample_id": "rs-1", "dataset": "casia_fasd", "label_live_spoof": "spoof",
     "source_record_id": "rec-A"},
    {"sample_id": "live-1", "dataset": "casia_fasd", "label_live_spoof": "live",
     "source_record_id": "rec-A"},
    {"sample_id": "rs-2", "dataset": "msu_mfsd", "label_live_spoof": "spoof",
     "source_record_id": "rec-B"},
    {"sample_id": "live-2", "dataset": "msu_mfsd", "label_live_spoof": "live",
     "source_record_id": "rec-B"},
]


def test_real_population_stable_group_identity_is_source_record_id(monkeypatch, tmp_path) -> None:
    _install_fake_source_train(monkeypatch, _SOURCE_TRAIN_ROWS)
    records = probe.resolve_real_spoof_population(tmp_path, domains=("casia_fasd", "msu_mfsd"))
    by_sample = {r.sample_identity: r for r in records}
    assert by_sample["rs-1"].stable_group_identity == "rec-A"
    assert by_sample["rs-2"].stable_group_identity == "rec-B"
    assert {r.label for r in records} == {probe.REAL_SPOOF_CLASS}


def test_real_population_only_includes_spoof_rows(monkeypatch, tmp_path) -> None:
    _install_fake_source_train(monkeypatch, _SOURCE_TRAIN_ROWS)
    records = probe.resolve_real_spoof_population(tmp_path)
    assert {r.sample_identity for r in records} == {"rs-1", "rs-2"}


def test_real_population_fails_closed_on_empty_source_record_id(monkeypatch, tmp_path) -> None:
    rows = [{"sample_id": "rs-1", "dataset": "casia_fasd", "label_live_spoof": "spoof",
            "source_record_id": ""}]
    _install_fake_source_train(monkeypatch, rows)
    with pytest.raises(probe.SyntheticRealProbeError):
        probe.resolve_real_spoof_population(tmp_path)


def test_real_population_fails_closed_on_ambiguous_source_record_id(monkeypatch, tmp_path) -> None:
    rows = [
        {"sample_id": "dup-1", "dataset": "casia_fasd", "label_live_spoof": "spoof",
         "source_record_id": "rec-X"},
        {"sample_id": "dup-1", "dataset": "casia_fasd", "label_live_spoof": "spoof",
         "source_record_id": "rec-Y"},
    ]
    _install_fake_source_train(monkeypatch, rows)
    with pytest.raises(probe.SyntheticRealProbeError):
        probe.resolve_real_spoof_population(tmp_path)


def test_synthetic_population_stable_group_identity_via_live_target_sample_id(
        monkeypatch, tmp_path) -> None:
    _install_fake_source_train(monkeypatch, _SOURCE_TRAIN_ROWS)
    bank_rows = [
        {"synthetic_id": "syn-1", "live_target_dataset": "casia_fasd",
         "live_target_sample_id": "live-1"},
        {"synthetic_id": "syn-2", "live_target_dataset": "msu_mfsd",
         "live_target_sample_id": "live-2"},
    ]
    _install_fake_bank(monkeypatch, bank_rows)
    records = probe.resolve_synthetic_population(tmp_path, "DET")
    by_sample = {r.sample_identity: r for r in records}
    assert by_sample["syn-1"].stable_group_identity == "rec-A"
    assert by_sample["syn-2"].stable_group_identity == "rec-B"
    assert {r.label for r in records} == {probe.SYNTHETIC_SPOOF_CLASS}


def test_synthetic_population_shares_group_identity_with_the_real_record_it_was_generated_from(
        monkeypatch, tmp_path) -> None:
    """The exact case the V1 defect missed: a synthetic candidate generated
    from source record A and a REAL spoof sample also from record A must
    resolve to the SAME stable_group_identity, so they land in the same
    split bucket together — proving cross-population group safety."""
    _install_fake_source_train(monkeypatch, _SOURCE_TRAIN_ROWS)
    bank_rows = [{"synthetic_id": "syn-1", "live_target_dataset": "casia_fasd",
                 "live_target_sample_id": "live-1"}]
    _install_fake_bank(monkeypatch, bank_rows)
    real = probe.resolve_real_spoof_population(tmp_path)
    synthetic = probe.resolve_synthetic_population(tmp_path, "DET")
    real_group = next(r.stable_group_identity for r in real if r.sample_identity == "rs-1")
    synth_group = next(r.stable_group_identity for r in synthetic if r.sample_identity == "syn-1")
    assert real_group == synth_group == "rec-A"
    bucket_real = probe.split_bucket("ns", 1, "casia_fasd", real_group)
    bucket_synth = probe.split_bucket("ns", 1, "casia_fasd", synth_group)
    assert bucket_real == bucket_synth


def test_synthetic_population_fails_closed_on_unmapped_live_target_sample_id(
        monkeypatch, tmp_path) -> None:
    _install_fake_source_train(monkeypatch, _SOURCE_TRAIN_ROWS)
    bank_rows = [{"synthetic_id": "syn-orphan", "live_target_dataset": "casia_fasd",
                 "live_target_sample_id": "does-not-exist"}]
    _install_fake_bank(monkeypatch, bank_rows)
    with pytest.raises(probe.SyntheticRealProbeError, match="syn-orphan"):
        probe.resolve_synthetic_population(tmp_path, "DET")


def test_synthetic_population_fails_closed_on_empty_live_target_sample_id(
        monkeypatch, tmp_path) -> None:
    _install_fake_source_train(monkeypatch, _SOURCE_TRAIN_ROWS)
    bank_rows = [{"synthetic_id": "syn-empty", "live_target_dataset": "casia_fasd",
                 "live_target_sample_id": ""}]
    _install_fake_bank(monkeypatch, bank_rows)
    with pytest.raises(probe.SyntheticRealProbeError):
        probe.resolve_synthetic_population(tmp_path, "DET")


def test_synthetic_population_never_uses_synthetic_id_as_group_identity(
        monkeypatch, tmp_path) -> None:
    """The V1 defect, made unreachable: two DIFFERENT synthetic_id values
    sharing one live_target_sample_id must resolve to the SAME group."""
    _install_fake_source_train(monkeypatch, _SOURCE_TRAIN_ROWS)
    bank_rows = [
        {"synthetic_id": "syn-a", "live_target_dataset": "casia_fasd",
         "live_target_sample_id": "live-1"},
        {"synthetic_id": "syn-b", "live_target_dataset": "casia_fasd",
         "live_target_sample_id": "live-1"},
    ]
    _install_fake_bank(monkeypatch, bank_rows)
    records = probe.resolve_synthetic_population(tmp_path, "DET")
    groups = {r.stable_group_identity for r in records}
    samples = {r.sample_identity for r in records}
    assert groups == {"rec-A"}
    assert samples == {"syn-a", "syn-b"}


# ==============================================================================
# D. split (group) vs selection (sample) are two different axes
# ==============================================================================

def test_split_bucket_signature_takes_group_identity_not_sample_identity() -> None:
    params = list(inspect.signature(probe.split_bucket).parameters)
    assert "stable_group_identity" in params
    assert "sample_identity" not in params


def test_selection_order_key_partitions_on_sample_identity() -> None:
    params = list(inspect.signature(probe._selection_order_key).parameters)
    assert "sample_identity" in params
    assert "stable_group_identity" not in params


def test_split_and_selection_disagree_when_sample_and_group_identity_differ() -> None:
    """Two records sharing one stable_group_identity but with DIFFERENT
    sample_identity values must land in the SAME split bucket (group axis)
    while still being independently, deterministically ORDERED for
    selection (sample axis) — proving the two rules are genuinely separate,
    not the same key reused under two names."""
    a = probe.PopulationRecord(sample_identity="sample-a", stable_group_identity="shared-group",
                               source_domain="casia_fasd", label=0)
    b = probe.PopulationRecord(sample_identity="sample-b", stable_group_identity="shared-group",
                               source_domain="casia_fasd", label=0)
    bucket_a = probe.split_bucket("ns", 1, a.source_domain, a.stable_group_identity)
    bucket_b = probe.split_bucket("ns", 1, b.source_domain, b.stable_group_identity)
    assert bucket_a == bucket_b   # same group -> same bucket

    key_a = probe._selection_order_key("proto", 1, "train", "casia_fasd", a.sample_identity)
    key_b = probe._selection_order_key("proto", 1, "train", "casia_fasd", b.sample_identity)
    assert key_a != key_b   # different sample identity -> different order key


def test_balance_classes_orders_by_sample_identity_not_group_identity() -> None:
    """Records sharing one stable_group_identity but different sample
    identities must be selectable independently by balance_classes (proving
    selection does not collapse same-group records together)."""
    real = [probe.PopulationRecord(sample_identity=f"r{i}", stable_group_identity="one-group",
                                   source_domain="casia_fasd", label=0) for i in range(5)]
    synthetic = {arm: [probe.PopulationRecord(sample_identity=f"{arm}-{i}",
                                              stable_group_identity="one-group",
                                              source_domain="casia_fasd", label=1)
                       for i in range(5)] for arm in probe.ARMS}
    selected_real, _ = probe.balance_classes(
        protocol_id="p", probe_seed=1, split="train", source_domain="casia_fasd",
        real_spoof=real, synthetic_by_arm=synthetic)
    assert len(selected_real) == 5
    assert len({r.sample_identity for r in selected_real}) == 5


def test_no_selection_function_reads_stable_group_identity_for_ordering() -> None:
    """The executable material line — not the prose docstring explaining the
    distinction — must never build its hash key from stable_group_identity."""
    source = inspect.getsource(probe._selection_order_key)
    material_line = next(line for line in source.splitlines() if "material = f" in line)
    assert "stable_group_identity" not in material_line
    assert "sample_identity" in material_line


# ==============================================================================
# E. verify_group_safe_split
# ==============================================================================

def test_verify_group_safe_split_accepts_a_genuinely_safe_split() -> None:
    split = {probe.TRAIN_LABEL: [probe.PopulationRecord("s1", "g1", "casia_fasd", 0)],
            probe.VALIDATION_LABEL: [probe.PopulationRecord("s2", "g2", "casia_fasd", 0)]}
    probe.verify_group_safe_split(split)   # must not raise


def test_verify_group_safe_split_rejects_a_leaking_split() -> None:
    split = {probe.TRAIN_LABEL: [probe.PopulationRecord("s1", "leaked-group", "casia_fasd", 0)],
            probe.VALIDATION_LABEL: [probe.PopulationRecord("s2", "leaked-group", "casia_fasd", 0)]}
    with pytest.raises(probe.SyntheticRealProbeError, match="leaked-group"):
        probe.verify_group_safe_split(split)


def test_verify_group_safe_split_handles_missing_buckets() -> None:
    probe.verify_group_safe_split({})   # neither key present: vacuously safe


def test_assign_splits_output_always_passes_verify_group_safe_split() -> None:
    records = [probe.PopulationRecord(f"s{i}", f"g{i}", "casia_fasd", 0) for i in range(500)]
    split = probe.assign_splits(records, namespace="ns", probe_seed=1)
    probe.verify_group_safe_split(split)   # must not raise: construction is safe by design


# ==============================================================================
# F. balance_report — group-count reporting
# ==============================================================================

def test_balance_report_includes_unique_source_record_id_counts() -> None:
    real = [probe.PopulationRecord(f"r{i}", f"g{i // 2}", "casia_fasd", 0) for i in range(10)]
    synthetic = {arm: [probe.PopulationRecord(f"{arm}{i}", f"g{i // 2}", "casia_fasd", 1)
                       for i in range(10)] for arm in probe.ARMS}
    report = probe.balance_report(protocol_id="p", probe_seed=1, split="train",
                                  source_domain="casia_fasd", real_spoof=real,
                                  synthetic_by_arm=synthetic)
    assert report["unique_source_record_id_counts"]["real_pre"] == 5
    assert "real_post" in report["unique_source_record_id_counts"]
    assert report["n"] == 10


def test_balance_report_pre_and_post_counts_match_balance_classes() -> None:
    real = [probe.PopulationRecord(f"r{i}", f"r{i}", "casia_fasd", 0) for i in range(30)]
    synthetic = {arm: [probe.PopulationRecord(f"{arm}{i}", f"{arm}{i}", "casia_fasd", 1)
                       for i in range(10 + index * 5)]
                for index, arm in enumerate(probe.ARMS)}
    selected_real, selected_synthetic = probe.balance_classes(
        protocol_id="p", probe_seed=1, split="train", source_domain="casia_fasd",
        real_spoof=real, synthetic_by_arm=synthetic)
    report = probe.balance_report(protocol_id="p", probe_seed=1, split="train",
                                  source_domain="casia_fasd", real_spoof=real,
                                  synthetic_by_arm=synthetic)
    assert report["post_balance_counts"]["real"] == len(selected_real)
    for arm in probe.ARMS:
        assert report["post_balance_counts"][arm] == len(selected_synthetic[arm])
    assert report["pre_balance_counts"]["real"] == len(real)


# ==============================================================================
# G. the all-arm hard verdict rule
# ==============================================================================

def test_hard_verdict_passes_when_all_arms_at_or_under_ceiling() -> None:
    result = probe.hard_verdict({"RND": 0.75, "DET": 0.70, "LLM": 0.50})
    assert result["verdict"] == "PASS"
    assert result["failing_arms"] == []


def test_hard_verdict_fails_when_any_single_arm_exceeds_the_ceiling() -> None:
    result = probe.hard_verdict({"RND": 0.60, "DET": 0.76, "LLM": 0.50})
    assert result["verdict"] == "FAIL"
    assert result["failing_arms"] == ["DET"]


def test_hard_verdict_fails_closed_on_missing_arm() -> None:
    with pytest.raises(probe.SyntheticRealProbeError):
        probe.hard_verdict({"RND": 0.5, "DET": 0.5})


def test_hard_verdict_uses_the_frozen_ceiling() -> None:
    result = probe.hard_verdict({"RND": 0.75, "DET": 0.75, "LLM": 0.75})
    assert result["ba_ceiling"] == 0.75
    assert result["verdict"] == "PASS"   # boundary is inclusive (<=)


def test_hard_verdict_boundary_just_over_ceiling_fails() -> None:
    result = probe.hard_verdict({"RND": 0.7501, "DET": 0.5, "LLM": 0.5})
    assert result["verdict"] == "FAIL"
    assert result["failing_arms"] == ["RND"]


def test_hard_verdict_is_frozen_in_the_v2_protocol_before_any_metric() -> None:
    protocol = probe.load_protocol(REPO)
    rule = protocol["hard_verdict_rule"]
    assert rule["pass_condition"] == \
        "BA_sep_RND <= 0.75 AND BA_sep_DET <= 0.75 AND BA_sep_LLM <= 0.75"


# ==============================================================================
# H0. shared fixtures for the joint (all-arm) runner
# ==============================================================================

def _fixture_checkpoint(arm: str, seed: int) -> "probe.CheckpointBinding":
    import hashlib as _hashlib

    return probe.CheckpointBinding(
        arm=arm, seed=seed, row_id=f"C-G-{arm}-P3READY-s{seed}",
        run_identity=f"run-{arm}-{seed}", config_identity="cfg" + "0" * 61,
        checkpoint_sha256=_hashlib.sha256(f"{arm}-{seed}".encode()).hexdigest(),
        checkpoint_path=f"runs/full/c8/P3/C-G-{arm}/cfg/{seed}/checkpoints/best.pt",
        checkpoint_kind="best", decision_logit_name="global_logit_G",
        decision_graph_hash="graph" + "0" * 59)


def _fixture_checkpoints_by_arm(*, seeds: Sequence[int] = (1, 2, 3, 4, 5)
                                ) -> dict[str, list["probe.CheckpointBinding"]]:
    return {arm: [_fixture_checkpoint(arm, seed) for seed in seeds] for arm in probe.ARMS}


def _fixture_population(prefix: str, group_prefix: str, *, count: int, label: int,
                        domain: str = "casia_fasd") -> list["probe.PopulationRecord"]:
    return [probe.PopulationRecord(sample_identity=f"{prefix}{i}",
                                   stable_group_identity=f"{group_prefix}{i}",
                                   source_domain=domain, label=label)
           for i in range(count)]


def _install_joint_bind_fixtures(monkeypatch, *, probe_seed_values=(1, 2, 3),
                                 real_count: int = 60, synthetic_count: int = 60
                                 ) -> dict[str, Any]:
    """Everything `build_checkpoint_binding` and `build_population_plan` need,
    monkeypatched at the exact seams those functions call through — never
    touches a real repo, a real image or a real checkpoint's bytes."""
    from prism_fas.evaluation import c6_evidence
    from prism_fas.pipeline.adapters import sources

    fake_protocol = {
        "detector_checkpoint_identity": {"required_decision_logit_name": "global_logit_G"},
        "matched_source_split": {"split_hash_namespace": "test-joint-ns"},
        "probe_seed_values": list(probe_seed_values),
        "source_domains": ["casia_fasd", "msu_mfsd"],
        "ba_ceiling": 0.75,
        "target_access": 0,
        "target_firewall": {"target_access": 0},
    }
    monkeypatch.setattr(probe, "load_protocol", lambda repo: fake_protocol)
    monkeypatch.setattr(probe, "protocol_identity", lambda repo: "a" * 64)

    checkpoints_by_arm = _fixture_checkpoints_by_arm()
    monkeypatch.setattr(probe, "resolve_checkpoint_set",
                        lambda repo, arm: checkpoints_by_arm[arm])

    real = (_fixture_population("r-c", "greal-c", count=real_count // 2, label=0,
                                domain="casia_fasd")
           + _fixture_population("r-m", "greal-m", count=real_count // 2, label=0,
                                 domain="msu_mfsd"))
    synthetic_by_arm = {
        arm: (_fixture_population(f"{arm.lower()}-c", f"g{arm.lower()}-c",
                                  count=synthetic_count // 2, label=1, domain="casia_fasd")
             + _fixture_population(f"{arm.lower()}-m", f"g{arm.lower()}-m",
                                   count=synthetic_count // 2, label=1, domain="msu_mfsd"))
        for arm in probe.ARMS}
    monkeypatch.setattr(probe, "resolve_real_spoof_population", lambda repo, **_: real)
    monkeypatch.setattr(probe, "resolve_synthetic_population",
                        lambda repo, arm, **_: synthetic_by_arm[arm])

    monkeypatch.setattr(sources, "verify_detector_inputs",
                        lambda repo, arms=None: {
                            "package_identity": "pkg" + "0" * 61,
                            "c6": {"banks": {arm: {"selected_set_sha256": f"bank-{arm}" + "0" * 55}
                                            for arm in probe.ARMS}}})

    class _FakeEvidence:
        def bank(self, arm: str) -> None:
            return None

    monkeypatch.setattr(c6_evidence, "verify_c6_evidence", lambda repo: _FakeEvidence())

    return {"protocol": fake_protocol, "checkpoints_by_arm": checkpoints_by_arm,
           "real": real, "synthetic_by_arm": synthetic_by_arm}


# ==============================================================================
# H1. build_checkpoint_binding / build_population_plan — joint mechanics
# ==============================================================================

def test_build_checkpoint_binding_includes_all_three_arms_atomically(monkeypatch, tmp_path) -> None:
    _install_joint_bind_fixtures(monkeypatch)
    binding = probe.build_checkpoint_binding(tmp_path)
    assert set(binding["checkpoints_per_arm"]) == set(probe.ARMS)
    assert all(binding["checkpoints_per_arm"][arm] == 5 for arm in probe.ARMS)
    assert binding["total_checkpoints"] == 15


def test_build_checkpoint_binding_fails_closed_when_one_arm_is_short(monkeypatch, tmp_path) -> None:
    fixtures = _install_joint_bind_fixtures(monkeypatch)
    short = dict(fixtures["checkpoints_by_arm"])
    short["LLM"] = short["LLM"][:4]                    # only 4/5

    def _resolve(repo, arm):
        if arm == "LLM":
            raise probe.SyntheticRealProbeError("LLM: 4/5 P3-ready Track-G checkpoints resolved")
        return fixtures["checkpoints_by_arm"][arm]

    monkeypatch.setattr(probe, "resolve_checkpoint_set", _resolve)
    with pytest.raises(probe.SyntheticRealProbeError):
        probe.build_checkpoint_binding(tmp_path)


def test_checkpoint_binding_identity_is_deterministic_and_covers_all_checkpoints(
        monkeypatch, tmp_path) -> None:
    _install_joint_bind_fixtures(monkeypatch)
    first = probe.build_checkpoint_binding(tmp_path)
    second = probe.build_checkpoint_binding(tmp_path)
    assert first["checkpoint_binding_identity_sha256"] == second["checkpoint_binding_identity_sha256"]
    assert len(first["checkpoint_binding_identity_sha256"]) == 64


def test_build_population_plan_gives_every_arm_a_non_empty_synthetic_selection(
        monkeypatch, tmp_path) -> None:
    _install_joint_bind_fixtures(monkeypatch)
    plan = probe.build_population_plan(tmp_path)
    assert plan["cells"], "expected at least one cell"
    for cell in plan["cells"]:
        assert cell["n"] > 0
        for arm in probe.ARMS:
            assert len(cell["synthetic_selected"][arm]) == cell["n"] > 0


def test_build_population_plan_n_is_min_across_real_and_all_three_arms(monkeypatch, tmp_path) -> None:
    """The exact joint-balancing bug this task fixes: N must reflect all
    three arms' pools simultaneously, never a single arm against an empty
    pair. Here DET is deliberately starved to 15 candidates per domain,
    well below the other arms' 30, so DET must set the ceiling for N."""
    fixtures = _install_joint_bind_fixtures(monkeypatch)
    starved_det = (_fixture_population("det-c", "gdet-c", count=15, label=1, domain="casia_fasd")
                  + _fixture_population("det-m", "gdet-m", count=15, label=1, domain="msu_mfsd"))

    def _resolve_synthetic(repo, arm, **_):
        if arm == "DET":
            return starved_det
        return fixtures["synthetic_by_arm"][arm]

    monkeypatch.setattr(probe, "resolve_synthetic_population", _resolve_synthetic)
    plan = probe.build_population_plan(tmp_path)
    casia_train_cells = [c for c in plan["cells"]
                         if c["source_domain"] == "casia_fasd" and c["split"] == "train"]
    for cell in casia_train_cells:
        assert cell["n"] <= 15
        assert len(cell["synthetic_selected"]["RND"]) == cell["n"]
        assert len(cell["synthetic_selected"]["LLM"]) == cell["n"]


def test_build_population_plan_fails_closed_on_a_zero_sample_cell(monkeypatch, tmp_path) -> None:
    _install_joint_bind_fixtures(monkeypatch)
    monkeypatch.setattr(probe, "resolve_synthetic_population", lambda repo, arm, **_: [])
    with pytest.raises(probe.SyntheticRealProbeError, match="N=0"):
        probe.build_population_plan(tmp_path)


def test_build_population_plan_reuses_the_identical_real_subset_across_arms(
        monkeypatch, tmp_path) -> None:
    _install_joint_bind_fixtures(monkeypatch)
    plan = probe.build_population_plan(tmp_path)
    for cell in plan["cells"]:
        real_ids = {entry["sample_identity"] for entry in cell["real_selected"]}
        assert len(real_ids) == cell["n"]
        # every arm's synthetic selection has exactly n members too (1:1 balance)
        for arm in probe.ARMS:
            assert len(cell["synthetic_selected"][arm]) == cell["n"]


def test_build_population_plan_leakage_audit_reports_no_leak_on_a_safe_plan(
        monkeypatch, tmp_path) -> None:
    _install_joint_bind_fixtures(monkeypatch)
    plan = probe.build_population_plan(tmp_path)
    assert plan["leakage_audit"]["leaked"] == {}
    assert set(plan["leakage_audit"]["checked_seeds"]) == {"1", "2", "3"} or \
        set(str(s) for s in plan["leakage_audit"]["checked_seeds"]) == {"1", "2", "3"}


def test_population_plan_identity_is_deterministic(monkeypatch, tmp_path) -> None:
    _install_joint_bind_fixtures(monkeypatch)
    first = probe.build_population_plan(tmp_path)
    second = probe.build_population_plan(tmp_path)
    assert first["population_plan_identity_sha256"] == second["population_plan_identity_sha256"]


# ==============================================================================
# H2. the CLI runner — --preflight-only (no --arm anywhere)
# ==============================================================================

def test_preflight_only_takes_no_arm_flag() -> None:
    with pytest.raises(SystemExit):
        runner.main(["--repo", str(REPO), "--preflight-only", "--arm", "RND"])


def test_preflight_only_blocked_on_this_laptop_with_no_gpu_artifacts() -> None:
    """This laptop has neither the M3B source package nor runs/full/c8/;
    the strengthened production preflight must report BLOCKED, honestly."""
    exit_code = runner.main(["--repo", str(REPO), "--preflight-only"])
    assert exit_code == runner.EXIT_BLOCKED


def test_preflight_only_reports_zero_target_access(capsys) -> None:
    runner.main(["--repo", str(REPO), "--preflight-only"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["target_access"] == 0


def test_preflight_only_never_writes_state(capsys) -> None:
    runner.main(["--repo", str(REPO), "--preflight-only"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["probe_fit_executed"] is False
    assert payload["ba_metric_computed"] is False
    assert payload["state_modified"] is False
    assert payload["scientific_artifacts_written"] is False


def test_preflight_only_passes_when_every_input_resolves(monkeypatch, tmp_path) -> None:
    _install_joint_bind_fixtures(monkeypatch)
    exit_code, payload = runner._preflight(tmp_path)
    assert exit_code == runner.EXIT_PASS
    assert payload["ready_for_bind"] is True
    assert payload["checkpoints_per_arm"] == {arm: 5 for arm in probe.ARMS}


def test_preflight_only_blocked_when_one_arm_is_short_a_checkpoint(monkeypatch, tmp_path) -> None:
    fixtures = _install_joint_bind_fixtures(monkeypatch)

    def _resolve(repo, arm):
        if arm == "LLM":
            raise probe.SyntheticRealProbeError("LLM: 4/5 resolved")
        return fixtures["checkpoints_by_arm"][arm]

    monkeypatch.setattr(probe, "resolve_checkpoint_set", _resolve)
    exit_code, payload = runner._preflight(tmp_path)
    assert exit_code == runner.EXIT_BLOCKED
    assert payload["ready_for_bind"] is False
    assert payload["checkpoints_resolved"] is False


def test_preflight_only_writes_nothing_even_when_ready(monkeypatch, tmp_path) -> None:
    _install_joint_bind_fixtures(monkeypatch)
    runner._preflight(tmp_path)
    assert not (tmp_path / "reports").exists()


def test_preflight_source_opens_no_image_and_computes_no_ba() -> None:
    source = inspect.getsource(runner._preflight)
    for forbidden in ("Image.open", "cv2.imread", "compute_ba_sep_for_seed(",
                      "fit_linear_probe(", "load_state_dict", "torch.load"):
        assert forbidden not in source, forbidden


# ==============================================================================
# H3. the CLI runner — --bind-only (joint, atomic, two artifacts)
# ==============================================================================

def test_bind_only_takes_no_arm_flag() -> None:
    with pytest.raises(SystemExit):
        runner.main(["--repo", str(REPO), "--bind-only", "--arm", "DET"])


def test_bind_only_fails_closed_on_this_laptop_with_no_gpu_artifacts() -> None:
    exit_code = runner.main(["--repo", str(REPO), "--bind-only"])
    assert exit_code == runner.EXIT_BLOCKED
    assert not (REPO / runner.EXECUTION_BINDING_PATH).exists()
    assert not (REPO / runner.POPULATION_PLAN_PATH).exists()


def test_bind_only_writes_exactly_two_global_artifacts_on_success(monkeypatch, tmp_path) -> None:
    _install_joint_bind_fixtures(monkeypatch)
    exit_code, payload = runner._bind_only(tmp_path)
    assert exit_code == runner.EXIT_PASS
    assert payload["bound"] is True
    binding_path = tmp_path / runner.EXECUTION_BINDING_PATH
    plan_path = tmp_path / runner.POPULATION_PLAN_PATH
    assert binding_path.is_file()
    assert plan_path.is_file()
    written = {p.name for p in (tmp_path / runner.RELIABILITY_DIR).glob("*.json")}
    assert written == {runner.EXECUTION_BINDING_PATH.rsplit("/", 1)[-1],
                       runner.POPULATION_PLAN_PATH.rsplit("/", 1)[-1]}


def test_bind_only_execution_binding_has_all_15_checkpoints_with_required_fields(
        monkeypatch, tmp_path) -> None:
    _install_joint_bind_fixtures(monkeypatch)
    runner._bind_only(tmp_path)
    binding = json.loads((tmp_path / runner.EXECUTION_BINDING_PATH).read_text())
    assert len(binding["checkpoints"]) == 15
    for entry in binding["checkpoints"]:
        for field in ("arm", "seed", "row_id", "run_identity", "config_identity",
                      "checkpoint_relative_path", "checkpoint_sha256",
                      "decision_graph_hash", "decision_logit_name"):
            assert field in entry
        assert entry["decision_logit_name"] == "global_logit_G"
    assert binding["checkpoints_per_arm"] == {arm: 5 for arm in probe.ARMS}
    assert "checkpoint_binding_identity_sha256" in binding
    assert "source_package_identity" in binding
    assert set(binding["c6_bank_identities"]) == set(probe.ARMS)


def test_bind_only_population_plan_has_no_performance_metric(monkeypatch, tmp_path) -> None:
    _install_joint_bind_fixtures(monkeypatch)
    runner._bind_only(tmp_path)
    plan_text = (tmp_path / runner.POPULATION_PLAN_PATH).read_text()
    for forbidden in ("balanced_accuracy", "BA_sep", "ba_sep"):
        assert forbidden not in plan_text
    binding_text = (tmp_path / runner.EXECUTION_BINDING_PATH).read_text()
    for forbidden in ("balanced_accuracy", "BA_sep", "ba_sep"):
        assert forbidden not in binding_text


def test_bind_only_never_loads_checkpoint_weights_or_opens_an_image(monkeypatch, tmp_path) -> None:
    _install_joint_bind_fixtures(monkeypatch)
    exit_code, payload = runner._bind_only(tmp_path)
    assert exit_code == runner.EXIT_PASS
    assert payload["checkpoint_weights_loaded"] is False
    assert payload["images_forwarded"] is False
    assert payload["ba_metric_computed"] is False


def test_bind_only_fails_closed_on_a_zero_sample_cell_and_writes_nothing(
        monkeypatch, tmp_path) -> None:
    _install_joint_bind_fixtures(monkeypatch)
    monkeypatch.setattr(probe, "resolve_synthetic_population", lambda repo, arm, **_: [])
    exit_code, payload = runner._bind_only(tmp_path)
    assert exit_code == runner.EXIT_BLOCKED
    assert not (tmp_path / runner.RELIABILITY_DIR).exists()


def test_bind_only_is_deterministic_across_repeated_calls(monkeypatch, tmp_path) -> None:
    _install_joint_bind_fixtures(monkeypatch)
    runner._bind_only(tmp_path)
    first = (tmp_path / runner.EXECUTION_BINDING_PATH).read_text()
    exit_code, payload = runner._bind_only(tmp_path)
    second = (tmp_path / runner.EXECUTION_BINDING_PATH).read_text()
    assert first == second
    assert exit_code == runner.EXIT_PASS
    assert payload["reused"] is True
    assert payload["artifacts_written"] is False


def test_bind_only_refuses_to_overwrite_a_mismatched_existing_binding(
        monkeypatch, tmp_path) -> None:
    _install_joint_bind_fixtures(monkeypatch)
    runner._bind_only(tmp_path)
    original = (tmp_path / runner.EXECUTION_BINDING_PATH).read_text()

    # A different protocol identity changes the checkpoint binding identity
    # too (it is bound into the material) — simulating a genuinely different
    # preregistration trying to bind over the first one.
    monkeypatch.setattr(probe, "protocol_identity", lambda repo: "b" * 64)
    exit_code, payload = runner._bind_only(tmp_path)
    assert exit_code == runner.EXIT_BLOCKED
    assert "DIFFERENT identity" in payload["error"]
    assert (tmp_path / runner.EXECUTION_BINDING_PATH).read_text() == original


# ==============================================================================
# H4. the CLI runner — --execute (joint, real code, not NotImplementedError)
# ==============================================================================

def test_execute_takes_no_arm_flag() -> None:
    with pytest.raises(SystemExit):
        runner.main(["--repo", str(REPO), "--execute", "--arm", "RND"])


def test_execute_requires_bind_only_to_have_run_first(monkeypatch, tmp_path) -> None:
    _install_joint_bind_fixtures(monkeypatch)   # protocol resolves; --bind-only never ran
    exit_code, payload = runner._execute(tmp_path)
    assert exit_code == runner.EXIT_BLOCKED
    assert payload["executed"] is False
    assert "bind-only" in payload["error"]


def test_execute_requires_both_artifacts_not_just_one(monkeypatch, tmp_path) -> None:
    _install_joint_bind_fixtures(monkeypatch)
    runner._bind_only(tmp_path)
    (tmp_path / runner.POPULATION_PLAN_PATH).unlink()
    exit_code, payload = runner._execute(tmp_path)
    assert exit_code == runner.EXIT_BLOCKED
    assert payload["executed"] is False


def test_execute_refuses_on_this_laptop_with_no_gpu_artifacts() -> None:
    exit_code = runner.main(["--repo", str(REPO), "--execute"])
    assert exit_code == runner.EXIT_BLOCKED


def test_execute_reports_zero_target_access_when_blocked(capsys) -> None:
    runner.main(["--repo", str(REPO), "--execute"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["executed"] is False
    assert payload["target_access"] == 0


def _install_execute_fixtures(monkeypatch, tmp_path, *, real_value, synthetic_value_by_arm):
    """Bind fixtures, run a real --bind-only, then monkeypatch the ONE
    genuinely-unverifiable-on-this-laptop boundary
    (`construct_row_trainer`/`forward_evidence_for_records`) so
    `execute_joint_probe` can be exercised end to end with known evidence.

    `real_value` / each `synthetic_value_by_arm[arm]` is either a fixed
    `[x, y]` pair or a `callable(sample_identity) -> [x, y]` — the caller
    never needs to know the exact `sample_identity` strings the joint
    population plan assigned (those are an implementation detail of
    `_install_joint_bind_fixtures`, not of what a test is proving).
    """
    _install_joint_bind_fixtures(monkeypatch)
    exit_code, _ = runner._bind_only(tmp_path)
    assert exit_code == runner.EXIT_PASS

    monkeypatch.setattr(probe, "construct_row_trainer", lambda repo, binding: binding)

    def _resolve(value_or_fn, sample_identity: str):
        return value_or_fn(sample_identity) if callable(value_or_fn) else value_or_fn

    def _forward(trainer_binding: "probe.CheckpointBinding", records):
        out = {}
        for record in records:
            if record.label == probe.REAL_SPOOF_CLASS:
                vector = _resolve(real_value, record.sample_identity)
            else:
                vector = _resolve(synthetic_value_by_arm[trainer_binding.arm], record.sample_identity)
            out[record.sample_identity] = np.asarray(vector, dtype=np.float64)
        return out

    monkeypatch.setattr(probe, "forward_evidence_for_records", _forward)


def test_execute_is_no_longer_notimplementederror(monkeypatch, tmp_path) -> None:
    """The exact defect this task fixes: --execute used to route to
    run_scientific_probe, which always raised NotImplementedError. It must
    not do that any more."""
    _install_execute_fixtures(
        monkeypatch, tmp_path, real_value=[0.0, 0.1],
        synthetic_value_by_arm={arm: [5.0, 0.9] for arm in probe.ARMS})
    exit_code, payload = runner._execute(tmp_path)
    assert payload["executed"] is True
    assert exit_code in (runner.EXIT_PASS, runner.EXIT_FAIL)   # a real verdict, not a crash


def test_execute_writes_all_five_result_artifacts_on_success(monkeypatch, tmp_path) -> None:
    _install_execute_fixtures(
        monkeypatch, tmp_path, real_value=[0.0, 0.1],
        synthetic_value_by_arm={arm: [0.0, 0.1] for arm in probe.ARMS})
    runner._execute(tmp_path)
    for relative in (runner.RESULT_PATH, runner.PER_SEED_PATH, runner.PARAMETERS_PATH,
                    runner.EVIDENCE_MANIFEST_PATH, runner.VERDICT_PATH):
        assert (tmp_path / relative).is_file(), relative


def test_execute_pass_verdict_when_evidence_is_indistinguishable(monkeypatch, tmp_path) -> None:
    """Real and synthetic evidence drawn from the SAME distribution: the
    probe cannot separate them, BA_sep should sit near chance, well under
    the 0.75 ceiling for every arm."""
    rng = np.random.RandomState(20260806)

    def _random_vector(_sample_identity: str) -> list[float]:
        return rng.normal(size=2).tolist()

    _install_execute_fixtures(
        monkeypatch, tmp_path, real_value=_random_vector,
        synthetic_value_by_arm={arm: _random_vector for arm in probe.ARMS})
    exit_code, payload = runner._execute(tmp_path)
    assert payload["executed"] is True
    assert payload["verdict"] == "PASS"
    assert exit_code == runner.EXIT_PASS
    assert all(value <= 0.75 for value in payload["ba_sep_by_arm"].values())


def test_execute_fail_verdict_when_evidence_is_cleanly_separable(monkeypatch, tmp_path) -> None:
    """Real and synthetic evidence drawn from well-separated clusters: the
    probe should separate them near-perfectly, BA_sep well over 0.75 for
    every arm — a real, honestly-reported scientific FAILED verdict."""
    _install_execute_fixtures(
        monkeypatch, tmp_path, real_value=[-5.0, -5.0],
        synthetic_value_by_arm={arm: [5.0, 5.0] for arm in probe.ARMS})
    exit_code, payload = runner._execute(tmp_path)
    assert payload["executed"] is True
    assert payload["verdict"] == "FAIL"
    assert exit_code == runner.EXIT_FAIL
    assert all(value > 0.75 for value in payload["ba_sep_by_arm"].values())
    # a scientific FAILED result is still WRITTEN, never hidden
    assert (tmp_path / runner.VERDICT_PATH).is_file()


def test_execute_fails_closed_when_a_checkpoint_is_missing_evidence_for_a_sample(
        monkeypatch, tmp_path) -> None:
    """4/5 checkpoints contributing evidence for a sample must never be
    silently averaged as if it were 5/5."""
    _install_joint_bind_fixtures(monkeypatch)
    exit_code, _ = runner._bind_only(tmp_path)
    assert exit_code == runner.EXIT_PASS

    monkeypatch.setattr(probe, "construct_row_trainer", lambda repo, binding: binding)
    call_count = {"n": 0}

    def _forward(trainer_binding, records):
        call_count["n"] += 1
        out = {}
        for index, record in enumerate(records):
            if call_count["n"] == 1 and index == 0:
                continue                               # one sample missing from one checkpoint
            out[record.sample_identity] = np.array([0.0, 0.1])
        return out

    monkeypatch.setattr(probe, "forward_evidence_for_records", _forward)
    exit_code, payload = runner._execute(tmp_path)
    assert exit_code == runner.EXIT_BLOCKED
    assert payload["executed"] is False


def test_execute_reuses_the_bound_protocol_and_checkpoint_identities_in_results(
        monkeypatch, tmp_path) -> None:
    _install_execute_fixtures(
        monkeypatch, tmp_path, real_value=[0.0, 0.1],
        synthetic_value_by_arm={arm: [0.0, 0.1] for arm in probe.ARMS})
    runner._execute(tmp_path)
    verdict_doc = json.loads((tmp_path / runner.VERDICT_PATH).read_text())
    binding_doc = json.loads((tmp_path / runner.EXECUTION_BINDING_PATH).read_text())
    assert verdict_doc["protocol_identity"] == binding_doc["protocol_identity"]
    assert verdict_doc["checkpoint_binding_identity"] == \
        binding_doc["checkpoint_binding_identity_sha256"]
    assert verdict_doc["target_access"] == 0
    assert "c_h4_support_rule_is_separate" in verdict_doc


def test_execute_never_reports_c_h4_as_supported_by_a_hard_pass(monkeypatch, tmp_path) -> None:
    _install_execute_fixtures(
        monkeypatch, tmp_path, real_value=[0.0, 0.1],
        synthetic_value_by_arm={arm: [0.0, 0.1] for arm in probe.ARMS})
    runner._execute(tmp_path)
    verdict_doc = json.loads((tmp_path / runner.VERDICT_PATH).read_text())
    assert "C-H4" in verdict_doc["c_h4_support_rule_is_separate"]
    assert "supported" not in json.dumps(verdict_doc["verdict"]).lower()


# ==============================================================================
# H5. real construction/forwarding mechanics — source-level safety checks
# ==============================================================================

def test_construct_row_trainer_never_calls_training_methods() -> None:
    source = inspect.getsource(probe.construct_row_trainer)
    for forbidden in ("run_source_only_flow", ".backward(", ".optimizer.step(",
                      "trainer.save(", ".step(closure"):
        assert forbidden not in source, forbidden


def test_construct_row_trainer_sets_eval_mode() -> None:
    source = inspect.getsource(probe.construct_row_trainer)
    assert "trainer.model.eval()" in source


def test_construct_row_trainer_strict_loads_with_identity_check() -> None:
    source = inspect.getsource(probe.construct_row_trainer)
    assert "expected_identity=trainer.identity" in source
    assert "apply_checkpoint(" in source


def test_forward_evidence_for_records_reuses_extract_evidence() -> None:
    """Batched evaluation slices one sample's [global_logit, p_global] out
    of the batched ModelOutput and hands it to the SAME frozen
    `extract_evidence` every other evidence path uses — never a second
    2-field extraction rule."""
    source = inspect.getsource(probe.forward_evidence_for_records)
    assert "extract_evidence(" in source
    for forbidden in ("Image.open", "cv2.imread", "PIL."):
        assert forbidden not in source, forbidden


def test_forward_evidence_for_records_fails_closed_on_unresolvable_identity() -> None:
    class _FakeDataset:
        _real_position: dict[str, int] = {}
        bank = type("Bank", (), {"rows": []})()

    class _FakeTrainer:
        dataset = _FakeDataset()
        device = "cpu"

    record = probe.PopulationRecord("missing", "group", "casia_fasd", probe.REAL_SPOOF_CLASS)
    with pytest.raises(probe.SyntheticRealProbeError):
        probe.forward_evidence_for_records(_FakeTrainer(), [record])


def test_execute_joint_probe_rejects_a_binding_bound_to_a_different_protocol(
        monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(probe, "protocol_identity", lambda repo: "current" + "0" * 57)
    mismatched_binding = {"protocol_identity": "stale" + "0" * 59,
                          "checkpoint_binding_identity_sha256": "x" * 64, "checkpoints": []}
    with pytest.raises(probe.SyntheticRealProbeError, match="protocol identity"):
        probe.execute_joint_probe(tmp_path, checkpoint_binding=mismatched_binding,
                                  population_plan={"protocol_identity": "current" + "0" * 57,
                                                   "population_plan_identity_sha256": "y" * 64,
                                                   "cells": []})


# ==============================================================================
# H6. inference parity with C8 — device resolution, CUDA-safe evidence,
#     batched forward, and explicit pre-forward package/bank reverification
# ==============================================================================

def test_construct_row_trainer_never_hard_codes_cpu() -> None:
    source = inspect.getsource(probe.construct_row_trainer)
    assert 'device="cpu"' not in source
    assert "device='cpu'" not in source


def test_construct_row_trainer_uses_the_c8_scientific_device_resolver() -> None:
    source = inspect.getsource(probe.construct_row_trainer)
    assert "_scientific_device" in source
    assert "from prism_fas.pipeline.adapters.c7 import" in source


def test_construct_row_trainer_passes_the_resolved_device_to_m9trainer() -> None:
    source = inspect.getsource(probe.construct_row_trainer)
    assert "device = _scientific_device()" in source
    assert "device=device" in source


def test_construct_row_trainer_blocks_before_any_resolution_when_device_unavailable(
        monkeypatch) -> None:
    """Device resolution must fail closed BEFORE the C7 lock, the C6 bank or
    the trainer are ever touched — never a partial construction on a host
    with no scientific device."""
    import prism_fas.pipeline.adapters.c7 as c7_module

    class _DeviceUnavailable(RuntimeError):
        pass

    def _raise() -> str:
        raise _DeviceUnavailable("no CUDA on this host")

    monkeypatch.setattr(c7_module, "_scientific_device", _raise)

    called = {"n": 0}

    def _spy(*args, **kwargs):
        called["n"] += 1
        raise AssertionError("verify_detector_config_lock must not be reached")

    monkeypatch.setattr(c7_module, "verify_detector_config_lock", _spy)

    binding = probe.CheckpointBinding(
        arm="RND", seed=1, row_id="C-G-RND-P3READY-s20260806", run_identity="run",
        config_identity="c" * 64, checkpoint_sha256="a" * 64, checkpoint_path="p",
        checkpoint_kind="best", decision_logit_name="global_logit_G",
        decision_graph_hash="g" * 64)
    with pytest.raises(_DeviceUnavailable):
        probe.construct_row_trainer(REPO, binding)
    assert called["n"] == 0


def test_evidence_scalar_converts_before_numpy() -> None:
    source = inspect.getsource(probe._evidence_scalar)
    assert ".detach()" in source
    assert ".float()" in source
    assert ".cpu()" in source


def test_evidence_scalar_handles_a_tensor_requiring_detach_before_numpy() -> None:
    """A `requires_grad` tensor refuses `.numpy()` directly — the same shape
    of failure a CUDA tensor produces for `np.asarray` (for a different
    reason: device, not grad). Proving `_evidence_scalar` survives THIS
    case, entirely on CPU, proves the `.detach().float().cpu()` chain
    actually runs rather than being dead code."""
    import torch

    tensor = torch.tensor(3.25, requires_grad=True)
    with pytest.raises((RuntimeError, TypeError)):
        np.asarray(tensor)
    assert probe._evidence_scalar(tensor) == pytest.approx(3.25)


def test_evidence_scalar_passes_through_plain_numbers_and_arrays() -> None:
    assert probe._evidence_scalar(1.5) == 1.5
    assert probe._evidence_scalar(np.array([2.5])) == 2.5


def test_extract_evidence_reads_global_logit_correctly_from_a_tensor() -> None:
    import torch

    class _Output:
        global_logit = torch.tensor(1.75)
        p_global = torch.tensor(0.6)

    vector = probe.extract_evidence(_Output())
    assert vector[0] == pytest.approx(1.75)


def test_extract_evidence_reads_p_global_correctly_from_a_tensor() -> None:
    import torch

    class _Output:
        global_logit = torch.tensor(1.75)
        p_global = torch.tensor(0.6)

    vector = probe.extract_evidence(_Output())
    assert vector[1] == pytest.approx(0.6)


def test_extract_evidence_dimension_is_exactly_two() -> None:
    import torch

    class _Output:
        global_logit = torch.tensor(0.0)
        p_global = torch.tensor(0.0)

    vector = probe.extract_evidence(_Output())
    assert vector.shape == (2,) == (probe.EVIDENCE_DIMENSION,)


class _SpyModel:
    """Records every batch it is called with and the grad-enabled state at
    call time; returns per-sample evidence encoded from each item's own
    `sample_ids` entry, so a test can verify no sample was reordered,
    dropped or duplicated across batch boundaries."""

    def __init__(self) -> None:
        import torch

        self.torch = torch
        self.calls: list[list[str]] = []
        self.grad_enabled_during_call: list[bool] = []
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True

    def __call__(self, batch: Any) -> Any:
        self.calls.append(list(batch.sample_ids))
        self.grad_enabled_during_call.append(self.torch.is_grad_enabled())
        n = len(batch.sample_ids)
        # value encodes the sample's own position in the GLOBAL id list
        # (via a deterministic function of its id), never batch position.
        values = [float(int(sid.split("-")[-1])) for sid in batch.sample_ids]
        global_logit = self.torch.tensor(values, dtype=self.torch.float32)
        p_global = self.torch.tensor([v / 100.0 for v in values], dtype=self.torch.float32)
        return type("Output", (), {"global_logit": global_logit, "p_global": p_global})()


class _FakeBatch:
    """Stands in for a real `DetectorBatch`: carries only `sample_ids` (what
    `_SpyModel` and `forward_evidence_for_records` actually use) and a
    no-op `.to(device)`, so these tests exercise `forward_evidence_for_records`'s
    OWN chunking/ordering/no_grad logic without needing to satisfy
    `DetectorBatch.validate()`'s real image/region-prior shape contract —
    that contract has its own coverage elsewhere (`dataset.py`'s tests)."""

    def __init__(self, sample_ids: list[str]) -> None:
        self.sample_ids = tuple(sample_ids)

    def to(self, device: str) -> "_FakeBatch":
        return self


def _fake_trainer_for_forwarding(monkeypatch, *, batch_size: int, real_ids: list[str],
                                 synthetic_ids: list[str]):
    import prism_fas.detector.dataset as dataset_module

    monkeypatch.setattr(
        dataset_module, "collate_items",
        lambda items: _FakeBatch([item.sample_id for item in items]))

    class _FakeConfig:
        validation_batch_size = batch_size

    class _FakeItem:
        def __init__(self, sample_id: str) -> None:
            self.sample_id = sample_id

    real_position = {sid: i for i, sid in enumerate(real_ids)}
    synthetic_rows = [{"synthetic_id": sid} for sid in synthetic_ids]

    class _FakeBank:
        rows = synthetic_rows

    class _FakeDataset:
        _real_position = real_position
        bank = _FakeBank()

        def real_item(self, position: int):
            return _FakeItem(real_ids[position])

        def synthetic_item(self, position: int):
            return _FakeItem(synthetic_ids[position])

    class _FakeTrainer:
        config = _FakeConfig()
        dataset = _FakeDataset()
        device = "cpu"
        model = _SpyModel()

    return _FakeTrainer()


def test_forward_evidence_for_records_batches_with_validation_batch_size() -> None:
    source = inspect.getsource(probe.forward_evidence_for_records)
    assert "trainer.config.validation_batch_size" in source


def test_forward_evidence_for_records_calls_model_in_chunks_of_batch_size(monkeypatch) -> None:
    real_ids = [f"r-{i}" for i in range(5)]
    trainer = _fake_trainer_for_forwarding(monkeypatch, batch_size=2, real_ids=real_ids,
                                           synthetic_ids=[])
    records = [probe.PopulationRecord(sid, sid, "casia_fasd", probe.REAL_SPOOF_CLASS)
              for sid in real_ids]
    probe.forward_evidence_for_records(trainer, records)
    # 5 samples at batch_size=2 -> 3 forward calls (2, 2, 1)
    assert len(trainer.model.calls) == 3
    assert [len(c) for c in trainer.model.calls] == [2, 2, 1]


def test_forward_evidence_for_records_runs_under_no_grad(monkeypatch) -> None:
    real_ids = [f"r-{i}" for i in range(3)]
    trainer = _fake_trainer_for_forwarding(monkeypatch, batch_size=2, real_ids=real_ids,
                                           synthetic_ids=[])
    records = [probe.PopulationRecord(sid, sid, "casia_fasd", probe.REAL_SPOOF_CLASS)
              for sid in real_ids]
    probe.forward_evidence_for_records(trainer, records)
    assert trainer.model.grad_enabled_during_call
    assert all(enabled is False for enabled in trainer.model.grad_enabled_during_call)


def test_forward_evidence_for_records_gives_every_sample_exactly_one_vector(monkeypatch) -> None:
    real_ids = [f"r-{i}" for i in range(4)]
    synthetic_ids = [f"s-{i}" for i in range(100, 104)]
    trainer = _fake_trainer_for_forwarding(monkeypatch, batch_size=3, real_ids=real_ids,
                                           synthetic_ids=synthetic_ids)
    records = ([probe.PopulationRecord(sid, sid, "casia_fasd", probe.REAL_SPOOF_CLASS)
               for sid in real_ids]
              + [probe.PopulationRecord(sid, sid, "casia_fasd", probe.SYNTHETIC_SPOOF_CLASS)
                 for sid in synthetic_ids])
    result = probe.forward_evidence_for_records(trainer, records)
    assert set(result) == set(real_ids) | set(synthetic_ids)
    assert len(result) == len(real_ids) + len(synthetic_ids)


def test_forward_evidence_for_records_batching_does_not_reorder_or_mix_samples(monkeypatch) -> None:
    """Each sample's evidence must come from ITS OWN row of whichever batch
    it landed in — not from a neighbor's row, and not shifted by chunk
    boundaries. `_SpyModel` encodes each sample's evidence deterministically
    from its own id, so a mismatch here would mean batching corrupted the
    identity <-> evidence mapping."""
    real_ids = [f"r-{i}" for i in range(7)]
    trainer = _fake_trainer_for_forwarding(monkeypatch, batch_size=3, real_ids=real_ids,
                                           synthetic_ids=[])
    records = [probe.PopulationRecord(sid, sid, "casia_fasd", probe.REAL_SPOOF_CLASS)
              for sid in real_ids]
    result = probe.forward_evidence_for_records(trainer, records)
    for sid in real_ids:
        expected = float(int(sid.split("-")[-1]))
        assert result[sid][0] == pytest.approx(expected)
        assert result[sid][1] == pytest.approx(expected / 100.0)


def _bare_bound_artifacts(*, package_identity: str = "pkg" + "0" * 61,
                          bank_identities: dict[str, str] | None = None) -> tuple[dict, dict]:
    protocol_id = "a" * 64
    checkpoint_binding = {
        "protocol_identity": protocol_id,
        "source_package_identity": package_identity,
        "c6_bank_identities": bank_identities or {arm: f"bank-{arm}" + "0" * 55
                                                  for arm in probe.ARMS},
        "checkpoints": [],
    }
    checkpoint_binding["checkpoint_binding_identity_sha256"] = \
        probe.checkpoint_binding_identity(checkpoint_binding)
    population_plan = {"protocol_identity": protocol_id, "cells": []}
    population_plan["population_plan_identity_sha256"] = \
        probe.population_plan_identity(population_plan)
    return checkpoint_binding, population_plan


def _install_pre_forward_guard_fixtures(monkeypatch, *, current_package_identity: str,
                                        current_bank_identities: dict[str, str]):
    from prism_fas.pipeline.adapters import sources

    monkeypatch.setattr(probe, "protocol_identity", lambda repo: "a" * 64)
    monkeypatch.setattr(
        sources, "verify_detector_inputs",
        lambda repo, arms=None: {
            "package_identity": current_package_identity,
            "c6": {"banks": {arm: {"selected_set_sha256": current_bank_identities[arm]}
                            for arm in probe.ARMS}}})

    called = {"n": 0}

    def _spy(repo, binding):
        called["n"] += 1
        raise AssertionError("construct_row_trainer must not be reached")

    monkeypatch.setattr(probe, "construct_row_trainer", _spy)
    return called


def test_execute_blocks_on_source_package_identity_mismatch_before_construction(
        monkeypatch, tmp_path) -> None:
    checkpoint_binding, population_plan = _bare_bound_artifacts(package_identity="bound-pkg" + "0" * 55)
    called = _install_pre_forward_guard_fixtures(
        monkeypatch, current_package_identity="DIFFERENT-pkg" + "0" * 51,
        current_bank_identities=checkpoint_binding["c6_bank_identities"])
    # give the checkpoint binding SOME checkpoints so a later count-check
    # would not itself explain a raise before the package check does
    with pytest.raises(probe.SyntheticRealProbeError, match="source package identity"):
        probe.execute_joint_probe(tmp_path, checkpoint_binding=checkpoint_binding,
                                  population_plan=population_plan)
    assert called["n"] == 0


@pytest.mark.parametrize("arm", ["RND", "DET", "LLM"])
def test_execute_blocks_on_a_single_arm_c6_bank_mismatch_before_construction(
        monkeypatch, tmp_path, arm: str) -> None:
    checkpoint_binding, population_plan = _bare_bound_artifacts()
    current_banks = dict(checkpoint_binding["c6_bank_identities"])
    current_banks[arm] = "DIFFERENT-bank" + "0" * 50
    called = _install_pre_forward_guard_fixtures(
        monkeypatch, current_package_identity=checkpoint_binding["source_package_identity"],
        current_bank_identities=current_banks)
    with pytest.raises(probe.SyntheticRealProbeError, match=arm):
        probe.execute_joint_probe(tmp_path, checkpoint_binding=checkpoint_binding,
                                  population_plan=population_plan)
    assert called["n"] == 0


def test_execute_still_requires_exactly_fifteen_checkpoints(monkeypatch, tmp_path) -> None:
    checkpoint_binding, population_plan = _bare_bound_artifacts()
    _install_pre_forward_guard_fixtures(
        monkeypatch, current_package_identity=checkpoint_binding["source_package_identity"],
        current_bank_identities=checkpoint_binding["c6_bank_identities"])
    # package/bank checks pass (identities agree); checkpoints list is still
    # empty, so the existing per-arm count guard must still fire.
    with pytest.raises(probe.SyntheticRealProbeError, match="checkpoints"):
        probe.execute_joint_probe(tmp_path, checkpoint_binding=checkpoint_binding,
                                  population_plan=population_plan)


def test_construct_row_trainer_still_verifies_checkpoint_sha_on_disk() -> None:
    source = inspect.getsource(probe.construct_row_trainer)
    assert "on_disk_sha256 != binding.checkpoint_sha256" in source


def test_v2_protocol_identity_is_unchanged_by_the_inference_parity_fix() -> None:
    assert probe.protocol_identity(REPO) == \
        "720a2e344017d588d71005b81fdf0e7d2062081ae2f3881a61a306d952dc4ac8"


def test_pre_forward_guard_functions_never_reference_target_paths() -> None:
    source = inspect.getsource(probe.execute_joint_probe)
    for forbidden in ("siw", "SiW", "target_test", "target_taxonomy"):
        assert forbidden not in source, forbidden
    assert '"target_access": 0' in inspect.getsource(probe) or \
        "target_access" in source   # the module carries target_access=0 throughout


# ==============================================================================
# I. runner safety / import purity
# ==============================================================================

def test_bind_only_source_never_loads_checkpoint_weights_or_images() -> None:
    source = inspect.getsource(runner._bind_only)
    for forbidden in ("torch.load", "load_state_dict", "Image.open", "cv2.imread",
                      "fit_linear_probe(", "compute_ba_sep_for_seed(",
                      "forward_checkpoint_evidence(", "construct_row_trainer("):
        assert forbidden not in source, forbidden


def test_runner_module_has_no_target_path() -> None:
    source = Path(inspect.getfile(runner)).read_text(encoding="utf-8")
    for forbidden in ("siw", "SiW", "target_test", "target_taxonomy",
                      "resolve_target", "_real_target_roots"):
        assert forbidden not in source, forbidden


def test_importing_the_runner_touches_no_filesystem_state() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import prism_fas.evaluation.synthetic_real_probe_runner"],
        capture_output=True, text=True, cwd=str(REPO))
    assert result.returncode == 0, result.stderr
    importlib.import_module("prism_fas.evaluation.synthetic_real_probe_runner")


def test_runner_invocable_as_python_dash_m_module() -> None:
    """`--preflight-only` on THIS laptop's real repo, unmocked: the
    strengthened production preflight is honestly BLOCKED here (no GPU
    artifacts), which is itself the contract being proven — the CLI must
    not crash, must not write anything, and must report why."""
    result = subprocess.run(
        [sys.executable, "-m", "prism_fas.evaluation.synthetic_real_probe_runner",
         "--repo", str(REPO), "--preflight-only"],
        capture_output=True, text=True, cwd=str(REPO / "src"))
    assert result.returncode == runner.EXIT_BLOCKED, result.stderr
    payload = json.loads(result.stdout)
    assert payload["target_access"] == 0
    assert payload["ready_for_bind"] is False


def test_run_scientific_probe_single_arm_entry_point_is_retired() -> None:
    """Replaced by the joint `execute_joint_probe`; the old single-arm
    signature must refuse rather than silently compute a partial result."""
    with pytest.raises(probe.SyntheticRealProbeError, match="joint"):
        probe.run_scientific_probe(REPO, "DET")


def test_module_docstring_references_v2_as_current() -> None:
    source = Path(inspect.getfile(probe)).read_text(encoding="utf-8")
    assert "C9_DETECTOR_BA_SEP_OPTION1_V2" in source
