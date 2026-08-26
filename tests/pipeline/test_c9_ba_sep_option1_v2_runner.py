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
# H. the CLI runner — contracts per mode
# ==============================================================================

def test_preflight_only_exits_pass_when_protocol_resolves() -> None:
    exit_code = runner.main(["--repo", str(REPO), "--preflight-only"])
    assert exit_code == runner.EXIT_PASS


def test_preflight_only_reports_zero_target_access(capsys) -> None:
    runner.main(["--repo", str(REPO), "--preflight-only"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["target_access"] == 0


def test_preflight_only_never_fits_a_probe_or_writes_state(capsys) -> None:
    runner.main(["--repo", str(REPO), "--preflight-only"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["probe_fit_executed"] is False
    assert payload["ba_metric_computed"] is False
    assert payload["state_modified"] is False
    assert payload["scientific_artifacts_written"] is False


def test_preflight_only_reports_per_arm_checkpoint_resolution(capsys) -> None:
    runner.main(["--repo", str(REPO), "--preflight-only"])
    payload = json.loads(capsys.readouterr().out)
    assert set(payload["checkpoints_by_arm"]) == set(probe.ARMS)


def test_preflight_only_with_arm_is_a_usage_error() -> None:
    exit_code = runner.main(["--repo", str(REPO), "--preflight-only", "--arm", "RND"])
    assert exit_code == runner.EXIT_USAGE


def test_bind_only_without_arm_is_a_usage_error() -> None:
    exit_code = runner.main(["--repo", str(REPO), "--bind-only"])
    assert exit_code == runner.EXIT_USAGE


def test_execute_without_arm_is_a_usage_error() -> None:
    exit_code = runner.main(["--repo", str(REPO), "--execute"])
    assert exit_code == runner.EXIT_USAGE


def test_no_mode_flag_is_an_argparse_usage_error() -> None:
    with pytest.raises(SystemExit):
        runner.main(["--repo", str(REPO)])


def test_bind_only_fails_closed_on_this_laptop_with_no_c8_artifacts(tmp_path) -> None:
    """This development clone has no runs/full/c8/; bind-only must refuse,
    write nothing, and never fabricate a partial binding."""
    exit_code = runner.main(["--repo", str(REPO), "--bind-only", "--arm", "DET"])
    assert exit_code == runner.EXIT_BLOCKED
    assert not (REPO / runner.BINDING_ARTIFACT_DIR).exists() or \
        not (REPO / runner.BINDING_ARTIFACT_DIR / "BINDING_DET.json").exists()


def test_execute_refuses_on_every_arm() -> None:
    for arm in probe.ARMS:
        exit_code = runner.main(["--repo", str(REPO), "--execute", "--arm", arm])
        assert exit_code == runner.EXIT_BLOCKED


def test_execute_reports_zero_target_access_and_no_execution(capsys) -> None:
    runner.main(["--repo", str(REPO), "--execute", "--arm", "RND"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["executed"] is False
    assert payload["target_access"] == 0


def test_bind_only_writes_one_artifact_on_full_success(tmp_path, monkeypatch) -> None:
    fake_protocol = {
        "detector_checkpoint_identity": {"required_decision_logit_name": "global_logit_G"},
        "matched_source_split": {"split_hash_namespace": "test-ns"},
        "probe_seed_values": [1, 2, 3],
        "source_domains": ["casia_fasd"],
    }
    monkeypatch.setattr(probe, "load_protocol", lambda repo: fake_protocol)
    monkeypatch.setattr(probe, "protocol_identity", lambda repo: "f" * 64)
    checkpoints = [probe.CheckpointBinding(arm="DET", seed=s, row_id=f"row-{s}",
                                           run_identity="run", checkpoint_sha256="a" * 64,
                                           checkpoint_path="p") for s in range(5)]
    monkeypatch.setattr(probe, "resolve_checkpoint_set", lambda repo, arm: checkpoints)
    real = [probe.PopulationRecord(f"r{i}", f"g{i}", "casia_fasd", 0) for i in range(20)]
    synthetic = [probe.PopulationRecord(f"s{i}", f"g{i}", "casia_fasd", 1) for i in range(20)]
    monkeypatch.setattr(probe, "resolve_arm_populations", lambda repo, arm: (real, synthetic))

    exit_code = runner.main(["--repo", str(tmp_path), "--bind-only", "--arm", "DET"])
    assert exit_code == runner.EXIT_PASS
    artifact = tmp_path / runner.BINDING_ARTIFACT_DIR / "BINDING_DET.json"
    assert artifact.is_file()
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["bound"] is True
    assert payload["checkpoint_weights_loaded"] is False
    assert payload["images_forwarded"] is False
    assert payload["ba_metric_computed"] is False
    assert len(payload["checkpoint_bindings"]) == 5
    assert all(entry["decision_logit_name"] == "global_logit_G"
              for entry in payload["checkpoint_bindings"])


def test_bind_only_refuses_a_protocol_with_the_wrong_decision_logit(tmp_path, monkeypatch) -> None:
    fake_protocol = {
        "detector_checkpoint_identity": {"required_decision_logit_name": "some_other_logit"},
        "matched_source_split": {"split_hash_namespace": "test-ns"},
        "probe_seed_values": [1], "source_domains": ["casia_fasd"],
    }
    monkeypatch.setattr(probe, "load_protocol", lambda repo: fake_protocol)
    monkeypatch.setattr(probe, "protocol_identity", lambda repo: "e" * 64)
    exit_code = runner.main(["--repo", str(tmp_path), "--bind-only", "--arm", "DET"])
    assert exit_code == runner.EXIT_BLOCKED
    assert not (tmp_path / runner.BINDING_ARTIFACT_DIR).exists()


def test_bind_only_writes_nothing_when_checkpoint_resolution_fails(tmp_path, monkeypatch) -> None:
    fake_protocol = {
        "detector_checkpoint_identity": {"required_decision_logit_name": "global_logit_G"},
        "matched_source_split": {"split_hash_namespace": "test-ns"},
        "probe_seed_values": [1], "source_domains": ["casia_fasd"],
    }
    monkeypatch.setattr(probe, "load_protocol", lambda repo: fake_protocol)
    monkeypatch.setattr(probe, "protocol_identity", lambda repo: "d" * 64)

    def _raise(repo, arm):
        raise probe.SyntheticRealProbeError("no checkpoints on this host")

    monkeypatch.setattr(probe, "resolve_checkpoint_set", _raise)
    exit_code = runner.main(["--repo", str(tmp_path), "--bind-only", "--arm", "DET"])
    assert exit_code == runner.EXIT_BLOCKED
    assert not (tmp_path / runner.BINDING_ARTIFACT_DIR).exists()


def test_bind_only_is_deterministic_across_repeated_calls(tmp_path, monkeypatch) -> None:
    fake_protocol = {
        "detector_checkpoint_identity": {"required_decision_logit_name": "global_logit_G"},
        "matched_source_split": {"split_hash_namespace": "test-ns"},
        "probe_seed_values": [1, 2], "source_domains": ["casia_fasd"],
    }
    monkeypatch.setattr(probe, "load_protocol", lambda repo: fake_protocol)
    monkeypatch.setattr(probe, "protocol_identity", lambda repo: "c" * 64)
    checkpoints = [probe.CheckpointBinding(arm="RND", seed=s, row_id=f"row-{s}",
                                           run_identity="run", checkpoint_sha256="b" * 64,
                                           checkpoint_path="p") for s in range(5)]
    monkeypatch.setattr(probe, "resolve_checkpoint_set", lambda repo, arm: checkpoints)
    real = [probe.PopulationRecord(f"r{i}", f"g{i}", "casia_fasd", 0) for i in range(12)]
    synthetic = [probe.PopulationRecord(f"s{i}", f"g{i}", "casia_fasd", 1) for i in range(12)]
    monkeypatch.setattr(probe, "resolve_arm_populations", lambda repo, arm: (real, synthetic))

    runner.main(["--repo", str(tmp_path), "--bind-only", "--arm", "RND"])
    first = (tmp_path / runner.BINDING_ARTIFACT_DIR / "BINDING_RND.json").read_text()
    runner.main(["--repo", str(tmp_path), "--bind-only", "--arm", "RND"])
    second = (tmp_path / runner.BINDING_ARTIFACT_DIR / "BINDING_RND.json").read_text()
    assert first == second


def test_bind_only_verifies_group_safety_before_writing(tmp_path, monkeypatch) -> None:
    """A split that leaks (train/validation share a group) must block the
    write, not silently ship a leaking binding."""
    fake_protocol = {
        "detector_checkpoint_identity": {"required_decision_logit_name": "global_logit_G"},
        "matched_source_split": {"split_hash_namespace": "test-ns"},
        "probe_seed_values": [1], "source_domains": ["casia_fasd"],
    }
    monkeypatch.setattr(probe, "load_protocol", lambda repo: fake_protocol)
    monkeypatch.setattr(probe, "protocol_identity", lambda repo: "9" * 64)
    checkpoints = [probe.CheckpointBinding(arm="LLM", seed=s, row_id=f"row-{s}",
                                           run_identity="run", checkpoint_sha256="7" * 64,
                                           checkpoint_path="p") for s in range(5)]
    monkeypatch.setattr(probe, "resolve_checkpoint_set", lambda repo, arm: checkpoints)
    monkeypatch.setattr(probe, "resolve_arm_populations", lambda repo, arm: ([], []))

    def _raise_leak(split):
        raise probe.SyntheticRealProbeError("group-safety violated: forced test failure")

    monkeypatch.setattr(probe, "verify_group_safe_split", _raise_leak)
    exit_code = runner.main(["--repo", str(tmp_path), "--bind-only", "--arm", "LLM"])
    assert exit_code == runner.EXIT_BLOCKED
    assert not (tmp_path / runner.BINDING_ARTIFACT_DIR).exists()


# ==============================================================================
# I. runner safety / import purity
# ==============================================================================

def test_execute_source_never_loads_checkpoint_weights_or_images() -> None:
    source = inspect.getsource(runner._execute)
    for forbidden in ("torch.load", "load_state_dict", "Image.open", "cv2.imread",
                      "fit_linear_probe(", "compute_ba_sep_for_seed("):
        assert forbidden not in source, forbidden


def test_bind_only_source_never_loads_checkpoint_weights_or_images() -> None:
    source = inspect.getsource(runner._bind_only)
    for forbidden in ("torch.load", "load_state_dict", "Image.open", "cv2.imread",
                      "fit_linear_probe(", "compute_ba_sep_for_seed(",
                      "forward_checkpoint_evidence("):
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
    result = subprocess.run(
        [sys.executable, "-m", "prism_fas.evaluation.synthetic_real_probe_runner",
         "--repo", str(REPO), "--preflight-only"],
        capture_output=True, text=True, cwd=str(REPO / "src"))
    assert result.returncode == runner.EXIT_PASS, result.stderr
    payload = json.loads(result.stdout)
    assert payload["target_access"] == 0


def test_run_scientific_probe_remains_unwired_after_the_v2_correction() -> None:
    with pytest.raises(NotImplementedError):
        probe.run_scientific_probe(REPO, "DET")


def test_module_docstring_references_v2_as_current() -> None:
    source = Path(inspect.getfile(probe)).read_text(encoding="utf-8")
    assert "C9_DETECTOR_BA_SEP_OPTION1_V2" in source
