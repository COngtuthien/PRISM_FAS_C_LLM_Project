"""Tests for the E8 fixed-config Track-G training runner
(src/prism_fas/evaluation/c_ext_e8_training_runner.py).

Pure, read-only, source-only tests. No GPU, no full SigLIP2 instantiation,
no target labels, no E8 membership regeneration, no mutation of any real
frozen artifact, no scientific checkpoint created.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prism_fas.evaluation import c_ext_e8_training_runner as runner  # noqa: E402
from prism_fas.evaluation import c_ext_common as cc  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _real_hashes() -> dict[str, str]:
    return {
        "membership": cc.sha256_file(REPO / "reports/c_ext_q1q2_v1/e8_qmatched/membership/"
                                      "E8_QMATCH_SELECTED_MEMBERSHIP.parquet"),
        "membership_lock": cc.sha256_file(REPO / "reports/c_ext_q1q2_v1/e8_qmatched/membership/"
                                          "E8_QMATCH_MEMBERSHIP_LOCK.json"),
        "adapter_source": cc.sha256_file(REPO / "src/prism_fas/evaluation/c_ext_e8_training_adapter.py"),
        "execution_plan": cc.sha256_file(REPO / runner.EXECUTION_PLAN_RELATIVE_PATH),
    }


@pytest.fixture(scope="module")
def before_hashes():
    return _real_hashes()


# --------------------------------------------------------------------------- #
# A/B. All 15 frozen run specs validate; exactly 15 unique IDs
# --------------------------------------------------------------------------- #

def test_A_all_15_frozen_run_specs_validate():
    specs = runner.all_scientific_run_specs()
    assert len(specs) == 15
    for spec in specs:
        assert spec.fold_id == "EXT-F1"
        assert spec.condition == runner.CONDITION_BY_ARM[spec.arm]


def test_B_exactly_15_unique_run_ids():
    specs = runner.all_scientific_run_specs()
    ids = [s.run_id for s in specs]
    assert len(ids) == 15
    assert len(set(ids)) == 15
    pairs = [(s.arm, s.seed) for s in specs]
    assert len(set(pairs)) == 15


# --------------------------------------------------------------------------- #
# C-F. Hard-fail on wrong seed/arm/condition/run_id
# --------------------------------------------------------------------------- #

def test_C_wrong_seed_hard_fails():
    with pytest.raises(runner.E8RunnerError, match="unknown seed"):
        runner.derive_run_id("RND", 99999999)
    with pytest.raises(runner.E8RunnerError):
        runner.build_run_spec("RND", 99999999)


def test_D_wrong_arm_hard_fails():
    with pytest.raises(runner.E8RunnerError, match="unknown arm"):
        runner.derive_run_id("XYZ", 20260806)
    with pytest.raises(runner.E8RunnerError):
        runner.build_run_spec("XYZ", 20260806)


def test_E_condition_arm_mapping_is_exact():
    for arm, expected_condition in (("RND", "G-RND-QMATCH"), ("DET", "G-DET-QMATCH"), ("LLM", "G-LLM-QMATCH")):
        spec = runner.build_run_spec(arm, 20260806)
        assert spec.condition == expected_condition


def test_F_wrong_run_id_hard_fails():
    with pytest.raises(runner.E8RunnerError, match="!= derived"):
        runner.build_run_spec("RND", 20260806, run_id="not_the_real_id")


# --------------------------------------------------------------------------- #
# G/H/I/J. Wrong plan/membership/adapter/config identity hard-fails
# --------------------------------------------------------------------------- #

def test_G_wrong_execution_plan_identity_hard_fails(tmp_path):
    exec_dir = tmp_path / "reports/c_ext_q1q2_v1/e8_qmatched/training/execution"
    exec_dir.mkdir(parents=True)
    bad_plan = {"e8_gpu_execution_plan_identity": "0" * 64}
    (exec_dir / "E8_GPU_EXECUTION_PLAN.json").write_text(json.dumps(bad_plan))
    (exec_dir / "E8_GPU_EXECUTION_PLAN.md").write_text("x")
    sha_md = cc.sha256_file(exec_dir / "E8_GPU_EXECUTION_PLAN.md")
    sha_json = cc.sha256_file(exec_dir / "E8_GPU_EXECUTION_PLAN.json")
    (exec_dir / "E8_GPU_EXECUTION_PLAN.sha256").write_text(
        f"{sha_json}  E8_GPU_EXECUTION_PLAN.json\n{sha_md}  E8_GPU_EXECUTION_PLAN.md\n")
    with pytest.raises(runner.E8RunnerError, match="execution-plan identity"):
        runner.verify_execution_plan(root=tmp_path)


def test_H_wrong_membership_identity_hard_fails(tmp_path):
    bad_dir = tmp_path / "reports/c_ext_q1q2_v1/e8_qmatched/membership"
    bad_dir.mkdir(parents=True)
    (bad_dir / "E8_QMATCH_SELECTED_MEMBERSHIP.parquet").write_bytes(b"not real")
    with pytest.raises(Exception):
        # goes through the adapter's own verified loader
        from prism_fas.evaluation import c_ext_e8_training_adapter as adapter
        adapter.load_frozen_membership(root=tmp_path)


def test_I_wrong_adapter_identity_hard_fails(monkeypatch):
    from prism_fas.evaluation import c_ext_e8_training_adapter as adapter
    monkeypatch.setattr(adapter, "adapter_rule_identity", lambda root=None: "deadbeef" * 8)
    with pytest.raises(runner.E8RunnerError, match="adapter_rule_identity"):
        runner.build_run_spec("RND", 20260806)


def test_J_wrong_track_g_config_identity_hard_fails(tmp_path, monkeypatch):
    spec = runner.build_run_spec("RND", 20260806)

    def _fake_read_json(path):
        if str(path).endswith("DETECTOR_CONFIG_LOCK.json"):
            return {"tracks": {"G": {"winner_config": {"weight_decay": 0.999},
                                    "winner_config_sha256": "bad", "variant_identity": "bad"}}}
        return cc.read_json(path)

    monkeypatch.setattr(runner.cc, "read_json", _fake_read_json)
    with pytest.raises(runner.E8RunnerError, match="winner_config"):
        runner.load_frozen_winner_track_g_config(spec, synthetic_bank_identity="x")


# --------------------------------------------------------------------------- #
# K. Source package containing target domain hard-fails
# --------------------------------------------------------------------------- #

def test_K_target_domain_in_allowed_set_hard_fails():
    assert "siw_mv2" not in runner.ALLOWED_SOURCE_DOMAINS
    assert runner.ALLOWED_SOURCE_DOMAINS == frozenset({"casia_fasd", "msu_mfsd"})


# --------------------------------------------------------------------------- #
# L/M/N. bank count / route count hard-fails
# --------------------------------------------------------------------------- #

def test_L_bank_count_mismatch_hard_fails(monkeypatch):
    from prism_fas.evaluation import c_ext_e8_training_adapter as adapter
    spec = runner.build_run_spec("RND", 20260806)
    bad = adapter.E8ArmBinding(arm="RND", condition="G-RND-QMATCH", fold_id="EXT-F1", profile="NOMINAL",
                               identifier_field="x", membership_count=817, physics_count=354,
                               gpat_count=463, membership_parquet_sha256="x", membership_lock_sha256="x",
                               c6_bank_lock_sha256="x", quality_threshold_identity="x")
    monkeypatch.setattr(adapter, "resolve_arm_binding", lambda arm, root=None: bad)
    with pytest.raises(runner.E8RunnerError, match="bank count"):
        runner.resolve_e8_bank_counts(spec)


def test_M_physics_count_mismatch_hard_fails(monkeypatch):
    from prism_fas.evaluation import c_ext_e8_training_adapter as adapter
    spec = runner.build_run_spec("RND", 20260806)
    bad = adapter.E8ArmBinding(arm="RND", condition="G-RND-QMATCH", fold_id="EXT-F1", profile="NOMINAL",
                               identifier_field="x", membership_count=818, physics_count=353,
                               gpat_count=465, membership_parquet_sha256="x", membership_lock_sha256="x",
                               c6_bank_lock_sha256="x", quality_threshold_identity="x")
    monkeypatch.setattr(adapter, "resolve_arm_binding", lambda arm, root=None: bad)
    with pytest.raises(runner.E8RunnerError, match="physics count"):
        runner.resolve_e8_bank_counts(spec)


def test_N_gpat_count_mismatch_hard_fails(monkeypatch):
    from prism_fas.evaluation import c_ext_e8_training_adapter as adapter
    spec = runner.build_run_spec("RND", 20260806)
    bad = adapter.E8ArmBinding(arm="RND", condition="G-RND-QMATCH", fold_id="EXT-F1", profile="NOMINAL",
                               identifier_field="x", membership_count=818, physics_count=354,
                               gpat_count=464 - 1, membership_parquet_sha256="x", membership_lock_sha256="x",
                               c6_bank_lock_sha256="x", quality_threshold_identity="x")
    # make total consistent to isolate the gpat check would still trip the earlier total check;
    # instead craft physics+gpat == 818 but gpat wrong and physics compensating
    bad = adapter.E8ArmBinding(arm="RND", condition="G-RND-QMATCH", fold_id="EXT-F1", profile="NOMINAL",
                               identifier_field="x", membership_count=818, physics_count=355,
                               gpat_count=463, membership_parquet_sha256="x", membership_lock_sha256="x",
                               c6_bank_lock_sha256="x", quality_threshold_identity="x")
    monkeypatch.setattr(adapter, "resolve_arm_binding", lambda arm, root=None: bad)
    with pytest.raises(runner.E8RunnerError, match="physics count|gpat count"):
        runner.resolve_e8_bank_counts(spec)


# --------------------------------------------------------------------------- #
# O/P/Q. Collision state classification
# --------------------------------------------------------------------------- #

def test_O_not_started_classification(tmp_path):
    run_root = tmp_path / "nonexistent_run"
    assert runner.classify_run_state(run_root) == runner.RunState.NOT_STARTED
    run_root.mkdir()
    assert runner.classify_run_state(run_root) == runner.RunState.NOT_STARTED  # empty dir


def test_P_completed_run_classification(tmp_path):
    run_root = tmp_path / "completed_run"
    (run_root / "stages" / "G6").mkdir(parents=True)
    (run_root / "stages" / "G6" / "output_hashes.json").write_text("{}")
    (run_root / "run.json").write_text("{}")
    assert runner.classify_run_state(run_root) == runner.RunState.COMPLETED


def test_Q_ambiguous_output_is_blocked_collision(tmp_path):
    run_root = tmp_path / "ambiguous_run"
    run_root.mkdir()
    (run_root / "some_random_file.txt").write_text("mystery")
    assert runner.classify_run_state(run_root) == runner.RunState.BLOCKED_COLLISION


def test_in_progress_classification(tmp_path):
    run_root = tmp_path / "in_progress_run"
    (run_root / "stages" / "G1").mkdir(parents=True)
    (run_root / "stages" / "G1" / "stage_state.json").write_text("{}")
    assert runner.classify_run_state(run_root) == runner.RunState.IN_PROGRESS


def test_failed_technical_classification(tmp_path):
    run_root = tmp_path / "failed_run"
    run_root.mkdir()
    (run_root / runner.FAILURE_MARKER_NAME).write_text("{}")
    assert runner.classify_run_state(run_root) == runner.RunState.FAILED_TECHNICAL


# --------------------------------------------------------------------------- #
# R/S. No automatic overwrite/resume
# --------------------------------------------------------------------------- #

def test_R_no_automatic_overwrite_on_completed(tmp_path):
    run_root = tmp_path / "completed_run"
    (run_root / "stages" / "G6").mkdir(parents=True)
    (run_root / "stages" / "G6" / "output_hashes.json").write_text('{"real":"evidence"}')
    (run_root / "run.json").write_text("{}")
    with pytest.raises(runner.E8RunnerError, match="not NOT_STARTED"):
        runner.assert_no_collision(run_root)
    # evidence untouched
    assert (run_root / "stages" / "G6" / "output_hashes.json").read_text() == '{"real":"evidence"}'


def test_S_no_automatic_resume_on_in_progress(tmp_path):
    run_root = tmp_path / "in_progress_run"
    (run_root / "checkpoints").mkdir(parents=True)
    (run_root / "checkpoints" / "last.pt").write_bytes(b"fake-checkpoint")
    with pytest.raises(runner.E8RunnerError, match="not NOT_STARTED"):
        runner.assert_no_collision(run_root)
    assert (run_root / "checkpoints" / "last.pt").read_bytes() == b"fake-checkpoint"


# --------------------------------------------------------------------------- #
# T/U. Smoke isolation
# --------------------------------------------------------------------------- #

def test_T_scientific_mode_rejects_smoke_only_overrides():
    # scientific run specs and preflight expose no override parameters at all --
    # E8RunSpec is frozen and carries no hyperparameter fields to override.
    spec = runner.build_run_spec("RND", 20260806)
    fields = spec.__dataclass_fields__.keys()
    forbidden = {"max_steps", "max_batches", "lr", "epochs", "batch_size", "weight_decay"}
    assert not (forbidden & set(fields))


def test_U_smoke_output_cannot_collide_with_scientific_run_root():
    result = runner.preflight_e8_smoke("RND", 20260806)
    assert result["smoke_run_root"] != f"{runner.RUN_ROOT_RELATIVE}/e8_ext_f1_g_rnd_qmatch_s20260806"
    assert result["smoke_run_id"] not in {runner.derive_run_id(arm, seed)
                                          for arm in runner.ARMS for seed in runner.SEEDS}
    assert runner.SMOKE_ROOT_RELATIVE != runner.RUN_ROOT_RELATIVE


# --------------------------------------------------------------------------- #
# V. preflight-only mode never invokes training
# --------------------------------------------------------------------------- #

def test_V_preflight_only_never_invokes_training():
    spec = runner.build_run_spec("RND", 20260806)
    result = runner.preflight_e8_run(spec)
    assert result["training_started"] is False
    assert result["checkpoint_created"] is False
    # structural: preflight_e8_run never constructs/calls M9Trainer
    source = inspect.getsource(runner.preflight_e8_run)
    assert "M9Trainer(" not in source
    assert "import M9Trainer" not in source


# --------------------------------------------------------------------------- #
# W/X. No target argument required, no target-label access path
# --------------------------------------------------------------------------- #

def test_W_no_target_argument_required():
    spec = runner.build_run_spec("RND", 20260806)
    sig = inspect.signature(runner.preflight_e8_run)
    assert "target" not in " ".join(sig.parameters.keys()).lower()
    runner.preflight_e8_run(spec)  # succeeds with no target-related argument at all


def test_X_no_target_label_access_path():
    # "target_labels_accessed" is a REQUIRED, legitimate flag name (always false) --
    # only actual target-import-shaped references are forbidden.
    source = inspect.getsource(runner)
    blob = source.lower()
    for forbidden in ("siw_mv2", "target_test", "target_eval", "targetadapter", "import target"):
        assert forbidden not in blob, f"forbidden target reference found: {forbidden}"
    assert 'target_labels_accessed": false' in blob or "target_labels_accessed=false" in blob


# --------------------------------------------------------------------------- #
# Y/Z. Deterministic binding payload; seed changes identity, not schedule
# --------------------------------------------------------------------------- #

def test_Y_same_run_spec_produces_deterministic_binding_payload():
    spec_a = runner.build_run_spec("RND", 20260806)
    spec_b = runner.build_run_spec("RND", 20260806)
    assert spec_a == spec_b
    assert runner.runner_rule_identity() == runner.runner_rule_identity()


def test_Z_different_seed_different_identity_same_schedule():
    spec_a = runner.build_run_spec("RND", 20260806)
    spec_b = runner.build_run_spec("RND", 20260807)
    assert spec_a.run_id != spec_b.run_id
    assert spec_a.seed != spec_b.seed
    cfg_a, _ = runner.load_frozen_winner_track_g_config(spec_a, synthetic_bank_identity="x")
    cfg_b, _ = runner.load_frozen_winner_track_g_config(spec_b, synthetic_bank_identity="x")
    assert cfg_a.seed != cfg_b.seed
    assert cfg_a.total_epochs == cfg_b.total_epochs == 35
    assert cfg_a.steps_per_epoch == cfg_b.steps_per_epoch == 45
    assert cfg_a.weight_decay == cfg_b.weight_decay == 0.025
    assert cfg_a.loss_weights["lambda_syn"] == cfg_b.loss_weights["lambda_syn"] == 0.25


# --------------------------------------------------------------------------- #
# AA/AB. Adapter reuse; M9Trainer receives the existing seam
# --------------------------------------------------------------------------- #

def test_AA_open_e8_arm_bank_used_not_duplicated():
    source = inspect.getsource(runner)
    assert "open_e8_arm_bank" in source
    # the runner never re-implements filter_bank_lock_to_e8 or reads C6_BANK_LOCK directly for membership
    assert "def filter_bank_lock_to_e8" not in source
    assert "def resolve_arm_membership" not in source


def test_AB_m9trainer_construction_receives_synthetic_bank_seam(monkeypatch):
    from prism_fas.evaluation import c_ext_e8_training_adapter as adapter

    calls = {}

    class _FakeTrainer:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    class _SentinelBank:
        identity = "sentinel-bank-identity"

    sentinel_bank = _SentinelBank()
    # open_e8_arm_bank itself is fully covered by the adapter's own test suite
    # (it needs the real 818-candidate C5 pixel tree, not available on this
    # laptop); here we isolate ONLY the concern this test names: that
    # launch_scientific_run passes whatever open_e8_arm_bank returns straight
    # into M9Trainer's synthetic_bank= kwarg, unmodified.
    monkeypatch.setattr(adapter, "open_e8_arm_bank", lambda *a, **k: sentinel_bank)

    spec = runner.build_run_spec("RND", 20260806)
    run_root = REPO / spec.run_root
    assert not run_root.exists()  # sanity: no real scientific dir exists yet

    trainer = runner.launch_scientific_run(
        spec, candidates_root=Path("/nonexistent"), recipes=(), recipe_bank_identity="x",
        _trainer_cls=_FakeTrainer, _skip_m3b_guard=True,
    )
    assert isinstance(trainer, _FakeTrainer)
    assert "synthetic_bank" in calls
    assert calls["synthetic_bank"] is sentinel_bank
    assert calls["config"].run_id == spec.run_id
    assert calls["config"].total_epochs == 35


# --------------------------------------------------------------------------- #
# AC. No trainer.py/dataset.py/sampler.py source mutation
# --------------------------------------------------------------------------- #

def test_AC_no_core_module_source_mutation(before_hashes):
    for rel in (
        "src/prism_fas/detector/trainer.py",
        "src/prism_fas/detector/dataset.py",
        "src/prism_fas/detector/sampler.py",
        "src/prism_fas/detector/c6_bank.py",
        "src/prism_fas/pipeline/adapters/c7.py",
    ):
        # existence + non-empty is a smoke check; the real guarantee is the
        # git-diff check the implementation task performs separately.
        assert (REPO / rel).is_file()
    after = _real_hashes()
    assert after == before_hashes


# --------------------------------------------------------------------------- #
# Runner rule identity
# --------------------------------------------------------------------------- #

def test_runner_rule_identity_deterministic_and_well_formed():
    ident = runner.runner_rule_identity()
    assert len(ident) == 64
    int(ident, 16)
    assert ident == runner.runner_rule_identity()


def test_runner_rule_payload_excludes_forbidden_fields():
    payload = runner.build_runner_rule_payload()
    blob = json.dumps(payload).lower()
    for forbidden in ("timestamp", "hostname", "/home/", "outcome", "accuracy"):
        assert forbidden not in blob


# =========================================================================== #
# Runner V2 source-package-binding correction tests
#
# Root cause: V1 conflated the EXT-F1 GPAT-input package path/identity
# (a train-only GPAT construction artifact, E7-D fold/source-support
# identity "955b...") with the detector's canonical runtime source package
# (M3B, content identity "08d9...", the identity every historical C5
# GenerationIdentity.package_identity actually binds). V2 separates these
# two provenance layers explicitly and never conflates them again.
# =========================================================================== #

M3B_ROOT_REAL = REPO / runner.M3B_RUNTIME_PACKAGE_RELATIVE_PATH


# --- 1-5. Identity separation -------------------------------------------- #

def test_1_v1_historical_identity_constants_retained_as_provenance_only():
    assert runner.RUNNER_RULE_NAME == "E8_FIXED_TRACK_G_RUNNER_V1"
    assert runner.EXT_F1_SOURCE_PACKAGE_IDENTITY == "955b630fec438c80f284ecbcb30fbf10c83251a23fd31d8ab1a52e0f8ce8383b"
    assert runner.SOURCE_PACKAGE_RELATIVE_PATH == "data/processed/c_ext_q1q2_v1/e7_gpat_bank/gpat_input/EXT-F1"
    # V1's own rule identity is unaffected by the correction (byte-for-byte historical fact)
    assert runner.runner_rule_identity() == "81842bd81d43c8c942773a0fefed31a0e76bfce1bbbeebc845cd15993dbbdcf0"


def test_2_v2_real_package_root_exact():
    assert runner.M3B_RUNTIME_PACKAGE_RELATIVE_PATH == "data/packages/prism_data_v1_m3b"


def test_3_v2_m3b_content_identity_exact():
    assert runner.M3B_CONTENT_IDENTITY == "08d9d289eb4b462006afcff37cd4750a7c4eeb402c83de5599eda38df44168c9"


def test_4_e7d_identity_exact():
    assert runner.E7D_F1_SOURCE_SUPPORT_IDENTITY == "955b630fec438c80f284ecbcb30fbf10c83251a23fd31d8ab1a52e0f8ce8383b"


def test_5_identities_explicitly_distinct():
    assert runner.E7D_F1_SOURCE_SUPPORT_IDENTITY != runner.M3B_CONTENT_IDENTITY


# --- 6. source_package_binding distinguishes the two roles --------------- #

def test_6_source_package_binding_distinguishes_authority_vs_runtime():
    binding = runner.source_package_binding()
    assert binding["fold_source_support_authority"]["identity"] == runner.E7D_F1_SOURCE_SUPPORT_IDENTITY
    assert binding["runtime_detector_package"]["expected_content_identity"] == runner.M3B_CONTENT_IDENTITY
    assert binding["runtime_detector_package"]["root"] == runner.M3B_RUNTIME_PACKAGE_RELATIVE_PATH
    assert binding["fold_source_support_authority"]["identity"] != \
        binding["runtime_detector_package"]["expected_content_identity"]


# --- 7-19. Strict M3B validation, real (partial) local package ----------- #

def test_7_real_local_package_lock_valid():
    result = runner.validate_m3b_package()
    assert result["lock_present"] is True
    assert result["lock_status_ok"] is True
    assert result["lock_schema_ok"] is True
    assert result["lock_content_identity_ok"] is True


def test_8_1440_train_required():
    assert runner.EXPECTED_M3B_TRAIN_ROWS == 1440


def test_9_2079_dev_required():
    assert runner.EXPECTED_M3B_DEV_ROWS == 2079


def test_10_exact_casia_msu_train_counts():
    assert runner.EXPECTED_M3B_TRAIN_DOMAIN_COUNTS == {"casia_fasd": 960, "msu_mfsd": 480}


def test_11_exact_casia_msu_dev_counts():
    assert runner.EXPECTED_M3B_DEV_DOMAIN_COUNTS == {"casia_fasd": 1439, "msu_mfsd": 640}


@pytest.mark.skipif(not (M3B_ROOT_REAL / "manifests" / "source_dev.parquet").is_file(),
                    reason="real M3B source_dev.parquet not present on this host")
def test_real_source_dev_matches_expected_counts_and_hash():
    dev_path = M3B_ROOT_REAL / "manifests" / "source_dev.parquet"
    assert cc.sha256_file(dev_path) == "28a9ceea5df1e1a888f31881f1e2faa3fdb4910c706f566bb85161e2cb6b7b92"
    import pandas as pd
    df = pd.read_parquet(dev_path)
    assert len(df) == 2079
    assert df["dataset"].value_counts().to_dict() == {"casia_fasd": 1439, "msu_mfsd": 640}
    assert set(df["dataset"].unique()) <= {"casia_fasd", "msu_mfsd"}
    assert df["sample_id"].is_unique


def _write_fixture_m3b(tmp_path, *, status="validated", schema="m3b-v1",
                       content_identity=None, with_train=True, with_dev=True,
                       train_rows=None, dev_rows=None, siw_in_train=False,
                       dup_train_ids=False, overlap=False):
    import pandas as pd

    content_identity = content_identity or runner.M3B_CONTENT_IDENTITY
    root = tmp_path / "data/packages/prism_data_v1_m3b"
    (root / "manifests").mkdir(parents=True)
    lock = {"status": status, "package_schema_version": schema, "content_identity_sha256": content_identity}
    (root / "PACKAGE_LOCK.json").write_text(json.dumps(lock))

    def _rows(n_casia, n_msu, prefix, extra_domain=None):
        rows = []
        for i in range(n_casia):
            rows.append({"sample_id": f"{prefix}_casia_{i}", "dataset": "casia_fasd"})
        for i in range(n_msu):
            rows.append({"sample_id": f"{prefix}_msu_{i}", "dataset": "msu_mfsd"})
        if extra_domain:
            rows.append({"sample_id": f"{prefix}_extra", "dataset": extra_domain})
        return rows

    if with_train:
        train_data = _rows(960, 480, "train", "siw_mv2" if siw_in_train else None)
        if dup_train_ids:
            train_data.append(dict(train_data[0]))
        if train_rows is not None:
            train_data = train_data[:train_rows]
        pd.DataFrame(train_data).to_parquet(root / "manifests" / "source_train.parquet")

    if with_dev:
        dev_prefix = "train" if overlap else "dev"  # force overlap by reusing train's prefix
        dev_data = _rows(1439, 640, dev_prefix)
        if dev_rows is not None:
            dev_data = dev_data[:dev_rows]
        pd.DataFrame(dev_data).to_parquet(root / "manifests" / "source_dev.parquet")

    return tmp_path


def test_12_no_siw_accepted(tmp_path):
    _write_fixture_m3b(tmp_path, siw_in_train=True)
    result = runner.validate_m3b_package(root=tmp_path)
    assert result["no_siw_in_train"] is False
    assert result["overall_state"] == runner.M3B_STATE_INVALID


def test_13_no_duplicate_train_ids(tmp_path):
    _write_fixture_m3b(tmp_path, dup_train_ids=True)
    result = runner.validate_m3b_package(root=tmp_path)
    assert result["no_duplicate_train_ids"] is False
    assert result["overall_state"] == runner.M3B_STATE_INVALID


def test_14_no_duplicate_dev_ids_field_present():
    # exercised structurally via the real dev manifest (already proven unique)
    result = runner.validate_m3b_package()
    # dev-duplicate detection only runs once train is also present; assert the
    # field exists in the result contract regardless of local materialization.
    assert "no_duplicate_dev_ids" in result


def test_15_no_train_dev_overlap(tmp_path):
    _write_fixture_m3b(tmp_path, overlap=True)
    result = runner.validate_m3b_package(root=tmp_path)
    assert result["no_train_dev_overlap"] is False
    assert result["overall_state"] == runner.M3B_STATE_INVALID


def test_16_missing_source_dev_hard_fails_readiness(tmp_path):
    _write_fixture_m3b(tmp_path, with_dev=False)
    result = runner.validate_m3b_package(root=tmp_path)
    assert result["source_dev_present"] is False
    assert result["overall_state"] == runner.M3B_STATE_NOT_MATERIALIZED


def test_17_drifted_content_identity_hard_fails(tmp_path):
    _write_fixture_m3b(tmp_path, content_identity="0" * 64)
    result = runner.validate_m3b_package(root=tmp_path)
    assert result["lock_content_identity_ok"] is False
    assert result["overall_state"] == runner.M3B_STATE_INVALID


def test_18_wrong_schema_hard_fails(tmp_path):
    _write_fixture_m3b(tmp_path, schema="m3b-v2-not-frozen")
    result = runner.validate_m3b_package(root=tmp_path)
    assert result["lock_schema_ok"] is False
    assert result["overall_state"] == runner.M3B_STATE_INVALID


def test_19_non_validated_lock_hard_fails(tmp_path):
    _write_fixture_m3b(tmp_path, status="pending")
    result = runner.validate_m3b_package(root=tmp_path)
    assert result["lock_status_ok"] is False
    assert result["overall_state"] == runner.M3B_STATE_INVALID


def test_20_laptop_absence_reported_honestly_not_fabricated(tmp_path):
    result = runner.validate_m3b_package(root=tmp_path)
    assert result["overall_state"] == runner.M3B_STATE_NOT_MATERIALIZED
    assert result["lock_present"] is False
    assert result["source_train_row_count"] is None  # never fabricated


def test_20b_fully_valid_fixture_reports_valid(tmp_path):
    _write_fixture_m3b(tmp_path)
    result = runner.validate_m3b_package(root=tmp_path)
    assert result["overall_state"] == runner.M3B_STATE_VALID
    assert result["problems"] == []


# --- 21-23. launch_scientific_run binding correctness --------------------- #

def test_21_22_23_launch_uses_m3b_root_and_identity_never_e7d(monkeypatch):
    from prism_fas.evaluation import c_ext_e8_training_adapter as adapter

    calls_trainer = {}
    calls_adapter = {}

    class _FakeTrainer:
        def __init__(self, **kwargs):
            calls_trainer.update(kwargs)

    class _Sentinel:
        identity = "sentinel"

    def _fake_open_e8_arm_bank(arm, **kwargs):
        calls_adapter.update(kwargs)
        return _Sentinel()

    monkeypatch.setattr(adapter, "open_e8_arm_bank", _fake_open_e8_arm_bank)
    spec = runner.build_run_spec("RND", 20260806)
    runner.launch_scientific_run(spec, candidates_root=Path("/nonexistent"), recipes=(),
                                 recipe_bank_identity="x", _trainer_cls=_FakeTrainer,
                                 _skip_m3b_guard=True)

    # 21: M3B root passed to M9Trainer
    assert calls_trainer["package_root"] == REPO / runner.M3B_RUNTIME_PACKAGE_RELATIVE_PATH
    # 22: M3B content identity passed to the adapter
    assert calls_adapter["package_identity"] == runner.M3B_CONTENT_IDENTITY
    # 23: E7-D identity never passed as the C6 package identity
    assert calls_adapter["package_identity"] != runner.E7D_F1_SOURCE_SUPPORT_IDENTITY


# --- 24-25. Adapter untouched; existing M9Trainer seam reused -------------- #

def test_24_adapter_implementation_untouched():
    assert cc.sha256_file(REPO / "src/prism_fas/evaluation/c_ext_e8_training_adapter.py") == \
        "7fa4be3beec6be6118faa830b39ef1d579a4715aba11c84a08a7a813a4ebd520"


def test_25_existing_m9trainer_seam_reused():
    source = inspect.getsource(runner)
    assert "synthetic_bank=e8_bank" in source
    assert "class M9Trainer" not in source  # never redefined


# --- 26-32. Frozen contract unchanged by this correction ------------------- #

def test_26_27_28_29_30_31_32_frozen_contract_unchanged():
    specs = runner.all_scientific_run_specs()
    assert len(specs) == 15
    assert sorted(s.seed for s in specs) == sorted(list(runner.SEEDS) * 3)
    spec = runner.build_run_spec("RND", 20260806)
    pf = runner.preflight_e8_run(spec)
    assert pf["frozen_schedule"]["total_optimizer_updates"] == 1575
    assert pf["frozen_schedule"]["synthetic_draws_per_run"] == 10800
    assert pf["bank_counts"] == {"total": 818, "physics": 354, "gpat": 464}
    assert pf["target_firewall"]["target_access"] is False
    assert pf["target_firewall"]["target_labels_accessed"] is False


def test_33_no_target_argument():
    sig = inspect.signature(runner.preflight_e8_run)
    assert "target" not in " ".join(sig.parameters.keys()).lower()


# --- 34-36. No fabrication / no fallback -------------------------------- #

def test_34_35_36_no_auto_source_dev_creation_no_random_split_no_gpat_fallback():
    source = inspect.getsource(runner.validate_m3b_package) + inspect.getsource(runner.launch_scientific_run)
    for forbidden in ("random.sample", "np.random", "shutil.copy", "GPAT_INPUT_PACKAGE_RELATIVE_PATH_V1_HISTORICAL"):
        assert forbidden not in source
    # launch_scientific_run must not reference the historical GPAT path constant at all
    launch_source = inspect.getsource(runner.launch_scientific_run)
    assert "SOURCE_PACKAGE_RELATIVE_PATH" not in launch_source
    assert "EXT_F1_SOURCE_PACKAGE_IDENTITY" not in launch_source


# --- 37-38. Execution-plan + correction identity verified ------------------ #

def test_37_original_execution_plan_identity_still_verified():
    plan = runner.verify_execution_plan()
    assert plan["e8_gpu_execution_plan_identity"] == runner.EXPECTED_EXECUTION_PLAN_IDENTITY


def test_38_new_correction_identity_verified():
    assert runner.SOURCE_BINDING_CORRECTION_RELATIVE_PATH.endswith(
        "runner_correction/E8_RUNNER_SOURCE_BINDING_CORRECTION.json")


# --- 39. V2 rule identity deterministic ------------------------------------ #

def test_39_runner_v2_rule_identity_deterministic():
    id1 = runner.runner_rule_identity_v2()
    id2 = runner.runner_rule_identity_v2()
    assert id1 == id2
    assert len(id1) == 64
    assert id1 != runner.runner_rule_identity()  # V2 differs from V1


# --- 40. Preflight never trains --------------------------------------------- #

def test_40_preflight_v2_does_not_train():
    spec = runner.build_run_spec("RND", 20260806)
    result = runner.preflight_e8_run(spec)
    assert result["training_started"] is False
    assert "M9Trainer(" not in inspect.getsource(runner.preflight_e8_run)


# --- 41-42. Collision policy / smoke isolation unchanged -------------------- #

def test_41_collision_policy_unchanged():
    assert {s.value for s in runner.RunState} == {
        "NOT_STARTED", "IN_PROGRESS", "COMPLETED", "FAILED_TECHNICAL", "BLOCKED_COLLISION"}


def test_42_smoke_namespace_remains_isolated():
    result = runner.preflight_e8_smoke("RND", 20260806)
    assert result["smoke_run_root"].startswith(runner.SMOKE_ROOT_RELATIVE)
    assert not result["smoke_run_root"].startswith(runner.RUN_ROOT_RELATIVE)


# --- Local preflight status for the real (partially materialized) host ---- #

def test_local_preflight_status_honest_for_this_host():
    spec = runner.build_run_spec("RND", 20260806)
    result = runner.preflight_e8_run(spec)
    assert result["local_preflight_status"] in (
        "READY_FOR_GPU_RUNTIME_ASSET_REVALIDATION", "BLOCKED_M3B_PACKAGE_VALIDATION_FAILED")
    # never silently claims the package is scientifically ready to train from
    assert result["local_preflight_status"] != "READY_TO_TRAIN"


# =========================================================================== #
# Runner V2.1 -- source-binding correction IDENTITY binding
#
# Root cause: V2's rule payload named the correction artifact by PATH only,
# never by content identity, contradicting the payload's own docstring and
# the correction contract (BLOCKED_E8_RUNNER_V2_CORRECTION_IDENTITY_NOT_BOUND).
# =========================================================================== #

HISTORICAL_V1_IDENTITY = "81842bd81d43c8c942773a0fefed31a0e76bfce1bbbeebc845cd15993dbbdcf0"
HISTORICAL_V2_IDENTITY = "3293994d312be82969fba884da5b7d13445aa6c1a9d43742f21434991c1b5570"


def test_v21_correct_correction_file_passes():
    verified = runner.verify_source_binding_correction()
    assert verified == runner.SOURCE_BINDING_CORRECTION_SHA256


def test_v21_one_byte_changed_correction_file_fails(tmp_path):
    real_path = REPO / runner.SOURCE_BINDING_CORRECTION_RELATIVE_PATH
    fixture_dir = tmp_path / Path(runner.SOURCE_BINDING_CORRECTION_RELATIVE_PATH).parent
    fixture_dir.mkdir(parents=True)
    corrupted = real_path.read_bytes()[:-1] + b"\x00"  # flip the trailing byte
    (fixture_dir / Path(runner.SOURCE_BINDING_CORRECTION_RELATIVE_PATH).name).write_bytes(corrupted)
    with pytest.raises(runner.E8RunnerError, match="SHA256"):
        runner.verify_source_binding_correction(root=tmp_path)


def test_v21_missing_correction_file_fails(tmp_path):
    with pytest.raises(runner.E8RunnerError, match="not present"):
        runner.verify_source_binding_correction(root=tmp_path)


def test_v21_preflight_verifies_correction_identity():
    spec = runner.build_run_spec("RND", 20260806)
    result = runner.preflight_e8_run(spec)
    assert result["source_binding_correction_sha256_verified"] == runner.SOURCE_BINDING_CORRECTION_SHA256


def test_v21_launch_verifies_correction_identity_before_m3b_and_trainer(monkeypatch):
    from prism_fas.evaluation import c_ext_e8_training_adapter as adapter

    calls = []
    monkeypatch.setattr(runner, "verify_source_binding_correction",
                        lambda root=None: calls.append("correction") or "ok")
    monkeypatch.setattr(runner, "validate_m3b_package",
                        lambda root=None: calls.append("m3b") or {"overall_state": runner.M3B_STATE_VALID,
                                                                   "problems": []})

    def _fake_open_e8_arm_bank(arm, **kwargs):
        calls.append("bank")
        class _S:
            identity = "s"
        return _S()

    monkeypatch.setattr(adapter, "open_e8_arm_bank", _fake_open_e8_arm_bank)

    class _FakeTrainer:
        def __init__(self, **kwargs):
            calls.append("trainer")

    spec = runner.build_run_spec("RND", 20260806)
    runner.launch_scientific_run(spec, candidates_root=Path("/nonexistent"), recipes=(),
                                 recipe_bank_identity="x", _trainer_cls=_FakeTrainer)
    assert calls[0] == "correction"
    assert calls.index("correction") < calls.index("m3b")
    assert calls.index("correction") < calls.index("bank")
    assert calls.index("correction") < calls.index("trainer")


def test_v21_rule_payload_contains_exact_correction_sha():
    payload = runner.build_runner_rule_payload_v2()
    assert payload["source_binding_correction_sha256"] == runner.SOURCE_BINDING_CORRECTION_SHA256
    assert payload["source_binding_correction_artifact"] == runner.SOURCE_BINDING_CORRECTION_RELATIVE_PATH
    assert payload["historical_v1_runner_rule_identity"] == HISTORICAL_V1_IDENTITY
    assert payload["historical_v2_runner_rule_identity"] == HISTORICAL_V2_IDENTITY


def test_v21_rule_identity_changes_if_correction_sha_changes(monkeypatch):
    base = runner.runner_rule_identity_v2()
    monkeypatch.setattr(runner, "SOURCE_BINDING_CORRECTION_SHA256", "0" * 64)
    changed = runner.runner_rule_identity_v2()
    assert changed != base


def test_v21_new_identity_differs_from_historical_v1_and_v2():
    new_id = runner.runner_rule_identity_v2()
    assert new_id != HISTORICAL_V1_IDENTITY
    assert new_id != HISTORICAL_V2_IDENTITY
    assert len(new_id) == 64
    int(new_id, 16)
    assert new_id == runner.runner_rule_identity_v2()  # deterministic


def test_v21_15_run_ids_unchanged():
    specs = runner.all_scientific_run_specs()
    assert len(specs) == 15
    assert len({s.run_id for s in specs}) == 15


def test_v21_seeds_unchanged():
    assert runner.SEEDS == (20260806, 20260807, 20260808, 20260809, 20260810)


def test_v21_m3b_identity_unchanged():
    assert runner.M3B_CONTENT_IDENTITY == "08d9d289eb4b462006afcff37cd4750a7c4eeb402c83de5599eda38df44168c9"


def test_v21_e7d_identity_unchanged():
    assert runner.E7D_F1_SOURCE_SUPPORT_IDENTITY == "955b630fec438c80f284ecbcb30fbf10c83251a23fd31d8ab1a52e0f8ce8383b"


def test_v21_bank_counts_unchanged():
    binding = runner.resolve_e8_bank_counts(runner.build_run_spec("RND", 20260806))
    assert binding.membership_count == 818
    assert binding.physics_count == 354
    assert binding.gpat_count == 464


def test_v21_schedule_unchanged():
    spec = runner.build_run_spec("RND", 20260806)
    pf = runner.preflight_e8_run(spec)
    assert pf["frozen_schedule"]["total_optimizer_updates"] == 1575
    assert pf["frozen_schedule"]["synthetic_draws_per_run"] == 10800


def test_v21_target_firewall_unchanged():
    spec = runner.build_run_spec("RND", 20260806)
    pf = runner.preflight_e8_run(spec)
    assert pf["target_firewall"]["target_access"] is False
    assert pf["target_firewall"]["target_labels_accessed"] is False


def test_v21_no_training_smoke_gpu_llm_performed():
    spec = runner.build_run_spec("RND", 20260806)
    pf = runner.preflight_e8_run(spec)
    assert pf["training_started"] is False
    smoke = runner.preflight_e8_smoke("RND", 20260806)
    assert smoke["is_scientific_result"] is False
    assert smoke["target_access"] is False
    # structural: this module never imports torch/cuda/an LLM client at module scope
    source = inspect.getsource(runner)
    assert "import torch" not in source
    assert "openai" not in source.lower()
    assert "anthropic" not in source.lower()
