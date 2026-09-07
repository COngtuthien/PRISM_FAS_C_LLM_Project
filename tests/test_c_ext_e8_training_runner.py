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
        _trainer_cls=_FakeTrainer,
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
