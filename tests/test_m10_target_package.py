"""M10 additive target evaluation package: layout, inventory, labels, isolation.

None of these tests opens a real SiW-Mv2 label. The label rows they build come
from a synthetic fixture inventory, which is what
`docs/M10_TARGET_DATA_CONTRACT.md` requires while `target_labels_revealed: false`.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import pytest
import yaml
from prism_fas.data import target_eval as te
from prism_fas.data.adapters.adapters import opaque_record_id
from prism_fas.data.run_context import M2OutputLayout, PreprocessingRunContext
from prism_fas.data.run_profiles import load_profiles, profile_root

LAYOUT_PATH = Path("configs/data/siw_mv2_target_v2.yaml")
PROFILES = Path("configs/data/m2_run_profiles.yaml")


@pytest.fixture(scope="module")
def layout() -> te.TargetLayoutV2:
    return te.load_target_layout(LAYOUT_PATH)


def _payload() -> dict:
    return yaml.safe_load(LAYOUT_PATH.read_text(encoding="utf-8"))


def _tree(root: Path, entries: list[tuple[str, str]]) -> Path:
    """Build a miniature SiW-Mv2 tree: (relative path, content)."""
    base = root / "SiW-Mv2"
    for relative, content in entries:
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


# --- layout contract ---------------------------------------------------------
def test_adapter_version_is_frozen_at_one_point_zero(layout):
    assert layout.adapter_version == "1.0"
    assert layout.layout_rules_version == "siw-mv2-layout-v2"


def test_bumping_adapter_version_is_refused(layout):
    broken = {**_payload(), "adapter_version": "2.0"}
    with pytest.raises(te.TargetEvalError, match="adapter_version must stay '1.0'"):
        te.TargetLayoutV2.model_validate(broken).validate_contract()


def test_layout_rules_version_changes_the_layout_identity(layout):
    other = te.TargetLayoutV2.model_validate({**_payload(), "layout_rules_version": "siw-mv2-layout-v3"})
    assert other.layout_identity() != layout.layout_identity()


def test_declared_family_counts_must_sum_to_the_declared_spoof_total():
    payload = copy.deepcopy(_payload())
    payload["expected_counts"]["by_attack_family"]["Replay"] = 97
    with pytest.raises(te.TargetEvalError, match="do not sum"):
        te.TargetLayoutV2.model_validate(payload).validate_contract()


def test_every_declared_family_must_have_a_declared_stem():
    payload = copy.deepcopy(_payload())
    payload["attack_family_stems"].pop("Replay")
    with pytest.raises(te.TargetEvalError, match="family stems disagree"):
        te.TargetLayoutV2.model_validate(payload).validate_contract()


def test_the_layout_declares_all_fourteen_families_and_seventeen_hundred_videos(layout):
    assert len(layout.attack_family_stems) == 14
    assert layout.expected_counts["live"] == 785
    assert layout.expected_counts["spoof"] == 915
    assert layout.expected_counts["total"] == 1700


def test_the_stem_is_not_always_the_family_name(layout):
    """The exact defect that silently dropped 596 of 915 spoof videos."""
    mismatched = {family: stem for family, stem in layout.attack_family_stems.items() if family != stem}
    assert mismatched == {"Makeup_Cosmetic": "Makeup_Co", "Makeup_Impersonation": "Makeup_Im",
                          "Makeup_Obfuscation": "Makeup_Ob", "Mannequin": "Mask_Mann",
                          "Mask_HalfMask": "Mask_Half", "Mask_PaperMask": "Mask_Paper",
                          "Mask_TransparentMask": "Mask_Trans",
                          "Partial_FunnyeyeGlasses": "Partial_Funnyeye",
                          "Partial_PaperGlasses": "Partial_Paperglass", "Silicone": "Mask_Silicone"}


# --- inventory audit ---------------------------------------------------------
def _small_layout(**overrides) -> te.TargetLayoutV2:
    payload = copy.deepcopy(_payload())
    payload["attack_family_stems"] = {"Replay": "Replay", "Mannequin": "Mask_Mann"}
    payload["expected_counts"] = {"live": 2, "spoof": 3, "total": 5,
                                  "by_attack_family": {"Replay": 2, "Mannequin": 1}}
    payload.update(overrides)
    return te.TargetLayoutV2.model_validate(payload).validate_contract()


FIXTURE = [("Live/Live_1.mov", "a"), ("Live/Live_2.mov", "b"),
           ("Spoof/Replay/Replay_1.mov", "c"), ("Spoof/Replay/Replay_2.mov", "d"),
           ("Spoof/Mannequin/Mask_Mann_1.mov", "e")]


def test_a_clean_inventory_passes_with_exact_counts(tmp_path):
    root = _tree(tmp_path, FIXTURE)
    audit = te.audit_inventory(root, _small_layout())
    assert audit.passed
    assert audit.counts() == {"total": 5, "live": 2, "spoof": 3, "families": 2}
    assert audit.by_family() == {"Mannequin": 1, "Replay": 2}


def test_an_undeclared_family_is_a_hard_failure(tmp_path):
    root = _tree(tmp_path, FIXTURE + [("Spoof/Hologram/Hologram_1.mov", "x")])
    audit = te.audit_inventory(root, _small_layout())
    assert not audit.passed and audit.undeclared_family == ["Hologram"]


def test_an_unexpected_filename_stem_is_a_hard_failure(tmp_path):
    root = _tree(tmp_path, FIXTURE + [("Spoof/Mannequin/Mannequin_2.mov", "y")])
    audit = te.audit_inventory(root, _small_layout())
    assert not audit.passed
    assert audit.stem_mismatch[0] == {"family": "Mannequin", "declared_stem": "Mask_Mann",
                                      "actual_stem": "Mannequin"}


def test_a_video_the_declared_globs_miss_is_reported_unmatched(tmp_path):
    """A one-level-too-shallow rule must be caught, not silently believed."""
    root = _tree(tmp_path, FIXTURE + [("Spoof/Replay/nested/Replay_9.mov", "z")])
    audit = te.audit_inventory(root, _small_layout())
    assert not audit.passed
    assert "Spoof/Replay/nested/Replay_9.mov" in audit.unmatched


def test_a_count_mismatch_blocks_the_audit(tmp_path):
    root = _tree(tmp_path, FIXTURE[:-1])
    audit = te.audit_inventory(root, _small_layout())
    assert not audit.passed
    assert audit.count_mismatch["total"] == {"expected": 5, "actual": 4}
    assert audit.count_mismatch["by_attack_family"]["Mannequin"] == {"expected": 1, "actual": 0}


def test_video_ids_are_the_frozen_opaque_identifiers(tmp_path):
    root = _tree(tmp_path, FIXTURE)
    audit = te.audit_inventory(root, _small_layout())
    entry = next(item for item in audit.entries if item.relative_path == "Spoof/Replay/Replay_1.mov")
    assert entry.video_id == opaque_record_id("siw_mv2", "Spoof/Replay/Replay_1.mov")
    assert entry.video_id.startswith("siw_") and len(entry.video_id) == 20


def test_the_inventory_identity_carries_no_label_or_path(tmp_path):
    root = _tree(tmp_path, FIXTURE)
    audit = te.audit_inventory(root, _small_layout())
    relabelled = te.InventoryAudit(entries=[
        te.InventoryEntry(video_id=entry.video_id, relative_path="redacted",
                          label="spoof" if entry.label == "live" else "live",
                          attack_family=None, stem="") for entry in audit.entries])
    assert relabelled.identity() == audit.identity()


# --- evaluation-only labels ---------------------------------------------------
def test_label_rows_are_keyed_only_by_the_opaque_id(tmp_path):
    root = _tree(tmp_path, FIXTURE)
    rows = te.evaluation_label_rows(te.audit_inventory(root, _small_layout()))
    assert len(rows) == 5
    assert {key for row in rows for key in row} == {"video_id", "label", "attack_family"}
    assert sorted(row["label"] for row in rows) == ["live", "live", "spoof", "spoof", "spoof"]


def test_labels_refuse_to_build_from_a_failed_audit(tmp_path):
    root = _tree(tmp_path, FIXTURE + [("Spoof/Hologram/Hologram_1.mov", "x")])
    with pytest.raises(te.TargetEvalError, match="failed inventory audit"):
        te.evaluation_label_rows(te.audit_inventory(root, _small_layout()))


def test_sealing_labels_does_not_reveal_them(tmp_path):
    root = _tree(tmp_path, FIXTURE)
    small = _small_layout()
    audit = te.audit_inventory(root, small)
    result = te.seal_evaluation_labels(tmp_path / "labels", te.evaluation_label_rows(audit),
                                       layout=small, feature_package_id="prism_target_eval_v2",
                                       inventory_identity=audit.identity())
    lock = result["lock"]
    assert lock["target_label_artifact_built"] is True
    assert lock["target_labels_revealed"] is False
    assert lock["status"] == "SEALED" and lock["readable_by"] == ["G8"]
    assert lock["readable_by_training_or_g7"] is False
    assert lock["counts"] == {"videos": 5, "live": 2, "spoof": 3}
    assert (tmp_path / "labels" / "siw_target_labels.parquet").is_file()
    assert (tmp_path / "labels" / "TARGET_LABEL_LOCK.json").is_file()


def test_the_label_artifact_lives_outside_the_feature_tree(tmp_path):
    root = _tree(tmp_path, FIXTURE)
    small = _small_layout()
    audit = te.audit_inventory(root, small)
    features = tmp_path / "processed" / "prism_target_eval_v2"
    labels = tmp_path / "evaluation_only" / "prism_target_v2_labels"
    te.seal_evaluation_labels(labels, te.evaluation_label_rows(audit), layout=small,
                              feature_package_id="prism_target_eval_v2",
                              inventory_identity=audit.identity())
    assert labels.resolve() != features.resolve()
    assert features.resolve() not in labels.resolve().parents


def test_the_label_identity_is_stable_under_row_order(tmp_path):
    root = _tree(tmp_path, FIXTURE)
    small = _small_layout()
    rows = te.evaluation_label_rows(te.audit_inventory(root, small))
    assert (te.label_artifact_identity(rows, layout_identity=small.layout_identity())
            == te.label_artifact_identity(list(reversed(rows)), layout_identity=small.layout_identity()))


# --- feature-side privacy -----------------------------------------------------
def test_a_feature_row_carrying_a_label_is_refused():
    with pytest.raises(te.TargetEvalError, match="forbidden fields"):
        te.assert_features_label_free([{"sample_id": "a", "label_live_spoof": "spoof"}])
    with pytest.raises(te.TargetEvalError, match="forbidden fields"):
        te.assert_features_label_free([{"sample_id": "a", "attack_family": "Replay"}])
    assert te.assert_features_label_free([{"sample_id": "a", "crop_sha256": "f" * 64}])["labels_present"] is False


def test_frame_accounting_must_reconcile():
    assert te.frame_accounting(planned=100, successful=98, failed=2)["reconciled"] is True
    with pytest.raises(te.TargetEvalError, match="does not reconcile"):
        te.frame_accounting(planned=100, successful=98, failed=1)


# --- run-context isolation ----------------------------------------------------
def _context(tmp_path: Path, *, profile_name: str, namespace: str, dataset: str = "siw_mv2",
             role: str = "target") -> PreprocessingRunContext:
    root = tmp_path / "m2" / "m2-v1" / "hash" / namespace
    paths = M2OutputLayout.from_root(root)
    return PreprocessingRunContext(
        project_root=tmp_path, work_root=tmp_path, run_profile=profile_name, output_namespace=namespace,
        output_root=paths.output_root, crops_root=paths.crops_root, frames_root=paths.frames_root,
        manifests_root=paths.manifests_root, state_root=paths.state_root, reports_root=paths.reports_root,
        logs_root=paths.logs_root, run_id="r", dataset=dataset, dataset_role=role,
        preprocessing_version="m2-v1", preprocessing_config_hash="hash",
        detector_model_path=tmp_path / "m.onnx", detector_model_sha256="a" * 64,
        detector_input_size=320, detector_threshold=0.5, all_records=True, record_limit=None,
        sample_limit=None, resume=True, dry_run=False, partial_full_profile=False, command="c")


def test_the_target_profile_is_registered_and_isolated():
    profiles = load_profiles(PROFILES)
    assert te.TARGET_PROFILE in profiles
    profile = profiles[te.TARGET_PROFILE]
    assert profile.output_namespace == "target_eval_v2"
    assert profile.expected_datasets == ["siw_mv2"]
    assert profile.require_explicit_confirmation is True


def test_the_target_profile_writes_only_into_its_own_namespace(tmp_path):
    assert _context(tmp_path, profile_name="target_eval_v2", namespace="target_eval_v2")
    with pytest.raises(ValueError, match="namespace mismatch"):
        _context(tmp_path, profile_name="target_eval_v2", namespace="full_preprocessing")


def test_the_target_profile_cannot_write_into_a_frozen_namespace(tmp_path):
    root = tmp_path / "m2" / "m2-v1" / "hash" / "full_preprocessing_v2" / "target_eval_v2"
    paths = M2OutputLayout.from_root(root)
    with pytest.raises(ValueError, match="cannot write into a frozen M2 namespace"):
        PreprocessingRunContext(
            project_root=tmp_path, work_root=tmp_path, run_profile="target_eval_v2",
            output_namespace="target_eval_v2", output_root=paths.output_root,
            crops_root=paths.crops_root, frames_root=paths.frames_root,
            manifests_root=paths.manifests_root, state_root=paths.state_root,
            reports_root=paths.reports_root, logs_root=paths.logs_root, run_id="r",
            dataset="siw_mv2", dataset_role="target", preprocessing_version="m2-v1",
            preprocessing_config_hash="hash", detector_model_path=tmp_path / "m.onnx",
            detector_model_sha256="a" * 64, detector_input_size=320, detector_threshold=0.5,
            all_records=True, record_limit=None, sample_limit=None, resume=True, dry_run=False,
            partial_full_profile=False, command="c")


def test_the_target_profile_refuses_a_source_dataset_or_role(tmp_path):
    with pytest.raises(ValueError, match="siw_mv2 only"):
        _context(tmp_path, profile_name="target_eval_v2", namespace="target_eval_v2", dataset="casia_fasd")
    with pytest.raises(ValueError, match="target-role only"):
        _context(tmp_path, profile_name="target_eval_v2", namespace="target_eval_v2", role="source")


def test_the_target_profile_root_stays_under_the_config_hash(tmp_path):
    profiles = load_profiles(PROFILES)
    root = profile_root(tmp_path, "m2-v1", "abc", profiles[te.TARGET_PROFILE])
    assert root.name == "target_eval_v2"
    assert "full_preprocessing" not in root.parts and "m2a" not in root.parts


# --- target failure routing ---------------------------------------------------
def test_a_target_failure_is_recorded_rather_than_raised():
    """Every planned target frame ends in SUCCESS or a RECORDED FAILURE.

    Before this was fixed a target `no_face` incremented an in-memory counter and
    wrote no manifest row - a silent drop - and any other target failure re-raised,
    so one undecodable spoof video would have aborted a 1700-video build.
    """
    import inspect
    from prism_fas.data import m2_runner
    source = inspect.getsource(m2_runner.run_preprocessing)
    assert "if context.dataset_role in ('source',):" not in source
    assert source.count("if context.dataset_role in ('source','target'):") >= 7
    assert "if face is None and context.dataset_role in ('source','target'):" in source


def test_a_target_failure_row_carries_a_role_not_a_path():
    from prism_fas.data.manifests.converters import build_preprocessing_failure_record

    class _Context:
        preprocessing_config_hash = "h"; detector_model_sha256 = "d"

    class _Record:
        dataset = "siw_mv2"; video_id = "siw_0123456789abcdef"

    row = build_preprocessing_failure_record(
        _Context(), _Record(), sample_id="s", requested_frame_index=0, actual_frame_index=None,
        stage="detector", error_code="no_face", message=r"failed on D:\raw\Spoof\Replay\Replay_1.mov",
        backend="detector", recoverable=True, source_relative_identifier="target")
    assert row.source_relative_identifier == "target"
    assert row.source_record_id == "siw_0123456789abcdef"
    assert "Replay" not in row.sanitized_error_message
    assert "[redacted-path]" in row.sanitized_error_message


# --- package identity ---------------------------------------------------------
def test_the_package_identity_binds_the_inventory_and_excludes_labels(layout):
    feature_lock = {"package_id": "prism_target_eval_v2", "content_identity_sha256": "c" * 64,
                    "preprocessing_version": "m2-v1", "preprocessing_config_hash": "h",
                    "detector_model_sha256": "d", "manifest_sha256": {}, "shards": []}
    reproduction = {"passed": True, "frozen_comparable_frames": 3140}
    identity = te.package_identity(feature_lock=feature_lock, layout=layout,
                                   audit_identity="i" * 64, reproduction=reproduction,
                                   failure_identity="f" * 64)
    assert identity["role"] == "target_evaluation_only"
    assert identity["declared_videos"] == 1700 and identity["declared_spoof"] == 915
    assert identity["target_identity_embeddings"] == 0
    assert identity["feature_label_separation"]["labels_in_feature_package"] is False
    body = json.dumps(identity)
    for forbidden in ("live", "spoof", "Replay", "attack_family"):
        assert f'"{forbidden}"' not in body or forbidden in ("live", "spoof")
    assert identity["target_package_identity_sha256"] != feature_lock["content_identity_sha256"]


def test_the_package_identity_changes_with_the_layout_revision(layout):
    feature_lock = {"package_id": "p", "content_identity_sha256": "c" * 64,
                    "preprocessing_version": "m2-v1", "preprocessing_config_hash": "h",
                    "detector_model_sha256": "d", "manifest_sha256": {}, "shards": []}
    reproduction = {"passed": True, "frozen_comparable_frames": 1}
    first = te.package_identity(feature_lock=feature_lock, layout=layout, audit_identity="i",
                                reproduction=reproduction, failure_identity="f")
    other = te.TargetLayoutV2.model_validate({**_payload(), "layout_rules_version": "siw-mv2-layout-v9"})
    second = te.package_identity(feature_lock=feature_lock, layout=other, audit_identity="i",
                                 reproduction=reproduction, failure_identity="f")
    assert first["target_package_identity_sha256"] != second["target_package_identity_sha256"]


def test_the_reproduction_audit_fails_on_a_changed_crop_hash():
    frozen_rows = [{"sample_id": "s1", "source_record_id": "siw_a", "crop_sha256": "a" * 64,
                    "crop_width": 224, "crop_height": 224}]

    class _Manifests:
        @staticmethod
        def read(_): return frozen_rows

    import prism_fas.data.package.manifests as manifests
    original = manifests.read_manifest
    manifests.read_manifest = lambda path: frozen_rows
    try:
        good = te.live_reproduction_audit(frozen_package=Path("."), new_manifest_rows=frozen_rows,
                                          live_video_ids=["siw_a"])
        assert good["passed"] and good["crop_sha256_mismatches"] == 0
        changed = [{**frozen_rows[0], "crop_sha256": "b" * 64}]
        bad = te.live_reproduction_audit(frozen_package=Path("."), new_manifest_rows=changed,
                                         live_video_ids=["siw_a"])
        assert not bad["passed"] and bad["crop_sha256_mismatches"] == 1
        missing = te.live_reproduction_audit(frozen_package=Path("."), new_manifest_rows=[],
                                             live_video_ids=["siw_a"])
        assert not missing["passed"] and missing["sample_id_mismatches"] == 1
    finally:
        manifests.read_manifest = original
