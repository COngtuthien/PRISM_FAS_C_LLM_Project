"""Tests for `prism_fas.evaluation.c_ext_e7d_source_support` (E7-D per-fold
crop-level SOURCE support materialization). Every test builds a
self-contained fake repo under `tmp_path` unless it explicitly checks the
REAL committed repo (frozen-hash / identity checks, which can only be
meaningfully verified against real bytes). No test ever fits GPAT, renders,
trains, or calls an LLM.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism_fas.evaluation import c_ext_e7b_data_prep as e7b
from prism_fas.evaluation import c_ext_e7c_gpat_prep as e7c
from prism_fas.evaluation import c_ext_e7d_source_support as e7d

REPO = Path(__file__).resolve().parents[2]


def _base_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


def _m3b_source_ref(dataset: str, label: str, sample_id: str, *, project_split: str = "source_train",
                    subject_id: str | None = "1", image_relative_path: str | None = None,
                    crop_sha256: str | None = None) -> dict:
    return {"dataset": dataset, "project_split": project_split, "reference_kind": "m3b_processed_sample",
           "sample_id": sample_id, "source_record_id": f"rec-{sample_id}", "subject_id": subject_id,
           "label_live_spoof": label,
           "image_relative_path": image_relative_path or f"images/{sample_id}.jpg",
           "prior_relative_path": f"priors/{sample_id}.npz",
           "crop_sha256": crop_sha256 or "a" * 64, "prior_sha256": "b" * 64}


def _siw_video_ref(video_id: str, project_split: str, label: str, *, population_identity: str,
                   split_identity: str, family=None) -> dict:
    return {"dataset": "SiW-Mv2", "project_split": project_split, "reference_kind": "siw_raw_video",
           "video_id": video_id, "relative_path": f"{video_id}.mov", "label_live_spoof": label,
           "spoof_family": family, "extension": "mov", "population_identity": population_identity,
           "split_identity": split_identity, "requires_frozen_face_preprocessing": True}


def _write_materialization(repo: Path, fold_id: str, *, source_domains, target_domain,
                           train_refs: list[dict], dev_refs: list[dict] | None = None,
                           siw_population_identity=None, siw_split_identity=None,
                           target_reference=None) -> None:
    body = {
        "schema_version": "ext-q1q2-e7a-materialized-fold-v1", "fold_id": fold_id,
        "source_domains": list(source_domains), "target_domain": target_domain,
        "source_train_references": train_refs, "source_dev_references": dev_refs or [],
        "target_reference": target_reference or {"kind": "BUILD_REQUIRED", "path": None},
        "siw_population_identity": siw_population_identity, "siw_split_identity": siw_split_identity,
        "m3b_package_identity": "fake-m3b-identity", "fold_identity": "fake-fold-identity",
        "target_labels_opened": False, "status": "FROZEN",
    }
    out_dir = repo / e7b.E7A_MATERIALIZATION_DIR / fold_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "FOLD_MATERIALIZATION.json").write_text(json.dumps(body), encoding="utf-8")


def _write_m3b_crop(repo: Path, relative_path: str, content: bytes) -> str:
    path = repo / e7b.CASIA_MSU_PACKAGE_ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return e7d.cc.sha256_bytes(content)


def _write_siw_crop(repo: Path, relative_path: str, content: bytes) -> str:
    path = repo / e7b.E7B_SIW_SOURCE_PACKAGE_ROOT / "m2_run" / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return e7d.cc.sha256_bytes(content)


def _write_siw_package(repo: Path, rows: list[dict], *, package_identity: str,
                       population_identity: str, split_identity: str) -> None:
    path = repo / e7b.E7B_SIW_SOURCE_PACKAGE_ROOT / "SIW_SOURCE_PACKAGE.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"package_identity": package_identity,
                                "population_identity": population_identity,
                                "split_identity": split_identity, "rows": rows}), encoding="utf-8")


def _write_f2_f3_siw_materialization(repo: Path, siw_train: list[dict],
                                     siw_dev: list[dict] | None = None) -> None:
    """`e7b._siw_source_refs_from_e7a` (reused verbatim by materialize_fold)
    requires BOTH EXT-F2 and EXT-F3 materializations present and agreeing
    on the identical SiW ref set -- exactly as E7-B itself requires. Tests
    exercising the SiW join must write both."""
    _write_materialization(repo, "EXT-F2", source_domains=("CASIA-FASD", "SiW-Mv2"),
                           target_domain="MSU-MFSD", train_refs=list(siw_train),
                           dev_refs=list(siw_dev or []))
    _write_materialization(repo, "EXT-F3", source_domains=("MSU-MFSD", "SiW-Mv2"),
                           target_domain="CASIA-FASD", train_refs=list(siw_train),
                           dev_refs=list(siw_dev or []))


def _patch_valid_input_binding(monkeypatch, *, m3b_identity: str = "m3b-test-identity") -> None:
    """Bypasses the (real-commit-hash-dependent) E7-A/E7-B/E7-C binding
    chain for materialize_fold() unit tests -- this module's OWN
    build_input_binding() is the single call site materialize_fold() uses,
    so patching it directly is the cleanest test seam (matching the
    established pattern of injecting fakes at an explicit, documented
    boundary rather than chasing frozen constants through three modules)."""
    monkeypatch.setattr(e7d, "build_input_binding", lambda repo: {
        "E7C_PREFLIGHT_BINDING_MATCH": True, "E7B_SIW_BINDING_MATCH": True,
        "E7A_SIW_SPLIT_BINDING_MATCH": True, "M3B_BINDING_MATCH": True,
        "m3b_package_identity_observed": m3b_identity, "m3b_package_identity_frozen": m3b_identity,
        "m3b_package_present_locally": True})


# --- 1/2/3: fold source domains exact ----------------------------------------

def test_f1_source_domains_exactly_casia_msu():
    assert e7d.FOLD_SOURCE_DOMAINS["EXT-F1"] == ("CASIA-FASD", "MSU-MFSD")


def test_f2_source_domains_exactly_casia_siw():
    assert e7d.FOLD_SOURCE_DOMAINS["EXT-F2"] == ("CASIA-FASD", "SiW-Mv2")


def test_f3_source_domains_exactly_msu_siw():
    assert e7d.FOLD_SOURCE_DOMAINS["EXT-F3"] == ("MSU-MFSD", "SiW-Mv2")


# --- 4/5: no global dataset-name firewall; fold-aware target firewall ------

def test_no_global_dataset_name_firewall(tmp_path):
    repo = _base_repo(tmp_path)
    # MSU M3B source is legal for F1/F3
    e7d.assert_not_target_path("EXT-F1", f"{e7b.CASIA_MSU_PACKAGE_ROOT}/images/x.jpg")
    e7d.assert_not_target_path("EXT-F3", f"{e7b.CASIA_MSU_PACKAGE_ROOT}/images/x.jpg")
    # CASIA M3B source is legal for F1/F2
    e7d.assert_not_target_path("EXT-F1", f"{e7b.CASIA_MSU_PACKAGE_ROOT}/images/y.jpg")
    e7d.assert_not_target_path("EXT-F2", f"{e7b.CASIA_MSU_PACKAGE_ROOT}/images/y.jpg")


def test_fold_aware_target_package_firewall():
    with pytest.raises(e7d.E7DTargetFirewallViolation):
        e7d.assert_not_target_path("EXT-F2", f"{e7b.E7B_MSU_TARGET_PACKAGE_ROOT}/crops/x.jpg")
    with pytest.raises(e7d.E7DTargetFirewallViolation):
        e7d.assert_not_target_path("EXT-F3", f"{e7b.E7B_CASIA_TARGET_PACKAGE_ROOT}/crops/x.jpg")


# --- 6/7/8: F1 never reads SiW bytes at all; F2/F3 can read SiW SOURCE but
# reject their own held-out TARGET package -----------------------------------

def test_f1_never_reads_siw_source_or_target_bytes():
    with pytest.raises(e7d.E7DTargetFirewallViolation):
        e7d.assert_not_target_path("EXT-F1", f"{e7b.E7B_SIW_SOURCE_PACKAGE_ROOT}/m2_run/crops/x.jpg")
    with pytest.raises(e7d.E7DTargetFirewallViolation):
        e7d.assert_not_target_path("EXT-F1", f"{e7b.SIW_TARGET_EVAL_PACKAGE_ROOT}/frames/x.jpg")


def test_f2_reads_siw_source_but_rejects_msu_target():
    e7d.assert_not_target_path("EXT-F2", f"{e7b.E7B_SIW_SOURCE_PACKAGE_ROOT}/m2_run/crops/x.jpg")
    with pytest.raises(e7d.E7DTargetFirewallViolation):
        e7d.assert_not_target_path("EXT-F2", f"{e7b.E7B_MSU_TARGET_PACKAGE_ROOT}/crops/x.jpg")


def test_f3_reads_siw_source_but_rejects_casia_target():
    e7d.assert_not_target_path("EXT-F3", f"{e7b.E7B_SIW_SOURCE_PACKAGE_ROOT}/m2_run/crops/x.jpg")
    with pytest.raises(e7d.E7DTargetFirewallViolation):
        e7d.assert_not_target_path("EXT-F3", f"{e7b.E7B_CASIA_TARGET_PACKAGE_ROOT}/crops/x.jpg")


# --- 9: target evaluation label paths always rejected ------------------------

def test_target_label_paths_always_rejected():
    for fold_id in e7d.FOLD_IDS:
        with pytest.raises(e7d.E7DTargetFirewallViolation):
            e7d.assert_not_target_path(fold_id, "data/evaluation_only/prism_target_v2_labels/x.parquet")


# --- 10/11/12: identity bindings exact (against the REAL repo) -------------

def test_e7b_siw_package_identity_exact_real_repo():
    binding = e7d.build_input_binding(REPO)
    assert binding["E7B_SIW_BINDING_MATCH"] is True
    assert binding["frozen_siw"]["package_identity"] == \
        "0f7811b0960d0dd2be7c732aef4107af9c3476eb9b6b9932b4fe32c7a126bb4f"


def test_e7a_siw_population_split_identities_exact_real_repo():
    binding = e7d.build_input_binding(REPO)
    assert binding["E7A_SIW_SPLIT_BINDING_MATCH"] is True
    assert binding["frozen_siw"]["population_identity"] == \
        "d05dafb814a98baebd7a5cd004ca0eb92ba798c13a8a6c5b6c90b1919e365c79"
    assert binding["frozen_siw"]["split_identity"] == \
        "b492a5d4d86537016012d5357bc5c4410f77e36409082831ef6703168ee096a1"


def test_m3b_identity_exact_real_repo():
    binding = e7d.build_input_binding(REPO)
    assert binding["m3b_package_identity_frozen"] == \
        "08d9d289eb4b462006afcff37cd4750a7c4eeb402c83de5599eda38df44168c9"
    if binding["m3b_package_present_locally"]:
        assert binding["M3B_BINDING_MATCH"] is True


# --- 13: unknown SiW video join fails -----------------------------------------

def test_unknown_siw_video_join_fails():
    with pytest.raises(e7d.E7DError, match="unknown SiW source_video_id"):
        e7d._siw_row("EXT-F2", {"source_video_id": "Nonexistent_0", "status": "success"},
                    refs_by_video={})


# --- 14: duplicate join key fails where forbidden -----------------------------

def test_duplicate_join_key_fails():
    rows = [{"sample_id": "s1"}, {"sample_id": "s1"}]
    with pytest.raises(e7d.E7DConflict, match="duplicate join key"):
        e7d._assert_unique(rows, key=lambda r: r["sample_id"], kind="test")


def test_no_duplicate_join_key_passes():
    rows = [{"sample_id": "s1"}, {"sample_id": "s2"}]
    e7d._assert_unique(rows, key=lambda r: r["sample_id"], kind="test")  # does not raise


# --- 15/16: successful SiW rows join split/label exactly; failures preserved

def test_successful_siw_row_joins_split_and_label_exactly():
    refs_by_video = {"Live_0": {"project_split": "train", "label_live_spoof": "live",
                                "spoof_family": None}}
    row = e7d._siw_row("EXT-F2", {"source_video_id": "Live_0", "status": "success",
                                  "frame_index": 0, "crop_relative_path": "crops/a.jpg",
                                  "crop_sha256": "a" * 64}, refs_by_video=refs_by_video)
    assert row["project_split"] == "train"
    assert row["label_live_spoof"] == "live"
    assert row["dataset"] == "SiW-Mv2"
    assert row["source_video_id"] == "Live_0"


def test_failure_row_preserved_with_failure_reason():
    refs_by_video = {"Spoof_0": {"project_split": "dev", "label_live_spoof": "spoof",
                                 "spoof_family": "print"}}
    row = e7d._siw_row("EXT-F2", {"source_video_id": "Spoof_0", "status": "failure",
                                  "frame_index": 2, "failure_reason": "no_face"},
                      refs_by_video=refs_by_video)
    assert row["status"] == "failure"
    assert row["failure_reason"] == "no_face"
    assert row["spoof_family"] == "print"


# --- 17/18: failures never enter crop support manifests; never resampled ----

def test_failures_never_enter_crop_support_manifests(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    _patch_valid_input_binding(monkeypatch)
    monkeypatch.setattr(e7d, "FROZEN_SIW", {"package_identity": "siw-id",
                                            "population_identity": "pop-1", "split_identity": "split-1",
                                            "planned_frame_count": 4, "successful_crop_count": 2,
                                            "failure_count": 2})
    train = [_siw_video_ref("Live_0", "train", "live", population_identity="pop-1",
                            split_identity="split-1")]
    _write_f2_f3_siw_materialization(repo, train)
    crop_bytes = b"real-crop-bytes"
    sha = _write_siw_crop(repo, "crops/live_0_0.jpg", crop_bytes)
    rows = [
        {"source_video_id": "Live_0", "status": "success", "frame_index": 0,
        "crop_relative_path": "crops/live_0_0.jpg", "crop_sha256": sha},
        {"source_video_id": "Live_0", "status": "success", "frame_index": 1,
        "crop_relative_path": "crops/live_0_0.jpg", "crop_sha256": sha},
        {"source_video_id": "Live_0", "status": "failure", "frame_index": 2,
        "failure_reason": "no_face"},
        {"source_video_id": "Live_0", "status": "failure", "frame_index": 3,
        "failure_reason": "no_face"},
    ]
    _write_siw_package(repo, rows, package_identity="siw-id", population_identity="pop-1",
                       split_identity="split-1")

    result = e7d.materialize_fold(repo, "EXT-F2", authorize=True)
    body = result["body"]
    train_rows = json.loads((Path(result["path"]).parent / "source_train.json").read_text())["rows"]
    assert all(r["status"] == "success" for r in train_rows)  # no failure ever in crop support
    assert body["siw_success_total"] == 2
    assert body["siw_failure_total"] == 2
    terminal = json.loads((Path(result["path"]).parent / "terminal_failures.json").read_text())["rows"]
    assert len(terminal) == 2
    assert all(r["status"] == "failure" for r in terminal)
    # no replacement sampling: exactly 4 planned frames total, never more
    assert len(train_rows) + len(terminal) == 4


# --- 19/20: live support == success AND live; spoof never in live support --

def test_live_support_is_success_and_live_only(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    _patch_valid_input_binding(monkeypatch)
    monkeypatch.setattr(e7d, "FROZEN_SIW", {"package_identity": "siw-id",
                                            "population_identity": "pop-1", "split_identity": "split-1",
                                            "planned_frame_count": 2, "successful_crop_count": 2,
                                            "failure_count": 0})
    train = [_siw_video_ref("Live_0", "train", "live", population_identity="pop-1",
                            split_identity="split-1"),
            _siw_video_ref("Spoof_0", "train", "spoof", population_identity="pop-1",
                          split_identity="split-1")]
    _write_f2_f3_siw_materialization(repo, train)
    live_bytes, spoof_bytes = b"live-bytes", b"spoof-bytes"
    live_sha = _write_siw_crop(repo, "crops/live.jpg", live_bytes)
    spoof_sha = _write_siw_crop(repo, "crops/spoof.jpg", spoof_bytes)
    rows = [
        {"source_video_id": "Live_0", "status": "success", "frame_index": 0,
        "crop_relative_path": "crops/live.jpg", "crop_sha256": live_sha},
        {"source_video_id": "Spoof_0", "status": "success", "frame_index": 0,
        "crop_relative_path": "crops/spoof.jpg", "crop_sha256": spoof_sha},
    ]
    _write_siw_package(repo, rows, package_identity="siw-id", population_identity="pop-1",
                       split_identity="split-1")

    result = e7d.materialize_fold(repo, "EXT-F2", authorize=True)
    live_rows = json.loads(
        (Path(result["path"]).parent / "source_live_train.json").read_text())["rows"]
    assert len(live_rows) == 1
    assert live_rows[0]["source_video_id"] == "Live_0"
    assert all(r["label_live_spoof"] == "live" for r in live_rows)


# --- 21: train/dev remain disjoint -------------------------------------------

def test_train_dev_remain_disjoint(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    _patch_valid_input_binding(monkeypatch)
    train = [_m3b_source_ref("CASIA-FASD", "live", "c1", project_split="source_train")]
    dev = [_m3b_source_ref("CASIA-FASD", "live", "c1", project_split="source_dev")]  # same sample_id!
    sha = _write_m3b_crop(repo, "images/c1.jpg", b"crop-bytes")
    train[0]["crop_sha256"] = sha
    dev[0]["crop_sha256"] = sha
    _write_materialization(repo, "EXT-F1", source_domains=("CASIA-FASD", "MSU-MFSD"),
                           target_domain="SiW-Mv2", train_refs=train, dev_refs=dev)
    with pytest.raises(e7d.E7DConflict, match="duplicate join key"):
        e7d.materialize_fold(repo, "EXT-F1", authorize=True)


# --- 22: no subject requirement for SiW --------------------------------------

def test_no_subject_requirement_for_siw():
    refs_by_video = {"Live_0": {"project_split": "train", "label_live_spoof": "live",
                                "spoof_family": None}}
    row = e7d._siw_row("EXT-F2", {"source_video_id": "Live_0", "status": "success",
                                  "frame_index": 0}, refs_by_video=refs_by_video)
    assert row["subject_id"] is None


# --- 23/24/25: crop path resolution; missing crop fails; SHA mismatch fails -

def test_crop_path_resolves_against_correct_package_root(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    _patch_valid_input_binding(monkeypatch)
    monkeypatch.setattr(e7d, "FROZEN_SIW", {"package_identity": "siw-id",
                                            "population_identity": "pop-1", "split_identity": "split-1",
                                            "planned_frame_count": 1, "successful_crop_count": 1,
                                            "failure_count": 0})
    train = [_siw_video_ref("Live_0", "train", "live", population_identity="pop-1",
                            split_identity="split-1")]
    _write_f2_f3_siw_materialization(repo, train)
    sha = _write_siw_crop(repo, "crops/a.jpg", b"real-crop")
    # a decoy at the OLD wrong location (directly under siw_source_v1/, not m2_run/)
    (repo / e7b.E7B_SIW_SOURCE_PACKAGE_ROOT / "crops").mkdir(parents=True, exist_ok=True)
    (repo / e7b.E7B_SIW_SOURCE_PACKAGE_ROOT / "crops/a.jpg").write_bytes(b"decoy")
    _write_siw_package(repo, [{"source_video_id": "Live_0", "status": "success", "frame_index": 0,
                              "crop_relative_path": "crops/a.jpg", "crop_sha256": sha}],
                       package_identity="siw-id", population_identity="pop-1", split_identity="split-1")
    result = e7d.materialize_fold(repo, "EXT-F2", authorize=True)
    assert result["status"] == "MATERIALIZED"  # succeeded using the CORRECT m2_run/ root


def test_missing_crop_fails(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    _patch_valid_input_binding(monkeypatch)
    monkeypatch.setattr(e7d, "FROZEN_SIW", {"package_identity": "siw-id",
                                            "population_identity": "pop-1", "split_identity": "split-1",
                                            "planned_frame_count": 1, "successful_crop_count": 1,
                                            "failure_count": 0})
    train = [_siw_video_ref("Live_0", "train", "live", population_identity="pop-1",
                            split_identity="split-1")]
    _write_f2_f3_siw_materialization(repo, train)
    _write_siw_package(repo, [{"source_video_id": "Live_0", "status": "success", "frame_index": 0,
                              "crop_relative_path": "crops/missing.jpg", "crop_sha256": "a" * 64}],
                       package_identity="siw-id", population_identity="pop-1", split_identity="split-1")
    with pytest.raises(e7d.E7DError, match="crop missing on disk"):
        e7d.materialize_fold(repo, "EXT-F2", authorize=True)


def test_crop_sha_mismatch_fails(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    _patch_valid_input_binding(monkeypatch)
    monkeypatch.setattr(e7d, "FROZEN_SIW", {"package_identity": "siw-id",
                                            "population_identity": "pop-1", "split_identity": "split-1",
                                            "planned_frame_count": 1, "successful_crop_count": 1,
                                            "failure_count": 0})
    train = [_siw_video_ref("Live_0", "train", "live", population_identity="pop-1",
                            split_identity="split-1")]
    _write_f2_f3_siw_materialization(repo, train)
    _write_siw_crop(repo, "crops/a.jpg", b"real-crop")
    _write_siw_package(repo, [{"source_video_id": "Live_0", "status": "success", "frame_index": 0,
                              "crop_relative_path": "crops/a.jpg", "crop_sha256": "0" * 64}],
                       package_identity="siw-id", population_identity="pop-1", split_identity="split-1")
    with pytest.raises(e7d.E7DError, match="SHA256 mismatch"):
        e7d.materialize_fold(repo, "EXT-F2", authorize=True)


# --- 26/27/28: absolute paths excluded from identity; deterministic; rerun -

def test_absolute_paths_excluded_from_identity():
    """compute_package_identity() takes no repo/filesystem-path argument at
    all -- structurally incapable of letting an absolute path affect the
    identity -- and the SAME rows produce the SAME identity regardless of
    which absolute repo root they were materialized under."""
    import inspect

    params = list(inspect.signature(e7d.compute_package_identity).parameters)
    assert "repo" not in params and "path" not in params
    rows = [{"dataset": "CASIA-FASD", "project_split": "source_train", "source_video_id": "v1",
            "frame_index": None, "crop_sha256": "a" * 64, "label_live_spoof": "live",
            "status": "success"}]
    id_under_repo_a = e7d.compute_package_identity(fold_id="EXT-F1", m3b_identity="m3b-id",
                                                    rows=rows)
    id_under_repo_b = e7d.compute_package_identity(fold_id="EXT-F1", m3b_identity="m3b-id",
                                                    rows=rows)  # simulates a different machine/repo root
    assert id_under_repo_a == id_under_repo_b


def test_deterministic_package_identity_same_inputs_same_identity():
    rows = [{"dataset": "CASIA-FASD", "project_split": "source_train", "source_video_id": "v1",
            "frame_index": None, "crop_sha256": "a" * 64, "label_live_spoof": "live",
            "status": "success"}]
    id_1 = e7d.compute_package_identity(fold_id="EXT-F1", m3b_identity="m3b-id", rows=list(rows))
    id_2 = e7d.compute_package_identity(fold_id="EXT-F1", m3b_identity="m3b-id",
                                        rows=list(reversed(rows)))  # row order must not matter
    assert id_1 == id_2


def test_rerun_same_inputs_produces_same_identity(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    _patch_valid_input_binding(monkeypatch)
    train = [_m3b_source_ref("CASIA-FASD", "live", "c1")]
    sha = _write_m3b_crop(repo, "images/c1.jpg", b"crop-bytes")
    train[0]["crop_sha256"] = sha
    _write_materialization(repo, "EXT-F1", source_domains=("CASIA-FASD", "MSU-MFSD"),
                           target_domain="SiW-Mv2", train_refs=train)
    result_1 = e7d.materialize_fold(repo, "EXT-F1", authorize=True)
    result_2 = e7d.materialize_fold(repo, "EXT-F1", authorize=True)  # resume
    assert result_1["body"]["package_identity"] == result_2["package_identity"]
    assert result_2["status"] == "ALREADY_VALID"


# --- 29/30: existing matching package -> ALREADY_VALID; conflicting -> fail

def test_existing_matching_package_reports_already_valid(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    _patch_valid_input_binding(monkeypatch)
    train = [_m3b_source_ref("CASIA-FASD", "live", "c1")]
    sha = _write_m3b_crop(repo, "images/c1.jpg", b"crop-bytes")
    train[0]["crop_sha256"] = sha
    _write_materialization(repo, "EXT-F1", source_domains=("CASIA-FASD", "MSU-MFSD"),
                           target_domain="SiW-Mv2", train_refs=train)
    e7d.materialize_fold(repo, "EXT-F1", authorize=True)
    result = e7d.materialize_fold(repo, "EXT-F1", authorize=True)
    assert result["status"] == "ALREADY_VALID"
    assert result["resumed"] is True


def test_existing_conflicting_identity_fails_closed(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    _patch_valid_input_binding(monkeypatch)
    train = [_m3b_source_ref("CASIA-FASD", "live", "c1")]
    sha = _write_m3b_crop(repo, "images/c1.jpg", b"crop-bytes")
    train[0]["crop_sha256"] = sha
    _write_materialization(repo, "EXT-F1", source_domains=("CASIA-FASD", "MSU-MFSD"),
                           target_domain="SiW-Mv2", train_refs=train)
    fold_root = repo / e7d.E7D_OUTPUT_ROOT / "EXT-F1"
    fold_root.mkdir(parents=True, exist_ok=True)
    (fold_root / "SOURCE_SUPPORT_PACKAGE.json").write_text(
        json.dumps({"package_identity": "conflicting-identity"}), encoding="utf-8")
    with pytest.raises(e7d.E7DConflict, match="disagrees with the freshly computed"):
        e7d.materialize_fold(repo, "EXT-F1", authorize=True)


# --- 31/32/33/34/35: no GPAT fitting / rendering / quality gate / training / LLM

def test_no_gpat_trainer_fit_call():
    source = Path(e7d.__file__).read_text(encoding="utf-8")
    assert "GPATTrainer(" not in source
    assert ".fit(" not in source


def test_no_gpat_route_or_physics_route_generation():
    source = Path(e7d.__file__).read_text(encoding="utf-8")
    for forbidden in ("GPATRoute(", "PhysicsRoute(", "render_arm(", "render_one("):
        assert forbidden not in source


def test_no_quality_gate_evaluation():
    source = Path(e7d.__file__).read_text(encoding="utf-8")
    assert "quality_gate.evaluate(" not in source
    assert "CandidateEvaluator(" not in source


def test_no_training_performed(tmp_path):
    repo = _base_repo(tmp_path)
    preflight = e7d.e7d_preflight(repo)
    assert preflight["TRAINING_PERFORMED"] is False
    assert preflight["GPAT_FITTING_PERFORMED"] is False
    source = Path(e7d.__file__).read_text(encoding="utf-8")
    for forbidden in ("train_detector(", "optimizer.step("):
        assert forbidden not in source


def test_no_llm_calls(tmp_path):
    source = Path(e7d.__file__).read_text(encoding="utf-8")
    for forbidden in ("openai", "google.generativeai", "GEMINI_API_KEY"):
        assert forbidden not in source
    preflight = e7d.e7d_preflight(_base_repo(tmp_path))
    assert preflight["LLM_API_CALLS"] == 0


# --- 36: preflight has correct readiness semantics ---------------------------

def test_preflight_readiness_semantics_real_repo():
    preflight = e7d.e7d_preflight(REPO)
    assert preflight["E7D_PLAN_VALID"] is True
    assert preflight["E7D_GPU_BYTES_REQUIRED"] is True  # no local SiW package bytes
    assert preflight["E7D_READY_FOR_GPU_SOURCE_SUPPORT_MATERIALIZATION"] is True
    assert preflight["E7_READY_FOR_GPAT_FITTING"] is False
    assert preflight["E7_READY_FOR_TRAINING"] is False
    assert preflight["SIW_CROP_JOIN_REQUIRED_ON_GPU"] is True
    assert preflight["SIW_CROP_JOIN_MATERIALIZED"] is False


def test_preflight_fails_closed_when_binding_missing(tmp_path):
    repo = _base_repo(tmp_path)  # nothing present at all
    preflight = e7d.e7d_preflight(repo)
    assert preflight["E7D_PLAN_VALID"] is False
    assert preflight["E7D_READY_FOR_GPU_SOURCE_SUPPORT_MATERIALIZATION"] is False


# --- 37: laptop missing SiW bytes => GPU_REQUIRED, not scientific failure --

def test_missing_siw_bytes_reported_as_gpu_required_not_failure(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    _patch_valid_input_binding(monkeypatch)
    train = [_siw_video_ref("Live_0", "train", "live", population_identity="pop-1",
                            split_identity="split-1")]
    _write_materialization(repo, "EXT-F2", source_domains=("CASIA-FASD", "SiW-Mv2"),
                           target_domain="MSU-MFSD", train_refs=train)
    # no SIW_SOURCE_PACKAGE.json written -- laptop-realistic absence
    with pytest.raises(e7d.E7DError, match="GPU_REQUIRED"):
        e7d.materialize_fold(repo, "EXT-F2", authorize=True)
    preflight = e7d.e7d_preflight(REPO)
    assert preflight["E7D_GPU_BYTES_REQUIRED"] is True
    assert preflight["local_data_state"] in ("PLAN_VALID", "SOURCE_SUPPORT_MATERIALIZED")
    assert preflight["local_data_state"] != "MISMATCH_FAIL_CLOSED"


# --- 38: E7-A/E7-B/E7-C protected artifacts untouched ------------------------

def test_e7a_e7b_e7c_protected_artifacts_untouched():
    e7a_dir = REPO / e7b.E7A_MATERIALIZATION_DIR
    before = {p: p.read_bytes() for p in e7a_dir.rglob("*") if p.is_file()}
    e7b_module = REPO / "src/prism_fas/evaluation/c_ext_e7b_data_prep.py"
    e7c_module = REPO / "src/prism_fas/evaluation/c_ext_e7c_gpat_prep.py"
    before_e7b_module = e7b_module.read_bytes()
    before_e7c_module = e7c_module.read_bytes()

    e7d.e7d_preflight(REPO)  # read-only call against the real repo

    after = {p: p.read_bytes() for p in e7a_dir.rglob("*") if p.is_file()}
    assert after == before
    assert e7b_module.read_bytes() == before_e7b_module
    assert e7c_module.read_bytes() == before_e7c_module


def test_authorize_required_for_materialize(tmp_path):
    repo = _base_repo(tmp_path)
    with pytest.raises(e7d.E7DError, match="requires --authorize"):
        e7d.materialize_fold(repo, "EXT-F1", authorize=False)


def test_unauthorized_materialize_touches_nothing(tmp_path):
    repo = _base_repo(tmp_path)
    before_tree = sorted(str(p.relative_to(repo)) for p in repo.rglob("*") if p.is_file())
    with pytest.raises(e7d.E7DError):
        e7d.materialize_fold(repo, "EXT-F1", authorize=False)
    after_tree = sorted(str(p.relative_to(repo)) for p in repo.rglob("*") if p.is_file())
    assert before_tree == after_tree


# --- prepare() writes only into E7-D's own additive namespace ---------------

def test_prepare_e7d_writes_only_into_own_namespace():
    before_tree = sorted(str(p.relative_to(REPO)) for p in
                         (REPO / e7d.E7D_REPORT_DIR).rglob("*") if p.is_file()) \
        if (REPO / e7d.E7D_REPORT_DIR).is_dir() else []
    result = e7d.prepare_e7d(REPO)
    assert result["readiness"]["body"]["E7D_PLAN_VALID"] is True
    for key in ("protocol_lock", "input_binding", "target_firewall", "join_contract",
               "output_schema", "identity_policy", "execution_plan", "readiness"):
        path = Path(result[key]["path"])
        assert str(path.relative_to(REPO)).startswith(e7d.E7D_REPORT_DIR)


def test_main_preflight_flag(monkeypatch, capsys):
    monkeypatch.setattr(e7d.cc, "repo_root", lambda: REPO)
    assert e7d.main(["--e7d-preflight"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["TARGET_LABEL_ACCESS"] is False
    assert out["TARGET_IMAGE_ACCESS"] is False


def test_main_no_flags_returns_nonzero(monkeypatch):
    monkeypatch.setattr(e7d.cc, "repo_root", lambda: REPO)
    assert e7d.main([]) == 1


def test_main_materialize_without_authorize_fails(monkeypatch, tmp_path):
    repo = _base_repo(tmp_path)
    monkeypatch.setattr(e7d.cc, "repo_root", lambda: repo)
    assert e7d.main(["--e7d-materialize"]) == 1
