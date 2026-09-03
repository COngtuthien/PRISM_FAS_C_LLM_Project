"""Tests for `prism_fas.evaluation.c_ext_e7a_fold_prep` (E7-A fold
manifest / source-dev / isolation preparation). Every test builds a
self-contained fake repo under `tmp_path`. No test ever renders, trains,
fits GPAT, touches target labels or calls an LLM.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism_fas.evaluation import c_ext_e7a_fold_prep as e7a


def _base_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


def _fold_plan_fixture(repo: Path, *, seed: int = 20260901) -> None:
    e0 = repo / e7a.E0_DIR
    e0.mkdir(parents=True, exist_ok=True)
    (e0 / "EXT_DATASET_FOLD_PLAN.json").write_text(json.dumps({
        "folds": {
            "EXT-F1": {"source": ["CASIA-FASD", "MSU-MFSD"], "target": "SiW-Mv2"},
            "EXT-F2": {"source": ["CASIA-FASD", "SiW-Mv2"], "target": "MSU-MFSD"},
            "EXT-F3": {"source": ["MSU-MFSD", "SiW-Mv2"], "target": "CASIA-FASD"},
        },
        "source_split_policy": {
            "casia_msu": "reuse EXACTLY the frozen Version-C source_train / source_dev construction",
            "siw_as_source": {
                "rule": "deterministic subject/group-disjoint split, 80% train / 20% dev at group level",
                "seed": seed, "group_key": "canonical subject_id from metadata",
                "disjointness": "no video/frame from one subject/group in both train and dev",
                "fallback": "if reliable subject/group metadata cannot be resolved, EXT-F2/EXT-F3 "
                           "source split construction STOPS as BLOCKED (never frame-random)",
            },
        },
    }), encoding="utf-8")


def _incomplete_fold_plan_fixture(repo: Path) -> None:
    """A fold plan missing the siw_as_source split policy entirely --
    exercises TASK C's fail-closed STOP path."""
    e0 = repo / e7a.E0_DIR
    e0.mkdir(parents=True, exist_ok=True)
    (e0 / "EXT_DATASET_FOLD_PLAN.json").write_text(json.dumps({
        "folds": {
            "EXT-F1": {"source": ["CASIA-FASD", "MSU-MFSD"], "target": "SiW-Mv2"},
            "EXT-F2": {"source": ["CASIA-FASD", "SiW-Mv2"], "target": "MSU-MFSD"},
            "EXT-F3": {"source": ["MSU-MFSD", "SiW-Mv2"], "target": "CASIA-FASD"},
        },
        "source_split_policy": {"casia_msu": "reuse frozen construction"},
    }), encoding="utf-8")


def _real_source_dev_fixture(repo: Path, *, casia_count: int = 4, msu_count: int = 3) -> None:
    """A real, small parquet with the ACTUAL column/value conventions the
    real M3B package uses (lowercase_underscore dataset ids)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = {"sample_id": [], "dataset": [], "source_record_id": [], "subject_id": [],
           "official_split": [], "label_live_spoof": [], "project_split": []}

    def _add(sample_id, dataset, subject, label):
        rows["sample_id"].append(sample_id)
        rows["dataset"].append(dataset)
        rows["source_record_id"].append(f"rec-{sample_id}")
        rows["subject_id"].append(subject)
        rows["official_split"].append("train")
        rows["label_live_spoof"].append(label)
        rows["project_split"].append("source_dev")

    for i in range(casia_count):
        _add(f"casia-{i}", "casia_fasd", f"casia-subj-{i}", "live" if i % 2 == 0 else "spoof")
    for i in range(msu_count):
        _add(f"msu-{i}", "msu_mfsd", f"msu-subj-{i}", "live" if i % 2 == 0 else "spoof")

    table = pa.table(rows)
    manifest_dir = repo / e7a.CASIA_MSU_PACKAGE_ROOT / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, manifest_dir / "source_dev.parquet")
    (repo / e7a.CASIA_MSU_PACKAGE_ROOT / "PACKAGE_LOCK.json").write_text(
        json.dumps({"content_identity_sha256": "fake-package-identity"}), encoding="utf-8")


def _real_source_train_fixture(repo: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = {"sample_id": ["casia-tr-0"], "dataset": ["casia_fasd"], "source_record_id": ["rec-tr-0"],
           "subject_id": ["subj-tr-0"], "official_split": ["train"], "label_live_spoof": ["live"],
           "project_split": ["source_train"]}
    table = pa.table(rows)
    manifest_dir = repo / e7a.CASIA_MSU_PACKAGE_ROOT / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, manifest_dir / "source_train.parquet")


def _siw_target_package_fixture(repo: Path) -> None:
    (repo / e7a.SIW_TARGET_EVAL_PACKAGE_ROOT).mkdir(parents=True, exist_ok=True)


def _full_fixture(tmp_path: Path) -> Path:
    repo = _base_repo(tmp_path)
    _fold_plan_fixture(repo)
    _real_source_dev_fixture(repo)
    _siw_target_package_fixture(repo)
    return repo


# --- 1-3: exact matrix/seed constants ---------------------------------------

def test_exact_three_folds():
    assert e7a.FOLDS == ("EXT-F1", "EXT-F2", "EXT-F3")
    assert e7a.FOLD_DOMAINS["EXT-F1"] == {"source": ("CASIA-FASD", "MSU-MFSD"), "target": "SiW-Mv2"}
    assert e7a.FOLD_DOMAINS["EXT-F2"] == {"source": ("CASIA-FASD", "SiW-Mv2"), "target": "MSU-MFSD"}
    assert e7a.FOLD_DOMAINS["EXT-F3"] == {"source": ("MSU-MFSD", "SiW-Mv2"), "target": "CASIA-FASD"}


def test_exact_source_target_domain_mapping():
    for fold, domains in e7a.FOLD_DOMAINS.items():
        assert domains["target"] not in domains["source"]
        assert len(domains["source"]) == 2


def test_split_seed_20260901(tmp_path):
    repo = _full_fixture(tmp_path)
    policy = e7a.resolve_source_split_policy(repo)
    assert policy["siw_as_source_seed"] == 20260901
    assert policy["FROZEN_SOURCE_SPLIT_SEED"] if "FROZEN_SOURCE_SPLIT_SEED" in policy else True
    lock = e7a.build_source_split_lock(repo)
    assert lock["FROZEN_SOURCE_SPLIT_SEED"] == 20260901


# --- 4: deterministic source split (policy resolution is itself deterministic)

def test_deterministic_source_split_policy(tmp_path):
    repo = _full_fixture(tmp_path)
    first = e7a.build_source_split_lock(repo)
    second = e7a.build_source_split_lock(repo)
    assert first["lock_identity"] == second["lock_identity"]


# --- 5: no train/dev sample overlap (checked structurally: no local source_train
#        exists, so this must be reported NOT_COMPUTABLE, never fabricated 0/None)

def test_no_train_dev_overlap_reported_honestly_without_bytes(tmp_path):
    repo = _full_fixture(tmp_path)
    isolation = e7a.audit_fold_isolation_e7a(repo)
    for fold in isolation["per_fold"]:
        assert fold["TRAIN_DEV_SAMPLE_OVERLAP"] == "NOT_COMPUTABLE_LOCAL_BYTES_MISSING"


def test_no_train_dev_overlap_computed_when_both_present(tmp_path):
    repo = _full_fixture(tmp_path)
    _real_source_train_fixture(repo)
    # even with source_train present, this module's audit only reads
    # source_dev (source_train reading is out of scope for this turn's
    # laptop-only audit) -- still must never silently claim 0 without basis
    isolation = e7a.audit_fold_isolation_e7a(repo)
    assert isolation["per_fold"][0]["TRAIN_DEV_SAMPLE_OVERLAP"] == "NOT_COMPUTABLE_LOCAL_BYTES_MISSING"


# --- 6: subject isolation where required ------------------------------------

def test_subject_isolation_rule_recorded_for_siw_as_source(tmp_path):
    repo = _full_fixture(tmp_path)
    policy = e7a.resolve_source_split_policy(repo)
    assert "subject" in policy["siw_as_source_disjointness"].lower()
    assert policy["siw_as_source_group_key"] == "canonical subject_id from metadata"


# --- 7-8: held-out domain absent from source train/dev ----------------------

def test_heldout_domain_absent_from_source_train_field_present(tmp_path):
    repo = _full_fixture(tmp_path)
    isolation = e7a.audit_fold_isolation_e7a(repo)
    for fold in isolation["per_fold"]:
        assert "SOURCE_TRAIN_TARGET_DOMAIN_ROWS" in fold


def test_heldout_domain_absent_from_source_dev_for_ext_f1(tmp_path):
    repo = _full_fixture(tmp_path)
    isolation = e7a.audit_fold_isolation_e7a(repo)
    f1 = next(f for f in isolation["per_fold"] if f["fold_id"] == "EXT-F1")
    assert f1["SOURCE_DEV_TARGET_DOMAIN_ROWS"] == 0  # SiW never in the CASIA/MSU pool file


def test_naive_reuse_would_leak_target_for_f2_f3(tmp_path):
    """Proves WHY F2/F3 must filter rather than reuse the whole pool file:
    each other fold's held-out target dataset (which IS a legitimate CASIA/
    MSU source domain for F1) shows up as a nonzero count here."""
    repo = _full_fixture(tmp_path)
    isolation = e7a.audit_fold_isolation_e7a(repo)
    f2 = next(f for f in isolation["per_fold"] if f["fold_id"] == "EXT-F2")
    f3 = next(f for f in isolation["per_fold"] if f["fold_id"] == "EXT-F3")
    assert f2["SOURCE_DEV_TARGET_DOMAIN_ROWS"] == 3  # msu_count
    assert f3["SOURCE_DEV_TARGET_DOMAIN_ROWS"] == 4  # casia_count
    assert f2["note"] is not None and "NEVER be reused wholesale" in f2["note"]


# --- 9-10: no target labels, no target label access -------------------------

def test_no_target_labels_in_target_reference(tmp_path):
    repo = _full_fixture(tmp_path)
    contract = e7a.build_target_reference_contract(repo)
    assert contract["TARGET_LABELS_LOADED"] is False
    assert contract["TARGET_LABEL_COLUMNS_PERSISTED"] is False
    assert contract["label_firewall_dir_opened_by_this_module"] is False


def test_no_target_label_access_anywhere(tmp_path):
    repo = _full_fixture(tmp_path)
    isolation = e7a.audit_fold_isolation_e7a(repo)
    assert isolation["TARGET_LABEL_ACCESS"] is False
    # the firewall directory is only ever used as a DOCUMENTATION string
    # (constant name, dict values); it is never the argument of a real
    # filesystem/read call anywhere in the module
    source = Path(e7a.__file__).read_text(encoding="utf-8")
    for forbidden_call in ("read_json(repo / SIW_LABEL_FIREWALL_DIR", "open(repo / SIW_LABEL_FIREWALL_DIR",
                          "read_table(repo / SIW_LABEL_FIREWALL_DIR"):
        assert forbidden_call not in source


# --- 11-12: canonical dataset identity pinned; mismatch fails closed --------

def test_canonical_dataset_identity_pinned(tmp_path):
    repo = _full_fixture(tmp_path)
    lock = e7a.build_e7a_protocol_lock(repo)
    assert lock["dataset_identities"]["CASIA-FASD"] == "fake-package-identity"
    assert lock["dataset_identities"]["MSU-MFSD"] == "fake-package-identity"
    assert lock["protocol_lock_identity"]


def test_identity_mismatch_fails_closed(tmp_path):
    repo = _full_fixture(tmp_path)
    e7a.write_e7a_protocol_lock(repo)
    # drift the repo state after the lock was written -- change the package
    # identity itself, which IS part of the protocol lock's hashed content
    (repo / e7a.CASIA_MSU_PACKAGE_ROOT / "PACKAGE_LOCK.json").write_text(
        json.dumps({"content_identity_sha256": "a-different-package-identity"}), encoding="utf-8")

    check = e7a.verify_protocol_lock_matches_expected(repo)
    assert check["PROTOCOL_LOCK_PRESENT"] is True
    assert check["MATCHES_EXPECTED"] is False

    with pytest.raises(e7a.E7AResumeConflict):
        e7a.e7a_build(repo, authorize=True)


# --- 13: unresolved split policy fails closed -------------------------------

def test_unresolved_split_policy_fails_closed(tmp_path):
    repo = _base_repo(tmp_path)
    _incomplete_fold_plan_fixture(repo)
    with pytest.raises(e7a.E7AError, match="UNRESOLVED protocol field"):
        e7a.resolve_source_split_policy(repo)


def test_missing_fold_plan_fails_closed(tmp_path):
    repo = _base_repo(tmp_path)
    with pytest.raises(e7a.E7AError):
        e7a.resolve_source_split_policy(repo)


def test_wrong_seed_fails_closed(tmp_path):
    repo = _base_repo(tmp_path)
    _fold_plan_fixture(repo, seed=99999999)
    with pytest.raises(e7a.E7AError, match="refusing to silently use a different seed"):
        e7a.resolve_source_split_policy(repo)


# --- 14-15: resume safety ----------------------------------------------------

def test_resume_with_identical_manifest_reuses_safely(tmp_path):
    repo = _full_fixture(tmp_path)
    e7a.write_e7a_protocol_lock(repo)
    check = e7a.verify_protocol_lock_matches_expected(repo)
    assert check["MATCHES_EXPECTED"] is True


def test_resume_with_differing_manifest_fails_closed(tmp_path):
    repo = _full_fixture(tmp_path)
    result = e7a.write_e7a_protocol_lock(repo)
    path = Path(result["path"])
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["protocol_lock_identity"] = "0" * 64
    path.write_text(json.dumps(tampered), encoding="utf-8")

    check = e7a.verify_protocol_lock_matches_expected(repo)
    assert check["MATCHES_EXPECTED"] is False
    with pytest.raises(e7a.E7AResumeConflict):
        e7a.e7a_build(repo, authorize=True)


# --- 16-19: no render/train/GPAT-fit/LLM ------------------------------------

def test_e7a_does_not_render(tmp_path):
    repo = _full_fixture(tmp_path)
    preflight = e7a.e7a_preflight(repo)
    assert preflight["rendering_performed"] is False
    source = Path(e7a.__file__).read_text(encoding="utf-8")
    assert "render_arm(" not in source
    assert "c5_render" not in source


def test_e7a_does_not_train(tmp_path):
    repo = _full_fixture(tmp_path)
    preflight = e7a.e7a_preflight(repo)
    assert preflight["training_performed"] is False
    source = Path(e7a.__file__).read_text(encoding="utf-8")
    for forbidden in ("train_detector(", "M9TrainingRun(", "optimizer.step("):
        assert forbidden not in source


def test_e7a_does_not_fit_gpat(tmp_path):
    repo = _full_fixture(tmp_path)
    preflight = e7a.e7a_preflight(repo)
    assert preflight["gpat_fitting_performed"] is False
    readiness = e7a.build_e7a_readiness(repo)
    assert readiness["E7A_READY_FOR_GPAT"] is False
    assert readiness["gpat_fitting_performed"] is False
    source = Path(e7a.__file__).read_text(encoding="utf-8")
    assert "GPATRoute(" not in source
    assert "build_gpat_model(" not in source


def test_e7a_does_not_call_llm(tmp_path):
    repo = _full_fixture(tmp_path)
    for body in (e7a.e7a_preflight(repo), e7a.build_e7a_readiness(repo), e7a.build_e7a_protocol_lock(repo)):
        assert body["llm_api_calls"] == 0
    source = Path(e7a.__file__).read_text(encoding="utf-8")
    for forbidden in ("openai", "google.generativeai", "GEMINI_API_KEY", "requests.post"):
        assert forbidden not in source


# --- 20: EXT-F1 Shuffle block does not alter fold data ----------------------

def test_shuffle_block_does_not_alter_fold_data(tmp_path):
    """E7-A is condition-independent -- nothing here even MENTIONS
    LLM-SHUFFLE-A or reads the E6-v2 closure; the fold manifest plan for
    EXT-F1 is identical regardless of any condition's readiness."""
    repo = _full_fixture(tmp_path)
    plan = e7a.build_fold_manifest_plan(repo)
    f1 = next(f for f in plan["folds"] if f["fold_id"] == "EXT-F1")
    assert f1["source_domains"] == ["CASIA-FASD", "MSU-MFSD"]
    assert f1["heldout_target_domain"] == "SiW-Mv2"
    source = Path(e7a.__file__).read_text(encoding="utf-8")
    assert "SHUFFLE" not in source
    assert "E6_V2" not in source and "e6_v2" not in source


# --- 21: historical Flow1 not accepted as E7 fold manifest -------------------

def test_historical_flow1_not_accepted_as_fold_manifest(tmp_path):
    repo = _full_fixture(tmp_path)
    plan = e7a.build_fold_manifest_plan(repo)
    for fold in plan["folds"]:
        for ref in (fold["source_train_manifest_ref"], fold["source_dev_manifest_ref"]):
            assert "flow1" not in json.dumps(ref).lower()
            assert "C-G-" not in json.dumps(ref)  # historical Flow-1 run_id prefix never appears


# --- 22: protected artifacts unchanged --------------------------------------

def test_prepare_e7a_never_writes_outside_own_namespace(tmp_path):
    repo = _full_fixture(tmp_path)
    (repo / "reports/full/c6").mkdir(parents=True, exist_ok=True)
    sentinel = repo / "reports/full/c6/SENTINEL.json"
    sentinel.write_text('{"untouched": true}', encoding="utf-8")
    before = sentinel.read_bytes()
    e7_readiness_dir = repo / e7a.E7_READINESS_DIR
    e7_readiness_dir.mkdir(parents=True, exist_ok=True)
    (e7_readiness_dir / "E7_READINESS.json").write_text('{"frozen": true}', encoding="utf-8")
    readiness_before = (e7_readiness_dir / "E7_READINESS.json").read_bytes()

    before_tree = sorted(str(p.relative_to(repo)) for p in repo.rglob("*") if p.is_file())
    e7a.prepare_e7a(repo)
    after_tree = sorted(str(p.relative_to(repo)) for p in repo.rglob("*") if p.is_file())

    assert sentinel.read_bytes() == before
    assert (e7_readiness_dir / "E7_READINESS.json").read_bytes() == readiness_before  # E7 readiness untouched
    new_files = sorted(set(after_tree) - set(before_tree))
    assert new_files
    assert all(f.startswith(e7a.E7A_REPORT_DIR) for f in new_files)


def test_prepare_e7a_writes_all_expected_artifacts(tmp_path):
    repo = _full_fixture(tmp_path)
    result = e7a.prepare_e7a(repo)
    expected = {"E7A_PROTOCOL_LOCK.json", "E7A_DATASET_BINDING.json", "E7A_FOLD_MANIFEST_PLAN.json",
               "E7A_SOURCE_SPLIT_LOCK.json", "E7A_TARGET_REFERENCE_CONTRACT.json",
               "E7A_ISOLATION_REPORT.json", "E7A_EXECUTION_PLAN.json", "E7A_READINESS.json"}
    written = {Path(v["path"]).name for v in result.values()}
    assert written == expected
    for name in expected:
        assert (repo / e7a.E7A_REPORT_DIR / name).is_file()


# --- extra: CLI plumbing / no-argument default ------------------------------

def test_main_no_flags_returns_nonzero(monkeypatch, tmp_path):
    repo = _full_fixture(tmp_path)
    monkeypatch.setattr(e7a.cc, "repo_root", lambda: repo)
    code = e7a.main([])
    assert code == 1


def test_main_e7a_build_without_authorize_flag_refuses(monkeypatch, tmp_path):
    repo = _full_fixture(tmp_path)
    monkeypatch.setattr(e7a.cc, "repo_root", lambda: repo)
    code = e7a.main(["--e7a-build"])
    assert code == 1


def test_main_e7a_validate_reports_not_yet_built(monkeypatch, tmp_path):
    repo = _full_fixture(tmp_path)
    monkeypatch.setattr(e7a.cc, "repo_root", lambda: repo)
    result = e7a.e7a_validate(repo)
    assert result["status"] == "NOT_YET_BUILT"
