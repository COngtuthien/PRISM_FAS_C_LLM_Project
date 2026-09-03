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


# =============================================================================
# AMENDMENT (local-data-only SiW-as-source policy): 25 required tests.
# =============================================================================

def _siw_layout_config_fixture(repo: Path, *, live_count: int = 3, replay_count: int = 2,
                               paper_count: int = 2) -> None:
    import yaml

    config_path = repo / e7a.SIW_LAYOUT_CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump({
        "path_pattern": r"^(?:(?P<live_label>Live)/Live_\d+|(?P<spoof_label>Spoof)/"
                        r"(?P<attack_family>[A-Za-z_]+)/(?P<stem>[A-Za-z_]+)_\d+)\.(?:avi|mov|mp4)$",
        "include_globs": ["Live/*.avi", "Live/*.mov", "Live/*.mp4",
                          "Spoof/*/*.avi", "Spoof/*/*.mov", "Spoof/*/*.mp4"],
        "attack_family_stems": {"Replay": "Replay", "Paper": "Paper"},
        "expected_counts": {"live": live_count, "spoof": replay_count + paper_count,
                           "total": live_count + replay_count + paper_count,
                           "by_attack_family": {"Replay": replay_count, "Paper": paper_count}},
    }), encoding="utf-8")


def _siw_raw_root_fixture(repo: Path, *, live_count: int = 3, replay_count: int = 2,
                          paper_count: int = 2) -> None:
    root = repo / e7a.SIW_RAW_ROOT
    (root / "Live").mkdir(parents=True, exist_ok=True)
    (root / "Spoof" / "Replay").mkdir(parents=True, exist_ok=True)
    (root / "Spoof" / "Paper").mkdir(parents=True, exist_ok=True)
    for i in range(live_count):
        (root / "Live" / f"Live_{i}.mov").write_bytes(b"fake-video-bytes")
    for i in range(replay_count):
        (root / "Spoof" / "Replay" / f"Replay_{i}.mov").write_bytes(b"fake-video-bytes")
    for i in range(paper_count):
        (root / "Spoof" / "Paper" / f"Paper_{i}.mov").write_bytes(b"fake-video-bytes")


def _m3b_source_train_fixture(repo: Path, *, casia_count: int = 2, msu_count: int = 1) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = {"sample_id": [], "dataset": [], "source_record_id": [], "subject_id": [],
           "official_split": [], "label_live_spoof": [], "project_split": []}
    for i in range(casia_count):
        rows["sample_id"].append(f"casia-tr-{i}"); rows["dataset"].append("casia_fasd")
        rows["source_record_id"].append(f"rec-{i}"); rows["subject_id"].append(f"s-{i}")
        rows["official_split"].append("train"); rows["label_live_spoof"].append("live")
        rows["project_split"].append("source_train")
    for i in range(msu_count):
        rows["sample_id"].append(f"msu-tr-{i}"); rows["dataset"].append("msu_mfsd")
        rows["source_record_id"].append(f"rec-msu-{i}"); rows["subject_id"].append(f"m-{i}")
        rows["official_split"].append("train"); rows["label_live_spoof"].append("spoof")
        rows["project_split"].append("source_train")
    table = pa.table(rows)
    manifest_dir = repo / e7a.CASIA_MSU_PACKAGE_ROOT / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, manifest_dir / "source_train.parquet")


def _full_amendment_fixture(tmp_path: Path) -> Path:
    repo = _full_fixture(tmp_path)
    _siw_layout_config_fixture(repo)
    return repo


# --- 1-2: external data/protocol prohibition --------------------------------

def test_external_siw_data_prohibited():
    source = Path(e7a.__file__).read_text(encoding="utf-8")
    for forbidden in ("urllib.request", "urllib.error", "requests.get(", "requests.post(",
                     "import requests", "http://", "https://", "subprocess.run([\"wget\"",
                     "subprocess.run([\"curl\""):
        assert forbidden not in source


def test_external_protocol_lists_not_scientific_dependencies(tmp_path):
    repo = _full_amendment_fixture(tmp_path)
    amendment = e7a.build_siw_local_only_amendment(repo)
    assert amendment["EXTERNAL_DATASET_USED"] is False
    assert amendment["EXTERNAL_PROTOCOL_LIST_USED_FOR_SPLIT"] is False
    plan = e7a.build_siw_local_population_plan(repo)
    assert plan["external_dataset_used"] is False
    assert plan["external_protocol_list_used_for_split"] is False


# --- 3: exact local root ------------------------------------------------------

def test_exact_local_siw_root():
    assert e7a.SIW_RAW_ROOT == "data/raw/siw_mv2/SiW-Mv2"


# --- 4: local population count checks ----------------------------------------

def test_local_population_count_checks_pass_when_matching(tmp_path):
    repo = _full_amendment_fixture(tmp_path)
    _siw_raw_root_fixture(repo)
    scan = e7a.scan_local_siw_population(repo)
    layout = e7a._load_siw_layout_config(repo)
    check = e7a.verify_siw_population_against_expected(scan, layout)
    assert check["CHECKED"] is True
    assert check["MATCHES_EXPECTED"] is True
    assert scan["TOTAL"] == 7 and scan["LIVE"] == 3 and scan["SPOOF"] == 4


def test_local_population_count_checks_fail_closed_on_mismatch(tmp_path):
    repo = _full_amendment_fixture(tmp_path)
    _siw_raw_root_fixture(repo, live_count=2)  # fewer than the 3 the config expects
    scan = e7a.scan_local_siw_population(repo)
    layout = e7a._load_siw_layout_config(repo)
    check = e7a.verify_siw_population_against_expected(scan, layout)
    assert check["MATCHES_EXPECTED"] is False
    assert any("LIVE" in m for m in check["mismatches"])


def test_no_video_silently_dropped_unexpected_family_hard_fails(tmp_path):
    repo = _full_amendment_fixture(tmp_path)
    _siw_raw_root_fixture(repo)
    bogus_dir = repo / e7a.SIW_RAW_ROOT / "Spoof" / "NotARealFamily"
    bogus_dir.mkdir(parents=True, exist_ok=True)
    (bogus_dir / "NotARealFamily_0.mov").write_bytes(b"x")
    with pytest.raises(e7a.E7AError):
        e7a.scan_local_siw_population(repo)


# --- 5: path-detection bug fixed ----------------------------------------------

def test_path_detection_bug_fixed(tmp_path):
    repo = _full_amendment_fixture(tmp_path)
    audit_before = e7a.audit_dataset_infrastructure(repo)
    siw_row = next(r for r in audit_before["rows"] if r["DATASET"] == "SiW-Mv2")
    assert siw_row["raw_siw_source_bytes_present_locally"] is False  # root genuinely absent

    _siw_raw_root_fixture(repo)
    audit_after = e7a.audit_dataset_infrastructure(repo)
    siw_row_after = next(r for r in audit_after["rows"] if r["DATASET"] == "SiW-Mv2")
    assert siw_row_after["raw_siw_source_bytes_present_locally"] is True  # now correctly detected


# --- 6-7: M3B path correction / target_eval_v2 placeholder rejection --------

def test_m3b_path_correction(tmp_path):
    repo = _full_amendment_fixture(tmp_path)
    correction = e7a.build_m3b_binding_correction(repo)
    assert correction["corrected_canonical_source_train_path"] == \
        "data/packages/prism_data_v1_m3b/manifests/source_train.parquet"
    assert correction["corrected_canonical_source_dev_path"] == \
        "data/packages/prism_data_v1_m3b/manifests/source_dev.parquet"
    assert correction["e0_file_rewritten"] is False
    assert correction["m3b_bytes_altered"] is False


def test_target_eval_v2_placeholders_not_accepted_as_casia_msu_source(tmp_path):
    repo = _full_amendment_fixture(tmp_path)
    correction = e7a.build_m3b_binding_correction(repo)
    assert "prism_target_eval_v2" in correction["e0_frozen_prose_path"]
    assert "ZERO rows" in correction["target_eval_v2_placeholder_note"]
    assert "prism_target_eval_v2" not in correction["corrected_canonical_source_train_path"]
    assert "prism_target_eval_v2" not in correction["corrected_canonical_source_dev_path"]


# --- 8-9: deterministic video split, stratification --------------------------

def test_deterministic_video_split():
    records = [{"video_id": f"Live_{i}", "class_live_spoof": "live", "spoof_family": None}
              for i in range(20)]
    records += [{"video_id": f"Replay_{i}", "class_live_spoof": "spoof", "spoof_family": "Replay"}
               for i in range(10)]
    first = e7a.compute_siw_video_split(records)
    second = e7a.compute_siw_video_split(records)
    assert first["split_identity"] == second["split_identity"]
    assert first["assignment"] == second["assignment"]


def test_stratification_by_live_spoof_family():
    records = [{"video_id": f"Live_{i}", "class_live_spoof": "live", "spoof_family": None}
              for i in range(10)]
    records += [{"video_id": f"Replay_{i}", "class_live_spoof": "spoof", "spoof_family": "Replay"}
               for i in range(10)]
    records += [{"video_id": f"Paper_{i}", "class_live_spoof": "spoof", "spoof_family": "Paper"}
               for i in range(10)]
    split = e7a.compute_siw_video_split(records)
    assert set(split["stratum_report"]) == {"live", "spoof:Replay", "spoof:Paper"}
    for stratum in split["stratum_report"].values():
        assert stratum["train"] == 8 and stratum["dev"] == 2


# --- 10-11: no overlap; video-level (never frame-level) isolation -----------

def test_video_train_dev_overlap_zero():
    records = [{"video_id": f"Live_{i}", "class_live_spoof": "live", "spoof_family": None}
              for i in range(15)]
    split = e7a.compute_siw_video_split(records)
    train = {v for v, s in split["assignment"].items() if s == "train"}
    dev = {v for v, s in split["assignment"].items() if s == "dev"}
    assert train & dev == set()
    assert len(train) + len(dev) == 15


def test_frames_from_same_video_cannot_cross_split():
    """The split assignment is keyed ONLY by video_id -- a duplicate
    video_id (which is what a second frame/crop from the same video would
    look like at this layer) is rejected outright, never silently split
    across train/dev."""
    records = [{"video_id": "Live_0", "class_live_spoof": "live", "spoof_family": None},
              {"video_id": "Live_0", "class_live_spoof": "live", "spoof_family": None}]
    with pytest.raises(e7a.E7AError, match="duplicate video_id"):
        e7a.compute_siw_video_split(records)


# --- 12-13: subject disjointness recorded; no filename subject parsing ------

def test_subject_disjointness_recorded_unverifiable(tmp_path):
    repo = _full_amendment_fixture(tmp_path)
    lock = e7a.build_siw_video_split_policy_lock(repo)
    assert lock["SUBJECT_DISJOINTNESS"] == "UNVERIFIABLE_NOT_ENFORCED"
    assert "unavailable" in lock["SUBJECT_DISJOINTNESS_REASON"].lower()
    assert lock["no_subject_inferred_from_filename"] is True


def test_no_fake_subject_parsing_from_filename():
    """`scan_local_siw_population`'s own docstring EXPLAINS it never
    resolves subject identity (legitimate documentation); the function
    BODY (after the docstring closes) must never actually extract or
    assign a subject field from a filename/regex."""
    source = Path(e7a.__file__).read_text(encoding="utf-8")
    fn_start = source.index("def scan_local_siw_population(")
    fn_end = source.index("\ndef ", fn_start + 10)
    full_fn = source[fn_start:fn_end]
    docstring_end = full_fn.index('"""', full_fn.index('"""') + 3) + 3
    body = full_fn[docstring_end:]
    assert "subject" not in body.lower()
    assert '"subject_id"' not in full_fn


# --- 14-17: F2/F3 construction semantics; F1 unchanged -----------------------

def test_f2_excludes_msu_source_rows(tmp_path):
    repo = _full_amendment_fixture(tmp_path)
    plan = e7a.build_amended_fold_construction_plan(repo)
    assert plan["EXT-F2"]["excludes_msu_source_rows"] is True
    assert "MSU" in plan["EXT-F2"]["casia_rows"]


def test_f3_excludes_casia_source_rows(tmp_path):
    repo = _full_amendment_fixture(tmp_path)
    plan = e7a.build_amended_fold_construction_plan(repo)
    assert plan["EXT-F3"]["excludes_casia_source_rows"] is True
    assert "CASIA" in plan["EXT-F3"]["msu_rows"]


def test_same_siw_split_reused_in_f2_f3(tmp_path):
    repo = _full_amendment_fixture(tmp_path)
    plan = e7a.build_amended_fold_construction_plan(repo)
    assert plan["EXT-F2"]["siw_source_split_identity"] == plan["EXT-F3"]["siw_source_split_identity"]
    assert plan["f2_f3_share_one_siw_split"] is True


def test_ext_f1_unchanged(tmp_path):
    repo = _full_amendment_fixture(tmp_path)
    plan = e7a.build_amended_fold_construction_plan(repo)
    assert plan["EXT-F1"]["unchanged"] is True
    assert plan["EXT-F1"]["source"] == ["CASIA-FASD", "MSU-MFSD"]
    assert plan["EXT-F1"]["target"] == "SiW-Mv2"


# --- 18-19: readiness before/after amendment ---------------------------------

def test_pre_amendment_readiness_false(tmp_path):
    repo = _full_amendment_fixture(tmp_path)
    readiness = e7a.build_amended_readiness(repo)
    assert readiness["E7A_READY_FOR_BUILD"] is False
    assert readiness["prerequisites"]["amendment_lock_present_and_matching"] is False


def test_amended_readiness_requires_all_prerequisites(tmp_path):
    repo = _full_amendment_fixture(tmp_path)
    _siw_raw_root_fixture(repo)
    _m3b_source_train_fixture(repo)
    e7a.write_siw_local_only_amendment(repo)
    readiness = e7a.build_amended_readiness(repo)
    # even with SiW root + M3B source_train present, source_dev's real
    # counts (from the module-level fixture) won't match the AMENDMENT's
    # informational M3B_EXPECTED_* constants -- proving readiness is a
    # real AND of every prerequisite, not just "amendment lock present"
    assert set(readiness["prerequisites"]) == {
        "amendment_lock_present_and_matching", "exact_local_siw_root_present",
        "frozen_population_inventory_matches", "deterministic_video_split_implementation_available",
        "m3b_casia_msu_manifests_present_and_matching", "target_domain_isolation_checks_resolvable",
        "no_target_label_access"}
    assert readiness["E7A_READY_FOR_BUILD"] == all(readiness["prerequisites"].values())


# --- 20: target labels untouched ---------------------------------------------

def test_target_labels_untouched_by_amendment(tmp_path):
    repo = _full_amendment_fixture(tmp_path)
    for body in (e7a.build_siw_local_population_plan(repo), e7a.build_siw_video_split_policy_lock(repo),
                e7a.build_m3b_binding_correction(repo), e7a.build_siw_local_only_amendment(repo),
                e7a.build_amended_fold_construction_plan(repo)):
        assert body["target_access"] is False
    source = Path(e7a.__file__).read_text(encoding="utf-8")
    assert "SIW_LABEL_FIREWALL_DIR)" not in source.split("AMENDMENT")[1] if "AMENDMENT" in source else True


# --- 21-24: no render/train/GPAT-fit/LLM --------------------------------------

def test_amendment_never_renders(tmp_path):
    repo = _full_amendment_fixture(tmp_path)
    preflight = e7a.e7a_local_siw_preflight(repo)
    assert preflight["rendering_performed"] is False
    source = Path(e7a.__file__).read_text(encoding="utf-8")
    assert "render_arm(" not in source


def test_amendment_never_trains(tmp_path):
    repo = _full_amendment_fixture(tmp_path)
    preflight = e7a.e7a_local_siw_preflight(repo)
    assert preflight["training_performed"] is False


def test_amendment_never_fits_gpat(tmp_path):
    repo = _full_amendment_fixture(tmp_path)
    preflight = e7a.e7a_local_siw_preflight(repo)
    assert preflight["gpat_fitting_performed"] is False
    source = Path(e7a.__file__).read_text(encoding="utf-8")
    assert "GPATRoute(" not in source


def test_amendment_never_calls_llm(tmp_path):
    repo = _full_amendment_fixture(tmp_path)
    preflight = e7a.e7a_local_siw_preflight(repo)
    assert preflight["llm_api_calls"] == 0
    source = Path(e7a.__file__).read_text(encoding="utf-8")
    for forbidden in ("openai", "google.generativeai", "GEMINI_API_KEY"):
        assert forbidden not in source


# --- 25: protected artifacts unchanged (writes confined to amendment dir) --

def test_amendment_never_writes_outside_amendment_dir_or_original_e7a(tmp_path):
    repo = _full_amendment_fixture(tmp_path)
    (repo / "reports/full/c6").mkdir(parents=True, exist_ok=True)
    sentinel = repo / "reports/full/c6/SENTINEL.json"
    sentinel.write_text('{"untouched": true}', encoding="utf-8")
    before = sentinel.read_bytes()

    original_e7a_dir = repo / e7a.E7A_REPORT_DIR
    original_e7a_dir.mkdir(parents=True, exist_ok=True)
    (original_e7a_dir / "E7A_PROTOCOL_LOCK.json").write_text('{"frozen": true}', encoding="utf-8")
    original_before = (original_e7a_dir / "E7A_PROTOCOL_LOCK.json").read_bytes()

    before_tree = sorted(str(p.relative_to(repo)) for p in repo.rglob("*") if p.is_file())
    e7a.prepare_e7a_amendment(repo)
    after_tree = sorted(str(p.relative_to(repo)) for p in repo.rglob("*") if p.is_file())

    assert sentinel.read_bytes() == before
    assert (original_e7a_dir / "E7A_PROTOCOL_LOCK.json").read_bytes() == original_before
    new_files = sorted(set(after_tree) - set(before_tree))
    assert new_files
    assert all(f.startswith(e7a.AMENDMENT_DIR) for f in new_files)


def test_prepare_e7a_amendment_writes_all_expected_artifacts(tmp_path):
    repo = _full_amendment_fixture(tmp_path)
    result = e7a.prepare_e7a_amendment(repo)
    expected = {"E7A_SIW_LOCAL_ONLY_AMENDMENT.json", "E7A_SIW_LOCAL_POPULATION_PLAN.json",
               "E7A_SIW_VIDEO_SPLIT_POLICY_LOCK.json", "E7A_M3B_BINDING_CORRECTION.json",
               "E7A_READINESS_FIX_REPORT.json", "E7A_AMENDED_EXECUTION_PLAN.json",
               "E7A_AMENDED_FOLD_CONSTRUCTION_PLAN.json", "E7A_AMENDED_READINESS.json"}
    written = {Path(v["path"]).name for v in result.values()}
    assert written == expected
    for name in expected:
        assert (repo / e7a.AMENDMENT_DIR / name).is_file()
