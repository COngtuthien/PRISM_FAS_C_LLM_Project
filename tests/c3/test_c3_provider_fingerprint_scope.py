"""The provider-evidence fingerprint must move only when provider evidence moves.

`scripts/c3_pre_live_audit.py` originally fingerprinted `reports/c1`..`reports/c3`
wholesale. The audit writes its own artifacts into `reports/c3/v15_pre_live_audit/`,
so each run hashed the previous run's output and produced a different fingerprint
with no provider call anywhere in between. A number that always changes cannot
witness "nothing called the provider", and a reader could reasonably misread the
change as evidence that something did.

These tests pin the corrected scope. They are offline: the shared `no_network`
fixture in conftest blocks sockets, and nothing here imports a provider SDK.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import c3_pre_live_audit as audit  # noqa: E402


# --------------------------------------------------------------- synthetic repo
def build_repo(root: Path) -> None:
    """A miniature repo carrying one artifact of each relevant kind."""
    for rel, payload in {
        "reports/c2/C2_PILOT_RAW_ARCHIVE.json": {"records": [{"id": 1}] * 42},
        "reports/c2/C2_SMOKE_RAW_ARCHIVE.json": {"records": [{"id": 1}] * 2},
        "reports/c2b/C2B_RAW_ARCHIVE.json": {"records": [{"id": 1}]},
        "reports/c2c/C2C_RAW_ARCHIVE.json": {"records": [{"id": 1}]},
        "reports/c2c/C2C_PROVENANCE.json": {"provider": "recorded"},
    }.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    raw = root / "reports/c2c/raw_responses"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "C2C_BATCH_000__seq01.json").write_text('{"text": "response"}', encoding="utf-8")

    out = root / audit.AUDIT_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "C3_PRE_LIVE_AUDIT.json").write_text('{"run": 1}', encoding="utf-8")

    # a non-provider C3 report that must stay outside the fingerprint
    contract = root / "reports/c3/v15_selection_contract"
    contract.mkdir(parents=True, exist_ok=True)
    (contract / "C3_SELECTION_CONTRACT.json").write_text('{"v": 1}', encoding="utf-8")


@pytest.fixture()
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    build_repo(tmp_path)
    monkeypatch.setattr(audit, "REPO", tmp_path)
    return tmp_path


def digest() -> str:
    return audit.provider_fingerprint()["fingerprint_sha256"]


# ------------------------------------------------------------------- stability
def test_two_runs_with_no_provider_change_give_the_same_fingerprint(repo: Path) -> None:
    assert digest() == digest()


def test_the_real_repository_fingerprint_is_stable_across_runs() -> None:
    """The regression as it actually manifested, against the real tree."""
    assert audit.provider_fingerprint()["fingerprint_sha256"] == \
           audit.provider_fingerprint()["fingerprint_sha256"]


def test_rewriting_an_audit_output_does_not_change_the_fingerprint(repo: Path) -> None:
    before = digest()
    written = repo / audit.AUDIT_OUTPUT_DIR / "C3_PRE_LIVE_AUDIT.json"
    written.write_text('{"run": 2, "regenerated": true}', encoding="utf-8")
    assert digest() == before


def test_a_brand_new_audit_output_file_does_not_change_the_fingerprint(repo: Path) -> None:
    before = digest()
    (repo / audit.AUDIT_OUTPUT_DIR / "C3_NEW_EVIDENCE.json").write_text(
        '{"added": true}', encoding="utf-8")
    assert digest() == before


def test_a_non_provider_c3_report_does_not_change_the_fingerprint(repo: Path) -> None:
    before = digest()
    (repo / "reports/c3/v15_selection_contract/C3_SELECTION_CONTRACT.json").write_text(
        '{"v": 2}', encoding="utf-8")
    assert digest() == before


# ------------------------------------------------------------------ sensitivity
def test_changing_a_provider_archive_changes_the_fingerprint(repo: Path) -> None:
    before = digest()
    (repo / "reports/c2c/C2C_RAW_ARCHIVE.json").write_text(
        json.dumps({"records": [{"id": 1}, {"id": 2}]}), encoding="utf-8")
    assert digest() != before


def test_changing_a_raw_response_payload_changes_the_fingerprint(repo: Path) -> None:
    before = digest()
    (repo / "reports/c2c/raw_responses/C2C_BATCH_000__seq01.json").write_text(
        '{"text": "a different response"}', encoding="utf-8")
    assert digest() != before


def test_a_new_c3_raw_archive_is_covered_the_moment_it_appears(repo: Path) -> None:
    """C3 generation must not be invisible to the very audit that guards it."""
    before = digest()
    (repo / "reports/c3/C3_RAW_ARCHIVE.json").write_text(
        json.dumps({"records": [{"slot": 0}]}), encoding="utf-8")
    assert digest() != before
    assert "reports/c3/C3_RAW_ARCHIVE.json" in \
           audit.provider_fingerprint()["fingerprinted_paths"]


def test_a_new_c3_raw_response_payload_is_covered(repo: Path) -> None:
    before = digest()
    payloads = repo / "reports/c3/raw_responses"
    payloads.mkdir(parents=True, exist_ok=True)
    (payloads / "C3_BATCH_000__seq01.json").write_text('{"text": "x"}', encoding="utf-8")
    assert digest() != before


# --------------------------------------------------------------------- scoping
def test_the_fingerprint_never_covers_audit_output(repo: Path) -> None:
    paths = audit.provider_fingerprint()["fingerprinted_paths"]
    assert paths
    assert not any(audit.is_audit_output(p) for p in paths)


def test_the_real_fingerprint_excludes_audit_output_and_contract_reports() -> None:
    fingerprint = audit.provider_fingerprint()
    paths = fingerprint["fingerprinted_paths"]
    assert paths, "the allowlist resolved to nothing; the scope is broken"
    assert not any(audit.is_audit_output(p) for p in paths)
    assert not any(p.startswith("reports/c3/v15_selection_contract") for p in paths)
    assert fingerprint["fingerprint_scope"]["covers_audit_output"] is False


def test_audit_output_paths_are_recognised() -> None:
    assert audit.is_audit_output(audit.AUDIT_OUTPUT_DIR)
    assert audit.is_audit_output(audit.AUDIT_OUTPUT_DIR + "/C3_PRE_LIVE_AUDIT.json")
    assert not audit.is_audit_output("reports/c3/C3_BANK_LOCK.json")
    assert not audit.is_audit_output("reports/c3/v15_pre_live_audit_other/x.json")


# ---------------------------------------------- preserved scientific quantities
def test_archived_provider_record_counts_are_preserved(repo: Path) -> None:
    fingerprint = audit.provider_fingerprint()
    assert fingerprint["archived_provider_records_by_milestone"] == {
        "reports/c2/C2_PILOT_RAW_ARCHIVE.json": 42,
        "reports/c2/C2_SMOKE_RAW_ARCHIVE.json": 2,
        "reports/c2b/C2B_RAW_ARCHIVE.json": 1,
        "reports/c2c/C2C_RAW_ARCHIVE.json": 1,
    }
    assert fingerprint["archived_provider_records_total"] == 46


def test_the_real_archived_total_is_the_recorded_46() -> None:
    assert audit.provider_fingerprint()["archived_provider_records_total"] == 46


def test_provider_delta_is_zero_across_a_fingerprint_pair(repo: Path) -> None:
    before = audit.provider_fingerprint()
    after = audit.provider_fingerprint()
    assert after["archived_provider_records_total"] - \
           before["archived_provider_records_total"] == 0
    assert before["fingerprint_sha256"] == after["fingerprint_sha256"]


def test_no_c3_generation_shaped_artifact_exists_in_the_real_tree() -> None:
    assert audit.provider_fingerprint()["c3_generation_shaped_artifacts"] == []


def test_the_generation_tripwire_still_fires(repo: Path) -> None:
    """Narrowing the fingerprint must not narrow the tripwire."""
    assert audit.generation_shaped_artifacts() == []
    (repo / "reports/c3/C3_RAW_ARCHIVE.json").write_text('{"records": []}', encoding="utf-8")
    assert "reports/c3/C3_RAW_ARCHIVE.json" in audit.generation_shaped_artifacts()


def test_the_tripwire_catches_an_unpredicted_generation_artifact(repo: Path) -> None:
    (repo / "reports/c3/nested").mkdir(parents=True, exist_ok=True)
    (repo / "reports/c3/nested/RECIPE_BANK_LOCK.json").write_text("{}", encoding="utf-8")
    assert "reports/c3/nested/RECIPE_BANK_LOCK.json" in audit.generation_shaped_artifacts()


def test_the_tripwire_ignores_its_own_audit_output(repo: Path) -> None:
    (repo / audit.AUDIT_OUTPUT_DIR / "C3_BATCH_notes.json").write_text("{}", encoding="utf-8")
    assert audit.generation_shaped_artifacts() == []
