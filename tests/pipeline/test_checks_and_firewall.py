"""Validate-profile checks (L.2) and the firewall around them.

Two jobs. First, the checks must actually measure the repository rather than
agree with a document — so each one is also run against a deliberately broken
copy and must fail there. A check that cannot fail is not a check.

Second, nothing on the validate path may reach a live provider, a GPU job or a
target label. That is asserted structurally: the modules are inspected for the
imports that would make it possible, not merely trusted not to use them.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from prism_fas.pipeline import checks
from prism_fas.pipeline.checks import CHECKS, run_check
from prism_fas.pipeline.stages import STAGES

PIPELINE_DIR = Path(__file__).resolve().parents[2] / "src" / "prism_fas" / "pipeline"


# --- the checks measure something -------------------------------------------

def test_every_declared_check_exists() -> None:
    """A stage cannot reference a check that was never implemented."""
    declared = {check_id for stage in STAGES for check_id in stage.validate_checks}
    assert declared <= set(CHECKS)


def test_every_implemented_check_is_declared_by_a_stage() -> None:
    declared = {check_id for stage in STAGES for check_id in stage.validate_checks}
    assert set(CHECKS) == declared


def test_all_checks_pass_against_the_live_repository(repo: Path) -> None:
    failures = {check_id: run_check(check_id, repo).summary
                for check_id in CHECKS if not run_check(check_id, repo).ok}
    assert failures == {}


def test_an_unknown_check_id_fails_rather_than_raising(repo: Path) -> None:
    result = run_check("no_such_check", repo)
    assert not result.ok
    assert "no such check" in result.summary


def test_a_check_that_raises_becomes_a_failed_measurement(
        repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(_: Path):
        raise RuntimeError("boom")

    monkeypatch.setitem(CHECKS, "spec_sha256", explode)
    result = run_check("spec_sha256", repo)
    assert not result.ok
    assert "RuntimeError" in result.summary


# --- the checks can fail ----------------------------------------------------

def test_a_wrong_spec_is_detected(tmp_path: Path) -> None:
    target = tmp_path / checks.SPEC_RELPATH
    target.parent.mkdir(parents=True)
    target.write_bytes(b"not the spec")
    result = checks.check_spec_sha256(tmp_path)
    assert not result.ok
    assert result.detail["expected"] == checks.EXPECTED_SPEC_SHA256
    assert result.detail["actual"] != checks.EXPECTED_SPEC_SHA256


def test_a_missing_spec_is_detected(tmp_path: Path) -> None:
    assert not checks.check_spec_sha256(tmp_path).ok


def test_a_missing_acceptance_file_is_detected(tmp_path: Path) -> None:
    assert not checks.check_c0_acceptance(tmp_path).ok


def test_a_drifted_contract_identity_is_detected(repo: Path, tmp_path: Path) -> None:
    """Rewrite the lock's expected value; the live code must disagree with it."""
    import shutil

    for relative in ("configs", "src"):
        shutil.copytree(repo / relative, tmp_path / relative,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    lock_source = repo / checks.PRELIMINARY_LOCK
    lock_target = tmp_path / checks.PRELIMINARY_LOCK
    lock_target.parent.mkdir(parents=True)
    payload = json.loads(lock_source.read_text(encoding="utf-8"))
    payload["components"]["ontology_identity"] = "0" * 64
    lock_target.write_text(json.dumps(payload), encoding="utf-8")

    result = checks.check_contract_identities(tmp_path)
    assert not result.ok
    assert "ontology_identity" in result.detail["drifted"]


def test_c3_generation_evidence_would_be_detected(repo: Path, tmp_path: Path) -> None:
    """The prohibition is measured, so planting evidence must trip the check."""
    assert checks.check_c3_generation_not_started(tmp_path).ok

    planted = tmp_path / "reports" / "c3" / "C3_RAW_ARCHIVE.json"
    planted.parent.mkdir(parents=True)
    planted.write_text("{}", encoding="utf-8")
    result = checks.check_c3_generation_not_started(tmp_path)
    assert not result.ok
    assert "reports/c3/C3_RAW_ARCHIVE.json" in result.detail["found"]


def test_no_c3_generation_evidence_exists_in_the_real_repository(repo: Path) -> None:
    result = checks.check_c3_generation_not_started(repo)
    assert result.ok, result.detail["found"]


def test_version_b_is_at_the_frozen_commit(repo: Path) -> None:
    result = checks.check_version_b_integrity(repo)
    assert result.ok
    assert result.detail["head"] == checks.EXPECTED_VERSION_B_HEAD
    assert result.detail["tag_peeled_commit"] == checks.EXPECTED_VERSION_B_HEAD
    assert result.detail["clean"] is True


def test_a_missing_version_b_is_detected(monkeypatch: pytest.MonkeyPatch,
                                         repo: Path, tmp_path: Path) -> None:
    monkeypatch.setattr(checks, "VERSION_B_PATH", tmp_path / "absent")
    assert not checks.check_version_b_integrity(repo).ok


def test_the_route_contract_is_checked_for_equality_not_containment(repo: Path) -> None:
    result = checks.check_route_contract_exact(repo)
    assert result.ok
    assert result.detail["actual"] == ["physics", "gpat"]


def test_both_c3_locks_still_verify(repo: Path) -> None:
    result = checks.check_c3_locks_verify(repo)
    assert result.ok, result.detail["problems"]
    assert result.detail["preliminary"]["body_hash_reproduces"]
    assert result.detail["superseding"]["status"] == "PRE_SCIENTIFIC_SUPERSEDING_CONTRACT_LOCK"


def test_the_c3_identities_re_derive_from_live_code(repo: Path) -> None:
    result = checks.check_c3_contract_identities(repo)
    assert result.ok, result.detail["drifted"]
    assert len(result.detail["comparisons"]) == 4


# --- firewall ---------------------------------------------------------------

#: Modules that can reach a provider, a GPU job or the target labels. None of
#: them may be reachable from the execution layer while no adapter exists.
FORBIDDEN_IMPORTS = (
    "prism_fas.llm.providers",
    "prism_fas.llm.pipeline",
    "prism_fas.cloud",
    "prism_fas.data.target_eval",
    "prism_fas.evaluation.target_prediction",
    "google.genai",
    "modal",
    "torch",
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("module_path", sorted(PIPELINE_DIR.glob("*.py")),
                         ids=lambda path: path.name)
def test_no_pipeline_module_imports_a_provider_gpu_or_target_module(
        module_path: Path) -> None:
    imported = _imported_modules(module_path)
    offending = {name for name in imported
                 if any(name == forbidden or name.startswith(f"{forbidden}.")
                        for forbidden in FORBIDDEN_IMPORTS)}
    assert not offending, f"{module_path.name} imports {sorted(offending)}"


def test_train_py_imports_no_provider_gpu_or_target_module(repo: Path) -> None:
    imported = _imported_modules(repo / "train.py")
    offending = {name for name in imported
                 if any(name == forbidden or name.startswith(f"{forbidden}.")
                        for forbidden in FORBIDDEN_IMPORTS)}
    assert not offending, f"train.py imports {sorted(offending)}"


def test_train_py_delegates_rather_than_implementing(repo: Path) -> None:
    """L.4: the entrypoint must not absorb recipe, GPAT, synthesis or lock logic."""
    tree = ast.parse((repo / "train.py").read_text(encoding="utf-8"))
    functions = {node.name for node in ast.walk(tree)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert functions == {"build_parser", "main"}
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]


def test_a_validate_run_leaves_no_provider_credential_in_its_artifacts(
        repo: Path) -> None:
    """No artifact may serialize anything shaped like a key."""
    for path in (repo / "reports" / "validate").rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert "AIza" not in text
        assert "GEMINI_API_KEY" not in text
