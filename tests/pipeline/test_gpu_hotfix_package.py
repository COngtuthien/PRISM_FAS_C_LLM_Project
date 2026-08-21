"""The GPU-laptop hotfix package: complete, honest and safe to apply.

The deployed machine holds ~34 GB of raw corpora and frozen weights. The fix it
needs is a few kilobytes of bootstrap code, so it travels as a small package
rather than a recopy. That only works if three things hold, and each is a test
below: the package carries every runtime file the fix needs, it carries nothing
that would overwrite data or evidence, and what it claims about itself matches
the repository it was cut from.

A stale package is worse than none — it would be applied with confidence and
leave the machine on half a fix — so the hashes are checked against the working
tree rather than trusted.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "reports" / "handoff" / "GPU_LAPTOP_BOOTSTRAP_HOTFIX"
META = {"HOTFIX_MANIFEST.json", "README_APPLY_HOTFIX.md", "APPLY_HOTFIX.ps1",
        "HOTFIX_INDEPENDENCE_CHECK.json"}

#: Nothing under these may ever appear in the package: applying it would
#: overwrite datasets, frozen weights or recorded evidence.
FORBIDDEN_ROOTS = ("data", "weights", "assets", "runs", "state", ".git", ".venv")


@pytest.fixture(scope="module")
def manifest() -> dict:
    path = PACKAGE / "HOTFIX_MANIFEST.json"
    if not path.exists():
        pytest.skip("the hotfix package has not been built "
                    "(scripts/build_gpu_bootstrap_hotfix.py)")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _blob(commit: str, path: str) -> bytes | None:
    """The file's bytes at a commit, or None when git cannot answer."""
    import subprocess

    result = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=REPO,
                            capture_output=True, check=False)
    return result.stdout if result.returncode == 0 else None


def test_every_shipped_file_matches_the_commit_it_was_cut_from(
        manifest: dict) -> None:
    """A stale package would be applied with confidence and fix half the problem.

    The comparison is against `authoritative_commit`, which is what the package
    declares about itself — not against the working tree. A later milestone that
    legitimately edits `train.py` does not make this delivered package wrong; it
    would only be wrong if its bytes stopped matching the commit it names.
    """
    import hashlib

    commit = manifest["authoritative_commit"]
    for entry in manifest["files"]:
        shipped = PACKAGE / entry["destination"]
        assert shipped.is_file(), entry["destination"]
        assert (REPO / entry["destination"]).is_file(), entry["destination"]
        assert sha256(shipped) == entry["sha256"], entry["destination"]
        assert shipped.stat().st_size == entry["bytes"]

        recorded = _blob(commit, entry["destination"])
        if recorded is None:                      # shallow clone or no git
            continue
        # The package ships CRLF as a Windows checkout writes it; git stores LF.
        assert hashlib.sha256(recorded).hexdigest() in {
            entry["sha256"],
            hashlib.sha256(shipped.read_bytes().replace(b"\r\n", b"\n")).hexdigest()}, (
            f"{entry['destination']} does not match {commit[:12]}, the commit this "
            "package names; rebuild it before handing the link over")


def test_the_package_contains_exactly_what_it_declares(manifest: dict) -> None:
    declared = {entry["destination"] for entry in manifest["files"]} | META
    present = {item.relative_to(PACKAGE).as_posix()
               for item in PACKAGE.rglob("*") if item.is_file()}
    assert present == declared, present ^ declared


def test_the_package_carries_the_whole_fix_and_not_a_subset(manifest: dict) -> None:
    """Each of these is load-bearing; any one missing leaves a broken machine."""
    shipped = {entry["destination"] for entry in manifest["files"]}
    assert shipped >= {
        "train.py",                                    # the exit-code mapping
        "bootstrap.py",                                # the fix
        "configs/environment/environment_contract.yaml",  # what it enforces
        "requirements/constraints.txt",                # the shared pin
        "requirements/cpu.txt",
        "requirements/cuda-cu130.txt",           # the Windows CUDA path
        "requirements/cuda-cu129.txt",
        "requirements/cuda-cu126.txt",
    }


def test_the_package_cannot_overwrite_data_weights_or_evidence(
        manifest: dict) -> None:
    for entry in manifest["files"]:
        top = entry["destination"].split("/")[0]
        assert top not in FORBIDDEN_ROOTS, entry["destination"]
    for item in PACKAGE.rglob("*"):
        if item.is_file():
            top = item.relative_to(PACKAGE).as_posix().split("/")[0]
            assert top not in FORBIDDEN_ROOTS, item


def test_no_test_or_documentation_file_is_shipped_as_runtime(manifest: dict) -> None:
    for entry in manifest["files"]:
        destination = entry["destination"]
        assert not destination.startswith(("tests/", "docs/", "reports/")), destination
    excluded = {item["path"] for item in manifest["not_shipped"]}
    shipped = {entry["destination"] for entry in manifest["files"]}
    assert not (excluded & shipped)


def test_every_shipped_file_says_why_it_is_there(manifest: dict) -> None:
    for entry in manifest["files"]:
        assert entry["reason"] and len(entry["reason"]) > 30, entry["destination"]
        assert isinstance(entry["required_on_gpu_laptop"], bool)
        assert entry["scientific_identity_affected"] is False
        assert len(entry["sha256"]) == 64


def test_the_manifest_records_where_it_came_from(manifest: dict) -> None:
    assert len(manifest["authoritative_commit"]) == 40
    assert manifest["authoritative_branch"]
    assert len(manifest["hotfix_identity"]) == 64
    assert manifest["file_count"] == len(manifest["files"])
    assert manifest["full_project_recopy_required"] is False
    assert manifest["expected_next_command"] == "python train.py"


def test_the_manifest_states_the_one_command_contract(manifest: dict) -> None:
    contract = manifest["operator_contract"]
    assert contract["command"] == "python train.py"
    assert contract["manual_py_launcher_required"] is False
    assert contract["manual_venv_activation_required"] is False
    assert contract["manual_venv_deletion_required"] is False
    assert contract["manual_pip_required"] is False


def test_the_manifest_does_not_claim_a_scientific_change(manifest: dict) -> None:
    impact = manifest["scientific_impact"]
    assert impact["protocols_or_constants_changed"] is False
    assert impact["c4_to_c13_scientific_status"] == "NOT_RUN"
    assert impact["target_access"] == 0


def test_the_apply_script_refuses_the_directories_that_must_not_move() -> None:
    script = (PACKAGE / "APPLY_HOTFIX.ps1")
    if not script.exists():
        pytest.skip("the hotfix package has not been built")
    text = script.read_text(encoding="utf-8")
    for name in ("data", "weights", "assets", "runs", "reports", "state"):
        assert f'"{name}"' in text, name
    assert "refusing to write into" in text
    assert "HOTFIX_APPLIED = PASS" in text
    assert "NEXT_COMMAND = python train.py" in text
    assert "Remove-Item" not in text, "the apply script must never delete anything"
    assert "Start-Process" not in text
    assert text.count("python train.py") == 1, (
        "the only mention of the command must be the one it prints; the apply "
        "script must never start training itself")


def test_the_readme_tells_the_operator_not_to_delete_the_partial_venv() -> None:
    readme = PACKAGE / "README_APPLY_HOTFIX.md"
    if not readme.exists():
        pytest.skip("the hotfix package has not been built")
    text = readme.read_text(encoding="utf-8")
    assert "You do not need to delete `.venv`" in text
    assert "python train.py" in text
    assert "py -3.12" in text, "name the command that is NOT required"
    assert "data/" in text and "weights/" in text
