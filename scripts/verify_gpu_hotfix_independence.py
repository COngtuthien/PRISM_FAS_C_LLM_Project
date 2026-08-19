"""Prove the GPU laptop needs only the hotfix package, not a 34 GB recopy.

The claim under test is narrow and falsifiable: take the tree the deployed
machine actually has, replace nothing but the files in
`reports/handoff/GPU_LAPTOP_BOOTSTRAP_HOTFIX/`, and the bootstrap fix works.

The fixture is the real pre-fix tree, reconstructed from Git rather than
hand-written, with sentinel files standing in for the datasets and weights that
must not move. The functional check then runs inside that fixture, so a hidden
dependency on some other changed file would surface as an import or attribute
error instead of an opinion.

    python scripts/verify_gpu_hotfix_independence.py [base-commit]
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "reports" / "handoff" / "GPU_LAPTOP_BOOTSTRAP_HOTFIX"
META = {"HOTFIX_MANIFEST.json", "README_APPLY_HOTFIX.md", "APPLY_HOTFIX.ps1"}

#: Categories that may legitimately change without the GPU laptop needing them.
NON_RUNTIME_PREFIXES = ("tests/", "docs/", "reports/", "scripts/", "state/",
                        "CHANGELOG.md", "MILESTONES.md", "DECISIONS.md",
                        "PROJECT_STATUS.md", "README")

#: Stand-ins for the ~34 GB that must survive the operation untouched.
SENTINELS = {
    "data/raw/casia_fasd/train/live/sentinel.txt": "raw corpus",
    "weights/face_detectors/sentinel.txt": "frozen detector",
    "assets/sentinel.txt": "recipe bank",
    "runs/full/sentinel.txt": "scientific run evidence",
    "reports/full/sentinel.txt": "scientific report evidence",
    "state/sentinel.txt": "pipeline state",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=str(REPO),
                                   text=True).strip()


def extract_tree(commit: str, destination: Path) -> None:
    """The deployed machine's tree, as Git recorded it."""
    archive = destination.parent / "old_tree.tar"
    with archive.open("wb") as stream:
        subprocess.check_call(["git", "archive", "--format=tar", commit],
                              cwd=str(REPO), stdout=stream)
    with tarfile.open(archive) as tar:
        tar.extractall(destination, filter="data")
    archive.unlink()


def main() -> int:
    manifest = json.loads((PACKAGE / "HOTFIX_MANIFEST.json").read_text(
        encoding="utf-8"))
    head = git("rev-parse", "HEAD")
    base = sys.argv[1] if len(sys.argv) > 1 else manifest[
        "previous_commit_for_old_hashes"]
    shipped = [entry["destination"] for entry in manifest["files"]]

    findings: dict[str, object] = {
        "schema_version": "prism-hotfix-independence-check-v1",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "authoritative_commit": head,
        "fixture_commit": base,
        "hotfix_identity": manifest["hotfix_identity"],
        "shipped_files": shipped,
    }

    with tempfile.TemporaryDirectory(prefix="prism-hotfix-") as raw:
        root = Path(raw) / "PRISM_FAS_C_LLM_Project"
        root.mkdir(parents=True)
        extract_tree(base, root)

        # The datasets and weights that must not move.
        sentinel_hashes = {}
        for relative, description in SENTINELS.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = f"{description}: do not touch\n".encode()
            path.write_bytes(payload)
            sentinel_hashes[relative] = sha256_bytes(payload)

        before = {relative: sha256(root / relative) for relative in shipped
                  if (root / relative).is_file()}

        # --- apply ONLY the hotfix package --------------------------------
        applied = []
        for relative in shipped:
            source = PACKAGE / relative
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            applied.append(relative)

        # --- 1. the runtime files now match authoritative HEAD ------------
        matches = {}
        for relative in shipped:
            matches[relative] = {
                "before": before.get(relative),
                "after": sha256(root / relative),
                "authoritative": sha256(REPO / relative),
                "changed_by_the_hotfix": before.get(relative) != sha256(
                    root / relative),
            }
        findings["runtime_files"] = matches
        findings["all_runtime_files_match_head"] = all(
            item["after"] == item["authoritative"] for item in matches.values())

        # --- 2. nothing else moved ----------------------------------------
        untouched = {relative: sha256(root / relative) == expected
                     for relative, expected in sentinel_hashes.items()}
        findings["sentinels_untouched"] = untouched
        findings["data_weights_evidence_untouched"] = all(untouched.values())

        # --- 3. the fix is functional with nothing else replaced ----------
        probe = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import bootstrap as boot

contract = boot.read_contract()
msys = {"executable": r"C:\msys64\mingw64\bin\python.exe", "version": "3.12.11",
        "version_info": [3, 12, 11], "implementation": "CPython",
        "sys_platform": "win32", "os_name": "nt",
        "sysconfig_platform": "mingw_x86_64_ucrt", "default_scheme": "posix_prefix",
        "venv_scripts_path": "base/bin", "prefix": r"C:\msys64\mingw64",
        "base_prefix": r"C:\msys64\mingw64", "msystem": "MINGW64", "usable": True}
verdict = boot.classify_interpreter(msys)


def selector(_contract):
    return boot.select_windows_cpython(
        _contract,
        candidates=[{"source": "py_launcher", "tag": "3.12",
                     "executable": r"C:\Python312\python.exe", "default": True}],
        probe=lambda executable: {
            "executable": str(executable), "version": "3.12.8",
            "version_info": [3, 12, 8], "implementation": "CPython",
            "sys_platform": "win32", "os_name": "nt",
            "sysconfig_platform": "win-amd64", "default_scheme": "nt",
            "venv_scripts_path": r"base\Scripts", "prefix": r"C:\Python312",
            "base_prefix": r"C:\Python312", "msystem": None, "usable": True})


host = boot.resolve_host_interpreter(contract, evidence=msys, selector=selector)
print(json.dumps({
    "msys2_classified": verdict["classification"],
    "msys2_may_build": verdict["may_build_the_project_environment"],
    "fallback_selection": host["selection"],
    "fallback_executable": host["executable"],
    "fallback_scheme": host["venv_scheme"],
    "venv_interpreter": str(boot.venv_python(boot.VENV, scheme=host["venv_scheme"])),
    "pip_policy": boot.pip_policy(contract),
    "onnxruntime_pin": contract["dependencies"]["onnxruntime"]["pin"],
    "requirement_closure": sorted(
        path.name for path in boot.requirement_files(
            contract["profiles"]["cuda-cu129"])),
    "science_import_groups": boot.import_groups(contract, scientific=True),
}))
"""
        output = subprocess.check_output([sys.executable, "-c", probe, str(root)],
                                         text=True, cwd=str(root))
        behaviour = json.loads(output.strip().splitlines()[-1])
        findings["behaviour_in_the_patched_fixture"] = behaviour
        findings["fix_is_functional_with_only_these_files"] = (
            behaviour["msys2_classified"] == "MSYS2_MINGW_PYTHON"
            and behaviour["msys2_may_build"] is False
            and behaviour["fallback_selection"] == "WINDOWS_CPYTHON_FALLBACK"
            and behaviour["fallback_scheme"] == "windows"
            and behaviour["venv_interpreter"].endswith("Scripts\\python.exe")
            and behaviour["onnxruntime_pin"] == "1.24.1"
            and behaviour["pip_policy"]["upgrade_policy"] == "BOUNDED_MINIMUM_ONLY"
            and "science_only" in behaviour["science_import_groups"])

        findings["files_applied"] = applied

    # --- 4. everything else that changed is genuinely non-runtime ---------
    changed = [line for line in git("diff", "--name-only", base, head).splitlines()
               if line]
    leftover = [path for path in changed
                if path not in shipped
                and not path.startswith(NON_RUNTIME_PREFIXES)
                and not path.startswith("reports/handoff/")]
    findings["repository_files_changed"] = changed
    findings["runtime_changes_not_shipped"] = leftover
    findings["no_unshipped_runtime_change"] = not leftover

    findings["verdict"] = (
        "PASS" if (findings["all_runtime_files_match_head"]
                   and findings["data_weights_evidence_untouched"]
                   and findings["fix_is_functional_with_only_these_files"]
                   and findings["no_unshipped_runtime_change"]) else "FAIL")

    out = PACKAGE / "HOTFIX_INDEPENDENCE_CHECK.json"
    out.write_text(json.dumps(findings, indent=2) + "\n", encoding="utf-8",
                   newline="\n")
    print(json.dumps({key: findings[key] for key in (
        "verdict", "all_runtime_files_match_head",
        "data_weights_evidence_untouched",
        "fix_is_functional_with_only_these_files",
        "no_unshipped_runtime_change", "runtime_changes_not_shipped")}, indent=2))
    print("wrote", out.relative_to(REPO).as_posix())
    return 0 if findings["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
