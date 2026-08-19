"""Build the small runtime hotfix package for the deployed GPU laptop.

The project folder on that machine is ~34 GB of raw datasets and frozen weights.
Recopying it to deliver a bootstrap fix would be absurd, so this assembles the
minimum: the runtime files the fix actually touches, in their project-relative
positions, plus a manifest, an apply script and instructions.

Only files that the RUNNING pipeline reads go in. Tests, docs, PROJECT_STATE,
data, weights and evidence stay out; they are listed in the manifest as
deliberately excluded so the omission is a recorded decision rather than a gap.

    python scripts/build_gpu_bootstrap_hotfix.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "reports" / "handoff" / "GPU_LAPTOP_BOOTSTRAP_HOTFIX"

#: Every runtime file the fix changes, with why it must travel.
RUNTIME_FILES: list[dict[str, object]] = [
    {
        "path": "train.py",
        "reason": "entrypoint: maps the new SUPPORTED_WINDOWS_CPYTHON_NOT_FOUND "
                  "blocker to a BLOCKED exit, and reports the host-interpreter "
                  "fallback and any .venv rebuild before handing over",
        "required": True,
        "scientific_identity_affected": False,
    },
    {
        "path": "bootstrap.py",
        "reason": "the fix itself: host-interpreter classification, the Windows "
                  "standard-CPython fallback, venv layout validation, partial and "
                  "foreign .venv recovery, the self-recreation guard and the "
                  "bounded pip policy",
        "required": True,
        "scientific_identity_affected": False,
    },
    {
        "path": "configs/environment/environment_contract.yaml",
        "reason": "declares what bootstrap.py now enforces: the interpreter "
                  "classifications, the preferred Python minors, the venv recovery "
                  "states, the bounded pip policy and the ONNX Runtime pin record",
        "required": True,
        "scientific_identity_affected": False,
    },
    {
        "path": "requirements/constraints.txt",
        "reason": "adds the onnxruntime pin so a drifting profile file becomes a "
                  "resolution error instead of a different preprocessing runtime",
        "required": True,
        "scientific_identity_affected": False,
    },
    {
        "path": "requirements/cpu.txt",
        "reason": "onnxruntime 1.24.0 -> 1.24.1; 1.24.0 does not exist on PyPI and "
                  "is what the deployment install died on",
        "required": True,
        "scientific_identity_affected": False,
    },
    {
        "path": "requirements/cuda-cu129.txt",
        "reason": "same pin repair on the CUDA 12.9 profile, which is the one a "
                  "Blackwell or Ada card selects",
        "required": True,
        "scientific_identity_affected": False,
    },
    {
        "path": "requirements/cuda-cu126.txt",
        "reason": "same pin repair on the CUDA 12.6 profile; which profile applies "
                  "is decided by the GPU detected on that machine, not from here",
        "required": True,
        "scientific_identity_affected": False,
    },
]

#: Changed in the same work, deliberately NOT shipped.
EXCLUDED: list[dict[str, str]] = [
    {"path": "tests/pipeline/test_bootstrap_host_interpreter.py",
     "category": "test",
     "why_not_shipped": "the GPU laptop runs the pipeline, not the suite; ship it "
                        "only if you intend to run pytest there"},
    {"path": "tests/pipeline/test_dependency_contract.py",
     "category": "test",
     "why_not_shipped": "same"},
    {"path": "tests/pipeline/test_gpu_hotfix_package.py",
     "category": "test",
     "why_not_shipped": "verifies this package on the build machine"},
    {"path": "reports/handoff/ONNXRUNTIME_PIN_EVIDENCE.json",
     "category": "evidence",
     "why_not_shipped": "the record of why the pin changed; read on the build "
                        "machine, never executed"},
    {"path": "docs/PROJECT_STATE.md",
     "category": "state",
     "why_not_shipped": "derived handoff state, not runtime input"},
    {"path": "scripts/build_gpu_bootstrap_hotfix.py",
     "category": "tooling",
     "why_not_shipped": "builds this package; nothing on the GPU laptop calls it"},
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*arguments: str) -> str:
    try:
        return subprocess.check_output(["git", *arguments], cwd=str(REPO), text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def previous_hash(relative: str, base: str) -> str | None:
    """What the deployed copy most likely still holds, if Git can tell us."""
    if not base:
        return None
    try:
        blob = subprocess.check_output(["git", "show", f"{base}:{relative}"],
                                       cwd=str(REPO), stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None
    return hashlib.sha256(blob).hexdigest()


README = """# Applying the bootstrap hotfix to the GPU laptop

This replaces {count} runtime files. It does **not** touch your datasets, your
weights or any evidence, so the ~34 GB folder is not recopied.

Built from branch `{branch}` at commit `{commit}`.
Hotfix identity `{identity}`.

## 1. Stop anything running

Close any Python process using the project — a running `train.py`, an open
notebook, an activated `.venv` in a terminal. Windows will not let a file be
replaced while it is loaded.

## 2. What must not be touched

Leave these exactly as they are:

    data/                 the raw CASIA / MSU / SiW-Mv2 corpora
    weights/              the frozen SCRFD, SigLIP2 and backbone weights
    assets/               recipe banks
    runs/  reports/  state/    scientific and rehearsal evidence

Nothing in this hotfix writes to any of them.

## 3. Copy the files

Copy the **contents** of `GPU_LAPTOP_BOOTSTRAP_HOTFIX/` over your project root,
preserving relative paths, and overwrite when asked:

    PRISM_FAS_C_LLM_Project/
    ├── train.py
    ├── bootstrap.py
    ├── configs/environment/environment_contract.yaml
    └── requirements/{{constraints,cpu,cuda-cu126,cuda-cu129}}.txt

Or let the script do it and verify every hash:

    powershell -ExecutionPolicy Bypass -File APPLY_HOTFIX.ps1 -ProjectRoot "C:\\path\\to\\PRISM_FAS_C_LLM_Project"

It prints `HOTFIX_APPLIED = PASS` and `FILES_VERIFIED = {count}/{count}`.
It never starts training.

`HOTFIX_MANIFEST.json`, `README_APPLY_HOTFIX.md` and `APPLY_HOTFIX.ps1` are
documentation and tooling for this package. Do **not** copy them into the project
root — the PowerShell script already excludes them.

## 4. The half-finished `.venv`

**You do not need to delete `.venv`.** The failed run left one that exists, runs,
and is missing packages. The new bootstrap classifies that as
`DEPENDENCIES_INCOMPLETE` and installs the rest into it — no rebuild, no manual
deletion. If instead it finds an environment that cannot work on this host (the
POSIX `bin/` layout MSYS2 Python produces, a Python of the wrong minor version,
an interpreter that will not launch), it rebuilds `<project>/.venv` itself and
says so. It will never delete anything outside the project.

## 5. Run it

    python train.py

That is the whole command. In particular:

* **not** `py -3.12 train.py` — if PATH `python` is MSYS2/MinGW Python, the runner
  now detects that and finds a supported standard Windows CPython through the
  Python Launcher on its own;
* **not** `pip install ...` — the runner installs the profile for your GPU;
* **not** activating `.venv` — the runner re-execs into it.

If no supported standard Windows CPython exists on the machine at all, the run
stops immediately with `SUPPORTED_WINDOWS_CPYTHON_NOT_FOUND` and prints the
interpreter it found, why it was refused, the supported version range and every
interpreter it discovered. Install a CPython from python.org in that range
(3.11-3.13, 3.12 or 3.13 preferred) and run the same command again.

## 6. What you should see

    host interpreter    MSYS2_MINGW_PYTHON at C:\\msys64\\mingw64\\bin\\python.exe
    using instead       standard Windows CPython 3.12.x at C:\\...\\Python312\\python.exe
    environment         INSTALLED  profile=cuda-cu129  id=...

then the preflight table, the GPU preflight, derived-data preparation and the
pipeline. The first run installs packages and needs the network; later runs do
not.
"""

APPLY_PS1 = r'''<#
    Applies the PRISM-FAS-C bootstrap hotfix to a deployed project folder.

    Copies only the runtime files listed in HOTFIX_MANIFEST.json, verifies every
    SHA256 after the copy, and stops on the first mismatch. It never touches
    data/, weights/, assets/, runs/, reports/ or state/, never deletes anything,
    and never starts training.

    Usage:
        powershell -ExecutionPolicy Bypass -File APPLY_HOTFIX.ps1 `
            -ProjectRoot "C:\path\to\PRISM_FAS_C_LLM_Project"
#>
[CmdletBinding()]
param(
    [string]$ProjectRoot,
    [switch]$WhatIfOnly
)

$ErrorActionPreference = "Stop"
$package = Split-Path -Parent $MyInvocation.MyCommand.Path
$manifestPath = Join-Path $package "HOTFIX_MANIFEST.json"

if (-not (Test-Path $manifestPath)) {
    Write-Error "HOTFIX_MANIFEST.json is missing next to this script."
}
$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json

# Default only to a project root that is unmistakably one.
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $candidate = (Resolve-Path (Join-Path $package "..\..\..")).Path
    $looksRight = (Test-Path (Join-Path $candidate "train.py")) -and
                  (Test-Path (Join-Path $candidate "bootstrap.py")) -and
                  (Test-Path (Join-Path $candidate "configs\environment\environment_contract.yaml"))
    if (-not $looksRight) {
        Write-Error "Pass -ProjectRoot explicitly: the folder above this package does not look like a project root."
    }
    $ProjectRoot = $candidate
    Write-Host "ProjectRoot not given; using the enclosing project at $ProjectRoot"
}

$ProjectRoot = (Resolve-Path $ProjectRoot).Path
foreach ($required in @("train.py", "bootstrap.py", "configs\environment\environment_contract.yaml")) {
    if (-not (Test-Path (Join-Path $ProjectRoot $required))) {
        Write-Error "$ProjectRoot does not look like a PRISM-FAS-C project (missing $required). Nothing was copied."
    }
}

# Paths this script must never write to, whatever a manifest says.
$forbidden = @("data", "weights", "assets", "runs", "reports", "state", ".git", ".venv")

$verified = 0
$total = ($manifest.files | Measure-Object).Count
foreach ($file in $manifest.files) {
    $relative = $file.destination -replace "/", "\"
    $top = ($relative -split "\\")[0]
    if ($forbidden -contains $top) {
        Write-Error "refusing to write into $top ($relative). Nothing further was copied."
    }

    $source = Join-Path $package $relative
    $destination = Join-Path $ProjectRoot $relative
    if (-not (Test-Path $source)) {
        Write-Error "the package is incomplete: $relative is missing."
    }

    if ($WhatIfOnly) {
        Write-Host ("WOULD COPY  {0}" -f $relative)
        continue
    }

    $parent = Split-Path -Parent $destination
    if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent | Out-Null }
    Copy-Item -Path $source -Destination $destination -Force

    $actual = (Get-FileHash -Path $destination -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $file.sha256) {
        Write-Error "SHA256 mismatch after copying $relative (got $actual, expected $($file.sha256))."
    }
    $verified += 1
    Write-Host ("OK  {0}  {1}" -f $file.sha256.Substring(0, 12), $relative)
}

if ($WhatIfOnly) {
    Write-Host "WHAT-IF only: nothing was written."
    exit 0
}

Write-Host ""
Write-Host "HOTFIX_APPLIED = PASS"
Write-Host ("FILES_VERIFIED = {0}/{1}" -f $verified, $total)
Write-Host "NEXT_COMMAND = python train.py"
'''


def main() -> int:
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    commit = git("rev-parse", "HEAD")
    base = git("rev-parse", "HEAD~1")

    if PACKAGE.exists():
        for item in sorted(PACKAGE.rglob("*"), reverse=True):
            item.unlink() if item.is_file() else item.rmdir()
    PACKAGE.mkdir(parents=True, exist_ok=True)

    entries = []
    for spec in RUNTIME_FILES:
        relative = str(spec["path"])
        source = REPO / relative
        if not source.is_file():
            raise SystemExit(f"missing runtime file: {relative}")
        destination = PACKAGE / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        entries.append({
            "destination": relative,
            "sha256": sha256(source),
            "bytes": source.stat().st_size,
            "reason": spec["reason"],
            "required_on_gpu_laptop": spec["required"],
            "previous_sha256": previous_hash(relative, base),
            "previous_sha256_source": f"git {base[:12]} (the commit the deployed "
                                      f"copy was taken from, if unchanged since)"
                                      if base else None,
            "scientific_identity_affected": spec["scientific_identity_affected"],
        })

    identity = hashlib.sha256(json.dumps(
        [[entry["destination"], entry["sha256"]] for entry in entries],
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    manifest = {
        "schema_version": "prism-gpu-bootstrap-hotfix-v1",
        "title": "Windows MSYS2 host-Python and ONNX Runtime bootstrap hotfix",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authoritative_branch": branch,
        "authoritative_commit": commit,
        "previous_commit_for_old_hashes": base,
        "hotfix_identity": identity,
        "file_count": len(entries),
        "full_project_recopy_required": False,
        "defects_fixed": [
            {"id": "MSYS2_HOST_PYTHON",
             "symptom": "PATH `python` resolved to C:\\msys64\\mingw64\\bin\\"
                        "python.exe, which reports os.name == 'nt' and then creates "
                        "a POSIX-scheme .venv/bin/python.exe on Windows",
             "fix": "explicit host-interpreter classification decided by the venv "
                    "scripts scheme, plus an automatic fallback to a supported "
                    "standard Windows CPython found through the Python Launcher"},
            {"id": "ONNXRUNTIME_PIN_NEVER_EXISTED",
             "symptom": "pip reported no matching distribution for "
                        "onnxruntime==1.24.0 after resolving torch",
             "fix": "pinned to 1.24.1, the smallest published patch in the same "
                    "release family, verified against the package index and "
                    "measured bit-identical to the historical runtime on the "
                    "frozen SCRFD detector"},
            {"id": "PARTIAL_VENV_NOT_RECOVERABLE",
             "symptom": "the failed install left a .venv that existed, ran and was "
                        "missing packages",
             "fix": "the environment is classified and repaired: dependencies are "
                    "installed into a structurally sound .venv, and only an "
                    "unusable one is rebuilt. No manual deletion."},
        ],
        "operator_contract": {
            "command": "python train.py",
            "manual_py_launcher_required": False,
            "manual_venv_activation_required": False,
            "manual_venv_deletion_required": False,
            "manual_pip_required": False,
        },
        "scientific_impact": {
            "protocols_or_constants_changed": False,
            "scientific_identity_affected": False,
            "c4_to_c13_scientific_status": "NOT_RUN",
            "target_access": 0,
            "evidence": "reports/handoff/ONNXRUNTIME_PIN_EVIDENCE.json "
                        "(on the build repository)",
        },
        "must_not_be_touched": ["data/", "weights/", "assets/", "runs/", "reports/",
                                "state/", ".git/"],
        "not_shipped": EXCLUDED,
        "package_own_files": ["HOTFIX_MANIFEST.json", "README_APPLY_HOTFIX.md",
                              "APPLY_HOTFIX.ps1", "HOTFIX_INDEPENDENCE_CHECK.json"],
        "expected_next_command": "python train.py",
        "files": entries,
    }

    (PACKAGE / "HOTFIX_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8", newline="\n")
    (PACKAGE / "README_APPLY_HOTFIX.md").write_text(
        README.format(count=len(entries), branch=branch or "(unknown)",
                      commit=(commit or "(unknown)")[:12], identity=identity[:16]),
        encoding="utf-8", newline="\n")
    (PACKAGE / "APPLY_HOTFIX.ps1").write_text(APPLY_PS1, encoding="utf-8",
                                              newline="\r\n")

    print(f"package     {PACKAGE.relative_to(REPO).as_posix()}")
    print(f"files       {len(entries)}")
    print(f"identity    {identity}")
    for entry in entries:
        print(f"  {entry['sha256'][:12]}  {entry['bytes']:>7}  "
              f"{entry['destination']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
