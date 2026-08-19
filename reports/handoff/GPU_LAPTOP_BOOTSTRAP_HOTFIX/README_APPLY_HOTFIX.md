# Applying the bootstrap hotfix to the GPU laptop

This replaces 8 runtime files. It does **not** touch your datasets, your
weights or any evidence, so the ~34 GB folder is not recopied.

Built from branch `portable-one-command-full-run` at commit `8d810aab0cfc`.
Hotfix identity `8f0fc9467972abfb`.

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
    └── requirements/{constraints,cpu,cuda-cu126,cuda-cu129,cuda-cu130}.txt

Or let the script do it and verify every hash:

    powershell -ExecutionPolicy Bypass -File APPLY_HOTFIX.ps1 -ProjectRoot "C:\path\to\PRISM_FAS_C_LLM_Project"

It prints `HOTFIX_APPLIED = PASS` and `FILES_VERIFIED = 8/8`.
It never starts training.

`HOTFIX_MANIFEST.json`, `README_APPLY_HOTFIX.md`, `APPLY_HOTFIX.ps1` and
`HOTFIX_INDEPENDENCE_CHECK.json` are documentation and tooling for this package.
Do **not** copy them into the project root — the PowerShell script already
excludes them.

## 4. The CUDA profile your card will get

A third problem turned up while auditing the plan, before it could bite you: the
CUDA 12.9 index publishes torch 2.13.0 for Linux only. On Windows that profile
would not have failed — pip would have installed a different build from PyPI
while the environment manifest went on naming an index it never used. The host
platform is now checked first, and a Windows Blackwell card resolves to the new
`cuda-cu130` profile, which does publish Windows wheels. An Ada or Ampere card
resolves to `cuda-cu126` as before. After installing, the runner reads back what
torch actually is and refuses a build that is not the one the profile named.

CUDA 13.0 needs an R580 driver or newer. If your driver is older and your card is
Blackwell, the run stops and says so; update the driver and run again.

## 5. The half-finished `.venv`

**You do not need to delete `.venv`.** The failed run left one that exists, runs,
and is missing packages. The new bootstrap classifies that as
`DEPENDENCIES_INCOMPLETE` and installs the rest into it — no rebuild, no manual
deletion. If instead it finds an environment that cannot work on this host (the
POSIX `bin/` layout MSYS2 Python produces, a Python of the wrong minor version,
an interpreter that will not launch), it rebuilds `<project>/.venv` itself and
says so. It will never delete anything outside the project.

## 6. Run it

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

## 7. What you should see

    host interpreter    MSYS2_MINGW_PYTHON at C:\msys64\mingw64\bin\python.exe
    using instead       standard Windows CPython 3.12.x at C:\...\Python312\python.exe
    environment         INSTALLED  profile=cuda-cu130  id=...

(`cuda-cu126` instead of `cuda-cu130` if your card is Ada or Ampere, or if
your driver is below R580. Never `cuda-cu129` on Windows — see section 4.)

then the preflight table, the GPU preflight, derived-data preparation and the
pipeline. The first run installs packages and needs the network; later runs do
not.
