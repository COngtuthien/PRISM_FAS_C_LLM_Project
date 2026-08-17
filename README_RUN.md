# Running PRISM-FAS-C-LLM

Open a terminal in the project root and run:

```
python train.py
```

That is the whole normal workflow. No `pip install`, no flags, no setup script.

The first run creates `.venv/`, installs the right dependencies for your
hardware, checks that the folder is complete, works out what to do, and does it.
Every later run reuses that environment and contacts no package index.

---

## What you need on the machine

| | |
|---|---|
| **Python** | 3.11 – 3.13 (tested on 3.13.11). The runner never installs or replaces your Python. |
| **NVIDIA driver** | Only for scientific runs. Blackwell (RTX 50-series) needs ≥ 570; Ada/Ampere ≥ 525.60. |
| **Internet** | First run only, to install dependencies. Training itself is offline. |
| **Disk** | ≥ 5 GB free for a rehearsal, ≥ 50 GB for a scientific run. |

Nothing else. If a prerequisite is missing the runner stops before doing any
work and prints exactly what it needs.

---

## What the one command actually does

```
python train.py
  → check the Python version
  → detect CPU or CUDA, match it to a declared dependency profile
  → create .venv/ and install, or reuse an unchanged environment
  → re-exec inside the project environment
  → verify the folder has everything the resolved run needs
  → pick the execution intent
  → resume whatever is already finished
  → run, then write plots, tables and a report
```

It prints a preflight block before any long work starts, and stops there if
anything is missing.

### Two intents, and why they are separate

**No GPU → `CPU_FULL_REHEARSAL`.** The real implementation runs end to end on
fixtures: GPAT training, rendering, the quality gate, both detector tracks, the
matrix scheduler, the locks, the target-firewall logic, the scorer, the
acceptance machinery. It proves the code works.

It writes only to `reports/rehearsal/` and `runs/rehearsal/`, is never
scientifically eligible, and uses a fixture target package — it cannot open a
real SiW label. **A rehearsal can never produce a Version-C P3 result**, which is
the point: proving the implementation must not spend the held-out target.

It may be slow. That is expected and fine.

**Compatible GPU → `GPU_SCIENTIFIC_FULL`.** The frozen scientific pipeline, from
the first milestone that is not already complete through C13. Today that starts
at **C4**, because C0–C3 are scientifically complete and their evidence is
verified on every run rather than assumed.

Every gate stays in force. If a milestone's hard acceptance fails, or a new
decision needs your approval, the run stops cleanly and says so — automatic
progression means progression through *already-approved* contracts only.

---

## What you get when it finishes

```
reports/<profile>/
    plots/     training curves, learning rates, coverage, ROC, calibration, forest plots
    tables/    main_results.csv, per_seed_results.csv, hypothesis_tests.csv, ...
    final/
        report.html                 self-contained; open it in a browser
        FINAL_BUNDLE_MANIFEST.json  every artifact, by path and hash
    c4/ ... c13/                    per-stage evidence, including failures
runs/<profile>/<protocol>/<method>/<config>/<seed>/
    config.json  run_manifest.json  train_history.jsonl  checkpoint/  status.json
state/
    PIPELINE_STATE.json      where to resume
    MASTER_RUN_INDEX.json    every run ever recorded, including losing ones
```

`report.html` is written to be read with no other context. Sections with no
evidence say so rather than showing a placeholder number.

---

## Stopping and resuming

Interrupt it however you like — Ctrl-C, power loss, a closed laptop. To carry on:

```
python train.py
```

The same command. Resume is the default and is identity-aware: completed work is
skipped only after its parent identities, config identity, content hash and
acceptance state all validate. If something upstream changed, the run fails
closed rather than reusing a stale descendant.

---

## Moving the folder

Copy the whole project directory anywhere — another drive, another machine, a
path with spaces. Scientific identities do not depend on the location, the
hostname or the GPU.

Before a long run the runner checks the copy is complete and, if not, prints the
missing paths and stops. You can ask for that check on its own:

```
python train.py --preflight-only
```

`PORTABLE_ASSET_MANIFEST.json` in the project root lists every external
dependency: what it is, where it goes, its hash, which stage needs it, and
whether a rehearsal or a scientific run requires it.

### What travels in the folder, and what does not

**Travels:** code, configs, the frozen C3 recipe banks, all evidence and locks,
the environment contract, the dependency manifests.

**Does not travel** (too large or licensed — point at them in
`configs/paths.local.yaml`): the CASIA-FASD, MSU-MFSD and SiW-Mv2 datasets, the
pinned model weights (~2.8 GB), and the derived `data/processed` and
`data/packages` trees, which are rebuildable from the raw datasets.

---

## Troubleshooting

**`UNSUPPORTED_PYTHON`** — install a Python in 3.11–3.13 and rerun.

**`CUDA_ENVIRONMENT_NOT_VALIDATED`** — your GPU or driver matches no declared
profile. The message prints your GPU, driver and compute capability alongside
every profile and its requirement. Update the driver, or extend
`configs/environment/environment_contract.yaml` deliberately. The runner will not
guess a CUDA wheel and will not silently fall back to CPU for science.

**`the portable bundle is missing N required item(s)`** — the preflight lists
each missing path and how to obtain it. Nothing was executed.

**Environment seems wrong** — delete `.venv/` and `state/ENVIRONMENT_MANIFEST.json`
and rerun; the bootstrap rebuilds from the locked manifests.

---

## Expert flags

Not needed for the normal workflow, still supported for debugging:

```
python train.py --profile validate                        # static checks only
python train.py --profile smoke --from C0 --to C13        # fast fixture traversal
python train.py --profile rehearsal                       # force the CPU rehearsal
python train.py --profile full --from C4 --to C4 --resume # one scientific stage
python train.py --preflight-only                          # resolve and report, run nothing
python train.py --no-bootstrap                            # use the current interpreter
```

Exit codes: `0` pass, `1` fail, `2` blocked, `3` usage error.

Dependency profiles live in `requirements/`: `base.txt` (hardware-independent),
`cpu.txt`, `cuda-cu129.txt` (Blackwell), `cuda-cu126.txt` (Ada/Ampere),
`dev.txt`, and `constraints.txt` holding the exact tested versions. Root
`requirements.txt` points at the CPU profile as a safe default. You normally
never run pip yourself.
