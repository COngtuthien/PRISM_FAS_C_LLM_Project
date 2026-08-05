# PRISM-FAS-B

Local, reproducible data factory for face anti-spoofing research: read-only dataset auditing,
explicit-rule canonical adapters, and deterministic M2 preprocessing that turns raw source and
target media into face crops with strict Parquet manifests.

**Status: M2 complete, M3 not started.** See [PROJECT_STATUS.md](PROJECT_STATUS.md).

## Install and test

```bash
python -m pip install -e .[dev]
python -m pytest -q
prism --help
```

## Configuration

Copy `configs/paths.example.yaml` to `configs/paths.local.yaml` and fill in your local dataset,
model-cache and work roots. The local file is git-ignored because it contains machine-specific
absolute paths. Preprocessing parameters are frozen in `configs/data/preprocess_m2.yaml`; its hash
is part of the output path, so changing it produces a new namespace instead of mutating an old one.

## Full-profile preprocessing

```bash
python -m prism_fas.cli.main data preprocess run \
  --dataset <casia_fasd|msu_mfsd|siw_mv2> \
  --config configs/paths.local.yaml \
  --preprocess-config configs/data/preprocess_m2.yaml \
  --run-profile full_preprocessing --confirm-full-run --all-records \
  --output-dir-name full_preprocessing_v2
```

Validate a completed run:

```bash
python -m prism_fas.cli.main data preprocess validate \
  --config configs/paths.local.yaml \
  --preprocess-config configs/data/preprocess_m2.yaml \
  --output-root <work_root>/m2/<version>/<config_hash>/full_preprocessing_v2 \
  --validation-profile full_preprocessing
```

## Output layout

Runs write to `<work_root>/m2/<preprocessing_version>/<preprocessing_config_hash>/<namespace>/`
with `crops/`, `manifests/`, `state/`, `reports/` and `logs/` beneath it. `full_preprocessing_v2` is
the official M2 namespace for downstream work; earlier namespaces are kept as audit artifacts.

## What is not in this repository

Raw datasets (CASIA-FASD, MSU-MFSD, SiW-Mv2), SCRFD model weights, generated crop images, Parquet
manifests, run logs and local path configuration are intentionally excluded. They are large, license
restricted or machine-specific, and every artifact is reproducible from source with the commands
above.

## Target isolation

SiW-Mv2 is treated as an inference-only target. Its manifests carry deterministic opaque
identifiers and no label, attack, taxonomy, subject or session information; validation fails if any
private token reaches a target row.
