# PRISM-FAS-B

Local, reproducible data factory for face anti-spoofing research: read-only dataset auditing,
explicit-rule canonical adapters, and deterministic M2 preprocessing that turns raw source and
target media into face crops with strict Parquet manifests.

**Status: M2 and M3 complete; M4 not started.** See [PROJECT_STATUS.md](PROJECT_STATUS.md).

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

## M3A package build

```bash
python -m prism_fas.cli.main data package build \
  --config configs/paths.local.yaml \
  --input-root <work_root>/m2/<version>/<config_hash>/full_preprocessing_v2 \
  --package-root <processed_root>/prism_data_v1_m3a --resume

python -m prism_fas.cli.main data package validate \
  --package-root <processed_root>/prism_data_v1_m3a
```

The package holds `images/`, `priors/`, `manifests/`, `shards/`, `audit/` and `PACKAGE_LOCK.json`.
`prism_data_v1_m3a` is the M3A package root; `prism_data_v1_m3b` adds model-dependent priors
(FaceXFormer parsing/pose, derived nine-region visibility, AdaFace IR-50 identity) and is the
official package for downstream work:

```bash
python -m prism_fas.cli.main data priors model-build \
  --input-package <processed_root>/prism_data_v1_m3a \
  --output-package <processed_root>/prism_data_v1_m3b \
  --model-config configs/models/m3b_priors.yaml \
  --config configs/paths.local.yaml --resume
```

Model weights are pinned by revision and SHA-256 in `configs/models/m3b_priors.yaml`, downloaded
into the ignored model cache, and never committed. Source splits (`source_train`, `source_dev`)
carry labels; `target_test` is feature-only and a training-mode selector cannot request it.

## What is not in this repository

Raw datasets (CASIA-FASD, MSU-MFSD, SiW-Mv2), SCRFD model weights, generated crop images, Parquet
manifests, NPZ priors, tar shards, generated packages, run logs and local path configuration are
intentionally excluded. They are large, license
restricted or machine-specific, and every artifact is reproducible from source with the commands
above.

## Target isolation

SiW-Mv2 is treated as an inference-only target. Its manifests carry deterministic opaque
identifiers and no label, attack, taxonomy, subject or session information; validation fails if any
private token reaches a target row.
