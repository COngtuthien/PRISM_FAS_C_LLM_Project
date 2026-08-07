# PRISM-FAS-B

Local, reproducible data factory for face anti-spoofing research: read-only dataset auditing,
explicit-rule canonical adapters, and deterministic M2 preprocessing that turns raw source and
target media into face crops with strict Parquet manifests.

**Status: M2–M8 complete; M9 not started.** See [PROJECT_STATUS.md](PROJECT_STATUS.md).

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

## M4 loader and sampler

The canonical loader reads the immutable `prism_data_v1_m3b` package through either the loose files
or the tar shards, with identical sample contracts:

```bash
python -m prism_fas.cli.main data loader inspect \
  --package-root <processed_root>/prism_data_v1_m3b --split source_train --backend loose

python -m prism_fas.cli.main data loader audit \
  --package-root <processed_root>/prism_data_v1_m3b --config configs/data/loader_m4.yaml

python -m prism_fas.cli.main data sampler audit \
  --package-root <processed_root>/prism_data_v1_m3b --config configs/data/loader_m4.yaml \
  --epochs 2 --batches 50
```

Labels are mapped explicitly (`live=0`, `spoof=1`) in `configs/data/loader_m4.yaml`. The balanced
sampler draws equal quotas from each `(dataset, label)` pool and is deterministic from the package
content identity, seed and epoch. Training mode cannot open `target_test`.

## M5 B00 local baseline

B00 is a plain ConvNeXt V2 binary classifier (live=0, spoof=1) trained only on balanced
`source_train` batches; it deliberately ignores the M3B priors and establishes the reproducible
training/calibration pipeline:

```bash
python -m prism_fas.cli.main train b00 run \
  --package-root <processed_root>/prism_data_v1_m3b \
  --config configs/train/b00_local.yaml --run-id b00_local_seed42

python -m prism_fas.cli.main train b00 calibrate \
  --run-root runs/b00_local_seed42 --package-root <processed_root>/prism_data_v1_m3b \
  --config configs/train/b00_local.yaml

python -m prism_fas.cli.main train b00 predict-target \
  --run-root runs/b00_local_seed42 --package-root <processed_root>/prism_data_v1_m3b \
  --config configs/train/b00_local.yaml

python -m prism_fas.cli.main train b00 report --run-root runs/b00_local_seed42
```

The best checkpoint, temperature and decision threshold are all chosen on `source_dev` only. Target
predictions are then produced once under that frozen calibration; target labels are never accessed,
so target score distributions are reported but target accuracy/FAS metrics are not. Runs are written
under `runs/<run_id>/` and are git-ignored.

## M6 Modal wrapper and parity

The Modal wrapper runs the *same* TrainerCore remotely — `src/prism_fas/train/**` never imports
`modal`. Remote jobs stream the package's 9 tar shards rather than its loose files.

```bash
export PYTHONIOENCODING=utf-8
python -m modal run modal_app.py --stage verify      # shard-first package validation
python -m modal run modal_app.py --stage forward     # CPU/GPU fp32 forward parity
python -m modal run modal_app.py --stage smoke       # 5-step GPU train + resume
python -m modal run modal_app.py --stage inference   # frozen-calibration inference parity
```

Volume names and mounts live in `configs/data/../cloud/modal_m6.yaml`; no credentials or local
paths are committed. M6 verifies execution-contract and numerical smoke parity, not equality of
complete training trajectories.

## M7 recipe compiler and physics engine

A recipe is a strict, dataset-agnostic description of presentation physics (schema **v1.1**). The
ontology in `configs/recipes/ontology_m7.yaml` owns every allowed value, safe range and compatibility
rule; the compiler turns a validated recipe into a deterministic operator graph, a region-mask policy
and a fixed 41-dimension conditioning vector. The frozen bank of 128 recipes is committed under
`assets/recipe_banks/prism_recipe_bank_m7_v1/` and was produced by an **offline deterministic
generator** — no external LLM, network call or credential is involved, and `prompt.txt` is frozen
alongside it as the contract a future pinned provider would receive.

```bash
python -m prism_fas.cli.main recipe build-bank \
  --ontology configs/recipes/ontology_m7.yaml --config configs/recipes/bank_m7.yaml \
  --output assets/recipe_banks/prism_recipe_bank_m7_v1
python -m prism_fas.cli.main recipe validate-bank --bank assets/recipe_banks/prism_recipe_bank_m7_v1
python -m prism_fas.cli.main recipe compile-bank  --bank assets/recipe_banks/prism_recipe_bank_m7_v1

python -m prism_fas.cli.main synthesis physics-audit \
  --package-root data/processed/prism_data_v1_m3b \
  --bank assets/recipe_banks/prism_recipe_bank_m7_v1 \
  --config configs/synthesis/physics_m7.yaml --output reports/m7
```

Rebuilding an existing bank is a no-op; a destination holding a different lock is never overwritten.
The physics engine applies the eight operators (halftone, pixel grid, moire, specular reflection,
texture smoothing, colour shift, boundary inconsistency, blur) on CPU inside deterministic
nine-region masks and composites exactly, so `max |output − input|` outside the exact edit mask is
**0**. Region mask rules are documented in
[docs/M7_REGION_MASK_MAPPING.md](docs/M7_REGION_MASK_MAPPING.md).

The audit preview runs on 32 real `source_train` **live** samples only (16 CASIA + 16 MSU); it never
opens `source_dev` or `target_test`, and uses no Modal, GPU or SSH. Its outputs under `reports/m7/`
are git-ignored preview/audit artifacts, **not** quality-gated attacks — those are produced by the
M8 quality gate below.

## M8 GPAT, quality gate and the versioned synthetic bank

M8 turns the M7 recipes into a **quality-gated synthetic bank**. Two generation routes produce the
same 1120 planned candidates: the frozen M7 `PhysicsEngine`, and **GPAT**, a 910,538-parameter Haar
wavelet residual generator whose LL band is structurally absent so low-frequency identity content
cannot be edited. Every candidate is finalized to discrete uint8 with an exact changed-pixel mask, so
`max |output − input|` outside that mask is **0**, re-proven after PNG decode.

Each candidate is scored against eight hard gates — face detection, identity cosine, landmark NME,
outside-mask parsing Dice, outside-mask error, artifact strength, high-frequency fingerprint and
support overlap — plus a quality weight `q`. **`q` is an M9 sample weight, never a live/spoof
label**, and a candidate is never rejected for a low `q` when every hard gate passes.

```bash
modal run modal_m8.py --stage calibrate                  # v1 quality calibration
modal run modal_m8.py --stage calibrate_identity_v2      # v2 identity calibration
modal run modal_m8.py --stage calibrate_structural_v3    # v3 structural calibration
modal run --detach modal_m8.py --stage generate_v3_spawn --interrupt-after 96
modal run modal_m8.py --stage export_v3 --bank-id <bank_id>

python scripts/m8_validate_downloaded_bank.py --archive <bank_id>.tar \
  --expected-archive-sha256 <sha> --expected-identity <identity>
```

Every threshold is calibrated on `source_train` **only**, and every rule was declared before the
candidates it judges were re-evaluated. The calibration population was versioned twice, and both the
superseded runs are retained rather than deleted:

| | calibration population | outcome |
|---|---|---|
| v1 | same image under ±2 % brightness/contrast | retained, missed two operational minimums |
| v2 | real same-identity cross-record pairs | retained, missed one operational minimum |
| **v3** | same image under a **localized benign appearance edit** | **all minimums met** |

Protocols: [docs/M8_QUALITY_GATE_CONTRACT.md](docs/M8_QUALITY_GATE_CONTRACT.md),
[docs/M8_IDENTITY_CALIBRATION_V2.md](docs/M8_IDENTITY_CALIBRATION_V2.md),
[docs/M8_STRUCTURAL_CALIBRATION_V3.md](docs/M8_STRUCTURAL_CALIBRATION_V3.md),
[docs/M8_GPAT_CONTRACT.md](docs/M8_GPAT_CONTRACT.md),
[docs/M8_SYNTHETIC_BANK_CONTRACT.md](docs/M8_SYNTHETIC_BANK_CONTRACT.md).

The frozen bank `prism_synthetic_bank_m8_v3_e84c78cd2a9b` holds **1120** candidates — 871 accepted,
249 rejected, 0 failed — covering all 8 artifact types and all 9 semantic regions, with both GPAT
domain relations present. It validates with 39/39 checks and 0 errors, regenerates a 32-candidate
subset with 0 mismatches, and its identity survives the archive round trip to Windows unchanged.

**M8 produces synthetic training material, not a detector result.** No detector was trained on this
bank, no target label was read, and no FAS or target-test performance is claimed.

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
