# PRISM-FAS-B

Local, reproducible data factory for face anti-spoofing research: read-only dataset auditing,
explicit-rule canonical adapters, and deterministic M2 preprocessing that turns raw source and
target media into face crops with strict Parquet manifests.

**Status: M2–M9 complete; M10 in progress (framework frozen, 1700-video target evaluation package built, no target label opened).** See [PROJECT_STATUS.md](PROJECT_STATUS.md).

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

## M9 regional detector, PromptHead, manifolds and the reference training run

M9 implements the PRISM detector exactly as the spec states it and trains **one** reference
configuration on source data only.

```
F_local            = ConvNeXtV2(x_rgb)                      # Atto, weight SHA 6389c2f5...7ebb
T_global, z_global = SigLIP2.image_encoder(x_rgb)           # Base P16-224, FROZEN
q_r = RegionQuery(parsing_r, landmarks_r, learnable_token_r)
z_r = CrossAttention(q_r, K=F_local, V=F_local) + RegionPool(T_global, parsing_r)
p_global = GlobalHead(z_global);  d_r = RealManifold.distance(z_r)
p_prompt = PromptHead(z_r, frozen_recipe_text_embeddings)
s_region = TopKMean(normalize(d_r), k=2)
s_final  = 1 - (1 - p_global) * (1 - s_region) * (1 - p_prompt_spoof)
```

The nine region priors are **soft masks**, never nine hard crops: the image is encoded once and the
regions are read out by prior-biased attention and prior-weighted pooling. SigLIP2 is pinned to
`google/siglip2-base-patch16-224` @ `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` with all seven file
SHA-256 values re-hashed inside the container; it is frozen and kept out of the checkpoint, bound by
its SHA identity instead. PromptHead scores the attacked regions against **cached** frozen recipe
text embeddings - a 128 x 768 uploaded artifact, so no text encoder, network call or LLM is involved
at train or inference time, and the M8 `recipe_match = not_applicable` placeholder is never used as
a target.

Training runs the spec stages **G1 -> G2 -> G5 -> G6** (G7/G8 are M10, where target data first
becomes readable). Batches are exactly 12 real live / 12 real spoof / 8 accepted-synthetic,
CASIA/MSU-balanced on both real partitions, deterministic from `seed + epoch` via SHA-256.
Prototypes are K=4 per region over the 280 `source_train` LIVE samples only (CASIA 160 / MSU 120);
running the initialization twice reproduces an identical prototype identity.

The single reference run `m9_reference_seed20260806` (seed 20260806, EMA disabled) completed all
four stages on an NVIDIA L4 with **0 non-finite losses** across all 1350 G5 steps and every batch at
the declared composition. Selected by the frozen `source_dev` ACER -> BPCER -> NLL criterion:

| source_dev (2079 rows) | ACER | APCER | BPCER | ROC-AUC |
|---|---|---|---|---|
| best checkpoint, epoch 35 | **0.136953** | 0.161406 | 0.112500 | 0.917853 |

G6 fits temperature scaling on `source_dev` alone (T 0.348756, threshold 0.837451), improving NLL
0.484347 -> 0.371227 and ECE 0.219956 -> 0.128980.

Contracts: [docs/M9_DETECTOR_CONTRACT.md](docs/M9_DETECTOR_CONTRACT.md),
[docs/M9_TRAINING_CONTRACT.md](docs/M9_TRAINING_CONTRACT.md).

**M9 validates the implementation and training of the reference detector - nothing more.** It does
not establish SiW-Mv2 performance, target APCER/BPCER/ACER, cross-domain generalization, ablation
superiority or any final research claim. `source_dev` was read only for checkpoint selection and
source calibration and produced no gradient; `target_test` was never opened. The experiment matrix
and the controlled blind target evaluation are M10.

## M10 experiment matrix and blind target evaluation (in progress)

```
target_labels_revealed: false
```

No SiW-Mv2 label has ever been opened, no target metric exists, no matrix row has been launched and
no M10 Modal app has ever been created. The framework is implemented, frozen and tested first, on
purpose: every contract that could be bent by a result is fixed before any result can exist.

Three contracts, all frozen before execution:
[docs/M10_EXPERIMENT_CONTRACT.md](docs/M10_EXPERIMENT_CONTRACT.md) (Table 59 B00-B08 as
configuration switches over one shared implementation, the Table 60 ablation deltas, the replication
policy), [docs/M10_TARGET_EVALUATION_CONTRACT.md](docs/M10_TARGET_EVALUATION_CONTRACT.md) (Table 57
prediction schema, video aggregation, PREDICTION_LOCK, the G7/G8 permission split, metric
definitions) and [docs/M10_STATISTICS_CONTRACT.md](docs/M10_STATISTICS_CONTRACT.md) (bootstrap unit,
seed, resamples, confidence level, comparison statistic, multiple-comparison policy). The target
data contract is [docs/M10_TARGET_DATA_CONTRACT.md](docs/M10_TARGET_DATA_CONTRACT.md).

```bash
python -m prism_fas.cli.main m10 matrix plan --dry-run      # plans twice, requires one identity
python -m prism_fas.cli.main m10 matrix validate
python -m prism_fas.cli.main m10 registry --action init
python -m prism_fas.cli.main m10 reliability --dry-run
python -m prism_fas.cli.main m10 report --dry-run
python -m prism_fas.cli.main m10 acceptance --dry-run
```

Every writing command supports `--dry-run`, which writes nothing, launches no Modal job and opens no
target label.

| matrix | |
|---|---|
| `m10_matrix_identity` | `a4972b0dc23946c4ad169f2c856fc9b5e0387baca45b2c9a4895f8180d9c2dd5` |
| logical rows | 42 |
| executable | 38 = 37 training + 1 bounded backend-parity run |
| new GPU training runs | 36 (B08 seed 20260806 binds the M9 reference run under an identity check) |
| blocked | 4, each carrying its reason |
| seeds | 20260806, 20260807, 20260808 |

### The target evaluation package

```
prism_target_eval_v2   1700 videos = 785 live + 915 spoof, 14/14 attack families
                       6776 frames = 6800 planned - 24 recorded no-face failures
                       identity c3a29e695ad08c4b31e01533f1d12374f4e30c51f0167c6622cf8168792e48a8
```

Additive and immutable: `prism_data_v1_m3b` (whose `target_test` split holds 785 live and **zero**
spoof videos, making APCER/ACER/HTER/ROC-AUC/EER undefined) is not modified, so every M8 and M9
identity bound to it still holds. Features and labels live in separate trees; the sealed
evaluator-only label artifact is git-ignored and has never been opened.

The primary acceptance gate passed on the full population: **3140/3140** comparable frozen live
frames reproduce their `sample_id` and `crop_sha256` byte-for-byte, 0 mismatches.

```bash
python -m prism_fas.cli.main m10 target-package inventory
python -m prism_fas.cli.main m10 target-package plan          # dry-run, writes nothing
python -m prism_fas.cli.main m10 target-package reproduce     # the byte-reproduction gate
python -m prism_fas.cli.main m10 target-package acceptance
```

**Replication is frozen by declared scientific role.** B00 and B08 carry 3 seeds because the spec
names them; B06, B07 and the four ablations that test H1/H3/H4/H5 carry 3 seeds because a declared
hypothesis rides on them; expensive diagnostic rows carry 1 seed and are reported
`single_seed_descriptive` with no interval and no superiority claim. A comparison whose either side
is a single-seed row is refused, not silently downgraded.

**The firewall is structural.** `TRAIN` cannot resolve the target feature or label roots; G7 reads
features and cannot resolve labels; G8 reads labels and cannot write `.pt`, `.pth`, `.ckpt`,
`.safetensors`, `optimizer`, `calibration/` or `checkpoints/`. Paths are compared after
`Path.resolve()`, isolation declarations are skipped structurally so the proof of isolation cannot
read as a leak, and G8's lack of a training capability is proved by an AST audit of its imports.

**Nothing is fabricated.** A metric that cannot be computed reports a status and a specific reason,
never `0.0` or `null`. A variant without a regional or prompt branch writes `null`, not zero. An
unscored report contains no target number, and a structural check enforces that.

**No claim yet.** M10 makes no SiW-Mv2, cross-domain, ablation or state-of-the-art claim, because no
target label has been opened. The next step is the additive 1700-video target evaluation package
(785 live + 915 spoof across 14 attack families at the frozen 4 frames/video) with its
evaluation-only label artifact physically and logically separated from the feature package.

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
