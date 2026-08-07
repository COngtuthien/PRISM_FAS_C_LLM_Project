# Project status

- Current milestone: **M8 COMPLETED** — GPAT residual generator, source-only quality calibration
  (three frozen versions), the 1120-candidate quality gate and the frozen versioned synthetic bank
  `prism_synthetic_bank_m8_v3_e84c78cd2a9b`. **M9 is NOT STARTED.**
- M0–M8: COMPLETED; M9: NOT STARTED.
- M3A package foundation and deterministic quality priors: **COMPLETED**.
- M3B model-dependent priors: **COMPLETED**.
- Official package for downstream work: `prism_data_v1_m3b` (parent `prism_data_v1_m3a`, immutable).

## M3B model priors (prism_data_v1_m3b)

Parent package `prism_data_v1_m3a` (immutable). Backends, pinned by revision and weight SHA-256 in
`configs/models/m3b_priors.yaml`:

| Prior | Backend | Revision |
|---|---|---|
| parsing (11-class LaPa) | FaceXFormer | `fd12148d0b19` |
| pose (yaw/pitch/roll, radians) | FaceXFormer | `fd12148d0b19` |
| visibility (9 regions) | derived from parsing + geometry + pose | `m3b-visibility-v1` |
| identity (512-d, L2 normalized) | AdaFace IR-50 (WebFace4M) | `60a65befbcf7` |

Identity embeddings are computed only for `source_train` samples labelled `live`; every other sample
records `identity_status="not_applicable"` and carries no embedding array. Target priors never
receive an identity embedding.

### M3B final result

| Prior | Computed | Failures |
|---|---|---|
| parsing | 6659 | 0 |
| pose | 6659 | 0 |
| visibility | 6659 | 0 |
| identity | 280 | 0 |

Identity applicable count (`source_train` AND `live`) derived from the manifests is **280**;
`identity_not_applicable` is 6379; target identity embeddings **0**; source_dev identity **0**.
Splits unchanged from the parent: source_train 1440, source_dev 2079, target_test 3140 (6659 total).
9 shards, 489,779,200 bytes. Package validation **passed = true**, 0 errors, 59/59 checks,
`PACKAGE_LOCK.status = validated`, target isolation clean.

Parent content identity `a968caeb8e6e55a2afdba724923073161d2315e33c57733cf1be2b967b469769`;
M3B content identity `b1cf29b69a165ed5d9e074fc8127c17fbf057723edf9e272048ec3a564eb9dc6`.
Two consecutive `--resume` reruns reused all 6659 priors, rebuilt 0, and left every manifest, shard
and prior byte-identical with an unchanged content identity. Tests: 298 passed.

## M3A package (prism_data_v1_m3a)

Official package root: `<processed_root>/prism_data_v1_m3a`, built from the frozen
`full_preprocessing_v2` M2 namespace.

| Split | Samples |
|---|---|
| source_train | 1440 |
| source_dev | 2079 |
| target_test | 3140 |
| **Total** | **6659** |

6659 images, 6659 NPZ priors, 9 uncompressed tar shards (150,220,800 bytes), package validation
**passed = true** with 42/42 checks and target isolation clean. A `--resume` rerun reused all 6659
priors and reproduced byte-identical shards, manifests and lock identity.
Generated packages are git-ignored and are not pushed to GitHub.
- Official M2 dataset namespace: `full_preprocessing_v2`, written under
  `<work_root>/m2/<preprocessing_version>/<preprocessing_config_hash>/full_preprocessing_v2`.
  The earlier `full_preprocessing` namespace is retained locally as an audit artifact and must not be modified.

## Final M2 execution (full_preprocessing_v2)

| Dataset | Role | Records | Selected | Successful | Failed |
|---|---|---|---|---|---|
| casia_fasd | source | 600 | 2400 | 2399 | 1 |
| msu_mfsd | source | 280 | 1120 | 1120 | 0 |
| siw_mv2 | target | 785 | 3140 | 3140 | 0 |
| **Total** | | **1665** | **6660** | **6659** | **1** |

Manifest counts: `source_frames 3519`, `source_crops 3519`, `target_frames 3140`,
`target_crops 3140`, `preprocessing_failures 1`; 6659 crop files on disk.

Reconciliation: `selected = successful + failed` and `successful = source_crops + target_crops`.

The single failure is a genuine detector outcome (`casia_fasd`, `error_code=no_face`,
`stage=detector`, `recoverable=true`), routed to the failure manifest with the run continuing.

## Validation

- Full-profile validation: **passed = true**, 0 errors, 35/35 checks.
- Missing crops 0, orphan crops 0, SHA mismatches 0, duplicate sample IDs 0,
  unreadable crops 0, wrong dimensions 0, temporary artifacts 0.
- Target privacy/isolation: 0 matches.
- Tests: `266 passed, 0 failed, 0 skipped` via `python -m pytest -q`.

## Target isolation

SiW-Mv2 target identifiers are opaque `siw_<16 hex>` values derived deterministically from the
dataset-root-relative path, so no class, subject, session or raw filename information reaches the
target manifests. Target rows carry no label, attack, taxonomy, subject or session field.

## Data policy

Generated data is intentionally **not** committed to GitHub: raw datasets, model weights,
crop images, Parquet manifests, run logs and local path configuration are ignored. The dataset is
reproducible from source using the documented CLI and the frozen preprocessing configuration.

- Blockers: none.
- Next milestone: M9 — PromptHead and detector training (NOT STARTED).

## M8 GPAT, quality calibration and the versioned synthetic bank

Source-only throughout: every M8 stage opens `manifests/source_train.parquet`, `source_train` images
and priors, the frozen M7 recipe bank and the pinned quality weights, and nothing else.
`source_dev_opened false`, `target_test_opened false`, `raw_dataset_path_opened false` on every run.

### Frozen inputs

| Item | Value |
|---|---|
| Source package | `prism_data_v1_m3b`, `b1cf29b69a165ed5d9e074fc8127c17fbf057723edf9e272048ec3a564eb9dc6` |
| M7 recipe bank | `fa989938cafdc4887518cc45c35d559d00278358439dc68c2486da10309210cb` |
| GPAT pair plan | `301868301dd11739ec018eed438704f9e4da7896ea52a0e60d50de563f2ccad3` (896 train / 224 validation) |
| GPAT checkpoint | epoch 11, `2047cdb513767010cfdf368c6f53a3664922451c56e1e837ec59cb96918a5b63` |
| Candidate plan | `b167c169dcb92426c0dc2ee96a80eb69f4645fbf887360a1b67abfc8890f40b8` — **1120** = 560 physics + 560 GPAT |

GPAT is a 910,538-parameter Haar-DWT residual generator whose LL band is structurally absent, trained
15 epochs on L4; best `validation_total_loss` 0.048736, `validation_identity_cosine` 0.999893.

### Quality calibration history — three frozen versions, honestly recorded

The calibration population was versioned twice. **Both revisions changed a population, not a
threshold**, and each rule was declared before any candidate was re-evaluated under it.

| | population | result |
|---|---|---|
| **v1** | same image under ±2 % brightness/contrast, noise 0.002 | 391 accepted; **failed** `accepted_total` (391 < 400) and `accepted_physics` (71 < 200) |
| **v2** | real same-identity cross-record pairs (560 genuine / 13440 impostor) | `tau_id` 0.99952 → **0.547440037939055**; identity rejections fell to **0**; 475 accepted; physics 151, so `accepted_physics` still **failed**; `landmark` (467) and `parsing_dice` (233) became the blockers |
| **v3** | same image under a **localized benign appearance edit** (280 live × 8 frozen transforms = 2240 observations) | `tau_lm` → **0.00836817528937794**, `tau_parse` → **0.7094826178704915**; **871 accepted, every minimum met** |

v3 versions only `tau_lm` and `tau_parse`. `tau_fd` (0.5), `tau_id` (the v2 value), `tau_out` (0),
`tau_fp` (5.687657785453908), the fingerprint reference, the artifact-strength and support-overlap
rules, the `q` formula and every operational minimum are carried over unchanged. The v1 and v2 runs,
reports, locks, namespaces and archives are retained unmodified and still read
`operational_minimums_failed`. Protocols: `docs/M8_IDENTITY_CALIBRATION_V2.md`,
`docs/M8_STRUCTURAL_CALIBRATION_V3.md`.

The v3 calibration measured 2240/2240 valid landmark comparisons, 0 detection failures, 0 region-mask
failures and an outside-support uint8 error of exactly 0 on every observation, and reproduced with
**0 mismatches** across two runs. The 560 cross-record genuine pairs were computed as a **diagnostic
only** and set no threshold: their landmark NME median is 0.0828 against 0.0013 for the same-image
population, so they measure real pose/expression/crop variation rather than detector jitter.

### The frozen bank

| Item | Value |
|---|---|
| Bank id | `prism_synthetic_bank_m8_v3_e84c78cd2a9b` |
| Content identity | `e84c78cd2a9b548244e243de0380998d04bc6770b91caf32ac7be96f489bb542` |
| Status | **`validated`** (derived from the operational-minimum result, never unconditional) |
| Candidates | **1120** = 871 accepted + 249 rejected + **0** failed |
| Accepted by route | physics **419** (min 200), gpat **452** (min 100) |
| Accepted by domain | CASIA **491**, MSU **380** (min 100 each) |
| Coverage | **8/8** artifact types, **9/9** semantic regions; gpat same-domain 226 / cross-domain 226 |
| `q` | min 0.2174, median 0.7717, mean 0.7544, max 0.9421 |
| Threshold SHA | `8fa2648643cd526730497ae2d717e17684dda3ecea361fc84929db07ac03bb19` |
| Shards | 2, loose/shard parity verified |
| Frozen path | `prism-fas-b-data:/synthetic_banks/prism_synthetic_bank_m8_v3_e84c78cd2a9b` |

Remaining rejections are dominated by the **unchanged** `artifact_strength` gate (158), then
`landmark` 86, `parsing_dice` 23, `fingerprint` 6, `identity` **0**. Relaxing only the landmark and
parsing gates cannot reject a previously accepted candidate, and the measured comparison confirms
**0 accepted → rejected** transitions in either route.

### Verification

- bank validation **39/39 checks, 0 errors**, including exact outside-mask uint8 error 0, artifact
  maps finite in [0,1] and exactly 0 outside the exact mask, NPZ loaded with `allow_pickle=False`,
  every accepted row passing every hard gate and every rejected row naming its failed gates
- resume audit **passed**: interrupted after 96 rebuilds, resumed reusing 95, the deliberately
  truncated candidate detected by hash and rebuilt byte-identically, completed rerun **1120 examined /
  1120 reused / 0 rebuilt**
- determinism audit **passed**: 32 candidates regenerated in a separate namespace, **0 mismatches**
- payload provenance: **475 payloads reused byte-identically** from the retained v2 run and 645
  regenerated deterministically under their original ids, inputs and seeds — payload recovery, never
  candidate replacement
- Windows round trip: archive **126,689,280 bytes**, SHA-256
  `38a92fda0c7ae004f092624d99bfb0b46605484aa433b8fe91a47527e68d0db2`, 3501 members; downloaded,
  extracted and re-validated locally with **local bank identity equal to the remote identity** and
  0 errors

`q` is an M9 sample weight, not a live/spoof label. **M8 makes no claim about FAS detector quality or
target-test performance**: no detector was trained on this bank and no target label was ever read.

## M7 recipe compiler and physics engine

Strict recipe schema **v1.1** (`recipe_id`, `medium`, `geometry`, `regions`, `artifacts`, `capture`,
`forbidden_shortcuts`, `generator_route`, `seed`, `schema_version`; unknown keys rejected). Ontology
`m7-ontology-v1`, SHA-256 `90694441c2ef1477ca8f6c4dd724a4997a3e166cbf5a067d52c101892f952bbd`, holds
every allowed value, safe range, medium/artifact and geometry/region compatibility table and the
alias map. Engineering safety defaults: max 3 artifacts, max 3 regions, max total artifact strength
1.0.

| Item | Value |
|---|---|
| Frozen bank | `assets/recipe_banks/prism_recipe_bank_m7_v1` (`status = frozen`) |
| Bank content identity | `fa989938cafdc4887518cc45c35d559d00278358439dc68c2486da10309210cb` |
| Recipes | 128 (`R-000001`–`R-000128`), bank seed `20260806`, 0 duplicate recipe or graph hashes |
| Generator | `deterministic_local` / `deterministic-source-only-recipe-generator` / `m7-v1`; **no external LLM, no network, no credential** |
| Compiler | `m7-compiler-v1`, graph schema `m7-graph-v1`, 128/128 compile, 128 unique graph hashes |
| Conditioning | `recipe_conditioning_v1`, fixed **41** float32 dims (5+6+9+8+6+5+2) |
| Coverage | all 5 media, 6 geometries, 9 regions, 8 artifacts, 6 illuminations; physics route in every recipe; artifact counts 1/2/3 and single- and multi-region all present |
| Diversity | offline TF-IDF cosine (no text model, no network): max pairwise 0.4875, mean 0.2436, thresholds 0.98 / 0.90 |

Physics: nine deterministic semantic-region masks (parsing-first, geometry-fallback, mask policy
`m7-mask-v1`) and all eight operators — halftone, pixel_grid, moire, specular_reflection,
texture_smoothing, color_shift, boundary_inconsistency, blur — on CPU with local PCG64 generators
only.

| Real audit stage | Result |
|---|---|
| Preview rows | 64 = 32 source_train live samples x 2 recipes |
| Inputs | 16 CASIA + 16 MSU source_train live; source_dev 0; target 0 |
| Outside-mask max abs error | **exactly 0.0** on every one of the 64 outputs |
| Masks | no empty masks; every preview changed pixels inside the mask |
| Exercised | 8/8 operators, 9/9 regions, 5/5 media, 6/6 geometries, 6/6 illuminations |
| Determinism rerun | 2 independent reruns, **0 mismatches**; identical image/mask/strength/graph hashes |
| Seed sensitivity | changing only the recipe seed changed the graph hash and the output (max abs diff 0.2542) |
| Frozen-bank rebuild | `reused`, 0 files written, BANK_LOCK unchanged |
| Package | `prism_data_v1_m3b` identity unchanged and never written |

Device **CPU**; no Modal job, no GPU and no SSH were used in M7. The previews are audit artifacts,
not quality-gated attacks: the M8 quality gate has not run. M8 (GPAT, quality gate, versioned
synthetic bank) is NOT STARTED.

## M6 Modal wrapper and parity

Modal CLI/SDK 1.5.3 in workspace `hysonlab-weather-forecast`; volumes `prism-fas-b-data`,
`prism-fas-b-models`, `prism-fas-b-runs`. The wrapper calls the same TrainerCore; remote jobs use the
shard backend (9 tars) rather than 13,337 loose files.

| Stage | Result |
|---|---|
| Remote package validation (`remote_parity`) | passed, 58/58 checks, 10.2 s |
| GPU | NVIDIA L4, 23.66 GB, CUDA 12.1, cuDNN 90100 |
| Forward parity (fp32, TF32 off) | logits max 1.86e-05 (tol 1e-3), probs 6.94e-07 (tol 2.5e-4), BCE 7.15e-07 (tol 5e-4), feature cosine 1.000000000 |
| Training smoke | 5 steps then resumed to 6; every batch 8/8/8/8; losses finite |
| Checkpoint portability | remote `last.pt` loads on local CPU at step 6, forward finite |
| Inference parity | 16 target rows, frozen calibration, 0 decision disagreements |
| Target isolation | no labels or private fields in any remote target row |

M6 verifies execution-contract and numerical smoke parity, **not** equality of complete training
trajectories. Remote package validation is a shard-first transfer-integrity profile; the exhaustive
59/59 loose-file validator remains local and unchanged.

## M5 B00 local baseline

ConvNeXt V2 Atto (`convnextv2_atto.fcmae_ft_in1k`, timm, weight SHA `6389c2f5…b7ebb`, 3.39M backbone
params, 320-d features) trained only on domain/class-balanced `source_train` batches with mean BCE
with logits. Device: **CPU** (no CUDA on this host).

| Stage | Result |
|---|---|
| Training | 8 epochs, 360 steps, early-stopped on patience 5 |
| Best checkpoint | epoch 2, selected by max source_dev ROC-AUC (tie-break min NLL) |
| source_dev ROC-AUC | 0.98312 |
| Temperature (source_dev only) | 2.5385 — NLL 0.2938 → 0.1596, ECE 0.0460 → 0.0393 |
| Threshold (source_dev min-ACER) | 0.34145 |
| source_dev frame metrics | APCER 0.0322, BPCER 0.0925, ACER 0.0623, accuracy 0.9562, EER 0.0650 |
| source_dev predictions | 2079 frames / 520 videos |
| target predictions | 3140 frames / 785 videos, blind |

Target labels were not accessed: no target metrics, and no target signal entered checkpoint
selection, temperature or threshold. B00 deliberately ignores the M3B parsing/pose/visibility/identity
priors; `reject_policy = disabled_for_b00`.

## M4 loader and sampler

Reads the immutable `prism_data_v1_m3b` package (never mutates it). Label mapping is explicit:
`live = 0`, `spoof = 1` (spoof is the positive attack class).

- Loose-file and tar-shard backends produce the same `CanonicalSourceSample` / `CanonicalTargetSample`
  contracts; full scans covered **6659** samples on both backends (source_train 1440, source_dev 2079,
  target_test 3140) with identical sample-ID sets and 0 parity mismatches.
- `BalancedDomainClassBatchSampler` draws an equal quota from each `(dataset, label)` pool:
  batch 32 = 8 casia live + 8 casia spoof + 8 msu live + 8 msu spoof, 45 steps per epoch
  (`ceil(1440/32)`). Deterministic from package content identity + seed + epoch; replay matches and
  epoch 0 differs from epoch 1.
- Target isolation: `target_test` cannot be opened in training mode, the sampler rejects any
  non-`source_train` split, target samples carry no label/identity, and target batches expose no
  training fields. Measured target identity availability: 0.
- Continue command: `python -m pytest -q`
