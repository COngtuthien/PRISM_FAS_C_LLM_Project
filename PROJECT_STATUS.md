# Project status

- Current milestone: **M5 COMPLETED** — B00 local baseline, source calibration and blind target prediction. M6 is NOT STARTED.
- M0: COMPLETED; M1: COMPLETED; M2: COMPLETED; M3: COMPLETED; M4: COMPLETED; M5: COMPLETED; M6: NOT STARTED.
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

left shield left shield left shield one seminar thecrops 3519`, `target_frames 3140`,
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
- Next milestone: M6 — Modal wrapper and parity smoke test (NOT STARTED).

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
