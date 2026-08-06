# Changelog

## M7 Recipe Compiler and Physics Engine — 2026-08-06

- Added `src/prism_fas/recipes/`: strict recipe schema v1.1 (extra keys forbidden), the source-only
  ontology loader with alias canonicalization and leakage scanning, canonical JSON/hashing, twelve-stage
  validation, the deterministic compiler `m7-compiler-v1`, the fixed 41-dimension conditioning contract
  `recipe_conditioning_v1`, the offline structured generator, the frozen-bank builder/validator and the
  coverage/diversity audit.
- Added `src/prism_fas/synthesis/`: `RegionMaskBuilder` (parsing-first, geometry-fallback nine-region
  masks), the eight physics operators, the CPU `PhysicsEngine` with exact-support compositing, and the
  real source-only preview/determinism/isolation audit.
- Added `configs/recipes/ontology_m7.yaml`, `configs/recipes/bank_m7.yaml`,
  `configs/synthesis/physics_m7.yaml`, `scripts/m7_physics_audit.py`, four CLI commands
  (`recipe build-bank|validate-bank|compile-bank`, `synthesis physics-audit`) and
  `docs/M7_REGION_MASK_MAPPING.md`.
- Committed the frozen bank `assets/recipe_banks/prism_recipe_bank_m7_v1/` — 128 recipes, bank seed
  20260806, content identity `fa989938cafdc488…`, produced by an offline deterministic generator with
  **no external LLM, network access or credential**. `.gitignore` now exempts
  `assets/recipe_banks/**/*.jsonl` from the blanket `*.jsonl` rule.
- Verified the LaPa parsing semantics against the real M3B priors before writing mask code: class 8
  (inner mouth) is present on only 25/80 sampled live priors and class 5 (right eye) on 72/80, which is
  why the geometry fallback exists rather than being decorative.
- Fixed a motion-blur defect found by a focused test: integer-rounded tap offsets collapsed short
  streaks onto identical lattice shifts and silently discarded the motion angle. Taps are now sampled
  with bilinear sub-pixel interpolation.
- Fixed the bank validation report leaking an absolute machine path (`root`) into a written audit
  artifact; it now reports only the bank directory name, and the isolation scan covers all 69 written
  artifacts including every per-preview metadata file.
- Ran the real audit on CPU: 64 previews from 32 `source_train` live samples (16 CASIA + 16 MSU),
  outside-mask max abs error exactly 0.0 on every output, all 8 operators / 9 regions / 5 media /
  6 geometries / 6 illuminations exercised, two determinism reruns with 0 mismatches, and a frozen-bank
  rebuild that wrote nothing. No source_dev, no target data, no Modal, no GPU, no SSH.
- Repaired an M3-era corrupted line in `PROJECT_STATUS.md` ("left shield left shield …") back to the
  intended manifest-count sentence. Documentation-only; no data or result changed.
- Tests: 465 passed, 0 failed, 0 skipped (was 363).

## M6 Modal Wrapper and Parity Smoke — 2026-08-06

- Added `modal_app.py` and `src/prism_fas/cloud/` (config guards, parity comparators, parity fixture
  builder, shard-first remote verifier) plus `configs/cloud/modal_m6.yaml`. TrainerCore stays
  backend-neutral: a test asserts `src/prism_fas/train/**` never imports `modal`.
- Uploaded only the validated `prism_data_v1_m3b` package, the pinned ConvNeXt V2 weight and the M5
  parity artifacts to three new volumes; raw datasets were never uploaded.
- Replaced the loose-file remote validator with a shard-first `remote_parity` profile: hashing 13,337
  small files over a network Volume exceeded the container lifetime and was preempted, while the 9
  shard hashes plus sampled real triplets verify the bytes the remote trainer actually reads in 10 s.
- Fixed forward-parity drift of 1.03e-02: Ada GPUs enable TF32 for conv/matmul by default, which is
  not fp32. Disabling TF32 brought logit drift to 1.86e-05 without widening any tolerance.
- Fixed a trainer reporting bug where `batch_composition` logged cumulative counts (8/16/24/32/40)
  instead of per-batch counts; cumulative totals now live under a separate key.
- Threaded an offline `weight_file` through the model builder, trainer and inference so timm loads
  the pinned volume weight instead of reaching the Hub inside offline containers.
- Verified on NVIDIA L4: 5-step GPU smoke with exact 8/8/8/8 batches, resume to step 6, remote
  checkpoint loading on local CPU, and inference parity with 0 decision disagreements.

## M5 B00 Local Baseline — 2026-08-06

- Added `prism_fas.train`: B00 ConvNeXt V2 binary classifier, exact BCE-with-logits loss, FAS metrics
  (APCER/BPCER/ACER, ROC-AUC, EER, NLL, Brier, ECE), atomic checkpointing with strict resume guards,
  local trainer, source-dev temperature scaling, min-ACER threshold selection, blind target
  inference, video aggregation and an HTML/JSON report.
- Added `configs/train/b00_local.yaml` pinning the model (`convnextv2_atto.fcmae_ft_in1k`, weight
  SHA-256), optimizer groups, cosine schedule, selection rule and calibration policy.
- Added `train b00 smoke|run|calibrate|predict-target|report` CLI commands with dry-run support.
- Ran the real pipeline on CPU: 5-step smoke with verified resume, then 8 epochs / 360 steps
  (early-stopped), best epoch 2 by source_dev ROC-AUC 0.98312; temperature 2.5385 improved source_dev
  NLL 0.2938 → 0.1596; min-ACER threshold 0.34145 gives APCER 0.0322 / BPCER 0.0925 / ACER 0.0623;
  2079 source_dev and 3140 blind target predictions written.
- Target labels were never accessed: no target metrics are reported and no target signal influenced
  checkpoint selection, temperature or threshold.

## M4 Canonical Loader and Balanced Sampler — 2026-08-05

- Added `prism_fas.data.loader`: validated package index, `CanonicalSourceSample`/`CanonicalTargetSample`
  contracts, loose-file and tar-shard datasets, deterministic image/prior transforms, separate source
  and target collate paths, and `BalancedDomainClassBatchSampler`.
- Added `configs/data/loader_m4.yaml` with an explicit `live=0 / spoof=1` label mapping, image decode
  contract, sampler policy and DataLoader defaults; no absolute paths.
- Added `data loader inspect`, `data loader audit` and `data sampler audit` CLI commands.
- Audited the real `prism_data_v1_m3b` package: 6659 samples scanned on both backends with matching
  sample-ID sets and 0 parity mismatches; sampler produced exact 8/8/8/8 pool quotas over 45 steps per
  epoch with 0 duplicate IDs and 0 repeated records per batch.
- DataLoader worker smoke passed at `num_workers=0` and `num_workers=2` on Windows.

## M3B Model-Dependent Priors — 2026-08-05

- Added `prism_fas.data.package.model_priors` and `m3b`: pinned FaceXFormer parsing/pose backend,
  AdaFace IR-50 identity backend, parsing+geometry+pose visibility derivation, an M3B package
  builder with resume, an extended priors index and a strict model-prior failure manifest.
- Added `configs/models/m3b_priors.yaml` pinning backend revisions, weight SHA-256 values, input
  normalization, parsing class mapping, pose convention and the nine-region visibility order.
- Added `data priors model-build` CLI plus M3B validator checks (parsing/pose/visibility coverage,
  identity scope and norms, parent package identity, image SHA stability).
- Third-party model code is fetched into the ignored model cache rather than vendored; weights are
  verified against their pinned SHA-256 on every load.
- Real-model smoke package validated: 39 samples, parsing/pose/visibility complete, 9 identity
  embeddings on source_train live only, 0 on target, 0 failures.
- Built the full `prism_data_v1_m3b` package from all 6659 M3A samples on CPU: parsing 6659,
  pose 6659, visibility 6659, identity 280 (derived source_train-live count), 0 model-prior
  failures, 9 shards totalling 489,779,200 bytes. Validation passed with 59/59 checks.
- Fixed the package content-identity contract: the promoted lock hashed the wall-clock
  `build_seconds` field, so rebuilding byte-identical artifacts changed `content_identity_sha256`.
  Wall-clock and host-dependent fields are now excluded, proven by two consecutive `--resume`
  reruns that reused 6659 priors, rebuilt 0 and produced an identical identity.

## M3A Package Foundation — 2026-08-05

- Added `prism_fas.data.package`: deterministic quality priors, per-sample NPZ prior builder with
  resume and atomic writes, canonical split manifests, WebDataset-compatible tar shards,
  PACKAGE_LOCK with a content identity hash, a 42-check package validator and a training-mode
  target-isolation selector.
- Added `data priors build`, `data package build` and `data package validate` CLI commands plus
  `configs/data/package_m3a.yaml` (no local absolute paths).
- Built `prism_data_v1_m3a` from the frozen `full_preprocessing_v2` M2 namespace: 6659 samples,
  6659 priors, 6659 images, 9 shards (150,220,800 bytes), splits 1440/2079/3140.
- Package validation passed with 42/42 checks; a `--resume` rerun reused 6659 priors and produced
  byte-identical shards, manifests and lock identity.
- Model-dependent priors (parsing masks, pose, visibility, identity embeddings) are deferred and
  recorded as `not_computed`; no placeholder arrays are written.

## M2 Full Preprocessing Checkpoint — 2026-08-05

- Switched the production full-profile CLI to the context-aware `run_preprocessing()` runner; the
  legacy small-acceptance path is unchanged.
- Completed source-success, source-failure and target-success routing behind one shared
  media → detector → geometry → crop → write → SHA-256 pipeline.
- Added an atomic crop writer with partial-artifact cleanup, a typed `OutputWriteError` boundary and
  a typed `HashComputationError` boundary with an unexpected-exception propagation guard.
- Fixed CASIA canonical identity: `train/` and `test/` shared `s<subject>v<video>` stems, which
  collapsed into one `video_id` and produced conflicting duplicate sample IDs. The official split is
  now part of the identity, with a uniqueness guard.
- Fixed crop `source_media_type`, which was hardcoded to `image_sequence` for every crop record; it
  is now an explicit typed input, so video sources record `video_file` on both frame and crop rows.
- Replaced raw SiW-Mv2 filename-derived identifiers with deterministic opaque `siw_<16 hex>` IDs
  derived from the dataset-root-relative path, with collision detection.
- Added a `full_preprocessing` validation profile that checks structural consistency, crop
  integrity, failure-record validity and target isolation without small-acceptance row-count
  constants or legacy completed-index/run-state/M2A artifacts.
- Made the atomic Parquet replace tolerant of transient Windows sharing violations.
- Ran full preprocessing into `full_preprocessing_v2`: 1665 canonical records, 6660 samples selected,
  6659 successful, 1 failed (`no_face`), 3519 source frames/crops, 3140 target frames/crops,
  6659 crop files, 0 missing/orphan/SHA/duplicate/dimension/temporary issues.
- Full-profile validation passed (35/35 checks); test suite at 266 passed.

## M2B2

- Added production M2 validation/status CLI commands and final small-acceptance reports.

## M2B1a

- Added strict source/target/failure Parquet manifests, target-isolation validation, atomic deterministic writer, and M2A migration.

## M2B1b

- Added completed-sample index, atomic run state, Windows-safe output lock and resume verification coverage.

## 0.1.0

- Added M0/M1 local infrastructure, explicit YAML adapters, raw audits, and tests.
- Resolved MSU-MFSD mapping from its README and official train/test subject lists.
- Added M2 configuration, deterministic sampling/IDs, media helpers, crop policy, SCRFD ONNX interface, CLI stubs and synthetic tests.
- Added source-only SCRFD 256/320 validation script and froze 320 input policy.
- Added M2A production image-sequence/video readers and small-run crop/JSONL runner.
