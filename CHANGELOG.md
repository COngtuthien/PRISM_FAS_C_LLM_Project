# Changelog

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
