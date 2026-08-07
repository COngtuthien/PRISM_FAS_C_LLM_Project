# Milestones

| Milestone | Objective | Expected files/tests | Acceptance | Status |
|---|---|---|---|---|
| M0 | Repo skeleton, strict config, logging and hashing | `config`, `utils`, CLI; unit tests | config resolves and tests pass | implemented/tested |
| M1 | Dataset adapters and raw audit | adapters, audit reports; unit + CLI integration | explicit rules; target isolation; reports | implemented/tested |
| M2 | Deterministic extraction, SCRFD crops, manifests | full_preprocessing_v2 manifests; 266 tests | deterministic manifests; full-profile validation passed | **completed** |
| M3 | Offline priors, packages/shards, PACKAGE_LOCK | package modules/tests | target isolation | **completed** |
| M4 | Canonical loader and balanced sampler | loader/tests | sampler checks | **completed** |
| M5 | B00 local baseline and source calibration | trainer/tests | target prediction report | **completed** |
| M6 | Modal wrapper and parity smoke test | modal modules/tests | PC/Modal parity | **completed** |
| M7 | Recipe compiler and physics engine | recipe modules/tests | deterministic recipes | **completed** |
| M8 | GPAT and synthetic bank | synthetic modules/tests | versioned bank | planned |
| M9 | Regional CNN–ViT fusion/manifolds/losses | model modules/tests | loss checks | planned |
| M10 | Experiment matrix and report | experiments/tests | reproducible report | planned |

Each milestone requires: objective, implementation tasks, expected files, unit/integration tests, a command, and an acceptance report. Detailed task expansion occurs only when that milestone is assigned.

## M2 components (all completed)

| Component | Status |
|---|---|
| Production CLI switch to the context-aware runner | COMPLETED |
| Source-success routing (frame + crop manifests) | COMPLETED |
| Source-failure routing (no_face, frame_index_unavailable, unreadable_image, decode_failed, detector_failed, invalid_bbox, invalid_landmarks, crop_failed, output_write_failed, hash_failed) | COMPLETED |
| Target-success routing | COMPLETED |
| Opaque target identifiers (`siw_<16 hex>`) | COMPLETED |
| Atomic crop writer and typed SHA-256 boundary | COMPLETED |
| Real smoke preprocessing (3 records per dataset) | COMPLETED |
| Full preprocessing v2 (1665 records, 6659 crops) | COMPLETED |
| Full-profile validation (35/35 checks) | COMPLETED |
| Target privacy isolation (0 matches) | COMPLETED |

## M3A components (completed)

| Component | Status |
|---|---|
| Deterministic quality priors (blur/brightness/contrast/saturation/face-size) | COMPLETED |
| Per-sample NPZ priors with resume and atomic writes | COMPLETED |
| Canonical split manifests from official source metadata | COMPLETED |
| Target feature manifest (label-free) | COMPLETED |
| WebDataset-compatible deterministic tar shards | COMPLETED |
| PACKAGE_LOCK with content identity hash | COMPLETED |
| Package validator (42 checks) | COMPLETED |
| Training-mode target isolation guard | COMPLETED |
| Real smoke package + full 6659-sample package | COMPLETED |
| Model-dependent priors (parsing, pose, visibility, identity) | COMPLETED (M3B) |

## M3B components

| Component | Status |
|---|---|
| Pinned FaceXFormer parsing + pose backend | COMPLETED |
| Pinned AdaFace IR-50 identity backend | COMPLETED |
| Nine-region visibility from parsing/geometry/pose | COMPLETED |
| M3B package builder with resume | COMPLETED |
| M3B validator checks | COMPLETED |
| Real-model smoke package | COMPLETED |
| Full 6659-sample package build | COMPLETED |
| Full package validation (59/59 checks) | COMPLETED |
| Resume idempotency (6659 reused, 0 rebuilt) | COMPLETED |

M3 is COMPLETED.

## M4 components (completed)

| Component | Status |
|---|---|
| Canonical source/target sample contracts | COMPLETED |
| Loose-file package dataset | COMPLETED |
| Tar-shard streaming dataset | COMPLETED |
| Loose/shard parity | COMPLETED |
| Source and target collate contracts | COMPLETED |
| Balanced domain/class batch sampler | COMPLETED |
| Deterministic seed/epoch replay | COMPLETED |
| Target isolation guards | COMPLETED |
| Full 6659-sample loose and shard audits | COMPLETED |
| DataLoader worker smoke (0 and 2 workers) | COMPLETED |

M4 is COMPLETED.

## M5 components (completed)

| Component | Status |
|---|---|
| B00 ConvNeXt V2 binary classifier | COMPLETED |
| Exact BCE-with-logits loss | COMPLETED |
| Local trainer with checkpoint/resume | COMPLETED |
| Real 5-step smoke with resume | COMPLETED |
| Full local training (8 epochs, early stop) | COMPLETED |
| Best checkpoint by source_dev ROC-AUC | COMPLETED |
| source_dev temperature scaling | COMPLETED |
| source_dev min-ACER threshold | COMPLETED |
| source_dev predictions (2079) and metrics | COMPLETED |
| Blind target predictions (3140) | COMPLETED |
| HTML/JSON report and COMPLETE.json | COMPLETED |

M5 is COMPLETED.

## M6 components (completed)

| Component | Status |
|---|---|
| Modal app with volume mounts and GPU allow-list | COMPLETED |
| Shard-first remote package verification (`remote_parity`) | COMPLETED |
| L4 GPU environment probe with CUDA assertion | COMPLETED |
| CPU/GPU fp32 forward parity (TF32 disabled) | COMPLETED |
| 5-step GPU training smoke + resume to step 6 | COMPLETED |
| Remote checkpoint download and local CPU portability | COMPLETED |
| Inference parity under frozen calibration | COMPLETED |
| Reusable Modal training entrypoint (not run fully) | COMPLETED |

M6 is COMPLETED.

## M7 components (completed)

| Component | Status |
|---|---|
| Strict recipe schema v1.1 (unknown keys rejected) | COMPLETED |
| Source-only ontology `m7-ontology-v1` with visible compatibility tables | COMPLETED |
| Rule validation, severity budget and source-only leakage guards | COMPLETED |
| Deterministic recipe compiler `m7-compiler-v1` | COMPLETED |
| Fixed 41-dimension conditioning contract `recipe_conditioning_v1` | COMPLETED |
| Frozen 128-recipe bank `prism_recipe_bank_m7_v1` with BANK_LOCK | COMPLETED |
| Offline coverage and TF-IDF diversity audit (no external text model) | COMPLETED |
| Deterministic nine-region mask builder (parsing-first, geometry-fallback) | COMPLETED |
| All eight physics operators | COMPLETED |
| CPU physics engine with exact outside-mask preservation | COMPLETED |
| Real 64-preview source_train-live audit (16 CASIA + 16 MSU) | COMPLETED |
| Determinism rerun with zero mismatches | COMPLETED |

M7 is COMPLETED.

## M8 components (completed)

| Component | Status |
|---|---|
| Deterministic source-only GPAT pair plan (896 train / 224 validation) | COMPLETED |
| Differentiable Haar DWT/IDWT with a structurally absent ΔLL band | COMPLETED |
| GPAT residual model (910,538 parameters) and the declared loss set | COMPLETED |
| GPAT trainer, strict checkpoint/resume and the Modal M8 wrapper | COMPLETED |
| Full GPAT training on L4 (15 epochs, best epoch 11) | COMPLETED |
| Pinned SCRFD / FaceXFormer / AdaFace quality-model registry | COMPLETED |
| Quality calibration **v1** (same-image mild photometric population) | COMPLETED — retained, failed two minimums |
| Identity calibration **v2** (real cross-record genuine/impostor pairs) | COMPLETED — retained, failed one minimum |
| Structural calibration **v3** (same-image localized benign appearance edits) | COMPLETED |
| 24-d high-frequency fingerprint reference, leave-one-record-out `tau_fp` | COMPLETED |
| Quality gate: 8 hard gates, `q` weight, `recipe_match = not_applicable` | COMPLETED |
| Deterministic 1120-candidate plan (560 physics + 560 GPAT) | COMPLETED |
| Discrete uint8 finalization with exact masks and zero outside-mask error | COMPLETED |
| Resume-safe generation of all 1120 candidates, both routes | COMPLETED |
| Operational minimums (all nine) | COMPLETED — met under v3 |
| Versioned bank layout, deterministic shards and BANK_LOCK | COMPLETED |
| Full bank validation (39/39 checks, 0 errors) | COMPLETED |
| Resume audit and 32-candidate determinism audit (0 mismatches) | COMPLETED |
| Remote freeze, transport archive, Windows download and local re-validation | COMPLETED |

M8 is COMPLETED: the frozen bank is `prism_synthetic_bank_m8_v3_e84c78cd2a9b`
(`e84c78cd2a9b548244e243de0380998d04bc6770b91caf32ac7be96f489bb542`, status `validated`,
1120 candidates = 871 accepted + 249 rejected + 0 failed).

M8 delivers **synthetic training material and a sample weight**, not a detector result. No detector
was trained on it and no target label was read, so no FAS or target-test claim is made.

Next milestone: M9 — PromptHead and detector training (NOT STARTED).
