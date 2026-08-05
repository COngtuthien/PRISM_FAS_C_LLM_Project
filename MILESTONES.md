# Milestones

| Milestone | Objective | Expected files/tests | Acceptance | Status |
|---|---|---|---|---|
| M0 | Repo skeleton, strict config, logging and hashing | `config`, `utils`, CLI; unit tests | config resolves and tests pass | implemented/tested |
| M1 | Dataset adapters and raw audit | adapters, audit reports; unit + CLI integration | explicit rules; target isolation; reports | implemented/tested |
| M2 | Deterministic extraction, SCRFD crops, manifests | full_preprocessing_v2 manifests; 266 tests | deterministic manifests; full-profile validation passed | **completed** |
| M3 | Offline priors, packages/shards, PACKAGE_LOCK | package modules/tests | target isolation | **in progress (M3A completed)** |
| M4 | Canonical loader and balanced sampler | loader/tests | sampler checks | planned |
| M5 | B00 local baseline and source calibration | trainer/tests | target prediction report | planned |
| M6 | Modal wrapper and parity smoke test | modal modules/tests | PC/Modal parity | planned |
| M7 | Recipe compiler and physics engine | recipe modules/tests | deterministic recipes | planned |
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

M3 is IN PROGRESS.

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
| Model-dependent priors (parsing, pose, visibility, identity) | IN PROGRESS (M3B) |

## M3B components

| Component | Status |
|---|---|
| Pinned FaceXFormer parsing + pose backend | COMPLETED |
| Pinned AdaFace IR-50 identity backend | COMPLETED |
| Nine-region visibility from parsing/geometry/pose | COMPLETED |
| M3B package builder with resume | COMPLETED |
| M3B validator checks | COMPLETED |
| Real-model smoke package | COMPLETED |
| Full 6659-sample package build | IN PROGRESS |
