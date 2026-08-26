# C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V1 — protocol freeze

**NOT A BA_sep REVISION. NOT A RELIABILITY-BARRIER RESCUE PROTOCOL. NOT A C9
PASS PROTOCOL. NOT A TARGET PROTOCOL.**

`synthetic_vs_real_spoof_probe` has already, permanently, FAILED under
`C9_DETECTOR_BA_SEP_OPTION1_V2` (protocol identity
`720a2e344017d588d71005b81fdf0e7d2062081ae2f3881a61a306d952dc4ac8`; observed
`BA_sep_RND=0.7843079833902619`, `BA_sep_DET=0.8514170182841069`,
`BA_sep_LLM=0.7902658339472685`, all over the 0.75 ceiling). The user-reported
GPU registration of that result (`DETECTOR_RELIABILITY_LOCK_C.json`,
`identity_sha256 = 40825a5fffcbbdd681e5d8b0354e8371dcccabc36b51d1b9a12cd8bd29e73fbe`,
`overall = FAILED`, `c9_may_close = false`) is permanent, immutable
scientific evidence. This document freezes a NEW, separately identified
protocol — **`C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V1`**,
`configs/evaluation/c9_post_failure_source_diagnostics_v1.yaml`,
identity `cb05271e26d9a421f2f9277599523e185026e1eab644febc07c75432d26f3fc5`
— for bounded, source-only, MECHANISTIC diagnostics. **It does not run
them.**

---

## 1. What this protocol cannot do

Nothing in this protocol, its implementation, or its future execution can:
change `BA_sep`, overwrite the BA_sep result files, turn the current
reliability barrier PASS, reopen current C9, authorize target access, alter
the C8 matrix, retrain C8, select a better seed, change the 0.75 ceiling,
change the V2 BA_sep protocol, create BA_sep V3, alter the C6 banks, or
cherry-pick checkpoints. Every runner artifact this protocol produces
explicitly records `c9_may_close: false` and the BA_sep observed verdict as
`FAIL` — hard-coded, never derived from a diagnostic outcome.

## 2. Re-audit method

Before writing this protocol, the actual code was re-read: detector
contracts (`src/prism_fas/detector/contracts.py`), the model's forward pass
(`src/prism_fas/detector/prism_detector.py`), the loss module
(`src/prism_fas/detector/losses.py`), C7's track flags
(`src/prism_fas/pipeline/adapters/c7.py`), the C6 matched-bank row schema
(`src/prism_fas/detector/c6_bank.py`), the synthesis route definitions
(`src/prism_fas/synthesis/candidate_plan.py`), and the frozen package's
manifest schema. This re-audit found MORE genuine evidence available for two
tests than a prior pass assumed (§5 below), and confirmed two others remain
genuinely blocked for precise, structural reasons (§6).

## 3. Executability classifications

| test | classification | GPU ready |
|---|---|---|
| `benign_jpeg_corruption` | `EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL` | yes |
| `benign_resize_corruption` | `EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL` | yes |
| `benign_color_corruption` | `EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL` | yes |
| `cross_route_synthetic` | `EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL` | yes |
| `recipe_region_shift` | `NEEDS_SCIENTIFIC_DECISION` | no |
| `artifact_map_swap` | `NEEDS_SCIENTIFIC_DECISION` | no |
| `residual_scale_zero` | `STRUCTURALLY_MODEL_BLOCKED` | no |
| `crop_padding_interpolation` | `STRUCTURALLY_DATA_BLOCKED` | no |

## 4. The three benign-corruption tests

Population: `source_dev` LIVE only, both domains, never spoof, never target.
Group-safe 50/50 calibration/evaluation split on `source_record_id`
(reusing `synthetic_real_probe.split_bucket`'s exact hash rule, new
namespace `c9-post-failure-diagnostics-v1`, single seed `20260806` — the
split is a deterministic hash, not a stochastic draw). Evidence
`[global_logit_G, p_global]`, averaged over each arm's 5 checkpoints exactly
as BA_sep's own `average_checkpoint_evidence` does. Metric:
`reliability.score_shift` (the existing, reused, unmodified paired-shift
measurement) on `p_global`, evaluation group only.

No existing declared threshold was found (searched first, per the task's
scientific-decision policy —
`configs/evaluation/m10_target.yaml` declares all three `status: planned`
with no numeric criterion). A calibration procedure was defined instead:
`threshold = calibration_mean_shift + 3 * calibration_std_shift` (population
std, `ddof=0`), computed ONLY from the calibration group. `PASS` iff
`evaluation_group.mean_shift <= threshold` (ties PASS). Fails closed on a
degenerate (empty) calibration or evaluation group.

Exact frozen perturbations:

- **JPEG**: PIL/Pillow, `format="JPEG", quality=50`, RGB, encode-then-decode.
- **Resize**: PIL bilinear downscale by `0.5`, bilinear upscale back to
  `224x224`; aspect ratio unchanged.
- **Color**: fixed per-channel RGB gain `{R: 1.15, G: 1.00, B: 0.90}`,
  clipped to `[0,1]`.

## 5. `cross_route_synthetic`

Routes confirmed from `prism_fas.synthesis.candidate_plan.ROUTES = ("physics", "gpat")`,
present as the `route` field on every C6 matched-bank row. Population: the
real-spoof population BA_sep V2 already resolves
(`synthetic_real_probe.resolve_real_spoof_population`, unchanged), and each
arm's frozen C6 bank partitioned by its own `route` field — never a second
bank resolver, never bank regeneration.

Symmetric design, both directions (`physics_to_gpat`, `gpat_to_physics`).
Per direction: fit the FROZEN BA_sep linear probe
(`synthetic_real_probe.fit_linear_probe`, unchanged hyperparameters — same
z-score/train-only normalization, same LBFGS config) on the TRAIN route's
real-vs-synthetic evidence, score it on the OTHER route's held-out evidence
via the unchanged `compute_ba_sep_for_seed`. Aggregated across the same
three frozen probe seeds (`20260806, 20260807, 20260808`, newly
preregistered for this test) and the two directions.

Threshold: reuses the SAME `0.75` ceiling BA_sep V2 already froze — its
scientific meaning (a linear probe should not separate the classes well)
applies identically here, just to a cross-route rather than same-route
comparison. This is reuse of an existing frozen threshold, not an invented
one, per §5.B of the freeze task.

## 6. The two `NEEDS_SCIENTIFIC_DECISION` tests — closer than previously known

`recipe_region_shift`: the re-audit found `region_embeddings` and the
fusion's own `region_attention` softmax weights ARE populated for primary
Track R (DET/LLM) — gated by `variant.has_region_path` (`region: on`),
independent of the manifold flag. What remains undecided is WHICH existing
signal constitutes the spec's "anomaly heat-map peak" — `region_attention`
and a distance-from-mean over `region_embeddings` are both plausible,
non-equivalent candidates, and this task does not choose one silently.
Whether the frozen M7 recipe bank actually contains recipe pairs differing
only in attacked region is also unverified in the time available.

`artifact_map_swap`: the re-audit found `local_logits` IS populated for
primary Track R (`variant.has_local`, true for `local_branch: convnext`),
`attack_region_mask` already exists per accepted C6 bank row, and
`weighted_local_loss` (`src/prism_fas/detector/losses.py`) already computes
a real, reusable local-supervision quantity from exactly these two — today
as a TRAINING loss, not yet adapted into a read-only diagnostic score with
its own frozen acceptance threshold. Recorded honestly as closer to
executable than a prior audit stated, not as executable today.

## 7. The two blocked tests

`residual_scale_zero` — `STRUCTURALLY_MODEL_BLOCKED`, two independent
reasons: its declared population needs a SYNTHESIS-time GPAT residual
control this task may not exercise (no C6 bank regeneration permitted), and
its declared measurement needs `region_distances`, populated only when
`self.manifold is not None` — true only for the secondary
`TRACK_R_K4_FLAGS` variant, never for Track G or primary Track R, the only
variants with real, trained, P3-ready Version-C checkpoints.

`crop_padding_interpolation` — `STRUCTURALLY_DATA_BLOCKED`, re-confirmed
unchanged: the frozen package stores only the 224×224 crop and its hash, no
bounding box, no path back to the source frame.

## 8. Checkpoint policy

Identical to BA_sep V2: 5 P3-ready Track-G checkpoints per arm (seeds
`20260806`–`20260810`), 15 total, resolved via
`synthetic_real_probe.resolve_checkpoint_set`/`resolve_all_checkpoint_sets`
(reused, never duplicated), all 15 hashes bound before execution, no
best-seed selection, no favorable-seed-only averaging.

## 9. Result namespace

An entirely separate artifact namespace,
`reports/full/c8/reliability/post_failure_source_diagnostics_v1/` — never
`.../synthetic_vs_real_spoof_probe/`, never `DETECTOR_RELIABILITY_LOCK_C.json`.
Seven artifacts across three modes: `DIAGNOSTICS_PROTOCOL_BINDING.json`,
`DIAGNOSTICS_POPULATION_BINDING.json`, `DIAGNOSTICS_CHECKPOINT_BINDING.json`
(`--bind-only`, zero scientific metric), `DIAGNOSTICS_RESULT.json`,
`DIAGNOSTICS_PER_TEST.json`, `DIAGNOSTICS_PROVENANCE.json`,
`DIAGNOSTICS_VERDICT.json` (`--execute`). Every one binds the BA_sep
protocol identity, the BA_sep observed verdict (`FAIL`), the
`DETECTOR_RELIABILITY_LOCK_C` observed overall (`FAILED`), and
`target_access: 0`.

## 10. Implementation

`src/prism_fas/evaluation/post_failure_diagnostics.py` — protocol
loading/identity; deterministic corruption transforms; the group-safe
calibration/evaluation split (reusing `split_bucket`); threshold derivation
and verdict; `source_dev` LIVE resolution; cross-route population
resolution; the real forward-pass orchestration for both test families,
reusing `synthetic_real_probe.construct_row_trainer` and
`M9ValidationDataset`/`collate_items` exactly.

`src/prism_fas/evaluation/post_failure_diagnostics_runner.py` — the CLI,
four modes (`--preflight-only`, `--bind-only`, `--status`, `--execute`),
each mirroring the BA_sep runner's exact fail-closed/no-rerun/partial-block
contract, on its own separate artifact namespace.

## 11. Tests

`tests/pipeline/test_c9_post_failure_source_diagnostics.py` — 60 tests,
fixture/engineering only: protocol identity determinism; `target_access`
everywhere; BA_sep protocol and artifacts provably untouched; `c9_may_close`
hard-coded false on every path; preflight read-only; bind-only zero-metric
and idempotent; complete/partial/clean no-rerun states; fixed checkpoint
policy; group-safe calibration/evaluation split; deterministic corruption
transforms; `score_shift` regression; every blocked test's reason present
and precise; `crop_padding_interpolation` never renamed to resize; no
forbidden evidence field read; no target path reachable; corrupted/mismatched
bindings fail closed.

60/60 pass. Full C9-scoped suite (five files): 277/277 pass. Broad
regression (`tests/c7 tests/pipeline`): 33 failed / 1790 passed / 22 skipped
— the exact, unchanged pre-existing baseline.

## 12. What this freeze does NOT do

Does not compute a real diagnostic value. `--preflight-only`, `--bind-only`,
`--status` and `--execute` were each run once against this real repo on this
laptop and each correctly reported an unresolved precondition, exit 2,
writing nothing (no CUDA, no M3B package). Does not touch the BA_sep
protocol, its artifacts, or `DETECTOR_RELIABILITY_LOCK_C.json` — verified by
regression and by direct byte-comparison tests. Does not access target data.
Does not implement the exploratory target protocol (a separate, future
task, only after these diagnostics are observed and frozen).
