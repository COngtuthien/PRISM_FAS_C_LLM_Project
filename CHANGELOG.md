# Changelog

## M10 COMPLETED - blind SiW-Mv2 target evaluation - 2026-08-09

`target_labels_revealed: false -> true`. The transition is one-way, happened exactly once, and is
recorded in `reports/m10/TARGET_LABEL_REVEAL.json` with the lockset identity, the pre-reveal audit
identity and the sealed artifact's own SHA-256. It happened only after all 37 prediction-eligible
rows were frozen.

### The order the milestone was executed in, and what gated each step

- `SOURCE_MATRIX_LOCK` (`c06944344eab...`) validated a **third** time, this time REMOTELY inside a
  Modal container against the bytes on the runs volume: for all 37 eligible rows the checkpoint
  exists with the locked SHA, the calibration exists with the locked SHA and hash, the run is
  COMPLETED and `target_test_opened` is false (`reports/m10/SOURCE_MATRIX_REVERIFICATION.json`).
- The label firewall re-verified in-container: 7 shards / 6776 rows / 1700 videos, every shard and
  manifest hash matching, `feature_label_leakage: 0`, **`label_artifact_resolvable: false`** across
  all three mounted volumes.
- **Scientific G7 over 37 rows**, each 6776 frames across 1700 videos (1676 videos x 4 frames,
  24 x 3). A scientific row now reads `SOURCE_MATRIX_LOCK` INSIDE the container and refuses unless
  the opened checkpoint SHA, the calibration SHA and the calibration hash are the ones the lock
  froze, and unless the prediction covers the whole package. `p_prompt` is `not_applicable` on
  every target row, as the contract predicted.
- `TARGET_PREDICTION_LOCKSET.json`: 37 entries, 250712 frame rows, built twice with an identical
  identity, binding the source lock, the matrix, the target FEATURE identity, every row status, the
  4 BLOCKED rows and the 5 rows that legitimately produce no prediction.
- Pre-reveal audit **15/15**, then the reveal, then an isolated G8 over all 37 rows.

### Measured results - two of five hypotheses supported

1700 videos (785 live / 915 spoof), 6776 frames. Each row decided at its OWN frozen source-dev
threshold.

```
B01  ACER 0.21798            AUC 0.87436      (1 seed, diagnostic)
B06  ACER 0.31190 +/- 0.01040 AUC 0.78822     (3 seeds)
B07  ACER 0.35962 +/- 0.02598 AUC 0.72050     (3 seeds)
B08  ACER 0.36088 +/- 0.04730 AUC 0.69327     (3 seeds, the full method)
B00  ACER 0.42307 +/- 0.10879 AUC 0.78588     (3 seeds)
A01  ACER 0.51560 +/- 0.00409 AUC 0.32330     (3 seeds, naive_concat)
```

| H | comparison | dACER | 95% CI | p (Holm) | outcome |
|---|---|---|---|---|---|
| H1 | B08 vs A01 naive_concat | -0.15593 | [-0.17932, -0.13239] | 0.0000 | **supported** |
| H2 | B07 vs B06 | +0.05214 | [+0.03967, +0.06531] | 0.0000 | **not_supported** |
| H3 | B08 vs A07 image_level | +0.01293 | [+0.00752, +0.01889] | 0.0000 | **not_supported** |
| H4 | B08 vs A02 random_operators | +0.00974 | [+0.00393, +0.01590] | 0.0004 | **not_supported** |
| H5 | B08 vs A04 hard_gate_only | -0.01540 | [-0.02535, -0.00550] | 0.0018 | **supported** |

All five cleared Holm-Bonferroni, but **three cleared it in the direction opposite to the
prediction**. Those are negative results and are reported as such, in full, rather than dropped,
re-tested under another metric or re-run with another seed.

### Two real defects, both found and fixed BEFORE any label was opened

- **G7 decided on the wrong quantity.** It compared the fused `s_final` against "the frozen
  source-dev threshold", but `run_g6` fits both the temperature and that threshold on
  `output.global_logit` alone, and the checkpoint is selected on `sigmoid(global_logit)`. Measured
  on `source_dev`: calibrated `p_global` at the frozen threshold gives ACER 0.1382 (reproducing the
  frozen G6 record's 0.13695), while `s_final` at the same threshold gives APCER 0.0 / BPCER 1.0 /
  **ACER 0.5** - a degenerate all-spoof classifier, because mean `s_region` on bona-fide samples is
  **0.9644**. The prediction schema was versioned **v1 -> v2** with a `decision_score` column, and
  all 37 predictions were regenerated from the SAME frozen checkpoints and calibrations. Nothing
  was retrained, refitted or reselected; the superseded v1 predictions are retained under
  `reports/m10/g7_v1_superseded/`. `docs/M10_TARGET_EVALUATION_CONTRACT.md` section 2b.
- **G8 was not actually torch-free.** `isolation_report()` reported `g8_imports_torch: false`, and
  that was **false**: `import_closure_audit` resolved a relative import (`from .metrics import ...`)
  to the bare name `metrics`, silently dropping whole subtrees, and `evaluation.metrics` imports
  `prism_fas.train.metrics`, which resolves the `prism_fas.train` package first, whose `__init__`
  imported `.models` and with it torch. The audit now resolves relative imports and walks the
  transitive first-party graph (16 modules), and `prism_fas/train/__init__.py` re-exports lazily.
  A test asserts it in a FRESH interpreter, because asserting it inside the test session would
  prove nothing.

### Also corrected

- The paired bootstrap scored both sides at ONE threshold. Two experiments never share a
  `source_dev` threshold, so `paired_bootstrap` now takes `threshold_a`/`threshold_b` and each side
  is scored at its own frozen operating point.
- `m10 reveal --dry-run` opened the sealed artifact to report its counts. It now returns before the
  read; the one occurrence is disclosed inside `TARGET_LABEL_REVEAL.json` itself, with why it
  changed nothing (the audit had passed, every prediction was already immutable, and the counts it
  printed were already public in the frozen package inventory).

### Reliability - 6 PASSED, 2 FAILED, 2 BLOCKED

Every acceptance rule was given its number in `modal_m10_reliability.py` BEFORE the tests ran.

- FAILED `synthetic_vs_real_spoof_probe`: a linear probe on the detector's own evidence separates
  synthetic from real spoof at **0.9375** balanced accuracy against a declared ceiling of 0.75.
- FAILED `residual_scale_zero`: scaling the residual to zero moves the decision score by only
  **0.022** against a declared minimum of 0.10, although the mean regional distance does fall
  (0.772 -> 0.622). The regional evidence responds to the artefact; the decision largely does not.
- PASSED: the three benign corruptions, `recipe_region_shift`, `artifact_map_swap`,
  `cross_route_synthetic`.
- BLOCKED: `benign_glasses_makeup_lowlight` (unchanged) and `crop_padding_interpolation`, because
  the frozen package stores only the 224x224 crop - no bbox, no path back to a source frame.

### Compute, measured

**12.56 GPU-hours** of training across the 37 rows, read from each run's own
`stages/<G>/output_hashes.json` rather than estimated. Offline synthetic/preprocessing cost and
online G7 inference cost are reported separately.

### Artifacts and tests

`report.html`, `summary.json`, `M10_ACCEPTANCE.json` (**32/32**), `TARGET_PREDICTION_LOCKSET.json`,
`TARGET_LABEL_REVEAL.json`, `PRE_REVEAL_AUDIT.json`, `G7_PREDICTION_VALIDATION.json`,
`SOURCE_MATRIX_REVERIFICATION.json`, `DECISION_SCORE_DIAGNOSTIC.json`, `RELIABILITY.json`,
`M10_COMPUTE_RAW.json`, `TEST_SUITE.json`.

Full suite: **1081 passed, 0 failed, 0 skipped** (1057 before this closure; 24 added).

**No state-of-the-art claim, no first-method claim, and no comparison outside the five predeclared
hypotheses.**

## M10 Additive SiW-Mv2 Target Evaluation Package - 2026-08-08

`target_labels_revealed: false`, `target_label_artifact_built: true`. The sealed
evaluation label artifact exists and **has never been opened**; no target metric has
been computed, no matrix row has been launched and no M10 Modal app has ever been
created.

- Built the additive, immutable **`prism_target_eval_v2`** target evaluation package:
  **1700 videos = 785 live + 915 spoof across all 14 attack families**, at the frozen
  4 frames/video plan. `prism_data_v1_m3b`, `configs/data/siw_mv2.yaml`, the M8 bank
  and every M9 artifact are untouched, so every identity they bind still holds.
- **Frame accounting reconciles exactly: planned 6800 = 6776 successful + 24 recorded
  failures.** The 24 are genuine `no_face` detector outcomes on 24 distinct videos,
  each recorded with its opaque record id. All 1700 videos survive - 1676 with four
  frames, 24 with three - and none was dropped.
- **Fixed a real defect this build would otherwise have hidden.** Every failure
  handler in `run_preprocessing` routed for `source` and re-raised for `target`, and
  the target `no_face` branch incremented an in-memory counter without writing a
  manifest row at all. One undecodable spoof video would have aborted the whole
  1700-video build, and a no-face frame would have vanished from the accounting.
  Target failures are now routed; source behaviour is unchanged.
- **The 785-live byte-for-byte reproduction gate PASSED on the full population:**
  3140/3140 comparable frozen live frames reproduce their `sample_id` AND their
  `crop_sha256` exactly, with 0 sample-id mismatches, 0 crop-hash mismatches and 0
  field mismatches. `adapter_version` stays `"1.0"` precisely so this check remains
  possible; the layout revision is carried by `layout_rules_version:
  siw-mv2-layout-v2`, the layout content identity and a new package id.
- Added an **isolated `target_eval_v2` M2 run profile** whose run context refuses to
  construct itself if pointed at a frozen namespace, at a dataset other than
  `siw_mv2`, or at a role other than `target`. The package is additive by
  construction, not by convention.
- Added `src/prism_fas/data/target_eval.py` and `prism m10 target-package`
  (`inventory`, `plan`, `extract`, `package`, `labels`, `reproduce`, `acceptance`),
  plus `configs/data/package_m10_target.yaml` and `configs/data/loader_m10_target.yaml`.
  The inventory audit compares the declared globs against **every video on disk** and
  hard-fails on an undeclared family, an unexpected filename stem, an unmatched video,
  a duplicate opaque id or any count mismatch - the check the v1 layout never had, and
  the reason 596 of 915 spoof videos were once dropped in silence.
- **Sealed the evaluator-only label artifact** (1700 rows, 785 live / 915 spoof, 14
  families, identity `863b80f0...`) into `data/evaluation_only/`, which is now
  git-ignored. Building and sealing is **not** revealing: the builder canonicalizes,
  hashes and seals, and trains nothing, tunes nothing, selects nothing and computes no
  target metric. The two states are recorded separately.
- **0 target identity embeddings**, asserted structurally rather than assumed:
  identity is applicable only to `source_train` live rows, so a target row can never
  qualify. `identity_not_applicable: 6776`.
- **0 label leakage on the feature side**: the target feature manifest carries no
  label, attack family, taxonomy, subject, session or official-split column.
- Idempotency verified: a completed rerun rebuilt nothing and reproduced the feature
  content identity `c3a29e69...`, the manifest SHAs and the shard set exactly.
- Ran a bounded **label-free G7 smoke against the new package** (240 frames / 225
  videos): strict identity pin, feature loading, schema, finite outputs, video
  grouping, prediction lock and firewall all verified. No target metric computed.
- `M3B_PACKAGE_ID` became an input rather than a module constant, so an additive
  package can no longer silently claim the frozen `prism_data_v1_m3b` identity.
- Tests: **894 passed, 0 failed, 0 skipped** (863 at the framework checkpoint;
  31 focused target-package tests added).

## M10 Experiment Matrix and Blind Target Evaluation Framework - 2026-08-08

`target_labels_revealed: false`. No SiW-Mv2 label has ever been opened, no target metric exists, no
matrix row has been launched and no M10 Modal app has ever been created.

- **Froze three contracts before running anything**: `docs/M10_EXPERIMENT_CONTRACT.md` (Table 59
  B00-B08 as configuration switches over one shared implementation, the Table 60 ablation deltas and
  the replication policy), `docs/M10_TARGET_EVALUATION_CONTRACT.md` (Table 57 prediction schema,
  video aggregation, PREDICTION_LOCK, the G7/G8 permission split, metric definitions) and
  `docs/M10_STATISTICS_CONTRACT.md` (bootstrap unit/seed/resamples/confidence/statistic and the
  multiple-comparison policy).
- **Froze the replication policy by declared scientific role** rather than assuming the spec's
  3 seeds applies to every row. §20.1 names only "main baseline and full method"; every auxiliary
  count is genuinely `SPEC_UNDERSPECIFIED`. B00 and B08 carry 3 seeds because the spec says so;
  B06, B07 and the four ablations that test H1/H3/H4/H5 carry 3 seeds because a declared hypothesis
  rides on them; expensive diagnostic rows carry 1 seed and are reported `single_seed_descriptive`
  with no interval and no superiority claim. A comparison whose either side is a single-seed row is
  **refused** by the statistics module, never silently downgraded.
- Added `src/prism_fas/evaluation/`: the deterministic matrix planner, the experiment registry, the
  structural target-label firewall, G7 label-free prediction with the versioned nullable Table 57
  schema, video aggregation, metrics, the deterministic paired video bootstrap, the isolated G8
  scorer, the reliability framework and the report assembler. **No module under
  `src/prism_fas/evaluation/` imports modal**, and G8 imports no torch and no trainer.
- Added `configs/experiments/m10_matrix.yaml`, `configs/evaluation/m10_target.yaml` and
  `configs/cloud/modal_m10.yaml` (declaration only - the app has never been created), plus the
  `prism m10` CLI with `--dry-run` on every writing command.
- **Materialized the matrix twice with an identical identity**
  `a4972b0dc23946c4ad169f2c856fc9b5e0387baca45b2c9a4895f8180d9c2dd5`: 42 logical rows, 38 executable
  (37 training + 1 bounded parity), 36 needing new GPU work because B08 seed 20260806 binds the
  existing M9 reference run under an identity check, and 4 blocked rows that stay in the matrix
  carrying their reasons.
- **Kept the blocked rows visible instead of deleting them.** The Table 60 frame-count ablation
  (16/32/48-64) is BLOCKED because the frozen plan is 4 frames/video and re-extracting SOURCE would
  break every M8/M9 identity, while uniform-4 is not a subset of uniform-16 so a denser TARGET
  package would forfeit the byte-for-byte live reproduction check. The full-length PC-vs-Modal
  training comparison is BLOCKED for want of a local CUDA device; the bounded step-parity protocol
  runs instead and is reported as a parity result, never as a second training result.
- **Implemented the firewall structurally.** Permissions are checked against resolved paths, so
  `..`, a symlink or a relative spelling cannot slip past; isolation declarations are skipped by
  the config scan so the proof of isolation cannot read as a leak (the M8/M9 lesson); and G8's
  absence of a training capability is proved by an AST audit of its imports and its import closure,
  not by a comment. G8's report writer refuses `.pt`, `.pth`, `.ckpt`, `.safetensors`, `optimizer`,
  `calibration/` and `checkpoints/` before it looks at the destination root.
- **Ran a label-free G7 engineering smoke on real target features**: 256 frames / 228 videos of the
  frozen `target_test` FEATURE package through the full M9 detector on CPU, verifying feature
  loading, region priors, the checkpoint identity guard, the forward pass, the Table 57 schema,
  finite outputs, video aggregation, the prediction lock and the firewall. No target metric was
  computed and none may be quoted from it.
- **The smoke found a real defect, and it was fixed rather than reported.** `p_prompt` is a
  structural zero on target data for B08 as well, because `PromptHead.applicability` is
  `is_synthetic AND attacked-region AND visible` and a target sample is neither synthetic nor
  masked. The first run wrote that `0.0` as `computed` - a constant presented as a measurement. G7
  now reads the model's own `prompt_applicable` mask and writes `null`/`not_applicable`. The fused
  score is unchanged, so the prediction logical identity is identical before and after, which is
  itself the proof that only the honesty of the column changed. Two consequences are recorded in the
  contract: on target data B08's fusion reduces to `1 - (1 - p_global)(1 - s_region)`, and the A08
  prompt ablation can only differ on target through training, never through inference-time fusion.
- Tests: **863 passed, 0 failed, 0 skipped** (745 at the M9 checkpoint; 118 focused M10 tests).

## M9 Regional Detector, PromptHead, Manifolds and the Reference Training Run - 2026-08-07

- Added `src/prism_fas/detector/`: the pinned pretrained registry, `GlobalHead`/`PromptHead` with the
  cached frozen recipe text embeddings, the Table 32 detector with the Table 34 fusion and a typed
  `ModelOutput`, the fail-closed M8 v3 bank reader, deterministic region-prior and attack-mask
  caches, the real+synthetic dataset, the exact 12/12/8 sampler, strict identity-guarded checkpoints
  and the G1/G2/G5/G6 stage trainer. No module under `src/prism_fas/detector/` imports modal.
- Added `modal_m9.py` (app `prism-fas-b-m9`) with `m9_environment_probe`, `m9_detector_smoke`,
  `m9_initialize_prototypes`, `m9_train_reference` and `m9_validate_checkpoint`, plus
  `scripts/m9_local_smoke.py` and `scripts/m9_acceptance.py`. The existing `prism-fas-b-data`,
  `prism-fas-b-models` and `prism-fas-b-runs` volumes are reused; the package and bank were not
  re-uploaded.
- **Pinned SigLIP2** to `google/siglip2-base-patch16-224` @ `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`
  - a resolved commit, never `main` - recording all seven file SHA-256 values, uploading them to
  `prism-fas-b-models:/pretrained/m9/` and re-hashing every one inside the L4 container.
- **Froze the recipe text cache as an uploaded artifact** (128 x 768, identity `10f4ec35...c141`,
  file SHA `bb7d3fb4...83aa`). The remote smoke showed encoding is deterministic within an
  environment but not bit-identical across transformers 4.49/torch 2.5.1 CUDA and transformers
  5.14/torch 2.13 CPU, so a silent rebuild would hand a run a different content identity for
  identical science. It is now minted once and only ever verified.
- Verified on real data before training: a local CPU smoke over real `source_train` and real accepted
  M8 samples; an L4 smoke of 5 steps, checkpoint and strict resume to step 6 without restarting at
  zero; and prototype initialization run twice independently with an **identical** prototype identity
  and identical centre/variance digests over the 280-sample live population (CASIA 160 / MSU 120).
- Ran **one** reference training run `m9_reference_seed20260806` (seed 20260806, EMA disabled) on
  NVIDIA L4: G1 3 epochs / 135 steps, G2 K-means initialization plus 2 manifold warm-up epochs to
  225 steps, G5 30 mixed epochs to 1575 steps (2247.94 s), G6 source calibration on 2079 `source_dev`
  rows. **0 non-finite** `L_total` and 0 non-finite loss terms across all 1350 G5 steps, and every
  one of those batches carried the exact declared 12/12/8 composition.
- Measured source-side results, selected by the frozen ACER -> BPCER -> NLL criterion at epoch 35:
  `source_dev` ACER **0.136953**, APCER 0.161406, BPCER 0.112500, ROC-AUC 0.917853 on all 2079 rows.
  Across the 30 validations NLL fell monotonically 0.521757 -> 0.484347 and ROC-AUC rose
  monotonically 0.906608 -> 0.917853. G6 temperature scaling on `source_dev` only: T 0.348756,
  threshold 0.837451, NLL -> 0.371227, ECE 0.219956 -> 0.128980, Brier 0.151305 -> 0.122078.
  Best checkpoint `06ead67e...f3f`, last `3f2be3c4...640`; the best checkpoint was re-opened under
  strict identity in a separate invocation and reproduced its selection metrics exactly.
- **SPEC_UNDERSPECIFIED, declared not assumed:** the diagonal Mahalanobis distance uses the
  per-dimension mean rather than the raw sum over `D`. Observed on the first real batch - a raw
  256-dim squared Mahalanobis measures ~250 in-distribution, which makes the spec's own
  `margin_out = clean_cap = 3.0` inoperative (`L_out` identically 0, `L_clean` identically 3).
  Dividing by `D` gives `E[d] = 1`. No formula and no declared weight changed; the convention lives
  in the contract, the config, the architecture identity and DECISIONS.md.
- Six defects were found by integration and fixed rather than worked around: `L_prompt` dotted the
  unprojected `z_r` against the text matrix; the variance floor rejected its own float32 round trip;
  the trainer never seeded, so two runs of one config built different weights; `run_id` polluted the
  config hash and therefore the prototype identity; `AutoModel.from_pretrained(dtype=...)` is
  transformers 5.x only; and `run_g6` called `calibrate_source_dev` with a keyword it does not take -
  caught mid-run, fixed, and recovered by resuming the SAME run id so G1/G2/G5 were reused and only
  G6 re-executed.
- Tests: **745 passed, 0 failed, 0 skipped** (662 before M9, 83 new in
  `tests/test_m9_regional_detector.py`).
- **M9 validates the reference detector and its training pipeline. It does not establish SiW-Mv2
  performance, target APCER/BPCER/ACER, cross-domain superiority or any final research claim.**

## M8 GPAT, Quality Gate and Versioned Synthetic Bank — 2026-08-07

- Added `src/prism_fas/synthesis/` M8 modules: differentiable Haar DWT/IDWT, the GPAT residual model
  (910,538 parameters, ΔLL structurally absent), the declared loss set, the trainer with strict
  checkpoint/resume, the pinned quality-model registry, the source-only pair plan, quality
  calibration, the quality gate, the 24-d high-frequency fingerprint, the candidate plan, discrete
  uint8 finalization, resume-safe generation, deterministic shards, bank validation and export.
- Added `modal_m8.py` with GPAT training and every calibration, pilot, generation, re-assembly and
  export stage, plus `scripts/m8_validate_downloaded_bank.py` and the `prism synthesis` commands. No
  module under `src/prism_fas/synthesis/` imports modal.
- Trained GPAT on L4 for 15 epochs; best epoch 11, `validation_total_loss` 0.048736,
  `validation_identity_cosine` 0.999893, checkpoint SHA `2047cdb513767010…`.
- Generated the frozen 1120-candidate plan (560 physics + 560 GPAT over all 280 `source_train` live
  samples, 128 distinct recipes), identity `b167c169dcb92426…`.
- **Calibrated the quality gate three times, and retained every superseded run.** Each revision
  changed a calibration *population*, never a threshold, and each rule was declared before the
  candidates it judges were re-evaluated:
  - **v1** — same image under ±2 % brightness/contrast and noise 0.002. 391 accepted; missed
    `accepted_total` (391 < 400) and `accepted_physics` (71 < 200). Retained as
    `prism_synthetic_bank_m8_v1_ef6a76ed46f0`, status `operational_minimums_failed`.
  - **v2** — real same-identity cross-record pairs (560 genuine / 13440 impostor, no distribution
    overlap). `tau_id` 0.9995203357934952 → **0.547440037939055**; identity rejections fell to 0 and
    475 were accepted, but `accepted_physics` was still 151 < 200. Retained as
    `prism_synthetic_bank_m8_v2_1abc9e83c2a5`, status `operational_minimums_failed`.
  - **v3** — same image under a **localized benign appearance edit**: 280 live × 8 frozen transforms
    (brightness 0.90/1.10, contrast 0.90/1.10, gamma 0.90/1.10, noise std 0.005, blur sigma 0.75)
    applied inside one deterministic semantic region. `tau_lm` → **0.00836817528937794** (p99),
    `tau_parse` → **0.7094826178704915** (p01).
- v3 versions only `tau_lm` and `tau_parse`; `tau_fd`, `tau_id`, `tau_out`, `tau_fp`, the fingerprint
  reference, the artifact-strength and support-overlap rules, the `q` formula and every operational
  minimum are unchanged, and the v1/v2 configs and artifacts were never modified.
- v3 calibration measured 2240/2240 valid landmark comparisons, 0 face-detection failures, 0
  region-mask failures and an outside-support uint8 error of exactly 0 on every observation, and
  reproduced with **0 mismatches** across two runs. Its logical row digest was re-verified locally
  under pyarrow 25.0.0 against the remote 18.1.0 write.
- Computed the 560 cross-record genuine pairs as a **diagnostic only**, and refused to derive a
  structural threshold from them: their landmark NME median is 0.0828 against 0.0013 for the
  same-image population, so they measure real pose/expression/crop variation rather than detector
  jitter under an appearance-only edit. Their 1st percentile alone is 3.3× larger than `tau_lm_v3`.
- Froze `prism_synthetic_bank_m8_v3_e84c78cd2a9b`
  (`e84c78cd2a9b548244e243de0380998d04bc6770b91caf32ac7be96f489bb542`, status **`validated`**):
  1120 candidates = 871 accepted + 249 rejected + 0 failed; physics 419, gpat 452; CASIA 491,
  MSU 380; 8/8 artifact types, 9/9 regions; same- and cross-domain gpat 226 each; 2 shards.
- Reused **475 payloads byte-identically** from the retained v2 run and regenerated 645 under their
  original ids, inputs and seeds. Re-calibration changes a decision, never a payload byte, and the
  measured comparison shows **0 accepted → rejected** transitions in either route.
- Verified: bank validation 39/39 checks with 0 errors; resume audit with a real interruption, a
  hash-detected truncation rebuilt byte-identically and a completed rerun of 1120 examined / 1120
  reused / **0 rebuilt**; a 32-candidate determinism audit with **0 mismatches**; and a Windows round
  trip of the 126,689,280-byte archive (SHA `38a92fda0c7ae004…`) whose extracted local bank identity
  **equals** the remote identity.
- Fixed real defects found during M8, each documented in `DECISIONS.md`: the physics artifact-map
  units error (a recipe requesting 0.0915 measured 0.780, rejecting every physics candidate); a
  non-portable pair-plan identity that folded in parquet bytes; a `PYTHONHASHSEED`-dependent batch
  order that made training irreproducible across processes; duplicate candidate ids from drawing
  recipes with replacement; an unconditional `BANK_LOCK.status = "validated"` that made both the
  field and the validator's check vacuous; config loaders and a leak scanner that rejected artifacts
  for *honestly declaring* the splits they refuse to open; and a v2 bank named with a v1 prefix.
- Source isolation on every M8 stage: only `manifests/source_train.parquet`, `images/` and `priors/`
  were opened; `source_dev_opened false`, `target_test_opened false`, `raw_dataset_path_opened false`.
- **No FAS detector quality or target-test performance is claimed.** `q` is an M9 sample weight, not
  a live/spoof label. M9 is NOT STARTED.
- Tests: 662 passed, 0 failed, 0 skipped (was 465 after M7).

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
