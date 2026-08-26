# POST_FAILURE_EXPLORATORY_TARGET_V2 — pre-target scientific + execution correction

**PRE-TARGET CORRECTION, NOT A REVISION OF AN OBSERVED TARGET RESULT.**
`POST_FAILURE_EXPLORATORY_TARGET_V1`
(`configs/evaluation/post_failure_exploratory_target_v1.yaml`, identity
`8fb806d25a80ecd3c7d44cfeba8c893a5f115b8b51797220a51132ba16708b51`) was
frozen but never scientifically executed: no SiW-Mv2 prediction has run, no
target label has been opened, no target metric has been observed. A
pre-target audit found nine defects in V1's execution/binding/lockset/
statistics design, corrected here in a separately identified protocol —
`POST_FAILURE_EXPLORATORY_TARGET_V2`
(`configs/evaluation/post_failure_exploratory_target_v2.yaml`, identity
`2f1beb0b95f01051e06c0ef8a82d06a759d0fe8f81f693c5d3a4d777845196a9`). **V1 is
preserved byte-for-byte, unchanged; V2 never writes into its namespace.**

BA_sep remains permanently FAILED; `DETECTOR_RELIABILITY_LOCK_C` remains
`overall=FAILED`; `POST_FAILURE_SOURCE_DIAGNOSTICS_V2` remains
`overall=FAIL`; the original C9 confirmatory path remains BLOCKED. `target_access=0`
throughout this task.

---

## Defects corrected

**A — execution not authoritatively bound to the frozen plan.** V1's
`--predict` re-resolved `source_matrix`/run manifests/checkpoint+calibration
live at execution time, letting scientific inputs drift silently after
binding. **Corrected:** `PREDICTION_PLAN_BINDING.json` is now the sole
authoritative execution source. `verify_binding_unchanged` recomputes the
candidate binding READ-ONLY and requires EXACT equality with the frozen
document; any difference sets `error: PREDICTION_PLAN_BINDING_DRIFTED` and
BLOCKS. `_predict` then drives inference from `frozen_binding["rows"]`
alone — proven by test that its source never calls `resolve_all_row_bindings`
or `resolve_target_matrix`.

**B — target feature package unverified at bind.** V1's `--bind-prediction-plan`
could succeed with `target_feature_package.verified=false`. **Corrected:**
`verify_target_feature_package_required` raises unless
`present_on_this_host AND verified AND computed_identity==expected`;
`target_feature_package_identity_verified` and `target_label_access=0` are
recorded as separate fields, never blurred with label access. `--preflight-only`
still MAY report the package absent/unverified without raising (proven by
test).

**C — the legacy lockset cannot represent 24 seeded rows.** Audited
`target_prediction.build_lockset`: it rejects duplicate `experiment_id`
values, but every Track-G/Track-R arm in the frozen matrix has 3-5 rows
sharing one `experiment_id` by design. The legacy function is left
**completely unchanged** — a regression test proves it still rejects
duplicates, exactly as designed for its own non-seed-replicated use.
**Corrected:** `build_v2_prediction_lockset`, a new, `row_id`-keyed
lockset, is used instead — 24 unique entries, `entry_count==24` enforced,
repeated `experiment_id` values fully valid.

**D — NOPROMPT identity collapse.** `variant = row.arm` made `C-R-LLM` and
`C-R-NOPROMPT` (both `arm=LLM`) indistinguishable in every prediction row,
`inference_config_hash`, and lock. **Corrected:**
`prediction_variant_id = row.experiment_id` (`C-G-RND`, `C-G-DET`,
`C-G-LLM`, `C-R-DET`, `C-R-LLM`, `C-R-NOPROMPT`) is used consistently for
the prediction row's `variant`, the inference-config-hash variant
component, and the per-row lock's `variant`; `arm` is retained as a
separate field.

**E — an existing lock was re-reported without validation.**
**Corrected:** `validate_existing_exploratory_prediction_result` checks the
active protocol identity, the binding identity, the target-matrix identity,
the C8 matrix identity, every row's identity fields/checkpoint/calibration
hash against the frozen binding, every prediction file's real SHA-256 and
schema (reusing `target_prediction.validate_predictions`/`read_predictions`
verbatim — so a forbidden label column would be caught structurally too),
every per-row lock's identity, and the lockset's own `lockset_identity`.
A complete-but-invalid lock BLOCKS (`EXISTING_RESULT_FAILED_VALIDATION`) —
never recomputed, never overwritten.

**F — weak partial-result detection.** V1 only counted
`target_predictions.parquet` files. **Corrected:** `_detect_partial_state`
checks BOTH prediction files and per-row `PREDICTION_LOCK.json` files for
all rows, and separately detects "all row artifacts present but the
overall lock missing" and "overall lock present but rows missing" — every
case BLOCKS.

**G — hard-coded target feature root.** `predict_one_row` no longer
hard-codes `data/processed/prism_target_eval_v2`; the root comes from the
caller's already-verified protocol/binding path.

**H — "paired ACER" was a raw class-blind error rate.** V1's
`compute_exploratory_comparisons` reduced every video to a binary error and
took `mean(error_A) - mean(error_B)` — equal to ACER only when live/spoof
counts are equal, which they are not (785 live vs 915 spoof on the frozen
target). **Corrected:** `apcer_bpcer_acer` computes
`APCER = count(spoof predicted NOT spoof)/count(spoof)`,
`BPCER = count(live predicted spoof)/count(live)`,
`ACER = 0.5*(APCER+BPCER)` from the correct class populations. Verified by
test with unequal class counts that the two definitions diverge.

**I — multiple seeds silently overwritten.** V1 built `video_id -> error`
via repeated `.update()` across every seed sharing an arm; later seeds
overwrote earlier ones, discarding most replication evidence.
**Corrected:** `class_stratified_paired_bootstrap` keeps each matched
seed's per-video decisions in its own mapping and only averages at the
final `delta_seed` step — verified by test that five seeds with distinct,
deliberately different APCER values all survive into
`per_seed_observed`, none overwritten.

**J — Holm family mis-specified.** V1 declared `family_size: 4`, one
p-value per E-H, undercounting E-H1's three pairwise arms (RND-DET,
RND-LLM, DET-LLM) and E-H4's two matched-arm pairs (Track-G vs Track-R for
DET and for LLM). **Corrected:** all seven atomic comparisons
(`E-H1_RND_vs_DET`, `E-H1_RND_vs_LLM`, `E-H1_DET_vs_LLM`, `E-H2`, `E-H3`,
`E-H4_DET`, `E-H4_LLM`) enter one Holm-Bonferroni family,
`family_size: 7`.

## The class-stratified paired video bootstrap (frozen before target access)

```
match compared rows by exact seed
partition target video IDs by true class (LIVE, SPOOF) after label unlock
for each of 10000 replicates:
    resample LIVE video IDs with replacement from LIVE only
    resample SPOOF video IDs with replacement from SPOOF only
    use the SAME resampled IDs for BOTH compared methods, every matched seed
    for each matched seed: compute APCER/BPCER/ACER for both methods; delta_seed = ACER_A - ACER_B
    replicate_statistic = mean(delta_seed across ALL matched seeds)
observed_statistic = mean over matched seeds of (ACER_A(seed) - ACER_B(seed)) on the real video set
CI = 2.5th/97.5th percentile of the 10000 replicate statistics
p = min(1, 2*min(P(replicate<=0), P(replicate>=0))) — a replicate exactly zero counts
    toward BOTH tails (inclusive <=/>=), one single frozen implementation
```

Seeds are fixed replications and are never themselves resampled. Matched
seeds per comparison follow automatically from set intersection of the two
arms' available seeds: E-H1 uses all 5 Track-G seeds (verified by test);
E-H2/E-H3/E-H4 use the 3 seeds common to both sides (Track-R's 3, or the
Track-G/Track-R intersection for E-H4) — verified by test.
`bootstrap_seed=20260810`, `resamples=10000`, `confidence_level=0.95`,
determinism verified by test (two independent calls with the same seed
produce byte-identical CI/p-value).

## Namespaces, reuse, and what stays untouched

- `configs/evaluation/post_failure_exploratory_target_v2.yaml`,
  `reports/full/exploratory_target_v2/`, `runs/exploratory_target_v2/`,
  `state/exploratory_target_v2/` — disjoint from V1's namespace (proven by
  test) and from C9-C13.
- V1's own modules
  (`post_failure_exploratory_target.py`/`post_failure_exploratory_target_scorer.py`)
  are **completely unmodified** (`git diff --stat` empty, proven by test);
  V2 imports their row/matrix/package/firewall resolution functions
  verbatim rather than duplicating them.
- The legacy M10 library (`target_prediction.py`, `firewall.py`,
  `video_aggregation.py`, `scoring.py`) remains reused verbatim except for
  `build_lockset`, which V2 never calls and which is proven unmodified.
- `synthetic_real_probe.construct_row_trainer` remains track-agnostic and
  unchanged, used for every one of the 24 rows regardless of Track G/R.

## What this correction does NOT do

Does not open the target feature package or label artifact. Does not run
SiW-Mv2 inference. Does not compute a target metric. Every CLI mode
(`--preflight-only`, `--bind-prediction-plan`, `--status`, `--predict` on
the predictor; `--preflight-score`, `--score` on the scorer) was run once
against this real repo; the real 24-row matrix resolution and its 15
Track-G/9 Track-R breakdown succeeded (proof the matrix audit remains
correct); every step needing GPU checkpoints, the target package, or
target labels correctly reported an unresolved precondition, exit 2,
writing nothing. `target_access=0` throughout.

## Tests

`tests/pipeline/test_post_failure_exploratory_target_v2.py` — 52 tests
covering all thirty-two required items (A-AF): V1 byte-identical/unchanged;
the legacy `build_lockset` source untouched and still correctly rejecting
duplicates; V2's distinct protocol identity and disjoint namespace; the
real 24-row/15-9 matrix; the frozen binding's completeness (including
`flags`); binding-drift detection for a changed checkpoint, a changed
calibration, and a changed source matrix, each blocking `--predict`;
package verification required before bind (not merely at preflight); the
predictor's import statements never mention the label reader;
`C-R-NOPROMPT` vs `C-R-LLM` distinct `prediction_variant_id`s; the V2
lockset's 24 unique row_ids with valid repeated `experiment_id`s; complete/
partial/missing-lock/missing-rows prediction states; the corrected ACER
formula and its divergence from a class-blind error rate under unequal
counts; every seed surviving the paired comparison; the bootstrap's
determinism under the frozen seed; all seven comparisons entering one Holm
family; the scorer's no-checkpoint-loading audit and its requirement of a
valid frozen lock; existing-score-result validation failing closed on
tampering; and `target_access=0`/labels-closed in every laptop-run mode.

Combined C9-scoped suite (ten files): 480/480 pass.
