# POST_FAILURE_EXPLORATORY_TARGET_V1 — protocol freeze

**NOT the original C9-C13 confirmatory path. NOT a C9 PASS. NOT a
reliability waiver. NOT evidence that BA_sep passed. NOT a replacement for
the observed negative reliability evidence.** `synthetic_vs_real_spoof_probe`
remains permanently FAILED (`C9_DETECTOR_BA_SEP_OPTION1_V2`,
`720a2e344017d588d71005b81fdf0e7d2062081ae2f3881a61a306d952dc4ac8`;
BA_sep_RND=0.7843079833902619, BA_sep_DET=0.8514170182841069,
BA_sep_LLM=0.7902658339472685). `DETECTOR_RELIABILITY_LOCK_C` remains
`overall=FAILED`, `c9_may_close=false`. `C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2`
remains `overall=FAIL` (color PASS all arms; JPEG FAIL via RND's tail only;
resize FAIL via all three arms' tail only). The original C9 confirmatory
path remains `BLOCKED_BY_DETECTOR_RELIABILITY_FAILURE`. This document
freezes a SEPARATE, EXPLORATORY, POST-FAILURE branch —
`POST_FAILURE_EXPLORATORY_TARGET_V1`
(`configs/evaluation/post_failure_exploratory_target_v1.yaml`, identity
`8fb806d25a80ecd3c7d44cfeba8c893a5f115b8b51797220a51132ba16708b51`) — that
asks a secondary, hypothesis-generating/descriptive question: given the
already-frozen C8 detectors, how do the different recipe-generator arms and
detector variants generalize to SiW-Mv2, despite the failed reliability
gate? Nothing here reopens, weakens, or is presented as a substitute for
the confirmatory evidence above.

---

## 1. Why the original path cannot continue

`barrier_state`'s `overall=FAILED` is sticky under the current BA_sep
protocol version once one required test has genuinely failed; C9's
`SOURCE_MATRIX_LOCK_C` cannot close, and C10-C13 cannot scientifically run
downstream of it. Reopening a confirmatory path would require either
accepting Option A (closure) or a new, separately versioned reliability
protocol — neither is what this task does. This branch does not touch that
decision; it exists entirely outside it.

## 2. Legacy M10 config audit (section 3/20)

`configs/evaluation/m10_target.yaml` (the ORIGINAL confirmatory G7/G8
target-evaluation contract) currently declares `target_labels_revealed:
true`. Git history (`git log -p -- configs/evaluation/m10_target.yaml`)
shows this file transitioned `false -> true` in commit `ee80048 "chore:
finalize M10 blind evaluation"`, an engineering-fixture exercise of the
ORIGINAL M10 code path's own one-way reveal CLI
(`prism_fas.cli.m10.py`), never a genuine scientific reveal reachable under
the current, FAILED reliability gate (`C11` — which is the only stage
permitted to resolve label-free SiW features toward that confirmatory
path — requires `reports/full/c9/SOURCE_MATRIX_LOCK_C.json`, which cannot
exist while `DETECTOR_RELIABILITY_LOCK_C.overall=FAILED`). This protocol
therefore treats `m10_target.yaml` strictly as **LEGACY CONTRACT / SOURCE
OF EXISTING DESIGN DECISIONS**, never loads it as its own state, and never
modifies it (`git diff --stat` proven empty by test). The new exploratory
config starts from a clean, explicit `target_labels_opened: false`,
`target_labels_revealed: false`, `target_predictions_observed: false`,
`target_metrics_observed: false`.

## 3. Reused legacy Python components (never reimplemented)

The legacy M10 evaluation library
(`src/prism_fas/evaluation/{target_prediction,firewall,video_aggregation,scoring}.py`)
is a complete, real, working implementation of exactly the blind-prediction
/ label-unlock-scoring contract this exploratory branch needs. Every
function below is reused **verbatim**, never duplicated:

| component | reused from |
|---|---|
| Table 57 prediction schema + forbidden columns | `target_prediction.PREDICTION_SCHEMA` / `FORBIDDEN_PREDICTION_COLUMNS` |
| label-free inference batch contract | `target_prediction.TargetInferenceBatch` |
| variant capability resolution | `target_prediction.VariantCapabilities` |
| prediction row construction, validation, identity | `build_prediction_row` / `validate_predictions` / `prediction_logical_identity` |
| prediction I/O | `write_predictions` / `read_predictions` |
| per-row prediction lock | `build_prediction_lock` / `write_prediction_lock` / `validate_prediction_lock` |
| cross-row lockset (this branch's `TARGET_PREDICTION_LOCK.json`) | `build_lockset` |
| inference config identity | `inference_config_hash` |
| checkpoint loading for inference | `load_checkpoint_for_inference` |
| label-free target batch streaming | `target_batches` |
| the inference driver | `predict_target` |
| the structural firewall | `firewall.FirewallConfig` / `TargetLabelFirewall` |
| video aggregation (trimmed mean, `trim_count=0` at 4 frames) | `video_aggregation.aggregate_frames` / `aggregation_config` |
| frame+video scoring | `scoring.score` / `load_evaluation_labels` |
| the no-training-capability AST audit | `scoring.static_import_audit` |
| the one-way label reveal | `scoring.target_label_reveal` |
| checkpoint/config construction for ANY row (verified track-agnostic — Track G AND Track R) | `synthetic_real_probe.construct_row_trainer` |
| the 24-row filter | `SourceRow.target_prediction_required` (`source_matrix.build_plan`) |
| checkpoint/calibration evidence | `source_evidence.load_row_evidence` |
| the real, cross-checked C8 matrix identity | `post_failure_diagnostics_v2.bind_c8_matrix_identity` |

Intentionally changed semantics (documented, never silent): `target_labels_revealed`
starts `false` here regardless of the legacy file's state;
`prediction_root` is `runs/exploratory_target_v1` (its own namespace, not
the legacy `runs`); hypotheses are named `E-H1..E-H4` (never `C-H*`,
never presented as confirmatory evidence); exactly four frozen comparisons
are declared, never invented after seeing a result.

## 4. The 24-row target matrix (section 5)

Resolved via `source_matrix.build_plan()` filtered on
`row.target_prediction_required` — never a hand-picked or renamed subset,
verified against the REAL repository plan by test:

- Track G, P3-ready: 15 rows — `RND`/`DET`/`LLM` × 5 seeds
  (`20260806..20260810`)
- Track R, P3: 9 rows — `C-R-DET` × 3 seeds, `C-R-LLM` × 3 seeds,
  `C-R-NOPROMPT` × 3 seeds
- **Total: 24**

`resolve_target_matrix` fails closed if the count or any breakdown drifts
from these exact numbers. No retraining, no best-seed selection, no
dropped arms — asserted by test against the actual function source.

## 5. Checkpoint + calibration binding (sections 5, 7)

Per row: `checkpoint_sha256`/`path`/`kind`, `decision_logit_name`/
`decision_score_name`/`decision_graph_hash` from the row's own
`run_manifest.json`; `calibration_hash`, `threshold`, `temperature` from
its own `calibration.json`. Fails closed unless `calibration.split ==
"source_dev"` (the exact source-only threshold enforcement C9's own
evidence reader already requires) and `target_labels_resolved == 0`. No
target quantile is ever read to set an operating threshold —
`resolve_row_binding`'s own source never references `target_feature_root`,
`target_label_root`, or the target package id, proven by test.

## 6. Target feature package (section 4)

Expected identity `c3a29e695ad08c4b31e01533f1d12374f4e30c51f0167c6622cf8168792e48a8`
(1700 videos: 785 live, 915 spoof, 14 attack families, 4 frames/video) is
bound from the legacy frozen contract as `EXPECTED /
TO_BE_VERIFIED_ON_GPU_BEFORE_FIRST_TARGET_ACCESS`. `verify_target_feature_package_expected`
computes the real identity (sorted `(relative_path, sha256)` pairs,
identical algorithm to `pipeline.adapters.c10`'s own private
`_package_identity`, independently implemented since that function is
private) ONLY if the package is present on disk, and fails closed on any
mismatch. On this laptop the package is absent — the function reports
`present_on_this_host: false`, never a fabricated verified identity.

## 7. Target label root (section 4, 9)

Declared (`data/evaluation_only/prism_target_v2_labels/siw_target_labels.parquet`)
so the firewall's permission table can structurally forbid it to every
stage but the Phase-E2 scorer, and even then only after a `FROZEN`
prediction lock exists. Declaring the path is not access — proven by
`TargetLabelFirewall.assert_cannot_resolve_labels("G7")`, reused verbatim.

## 8. Two-phase firewall (section 9)

**Phase E1 (blind prediction, `post_failure_exploratory_target.py`):** may
open the target feature package and the frozen checkpoint/calibration;
forbidden every label-bearing column (Table 57's forbidden-column list,
reused, unmodified). Writes predictions plus a per-row `PREDICTION_LOCK.json`,
then an overall `TARGET_PREDICTION_LOCK.json` (this branch's lockset,
reusing `build_lockset` verbatim) binding the exploratory protocol identity,
the target-matrix identity, the C8 matrix identity, every row's checkpoint/
calibration/inference-config hash, prediction row/video counts, and
`target_labels_opened: false`. Once locked, predictions are immutable.

**Phase E2 (label unlock + scoring, `post_failure_exploratory_target_scorer.py`):**
may run only after that lock's `status == "FROZEN"`. Holds no
checkpoint-loading capability — `assert_no_training_capability` (reusing
`scoring.static_import_audit`) proves it from the module's own AST, and a
regression test greps the source for `torch`/`construct_row_trainer`/
`M9Trainer`/`load_checkpoint` and finds none. Never retrains, recalibrates,
selects a checkpoint, or mutates a prediction.

## 9. Thresholds, reject policy (sections 7, 8)

Every row's operating threshold is its own frozen `source_dev` calibration
— never a target quantile. EER may be computed as a descriptive metric but
never feeds back into model selection, the ACER operating threshold,
checkpoint selection, or future inference. No source-only reject/unknown
fit has been performed in this repository; `unknown_threshold: null`,
`reject_policy: NOT_APPLICABLE_NOT_FITTED`, and every reject-dependent
metric is `NOT_APPLICABLE` with this explicit reason — never derived from a
target quantile, never silently turning a low-confidence row into a reject.

## 10. Aggregation, metrics, comparisons, statistics (sections 10-13)

4 frames/video; `trimmed_mean` at `trim_fraction=0.10` reduces to the plain
mean (`floor(4*0.10)=0`, stated explicitly); grouped by `video_id`, frame
order `sample_id`; reused verbatim from `video_aggregation.py`. Frame and
video-level APCER/BPCER/ACER/ROC-AUC/EER/ECE/Brier/NLL; HTER retained only
as an explicitly labeled evaluation-only descriptive metric. Per-seed values
are never hidden; cross-seed aggregation is mean/std (`ddof=0`). Four,
and only four, frozen exploratory comparisons: **E-H1** (Track-G generator:
RND/DET/LLM), **E-H2** (Track-R DET vs LLM), **E-H3** (Track-R LLM vs
R-NOPROMPT), **E-H4** (Track-G vs Track-R, matched arms DET/LLM only — RND
has no Track-R row). Statistics reuse the legacy contract's frozen values
(`bootstrap_unit=video`, `resamples=10000`, `seed=20260810`,
`confidence=0.95`, percentile, paired, Holm-Bonferroni over the E-H1..E-H4
family) — implemented as pure, tested functions
(`paired_bootstrap_acer_difference`, `holm_bonferroni`) in the scorer
module. Every exploratory p-value/CI is explicitly `EXPLORATORY_COMPARISON`
evidence, never confirmatory evidence for C-H1..C-H5.

## 11. No-rerun / immutability (section 17)

Prediction: no artifacts on disk → run once; a complete `FROZEN` lock →
validate/re-report only, zero model inference (`checkpoint_weights_loaded:
false`, `images_forwarded: false`); a partial prediction set (some but not
all 24 rows) → BLOCK (`PARTIAL_SCIENTIFIC_RESULT_SET`). Scoring: the same
three-way contract over per-row score files and the combined
`EXPLORATORY_TARGET_SCORE_RESULT.json`. All four transitions are proven by
fixture test (never against real GPU data).

## 12. What this freeze does NOT do

Does not open the target feature package or the target label artifact. Does
not run SiW-Mv2 inference. Does not compute a target metric. Does not
fabricate a GPU-verified package identity. `--preflight-only`,
`--bind-prediction-plan`, `--status`, and `--predict` (predictor) and
`--preflight-score`/`--score` (scorer) were each run once against this real
repo on this laptop; the matrix resolution and its exact 24-row/15+9
breakdown succeeded against the REAL repository plan (proof the matrix
audit is correct), while every step needing GPU checkpoints, the target
package, or target labels correctly reported an unresolved precondition,
exit 2, writing nothing. `target_access = 0` throughout.

## 13. Tests

`tests/pipeline/test_post_failure_exploratory_target_v1.py` — 57 tests
covering all twenty-six required items (A-Z): deterministic protocol
identity; namespace disjointness from C9-C13, BA_sep, and both diagnostics
versions; `c9_may_close` cannot be set true; BA_sep/detector-reliability/
diagnostics-V2 failure bindings present; the real 24-row matrix and its
Track-G/Track-R breakdown; no best-seed-selection language; checkpoint and
calibration binding (including a `source_dev`-only fail-closed case and a
`target_labels_resolved != 0` fail-closed case); the predictor's import
statements never mention the label reader; the prediction schema's
forbidden-column list; the scorer's no-training-capability audit and its
literal absence of `torch`/checkpoint-construction references; the scorer's
refusal before any lock exists and before a `FROZEN` one exists; no attack
taxonomy in the predictor's source; threshold binding never touches target
paths; the reject policy; partial/complete prediction and partial/complete
scoring no-rerun behavior (fixture-driven); the legacy `m10_target.yaml`'s
byte-identical, untouched state; and `target_access=0` in every mode.

Combined C9-scoped suite (eight files, including this one): 428/428 pass.
