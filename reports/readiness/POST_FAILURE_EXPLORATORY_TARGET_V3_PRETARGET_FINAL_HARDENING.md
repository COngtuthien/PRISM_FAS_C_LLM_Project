# POST_FAILURE_EXPLORATORY_TARGET_V3 — final pre-target hardening

**FINAL PRE-TARGET CORRECTION.** `POST_FAILURE_EXPLORATORY_TARGET_V2`
(`configs/evaluation/post_failure_exploratory_target_v2.yaml`, identity
`2f1beb0b95f01051e06c0ef8a82d06a759d0fe8f81f693c5d3a4d777845196a9`) was
frozen but never scientifically executed: no target feature package has
been scientifically opened, no SiW-Mv2 prediction generated, no target
label opened, no target metric observed. A final pre-target audit found
provenance/access/statistics defects, corrected here in
`POST_FAILURE_EXPLORATORY_TARGET_V3`
(`configs/evaluation/post_failure_exploratory_target_v3.yaml`, identity
`a2b54f8844a2a36540e62470c2f5f30de52fbf509a37f03feb7f6d769d5c702c`) — the
final corrected protocol before any target access. **V1 and V2 are
preserved byte-for-byte, unchanged; V3 never writes into either
namespace.**

BA_sep remains permanently FAILED; `DETECTOR_RELIABILITY_LOCK_C` remains
`overall=FAILED`; `POST_FAILURE_SOURCE_DIAGNOSTICS_V2` remains
`overall=FAIL`; the original C9 confirmatory path remains BLOCKED. No
target feature or label has been accessed, on this laptop, by this task.

---

## Defects corrected

**A — per-row target package identity was empty.** `predict_one_row`
(V2) read `binding.get("target_feature_package_identity", "")`, which the
binding never actually set — every prediction could carry an empty
identity. **Corrected:** every row's binding now carries the REAL,
top-level VERIFIED `target_feature_package_identity`
(`resolve_all_row_bindings_v3`), required non-empty
(`ExploratoryTargetV3Error` on an empty value); `inference_config_hash`,
the per-row lock, and the hardened validator all require exact equality
with it.

**B — code commit never bound to a prediction.** `target_prediction.build_prediction_lock`
already accepts a `code_commit` parameter, but V2 never populated it.
**Corrected:** `current_code_commit` (reusing `detector.checkpoint.git_commit`
verbatim) is read ONCE at execution start
(`prediction_execution_code_commit`), passed into every row's lock and the
overall lockset. A second `--predict` re-verifies the recorded lineage
without rerunning anything; a lockset built with a per-row `code_commit`
disagreeing with the execution commit is refused.

**C — `target_access` was scientifically ambiguous.** A single
`target_access: 0` conflated feature-identity reads, prediction-feature
reads, and label reads. **Corrected:** `target_access_state` — three
explicit booleans (`target_feature_identity_accessed`,
`target_prediction_features_accessed`, `target_labels_accessed`) plus
counts — frozen with exact before/after semantics for each access type. The
protocol refuses to load if it starts with any flag true.

**D — the label reveal artifact was declared but never written.** V2's
scorer called `load_evaluation_labels` directly. **Corrected:**
`reveal_target_labels` writes a real, one-way `TARGET_LABEL_REVEAL.json`
BEFORE the first label load: it validates the full E1 lock, verifies the
scorer's no-model-capability, verifies the label artifact exists, hashes it
WITHOUT scoring, then atomically creates the reveal (exact match reuses it,
any mismatch — including the label file itself being tampered with after
reveal — BLOCKS; it is never overwritten).

**E — score completeness was under-specified.** V2 would rescore (and
reopen labels) if exactly 24 per-row score files existed with no final
result. **Corrected:** that state is `INCOMPLETE_FINALIZATION` and BLOCKS —
no label reopen, no row rewrite. An unexpected extra score-row file also
BLOCKS (`UNEXPECTED_SCORE_ROW_FILES`).

**F — the score-result validator is now comprehensive.**
`validate_existing_exploratory_score_result_v3` cross-checks the label
reveal identity, the prediction execution code commit, the exact 24 row
IDs, the exact matched-seed set per comparison, and recomputes
Holm-Bonferroni from the RECORDED randomization p-values (never
recomputing a metric from raw predictions, never reopening labels, never
loading a model).

**G/H/I — the bootstrap CI and the exploratory p-value are now separate.**
V2's `class_stratified_paired_bootstrap` derived its p-value from the same
bootstrap distribution used for the CI. **Corrected:** the class-stratified
bootstrap (`class_stratified_bootstrap_ci`) is CI-only (its return value
carries no `p_value_two_sided` field, verified by test); a SEPARATE,
frozen **paired video-level sign-flip randomization test**
(`paired_randomization_test`) produces the p-value:

```
for each video, its per-seed method-difference contribution is averaged
across ALL matched seeds first (video is the paired observational unit;
averaging is linear, so sign-flipping the averaged value is identical to
flipping the sign consistently across every seed for that video)
  LIVE videos:  contribution = mean_seed(BPCER_error_A - BPCER_error_B) * 1/(2*N_live)
  SPOOF videos: contribution = mean_seed(APCER_error_A - APCER_error_B) * 1/(2*N_spoof)
sum(contributions) == mean_seed(ACER_A - ACER_B)  (the observed statistic)
10000 replicates: independently flip each video's contribution's sign
p = (1 + count(|T_perm| >= |T_obs|)) / (1 + 10000)   — ">=" inclusive of ties
```

Verified by test: identical arms give `p=1.0`; very different arms give a
small p; a zero observed statistic makes every replicate satisfy
`|T_perm| >= 0`, giving the maximal finite-sample p (`(1+N)/(1+N)`).

**J — matched-seed sets are now asserted exactly.** V2 used a silent
`set(a) & set(b)` intersection. **Corrected:** every comparison's matched
seeds are checked against a frozen, named requirement
(`REQUIRED_MATCHED_SEEDS`); a genuinely missing required seed BLOCKS. A
comparison's side may still legitimately carry additional seeds it needs
for a *different* comparison (e.g. Track-G DET's 5 seeds also serve E-H1)
— only a missing REQUIRED seed is an error, never an unrelated extra one.

**Cross-seed summary (section 12) — actually implemented.**
`build_cross_seed_summary` emits, for each of the 6 configurations
(`C-G-RND`/`C-G-DET`/`C-G-LLM`/`C-R-DET`/`C-R-LLM`/`C-R-NOPROMPT`) and each
of the 8 metrics (APCER/BPCER/ACER/ROC-AUC/EER/ECE/Brier/NLL), the per-seed
values plus `mean`/`std(ddof=0)` over the complete frozen seed family —
verified by test that every frozen seed is present and that predictions
are never pooled across seeds before this summary.

**Crash-safe first prediction (section 14).** `predict_one_row_to_staging`
writes ONLY into a disposable staging namespace
(`runs/exploratory_target_v3/.staging/<execution_identity>/`), each row
explicitly marked `ENGINEERING_STAGING_NOT_SCIENTIFICALLY_LOCKED`.
`promote_staged_rows` atomically renames each staged row directory into its
final scientific location ONLY after all 24 rows succeed; the overall
`TARGET_PREDICTION_LOCK.json` is written last. A crash before promotion
leaves only disposable staging — no scientific result was ever frozen.
`_detect_partial_state` scans only final row directories, never staging.

## Namespaces and reuse

- `configs/evaluation/post_failure_exploratory_target_v3.yaml`,
  `reports/full/exploratory_target_v3/`, `runs/exploratory_target_v3/`
  (with its own `.staging/` subtree), `state/exploratory_target_v3/` —
  disjoint from V1 and V2 (proven by test).
- V1's and V2's own modules are completely unmodified (`git diff --stat`
  empty, proven by test); V3 imports their resolution functions verbatim
  (`resolve_target_matrix`, `target_matrix_identity`,
  `bind_c8_matrix_identity`, `verify_target_feature_package_expected`/
  `_required`, `verify_target_label_root_sealed`, `build_firewall`,
  `resolve_all_row_bindings_v2`) rather than duplicating them.
- The legacy M10 library remains reused verbatim, including
  `build_prediction_lock` — V3 simply, finally, populates the
  `code_commit` parameter that function already supported.

## What this correction does NOT do

Does not open the target feature package or label artifact. Does not run
SiW-Mv2 inference. Does not compute a target metric. Every CLI mode
(`--preflight-only`, `--bind-prediction-plan`, `--status`, `--predict` on
the predictor; `--preflight-score`, `--score` on the scorer) was run once
against this real repo; the real 24-row matrix resolution succeeded; every
step needing GPU checkpoints, the target package, or target labels
correctly reported an unresolved precondition, exit 2, writing nothing.
`target_access_state` stayed all-false throughout.

## Tests

`tests/pipeline/test_post_failure_exploratory_target_v3.py` — 50 tests
covering the full A-AJ list: V1/V2 byte-identical/unchanged; V3's distinct
identity and disjoint namespace; per-row package-identity non-emptiness and
equality with the top-level value; the inference-config hash and per-row
lock binding that identity; code-commit binding and mismatch detection;
the `target_access_state` transitions (refusing a protocol that starts
opened, binding = feature-identity-only, lockset = feature+prediction);
the label reveal's mandatory/idempotent/conflict/tamper behavior and its
strict before-load ordering; the four-way score state machine (zero/
partial/24-without-final/complete) plus extra-row rejection; score-result
identity validation; the bootstrap CI's determinism, seed retention, and
p-value-free shape; the randomization test's determinism, finite-sample
correction, and `>=` tie handling; exact matched-seed assertion (missing
blocks, unrelated extra seeds on one side do not); all seven comparisons
entering one Holm family; the cross-seed summary's completeness,
`ddof=0`, and no-pooling; the scorer's no-training-capability audit; and
that no laptop-run mode ever opens a target label.

Combined C9-scoped suite (eleven files): 530/530 pass.
