# POST_FAILURE_EXPLORATORY_TARGET_V3 — implementation reconciliation

**NO NEW PROTOCOL VERSION.** This is not V4. `configs/evaluation/post_failure_exploratory_target_v3.yaml`
is untouched (`git diff` against the starting HEAD
`c02113bb7a4296fc61860acd2ca41df06f347d31` is empty) and its protocol
identity is unchanged:
`a2b54f8844a2a36540e62470c2f5f30de52fbf509a37f03feb7f6d769d5c702c`. No
scientific decision — metric, threshold, hypothesis/comparison family, the
24-row matrix, seeds, bootstrap design, randomization design, Holm family,
access semantics, target package identity, or the label firewall — changed.
This task reconciles eight IMPLEMENTATION defects found by a pre-target
audit of the already-frozen V3 code, in
`src/prism_fas/evaluation/post_failure_exploratory_target_v3.py` and
`..._v3_scorer.py`, edited in place.

BA_sep remains permanently FAILED; `DETECTOR_RELIABILITY_LOCK_C` remains
`overall=FAILED`; `POST_FAILURE_SOURCE_DIAGNOSTICS_V2` remains
`overall=FAIL`; the original C9 confirmatory path remains BLOCKED. No
target feature, prediction, or label was accessed on this laptop by this
task.

---

## Defects corrected

**A — E1's "atomic promotion" was only per-row.** `promote_staged_rows`
renamed each staged row directory one at a time with no crash-recovery
manifest; a crash after promoting `N < 24` rows left a permanently partial
final state with no way to resume without risking either re-inference or a
silently incomplete result. **Corrected:** a
`PREDICTION_PROMOTION_TRANSACTION_<execution_id>.json` manifest
(`build_promotion_transaction`) is written with `state: READY_TO_PROMOTE`
BEFORE any row is renamed, binding the protocol identity, the plan binding
identity, the execution identity, the exact row-ID set, and each staged
row's `prediction_file_sha256` / `lock_file_sha256`. Recovery
(`promote_staged_rows` called again) validates every already-promoted row
in its final directory, and every still-staged row in its staging
directory, against those recorded hashes — a mismatch BLOCKS — then
resumes file-renames only (zero model inference), and the manifest is
marked `COMPLETE` only after all rows are confirmed promoted. `_predict`
locates a matching transaction purely from `(prediction_lock_identity-free)`
`(plan_binding_identity, code_commit)` — i.e. the SAME deterministic
`execution_identity` already used for the staging path — and, when found,
reconstructs `row_results` by reading the already-written per-row
`PREDICTION_LOCK.json` files back from disk (`_row_result_from_recovered_artifacts`)
instead of re-invoking `predict_one_row_to_staging`. Proven by test: a
simulated crash after 1 and after 2 of a 3-row fake matrix recovers to the
identical final `TARGET_PREDICTION_LOCK.json`, with the inference
call-counter at exactly zero during recovery.

**B — E2's row metadata still violated the post-E1 authority rule.**
`_score` called `post_failure_exploratory_target.resolve_target_matrix` —
the V1 source-matrix resolver — to build `row_meta`, even though the
frozen rule (established when E1 closes) is that the validated
`TARGET_PREDICTION_LOCK.json` is the sole scientific authority for row
identity from that point on. **Corrected:** `_row_meta_from_lockset` builds
`row_meta` exclusively from the lockset's own `entries` (`row_id`,
`experiment_id`, `track`, `arm`, `seed`, `prediction_variant_id`,
`threshold`, `checkpoint_sha256`, `calibration_hash`,
`prediction_lock_identity`, `prediction_logical_identity`); `_score` no
longer imports or calls `resolve_target_matrix` at all. Proven by test:
poisoning `resolve_target_matrix` to raise does not affect `_score`'s
outcome.

**C — the label reveal conflated two distinct commits.**
`first_authorized_reveal_code_commit` was set to
`lockset["prediction_execution_code_commit"]` — the E1 inference commit —
never a genuinely fresh E2 commit. **Corrected:** `build_label_reveal` now
takes two separate parameters, `prediction_execution_code_commit` (read
verbatim from the frozen E1 lockset, unchanged) and
`first_authorized_reveal_code_commit` (a FRESH `git rev-parse HEAD` read at
reveal time via a new local helper, `_current_scorer_git_commit`, that
shells out directly rather than importing
`prism_fas.detector.checkpoint.git_commit` — that import stays forbidden
in this scorer's `FORBIDDEN_IMPORTS`, since it can pull in model/checkpoint
machinery transitively). Proven by test: the two commit fields differ when
the underlying commits differ, and `assert_no_training_capability()` still
passes.

**D — `target_access_state` was missing or misleading on real artifacts.**
**Corrected:** the label reveal and the final score result each carry an
explicit `access_state` (`target_feature_identity_accessed`,
`target_prediction_features_accessed`, `target_labels_accessed` all `true`,
`target_feature_access_count`/`target_label_access_count` both `1`).
Counts have a single documented meaning each (authorized feature-access
phases; first-authorized-reveal transitions) and never increment on
idempotent reuse — proven by test: calling `reveal_target_labels` twice
leaves `target_label_access_count == 1` both times.

**E — label tampering after reveal was not validated.**
`validate_existing_exploratory_score_result_v3` checked the reveal's
identity but never re-verified the label artifact's bytes. **Corrected:**
the validator now (only when a reveal already exists on disk — never
before) re-hashes the CURRENT label file and compares it against the
frozen reveal's `target_label_artifact_sha256`, and recomputes the
reveal's own `reveal_identity`; either mismatch BLOCKS with a message
naming possible tampering. Proven by test: tampering with the label file's
bytes after a reveal exists is detected; no code path hashes label bytes
when no reveal exists yet.

**F — per-row score artifacts had no immutable identity.** `score_one_row`
returned the raw `scoring.score()` payload with no binding to its
originating prediction lock. **Corrected:** every per-row score file is now
wrapped in a self-hashing envelope — `row_id`, `prediction_lock_identity`,
`prediction_logical_identity`, `checkpoint_sha256`, `calibration_hash`,
`seed`, `track`, `arm`, `prediction_variant_id`, the original metrics under
`metrics`, and `score_artifact_identity` (a hash of the whole envelope).
The validator now checks: exactly 24 files exist, no extras, each file's
CURRENT SHA-256 matches the frozen `per_row_score_artifacts` entry, each
file's internal `score_artifact_identity` is self-consistent, and each
file's identity fields cross-check against the frozen lockset entry.
Proven by test: a one-byte tamper to any per-row file is detected;
missing/extra files are detected.

**G — the final score result's provenance was incomplete.**
**Corrected:** `EXPLORATORY_TARGET_SCORE_RESULT.json` now additionally
binds `scoring_execution_code_commit` (the E2 execution's own fresh
commit), `target_label_artifact_sha256`, `target_feature_package_identity`,
and `per_row_score_artifacts` (all 24 row identities/hashes), alongside the
already-frozen `prediction_lock_identity`, `prediction_execution_code_commit`,
`label_reveal_identity`, `cross_seed_summary`, `exploratory_comparisons`,
`access_state`, and the immutable failure-context fields
(`ba_sep_observed_verdict=FAIL`, `detector_reliability_lock_c_observed_overall=FAILED`,
`post_failure_diagnostics_v2=FAIL`, `c9_original_confirmatory_path=BLOCKED`,
`c9_may_close=false`). `score_result_identity` hashes the complete,
extended result.

**H — E2 scoring was not crash-recoverable.** A crash between scoring some
rows and writing the final result risked either a re-scored row or a
second, redundant label load. **Corrected**, mirroring Defect A: all 24
rows are scored into a disposable `.score_staging/<execution_id>/`
namespace first; a `SCORE_PROMOTION_TRANSACTION_<execution_id>.json`
manifest (`build`/validate via `promote_staged_score_rows`) is written
before any row file is promoted into `reports/full/exploratory_target_v3/scores/`;
recovery validates already-promoted and still-staged rows against the
manifest's hashes and resumes file-renames only. On recovery, `_score`
reads the ALREADY-WRITTEN `TARGET_LABEL_REVEAL.json` directly (never
re-derives or re-hashes it) and reconstructs `scored_rows` by reading each
row's already-computed JSON back from disk
(`_recover_scored_rows_from_disk`) — zero rescoring, zero label reopen.
`cross_seed_summary`/`exploratory_comparisons` (pure, deterministic
recomputation from already-scored data) are still recomputed on recovery,
as this is not label access. Proven by test: a simulated crash after one of
24 rows promoted recovers to the identical final result, with the scoring
call-counter, the label-load call-counter, and the reveal call-counter all
at exactly zero during recovery.

## Statistics — unchanged, byte-for-byte

ACER definition, the class-stratified paired-video bootstrap (10000
resamples, seed 20260810, CI-only), the paired video-level sign-flip
randomization test (10000 resamples, seed 20260810, two-sided, finite-sample
`+1` correction, `>=` tie handling), the seven atomic comparisons and their
`REQUIRED_MATCHED_SEEDS`, Holm-Bonferroni with `family_size=7` over
randomization p-values only, and the cross-seed mean/std(`ddof=0`) summary
are all byte-for-byte unchanged — verified by rerunning the original,
unmodified `tests/pipeline/test_post_failure_exploratory_target_v3.py`
(50/50 pass) alongside the new reconciliation suite.

## What this correction does NOT do

Does not create a new protocol version. Does not touch
`configs/evaluation/post_failure_exploratory_target_v3.yaml`,
`post_failure_exploratory_target_v1.yaml`, `..._v2.yaml`, or either V1/V2
module (`git diff --stat` against the starting HEAD empty, proven by test).
Does not open the target feature package or label artifact. Does not run
SiW-Mv2 inference. Does not compute a target metric.

## Tests

`tests/pipeline/test_post_failure_exploratory_target_v3_reconciliation.py`
— 25 new tests covering all eight defects, including two crash-recovery
integration tests (Defect A: `--predict` recovers from a simulated crash
after 1 and after 2 of a 3-row fake matrix's promotions, with zero
inference; Defect H: `--score` recovers from a simulated crash after 1 of
24 rows promoted, with zero rescoring and no second label reopen).

The original, unmodified `tests/pipeline/test_post_failure_exploratory_target_v3.py`
(50 tests) and the V1/V2 suites (57 + 52 tests) all pass unchanged,
confirming zero regression and byte-identical historical behavior.

## Files changed

- Edited in place: `src/prism_fas/evaluation/post_failure_exploratory_target_v3.py`,
  `src/prism_fas/evaluation/post_failure_exploratory_target_v3_scorer.py`
- Added: `tests/pipeline/test_post_failure_exploratory_target_v3_reconciliation.py`,
  this report and its `.json` companion
- Updated: `reports/readiness/POST_FAILURE_EXPLORATORY_TARGET_V3_GPU_HANDOFF.md`
  (same eight-step authorization scope, still stopping before `--predict`),
  `docs/PROJECT_STATE.md`
- Untouched (verified by test): `configs/evaluation/post_failure_exploratory_target_v1.yaml`,
  `..._v2.yaml`, `..._v3.yaml`, `src/prism_fas/evaluation/post_failure_exploratory_target.py`,
  `..._scorer.py`, `..._v2.py`, `..._v2_scorer.py`
