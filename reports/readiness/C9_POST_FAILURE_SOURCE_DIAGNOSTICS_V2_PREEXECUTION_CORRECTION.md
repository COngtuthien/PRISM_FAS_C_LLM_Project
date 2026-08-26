# C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2 — pre-execution scientific correction

**PRE-EXECUTION CORRECTION, NOT A REVISION OF AN OBSERVED RESULT.** V1
(`configs/evaluation/c9_post_failure_source_diagnostics_v1.yaml`, identity
`cb05271e26d9a421f2f9277599523e185026e1eab644febc07c75432d26f3fc5`) was never
scientifically executed:
`no_diagnostic_metric_observed_before_freeze: true`, and every mode of its
runner, run once against this real repo, correctly reported an unresolved
precondition (no CUDA, no M3B package) and wrote nothing. This document
freezes a corrected, separately identified protocol —
**`C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2`**,
`configs/evaluation/c9_post_failure_source_diagnostics_v2.yaml`, identity
`05ffa20ee71ff7168732436be6b8a98b613351d434f993226ff4308ed32a5523` — after a
pre-execution scientific audit found five defects in V1's design. **V1 is
preserved byte-for-byte, unchanged, as historical pre-execution design
evidence; V2 never writes into V1's artifact namespace.**

`synthetic_vs_real_spoof_probe` remains permanently, observedly FAILED under
`C9_DETECTOR_BA_SEP_OPTION1_V2`
(`720a2e344017d588d71005b81fdf0e7d2062081ae2f3881a61a306d952dc4ac8`;
BA_sep_RND=0.7843079833902619, BA_sep_DET=0.8514170182841069,
BA_sep_LLM=0.7902658339472685, all over 0.75). `DETECTOR_RELIABILITY_LOCK_C`
remains `overall=FAILED`, `c9_may_close=false`. Nothing in V2 changes,
reruns, or can reopen either. `target_access=0` throughout.

---

## 1. Starting state

| field | value |
|---|---|
| starting HEAD | `fcee1e9074a65b07d0a7112f03a5dc3b77b349ad` |
| V1 protocol identity | `cb05271e26d9a421f2f9277599523e185026e1eab644febc07c75432d26f3fc5` |
| V1 scientifically executed | NO |
| BA_sep verdict | FAIL (immutable, observed) |
| DETECTOR_RELIABILITY_LOCK_C | FAILED, `c9_may_close=false` |
| GPU diagnostic `--execute` run | NO |
| target access | 0 |

## 2. Defect A — benign corruption threshold was self-normalizing

**Problem.** V1 derived `threshold = calibration_mean_shift + 3 *
calibration_std_shift` from the SAME corruption's own calibration-group
shift, then tested that corruption's evaluation-group shift against it. A
corruption causing a large but consistent spoof-score increase would inflate
its own threshold and could still PASS — not a faithful test of "no
systematic large increase in spoof score under benign corruption."

**Audit.** Searched the repository for a pre-existing, source-only, benign
reference-variability population frozen independently of this protocol.
Found one: `src/prism_fas/synthesis/quality_calibration.py`
(`BENIGN_VARIANTS` — `brightness_098`, `brightness_102`, `contrast_098`,
`contrast_102`; `BENIGN_NOISE_STD = 0.002`), frozen at M8 for synthesis
quality-gate calibration (tau_id/tau_lm/tau_parse), on `source_train`, over
an unrelated metric space (identity cosine, landmark NME, parsing dice) —
independent of this diagnostics protocol both in purpose and in perturbation
family. `structural_calibration.py` carries an analogous ±10% brightness/
contrast family for a different threshold pair; the ±2%/noise-std family was
selected as the closer match to "mild" as the task specified.

**Correction — disjoint calibration design, frozen exactly:**

```
REFERENCE BENIGN CONTROLS (M8, pre-existing, independent)
    -> forwarded on the CALIBRATION group ONLY
    -> tau_mean = reference_mean_delta_plus + 3 * reference_std_delta_plus
       tau_tail = reference_p95_delta_plus  + 3 * reference_std_delta_plus
    -> held-out JPEG/resize/color, forwarded on the EVALUATION group ONLY
    -> PASS iff evaluation.mean(delta_plus) <= tau_mean
           AND evaluation.p95(delta_plus) <= tau_tail
```

`delta_plus = max(p_after - p_before, 0)` (positive spoof-score shift only,
per the task's preference), computed identically for the reference
population and the tested corruption. The tested corruption never
contributes to its own threshold: disjoint by sample group (calibration vs.
evaluation) AND by perturbation family (reference vs. tested corruption). No
new numerical constant is invented — the brightness/contrast/noise
parameters are the exact, already-frozen M8 values, reused verbatim via
`quality_calibration.benign_variant`; the `+3*std` margin is the same
convention V1 already used, now applied to two reference statistics (mean
and p95) instead of one self-referential one. Implemented in
`post_failure_diagnostics_v2.py`:
`forward_reference_benign_evidence_for_arm`, `reference_delta_plus_for_arm`,
`derive_reference_threshold`, `corruption_verdict`,
`run_benign_corruption_diagnostic_for_arm`.

Both mean and tail criteria are required to PASS, protecting against a
systematic mean increase and a large upper-tail increase independently, per
the task's explicit requirement.

## 3. Defect B — `cross_route_synthetic` semantic mismatch

**Problem.** V1 fit a real-vs-synthetic separability probe on one synthesis
route and scored it on the other, reusing the frozen BA_sep `<=0.75` ceiling
(LOW balanced accuracy is the desired outcome there). The test's own
canonical declaration
(`prism_fas.evaluation.reliability.DECLARED_TESTS["cross_route_synthetic"]`,
spec §17.3.7) states: population "train on one synthetic route, evaluate on
the other"; measures "cross-route generalization of the synthetic evidence";
pass_rule **"performance is retained across routes"** — a retention
question (HIGH performance is desired), the opposite direction from a
separability ceiling.

**Audit.** No already-frozen, faithful definition of "performance" (which
metric), "within-route reference" (retained relative to what), or a numeric
retention threshold exists anywhere in this repository or spec §17.3.7.
Reusing the BA_sep ceiling under this name risks silently reversing the
test's declared scientific meaning.

**Correction.** `cross_route_synthetic` reclassified
`EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL` → `NEEDS_SCIENTIFIC_DECISION`,
`gpu_ready: false`. V1's separability-probe attempt
(`post_failure_diagnostics.resolve_synthetic_population_by_route`,
`cross_route_ba`, `run_cross_route_diagnostic_for_arm`) is preserved
unchanged in the V1 module and is NOT called by the V2 runner. It becomes
executable only if a normative audit or explicit user decision freezes the
performance metric, the within-route reference, and a numeric retention
threshold — none invented here.

## 4. Defect C — existing-result validation was too weak

**Problem.** V1's `--execute` (with all four result artifacts present) read
bare `verdict`/`per-test` fields and re-reported them without cross-checking
bindings, identities, or internal consistency.

**Correction.**
`post_failure_diagnostics_v2.validate_existing_diagnostics_result(repo)` is
now canonical and gates every second-or-later `--execute` and every
`--status` call. It checks: all four result artifacts and all three
bindings present and parseable; every artifact's `protocol_identity` equals
the currently active V2 protocol identity; `result.per_test ==
per_test.per_test` (cross-artifact agreement); checkpoint
hashes/cardinality/seeds and C6 bank identities re-derived and compared;
population identities (calibration/evaluation sample IDs, per-domain group
safety) re-derived byte-for-byte; the executable/blocked test set matches
the frozen V2 protocol; each test's recorded status is recomputed from its
own recorded per-arm verdicts; the overall verdict is recomputed from
RECORDED per-test statuses only (never from raw predictions); `c9_may_close`
is `False` everywhere it appears; `ba_sep_observed_verdict == "FAIL"` and
`detector_reliability_lock_c_observed_overall == "FAILED"` in every result
artifact; `target_access == 0` everywhere; and the LIVE, on-host
`DETECTOR_RELIABILITY_LOCK_C` is still `FAILED` (fails closed if it has ever
flipped to `PASSED` behind the scenes). A complete-but-invalid result set
returns `EXIT_BLOCKED = 2` with the `problems` list — no recomputation, no
overwrite.

## 5. Defect D — C8 matrix identity was placeholder text

**Problem.** V1's result/provenance artifacts recorded
`"c8_matrix_identity": "see reports/full/c8/C8_ACCEPTANCE.json ..."` — a
handwritten string, not a real identity.

**Correction.** `post_failure_diagnostics_v2.bind_c8_matrix_identity(repo)`
binds the REAL canonical identity,
`prism_fas.evaluation.source_matrix.build_plan().identity` — the exact same
value C8's own acceptance gate already writes as `C8_ACCEPTANCE.json`'s
`matrix_identity` field — and cross-checks the two, failing closed if
`C8_ACCEPTANCE.json` is absent or the identities disagree. Bound at
`--bind-only` (stored in `DIAGNOSTICS_PROTOCOL_BINDING.json`) and
re-verified fresh at `--execute` (fails closed if it has drifted since
binding); every result/provenance/verdict artifact carries the real value.

## 6. Defect E — group-safe split proven only globally

**Problem.** V1's calibration/evaluation split ensured a non-empty
calibration and evaluation set globally, but the frozen design covers two
domains (`casia_fasd`, `msu_mfsd`) and a global check cannot detect one
domain collapsing into a single group.

**Correction.**
`post_failure_diagnostics_v2.verify_per_domain_group_safety(records, split,
domains=...)` fails closed unless EVERY domain has a non-empty calibration
group set, a non-empty evaluation group set, and an empty
calibration/evaluation intersection — recording per-domain sample counts,
unique group counts, and calibration/evaluation group-set identities. Run at
`--bind-only` and re-derived by the result validator.

## 7. Re-audited executability table

| test | V1 | V2 | reason for change |
|---|---|---|---|
| `benign_jpeg_corruption` | EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL | EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL | threshold design corrected (Defect A); still executable |
| `benign_resize_corruption` | EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL | EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL | threshold design corrected (Defect A); still executable |
| `benign_color_corruption` | EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL | EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL | threshold design corrected (Defect A); still executable |
| `cross_route_synthetic` | EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL | NEEDS_SCIENTIFIC_DECISION | Defect B — semantic mismatch with the canonical declared test |
| `recipe_region_shift` | NEEDS_SCIENTIFIC_DECISION | NEEDS_SCIENTIFIC_DECISION | unchanged |
| `artifact_map_swap` | NEEDS_SCIENTIFIC_DECISION | NEEDS_SCIENTIFIC_DECISION | unchanged |
| `residual_scale_zero` | STRUCTURALLY_MODEL_BLOCKED | STRUCTURALLY_MODEL_BLOCKED | unchanged |
| `crop_padding_interpolation` | STRUCTURALLY_DATA_BLOCKED | STRUCTURALLY_DATA_BLOCKED | unchanged |

`gpu_ready_count`: V1 = 4, V2 = 3. V2 does NOT maximize executable-test count
— it reduces it, because Defect B's correction removes one test from the
executable set rather than inventing a threshold to keep it executable.

## 8. Namespaces

V1: `reports/full/c8/reliability/post_failure_source_diagnostics_v1/`
(unchanged, historical). V2:
`reports/full/c8/reliability/post_failure_source_diagnostics_v2/` (new,
disjoint — neither is a prefix of the other, proven by regression test).
Neither overlaps `reports/full/c8/reliability/synthetic_vs_real_spoof_probe/`
(BA_sep) or `reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json`.

## 9. Implementation

`src/prism_fas/evaluation/post_failure_diagnostics_v2.py` — protocol
loading/identity; C8 matrix identity binding; per-domain group-safety
verification; reference-benign forwarding/threshold derivation; the
corrected benign-corruption orchestration; the canonical existing-result
validator. Reuses, never reimplements: `jpeg_corrupt`, `resize_corrupt`,
`color_corrupt`, `CORRUPTION_FUNCTIONS`, `calibration_evaluation_split`,
`resolve_source_dev_live_records`, `forward_corruption_evidence_for_arm` from
V1's module, and `quality_calibration.benign_variant`/`BENIGN_VARIANTS`/
`BENIGN_NOISE_STD` from the frozen M8 module.

`src/prism_fas/evaluation/post_failure_diagnostics_v2_runner.py` — the CLI,
the same four modes as V1
(`--preflight-only`, `--bind-only`, `--status`, `--execute`), on the V2
namespace, with the validator wired into `--status` and every
second-or-later `--execute`.

## 10. Tests

`tests/pipeline/test_c9_post_failure_source_diagnostics_v2.py` — 59 tests:
V1 byte-identical/unchanged; V2's distinct protocol identity; the threshold
can never be derived from the tested corruption; the reference population is
the frozen M8 family; `cross_route_synthetic` correctly
`NEEDS_SCIENTIFIC_DECISION`; the validator detects a tampered result,
provenance, verdict, `c9_may_close`, and mismatched bindings; a complete
INVALID result blocks without recomputing or overwriting; a complete VALID
result re-reports with zero recomputation; the real canonical C8 matrix
identity is bound and cross-checked (agreement and mismatch); per-domain
group safety (both domains, degenerate-domain failure, split-intersection
failure, and a case proving a global-only check would have missed a
per-domain degeneracy); `target_access=0` everywhere; a FAILED BA_sep/
reliability barrier can never reopen C9 (including a live-lock-flip
detection); and the V1/V2/BA_sep namespaces never collide.

V1's own 70-test suite was re-run unmodified and still passes (proof V1's
observable behavior is untouched by this correction).

Test-environment note: this laptop sandbox initially lacked `pytest`,
`torch`, `pyarrow`, and `opencv-python-headless`; these were installed into
an isolated project-local virtualenv (`get-pip.py` bootstrap, no system
package changes) purely to execute the test suite in this session — no
scientific artifact was produced by that environment, and no GPU/CUDA
package was installed.

## 11. Results

- `tests/pipeline/test_c9_post_failure_source_diagnostics_v2.py`: 59/59 pass.
- `tests/pipeline/test_c9_post_failure_source_diagnostics.py` (V1, unmodified): 70/70 pass.
- Combined C9-scoped suite (six files): 346/346 pass.
- Broad regression (`tests/c7 tests/pipeline`): 39 failed, 1848 passed, 22
  skipped, 5 errors (330.57s). This differs from the previously documented
  V1-freeze baseline (33 failed / 1790 passed / 22 skipped / 0 errors)
  because this session's sandboxed environment lacks a real GPU/CUDA device,
  the frozen Version-B checkout, the real M3B source package, and `modal` —
  every failing/erroring test file was grepped for `post_failure_diagnostics`
  and `c9_post_failure`; none matched. None of the 39 failures or 5 errors
  are in a file this task touched (`test_decision_contract.py`,
  `test_adapters_c3_modes.py`, `test_bootstrap_host_interpreter.py`,
  `test_checks_and_firewall.py`, `test_dependency_contract.py`,
  `test_gpu_hotfix_package.py`, `test_gpu_preflight_autograd.py`,
  `test_m2_m3a_contract.py`, `test_orchestrator.py`,
  `test_package_lifecycle.py`) — all environment-dependent, pre-existing,
  and unrelated to this correction.

## 12. What this correction does NOT do

Does not compute a real diagnostic value — `--preflight-only`, `--bind-only`,
`--status` and `--execute` were each run once against this real repo on this
laptop and each correctly reported an unresolved precondition (no CUDA, no
M3B package, no `C8_ACCEPTANCE.json`), exit 2, writing nothing. Does not
touch the BA_sep protocol, its artifacts, `DETECTOR_RELIABILITY_LOCK_C.json`,
`C8_ACCEPTANCE.json`, or V1's diagnostics namespace — verified by regression
and by direct byte-comparison tests. Does not access target data. Does not
resolve `cross_route_synthetic`, `recipe_region_shift` or `artifact_map_swap`
under an invented metric.
