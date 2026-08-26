# C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2 — observed-result interpretation (preview)

**THIS IS A PREVIEW, NOT THE REGISTERED ARTIFACT.** The real
`DIAGNOSTICS_INTERPRETATION.json`/`.md` are written by
`prism_fas.evaluation.post_failure_diagnostics_v2_closure --register-interpretation`,
run on the GPU host against the real, already-executed, already-validated
result files. This laptop has not independently read those files and has
not computed real `result_file_sha256`, `checkpoint_binding_identity`,
`population_binding_identity`, or `source_package_identity` values for this
run. Every number below is either (a) the user-reported GPU scientific
observation from this task's prompt, or (b) arithmetic derived from those
numbers by the same pure function (`post_failure_diagnostics_v2_interpretation.derive_full_interpretation`)
the real registration calls — proven byte-identical by
`tests/pipeline/test_c9_post_failure_source_diagnostics_v2_closure.py`.

`C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2` (protocol identity
`05ffa20ee71ff7168732436be6b8a98b613351d434f993226ff4308ed32a5523`) has now
been scientifically executed exactly once, at code commit
`c36a0214358fa21c905f9e746796611968c03221`, C8 matrix identity
`a777671fb9142a75369a905f66eee5f0f2ab5c3827f33d3803d52426e2e29af8` (all
user-reported). `overall_diagnostics_verdict = FAIL`. `target_access = 0`.
`c9_may_close = false`. Both `--execute` (first and second call) and
`--status` returned exit `1`, consistent with a genuine observed
`overall_diagnostics_verdict = FAIL` plus `existing_result_validation.valid
= true` — this is a correctly-behaving negative scientific result, not an
error.

---

## A. `benign_color_corruption` — TEST VERDICT: PASS

| arm | mean_delta_plus | tau_mean | p95_delta_plus | tau_tail | verdict |
|---|---|---|---|---|---|
| DET | 0.056465106 | 0.325571173 | 0.257891305 | 0.505300498 | PASS |
| LLM | 0.060303564 | 0.377137697 | 0.278645410 | 0.594541305 | PASS |
| RND | 0.074588124 | 0.326370837 | 0.325204827 | 0.505645819 | PASS |

**INTERPRETATION:** No evidence, under this protocol, of excessive
sensitivity to the frozen benign color-gain perturbation for any arm (RND,
DET, LLM).

**NOT_SUPPORTED:** generalizing this PASS to all color/illumination
conditions outside this protocol's exact frozen ±15%/±10% RGB-gain
perturbation and `source_dev` LIVE population.

## B. `benign_jpeg_corruption` — TEST VERDICT: FAIL (RND only)

| arm | mean_delta_plus | tau_mean | mean_ok | p95_delta_plus | tau_tail | tail_ok | verdict |
|---|---|---|---|---|---|---|---|
| DET | 0.107911873 | 0.325571173 | yes | 0.433825017 | 0.505300498 | yes | PASS |
| LLM | 0.114902634 | 0.377137697 | yes | 0.500528473 | 0.594541305 | yes | PASS |
| RND | 0.151063727 | 0.326370837 | yes | **0.548018453** | **0.505645819** | **no** | FAIL |

**OBSERVED:** RND: `mean_delta_plus = 0.1510637 <= tau_mean = 0.3263708`
(mean criterion satisfied); `p95_delta_plus = 0.5480185 > tau_tail =
0.5056458` (tail criterion NOT satisfied).

**DERIVED_ARITHMETIC:** `tail_exceedance = 0.5480185 - 0.5056458 =
0.0423726`.

**INTERPRETATION:** RND does not show excessive average JPEG-induced
spoof-score increase, but a high-sensitivity tail of `source_dev` LIVE
samples exceeds the frozen benign-reference tolerance. DET and LLM show no
such tail effect under this protocol.

**NOT_SUPPORTED:** any claim that this failure reflects a broad,
average-case JPEG sensitivity in RND (the mean criterion PASSES).

## C. `benign_resize_corruption` — TEST VERDICT: FAIL (all three arms)

| arm | mean_delta_plus | tau_mean | mean_ok | p95_delta_plus | tau_tail | tail_ok | verdict |
|---|---|---|---|---|---|---|---|
| DET | 0.189705961 | 0.325571173 | yes | 0.543240921 | 0.505300498 | no | FAIL |
| LLM | 0.191888028 | 0.377137697 | yes | 0.617576063 | 0.594541305 | no | FAIL |
| RND | 0.209111999 | 0.326370837 | yes | 0.606440059 | 0.505645819 | no | FAIL |

**OBSERVED:** all three arms' mean criterion PASSES; all three arms' tail
criterion FAILS.

**DERIVED_ARITHMETIC — exact tail exceedances:**

- DET: `0.543240921 − 0.505300498 = 0.037940423`
- LLM: `0.617576063 − 0.594541305 = 0.023034758`
- RND: `0.606440059 − 0.505645819 = 0.100794240`

**INTERPRETATION:** the failure is not a broad average score inflation; it
is primarily a tail-sensitivity / subset-of-samples effect. Ordered most to
least frozen-threshold exceedance: **RND (0.1007942) > DET (0.0379404) >
LLM (0.0230348)**. RND has the largest exceedance, LLM the smallest, DET
intermediate. This is a descriptive comparison only.

**NOT_SUPPORTED:** a statistical ranking among RND/DET/LLM tail exceedance
— no preregistered paired statistical comparison exists for this quantity.

## D. Connection to the BA_sep FAILURE

**INTERPRETATION (consistency, not causal):** the resize/JPEG diagnostic
results are consistent with the possibility that some detectors use
preprocessing/resampling/compression-sensitive cues, which is
mechanistically compatible with the shortcut sensitivity already implicated
by the BA_sep FAILURE.

**NOT_SUPPORTED:** "resize sensitivity caused BA_sep to fail" — causality
has not been established by this protocol.

## E. Blocked tests remain BLOCKED

`artifact_map_swap`, `recipe_region_shift`, `cross_route_synthetic` remain
`NEEDS_SCIENTIFIC_DECISION`; `residual_scale_zero` remains
`STRUCTURALLY_MODEL_BLOCKED`; `crop_padding_interpolation` remains
`STRUCTURALLY_DATA_BLOCKED`. **None of these is treated as an observed FAIL
or as negative evidence — BLOCKED != FAIL.**

## F. Immutable scientific state (unchanged by this diagnostic result)

| field | value |
|---|---|
| BA_sep (Option-1 V2) | RND=0.7843079833902619, DET=0.8514170182841069, LLM=0.7902658339472685 — **FAIL, immutable, rerun FORBIDDEN** |
| BA_sep protocol identity | `720a2e344017d588d71005b81fdf0e7d2062081ae2f3881a61a306d952dc4ac8` |
| `DETECTOR_RELIABILITY_LOCK_C` | `overall = FAILED`, `c9_may_close = false` |
| Original C9 confirmatory path | BLOCKED |
| target_access | 0 |

## G. What must happen for real, on the GPU host

```
python -m prism_fas.evaluation.post_failure_diagnostics_v2_closure --repo . --status
python -m prism_fas.evaluation.post_failure_diagnostics_v2_closure --repo . --register-interpretation
```

See `C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2_INTERPRETATION_GPU_HANDOFF.md` for
the exact, safe, non-destructive sequence and the required before/after
checksum proofs.
