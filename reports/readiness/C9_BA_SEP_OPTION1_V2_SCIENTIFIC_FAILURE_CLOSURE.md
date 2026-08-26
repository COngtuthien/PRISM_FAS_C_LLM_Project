# C9_DETECTOR_BA_SEP_OPTION1_V2 — scientific FAILURE closure

**THE FIRST AND ONLY AUTHORIZED REAL BA_sep OPTION-1 V2 EXECUTION HAS
COMPLETED, ON THE GPU SCIENTIFIC HOST.** This is an OBSERVED scientific
result. **It is not rerun. Nothing about it is retried, tuned, or
selectively reread to obtain a different outcome.**

This document is written on the development laptop, which does not possess
the GPU host's real artifacts. Every numeric value below in §1–§2 is
**OBSERVED_ON_GPU_USER_REPORTED** — reported to this laptop, in this task, by
the person directing the work — not independently verified by this laptop
reading real files it does not have. Verifying these numbers against the
real artifacts on disk is exactly what
`python -m prism_fas.evaluation.detector_reliability_runner --register-ba-sep-result`
does, and that verification is **GPU_ARTIFACT_REGISTRATION_VERIFIED** only
once it is actually run, on the GPU host, against the real files. It has not
been run yet.

---

## 1. The observed result

Execution exit code: `1` — **scientific execution COMPLETED**, hard verdict
**FAIL**. Not a technical `BLOCKED`, not a crash, not an incomplete run.

Aggregate `BA_sep` by arm:

```
RND = 0.7843079833902619
DET = 0.8514170182841069
LLM = 0.7902658339472685
```

Frozen hard ceiling: `0.75`. Frozen rule (unchanged, from
`configs/evaluation/c9_detector_ba_sep_option1_v2.yaml::hard_verdict_rule`):

```
PASS iff BA_sep_RND <= 0.75 AND BA_sep_DET <= 0.75 AND BA_sep_LLM <= 0.75
```

All three arms exceed the ceiling. Verdict: **`synthetic_vs_real_spoof_probe = FAILED`**.

Per-probe-seed values:

| arm | 20260806 | 20260807 | 20260808 |
|-----|----------|----------|----------|
| RND | 0.753968253968254 | 0.8164556962025317 | 0.7825 |
| DET | 0.8333333333333333 | 0.8734177215189873 | 0.8474999999999999 |
| LLM | 0.7486772486772486 | 0.819620253164557 | 0.8025 |

The arithmetic mean of each arm's three per-seed values reproduces its
aggregate exactly (verified on this laptop with plain arithmetic on the
reported numbers — `sum(values)/3` for each arm equals the aggregate above
to full float64 precision). The fact that RND's `20260806` seed
(`0.753968...`) and LLM's `20260806` seed (`0.748677...`) individually sit
under 0.75 is **irrelevant to the frozen aggregate rule**, which operates on
the three-seed MEAN per arm, never on any single seed. No seed is selected,
removed, replaced, or rerun on the strength of an individual value.

## 2. Frozen identities from the real GPU run (user-reported)

```
protocol identity:            720a2e344017d588d71005b81fdf0e7d2062081ae2f3881a61a306d952dc4ac8
checkpoint binding identity:  fa380fa8e732f8536fe175d449542e636563d92d8d75f64bb07b40ca180f63b0
population plan identity:     90d00d9f4bb50a93724d1ac6a632d6fa5052cf2d7ec0d08989c4c7004fa6cae1
source package identity:      08d9d289eb4b462006afcff37cd4750a7c4eeb402c83de5599eda38df44168c9
execution code commit:        7d0d3267d4e9208744b55cd6d16793780b297930

c6_bank_identities:
  RND: 1a5be34ad5104b9c8fe18f90778e86b6710180e757ea066c4f931565928d50ba
  DET: 32f6e0e129da277d0be76abc0758bc3d63ad16caba111efc801ac92de291f5b0
  LLM: b7edd00b7ea87e558814b96955452f3bf4bdbeda0da4374461b4ed4ddd29525b

target_access: 0
```

The V2 protocol identity is exactly the one this project already froze and
has verified unchanged across two prior implementation-fix tasks
(`C9_BA_SEP_OPTION1_V2_RUNNER_INTEGRATION_FIX.md`,
`C9_BA_SEP_OPTION1_V2_INFERENCE_PARITY_FIX.md`) — the protocol that PRODUCED
this result is the SAME protocol that was frozen before any BA_sep value was
ever observed, at any point.

## 3. What is FORBIDDEN now that a result exists

Per `detector_reliability.barrier_state`'s own `on_failure` clause (unchanged
by this task) and the frozen protocol's own `hard_verdict_rule.post_observation_prohibitions`:

- **No rerun.** `synthetic_real_probe_runner --execute` is now
  scientifically NO-RERUN: with all five result artifacts present, it
  re-reports the existing result and performs zero trainer construction,
  zero checkpoint load, zero forward pass, zero probe fit, zero BA
  recomputation (`reports/readiness/C9_BA_SEP_OPTION1_V2_SCIENTIFIC_FAILURE_CLOSURE.json::no_rerun_guard`).
- **No threshold tuning.** The 0.75 ceiling is not loosened, narrowed, or
  made per-arm.
- **No seed replacement.** The three probe seeds
  (`20260806, 20260807, 20260808`) are not swapped, dropped, or
  supplemented — regardless of any individual seed's value.
- **No population replacement.** The frozen population plan
  (`90d00d9...4fa6cae1`) is not rebuilt with different sampling, balancing,
  or split rules to seek a different outcome.
- **No checkpoint cherry-picking.** All 15 bound checkpoints (5/arm)
  contributed to the averaged evidence; none may be dropped, substituted,
  or reselected by any post-hoc criterion.
- **No C6 bank reopening under the current protocol.** The three bank
  identities above (`1a5be34a...`, `32f6e0e1...`, `b7edd00b...`) are the
  banks this result was produced against; they are not regenerated, retuned
  or reselected under Option-1 V2.
- **The negative result is retained, not discarded.** It becomes the
  permanent, immutable scientific record for `synthetic_vs_real_spoof_probe`
  under Option-1 V2. Any future redesign is a NEW, separately versioned
  protocol (Option C in
  `reports/readiness/C9_DETECTOR_RELIABILITY_POST_BA_FAILURE_DECISION_DOSSIER.md`)
  — never a retroactive edit of this one.

## 4. The no-rerun guard (implementation, this task)

`prism_fas.evaluation.synthetic_real_probe_runner._execute` now checks, BEFORE
any protocol read, any binding read, or any trainer consideration, how many
of the five result artifacts already exist:

- **none** — the original first-execution path is reachable (a clean host
  may still run the probe once);
- **all five** — read and cross-validate them
  (`synthetic_real_probe.validate_existing_scientific_result`) and
  RE-REPORT the existing result; the exit code still reflects the real
  recorded verdict (`PASS` → 0, `FAIL` → 1);
- **some but not all** — `BLOCKED` with reason `PARTIAL_SCIENTIFIC_RESULT_SET`;
  nothing is overwritten, completed, or inferred.

`validate_existing_scientific_result` cross-checks all seven artifacts
(the two `--bind-only` artifacts plus the five result artifacts) for
identity agreement (`protocol_identity`, `checkpoint_binding_identity`,
`population_plan_identity`, `source_package_identity`, all three C6 bank
identities, `target_access == 0`), requires exactly 5/5/5 checkpoint hashes
(15 total) matching the execution binding, requires exactly the three
frozen probe seeds per arm in the per-seed record, RECOMPUTES (never
re-fits) each arm's aggregate from its own recorded per-seed values via the
unchanged `aggregate_ba_sep`, and RECOMPUTES the hard verdict from the
recorded `ba_sep_by_arm` via the unchanged `hard_verdict` — requiring both
to equal what is recorded. This is bookkeeping over JSON already on disk;
it never constructs a trainer, loads a checkpoint's weights, opens an
image, or fits a probe.

## 5. Failed barrier registration (this task, pending GPU verification)

`python -m prism_fas.evaluation.detector_reliability_runner --register-ba-sep-result`
is the sanctioned path to bind this OBSERVED result into
`reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json`, via the existing,
unchanged `detector_reliability.lock_payload`. It requires the complete,
valid result set (§4 above) and binds ONLY
`synthetic_vs_real_spoof_probe = FAILED`; every other
`REQUIRED_DETECTOR_RELIABILITY_TESTS` entry stays `UNRESOLVED` — this
module never infers or fabricates another test's outcome. Idempotent:
identical existing scientific content is verified and reused; different
existing content blocks rather than being overwritten. It never mutates any
of the seven BA_sep artifacts it reads.

This registration has **NOT** been run — not on this laptop (which has none
of the real artifacts: `--status` and `--register-ba-sep-result` were both
run against this real repo as part of this task and both correctly reported
the result set absent, exit code 2, writing nothing) and not yet on the GPU
host either. Until it runs there, `DETECTOR_RELIABILITY_LOCK_C.json` does
not yet exist as a real, registered, GPU-verified artifact — the barrier
state below is the state registration WOULD produce, computed here from the
fixture-proven mechanics and the reported numbers, not yet
GPU-artifact-verified.

## 6. The barrier state this observed result produces

```
synthetic_vs_real_spoof_probe = FAILED
residual_scale_zero           = UNRESOLVED
recipe_region_shift           = UNRESOLVED
artifact_map_swap             = UNRESOLVED
cross_route_synthetic         = UNRESOLVED
benign_jpeg_corruption        = UNRESOLVED
benign_resize_corruption      = UNRESOLVED
benign_color_corruption       = UNRESOLVED
crop_padding_interpolation    = UNRESOLVED
benign_glasses_makeup_lowlight = BLOCKED   (canonical, unrequired)

overall      = FAILED
c9_may_close = false
target_access = 0
```

`detector_reliability.barrier_state`'s existing, unmodified rule (`if
failed: overall = FAILED`) means this is correct even though eight required
tests remain unresolved — a single required FAILED test holds the barrier
shut regardless of how many others are still open. Resolving all eight
later to PASSED cannot change `overall` back to `PASSED`: FAILED is sticky
once any required test has genuinely failed, under this protocol.

## 7. `detector_reliability.verify_lock` remains strict, PASS-only

`verify_lock` was NOT weakened. It still requires `overall == PASSED`. A
registered FAILED barrier makes `verify_lock(...).valid == False` — exactly
as an absent lock does. A new, separate function,
`detector_reliability.validate_lock_record`, answers the different question
"is this a structurally well-formed record" (true for a genuine FAILED
record) without ever making `verify_lock` accept it. C9's own precondition
gate (`pipeline.adapters.c9.C9Adapter.semantic_preconditions`) calls
`verify_lock` — unmodified — so C9 rejects a FAILED reliability record
exactly as it rejects an absent one, blocking `workflow()` and therefore
every target-adjacent path C9 could ever reach.

## 8. C-H4

Under the observed values:

```
BA_sep_LLM = 0.7902658339472685
BA_sep_DET = 0.8514170182841069
BA_sep_RND = 0.7843079833902619
```

`C-H4` requires `BA_sep_LLM <= 0.75 AND BA_sep_LLM < BA_sep_DET AND
BA_sep_LLM < BA_sep_RND` (plus a bootstrap-CI condition, a validity
condition and a recipe-diversity condition, none of which is ever
fabricated here). LLM is over the 0.75 ceiling
(`0.7902... > 0.75` — hard gate fails) and LLM does NOT beat RND
(`0.7902... > 0.7843...`), even though LLM does beat DET
(`0.7902... < 0.8514...`). The basic hard-gate preconditions already fail
before any bootstrap CI is considered:

```
C-H4 = NOT_SUPPORTED_BY_CURRENT_BA_SEP_RESULT
```

`synthetic_real_probe.c_h4_preconditions` computes exactly this, from the
observed values, and never computes, fits, or fabricates a bootstrap CI.

## 9. Scientific interpretation — what this DOES and does NOT show

The result supports: under the frozen Option-1 V2 evidence representation
(`[global_logit_G, p_global]`, common Track-G decision evidence), a simple
linear probe can separate synthetic spoof from real spoof above the frozen
0.75 reliability ceiling, for all three generator arms.

It does NOT show: that the entire synthetic generation method is useless —
BA_sep measures SEPARABILITY of the evidence representation, not detector
performance, and no target metric was ever computed. It does NOT show that
target performance is bad — target has not been accessed;
`target_access = 0` throughout every artifact this result touches. It does
NOT show that LLM is universally worse than deterministic generation: LLM's
BA_sep is LOWER than DET's (`0.7902... < 0.8514...`) even though it is
still over ceiling and still not below RND's. Every one of these
distinctions is stated precisely, once, here, and is not re-litigated
elsewhere in this closure record.

## 10. What this closure does NOT do

- Does not rerun the probe, on any arm, under any circumstance.
- Does not tune the 0.75 ceiling, replace a seed, rebuild the population, or
  cherry-pick a checkpoint.
- Does not reopen the C6 banks under the current (Option-1 V2) protocol.
- Does not register the barrier on this laptop — no real artifacts exist
  here to register.
- Does not resolve any of the other eight `REQUIRED_DETECTOR_RELIABILITY_TESTS`.
- Does not run C9, C10, C11, C12 or C13.
- Does not access target data, anywhere, under any code path this task adds.
- Does not create a V3 protocol or edit the V2 config.

**C9 remains, and after this closure record remains, correctly
`BLOCKED_BY_DETECTOR_RELIABILITY_FAILURE`** — a stronger, more specific
condition than the prior `BLOCKED_PENDING_DETECTOR_RELIABILITY_SCIENTIFIC_DECISION`,
because the barrier now carries a genuine, observed, negative scientific
result rather than an open question.
