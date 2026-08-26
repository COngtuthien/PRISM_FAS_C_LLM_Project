# C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2 — GPU handoff

**DO NOT RUN THE SCIENTIFIC COMMANDS BELOW ON THIS LAPTOP.** They are the
exact future commands for the GPU scientific host, which has CUDA, the real
M3B source package, `pyarrow`/`opencv`/`torch`, and the real `runs/full/c8/`
checkpoints this laptop does not have. This document does not execute
anything; it hands off exact steps.

This supersedes `C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V1_GPU_HANDOFF.md`
(V1 was never scientifically executed and remains unchanged, historical).
V2's frozen protocol identity is
`05ffa20ee71ff7168732436be6b8a98b613351d434f993226ff4308ed32a5523`
(`configs/evaluation/c9_post_failure_source_diagnostics_v2.yaml`). Three of
eight tests are `EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL`
(`benign_jpeg_corruption`, `benign_resize_corruption`,
`benign_color_corruption`); `cross_route_synthetic` is now
`NEEDS_SCIENTIFIC_DECISION` (Defect B correction) and the remaining four stay
`NEEDS_SCIENTIFIC_DECISION` / `STRUCTURALLY_MODEL_BLOCKED` /
`STRUCTURALLY_DATA_BLOCKED`, none run by `--execute`.

**The GPU host's git tree is NOT assumed clean.** It intentionally holds
modified/untracked scientific artifacts from the BA_sep execution and
`DETECTOR_RELIABILITY_LOCK_C` registration. Every command below is
non-destructive to that state.

---

## 1. Safe git update (never destructive)

```
git status --short
git fetch origin
git log --oneline -1 origin/portable-one-command-full-run
git merge --ff-only origin/portable-one-command-full-run
```

If `merge --ff-only` refuses because local commits exist that are not on
`origin`, STOP and report — do not force, rebase, or discard anything.
**NEVER** run on this branch: `git reset --hard`, `git clean`, a casual
`git stash` that is not immediately and intentionally popped back, `git add .`,
or `git add -A`. If local scientific artifacts are untracked and you need to
inspect what would be staged, use `git status --short` and add files
explicitly by path.

## 2. Protected-state checksum snapshot (BEFORE anything else)

```
sha256sum configs/evaluation/c9_detector_ba_sep_option1_v2.yaml
sha256sum configs/evaluation/c9_post_failure_source_diagnostics_v1.yaml
sha256sum configs/evaluation/c9_post_failure_source_diagnostics_v2.yaml
sha256sum reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json
sha256sum reports/full/c8/C8_ACCEPTANCE.json
sha256sum reports/full/c8/reliability/synthetic_vs_real_spoof_probe/BA_SEP_RESULT.json
sha256sum reports/full/c8/reliability/synthetic_vs_real_spoof_probe/BA_SEP_PER_SEED.json
sha256sum reports/full/c8/reliability/synthetic_vs_real_spoof_probe/BA_SEP_PROBE_PARAMETERS.json
sha256sum reports/full/c8/reliability/synthetic_vs_real_spoof_probe/BA_SEP_EVIDENCE_MANIFEST.json
sha256sum reports/full/c8/reliability/synthetic_vs_real_spoof_probe/SYNTHETIC_VS_REAL_SPOOF_PROBE_VERDICT.json
sha256sum reports/full/c8/reliability/synthetic_vs_real_spoof_probe/C9_BA_SEP_EXECUTION_BINDING.json
sha256sum reports/full/c8/reliability/synthetic_vs_real_spoof_probe/C9_BA_SEP_POPULATION_PLAN.json
find reports/full/c8/reliability/post_failure_source_diagnostics_v1 -type f -exec sha256sum {} \; 2>/dev/null || true
```

Save this output (e.g. `pre_v2_diagnostics_checksums.txt`) somewhere outside
the repo tree — it is what steps 8/9/10 below compare against. If V1's
namespace (`post_failure_source_diagnostics_v1/`) has artifacts on this host
from an earlier session, that command records them too — V2 must never
touch them.

## 3. Diagnostics `--preflight-only`

```
python -m prism_fas.evaluation.post_failure_diagnostics_v2_runner --repo . --preflight-only
```

Expect `ready_for_bind: true`; `per_test_gpu_ready` `true` for exactly
`benign_jpeg_corruption`, `benign_resize_corruption`, `benign_color_corruption`
and `false` for all five others (including `cross_route_synthetic`, now
`false` — this is the Defect B correction, not a regression);
`c8_matrix_identity_resolvable: true`; and
`ba_sep_canary.ba_sep_protocol_identity ==
720a2e344017d588d71005b81fdf0e7d2062081ae2f3881a61a306d952dc4ac8`. Exit code
`0`. If anything reports `false`, or `c8_matrix_identity_error` is non-empty,
or the canary identity differs, STOP — resolve the reported precondition
before continuing; do not force `--bind-only`.

## 4. Diagnostics `--bind-only`

```
python -m prism_fas.evaluation.post_failure_diagnostics_v2_runner --repo . --bind-only
```

Expect exit code `0`, `bound: true`, `reused: false` (first run on this
host), and the three binding artifacts written under
`reports/full/c8/reliability/post_failure_source_diagnostics_v2/` — a
namespace that must NOT already contain files from V1 (it is a distinct
directory: `..._v2/`, not `..._v1/`). Record `protocol_identity` — it must
equal `05ffa20ee71ff7168732436be6b8a98b613351d434f993226ff4308ed32a5523` —
and `c8_matrix_identity`, which must equal the `matrix_identity` field
already recorded in `reports/full/c8/C8_ACCEPTANCE.json` (the bind fails
closed with a clear error if these two disagree or if
`C8_ACCEPTANCE.json` is absent).

## 5. Binding verification

```
python -m prism_fas.evaluation.post_failure_diagnostics_v2_runner --repo . --bind-only
```

Run it a SECOND time. Expect exit code `0`, `bound: true`, `reused: true`,
`artifacts_written: false` — proof the bind is deterministic and idempotent.

```
sha256sum reports/full/c8/reliability/post_failure_source_diagnostics_v2/*.json
```

Confirm the three binding files' hashes are unchanged from step 4. Confirm
`reports/full/c8/reliability/post_failure_source_diagnostics_v1/` (if it has
artifacts from a prior V1 session on this host) is byte-identical to the
step-2 snapshot — V2's bind must not have touched it.

## 6. Scientific `--execute`, EXACTLY ONCE

```
python -m prism_fas.evaluation.post_failure_diagnostics_v2_runner --repo . --execute
```

This is the ONE authorized real V2 diagnostics execution. Exit code `0` if
every executed test PASSED, `1` if at least one FAILED (a real result, not
an error) — either is a valid, acceptable outcome; do not treat exit `1` as
a reason to retry. Record the full JSON output, including
`overall_diagnostics_verdict` and `per_test`. Note the per-test result now
reports `mean_delta_plus`/`p95_delta_plus` against `tau_mean`/`tau_tail`
rather than V1's `mean_shift`/`threshold` — this is the Defect A correction,
not a bug.

## 7. No-rerun verification, via the canonical validator

```
python -m prism_fas.evaluation.post_failure_diagnostics_v2_runner --repo . --execute
```

Run it a SECOND time, immediately. This call now runs
`post_failure_diagnostics_v2.validate_existing_diagnostics_result` before
re-reporting (Defect C correction) — expect the SAME exit code as step 6,
`reused_existing_diagnostics_result: true`, `checkpoint_weights_loaded: false`,
`images_forwarded: false`, `ba_metric_recomputed: false`. If the validator
finds ANY inconsistency (a tampered artifact, a mismatched binding, a
recomputed-per-test/overall verdict mismatch, a live
`DETECTOR_RELIABILITY_LOCK_C` that is no longer `FAILED`, etc.) it BLOCKS
with exit code `2` and `error: "EXISTING_RESULT_FAILED_VALIDATION"` — this is
correct fail-closed behavior, not a bug; report the `problems` list rather
than deleting anything.

```
sha256sum reports/full/c8/reliability/post_failure_source_diagnostics_v2/DIAGNOSTICS_RESULT.json
sha256sum reports/full/c8/reliability/post_failure_source_diagnostics_v2/DIAGNOSTICS_VERDICT.json
```

Confirm both hashes are identical to what step 6 produced.

## 8. Result verification

```
python -m prism_fas.evaluation.post_failure_diagnostics_v2_runner --repo . --status
```

`--status` also runs the canonical validator now. Expect
`diagnostics_result_available: true`, `overall_diagnostics_verdict` matching
step 6, `c9_may_close: false`, `existing_result_validation.valid: true`, and
`ba_sep_canary` unchanged from step 3.

```
python -c "import json; d = json.load(open('reports/full/c8/reliability/post_failure_source_diagnostics_v2/DIAGNOSTICS_VERDICT.json')); print(d['ba_sep_observed_verdict'], d['detector_reliability_lock_c_observed_overall'], d['c9_may_close'])"
```

Expect exactly: `FAIL FAILED False`.

## 9. Proof the BA_sep artifacts are unchanged

```
sha256sum reports/full/c8/reliability/synthetic_vs_real_spoof_probe/*.json
```

Diff against the step-2 snapshot. Every hash must be byte-identical.

## 10. Proof `DETECTOR_RELIABILITY_LOCK_C.json` and `C8_ACCEPTANCE.json` are unchanged

```
sha256sum reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json reports/full/c8/C8_ACCEPTANCE.json
python -c "import json; d = json.load(open('reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json')); print(d['overall'], d['c9_may_close'])"
```

Both hashes must match the step-2 snapshot exactly. The printed values must
be `FAILED False`.

## 11. Proof V1's namespace is untouched

```
find reports/full/c8/reliability/post_failure_source_diagnostics_v1 -type f -exec sha256sum {} \; 2>/dev/null || echo "no V1 artifacts on this host"
```

Must be byte-identical to the step-2 snapshot (or still absent, if it was
absent then).

## 12. `target_access = 0`

```
grep -r '"target_access"' reports/full/c8/reliability/post_failure_source_diagnostics_v2/
```

Every artifact must report `"target_access": 0`. No command in this handoff
opens, reads, or resolves any `siw_mv2`, `target_test`,
`prism_target_eval_v2`, or target-label path — confirm none appear in any
output above.

## 13. After this — what is NOT authorized yet

Nothing in this handoff authorizes: rerunning `--execute` a third time with
different inputs, editing the BA_sep protocol or its result files, editing
`DETECTOR_RELIABILITY_LOCK_C.json` or `C8_ACCEPTANCE.json`, running C10, C11,
C12 or C13, opening any target artifact, or resolving
`cross_route_synthetic`, `recipe_region_shift` or `artifact_map_swap` under
an invented metric. The observed diagnostics results (once produced) are
themselves then permanent evidence — feeding into the interpretation and
next-decision work of a SEPARATE, future task, not this one.
