# C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V1 — GPU handoff

**DO NOT RUN THE SCIENTIFIC COMMANDS BELOW ON THIS LAPTOP.** They are the
exact future commands for the GPU scientific host, which has CUDA, the real
M3B source package, and the real `runs/full/c8/` checkpoints this laptop
does not have. This document does not execute anything; it hands off exact
steps.

The frozen diagnostics protocol identity is
`cb05271e26d9a421f2f9277599523e185026e1eab644febc07c75432d26f3fc5`
(`configs/evaluation/c9_post_failure_source_diagnostics_v1.yaml`). Four of
eight tests are `EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL`
(`benign_jpeg_corruption`, `benign_resize_corruption`,
`benign_color_corruption`, `cross_route_synthetic`); the other four remain
`NEEDS_SCIENTIFIC_DECISION` / `STRUCTURALLY_MODEL_BLOCKED` /
`STRUCTURALLY_DATA_BLOCKED` and are not run by `--execute`.

---

## 1. Safe git fast-forward

```
git fetch origin
git status --short
git rev-parse HEAD
git log --oneline -1
```

Expect a clean tree and `HEAD` matching the commit this task pushed. If the
GPU host's checkout is behind, fast-forward only — never rebase, never
force:

```
git checkout portable-one-command-full-run
git pull --ff-only origin portable-one-command-full-run
```

## 2. Protected-state checksum snapshot (BEFORE anything else)

Record hashes of everything this protocol must never move, so any drift is
immediately provable:

```
sha256sum configs/evaluation/c9_detector_ba_sep_option1_v2.yaml
sha256sum configs/evaluation/c9_post_failure_source_diagnostics_v1.yaml
sha256sum reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json
sha256sum reports/full/c8/reliability/synthetic_vs_real_spoof_probe/BA_SEP_RESULT.json
sha256sum reports/full/c8/reliability/synthetic_vs_real_spoof_probe/BA_SEP_PER_SEED.json
sha256sum reports/full/c8/reliability/synthetic_vs_real_spoof_probe/BA_SEP_PROBE_PARAMETERS.json
sha256sum reports/full/c8/reliability/synthetic_vs_real_spoof_probe/BA_SEP_EVIDENCE_MANIFEST.json
sha256sum reports/full/c8/reliability/synthetic_vs_real_spoof_probe/SYNTHETIC_VS_REAL_SPOOF_PROBE_VERDICT.json
sha256sum reports/full/c8/reliability/synthetic_vs_real_spoof_probe/C9_BA_SEP_EXECUTION_BINDING.json
sha256sum reports/full/c8/reliability/synthetic_vs_real_spoof_probe/C9_BA_SEP_POPULATION_PLAN.json
```

Save this output (e.g. `pre_diagnostics_checksums.txt`) somewhere outside
the repo tree — it is what step 9/10 below compares against.

## 3. Diagnostics `--preflight-only`

```
python -m prism_fas.evaluation.post_failure_diagnostics_runner --repo . --preflight-only
```

Expect `ready_for_bind: true`, `per_test_gpu_ready` matching the four
`EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL` tests as `true` and the other four as
`false`, and `ba_sep_canary.ba_sep_protocol_identity ==
720a2e344017d588d71005b81fdf0e7d2062081ae2f3881a61a306d952dc4ac8`. Exit code
`0`. If anything reports `false` or the canary identity differs, STOP —
resolve the reported precondition before continuing; do not force `--bind-only`.

## 4. Diagnostics `--bind-only`

```
python -m prism_fas.evaluation.post_failure_diagnostics_runner --repo . --bind-only
```

Expect exit code `0`, `bound: true`, `reused: false` (first run), and the
three binding artifacts written under
`reports/full/c8/reliability/post_failure_source_diagnostics_v1/`. Record
`protocol_identity` from the output — it must equal
`cb05271e26d9a421f2f9277599523e185026e1eab644febc07c75432d26f3fc5`.

## 5. Binding verification

```
python -m prism_fas.evaluation.post_failure_diagnostics_runner --repo . --bind-only
```

Run it a SECOND time. Expect exit code `0`, `bound: true`, `reused: true`,
`artifacts_written: false` — proof the bind is deterministic and idempotent,
not a fresh (possibly different) resolution each time.

```
sha256sum reports/full/c8/reliability/post_failure_source_diagnostics_v1/*.json
```

Confirm the three binding files' hashes are unchanged from step 4.

## 6. Scientific `--execute`, EXACTLY ONCE

```
python -m prism_fas.evaluation.post_failure_diagnostics_runner --repo . --execute
```

This is the ONE authorized real diagnostics execution. Exit code `0` if
every executed test PASSED, `1` if at least one FAILED (a real result, not
an error) — either is a valid, acceptable outcome; do not treat exit `1` as
a reason to retry. Record the full JSON output, including
`overall_diagnostics_verdict` and `per_test`.

## 7. No-rerun verification

```
python -m prism_fas.evaluation.post_failure_diagnostics_runner --repo . --execute
```

Run it a SECOND time, immediately. Expect the SAME exit code as step 6,
`reused_existing_diagnostics_result: true`, `checkpoint_weights_loaded: false`,
`images_forwarded: false`, `ba_metric_recomputed: false` — proof the second
call re-reported rather than recomputing anything.

```
sha256sum reports/full/c8/reliability/post_failure_source_diagnostics_v1/DIAGNOSTICS_RESULT.json
sha256sum reports/full/c8/reliability/post_failure_source_diagnostics_v1/DIAGNOSTICS_VERDICT.json
```

Confirm both hashes are identical to what step 6 produced.

## 8. Result verification

```
python -m prism_fas.evaluation.post_failure_diagnostics_runner --repo . --status
```

Expect `diagnostics_result_available: true`, `overall_diagnostics_verdict`
matching step 6, `c9_may_close: false`, and `ba_sep_canary` unchanged from
step 3.

```
python -c "import json; d = json.load(open('reports/full/c8/reliability/post_failure_source_diagnostics_v1/DIAGNOSTICS_VERDICT.json')); print(d['ba_sep_observed_verdict'], d['detector_reliability_lock_c_observed_overall'], d['c9_may_close'])"
```

Expect exactly: `FAIL FAILED False`.

## 9. Proof the BA_sep artifacts are unchanged

```
sha256sum reports/full/c8/reliability/synthetic_vs_real_spoof_probe/*.json
```

Diff this output against the step-2 snapshot. Every hash must be
byte-identical — the diagnostics execution must not have touched a single
byte of the BA_sep result set.

## 10. Proof `DETECTOR_RELIABILITY_LOCK_C.json` is unchanged

```
sha256sum reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json
python -c "import json; d = json.load(open('reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json')); print(d['overall'], d['c9_may_close'])"
```

The hash must match the step-2 snapshot exactly. The printed values must be
`FAILED False` — unchanged by anything this diagnostics run computed,
regardless of the diagnostics verdict.

## 11. `target_access = 0`

```
grep -r '"target_access"' reports/full/c8/reliability/post_failure_source_diagnostics_v1/
```

Every artifact must report `"target_access": 0`. No command in this handoff
opens, reads, or resolves any `siw_mv2`, `target_test`,
`prism_target_eval_v2`, or target-label path — confirm none appear in any
output above.

## 12. After this — what is NOT authorized yet

Nothing in this handoff authorizes: rerunning `--execute` a third time with
different inputs, editing the BA_sep protocol or its result files, editing
`DETECTOR_RELIABILITY_LOCK_C.json`, running C10, C11, C12 or C13, or opening
any target artifact. The observed diagnostics results (once produced) are
themselves then permanent evidence — feeding into the interpretation and
next-decision work of a SEPARATE, future task, not this one.
