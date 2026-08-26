# C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2 — interpretation registration GPU handoff

**DO NOT RUN A DIAGNOSTIC AGAIN.** `C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2`
has already been scientifically executed exactly once on this host
(`overall_diagnostics_verdict = FAIL`, user-reported: color PASS all arms,
JPEG FAIL via RND's upper-tail only, resize FAIL via all three arms'
upper-tail only). Rerunning `post_failure_diagnostics_v2_runner --execute`
is FORBIDDEN — it is already scientifically no-rerun by construction
(`validate_existing_diagnostics_result` blocks recomputation), but this
handoff exists so the closure/interpretation step is never confused with a
re-execution.

This phase does exactly one new thing: it validates the existing result and
registers a bounded, arithmetic-only interpretation of it. It does **not**
open a target path, retrain anything, or touch `DETECTOR_RELIABILITY_LOCK_C`,
`C8_ACCEPTANCE.json`, or the BA_sep artifacts.

**The GPU host's git tree is NOT assumed clean** — same as the prior
handoff. Every command below is non-destructive.

---

## 1. Safe git update (never destructive)

```
git status --short
git fetch origin
git log --oneline -1 origin/portable-one-command-full-run
git merge --ff-only origin/portable-one-command-full-run
```

STOP and report if `merge --ff-only` refuses. **NEVER** run `git reset
--hard`, `git clean`, a casual `git stash` left unpopped, `git add .`, or
`git add -A`.

## 2. Protected-state checksum snapshot (BEFORE anything else)

```
sha256sum configs/evaluation/c9_detector_ba_sep_option1_v2.yaml
sha256sum configs/evaluation/c9_post_failure_source_diagnostics_v1.yaml
sha256sum configs/evaluation/c9_post_failure_source_diagnostics_v2.yaml
sha256sum reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json
sha256sum reports/full/c8/C8_ACCEPTANCE.json
sha256sum reports/full/c8/reliability/synthetic_vs_real_spoof_probe/*.json
find reports/full/c8/reliability/post_failure_source_diagnostics_v1 -type f -exec sha256sum {} \; 2>/dev/null || true
find reports/full/c8/reliability/post_failure_source_diagnostics_v2 -type f -exec sha256sum {} \;
```

Save this output (e.g. `pre_interpretation_checksums.txt`) outside the repo
tree. The V2 diagnostics directory's four result artifacts and three
bindings MUST already exist and be non-empty (the scientific execution
already happened) — if any is missing, STOP; this handoff is not the
recovery path for a missing execution.

## 3. Closure `--status` (read-only)

```
python -m prism_fas.evaluation.post_failure_diagnostics_v2_closure --repo . --status
```

Expect `diagnostics_result_valid: true`, `diagnostics_result_problems: []`,
`interpretation_registered: false` (first run), `checkpoint_weights_loaded:
false`, `images_forwarded: false`, `diagnostic_metric_recomputed: false`,
`target_access: 0`. Exit code `2` is CORRECT here (`reason:
INTERPRETATION_NOT_YET_REGISTERED`) — it does not mean anything is broken.
If `diagnostics_result_valid` is `false`, STOP and report the
`diagnostics_result_problems` list; do not proceed to registration and do
not attempt to fix it by rerunning `--execute`.

## 4. Register the interpretation, EXACTLY ONCE

```
python -m prism_fas.evaluation.post_failure_diagnostics_v2_closure --repo . --register-interpretation
```

This re-validates the existing result (same check as step 3), then derives
the bounded interpretation purely from the recorded `DIAGNOSTICS_PER_TEST.json`
values, hashes the four already-written result files, and writes:

```
reports/full/c8/reliability/post_failure_source_diagnostics_v2/DIAGNOSTICS_INTERPRETATION.json
reports/full/c8/reliability/post_failure_source_diagnostics_v2/DIAGNOSTICS_INTERPRETATION.md
```

Expect exit code `0`, `registered: true`, `reused: false`, `written: true`,
`checkpoint_weights_loaded: false`, `images_forwarded: false`,
`diagnostic_metric_recomputed: false`. Compare the written JSON's
`interpretation` section against
`reports/readiness/C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2_OBSERVED_INTERPRETATION_PREVIEW.json`'s
`interpretation_preview` — the per-test `interpretation`/`not_supported`
text must match exactly (same pure function, same recorded numbers).

## 5. No-rerun / idempotence verification

```
python -m prism_fas.evaluation.post_failure_diagnostics_v2_closure --repo . --register-interpretation
```

Run it a SECOND time, immediately. Expect exit code `0`, `registered: true`,
`reused: true`, `written: false`.

```
sha256sum reports/full/c8/reliability/post_failure_source_diagnostics_v2/DIAGNOSTICS_INTERPRETATION.json
```

Confirm the hash is identical to what step 4 produced.

```
python -m prism_fas.evaluation.post_failure_diagnostics_v2_closure --repo . --status
```

Expect exit code `0`, `interpretation_registered: true`.

## 6. Proof the four original result artifacts are unchanged

```
sha256sum reports/full/c8/reliability/post_failure_source_diagnostics_v2/DIAGNOSTICS_RESULT.json
sha256sum reports/full/c8/reliability/post_failure_source_diagnostics_v2/DIAGNOSTICS_PER_TEST.json
sha256sum reports/full/c8/reliability/post_failure_source_diagnostics_v2/DIAGNOSTICS_PROVENANCE.json
sha256sum reports/full/c8/reliability/post_failure_source_diagnostics_v2/DIAGNOSTICS_VERDICT.json
```

Byte-identical to the step-2 snapshot — closure registration must never
rewrite the original result files (it only reads and hashes them).

## 7. Proof BA_sep / `DETECTOR_RELIABILITY_LOCK_C` / `C8_ACCEPTANCE.json` are unchanged

```
sha256sum reports/full/c8/reliability/synthetic_vs_real_spoof_probe/*.json
sha256sum reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json reports/full/c8/C8_ACCEPTANCE.json
python -c "import json; d = json.load(open('reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json')); print(d['overall'], d['c9_may_close'])"
```

All hashes byte-identical to the step-2 snapshot. Printed values must be
`FAILED False`.

## 8. Proof V1's namespace is untouched

```
find reports/full/c8/reliability/post_failure_source_diagnostics_v1 -type f -exec sha256sum {} \; 2>/dev/null || echo "no V1 artifacts on this host"
```

Byte-identical to the step-2 snapshot (or still absent).

## 9. `target_access = 0`

```
grep -r '"target_access"' reports/full/c8/reliability/post_failure_source_diagnostics_v2/DIAGNOSTICS_INTERPRETATION.json
```

Must report `"target_access": 0`.

## 10. Commit and push the registered interpretation

```
git status --short
git add reports/full/c8/reliability/post_failure_source_diagnostics_v2/DIAGNOSTICS_INTERPRETATION.json
git add reports/full/c8/reliability/post_failure_source_diagnostics_v2/DIAGNOSTICS_INTERPRETATION.md
git commit -m "feat(c9): register observed post-failure diagnostic interpretation"
git push origin portable-one-command-full-run
```

Review `git status --short` before staging — add ONLY the two named files
by explicit path. Never `git add .` / `git add -A`.

## 11. After this — what is NOT authorized yet

Nothing in this handoff authorizes: rerunning any diagnostic, editing
`DIAGNOSTICS_RESULT.json`/`DIAGNOSTICS_PER_TEST.json`/
`DIAGNOSTICS_PROVENANCE.json`/`DIAGNOSTICS_VERDICT.json`, editing the BA_sep
protocol or `DETECTOR_RELIABILITY_LOCK_C.json`, resolving
`cross_route_synthetic`/`recipe_region_shift`/`artifact_map_swap` under an
invented metric, or starting
`POST_FAILURE_EXPLORATORY_TARGET_V1`. That protocol is a SEPARATE, future
task — it will reuse the frozen C8 checkpoints but must not pretend the
original reliability gate passed. No target root, path, or label may be
opened by this task or its outputs.
