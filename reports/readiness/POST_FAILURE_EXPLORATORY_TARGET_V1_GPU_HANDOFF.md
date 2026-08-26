# POST_FAILURE_EXPLORATORY_TARGET_V1 — GPU handoff

**DO NOT RUN A SCIENTIFIC TARGET COMMAND ON THIS LAPTOP.** This document
hands off exact steps for the GPU scientific host, which has CUDA, the real
`prism_target_eval_v2` feature package, and the real `runs/full/c8/`
checkpoints this laptop does not have. It authorizes six read-only/binding
actions now. It does **not** yet authorize a scientific `--predict` — that
is named separately, at the end, as the clearly-distinguished next
execution step.

This is a SEPARATE, EXPLORATORY, POST-FAILURE branch. It does not reopen,
weaken, or substitute for the original C9 confirmatory path, which remains
`BLOCKED_BY_DETECTOR_RELIABILITY_FAILURE`. `synthetic_vs_real_spoof_probe`
remains permanently FAILED; `DETECTOR_RELIABILITY_LOCK_C` remains
`overall=FAILED`; `C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2` remains
`overall=FAIL`. The exploratory protocol identity is
`8fb806d25a80ecd3c7d44cfeba8c893a5f115b8b51797220a51132ba16708b51`.

**The GPU host's git tree is NOT assumed clean.**

---

## 1. Safe git update (never destructive)

```
git status --short
git fetch origin
git log --oneline -1 origin/portable-one-command-full-run
git merge --ff-only origin/portable-one-command-full-run
```

STOP and report if `merge --ff-only` refuses. **NEVER** `git reset --hard`,
`git clean`, an unpopped `git stash`, `git add .`, or `git add -A`.

## 2. Protected-state checksum snapshot (BEFORE anything else)

```
sha256sum configs/evaluation/c9_detector_ba_sep_option1_v2.yaml
sha256sum configs/evaluation/c9_post_failure_source_diagnostics_v1.yaml
sha256sum configs/evaluation/c9_post_failure_source_diagnostics_v2.yaml
sha256sum configs/evaluation/post_failure_exploratory_target_v1.yaml
sha256sum configs/evaluation/m10_target.yaml
sha256sum reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json
sha256sum reports/full/c8/C8_ACCEPTANCE.json
sha256sum reports/full/c8/reliability/synthetic_vs_real_spoof_probe/*.json
find reports/full/c8/reliability/post_failure_source_diagnostics_v1 -type f -exec sha256sum {} \; 2>/dev/null || true
find reports/full/c8/reliability/post_failure_source_diagnostics_v2 -type f -exec sha256sum {} \;
find reports/full/exploratory_target_v1 -type f -exec sha256sum {} \; 2>/dev/null || echo "no exploratory artifacts on this host yet"
```

Save this output outside the repo tree.

## 3. Exploratory `--preflight-only` (read-only)

```
python -m prism_fas.evaluation.post_failure_exploratory_target --repo . --preflight-only
```

Expect `protocol_resolved: true`, `matrix_resolved: true`, `row_count: 24`,
`rows_bindable: true` (all 24 rows' checkpoint/calibration resolve for
real), `c8_matrix_identity_resolvable: true`, `target_feature_package`
reporting `present_on_this_host: true` and `verified: true` with
`computed_identity == expected_identity ==
c3a29e695ad08c4b31e01533f1d12374f4e30c51f0167c6622cf8168792e48a8`, and
`target_label_root_sealed.target_labels_opened: false`. Exit `0`
(`ready_for_bind: true`). If ANY of these is false, STOP — resolve the
reported precondition; do not force the next step.

## 4. Bind the exact prediction plan

```
python -m prism_fas.evaluation.post_failure_exploratory_target --repo . --bind-prediction-plan
```

Expect exit `0`, `bound: true`, `reused: false` (first run), and
`reports/full/exploratory_target_v1/PREDICTION_PLAN_BINDING.json` written —
zero scientific metric, pure identity/plan resolution. Record
`protocol_identity` and `target_matrix_identity` from the output.

```
python -m prism_fas.evaluation.post_failure_exploratory_target --repo . --bind-prediction-plan
```

Run a SECOND time. Expect `reused: true`, identical bytes to the first
write — proof the bind is deterministic.

## 5. Verify the target feature package identity WITHOUT opening labels

The `--preflight-only`/`--bind-prediction-plan` steps above already
compute the real package identity (a hash over the package's own files) and
compare it to the frozen expected value — this never opens
`data/evaluation_only/prism_target_v2_labels`. Confirm independently:

```
grep -r '"target_labels_opened"' reports/full/exploratory_target_v1/PREDICTION_PLAN_BINDING.json
```

Must report `false`.

## 6. Confirm the target label root remains unopened

```
python -c "import json; d = json.load(open('reports/full/exploratory_target_v1/PREDICTION_PLAN_BINDING.json')); print(d['target_label_root_seal'])"
```

Expect `{'stage': 'G7', 'label_root_permission': 'deny', 'label_root_declared': True, 'label_root_exists': True, 'target_labels_opened': False}`.

```
sha256sum reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json reports/full/c8/C8_ACCEPTANCE.json configs/evaluation/m10_target.yaml
```

Must be byte-identical to the step-2 snapshot. `configs/evaluation/m10_target.yaml`
in particular must remain untouched — this exploratory branch never writes
to it.

---

## NEXT SCIENTIFIC EXECUTION STEP (separate authorization required)

Everything above is read-only or identity-binding. The following is the
FIRST genuine scientific target access this branch would ever make, and is
named here only as the clearly-separated next step — running it is **not**
authorized by this handoff alone; it requires the user's explicit go-ahead
in a future task, exactly as the original diagnostics `--execute` steps
each required their own explicit authorization:

```
python -m prism_fas.evaluation.post_failure_exploratory_target --repo . --predict
```

This runs REAL, label-free inference for all 24 rows (reusing
`target_prediction.target_batches`/`predict_target` verbatim), writes each
row's predictions plus its own `PREDICTION_LOCK.json`, then the overall
`TARGET_PREDICTION_LOCK.json`. It is scientifically no-rerun by
construction (a complete `FROZEN` lock re-reports with zero inference; a
partial prediction set BLOCKS) — but it is still real GPU compute the user
must knowingly authorize, EXACTLY ONCE. After it runs, `--status` and a
second `--predict` call should be exercised to prove no-rerun, and the four
original result artifacts / BA_sep artifacts / `DETECTOR_RELIABILITY_LOCK_C.json`
/ `C8_ACCEPTANCE.json` should be byte-compared against the step-2 snapshot
to prove none moved.

`--preflight-score`/`--score` (Phase E2, `post_failure_exploratory_target_scorer.py`)
are a FURTHER-separate authorization, reachable only after a `FROZEN`
`TARGET_PREDICTION_LOCK.json` exists — they are the step that finally opens
`siw_target_labels.parquet`, and are not addressed by this handoff at all.

## After this — what is NOT authorized yet

Nothing in this handoff authorizes: opening the target label artifact,
running `--predict` without a separate explicit go-ahead, running
`--preflight-score`/`--score`, editing the BA_sep protocol,
`DETECTOR_RELIABILITY_LOCK_C.json`, `C8_ACCEPTANCE.json`, or
`configs/evaluation/m10_target.yaml`, running C9, C10, C11, C12 or C13, or
resolving `cross_route_synthetic`/`recipe_region_shift`/`artifact_map_swap`
under an invented metric.
