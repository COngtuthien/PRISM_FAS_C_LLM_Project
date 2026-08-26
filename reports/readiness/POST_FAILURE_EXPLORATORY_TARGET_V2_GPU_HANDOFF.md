# POST_FAILURE_EXPLORATORY_TARGET_V2 — GPU handoff

**DO NOT RUN TARGET INFERENCE MERELY BECAUSE THE CODE COMPILES.** This
handoff authorizes exactly the nine steps below — through binding and
status verification — and STOPS. It does **not** authorize `--predict`.
The future `--predict` command is provided at the end, clearly labeled
`NOT YET EXECUTED / REQUIRES FINAL HUMAN AUDIT`.

This supersedes `POST_FAILURE_EXPLORATORY_TARGET_V1_GPU_HANDOFF.md`. V1 was
never scientifically executed and remains unchanged, historical. V2's
frozen protocol identity is
`2f1beb0b95f01051e06c0ef8a82d06a759d0fe8f81f693c5d3a4d777845196a9`.

BA_sep remains permanently FAILED; `DETECTOR_RELIABILITY_LOCK_C` remains
`overall=FAILED`; `POST_FAILURE_SOURCE_DIAGNOSTICS_V2` remains
`overall=FAIL`; the original C9 confirmatory path remains BLOCKED. This is
a separate, exploratory, post-failure branch — not a substitute for any of
the above.

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

## 2. Protected upstream checksum snapshot

```
sha256sum configs/evaluation/c9_detector_ba_sep_option1_v2.yaml
sha256sum configs/evaluation/c9_post_failure_source_diagnostics_v1.yaml
sha256sum configs/evaluation/c9_post_failure_source_diagnostics_v2.yaml
sha256sum configs/evaluation/post_failure_exploratory_target_v1.yaml
sha256sum configs/evaluation/post_failure_exploratory_target_v2.yaml
sha256sum configs/evaluation/m10_target.yaml
sha256sum reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json
sha256sum reports/full/c8/C8_ACCEPTANCE.json
find reports/full/exploratory_target_v1 -type f -exec sha256sum {} \; 2>/dev/null || echo "no V1 artifacts on this host"
find reports/full/exploratory_target_v2 -type f -exec sha256sum {} \; 2>/dev/null || echo "no V2 artifacts on this host yet"
```

Save this output outside the repo tree.

## 3. V2 `--preflight-only` (read-only)

```
python -m prism_fas.evaluation.post_failure_exploratory_target_v2 --repo . --preflight-only
```

Expect `protocol_resolved: true`, `matrix_resolved: true`, `row_count: 24`,
`rows_bindable: true`, `c8_matrix_identity_resolvable: true`,
`target_feature_package.present_on_this_host: true` and `.verified: true`
with `computed_identity ==
c3a29e695ad08c4b31e01533f1d12374f4e30c51f0167c6622cf8168792e48a8`, and
`target_label_root_sealed.target_labels_opened: false`. Exit `0`
(`ready_for_bind: true`). STOP if any of these is false.

## 4. Verify the target FEATURE package identity ONLY

Step 3 already computed and compared the real package identity without
opening any label. Confirm independently, still without touching the label
root:

```
grep -c '"target_labels_opened": false' <(python -m prism_fas.evaluation.post_failure_exploratory_target_v2 --repo . --preflight-only)
```

## 5. Prove the label firewall is sealed

```
python -c "
from pathlib import Path
from prism_fas.evaluation.post_failure_exploratory_target_v2 import load_protocol
from prism_fas.evaluation.post_failure_exploratory_target import verify_target_label_root_sealed
protocol = load_protocol(Path('.'))
print(verify_target_label_root_sealed(Path('.'), protocol))
"
```

Expect `{'stage': 'G7', 'label_root_permission': 'deny', 'label_root_declared': True, 'label_root_exists': True, 'target_labels_opened': False}`.

## 6. V2 `--bind-prediction-plan`

```
python -m prism_fas.evaluation.post_failure_exploratory_target_v2 --repo . --bind-prediction-plan
```

Expect exit `0`, `bound: true`, `reused: false` (first run),
`target_feature_package_identity_verified: true`, `target_label_access: 0`,
and `reports/full/exploratory_target_v2/PREDICTION_PLAN_BINDING.json`
written with all 24 rows' full field sets (including `flags`). Record
`protocol_identity` and `target_matrix_identity`.

## 7. Second bind proving idempotence

```
python -m prism_fas.evaluation.post_failure_exploratory_target_v2 --repo . --bind-prediction-plan
```

Expect `reused: true`, byte-identical to step 6's write.

```
sha256sum reports/full/exploratory_target_v2/PREDICTION_PLAN_BINDING.json
```

Confirm unchanged from step 6.

## 8. Validate the frozen binding

```
python -c "
from pathlib import Path
from prism_fas.evaluation.post_failure_exploratory_target_v2 import build_prediction_plan_binding, verify_binding_unchanged
import json
frozen = json.load(open('reports/full/exploratory_target_v2/PREDICTION_PLAN_BINDING.json'))
print(verify_binding_unchanged(Path('.'), frozen))
"
```

Expect `{'unchanged': True}` — a read-only recomputation matches the frozen
binding exactly. If it does not, STOP and report the drift; do not proceed
to `--predict` under any circumstance until the drift's cause is understood.

## 9. `--status`

```
python -m prism_fas.evaluation.post_failure_exploratory_target_v2 --repo . --status
```

Expect `prediction_plan_bound: true`, `prediction_lock_exists: false`,
`target_labels_opened: false`, `reason: NO_PREDICTION_LOCK_YET`, exit `2`
(correct — no prediction has been made yet, this is not an error).

**STOP HERE.** Nothing beyond this point is authorized by this handoff.

---

## NOT YET EXECUTED / REQUIRES FINAL HUMAN AUDIT

Everything above is read-only or identity-binding. The command below is
the FIRST genuine scientific target access this branch would ever make.
It is provided here only as the documented next step — running it requires
a SEPARATE, explicit, future authorization, exactly like every prior
`--execute`/`--predict` step in this project's history:

```
python -m prism_fas.evaluation.post_failure_exploratory_target_v2 --repo . --predict
```

Before that authorization is given, a human should re-read: `verify_binding_unchanged`'s
step-8 output (must be `{'unchanged': True}` AGAIN, freshly, immediately
before running `--predict`, since time may have passed since step 8); the
exact 24 rows about to be predicted; and that
`reports/full/exploratory_target_v2/` on the GPU host is genuinely empty of
prior partial artifacts (or, if not, that `--status`/a direct listing shows
either nothing or a complete, previously-validated `TARGET_PREDICTION_LOCK.json`).

`--preflight-score`/`--score`
(`post_failure_exploratory_target_v2_scorer.py`) — Phase E2, which finally
opens `siw_target_labels.parquet` — require a FURTHER separate
authorization, reachable only after a valid, `FROZEN`, 24-entry V2
`TARGET_PREDICTION_LOCK.json` exists, and are not addressed by this
handoff at all.

## After this — what is NOT authorized yet

Nothing in this handoff authorizes: running `--predict`, opening the target
label artifact, running `--preflight-score`/`--score`, editing the BA_sep
protocol, `DETECTOR_RELIABILITY_LOCK_C.json`, `C8_ACCEPTANCE.json`, or
`configs/evaluation/m10_target.yaml`, running C9, C10, C11, C12 or C13, or
resolving `cross_route_synthetic`/`recipe_region_shift`/`artifact_map_swap`
under an invented metric.
