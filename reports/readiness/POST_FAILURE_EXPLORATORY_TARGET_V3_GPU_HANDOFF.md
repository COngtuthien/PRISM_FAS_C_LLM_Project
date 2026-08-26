# POST_FAILURE_EXPLORATORY_TARGET_V3 — GPU handoff

**DO NOT RUN `--predict` MERELY BECAUSE THE CODE COMPILES.** This handoff
authorizes exactly the eight steps below and STOPS. It does **not**
authorize `--predict`; the future command is provided at the end, marked
`NOT AUTHORIZED UNTIL FINAL AUDIT`.

This supersedes both `POST_FAILURE_EXPLORATORY_TARGET_V1_GPU_HANDOFF.md`
and `..._V2_GPU_HANDOFF.md`. Neither V1 nor V2 was ever scientifically
executed and both remain unchanged, historical. V3's frozen protocol
identity is `a2b54f8844a2a36540e62470c2f5f30de52fbf509a37f03feb7f6d769d5c702c`.

BA_sep remains permanently FAILED; `DETECTOR_RELIABILITY_LOCK_C` remains
`overall=FAILED`; `POST_FAILURE_SOURCE_DIAGNOSTICS_V2` remains
`overall=FAIL`; the original C9 confirmatory path remains BLOCKED.

**The GPU host's git tree is NOT assumed clean.**

---

## 1. Safe git fast-forward

```
git status --short
git fetch origin
git log --oneline -1 origin/portable-one-command-full-run
git merge --ff-only origin/portable-one-command-full-run
```

STOP and report if `merge --ff-only` refuses. **NEVER** `git reset --hard`,
`git clean`, an unpopped `git stash`, `git add .`, or `git add -A`.

## 2. Checksum upstream science

```
sha256sum configs/evaluation/c9_detector_ba_sep_option1_v2.yaml
sha256sum configs/evaluation/post_failure_exploratory_target_v1.yaml
sha256sum configs/evaluation/post_failure_exploratory_target_v2.yaml
sha256sum configs/evaluation/post_failure_exploratory_target_v3.yaml
sha256sum configs/evaluation/m10_target.yaml
sha256sum reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json
sha256sum reports/full/c8/C8_ACCEPTANCE.json
find reports/full/exploratory_target_v1 -type f -exec sha256sum {} \; 2>/dev/null || echo "no V1 artifacts"
find reports/full/exploratory_target_v2 -type f -exec sha256sum {} \; 2>/dev/null || echo "no V2 artifacts"
find reports/full/exploratory_target_v3 -type f -exec sha256sum {} \; 2>/dev/null || echo "no V3 artifacts yet"
```

Save this output outside the repo tree.

## 3. V3 `--preflight-only`

```
python -m prism_fas.evaluation.post_failure_exploratory_target_v3 --repo . --preflight-only
```

Expect `protocol_resolved: true`, `matrix_resolved: true`, `row_count: 24`,
`rows_bindable: true`, `c8_matrix_identity_resolvable: true`,
`target_feature_package.verified: true` with `computed_identity ==
c3a29e695ad08c4b31e01533f1d12374f4e30c51f0167c6622cf8168792e48a8`, and
`target_label_root_sealed.target_labels_opened: false`. Exit `0`
(`ready_for_bind: true`). `access_state` must be all-false. STOP if any of
this is not true.

## 4. Feature-package identity verification (already performed by step 3)

Confirm independently, still without touching any label:

```
python -c "
from pathlib import Path
from prism_fas.evaluation.post_failure_exploratory_target_v3 import load_protocol
from prism_fas.evaluation.post_failure_exploratory_target import verify_target_feature_package_expected
protocol = load_protocol(Path('.'))
print(verify_target_feature_package_expected(Path('.'), protocol))
"
```

## 5. Prove the label root remains sealed

```
python -c "
from pathlib import Path
from prism_fas.evaluation.post_failure_exploratory_target_v3 import load_protocol
from prism_fas.evaluation.post_failure_exploratory_target import verify_target_label_root_sealed
protocol = load_protocol(Path('.'))
print(verify_target_label_root_sealed(Path('.'), protocol))
"
```

Expect `target_labels_opened: False`, `label_root_permission: 'deny'`.

## 6. Bind the prediction plan

```
python -m prism_fas.evaluation.post_failure_exploratory_target_v3 --repo . --bind-prediction-plan
```

Expect exit `0`, `bound: true`, `reused: false`,
`target_feature_package_identity` equal to the value from step 3/4, and
`access_state.target_feature_identity_accessed: true` with
`target_prediction_features_accessed: false` and
`target_labels_accessed: false`. `reports/full/exploratory_target_v3/PREDICTION_PLAN_BINDING.json`
is written with all 24 rows carrying a non-empty
`target_feature_package_identity`.

## 7. Bind again — exact idempotence

```
python -m prism_fas.evaluation.post_failure_exploratory_target_v3 --repo . --bind-prediction-plan
```

Expect `reused: true`, byte-identical to step 6.

```
sha256sum reports/full/exploratory_target_v3/PREDICTION_PLAN_BINDING.json
```

Confirm unchanged from step 6.

## 8. `--status`

```
python -m prism_fas.evaluation.post_failure_exploratory_target_v3 --repo . --status
```

Expect `prediction_plan_bound: true`, `prediction_lock_exists: false`,
`reason: NO_PREDICTION_LOCK_YET`, exit `2` (correct — no prediction has
been made yet).

**STOP HERE.** Nothing beyond this point is authorized by this handoff.

---

## NOT AUTHORIZED UNTIL FINAL AUDIT

```
python -m prism_fas.evaluation.post_failure_exploratory_target_v3 --repo . --predict
```

This is the FIRST genuine scientific target access this branch would ever
make. It writes to a disposable staging namespace first and promotes to
the final scientific row directories only after all 24 rows succeed and
validate, with the overall `TARGET_PREDICTION_LOCK.json` written last — but
it is still real GPU compute requiring its own separate, explicit, future
authorization. Immediately before running it, re-verify: `verify_binding_unchanged`
against the frozen `PREDICTION_PLAN_BINDING.json` returns
`{'unchanged': True}` (recompute fresh, do not trust a stale note); the
current git commit is the one intended for `prediction_execution_code_commit`;
and `reports/full/exploratory_target_v3/` and
`runs/exploratory_target_v3/` are either empty or hold only a
previously-validated, complete result.

`--preflight-score`/`--score`
(`post_failure_exploratory_target_v3_scorer.py`) — Phase E2, which writes
the one-way `TARGET_LABEL_REVEAL.json` and finally opens
`siw_target_labels.parquet` — require a FURTHER separate authorization,
reachable only after a valid, `FROZEN`, 24-entry V3
`TARGET_PREDICTION_LOCK.json` exists, and are not addressed by this
handoff at all.

## After this — what is NOT authorized yet

Nothing in this handoff authorizes: running `--predict`, opening the
target label artifact, running `--preflight-score`/`--score`, editing the
BA_sep protocol, `DETECTOR_RELIABILITY_LOCK_C.json`, `C8_ACCEPTANCE.json`,
or `configs/evaluation/m10_target.yaml`, running C9, C10, C11, C12 or C13,
or resolving `cross_route_synthetic`/`recipe_region_shift`/`artifact_map_swap`
under an invented metric.
