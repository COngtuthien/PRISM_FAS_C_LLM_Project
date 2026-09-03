# E6-v2 PAIRED_CURRENT_RUNTIME — Final Scientific Closure

## 1. Why this paired rerender existed

The historical ORIGINAL_LLM C5/C6 bank could not serve as E6's "original"
condition: reproducing its `trace.requested_coverage` under the current
production runtime was proven code-level impossible, and the historical
implementation that produced it is unrecoverable from git history. E6-v2
answers the underlying question cleanly instead — does the frozen LLM recipe
bank's benefit depend on cross-field joint associations, or on field
marginals alone? — by rendering BOTH `LLM_ORIGINAL_CURRENT_V2` and
`LLM_SHUFFLE_A_CURRENT_V2` fresh, under the identical current runtime, from
the identical frozen source-pair schedule.

## 2. Three technical execution failures, and their fixes

| Attempt | Real production observation | Root cause | Fix |
|---|---|---|---|
| 1 | Technical path-binding failure during render/quality resolution | Render work_root double-appended the arm segment; quality lookup fell back to the historical E6 root/arm | Corrected work_root contract (`v2_render_work_root`), explicit `candidates_root`/`arm` parameterization |
| 2 | Physics fillable = 16/512 | Quality reconstruction (`requested_support_for`) used LLM-SHUFFLE-A's recipe bank for EVERY arm, including ORIGINAL | `default_metrics_provider`/`default_quality_matcher` gained an explicit `quality_bank` override, built per-arm from that arm's own recipes |
| 3 | Physics fillable = 26/512 | `_support_masks`/`SampleStore.cached_mask` memoized region masks keyed by `(sample_id, recipe_id)` — identical across arms at the same schedule position by construction — while recipe CONTENT differs; the cache is shared for the life of the process across both arms | Cache keys extended to include `graph.recipe_hash`, a content hash, so identical-`recipe_id`/different-content requests never collide |

Existing rendered bytes were preserved at every step; no candidate was ever
re-rendered to work around a bug. Each attempt's provenance is recorded,
unmodified, in `E6_V2_ATTEMPT{1,2,3}_PROVENANCE.json`.

## 3. Why 16 and 26 are not scientific results

Both numbers were produced by code defects in the MEASUREMENT/MATCHING path,
not by anything about the rendered candidates or the frozen quality gate
itself. A read-only `--matching-sequence-preflight` diagnostic, run on the
real GPU host after the attempt-3 fix, reproduced the production execution
order (ORIGINAL then SHUFFLE, one process) and its reverse (SHUFFLE then
ORIGINAL) and found the two orders now agree exactly:

- Forward SHUFFLE Physics max fillable = 479
- Reverse SHUFFLE Physics max fillable = 479
- `ORDER_DEPENDENCE_PRESENT = False`

This confirms the cross-arm cache contamination is fully removed, and that
479 — not 16, not 26 — is the clean, reproducible, order-independent result.

## 4. Clean ORIGINAL feasibility

ORIGINAL_LLM_CURRENT_V2: GPAT max fillable = 512/512, Physics max fillable =
512/512. Both routes fill their full frozen quota. PASS.

## 5. Clean SHUFFLE infeasibility

LLM_SHUFFLE_A_CURRENT_V2: GPAT max fillable = 512/512 (PASS). Physics: 512
candidates pass the frozen quality gate in total — numerically enough to
fill 512 slots — but only 479 of the 512 required slots can actually be
FILLED once the frozen per-source-domain quota is applied. FAIL.

## 6. The exact CASIA deficit

| Domain | Quality-pass available | Frozen quota required | Fillable |
|---|---|---|---|
| CASIA-FASD | 231 | 264 | 231 |
| MSU-MFSD | 281 | 248 | 248 |
| **Total** | **512** | **512** | **479** |

CASIA deficit = 264 required − 231 available = **33**.
MSU surplus = 281 available − 248 required = **33**.

## 7. Why the MSU surplus cannot compensate

`select_route_bank` (the frozen §11.3 selector) fills each source domain
INDEPENDENTLY, up to `min(quota[domain], available[domain])`. The frozen
common-domain quota vector is exact per domain, not a single pooled total —
it exists specifically so no arm's bank can substitute an easy domain for a
hard one. MSU's 33-candidate surplus has no mechanism to fill CASIA's
33-candidate deficit under this rule; the two are numerically equal by
coincidence, not fungible by design.

## 8. Why training is blocked

`E6_V2_READY_FOR_TRAINING = False`. `E6_V2_TRAINING_PLAN_LOCK.json` already
plans 5 detector-training seeds per arm but records
`training_authorized_this_turn: false` and
`synthetic_bank_identity_status: "PENDING (no bank rendered this turn)"` —
no final SHUFFLE matched bank was ever selected or written this turn (or
any prior turn); TRAINING on an infeasible/incomplete bank is not attempted.
This repository's canonical milestone sequence is C0-C13 (see CLAUDE.md's 'Canonical entrypoint'); the c_ext_q1q2_v1 extension's own milestones are named E0-E6 (this being E6-v2, a paired rerender within E6). No milestone literally named 'E7' is defined anywhere in docs/PROJECT_STATE.md, .claude/skills/, or the source tree -- inventing one here would itself be a scope violation. The nearer, real downstream consumer —
`E6_V2_TRAINING_PLAN_LOCK.json` — is blocked for exactly this reason.

## 9. No target access, no LLM calls, no scientific parameter changes

`target_access = False` and `llm_api_calls = 0` throughout every attempt and
this closure. Across all three technical fixes and this closure: quota,
`q`, quality thresholds, the matching algorithm, source-domain policy, GPAT,
Physics, recipes and the source schedule were never changed.

## The one valid claim

Under this frozen rendering/quality/matching protocol, the shuffled arm
cannot satisfy the frozen Physics source-domain quota after quality gating.
This says nothing about whether shuffled recipes produce lower-quality
candidates in general, and nothing about any causal LLM-reasoning
advantage — the ORIGINAL arm's own quality-pass candidates happen to fall
into source domains the frozen quota can fill; the SHUFFLE arm's do not, by
33 candidates, in exactly one domain.
