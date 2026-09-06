# E7-v1.1 Terminal Run-State Bookkeeping Correction

**Classification:** technical metadata / bookkeeping fix only. **No scientific result changes.**
**Discovered after:** EXT-F1's real terminal Reserve Tranche 4 run, implementation commit `db87b1c721da99d89d6b9439418e94ab46e68767`.
**Fixed in:** `src/prism_fas/evaluation/c_ext_e7_reserve_orchestrator.py` (this correction), NOT `c_ext_e7_reserve_schedule.py` (untouched).
**Reserve schedule rule identity:** unchanged — `7871404603876a7b015af80cefaca26d94d54f102a9eb445099254aa3bbd1822`.

## Historical sequence (audit trail — preserved in order, nothing overwritten)

1. **E7-v1.0** — EXT-F1's initial frozen 2048-candidate pool terminated `BLOCKED_PROTOCOL_GAP_RESERVE_SCHEDULE` on every one of G-RND/G-DET/G-LLM (no authoritative reserve schedule existed yet).
2. **E7-v1.1 amendment** — the reserve mechanism (`c_ext_e7_reserve_schedule.py`, rule identity `7871404603876a7b015af80cefaca26d94d54f102a9eb445099254aa3bbd1822`) was ratified and frozen *before* any reserve candidate was rendered, referencing observation 1 above as its trigger.
3. **EXT-F1 Tranche 1** — rendered and evaluated; cumulative result `NEEDS_NEXT_RESERVE_TRANCHE`.
4. **EXT-F1 Tranche 2** — rendered and evaluated; cumulative result `NEEDS_NEXT_RESERVE_TRANCHE`.
5. **EXT-F1 Tranche 3** — rendered and evaluated; cumulative result `NEEDS_NEXT_RESERVE_TRANCHE`.
6. **EXT-F1 Tranche 4** — rendered and evaluated under implementation commit `db87b1c721da99d89d6b9439418e94ab46e68767`; cumulative result `SCIENTIFICALLY_BLOCKED_AFTER_RESERVE_CAP` — **this is EXT-F1's scientific result of record and is immutable.**
7. **Post-run technical observation** — inspecting the artifacts from step 6 after the fact found that `RESERVE_RUN_STATE.json`'s `cumulative_counts` field had been left at tranche 3's stale `1792/1792` instead of tranche 4's true `2048/2048` (the scientific closure itself, `RESERVE_FINAL_CLOSURE.json`, was and remains correct).
8. **Correction** (this note, this commit) — `c_ext_e7_reserve_orchestrator.py` fixed so every *future* fold's terminal cap-block run records `2048/2048` correctly in its own `RESERVE_RUN_STATE.json` directly; EXT-F1's own historical `RESERVE_RUN_STATE.json` is left exactly as step 6 produced it, not repaired in place.

## What happened

EXT-F1 completed Reserve Tranche 4 under commit `db87b1c...`. The scientific final status is:

```
SCIENTIFICALLY_BLOCKED_AFTER_RESERVE_CAP
```

`RESERVE_TRANCHE_04_CLOSURE.json`/`RESERVE_FINAL_CLOSURE.json` prove tranche 4 was genuinely included in the cumulative scientific evaluation (real PERMISSIVE-profile assessment against the full cumulative pool: RND Physics=783/GPAT=1155, DET Physics=774/GPAT=1105, LLM Physics=295/GPAT=968; common quota Physics=295/512, GPAT=512/512; Physics shortfall=217; final cumulative planned route counts Physics casia_fasd=1181+msu_mfsd=867=2048, GPAT casia_fasd=1164+msu_mfsd=884=2048). **T4 genuinely used 2048 candidates/route/arm.** This scientific result is correct and is not altered by this correction in any way.

However, `RESERVE_RUN_STATE.json`, written immediately after that same T4 run, incorrectly retained:

```
cumulative_counts: {physics: 1792, gpat: 1792}   (per arm)
```

instead of the correct terminal value:

```
cumulative_counts: {physics: 2048, gpat: 2048}   (per arm)
```

`next_tranche` correctly became `null` and the terminal `status` correctly became `SCIENTIFICALLY_BLOCKED_AFTER_RESERVE_CAP` — only the `cumulative_counts` metadata field was stale.

## Root cause (confirmed from code, not assumed)

In `open_and_close_tranche`, the `decision.selected is None` branch had two paths:

```python
if tranche < rs.MAX_TRANCHES:
    new_status = STATUS_NEEDS_NEXT_TRANCHE
    new_state = {**state, "status": new_status, "next_tranche": tranche + 1,
                "cumulative_counts": {arm: {route: V1_0_CUMULATIVE_PER_ROUTE[route] + 256 * tranche
                                           for route in (spp.PHYSICS, spp.GPAT)}
                                     for arm in spp.ARMS}}
else:
    new_status = STATUS_CAP_BLOCKED
    new_state = {**state, "status": new_status, "next_tranche": None, "closed_at_tranche": None}
```

The non-terminal branch recomputes `cumulative_counts` from `tranche`. The terminal cap-block branch (`tranche == rs.MAX_TRANCHES`) built `new_state` by spreading `**state` (the run state as it stood *before* this call, i.e. still carrying tranche 3's `1792/1792`) and never recomputed `cumulative_counts` at all — so it silently inherited the previous tranche's stale value.

## Fix

Factored the count computation into one pure helper, `_cumulative_counts_after_tranche(tranche)` (`V1_0_CUMULATIVE_PER_ROUTE[route] + 256 * tranche`, per arm), and called it from **both** branches — including the terminal cap-block branch, which now correctly computes `2048/2048` for `tranche = 4`. No other behavior changed: the non-terminal branch's numeric formula is byte-identical to before (just no longer duplicated inline), and the `CLOSED_MATCHED` (feasible) branch — a structurally separate code path (`decision.selected is not None`) — was not touched at all.

## What this correction does NOT do

- Does not rerun, modify, delete, or repair any EXT-F1 scientific artifact.
- Does not regenerate any candidate, re-evaluate any quality metric, or reopen tranche 4.
- Does not mutate EXT-F1's `RESERVE_FINAL_CLOSURE.json` or any `RESERVE_TRANCHE_0N_CLOSURE.json`.
- Does not change reserve schedule membership, candidate IDs, tranche boundaries, the STRICT/NOMINAL/PERMISSIVE thresholds or profile ordering, eligibility decisions, route quotas, the matched-bank selector, the final 512+512 requirement, the cap size, GPAT checkpoints, the Physics renderer, the target firewall, or LLM-call policy.
- Does not alter `c_ext_e7_reserve_schedule.py`, `c_ext_e7_gpat_bank.py`, or any frozen C5/C6/GPAT primitive (verified: zero diff).
- Does not claim this correction was present before EXT-F1 was run — it was **not**; it is dated and attributed to this correction, discovered after observing EXT-F1's real terminal run under commit `db87b1c721da99d89d6b9439418e94ab46e68767`.

## How to interpret the historical EXT-F1 run

The **scientific result of record** for EXT-F1's terminal reserve run is `RESERVE_FINAL_CLOSURE.json` (status `SCIENTIFICALLY_BLOCKED_AFTER_RESERVE_CAP`, with its `assessments`/`profile_decision` evidence) — that file was correct at the time it was written and remains correct. The historical `RESERVE_RUN_STATE.json` produced by that same run carries a stale `cumulative_counts` value (`1792/1792` instead of `2048/2048`) and **must not be read as evidence of how many reserve candidates were actually evaluated** — for that, use `RESERVE_FINAL_CLOSURE.json`'s own assessment data, which independently reports the true cumulative route counts (Physics 2048, GPAT 2048, as reproduced above). This historical run-state artifact is left exactly as it was written (not corrected in place) so that EXT-F1's run remains attributable, byte-for-byte, to implementation commit `db87b1c721da99d89d6b9439418e94ab46e68767` — the commit that actually produced it.

## Effect on future folds

Any fold's terminal cap-block run *after* this correction is applied will record the correct `2048/2048` cumulative count in its own `RESERVE_RUN_STATE.json` directly — no separate correction note will be needed for those folds.
