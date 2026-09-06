# E7-v1.1 RESERVE AMENDMENT

**Status:** `FROZEN_POST_OBSERVATION_PRE_RENDER` — ratified by the protocol owner; NOT an E0-preregistered rule.
**Reserve-schedule rule identity:** `7871404603876a7b015af80cefaca26d94d54f102a9eb445099254aa3bbd1822`
**Governing spec:** EXT-Q1Q2 Detailed Spec v1.0, section 6.1 (Unified synthetic candidate budget).
**Authority audit this amendment resolves:** `../RESERVE_SCHEDULE_AUTHORITY_AUDIT.md` / `.json` — found no authoritative exact reserve schedule anywhere; Option 1 there is what this amendment ratifies.
**Implementation module:** `src/prism_fas/evaluation/c_ext_e7_reserve_schedule.py` (additive; no frozen C5/C6/GPAT synthesis primitive modified).

## 1. Why this amendment exists

The first authorized E7 `generate_and_match` attempt ran for all three extension folds under protocol v1.0. Every one of G-RND/G-DET/G-LLM, in every fold, terminated:

| Fold | G-RND | G-DET | G-LLM |
|---|---|---|---|
| EXT-F1 | `BLOCKED_PROTOCOL_GAP_RESERVE_SCHEDULE` | `BLOCKED_PROTOCOL_GAP_RESERVE_SCHEDULE` | `BLOCKED_PROTOCOL_GAP_RESERVE_SCHEDULE` |
| EXT-F2 | `BLOCKED_PROTOCOL_GAP_RESERVE_SCHEDULE` | `BLOCKED_PROTOCOL_GAP_RESERVE_SCHEDULE` | `BLOCKED_PROTOCOL_GAP_RESERVE_SCHEDULE` |
| EXT-F3 | `BLOCKED_PROTOCOL_GAP_RESERVE_SCHEDULE` | `BLOCKED_PROTOCOL_GAP_RESERVE_SCHEDULE` | `BLOCKED_PROTOCOL_GAP_RESERVE_SCHEDULE` |

No fold produced a valid core matched synthetic bank; no detector training was performed; no target labels were accessed; no LLM API calls occurred. These three v1.0 terminal observations are the **sole reason** the pre-existing §6.1 reserve-tranche mechanism became scientifically necessary, and they are the trigger this amendment exists to resolve. **These v1.0 observations are immutable**: their `GENERATION_CLOSURE.json` files are never rewritten, rescued, or reinterpreted as successful by this amendment or its implementation module, and the original initial candidate pools are never deleted, rerun, or replaced.

This amendment is drafted and ratified **after** observing that trigger and **before** any reserve candidate is rendered. It is explicitly **not** part of, and must never be recorded as, the E0 preregistration freeze.

## 2. What was ratified (Option 1, exactly as documented in the audit)

1. Reserve scheduling is an **additive**, deterministic extension of the existing frozen C5 position-keyed source-scheduling semantics.
2. `c5_source_pair_plan.load_source_rows`/`live_for_position`/`spoof_for_position`/`candidate_identity` and `c5_arm_plan.load_arm_bank`/`_recipe_id` are reused **verbatim**, unmodified.
3. Initial candidate positions/identities `0..2047` remain **completely unchanged**.
4. **No** frozen C5/C6/GPAT synthesis primitive file is modified.
5. **Block route assignment** for reserve positions (the first 256 positions of a tranche are Physics, the next 256 are GPAT) — as proposed in the audit, not the alternative per-recipe interleaving.

## 3. The complete deterministic schedule

For tranche `t` in `1..4`, one arm/fold's 512 reserve positions are numbered:

```
position = 2048 + 512*(t-1) + route_offset + recipe_ordinal
route_offset = 0   for PHYSICS  (positions  local 0..255)
route_offset = 256 for GPAT     (positions  local 256..511)
recipe_ordinal ∈ 0..255
```

- **Position range** — `2048..4095` total across all 4 tranches/both routes (`RESERVE_POSITION_BASE = 2048` = `c5_source_pair_plan.CANDIDATES_PER_ARM`, the exact one-past-the-end of the frozen v1.0 pool). Strictly disjoint from, and never overlapping, `0..2047`.
- **Route assignment** — BLOCK: within one tranche, positions `0..255` (local) are Physics, `256..511` (local) are GPAT.
- **Recipe-ordinal assignment** — `recipe_ordinal = local_index_within_route_block (0..255)` — one additional full cycle through the frozen 256-recipe bank per tranche per route (the same shape the v1.0 schedule already uses: 4 Physics + 4 GPAT renders/recipe before any reserve tranche opens).
- **Domain-relation assignment (GPAT only)** — whole-tranche alternation: `SAME_DOMAIN` for odd tranches (1, 3), `CROSS_DOMAIN` for even tranches (2, 4). At the 4-tranche cap this is exactly 2 SAME + 2 CROSS tranches (512 + 512 GPAT reserve candidates), the same 50/50 balance the frozen schedule already guarantees, achieved cumulatively across tranches rather than within one.
- **Live sample assignment** — `live_for_position(live_list, position)` reused verbatim, called with the candidate's own (≥2048) global position. Pure function of `(position, live_list)`, well-defined for any position.
- **Spoof-partner assignment (GPAT only)** — `spoof_for_position(spoof_list, live, position, relation, seed=PLAN_SEED)` reused verbatim.
- **Seed/hash material** — the exact frozen `PLAN_SEED = 20260806` and the exact frozen SHA256-keyed spoof-selection rule; no new seed, no new hash scheme.
- **Slot numbering** — extends the frozen v1.0 per-recipe slot numbering (`0..7`) naturally: `slot = 8 + 2*(tranche-1) + (0 if PHYSICS else 1)`, giving `8,9` (tranche 1), `10,11` (tranche 2), `12,13` (tranche 3), `14,15` (tranche 4) — never colliding with a v1.0 slot, never reused across tranche/route.
- **Candidate-ID construction** — `candidate_identity(...)` reused verbatim, over the reserve row's own `(route, recipe_ordinal, live, spoof, domain_relation, position)` tuple. A reserve candidate's identity is therefore automatically distinct from every `0..2047` candidate (different `position` input) and automatically identical in construction rule across RND/DET/LLM — the arms differ only through their own already-frozen `recipe_bank_identity`/`recipe_id`.
- **Relationship to the original frozen C5 plan** — every reserve row carries the SAME `source_pair_plan_identity`/`package_identity` the fold's v1.0 base plan already locked (verified, not merely assumed: `build_reserve_base_schedule` fails closed if the live/spoof pool resolved right now has drifted from what the base plan recorded).

## 4. Budget (unchanged from §6.1)

| | Physics | GPAT |
|---|---|---|
| Initial | 1024 | 1024 |
| Goal accepted | 512 | 512 |
| Reserve tranche | +256/tranche | +256/tranche |
| Cap | 2048 | 2048 |

Maximum 4 tranches, 4096 total candidates/arm at the cap. Never lower the quality threshold. If still short of 512 accepted on any arm-route after the cap: `SCIENTIFICALLY_BLOCKED`, no oversampling.

**Orchestration (future work, not implemented by this module):** after each tranche opens (synchronously, all 3 arms × 2 routes), re-run the exact frozen `c5_render.render_arm` / `c6_scientific.evaluate_pool`/`gate_candidates` / `c6_matched_bank.build_matched_banks` over the **cumulative** pool; if still infeasible and `t < 4`, open tranche `t+1`; if infeasible at `t = 4`, `SCIENTIFICALLY_BLOCKED`. This loop is **not** implemented by `c_ext_e7_reserve_schedule.py` — only the deterministic membership this loop would consume is.

## 5. Outcome independence (mandatory, structural)

No function in `c_ext_e7_reserve_schedule.py` accepts an accepted-count, a quality score, a `q` value, a prior tranche's result, target data, or any other observed outcome as an input. `materialize_reserve_schedule_for_fold`'s signature has no `arm=`/`route=` parameter at all — synchronization across all 3 arms and both routes is a structural guarantee of the API shape, not a runtime check performed after the fact. Membership is a pure function of `(fold_id, tranche, route, recipe_ordinal, arm)` only. Verified by test (`test_f_*`, `test_e_*` in `tests/pipeline/test_c_ext_e7_reserve_schedule.py`).

## 6. Shuffle-A policy (unchanged, not rescued)

The core RND/DET/LLM anchor bank for a fold must close (reach a common matched-feasible profile) **before** that fold's Shuffle-A policy is evaluated at all. This amendment does not implement Shuffle reserve as an independent rescue mechanism. EXT-F1's historical Shuffle scientific infeasibility (`BLOCKED_TRUE_FROZEN_MATCHED_BANK_INFEASIBILITY`) is **not** rescued by this amendment and is not touched by `c_ext_e7_reserve_schedule.py` at all.

## 7. What this task does NOT do

Laptop-only. No reserve candidate is rendered, no image is generated, no quality evaluation runs, no GPU rendering occurs, no rsync to GPU, no detector training, no LLM call, no commit, no push. Only the deterministic schedule/identity construction and its lock/tests/dry-run preview are produced. Authorizing and executing the first reserve tranche is a separate, later, explicit decision.
