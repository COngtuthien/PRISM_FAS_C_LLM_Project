# Reserve-Schedule Authority Audit

**Schema:** `ext-q1q2-e7-gpat-bank-reserve-schedule-authority-audit-v1`
**Scope:** READ-ONLY protocol/source audit — no reserve candidate invented or executed, no frozen artifact touched.
**JSON equivalent:** `RESERVE_SCHEDULE_AUTHORITY_AUDIT.json` (same directory), `audit_identity` = `68b19a31613c0deff341b2d669facfc97c152a2aeee73a033fb191e42c953817`.

## 1. Trigger context

The first authorized E7 `generate_and_match` attempt has run for all three extension folds. Observed terminal states:

| Fold | G-RND | G-DET | G-LLM |
|---|---|---|---|
| EXT-F1 | `BLOCKED_PROTOCOL_GAP_RESERVE_SCHEDULE` | `BLOCKED_PROTOCOL_GAP_RESERVE_SCHEDULE` | `BLOCKED_PROTOCOL_GAP_RESERVE_SCHEDULE` |
| EXT-F2 | `BLOCKED_PROTOCOL_GAP_RESERVE_SCHEDULE` | `BLOCKED_PROTOCOL_GAP_RESERVE_SCHEDULE` | `BLOCKED_PROTOCOL_GAP_RESERVE_SCHEDULE` |
| EXT-F3 | `BLOCKED_PROTOCOL_GAP_RESERVE_SCHEDULE` | `BLOCKED_PROTOCOL_GAP_RESERVE_SCHEDULE` | `BLOCKED_PROTOCOL_GAP_RESERVE_SCHEDULE` |

No fold produced a valid core matched synthetic bank. No detector training performed, no target labels accessed, no LLM API calls. This laptop checkout holds no local run/`BANK_LOCK`/`GENERATION_CLOSURE` artifacts from that attempt (it ran on the GPU host) — none were inspected, modified, deleted or re-run to produce this audit. This audit is a pure source/spec/history read.

## 2. Governing spec citation

**Document:** `/home/cong/PRISM_FAS_C_EXT_Q1Q2_Detailed_Spec_v1_0.docx` — named as the extension's governing spec by `src/prism_fas/evaluation/c_ext_e0_freeze.py:737-739` ("EXT-Q1Q2 Detailed Spec v1.0"). This is a **different** document from the main `docs/PRISM_FAS_C_LLM_v1_5_FINAL...docx` (CLAUDE.md's primary authority) and from `docs/spec_snapshot.md` (a snapshot of the unrelated PRISM-FAS-B v1.1 spec, whose own §6.1 is "Filesystem isolation").

**Extraction method:** read-only extraction of `word/document.xml` text from the `.docx` zip archive (no `python-docx` available in this environment); the result matches the existing repository paraphrases in `c_ext_e0_freeze.py` and `configs/c_ext_q1q2_v1/conditions.yaml` exactly.

**§6.1 "Unified synthetic candidate budget", verbatim:**

> Initial candidate generation per arm/fold: 1024 Physics + 1024 GPAT = 2048 candidates.
> Goal: exactly 512 accepted Physics + 512 accepted GPAT = 1024 final synthetic samples per arm.
> Nếu bất kỳ arm-route nào có <512 accepted, mở reserve tranche +256 candidates/route cho TẤT CẢ arms trong fold đó.
> Lặp đồng bộ tối đa đến 2048 candidates/route/arm (4096 total/arm).
> Không bao giờ hạ quality threshold. Nếu sau cap vẫn thiếu 512 ở bất kỳ arm-route nào: fold/condition = SCIENTIFICALLY BLOCKED, không dùng oversampling để giả đủ bank.
> Báo cáo candidate_count, passed_count, selected_count, reserve_tranche_count cho mọi arm-route.

**§6.1 confirms:**
- initial budget 1024 Physics + 1024 GPAT per arm/fold;
- goal 512 accepted Physics + 512 accepted GPAT;
- reserve trigger: any arm-route <512 accepted → open +256 candidates/route for **all** arms in that fold, synchronously;
- cap 2048 candidates/route/arm (4096 total/arm);
- never lower the quality threshold;
- underfill after cap → `SCIENTIFICALLY BLOCKED`, no oversampling to fake a full bank;
- must report `candidate_count`, `passed_count`, `selected_count`, `reserve_tranche_count` per arm-route.

**§6.1 does NOT specify:**
- reserve candidate ordinal / recipe ordinal mapping;
- reserve live-sample assignment;
- reserve GPAT spoof-partner assignment;
- reserve route/slot assignment;
- reserve same/cross-domain relation assignment;
- reserve `candidate_id` construction;
- any seed/hash rule for reserve positions.

**Adjacent section checked and ruled out:** a "3 deterministic schedule realizations" / mixed-radix "odometer mapping" mechanism appears later in the document (DET-A/B/C cyclic-offset construction: `candidate_combo_index(k) = (offset + k) mod N_combo`). This is the **C3 recipe-bank realization mechanism** for the E11 multi-bank feature (selecting 256 of 384 candidate recipes across independent bank realizations) — a wholly separate, already-frozen mechanism with no bearing on C5 candidate-render position scheduling or reserve tranches.

## 3. Repository audit

- **`configs/c_ext_q1q2_v1/conditions.yaml:49`**: `reserve_tranche_per_route: 256` — a numeric parameter matching the spec exactly, but grep-confirmed read by **zero** Python code anywhere in the repository.
- **`src/prism_fas/evaluation/c_ext_e0_freeze.py:317-325`**: echoes the same numeric budget into a descriptive `synthetic_candidate_budget` provenance dict. Computes and gates nothing.
- **Frozen synthesis primitives checked** (full-text grep for `reserve`/`tranche`): `c5_source_pair_plan.py`, `c5_arm_plan.py`, `c5_render.py`, `c5_raw_generation.py`, `c6_scientific.py`, `c6_matched_bank.py`, `gate_profiles.py`, `quality_gate.py`, `quality_calibration.py`, `synthetic_bank.py` — **zero matches** (the sole hit, `c5_raw_generation.py:260`, is the unrelated English word "Reserved" in a docstring about semantic-failure retention).
- **Hard-fixed-budget evidence** (why the primitives cannot silently grow the pool):
  - `c5_source_pair_plan.CANDIDATES_PER_ARM = RECIPES_PER_ARM(256) * RENDERS_PER_RECIPE(8) = 2048`, asserted by `_assert_schedule` on every build;
  - `c5_arm_plan._assert_arm_plan` asserts exactly 2048 candidates/arm, exactly 1024/1024 route split, exactly 8 renders/recipe, every time;
  - `c5_render.completeness()`'s own `rule` field, verbatim: *"a semantic generation failure is retained and reported; it is never resampled and the frozen 2048-per-arm budget never grows"*;
  - `gate_profiles.py`'s module docstring: *"Selection picks the strictest profile that yields the full matched cardinality in every arm... If none qualifies, C6 FAILS; it never relaxes the gate for the arm that fell short"*, and separately, `tau_out` is exempt from profiling because *"range/recipe-safe constraints are never relaxed beyond their frozen legal range"*.
- **Git history**: `git log --all --oneline -i --grep=reserve` → zero commits concern a reserve-candidate schedule.
- **Test suite**: all 19 `test_c5_*.py`/`test_c6_*.py` files — zero tests exercise a reserve-tranche candidate *schedule*; the only reserve-related tests anywhere (added this milestone, in `test_c_ext_e7_gpat_bank.py`) test that the **block** fires correctly and that its evidence is pre-bound — they do not assume or exercise a schedule, because none exists.
- **Docs/reports**: `docs/PROJECT_STATE.md`, `docs/spec_snapshot.md`, `reports/v15_reconciliation/*`, `docs/V15_PIPELINE_RESTRUCTURE_PLAN.md` — no reserve-tranche candidate-schedule content found. Every other repository-wide "reserve" substring hit is the English word "preserved"/"preserves" (false positive of case-insensitive search).

## 4. Answers

| Question | Answer |
|---|---|
| Does an authoritative exact reserve schedule already exist? | **No.** |
| What file/commit/identity freezes it? | None — no such freeze exists anywhere audited. |
| Reserve candidate ordinal / recipe ordinal rule? | Undefined. |
| Reserve live-sample assignment rule? | Undefined. |
| Reserve GPAT spoof-partner assignment rule? | Undefined. |
| Reserve route/slot assignment rule? | Undefined. |
| Reserve same/cross-domain relation rule? | Undefined. |
| Reserve seed/hash rule? | Undefined. |
| Reserve `candidate_id` construction rule? | Undefined. |
| Guarantees identical scheduling across RND/DET/LLM? | N/A for reserve positions — no rule exists to evaluate. The **existing** frozen initial-2048 schedule guarantees this today, via `c5_arm_plan.assert_arms_share_the_schedule`; any future reserve rule **must** also satisfy this exact invariant before ratification. |
| Guarantees no target access? | N/A for reserve positions — no rule exists to evaluate. The existing guarantee rests on `live_for_position`/`spoof_for_position`/`load_source_rows` reading `source_train` only; any future reserve rule must reuse those exact functions unmodified to inherit it. |
| Can it extend deterministically through tranche 1..4 without changing the frozen initial 2048? | Cannot answer as a fact — no rule exists. This is exactly the property the draft proposal below is designed, but not yet proven or ratified, to satisfy. |

## 5. Status: `PROTOCOL_AMENDMENT_REQUIRED`

No reserve candidate was invented, computed or executed in this task. No frozen synthesis primitive, GPAT checkpoint/lock, quality calibration artifact, Flow1/Flow2 artifact, or existing candidate bank/generation closure was modified, deleted, or re-run.

## 6. Draft protocol-amendment proposal — **NOT RATIFIED, NOT IMPLEMENTED, NOT EXECUTED**

**Framing.** This proposal is drafted **after** observing, under the existing source-only/target-blind protocol, that all three folds' initial frozen 2048-candidate pass fell short of the 512-accepted floor on at least one route, and **before** any reserve candidate is rendered. It is a genuine **mid-study protocol amendment** and must always be cited as such. It is explicitly **not** part of, and must never be described or recorded as, the E0 preregistration freeze — the trigger condition (which folds/routes would fall short) was not and could not have been known at E0.

**Hard requirements this draft is designed to satisfy:**
- global and outcome-independent: one identical deterministic rule for every fold, every arm, every route, every tranche — no branch reads an accepted count, a `q` value, a per-arm pass rate, or any other observed outcome;
- never reads target images, target labels, or target metrics;
- never makes an outcome-adaptive recipe or source choice — every assignment formula is fixed before substitution, mirroring how the initial schedule is already constructed;
- preserves positions `0..2047` and everything derived from them (identity, bytes, files already on disk) completely unchanged — reserve positions occupy a new, disjoint, strictly-higher numbering range, never a renumbering of existing slots;
- synchronized `+256`/route/all-arm tranches exactly as spec §6.1 requires, capped at 2048 candidates/route/arm;
- defines complete candidate membership (recipe ordinal, live sample, spoof partner, route, domain relation, candidate identity) for every reserve position **before** any reserve rendering could begin.

**Recommended mechanism (draft).** Extend `c5_source_pair_plan`'s existing frozen, position-keyed functions (`live_for_position`, `eligible_spoof_sources`/`spoof_for_position`, `candidate_identity`) **verbatim** into a new, disjoint position range. Nothing about *how* a position becomes a candidate is invented; only *which* new positions exist, and their `(route, recipe_ordinal, domain_relation)` triple, is proposed:

- **Position numbering** — for tranche `t` in `1..4`, one arm/fold's reserve positions are numbered `2048 + 512*(t-1) .. 2048 + 512*t - 1` (512 = 256/route × 2 routes) — strictly above, and disjoint from, the frozen `0..2047` range.
- **Route assignment** — within one tranche's 512 new positions, the first 256 (by position order) are Physics and the next 256 are GPAT — deterministic and identical for every arm/fold/tranche.
- **Recipe-ordinal assignment** — `recipe_ordinal = reserve_index mod 256`, where `reserve_index` is the position's 0-based rank *within its own route* across all tranches (`0..1023` at the `t=4` cap) — cycles back through the same frozen 256-recipe bank, at most 4 additional full cycles per route, the same shape the initial schedule already uses (4 Physics + 4 GPAT renders/recipe before any reserve tranche opens).
- **Domain-relation assignment (GPAT)** — continue the same alternating same/cross pattern the original 4 GPAT slots/recipe already use (2 same + 2 cross): `relation = SAME_DOMAIN if (reserve_index_within_route // 256) is even else CROSS_DOMAIN` — preserves the existing 50/50 balance, introduces no new ratio.
- **Live sample / spoof partner** — reuse `live_for_position`/`spoof_for_position` **verbatim**, unmodified, called with the reserve position's own (larger) position number — both are already pure functions of `(position, seed)` with no assumption that `position < 2048`.
- **Candidate identity** — reuse `candidate_identity()` **verbatim**, unmodified, over the reserve position's own tuple — a reserve candidate's identity is therefore automatically distinct from every `0..2047` candidate (different `position` input) and automatically identical in construction rule across RND/DET/LLM.
- **Fairness/firewall inheritance** — because every reserve position is produced by the same frozen, position-keyed functions that already prove fairness (`assert_arms_share_the_schedule`) and source-only access (`load_source_rows` reads `source_train` only), the extension inherits both guarantees *by construction* — subject to re-running the equivalent checks over the **extended** schedule as an explicit acceptance gate before any reserve rendering.

**Explicitly open decisions requiring user ratification (not pre-decided by this draft):**
1. Block route assignment (256 Physics then 256 GPAT) vs. an alternating slot pattern mirroring the original per-recipe 8-slot cycle — both are equally deterministic and outcome-independent; this draft recommends the simpler block form only, not as a foregone conclusion.
2. Whether to add new frozen constants inside `c5_source_pair_plan.py` itself (a frozen-primitive **edit**, requiring explicit sign-off under CLAUDE.md) versus implementing this as a wholly separate additive module that only *imports* the existing frozen functions with zero primitive edits — this draft recommends the additive-module path specifically so no frozen synthesis primitive is ever touched.
3. The exact acceptance test suite that must pass on the extended schedule before reserve rendering is authorized — this draft recommends re-running the cardinality assertions, the cross-arm fairness assertion, and a fresh source-only audit over the full extended position range, all strictly before any reserve GPU rendering begins.

**Not yet decided.** This entire section is a draft for the user/spec-owner to ratify, amend, or reject. No code in this task implements it, and no reserve candidate exists anywhere because of it. If ratified, the resulting frozen rule must be committed as an explicit, separate, additive change (e.g. a new `c5_reserve_schedule.py` module — never an edit to `c5_source_pair_plan.py`/`c5_arm_plan.py`/`c5_render.py`) with its own identity, before any reserve candidate is rendered — and the ratification record must state the exact date/commit and the fact that it was frozen *after* observing the reserve trigger and *before* any reserve rendering, never as an E0 preregistration.

## 7. Constraints observed this task

- No reserve candidate invented or executed.
- No frozen synthesis primitive modified.
- No existing generation artifact modified or re-run.
- No GPAT checkpoint or lock modified.
- No quality calibration modified.
- No Flow1/Flow2 artifact modified.
- No candidate bank or generation closure modified.
- `target_access = false`, `llm_api_calls = 0`.
