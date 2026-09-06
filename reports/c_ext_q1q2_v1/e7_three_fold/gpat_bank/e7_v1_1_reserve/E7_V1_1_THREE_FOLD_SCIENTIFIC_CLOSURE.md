# E7-v1.1 Three-Fold Scientific and Provenance Closure

**Scope:** reporting/provenance only. No experiment was run, no candidate was rendered, no GPAT/detector was trained, no LLM was called, and no EXT-F1/EXT-F2/EXT-F3 run artifact was reopened or modified to produce this document.

**Reserve schedule rule identity:** `7871404603876a7b015af80cefaca26d94d54f102a9eb445099254aa3bbd1822`
**EXT-F1 scientific execution commit:** `db87b1c721da99d89d6b9439418e94ab46e68767`
**EXT-F2 / EXT-F3 execution commit:** `fae929616219fad083a2ce6ca05fbf7d25bd0157`
**Bookkeeping-correction commit:** `fae929616219fad083a2ce6ca05fbf7d25bd0157`
**EXT-F1 is never retroactively attributed to the bookkeeping-fix commit** — confirmed directly from persisted evidence: every one of EXT-F1's four `RESERVE_TRANCHE0N_LOCK.json` files binds `implementation_commit=db87b1c721da99d89d6b9439418e94ab46e68767`; every one of EXT-F2's and EXT-F3's binds `implementation_commit=fae929616219fad083a2ce6ca05fbf7d25bd0157`.
**Master evidence manifest:** `gpu_evidence/e7_v1_1_reserve/E7_V1_1_THREE_FOLD_TERMINAL_EVIDENCE.sha256` — verified against the persisted files with `sha256sum -c --quiet`: **all hashes match, zero mismatches.**

## A. Protocol history (in order)

1. E7-v1.0's frozen initial candidate pools (2048/route/arm) were executed once for all three folds.
2. All three folds failed to form the required matched banks from the initial pool.
3. Repository audit found no authoritative reserve membership/schedule (`RESERVE_SCHEDULE_AUTHORITY_AUDIT.md`).
4. Status: `BLOCKED_PROTOCOL_GAP_RESERVE_SCHEDULE` (EXT-F1/F2/F3, G-RND/G-DET/G-LLM).
5. E7-v1.1 RESERVE AMENDMENT was frozen after observing this source-only feasibility failure and **before** any reserve rendering.
6. Reserve membership was deterministic and outcome-independent (position/tranche/route/recipe-ordinal formulas fixed before rendering; see `E7_V1_1_RESERVE_AMENDMENT.md`).
7. Four synchronized `+256`-candidate-per-route reserve tranches were permitted, per §6.1.
8. Maximum cumulative pool: 2048 candidates/route/arm (4096 total/arm).
9. Gate thresholds (STRICT/NOMINAL/PERMISSIVE), route quotas, the matched-bank selector, candidate-ID construction, and the target firewall were never adapted from observed outcomes at any tranche.
10. EXT-F1's terminal tranche-4 run (commit `db87b1c...`) revealed a post-run `RESERVE_RUN_STATE.json.cumulative_counts` metadata bug (stale `1792/1792` instead of `2048/2048`).
11. The metadata-only fix was committed as `fae9296...` (`fix(c-ext): correct E7 terminal reserve counts`).
12. EXT-F1 remains scientifically attributable to `db87b1c...` — never reattributed.
13. EXT-F2 and EXT-F3 executed under `fae9296...` and their own `RESERVE_RUN_STATE_AFTER_T4.json` confirm correct terminal `2048/2048` bookkeeping directly (verified below).

## B. Per-fold tranche progression (PERMISSIVE, Physics common quota / 512)

All figures below are read directly from each fold's `EXT_F*_RESERVE_TRANCHE{1..4}_CLOSURE.json` and cross-checked against the values supplied for this task — they match exactly.

| Fold | T1 | T2 | T3 | T4 (terminal) |
|---|---|---|---|---|
| EXT-F1 | 182/512 (shortfall 330) | 218/512 (shortfall 294) | 256/512 (shortfall 256) | **295/512 (shortfall 217)** |
| EXT-F2 | 304/512 (shortfall 208) | 360/512 (shortfall 152) | 413/512 (shortfall 99) | **467/512 (shortfall 45)** |
| EXT-F3 | 301/512 (shortfall 211) | 363/512 (shortfall 149) | 415/512 (shortfall 97) | **472/512 (shortfall 40)** |

GPAT reached `512/512` (shortfall 0) at **every** tranche, in every fold, including T1 — GPAT was never the constraining route at any point in the reserve sequence.

**Note on comparability:** the T1–T4 figures above are all *cumulative reserve* assessments (v1.0's frozen 2048/arm pool plus the stated number of reserve tranches), computed under the identical frozen STRICT→NOMINAL→PERMISSIVE procedure and the identical common-domain-quota selector at every step — so the T1→T4 progression within one fold is a fair, monotonic comparison. The user-supplied "initial pool" figure for EXT-F1 (~483/512 "before reserve") is **not** included as a comparable data point in this table: it was not located as a persisted PERMISSIVE-profile assessment artifact from the v1.0 initial-pool run in the evidence read for this closure, and the v1.0 initial assessment's own selection/calibration context (computed before the reserve amendment existed) is not asserted here to be directly comparable to the v1.1 cumulative reserve assessments without that artifact in hand. This closure deliberately does not imply monotonicity across the v1.0→v1.1 boundary; it reports only the four cumulative reserve tranches, which are internally consistent and directly comparable to one another.

## C. Final per-arm PERMISSIVE route counts (from each fold's authoritative `RESERVE_FINAL_CLOSURE.json`)

| Fold | RND Physics | DET Physics | LLM Physics | RND GPAT | DET GPAT | LLM GPAT |
|---|---|---|---|---|---|---|
| EXT-F1 | 783 | 774 | **295** | 1155 | 1105 | 968 |
| EXT-F2 | 1014 | 964 | **467** | 1071 | 979 | 1098 |
| EXT-F3 | 1027 | 965 | **472** | 1174 | 998 | 952 |

In every fold, **LLM has the fewest accepted Physics candidates of the three arms**, and since the frozen matched-bank selector (`c6_matched_bank.build_matched_banks`) requires one identical common-domain-quota vector shared by RND/DET/LLM, LLM's own Physics acceptance count is the binding constraint on the common Physics quota in all three folds (Physics common quota = min-constrained value: F1 quota_total=295 = LLM's own 295; F2 quota_total=467 = LLM's own 467; F3 quota_total=472 = LLM's own 472 — an exact match in every fold, confirming LLM-Physics as the limiting arm-route).

## D. Final scientific interpretation

- **0 / 3** folds produced a valid core matched synthetic bank.
- All 3 folds exhausted the frozen reserve cap (tranche 4, 2048 candidates/route/arm).
- All 3 folds terminated **`SCIENTIFICALLY_BLOCKED_AFTER_RESERVE_CAP`** (confirmed verbatim in each fold's `RESERVE_FINAL_CLOSURE.json.status`/`.final_status`).
- The common failure mode was the **Physics** route in all three folds; GPAT was feasible at **512/512** in the final PERMISSIVE assessment for every fold (shortfall 0, confirmed for all three).
- The limiting arm-route in the observed final folds was **LLM-Physics** (§C above).
- **This is a source-only feasibility result** under the frozen v1.0+v1.1 generation/gating/matching protocol.
- **It is NOT target ACER evidence** — no detector was trained, no target label or image was ever accessed (`target_access=false` in every closure).
- **It is NOT evidence that LLM recipes are generally inferior** — it reflects only how many of LLM's own generated Physics candidates survived the frozen, unmodified 8-criterion quality gate at PERMISSIVE, under this specific frozen recipe bank, generator, and threshold configuration.
- It only shows that, under this frozen generation/gating/matching protocol, the required common 512-Physics quota could not be constructed in any of the three folds.

## E. Integrity / leakage statement

- `target_access = false` in every fold's every closure.
- `llm_api_calls = 0` in every fold's every closure.
- No quality threshold was lowered at any tranche or fold.
- No arm-specific threshold was introduced.
- No target distribution was used to rescue feasibility.
- No selector was changed.
- No post-cap (tranche > 4) candidate was generated in any fold.
- No fold was rerun after its own terminal scientific closure was written.

## F. Recovery evidence (`mask_compatibility_recovery_count`, read from each fold's `RESERVE_FINAL_CLOSURE.json`)

| Fold | mask_compatibility_recovery_count |
|---|---|
| EXT-F1 | **0** |
| EXT-F2 | **0** |
| EXT-F3 | **1** |

All three values are read directly from persisted evidence, not assumed. The count reflects the frozen `e7-empty-cheek-crop-boundary-recovery-v1` mask-compatibility policy activating during rendering/evaluation; it is unrelated to reserve-schedule membership or matched-bank feasibility and is recorded here purely as provenance, never hidden.

## G. Downstream consequences

**Revision note (this section only):** this table corrects an earlier over-strong classification. It was re-derived directly from spec sections 14–18 and Appendix D (execution-order appendix, Q2-NoLLM/Q1-MultiBank execution profiles), not from E7 implementation convenience. Sections A–F above and the F1/F2/F3 numerical results and provenance bindings they report are **unchanged** by this revision. The E7 closure remains **0/3 valid core synthetic matched banks**; nothing below weakens that result. E8 is a different, explicitly pre-specified ablation with its own bank-construction rule — an E8 bank must never be called an E7 rescue bank, and it does not rewrite E7's `SCIENTIFICALLY_BLOCKED_AFTER_RESERVE_CAP` result.

| Experiment | Disposition | Basis |
|---|---|---|
| **E6 LLM-SHUFFLE-A** | `BLOCKED_BY_E7_CORE_BANK` (downstream detector comparison only) | The frozen fold-scoped Shuffle-A policy (`c_ext_e7_gpat_bank.py`; also recorded in `c_ext_e7_reserve_orchestrator.py`'s protocol lock `shuffle_policy` field) requires that fold's own RND/DET/LLM core anchor bank close *before* Shuffle-A candidate generation is even attempted. 0/3 folds closed their anchor bank, so Shuffle-A generation was never attempted for EXT-F2/EXT-F3; EXT-F1's own historical Shuffle-A infeasibility (`BLOCKED_TRUE_FROZEN_MATCHED_BANK_INFEASIBILITY`) remains separately frozen and is not rescued by this amendment. **Recipe-level Shuffle-A construction/audit is conceptually distinct** from this blocked downstream detector comparison and is not itself blocked. |
| **E7 detector experiment** | Per-condition (no single scalar) — see table below | A single experiment-wide status would misrepresent that REAL-ONLY does not require a synthetic bank. |
| **E8 q-matched detector experiment** | `READY` | Spec §15 defines its **own, separate** source-only bank-construction rule (bin q into 10 fixed bins per route; `n_b` = min accepted count across RND/DET/LLM per route/bin; select `n_b` per arm by stable hash `sample_id \|\| 'QMATCH-v1'`; concatenate; equal bank size on common q support; fixed optimizer updates via sampling with replacement) that does **not** require a successful E7 512-Physics+512-GPAT common matched bank. The spec's own pre-specified trigger (`\|SMD(q)\| >= 0.25`, frozen at E0) has already fired (`E7_E8_TRIGGER_RECORD.json`, `E8_TRIGGER_FROM_E2=true`). All three folds' PERMISSIVE accepted counts are nonzero for RND/DET/LLM on both routes, so `n_b > 0` is achievable. `READY` means the prescribed construction is executable now — it is **not** a guarantee the resulting bank will be scientifically adequate; realized unique-bank size and common q support must be reported when E8 is run. |
| **E9 candidate-pool perturbation** | `READY` — **conditional-on-need** | Spec §16 operates entirely on the frozen 384-candidate recipe pools and the MILP selector that picks 256 of them — a C3 recipe-bank-selection question, upstream of and independent from C5 rendering, quality gating, or C6 matched-bank construction. It consumes no E7 rendered-candidate or matched-bank artifact. Per Appendix D's Q2-NoLLM profile ("E8/E9 theo trigger/need"), E9 runs only if exact-256-bank sensitivity actually needs checking. The spec explicitly forbids calling this independent generator replication, and so does this closure. |
| **E10 decision gate** | `READY` | E10 is a **decision gate**, not a detector experiment — Appendix D's Q2-NoLLM profile explicitly includes E10 as a required step (no LLM API call), and its purpose is to decide whether E11 is justified. A gate can be evaluated, and can legitimately resolve to its negative branch, even when some inputs are unavailable. **Fail-closed rule:** because E7's core synthetic detector conditions are unavailable in every fold, three of E10's five required positive-evidence conditions (spec §17) cannot currently be demonstrated — cross-domain LLM detector ordering, LLM vs REAL-ONLY on ≥2/3 folds, and original vs Shuffle-A downstream detector result. E8 can address only the quality-confound condition and **cannot replace** the other three. When E10 is eventually executed, **absence of that evidence must not be interpreted as positive support for E11**; the gate fails closed per the spec's own fallback ("do not call the LLM again") unless the required evidence is actually available and satisfied. No substitute criteria are authorized. |
| **E11 independent multi-bank study** | `NOT_APPLICABLE_AFTER_E7_FAILURE` (preferred wording: `NOT_AUTHORIZED_UNDER_CURRENT_E10_EVIDENCE`) | E11 remains an optional experiment defined by spec §18; it is not authorized under the **current** frozen plan because E10's required positive core evidence (the three conditions above) cannot currently be demonstrated — E11 is explicitly conditioned on E10's gate being satisfied. This is **not** "scientifically impossible forever": launching E11 would require a separate protocol decision/amendment (with its own pre-API hash-locked subprotocol per §18) and must never be silently treated as already authorized by the original E10 Q1-mechanism pathway. No LLM API call is authorized by this closure. |

### E7 detector experiment — per-condition status

| Condition | Disposition | Basis |
|---|---|---|
| **G-REALONLY** | `READY` | Spec §14 table: recipe bank = "None", synthetic = "No" — consumes no E7 matched-bank artifact, structurally unaffected. |
| **G-RND** | `BLOCKED_BY_E7_CORE_BANK` | Requires that fold's own matched RND synthetic bank as training data; 0/3 folds produced one. |
| **G-DET** | `BLOCKED_BY_E7_CORE_BANK` | Requires that fold's own matched DET synthetic bank as training data; 0/3 folds produced one. |
| **G-LLM** | `BLOCKED_BY_E7_CORE_BANK` | Requires that fold's own matched LLM synthetic bank as training data; 0/3 folds produced one. |
| **G-LLM-SHUFFLE-A** | `BLOCKED_BY_E7_CORE_BANK` | Requires that fold's own matched Shuffle-A synthetic bank; Shuffle-A generation was never attempted (core anchor bank never closed). |

G-REALONLY alone does **not** answer RQ1/RQ2: the primary comparative hypothesis families EXT-H1 (LLM vs RND), EXT-H2 (LLM vs DET) and EXT-H4 (LLM-original vs LLM-SHUFFLE) (frozen in `c_ext_e0_freeze.py`) each require at least one `BLOCKED_BY_E7_CORE_BANK` condition and therefore cannot be tested in any fold under this protocol.

## Audit manifest

- Master manifest: `gpu_evidence/e7_v1_1_reserve/E7_V1_1_THREE_FOLD_TERMINAL_EVIDENCE.sha256` — header confirms `EXT-F1 scientific execution commit=db87b1c721da99d89d6b9439418e94ab46e68767`, `EXT-F2/EXT-F3 implementation commit=fae929616219fad083a2ce6ca05fbf7d25bd0157`, `reserve_rule=7871404603876a7b015af80cefaca26d94d54f102a9eb445099254aa3bbd1822`, and all three `F{1,2,3}=SCIENTIFICALLY_BLOCKED_AFTER_RESERVE_CAP` — **verified byte-for-byte against every listed file via `sha256sum -c --quiet`: zero mismatches.**
- Per-fold manifests: `EXT_F1_RESERVE_TERMINAL_EVIDENCE.sha256`, `EXT_F2_RESERVE_TERMINAL_EVIDENCE.sha256` / `EXT_F2_COMPLETE_RESERVE_AUDIT_TRAIL.sha256`, `EXT_F3_RESERVE_TERMINAL_EVIDENCE.sha256` / `EXT_F3_COMPLETE_RESERVE_AUDIT_TRAIL.sha256` — all present, all under the existing `gpu_evidence/e7_v1_1_reserve/` namespace, none modified by this task.

## Companion JSON

`E7_V1_1_THREE_FOLD_SCIENTIFIC_CLOSURE.json` (same directory) carries the same facts in structured form, including the full per-fold tranche-progression data, final PERMISSIVE `arm_route_counts`, provenance block, and the downstream-dependency table above.
