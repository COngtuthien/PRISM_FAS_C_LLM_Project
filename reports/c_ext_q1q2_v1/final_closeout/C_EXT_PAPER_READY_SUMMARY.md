# PRISM-FAS-C EXT-Q1Q2 -- paper-ready scientific summary (final closeout)

_Read-only evidence synthesis at the E10 scientific-closure commit `7723fc1444cd69a5c4db1fe73c4ee4edf96313dd`. No experiment was run; no LLM / provider / API call was made; no detector was trained; no synthetic data was generated; no raw target image, feature or label was accessed._

## What was attempted

An additive Q1/Q2 evidence-strengthening extension to Version C, executed under the frozen **Q2-NoLLM** profile (`ALLOW_LLM_API = FALSE` for the whole E0-E10 core). The pre-registered primary family (frozen at E0) is:

- **EXT-H1** LLM vs RND, **EXT-H2** LLM vs DET, **EXT-H3** LLM vs REAL-ONLY, **EXT-H4** LLM-original vs LLM-SHUFFLE-A;
- primary endpoint: video-level ACER by fold/seed on the held-out target SiW-Mv2, **conditional on the frozen recipe banks**;
- Holm-Bonferroni, alpha = 0.05;
- E0 claim ceiling: **no general 'LLM generation mechanism is superior' claim without E11.**

The intended mechanism test was a unified 3-fold cross-domain Track-G detector experiment (E7) with matched RND/DET/LLM/LLM-SHUFFLE-A synthetic banks, plus REAL-ONLY (E5), a recipe-bank structural analysis (E1), a q-confound ablation (E8), a bank-selection sensitivity check (E9), BA-separation negative controls (E3) and a threshold-transfer diagnostic (E4); then the E10 decision gate.

## What completed

- **E0** protocol / hypothesis-family / binding freeze -- COMPLETE and locked.
- **E1** recipe-bank structural analysis -- COMPLETE (CPU, descriptive). All 12 structural metrics generated for the 3 frozen 256-recipe banks (status `COMPLETE_CPU`, 0 LLM calls, no target access). The LLM bank is structurally distinguishable from RND/DET at the recipe level: larger cross-arm Jensen-Shannon divergence on the joint artifact x severity co-occurrence (RND-vs-LLM 0.1868, DET-vs-LLM 0.2103 vs RND-vs-DET 0.0318) and lower within-bank pairwise Gower dispersion (LLM mean 0.4271 vs RND 0.5163 / DET 0.5163). **Structural only** -- per E1's own interpretation boundary this does not measure semantic plausibility or a causal role of LLM reasoning, and is a mechanism signal only if the (blocked) E6 LLM-SHUFFLE ablation reduces downstream performance. E1 is a diagnostic, not in the Holm family, and is not consulted by any E10 gate criterion.
- **E2** quality-distribution reconstruction -- COMPLETE. A real quality confound is present: SMD(q) LLM-vs-RND = -0.3258, LLM-vs-DET = -0.3640 (both |SMD| >= 0.25; RND-vs-DET = -0.0462). This fired the frozen E8 trigger.
- **E5** REAL-ONLY target evaluation -- COMPLETED and scored for **EXT-F1 only**: mean target ACER 0.1999 (sample SD 0.0046, 5 detector seeds), mean ROC-AUC 0.8850. Frozen source-dev calibration, no target threshold fitted.
- **E8** q-matched Track-G target evaluation -- COMPLETED and scored for **EXT-F1 only** (arms RND/DET/LLM, 5 detector seeds/arm, 15 runs). Authoritative descriptive mean target ACER: **LLM = 0.3487 < DET = 0.3506 < RND = 0.3516** (mean ROC-AUC LLM 0.7071, DET 0.7086, RND 0.7051). Margins are ~0.002 ACER. **Descriptive only:** inferential_status = NOT_CLAIMED, no statistical-significance claim, no LLM-superiority claim, no comparable pre-q-match target baseline exists.
- **E9** conditional bank-selection robustness -- CLOSED. 9/9 deterministic candidate-pool perturbation realizations (RND/DET/LLM x masks P1/P2/P3, 0 blocked). With 64 of 384 frozen candidates masked, the 384->256 selection keeps a large descriptive Jaccard overlap with the reference selection (per-arm means RND 0.6917, DET 0.7032, LLM 0.6256). **Structural selection stability only** -- 0 LLM calls, no target access.
- **E10** decision gate -- CLOSED. Criteria: C1 BLOCKED, C2 NOT_ESTABLISHED, C3 BLOCKED, C4 PASS (descriptive only), C5 PASS (with limitations). **E11 NOT authorized; new LLM calls NOT authorized; Q1 mechanism claim NOT supported.**

## What was blocked

- **E7 (primary mechanism test) -- SCIENTIFICALLY BLOCKED.** All three folds terminated `SCIENTIFICALLY_BLOCKED_AFTER_RESERVE_CAP` with **0/3** valid core matched synthetic banks. The binding constraint was the LLM-Physics accepted count in every fold (EXT-F1 295, EXT-F2 467, EXT-F3 472) against the frozen common 512-Physics quota (GPAT was feasible 512/512 in all folds). E7 terminated **before any target ACER scoring**; G-RND/G-DET/G-LLM/G-LLM-SHUFFLE-A are `BLOCKED_BY_E7_CORE_BANK`. This is a source-only, target-blind generation/quality-gate/matched-bank **feasibility** result under the frozen protocol -- `is_target_acer_evidence = false`, `is_evidence_llm_recipes_generally_inferior = false`.
- **E6 (paired original / LLM-SHUFFLE-A rerender) -- CLOSED, matched bank infeasible.** The frozen SHUFFLE-A matched Physics bank fills only 479/512 under the frozen EXACT per-source-domain quota (CASIA deficit 33, MSU surplus 33, not fungible). No detector trained; no semantic-ablation target ACER.
- **E3 discriminating BA controls (C0/C1/C3) and C2** -- GPU-gated / unsupported; `delta_BA_excess` not computable.
- **E4 EXT-F2/EXT-F3 threshold transfer** -- GPU-required; only the EXT-F1 re-analysis of the frozen historical Flow-1 evaluation exists.
- **E11** -- NOT_AUTHORIZED_NOT_RUN (gate did not pass).

## Valid quantitative results

All numbers below are descriptive; none carries an inferential or significance claim. Full rows: `C_EXT_FINAL_RESULTS_TABLE.csv`.

| Evidence class | Fold | Quantity | Value |
|---|---|---|---|
| E1 recipe structural (descriptive) | frozen-256 bank | within-bank Gower dispersion mean LLM / DET / RND | 0.4271 / 0.5163 / 0.5163 |
| E1 recipe structural (descriptive) | frozen-256 bank | JS divergence artifact x severity: RND-LLM / DET-LLM / RND-DET | 0.1868 / 0.2103 / 0.0318 |
| E2 quality confound | frozen-F1 bank | SMD(q) LLM-vs-RND / LLM-vs-DET | -0.3258 / -0.3640 |
| E5 REAL-ONLY (completed, partial) | EXT-F1 | mean target ACER (5 seeds) | 0.1999 |
| E8 q-matched (descriptive) | EXT-F1 | mean target ACER LLM / DET / RND | 0.3487 / 0.3506 / 0.3516 |
| E9 bank selection (structural) | frozen-384 pool | mean Jaccard vs reference RND / DET / LLM | 0.6917 / 0.7032 / 0.6256 |
| E7 3-fold G-LLM/G-DET/G-RND/G-SHUFFLE target ACER | EXT-F1/F2/F3 | mean target ACER | UNAVAILABLE (upstream-blocked) |

## Limitations

- The primary mechanism test (E7) produced **no target evidence**: the matched-bank construction was infeasible under the frozen quota in every fold. EXT-H1, EXT-H2 and EXT-H4 are therefore untestable under this frozen protocol.
- Every completed target evaluation (E5, E8) is **EXT-F1 only** and **5 detector seeds**; effect sizes in E8 are ~0.002 ACER with per-seed SD an order of magnitude larger.
- E8 is a q-matched ablation with **no comparable pre-q-match target baseline**, so a ranking change cannot be assessed.
- No inferential procedure for multi-detector-seed arm-level comparison existed in the tracked frozen repository before target results, so **no significance is claimed anywhere**.
- E9 is **structural selection stability**, not detector performance and not independent generation replication.
- E3's discriminating BA controls are GPU-gated, so whether the synthetic/real separation is a generator fingerprint or a domain/pipeline confound is **not affirmatively resolved**.
- E4's operating-point diagnostic covers EXT-F1 only; T3-oracle is diagnostic-only and never a method.

## Claims that ARE allowed

- E1 shows the LLM recipe bank is **structurally distinguishable** from RND/DET at the recipe level (cross-arm JS divergence, lower within-bank Gower dispersion); this is a descriptive structural fact only and is **not** evidence of a semantic mechanism -- that would need the blocked E6 Shuffle ablation.
- E7 was **blocked by source-only matched-bank feasibility before target evaluation**; it is not evidence about LLM recipe quality either way.
- E8 provides **descriptive EXT-F1 q-matched evidence only**: after q-matching, the LLM arm has the lowest mean target ACER in that single fold, with no significance and no superiority claim.
- After q-matching, the descriptive LLM ordering **does not reverse** in EXT-F1 (the E10 C4 PASS, descriptive-only).
- E9 shows the frozen 384->256 recipe selection is **descriptively stable** to which exact 256-recipe subset is drawn from the frozen 384-candidate pools, under three deterministic perturbation masks.
- No bug or target-label leakage that would invalidate the completed EXT-F1 target evaluations was detected (E10 C5 PASS, with GPU-gated controls listed as limitations).
- **E10 did not authorize E11**; unavailable evidence is not negative evidence.

## Claims that must NOT be made

- “The LLM is superior overall” / “LLM generation generally outperforms RND/DET”.
- “The LLM mechanism is proven” / “the Q1 mechanism claim is supported”.
- “E7 shows the LLM recipes failed / are inferior”, or any reading of the E7 block as a negative LLM result.
- “E9 proves detector robustness” or “E9 shows robust LLM detector performance”.
- “E9 is an independent LLM-generation replication”.
- “LLM synthetic data beats (or loses to) REAL-ONLY”.
- “The Shuffle / semantic-coupling ablation shows an effect” (or that it reversed).
- “Statistically significant” anywhere -- no authoritative artifact supports it.

## Bottom line

The EXT-Q1Q2 extension is **CLOSED with the primary mechanism evidence blocked**. The frozen matched-bank protocol could not construct the E7 cross-domain banks in any fold, so the pre-registered LLM-vs-RND/DET/SHUFFLE mechanism comparison was never evaluated on the target. The completed descriptive work (E5, E8, E9) does not, and under the E0 claim ceiling cannot, support a general LLM-superiority or LLM-mechanism claim. E10 correctly declined to authorize another LLM call (E11).
