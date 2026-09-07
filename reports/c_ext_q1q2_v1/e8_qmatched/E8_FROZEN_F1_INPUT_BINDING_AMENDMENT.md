# E8 Frozen-F1 Input-Binding Amendment (Additive Clarification)

**Classification:** `E8_FROZEN_F1_INPUT_BINDING_CLARIFICATION` — additive protocol clarification, not a spec change.
**Governing spec:** `/home/cong/PRISM_FAS_C_EXT_Q1Q2_Detailed_Spec_v1_0.docx`, sha256 `692291d67ffa6a4877202dd18f586bee7dc138e7d74c198840cad982055dedb2` — **not modified**.
**Current implementation commit:** `86c32a2a870f4ceda47c26de30a161eb19550219`

## Disclosure

- Created **after** observing E7-v1.1 source-only fold-specific bank infeasibility (0/3 folds, `SCIENTIFICALLY_BLOCKED_AFTER_RESERVE_CAP`).
- Created **before** E8 q-matched membership selection, before E8 detector training, before E8 target inference/scoring.
- **Not** claimed as an E0 preregistration.
- **Not** an E7 rescue. Does **not** reopen E7.

**E7 remains: 0/3 valid core matched synthetic banks** (EXT-F1/F2/F3 all `SCIENTIFICALLY_BLOCKED_AFTER_RESERVE_CAP`). E8 uses a *different*, pre-existing frozen-F1 synthetic population solely because that exact population is the one on which the pre-specified E2 q-confound trigger was observed.

## Terminology (claim boundary)

Use only: **"E8 frozen-F1 q-matched ablation"**. Never: "E7 reserve bank", "E7 rescue bank", "E7 matched-bank continuation", or "fold-local E7-v1.1 q-match".

## 1. Frozen-F1 compatibility audit (A–F)

| Item | Result | Evidence |
|---|---|---|
| A. Historical Flow-1/C6 source domains = CASIA+MSU, held out SiW-Mv2 | **VERIFIED** | `C6_MATCHED_BANKS.json` exposure `by_source_domain` = `{casia_fasd, msu_mfsd}` only, every arm/route; no `siw_mv2` key anywhere |
| B. EXT-F1 = CASIA+MSU → SiW-Mv2 | **VERIFIED** | EXT-F1 `domain_pair` binding and `FOLD_SOURCE_DATASET_SLUGS['EXT-F1']` = `(casia_fasd, msu_mfsd)` — exact match to (A) |
| C. EXT-F1 reuses frozen Version-C CASIA/MSU construction | **VERIFIED** | `c_ext_e7_gpat_bank.py` docstring: "every scientific primitive below is REUSED VERBATIM"; `build_effective_fold_quality_calibration_binding()`'s own note: the frozen `quality_gate_m8.yaml` calibration population "describe[s] EXT-F1's historical population" — i.e. it *is* the same population |
| D. Historical C6 bank is source-only, no target samples | **VERIFIED** | Same exposure evidence as (A): zero `siw_mv2` samples in any of the 3072 selected candidates |
| E. Historical C6 q reconstruction is target-label blind | **VERIFIED** | `C6_Q_RECONSTRUCTION_LOCK.json`: `target_labels_accessed=false`, `gpu_used=false`, `llm_api_calls=0`, `historical_consistency.status=CONSISTENT` (0 disagreements across 3072 rows) |
| F. Frozen Track-G binding compatible with EXT-F1 protocol | **VERIFIED** | E0's `c7_winner_config_sha256` matches the leaderboard/selected winner config in `reports/full/c7/C7_FULL.json`/`DETECTOR_CONFIG_LOCK.json` — the same frozen config that originally trained Version-C Track-G using (among its inputs) this exact historical bank |

**Overall: ALL VERIFIED.** The amendment is not blocked on frozen-F1 compatibility.

## 2. Resolved input artifacts

| Artifact | Path | SHA256 |
|---|---|---|
| Per-sample q table | `reports/c_ext_q1q2_v1/e2_quality/reconstructed_q/C6_Q_RECONSTRUCTED.parquet` | `87fdc8ea594a487bfef5206c5a0f0c763c7d19451a5f915e67e2237fd7431cd9` |
| Reconstruction lock | `reports/c_ext_q1q2_v1/e2_quality/reconstructed_q/C6_Q_RECONSTRUCTION_LOCK.json` | `c2b5cd55fc2fc9f0662580d6cb64f8c8fb8ae05d67bcbe904b14e7092a8e08c6` (lock_identity `9fc202cb1c...`) |
| Matched-bank record | `reports/full/c6/C6_MATCHED_BANKS.json` | `7a134b6c125f5eb30d8f51932d844363e683fec45ba9a28c0bb52874d3241293` |
| Profile-selection lock | `reports/full/c6/C6_PROFILE_SELECTION_LOCK.json` | `9bb5d4fcac97c630d05f239be43e8322b0d08f791c79bde0abe06ba6983907fc` |
| Bank lock (RND/DET/LLM) | `reports/full/c6/C6_BANK_LOCK_{RND,DET,LLM}.json` | `451964130e...` / `bacd544d7d...` / `ff57043ac3...` |

Per-sample table: 3072 rows, columns `candidate_id, arm, route, historical_passed, historical_selected, q, source_artifact_identity, reconstruction_method`. `candidate_id` serves as the E8 stable-hash `sample_id`. Every row has `historical_passed=True` **and** `historical_selected=True` — this is already the final selected 1024/arm bank (512 physics + 512 gpat); "accepted" for this population coincides exactly with "selected".

**Quality-gate config filename:** `QUALITY_GATE_CONFIG_FILENAME_UNRESOLVED_BUT_THRESHOLD_IDENTITY_BOUND`. Three `quality_gate_m8*.yaml` files exist; E0's own `NEEDS_CONFIRMATION` flag on which one produced threshold identity `8fa2648643cd526730497ae2d717e17684dda3ecea361fc84929db07ac03bb19` was never resolved in any persisted E2 artifact. Per instruction, this filename ambiguity alone does **not** block E8: the effective threshold identity and `NOMINAL` profile semantics are uniquely bound, cryptographically identified, and reproducible independent of the literal filename. No filename was guessed.

**Image/crop data locator:** not resolved by this amendment. The parquet carries `candidate_id`/`source_artifact_identity` — sufficient for q-matched *membership* selection — but not the actual image/crop bytes (these live in the frozen Flow-1 C6 package artifacts on the GPU host). Resolving them is a prerequisite for E8 *training*, deferred to the (not-yet-written) selector/training implementation.

## 3. Quality-profile binding: `NOMINAL`

Confirmed: E0 froze `NOMINAL` before any target result; E2 actually reconstructed q under `NOMINAL`; the frozen F1 bank that fired the E8 trigger was the `NOMINAL` final-selected bank. **`PERMISSIVE` is excluded** — it was an E7-v1.1 reserve-feasibility fallback for a different scientific purpose and is not retroactively imported into E8.

## 4. Amendment scope

Applies **only** to E8's primary deployment: **EXT-F1 × 5 detector seeds**. The E8 source population is the exact historical frozen F1/C6 NOMINAL final selected synthetic population that produced the E2 trigger. This binding does **not** automatically extend to EXT-F2/EXT-F3 — no equivalent historical frozen bank exists for those folds (they introduce SiW-Mv2 as a *source* domain, which the original Version-C pipeline never did). If E8 later materially changes ranking and F2/F3 need consideration, that requires a separate input-binding decision.

## 5. Q-match rule (frozen exactly)

Routes: `physics`, `gpat`, handled separately.

q-bins: `[0.0,0.1), [0.1,0.2), [0.2,0.3), [0.3,0.4), [0.4,0.5), [0.5,0.6), [0.6,0.7), [0.7,0.8), [0.8,0.9), [0.9,1.0]`

**q = 1.0 rule (explicit amendment clarification):** include q==1.0 in the final closed bin `[0.9,1.0]` — never dropped. Recorded explicitly because the original spec's interval notation is underspecified at the right endpoint for every bin except the last, which the spec itself already writes closed.

`n_b = min(accepted_count_RND, accepted_count_DET, accepted_count_LLM)` per route × bin. Selection ordering: stable hash of `sample_id || 'QMATCH-v1'`; select exactly `n_b` per arm/route/bin. No quality score enters the tie-break beyond bin membership. No target information enters selection.

## 6. `e8_input_binding_rule_identity`

```
8223df2d5acb38483b7cfe94bbd0a293489a9200b20fd309fb679ea6621cc7d8
```

SHA256 of the canonical JSON payload containing only frozen scientific decisions (`input_population_identity`, `profile`, `profile_identity`, `quality_threshold_identity`, `routes`, `q_bins`, `q_equals_one_rule`, `min_count_rule`, `stable_hash_rule`, `primary_fold`, `training_seeds`, `target_firewall_rule`) — see the companion JSON's `e8_input_binding_rule_identity_material` for the exact material hashed. This identity is frozen **before** any membership selection.

## 7. Dry-run feasibility (READ-ONLY — no bank membership written)

### Physics

| q-bin | RND | DET | LLM | n_b | limiting arm(s) |
|---|---|---|---|---|---|
| [0.0,0.1) | 0 | 2 | 0 | **0** | RND, LLM |
| [0.1,0.2) | 0 | 0 | 0 | **0** | RND, DET, LLM |
| [0.2,0.3) | 0 | 0 | 4 | **0** | RND, DET |
| [0.3,0.4) | 3 | 4 | 11 | **3** | RND |
| [0.4,0.5) | 24 | 12 | 23 | **12** | DET |
| [0.5,0.6) | 30 | 38 | 74 | **30** | RND |
| [0.6,0.7) | 80 | 72 | 134 | **72** | DET |
| [0.7,0.8) | 161 | 136 | 165 | **136** | DET |
| [0.8,0.9) | 168 | 190 | 98 | **98** | LLM |
| [0.9,1.0] | 46 | 58 | 3 | **3** | LLM |

### GPAT

| q-bin | RND | DET | LLM | n_b | limiting arm(s) |
|---|---|---|---|---|---|
| [0.0,0.1) | 0 | 0 | 0 | **0** | RND, DET, LLM |
| [0.1,0.2) | 0 | 0 | 1 | **0** | RND, DET |
| [0.2,0.3) | 0 | 0 | 2 | **0** | RND, DET |
| [0.3,0.4) | 2 | 4 | 7 | **2** | RND |
| [0.4,0.5) | 17 | 19 | 22 | **17** | RND |
| [0.5,0.6) | 43 | 33 | 43 | **33** | DET |
| [0.6,0.7) | 95 | 115 | 107 | **95** | RND |
| [0.7,0.8) | 212 | 178 | 182 | **178** | DET |
| [0.8,0.9) | 134 | 152 | 143 | **134** | RND |
| [0.9,1.0] | 9 | 11 | 5 | **5** | LLM |

| Metric | Value |
|---|---|
| Physics common-support total (Σ n_b) | **354** |
| GPAT common-support total (Σ n_b) | **464** |
| Total unique E8 bank size per arm | **818** |
| Empty bins | physics:bin0, physics:bin1, physics:bin2, gpat:bin0, gpat:bin1, gpat:bin2 |
| q == 1.0 count | 0 |
| min(q) / max(q) | 0.085553 / 0.941574 |
| Source-domain composition | not available at per-sample/per-bin granularity (no source-domain column in the reconstructed parquet); whole-bank aggregate = `{casia_fasd, msu_mfsd}` only (see compatibility A/D) |

This is **read-only**. No final selected-sample membership was written.

## 8. Data-integrity rules observed

`target_access=false`, `target_labels_accessed=false`, `llm_api_calls=0`, no GPU run, no new candidate generation, no threshold relaxation, no E7 gate change, no candidate regeneration, no arm-specific quality threshold, no change to q, no final bank membership selected, no detector training performed, no scientific historical artifact modified, no scientific code modified.

## 9. Status

**`RATIFIED_READY_FOR_E8_SELECTOR_IMPLEMENTATION`**
