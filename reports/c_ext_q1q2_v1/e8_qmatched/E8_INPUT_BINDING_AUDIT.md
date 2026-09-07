# E8 Input-Binding Audit (Quality-Matched Synthetic-Bank Ablation)

**Classification:** audit / protocol-binding / dry-run-feasibility only. No detector was trained, no candidate was rendered, no GPU was run, no E7 evidence was modified, no LLM was called, and E7 was not reopened.
**Current implementation commit:** `86c32a2a870f4ceda47c26de30a161eb19550219`
**E7 status (referenced, not altered):** EXT-F1/F2/F3 = `SCIENTIFICALLY_BLOCKED_AFTER_RESERVE_CAP`, 0/3 core matched synthetic banks. E8 is a separate, pre-specified ablation and is never described here as an E7 rescue.

## 1. E8 trigger verification

| Item | Value |
|---|---|
| Source artifact | `reports/c_ext_q1q2_v1/e7_three_fold/E7_E8_TRIGGER_RECORD.json` |
| Frozen rule | `\|SMD(q)\| >= 0.25` between LLM and RND OR LLM and DET, on the final accepted q distribution, frozen at E0 |
| `E8_TRIGGER_FROM_E2` | `true` |
| Cross-checked against | `reports/c_ext_q1q2_v1/e2_quality/E2_QUALITY_ANALYSIS.json` |
| Observed SMD(LLM, RND) | **-0.3258** (abs ≥ 0.25 -> true) — matches the ≈0.326 expected value |
| Observed SMD(LLM, DET) | **-0.3640** (abs ≥ 0.25 -> true) — matches the ≈0.364 expected value |
| Observed SMD(RND, DET) | -0.0462 (abs ≥ 0.25 -> false) |
| n per arm in this computation | 1024 (3072 total) |
| Persisted `E8_REQUIRED` | `TRUE` |

Persisted artifacts are authoritative; the values above were read directly, not recomputed or overwritten. **The trigger has fired.**

## 2. Existing E8 implementation status: `ABSENT`

A repository-wide search for `E8`, `qmatched`, `q_match`, `QMATCH-v1`, `quality matched`, `quality_matched` under `src/prism_fas/evaluation/` and `tests/pipeline/` finds only incidental word-matches (e.g. `E8` appearing in `c_ext_e0_freeze.py`'s `milestone_order` list). `QMATCH-v1` as a literal string exists only inside the E7-v1.1 closure documents produced by a prior reporting task — never inside any executable module. There is no `c_ext_e8_*.py` module, no q-binning function, no QMATCH stable-hash selector, and no `test_c_ext_e8_*.py` file anywhere in the repository.

**Reusable frozen utilities identified** (for a future, additive E8 implementation — none were modified by this audit):

| Need | Reusable utility |
|---|---|
| Candidate loading | `c_ext_common.read_json`/`read_jsonl`; the parquet manifest I/O pattern (`read_manifest`/`write_manifest`) used throughout `c_ext_e7_gpat_bank.py` |
| Quality fields / q extraction | `prism_fas.synthesis.quality_gate.evaluate` (frozen 8-criterion gate); `prism_fas.synthesis.quality_calibration`; `c_ext_quality_reconstruct` + `c_ext_quality_analysis.build_q_distribution()` (reads a per-sample q/arm/route parquet, computes SMD via `c_ext_common.standardized_mean_difference`) |
| Stable hashing | `prism_fas.utils.core.stable_json_hash`; `prism_fas.recipes.canonical.stable_hash`; `c_ext_common.sha256_bytes`/`sha256_json`/`identity_hash` — any can implement `sample_id \|\| 'QMATCH-v1'` without new cryptographic logic |
| Fold identities | `c_ext_e7_gpat_bank.py`'s fold/source-domain binding helpers; each fold's own `RESERVE_FINAL_CLOSURE.json` `fold_id` field |
| Detector bank manifests | Historical Flow-1 `c6_matched_bank.py` + `gate_profiles.py` (STRICT/NOMINAL/PERMISSIVE selection logic) for the non-fold-specific bank; **no fold-specific equivalent manifest exists for EXT-F1/F2/F3** (see §3 below) |

Any future E8 implementation must live in an additive namespace (e.g. `src/prism_fas/evaluation/c_ext_e8_qmatched.py`) and must not reuse or mutate E7 selector code or state. None of E7's selector modules were modified to produce this audit.

## 3. Critical input-binding audit

### A. Which candidate table is authoritative?

**No persisted per-sample candidate table** (`sample_id`, `arm`, `route`, `q`, quality-gate acceptance state, fold/source-domain identity) exists in this repository for EXT-F1, EXT-F2, or EXT-F3's own fold-specific reserve-tranche candidates. Every `RESERVE_FINAL_CLOSURE.json` / `RESERVE_TRANCHE*_CLOSURE.json` contains only **aggregate** counts (`arm_route_counts`, `route_quotas`) and quota-redistribution math — never a per-candidate row. `runs/c_ext_q1q2_v1/EXT-F1/`, `EXT-F2/`, `EXT-F3/` contain only empty placeholder directories (verified: zero files under any of them).

The **only** real per-sample q table present locally is `reports/c_ext_q1q2_v1/e2_quality/reconstructed_q/C6_Q_RECONSTRUCTED.parquet` (3072 rows = 1024/arm) — the single **historical Flow-1/Version-C matched bank** (`reports/full/c6/C6_BANK_LOCK_{arm}.json`), not any EXT-Q1Q2 fold-specific candidate pool.

### B. Which frozen artifact does E8 consume?

Two candidates, neither uniquely named by the spec:

1. **EXT-F1's own cumulative reserve-accepted candidates** — fold-specific, source-only, LODO-consistent, the pool that actually trained EXT-F1's G-RND/G-DET/G-LLM detectors under E7. Only aggregate counts are persisted for this pool, at every profile; no per-sample record exists.
2. **The pre-extension historical Flow-1/C6 NOMINAL bank** (1024/arm, 3072 total) — per-sample q IS available locally for this one, and it is in fact the exact bank the real, persisted E2 trigger computation used. But this bank predates EXT-F1/F2/F3, was built under the original (non-LODO) Version-C protocol, and was never used to train any fold's Track-G detector under this extension.

**Resolution: NOT uniquely determined.** Spec §15's 9-step method and "Primary deployment: EXT-F1 × 5 seeds" name neither artifact explicitly. Candidate 2 is authoritative only for the *trigger decision* (whether to run E8 at all) — nothing states it is also authoritative for E8's own bank-construction input.

### C. Which quality profile governs E8 input?

**Resolved component: `NOMINAL`**, on two independent, E0-era authorities — this audit does **not** infer PERMISSIVE from E7's convenience:

1. `c_ext_e0_freeze.py`'s frozen quality-gate binding: `{"rule": "exact 8-criterion hard gate + q geometric-mean weight; NOMINAL profile", "profile": "NOMINAL", "profile_source": "reports/full/c6/C6_PROFILE_SELECTION_LOCK.json (STRICT infeasible)"}` — recorded before any target result and before the E7-v1.1 reserve amendment existed.
2. The real, persisted E2 reconstruction (`C6_Q_RECONSTRUCTION_LOCK.json`) that actually fired the trigger used `quality_profile: "NOMINAL"` for every arm.

`PERMISSIVE` never appears in E0's frozen binding — it is an ad hoc fallback profile E7-v1.1's reserve mechanism introduced later, specific to E7's own 512-common-quota feasibility problem.

**Unresolved sub-item:** E0 itself flags `config_binding_status: "NEEDS_CONFIRMATION"` for exactly which `quality_gate_m8*.yaml` file (of three present: unversioned, `_v2`, `_v3`) produced `quality_threshold_identity=8fa2648643cd526730497ae2d717e17684dda3ecea361fc84929db07ac03bb19`, stating this "must be confirmed against `C6_GATE_PROFILES.json` at E2 before any q value is consumed." A repository-wide search finds this confirmation was **never recorded** in any persisted E2 artifact.

**If the fold-specific pool is intended:** even granting NOMINAL, EXT-F1's own NOMINAL-profile aggregate accepted counts are known — physics `{RND: 409, DET: 394, LLM: 45}`, gpat `{RND: 964, DET: 928, LLM: 705}` — but per §A, no per-sample q breakdown exists for these counts, so they cannot be binned into E8's 10 q-bins.

### Conclusion

The **profile** question is substantially resolved (`NOMINAL`, dual E0/E2 authority), with one open sub-item (exact YAML confirmation). The **candidate-pool identity** question (fold-specific EXT-F1 pool vs. the historical Flow-1 bank) is **not** uniquely resolved by the spec, E0, E2, existing configs, or code. Independent of both questions, the per-sample q table required to execute E8's own method for any fold-specific candidate pool, under any profile, **does not exist** in this repository. A dry-run bin-count table cannot be honestly computed under these conditions without inventing data that is not present.

## 4. q-bin definition (frozen, from spec text — never changed here)

`[0.0,0.1) [0.1,0.2) [0.2,0.3) [0.3,0.4) [0.4,0.5) [0.5,0.6) [0.6,0.7) [0.7,0.8) [0.8,0.9) [0.9,1.0]`

**q = 1.0 handling:** the spec's own notation writes the final bin closed on both ends, `[0.9,1.0]`, unlike the preceding nine half-open bins. q = 1.0 is therefore explicitly included in bin 10 by the literal spec text — never dropped, never double-counted (bin 10 is the only bin with a closed upper bound).

## 5. Dry-run feasibility: NOT PERFORMED

Per this task's own instruction, the dry-run feasibility computation (per-route, per-bin accepted counts; `n_b`; Physics/GPAT common-support totals; total unique E8 bank size per arm; empty bins) is performed **only if** E8 input binding resolves uniquely. It did not (§3). No such table is reported here. EXT-F1's aggregate (non-per-bin) accepted counts under all three profiles are recorded in the companion JSON's `dry_run_feasibility.f1_available_aggregate_data_for_reference_only` block for audit transparency only — they are **not** an E8 feasibility answer.

## 6. Data-integrity rules observed

`target_labels_accessed=false`; no target-domain metric influenced profile choice, bin counts, selection, or bank size; no new candidate generation; no threshold relaxation; no E7 gate change; no candidate regeneration; no arm-specific quality threshold; no change to q; no balancing beyond the (unexecuted) E8 min-count/bin rule; `llm_api_calls=0`; no GPU run.

## 7. Status

**`BLOCKED_PROTOCOL_GAP_E8_INPUT_BINDING`**

The authoritative E8 input candidate pool is not uniquely determined by the governing spec, E0 freeze artifacts, E2 records, existing configs, protocol locks, or code; and, independently, no per-sample candidate-level q table exists in this repository for any EXT-F1/F2/F3 fold-specific candidate pool under any quality-gate profile. This audit does not choose a profile or candidate pool itself and does not compute a dry-run bin-count table under these conditions.

No selected bank was written. No detector checkpoint was written. No scientific artifact was written. E7 was not reopened; no E7 evidence was modified. No scientific code was modified.
