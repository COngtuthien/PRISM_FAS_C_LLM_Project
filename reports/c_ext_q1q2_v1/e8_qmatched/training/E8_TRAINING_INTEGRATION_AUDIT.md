# E8 Training-Integration Audit + 15-Run Preflight Plan

**Classification:** read-only training-integration audit + preflight plan only. No training code modified, no adapter created, no GPU access, no training, no inference, no target labels accessed, no calibration performed, no E8 membership regenerated, no LLM call, no commit/push performed by this task.

**Final status: `READY_FOR_E8_TRAINING_INTEGRATION_IMPLEMENTATION`**

## 1. Firewall verification

HEAD at start: `c8c87620462df4c670c876f2ceaa052fad5f2454`. Membership evidence manifest: PASS (5/5). Canonical membership SHA256 verified: `d2f91738c8a250997491eaabf869a2901f3d759ad19b2fb194df806becba656d`. Selector implementation commit: `1bc21807564650141d22b401003d91c6b10d7b32`. Selector rule identity: `95d60b94e3ff69be420eaf9d7e9cfd636cfe0adc667b30db05aadaeabe430126`. Input-binding rule identity: `8223df2d5acb38483b7cfe94bbd0a293489a9200b20fd309fb679ea6621cc7d8`.

## 2. Track-G binding — RESOLVED

Read from `EXT_MODEL_BINDING.json`, `EXT_SEED_REGISTRY.json`, `configs/train/m9_reference.yaml`, `configs/data/loader_m4.yaml` (each cross-cited to `DETECTOR_CONFIG_LOCK.json`/`MASTER_METHOD_FORMULA_SOURCE_MAP.md`).

- **Architecture:** frozen SigLIP2 backbone (`google/siglip2-base-patch16-224`, identity `7e059e40...`, 0 trainable params) + `Linear(256->1)(LayerNorm(Linear(dim->256)(z_global)))` head, single-logit fusion. `variant_identity = 1e26c31bba8922bff9fc15ce2e163d2f82f5a353614c55b52bc0db33b39fac8d`.
- **Loss:** `L_total = L_cls_real + lambda_syn*qbar*L_cls_syn + lambda_risk*L_risk`; **winner** values `lambda_syn=0.25`, `lambda_risk=0.05` (the generic `m9_reference.yaml` default `lambda_syn=0.50` is the untuned reference value — the C7-search winner governs).
- **Optimizer:** AdamW, `backbone_lr=1e-5`, `head_lr=1e-4`, **winner** `weight_decay=0.025` (generic anchor is 0.05), betas (0.9, 0.999), grad-clip 1.0, AMP bf16/fp16/fp32-CPU.
- **Scheduler:** cosine, warmup fraction 0.05 (winner), min_lr_scale 0.0.
- **Stages:** `flow=[G1,G2,G5,G6]` — G1 warmup_detector (3 epochs), G2 manifold_warmup (2 epochs, manifold OFF for Track G but epochs still counted), G5 mixed (30 epochs), **total_epochs=35**. G6 = source_dev calibration, not additional optimizer-update epochs. `m9_reference.yaml`'s own `SPEC_UNDERSPECIFIED` comment resolves the warm-up scope explicitly: **both G1 and G2 (5 epochs total) use the real-only warmup batch** (16 live/16 real_spoof/0 synthetic); only G5's 30 epochs draw synthetic samples.
- **Batch:** `batch_size=32`, `steps_per_epoch=45` (`ceil(1440/32)`, constant across stages), mixed G5 = 12 real-live/12 real-spoof/8 synthetic, warmup G1+G2 = 16/16/0, `accumulation_steps=1`.
- **Checkpoint selection:** every 250 steps, lexicographic min(`source_dev/acer`, `source_dev/bpcer`, `source_dev/nll`), tolerance 1e-12, `uses_target=false`.
- **Calibration:** source_dev only, Adam(lr=0.05, 500 iters, float64) temperature fit + ACER/APCER-lexicographic threshold search, `uses_target=false`.
- **Preprocessing:** opencv decode, RGB, 224×224, float32, [0,1], channels-first; backbone input contract `x_norm=2*x-1`.
- **Determinism:** cudnn deterministic, no benchmark, seeded sampler material `sha256(schema|pool_identity|seed|epoch|pool)`, 0 dataloader workers.
- **Warm-up/base checkpoint:** none beyond the frozen SigLIP2 backbone pin.
- **Pre-existing, non-blocking gap:** `quality_gate` config filename remains `NEEDS_CONFIRMATION` (identical to the gap already ratified as non-blocking in the E8 amendment — the threshold identity is cryptographically bound regardless of filename).

## 3. Detector seeds — RESOLVED

`[20260806, 20260807, 20260808, 20260809, 20260810]` — confirmed identical in both `EXT_SEED_REGISTRY.json` and `EXT_MODEL_BINDING.json`. The Shuffle perturbation seeds `[20260911, 20260912, 20260913]` (E6/E9 masks P1/P2/P3) are explicitly excluded — not detector seeds.

## 4. Baseline training exposure — RESOLVED

- Total epochs: **35**; steps/epoch: **45**; total optimizer updates/run: **1575**.
- Synthetic draws occur **only** during G5 (30 epochs): `30 * 45 * 8` = **10800** synthetic draws/run — unchanged for E8.
- E8 unique bank size: **818**. Effective exposure / bank size reuse ratio: **13.2029×**.
- The identical 10800-draw schedule already runs, unmodified, against the historical 1024-sample banks (~10.55× reuse); no optimizer-update count is changed to force a clean pass over 818.
- **Replacement-sampling mechanism already exists**, unmodified: `sampler.py`'s `_Stream`/`M9BatchSampler` draws a deterministic, seeded permutation per pool (per route) without replacement *within* a cycle, and deterministically reshuffles once a cycle is exhausted — functionally equivalent to sampling with replacement at the scale this schedule already requires. No hardcoded 1024/512 pool-size dependency exists anywhere in `sampler.py`, `dataset.py`, `config.py`, or `contracts.py` (verified by grep — zero matches).

## 5. Loader audit (A–J)

| Item | Finding |
|---|---|
| A. Expected input | `M9TrainingDataset` accepts either `bank_root` + `SyntheticBankReader.open()` (the single fail-closed M9-reference bank) **or** a pre-built `bank=` object passed in directly — documented in `dataset.py` as "the Version-C seam: C7/C8 train against one arm's frozen C6 matched bank ... opened by its own fail-closed reader (`detector.c6_bank`)" |
| B. Direct parquet/jsonl consumption | **No** — neither file carries the full row shape `C6MatchedBankReader` assembles |
| C. Locator sufficiency | `sample_id` (==`candidate_id`) is sufficient as a **join key**, not a standalone locator |
| D. Required join | Stage 1: `reports/full/c6/C6_BANK_LOCK_{RND,DET,LLM}.json` `selected[]`, keyed by `candidate_id`; stage 2: that arm's C5 `CANDIDATE.json` record at `candidate_dir(candidates_root, arm, candidate_id)`, exactly what `c6_bank.py:C6MatchedBankReader.open()` already does for historical arm banks |
| E. Join determinism | **Verified 1:1 and complete** — all 2454 E8-selected sample_ids (818/arm) found exactly once each in their arm's bank-lock `selected[]` (0 missing, 0 duplicate, per arm and overall) |
| F. q as loss weight | **Yes** — `qbar = mean(quality_weight[synthetic])` scales only the synthetic loss bracket |
| G. Route metadata preserved | **Yes** — flows through to `SyntheticSample.route`, validated against `{physics, gpat}`; sampler pools are keyed per route |
| H. Fixed 1024-bank dependency | **No** — no hardcoded pool-size literal found anywhere |
| I. Without-replacement default | Without replacement **within** a cycle; cycles repeat (functionally with-replacement) once exhausted |
| J. Global loader change needed? | **N/A** — no change to `sampler.py`/`dataset.py` is required; the existing `bank=` seam is explicitly designed for exactly this purpose |

**Additive E8 adapter required:** yes, but minimal — a thin function that filters an arm's real `C6_BANK_LOCK_<ARM>.json` `selected[]` down to that arm's 818 E8-selected `candidate_id`s (all other lock fields preserved verbatim), then calls the **existing, unmodified** `prism_fas.detector.c6_bank.C6MatchedBankReader.open()` with that filtered lock. No new reader class, no change to `dataset.py`, `sampler.py`, `c6_bank.py`, or `synthetic_bank.py` — none of these four files were modified by this task.

## 6. Asset locator audit (2454 rows) — RESOLVED AT IDENTITY LEVEL

| Arm | Rows | Present | Missing | Unique assets | Duplicates | Physics | GPAT | Source domains |
|---|---|---|---|---|---|---|---|---|
| RND | 818 | 818 | 0 | 818 | 0 | 354 | 464 | {'casia_fasd': 451, 'msu_mfsd': 367} |
| DET | 818 | 818 | 0 | 818 | 0 | 354 | 464 | {'casia_fasd': 465, 'msu_mfsd': 353} |
| LLM | 818 | 818 | 0 | 818 | 0 | 354 | 464 | {'casia_fasd': 430, 'msu_mfsd': 388} |

**Totals: 2454/2454 rows resolve, 0 missing, 0 duplicates.** Resolution basis: every E8-selected `sample_id` is a verified subset of `C6_Q_RECONSTRUCTED.parquet` (already proven `CONSISTENT`, 0 disagreements/3072, against these same bank-lock files during the ratified amendment); this audit additionally re-verified the 2454-row subset directly here.

**Byte-level caveat:** identity/manifest-level resolution is complete and deterministic; the actual rendered pixel payloads (`synthetic.png`/`exact_mask.png`/`artifact_map.npz`) were **not** verified byte-for-byte from this laptop checkout — this laptop holds C5 `CANDIDATE.json` metadata only for the LLM arm (no pixel files), and zero local candidate directories for RND/DET. This does not change the RESOLVED classification: assets are not missing or unresolvable, only not locally co-located for byte verification — that is the GPU asset plan's job (§7), not this audit's.

## 7. GPU asset availability plan (classification only — no transfer executed)

| Category | Contents |
|---|---|
| A. Already git-trackable | E8 membership/lock/selector, `C6_BANK_LOCK_{RND,DET,LLM}.json`, `m9_reference.yaml`, `loader_m4.yaml`, `EXT_MODEL_BINDING.json` |
| B. Present only on laptop | none identified beyond what's already committed and pushed |
| C. Expected already on GPU (historical) | the 2454 candidates' rendered `synthetic.png`/`exact_mask.png`/`artifact_map.npz` under `runs/full/c5/scientific/candidates/{RND,DET,LLM}/<candidate_id>/` — expected present since this exact bank already trained Version-C's original Track-G; **not to be regenerated** |
| D. Large weights | frozen SigLIP2 backbone (`weights/pretrained/m9/siglip2`, identity `7e059e40...`) — shared/frozen across all Track-G runs |
| E. Source CASIA/MSU data | EXT-F1 source package, `frozen_package_identity=955b630fec438c80f284ecbcb30fbf10c83251a23fd31d8ab1a52e0f8ce8383b`, `E7D_BINDING_MATCH=true`, `e7d_status=CLOSED_VALID` |
| F. Frozen synthetic assets selected by E8 | same set as category C |
| G. source_train/source_dev manifests | EXT-F1's fold-scoped M3B package manifests — already present (E7-D CLOSED_VALID) |
| H. Warm-up/base checkpoint | not applicable — none beyond the SigLIP2 backbone pin (category D) |

## 8. Source/target firewall — PASS

EXT-F1 source domains: `casia_fasd`, `msu_mfsd`. Target: `siw_mv2`. Training uses only source_train + E8 synthetic (verified zero `siw_mv2` rows across all 2454 selected samples). Checkpoint selection and calibration use only `source_dev` (`uses_target=false` explicit in the binding for both). `target_access=false`, `target_labels_accessed=false`.

## 9. Planned 15-run matrix (plan only — no run directories or checkpoints created)

Naming convention: `{fold}-{track}-{arm}-QMATCH-s{seed}`, following the spec Appendix A convention `{fold}-{track}-{condition}-{bank_or_variant}-s{seed}` (spec example: `EXT-F2-G-LLM-FROZENBANK-s20260808`); `QMATCH` replaces `FROZENBANK` since E8 draws from the q-matched bank, not the plain E7 bank. No competing convention was invented.

| Run ID | Condition | Arm | Seed | Planned optimizer updates | Planned synthetic draws |
|---|---|---|---|---|---|
| EXT-F1-G-RND-QMATCH-s20260806 | G-RND-QMATCH | RND | 20260806 | 1575 | 10800 |
| EXT-F1-G-RND-QMATCH-s20260807 | G-RND-QMATCH | RND | 20260807 | 1575 | 10800 |
| EXT-F1-G-RND-QMATCH-s20260808 | G-RND-QMATCH | RND | 20260808 | 1575 | 10800 |
| EXT-F1-G-RND-QMATCH-s20260809 | G-RND-QMATCH | RND | 20260809 | 1575 | 10800 |
| EXT-F1-G-RND-QMATCH-s20260810 | G-RND-QMATCH | RND | 20260810 | 1575 | 10800 |
| EXT-F1-G-DET-QMATCH-s20260806 | G-DET-QMATCH | DET | 20260806 | 1575 | 10800 |
| EXT-F1-G-DET-QMATCH-s20260807 | G-DET-QMATCH | DET | 20260807 | 1575 | 10800 |
| EXT-F1-G-DET-QMATCH-s20260808 | G-DET-QMATCH | DET | 20260808 | 1575 | 10800 |
| EXT-F1-G-DET-QMATCH-s20260809 | G-DET-QMATCH | DET | 20260809 | 1575 | 10800 |
| EXT-F1-G-DET-QMATCH-s20260810 | G-DET-QMATCH | DET | 20260810 | 1575 | 10800 |
| EXT-F1-G-LLM-QMATCH-s20260806 | G-LLM-QMATCH | LLM | 20260806 | 1575 | 10800 |
| EXT-F1-G-LLM-QMATCH-s20260807 | G-LLM-QMATCH | LLM | 20260807 | 1575 | 10800 |
| EXT-F1-G-LLM-QMATCH-s20260808 | G-LLM-QMATCH | LLM | 20260808 | 1575 | 10800 |
| EXT-F1-G-LLM-QMATCH-s20260809 | G-LLM-QMATCH | LLM | 20260809 | 1575 | 10800 |
| EXT-F1-G-LLM-QMATCH-s20260810 | G-LLM-QMATCH | LLM | 20260810 | 1575 | 10800 |

Every run additionally carries: `membership_parquet_sha256=d2f91738c8a250997491eaabf869a2901f3d759ad19b2fb194df806becba656d`, `selector_rule_identity=95d60b94e3ff69be420eaf9d7e9cfd636cfe0adc667b30db05aadaeabe430126`, `input_binding_rule_identity=8223df2d5acb38483b7cfe94bbd0a293489a9200b20fd309fb679ea6621cc7d8`, `source_split_identity` (EXT-F1 package identity) `955b630fec438c80f284ecbcb30fbf10c83251a23fd31d8ab1a52e0f8ce8383b`, `target_access=false`.

## 10. Integrity validation after audit

E8 membership evidence PASS (5/5); E8 membership canonical SHA256 unchanged; E8 selector unchanged; E8 input-binding/amendment unchanged; Flow1/Flow2 protected manifest **170/170 PASS**; E7 raw evidence **90/90 PASS**; historical C6 unchanged; E7 unchanged; **zero scientific source-code modifications** in this task; pre-existing unrelated `.gitignore`/`dataset.py` untouched.

**Note:** a prior turn's broader `pytest -k c_ext` regression sweep had regenerated 3 tracked non-scientific planning artifacts under `reports/c_ext_q1q2_v1/e7_three_fold/gpat_bank/` (`EXECUTION_PLAN.json`, `QUALITY_GATE_BINDING.json`, `READINESS.json`) as a known, documented test side effect. Found still dirty in the working tree at the start of this task; restored via `git checkout --` to their committed state before writing this report. This task itself performed no writes to E7.

## Status

**`READY_FOR_E8_TRAINING_INTEGRATION_IMPLEMENTATION`**
