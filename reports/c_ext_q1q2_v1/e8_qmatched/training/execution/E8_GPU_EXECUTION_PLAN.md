# E8 EXT-F1 GPU Execution Plan (Immutable Preflight)

**Classification:** `E8_GPU_EXECUTION_PLAN_V1` -- read-only validation + execution-contract generation + GPU preflight command generation ONLY. No GPU connection, no file copy, no training, no inference, no target access, no checkpoint, no prediction, no membership regeneration, no adapter modification, no C6/E7 modification, no LLM call, no commit/push performed by this task.

**Status: `READY_FOR_GPU_ASSET_PREFLIGHT`**

**`e8_gpu_execution_plan_identity`:** `74f9503f1fa22ffe96eca6585ff1184b3f6bc122380a3e39c485a35dc02a39de`

## Repository state

Branch `gpu-work/e7-gpat-bank-prep`, HEAD `9fb4e2fc7d30ad940e42999524f0d09b81bbf0e5` (parent `c8c87620462df4c670c876f2ceaa052fad5f2454`).

## Frozen identities

| Identity | Value |
|---|---|
| Membership commit | `c8c87620462df4c670c876f2ceaa052fad5f2454` |
| Membership parquet SHA256 | `d2f91738c8a250997491eaabf869a2901f3d759ad19b2fb194df806becba656d` |
| Membership JSONL SHA256 | `0122a47129dbaeac8950e47fb9a77e3bf884a45149433cf20a16a50ca2c27070` |
| Membership lock SHA256 | `16a25b0d777c37c9dc9ec775e7581e45d34c3b6ced77e328b4d94f72ebe33c07` |
| Selector rule identity | `95d60b94e3ff69be420eaf9d7e9cfd636cfe0adc667b30db05aadaeabe430126` |
| Input-binding rule identity | `8223df2d5acb38483b7cfe94bbd0a293489a9200b20fd309fb679ea6621cc7d8` |
| Adapter implementation commit | `9fb4e2fc7d30ad940e42999524f0d09b81bbf0e5` |
| Adapter rule identity | `56e649a49ba03febbadce6fc62d4362be753d95ccd0fce03d1886f8c5384bbdb` |
| Adapter source SHA256 | `7fa4be3beec6be6118faa830b39ef1d579a4715aba11c84a08a7a813a4ebd520` |
| Track-G variant identity | `1e26c31bba8922bff9fc15ce2e163d2f82f5a353614c55b52bc0db33b39fac8d` |
| C7 winner config SHA256 | `97d32c36745e1f4758cbc342b5f83f2fa9c87d69f4ba91605678164d32b5b5dd` |
| EXT-F1 source package identity | `955b630fec438c80f284ecbcb30fbf10c83251a23fd31d8ab1a52e0f8ce8383b` |

C6 bank-lock hashes: RND `451964130e59a084470edfac754dc097658b678d4a4f0811ba964c2cfbbe7d92`, DET `bacd544d7d25e411592262c80f092099ba16e5abaa442b00ebe5514eac79b96d`, LLM `ff57043ac3b2aa1591e630b48fb76e588e7fb3ce39a4894093750c01c5c0893a`.

## Scientific bank contract

3 conditions (`G-RND-QMATCH`, `G-DET-QMATCH`, `G-LLM-QMATCH`), each **818** unique (354 physics + 464 GPAT), grand total **2454** rows. Membership must never be expanded to 10800 rows -- repeated exposure is owned entirely by the existing, unmodified Track-G sampler.

## Track-G training contract

Model `google/siglip2-base-patch16-224` (frozen backbone) + existing frozen winner head. AdamW, backbone_lr=1e-5, head_lr=1e-4, weight_decay=0.025, betas=(0.9,0.999), grad-clip 1.0; cosine scheduler, warmup 0.05. Batch size 32.

| Stage | Epochs | Phase | Real-live | Real-spoof | Synthetic |
|---|---|---|---|---|---|
| G1 | 3 | real_only | 16 | 16 | 0 |
| G2 | 2 | real_only | 16 | 16 | 0 |
| G5 | 30 | mixed | 12 | 12 | 8 |

Steps/epoch **45**, total epochs **35**, total optimizer updates **1575**. Synthetic draws during G5: 30×45×8 = **10800**/run. E8 unique bank/arm: **818**. Descriptive exposure ratio: 10800/818 ≈ **13.2029339853**. The schedule is never altered to make one epoch equal one traversal of the 818-bank.

## Detector seeds

`[20260806, 20260807, 20260808, 20260809, 20260810]` -- Shuffle seeds `[20260911, 20260912, 20260913]` explicitly excluded (E6/E9 perturbation masks, never detector seeds).

## Execution order

`1-5 RND seeds ascending, 6-10 DET seeds ascending, 11-15 LLM seeds ascending`. `execution_order_is_not_selection_rule = true` -- operational only, must never change scientific semantics or be used for result-dependent reordering later.

## Frozen 15-run matrix

| Order | Run ID | Condition | Seed |
|---|---|---|---|
| 1 | e8_ext_f1_g_rnd_qmatch_s20260806 | G-RND-QMATCH | 20260806 |
| 2 | e8_ext_f1_g_rnd_qmatch_s20260807 | G-RND-QMATCH | 20260807 |
| 3 | e8_ext_f1_g_rnd_qmatch_s20260808 | G-RND-QMATCH | 20260808 |
| 4 | e8_ext_f1_g_rnd_qmatch_s20260809 | G-RND-QMATCH | 20260809 |
| 5 | e8_ext_f1_g_rnd_qmatch_s20260810 | G-RND-QMATCH | 20260810 |
| 6 | e8_ext_f1_g_det_qmatch_s20260806 | G-DET-QMATCH | 20260806 |
| 7 | e8_ext_f1_g_det_qmatch_s20260807 | G-DET-QMATCH | 20260807 |
| 8 | e8_ext_f1_g_det_qmatch_s20260808 | G-DET-QMATCH | 20260808 |
| 9 | e8_ext_f1_g_det_qmatch_s20260809 | G-DET-QMATCH | 20260809 |
| 10 | e8_ext_f1_g_det_qmatch_s20260810 | G-DET-QMATCH | 20260810 |
| 11 | e8_ext_f1_g_llm_qmatch_s20260806 | G-LLM-QMATCH | 20260806 |
| 12 | e8_ext_f1_g_llm_qmatch_s20260807 | G-LLM-QMATCH | 20260807 |
| 13 | e8_ext_f1_g_llm_qmatch_s20260808 | G-LLM-QMATCH | 20260808 |
| 14 | e8_ext_f1_g_llm_qmatch_s20260809 | G-LLM-QMATCH | 20260809 |
| 15 | e8_ext_f1_g_llm_qmatch_s20260810 | G-LLM-QMATCH | 20260810 |

Every run additionally carries: membership/lock SHA256, selector identity, input-binding identity, adapter identity + implementation commit, Track-G variant identity, C7 winner config SHA256, that arm's C6 bank-lock SHA256, `optimizer_updates=1575`, `synthetic_draws=10800`, `unique_synthetic_count=818`, `physics_unique=354`, `gpat_unique=464`, `target_access=false`, `target_labels_accessed=false` -- see the companion JSON for the full per-run record.

## Runner resolution

**Trainer class:** `prism_fas.detector.trainer.M9Trainer` (`src/prism_fas/detector/trainer.py`) -- already has a `synthetic_bank: Any = None` seam, documented as "An ALREADY-VERIFIED synthetic bank reader. Version C's C7/C8 pass one arm's frozen C6 matched bank here (`detector.c6_bank.C6MatchedBankReader`)" -- exactly what `c_ext_e8_training_adapter.open_e8_arm_bank()` returns. **No trainer modification is required.**

**Historical reuse precedent:** `src/prism_fas/pipeline/adapters/c7.py:_run_scientific_trial` already does `bank = open_arm_bank(...)` then `M9Trainer(..., synthetic_bank=bank)` -- proving this exact seam is already used in production.

**Why C7's adapter is not reused directly:** it is embedded in C7's own hyperparameter-*search* trial/workflow state machine (`AdapterRequest`, trial objects, `_trial_run_root`, unique-configuration counting, winner-selection logic) -- none of which applies to E8, which runs the single, already-determined winner config for a fixed (arm, seed), not a search.

**Existing bare CLI checked:** `scripts/m9_local_smoke.py` constructs `M9Trainer` directly from argparse flags but has no flag to accept an externally-supplied bank -- it always opens the frozen default M8 v3 bank.

**Status: `ADDITIVE_E8_RUNNER_REQUIRED`** -- a thin, additive launcher is needed to (1) call `open_e8_arm_bank`, (2) build `M9TrainingConfig` with the frozen winner hyperparameters + given seed, (3) point `run_root` at the new E8 namespace, and (4) enforce the run-collision/resume policy below before calling `M9Trainer()`. **Not implemented in this task.**

## Run-collision / resume policy

States: `NOT_STARTED`, `IN_PROGRESS`, `COMPLETED`, `FAILED_TECHNICAL`, `BLOCKED_COLLISION`. `M9Trainer.__post_init__` only does `run_root.mkdir(parents=True, exist_ok=True)` -- it does **not** itself refuse a pre-existing completed run. This classification must be enforced by the future additive E8 launcher before it calls `M9Trainer()`. Never silently overwrite/resume/delete; one failed technical job must not trigger rerun of all arms; no scientific rerun merely because accuracy looks bad.

## Target firewall

Source domains `casia_fasd`/`msu_mfsd`; target `siw_mv2`. Training inputs: source_train + E8 source synthetic bank only. Checkpoint selection and later calibration: source_dev only. Target forbidden in training, dataset construction, checkpoint selection, source calibration. `target_access=false`, `target_labels_accessed=false`. Target scoring is a later, separately authorized phase.

## GPU repository contract

GPU repo `/home/sparc/workdir/longnm/PRISM_FAS_C_LLM_Project`, branch `gpu-work/e7-gpat-bank-prep`, **required exact commit** `9fb4e2fc7d30ad940e42999524f0d09b81bbf0e5` -- "latest branch" or "equivalent working tree" is never acceptable. See `E8_GPU_PREFLIGHT_COMMANDS.sh` §1 for the exact verification commands (to be run manually by the user).

## Model-weight preflight

Model identity `google/siglip2-base-patch16-224`, revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`. Resolver: `prism_fas.detector.pretrained.SigLIP2Artifacts.resolve(weight_root)`. **File-level SHA256 hashes are frozen** (`src/prism_fas/detector/pretrained.py:SIGLIP2_PIN`), not fabricated here -- read directly from source; see the companion JSON for all 7 file hashes.

## Source data preflight

EXT-F1 package identity `955b630fec438c80f284ecbcb30fbf10c83251a23fd31d8ab1a52e0f8ce8383b`, root `data/processed/c_ext_q1q2_v1/e7_gpat_bank/gpat_input/EXT-F1`, manifests `source_train.parquet`/`source_dev.parquet`. Source domains `casia_fasd`/`msu_mfsd` only, `siw_in_fold=false`. `m3b_train_reference_count=1440` (from `GPAT_FOLD_SOURCE_BINDING_DETAIL.json`, independently cross-checked against `loader_m4.yaml`'s own `ceil(1440/32)=45` derivation). source_dev row count not separately recorded in any checked artifact -- not fabricated. `E7D_BINDING_MATCH=true`, `e7d_status=CLOSED_VALID`. Package not materialized on this laptop (large binary tree, as expected) -- GPU-side verification template is in `E8_GPU_PREFLIGHT_COMMANDS.sh` §4.

## Synthetic pixel-asset GPU verification (required, not yet executed)

For every arm: membership 818, resolved historical records 818 required, physics 354, GPAT 464, missing payload 0 required. Grand total **2454/2454** payloads present required. Where historical payload hashes exist, they must be verified. **No rerender is ever authorized.** If any payload is missing: future status = `BLOCKED_E8_GPU_ASSET_MISSING`. Template procedure: `E8_GPU_PREFLIGHT_COMMANDS.sh` §7 (adjust `candidates_root`/recipes/package identities to the real GPU paths before running).

## Storage assessment

**`STORAGE_REQUIREMENT_NOT_YET_BOUND`** -- no persisted historical Track-G checkpoint/run size figure was found in any report checked; no number is invented. `df -h`/`du -sh` commands are provided in the preflight script (§8) for the user to establish this on first GPU access.

## Future run root

`runs/c_ext_q1q2_v1/e8_qmatched/ext_f1` -- additive, per-`run_id` subdirectory; distinct from historical Version-C/E7/Flow1/Flow2 run namespaces. No directory created by this task.

## Engineering smoke policy

One arm, one seed, minimal steps, source-only, separate namespace `runs/c_ext_q1q2_v1/e8_qmatched/smoke`. Not a scientific result, must not consume target data, must not alter frozen membership or the 15-run matrix. `scripts/m9_local_smoke.py` demonstrates the reduced-step CPU-smoke pattern but has no flag for an externally-supplied bank -- same additive-launcher gap as the full runner. **Status: `SMOKE_REQUIRES_ADDITIVE_RUNNER_SUPPORT`.**

## Scientific run launch policy

Frozen schedule only -- no early stopping, adaptive LR, batch-size/epoch reduction, seed substitution, bank replacement, route rebalance, gate relaxation, or post-result hyperparameter change. Checkpoint selection: existing source-dev rule only. One failed technical job never triggers automatic rerun of all arms.

## Result artifact plan

Every future run must preserve: run config, run identity, implementation commit, membership identities, seed, console log, start/end timestamps, exit code, environment summary, GPU identity, training metrics, checkpoint hashes, best-checkpoint selection evidence, source-dev metrics, `target_access`/`target_labels_accessed` flags. No target predictions are produced.

## Flags

`target_access=false`, `target_labels_accessed=false`, `llm_api_calls=0`, `gpu_used=false`, `detector_training_started=false`, `detector_checkpoints_created=false`.

## Status

**`READY_FOR_GPU_ASSET_PREFLIGHT`**
