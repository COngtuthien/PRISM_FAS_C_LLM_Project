# E8 Fixed-Config Track-G Runner -- Local Source-Only Preflight

**Classification:** implementation + focused tests + local source-only preflight only. No GPU, no file copy, no detector training, no full SigLIP2 instantiation, no target inference, no SiW-Mv2 access, no scientific checkpoint, no E8 membership/adapter modification, no historical Track-G/C6/E7 modification, no LLM call, no stage/commit/push.

**Status: `READY_FOR_E8_RUNNER_COMMIT_REVIEW`**

This implementation resolves the operational gap recorded in `../execution/E8_GPU_READINESS_STATUS_CLARIFICATION.json` (`BLOCKED_E8_RUNNER_INTEGRATION_REQUIRED`).

## Runner identity

- Rule name: `E8_FIXED_TRACK_G_RUNNER_V1`
- Runner rule identity: `81842bd81d43c8c942773a0fefed31a0e76bfce1bbbeebc845cd15993dbbdcf0`
- Runner source SHA256: `e20309727587c021fddbca46a5d96a5b20766c06ae34dbc7cbd0514ea4e4ea2c`
- Implementation parent commit: `9fb4e2fc7d30ad940e42999524f0d09b81bbf0e5` (no implementation commit exists yet)

## 15-run validation

All 15 frozen run specs validate. Run IDs: `e8_ext_f1_g_{rnd,det,llm}_qmatch_s{20260806..20260810}`. Condition/arm mapping: `{'RND': 'G-RND-QMATCH', 'DET': 'G-DET-QMATCH', 'LLM': 'G-LLM-QMATCH'}`.

## Frozen Track-G config enforcement

Loaded via `prism_fas.detector.config.load_m9_configs` (the two frozen YAMLs), then the C7 winner delta from `DETECTOR_CONFIG_LOCK.json :: tracks.G.winner_config` is applied via the **same** `dataclasses.replace(base, **overrides)` pattern `c7._scientific_trial_config` already uses in production. `winner_config_sha256` is **recomputed and reasserted** against `97d32c36745e1f4758cbc342b5f83f2fa9c87d69f4ba91605678164d32b5b5dd` on every load -- never trusted blind.

**LR note:** `backbone_lr`/`head_lr` are **not** re-derived through the LR-anchor-multiplier mechanism (`prism_fas.search.lr_decision`) -- Track G's component there is `UNIQUE_INHERITED_ANCHOR`, and this implementation did not conclusively resolve what the recorded `learning_rate_multiplier=2.0` means for that interpretation (naive `anchor*multiplier` disagrees with the value already resolved and frozen in `EXT_MODEL_BINDING.json`). Instead, the runner **asserts** the base config's own LR defaults already equal the frozen resolved binding (1e-5/1e-4) and leaves them untouched -- a verified equality, not a guessed derivation. This is a deliberate scope boundary, documented rather than silently resolved.

Asserted schedule: 35 total epochs (G1=3/G2=2/G5=30), 45 steps/epoch, weight_decay=0.025, warmup_fraction=0.05, lambda_syn=0.25, lambda_risk=0.05.

## Adapter integration

`open_e8_arm_bank` is reused directly; membership filtering is **never** duplicated in the runner. Bank counts are hard-asserted: total=818, physics=354, gpat=464.

## Preflight-only behavior

`preflight_e8_run()` never imports or calls `M9Trainer` and never opens a pixel payload. It resolves: run specification, source package binding, E8 bank counts, frozen training config + hash, output path, collision state, model identity, target firewall -- then stops. Usable later on GPU unmodified.

## Collision-state behavior

States: `['NOT_STARTED', 'IN_PROGRESS', 'COMPLETED', 'FAILED_TECHNICAL', 'BLOCKED_COLLISION']`. Never deletes, never silently overwrites, never silently resumes. Ambiguous existing output classifies as `BLOCKED_COLLISION`. `assert_no_collision()` raises for any state other than `NOT_STARTED` before a launch may proceed.

## Smoke-mode result

**`SUPPORTED_VIA_M9TRAINER_SMOKE_METHOD`** -- `M9Trainer.smoke(steps=..., resume_steps=..., stage=...)` is an existing, already-frozen engineering mechanism that overrides only execution length, never batch composition/LR/weight-decay/any frozen scientific value. Namespace `runs/c_ext_q1q2_v1/e8_qmatched/smoke`, ID `e8_ext_f1_adapter_smoke` -- cannot collide with any of the 15 scientific run IDs. **Not launched by this task.**

## Target-firewall result

`target_access=false`, `target_labels_accessed=false`, allowed domains `['casia_fasd', 'msu_mfsd']`. No target argument exists anywhere in the runner's public API.

## Focused tests

```
pytest -q tests/test_c_ext_e8_training_runner.py
pytest -q tests/test_c_ext_e8_training_adapter.py
pytest -q tests/pipeline/test_c6_evidence_and_bank.py tests/test_m4_loader_sampler.py
```

Combined: **101 passed, 0 failed, 16 skipped** (pre-existing). Runner suite alone: **33 passed, 0 failed, 0 skipped**. No broad `-k c_ext` sweep run; the known E7-mutating test path was not invoked.

## Flags

`target_access=false`, `target_labels_accessed=false`, `llm_api_calls=0`, `gpu_used=false`, `detector_training_started=false`, `detector_checkpoints_created=false`. No historical/core module modified.

## Status

**`READY_FOR_E8_RUNNER_COMMIT_REVIEW`**
