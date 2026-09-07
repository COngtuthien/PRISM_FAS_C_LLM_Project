# E8 Training Runner V2.1 -- Correction-Identity-Bound Preflight

**Classification:** additive provenance correction only. No GPU, no training, no smoke, no target access, no LLM call, no commit/push.

**Status: `READY_FOR_E8_RUNNER_V2_1_COMMIT_REVIEW`**

## Correction verification

`verify_source_binding_correction()` returns `e018cf5bd1d23a88bd5018bb7c86a82dc6e209d9342cf078063a6de869a7f921`, matching the frozen expected value exactly.

## Ordering guarantee

`verify_source_binding_correction() called before validate_m3b_package(), before adapter.open_e8_arm_bank(), and before any M9Trainer import -- confirmed by test_v21_launch_verifies_correction_identity_before_m3b_and_trainer`

## Frozen contract -- unchanged by this correction

- 15 run IDs: unchanged (15 total)
- Seeds: `[20260806, 20260807, 20260808, 20260809, 20260810]`
- M3B content identity: `08d9d289eb4b462006afcff37cd4750a7c4eeb402c83de5599eda38df44168c9`
- E7-D source-support identity: `955b630fec438c80f284ecbcb30fbf10c83251a23fd31d8ab1a52e0f8ce8383b`
- Bank counts: `{'total': 818, 'physics': 354, 'gpat': 464}`
- Schedule: `{'total_optimizer_updates': 1575, 'synthetic_draws_per_run': 10800}`
- Target firewall: `{'target_access': False, 'target_labels_accessed': False}`
- `configs/models/m9_detector.yaml`: **not modified** (out of scope; see the provenance-correction report)

## Focused tests

```
pytest -q tests/test_c_ext_e8_training_runner.py
pytest -q tests/test_c_ext_e8_training_adapter.py
pytest -q tests/pipeline/test_c6_evidence_and_bank.py tests/test_m4_loader_sampler.py
```

Combined: **152 passed, 0 failed, 16 skipped**. Runner suite alone: **84 passed, 0 failed, 0 skipped**. No broad `-k c_ext` sweep.

## Historical integrity

V1 execution plan + runner reports unchanged; V2 correction reports unchanged; membership, adapter, historical C6, E7 unchanged; Flow1/Flow2 170/170 PASS; `configs/models/m9_detector.yaml` unchanged.

## Identities

- New runner rule name: `E8_FIXED_TRACK_G_RUNNER_V2_1_CORRECTION_IDENTITY_BOUND`
- New runner rule identity: `9dd689dfa013f75a5641f566493216718f7a8bfc6b617b06250b63d1cb3e68db`
- New runner source SHA256: `b9556da6275a2fc8ce6f1545e61dbbd1c4a1c454544226ebcff23b8bd0fc9058`
- Implementation parent commit: `eb6797aab6a3c46013b2f0a7b34c81bd9fb0c3c9`

## Status

**`READY_FOR_E8_RUNNER_V2_1_COMMIT_REVIEW`**
