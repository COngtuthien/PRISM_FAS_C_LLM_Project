# E8 Training Runner V2.3 -- CUDA-Execution-Bound Preflight

**Classification:** additive execution/runtime binding correction only. No GPU, no training, no
smoke, no target access, no LLM call, no commit/push.

**Status: `READY_FOR_E8_RUNNER_V2_3_COMMIT_REVIEW`**

## Scientific device contract

Metadata reported by `preflight_e8_run()` (never requires a real GPU):

```
{
  "required_device": "cuda",
  "cpu_fallback_permitted": false,
  "resolver": "prism_fas.pipeline.adapters.c7._scientific_device"
}
```

## Separate runtime device probe

`preflight_e8_scientific_device()` actually invokes the canonical C7 CUDA gate and may hard-fail on a
non-CUDA host. On THIS laptop:

- Result: `None`
- Error: `scientific CUDA device unavailable: scientific C7 requires CUDA and this host has none. A scientific detector trial may not run on the CPU: it would neither finish nor honour the frozen precision contract. Run the rehearsal profile on this machine, or run C7 on the GPU host.`

this laptop has no CUDA device, so the SEPARATE runtime probe honestly hard-fails here -- preflight_e8_run() above is unaffected and remains usable for metadata/preparation testing

## Ordering guarantee

`verify_source_binding_correction() -> resolve_e8_scientific_device() (CUDA-required, never CPU) -> canonical detector input verification (verify_detector_inputs) -> M3B+M7 identity checks -> target firewall zero-check -> C3 arm-bank identity/count check -> E8 membership + C6 bank opening -> frozen Track-G config resolution -> collision guard -> M9Trainer construction with device='cuda' -- confirmed by test_v23_5_no_trainer_construction_after_cuda_failure and test_v21_launch_verifies_correction_identity_before_m3b_and_trainer (extended with a 'device' call marker in V2.3)`

## Frozen contract -- unchanged by this correction

- 15 run IDs: unchanged (15 total)
- Seeds: `[20260806, 20260807, 20260808, 20260809, 20260810]`
- M3B content identity: `08d9d289eb4b462006afcff37cd4750a7c4eeb402c83de5599eda38df44168c9`
- M7 detector recipe bank identity: `fa989938cafdc4887518cc45c35d559d00278358439dc68c2486da10309210cb`
- C3 treatment bank identities by arm: `{'DET': '2802ca5f537c4278eefdb160049d52cb1b667234ec5e32736a733b272e9231c9', 'LLM': 'f225df13ad49eafb90fa9eb903d4dc85efec79c390ec42243a077c80f5d6cb59', 'RND': '07db567c2b432a9239b01d02bac80b95211baafd7f7047ddbad3af43a7ee1136'}`
- C5 candidates root: `runs/full/c5/scientific/candidates`
- Bank counts: `{'total': 818, 'physics': 354, 'gpat': 464}`
- Schedule: `{'total_optimizer_updates': 1575, 'synthetic_draws_per_run': 10800}`
- Target firewall: `{'target_access': False, 'target_labels_accessed': False}`
- `configs/models/m9_detector.yaml`: **not modified**

## Smoke contract

```
{
  "eligible_for_scientific_tables": false,
  "is_scientific_result": false,
  "smoke_arm": "RND",
  "smoke_resume_steps": 6,
  "smoke_run_id": "e8_ext_f1_adapter_smoke",
  "smoke_run_root": "runs/c_ext_q1q2_v1/e8_qmatched/smoke/e8_ext_f1_adapter_smoke",
  "smoke_seed": 20260806,
  "smoke_stage": "G5",
  "smoke_steps": 5
}
```

launch_e8_engineering_smoke() relabels ONLY run_id via dataclasses.replace() and hard-asserts smoke_config.hash() == scientific_config.hash() before ever constructing a trainer -- proven by test_v23_26_smoke_config_hash_equals_scientific_config_hash, which additionally confirms the constructed trainer's config.run_id equals SMOKE_RUN_ID while its .hash() is unchanged.

## Focused tests

```
pytest -q tests/test_c_ext_e8_training_runner.py
pytest -q tests/test_c_ext_e8_training_adapter.py
pytest -q tests/pipeline/test_c6_evidence_and_bank.py tests/test_m4_loader_sampler.py
```

Combined: **214 passed, 0 failed, 16 skipped**. Runner suite alone: **146 passed, 0 failed, 0
skipped** (119 pre-existing + 27 new V2.3 tests). No broad `-k c_ext` sweep.

tests/pipeline/test_c7_scientific_path.py -- uses heavy full-repo fixtures unrelated to this change and timed out regardless of test selection; not run to completion. C7's real _scientific_device delegation is instead proven directly by this suite's own test_v23_3_device_resolver_delegates_to_historical_c7_gate.

## Historical integrity

V1 execution plan + runner reports unchanged; V2, V2.1, and V2.2 correction reports unchanged;
membership, adapter, historical C6, E7 unchanged; C3 banks, M7 bank, `c7.py`, `trainer.py` unchanged;
`configs/models/m9_detector.yaml` unchanged.

## Identities

- New runner rule name: `E8_FIXED_TRACK_G_RUNNER_V2_3_CUDA_EXECUTION_BOUND`
- New runner rule identity: `027eb07d1be935adf440eb6f10db20e464c43febd0cb7a6354393c985adb0880`
- New runner source SHA256: `25bba8fbb89bce2ddc620d79d0a153c4a700aeffea78ec85d525789d42031d8c`
- Implementation parent commit: `1645603dc18300b918ca2836f91fdf352d589b6c`

## Status

**`READY_FOR_E8_RUNNER_V2_3_COMMIT_REVIEW`**
