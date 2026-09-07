# E8 Runner V2.3 -- CUDA Execution Binding Correction

**Classification:** `ADDITIVE_EXECUTION_RUNTIME_BINDING_CORRECTION` -- execution/runtime binding fix only, not a scientific
protocol change. Historical V1/V2/V2.1/V2.2 runner reports remain byte-unchanged.

**Bug classification: `BLOCKED_E8_RUNNER_V2_2_GPU_DEVICE_NOT_BOUND`**

launch_scientific_run() constructed M9Trainer(...) with no device= argument, so it silently inherited M9Trainer's device: str = "cpu" default. Zero device/cuda/_scientific_device references existed anywhere in the E8 runner. This is an execution/runtime binding bug only -- not a scientific protocol change: no hyperparameter, bank, membership, seed, q-matching, or source-split change.

## GPU operator audit

launch_scientific_run() constructs M9Trainer(...) with no device= argument. Actual M9Trainer signature has device: str = "cpu". Zero device/cuda/_scientific_device references anywhere in the E8 runner. Independently confirmed locally: grep for 'device=' in the pre-correction M9Trainer construction call found no match; src/prism_fas/detector/trainer.py:250 confirms device: str = "cpu".

## Historical canonical behavior reused

Source: `prism_fas.pipeline.adapters.c7`
Class: `ScientificDeviceUnavailable(AdapterError)`
Function: `_scientific_device() -> str`
Contract: `CUDA_REQUIRED_NO_CPU_FALLBACK`

Historical C7 scientific training does device = _scientific_device(); M9Trainer(..., device=device, synthetic_bank=bank). V2.3 reuses this EXACT gate rather than inventing a second CUDA-selection policy.

## Identities

| Layer | Identity |
|---|---|
| Historical V1 runner rule identity | `81842bd81d43c8c942773a0fefed31a0e76bfce1bbbeebc845cd15993dbbdcf0` |
| Historical V2 runner rule identity | `3293994d312be82969fba884da5b7d13445aa6c1a9d43742f21434991c1b5570` |
| Historical V2.1 runner rule identity | `9dd689dfa013f75a5641f566493216718f7a8bfc6b617b06250b63d1cb3e68db` |
| Historical V2.2 runner rule identity | `1e4a66fb2f49d5ad7b512ab01293fa8dd66e9b5ccaa75af3fa95b2a99689130b` |
| V2 source-binding correction identity | `e018cf5bd1d23a88bd5018bb7c86a82dc6e209d9342cf078063a6de869a7f921` |
| **New V2.3 runner rule identity** | **`027eb07d1be935adf440eb6f10db20e464c43febd0cb7a6354393c985adb0880`** |
| New V2.3 runner rule name | `E8_FIXED_TRACK_G_RUNNER_V2_3_CUDA_EXECUTION_BOUND` |
| New runner source SHA256 | `25bba8fbb89bce2ddc620d79d0a153c4a700aeffea78ec85d525789d42031d8c` |
| Implementation parent commit | `1645603dc18300b918ca2836f91fdf352d589b6c` |

## Fix summary

- added frozen constants SCIENTIFIC_DEVICE_REQUIRED='cuda', SCIENTIFIC_DEVICE_CPU_FALLBACK_PERMITTED=False, SCIENTIFIC_DEVICE_RESOLVER_QUALNAME='prism_fas.pipeline.adapters.c7._scientific_device'
- added resolve_e8_scientific_device(): read-only, delegates to the historical C7 gate (imported inside the function since it is private); wraps ScientificDeviceUnavailable as E8RunnerError; hard-asserts the resolved value is exactly 'cuda' (never 'cpu', never a silent fallback, never chosen based on target/scientific results)
- added preflight_e8_scientific_device(): a SEPARATE read-only runtime probe that actually invokes the canonical C7 gate and may hard-fail on a non-CUDA host -- distinct from preflight_e8_run(), which remains metadata-only and never requires a real GPU
- preflight_e8_run() now reports 'scientific_device_contract' {required_device, cpu_fallback_permitted, resolver} as METADATA only -- never fabricates runtime CUDA availability
- factored the shared read-only helper _resolve_e8_launch_bindings(): resolves, in order, (1) V2 source-binding correction identity, (2) CUDA scientific device, (3) canonical detector inputs, (4) M3B/M7 identity checks, (5) target firewall zero-assertion, (6) C3 treatment-bank validation, (7) E8 filtered bank opening, (8) frozen Track-G config resolution -- used by BOTH launch_scientific_run() and the new launch_e8_engineering_smoke(), so they can never drift into two implementations
- launch_scientific_run() now performs the collision guard (step 9) and M9Trainer construction (step 10) against the scientific run root, passing device=bindings['device'] explicitly -- there is no production scientific execution path on which M9Trainer receives its default CPU device
- removed the public 'device=' surface entirely from launch_scientific_run(); only the private, test-only _device_resolver seam exists, and even a seam returning anything other than 'cuda' hard-fails inside resolve_e8_scientific_device()
- added launch_e8_engineering_smoke(): the ONE official smoke launcher, using frozen SMOKE_ARM='RND'/SMOKE_SEED=20260806/SMOKE_STEPS=5/SMOKE_RESUME_STEPS=6/SMOKE_STAGE='G5'; builds the REAL frozen RND/20260806 config via the shared bindings resolver, relabels only the run_id via dataclasses.replace (hard-asserted not to change M9TrainingConfig.hash(), which excludes run_id), and calls ONLY the existing M9Trainer.smoke(steps=..., resume_steps=..., stage=...) mechanism -- no new shortened training loop
- smoke output is namespaced under smoke_run_root() (runs/c_ext_q1q2_v1/e8_qmatched/smoke/e8_ext_f1_adapter_smoke), verified disjoint from every one of the 15 scientific run roots; any existing smoke output not NOT_STARTED hard-fails -- never deletes, overwrites, or auto-resumes
- build_runner_rule_payload_v2() now binds scientific_device_required='cuda', cpu_fallback_permitted=false, and the canonical resolver qualname, plus an explicit historical_v2_2_runner_rule_identity field
- because the payload's content changed, runner_rule_identity_v2() now computes a NEW identity (027eb07d1be935adf440eb6f10db20e464c43febd0cb7a6354393c985adb0880); every prior value (V1/V2/V2.1/V2.2) is retained only as a recorded historical fact, never recomputed from this function again

## Does NOT change

- the 15 frozen run IDs
- the 5 detector seeds
- EXT-F1 fold only
- arms RND/DET/LLM
- the E8 membership identities / 818-354-464 per-arm counts
- the 35 epochs / 45 steps-per-epoch / 1575 optimizer updates / 10800 synthetic draws schedule
- the frozen Track-G winner config (weight_decay=0.025, warmup=0.05, lambda_syn=0.25, lambda_risk=0.05)
- the M3B content identity (08d9...)
- the M7 neutral bank identity (fa989938...)
- the per-arm C3 treatment bank identities
- the E7-D source-support identity (955b...)
- the q-matching rule
- the quality threshold
- the C6 locks
- the C7 lock
- the target firewall
- c7.py, trainer.py, dataset.py, sampler.py, c6_bank.py, synthetic_bank.py, c_ext_e8_training_adapter.py, sources.py -- all byte-unchanged
- the C3 banks, the M7 bank, the C5 candidates, the E8 membership, and every historical V1/V2/V2.1/V2.2 evidence file -- all byte-unchanged
- configs/models/m9_detector.yaml -- not modified

## Flags

`scientific_protocol_changed=false`, `hyperparameter_changed=false`, `bank_membership_changed=false`,
`seed_changed=false`, `q_matching_changed=false`, `source_split_changed=false`,
`target_informed_adaptation=false`, `target_access=false`, `target_labels_accessed=false`,
`training_performed=false`, `smoke_performed=false`, `gpu_used=false`, `llm_calls=0`.

See `E8_TRAINING_RUNNER_V2_3_BINDING.json` / `E8_TRAINING_RUNNER_V2_3_PREFLIGHT.{json,md}` for the
full V2.3 report.
