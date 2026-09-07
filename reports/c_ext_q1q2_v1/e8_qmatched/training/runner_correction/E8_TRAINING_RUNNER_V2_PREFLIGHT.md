# E8 Training Runner V2 -- Source-Binding Correction Preflight

**Classification:** scientific provenance correction. No GPU, no training, no smoke, no target access, no LLM call, no commit/push.

**Status: `READY_FOR_E8_RUNNER_V2_COMMIT_REVIEW`**
**Local preflight status: `READY_FOR_GPU_RUNTIME_ASSET_REVALIDATION`**

Correction reference: `E8_RUNNER_SOURCE_BINDING_CORRECTION.json`.

## 15-run validation

15 run specs, unchanged: `['e8_ext_f1_g_det_qmatch_s20260806', 'e8_ext_f1_g_det_qmatch_s20260807', 'e8_ext_f1_g_det_qmatch_s20260808']` ... (15 total).

## Track-G config validation

Unchanged by this correction: 35 epochs, 45 steps/epoch, 1575 optimizer updates, 10800 synthetic draws/run, weight_decay=0.025, warmup_fraction=0.05, lambda_syn=0.25, lambda_risk=0.05.

## Source-package-binding correction

| Role | Identity/Path |
|---|---|
| E7-D fold/source-support authority | `955b630fec438c80f284ecbcb30fbf10c83251a23fd31d8ab1a52e0f8ce8383b` |
| M3B runtime package root | `data/packages/prism_data_v1_m3b` |
| M3B content identity | `08d9d289eb4b462006afcff37cd4750a7c4eeb402c83de5599eda38df44168c9` |

Distinct roles, never conflated (`True`).

## Local M3B validation (this checkout)

| Field | Value |
|---|---|
| `overall_state` | `RUNTIME_ASSET_NOT_MATERIALIZED_ON_THIS_HOST` |
| `lock_present` / `lock_status_ok` / `lock_schema_ok` / `lock_content_identity_ok` | True / True / True / True |
| `source_train_present` | False |
| `source_dev_present` | True |
| `problems` | ['/home/cong/PRISM_FAS_C_LLM_Project/data/packages/prism_data_v1_m3b/manifests/source_train.parquet: source_train.parquet not present on this host'] |

PACKAGE_LOCK.json is present and fully valid on this laptop (status/schema/content-identity all confirmed); `source_dev.parquet` is present and matches the GPU-observed hash and counts exactly; `source_train.parquet` is not materialized here. Honestly reported, not fabricated, not treated as a scientific failure.

## Scientific launch hard guard

Enforced before any `M9Trainer` construction in `launch_scientific_run()`: PACKAGE_LOCK exists, status validated, schema `m3b-v1`, content identity `08d9d289...`, source_train present, source_dev present, source-only domains, correct train/dev counts, no overlap/duplicates. On failure: hard-fail before trainer construction -- no fallback to the GPAT-input package, no source_dev fabrication, no random split, no download, no SiW. Verified directly against this checkout's real, partially-materialized package (missing `source_train.parquet`): confirmed to raise `E8RunnerError`.

## Synthetic-bank compatibility

`open_e8_arm_bank()` is called with `package_identity = 08d9d289eb4b462006afcff37cd4750a7c4eeb402c83de5599eda38df44168c9`. The E7-D identity is never passed as the C6 package identity (tested directly).

## Target-firewall result

`target_access=false`, `target_labels_accessed=false`, allowed domains `['casia_fasd', 'msu_mfsd']`.

## Focused tests

```
pytest -q tests/test_c_ext_e8_training_runner.py
pytest -q tests/test_c_ext_e8_training_adapter.py
pytest -q tests/pipeline/test_c6_evidence_and_bank.py tests/test_m4_loader_sampler.py
```

Combined: **136 passed, 0 failed, 16 skipped**. Runner suite alone: **68 passed, 0 failed, 0 skipped**. No broad `-k c_ext` sweep.

## Historical integrity

Original execution plan, readiness clarification, and Runner V1 reports/evidence all verified byte-unchanged. Membership, adapter, historical C6, E7 unchanged. Flow1/Flow2 170/170 PASS.

## Identities

- New runner rule identity: `3293994d312be82969fba884da5b7d13445aa6c1a9d43742f21434991c1b5570`
- New runner source SHA256: `3cf0cb6234121cbbc0e443641193bb8201743d700c05f1954db8edc0a0a8e3cf`
- Implementation parent commit: `25ffb908e2fe32bf7c27ae0fb91d4546738db6eb`

## Status

**`READY_FOR_E8_RUNNER_V2_COMMIT_REVIEW`**
