# E8 QMATCH-v1 Selector Implementation Report

**Classification:** `E8_QMATCH_V1_SELECTOR_IMPLEMENTATION` — code + tests + protocol-lock implementation only. No scientific E8 membership was generated; no GPU run; no target labels accessed; no LLM call; E7 and Flow-1/Flow-2 untouched; the ratified amendment untouched.

**Implementation commit parent:** `acb5c4bf51c53406944be6d3a3d4542d0c218a2a`

## Source files

**Created:**
- `src/prism_fas/evaluation/c_ext_e8_qmatched.py` — pure q-bin/count/select selector + frozen `E8_QMATCH_V1` protocol payload/identity
- `tests/pipeline/test_c_ext_e8_qmatched.py` — dedicated unit + opt-in real-data reproduction tests

**Modified:** none. Zero modifications to any existing scientific module. Verified zero diff on `c_ext_e7_gpat_bank.py`, `c_ext_e7_reserve_schedule.py`, `c_ext_e7_reserve_orchestrator.py`, `c6_matched_bank.py`, `gate_profiles.py`, `quality_gate.py`.

## Tests

| Suite | Passed | Failed | Skipped |
|---|---|---|---|
| `test_c_ext_e8_qmatched.py` (dedicated) | 44 | 0 | 0 |
| E7 orchestrator + reserve-schedule + protected-manifest + quality-reconstruct (regression) | 76 | 0 | 0 |
| `tests/unit/test_core.py` (stable hashing utilities) | 2 | 0 | 0 |

**Real-data reproduction test** (`test_real_data_reproduction_opt_in`): gated by `pytest.mark.skipif` on the parquet's presence; ran (not skipped) since the artifact exists in this checkout. **PASSED.** No membership file was persisted (asserted inside the test itself via an empty-`tmp_path` check).

## Real-data reproduction result (read-only, in-memory only)

| Metric | Value |
|---|---|
| Input q-table SHA256 verified | yes — `87fdc8ea594a487bfef5206c5a0f0c763c7d19451a5f915e67e2237fd7431cd9` |
| Row count | 3072 (1024/arm) |
| Physics common-support total | **354** |
| GPAT common-support total | **464** |
| Total unique E8 bank size per arm | **818** |
| Empty bins | gpat:bin0, gpat:bin1, gpat:bin2, physics:bin0, physics:bin1, physics:bin2 |
| q == 1.0 count | 0 |
| Membership persisted | **No** |

Expected 354/464/818 totals: **matched exactly.**

## Identities

- `e8_input_binding_rule_identity` (ratified, unchanged): `8223df2d5acb38483b7cfe94bbd0a293489a9200b20fd309fb679ea6621cc7d8`
- `e8_qmatch_selector_rule_identity` (new, this implementation): `95d60b94e3ff69be420eaf9d7e9cfd636cfe0adc667b30db05aadaeabe430126`

## Flags

`target_access=false`, `target_labels_accessed=false`, `llm_api_calls=0`, `gpu_used=false`, `membership_written=false`, `e7_modified=false`, `flow1_flow2_modified=false`, `amendment_modified=false`.

## Status

**`READY_TO_FREEZE_E8_SELECTOR_IMPLEMENTATION`**

Scientific E8 membership generation has **not** been performed. This report documents only that the selector implementation is correct, deterministic, and reproduces the ratified dry-run totals — it does not itself constitute or authorize a scientific run.
