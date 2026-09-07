# E8 Training Adapter -- Source-Only Preflight

**Classification:** implementation + focused tests + source-only preflight only. No commit, no push, no GPU, no detector training, no scientific checkpoint, no target inference, no SiW-Mv2 access, no E8 membership regeneration, no historical C6/E7 modification, no LLM call.

**Status: `READY_FOR_E8_ADAPTER_COMMIT_REVIEW`**

## Adapter identity

- Rule name: `E8_TRACK_G_QMATCH_ADAPTER_V1`
- Adapter rule identity: `56e649a49ba03febbadce6fc62d4362be753d95ccd0fce03d1886f8c5384bbdb`
- Adapter source SHA256: `7fa4be3beec6be6118faa830b39ef1d579a4715aba11c84a08a7a813a4ebd520`
- Implementation parent commit: `c8c87620462df4c670c876f2ceaa052fad5f2454` (no implementation commit exists yet -- this task stops before commit)

## Per-arm validation

| Arm | Membership | Physics | GPAT | C6 bank-lock SHA256 |
|---|---|---|---|---|
| RND | 818 | 354 | 464 | `451964130e59a084470edfac754dc097658b678d4a4f0811ba964c2cfbbe7d92` |
| DET | 818 | 354 | 464 | `bacd544d7d25e411592262c80f092099ba16e5abaa442b00ebe5514eac79b96d` |
| LLM | 818 | 354 | 464 | `ff57043ac3b2aa1591e630b48fb76e588e7fb3ce39a4894093750c01c5c0893a` |

All three arms: **818/818/818**, **354 physics + 464 GPAT** each. Never rebalanced, never forced to 512/512.

## Validation results

- Identity-set equality: **PASS** (all 3 arms)
- Deterministic historical ordering (filtered list is a subsequence of the historical C6 `selected` order): **PASS** (all 3 arms)
- Retained records byte-identical to their historical C6 record: **PASS** (all 3 arms)
- `C6MatchedBankReader` compatibility: **PASS** -- tiny synthetic fixture (224x224 zero images, real `encode_png`/`encode_npz` round-trip through the actual reader's `.sample()`/`.validate()`)
- Sampler 818-bank compatibility: **PASS** -- 30 epochs x 45 steps x 8 synthetic = **10800 draws** from an 818-position pool (354 physics + 464 GPAT), every index in `[0,817]`, no fixed-1024 assumption, same-seed determinism verified, different-seed divergence verified, >700/818 unique positions touched within 30 epochs (heavy reuse, as expected). No SigLIP2 instantiated.
- Target firewall: **PASS** -- `target_access=false`, `target_labels_accessed=false`; source-domain firewall enforced in `filter_bank_lock_to_e8`, verified only `casia_fasd`/`msu_mfsd` present across all 3 arms' filtered output.
- Frozen artifact hashes: **unchanged** (see integrity check below).
- Historical/core scientific source: **zero modifications**.

## Focused tests run

```
pytest -q tests/test_c_ext_e8_training_adapter.py
pytest -q tests/pipeline/test_c6_evidence_and_bank.py
pytest -q tests/test_m4_loader_sampler.py
```

Combined: **68 passed, 0 failed, 16 skipped** (pre-existing, unrelated skips). The dedicated E8 adapter suite alone: **40 passed, 0 failed, 0 skipped**.

**A broad `pytest -k c_ext` sweep was deliberately NOT run** in this task -- it is known to regenerate 3 tracked E7 planning artifacts as a side effect via `test_c_ext_e7_gpat_bank.py`, which was not invoked here.

## Integrity check after implementation

| Check | Result |
|---|---|
| Membership parquet SHA256 | `d2f91738c8a250997491eaabf869a2901f3d759ad19b2fb194df806becba656d` |
| Membership JSONL SHA256 | `0122a47129dbaeac8950e47fb9a77e3bf884a45149433cf20a16a50ca2c27070` |
| Membership lock SHA256 | `16a25b0d777c37c9dc9ec775e7581e45d34c3b6ced77e328b4d94f72ebe33c07` |
| Membership evidence manifest | PASS (5/5) |
| E7 | unchanged |
| Historical C6 locks | unchanged |
| Flow1/Flow2 protected manifest | 170/170 PASS |
| E7 raw evidence | 90/90 PASS |
| `target_access` | false |
| `target_labels_accessed` | false |
| LLM API calls | 0 |
| GPU used | false |
| Detector training started | false |

## Physical bank-view files

**Not created.** `C6MatchedBankReader.open()` takes `bank_lock` as an in-memory mapping (the same pattern `prism_fas.detector.c6_bank.open_arm_bank` already uses: read the historical JSON into a dict, then pass it) -- no physical E8 bank-view JSON file is required by the reader's own API, so none was written. This binding plus the adapter's pure functions are the complete, reproducible provenance.

## Files created

- `src/prism_fas/evaluation/c_ext_e8_training_adapter.py`
- `tests/test_c_ext_e8_training_adapter.py`
- `reports/c_ext_q1q2_v1/e8_qmatched/training/adapter/E8_TRAINING_ADAPTER_BINDING.json`
- `reports/c_ext_q1q2_v1/e8_qmatched/training/adapter/E8_TRAINING_ADAPTER_PREFLIGHT.json`
- `reports/c_ext_q1q2_v1/e8_qmatched/training/adapter/E8_TRAINING_ADAPTER_PREFLIGHT.md`
- `reports/c_ext_q1q2_v1/e8_qmatched/training/adapter/E8_TRAINING_ADAPTER_EVIDENCE.sha256`

**Files modified:** none.

## Status

**`READY_FOR_E8_ADAPTER_COMMIT_REVIEW`**
