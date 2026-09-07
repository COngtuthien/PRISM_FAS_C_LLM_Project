# E8 Runner V2.1 -- Provenance Correction (Correction Identity Bound)

**Classification:** `ADDITIVE_PROVENANCE_CORRECTION` -- additive only. Historical Runner V2 (commit `eb6797aab6a3c46013b2f0a7b34c81bd9fb0c3c9`) and its correction reports remain byte-unchanged.

**Bug classification: `BLOCKED_E8_RUNNER_V2_CORRECTION_IDENTITY_NOT_BOUND`**

V2's rule payload named the source-binding correction artifact by PATH only, never by content IDENTITY -- contradicting the payload's own docstring ('plus the source-binding correction artifact's own identity') and the correction contract. This is a provenance-binding gap only, not a scientific protocol change.

## Identities

| Layer | Identity |
|---|---|
| Historical V1 runner rule identity | `81842bd81d43c8c942773a0fefed31a0e76bfce1bbbeebc845cd15993dbbdcf0` |
| Historical V2 runner rule identity | `3293994d312be82969fba884da5b7d13445aa6c1a9d43742f21434991c1b5570` |
| V2 source-binding correction identity | `e018cf5bd1d23a88bd5018bb7c86a82dc6e209d9342cf078063a6de869a7f921` |
| **New V2.1 runner rule identity** | **`9dd689dfa013f75a5641f566493216718f7a8bfc6b617b06250b63d1cb3e68db`** |
| New V2.1 runner rule name | `E8_FIXED_TRACK_G_RUNNER_V2_1_CORRECTION_IDENTITY_BOUND` |
| New runner source SHA256 | `b9556da6275a2fc8ce6f1545e61dbbd1c4a1c454544226ebcff23b8bd0fc9058` |

## Fix summary

- added frozen constant SOURCE_BINDING_CORRECTION_SHA256 = e018cf5bd1d23a88bd5018bb7c86a82dc6e209d9342cf078063a6de869a7f921
- added verify_source_binding_correction(): fail-closed, read-only; requires the file at SOURCE_BINDING_CORRECTION_RELATIVE_PATH to exist, computes its SHA256, requires exact equality with SOURCE_BINDING_CORRECTION_SHA256; performs no writes
- preflight_e8_run() now calls verify_source_binding_correction() and reports the verified SHA256 in its result
- launch_scientific_run() now calls verify_source_binding_correction() BEFORE M3B package validation, before adapter.open_e8_arm_bank(), and before any M9Trainer import/construction
- build_runner_rule_payload_v2() now binds BOTH source_binding_correction_artifact (path) AND source_binding_correction_sha256 (identity), plus explicit historical_v1_runner_rule_identity and historical_v2_runner_rule_identity fields
- because the payload's content changed, runner_rule_identity_v2() now computes a NEW identity (9dd689dfa013f75a5641f566493216718f7a8bfc6b617b06250b63d1cb3e68db); the prior value (3293994d312be82969fba884da5b7d13445aa6c1a9d43742f21434991c1b5570) is retained only as a recorded historical fact, never recomputed from this function again

## Does NOT change

- the 15 frozen run IDs
- the 5 detector seeds
- the M3B content identity (08d9...)
- the E7-D source-support identity (955b...)
- the 818/354/464 E8 bank counts
- the 1575 optimizer updates / 10800 synthetic draws schedule
- the target firewall
- configs/models/m9_detector.yaml (explicitly out of scope; see note below)

## `configs/models/m9_detector.yaml` -- explicitly out of scope

The stale 'b1cf...' pin in configs/models/m9_detector.yaml has been separately audited: it mismatches the current M3B content identity (08d9...); verify_pinned_identities() has no production caller in the current E8 path; the E8 dataset construction uses the actual canonical package identity directly (M3B_CONTENT_IDENTITY), not that stale pin. Therefore it is out of scope for this provenance fix and was not modified.

## Flags

`scientific_protocol_changed=false`, `train_dev_membership_changed=false`, `synthetic_membership_changed=false`, `quality_rule_changed=false`, `hyperparameters_changed=false`, `target_access=false`, `target_labels_accessed=false`, `training_performed=false`, `smoke_performed=false`, `gpu_used=false`, `llm_calls=0`.

See `E8_TRAINING_RUNNER_V2_1_BINDING.json` / `E8_TRAINING_RUNNER_V2_1_PREFLIGHT.{json,md}` for the full V2.1 report.
