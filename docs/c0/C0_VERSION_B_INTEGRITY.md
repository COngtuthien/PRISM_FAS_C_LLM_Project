# C0 — Version-B integrity record

Milestone C0. Version B was opened **for reading only**. Nothing in
`PRISM_FAS_B_Project` was committed, pushed, reset, deleted, retrained, recalibrated,
re-tagged or rewritten. G7 and G8 were not rerun, and no Version-B label was reopened.

The machine-readable form of this document is
[`reports/c0/VERSION_B_INTEGRITY_SNAPSHOT.json`](../../reports/c0/VERSION_B_INTEGRITY_SNAPSHOT.json),
regenerable with [`scripts/c0_fingerprint_version_b.py`](../../scripts/c0_fingerprint_version_b.py).
That script only reads.

## 1. Repository state

| Field | Value |
|---|---|
| Version-B repository | `D:\AI on IOT\Anti_spoofing\PRISM_FAS_B_Project` |
| `git status --short` | *(empty — clean working tree)* |
| `HEAD` | `7799f7decd35db6987ce4578824e5bd8d9eab4ae` |
| `main` | `7799f7decd35db6987ce4578824e5bd8d9eab4ae` |
| `origin/main` | `7799f7decd35db6987ce4578824e5bd8d9eab4ae` |
| `m10-blind-evaluation-checkpoint^{}` | `7799f7decd35db6987ce4578824e5bd8d9eab4ae` |
| expected checkpoint | `7799f7decd35db6987ce4578824e5bd8d9eab4ae` |
| match | **YES** — HEAD, `main`, `origin/main` and the peeled tag all agree |
| `origin` | `https://github.com/COngtuthien/PRISM_FAS_B_Project.git` |
| `git log -1` | `7799f7d (HEAD -> main, tag: m10-blind-evaluation-checkpoint, origin/main, ...) docs: reconcile final M10 project state` |

Tags present: `m2-full-preprocessing-v2`, `m3-prism-data-v1-m3b`, `m4-canonical-loader-sampler`,
`m5-b00-local-baseline`, `m6-modal-parity-checkpoint`, `m7-recipe-physics-checkpoint`,
`m8-gpat-synthetic-bank-checkpoint`, `m9-regional-detector-checkpoint`,
`m10-blind-evaluation-checkpoint`.

Version C was created with `git clone --no-hardlinks`, which reads Version B's object
store and writes only into the new directory. `--no-hardlinks` was used deliberately so
the two object stores share no files on disk and no future operation in Version C can
reach Version-B storage.

## 2. Frozen scientific artifact identities

Every identity below was **read from the actual repository artifacts**, not assumed.
Nothing is marked `NOT FOUND`: all expected artifacts resolved.

### 2.1 Frozen source package (`PACKAGE_LOCK`)

| Field | Value |
|---|---|
| package id | `prism_data_v1_m3b` |
| package identity | `b1cf29b69a165ed5d9e074fc8127c17fbf057723edf9e272048ec3a564eb9dc6` |
| lock file | `data/processed/prism_data_v1_m3b/PACKAGE_LOCK.json` (5 781 B) |
| status | `validated` |

Source of the identity: `reports/m10/SOURCE_MATRIX_LOCK.json → frozen_inputs`.

### 2.2 M7 recipe bank

| Field | Value |
|---|---|
| bank id | `prism_recipe_bank_m7_v1` |
| bank content identity | `fa989938cafdc4887518cc45c35d559d00278358439dc68c2486da10309210cb` |
| recipe count | **128** |
| status | `frozen` |
| bank seed | `20260806` |
| ontology version / SHA256 | `m7-ontology-v1` / `90694441c2ef1477ca8f6c4dd724a4997a3e166cbf5a067d52c101892f952bbd` |
| prompt SHA256 | `6181410db246d35c35c1a8726154a196b5c46fe98f8ea4dbc75905217cf7af02` |
| compiler version | `m7-compiler-v1` |
| conditioning | `recipe_conditioning_v1`, dimension **41** |
| generator provider | `deterministic_local` |
| generator model id | `deterministic-source-only-recipe-generator` |
| generator revision | `m7-v1` |
| **`external_llm_invoked`** | **`false`** |

The generator fields are the primary evidence for
[`C0_LLM_GAP_ANALYSIS.md`](C0_LLM_GAP_ANALYSIS.md).

### 2.3 M8 synthetic bank

| Field | Value |
|---|---|
| bank id | `prism_synthetic_bank_m8_v3_e84c78cd2a9b` |
| bank identity | `e84c78cd2a9b548244e243de0380998d04bc6770b91caf32ac7be96f489bb542` |
| lock file | `data/processed/prism_synthetic_bank_m8_v3_e84c78cd2a9b/BANK_LOCK.json` (7 598 B) |
| A02 random-operator recipe bank | `9351d08ac824cc67021445d1bb59bd9dc14ef7eb3dfa606414500d8fac49603f` |
| A02 random-operator synthetic bank | `f7f1e6ac20341d32d75dddd19cbf3231ea4eb7554eb49290aee32cf59ec17387` |
| A02 conditioning control | `28b96d0122f6e2493cc1daa6534d7d7e60fdb63a9071d7422f262ee41a1a8140` |

The three A02 identities matter to Version C: they are the artifacts of the Version-B
random-recipe control whose GPAT conditioning was out of distribution. Version C removes
that confound (see [`C0_FROZEN_DESIGN_DECISIONS.md`](C0_FROZEN_DESIGN_DECISIONS.md) §6).

### 2.4 M9 reference artifacts

| Field | Value |
|---|---|
| acceptance file | `reports/m9/M9_ACCEPTANCE.json` |
| `target_test_opened` | `false` |
| recipe-text cache identity | `10f4ec35b7563b2b658cacc94599d35b9f93b531963a065459d4694d5dc2c141` |
| SigLIP2 identity | `7e059e40dcc34913b51fc8d7bd25e6f0c023bc238261effee9bfb87b33f04822` |
| ConvNeXt weight SHA256 | `6389c2f5a427b01a922e66e6d352c707424cccb62390c6936bc612e3d10b7ebb` |
| M9↔M10 reference binding accepted | `true` |

### 2.5 M10 experiment matrix

| Field | Value |
|---|---|
| `m10_matrix_identity` | `a4972b0dc23946c4ad169f2c856fc9b5e0387baca45b2c9a4895f8180d9c2dd5` |
| `registry_identity` | `cfc08d8d19fd4f3d9fd36ceb0f679cfb3d6afb017922f3eaee275cc0331666c6` |
| logical rows | 42 |
| executable rows | 38 |
| blocked rows | 4 (each carrying a reason) |
| failed rows | 0 |
| acceptance | `passed: true`, `failed_checks: []` |
| acceptance identity | `4f17a23046e456f37dc96397f2250a8ab4628fee225ba9a7785859586e33d6d2` |

### 2.6 `SOURCE_MATRIX_LOCK`

| Field | Value |
|---|---|
| identity | `c06944344eab25820b4bf6327b9dd391a308a3ffd935ab2ed91264e5898517aa` |
| schema version | `m10-source-matrix-lock-v2` |
| `selection_used_target` | `false` |
| `target_labels_opened` (at lock time) | `false` |
| selection rule | metric `source_dev/acer`, tie-break `source_dev/bpcer`, calibration `source_dev/nll`, threshold `min_acer_on_calibrated_source_dev` |

Frozen inputs recorded inside the lock:

```
source_package_id          prism_data_v1_m3b
source_package_identity    b1cf29b69a165ed5d9e074fc8127c17fbf057723edf9e272048ec3a564eb9dc6
m7_recipe_bank_identity    fa989938cafdc4887518cc45c35d559d00278358439dc68c2486da10309210cb
m8_bank_id                 prism_synthetic_bank_m8_v3_e84c78cd2a9b
m8_bank_identity           e84c78cd2a9b548244e243de0380998d04bc6770b91caf32ac7be96f489bb542
recipe_text_cache_identity 10f4ec35b7563b2b658cacc94599d35b9f93b531963a065459d4694d5dc2c141
siglip2_identity           7e059e40dcc34913b51fc8d7bd25e6f0c023bc238261effee9bfb87b33f04822
convnext_weight_sha256     6389c2f5a427b01a922e66e6d352c707424cccb62390c6936bc612e3d10b7ebb
```

### 2.7 Final target package identity

| Field | Value |
|---|---|
| target feature package identity | `c3a29e695ad08c4b31e01533f1d12374f4e30c51f0167c6622cf8168792e48a8` |
| target package lock | `data/processed/prism_target_eval_v2/PACKAGE_LOCK.json` |
| target label lock | `data/evaluation_only/prism_target_v2_labels/TARGET_LABEL_LOCK.json` |
| `TARGET_PREDICTION_LOCKSET` identity | `34a28858457933df2e400aea8a3f75dd3e9883c7a15d53ffe2f8d31104cd6b8a` |
| lockset status / entries | `FROZEN` / 37 |
| target label reveal identity | `436fda25254f77d63878edb9ff26d942b92959f8395942fe64b4f0b049fb935f` |

**This is the artifact that makes Version-C P3 procedural rather than blind.** The
Version-B reveal record exists and is permanent. Version C inherits a *known* target and
says so.

### 2.8 Final scientific checkpoint

| Field | Value |
|---|---|
| commit | `7799f7decd35db6987ce4578824e5bd8d9eab4ae` |
| tag | `m10-blind-evaluation-checkpoint` (annotated) |
| peeled commit | `7799f7decd35db6987ce4578824e5bd8d9eab4ae` |

### 2.9 Version-B test baseline

| Field | Value |
|---|---|
| command | `python -m pytest -q` |
| passed | **1081** |
| failed | **0** |
| skipped | **0** |
| errors | 0 |
| raw | `1081 passed, 1 warning in 385.04s (0:06:25)` |

Recorded at Version-B closure in `reports/m10/TEST_SUITE.json`. Version C inherits this
baseline; its own count may exceed 1081 as C-milestone tests are added, but the failure
and unexpected-skip counts must stay at 0.

## 3. Post-creation re-verification

After the Version-C repository was created, Version B was re-checked:

- `git status --short` → empty;
- `HEAD` → `7799f7decd35db6987ce4578824e5bd8d9eab4ae`;
- `m10-blind-evaluation-checkpoint^{}` → `7799f7decd35db6987ce4578824e5bd8d9eab4ae`;
- `origin` → `https://github.com/COngtuthien/PRISM_FAS_B_Project.git`.

Unchanged. Changed scientific artifacts: **0**.

## 4. Standing rule for the rest of Version C

Version B is referenced by **identity**, never by mutation. Version-C code must not write
into the Version-B tree, must not use an absolute output path inside it, and must never
push to its remote. The Version-C clone keeps a read-only informational remote
`version-b-readonly` whose push URL is set to the non-URL sentinel
`DISABLED_NO_PUSH_TO_VERSION_B`, so an accidental `git push version-b-readonly` fails
instead of reaching Version B.
