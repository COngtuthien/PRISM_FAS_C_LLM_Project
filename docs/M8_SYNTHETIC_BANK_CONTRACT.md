# M8 synthetic bank contract

Frozen **before** generation. Source-only throughout.

---

## 1. Candidate plan

Fixed bank seed **20260806**. Every one of the **280** `source_train` live samples is used.

| route | per live sample | total |
|---|---|---|
| physics | 2 | 560 |
| gpat | 2 | 560 |
| **all** | 4 | **1120** |

M7's `configs/synthesis/physics_m7.yaml` (`recipes_per_sample: 2`) is **not** modified — it remains
the immutable 64-preview regression contract. M8 uses its own field in
`configs/synthesis/synthetic_bank_m8.yaml`:

```yaml
candidate_recipes_per_live:
  physics: 2
  gpat: 2
```

Recipe assignment is deterministic from `(candidate seed, live sample id, route)` and coverage-aware
over the frozen 128-recipe bank; no target information participates.

GPAT spoof-source assignment per live sample: one **same-domain** and one **cross-domain** spoof
source, different `source_record_id`, different explicit `subject_id` when both are available,
deterministic.

```
synthetic_id = "syn_" + sha256(
    parent_package_identity | recipe_bank_identity | route | live_target_sample_id |
    (spoof_source_sample_id or "physics-none") | recipe_id | candidate_seed |
    (gpat_checkpoint_sha or physics_engine_version)
)[:24]
```

A `synthetic_id` contains no raw path, filename, subject, target token or timestamp.

Plan artifacts (git-ignored): `reports/m8/candidate_plan.parquet`,
`reports/m8/CANDIDATE_PLAN_LOCK.json`. A plan rerun is byte-identical.

## 2. Generation

- physics route: the **exact final M7 `PhysicsEngine`**, no GPAT model — 560 candidates.
- gpat route: the frozen **best** M8 GPAT checkpoint — 560 candidates.
- Every candidate is quality-evaluated. Generation does not stop when enough candidates are
  accepted, and failed candidates are never resampled to inflate the accepted count.
- Each planned candidate is recorded exactly once as `accepted`, `rejected` or `failed_generation`.
  Failures record stage, reason, exception type and candidate id — never a raw path.

Resume: each candidate output is written atomically; an existing result is reused only when all
input hashes match; a mismatched partial output is rebuilt; no duplicate ids; no silent overwrite.

## 3. Discrete output and the exact mask

```
1. generated float -> uint8 (round-half-up, clip 0..255)
2. original live   -> uint8 with the identical convention
3. composite outside the binary support using the ORIGINAL uint8 pixels
4. exact_edit_mask = any-channel uint8 difference != 0
5. re-composite outside exact_edit_mask
6. write PNG
7. decode the written PNG
8. re-verify max|decoded - original_uint8| outside exact_edit_mask == 0
```

- image: RGB PNG, 224x224, uint8, lossless
- mask: PNG uint8, values only `{0,255}`, the **exact changed pixels**
- artifact map: compressed NPZ, key `artifact_map`, float16 `[1,224,224]`, finite, `[0,1]`, exactly
  zero outside the exact mask
- the requested support mask is preserved separately in the metadata (pixel count and hash), so the
  requested region is recoverable even though the saved mask is the exact one

## 4. Operational minimums (pre-declared)

```
candidates                    == 1120
accepted total                >= 400
accepted physics              >= 200
accepted gpat                 >= 100
accepted live-target CASIA    >= 100
accepted live-target MSU      >= 100
accepted covers all 8 artifact types
accepted covers all 9 semantic regions
accepted contains both same-domain and cross-domain GPAT pairs
```

Thresholds are never lowered after seeing the counts. A justified config change creates a new run and
config identity, keeps the failed run's report and is documented before the rerun.

## 5. Layout

```
prism_synthetic_bank_m8_v1_<short_identity>/
├── images/<synthetic_id>.png
├── artifact_maps/<synthetic_id>.npz
├── masks/<synthetic_id>.png
├── metadata/<synthetic_id>.json
├── manifests/{candidate_manifest,manifest,rejected,failures,
│              pair_manifest_train,pair_manifest_validation}.parquet
├── calibration/quality_gate.json
├── shards/synthetic-#####.tar
├── shards_index.parquet
├── quality_summary.json
├── generation_summary.json
└── BANK_LOCK.json
```

Accepted manifest columns: `synthetic_id, route, live_target_sample_id, spoof_source_sample_id`
(null for physics)`, live_target_dataset, spoof_source_dataset` (null)`, recipe_id, recipe_hash,
graph_hash, gpat_checkpoint_sha256` (null)`, image_relative_path, image_sha256, mask_relative_path,
mask_sha256, artifact_map_relative_path, artifact_map_sha256, q,` every quality metric`,
threshold_hash, exact_mask_pixels, requested_region_pixels, requested_coverage, achieved_coverage,
package_identity, recipe_bank_identity, generation_config_hash`.

Never present: raw paths, target metadata, target labels, source filenames, absolute paths.

## 6. Shards

Accepted samples only, sorted by `synthetic_id`, at most **500** per shard, **uncompressed** tar.
Fixed member metadata: `uid=gid=0`, `uname=gname=""`, `mode=0o644`, `mtime=0`, deterministic member
order. Each sample contributes `<id>.png`, `<id>.mask.png`, `<id>.npz`, `<id>.json`.

`shards_index.parquet`: shard name, row count, byte size, SHA-256, first/last `synthetic_id`, route
and domain counts. Loose and shard loaders must return equivalent values for audited samples.

## 7. BANK_LOCK

Contains: schema version, bank id, `status = validated`, parent source-package identity, frozen
recipe-bank identity, pair-plan identity, candidate-plan identity, GPAT architecture hash, GPAT
checkpoint SHA, quality-calibration SHA, generation-config SHA, seed, candidate/accepted/rejected/
failed counts, route counts, domain counts, artifact and region coverage, manifest hashes,
quality-summary hash, shards-index hash, every shard SHA and row count, generator versions, pinned
quality model revisions and SHAs, and the bank content identity.

**Excluded from the content identity**: absolute paths, machine name, timestamps, Modal App id,
temporary directories, workspace account name. A timestamp may appear only as informational metadata
outside the identity.

## 8. Validation

`reports/m8/synthetic_bank_validation.json` records: lock parses; `status == validated`; parent
identities match; candidates == 1120; accepted + rejected + failed == 1120; no duplicate ids; every
accepted file exists with a matching hash; every PNG decodes RGB 224x224; masks decode with only
`{0,255}`; artifact maps load with `allow_pickle=False`, finite, in `[0,1]`, zero outside the exact
mask; saved-image outside-mask error exactly 0; `q` finite in `[0,1]`; every accepted row passes every
hard gate; every rejected row names at least one failed gate; no target or private fields;
source-only identities; shards complete; loose/shard parity; operational minimums; package and
recipe bank unchanged.

## 9. Transport

Working remote path: `prism-fas-b-runs:/runs/synthetic_banks/m8_work/`. After validation the bank is
frozen under a versioned path on `prism-fas-b-data:/synthetic_banks/<bank_id>/`; an existing
*different* bank is never overwritten.

Because a Windows directory download failed in M6, a deterministic single-file export archive is
written to `/runs/exports/<bank_id>.tar` and that one file is downloaded, then extracted to
`data/processed/<bank_id>/` and validated locally. **The export archive is transport-only and is not
part of the BANK_LOCK identity.**
