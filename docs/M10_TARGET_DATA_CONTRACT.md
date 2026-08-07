# M10 target evaluation data contract

Frozen **before** any target label is opened and before any M10 experiment runs.

This contract exists because of a defect found at the start of M10, not because of a
design preference. It is written first so the data foundation is fixed before any
result can influence it.

---

## 1. The defect this fixes

`configs/data/siw_mv2.yaml` (v1) globs `Spoof/*.{avi,mov,mp4}` — one level deep. The
real SiW-Mv2 tree nests every attack one level deeper, `Spoof/<Family>/<stem>_<N>.mov`.
The v1 rule therefore matched **zero** spoof videos.

Proved by recomputing the adapter's opaque ids from the raw tree and joining them
against the frozen manifest:

```
Live files matched by "Live/*"                       785
Spoof files matched by "Spoof/*"                       0
Spoof files one level deeper (never ingested)        915
prism_data_v1_m3b target_test unique record ids      785    frames 3140
Live-derived opaque ids found inside target_test     785
target_test ids NOT explained by Live/                 0
```

So `prism_data_v1_m3b`'s `target_test` split is **785 videos, all LIVE, zero spoof**.
On that package APCER is undefined (there is no attack to misclassify), and ACER,
HTER, ROC-AUC and EER are undefined or degenerate. Only BPCER is computable.

Separately, `evaluation_only/siw_target_labels.parquet` — the label manifest spec
Table 24 scores against — was never built anywhere in the project.

## 2. What is NOT changed

`prism_data_v1_m3b` is immutable and is **not touched**. Its content identity
`b1cf29b69a165ed5d9e074fc8127c17fbf057723edf9e272048ec3a564eb9dc6` is bound into the
M8 bank lock and into every M9 checkpoint identity guard; changing it would invalidate
the entire M8/M9 chain. `configs/data/siw_mv2.yaml` is likewise left exactly as it is,
as the historical record of what v1 ingested.

The fix is **additive**: a separate, versioned, immutable target-only artifact.

## 3. The two artifacts, physically separated

```
data/processed/prism_target_v2_siwmv2_full/        FEATURES — label-free
  TARGET_PACKAGE_LOCK.json
  manifests/target_test_features.parquet
  images/  priors/  shards/

data/evaluation_only/prism_target_v2_labels/       LABELS — evaluator-only
  TARGET_LABEL_LOCK.json
  siw_target_labels.parquet
```

They are separate directory trees, separately locked, and separately mounted. The
feature package carries no label column and no attack column; the label artifact
carries no image, no prior and no feature.

## 4. Population

All 1700 SiW-Mv2 videos: **785 live + 915 spoof** across the 14 official attack
families, enumerated with their exact filename stems in
`configs/data/siw_mv2_target_v2.yaml`. The stem is not always the family name
(`Makeup_Cosmetic/Makeup_Co_*`, `Mannequin/Mask_Mann_*`, `Silicone/Mask_Silicone_*`),
so each family's stem is **declared and checked**; an unlisted family or an unexpected
stem is a hard failure, never a silently new class.

| family | videos | | family | videos |
|---|---|---|---|---|
| Partial_FunnyeyeGlasses | 179 | | Partial_Eye | 57 |
| Paper | 135 | | Makeup_Cosmetic | 52 |
| Replay | 98 | | Mannequin | 40 |
| Partial_PaperGlasses | 76 | | Partial_Mouth | 29 |
| Mask_HalfMask | 72 | | Makeup_Obfuscation | 22 |
| Makeup_Impersonation | 61 | | Mask_PaperMask | 17 |
| Mask_TransparentMask | 60 | | Silicone | 17 |

## 5. Join key — opaque ids only

The opaque record id is `siw_<sha256("siw_mv2\0" + dataset-root-relative POSIX path)[:16]>`,
exactly as `adapters.opaque_record_id` computes it. It is the ONLY key used to join a
prediction to a label.

The 785 live ids produced under v2 are **identical** to the 785 in
`prism_data_v1_m3b` — verified, not assumed. That makes the live half of the new
package the same population as the frozen one, and gives a byte-level reproduction
check (§7).

Labels are never joined by filename, path, attack name or folder structure at scoring
time. The directory organization is read exactly ONCE, at ingestion, by the
evaluator-only path, and is converted immediately into an opaque-id-keyed table.

## 6. Preprocessing — the frozen m2-v1 contract, unchanged

Ingestion uses `configs/data/preprocess_m2.yaml` verbatim, so the new crops are
produced by the identical pipeline as the existing ones:

```
preprocessing_version m2-v1     sampling uniform-v1     frames_per_video 4
start/end exclusion 0.05        SCRFD 10g_bnkps @ 320   detection_threshold 0.50
face policy largest_valid_face  crop_padding 0.25       crop_output_size 224
output jpg q95                  hash sha256
```

Priors follow `configs/models/m3b_priors.yaml`: FaceXFormer parsing + pose, derived
9-region visibility. **No identity embedding** — target priors never receive one.

## 7. Required verification before the package is locked

1. Every one of the 1700 paths matches the declared rule; 0 unmatched, 0 undeclared
   family, 0 stem mismatch, 0 duplicate opaque id.
2. Per-family counts equal the declared counts exactly.
3. The 785 live `crop_sha256` values **reproduce `prism_data_v1_m3b` byte-for-byte**.
   This is the strongest available proof that the new package's pipeline is the frozen
   pipeline. A mismatch blocks the package.
4. The feature manifest contains no label-bearing column.
5. Label and feature row sets are identical as sets of opaque ids.
6. Failures are recorded, never silently dropped.

## 8. Firewall

| Stage | May read |
|---|---|
| training (G1/G2/G5/G6) | source package + M8 bank only. Target features **not** mounted. |
| G7 target prediction | target FEATURE package + frozen checkpoint + frozen calibration. Labels **not** mounted. |
| G8 scoring | frozen predictions + label artifact. No optimizer, no checkpoint write. |

The label artifact is never uploaded to `prism-fas-b-data` or any training/G7 Modal
image. It is not referenced by any training or inference config. Enforcement is
structural — a resolvable-path permission check, not a string scan (the M8/M9 lesson:
a string scan flags the proof of isolation as a leak).

Labels may be opened only after `reports/m10/TARGET_PREDICTION_LOCKSET.json` is
frozen, and that transition is recorded once in
`reports/m10/TARGET_LABEL_REVEAL.json`.

## 9. Known constraint this contract does NOT resolve

The frozen frame plan is **4 frames per video** for every split. Table 60's frame-count
ablation asks for 16 / 32 / 48-64. Those densities are not available from the frozen
plan, and re-extracting the SOURCE splits at a different density would change the
source package identity and break the M8/M9 chain.

**Decision, taken before any target label was opened:** this package is built at the
frozen 4-frame plan. That keeps the target population directly comparable to source
training and preserves the byte-for-byte reproduction check of §7.3, which a denser
plan would forfeit — uniform-4 indices are not a subset of uniform-16, so the live
crops would no longer reproduce the frozen package.

The frame-count ablation is therefore recorded as **BLOCKED — declared data not
available at the frozen frame plan**, with this reason, rather than silently
substituted with a different ablation. Whether to build a denser target-side v3
package for that one ablation row is a separate decision, deliberately deferred until
after this package is locked and M10 can produce valid target metrics. It will not be
taken on the basis of any observed target result.
