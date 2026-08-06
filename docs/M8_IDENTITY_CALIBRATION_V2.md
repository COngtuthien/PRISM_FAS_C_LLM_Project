# M8 identity calibration v2

Frozen **before** any candidate was re-evaluated. Source-only throughout: this protocol opens
`manifests/source_train.parquet` and `source_train` live images, and nothing else. It never opens
`source_dev`, `target_test`, a target label, a target shard or a raw dataset path, and it never reads
a generated candidate.

`configs/synthesis/quality_gate_m8.yaml` and every v1 artifact are **unmodified**.

---

## 1. Why v2 exists

v1 fitted `tau_id` as the 1st percentile of the identity cosine between an image and **the same image**
under a mild photometric nudge:

- brightness x{0.98, 1.02}
- contrast x{0.98, 1.02}
- additive Gaussian noise, std 0.002

That population measures how much an AdaFace embedding moves when a picture is brightened by 2 %. It
does **not** model how the embedding moves between two genuinely different observations of the same
person, which is the quantity an "identity is preserved" gate is really asking about. The resulting
`tau_id_v1 = 0.9995203357934952` is therefore extremely tight, and the retained v1 run measured the
consequence: 483 of 729 rejections cited `identity`, and the physics route kept only 71 of 560
candidates.

**This is a change of calibration population, not a threshold nudge.** The v2 threshold is produced by
a rule declared here before any v2 acceptance count exists, and the v1 acceptance counts are not an
input to it.

## 2. Populations

Both populations are drawn from `source_train` **live** samples only.

**Identity key** — `dataset + "::" + normalized_subject_id`, where the subject id is stripped and
case-folded. CASIA subject `5` and MSU subject `5` are two different people; the dataset prefix is part
of the key, not decoration.

| population | rule |
|---|---|
| genuine | same dataset, same identity key, different `sample_id`, **different `source_record_id`** |
| impostor | same dataset, **different** identity key, different `sample_id` |

Two frames of one canonical record are two views of one observation, so a same-record pair understates
real variation and is excluded from the genuine population rather than down-weighted.

Impostor pairs are **same-dataset only**. A CASIA/MSU pair differs in capture domain as well as
identity, which would make the impostor problem artificially easy and push `tau_impostor` down.

## 3. Determinism and balance

Fixed seed **20260806**. Every pair is unordered and unique; `sample_id_a < sample_id_b`.

Genuine pairs are ordered per identity by

```
sha256(calibration_version | seed | dataset | normalized_subject_id |
       min(sample_id_a, sample_id_b) | max(sample_id_a, sample_id_b))
```

and capped at **20 per identity**, so an identity with many samples cannot dominate.

Impostor pairs are ordered by

```
sha256(calibration_version | seed | dataset | identity_key_hash_a | identity_key_hash_b |
       sample_id_a | sample_id_b)
```

Each unordered impostor pair is filed under exactly one identity (the one with the smaller identity-key
hash) and selection is **round-robin over identities**, so no identity dominates the tail that the
99.9th percentile is read from. Totals are capped at **20000** and balanced across CASIA and MSU as
closely as availability permits.

Pair-plan identities are computed over **canonical logical rows**, never over parquet bytes, so they are
portable across pyarrow versions — the same rule the M8 pair plan and candidate plan already use.

## 4. Embeddings

The **exact pinned M8 production wrapper** is reused:
`prism_fas.synthesis.quality_models.DifferentiableAdaFace` — RGB->BGR flip, bicubic resize to 112,
`(x - 0.5) / 0.5`, L2-normalized 512-d output — against AdaFace IR-50 WebFace4M, revision
`60a65befbcf7`, weight SHA-256
`43bd2d570584d95d4a17ce81f26449034c45dbeed750afcab651872abc0e1496`.

No alternate resize, alignment or normalization path is introduced, so a v2 cosine is directly
comparable to a candidate's `identity_cosine`. The wrapper's semantics are hashed into
`preprocessing_contract_identity_sha256` and bound in the lock.

Each unique image is embedded **once**, cached by the processed image **content identity**
(`array_hash` of the loaded float tensor) — never by an absolute path. Every embedding is asserted
finite and L2-normalized.

## 5. Threshold rule (declared before any result)

```
tau_genuine  = 1st percentile of the genuine same-identity cosines
tau_impostor = 99.9th percentile of the impostor different-identity cosines
tau_id_v2    = max(tau_genuine, tau_impostor)
```

Acceptance is unchanged: `identity_cosine >= tau_id_v2`.

The `max` is a **floor, not a choice**: it holds the impostor false-match rate at or below 0.1 % even
when the genuine distribution is loose. It is applied verbatim — no manual rounding, no search over
candidate acceptance outcomes, no "whichever of v1 and v2 is lower", no percentile changed after seeing
the value.

If `tau_impostor >= tau_genuine` the two distributions overlap at these percentiles. That is reported
truthfully as `distributions_are_well_separated: false` together with the fraction of the genuine
population at or below `tau_impostor`; it is not disguised.

## 6. What v2 does NOT change

`tau_fd`, `tau_lm`, `tau_parse`, `tau_out`, `tau_fp`, the fingerprint reference, the artifact-strength
bounds, the support-overlap rule, the `q` formula and `recipe_match = not_applicable` are all carried
over from v1 **unchanged**. The operational minimums are unchanged:

```
candidates == 1120, accepted total >= 400, accepted physics >= 200, accepted gpat >= 100,
accepted CASIA >= 100, accepted MSU >= 100, all 8 artifact types, all 9 regions,
both gpat domain relations
```

They are never re-declared after a v2 acceptance count is known.

## 7. Artifacts

Git-ignored, under `reports/m8/`:

```
identity_structure_v2.json               structure audit; the go/no-go for the protocol
identity_genuine_pairs_v2.parquet        privacy-safe rows + measured cosine
identity_impostor_pairs_v2.parquet       privacy-safe rows + measured cosine
quality_calibration_v2.json              full calibration report
identity_calibration_v2_summary.json     distributions and thresholds
IDENTITY_CALIBRATION_V2_LOCK.json        the binding lock
```

No pair row carries a raw filename, an absolute path or a plaintext subject id: identities appear only
as `identity_key_hash`.

The lock binds the calibration schema/version, the source-package identity, the source population hash,
the AdaFace id/revision/SHA, the preprocessing-contract identity, the v2 config hash, the seed, both
logical pair-plan identities, both pair counts, `tau_genuine`, `tau_impostor`, `tau_id_v2`, the
unchanged gate thresholds, the full threshold SHA, the source-isolation evidence, and the calibration
content identity.

**Excluded from the content identity**: absolute paths, machine name, Modal ids, timestamps, temporary
directories and physical parquet bytes.

## 8. Reproducibility

Calibration is run **twice**. Required: identical logical pair rows, identical pair-plan identities,
identical threshold values under the declared serialization, identical calibration content identity, and
cosines equal within a declared tolerance of **1e-6** (two runs on one device are expected to be
bit-identical; the tolerance exists so a real device change is reported rather than silently absorbed).

## 9. Minimum data requirements

The protocol refuses to proceed unless `source_train` really supports it:

- both CASIA and MSU represented
- at least 10 valid identities overall
- at least 100 valid genuine pairs overall
- every genuine pair uses two distinct `source_record_id` values

If any requirement fails, the run stops truthfully and M8 stays incomplete. Falling back to same-record
frames would reintroduce exactly the defect v2 exists to remove and is not permitted without a new
explicit decision.
