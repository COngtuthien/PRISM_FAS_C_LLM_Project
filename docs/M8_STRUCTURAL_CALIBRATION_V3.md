# M8 structural robustness calibration v3

Frozen **before** any candidate is re-evaluated under it. Source-only throughout: this protocol
opens `manifests/source_train.parquet`, `source_train` **live** images and their stored priors, and
nothing else. It never opens `source_dev`, `target_test`, a target label, a target shard or a raw
dataset path, and it never reads a generated candidate.

`configs/synthesis/quality_gate_m8.yaml` (v1) and `configs/synthesis/quality_gate_m8_v2.yaml` (v2)
are **unmodified**, and every v1 and v2 report, lock, namespace, archive and bank is immutable.

**This is the final planned calibration revision for M8.** If the v3 bank still misses an operational
minimum after a correct implementation, there is no v4: the milestone is reported as unable to meet
its predeclared operational specification. See §11.

---

## 1. Why v3 exists

Two thresholds are still fitted from the v1 benign population — the **same image** under

- brightness x{0.98, 1.02}
- contrast x{0.98, 1.02}
- additive Gaussian noise, std 0.002

That population measures how far SCRFD's landmarks and FaceXFormer's parsing move when a picture is
brightened by 2 %. It is a near-identity photometric nudge. It does **not** characterize stability
under a **non-geometric, identity-preserving local face appearance edit**, which is what a synthetic
candidate actually is.

`tau_id` carried exactly this defect and was versioned in v2. The retained v2 run then measured the
consequence for the two thresholds v2 left alone: of 645 rejections, `landmark` accounts for **467**
and `parsing_dice` for **233**, while `identity` accounts for **0**.

The implementation audit behind those rejections found **no defect** in coordinate transforms, mask
semantics, SCRFD preprocessing, FaceXFormer preprocessing, AdaFace preprocessing, discrete uint8
compositing, outside-mask equality, generation determinism or payload binding. `support_overlap` is
exactly 1.0 on all 1120 candidates and the outside-mask uint8 error is exactly 0 on every one.

So the remaining blocker is the same **calibration-population** defect v2 removed from `tau_id`, and
v3 removes it from `tau_lm` and `tau_parse` by the same method: a new population, a rule declared
here before any v3 acceptance count exists.

**This is a change of calibration population, not a threshold nudge.** The v2 and v3 acceptance
counts are not an input to any rule below.

## 2. What v3 does NOT change

Frozen and carried forward verbatim:

| item | value / source |
|---|---|
| `tau_fd` | **0.5** — pinned SCRFD production threshold, never fitted |
| `tau_id` | **0.547440037939055** — identity calibration v2, unchanged |
| `tau_out` | **0** — exact, not fitted |
| `tau_fp` | **5.687657785453908** — v1 real-spoof leave-one-record-out |
| fingerprint reference | v1, `c5c09cfa26819e125eafb4640eec6ab02eec5419ae6a83bad9a293ae4c4ebb39` |
| artifact-strength rule | `lower = max(0.01, 0.25a)`, `upper = min(0.50, 1.75a)` |
| support-overlap rule | `>= 0.90` |
| `q` formula | geometric mean of the seven components, eps 1e-6 |
| `recipe_match` | `not_applicable` |
| operational minimums | unchanged, see §11 |

**v3 versions exactly two values: `tau_lm` and `tau_parse`.**

## 3. Cross-record genuine pairs must NOT set these thresholds

The 560 v2 genuine identity pairs are valid for **identity** calibration and are reused here as a
**diagnostic only**.

They must not determine `tau_lm` or `tau_parse`. Two different `source_record_id` values are two
different observations, and can differ in real pose, expression, head-geometry projection, landmark
position, occlusion and crop/alignment. Those are **real scene variation**, not detector jitter
caused by an appearance-only edit of one input. A `tau_lm` read off cross-record landmark NME would
make the geometric gate conceptually far too permissive: it would license a candidate whose landmarks
moved as much as a genuinely different photograph of the same person.

The diagnostic is reported so the size of that gap is visible rather than asserted, and a test
asserts no threshold is derived from it.

## 4. Calibration population

`source_train` **live** images only: **280** samples, CASIA 160 / MSU 120.

Each image is compared against **itself** after a deterministic, appearance-only perturbation. The
geometric content is never warped. No rotation, translation, scaling, affine or perspective
transform, no crop change, no landmark movement, no face replacement.

## 5. Fixed localized benign appearance suite

Exactly **8** calibration variants per live image -> **280 x 8 = 2240** calibration observations.
These are calibration-only generic appearance changes. **No M7 spoof operator, no GPAT, and no
generated synthetic candidate participates.**

| slot | transform | parameter |
|---|---|---|
| 0 | brightness | factor 0.90 |
| 1 | brightness | factor 1.10 |
| 2 | contrast | factor 0.90 |
| 3 | contrast | factor 1.10 |
| 4 | gamma | 0.90 |
| 5 | gamma | 1.10 |
| 6 | additive Gaussian noise | std 0.005 |
| 7 | Gaussian blur | sigma 0.75 |

Definitions, applied to the whole `[3,224,224]` float image in `[0,1]` before masking, so the
transform is a well-defined function of the image alone and is independent of which region it is
later restricted to:

```
brightness(x, f) = x * f
contrast(x, f)   = (x - mean(x)) * f + mean(x)          # mean over the whole image
gamma(x, g)      = clip(x, 0, 1) ** g
noise(x, s)      = x + N(0, s), local PCG64 seeded per observation
blur(x, sigma)   = separable Gaussian, radius ceil(4*sigma) = 3, reflect padding
```

Every transform preserves image dimensions and pixel coordinates and performs no geometric
resampling except the fixed Gaussian filter, which is a fixed symmetric low-pass kernel that moves no
coordinate. All are deterministic; noise draws from a local PCG64 seeded by a SHA-256 digest and
never from global RNG state.

**JPEG is deliberately excluded**, on the same grounds as v1 and v2: `cv2` JPEG encoding is not
provably byte-deterministic across OpenCV builds, so it cannot be proven deterministic here.

**These values are frozen now and are not changed after seeing any calibration or candidate result.**

## 6. Deterministic semantic support

Candidate generation changes selected semantic regions rather than the whole image, so the
calibration must test **localized** appearance edits. The frozen M7 semantic-region machinery and the
`source_train` priors are reused **only to build support masks**; no M7 physics operator runs.

The nine canonical regions are the M7/M8 set: `left_eye`, `right_eye`, `nose`, `mouth`, `forehead`,
`left_cheek`, `right_cheek`, `face_boundary`, `context`.

For each (live image, transform slot):

```
observation_digest = SHA256(
    calibration_version | source_package_identity | sample_id | transform_slot | seed )
region = REGION_ORDER[ int(observation_digest[:16], 16) mod 9 ]
```

The mask is built by the exact `RegionMaskBuilder` used by M7 and M8 — the same parsing-first,
geometry-fallback policy and the same `crop_box` coordinate mapping — at coverage 1.0, so the support
is the whole canonical region.

Over the complete 2240-observation population the protocol **requires** all nine regions represented,
both datasets represented for every transform slot, deterministic region assignments, and no target
information. A shortfall stops the protocol; it is not patched by reassigning regions.

## 7. The calibration edit

For each observation:

```
original_uint8 = to_uint8(source_train live image)                 # the M8 discrete convention
edited_uint8   = where(support, to_uint8(transform(image)), original_uint8)
```

Outside the support the **exact original uint8 pixels** are copied, and the protocol asserts
`outside-support max error == 0` after the final uint8 conversion.

Both metrics are then computed on `from_uint8(...)` of each side, which is exactly what the candidate
evaluator does for its reference and its generated image, so a calibration measurement and a
candidate measurement are directly comparable.

A calibration observation is **not** a synthetic spoof candidate. It carries no spoof label, no
recipe id, no GPAT source and no M7 physics artifact. It is only a structural-stability sample.

## 8. Landmark calibration

The exact pinned SCRFD preprocessing and coordinate contract already used by M8: input 320,
threshold 0.5 (`scrfd_source_policy_v1`), best-scoring detection, landmarks in crop pixel
coordinates, the same validated crop mapping.

```
landmark_nme = mean(|| L_edited - L_original ||) / max(inter_ocular(L_original), 1e-6)
```

— the same `landmark_nme` function, the same inter-ocular normalization, and the **same image**
before and after. Cross-record ground-truth landmark differences are never used.

**Predeclared threshold:**

```
tau_lm_v3 = 99th percentile of the valid structural-calibration NME values
```

The percentile is fixed now and is not changed after seeing the result.

Detector failures are recorded **separately** and are never silently discarded because their NME
would be large. An observation where either side yields no detection has no defined NME and is
excluded from the percentile pool with an explicit count. The protocol **stops** if fewer than 95 %
of observations produce a valid landmark comparison, because a threshold fitted on a heavily censored
population would not describe the population it claims to.

Reported: total observations, valid landmark comparisons, face-detection failures, NME min, p50, p90,
p95, p99, max, and per-dataset, per-transform and per-region summaries.

## 9. Parsing calibration

The exact pinned FaceXFormer implementation and preprocessing already used by M8: RGB 224, ImageNet
mean/std, bicubic, task token 0, LaPa 11 classes.

```
parse_original = FaceXFormer(original)
parse_edited   = FaceXFormer(edited)
outside_support_parsing_dice = parsing_dice(parse_edited, parse_original, ~support)
```

— the same `parsing_dice` function and the same outside-mask semantics as the candidate gate. No
different surrogate metric is substituted.

**Predeclared threshold:**

```
tau_parse_v3 = 1st percentile of the valid outside-support parsing Dice values
```

The percentile is fixed now and is not altered after results.

**Declared asymmetry, measured not assumed.** The candidate gate scores Dice outside the *exact
changed-pixel* mask, which is a subset of the requested support, so the gate's outside region also
contains the in-support pixels that survived quantization unchanged. Those pixels agree by
construction and can only raise a Dice score. Scoring the calibration outside the *support* is
therefore the conservative side of that difference. The outside-exact variant is computed and
reported alongside as a labelled diagnostic so the size of the gap is a measurement, not a claim.
`tau_parse_v3` comes from the outside-support values.

Reported: observation count, valid comparisons, Dice min, p01, p05, p50, max, and per-dataset,
per-transform and per-region summaries.

## 10. Freeze, artifacts and determinism

Frozen threshold set:

```
tau_id    = 0.547440037939055        (identity calibration v2, unchanged)
tau_lm    = tau_lm_v3
tau_parse = tau_parse_v3
tau_fd    = 0.5                      (unchanged)
tau_out   = 0                        (unchanged)
tau_fp    = 5.687657785453908        (unchanged)
```

Git-ignored artifacts under `reports/m8/`:

```
structural_calibration_v3.parquet        per-observation logical rows and metrics
structural_calibration_v3_summary.json   distributions and thresholds
quality_calibration_v3.json              full calibration report
STRUCTURAL_CALIBRATION_V3_LOCK.json      the binding lock
```

The lock binds the calibration version and schema, the source-package identity, the `source_train`
logical population identity, the SCRFD identity/SHA, the FaceXFormer identity/revision/SHA, both
preprocessing-contract identities, the v3 config hash, the seed, the deterministic transform suite
and its parameters, the semantic-region assignment identity, the observation count, `tau_lm_v3`,
`tau_parse_v3`, the inherited `tau_id_v2`, the inherited `tau_fd`/`tau_out`/`tau_fp`, the threshold
hash, the source-isolation evidence and the calibration content identity.

**Excluded from every identity:** machine names, absolute paths, Modal ids, timestamps and physical
Parquet bytes. Identities are hashed over **canonical logical rows**, so they are portable across
pyarrow versions — the same rule the pair plan, the candidate plan and identity calibration v2 use.

**Determinism.** Calibration runs **twice** and refuses to write unless both runs agree on: the same
2240 logical observations, the same transform assignment, the same region assignment, the same noise
samples, the same logical observation identity, the same metric values within the declared tolerance,
the same `tau_lm_v3`, the same `tau_parse_v3`, the same threshold SHA and the same calibration
content identity. **Required mismatch count: 0.**

The declared numeric tolerance is **1e-6** on per-observation metric values. Two runs on one device
are expected to be bit-identical; the tolerance exists so a real device change is *reported* rather
than silently absorbed. The derived thresholds, the threshold SHA and the content identity must match
**exactly** — a tolerance never applies to them, and the tolerance is never widened after seeing a
result.

## 11. Operational minimums and the stop rule

Unchanged, and never re-declared after a v3 acceptance count is known:

```
candidates == 1120, accepted total >= 400, accepted physics >= 200, accepted gpat >= 100,
accepted CASIA live targets >= 100, accepted MSU live targets >= 100,
all 8 artifact types, all 9 semantic regions, both gpat domain relations
```

If the v3 bank still fails any operational minimum, then: the v3 run and every report are **retained**;
no v4 quality calibration is created; no threshold is lowered; `accepted_physics` is not redefined;
the physics route is not removed; nothing is resampled; `source_dev` and `target_test` stay unopened;
and **M8 is not marked complete**. The reported outcome is that M8 cannot satisfy the current
predeclared operational specification under the validated source-only calibration protocols, and the
milestone stops for an explicit project-level decision.

Calibration populations are **not** changed again until a minimum passes.
