# M7 region mask mapping

How the nine canonical regions are built from the **real** M3B priors, region by region.
Implementation: [`src/prism_fas/synthesis/masks.py`](../src/prism_fas/synthesis/masks.py)
(`RegionMaskBuilder`, mask policy version `m7-mask-v1`).

The policy is **parsing-first, geometry-fallback**: a region uses its FaceXFormer parsing classes when
they are actually present and large enough, and otherwise falls back to a documented deterministic
landmark/bbox construction. Nothing is silently assumed.

---

## 1. What M3B actually stores

Verified by reading real `priors/*.npz` files from `prism_data_v1_m3b`, not assumed from the spec:

| array | shape / dtype | coordinate space |
|---|---|---|
| `parsing_labels` | `[224,224]` uint8 | **crop pixels** (the 224x224 package image) |
| `landmarks` | `[5,2]` float32 | **original frame pixels** |
| `bbox` | `[4]` float32 | **original frame pixels** |
| `crop_box` | `[4]` float32 | **original frame pixels** — the crop rectangle |
| `frame_width/height`, `crop_width/height` | int32 | frame size; crop is always 224 |

Because landmarks and bbox live in frame space while the parsing mask and the image live in crop
space, a `crop_box` is required to move between them:

```
u = (x_frame - crop_x1) * 224 / (crop_x2 - crop_x1)
v = (y_frame - crop_y1) * 224 / (crop_y2 - crop_y1)
```

`RegionMaskBuilder` applies exactly this mapping. When `crop_box` is `None` the points are treated as
already being in crop space; unit-test fixtures use that form.

The bbox is clipped to the crop after mapping (real priors do contain partly out-of-frame boxes, e.g.
`bbox = [10.2, -7.2, 105.6, 105.3]`), and a bbox that degenerates to fewer than 2 px on a side falls
back to the whole crop.

## 2. Parsing label semantics

FaceXFormer (`kartiknarayan/facexformer`, revision `fd12148d0b19`) emits the 11-class **LaPa**
ordering. M3B verified the ordering empirically from class centroids; M7 re-verified it on 60 real
`source_train` live priors before writing any mask code:

| id | class | measured centroid (x, y) as a fraction of the crop |
|---|---|---|
| 0 | background | 0.446, 0.672 |
| 1 | skin | 0.519, 0.517 |
| 2 | left eyebrow | 0.341, 0.323 |
| 3 | right eyebrow | 0.676, 0.327 |
| 4 | left eye | 0.348, 0.417 |
| 5 | right eye | 0.681, 0.417 |
| 6 | nose | 0.510, 0.525 |
| 7 | upper lip | 0.510, 0.694 |
| 8 | inner mouth | 0.507, 0.727 |
| 9 | lower lip | 0.506, 0.727 |
| 10 | hair | 0.506, 0.205 |

**Left/right are image-space (viewer) sides**, not the subject's own left/right. Class 2/4 sit at
x ≈ 0.34 and class 3/5 at x ≈ 0.68, and SCRFD landmark 0 is likewise the image-left eye. This matches
the existing M3B `region_masks`/visibility convention, so M7 does not introduce a second convention.

`face` = classes 1–9 (skin, brows, eyes, nose, lips). Classes 0 and 10 are outside the face.

### 2.1 The measured gap that makes a fallback necessary

Over 80 real `source_train` live priors:

| class | samples where present |
|---|---|
| 8 (inner mouth) | **25 / 80** |
| 5 (right eye) | **72 / 80** |
| all others | 80 / 80 |

Inner mouth is absent from most samples and one eye can be missing under yaw. A parsing region is
therefore treated as usable only when it has at least `MIN_PARSING_PIXELS = 24` pixels; below that
the documented geometry fallback runs instead. This is a real property of the frozen package, not a
hypothetical.

## 3. Per-region rules

`face_w`, `face_h` are the mapped bbox dimensions; `eye_line` is the top row of classes {2,3,4,5} when
present, otherwise the mean eye-landmark row minus `0.12·face_h`; `mouth_line` is the top row of the
mouth classes, otherwise the mean mouth-landmark row.

| region | parsing classes | geometry fallback | morphology | coverage rule | boundary handling |
|---|---|---|---|---|---|
| `left_eye` | `{4}` | ellipse at landmark 0, radii `0.11·face_w × 0.07·face_h` | none | shared union rule (§4) | ellipse clipped by the crop array bounds |
| `right_eye` | `{5}` | ellipse at landmark 1, same radii | none | shared | as above |
| `nose` | `{6}` | ellipse at landmark 2, radii `0.11·face_w × 0.14·face_h` | none | shared | as above |
| `mouth` | `{7,8,9}` | ellipse spanning landmarks 3–4, radius_x `max(0.75·‖lm3−lm4‖, 0.12·face_w)`, radius_y `0.07·face_h` | none | shared | as above |
| `forehead` | derived: `face ∧ rows ∈ [bbox_top, eye_line)` | ellipse at `(face_cx, bbox_top+0.18·face_h)`, radii `0.36·face_w × 0.14·face_h` | none | shared | row band clipped to the crop |
| `left_cheek` | derived: `face ∧ rows ∈ [eye_line+0.10·face_h, mouth_line+0.05·face_h) ∧ cols < nose_x − 0.06·face_w` | ellipse at `nose_x − 0.22·face_w` | none | shared | band and column test clipped to the crop |
| `right_cheek` | mirror of `left_cheek` (`cols > nose_x + 0.06·face_w`) | ellipse at `nose_x + 0.22·face_w` | none | shared | as above |
| `face_boundary` | none (derived from the face mask) | face mask from bbox ellipse if parsing is unusable | ring `dilate(face, r) ∧ ¬erode(face, r)`, `r = max(2, round(0.04·min(face_w, face_h)))` | shared | erosion treats outside-the-crop as background, so a face touching the crop edge yields a ring there too |
| `context` | none (derived) | `outer_ellipse(0.75·face_w, 0.75·face_h) ∧ ¬face` if the dilation is too small | `dilate(face, R) ∧ ¬face`, `R = max(3, round(0.16·min(face_w, face_h)))` | shared | the dilation is computed on the crop array, so `context` is always inside the valid crop bounds |

Morphology uses the disk-structuring-element `binary_dilate` / `binary_erode` in
[`operators/base.py`](../src/prism_fas/synthesis/operators/base.py) — pure NumPy, so the result cannot
drift with an OpenCV version.

Every region records where it came from (`region_sources`): `parsing`, `landmark_geometry`,
`bbox_geometry`, `parsing+geometry`, `parsing+morphology`, `bbox_geometry+morphology`. On the real
64-preview audit the observed sources were `parsing` for eyes/nose/mouth, `parsing+geometry` for
forehead and cheeks, and `parsing+morphology` for `face_boundary`/`context`.

## 4. Coverage

- `requested_region_mask` = the union of the requested base region masks (full, coverage 1.0).
- `operator_support_mask` = the coverage-selected sub-mask that operators actually receive.
- `achieved_coverage` = `|operator_support_mask| / |requested_region_mask|` — i.e. coverage is
  measured **relative to the union of the requested base regions**, not to the crop or the face.

When `recipe.geometry.coverage < 1`, a deterministic procedural sub-mask is selected: a seeded 8×8
lattice is bilinearly upsampled to 224×224, and the top `round(coverage · area)` pixels of that field
*inside the union* are kept, with the flat pixel index as a stable tie-break. This yields blobs rather
than salt-and-pepper noise, and the achieved coverage is exact to within one pixel — far inside the
declared tolerance of **0.05**. The seed comes from the compiled graph
(`sha256(bank_id|recipe_id|recipe_seed|"<sample_id>|region_mask"|0)`), so the same sample/recipe pair
always yields the same mask.

The builder never returns an empty mask silently: an empty per-region mask, an empty union or an
empty coverage selection each raise `MaskBuildError`. At least one pixel is always kept.

## 5. Output contract

`RegionMaskResult` carries `requested_region_mask`, `operator_support_mask`, `per_region_masks`,
`region_sources`, `requested_coverage`, `achieved_coverage`, `mask_hash` and metadata. Masks are held
internally as `bool` and returned as `float32 [1,H,W]` containing **exactly** 0.0 or 1.0; the contract
validator rejects any other value. `mask_hash` is the SHA-256 of the shape-prefixed packed bits.
