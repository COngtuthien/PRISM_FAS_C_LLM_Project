# M8 quality gate contract

Frozen **before** any candidate was generated. Calibration uses **only `source_train`**; no
`source_dev`, no `target_test`, no target label or metadata.

---

## 1. Pinned quality models

Verified against the real files in the ignored model cache and against `PACKAGE_LOCK.json` /
`configs/models/m3b_priors.yaml` — not copied from a prompt.

| role | model | source | revision | file | SHA-256 | bytes |
|---|---|---|---|---|---|---|
| detection + 5 landmarks | SCRFD 10G bnkps (ONNX) | insightface, pinned local artifact | *no upstream revision string*; pinned by SHA | `scrfd_10g_bnkps.onnx` | `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91` | 16,923,827 |
| parsing + pose | FaceXFormer | HF `kartiknarayan/facexformer` | `fd12148d0b19` | `face_geometry/ckpts/model.pt` | `327a755849ba64d336fb96589ff87b27e84a12be1ecf8bcfaa503d66f803286d` | 1,104,869,851 |
| identity | AdaFace IR-50 WebFace4M | HF `minchul/cvlface_adaface_ir50_webface4m` | `60a65befbcf7` | `face_identity/pretrained_model/model.pt` | `43bd2d570584d95d4a17ce81f26449034c45dbeed750afcab651872abc0e1496` | 174,611,121 |

**SPEC GAP** — SCRFD has no upstream revision string in this repository; it is a pinned local ONNX
artifact whose SHA-256 already appears in `PACKAGE_LOCK.detector_model_sha256`. That SHA is the
pin. Input 320, threshold 0.5 (`scrfd_source_policy_v1`), unchanged from M2.

Normalization and output semantics are reused verbatim from M3 (`configs/models/m3b_priors.yaml`):
FaceXFormer RGB 224, ImageNet mean/std, bicubic, task token 0 (parsing) / 2 (pose), LaPa 11 classes;
AdaFace BGR 112, mean/std 0.5, bicubic, 512-d L2-normalized.

Models are never substituted after seeing quality results. Tests never download weights.

## 2. Calibration populations (source_train only)

1. **live**: all **280** `source_train` live samples (CASIA 160, MSU 120).
2. **real spoof**: all **1160** `source_train` spoof samples (CASIA 800, MSU 360).
3. **benign**: deterministic mild perturbations of the 280 live samples, fixed before calibration:
   - brightness factor in `{0.98, 1.02}`
   - contrast factor in `{0.98, 1.02}`
   - additive Gaussian noise, `std <= 0.002`, from a per-sample local PCG64 seed
   - (JPEG Q=95 is *not* used: `cv2.imencode` quality is not guaranteed byte-deterministic across
     OpenCV builds, so it is excluded rather than assumed)

   4 variants per live sample -> **1120** benign samples. No M7 spoof-producing operator is used as
   benign calibration.

## 3. Thresholds

| symbol | rule | note |
|---|---|---|
| `tau_fd` | **pinned SCRFD production threshold 0.5**, not fitted | from `scrfd_source_policy_v1` |
| `tau_id` | 1st percentile of benign identity cosine | raw percentile and final value both recorded |
| `tau_lm` | 99th percentile of benign landmark NME | NME normalized by inter-ocular distance with `eps = 1e-6` |
| `tau_parse` | 1st percentile of benign outside-mask parsing Dice | |
| `tau_out` | **exactly 0** after final discrete compositing | not fitted |
| `tau_fp` | 99th percentile of real-spoof leave-one-record-out fingerprint scores | |

Thresholds are frozen in `reports/m8/quality_calibration.json` together with the SHA-256 of the
source population, the model weights, the calibration config, the thresholds themselves and the
fingerprint reference. **Thresholds are never lowered after seeing accepted counts.**

## 4. Fingerprint reference

24-dimensional high-frequency feature vector per image:

| slice | length | feature |
|---|---|---|
| `0:9` | 9 | mean absolute value of the 9 Haar high-frequency channels |
| `9:18` | 9 | standard deviation of the 9 Haar high-frequency channels |
| `18:21` | 3 | per-RGB-channel edge energy (mean abs Sobel-like gradient magnitude) |
| `21` | 1 | Laplacian variance of the grey image |
| `22` | 1 | mean saturation (max-min over channels) |
| `23` | 1 | channel-balance / colour-shift magnitude (max abs deviation of channel means from the grey mean) |

Robust reference per **spoof source domain** (`casia_fasd`, `msu_mfsd`): per-dimension `median` and
`MAD` with an epsilon floor `1e-6` (MAD scaled by 1.4826). No trainable deep probe is used.

```
z_d(x, domain) = (x - median_domain) / max(1.4826*MAD_domain, eps)
fingerprint_score(x) = min over valid domains of mean(z_d(x, domain)^2)
```

`tau_fp` = 99th percentile of the leave-one-record-out scores of the 1160 real spoof samples (the
sample's own `source_record_id` is excluded from the domain statistics used to score it).

## 5. Artifact-strength gate

With `a` = requested mean recipe artifact strength:

```
lower = max(0.01, 0.25 * a)
upper = min(0.50, 1.75 * a)
```

Measured artifact strength = masked mean of the saved artifact map over the exact edit mask.

## 6. Per-candidate metrics

1. `face_detection_score`
2. `identity_cosine` (generated vs the **live target** embedding)
3. `landmark_nme`
4. `outside_mask_parsing_dice` (parsing argmax agreement outside the exact edit mask)
5. `outside_mask_max_error`
6. `measured_artifact_strength`
7. `requested_artifact_strength`
8. `fingerprint_score`
9. `support_overlap` (|exact_edit_mask ∩ requested_support| / |exact_edit_mask|)
10. `recipe_match` = **`not_applicable`** — M9's PromptHead does not exist, so no prompt score is
    invented.

## 7. Hard gates (all must pass)

```
face_detection_score      >= tau_fd
identity_cosine           >= tau_id
landmark_nme              <= tau_lm
outside_mask_parsing_dice >= tau_parse
outside_mask_max_error    == 0
lower <= measured_artifact_strength <= upper
fingerprint_score         <= tau_fp
support_overlap           >= 0.90
```

## 8. Quality weight q

```
q_fd       = clip((fd - tau_fd) / (1 - tau_fd), 0, 1)
q_id       = clip((identity - tau_id) / (1 - tau_id), 0, 1)
q_lm       = clip((tau_lm - nme) / max(tau_lm, eps), 0, 1)
q_parse    = clip((dice - tau_parse) / (1 - tau_parse), 0, 1)
q_strength = triangular peak at the requested strength, 1 at `a`, 0 at `lower` and `upper`
q_fp       = clip(1 - fingerprint_score / max(tau_fp, eps), 0, 1)
q_support  = clip(support_overlap, 0, 1)

q = exp(mean(log(max(q_component, eps))))          # geometric mean, eps = 1e-6
```

`q` is float32, finite, in `[0,1]`. **`q` is not a live/spoof label** — it is an M9 sample weight.
A candidate is **never** rejected for a low `q` alone when every hard gate passes.

## 9. Rejected candidates

A rejected row keeps: `synthetic_id`, `route`, `live_target_sample_id`, `spoof_source_sample_id`,
`recipe_id`, the list of failed gate names, every metric value and the threshold hash. Rejected
image binaries are not preserved.
