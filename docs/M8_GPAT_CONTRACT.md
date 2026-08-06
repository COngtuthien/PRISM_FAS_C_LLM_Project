# M8 GPAT contract

Frozen **before** implementation. Anything the repository or `docs/spec_snapshot.md` does not
determine is labelled **SPEC GAP** and resolved with the fixed engineering default recorded here.

Source-only: M8 opens `source_train` payloads, the frozen M7 recipe bank and pinned quality weights.
It never opens `source_dev`, `target_test`, target shards, target labels or a raw dataset path.

---

## 1. Tensor shapes

| name | shape | dtype | range |
|---|---|---|---|
| `live_image` | `[B,3,224,224]` | float32 (AMP-castable) | `[0,1]` RGB |
| `source_spoof_image` | `[B,3,224,224]` | float32 | `[0,1]` RGB |
| `recipe_conditioning` | `[B,41]` | float32 | `recipe_conditioning_v1` |
| `target_support_mask` | `[B,1,224,224]` | float32 | exactly `{0,1}` |
| `source_style_mask` | `[B,1,224,224]` | float32 | exactly `{0,1}` |
| DWT band (each) | `[B,3,112,112]` | float32 | unbounded |
| `high` (LH‖HL‖HH) | `[B,9,112,112]` | float32 | unbounded |
| `delta_high` | `[B,9,112,112]` | float32 | `[-0.15, 0.15]` |
| `artifact_map_logit` | `[B,1,112,112]` | float32 | unbounded |
| `artifact_map` (full res) | `[B,1,224,224]` | float32 | `[0,1]` |
| `synthetic_image` | `[B,3,224,224]` | float32 | `[0,1]` |
| `artifact_latent` `z_a` | `[B,128]` | float32 | unbounded |
| `recipe_latent` `z_recipe` | `[B,64]` | float32 | unbounded |

`target_support_mask` is the M7 `operator_support_mask` for the (live sample, recipe) pair.
`source_style_mask` is the M7 requested-region mask for the same recipe evaluated on the **spoof
source** sample — it selects where the spoof's artifact statistics are read from.

## 2. DWT convention

**SPEC GAP** — the spec names DWT but fixes no normalization. Default: **orthonormal Haar**.

```
a = 1/sqrt(2)
LL = a*(a*(x00 + x01) + a*(x10 + x11))   # separable, applied rows then columns
LH, HL, HH follow the standard (low,high), (high,low), (high,high) products
```

Concretely, with `x` split into 2x2 blocks `x00,x01,x10,x11` (row, column):

```
LL = (x00 + x01 + x10 + x11) / 2
LH = (x00 + x01 - x10 - x11) / 2      # low along columns, high along rows
HL = (x00 - x01 + x10 - x11) / 2      # high along columns, low along rows
HH = (x00 - x01 - x10 + x11) / 2
```

This transform is orthonormal (its own inverse up to transpose), so `IDWT(DWT(x)) == x` analytically.
Declared numerical tolerance in fp32: **max abs reconstruction error <= 1e-6**.

Implemented in pure PyTorch (`torch.nn.functional` slicing/`conv2d`), differentiable, CPU+CUDA,
fp32 and AMP. **No PyWavelets call appears in any model forward.** Input `H,W` must be even.

`high = cat([LH, HL, HH], dim=1)` in exactly that order — 9 channels, channel `3*k + c` is band `k`
channel `c`.

## 3. Architecture

`GPATResidualModel` — fixed engineering default (the spec fixes the DWT residual idea, not layer
counts; **SPEC GAP** for widths/blocks).

```
ArtifactEncoder(source_spoof_image [B,3,224,224])
  Conv3x3 s2  3  -> 32  + GroupNorm + SiLU
  Conv3x3 s2 32  -> 64  + GroupNorm + SiLU
  Conv3x3 s2 64  -> 128 + GroupNorm + SiLU
  Conv3x3 s2 128 -> 256 + GroupNorm + SiLU
  masked global average pool over source_style_mask (falls back to global mean if empty)
  Linear 256 -> 128                                  -> z_a [B,128]

RecipeEncoder(recipe_conditioning [B,41])
  Linear 41 -> 128 + SiLU + Linear 128 -> 64          -> z_recipe [B,64]

Generator
  input  = cat([LL_live (3), high_live (9), support_half (1)]) = 13 channels @ 112x112
  stem   = Conv3x3 13 -> 64
  4 x FiLMResidualBlock(64):
      h = Conv3x3(GN(SiLU(x))) ; h = h * (1+gamma) + beta ; h = Conv3x3(SiLU(GN(h))) ; x = x + h
      (gamma, beta) = Linear(cat([z_a, z_recipe]) [192]) -> 128 -> split
  delta_head        = Conv3x3 64 -> 9
  artifact_map_head = Conv3x3 64 -> 1
```

Parameter count is recorded in `reports/m8/local_gpat_smoke.json` and in the checkpoint's
architecture hash. No discriminator, no GAN loss, no detector head.

## 4. Residual and the LL hard lock

```
A_half   = sigmoid(artifact_map_logit) * support_half          # support_half in {0,1}
delta_high = tanh(delta_head) * A_half * max_high_frequency_delta      # default 0.15
high_out = high_live + delta_high
LL_out   = LL_live                    # identical tensor object, not a copy of a prediction
generated = IDWT(LL_live, high_out)
generated = where(target_support_mask, generated, live_image)         # exact composite
generated = clamp(generated, 0, 1)
```

**ΔLL is structurally absent**: the model has no `delta_LL` parameter, head or output field, so low
frequency geometry cannot be modified by training. `GPATOutput` exposes no `delta_LL` key, and a test
asserts the field does not exist on the dataclass.

Because `IDWT` is applied to the *unchanged* `LL_live`, the LL of the reconstruction (before the
`clamp`/composite, which act at full resolution) equals `LL_live` up to float round-off. Declared
tolerance for `max|LL(generated_pre_composite) - LL_live|` in fp32: **1e-5**.

## 5. Mask semantics

| mask | meaning | where used |
|---|---|---|
| `target_support_mask` | where the artifact may be written on the live face | composite, `L_map`, `L_strength`, style stats of `G` |
| `support_half` | `target_support_mask` downsampled to 112 by 2x2 max-pool, then re-binarized | generator input channel, `A_half`, `L_tv` |
| `source_style_mask` | where the spoof's artifact statistics are read | style stats of `S`, artifact-encoder pooling |
| `exact_edit_mask` | discrete uint8 pixel difference after saving | bank output only, not a training tensor |

A pair whose `target_support_mask` or `source_style_mask` has **zero** pixels is **failed**, never
silently turned into a zero loss.

## 6. Loss formulas and weights

`a_recipe` = mean of the recipe's artifact strengths (scalar per sample).
`mu_high(X, M)`, `std_high(X, M)` = per-high-frequency-channel masked mean / std at 112 resolution,
using the mask downsampled the same way as `support_half`.

```
L_style     = mean|mu_high(G, M_t)  - mu_high(S, M_s)|
            + mean|std_high(G, M_t) - std_high(S, M_s)|
L_identity  = mean(1 - cos(e_G, stopgrad(e_L)))
L_map       = mean((A_full - M_t * a_recipe)^2)
L_strength  = |masked_mean(A_full, M_t) - a_recipe|      (mean over batch)
L_residual  = mean|delta_high|
L_tv        = total_variation(delta_high * support_half)

L_total = 1.00*L_style + 0.50*L_identity + 0.50*L_map
        + 0.25*L_strength + 0.02*L_tv + 0.01*L_residual
```

All six coefficients live in `configs/synthesis/gpat_m8.yaml` and every component is logged.

`L_LL` and `L_outside` are **not** optimization terms — they are asserted invariants (§4, §7).

## 7. Gradient recipients

| module | gradients |
|---|---|
| artifact encoder | **yes** |
| recipe encoder | **yes** |
| residual generator + FiLM | **yes** |
| delta head, artifact-map head | **yes** |
| AdaFace IR-50 | **no** — `requires_grad_(False)`, `eval()`; gradient flows *through* it into `e_G` only |
| recipe bank / conditioning vectors | **no** — constant inputs |
| package image/prior tensors | **no** |
| quality-gate models (SCRFD, FaceXFormer) | **no** — not in the training graph at all |

`e_L` (live embedding) is cached and `stopgrad`-ed.

Asserted every step: total loss is a finite scalar; every component finite; backward finite;
LL invariant holds; outside-support difference is exactly 0 after the composite; zero-support pairs
rejected.

## 8. Optimization

AdamW, per-group LR (`encoder 2e-4`, `recipe 1e-4`, `generator 2e-4`), weight decay `1e-4`,
betas `(0.9, 0.999)`, cosine schedule with `warmup_fraction 0.05` and `min_lr 1e-6`, gradient clip
norm `1.0`, batch size 16, up to 15 epochs, seed `20260806`. Precision: fp16 AMP on CUDA, fp32 on CPU.
Early stopping: `min_epochs 5`, `patience_epochs 4`. Checkpoint selection: min
`validation_total_loss`, tie-break max `validation_identity_cosine`.

## 9. Checkpoint and resume

`schema_version = "m8-gpat-ckpt-v1"`. Contents: model, optimizer, scheduler, AMP scaler, epoch,
global step, best metrics, Python/NumPy/Torch-CPU/Torch-CUDA RNG state, pair-plan identity, package
identity, recipe-bank identity, config hash, architecture hash, AdaFace revision + weight SHA,
train/validation record-set hashes, git commit, resume lineage.

Resume **rejects** (raises, never partially loads) on mismatch of: package identity, recipe-bank
identity, pair-plan identity, config hash, architecture hash, AdaFace weight SHA.

## 10. Pair-plan rules

See `reports/m8/pairs/PAIR_PLAN_LOCK.json`. Fixed seed `20260806`.

- Every pair: live target from `source_train` **live**, spoof source from `source_train` **spoof**,
  recipe from the frozen M7 bank.
- Records partitioned by `sha256(source_record_id)` into GPAT train 80% / validation 20%, stratified
  by dataset, separately for the live and spoof roles. No `source_record_id` appears in both
  partitions for the same role.
- Counts: train **896**, validation **224**, total **1120**.
- Per live target: 4 training-plan pairs — 2 same-domain spoof sources, 2 cross-domain.
- `live.source_record_id != spoof.source_record_id`.
- If both explicit `subject_id` values are available, subjects must differ; otherwise the manifest
  records `different_subject_rule = "not_applicable"`.
- No raw path or filename enters a pair manifest.
- `pair_id = "gpatpair_" + sha256(package_identity|recipe_bank_identity|partition|live_sample_id|
  spoof_sample_id|recipe_id|pair_seed)[:20]`.

**SPEC GAP** — the spec asks to balance attack family. The M3B package carries no attack-family or
device column for source data, so `attack_family_balance = "unavailable"` is recorded and balancing
falls back to live target domain, spoof source domain, `source_record_id` and recipe attributes.
