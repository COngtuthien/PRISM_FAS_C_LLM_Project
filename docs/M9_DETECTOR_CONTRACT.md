# M9 detector contract

Frozen **before** any detector code was written. Every requirement below traces to a heading in
`docs/spec_snapshot.md`; the "Spec" column names the section and, where the content lives in an
extracted table, the table number.

Anything the local spec does not fix is marked **SPEC_UNDERSPECIFIED** and carries an explicit
engineering default that lives in config, is recorded in `DECISIONS.md`, and is **never** presented
as a spec requirement.

---

## 1. Scope

M9 implements the regional CNN-VLM detector, the PromptHead, the multi-prototype regional real
manifolds, the image-level fusion and the declared losses, and trains **one** reference
configuration. It does not run the M10 experiment matrix.

Spec acceptance for this milestone (§19.1, Table 62):

> `M9 | Regional fusion + manifolds + losses | Toy tests and source-dev training stable.`

That is the bar: toy tests plus stable source-dev training. Multi-seed runs, the `K in {2,4,6}`
search, the B00-B08 baselines and the ablation matrix are **M10** (Table 59, Table 60, Table 62).

## 2. Inputs and region set — §9.1

| Requirement | Spec | Implementation |
|---|---|---|
| RGB 224x224 train input from the canonical 256 crop | §9.1 | the existing M3/M4 canonical loader; **no second preprocessing pipeline** |
| Optional high-pass/wavelet detail computed on-the-fly | §9.1 | `optional_highpass`, config flag, default **off** |
| Region set MVP: `left_eye, right_eye, nose, mouth, forehead, left_cheek, right_cheek, face_boundary, context` | §9.1 | `REGION_ORDER`, identical to M3B `VISIBILITY_REGIONS` and M7 `masks.REGION_ORDER` |
| Region prior is a **soft mask / query initialization**, not nine hard crops | §9.1 | `RegionQuery` builds soft priors from stored parsing + landmarks; a test forbids a nine-crop path |
| Visibility mask blocks prototype update from occluded or mis-detected regions | §9.1 | `CanonicalGeometry.visibility` float16[9], thresholded; see §6 below |

The nine-region ordering is already frozen three times over in this repository (M3B priors, M7 masks,
M8 coverage), so M9 reuses it rather than redeclaring it.

## 3. Architecture — §9.2 (Table 32)

The spec states the architecture as:

```
F_local              = ConvNeXtV2(x_rgb, optional_highpass)
T_global, z_global   = SigLIP2.image_encoder(x_rgb)

q_r = RegionQuery(parsing_r, landmarks_r, learnable_token_r)
z_r = CrossAttention(q_r, K=F_local, V=F_local) + RegionPool(T_global, parsing_r)

p_global = GlobalHead(z_global)
d_r      = RealManifold.distance(z_r)
p_prompt = PromptHead(z_r, frozen_recipe_text_embeddings)
```

Backbones (§1.2, Table 5):

| Branch | Spec default | Pin |
|---|---|---|
| Local | **ConvNeXt V2 Atto or Tiny** | `convnextv2_atto.fcmae_ft_in1k`, already pinned by M5/M6 with weight SHA `6389c2f5a427b01a922e66e6d352c707424cccb62390c6936bc612e3d10b7ebb`. The spec allows either; Atto is chosen because it is already pinned, verified and uploaded, which removes a new unpinned dependency. |
| Global | **SigLIP2 Base P16-224, frozen; train fusion/heads** | `google/siglip2-base-patch16-224`, revision and file SHA-256 pinned from the real downloaded bytes in `configs/models/m9_detector.yaml` before the first run |

**The global branch is frozen. Only fusion and heads train on it** — that is explicit in Table 5
("SigLIP2 Base P16-224 frozen, train fusion/heads").

### Typed forward output — §14.2 (Table 53)

The spec fixes the forward contract:

```
ModelOutput(
    global_logit:        Tensor[B, 1],
    local_logits:        Tensor[B, P],
    region_embeddings:   Tensor[B, R, D],
    region_distances:    Tensor[B, R],
    prompt_logits:       Tensor[B, N_prompt] | None,
    confidence_features: dict[str, Tensor],
    aux:                 dict[str, Tensor],
)
```

M9 implements this as a frozen dataclass, not an untyped tuple, with `R = 9`. It additionally
carries the visibility mask and the fusion components as named fields so §11 ablations can read them
without re-deriving anything.

### SPEC_UNDERSPECIFIED in §9.2

| Symbol | Why undetermined | Engineering default | Config key |
|---|---|---|---|
| `D` (region embedding dim) | Table 53 names `D` but fixes no value | **256** | `model.region_embedding_dim` |
| `P` (local token count) | Table 53 names `P`; it is a property of the local feature map | `7*7 = 49` at 224 input with ConvNeXt V2 Atto stage-4 stride 32 | derived, asserted in a test |
| `CrossAttention` head count / hidden width | not stated | 4 heads, width `D` | `model.region_attention_heads` |
| `RegionPool` reduction | not stated | visibility-free soft-mask weighted mean of `T_global` patch tokens over the region prior, resized to the patch grid | `model.region_pool` |
| `GlobalHead` shape | not stated | single linear layer on `z_global` -> 1 logit | `model.global_head` |

Each default is the smallest thing that makes the spec's own equation type-check.

## 4. PromptHead — §9.2, §10.1

The spec gives PromptHead exactly two facts: its signature
`p_prompt = PromptHead(z_r, frozen_recipe_text_embeddings)` (Table 32) and its loss
`L_prompt = InfoNCE(z_attack_region, text_embedding(recipe))` (Table 35). It also appears in the
quality-gate table as `Recipe match | >= tau_prompt khi prompt head có sẵn` (Table 29) — that is an
**M8 gate** row, and M8 recorded it as `recipe_match = not_applicable` precisely because no
PromptHead existed. **That placeholder is not a training signal and is not reused here.**

| Item | Contract |
|---|---|
| Input feature | `z_r`, the regional embeddings `[B, R, D]` |
| Target representation | frozen recipe text embeddings — one vector per recipe in the frozen M7 bank |
| Text encoder frozen? | **Yes.** Table 5 freezes the global branch; §2.2 forbids fine-tuning the text encoder from the start ("Không tự động fine-tune text encoder từ đầu; chỉ LoRA/adapter sau ablation frozen backbone") |
| Text source | the frozen M7 recipe bank's own deterministic recipe description, encoded once by the **frozen SigLIP2 text encoder** and cached by recipe id. No target text, no attack taxonomy. |
| Output | `prompt_logits [B, N_prompt]` — cosine similarity between the projected attacked-region embedding and every recipe text embedding, scaled by an InfoNCE temperature |
| Normalization | both sides L2-normalized before the dot product |
| Applicable samples | **synthetic samples only**, and only on regions the recipe actually attacked (`attack_region_mask`) with valid visibility. Real live and real spoof carry no recipe, so PromptHead contributes **nothing** for them and its loss is masked to a finite zero. |
| Loss | `L_prompt = InfoNCE(z_attack_region, text_embedding(recipe))` (Table 35), weight `lambda_P = 0.20` (Table 38) |
| Fusion role | the spec's fusion uses `p_prompt_spoof` (Table 34), so PromptHead **does** participate in `s_final` |
| Deployment | it is not a separate deployment dependency at inference: the text embeddings are a cached constant matrix, and the head is one projection. A test asserts the fast path needs no text encoder at inference. |

### SPEC_UNDERSPECIFIED in PromptHead

| Symbol | Engineering default | Config key |
|---|---|---|
| `N_prompt` | 128 — one logit per frozen M7 recipe (the bank holds exactly 128) | derived from the bank, asserted |
| InfoNCE temperature | **0.07** | `model.prompt.temperature` |
| `p_prompt_spoof` | `max_n softmax(prompt_logits)_n` over the attacked regions, 0 when no region is applicable — a bounded [0,1] evidence term as `s_final` requires | `model.prompt.spoof_reduction` |
| projection of `z_r` into text space | one linear layer `D -> text_dim`, L2-normalized | `model.prompt.projection` |

## 5. Multi-prototype real manifolds — §9.3 (Table 33)

```
M_r_real      = { N(mu_rk, Sigma_rk) } for k = 1..K_r
d_r(x)        = min_k (z_r - mu_rk)^T Sigma_rk^{-1} (z_r - mu_rk)
assignment_rk = softmax(-d_rk / tau_prototype)
```

| Requirement | Spec |
|---|---|
| `K_r = 4` for every region in MVP | §9.3 |
| `K in {2,4,6}` tuned **on source_dev** | §9.3 — **M10**, not M9 |
| Init by **K-means on source_train live embeddings after detector warm-up** | §9.3 |
| **Diagonal covariance + epsilon floor** | §9.3 |
| Prototype update only from **live** samples with **valid region visibility** | §9.3 |
| Prototype state saved in checkpoint and exported as `prototypes.npz` | §9.3 |

Because the covariance is diagonal, the Mahalanobis form collapses to
`d_rk = sum_d (z_rd - mu_rkd)^2 / max(var_rkd, eps)` — that is the exact spec formula under the
spec's own diagonal constraint, not a substitution.

Hard invariants, each with a test: a real-spoof sample cannot alter prototype state; a synthetic
sample cannot alter prototype state; an invalid-visibility region cannot alter prototype state; and
K-means is deterministic given a fixed seed and a stable input ordering.

### SPEC_UNDERSPECIFIED in §9.3

| Symbol | Engineering default | Config key |
|---|---|---|
| `tau_prototype` | **1.0** | `model.manifold.tau_prototype` |
| epsilon floor | **1e-4** on the variance | `model.manifold.covariance_epsilon` |
| K-means implementation | deterministic Lloyd, k-means++ init from a seeded PCG64, fixed iteration cap 100, stable tie-break by index | `model.manifold.kmeans` |
| prototype update rule during training | EMA of assigned live embeddings, decay 0.99, updated under `torch.no_grad()` | `model.manifold.update` |

The spec fixes *what* updates prototypes (live + visible only) and *when they are initialized*
(K-means after warm-up); it does not fix the online update rule, so an EMA is used and declared.

## 6. Visibility gating

`CanonicalGeometry.visibility` is a float16[9] in [0,1] already computed and frozen in the M3B
package. A region is **valid** when `visibility_r >= visibility_threshold`.

Invalid regions are excluded from: prototype initialization, prototype updates, regional manifold
statistics, and every regional loss term. They are **not** silently replaced by zeros inside a mean —
each affected loss divides by the count of *applicable* entries and returns a finite 0 when that
count is 0.

**SPEC_UNDERSPECIFIED:** the visibility threshold. Default **0.30**, `model.visibility_threshold`.

## 7. Image-level fusion — §9.4 (Table 34)

```
s_region = TopKMean(normalize(d_r), k=2)
s_final  = 1 - (1 - p_global) * (1 - s_region) * (1 - p_prompt_spoof)
# Unknown/reject also considers entropy and global-local disagreement.
```

Implemented exactly. `p_global` is the sigmoid of `global_logit`. The three evidence terms are
exposed as separate named outputs so M10 can ablate them **by config flag on one implementation
path** — no ablation branch is hard-wired into the model (§11 of the task contract, §17.2 of the
spec).

`confidence_features` carries the spec's two named reject signals: predictive **entropy** and
**global-local disagreement** `|p_global - s_region|`.

### SPEC_UNDERSPECIFIED in §9.4

| Symbol | Engineering default | Config key |
|---|---|---|
| `normalize(d_r)` | `1 - exp(-d_r / d_scale)`, a monotone map of the spec's distance into [0,1] with `d_scale` = the running median live distance per region, frozen after warm-up | `model.fusion.distance_normalizer` |
| `TopKMean` over invalid regions | invalid regions are excluded before the top-k; if fewer than k remain, the mean is taken over what is valid; if none are valid `s_region = 0` | `model.fusion.topk` |

`k = 2` is specified by the spec and is not a default.

## 8. Immutable inputs

| Artifact | Identity | Rule |
|---|---|---|
| Source package | `prism_data_v1_m3b`, `b1cf29b69a165ed5d9e074fc8127c17fbf057723edf9e272048ec3a564eb9dc6` | read-only |
| M7 recipe bank | `fa989938cafdc4887518cc45c35d559d00278358439dc68c2486da10309210cb` | read-only; supplies recipe text for PromptHead |
| M8 synthetic bank | `prism_synthetic_bank_m8_v3_e84c78cd2a9b`, content identity `e84c78cd2a9b548244e243de0380998d04bc6770b91caf32ac7be96f489bb542`, status `validated` | read-only; **871 accepted samples only** |

M9 **fails closed** when the requested bank identity does not match, and never falls back to a
working namespace, to a v1/v2 bank, or to rejected candidates. `q` is loaded as a **sample weight**
and is never converted into a label (§8 Table 30: "Detector uses q_i as a weight; q_i is not a
live/spoof label").

## 9. What M9 must never touch

Per §3 (Table 8), §6 (Table 23) and the task contract: no SiW-Mv2 target sample, target label or
attack taxonomy in training, checkpoint selection or threshold tuning; no rejected M8 candidate; no
v1/v2 bank; no raw dataset. `source_dev` is opened **only** by the explicitly permitted validation
and checkpoint-selection path (§10.3 checkpoint criterion, §16.2 calibration).

The trainer core never imports `modal` (§12.1, Table 41).
