# M9 training contract

Frozen **before** any trainer code was written. Every requirement traces to a heading in
`docs/spec_snapshot.md`. Values the local spec does not fix are marked **SPEC_UNDERSPECIFIED**,
given the smallest engineering default that lets the spec's own equations run, placed in config and
recorded in `DECISIONS.md`.

Companion document: `docs/M9_DETECTOR_CONTRACT.md` (architecture, PromptHead, manifolds, fusion).

---

## 1. Losses — §10.1 (Table 35)

The spec states the loss set verbatim:

```
L_cls    = CE(p_global, y_image)
L_local  = weighted_BCE(local_token_logits, artifact_map)
L_MIL    = CE(LogSumExpMIL(token_logits), y_image)
L_real   = mean_live sum_r SoftMin_k d_rk
L_out    = mean_syn  sum_r m_r * max(0, margin_r - d_r)
L_clean  = mean_syn  sum_r (1-m_r) * min(d_r, clean_cap)
L_prompt = InfoNCE(z_attack_region, text_embedding(recipe))
L_cons   = |p_global - stopgrad(s_region)| + |s_region - stopgrad(p_global)|
L_risk   = Var(domain_risk) + Var(artifact_family_risk)

L_total = L_cls_real + lambda_syn * q * (L_cls_syn + lambda_local*L_local + lambda_out*L_out)
        + lambda_M*L_real + lambda_clean*L_clean + lambda_MIL*L_MIL
        + lambda_P*L_prompt + lambda_cons*L_cons + lambda_risk*L_risk
```

Read the total literally: **`q` multiplies only the synthetic bracket** — the synthetic
classification, local and outlier terms. It does not scale `L_real`, `L_clean`, `L_MIL`, `L_prompt`,
`L_cons` or `L_risk`, and it never touches `L_cls_real`. `L_cls` is split into a real part and a
synthetic part exactly as the total requires.

### Per-loss contract

| Loss | Applicable to | Normalization denominator | Zero behaviour |
|---|---|---|---|
| `L_cls_real` | real live + real spoof | count of real samples | finite 0 if none |
| `L_cls_syn` | synthetic | count of synthetic samples | finite 0 if none |
| `L_local` | synthetic **with an artifact map** | applicable tokens | finite 0 if none |
| `L_MIL` | all labelled samples | batch size | finite 0 if none |
| `L_real` | **live only**, valid regions | live count x valid regions | finite 0 if none |
| `L_out` | synthetic, **attacked** regions (`m_r = 1`), valid visibility | applicable (sample, region) pairs | finite 0 if none |
| `L_clean` | synthetic, **clean** regions (`m_r = 0`), valid visibility | applicable (sample, region) pairs | finite 0 if none |
| `L_prompt` | synthetic, attacked regions, valid visibility | applicable regions | finite 0 if none |
| `L_cons` | all samples | batch size | finite 0 if none |
| `L_risk` | all samples, needs >= 2 groups | variance over group means | finite 0 with < 2 groups |

`m_r` is the **regional attack mask**: region `r` of a synthetic sample is attacked when the M8
exact edit mask overlaps that region's prior. Real samples have no `m_r` and never enter `L_out`,
`L_clean` or `L_prompt`. Every mean is over applicable entries only; invalid-visibility regions are
excluded before the denominator is formed, never zero-filled inside it.

Every loss term is logged individually in `metrics.jsonl` **independently of the total**, so a term
that is legitimately zero for a batch is visible rather than inferred.

### Initial loss weights — §10.4 (Table 38)

```
lambda_syn=0.50  lambda_M=1.00     lambda_out=1.00   lambda_clean=0.25
lambda_local=1.00 lambda_MIL=0.50  lambda_P=0.20     lambda_cons=0.05
lambda_risk=0.10  margin_out=3.0   synthetic_ratio=0.25
```

`margin_r = margin_out = 3.0` — specified, not a default. §10.4 says these are starting points whose
search space must be declared in advance and tuned source-only; **M9 does not tune them at all**, and
certainly not after seeing any result.

### SPEC_UNDERSPECIFIED in §10.1

| Symbol | Engineering default | Config key |
|---|---|---|
| `clean_cap` | **margin_out = 3.0** — the same scale the outlier margin already fixes, so a clean region is pulled no further than an attacked region is pushed | `loss.clean_cap` |
| `weighted_BCE` weighting | per-token weight = artifact-map value, positives normalized by their own mass so a sparse map is not swamped by background | `loss.local_weighting` |
| `LogSumExpMIL` temperature | **1.0** | `loss.mil_temperature` |
| `SoftMin_k` temperature | reuses `tau_prototype` = 1.0 | `model.manifold.tau_prototype` |
| InfoNCE temperature | **0.07** | `model.prompt.temperature` |
| `domain_risk` | per-domain mean `L_cls`, groups = `{casia_fasd, msu_mfsd}` | `loss.risk.domain_groups` |
| `artifact_family_risk` | per-artifact-family mean `L_cls` over synthetic samples, family = the M7 recipe's primary artifact name | `loss.risk.family_groups` |
| `CE` on a binary head | BCE-with-logits on the single `global_logit`, the M5 convention (`live=0, spoof=1`) | `loss.classification` |

## 2. Batch composition — §10.2 (Table 36, Table 65)

| Partition | MVP ratio | Batch 32 | Rule |
|---|---|---|---|
| Real live | 37.5 % | **12** | CASIA/MSU domain-balanced; required every batch |
| Real spoof | 37.5 % | **12** | CASIA/MSU domain-balanced; required every batch |
| Synthetic spoof | 25 % | **8** | quality-weighted; mix physics and GPAT |

Table 65 confirms `batch: {live: 12, real_spoof: 12, synthetic_spoof: 8}` and `domain_balance: true`.
Table 9 adds: synthetic never exceeds 25 % of a batch, and every batch must contain real live **and**
real spoof.

Gradient accumulation preserves the **effective** composition across the accumulation window
(§10.2: "Nếu GPU nhỏ, dùng gradient accumulation nhưng giữ effective composition trên accumulation
window"). The microbatch may shrink; the ratio may not.

Synthetic route balance: the spec says "mix physics and GPAT", so both routes are required to be
present in the effective window; it fixes no ratio, so none is imposed beyond presence.

The synthetic live-target domain (`casia_fasd` / `msu_mfsd`) is the **live source domain** of the
image that was edited. It is **not** an attack taxonomy and is not balanced as one.

Sampler determinism: seeded by `base_seed + epoch` (§12.7: "DataLoader seed kết hợp base_seed + epoch
+ rank; worker seed deterministic"). Resume reproduces the next batch sequence exactly.

## 3. Optimization — §10.3 (Table 37)

| Parameter | Spec value |
|---|---|
| Optimizer | **AdamW** |
| Backbone LR | **1e-5** |
| Heads/manifold LR | **1e-4** |
| Weight decay | **0.05** |
| Scheduler | **5 % warm-up + cosine decay** |
| Epochs | **30 detector epochs after warm-up** |
| Warm-up | **3 detector + 2 manifold epochs** |
| Gradient clipping | **1.0** |
| AMP | **bf16 when supported, else fp16** |
| EMA | 0.999 **optional; report whether enabled** |
| Checkpoint criterion | **source_dev ACER primary, BPCER tie-break, then calibration NLL** |

All specified; none is a default. Table 65 independently confirms
`selection_metric: source_dev/acer`, `tie_break_metric: source_dev/bpcer` and `max_epochs: 30`.

**EMA decision:** the spec makes EMA optional and requires reporting whether it is enabled. M9 runs
the reference configuration with **EMA disabled**, and `run.json` records `ema_enabled: false`.
Enabling it would add a second weight set whose effect is untested at this milestone; that is an M10
ablation, not an M9 default.

## 4. Stage flow — §11 (Table 39, Table 40)

The spec's stages:

| Stage | Name | Input | Output | In M9? |
|---|---|---|---|---|
| G0 | Local data build | Raw datasets | Validated package | done in M2/M3 |
| G1 | **Baseline warm-up** | Source real live/spoof | Binary detector checkpoint | **yes** |
| G2 | **Real manifold init** | Source live embeddings | Prototype init + covariance | **yes** |
| G3 | Recipe bank freeze | Source-only ontology | BANK_LOCK | done in M7 |
| G4 | Synthetic bank | Source train + recipes | Quality-gated bank | done in M8 |
| G5 | **Mixed training** | Real + synthetic | Full PRISM checkpoint | **yes** |
| G6 | **Source calibration** | Source-dev predictions | thresholds.json | **yes** |
| G7 | Target prediction | SiW target features | prediction parquet | **M10** |
| G8 | Final scoring | prediction + eval labels | report | **M10** |

M9 runs **G1, G2, G5 and G6**. G7 and G8 are explicitly out of scope: they are where target data
first becomes readable, and M9 never opens it.

Stage transition rules (§11.1) implemented verbatim: each stage consumes an input lock/hash and emits
an output lock/hash; a stage may not read a future artifact and **G5 must not read the target
package**; resume from `last` only when the config hash and the data package hash match; changing the
data package, recipe bank or model revision forces a **new run id**; a synthetic bank built once may
be reused across detector runs when `BANK_LOCK` matches.

Run status machine (§11.2, Table 40):

```
PENDING -> RUNNING -> {COMPLETED | FAILED | INTERRUPTED}
INTERRUPTED -> RESUMING -> RUNNING
FAILED may resume only when the failure is marked recoverable.
```

Each stage writes `stage_state.json`, `resolved_config.yaml`, `input_hashes.json`, logs,
`metrics.jsonl` and `output_hashes.json`.

**Warm-up reading:** Table 37's "Warm-up | 3 detector + 2 manifold epochs" and §9.3's "K-means ...
after detector warm-up" are the same schedule seen from two sides. M9 implements: 3 epochs of G1
(classification only, prototypes absent), then G2 K-means initialization on `source_train` live
embeddings, then 2 manifold warm-up epochs in which the manifold terms are active, then 30 G5 epochs.

## 5. Checkpoint and resume — §15.3, §12.7

Save `last` every fixed number of optimizer steps and at epoch end; save `best` by the **source_dev**
criterion and never by a SiW result; atomic write with verification before the pointer moves; keep
`last`, `best` and top-3; commit the Modal Volume after important checkpoints; resume restores
Python, NumPy and Torch CPU/CUDA RNG states and the sampler epoch.

The checkpoint carries model, optimizer, scheduler, scaler, RNG states, **prototypes**, epoch and
global step (§12.7). M9 extends the M8 strict-identity pattern; **every** guarded field below causes
resume to raise rather than partially load, and there is no `strict=False` path:

```
schema version, source package identity, M8 bank identity, model architecture identity,
pretrained weight SHAs (ConvNeXt, SigLIP2), config hash, loss-contract hash, prototype-state
identity, training stage, sampler state, best source-dev metric, stage lineage, git commit
```

§12.7 also requires that a resolved-config mismatch **blocks** resume with a human-readable diff
rather than silently loading a partial state.

## 6. Prototype export — §9.3

`prototypes.npz`, arrays only, `allow_pickle=False` compatible, binding: the nine region names,
`K` per region, centers, diagonal variance, validity, counts, epsilon, the model feature-contract
identity, the `source_train` live population identity, the prototype config hash and a prototype
content identity. No paths and no machine metadata enter the identity. Export is deterministic for
unchanged inputs.

Prototype initialization uses **only** `source_train` AND `label = live` — the frozen package holds
exactly **280** such samples (CASIA 160 / MSU 120). Initialization fails explicitly if a region has
fewer than `K = 4` valid samples; K is **not** silently reduced.

## 7. Source isolation — §3, §6 (Table 23)

Required evidence for the reference run, written to `reports/m9/source_isolation.json`:

```
source_train_opened          = true
m8_validated_bank_opened     = true
source_dev_opened            = true   (validation / checkpoint-selection stage only)
target_test_opened           = false
target_labels_opened         = false
raw_dataset_opened           = false
m8_rejected_candidates_used  = false
m8_v1_v2_banks_used          = false
```

`source_dev` produces **no optimization gradient**: it is read only to select a checkpoint and to fit
the source-only calibration (§16.2 — temperature scaling fit on source_dev only, threshold by a
declared source-dev criterion).

## 8. What M9 does and does not claim

M9's acceptance is **implementation and training correctness** (Table 62: "Toy tests and source-dev
training stable"). M9 reports source_train and source_dev evidence only.

M9 does **not** claim that PRISM improves SiW-Mv2, any cross-domain gain, any target APCER/BPCER/ACER
or any state-of-the-art result. Target evaluation is a controlled blind M10 stage, and §20.1 requires
that no result is ever selected on SiW-Mv2 metrics.
