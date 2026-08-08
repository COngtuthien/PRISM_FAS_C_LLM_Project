# M10 experiment contract

Frozen **before** any M10 matrix run is launched and before any target label is opened.

`target_labels_revealed: false` at the time this document was written and frozen.

Everything here is a declaration about what will be run and how it will be counted.
No value in this document was chosen by looking at a SiW-Mv2 result, because no
SiW-Mv2 result exists.

---

## 1. What M10 is

M10 executes the Table 59 baselines B00-B08, the Table 60 ablations, the Table 58
statistics and the controlled blind target evaluation (G7 -> G8). M9 produced ONE
reference training run and made no target, cross-domain, baseline or ablation claim;
all of those belong here.

Stage vocabulary, extending the frozen M9 flow:

```
G1  detector warm-up            source_train                    (M9, unchanged)
G2  prototype init + warm-up    source_train live only          (M9, unchanged)
G5  mixed training              source_train + M8 v3 bank       (M9, unchanged)
G6  source calibration          source_dev only                 (M9, unchanged)
G7  target prediction           target FEATURES, NO labels      (M10, new)
G8  target scoring              predictions + evaluator labels  (M10, new)
```

G7 and G8 are separate processes with separate permissions. G7 cannot resolve a
label path; G8 cannot write a checkpoint, an optimizer state or a calibration.

## 2. Baseline semantics — Table 59, read literally

Each baseline is a **configuration of the shared implementation**, never a forked
model. The flag set below is the contract; `configs/experiments/m10_matrix.yaml`
carries the same flags in machine form and the planner materializes them.

| ID | Table 59 text | local | global | fusion | region | manifold | synthetic | prompt | sampler |
|---|---|---|---|---|---|---|---|---|---|
| B00 | ConvNeXt binary classifier; domain-balanced source training | convnext | off | single_logit | off | off | none | off | domain_class_balanced |
| B01 | SigLIP/ViT binary classifier | off | siglip2_frozen | single_logit | off | off | none | off | domain_class_balanced |
| B02 | CNN + ViT simple concat, no region/manifold/prompt | convnext | siglip2_frozen | simple_concat | off | off | none | off | domain_class_balanced |
| B03 | B02 + physics augmentation only | convnext | siglip2_frozen | simple_concat | off | off | physics_only | off | domain_class_balanced |
| B04 | B02 + GPAT synthetic bank, no recipe/manifold | convnext | siglip2_frozen | simple_concat | off | off | gpat_only | off | domain_class_balanced |
| B05 | B02 + one global real center/Gaussian | convnext | siglip2_frozen | simple_concat | off | global_center | none | off | domain_class_balanced |
| B06 | Regional detector + global center | convnext | siglip2_frozen | prism_noisy_or | on | global_center | none | off | domain_class_balanced |
| B07 | Regional detector + multi-prototype manifolds, no synthetic | convnext | siglip2_frozen | prism_noisy_or | on | multi_prototype (K=4) | none | off | domain_class_balanced |
| B08 | Full PRISM-FAS-B v1.1 | convnext | siglip2_frozen | prism_noisy_or | on | multi_prototype (K=4) | bank_physics_gpat | frozen_prompt | domain_class_balanced |

Notes that are part of the contract, not commentary:

- **B00 is the M5 baseline definition**, unchanged: ConvNeXt V2 Atto, single BCE
  logit, domain/class-balanced batches, no M3B prior consumption.
- **B03 vs B04 vs B08 route semantics.** `physics_only` and `gpat_only` restrict the
  accepted M8 v3 bank rows to one route; they do not regenerate a bank. The bank
  identity `e84c78cd…b542` is unchanged in every row that uses synthetic data.
- **B05's "one global real center/Gaussian"** is the same `RealManifold` code with
  `K=1` and `region_scope=global`: one center over the pooled image embedding, no
  regional decomposition. It is not a second implementation.
- **B06 vs B07 is exactly hypothesis H2** (multi-prototype regional manifold vs one
  global center) and is the reason both carry three seeds.
- **B08 is the predeclared full method** and `K=4` its frozen reference
  configuration, identical to the M9 reference configuration
  (`configs/train/m9_reference.yaml` + `configs/models/m9_detector.yaml`).
- No baseline may be redefined to match an implementation shortcut. Where a flag
  combination is not yet implemented the row is reported `BLOCKED` with a reason,
  never silently re-pointed at a neighbouring configuration.

### 2.1 B08 seed 20260806 reuses the M9 reference run

The M9 reference run `m9_reference_seed20260806` has the identical scientific
configuration and identical seed to B08's first replicate. The planner therefore
emits that row with `reference_run_id: m9_reference_seed20260806` and
`reuses_m9_reference: true`; the registry binds the existing run instead of
retraining it. This is an identity binding, not an approximation: the row is
accepted only if the recorded resolved-config hash, architecture identity, dataset
contract identity and seed all match. If any of them differs, the row is retrained
under a new run id.

## 3. Ablation matrix — Table 60, read literally

Every ablation is expressed as a delta against a declared parent. The parent's own
configuration is never duplicated as a new row.

| # | family | variants materialized | parent | question (Table 60) |
|---|---|---|---|---|
| A01 | data balance | `naive_concat` (balanced = parent) | B08 | Gain có phải do sampler? |
| A02 | recipe | `random_operators` (structured = parent) | B08 | Recipe composition có giá trị không? |
| A03 | synthetic route | `physics_only`, `gpat_only` (physics+GPAT = parent) | B08 | Route nào tạo gain? |
| A04 | quality weighting | `hard_gate_only` (q weighting = parent) | B08 | Soft reliability có ích không? |
| A05 | region | `global_only` (semantic regions = parent) | B08 | Local anomaly có cần region prior? |
| A06 | prototype K | `k1`, `k2`, `k6` (K=4 = parent) | B08 | Multiple real modes có cần thiết? |
| A07 | outlier | `image_level` (mask-aware = parent) | B08 | Vùng sạch có được bảo toàn? |
| A08 | prompt | `off`, `adapter` (frozen prompt = parent) | B08 | Prompt gain độc lập ra sao? |
| A09 | backend | `pc_bounded_parity` (Modal = parent) | B08 | Infrastructure có làm lệch kết quả? |
| A10 | frame count | `f16`, `f32`, `f48_64` | B08 | Sampling density ảnh hưởng thế nào? |

### 3.1 A10 frame count is BLOCKED and stays in the matrix

```
status: BLOCKED
reason: declared data not available at the frozen frame plan
```

The frozen sampling plan is `frames_per_video: 4` (`uniform-v1`, `m2-v1`) for every
split. Re-extracting SOURCE at another density would change the source package
identity `b1cf29b6…9dc6` and break every M8 bank lock and M9 checkpoint identity
that binds it; and uniform-4 indices are not a subset of uniform-16, so a denser
TARGET package would forfeit the byte-for-byte live reproduction check that
`docs/M10_TARGET_DATA_CONTRACT.md` §7.3 requires. All three declared variants stay
in the matrix as `BLOCKED` rows so the gap is visible in the report; they are never
deleted and never silently substituted with a different ablation.

A denser target-side v3 package is a separate, deferred decision that **will not be
taken on the basis of any observed target result**.

### 3.2 A09 backend parity is a bounded protocol, and its full form is BLOCKED

Table 60 asks for "PC vs Modal same seed/config". This host has no CUDA device
(`torch 2.13.0+cpu`), so a full-length 35-epoch B08 run on the PC backend is not
feasible within the milestone's compute budget. Two rows record this honestly:

```
A09-backend-pc_bounded_parity   PLANNED   bounded_step_parity, declared step budget,
                                          identical config + identical seed,
                                          compares batch identity, loss trajectory
                                          and forward outputs within declared tolerance
A09-backend-pc_full_training    BLOCKED   no local CUDA device; a full-length CPU
                                          B08 run exceeds the milestone compute budget
```

The bounded protocol is the M6 parity contract, reused verbatim. It answers the
Table 60 question at the level the available compute supports and is reported as a
parity result, never as a second full training result.

## 4. Replication policy — FROZEN, compute-aware

The spec is explicit for two rows and genuinely **SPEC_UNDERSPECIFIED** for the
rest:

```
§20.1     "At least 3 seeds for main baseline and full method; mean/std reported."
Table 58  "Mean +/- std qua tối thiểu 3 seeds; paired bootstrap theo video khi so model."
Table 62  M10: "3 seeds, baselines/ablations, reproducible summary."
```

Table 62's "3 seeds, baselines/ablations" is a milestone summary line, not a
per-row requirement, and §20.1 names only "main baseline and full method". Rather
than assume 3 seeds everywhere (which the compute budget cannot carry) or 1 seed
everywhere (which would forfeit every statistical claim), the count is frozen by
**declared scientific role**:

### 4.1 Canonical seeds

```
seeds = [20260806, 20260807, 20260808]
```

A 3-seed row uses all three in that order. A 1-seed row uses `20260806`. No row
may use a seed outside this list, and the list is never extended after a result is
seen.

### 4.2 The rule

| replication role | seeds | may carry a statistical claim | rows |
|---|---|---|---|
| `spec_mandated` | 3 | yes | B00, B08 |
| `hypothesis_critical` | 3 | yes | B06, B07, A01 `naive_concat`, A02 `random_operators`, A04 `hard_gate_only`, A07 `image_level` |
| `diagnostic` | 1 | **no** — descriptive only | B01, B02, B03, B04, B05, A03 `physics_only`/`gpat_only`, A05 `global_only`, A06 `k1`/`k2`/`k6`, A08 `off`/`adapter` |
| `parity` | 1 (same declared seed on both backends) | no | A09 `pc_bounded_parity` |
| `blocked` | 0 | no | A10 `f16`/`f32`/`f48_64`, A09 `pc_full_training` |

`hypothesis_critical` is exactly the set of rows a declared hypothesis H1-H5 is
tested with:

```
H1  domain-balanced vs naive concatenation        B08            vs A01 naive_concat
H2  multi-prototype regional vs one global center  B07           vs B06
H3  mask-aware vs image-level outlier loss         B08           vs A07 image_level
H4  structured recipe vs random augmentation       B08           vs A02 random_operators
H5  quality-weighted vs equal-weight synthetic     B08           vs A04 hard_gate_only
H6  PC vs Modal, same package and config           A09 bounded parity protocol
```

### 4.3 What a 1-seed row may and may not be used for

A `diagnostic` row is reported with its single observed value, labelled
`single_seed_descriptive`, and **no confidence interval, no significance test and
no superiority claim** is attached to it. The report assembler enforces this: a
statistical comparison whose either side is a 1-seed row is refused, not
downgraded silently.

### 4.4 What this policy is not

It is not a claim that 3 seeds is sufficient for a strong statistical result. It is
the largest replication the milestone's compute supports on the rows that carry the
declared hypotheses, declared in advance so the count can never be adjusted to suit
an outcome.

## 5. Deterministic matrix identity

`reports/m10/M10_MATRIX_PLAN.json` carries `m10_matrix_identity`, the SHA-256 of the
canonical scientific rows only. The identity **excludes** wall-clock timestamps,
machine paths, run directories, Modal app/call ids, host environment and backend
assignment for non-parity rows. It **includes**, per row:

```
experiment_id, category, variant, seed, scientific_config_hash, parent_experiment_id,
source_package_identity, m8_bank_identity, m7_recipe_bank_identity,
siglip2_identity, recipe_text_cache_identity, convnext_weight_sha256,
required_stages, source_selection_rule, target_prediction_required,
replication_role, status, blocked_reason
```

Planning twice must produce identical rows and an identical
`m10_matrix_identity`; the planner is run twice and the two are compared before the
matrix may be executed.

`scientific_config_hash` deliberately excludes `backend`, so the A09 parity row and
B08 seed 20260806 hash identically — that is the definition of the parity test. Row
uniqueness is carried by `experiment_id`.

## 6. Source-side selection rule — unchanged and never target-driven

```
selection_metric   source_dev/acer
tie_break_metric   source_dev/bpcer
calibration_metric source_dev/nll
temperature        fitted on source_dev only
live/spoof threshold  min-ACER on calibrated source_dev probabilities
unknown/reject threshold  source-dev corruption/synthetic exposure only (see §7)
```

No architecture, loss, hyper-parameter, checkpoint, seed, threshold, calibration or
ablation variant may be selected using a SiW-Mv2 result. Target results are
post-hoc analysis, never a selection procedure. `SOURCE_MATRIX_LOCK.json` freezes
every selected checkpoint and calibration before the first G7 prediction runs.

## 7. Unknown/reject threshold — declared, currently NOT FITTED

Spec §16.2 requires the unknown/reject threshold to come from source-dev
corruptions/synthetic unknown exposure and forbids a target quantile. That fit
(`G6b`) has not been performed. Until it is:

```
reject_policy: not_fitted
unknown_threshold: null
```

and every reject-dependent metric is reported `not_applicable` with that reason.
Risk-coverage does not need the threshold — it needs only a confidence ordering —
and is therefore still computed. No target quantile is ever used to invent one.

## 8. Failure handling

Allowed terminal statuses: `COMPLETED`, `FAILED`, `BLOCKED`. A failed row is kept in
the registry with its failure record and appears in the report's negative-results
section. Rows are never silently dropped, and a `FAILED` row never becomes a
missing row.

## 9. Frozen inputs every row binds

```
source package        b1cf29b69a165ed5d9e074fc8127c17fbf057723edf9e272048ec3a564eb9dc6
M8 v3 bank            e84c78cd2a9b548244e243de0380998d04bc6770b91caf32ac7be96f489bb542
M7 recipe bank        fa989938cafdc4887518cc45c35d559d00278358439dc68c2486da10309210cb
SigLIP2               7e059e40dcc34913b51fc8d7bd25e6f0c023bc238261effee9bfb87b33f04822
recipe text cache     10f4ec35b7563b2b658cacc94599d35b9f93b531963a065459d4694d5dc2c141
ConvNeXt weight       6389c2f5a427b01a922e66e6d352c707424cccb62390c6936bc612e3d10b7ebb
```

None of these is modified by M10. `prism_data_v1_m3b`, `configs/data/siw_mv2.yaml`,
the M8 bank and every M9 artifact stay exactly as they are; the target evaluation
package is additive and separately versioned.
