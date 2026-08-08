# M10 statistics contract

Frozen **before** any target score exists. `target_labels_revealed: false` at the
time this document was written and frozen.

Nothing here may be changed after a SiW-Mv2 result is seen. Every number below was
chosen from the spec text and the declared compute budget, not from an outcome.

---

## 1. What the spec fixes

```
Table 58   "Mean +/- std qua tối thiểu 3 seeds; paired bootstrap theo video khi so model."
§20.1      "At least 3 seeds for main baseline and full method; mean/std reported."
Table 62   M10: "3 seeds, baselines/ablations, reproducible summary."
```

Fixed by the spec: the **unit** of the bootstrap (video), the **pairing** (paired
when comparing models), the **minimum** replication for the main baseline and the
full method, and that mean ± std is the summary form.

Genuinely `SPEC_UNDERSPECIFIED` and frozen here: the number of resamples, the
bootstrap seed, the confidence level, the comparison statistic, the interval type,
the p-value convention, and the multiple-comparison policy.

## 2. Replication

The seed list and the per-row replication roles are frozen in
`docs/M10_EXPERIMENT_CONTRACT.md` §4 and are part of this contract by reference:

```
seeds = [20260806, 20260807, 20260808]

spec_mandated        3 seeds   B00, B08                             statistical claims allowed
hypothesis_critical  3 seeds   B06, B07, A01, A02, A04, A07         statistical claims allowed
diagnostic           1 seed    every other executable row            descriptive only
parity               1 seed    A09 bounded parity                    parity result only
blocked              0 seeds   A10 frame-count rows, A09 full        no result
```

A `diagnostic` row is reported as `single_seed_descriptive`. **A comparison whose
either side is a 1-seed row is refused by the statistics module**, not silently
downgraded to a point estimate with an interval.

Multi-seed summary form:

```
mean, std (population std, ddof = 0, over the 3 seed values), min, max, n_seeds
```

## 3. Paired bootstrap by video — FROZEN

```
unit                  video (the opaque siw_<16hex> id)
pairing               paired: one resample of video ids, both models scored on it
resamples             10000
bootstrap_seed        20260810
confidence_level      0.95
interval              percentile (2.5th, 97.5th)
statistic             delta = metric(model_A) - metric(model_B), video level
default metric        ACER at the frozen source-dev threshold
p_value               two-sided:  2 * min(P(delta <= 0), P(delta >= 0)), clipped to [0,1]
resample size         n = the number of videos common to both models
replacement           with replacement
rng                   numpy PCG64, seeded structurally (see §3.1)
```

### 3.1 The resample plan is deterministic and identity-bearing

The generator seed is **not** the raw integer; it is derived structurally so the
same comparison always draws the same videos and a different comparison cannot
accidentally reuse a plan:

```
seed_material = sha256( "m10-bootstrap-v1"
                        | bootstrap_seed | resamples | statistic_name
                        | sha256(join("\n", sorted(video_ids))) )
```

The resulting `[resamples, n]` integer index matrix has its own
`bootstrap_plan_identity = sha256(index matrix bytes)`, recorded beside every
interval. Running the same comparison twice must reproduce the plan identity and
every reported value **exactly**; the statistics module runs the plan builder twice
and compares before reporting.

### 3.2 What "paired" means here

One resample of video ids is drawn once and **both** models are evaluated on that
same resample, so the interval is on the difference and the between-video variance
that both models share is removed. Drawing two independent resamples would inflate
the interval and is not what Table 58 asks for.

### 3.3 Seeds and the bootstrap are separate axes

The bootstrap resamples **videos**, not seeds. For a 3-seed row the video-level
score used inside a resample is the **mean over the three seeds** of that video's
score, and the seed-to-seed spread is reported separately as mean ± std. The two
are never combined into one interval, because they answer different questions
(sampling variability of the evaluation set vs training variability).

## 4. Declared comparison family and multiple-comparison policy

Exactly five pre-declared comparisons carry a statistical claim, one per hypothesis:

```
H1  B08                vs  A01 naive_concat
H2  B07                vs  B06
H3  B08                vs  A07 image_level
H4  B08                vs  A02 random_operators
H5  B08                vs  A04 hard_gate_only
```

```
multiple_comparison_policy = holm_bonferroni
family_size                = 5
family                     = {H1, H2, H3, H4, H5}
```

Holm-Bonferroni is applied to the five two-sided p-values. Both the raw and the
adjusted p-value are reported, together with the adjusted decision at
`alpha = 0.05`. Any comparison outside this family is exploratory and is labelled
`exploratory_not_in_family`; it never receives an adjusted p-value and never
supports a superiority claim.

H6 (PC vs Modal) is a **parity** check, not a superiority test: it is reported as
agreement within a declared tolerance, with no p-value.

## 5. What a statistical result here can and cannot say

- It can say whether the observed difference on this 1700-video SiW-Mv2 evaluation
  set is larger than the resampling variability of that set, at three training
  seeds.
- It cannot say anything about a population of datasets, about other attack types,
  or about deployment. n = 3 seeds is the spec minimum, and it is small.
- A non-significant result is reported as a **negative result**, in full, in the
  report's negative-results section. It is never dropped, re-tested under a
  different metric, or re-run with another seed until it passes.
- Adding a seed, changing the metric, changing the resample count or changing the
  family after seeing a result is forbidden. If a genuine defect forces a change,
  it is versioned as a new contract revision with the reason recorded, exactly as
  the M8 calibration revisions were, and the superseded run is retained.

## 6. Calibration statistics

ECE (15 equal-width bins), Brier and NLL are reported for the calibrated target
probability of every scored row, alongside the frozen source-dev values, so
source-to-target calibration drift is visible. No target-derived temperature or
threshold is ever fitted.

## 7. Efficiency statistics

Params, FLOPs estimate, latency, peak VRAM and data throughput are reported per
variant with offline and online cost separated (Table 58). They are engineering
measurements and carry no confidence interval.
