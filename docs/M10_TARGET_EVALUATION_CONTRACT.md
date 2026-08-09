# M10 target evaluation contract — G7 prediction and G8 scoring

Frozen **before** any target label is opened and before any target metric exists.

`target_labels_revealed: false` at the time this document was written and frozen.

Companion documents: `docs/M10_TARGET_DATA_CONTRACT.md` (what the target package is),
`docs/M10_EXPERIMENT_CONTRACT.md` (what is run), `docs/M10_STATISTICS_CONTRACT.md`
(how differences are tested).

---

## 1. Two processes, two permissions

```
G7  target prediction   reads  frozen checkpoint, frozen source calibration,
                               target FEATURE manifest, target images/priors
                        writes frame prediction parquet (label-free) + PREDICTION_LOCK
                        CANNOT resolve any target label path

G8  target scoring      reads  frozen predictions, PREDICTION_LOCK,
                               evaluation-only label artifact, frozen evaluation config
                        writes metric/report artifacts only
                        CANNOT train, build an optimizer, write a checkpoint,
                        modify a calibration or modify a prediction
```

The separation is structural, not advisory: see §8.

## 2. Frame prediction schema — Table 57, versioned and nullable

```
prediction_schema_version = "m10-target-prediction-v1"
```

| column | type | applicability |
|---|---|---|
| `sample_id` | string | always |
| `video_id` | string | always — the opaque `siw_<16hex>` record id, never a path |
| `frame_id` | int64 | always — see the note below |
| `p_global` | float64 | always |
| `s_region` | float64 or **null** | null when `region_status = not_applicable` |
| `p_prompt` | float64 or **null** | null when `prompt_status = not_applicable` |
| `s_final` | float64 | always, finite, in [0,1] |
| `confidence` | float64 | always, finite, in [0,1] |
| `decision` | string | `live` \| `spoof` \| `reject` |
| `top_region_ids` | list[int64] | empty list when regions are not applicable |
| `region_distances` | list[float64] | empty list when regions are not applicable |
| `checkpoint_hash` | string | always |
| `calibration_hash` | string | always |
| `inference_config_hash` | string | always |
| `region_status` | string | `computed` \| `not_applicable` |
| `prompt_status` | string | `computed` \| `not_applicable` |
| `variant` | string | the experiment variant that produced the row |

**`frame_id` is an ordinal, not a timestamp.** The frozen package manifest does not
carry the real in-video frame index: M3A/M3B keep only the opaque `sample_id`, which
HASHES that index without exposing it. `frame_id` is therefore the deterministic
ordinal of the frame within its video in frozen `sample_id` order, and it is used
only as an ordering key for the aggregation join. It is never read as a video
position or a time. Recovering the real index would mean re-opening the raw video
tree, which G7 has no permission to touch.

**Nullable, never fabricated.** B00-B05 have no regional branch and no PromptHead.
Their rows carry `s_region = null`, `p_prompt = null`,
`region_status = not_applicable`, `prompt_status = not_applicable`, and empty
`top_region_ids` / `region_distances`. Writing `0.0` for a component the model does
not have would silently enter the fusion arithmetic and the report as a measured
value; `null` cannot.

**`p_prompt` is `not_applicable` on target data even for B08.** Measured, not
assumed: the M9 `PromptHead.applicability` mask is `is_synthetic AND attacked region
AND visible`. A target sample is never synthetic and carries no attack-region mask,
so no region is ever applicable and the head returns an EXACT structural zero. G7
therefore reads the model's own `prompt_applicable` mask and writes `null` when it
is empty, instead of recording a constant as a measurement.

Two consequences, stated so nobody has to rediscover them:

1.  On target data B08's fusion reduces to `s_final = 1 - (1 - p_global)(1 - s_region)`
    for every row. The prompt term contributes nothing at inference on real data.
2.  The A08 prompt ablation can therefore only differ on target through **training**
    — `L_prompt` shaping the shared region embeddings — never through inference-time
    fusion. A08's target difference must be interpreted that way and no other.

For a variant without the regional or prompt branch, `s_final = p_global`. This is
the Table 34 fusion with the absent factors omitted, not a second formula:
`1 - (1 - p_global)` collapses to `p_global` when the other two evidence terms do
not exist.

**A prediction parquet carries NO ground-truth label**, no attack family, no
taxonomy, no subject, no session and no raw path. The writer refuses a forbidden
column by name before the file is written.

## 2b. REVISION v1 → v2: which quantity the frozen threshold belongs to

```
prediction_schema_version   m10-target-prediction-v1  ->  m10-target-prediction-v2
revised                     before any target label was opened
justified by                source_dev only; no target observation exists
superseded run              retained (the v1 predictions are kept, not deleted)
```

**The defect.** v1 decided `spoof` when `s_final >= threshold`, where `threshold` is
"the frozen source-dev threshold". But `M9Trainer.run_g6` fits BOTH the temperature
and that threshold on `output.global_logit` alone
(`calibrate_source_dev(logits=global_logit, ...)`), and `M9Trainer.validate` selects
the checkpoint on `sigmoid(global_logit)`. The frozen operating point therefore
belongs to the **calibrated `p_global`**, and `s_final` is pointwise `>= p_global`.
Applying one quantity's threshold to the other is a category error, not a tuning
choice.

**The measurement, on `source_dev`, B08's frozen reference checkpoint, frozen
threshold 0.8374507610075123** (`reports/m10/DECISION_SCORE_DIAGNOSTIC.json`):

| score | APCER | BPCER | ACER | ROC-AUC | EER |
|---|---|---|---|---|---|
| calibrated `p_global` | 0.1614 | 0.1125 | **0.1382** | 0.9179 | 0.1449 |
| fused `s_final` | 0.0000 | 1.0000 | **0.5000** | 0.9063 | 0.1160 |

The `p_global` row reproduces the frozen G6 record (ACER 0.13695, BPCER 0.11250,
ROC-AUC 0.91785). The `s_final` row is a degenerate all-spoof classifier, and the
reason is measured rather than inferred: **mean `s_region` on bona-fide `source_dev`
samples is 0.9644** (0.9999 on attacks). The regional distance evidence is not on a
calibrated probability scale, so `1 - (1 - p)(1 - 0.96)` saturates for every sample.

**The revision.** A new column `decision_score` carries the quantity the frozen
calibration was fitted on — the calibrated `p_global`. `decision`, `confidence` and
the video aggregation all use it. `s_final` is unchanged, still recorded, and still
reported, but only on the metrics that need no calibrated scale (ROC-AUC, EER),
under `fused_evidence`. **No ACER is printed for `s_final` at the frozen threshold**,
because that number would not be a measurement.

For B00–B05 and every other variant without a regional or prompt branch,
`decision_score == s_final` by construction, so nothing changes for them.

**What this does NOT do.** No checkpoint, no temperature, no threshold and no
calibration was refitted; nothing was retrained; `SOURCE_MATRIX_LOCK` is untouched.
The predictions were regenerated from the same frozen checkpoints and the same
frozen calibrations. The fix is confined to how G7 forms a decision from the model's
own outputs, which is exactly the repair the milestone permits before a reveal.

**What it costs, stated plainly.** Under v2 the regional branch contributes to the
reported ranking (`fused_evidence`) but not to the frozen operating point. H2
(B07 vs B06) and every region-bearing comparison are therefore adjudicated on
`p_global`, i.e. on how the regional machinery shaped the global branch during
TRAINING, not on inference-time fusion. That is a real limitation of this evaluation
and is reported as one; the alternative — fitting a threshold for `s_final` — would
be a new source-side calibration, which M10 forbids after `SOURCE_MATRIX_LOCK`.

## 3. Video aggregation — FROZEN, Table 57

```
video_score       = trimmed_mean(frame_decision_score, trim = 0.10)   # revised, see 2b
video_fused_score = trimmed_mean(frame_s_final,        trim = 0.10)   # carried alongside
video_confidence  = median(frame_confidence)
video_decision    = threshold_and_reject(video_score, video_confidence)
```

The aggregation FUNCTION, its trim fraction, its grouping and its ordering are
unchanged and were never revisited. Only the column it is applied to moved, for the
reason in section 2b, and the fused score is aggregated identically beside it.

`trimmed_mean` is the deterministic form already frozen in M5
(`prism_fas.train.video_aggregation.trimmed_mean`): sort ascending,
`trim_count = floor(n * 0.10)`, drop `trim_count` from each end only when
`trim_count > 0` and `2 * trim_count < n`. At the frozen 4 frames per video
`trim_count = 0`, so aggregation reduces to the plain mean of 4 values — that is a
consequence of the frozen frame plan, stated rather than hidden.

`threshold_and_reject`:

```
reject  when unknown_threshold is not null and video_confidence < unknown_threshold
spoof   when video_score >= threshold
live    otherwise
```

`threshold` is the frozen source-dev threshold. `unknown_threshold` is currently
`null` (see the experiment contract §7), so no row is rejected and
`rejection_rate = 0.0` by construction rather than by measurement.

Grouping is by `video_id` in sorted order, frames within a video sorted by
`sample_id`. **Aggregation is never changed after target scoring.**

## 4. Frame-level and video-level metrics

Both are produced. Frame predictions stay label-free on disk; the label join happens
**only inside G8**, only in memory, and only through the opaque id:

```
frame  metrics: join frame prediction -> label on video_id (all frames of a video
                share its video label)
video  metrics: aggregate first, then join on video_id
```

The join key is the opaque `siw_<16hex>` id and nothing else. Labels are never
joined by filename, path, attack name or directory structure at scoring time.

## 5. PREDICTION_LOCK

Every prediction file is accompanied by `PREDICTION_LOCK.json`:

```json
{
  "prediction_lock_schema_version": "m10-prediction-lock-v1",
  "experiment_id": "...", "variant": "...", "seed": 20260806,
  "scientific_config_hash": "...", "source_matrix_lock_identity": "...",
  "code_commit": "...",
  "checkpoint_sha256": "...", "source_calibration_sha256": "...",
  "calibration_hash": "...", "inference_config_hash": "...",
  "target_feature_package_identity": "...", "target_package_id": "...",
  "prediction_logical_identity": "...", "row_count": 0, "video_count": 0,
  "aggregation": {"video_score": "trimmed_mean", "trim": 0.10,
                  "video_confidence": "median", "threshold": 0.0,
                  "unknown_threshold": null},
  "prediction_schema_version": "m10-target-prediction-v1",
  "target_labels_opened": false,
  "status": "LOCKED"
}
```

`scientific_config_hash`, `source_matrix_lock_identity` and `code_commit` were added
**before any prediction existed and before any label was opened**. They bind a
prediction to the frozen source-side decision that produced it, so a lock traces back
to a matrix row and to the code that ran it without consulting anything outside the
file. They are empty only for an engineering smoke, which has no matrix row to bind.

`prediction_logical_identity` hashes the logical rows — `(sample_id, video_id,
frame_id, s_final, confidence, decision)` rounded to 12 decimals — not the parquet
bytes, for the same reason M7/M8 bind logical rows: pyarrow versions write
different bytes for identical data.

G8 **refuses** to score when any of these holds:

1. the prediction has no lock, or the lock status is not `LOCKED`;
2. the lock's `checkpoint_sha256` differs from the registry's selected checkpoint;
3. the lock's `source_calibration_sha256` or `calibration_hash` differs from the
   frozen source calibration;
4. the lock's `inference_config_hash` differs from the frozen inference config;
5. `prediction_logical_identity` does not reproduce from the prediction file;
6. `row_count` or `video_count` disagree with the file;
7. the lock's target feature package identity differs from the declared package.

A refusal is an error, never a warning and never a fallback path.

`TARGET_PREDICTION_LOCKSET.json` freezes the complete set of PREDICTION_LOCKs for
the whole matrix. **Labels may be opened only after that lockset is frozen**, and
that transition is recorded exactly once in `reports/m10/TARGET_LABEL_REVEAL.json`.

The lockset binds more than the predictions, so a reader can tell a MISSING
prediction from an ABSENT one: the `SOURCE_MATRIX_LOCK` identity (and it refuses a
set whose locks do not all bind the same one), the matrix identity, the target
FEATURE identity, every logical row's terminal status, the BLOCKED rows with their
frozen reasons, and `rows_without_prediction` — the rows that legitimately produce
none, each with the specific reason. The A09 bounded-parity row is the only such row
in M10: it trains nothing to completion and selects no checkpoint, so there is
nothing to predict with. An engineering-smoke lock is refused outright.

Between the lockset and the reveal stands one more gate, `PRE_REVEAL_AUDIT.json`.
Every condition is checked against an artifact, never asserted: the source lock
revalidates, the lockset reproduces its own identity and covers exactly the eligible
rows, every prediction file on disk reproduces its locked logical identity and the
frozen checkpoint/calibration/config it names, no reveal file exists yet, and no
scoring artifact exists yet. `m10 reveal` refuses unless that audit passed **for the
same lockset identity**, and refuses a second time outright.

## 6. Metric definitions — fixed here so they cannot drift

Label convention is unchanged from M4/M5: `live = 0`, `spoof = 1`, spoof is the
positive attack class, decision is `p_spoof >= threshold`.

| metric | definition |
|---|---|
| APCER | attack presentations classified as live / total attacks |
| BPCER | bona-fide presentations classified as attack / total bona fide |
| ACER | (APCER + BPCER) / 2 **at the frozen source-dev threshold** |
| HTER | (FAR + FRR) / 2 **at the source-dev EER threshold**, FAR = APCER, FRR = BPCER |
| ROC-AUC | trapezoidal area under the ROC of the spoof score |
| EER | the point where FNR and FPR meet on the target ROC, reported with its threshold |
| ECE | 15 equal-width bins over the calibrated probability |
| Brier | mean squared error of the calibrated probability against the 0/1 label |
| NLL | mean binary cross-entropy of the calibrated probability |
| risk-coverage | error rate on the most-confident fraction, coverage swept over the ranked confidence; needs no threshold |
| rejection rate | fraction of videos decided `reject` |
| attack-wise APCER | APCER restricted to one attack family, computed **post-hoc**, never used in tuning |

ACER and HTER are numerically the same functional at the same threshold. They are
kept distinct because their **threshold source** differs: ACER at the frozen
operating threshold M10 must actually deploy, HTER at the EER threshold, which is
the conventional cross-domain reporting point. Both threshold sources are recorded
beside the value.

Where a metric cannot be computed from the available protocol it is reported as

```json
{"status": "not_applicable", "reason": "<the specific reason>"}
```

and never as `0.0`, `null` or an invented value. Open-set/unknown AUROC/AUPR is
`not_applicable` on this protocol: SiW-Mv2 gives no unknown-class label, and
inventing one from the attack families would be fabricating a label.

## 7. Degenerate populations

If a scored population has zero attacks, APCER, ACER, HTER, ROC-AUC and EER are
**undefined** and are reported `not_applicable` with the population counts — the
exact situation the frozen 785-live-only package produces. The scorer raises rather
than dividing by zero, and the report prints the counts.

## 8. The firewall is structural

Enforcement is a resolvable-path permission check plus a capability object, **not a
string scan**. The M8/M9 lesson is recorded and applies here: a blanket token match
flags the proof of isolation (`target_labels_opened: false`) as a leak.

```
roots declared in configs/evaluation/m10_target.yaml
  source_package_root         TRAIN, G7(no), G8(no)
  target_feature_root         TRAIN(no), G7 yes, G8(no)
  target_label_root           TRAIN(no), G7(no),  G8 yes
  prediction_root             TRAIN(no), G7 write, G8 read
```

`FirewallStage.TRAIN` may not resolve `target_feature_root` or `target_label_root`.
`FirewallStage.G7` may resolve `target_feature_root` and may not resolve
`target_label_root`. `FirewallStage.G8` may resolve `target_label_root` and
`prediction_root`, and may not write any of `.pt`, `.pth`, `.ckpt`,
`optimizer_state`, `calibration/`.

A path is checked after `Path.resolve()`, so `..`, a symlink or a relative spelling
cannot slip past. A declaration such as `target_labels_opened: false` is data, is
skipped by the taxonomy check, and never counts as a violation.

The attack taxonomy (`Replay`, `Paper`, `Silicone`, `Mask_*`, `Makeup_*`,
`Partial_*`, `Mannequin`) may not appear in a training config, in checkpoint
selection logic or in any G7 input. The check walks the config structurally and
skips declared isolation-evidence keys instead of pattern-matching the document.

## 9. Rules that outlive this document

- Aggregation, thresholds, metric definitions and the statistics contract are frozen
  before scoring and are never changed after a target result is seen.
- G8 never updates a model, an optimizer or a calibration (spec Table 54).
- The reporter never recomputes a prediction; it reads frozen artifacts (Table 54).
- `q` is a sample weight, never a label.
- `target_labels_revealed` is one-way. Once `true`, it is never reset.
