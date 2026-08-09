# C0 — Protocol manifest plan (P1 / P2 / P3)

C0 inspects the inherited adapters and manifests and **designs** protocol-specific
manifest generation. It does not run expensive preprocessing, and it does not open a
single SiW label.

## 1. What the inherited package already provides

The frozen Version-B source package `prism_data_v1_m3b`
(identity `b1cf29b69a165ed5d9e074fc8127c17fbf057723edf9e272048ec3a564eb9dc6`,
status `validated`, 59/59 validation checks) contains:

| Split | Samples |
|---|---|
| `source_train` | 1 440 |
| `source_dev` | 2 079 |
| `target_test` | 3 140 (SiW-Mv2 **live only**, feature-only, no labels) |
| total | 6 659 |

| Dataset | Samples |
|---|---|
| `casia_fasd` | 2 399 |
| `msu_mfsd` | 1 120 |
| `siw_mv2` | 3 140 |

Split mapping inherited from Version B §3.2: official **train** → `source_train`,
official **test** → `source_dev`. No random frame split; every frame of a video stays in
one split.

`target_isolation` in the package lock reads
`{"policy": "feature_only_no_labels_no_identity", "status": "passed"}`.

The full P3 evaluation population lives in the separate additive package
`prism_target_eval_v2` (identity `c3a29e695ad0…e48a8`): 1 700 videos = 785 live + 915
spoof, 6 776 frames, with features and labels in **separate trees**. `prism_data_v1_m3b`
is not modified by it, so every M8/M9 identity bound to it still holds.

## 2. Are the inherited adapter semantics sufficient for subject/video-disjoint manifests?

**Yes for CASIA and MSU. Deliberately not for SiW — and that is correct.**

Evidence from `src/prism_fas/data/adapters/adapters.py`:

- **CASIA-FASD** — records are grouped by `(split, label, subject_id, video_id)` parsed
  from an explicit YAML `path_pattern`, and `CanonicalVideoRecord` carries a real
  `subject_id` and a composed `video_id`. Subject **and** video are available.
- **MSU-MFSD** — the same pattern; `subject_id` is parsed and excluded from the capture
  metadata dict, and `video_id` uniqueness is tracked via a `seen` set.
- **SiW-Mv2 (target)** — `TargetInferenceRecord` carries an **opaque** `video_id`, no
  `subject_id`, and `metadata_provenance = "explicit YAML path_pattern; labels withheld"`.
  The private, label-bearing variant is a separate evaluator-only path explicitly marked
  `PRIVATE evaluator-only`.

`src/prism_fas/data/manifests/leakage.py` enforces this structurally: `subject_id`,
`label`, `attack_family`, `taxonomy`, `ground_truth` and related tokens are in a
`FORBIDDEN` set, and `assert_target_safe` fails if any of them reaches a target row.

**Conclusion:** the metadata needed to build subject-disjoint and video-disjoint P1/P2
manifests exists for both source datasets, and the target manifest is structurally
incapable of carrying the identifiers that would leak. No new preprocessing is required to
build the protocol manifests — only metadata-level selection over the existing frozen
manifests.

## 3. Protocol manifest design

All three protocols are **derived views** over the frozen package. They add no new crop,
no new prior and no new byte to the source package, so `prism_data_v1_m3b` keeps its
identity and every downstream Version-B identity bound to it remains valid.

### P1 — CASIA → MSU

```
train        casia_fasd  ∩  source_train
select/cal   casia_fasd  ∩  source_dev
cross-test   msu_mfsd    ∩  (source_train ∪ source_dev)   -> frozen cross-test manifest
```

### P2 — MSU → CASIA

```
train        msu_mfsd    ∩  source_train
select/cal   msu_mfsd    ∩  source_dev
cross-test   casia_fasd  ∩  (source_train ∪ source_dev)   -> frozen cross-test manifest
```

For P1 and P2 the cross-domain test set is the **whole other source dataset**, frozen
before any P1/P2 run. Its labels exist (it is a source dataset), so the firewall here is
procedural discipline plus a lock — not a capability boundary. The rule is unambiguous:
once a P1 or P2 cross-test result is seen, nothing in that protocol may be retuned.

### P3 — CASIA + MSU → SiW-Mv2 (MAIN)

```
train        (casia_fasd ∪ msu_mfsd)  ∩  source_train
select/cal   (casia_fasd ∪ msu_mfsd)  ∩  source_dev,  domain-balanced
target       prism_target_eval_v2  — 1700 videos, label-free features only at C11
```

P3 is the only protocol with a hard capability boundary: TRAIN, LLM and SYNTHESIS
processes cannot resolve SiW labels or target metrics at all.

### 3.1 Why the source `train`/`dev` sizes look inverted

`source_dev` (2 079) is larger than `source_train` (1 440) because the inherited mapping
sends the official **test** split to `source_dev`. This is the Version-B contract and is
inherited unchanged for byte-level package reuse. It is recorded here so that nobody later
"fixes" it and silently invalidates the frozen package identity. If a future protocol
version wants a different ratio, that is a new package and a new protocol version.

## 4. Required validation before any scientific training

For each of P1, P2, P3, before a single training step:

1. **Subject disjointness** — no `subject_id` appears in both the train side and the
   select/calibrate side, and none appears in the cross-domain test side. Where an adapter
   yields `subject_id = null`, the video group is the minimum unit and the null is
   recorded rather than guessed.
2. **Video disjointness** — no `video_id` appears in more than one role. Every frame of a
   video shares one role.
3. **Leakage count = 0**, computed by the inherited leakage checker over the generated
   manifests.
4. **Dataset restriction** — the train manifest contains only the datasets the protocol
   declares; a training-mode selector cannot request `target_test`.
5. **Label presence** — source manifests carry labels; the P3 target feature manifest
   carries none, and validation fails if a label column or a forbidden token appears.
6. **Manifest identity** — each protocol manifest gets its own content hash, and that hash
   enters the run identity of every row that consumes it.

> The Version-C spec §5.3 requires that C0 materialize protocol-specific manifests and
> prove subject/video leakage = 0 **before** any scientific training.

## 5. What C0 does and does not materialize

Materialization here is purely metadata-level: it reads existing Parquet manifests, filters
rows, and writes new manifests plus a leakage report. It decodes no video, runs no
detector, computes no prior and touches no GPU.

**C0 delivers the design and the validation contract above.** The manifest files
themselves are written by the first C1 step, because they must carry a Version-C manifest
schema version and land in the Version-C namespace, and because C0's frozen scope is
reconciliation and repository isolation. Nothing about that ordering permits a scientific
training run before the leakage proof exists — the requirement is that leakage = 0 is
proven before training, and no training happens in C0 or C1.

Explicitly **not** done in C0:

- no raw dataset copy (raw data stays at its existing approved local location);
- no re-preprocessing, no re-cropping, no new priors;
- no SiW label file opened, read or mounted;
- no target metric computed;
- no tuning of anything against SiW.

## 6. Target label firewall for P3 (inherited and extended)

| Stage | May resolve |
|---|---|
| TRAIN / LLM / SYNTHESIS | source data only. **Cannot** resolve SiW labels, attack-family metadata or target metrics |
| C11 PREDICT | label-free SiW features only |
| C12 SCORE | locked predictions + SiW labels, but **cannot** mutate any model, checkpoint or calibration artifact, and must not import the training stack |

Version C extends the Version-B firewall to a path that did not exist in Version B: **the
LLM prompt**. No SiW label, taxonomy, family count, attack-wise metric, target failure or
target prediction may appear in any prompt sent to the provider, and no face image may be
submitted at all.

Procedural, not blind: this is auditable isolation of the Version-C *process*. It does not
claim that the researchers have never seen SiW labels — Version B revealed them, and every
report must say so.
