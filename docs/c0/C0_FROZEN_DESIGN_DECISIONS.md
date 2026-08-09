# C0 — Frozen Version-C design decisions

Transcribed from the authoritative spec
`docs/spec/PRISM_FAS_C_LLM_v1_1_FINAL_Detailed_Spec_2026.docx`
(SHA256 `d9edbfefa6829f29bb075e2f3d12073bb6517be57c0debcf93c92c4346d2e2df`).

These are **FROZEN**. A `MUST` may not be downgraded to a `SHOULD` by an implementer, and
any scientifically material change requires a decision record written *before* the target
runs. The machine-readable mirror is
[`configs/version_c/c0_frozen_design.yaml`](../../configs/version_c/c0_frozen_design.yaml),
and `tests/c0/test_c0_frozen_design.py` asserts the two agree.

## 1. Datasets

**Exactly three. No fourth dataset in v1.1.**

| Dataset | Version-C role | Labels during development |
|---|---|---|
| CASIA-FASD | SOURCE | yes |
| MSU-MFSD | SOURCE | yes |
| SiW-Mv2 v2 | P3 FIXED HELD-OUT TARGET | historically known from Version B; must be unresolvable to C training/prompt/model-selection code |

Adding a dataset requires a new protocol version (PRISM-FAS-C v2). It is not an in-place
v1.1 edit. The pre-existing replay-only dataset is not part of this experiment matrix.

## 2. Protocols

| | P1 | P2 | P3 (MAIN) |
|---|---|---|---|
| train | CASIA `source_train` | MSU `source_train` | CASIA + MSU `source_train` |
| select / calibrate | CASIA `source_dev` | MSU `source_dev` | combined domain-balanced `source_dev` |
| cross-domain test | MSU frozen cross-test manifest | CASIA frozen cross-test manifest | SiW-Mv2 v2 (1700 videos = 785 live + 915 spoof) |
| Track-G seeds | 3 | 3 | 5 |
| role | cross-source evidence | cross-source evidence | fixed held-out target evidence |

P1 and P2 are separate experiments with separate locks. A dataset being the source in one
protocol and the cross-domain test in another does **not** permit using its test result to
retune that same protocol.

### 2.1 The SiW-Mv2 disclosure — mandatory wording

> SiW-Mv2 is **historically known from Version B**, where its labels were revealed and its
> results reported. For Version C it is a **fixed held-out cross-domain target** with
> **procedural label isolation**: the Version-C prediction process cannot resolve target
> labels, which is auditable, but this is **not** equivalent to never-seen blindness.

Version C **MUST NOT** call P3 a never-before-seen blind target, a first-ever reveal, or a
blind evaluation. The paper must disclose the Version-B exposure. This wording constraint
is frozen and applies to every report, plot caption and abstract produced by this project.

## 3. Generator arms

| Arm | Definition | Role |
|---|---|---|
| `RND` | rule-valid random operators sampled from the same ontology and ranges | lower semantic-planning control |
| `DET` | deterministic structured generator inherited/adapted from Version B | structured non-LLM control |
| `LLM` | frozen Gemini structured recipe planner | proposed treatment |

All three **MUST** share: the same ontology and schema, the same operator parameter
ranges, the same compiler interface, the same validation, exactly **256** final recipes,
matched coverage requirements, the same source live base-sample list, the same route
budget, the same frozen GPAT checkpoint, the same quality gate, and the same downstream
synthesis and training budget.

> **No generator arm may gain an advantage from extra training exposure.**

## 4. LLM provider contract (declared at C0, first called at C1)

**Zero live Gemini requests were made during C0.**

| Field | Frozen value |
|---|---|
| provider | Google Gemini Developer API |
| scientific model | `gemini-3.6-flash` (stable) |
| `thinking_level` | `medium` |
| modality | **TEXT ONLY** |
| structured response | `application/json` with a strict JSON Schema |
| SDK / API version | frozen at C1 and recorded — not guessed at C0 |

Must be **OFF**: tools, Google Search grounding, URL context, code execution, file search,
image input, audio input, video input.

Must **not be sent**: `temperature`, `top_p`, `top_k` — these sampling controls are
deprecated for Gemini 3.x and the spec forbids them.

### 4.1 Leakage rules — absolute

- No face image of any kind may ever be submitted to the LLM.
- No CASIA / MSU / SiW private image content may be submitted.
- No SiW label, attack-family taxonomy, family count, attack-wise metric, target failure
  analysis, target prediction or Version-B target feedback may appear in any prompt.
- The generation prompt may contain only frozen ontology-level coverage quotas and
  request-batch quotas — never dataset-derived CASIA/MSU deficits.
- Humans may not hand-pick "nice" outputs. Selection is machine-validated and
  deterministic after raw generation.

### 4.2 Secret handling

The key is read from the environment or a secret store under the name `GEMINI_API_KEY`.
It must never be logged, never serialized into JSON provenance, never committed, never
placed in a report or screenshot. `.env` is git-ignored; `.env.example` carries the
variable **name** only and no value. **C0 tests use mocks only**, and no unit or
integration test may ever incur a live API call or cost.

### 4.3 Retry and quota policy

| Condition | Action |
|---|---|
| non-JSON / schema-invalid / ontology-invalid output | REJECT; machine retry with machine-generated validation errors only, max **2** retries per request; then mark failed. Never silently repair a semantic field |
| 429 `rate_limit_exceeded` (transport) | retry the **exact same frozen request** with exponential backoff. Does not authorize changing batch size, prompt, model or schema |
| 429 `quota_exceeded` / daily quota exhausted | preserve every completed raw response and request-slot state atomically, write `API_QUOTA_BLOCKED.json` with completed call ids and hashes, **STOP cleanly**, notify the user |

Start on **Free Tier**. Code **MUST NEVER** enable billing, open a billing page, attach a
billing account, prepay credits or auto-reload. After a quota stop the user chooses: wait
for reset, or explicitly enable a Paid Tier. On resume, the *same* frozen request ids are
retried with an unchanged model, prompt, schema and selection algorithm. Billing tier is
recorded in provenance and is **not** a treatment factor.

## 5. Recipe bank budget

| Quantity | Value |
|---|---|
| C2 pilot candidates (disposable, never enter the final bank) | 32 |
| C3 LLM raw candidate slots | **384** (frozen schedule, recommended 12 calls × 32 objects) |
| minimum valid unique pool before selection | 320 — below this, **C3 fails**; the validator is never weakened after seeing results |
| final bank, each arm | **256** |
| final banks | 256 RND + 256 DET + 256 LLM |

Selection from 384 → 256 is deterministic and algorithmic. **No manual cherry-picking of
LLM recipes.** Raw responses become immutable provenance the moment C3 begins.

Per-recipe limits: 1–3 artifacts, 1–3 semantic regions. Every ontology artifact and all
nine region classes must appear in the final bank; every medium and geometry category must
meet a predeclared minimum count.

## 6. Generator-neutral GPAT-C

Version B's A02 confound must not recur: its random-recipe control ran against a GPAT
trained on a structured recipe identity, leaving an out-of-distribution conditioning
caveat.

```
ontology-derived neutral conditioning support
        |            (deterministic, from the ontology and the safe parameter
        |             manifold; INDEPENDENT of the RND/DET/LLM treatment banks)
        v
      GPAT-C   ---- trained once
        |
      FREEZE
        |
   +----+----+----+
   |         |    |
  RND       DET  LLM        <- the same frozen checkpoint for every arm
```

> **Normative.** No treatment arm may receive a GPAT checkpoint trained specifically on its
> own recipe bank in the primary C-H1/C-H2/C-H3 comparisons.

C0 documents this contract only. **GPAT-C is not trained in C0.**

## 7. Synthesis budget

Per generator arm:

```
256 recipes  x  ( 4 Physics + 4 GPAT renders )  =  2048 candidate renders / arm
3 arms                                          =  6144 candidate renders
```

After the **common** quality gate, each arm must provide exactly:

```
1024 accepted training samples / arm  =  512 Physics + 512 GPAT
```

> If one arm cannot reach 1024 under the common gate, **C6 FAILS**. The quality gate is
> never lowered for one arm.

Synthesis draws from `source_train` only; `source_dev` and every target are rejected by
code. All arms share one fixed ordered list of live source sample IDs.

`q` is a sample-quality weight, never a class label. It multiplies **only** the synthetic
loss bracket, never real samples and never the whole loss.

## 8. Detector tracks

### Track G — PRIMARY generator-effect detector

| Component | Value |
|---|---|
| image encoder | frozen SigLIP2 Base P16-224, Version-B model identity/revision |
| trainable | lightweight binary / fusion head only |
| regions | **OFF** |
| PromptHead | **OFF** |
| manifold | **OFF** |
| sampler | domain/class-balanced |

Purpose: measure the effect of the recipe generator with minimum architectural
confounding. This is where C-H1 and C-H2 are decided.

### Track R — SECONDARY spec-faithful regional PRISM

| Component | Value |
|---|---|
| local branch | ConvNeXt V2 **Atto** (inherited weight identity) — **not** Tiny in v1.1 |
| global branch | frozen SigLIP2 image tower |
| regions | 9: left_eye, right_eye, nose, mouth, forehead, left_cheek, right_cheek, face_boundary, context |
| region fusion | cross-attention over the local feature map + global patch tokens |
| PromptHead | **ON** (synthetic region-text alignment) |
| manifold | **OFF** in the primary Track-R variant; K=4 only as an explicit secondary row |
| outlier supervision | mask-aware synthetic losses; clean regions retain consistency |

> The Version-B multi-prototype regional manifold **must not be silently restored** into
> the main Version-C detector. Version B's H2 found it significantly worse, and v1.1
> excludes it from the primary variant. Track R exists to study semantic regional
> supervision, not to re-assert the full Version-B architecture.

PromptHead applicability: for ordinary real target samples no recipe or attack mask
exists, so target-time `p_prompt` **MUST be `null` / not_applicable**, never `0.0`. Any
target benefit is via representation learned during training, not a target-time text
measurement.

## 9. Replication

```
seed family (5-seed rows):  20260806  20260807  20260808  20260809  20260810
seed family (3-seed rows):  the first three of the above
```

| Rows | Seeds |
|---|---|
| P3 Track-G primary, C-H1 / C-H2 | **5** |
| P1 / P2 Track-G | 3 |
| Track-R and PromptHead rows (C-H3 / C-H5) | 3 |
| everything else | 1 — **diagnostic only** |

A single-seed row may **never** support a superiority claim. No best-seed reporting and no
cherry-picking. Report per-seed values plus mean ± std for every replicated row.

## 10. Statistics

| Field | Frozen value |
|---|---|
| primary endpoint | video ACER |
| unit | **VIDEO** |
| bootstrap | paired video bootstrap |
| resamples | **10 000** |
| confidence interval | 95 % percentile |
| bootstrap seed | **20260810** |
| multiple-comparison correction | Holm-Bonferroni |
| target hypothesis family | C-H1, C-H2, C-H3, C-H5 |
| reported separately | C-H4 (source-only mechanism/reliability) |
| Δ definition | ΔACER = treatment − control; negative favours LLM |
| video aggregation | trimmed mean, trim = 0.10, for the decision score; median confidence |

A hypothesis is supported only if the effect direction matches the prediction **and** the
corrected inference excludes the null under the frozen rule. No target result was computed
during C0.

## 11. Hypotheses

| ID | Treatment vs control | Endpoint | Direction | Seeds |
|---|---|---|---|---|
| C-H1 | LLM vs DET, same Track-G detector | P3 video ACER | ΔACER < 0 | 5 |
| C-H2 | LLM vs RND, same Track-G detector | P3 video ACER | ΔACER < 0 | 5 |
| C-H3 | LLM vs DET, same Track-R detector | P3 video ACER | ΔACER < 0 | 3 |
| C-H4 | LLM bank vs other banks: separability + coverage | source-only reliability | lower separability without worse validity; higher coverage | bank-level + 3 probe seeds |
| C-H5 | Track-R LLM PromptHead ON vs OFF | P3 video ACER | ΔACER < 0 | 3 |

## 12. What Version C deliberately does not do

- Does not use any LLM/VLM to look at a target image and guess live/spoof.
- Does not use text-to-image generation to redraw a face in the MVP.
- Does not call an LLM inside the training loop or the inference loop.
- Does not tune prompt, recipe, architecture, checkpoint, quality gate or calibration on
  SiW-Mv2 results.
- Does not promise a favourable result. The goal is a clean protocol and a strong
  source-side design.
- Does not modify Version-B history, tags or artifacts.
