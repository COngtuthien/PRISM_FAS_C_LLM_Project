# PRISM-FAS-C-LLM

**LLM-guided executable artifact planning for cross-domain face anti-spoofing.**

Version C asks one question: *does LLM recipe planning produce more useful synthetic
supervision than a deterministic structured generator or a random operator sampler?* It
answers it with three matched generator arms — RND, DET, LLM — run through the same
compiler, the same generator-neutral GPAT, the same quality gate and the same training
budget, so the recipe generator is the only changed factor.

**Status: C0 complete.** Spec reconciliation and repository isolation are done. No LLM has
been called, no model has been trained and no target has been scored.

---

## Relationship to PRISM-FAS-B

Version C derives from the **frozen Version-B scientific checkpoint**:

```
PRISM_FAS_B_Project @ m10-blind-evaluation-checkpoint = 7799f7decd35db6987ce4578824e5bd8d9eab4ae
        |
        |  git clone --no-hardlinks, checked out at the peeled tag
        v
PRISM_FAS_C_LLM_Project @ branch c0-spec-reconciliation
        origin -> https://github.com/COngtuthien/PRISM_FAS_C_LLM_Project.git
```

**Version B remains unchanged.** It is a completed scientific experiment and is treated as
read-only: no commit, no push, no reset, no re-tag, no retraining, no recalibration, no
history rewrite. Its artifacts are referenced here **by identity**, never copied or
mutated. The Version-C clone keeps an informational `version-b-readonly` remote whose push
URL is disabled, so an accidental push fails instead of reaching Version B.

**Version C is a separate scientific experiment.** It inherits Version B's infrastructure
— deterministic preprocessing, strict manifests, the recipe compiler, the physics
operators, the GPAT mechanism, the quality gate and `q` weighting, the target firewall,
the prediction lock and scorer isolation, the paired video bootstrap — and it does *not*
inherit the assumption that Version B's full regional model was the best one.

Full integrity record: [`docs/c0/C0_VERSION_B_INTEGRITY.md`](docs/c0/C0_VERSION_B_INTEGRITY.md).

## What the LLM does, and does not do

The LLM is an **offline semantic recipe planner**. It runs once, before training, and its
output is frozen.

```
Gemini structured recipe planner     gemini-3.6-flash, thinking_level=medium,
        |                            TEXT-ONLY, strict JSON schema, tools/grounding OFF
        v
strict JSON / ontology validator     fail-closed; max 2 machine retries
        |
        v
structured executable recipe         schema v1.1; 384 raw slots -> 256 final
        |
        v
the SAME inherited compiler          operator graph + mask policy + 41-D conditioning
        |
        +-------------------+
        v                   v
     Physics             GPAT-C      generator-neutral, trained once, frozen,
                                     shared by RND / DET / LLM
```

It is **not** the FAS classifier, not an image generator, not a target-image interpreter,
not part of inference and not part of the training loop. It never receives a face image,
a dataset filename, a SiW label, an attack taxonomy or any target metric.

Version B, by contrast, never invoked an external LLM at all: its frozen recipe bank
carries `external_llm_invoked: false` and was produced by
`deterministic-source-only-recipe-generator`. That gap is what Version C closes, and the
evidence is laid out in [`docs/c0/C0_LLM_GAP_ANALYSIS.md`](docs/c0/C0_LLM_GAP_ANALYSIS.md).

## Datasets and protocols

**Exactly three datasets. No fourth dataset in v1.1.**

| Dataset | Role |
|---|---|
| CASIA-FASD | source |
| MSU-MFSD | source |
| SiW-Mv2 v2 | P3 fixed held-out target |

| Protocol | Train | Select / calibrate | Cross-domain test | Track-G seeds |
|---|---|---|---|---|
| **P1** | CASIA | CASIA `source_dev` | MSU | 3 |
| **P2** | MSU | MSU `source_dev` | CASIA | 3 |
| **P3** (MAIN) | CASIA + MSU | combined domain-balanced `source_dev` | SiW-Mv2 v2 (1700 videos) | 5 |

### On SiW-Mv2 — read this before writing any claim

> SiW-Mv2 is **historically known from Version B**, where its labels were revealed and its
> results reported. In Version C it is a **fixed held-out cross-domain target** with
> **procedural label isolation**: the Version-C prediction process cannot resolve target
> labels, which is auditable, but that is **not** the same as never-seen blindness.

Version C **must never** describe P3 as a never-before-seen blind target, a blind
evaluation or a first-ever reveal. Every report must disclose the Version-B exposure.

## The experiment

| | |
|---|---|
| generator arms | RND (random operators), DET (deterministic structured), LLM (Gemini) |
| recipes per arm | **256**, matched; LLM selected deterministically from 384 raw candidate slots |
| synthesis | 8 renders/recipe (4 Physics + 4 GPAT) = 2048 candidates/arm |
| accepted per arm | **1024** = 512 Physics + 512 GPAT, under one common quality gate |
| Track G (primary) | frozen SigLIP2 Base P16-224 + light head; regions/PromptHead/manifold OFF |
| Track R (secondary) | ConvNeXt V2 Atto + frozen SigLIP2 + 9 semantic regions + PromptHead |
| statistics | paired video bootstrap, 10 000 resamples, 95 % percentile CI, seed 20260810, Holm-Bonferroni |

Hypotheses: C-H1 (LLM vs DET, Track G, 5 seeds), C-H2 (LLM vs RND, Track G, 5 seeds),
C-H3 (LLM vs DET, Track R, 3 seeds), C-H4 (bank-level reliability, source-only, reported
separately), C-H5 (PromptHead on/off, 3 seeds).

Frozen in [`docs/c0/C0_FROZEN_DESIGN_DECISIONS.md`](docs/c0/C0_FROZEN_DESIGN_DECISIONS.md)
and [`configs/version_c/c0_frozen_design.yaml`](configs/version_c/c0_frozen_design.yaml).

## Milestone status

| | Milestone | Status |
|---|---|---|
| **C0** | Spec reconciliation + isolated repository | **COMPLETE** |
| C1 | LLM provider + strict contract | not started |
| C2 | LLM pilot (32 disposable recipes) | not started |
| C3 | Frozen 256 RND / DET / LLM banks | not started |
| C4 | Generator-neutral GPAT-C | not started |
| C5 | Synthesis integration | not started |
| C6 | Quality gate + matched 1024 banks | not started |
| C7 | Track-G / Track-R implementation | not started |
| C8 | Source-only matrix | not started |
| C9 | `SOURCE_MATRIX_LOCK_C` | not started |
| C10 | SiW P3 package / label capability lock | not started |
| C11 | Label-isolated P3 prediction | not started |
| C12 | Isolated P3 scoring + statistics | not started |
| C13 | Final acceptance | not started |

Plan with inputs, outputs, tests, acceptance and stop conditions per milestone:
[`docs/c0/C0_MILESTONE_EXECUTION_PLAN.md`](docs/c0/C0_MILESTONE_EXECUTION_PLAN.md).

## C0 documents

| Document | Contents |
|---|---|
| [`C0_SPEC_RECONCILIATION.md`](docs/c0/C0_SPEC_RECONCILIATION.md) | 37 requirements across original B spec vs frozen B reality vs Version-C requirement, each classified and each with a Version-C action |
| [`C0_VERSION_B_INTEGRITY.md`](docs/c0/C0_VERSION_B_INTEGRITY.md) | read-only fingerprint of the frozen Version-B scientific state |
| [`C0_LLM_GAP_ANALYSIS.md`](docs/c0/C0_LLM_GAP_ANALYSIS.md) | evidence that Version B invoked no external LLM, and what Version C changes |
| [`C0_FROZEN_DESIGN_DECISIONS.md`](docs/c0/C0_FROZEN_DESIGN_DECISIONS.md) | the frozen v1.1 decisions |
| [`C0_COMPUTE_BACKEND_PLAN.md`](docs/c0/C0_COMPUTE_BACKEND_PLAN.md) | `modal_l4` / `ssh_lab` neutrality and the portability contract |
| [`C0_PROTOCOL_PLAN.md`](docs/c0/C0_PROTOCOL_PLAN.md) | P1/P2/P3 manifest design and the leakage-validation contract |
| [`C0_MILESTONE_EXECUTION_PLAN.md`](docs/c0/C0_MILESTONE_EXECUTION_PLAN.md) | C1–C13 |
| [`C0_ACCEPTANCE.json`](reports/c0/C0_ACCEPTANCE.json) | machine-readable C0 acceptance |

The three authoritative specs are copied verbatim into [`docs/spec/`](docs/spec/).

## Install and test

```bash
python -m pip install -e .[dev]
python -m pytest -q
```

Tests never download a dataset or a model, never call a provider and never incur an API
cost. The LLM provider abstraction ships with a mock/replay implementation, and that is
what the suite uses.

## Secrets

Copy [`.env.example`](.env.example) to `.env` (git-ignored) or export the variables from
your shell or secret store. `.env.example` carries variable **names only** and never a
value. The Gemini key is read from `GEMINI_API_KEY` and must never be logged, serialized
into JSON provenance, committed, or placed in a report or screenshot.

Billing is never enabled by code. If the Free Tier quota is exhausted, generation
checkpoints what it has, writes `API_QUOTA_BLOCKED.json`, stops cleanly and asks you
whether to wait for the reset or enable a Paid Tier.

## What is not in this repository

Raw datasets (CASIA-FASD, MSU-MFSD, SiW-Mv2), model weights and caches, generated crops,
Parquet manifests, NPZ priors, tar shards, generated recipe and synthetic banks, raw LLM
response archives, checkpoints, run logs, evaluation-only labels and local path
configuration. They are large, licence-restricted, private or machine-specific. Every one
is reproducible from source or referenced by hash.

Raw face datasets are never uploaded to GitHub and never sent to the LLM.

## Inherited Version-B contracts

These describe the frozen Version-B experiment and remain valid as the inherited
implementation reference: [`docs/M7_REGION_MASK_MAPPING.md`](docs/M7_REGION_MASK_MAPPING.md),
[`docs/M8_QUALITY_GATE_CONTRACT.md`](docs/M8_QUALITY_GATE_CONTRACT.md),
[`docs/M8_GPAT_CONTRACT.md`](docs/M8_GPAT_CONTRACT.md),
[`docs/M9_DETECTOR_CONTRACT.md`](docs/M9_DETECTOR_CONTRACT.md),
[`docs/M10_EXPERIMENT_CONTRACT.md`](docs/M10_EXPERIMENT_CONTRACT.md),
[`docs/M10_TARGET_EVALUATION_CONTRACT.md`](docs/M10_TARGET_EVALUATION_CONTRACT.md),
[`docs/M10_STATISTICS_CONTRACT.md`](docs/M10_STATISTICS_CONTRACT.md).
Where a Version-C document disagrees with one of them, the Version-C document governs
Version C and the Version-B document keeps governing the Version-B history.

## Claim policy

No claim is made before its evidence exists. Negative, blocked and failed results are
reported, never dropped. Version C does not claim state of the art, does not claim to be
the first LLM-based FAS method, does not equate valid JSON with physical understanding,
and does not describe P3 as a blind target.
