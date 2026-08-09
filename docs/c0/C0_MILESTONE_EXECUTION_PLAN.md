# C0 — Milestone execution plan, C1 → C13

Planned at C0. **None of these milestones is executed in C0.** No Gemini call, no
training, no GPU job, no synthesis, no SiW access has occurred.

Each milestone below records: inputs, actions, outputs, tests, hard acceptance,
GPU/API requirements, and stop conditions.

Global stop conditions that apply to every milestone:

- Version B is modified in any way → **STOP**.
- A SiW label, taxonomy, family count or target metric reaches a training, LLM, synthesis
  or selection process → **STOP**, invalidate the affected artifacts.
- An identity in the chain is missing or unknown → **fail closed**; never "best effort"
  resume.
- A behaviour branches on an `experiment_id` string → **STOP** and fix; all behaviour is
  resolved from typed configuration and enters the run identity.

---

## C1 — LLM provider and strict contract

| | |
|---|---|
| **Inputs** | frozen ontology (`m7-ontology-v1` lineage), recipe schema v1.1, Version-C leakage rules, `GEMINI_API_KEY` from the environment |
| **Actions** | provider abstraction with a real Gemini implementation **and** a mock/replay implementation; strict JSON Schema; system-prompt and request-template freeze with UTF-8 byte SHA256; validator (schema → ontology → compatibility → severity); retry policy (max 2 semantic retries, exponential transport backoff); 15-field provenance record; quota classifier distinguishing `rate_limit_exceeded` from `quota_exceeded` |
| **Outputs** | `docs/C_LLM_RECIPE_CONTRACT.md`, `configs/version_c/llm/*.yaml`, provider + validator modules, `reports/c1/` |
| **Tests** | malformed JSON; unknown enum; out-of-range severity; duplicate recipe; retry exhaustion; missing provenance field; prompt containing a forbidden target token is refused; secret never appears in a serialized artifact or log |
| **Hard acceptance** | no target information can enter a prompt; every invalid output fails closed; provenance complete; **all tests use the mock/replay provider** |
| **GPU / API** | no GPU. A live-provider integration test may generate **≤ 2 disposable recipes**, which never enter any scientific bank |
| **Stop conditions** | a real key would be written to disk, a log or provenance; a live test exceeds 2 recipes; the schema cannot express the inherited ontology without a version bump |

## C2 — LLM pilot (32 disposable recipes)

| | |
|---|---|
| **Inputs** | frozen C1 contract, Free-Tier Gemini access |
| **Actions** | generate 32 **disposable** pilot candidates; record the live quota snapshot from AI Studio; measure unique/invalid/retry rates and per-axis ontology coverage; produce human-readable sample recipes for review |
| **Outputs** | pilot bank + audit under `reports/c2/`, quota snapshot |
| **Tests** | pilot bank is structurally marked disposable and cannot be loaded as a scientific bank |
| **Hard acceptance** | no final bank produced; invalid/duplicate/retry/quota statistics documented; **the user approves and freezes `gemini-3.6-flash`, the prompt and the schema before C3** |
| **GPU / API** | no GPU. First real API usage; Free Tier |
| **Stop conditions** | daily quota exhausted → checkpoint and stop cleanly, notify the user, never auto-enable billing. Pilot recipes must never leak into the final bank. Prompt tuning here may use **source-only** validity and coverage — never any downstream target performance |

## C3 — Frozen recipe banks (256 RND / 256 DET / 256 LLM)

| | |
|---|---|
| **Inputs** | frozen prompt/model/schema from C2; inherited DET generator; RND sampler |
| **Actions** | generate exactly **384** LLM raw candidate slots under the frozen 12×32 schedule; archive every request and response; validate, dedup, then select 256 deterministically by coverage; build the 256-recipe DET and RND banks under the same ontology and ranges; compile-test all three |
| **Outputs** | three `BANK_LOCK`s, raw-response archive + metadata manifest, coverage report, `reports/c3/` |
| **Tests** | each bank has exactly 256 unique valid recipes; every artifact type and all 9 regions present; medium/geometry minimums met; all three compile through the same compiler; selection is reproducible from the archive |
| **Hard acceptance** | exactly 256 per arm; identities frozen; no target information present; final 256 chosen **algorithmically**, never hand-picked |
| **GPU / API** | no GPU. The scientific LLM generation run |
| **Stop conditions** | fewer than **320** valid unique recipes after the 384-slot budget → **C3 FAILS**; the validator is never weakened after seeing results. Quota exhaustion → preserve completed responses, write `API_QUOTA_BLOCKED.json`, stop, ask the user. Raw responses become immutable the moment C3 begins |

## C4 — Generator-neutral GPAT-C

| | |
|---|---|
| **Inputs** | ontology + safe parameter manifold; frozen source package; **not** the RND/DET/LLM treatment banks |
| **Actions** | build a deterministic ontology-derived neutral conditioning-support bank; train GPAT **once**; freeze the checkpoint |
| **Outputs** | GPAT-C checkpoint + `SUPPORT_LOCK` recording the exact support-bank identity, `reports/c4/` |
| **Tests** | support bank is provably independent of all three treatment banks; low-frequency geometry lock; identity-drift bound; outside-mask invariants |
| **Hard acceptance** | the **same** frozen GPAT checkpoint is valid for all arms; source-only; geometry invariants pass |
| **GPU / API** | GPU required (`modal_l4` acceptable). No API |
| **Stop conditions** | any overlap between the support bank and a treatment bank; geometry or outside-mask invariant fails; the checkpoint would be retrained per arm |

## C5 — Synthesis integration

| | |
|---|---|
| **Inputs** | three frozen recipe banks; frozen GPAT-C; `source_train` only |
| **Actions** | render 4 Physics + 4 GPAT candidates per recipe for each arm = 2048 candidates/arm, 6144 total, from one fixed ordered live base-sample list shared by all arms |
| **Outputs** | candidate synthetic artifacts + provenance, `reports/c5/` |
| **Tests** | identical base live list across arms; identical route budget; `source_dev` and every target rejected by code; exact mask, zero change outside mask |
| **Hard acceptance** | same base live list and route budget for every arm; no target touched |
| **GPU / API** | GPU required. No API |
| **Stop conditions** | any arm rendering a different count or drawing from a different base list; any read outside `source_train` |

## C6 — Quality gate and matched banks

| | |
|---|---|
| **Inputs** | 6144 candidates; one common quality gate identity |
| **Actions** | recalibrate the gate on the C population **before** scoring any arm; apply one gate to all arms; build matched banks of exactly 1024 accepted/arm = 512 Physics + 512 GPAT, balanced deterministically over source domain, route, recipe coverage and base live IDs; run the reliability suite |
| **Outputs** | three synthetic `BANK_LOCK`s, quality/reliability reports, `reports/c6/` |
| **Tests** | equal cardinality and equal route split per arm; `q` weights only the synthetic bracket; synthetic-vs-real probe; residual sensitivity; corruption, crop/interpolation, recipe-region intervention, artifact-map swap, cross-route, zero-artifact-map |
| **Hard acceptance** | exactly 1024 accepted per arm (512/512); **shortcut gates pass, or STOP** |
| **GPU / API** | GPU required. No API |
| **Stop conditions** | an arm cannot reach 1024 under the common gate → **C6 FAILS**; the gate is never relaxed for one arm. Synthetic-vs-real balanced accuracy above 0.75 → C6 fails or requires redesign **before** target. Residual-sensitivity movement below 0.10 → gate fails |

## C7 — Detector implementation (Track G + Track R)

| | |
|---|---|
| **Inputs** | frozen matched banks; inherited SigLIP2 and ConvNeXt V2 Atto identities |
| **Actions** | implement Track G (frozen SigLIP2 + light head; regions/PromptHead/manifold OFF) and Track R (ConvNeXt V2 Atto + frozen SigLIP2 + 9 regions + PromptHead; manifold OFF in the primary variant); freeze the batch and loss contracts; serialize variant resolution into the run identity |
| **Outputs** | instantiate/forward/backward/resume reports, `reports/c7/` |
| **Tests** | CPU-fixture smoke for **every** primary row: instantiate, forward, finite loss, backward, optimizer step, checkpoint, resume; target-time `p_prompt` is `null`, never `0.0`; `synthetic=none` truly omits the bracket; the §16 calibration guard — no fused score thresholded by a calibration fitted on a different quantity |
| **Hard acceptance** | no experiment-id branching; every primary row executable; no silent fallback to Track-R reference behaviour |
| **GPU / API** | CPU fixture sufficient for acceptance; GPU for preflight. No API |
| **Stop conditions** | the Version-B multi-prototype manifold appears in the primary Track-R variant; a variant resolves by string matching; the calibration guard is absent |

## C8 — Source-only matrix

| | |
|---|---|
| **Inputs** | frozen banks, frozen GPAT-C, C7 detectors, P1/P2/P3 manifests |
| **Actions** | run P1 and P2 Track-G RND/DET/LLM at 3 seeds; P3-ready Track-G RND/DET/LLM at **5** seeds; Track-R DET/LLM at 3 seeds; the PromptHead on/off ablation at 3 seeds; cross-source diagnostics; calibration-stability audit across CASIA-dev and MSU-dev |
| **Outputs** | complete source run set, `reports/c8/` |
| **Tests** | every declared seed present; no target opened; calibration fitted on `source_dev` only |
| **Hard acceptance** | **no SiW P3 scoring opened during C8**; all mandatory P1/P2/P3-ready runs and seed counts complete; cross-source and calibration-stability reports exist before the source lock |
| **GPU / API** | heaviest GPU stage. **`ssh_lab` support is mandatory before C8.** Hypothesis-critical rows should run on one homogeneous compute profile. No API |
| **Stop conditions** | Modal access expires mid-block → the affected rows are rerun on the final profile, never pooled. Lab GPU lacks BF16 → **STOP** and freeze an explicit FP16 profile first. Precision must never change silently inside a block |

## C9 — Source freeze

| | |
|---|---|
| **Inputs** | the complete C8 run set |
| **Actions** | build and validate `SOURCE_MATRIX_LOCK_C`; validate it **twice** |
| **Outputs** | `SOURCE_MATRIX_LOCK_C`, `reports/c9/` |
| **Tests** | every checkpoint, calibration and identity present and frozen; every blocked row carries a reason; `selection_used_target = false` |
| **Hard acceptance** | all identities frozen; **0 failed hidden rows**; the lock validates twice |
| **GPU / API** | none |
| **Stop conditions** | any hypothesis-critical row incomplete or silently dropped; any identity unresolved |

## C10 — P3 SiW package and label-capability lock

| | |
|---|---|
| **Inputs** | the existing SiW-Mv2 v2 feature package (Version-B identity `c3a29e69…e48a8`) |
| **Actions** | validate and freeze the feature package for Version C; create a Version-C evaluation-only label capability and target lock **without modifying the Version-B package** |
| **Outputs** | target package lock + sealed label lock, `reports/c10/` |
| **Tests** | feature identity verified; label files absent from every training environment and volume |
| **Hard acceptance** | no label leakage; raw labels absent from the training environment; P3 fixed to SiW-Mv2 v2 |
| **GPU / API** | none |
| **Stop conditions** | any change of target or dataset composition — that requires PRISM-FAS-C protocol v2, not an in-place v1.1 edit. Any label file mounted on a training process |

## C11 — P3 label-isolated prediction

| | |
|---|---|
| **Inputs** | `SOURCE_MATRIX_LOCK_C`; label-free SiW features |
| **Actions** | run C-G7 inference **exactly once** per preregistered P3 row under its frozen inference config; build per-row `PREDICTION_LOCK_C` and the global `TARGET_PREDICTION_LOCKSET_C`; write the procedural pre-score audit |
| **Outputs** | prediction locks + lockset, pre-score audit, `reports/c11/` |
| **Tests** | labels structurally unresolvable to the prediction process; predictions carry no ground truth, attack family, raw path, subject/session taxonomy or hidden target metadata; **lockset validated twice** |
| **Hard acceptance** | labels unresolvable; lockset validated twice. This is **procedural isolation**, not a claim of historical blindness |
| **GPU / API** | GPU for inference. No API |
| **Stop conditions** | any optimizer or checkpoint mutation; any row predicted more than once; any label resolvable from the prediction process |

## C12 — P3 isolated scoring and statistics

| | |
|---|---|
| **Inputs** | frozen predictions + evaluation-only SiW labels |
| **Actions** | grant scorer-only label capability **after** the lockset validates; run isolated C-G8; compute primary/secondary metrics, paired video bootstrap (10 000 resamples, 95 % percentile, seed 20260810) and Holm-Bonferroni over C-H1/C-H2/C-H3/C-H5; report C-H4 separately |
| **Outputs** | `statistics.json`, `summary.json`, `reliability.json`, report, `reports/c12/` |
| **Tests** | the **dry-run must not read label bytes**; the scorer has no training capability (import audit); no calibration mutation path |
| **Hard acceptance** | no training rerun; scorer isolation proven; no target recalibration |
| **GPU / API** | none — no torch training stack in the scorer |
| **Stop conditions** | any attempt to feed a P3 metric back into C0–C11, prompt design, bank selection, checkpoint selection or calibration. Any redesign after seeing P3 becomes a new protocol version, **not** a retry of v1.1. Negative results are preserved, never dropped |

## C13 — Final acceptance

| | |
|---|---|
| **Inputs** | every lock, report and statistic |
| **Actions** | full integrity verification of the identity chain; final report; paper evidence package; propose (do not create early) the Version-C scientific tag |
| **Outputs** | `C_ACCEPTANCE.json`, final report, `reports/c13/` and `reports/final/` |
| **Tests** | every preregistered check re-run; every claim mapped to its evidence |
| **Hard acceptance** | all preregistered checks pass; **negative, blocked and failed hypotheses all reported**; the SiW historical-exposure disclosure present |
| **GPU / API** | none |
| **Stop conditions** | any claim without its required evidence; any wording calling P3 a never-before-seen blind target; any claim of SOTA or first-method without the required external verification |

---

## Open items carried out of C0

| # | Item | Needed by | Owner decision |
|---|---|---|---|
| 1 | Lab GPU capability profile (name, VRAM, driver, CUDA, PyTorch build, BF16/FP16) is **unknown** | before C8 | run the capability probe; do not assume a GPU model |
| 2 | Exact Gemini SDK/API version to freeze | C1 | record the real installed version at C1 |
| 3 | Live Free-Tier quota snapshot (RPM/TPM/RPD) | C2 | read from AI Studio; never hard-code public numbers into the scientific identity |
| 4 | User approval of model + prompt + schema | end of C2, before C3 | explicit user decision |
| 5 | Paid-Tier escalation, if Free Tier blocks C3 | only if it happens | **user decision only**; code must never auto-enable billing |
| 6 | Whether Modal access survives long enough to cover C8 | before C8 | if not, the full hypothesis-critical matrix runs on `ssh_lab` |
| 7 | Version-C protocol manifests for P1/P2/P3 to be written into the C namespace | C1, before any training | design frozen in [`C0_PROTOCOL_PLAN.md`](C0_PROTOCOL_PLAN.md) |
| 8 | Resolution of the inherited synthetic-vs-real probe failure (Version B measured 0.9375 against a 0.75 ceiling) | C6, blocking P3 | in Version C this is a gate, not a disclosure |
