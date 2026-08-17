# Learning-rate anchor decision — C4 and C7

**Status: AWAITING USER APPROVAL. Nothing here is implemented.**

Machine-readable companion: `LR_ANCHOR_DECISION_DOSSIER.json`.

The C0–C13 engineering-readiness milestone stopped with one unresolved
result-affecting decision: `learning_rate` has no uniquely inherited Version-B
anchor, and §15.2.2 does not permit choosing among ambiguous anchors silently.
This document reconstructs the evidence and lays out the interpretations that are
actually legal. It approves nothing and changes no search plan.

---

## 1. What the ambiguity actually is

There is **no scalar anywhere named `learning_rate`**. What exists is a set of
per-group learning rates, all of which are simultaneously active.

### C4 / GPAT — one AdamW, three groups

| scalar | value | parameter group | trainable params | live? |
|---|---|---|---|---|
| `encoder_lr` | 2.0e-4 | `artifact_encoder` | 422,272 | yes |
| `recipe_lr` | 1.0e-4 | `recipe_encoder` | 13,632 | yes |
| `generator_lr` | 2.0e-4 | `generator` (stem + 4 FiLM blocks + delta head + map head) | 474,634 | yes |

The three groups sum to 910,538 parameters — **exactly the model's full trainable
count**. None is historical, none is inactive, none is a superset of the others.
Inherited ratio **2 : 1 : 2**.

### C7 / detector — one AdamW, up to two groups

| scalar | value | parameter group | Track G | Track R |
|---|---|---|---|---|
| `backbone_lr` | 1.0e-5 | `backbone` (ConvNeXt V2 Atto) | **absent** | 3,386,760 params |
| `head_lr` | 1.0e-4 | `heads` (everything else trainable) | 12,353 params | 32,654 params |

Inherited ratio **1 : 10** (backbone : heads).

Also present and deliberately **not** LR anchors: `min_lr` (1e-6, a GPAT schedule
floor), `min_lr_scale` (0.0, a relative detector floor), and B00's own
`backbone_lr`/`head_lr` — B00 is an inherited Version-B baseline that no
Version-C row runs.

All three LR-bearing configs are **byte-identical to Version B**. Version B
recorded **no learning-rate sweep**: every value is a single inherited setting,
never a search winner. So nothing in the inheritance elevates one scalar above
the others.

---

## 2. Track G is not a user decision

§13.4.1 makes Track G global-only and forbids instantiating ConvNeXt.
`PRISMDetector.parameter_groups` omits an empty group, so under Track G the
backbone group **does not exist** and `backbone_lr` controls **zero parameters**.

Measured, not assumed:

```
Track G:  groups=['heads']              backbone_lr controls anything: False
Track R:  groups=['backbone','heads']   backbone_lr controls anything: True
```

Exactly one LR scalar is applicable to Track G, which is what a *uniquely
inherited anchor* means. §15.2.3 already directs that non-applicable scalars are
"skipped, not invented".

Version B's own M10 evidence shows the same structure: 35 experiments recorded
`optimizer_groups = ["backbone","heads"]`, and row **B01** — the local-branch-off
variant — recorded `["heads"]` alone.

> **Track G: `head_lr` = 1.0e-4. `ALREADY_IMPLIED_BY_FROZEN_SPEC`.**
> Recording this as a user choice would invent a decision the architecture
> already makes.

The ambiguity is real only for **C4** and **C7 Track R**.

---

## 3. The interpretations, and what each costs

§15.2.2 fixes the search order as one named sequence — *learning rate → weight
decay → warm-up → lambda_syn → lambda_local → lambda_MIL → lambda_P →
lambda_risk → active K=4-only weights*. It declares **one** learning-rate
coordinate. Trial counts below are computed by the real `SearchPlan`, not by hand.

| | C4 coords / trials | Track R coords / trials | class |
|---|---|---|---|
| **D** — skip the LR coordinate *(today)* | 3 / 9 | 7 / 21 | `NOT_APPLICABLE` |
| **A** — elect one scalar | 4 / **12** | 8 / **24** | `COMPATIBLE_BUT_USER_APPROVAL_REQUIRED` |
| **B** — common multiplier over all groups | 4 / **12** | 8 / **24** | `COMPATIBLE_BUT_USER_APPROVAL_REQUIRED` |
| **C** — one coordinate per group | 6 / **18** | 9 / **27** | `SEARCH_ENVELOPE_EXPANSION` |

*(Track G, for reference: 5 coords / 15 trials with `head_lr`; 4 / 12 without.)*

**D** is where the plans sit today. It is a holding position, not an
interpretation: §15.2.3's skip rule covers scalars that are *absent* or
*non-applicable*, and for C4 and Track R none is.

**A** keeps one coordinate and leaves the unelected groups at their inherited
values. Legal — but electing one of two or three equally inherited scalars is
discretion with no evidential basis behind it.

**B** keeps one coordinate, searches every active group, and preserves every
inherited ratio: the candidate is a multiplier *m* ∈ {0.5, 1.0, 2.0} applied to
each group's own anchor, so *m*=1.0 reproduces Version B exactly. **Same trial
count as A.** Its inheritance support is strong: the detector's `_lr_lambda`
returns *one* multiplier that `LambdaLR` applies to every group, and GPAT's
cosine schedule anchors each group on its own base under one shared shape.
Version B's own training already treats "the learning rate" as a common
multiplier over grouped anchors.

**C** inserts coordinates the frozen order does not name — two extra for C4, one
for Track R — and breaks the inherited ratios by construction. §15.2.2 already
makes *expanding a candidate set* approval-required; adding coordinates is a
larger change than that.

---

## 4. Recommendation — `RECOMMENDATION_ONLY`

| milestone | recommended | anchor |
|---|---|---|
| **C4 / GPAT** | **B — common multiplier** | vector `{encoder 2e-4, recipe 1e-4, generator 2e-4}`, ratio 2:1:2 held |
| **C7 Track G** | *no decision required* | `head_lr` = 1e-4 (already unique) |
| **C7 Track R** | **B — common multiplier** | vector `{backbone 1e-5, heads 1e-4}`, ratio 1:10 held |

Against the five stated criteria: B is the minimum deviation from the frozen
envelope (one coordinate, same trial count as A); it rests on the strongest
inheritance evidence (all anchors byte-identical to Version B, all groups live);
it adds the least discretion (no election required); it is the only option that
preserves the optimizer-group relationships; and it enlarges nothing.

**What approving B means:** accepting that "the inherited anchor", for a
component whose inherited learning rate is a *vector*, means that vector scaled
as a unit.

Recommending B for both keeps one meaning of "the learning-rate coordinate"
across the pipeline. Approving different interpretations per milestone is legal
but would make two search plans mean different things by the same word.

**If you decline all of them**, the envelopes stay as they are, the LR coordinate
stays skipped, and C4/C7 remain scientifically blocked.

---

## 5. Consequence for the frozen plan identities

Both current plans were built with `learning_rate` AMBIGUOUS and therefore
skipped:

```
c4_gpat_coordinate_v1       ab77e964d9c035cf2c3bed209ffac307aebd85c6735879bc3fa3c5efce20d0ec
c7_detector_coordinate_v1   62d0022507e732ba89618845fab2c63fec2b7b07f6817b2d541a4f500f459d7b
```

**Any approved interpretation changes both identities**, and the changed identity
is what a later full run must execute against. That is why the decision has to
land before the external-GPU search plans are frozen.

---

## 6. The exact decision required

1. **C4** — approve one of `B_common_multiplier` (recommended), `A_single_scalar`
   (and name which scalar), `C_independent_per_group`, or decline.
2. **C7 Track R** — the same four options.
3. **C7 Track G** — nothing to approve; confirm you accept the reading that
   `head_lr` is already the unique anchor.

Until then: `learning_rate_decision = AWAITING_USER_APPROVAL`, and C4/C7 full
execution stays blocked.

---

*This audit opened no dataset, allocated no GPU, ran no training, made no
provider call and resolved no target. Five pinned weight files were hashed for
verification and none was loaded.*
